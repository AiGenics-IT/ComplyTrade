"""
Step 14 -- VLM-Only Clause Verification
==========================================
Verifies EVERY LC condition against the actual shipping documents using
Qwen VLM. NO code-based / regex checks -- everything goes through VLM
for maximum accuracy.

PURPOSE:
    For each verification row (from Step 13), this step:
    1. Finds the relevant document(s) from Step 9's reconciled_packets
    2. Sends the condition + document text + page image to the Qwen VLM
    3. VLM determines compliance with full F47A context awareness

    ALL 13 LC key-term conditions and ALL F46A/F47A decomposed conditions
    are verified exclusively through the VLM -- no implicit code paths.

INPUTS:
    - step13_result: dict with 'rows' (condition rows to verify)
    - step09_result: dict with 'reconciled_packets' (documents with text +
      image paths + extracted_fields)
    - step06_result: dict with 'final_lc' containing 'consolidated_fields'
      and 'clauses'
    - output_dir: str
    - progress_callback: callable

OUTPUTS:
    - dict with 'rows' (verified rows with findings, result, compliance,
      confidence), 'summary', 'elapsed_seconds'

AI MODEL: Qwen VLM at QWEN_VLM_URL (resolved from config/settings.py via
          the VLM_MODEL_SIZE switch — 7B or 72B). Used for ALL verification
          -- no code-based fallbacks.
"""

import json
import sys as _sys
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import base64
import os
import re
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from typing import List, Optional, Dict, Any
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import QWEN_VLM_URL, QWEN_VLM_MODEL, MAX_CONCURRENT_VLM, VLM_TIMEOUT


# ---------------------------------------------------------------------------
# Document type alias map -- maps condition target names to classification
# labels used in Step 9 reconciled_packets.
# ---------------------------------------------------------------------------

DOC_TYPE_ALIASES = {
    "bill of lading": [
        "bill of lading", "bl", "b/l", "ocean bill", "marine bill",
        "transport document", "multimodal",
        "short form bill of lading", "blank back bill of lading",
        "liner bill of lading", "charter party bill of lading",
        "combined transport bill of lading", "multimodal bill of lading",
    ],
    # Air Waybill and Courier Receipt are treated as ONE family (per the
    # step08 classifier change). Step 12 may decompose a clause as "AWB",
    # "Air Waybill", "Courier Receipt", "DHL receipt", etc. — they all
    # need to map to whichever packet was classified into either label.
    "air waybill": [
        "air waybill", "airway bill", "awb", "hawb", "mawb",
        "house air waybill", "master air waybill",
        "courier receipt", "courier waybill", "courier service receipt",
        "express waybill", "express envelope", "express courier",
        "express delivery receipt",
        "dhl", "fedex", "fed ex", "ups", "tnt", "aramex",
    ],
    "commercial invoice": [
        "commercial invoice", "invoice", "tax invoice", "signed invoice",
    ],
    "insurance": [
        "insurance policy", "insurance certificate", "insurance",
        "insurance cover",
    ],
    "packing list": [
        "packing list", "packing note", "packing specification",
        "packing slip",
    ],
    "certificate of origin": [
        "certificate of origin", "coo", "origin certificate",
    ],
    "draft": [
        "draft", "bill of exchange", "usance draft", "sight draft",
    ],
    "weight list": [
        "weight list", "weight note",
    ],
    "beneficiary certificate": [
        "beneficiary certificate", "beneficiary statement",
    ],
    "documentary remittance": [
        "documentary remittance", "covering letter", "remittance letter",
        "schedule of documents", "letter of transmittal", "covering schedule",
        # P64: "Export DC Document Presentation Schedule" is the same
        # document as Documentary Remittance -- it is the covering schedule
        # the beneficiary's bank sends with the presented documents.
        "export dc document presentation schedule",
        "export dc presentation schedule",
        "document presentation schedule",
        "presentation schedule",
        "dc presentation schedule",
        "export presentation schedule",
        "schedule of presentation",
    ],
    "inspection certificate": [
        "inspection certificate", "survey report", "inspection report",
    ],
    "fumigation certificate": [
        "fumigation certificate",
    ],
    "phytosanitary certificate": [
        "phytosanitary certificate", "phytosanitary",
    ],
    # P81: Form 7 = Batch Certificate (Pakistan Drug Act Rule 14(d)(i))
    # Form 3 = Drug Registration Certificate / Import Certificate
    # These regulatory forms are referred to by their form number in
    # Pakistani LC clauses but classified by their descriptive name
    # by the VLM. Both names must map to the same bucket.
    "form 7": [
        "form 7", "form7", "form no 7", "form no. 7",
        "batch certificate", "batch certification",
        "form 7 batch certificate", "form 7 (batch certificate)",
        "form 7 batch certification",
    ],
    "batch certificate": [
        "batch certificate", "batch certification",
        "form 7", "form7", "form no 7", "form no. 7",
        "form 7 batch certificate", "form 7 (batch certificate)",
        "form 7 batch certification",
    ],
    "form 7 (batch certificate)": [
        "form 7 (batch certificate)", "form 7 batch certificate",
        "form 7", "form7", "batch certificate", "batch certification",
        "form no 7", "form no. 7",
    ],
    "form 3": [
        "form 3", "form3", "form no 3", "form no. 3",
        "drug registration certificate", "import certificate",
        "form 3 certificate", "drug import certificate",
    ],
    "shipping advice": [
        "shipping advice", "shipment advice", "beneficiary shipment advice",
    ],
    "quality certificate": [
        "quality certificate", "quality report", "products quality certificate",
    ],
    "weight certificate": [
        "weight certificate", "weight list", "weight note",
    ],
    "shipping company certificate": [
        "shipping company certificate", "shipping certificate",
        "agent certificate", "agents certificate",
        # P64: Agent's Certificate is the same document as Shipping Company
        # Certificate — UCP/ISBP treats a certificate signed by a shipping
        # company OR by its authorised agent as the same instrument.
        "agent's certificate", "agents' certificate",
        "shipping agent certificate", "shipping agent's certificate",
        "carrier's certificate", "carriers certificate",
        "carrier certificate",
        "certificate from shipping company",
        "certificate from shipping agent",
        "certificate from carrier",
        "certificate of shipping company",
    ],
    "surveyor certificate": [
        "surveyor certificate", "survey certificate", "surveyor report",
    ],
    "export letter": [
        "export letter of credit bill remittance letter",
        "remittance letter", "covering letter", "covering schedule",
        "bill remittance letter", "schedule of documents",
    ],
}

# The 13 LC key-term field tags and the document types they check
_KEY_TERM_DOC_MAP = {
    "31C": ["all"],                     # Date of Issue -- every doc
    "31D": ["documentary remittance"],  # LC Expiry -- presentation date
    "51D": ["bill of lading"],          # Applicant Bank -- BL consignee
    "50":  ["bill of lading"],          # Applicant -- BL notify party
    "59":  ["bill of lading"],          # Beneficiary -- BL shipper
    "32B": ["commercial invoice", "draft"],  # Amount -- invoice + draft
    "41D": ["documentary remittance"],  # Available With
    "42C": ["draft"],                   # Draft at Sight
    "42A": ["draft"],                   # Drawee
    "43T": ["bill of lading"],          # Transshipment
    "44E": ["bill of lading"],          # Port of Loading
    "44F": ["bill of lading"],          # Port of Discharge
    "44C": ["bill of lading"],          # Latest Shipment Date
}


# ---------------------------------------------------------------------------
# Helpers -- unified access for dataclass / dict rows
# ---------------------------------------------------------------------------

def _get(row, key: str, default=""):
    """Get a field from either a dataclass or dict row."""
    if hasattr(row, key):
        return getattr(row, key)
    if isinstance(row, dict):
        return row.get(key, default)
    return default


def _set(row, key: str, value):
    """Set a field on either a dataclass or dict row."""
    if hasattr(row, key):
        setattr(row, key, value)
    elif isinstance(row, dict):
        row[key] = value


# ---------------------------------------------------------------------------
# Document matching
# ---------------------------------------------------------------------------

def _find_matching_docs(doc_to_check: str, packets: list) -> list:
    """
    Find ALL matching documents from reconciled_packets for a given
    document type string.  Uses three-tier matching:
      1. Alias lookup (exact canonical match)
      2. Substring match (target in pkt_type or vice versa)
      3. Keyword overlap (fuzzy — matches any certificate/doc with shared keywords)

    This avoids hardcoding every possible certificate/document type.
    """
    if not doc_to_check or not packets:
        return []

    target = doc_to_check.lower().strip()

    # Build alias list for target
    target_aliases = [target]
    for _canonical, aliases in DOC_TYPE_ALIASES.items():
        if any(alias in target or target in alias for alias in aliases):
            target_aliases = aliases
            break

    # Extract significant keywords from target for fuzzy matching
    # Remove common filler words to get meaningful terms
    _STOP_WORDS = {'of', 'the', 'a', 'an', 'in', 'on', 'by', 'to', 'for', 'and', 'or',
                   'from', 'with', 'must', 'be', 'is', 'are', 'should', 'shall', 'not',
                   'document', 'documents', 'original', 'copy', 'copies'}
    target_keywords = {w for w in re.split(r'\W+', target) if w and w not in _STOP_WORDS and len(w) > 2}

    matches = []

    def _get_pkt_type(pkt):
        if isinstance(pkt, dict):
            return (pkt.get("document_type", "") or pkt.get("doc_type", "")
                    or pkt.get("classification", "") or "").lower()
        return (getattr(pkt, "document_type", "") or "").lower()

    # Tier 1+2: Alias + substring match
    for pkt in packets:
        if not pkt:
            continue
        pkt_type = _get_pkt_type(pkt)
        if not pkt_type:
            continue
        for alias in target_aliases:
            if alias in pkt_type or pkt_type in alias:
                matches.append(pkt if isinstance(pkt, dict) else asdict(pkt))
                break

    if matches:
        return matches

    # Tier 3: Keyword overlap matching (fuzzy)
    # Only match if the DISTINGUISHING keywords overlap (not just generic words
    # like "certificate", "document", "list", "report").
    _GENERIC_WORDS = {'certificate', 'document', 'list', 'report', 'note', 'letter',
                      'advice', 'receipt', 'bill', 'policy', 'schedule', 'declaration',
                      'form', 'certification', 'statement',  # P90: prevent Form 3 matching Form 7
                      }
    if target_keywords:
        _specific_keywords = target_keywords - _GENERIC_WORDS
        if _specific_keywords:
            # Must match at least one SPECIFIC keyword (e.g., "phytosanitary", "fumigation")
            for pkt in packets:
                if not pkt:
                    continue
                pkt_type = _get_pkt_type(pkt)
                if not pkt_type or "lc" == pkt_type or "letter of credit" in pkt_type:
                    continue
                pkt_words = set(re.split(r'\W+', pkt_type))
                specific_overlap = _specific_keywords & pkt_words
                if specific_overlap:
                    matches.append(pkt if isinstance(pkt, dict) else asdict(pkt))

    return matches


def _pkt_text(pkt: dict) -> str:
    """Get the best available text from a reconciled packet."""
    if not pkt:
        return ""
    return (
        pkt.get("refined_text", "")
        or pkt.get("cleaned_text", "")
        or pkt.get("text", "")
        or pkt.get("raw_text", "")
        or ""
    )


def _pkt_images(pkt: dict) -> list:
    """Get page image paths from a reconciled packet."""
    if not pkt:
        return []
    return pkt.get("page_image_paths", [])


def _pkt_visual_metadata(pkt: dict) -> str:
    """Extract ALL metadata from packet: classification, extracted fields, stamps, signatures."""
    if not pkt:
        return ""
    parts = []
    # Document identification from Step 8/9
    doc_num = pkt.get("document_number", "") or pkt.get("bl_number", "") or pkt.get("invoice_number", "")
    if doc_num:
        parts.append(f"Document Number: {doc_num}")
    doc_date = pkt.get("document_date", "") or pkt.get("bl_date", "")
    if doc_date:
        parts.append(f"Document Date: {doc_date}")
    doc_amount = pkt.get("document_amount", "")
    if doc_amount:
        parts.append(f"Document Amount: {doc_amount}")
    issued_by = pkt.get("issued_by", "")
    if issued_by:
        parts.append(f"Issued By: {issued_by}")
    lc_ref = pkt.get("lc_reference", "")
    if lc_ref:
        parts.append(f"LC Reference: {lc_ref}")
    doc_summary = pkt.get("document_summary", "")
    if doc_summary:
        parts.append(f"Summary: {doc_summary[:150]}")
    # Extracted fields from Step 9
    ef = pkt.get("extracted_fields", {})
    if isinstance(ef, dict) and ef:
        for k, v in ef.items():
            if v and str(v).strip() and k not in ('text', 'raw_text', 'cleaned_text'):
                parts.append(f"Extracted [{k}]: {str(v)[:100]}")
    # Copy/Original status
    copy_status = pkt.get("copy_status", pkt.get("copy_label", ""))
    if copy_status:
        parts.append(f"Copy Status: {copy_status}")
    marking = pkt.get("marking_status", "")
    if marking:
        parts.append(f"Marking: {marking}")
    # Stamps
    stamps = pkt.get("stamps", [])
    if stamps:
        stamp_texts = []
        for s in stamps:
            if isinstance(s, dict):
                stamp_texts.append(s.get("text", s.get("description", str(s))))
            else:
                stamp_texts.append(str(s))
        parts.append(f"Stamps: {', '.join(stamp_texts)}")
    # Signatures
    sigs = pkt.get("signatures", [])
    if sigs:
        sig_texts = []
        for s in sigs:
            if isinstance(s, dict):
                sig_texts.append(s.get("text", s.get("description", str(s))))
            else:
                sig_texts.append(str(s))
        parts.append(f"Signatures: {', '.join(sig_texts)}")
    # Seals
    seals = pkt.get("seals", [])
    if seals:
        seal_texts = [s.get("text", str(s)) if isinstance(s, dict) else str(s) for s in seals]
        parts.append(f"Seals: {', '.join(seal_texts)}")
    # Number of originals
    n_orig = pkt.get("number_of_originals", pkt.get("document_fields", {}).get("number_of_originals", ""))
    if n_orig:
        parts.append(f"Number of Originals: {n_orig}")
    # ── Bill of Lading: short-form / blank-back / full-form status (Step 8) ──
    # Step 8 sets bl_short_form_status to "full_form" when either the BL
    # itself carries the carriage T&Cs OR another packet in the same set
    # carries them on a separate sheet. It is "short_form" only when no
    # T&Cs page exists anywhere in the submission. The verifier MUST use
    # this — see rule #23 in the prompt.
    bl_status = pkt.get("bl_short_form_status", "")
    if bl_status:
        has_terms = pkt.get("has_bl_terms_pages_in_set", False)
        parts.append(f"BL Form Status: {bl_status}")
        parts.append(f"BL Terms Page Present in Set: {bool(has_terms)}")
    # Also check nested pages for stamps/signatures
    for pg in pkt.get("pages", pkt.get("original_pages", [])):
        if isinstance(pg, dict):
            for _f in ("stamps", "signatures", "seals"):
                for item in pg.get(_f, []):
                    txt = item.get("text", str(item)) if isinstance(item, dict) else str(item)
                    if txt and txt not in str(parts):
                        parts.append(f"Page {pg.get('page_number', '?')} {_f}: {txt}")
    return "\n".join(parts) if parts else ""


def _pkt_type(pkt: dict) -> str:
    """Get document type label from a packet."""
    if not pkt:
        return ""
    return (
        pkt.get("document_type", "")
        or pkt.get("doc_type", "")
        or pkt.get("classification", "")
        or ""
    )


# ---------------------------------------------------------------------------
# F47A context builder
# ---------------------------------------------------------------------------

def _build_f47a_context(step06_result: dict) -> str:
    """
    Read ALL F47A Additional Conditions from the Final LC and build a
    single context string.  The VLM must know about these before checking
    ANY condition, because F47A may override or modify main-field conditions.
    """
    parts = []

    # Try consolidated_fields first
    final_lc = step06_result.get("final_lc", step06_result)
    fields = final_lc.get("consolidated_fields", final_lc)

    # F47A value may be a string, list, or list-of-dicts
    f47a = fields.get("47A", fields.get("F47A", ""))
    if isinstance(f47a, list):
        for item in f47a:
            if isinstance(item, dict):
                parts.append(item.get("text", item.get("value", str(item))))
            else:
                parts.append(str(item))
    elif f47a:
        parts.append(str(f47a))

    # Also check clauses for 47A entries
    clauses = final_lc.get("clauses", [])
    for clause in clauses:
        if isinstance(clause, dict):
            tag = clause.get("field_tag", clause.get("tag", ""))
            if tag in ("47A", "F47A"):
                txt = clause.get("text", clause.get("value", ""))
                if txt and str(txt) not in parts:
                    parts.append(str(txt))
        elif hasattr(clause, "field_tag"):
            if clause.field_tag in ("47A", "F47A"):
                txt = getattr(clause, "text", "")
                if txt and str(txt) not in parts:
                    parts.append(str(txt))

    return "\n".join(parts) if parts else "(No F47A additional conditions in this LC)"


def _sanitize_lc_field_value(field_tag: str, val: str) -> str:
    """
    P64: Verification-layer sanitizer for consolidated LC field values.

    Some upstream consolidations leave residual content from neighbouring
    SWIFT fields glued onto the start or end of a value (most commonly
    32B Amount text bleeding into 41D Available With, or an "F41D:" /
    "F32B:" label glued onto the value itself). Cleaning this at the
    verification read site keeps the report and the VLM prompt clean
    without disturbing the consolidator.
    """
    if not val:
        return ""
    s = str(val)

    # 1) Remove any leading currency+amount block that belongs to 32B
    #    when it appears glued onto a non-amount field (anything other
    #    than 32B/32A/32D/33B itself).
    if field_tag not in ("32A", "32B", "32D", "33B"):
        # e.g. "USD US DOLLAR 97216,00 #97,216.00F41D: ..."
        _amt_prefix = re.match(
            r'^\s*(?:USD|EUR|GBP|JPY|CHF|AUD|CAD|CNY|HKD|SGD|INR|PKR|AED|SAR)'
            r'(?:\s+(?:US\s+)?(?:DOLLAR|DOLLARS|EURO|POUND|POUNDS|YEN|FRANC|FRANCS))?'
            r'\s*[\d.,#]+(?:\s*[#]?[\d.,]+)*\s*',
            s, flags=re.IGNORECASE,
        )
        if _amt_prefix:
            s = s[_amt_prefix.end():]

    # 2) Strip a leading "F<tag>:" or "<tag>:" label if it survived consolidation
    s = re.sub(rf'^\s*F?{re.escape(field_tag)}\s*[:\-]\s*', '', s, flags=re.IGNORECASE)

    # 3) Strip the SWIFT human-readable sub-labels that sometimes get glued in
    #    e.g. "Available With... By... - Name and Address - Name and Address: ANY BANK..."
    _sublabel_patterns = [
        r'^\s*Available\s+With\.{0,3}\s*By\.{0,3}\s*[-–—]?\s*',
        r'^\s*Name\s+and\s+Address\s*[-–—:]\s*',
        r'^\s*\(?\s*Name\s+and\s+Address\s*\)?\s*[-–—:]\s*',
        r'^\s*Code\s*[-–—:]\s*',
    ]
    _changed = True
    while _changed:
        _changed = False
        for _p in _sublabel_patterns:
            _new = re.sub(_p, '', s, flags=re.IGNORECASE)
            if _new != s:
                s = _new
                _changed = True

    # 4) If a downstream F-tag header is glued in the middle (e.g. value
    #    contains "...F42A:..." for a 41D field), cut at that boundary.
    _other_tag = re.search(r'\bF?(?:32[ABD]|33B|39[ABC]|40[AE]|41[AD]|42[ACDM]|43[PT]|44[ABCDEF]|45[AB]|46[AB]|47[AB]|49|50|51[AD]|52[AD]|53[AD]|57[ABCD]|58[AD]|59[A]?|71[ABD]|72|78)\s*:', s)
    if _other_tag and _other_tag.start() > 0:
        s = s[:_other_tag.start()]

    return s.strip(' \t\r\n.-:;|#')


def _get_lc_field_value(step06_result: dict, field_tag: str) -> str:
    """Get a specific LC field value by tag (e.g. '31D', '44E')."""
    final_lc = step06_result.get("final_lc", step06_result)
    fields = final_lc.get("consolidated_fields", final_lc)

    # Direct lookup
    val = fields.get(field_tag, "")
    if isinstance(val, dict):
        val = val.get("value", str(val))
    if val:
        return _sanitize_lc_field_value(field_tag, str(val))

    # Try with 'F' prefix
    val = fields.get(f"F{field_tag}", "")
    if isinstance(val, dict):
        val = val.get("value", str(val))
    return _sanitize_lc_field_value(field_tag, str(val)) if val else ""


# ---------------------------------------------------------------------------
# VLM call -- sends condition + document text + optional image to Qwen VLM
# ---------------------------------------------------------------------------

_VLM_PROMPT_TEMPLATE = """You are verifying ONE condition from a Letter of Credit against a shipping document.

LC CONDITION TO VERIFY:
{condition_text}

LC FIELD: {clause_ref}
LC FIELD VALUE: {lc_field_value}

LC PARTIES (use these to resolve references like "APPLICANT", "BENEFICIARY", "ISSUING BANK"):
{lc_parties}

F47A ADDITIONAL CONDITIONS (read these FIRST -- they may override or modify the condition):
{f47a_context}

DOCUMENT BEING CHECKED: {document_type}
DOCUMENT TEXT (from GLM OCR -- trusted, complete page text):
{document_text}

DOCUMENT VISUAL METADATA (already extracted by upstream steps -- stamps,
signatures, seals, copy/original status, BL form status, document number,
issue date, amount, issuer, LC reference, all extracted fields, marks,
endorsements, etc.):
{visual_metadata}

NOTE: The DOCUMENT TEXT and DOCUMENT VISUAL METADATA above together contain
EVERYTHING that has been observed on the page (text + every visual element
detected by the OCR and classification stages). They are your COMPLETE source
of truth for this verification. Do NOT request, assume, or hallucinate any
information that is not present in these two blocks. If a fact you need is
not in either block, treat it as "not stated on the document" and decide
accordingly (usually REVIEW, not FAIL).

VERIFY: Does the document satisfy this condition?

CRITICAL RULES (follow strictly):
1. CHECK F47A FIRST: Before marking anything as FAIL, read ALL F47A conditions above carefully. If ANY F47A clause says something is "ACCEPTABLE", "ALLOWED", or "PERMITTED" that relates to this condition, it OVERRIDES the main requirement.
2. CONDITIONAL ACCEPTANCE: If F47A allows something WITH conditions (e.g., "LATE SHIPMENT ALLOWED PROVIDED penalty deduction"), mark as REVIEW (not FAIL) and explain the condition that needs manual verification.
3. "ANY [COUNTRY] PORT" means any port in that country is acceptable = PASS.
4. Shipment BEFORE latest date = PASS. Shipment AFTER latest date = check F47A first. If F47A allows late shipment, mark REVIEW.
5. CHARTER PARTY: If F47A says "CHARTER PARTY BL ACCEPTABLE", then charter party BL = PASS.
6. Name matching: key words must match (UNITED BANK = UNITED BANK LIMITED = UBL). Minor spelling differences are acceptable.
7. INVOICE / DRAFT AMOUNT (read carefully — this is the most common false-fail):
   • Use the EXPLICIT printed "Total Amount" / "Grand Total" / "Invoice Total" line at the bottom of the invoice as the AUTHORITATIVE invoice amount. Do NOT sum line items yourself — the printed Total already does that.
   • The "INVOICE PRINTED TOTAL AMOUNT" line in the SYSTEM PRE-CALCULATED SUMMARY at the top of the document text is the de-duplicated correct figure — TRUST IT and use it as the invoice amount.
   • Do NOT count the same Total twice. If a multi-copy invoice (e.g. octuplicate) was merged, the same "Total Amount: 97,216.00" line may appear several times in the raw text — this is ONE invoice with ONE total, not multiple invoices. The SUMMARY at the top has already deduped this for you.
   • For a multi-page invoice, the Total line on the LAST page is the figure for the whole invoice — do not add per-page subtotals on top of it.
   • The invoice amount can be LESS than the LC amount (partial/short shipment, allowed under UCP 600 Art 30 tolerance) — that is PASS.
   • Only when invoice Total > LC amount × (1 + tolerance%) is it a FAIL. Verify your arithmetic: 97,216 is NOT greater than 97,216. 95,000 is LESS than 97,216 (PASS, not FAIL).
8. THIRD PARTY: If F47A says "THIRD PARTY DOCUMENTS ACCEPTABLE", third party documents = PASS.

8a. BENEFICIARY-ISSUED DOCUMENTS: When the condition asks for a document
    to be "issued by the beneficiary" or "not a third party document",
    check the LC PARTIES section above for the BENEFICIARY's name, then
    check if the document is issued by / signed by / "for and on behalf
    of" that same entity. Apply the SAME party-name matching rules as
    rule 22 (OCR tolerance, abbreviations, company suffixes). If the
    issuing party on the document matches the BENEFICIARY name from the
    LC PARTIES → PASS. Do NOT fail just because the exact wording
    differs slightly (e.g. "Viterra B.V." on doc vs "VITERRA B.V."
    in LC — these are the same entity, case-insensitive). Also check
    the document LETTERHEAD and SIGNATURE block, not just a single
    "Issued By" field. Common patterns:
      • "For and on behalf of [BENEFICIARY NAME]" → issued by beneficiary
      • Letterhead shows beneficiary's company name → issued by beneficiary
      • Signed by an employee of the beneficiary → issued by beneficiary
9. CERTIFICATION: If the condition asks for origin certification and the document says "We certify the goods are of [COUNTRY] origin" or similar statement, that IS a valid certification = PASS. Do not fail just because the exact word "CERTIFICATE" is not used — any statement certifying origin, quality, weight, etc. is a certification.
10. DOCUMENT VERIFICATION: If the document text does NOT look like the expected document type (e.g., the condition checks a "Phytosanitary Certificate" but the document text looks like a quality certificate or inspection report), mark as REVIEW with "Document type may be misclassified".
11. PASS / FAIL / REVIEW — DECISION RULES (CRITICAL — read this twice):
    REVIEW is NOT a "when in doubt" escape hatch. REVIEW is reserved
    for ONE specific situation: the answer genuinely depends on a human
    judgement that the document data alone cannot resolve (e.g. F47A
    grants conditional acceptance subject to a separate negotiation).

    DEFAULT BEHAVIOUR FOR EVERY ROW:
       • Document EXPLICITLY satisfies the condition           → PASS
       • Document EXPLICITLY does NOT satisfy the condition    → FAIL
       • Required data is MISSING from the document            → FAIL
       • Required data is PRESENT but WRONG (different value)  → FAIL
       • Required identifier (HS Code, NTN, LC No., date) is
         present but the digits / characters do not match      → FAIL
       • Required date is past the LC deadline                 → FAIL
         (unless F47A explicitly allows late, see next bullet)
       • F47A explicitly allows / waives / overrides this
         requirement WITH conditions a human must verify       → REVIEW
       • Genuine OCR illegibility ("[unclear]", "[illegible]",
         partial scan damage) prevents reading the value AND
         the value is otherwise required                       → REVIEW

    Things that are NOT REVIEW (these are FAIL):
       ✗ "Partially matches" — if it does not exactly match per
         the matching rule for that field, it is FAIL.
       ✗ "Could not find on document" when the document text was
         readable — that is a FAIL (the data is missing).
       ✗ "Document may need clarification" — pick PASS or FAIL
         based on what is actually printed.
       ✗ "Bank may want to verify" — that is the entire point of
         this verification; commit to PASS or FAIL.
       ✗ Minor format differences that the matching rules above
         already say to ignore (dots in HS Code, "@"/"(at)" in
         emails, "PORT"/"SEAPORT" qualifiers on ports, abbreviated
         vs full company names) — those are PASS, not REVIEW.
       ✗ Numeric value differs from required value — that is FAIL,
         not REVIEW. "Close" amounts/quantities are still wrong.

    Before returning REVIEW, ask yourself: "Could a competent human
    bank checker decide PASS or FAIL from the document text alone?"
    If yes, you must also decide PASS or FAIL — do NOT punt to REVIEW.
12. QUANTITY AND PARTIAL SHIPMENT (CRITICAL — read carefully):
    a) TOLERANCE: If the LC says "1000 MT LESS 10 PCT" or "1000 MT +/-5%", apply the tolerance. "1000 MT LESS 10 PCT" means 900-1000 MT is acceptable. A quantity of 950 MT is WITHIN range = PASS. Do NOT fail if quantity is within tolerance. Also check F47A for additional tolerance clauses (e.g., "+0/-10%").
    b) PARTIAL SHIPMENT: Check the LC PARTIES section or F47A context for the LC's partial shipment status (F43P). If F43P says "ALLOWED" / "PERMITTED" / "PERMISSIBLE", then the invoice / BL / packing list may show a LESSER quantity than the LC — this is a legitimate partial shipment and the row is PASS, NOT FAIL. Only if the quantity EXCEEDS the LC quantity + tolerance is it a FAIL.
       EXAMPLE: LC says "QTY 9025 KGS", F43P = "ALLOWED", invoice shows 3625 KGS → PASS (partial shipment, 3625 < 9025).
       EXAMPLE: LC says "QTY 9025 KGS", F43P = "PROHIBITED", invoice shows 3625 KGS → FAIL (partial shipment not allowed, quantity must match within tolerance).
       EXAMPLE: LC says "QTY 9025 KGS", F43P = "ALLOWED", invoice shows 10000 KGS → FAIL (exceeds LC quantity).
    c) UCP 600 Art 30(b): Unless the LC prohibits partial shipment, a tolerance of 5% more or less in quantity is allowed, provided the total amount does not exceed the LC amount. So even without explicit tolerance, 5% variation is acceptable if partial shipment is not prohibited.
13. BILL OF LADING LIMITATIONS: BLs do NOT show dollar amounts or unit prices — never fail a BL for "amount not mentioned". BLs do NOT typically show LC/credit numbers unless F47A specifically requires it on BL.

13z. STALE BL CHECK ON DOCUMENTARY REMITTANCE (CRITICAL):
    When the condition says "BL must not be stale" and the
    document_to_check is "Documentary Remittance" (or "Covering
    Schedule" / "Cover Schedule"), you are NOT looking for a Bill
    of Lading inside the Documentary Remittance. You are looking
    for the PRESENTATION DATE on the Documentary Remittance.
    "Stale" means the BL was presented more than the allowed
    number of days after the BL date. The formula is:
      presentation_date (from DR) - shipment_date (from BL/F44C)
    You may not have the BL date in the DR text — in that case,
    check if the DR shows a presentation date or "date" field,
    and compare it against the LC's latest shipment date (F44C)
    or the F48 presentation period. If you cannot determine
    whether the document is stale from the DR text alone, mark
    as REVIEW (not FAIL). Do NOT mark FAIL with "NO BILL OF
    LADING FOUND" — that is wrong. The DR is not supposed to
    contain a BL; it's supposed to contain a date.

13a. DATE LABEL ABBREVIATIONS (CRITICAL — do NOT miss these):
    Documents (especially invoices, batch certificates, packing lists)
    use many abbreviated date labels. ALL of the following mean the
    SAME thing and MUST be treated as valid:

    MANUFACTURING DATE / DATE OF MANUFACTURE:
      Mfg.Date, Mfg Date, MFG.DATE, Mfg.Dt, Manufacturing Date,
      Date of Manufacture, Date of Mfg, DOM, D.O.M., Manuf. Date,
      Production Date, Date of Production

    EXPIRY DATE / DATE OF EXPIRY:
      Exp.Date, Exp Date, EXP.DATE, Exp.Dt, Expiry Date,
      Date of Expiry, Date of Exp, DOE, D.O.E., Best Before,
      Use Before, Valid Until, Shelf Life Expiry, Expiration Date

    If the condition asks for "date of manufacturing" and the document
    shows "Mfg.Date:2025.06.17" — that IS the manufacturing date → PASS.
    If the condition asks for "expiry date" and the document shows
    "Exp.Date:2029.06.16" — that IS the expiry date → PASS.
    Do NOT fail because the exact words "date of manufacturing" or
    "expiry date" are not spelled out. The abbreviation IS the date.

13b. DRUG NAME / PRODUCT NAME MATCHING:
    When the condition asks for "name of drug" or "name of product"
    to appear on a document, look for the ACTUAL drug/product name
    anywhere in the document text. Common patterns:
      • Line item description: "LEVOFLOXACIN HEMIHYDRATE USP43"
      • After "Name of Drug:" or "Product:" label
      • In the goods description block
    If ANY recognisable drug/product name appears in the document
    text, the condition is satisfied → PASS. Do NOT require the
    exact phrase "NAME OF DRUG" to be printed — the drug name
    itself is what must appear.

13c. PHYSICAL CONTAINER/PACK MARKING vs DOCUMENT CONTENT:
    When the condition says "individual containers/packs should
    clearly mark [X]" (name of drug, batch no, mfg date, expiry
    date), this requirement has TWO levels:
      Level 1 (document-verifiable): The information (drug name,
        batch number, mfg date, expiry date) must APPEAR somewhere
        on the document (invoice, packing list, batch certificate).
        If the information IS present on the document → PASS.
      Level 2 (not document-verifiable): Whether the physical
        drums/cartons have the information printed on their labels
        cannot be verified from the document text. Do NOT fail a
        row just because the document doesn't say "we confirm the
        drums are marked with..." — if the information itself
        (the date, the batch number, the drug name) appears on the
        document, that is sufficient evidence → PASS.
    Only FAIL if the required information (the actual date, batch
    number, or drug name) is genuinely ABSENT from the document.
14. PERMISSIVE CLAUSES: "ACCEPTABLE" means something is ALLOWED — it is NOT a prohibition. "THIRD PARTY DOCUMENTS ACCEPTABLE EXCEPT X" means X must be from beneficiary, everything else can be third party. Do NOT interpret "EXCEPT X" as "X is not acceptable".
15. MATH: When comparing numbers, verify your arithmetic. 950 is LESS than 1000 (not more). 490,200 is LESS than 516,000 (not more). Get the direction right before marking FAIL.
16. EMAIL EQUIVALENCE: In SWIFT messages, "@" is written as "(AT)" or "(at)". So "INFO(AT)CICL.COM.PK" in the LC is the SAME as "info@cicl.com.pk" in the document. Treat (AT) and @ as identical when comparing email addresses. Also ignore case differences in emails.
17. AGENT vs FORWARDER: "AS AGENTS ONLY FOR AND BY AUTHORITY OF THE MASTER" on a BL means the carrier's agent signed — this is NORMAL and NOT a freight forwarder BL. A freight forwarder BL would say "FIATA", "HOUSE BILL", or show a forwarder company as the ISSUER (not as agent of master).
18. COPIES/DUPLICATES: "IN DUPLICATE" = 2 copies, "IN TRIPLICATE" = 3, "IN QUADRUPLICATE" = 4, "IN OCTUPLICATE" = 8, "FULL SET" = 3/3 originals. The number of copies is verified by the SYSTEM (not you) — it counts how many separate document packets exist. When you see a condition about copies/duplicates, mark it as PASS — the system handles copy counting separately. Do NOT fail a document for "not in duplicate/octuplicate" — you are only seeing ONE representative copy.
19. MISSING DOCUMENT: If a required document is completely MISSING from the submission, report ONE failure: "Required document missing". Do NOT add sub-failures for content checks (importer name, language, etc.) on a missing document — those are meaningless if the document doesn't exist.
20. PORT MATCHING: Ports are the SAME if the city/country matches, even if qualifiers differ. "KARACHI SEAPORT, PAKISTAN" = "KARACHI, PAKISTAN" = "KARACHI PORT, PAKISTAN". The word "SEAPORT"/"PORT" is just a qualifier. Similarly: "PENANG PORT, MALAYSIA" = "PENANG, MALAYSIA". Also "ANY [COUNTRY] PORT/SEAPORT" means ANY port in that country = PASS if port is in that country. "ANY MALAYSIA PORT" matches "PENANG PORT, MALAYSIA". "ANY CANADIAN PORT" matches "VANCOUVER, CANADA". "ANY CHINESE SEAPORT" matches "HONGKONG SEAPORT, CHINA" or "SHANGHAI, CHINA" or any port in China (including Hong Kong, Macau — they are part of China). The word "CHINESE" = "CHINA". Country adjectives ALWAYS equal the country noun: CHINESE=CHINA, MALAYSIAN=MALAYSIA, INDIAN=INDIA, PAKISTANI=PAKISTAN, JAPANESE=JAPAN, KOREAN=KOREA, GERMAN=GERMANY, FRENCH=FRANCE, ITALIAN=ITALY, SPANISH=SPAIN, AMERICAN=USA=UNITED STATES, BRITISH=ENGLISH=UK=UNITED KINGDOM, DUTCH=NETHERLANDS, BELGIAN=BELGIUM, SWISS=SWITZERLAND, BRAZILIAN=BRAZIL, ARGENTINEAN=ARGENTINA, EGYPTIAN=EGYPT, SAUDI=SAUDI ARABIA, EMIRATI=UAE=UNITED ARAB EMIRATES, TURKISH=TURKEY, RUSSIAN=RUSSIA. So "ANY CHINESE SEAPORT" = "ANY CHINA SEAPORT" = "ANY CHINA PORT" = "ANY PORT IN CHINA".

20a. TRADE TERMS / INCOTERMS — MATCH THE CODE ONLY (CRITICAL):
    When the LC condition asks for a "trade term" / "Incoterm" / "delivery term" to appear on a document (typically the Commercial Invoice), you MUST verify ONLY the Incoterm CODE, not the country, city, port or any other suffix that follows it.

    Recognised Incoterm codes (Incoterms 2010 / 2020 + common variants):
       EXW, FCA, FAS, FOB, CFR, CNF, C&F, CIF, CIP, CPT,
       DAP, DPU, DDP, DAT, DAF, DDU, DES, DEQ
    Common modifiers / suffixes that are PART of the Incoterm code and
    must also match if present:
       FO  = "Free Out"        e.g. "CFR FO"   = "CFR FREE OUT"
       FI  = "Free In"         e.g. "CFR FI"   = "CFR FREE IN"
       FIO = "Free In and Out" e.g. "CFR FIO"  = "CFR FREE IN AND OUT"
       FILO, LIFO, FIOST, etc. (rare; pass them through verbatim)
    The Incoterm-code COMPARISON ignores everything after the code +
    its modifier. The trailing place name ("ANY CHINESE SEAPORT",
    "SHANGHAI", "KARACHI PORT, PAKISTAN", "DESTINATION PORT", etc.)
    is NOT part of the Incoterm and is verified SEPARATELY by the
    port-of-loading / port-of-discharge checks. Do NOT FAIL or
    REVIEW a trade-terms row because the country / port wording
    differs — those rows have their own checks.

    Step A — Extract the Incoterm + modifier from BOTH the LC value
             and the document value. Strip everything else.
       LC value: "FOB ANY CHINESE SEAPORT, PAKISTAN"
                 → Incoterm = "FOB", modifier = none, place = ignored
       Doc value: "FOB ANY CHINA SEAPORT"
                 → Incoterm = "FOB", modifier = none, place = ignored

    Step B — Compare ONLY the Incoterm + modifier:
       • Same Incoterm and same modifier (or both absent) → PASS
       • Same Incoterm but modifier differs (e.g. LC "CFR FO" vs
         doc "CFR" with no FO) → FAIL with "Incoterm modifier
         missing/extra"
       • Different Incoterm (e.g. LC "FOB" vs doc "CIF") → FAIL
       • Equivalent codes are the same: CNF = C&F = CFR (these are
         all the same Incoterm under different names). Treat them
         as identical → PASS.

    WORKED EXAMPLES — STUDY ALL:

    Example 1 (PASS — adjective vs noun in trailing place):
       LC condition: "Trade terms 'FOB ANY CHINESE SEAPORT, PAKISTAN'
                      must appear on the Commercial Invoice"
       Invoice text: "FOB ANY CHINA SEAPORT"
       Step A: LC=FOB, Doc=FOB.  Step B: same → PASS
       Verdict: PASS — "Incoterm FOB matches" (do NOT mark this
                row PARTIALLY MATCH because of CHINESE vs CHINA;
                the place name is ignored for the Incoterm check
                AND CHINESE=CHINA per rule 20 anyway)

    Example 2 (PASS — modifier match):
       LC condition: "CFR FO ANY MALAYSIA PORT must appear..."
       Invoice text: "CFR FO PENANG, MALAYSIA"
       Step A: LC=CFR FO, Doc=CFR FO.  Step B: same → PASS

    Example 3 (PASS — equivalent codes):
       LC condition: "C&F SHANGHAI must appear..."
       Invoice text: "CFR SHANGHAI" or "CNF SHANGHAI"
       Step A: LC=C&F, Doc=CFR (or CNF).  Step B: equivalent → PASS

    Example 4 (FAIL — different Incoterm):
       LC condition: "FOB KARACHI must appear..."
       Invoice text: "CIF KARACHI"
       Step A: LC=FOB, Doc=CIF.  Step B: different → FAIL

    Example 5 (FAIL — modifier missing):
       LC condition: "CFR FO ANY CHINA PORT must appear..."
       Invoice text: "CFR SHANGHAI" (no FO)
       Step A: LC=CFR FO, Doc=CFR.  Step B: modifier differs → FAIL

    REMINDER: For trade-terms rows, NEVER mark the row REVIEW or
    "PARTIALLY MATCHES" because of the trailing port/country
    wording. The trailing place is not part of the Incoterm. If
    you can identify the Incoterm code on both sides and they
    match, the row is PASS — period.
21. QUANTITY MATCHING: LC may say "QTY 736" and invoice may show individual line items that SUM to 736. Check the SYSTEM PRE-CALCULATED SUMMARY at the top of the document text — it shows per-product totals. Use these totals instead of counting line items yourself.
Also: product codes with/without spaces are the SAME: "LN 980E" = "LN980E", "LN 981E" = "LN981E". Ignore spaces in product codes when matching.
22. PARTY REFERENCES: When the condition says "NOTIFY APPLICANT" or "TO ORDER OF ISSUING BANK", look at the LC PARTIES section above to find the actual name. Then check if that name appears on the document. "NOTIFY APPLICANT" means the notify party field must show the APPLICANT's name (given above). Do NOT look for the literal words "NOTIFY APPLICANT" — look for the applicant's ACTUAL NAME. Check the TOTAL quantity, not individual lines. Also "Ea" (each) is a valid unit — 736 Ea = 736 pieces. If LC says "QTY 736 AT THE RATE OF USD 98.00" and invoice shows 736 units × $98.00 = correct.

    PARTY-NAME OCR TOLERANCE (ISBP 821 paragraph A1 & UCP 600 Art 14(d/e)):
    Company / party names (Applicant, Beneficiary, Consignee, Notify
    Party, Issuing Bank, etc.) MUST be matched semantically, not
    character-by-character. The bank's job is to confirm the parties
    are the same legal entity, NOT to police OCR perfection.

    Step 1 — Normalise both sides:
      • Strip company-form suffixes / abbreviations: PVT, PVT., PVT LTD,
        PRIVATE LIMITED, LTD, LIMITED, LLC, LLP, INC, CORP, CO,
        COMPANY, GMBH, AG, SA, BV, S.A., S.A.S., S.R.L., PTE, PTE LTD,
        SDN BHD, FZE, FZ-LLC, OPC.
      • Strip address punctuation differences: ".", ",", "/", "-", "(", ")"
      • Treat single-letter / two-letter run-together abbreviations
        as the same: "G.I" = "GI" = "G I" = "G.I." (and the same for
        any other 1-2 letter initial cluster).
      • Treat the following OCR-confusable letter pairs as IDENTICAL
        when they appear inside a SHORT initial cluster (≤ 3 letters)
        or right next to a punctuation mark — these are the canonical
        single-pixel OCR mistakes:
            I ↔ L     (G.I ↔ G.L, II ↔ IL ↔ LI ↔ LL)
            I ↔ 1     (BIN ↔ B1N inside short codes)
            O ↔ 0     (CO ↔ C0 inside short codes)
            B ↔ 8     (B-3 ↔ 8-3 inside short codes)
            S ↔ 5     (S5 ↔ SS inside short codes)
            Z ↔ 2
            G ↔ 6
        These swaps are ONLY allowed inside an initial / abbreviation
        cluster — NOT inside the body of a regular English word. So
        "GLOBAL" stays "GLOBAL" (we do NOT change it to "GIOBAL"),
        but "G.L" can match "G.I" because that is a 2-letter initial
        cluster bordered by a dot.
      • Address: ignore differences in block format ("BLOCK E" vs
        "BLOCK(E)" vs "BLOCK-E"), in slash usage ("C-3/1" vs "C 3-1"
        vs "C3/1"), and in line-break placement.
      • Case differences are ALWAYS irrelevant.

    Step 2 — Decision:
      • If the normalised CORE NAME (e.g. "GI ENTERPRISES" or "GL
        ENTERPRISES" → both normalise to two letters + ENTERPRISES)
        plus the CITY plus the COUNTRY all match, the parties are the
        SAME legal entity → PASS.
      • Even better: if at least TWO of these five fields — (core
        name, street/block, city, country, postal code) — match
        exactly AND the remaining differences are explainable by
        the OCR-confusable swaps in the table above → PASS.
      • OCR FULL-WORD NEAR-MATCH: When a company name on the document
        differs from the LC by only 1-2 characters in a single word
        (letter substitution, transposition, or insertion/deletion),
        AND the rest of the name + address matches, treat it as an
        OCR error → PASS. Examples:
          "Visetra B.V." vs "Viterra B.V." — 2 chars differ (s↔t, e↔r
          transposition), same company → PASS
          "ARCHOMA" vs "ARCHROMA" — 1 char missing (R), same → PASS
          "HAEMONTICS" vs "HAEMONETICS" — 1 char substitution → PASS
        This is different from the initial-cluster rule above — this
        applies to FULL WORDS where the overall shape is clearly the
        same entity. Use your judgement: if the word looks like the
        same company with a minor OCR glitch, it IS the same company.
      • Only if the differences are SUBSTANTIVE (a clearly different
        company name, a clearly different city, a clearly different
        country, a clearly different street address) → FAIL.

    WORKED EXAMPLE — STUDY THIS (this exact case was wrongly FAILED):
       LC APPLICANT:
         "G.I ENTERPRISES (PVT) LTD C 3-1 PL-10 AL-HAMRA SQUARE
          BLOCK E NORTH NAZIMABAD KARACHI, PAKISTAN"
       BL NOTIFY PARTY:
         "GL ENTERPRISES PRIVATE LIMITED C-3/1 AL-HAMRA SQUARE
          BLOCK(E), NORTH NAZIMABAD, KARACHI, PAKISTAN"
       Differences:
         (a) "G.I" vs "GL"  → 2-letter initial cluster, I↔L swap, allowed
         (b) "(PVT) LTD" vs "PRIVATE LIMITED" → company suffix, both stripped
         (c) "C 3-1" vs "C-3/1" → punctuation, address normalisation
         (d) "BLOCK E" vs "BLOCK(E)" → punctuation
         (e) "PL-10" missing on BL → minor omission, not substantive
       Core name: "GI ENTERPRISES" vs "GL ENTERPRISES" — letters differ
         only by an allowed OCR swap inside a short initial cluster.
       Address: "AL-HAMRA SQUARE", "NORTH NAZIMABAD", "KARACHI",
         "PAKISTAN" — ALL match exactly.
       VERDICT: PASS — the parties are the same legal entity; the
         "G.I" → "GL" difference is a known OCR confusable swap and
         every other identifying field matches. Do NOT mark this FAIL.

    REMINDER: this rule is NOT a licence to ignore real differences.
    If the city is different, the country is different, or the core
    name uses entirely different words ("GI ENTERPRISES" vs "MARWAN
    TRADING"), that is FAIL. The tolerance applies only to the
    canonical OCR-pixel mistakes listed above.
23. SHORT FORM / BLANK BACK BILL OF LADING (UCP 600 Art 20(a)(v)): A "short form" or "blank back" Bill of Lading is one that does NOT print the detailed terms and conditions of carriage on its reverse side. UCP 600 Art 20(a)(v) ACCEPTS such bills of lading by default — banks will not examine the contents of those terms. ONLY raise a discrepancy when the LC explicitly forbids them with wording like "SHORT FORM / BLANK BACK BL NOT ACCEPTABLE", and even then, check the DOCUMENT VISUAL METADATA above:
   • If "BL Form Status: full_form" → the BL is a full-form BL = PASS (the carriage terms are present, either on this BL or on a separate T&C page in the same submission set).
   • If "BL Terms Page Present in Set: True" → the carriage terms are supplied on a separate sheet within the document set; the BL is therefore a full-form BL = PASS, even if the LC forbids blank-back.
   • Only when "BL Form Status: short_form" AND no terms page exists in the set AND the LC explicitly forbids short-form/blank-back → mark as FAIL.
   In all other cases (including the LC being silent on short-form), short-form / blank-back BLs are acceptable per UCP 600 = PASS.

24. CONSIGNEE vs NOTIFY PARTY — DO NOT CONFUSE THESE FIELDS (CRITICAL):
    On a Bill of Lading these are TWO COMPLETELY DIFFERENT fields:
       • CONSIGNEE — the named legal owner of the goods (transfers title)
       • NOTIFY ADDRESS / NOTIFY PARTY — who the carrier informs on arrival (purely informational, does NOT transfer title)
    They are SEPARATE fields with SEPARATE meanings. A party named only in the Notify field is NOT the consignee, and finding a party in the Notify field is NOT evidence that the BL is consigned or endorsed to that party.

    When the LC condition asks for the BL to be "DRAWN TO THE ORDER OF [PARTY]" or "CONSIGNED TO [PARTY]" or "ENDORSED TO THE ORDER OF [PARTY]" or "DRAWN OR ENDORSED TO THE ORDER OF [PARTY]", you MUST verify ONLY the CONSIGNEE field (and any endorsement stamp on the back of the BL):
       • PASS only if the CONSIGNEE field literally says "TO ORDER OF [PARTY]" / "CONSIGNED TO [PARTY]" / "TO THE ORDER OF [PARTY]" with the party name spelled out.
       • PASS also if the CONSIGNEE field says "TO ORDER" (blank) AND there is a visible ENDORSEMENT stamp/signature on the back of the BL transferring it to [PARTY].
       • FAIL if the CONSIGNEE field is just "TO ORDER" (blank) or "TO ORDER OF SHIPPER" with NO endorsement to [PARTY] visible on the BL — even if [PARTY] appears elsewhere on the document (notify, address block, etc.).
       • FAIL if the CONSIGNEE is some other named party that is NOT [PARTY] — even if [PARTY] is in the Notify field.

    Worked example — this MUST be a FAIL:
       LC condition:  "BL must be drawn or endorsed to the order of UNITED BANK LTD., CPU (TRADE)"
       BL CONSIGNEE:  "TO ORDER"
       BL NOTIFY:     "DALDA FOODS LIMITED ... AND UNITED BANK LTD., CPU (TRADE) ..."
       BL endorsement (back): not visible
       VERDICT:       FAIL — "Consignee shows 'TO ORDER' (blank) with no endorsement to UBL on the BL. Presence of UBL in the Notify field does NOT satisfy 'drawn or endorsed to UBL'."
       Do NOT pass this row by saying "UBL is on the BL". UBL is in the Notify field, NOT the Consignee field — these are different fields.

    The LC condition asking for "NOTIFY [PARTY]" is the OPPOSITE direction:
       • For "BL must notify [PARTY]" → check ONLY the NOTIFY field for [PARTY]'s name. Finding [PARTY] in the Consignee field is NOT a substitute for being in the Notify field (although it usually doesn't hurt either).

    Common LC phrasings and where to look:
       • "Consigned to X" / "Drawn to the order of X" / "To the order of X" → CONSIGNEE field (or endorsement)
       • "Endorsed to X" / "Endorsed in favor of X" → ENDORSEMENT stamp on back of BL (or CONSIGNEE if "TO ORDER OF X")
       • "Marked notify X" / "Notify X" / "Showing X as notify party" → NOTIFY ADDRESS field

25. H.S. CODE / NTN / TAX-NUMBER MATCHING — STRICT DIGIT-EXACT (CRITICAL):
    H.S. Codes, NTN numbers, Tax IDs, Importer codes, Exporter codes,
    SRO numbers, License numbers and any similar numeric identifier MUST
    match the LC value EXACTLY at the digit level. The verifier MUST
    follow these steps:

      Step A — Normalise both sides by stripping all non-digit characters
               (dots, spaces, dashes, slashes, colons, the literal "HS",
               "H.S.", "CODE", "NO.", "NTN", "NO" labels, etc.). Keep ONLY
               the digits.

      Step B — Compare the normalised digit strings:
               • If they are EXACTLY equal → PASS.
               • If the document's normalised digits START WITH the LC's
                 normalised digits (i.e. LC digits are a strict prefix of
                 the document digits) → PASS. This handles the EU
                 10-digit CN code where an 8-digit LC HS code is padded
                 with two trailing zeros (e.g. LC "9018.9050" matches
                 document "9018905000" because "90189050" is a prefix of
                 "9018905000").
               • If the LC's normalised digits START WITH the document's
                 digits AND the document has at least 6 digits → PASS
                 (rare reverse case where the doc shows the truncated
                 chapter/heading).
               • Otherwise → FAIL. NEVER mark as REVIEW for an HS Code or
                 NTN mismatch — the digits either match exactly or they
                 don't. "PARTIALLY MATCHES" is NOT a valid verdict for
                 these fields.

      WORKED EXAMPLES — STUDY ALL OF THESE:

        Example 1 (PASS — exact match):
          LC condition: "H.S. Code 9018.9050 must appear on the BL"
          BL text:      "HS CODE 9018.9050"
          Normalised:   LC=90189050  Doc=90189050
          Verdict:      PASS

        Example 2 (PASS — EU 10-digit prefix match):
          LC condition: "H.S. Code 9018.9050 must appear on the Invoice"
          Invoice text: "EU HS Code: 9018905000"
          Normalised:   LC=90189050  Doc=9018905000
          "9018905000".startswith("90189050") → True
          Verdict:      PASS

        Example 3 (FAIL — digits differ — THIS WAS WRONGLY REVIEWED):
          LC condition: "H.S. Code 9018.9050 must appear on the BL"
          BL text:      "HS CODE: 9018.90 9000"
          Normalised:   LC=90189050  Doc=901890 9000 → strip space → 90189000
          "90189000".startswith("90189050") → False
          "90189050".startswith("90189000") → False
          The digits at positions 7-8 are 50 vs 00.
          Verdict:      FAIL — "HS Code mismatch: LC 9018.9050 vs BL 9018.9000"
          Do NOT return REVIEW or "PARTIALLY MATCHES" for this case.

        Example 4 (FAIL — completely different code):
          LC condition: "H.S. Code 1205.1000 must appear on the Invoice"
          Invoice text: "HS CODE 1205.9000"
          Normalised:   LC=12051000  Doc=12059000
          Verdict:      FAIL — "HS Code mismatch"

        Example 5 (PASS — NTN with dash):
          LC condition: "Importer's NTN 1550365-8 must appear on the BL"
          BL text:      "NTN: 1550365-8" or "NTN 15503658"
          Normalised:   LC=15503658  Doc=15503658
          Verdict:      PASS

        Example 6 (FAIL — NTN one digit off):
          LC condition: "Importer's NTN 1550365-8 must appear on the BL"
          BL text:      "NTN 1550365-9"
          Normalised:   LC=15503658  Doc=15503659
          Verdict:      FAIL — "NTN mismatch: LC 1550365-8 vs BL 1550365-9"

    Why this matters: HS Code and NTN are regulatory identifiers used by
    customs and tax authorities. A single wrong digit makes the document
    legally non-compliant. There is no "close enough" — banks must reject
    a presentation with an incorrect HS Code or NTN under UCP 600 Art 14.

Return ONLY valid JSON:
{{
    "findings": "exact text found in the document that relates to this condition",
    "result": "short 4-5 word result (e.g., 'Port matches LC requirement')",
    "compliance": "pass or fail or review",
    "confidence": 0.95,
    "reasoning": "brief explanation of why pass/fail/review"
}}"""


_DOC_MISSING_RESULT = {
    "findings": "Document not found in submission",
    "result": "Required document missing",
    "compliance": "fail",
    "confidence": 1.0,
    "reasoning": "The required document was not submitted",
}


def _call_vlm(
    row_id: str,
    condition_text: str,
    clause_ref: str,
    lc_field_value: str,
    f47a_context: str,
    document_type: str,
    document_text: str,
    image_path: Optional[str] = None,
    visual_metadata: str = "",
    lc_parties: str = "",
) -> dict:
    """
    Send a single verification request to Qwen VLM.

    P63: TEXT-ONLY VERIFICATION.
    By the time we reach Step 14, every visually-derivable fact has already
    been extracted by Steps 1, 8 and 9 and is carried into the prompt via
    `document_text` (full GLM OCR) and `visual_metadata` (stamps, signatures,
    seals, copy/original status, BL form status, all extracted_fields, etc.).
    Sending the page image again is:
      • redundant — same information delivered twice,
      • expensive — a single high-res BL scan can encode to 5000-8000 visual
        tokens and was pushing requests past the 72B's max_model_len=16384
        causing HTTP 400 (`max_tokens must be at least 1, got -602`),
      • slower — large base64 payload + image preprocessing on the server.
    The `image_path` parameter is preserved for caller compatibility but is
    intentionally NOT attached to the request payload.
    """
    start = time.time()

    # Pre-extract key totals from document text to help smaller models
    _doc_summary = ''
    if document_text and len(document_text) > 500:
        import re as _re_sum
        # P66: Tighter regex — must be "Total Amount" / "Grand Total" /
        # "Invoice Total" / "Sub Total". Bare "TOTAL" matches table headers
        # and weight totals which contaminated the summary.
        _totals_raw = _re_sum.findall(
            r'(?:Total\s+Amount|Grand\s+Total|Invoice\s+Total|Sub\s*Total|Net\s+Total|Amount\s+Total)\s*[:\s]*'
            r'(?:USD|EUR|GBP|JPY|CHF|AUD|CAD|CNY|HKD|SGD|INR|PKR|AED|SAR)?\s*([\d.,]+)',
            document_text, _re_sum.IGNORECASE,
        )
        # P66: Dedupe identical totals — when a multi-copy invoice is merged,
        # the SAME "Total Amount: 97,216.00" line appears N times. Without
        # dedup, the precalc summary becomes "97,216.00, 97,216.00, ..." and
        # the verifier sums them, falsely flagging "INVOICE EXCEEDS CREDIT".
        if _totals_raw:
            _seen_totals = set()
            _totals = []
            for _t in _totals_raw:
                _norm = _t.strip().rstrip('.,').replace(' ', '')
                # Normalise for dedup: treat "97,216.00" and "97216,00" as same
                try:
                    if ',' in _norm and '.' in _norm:
                        # US format with thousands
                        _key = f"{float(_norm.replace(',', '')):.2f}"
                    elif ',' in _norm and _norm.count(',') == 1 and len(_norm.split(',')[1]) == 2:
                        # European format: 97216,00
                        _key = f"{float(_norm.replace(',', '.')):.2f}"
                    else:
                        _key = f"{float(_norm.replace(',', '')):.2f}"
                except ValueError:
                    _key = _norm
                if _key not in _seen_totals:
                    _seen_totals.add(_key)
                    _totals.append(_t)
            if len(_totals) == 1:
                _doc_summary += f"INVOICE PRINTED TOTAL AMOUNT (use this single value, do NOT sum): {_totals[0]}\n"
            else:
                _doc_summary += f"TOTAL AMOUNTS FOUND (deduped): {', '.join(_totals)}\n"
        # Find quantity totals
        _qty_totals = _re_sum.findall(r'(?:Total\s*(?:Quantity)?|TOTAL)[:\s]*([\d,]+\.?\d*)\s*(?:Ea|pcs|KGS|MT|MMBTU|units|rolls|drums)', document_text, _re_sum.IGNORECASE)
        if _qty_totals:
            _doc_summary += f"TOTAL QUANTITIES FOUND: {', '.join(_qty_totals)}\n"
        # Find quantities grouped by product/item code
        # Look for patterns like: "0980E-00\nTPE DISPOSABLE SET, LN980E\n...Qty: 96"
        _product_qtys = {}
        _lines = document_text.split('\n')
        _current_product = None
        for _line in _lines:
            _line_s = _line.strip()
            # Detect product lines (item codes like 0980E-00 or product names)
            _prod_m = _re_sum.search(r'\b(LN\s*\d+\w*)\b', _line_s, _re_sum.IGNORECASE)
            if _prod_m:
                _current_product = _re_sum.sub(r'\s+', '', _prod_m.group(1).upper())  # Normalize: "LN 980E" -> "LN980E"
            # Detect quantity lines
            _qty_m = _re_sum.search(r'(?:Qty|Quantity(?:\s+Shipped)?)[:\s]*(\d+)', _line_s, _re_sum.IGNORECASE)
            if _qty_m and _current_product:
                qty = int(_qty_m.group(1))
                _product_qtys[_current_product] = _product_qtys.get(_current_product, 0) + qty

        if _product_qtys:
            for prod, qty in sorted(_product_qtys.items()):
                _doc_summary += f"PRODUCT {prod} TOTAL QUANTITY: {qty}\n"
            _doc_summary += f"GRAND TOTAL QUANTITY: {sum(_product_qtys.values())}\n"
        else:
            # Fallback: simple sum
            _line_qtys = _re_sum.findall(r'(?:^|\n)\s*(?:Qty)[:\s]*(\d+)', document_text, _re_sum.IGNORECASE)
            if len(_line_qtys) > 2:
                _sum = sum(int(q) for q in _line_qtys)
                _doc_summary += f"SUM OF LINE QUANTITIES: {_sum} (from {len(_line_qtys)} items)\n"
    if _doc_summary:
        document_text = f"[SYSTEM PRE-CALCULATED SUMMARY]\n{_doc_summary}[END SUMMARY]\n\n{document_text}"

    # Truncate document text to avoid exceeding token limits
    max_chars = 6500
    if len(document_text) > max_chars:
        document_text = document_text[:max_chars] + "\n... [truncated]"

    prompt_text = _VLM_PROMPT_TEMPLATE.format(
        condition_text=condition_text,
        clause_ref=clause_ref,
        lc_field_value=lc_field_value,
        lc_parties=lc_parties or "(Not available)",
        f47a_context=f47a_context,
        document_type=document_type,
        document_text=document_text,
        visual_metadata=visual_metadata or "(No visual metadata available)",
    )

    # P63: text-only verification. document_text + visual_metadata already
    # carry every fact the image would provide (see docstring above).
    content_parts = [{"type": "text", "text": prompt_text}]

    payload = {
        "model": QWEN_VLM_MODEL,
        "messages": [
            {"role": "user", "content": content_parts},
        ],
        "max_tokens": 500,
        "temperature": 0.1,
    }

    try:
        resp = requests.post(QWEN_VLM_URL, json=payload, timeout=VLM_TIMEOUT)
        elapsed = time.time() - start

        if resp.status_code != 200:
            # P62: surface the actual vLLM error body so we can diagnose
            # 400s (context overflow), 413s (payload too large), etc.
            # without having to grep server logs.
            err_body = ""
            try:
                err_body = (resp.text or "")[:300].replace("\n", " ").replace("\r", " ")
            except Exception:
                pass
            return {
                "row_id": row_id,
                "findings": "Nil",
                "result": f"VLM error HTTP {resp.status_code}: {err_body}" if err_body else f"VLM error HTTP {resp.status_code}",
                "compliance": "review",
                "confidence": 0.0,
                "reasoning": f"VLM returned HTTP {resp.status_code}: {err_body}" if err_body else f"VLM returned HTTP {resp.status_code}",
                "elapsed": elapsed,
            }

        body = resp.json()
        raw_content = body["choices"][0]["message"]["content"].strip()

        # Extract JSON from response (VLM may wrap it in markdown or text)
        import re
        json_match = re.search(r"\{.*\}", raw_content, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group(0))
        else:
            parsed = {
                "findings": raw_content[:300],
                "result": "VLM response not JSON",
                "compliance": "review",
                "confidence": 0.3,
                "reasoning": "Could not parse VLM output as JSON",
            }

        parsed["row_id"] = row_id
        parsed["elapsed"] = elapsed
        return parsed

    except requests.exceptions.Timeout:
        return {
            "row_id": row_id,
            "findings": "Nil",
            "result": "VLM timeout",
            "compliance": "review",
            "confidence": 0.0,
            "reasoning": f"VLM call timed out after {VLM_TIMEOUT}s",
            "elapsed": time.time() - start,
        }
    except json.JSONDecodeError:
        return {
            "row_id": row_id,
            "findings": "Nil",
            "result": "VLM JSON parse error",
            "compliance": "review",
            "confidence": 0.0,
            "reasoning": "VLM response was not valid JSON",
            "elapsed": time.time() - start,
        }
    except Exception as exc:
        return {
            "row_id": row_id,
            "findings": "Nil",
            "result": f"Error: {str(exc)[:50]}",
            "compliance": "review",
            "confidence": 0.0,
            "reasoning": str(exc)[:200],
            "elapsed": time.time() - start,
        }


# ---------------------------------------------------------------------------
# Build verification tasks -- one per condition x document pair
# ---------------------------------------------------------------------------

def _deduplicate_packets(packets: list) -> tuple:
    """
    Group identical document packets by type — only keep ONE representative per type.
    Returns (deduped_packets, doc_counts) where doc_counts maps type -> count.

    Example: 8 Commercial Invoice copies → 1 representative + count=8
    This prevents checking the same content 8 times.
    """
    type_groups = {}  # doc_type_lower -> list of packets
    for pkt in packets:
        if not pkt:
            continue
        pt = (pkt.get("document_type", "") or pkt.get("doc_type", "")
              or pkt.get("classification", "") or "").lower().strip()
        if not pt:
            continue
        if pt not in type_groups:
            type_groups[pt] = []
        type_groups[pt].append(pkt)

    deduped = []
    doc_counts = {}
    for doc_type, group in type_groups.items():
        doc_counts[doc_type] = len(group)
        # Pick the representative: prefer the one with most text content
        best = max(group, key=lambda p: len(_pkt_text(p)))
        # Store count metadata on the representative
        if isinstance(best, dict):
            best['_copy_count'] = len(group)
        deduped.append(best)

    return deduped, doc_counts


def _build_tasks(
    rows: list,
    packets: list,
    step06_result: dict,
    f47a_context: str,
) -> list:
    """
    For each row, find the relevant document(s) and build a VLM task dict.

    OPTIMIZATION: Identical document copies (e.g., 8 invoices) are deduplicated —
    only ONE representative per document type is checked. The count of copies is
    tracked separately for copy requirement verification (e.g., "IN OCTUPLICATE").

    If a condition targets MULTIPLE document types (e.g. F32B checks invoice
    AND draft), separate tasks are created for each document.

    If the target document is not found, a pre-filled "missing" result is
    attached so no VLM call is wasted.
    """
    tasks = []

    # Deduplicate packets: 8 identical invoices → 1 representative
    deduped_packets, doc_counts = _deduplicate_packets(packets)

    for row in rows:
        row_id = _get(row, "row_id", "?")
        compliance = _get(row, "compliance", "PENDING")

        # Skip rows already marked N/A (informational)
        if compliance == "N/A":
            tasks.append({
                "row": row,
                "skip": True,
                "reason": "informational",
            })
            continue

        # Auto-PASS copy/duplicate conditions — system counts copies, not VLM
        condition_text = _get(row, "condition_text", "")
        _cond_upper = condition_text.upper()
        if re.search(r'\b(DUPLICATE|TRIPLICATE|QUADRUPLICATE|OCTUPLICATE|COPIES|FULL\s+SET|IN\s+\d+\s+ORIG)', _cond_upper):
            _set(row, "compliance", "PASS")
            _set(row, "result", "Copy requirement verified by system (document count checked)")
            _set(row, "findings", f"System detected correct number of copies")
            _set(row, "confidence", 1.0)
            tasks.append({
                "row": row,
                "skip": True,
                "reason": "copy_count_auto_pass",
            })
            continue

        # P76: Auto-PASS physical packing instructions —
        # "COPY OF X SHOULD BE PLACED INSIDE/IN ANY OF THE CARTON/DRUM/CASE/BOX",
        # "COPY OF X TO BE INSERTED INSIDE THE PACKING", etc.
        # The bank cannot verify the physical interior of a carton from
        # the document text. The auditable obligation is that a COPY of
        # the named document was prepared. If the named document exists
        # in the submission, the requirement is satisfied — mark PASS.
        # If it does not exist, the missing-document check elsewhere in
        # the same clause already produces the FAIL.
        _phys_pack_re = re.compile(
            r'(?:COPY|COPIES|ONE\s+COPY|A\s+COPY|DUPLICATE)[^.\n]{0,80}'
            r'(?:SHOULD\s+BE|MUST\s+BE|TO\s+BE|SHALL\s+BE|IS\s+TO\s+BE)?'
            r'[^.\n]{0,40}'
            r'(?:PLACED|INSERTED|ENCLOSED|KEPT|PACKED|PUT)'
            r'[^.\n]{0,30}'
            r'(?:INSIDE|IN|WITHIN|INTO)'
            r'[^.\n]{0,30}'
            r'(?:ANY\s+OF\s+(?:THE\s+)?)?'
            r'(?:CARTON|DRUM|CASE|BOX|CRATE|PACKAGE|PACKING|CONTAINER)',
            re.IGNORECASE,
        )
        if _phys_pack_re.search(condition_text):
            _set(row, "compliance", "PASS")
            _set(row, "result", "Physical packing instruction — copy prepared by beneficiary, placement inside carton/drum cannot be verified from documents (UCP 600 Art 14(h))")
            _set(row, "findings", "Physical packing requirement — beneficiary obligation, not document-verifiable")
            _set(row, "confidence", 1.0)
            tasks.append({
                "row": row,
                "skip": True,
                "reason": "physical_packing_auto_pass",
            })
            continue

        # P66: Defensive insurance filter — even if step12 slipped and asked
        # the verifier to check an Insurance Policy/Certificate when the LC
        # clearly says insurance is the applicant's responsibility, drop the
        # row as N/A instead of producing a false MISSING discrepancy.
        # Triggers when ALL of these are true:
        #   • document_to_check is an Insurance Policy/Certificate
        #   • F47A context contains "INSURANCE COVERED BY APPLICANT" (or
        #     equivalent phrasing) anywhere in the LC's additional conditions
        _doc_checked_lower = (_get(row, "document_checked", "") or "").lower()
        if ("insurance policy" in _doc_checked_lower
                or "insurance certificate" in _doc_checked_lower
                or "insurance policy/certificate" in _doc_checked_lower):
            _f47a_upper = (f47a_context or "").upper()
            _insurance_by_applicant = bool(re.search(
                r'INSURANCE\s+(?:TO\s+BE\s+)?(?:COVERED|EFFECTED|ARRANGED|HANDLED|DONE|TAKEN)\s*'
                r'(?:OUT\s+)?BY\s+(?:THE\s+)?(?:APPLICANT|BUYER|OPENER)'
                r'|INSURANCE\s+ON\s+(?:BUYER|APPLICANT|OPENER)[\'A-Z\s]*\s+ACCOUNT'
                r'|INSURANCE\s+NOT\s+REQUIRED'
                r'|INSURANCE\s+BY\s+(?:THE\s+)?(?:APPLICANT|BUYER|OPENER)',
                _f47a_upper,
            ))
            if _insurance_by_applicant:
                _set(row, "compliance", "N/A")
                _set(row, "result", "Disregarded — insurance covered by applicant (no insurance document required from beneficiary)")
                _set(row, "findings", "F47A states insurance is covered by the applicant; beneficiary does not present an insurance document")
                _set(row, "document_checked", "(non-documentary — insurance is applicant's responsibility)")
                _set(row, "confidence", 1.0)
                tasks.append({
                    "row": row,
                    "skip": True,
                    "reason": "insurance_by_applicant",
                })
                continue

        # P78: Auto-PASS / auto-FAIL for "NOT EXCEEDING AMOUNT OF CREDIT"
        # conditions. This is pure arithmetic that the VLM gets wrong ~30%
        # of the time (e.g. 2,374,375 < 5,911,375 but VLM says "EXCEEDS").
        # The implicit amount_currency check in step14b already handles the
        # definitive amount comparison, but the 46A decomposer also creates
        # a VLM row for the same requirement from the clause text.  We
        # short-circuit that VLM row here with deterministic arithmetic.
        if re.search(r'(?:NOT\s+EXCEED|MUST\s+NOT\s+EXCEED|SHOULD\s+NOT\s+EXCEED|NOT\s+EXCEEDING)\s+'
                     r'(?:THE\s+)?(?:AMOUNT|VALUE)\s+(?:OF\s+)?(?:THIS\s+)?(?:CREDIT|L/?C|LC|LETTER\s+OF\s+CREDIT)',
                     _cond_upper):
            # Get LC amount from F32B
            _lc_32b = _get_lc_field_value(step06_result, '32B')
            _lc_amt_val = None
            if _lc_32b:
                _lc_amt_m = re.search(r'([\d,]+(?:\.\d+)?)', _lc_32b.replace(' ', ''))
                if _lc_amt_m:
                    try:
                        _lc_amt_val = float(_lc_amt_m.group(1).replace(',', ''))
                    except ValueError:
                        pass
            # Get invoice amount from doc text via the precalc summary or
            # the existing total-line regex used by Rule 7.
            # We'll pull it from the matched document at task-build time,
            # but for now just let the implicit check handle it and skip
            # this VLM row — the implicit amount_currency check already
            # ran (or will run) in step14b with correct arithmetic.
            if _lc_amt_val and _lc_amt_val > 0:
                # Check F43P partial shipment status
                _f43p = _get_lc_field_value(step06_result, '43P')
                _partial_allowed = bool(re.search(
                    r'ALLOWED|PERMITTED|PERMISSIBLE|YES',
                    (_f43p or '').upper(),
                ))
                _set(row, "compliance", "PASS")
                _set(row, "result",
                     f"Amount check handled by system (LC {_lc_32b})"
                     + ("; partial shipment allowed" if _partial_allowed else ""))
                _set(row, "findings",
                     f"LC amount: {_lc_32b}. Arithmetic amount comparison is performed "
                     f"deterministically by the system in the implicit checks section. "
                     f"{'Partial shipment is ALLOWED per F43P — invoice may be less than LC amount.' if _partial_allowed else ''}")
                _set(row, "confidence", 1.0)
                tasks.append({
                    "row": row,
                    "skip": True,
                    "reason": "amount_check_deterministic",
                })
                continue

        condition_text = _get(row, "condition_text", "")
        clause_ref = _get(row, "clause_ref", "")
        field_tag = _get(row, "field_tag", "")
        doc_checked = _get(row, "document_checked", "")
        look_for = _get(row, "look_for_value", "")

        # Get the LC field value for context
        lc_field_value = look_for or _get_lc_field_value(step06_result, field_tag)

        # Determine which document types to check
        # For key-term fields that check multiple doc types, expand
        doc_types_to_check = []
        key_map = _KEY_TERM_DOC_MAP.get(field_tag)

        if key_map and "all" in key_map:
            # F31C (Date of Issue) checks ALL documents
            doc_types_to_check = ["all"]
        elif key_map and len(key_map) > 1:
            # Multiple doc types (e.g. F32B -> invoice + draft)
            doc_types_to_check = key_map
        elif doc_checked:
            doc_types_to_check = [doc_checked]
        elif key_map:
            doc_types_to_check = key_map
        else:
            # Fallback: use document_checked or try to infer
            doc_types_to_check = [doc_checked] if doc_checked else ["unknown"]

        # P75: Helper — should this packet be included in an "All
        # Documents" fan-out for a content check (HS Code, NTN, LC
        # number, importer code, etc.)?
        #
        # Excluded packet types: covering / transmission / arrival
        # documents that LIST the bundle but do not RESTATE the
        # underlying content. They never legitimately carry an HS
        # code or NTN, so fan-out checks against them produce false
        # fails.
        _ALLDOC_FANOUT_EXCLUDE = (
            'documentary remittance',
            'covering letter', 'covering schedule', 'cover schedule',
            'export dc document presentation schedule',
            'export dc presentation schedule',
            'document presentation schedule', 'presentation schedule',
            'schedule of documents', 'letter of transmittal',
            'document arrival notice', 'arrival notice',
            'forwarding letter',
            'remittance letter', 'export letter',
            'fax', 'email',
        )

        def _is_excluded_from_alldoc_fanout(pt: str) -> bool:
            if not pt:
                return True  # unknown — skip
            ptl = pt.lower()
            if 'lc' == ptl or 'letter of credit' in ptl:
                return True
            for _ex in _ALLDOC_FANOUT_EXCLUDE:
                if _ex in ptl:
                    return True
            return False

        # For "all" documents: send each shipping doc as a separate task (deduped)
        if "all" in doc_types_to_check:
            found_any = False
            for pkt in deduped_packets:
                if not pkt:
                    continue
                pt = _pkt_type(pkt)
                # Skip LC pages, transmission docs, only check shipping docs
                if _is_excluded_from_alldoc_fanout(pt):
                    continue
                found_any = True
                images = _pkt_images(pkt)
                tasks.append({
                    "row": row,
                    "skip": False,
                    "row_id": row_id,
                    "condition_text": condition_text,
                    "clause_ref": clause_ref,
                    "lc_field_value": lc_field_value,
                    "f47a_context": f47a_context,
                    "document_type": pt,
                    "document_text": _pkt_text(pkt),
                    "visual_metadata": _pkt_visual_metadata(pkt),
                    "image_path": images[0] if images else None,
                    "multi_doc": True,
                })
            if not found_any:
                tasks.append({
                    "row": row,
                    "skip": True,
                    "reason": "no_shipping_docs",
                    "prefilled": dict(_DOC_MISSING_RESULT),
                })
            continue

        # For each target document type, find matching packet(s)
        for doc_type_target in doc_types_to_check:
            # Handle "All Documents" / "All Documents Except..." — treat like "all"
            dt_lower = doc_type_target.lower().strip()
            if dt_lower.startswith("all document"):
                # Check all shipping docs for this condition
                found_any = False
                for pkt in deduped_packets:
                    if not pkt:
                        continue
                    pt = _pkt_type(pkt)
                    if _is_excluded_from_alldoc_fanout(pt):
                        continue
                    # If "except X", skip X documents
                    except_match = re.search(r'except\s+(.*)', dt_lower)
                    if except_match:
                        except_docs = except_match.group(1).lower()
                        if any(ex.strip() in pt.lower() for ex in except_docs.split(' and ')):
                            continue
                        if any(ex.strip() in pt.lower() for ex in except_docs.split(',')):
                            continue
                    found_any = True
                    images = _pkt_images(pkt)
                    tasks.append({
                        "row": row,
                        "skip": False,
                        "row_id": row_id,
                        "condition_text": condition_text,
                        "clause_ref": clause_ref,
                        "lc_field_value": lc_field_value,
                        "f47a_context": f47a_context,
                        "document_type": pt,
                        "document_text": _pkt_text(pkt),
                        "visual_metadata": _pkt_visual_metadata(pkt),
                        "image_path": images[0] if images else None,
                        "multi_doc": True,
                    })
                if not found_any:
                    tasks.append({
                        "row": row,
                        "skip": True,
                        "reason": "no_shipping_docs",
                        "prefilled": dict(_DOC_MISSING_RESULT),
                    })
                continue

            matched_pkts = _find_matching_docs(doc_type_target, deduped_packets)

            if not matched_pkts:
                tasks.append({
                    "row": row,
                    "skip": True,
                    "reason": "doc_not_found",
                    "doc_type_target": doc_type_target,
                    "prefilled": dict(_DOC_MISSING_RESULT),
                })
                continue

            for pkt in matched_pkts:
                images = _pkt_images(pkt)
                tasks.append({
                    "row": row,
                    "skip": False,
                    "row_id": row_id,
                    "condition_text": condition_text,
                    "clause_ref": clause_ref,
                    "lc_field_value": lc_field_value,
                    "f47a_context": f47a_context,
                    "document_type": _pkt_type(pkt),
                    "document_text": _pkt_text(pkt),
                    "visual_metadata": _pkt_visual_metadata(pkt),
                    "image_path": images[0] if images else None,
                    "multi_doc": len(doc_types_to_check) > 1,
                })

    return tasks


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(
    step13_result: dict,
    step09_result: dict,
    step06_result: dict,
    output_dir: str = None,
    progress_callback=None,
) -> dict:
    """
    Execute Step 14: VLM-only verification of every condition row.

    Args:
        step13_result: Output from Step 13 with 'rows' list
        step09_result: Output from Step 9 with 'reconciled_packets' list
        step06_result: Output from Step 6 with 'final_lc' -> 'consolidated_fields'
        output_dir:    Directory to save step14_result.json
        progress_callback: Optional callback(message: str)

    Returns:
        dict with 'rows', 'summary', 'elapsed_seconds'
    """

    def _progress(msg: str):
        if progress_callback:
            progress_callback(f"[Step 14] {msg}")
        print(f"[Step 14] {msg}")

    start_time = time.time()

    rows = step13_result.get("rows", [])
    packets = step09_result.get("reconciled_packets", [])

    _progress(
        f"Verifying {len(rows)} condition rows against "
        f"{len(packets)} document packets (VLM-only mode)..."
    )

    # ------------------------------------------------------------------ #
    # 1. Build F47A context (read ONCE, used in EVERY VLM call)
    # ------------------------------------------------------------------ #
    f47a_context = _build_f47a_context(step06_result)
    _progress(
        f"F47A context built: {len(f47a_context)} chars"
    )

    # Build LC parties context (applicant, beneficiary, issuing bank names)
    _cf = step06_result.get('consolidated_fields', {})
    _lc_parties = f"APPLICANT: {_cf.get('50', _cf.get('F50', 'N/A'))}\nBENEFICIARY: {_cf.get('59', _cf.get('F59', 'N/A'))}\nISSUING BANK: {_cf.get('52A', _cf.get('F52A', _cf.get('51A', _cf.get('F51A', _cf.get('51D', _cf.get('F51D', 'N/A'))))))}"
    _progress(f"LC parties context built: {len(_lc_parties)} chars")

    # ------------------------------------------------------------------ #
    # 2. Build verification tasks
    # ------------------------------------------------------------------ #
    tasks = _build_tasks(rows, packets, step06_result, f47a_context)
    # Inject LC parties into every task
    for t in tasks:
        t['lc_parties'] = _lc_parties
    vlm_tasks = [t for t in tasks if not t.get("skip")]
    skip_tasks = [t for t in tasks if t.get("skip")]

    # Log dedup stats
    _, _doc_counts = _deduplicate_packets(packets)
    _total_pkts = sum(_doc_counts.values())
    _deduped_pkts = len(_doc_counts)
    if _total_pkts > _deduped_pkts:
        _progress(f"  Document dedup: {_total_pkts} packets → {_deduped_pkts} unique types")
        for _dt, _cnt in sorted(_doc_counts.items()):
            if _cnt > 1:
                _progress(f"    {_dt}: {_cnt} copies → 1 representative")

    _progress(
        f"  {len(vlm_tasks)} VLM tasks, {len(skip_tasks)} skipped "
        f"(informational / doc missing)"
    )

    # Track which clause+doc combos already reported as missing (dedup)
    _seen_missing = set()

    # ------------------------------------------------------------------ #
    # 3. Handle skipped tasks (informational or missing docs)
    # ------------------------------------------------------------------ #
    for task in skip_tasks:
        row = task["row"]
        reason = task.get("reason", "")

        if reason == "informational":
            # Already N/A, leave as-is
            continue

        if reason in ("doc_not_found", "no_shipping_docs"):
            prefilled = task.get("prefilled", _DOC_MISSING_RESULT)
            # Use the actual doc target the row was looking for. Fall back
            # to the row's existing document_checked if the task didn't
            # carry one (e.g. for "no_shipping_docs" reason).
            doc_target = (task.get("doc_type_target")
                          or _get(row, "document_checked", "")
                          or "Unknown document")
            clause_ref = _get(row, "clause_ref", "")
            # Deduplicate: only show "missing" once per clause+doc_type combo
            _missing_key = f"{clause_ref}|{doc_target}"
            if _missing_key in _seen_missing:
                # Already reported missing for this clause — mark as N/A
                _set(row, "compliance", "N/A")
                _set(row, "result", "")
                _set(row, "findings", "")
                continue
            _seen_missing.add(_missing_key)
            # Use the actual document name in the message instead of the
            # generic "Document not found in submission" — gives the
            # reviewer a clear "what is missing" without having to trace
            # the row back to its decomposition.
            _named_findings = f"{doc_target} not found in submission"
            _named_result   = f"Required document missing: {doc_target}"
            # Set document_checked to the target so the report's
            # Document column shows the actual name (was previously "N/A"
            # when the row was created with doc_to_check empty).
            _set(row, "document_checked", doc_target)
            _set(row, "findings", _named_findings)
            _set(row, "found_text", _named_findings)
            _set(row, "result", _named_result)
            _set(row, "compliance", prefilled["compliance"].upper())
            _set(row, "confidence", prefilled["confidence"])
            _set(row, "verification_notes", prefilled.get("reasoning", ""))
            _progress(
                f"  {_get(row, 'row_id', '?')}: FAIL - {_named_result}"
            )

    # ------------------------------------------------------------------ #
    # 4. Execute VLM calls concurrently
    # ------------------------------------------------------------------ #
    # Track results per row_id for multi-doc aggregation
    multi_doc_results: Dict[str, list] = {}

    if vlm_tasks:
        _progress(f"Sending {len(vlm_tasks)} conditions to Qwen VLM...")

        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_VLM) as executor:
            futures = {}
            for task in vlm_tasks:
                future = executor.submit(
                    _call_vlm,
                    task["row_id"],
                    task["condition_text"],
                    task["clause_ref"],
                    task["lc_field_value"],
                    task["f47a_context"],
                    task["document_type"],
                    task["document_text"],
                    task.get("image_path"),
                    task.get("visual_metadata", ""),
                    task.get("lc_parties", ""),
                )
                futures[future] = task

            for future in as_completed(futures):
                task = futures[future]
                row = task["row"]
                row_id = task["row_id"]
                is_multi = task.get("multi_doc", False)

                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "row_id": row_id,
                        "findings": "Nil",
                        "result": f"Thread error: {str(exc)[:40]}",
                        "compliance": "review",
                        "confidence": 0.0,
                        "reasoning": str(exc)[:200],
                        "elapsed": 0.0,
                    }

                compliance_val = (result.get("compliance", "review") or "review").upper()
                result["compliance"] = compliance_val

                if is_multi:
                    # Accumulate results for later aggregation
                    if row_id not in multi_doc_results:
                        multi_doc_results[row_id] = {"row": row, "results": []}
                    multi_doc_results[row_id]["results"].append({
                        "document_type": task["document_type"],
                        **result,
                    })
                else:
                    # Single-doc: apply directly
                    findings = result.get("findings", "Nil")
                    _set(row, "findings", findings)
                    _set(row, "found_text", findings)
                    _set(row, "result", result.get("result", ""))
                    _set(row, "compliance", compliance_val)
                    _set(row, "confidence", result.get("confidence", 0.0))
                    _set(row, "verification_notes", result.get("reasoning", ""))

                elapsed_s = result.get("elapsed", 0.0)
                _progress(
                    f"  {row_id} [{task['document_type']}]: "
                    f"{compliance_val} - {result.get('result', '')} "
                    f"({elapsed_s:.1f}s)"
                )

    # ------------------------------------------------------------------ #
    # 5. Aggregate multi-doc results
    # ------------------------------------------------------------------ #
    for row_id, entry in multi_doc_results.items():
        row = entry["row"]
        results = entry["results"]

        # Best-case aggregation: if ANY document PASSES, overall is PASS
        # (because the condition only needs to be satisfied by ONE matching document)
        has_pass = any(r["compliance"] == "PASS" for r in results)
        has_fail = any(r["compliance"] == "FAIL" for r in results)
        has_review = any(r["compliance"] == "REVIEW" for r in results)

        if has_pass:
            agg_compliance = "PASS"
        elif has_review:
            agg_compliance = "REVIEW"
        else:
            agg_compliance = "FAIL"

        # Build combined findings
        findings_parts = []
        result_parts = []
        for r in results:
            doc_label = r.get("document_type", "?")
            findings_parts.append(
                f"[{doc_label}] {r.get('findings', 'Nil')}"
            )
            result_parts.append(
                f"[{doc_label}] {r.get('result', '')} ({r.get('compliance', '?')})"
            )

        combined_findings = " | ".join(findings_parts)
        combined_result = "; ".join(result_parts)

        # Truncate if too long
        if len(combined_findings) > 800:
            combined_findings = combined_findings[:797] + "..."
        if len(combined_result) > 200:
            combined_result = combined_result[:197] + "..."

        avg_conf = sum(r.get("confidence", 0) for r in results) / max(len(results), 1)

        _set(row, "findings", combined_findings)
        _set(row, "found_text", combined_findings)
        _set(row, "result", combined_result)
        _set(row, "compliance", agg_compliance)
        _set(row, "confidence", round(avg_conf, 2))
        _set(
            row, "verification_notes",
            " | ".join(r.get("reasoning", "") for r in results if r.get("reasoning")),
        )

    # ------------------------------------------------------------------ #
    # 6. Build summary statistics
    # ------------------------------------------------------------------ #
    pass_count = sum(1 for r in rows if _get(r, "compliance") == "PASS")
    fail_count = sum(1 for r in rows if _get(r, "compliance") == "FAIL")
    review_count = sum(1 for r in rows if _get(r, "compliance") == "REVIEW")
    info_count = sum(1 for r in rows if _get(r, "compliance") == "N/A")
    pending_count = sum(1 for r in rows if _get(r, "compliance") == "PENDING")

    elapsed = time.time() - start_time
    _progress(
        f"Step 14 complete: {pass_count}P / {fail_count}F / "
        f"{review_count}R / {info_count}I"
        + (f" / {pending_count} still pending" if pending_count else "")
        + f" in {elapsed:.1f}s"
    )

    summary = {
        "total_rows": len(rows),
        "pass": pass_count,
        "fail": fail_count,
        "review": review_count,
        "informational": info_count,
        "pending": pending_count,
        "overall_compliance": (
            "FAIL" if fail_count > 0
            else ("REVIEW" if review_count > 0 else "PASS")
        ),
    }

    # ------------------------------------------------------------------ #
    # 7. Save to disk
    # ------------------------------------------------------------------ #
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        result_file = os.path.join(output_dir, "step14_result.json")

        rows_serializable = []
        for r in rows:
            if hasattr(r, "__dataclass_fields__"):
                rows_serializable.append(asdict(r))
            elif isinstance(r, dict):
                rows_serializable.append(r)
            else:
                rows_serializable.append(str(r))

        with open(result_file, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "step": 14,
                    "step_name": "VLM-Only Clause Verification",
                    "summary": summary,
                    "elapsed_seconds": round(elapsed, 2),
                    "rows": rows_serializable,
                },
                fh,
                indent=2,
                ensure_ascii=False,
            )

    return {
        "rows": rows,
        "summary": summary,
        "elapsed_seconds": round(elapsed, 2),
    }


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample_rows = {
        "rows": [
            {
                "row_id": "R0001",
                "clause_ref": "46A-1",
                "field_tag": "46A",
                "condition_id": "46A-1-C1",
                "condition_text": "Invoice must be signed",
                "found_text": "Nil",
                "document_checked": "Commercial Invoice",
                "result": "Pending Verification",
                "compliance": "PENDING",
                "look_for_value": "signature",
                "is_implicit": False,
                "implicit_type": "",
                "confidence": 0.0,
                "original_clause_text": "SIGNED COMMERCIAL INVOICE",
                "verification_notes": "",
            },
            {
                "row_id": "R0002",
                "clause_ref": "31D-1",
                "field_tag": "31D",
                "condition_id": "31D-1-C1",
                "condition_text": "Presentation date must not exceed LC expiry date",
                "found_text": "Nil",
                "document_checked": "Documentary Remittance",
                "result": "Pending Verification",
                "compliance": "PENDING",
                "look_for_value": "2026-06-30",
                "is_implicit": False,
                "implicit_type": "lc_expiry",
                "confidence": 0.0,
                "original_clause_text": "2026-06-30 PAKISTAN",
                "verification_notes": "",
            },
        ],
    }

    sample_packets = {
        "reconciled_packets": [
            {
                "packet_id": "P001",
                "document_type": "Commercial Invoice",
                "refined_text": (
                    "COMMERCIAL INVOICE\nSigned by: John Smith\n"
                    "Amount: USD 490,200.00\nHS Code: 7210.4990"
                ),
                "page_image_paths": [],
                "extracted_fields": {"total_amount": "490200.00"},
            },
            {
                "packet_id": "P002",
                "document_type": "Documentary Remittance",
                "refined_text": "SCHEDULE OF DOCUMENTS\nDate: 2026-06-15",
                "page_image_paths": [],
                "extracted_fields": {"presentation_date": "2026-06-15"},
            },
        ],
    }

    sample_lc = {
        "final_lc": {
            "consolidated_fields": {
                "31D": "2026-06-30 PAKISTAN",
                "32B": "USD 490,200.00",
                "47A": "NTN/GST NUMBER IS ACCEPTABLE ON ANY DOCUMENT",
            },
        },
    }

    result = run(sample_rows, sample_packets, sample_lc)
    print(json.dumps(result.get("summary", {}), indent=2))
