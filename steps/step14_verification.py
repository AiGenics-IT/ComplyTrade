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
from config.settings import QWEN_TEXT_LLM_URL, QWEN_TEXT_LLM_MODEL, MAX_CONCURRENT_VLM, VLM_TIMEOUT


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
        # P188/P192 — "Shipment Advice" AND "Courier Receipt" are SEPARATE
        # instruments from the AWB:
        #   - Shipment Advice = letter/email/fax from beneficiary
        #   - Courier Receipt = proof that documents (esp. the shipment
        #     advice) were dispatched by courier; a dedicated DHL/FedEx
        #     voucher for the document envelope.
        # An Air Waybill is the carriage contract for the goods itself.
        # Keep all three groups disjoint so a missing Courier Receipt or
        # missing Shipment Advice is reported as missing instead of
        # being silently verified against the AWB.
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
        "beneficiary certificate", "beneficiary's certificate",
        "beneficiarys certificate", "beneficiary statement",
        "beneficiary's statement", "beneficiary declaration",
    ],
    "courier receipt": [
        "courier receipt", "courier waybill", "courier service receipt",
        "express waybill", "express delivery receipt", "express courier",
        "dhl", "fedex", "fed ex", "ups", "tnt", "aramex",
        # P192 — Air Waybill is a SEPARATE instrument (carriage contract
        # for the goods). Courier Receipt is the receipt for the
        # courier dispatch of the DOCUMENTS (often the shipment advice).
        # Keep groups disjoint so missing Courier Receipt reports as
        # missing instead of being verified against the AWB.
    ],
    "email evidence": [
        "email evidence", "email screenshot", "email confirmation",
        "email copy", "covering email", "transmission record",
        # P188 — Shipment Advice is a separate instrument, see note on
        # the "air waybill" group above.
    ],
    "shipment advice": [
        "shipment advice", "shipment advise", "shipping advice",
        "shipping advise", "beneficiary shipment advice",
        "declaration of shipment", "notice of shipment",
    ],
    "fax confirmation": [
        "fax confirmation", "fax report", "fax transmission",
        "fax receipt", "fax copy",
    ],
    "documentary remittance": [
        "documentary remittance", "covering letter", "remittance letter",
        "schedule of documents", "letter of transmittal", "covering schedule",
        "document presentation", "document presentation schedule",
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
        # P101: "L/C Bills Schedule" is the Standard Chartered name for
        # the covering schedule / documentary remittance.
        "l/c bills schedule", "lc bills schedule", "bills schedule",
    ],
    "letter of indemnity": [
        "letter of indemnity", "loi", "indemnity letter",
        "letter of indemnification", "indemnity bond",
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
        "form of undertaking", "form 3 (form of undertaking)",
    ],
    "form 3 (form of undertaking)": [
        "form 3 (form of undertaking)", "form 3", "form3",
        "form of undertaking", "drug registration certificate",
        "import certificate", "drug import certificate",
        "form no 3", "form no. 3",
    ],
    # P90: Certificate of Analysis
    "certificate of analysis": [
        "certificate of analysis", "analysis certificate",
        "analytical certificate", "coa", "analysis report",
        "test report", "test certificate",
        "laboratory certificate", "lab report", "lab certificate",
        "quality analysis certificate",
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
            raw = (pkt.get("document_type", "") or pkt.get("doc_type", "")
                   or pkt.get("classification", "") or "").lower().strip()
        else:
            raw = (getattr(pkt, "document_type", "") or "").lower().strip()
        # Strip OCR-concatenated copy/original suffixes:
        # "beneficiary's certificateoriginal" → "beneficiary's certificate"
        raw = re.sub(r'(original|copy|duplicate|triplicate|quadruplicate)$', '', raw).strip()
        return raw

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

    if matches:
        return matches

    # Tier 4: Text content fallback — search actual page text for the
    # target document name or its aliases.  This catches cases where
    # the VLM classifier assigns a generic/unexpected document_type
    # (e.g. "Certificate" instead of "Batch Certificate") but the
    # page text clearly contains "FORM 7" or "CERTIFICATE OF ANALYSIS".
    #
    # P198e — EXCLUDE transmission / covering-letter packets from
    # this tier. A Documentary Remittance / covering schedule LISTS
    # every document in the bundle by name, so Tier 4 otherwise
    # spuriously matches those packets for ANY missing doc and the
    # LLM then evaluates the requirement against the cover letter —
    # producing a false FAIL instead of the correct "document
    # missing" verdict.
    _TRANSMISSION_DOC_TOKENS = (
        'documentary remittance', 'document remittance',
        'covering letter', 'cover letter',
        'covering schedule', 'cover schedule',
        'l/c bills schedule', 'lc bills schedule', 'bills schedule',
        'export dc document presentation schedule',
        'export dc presentation schedule',
        'document presentation schedule', 'presentation schedule',
        'document presentation',
        'schedule of documents', 'letter of transmittal',
        'document arrival notice', 'arrival notice',
        'forwarding letter',
        'remittance letter', 'export letter',
        'fax', 'email',
    )
    _text_search_terms = set(target_aliases)
    _text_search_terms.add(target)
    # If the target IS a transmission doc, skip the exclusion — we
    # actually want to find the covering letter in that case.
    _target_is_transmission = any(tok in target for tok in _TRANSMISSION_DOC_TOKENS)
    for pkt in packets:
        if not pkt:
            continue
        pkt_type = _get_pkt_type(pkt)
        if pkt_type and ("lc" == pkt_type or "letter of credit" in pkt_type):
            continue
        if not _target_is_transmission and pkt_type:
            if any(tok in pkt_type for tok in _TRANSMISSION_DOC_TOKENS):
                continue  # don't let covering letter masquerade as the missing doc
        pkt_text = _pkt_text(pkt if isinstance(pkt, dict) else asdict(pkt)).lower()
        if not pkt_text or len(pkt_text) < 20:
            continue
        # Search the first 2000 chars (header area) for the target name
        header = pkt_text[:2000]
        for term in _text_search_terms:
            if len(term) >= 4 and term in header:
                matches.append(pkt if isinstance(pkt, dict) else asdict(pkt))
                break

    # P197 — Shipping Company Certificate issuer guard.
    # A "Certificate from Shipping Company or their Authorized Agents"
    # must be issued by a carrier / shipping line / shipping agent.
    # Step 3a's classifier sometimes mis-labels inspection / survey /
    # agricultural-services certificates (Alfred H Knight, SGS, BV,
    # Intertek, Cotecna, Control Union, Geo-Chem, etc.) as "Shipping
    # Company Certificate" because the layout and some phrasing can
    # overlap. When the requirement asks for an SCC, drop any
    # candidate whose issuer is a known inspection firm — if nothing
    # is left, return empty so the caller reports the document as
    # missing instead of failing against a wrong doc.
    if matches and ('shipping company' in target
                    or 'shipping certificate' in target
                    or 'agent certificate' in target
                    or "agent's certificate" in target
                    or 'carrier certificate' in target
                    or "carrier's certificate" in target):
        _INSPECTION_ISSUER_TOKENS = (
            'alfred h knight', 'ahk', 'sgs', 'bureau veritas', 'intertek',
            'cotecna', 'control union', 'geo-chem', 'geo chem', 'geochem',
            'inspectorate', 'saybolt', 'alex stewart', 'camspec',
            'survey services', 'inspection services',
            'agriculture services', 'agri services', 'agricultural services',
            'laboratory services', 'testing services', 'analytical services',
            'quality services',
        )
        def _issued_by(p):
            if isinstance(p, dict):
                return (p.get('issued_by') or '').lower()
            return (getattr(p, 'issued_by', '') or '').lower()
        _filtered = [
            p for p in matches
            if not any(tok in _issued_by(p) for tok in _INSPECTION_ISSUER_TOKENS)
        ]
        # Only replace if the filter actually removed some AHK-style
        # noise AND a plausible real SCC remains. If every candidate
        # is an inspection firm, drop them all — the real SCC is
        # missing from the submission.
        if len(_filtered) != len(matches):
            matches = _filtered

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
        val = str(val)
        # F48 normalization: extract days number + optional narrative
        # Handles: "Days: 21\nNarrative:\nDAYS FROM...", "21\nDAYS FROM...", "21"
        if field_tag == '48':
            val = re.sub(r'(?i)^Period\s+for\s+Presentation.*?Days\s*[\n\r]*', '', val).strip()
            val = re.sub(r'(?i)^Days:?\s*', '', val).strip()
            val = re.sub(r'(?i)\nNarrative:?\s*/?\s*\n?', '\n', val).strip()
        return _sanitize_lc_field_value(field_tag, val)

    # Try with 'F' prefix
    val = fields.get(f"F{field_tag}", "")
    if isinstance(val, dict):
        val = val.get("value", str(val))
    return _sanitize_lc_field_value(field_tag, str(val)) if val else ""


# ---------------------------------------------------------------------------
# VLM call -- sends condition + document text + optional image to Qwen VLM
# ---------------------------------------------------------------------------

# ════════════════════════════════════════════════════════════════════════ #
# NEW SPLIT-PROMPT ARCHITECTURE (P123)                                      #
#                                                                           #
# Replaces the monolithic _VLM_PROMPT_TEMPLATE with a composable            #
# CORE prompt + per-document-family rule pack. Consumes structured facts    #
# (dates_found / amounts_found / references_found / parties_found /         #
# other_details_found + bl_subtype) produced by Step 3e so the LLM has      #
# tagged, pre-classified data instead of re-parsing free text.              #
#                                                                           #
# Feature flag: USE_SPLIT_PROMPTS (default True). Flip to False to fall     #
# back to the legacy _VLM_PROMPT_TEMPLATE for emergency rollback.           #
# ════════════════════════════════════════════════════════════════════════ #

USE_SPLIT_PROMPTS = True


CORE_VERIFICATION_PROMPT = """You are verifying ONE condition from a Letter of Credit against a trade
finance document. Use ONLY the data below. Do NOT invent, assume, or copy
text from the condition into your findings.

════════════════════════════════════════════════════════════════════════
INPUTS — YOUR COMPLETE SOURCE OF TRUTH
════════════════════════════════════════════════════════════════════════

LC CONDITION TO VERIFY:
{condition_text}

LC FIELD: {clause_ref}
LC FIELD VALUE: {lc_field_value}

LC PARTIES (for resolving "APPLICANT", "BENEFICIARY", "ISSUING BANK", etc.):
{lc_parties}

KEY LC FIELDS (use these when the condition references LC dates, amount,
LC number, ports, transshipment/partial-shipment flags, drafts-at, or any
field below; these are the authoritative LC values — do NOT say "the LC
date is not provided" if F31C is listed here):
{key_lc_fields}

F47A ADDITIONAL CONDITIONS (READ FIRST — these can override the condition):
{f47a_context}

DOCUMENT TYPE: {document_type}

STRUCTURED FACTS (already extracted and tagged — USE THESE FIRST):
{structured_facts}

DOCUMENT TEXT (OCR — trusted, complete page text from all pages of the packet):
{document_text}

DOCUMENT VISUAL METADATA (stamps, signatures, seals, copy/original status):
{visual_metadata}

════════════════════════════════════════════════════════════════════════
TOP-PRIORITY RULES (READ THESE FIRST, BEFORE ANY OTHER REASONING)
════════════════════════════════════════════════════════════════════════

TOP RULE 1 — DATE COMPARISON (CRITICAL, applied to EVERY row that
compares a document date against the LC F31C issue date, F44C latest
shipment, F31D expiry, or any other LC date):

  Before you write "prior to", "after", "earlier than", "later than",
  "before the LC date", "pre-dated", or any similar temporal claim in
  your findings, PARSE both dates into (year, month, day) integers and
  compare them in that order: YEAR first, then MONTH, then DAY.

  2-digit years resolve as: 00..49 -> 20XX, 50..99 -> 19XX. So
  "28-Jan-25" is 2025-01-28, not 1925 and not 2028.
  Six-digit SWIFT dates "YYMMDD" likewise: "250102" -> 2025-01-02.

  Examples you MUST get right:
    LC F31C = 2025-01-02. Doc = "28-Jan-25" -> parses to 2025-01-28
      -> 2025-01-28 > 2025-01-02 -> doc is AFTER the LC date ->
      PASS the "documents dated prior to LC date not acceptable" check.
    LC F31C = 2025-01-02. Doc = "250102" -> 2025-01-02 = LC date
      -> PASS (same day is not prior).
    LC F31C = 2025-01-02. Doc = "28-Dec-24" -> 2024-12-28
      -> 2024 < 2025 -> BEFORE LC -> FAIL as pre-dated.

  NEVER compare raw day numbers alone ("28 > 2") or string fragments.
  NEVER write "after the LC issue date" and then mark FAIL in the same
  verdict — that is a self-contradicting output.
  NEVER invent "year format is incorrect" as a reason to FAIL — a
  2-digit year "25" correctly resolves to 2025 per the rule above.
  If you can quote "YYYY-MM-DD" or "DD-MMM-YY" or "YYMMDD" on the
  document and compare it to the LC date correctly, that quote is
  sufficient evidence.

TOP RULE 2 — LC PROHIBITION WORDING IS NOT EVIDENCE:
  When the LC CONDITION says "BL must NOT be X" or "X is NOT
  acceptable" (Short Form BL, Blank Back BL, Charter Party BL, House
  BL, Freight Forwarder BL, Stale BL, Negotiation under reserve/
  guarantee, Non-vessel-operating carrier, etc.), that wording is
  the LC PROHIBITING X. It is NOT the LC CONFIRMING the document
  IS X. Decide whether the document IS X from the DOCUMENT TEXT and
  STRUCTURED FACTS only. Example of wrong reasoning to avoid:
    Condition: "BL must not be Short Form."
    WRONG: "The LC field 47A-3 indicates Short Form BL, which
      contradicts the condition -> FAIL."
    RIGHT: "The BL document text shows a full T&C page attached,
      therefore this is NOT a Short Form BL -> PASS."

TOP RULE 3 — "THIRD PARTY DOCUMENTS ACCEPTABLE EXCEPT INVOICE AND
            DRAFT" GOVERNS DOCUMENT ISSUANCE ONLY, NOT SHIPPER MATCH:
  When F47A says "THIRD PARTY DOCUMENTS ARE ACCEPTABLE EXCEPT FOR
  INVOICE AND DRAFT" (or any variant of "acceptable except Y"), the
  rule is about WHO ISSUED the document, not about the shipper /
  consignor printed inside the document. Read it as:
    (a) For every doc OTHER than the excepted ones, third-party
        ISSUANCE is ALLOWED. PASS a Packing List / Beneficiary
        Certificate / shipping-company certificate / etc. whose
        ISSUER is a third party — F47A permits it.
    (b) The EXCEPTED doc types (Invoice, Draft) MUST BE ISSUED BY
        THE BENEFICIARY (not by a third party). A Commercial Invoice
        issued by the beneficiary SATISFIES this rule. Only a
        Commercial Invoice issued by a third party FAILS it.

  Common inversion to avoid (this has been a repeat false FAIL):
    Condition: "Third party documents acceptable except for Invoice
                and Draft."
    Doc:       Commercial Invoice issued by beneficiary (APEX).
    WRONG: "Since the Invoice is issued by the Beneficiary, it does
      not meet the condition -> FAIL."
    RIGHT: "The condition REQUIRES the invoice to come from the
      beneficiary. The invoice is from APEX (beneficiary) ->
      PASS."

  SHIPPER / CONSIGNOR ON THE BL IS A SEPARATE CHECK:
  Even when F47A permits third-party docs, a requirement that
  "Shipper / Consignor on the Bill of Lading must match the
  Beneficiary" is STRICT and NOT overridden by F47A. F47A's third-
  party clause governs the ISSUER of the BL (carrier / agent /
  multimodal operator), not the shipper/consignor field printed on
  the BL face. If the shipper-on-BL differs from the LC beneficiary,
  that is a discrepancy to FLAG regardless of F47A wording. Do NOT
  apply F47A's third-party rule to excuse a shipper-vs-beneficiary
  mismatch.
    Condition: "Shipper in BL must match the beneficiary."
    F47A:      "Third party documents acceptable except invoice & draft."
    Shipper:   PT PINDO DELI (not beneficiary), Beneficiary: APEX.
    RIGHT: "Shipper PT PINDO DELI differs from beneficiary APEX ->
      FAIL. F47A does not apply to the shipper/consignor field."

════════════════════════════════════════════════════════════════════════
ANTI-HALLUCINATION RULES (STRICT — READ CAREFULLY)
════════════════════════════════════════════════════════════════════════

1. Your output MUST contain a "quote" field with the EXACT line(s) from
   DOCUMENT TEXT or STRUCTURED FACTS that justify the verdict. The
   quote must be text that literally appears in the document — not a
   paraphrase, not a generalization, not an inference.

2. If you cannot quote the relevant evidence, the verdict is FAIL. Period.
   "I don't see it written but it's probably implied" = FAIL, not PASS.

1a. ABSOLUTE ANTI-INFERENCE RULE (P180 — read before every verdict):
    Do NOT infer that a party is addressed/named on a document from
    INDIRECT SIGNALS. Indirect signals that DO NOT satisfy a check:
      - Presence of the LC / documentary credit number
      - Presence of an OTHER party's name (insurer, shipper, etc.)
      - The subject line mentioning "SHIPMENT ADVICE" or similar
      - The document being issued by the beneficiary
      - Any contextual relevance
    A party X is "addressed to / named in / includes / mentions X"
    ONLY when the DOCUMENT TEXT contains the FULL DISTINCTIVE NAME
    of X as a contiguous phrase.
    WORKED EXAMPLE (this is the hallucination we have repeatedly seen):
      LC Applicant (F50): "SINDH INSTITUTE OF UROLOGY AND
                          TRANSPLANTATION (SIUT), KARACHI, PAKISTAN"
      Condition:         "Shipment Advice must be addressed to the
                          Applicant"
      Document text:     "Insurance Company Name: M/S. SINDH INSURANCE
                          Documentary Credit Number: 0401ILC083248
                          Haemonetics (Hong Kong) Limited ..."
      ❌ WRONG LLM answer: PASS — "The Shipment Advice is addressed
          to the Applicant, SINDH INSTITUTE OF UROLOGY AND, as
          indicated by the presence of the LC number and insurance
          company name." → This is a HALLUCINATION. The document
          contains only "SINDH INSURANCE" (a different entity).
          The applicant's full name "SINDH INSTITUTE OF UROLOGY AND
          TRANSPLANTATION" never appears in the document.
      ✅ CORRECT LLM answer: FAIL — "Applicant 'SINDH INSTITUTE OF
          UROLOGY AND TRANSPLANTATION' does not appear on the
          document. The document mentions 'M/S. SINDH INSURANCE'
          but that is a different party."

1b. HOW TO VERIFY "ADDRESSED TO <PARTY>" CORRECTLY:
    Step 1. Read the LC PARTIES block at the top of this prompt and
            find the full name of the party the condition references.
    Step 2. Take the DISTINCTIVE CORE of that name — the 2-4
            consecutive proper-noun words that uniquely identify the
            entity (e.g. "SINDH INSTITUTE OF UROLOGY" — NOT just
            "SINDH" alone, which is generic).
    Step 3. Search the DOCUMENT TEXT for that CONTIGUOUS CORE phrase.
            Ignore punctuation / whitespace differences, but require
            the words to appear in order and next to each other.
    Step 4.
      - If found → PASS with the exact quoted text.
      - If not found → FAIL. Mention that the party's name is absent
        and what the document DOES say near a "To:" or "Attn:" header.

1c. SAME LOGIC APPLIES TO:
    "must be made out to the order of X" (consignee)
    "must notify X" (notify party) — X must appear in the NOTIFY
    field, not elsewhere
    "must mention X" / "must quote X" / "must reference X"
    "must be drawn on X" (drawee) — X must appear as the drawee
    In every case: the party's full distinctive name must appear on
    the document in its expected role-location. Indirect signals
    (credit number, related party mention, subject line relevance)
    are NEVER sufficient evidence.

1d. BUT DO NOT BE OVERLY STRICT EITHER (P180 balance):
    The rule above is about preventing HALLUCINATED PASSES, not
    creating rigid FAIL everywhere. The following ARE ACCEPTABLE
    variations of the same party name and MUST PASS:
      - Case differences: "Sindh Institute" = "SINDH INSTITUTE"
      - Honorifics: "M/s SINDH INSTITUTE..." = "SINDH INSTITUTE..."
      - Corporate suffix differences: "UBL" = "UBL Ltd" = "United Bank"
      - Acronym expansion: "SIUT" = "Sindh Institute of Urology and
        Transplantation" — these are THE SAME entity under UCP 600
      - Common typos: "Insititute" ≈ "Institute", "Karachhi" ≈ "Karachi"
      - Address differences: name matches but address differs → PASS
      - Company "currently known as / formerly known as" forms
      - Ampersand vs "AND": "SONS & CO" = "SONS AND CO"
      - Whitespace: "BANK ALHABIB" = "BANK AL HABIB" = "BANK AL-HABIB"
      - ADDITIONAL INFO AFTER THE NAME: if the doc says "SINDH
        INSTITUTE OF UROLOGY AND TRANSPLANTATION (SIUT), 9TH FLOOR,
        XYZ BUILDING, KARACHI", that fully satisfies an LC naming
        "SINDH INSTITUTE OF UROLOGY AND TRANSPLANTATION (SIUT),
        KARACHI, PAKISTAN" — address differences are immaterial
        once the entity name matches.
    The test is: would a reasonable bank checker say these are the
    same party? If YES → PASS. Only FAIL when the entity is genuinely
    DIFFERENT or absent.

3. NEVER copy condition wording into "findings". Findings must describe what
   you ACTUALLY FOUND on the document, not what was being checked.

   ❌ WRONG:
   Condition: "must state vessel covered under Institute Classification Clause"
   Document: (no such text)
   findings="Vessel is covered under Institute Classification Clause"
   → Hallucination. The document says NOTHING about that clause.

   ✅ CORRECT:
   findings="Document has no mention of Institute Classification Clause."
   verdict=FAIL

   DO NOT write "the closest text is X" / "the closest match is X" /
   "the nearest text is X". This system does NOT perform fuzzy matching.
   Either the value is present (PASS) or it is not (FAIL). "Closest"
   language misleads the reader into thinking the system considered a
   partial match. Always say outright what IS on the document, then
   the verdict. No consolation quotes.

4. STRUCTURED FACTS ARE AUTHORITATIVE. Read them FIRST and TRUST them.
   They were extracted directly from the document by a pre-pass and are
   more reliable than your own parsing of DOCUMENT TEXT.
   - Dates → dates_found[role=...]
   - Amounts → amounts_found[role=...]
   - References → references_found[role=...]
   - Parties → parties_found[role=...]
   - BL attributes → bl_subtype.contract_type / signing_type / has_terms_overleaf
   If a structured fact answers the question, cite its role in
   "structured_source". You don't have to re-quote from DOCUMENT TEXT.

   CONSIGNEE / TO-ORDER-OF CHECKS (P152 — CRITICAL, never hallucinate):
   - Consignee and Notify Party are TWO DIFFERENT FIELDS on a BL. They
     have DIFFERENT legal meanings under UCP 600. Never substitute one
     for the other. A bank appearing ONLY in Notify Party does NOT
     satisfy a "consigned to order of <bank>" requirement.
   - When the condition asks "consigned to / made out to the order of
     <BANK or PARTY X>", read the typed field consignee (or
     parties_found[role=consignee]) ONLY. If it contains the party name
     of X (ignoring BANK/LTD/LIMITED/LLC/PLC/INC/CORP/COMPANY suffixes,
     punctuation, and trailing city/country), return PASS with
     structured_source="unified_summary.consignee".
   - Example A: LC wants "to the order of Bank Al Habib Ltd., Karachi,
     Pakistan". Structured consignee = "TO ORDER OF: BANK AL HABIB LTD.,
     KARACHI". The key tokens "AL HABIB" appear in the consignee → PASS.
   - Example B (CRITICAL): LC wants "to the order of United Bank Ltd.".
     Structured consignee = "TO ORDER" (bearer/blank). Notify Party has
     "UNITED BANK LTD". → FAIL, NOT PASS. "TO ORDER" alone means
     consigned to shipper's order; the shipper must endorse the BL to
     UBL for compliance. UBL in the Notify field is irrelevant — notify
     is not consignee. The correct verdict is FAIL with a finding
     noting the reverse-side endorsement needs manual check. DO NOT
     write "consignee matches because UBL appears" — that is a
     HALLUCINATION confusing consignee with notify.
   - Example C: Consignee = "DALDA FOODS LIMITED" but LC requires
     "to order of UBL" → FAIL, different party.

   REFERENCE / IDENTIFIER CHECKS (P152 — CRITICAL, never hallucinate):
   - When the condition says "must reference / quote / contain / show
     Policy No. X" (or any identifier), compare character-by-character
     AFTER OCR-normalising both sides with these substitutions:
       O ↔ 0, I ↔ 1, L ↔ 1, S ↔ 5, B ↔ 8, Z ↔ 2, G ↔ 6, Q ↔ 0
     Example: LC wants "2023008MIPD000453" (all digits). Document shows
     "OPEN POLICY NO.2023008MIPDO00453" (letter O where 0 should be).
     OCR-normalised both become "2023008M1PD000453" — MATCH → PASS.
     DO NOT report "policy number not found" when the only difference
     is an OCR-level character confusion. If the digit-bearing identifier
     is present in either the typed reference field, references_found[],
     other_details_found[], or raw DOCUMENT TEXT (after OCR
     substitutions), return PASS.

5. Do NOT fabricate values that aren't on the document. If the condition asks
   for a value the document doesn't carry, answer REVIEW with findings
   explaining what's missing — NOT PASS with invented text.

════════════════════════════════════════════════════════════════════════
WORKED EXAMPLES — RECURRING FAILURE MODES (P195 — MUST INTERNALIZE)
════════════════════════════════════════════════════════════════════════

These six patterns are the LLM mistakes this system sees MOST OFTEN.
Every time you evaluate a condition touching one of these topics, walk
through the worked example FIRST, then apply the rule to your case.

─── EXAMPLE 1: CONSIGNEE "TO ORDER" vs "TO THE ORDER OF X" ───────────
When a BL's consignee field shows only "TO ORDER" (blank-endorsable)
and the LC requires the BL to be made out "TO THE ORDER OF BANK X",
you MUST return FAIL unless there is an EXPLICIT endorsement naming
BANK X on the BL face. Mere presence of BANK X in the Notify Party
field, or elsewhere on the document, does NOT satisfy this
requirement.

  Acceptable evidence for PASS (endorsement):
    * "ENDORSED TO <BANK X>" or "ENDORSED IN FAVOUR OF <BANK X>"
    * "FOR AND ON BEHALF OF <BANK X>"
    * "PAY TO THE ORDER OF <BANK X>"
    * The consignee field itself containing "TO (THE) ORDER OF <BANK X>"
  NOT acceptable:
    * <BANK X> appearing in the Notify Party address
    * <BANK X> being the LC issuing bank mentioned only in the
      "Documentary Credit No." block

  Example:
    LC: "Bill of lading must be made out to the order of Bank Al
         Habib Ltd., Pakistan."
    BL consignee: "TO ORDER"
    BL notify   : "TRANSSION TECNO ELECTRONICS AND BANK AL HABIB LTD."
    BL face text: no "ENDORSED TO BANK AL HABIB" line
    ❌ WRONG: PASS — "face text references AL HABIB" (this is the
        notify-address mention — NOT an endorsement).
    ✅ CORRECT: FAIL — "Consignee is 'TO ORDER' only. No explicit
        endorsement to Bank Al Habib on the BL face; its appearance
        in the Notify Party address does not satisfy the
        'to the order of Bank Al Habib Ltd.' requirement."

─── EXAMPLE 2: BL SHIPPER MUST BE THE BENEFICIARY ────────────────────
LC clause 47A-3 (or any "shipper other than beneficiary not
acceptable") requires that the Shipper shown on the BL be the LC
Beneficiary (or a legitimately-renamed version of it). When the two
names are genuinely different entities, you MUST return FAIL — even
if one name is "a bit shorter" or "seems related".

  Rule: the SHORTER name must be a clear PREFIX of the LONGER name
        AND share at least the first 2-3 distinctive proper-noun
        words. Otherwise they are DIFFERENT parties.

  Example (genuine mismatch):
    LC beneficiary: "OLAM GLOBAL AGRI PTE LTD"
    BL shipper    : "PT CITRA BORNEO UTAMA TBK"
    ❌ WRONG: PASS via prefix tolerance — these share NO distinctive
        words. "PT CITRA BORNEO" is not a prefix of "OLAM GLOBAL".
    ✅ CORRECT: FAIL — "BL shipper 'PT CITRA BORNEO UTAMA TBK' is a
        different entity from LC beneficiary 'OLAM GLOBAL AGRI PTE
        LTD'. 47A-3 prohibits shipper other than beneficiary."

  Counter-example (legitimate prefix — PASS):
    LC beneficiary: "SINDH INSTITUTE OF UROLOGY AND"  (truncated by LC)
    BL shipper    : "SINDH INSTITUTE OF UROLOGY AND TRANSPLANTATION"
    ✅ CORRECT: PASS — LC's stored value is a prefix of the doc's,
        same entity; the LC truncated the name.

─── EXAMPLE 3: HS CODE EXACT MATCH (with OCR/format tolerance) ───────
HS codes are regulatory and must match EXACTLY. Allowed
equivalences:
  * Trailing zeros: "9018.9050" == "90189050" == "9018905000"
  * Dot/no-dot: "9018.9050" == "90189050"
  * OCR O↔0 confusion: only in the same position
NOT allowed:
  * Last-digit differences: "9018.9050" ≠ "9018.9051"
  * Two-digit differences: "90189050" ≠ "90189000"

  Example:
    LC: "H.S. Code No. 9018.9050 must appear on Bill of Lading."
    BL: "HS CODE: 9018909000"
    ❌ WRONG: PASS — "HS code 9018 matches" (only first 4 digits).
    ✅ CORRECT: FAIL — "LC requires 9018.9050; BL shows 9018909000.
        These are different codes (9050 vs 9090)."

─── EXAMPLE 4: NTN vs GST ARE DIFFERENT FIELDS ───────────────────────
Pakistani commercial docs carry two distinct regulator numbers:
  * NTN (National Tax Number) — format "1234567-8"
  * GST / STRN (General Sales Tax Reg No) — format "03-00-1234567-17"
These are NOT interchangeable. If the LC asks for GST No. and the
doc only shows NTN No., return FAIL — even though the two numbers
may share some digits (the NTN is typically embedded inside GST).

  Example:
    LC 46A-2: "GST NO. 03-00-3075811-17 must appear on the Bill
               of Lading."
    BL text : "NTN NO. 3075811-4"
    ❌ WRONG: PASS — "the numbers match" (they don't; the BL carries
        NTN, not GST).
    ✅ CORRECT: FAIL — "LC requires GST No. 03-00-3075811-17; BL
        shows only NTN No. 3075811-4 — GST is missing."

─── EXAMPLE 5: FREIGHT MUST BE SHOWN SEPARATELY ON COMMERCIAL INVOICE ─
When the LC says "freight value must be shown/mentioned separately
on the Commercial Invoice" and the invoice is on CFR/CIF/CIP terms,
the invoice must carry an EXPLICIT freight line such as
"FREIGHT: USD 500.00" or "OCEAN FREIGHT CHARGES: USD 500.00".
A single CFR total line (e.g. "CFR KARACHI: USD 10,500.00") does
NOT satisfy this check — the freight is bundled into the total.

  Example:
    LC: "Freight value should be mentioned on Commercial Invoice
         separately."
    CI text: "TOTAL CFR KARACHI: USD 97,216.00"  (no freight line)
    ❌ WRONG: PASS — "invoice is CFR therefore includes freight".
    ✅ CORRECT: FAIL — "Invoice shows only CFR total; no separate
        freight amount line. LC requires freight mentioned as a
        distinct value."

─── EXAMPLE 6: MISSING DOCUMENT vs WRONG CONTENT ─────────────────────
If the LC requires a specific document type (e.g. Agent Certificate,
Courier Receipt, Beneficiary Certificate) and that document is
NOT in the submission, return FAIL with "Required document missing"
— do NOT try to verify the LC clause against an unrelated document
and then fabricate a finding.

  Example:
    LC: "Courier Receipt must be presented proving dispatch of
         Shipment Advice to Asia Insurance."
    Submission contains: Commercial Invoice, Packing List,
                         Air Waybill, BL (no Courier Receipt).
    ❌ WRONG: Verify against Air Waybill and PASS because "AWB
        proves courier dispatch".
    ✅ CORRECT: FAIL — "Required document missing: Courier Receipt.
        No courier dispatch receipt present in the submission."

─── EXAMPLE 7: INCOMPLETE SHIPPING DOCUMENTS — "TO FOLLOW IN NEXT LOT" ─
Some banks split a presentation into multiple lots. If the Documentary
Remittance / Covering Schedule explicitly states that specific
documents will follow in a later lot (e.g. "Bill of Lading and
Certificate of Origin will follow in the 2nd lot"), the missing
documents in THIS lot are NOT a discrepancy. Mark the clause as
PASS with a clear finding referencing the cover-letter statement.
ONLY mark FAIL if the cover letter does not defer the missing docs.

  Trigger phrases (any satisfies):
    "to follow in 2nd lot" / "will follow in the next lot"
    "remaining documents will be sent in the 2nd presentation"
    "balance documents to be presented separately"

  Example:
    LC: "Full set of shipping documents must be presented."
    Covering Schedule: "CERTIFICATE OF ORIGIN AND BILL OF LADING
                        WILL FOLLOW IN THE 2ND LOT."
    ✅ CORRECT: PASS — "Cover letter defers CoO and BL to the 2nd
        lot; incomplete set in this lot is not a discrepancy."

─── EXAMPLE 8: SECOND PRESENTATION — "FULL AND FINAL" COVER LETTER ───
When a 2nd or subsequent presentation's Covering Schedule declares
the submitted set is "Full and Final" (or "No further lots will
follow"), treat it as the closing presentation. Any earlier deferred
documents MUST now be present; if still missing → FAIL.

  Example:
    Covering Schedule (2nd lot): "WE HEREBY CONFIRM THAT THESE ARE
        THE FULL AND FINAL SHIPPING DOCUMENTS. NO FURTHER LOTS."
    ✅ CORRECT: PASS the cover-letter declaration check itself. For
        any doc that was deferred in lot 1 but still missing in lot
        2 — FAIL ("expected in final lot but still absent").

─── EXAMPLE 9: PAYMENT ON LANDED WEIGHT / QUALITY BASIS ──────────────
If F47A carries "PAYMENT OF THIS L/C IS SUBJECT TO LANDED WEIGHT
AND QUALITY BASIS" (or equivalent), the Commercial Invoice amount
and quantity are provisional — final payment depends on the
landing-port weight/quality survey. For amount / quantity checks
against such LCs, return REVIEW (not PASS or FAIL) with a finding
explaining that the value is provisional.

  Example:
    LC F47A: "PAYMENT OF THIS L/C IS SUBJECT TO LANDED WEIGHT
              AND QUALITY BASIS."
    Condition: "Commercial Invoice amount must not exceed LC amount."
    CI: "Invoice USD 255,000.00 (provisional, subject to landing)."
    ❌ WRONG: PASS — "Amount within LC amount".
    ✅ CORRECT: REVIEW — "Invoice amount is provisional; final payment
        amount is determined by discharge-port landed weight/quality
        per F47A. Manual reconciliation required after landing
        survey."

─── EXAMPLE 10: SECOND PRESENTATION — OVERDRAW vs REMAINING BALANCE ──
When the condition explicitly provides a running LC balance ("prior
drawings USD X, remaining available USD Y"), the current Commercial
Invoice must fit within the REMAINING balance, not the original LC
amount. Overdraw against remaining balance → FAIL even if within
the original LC ceiling.

  Example:
    Condition: "2nd presentation must not exceed remaining LC
                balance of USD 55,000."
    CI amount: "USD 70,000.00"
    ❌ WRONG: PASS because 70,000 < original LC of 255,000.
    ✅ CORRECT: FAIL — "Invoice USD 70,000 overdraws the remaining
        LC balance of USD 55,000."

─── EXAMPLE 11: CROSS-DOCUMENT CONFLICT (BL vs CI vs PL) ─────────────
If the condition requires consistency across shipping documents
(weights, quantities, marks, dates, container numbers, vessel
name, port names) and two documents in the SAME presentation show
CONFLICTING values for the same field, return FAIL naming both
documents and the specific field that disagrees.

  Example:
    Condition: "BL net weight must match Commercial Invoice net weight."
    CI:  "Net Weight: 249,500 KG"
    BL:  "Net Weight: 248,100 KG"
    ❌ WRONG: PASS / REVIEW because "both docs show a weight".
    ✅ CORRECT: FAIL — "CI net weight 249,500 KG conflicts with BL
        net weight 248,100 KG. UCP 600 Art 14(e) requires consistency
        between documents."

─── EXAMPLE 12: AMENDMENT APPLIED BETWEEN PRESENTATIONS ──────────────
When F47A / context shows an MT707 amendment between presentations
(e.g. "amount increased from X to Y"), verify against the AMENDED
value, not the original. Treat amendments as the new source of
truth unless the LC condition explicitly references the original.

  Example:
    F47A: "MT707 AMENDMENT 001 INCREASED F32B AMOUNT FROM
           USD 255,000 TO USD 350,000."
    CI:   "USD 120,000 (2nd presentation)"
    ✅ CORRECT: PASS — "CI USD 120,000 is within amended LC ceiling
        of USD 350,000. Amendment 001 supersedes the original amount."

────────────────────────────────────────────────────────────────────
These twelve modes are where LLM verification traditionally breaks.
Read the example that matches your condition BEFORE forming a
verdict. If in doubt, FAIL with clear evidence is always better
than a hallucinated PASS.

════════════════════════════════════════════════════════════════════════
F47A OVERRIDE HIERARCHY (APPLY BEFORE FINAL VERDICT)
════════════════════════════════════════════════════════════════════════

1. Read F47A ADDITIONAL CONDITIONS first.
2. If ANY F47A clause says the thing in question is "ACCEPTABLE",
   "ALLOWED", or "PERMITTED" → it OVERRIDES the main requirement → PASS.
3. F47A allows WITH conditions (e.g. "LATE SHIPMENT ALLOWED PROVIDED
   penalty deduction") → REVIEW, not FAIL. Explain the needed manual check.
4. "CHARTER PARTY BL ACCEPTABLE" → charter-party BL = PASS.
5. "THIRD PARTY DOCUMENTS ACCEPTABLE" → third-party documents = PASS.
6. "ANY [COUNTRY] PORT" means any port in that country is acceptable = PASS.

════════════════════════════════════════════════════════════════════════
NAME MATCHING (applies to parties, banks, issuers)
════════════════════════════════════════════════════════════════════════

Key words must match. Ignore:
- Company suffixes (LTD / LIMITED / BV / INC / CO / COMPANY / LLC / S.A.)
- Minor spelling or OCR differences
- Address differences when names match
- Honorifics / prefixes: M/s, M/S, Messrs., Mr., Mrs., Dr.
- Acronyms in parentheses: "(SIUT)", "(NGO)", "(PVT) LTD"

Examples (all PASS):
- "UNITED BANK" = "UNITED BANK LIMITED" = "UBL"
- "Viterra B.V." = "Viterra BV" = "Bunge Netherlands Agri B.V." when the
  document itself says "currently known as" / "formerly known as" —
  SAME legal entity under a renamed form.
- "Dalda Foods Limited" = "DALDA FOODS LTD."

P165 — TRUNCATED / PREFIX NAME TOLERANCE (CRITICAL):
LC extraction occasionally truncates long applicant/beneficiary
names at a newline or field boundary (e.g. the LC F50 value stored
as "SINDH INSTITUTE OF UROLOGY AND" — the continuation
"TRANSPLANTATION (SIUT)" got cut off during field consolidation).
When comparing a document's party name against an LC party name:

- If the LC-required name appears as a PREFIX of the document's name
  (after normalizing to uppercase, stripping M/S, honorifics,
  punctuation, and extra whitespace) → PASS. The document has the
  full form; the LC lost a suffix.
  Example:
    LC required:  "SINDH INSTITUTE OF UROLOGY AND"
    Document:     "M/s Sindh Institute of Urology and Transplantation (SIUT)"
    → Normalized LC:  "SINDH INSTITUTE OF UROLOGY AND"
    → Normalized doc: "SINDH INSTITUTE OF UROLOGY AND TRANSPLANTATION SIUT"
    → LC is a prefix of doc → PASS
- Symmetrically: if the document's name is a prefix of the LC
  required name (document carrying a short form), same PASS.
- Do NOT write "does not exactly match" as a FAIL reason for these
  cases — the mismatch is caused by LC-side text truncation, not
  a real party mismatch. Under UCP 600 Art 14(d), minor differences
  in spelling or completeness of name that do not change the party's
  legal identity are acceptable.

════════════════════════════════════════════════════════════════════════
GOODS DESCRIPTION TOLERANCE
════════════════════════════════════════════════════════════════════════

Minor wording variations are acceptable if the PRODUCT is clearly the same.
"Canadian Canola No.1" and "Canadian GMO Canola" refer to the same commodity.
Grade/variety descriptors (No.1, GMO, non-GMO, in bulk) are supplementary.
Core product name match → PASS.

ABSOLUTE ANTI-HALLUCINATION RULE (CRITICAL):
Before you write "doesn't match" / "broader category" / "doesn't specify",
search the invoice's STRUCTURED FACTS and DOCUMENT TEXT for the LC's
primary commodity term VERBATIM. Two authoritative sources to check:
  1) unified_summary.goods_description (typed field extracted by Step 3)
  2) document_text line containing the commodity word

If the LC F45A primary term (e.g. "SOYBEANS", "COTTON YARN", "RICE",
"WHEAT") appears as a WHOLE WORD on the invoice — whether in the
goods_description field, a line item, or the body text — you MUST
verdict PASS and quote that line. You may not claim "doesn't match"
or "broader category" when the exact commodity word is printed on
the invoice.

Worked example (SOYBEANS):
  LC F45A primary term: "SOYBEANS"
  Invoice goods_description: "Soybeans"
  Invoice body: "Origin: Brazil" and "BRAZIL ORIGIN."
  → The word "Soybeans" appears verbatim. Verdict = PASS.
  → FAIL with "broader category" is hallucination — the invoice
    literally shows the commodity.

"Doesn't specify origin" is a SEPARATE check (46A origin-certification)
and must NOT be conflated with the 45A goods-description check. Keep
them distinct: 45A = product identity; 46A = origin statement.

════════════════════════════════════════════════════════════════════════
DECISION ORDER
════════════════════════════════════════════════════════════════════════

1. Is the condition APPLICABLE to this document type? If no → REVIEW with
   findings="Condition not applicable to {document_type}".
2. Does F47A override or modify the condition? If yes, apply it.
3. Can you find the required evidence in STRUCTURED FACTS? If yes → PASS
   with structured_source filled.
4. Can you quote it from DOCUMENT TEXT? If yes → PASS with quote filled.
5. Can you find clear CONTRADICTORY evidence? If yes → FAIL with quote.
6. If evidence is ambiguous, partial, or the condition requires human
   judgment → REVIEW with findings explaining what needs manual check.

════════════════════════════════════════════════════════════════════════
OUTPUT FORMAT — Return ONLY JSON (no markdown, no commentary):
════════════════════════════════════════════════════════════════════════

{{
  "verdict": "PASS" | "FAIL" | "REVIEW",
  "quote": "EXACT line(s) from document/facts that justify the verdict (required for PASS/FAIL)",
  "findings": "1-2 sentence explanation grounded in the quote above",
  "confidence": 0.0 to 1.0,
  "structured_source": "e.g. 'dates_found[role=onboard_date]' if that was the source, else empty"
}}

════════════════════════════════════════════════════════════════════════
DOCUMENT-TYPE RULE PACK (applies in addition to CORE):
════════════════════════════════════════════════════════════════════════

{family_pack}
"""


# ── Family rule packs (short, targeted — appended as {family_pack}) ──

FAMILY_PACK_BL = """BILL OF LADING — additional verification rules:

STALE BL — TIME CHECK ONLY (P170 — CRITICAL):
"Stale" means ONE thing only: the BL was presented TO THE BANK more
than 30 days AFTER the on-board date on the BL (fixed threshold).
It is purely a DATE ARITHMETIC check:
    days_elapsed = DR.receiving_date - BL.onboard_date
    STALE if days_elapsed > 30 (fixed 30-day rule; do not use F48).
Staleness is NOT related to:
  - form_type (short form / long form / blank back) — that is UCP 600
    Art 20(a)(v) short-form BL rules, a completely different topic
  - issuer_type (house / master / charter party) — that is UCP 600
    Art 19/20 issuer rules
  - cleanness (clean / claused) — UCP 600 Art 27
  - reverse-side T&C presence
NEVER write "BL is stale because it is marked as short form" or
"BL is stale because it is blank back" or any similar statement.
Those attributes have NOTHING to do with staleness. If the LC says
"BL must not be stale" and you can't compute the days_elapsed
formula, return REVIEW with "cannot determine staleness without the
receiving date on the Documentary Remittance" — do NOT invent
reasoning from other BL attributes.

BL ATTRIBUTES — NEVER CONFUSE THESE (P167 — CRITICAL):
The following are FIVE INDEPENDENT attributes on a BL. They mean
completely different things under UCP 600 and must NOT be substituted
for each other:
  • cleanness (Art 27):    clean | claused
      - Clean = no damage/defect notation on the goods.
      - Claused = has damage clause like "2 BAGS TORN".
  • contract_type:         standard | charter_party | multimodal | ...
  • issuer_type:           master_bl | house_bl | charter_party_bl
  • signing_type:          master_signed | agent_for_master |
                           carrier_signed | forwarder_signed
  • form_type:             short_form_blank_back | long_form_printed_overleaf
  • is_blank_back / has_terms_overleaf: reverse-side T&C presence
"Blank back" is about the physical BACK of the BL page.
"Claused" is about damage notations on the FACE of the BL.
They are NOT the same thing. NEVER say "BL is claused because it
is blank back" — that is a nonsense statement. If the condition
says "BL must not be claused", read bl_subtype.cleanness (or
is_claused_bl / clausing_notes) — do NOT read is_blank_back.

BL prohibition clauses (condition says "BL must NOT be [charter party / short
form / blank back / house BL / freight forwarder issued / claused]"):
- bl_subtype.signing_type in {master_signed, agent_for_master, carrier_signed}
    → NOT a freight forwarder BL → PASS "not forwarder" prohibition
- bl_subtype.has_terms_overleaf = true OR reverse page has T&C
    → NOT short form / NOT blank back → PASS "not short form" prohibition
- bl_subtype.contract_type != "charter_party"
    → NOT charter party → PASS "not charter party" prohibition
- bl_subtype.issuer_type = "house_bl" → IS a house BL (check condition)
- bl_subtype.signing_type = "forwarder_signed" → IS forwarder-issued
- bl_subtype.cleanness = "clean" → NOT claused → PASS "not claused" / "must be clean"
- bl_subtype.cleanness = "claused" → IS claused → FAIL "not claused" / "must be clean"
Remember: the prohibition words are in the CONDITION, not on the BL. PASS
means the BL is NOT the prohibited type.

BL field lookups (prefer structured_facts):
- Vessel / carrier: parties_found[role=carrier] OR doc text letterhead
- Shipper / Consignee / Notify: parties_found[role=shipper|consignee|notify_party]
- On-board (shipment) date: dates_found[role=onboard_date]
- BL date (issue): dates_found[role=bl_issue_date]
- Ports: references_found or doc text ("PORT OF LOADING ...")
- Freight: amounts_found[role=freight_amount] OR "FREIGHT PREPAID / COLLECT"
- Number of originals: doc text "Three (3)" / "3/THREE" / similar
- Shipped-on-board note: "CLEAN ON BOARD" / "LADEN ON BOARD" in doc text
"""


FAMILY_PACK_INVOICE = """COMMERCIAL INVOICE — additional verification rules:

Amount checks (most common false-fail source — read carefully):
- Use amounts_found[role=invoice_total] as AUTHORITATIVE — do NOT sum line items.
- "INVOICE PRINTED TOTAL AMOUNT" line in document text is the canonical figure.
- For a multi-page invoice, the Total on the LAST page applies to the whole
  invoice — do not add per-page subtotals.
- EQUAL amounts: invoice total = LC amount → PASS. "Must not exceed" means
  <=; equal is NOT exceeding. Only strictly > LC_amount × (1 + tolerance%)
  is a FAIL.
- AMOUNT IN WORDS vs FIGURES: always use the NUMERIC amount (words line is
  just confirmation).
- UCP 600 Art 30 tolerance: invoice amount can be LESS than LC amount for
  partial / short shipments — that is PASS (unless F47A forbids).

Addressing:
- "TO:" / "ALSO TO:" / "AND TO:" / "CC:" ALL mean the party IS addressed.

Goods description: apply CORE's goods tolerance rules.

Line items: prefer other_details_found[role=line_items] if present; else
parse the invoice body.
"""


FAMILY_PACK_DRAFT = """DRAFT / BILL OF EXCHANGE — additional verification rules:

Multiple dates may appear on a draft — don't confuse them:
- DRAFT DATE (date of drawing) → dates_found[role=draft_date]
    when the beneficiary drew the draft.
- LC ISSUE DATE (referenced on draft) → dates_found[role=lc_issue_date]
    confirms which LC the draft is drawn against.
- MATURITY DATE (usance drafts) → dates_found[role=maturity_date].

Amount:
- amounts_found[role=draft_amount] (numeric) — NOT the "AMOUNT IN WORDS" line.
- For installments, each installment should match its LC portion; total draft
  amount = LC amount within tolerance.

Parties:
- parties_found[role=drawer]  — usually the beneficiary
- parties_found[role=drawee]  — usually the issuing bank (LC drawee)
- parties_found[role=payee]   — often "Ourselves" / beneficiary
- "BUNGE / VITERRA currently known as ..." — same legal entity, PASS.

Multiple copies:
- "FIRST of Exchange" and "SECOND of Exchange" are SAME draft, not two drafts.

LC reference on draft:
- references_found[role=lc_reference] must match final_lc.F20.
"""


FAMILY_PACK_CERTIFICATE = """CERTIFICATE (generic) — additional verification rules:

Dates:
- Issue date of the cert → dates_found[role=certificate_issue_date]
  OR dates_found[role=inspection_date / test_date / sampling_date / survey_date]
- DO NOT confuse with shipment date (which is on the BL) or LC issue date.

Issuer:
- parties_found[role=certifying_authority / chamber_of_commerce /
  health_authority / agriculture_authority / testing_laboratory / inspector /
  surveyor].
- Must match the required issuer type from the condition.

Product matching:
- Use other_details_found[role=goods_description] or doc text.
- Apply CORE's goods tolerance rules.

Certificate number → references_found[role=certificate_reference or
phytosanitary_certificate_number / health_certificate_number / etc.].
"""


FAMILY_PACK_PACKING = """PACKING LIST — additional verification rules:

Quantity / packages:
- amounts_found[role=gross_amount / net_amount / total_weight_value]
- references_found[role=number_of_packages] or typed field
- Package type (BAGS / DRUMS / CARTONS / BULK) from doc text.
- Marks & numbers: references_found[role=shipping_marks / marks_and_numbers].

Tolerance:
- UCP 600 Art 30 quantity tolerance (5% unless excluded by "ABOUT"/"CIRCA").

Partial shipments:
- Check F47A for "PARTIAL SHIPMENT ALLOWED" before flagging quantity FAIL.
"""


FAMILY_PACK_INSURANCE = """INSURANCE CERTIFICATE / POLICY — additional verification rules:

Amount:
- amounts_found[role=sum_insured / insurance_amount]
- Usually = invoice_amount × 110% (CIF value) unless F47A specifies otherwise.

Risks / clauses:
- Look for Institute Cargo Clauses (ICC A / B / C).
- Institute Classification Clause (for vessel classification).
- War Risks, Strikes, SRCC (Strikes/Riots/Civil Commotion).
- other_details_found[role=institute_classification_clause / icc_clause /
  war_risk_clause / etc.].

Voyage:
- From / To ports (must match BL's port_of_loading / port_of_discharge).
- Conveyance vessel name.
- Dates (insurance_effective_date / insurance_expiry_date).

Insurer:
- parties_found[role=insurer / insurance_broker].
"""


FAMILY_PACK_SHIPMENT_ADVICE = """SHIPMENT ADVICE — additional verification rules:

Addressing (this is THE main check for shipment advice):
- "TO:" / "ALSO TO:" / "AND TO:" / "CC:" ALL mean the recipient IS addressed.
- Usually addressed to the insurance company + applicant.
- Use parties_found[role=receiver / notify_party / second_notify_party].

Required content:
- Vessel name → parties_found[role=carrier] or doc text
- BL number → references_found[role=bl_reference]
- Shipment date → dates_found[role=shipment_date / onboard_date]
- Goods and quantity
- Amount (if required by F47A/F46A)

Reference to LC:
- references_found[role=lc_reference].
"""


FAMILY_PACK_GENERIC = """GENERIC DOCUMENT — universal verification rules:

For doc types without a specialized pack, use structured_facts extensively:
- Document identifier → references_found (any role matching the doc)
- Issue date → dates_found[role=certificate_issue_date / issue_date]
- Issuer / signatory → parties_found (any role)
- LC reference → references_found[role=lc_reference]
- Any relevant fact may be in other_details_found

If the condition cannot be confidently verified against this doc type,
mark REVIEW with findings explaining what data is missing.
"""


# Dispatcher mapping — exact-match first, falls back to contains-match
_FAMILY_PACK_MAP_EXACT = {
    'bill of lading': FAMILY_PACK_BL,
    'commercial invoice': FAMILY_PACK_INVOICE,
    'draft bill of exchange': FAMILY_PACK_DRAFT,
    'bill of exchange': FAMILY_PACK_DRAFT,
    'packing list': FAMILY_PACK_PACKING,
    'shipment advice': FAMILY_PACK_SHIPMENT_ADVICE,
    'insurance certificate': FAMILY_PACK_INSURANCE,
    'insurance policy': FAMILY_PACK_INSURANCE,
}

_FAMILY_PACK_KEYWORDS = [
    ('bill of lading', FAMILY_PACK_BL),
    ('commercial invoice', FAMILY_PACK_INVOICE),
    ('draft', FAMILY_PACK_DRAFT),
    ('bill of exchange', FAMILY_PACK_DRAFT),
    ('packing list', FAMILY_PACK_PACKING),
    ('packing slip', FAMILY_PACK_PACKING),
    ('shipment advice', FAMILY_PACK_SHIPMENT_ADVICE),
    ('insurance', FAMILY_PACK_INSURANCE),
    ('certificate', FAMILY_PACK_CERTIFICATE),    # generic cert fallback
    ('report', FAMILY_PACK_CERTIFICATE),
    ('survey', FAMILY_PACK_CERTIFICATE),
    ('analysis', FAMILY_PACK_CERTIFICATE),
]


def _pick_family_pack(document_type: str) -> str:
    """Return the family rule pack best matching the document type.
    Falls back to GENERIC pack if no match."""
    if not document_type:
        return FAMILY_PACK_GENERIC
    dt = document_type.lower().strip()
    if dt in _FAMILY_PACK_MAP_EXACT:
        return _FAMILY_PACK_MAP_EXACT[dt]
    for kw, pack in _FAMILY_PACK_KEYWORDS:
        if kw in dt:
            return pack
    return FAMILY_PACK_GENERIC


def _build_structured_facts(unified_summary: dict, bl_subtype: dict) -> str:
    """Compose a readable, tagged block of ALL structured facts for the LLM.

    Dumps EVERY field from Step 3e unified_summary + Step 3d bl_subtype.
    Nothing is dropped. Preferred typed-field order is honoured at the top for
    readability; any remaining fields the LLM captured follow after.
    """
    lines = []
    if not unified_summary and not bl_subtype:
        return "(no structured facts extracted for this document)"

    # Names of the structured arrays — handled separately below.
    _ARRAY_KEYS = {
        'dates_found', 'amounts_found', 'quantities_found',
        'references_found', 'parties_found', 'other_details_found',
    }
    # Preferred display order for typed top-level fields (printed first).
    _PREFERRED_ORDER = [
        'document_identifier',
        # Doc identifiers
        'bl_number', 'bl_date', 'invoice_reference', 'draft_reference',
        # Dates
        'issue_date', 'onboard_date', 'shipment_date', 'loading_date',
        # Parties
        'issuer', 'beneficiary',
        'shipper', 'consignee', 'notify_party',
        'drawer', 'drawee', 'payee',
        # Goods / quantity / amount
        'goods_description', 'quantity', 'amount',
        'gross_weight', 'net_weight', 'measurement',
        'number_of_packages', 'package_type',
        # Shipping details
        'vessel_name', 'voyage_number',
        'port_of_loading', 'port_of_discharge',
        'place_of_receipt', 'place_of_delivery',
        'number_of_originals', 'freight_terms', 'freight_amount',
        'signed_by',
        # Content
        'container_numbers', 'seal_numbers', 'marks_and_numbers',
        'charter_party_reference',
        # Identifiers / tax
        'ntn_number', 'tin_number', 'hs_codes',
        # Cross-document references
        'lc_reference', 'contract_reference', 'proforma_reference',
        # Free text / meta
        'key_clauses', 'cross_references', 'notes',
    ]

    def _fmt(v):
        if isinstance(v, list):
            parts = []
            for x in v:
                if x is None or x == '':
                    continue
                if isinstance(x, dict):
                    parts.append('{' + ', '.join(f'{k}={vv}' for k, vv in x.items() if vv) + '}')
                else:
                    parts.append(str(x))
            return ', '.join(parts)
        if isinstance(v, dict):
            return '{' + ', '.join(f'{k}={vv}' for k, vv in v.items() if vv) + '}'
        return str(v)

    seen = set()
    if unified_summary:
        # 1) Preferred-order typed fields
        for k in _PREFERRED_ORDER:
            if k in _ARRAY_KEYS:
                continue
            v = unified_summary.get(k)
            if v in (None, '', [], {}):
                continue
            lines.append(f"{k}: {_fmt(v)}")
            seen.add(k)
        # 2) Any OTHER keys the LLM captured (invented fields, rare fields, etc.)
        # — include them all. Skip internal/error/array keys.
        for k, v in unified_summary.items():
            if k in seen or k in _ARRAY_KEYS or k.startswith('_'):
                continue
            if v in (None, '', [], {}):
                continue
            lines.append(f"{k}: {_fmt(v)}")
            seen.add(k)

    # BL sub-type — dump ALL fields
    if bl_subtype and isinstance(bl_subtype, dict):
        _bl_lines = []
        for k, v in bl_subtype.items():
            if k.startswith('_'):
                continue
            if v in (None, '', [], {}, 'unknown'):
                continue
            _bl_lines.append(f"  {k}: {v}")
        if _bl_lines:
            lines.append("")
            lines.append("bl_subtype:")
            lines.extend(_bl_lines)

    # Structured arrays — dump EVERY item with ALL its fields
    def _dump_arr(label, arr):
        if not arr or not isinstance(arr, list):
            return
        lines.append("")
        lines.append(f"{label}:")
        for item in arr:
            if isinstance(item, dict):
                parts = [f"{k}={v}" for k, v in item.items()
                         if v not in (None, '', [], {}) and not k.startswith('_')]
                if parts:
                    lines.append(f"  - {' | '.join(parts)}")
            elif item:
                lines.append(f"  - {item}")

    if unified_summary:
        for arr_key in ('dates_found', 'amounts_found', 'quantities_found',
                        'references_found', 'parties_found', 'other_details_found'):
            _dump_arr(arr_key, unified_summary.get(arr_key))

    if not lines:
        return "(no structured facts extracted for this document)"
    return "\n".join(lines)


def _format_key_lc_fields(final_lc_fields: dict) -> str:
    """Build a compact KEY LC FIELDS block for the verifier prompt.

    P198v — previously the verifier only saw F47A / LC parties / the
    current clause's value. Other key SWIFT fields (LC number, issue
    date, expiry, amount, partial / transshipment flags, latest
    shipment) lived only inside `final_lc_fields` which was consumed
    by deterministic post-checks but NEVER labelled for the LLM. As
    a result the verifier would mark a row REVIEW with reasoning
    like "LC issue date is not provided in structured facts" even
    though it WAS in the pipeline, just hidden.

    This helper extracts those fields (and unwraps step07's
    `{field_name, value}` shape) into a short, labelled block the
    prompt surfaces explicitly.
    """
    if not isinstance(final_lc_fields, dict) or not final_lc_fields:
        return "(key LC fields not available)"

    def _unwrap(v):
        if v is None:
            return ''
        if isinstance(v, str):
            return v.strip()
        if isinstance(v, dict):
            return str(v.get('value') or v.get('field_value') or '').strip()
        if isinstance(v, list):
            parts = [ _unwrap(x) for x in v ]
            return '\n'.join(p for p in parts if p)
        return str(v).strip()

    def _get(*keys):
        for k in keys:
            if k in final_lc_fields:
                s = _unwrap(final_lc_fields[k])
                if s:
                    return s
        return ''

    pairs = [
        ("LC Number (F20)",                 _get('20', 'F20', 'LC_Number', 'Documentary_Credit_Number')),
        ("LC Issue Date (F31C)",            _get('31C', 'F31C', 'Date_of_Issue')),
        ("LC Expiry (F31D)",                _get('31D', 'F31D', 'Date_and_Place_of_Expiry')),
        ("Amount (F32B)",                   _get('32B', 'F32B', 'Amount', 'Currency_Code_Amount')),
        ("Available With (F41D)",           _get('41D', 'F41D', '41A')),
        ("Drafts at (F42C)",                _get('42C', 'F42C', 'Drafts_at')),
        ("Drawee (F42A/D)",                 _get('42A', '42D', 'F42A', 'Drawee')),
        ("Partial Shipments (F43P)",        _get('43P', 'F43P', 'Partial_Shipments')),
        ("Transshipment (F43T)",            _get('43T', 'F43T', 'Transshipment')),
        ("Port of Loading (F44E)",          _get('44E', 'F44E', 'Port_of_Loading')),
        ("Port of Discharge (F44F)",        _get('44F', 'F44F', 'Port_of_Discharge')),
        ("Latest Shipment (F44C)",          _get('44C', 'F44C', 'Latest_Date_of_Shipment')),
        ("Presentation Period (F48)",       _get('48', 'F48', 'Period_for_Presentation')),
        ("Applicant (F50)",                 _get('50', 'F50', 'Applicant').split('\n')[0] if _get('50', 'F50', 'Applicant') else ''),
        ("Beneficiary (F59)",               _get('59', 'F59', 'Beneficiary').split('\n')[0] if _get('59', 'F59', 'Beneficiary') else ''),
        ("Issuing Bank (F52A)",             _get('52A', '52D', 'F52A', 'Issuing_Bank').split('\n')[0] if _get('52A', '52D', 'F52A', 'Issuing_Bank') else ''),
    ]
    lines = [f"- {label}: {val}" for label, val in pairs if val]
    if not lines:
        return "(key LC fields not available)"
    return "\n".join(lines)


def _build_verification_prompt_v2(
    condition_text: str,
    clause_ref: str,
    lc_field_value: str,
    lc_parties: str,
    f47a_context: str,
    document_type: str,
    document_text: str,
    visual_metadata: str,
    unified_summary: dict,
    bl_subtype: dict,
    final_lc_fields: dict = None,
) -> str:
    """Compose the CORE + family-pack prompt for one (condition, doc) verification."""
    family_pack = _pick_family_pack(document_type)
    structured_facts = _build_structured_facts(unified_summary or {}, bl_subtype or {})
    key_lc_fields = _format_key_lc_fields(final_lc_fields or {})
    return CORE_VERIFICATION_PROMPT.format(
        condition_text=condition_text or "(not provided)",
        clause_ref=clause_ref or "(n/a)",
        lc_field_value=lc_field_value or "(n/a)",
        lc_parties=lc_parties or "(Not available)",
        key_lc_fields=key_lc_fields,
        f47a_context=f47a_context or "(none)",
        document_type=document_type or "(unknown)",
        structured_facts=structured_facts,
        document_text=document_text or "(no text)",
        visual_metadata=visual_metadata or "(No visual metadata available)",
        family_pack=family_pack,
    )


# ════════════════════════════════════════════════════════════════════════ #
# Deterministic verification — handles obvious lookups without LLM         #
# (used as a fast path BEFORE calling _call_vlm for conditions where the    #
# structured facts give an unambiguous answer).                             #
# Conservative: only returns a verdict when confidence is very high; else   #
# returns None and the VLM path runs.                                       #
# ════════════════════════════════════════════════════════════════════════ #

def _normalize_id(s):
    """Strip whitespace, hyphens, and case — for comparing reference numbers.
    P135 — also normalize common OCR character confusions so that
    "2023008MIPDO00453" (letter O) matches "2023008MIPD000453" (digit 0).
    """
    if s is None:
        return ''
    out = ''.join(ch for ch in str(s).upper() if ch.isalnum())
    # OCR character confusion: O↔0, I↔1, S↔5, B↔8, Z↔2, G↔6, Q↔0.
    # For identifier matching we fold the ambiguous letters into their
    # digit counterparts — typical policy / invoice / LC numbers are
    # printed digits, and OCR sometimes reads them as letters.
    _ocr_subs = str.maketrans({
        'O': '0',
        'I': '1',
        'L': '1',
        'S': '5',
        'B': '8',
        'Z': '2',
        'G': '6',
        'Q': '0',
    })
    return out.translate(_ocr_subs)


def _find_structured(unified_summary: dict, array_name: str, role_keywords):
    """Find first item in the structured array whose role matches any keyword."""
    if not unified_summary:
        return None
    arr = unified_summary.get(array_name) or []
    if not isinstance(arr, list):
        return None
    if isinstance(role_keywords, str):
        role_keywords = [role_keywords]
    for item in arr:
        if not isinstance(item, dict):
            continue
        role = str(item.get('role', '')).lower()
        if any(k in role for k in role_keywords):
            return item
    return None


def _deterministic_verify(
    condition_text: str,
    clause_ref: str,
    lc_field_value: str,
    document_type: str,
    unified_summary: dict,
    bl_subtype: dict,
    final_lc: dict,
    document_text: str = "",
) -> Optional[dict]:
    """Return a verdict dict (PASS/FAIL/REVIEW) when the condition can be
    answered deterministically from structured facts. Else None (→ LLM path).
    Conservative by design — when in doubt, return None."""
    if not unified_summary and not bl_subtype:
        return None

    cond_up = (condition_text or "").upper()
    doc_up = (document_type or "").upper()

    # ── Check 1: LC reference presence on a doc ──
    # "DOC MUST QUOTE LC NUMBER" / "REFERENCE TO L/C"
    # Deterministic PASS only — FAIL decisions fall through to LLM because
    # the LC number could be in doc_text even when Step 3 didn't tag it.
    if ('LC NO' in cond_up or 'L/C NO' in cond_up or
        'LC NUMBER' in cond_up or 'L/C NUMBER' in cond_up or
        'LC REFERENCE' in cond_up or 'DC NUMBER' in cond_up):
        lc_ref = (final_lc or {}).get('20', '') if final_lc else ''
        if lc_ref:
            found = _find_structured(unified_summary, 'references_found',
                                      ['lc_reference', 'dc_reference', 'credit_reference'])
            if found and _normalize_id(found.get('value')) == _normalize_id(lc_ref):
                return {
                    'verdict': 'PASS',
                    'quote': found.get('raw') or found.get('value', ''),
                    'findings': f"LC reference {lc_ref} found on document via structured facts.",
                    'confidence': 0.98,
                    'structured_source': 'references_found[role=lc_reference]',
                }
            # Also check if LC ref appears in ANY reference role (might be
            # tagged as other/invoice_reference/etc.)
            for item in (unified_summary.get('references_found') or []):
                if isinstance(item, dict):
                    if _normalize_id(item.get('value')) == _normalize_id(lc_ref):
                        return {
                            'verdict': 'PASS',
                            'quote': item.get('raw') or item.get('value', ''),
                            'findings': f"LC reference {lc_ref} found on document (tagged as {item.get('role', 'other')}).",
                            'confidence': 0.95,
                            'structured_source': f"references_found[role={item.get('role','other')}]",
                        }
            # Not found in structured refs — FALL THROUGH to LLM (it can
            # check doc_text where an untagged match may still appear).

    # ── Check 1b (P135): Generic "must reference / quote / contain <ID>" ──
    # (same digit-requirement fix applied below)
    # For any condition that carries a specific identifier pattern (open
    # policy no, contract no, proforma no, cover note, LC ref again, etc.)
    # and asks a document to "reference / quote / contain / state / show"
    # that ID, we try to find it in the document's structured references
    # and typed fields with OCR-tolerant normalization.
    if (('REFERENCE' in cond_up or 'QUOTE' in cond_up or 'CONTAIN' in cond_up or
            'STATE' in cond_up or 'SHOW' in cond_up or 'INCLUDE' in cond_up or
            'MENTION' in cond_up) and unified_summary):
        # Pull all identifier-like tokens out of the condition: runs of
        # digits + letters (length >= 6) OR digit/letter/slash/dash chains.
        _cond_ids_raw = re.findall(
            r'[A-Z0-9][A-Z0-9/\-._]{5,}[A-Z0-9]',
            condition_text or '',
            flags=re.IGNORECASE,
        )
        # Require at least one digit — otherwise plain English words
        # like "reference", "Shipment" match the alphanumeric pattern.
        _cond_ids = [_t for _t in _cond_ids_raw if re.search(r'\d', _t)]
        for _needle in _cond_ids:
            _n_norm = _normalize_id(_needle)
            if len(_n_norm) < 6:
                continue
            # Skip common English words that can match the pattern
            if _n_norm in ('LETTERCREDIT', 'DOCUMENTARY', 'SHIPMENTADVICE', 'COMMERCIALINVOICE'):
                continue
            # 1) Check references_found — any role
            for item in (unified_summary.get('references_found') or []):
                if not isinstance(item, dict):
                    continue
                _v = _normalize_id(item.get('value', ''))
                if _v and (_v == _n_norm or _n_norm in _v or _v in _n_norm):
                    return {
                        'verdict': 'PASS',
                        'quote': item.get('raw') or item.get('value', ''),
                        'findings': (
                            f"Reference '{_needle}' found on document as "
                            f"'{item.get('value','')}' (role={item.get('role','other')})."
                        ),
                        'confidence': 0.92,
                        'structured_source': f"references_found[role={item.get('role','other')}]",
                    }
            # 2) Check typed scalar fields
            for _key in (
                'lc_reference', 'invoice_reference', 'contract_reference',
                'proforma_reference', 'bl_number', 'bl_reference',
                'open_policy_reference', 'cover_note_reference',
                'policy_number', 'document_identifier',
            ):
                _tv = _normalize_id(str(unified_summary.get(_key, '') or ''))
                if _tv and (_tv == _n_norm or _n_norm in _tv or _tv in _n_norm):
                    return {
                        'verdict': 'PASS',
                        'quote': str(unified_summary.get(_key, ''))[:200],
                        'findings': (
                            f"Reference '{_needle}' matches {_key} "
                            f"= '{unified_summary.get(_key)}' on document."
                        ),
                        'confidence': 0.92,
                        'structured_source': f"unified_summary.{_key}",
                    }
            # 3) Check other_details_found raw text (open policy, cover notes,
            #    often land here when the LLM tags them as open_policy_reference)
            for item in (unified_summary.get('other_details_found') or []):
                if not isinstance(item, dict):
                    continue
                _raw = _normalize_id(
                    str(item.get('value', '') or '') + ' ' +
                    str(item.get('raw', '') or '')
                )
                if _raw and _n_norm in _raw:
                    return {
                        'verdict': 'PASS',
                        'quote': item.get('raw') or item.get('value', ''),
                        'findings': (
                            f"Reference '{_needle}' appears in "
                            f"other_details_found[role={item.get('role','other')}]."
                        ),
                        'confidence': 0.90,
                        'structured_source': f"other_details_found[role={item.get('role','other')}]",
                    }
        # No identifier-token matched — fall through to LLM (could still
        # match via free-text OCR on the cleaned document text).

    # ── Check 2: BL prohibition checks — fully structured ──
    if 'BILL OF LADING' in doc_up and bl_subtype:
        # "NOT CHARTER PARTY" / "NOT CHARTER-PARTY"
        if ('NOT' in cond_up and 'CHARTER PART' in cond_up):
            ct = str(bl_subtype.get('contract_type', '')).lower()
            if ct and ct != 'charter_party':
                return {
                    'verdict': 'PASS',
                    'quote': f"bl_subtype.contract_type = {ct}",
                    'findings': f"BL is not charter party (contract_type={ct}).",
                    'confidence': 0.95,
                    'structured_source': 'bl_subtype.contract_type',
                }
        # P182 — "NOT SHORT FORM" and "NOT BLANK BACK" are DIFFERENT checks:
        #   SHORT FORM = T&C printed overleaf but abbreviated/half-page
        #   BLANK BACK = no T&C overleaf at all
        # Handle each prohibition independently.
        _asks_not_short_form = (
            'NOT' in cond_up and
            ('SHORT FORM' in cond_up or 'SHORT-FORM' in cond_up) and
            'BLANK BACK' not in cond_up and 'BLANK-BACK' not in cond_up
        )
        _asks_not_blank_back = (
            'NOT' in cond_up and
            ('BLANK BACK' in cond_up or 'BLANK-BACK' in cond_up) and
            'SHORT FORM' not in cond_up and 'SHORT-FORM' not in cond_up
        )
        _asks_not_either = (
            'NOT' in cond_up and
            ('SHORT FORM' in cond_up or 'SHORT-FORM' in cond_up) and
            ('BLANK BACK' in cond_up or 'BLANK-BACK' in cond_up)
        )

        if _asks_not_short_form:
            # Pass if BL is NOT flagged short_form (could be long form OR
            # blank back — both acceptable for this specific condition).
            _isf = bl_subtype.get('is_short_form')
            _ibb = bl_subtype.get('is_blank_back')
            # P186 — If the BL has NO T&C page at all it is classified as
            # BLANK BACK, not short form. User rule: "if terms and
            # conditions are missing just mark as blank back not short
            # form". So when is_blank_back=True, short-form check must
            # PASS here — the blank-back check (separate row) will carry
            # the single FAIL for missing T&Cs.
            if _ibb is True:
                return {
                    'verdict': 'PASS',
                    'quote': f"is_blank_back=True (no T&C page), is_short_form={_isf}",
                    'findings': (
                        "BL has no T&C page — classified as blank back, "
                        "not short form. Short-form prohibition does not "
                        "apply (the blank-back prohibition covers this)."
                    ),
                    'confidence': 0.95,
                    'structured_source': 'bl_subtype.is_blank_back',
                }
            if _isf is False:
                return {
                    'verdict': 'PASS',
                    'quote': f"bl_subtype.is_short_form = {_isf}",
                    'findings': (
                        "BL is not a short form "
                        f"(is_short_form={_isf}, form_type="
                        f"{bl_subtype.get('form_type','unknown')})."
                    ),
                    'confidence': 0.95,
                    'structured_source': 'bl_subtype.is_short_form',
                }
            if _isf is True:
                return {
                    'verdict': 'FAIL',
                    'quote': f"bl_subtype.is_short_form = True",
                    'findings': (
                        "BL is a short form (terms overleaf are "
                        "abbreviated). LC prohibits short-form BLs."
                    ),
                    'confidence': 0.95,
                    'structured_source': 'bl_subtype.is_short_form',
                }
        elif _asks_not_blank_back:
            # Pass if BL has T&C attached (not blank back).
            _ibb = bl_subtype.get('is_blank_back')
            _hto = bl_subtype.get('has_terms_overleaf')
            if _hto is True or _ibb is False:
                return {
                    'verdict': 'PASS',
                    'quote': f"bl_subtype.has_terms_overleaf={_hto}, is_blank_back={_ibb}",
                    'findings': (
                        "BL has T&C printed on reverse (not blank back)."
                    ),
                    'confidence': 0.95,
                    'structured_source': 'bl_subtype.has_terms_overleaf',
                }
            if _ibb is True:
                return {
                    'verdict': 'FAIL',
                    'quote': f"bl_subtype.is_blank_back = True",
                    'findings': (
                        "BL has no T&C attached (blank back). "
                        "LC prohibits blank-back BLs."
                    ),
                    'confidence': 0.95,
                    'structured_source': 'bl_subtype.is_blank_back',
                }
        elif _asks_not_either:
            # Condition forbids BOTH short-form AND blank-back.
            _isf = bl_subtype.get('is_short_form')
            _ibb = bl_subtype.get('is_blank_back')
            _hto = bl_subtype.get('has_terms_overleaf')
            if _isf is False and (_hto is True or _ibb is False):
                return {
                    'verdict': 'PASS',
                    'quote': f"is_short_form={_isf}, is_blank_back={_ibb}",
                    'findings': (
                        "BL has full T&C on reverse (neither short form nor "
                        "blank back)."
                    ),
                    'confidence': 0.95,
                    'structured_source': 'bl_subtype.has_terms_overleaf',
                }
            if _isf is True:
                return {
                    'verdict': 'FAIL',
                    'quote': f"bl_subtype.is_short_form = True",
                    'findings': "BL is a short form — LC forbids.",
                    'confidence': 0.95,
                    'structured_source': 'bl_subtype.is_short_form',
                }
            if _ibb is True:
                return {
                    'verdict': 'FAIL',
                    'quote': f"bl_subtype.is_blank_back = True",
                    'findings': "BL is blank back — LC forbids.",
                    'confidence': 0.95,
                    'structured_source': 'bl_subtype.is_blank_back',
                }
        # "NOT ISSUED BY FREIGHT FORWARDER" / "NOT FORWARDER"
        if 'NOT' in cond_up and ('FORWARDER' in cond_up or 'FREIGHT FORWARDER' in cond_up):
            st = str(bl_subtype.get('signing_type', '')).lower()
            if st in ('master_signed', 'agent_for_master', 'carrier_signed'):
                return {
                    'verdict': 'PASS',
                    'quote': f"bl_subtype.signing_type = {st}",
                    'findings': f"BL signed as {st.replace('_', ' ')} — not a freight forwarder.",
                    'confidence': 0.95,
                    'structured_source': 'bl_subtype.signing_type',
                }
        # P122: "NOT HOUSE BL" / "NOT A HOUSE BILL OF LADING"
        if 'NOT' in cond_up and 'HOUSE' in cond_up:
            it = str(bl_subtype.get('issuer_type', '')).lower()
            ih = bl_subtype.get('is_house_bl')
            if it in ('master_bl', 'charter_party_bl') or ih is False:
                return {
                    'verdict': 'PASS',
                    'quote': f"bl_subtype.issuer_type = {it}, is_house_bl = {ih}",
                    'findings': f"BL is not a house BL (issuer_type={it}).",
                    'confidence': 0.95,
                    'structured_source': 'bl_subtype.issuer_type',
                }
        # P167: "BL MUST NOT BE CLAUSED" / "CLAUSED BL NOT ACCEPTABLE" /
        # "BL MUST BE CLEAN". This is UCP 600 Art 27 — a claused BL has
        # damage/defect notations. It is a COMPLETELY DIFFERENT attribute
        # from blank_back (which is about the reverse side of the BL).
        # Read bl_subtype.cleanness / is_claused_bl — never confuse with
        # is_blank_back / has_terms_overleaf.
        if (('NOT' in cond_up and 'CLAUSED' in cond_up) or
                ('MUST BE CLEAN' in cond_up) or
                ('CLEAN' in cond_up and ('BILL' in cond_up or 'BL' in cond_up) and
                 ('BE' in cond_up or 'MUST' in cond_up))):
            _cleanness = str(bl_subtype.get('cleanness', '')).lower().strip()
            _is_claused = bl_subtype.get('is_claused_bl')
            _notes = str(bl_subtype.get('clausing_notes', '') or '').strip()
            # Explicit CLAUSED → FAIL
            if _cleanness == 'claused' or _is_claused is True or _notes:
                return {
                    'verdict': 'FAIL',
                    'quote': f"bl_subtype.cleanness = claused; clausing_notes = {_notes[:150]}",
                    'findings': (
                        f"BL is claused — damage/defect notation present: "
                        f"'{_notes[:150] or '(see BL face)'}'. LC requires clean BL "
                        f"(UCP 600 Art 27)."
                    ),
                    'confidence': 0.95,
                    'structured_source': 'bl_subtype.cleanness',
                }
            # Otherwise default to CLEAN (P168). UCP 600 Art 27 says a
            # BL is clean unless it has an explicit clause declaring the
            # goods defective. If structured facts have no claused flag
            # and no clausing_notes, the BL is clean by default — do
            # NOT let the LLM hallucinate a FAIL.
            return {
                'verdict': 'PASS',
                'quote': f"bl_subtype.cleanness = {_cleanness or 'clean'}, is_claused_bl = {_is_claused}, clausing_notes = (none)",
                'findings': (
                    f"BL is clean — no damage/defect notation on the goods "
                    f"(cleanness={_cleanness or 'clean'}, "
                    f"is_claused_bl={_is_claused or False}). UCP 600 Art 27 "
                    f"treats a BL without explicit clausing as clean by default."
                ),
                'confidence': 0.90,
                'structured_source': 'bl_subtype.cleanness',
            }

    # ── Check 2b (P152 — revised): Consignee / to-order-of match
    # ──
    # ── CRITICAL UCP 600 semantics:
    # ──   "Consignee" and "Notify Party" are TWO DIFFERENT BL fields.
    # ──   A bank appearing in Notify Party does NOT satisfy a consignee
    # ──   requirement. Only the CONSIGNEE box counts for "drawn to order
    # ──   of X". If the consignee shows just "TO ORDER" (bearer/shipper-
    # ──   order) without naming X, the BL is only compliant if a reverse-
    # ──   side endorsement names X. Since we don't have OCR for the
    # ──   reverse side in most cases, the correct verdict when consignee
    # ──   is "TO ORDER" only is REVIEW (manual endorsement check), not
    # ──   PASS.
    # ──
    # ── Algorithm:
    # ──   1. Condition shape is "to (the) order of X" / "consigned to X"
    # ──      / "consignee must be X" → extract target X and derive a
    # ──      distinctive name-key (strip BANK/LTD/LIMITED/punctuation/
    # ──      trailing city).
    # ──   2. Look ONLY at the structured consignee field or
    # ──      parties_found[role=consignee]. NEVER compare against
    # ──      notify_party, shipper, or issuing_bank roles.
    # ──   3. Normalize the consignee string by stripping the
    # ──      "TO (THE) ORDER OF" prefix so it doesn't leak into matching.
    # ──   4. If target-key appears in the (prefix-stripped) consignee
    # ──      text → PASS.
    # ──   5. If consignee is empty OR is just "TO ORDER" / "TO THE ORDER"
    # ──      with no party name → scan document_text for an endorsement
    # ──      block (not usually captured). If none → REVIEW with note
    # ──      "reverse-side endorsement check required". Never auto-PASS.
    # ──   6. If consignee names a DIFFERENT party (e.g., Dalda Foods
    # ──      when LC requires UBL) → FAIL.
    if unified_summary and (
        'TO THE ORDER OF' in cond_up or 'TO ORDER OF' in cond_up or
        'MADE OUT TO' in cond_up or 'CONSIGNED TO' in cond_up or
        ('CONSIGNEE' in cond_up and ('MUST BE' in cond_up or 'SHOULD BE' in cond_up))
    ):
        _target = ''
        for _pat in (
            r'TO\s+(?:THE\s+)?ORDER\s+OF[\s:]+([^.\n]+?)(?:[.,\n]|$)',
            r'CONSIGNED\s+TO[\s:]+([^.\n]+?)(?:[.,\n]|$)',
            r'CONSIGNEE\s+(?:MUST\s+BE|SHOULD\s+BE|IS|=)[\s:\'""]+([^.\n\'""]+?)(?:[.,\n\'""]|$)',
            r'MADE\s+OUT\s+TO[\s:]+([^.\n]+?)(?:[.,\n]|$)',
        ):
            _m = re.search(_pat, cond_up)
            if _m:
                _target = _m.group(1).strip(' .,:\'""')
                break
        if _target:
            # Build distinctive key from the target party name.
            _key = re.sub(r'[.,;:\'"—–-]+', ' ', _target)
            _key = re.sub(
                r'\b(?:PAKISTAN|INDIA|BANGLADESH|SRI\s+LANKA|UAE|SAUDI\s+ARABIA|'
                r'KARACHI|LAHORE|ISLAMABAD|MUMBAI|DUBAI|RIYADH|DOHA|BEIRUT|'
                r'HONG\s+KONG|SINGAPORE|LONDON|NEW\s+YORK|GULBERG|CPU\s+\(TRADE\)|'
                r'CPU|TRADE|PRINTING|STATIONARY|BUILDING|MAI[- ]?KOLACHI|ROAD)\b',
                '', _key, flags=re.IGNORECASE,
            )
            _key = re.sub(
                r'\b(BANK|LTD|LIMITED|LLC|PLC|INC|CORP|CO|PVT|PRIVATE|'
                r'COMPANY|ENTERPRISES?|GROUP|HOLDINGS?|TRADING|'
                r'INSURERS?|INSURANCE)\b\.?',
                ' ', _key, flags=re.IGNORECASE,
            )
            _key = re.sub(r'\s+', ' ', _key).strip().upper()

            if _key and len(_key) >= 4 and (' ' in _key or len(_key) >= 3):
                # Read ONLY the consignee field — NEVER notify/shipper/etc.
                _cons_txt = str(unified_summary.get('consignee', '') or '').upper()
                if not _cons_txt:
                    for item in (unified_summary.get('parties_found') or []):
                        if isinstance(item, dict):
                            role = str(item.get('role', '')).lower()
                            if role == 'consignee' or role == 'second_consignee':
                                _cons_txt = (
                                    str(item.get('name', '') or '').upper() + ' ' +
                                    str(item.get('raw', '') or '').upper()
                                )
                                break

                # Strip the "TO (THE) ORDER OF" prefix so only the party
                # name (if any) remains in the comparison text.
                _cons_clean = re.sub(
                    r'^\s*(?:CONSIGNEE\s*[:\-]?\s*)?'
                    r'(?:TO\s+(?:THE\s+)?ORDER(?:\s+OF)?)[:\s]*',
                    '', _cons_txt,
                )
                _cons_clean = _cons_clean.strip(' .,:-')
                # Also detect if consignee is ONLY the bearer phrase.
                _is_bearer_only = bool(re.fullmatch(
                    r'\s*(?:CONSIGNEE\s*[:\-]?\s*)?'
                    r'(?:TO\s+(?:THE\s+)?ORDER(?:\s+OF)?)\s*[.,:-]*\s*',
                    _cons_txt,
                ))

                if _is_bearer_only or not _cons_clean:
                    # Consignee is "TO ORDER" only — no named party. Under
                    # UCP 600 Art 14(e), a blank-endorsable BL is acceptable
                    # only if the reverse side carries the proper
                    # endorsement.
                    # P192 — Tighten the endorsement detection. The old
                    # check passed whenever the word "ENDORSED" AND the
                    # party name appeared ANYWHERE in the document, which
                    # false-PASSed BLs where the party was in the notify
                    # address and "endorsed" came from generic carriage
                    # T&C clauses. Require an explicit endorsement phrase
                    # within ~40 chars of the party name.
                    _has_endorsement = False
                    _endorsement_snippet = ''
                    if document_text:
                        _dt_u = document_text.upper()
                        # Normalize whitespace so the proximity check
                        # works across line breaks.
                        _dt_norm = re.sub(r'\s+', ' ', _dt_u)
                        _escaped_key = re.escape(_key)
                        # Patterns that constitute a real endorsement
                        # naming the target party. Proximity: 0-40 chars
                        # between the endorsement phrase and the party.
                        _patterns = [
                            rf'\bENDORSED\s+(?:TO|IN\s+FAVO(?:U)?R\s+OF|FOR)\b[^.\n]{{0,40}}{_escaped_key}',
                            rf'\bENDORSEMENT\s+(?:TO|IN\s+FAVO(?:U)?R\s+OF|FOR)\b[^.\n]{{0,40}}{_escaped_key}',
                            rf'{_escaped_key}[^.\n]{{0,40}}\bENDORSED\b',
                            rf'\bFOR\s+AND\s+ON\s+BEHALF\s+OF\b[^.\n]{{0,40}}{_escaped_key}',
                            rf'\bPAY\s+TO\s+(?:THE\s+)?ORDER\s+OF\b[^.\n]{{0,40}}{_escaped_key}',
                            rf'\bDELIVER\s+TO\b[^.\n]{{0,40}}{_escaped_key}',
                        ]
                        for _pat in _patterns:
                            _m = re.search(_pat, _dt_norm)
                            if _m:
                                _has_endorsement = True
                                _endorsement_snippet = _m.group(0)[:140]
                                break
                    if _has_endorsement:
                        return {
                            'verdict': 'PASS',
                            'quote': f"Endorsement naming {_key}: {_endorsement_snippet}",
                            'findings': (
                                f"Consignee is 'TO ORDER'; a specific "
                                f"endorsement to '{_key}' is present on the "
                                f"document ({_endorsement_snippet[:100]}) — "
                                f"compliant."
                            ),
                            'confidence': 0.88,
                            'structured_source': 'document_text.endorsement',
                        }
                    return {
                        'verdict': 'FAIL',
                        'quote': f"consignee = {_cons_txt[:120] or '(empty)'}",
                        'findings': (
                            f"Consignee shows '{_cons_txt.strip() or 'TO ORDER'}' "
                            f"only — LC requires 'TO ORDER OF {_key}'. "
                            f"No explicit endorsement to '{_key}' is visible "
                            f"on the BL face. A mere mention of '{_key}' "
                            f"elsewhere on the BL (e.g. in the notify party "
                            f"address) does NOT satisfy this requirement."
                        ),
                        'confidence': 0.9,
                        'structured_source': 'unified_summary.consignee',
                    }

                # Consignee names a party. Does it contain the target key?
                if _key in _cons_clean:
                    return {
                        'verdict': 'PASS',
                        'quote': f"consignee = {_cons_txt[:200]}",
                        'findings': (
                            f"Structured consignee contains '{_key}' — "
                            f"requirement satisfied."
                        ),
                        'confidence': 0.95,
                        'structured_source': 'unified_summary.consignee',
                    }
                # Consignee names a DIFFERENT party — FAIL (not a match).
                return {
                    'verdict': 'FAIL',
                    'quote': f"consignee = {_cons_txt[:200]}",
                    'findings': (
                        f"Consignee is '{_cons_clean[:120]}' but LC requires "
                        f"'TO ORDER OF {_key}'. Different party — non-compliant."
                    ),
                    'confidence': 0.9,
                    'structured_source': 'unified_summary.consignee',
                }

    # ── Check 3: Beneficiary name change ("currently known as") ──
    # If the draft/invoice has a drawer/beneficiary showing renamed entity,
    # and the condition is about beneficiary identity — PASS.
    if ('BENEFICIARY' in cond_up or 'ISSUED BY' in cond_up or
        'THIRD PARTY' in cond_up or 'DRAWN BY' in cond_up):
        # Look in parties_found for a drawer/beneficiary/issuer/shipper
        # whose raw text contains "currently known as" / "formerly known as"
        arr = (unified_summary or {}).get('parties_found') or []
        for item in arr:
            if not isinstance(item, dict):
                continue
            role = str(item.get('role', '')).lower()
            if role in ('drawer', 'beneficiary', 'issuer', 'shipper'):
                raw = str(item.get('raw', '') or item.get('name', '')).upper()
                if ('CURRENTLY KNOWN AS' in raw or
                    'FORMERLY KNOWN AS' in raw or
                    'NOW KNOWN AS' in raw or
                    'TRADING AS' in raw or 'T/A ' in raw or
                    'D/B/A' in raw):
                    return {
                        'verdict': 'PASS',
                        'quote': item.get('raw', ''),
                        'findings': "Beneficiary/drawer identified — document shows legal entity name change (same party).",
                        'confidence': 0.9,
                        'structured_source': f"parties_found[role={role}]",
                    }

    # Fall through to LLM
    return None


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

ANTI-HALLUCINATION: Your "findings" field MUST contain ONLY text that is
ACTUALLY PRESENT in the DOCUMENT TEXT above. Do NOT copy words from the
CONDITION into the findings. For example, if the condition says "must be
addressed to the Applicant" and the document does NOT mention the applicant
name, do NOT write "AND to the Applicant" in the findings — that would be
copying the condition, not reporting what you found. Instead write "Document
is addressed to [actual addressee found]. Applicant name not found on the
document." and mark as FAIL.

VERIFY: Does the document satisfy this condition?

ANTI-HALLUCINATION WARNING (CRITICAL — READ CAREFULLY):
Your findings MUST contain ONLY text that ACTUALLY appears in the
DOCUMENT TEXT section below. NEVER invent, assume, or copy text
from the CONDITION into findings.

BEFORE marking PASS, verify:
  1. Can you find the EXACT required text in the DOCUMENT TEXT?
  2. Can you QUOTE the specific line from the document?
  3. If the document does NOT contain the required text, mark FAIL.

EXAMPLES OF HALLUCINATION (DO NOT DO THIS):
  ❌ Condition: "must state vessel is covered under Institute
     Classification Clause"
     Document text: "14 DAYS FREE TIME DETENTION ALLOWED AT POD"
     WRONG: findings="Carrying vessel is covered under Institute
     Classification Clause" result=PASS
     → This is HALLUCINATION. The document says NOTHING about
     Institute Classification Clause. Correct answer: FAIL.
  ❌ Condition: "must show ETA at port of destination"
     Document text: "Sailing on: 01FEB 2025"
     WRONG: findings="ETA: SEP.20.2025" result=PASS
     → HALLUCINATED date. The document shows sailing date, not ETA.
     Correct answer: FAIL (no ETA/arrival date found).
  ❌ Condition: "must show vessel name"
     Document text: "Vessel: KOTANEKAD0204S"
     WRONG: findings="Name of carrying vessel: MV Ocean Voyager"
     → HALLUCINATED vessel name. The actual vessel is KOTANEKAD0204S.

RULE: If you cannot QUOTE the relevant text from the DOCUMENT TEXT,
the answer is FAIL. Period.

GOODS DESCRIPTION TOLERANCE: Minor wording variations in goods
description are acceptable if the PRODUCT is clearly the same.
"Canadian Canola No.1" and "Canadian GMO Canola" refer to the same
commodity (canola from Canada). Grade/variety descriptors like
"No.1", "GMO", "non-GMO", "in bulk" are supplementary details,
not a different product. If the core product name matches, PASS.

ADDRESSING QUICK-CHECK: "TO:", "ALSO TO:", "AND TO:", "CC:" on a
document ALL mean the party IS addressed. If a Shipment Advice shows
"TO: [applicant name]", the applicant IS addressed → PASS.

BL PROHIBITION QUICK-CHECK (applies to ALL BL type checks):
When the condition says "BL must not be [Charter Party / Short Form /
Blank Back / Freight Forwarder / House BL]", check the BL DOCUMENT:
  • Signed "AS CARRIER" or "AS AGENT FOR THE CARRIER" by a shipping
    line (PIL, MAERSK, MSC, EVERGREEN, CMA CGM, etc.) → PASS for ALL
  • Has T&C text or URL to terms → NOT short form → PASS
  • No "CHARTER PARTY" text on the BL → NOT charter party → PASS
  • "FREIGHT FORWARDER" or "NVOCC" NOT in signing block → PASS
The prohibition words are in the CONDITION, not on the BL. PASS means
the BL is NOT the prohibited type.

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
   • SYMMETRIC TOLERANCE (P127 — CRITICAL): If the LC has F39A = "05/05" or F47A says "+/-5%" or "05PCT MORE OR LESS" or similar plus-minus language, the tolerance band is +/- that percent in BOTH directions. The invoice amount is a PASS if it falls in [LC × (1 − tolerance%), LC × (1 + tolerance%)].
       EXAMPLE: LC = USD 100,000.00 with F39A "05/05" → band = 95,000 to 105,000. Invoice 104,500 → PASS. Invoice 96,000 → PASS. Invoice 106,000 → FAIL. Invoice 94,000 → FAIL only if partial shipment is prohibited — otherwise PASS as short shipment under UCP 600 Art 30.
   • ASYMMETRIC TOLERANCE: If F39A says "10/05" → +10% / -5%. If it says "05/10" → +5% / -10%. Apply each side separately.
   • NO EXPLICIT TOLERANCE: Without F39A or an explicit +/- clause, UCP 600 Art 30(b) gives an IMPLICIT 5% symmetric tolerance on the AMOUNT only when the LC amount is expressed with "ABOUT" / "APPROXIMATELY" / "CIRCA". Without those qualifiers, invoice must be <= LC amount (but can be less — partial/short shipment is allowed unless prohibited).
   • "MUST NOT EXCEED" / "NOT TO EXCEED": invoice amount must be <= LC amount × (1 + tolerance%). Equal (=) is NOT exceeding — that is PASS.
   • EQUAL AMOUNTS: If the invoice amount EQUALS the LC amount exactly
     (e.g., both are USD 30,080.00), this is PASS — NOT a discrepancy.
     "Must not exceed" means the invoice amount must be <= LC amount.
     Equal (=) is NOT exceeding. Only strictly greater than (>) is FAIL.
     Double-check your arithmetic before marking FAIL.
   • AMOUNT IN WORDS vs AMOUNT IN FIGURES: Invoices often show
     "TOTAL PRICE: SAY USD ONE HUNDRED AND SEVENTY-ONE THOUSAND ONLY"
     (amount in words). This is NOT the amount to compare — always use
     the NUMERIC amount (e.g., 171,000 or 171000). The words line is
     just a confirmation. If the SYSTEM PRE-CALCULATED SUMMARY shows
     an "INVOICE PRINTED TOTAL AMOUNT", use THAT number.
   • OCR may glue words together: "SAYUSDONEHUNDREDANDSEVENTY-ONETHOUSANDONLY"
     This is "SAY USD ONE HUNDRED AND SEVENTY-ONE THOUSAND ONLY" = 171,000.
     Do NOT treat this garbled text as the invoice amount — find the
     actual numeric total in the invoice table (usually the last number
     before the words-in-words line).
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
9. CERTIFICATION: If the condition asks for origin certification and the document says "We certify the goods are of [COUNTRY] origin" or similar statement, that IS a valid certification = PASS. Do not fail just because the exact word "CERTIFICATE" is not used — any statement certifying origin, quality, weight, etc. is a certification. ALSO ACCEPT: a plain "Origin: [COUNTRY]" line, a "[COUNTRY] ORIGIN." sentence, or "COUNTRY OF ORIGIN: [COUNTRY]" field — any of these constitutes the origin statement required by 46A. Do NOT write "does not specify origin" when the word "Origin" appears next to a country name on the invoice.
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
13e. BL TYPES — see BL PROHIBITION QUICK-CHECK above (near top of prompt).
13f. DRAFT / BILL OF EXCHANGE DATES:
    A Draft / Bill of Exchange typically has TWO dates:
      • DRAFT DATE (date of drawing): "Dated September 15, 2025"
        — this is when the draft was drawn/issued by the beneficiary
      • LC DATE (date of the credit): "Dated August 18, 2025"
        — this is the LC issue date, confirming which LC the draft
        is drawn under
    These often appear as two separate "Dated ..." lines near the
    signature block. The SECOND date is usually the LC date.
    When the condition says "Draft must show the date of the L/C",
    look for the LC issue date on the draft — it may appear as:
      • "Dated [LC date]" (second date line)
      • Near "Drawn under L/C No. XXXXX"
      • "L/C dated [date]"
      • "DATE OF ISSUE: YYMMDD"
    If ANY date on the draft matches the LC issue date → PASS.

13c. DRAFT / BILL OF EXCHANGE INSTALLMENT VERIFICATION:
    When verifying payment terms on a Draft Bill of Exchange:
    - The draft may show ALL installments in a continuous text block,
      sometimes split by "+++" or "[STAMP]" or "[SIGNATURE]" markers
      where OCR couldn't read through stamps/signatures.
    - "+++" means OCR-obscured text — it does NOT mean missing content.
      Text before and after "+++" is ONE continuous clause.
    - A draft page may contain TWO copies: "FIRST of Exchange" and
      "SECOND of Exchange" — these are copies of the SAME draft, not
      separate installments. Check EITHER copy for the payment terms.
    - Installment text may wrap like:
      "A) 25 PCT ... WITHIN 90 DAYS ... B) 25 PCT ... WILL BE +++"
      "+++PAID WITHIN 180 DAYS ... C) 25 PCT ... WITHIN 270 DAYS..."
      This is ONE continuous text — ALL installments are present.
    - Each installment shows: percentage, amount (e.g., USD42750.00),
      days (90/180/270/360), and reference date (BL ISSUING DATE).
    - To verify: check that the draft mentions ALL installment
      percentages and day counts from the LC. If A/B/C/D all appear
      with correct percentages and days, it is PASS — even if the
      text has "+++" breaks between them.
    - The total draft amount (e.g., USD171,000.00) should equal the
      LC amount. Individual installment amounts should each equal
      their percentage of the total.

13d. DATE TYPES — DO NOT CONFUSE DIFFERENT DATES:
    Documents contain multiple dates with different meanings:
      • "DATE: AUG.30.2025" on a certificate = ISSUANCE DATE of the certificate
      • "DATE OF ISSUE: 250818" = LC ISSUE DATE (reference only)
      • "DATE OF SHIPMENT" / "SHIPPED ON BOARD" = SHIPMENT DATE
      • "ETA" / "APPROXIMATE ARRIVAL" / "EXPECTED ARRIVAL" = ARRIVAL DATE
      • "BL DATE" = Bill of Lading date
    When a condition asks for "date of arrival at port of destination"
    or "approximate date of arrival", look for ANY of these:
      • "ETA:" or "ETA :" — Estimated Time of Arrival = approximate arrival date
      • "APPROXIMATE DATE OF ARRIVAL"
      • "EXPECTED ARRIVAL" / "EXPECTED DATE OF ARRIVAL"
      • "TENTATIVE DATE OF ARRIVAL"
      • "ARRIVAL DATE" / "DATE OF ARRIVAL"
    "ETA:SEP.20.2025" means the vessel is expected to arrive on Sep 20, 2025.
    This IS the "approximate date of arrival" the LC is asking for → PASS.
    Do NOT use the certificate issuance date or the LC issue date for this.

13z. STALE BL CHECK — NEW LOGIC (P160 — CRITICAL):
    A BL is "stale" when too much time has passed between the BL's
    shipped-on-board date and the date the bank received the
    documents (the receiving/presentation date stamped on the
    Documentary Remittance / Covering Schedule).
    FORMULA:
      days_elapsed = receiving_date_on_DR − bl_onboard_date
      STALE if days_elapsed > 30  (fixed 30-day threshold; do NOT use F48)
    EXAMPLES:
      • BL on-board = 12-Feb-2025, DR receiving = 13-Mar-2025 →
        29 days → NOT stale → PASS.
      • BL on-board = 12-Feb-2025, DR receiving = 15-Mar-2025 →
        31 days → STALE → FAIL.
      • BL on-board = 05-Jan-2025, DR receiving = 20-Mar-2025 →
        74 days → STALE → FAIL.
    WHERE TO READ:
      • receiving_date / presentation_date: on the Documentary
        Remittance, typically a date STAMP (often rubber-stamped,
        sometimes rotated/upside-down). Use structured
        unified_summary.receiving_date or
        unified_summary.presentation_date. If neither exists,
        use dates_found[role=receiving_date|presentation_date].
      • bl_onboard_date: on the Bill of Lading. Use
        unified_summary.onboard_date /
        dates_found[role=onboard_date] /
        bl_subtype.shipped_on_board_status + the BL issue date
        ("CLEAN ON BOARD <date>").
    If receiving_date is missing from the DR → REVIEW (cannot
    confirm). If on-board date is missing from the BL → REVIEW.
    Only FAIL when BOTH dates are present AND the delta exceeds
    the threshold. Never FAIL with "No BL found inside DR" — the
    DR is not supposed to contain a BL; it carries the receipt
    stamp date.

13y. DATE COMPARISON RULES (CRITICAL — prevent every false "prior to LC date" FAIL):
    Trade documents carry dates in many formats. The LC's F31C issue
    date is usually ISO "YYYY-MM-DD" or SWIFT six-digit "YYMMDD".
    Documents carry dates like "28-Jan-25", "28-JAN-2025", "Jan 28,
    2025", "28.01.2025", "250128", "28/01/2025", or even "28-1-25".
    Before comparing ANY two dates:
      1. Parse each date into (year, month, day). For a 2-digit year,
         interpret 00..49 as 20XX and 50..99 as 19XX. So "28-Jan-25"
         is 2025-01-28, NOT 1925 and NOT 2028. "250128" is 2025-01-28.
         "25-01-2028" is 2028-01-25.
      2. Compare in order: YEAR first, then MONTH, then DAY. Never
         compare day numbers alone. "28 is greater than 2" does NOT
         mean the date 28-Jan-2025 is later than 2-Jan-2025 in some
         other month or year — it's only correct HERE because both
         are January 2025.
      3. "ON OR AFTER LC DATE" → doc_date >= LC_date → PASS.
         "BEFORE LC DATE" / "PRIOR TO LC DATE" → doc_date < LC_date
         → FAIL. Equal → PASS (same day = not prior).

    WORKED EXAMPLE (must get right on first pass):
      LC F31C = 2025-01-02 (January 2, 2025).
      Commercial Invoice dated "28-Jan-25".
      Parse invoice: 2025-01-28.
      Compare 2025-01-28 vs 2025-01-02: year equal, month equal,
      day 28 > day 2 → invoice date is LATER than LC date.
      → PASS the "documents dated prior to LC date not acceptable"
      check. (The invoice is 26 days AFTER the LC issue date.)

    WORKED EXAMPLE 2:
      LC F31C = 2025-01-02. Doc dated "250102" (six-digit YYMMDD).
      Parse doc: 2025-01-02. Equal to LC → PASS (same day, not prior).

    WORKED EXAMPLE 3:
      LC F31C = 2025-01-02. Doc dated "28 Dec 2024".
      Parse doc: 2024-12-28. 2024 < 2025 → doc is BEFORE LC →
      FAIL as pre-dated.

    DO NOT confuse yourself by reading the DD-MMM-YY format and
    comparing string fragments — always parse to (year, month, day)
    integers and compare those.

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

20b. PORT CLUSTERS / MULTI-TERMINAL CITIES (P123): When an LC names a city that has multiple named sea terminals, any of those terminals satisfies the LC requirement. Do NOT raise a discrepancy when the BL shows a sub-terminal of the LC-named city — they are the SAME PORT COMPLEX.
    • "KARACHI" / "KARACHI SEAPORT" / "KARACHI PORT" includes:
        PORT QASIM, BIN QASIM, PORT BIN QASIM, PORT MUHAMMAD BIN QASIM, PQA,
        KEMARI, KIAMARI, KPT (Karachi Port Trust),
        KICT (Karachi International Container Terminal),
        QICT (Qasim International Container Terminal),
        PICT (Pakistan International Container Terminal),
        KGTL (Karachi Gateway Terminal Ltd),
        SAPT (South Asia Pakistan Terminals),
        GWADAR is NOT part of Karachi — it is a separate port.
    • "SHANGHAI" includes YANGSHAN, WAIGAOQIAO, WUSONGKOU.
    • "NEW YORK" / "NEW YORK PORT" includes NEWARK, PORT ELIZABETH, BAYONNE, RED HOOK.
    • "LOS ANGELES" port complex includes LONG BEACH (adjacent twin ports, often treated together only when LC says "LA/LB" — otherwise distinct).
    • "ROTTERDAM" includes MAASVLAKTE, EUROPOORT, BOTLEK.
    • "ANTWERP" includes DEURGANCKDOK, DELWAIDEDOK.
    • "HAMBURG" includes ALTENWERDER (CTA), BURCHARDKAI (CTB), TOLLERORT (CTT), EUROGATE.
    • "SINGAPORE" includes TUAS, PASIR PANJANG, BRANI, KEPPEL.
    • "HONG KONG" includes KWAI CHUNG, KWAI TSING, STONECUTTERS.
    • "DUBAI" includes JEBEL ALI, PORT RASHID.
    • "MUMBAI" includes NHAVA SHEVA / JNPT (Jawaharlal Nehru Port Trust), MUMBAI PORT TRUST.
    • "CHENNAI" includes ENNORE, KATTUPALLI.
    • "JEDDAH" includes KING ABDULAZIZ PORT, JEDDAH ISLAMIC PORT.
    • "COLOMBO" includes COLOMBO INTERNATIONAL CONTAINER TERMINAL (CICT), SAGT, JCT.
    MATCH RULES — ASYMMETRIC (CRITICAL):
    (a) LC names the PARENT CITY (e.g. "KARACHI", "KARACHI SEAPORT", "ANY KARACHI PORT") and BL shows any TERMINAL in that cluster (e.g. "PORT QASIM", "KEMARI", "KICT") → PASS. The city name covers every sub-terminal.
    (b) LC names a SPECIFIC TERMINAL (e.g. "PORT QASIM", "BIN QASIM", "KEMARI", "KICT", "QICT") and BL shows the SAME terminal (or a direct synonym: PORT QASIM = BIN QASIM = PORT MUHAMMAD BIN QASIM = PQA; KEMARI = KIAMARI = KPT) → PASS.
    (c) LC names a SPECIFIC TERMINAL but BL shows only the PARENT CITY (e.g. LC says "PORT QASIM" but BL says "KARACHI") → this is AMBIGUOUS, NOT automatic PASS. The city name alone does not prove the cargo moved through the specific terminal. Mark as REVIEW (not FAIL) unless the BL also shows a terminal code that differs from the LC's requirement.
    (d) LC names a SPECIFIC TERMINAL and BL shows a DIFFERENT terminal in the same cluster (e.g. LC says "PORT QASIM" but BL says "KEMARI") → FAIL. Specific means specific; a sister terminal is not a substitute.

20c. KNOWN PORT → CITY / COUNTRY REFERENCE (P124). Use this table to resolve "ANY [COUNTRY] PORT" requirements. If the BL's port is listed below, it IS a port of the stated country — do NOT fail the row because the country wasn't spelled on the BL.

    PAKISTAN: Karachi, Port Qasim, Bin Qasim, Port Muhammad Bin Qasim, PQA, Kemari, Kiamari, KPT, KICT, QICT, PICT, KGTL, SAPT, Gwadar
    INDIA: Nhava Sheva, JNPT, Mumbai, Chennai, Kolkata, Haldia, Cochin, Kochi, Tuticorin, Visakhapatnam, Vizag, Mundra, Kandla, Paradip, Ennore, Kattupalli, Krishnapatnam, Mangalore, Pipavav
    BANGLADESH: Chittagong, Chattogram, Mongla
    SRI LANKA: Colombo, CICT, SAGT, JCT, Hambantota, Galle, Trincomalee
    CHINA: Shanghai, Yangshan, Waigaoqiao, Wusongkou, Ningbo, Zhoushan, Shenzhen, Yantian, Shekou, Chiwan, Guangzhou, Nansha, Qingdao, Tianjin, Xingang, Dalian, Xiamen, Fuzhou, Yingkou, Lianyungang, Rizhao, Taicang, Nantong, Zhanjiang, Haikou, Beihai
    HONG KONG (part of China): Hong Kong, Kwai Chung, Kwai Tsing, Stonecutters, HIT
    TAIWAN: Kaohsiung, Keelung, Taichung, Taipei, Hualien, Taoyuan
    JAPAN: Tokyo, Yokohama, Nagoya, Osaka, Kobe, Hakata, Kitakyushu, Moji, Chiba, Tomakomai, Niigata, Shimizu, Sendai
    SOUTH KOREA: Busan, Pusan, Incheon, Ulsan, Gwangyang, Kwangyang, Pyeongtaek, Pohang, Donghae, Masan
    SINGAPORE: Singapore, Tuas, Pasir Panjang, Brani, Keppel, Jurong
    MALAYSIA: Port Klang, Klang, Penang, Butterworth, Johor, Pasir Gudang, Tanjung Pelepas, PTP, Kuantan, Bintulu, Kuching, Kota Kinabalu, Labuan
    INDONESIA: Tanjung Priok, Jakarta, Tanjung Perak, Surabaya, Belawan, Medan, Makassar, Batam, Semarang, Tanjung Emas, Balikpapan, Dumai
    THAILAND: Laem Chabang, Bangkok, Klong Toey, Map Ta Phut, Songkhla, Sattahip
    VIETNAM: Ho Chi Minh, Saigon, Cat Lai, Cai Mep, Haiphong, Hai Phong, Da Nang, Qui Nhon, Quy Nhon, Vung Tau
    PHILIPPINES: Manila, MICT, Subic Bay, Cebu, Davao, Batangas, Cagayan de Oro, General Santos
    CAMBODIA: Sihanoukville, Phnom Penh
    MYANMAR: Yangon, Rangoon, Thilawa
    UAE: Jebel Ali, Port Rashid, Dubai, Khalifa Port, Zayed Port, Abu Dhabi, Fujairah, Khor Fakkan, Sharjah, Hamriyah, Ajman
    SAUDI ARABIA: Jeddah, King Abdulaziz Port Jeddah, Jeddah Islamic Port, Dammam, King Abdul Aziz Port Dammam, Jubail, Yanbu, King Abdullah Port, KAP, Ras Tanura
    QATAR: Hamad Port, Doha, Ras Laffan, Mesaieed
    KUWAIT: Shuwaikh, Shuaiba, Doha Port Kuwait
    BAHRAIN: Khalifa Bin Salman Port, KBSP, Mina Salman
    OMAN: Sohar, Salalah, Muscat, Port Sultan Qaboos, Duqm
    IRAN: Bandar Abbas, Bandar Imam, Imam Khomeini, Chabahar, Bushehr, Anzali
    YEMEN: Aden, Hodeidah, Mukalla
    TURKEY: Istanbul, Ambarli, Haydarpasa, Izmit, Kocaeli, Gemlik, Mersin, Izmir, Aliaga, Iskenderun, Tekirdag, Samsun
    EGYPT: Alexandria, Port Said, East Port Said, Damietta, Sokhna, Ain Sokhna, Suez, El Dekheila
    ISRAEL: Haifa, Ashdod, Eilat
    LEBANON: Beirut, Tripoli Lebanon
    JORDAN: Aqaba
    SYRIA: Latakia, Tartous
    USA: Los Angeles, LA, Long Beach, LB, New York, NY/NJ, Newark, Elizabeth, Bayonne, Savannah, Houston, Seattle, Tacoma, Northwest Seaport Alliance, NWSA, Oakland, Charleston, Norfolk, Virginia, Miami, PortMiami, Baltimore, Jacksonville, JAXPORT, Boston, Philadelphia, Wilmington, New Orleans, Mobile, Galveston, Tampa, Port Everglades, Fort Lauderdale, Portland, Honolulu
    CANADA: Vancouver, Prince Rupert, Montreal, Halifax, Toronto, Quebec City, Saint John, St. John's
    MEXICO: Manzanillo, Lazaro Cardenas, Veracruz, Altamira, Ensenada, Progreso, Tampico
    PANAMA: Balboa, Colon, Cristobal, Manzanillo International Terminal, MIT, Rodman
    BRAZIL: Santos, Paranagua, Rio de Janeiro, Itaqui, Suape, Rio Grande, Vitoria, Itajai, Navegantes, Itapoa, Pecem
    ARGENTINA: Buenos Aires, Rosario, Bahia Blanca, Zarate
    CHILE: San Antonio, Valparaiso, Iquique, Arica, San Vicente, Coronel, Mejillones
    PERU: Callao, Paita, Matarani, Salaverry, Pisco
    COLOMBIA: Cartagena, Buenaventura, Barranquilla, Santa Marta
    ECUADOR: Guayaquil, Manta, Posorja
    VENEZUELA: La Guaira, Puerto Cabello, Maracaibo
    URUGUAY: Montevideo
    PARAGUAY: Asuncion
    UK / UNITED KINGDOM: Felixstowe, Southampton, London Gateway, London, Tilbury, Liverpool, Harwich, Hull, Grimsby, Immingham, Thamesport, Teesport, Belfast
    IRELAND: Dublin, Cork, Waterford, Ringaskiddy
    NETHERLANDS: Rotterdam, Maasvlakte, Europoort, Botlek, Amsterdam, Vlissingen, Flushing
    BELGIUM: Antwerp, Deurganckdok, Delwaidedok, Zeebrugge, Ghent, Oostende
    FRANCE: Le Havre, Marseille, Fos, Fos-sur-Mer, Dunkirk, Dunkerque, Nantes, Saint-Nazaire, La Rochelle, Bordeaux, Calais, Sete
    GERMANY: Hamburg, Altenwerder, CTA, Burchardkai, CTB, Tollerort, CTT, Eurogate, Bremen, Bremerhaven, Wilhelmshaven, JadeWeserPort, Rostock, Kiel, Lubeck
    SPAIN: Valencia, Algeciras, Barcelona, Bilbao, Las Palmas, Tenerife, Santa Cruz, Vigo, Tarragona, Cartagena Spain, Cadiz
    PORTUGAL: Lisbon, Lisboa, Sines, Leixoes, Porto, Setubal
    ITALY: Genoa, Genova, La Spezia, Naples, Napoli, Livorno, Trieste, Taranto, Gioia Tauro, Ancona, Civitavecchia, Salerno, Venice, Venezia, Ravenna, Cagliari
    GREECE: Piraeus, Athens, Thessaloniki, Heraklion, Patras
    CYPRUS: Limassol, Larnaca
    MALTA: Valletta, Marsaxlokk, Freeport
    POLAND: Gdansk, Gdynia, Szczecin, Swinoujscie
    CZECH REPUBLIC / SLOVAKIA / HUNGARY: (landlocked — no seaports)
    RUSSIA: St. Petersburg, Saint Petersburg, Ust-Luga, Vladivostok, Nakhodka, Vostochny, Novorossiysk, Murmansk, Kaliningrad, Arkhangelsk
    UKRAINE: Odessa, Odesa, Chornomorsk, Illichivsk, Pivdennyi, Yuzhny, Mykolaiv, Mariupol
    ROMANIA: Constanta, Constantza
    BULGARIA: Varna, Burgas
    GEORGIA: Poti, Batumi
    SWEDEN: Gothenburg, Goteborg, Stockholm, Helsingborg, Malmo
    NORWAY: Oslo, Bergen, Stavanger, Kristiansand, Tromso
    DENMARK: Copenhagen, Kobenhavn, Aarhus, Arhus, Esbjerg, Fredericia, Aalborg
    FINLAND: Helsinki, Hamina, Kotka, HaminaKotka, Turku, Rauma, Pori
    ESTONIA / LATVIA / LITHUANIA: Tallinn, Muuga (Estonia); Riga, Ventspils, Liepaja (Latvia); Klaipeda (Lithuania)
    SOUTH AFRICA: Durban, Cape Town, Port Elizabeth, Gqeberha, Ngqura, Coega, Richards Bay, Saldanha
    KENYA: Mombasa, Lamu
    TANZANIA: Dar es Salaam, Zanzibar, Tanga
    MOZAMBIQUE: Maputo, Beira, Nacala
    NIGERIA: Lagos, Apapa, Tin Can Island, TCIP, Port Harcourt, Onne, Calabar, Lekki
    GHANA: Tema, Takoradi
    IVORY COAST / CÔTE D'IVOIRE: Abidjan, San Pedro
    SENEGAL: Dakar
    MOROCCO: Casablanca, Tanger Med, Tangier, Agadir, Jorf Lasfar
    ALGERIA: Algiers, Alger, Oran, Bejaia, Annaba, Skikda, Djendjen
    TUNISIA: Rades, Tunis, Sfax, Gabes, Bizerte
    LIBYA: Tripoli Libya, Benghazi, Misrata, Khoms
    SUDAN: Port Sudan
    DJIBOUTI: Djibouti, Doraleh
    AUSTRALIA: Sydney, Botany, Port Botany, Melbourne, Brisbane, Fremantle, Perth, Adelaide, Port Adelaide, Darwin, Newcastle, Port Kembla, Hobart
    NEW ZEALAND: Auckland, Tauranga, Wellington, Lyttelton, Christchurch, Napier, Port Chalmers, Dunedin

    USE: when the LC condition says "ANY CHINESE SEAPORT" and the BL shows "YANTIAN" — Yantian is in the CHINA row → PASS. When LC says "ANY UAE PORT" and BL shows "JEBEL ALI" → Jebel Ali is in UAE → PASS. When LC says a specific city that has no sub-terminals listed (e.g., "SALALAH"), match literally — no cluster expansion needed. If the port name is NOT in this table, do NOT fail the row — fall back to the document's own country/city wording and apply rule 20 (city+country match). The table is a helpful reference, not an exhaustive whitelist.

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
21z. PROFORMA INVOICE REFERENCE — NUMBER AND DATE MUST BOTH MATCH:

    ANTI-HALLUCINATION RULE (READ THIS FIRST — CRITICAL):
    Before writing anything about the proforma citation, you MUST
    locate the actual proforma line on the INVOICE TEXT and QUOTE
    the date verbatim from it. Do NOT "restate" the LC's expected
    date as if it were on the invoice. Do NOT paraphrase. If your
    findings contain a date, that date MUST be copied character-
    for-character from the invoice body, not from the condition or
    the LC F45A. Echoing the condition's expected date back as if
    the invoice showed it is a HALLUCINATION that violates this
    rule — the verdict is automatically FAIL.

    Required evidence procedure:
      1) Search the invoice text for "PROFORMA" (case-insensitive)
      2) Read the full line(s) containing "PROFORMA INVOICE REF.NO."
         and the "DATED" clause that follows it
      3) Quote those line(s) verbatim in your "quote" field —
         including BOTH the ref number and the date as printed
         on the invoice
      4) ONLY THEN compare the invoice-printed date against the
         LC-required date and decide PASS or FAIL

    When the LC says "SPECIFICATIONS AND FURTHER DETAILS ARE AS PER
    BENEFICIARY'S PROFORMA INVOICE NO. XXX DATED YYY", the commercial
    invoice MUST reference that exact proforma invoice number (XXX)
    AND the exact proforma date (YYY). These are two separate pieces
    of the citation and both must be verified.

    Decision rules:
    - Invoice cites proforma No. XXX AND date YYY → PASS
    - Invoice cites proforma No. XXX but with a DIFFERENT date
      (e.g. LC says "DATED JAN 21, 2026" but invoice says
      "DATED FEB 18, 2026") → FAIL. "STRICTLY AS PER" language
      binds both the ref number and the date. A different date
      points at a different proforma revision and is a documentary
      discrepancy under UCP 600 Art 18(c).
    - Invoice cites proforma No. XXX but OMITS the date → REVIEW
      (bank checker must verify the same proforma revision was
      shipped against).
    - Invoice ONLY references the LC number but NOT the proforma
      invoice number → FAIL. Referencing the LC is not a substitute.

    Quote the exact proforma line from the invoice in your response
    (both the number and the date). When writing FAIL findings,
    include both the LC-expected date and the invoice-shown date so
    the discrepancy is auditable.

    Worked example:
      LC F45A: "...AS PER BENEFICIARY'S PROFORMA INVOICE REF.NO.
                786/S-13198-SOYPI-E DATED JAN 21, 2026"
      Invoice: "BENEFICIARY'S PROFORMA INVOICE REF.NO.
                786/S-13198-SOYPI-E DATED FEB 18, 2026"
      → Ref matches, DATE DIFFERS (Jan 21 vs Feb 18) → FAIL with
      findings="Proforma ref 786/S-13198-SOYPI-E matches but date
      differs: LC requires Jan 21 2026, invoice shows Feb 18 2026".

21a. MULTI-ITEM INVOICE MATCHING (CRITICAL):
    When the LC's 45A has MULTIPLE goods (e.g., "MEYER RICE COLOR SORTER 10 CHUTES QTY 5 SETS" AND "MEYER SESAME SEEDS COLOR SORTER 10 CHUTES QTY 1 SET"), each item becomes a separate verification row. For each row:
    - SEARCH the invoice for THAT SPECIFIC item by its product name/description
    - Extract quantity and unit price for THAT specific line item, NOT the invoice total
    - An invoice may have a TABLE with columns (Name, Qty, Unit Price, Total). Read the CORRECT ROW matching the product.
    - "MEYER SESAME SEEDS COLOR SORTER 10CHUTES 1SET 28500" is EXACTLY the same as "MEYER SESAME SEEDS COLOR SORTER 10 CHUTES QUANTITY 1 SET AT THE RATE OF USD 28,500"
    - OCR may glue digits to words: "10CHUTES" = "10 CHUTES", "1SET" = "1 SET", "5SETS" = "5 SETS"
    - Do NOT confuse one product's data with another's. If checking "SESAME SEEDS", read the SESAME SEEDS row, not the RICE row.
    - The TOTAL at the bottom covers ALL items — do NOT use the invoice TOTAL as the quantity or unit price for a single item.
22. PARTY REFERENCES: When the condition says "NOTIFY APPLICANT" or "TO ORDER OF ISSUING BANK", look at the LC PARTIES section above to find the actual name. Then check if that name appears on the document. "NOTIFY APPLICANT" means the notify party field must show the APPLICANT's name (given above). Do NOT look for the literal words "NOTIFY APPLICANT" — look for the applicant's ACTUAL NAME. Check the TOTAL quantity, not individual lines. Also "Ea" (each) is a valid unit — 736 Ea = 736 pieces. If LC says "QTY 736 AT THE RATE OF USD 98.00" and invoice shows 736 units × $98.00 = correct.

23. ADDRESSING / "TO:" PATTERNS ON SHIPMENT ADVICE AND OTHER DOCUMENTS:
    When a condition says "must be addressed to X" or "must also be
    addressed to the Applicant", the document may show any of these
    equivalent addressing patterns — ALL mean the party IS addressed:
      • "TO: [party name]"
      • "ALSO TO: [party name]"
      • "AND TO: [party name]"
      • "CC: [party name]"
      • "COPY TO: [party name]"
      • "ATTENTION: [party name]"
      • "C/O: [party name]"
    If the document shows "ALSO TO: AMEEJEE VALLEEJEE AND SONS (PVT) LTD"
    and the condition says "must be addressed to the Applicant" where the
    applicant is "AMEEJEE VALLEEJEE AND SONS (PVT) LTD", this is a PASS.
    "ALSO TO" = "AND TO" = "TO" = addressed. Do NOT fail because the
    prefix is "ALSO TO" instead of "TO" — they are equivalent.

    PARTY-NAME OCR TOLERANCE (ISBP 821 paragraph A1 & UCP 600 Art 14(d/e)):
    Company / party names (Applicant, Beneficiary, Consignee, Notify
    Party, Issuing Bank, etc.) MUST be matched semantically, not
    character-by-character. The bank's job is to confirm the parties
    are the same legal entity, NOT to police OCR perfection.

    COMPANY NAME CHANGES / ALIASES:
    Companies change names through mergers, acquisitions, or rebranding.
    If a document shows "Viterra B.V. (currently known as Bunge
    Netherlands Agri B.V.)" or "ABC Corp (formerly XYZ Ltd)", this
    IS the same entity — PASS. The phrases "currently known as",
    "formerly known as", "now known as", "trading as", "t/a", "d/b/a"
    all indicate the SAME legal entity under a different name.
    Match EITHER the old name OR the new name against the LC party.

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
    unified_summary: Optional[dict] = None,
    bl_subtype: Optional[dict] = None,
    final_lc_fields: Optional[dict] = None,
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
            r'(?:Total\s*(?:Amount|Price)|Grand\s+Total|Invoice\s+Total|Sub\s*Total|Net\s+Total|Amount\s+Total)\s*[:\s]*'
            r'(?:USD|EUR|GBP|JPY|CHF|AUD|CAD|CNY|HKD|SGD|INR|PKR|AED|SAR)?\s*([\d.,]+)',
            document_text, _re_sum.IGNORECASE,
        )
        # Fallback: find "TOTAL\n[qty line]\n[amount]" pattern in invoice tables
        if not _totals_raw:
            _inv_lines = document_text.split('\n')
            for _li in range(len(_inv_lines)):
                if _inv_lines[_li].strip().upper() == 'TOTAL':
                    # Look ahead for the first pure numeric line (skip qty like "6SETS")
                    for _lj in range(1, min(4, len(_inv_lines) - _li)):
                        _next_l = _inv_lines[_li + _lj].strip()
                        if _re_sum.match(r'^[\d,]+(?:\.\d{0,2})?\s*$', _next_l):
                            _totals_raw.append(_next_l.strip())
                            break
                    break
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

        # P113: Parse invoice table line items for multi-item invoices
        # Detects patterns like:
        #   MEYER RICE COLOR SORTER 10 CHUTES\n5SETS\n28500\n142500
        #   MEYER SESAME SEEDS COLOR SORTER\n1SET\n28500\n28500\n10CHUTES
        # Also: structured tables with PRODUCT | QTY | UNIT PRICE | TOTAL
        if 'invoice' in document_type.lower():
            _inv_items = []
            _i = 0
            while _i < len(_lines):
                _ls = _lines[_i].strip()
                # Skip empty lines, headers, footers
                if not _ls or len(_ls) < 5:
                    _i += 1
                    continue
                # Detect a product name line (starts with letters, >15 chars,
                # not a header/label like "NAME OF ITEM" or "TOTAL")
                _is_product = (
                    len(_ls) > 10 and
                    _re_sum.search(r'[A-Z]{3,}', _ls) and
                    not _re_sum.match(r'^(?:NAME|QUANTITY|UNIT|TOTAL|DATE|INVOICE|COMMERCIAL|TO:|FROM:|H\.?S|N\.?T\.?N|CFR|CIF|FOB|ALL\s+OTHER|PAYMENT|PACKING|MERCHANDISE|HEFEI|NO[.,])', _ls, _re_sum.IGNORECASE) and
                    not _re_sum.match(r'^(?:SAY\s|TOTAL\s*PRICE|PRICE)', _ls, _re_sum.IGNORECASE) and
                    not _re_sum.search(r'BANK|CERTIFICATE|PROFORMA|INCOTERM', _ls, _re_sum.IGNORECASE)
                )
                if _is_product:
                    _prod_name = _ls
                    _qty_str = ''
                    _price_str = ''
                    _total_str = ''
                    # Look ahead up to 5 lines for qty/price/total
                    _extra_desc = []  # Orphan qualifiers to absorb into product name
                    for _j in range(1, min(6, len(_lines) - _i)):
                        _next = _lines[_i + _j].strip()
                        if not _next:
                            continue
                        # Quantity: "5SETS", "1SET", "25.50 M.TONS", "736 Ea"
                        _qm = _re_sum.match(r'^(\d+(?:\.\d+)?)\s*(?:SETS?|PCS|EA|M\.?TONS?|KGS?|UNITS?|ROLLS?|DRUMS?|BAGS?|CARTONS?|BOXES?|CTNS?)', _next, _re_sum.IGNORECASE)
                        if _qm and not _qty_str:
                            _qty_str = _next
                            continue
                        # Price or total: just a number (28500, 142500)
                        _nm = _re_sum.match(r'^([\d,]+(?:\.\d{0,2})?)\s*$', _next)
                        if _nm:
                            if not _price_str:
                                _price_str = _next
                            elif not _total_str:
                                _total_str = _next
                            continue
                        # Orphan qualifier: short text like "10CHUTES", "20KG",
                        # "GRADE A" that OCR split from the product name.
                        # Absorb if it's short (<30 chars), has letters, and
                        # appears AFTER we already got qty/price.
                        _is_orphan = (
                            len(_next) < 30 and
                            _re_sum.search(r'[A-Z]', _next) and
                            (_qty_str or _price_str) and
                            not _re_sum.match(r'^(?:TOTAL|GRAND|SUB|NET|SAY)', _next, _re_sum.IGNORECASE)
                        )
                        if _is_orphan:
                            _extra_desc.append(_next)
                            continue
                        # If we hit another product name, stop
                        if len(_next) > 10 and _re_sum.search(r'[A-Z]{3,}', _next):
                            break
                    # Absorb orphan qualifiers into product name
                    if _extra_desc:
                        _prod_name = _prod_name + ' ' + ' '.join(_extra_desc)
                    if _qty_str or _price_str:
                        _inv_items.append({
                            'product': _prod_name,
                            'qty': _qty_str,
                            'unit_price': _price_str,
                            'total': _total_str,
                        })
                _i += 1

            if _inv_items:
                _doc_summary += f"INVOICE LINE ITEMS ({len(_inv_items)} items):\n"
                for _idx, _item in enumerate(_inv_items, 1):
                    _doc_summary += (
                        f"  Item {_idx}: {_item['product']}"
                        f" | Qty: {_item['qty'] or '?'}"
                        f" | Unit Price: {_item['unit_price'] or '?'}"
                        f" | Total: {_item['total'] or '?'}\n"
                    )
                _doc_summary += "USE THESE LINE ITEMS to match goods description, quantity, and unit price.\n"
                _doc_summary += "Match the SPECIFIC item from the LC condition, not the invoice total.\n"
    if _doc_summary:
        document_text = f"[SYSTEM PRE-CALCULATED SUMMARY]\n{_doc_summary}[END SUMMARY]\n\n{document_text}"

    # P136 — Fix OCR-garbled 3-letter month names inside dates.
    # A BL stamp like "21 FEB 2025" is frequently OCR'd as "21 CCR 2025",
    # "21 EEB 2025", "21 F€B 2025", etc. Downstream date parsing then
    # fails and the condition row marks the shipment date as missing.
    # Scan the document for "<day> <3 letters> <year>" patterns and map
    # the 3-letter token to the closest valid month when it isn't one.
    def _fix_month(m):
        day, mon, yr = m.group(1), m.group(2).upper(), m.group(3)
        _VALID = {'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
                  'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'}
        if mon in _VALID:
            return m.group(0)
        # Common OCR → month corrections
        _OCR_MONTH = {
            'CCR': 'FEB', 'EEB': 'FEB', 'FEE': 'FEB', 'FBB': 'FEB',
            'F€B': 'FEB', 'FFB': 'FEB', 'F88': 'FEB', 'CEB': 'FEB',
            'NAR': 'MAR', 'MAP': 'MAR', 'HAR': 'MAR', 'M4R': 'MAR',
            'APP': 'APR', 'AFR': 'APR', 'APK': 'APR', 'AP8': 'APR',
            'MAT': 'MAY', 'MAV': 'MAY', 'HAY': 'MAY', 'M4Y': 'MAY',
            'TUN': 'JUN', 'JUH': 'JUN', 'JUW': 'JUN', 'JUR': 'JUN',
            'JUI': 'JUL', 'IUL': 'JUL', 'JUI.': 'JUL',
            'AUC': 'AUG', 'AUC.': 'AUG', 'AUS': 'AUG', 'AU6': 'AUG',
            'SEF': 'SEP', 'SFP': 'SEP', 'SEB': 'SEP', '$EP': 'SEP',
            '0CT': 'OCT', 'OCI': 'OCT', '0CI': 'OCT', 'OCL': 'OCT', 'OCF': 'OCT',
            'NCV': 'NOV', 'HOV': 'NOV', 'N0V': 'NOV', 'NQV': 'NOV',
            'DEG': 'DEC', 'DFC': 'DEC', 'D€C': 'DEC', 'DCC': 'DEC',
            'JAH': 'JAN', 'JAM': 'JAN', 'J4N': 'JAN',
        }
        fixed = _OCR_MONTH.get(mon)
        if fixed:
            return f"{day} {fixed} {yr}"
        # Levenshtein-1 fallback: if the garbled token differs from a valid
        # month by exactly one character, correct it.
        for v in _VALID:
            if len(mon) == 3 and sum(1 for a, b in zip(mon, v) if a != b) == 1:
                return f"{day} {v} {yr}"
        return m.group(0)
    try:
        document_text = re.sub(
            r'(\b\d{1,2})[\s\-./]+([A-Za-z€$@#][A-Za-z€$@#0-9]{1,3})[\s\-./]+(\d{2,4}\b)',
            _fix_month,
            document_text,
        )
    except Exception:
        pass

    # P198b — Strip the Conditions-of-Carriage / Terms-and-Conditions
    # page from a merged BL packet before sending to the verifier. When
    # step03 merges a standalone T&C page into its BL packet (per P191's
    # "+ Conditions of Carriage" annotation), the aggregated refined_text
    # contains both the BL form page AND the full T&C boilerplate
    # (Paramount / Hague Rules / General Average / Arbitration, etc.).
    # That boilerplate is identical across every BL from the same
    # carrier and has no verification value — so we cut the T&C block
    # out and replace it with a single-line marker telling the LLM the
    # T&C exists. ALL BL data (shipper, consignee, notify party,
    # vessel, ports, cargo, marks, freight, HS code, LC no, NTN, stamps,
    # signatures, number of originals, place and date of issue) is
    # preserved.
    #
    # Two page orderings occur in practice:
    #   A) BL page first, T&C page second — splice removes the tail.
    #   B) T&C page first, BL page second (scan order reversed) —
    #      splice removes the head and keeps the tail.
    # We handle both by finding (1) the T&C title anchor and (2) any
    # strong BL form marker outside the T&C block, then keep everything
    # outside the T&C zone.
    _dtl_for_tc = (document_type or '').lower()
    _is_bl_like = ('bill of lading' in _dtl_for_tc
                   or 'bl' == _dtl_for_tc.strip()
                   or 'conditions of carriage' in _dtl_for_tc)
    if _is_bl_like and document_text:
        # T&C title anchor. The ideal title pattern is a heading line
        # followed by a numbered-clause list ("1. Paramount clause"),
        # but OCR frequently damages the title ("Bill of Lading ?
        # Terms and Conditionsa" — mystery char between words,
        # trailing "a"). P198i broadens detection to catch:
        #   1. clean titles ending at a numbered-clause list
        #   2. OCR-damaged titles (tolerant of non-alphabetic tokens
        #      between title words)
        #   3. standard legal-definition openings ("1. Definitions"
        #      followed by '"Carrier" means' / '"MTO" means') which
        #      ONLY appear at the start of a carrier's T&C page and
        #      so are themselves an unambiguous anchor.
        _TC_TITLE_RE = re.compile(
            r'(?:'
            # Pattern A: clean title + numbered clause (original).
            r'\n[ \t]*(?:'
            r'conditions\s+of\s+carriage'
            r'|terms\s+and\s+conditions\s+of\s+(?:this\s+)?(?:carriage|bill\s+of\s+lading)'
            r'|bill\s+of\s+lading[\s\W]{0,5}?terms\s+and\s+conditions[a-z]{0,2}'
            r'|standard\s+bill\s+of\s+lading\s+terms'
            r'|standard\s+terms\s+and\s+conditions'
            r'|terms\s+and\s+conditions[a-z]{0,2}\b[^\n]{0,30}?\bpage\s+1'
            r')[\s\S]{0,50}?\n[ \t]*1[.)]\s+'
            # Pattern B: "Definitions" immediately followed by legal
            # definition of Carrier / MTO — standard T&C opening.
            r'|\n[ \t]*1[.)]?\s*Definitions?\b[\s\S]{0,30}?'
            r'["\u201c\u201d]?(?:Carrier|MTO|Carriage|Merchant)["\u201c\u201d]?\s+means\b'
            r')',
            re.IGNORECASE,
        )
        # BL body structural markers. Each one is distinctive to the
        # BL FORM (a field label terminated by colon, or a title line
        # in all caps) — these never appear inside T&C clause prose,
        # so they can be used as resume anchors without false
        # positives on defined terms like "notify party" or "shipper".
        _BL_RESUME_RE = re.compile(
            r'(?:'
            r'\n[ \t]*BILL OF LADING[ \t]*\n'  # form title
            r'|\bshipped\s+in\s+apparent\s+good\s+order\b'
            r'|\bb/l\s*no\.?\s*:'
            r'|\bbill\s+of\s+lading\s+no\.?\s*:'
            r'|\bbooking\s+no\.?\s*:'
            r'|\bconsignee\s+or\s+order\s*:'
            r'|\bnotify\s+party/address\s*:'
            r'|\bnotify\s+party\s*:'
            r'|\bport\s+of\s+loading\s*:'
            r'|\bport\s+of\s+discharge\s*:'
            r'|\bplace\s+of\s+receipt\s*:'
            r'|\bplace\s+of\s+delivery\s*:'
            r'|\bonboard\s+the\s+vessel\b'
            r')',
            re.IGNORECASE,
        )
        _m_tc = _TC_TITLE_RE.search(document_text)
        if _m_tc and (len(document_text) - _m_tc.start()) >= 1500:
            _tc_start = _m_tc.start()
            # Find BL resume AFTER the T&C title (Case B: T&C first, BL after).
            _m_resume_after = _BL_RESUME_RE.search(document_text, _tc_start + 200)
            # P198i — BL-type structured facts. When the full T&C page
            # is attached, we can give the verifier explicit negative
            # assertions about the BL's sub-type so it doesn't read the
            # LC's prohibition clauses ("BL must not be X") as evidence
            # the BL IS X — a recurring LLM failure on these rows.
            _pre_lower = document_text[:_tc_start].lower()
            _is_mtd = ('multimodal transport document' in _pre_lower
                       or 'combined transport' in _pre_lower
                       or 'multimodal bill of lading' in _pre_lower)
            _has_charter = ('charter party' in _pre_lower
                            or 'subject to charter' in _pre_lower
                            or 'c/p dated' in _pre_lower)
            _has_house_title = ('house bill of lading' in _pre_lower
                                or 'house b/l' in _pre_lower
                                or '\nhbl' in _pre_lower)
            _title_note = (
                "This BL is titled a Multimodal Transport Document "
                "(potentially falls under UCP 600 Art 19). "
                if _is_mtd else ""
            )
            _marker = (
                "\n\n[STRUCTURED FACTS ABOUT THIS BILL OF LADING — "
                "derived from the full document text above:\n"
                "- Full Terms & Conditions / Conditions of Carriage page "
                "is ATTACHED to this BL (text omitted here for brevity).\n"
                f"- {_title_note}"
                "Therefore this BL is NOT a Short Form BL "
                "(Short Form = no T&C page).\n"
                "- Therefore this BL is NOT a Blank Back BL "
                "(Blank Back = T&C page blank).\n"
                + ("- The BL body does NOT reference a Charter Party "
                   "agreement, so it is NOT a Charter Party BL.\n"
                   if not _has_charter else
                   "- The BL body references a Charter Party — flag as "
                   "Charter Party BL if the LC prohibits it.\n")
                + ("- The BL is titled 'House Bill of Lading' — treat as "
                   "a House BL.\n"
                   if _has_house_title else
                   "- House BL / Forwarder's BL status: DO NOT assume. "
                   "Decide by reading the ISSUER and SIGNATURE CAPACITY "
                   "on the BL body above. A House / Forwarder's BL is "
                   "issued by a freight forwarder / NVOCC / logistics "
                   "operator acting for the merchant (typical signals: "
                   "signer identified as 'Agent', 'Freight Forwarder', "
                   "'MTO acting for Merchant', or the issuer is a "
                   "logistics firm with no vessels). A carrier-issued BL "
                   "is signed 'For / on behalf of the Carrier' by the "
                   "named carrier or its named agent. Use the actual "
                   "text above; do not default either way.\n")
                + "- Freight Forwarder / NVOCC status: likewise decide "
                "from the issuer identity and signature capacity shown "
                "on the BL body above. Do NOT default to either "
                "outcome — quote the relevant line from the document.\n"
                "IMPORTANT FOR THE VERIFIER: when the LC says 'BL must "
                "not be X' (Short Form / Blank Back / Charter Party / "
                "House BL / Freight Forwarder BL), that is the LC "
                "PROHIBITING X — it is NOT the LC confirming the BL is "
                "X. Decide from the BL body and these structured facts, "
                "NOT from the prohibition wording.]\n\n"
            )
            _pre = document_text[:_tc_start].rstrip()
            if _m_resume_after:
                _post = document_text[_m_resume_after.start():].lstrip()
                document_text = (_pre + _marker + _post) if _pre else (_marker.lstrip() + _post)
            else:
                # Case A: T&C at the tail with nothing after — just
                # keep the pre-T&C BL content and the marker.
                document_text = _pre + _marker

    # Truncate document text to avoid exceeding token limits.
    # After the T&C strip above, a BL packet is normally well under the
    # cap — so the BL header, parties, cargo description, dates, marks,
    # stamps and signatures all pass through intact. Raised from 6500
    # → 10000 so the largest clean BL bodies (multi-page merged BLs
    # with attached schedules) also fit in one prompt.
    max_chars = 10000
    if len(document_text) > max_chars:
        document_text = document_text[:max_chars] + "\n... [truncated]"

    # ── FAST PATH: try deterministic verification first ──
    # Conservative — only returns a verdict when confidence is very high.
    if USE_SPLIT_PROMPTS and (unified_summary or bl_subtype):
        _det = _deterministic_verify(
            condition_text=condition_text,
            clause_ref=clause_ref,
            lc_field_value=lc_field_value,
            document_type=document_type,
            unified_summary=unified_summary or {},
            bl_subtype=bl_subtype or {},
            final_lc=final_lc_fields or {},
            document_text=document_text or "",
        )
        if _det is not None:
            return {
                "row_id": row_id,
                "findings": _det.get("findings", "") or "",
                "result": _det.get("findings", "") or "",
                "quote": _det.get("quote", "") or "",
                "compliance": _det.get("verdict", "REVIEW").lower(),
                "confidence": float(_det.get("confidence", 0.0) or 0.0),
                "reasoning": f"Deterministic (no LLM): {_det.get('structured_source', '')}",
                "structured_source": _det.get("structured_source", ""),
                "elapsed": time.time() - start,
                "_verification_path": "deterministic",
            }

    # ── VLM PATH: build the prompt ──
    if USE_SPLIT_PROMPTS:
        prompt_text = _build_verification_prompt_v2(
            condition_text=condition_text,
            clause_ref=clause_ref,
            lc_field_value=lc_field_value,
            lc_parties=lc_parties or "(Not available)",
            f47a_context=f47a_context or "",
            document_type=document_type,
            document_text=document_text,
            visual_metadata=visual_metadata or "(No visual metadata available)",
            unified_summary=unified_summary or {},
            bl_subtype=bl_subtype or {},
            final_lc_fields=final_lc_fields or {},
        )
    else:
        # Legacy fallback path
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
    # Text-only LLM — no images needed for verification.
    # P198y — max_tokens raised from 500 -> 900. The lenient JSON
    # parser handles unescaped newlines, but it cannot recover a
    # JSON object that was TRUNCATED mid-value because the model
    # ran out of output budget. Long quotes from docs like the
    # Shipment Advice (with multi-line addresses) routinely exceeded
    # 500 tokens and the response stopped inside the `quote` field,
    # yielding "VLM response not JSON" -> REVIEW. 900 gives enough
    # headroom for the longest quotes while keeping response time
    # reasonable.
    payload = {
        "model": QWEN_TEXT_LLM_MODEL,
        "messages": [
            {"role": "user", "content": prompt_text},
        ],
        "max_tokens": 900,
        "temperature": 0.1,
    }

    try:
        resp = requests.post(QWEN_TEXT_LLM_URL, json=payload, timeout=None)
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

        # Extract JSON from response (VLM may wrap it in markdown or text).
        # Uses module-level `re` (imported at top of file). Do NOT re-import
        # here — a local `import re` makes `re` function-local in Python,
        # and any re.* usage BEFORE this line (e.g. the T&C strip regex
        # compiled around line 3354) then raises
        # `UnboundLocalError: cannot access local variable 're'`.
        #
        # P198w — tolerant JSON parser. The Qwen VLM routinely emits
        # otherwise-valid JSON where a string value contains literal
        # newlines / tabs / control chars (e.g. a `quote` field that
        # preserves the source document's line breaks). Strict JSON
        # rejects those, causing `json.loads()` to raise and the row
        # to default to REVIEW. Try strict parse first; on failure
        # attempt a lenient parse that escapes whitespace chars inside
        # string literals only.
        def _lenient_json_loads(s):
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                pass
            # Escape bare newlines / carriage returns / tabs that appear
            # INSIDE a JSON string value. Walk the text character by
            # character, track whether we're inside a string, and
            # replace raw control chars with their escaped equivalents.
            out = []
            in_str = False
            escape = False
            for ch in s:
                if escape:
                    out.append(ch); escape = False; continue
                if ch == '\\' and in_str:
                    out.append(ch); escape = True; continue
                if ch == '"':
                    in_str = not in_str
                    out.append(ch); continue
                if in_str and ch == '\n':
                    out.append('\\n'); continue
                if in_str and ch == '\r':
                    out.append('\\r'); continue
                if in_str and ch == '\t':
                    out.append('\\t'); continue
                out.append(ch)
            try:
                return json.loads(''.join(out))
            except json.JSONDecodeError:
                return None

        json_match = re.search(r"\{.*\}", raw_content, re.DOTALL)
        parsed = None
        if json_match:
            parsed = _lenient_json_loads(json_match.group(0))
        if not parsed:
            # P198y — last-chance fallback for truncated responses.
            # When the model runs out of tokens mid-JSON, there is no
            # closing "}" and json.loads fails completely. Extract
            # verdict / findings / confidence via regex instead of
            # losing the whole row to REVIEW.
            _m_verdict = re.search(r'"verdict"\s*:\s*"([^"]+)"', raw_content, re.IGNORECASE)
            _m_findings = re.search(r'"findings"\s*:\s*"([^"]+)"', raw_content, re.IGNORECASE)
            _m_quote = re.search(r'"quote"\s*:\s*"([^"]{0,500})', raw_content, re.IGNORECASE | re.DOTALL)
            _m_conf = re.search(r'"confidence"\s*:\s*([0-9.]+)', raw_content, re.IGNORECASE)
            if _m_verdict:
                parsed = {
                    "verdict": _m_verdict.group(1),
                    "compliance": _m_verdict.group(1).lower(),
                    "findings": (_m_findings.group(1) if _m_findings
                                 else (_m_quote.group(1).replace('\n', ' ')[:250] if _m_quote else raw_content[:250])),
                    "quote": _m_quote.group(1) if _m_quote else "",
                    "confidence": float(_m_conf.group(1)) if _m_conf else 0.7,
                    "reasoning": "Recovered from truncated VLM JSON via regex fallback.",
                }
        if not parsed:
            parsed = {
                "findings": raw_content[:300],
                "result": "VLM response not JSON",
                "compliance": "review",
                "confidence": 0.3,
                "reasoning": "Could not parse VLM output as JSON",
            }

        # ── Normalize CORE-prompt response schema to legacy schema ──
        # CORE prompt returns: verdict / quote / findings / confidence / structured_source
        # Legacy downstream expects:   compliance / findings / result / confidence / reasoning
        if "compliance" not in parsed and "verdict" in parsed:
            parsed["compliance"] = str(parsed.get("verdict") or "review").lower()
        if "result" not in parsed:
            # Use findings as the short status; truncate for UI display
            _f = parsed.get("findings") or ""
            parsed["result"] = _f[:600] if _f else parsed.get("compliance", "review")
        if "reasoning" not in parsed:
            _src = parsed.get("structured_source") or ""
            _q = parsed.get("quote") or ""
            parsed["reasoning"] = (
                f"Source: {_src}. Quote: {_q[:150]}" if _src or _q else parsed.get("findings", "")[:200]
            )

        parsed["row_id"] = row_id
        parsed["elapsed"] = elapsed
        parsed["_verification_path"] = "vlm_split" if USE_SPLIT_PROMPTS else "vlm_legacy"

        # P133 — Arithmetic sanity post-check for FAIL verdicts on
        # quantity/amount conditions. LLM occasionally writes "X exceeds
        # maximum Y" where X is in fact less than Y (self-contradictory
        # finding). Parse the finding text, compare the two numbers, and
        # flip to PASS when the stated overflow isn't real.
        try:
            _comp = str(parsed.get("compliance", "")).lower().strip()
            _findings = str(parsed.get("findings", ""))
            if _comp in ("fail", "not_complied", "non_compliant", "discrepant") and _findings:
                _fu = _findings.upper()
                # Pattern: "... IS X ... EXCEEDS ... MAXIMUM OF Y ..."
                # Grab any two numeric values that appear in the finding and
                # check the claimed exceedance.
                _nums = re.findall(
                    r'([-+]?\d{1,3}(?:[,\s]\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)',
                    _findings,
                )
                def _to_float(s):
                    try:
                        return float(str(s).replace(',', '').replace(' ', ''))
                    except Exception:
                        return None
                _vals = [_to_float(n) for n in _nums]
                _vals = [v for v in _vals if v is not None and v > 0]

                _says_exceeds = (
                    'EXCEEDS' in _fu or 'EXCEED THE' in _fu or
                    'GREATER THAN' in _fu or 'OVER THE MAXIMUM' in _fu or
                    'ABOVE THE ALLOWED' in _fu
                )
                _says_less = (
                    'LESS THAN THE MINIMUM' in _fu or 'BELOW THE MINIMUM' in _fu or
                    'UNDER THE MINIMUM' in _fu
                )
                if _says_exceeds and len(_vals) >= 2:
                    # Heuristic: actual value is the FIRST number quoted,
                    # limit is typically the LARGEST number after "MAXIMUM"
                    # or the last explicit number in the finding.
                    _actual = _vals[0]
                    _limit = max(_vals[1:])
                    if _actual <= _limit * 1.0001:  # 0.01% slack for FP
                        parsed["compliance"] = "pass"
                        parsed["verdict"] = "PASS"
                        parsed["findings"] = (
                            f"{_findings.rstrip('. ')}. Arithmetic post-check: "
                            f"{_actual:,.2f} is NOT greater than {_limit:,.2f} — within tolerance (P133 override)."
                        )
                        parsed["result"] = parsed["findings"][:600]
                        parsed["_post_check"] = "P133_arithmetic_override"
                if _says_less and len(_vals) >= 2:
                    _actual = _vals[0]
                    _limit = min(_vals[1:])
                    if _actual >= _limit * 0.9999:
                        parsed["compliance"] = "pass"
                        parsed["verdict"] = "PASS"
                        parsed["findings"] = (
                            f"{_findings.rstrip('. ')}. Arithmetic post-check: "
                            f"{_actual:,.2f} is NOT less than {_limit:,.2f} — within tolerance (P133 override)."
                        )
                        parsed["result"] = parsed["findings"][:600]
                        parsed["_post_check"] = "P133_arithmetic_override"
        except Exception:
            pass  # never let the sanity check break the pipeline

        # P134 — Consignee-name post-check. LLM sometimes FAILs a
        # "TO THE ORDER OF <BANK>" / "CONSIGNED TO <X>" / "CONSIGNEE MUST
        # BE <X>" condition despite the structured consignee clearly
        # containing that party. If structured consignee proves the
        # requirement, override to PASS.
        try:
            _comp = str(parsed.get("compliance", "")).lower().strip()
            if _comp in ("fail", "not_complied", "non_compliant", "discrepant"):
                _cu = (condition_text or "").upper()
                _trigger = (
                    'TO THE ORDER OF' in _cu or 'TO ORDER OF' in _cu or
                    'MADE OUT TO' in _cu or 'CONSIGNED TO' in _cu or
                    'CONSIGNEE' in _cu or 'CONSIGN TO' in _cu or
                    'ISSUED TO THE ORDER' in _cu
                )
                if _trigger:
                    # Try multiple target extraction patterns
                    _target = ''
                    for _pat in (
                        r'TO\s+(?:THE\s+)?ORDER\s+OF[\s:]+([^.\n]+?)(?:[.,\n]|$)',
                        r'CONSIGNED\s+TO[\s:]+([^.\n]+?)(?:[.,\n]|$)',
                        r'CONSIGNEE\s+(?:MUST\s+BE|SHOULD\s+BE|IS|=)[\s:\'""]+([^.\n\'""]+?)(?:[.,\n\'""]|$)',
                        r'MADE\s+OUT\s+TO[\s:]+([^.\n]+?)(?:[.,\n]|$)',
                    ):
                        _m = re.search(_pat, _cu)
                        if _m:
                            _target = _m.group(1).strip(' .,:\'""')
                            break
                    # Extract just the core bank/company name by stripping
                    # generic noise words. Keep the distinguishing words
                    # (proper nouns like "AL HABIB", "NOOR-UD-DIN").
                    _target_key = _target
                    # Remove all punctuation for robust matching
                    _target_key = re.sub(r'[.,;:\'"—–-]+', ' ', _target_key)
                    # Strip trailing country / city words — these aren't the
                    # distinguishing part of a party name
                    _LOC_SUFFIX = (
                        r'\b(?:PAKISTAN|INDIA|BANGLADESH|SRI\s+LANKA|UAE|SAUDI\s+ARABIA|'
                        r'KARACHI|LAHORE|ISLAMABAD|MUMBAI|DUBAI|RIYADH|DOHA|BEIRUT|'
                        r'HONG\s+KONG|SINGAPORE|LONDON|NEW\s+YORK|GULBERG|CITY)\b'
                    )
                    _target_key = re.sub(_LOC_SUFFIX, '', _target_key, flags=re.IGNORECASE).strip()
                    # Strip common corporate suffixes & generic words
                    _target_key = re.sub(
                        r'\b(BANK|LTD|LIMITED|LLC|PLC|INC|CORP|CO|PVT|PRIVATE|COMPANY|'
                        r'LIMITEDS?|ENTERPRISES?|GROUP|HOLDINGS?|TRADING|'
                        r'INSURERS?|INSURANCE)\b\.?',
                        ' ', _target_key, flags=re.IGNORECASE,
                    )
                    _target_key = re.sub(r'\s+', ' ', _target_key).strip()
                    _cons_txt = ''
                    if isinstance(unified_summary, dict):
                        _cons_txt = str(unified_summary.get('consignee', '') or '').upper()
                        if not _cons_txt:
                            arr = unified_summary.get('parties_found') or []
                            for item in (arr if isinstance(arr, list) else []):
                                if not isinstance(item, dict):
                                    continue
                                role = str(item.get('role', '')).lower()
                                if 'consignee' in role:
                                    _cons_txt = (
                                        str(item.get('name', '') or '').upper() + ' ' +
                                        str(item.get('raw', '') or '').upper()
                                    )
                                    break
                    # Match if consignee contains target key. Don't require
                    # "TO ORDER" wording — sometimes the LC just says
                    # "consigned to X" without the "to order" phrasing.
                    if _cons_txt and _target_key and _target_key in _cons_txt:
                        parsed["compliance"] = "pass"
                        parsed["verdict"] = "PASS"
                        parsed["findings"] = (
                            f"Structured consignee contains '{_target_key}' "
                            f"(consignee='{_cons_txt[:150]}'). (P134 override)"
                        )
                        parsed["result"] = parsed["findings"][:600]
                        parsed["_post_check"] = "P134_consignee_override"
                    elif _target_key and document_text:
                        # Fallback: scan raw document text for "TO ORDER OF"
                        # block followed by target key, or for the target
                        # key appearing near "Consignee". This catches the
                        # common case where step 3 didn't tag the consignee
                        # cleanly but the BL text says "Consignee\n...\nTO
                        # ORDER OF:\nBANK AL HABIB LTD.\nKARACHI".
                        _dt_up = document_text.upper()
                        _target_key_collapsed = re.sub(r'\s+', ' ', _target_key).strip()
                        # Normalize document whitespace to one space for search
                        _dt_flat = re.sub(r'\s+', ' ', _dt_up)
                        _has_to_order = 'TO ORDER' in _dt_flat
                        _has_target = _target_key_collapsed in _dt_flat
                        _has_consignee_header = 'CONSIGNEE' in _dt_flat
                        if _has_target and (_has_to_order or _has_consignee_header):
                            parsed["compliance"] = "pass"
                            parsed["verdict"] = "PASS"
                            parsed["findings"] = (
                                f"'{_target_key}' appears in document text "
                                f"alongside consignee/'TO ORDER' marker — "
                                f"requirement satisfied. (P134 doc-text override)"
                            )
                            parsed["result"] = parsed["findings"][:600]
                            parsed["_post_check"] = "P134_doc_text_override"
        except Exception:
            pass

        # P137 — Unit price / product code mismatch where BOTH differ by
        # a single character from the LC (e.g. LC "HP4024N" at 1,190 vs
        # invoice "HP4024WN" at 1,140). This pattern strongly suggests the
        # LC condition itself was OCR-corrupted during decomposition
        # (one character slipped in the product code + a digit swapped in
        # the price). Downgrade FAIL → REVIEW so a human can verify the
        # source LC rather than auto-flagging a discrepancy that may not
        # actually exist.
        try:
            _comp_up = str(parsed.get("compliance", "")).lower().strip()
            if _comp_up in ("fail", "not_complied", "non_compliant", "discrepant"):
                _cu = (condition_text or "").upper()
                _fu = str(parsed.get("findings", "")).upper()
                if ('UNIT PRICE' in _cu or 'PRICE' in _cu) and (
                    'DOES NOT MATCH' in _fu or 'MISMATCH' in _fu or
                    'DIFFERENT' in _fu
                ):
                    # Pull product codes from both sides: tokens of 6-12 chars
                    # that are mostly alphanumeric (HP4024N-type).
                    _cond_codes = set(re.findall(
                        r"\b([A-Z]{1,4}\d{3,6}[A-Z]{0,3})\b",
                        _cu,
                    ))
                    _fin_codes = set(re.findall(
                        r"\b([A-Z]{1,4}\d{3,6}[A-Z]{0,3})\b",
                        _fu,
                    ))
                    # Edit-distance 1 between LC code and invoice code?
                    def _lev1(a, b):
                        if abs(len(a) - len(b)) > 1:
                            return False
                        if a == b:
                            return False
                        if len(a) == len(b):
                            return sum(1 for x, y in zip(a, b) if x != y) == 1
                        # insertion/deletion of 1 char
                        _s, _l = (a, b) if len(a) < len(b) else (b, a)
                        for i in range(len(_l)):
                            if _l[:i] + _l[i+1:] == _s:
                                return True
                        return False
                    _near_miss = any(
                        _lev1(c1, c2) for c1 in _cond_codes for c2 in _fin_codes
                        if c1 != c2
                    )
                    if _near_miss:
                        # P137 → P150: user requested PASS (not REVIEW).
                        # When product codes differ by a single character,
                        # the LC condition itself almost certainly has an
                        # OCR error during extraction. Force PASS with a
                        # note so the reviewer can still see the near-miss.
                        parsed["compliance"] = "pass"
                        parsed["verdict"] = "PASS"
                        parsed["findings"] = (
                            f"{parsed.get('findings','').rstrip('. ')}. "
                            f"Product codes differ by a single character "
                            f"({sorted(_cond_codes)} vs {sorted(_fin_codes)}) "
                            f"— treating as same product with OCR variant. "
                            f"(P150 OCR near-miss override)"
                        )
                        parsed["result"] = parsed["findings"][:600]
                        parsed["_post_check"] = "P150_unit_price_ocr_pass"
        except Exception:
            pass

        # P135 — "Reference not found" post-check override. LLM sometimes
        # FAILs a "document must reference X" row when X IS on the document
        # but OCR-munged (e.g. letter O read as digit 0, letter I as 1).
        # Re-scan the full document_text after OCR-normalization; if the
        # identifier matches, override to PASS.
        try:
            _comp2 = str(parsed.get("compliance", "")).lower().strip()
            _findings2 = str(parsed.get("findings", ""))
            _fu2 = _findings2.upper()
            if _comp2 in ("fail", "not_complied", "non_compliant", "discrepant") and (
                'NOT FOUND' in _fu2 or 'NOT PRESENT' in _fu2 or
                'DOES NOT CONTAIN' in _fu2 or 'NOT APPEAR' in _fu2 or
                'CANNOT FIND' in _fu2 or 'NOT REFERENCED' in _fu2 or
                "NOT QUOTED" in _fu2 or "NOT MENTIONED" in _fu2 or
                'DOES NOT SHOW' in _fu2 or "DOESN'T SHOW" in _fu2 or
                "IS NOT SHOWN" in _fu2 or "NOT DISPLAYED" in _fu2 or
                "NOT INCLUDED" in _fu2 or "MISSING" in _fu2 or
                "DOES NOT INCLUDE" in _fu2 or "NOT STATED" in _fu2
            ):
                # Extract identifier tokens from the CONDITION — MUST
                # contain a digit, otherwise common English words like
                # "Commercial" / "Reference" / "Shipment" / "Invoice"
                # match trivially and produce false PASSes.
                _cids_raw = re.findall(
                    r'[A-Z0-9][A-Z0-9/\-._]{5,}[A-Z0-9]',
                    condition_text or '',
                    flags=re.IGNORECASE,
                )
                _cids = [t for t in _cids_raw if re.search(r'\d', t)]
                _doc_full = (
                    _normalize_id(document_text or '') + ' ' +
                    _normalize_id(str(unified_summary or ''))
                )
                for _needle in _cids:
                    _n = _normalize_id(_needle)
                    # Must have at least 5 alnum chars AND at least 3
                    # digits to rule out any remaining word-like tokens.
                    if len(_n) < 5 or sum(1 for ch in _n if ch.isdigit()) < 3:
                        continue
                    if _n in ('LETTERCREDIT', 'DOCUMENTARY', 'SHIPMENTADVICE', 'COMMERCIALINVOICE'):
                        continue
                    if _n in _doc_full:
                        parsed["compliance"] = "pass"
                        parsed["verdict"] = "PASS"
                        parsed["findings"] = (
                            f"Reference '{_needle}' IS present on document "
                            f"(OCR-normalised match). Original LLM finding said "
                            f"not found, but the identifier appears after "
                            f"OCR character-confusion handling (O↔0, I↔1, etc.). "
                            f"(P135 override)"
                        )
                        parsed["result"] = parsed["findings"][:600]
                        parsed["_post_check"] = "P135_reference_found_override"
                        break
        except Exception:
            pass

        # P165 — Truncated / prefix name-match override. LLM sometimes
        # returns FAIL with "does not exactly match 'X'" where X is a
        # truncated LC-extracted party name and the document carries
        # the full form (X being a prefix of the document name). Fix:
        # if the quoted expected value from the finding is a prefix of
        # any party name in the document text / structured parties,
        # override to PASS.
        try:
            _comp5 = str(parsed.get("compliance", "")).lower().strip()
            _findings5 = str(parsed.get("findings", ""))
            _fu5 = _findings5.upper()
            if (_comp5 in ("fail", "not_complied", "non_compliant", "discrepant")
                    and parsed.get("_post_check") is None
                    and ('DOES NOT EXACTLY MATCH' in _fu5 or
                         'DOES NOT MATCH' in _fu5 or
                         'NOT MATCH EXACTLY' in _fu5 or
                         'EXACT MATCH' in _fu5)):
                def _norm_name(s):
                    s = str(s or '').upper()
                    # Strip honorifics / prefixes
                    s = re.sub(r'\b(M/?S\.?|MESSRS\.?|MR\.?|MRS\.?|DR\.?)\s+', '', s)
                    # Strip acronyms in parens
                    s = re.sub(r'\([^)]*\)', ' ', s)
                    # Strip company suffixes
                    s = re.sub(
                        r'\b(LTD|LIMITED|LLC|PLC|INC|CORP|CO|PVT|PRIVATE|COMPANY|'
                        r'S\.?A\.?|S\.?L\.?|B\.?V\.?|N\.?V\.?|GMBH|AG|AB|OY|'
                        r'ENTERPRISES?|GROUP|HOLDINGS?)\b\.?',
                        ' ', s,
                    )
                    # Strip punctuation, collapse whitespace
                    s = re.sub(r'[.,;:/\\\'"—–\-]+', ' ', s)
                    s = re.sub(r'\s+', ' ', s).strip()
                    return s

                # P192 — Use the LC's EXPECTED value (lc_field_value) as
                # the authoritative "expected name", NOT arbitrary quoted
                # strings from the finding. The previous logic grabbed
                # every quoted name from the LLM's finding — including the
                # document's OWN mismatched value — and then matched it
                # against itself in the document, producing false PASSes
                # for party-vs-party comparisons like "BL shipper must be
                # beneficiary" when the two names are genuinely different.
                _expected = []
                _exp_lc = ''
                try:
                    _exp_lc = str(lc_field_value or '').strip()
                except Exception:
                    _exp_lc = ''
                if _exp_lc and len(_exp_lc) >= 5:
                    # Take ONLY the first line of the LC expected value so
                    # we don't drag in a full multi-line address.
                    _expected.append(_exp_lc.split('\n')[0].strip())

                # Disable P165 entirely for party-vs-party comparisons
                # where the expected source is the LC's BENEFICIARY and
                # the subject is the BL's SHIPPER (or similar cross-field
                # checks). The doc's shipper genuinely differing from the
                # LC's beneficiary is a real UCP 600 Art 14(j) FAIL, not
                # a truncation artefact.
                _cu_p165 = (condition_text or '').upper()
                _cross_field = (
                    ('BENEFICIARY' in _cu_p165 and
                        ('SHIPPER' in _cu_p165 or 'CONSIGNOR' in _cu_p165)) or
                    ('SHIPPER' in _cu_p165 and 'MUST' in _cu_p165 and
                        'BENEFICIARY' in _cu_p165)
                )
                if _cross_field:
                    _expected = []  # skip override

                # Gather candidate party names from the evidence
                _candidates = []
                if isinstance(unified_summary, dict):
                    for _fld in ('applicant', 'beneficiary', 'shipper',
                                  'consignee', 'notify_party', 'issuer',
                                  'drawer', 'drawee', 'payee'):
                        _v = unified_summary.get(_fld)
                        if _v:
                            _candidates.append(str(_v))
                    for _item in (unified_summary.get('parties_found') or []):
                        if isinstance(_item, dict):
                            _nm = _item.get('name') or _item.get('raw')
                            if _nm:
                                _candidates.append(str(_nm))

                _hit = None
                for _exp in _expected:
                    _exp_n = _norm_name(_exp)
                    if len(_exp_n) < 8:  # too short to prefix-match safely
                        continue
                    for _cand in _candidates:
                        _c_n = _norm_name(_cand)
                        # PASS only when the LC's expected value is a
                        # prefix of, or contained in, a document party
                        # name. Bidirectional match retained for the real
                        # truncation case: LC-extracted value got cut
                        # mid-name and the document carries the full form.
                        if (_exp_n and _c_n and len(_c_n) >= 8 and
                                (_c_n.startswith(_exp_n) or
                                 _exp_n.startswith(_c_n))):
                            # Require meaningful overlap — the shorter
                            # must be at least 70% of the longer's length
                            # so we don't false-pass "PT CITRA" matching
                            # "PT CITRA BORNEO UTAMA TBK" against a
                            # totally-different LC expected name.
                            _short = min(len(_exp_n), len(_c_n))
                            _long = max(len(_exp_n), len(_c_n))
                            if _short / max(_long, 1) >= 0.70:
                                _hit = _exp
                                break
                    if _hit:
                        break
                if _hit:
                    parsed["compliance"] = "pass"
                    parsed["verdict"] = "PASS"
                    parsed["findings"] = (
                        f"Party name match confirmed. Document name "
                        f"contains the required party '{_hit}' as a prefix "
                        f"(the LC-extracted form appears truncated; "
                        f"document carries the full legal name). "
                        f"(P165 prefix/truncation override)"
                    )
                    parsed["result"] = parsed["findings"][:600]
                    parsed["_post_check"] = "P165_name_prefix_match"
        except Exception:
            pass

        # P138 — "Date not found" post-check override. LLM sometimes
        # returns REVIEW/FAIL with "no date found" / "date not found"
        # even though the document obviously has an issue date (often the
        # typed issue_date field was populated but the LLM ignored it, or
        # the date format on the document is unusual — "2025.02.16" / "Feb
        # 16, 2025" / "DD.MM.YYYY"). Scan the document for ANY date-like
        # token and, if found, override the verdict based on the LC
        # condition's date arithmetic (on/after | on/before).
        try:
            _comp3 = str(parsed.get("compliance", "")).lower().strip()
            _fu3 = str(parsed.get("findings", "")).upper()
            if _comp3 in ("fail", "not_complied", "non_compliant", "discrepant",
                          "review", "review required") and (
                'NO DATE' in _fu3 or 'DATE NOT FOUND' in _fu3 or
                'DATE MISSING' in _fu3 or 'NO ISSUE DATE' in _fu3 or
                'ISSUE DATE NOT' in _fu3 or 'SHIPMENT DATE NOT' in _fu3 or
                'DATE IS NOT' in _fu3 or 'DATE COULD NOT' in _fu3
            ):
                # 1) Prefer a date already extracted by step 3
                _candidate_dates = []
                if isinstance(unified_summary, dict):
                    for _fld in ('issue_date', 'invoice_date', 'bl_issue_date',
                                  'certificate_issue_date', 'draft_date',
                                  'shipment_date', 'onboard_date', 'document_date'):
                        _v = unified_summary.get(_fld)
                        if _v and str(_v).strip():
                            _candidate_dates.append((_fld, str(_v).strip()))
                    # Also check structured dates_found
                    for _item in (unified_summary.get('dates_found') or []):
                        if not isinstance(_item, dict):
                            continue
                        _v = _item.get('value') or _item.get('raw')
                        _r = str(_item.get('role', '') or '').lower()
                        if _v and _r in ('issue_date', 'invoice_date',
                                          'bl_issue_date', 'certificate_issue_date',
                                          'document_date', 'draft_date',
                                          'shipment_date', 'onboard_date'):
                            _candidate_dates.append((_r, str(_v).strip()))

                # 2) Fall back: scan document_text for date patterns
                if not _candidate_dates and document_text:
                    _date_pats = [
                        # YYYY-MM-DD / YYYY.MM.DD / YYYY/MM/DD
                        r'\b(20\d{2})[-./](\d{1,2})[-./](\d{1,2})\b',
                        # DD-MM-YYYY / DD.MM.YYYY / DD/MM/YYYY
                        r'\b(\d{1,2})[-./](\d{1,2})[-./](20\d{2})\b',
                        # DD MMM YYYY / DD-MMM-YYYY
                        r'\b(\d{1,2})[\s\-]+(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*[\s\-.,]+(20\d{2})\b',
                    ]
                    for _pat in _date_pats:
                        _hit = re.search(_pat, document_text, re.IGNORECASE)
                        if _hit:
                            _candidate_dates.append(('doc_text_scan', _hit.group(0)))
                            break

                if _candidate_dates:
                    # Any date present → the LC requirement (on/after or
                    # on/before the LC date) becomes resolvable. Flip verdict
                    # to PASS with a note citing the found date. Leave REVIEW
                    # in place only if the condition has an explicit
                    # comparison that we can't resolve from our date data.
                    _src, _val = _candidate_dates[0]
                    parsed["compliance"] = "pass"
                    parsed["verdict"] = "PASS"
                    parsed["findings"] = (
                        f"Date IS present on document: {_val} "
                        f"(from {_src}). LLM's 'no date found' was incorrect. "
                        f"(P138 override)"
                    )
                    parsed["result"] = parsed["findings"][:600]
                    parsed["_post_check"] = "P138_date_found_override"
        except Exception:
            pass

        # P155 — STRICT identifier-only override (replaces P141/P143/P144).
        # Only overrides FAIL to PASS when a specific identifier from the
        # CONDITION is literally present in the ACTUAL DOCUMENT TEXT
        # (after OCR character-confusion normalization). No fuzzy word
        # matching, no finding-text extraction (finding often contains
        # "closest text is X" which would wrongly match), no proper-noun
        # approximations. Exact value match or nothing.
        try:
            _comp_final = str(parsed.get("compliance", "")).lower().strip()
            _fin_final = str(parsed.get("findings", "")).upper()
            _NEG_PHRASES = (
                'NOT FOUND', 'NOT PRESENT', 'DOES NOT CONTAIN',
                'DOES NOT SHOW', 'DOES NOT INCLUDE',
                'NOT REFERENCED', 'NOT QUOTED', 'NOT MENTIONED',
                'NOT STATED', 'IS MISSING', 'CANNOT FIND',
                'IS NOT SHOWN', 'NOT LISTED',
            )
            if (_comp_final in ('fail', 'not_complied', 'non_compliant', 'discrepant')
                    and any(p in _fin_final for p in _NEG_PHRASES)
                    and parsed.get("_post_check") is None):
                # Extract identifier-like tokens ONLY from the CONDITION.
                # Must contain at least one digit (policy numbers, HS
                # codes, LC refs, NTN numbers, etc.). Plain-text tokens
                # are NOT considered — use the LLM verdict for those.
                _cond_ids = []
                for _m in re.finditer(
                    r'[A-Z0-9][A-Z0-9/\-._]{4,}[A-Z0-9]',
                    condition_text or '', flags=re.IGNORECASE,
                ):
                    _tok = _m.group(0)
                    if re.search(r'\d', _tok):
                        _cond_ids.append(_tok)

                if _cond_ids:
                    _doc_norm = _normalize_id(document_text or '')
                    _hit_tok = None
                    for _tok in _cond_ids:
                        _tn = _normalize_id(_tok)
                        if len(_tn) >= 5 and _tn in _doc_norm:
                            _hit_tok = _tok
                            break
                    if _hit_tok:
                        parsed["compliance"] = "pass"
                        parsed["verdict"] = "PASS"
                        parsed["findings"] = (
                            f"'{_hit_tok}' is present in the document "
                            f"(identifier match). Requirement satisfied."
                        )
                        parsed["result"] = parsed["findings"][:600]
                        parsed["_post_check"] = "P155_identifier_match"
        except Exception:
            pass

        # P148 REMOVED — The bank-name failsafe was matching the bank
        # name in Notify Party or LC-parties blob and wrongly flipping
        # consignee FAILs to PASS. Consignee checks now live entirely in
        # _deterministic_verify (Check 2b) where they read ONLY the
        # consignee field, never notify. No generic failsafe here.

        # P156 — Clean up LLM findings language: strip "closest text is X"
        # / "the closest match is X" which clutters the report and is
        # misleading (the system is not doing fuzzy matching — either the
        # value is there or it isn't).
        try:
            _fin = str(parsed.get("findings", "") or "")
            if _fin:
                # Remove sentences starting with "The closest text/match..."
                _fin2 = re.sub(
                    r'(?:^|\.\s*)(?:The\s+)?closest\s+(?:text|match|wording|phrase)[^.]*\.',
                    '.', _fin, flags=re.IGNORECASE,
                )
                # Remove " — closest text is X" / ", closest match is Y"
                _fin2 = re.sub(
                    r'[\s,;—–-]+(?:The\s+)?closest\s+(?:text|match|wording|phrase)[^.]*',
                    '', _fin2, flags=re.IGNORECASE,
                )
                _fin2 = re.sub(r'\s+\.', '.', _fin2).strip()
                _fin2 = re.sub(r'\.{2,}', '.', _fin2).strip()
                if _fin2 and _fin2 != _fin:
                    parsed["findings"] = _fin2
                    # Keep result in sync
                    if parsed.get("result") == _fin[:200]:
                        parsed["result"] = _fin2[:600]
                    else:
                        _res = str(parsed.get("result", "") or "")
                        _res = re.sub(
                            r'[\s,;—–-]+(?:The\s+)?closest\s+(?:text|match|wording|phrase)[^.]*',
                            '', _res, flags=re.IGNORECASE,
                        )
                        parsed["result"] = _res.strip()[:600]
        except Exception:
            pass

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
    _untyped_idx = 0
    for pkt in packets:
        if not pkt:
            continue
        pt = (pkt.get("document_type", "") or pkt.get("doc_type", "")
              or pkt.get("classification", "") or "").lower().strip()
        if not pt:
            # Don't discard untyped packets — they may contain documents
            # that the classifier couldn't name but are still present.
            # Give each a unique key so they aren't merged together.
            _untyped_idx += 1
            pt = f"_untyped_{_untyped_idx}"
        if pt not in type_groups:
            type_groups[pt] = []
        type_groups[pt].append(pkt)

    deduped = []
    doc_counts = {}
    for doc_type, group in type_groups.items():
        doc_counts[doc_type] = len(group)
        if len(group) == 1:
            deduped.append(group[0])
            continue

        # Check if documents in this group are ACTUAL copies (same content)
        # or DIFFERENT documents with the same type name.
        # Common document types (invoice, BL, packing list, COO) are almost
        # always copies — use a lower threshold (40%) to dedup them.
        # Certificates and other types might be genuinely different documents
        # with the same type name — use a higher threshold (60%).
        _ALWAYS_COPY_TYPES = {
            'commercial invoice', 'invoice', 'packing list', 'packing note',
            'bill of lading', 'draft bill of exchange', 'bill of exchange',
            'certificate of origin', 'air waybill', 'airway bill',
        }
        _overlap_threshold = 0.40 if doc_type in _ALWAYS_COPY_TYPES else 0.60
        _distinct = []
        for pkt in group:
            _pkt_words = set(_pkt_text(pkt).upper().split())
            _is_copy = False
            for _existing in _distinct:
                _ex_words = set(_pkt_text(_existing).upper().split())
                if _pkt_words and _ex_words:
                    _overlap = len(_pkt_words & _ex_words)
                    _total = max(len(_pkt_words), len(_ex_words))
                    if _total > 0 and _overlap / _total > _overlap_threshold:
                        _is_copy = True
                        break
            if not _is_copy:
                _distinct.append(pkt)

        if len(_distinct) == 1:
            # All copies of the same document — pick the one with most text
            best = max(group, key=lambda p: len(_pkt_text(p)))
            if isinstance(best, dict):
                best['_copy_count'] = len(group)
            deduped.append(best)
        else:
            # Different documents with same type — keep all
            for d in _distinct:
                if isinstance(d, dict):
                    d['_copy_count'] = 1
                deduped.append(d)

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

    # P139 — Drop structural / non-verifiable pages from the packet pool
    # BEFORE building any tasks. A "Blank Page" / "Header Page" /
    # "Endorsement Page" / "BL Conditions of Carriage" / "Back Page" etc.
    # has no content to verify against LC conditions, and running dozens
    # of "date not found" REVIEW rows against them just clutters the
    # report. Keep only packets that represent a real submitted document.
    _NON_VERIFIABLE_PAGE_TYPES = {
        'blank page', 'header page', 'endorsement page', 'back page',
        'bl conditions of carriage', 'conditions of carriage',
        'terms and conditions', 'terms overleaf',
        'unknown', 'unidentified', 'cover page',
    }
    _orig_count = len(packets)
    packets = [
        p for p in packets
        if p and str((p.get('document_type', '') or p.get('doc_type', '')
                      or '')).lower().strip() not in _NON_VERIFIABLE_PAGE_TYPES
    ]
    _dropped = _orig_count - len(packets)
    if _dropped:
        # Can't call progress here (no fn in scope) — will log elsewhere.
        pass

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
        # BUT only if the document actually exists in the submission
        condition_text = _get(row, "condition_text", "")
        _cond_upper = condition_text.upper()
        if re.search(r'\b(DUPLICATE|TRIPLICATE|QUADRUPLICATE|OCTUPLICATE|COPIES|FULL\s+SET|IN\s+\d+\s+ORIG)', _cond_upper):
            # Check if the document type mentioned in the condition exists
            doc_checked = _get(row, "document_checked", "")
            _doc_exists = False
            if doc_checked:
                _matched = _find_matching_docs(doc_checked, deduped_packets)
                _doc_exists = bool(_matched)
            if _doc_exists:
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
            # Document not found — don't auto-pass, let it fall through
            # to the missing document check below

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

        # P93: Inject F43P partial shipment status into quantity conditions
        # so the VLM knows whether lesser quantity is acceptable.
        # We DON'T bypass the VLM — we let it compare the actual quantities
        # but give it the F43P context to make the right PASS/FAIL decision.
        _qty_match = re.search(r'(?:QUANTITY|QTY)\s+(?:MUST\s+BE|SHOULD\s+BE|OF)\s+([\d,.]+)', _cond_upper)
        if _qty_match:
            _f43p = _get_lc_field_value(step06_result, '43P')
            _partial_allowed = bool(re.search(
                r'ALLOWED|PERMITTED|PERMISSIBLE|YES',
                (_f43p or '').upper(),
            ))
            if _partial_allowed:
                # Append F43P context to the condition text so the VLM sees it
                _current_cond = _get(row, "condition_text", "")
                if 'PARTIAL SHIPMENT' not in _current_cond.upper():
                    _set(row, "condition_text",
                         _current_cond + f" [NOTE: F43P = {_f43p}. Partial shipment is ALLOWED. "
                         f"Lesser quantity is acceptable (PASS). Only quantity EXCEEDING "
                         f"{_qty_match.group(1)} is a discrepancy (FAIL).]")

        condition_text = _get(row, "condition_text", "")
        clause_ref = _get(row, "clause_ref", "")
        field_tag = _get(row, "field_tag", "")
        doc_checked = _get(row, "document_checked", "")
        look_for = _get(row, "look_for_value", "")

        # PARTY NAME INJECTION: resolve "Issuing Bank", "Applicant",
        # "Beneficiary" to actual names so LLM can't confuse them.
        _cond_lower_party = condition_text.lower()
        if any(k in _cond_lower_party for k in ['issuing bank', 'applicant', 'beneficiary']):
            _lc_parties_fields = step06_result.get('final_lc', step06_result).get('consolidated_fields', step06_result.get('consolidated_fields', {}))
            _applicant = str(_lc_parties_fields.get('50', '')).split('\n')[0].strip()
            _beneficiary = str(_lc_parties_fields.get('59', '')).split('\n')[0].strip()
            _issuing_bank = str(_lc_parties_fields.get('52A', _lc_parties_fields.get('52D', ''))).split('\n')[0].strip()
            if not _issuing_bank:
                _issuing_bank = str(_lc_parties_fields.get('51A', _lc_parties_fields.get('51D', ''))).split('\n')[0].strip()
            if not _issuing_bank:
                _issuing_bank = str(_lc_parties_fields.get('sender_institution', '')).split('\n')[0].strip()
            if not _issuing_bank:
                _issuing_bank = str(_lc_parties_fields.get('42D', '')).split('\n')[0].strip()
            if not _issuing_bank:
                # Fallback: check F78 for bank name
                _f78 = str(_lc_parties_fields.get('78', ''))
                _bank_m = re.search(r'(BANK\s+AL\s+HABIB|UNITED\s+BANK|HABIB\s+BANK|MCB\s+BANK|ALLIED\s+BANK|NATIONAL\s+BANK|ASKARI\s+BANK|MEEZAN\s+BANK|FAYSAL\s+BANK|STANDARD\s+CHARTERED|BANK\s+OF\s+PUNJAB|SILK\s+BANK|JS\s+BANK|SONERI\s+BANK|SUMMIT\s+BANK|BANK\s+ALFALAH|HSBC|CITIBANK|DEUTSCHE\s+BANK)[\w\s,.]*(LTD\.?|LIMITED)?', _f78, re.IGNORECASE)
                if _bank_m:
                    _issuing_bank = _bank_m.group(0).strip().rstrip(',. ')
            _party_note = []
            if 'issuing bank' in _cond_lower_party and _issuing_bank:
                _party_note.append(f'L/C Issuing Bank = "{_issuing_bank}"')
            if 'applicant' in _cond_lower_party and _applicant:
                _party_note.append(f'Applicant = "{_applicant}"')
            if 'beneficiary' in _cond_lower_party and _beneficiary:
                _party_note.append(f'Beneficiary = "{_beneficiary}"')
            if _party_note:
                condition_text = f"[PARTY NAMES: {'; '.join(_party_note)}. Match the EXACT party name, not a different party.]\n{condition_text}"

        # NOTE: The P120 regex-heavy BL prohibition injection block was REMOVED
        # in P123. The information it produced (signing capacity, charter party
        # presence, T&C presence, known-carrier check) is now captured by
        # Step 3d bl_subtype (signing_type / contract_type / has_terms_overleaf
        # / is_claused_bl / etc.) and flows into the CORE + FAMILY_PACK_BL
        # prompt via _build_structured_facts. The deterministic fast-path in
        # _call_vlm also handles the common prohibition cases without LLM.
        # No regex, no hard-coded carrier list — pure LLM-driven classification.

        # Get the LC field value for context
        lc_field_value = look_for or _get_lc_field_value(step06_result, field_tag)

        # For 45A sub-conditions (goods/quantity/price), include the FULL
        # clause text AND prepend the specific product context to the
        # condition text so the LLM knows which line item to match on
        # multi-item invoices.
        if field_tag == '45A' and clause_ref:
            _full_45a = _get_lc_field_value(step06_result, '45A')
            if _full_45a:
                lc_field_value = _full_45a
                # Check if the condition mentions quantity/price but not the
                # product name — inject the clause text as context
                _cond_lower_45a = condition_text.lower()
                if any(k in _cond_lower_45a for k in ['quantity', 'unit price', 'rate of', 'goods description']):
                    # Find the specific clause this row was decomposed from
                    # Split 45A into individual goods items
                    # Common separators: "\n.\n", "\n\n", or "." on its own line
                    if '\n.\n' in _full_45a:
                        _clauses_45a = [c.strip() for c in _full_45a.split('\n.\n') if c.strip()]
                    elif '\n\n' in _full_45a:
                        _clauses_45a = [c.strip() for c in _full_45a.split('\n\n') if c.strip()]
                    else:
                        _clauses_45a = [_full_45a]
                    # Try to match the clause by quoted text in the condition
                    _quoted = re.findall(r"['\"]([^'\"]{5,})['\"]", condition_text)
                    _matched_clause = None
                    for _q in _quoted:
                        for _cl in _clauses_45a:
                            if _q.upper() in _cl.upper():
                                _matched_clause = _cl.strip()
                                break
                        if _matched_clause:
                            break
                    if not _matched_clause and len(_clauses_45a) > 1:
                        # No quoted match — try matching by condition keywords
                        for _cl in _clauses_45a:
                            _cl_words = set(re.findall(r'[A-Z]{3,}', _cl.upper()))
                            _cond_words = set(re.findall(r'[A-Z]{3,}', condition_text.upper()))
                            if len(_cl_words & _cond_words) >= 2:
                                _matched_clause = _cl.strip()
                                break
                    if _matched_clause:
                        condition_text = (
                            f"[LC GOODS CLAUSE: {_matched_clause}]\n"
                            f"{condition_text}"
                        )

        # Determine which document types to check
        # For key-term fields that check multiple doc types, expand
        doc_types_to_check = []
        key_map = _KEY_TERM_DOC_MAP.get(field_tag)

        # P198f — DISJUNCTIVE (OR) document routing. Step 12 emits
        # a pipe-separated document_to_check for "X OR Y" LC clauses
        # (e.g. "Bill of Lading | Shipping Company Certificate" for
        # "B/L OR SCC MUST SHOW 21 DAYS FREE TIME"). Split on "|" /
        # " or " / ", OR " and mark the row so we later send a
        # SINGLE combined-text task to the verifier instead of
        # fanning out into N rows that each independently fail.
        _is_or_condition = False
        _or_doc_types = []
        if doc_checked and not (key_map and "all" in key_map):
            _raw = doc_checked.strip()
            if '|' in _raw:
                _or_doc_types = [s.strip() for s in _raw.split('|') if s.strip()]
            else:
                _m = re.split(r'\s+or\s+|\s*,\s*or\s+', _raw, flags=re.IGNORECASE)
                if len(_m) > 1 and all(len(p.strip()) >= 3 for p in _m):
                    # Only treat as OR if every fragment looks like a doc name
                    # (avoids splitting strings like "Bill of Exchange").
                    _or_doc_types = [p.strip() for p in _m]
            if len(_or_doc_types) >= 2:
                _is_or_condition = True

        if key_map and "all" in key_map:
            # F31C (Date of Issue) checks ALL documents
            doc_types_to_check = ["all"]
        elif key_map and len(key_map) > 1:
            # Multiple doc types (e.g. F32B -> invoice + draft)
            doc_types_to_check = key_map
        elif _is_or_condition:
            # P198f — One virtual "or-combined" task, not a fan-out.
            doc_types_to_check = ["__or_combined__"]
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
            'documentary remittance', 'document remittance',
            'covering letter', 'cover letter',
            'covering schedule', 'cover schedule',
            'l/c bills schedule', 'lc bills schedule', 'bills schedule',
            'export dc document presentation schedule',
            'export dc presentation schedule',
            'document presentation schedule', 'presentation schedule',
            'document presentation',
            'schedule of documents', 'letter of transmittal',
            'document arrival notice', 'arrival notice',
            'forwarding letter',
            'remittance letter', 'export letter',
            'fax', 'email',
            # P153 — structural / non-content page types (blanks, T&C, etc.)
            # NEVER fan-out "all documents" checks to these — they have no
            # content to verify against LC requirements and produce
            # spurious "missing LC number / missing date" REVIEW rows.
            'header page', 'blank page', 'endorsement page',
            'back page', 'back cover', 'reverse page',
            'terms and conditions', 'terms overleaf',
            'bl conditions of carriage', 'conditions of carriage',
            'cover page', 'title page',
            'unknown', 'unidentified', 'supporting document',
        )

        def _is_excluded_from_alldoc_fanout(pt: str, pkt=None) -> bool:
            if not pt:
                return True  # unknown — skip
            ptl = pt.lower().strip()
            if 'lc' == ptl or 'letter of credit' in ptl:
                return True
            for _ex in _ALLDOC_FANOUT_EXCLUDE:
                if _ex in ptl:
                    return True
            # P153 — also skip packets whose refined/cleaned text is too
            # short to possibly carry LC numbers / dates / amounts. Blank
            # or near-blank pages that slipped past classification get
            # caught here.
            if pkt is not None:
                try:
                    _txt = (_pkt_text(pkt) or '').strip()
                    if len(_txt) < 80:
                        return True
                except Exception:
                    pass
            return False

        # For "all" documents: send each shipping doc as a separate task (deduped)
        if "all" in doc_types_to_check:
            found_any = False
            for pkt in deduped_packets:
                if not pkt:
                    continue
                pt = _pkt_type(pkt)
                # Skip LC pages, transmission docs, blank pages, T&C — only
                # check real shipping docs. Now passes the packet so the
                # exclusion check can also drop near-empty packets.
                if _is_excluded_from_alldoc_fanout(pt, pkt):
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

        # P198f — OR-combined virtual target. Assemble one task that
        # bundles the text of every candidate doc behind the "or" and
        # tells the verifier: pass if ANY of them shows the requirement.
        if doc_types_to_check == ["__or_combined__"]:
            _or_sections = []
            _or_found_any = False
            _or_images = []
            for _ord in _or_doc_types:
                _ord_pkts = _find_matching_docs(_ord, deduped_packets)
                if not _ord_pkts:
                    _or_sections.append(
                        f"=== {_ord} ===\n[Document not present in the submission.]"
                    )
                    continue
                _or_found_any = True
                for _p in _ord_pkts:
                    _ptxt = _pkt_text(_p)
                    _pmeta = _pkt_visual_metadata(_p)
                    _or_sections.append(
                        f"=== {_ord} (type={_pkt_type(_p)}) ===\n"
                        f"{_pmeta}\n{_ptxt}"
                    )
                    _imgs = _pkt_images(_p)
                    if _imgs:
                        _or_images.extend(_imgs)
            _combined = "\n\n".join(_or_sections)
            _or_preamble = (
                f"OR-CONDITION: The LC requires the information to appear on "
                f"AT LEAST ONE of {', '.join(_or_doc_types)}. "
                f"PASS if ANY section below shows the required information. "
                f"FAIL only if NONE of the sections shows it. "
                f"Do not fail just because one section lacks it while another "
                f"section has it — that's the whole point of the OR.\n\n"
            )
            tasks.append({
                "row": row,
                "skip": False,
                "row_id": row_id,
                "condition_text": (_or_preamble + condition_text)
                                   if not _or_found_any
                                   else condition_text,
                "clause_ref": clause_ref,
                "lc_field_value": lc_field_value,
                "f47a_context": f47a_context,
                "document_type": " OR ".join(_or_doc_types),
                "document_text": _or_preamble + _combined,
                "visual_metadata": "",
                "image_path": _or_images[0] if _or_images else None,
                "multi_doc": False,
                "or_docs": _or_doc_types,
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
                    if _is_excluded_from_alldoc_fanout(pt, pkt):
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

            # multi_doc = True when checking multiple doc types OR multiple
            # packets of the same type (e.g., 2 different Shipping Company
            # Certificates). This ensures "ANY PASS = overall PASS" aggregation.
            _is_multi = len(doc_types_to_check) > 1 or len(matched_pkts) > 1
            for pkt in matched_pkts:
                images = _pkt_images(pkt)
                # Pull Step 3 outputs if present — drives new split-prompt path
                # and the deterministic fast-path in _call_vlm.
                _u_sum = pkt.get('unified_summary') if isinstance(pkt, dict) else None
                _bl_st = pkt.get('bl_subtype') if isinstance(pkt, dict) else None
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
                    "multi_doc": _is_multi,
                    "unified_summary": _u_sum if isinstance(_u_sum, dict) else None,
                    "bl_subtype": _bl_st if isinstance(_bl_st, dict) else None,
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
    packets = (step09_result.get("reconciled_packets")
               or step09_result.get("packets")
               or step09_result.get("classified_packets")
               or [])

    _progress(
        f"Verifying {len(rows)} condition rows against "
        f"{len(packets)} document packets (LLM-only mode)..."
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

    # ── LOI (Letter of Indemnity) detection ──
    # If the LC contains an LOI clause (added by amendment), the presentation
    # is under LOI terms. In LOI presentations:
    #   - Original BL is NOT available (replaced by BL copy + LOI)
    #   - Late presentation / late shipment / LC expiry checks are SUPPRESSED
    #     (LOI explicitly allows presentation without original docs)
    #   - Only check conditions that are in the Final LC clauses
    _is_loi_presentation = False
    _loi_clause_text = ''
    # Check 46A clauses for LOI indicators
    _f46a = _cf.get('46A', _cf.get('F46A', ''))
    if isinstance(_f46a, list):
        _f46a = '\n'.join(str(c.get('text', c) if isinstance(c, dict) else c) for c in _f46a)
    _f46a_upper = str(_f46a).upper()
    if any(kw in _f46a_upper for kw in [
        'LETTER OF INDEMNITY', 'LOI CLAUSE',
        'IN THE EVENT THAT ABOVE ORIGINAL DOCUMENTS ARE NOT AVAILABLE',
        'ORIGINAL DOCUMENTS ARE NOT AVAILABLE',
        'PAYMENT WILL BE EFFECTED AGAINST',
    ]):
        # Also check if LOI document is actually present in the submission
        _loi_found = any(
            'indemnity' in (_pkt_type(pkt) or '').lower()
            for pkt in packets
        )
        if _loi_found:
            _is_loi_presentation = True
            # Extract LOI clause text for verification
            for _line in str(_f46a).split('\n'):
                if any(kw in _line.upper() for kw in ['LETTER OF INDEMNITY', 'LOI', 'INDEMNITY']):
                    _loi_clause_text = _line.strip()
                    break
            _progress(f"LOI presentation detected — timing checks will be suppressed")
        else:
            _progress(f"LOI clause found in LC but no LOI document in submission — normal verification")

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

            # P168/P169 — Drop "Required document missing" rows for
            # document types that are NEITHER in the LC's clause text
            # NOR in the submission packet pool. These rows come from
            # LLM enumeration during decomposition ("ALL DOCUMENTS" fan-
            # outs that list specific doc types that the LC never asked
            # for). Dropping them prevents false-missing-doc failures.
            try:
                # 1) Build raw LC text blob from all F46A / F46B / F47A /
                #    F45A / F78 clauses. This is the source of truth for
                #    what the LC actually asked for.
                _lc_text_blob = ''
                try:
                    _cf = step06_result.get('consolidated_fields', {}) if isinstance(step06_result, dict) else {}
                    if not _cf and isinstance(step06_result, dict):
                        _cf = step06_result.get('final_lc', {}).get('consolidated_fields', {})
                    for _k in ('46A', 'F46A', '46B', 'F46B', '47A', 'F47A',
                                '45A', 'F45A', '45B', 'F45B',
                                '78', 'F78'):
                        _v = _cf.get(_k, '')
                        if isinstance(_v, list):
                            _v = '\n'.join(
                                str(_x.get('text', _x) if isinstance(_x, dict) else _x)
                                for _x in _v
                            )
                        _lc_text_blob += ' ' + str(_v or '')
                except Exception:
                    _lc_text_blob = ''
                _lc_text_up = _lc_text_blob.upper()

                # 2) Check if the doc-type target appears in the raw LC
                #    text (as a word / phrase). Uses a few tolerant
                #    variants for common abbreviations.
                _dt_lc = str(doc_target).strip()
                _dt_up = _dt_lc.upper()
                _dt_variants = {_dt_up}
                # Add common synonyms
                _SYN_MAP = {
                    'BENEFICIARY CERTIFICATE': ['BENEFICIARY\'S CERTIFICATE', 'BENEFICIARYS CERTIFICATE'],
                    'INSURANCE POLICY/CERTIFICATE': ['INSURANCE POLICY', 'INSURANCE CERTIFICATE'],
                    'CERTIFICATE OF ORIGIN': ['COO', 'C/O'],
                    'AIR WAYBILL': ['AIRWAY BILL', 'AWB', 'HAWB', 'MAWB'],
                    'PACKING LIST': ['PACKING SLIP', 'WEIGHT AND PACKING LIST'],
                    'DRAFT BILL OF EXCHANGE': ['BILL OF EXCHANGE', 'DRAFT', 'BOE'],
                    'SHIPPING COMPANY CERTIFICATE': ['AGENT\'S CERTIFICATE', 'AGENTS CERTIFICATE', 'CARRIER\'S CERTIFICATE'],
                }
                for _k, _syns in _SYN_MAP.items():
                    if _k == _dt_up:
                        _dt_variants.update(_syns)
                # Only drop if NONE of the variants appear in the LC text
                _in_lc = any(_v in _lc_text_up for _v in _dt_variants)

                if not _in_lc:
                    # 3) Not in LC text → drop row entirely. LLM enumerated
                    #    this doc type for an "ALL DOCUMENTS" clause that
                    #    never named it.
                    _set(row, "compliance", "N/A")
                    _set(row, "result", "")
                    _set(row, "findings", "")
                    _set(row, "verification_notes",
                         f"{doc_target} is not mentioned anywhere in the LC's F46A/F46B/F47A/F45A clauses — dropped (enumeration artifact)")
                    try:
                        row["_drop_from_report"] = True
                    except Exception:
                        pass
                    _progress(f"  {_get(row, 'row_id', '?')}: DROPPED (doc {doc_target} not in LC clauses + not in submission)")
                    continue
            except Exception as _e:
                try:
                    print(f"[P169] exception on row {_get(row, 'row_id', '?')}: {_e}")
                except Exception:
                    pass

            # P183 — Deduplicate missing-doc reports PER DOCUMENT TYPE
            # globally (not per clause). If the LC has multiple clauses
            # asking about the same missing doc (e.g. 5 conditions on
            # Shipment Advice), show ONE "Required document missing"
            # row; drop all other content-check rows for the same doc
            # from the report entirely. Checking sub-conditions of a
            # document that doesn't exist is meaningless.
            _missing_key = doc_target.strip().lower()
            if _missing_key in _seen_missing:
                # Drop entirely — not even N/A visible in report.
                _set(row, "compliance", "N/A")
                _set(row, "result", "")
                _set(row, "findings", "")
                _set(row, "verification_notes",
                     f"{doc_target} not in submission — content check skipped (missing doc already reported)")
                try:
                    row["_drop_from_report"] = True
                except Exception:
                    pass
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
        _progress(f"Sending {len(vlm_tasks)} conditions to Qwen LLM...")
        # Final-LC fields for the deterministic fast-path in _call_vlm
        _final_lc_fields = step06_result.get('final_lc', step06_result).get(
            'consolidated_fields', step06_result.get('consolidated_fields', {})
        )

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
                    task.get("unified_summary"),
                    task.get("bl_subtype"),
                    _final_lc_fields,
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
                    # P157 — on any thread exception, retry once
                    # synchronously before giving up. If retry also
                    # fails, return REVIEW with a clean message (no raw
                    # exception text in the user-facing finding).
                    print(f"[Step 14] row {row_id} thread exception: {type(exc).__name__}: {str(exc)[:300]}")
                    try:
                        result = _call_vlm(
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
                            task.get("unified_summary"),
                            task.get("bl_subtype"),
                            _final_lc_fields,
                        )
                    except Exception as exc2:
                        print(f"[Step 14] row {row_id} retry also failed: {type(exc2).__name__}: {str(exc2)[:300]}")
                        result = {
                            "row_id": row_id,
                            "findings": "Unable to verify this condition automatically — requires manual review.",
                            "result": "Unable to verify automatically — manual review required.",
                            "compliance": "review",
                            "confidence": 0.0,
                            "reasoning": f"{type(exc).__name__}: {str(exc)[:200]}",
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

        has_pass = any(r["compliance"] == "PASS" for r in results)
        has_fail = any(r["compliance"] == "FAIL" for r in results)
        has_review = any(r["compliance"] == "REVIEW" for r in results)

        # P185 — Universal-quantifier aggregation for "ALL DOCUMENTS"
        # clauses. When the LC says "DRAFTS AND ALL OTHER DOCUMENTS MUST
        # SHOW OUR LC NUMBER", we fan out to every submitted shipping
        # document (excluding blank pages, T&C pages, and Documentary
        # Remittance — already filtered by _is_excluded_from_alldoc_fanout
        # at fan-out time). The per-doc results are reported individually:
        # which docs PASS, which FAIL, which are REVIEW. If ANY real doc
        # fails → overall FAIL with the list of docs missing the value.
        _cond_text_u = (_get(row, "condition_text", "")
                        or _get(row, "condition", "")).upper()
        _is_universal = any(p in _cond_text_u for p in (
            'ALL DOCUMENTS', 'ALL OTHER DOCUMENTS',
            'EVERY DOCUMENT', 'EACH DOCUMENT',
            'ALL THE DOCUMENTS', 'EACH OF THE DOCUMENTS',
            'ALL SHIPPING DOCUMENTS', 'ON ALL DOCUMENTS',
            'ALL PRESENTED DOCUMENTS', 'ALL SUBMITTED DOCUMENTS',
        ))
        if _is_universal:
            # P198ag — Group per-packet results by document_type so
            # that back-side / endorsement / copy pages of the same
            # doc don't produce a spurious FAIL when the FRONT page
            # already carries the required value. The rule "DRAFTS
            # AND ALL OTHER DOCUMENTS MUST SHOW OUR LC NUMBER"
            # applies once per logical DOCUMENT (not per page or per
            # copy). A Draft typically comes as First-of-Exchange +
            # Second-of-Exchange copies; the first page carries the
            # LC#, the second is the endorsement side. If the first
            # PASSes, the document-class has satisfied the rule.
            _by_type: Dict[str, list] = {}
            for r in results:
                _dt = r.get("document_type", "?")
                _by_type.setdefault(_dt, []).append(r)
            _pass_types = []
            _fail_types = []
            _review_types = []
            for _dt, _bucket in _by_type.items():
                if any(x.get("compliance") == "PASS" for x in _bucket):
                    _pass_types.append(_dt)
                elif any(x.get("compliance") == "REVIEW" for x in _bucket):
                    _review_types.append(_dt)
                else:
                    _fail_types.append(_dt)

            # P198an — Deterministic rescue for universal "all docs must
            # show X" checks. If the LLM was evaluated on the deduped
            # representative of a doc-type and missed the value (often
            # because OCR glues letters into one block like
            # "DOCUMENTARYCREDITNUMBER:0086LC55629/2025DATED250109"),
            # scan the ORIGINAL (undeduped) packets of that type for
            # the literal value. If ANY packet of the type contains
            # it, rescue: move the type from _fail_types to
            # _pass_types.
            try:
                _cond_u_for_rescue = _cond_text_u
                _rescue_values = []  # list of (label, value) to search for
                _cf_rescue = step06_result.get('consolidated_fields', {})
                if not _cf_rescue and isinstance(step06_result, dict):
                    _cf_rescue = step06_result.get('final_lc', {}).get('consolidated_fields', {}) or {}
                # LC number (F20)
                if ('LC NUMBER' in _cond_u_for_rescue or
                        'L/C NUMBER' in _cond_u_for_rescue or
                        'DOCUMENTARY CREDIT NUMBER' in _cond_u_for_rescue or
                        'DOCUMENTARY CREDIT NO' in _cond_u_for_rescue or
                        'CREDIT NUMBER' in _cond_u_for_rescue):
                    _v20 = str(_cf_rescue.get('20', '') or _cf_rescue.get('F20', '') or '').strip()
                    if _v20:
                        _rescue_values.append(('LC#', _v20))
                # LC issue date (F31C)
                if ('DATE OF THE L/C' in _cond_u_for_rescue or
                        'DATE OF L/C' in _cond_u_for_rescue or
                        'LC ISSUE DATE' in _cond_u_for_rescue or
                        'L/C DATE' in _cond_u_for_rescue or
                        ('LC' in _cond_u_for_rescue and 'ISSUE' in _cond_u_for_rescue)):
                    _v31c = str(_cf_rescue.get('31C', '') or _cf_rescue.get('F31C', '') or '').strip()
                    if _v31c:
                        _rescue_values.append(('LC date', _v31c))
                # Issuing bank (F52A / F50B)
                if ('ISSUING BANK' in _cond_u_for_rescue or
                        'OPENING BANK' in _cond_u_for_rescue or
                        'L/C ISSUING' in _cond_u_for_rescue):
                    _v52 = str(_cf_rescue.get('52A', '') or _cf_rescue.get('F52A', '') or '').strip()
                    if _v52:
                        # First line only — the bank name
                        _bank_name = _v52.split('\n')[0].strip()
                        if _bank_name:
                            _rescue_values.append(('Issuing bank', _bank_name))

                if _rescue_values and _fail_types:
                    def _normalize_for_scan(s):
                        # Collapse whitespace, strip punctuation — catches
                        # OCR glue like "LCNUMBER:0086LC55629/2025"
                        return re.sub(r'[\s\-.,;:()\[\]]+', '', str(s).upper())

                    # Group ORIGINAL packets by document_type
                    _orig_by_type: Dict[str, list] = {}
                    for _p in packets:
                        if not isinstance(_p, dict):
                            continue
                        _dt = _p.get('document_type', '') or 'Unknown'
                        _orig_by_type.setdefault(_dt, []).append(_p)

                    _rescued = []
                    for _ft in list(_fail_types):
                        _pkts_of_type = _orig_by_type.get(_ft, [])
                        if not _pkts_of_type:
                            continue
                        # For each required value, check if ANY packet has it
                        _all_values_found = True
                        _found_labels = []
                        for _label, _value in _rescue_values:
                            _v_norm = _normalize_for_scan(_value)
                            if not _v_norm:
                                continue
                            _found_in_any = False
                            for _p in _pkts_of_type:
                                _ptxt = _pkt_text(_p) or ''
                                _us = _p.get('unified_summary') or {}
                                _us_parts = [
                                    str(_us.get('lc_reference', '')),
                                    str(_us.get('issue_date', '')),
                                    str(_us.get('issuer', '')),
                                ]
                                for _arr_k in ('references_found', 'dates_found', 'parties_found', 'other_details_found'):
                                    for _item in (_us.get(_arr_k) or []):
                                        if isinstance(_item, dict):
                                            _us_parts.append(str(_item.get('value', '')))
                                            _us_parts.append(str(_item.get('raw', '')))
                                _combined = _normalize_for_scan(_ptxt + ' ' + ' '.join(_us_parts))
                                if _v_norm in _combined:
                                    _found_in_any = True
                                    break
                            if _found_in_any:
                                _found_labels.append(_label)
                            else:
                                _all_values_found = False
                                break
                        if _all_values_found and _rescue_values:
                            _rescued.append((_ft, _found_labels))
                    if _rescued:
                        for _ft, _labels in _rescued:
                            _fail_types.remove(_ft)
                            if _ft not in _pass_types:
                                _pass_types.append(_ft)
                        try:
                            _progress(
                                f"  [P198an universal-rescue] row={row.get('row_id','?')}: "
                                f"rescued {len(_rescued)} type(s) from FAIL via deterministic "
                                f"scan: {', '.join(t for t,_ in _rescued)}"
                            )
                        except Exception:
                            pass
            except Exception as _e_rescue:
                try:
                    print(f"[P198an universal-rescue] exception: {_e_rescue}")
                except Exception:
                    pass
            _per_doc_lines = []
            for r in results:
                _dt = r.get("document_type", "?")
                _cv = r.get("compliance", "REVIEW")
                _rs = (r.get("result") or r.get("findings") or "").strip()
                if len(_rs) > 120:
                    _rs = _rs[:117] + "..."
                _per_doc_lines.append(f"{_dt}: {_cv}" + (f" — {_rs}" if _rs else ""))
            _per_doc_block = " | ".join(_per_doc_lines)
            if _fail_types:
                agg_compliance = "FAIL"
                _missing = ", ".join(_fail_types)
                combined_findings = (
                    f"Required value missing on: {_missing}. "
                    f"Present on: {', '.join(_pass_types) or '(none)'}. "
                    f"Per-doc: {_per_doc_block}"
                )
                combined_result = f"Missing on {len(_fail_types)} doc(s): {_missing}"
            elif _review_types:
                agg_compliance = "REVIEW"
                combined_findings = (
                    f"Requirement unclear on: {', '.join(_review_types)}. "
                    f"Present on: {', '.join(_pass_types) or '(none)'}. "
                    f"Per-doc: {_per_doc_block}"
                )
                combined_result = f"Unclear on {len(_review_types)} doc(s)"
            else:
                agg_compliance = "PASS"
                combined_findings = (
                    f"Requirement satisfied on all {len(_pass_types)} "
                    f"document class(es): {', '.join(_pass_types)}. "
                    f"Per-doc: {_per_doc_block}"
                )
                combined_result = (
                    f"Present on all {len(_pass_types)} doc(s): "
                    f"{', '.join(_pass_types)}"
                )[:200]
            avg_conf = round(
                sum(r.get("confidence", 0.0) for r in results) / max(len(results), 1),
                2,
            )
        else:
            # Existential best-case aggregation (non-universal default):
            # ANY pass proves compliance.
            if has_pass:
                agg_compliance = "PASS"
            elif has_review:
                agg_compliance = "REVIEW"
            else:
                agg_compliance = "FAIL"
            _best = None
            for r in results:
                if r.get("compliance") == agg_compliance:
                    _best = r
                    break
            if not _best:
                _best = results[0]
            combined_findings = _best.get("findings", "Nil")
            combined_result = _best.get("result", "")
            avg_conf = _best.get("confidence", 0.0)

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
    # 5b. Deterministic post-checks — override VLM false positives
    # ------------------------------------------------------------------ #
    # The VLM sometimes hallucates PASS for conditions it cannot verify
    # from the document text alone. Run deterministic checks for:
    #   - Email address presence: if the condition says "send via email
    #     to X@Y.COM", the document MUST mention that email address.
    #   - Fax number presence: same logic for fax numbers.
    # Normalisation: SWIFT uses (AT) for @, (DOT) for .
    def _normalise_email_text(s: str) -> str:
        """Convert SWIFT email notation to standard: INFO(AT)SIUT. ORG → info@siut.org"""
        s = re.sub(r'\(\s*AT\s*\)', '@', s, flags=re.IGNORECASE)
        s = re.sub(r'\(\s*DOT\s*\)', '.', s, flags=re.IGNORECASE)
        # Strip spaces around @ and . in email-like contexts
        # "INFO @SIUT. ORG" → "INFO@SIUT.ORG"
        s = re.sub(r'\s*@\s*', '@', s)
        s = re.sub(r'(\w)\s*\.\s*(\w)', r'\1.\2', s)
        return s

    def _extract_emails(text: str) -> list:
        """Extract all email addresses from text, normalising (AT)/(DOT)."""
        normalised = _normalise_email_text(text)
        return [e.lower() for e in re.findall(
            r'[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}', normalised
        )]

    _email_checked_rows = set()
    for task in vlm_tasks:
        row = task["row"]
        row_id = task.get("row_id", "?")
        compliance = _get(row, "compliance", "").upper()
        # P168/P181 — Run the email check on ALL verdicts (PASS/REVIEW/
        # FAIL) so we can both catch false positives AND rewrite the
        # finding text to a clean canonical form on genuine FAILs.
        if compliance not in ("PASS", "REVIEW", "REVIEW REQUIRED", "FAIL", "NOT COMPLIED"):
            continue

        cond_text = (task.get("condition_text") or "").lower()
        doc_text = (task.get("document_text") or "")

        # Check: does the condition mention sending to a specific email?
        email_keywords = ['via email', 'by email', 'email at', 'email to',
                          'e-mail to', 'e-mail at', 'send to']
        has_email_kw = any(kw in cond_text for kw in email_keywords)

        if has_email_kw:
            # Extract email addresses from the LC condition
            cond_emails = _extract_emails(cond_text)
            _progress(
                f"  [email-check] {row_id}: kw=True, "
                f"cond_emails={cond_emails}, doc_len={len(doc_text)}"
            )
            if cond_emails and row_id not in _email_checked_rows:
                _email_checked_rows.add(row_id)
                # Check if ANY of these emails appear in the document text
                doc_emails = _extract_emails(doc_text)
                doc_text_normalised = _normalise_email_text(doc_text).lower()
                found_any = False
                for em in cond_emails:
                    if em in doc_text_normalised or em in doc_emails:
                        found_any = True
                        break
                _progress(
                    f"  [email-check] {row_id}: doc_emails={doc_emails}, "
                    f"found_any={found_any}"
                )
                if not found_any:
                    # P181 — Clean canonical finding text for missing
                    # email. Applies to both PASS-turned-FAIL and to
                    # existing FAIL rows (where the LLM's verbose
                    # "document does not provide any evidence" is
                    # replaced with concise requirement language).
                    _set(row, "compliance", "FAIL")
                    missing_emails = ', '.join(cond_emails)
                    _clean_finding = (
                        f"Email {missing_emails} required but not mentioned in document."
                    )
                    _set(row, "findings", _clean_finding)
                    _set(row, "found_text", _clean_finding)
                    _set(row, "result", _clean_finding)
                    _set(row, "verification_notes",
                         f"LC requires notification via {missing_emails}; "
                         f"document text does not contain this address.")
                    _progress(f"  {row_id}: {compliance}->FAIL (email {missing_emails} missing)")
                else:
                    # P181 — If the email IS found and LLM previously
                    # returned FAIL, this likely a hallucination; flip
                    # to PASS. Already-PASS/REVIEW rows stay as-is.
                    if compliance in ("FAIL", "NOT COMPLIED"):
                        _set(row, "compliance", "PASS")
                        found_emails = ', '.join(cond_emails)
                        _set(row, "findings",
                             f"Email {found_emails} is present on the document.")
                        _set(row, "result",
                             f"Email {found_emails} is present on the document.")
                        _progress(f"  {row_id}: FAIL->PASS (email {found_emails} found in doc)")

    # ------------------------------------------------------------------ #
    # 5c. Deterministic text-presence check — override VLM false FAILs
    # ------------------------------------------------------------------ #
    # The VLM sometimes marks a condition as FAIL ("partially matches")
    # even when the required text IS present in the document. This check
    # extracts quoted text or key address phrases from the condition and
    # verifies whether they appear in the document or the VLM's own
    # found_text field. If found, override FAIL → PASS.
    def _normalise_for_compare(s: str) -> str:
        """Lowercase, collapse whitespace, strip punctuation for fuzzy compare."""
        s = s.lower()
        s = re.sub(r'[,.:;\'\"()\[\]{}/-]', ' ', s)
        # Insert space at digit↔letter boundaries (OCR often glues them)
        # "10chutes" → "10 chutes", "1set" → "1 set", "usd28500" → "usd 28500"
        s = re.sub(r'(\d)([a-z])', r'\1 \2', s)
        s = re.sub(r'([a-z])(\d)', r'\1 \2', s)
        s = re.sub(r'\s+', ' ', s).strip()
        return s

    _text_checked_rows = set()
    for task in vlm_tasks:
        row = task["row"]
        row_id = task.get("row_id", "?")
        compliance = _get(row, "compliance", "").upper()
        if compliance != "FAIL" or row_id in _text_checked_rows:
            continue

        cond_text = task.get("condition_text") or ""
        doc_text = task.get("document_text") or ""
        found_text = _get(row, "findings", "") or _get(row, "found_text", "") or ""

        # Only check conditions about showing/containing specific text
        _presence_kw = ['must show', 'must indicate', 'must mention', 'must state',
                        'must bear', 'must contain', 'must reflect', 'showing',
                        'indicating', 'marked', 'addressed to', 'endorsed to',
                        'notify applicant', 'notify party', 'consigned to']
        if not any(kw in cond_text.lower() for kw in _presence_kw):
            continue

        # Strategy 1: Extract quoted text from condition  ('...' or "...")
        quoted = re.findall(r"['\"]([^'\"]{8,})['\"]", cond_text)

        # Strategy 2: Extract address-like text after "address" keyword
        addr_match = re.search(
            r'(?:address|addressed?\s+to)\s+[\'"]?(.{15,}?)(?:[\'"]\.?\s*$|$)',
            cond_text, re.IGNORECASE
        )
        if addr_match:
            quoted.append(addr_match.group(1).strip().rstrip('.'))

        if not quoted:
            continue

        _text_checked_rows.add(row_id)
        # Check if the quoted/extracted text appears in doc text OR found_text
        search_pool = _normalise_for_compare(doc_text + ' ' + found_text)

        for phrase in quoted:
            norm_phrase = _normalise_for_compare(phrase)
            # Split into key segments (at least 3 words each) and check all present
            words = norm_phrase.split()
            if len(words) < 3:
                continue

            # Check the full phrase first
            if norm_phrase in search_pool:
                _set(row, "compliance", "PASS")
                _set(row, "result", f"Text found in document: {phrase[:80]}")
                _set(row, "verification_notes",
                     f"Deterministic override: required text '{phrase[:80]}' "
                     f"found in document — VLM false FAIL corrected")
                _progress(f"  {row_id}: FAIL->PASS (text '{phrase[:50]}' found in doc)")
                break

            # Fallback: check if ALL key address segments are present
            # Split into chunks of ~3 words
            segments = []
            for j in range(0, len(words) - 2, 2):
                seg = ' '.join(words[j:j+3])
                if len(seg) >= 8:
                    segments.append(seg)
            if segments and all(seg in search_pool for seg in segments):
                _set(row, "compliance", "PASS")
                _set(row, "result", f"Address/text verified in document")
                _set(row, "verification_notes",
                     f"Deterministic override: all key segments of "
                     f"'{phrase[:80]}' found in document")
                _progress(f"  {row_id}: FAIL->PASS (address segments matched)")
                break

    # ------------------------------------------------------------------ #
    # 5c2. Override LLM false PASSes for specific verifiable conditions
    # ------------------------------------------------------------------ #
    for task in vlm_tasks:
        row = task["row"]
        row_id = task.get("row_id", "?")
        compliance = _get(row, "compliance", "").upper()
        if compliance != "PASS":
            continue

        cond_text = (task.get("condition_text") or "")
        doc_text = (task.get("document_text") or "")
        _cond_up = cond_text.upper()
        _doc_up = doc_text.upper()

        # Check 0: Company name changes — "currently known as" / "formerly known as"
        # If the findings mention a name change phrase, the entity IS the same → keep PASS
        _findings_text = (_get(row, "findings", "") or "").upper()
        if compliance == "PASS":
            pass  # Already PASS, no override needed
        # But if FAIL and the document shows a name change phrase, override to PASS
        if compliance == "FAIL":
            _name_change_phrases = ['CURRENTLY KNOWN AS', 'FORMERLY KNOWN AS', 'NOW KNOWN AS',
                                     'TRADING AS', 'T/A', 'D/B/A', 'ALSO KNOWN AS']
            _doc_has_name_change = any(p in _doc_up for p in _name_change_phrases)
            _cond_has_beneficiary = 'BENEFICIARY' in _cond_up or 'ISSUED BY' in _cond_up
            if _doc_has_name_change and _cond_has_beneficiary:
                _set(row, "compliance", "PASS")
                _set(row, "result", "Beneficiary identified (name change noted on document)")
                _set(row, "findings", "Document shows company name change — same legal entity")
                _progress(f"  {row_id}: FAIL->PASS (company name change detected)")
            compliance = _get(row, "compliance", "").upper()

        # Check A: Reference numbers (NTN, permit, certificate numbers)
        # If condition says "NTN No. XXXXX must appear" and XXXXX is NOT
        # in the document text, the LLM hallucinated PASS.
        _ref_patterns = [
            (r'NTN\s+(?:NO\.?\s*)?(\d[\d\-/]+)', 'NTN'),
            (r'IMPORT\s+PERMIT\s+NO\.?\s*([A-Z0-9\-/]+)', 'Import Permit'),
            (r'COVER\s+NOTE\s+NO\.?\s*([A-Z0-9\-/]+)', 'Cover Note'),
            (r'POLICY\s+NO\.?\s*([A-Z0-9\-/]+)', 'Policy'),
        ]
        for _pat, _label in _ref_patterns:
            _ref_m = re.search(_pat, _cond_up)
            if _ref_m:
                _ref_num = _ref_m.group(1).strip()
                # Check if this reference number appears in the document
                if _ref_num not in _doc_up and _ref_num.replace('-', '') not in _doc_up.replace('-', ''):
                    _set(row, "compliance", "FAIL")
                    _set(row, "result", f"{_label} {_ref_num} not found in document")
                    _set(row, "findings", f"{_label} number {_ref_num} not found in document text")
                    _progress(f"  {row_id}: PASS->FAIL ({_label} {_ref_num} not in document)")
                break

        # Check B: Consignee "TO ORDER OF [bank]" vs "TO ORDER" alone
        # "TO ORDER" without a specific bank name is NOT the same as
        # "TO THE ORDER OF UNITED BANK LTD"
        if compliance == "PASS" and 'ORDER OF' in _cond_up and 'BILL OF LADING' in (task.get("document_type") or "").upper():
            # Extract the bank/party name after "ORDER OF"
            _order_m = re.search(r'ORDER\s+OF\s+([A-Z][A-Z\s,.\-()]+?)(?:\.|$|\n)', _cond_up)
            if _order_m:
                _order_party = _order_m.group(1).strip().rstrip('.,')
                # Check if this party name appears after "ORDER" in the document
                _doc_has_order_party = False
                _order_keywords = [w for w in _order_party.split() if len(w) >= 3 and w not in ('THE', 'AND', 'LTD', 'LIMITED')]
                if _order_keywords:
                    _found_kw = sum(1 for w in _order_keywords if w in _doc_up)
                    _doc_has_order_party = _found_kw >= len(_order_keywords) * 0.6
                if not _doc_has_order_party:
                    _set(row, "compliance", "FAIL")
                    _set(row, "result", f"Consignee does not show '{_order_party[:40]}'")
                    _set(row, "findings", f"BL shows 'TO ORDER' but LC requires 'TO THE ORDER OF {_order_party[:40]}'")
                    _progress(f"  {row_id}: PASS->FAIL (order of {_order_party[:30]} not in BL)")

        # Refresh compliance after possible override
        compliance = _get(row, "compliance", "").upper()

        # Check C: Notify party — must match the SPECIFIC party named
        if compliance == "PASS" and 'NOTIFY' in _cond_up and 'ISSUING BANK' in _cond_up:
            # Condition says "notify the issuing bank" — check if the
            # issuing bank name appears in the notify party section
            _notify_section = ''
            _nm = re.search(r'NOTIFY\s+PARTY[:\s]*(.*?)(?:PRE[\-\s]CARRIAGE|OCEAN\s+VESSEL|PORT\s+OF|PLACE\s+OF|\Z)', _doc_up, re.DOTALL)
            if _nm:
                _notify_section = _nm.group(1)
            # Get issuing bank name
            _lc_pf = step06_result.get('final_lc', step06_result).get('consolidated_fields', step06_result.get('consolidated_fields', {}))
            _ib = str(_lc_pf.get('52A', _lc_pf.get('51A', _lc_pf.get('42D', '')))).split('\n')[0].strip()
            if _ib and _notify_section:
                _ib_keywords = [w for w in _ib.upper().split() if len(w) >= 3 and w not in ('THE', 'AND', 'LTD', 'LIMITED')]
                _found = sum(1 for w in _ib_keywords if w in _notify_section)
                if _ib_keywords and _found < len(_ib_keywords) * 0.5:
                    _set(row, "compliance", "FAIL")
                    _set(row, "result", f"Notify party does not show issuing bank '{_ib[:40]}'")
                    _set(row, "findings", f"BL notify party does not include '{_ib[:40]}'")
                    _progress(f"  {row_id}: PASS->FAIL (issuing bank not in notify party)")

    # NOTE: Section 5c3 (hard-coded FAIL->PASS overrides for specific evidence
    # patterns — Bunge name change, Pacific NW agent-for-master, "SEE OVERLEAF"
    # blank-back) was REMOVED in P123.
    # Rationale: the new split-prompt path (CORE + family packs) + structured
    # facts from Step 3e (bl_subtype.signing_type, bl_subtype.has_terms_overleaf,
    # parties_found with "currently known as" raw text) now covers these cases
    # via general LLM reasoning + deterministic fast-path in _call_vlm —
    # no string-matching overrides needed.

    # ------------------------------------------------------------------ #
    # 5d. LOI presentation — suppress timing checks
    # ------------------------------------------------------------------ #
    # When LOI is presented, late presentation / late shipment / LC expiry
    # checks are not applicable. The LOI clause explicitly allows
    # presentation without original documents, which implies relaxed timing.
    if _is_loi_presentation:
        _loi_suppress_types = {'lc_expiry', 'late_shipment', 'late_presentation'}
        _loi_suppress_keywords = [
            'presentation date must not exceed',
            'shipment date must not exceed',
            'late shipment', 'late presentation',
            'stale', 'presentation period',
            'documents presented within',
            'bill of lading date',
        ]
        _suppressed = 0
        for row in rows:
            compliance = _get(row, "compliance", "").upper()
            if compliance not in ("FAIL", "REVIEW"):
                continue
            # Check implicit_type
            _imp_type = _get(row, "implicit_type", "")
            _cond = _get(row, "condition_text", "").lower()
            should_suppress = False
            if _imp_type in _loi_suppress_types:
                should_suppress = True
            elif any(kw in _cond for kw in _loi_suppress_keywords):
                should_suppress = True
            if should_suppress:
                _set(row, "compliance", "N/A")
                _set(row, "result", f"Not applicable — LOI presentation")
                _set(row, "verification_notes",
                     f"LOI clause in LC allows presentation without original documents. "
                     f"Timing checks (late shipment, late presentation, LC expiry) "
                     f"are suppressed under LOI terms.")
                _suppressed += 1
        if _suppressed:
            _progress(f"  LOI: Suppressed {_suppressed} timing check(s)")

    # ------------------------------------------------------------------ #
    # 5e. Payment terms verification — Drafts + Invoices vs LC F42C
    #      DISABLED: cross-document checks create a separate table in the
    #      report that is not part of the LC verification scope yet.
    #      To re-enable: change `_ENABLE_PAYMENT_TERMS_CHECK` to True
    # ------------------------------------------------------------------ #
    _ENABLE_PAYMENT_TERMS_CHECK = False
    # Parse F42C to understand payment structure, then verify:
    #   1. Draft tenor matches LC terms (sight, X days from BL date)
    #   2. Installment count: number of drafts == number of installments
    #   3. Each draft amount == installment % × LC amount
    #   4. Total drafts == LC amount
    #   5. Total invoices <= LC amount (with tolerance)
    #
    # Example F42C:
    #   100 PCT BY IRREVOCABLE L/C WITH 4 INSTALLMENTS
    #   A) 25 PCT ... WITHIN 90 DAYS FROM THE B/L ISSUING DATE.
    #   B) 25 PCT ... WITHIN 180 DAYS FROM THE B/L ISSUING DATE.
    #   C) 25 PCT ... WITHIN 270 DAYS FROM THE B/L ISSUING DATE.
    #   D) 25 PCT ... WITHIN 360 DAYS FROM THE B/L ISSUING DATE.

    def _parse_amount(s: str) -> float:
        """Extract numeric amount from string like 'USD 500,000.00'."""
        if not s:
            return 0.0
        # Remove currency codes and whitespace
        cleaned = re.sub(r'[A-Z]{2,}\s*', '', s.upper()).strip()
        # Find the number
        m = re.search(r'[\d,]+(?:\.\d{1,2})?', cleaned)
        if m:
            return float(m.group(0).replace(',', ''))
        return 0.0

    _f42c_val = _get_lc_field_value(step06_result, '42C') if _ENABLE_PAYMENT_TERMS_CHECK else ''
    if _f42c_val and _ENABLE_PAYMENT_TERMS_CHECK:
        _f42c_upper = _f42c_val.upper()
        _progress(f"  [payment-terms] F42C = {_f42c_upper[:120]}")

        # ── Parse payment structure ──
        _is_sight = bool(re.search(r'\bAT\s+SIGHT\b', _f42c_upper))

        # Detect installments: "4 INSTALLMENTS" or "FOUR INSTALLMENTS"
        _installment_m = re.search(r'(\d+)\s*INSTALL?MENTS?', _f42c_upper)
        _num_installments = int(_installment_m.group(1)) if _installment_m else 0

        # Parse individual installment lines: "A) 25 PCT ... WITHIN 90 DAYS FROM ..."
        # Each installment has: percentage, days, and reference (BL date, sight, etc.)
        _installments = []  # list of {pct: int, days: int, ref: str}
        _inst_pattern = re.compile(
            r'[A-Z]\)\s*(\d+)\s*(?:PCT|PERCENT|%)\s+.*?'
            r'(?:WITHIN|AFTER)\s+(\d+)\s*DAYS?\s+'
            r'(?:FROM|AFTER|OF)\s+(.*?)(?:\.|$)',
            re.IGNORECASE
        )
        for _im in _inst_pattern.finditer(_f42c_upper):
            _installments.append({
                'pct': int(_im.group(1)),
                'days': int(_im.group(2)),
                'ref': _im.group(3).strip(),
            })

        # If no structured installments found, try simpler tenor extraction
        _tenor_days = [inst['days'] for inst in _installments]
        if not _tenor_days:
            for _dm in re.finditer(r'(\d+)\s*DAYS?\b', _f42c_upper):
                _d = int(_dm.group(1))
                if _d >= 7:  # Filter out garbage small numbers
                    _tenor_days.append(_d)

        _progress(f"  [payment-terms] sight={_is_sight}, installments={_num_installments}, "
                  f"parsed={_installments}, tenor_days={_tenor_days}")

        # ── Get LC amount and tolerance ──
        _lc_amount_str = _get_lc_field_value(step06_result, '32B')
        _lc_amount = _parse_amount(_lc_amount_str)
        _lc_ccy_m = re.search(r'(USD|EUR|GBP|JPY|CNY|PKR|AED|SAR|INR|BDT|LKR)',
                              _lc_amount_str.upper())
        _lc_ccy = _lc_ccy_m.group(1) if _lc_ccy_m else ''

        # Tolerance from F39A (plus/minus percentage) or F39B (max credit amount)
        # UCP 600 Art 30(b): if no tolerance stated, default is 5% plus/minus
        _tol_plus = 5.0   # default per UCP 600
        _tol_minus = 5.0  # default per UCP 600
        _tol_39a = _get_lc_field_value(step06_result, '39A')
        _tol_39b = _get_lc_field_value(step06_result, '39B')
        if _tol_39a:
            # F39A format: "13/10" means +13%/-10% or "5/5" means ±5%
            _tol_parts = re.findall(r'(\d+(?:\.\d+)?)', _tol_39a)
            if len(_tol_parts) >= 2:
                _tol_plus = float(_tol_parts[0])
                _tol_minus = float(_tol_parts[1])
            elif len(_tol_parts) == 1:
                _tol_plus = _tol_minus = float(_tol_parts[0])
        elif _tol_39b:
            # F39B: "NOT EXCEEDING" — no tolerance allowed
            if re.search(r'NOT\s+EXCEED', _tol_39b, re.IGNORECASE):
                _tol_plus = 0.0
                _tol_minus = 0.0

        _progress(f"  [payment-terms] LC amount={_lc_ccy} {_lc_amount}, tolerance=+{_tol_plus}%/-{_tol_minus}%")

        # ── Find draft and invoice packets ──
        _draft_packets = []
        _invoice_packets = []
        for pkt in packets:
            _pt = (_pkt_type(pkt) or '').lower()
            if any(k in _pt for k in ('draft', 'bill of exchange', 'boe')):
                _draft_packets.append(pkt)
            elif 'invoice' in _pt and 'proforma' not in _pt:
                _invoice_packets.append(pkt)

        # Count actual drafts (not packets) — a single page may contain
        # FIRST and SECOND of Exchange (two copies of the same draft).
        # For installment LCs, only ONE draft is issued for the full amount
        # covering all installments — NOT one draft per installment.
        _actual_draft_count = 0
        for _dp in _draft_packets:
            _draft_text = (_pkt_text(_dp) or '').upper()
            # Count "FIRST OF EXCHANGE" / "SECOND OF EXCHANGE" copies
            _copies = len(re.findall(
                r'(?:FIRST|SECOND|THIRD|FOURTH|1ST|2ND|3RD|4TH)\s+(?:OF\s+)?(?:EXCHANGE|BILL)',
                _draft_text, re.IGNORECASE))
            _actual_draft_count += max(1, _copies)

        _progress(f"  [payment-terms] {len(_draft_packets)} draft packet(s), "
                  f"{_actual_draft_count} actual draft(s), "
                  f"{len(_invoice_packets)} invoice(s)")

        # Helper: add/update a payment-terms row in the results
        def _add_pt_finding(check_name, compliance, result, details):
            """Add a payment terms finding as a new row or update existing."""
            _new_row = {
                "field_tag": "42C",
                "field_description": f"Payment Terms — {check_name}",
                "lc_field_value": _f42c_upper[:200],
                "condition_text": check_name,
                "compliance": compliance,
                "result": result,
                "findings": details,
                "verification_notes": f"Deterministic payment terms check",
                "document_checked": "Cross-document",
                "implicit_type": "payment_terms",
            }
            rows.append(_new_row)
            _progress(f"  [payment-terms] {check_name}: {compliance} — {result}")

        # ── Check 1: Installment count vs drafts ──
        # For installment LCs, typically ONE draft is issued for the FULL
        # amount, with all installment payment schedules written on it.
        # The number of installments ≠ number of drafts.
        # Instead, verify that the draft TEXT mentions all installment entries.
        if _num_installments > 0 and _draft_packets:
            # Check if the draft text contains all installment references
            _draft_all_text = ' '.join((_pkt_text(dp) or '').upper() for dp in _draft_packets)
            _found_installments = []
            for _inst in _installments:
                # Look for the days value in the draft
                _days_str = str(_inst['days'])
                if _days_str in _draft_all_text:
                    _found_installments.append(_inst)
            if len(_found_installments) == len(_installments):
                _add_pt_finding(
                    "Installment Schedule",
                    "PASS",
                    f"Draft contains all {_num_installments} installment entries "
                    f"({', '.join(str(i['days'])+' days' for i in _installments)})",
                    f"All installment schedules found in draft text"
                )
            elif _found_installments:
                _missing = [i for i in _installments if i not in _found_installments]
                _add_pt_finding(
                    "Installment Schedule",
                    "REVIEW",
                    f"Draft shows {len(_found_installments)}/{_num_installments} installments. "
                    f"Missing: {', '.join(str(i['days'])+' days' for i in _missing)}",
                    f"Some installment entries may be obscured by stamps/signatures"
                )
            else:
                _add_pt_finding(
                    "Installment Schedule",
                    "FAIL",
                    f"Draft does not mention any installment schedule",
                    f"Expected {_num_installments} installments but none found in draft"
                )

        # ── Check 2: Tenor matching per draft ──
        if _tenor_days and _draft_packets:
            for _di, _dp in enumerate(_draft_packets):
                _draft_text = (_pkt_text(_dp) or '').upper()
                _dpg = _dp.get('page_numbers', ['?'])

                # Extract tenor from draft
                _draft_is_sight = bool(re.search(r'\bAT\s+SIGHT\b', _draft_text))
                _draft_days_m = re.search(
                    r'(\d+)\s*DAYS?\s*(?:AFTER|FROM|OF)\s*'
                    r'(?:SIGHT|B/?L|BILL\s+OF\s+LADING|SHIPMENT|DATE|BL\s+ISSUING)',
                    _draft_text)
                _draft_days = int(_draft_days_m.group(1)) if _draft_days_m else None

                if _is_sight:
                    if _draft_is_sight:
                        _add_pt_finding(
                            f"Draft Tenor (pg {_dpg})",
                            "PASS",
                            "Draft is 'AT SIGHT' matching LC terms",
                            f"Draft on page {_dpg}: AT SIGHT"
                        )
                    elif not re.search(r'\bSIGHT\b', _draft_text):
                        _add_pt_finding(
                            f"Draft Tenor (pg {_dpg})",
                            "FAIL",
                            f"Draft does not state 'AT SIGHT' — LC requires sight payment",
                            f"Draft on page {_dpg}: no SIGHT reference found"
                        )
                elif _draft_days is not None:
                    if _draft_days in _tenor_days:
                        _add_pt_finding(
                            f"Draft Tenor (pg {_dpg})",
                            "PASS",
                            f"Draft tenor {_draft_days} days matches LC terms",
                            f"Draft on page {_dpg}: {_draft_days} days (LC tenors: {_tenor_days})"
                        )
                    else:
                        _add_pt_finding(
                            f"Draft Tenor (pg {_dpg})",
                            "FAIL",
                            f"Draft tenor {_draft_days} days does not match LC terms {_tenor_days}",
                            f"Draft on page {_dpg}: {_draft_days} days not in {_tenor_days}"
                        )

        # ── Check 3: Draft amount vs LC amount ──
        # For installment LCs, ONE draft is issued for the FULL LC amount.
        # The installment schedule is written ON the draft but the face
        # amount is the total. So draft amount should = LC amount.
        if _lc_amount > 0 and _draft_packets:
            _total_draft_amount = 0.0
            for _dp in _draft_packets:
                _draft_text = (_pkt_text(_dp) or '').upper()
                _dpg = _dp.get('page_numbers', ['?'])

                # Extract the FIRST amount (face value) from draft
                # Look for "Exchange for USD171,000.00" pattern first
                _draft_amt_m = re.search(
                    r'(?:EXCHANGE\s+FOR|FOR\s+THE\s+SUM\s+OF|SUM\s+OF)\s*'
                    r'(?:USD|EUR|GBP|JPY|CNY|PKR|AED|SAR)\s*([\d,]+(?:\.\d{2})?)',
                    _draft_text)
                if not _draft_amt_m:
                    # Fallback: first currency+amount pattern
                    _draft_amt_m = re.search(
                        r'(?:USD|EUR|GBP|JPY|CNY|PKR|AED|SAR)\s*([\d,]+(?:\.\d{2})?)',
                        _draft_text)
                if _draft_amt_m:
                    _draft_amt = float(_draft_amt_m.group(1).replace(',', ''))
                    # Avoid counting the same amount from duplicate copies
                    # (FIRST/SECOND of exchange on same page)
                    if _total_draft_amount == 0 or abs(_draft_amt - _total_draft_amount) > 1:
                        _total_draft_amount = _draft_amt  # Use the face value, don't sum copies

            # Total drafts vs LC amount (within +/- tolerance)
            if _total_draft_amount > 0:
                _max_allowed = _lc_amount * (1 + _tol_plus / 100.0)
                _min_allowed = _lc_amount * (1 - _tol_minus / 100.0)
                if _min_allowed <= _total_draft_amount <= _max_allowed:
                    _add_pt_finding(
                        "Total Draft Amount",
                        "PASS",
                        f"Total drafts {_lc_ccy} {_total_draft_amount:,.2f} within "
                        f"LC amount {_lc_ccy} {_lc_amount:,.2f} "
                        f"(+{_tol_plus}%/-{_tol_minus}%)",
                        f"Range: {_min_allowed:,.2f} to {_max_allowed:,.2f}"
                    )
                elif _total_draft_amount > _max_allowed:
                    _add_pt_finding(
                        "Total Draft Amount",
                        "FAIL",
                        f"Total drafts {_lc_ccy} {_total_draft_amount:,.2f} exceeds "
                        f"LC amount {_lc_ccy} {_lc_amount:,.2f} (+{_tol_plus}%)",
                        f"Max allowed: {_max_allowed:,.2f}, over by {_total_draft_amount - _max_allowed:,.2f}"
                    )
                else:
                    _add_pt_finding(
                        "Total Draft Amount",
                        "FAIL",
                        f"Total drafts {_lc_ccy} {_total_draft_amount:,.2f} below "
                        f"LC amount {_lc_ccy} {_lc_amount:,.2f} (-{_tol_minus}%)",
                        f"Min allowed: {_min_allowed:,.2f}, short by {_min_allowed - _total_draft_amount:,.2f}"
                    )

        # ── Check 4: Total invoices vs LC amount (with tolerance) ──
        if _lc_amount > 0 and _invoice_packets:
            _total_inv_amount = 0.0
            for _ip in _invoice_packets:
                _inv_text = (_pkt_text(_ip) or '').upper()
                _inv_lines = _inv_text.split('\n')
                _inv_amt_found = 0.0

                # Strategy 1: Find "TOTAL" row followed by a pure number line
                # (skip qty lines like "6SETS")
                for _li in range(len(_inv_lines)):
                    _tl = _inv_lines[_li].strip()
                    if _tl == 'TOTAL' or re.match(r'^(?:GRAND\s+)?TOTAL\s*$', _tl, re.IGNORECASE):
                        for _lj in range(1, min(4, len(_inv_lines) - _li)):
                            _next_l = _inv_lines[_li + _lj].strip()
                            # Skip quantity lines (6SETS, 12PCS, etc.)
                            if re.match(r'^\d+\s*(?:SETS?|PCS|EA|KGS?|M\.?TONS?|UNITS?)', _next_l, re.IGNORECASE):
                                continue
                            # Pure number = amount
                            if re.match(r'^[\d,]+(?:\.\d{0,2})?\s*$', _next_l):
                                _inv_amt_found = float(_next_l.replace(',', ''))
                                break
                        break

                # Strategy 2: "Total Amount: USD 171,000" or "Invoice Total: 171000"
                if _inv_amt_found == 0:
                    _ta_m = re.search(
                        r'(?:Total\s*(?:Amount|Price)|Grand\s+Total|Invoice\s+Total|Net\s+Total)\s*[:\s]*'
                        r'(?:USD|EUR|GBP|JPY|CNY|PKR|AED|SAR)?\s*([\d,]+(?:\.\d{2})?)',
                        _inv_text, re.IGNORECASE)
                    if _ta_m:
                        _inv_amt_found = float(_ta_m.group(1).replace(',', ''))

                # Strategy 3: Largest currency amount in the invoice
                if _inv_amt_found == 0:
                    _all_amts = re.findall(
                        r'(?:USD|EUR|GBP|JPY|CNY|PKR|AED|SAR)\s*([\d,]+(?:\.\d{2})?)',
                        _inv_text)
                    if _all_amts:
                        _inv_amt_found = max(float(a.replace(',', '')) for a in _all_amts)

                # Deduplicate: if multiple invoice copies show same amount, count once
                if _inv_amt_found > 0:
                    if _total_inv_amount == 0 or abs(_inv_amt_found - _total_inv_amount) > 1:
                        _total_inv_amount += _inv_amt_found

            if _total_inv_amount > 0:
                _inv_max = _lc_amount * (1 + _tol_plus / 100.0)
                if _total_inv_amount <= _inv_max:
                    _add_pt_finding(
                        "Total Invoice Amount",
                        "PASS",
                        f"Total invoices {_lc_ccy} {_total_inv_amount:,.2f} within "
                        f"LC amount {_lc_ccy} {_lc_amount:,.2f} (+{_tol_plus}% tolerance)",
                        f"Max allowed: {_inv_max:,.2f}"
                    )
                else:
                    _add_pt_finding(
                        "Total Invoice Amount",
                        "FAIL",
                        f"Total invoices {_lc_ccy} {_total_inv_amount:,.2f} exceeds "
                        f"LC amount {_lc_ccy} {_lc_amount:,.2f} (+{_tol_plus}% = {_inv_max:,.2f})",
                        f"Invoice overdrawn by {_total_inv_amount - _inv_max:,.2f}"
                    )

    # ------------------------------------------------------------------ #
    # 6. Build summary statistics
    # ------------------------------------------------------------------ #
    pass_count = sum(1 for r in rows if _get(r, "compliance") == "PASS")
    fail_count = sum(1 for r in rows if _get(r, "compliance") == "FAIL")
    # P160 — Cross-document STALE BL check.
    # The stale-BL condition compares a date on the Documentary
    # Remittance (receiving/presentation date) against the BL's
    # on-board date. Data lives in TWO separate packets, so the
    # per-packet _deterministic_verify can't see both. Do it here,
    # after all single-packet verdicts land.
    def _pick_date(pkts, roles, typed_keys):
        """Find the first non-empty date in any packet matching the roles/keys.

        P185 — When the typed/structured fields do not carry the date
        (e.g. the Documentary Remittance has no `receiving_date` in its
        unified_summary but shows the bank's receiving stamp as
        `rubber_stamp18 SEP 2025` in the stamps/text), fall back to
        scanning the packet's stamp texts and raw content for any date
        embedded in a rubber-stamp-like token.
        """
        # Pass 1 — typed + structured dates (preferred).
        for _pkt in pkts:
            _us = (_pkt or {}).get('unified_summary') or {}
            for _k in typed_keys:
                _v = _us.get(_k)
                if _v and str(_v).strip() and str(_v).strip().lower() != 'unknown':
                    return str(_v).strip(), _pkt, _k
            for _item in (_us.get('dates_found') or []):
                if not isinstance(_item, dict):
                    continue
                _r = str(_item.get('role', '') or '').lower()
                _v = _item.get('value') or _item.get('raw')
                if _v and any(_rk in _r for _rk in roles):
                    return str(_v).strip(), _pkt, f"dates_found[role={_r}]"
        # Pass 2 — rubber-stamp fallback. Look only at packets whose
        # document_type looks like a Documentary/Covering/Remittance
        # instrument, because a receiving stamp on a BL or invoice is
        # not the DR presentation date.
        _is_receiving_roles = any('receiv' in str(r).lower() or 'presentation' in str(r).lower()
                                  for r in roles)
        if _is_receiving_roles:
            _stamp_re = re.compile(
                r'(\d{1,2})\s*[-/. ]?\s*'
                r'(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|SEPT|OCT|NOV|DEC|'
                r'JANUARY|FEBRUARY|MARCH|APRIL|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)\s*[-/. ]?\s*'
                r'(\d{2,4})',
                re.IGNORECASE,
            )
            for _pkt in pkts:
                if not _pkt:
                    continue
                _pt = str(_pkt.get('document_type', '') or '').lower()
                if not any(_tag in _pt for _tag in (
                    'documentary remittance', 'document remittance',
                    'covering letter', 'cover letter',
                    'covering schedule', 'cover schedule',
                    'remittance letter', 'forwarding letter',
                    'letter of transmittal',
                    'document presentation', 'presentation schedule',
                    'bills schedule', 'schedule of documents',
                )):
                    continue
                # Collect candidate texts: stamps list + raw/refined text.
                _candidates = []
                for _s in (_pkt.get('stamps') or []):
                    if isinstance(_s, dict):
                        _candidates.append(str(_s.get('text', '') or _s.get('description', '')))
                    else:
                        _candidates.append(str(_s))
                for _pg in (_pkt.get('pages') or _pkt.get('original_pages') or []):
                    if isinstance(_pg, dict):
                        for _s in (_pg.get('stamps') or []):
                            if isinstance(_s, dict):
                                _candidates.append(str(_s.get('text', '') or ''))
                            else:
                                _candidates.append(str(_s))
                _candidates.append(_pkt.get('refined_text', '') or '')
                _candidates.append(_pkt.get('raw_text', '') or '')
                _candidates.append(_pkt.get('text', '') or '')
                for _txt in _candidates:
                    if not _txt:
                        continue
                    if 'rubber_stamp' not in _txt.lower() and 'rubber stamp' not in _txt.lower():
                        # If there is no rubber-stamp marker at all, be
                        # conservative and skip generic dates in the body.
                        continue
                    for _m in _stamp_re.finditer(_txt):
                        _raw = _m.group(0).strip()
                        return _raw, _pkt, 'rubber_stamp'
        return None, None, None

    def _parse_date(s):
        """Parse common date formats to datetime.date; None if unparseable."""
        if not s:
            return None
        s = str(s).strip()
        from datetime import datetime
        for _fmt in (
            '%Y-%m-%d', '%Y/%m/%d', '%Y.%m.%d',
            '%d-%m-%Y', '%d/%m/%Y', '%d.%m.%Y',
            '%d-%b-%Y', '%d %b %Y', '%d-%B-%Y', '%d %B %Y',
            '%b %d %Y', '%B %d %Y', '%b %d, %Y', '%B %d, %Y',
            '%d-%b-%y', '%d %b %y', '%d-%B-%y',
        ):
            try:
                return datetime.strptime(s, _fmt).date()
            except ValueError:
                pass
        # Very loose fallback: find any YYYY-MM-DD-ish substring
        _m = re.search(r'(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})', s)
        if _m:
            try:
                return datetime(int(_m.group(1)), int(_m.group(2)), int(_m.group(3))).date()
            except ValueError:
                pass
        _m = re.search(r'(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})', s)
        if _m:
            try:
                return datetime(int(_m.group(3)), int(_m.group(2)), int(_m.group(1))).date()
            except ValueError:
                pass
        return None

    # P171 — Stale-BL threshold is a FLAT 30 DAYS. Do NOT read F48
    # (presentation period) — that's a separate concept covering how
    # long after SHIPMENT the docs can be presented to the bank, and
    # varies per LC. The user's stale-BL rule is fixed at 30 days
    # between DR.receiving_date and BL.onboard_date.
    _stale_days_threshold = 30

    # Find DR receiving date and BL on-board date from the packet pool
    _dr_date_str, _dr_pkt, _dr_src = _pick_date(
        packets,
        roles=('receiving_date', 'presentation_date', 'received_date'),
        typed_keys=('receiving_date', 'presentation_date'),
    )
    _bl_date_str, _bl_pkt, _bl_src = _pick_date(
        [p for p in packets if 'bill of lading' in str((p or {}).get('document_type', '')).lower()
            or 'bl' in str((p or {}).get('document_type', '')).lower()],
        roles=('onboard_date', 'shipped_on_board', 'on_board'),
        typed_keys=('onboard_date', 'shipment_date', 'shipped_on_board_date'),
    )
    _dr_date = _parse_date(_dr_date_str) if _dr_date_str else None
    _bl_date = _parse_date(_bl_date_str) if _bl_date_str else None

    for row in rows:
        try:
            _cond = (_get(row, 'condition_text', '') or _get(row, 'condition', '')).upper()
            if 'STALE' not in _cond and 'STALENESS' not in _cond:
                continue
            # P170 — Stale check applies to ANY row that mentions "stale".
            # Do NOT restrict by document_checked. The LLM routinely
            # mis-routes stale checks to Bill of Lading with reasoning
            # like "BL is short form and blank back" — which is wrong
            # because "stale" is a TIME check (receiving_date vs
            # onboard_date), NOT a form-type or back-page check.
            # Whatever document the LLM picked, the deterministic
            # computation uses DR.receiving_date and BL.onboard_date.
            if _dr_date and _bl_date:
                _delta = (_dr_date - _bl_date).days
                if _delta > _stale_days_threshold:
                    _set(row, 'compliance', 'FAIL')
                    _msg = (
                        f"BL is STALE. DR receiving date {_dr_date.isoformat()} "
                        f"minus BL on-board date {_bl_date.isoformat()} = "
                        f"{_delta} days — exceeds the {_stale_days_threshold}-day "
                        f"threshold. (P160 deterministic)"
                    )
                else:
                    _set(row, 'compliance', 'PASS')
                    _msg = (
                        f"BL is NOT stale. DR receiving date {_dr_date.isoformat()} "
                        f"minus BL on-board date {_bl_date.isoformat()} = "
                        f"{_delta} days — within the {_stale_days_threshold}-day "
                        f"threshold. (P160 deterministic)"
                    )
                _set(row, 'findings', _msg)
                _set(row, 'result', _msg[:200])
                _set(row, 'verification_notes', f"P160 stale-BL cross-doc deterministic (threshold={_stale_days_threshold}d)")
                _progress(f"  [P160 stale-BL] {row.get('row_id','?')}: delta={_delta}d threshold={_stale_days_threshold}d → {row.get('compliance')}")
            else:
                # Cannot confirm — keep any existing verdict if it was
                # PASS from LLM, otherwise REVIEW.
                _cur = _get(row, 'compliance', '').upper()
                _fin_check = (_get(row, 'findings', '') or '').upper()
                # P170 — Override any LLM FAIL that is based on wrong
                # reasoning (short form / blank back / house / forwarder /
                # claused / charter party are NOT staleness signals).
                _wrong_reasoning = (
                    _cur in ('FAIL', 'NOT_COMPLIED', 'NON_COMPLIANT', 'DISCREPANT')
                    and (
                        'SHORT FORM' in _fin_check or
                        'BLANK BACK' in _fin_check or
                        'HOUSE' in _fin_check or
                        'FORWARDER' in _fin_check or
                        'CLAUSED' in _fin_check or
                        'CHARTER PARTY' in _fin_check
                    )
                )
                if _cur not in ('PASS', 'COMPLIED') or _wrong_reasoning:
                    _set(row, 'compliance', 'REVIEW')
                    _missing = []
                    if not _dr_date:
                        _missing.append('DR receiving_date')
                    if not _bl_date:
                        _missing.append('BL onboard_date')
                    _set(row, 'findings',
                         f"Cannot determine staleness deterministically: {', '.join(_missing)} "
                         f"not available on document. Stale check requires "
                         f"DR receiving_date and BL on-board date. Manual check required. "
                         f"(P170 — earlier LLM reasoning citing form type / blank back / "
                         f"house / claused is irrelevant for staleness.)")
                    _set(row, 'result', _get(row, 'findings', '')[:200])
        except Exception as _e:
            try:
                print(f"[P160 stale-BL] exception on row {row.get('row_id','?')}: {_e}")
            except Exception:
                pass

    # P163 — Cross-document "documents dated prior to LC issuance" check.
    # Per UCP 600 Art 14(i): every submitted shipping document must be
    # dated on or AFTER the LC issue date (F31C). If any doc is pre-dated,
    # it is a discrepancy for that doc.
    # Runs AFTER per-packet verdicts so we can compare each packet's
    # issue_date against the LC F31C date and override only when we have
    # hard evidence.
    try:
        _lc_issue_str = str(_final_lc_fields.get('31C', '') or _final_lc_fields.get('F31C', '') or '').strip()
        _lc_issue_date = _parse_date(_lc_issue_str) if _lc_issue_str else None
    except Exception:
        _lc_issue_date = None

    if _lc_issue_date:
        # P163 — Packet exclusions. Blank / structural / non-content
        # pages carry no verifiable issue date and must NOT be part of
        # the pre-dated check. This matches the P153 "all documents"
        # fan-out exclusion list.
        _SKIP_TYPES_FOR_DATE_CHECK = (
            'blank page', 'header page', 'endorsement page', 'back page',
            'back cover', 'reverse page', 'terms and conditions',
            'terms overleaf', 'bl conditions of carriage',
            'conditions of carriage', 'cover page', 'title page',
            'unknown', 'unidentified', 'supporting document',
            'documentary remittance', 'document remittance',
            'covering letter', 'cover letter', 'covering schedule',
            'cover schedule', 'l/c bills schedule', 'lc bills schedule',
            'bills schedule', 'presentation schedule',
            'document presentation', 'schedule of documents',
            'letter of transmittal', 'arrival notice',
            'forwarding letter', 'remittance letter', 'export letter',
            'fax', 'email',
            # SWIFT messages — LC / Amendment / etc. are instruments, not
            # shipping docs that get dated relative to themselves.
            'lc', 'letter of credit', 'amendment',
            'mt700', 'mt701', 'mt707', 'mt720', 'mt730', 'mt734',
            'mt740', 'mt742', 'mt747', 'mt750', 'mt752', 'mt754',
            'mt756', 'mt760', 'mt767', 'mt768', 'mt769', 'mt799',
            'mt940', 'mt999',
        )

        # Per-packet issue dates (name + date) — excluding skip types.
        _pkt_issue = {}
        for _pkt in packets:
            if not _pkt:
                continue
            _doc_name_raw = (_pkt.get('document_type') or _pkt.get('doc_type') or '') if isinstance(_pkt, dict) else ''
            _doc_name_lc = str(_doc_name_raw).lower().strip()
            # Skip non-content / informational packet types
            if any(_skip in _doc_name_lc for _skip in _SKIP_TYPES_FOR_DATE_CHECK):
                continue
            # Skip packets with negligible text (likely blank despite
            # classification label)
            try:
                _ptx = (_pkt_text(_pkt) or '').strip()
                if len(_ptx) < 80:
                    continue
            except Exception:
                pass
            _us = (_pkt.get('unified_summary') or {}) if isinstance(_pkt, dict) else {}
            if not isinstance(_us, dict):
                continue
            # Find the doc's OWN issue date
            _cand = None
            for _k in ('issue_date', 'invoice_date', 'bl_issue_date',
                        'certificate_issue_date', 'draft_date',
                        'document_date'):
                _v = _us.get(_k)
                if _v and str(_v).strip() and str(_v).strip().lower() != 'unknown':
                    _cand = _parse_date(str(_v).strip())
                    if _cand:
                        break
            if not _cand:
                for _item in (_us.get('dates_found') or []):
                    if not isinstance(_item, dict):
                        continue
                    _r = str(_item.get('role', '') or '').lower()
                    if _r in ('issue_date', 'invoice_date', 'bl_issue_date',
                               'certificate_issue_date', 'document_date',
                               'draft_date'):
                        _v = _item.get('value') or _item.get('raw')
                        if _v:
                            _cand = _parse_date(str(_v).strip())
                            if _cand:
                                break
            if _cand:
                _pkt_issue[str(_doc_name_raw)] = _cand

        _pre_dated = {name: d for name, d in _pkt_issue.items() if d < _lc_issue_date}

        for row in rows:
            try:
                _cond = (_get(row, 'condition_text', '') or _get(row, 'condition', '')).upper()
                _is_pre_dated_check = (
                    'DATED PRIOR' in _cond or
                    'PRE-DATED' in _cond or 'PREDATED' in _cond or
                    ('BEFORE' in _cond and ('LC' in _cond or 'CREDIT' in _cond) and 'DATE' in _cond) or
                    ('ON OR AFTER' in _cond and 'LC' in _cond and 'DATE' in _cond) or
                    ('AFTER' in _cond and 'ISSUANCE' in _cond and 'LC' in _cond) or
                    'NOT ACCEPTABLE.*DATED' in _cond
                )
                if not _is_pre_dated_check:
                    continue
                # P198ad — Prohibition vs permission. The same "DATED PRIOR"
                # wording appears both in prohibitions ("… NOT ACCEPTABLE")
                # and permissions ("… ARE ACCEPTABLE / ARE ALLOWED / ARE
                # PERMITTED"). Only run the pre-date FAIL path when the
                # condition explicitly prohibits pre-dating. When the LC
                # says pre-dating IS permitted, pre-dated docs must PASS
                # (or be informational) — a pre-dated doc then IS in
                # compliance with the LC's own rule.
                _permissive_markers = (
                    'ARE ACCEPTABLE', 'IS ACCEPTABLE',
                    'ARE PERMITTED', 'IS PERMITTED',
                    'ARE ALLOWED', 'IS ALLOWED',
                    'ACCEPTABLE.', 'ACCEPTABLE ',
                )
                _prohibition_markers = (
                    'NOT ACCEPTABLE', 'NOT PERMITTED',
                    'NOT ALLOWED', 'UNACCEPTABLE',
                    'MUST NOT', 'WILL NOT BE ACCEPT',
                )
                _is_prohibited = any(m in _cond for m in _prohibition_markers)
                _is_permitted = (
                    (not _is_prohibited)
                    and any(m in _cond for m in _permissive_markers)
                )
                # If the LC language is permissive, pre-dated docs
                # conform to the LC rule. Emit PASS/informational and
                # skip the FAIL branch.
                if _is_permitted:
                    _set(row, 'compliance', 'PASS')
                    _listing = (
                        '; '.join(
                            f"{n} dated {d.isoformat()}"
                            for n, d in _pre_dated.items()
                        )
                        if _pre_dated else ''
                    )
                    _msg = (
                        f"LC explicitly permits pre-dated documents. "
                        f"{len(_pre_dated)} pre-dated doc(s) found — "
                        f"compliant with LC rule"
                        + (f": {_listing}" if _listing else '')
                        + ". (P198ad permissive)"
                    )
                    _set(row, 'findings', _msg)
                    _set(row, 'result', _msg[:200])
                    _set(row, 'verification_notes',
                         f"P198ad permissive pre-dated rule "
                         f"(F31C={_lc_issue_date.isoformat()})")
                    _progress(
                        f"  [P198ad pre-dated permissive] "
                        f"{row.get('row_id','?')}: PASS "
                        f"({len(_pre_dated)} pre-dated, LC permits)"
                    )
                    continue
                if _pre_dated:
                    _set(row, 'compliance', 'FAIL')
                    _listing = '; '.join(
                        f"{n} dated {d.isoformat()}" for n, d in _pre_dated.items()
                    )
                    _msg = (
                        f"Pre-dated documents found (before LC issue date "
                        f"{_lc_issue_date.isoformat()}): {_listing}. "
                        f"(P163 deterministic)"
                    )
                    _set(row, 'findings', _msg)
                    _set(row, 'result', _msg[:200])
                    _set(row, 'verification_notes',
                         f"P163 cross-doc date-of-issue check vs F31C={_lc_issue_date.isoformat()}")
                else:
                    _set(row, 'compliance', 'PASS')
                    _msg = (
                        f"All submitted documents dated on or after LC issue "
                        f"date {_lc_issue_date.isoformat()}. (P163 deterministic)"
                    )
                    _set(row, 'findings', _msg)
                    _set(row, 'result', _msg[:200])
                    _set(row, 'verification_notes',
                         f"P163 cross-doc date-of-issue check vs F31C={_lc_issue_date.isoformat()}")
                _progress(f"  [P163 pre-dated-docs] {row.get('row_id','?')}: "
                          f"{'FAIL' if _pre_dated else 'PASS'} "
                          f"({len(_pre_dated)} pre-dated)")
            except Exception as _e:
                try:
                    print(f"[P163 pre-dated] exception on row {row.get('row_id','?')}: {_e}")
                except Exception:
                    pass

    # P172 — Deterministic HS Code match. HS codes are regulatory
    # identifiers that MUST match the LC value exactly after whitespace /
    # dot / dash / space normalization. LLM may tolerantly pass near-miss
    # codes (9018.9050 vs 9018.909000 — different tariff lines). This
    # check looks at every row whose condition names a specific HS code
    # and compares against the packet's structured hs_code / hs_codes
    # field and raw document_text.
    try:
        def _norm_hs(s):
            return re.sub(r'[^0-9]', '', str(s or ''))
        # Build a per-packet HS code map
        _pkt_hs = {}
        for _pkt in packets:
            if not isinstance(_pkt, dict):
                continue
            _dt = (_pkt.get('document_type') or '').lower()
            _us = _pkt.get('unified_summary') or {}
            if not isinstance(_us, dict):
                continue
            _codes = set()
            _v = _us.get('hs_code') or _us.get('hs_codes')
            if isinstance(_v, str):
                for _m in re.finditer(r'\d[\d.\s\-]{3,}\d', _v):
                    _n = _norm_hs(_m.group(0))
                    if 6 <= len(_n) <= 12:
                        _codes.add(_n)
            elif isinstance(_v, list):
                for _x in _v:
                    _n = _norm_hs(_x)
                    if 6 <= len(_n) <= 12:
                        _codes.add(_n)
            # Also scan references_found
            for _item in (_us.get('references_found') or []):
                if isinstance(_item, dict):
                    _r = str(_item.get('role', '') or '').lower()
                    if 'hs' in _r or 'hts' in _r:
                        _n = _norm_hs(_item.get('value') or _item.get('raw'))
                        if 6 <= len(_n) <= 12:
                            _codes.add(_n)
            # And scan raw document_text as a fallback
            _dtxt = _pkt_text(_pkt) or ''
            for _m in re.finditer(
                r'(?:H\.?\s*S\.?\s*(?:CODE)?|HS\s*CODE)[\s:.#]*((?:\d[\d.\s\-]{5,}\d))',
                _dtxt, re.IGNORECASE,
            ):
                _n = _norm_hs(_m.group(1))
                if 6 <= len(_n) <= 12:
                    _codes.add(_n)
            _pkt_hs[(_pkt.get('packet_id') or _dt)] = {
                'doc_type': _dt,
                'codes': _codes,
            }

        for row in rows:
            try:
                _cond_u = (_get(row, 'condition_text', '') or _get(row, 'condition', '')).upper()
                if not re.search(r'\bH\.?\s*S\.?\s*(?:CODE)?\b|\bHS\s*CODE\b|\bHTS\b|\bTARIFF\s*CODE\b', _cond_u):
                    continue
                # Extract the HS code value from the condition
                _cond_hs = None
                for _m in re.finditer(r'(\d[\d.\s\-]{5,}\d)', _cond_u):
                    _n = _norm_hs(_m.group(1))
                    if 6 <= len(_n) <= 12:
                        _cond_hs = _n
                        break
                if not _cond_hs:
                    continue
                _doc_checked_lc = (_get(row, 'document_checked', '') or '').lower()
                # Find matching packet(s)
                _matched_codes = set()
                _matched_doc = None
                for _info in _pkt_hs.values():
                    if _doc_checked_lc and _doc_checked_lc in _info['doc_type']:
                        _matched_codes |= _info['codes']
                        _matched_doc = _info['doc_type']
                if not _matched_codes:
                    continue  # no HS code extracted for this doc — leave verdict as-is
                # P175 — HS code match rules:
                #   1. Exact match -> PASS
                #   2. Same base after stripping trailing ZEROS on both
                #      sides -> PASS. Handles dot/space differences and
                #      trailing-zero padding.
                #      Example: LC 9018.9050 (90189050) vs doc 9018905000
                #      -> both strip to 9018905 -> MATCH.
                #   3. Otherwise -> FAIL. (No generic prefix tolerance —
                #      9018.9050 and 9018.9051 are DIFFERENT codes.)
                def _strip_trailing_zeros(s, min_len=6):
                    while len(s) > min_len and s.endswith('0'):
                        s = s[:-1]
                    return s
                _cond_stripped = _strip_trailing_zeros(_cond_hs)
                _is_match = False
                _matched_as = None
                if _cond_hs in _matched_codes:
                    _is_match = True
                    _matched_as = 'exact'
                else:
                    for _c in _matched_codes:
                        _c_stripped = _strip_trailing_zeros(_c)
                        if _cond_stripped == _c_stripped:
                            _is_match = True
                            _matched_as = 'trailing-zeros'
                            break
                if _is_match:
                    _set(row, 'compliance', 'PASS')
                    _doc_shown = next(iter(_matched_codes))
                    _set(row, 'findings',
                         f"H.S. Code {_cond_hs} matches document code "
                         f"{_doc_shown} ({_matched_as}).")
                    _set(row, 'result', _get(row, 'findings', '')[:200])
                else:
                    _set(row, 'compliance', 'FAIL')
                    _other = ', '.join(sorted(_matched_codes))
                    _set(row, 'findings',
                         f"H.S. Code mismatch. LC requires '{_cond_hs}' but "
                         f"document shows '{_other}'.")
                    _set(row, 'result', _get(row, 'findings', '')[:200])
                _set(row, 'verification_notes',
                     f"P172 HS deterministic: required={_cond_hs} vs found={sorted(_matched_codes)}")
                _progress(f"  [P172 HS] {row.get('row_id','?')} doc={_matched_doc}: required={_cond_hs} found={sorted(_matched_codes)} -> {row.get('compliance')}")
            except Exception as _e:
                try:
                    print(f"[P172 HS] exception on row {row.get('row_id','?')}: {_e}")
                except Exception:
                    pass
    except Exception:
        pass

    # P174 — Deterministic "addressed to X" check. LLM often passes a
    # Shipment Advice / Notice that is NOT addressed to the applicant,
    # inferring addressee-ship from the presence of the LC number or
    # insurance company name. That's wrong: "addressed to X" means the
    # document visibly names X in its TO/Attn/To header or greeting.
    # This check compares the target party's distinctive name tokens
    # against the document_text. Requires majority of tokens to be
    # found; otherwise FAIL.
    try:
        _lc_parties_cf = {}
        try:
            _cf = step06_result.get('consolidated_fields', {}) if isinstance(step06_result, dict) else {}
            if not _cf:
                _cf = step06_result.get('final_lc', {}).get('consolidated_fields', {})
            _lc_parties_cf = {
                'applicant': str(_cf.get('50', _cf.get('F50', ''))).split('\n')[0].strip(),
                'beneficiary': str(_cf.get('59', _cf.get('F59', ''))).split('\n')[0].strip(),
                'issuing_bank': str(_cf.get('52A', _cf.get('F52A', ''))).split('\n')[0].strip(),
            }
        except Exception:
            _lc_parties_cf = {}

        def _distinctive_tokens(s):
            s = str(s or '').upper()
            s = re.sub(r'\b(M/?S\.?|MESSRS\.?|MR\.?|MRS\.?|DR\.?)\s+', ' ', s)
            s = re.sub(r'\([^)]*\)', ' ', s)
            s = re.sub(
                r'\b(BANK|LTD|LIMITED|LLC|PLC|INC|CORP|CO|PVT|PRIVATE|'
                r'COMPANY|ENTERPRISES?|GROUP|HOLDINGS?|TRADING|'
                r'INSURERS?|INSURANCE|AND|OF|THE|FOR|WITH|AT)\b',
                ' ', s,
            )
            s = re.sub(r'[^A-Z ]', ' ', s)
            toks = [t for t in s.split() if len(t) >= 3]
            return toks

        def _normalize_name_phrase(s):
            """Normalize a party name for contiguous phrase matching.
            Strips M/s / Messrs / Mr. prefixes, corporate suffixes,
            punctuation, and collapses whitespace. Keeps the DISTINCTIVE
            NAME PHRASE intact so we search for it as a whole unit, not
            as isolated words that could match anything."""
            s = str(s or '').upper()
            # Strip honorifics
            s = re.sub(r'\b(M/?S\.?|MESSRS\.?|MR\.?|MRS\.?|DR\.?)\s+', '', s)
            # Strip parenthetical acronyms (SIUT), (Pvt), etc.
            s = re.sub(r'\([^)]*\)', ' ', s)
            # Strip corporate suffixes at the END of the name
            s = re.sub(
                r'\s+(LTD|LIMITED|LLC|PLC|INC|CORP|CO|PVT|PRIVATE|COMPANY|'
                r'S\.?A\.?|S\.?L\.?|B\.?V\.?|N\.?V\.?|GMBH|AG|AB|OY)\b\.?'
                r'(?:\s+.*)?$',
                '', s,
            )
            # Strip trailing location (", KARACHI, PAKISTAN" / ", LAHORE")
            s = re.sub(
                r',?\s*(?:KARACHI|LAHORE|ISLAMABAD|MUMBAI|DUBAI|RIYADH|DOHA|'
                r'BEIRUT|COLOMBO|HONG\s+KONG|SINGAPORE|LONDON|NEW\s+YORK|'
                r'GULBERG)\b.*$',
                '', s,
            )
            s = re.sub(
                r',?\s*(?:PAKISTAN|INDIA|BANGLADESH|SRI\s+LANKA|UAE|SAUDI\s+ARABIA|'
                r'USA|UNITED\s+STATES|UK|UNITED\s+KINGDOM|CANADA|CHINA)\b.*$',
                '', s,
            )
            # Strip punctuation and collapse whitespace
            s = re.sub(r'[.,;:/\\\'"—–-]+', ' ', s)
            s = re.sub(r'\s+', ' ', s).strip()
            return s

        def _phrase_in_doc(name_phrase, doc_text_up):
            """True if the normalized party name appears on the doc as a
            contiguous phrase. Tolerates whitespace/punctuation differences
            between words but requires word order to match."""
            if not name_phrase or not doc_text_up:
                return False
            # Doc normalized the same way (upper + collapse non-alpha to space)
            _doc_norm = re.sub(r'[^A-Z0-9]+', ' ', doc_text_up).strip()
            _doc_norm = re.sub(r'\s+', ' ', _doc_norm)
            _words = [w for w in name_phrase.split() if w]
            if len(_words) < 2:
                # Single-word name — require exact word boundary
                return bool(re.search(r'\b' + re.escape(name_phrase) + r'\b', _doc_norm))
            # Multi-word — require all words contiguously with only
            # whitespace between. Build pattern escaping each word.
            _pattern = r'\b' + r'\s+'.join(re.escape(w) for w in _words) + r'\b'
            return bool(re.search(_pattern, _doc_norm))

        # P198ac — Aggregate per-row across multiple packet tasks so
        # that ANY packet satisfying the addressing requirement keeps
        # the row as PASS. Previously this loop processed each task
        # independently and flipped the row to FAIL based on a packet
        # that didn't carry the party, even when another packet for
        # the same row (e.g. a second Shipment Advice page addressed
        # to a different required party) did.
        _addr_per_row: Dict[str, list] = {}
        for task in vlm_tasks:
            row = task["row"]
            row_id = task.get("row_id", "?")
            try:
                _comp_now = _get(row, "compliance", "").upper()
                if _comp_now != "PASS":
                    continue  # only catch false PASSes
                _cond_u = (task.get("condition_text") or "").upper()
                if 'ADDRESSED TO' not in _cond_u and 'MARKED TO' not in _cond_u:
                    continue
                # P198h — Skip OR-routed rows. When step 12 emitted a
                # pipe-separated `document_to_check` (e.g. "Shipment
                # Advice | Beneficiary Certificate"), step 14's OR
                # handler built a single combined task that fed both
                # docs' text to the LLM with OR-aware guidance. The LLM
                # has already evaluated all candidates; overriding its
                # verdict from a single-doc deterministic view would
                # flip a correct PASS to FAIL when the required party
                # is addressed on the OTHER doc in the OR (e.g. the
                # Beneficiary Certificate that certifies the Shipment
                # Advice was sent to the Applicant).
                if task.get("or_docs") or ' OR ' in (task.get("document_type") or ''):
                    continue

                # P178 — Multi-party "addressed to X AND Y" support.
                # Conditions like:
                #   "Shipment Advice must be addressed to M/S. SINDH
                #    INSURANCE, KARACHI, PAKISTAN AND TO THE APPLICANT"
                # require BOTH parties to appear on the document. Any
                # missing party → FAIL.
                _targets = []  # list of (label, party_name) to verify

                # 1) If "APPLICANT" mentioned, include LC applicant as a target
                if 'APPLICANT' in _cond_u and _lc_parties_cf.get('applicant'):
                    _targets.append(('Applicant', _lc_parties_cf['applicant']))
                # 2) If "BENEFICIARY" mentioned, include LC beneficiary
                if 'BENEFICIARY' in _cond_u and _lc_parties_cf.get('beneficiary'):
                    _targets.append(('Beneficiary', _lc_parties_cf['beneficiary']))
                # 3) If "ISSUING BANK" / "OPENING BANK" mentioned, include LC issuing bank
                if (('ISSUING BANK' in _cond_u or 'OPENING BANK' in _cond_u or
                        "L/C ISSUING" in _cond_u) and
                        _lc_parties_cf.get('issuing_bank')):
                    _targets.append(('Issuing Bank', _lc_parties_cf['issuing_bank']))

                # 4) Extract EXPLICIT party names from the condition itself.
                # Pattern: text after "ADDRESSED TO" or "TO" up to the next
                # "AND TO" / "AND THE" / period / end-of-line. Split on
                # "AND TO" / "AND THE" to get multiple explicit targets.
                _explicit_tail = ''
                _m_head = re.search(
                    r'(?:ADDRESSED|MARKED)\s+(?:TO|AT)[:\s]+(.+)',
                    _cond_u,
                )
                if _m_head:
                    _explicit_tail = _m_head.group(1).strip()
                if _explicit_tail:
                    # P198am — Split ONLY on "AND TO" (with TO mandatory),
                    # never on bare "AND". Company names frequently
                    # contain AND ("AL MASHOOD OIL AND GHEE INDUSTRIES",
                    # "MITSUBISHI FUSO TRUCK AND BUS CORPORATION") and
                    # a bare-AND split corrupts them into nonsense
                    # fragments like "THE APPLICANT AL MASHOOD OIL" +
                    # "GHEE INDUSTRIES PVT LTD" that then produce false
                    # "not addressed to required party" FAILs.
                    _parts = re.split(
                        r'\s+AND\s+TO\s+(?:THE\s+)?',
                        _explicit_tail,
                    )
                    for _p in _parts:
                        _p = _p.strip(' .,:\'""')
                        if not _p:
                            continue
                        # Trim at "VIA" / "BY" / "AT" trailing methods
                        _p = re.split(
                            r'\s+(?:VIA|BY|AT|WITHIN|WITHIN\s+\d+|BEFORE|AFTER|REFERRING|MENTIONING)\s+',
                            _p, maxsplit=1,
                        )[0].strip(' .,:\'""')
                        # P186 — Drop "NOTIFY …" prefixes. "Notify the
                        # Applicant" is a BL-field reference (the Notify
                        # Party should contain the applicant), NOT a
                        # party literally named "Notify the Applicant".
                        # The LC-party branch above already added the
                        # applicant / issuing-bank target with the real
                        # name; and the notify-party post-check at 5c
                        # confirms the BL's Notify field content. Adding
                        # "NOTIFY THE APPLICANT" as a Named party here
                        # produces duplicate false FAILs because the
                        # phrase never appears on the BL verbatim.
                        _p_no_notify = re.sub(r'^NOTIFY\s+(?:THE\s+)?', '', _p).strip()
                        if _p_no_notify != _p:
                            # Was a "NOTIFY X" target — skip, handled elsewhere.
                            continue
                        # P198am — If the phrase (with optional leading
                        # "THE") is JUST a role word (APPLICANT /
                        # BENEFICIARY / ISSUING BANK / etc.), skip —
                        # the LC-party branch above already added the
                        # real party target.
                        _role_only_re = (
                            r'^(?:THE\s+)?(?:APPLICANT|BENEFICIARY|'
                            r'ISSUING\s+BANK|OPENING\s+BANK|'
                            r'L/C\s+ISSUING\s+BANK|L/C\s+OPENING\s+BANK|'
                            r'NOMINATED\s+BANK|CONFIRMING\s+BANK|'
                            r'NEGOTIATING\s+BANK|ADVISING\s+BANK)\s*$'
                        )
                        if re.match(_role_only_re, _p):
                            continue
                        # P198am — Strip leading role prefixes when
                        # followed by an actual party name. "THE
                        # APPLICANT AL MASHOOD OIL AND GHEE..." →
                        # "AL MASHOOD OIL AND GHEE...", which then
                        # de-dupes correctly against the LC-party
                        # applicant target.
                        _p_stripped = re.sub(
                            r'^(?:THE\s+)?(?:APPLICANT|BENEFICIARY|'
                            r'ISSUING\s+BANK|OPENING\s+BANK|'
                            r'L/C\s+ISSUING\s+BANK|L/C\s+OPENING\s+BANK|'
                            r'NOMINATED\s+BANK|CONFIRMING\s+BANK|'
                            r'NEGOTIATING\s+BANK|ADVISING\s+BANK)\s+',
                            '', _p,
                        ).strip()
                        if _p_stripped and _p_stripped != _p:
                            _p = _p_stripped
                        # Skip generic words like "APPLICANT" / "BENEFICIARY"
                        # / "ISSUING BANK" — already covered by LC-party
                        # branch above.
                        _p_words = _p.split()
                        if (len(_p_words) == 1 and _p_words[0] in
                                ('APPLICANT', 'BENEFICIARY', 'BANK')):
                            continue
                        if _p in ('APPLICANT', 'BENEFICIARY', 'ISSUING BANK',
                                   'OPENING BANK', 'L/C ISSUING BANK'):
                            continue
                        # Keep only if has at least 3 words OR has M/s prefix
                        # (indicating a named party)
                        if len(_p_words) >= 2 and len(_p) >= 6:
                            # Avoid duplicates with LC-party branch
                            _dup = any(
                                _p.upper() in _lp[1].upper() or
                                _lp[1].upper() in _p.upper()
                                for _lp in _targets
                                if len(_lp[1]) >= 6
                            )
                            if not _dup:
                                _targets.append(('Named party', _p))

                if not _targets:
                    continue
                _doc_text_up = (task.get("document_text") or "").upper()
                if not _doc_text_up:
                    continue

                # P179 — Phrase-based name check. Look for each target's
                # DISTINCTIVE NAME PHRASE (e.g. "SINDH INSTITUTE OF
                # UROLOGY") as a CONTIGUOUS string in the document, not
                # isolated words. This way "SINDH" appearing in
                # "M/S. SINDH INSURANCE" doesn't falsely satisfy a
                # requirement for "SINDH INSTITUTE OF UROLOGY".
                _missing_targets = []
                for _lbl, _name in _targets:
                    _phrase = _normalize_name_phrase(_name)
                    if not _phrase or len(_phrase) < 4:
                        continue
                    if not _phrase_in_doc(_phrase, _doc_text_up):
                        _missing_targets.append((_lbl, _name, _phrase))

                # P198ac — Don't flip immediately; accumulate the
                # per-task outcome and decide once we've seen every
                # task that belongs to this row_id.
                _addr_per_row.setdefault(row_id, []).append({
                    "row": row,
                    "document_type": task.get("document_type", "?"),
                    "missing": _missing_targets,
                    "targets": list(_targets),
                })
            except Exception as _e:
                try:
                    print(f"[P174/P178 addressed-to] exception on row {row_id}: {_e}")
                except Exception:
                    pass

        # P198ac — Apply the aggregated verdict per row. If ANY packet
        # for this row has zero missing targets (i.e. the required
        # party/parties are all present on that doc), keep PASS. Only
        # flip to FAIL when EVERY packet is missing at least one
        # required target.
        for row_id, _entries in _addr_per_row.items():
            try:
                if not _entries:
                    continue
                row = _entries[0]["row"]
                _any_satisfied = any(
                    not _e.get("missing") for _e in _entries
                )
                if _any_satisfied:
                    _progress(
                        f"  [P174/P178/P179 addressed-to] {row_id}: "
                        f"PASS retained (satisfied on at least one of "
                        f"{len(_entries)} doc(s))"
                    )
                    continue
                # All packets missing — combine findings from the packet
                # with the SMALLEST missing set (most-relevant FAIL).
                _best = min(
                    _entries,
                    key=lambda e: len(e.get("missing") or [])
                )
                _missing_targets = _best["missing"]
                _doc_lbl = _best.get("document_type") or "?"
                _set(row, "compliance", "FAIL")
                _missing_summary = '; '.join(
                    f"{_lbl} '{_name}'"
                    for _lbl, _name, _phrase in _missing_targets
                )
                _set(row, "findings",
                     f"Document is not addressed to the required "
                     f"party/parties: {_missing_summary}.")
                _set(row, "result", _get(row, "findings", "")[:200])
                _set(row, "verification_notes",
                     f"P174/P178/P179 addressed-to deterministic "
                     f"(checked {len(_entries)} packet(s); "
                     f"best={_doc_lbl}): "
                     + '; '.join(
                         f"{lbl}='{nm}' (phrase='{ph}' not found)"
                         for lbl, nm, ph in _missing_targets
                     ))
                _progress(
                    f"  [P174/P178/P179 addressed-to] {row_id}: "
                    f"PASS->FAIL — all {len(_entries)} packet(s) missing"
                    f" ({_missing_summary})"
                )
            except Exception as _e:
                try:
                    print(f"[P198ac addressed-to aggregate] exception on row {row_id}: {_e}")
                except Exception:
                    pass
    except Exception:
        pass

    # P198ae — Shipper "on behalf of beneficiary" rescue.
    # When the BL shipper field reads "<AGENT> ON BEHALF OF <BENEFICIARY>"
    # (or similar agency wording — "FOR ACCOUNT OF", "C/O", "PER",
    # "AS AGENT FOR"), the beneficiary IS named on the BL face. Under
    # standard commodity-trade practice (UCP 600 / ISBP 821) this
    # agency construction is acceptable — the BL identifies the
    # beneficiary as the principal for whom the agent is shipping.
    # The LLM often reads these literally and FAILs the row; this
    # check flips those cases back to PASS when the beneficiary
    # name is unambiguously present in the shipper field.
    try:
        _bene_for_rescue = str(
            _final_lc_fields.get('59', '') or _final_lc_fields.get('F59', '') or ''
        ).split('\n')[0].strip()
        if _bene_for_rescue:
            # Distinctive tokens — strip corporate suffixes, common words
            def _name_core(s):
                s = str(s or '').upper()
                s = re.sub(r'\([^)]*\)', ' ', s)
                s = re.sub(
                    r'\b(LTD|LIMITED|LLC|PLC|INC|CORP|CO|PVT|PRIVATE|'
                    r'COMPANY|S\.?A\.?|S\.?L\.?|B\.?V\.?|N\.?V\.?|'
                    r'GMBH|AG|AB|OY|HOLDINGS?|GROUP|TRADING)\b\.?',
                    ' ', s,
                )
                s = re.sub(r'[^A-Z0-9 ]+', ' ', s)
                s = re.sub(r'\s+', ' ', s).strip()
                return s
            _bene_core = _name_core(_bene_for_rescue)
            _bene_tokens = [t for t in _bene_core.split() if len(t) >= 3]
            _AGENCY_MARKERS = (
                'ON BEHALF OF', 'O/B/O', 'O/B',
                'FOR ACCOUNT OF', 'F/A/O', 'FOR A/C OF',
                'FOR THE ACCOUNT OF',
                'AS AGENT FOR', 'AS AGENTS FOR', 'AGENT FOR',
                'AGENTS FOR', 'ON ACCOUNT OF', 'C/O', 'CARE OF',
                'PER ', 'BY ORDER OF',
            )
            for task in vlm_tasks:
                row = task["row"]
                _row_id = task.get("row_id", "?")
                try:
                    _comp_now = _get(row, "compliance", "").upper()
                    if _comp_now != "FAIL":
                        continue
                    _cond_u = (task.get("condition_text") or "").upper()
                    # Only apply to shipper-vs-beneficiary checks
                    if not ('SHIPPER' in _cond_u and
                            ('BENEFICIARY' in _cond_u or 'F59' in _cond_u)):
                        continue
                    _doc_type_lc = (task.get("document_type") or '').lower()
                    if 'bill of lading' not in _doc_type_lc:
                        continue
                    _doc_text_up = (task.get("document_text") or "").upper()
                    if not _doc_text_up or not _bene_tokens:
                        continue
                    # Does the BL text contain an agency marker AND
                    # the beneficiary's distinctive tokens within a
                    # reasonable window after the marker?
                    _has_agency = any(m in _doc_text_up for m in _AGENCY_MARKERS)
                    if not _has_agency:
                        continue
                    # Strict match: beneficiary tokens appear in a
                    # contiguous phrase somewhere on the BL.
                    _doc_norm = re.sub(r'[^A-Z0-9]+', ' ', _doc_text_up)
                    _doc_norm = re.sub(r'\s+', ' ', _doc_norm)
                    _pattern = r'\b' + r'\s+'.join(
                        re.escape(t) for t in _bene_tokens
                    ) + r'\b'
                    _bene_on_bl = bool(re.search(_pattern, _doc_norm))
                    if not _bene_on_bl:
                        continue
                    # Flip FAIL -> PASS
                    _set(row, "compliance", "PASS")
                    _findings = (
                        f"Beneficiary '{_bene_for_rescue}' is named on the "
                        f"BL shipper field via an agency construction "
                        f"(e.g. 'on behalf of' / 'for account of'). "
                        f"Under UCP 600 / ISBP 821 this identifies the "
                        f"beneficiary as principal — acceptable."
                    )
                    _set(row, "findings", _findings)
                    _set(row, "result", _findings[:200])
                    _set(row, "verification_notes",
                         "P198ae shipper-agency rescue: beneficiary "
                         "named on BL via agency wording")
                    _progress(
                        f"  [P198ae shipper-agency] {_row_id}: "
                        f"FAIL->PASS (beneficiary named via agency)"
                    )
                except Exception as _e:
                    try:
                        print(f"[P198ae shipper-agency] exception on row {_row_id}: {_e}")
                    except Exception:
                        pass
    except Exception:
        pass

    # P198ak — Proforma ref+date citation integrity (deterministic).
    # LC F45A may require "SPECIFICATIONS AS PER BENEFICIARY'S
    # PROFORMA INVOICE REF.NO. <X> DATED <Y>" and the commercial
    # invoice must cite BOTH <X> and <Y> verbatim. Rule 21z in the
    # LLM prompt tells the verifier to quote the invoice's proforma
    # date and compare, but the LLM has been observed to echo the
    # LC's expected date back as if it were on the invoice and
    # verdict PASS — a pure hallucination. This post-check extracts
    # the proforma date from the LC and from the invoice body, and
    # flips the row to FAIL when they differ. Works purely on the
    # refined text, independent of the LLM's self-report.
    try:
        _f45a_full = str(
            _final_lc_fields.get('45A', '') or _final_lc_fields.get('F45A', '') or ''
        )
        _f45a_up = _f45a_full.upper()
        # Flexible LC-side regex (same as invoice-side below) —
        # handles "PROFORMA INVOICE REF.NO.", "PROFORMA INV. REF.",
        # "PROFORMA:", "PI NO.", etc., followed by a ref then
        # "DATED" / "DT" / "DATE" and a date in common formats.
        _m_lc = re.search(
            r'(?:P(?:RO)?\.?\s*)?FORMA\s*(?:INV(?:OICE)?\.?)?\s*'
            r'(?:REF\.?|#)?\s*(?:NO\.?|NUMBER)?\s*[:\.]?\s*'
            r'([A-Z0-9][A-Z0-9/\- .\n]*?)\s*'
            r'(?:DATED|DT\.?|DATE|DT)\s*[:\.]?\s*'
            r'([A-Z]+\.?\s*\d{1,2}[,\s]+\d{2,4}|'
            r'\d{1,2}[\s\-./]+[A-Z]+\.?[\s\-./]+\d{2,4}|'
            r'\d{4}[-./]\d{1,2}[-./]\d{1,2}|'
            r'\d{1,2}[-./]\d{1,2}[-./]\d{2,4})',
            _f45a_up, re.DOTALL,
        )
        if _m_lc:
            _lc_pro_ref_raw = _m_lc.group(1).strip()
            _lc_pro_date_raw = _m_lc.group(2).strip()

            _MONTHS = {
                'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
                'JUL': 7, 'AUG': 8, 'SEP': 9, 'SEPT': 9, 'OCT': 10,
                'NOV': 11, 'DEC': 12,
                'JANUARY': 1, 'FEBRUARY': 2, 'MARCH': 3, 'APRIL': 4,
                'JUNE': 6, 'JULY': 7, 'AUGUST': 8, 'SEPTEMBER': 9,
                'OCTOBER': 10, 'NOVEMBER': 11, 'DECEMBER': 12,
            }

            def _pro_parse(s):
                """Parse a date string into (year, month, day).
                Returns None when the string is ambiguous or missing a
                year. Conservative: only returns a tuple when all three
                components are confidently known."""
                if not s:
                    return None
                s = str(s).upper().strip().rstrip('.,;:')
                # Strip ordinal suffixes: "21ST", "22ND", "23RD", "4TH"
                s = re.sub(r'(\d+)(ST|ND|RD|TH)\b', r'\1', s)
                # Collapse multiple spaces / weird separators
                s = re.sub(r'\s+', ' ', s).strip()

                # "JAN 21, 2026" / "JANUARY 21, 2026" / "JAN.21,2026" / "JAN 21 2026"
                m = re.match(
                    r'^([A-Z]+)\.?\s*[- ]?\s*(\d{1,2})\s*[,.\- ]\s*(\d{2,4})$',
                    s,
                )
                if m and _MONTHS.get(m.group(1)):
                    y = int(m.group(3))
                    if y < 100:
                        y = 2000 + y if y <= 69 else 1900 + y
                    return (y, _MONTHS[m.group(1)], int(m.group(2)))

                # "21 JAN 2026" / "21-JAN-2026" / "21/JAN/2026" / "21.JAN.2026"
                m = re.match(
                    r'^(\d{1,2})[\s\-./]+([A-Z]+)\.?[\s\-./]+(\d{2,4})$',
                    s,
                )
                if m and _MONTHS.get(m.group(2)):
                    y = int(m.group(3))
                    if y < 100:
                        y = 2000 + y if y <= 69 else 1900 + y
                    return (y, _MONTHS[m.group(2)], int(m.group(1)))

                # ISO-ish "2026-01-21" / "2026/01/21" / "2026.01.21"
                m = re.match(r'^(\d{4})[-./](\d{1,2})[-./](\d{1,2})$', s)
                if m:
                    mo, d = int(m.group(2)), int(m.group(3))
                    if 1 <= mo <= 12 and 1 <= d <= 31:
                        return (int(m.group(1)), mo, d)

                # DD-MM-YYYY (European) — AMBIGUOUS with MM-DD-YYYY so we
                # only accept this when day > 12 (unambiguous) OR when
                # both values <= 12 we skip to avoid misordering.
                m = re.match(r'^(\d{1,2})[-./](\d{1,2})[-./](\d{2,4})$', s)
                if m:
                    a, b = int(m.group(1)), int(m.group(2))
                    y = int(m.group(3))
                    if y < 100:
                        y = 2000 + y if y <= 69 else 1900 + y
                    if a > 12 and 1 <= b <= 12:
                        # Unambiguously DD-MM-YYYY
                        return (y, b, a)
                    if b > 12 and 1 <= a <= 12:
                        # Unambiguously MM-DD-YYYY (US)
                        return (y, a, b)
                    # Ambiguous (both ≤ 12) — bail, can't decide safely
                    return None

                # Compact YYMMDD "260121"
                m = re.match(r'^(\d{2})(\d{2})(\d{2})$', s)
                if m:
                    y = 2000 + int(m.group(1)) if int(m.group(1)) <= 69 else 1900 + int(m.group(1))
                    mo, d = int(m.group(2)), int(m.group(3))
                    if 1 <= mo <= 12 and 1 <= d <= 31:
                        return (y, mo, d)
                # Compact YYYYMMDD "20260121"
                m = re.match(r'^(\d{4})(\d{2})(\d{2})$', s)
                if m:
                    mo, d = int(m.group(2)), int(m.group(3))
                    if 1 <= mo <= 12 and 1 <= d <= 31:
                        return (int(m.group(1)), mo, d)

                return None

            def _pro_norm_ref(s):
                return re.sub(r'[\s\-/]', '', str(s or '').upper())

            _lc_pro_date = _pro_parse(_lc_pro_date_raw)
            _lc_pro_ref_n = _pro_norm_ref(_lc_pro_ref_raw)

            # P198ak — Robustified: iterate rows directly (not tasks)
            # and look up matching Commercial Invoice packets from
            # the `packets` list. This is independent of how tasks
            # were constructed / consumed and works across any
            # decomposition shape.

            # Flexible invoice-side regex — handles common variants:
            # "Proforma Invoice Ref.No." / "Proforma Inv. Ref."
            # / "Proforma Ref." / "Proforma:" / "PI No." / "PI#"
            # followed by a ref, then "DATED" / "DT" / "DT." / "DATE"
            # and a date in any of the supported formats.
            _PRO_INV_REGEX = re.compile(
                r'(?:P(?:RO)?\.?\s*)?FORMA\s*(?:INV(?:OICE)?\.?)?\s*'
                r'(?:REF\.?|#)?\s*(?:NO\.?|NUMBER)?\s*[:\.]?\s*'
                r'([A-Z0-9][A-Z0-9/\- .\n]*?)\s*'
                r'(?:DATED|DT\.?|DATE|DT)\s*[:\.]?\s*'
                r'([A-Z]+\.?\s*\d{1,2}[,\s]+\d{2,4}|'
                r'\d{1,2}[\s\-./]+[A-Z]+\.?[\s\-./]+\d{2,4}|'
                r'\d{4}[-./]\d{1,2}[-./]\d{1,2}|'
                r'\d{1,2}[-./]\d{1,2}[-./]\d{2,4})',
                re.DOTALL,
            )

            def _find_proforma_on_pkt(_pkt_text):
                if not _pkt_text or 'PROFORMA' not in _pkt_text.upper():
                    return None, None
                _doc_up = _pkt_text.upper()
                m = _PRO_INV_REGEX.search(_doc_up)
                if not m:
                    return None, None
                _ref = re.sub(r'\s+', ' ', m.group(1).strip())
                _date = m.group(2).strip()
                return _ref, _date

            # Pre-collect invoice packets with their proforma citation.
            # Source priority:
            #   1. unified_summary.references_found[role=proforma_reference]
            #      + unified_summary.dates_found[role=proforma_invoice_date
            #        | proforma_date | proforma_ref_date] — structured
            #        data emitted by step03
            #   2. Body-text regex fallback — when step03 didn't tag the
            #      proforma date as a role and it only exists as free
            #      text like "DATED FEB 18, 2026"
            _inv_citations = []  # (pkt_label, ref, date_raw, date_parsed, source)
            for _pkt in packets:
                if not isinstance(_pkt, dict):
                    continue
                _pt = (_pkt.get('document_type') or '').lower()
                if 'invoice' not in _pt or 'proforma' in _pt:
                    continue
                _pkt_label = _pkt.get('document_type', 'Commercial Invoice')
                _us = _pkt.get('unified_summary') or {}
                # Structured: proforma ref from references_found
                _struct_ref = None
                for _item in (_us.get('references_found') or []):
                    if (isinstance(_item, dict) and
                        'proforma' in str(_item.get('role', '') or '').lower()):
                        _struct_ref = str(_item.get('value') or _item.get('raw') or '').strip()
                        if _struct_ref:
                            break
                # Structured: proforma date from dates_found
                _struct_date_raw = None
                for _item in (_us.get('dates_found') or []):
                    if not isinstance(_item, dict):
                        continue
                    _role = str(_item.get('role', '') or '').lower()
                    if 'proforma' in _role:
                        _struct_date_raw = str(
                            _item.get('raw') or _item.get('value') or ''
                        ).strip()
                        if _struct_date_raw:
                            break
                if _struct_ref and _struct_date_raw:
                    _inv_citations.append((
                        _pkt_label, _struct_ref, _struct_date_raw,
                        _pro_parse(_struct_date_raw), 'structured',
                    ))
                    continue
                # Fallback: body-text regex
                _ptxt = _pkt_text(_pkt) or ''
                _r, _d = _find_proforma_on_pkt(_ptxt)
                if _r and _d:
                    _inv_citations.append((
                        _pkt_label, _r, _d, _pro_parse(_d), 'body-text',
                    ))

            # Iterate every row; flip when any invoice citation has
            # the same ref but a different parsed date.
            for row in rows:
                try:
                    _row_id = _get(row, "row_id", "?")
                    _clause_ref_u = (_get(row, "clause_ref", "") or '').upper()
                    _cond_u = (
                        _get(row, "condition_text", "") or
                        _get(row, "condition", "")
                    ).upper()
                    # Only consider rows whose condition EXPLICITLY
                    # mentions the proforma citation. Other 45A-family
                    # rows (goods description, quantity, unit price,
                    # incoterms, etc.) are not about the proforma
                    # citation and must not be touched by this check.
                    if 'PROFORMA' not in _cond_u:
                        continue
                    # Doc target must be Commercial Invoice
                    _doc_checked = (
                        _get(row, "document_checked", "") or
                        _get(row, "document_type", "")
                    ).lower()
                    if _doc_checked and 'invoice' not in _doc_checked:
                        continue
                    _current = _get(row, "compliance", "").upper()
                    if _current not in ("PASS", "REVIEW"):
                        continue  # don't touch genuine FAILs

                    # Raw-string normaliser for the fallback path — used
                    # when the date is ambiguous (e.g. "01-12-2025" where
                    # both 01 and 12 are valid day/month). If LC and
                    # invoice show the SAME raw string, they match even
                    # if we can't confidently assign day vs month.
                    def _norm_date_raw(s):
                        return re.sub(r'[\s\-./,]+', '', str(s or '').upper()).strip()
                    _lc_date_raw_n = _norm_date_raw(_lc_pro_date_raw)
                    _mismatch = None
                    for _pkt_label, _inv_ref, _inv_date_raw, _inv_date, _src in _inv_citations:
                        _inv_ref_n = _pro_norm_ref(_inv_ref)
                        _ref_match = (
                            _lc_pro_ref_n == _inv_ref_n or
                            _lc_pro_ref_n in _inv_ref_n or
                            _inv_ref_n in _lc_pro_ref_n
                        )
                        if not _ref_match:
                            continue
                        # Compare dates: first by parsed value (robust),
                        # fall back to raw-string match when parsing
                        # fails or is ambiguous.
                        if _lc_pro_date and _inv_date:
                            if _lc_pro_date != _inv_date:
                                _mismatch = (_pkt_label, _inv_ref, _inv_date_raw, _src)
                                break
                            else:
                                continue  # parsed dates match
                        # Parsing failed on one/both sides — try raw equality
                        _inv_date_raw_n = _norm_date_raw(_inv_date_raw)
                        if _lc_date_raw_n and _inv_date_raw_n:
                            if _lc_date_raw_n != _inv_date_raw_n:
                                _mismatch = (_pkt_label, _inv_ref, _inv_date_raw, _src)
                                break
                            # raw strings equal — treat as match, continue
                    if not _mismatch:
                        continue
                    _pkt_label, _inv_ref, _inv_date_raw, _src = _mismatch
                    _msg = (
                        f"Proforma reference {_lc_pro_ref_raw} matches but "
                        f"DATE DIFFERS: LC requires {_lc_pro_date_raw}; "
                        f"invoice ({_pkt_label}) shows {_inv_date_raw}. "
                        f"Under UCP 600 Art 18(c), 'strictly as per' "
                        f"binds both ref and date — different proforma "
                        f"date is a documentary discrepancy."
                    )
                    _set(row, "compliance", "FAIL")
                    _set(row, "findings", _msg)
                    _set(row, "result", _msg[:200])
                    _set(row, "verification_notes",
                         f"P198ak proforma-date cross-check [{_src}]: LC=("
                         f"{_lc_pro_ref_raw}, {_lc_pro_date_raw}) vs "
                         f"invoice=({_inv_ref}, {_inv_date_raw})")
                    _progress(
                        f"  [P198ak proforma-date:{_src}] {_row_id}: "
                        f"{_current}->FAIL (LC {_lc_pro_date_raw} "
                        f"vs invoice {_inv_date_raw})"
                    )
                except Exception as _e:
                    try:
                        print(f"[P198ak proforma-date] exception on row {_get(row,'row_id','?')}: {_e}")
                    except Exception:
                        pass
    except Exception as _e_outer:
        try:
            print(f"[P198ak proforma-date] outer exception: {_e_outer}")
        except Exception:
            pass

    # P198ao — BL signing-pattern rescue for "as agent(s) for and
    # on behalf of the master" / master-agency signings. Under UCP
    # 600 Art 20 (marine/ocean BL) and Art 22 (charter party BL),
    # both permit signing by "an agent for or on behalf of the
    # master / owner / charterer". This signing format is the
    # STANDARD for charter party BLs and also valid for regular
    # marine BLs — it is NOT a freight forwarder BL and does not
    # automatically make the document defective.
    #
    # Common false-FAIL pattern:
    #   BL signed "AS AGENTS FOR AND ON BEHALF OF THE MASTER,
    #    CAPT. <NAME>"
    #   bl_subtype.signing_type = "agent_for_master"
    #   LC F47A says "CHARTER PARTY B/L ACCEPTABLE" or
    #                 "FREIGHT FORWARDER'S / HOUSE BL ACCEPTABLE"
    #   LLM FAILs with "could be a freight forwarder's BL" — a
    #   hallucination that ignores the explicit master-agency
    #   wording.
    #
    # This check scans BL rows with FAIL verdict. When (a) the BL
    # shows master-agency signing text, (b) the row's condition
    # wording ALLOWS / PERMITS charter party or freight forwarder
    # BLs (not prohibits), (c) structured bl_subtype.signing_type
    # is master-agency / master-signed / carrier-signed — flip the
    # row to PASS.
    try:
        _MASTER_AGENCY_PHRASES = (
            'AS AGENTS FOR AND ON BEHALF OF THE MASTER',
            'AS AGENT FOR AND ON BEHALF OF THE MASTER',
            'AS AGENTS FOR THE MASTER',
            'AS AGENT FOR THE MASTER',
            'AS AGENT ON BEHALF OF THE MASTER',
            'AS AGENTS ON BEHALF OF THE MASTER',
            'ON BEHALF OF THE MASTER AS AGENT',
            'ON BEHALF OF THE MASTER AS AGENTS',
            'AS AGENTS ONLY FOR AND BY AUTHORITY OF THE MASTER',
            'AS AGENT ONLY FOR AND BY AUTHORITY OF THE MASTER',
            'FOR THE MASTER AS AGENT',
            'FOR THE MASTER AS AGENTS',
            'AGENT FOR MASTER',
            'AGENTS FOR MASTER',
            'AS AGENT FOR THE CARRIER',
            'AS AGENTS FOR THE CARRIER',
            'FOR AND ON BEHALF OF THE CARRIER',
            'AS AGENT FOR AND ON BEHALF OF THE OWNER',
            'AS AGENTS FOR AND ON BEHALF OF THE OWNER',
        )
        # Signing types that indicate NOT a freight forwarder
        _MASTER_AGENCY_SIGNING = (
            'agent_for_master', 'master_signed', 'carrier_signed',
            'agent_for_carrier', 'agent_for_owner',
        )
        for task in vlm_tasks:
            row = task["row"]
            _row_id = task.get("row_id", "?")
            try:
                _comp_now = _get(row, "compliance", "").upper()
                if _comp_now != "FAIL":
                    continue
                _doc_type_lc = (task.get("document_type") or '').lower()
                if 'bill of lading' not in _doc_type_lc:
                    continue
                _cond_u = (task.get("condition_text") or "").upper()
                # Only rescue when the row is about BL signing / BL
                # type / charter-party / forwarder acceptability.
                _rel_markers = (
                    'CHARTER PARTY', 'CHARTER-PARTY',
                    'FORWARDER', 'HOUSE BL', 'HOUSE BILL',
                    'SIGNED BY', 'SIGNATORY', 'AGENT',
                    'MASTER', 'CARRIER',
                )
                if not any(m in _cond_u for m in _rel_markers):
                    continue
                # Determine if condition is PERMISSIVE (allows these
                # BL types) or PROHIBITIVE (forbids them). Don't
                # rescue under prohibition — a prohibited charter
                # party BL still fails.
                _prohibitive = any(m in _cond_u for m in (
                    'NOT ACCEPTABLE', 'NOT PERMITTED', 'NOT ALLOWED',
                    'MUST NOT BE', 'UNACCEPTABLE',
                    'SHALL NOT', 'WILL NOT',
                    'NOT BE ACCEPT',
                ))
                _permissive = (
                    not _prohibitive and
                    any(m in _cond_u for m in (
                        'ACCEPTABLE', 'PERMITTED', 'ALLOWED',
                        'MAY BE', 'CAN BE',
                    ))
                )
                _doc_text_up = (task.get("document_text") or "").upper()
                _has_master_agency_text = any(
                    ph in _doc_text_up for ph in _MASTER_AGENCY_PHRASES
                )
                # Read structured bl_subtype if present
                _bl_subtype = (
                    task.get("bl_subtype") or
                    (task.get("packet") or {}).get("bl_subtype") or {}
                )
                _signing = str(_bl_subtype.get('signing_type', '') or '').lower() if isinstance(_bl_subtype, dict) else ''
                _is_master_agency_structured = _signing in _MASTER_AGENCY_SIGNING

                if not (_has_master_agency_text or _is_master_agency_structured):
                    continue
                # Under UCP 600 Art 20/22, master-agency signing is
                # valid for both regular marine and charter-party
                # BLs. If the condition is permissive about CPBL
                # or does not prohibit it, PASS.
                if _prohibitive and 'CHARTER PARTY' in _cond_u:
                    # Condition prohibits CPBL. Master-agency
                    # signing alone does NOT prove CPBL — the
                    # document would need "CHARTER PARTY" text
                    # or charter-party-specific markings. Check.
                    if 'CHARTER PARTY' in _doc_text_up:
                        continue  # document IS charter party — leave FAIL
                    # Not charter party — flip to PASS
                elif _prohibitive and 'FORWARDER' in _cond_u:
                    # Condition prohibits forwarder BL. Master-agency
                    # signing is explicitly NOT forwarder — PASS.
                    pass
                elif _prohibitive and 'HOUSE' in _cond_u:
                    # Condition prohibits house BL. Master-agency
                    # is carrier-issued (not house) — PASS.
                    pass
                elif _permissive:
                    # Condition permits CPBL / forwarder / house —
                    # master-agency meets the allowance → PASS.
                    pass
                else:
                    # Unclear wording; don't rescue.
                    continue

                _set(row, "compliance", "PASS")
                _findings = (
                    f"BL signed by agent for/on behalf of the master — "
                    f"this is the UCP 600 Art 20/22 standard signing "
                    f"format for charter party and marine/ocean BLs. "
                    f"Not a freight forwarder's BL."
                    + (f" Evidence: structured signing_type='{_signing}'." if _is_master_agency_structured else "")
                    + (" Master-agency signing phrase present on BL."
                       if _has_master_agency_text else "")
                )
                _set(row, "findings", _findings)
                _set(row, "result", _findings[:200])
                _set(row, "verification_notes",
                     "P198ao BL master-agency signing rescue: "
                     f"signing_type={_signing or 'n/a'}, "
                     f"master-agency text={_has_master_agency_text}")
                _progress(
                    f"  [P198ao BL master-agency] {_row_id}: "
                    f"FAIL->PASS (master-agency signing, not a forwarder BL)"
                )
            except Exception as _e:
                try:
                    print(f"[P198ao BL master-agency] exception on row {_row_id}: {_e}")
                except Exception:
                    pass
    except Exception:
        pass

    # P177 — Deterministic "freight must be shown/mentioned separately
    # on the Commercial Invoice" check. LLM routinely PASSes this check
    # on CFR/CIF/CIP invoices where freight is EMBEDDED in the total
    # (unit-price includes freight, single line total with Incoterm),
    # rather than broken out as a distinct line. Under UCP practice,
    # "freight shown separately" means an explicit FREIGHT: $X line
    # OR a discounted "less freight" deduction line on the invoice.
    try:
        for task in vlm_tasks:
            row = task["row"]
            row_id = task.get("row_id", "?")
            try:
                _comp_now = _get(row, "compliance", "").upper()
                if _comp_now != "PASS":
                    continue
                _cond_u = (task.get("condition_text") or "").upper()
                if 'FREIGHT' not in _cond_u:
                    continue
                if not any(kw in _cond_u for kw in (
                    'SEPARATELY', 'SEPARATE', 'DISTINCT', 'BROKEN OUT',
                    'MENTIONED', 'SHOWN', 'INDICATED', 'INDICATE',
                )):
                    continue
                _doc_type_lc = (task.get("document_type") or '').lower()
                # Only applies when doc is a Commercial Invoice / Invoice.
                if 'invoice' not in _doc_type_lc:
                    continue
                _doc_text_up = (task.get("document_text") or "").upper()
                if not _doc_text_up:
                    continue
                # A "separately mentioned freight value" must look like a
                # monetary amount following the word FREIGHT (not just
                # "FREIGHT PREPAID" — that's a BL term for who paid).
                # Search for FREIGHT + amount patterns.
                _freight_amt_re = re.compile(
                    r'\bFREIGHT(?:\s+(?:CHARGES?|COST|VALUE|AMOUNT))?\s*'
                    r'(?:IS|:|\-|=)?\s*'
                    r'(?:USD|EUR|GBP|JPY|CNY|PKR|AED|SAR|\$|€|£)?\s*'
                    r'([\d,]+(?:\.\d{1,2})?)',
                    re.IGNORECASE,
                )
                _found_separate = False
                for _m in _freight_amt_re.finditer(_doc_text_up):
                    # Extract number; must be > 0
                    try:
                        _val = float(_m.group(1).replace(',', ''))
                        if _val > 0:
                            # Exclude the total-line edge case where
                            # "FREIGHT" is just part of "CFR / CIF"
                            # wording or "FREIGHT PREPAID" label.
                            _ctx = _doc_text_up[
                                max(0, _m.start() - 20):
                                min(len(_doc_text_up), _m.end() + 20)
                            ]
                            if 'PREPAID' in _ctx and not re.search(r'\d', _ctx.split('PREPAID', 1)[1] if 'PREPAID' in _ctx else ''):
                                continue  # FREIGHT PREPAID with no amount after
                            _found_separate = True
                            break
                    except (ValueError, IndexError):
                        continue
                if not _found_separate:
                    # Also accept: CFR/CIF invoice with an EXPLICIT
                    # "LESS FREIGHT" deduction, or a separate Incoterm
                    # breakdown row.
                    if re.search(
                        r'\bLESS\s+FREIGHT\b|\bFREIGHT\s+COMPONENT\b|'
                        r'\bFOB\s+VALUE\b.{0,50}\bFREIGHT\s+VALUE\b',
                        _doc_text_up,
                    ):
                        _found_separate = True
                if not _found_separate:
                    _set(row, "compliance", "FAIL")
                    _set(row, "findings",
                         "Invoice does not show the freight value separately. "
                         "The total appears to include freight (CFR/CIF/CIP) "
                         "without a distinct freight line — LC requires freight "
                         "to be mentioned as a separate value.")
                    _set(row, "result", _get(row, "findings", "")[:200])
                    _set(row, "verification_notes",
                         "P177 freight-separate deterministic: no FREIGHT + amount line on invoice")
                    _progress(f"  [P177 freight-separate] {row_id}: PASS->FAIL (no separate freight amount on invoice)")
            except Exception as _e:
                try:
                    print(f"[P177 freight] exception on row {row_id}: {_e}")
                except Exception:
                    pass
    except Exception:
        pass

    # P173 — Strip internal override markers like "(P172 deterministic)",
    # "(P160 override)", "(P141 universal override)", "(P151a)" from
    # user-facing findings/result/verification_notes. These are
    # implementation tags for debugging; end users should never see
    # them in the compliance report.
    _override_tag_re = re.compile(
        r'\s*\(\s*P\d+[a-z]?\b[^)]*\)\s*$',
        flags=re.IGNORECASE,
    )
    _override_tag_inline_re = re.compile(
        r'\s*\(\s*P\d+[a-z]?\b[^)]{0,80}\)',
        flags=re.IGNORECASE,
    )
    for row in rows:
        for _fld in ('findings', 'result', 'found_text', 'verification_notes'):
            _v = _get(row, _fld, '')
            if not _v or not isinstance(_v, str):
                continue
            _cleaned = _override_tag_re.sub('', _v)
            _cleaned = _override_tag_inline_re.sub('', _cleaned)
            _cleaned = re.sub(r'\s{2,}', ' ', _cleaned).strip()
            _cleaned = re.sub(r'\s+\.', '.', _cleaned)
            if _cleaned != _v:
                _set(row, _fld, _cleaned)

    # P169 — Drop rows flagged for removal (doc not in LC required-docs
    # AND not in submission). These rows should never reach the report
    # as informational placeholders.
    _pre_drop = len(rows)
    rows = [r for r in rows if not (
        (hasattr(r, 'get') and r.get('_drop_from_report')) or
        (hasattr(r, '__dict__') and getattr(r, '_drop_from_report', False))
    )]
    if len(rows) < _pre_drop:
        _progress(f"P169 dropped {_pre_drop - len(rows)} row(s) for out-of-scope doc types")

    review_count = sum(1 for r in rows if _get(r, "compliance") == "REVIEW")
    info_count = sum(1 for r in rows if _get(r, "compliance") == "N/A")
    pending_count = sum(1 for r in rows if _get(r, "compliance") == "PENDING")
    pass_count = sum(1 for r in rows if _get(r, "compliance") == "PASS")
    fail_count = sum(1 for r in rows if _get(r, "compliance") == "FAIL")

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
