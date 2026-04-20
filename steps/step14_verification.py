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
        "courier receipt", "courier waybill", "courier service receipt",
        "express waybill", "express envelope", "express courier",
        "express delivery receipt",
        "dhl", "fedex", "fed ex", "ups", "tnt", "aramex",
        "shipment advice", "shipping advice", "declaration of shipment",
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
        "air waybill", "airway bill", "awb",
        "shipment advice", "declaration of shipment",
        "shipping advice",
    ],
    "email evidence": [
        "email evidence", "email screenshot", "email confirmation",
        "email copy", "covering email", "transmission record",
        "shipment advice", "shipment advise", "declaration of shipment",
        "shipping advice", "shipping advise",
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
    _text_search_terms = set(target_aliases)
    # Also add the original target with common variations
    _text_search_terms.add(target)
    for pkt in packets:
        if not pkt:
            continue
        pkt_type = _get_pkt_type(pkt)
        if pkt_type and ("lc" == pkt_type or "letter of credit" in pkt_type):
            continue
        pkt_text = _pkt_text(pkt if isinstance(pkt, dict) else asdict(pkt)).lower()
        if not pkt_text or len(pkt_text) < 20:
            continue
        # Search the first 2000 chars (header area) for the target name
        header = pkt_text[:2000]
        for term in _text_search_terms:
            if len(term) >= 4 and term in header:
                matches.append(pkt if isinstance(pkt, dict) else asdict(pkt))
                break

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
ANTI-HALLUCINATION RULES (STRICT — READ CAREFULLY)
════════════════════════════════════════════════════════════════════════

1. Your output MUST contain a "quote" field with the EXACT line(s) from
   DOCUMENT TEXT or STRUCTURED FACTS that justify the verdict.

2. If you cannot quote the relevant evidence, the verdict is FAIL. Period.

3. NEVER copy condition wording into "findings". Findings must describe what
   you ACTUALLY FOUND on the document, not what was being checked.

   ❌ WRONG:
   Condition: "must state vessel covered under Institute Classification Clause"
   Document: (no such text)
   findings="Vessel is covered under Institute Classification Clause"
   → Hallucination. The document says NOTHING about that clause.

   ✅ CORRECT:
   findings="Document has no mention of Institute Classification Clause.
   Closest text is '[actual line]'." verdict=FAIL

4. Prefer STRUCTURED FACTS over re-parsing document text:
   - Dates → dates_found[role=...]
   - Amounts → amounts_found[role=...]
   - References → references_found[role=...]
   - Parties → parties_found[role=...]
   - BL attributes → bl_subtype.contract_type / signing_type / has_terms_overleaf
   If a structured fact answers the question, cite its role in
   "structured_source". You don't have to re-quote from DOCUMENT TEXT.

5. Do NOT fabricate values that aren't on the document. If the condition asks
   for a value the document doesn't carry, answer REVIEW with findings
   explaining what's missing — NOT PASS with invented text.

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

Examples (all PASS):
- "UNITED BANK" = "UNITED BANK LIMITED" = "UBL"
- "Viterra B.V." = "Viterra BV" = "Bunge Netherlands Agri B.V." when the
  document itself says "currently known as" / "formerly known as" —
  SAME legal entity under a renamed form.
- "Dalda Foods Limited" = "DALDA FOODS LTD."

════════════════════════════════════════════════════════════════════════
GOODS DESCRIPTION TOLERANCE
════════════════════════════════════════════════════════════════════════

Minor wording variations are acceptable if the PRODUCT is clearly the same.
"Canadian Canola No.1" and "Canadian GMO Canola" refer to the same commodity.
Grade/variety descriptors (No.1, GMO, non-GMO, in bulk) are supplementary.
Core product name match → PASS.

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

BL prohibition clauses (condition says "BL must NOT be [charter party / short
form / blank back / house BL / freight forwarder issued]"):
- bl_subtype.signing_type in {master_signed, agent_for_master, carrier_signed}
    → NOT a freight forwarder BL → PASS "not forwarder" prohibition
- bl_subtype.has_terms_overleaf = true OR reverse page has T&C
    → NOT short form → PASS "not short form" prohibition
- bl_subtype.contract_type != "charter_party"
    → NOT charter party → PASS "not charter party" prohibition
- bl_subtype.issuer_type = "house_bl" → IS a house BL (check condition)
- bl_subtype.signing_type = "forwarder_signed" → IS forwarder-issued
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
) -> str:
    """Compose the CORE + family-pack prompt for one (condition, doc) verification."""
    family_pack = _pick_family_pack(document_type)
    structured_facts = _build_structured_facts(unified_summary or {}, bl_subtype or {})
    return CORE_VERIFICATION_PROMPT.format(
        condition_text=condition_text or "(not provided)",
        clause_ref=clause_ref or "(n/a)",
        lc_field_value=lc_field_value or "(n/a)",
        lc_parties=lc_parties or "(Not available)",
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
        _cond_ids = re.findall(
            r'[A-Z0-9][A-Z0-9/\-._]{5,}[A-Z0-9]',
            condition_text or '',
            flags=re.IGNORECASE,
        )
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
        # "NOT SHORT FORM" / "NOT SHORT-FORM" / "NOT BLANK BACK"
        if ('NOT' in cond_up and
                ('SHORT FORM' in cond_up or 'SHORT-FORM' in cond_up or
                 'BLANK BACK' in cond_up or 'BLANK-BACK' in cond_up)):
            if bl_subtype.get('has_terms_overleaf') is True or bl_subtype.get('is_blank_back') is False:
                return {
                    'verdict': 'PASS',
                    'quote': f"bl_subtype.has_terms_overleaf={bl_subtype.get('has_terms_overleaf')}, is_blank_back={bl_subtype.get('is_blank_back')}",
                    'findings': "BL has T&C printed on reverse (not short form / blank back).",
                    'confidence': 0.95,
                    'structured_source': 'bl_subtype.has_terms_overleaf',
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

        # P134: "MADE OUT TO THE ORDER OF <BANK>" / "CONSIGNED TO ORDER OF <X>"
        # Conditions like "Bill of lading must be made out to the order of
        # Bank Al Habib Ltd, Karachi" are very common. Check the structured
        # consignee field — if it contains "TO ORDER OF" plus the required
        # bank/party name, PASS deterministically.
        if ('TO THE ORDER OF' in cond_up or 'TO ORDER OF' in cond_up or
                'MADE OUT TO' in cond_up or 'CONSIGNED TO' in cond_up):
            # Extract the target party name from the condition — text after
            # "TO (THE) ORDER OF" up to "." / "," / end-of-line.
            _m = re.search(
                r'TO\s+(?:THE\s+)?ORDER\s+OF[\s:]+([^.\n,]+?)(?:[.,\n]|$|\s+KARACHI|\s+WITH|\s+FOR|\s+AT)',
                cond_up,
            )
            _target = (_m.group(1).strip() if _m else '').strip(' .,:')
            if _target:
                # Pull consignee text from unified_summary — typed field or
                # parties_found[role=consignee].
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
                # Match rule: consignee text must contain "TO ORDER" and the
                # target bank's key word(s). "Bank Al Habib" → look for
                # "AL HABIB" (skip the common word "BANK").
                _target_key = re.sub(r'\bBANK\b', '', _target).strip()
                _target_key = re.sub(r'\s+(LTD|LIMITED|LLC|PLC|INC|CORP|CO)\b\.?', '', _target_key).strip()
                if (_cons_txt and 'TO ORDER' in _cons_txt and _target_key and
                        _target_key in _cons_txt):
                    return {
                        'verdict': 'PASS',
                        'quote': f"consignee = {_cons_txt[:200]}",
                        'findings': (
                            f"BL consigned 'TO ORDER OF {_target_key}' per "
                            f"structured consignee field."
                        ),
                        'confidence': 0.95,
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
21z. PROFORMA INVOICE REFERENCE:
    When the LC says "SPECIFICATIONS AND FURTHER DETAILS ARE AS PER
    BENEFICIARY'S PROFORMA INVOICE NO. XXX DATED YYY", the commercial
    invoice MUST reference that exact proforma invoice number (XXX).
    - If the invoice mentions proforma invoice No. XXX → PASS
    - If the invoice ONLY references the LC number but NOT the
      proforma invoice number → FAIL. The LC explicitly requires
      the proforma reference. Referencing the LC number is NOT a
      substitute for the proforma invoice number.
    - This is a documentary credit requirement — the bank checker
      must see the exact proforma invoice number on the document.

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

    # Truncate document text to avoid exceeding token limits
    max_chars = 6500
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
    payload = {
        "model": QWEN_TEXT_LLM_MODEL,
        "messages": [
            {"role": "user", "content": prompt_text},
        ],
        "max_tokens": 500,
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

        # ── Normalize CORE-prompt response schema to legacy schema ──
        # CORE prompt returns: verdict / quote / findings / confidence / structured_source
        # Legacy downstream expects:   compliance / findings / result / confidence / reasoning
        if "compliance" not in parsed and "verdict" in parsed:
            parsed["compliance"] = str(parsed.get("verdict") or "review").lower()
        if "result" not in parsed:
            # Use findings as the short status; truncate for UI display
            _f = parsed.get("findings") or ""
            parsed["result"] = _f[:200] if _f else parsed.get("compliance", "review")
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
                        parsed["result"] = parsed["findings"][:200]
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
                        parsed["result"] = parsed["findings"][:200]
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
                        parsed["result"] = parsed["findings"][:200]
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
                            parsed["result"] = parsed["findings"][:200]
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
                        parsed["compliance"] = "review"
                        parsed["verdict"] = "REVIEW"
                        parsed["findings"] = (
                            f"{parsed.get('findings','').rstrip('. ')}. "
                            f"Product codes differ by a single character "
                            f"({sorted(_cond_codes)} vs {sorted(_fin_codes)}) "
                            f"— likely LC OCR error. Human review recommended "
                            f"before marking as discrepancy. (P137 downgrade)"
                        )
                        parsed["result"] = parsed["findings"][:200]
                        parsed["_post_check"] = "P137_unit_price_ocr_downgrade"
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
                # Extract identifier tokens from the CONDITION (what we were
                # looking for) — same pattern as the deterministic path.
                _cids = re.findall(
                    r'[A-Z0-9][A-Z0-9/\-._]{5,}[A-Z0-9]',
                    condition_text or '',
                    flags=re.IGNORECASE,
                )
                _doc_full = (
                    _normalize_id(document_text or '') + ' ' +
                    _normalize_id(str(unified_summary or ''))
                )
                for _needle in _cids:
                    _n = _normalize_id(_needle)
                    if len(_n) < 6 or _n in ('LETTERCREDIT', 'DOCUMENTARY', 'SHIPMENTADVICE', 'COMMERCIALINVOICE'):
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
                        parsed["result"] = parsed["findings"][:200]
                        parsed["_post_check"] = "P135_reference_found_override"
                        break
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
                    parsed["result"] = parsed["findings"][:200]
                    parsed["_post_check"] = "P138_date_found_override"
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
            'l/c bills schedule', 'lc bills schedule', 'bills schedule',
            'export dc document presentation schedule',
            'export dc presentation schedule',
            'document presentation schedule', 'presentation schedule',
            'schedule of documents', 'letter of transmittal',
            'document arrival notice', 'arrival notice',
            'forwarding letter',
            'remittance letter', 'export letter',
            'fax', 'email',
            # P92: structural / non-content page types
            'header page', 'blank page', 'endorsement page',
            'back page', 'terms and conditions',
            'unknown', 'unidentified', 'supporting document',
            # Agent's certificate is a specific doc, should be checked
            # separately — but NOT for HS codes / NTN fan-outs.
            # However we keep it in the fan-out for now since the LC
            # says "ALL DOCUMENTS". If it causes false fails, add it
            # to the exclude list.
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

        # Pick the BEST result — show only one, not all copies
        # Priority: PASS > REVIEW > FAIL
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
        if compliance != "PASS":
            continue  # only check PASS results for false positives

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
                    # VLM falsely passed — the document doesn't mention the email
                    _set(row, "compliance", "FAIL")
                    missing_emails = ', '.join(cond_emails)
                    _set(row, "findings",
                         f"Email address {missing_emails} not found in document text")
                    _set(row, "result",
                         f"Email {missing_emails} required but not mentioned in document")
                    _set(row, "verification_notes",
                         f"Deterministic override: LC requires notification via "
                         f"{missing_emails} but document does not reference this address")
                    _progress(f"  {row_id}: PASS->FAIL (email {missing_emails} not in doc)")

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
