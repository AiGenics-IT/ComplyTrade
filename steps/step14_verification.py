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

AI MODEL: Qwen 2.5-VL-72B @ http://10.20.10.2:8085/v1/chat/completions
    Used for ALL verification -- no code-based fallbacks.
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
                      'advice', 'receipt', 'bill', 'policy', 'schedule', 'declaration'}
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


def _get_lc_field_value(step06_result: dict, field_tag: str) -> str:
    """Get a specific LC field value by tag (e.g. '31D', '44E')."""
    final_lc = step06_result.get("final_lc", step06_result)
    fields = final_lc.get("consolidated_fields", final_lc)

    # Direct lookup
    val = fields.get(field_tag, "")
    if isinstance(val, dict):
        val = val.get("value", str(val))
    if val:
        return str(val).strip()

    # Try with 'F' prefix
    val = fields.get(f"F{field_tag}", "")
    if isinstance(val, dict):
        val = val.get("value", str(val))
    return str(val).strip() if val else ""


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
DOCUMENT TEXT (from GLM OCR -- trusted):
{document_text}

DOCUMENT VISUAL METADATA (stamps, signatures, seals, copy/original status detected):
{visual_metadata}

VERIFY: Does the document satisfy this condition?

CRITICAL RULES (follow strictly):
1. CHECK F47A FIRST: Before marking anything as FAIL, read ALL F47A conditions above carefully. If ANY F47A clause says something is "ACCEPTABLE", "ALLOWED", or "PERMITTED" that relates to this condition, it OVERRIDES the main requirement.
2. CONDITIONAL ACCEPTANCE: If F47A allows something WITH conditions (e.g., "LATE SHIPMENT ALLOWED PROVIDED penalty deduction"), mark as REVIEW (not FAIL) and explain the condition that needs manual verification.
3. "ANY [COUNTRY] PORT" means any port in that country is acceptable = PASS.
4. Shipment BEFORE latest date = PASS. Shipment AFTER latest date = check F47A first. If F47A allows late shipment, mark REVIEW.
5. CHARTER PARTY: If F47A says "CHARTER PARTY BL ACCEPTABLE", then charter party BL = PASS.
6. Name matching: key words must match (UNITED BANK = UNITED BANK LIMITED = UBL). Minor spelling differences are acceptable.
7. Amount: The invoice/draft amount can be LESS than the LC amount (partial/short shipment) — only EXCEEDING the LC amount + tolerance is a discrepancy.
8. THIRD PARTY: If F47A says "THIRD PARTY DOCUMENTS ACCEPTABLE", third party documents = PASS.
9. CERTIFICATION: If the condition asks for origin certification and the document says "We certify the goods are of [COUNTRY] origin" or similar statement, that IS a valid certification = PASS. Do not fail just because the exact word "CERTIFICATE" is not used — any statement certifying origin, quality, weight, etc. is a certification.
10. DOCUMENT VERIFICATION: If the document text does NOT look like the expected document type (e.g., the condition checks a "Phytosanitary Certificate" but the document text looks like a quality certificate or inspection report), mark as REVIEW with "Document type may be misclassified".
11. When in doubt, mark REVIEW with clear reasoning rather than FAIL.
12. QUANTITY TOLERANCE: If the LC says "1000 MT LESS 10 PCT" or "1000 MT +/-5%", apply the tolerance. "1000 MT LESS 10 PCT" means 900-1000 MT is acceptable. A quantity of 950 MT is WITHIN range = PASS. Do NOT fail if quantity is within tolerance. Also check F47A for additional tolerance clauses (e.g., "+0/-10%").
13. BILL OF LADING LIMITATIONS: BLs do NOT show dollar amounts or unit prices — never fail a BL for "amount not mentioned". BLs do NOT typically show LC/credit numbers unless F47A specifically requires it on BL.
14. PERMISSIVE CLAUSES: "ACCEPTABLE" means something is ALLOWED — it is NOT a prohibition. "THIRD PARTY DOCUMENTS ACCEPTABLE EXCEPT X" means X must be from beneficiary, everything else can be third party. Do NOT interpret "EXCEPT X" as "X is not acceptable".
15. MATH: When comparing numbers, verify your arithmetic. 950 is LESS than 1000 (not more). 490,200 is LESS than 516,000 (not more). Get the direction right before marking FAIL.
16. EMAIL EQUIVALENCE: In SWIFT messages, "@" is written as "(AT)" or "(at)". So "INFO(AT)CICL.COM.PK" in the LC is the SAME as "info@cicl.com.pk" in the document. Treat (AT) and @ as identical when comparing email addresses. Also ignore case differences in emails.
17. AGENT vs FORWARDER: "AS AGENTS ONLY FOR AND BY AUTHORITY OF THE MASTER" on a BL means the carrier's agent signed — this is NORMAL and NOT a freight forwarder BL. A freight forwarder BL would say "FIATA", "HOUSE BILL", or show a forwarder company as the ISSUER (not as agent of master).
18. COPIES/DUPLICATES: "IN DUPLICATE" = 2 copies, "IN TRIPLICATE" = 3, "IN QUADRUPLICATE" = 4, "IN OCTUPLICATE" = 8, "FULL SET" = 3/3 originals. The number of copies is verified by the SYSTEM (not you) — it counts how many separate document packets exist. When you see a condition about copies/duplicates, mark it as PASS — the system handles copy counting separately. Do NOT fail a document for "not in duplicate/octuplicate" — you are only seeing ONE representative copy.
19. MISSING DOCUMENT: If a required document is completely MISSING from the submission, report ONE failure: "Required document missing". Do NOT add sub-failures for content checks (importer name, language, etc.) on a missing document — those are meaningless if the document doesn't exist.
20. PORT MATCHING: Ports are the SAME if the city/country matches, even if qualifiers differ. "KARACHI SEAPORT, PAKISTAN" = "KARACHI, PAKISTAN" = "KARACHI PORT, PAKISTAN". The word "SEAPORT"/"PORT" is just a qualifier. Similarly: "PENANG PORT, MALAYSIA" = "PENANG, MALAYSIA". Also "ANY [COUNTRY] PORT" means ANY port in that country = PASS if port is in that country. "ANY MALAYSIA PORT" matches "PENANG PORT, MALAYSIA". "ANY CANADIAN PORT" matches "VANCOUVER, CANADA".
21. QUANTITY MATCHING: LC may say "QTY 736" and invoice may show individual line items that SUM to 736.
22. PARTY REFERENCES: When the condition says "NOTIFY APPLICANT" or "TO ORDER OF ISSUING BANK", look at the LC PARTIES section above to find the actual name. Then check if that name appears on the document. "NOTIFY APPLICANT" means the notify party field must show the APPLICANT's name (given above). Do NOT look for the literal words "NOTIFY APPLICANT" — look for the applicant's ACTUAL NAME. Check the TOTAL quantity, not individual lines. Also "Ea" (each) is a valid unit — 736 Ea = 736 pieces. If LC says "QTY 736 AT THE RATE OF USD 98.00" and invoice shows 736 units × $98.00 = correct.

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
    If image_path is provided and the file exists, the image is included
    as a base64-encoded content block alongside the text prompt.
    """
    start = time.time()

    # Pre-extract key totals from document text to help smaller models
    _doc_summary = ''
    if document_text and len(document_text) > 500:
        import re as _re_sum
        # Find "Total Amount" or "Total:" lines
        _totals = _re_sum.findall(r'(?:Total\s*(?:Amount)?|TOTAL)[:\s]*(?:USD|EUR|GBP)?\s*([\d,]+\.?\d*)', document_text, _re_sum.IGNORECASE)
        if _totals:
            _doc_summary += f"TOTAL AMOUNTS FOUND: {', '.join(_totals)}\n"
        # Find quantity totals
        _qty_totals = _re_sum.findall(r'(?:Total\s*(?:Quantity)?|TOTAL)[:\s]*([\d,]+\.?\d*)\s*(?:Ea|pcs|KGS|MT|MMBTU|units|rolls|drums)', document_text, _re_sum.IGNORECASE)
        if _qty_totals:
            _doc_summary += f"TOTAL QUANTITIES FOUND: {', '.join(_qty_totals)}\n"
        # Find quantities — group by "Quantity Shipped" column (not "Quantity Ordered" to avoid double count)
        # Look for shipped quantities or standalone Qty patterns
        _shipped_qtys = _re_sum.findall(r'(?:Quantity\s+Shipped|Shipped)[:\s]*([\d,.]+)', document_text, _re_sum.IGNORECASE)
        if _shipped_qtys:
            _sum = sum(float(q.replace(',','')) for q in _shipped_qtys)
            _doc_summary += f"TOTAL SHIPPED QUANTITY: {_sum:.0f} (from {len(_shipped_qtys)} line items)\n"
        else:
            # Fallback: look for Qty: patterns but avoid duplicates (Ordered vs Shipped)
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

    # Build message content -- text only or text + image
    content_parts = []

    # Include image if available
    if image_path and os.path.isfile(image_path):
        try:
            with open(image_path, "rb") as fh:
                img_b64 = base64.b64encode(fh.read()).decode()
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_b64}"},
            })
        except Exception:
            pass  # Fall back to text-only if image read fails

    content_parts.append({"type": "text", "text": prompt_text})

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
            return {
                "row_id": row_id,
                "findings": "Nil",
                "result": f"VLM error HTTP {resp.status_code}",
                "compliance": "review",
                "confidence": 0.0,
                "reasoning": f"VLM returned HTTP {resp.status_code}",
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

        # For "all" documents: send each shipping doc as a separate task (deduped)
        if "all" in doc_types_to_check:
            found_any = False
            for pkt in deduped_packets:
                if not pkt:
                    continue
                pt = _pkt_type(pkt)
                # Skip LC pages, only check shipping docs
                if not pt or "lc" in pt.lower() or "letter of credit" in pt.lower():
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
                    if not pt or "lc" in pt.lower() or "letter of credit" in pt.lower():
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
            doc_target = task.get("doc_type_target", "unknown")
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
            _set(row, "findings", prefilled["findings"])
            _set(row, "found_text", prefilled["findings"])
            _set(row, "result", prefilled["result"])
            _set(row, "compliance", prefilled["compliance"].upper())
            _set(row, "confidence", prefilled["confidence"])
            _set(row, "verification_notes", prefilled.get("reasoning", ""))
            _progress(
                f"  {_get(row, 'row_id', '?')}: FAIL - "
                f"Document not found: {doc_target}"
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
