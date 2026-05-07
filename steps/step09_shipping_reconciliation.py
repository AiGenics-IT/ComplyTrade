"""
Step 9 -- Shipping OCR Reconciliation
=======================================
For each classified shipping packet, sends image + GLM text + classification
to Qwen VLM. The VLM:
  1. Reviews GLM text against the page image
  2. ADDS any text GLM missed (stamps, handwritten notes, small print)
  3. Does NOT change or rewrite existing GLM text
  4. Extracts document-type-specific fields

KEY PRINCIPLE: GLM OCR already extracted trusted text (Step 1).
Qwen VLM only adds missing things -- never rewrites GLM text.

TEXT PROVENANCE (after this step, shipping docs have 3 layers):
    raw_text    -- GLM OCR output (Step 1)
    refined_text -- GLM text + VLM additions (this step)
    change_log  -- audit trail of what was added and why

INPUT:  Classified packets from Step 8 + required docs from Step 7
OUTPUT: Reconciled packets with refined text, extracted fields, change_log
"""

import json
import sys as _sys
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import os
import re
import time
import base64
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import QWEN_VLM_URL, QWEN_VLM_MODEL, MAX_CONCURRENT_VLM, VLM_TIMEOUT


# ── Dataclasses ──

@dataclass
class ChangeLogEntry:
    """Records a single change made during reconciliation."""
    field: str                         # "text_addition", "document_type", "field_extraction"
    old_value: str = ""
    new_value: str = ""
    reason: str = ""
    source: str = "vlm_reconciliation"
    timestamp: float = 0.0


@dataclass
class ReconciledPacket:
    """A shipping packet after VLM-based OCR reconciliation."""
    packet_id: str
    original_pages: List[int] = field(default_factory=list)
    page_image_paths: List[str] = field(default_factory=list)

    # Text provenance chain
    raw_text: str = ""                      # GLM OCR output (Step 1)
    cleaned_text: str = ""                  # After deterministic cleaning (Step 2)
    refined_text: str = ""                  # GLM text + VLM additions (this step)

    # Classification (carried from Step 8, may be updated)
    document_type: str = ""
    classification_status: str = "unknown"
    match_confidence: float = 0.0
    matched_requirement_index: int = -1
    matched_requirement_name: str = ""
    was_reclassified: bool = False
    previous_document_type: str = ""

    # Document summary and visual elements (carried from Step 8)
    document_summary: str = ""
    document_number: str = ""
    document_date: str = ""
    document_amount: str = ""
    stamps: List[dict] = field(default_factory=list)
    signatures: List[dict] = field(default_factory=list)
    seals: List[dict] = field(default_factory=list)
    logos: List[dict] = field(default_factory=list)
    copy_status: str = ""
    copy_label: str = ""
    marking_status: str = ""
    issued_by: str = ""
    lc_reference: str = ""

    # Document-type-specific extracted fields
    extracted_fields: Dict[str, Any] = field(default_factory=dict)

    # P129 — Carry forward Step 3 structured facts for deterministic Step 14
    unified_summary: Dict[str, Any] = field(default_factory=dict)
    bl_subtype: Dict[str, Any] = field(default_factory=dict)
    validation_status: str = ""

    # Change tracking
    change_log: List[dict] = field(default_factory=list)

    # Metadata
    source_step: int = 9
    confidence: float = 0.0
    ambiguity_flag: bool = False
    ambiguity_notes: str = ""
    elapsed_seconds: float = 0.0


# ── Document-Type-Specific Field Lists ──

_DOC_TYPE_FIELDS = {
    "Bill of Lading": (
        "shipper, consignee, notify_party, vessel, voyage, "
        "port_of_loading, port_of_discharge, bl_number, bl_date, "
        "shipped_on_board, shipped_on_board_date, freight_status, "
        "number_of_originals, goods_description, gross_weight, "
        "net_weight, number_of_packages, marks_and_numbers, "
        "place_of_receipt, place_of_delivery"
    ),
    "Commercial Invoice": (
        "invoice_number, invoice_date, amount, currency, "
        "goods_description, hs_code, ntn_number, "
        "incoterms (FULL term including suffixes like FO/FI e.g. 'CFR FO' not just 'CFR'), "
        "unit_price, quantity, total_amount, amount_in_words, "
        "buyer_name, seller_name, lc_reference"
    ),
    "Draft": (
        "drawee, drawer, amount, amount_in_words, currency, "
        "tenor, at_sight, lc_reference, draft_date, draft_number"
    ),
    "Insurance Policy/Certificate": (
        "policy_number, insured_party, sum_insured, currency, "
        "coverage_type, voyage_from, voyage_to, goods_description, "
        "claims_agent, premium, effective_date"
    ),
    "Certificate of Origin": (
        "certificate_number, country_of_origin, exporter_name, "
        "importer_name, goods_description, issuing_authority, "
        "certification_date, hs_code"
    ),
    "Packing List": (
        "total_packages, net_weight, gross_weight, dimensions, "
        "marks_and_numbers, goods_description, packing_details"
    ),
    "Weight List": (
        "total_net_weight, total_gross_weight, tare_weight, "
        "weight_per_package, number_of_packages"
    ),
    "Beneficiary Certificate": (
        "certificate_text, certifying_party, certification_date, "
        "lc_reference"
    ),
    "Inspection Certificate": (
        "certificate_number, inspector_name, inspection_date, "
        "goods_description, inspection_result, surveyor_company"
    ),
    "Shipping Advice": (
        "vessel_name, voyage_number, bl_number, bl_date, "
        "port_of_loading, port_of_discharge, goods_description, "
        "estimated_arrival"
    ),
    "Fumigation Certificate": (
        "certificate_number, fumigation_date, chemicals_used, "
        "treatment_duration, goods_description"
    ),
    "Phytosanitary Certificate": (
        "certificate_number, issuing_authority, inspection_date, "
        "place_of_origin, goods_description"
    ),
    "Health Certificate": (
        "certificate_number, issuing_authority, inspection_date, "
        "goods_description, fitness_declaration"
    ),
}

_DEFAULT_FIELDS = (
    "document_number, document_date, issuer, reference_numbers, "
    "key_text, any_amounts"
)


# ── Reconciliation Prompt ──

_RECONCILIATION_PROMPT = """This document was classified as: {document_type}

GLM OCR extracted this text:
{glm_text}

Review the image against the GLM text.
- Only ADD text that GLM missed (e.g., text in stamps, handwritten notes, small print)
- Do NOT change or rewrite existing GLM text
- Report what you added and why

Also extract these fields specific to {document_type}:
[{field_list}]

Return ONLY valid JSON:
{{
    "refined_text": "the GLM text with any additions appended at the end",
    "additions": [
        {{"text": "added text", "location": "where on the page", "reason": "why GLM missed it"}}
    ],
    "document_fields": {{
        "field_name": "field_value"
    }},
    "reclassify": false,
    "new_document_type": "",
    "reclassify_reason": "",
    "reclassify_confidence": 0.0
}}"""


# ─────────────────────────────────────────────────────────────────
# P198gz46 — Stamp-date sanity guard (year + day/month).
#
# When the VLM/OCR extracts a date from a rubber stamp / received-
# stamp on a document and the extracted date is BEFORE the document's
# own issue date, the date was almost certainly mis-read because a
# digit was cut off, blurred, or overlapped by other ink.
#
# Common patterns:
#   • Year cut off: "30 APR 2026" read as "30 APR 2020" (trailing
#     "6" looked like "0"). Anchor: cb7d7bbf pg32.
#   • Day digit cut off: "30 APR 2026" read as "10 APR 2026"
#     (top bar of "3" was cut, looks like "1"). Anchor: c1d9277c pg9
#     where doc issue is 27-APR-26 but extracted stamp says 10 APR.
#
# Rule: a received stamp on a document MUST be on or after the
# document's own issue date — that's a hard physical reality
# (the bank can't stamp a doc before the doc was issued).
#
# Correction strategy:
#   1. If extracted year is >1 year before doc issue year → replace
#      year with issue year (year-misread case).
#   2. If extracted year matches but date is still before issue
#      date → try common digit substitutions in the day position
#      (1→3, 1→7, 1→2, 0→8, 5→6, 5→8, 2→3) and pick the first
#      candidate that produces a date >= doc issue date AND within
#      60 days of it.
#   3. If no correction works, leave as-is and let downstream
#      verification flag it.
# ─────────────────────────────────────────────────────────────────

_STAMP_DATE_RE = re.compile(
    r'\b(\d{1,2})[\s\-/.]+'
    r'(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC|'
    r'JANUARY|FEBRUARY|MARCH|APRIL|JUNE|JULY|AUGUST|'
    r'SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)[\s\-/.]+'
    r'(\d{4})\b',
    flags=re.IGNORECASE,
)

_MONTH_TO_NUM = {
    'JAN': 1, 'JANUARY': 1, 'FEB': 2, 'FEBRUARY': 2,
    'MAR': 3, 'MARCH': 3, 'APR': 4, 'APRIL': 4,
    'MAY': 5, 'JUN': 6, 'JUNE': 6, 'JUL': 7, 'JULY': 7,
    'AUG': 8, 'AUGUST': 8, 'SEP': 9, 'SEPTEMBER': 9,
    'OCT': 10, 'OCTOBER': 10, 'NOV': 11, 'NOVEMBER': 11,
    'DEC': 12, 'DECEMBER': 12,
}

# OCR-misread digit pairs (visual confusion). Each tuple is
# (wrong_digit, candidate_correct_digit). Order matters — most
# common confusions first.
_DIGIT_CONFUSIONS = (
    ('1', '3'),  # 3 with cut top-bar reads as 1
    ('1', '7'),  # 7 with serif reads as 1
    ('1', '2'),  # 2 partial reads as 1
    ('0', '8'),  # 8 with broken sides reads as 0
    ('5', '6'),  # similar curves
    ('5', '8'),  # similar curves
    ('2', '3'),  # similar shape
    ('7', '1'),  # reverse case
    ('8', '0'),  # reverse case
)


def _try_day_correction(day_str, month_num, year, ref_date_tuple):
    """Try substituting digits in day_str. Return (new_day, new_day_str)
    if a valid candidate produces a date >= ref AND within 60 days, else
    None."""
    import datetime as _dt
    if len(day_str) > 2 or not day_str.isdigit():
        return None
    try:
        ref = _dt.date(*ref_date_tuple)
    except Exception:
        return None
    for pos in range(len(day_str)):
        ch = day_str[pos]
        for _wrong, _right in _DIGIT_CONFUSIONS:
            if ch != _wrong:
                continue
            new_day_str = day_str[:pos] + _right + day_str[pos + 1:]
            try:
                new_day = int(new_day_str)
                if not (1 <= new_day <= 31):
                    continue
                cand = _dt.date(year, month_num, new_day)
            except Exception:
                continue
            if cand >= ref and (cand - ref).days <= 60:
                return (new_day, new_day_str)
    return None


def _sanity_correct_stamp_years(text, doc_date, doc_issue_date, change_log):
    """Scan text for date-like tokens that are inconsistent with the
    document's own issue date and correct them.

    1. Year is far before doc year → bump year to doc year.
    2. Date is before doc issue date (same year) → try day-digit
       substitutions to find a plausible OCR correction.
    """
    if not text:
        return text
    try:
        import datetime as _dt
        # Reference year = the doc's own issue date year (or document_date)
        _ref_year = None
        _ref_date_tuple = None
        for _src in (doc_issue_date, doc_date):
            if _src:
                _m_full = re.search(r'\b(20\d{2})-(\d{1,2})-(\d{1,2})\b', str(_src))
                if _m_full:
                    _ref_year = int(_m_full.group(1))
                    _ref_date_tuple = (
                        int(_m_full.group(1)),
                        int(_m_full.group(2)),
                        int(_m_full.group(3)),
                    )
                    break
                _m_y = re.search(r'\b(20\d{2})\b', str(_src))
                if _m_y:
                    _ref_year = int(_m_y.group(1))
                    break
        if _ref_year is None:
            return text

        def _fix_date(m):
            day_str = m.group(1)
            mon_label = m.group(2)
            yr = int(m.group(3))
            mon_num = _MONTH_TO_NUM.get(mon_label.upper())
            # Case 1: year is implausibly old → bump to ref year
            if yr < _ref_year - 1:
                new_yr = _ref_year
                if change_log is not None:
                    try:
                        change_log.append(asdict(ChangeLogEntry(
                            field="stamp_year_corrected",
                            old_value=m.group(0),
                            new_value=f"{day_str} {mon_label} {new_yr}",
                            reason=(f"P198gz46: extracted year {yr} is "
                                    f"{_ref_year - yr} years before doc "
                                    f"issue year {_ref_year} — likely "
                                    f"OCR misread of cut-off digit; "
                                    f"corrected year to {new_yr}"),
                            timestamp=time.time(),
                        )))
                    except Exception:
                        pass
                return f"{day_str} {mon_label} {new_yr}"

            # Case 2: same/close year but DATE before doc issue date
            # → try day-digit substitution
            if (_ref_date_tuple is not None and yr == _ref_year
                    and mon_num is not None):
                try:
                    cand_date = _dt.date(yr, mon_num, int(day_str))
                    ref_date = _dt.date(*_ref_date_tuple)
                    if cand_date < ref_date:
                        fix = _try_day_correction(
                            day_str, mon_num, yr, _ref_date_tuple,
                        )
                        if fix is not None:
                            new_day, new_day_str = fix
                            if change_log is not None:
                                try:
                                    change_log.append(asdict(ChangeLogEntry(
                                        field="stamp_day_corrected",
                                        old_value=m.group(0),
                                        new_value=f"{new_day_str} {mon_label} {yr}",
                                        reason=(
                                            f"P198gz46: extracted date "
                                            f"{day_str}-{mon_label}-{yr} is "
                                            f"before doc issue date "
                                            f"{ref_date.isoformat()} — "
                                            f"likely OCR digit misread "
                                            f"({day_str} -> {new_day_str}); "
                                            f"corrected"
                                        ),
                                        timestamp=time.time(),
                                    )))
                                except Exception:
                                    pass
                            return f"{new_day_str} {mon_label} {yr}"
                except Exception:
                    pass
            return m.group(0)
        return _STAMP_DATE_RE.sub(_fix_date, text)
    except Exception:
        return text


def _reconcile_single_packet(packet: dict, expected_docs: List[dict], packet_index: int) -> dict:
    """
    Reconcile a single shipping packet: send image + GLM text + classification
    to Qwen VLM for text additions and field extraction.
    """
    start = time.time()

    doc_type = packet.get('document_type', '')
    glm_text = packet.get('cleaned_text', packet.get('raw_text', ''))
    image_paths = packet.get('page_image_paths', [])
    original_status = packet.get('classification_status', 'unknown')

    change_log = []

    # Get document-type-specific field list
    field_list = _DOC_TYPE_FIELDS.get(doc_type, _DEFAULT_FIELDS)

    # Build prompt
    prompt = _RECONCILIATION_PROMPT.format(
        document_type=doc_type if doc_type else "Unknown",
        glm_text=glm_text[:5000],
        field_list=field_list,
    )

    # Build VLM request with images
    content_parts = []
    if image_paths:
        for img_path in image_paths[:2]:  # Max 2 pages
            img_str = str(img_path)
            if os.path.exists(img_str):
                try:
                    with open(img_str, 'rb') as f:
                        img_b64 = base64.b64encode(f.read()).decode('utf-8')
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,%s" % img_b64}
                    })
                except Exception:
                    pass

    content_parts.append({"type": "text", "text": prompt})

    # Call VLM
    refined_text = glm_text  # Default: keep GLM text unchanged
    extracted_fields = {}
    was_reclassified = False
    previous_type = ""
    additions = []

    try:
        resp = requests.post(QWEN_VLM_URL, json={
            "model": QWEN_VLM_MODEL,
            "messages": [{"role": "user", "content": content_parts}],
            "max_tokens": 4000,
            "temperature": 0.1,
        }, timeout=None)

        if resp.status_code == 200:
            result = resp.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")

            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                vlm_data = json.loads(json_match.group(0))

                # Refined text (GLM text + additions)
                new_text = vlm_data.get('refined_text', '')
                if new_text and len(new_text) >= len(glm_text) * 0.9:
                    refined_text = new_text
                    change_log.append(asdict(ChangeLogEntry(
                        field="text_addition",
                        old_value="[%d chars]" % len(glm_text),
                        new_value="[%d chars]" % len(refined_text),
                        reason="VLM added missing text",
                        timestamp=time.time(),
                    )))

                # Log individual additions
                additions = vlm_data.get('additions', [])
                if isinstance(additions, list):
                    for add in additions:
                        if isinstance(add, dict) and add.get('text'):
                            change_log.append(asdict(ChangeLogEntry(
                                field="text_addition",
                                old_value="",
                                new_value=str(add.get('text', '')),
                                reason="VLM found: %s (%s)" % (
                                    add.get('location', 'unknown'),
                                    add.get('reason', 'missed by GLM')),
                                timestamp=time.time(),
                            )))

                # Reclassification check
                if vlm_data.get('reclassify', False):
                    new_type = vlm_data.get('new_document_type', '')
                    reclass_conf = float(vlm_data.get('reclassify_confidence', 0.0))
                    reclass_reason = vlm_data.get('reclassify_reason', '')

                    # ── P198fp — Reclassification veto for specific
                    # commodity / cert names that the VLM tends to
                    # over-collapse into generic buckets.
                    #
                    # Example: a packet originally classified as
                    #   "COAL SPECIFICATIONS AT THE LOADING PORT"
                    # is a SPECIFIC named report — coal-trade convention
                    # for the load-port quality / sampling cert. Step 9
                    # was reclassifying it to the generic "Inspection
                    # Certificate" bucket because the LC's expected-doc
                    # list happens to contain that name. That collapse
                    # loses the trade-specific context and breaks the
                    # coal-quality verifier (P198fb) which keys off the
                    # specific name.
                    #
                    # Veto rule: when the ORIGINAL doc_type contains a
                    # commodity / cert-family marker (COAL, IRON ORE,
                    # PETROLEUM, SUGAR, GRAIN, FERTILIZER, DRAFT SURVEY,
                    # SAMPLING, ANALYSIS, WEIGHT, QUALITY) AND the
                    # proposed new_type is a generic family bucket
                    # (Inspection Certificate, Quality Certificate,
                    # Survey Report, Test Certificate, Quantity
                    # Certificate, Generic Certificate), reject the
                    # reclassification. The original specific name is
                    # the trade-finance truth.
                    _orig_u = (doc_type or '').upper()
                    _new_u  = (new_type or '').upper()
                    _SPECIFIC_MARKERS = (
                        'COAL', 'IRON ORE', 'PETROLEUM', 'CRUDE',
                        'SUGAR', 'WHEAT', 'BARLEY', 'CORN', 'RICE',
                        'GRAIN', 'OILSEED', 'PALM', 'FERTILIZER',
                        'UREA', 'CEMENT', 'CLINKER', 'STEEL',
                        'PETCOKE', 'LIGNITE', 'BITUMINOUS',
                        'DRAFT SURVEY', 'SAMPLING', 'ANALYSIS',
                        'WEIGHT', 'CALORIFIC', 'PROXIMATE',
                        'ULTIMATE', 'SPECIFICATION',
                    )
                    _GENERIC_BUCKETS = (
                        'INSPECTION CERTIFICATE',
                        'QUALITY CERTIFICATE',
                        'TEST CERTIFICATE',
                        'GENERIC CERTIFICATE',
                        'CERTIFICATE',  # very generic — only triggers if BOTH conditions met
                        'SURVEY REPORT',
                        'INSPECTION REPORT',
                        'QUANTITY CERTIFICATE',
                    )
                    _is_specific_orig = any(m in _orig_u for m in _SPECIFIC_MARKERS)
                    _is_generic_new  = any(g == _new_u or _new_u.endswith(' ' + g)
                                           or _new_u.startswith(g + ' ') or _new_u == g
                                           for g in _GENERIC_BUCKETS)
                    # Only veto when the NEW type is PURELY generic — if
                    # the new type itself carries a commodity / cert
                    # marker (e.g. "Coal Quality Certificate"), trust
                    # it; the reclass is preserving specificity.
                    _new_also_specific = any(m in _new_u for m in _SPECIFIC_MARKERS)
                    if _is_specific_orig and _is_generic_new and not _new_also_specific:
                        change_log.append(asdict(ChangeLogEntry(
                            field="document_type_reclassify_vetoed",
                            old_value=doc_type,
                            new_value=new_type,
                            reason=("P198fp veto: original name carries a "
                                    "specific commodity/cert marker; refusing "
                                    "to collapse to generic bucket '%s'. "
                                    "Reason given: %s" % (new_type, reclass_reason)),
                            timestamp=time.time(),
                        )))
                        # Skip the reclass — keep the original doc_type.
                        new_type = ''   # disable the apply block below

                    if new_type and reclass_conf > 0.7 and new_type != doc_type:
                        previous_type = doc_type
                        doc_type = new_type
                        was_reclassified = True

                        change_log.append(asdict(ChangeLogEntry(
                            field="document_type",
                            old_value=previous_type,
                            new_value=doc_type,
                            reason="VLM reclassification (%.2f): %s" % (
                                reclass_conf, reclass_reason),
                            timestamp=time.time(),
                        )))

                        # Try to match new type to expected docs
                        matched_idx = -1
                        for i, ed in enumerate(expected_docs):
                            ed_name = ed.get('document_name', '').upper()
                            dt_upper = doc_type.upper()
                            if dt_upper in ed_name or ed_name in dt_upper:
                                matched_idx = i
                                break

                        if matched_idx >= 0:
                            packet['matched_requirement_index'] = matched_idx
                            packet['matched_requirement_name'] = expected_docs[matched_idx].get('document_name', '')
                            packet['classification_status'] = 'matched_document'
                        else:
                            packet['classification_status'] = 'alien_document'

                # Extracted fields
                extracted_fields = vlm_data.get('document_fields', {})
                if not isinstance(extracted_fields, dict):
                    extracted_fields = {}

    except Exception as e:
        print("[Step 9] VLM reconciliation failed for packet %d: %s" % (packet_index, e))
        change_log.append(asdict(ChangeLogEntry(
            field="reconciliation",
            old_value="",
            new_value="",
            reason="VLM call failed: %s" % e,
            timestamp=time.time(),
        )))

    elapsed = time.time() - start

    # Confidence adjustment
    confidence = float(packet.get('match_confidence', 0.0))
    if refined_text != glm_text:
        confidence = min(confidence + 0.05, 1.0)
    if was_reclassified:
        confidence = max(confidence - 0.1, 0.0)

    # Ambiguity
    ambiguity = packet.get('ambiguity_flag', False)
    ambiguity_notes = packet.get('ambiguity_notes', '')
    if was_reclassified:
        ambiguity = True
        if ambiguity_notes:
            ambiguity_notes += "; "
        ambiguity_notes += "Reclassified from %s to %s" % (previous_type, doc_type)

    # Carry forward Step 8 visual elements
    reconciled = ReconciledPacket(
        packet_id=packet.get('packet_id', "packet_%03d" % packet_index),
        original_pages=packet.get('original_pages', []),
        page_image_paths=packet.get('page_image_paths', []),
        raw_text=packet.get('raw_text', ''),
        cleaned_text=packet.get('cleaned_text', glm_text),
        refined_text=_sanity_correct_stamp_years(
            refined_text,
            packet.get('document_date', ''),
            (packet.get('unified_summary') or {}).get('issue_date', ''),
            change_log,
        ),
        document_type=doc_type,
        classification_status=packet.get('classification_status', original_status),
        match_confidence=confidence,
        matched_requirement_index=packet.get('matched_requirement_index', -1),
        matched_requirement_name=packet.get('matched_requirement_name', ''),
        was_reclassified=was_reclassified,
        previous_document_type=previous_type,
        # Carry forward Step 8 visual elements
        document_summary=packet.get('document_summary', ''),
        document_number=packet.get('document_number', ''),
        document_date=packet.get('document_date', ''),
        document_amount=packet.get('document_amount', ''),
        stamps=[
            (lambda _s: ({**_s, 'text': _sanity_correct_stamp_years(
                _s.get('text', ''),
                packet.get('document_date', ''),
                (packet.get('unified_summary') or {}).get('issue_date', ''),
                change_log,
            )} if isinstance(_s, dict) else _s))(_s)
            for _s in (packet.get('stamps', []) or [])
        ],
        signatures=packet.get('signatures', []),
        seals=packet.get('seals', []),
        logos=packet.get('logos', []),
        copy_status=packet.get('copy_status', ''),
        copy_label=packet.get('copy_label', ''),
        marking_status=packet.get('marking_status', ''),
        issued_by=packet.get('issued_by', ''),
        lc_reference=packet.get('lc_reference', ''),
        # New from this step
        extracted_fields=extracted_fields,
        change_log=change_log,
        confidence=confidence,
        ambiguity_flag=ambiguity,
        ambiguity_notes=ambiguity_notes,
        elapsed_seconds=round(elapsed, 2),
        # P129 — carry forward Step 3 structured facts
        unified_summary=packet.get('unified_summary', {}) or {},
        bl_subtype=packet.get('bl_subtype', {}) or {},
        validation_status=packet.get('validation_status', '') or '',
    )

    return asdict(reconciled)


# ── Main Run Function ──

def run(step8_result: dict, step7_result: dict = None, output_dir: str = None, progress_callback=None) -> dict:
    """
    Execute Step 9: Reconcile OCR text and extract document-specific fields.

    Args:
        step8_result: Output from Step 8 with 'classified_packets'
        step7_result: Output from Step 7 with 'required_documents' (optional)
        output_dir: Directory to save results
        progress_callback: Optional callback for progress updates

    Returns:
        dict with 'reconciled_packets', 'summary', 'elapsed_seconds'
    """
    def _progress(msg):
        if progress_callback:
            progress_callback("[Step 9] %s" % msg)
        print("[Step 9] %s" % msg)

    start_time = time.time()

    packets = step8_result.get('classified_packets', [])
    required_docs = (step7_result or step8_result).get('required_documents', [])

    _progress("Reconciling %d classified packets..." % len(packets))

    if not packets:
        return {
            'reconciled_packets': [],
            'summary': {'total': 0, 'refined': 0, 'reclassified': 0},
            'elapsed_seconds': 0,
        }

    # Process packets concurrently
    reconciled_packets = [None] * len(packets)

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_VLM) as executor:
        futures = {}
        for idx, packet in enumerate(packets):
            future = executor.submit(_reconcile_single_packet, packet, required_docs, idx)
            futures[future] = idx

        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                reconciled_packets[idx] = result
                doc_type = result.get('document_type', '?')
                reclass = " [RECLASSIFIED]" if result.get('was_reclassified') else ""
                changes = len(result.get('change_log', []))
                n_fields = len(result.get('extracted_fields', {}))
                _progress("  Packet %d: %s%s (%d changes, %d fields)" % (
                    idx, doc_type, reclass, changes, n_fields))
            except Exception as e:
                _progress("  Packet %d: ERROR - %s" % (idx, e))
                reconciled_packets[idx] = asdict(ReconciledPacket(
                    packet_id="packet_%03d" % idx,
                    ambiguity_flag=True,
                    ambiguity_notes="Reconciliation error: %s" % e,
                ))

    # Summary
    total_refined = sum(
        1 for p in reconciled_packets
        if p and p.get('refined_text', '') != p.get('cleaned_text', '')
    )
    total_reclassified = sum(
        1 for p in reconciled_packets
        if p and p.get('was_reclassified', False)
    )

    summary = {
        'total': len(reconciled_packets),
        'refined': total_refined,
        'reclassified': total_reclassified,
        'matched': sum(1 for p in reconciled_packets if p and p.get('classification_status') == 'matched_document'),
        'alien': sum(1 for p in reconciled_packets if p and p.get('classification_status') == 'alien_document'),
        'extra': sum(1 for p in reconciled_packets if p and p.get('classification_status') == 'extra_document'),
        'unknown': sum(1 for p in reconciled_packets if p and p.get('classification_status') == 'unknown'),
        'ambiguous': sum(1 for p in reconciled_packets if p and p.get('ambiguity_flag')),
        'fields_extracted': sum(
            len(p.get('extracted_fields', {}))
            for p in reconciled_packets if p
        ),
    }

    elapsed = time.time() - start_time

    # Save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        result_file = os.path.join(output_dir, 'step09_result.json')
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({
                'step': 9,
                'step_name': 'Shipping OCR Reconciliation',
                'total_packets': len(reconciled_packets),
                'summary': summary,
                'elapsed_seconds': round(elapsed, 2),
                'reconciled_packets': reconciled_packets,
            }, f, indent=2, ensure_ascii=False)

    _progress("Step 9 complete: %d refined, %d reclassified, %d fields in %.1fs" % (
        total_refined, total_reclassified, summary['fields_extracted'], elapsed))

    return {
        'reconciled_packets': reconciled_packets,
        'summary': summary,
        'elapsed_seconds': round(elapsed, 2),
    }


if __name__ == '__main__':
    import sys as _main_sys
    if len(_main_sys.argv) < 2:
        print("Usage: python step09_shipping_reconciliation.py <step08_result.json> [step07_result.json]")
        _main_sys.exit(1)
    with open(_main_sys.argv[1], 'r', encoding='utf-8') as f:
        step8 = json.load(f)
    step7 = None
    if len(_main_sys.argv) >= 3:
        with open(_main_sys.argv[2], 'r', encoding='utf-8') as f:
            step7 = json.load(f)
    result = run(step8, step7, output_dir=os.path.dirname(_main_sys.argv[1]))
    print("\nResult: %s" % result['summary'])
