"""
Step 4 -- MT Document Identification
======================================
Classifies each document packet from Step 3 into MT-side (SWIFT LC messages)
or shipping-side (commercial documents).

MT-SIDE (LC documents):
    MT700 (original LC), MT707 (amendment), MT799 (free-format), MT710/720 (transfers)
    These define the RULES -- what documents are required, amounts, dates, conditions.

SHIPPING-SIDE (commercial documents):
    Bills of Lading, Invoices, Certificates, Packing Lists, Drafts, etc.
    These must COMPLY with the LC rules.

HOW IT WORKS:
    Phase 1 -- GLM text pattern matching:
        Searches for SWIFT-specific patterns (F-tags, MT700/707 headers, Alliance
        :20: format) and shipping document patterns (BILL OF LADING, INVOICE, etc.)
    Phase 2 -- VLM classification (Qwen):
        For packets with low confidence (<0.70) or "unknown" classification, sends
        the first page IMAGE + GLM text to Qwen for visual classification.

SWIFT FORMATS:
    Alliance: :20:, :31C: (used by UBL and most banks)
    Fusion:   F20:, F31C: (used by HBL)

INPUT:  Step 3 output -- list of DocumentPacket objects
OUTPUT: List of ClassifiedPacket with mt_type, mt_confidence, swift_format

MODEL:  Qwen VLM at QWEN_VLM_URL (7B or 72B per VLM_MODEL_SIZE switch in
        config/settings.py — only for ambiguous classification)
"""

import os
import sys as _sys; _sys.stdout.reconfigure(encoding="utf-8", errors="replace") if hasattr(_sys.stdout, "reconfigure") else None
import re
import json
import time
import base64
import requests
from dataclasses import dataclass, field, asdict
from typing import List, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import QWEN_VLM_URL, QWEN_VLM_MODEL, VLM_TIMEOUT, MAX_CONCURRENT_VLM


# == Dataclasses ==============================================================

@dataclass
class PacketPage:
    """Page within a classified packet (carried forward from Step 3)."""
    page_number: int
    raw_text: str
    cleaned_text: str
    page_image_path: str
    is_continuation: bool = False
    continuation_label: str = ""
    boundary_reason: str = ""


@dataclass
class ClassifiedPacket:
    """
    A document packet with MT classification.
    mt_type determines routing:
      MT700/MT707/MT799 -> Step 6 (Final LC Consolidation)
      shipping          -> Step 8 (Shipping Classification)
      covering_letter   -> Documentary Remittance processing
    """
    packet_id: int
    pages: List[PacketPage] = field(default_factory=list)
    page_numbers: List[int] = field(default_factory=list)
    page_count: int = 0
    boundary_confidence: float = 0.0
    boundary_method: str = ""
    doc_hint: str = ""
    # Classification fields
    mt_type: str = ""                  # MT700, MT707, MT799, MT710, MT720, shipping, covering_letter, unknown
    mt_confidence: float = 0.0
    classification_method: str = ""    # "text_pattern" | "vlm" | "hybrid"
    classification_reason: str = ""
    swift_format: str = ""             # "alliance" | "fusion" | "unknown"
    # MT799 free-format flagging: when an MT799 carries amendment instructions
    # we promote mt_type to MT707 but remember the original MT here so the
    # downstream pipeline (and report) can show "MT799 amendment".
    source_mt: str = ""                # original MT type before any promotion (e.g. "MT799")
    is_799_amendment: bool = False     # True when MT799/MT999 contained amendment instructions


# == Text-pattern classification ==============================================

# Direct MT type references -- highest confidence
# IMPORTANT: MT799/MT999 (free format) checked FIRST so a 799 message that
# REFERENCES an MT700 (e.g. "UNDER OUR MT700") is not misclassified as MT700.
#
# This list now covers the FULL SWIFT category-7 (Trade Services) range used
# in documentary-credit operations:
#
#   Free-format          : MT799, MT999
#   Documentary credit   : MT700/701 (issue), MT705 (pre-advice),
#                          MT707/708 (amendment), MT710/711 (advice of third
#                          bank), MT720/721 (transfer), MT730 (acknowledgement),
#                          MT732 (advice of discharge), MT734 (advice of refusal),
#                          MT735 (full refusal under reserve)
#   Reimbursement        : MT740 (authorisation), MT742 (claim),
#                          MT744 (notice of non-conforming claim),
#                          MT747 (amendment to authorisation)
#   Discrepancy / payment: MT750 (advice of discrepancy),
#                          MT752 (authorisation to pay/accept/negotiate),
#                          MT754 (advice of payment/acceptance/negotiation),
#                          MT756 (advice of reimbursement or payment)
#   Standby / guarantee  : MT760 (issue), MT767 (amendment),
#                          MT768 (acknowledgement), MT769 (release/reduction),
#                          MT775 (amendment), MT785 (notice of demand),
#                          MT786 (notice of receipt), MT787 (acknowledgement)
_MT_PATTERNS = [
    # Free format -- checked FIRST so a 799 referencing MT700 stays a 799.
    (re.compile(r'\bMT\s*799\b', re.IGNORECASE), 'MT799', 0.95),
    (re.compile(r'\bMT\s*999\b', re.IGNORECASE), 'MT999', 0.95),
    # Documentary credit issuance / amendment
    (re.compile(r'\bMT\s*700\b', re.IGNORECASE), 'MT700', 0.95),
    (re.compile(r'\bMT\s*701\b', re.IGNORECASE), 'MT701', 0.90),
    (re.compile(r'\bMT\s*705\b', re.IGNORECASE), 'MT705', 0.90),
    (re.compile(r'\bMT\s*707\b', re.IGNORECASE), 'MT707', 0.95),
    (re.compile(r'\bMT\s*708\b', re.IGNORECASE), 'MT708', 0.90),
    # Advice of third bank's LC / transfer
    (re.compile(r'\bMT\s*710\b', re.IGNORECASE), 'MT710', 0.90),
    (re.compile(r'\bMT\s*711\b', re.IGNORECASE), 'MT711', 0.90),
    (re.compile(r'\bMT\s*720\b', re.IGNORECASE), 'MT720', 0.90),
    (re.compile(r'\bMT\s*721\b', re.IGNORECASE), 'MT721', 0.90),
    # Acknowledgement / discharge / refusal
    (re.compile(r'\bMT\s*730\b', re.IGNORECASE), 'MT730', 0.90),
    (re.compile(r'\bMT\s*732\b', re.IGNORECASE), 'MT732', 0.90),
    (re.compile(r'\bMT\s*734\b', re.IGNORECASE), 'MT734', 0.90),
    (re.compile(r'\bMT\s*735\b', re.IGNORECASE), 'MT735', 0.90),
    # Reimbursement
    (re.compile(r'\bMT\s*740\b', re.IGNORECASE), 'MT740', 0.90),
    (re.compile(r'\bMT\s*742\b', re.IGNORECASE), 'MT742', 0.90),
    (re.compile(r'\bMT\s*744\b', re.IGNORECASE), 'MT744', 0.90),
    (re.compile(r'\bMT\s*747\b', re.IGNORECASE), 'MT747', 0.90),
    # Discrepancy / authorisation / payment advice
    (re.compile(r'\bMT\s*750\b', re.IGNORECASE), 'MT750', 0.90),
    (re.compile(r'\bMT\s*752\b', re.IGNORECASE), 'MT752', 0.90),
    (re.compile(r'\bMT\s*754\b', re.IGNORECASE), 'MT754', 0.90),
    (re.compile(r'\bMT\s*756\b', re.IGNORECASE), 'MT756', 0.90),
    # Guarantee / Standby LC
    (re.compile(r'\bMT\s*760\b', re.IGNORECASE), 'MT760', 0.90),
    (re.compile(r'\bMT\s*767\b', re.IGNORECASE), 'MT767', 0.90),
    (re.compile(r'\bMT\s*768\b', re.IGNORECASE), 'MT768', 0.90),
    (re.compile(r'\bMT\s*769\b', re.IGNORECASE), 'MT769', 0.90),
    (re.compile(r'\bMT\s*775\b', re.IGNORECASE), 'MT775', 0.90),
    (re.compile(r'\bMT\s*785\b', re.IGNORECASE), 'MT785', 0.90),
    (re.compile(r'\bMT\s*786\b', re.IGNORECASE), 'MT786', 0.90),
    (re.compile(r'\bMT\s*787\b', re.IGNORECASE), 'MT787', 0.90),
    # fin.XXX identifier format
    (re.compile(r'\bfin\.(\d{3})\b', re.IGNORECASE), None, 0.90),  # handled below
    # SWIFT message type headers (alternate "Type: 700" / "Message: 707" format)
    (re.compile(r'(?:Type|Message)\s*:?\s*700', re.IGNORECASE), 'MT700', 0.85),
    (re.compile(r'(?:Type|Message)\s*:?\s*707', re.IGNORECASE), 'MT707', 0.85),
    (re.compile(r'(?:Type|Message)\s*:?\s*710', re.IGNORECASE), 'MT710', 0.85),
    (re.compile(r'(?:Type|Message)\s*:?\s*734', re.IGNORECASE), 'MT734', 0.85),
    (re.compile(r'(?:Type|Message)\s*:?\s*735', re.IGNORECASE), 'MT735', 0.85),
    (re.compile(r'(?:Type|Message)\s*:?\s*750', re.IGNORECASE), 'MT750', 0.85),
    (re.compile(r'(?:Type|Message)\s*:?\s*754', re.IGNORECASE), 'MT754', 0.85),
    (re.compile(r'(?:Type|Message)\s*:?\s*760', re.IGNORECASE), 'MT760', 0.85),
    (re.compile(r'(?:Type|Message)\s*:?\s*767', re.IGNORECASE), 'MT767', 0.85),
    (re.compile(r'(?:Type|Message)\s*:?\s*799', re.IGNORECASE), 'MT799', 0.85),
    (re.compile(r'(?:Type|Message)\s*:?\s*999', re.IGNORECASE), 'MT999', 0.85),
]

# F-tag patterns -- strong indicator of SWIFT (Fusion format)
_FTAG_RE = re.compile(
    r'\bF(20|31[CD]|32[AB]|39[AB]|4[0-9][A-Z]|50|59|7[0-9][A-Z]?|78)\s*:',
    re.IGNORECASE,
)

# Alliance SWIFT format indicators (:20:, :31C:)
_ALLIANCE_PATTERNS = [
    re.compile(r'Alliance\s+(?:Lite2?|Access|Message)', re.IGNORECASE),
    re.compile(r'\{1:F01', re.IGNORECASE),
    re.compile(r':20:', re.IGNORECASE),
    re.compile(r':31[CD]:', re.IGNORECASE),
    re.compile(r':32[AB]:', re.IGNORECASE),
    re.compile(r':46[AB]:', re.IGNORECASE),
    re.compile(r':47[AB]:', re.IGNORECASE),
]

# Fusion (HBL) format indicators (F20:, F31C:)
_FUSION_PATTERNS = [
    re.compile(r'Fusion\s+(?:Banking|Global)', re.IGNORECASE),
    re.compile(r'\bF20\s*:', re.IGNORECASE),
    re.compile(r'\bF31[CD]\s*:', re.IGNORECASE),
    re.compile(r'\bF32[AB]\s*:', re.IGNORECASE),
    re.compile(r'\bF46[AB]\s*:', re.IGNORECASE),
    re.compile(r'\bF47[AB]\s*:', re.IGNORECASE),
]

# Shipping document patterns
_SHIPPING_PATTERNS = [
    (re.compile(r'\bBILL\s+OF\s+LADING\b', re.IGNORECASE), 'bill_of_lading'),
    (re.compile(r'\bCOMMERCIAL\s+INVOICE\b', re.IGNORECASE), 'commercial_invoice'),
    (re.compile(r'\bPACKING\s+LIST\b', re.IGNORECASE), 'packing_list'),
    (re.compile(r'\bWEIGHT\s+LIST\b', re.IGNORECASE), 'weight_list'),
    (re.compile(r'\bCERTIFICATE\s+OF\s+ORIGIN\b', re.IGNORECASE), 'certificate_of_origin'),
    (re.compile(r'\bINSURANCE\s+(?:POLICY|CERTIFICATE)\b', re.IGNORECASE), 'insurance'),
    (re.compile(r'\bDRAFT\b.*\bEXCHANGE\b', re.IGNORECASE), 'draft'),
    (re.compile(r'\bBENEFICIARY\s+CERT', re.IGNORECASE), 'beneficiary_certificate'),
    (re.compile(r'\bINSPECTION\s+CERT', re.IGNORECASE), 'inspection_certificate'),
    (re.compile(r'\bFUMIGATION\s+CERT', re.IGNORECASE), 'fumigation_certificate'),
    (re.compile(r'\bPHYTOSANITARY', re.IGNORECASE), 'phytosanitary_certificate'),
    (re.compile(r'\bSHIPPING\s+ADVICE\b', re.IGNORECASE), 'shipping_advice'),
]

# Covering letter / Documentary Remittance
_COVERING_LETTER_PATTERNS = [
    re.compile(r'\bCOVERING\s+(?:LETTER|SCHEDULE)\b', re.IGNORECASE),
    re.compile(r'\bDOCUMENTARY\s+REMITTANCE\b', re.IGNORECASE),
    re.compile(r'\bENCLOSED\s+HEREWITH\b', re.IGNORECASE),
    re.compile(r'\bWE\s+ARE\s+ENCLOSING\b', re.IGNORECASE),
]

# Amendment indicators
_AMENDMENT_PATTERNS = [
    re.compile(r'\bAMENDMENT\b', re.IGNORECASE),
    re.compile(r'\bAMEND(?:ED|ING|S)\b', re.IGNORECASE),
    re.compile(r'\bF26E\s*:', re.IGNORECASE),
    re.compile(r':26E:', re.IGNORECASE),
]

# MT799 free-format amendment instruction patterns.
# A 799 is an amendment when it contains explicit "should read as / instead of"
# language OR explicit "please amend / replace / delete / add clause" instructions
# referencing a field tag or LC clause.
_MT799_AMENDMENT_INSTRUCTION_PATTERNS = [
    re.compile(r'\bSHOULD\s+READ\s+AS\b', re.IGNORECASE),
    re.compile(r'\bTO\s+READ\s+AS\b', re.IGNORECASE),
    re.compile(r'\bNOW\s+READ\s+AS\b', re.IGNORECASE),
    re.compile(r'\bI\s*/\s*O\b', re.IGNORECASE),               # I/O = "instead of"
    re.compile(r'\bINSTEAD\s+OF\b', re.IGNORECASE),
    re.compile(r'\bPLEASE\s+(?:AMEND|CORRECT|REPLACE|CHANGE)\b', re.IGNORECASE),
    re.compile(r'\b(?:DELETE|ADD)\s+(?:AND\s+REPLACE|CLAUSE)\b', re.IGNORECASE),
    re.compile(r'\bUNDER\s+FIELD\s+\d{2}[A-Z]?\b', re.IGNORECASE),
    re.compile(r'\bAMEND(?:ED|MENT)\s+(?:AS\s+)?(?:UNDER|FOLLOWS?)\b', re.IGNORECASE),
]

# LC reference indicators (must accompany amendment language)
_MT799_LC_REFERENCE_PATTERNS = [
    re.compile(r'\bMT\s*700\b', re.IGNORECASE),
    re.compile(r'\bOUR\s+(?:LC|CREDIT|MT|REF)\b', re.IGNORECASE),
    re.compile(r'\bYOUR\s+(?:LC|CREDIT|MT|REF)\b', re.IGNORECASE),
    re.compile(r'\bLETTER\s+OF\s+CREDIT\b', re.IGNORECASE),
    re.compile(r'\bDOCUMENTARY\s+CREDIT\b', re.IGNORECASE),
    re.compile(r'\bUNDER\s+(?:THE\s+)?(?:ABOVE|CAPTIONED|REFERENCED)\b', re.IGNORECASE),
]


def _is_mt799_amendment(text: str) -> bool:
    """
    Decide whether a free-format MT799 message contains amendment INSTRUCTIONS
    (vs. a generic bank-to-bank notice / acknowledgement / status update).

    Returns True only when BOTH:
      - The message uses amendment-instruction language
        ("should read as", "instead of", "please amend...", "under field 45A...")
      - It references the underlying LC ("MT700", "our LC", "letter of credit",
        "above captioned credit", etc.)
    """
    if not text:
        return False
    has_instruction = any(p.search(text) for p in _MT799_AMENDMENT_INSTRUCTION_PATTERNS)
    if not has_instruction:
        return False
    has_lc_ref = any(p.search(text) for p in _MT799_LC_REFERENCE_PATTERNS)
    return has_lc_ref


def _get_packet_text(packet) -> str:
    """Concatenate cleaned text from all pages in a packet."""
    pages = packet.pages if hasattr(packet, 'pages') else packet.get('pages', [])
    texts = []
    for p in pages:
        t = p.cleaned_text if hasattr(p, 'cleaned_text') else p.get('cleaned_text', '')
        texts.append(t)
    return '\n'.join(texts)


def _get_first_image(packet) -> str:
    """Get path of first page image in packet."""
    pages = packet.pages if hasattr(packet, 'pages') else packet.get('pages', [])
    if pages:
        p = pages[0]
        return p.page_image_path if hasattr(p, 'page_image_path') else p.get('page_image_path', '')
    return ''


def _detect_swift_format(text: str) -> str:
    """
    Detect Alliance vs Fusion SWIFT format.
    Alliance: :20:VALUE (colon-wrapped)
    Fusion:   F20: VALUE (F-prefixed)
    """
    alliance_score = sum(1 for p in _ALLIANCE_PATTERNS if p.search(text))
    fusion_score = sum(1 for p in _FUSION_PATTERNS if p.search(text))
    if fusion_score > alliance_score:
        return 'fusion'
    elif alliance_score > 0:
        return 'alliance'
    return 'unknown'


def _classify_by_text(text: str) -> dict:
    """
    Classify a document packet by GLM text pattern matching.

    Priority:
    1. Covering letter (checked first -- can contain MT references as false positives)
    2. Explicit MT type reference (MT700, MT707, etc.)
    3. F-tag count (3+ F-tags = likely SWIFT message)
    4. Alliance tag count
    5. Shipping document patterns
    6. Default: unknown
    """
    # Covering letter check first
    cl_score = sum(1 for p in _COVERING_LETTER_PATTERNS if p.search(text))
    if cl_score >= 2:
        return {
            'mt_type': 'covering_letter',
            'confidence': 0.90,
            'reason': f'{cl_score} covering letter patterns',
            'swift_format': '',
        }

    # Check for explicit MT type
    for pattern, mt_type, conf in _MT_PATTERNS:
        if pattern.search(text):
            is_amendment = any(p.search(text) for p in _AMENDMENT_PATTERNS)
            if is_amendment and mt_type == 'MT700':
                mt_type = 'MT707'
                conf *= 0.9

            swift_fmt = _detect_swift_format(text)
            result = {
                'mt_type': mt_type,
                'confidence': conf,
                'reason': f'matched {mt_type} pattern',
                'swift_format': swift_fmt,
            }
            # MT799/MT999 free format: inspect content. If it carries
            # amendment INSTRUCTIONS referencing the underlying LC, promote
            # it to MT707 so step06 will apply the amendment. Otherwise
            # keep it as MT799 (stored as supplementary narrative).
            if mt_type in ('MT799', 'MT999'):
                if _is_mt799_amendment(text):
                    result['mt_type'] = 'MT707'
                    result['confidence'] = max(conf * 0.9, 0.80)
                    result['reason'] = (
                        f'{mt_type} free-format with amendment instructions '
                        f'(promoted to MT707)'
                    )
                    result['source_mt'] = mt_type
                    result['is_799_amendment'] = True
                else:
                    result['reason'] = f'{mt_type} free-format message (no amendment instructions)'
                    result['is_799_amendment'] = False
            return result

    # Check F-tags (Fusion format)
    ftag_matches = _FTAG_RE.findall(text)
    if len(ftag_matches) >= 3:
        is_amendment = any(p.search(text) for p in _AMENDMENT_PATTERNS)
        mt_type = 'MT707' if is_amendment else 'MT700'
        return {
            'mt_type': mt_type,
            'confidence': 0.80,
            'reason': f'{len(ftag_matches)} F-tags found (Fusion)',
            'swift_format': 'fusion',
        }

    # Check Alliance tags (:20:, :31C:, etc.)
    alliance_tags = re.findall(r':(\d{2}[A-Z]?):', text)
    swift_tags = [t for t in alliance_tags if t in (
        '20', '21', '23', '26E', '27', '30', '31C', '31D', '32B', '33B',
        '34B', '39A', '40A', '40E', '41A', '41D', '42A', '42C',
        '43P', '43T', '44A', '44B', '44C', '44E', '44F',
        '45A', '46A', '47A', '48', '49', '50', '59', '71D', '78',
    )]
    if len(swift_tags) >= 3:
        is_amendment = any(p.search(text) for p in _AMENDMENT_PATTERNS)
        mt_type = 'MT707' if is_amendment else 'MT700'
        return {
            'mt_type': mt_type,
            'confidence': 0.80,
            'reason': f'{len(swift_tags)} Alliance tags found',
            'swift_format': 'alliance',
        }

    # Weak F-tag signal
    if len(ftag_matches) >= 1 or len(swift_tags) >= 1:
        swift_fmt = _detect_swift_format(text)
        return {
            'mt_type': 'MT700',
            'confidence': 0.50,
            'reason': f'{len(ftag_matches)} F-tag(s) + {len(swift_tags)} Alliance tag(s), weak signal',
            'swift_format': swift_fmt,
        }

    # Check shipping document patterns
    for pattern, doc_type in _SHIPPING_PATTERNS:
        if pattern.search(text[:1500]):
            return {
                'mt_type': 'shipping',
                'confidence': 0.85,
                'reason': f'shipping doc: {doc_type}',
                'swift_format': '',
            }

    # No clear match
    return {
        'mt_type': 'unknown',
        'confidence': 0.30,
        'reason': 'no matching patterns',
        'swift_format': '',
    }


# == VLM classification (Qwen) ===========================================

_VLM_CLASSIFY_PROMPT = """You are analyzing a scanned trade finance document page.

Here is the GLM OCR text for this page (first 1000 chars):
---
{text_preview}
---

Classify this document into ONE of these categories:
- MT700: Original Letter of Credit (SWIFT LC issuance) -- contains fields like F20, F31C, F32B, F46A, F47A or :20:, :31C:, :32B:, :46A:, :47A:
- MT701: Continuation of MT700 (overflow page)
- MT705: Pre-Advice of a Documentary Credit
- MT707: LC Amendment -- contains amendment number (F26E or :26E:), references to amending an existing LC
- MT708: Continuation of MT707
- MT710: Advice of Third Bank's LC
- MT711: Continuation of MT710
- MT720: Transfer of LC
- MT721: Continuation of MT720
- MT730: Acknowledgement
- MT732: Advice of Discharge
- MT734: Advice of Refusal
- MT735: Full Refusal Under Reserve
- MT740: Authorisation to Reimburse
- MT742: Reimbursement Claim
- MT744: Notice of Non-Conforming Reimbursement Claim
- MT747: Amendment to Authorisation to Reimburse
- MT750: Advice of Discrepancy
- MT752: Authorisation to Pay / Accept / Negotiate
- MT754: Advice of Payment / Acceptance / Negotiation
- MT756: Advice of Reimbursement or Payment
- MT760: Issue of Demand Guarantee / Standby LC
- MT767: Amendment to Demand Guarantee / Standby
- MT768: Acknowledgement of Guarantee Amendment
- MT769: Notice of Reduction or Release
- MT775: Further Amendment to Documentary Credit
- MT785/MT786/MT787: Guarantee notice / receipt / acknowledgement
- MT799: Free format SWIFT message -- bank-to-bank communication. IMPORTANT: an MT799 may CARRY amendment instructions (e.g. "UNDER FIELD 45A SHOULD READ AS X I/O Y", "PLEASE AMEND ... INSTEAD OF ..."). When the 799 contains such instructions, classify it as MT707 (treat as amendment), not MT799. Use MT799 ONLY when the message is a generic notice / acknowledgement / status with NO amendment instructions.
- MT999: Free format SWIFT message (general financial — same handling as MT799)
- shipping: Shipping/commercial document (Bill of Lading, Invoice, Packing List, Certificate, Draft, etc.)
- covering_letter: Documentary Remittance / Covering Letter / Covering Schedule
- unknown: Cannot determine

Also determine the SWIFT format if applicable:
- alliance: Uses colon notation like :20:, :31C:, :46A:
- fusion: Uses F-prefix notation like F20:, F31C:, F46A:

Respond in EXACTLY this JSON format:
{{"mt_type": "MT700|MT707|MT799|shipping|covering_letter|unknown", "swift_format": "alliance|fusion|unknown", "confidence": 0.0-1.0, "reason": "brief explanation", "document_subtype": "e.g. Bill of Lading, Commercial Invoice, etc."}}"""


def _vlm_classify(image_path: str, text: str) -> dict:
    """Send first page image + GLM text to Qwen for classification."""
    prompt = _VLM_CLASSIFY_PROMPT.format(text_preview=text)

    messages = [{"role": "user", "content": []}]

    if image_path and os.path.exists(image_path):
        try:
            with open(image_path, 'rb') as f:
                img_b64 = base64.b64encode(f.read()).decode('utf-8')
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_b64}"},
            })
        except Exception:
            pass  # proceed without image

    messages[0]["content"].append({"type": "text", "text": prompt})

    try:
        resp = requests.post(
            QWEN_VLM_URL,
            json={
                "model": QWEN_VLM_MODEL,
                "messages": messages,
                "max_tokens": 400,
                "temperature": 0.1,
            },
            timeout=VLM_TIMEOUT,
        )
        if resp.status_code != 200:
            return {'error': f'HTTP {resp.status_code}'}

        content = resp.json()['choices'][0]['message']['content']
        json_match = re.search(r'\{[^{}]+\}', content, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            return {
                'mt_type': parsed.get('mt_type', 'unknown'),
                'confidence': float(parsed.get('confidence', 0.5)),
                'reason': parsed.get('reason', 'vlm'),
                'swift_format': parsed.get('swift_format', 'unknown'),
                'doc_subtype': parsed.get('document_subtype', ''),
            }
        return {'error': 'no JSON in response', 'raw': content[:200]}

    except requests.exceptions.Timeout:
        return {'error': 'vlm_timeout'}
    except Exception as e:
        return {'error': str(e)}


# == Main run function ========================================================

def run(step3_result: dict, output_dir: str = None, progress_callback=None) -> dict:
    """
    Execute Step 4: MT Document Identification.

    Args:
        step3_result: Output from Step 3 (with 'packets' list)
        output_dir: Directory to save results
        progress_callback: Optional callback(message: str)

    Returns:
        dict with 'packets', 'mt_packets', 'shipping_packets', 'elapsed_seconds'
    """
    def _progress(msg):
        if progress_callback:
            progress_callback(f"[Step 4] {msg}")
        print(f"[Step 4] {msg}")

    start_time = time.time()
    packets_in = step3_result.get('packets', [])
    _progress(f"Classifying {len(packets_in)} packets...")

    classified: List[ClassifiedPacket] = []
    vlm_calls = 0
    vlm_needed = []  # (index, packet_data) for VLM calls

    # -- Phase 1: Text-based classification for all packets --
    text_results = []
    for i, pkt_in in enumerate(packets_in):
        if hasattr(pkt_in, 'packet_id'):
            pkt_id = pkt_in.packet_id
            pages_data = pkt_in.pages
            page_numbers = pkt_in.page_numbers
            page_count = pkt_in.page_count
            b_conf = pkt_in.boundary_confidence
            b_method = pkt_in.boundary_method
            doc_hint = pkt_in.doc_hint
        else:
            pkt_id = pkt_in.get('packet_id', 0)
            pages_data = pkt_in.get('pages', [])
            page_numbers = pkt_in.get('page_numbers', [])
            page_count = pkt_in.get('page_count', 0)
            b_conf = pkt_in.get('boundary_confidence', 0)
            b_method = pkt_in.get('boundary_method', '')
            doc_hint = pkt_in.get('doc_hint', '')

        # Build PacketPage list
        pkt_pages = []
        for p in pages_data:
            if hasattr(p, 'page_number'):
                pkt_pages.append(PacketPage(
                    page_number=p.page_number, raw_text=p.raw_text,
                    cleaned_text=p.cleaned_text, page_image_path=p.page_image_path,
                    is_continuation=p.is_continuation, continuation_label=p.continuation_label,
                    boundary_reason=p.boundary_reason,
                ))
            else:
                pkt_pages.append(PacketPage(
                    page_number=p.get('page_number', 0), raw_text=p.get('raw_text', ''),
                    cleaned_text=p.get('cleaned_text', ''), page_image_path=p.get('page_image_path', ''),
                    is_continuation=p.get('is_continuation', False),
                    continuation_label=p.get('continuation_label', ''),
                    boundary_reason=p.get('boundary_reason', ''),
                ))

        full_text = '\n'.join(p.cleaned_text for p in pkt_pages)
        first_img = pkt_pages[0].page_image_path if pkt_pages else ''

        text_cls = _classify_by_text(full_text)

        text_results.append({
            'pkt_id': pkt_id,
            'pkt_pages': pkt_pages,
            'page_numbers': page_numbers if page_numbers else [p.page_number for p in pkt_pages],
            'page_count': page_count if page_count else len(pkt_pages),
            'b_conf': b_conf,
            'b_method': b_method,
            'doc_hint': doc_hint,
            'full_text': full_text,
            'first_img': first_img,
            'text_cls': text_cls,
        })

        # Check if VLM needed
        if text_cls['confidence'] < 0.70 or text_cls['mt_type'] == 'unknown':
            vlm_needed.append(i)

    # -- Phase 2: VLM for uncertain packets (concurrent) --
    if vlm_needed:
        _progress(f"  {len(vlm_needed)} packets need VLM classification...")

        def _do_vlm(idx):
            tr = text_results[idx]
            result = _vlm_classify(tr['first_img'], tr['full_text'])
            return (idx, result)

        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_VLM) as executor:
            futures = {executor.submit(_do_vlm, idx): idx for idx in vlm_needed}
            for future in as_completed(futures):
                try:
                    idx, vlm_cls = future.result()
                    vlm_calls += 1
                    tr = text_results[idx]

                    if 'error' not in vlm_cls:
                        if vlm_cls['confidence'] > tr['text_cls']['confidence']:
                            # Preserve MT799-amendment classification if the
                            # text-pattern already promoted it; VLM rarely sees
                            # the "should read as / I/O" instruction language.
                            prior = tr['text_cls']
                            keep_799_amend = bool(prior.get('is_799_amendment'))
                            tr['text_cls'] = {
                                'mt_type': vlm_cls['mt_type'],
                                'confidence': vlm_cls['confidence'],
                                'reason': f"vlm: {vlm_cls['reason']}",
                                'swift_format': vlm_cls.get('swift_format', prior.get('swift_format', '')),
                            }
                            if keep_799_amend:
                                tr['text_cls']['mt_type'] = 'MT707'
                                tr['text_cls']['source_mt'] = prior.get('source_mt') or 'MT799'
                                tr['text_cls']['is_799_amendment'] = True
                                tr['text_cls']['reason'] = (
                                    'MT799 free-format with amendment instructions '
                                    f"(promoted to MT707); vlm: {vlm_cls['reason']}"
                                )
                            elif vlm_cls['mt_type'] in ('MT799', 'MT999'):
                                # VLM identified as 799 — re-run amendment detection
                                if _is_mt799_amendment(tr['full_text']):
                                    tr['text_cls']['mt_type'] = 'MT707'
                                    tr['text_cls']['source_mt'] = vlm_cls['mt_type']
                                    tr['text_cls']['is_799_amendment'] = True
                                    tr['text_cls']['reason'] = (
                                        f"{vlm_cls['mt_type']} free-format with amendment "
                                        f"instructions (promoted to MT707)"
                                    )
                                else:
                                    tr['text_cls']['is_799_amendment'] = False
                            tr['cls_method'] = 'vlm'
                            if vlm_cls.get('doc_subtype'):
                                tr['doc_hint'] = vlm_cls['doc_subtype']
                        else:
                            tr['cls_method'] = 'hybrid'
                        _progress(f"    VLM packet {tr['pkt_id']}: {vlm_cls.get('mt_type', '?')} "
                                  f"(conf={vlm_cls.get('confidence', 0):.2f})")
                    else:
                        _progress(f"    VLM error packet {tr['pkt_id']}: {vlm_cls['error']}")
                except Exception as e:
                    _progress(f"    VLM future error: {e}")

    # -- Build classified packets --
    for tr in text_results:
        text_cls = tr['text_cls']
        cls_method = tr.get('cls_method', 'text_pattern')

        cpkt = ClassifiedPacket(
            packet_id=tr['pkt_id'],
            pages=tr['pkt_pages'],
            page_numbers=tr['page_numbers'],
            page_count=tr['page_count'],
            boundary_confidence=tr['b_conf'],
            boundary_method=tr['b_method'],
            doc_hint=tr['doc_hint'],
            mt_type=text_cls['mt_type'],
            mt_confidence=text_cls['confidence'],
            classification_method=cls_method,
            classification_reason=text_cls['reason'],
            swift_format=text_cls.get('swift_format', ''),
            source_mt=text_cls.get('source_mt', ''),
            is_799_amendment=bool(text_cls.get('is_799_amendment', False)),
        )
        classified.append(cpkt)

        _progress(f"  Packet {cpkt.packet_id} (pages {cpkt.page_numbers}): "
                  f"{cpkt.mt_type} (conf={cpkt.mt_confidence:.2f}, fmt={cpkt.swift_format}, method={cls_method})")

    # Summary
    mt_packets = [p for p in classified if p.mt_type.startswith('MT')]
    shipping_packets = [p for p in classified if p.mt_type == 'shipping']
    other_packets = [p for p in classified if p.mt_type not in ('shipping',) and not p.mt_type.startswith('MT')]

    elapsed = time.time() - start_time
    _progress(f"Step 4 complete: {len(mt_packets)} MT, {len(shipping_packets)} shipping, "
              f"{len(other_packets)} other in {elapsed:.1f}s ({vlm_calls} VLM calls)")

    # Save results
    result_file = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        result_file = os.path.join(output_dir, 'step04_result.json')
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({
                'step': 4,
                'step_name': 'MT Document Identification',
                'total_packets': len(classified),
                'mt_packets': len(mt_packets),
                'shipping_packets': len(shipping_packets),
                'other_packets': len(other_packets),
                'vlm_calls': vlm_calls,
                'elapsed_seconds': round(elapsed, 2),
                'packets': [asdict(p) for p in classified],
            }, f, indent=2, ensure_ascii=False)

    return {
        'packets': classified,
        'mt_packets': len(mt_packets),
        'shipping_packets': len(shipping_packets),
        'elapsed_seconds': round(elapsed, 2),
        'errors': [],
        'result_file': result_file,
    }


if __name__ == '__main__':
    import sys as _sys
    if len(_sys.argv) < 2:
        print("Usage: python step04_mt_identification.py <step03_result.json>")
        _sys.exit(1)
    with open(_sys.argv[1], 'r', encoding='utf-8') as f:
        step3 = json.load(f)
    result = run(step3, output_dir=os.path.dirname(_sys.argv[1]))
    print(f"\nResult: {result['mt_packets']} MT, {result['shipping_packets']} shipping, {result['elapsed_seconds']}s")
