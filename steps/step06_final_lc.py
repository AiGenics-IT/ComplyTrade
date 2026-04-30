"""
Step 6 -- Final LC Consolidation
==================================
Takes all MT-side document packets (original LC + amendments) and produces a
single consolidated "Final LC" -- the definitive version of the Letter of Credit
with all amendments applied in sequence.

WHY THIS MATTERS:
    An LC starts as MT700 (original) and may be modified by MT707 (amendments).
    Each amendment can change amounts, dates, required documents, shipping terms.
    Compliance checking must use the FINAL version after all amendments.

HOW IT WORKS:
    1. Find MT700 packet -> extract SWIFT fields from GLM text using regex
    2. Sort MT707 packets by amendment number or date
    3. Apply each amendment in sequence (later overrides earlier)
    4. Split clause fields (F46A, F47A, F45A, F78) into numbered clauses
    5. Store MT799 (free-format) as supplementary information

SWIFT FIELD EXTRACTION:
    Supports both formats via regex:
    - Alliance: :20:VALUE, :31C:VALUE, :46A:VALUE (colon-wrapped)
    - Fusion:   F20: VALUE, F31C: VALUE, F46A: VALUE (F-prefixed)

AMENDMENT RULES:
    - F34B (New Amount) replaces F32B (Original Amount)
    - B-suffix tags replace A-suffix: F45B->F45A, F46B->F46A, F47B->F47A
    - F26E = amendment number, F30 = amendment date

INPUT:  Step 5 output -- reconciled packets with mt_type and refined_text
OUTPUT: FinalLC with dc_number, consolidated_fields, clauses, amendment_log

MODEL:  None -- pure deterministic regex extraction and merging
"""

import os
import sys as _sys; _sys.stdout.reconfigure(encoding="utf-8", errors="replace") if hasattr(_sys.stdout, "reconfigure") else None
import re
import json
import time
import base64
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import QWEN_VLM_URL, QWEN_VLM_MODEL, MAX_CONCURRENT_VLM, VLM_TIMEOUT
try:
    from config.settings import QWEN_TEXT_LLM_URL, QWEN_TEXT_LLM_MODEL
except ImportError:
    QWEN_TEXT_LLM_URL = None
    QWEN_TEXT_LLM_MODEL = None


# == Dataclasses ==============================================================

@dataclass
class SwiftField:
    """A single extracted SWIFT field from an LC message."""
    tag: str                   # e.g. "20", "31C", "46A"
    label: str                 # e.g. "Documentary Credit Number"
    value: str
    source_page: int = 0
    source_mt: str = ""        # MT700, MT707, etc.


@dataclass
class Clause:
    """
    A single clause extracted from a multi-clause SWIFT field.
    e.g. F46A clause 1: "COMMERCIAL INVOICE IN 3 ORIGINALS"
    """
    clause_number: int         # 1-based position
    clause_id: str             # e.g. "46A-1", "47A-3"
    text: str
    parent_tag: str            # e.g. "46A"


@dataclass
class AmendmentRecord:
    """Record of a single amendment (MT707) applied to the LC."""
    amendment_number: int
    source_packet_id: int
    amendment_date: str
    fields_changed: List[str] = field(default_factory=list)
    change_details: Dict[str, dict] = field(default_factory=dict)


@dataclass
class FinalLC:
    """The consolidated Final LC after all amendments."""
    dc_number: str = ""
    consolidated_fields: Dict[str, str] = field(default_factory=dict)
    clauses: Dict[str, List[Clause]] = field(default_factory=dict)
    original_fields: Dict[str, str] = field(default_factory=dict)
    amendment_log: List[AmendmentRecord] = field(default_factory=list)
    amendment_count: int = 0
    swift_format: str = ""
    source_packets: List[int] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# == SWIFT Field Definitions ==================================================

SWIFT_FIELD_LABELS = {
    '20': 'Documentary Credit Number',
    '21': 'Related Reference',
    '23': 'Reference to Pre-Advice',
    '26E': 'Number of Amendment',
    '27': 'Sequence of Total',
    '30': 'Date of Amendment',
    '31C': 'Date of Issue',
    '31D': 'Date and Place of Expiry',
    '32B': 'Currency Code, Amount',
    '33B': 'Additional Amount Covered',
    '34B': 'New Amount After Amendment',
    '39A': 'Percentage Credit Amount Tolerance',
    '39B': 'Maximum Credit Amount',
    '39C': 'Additional Amounts Covered',
    '40A': 'Form of Documentary Credit',
    '40E': 'Applicable Rules',
    '41A': 'Available With... By...',
    '41D': 'Available With... By...',
    '42A': 'Drawee',
    '42D': 'Drawee - Name and Address',
    '42C': 'Drafts at...',
    '42M': 'Mixed Payment Details',
    '42P': 'Negotiation/Deferred Payment Details',
    '43P': 'Partial Shipments',
    '43T': 'Transhipment',
    '44A': 'Place of Taking in Charge',
    '44B': 'Place of Final Destination',
    '44C': 'Latest Date of Shipment',
    '44D': 'Shipment Period',
    '44E': 'Port of Loading',
    '44F': 'Port of Discharge',
    '45A': 'Description of Goods and/or Services',
    '45B': 'Description of Goods (Amendment)',
    '46A': 'Documents Required',
    '46B': 'Documents Required (Amendment)',
    '47A': 'Additional Conditions',
    '47B': 'Additional Conditions (Amendment)',
    '48': 'Period for Presentation',
    '49': 'Confirmation Instructions',
    '50': 'Applicant',
    '51A': 'Applicant Bank',
    '51D': 'Applicant Bank',
    '52A': 'Issuing Bank',
    '52D': 'Issuing Bank',
    '53A': 'Reimbursing Bank',
    '57A': 'Advising Through Bank',
    '57D': 'Advising Through Bank',
    '59': 'Beneficiary',
    '71D': 'Charges',
    '72': 'Sender to Receiver Information',
    '78': 'Instructions to Paying/Accepting/Negotiating Bank',
    '79': 'Narrative',
}

# Tags that contain clause-type content (split into individual clauses)
CLAUSE_TAGS = {'45A', '45B', '46A', '46B', '47A', '47B', '78', '79', '72'}

# Tags to extract -- ordered by typical SWIFT message appearance
EXTRACTION_TAGS = [
    '20', '21', '23', '26E', '27', '30', '31C', '31D', '32B', '33B', '34B',
    '39A', '39B', '39C', '40A', '40E', '41A', '41D', '42A', '42C', '42D',
    '42M', '42P', '43P', '43T', '44A', '44B', '44C', '44D', '44E', '44F',
    '45A', '45B', '46A', '46B', '47A', '47B', '48', '49', '50', '51A',
    '51D', '52A', '52D', '53A', '57A', '57D', '59', '71D', '72', '78', '79',
]


# == SWIFT field extraction ===================================================

def _build_tag_patterns(tags: list) -> list:
    """
    Build regex patterns for SWIFT field extraction in three formats.

    Alliance:       :20:0491ILC081972          (colon-wrapped tag)
    Fusion (F-tag): F20: 0491ILC081972         (F-prefix tag)
    Fusion (bare):  20: Documentary Credit Number\n0491ILC081972  (bare tag with label)

    Each pattern captures the value until the next tag or end of text.
    """
    patterns = []
    for tag in tags:
        # Alliance format: :TAG:VALUE
        patterns.append({
            'tag': tag,
            'format': 'alliance',
            'regex': re.compile(
                r'(?:^|\n)\s*:' + re.escape(tag) + r':\s*(.*?)(?=\n\s*:[A-Z0-9]{2,4}:|\Z)',
                re.DOTALL,
            ),
        })
        # Fusion format: FTAG: VALUE
        patterns.append({
            'tag': tag,
            'format': 'fusion',
            'regex': re.compile(
                r'(?:^|\n)\s*F' + re.escape(tag) + r'\s*:\s*(.*?)(?=\n\s*F[A-Z0-9]{2,4}\s*:|\Z)',
                re.DOTALL,
            ),
        })
        # Bare Fusion format: TAG: Label\nVALUE  (OCR'd Fusion pages without F prefix)
        # Lookahead stops at next bare TAG: pattern or end of text
        patterns.append({
            'tag': tag,
            'format': 'bare_fusion',
            'regex': re.compile(
                r'(?:^|\n)\s*' + re.escape(tag) + r':\s*(.*?)(?=\n\s*\d{2}[A-Z]?:\s|\Z)',
                re.DOTALL,
            ),
        })
    return patterns


_TAG_PATTERNS = _build_tag_patterns(EXTRACTION_TAGS)


def _extract_swift_fields(text: str, source_page: int = 0, source_mt: str = '') -> List[SwiftField]:
    """
    Extract all known SWIFT fields from GLM text using regex.
    First match wins per tag (avoids duplicates when both formats match).
    """
    # P120g: Normalize full-width colons (U+FF1A) to standard colons.
    # Some OCR engines (especially for CJK documents) produce F43T：
    # instead of F43T: — the regex won't match full-width colons.
    text = text.replace('\uff1a', ':')

    # P101: Pre-normalize — insert newline before F-tags that are glued to
    # the previous field's value (OCR artifact).
    # e.g. "#31,674.67F45B: Description..." → "#31,674.67\nF45B: Description..."
    text = re.sub(r'([^\n])(F\d{2}[A-Z]?\s*:)', r'\1\n\2', text)
    # Also for Alliance colon-format: "...valueF:32B:" → "...value\n:32B:"
    text = re.sub(r'([^\n])(:\d{2}[A-Z]?:)', r'\1\n\2', text)

    # P107: Fix OCR-truncated tags where leading digits are lost.
    # e.g. "D: Date and Place of Expiry" → "31D: Date and Place of Expiry"
    # Only fix known SWIFT field label patterns to avoid false positives.
    _TRUNCATED_TAG_FIXES = [
        (r'(?<=\n)\s*D:\s*(?=Date\s+and\s+Place\s+of\s+Expiry)',   '31D: '),
        (r'(?<=\n)\s*C:\s*(?=Date\s+of\s+Issue)',                   '31C: '),
        (r'(?<=\n)\s*B:\s*(?=Currency\s+Code)',                     '32B: '),
        (r'(?<=\n)\s*A:\s*(?=Form\s+of\s+Documentary\s+Credit)',    '40A: '),
        (r'(?<=\n)\s*A:\s*(?=Available\s+With)',                    '41A: '),
        (r'(?<=\n)\s*D:\s*(?=Available\s+With)',                    '41D: '),
        (r'(?<=\n)\s*C:\s*(?=Drafts\s+at)',                         '42C: '),
        (r'(?<=\n)\s*D:\s*(?=Drawee)',                              '42D: '),
        (r'(?<=\n)\s*P:\s*(?=Partial\s+Shipment)',                  '43P: '),
        (r'(?<=\n)\s*T:\s*(?=Transship)',                           '43T: '),
        (r'(?<=\n)\s*A:\s*(?=(?:Place|Port)\s+of\s+(?:Loading|Taking))', '44A: '),
        (r'(?<=\n)\s*E:\s*(?=Port\s+of\s+(?:Loading|Discharge))',  '44E: '),
        (r'(?<=\n)\s*F:\s*(?=Port\s+of\s+Discharge)',              '44F: '),
        (r'(?<=\n)\s*B:\s*(?=Place\s+of\s+Final\s+Destination)',   '44B: '),
        (r'(?<=\n)\s*D:\s*(?=Charges)',                             '71D: '),
    ]
    for _pat, _repl in _TRUNCATED_TAG_FIXES:
        text = re.sub(_pat, _repl, text, count=1, flags=re.IGNORECASE)

    fields = []
    found_tags = set()

    for pat in _TAG_PATTERNS:
        if pat['tag'] in found_tags:
            continue
        m = pat['regex'].search(text)
        if m:
            value = m.group(1).strip()
            # Collapse excessive blank lines
            value = re.sub(r'\n{3,}', '\n\n', value)
            value = value.strip()
            if value:
                fields.append(SwiftField(
                    tag=pat['tag'],
                    label=SWIFT_FIELD_LABELS.get(pat['tag'], f'Field {pat["tag"]}'),
                    value=value,
                    source_page=source_page,
                    source_mt=source_mt,
                ))
                found_tags.add(pat['tag'])

    return fields


# MT799 free-format → amendment field map
_MT799_TAG_TO_FIELD = {
    '45A': '45A', '45B': '45A',
    '46A': '46A', '46B': '46A',
    '47A': '47A', '47B': '47A',
    '48': '48',
    '44C': '44C', '44E': '44E', '44F': '44F',
    '32B': '32B',
    '31D': '31D',
    '39A': '39A',
    '49': '49',
    '50': '50', '59': '59',
}


def _extract_mt799_amendment_fields(
    text: str, source_page: int = 0, source_mt: str = 'MT799'
) -> List[SwiftField]:
    """
    Parse field-level amendments embedded in a free-format MT799/MT999 message.

    Recognises the patterns banks use when sending an amendment via 799 instead
    of MT707, e.g.:
        UNDER FIELD 45A RATE SHOULD READ AS 'EUR 141.00' I/O 'EUR 140.00'
        FIELD 47A SHOULD READ AS '...' INSTEAD OF '...'
        PLEASE AMEND LATEST DATE OF SHIPMENT TO READ AS 15.05.2026 I/O 30.04.2026

    Real-world MT799 messages from Alliance often format the values on the
    NEXT line after "SHOULD READ AS" and wrap them in DOUBLE single-quotes
    (`''X''`), e.g.:
        UNDER FIELD 45A RATE SHOULD READ AS
        ''EUR 141,396.00'' I/O ''EUR 141,396.56''
    so the parser must (a) tolerate one or more newlines between the verb
    and the value, and (b) tolerate one or more quote characters of either
    style around the value.

    Returns a list of SwiftField records carrying amendment OPERATIONS that
    `_apply_amendment` / `_apply_text_amendment` will then merge into the base
    LC fields. The value is wrapped as "TO READ AS '<new>' INSTEAD OF '<old>'"
    so the existing text-amendment path applies it.
    """
    if not text:
        return []

    # Pre-normalise: collapse runs of single/double quotes (e.g. `''X''` →
    # `'X'`) so the value-capture regexes only need to handle one quote on
    # each side. Also normalise smart quotes to ASCII.
    norm = text
    norm = norm.replace('\u2018', "'").replace('\u2019', "'")
    norm = norm.replace('\u201c', '"').replace('\u201d', '"')
    norm = re.sub(r"'{2,}", "'", norm)
    norm = re.sub(r'"{2,}', '"', norm)

    out: List[SwiftField] = []
    seen_tags = set()

    def _record(canon_tag, new_val, old_val):
        if canon_tag in seen_tags or not new_val:
            return
        seen_tags.add(canon_tag)
        out.append(SwiftField(
            tag=canon_tag,
            label=SWIFT_FIELD_LABELS.get(canon_tag, f'Field {canon_tag}'),
            value=f"TO READ AS '{new_val}' INSTEAD OF '{old_val}'",
            source_page=source_page,
            source_mt=source_mt,
        ))

    # ── Pattern 1a: QUOTED form ──
    #   UNDER FIELD 45A [RATE] SHOULD READ AS
    #   'EUR 141,396.00' I/O 'EUR 141,396.56'
    # The value char class excludes quotes and newlines, so the closing
    # quote bounds the capture cleanly even with periods inside the value.
    pat_field_quoted = re.compile(
        r'(?:UNDER\s+)?FIELD\s+(\d{2}[A-Z]?)\b[^\n]{0,80}?'   # FIELD 45A [RATE]
        r'(?:SHOULD|SHALL|NOW|TO)\s+READ\s+AS'                # SHOULD READ AS
        r'[\s\n\r]*'                                          # whitespace incl. newline
        r'[\'"]\s*([^\'"\n\r]+?)\s*[\'"]'                     # 'new value'
        r'\s*(?:I\s*/\s*O|INSTEAD\s+OF)\s*'                   # I/O / INSTEAD OF
        r'[\'"]\s*([^\'"\n\r]+?)\s*[\'"]',                    # 'old value'
        re.IGNORECASE,
    )
    for m in pat_field_quoted.finditer(norm):
        raw_tag = m.group(1).upper().strip()
        new_val = m.group(2).strip()
        old_val = m.group(3).strip()
        canon_tag = _MT799_TAG_TO_FIELD.get(raw_tag, raw_tag)
        _record(canon_tag, new_val, old_val)

    # ── Pattern 1b: UNQUOTED form ──
    #   UNDER FIELD 45A SHOULD READ AS EUR 141.00 I/O EUR 140.00
    # Bound by newline, period, or another keyword.
    pat_field_unquoted = re.compile(
        r'(?:UNDER\s+)?FIELD\s+(\d{2}[A-Z]?)\b[^\n]{0,80}?'
        r'(?:SHOULD|SHALL|NOW|TO)\s+READ\s+AS\s+'
        r'([^\n\r\'"]{1,200}?)'                                # new value (no quotes, no newline)
        r'\s+(?:I\s*/\s*O|INSTEAD\s+OF)\s+'
        r'([^\n\r\'"]{1,200}?)'                                # old value
        r'(?=\s*(?:[\n\r]|$|REGARDS\b|THANKS\b))',
        re.IGNORECASE,
    )
    for m in pat_field_unquoted.finditer(norm):
        raw_tag = m.group(1).upper().strip()
        new_val = m.group(2).strip().rstrip('.,;')
        old_val = m.group(3).strip().rstrip('.,;')
        canon_tag = _MT799_TAG_TO_FIELD.get(raw_tag, raw_tag)
        _record(canon_tag, new_val, old_val)

    # ── Pattern 2: "<TAG>: <label> SHOULD READ AS X I/O Y" ──
    # Same dual quoted/unquoted handling.
    pat_colon_quoted = re.compile(
        r'(?:^|\n)\s*(\d{2}[A-Z]?)\s*:[^\n]{0,80}?'
        r'(?:SHOULD|SHALL|NOW|TO)\s+READ\s+AS'
        r'[\s\n\r]*'
        r'[\'"]\s*([^\'"\n\r]+?)\s*[\'"]'
        r'\s*(?:I\s*/\s*O|INSTEAD\s+OF)\s*'
        r'[\'"]\s*([^\'"\n\r]+?)\s*[\'"]',
        re.IGNORECASE,
    )
    for m in pat_colon_quoted.finditer(norm):
        raw_tag = m.group(1).upper().strip()
        new_val = m.group(2).strip()
        old_val = m.group(3).strip()
        canon_tag = _MT799_TAG_TO_FIELD.get(raw_tag, raw_tag)
        _record(canon_tag, new_val, old_val)

    pat_colon_unquoted = re.compile(
        r'(?:^|\n)\s*(\d{2}[A-Z]?)\s*:[^\n]{0,80}?'
        r'(?:SHOULD|SHALL|NOW|TO)\s+READ\s+AS\s+'
        r'([^\n\r\'"]{1,200}?)'
        r'\s+(?:I\s*/\s*O|INSTEAD\s+OF)\s+'
        r'([^\n\r\'"]{1,200}?)'
        r'(?=\s*(?:[\n\r]|$|REGARDS\b|THANKS\b))',
        re.IGNORECASE,
    )
    for m in pat_colon_unquoted.finditer(norm):
        raw_tag = m.group(1).upper().strip()
        new_val = m.group(2).strip().rstrip('.,;')
        old_val = m.group(3).strip().rstrip('.,;')
        canon_tag = _MT799_TAG_TO_FIELD.get(raw_tag, raw_tag)
        _record(canon_tag, new_val, old_val)

    # Pattern 3: amount-only "AMOUNT INCREASED/DECREASED BY <CCY> <NUM>"
    if '32B' not in seen_tags:
        m = re.search(
            r'(?:AMOUNT|VALUE)\s+(INCREASED|DECREASED|REDUCED|RAISED)\s+BY\s+'
            r'([A-Z]{3})\s*([\d,]+(?:\.\d+)?)',
            norm, re.IGNORECASE,
        )
        if m:
            op = m.group(1).upper()
            ccy = m.group(2).upper()
            amt = m.group(3)
            verb = 'INCREASED' if op in ('INCREASED', 'RAISED') else 'DECREASED'
            out.append(SwiftField(
                tag='32B',
                label=SWIFT_FIELD_LABELS['32B'],
                value=f"{verb} BY {ccy} {amt}",
                source_page=source_page,
                source_mt=source_mt,
            ))
            seen_tags.add('32B')

    # Pattern 4: simple "PLEASE AMEND <description> TO READ AS X I/O Y"
    # Lands in 47A (Additional Conditions) since no specific tag is given.
    if '47A' not in seen_tags:
        m = re.search(
            r'PLEASE\s+(?:AMEND|CHANGE|CORRECT|REPLACE)\s+(.{5,80}?)\s+'
            r'TO\s+READ\s+AS[\s\n\r]*["\']?\s*([^\'"\n\r]+?)\s*["\']?\s*'
            r'(?:I\s*/\s*O|INSTEAD\s+OF)\s*["\']?\s*([^\'"\n\r]+?)\s*["\']?'
            r'(?=\s*(?:[\.\n\r]|$))',
            norm, re.IGNORECASE,
        )
        if m:
            desc = m.group(1).strip()
            new_val = m.group(2).strip().strip("'\"")
            old_val = m.group(3).strip().strip("'\"")
            out.append(SwiftField(
                tag='47A',
                label=SWIFT_FIELD_LABELS['47A'],
                value=f"{desc}: TO READ AS '{new_val}' INSTEAD OF '{old_val}'",
                source_page=source_page,
                source_mt=source_mt,
            ))

    return out


def _detect_format_from_text(text: str) -> str:
    """Detect SWIFT format from GLM text content."""
    fusion_count = len(re.findall(r'\bF\d{2}[A-Z]?\s*:', text))
    alliance_count = len(re.findall(r':\d{2}[A-Z]?:', text))
    # Bare fusion: "20: Documentary Credit Number" pattern (number colon space label)
    bare_fusion_count = len(re.findall(r'(?:^|\n)\s*\d{2}[A-Z]?:\s+[A-Z]', text))
    if fusion_count > alliance_count and fusion_count > bare_fusion_count:
        return 'fusion'
    elif alliance_count > 0 and alliance_count >= bare_fusion_count:
        return 'alliance'
    elif bare_fusion_count > 0:
        return 'bare_fusion'
    return 'unknown'


# == Clause splitting =========================================================

def _split_into_clauses(tag: str, text: str) -> List[Clause]:
    """
    Split a multi-clause SWIFT field into individual clauses.

    Tries splitting formats in order:
    1. Numbered: "1. ..." "2. ..."
    2. Lettered: "A. ..." "B. ..."
    3. Dashed/bulleted: "- ..." "- ..."
    4. Line-by-line grouping
    5. Fallback: entire text as one clause
    """
    if not text or not text.strip():
        return []

    # Handle list input (from JSON deserialization)
    if isinstance(text, list):
        text = '\n'.join(str(item) for item in text)

    clauses = []
    text = text.strip()

    # F45A: split if it has numbered sub-items (1-, 2-, 1., 2.), otherwise keep as one
    if tag in ('45A', 'F45A'):
        # Check if it has numbered sub-items like "1-", "2-", "1.", "2)"
        if not re.search(r'(?:^|\n)\s*\d+\s*[-.)]\s+', text):
            return [Clause(clause_number=1, clause_id=f"{tag}-1", text=text, parent_tag=tag)]
        # Fall through to normal splitting

    # Try numbered: "1.", "2.", "1)", "2)", "1-", "2-"
    # P77/P80: Also handles "7.THE" (no space after the period) and
    # mid-line clause starts like "...EXPIRY. 7.THE CARRIER..." where
    # there is no newline before the clause number. We pre-normalize
    # the text by inserting \n before any mid-line "N." / "N)" / "N-"
    # pattern that follows a sentence-ending period or whitespace.
    _normalized = text
    # Insert \n before mid-line numbered clause starts:
    # Match: (sentence-end punctuation + space(s)) + (digit(s)) + (clause delimiter)
    # e.g. "EXPIRY. 7.THE" → "EXPIRY.\n7.THE"
    # e.g. "SEPARATELY. 10. SOME" → "SEPARATELY.\n10. SOME"
    # P90: Added (?!\d) after the delimiter to exclude dates like
    # "DATED: 07-01-2025" where "07-" looks like a clause number
    # but is actually a date. A real clause number is followed by
    # a letter (the clause text), not another digit.
    # P198au — Broaden the lookbehind to include whitespace, comma,
    # and digit-end (so "NTN NO. 3075811-4 2)" also gets split).
    # Guards: (?!\d) after the delimiter still excludes dates like
    # "07-01-2025"; and the uppercase/"(" lookahead still rejects
    # matches whose RHS is not a clause-body start.
    _normalized = re.sub(
        r'(?<=[\s.;:!?,])(\d{1,2})\s*([.\-)])\s*(?=[A-Z\(])',
        r'\n\1\2 ',
        _normalized,
    )
    # P90: The split regex must NOT match dates like "07-01-2025" or
    # "01.2025" that start a line after a colon ("DATED:\n07-01-2025").
    # A clause number is followed by text content, not by more digits.
    # Use a negative lookahead (?!\d) to exclude digit-after-delimiter.
    numbered = re.split(r'\n\s*(\d{1,2})\s*[-.)]\s*(?!\d)', '\n' + _normalized)
    if len(numbered) >= 3:
        for i in range(1, len(numbered) - 1, 2):
            clause_text = numbered[i + 1].strip()
            if clause_text:
                clauses.append(Clause(
                    clause_number=len(clauses) + 1,
                    clause_id=f"{tag}-{len(clauses) + 1}",
                    text=clause_text,
                    parent_tag=tag,
                ))
        if clauses:
            return clauses

    # Try lettered: "A.", "B.", "A)", "B)"
    lettered = re.split(r'\n\s*([A-Z])\s*[.)]\s+', '\n' + text)
    if len(lettered) >= 3:
        for i in range(1, len(lettered) - 1, 2):
            clause_text = lettered[i + 1].strip()
            if clause_text:
                clauses.append(Clause(
                    clause_number=len(clauses) + 1,
                    clause_id=f"{tag}-{len(clauses) + 1}",
                    text=clause_text,
                    parent_tag=tag,
                ))
        if clauses:
            return clauses

    # Try dash/bullet
    dashed = re.split(r'\n\s*[-+*]\s+', '\n' + text)
    if len(dashed) >= 3:
        for part in dashed:
            part = part.strip()
            if part and len(part) > 10:
                clauses.append(Clause(
                    clause_number=len(clauses) + 1,
                    clause_id=f"{tag}-{len(clauses) + 1}",
                    text=part,
                    parent_tag=tag,
                ))
        if clauses:
            return clauses

    # Try line-by-line grouping
    lines = [ln.strip() for ln in text.split('\n') if ln.strip()]
    if len(lines) >= 2:
        grouped = []
        current = []
        for ln in lines:
            # New clause starts with uppercase, is long enough, and not a continuation word
            if current and (
                re.match(r'^[A-Z0-9]', ln)
                and len(ln) > 20
                and not ln.startswith(('AND ', 'OR ', 'THE ', 'TO ', 'IN ', 'OF ', 'AT ', 'BY '))
            ):
                grouped.append('\n'.join(current))
                current = [ln]
            else:
                current.append(ln)
        if current:
            grouped.append('\n'.join(current))

        if len(grouped) >= 2:
            for part in grouped:
                part = part.strip()
                if part:
                    clauses.append(Clause(
                        clause_number=len(clauses) + 1,
                        clause_id=f"{tag}-{len(clauses) + 1}",
                        text=part,
                        parent_tag=tag,
                    ))
            return clauses

    # Fallback: entire text as one clause
    clauses.append(Clause(
        clause_number=1,
        clause_id=f"{tag}-1",
        text=text,
        parent_tag=tag,
    ))
    return clauses


# == Amendment text operations ================================================

def _extract_swift_ops(amd_text: str) -> list:
    """P198du — Parse SWIFT amendment operation tokens out of an
    amendment value so the audit log can carry an explicit list of
    what each amendment did. Surfaces tokens like /DELETE/, /ADD/,
    /REPALL/, /DEL/ together with the target clause numbers when
    present (e.g. 'CLAUSE 6', 'CLAUSE NO. 7'). The tooltip in the
    final-LC viewer reads these and shows e.g. 'DELETE Clause 6'."""
    ops = []
    if not amd_text:
        return ops
    t = str(amd_text)
    # Normalize fancy quotes
    t = (t.replace('‘', "'").replace('’', "'")
          .replace('“', '"').replace('”', '"')
          .replace("''", '"'))

    def _clause_targets(scope: str):
        return [m.group(1) for m in
                re.finditer(r'CLAUSE\s+(?:NO\.?\s+)?(\d+)',
                             scope, re.IGNORECASE)]

    # /DELETE/ or /DEL/ — typically followed by CLAUSE N or
    # 'PLEASE READ WORDS IN FIELD ... AS DELETED'
    for m in re.finditer(r'/DEL(?:ETE)?/([^/]*?)(?=/[A-Z]+/|$)',
                          t, re.IGNORECASE | re.DOTALL):
        scope = m.group(1) or ''
        targets = _clause_targets(scope)
        if targets:
            for tgt in targets:
                ops.append({'op': 'DELETE', 'target': f'Clause {tgt}'})
        else:
            # Try to capture a quoted phrase to delete
            qm = re.search(r"['\"]([^'\"]+)['\"]\s*AS\s+DELETED",
                           scope, re.IGNORECASE)
            if qm:
                ops.append({'op': 'DELETE',
                            'target': f"Words: {qm.group(1).strip()}"})
            else:
                ops.append({'op': 'DELETE'})

    # /ADD/ — typically followed by CLAUSE N or new clause text
    for m in re.finditer(r'/ADD/([^/]*?)(?=/[A-Z]+/|$)',
                          t, re.IGNORECASE | re.DOTALL):
        scope = m.group(1) or ''
        targets = _clause_targets(scope)
        if targets:
            for tgt in targets:
                ops.append({'op': 'ADD', 'target': f'Clause {tgt}'})
        else:
            ops.append({'op': 'ADD'})

    # /REPALL/ — full-field replacement
    if re.search(r'/REPALL/', t, re.IGNORECASE):
        ops.append({'op': 'REPLACE-ALL'})

    # "TO READ AS '<new>' INSTEAD OF '<old>'" — narrative-form replacement
    for m in re.finditer(
        r'(?:CLAUSE\s+(?:NO\.?\s+)?(\d+)\s+)?'
        r'TO\s+READ\s+AS\s*[\'"]([^\'"]+)[\'"]'
        r'(?:\s*(?:I/?O|INSTEAD\s+OF)\s*[\'"]([^\'"]+)[\'"])?',
        t, re.IGNORECASE | re.DOTALL):
        cl = m.group(1)
        ops.append({
            'op': 'REPLACE',
            'target': f'Clause {cl}' if cl else None,
        })

    # De-dupe while preserving order
    seen = set()
    out = []
    for o in ops:
        key = (o.get('op'), o.get('target'))
        if key in seen:
            continue
        seen.add(key)
        out.append(o)
    return out


def _apply_text_amendment(base_text: str, amendment_text: str) -> str:
    """
    Apply amendment operations to base field text.

    Handles ALL known amendment formats:

    FUSION FORMAT:
      /ADD/+) CLAUSE NO. 10 TO READ AS "new" INSTEAD OF "old"
      /ADD/+) CLAUSE NO. 27 TO READ AS "new text"
      /ADD/+) DELETE ''old'' REPLACE BY ''new''
      /ADD/+) COAUSE NO.5: DELETE ''30 DAYS'' REPLACE BY ''45 DAYS''
      /DEL/) CLAUSE NO. 3

    ALLIANCE FORMAT:
      /DELETE/ PLEASE READ WORDS IN FIELD 46A-1 "text" AS DELETED
      /REPALL/ PLEASE READ FIELD 47A-11 AS "new text"
      /ADD/ PLEASE READ FIELD 47A-19 AS "new clause text"

    If no patterns match, return base text unchanged (don't corrupt it).
    """
    if not base_text:
        return base_text

    result = base_text
    # Normalize all quote types to regular double quote for matching
    amd = amendment_text
    amd = amd.replace('\u2018', "'").replace('\u2019', "'")  # smart single quotes
    amd = amd.replace('\u201C', '"').replace('\u201D', '"')  # smart double quotes
    amd = amd.replace("''", '"')  # two single quotes = one double quote

    # Quote pattern: matches any combination of single/double quotes
    Q = r"""['"]+"""

    # ── FULL REPLACEMENT ──
    # If amendment is just /REPALL/ followed by unquoted text, replace everything
    _repall_m = re.match(r'\s*/REPALL/\s*(.+)', amd, re.IGNORECASE | re.DOTALL)
    if _repall_m:
        new_content = _repall_m.group(1).strip()
        # Clean Narrative: prefixes from Alliance format
        new_content = re.sub(r'(?:Narrative\d?:\s*)+', '', new_content).strip()
        new_content = re.sub(r'\n\s*Narrative\d?:\s*', '\n', new_content).strip()
        new_content = re.sub(r'^Lines?\s*\d*(?:\s*-\s*\d+)?(?:\s*to\s*\d+)?\s*:?\s*$', '', new_content, flags=re.MULTILINE).strip()
        new_content = re.sub(r'^\s*Line\s+\d+\s*$', '', new_content, flags=re.MULTILINE).strip()
        new_content = re.sub(r'^\s*Code\s*[-:]?\s*(?:/REPALL/)?\s*$', '', new_content, flags=re.MULTILINE).strip()
        new_content = re.sub(r'^\s*Code\s*-\s*Narrative\s*$', '', new_content, flags=re.MULTILINE).strip()
        # P87: Detect structured amendment instructions inside /REPALL/ blocks.
        # Alliance MT707 often puts instructions like:
        #   "FIELD 47A-1 TO READ AS ..."
        #   "FIELD 47A-2 DELETE WORDING AS ..."
        #   "UNDER FIELD 47A ADD CLAUSE AS ..."
        # inside a /REPALL/ block. These are NOT replacement text — they're
        # operations that modify the existing base field.
        _has_field_instructions = bool(re.search(
            r'(?:^|\n)\s*(?:UNDER\s+)?FIELD\s+\d{2}[A-Z]?(?:-\d+)?\s+'
            r'(?:TO\s+READ\s+AS|WORD\s+TO\s+READ\s+AS|DELETE\s+WORDING|ADD\s+(?:CLAUSE|LOI|WORDING))',
            new_content, re.IGNORECASE,
        ))

        # P102: Handle "UNDER FIELD XXA, NOW TO BE READ AS, 'new content'"
        # This is a full replacement where the new content is quoted after
        # "NOW TO BE READ AS" / "TO BE READ AS" / "TO READ AS".
        _read_as_m = re.search(
            r'(?:NOW\s+)?TO\s+(?:BE\s+)?READ\s+AS\s*,?\s*' + Q + r'(.+?)' + Q + r'\s*(?:I/?O\s|$)',
            new_content, re.IGNORECASE | re.DOTALL,
        )
        if not _read_as_m:
            # Try without closing quote (content runs to end)
            _read_as_m = re.search(
                r'(?:NOW\s+)?TO\s+(?:BE\s+)?READ\s+AS\s*,?\s*' + Q + r'(.+)',
                new_content, re.IGNORECASE | re.DOTALL,
            )
        if _read_as_m:
            _extracted = _read_as_m.group(1).strip().rstrip("'\"")
            if _extracted:
                result = _extracted

        # P102: Handle "UNDER FIELD XX ADD <thing> AS 'value'" inside REPALL
        # This appends the quoted value to the base text.
        if not _read_as_m:
            _add_as_m = re.search(
                r'UNDER\s+FIELD\s+\d{2}[A-Z]?\s+ADD\s+(.+?)\s+AS\s*\n?\s*' + Q + r'(.+?)' + Q,
                new_content, re.IGNORECASE | re.DOTALL,
            )
            if _add_as_m:
                _add_label = _add_as_m.group(1).strip()
                _add_value = _add_as_m.group(2).strip()
                if _add_value and _add_value not in result:
                    result = result.rstrip() + '\nAND ADD ' + _add_label + ' ' + _add_value

        # If it contains PLEASE READ patterns or FIELD instruction patterns,
        # don't do full replace — let patterns handle it below
        elif not re.search(r'PLEASE\s+READ', new_content, re.IGNORECASE) and not _has_field_instructions:
            if new_content:
                result = new_content
                # Still process remaining patterns (there might be /ADD/ blocks after)

    # ── ALLIANCE FORMAT ──

    # A1: /DELETE/ PLEASE READ WORDS IN FIELD XX-N "text" AS DELETED
    for m in re.finditer(
            r'PLEASE\s+READ\s+WORDS\s+IN\s+FIELD\s+\w+-?\d*\s+' + Q + r'(.+?)' + Q + r'\s+AS\s+DELETED',
            amd, re.IGNORECASE | re.DOTALL):
        del_text = m.group(1).strip()
        if del_text in result:
            result = result.replace(del_text, '')
            # Clean up double spaces / empty lines left behind
            result = re.sub(r'  +', ' ', result)
            result = re.sub(r'\n\s*\n\s*\n', '\n\n', result)

    # A2: /REPALL/ or /ADD/ PLEASE READ FIELD XX-N AS "new text"
    for m in re.finditer(
            r'PLEASE\s+READ\s+FIELD\s+\w+-(\d+)\s+AS\s+' + Q + r'(.+?)' + Q,
            amd, re.IGNORECASE | re.DOTALL):
        clause_num = int(m.group(1))
        new_text = m.group(2).strip()
        pre_text = amd[:m.start()].upper()
        is_repall = '/REPALL/' in pre_text
        is_add = '/ADD/' in pre_text and '/REPALL/' not in pre_text

        if is_repall:
            clause_pat = re.compile(
                r'(\(' + str(clause_num) + r'\)|' + str(clause_num) + r'[\).:])\s*(.+?)(?=\n\s*(?:\(\d+\)|\d+[\).:])\s|\Z)',
                re.DOTALL)
            cm = clause_pat.search(result)
            if cm:
                result = result[:cm.start(2)] + new_text + result[cm.end(2):]
        elif is_add:
            clause_line = f"\n{clause_num}.{new_text}"
            if new_text not in result:
                result = result.rstrip() + clause_line

    # ── FUSION FORMAT ──

    # F1: DELETE ''old'' REPLACE BY ''new''
    for m in re.finditer(r'DELETE\s+' + Q + r'(.+?)' + Q + r'\s+REPLACE\s+BY\s+' + Q + r'(.+?)' + Q,
                         amd, re.IGNORECASE | re.DOTALL):
        old_text = m.group(1).strip()
        new_text = m.group(2).strip()
        if old_text in result:
            result = result.replace(old_text, new_text)

    # F1b: bare "TO READ AS 'new' INSTEAD OF 'old'" (no clause-number prefix).
    # Produced by _extract_mt799_amendment_fields() for MT799 free-format
    # rate / value corrections like:
    #     UNDER FIELD 45A RATE SHOULD READ AS
    #     ''EUR 141,396.00'' I/O ''EUR 141,396.56''
    # The parser converts this to: TO READ AS 'EUR 141,396.00' INSTEAD OF 'EUR 141,396.56'
    # which we apply as a targeted substring replace inside the base field
    # — leaving the rest of the goods description intact.
    #
    # Real-world wrinkle: the amendment usually quotes the value with its
    # currency prefix ("EUR 141,396.56"), but the base F45A may have the
    # currency on a different line ("EURO\n141,396.56"). So an exact-match
    # replace fails. We try several candidate forms in priority order:
    #   1. Exact text from the amendment
    #   2. Without the leading currency code (3-letter or "EURO" word)
    #   3. Just the bare numeric value (digits + grouping + decimals)
    # The first form that's actually present in the base field wins.
    def _replace_old_with_new(base: str, old_text: str, new_text: str) -> str:
        if not old_text:
            return base
        candidates_old = [old_text]
        candidates_new = [new_text]
        # Strip leading currency code (USD, EUR, GBP, etc. or "EURO")
        _strip_ccy = re.compile(
            r'^(?:[A-Z]{3}|EURO|DOLLAR|POUND|YEN)\s+',
            re.IGNORECASE,
        )
        _bare_old = _strip_ccy.sub('', old_text).strip()
        _bare_new = _strip_ccy.sub('', new_text).strip()
        if _bare_old != old_text:
            candidates_old.append(_bare_old)
            candidates_new.append(_bare_new)
        # Just the numeric part — last resort, only when both old and new
        # parse as numbers. This catches "EUR 141,396.56" → "141,396.56".
        _num_re = re.compile(r'[\d,]+(?:\.\d+)?')
        _num_old = _num_re.search(old_text)
        _num_new = _num_re.search(new_text)
        if _num_old and _num_new:
            candidates_old.append(_num_old.group(0))
            candidates_new.append(_num_new.group(0))
        for _co, _cn in zip(candidates_old, candidates_new):
            if _co and _co in base:
                return base.replace(_co, _cn)
        return base

    for m in re.finditer(
        r'TO\s+READ\s+AS\s+' + Q + r'(.+?)' + Q + r'\s+INSTEAD\s+OF\s+' + Q + r'(.+?)' + Q,
        amd, re.IGNORECASE | re.DOTALL,
    ):
        new_text = m.group(1).strip()
        old_text = m.group(2).strip()
        result = _replace_old_with_new(result, old_text, new_text)

    # F2a: CLAUSE NO. X TO READ AS "new" INSTEAD OF "old"
    for m in re.finditer(r'C\w{1,5}E\s+NO\.?\s*(\d+)\s+(?:NOW\s+)?TO\s+READ\s+AS\s+' + Q + r'(.+?)' + Q + r'\s+INSTEAD\s+OF\s+' + Q + r'(.+?)' + Q,
                         amd, re.IGNORECASE | re.DOTALL):
        old_text = m.group(3).strip()
        new_text = m.group(2).strip()
        if old_text in result:
            result = result.replace(old_text, new_text)

    # F2b: CLAUSE NO. X TO READ AS "new text" (no INSTEAD OF)
    for m in re.finditer(r'C\w{1,5}E\s+NO\.?\s*(\d+)\s+(?:NOW\s+)?TO\s+READ\s+AS\s+' + Q + r'(.+?)' + Q,
                         amd, re.IGNORECASE | re.DOTALL):
        full = m.group(0)
        if re.search(r'INSTEAD\s+OF', full, re.IGNORECASE):
            continue
        clause_num = int(m.group(1))
        new_text = m.group(2).strip()
        clause_pat = re.compile(
            r'(\(' + str(clause_num) + r'\)|' + str(clause_num) + r'[\).:])\s*(.+?)(?=\n\s*(?:\(\d+\)|\d+[\).:])\s|\Z)',
            re.DOTALL)
        cm = clause_pat.search(result)
        if cm:
            result = result[:cm.start(2)] + new_text + result[cm.end(2):]

    # F3: /ADD/ new clause text (generic — append if not an instruction)
    for m in re.finditer(r'/ADD/\s*\+?\s*\)?\s*(.+?)(?=\n\s*/(?:ADD|DEL|REPALL)/|\Z)',
                         amd, re.IGNORECASE | re.DOTALL):
        add_text = m.group(1).strip()
        # Skip if it's an instruction already handled above
        if re.search(r'C\w{1,5}E\s+NO', add_text, re.IGNORECASE):
            continue
        if re.search(r'DELETE.*REPLACE', add_text, re.IGNORECASE):
            continue
        if re.search(r'PLEASE\s+READ\s+FIELD', add_text, re.IGNORECASE):
            continue
        if re.search(r'PLEASE\s+READ\s+WORDS', add_text, re.IGNORECASE):
            continue
        if re.search(r'PLEASE\s+READ\s+C\w{1,5}E', add_text, re.IGNORECASE):
            continue
        if re.search(r'TO\s+READ\s+AS', add_text, re.IGNORECASE):
            continue
        # Append as new content
        if add_text and add_text not in result:
            result = result.rstrip() + '\n' + add_text

    # ── ADDITIONAL PATTERNS ──

    # P1: /ADD/+)TO READ AS "new text" (no clause number — replaces entire field, e.g. 45A goods)
    for m in re.finditer(r'/ADD/\s*\+?\s*\)?\s*TO\s+READ\s+AS\s+' + Q + r'(.+?)' + Q,
                         amd, re.IGNORECASE | re.DOTALL):
        new_text = m.group(1).strip()
        # This replaces a portion of the field text — check if there's a DELETE..REPLACE nearby
        if not re.search(r'DELETE.*REPLACE', amd[max(0,m.start()-50):m.start()], re.IGNORECASE):
            # Full field replacement
            if new_text and new_text not in result:
                result = new_text

    # P2: PLEASE READ CLAUSE XX-N AS "new text" (Alliance, uses CLAUSE instead of FIELD)
    for m in re.finditer(r'PLEASE\s+READ\s+C\w{1,5}E\s+\w+-(\d+)\s+AS\s+' + Q + r'(.+?)' + Q,
                         amd, re.IGNORECASE | re.DOTALL):
        clause_num = int(m.group(1))
        new_text = m.group(2).strip()
        clause_pat = re.compile(
            r'(\(' + str(clause_num) + r'\)|' + str(clause_num) + r'[\).:])\s*(.+?)(?=\n\s*(?:\(\d+\)|\d+[\).:])\s|\Z)',
            re.DOTALL)
        cm = clause_pat.search(result)
        if cm:
            result = result[:cm.start(2)] + new_text + result[cm.end(2):]

    # P3: PLEASE READ FIELD XXX AS "new text" (no clause number — replaces entire field)
    for m in re.finditer(r'PLEASE\s+READ\s+FIELD\s+\w+\s+AS\s+' + Q + r'(.+?)' + Q,
                         amd, re.IGNORECASE | re.DOTALL):
        # Only if no clause number (already handled by A2)
        if not re.search(r'FIELD\s+\w+-\d+', m.group(0)):
            new_text = m.group(1).strip()
            if new_text:
                result = new_text

    # P4: IN FIELD XXX: ADD CLAUSE NO.N "text" (PPL6 style)
    for m in re.finditer(r'IN\s+FIELD\s+\w+:\s*ADD\s+C\w{1,5}E\s+NO\.?\s*(\d+)',
                         amd, re.IGNORECASE):
        clause_num = int(m.group(1))
        # Get remaining text after the match as the clause content
        remaining = amd[m.end():].strip()
        if remaining and remaining not in result:
            result = result.rstrip() + f'\n{clause_num}.{remaining}'

    # P5: /DELETE/ CLAUSE 2,3,5,7 — delete multiple clauses by number
    for m in re.finditer(r'C\w{1,5}E\s+([\d,\s]+)', amd):
        pre = amd[:m.start()].upper()
        if '/DELETE/' not in pre[max(0,pre.rfind('/')-10):]:
            continue
        nums = [int(n.strip()) for n in m.group(1).split(',') if n.strip().isdigit()]
        for clause_num in nums:
            # Remove the clause from base text
            clause_pat = re.compile(
                r'(\(' + str(clause_num) + r'\)|' + str(clause_num) + r'[\).:])\s*(.+?)(?=\n\s*(?:\(\d+\)|\d+[\).:])\s|\Z)',
                re.DOTALL)
            result = clause_pat.sub('', result)
        result = re.sub(r'\n\s*\n\s*\n', '\n\n', result)

    # P6: /ADD/ CLAUSE N)text (direct clause add, no quotes)
    for m in re.finditer(r'C\w{1,5}E\s+(\d+)\)\s*(.+?)(?=\n\s*C\w{1,5}E\s+\d+\)|\n\s*Code:|\Z)',
                         amd, re.IGNORECASE | re.DOTALL):
        pre = amd[:m.start()].upper()
        if '/ADD/' not in pre[max(0,len(pre)-100):]:
            continue
        clause_num = int(m.group(1))
        new_text = m.group(2).strip()
        # Clean up Narrative: prefixes from Alliance format
        new_text = re.sub(r'(?:Narrative\d?:\s*)+', '', new_text).strip()
        new_text = re.sub(r'\n\s*Narrative\d?:\s*', '\n', new_text).strip()
        if new_text and new_text not in result:
            result = result.rstrip() + f'\n{clause_num}){new_text}'

    # P7: IN FIELD XX: FOR EXISTING PLEASE READ "value" (replace field value)
    for m in re.finditer(r'IN\s+FIELD\s+\w+:\s*FOR\s+EXISTING\s+PLEASE\s+READ\s+' + Q + r'(.+?)' + Q,
                         amd, re.IGNORECASE | re.DOTALL):
        new_val = m.group(1).strip()
        if new_val:
            result = new_val

    # ── P87: ALLIANCE FIELD-INSTRUCTION FORMAT ──
    # These appear inside /REPALL/ blocks as structured operations:
    #
    #   "FIELD 47A-1 TO READ AS\n<quoted or unquoted new clause text>"
    #   "FIELD 47A-2 DELETE WORDING AS\n<quoted text to delete>"
    #   "UNDER FIELD 47A ADD CLAUSE AS\n<quoted text for new clause(s)>"
    #   "UNDER FIELD 46A ADD LOI CLAUSE AS\n<text>"
    #   "UNDER FIELD 46A-2 WORD TO READ AS\n<quoted new> I/O <quoted old>"

    Q = r"""['"]+"""  # re-define for this section

    # I1: FIELD XX-N TO READ AS "new clause text"
    # Replaces clause N entirely with the new text.
    for m in re.finditer(
        r'FIELD\s+\d{2}[A-Z]?-(\d+)\s+(?:NOW\s+)?TO\s+READ\s+AS\s*\n?\s*' + Q + r'(.+?)' + Q,
        amd, re.IGNORECASE | re.DOTALL,
    ):
        clause_num = int(m.group(1))
        new_text = m.group(2).strip()
        # Find and replace clause N in the result
        clause_pat = re.compile(
            r'(' + str(clause_num) + r'[\).:\s])\s*(.+?)(?=\n\s*(?:\d+[\).:])\s|\Z)',
            re.DOTALL,
        )
        cm = clause_pat.search(result)
        if cm:
            result = result[:cm.start(2)] + ' ' + new_text + result[cm.end(2):]

    # I2: FIELD XX-N DELETE WORDING AS "text to delete"
    # Deletes the quoted text from the field.
    for m in re.finditer(
        r'FIELD\s+\d{2}[A-Z]?-?(\d*)\s+DELETE\s+WORDING\s+AS\s*\n?\s*' + Q + r'(.+?)' + Q,
        amd, re.IGNORECASE | re.DOTALL,
    ):
        del_text = m.group(2).strip()
        if del_text and del_text in result:
            result = result.replace(del_text, '')
            result = re.sub(r'  +', ' ', result)
            result = re.sub(r'\n\s*\n\s*\n', '\n\n', result)

    # I2b: [UNDER] FIELD XX-N WORD TO READ AS "new" I/O "old"
    for m in re.finditer(
        r'(?:UNDER\s+)?FIELD\s+\d{2}[A-Z]?-?(\d*)\s+WORD\s+TO\s+READ\s+AS\s*\n?\s*'
        + Q + r'(.+?)' + Q + r'\s+I/O\s+' + Q + r'(.+?)' + Q,
        amd, re.IGNORECASE | re.DOTALL,
    ):
        new_text = m.group(2).strip()
        old_text = m.group(3).strip()
        if old_text in result:
            result = result.replace(old_text, new_text)

    # I3: UNDER FIELD XX ADD CLAUSE AS "new clause text"
    # or  UNDER FIELD XX ADD LOI CLAUSE AS "text"
    # Appends new clause(s) to the field.
    for m in re.finditer(
        r'UNDER\s+FIELD\s+\d{2}[A-Z]?(?:-\d+)?\s+ADD\s+(?:LOI\s+)?(?:CLAUSE|WORDING)\s+AS\s*\n?\s*'
        + Q + r'?(.+?)(?=' + Q + r'?\s*$|\n\s*(?:FIELD|UNDER)\s+\d{2}[A-Z])',
        amd, re.IGNORECASE | re.DOTALL,
    ):
        add_text = m.group(1).strip()
        # Strip trailing quotes
        add_text = re.sub(r'["\']$', '', add_text).strip()
        if add_text and add_text not in result:
            result = result.rstrip() + '\n' + add_text

    # P102: UNDER FIELD XX ADD <anything> AS "value"
    # e.g. "UNDER FIELD 45A ADD PROFORMA INVOICE NO. AS 'HN/2026/43 DATED 01-01-2026'"
    # This appends the quoted value to the existing field text.
    for m in re.finditer(
        r'UNDER\s+FIELD\s+\d{2}[A-Z]?(?:-\d+)?\s+ADD\s+(.+?)\s+AS\s*\n?\s*'
        + Q + r'(.+?)' + Q,
        amd, re.IGNORECASE | re.DOTALL,
    ):
        add_label = m.group(1).strip()  # e.g. "PROFORMA INVOICE NO."
        add_value = m.group(2).strip()  # e.g. "HN/2026/43 DATED 01-01-2026"
        if add_value:
            # Append as "AND ADD <label> <value>" to the existing field
            append_text = f"AND ADD {add_label} {add_value}"
            if add_value not in result:
                result = result.rstrip() + '\n' + append_text

    return result


# == Amendment application ====================================================

# Label-strip patterns reused by both base extraction and amendment cleanup.
# Each entry strips the leading label that the SWIFT regex captured along
# with the actual value (e.g. "Latest Date of Shipment\n251030 Oct 30").
_FIELD_LABEL_STRIP = {
    '20':  r'^(?:Documentary\s+Credit\s+Number|Sender\'?s?\s+Reference|Transaction\s+Reference\s+Number)\s*[\n\r]*',
    '21':  r'^(?:Related\s+Reference|Receiver\'?s?\s+Reference|Reimbursing\s+Bank\'?s?\s+Reference)\s*[\n\r]*',
    '27':  r'^Sequence\s+of\s+Total\s*[\n\r]*',
    '31C': r'^Date\s+of\s+Issue\s*[\n\r]*',
    '31D': r'^Date\s+and\s+Place\s+of\s+Expiry\s*[\n\r]*',
    '32B': r'^(?:Currency\s+Code,?\s*Amount|Increase\s+of\s+Documentary\s+Credit\s+Amount)\s*[\n\r]*',
    '39A': r'^Percentage\s+Credit\s+Amount\s+Tolerance\s*[\n\r]*',
    '40A': r'^Form\s+of\s+Documentary\s+Credit\s*[\n\r]*',
    '40E': r'^Applicable\s+Rules\s*[\n\r]*',
    '41A': r'^(?:Available\s+With.*?(?:By\.{0,3})?|\.{2,}\s*By\s*\.{2,}.*?(?:Name\s+and\s+Address)?:?)\s*[\n\r]*',
    '41D': r'^(?:Available\s+With.*?(?:Code|By\.{0,3})?|\.{2,}\s*By\s*\.{2,}.*?(?:Name\s+and\s+Address)?:?)\s*[\n\r]*',
    '42A': r'^(?:Drawee|Issuing\s+Bank).*?(?:Identifier\s+Code)?\s*[\n\r]*',
    '42D': r'^Drawee[\s\S]*?(?:Name\s+and\s+Address\s*:?\s*[\n\r]*)',
    '42C': r'^Drafts\s+at\s*\.{0,3}\s*[\n\r]*',
    '42P': r'^(?:Negotiation/)?Deferred\s+Payment\s+Details\s*[\n\r]*',
    '43P': r'^Partial\s+Shipments?[\.,;:]?\s*[\n\r]*',
    '43T': r'^Trans[sh]?ipment[\.,;:]?\s*[\n\r]*',
    '44A': r'^Place\s+of\s+Taking.*?\s*[\n\r]*',
    '44C': r'^Latest\s+Date\s+of\s+Shipment\s*[\n\r]*',
    '44E': r'^Port\s+of\s+Loading.*?Departure\s*[\n\r]*',
    '44F': r'^Port\s+of\s+Discharge.*?Destination\s*[\n\r]*',
    '45A': r'^Description\s+of\s+Goods.*?Services?\s*[\n\r]*',
    '46A': r'^Documents?\s+Required\s*[\n\r]*',
    '47A': r'^Additional\s+Conditions\s*[\n\r]*',
    '48':  r'^Period\s+for\s+Presentation.*?Days\s*[\n\r]*',
    '49':  r'^Confirmation\s+Instructions\s*[\n\r]*',
    '50':  r'^Applicant\s*[\n\r]*',
    '51A': r'^Applicant\s+Bank.*?(?:Identifier\s+Code)?\s*[\n\r]*',
    '51D': r'^(?:Applicant\s+Bank|Bank)\s*-?\s*(?:Party)?.*?(?:Name\s+and\s+Address)?:?\s*[\n\r]*',
    '52A': r'^(?:Issuing\s+Bank|Applicant\s+Bank).*?(?:Identifier\s+Code)?\s*[\n\r]*',
    '53A': r'^Reimbursing\s+Bank.*?(?:Identifier\s+Code)?\s*[\n\r]*',
    '57A': r'^[\'"]?Advise\s+Through[\'"]?\s+Bank.*?(?:Identifier\s+Code\s*:?\s*)?\s*[\n\r]*',
    '59':  r'^Beneficiary\s*[\n\r]*(?:Name\s+and\s+Address:?\s*[\n\r]*)?',
    '71D': r'^Charges\s*[\n\r]*',
    '78':  r'^Instructions\s+to\s+the\s+Paying.*?Bank\s*[\n\r]*',
}


def _clean_consolidated_field_value(tag: str, value: str) -> str:
    """
    Apply the FULL cleanup pipeline used by the base-field extraction loop
    to a single (tag, value) pair. Used for both base values and amendment
    values so the consolidated LC has consistent cleaning.

    Strips:
      • SWIFT label prefixes (per _FIELD_LABEL_STRIP)
      • "- Party Identifier - Identifier Code / Identifier Code:" sub-labels
      • "Name and Address:" headers
      • Fusion "... By ..." availability prefixes
      • Drawee / Issuing Bank / Reimbursing Bank "- Party ... Code" headers
      • "Available With ... Code" prefix (41A/41D)
      • "Applicable Rules" prefix (40E)
      • "Other\\nDelivery overdue / Network delivery / Payment Confirmation"
        SWIFT report footer that bleeds in when a field is the LAST tag
        on the message (typically F44C, F47A, F78)
      • "Report Footer\\nNumber of Entities ... End of Report" PDF footer
      • "(CONT FROM FIELD ...)" cross-references
      • Trailing F-tag merge (e.g. "...F41D: Available With...")
      • "Page X of Y" page numbering
      • OCR garbage from blank/unreadable pages
      • Inline "Date:", "Place:", "Currency:", "Amount:", "Days:",
        "Narrative:", "Number:", "Total:", "Tolerance N:", "Code:" sub-labels
    Converts:
      • SWIFT date format "260131 2026 Jan 31" → "2026-01-31" (for 31C/31D/44C)
      • F32B amount: "USD #516,000.00#" → "USD 516,000.00", European
        format "516000,00" → "516,000.00"
      • Removes raw "260131" 6-digit codes after they've been converted
    """
    if not value:
        return value
    v = value

    # 1. Strip leading SWIFT label per tag
    _strip_pat = _FIELD_LABEL_STRIP.get(tag, '')
    if _strip_pat:
        v = re.sub(_strip_pat, '', v, flags=re.IGNORECASE).strip()

    # 1b. Strip "Days:" and "Narrative:" sub-labels (common in F48, F47A)
    if tag in ('48', '47A', '46A', '45A', '78', '72'):
        v = re.sub(r'(?:^|\n)\s*Days:?\s*', '\n', v).strip()
        v = re.sub(r'(?:^|\n)\s*Narrative:?\s*/?\s*', '\n', v).strip()

    # 2. Sub-label chains: "- Party Identifier - Identifier Code\nIdentifier Code:\n..."
    v = re.sub(r'-?\s*Party\s+Identifier\s*-?\s*Identifier\s*(?:Code)?\s*\n?', '', v, flags=re.IGNORECASE).strip()
    v = re.sub(r'^-\s+', '', v).strip()
    v = re.sub(r'Identifier\s+Code:?\s*\n?', '', v, flags=re.IGNORECASE).strip()
    v = re.sub(r'^Identifier:?\s*\n?', '', v, flags=re.IGNORECASE).strip()
    v = re.sub(r'^Name\s+and\s+Address:?\s*\n?', '', v, flags=re.IGNORECASE).strip()

    # 3. Fusion availability prefix
    v = re.sub(r'\.{2,}\s*By\s*\.{2,}\s*-?\s*(?:Name\s+and\s+Address\s*-?\s*)*:?\s*[\n\r]*',
               '', v, flags=re.IGNORECASE).strip()
    v = re.sub(r'-\s*Name\s+and\s+Address\s*-?\s*(?:Name\s+and\s+Address)?:?\s*[\n\r]*',
               '', v, flags=re.IGNORECASE).strip()

    # 3b. Strip amendment instruction wrappers from field values
    # After _apply_text_amendment, 45A may still contain:
    #   "UNDER FIELD 45A, NOW TO BE READ AS, '..." and "...'' I/O EXISTING"
    # Strip these instruction wrappers to get the actual content.
    v = re.sub(
        r'^(?:UNDER\s+)?FIELD\s+\d{2}[A-Z]?\s*,?\s*(?:NOW\s+)?TO\s+(?:BE\s+)?READ\s+AS\s*,?\s*[\'"]?\s*',
        '', v, flags=re.IGNORECASE).strip()
    # Strip trailing "I/O EXISTING" or "I/O <old text>" (may appear mid-line or at end)
    v = re.sub(r'[\'"]?\s*I\s*/?\s*O\s+EXISTING\s*', '', v, flags=re.IGNORECASE).strip()
    v = re.sub(r'[\'"]?\s*I/O\s+[\'"].*?[\'"]', '', v, flags=re.IGNORECASE).strip()

    # 4. Currency-name strip (32B)
    if tag == '32B':
        v = re.sub(r'\b(?:US\s+DOLLAR|EURO|POUND\s+STERLING|JAPANESE\s+YEN)\s*[\n\r]*',
                   '', v, flags=re.IGNORECASE).strip()

    # 5. Other prefix headers
    v = re.sub(r'^Applicable\s+Rules:?\s*\n?', '', v, flags=re.IGNORECASE).strip()
    v = re.sub(r'^Available\s+With.*?Code\s*\n?', '', v, flags=re.IGNORECASE).strip()
    v = re.sub(r'^Drawee\s*-?\s*Party.*?Code\s*\n?', '', v, flags=re.IGNORECASE).strip()
    v = re.sub(r'^Applicant\s+Bank\s*-?\s*Party.*?Code\s*\n?', '', v, flags=re.IGNORECASE).strip()
    v = re.sub(r'^Issuing\s+Bank\s*-?\s*Party.*?Code\s*\n?', '', v, flags=re.IGNORECASE).strip()
    v = re.sub(r'^Reimbursing\s+Bank\s*-?\s*Party.*?Code\s*\n?', '', v, flags=re.IGNORECASE).strip()
    v = re.sub(r'^[\'"]?Advise\s+Through[\'"]?\s+Bank\s*-?\s*Party.*?Code\s*\n?',
               '', v, flags=re.IGNORECASE).strip()

    # 6. SWIFT message footer — strip aggressively.
    # "Other\nDelivery overdue..." appears at end of last field on a page.
    # Also catches: "OtherDelivery..." (no newline), "Confirmed Confirmed...",
    # "Page X of Y", "Report Footer", "Message Details #N", etc.
    v = re.sub(r'\s*Other\s*\n?\s*Delivery\s+overdue.*$', '', v, flags=re.IGNORECASE | re.DOTALL).strip()
    v = re.sub(r'\s*Delivery\s+overdue\s+warning.*$', '', v, flags=re.IGNORECASE | re.DOTALL).strip()
    v = re.sub(r'\s*Network\s+delivery\s+notif.*$', '', v, flags=re.IGNORECASE | re.DOTALL).strip()
    v = re.sub(r'\s*Payment\s+Confirmation\s+Status.*$', '', v, flags=re.IGNORECASE | re.DOTALL).strip()
    v = re.sub(r'\s*Confirmed\s+(?:Currency|Amount|Date)\s*:?\s*\n?.*$', '', v, flags=re.IGNORECASE | re.DOTALL).strip()
    # "Confirmed Confirmed Confirmed" (repeated word without label)
    v = re.sub(r'\s*(?:Confirmed\s+){2,}.*$', '', v, flags=re.IGNORECASE | re.DOTALL).strip()
    # Page pagination: "Page N of M"
    v = re.sub(r'\s*Page\s+\d+\s+of\s+\d+\s*', ' ', v).strip()

    # 7. PDF / report footer that follows the SWIFT footer ("Report Footer
    # / Number of Entities / End of Report"). Sometimes the SWIFT footer
    # was already stripped at message-export time, so the PDF footer is
    # the only remaining trailing garbage.
    v = re.sub(
        r'\n\s*Report\s+Footer\b.*$',
        '', v, flags=re.IGNORECASE | re.DOTALL).strip()
    v = re.sub(
        r'\n\s*Number\s+of\s+Entities\b.*$',
        '', v, flags=re.IGNORECASE | re.DOTALL).strip()
    v = re.sub(
        r'\n\s*End\s+of\s+Report\b.*$',
        '', v, flags=re.IGNORECASE | re.DOTALL).strip()

    # 7b. Report structure headers that leak into field values
    v = re.sub(r'\s*Report\s+Content\b.*?(?=\n[A-Z]|\Z)', '', v, flags=re.IGNORECASE | re.DOTALL).strip()
    v = re.sub(r'\s*Message\s+Details\s+#\s*\d+\b.*?(?=\n[A-Z]|\Z)', '', v, flags=re.IGNORECASE | re.DOTALL).strip()
    v = re.sub(r'\s*Message\s+Text\s*\n?', '', v, flags=re.IGNORECASE).strip()
    v = re.sub(r'\s*Block\s+[45]\s*\n?', '', v, flags=re.IGNORECASE).strip()
    v = re.sub(r'\s*Message\s+Preparation\s+Application.*$', '', v, flags=re.IGNORECASE | re.DOTALL).strip()
    v = re.sub(r'\s*Unique\s+Message\s+Identifier.*$', '', v, flags=re.IGNORECASE | re.DOTALL).strip()
    v = re.sub(r'\s*Message\s+(?:Header|Identifier)\s*\n?', '', v, flags=re.IGNORECASE).strip()
    v = re.sub(r'\s*Applic\.?\s+Interface\b.*$', '', v, flags=re.IGNORECASE | re.DOTALL).strip()
    v = re.sub(r'\s*SWIFT\s+Interface\b.*$', '', v, flags=re.IGNORECASE | re.DOTALL).strip()
    v = re.sub(r'\{CHK:[A-F0-9]+\}', '', v).strip()

    # 7c. For 47A: strip F48 presentation period text at the end.
    # Even if the LLM removed the "(CONT FROM FIELD 48)" marker, the
    # continuation text may remain. Strip known patterns.
    if tag == '47A':
        v = re.sub(
            r'\n\s*DOCUMENTS\s+PRESENTED\s+\d+\s+DAYS\s+AFTER\s+BILL\s+OF\s+LADING.*$',
            '', v, flags=re.IGNORECASE | re.DOTALL).strip()
        v = re.sub(
            r'\n\s*\d+\s+DAYS\s+FROM\s+SHIPMENT\s+DATE\s+BUT\s+WITHIN.*$',
            '', v, flags=re.IGNORECASE | re.DOTALL).strip()

    # 8. (CONT FROM/IN FIELD ...) cross-references — P87: also match "IN"
    # P101: Strip the marker AND any trailing continuation text that belongs
    # to the referenced field, not to the current field.
    # e.g. "...\n(CONT. FROM FIELD 48)\nDAYS FROM SHIPMENT DATE..." →
    # the "DAYS FROM..." belongs to F48, not to this field.
    v = re.sub(r'\s*/?\(CONT\.?\s*(?:FROM|IN)\s+FIELD\s+\w+\).*', '', v, flags=re.IGNORECASE | re.DOTALL).strip()
    v = re.sub(r'/\(CONT\.?\s*(?:FROM|IN)\s+FIELD\s+\w+\).*', '', v, flags=re.IGNORECASE | re.DOTALL).strip()

    # 9. Truncate at next F-tag if it merged in
    _ftag_merge = re.search(r'F\d{2}[A-Z]?\s*:', v)
    if _ftag_merge:
        v = v[:_ftag_merge.start()].strip()

    # 10. "Page X of Y" page numbering
    v = re.sub(r'\bPage\s+\d+\s+of\s+\d+\b', '', v, flags=re.IGNORECASE).strip()

    # 11. OCR garbage
    v = re.sub(r'There is no visible text.*?(?:clearly visible|another version)[.\s]*',
               '', v, flags=re.IGNORECASE | re.DOTALL).strip()
    v = re.sub(r'The image appears to be blank[.\s]*',
               '', v, flags=re.IGNORECASE).strip()

    # 11a. Common SWIFT/OCR abbreviation corrections
    v = re.sub(r'\bFRM\b', 'FROM', v)
    v = re.sub(r'\bWTHN\b', 'WITHIN', v)
    v = re.sub(r'\bSHPMNT\b', 'SHIPMENT', v)
    v = re.sub(r'\bDOCS?\b(?=\s+(?:PRESENTED|REQUIRED|MUST))', 'DOCUMENTS', v)

    # 11b. P88: Line-ending hyphen continuation join.
    # SWIFT/OCR text sometimes breaks a word or reference number across
    # lines with a hyphen: "PL-0725-\n201501-M05-002828". Join these
    # so the full reference stays on one line. Only join when the hyphen
    # is at the very end of a line (not mid-line hyphens like "MAI-KOLACHI").
    v = re.sub(r'-\s*\n\s*', '-', v)

    # 12. Inline sub-labels: "Date:", "Place:", "Currency:", "Amount:", etc.
    v = re.sub(r'\bDate:?\s*', '', v).strip()
    v = re.sub(r'\bPlace:?\s*', '', v).strip()
    v = re.sub(r'\bCurrency:?\s*', '', v).strip()
    v = re.sub(r'\bAmount:?\s*', '', v).strip()
    v = re.sub(r'\bDays:?\s*', '', v).strip()
    v = re.sub(r'\bNarrative:?\s*/?', '', v).strip()
    v = re.sub(r'\bNumber:?\s*', '', v).strip()
    v = re.sub(r'\bTotal:?\s*', '', v).strip()
    v = re.sub(r'\bTolerance\s+\d:?\s*', '', v).strip()
    v = re.sub(r'\bCode:?\s*', '', v).strip()

    # 13. SWIFT date conversion (31C, 31D, 44C)
    if tag in ('31C', '31D', '44C'):
        # Format 1: "250103 2025 Jan 03" (Fusion long format) → "2025-01-03"
        _dm = re.search(r'(\d{6})\s+(\d{4})\s+(\w{3})\s+(\d{1,2})', v)
        if _dm:
            _months = {'Jan':'01','Feb':'02','Mar':'03','Apr':'04','May':'05','Jun':'06',
                       'Jul':'07','Aug':'08','Sep':'09','Oct':'10','Nov':'11','Dec':'12'}
            _date_str = f"{_dm.group(2)}-{_months.get(_dm.group(3),'01')}-{int(_dm.group(4)):02d}"
            v = (v[:_dm.start()] + _date_str + v[_dm.end():]).strip()
            # Only strip leftover 6-digit codes AFTER successful conversion
            # (the converted date is now YYYY-MM-DD, remove any duplicate raw code)
            v = re.sub(r'\b\d{6}\b\s*', '', v).strip()
        else:
            # Format 2: "250103" alone (Alliance raw date) → convert to "2025-01-03"
            _raw_date = re.search(r'\b(\d{2})(\d{2})(\d{2})\b', v)
            if _raw_date:
                _yy, _mm, _dd = _raw_date.group(1), _raw_date.group(2), _raw_date.group(3)
                _year = f"20{_yy}" if int(_yy) < 80 else f"19{_yy}"
                _date_str = f"{_year}-{_mm}-{_dd}"
                v = v[:_raw_date.start()] + _date_str + v[_raw_date.end():]
                v = v.strip()

    # 14a. Strip a leading punctuation-only line (e.g. '.\nALLOWED' →
    # 'ALLOWED'). This catches the residue when the SWIFT label was
    # followed by a stray '.' / ',' / ';' / ':' before the newline that
    # the label-strip regex couldn't consume because punctuation isn't
    # whitespace. Safe because no LC field value legitimately STARTS
    # with a punctuation-only line.
    v = re.sub(r'^[\.,;:]\s*[\n\r]+', '', v).strip()
    v = re.sub(r'^[\.,;:]+\s*(?=[A-Za-z0-9])', '', v).strip()

    # 14b. F32B amount cleanup
    if tag == '32B':
        _ccy = re.search(r'([A-Z]{3})(?=\s|\d|$)', v)
        _ccy_str = _ccy.group(1) if _ccy else 'USD'
        # Format 1: #123,456.78# (hash-wrapped)
        _am = re.search(r'#([\d,]+\.\d+)#?', v)
        if _am:
            v = f"{_ccy_str} {_am.group(1)}"
        else:
            # Format 2: European 123.456,78
            _am2 = re.search(r'(\d[\d.]*,\d{2})\b', v)
            if _am2:
                _amt = _am2.group(1).replace('.', '').replace(',', '.')
                try:
                    v = f"{_ccy_str} {float(_amt):,.2f}"
                except ValueError:
                    pass
            else:
                # Format 3: USD59415, or USD 59415 (no decimals, trailing comma/period)
                # Also handles: USD1,234,567 or USD 1234567.00
                _am3 = re.search(r'([A-Z]{3})\s*([\d,]+(?:\.\d{0,2})?)[,.\s]*$', v)
                if _am3:
                    _raw = _am3.group(2).rstrip(',.')
                    try:
                        # If it has dots as decimal (USD59415.00)
                        if '.' in _raw:
                            _val = float(_raw.replace(',', ''))
                        else:
                            _val = float(_raw.replace(',', ''))
                        v = f"{_am3.group(1)} {_val:,.2f}"
                    except ValueError:
                        pass

    return v.strip()


def _strip_field_sub_labels(tag: str, value: str) -> str:
    """
    Strip the full chain of SWIFT sub-labels that bleed into a field value
    after the main label has been removed. Used by both the base-field
    cleanup loop AND _apply_amendment so that amendment values get the
    same treatment as base values.

    Handles patterns like:
        - Party Identifier - Identifier Code
        Identifier Code:
        UNILPKKA
        UNITED BANK LIMITED
        KARACHI PK
    by stripping "- Party Identifier - Identifier Code" and "Identifier Code:"
    while preserving the actual SWIFT BIC + bank name lines below.
    """
    if not value:
        return value
    v = value
    # "- Party Identifier - Identifier Code"
    v = re.sub(r'-?\s*Party\s+Identifier\s*-?\s*Identifier\s*(?:Code)?\s*\n?',
               '', v, flags=re.IGNORECASE).strip()
    # leading "- "
    v = re.sub(r'^-\s+', '', v).strip()
    # "Identifier Code:" / "Identifier:"
    v = re.sub(r'Identifier\s+Code:?\s*\n?', '', v, flags=re.IGNORECASE).strip()
    v = re.sub(r'^Identifier:?\s*\n?', '', v, flags=re.IGNORECASE).strip()
    # "Name and Address:" header rows
    v = re.sub(r'^Name\s+and\s+Address:?\s*\n?', '', v, flags=re.IGNORECASE).strip()
    v = re.sub(r'-\s*Name\s+and\s+Address\s*-?\s*(?:Name\s+and\s+Address)?:?\s*[\n\r]*',
               '', v, flags=re.IGNORECASE).strip()
    # Fusion "... By ..." availability prefix
    v = re.sub(r'\.{2,}\s*By\s*\.{2,}\s*-?\s*(?:Name\s+and\s+Address\s*-?\s*)*:?\s*[\n\r]*',
               '', v, flags=re.IGNORECASE).strip()
    # Drawee / Applicant Bank "- Party ... Code" sub-headers
    v = re.sub(r'^Drawee\s*-?\s*Party.*?Code\s*\n?', '', v, flags=re.IGNORECASE).strip()
    v = re.sub(r'^Applicant\s+Bank\s*-?\s*Party.*?Code\s*\n?', '', v, flags=re.IGNORECASE).strip()
    v = re.sub(r'^Issuing\s+Bank\s*-?\s*Party.*?Code\s*\n?', '', v, flags=re.IGNORECASE).strip()
    v = re.sub(r'^Reimbursing\s+Bank\s*-?\s*Party.*?Code\s*\n?', '', v, flags=re.IGNORECASE).strip()
    v = re.sub(r'^[\'"]?Advise\s+Through[\'"]?\s+Bank\s*-?\s*Party.*?Code\s*\n?',
               '', v, flags=re.IGNORECASE).strip()
    # "Available With ... By ... Code" prefix (used by 41A/41D)
    v = re.sub(r'^Available\s+With.*?Code\s*\n?', '', v, flags=re.IGNORECASE).strip()
    # "Applicable Rules" prefix (40E)
    v = re.sub(r'^Applicable\s+Rules:?\s*\n?', '', v, flags=re.IGNORECASE).strip()
    return v


def _apply_amendment(
    base_fields: Dict[str, str],
    amendment_fields: List[SwiftField],
    amendment_number: int,
    source_packet_id: int,
) -> AmendmentRecord:
    """
    Apply a single amendment to the base LC fields (in-place modification).

    Rules:
    - F26E (amendment number) and F27 (sequence) are metadata only
    - F30 (amendment date) is recorded
    - F34B (new amount) replaces F32B
    - B-suffix fields replace A-suffix: F45B->F45A, F46B->F46A, F47B->F47A
    - All other fields replace their existing values
    """
    record = AmendmentRecord(
        amendment_number=amendment_number,
        source_packet_id=source_packet_id,
        amendment_date='',
    )

    for sf in amendment_fields:
        tag = sf.tag

        # Skip metadata tags
        if tag in ('26E', '27', '23'):
            if tag == '26E':
                record.amendment_number = _parse_amendment_number(sf.value) or amendment_number
            continue

        # Amendment date
        if tag == '30':
            record.amendment_date = sf.value
            continue

        # New amount — may be replacement or increment
        # P101: In Alliance MT707, the increase amount comes as F32B with
        # "Increase of Documentary Credit Amount" label, not F34B.
        # Handle both tag 34B and tag 32B with "increase" in the value.
        _is_amount_field = (tag == '34B' or
                            (tag == '32B' and re.search(r'increase', sf.value, re.IGNORECASE)))
        if _is_amount_field:
            old_val = base_fields.get('32B', '')
            new_val = sf.value
            # Strip "Increase of Documentary Credit Amount" label
            _is_increase = bool(re.search(r'increase', new_val, re.IGNORECASE))
            new_val = re.sub(r'(?i)^(?:Increase\s+of\s+Documentary\s+Credit\s+Amount|'
                             r'Currency\s+Code,?\s*Amount)\s*[\n\r]*', '', new_val).strip()
            if _is_increase and old_val:
                # P87: Robust amount parsing for increase calculation.
                # Alliance exports amounts in multiple formats:
                #   "761452,56"  (European, comma decimal)
                #   "#761,452.56#"  (US format with # delimiters)
                #   "761,452.56"  (plain US)
                # We need to extract a float from each, using a priority:
                # 1. #-delimited US format: #761,452.56#
                # 2. US format with commas: 761,452.56
                # 3. European format: 761452,56
                # 4. Plain number: 761452.56
                def _parse_amt_str(s):
                    if not s:
                        return None
                    s = s.replace(' ', '')
                    # Try #-delimited first
                    m = re.search(r'#([\d,]+\.\d+)#?', s)
                    if m:
                        return float(m.group(1).replace(',', ''))
                    # US format with thousands
                    m = re.search(r'(\d{1,3}(?:,\d{3})+\.\d+)', s)
                    if m:
                        return float(m.group(1).replace(',', ''))
                    # European format with comma decimal
                    m = re.search(r'(\d+,\d{2})\b', s)
                    if m:
                        return float(m.group(1).replace(',', '.'))
                    # Plain decimal
                    m = re.search(r'(\d+\.\d+)', s)
                    if m:
                        return float(m.group(1))
                    # Bare integer
                    m = re.search(r'(\d{3,})', s)
                    if m:
                        return float(m.group(1))
                    return None

                _ccy = re.search(r'\b([A-Z]{3})\b', old_val) or re.search(r'\b([A-Z]{3})\b', new_val)
                _o = _parse_amt_str(old_val)
                _n = _parse_amt_str(new_val)
                if _o and _n and _ccy:
                    _total = _o + _n
                    new_val = f"{_ccy.group(1)} {_total:,.2f}"
            base_fields['32B'] = new_val
            record.fields_changed.append('32B')
            record.change_details['32B'] = {'old': old_val, 'new': new_val, 'via': '34B',
                                            'operation': 'increase' if _is_increase else 'replace'}
            continue

        # B-suffix -> A-suffix replacement
        actual_tag = tag
        if tag.endswith('B') and tag[:-1] + 'A' in SWIFT_FIELD_LABELS:
            base_tag = tag[:-1] + 'A'
            if base_tag in base_fields or tag in ('45B', '46B', '47B'):
                actual_tag = base_tag

        old_val = base_fields.get(actual_tag, '')

        # Check if amendment value contains /ADD/ or /DEL/ instructions
        # These are amendment OPERATIONS, not replacement values
        amd_val = sf.value.strip()

        # P101: Clean Alliance MT707 formatting artifacts.
        # Alliance wraps amendment text in structured prefixes:
        #   "Description of Goods and/or Services\nLine 1\nCode: /REPALL/\n
        #    Narrative: UNDER FIELD 45A,\nLines 2-100\nNarrative: text..."
        # Strip these so the amendment parser sees clean content.
        # 1. Strip leading field label (Description of Goods, Documents Required, etc.)
        _label_strip = _FIELD_LABEL_STRIP.get(actual_tag, _FIELD_LABEL_STRIP.get(tag))
        if _label_strip:
            amd_val = re.sub(_label_strip, '', amd_val, flags=re.IGNORECASE).strip()
        # 2. Strip "Line N" and "Lines N-M" markers
        amd_val = re.sub(r'(?:^|\n)\s*Lines?\s+\d+(?:\s*[-–]\s*\d+)?\s*(?:\n|$)', '\n', amd_val).strip()
        # 3. Strip "Code:" prefix (e.g. "Code: /REPALL/")
        amd_val = re.sub(r'(?:^|\n)\s*Code\s*:\s*', '\n', amd_val).strip()
        # 4. Strip "Narrative:" prefixes from each line
        amd_val = re.sub(r'(?:^|\n)\s*Narrative\s*:\s*', '\n', amd_val).strip()

        # Normalize double slashes around SWIFT keywords: //REPALL// -> /REPALL/
        # P102: Only normalize slashes around known operation keywords, NOT ''
        # (two single quotes used as SWIFT double-quote delimiter).
        amd_val = re.sub(r'//(REPALL|ADD|DEL|DELETE)//', r'/\1/', amd_val, flags=re.IGNORECASE)
        amd_val = re.sub(r'//(REPALL|ADD|DEL|DELETE)/', r'/\1/', amd_val, flags=re.IGNORECASE)
        amd_val = re.sub(r'/(REPALL|ADD|DEL|DELETE)//', r'/\1/', amd_val, flags=re.IGNORECASE)
        # ALSO trigger the operation path for the bare "TO READ AS 'X' INSTEAD
        # OF 'Y'" form that the MT799 free-format parser produces. Without
        # this, an MT799 amendment would fall into the wholesale-replace
        # branch below and clobber the entire base field with the literal
        # instruction string.
        _is_to_read_as = bool(re.search(
            r'TO\s+READ\s+AS\b.*?\bINSTEAD\s+OF\b',
            amd_val, re.IGNORECASE | re.DOTALL,
        ))
        if (re.search(r'/ADD/|/DEL/|/DELETE/|/REPALL/|PLEASE\s+READ', amd_val, re.IGNORECASE)
                or re.search(r'(?:^|\n)\+?\s*\)', amd_val)
                or _is_to_read_as
                or re.search(r'UNDER\s+FIELD\s+\d{2}[A-Z]?\s+ADD\b', amd_val, re.IGNORECASE)):
            # This is an amendment instruction — apply operations to base value
            new_val = _apply_text_amendment(old_val, amd_val)
            base_fields[actual_tag] = new_val
            if old_val != new_val:
                record.fields_changed.append(actual_tag)
                record.change_details[actual_tag] = {
                    'old': old_val,
                    'new': new_val,
                    'operation': 'text_amendment',
                    'ops': _extract_swift_ops(amd_val),
                }
        else:
            # Strip common field labels that bleed into values
            clean_val = re.sub(
                r'^(?:Sender\'?s?\s+Reference|Documentary\s+Credit\s+Number|'
                r'Date\s+of\s+Issue|Date\s+of\s+Amendment|Date\s+and\s+Place\s+of\s+Expiry|'
                r'Documents?\s+Required|Additional\s+Conditions|'
                r'Description\s+of\s+Goods.*?Services?|Period\s+for\s+Presentation.*?Days|'
                r'Increase\s+of\s+Documentary\s+Credit\s+Amount|Currency\s+Code,?\s*Amount|'
                r'Issuing\s+Bank|Reimbursing\s+Bank|[\'"]?Advise\s+Through[\'"]?\s+Bank|'
                r'Applicant\s+Bank|Available\s+With.*?By\.{0,3}|'
                r'Negotiation/?Deferred\s+Payment\s+Details|'
                r'Partial\s+Shipments?|Trans[sh]?ipment|'
                r'Latest\s+Date\s+of\s+Shipment|Confirmation\s+Instructions|'
                r'Port\s+of\s+Loading.*?Departure|Port\s+of\s+Discharge.*?Destination|'
                r'Beneficiary|Applicant|Charges)\s*[\n\r]*',
                '', amd_val, flags=re.IGNORECASE).strip()

            # Run the same sub-label cleanup the base-field loop uses,
            # so amendment values for fields like F52A (Issuing Bank) end
            # up as just "UNILPKKA / UNITED BANK LIMITED / KARACHI PK"
            # instead of the full label chain.
            clean_val = _strip_field_sub_labels(actual_tag, clean_val)

            base_fields[actual_tag] = clean_val if clean_val else amd_val
            if old_val != base_fields[actual_tag]:
                record.fields_changed.append(actual_tag)
                record.change_details[actual_tag] = {'old': old_val, 'new': base_fields[actual_tag]}

    return record


# == VLM-based amendment application ===========================================

_VLM_AMENDMENT_PROMPT = """You are an expert SWIFT MT707 amendment processor for Letters of Credit.

I have a base Letter of Credit with the following field values, and an MT707 amendment that modifies some fields.

BASE LC FIELDS (current values before this amendment):
{base_fields_text}

AMENDMENT TEXT (MT707 message):
{amendment_text}

YOUR TASK: Apply the amendment instructions to produce the UPDATED field values.

AMENDMENT INSTRUCTION TYPES:
- "/REPALL/" followed by "UNDER FIELD XXA, NOW TO BE READ AS 'new text'" = Replace the ENTIRE field with the new quoted text
- "FIELD XXA-N TO READ AS 'new text'" = Replace clause N of field XXA with the new text
- "FIELD XXA-N WORD TO READ AS 'X' I/O 'Y'" = In clause N, replace word Y with word X
- "FIELD XXA-N DELETE WORDING AS 'text'" = Delete that text from clause N
- "UNDER FIELD XXA ADD CLAUSE AS 'text1' 'text2'" = Add new clauses to the end of field XXA
- "UNDER FIELD XXA ADD <something> AS 'value'" = Add the value to the field
- F32B with "Increase of Documentary Credit Amount" = ADD the new amount to the existing 32B amount
- F34B = New total amount (replaces 32B)
- F45B replaces/modifies F45A, F46B replaces/modifies F46A, F47B replaces/modifies F47A
- F31D = New expiry date/place (replaces existing 31D)
- F44C = New latest shipment date (replaces existing 44C)
- F44E = New port of loading (replaces existing 44E)

IMPORTANT RULES:
1. Only return fields that CHANGED. Do NOT return unchanged fields.
2. For F32B increases: calculate old_amount + increase_amount = new_amount. Return the new total as "CCY new_amount" (e.g. "USD 777,059.70")
3. For clause fields (46A, 47A, 45A) with WORD CHANGES or CLAUSE REPLACEMENTS:
   Return the COMPLETE field text with ALL clauses (existing + modified).
4. For ADD operations ("ADD CLAUSE AS", "ADD LOI CLAUSE AS"):
   Use the special key format "46A_ADD" (or "47A_ADD") and return ONLY the new text to append.
   I will merge it with the existing field programmatically.
   Include the FULL text verbatim — do NOT summarize or truncate.
5. Strip "Narrative:" prefixes, "Line N", "Code:", "Lines N-M" formatting.
6. Strip SWIFT footer text ("Other", "Delivery overdue", "Payment Confirmation", "Page X of Y").
7. Strip "{CHK:...}" checksum blocks.
8. Do NOT include amendment metadata fields (26E, 27, 30, 22A, 23, 21).
9. For quoted text ('text' or ''text''), extract the content between quotes.
10. CROSS-FIELD CONTINUATION: "(CONT FROM FIELD XX)" means the text after belongs to field XX. Remove it from the current field and output field XX with the continuation appended.
11. F48: If it has "/(CONT IN FIELD 47A)", combine the continuation text into F48.

Example for ADD: If amendment says "UNDER FIELD 47A ADD CLAUSE AS 'CHARTER PARTY B/L ACCEPTABLE'",
return: {{"47A_ADD": "CHARTER PARTY B/L ACCEPTABLE"}}

Example for word change: If amendment says "FIELD 46A-2 WORD TO READ AS 'CLEAN ON BOARD' I/O 'CLEAN ON BOARD'",
return the full 46A with the word changed in clause 2.

Return ONLY valid JSON with the changed field tags as keys and their new values.
Example: {{"45A": "MOGAS 92 RON\\nQUANTITY: 10,347...", "32B": "USD 777,059.70"}}
"""


def _apply_amendment_vlm(
    base_fields: dict,
    amendment_text: str,
    amendment_number: int,
    source_packet_id: str,
    _progress=None,
) -> AmendmentRecord:
    """
    Use VLM to apply an MT707 amendment to base LC fields.
    Falls back to regex-based _apply_amendment if VLM fails.
    """
    record = AmendmentRecord(
        amendment_number=amendment_number,
        source_packet_id=source_packet_id,
        amendment_date='',
    )

    # Extract amendment metadata before sending to VLM
    _amd_num_m = re.search(r'(?:F?26E|Number\s+of\s+Amendment)\s*:?\s*(\d+)', amendment_text, re.IGNORECASE)
    if _amd_num_m:
        record.amendment_number = int(_amd_num_m.group(1))
    _date_m = re.search(r'(?:F?30|Date\s+of\s+Amendment)\s*:?\s*(\d{6})\s+(\d{4}\s+\w+\s+\d+)?', amendment_text, re.IGNORECASE)
    if _date_m:
        record.amendment_date = _date_m.group(0).strip()

    # Build base fields text for prompt — only send fields that the amendment
    # is likely to touch, plus a few key reference fields. This keeps the
    # prompt small enough for 16K context models.
    # Detect which fields the amendment mentions
    _amd_upper = amendment_text.upper()
    _touched_tags = set()
    for _t in ['45A', '45B', '46A', '46B', '47A', '47B', '32B', '34B',
               '31D', '44C', '44E', '44F', '48', '50', '59', '71D', '78']:
        if _t in _amd_upper or f'F{_t}' in _amd_upper or f'FIELD {_t}' in _amd_upper:
            # Map B-suffix to A-suffix
            actual = _t[:-1] + 'A' if _t.endswith('B') and _t not in ('32B', '34B', '71B') else _t
            if actual == '34B':
                actual = '32B'
            _touched_tags.add(actual)
    # Always include a few reference fields
    _touched_tags.update(['20', '32B'])

    base_text_parts = []
    for tag in sorted(_touched_tags):
        val = base_fields.get(tag, '')
        if val:
            # Full text for clause fields, truncate others
            max_len = 2500 if tag in ('46A', '47A', '45A', '78') else 300
            val_preview = str(val)[:max_len]
            base_text_parts.append(f"F{tag}: {val_preview}")
    base_fields_text = '\n\n'.join(base_text_parts)

    # Detect ADD-heavy amendments (LOI clauses etc.) and use a focused
    # LLM call that only extracts the ADD text, not the full field.
    _amd_upper_check = amendment_text.upper()
    # Only trigger focused ADD extraction if the amendment has "ADD ... CLAUSE AS"
    # or "ADD LOI CLAUSE AS" — NOT just any mention of "ADD" (which could be
    # in other contexts like field names or addresses)
    _has_add_clause = bool(re.search(
        r'ADD\s+(?:LOI\s+)?CLAUSE\s+AS\b', _amd_upper_check
    ))
    _has_other_ops = any(kw in _amd_upper_check for kw in [
        'WORD TO READ AS', 'DELETE WORDING', 'I/O', 'INCREASE OF DOCUMENTARY',
        'TO READ AS', 'REPALL',
    ])

    if _has_add_clause and not _has_other_ops:
        # Use focused LLM call for ADD extraction
        _add_prompt = """Extract the text being ADDED to the LC from this MT707 amendment message.

AMENDMENT TEXT:
{amd_text}

The amendment contains an instruction like "UNDER FIELD XXA ADD [LOI] CLAUSE AS '...'".
Extract ONLY the quoted text being added. Include the COMPLETE text verbatim — every word, every line.
Strip "Narrative:" prefixes, "Line N", "Code:" labels, "Page X of Y", and SWIFT footer garbage.

Return JSON: {{"field_tag": "46A", "add_text": "the complete extracted text..."}}
If there are also other changes (word changes, amount increases, field replacements), include those too as separate keys.
"""
        _add_clean = re.sub(r'(?:^|\n)\s*Narrative\s*:\s*', '\n', amendment_text).strip()
        _add_clean = re.sub(r'(?:^|\n)\s*(?:Lines?\s+\d+|Code\s*:)\s*(?:\n|$)', '\n', _add_clean).strip()
        _add_clean = re.sub(r'\s*Other\s*\n?\s*Delivery\s+overdue.*$', '', _add_clean, flags=re.IGNORECASE | re.DOTALL).strip()
        _add_clean = re.sub(r'\s*Page\s+\d+\s+of\s+\d+\s*', ' ', _add_clean).strip()
        _add_clean = re.sub(r'\{CHK:[A-F0-9]+\}', '', _add_clean).strip()
        _add_clean = re.sub(r'Block\s+5\s*', '', _add_clean).strip()
        _add_clean = re.sub(r'Report\s+(?:Header|Footer|Content).*?(?=\n[A-Z]|\Z)', '', _add_clean, flags=re.IGNORECASE | re.DOTALL).strip()
        _add_clean = re.sub(r'(?:Delivery\s+overdue|Network\s+delivery|Payment\s+Confirmation|Confirmed\s+(?:Currency|Amount|Date)).*$', '', _add_clean, flags=re.IGNORECASE | re.DOTALL).strip()

        _filled_prompt = _add_prompt.replace('{amd_text}', _add_clean[:10000])
        _llm_url = QWEN_TEXT_LLM_URL or QWEN_VLM_URL
        _llm_model = QWEN_TEXT_LLM_MODEL or QWEN_VLM_MODEL
        try:
            if _progress:
                _progress(f"      LLM ADD extraction: {_llm_url}")
            _resp = requests.post(_llm_url, json={
                "model": _llm_model,
                "messages": [{"role": "user", "content": _filled_prompt}],
                "max_tokens": 8000, "temperature": 0.1,
            }, timeout=None)
            if _resp.status_code == 200:
                _content = _resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
                if _progress:
                    _progress(f"      LLM ADD response: {len(_content)} chars")
                    _progress(f"      LLM ADD raw (first 300): {_content[:300]}")
                _jm = re.search(r'\{.*\}', _content, re.DOTALL)
                if _jm:
                    _raw_json = _jm.group(0)
                    try:
                        _add_result = json.loads(_raw_json)
                    except json.JSONDecodeError:
                        # Try fixing common issues: trailing comma, extra text
                        # Find matching braces manually
                        _depth = 0
                        _end = 0
                        for _ci, _ch in enumerate(_raw_json):
                            if _ch == '{': _depth += 1
                            elif _ch == '}':
                                _depth -= 1
                                if _depth == 0:
                                    _end = _ci + 1
                                    break
                        if _end:
                            try:
                                _add_result = json.loads(_raw_json[:_end])
                            except json.JSONDecodeError:
                                if _progress:
                                    _progress(f"      LLM ADD JSON parse failed, raw: {_raw_json[:200]}")
                                _add_result = {}
                        else:
                            _add_result = {}
                    _add_tag = re.sub(r'^F', '', _add_result.get('field_tag', ''))
                    _add_text = _add_result.get('add_text', '')
                    if _add_tag and _add_text:
                        old_val = base_fields.get(_add_tag, '')
                        _existing_nums = re.findall(r'^(\d+)[\.\)]\s', old_val, re.MULTILINE)
                        _next_num = max([int(n) for n in _existing_nums] + [0]) + 1
                        new_val = old_val.rstrip() + f'\n{_next_num}. ' + _add_text
                        base_fields[_add_tag] = new_val
                        record.fields_changed.append(_add_tag)
                        record.change_details[_add_tag] = {
                            'old': old_val, 'new': new_val,
                            'operation': 'llm_add_clause',
                        }
                        if _progress:
                            _progress(f"      ADD clause to {_add_tag}: {len(_add_text)} chars via LLM")
                    # Also check for other field changes in the response
                    for _k, _v in _add_result.items():
                        if _k in ('field_tag', 'add_text'):
                            continue
                        _norm_k = re.sub(r'^F', '', _k)
                        if _norm_k in ('26E', '27', '30', '22A', '23', '21'):
                            continue
                        _v = str(_v).strip()
                        if _v:
                            _old = base_fields.get(_norm_k, '')
                            base_fields[_norm_k] = _v
                            if _old != _v:
                                record.fields_changed.append(_norm_k)
                                record.change_details[_norm_k] = {
                                    'old': _old, 'new': _v,
                                    'operation': 'llm_amendment',
                                }

                    if not _has_other_ops or record.fields_changed:
                        return record
        except Exception as e:
            if _progress:
                _progress(f"      LLM ADD extraction failed: {e}")

    # Clean amendment text
    clean_amd = amendment_text
    clean_amd = re.sub(r'(?:^|\n)\s*Narrative\s*:\s*', '\n', clean_amd).strip()
    clean_amd = re.sub(r'(?:^|\n)\s*Lines?\s+\d+(?:\s*[-–]\s*\d+)?\s*(?:\n|$)', '\n', clean_amd).strip()
    clean_amd = re.sub(r'(?:^|\n)\s*Code\s*:\s*', '\n', clean_amd).strip()
    clean_amd = re.sub(r'\s*Other\s*\n?\s*Delivery\s+overdue.*$', '', clean_amd, flags=re.IGNORECASE | re.DOTALL).strip()
    clean_amd = re.sub(r'\s*Page\s+\d+\s+of\s+\d+\s*', ' ', clean_amd).strip()
    clean_amd = re.sub(r'\{CHK:[A-F0-9]+\}', '', clean_amd).strip()
    clean_amd = re.sub(r'Block\s+5\s*', '', clean_amd).strip()
    # Remove report headers/footers
    clean_amd = re.sub(r'Report\s+(?:Header|Footer|Content).*?(?=\n[A-Z]|\Z)', '', clean_amd, flags=re.IGNORECASE | re.DOTALL).strip()
    clean_amd = re.sub(r'Message\s+(?:Header|Identifier|Details).*?(?=\n[A-Z]|\Z)', '', clean_amd, flags=re.IGNORECASE | re.DOTALL).strip()
    clean_amd = re.sub(r'(?:Delivery\s+overdue|Network\s+delivery|Payment\s+Confirmation|Confirmed\s+(?:Currency|Amount|Date)).*$', '', clean_amd, flags=re.IGNORECASE | re.DOTALL).strip()

    # Use string concatenation instead of .format() to avoid {CHK:...} issues
    prompt = _VLM_AMENDMENT_PROMPT.replace('{base_fields_text}', base_fields_text).replace('{amendment_text}', clean_amd[:12000])

    # Prefer text-only LLM for amendments (faster, no image overhead)
    _llm_url = QWEN_TEXT_LLM_URL or QWEN_VLM_URL
    _llm_model = QWEN_TEXT_LLM_MODEL or QWEN_VLM_MODEL

    try:
        payload = {
            "model": _llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 8000,
            "temperature": 0.1,
        }
        resp = requests.post(_llm_url, json=payload, timeout=None)
        if _progress:
            _progress(f"      LLM request: {_llm_url} model={_llm_model} prompt={len(prompt)} chars")
        if resp.status_code == 200:
            content = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
            if _progress:
                _progress(f"      LLM response: {len(content)} chars")
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                changes = json.loads(json_match.group(0))
                if isinstance(changes, dict) and changes:
                    for raw_tag, new_val in changes.items():
                        # Normalize tag: strip F-prefix (VLM may return "F32B" vs "32B")
                        tag = re.sub(r'^F', '', raw_tag)
                        # Handle _ADD suffix: append to existing field
                        is_add = tag.endswith('_ADD')
                        if is_add:
                            tag = tag[:-4]  # "46A_ADD" -> "46A"
                        # Skip amendment metadata
                        if tag in ('26E', '27', '30', '22A', '23', '21'):
                            continue
                        new_val = str(new_val).strip()
                        if not new_val:
                            continue
                        old_val = base_fields.get(tag, '')
                        if is_add and old_val:
                            # Append: add new text after existing
                            new_val = old_val.rstrip() + '\n' + new_val
                        base_fields[tag] = new_val
                        if old_val != new_val:
                            record.fields_changed.append(tag)
                            record.change_details[tag] = {
                                'old': old_val,
                                'new': new_val,
                                'operation': 'vlm_amendment',
                            }
                    if _progress:
                        _progress(f"      VLM amendment applied: {record.fields_changed}")
                    return record
        if _progress:
            _err_body = resp.text[:300] if resp.text else ''
            _progress(f"      LLM amendment failed (status {resp.status_code}): {_err_body}, falling back to regex")
    except Exception as e:
        if _progress:
            _progress(f"      VLM amendment error: {e}, falling back to regex")

    # Fallback: return empty record (regex will be tried by caller)
    return None


def _parse_amendment_number(value: str) -> Optional[int]:
    """Extract amendment number from F26E value."""
    m = re.search(r'(\d+)', value)
    return int(m.group(1)) if m else None


def _sort_amendments(amendment_packets: list) -> list:
    """Sort amendment packets by F26E number or F30 date."""
    def _get_sort_key(pkt):
        text = _get_packet_refined_text(pkt)
        # Try amendment number
        m = re.search(r'(?:F?26E\s*:?\s*|:26E:|amendment\s*(?:no\.?|number)\s*:?\s*)(\d+)', text, re.IGNORECASE)
        if m:
            return int(m.group(1))
        # Try date
        m = re.search(r'(?:F?30\s*:?\s*|:30:)(\d{6})', text)
        if m:
            return int(m.group(1))
        return 9999

    return sorted(amendment_packets, key=_get_sort_key)


# == Helpers ==================================================================

# Module-level page text lookup (set by run() before extraction)
_PAGE_TEXT_LOOKUP = {}

def _get_packet_refined_text(pkt) -> str:
    """Get concatenated text from a packet using page_numbers -> page_texts lookup."""
    # First try page_numbers + global lookup (set by run())
    page_nums = pkt.get('page_numbers', []) if isinstance(pkt, dict) else getattr(pkt, 'page_numbers', [])
    if page_nums and _PAGE_TEXT_LOOKUP:
        texts = []
        for pn in page_nums:
            t = _PAGE_TEXT_LOOKUP.get(pn, '')
            if t:
                texts.append(t)
        if texts:
            return '\n'.join(texts)

    # Fallback: try pages list with text fields
    pages = pkt.pages if hasattr(pkt, 'pages') else pkt.get('pages', [])
    texts = []
    for p in pages:
        if hasattr(p, 'refined_text'):
            t = p.refined_text
        elif isinstance(p, dict):
            t = p.get('refined_text', p.get('cleaned_text', p.get('raw_text', '')))
        elif isinstance(p, int):
            # Page number — look up in global lookup
            t = _PAGE_TEXT_LOOKUP.get(p, '')
        else:
            t = getattr(p, 'refined_text', getattr(p, 'cleaned_text', ''))
        if t:
            texts.append(t)
    return '\n'.join(texts)


def _get_packet_first_page(pkt) -> int:
    """Get first page number from packet."""
    pages = pkt.pages if hasattr(pkt, 'pages') else pkt.get('pages', [])
    if pages:
        p = pages[0]
        return p.page_number if hasattr(p, 'page_number') else p.get('page_number', 0)
    return 0


def _get_packet_field(pkt, field_name: str, default=''):
    """Get a field from packet (handles both dataclass and dict)."""
    if hasattr(pkt, field_name):
        return getattr(pkt, field_name)
    return pkt.get(field_name, default)


# == VLM-based field extraction ===============================================

_VLM_EXTRACT_PROMPT = """You are a SWIFT message parser. Extract ALL SWIFT fields from this Letter of Credit page.

OCR TEXT:
{text}

Return ONLY valid JSON with SWIFT field tags as keys and their values. Example:
{{
    "20": "ILC07860560723PK",
    "31C": "230509",
    "31D": "230810UNITED KINGDOM",
    "40A": "IRREVOCABLE",
    "40E": "UCP LATEST VERSION",
    "50": "PAKISTAN STATE OIL COMPANY LTD.\\nP.S.O. HOUSE, KHAYABAN-E-IQBAL\\nCLIFTON, P.O.BOX 3983\\nKARACHI - PAKISTAN",
    "59": "SAHARA ENERGY RESOURCE LTD\\n21-23 VICTORIA STREET, 2ND FLOOR,\\nDOUGLAS, IM1 2LW, ISLE OF MAN",
    "32B": "USD4583829,00",
    "39A": "05/05",
    "46A": "1.COPY OF VESSEL'S NOTICE OF READINESS...",
    "47A": "(1) PHOTOCOPIES OF SIGNED DOCUMENTS ACCEPTABLE...",
    "78": "Instructions to the bank..."
}}

RULES:
- Use standard SWIFT field tags (20, 27, 31C, 31D, 32B, 39A, 40A, 40E, 41A, 42A, 42C, 42P, 43P, 43T, 44A, 44C, 44E, 44F, 45A, 45B, 46A, 46B, 47A, 47B, 48, 49, 50, 51A, 52A, 53A, 57A, 59, 71D, 78)
- Extract ONLY the field VALUE, not the label (e.g., for "20: Documentary Credit Number\\nILC07860560723PK", extract only "ILC07860560723PK")
- For multi-line values (like 46A, 47A, 45A), include the COMPLETE text with newlines
- Preserve clause numbering (1., 2., 3...) in 46A/47A
- If a field spans multiple pages, extract what's on THIS page
- Return empty object {{}} if no SWIFT fields found
"""


def _extract_fields_vlm_page(page_num: int, image_path: str, text: str) -> dict:
    """Send one LC page to VLM for field extraction."""
    try:
        payload = {
            "model": QWEN_VLM_MODEL,
            "messages": [{"role": "user", "content": []}],
            "max_tokens": 4000, "temperature": 0.1
        }
        content_parts = []
        if image_path and os.path.exists(image_path):
            img_b64 = base64.b64encode(open(image_path, 'rb').read()).decode()
            content_parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}})
        content_parts.append({"type": "text", "text": _VLM_EXTRACT_PROMPT.format(text=text[:6000])})
        payload["messages"][0]["content"] = content_parts

        resp = requests.post(QWEN_VLM_URL, json=payload, timeout=None)
        if resp.status_code != 200:
            return {}
        content = resp.json().get('choices', [{}])[0].get('message', {}).get('content', '')
        content = content.strip()
        if content.startswith('```'):
            content = content.split('\n', 1)[1] if '\n' in content else content[3:]
        if content.endswith('```'):
            content = content[:-3]
        content = content.strip()
        json_start = content.find('{')
        json_end = content.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            return json.loads(content[json_start:json_end])
    except Exception:
        pass
    return {}


def _extract_fields_vlm(base_pkt, base_text: str, base_page: int, _progress) -> list:
    """Use VLM to extract SWIFT fields from LC pages."""
    page_nums = base_pkt.get('page_numbers', []) if isinstance(base_pkt, dict) else getattr(base_pkt, 'page_numbers', [])

    # Get image paths for each page
    results_dir = None
    for pn in page_nums:
        # Try to find image path from packet pages data
        pages_data = base_pkt.get('pages', []) if isinstance(base_pkt, dict) else getattr(base_pkt, 'pages', [])
        for pd in pages_data:
            img = pd.get('page_image_path', '') if isinstance(pd, dict) else getattr(pd, 'page_image_path', '')
            if img and os.path.exists(img):
                results_dir = os.path.dirname(os.path.dirname(img))
                break
        if results_dir:
            break

    # Send each LC page to VLM concurrently
    page_items = []
    for pn in page_nums:
        txt = _PAGE_TEXT_LOOKUP.get(pn, '')
        img_path = ''
        if results_dir:
            candidate = os.path.join(results_dir, 'images', f'page_{pn:03d}.png')
            if os.path.exists(candidate):
                img_path = candidate
        page_items.append((pn, img_path, txt))

    _progress(f"  VLM extracting from {len(page_items)} LC pages concurrently...")
    merged_fields = {}
    with ThreadPoolExecutor(max_workers=min(MAX_CONCURRENT_VLM, len(page_items))) as executor:
        futures = {executor.submit(_extract_fields_vlm_page, pn, img, txt): pn
                   for pn, img, txt in page_items}
        for fut in as_completed(futures):
            pn = futures[fut]
            try:
                page_fields = fut.result()
                for tag, val in page_fields.items():
                    if val and tag not in merged_fields:
                        merged_fields[tag] = val
                    elif val and tag in merged_fields and tag in ('46A', '47A', '45A', '78'):
                        # Append continuation text for clause fields
                        merged_fields[tag] = merged_fields[tag] + '\n' + val
                _progress(f"    Page {pn}: {len(page_fields)} fields")
            except Exception as e:
                _progress(f"    Page {pn}: VLM error: {e}")

    # Convert to SwiftField list
    fields = []
    for tag, val in merged_fields.items():
        fields.append(SwiftField(
            tag=tag,
            label=SWIFT_FIELD_LABELS.get(tag, f'Field {tag}'),
            value=str(val).strip(),
            source_page=base_page,
            source_mt='MT700',
        ))
    return fields


# == Main run function ========================================================

def run(step5_result: dict, output_dir: str = None, progress_callback=None) -> dict:
    """
    Execute Step 6: Final LC Consolidation.

    Extracts SWIFT fields from GLM text using regex, applies amendments
    in sequence, splits clause fields into individual clauses.

    Args:
        step5_result: Output from Step 5 (with 'packets' list of ReconciledPacket)
        output_dir: Directory to save results
        progress_callback: Optional callback(message: str)

    Returns:
        dict with 'final_lc', 'shipping_packets', 'other_packets', 'elapsed_seconds'
    """
    def _progress(msg):
        if progress_callback:
            progress_callback(f"[Step 6] {msg}")
        print(f"[Step 6] {msg}")

    start_time = time.time()

    # Set up page text lookup from Step 2 cleaned text
    global _PAGE_TEXT_LOOKUP
    _page_texts = step5_result.get('page_texts', {})
    if _page_texts:
        _PAGE_TEXT_LOOKUP = {int(k): v for k, v in _page_texts.items() if v}
        _progress(f"Page text lookup: {len(_PAGE_TEXT_LOOKUP)} pages with text")

    packets_in = step5_result.get('packets', [])
    _progress(f"Consolidating Final LC from {len(packets_in)} packets...")

    # -- Separate packet types --
    mt700_packets = []
    mt707_packets = []
    mt799_packets = []
    other_mt_packets = []
    shipping_packets = []
    other_packets = []

    for pkt in packets_in:
        mt = _get_packet_field(pkt, 'mt_type', '')
        # ── LC issuance family ──
        # MT700  = Issue of Documentary Credit
        # MT701  = Issue (continuation)
        # MT705  = Pre-Advice of Documentary Credit
        # MT710  = Advice of Third Bank's LC
        # MT711  = Advice (continuation)
        # MT720  = Transfer of Documentary Credit
        # MT721  = Transfer (continuation)
        # MT760  = Issue of Demand Guarantee / Standby LC
        if mt in ('MT700', 'MT701', 'MT705', 'MT710', 'MT711',
                  'MT720', 'MT721', 'MT760'):
            mt700_packets.append(pkt)
        # ── Amendment family ──
        # MT707 = Amendment to Documentary Credit
        # MT708 = Amendment (continuation)
        # MT747 = Amendment to Authorisation to Reimburse
        # MT767 = Amendment to Demand Guarantee / Standby
        # MT775 = Further Amendment to Documentary Credit
        elif mt in ('MT707', 'MT708', 'MT747', 'MT767', 'MT775'):
            mt707_packets.append(pkt)
        # ── Free format ──
        elif mt in ('MT799', 'MT999'):
            mt799_packets.append(pkt)
        # ── Bank-to-bank acknowledgements / advices / claims ──
        # MT730 acknowledgement, MT732 discharge, MT734 refusal,
        # MT735 full refusal under reserve, MT740 reimb auth,
        # MT742 reimb claim, MT744 non-conforming claim, MT750 discrepancy,
        # MT752 authorisation, MT754 advice of payment/acceptance/negotiation,
        # MT756 reimbursement advice, MT768 guarantee ack, MT769 release,
        # MT785/MT786/MT787 guarantee notices.
        elif mt.startswith('MT'):
            other_mt_packets.append(pkt)
        elif mt == 'shipping':
            shipping_packets.append(pkt)
        else:
            other_packets.append(pkt)

    _progress(f"  MT700: {len(mt700_packets)}, MT707: {len(mt707_packets)}, "
              f"MT799: {len(mt799_packets)}, shipping: {len(shipping_packets)}, "
              f"other: {len(other_mt_packets) + len(other_packets)}")

    final_lc = FinalLC()
    warnings = []

    # -- Extract fields from original LC (MT700) --
    if not mt700_packets:
        warnings.append("No MT700 (original LC) packet found")
        _progress("  WARNING: No MT700 found")
        if mt707_packets:
            warnings.append("Using MT707 as base LC (no MT700 available)")
            mt700_packets = [mt707_packets.pop(0)]
        elif other_mt_packets:
            warnings.append("Using other MT packet as base LC")
            mt700_packets = [other_mt_packets.pop(0)]

    if mt700_packets:
        if len(mt700_packets) > 1:
            warnings.append(f"Multiple MT700 packets found ({len(mt700_packets)}), using first")
            _progress(f"  WARNING: {len(mt700_packets)} MT700 packets, using first")

        base_pkt = mt700_packets[0]
        base_text = _get_packet_refined_text(base_pkt)
        base_page = _get_packet_first_page(base_pkt)

        # Detect format from GLM text
        final_lc.swift_format = _detect_format_from_text(base_text)
        _progress(f"  Base LC format: {final_lc.swift_format}")

        # Extract fields using regex on GLM text
        base_fields = _extract_swift_fields(base_text, source_page=base_page, source_mt='MT700')
        _progress(f"  Extracted {len(base_fields)} fields from MT700")

        # VLM fallback: if regex extracted fewer than 5 fields, use VLM to extract
        if len(base_fields) < 5:
            _progress(f"  Regex extracted only {len(base_fields)} fields — trying VLM extraction...")
            try:
                vlm_fields = _extract_fields_vlm(base_pkt, base_text, base_page, _progress)
                if len(vlm_fields) > len(base_fields):
                    _progress(f"  VLM extracted {len(vlm_fields)} fields (better than regex {len(base_fields)})")
                    base_fields = vlm_fields
                else:
                    _progress(f"  VLM extracted {len(vlm_fields)} fields (not better, keeping regex)")
            except Exception as e:
                _progress(f"  VLM extraction failed: {e}")

        # Clean field values: strip SWIFT label text from values
        # GLM text has: "F20: Documentary Credit Number\n05251LC082463"
        # Regex captures: "Documentary Credit Number\n05251LC082463"
        # We need just: "05251LC082463"
        _LABEL_STRIP = {
            '20': r'^(?:Documentary\s+Credit\s+Number|Sender\'?s?\s+Reference|Transaction\s+Reference)\s*[\n\r]*',
            '27': r'^Sequence\s+of\s+Total\s*[\n\r]*',
            '31C': r'^Date\s+of\s+Issue\s*[\n\r]*',
            '31D': r'^Date\s+and\s+Place\s+of\s+Expiry\s*[\n\r]*',
            '32B': r'^(?:Currency\s+Code,?\s*Amount|Increase\s+of\s+Documentary\s+Credit\s+Amount)\s*[\n\r]*',
            '39A': r'^Percentage\s+Credit\s+Amount\s+Tolerance\s*[\n\r]*',
            '40A': r'^Form\s+of\s+Documentary\s+Credit\s*[\n\r]*',
            '40E': r'^Applicable\s+Rules\s*[\n\r]*',
            '41A': r'^(?:Available\s+With.*?(?:By\.{0,3})?|\.{2,}\s*By\s*\.{2,}.*?(?:Name\s+and\s+Address)?:?)\s*[\n\r]*',
            '41D': r'^(?:Available\s+With.*?(?:Code|By\.{0,3})?|\.{2,}\s*By\s*\.{2,}.*?(?:Name\s+and\s+Address)?:?)\s*[\n\r]*',
            '42A': r'^(?:Drawee|Issuing\s+Bank).*?(?:Identifier\s+Code)?\s*[\n\r]*',
            '42C': r'^Drafts\s+at\s*\.{0,3}\s*[\n\r]*',
            '42P': r'^(?:Negotiation/)?Deferred\s+Payment\s+Details\s*[\n\r]*',
            '43P': r'^Partial\s+Shipments?[\.,;:]?\s*[\n\r]*',
            '43T': r'^Trans[sh]?ipment[\.,;:]?\s*[\n\r]*',
            '44A': r'^Place\s+of\s+Taking.*?\s*[\n\r]*',
            '44C': r'^Latest\s+Date\s+of\s+Shipment\s*[\n\r]*',
            '44E': r'^Port\s+of\s+Loading.*?Departure\s*[\n\r]*',
            '44F': r'^Port\s+of\s+Discharge.*?Destination\s*[\n\r]*',
            '45A': r'^Description\s+of\s+Goods.*?Services?\s*[\n\r]*',
            '46A': r'^Documents?\s+Required\s*[\n\r]*',
            '47A': r'^Additional\s+Conditions\s*[\n\r]*',
            '48': r'^Period\s+for\s+Presentation.*?Days\s*[\n\r]*',
            '49': r'^Confirmation\s+Instructions\s*[\n\r]*',
            '50': r'^Applicant\s*[\n\r]*',
            '51A': r'^Applicant\s+Bank.*?(?:Identifier\s+Code)?\s*[\n\r]*',
            '51D': r'^(?:Applicant\s+Bank|Bank)\s*-?\s*(?:Party)?.*?(?:Name\s+and\s+Address)?:?\s*[\n\r]*',
            '52A': r'^(?:Issuing\s+Bank|Applicant\s+Bank).*?(?:Identifier\s+Code)?\s*[\n\r]*',
            '53A': r'^Reimbursing\s+Bank.*?(?:Identifier\s+Code)?\s*[\n\r]*',
            '57A': r'^[\'"]?Advise\s+Through[\'"]?\s+Bank.*?(?:Identifier\s+Code\s*:?\s*)?\s*[\n\r]*',
            '59': r'^Beneficiary\s*[\n\r]*(?:Name\s+and\s+Address:?\s*[\n\r]*)?',
            '71D': r'^Charges\s*[\n\r]*',
            '78': r'^Instructions\s+to\s+the\s+Paying.*?Bank\s*[\n\r]*',
        }
        for sf in base_fields:
            # Strip label prefix from value
            _strip_pat = _LABEL_STRIP.get(sf.tag, '')
            if _strip_pat:
                sf.value = re.sub(_strip_pat, '', sf.value, flags=re.IGNORECASE).strip()
            # Also clean sub-labels that appear inside field values
            # Handle "- Party Identifier - Identifier Code\nIdentifier Code:\nUNILPKKA"
            sf.value = re.sub(r'-?\s*Party\s+Identifier\s*-?\s*Identifier\s*(?:Code)?\s*\n?', '', sf.value, flags=re.IGNORECASE).strip()
            sf.value = re.sub(r'^-\s+', '', sf.value).strip()  # Strip leading "- "
            sf.value = re.sub(r'Identifier\s+Code:?\s*\n?', '', sf.value, flags=re.IGNORECASE).strip()
            sf.value = re.sub(r'Identifier:?\s*\n?', '', sf.value, flags=re.IGNORECASE).strip()
            sf.value = re.sub(r'^Name\s+and\s+Address:?\s*\n?', '', sf.value, flags=re.IGNORECASE).strip()
            # Fusion format cleanup: "... By ... - Name and Address - Name and Address:"
            sf.value = re.sub(r'\.{2,}\s*By\s*\.{2,}\s*-?\s*(?:Name\s+and\s+Address\s*-?\s*)*:?\s*[\n\r]*', '', sf.value, flags=re.IGNORECASE).strip()
            sf.value = re.sub(r'-\s*Name\s+and\s+Address\s*-?\s*(?:Name\s+and\s+Address)?:?\s*[\n\r]*', '', sf.value, flags=re.IGNORECASE).strip()
            # Remove "US DOLLAR" currency name (keep just currency code)
            if sf.tag == '32B':
                sf.value = re.sub(r'\b(?:US\s+DOLLAR|EURO|POUND\s+STERLING|JAPANESE\s+YEN)\s*[\n\r]*', '', sf.value, flags=re.IGNORECASE).strip()
            sf.value = re.sub(r'^Applicable\s+Rules:?\s*\n?', '', sf.value, flags=re.IGNORECASE).strip()
            # Clean "Available With ... By ... - Name and Address - Code" prefix
            sf.value = re.sub(r'^Available\s+With.*?Code\s*\n?', '', sf.value, flags=re.IGNORECASE).strip()
            sf.value = re.sub(r'^Drawee\s*-?\s*Party.*?Code\s*\n?', '', sf.value, flags=re.IGNORECASE).strip()
            sf.value = re.sub(r'^Applicant\s+Bank\s*-?\s*Party.*?Code\s*\n?', '', sf.value, flags=re.IGNORECASE).strip()

            # Strip SWIFT message footer ("Other\nDelivery overdue..." section)
            sf.value = re.sub(
                r'\n\s*Other\s*\n\s*(?:Delivery\s+overdue|Network\s+delivery|Payment\s+Confirmation).*$',
                '', sf.value, flags=re.IGNORECASE | re.DOTALL).strip()

            # Strip "(CONT FROM/IN FIELD ...)" cross-references — P87
            sf.value = re.sub(r'\(CONT\.?\s*(?:FROM|IN)\s+FIELD\s+\w+\)', '', sf.value, flags=re.IGNORECASE).strip()
            sf.value = re.sub(r'/\(CONT\.?\s*(?:FROM|IN)\s+FIELD\s+\w+\)', '', sf.value, flags=re.IGNORECASE).strip()

            # Truncate if another F-tag got merged in (e.g. "...F41D: Available With...")
            _ftag_merge = re.search(r'F\d{2}[A-Z]?\s*:', sf.value)
            if _ftag_merge:
                sf.value = sf.value[:_ftag_merge.start()].strip()

            # Strip "Page X of Y" page numbering from PDF
            sf.value = re.sub(r'\bPage\s+\d+\s+of\s+\d+\b', '', sf.value, flags=re.IGNORECASE).strip()

            # Strip OCR garbage from blank/unreadable pages
            sf.value = re.sub(
                r'There is no visible text.*?(?:clearly visible|another version)[.\s]*',
                '', sf.value, flags=re.IGNORECASE | re.DOTALL).strip()
            sf.value = re.sub(
                r'The image appears to be blank[.\s]*',
                '', sf.value, flags=re.IGNORECASE).strip()

            # Skip Sequence of Total (tag 27) — not useful in Final LC
            if sf.tag == '27':
                continue

            # Strip inline sub-labels: "Date: 260131\nPlace: NETHERLANDS" -> "260131 NETHERLANDS"
            sf.value = re.sub(r'\bDate:?\s*', '', sf.value).strip()
            sf.value = re.sub(r'\bPlace:?\s*', '', sf.value).strip()
            sf.value = re.sub(r'\bCurrency:?\s*', '', sf.value).strip()
            sf.value = re.sub(r'\bAmount:?\s*', '', sf.value).strip()
            sf.value = re.sub(r'\bDays:?\s*', '', sf.value).strip()
            sf.value = re.sub(r'\bNarrative:?\s*/?', '', sf.value).strip()
            sf.value = re.sub(r'\bNumber:?\s*', '', sf.value).strip()
            sf.value = re.sub(r'\bTotal:?\s*', '', sf.value).strip()
            sf.value = re.sub(r'\bTolerance\s+\d:?\s*', '', sf.value).strip()
            sf.value = re.sub(r'\bCode:?\s*', '', sf.value).strip()
            # Clean SWIFT date format: "260131 2026 Jan 31" -> "2026-01-31"
            _dm = re.search(r'(\d{6})\s+(\d{4})\s+(\w{3})\s+(\d{1,2})', sf.value)
            if _dm and sf.tag in ('31C', '31D', '44C'):
                _months = {'Jan':'01','Feb':'02','Mar':'03','Apr':'04','May':'05','Jun':'06',
                           'Jul':'07','Aug':'08','Sep':'09','Oct':'10','Nov':'11','Dec':'12'}
                _date_str = f"{_dm.group(2)}-{_months.get(_dm.group(3),'01')}-{int(_dm.group(4)):02d}"
                sf.value = sf.value[:_dm.start()] + _date_str + sf.value[_dm.end():]
                sf.value = sf.value.strip()
            # Clean amount format for F32B:
            # Fusion: "USD\nUS DOLLAR\n516000,00\n#516,000.00" or "USD 516000,00 #516,000.00#"
            if sf.tag == '32B':
                _ccy = re.search(r'\b([A-Z]{3})\b', sf.value)
                _ccy_str = _ccy.group(1) if _ccy else 'USD'
                # Try formatted amount: #516,000.00# or #516,000.00
                _am = re.search(r'#([\d,]+\.\d+)#?', sf.value)
                if _am:
                    sf.value = f"{_ccy_str} {_am.group(1)}"
                else:
                    # Try European format: 516000,00
                    _am2 = re.search(r'(\d[\d.]*,\d{2})\b', sf.value)
                    if _am2:
                        _amt = _am2.group(1).replace('.', '').replace(',', '.')
                        sf.value = f"{_ccy_str} {float(_amt):,.2f}"
            # Convert raw SWIFT date codes: "250103" → "2025-01-03"
            # Only strip AFTER converting — the old code stripped unconditionally
            # which wiped Alliance dates that are just 6 digits with no long form.
            if sf.tag in ('31C', '31D', '44C'):
                _raw_dm = re.search(r'\b(\d{2})(\d{2})(\d{2})\b', sf.value)
                if _raw_dm and not re.search(r'\d{4}-\d{2}-\d{2}', sf.value):
                    _yy, _mm, _dd = _raw_dm.group(1), _raw_dm.group(2), _raw_dm.group(3)
                    _year = f"20{_yy}" if int(_yy) < 80 else f"19{_yy}"
                    sf.value = sf.value[:_raw_dm.start()] + f"{_year}-{_mm}-{_dd}" + sf.value[_raw_dm.end():]
                    sf.value = sf.value.strip()

            # Strip a leading punctuation-only line (e.g. ".\nALLOWED" →
            # "ALLOWED"). The label-strip regex doesn't consume a stray
            # "." or "," that the SWIFT export sometimes leaves between
            # the field label and the value (typically on F43P / F43T
            # short enums like "Transhipment.\n  ALLOWED").
            sf.value = re.sub(r'^[\.,;:]\s*[\n\r]+', '', sf.value).strip()
            sf.value = re.sub(r'^[\.,;:]+\s*(?=[A-Za-z0-9])', '', sf.value).strip()

            final_lc.consolidated_fields[sf.tag] = sf.value
            final_lc.original_fields[sf.tag] = sf.value
            _progress(f"    F{sf.tag}: {sf.value[:80]}{'...' if len(sf.value) > 80 else ''}")

        # Set DC number — strip any remaining label text
        _raw_dc = final_lc.consolidated_fields.get('20', '')
        _raw_dc = re.sub(r"(?i)^(?:Sender'?s?\s+Reference|Documentary\s+Credit\s+Number|Transaction\s+Reference\s+Number)\s*[\n\r]*", '', _raw_dc).strip()
        final_lc.dc_number = _raw_dc
        final_lc.consolidated_fields['20'] = _raw_dc

        # BAHL fallback: If F20 is an internal reference (no LC pattern) but F21
        # (Related Reference) contains an LC number, use F21 as the DC number.
        # BAHL uses Transaction Reference in F20 and LC number in F21.
        _raw_f21 = final_lc.consolidated_fields.get('21', '')
        if _raw_f21:
            _raw_f21 = re.sub(r"(?i)^(?:Related\s+Reference|Receiver'?s?\s+Reference)\s*[\n\r]*", '', _raw_f21).strip()
            # Check if F20 looks like a bank internal ref (no LC pattern)
            # and F21 looks like an LC number. Use word boundary \b to avoid
            # matching "LC" inside words like "ELCT183292".
            _f20_has_lc = bool(re.search(r'(?<![A-Z])(?:LC|ILC|ALS|DLC)\d', final_lc.dc_number, re.IGNORECASE))
            _f21_has_lc = bool(re.search(r'(?<![A-Z])(?:LC|ILC|ALS|DLC)\d', _raw_f21, re.IGNORECASE))
            if _raw_f21 and not _f20_has_lc and _f21_has_lc:
                _progress(f"  DC number: using Related Reference (F21) '{_raw_f21}' over Transaction Reference (F20) '{final_lc.dc_number}'")
                final_lc.dc_number = _raw_f21
                final_lc.consolidated_fields['20'] = _raw_f21
        final_lc.source_packets.append(_get_packet_field(base_pkt, 'packet_id', 0))

        # If multiple MT700 packets, extract from subsequent ones too
        # (multi-page LCs where each page was a separate packet)
        #
        # Page-break continuation: when a field value spans pages, e.g.:
        #   Page 7: "F43T: Transhipment\nPage 7 of 10"
        #   Page 8: "ALLOWED\nF44E: Port of Loading..."
        # The value "ALLOWED" at the start of page 8 belongs to F43T.
        # Check if the next page starts with a short value (like ALLOWED,
        # PROHIBITED, WITHOUT) before the first F-tag.
        _SHORT_ENUM_VALUES = {
            'ALLOWED', 'PROHIBITED', 'NOT ALLOWED', 'PERMITTED',
            'WITHOUT', 'CONFIRM', 'MAY ADD', 'IRREVOCABLE',
        }
        for extra_pkt in mt700_packets[1:]:
            extra_text = _get_packet_refined_text(extra_pkt)
            extra_page = _get_packet_first_page(extra_pkt)

            # Check for page-break continuation: if page starts with a
            # short enum value before any F-tag, it belongs to the LAST
            # field from the previous page.
            _first_line = extra_text.strip().split('\n')[0].strip().upper() if extra_text else ''
            if _first_line in _SHORT_ENUM_VALUES:
                # Find the last field that has an empty/label-only value
                _label_only_fields = {'43T', '43P', '49', '40A'}
                for _lof_tag in _label_only_fields:
                    _existing_val = final_lc.consolidated_fields.get(_lof_tag, '')
                    _cleaned = re.sub(r'(?i)^(?:Trans[sh]?ipment|Partial\s+Shipments?|Confirmation\s+Instructions|Form\s+of\s+Documentary\s+Credit)\s*$', '', _existing_val).strip()
                    if not _cleaned and _existing_val:
                        final_lc.consolidated_fields[_lof_tag] = _first_line
                        final_lc.original_fields[_lof_tag] = _first_line
                        _progress(f"    F{_lof_tag}: page-break continuation → '{_first_line}'")
                        break

            extra_fields = _extract_swift_fields(extra_text, source_page=extra_page, source_mt='MT700')
            _clause_tags = {'46A', '47A', '45A', '78', '72', '79'}
            for sf in extra_fields:
                if sf.tag not in final_lc.consolidated_fields:
                    final_lc.consolidated_fields[sf.tag] = sf.value
                    final_lc.original_fields[sf.tag] = sf.value
                    _progress(f"    F{sf.tag} (from extra MT700): {sf.value[:60]}...")
                elif sf.tag in _clause_tags and sf.value:
                    # Append continuation text for multi-page clause fields
                    existing = final_lc.consolidated_fields[sf.tag]
                    if sf.value not in existing:
                        final_lc.consolidated_fields[sf.tag] = existing.rstrip() + '\n' + sf.value
                        final_lc.original_fields[sf.tag] = final_lc.consolidated_fields[sf.tag]
                        _progress(f"    F{sf.tag} (appended from extra MT700 page): +{len(sf.value)} chars")
            final_lc.source_packets.append(_get_packet_field(extra_pkt, 'packet_id', 0))

    # -- Apply amendments (MT707) in sequence --
    if mt707_packets:
        sorted_amendments = _sort_amendments(mt707_packets)
        _progress(f"  Applying {len(sorted_amendments)} amendments...")

        for i, amd_pkt in enumerate(sorted_amendments):
            amd_text = _get_packet_refined_text(amd_pkt)
            amd_page = _get_packet_first_page(amd_pkt)
            pkt_id = _get_packet_field(amd_pkt, 'packet_id', 0)
            is_799 = bool(_get_packet_field(amd_pkt, 'is_799_amendment', False))
            src_mt_label = _get_packet_field(amd_pkt, 'source_mt', '') or 'MT707'

            if is_799:
                # MT799 free-format amendment: use the free-format parser ONLY.
                # We must NOT fall back to _extract_swift_fields() on a 799 page
                # because that would extract F20/F79 (the narrative tag) and
                # apply them as regular consolidated fields, polluting the LC.
                # If the parser can't recognise the amendment instructions, we
                # leave amd_fields empty and skip the amendment — better to
                # under-apply than to corrupt the consolidated LC.
                amd_fields = _extract_mt799_amendment_fields(
                    amd_text, source_page=amd_page, source_mt=src_mt_label or 'MT799',
                )
                # Defensive filter: drop any tag that should NEVER come from a
                # 799 amendment regardless of source. F79 (narrative), F20
                # (transaction reference), F21 (related reference), F23
                # (issuing bank reference) are 799-specific routing fields,
                # not LC fields.
                amd_fields = [f for f in amd_fields if f.tag not in ('79', '20', '21', '23')]
                if not amd_fields:
                    _progress(
                        f"    Amendment {i + 1}: MT799 free-format from packet {pkt_id} "
                        f"— parser found 0 amendment instructions; skipping (no fields applied)"
                    )
                else:
                    _progress(
                        f"    Amendment {i + 1}: MT799 free-format → "
                        f"{len(amd_fields)} amendment field(s) from packet {pkt_id}: "
                        f"{[f.tag for f in amd_fields]}"
                    )
            else:
                amd_fields = _extract_swift_fields(amd_text, source_page=amd_page, source_mt='MT707')
                # VLM fallback for STRUCTURED amendments only — never for 799
                # (the VLM has no notion of "should read as / I/O" semantics).
                if len(amd_fields) < 2:
                    try:
                        vlm_amd = _extract_fields_vlm(amd_pkt, amd_text, amd_page, _progress)
                        if len(vlm_amd) > len(amd_fields):
                            amd_fields = vlm_amd
                    except Exception:
                        pass
                _progress(f"    Amendment {i + 1}: {len(amd_fields)} fields from packet {pkt_id}")

            # Try VLM-based amendment first (more accurate for complex instructions)
            vlm_record = _apply_amendment_vlm(
                final_lc.consolidated_fields,
                amd_text,
                amendment_number=i + 1,
                source_packet_id=pkt_id,
                _progress=_progress,
            )
            if vlm_record and vlm_record.fields_changed:
                record = vlm_record
                _progress(f"      Applied via VLM: {record.fields_changed}")
            else:
                # Fallback to regex-based amendment
                record = _apply_amendment(
                    final_lc.consolidated_fields,
                    amd_fields,
                    amendment_number=i + 1,
                    source_packet_id=pkt_id,
                )
                _progress(f"      Applied via regex: {record.fields_changed}")
            final_lc.amendment_log.append(record)
            final_lc.source_packets.append(pkt_id)

            # Re-run the FULL base-field cleanup pipeline on every field
            # this amendment changed. _apply_amendment writes raw amendment
            # values directly to consolidated_fields, so without this pass
            # an amendment to F44C / F47A / F78 brings the SWIFT message
            # footer ("Other / Delivery overdue / Network delivery / Payment
            # Confirmation / Report Footer") and unconverted SWIFT date
            # codes ("251030 2025 Oct 30") into the Final LC.
            for _ch_tag in record.fields_changed:
                _raw = final_lc.consolidated_fields.get(_ch_tag, '')
                _cleaned = _clean_consolidated_field_value(_ch_tag, _raw)
                if _cleaned != _raw:
                    final_lc.consolidated_fields[_ch_tag] = _cleaned
                    # Keep the change_details in-sync so the audit log
                    # reflects the final visible value, not the raw one.
                    if _ch_tag in record.change_details:
                        record.change_details[_ch_tag]['new'] = _cleaned

            if record.fields_changed:
                _progress(f"      Changed: {', '.join(record.fields_changed)}")
            if record.amendment_date:
                _progress(f"      Date: {record.amendment_date}")

        final_lc.amendment_count = len(final_lc.amendment_log)
        # Update DC number if changed by amendment
        if '20' in final_lc.consolidated_fields:
            final_lc.dc_number = final_lc.consolidated_fields['20']

    # -- Post-processing: clean and resolve cross-references --
    _cf = final_lc.consolidated_fields

    # Track which fields were amended (for UI annotation)
    _amended_fields = set()
    for rec in final_lc.amendment_log:
        _amended_fields.update(rec.fields_changed)
    if _amended_fields:
        _cf['_amended_fields'] = list(_amended_fields)

    # 1. Format SWIFT dates: "230509" → "2023-05-09", "260131" → "2026-01-31"
    for _dt_tag in ('31C', '31D', '44C'):
        _dv = _cf.get(_dt_tag, '')
        if _dv:
            _dm = re.match(r'^(\d{6})\s*$', _dv.strip().split('\n')[0])
            if _dm:
                _raw = _dm.group(1)
                _yr = int(_raw[:2])
                _mo = _raw[2:4]
                _dy = _raw[4:6]
                _full_yr = 2000 + _yr if _yr < 80 else 1900 + _yr
                _formatted = f"{_full_yr}-{_mo}-{_dy}"
                _rest = _dv[_dm.end():].strip()
                _cf[_dt_tag] = f"{_formatted}\n{_rest}".strip() if _rest else _formatted
                _progress(f"  F{_dt_tag}: formatted date {_raw} → {_formatted}")

    # 1b. P73: SWIFT continuation marker resolution.
    #
    # Some Fusion / Alliance exports break a long field across two physical
    # SWIFT fields and stitch them with "(CONT FROM FIELD XX)" markers.
    # Example seen in the wild:
    #   F48: "Period for Presentation in Days
    #         Days: 21
    #         Narrative: /(CONT FROM FIELD 47A)"
    #   F47A: "(CONT FROM FIELD 48)
    #          DAYS FROM DATE OF SHIPMENT BUT WITHIN THE VALIDITY OF LC
    #          ...real F47A clauses..."
    #
    # The continuation chunk in F47A actually belongs to F48, not F47A.
    # This pre-pass walks every field, finds "(CONT FROM FIELD XX)" markers,
    # extracts the chunk that belongs to field XX, appends it to XX, and
    # removes it from the current field. After this pass the regular
    # cross-reference resolver and the per-field cleanup see clean values.
    #
    # The chunk boundary is one of:
    #   • a blank line followed by an UPPERCASE label like "F47A:" / "47A:"
    #   • a blank line followed by a tag-like prefix
    #   • a numbered clause start ("1.", "2)" etc.) at column 0
    #   • end of value
    # P87: Match both "(CONT FROM FIELD XX)" and "(CONT. IN FIELD XX)"
    # and "(CONT IN FIELD XX)" — Alliance uses "IN" while Fusion uses "FROM"
    _cont_marker_re = re.compile(
        r'[/\\]?\s*'
        r'\(\s*CONT\.?\s*(?:INUED|INUATION)?\s+(?:FROM|IN)\s+FIELD\s+(?P<src>\d{2}[A-Z]?)\s*\)\s*'
        r'(?P<rest>(?:(?!\n\s*\d+\s*[.\-\)]\s)(?!\n\s*F?\d{2}[A-Z]?\s*:).)*)',
        re.IGNORECASE | re.DOTALL,
    )
    _cont_pulled: Dict[str, list] = {}
    for _tag in list(_cf.keys()):
        if _tag.startswith('_'):
            continue
        _val = _cf.get(_tag, '')
        if not isinstance(_val, str) or 'CONT' not in _val.upper():
            continue
        _new_val = _val
        _had_match = False
        # Iterate matches from end to start so the indexes stay valid as
        # we slice them out.
        _matches = list(_cont_marker_re.finditer(_val))
        for _m in reversed(_matches):
            _src_tag = _m.group('src').upper()
            _chunk = (_m.group('rest') or '').strip()
            # Skip the empty self-reference seen in F48 ("Narrative: /(CONT
            # FROM FIELD 47A)") — there's no payload to move, the payload
            # lives in F47A and will be handled when we process F47A.
            if not _chunk:
                _new_val = (_new_val[:_m.start()] + _new_val[_m.end():])
                _had_match = True
                continue
            # Don't move content into itself.
            if _src_tag == _tag.upper():
                continue
            _cont_pulled.setdefault(_src_tag, []).append(_chunk)
            _new_val = (_new_val[:_m.start()] + _new_val[_m.end():])
            _had_match = True
        if _had_match:
            _cf[_tag] = re.sub(r'\n{3,}', '\n\n', _new_val).strip()
    # Append the pulled continuation chunks into their target fields.
    for _src_tag, _chunks in _cont_pulled.items():
        _existing = _cf.get(_src_tag, '') or ''
        # Strip a stray "(CONT FROM FIELD XX)" / "/(CONT FROM FIELD XX)"
        # back-reference left in the target so we don't have a marker
        # next to the merged content.
        _existing = re.sub(
            r'(?:^|\n)\s*[/\\]?\s*\(\s*CONT\.?\s*(?:INUED|INUATION)?\s+(?:FROM|IN)\s+FIELD\s+\d{2}[A-Z]?\s*\)\s*',
            '\n', _existing, flags=re.IGNORECASE,
        ).strip()
        _merged_chunks = '\n'.join(_chunks).strip()
        if _existing:
            _cf[_src_tag] = f"{_existing}\n{_merged_chunks}".strip()
        else:
            _cf[_src_tag] = _merged_chunks
        _progress(f"  F{_src_tag}: merged continuation chunk(s) "
                  f"({sum(len(c) for c in _chunks)} chars from "
                  f"{len(_chunks)} marker(s))")

    # 2. Resolve cross-references
    # Pattern A: "++++SEE FIELD 47A++++" → look up value from 47A (marker-based)
    # Pattern B: "REFER CLAUSE NO.10 OF FIELD 47A" → look up clause 10 from 47A
    # Pattern C: "PLS REFER CLAUSE NO.5 OF FIELD 47A" → look up clause 5 from 47A
    for _tag, _val in list(_cf.items()):
        if not isinstance(_val, str):
            continue

        # Pattern A: ++++SEE FIELD XX++++
        _ref_m = re.search(r'\+{3,}SEE\s+FIELD\s+(\d{2}[A-Z]?)\+{3,}', _val, re.IGNORECASE)
        if _ref_m:
            _ref_tag = _ref_m.group(1)
            _ref_val = _cf.get(_ref_tag, '')
            if _ref_val:
                # Look for +++FIELD XX+++ marker in the referenced field
                _marker_pat = r'\+{3,}FIELD\s+' + re.escape(_tag) + r'\+{3,}\s*\n?(.*?)(?=\n\+{3,}FIELD|\Z)'
                _marker_m = re.search(_marker_pat, _ref_val, re.IGNORECASE | re.DOTALL)
                if _marker_m:
                    _resolved = _marker_m.group(1).strip()
                    _cf[_tag] = _resolved
                    _progress(f"  F{_tag}: resolved cross-ref (marker) from F{_ref_tag} → {_resolved[:60]}")
                    continue

        # Pattern B: "REFER CLAUSE NO.X OF FIELD YY" / "PLS REFER CLAUSE NO.X OF FIELD YY"
        # P198fr — make 'NO.' optional so 'REFER CLAUSE 10 OF FIELD 47A'
        # (no 'NO.') also resolves correctly. Same regex handles
        # "CLAUSE NO. 10", "CLAUSE NO.10", "CLAUSE 10".
        _clause_ref_m = re.search(
            r'(?:PLS\s+)?REFER\s+(?:TO\s+)?CLAUSE\s+(?:NO\.?\s*)?(\d+)\s+OF\s+FIELD\s+(\d{2}[A-Z]?)',
            _val, re.IGNORECASE)
        if _clause_ref_m:
            _clause_num = int(_clause_ref_m.group(1))
            _ref_tag = _clause_ref_m.group(2)
            _ref_val = _cf.get(_ref_tag, '')
            if _ref_val:
                _ref_clauses = _split_into_clauses(_ref_tag, _ref_val)
                for _rc in _ref_clauses:
                    if _rc.clause_number == _clause_num:
                        _resolved = _rc.text.strip()
                        _cf[_tag] = _resolved
                        _progress(f"  F{_tag}: resolved clause #{_clause_num} from F{_ref_tag} → {_resolved[:60]}")
                        break
            continue

        # Pattern C1: "REFER TO FIELD 47A(10)" — clause number in parentheses
        _paren_ref_m = re.search(
            r'(?:SEE|REFER\s+(?:TO)?|AS\s+PER)\s+FIELD\s+(\d{2}[A-Z]?)\s*\(\s*(\d+)\s*\)',
            _val, re.IGNORECASE)
        if _paren_ref_m:
            _ref_tag = _paren_ref_m.group(1)
            _clause_num = int(_paren_ref_m.group(2))
            _ref_val = _cf.get(_ref_tag, '')
            if _ref_val:
                _ref_clauses = _split_into_clauses(_ref_tag, _ref_val)
                for _rc in _ref_clauses:
                    if _rc.clause_number == _clause_num:
                        _resolved = _rc.text.strip()
                        _cf[_tag] = _resolved
                        _progress(f"  F{_tag}: resolved ref from F{_ref_tag} clause ({_clause_num}) → {_resolved[:60]}")
                        break
                else:
                    # Clause number not found — try matching by content keyword
                    # e.g., clause 10 might be labelled differently
                    _progress(f"  F{_tag}: clause ({_clause_num}) not found in F{_ref_tag} ({len(_ref_clauses)} clauses)")
            continue

        # P198fr — "REFER FIELD YY CLAUSE N" / "REFER FIELD YY CLAUSE NO. N"
        # MUST be tried BEFORE the bare "REFER FIELD YY" pattern below,
        # otherwise C2's whole-field-replacement fires first and wipes
        # out F48's content (replacing it with the entire F47A body
        # instead of only clause N). Real example: F48 contains
        # "Narrative: /REFER FIELD 47A CLAUSE 10" — this used to
        # collapse F48 to all of F47A. Now it correctly extracts
        # clause 10 only.
        _rev_clause_m = re.search(
            r'REFER\s+FIELD\s+(\d{2}[A-Z]?)\s+CLAUSE\s+(?:NO\.?\s*)?(\d+)',
            _val, re.IGNORECASE)
        if _rev_clause_m:
            _ref_tag = _rev_clause_m.group(1)
            _clause_num = int(_rev_clause_m.group(2))
            _ref_val = _cf.get(_ref_tag, '')
            if _ref_val:
                _ref_clauses = _split_into_clauses(_ref_tag, _ref_val)
                _resolved_clause = None
                for _rc in _ref_clauses:
                    if _rc.clause_number == _clause_num:
                        _resolved_clause = _rc.text.strip()
                        break
                if _resolved_clause:
                    # Replace the reference marker with the resolved
                    # clause text — keep the surrounding F48 wording
                    # ("Period for Presentation ...", "Days: 21",
                    # "Narrative: ") intact.
                    _new_val = (
                        _val[:_rev_clause_m.start()]
                        + _resolved_clause
                        + _val[_rev_clause_m.end():]
                    )
                    _cf[_tag] = re.sub(r'\n{3,}', '\n\n', _new_val).strip()
                    _progress(
                        f"  F{_tag}: P198fr resolved CLAUSE {_clause_num} from "
                        f"F{_ref_tag} (substring replace, kept F{_tag} "
                        f"surrounding text)"
                    )
                    continue

        # Pattern C2: "SEE FIELD YY" / "AS PER FIELD YY" / "REFER TO FIELD YY"
        # (no clause number)
        #
        # P198di — Only apply this whole-field replacement when the
        # current field's content is ESSENTIALLY just a reference
        # ("PLS REFER FIELD 47A", "AS PER FIELD 47A.") — typical of
        # short fields like F48 that say nothing but "see X". When
        # the reference is just one phrase inside a longer clause
        # field (e.g. F47A clause 10: "ALL DOCUMENTS SHOWING
        # DESCRIPTION OF GOODS AS SOYBEANS ACCEPTABLE ONLY INVOICE
        # TO SHOW FULL DESCRIPTION AS PER FIELD 45A"), we MUST NOT
        # replace the entire F47A with F45A's content — doing so
        # wipes out the other 16 conditions.
        # P198di — fixed pre-existing regex: 'REFER\\s+(?:TO)?' required
        # double whitespace when TO was absent, so 'REFER FIELD' (single
        # space, no 'TO') never matched. Now 'REFER(?:\\s+TO)?'.
        _simple_ref_m = re.search(
            r'(?:SEE|REFER(?:\s+TO)?|AS\s+PER)\s+FIELD\s+(\d{2}[A-Z]?)',
            _val, re.IGNORECASE)
        if _simple_ref_m:
            # Heuristic: only treat as a whole-field replacement if
            # stripping the reference leaves essentially no other
            # content (≤ 30 chars after removing the marker and
            # surrounding label words). Long multi-clause fields
            # (≥ 200 chars or with numbered list items "1)" / "2)")
            # are NEVER replaced — the reference is just text inside
            # one of their clauses.
            _val_after = (
                _val[:_simple_ref_m.start()]
                + _val[_simple_ref_m.end():]
            )
            # Strip common label / connective words that surround a
            # bare-reference shape ("Period for Presentation in
            # Days", "Days:", "Narrative:", whitespace, slashes)
            _val_residual = re.sub(
                r'(?:Period\s+for\s+Presentation(?:\s+in\s+Days)?|'
                r'Additional\s+Conditions|Documents\s+Required|'
                r'Description\s+of\s+Goods(?:\s+and/or\s+Services)?|'
                r'Days?\s*:|Narrative\s*:|/|\\|\.|\s)+',
                ' ', _val_after, flags=re.IGNORECASE,
            ).strip()
            _is_multi_clause = bool(
                re.search(r'(?m)^\s*\d+\s*[\)\.]\s', _val)
                or len(_val) > 200
            )
            # P198fr — Additional guard: if the value mentions
            # "CLAUSE <num>" anywhere, the reference is to a SPECIFIC
            # clause within FYY, not the whole field. Don't replace
            # the entire current field with all of FYY's content.
            _has_clause_ref = bool(re.search(
                r'CLAUSE\s+(?:NO\.?\s*)?\d+', _val, re.IGNORECASE))
            if (not _is_multi_clause) and len(_val_residual) <= 30 \
                    and not _has_clause_ref:
                _ref_tag = _simple_ref_m.group(1)
                _ref_val = _cf.get(_ref_tag, '')
                if _ref_val:
                    # Replace the reference with the actual value
                    _cf[_tag] = _ref_val
                    _progress(
                        f"  F{_tag}: resolved simple ref from F{_ref_tag} "
                        f"→ {_ref_val[:60]}"
                    )
            else:
                # Skip — reference is just one phrase inside a
                # larger clause field; do NOT replace the whole
                # field with the referenced field's content.
                _progress(
                    f"  F{_tag}: skipped simple-ref replacement "
                    f"(field has multi-clause content; reference "
                    f"is in-clause text only)"
                )
            continue

        # Pattern D: "REFER FIELD YY CLAUSE NO.X" (reversed order)
        _rev_ref_m = re.search(
            r'REFER\s+FIELD\s+(\d{2}[A-Z]?)\s+CLAUSE\s+NO\.?\s*(\d+)',
            _val, re.IGNORECASE)
        if _rev_ref_m:
            _ref_tag = _rev_ref_m.group(1)
            _clause_num = int(_rev_ref_m.group(2))
            _ref_val = _cf.get(_ref_tag, '')
            if _ref_val:
                _ref_clauses = _split_into_clauses(_ref_tag, _ref_val)
                for _rc in _ref_clauses:
                    if _rc.clause_number == _clause_num:
                        _resolved = _rc.text.strip()
                        _cf[_tag] = _resolved
                        _progress(f"  F{_tag}: resolved ref from F{_ref_tag} clause #{_clause_num} → {_resolved[:60]}")
                        break

    # 2a-bis. Remove junk fields that shouldn't be in the consolidated LC
    # F21 "NONREF" is a filler value — remove it
    if _cf.get('21', '').strip().upper() in ('NONREF', 'NON-REF', 'NONE', 'N/A', ''):
        _cf.pop('21', None)
    # F23 often captures report structure text — remove if it's garbage
    _f23 = _cf.get('23', '')
    if _f23 and (re.search(r'Message\s+Text|Block\s+\d|Report\s+Content', _f23, re.IGNORECASE)
                 or len(_f23) > 100):
        _cf.pop('23', None)
    # F30 is amendment date — only valid in amendment context, not in consolidated LC
    # If it contains garbage (report headers), remove
    _f30 = _cf.get('30', '')
    if _f30 and re.search(r'Report\s+Content|Message\s+Details|Applic', _f30, re.IGNORECASE):
        _cf.pop('30', None)
    # F30 should not be in the final LC (it's amendment metadata)
    _cf.pop('30', None)

    # 2b. Special handling for F48 (Presentation Period) — extract days + resolve reference
    _f48 = _cf.get('48', '')
    if _f48:
        _days_m = re.match(r'(\d+)\s*/?\s*(?:PLS\s+)?REFER', _f48, re.IGNORECASE)
        if _days_m:
            _cf['48'] = _days_m.group(1)
            _progress(f"  F48: extracted {_days_m.group(1)} days from presentation period")

    # P126 — Rescue "DAYS FROM SHIPMENT DATE BUT WITHIN VALIDITY" from any
    # field it leaked into (typically F47A), fold it back into F48. This
    # handles Alliance/Fusion SWIFT exports where the F48 presentation
    # period straddles fields without a proper (CONT FROM FIELD 48) marker.
    _days_pat = re.compile(
        r'(?:^|\n)\s*(?P<num>\d{1,3})?\s*DAYS?\s+FROM\s+(?:DATE\s+OF\s+)?SHIPMENT(?:\s+DATE)?\s+BUT\s+WITHIN\s+(?:THE\s+)?(?:VALIDITY|EXPIRY)(?:\s+OF\s+(?:THIS\s+)?(?:LC|L/C|CREDIT))?[.\s]*',
        re.IGNORECASE,
    )
    _rescued_days = None
    _rescued_num = None
    for _tag in list(_cf.keys()):
        if _tag == '48' or _tag.startswith('_'):
            continue
        _val = _cf.get(_tag, '')
        if not isinstance(_val, str) or not _val:
            continue
        _m = _days_pat.search(_val)
        if not _m:
            continue
        _rescued_days = _m.group(0).strip().rstrip('.').strip()
        _rescued_num = _m.group('num') or ''
        _cf[_tag] = (_val[:_m.start()] + _val[_m.end():]).strip()
        _progress(f"  P126: rescued '{_rescued_days[:60]}' from F{_tag} → F48")
        break
    if _rescued_days:
        _existing_f48 = str(_cf.get('48', '')).strip()
        _has_days_text = bool(re.search(r'DAYS?\s+FROM\s+SHIPMENT', _existing_f48, re.IGNORECASE))
        _has_num = bool(re.search(r'\d+', _existing_f48))
        if not _has_days_text:
            if _rescued_num and not _has_num:
                _cf['48'] = f"{_rescued_num} {_rescued_days if 'DAYS' in _rescued_days.upper() else 'DAYS ' + _rescued_days}".strip()
            elif _has_num and not re.search(r'DAYS?\b', _existing_f48, re.IGNORECASE):
                _cf['48'] = f"{_existing_f48} {_rescued_days}".strip()
            else:
                _cf['48'] = f"{_existing_f48}\n{_rescued_days}".strip() if _existing_f48 else _rescued_days

    # P198df — Display normalisation for F48. SWIFT BAHL notation
    # writes the period as "15/FRM SHIPMENT DATE BUT WITH IN EXPIRY"
    # — a slash form with "FRM" / "WITH IN" abbreviations that
    # reads awkwardly in the final LC report. Rewrite it to a
    # clean English form ("15 days from shipment date but within
    # expiry") so the consolidated final LC is readable, while
    # keeping the original numeric value and intent intact. Skips
    # already-clean wording (e.g. "21 DAYS FROM SHIPMENT DATE").
    try:
        _f48_v = str(_cf.get('48', '') or '').strip()
        if _f48_v:
            _norm = _f48_v
            # Slash form: "15/FROM SHIPMENT DATE BUT WITHIN EXPIRY"
            #         or "21/FRM SHIPMENT DATE BUT WITH IN EXPIRY"
            _slash_m = re.match(
                r'^\s*(\d{1,3})\s*/\s*(FROM|FRM)\s+(.+)$',
                _norm,
                flags=re.IGNORECASE,
            )
            if _slash_m:
                _norm = (
                    f"{_slash_m.group(1)} days from "
                    f"{_slash_m.group(3).strip()}"
                )
            # Abbreviation cleanup that applies regardless of form:
            #   FRM    → from
            #   WITH IN → within
            #   B/L DATE / SHIPMENT DATE: leave date words intact
            _norm = re.sub(r'\bFRM\b', 'from', _norm, flags=re.IGNORECASE)
            _norm = re.sub(r'\bWITH\s+IN\b', 'within', _norm, flags=re.IGNORECASE)
            # Drop the ALL-CAPS look so it reads naturally. Only
            # rewrite when the source was the BAHL slash/abbrev form
            # (i.e. we actually changed something) to avoid touching
            # already-clean wording.
            if _norm != _f48_v:
                # Title-cased English with a leading number form.
                _final_lower = _norm.lower()
                _cf['48'] = _final_lower
                _progress(
                    f"  P198df: F48 reformatted "
                    f"{_f48_v!r} -> {_final_lower!r}"
                )
    except Exception as _e:
        try:
            _progress(f"  P198df F48 reformat exception: {_e}")
        except Exception:
            pass

    # 2c. Run full cleanup on ALL consolidated fields (not just amended ones)
    for _tag in list(_cf.keys()):
        if _tag.startswith('_'):
            continue
        _val = _cf[_tag]
        if isinstance(_val, str):
            _cleaned = _clean_consolidated_field_value(_tag, _val)
            if _cleaned != _val:
                _cf[_tag] = _cleaned

    # 3. Clean junk: URLs, pagination, "Select 'Print' to output..."
    for _tag in list(_cf.keys()):
        if _tag.startswith('_'):
            continue
        _val = _cf[_tag]
        if isinstance(_val, str):
            # Remove URLs
            _val = re.sub(r'https?://\S+', '', _val)
            # Remove "Select 'Print' to output..."
            _val = re.sub(r"Select\s+'Print'\s+to\s+output.*", '', _val, flags=re.IGNORECASE)
            # Remove pagination: "1/1", "2/2", "Page X of Y", "SWIFT_MT7012/2"
            _val = re.sub(r'\bSWIFT_MT\d+/?\d*', '', _val)
            _val = re.sub(r'\n\s*\d+/\d+\s*$', '', _val)
            # Remove IP addresses
            _val = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}:\d+\S*', '', _val)
            # Collapse multiple blank lines
            _val = re.sub(r'\n{3,}', '\n\n', _val)
            _cf[_tag] = _val.strip()

    # 4. Reformat amount if "Increase" label is present
    _amt = _cf.get('32B', '')
    if 'increase' in _amt.lower():
        _amt = re.sub(r'(?i)^Increase\s+of\s+Documentary\s+Credit\s+Amount\s*[\n\r]*', '', _amt).strip()
        _cf['32B'] = _amt

    # 4b. P66: Final 32B hardening pass.
    # Some extraction paths (Alliance OCR fallback, VLM fallback, multi-page
    # SWIFT continuation) write 32B without going through the inline cleanup
    # at extraction time. This catches any residual bleeding such as:
    #   "USD US DOLLAR 97216,00 #97,216.00F41D: Available With..."
    #   "USD\nUS DOLLAR\n516000,00\n#516,000.00\nF41D ..."
    #   "F32B: USD 97,216.00"
    # The output is always normalised to "{CCY} {amount with US thousands
    # format}" — e.g. "USD 97,216.00".
    _amt = _cf.get('32B', '')
    if _amt and isinstance(_amt, str):
        _v = _amt
        # Strip a leading "F32B:" / "32B:" label
        _v = re.sub(r'^\s*F?32B\s*[:\-]\s*', '', _v, flags=re.IGNORECASE)
        # Truncate at any downstream F-tag header glued in (F41D, F39A, etc.)
        _next = re.search(r'\bF?\d{2}[A-Z]?\s*:', _v)
        if _next and _next.start() > 0:
            _v = _v[:_next.start()]
        # Strip the spelt-out currency word(s) — keep just the ISO code
        _v = re.sub(
            r'\b(?:US\s*DOLLAR|US\s*DOLLARS|DOLLAR|DOLLARS|EURO|EUROS|POUND\s*STERLING|'
            r'POUNDS|JAPANESE\s*YEN|YEN|FRANC|FRANCS|RUPEE|RUPEES|YUAN|RIYAL|DIRHAM)\b',
            '', _v, flags=re.IGNORECASE,
        )
        # Strip "#" separators
        _v = _v.replace('#', ' ')
        # Find ISO currency code (3 uppercase letters, default USD)
        _ccy_m = re.search(r'\b([A-Z]{3})\b', _v)
        _ccy_str = _ccy_m.group(1) if _ccy_m else 'USD'
        # Try to find the amount in any common format
        _amt_value = None
        # 1. US format: 97,216.00 / 1,234,567.89
        _us = re.search(r'(\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?)', _v)
        if _us:
            try:
                _amt_value = float(_us.group(1).replace(',', ''))
            except ValueError:
                pass
        # 2. European format: 97216,00 / 1.234.567,89
        if _amt_value is None:
            _eu = re.search(r'(\d{1,3}(?:\.\d{3})+,\d{1,2})', _v)
            if _eu:
                try:
                    _amt_value = float(_eu.group(1).replace('.', '').replace(',', '.'))
                except ValueError:
                    pass
        # 3. Plain digits with European decimal comma: 97216,00
        if _amt_value is None:
            _eu2 = re.search(r'(\d+,\d{2})\b', _v)
            if _eu2:
                try:
                    _amt_value = float(_eu2.group(1).replace(',', '.'))
                except ValueError:
                    pass
        # 4. Plain digits with US decimal point: 97216.00
        if _amt_value is None:
            _us2 = re.search(r'(\d+\.\d{2})\b', _v)
            if _us2:
                try:
                    _amt_value = float(_us2.group(1))
                except ValueError:
                    pass
        # 5. Bare integer: 97216
        if _amt_value is None:
            _int = re.search(r'\b(\d{3,})\b', _v)
            if _int:
                try:
                    _amt_value = float(_int.group(1))
                except ValueError:
                    pass
        if _amt_value is not None and _amt_value > 0:
            _cf['32B'] = f"{_ccy_str} {_amt_value:,.2f}"

    # -- Split clause fields --
    for tag in CLAUSE_TAGS:
        value = final_lc.consolidated_fields.get(tag, '')
        if value:
            clause_list = _split_into_clauses(tag, value)
            if clause_list:
                final_lc.clauses[tag] = clause_list
                _progress(f"  F{tag} ({SWIFT_FIELD_LABELS.get(tag, '')}): {len(clause_list)} clauses")

    # -- Extract MT799 narrative (only if it contains amendment instructions) --
    for pkt in mt799_packets:
        text = _get_packet_refined_text(pkt)
        pkt_id = _get_packet_field(pkt, 'packet_id', 0)
        # Only store MT799s that reference LC fields or contain amendment instructions
        _has_amendment_content = bool(re.search(
            r'FIELD\s+\d{2}[A-Z]|UNDER\s+FIELD|TO\s+READ\s+AS|I/O\s+EXISTING|PLEASE\s+AMEND',
            text, re.IGNORECASE
        ))
        if _has_amendment_content:
            final_lc.source_packets.append(pkt_id)
            idx = len([k for k in final_lc.consolidated_fields if k.startswith('799_')])
            final_lc.consolidated_fields[f'799_{idx + 1}'] = text
            _progress(f"  MT799 packet {pkt_id}: stored as 799_{idx + 1} (has amendment content)")
        else:
            _progress(f"  MT799 packet {pkt_id}: skipped (no amendment content)")

    final_lc.warnings = warnings

    elapsed = time.time() - start_time
    _progress(f"Step 6 complete: DC# {final_lc.dc_number}, "
              f"{len(final_lc.consolidated_fields)} fields, "
              f"{final_lc.amendment_count} amendments, "
              f"{sum(len(v) for v in final_lc.clauses.values())} total clauses, "
              f"{elapsed:.1f}s")

    # -- Save results --
    result_file = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        result_file = os.path.join(output_dir, 'step06_result.json')

        clauses_serialized = {}
        for tag, clause_list in final_lc.clauses.items():
            clauses_serialized[tag] = [asdict(c) for c in clause_list]

        amendment_log_serialized = [asdict(a) for a in final_lc.amendment_log]

        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({
                'step': 6,
                'step_name': 'Final LC Consolidation',
                'dc_number': final_lc.dc_number,
                'swift_format': final_lc.swift_format,
                'total_fields': len(final_lc.consolidated_fields),
                'amendment_count': final_lc.amendment_count,
                'total_clauses': sum(len(v) for v in final_lc.clauses.values()),
                'elapsed_seconds': round(elapsed, 2),
                'warnings': final_lc.warnings,
                'consolidated_fields': final_lc.consolidated_fields,
                'original_fields': final_lc.original_fields,
                'clauses': clauses_serialized,
                'amendment_log': amendment_log_serialized,
                'source_packets': final_lc.source_packets,
                'shipping_packets_count': len(shipping_packets),
                'other_packets_count': len(other_packets),
            }, f, indent=2, ensure_ascii=False)

    return {
        'final_lc': final_lc,
        'shipping_packets': shipping_packets,
        'other_packets': other_packets + other_mt_packets,
        'elapsed_seconds': round(elapsed, 2),
        'errors': [],
        'warnings': warnings,
        'result_file': result_file,
    }


if __name__ == '__main__':
    import sys as _sys
    if len(_sys.argv) < 2:
        print("Usage: python step06_final_lc.py <step05_result.json>")
        _sys.exit(1)
    with open(_sys.argv[1], 'r', encoding='utf-8') as f:
        step5 = json.load(f)
    result = run(step5, output_dir=os.path.dirname(_sys.argv[1]))
    flc = result['final_lc']
    print(f"\nFinal LC: DC# {flc.dc_number}")
    print(f"  Fields: {len(flc.consolidated_fields)}")
    print(f"  Amendments: {flc.amendment_count}")
    print(f"  Clauses: {sum(len(v) for v in flc.clauses.values())}")
    print(f"  Elapsed: {result['elapsed_seconds']}s")
