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
    '20', '23', '26E', '27', '30', '31C', '31D', '32B', '33B', '34B',
    '39A', '39B', '39C', '40A', '40E', '41A', '41D', '42A', '42C',
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
    _normalized = re.sub(
        r'(?<=[.;:!?\s])\s+(\d{1,2})\s*([.\-)])\s*(?=[A-Z])',
        r'\n\1\2 ',
        _normalized,
    )
    numbered = re.split(r'\n\s*(\d+)\s*[-.)]\s*', '\n' + _normalized)
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
        new_content = re.sub(r'^Lines?\d*(?:to\d+)?:\s*', '', new_content, flags=re.MULTILINE).strip()
        new_content = re.sub(r'^\s*Code\s*-\s*Narrative\s*$', '', new_content, flags=re.MULTILINE).strip()
        # If it contains PLEASE READ patterns, don't do full replace — let patterns handle it
        if not re.search(r'PLEASE\s+READ', new_content, re.IGNORECASE):
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

    return result


# == Amendment application ====================================================

# Label-strip patterns reused by both base extraction and amendment cleanup.
# Each entry strips the leading label that the SWIFT regex captured along
# with the actual value (e.g. "Latest Date of Shipment\n251030 Oct 30").
_FIELD_LABEL_STRIP = {
    '20':  r'^(?:Documentary\s+Credit\s+Number|Sender\'?s?\s+Reference)\s*[\n\r]*',
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
    '57A': r'^[\'"]?Advise\s+Through[\'"]?\s+Bank.*?(?:Identifier\s+Code)?\s*[\n\r]*',
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

    # 6. SWIFT message footer ("Other\nDelivery overdue / Network delivery /
    # Payment Confirmation" section that bleeds into the LAST tag on the
    # message — typically F44C, F47A, F78). Strip everything from "Other"
    # to end of value.
    v = re.sub(
        r'\n\s*Other\s*\n\s*(?:Delivery\s+overdue|Network\s+delivery|Payment\s+Confirmation).*$',
        '', v, flags=re.IGNORECASE | re.DOTALL).strip()

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

    # 8. (CONT FROM FIELD ...) cross-references
    v = re.sub(r'\(CONT\s+FROM\s+FIELD\s+\w+\)', '', v, flags=re.IGNORECASE).strip()
    v = re.sub(r'/\(CONT\s+FROM\s+FIELD\s+\w+\)', '', v, flags=re.IGNORECASE).strip()

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
        _dm = re.search(r'(\d{6})\s+(\d{4})\s+(\w{3})\s+(\d{1,2})', v)
        if _dm:
            _months = {'Jan':'01','Feb':'02','Mar':'03','Apr':'04','May':'05','Jun':'06',
                       'Jul':'07','Aug':'08','Sep':'09','Oct':'10','Nov':'11','Dec':'12'}
            _date_str = f"{_dm.group(2)}-{_months.get(_dm.group(3),'01')}-{int(_dm.group(4)):02d}"
            v = (v[:_dm.start()] + _date_str + v[_dm.end():]).strip()
        # Remove leftover raw 6-digit SWIFT date codes
        v = re.sub(r'\b\d{6}\b\s*', '', v).strip()

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
        _ccy = re.search(r'\b([A-Z]{3})\b', v)
        _ccy_str = _ccy.group(1) if _ccy else 'USD'
        _am = re.search(r'#([\d,]+\.\d+)#?', v)
        if _am:
            v = f"{_ccy_str} {_am.group(1)}"
        else:
            _am2 = re.search(r'(\d[\d.]*,\d{2})\b', v)
            if _am2:
                _amt = _am2.group(1).replace('.', '').replace(',', '.')
                try:
                    v = f"{_ccy_str} {float(_amt):,.2f}"
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
        if tag == '34B':
            old_val = base_fields.get('32B', '')
            new_val = sf.value
            # Strip "Increase of Documentary Credit Amount" label
            _is_increase = bool(re.search(r'increase', new_val, re.IGNORECASE))
            new_val = re.sub(r'(?i)^(?:Increase\s+of\s+Documentary\s+Credit\s+Amount|'
                             r'Currency\s+Code,?\s*Amount)\s*[\n\r]*', '', new_val).strip()
            if _is_increase and old_val:
                # Parse and add amounts
                _old_amt = re.search(r'([\d,]+[.,]\d+)', old_val.replace(' ', ''))
                _new_amt = re.search(r'([\d,]+[.,]\d+)', new_val.replace(' ', ''))
                _ccy = re.search(r'\b([A-Z]{3})\b', old_val) or re.search(r'\b([A-Z]{3})\b', new_val)
                if _old_amt and _new_amt and _ccy:
                    _o = float(_old_amt.group(1).replace(',', ''))
                    _n = float(_new_amt.group(1).replace(',', '').replace('.', '.'))
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
        # Normalize double slashes: //REPALL// -> /REPALL/
        amd_val = re.sub(r'//', '/', amd_val)
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
                or _is_to_read_as):
            # This is an amendment instruction — apply operations to base value
            new_val = _apply_text_amendment(old_val, amd_val)
            base_fields[actual_tag] = new_val
            if old_val != new_val:
                record.fields_changed.append(actual_tag)
                record.change_details[actual_tag] = {'old': old_val, 'new': new_val, 'operation': 'text_amendment'}
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

        resp = requests.post(QWEN_VLM_URL, json=payload, timeout=VLM_TIMEOUT)
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
            '20': r'^(?:Documentary\s+Credit\s+Number|Sender\'?s?\s+Reference)\s*[\n\r]*',
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
            '57A': r'^[\'"]?Advise\s+Through[\'"]?\s+Bank.*?(?:Identifier\s+Code)?\s*[\n\r]*',
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

            # Strip "(CONT FROM FIELD ...)" cross-references
            sf.value = re.sub(r'\(CONT\s+FROM\s+FIELD\s+\w+\)', '', sf.value, flags=re.IGNORECASE).strip()
            sf.value = re.sub(r'/\(CONT\s+FROM\s+FIELD\s+\w+\)', '', sf.value, flags=re.IGNORECASE).strip()

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
            # Remove raw SWIFT date codes like "260131" if already converted
            if sf.tag in ('31C', '31D', '44C'):
                sf.value = re.sub(r'\b\d{6}\b\s*', '', sf.value).strip()

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
        _raw_dc = re.sub(r"(?i)^(?:Sender'?s?\s+Reference|Documentary\s+Credit\s+Number)\s*[\n\r]*", '', _raw_dc).strip()
        final_lc.dc_number = _raw_dc
        final_lc.consolidated_fields['20'] = _raw_dc
        final_lc.source_packets.append(_get_packet_field(base_pkt, 'packet_id', 0))

        # If multiple MT700 packets, extract from subsequent ones too
        # (multi-page LCs where each page was a separate packet)
        for extra_pkt in mt700_packets[1:]:
            extra_text = _get_packet_refined_text(extra_pkt)
            extra_page = _get_packet_first_page(extra_pkt)
            extra_fields = _extract_swift_fields(extra_text, source_page=extra_page, source_mt='MT700')
            for sf in extra_fields:
                if sf.tag not in final_lc.consolidated_fields:
                    final_lc.consolidated_fields[sf.tag] = sf.value
                    final_lc.original_fields[sf.tag] = sf.value
                    _progress(f"    F{sf.tag} (from extra MT700): {sf.value[:60]}...")
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

            record = _apply_amendment(
                final_lc.consolidated_fields,
                amd_fields,
                amendment_number=i + 1,
                source_packet_id=pkt_id,
            )
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
    _cont_marker_re = re.compile(
        r'[/\\]?\s*'
        r'\(\s*CONT(?:INUED|INUATION)?\s+FROM\s+FIELD\s+(?P<src>\d{2}[A-Z]?)\s*\)\s*'
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
            r'(?:^|\n)\s*[/\\]?\s*\(\s*CONT(?:INUED|INUATION)?\s+FROM\s+FIELD\s+\d{2}[A-Z]?\s*\)\s*',
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
        _clause_ref_m = re.search(
            r'(?:PLS\s+)?REFER\s+(?:TO\s+)?CLAUSE\s+NO\.?\s*(\d+)\s+OF\s+FIELD\s+(\d{2}[A-Z]?)',
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
                        _cf[_tag] = f"{_val.strip()}\n[Resolved: {_resolved}]"
                        _progress(f"  F{_tag}: resolved clause #{_clause_num} from F{_ref_tag} → {_resolved[:60]}")
                        break
            continue

        # Pattern C: "SEE FIELD YY" / "AS PER FIELD YY" / "REFER TO FIELD YY" (no clause number)
        _simple_ref_m = re.search(
            r'(?:SEE|REFER\s+(?:TO)?|AS\s+PER)\s+FIELD\s+(\d{2}[A-Z]?)',
            _val, re.IGNORECASE)
        if _simple_ref_m:
            _ref_tag = _simple_ref_m.group(1)
            _ref_val = _cf.get(_ref_tag, '')
            if _ref_val:
                # Replace the reference with the actual value
                _cf[_tag] = _ref_val
                _progress(f"  F{_tag}: resolved simple ref from F{_ref_tag} → {_ref_val[:60]}")
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
                        _cf[_tag] = f"{_val.strip()}\n[Resolved: {_resolved}]"
                        _progress(f"  F{_tag}: resolved ref from F{_ref_tag} clause #{_clause_num} → {_resolved[:60]}")
                        break

    # 2b. Special handling for F48 (Presentation Period) — extract days + resolve reference
    _f48 = _cf.get('48', '')
    if _f48:
        _days_m = re.match(r'(\d+)\s*/?\s*(?:PLS\s+)?REFER', _f48, re.IGNORECASE)
        if _days_m:
            _cf['48'] = _days_m.group(1)
            _progress(f"  F48: extracted {_days_m.group(1)} days from presentation period")

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

    # -- Extract MT799 narrative --
    for pkt in mt799_packets:
        text = _get_packet_refined_text(pkt)
        pkt_id = _get_packet_field(pkt, 'packet_id', 0)
        final_lc.source_packets.append(pkt_id)
        idx = len([k for k in final_lc.consolidated_fields if k.startswith('799_')])
        final_lc.consolidated_fields[f'799_{idx + 1}'] = text
        _progress(f"  MT799 packet {pkt_id}: stored as 799_{idx + 1}")

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
