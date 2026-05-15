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

@dataclass
class SwiftField:
    """A single extracted SWIFT field from an LC message."""
    tag: str
    label: str
    value: str
    source_page: int = 0
    source_mt: str = ""

    mt799_replace_anchor: Optional[str] = None

@dataclass
class Clause:
    """
    A single clause extracted from a multi-clause SWIFT field.
    e.g. F46A clause 1: "COMMERCIAL INVOICE IN 3 ORIGINALS"
    """
    clause_number: int
    clause_id: str
    text: str
    parent_tag: str

    section: str = ''
    is_section_header: bool = False

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

SWIFT_FIELD_LABELS = {
    '20': 'Documentary Credit Number',
    '21': 'Related Reference',

    '22A': 'Purpose of Message',
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

    '58A': 'Requested Confirmation Party',
    '59': 'Beneficiary',
    '71D': 'Charges',
    '72': 'Sender to Receiver Information',
    '78': 'Instructions to Paying/Accepting/Negotiating Bank',
    '79': 'Narrative',
}

CLAUSE_TAGS = {'45A', '45B', '46A', '46B', '47A', '47B', '78', '79', '72'}

EXTRACTION_TAGS = [

    '20', '21', '22A', '23', '26E', '27', '30', '31C', '31D', '32B', '33B',
    '34B',
    '39A', '39B', '39C', '40A', '40E', '41A', '41D', '42A', '42C', '42D',
    '42M', '42P', '43P', '43T', '44A', '44B', '44C', '44D', '44E', '44F',
    '45A', '45B', '46A', '46B', '47A', '47B', '48', '49', '50', '51A',

    '51D', '52A', '52D', '53A', '57A', '57D', '58A', '59', '71D', '72', '78', '79',
]

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

        patterns.append({
            'tag': tag,
            'format': 'alliance',
            'regex': re.compile(
                r'(?:^|\n)\s*:' + re.escape(tag) + r':\s*(.*?)(?=\n\s*:[A-Z0-9]{2,4}:|\Z)',
                re.DOTALL,
            ),
        })

        patterns.append({
            'tag': tag,
            'format': 'fusion',
            'regex': re.compile(
                r'(?:^|\n)\s*F' + re.escape(tag) + r'\s*:\s*(.*?)(?=\n\s*F[A-Z0-9]{2,4}\s*:|\Z)',
                re.DOTALL,
            ),
        })

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

    text = text.replace('\uff1a', ':')

    _TS_PROTECT = '___TIMECOLON___'
    text = re.sub(
        r'(\d{1,2}):(\d{2}):(\d{2})',
        lambda _m: f"{_m.group(1)}{_TS_PROTECT}{_m.group(2)}{_TS_PROTECT}{_m.group(3)}",
        text,
    )

    text = re.sub(r'([^\n])(F\d{2}[A-Z]?\s*:)', r'\1\n\2', text)

    text = re.sub(r'([^\n])(:\d{2}[A-Z]?:)', r'\1\n\2', text)

    # P278 \u2014 Alliance UI label-injection. The Alliance Lite UI export
    # format prints LC field labels as standalone lines (e.g. "Documents
    # Required", "Partial Shipments", "Description of Goods and/or
    # Services") and the value on the following line(s). Some LCs are
    # MIXED: F-tags present for some fields (F20, F31C, F40A, F42D,
    # F47A) but later fields like F43P, F43T, F44E/F, F44C, F45A, F46A
    # use label-only sections with no F-prefix. The regex extractor
    # then greedily consumes everything between the last F-tag and the
    # NEXT F-tag, swallowing the label-only fields. Worst case: F42D
    # eats F43P/T, F44, F45A, F46A \u2014 losing the documents-required
    # list entirely and breaking missing-doc detection downstream.
    #
    # Fix: detect whole-line Alliance UI labels and inject the matching
    # F-tag prefix BEFORE the regex split. Only fires when the label
    # occupies an entire line (possibly with surrounding whitespace /
    # trailing punctuation) so it doesn't false-positive on clause-body
    # phrases like "Partial Shipments are not allowed under this LC".
    _ALLIANCE_LABEL_TO_TAG = [
        # Order: longest / most specific labels FIRST so they win over
        # generic ones (e.g. "Description of Goods and/or Services"
        # before bare "Description of Goods").
        ('Sender to Receiver Information',                          'F72'),
        ('Instructions to the Paying/Accepting/Negotiating Bank',   'F78'),
        ('Instructions to the Paying or Accepting or Negotiating Bank', 'F78'),
        ('Instructions to the Negotiating Bank',                    'F78'),
        ('Instructions to the Accepting Bank',                      'F78'),
        ('Instructions to the Paying Bank',                         'F78'),
        ('Port of Discharge/Airport of Destination',                'F44F'),
        ('Port of Loading/Airport of Departure',                    'F44E'),
        ('Description of Goods and/or Services',                    'F45A'),
        ('Description of Goods and or Services',                    'F45A'),
        ('Percentage Credit Amount Tolerance',                      'F39A'),
        ('Date and Place of Expiry',                                'F31D'),
        ('Available With Bank By Negotiation',                      'F41D'),
        ('Available With Bank By',                                  'F41D'),
        ('Available With ... By ...',                               'F41A'),
        ('Available With By',                                       'F41A'),
        ('Available With',                                          'F41A'),
        ('Place of Final Destination',                              'F44B'),
        ('Place of Taking in Charge',                               'F44A'),
        ('Form of Documentary Credit',                              'F40A'),
        ('Documentary Credit Number',                               'F20'),
        ('Latest Date of Shipment',                                 'F44C'),
        ('Period for Presentation in Days',                         'F48'),
        ('Period for Presentation',                                 'F48'),
        ('Confirmation Instructions',                               'F49'),
        ('Requested Confirmation Party',                            'F58A'),
        ('Advising Through Bank',                                   'F57A'),
        ('Reimbursing Bank',                                        'F53A'),
        ('Applicant Bank',                                          'F51A'),
        ('Issuing Bank',                                            'F52A'),
        ('Additional Conditions',                                   'F47A'),
        ('Documents Required',                                      'F46A'),
        ('Description of Goods',                                    'F45A'),
        ('Partial Shipments',                                       'F43P'),
        ('Partial Shipment',                                        'F43P'),
        ('Transshipment',                                           'F43T'),
        ('Transhipment',                                            'F43T'),
        ('Port of Discharge',                                       'F44F'),
        ('Port of Loading',                                         'F44E'),
        ('Applicable Rules',                                        'F40E'),
        ('Currency Code, Amount',                                   'F32B'),
        ('Currency Code Amount',                                    'F32B'),
        ('Currency Amount',                                         'F32B'),
        ('Date of Issue',                                           'F31C'),
        ('Drafts at',                                               'F42C'),
        ('Drawee',                                                  'F42D'),
        ('Applicant',                                               'F50'),
        ('Beneficiary',                                             'F59'),
        ('Charges',                                                 'F71D'),
    ]
    # Don't run injection if the text is ALREADY fully F-tagged or
    # Alliance-colon-tagged (extracting would already work).
    _f_tag_count = len(re.findall(r'\bF\d{2}[A-Z]?\s*:', text))
    _colon_tag_count = len(re.findall(r'\n\s*:\d{2}[A-Z]?:', text))
    if _f_tag_count < 20 and _colon_tag_count < 20:
        _present_tags = set(re.findall(r'\bF(\d{2}[A-Z]?)\s*:', text))
        _injection_count = 0
        # Pad with leading \n so the first line can match too.
        text_buf = '\n' + text
        for _label, _ftag in _ALLIANCE_LABEL_TO_TAG:
            _tag_num = _ftag[1:]
            if _tag_num in _present_tags:
                continue  # tag already in the text, don't double-inject
            # Whole-line label match: line must contain ONLY the label
            # (case-insensitive), possibly with trailing punctuation /
            # whitespace. \s+ tolerates OCR spacing variance inside.
            _label_pat = re.escape(_label).replace(r'\ ', r'\s+')
            _line_re = re.compile(
                r'(?<=\n)[ \t]*'
                + _label_pat
                + r'[ \t.,:;]*\n',
                flags=re.IGNORECASE,
            )
            text_buf, _n = _line_re.subn(_ftag + ': \n', text_buf, count=1)
            if _n > 0:
                _injection_count += 1
                _present_tags.add(_tag_num)
        # Apply only if we injected at least 3 \u2014 fewer suggests this
        # isn't really an Alliance UI page and the false-positive risk
        # outweighs the gain.
        if _injection_count >= 3:
            text = text_buf.lstrip('\n')

    _TRUNCATED_TAG_FIXES = [
        (r'(?<=\n)\s*\d{0,2}D:\s*(?=Date\s+and\s+Place\s+of\s+Expiry)',   'F31D: '),
        (r'(?<=\n)\s*\d{0,2}C:\s*(?=Date\s+of\s+Issue)',                   'F31C: '),
        (r'(?<=\n)\s*\d{0,2}B:\s*(?=Currency\s+Code)',                     'F32B: '),
        (r'(?<=\n)\s*\d{0,2}A:\s*(?=Form\s+of\s+Documentary\s+Credit)',    'F40A: '),
        (r'(?<=\n)\s*\d{0,2}A:\s*(?=Available\s+With)',                    'F41A: '),
        (r'(?<=\n)\s*\d{0,2}D:\s*(?=Available\s+With)',                    'F41D: '),
        (r'(?<=\n)\s*\d{0,2}C:\s*(?=Drafts\s+at)',                         'F42C: '),
        (r'(?<=\n)\s*\d{0,2}D:\s*(?=Drawee)',                              'F42D: '),
        (r'(?<=\n)\s*\d{0,2}P:\s*(?=Partial\s+Shipment)',                  'F43P: '),
        (r'(?<=\n)\s*\d{0,2}T:\s*(?=Transship)',                           'F43T: '),
        (r'(?<=\n)\s*\d{0,2}A:\s*(?=(?:Place|Port)\s+of\s+(?:Loading|Taking))', 'F44A: '),
        (r'(?<=\n)\s*\d{0,2}E:\s*(?=Port\s+of\s+(?:Loading|Discharge))',  'F44E: '),
        (r'(?<=\n)\s*\d{0,2}F:\s*(?=Port\s+of\s+Discharge)',              'F44F: '),
        (r'(?<=\n)\s*\d{0,2}B:\s*(?=Place\s+of\s+Final\s+Destination)',   'F44B: '),
        (r'(?<=\n)\s*\d{0,2}D:\s*(?=Charges)',                             'F71D: '),
    ]
    for _pat, _repl in _TRUNCATED_TAG_FIXES:
        text = re.sub(_pat, _repl, text, count=1, flags=re.IGNORECASE)

    text = text.replace(_TS_PROTECT, ':')

    fields = []
    found_tags = set()

    for pat in _TAG_PATTERNS:
        if pat['tag'] in found_tags:
            continue
        m = pat['regex'].search(text)
        if m:
            value = m.group(1).strip()

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
_MT799_CANON_TAGS = frozenset(_MT799_TAG_TO_FIELD.values())

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

    norm = text
    norm = norm.replace('\u2018', "'").replace('\u2019', "'")
    norm = norm.replace('\u201c', '"').replace('\u201d', '"')
    norm = re.sub(r"'{2,}", "'", norm)
    norm = re.sub(r'"{2,}', '"', norm)

    out: List[SwiftField] = []
    seen_tags = set()

    def _record(canon_tag, new_val, old_val):

        if canon_tag not in _MT799_CANON_TAGS:
            return
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

    pat_field_quoted = re.compile(
        r'(?:UNDER\s+)?FIELD\s+(\d{2}[A-Z]?)\b[^\n]{0,80}?'
        r'(?:SHOULD|SHALL|NOW|TO)\s+READ\s+AS'
        r'[\s\n\r]*'
        r'[\'"]\s*([^\'"\n\r]+?)\s*[\'"]'
        r'\s*(?:I\s*/\s*[OE]|INSTEAD\s+OF)\s*'
        r'[\'"]\s*([^\'"\n\r]+?)\s*[\'"]',
        re.IGNORECASE,
    )
    for m in pat_field_quoted.finditer(norm):
        raw_tag = m.group(1).upper().strip()
        new_val = m.group(2).strip()
        old_val = m.group(3).strip()
        canon_tag = _MT799_TAG_TO_FIELD.get(raw_tag, raw_tag)
        _record(canon_tag, new_val, old_val)

    _SINGLE_VAL_TAGS = {'31D', '42C', '44C', '44E', '44F', '32B', '39A'}
    _CLAUSE_TAGS = {'45A', '46A', '47A'}
    pat_field_no_old = re.compile(
        r'(?:UNDER\s+)?FIELD\s+(\d{2}[A-Z]?)\b[^\n]{0,80}?'
        r'(?:SHOULD|SHALL|NOW|TO)\s+READ\s+AS'
        r'[\s\n\r]*'
        r'[\'"]\s*([^\'"\n\r]+?)\s*[\'"]'
        r'\s*(?:I\s*/\s*[OE]|INSTEAD\s+OF\s+EXISTING)\s*'
        r'(?=\s*(?:[\.\n\r]|$|REGARDS\b|THANKS\b))',
        re.IGNORECASE,
    )
    for m in pat_field_no_old.finditer(norm):
        raw_tag = m.group(1).upper().strip()
        new_val = m.group(2).strip()
        canon_tag = _MT799_TAG_TO_FIELD.get(raw_tag, raw_tag)

        if canon_tag not in _SINGLE_VAL_TAGS or canon_tag in seen_tags:
            continue
        seen_tags.add(canon_tag)
        out.append(SwiftField(
            tag=canon_tag,
            label=SWIFT_FIELD_LABELS.get(canon_tag, f'Field {canon_tag}'),
            value=new_val,
            source_page=source_page,
            source_mt=source_mt,
        ))

    pat_field_unquoted = re.compile(
        r'(?:UNDER\s+)?FIELD\s+(\d{2}[A-Z]?)\b[^\n]{0,80}?'
        r'(?:SHOULD|SHALL|NOW|TO)\s+READ\s+AS\s+'
        r'([^\n\r\'"]{1,200}?)'
        r'\s+(?:I\s*/\s*[OE]|INSTEAD\s+OF)\s+'
        r'([^\n\r\'"]{1,200}?)'
        r'(?=\s*(?:[\n\r]|$|REGARDS\b|THANKS\b))',
        re.IGNORECASE,
    )
    for m in pat_field_unquoted.finditer(norm):
        raw_tag = m.group(1).upper().strip()
        new_val = m.group(2).strip().rstrip('.,;')
        old_val = m.group(3).strip().rstrip('.,;')
        canon_tag = _MT799_TAG_TO_FIELD.get(raw_tag, raw_tag)
        _record(canon_tag, new_val, old_val)

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
        r'\s+(?:I\s*/\s*[OE]|INSTEAD\s+OF)\s+'
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

    if '47A' not in seen_tags:
        m = re.search(
            r'PLEASE\s+(?:AMEND|CHANGE|CORRECT|REPLACE)\s+(.{5,80}?)\s+'
            r'TO\s+READ\s+AS[\s\n\r]*["\']?\s*([^\'"\n\r]+?)\s*["\']?\s*'
            r'(?:I\s*/\s*[OE]|INSTEAD\s+OF)\s*["\']?\s*([^\'"\n\r]+?)\s*["\']?'
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

    pat_bahl_under = re.compile(
        r'(?:^|\n)\s*\+\s*FIELD\s+(\d{2}[A-Z]?)\b[^\n]*?'
        r'TO\s+BE\s+READ\s+AS\s+UNDER\s+'
        r'I\s*/?\s*[OE]\s+EXISTING\s*\n'
        r'(.+?)'
        r'(?=\n\s*\+\s*FIELD\b'
        r'|\n\s*\.?\s*\n\s*(?:REGRET|PLEASE\s+(?:INFORM|NOTE)|REGARDS|CENTRALIZED|YOURS\s+(?:TRULY|FAITHFULLY)|BANK\s+AL\s+HABIB)\b'
        r'|\Z)',
        re.IGNORECASE | re.DOTALL,
    )
    for m in pat_bahl_under.finditer(norm):
        raw_tag = m.group(1).upper().strip()
        new_val = m.group(2).strip()

        new_val = re.sub(r'^\s*\.\s*\n', '', new_val)
        new_val = re.sub(r'\n\s*\.\s*$', '', new_val).strip()

        new_val = re.sub(
            r'\n\s*\.?\s*(?:REGRET\s+ERROR|PLEASE\s+(?:INFORM|NOTE)|REGARDS|CENTRALIZED\s+OPERATIONS|BANK\s+AL\s+HABIB|YOURS\s+(?:TRULY|FAITHFULLY))\b.*$',
            '', new_val, flags=re.IGNORECASE | re.DOTALL,
        ).strip()
        if not new_val:
            continue
        canon_tag = _MT799_TAG_TO_FIELD.get(raw_tag, raw_tag)
        if canon_tag in seen_tags:
            continue
        seen_tags.add(canon_tag)

        anchor = None
        if canon_tag in _CLAUSE_TAGS:
            m_anchor = re.search(
                r'CONTRACT\s+(?:NO\.?|NOS?\.?)\s*'
                r'([A-Z][A-Z0-9][A-Z0-9\-]{3,30})',
                new_val, re.IGNORECASE,
            )
            if m_anchor:
                anchor = m_anchor.group(1).upper().rstrip('.,;')
        out.append(SwiftField(
            tag=canon_tag,
            label=SWIFT_FIELD_LABELS.get(canon_tag, f'Field {canon_tag}'),
            value=new_val,
            source_page=source_page,
            source_mt=source_mt,
            mt799_replace_anchor=anchor,
        ))

    pat_ubl_field = re.compile(
        r'(?:^|\n)\s*(\d{2}[A-Z]?)\s*:-\s*[^\n]{0,80}?'
        r'\bRead\s+as\b[\s\n\r]*[\'"]\s*([^\'"\n\r]+?)\s*[\'"]'
        r'\s*(?:instead\s+of\s+existing|i\s*/\s*[oe])',
        re.IGNORECASE,
    )
    for m in pat_ubl_field.finditer(norm):
        raw_tag = m.group(1).upper().strip()
        new_val = m.group(2).strip()
        canon_tag = _MT799_TAG_TO_FIELD.get(raw_tag, raw_tag)
        if canon_tag not in _SINGLE_VAL_TAGS or canon_tag in seen_tags:
            continue
        seen_tags.add(canon_tag)
        out.append(SwiftField(
            tag=canon_tag,
            label=SWIFT_FIELD_LABELS.get(canon_tag, f'Field {canon_tag}'),
            value=new_val,
            source_page=source_page,
            source_mt=source_mt,
        ))

    _clause_ops: Dict[str, List[str]] = {}

    def _add_op(canon_tag: str, op_line: str):
        if canon_tag not in _CLAUSE_TAGS:
            return
        _clause_ops.setdefault(canon_tag, []).append(op_line)

    for m in re.finditer(
        r'(?:^|\n)\s*(\d{2}[A-Z]?)\s*\(\s*(\d{1,3})\s*\)\s*:?\s*-?\s*'
        r'PLEASE\s+DELETE\s+COMPLETELY',
        norm, re.IGNORECASE,
    ):
        raw_tag = m.group(1).upper()
        clause_n = int(m.group(2))
        _add_op(_MT799_TAG_TO_FIELD.get(raw_tag, raw_tag), f"/DELETE/ CLAUSE {clause_n}")

    for m in re.finditer(
        r'(?:^|\n)\s*(\d{2}[A-Z]?)\s+(\d{1,3})\s+PLEASE\s+DELETE\s+COMPLETELY',
        norm, re.IGNORECASE,
    ):
        raw_tag = m.group(1).upper()
        clause_n = int(m.group(2))
        _add_op(_MT799_TAG_TO_FIELD.get(raw_tag, raw_tag), f"/DELETE/ CLAUSE {clause_n}")

    for canon_tag, ops in _clause_ops.items():
        if canon_tag in seen_tags or not ops:
            continue
        seen_tags.add(canon_tag)
        out.append(SwiftField(
            tag=canon_tag,
            label=SWIFT_FIELD_LABELS.get(canon_tag, f'Field {canon_tag}'),
            value='\n'.join(ops),
            source_page=source_page,
            source_mt=source_mt,
        ))

    return out

def _strip_mt799_anchor_block(text: str, anchor: str) -> str:
    """P273 — remove the F45A/F46A/F47A block referencing `anchor`.

    Used when an MT799 "+FIELD XX TO BE READ AS UNDER I/O EXISTING"
    amendment carries an explicit contract reference. The bank's
    semantic is REPLACE the existing /ADD/ block for that contract;
    without this strip, the new /ADD/ appends and the previous block
    remains as a stale duplicate (eg. BAHL 1003LC55989's BKKJP26-0090
    appearing once with old goods description and once with the
    corrected one).

    Strip rule (conservative — only the contract-ref pair):
      • Find the line containing the anchor.
      • Drop that line.
      • If the line above is a "AS PER BENEFICIARY... PROFORMA INVOICE
        CONTRACT [NO.]" continuation line, drop it too.

    The goods description line above the "AS PER ..." pair is left
    intact: in real consolidated F45As that line is often shared with
    a sibling contract's block (cross-amendment dedup), so removing it
    would orphan another contract's goods. The MT799's own /ADD/ then
    appends its full new block (goods + AS PER + contract ref),
    producing the expected single corrected block.

    No-op when anchor is missing or not present in `text` — preserves
    the pre-P273 append behaviour as a safe fallback.
    """
    if not anchor or anchor not in text:
        return text
    lines = text.split('\n')
    out: List[str] = []
    as_per_re = re.compile(
        r'AS\s+PER\s+BENEFICIARY[^\n]*PROFORMA\s+INVOICE\s+CONTRACT',
        re.IGNORECASE,
    )
    i = 0
    while i < len(lines):
        ln = lines[i]
        if anchor in ln:

            if out and as_per_re.search(out[-1]):
                out.pop()
            i += 1
            continue
        out.append(ln)
        i += 1
    return '\n'.join(out)

def _trim_resolved_for_name_address_field(text: str) -> str:
    """P276 (+P277/P278 ext) — trim a cross-ref resolved clause when
    destination is F50/F59 (name+address fields).

    Some banks place the beneficiary's name+address in a numbered
    F47A clause and reference it from F59 ("REFER TO FIELD 47A(10)
    FOR COMPLETE NAME AND ADDRESS"). The bank then appends an
    UNRELATED legal/disclaimer paragraph to the same clause,
    separated by a SWIFT lone-'.' or lone-'+' line (paragraph
    delimiter). The cross-ref resolver pulls the whole clause, so
    the consolidated F59 ends up with both the address AND the
    disclaimer.

    Trim rule (additive, generic):
      1. Truncate at the first lone-'.' OR lone-'+' line — both are
         SWIFT paragraph delimiters in different bank exports.
         Pure-address clauses with no break = no-op.
      2. Strip leading SWIFT decorator line that is only "+" chars
         (≥3 chars, e.g. "+++++++"). Banks insert this under the
         label as a visual separator; it is not address content.
         Phone-style "+92-..." inside a real address line is
         untouched (full-line match required).
      3. Strip leading clause-level label preambles — generic
         meta-labels that identify the clause as F50/F59's value:
           • "<X> [COMPLETE] NAME AND ADDRESS" — party-name form
             (BAHL/UBL/FUSION typical).
           • "FIELD <N> TO BE READ AS UNDER" — MT707 amendment
             narrative form (BAHL job 10cf936f case). Requires
             both "FIELD <N>" and "READ AS" tokens in the line so
             body text never falsely matches.
         Both label forms enforced ≤10 words to keep body lines
         intact.

    No-op when none of the conditions match → safe fallback.
    """
    if not text:
        return text
    lines = text.splitlines()
    para_end = None
    for i, ln in enumerate(lines):
        s = ln.strip()

        if s == '.' or s == '+':
            para_end = i
            break
    if para_end is not None:
        lines = lines[:para_end]
    while lines:
        first = lines[0].strip()
        if not first:
            lines.pop(0)
            continue

        if re.fullmatch(r'\+{3,}', first):
            lines.pop(0)
            continue

        if len(first.split()) <= 10:
            up = first.upper()

            if re.search(r'\bNAME\s+AND\s+ADDRESS\b', up):
                lines.pop(0)
                continue

            if (re.search(r'\bFIELD\s+\d{2}[A-Z]?\b', up)
                    and re.search(r'\bREAD\s+AS\b', up)):
                lines.pop(0)
                continue
        break
    return '\n'.join(lines).strip()

def _find_name_address_clause(clauses, party_keyword: str, dest_tag: str = ''):
    """P277 (+P278 ext) — locate a clause whose first non-empty line
    is a META-LABEL identifying it as F<dest_tag>'s value.

    Used when F50/F59 references its parent F47A WITHOUT a clause
    number ("REFER FIELD 47A" / "(REFER TO FIELD 47A)" / "REFER 47A").
    Banks embed the actual name+address inside one of F47A's clauses
    using one of two equivalent label conventions — both are short
    label/header lines (≤10 words):

      • Form A — party-name: line contains `party_keyword`
        (e.g. "APPLICANT" / "BENEFICIARY") AND the phrase
        "NAME AND ADDRESS". Examples:
            APPLICANT'S COMPLETE NAME AND ADDRESS:
            BENEFICIARY COMPLETE NAME AND ADDRESS:

      • Form B — field-number reference: line matches
        "FIELD <dest_tag>" or "F<dest_tag>" as a token. Used by
        BAHL MT707 amendment narratives (job 10cf936f). Example:
            FIELD 59 TO BE READ AS UNDER

    Match rules (strict, to avoid false positives):
      • Only the first non-empty line of each clause is examined.
      • Line length ≤ 10 words — body text never matches.
      • Form A requires both party keyword + "NAME AND ADDRESS".
      • Form B requires field-number token (FIELD / F prefix).

    Returns the matched Clause or None. None = caller falls back to
    its existing behaviour (no replacement) — zero regression.
    """
    if not clauses:
        return None
    pk = (party_keyword or '').upper()
    dt = (dest_tag or '').upper()
    for c in clauses:
        text = getattr(c, 'text', '') or ''
        for ln in text.splitlines():
            stripped = ln.strip()
            if not stripped:
                continue
            up = stripped.upper()
            if len(stripped.split()) > 10:
                break

            if (pk and pk in up and 'NAME AND ADDRESS' in up):
                return c

            if dt and re.search(
                rf'\b(?:FIELD\s+|F){re.escape(dt)}\b', up):
                return c
            break
    return None

def _detect_format_from_text(text: str) -> str:
    """Detect SWIFT format from GLM text content."""
    fusion_count = len(re.findall(r'\bF\d{2}[A-Z]?\s*:', text))
    alliance_count = len(re.findall(r':\d{2}[A-Z]?:', text))

    bare_fusion_count = len(re.findall(r'(?:^|\n)\s*\d{2}[A-Z]?:\s+[A-Z]', text))
    if fusion_count > alliance_count and fusion_count > bare_fusion_count:
        return 'fusion'
    elif alliance_count > 0 and alliance_count >= bare_fusion_count:
        return 'alliance'
    elif bare_fusion_count > 0:
        return 'bare_fusion'
    return 'unknown'

def _has_top_level_sections(text: str) -> bool:
    """Cheap structural gate: does this field begin with a top-level
    letter-paren section marker AND contain at least one more such marker
    at column 0? Minimum-possible heuristic — enumerates no specific
    keywords (no PART / SET / release%); just checks that the first
    non-blank content of the field starts with '<UPPERCASE>) ' (signalling
    a real section grouping rather than a sub-bullet inside a numbered
    clause body, since sub-bullets never occupy the field-start position).
    """
    if not text or not text.strip():
        return False
    stripped = text.lstrip()
    first_line = stripped.split('\n', 1)[0].strip()
    if not re.match(r'^[A-Z]\)\s+\S', first_line):
        return False

    distinct = set()
    for m in re.finditer(r'(?:^|\n)([A-Z])\)\s+\S', text):
        distinct.add(m.group(1))
    return len(distinct) >= 2

_LLM_CLAUSE_SPLIT_PROMPT = """You are parsing a SWIFT trade-finance text field. Split it into individual clauses and return structured JSON.

═══════════════════════════════════════════════════════════════════════
RULE — MULTI-SECTION (HIERARCHICAL) STRUCTURE
═══════════════════════════════════════════════════════════════════════

CONDITION (when this rule applies):
The field starts with a top-level letter-paren marker like "A)" at the very
beginning AND contains at least one more such marker like "B)" / "C)" /
... at the start of a line (column 0). This pattern means the field is
DIVIDED into multiple top-level groups, each governing a different set
of nested sub-items.

Real examples of when this division pattern shows up in LCs:
  • Partial-release credits — "A) FOR RELEASE OF 90 PERCENT PAYMENT" /
    "B) FOR RELEASE OF 10 PERCENT" — each section lists the documents
    required to release that payment portion.
  • Multi-tranche credits — "A) FOR FIRST SHIPMENT" / "B) FOR SECOND
    SHIPMENT" — each section lists shipment-specific documents.
  • Conditional document sets — "A) IF SHIPMENT IS BY VESSEL" /
    "B) IF SHIPMENT IS BY AIR" — alternative document requirements.
  • Any other A)/B)/C)/... division regardless of the heading wording.

WHAT TO PRODUCE for each section when the condition holds:
  1. ONE clause for the section's introductory line(s) — the "X) ..."
     line plus any descriptive header text BEFORE the first sub-item.
     This clause carries:
         section          = the section letter (e.g. "A")
         is_section_header = true
  2. ONE clause for EACH numbered/Roman sub-item in the section's body
     (1), 2), 3), i), ii), iii), 1., 2., etc.). Each carries:
         section          = the SAME section letter as the parent
         is_section_header = false
  3. Sub-items belong to the NEAREST preceding section letter — never to
     the wrong section.

OCR TOLERANCE: If the very first sub-item of a section is "i)" (lowercase
roman) but the rest are Arabic ("2)", "3)", ...), this is OCR mis-reading
the digit "1" as the letter "i" because they look near-identical in some
fonts. Treat "i)" as the first sub-item — split it like any other.

═══════════════════════════════════════════════════════════════════════
RULE — FLAT (NON-HIERARCHICAL) STRUCTURE
═══════════════════════════════════════════════════════════════════════

CONDITION: The field has NO top-level letter-paren division at column 0,
or only one such marker. It's a flat list (or a single block).

WHAT TO PRODUCE: All clauses with section="" and is_section_header=false.
Split the same way the field would normally split — one clause per
numbered/Roman sub-item, or one clause for the whole field if nothing
splits cleanly.

═══════════════════════════════════════════════════════════════════════
WORKED EXAMPLE
═══════════════════════════════════════════════════════════════════════

INPUT:
A) FOR RELEASE OF 90 PERCENT PAYMENT OF LC VALUE, FOLLOWING
DOCUMENTS ARE REQUIRED
i) BENEFICIARY MANUALLY SIGNED COMMERCIAL INVOICE
2) FULL SET OF THREE ORIGINAL BILLS OF LADING

B) FOR RELEASE OF 10 PERCENT OF LC VALUE FOLLOWING
DOCUMENTS ARE REQUIRED:
1) BENEFICIARY MANUALLY SIGNED COMMERCIAL INVOICE
2) CERTIFICATE OF WEIGHT IN ONE ORIGINAL

EXPECTED OUTPUT:
{
  "clauses": [
    {"text": "A) FOR RELEASE OF 90 PERCENT PAYMENT OF LC VALUE, FOLLOWING\\nDOCUMENTS ARE REQUIRED", "section": "A", "is_section_header": true},
    {"text": "i) BENEFICIARY MANUALLY SIGNED COMMERCIAL INVOICE", "section": "A", "is_section_header": false},
    {"text": "2) FULL SET OF THREE ORIGINAL BILLS OF LADING", "section": "A", "is_section_header": false},
    {"text": "B) FOR RELEASE OF 10 PERCENT OF LC VALUE FOLLOWING\\nDOCUMENTS ARE REQUIRED:", "section": "B", "is_section_header": true},
    {"text": "1) BENEFICIARY MANUALLY SIGNED COMMERCIAL INVOICE", "section": "B", "is_section_header": false},
    {"text": "2) CERTIFICATE OF WEIGHT IN ONE ORIGINAL", "section": "B", "is_section_header": false}
  ]
}

═══════════════════════════════════════════════════════════════════════
CRITICAL CONSTRAINTS (apply in BOTH cases)
═══════════════════════════════════════════════════════════════════════

1. Return text VERBATIM — do NOT paraphrase, fix typos, alter wording,
   normalise quotes, expand abbreviations, or summarise.
2. Preserve ALL original markers inside each clause's text exactly
   (1), 2), A), B), i), ii), 1., 2., dashes, parentheses, etc.).
3. Concatenating every clause's "text" with newlines must reproduce the
   ORIGINAL input modulo whitespace. Do NOT drop, add, or merge content.
4. Output ONLY the JSON object. No prose, no markdown fences, no
   commentary, no leading or trailing text.

INPUT TEXT:
{text}
"""

def _validate_clause_split(clauses: List['Clause'], original_text: str) -> bool:
    """Reject the LLM's split if joining its clauses doesn't reproduce the
    original text. Whitespace-normalised character compare — catches drops,
    paraphrases, hallucinations, single-character substitutions. The only
    differences allowed are runs of whitespace (newlines, tabs, multiple
    spaces) collapsed to a single space."""
    if not clauses:
        return False
    if len(clauses) > 60:
        return False
    joined = '\n'.join(c.text for c in clauses)

    def _norm(s: str) -> str:

        return re.sub(r'\s+', ' ', s).strip().lower()

    return _norm(joined) == _norm(original_text)

def _split_into_clauses_llm(tag: str, text: str, _progress=None) -> Optional[List['Clause']]:
    """LLM-driven clause splitter. Used only when `_has_top_level_sections`
    detects a multi-section hierarchy. Returns None on any failure
    (network, parse, validation) so the caller falls back to the legacy
    regex splitter — i.e. the worst case is the legacy behaviour, never
    a corrupted output.

    Endpoint strategy: try the configured Text-LLM first, then fall through
    to the VLM endpoint (which accepts text-only chat completions). This
    keeps the call working when one endpoint is mid-migration or returning
    a transient error — `_apply_amendment_vlm` uses the same chain.
    """
    if not text or not text.strip():
        return None

    candidates = []
    if QWEN_TEXT_LLM_URL and QWEN_TEXT_LLM_MODEL:
        candidates.append((QWEN_TEXT_LLM_URL, QWEN_TEXT_LLM_MODEL))
    if QWEN_VLM_URL and QWEN_VLM_MODEL:
        candidates.append((QWEN_VLM_URL, QWEN_VLM_MODEL))
    if not candidates:
        return None

    prompt = _LLM_CLAUSE_SPLIT_PROMPT.replace('{text}', text)
    content = ''
    for _url, _model in candidates:
        try:
            resp = requests.post(_url, json={
                "model": _model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 8000,
                "temperature": 0.0,
            }, timeout=None)
            if resp.status_code != 200:
                if _progress:
                    _progress(
                        f"  F{tag}: LLM split HTTP {resp.status_code} on "
                        f"{_url} — trying next endpoint"
                    )
                continue
            content = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
            if content:
                break
        except Exception as e:
            if _progress:
                _progress(f"  F{tag}: LLM split error on {_url} ({e}) — trying next endpoint")
            continue

    if not content:
        if _progress:
            _progress(f"  F{tag}: all LLM endpoints failed — falling back to regex")
        return None

    content = re.sub(r'```(?:json|plaintext|text)?\s*', '', content, flags=re.IGNORECASE)
    content = re.sub(r'\s*```\s*$', '', content).strip()

    m = re.search(r'\{.*\}', content, re.DOTALL)
    if not m:
        if _progress:
            _progress(f"  F{tag}: LLM returned no JSON — falling back to regex")
        return None
    try:
        parsed = json.loads(m.group(0))
    except json.JSONDecodeError:
        if _progress:
            _progress(f"  F{tag}: LLM JSON parse failed — falling back to regex")
        return None

    items = parsed.get('clauses', [])
    if not isinstance(items, list) or not items:
        return None

    out: List[Clause] = []
    for it in items:
        if not isinstance(it, dict):
            return None
        clause_text = str(it.get('text', '') or '').strip()
        if not clause_text:
            continue
        section = str(it.get('section', '') or '').strip()
        is_header = bool(it.get('is_section_header', False))
        out.append(Clause(
            clause_number=len(out) + 1,
            clause_id=f"{tag}-{len(out) + 1}",
            text=clause_text,
            parent_tag=tag,
            section=section,
            is_section_header=is_header,
        ))
    if not out:
        return None

    if not _validate_clause_split(out, text):
        if _progress:
            _progress(f"  F{tag}: LLM split rejected by integrity check — falling back to regex")
        return None

    if any(c.section for c in out):
        _sub_count: Dict[str, int] = {}
        for c in out:
            if not c.section:
                continue
            if c.is_section_header:
                c.clause_id = f"{tag}-{c.section}0"
            else:
                _sub_count[c.section] = _sub_count.get(c.section, 0) + 1
                c.clause_id = f"{tag}-{c.section}{_sub_count[c.section]}"

    return out

def _split_into_clauses(tag: str, text: str, _progress=None) -> List[Clause]:
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

    if isinstance(text, list):
        text = '\n'.join(str(item) for item in text)

    _normalised = text.strip()
    if tag in ('46A', '47A', '45A', '78') and _has_top_level_sections(_normalised):
        _llm_clauses = _split_into_clauses_llm(tag, _normalised, _progress=_progress)
        if _llm_clauses:
            if _progress:
                _progress(
                    f"  F{tag}: split into {len(_llm_clauses)} clauses via LLM "
                    f"(hierarchical multi-section)"
                )
            return _llm_clauses

    clauses = []
    text = text.strip()

    if tag in ('45A', 'F45A'):

        if not re.search(r'(?:^|\n)\s*\d+\s*[-.)]\s+', text):
            return [Clause(clause_number=1, clause_id=f"{tag}-1", text=text, parent_tag=tag)]

    if tag in ('78', 'F78', '72', 'F72', '79', 'F79'):
        _has_structural_marker = bool(re.search(
            r'(?:^|\n)\s*(?:'
            r'\d{1,2}\s*[-.)\)]'
            r'|\(\s*\d{1,2}\s*\)'
            r'|[A-Z]\s*[.)\)]'
            r'|\+\s'
            r'|[-*]\s'
            r')',
            text,
        ))
        if not _has_structural_marker:
            return [Clause(clause_number=1, clause_id=f"{tag}-1",
                           text=text, parent_tag=tag)]

    _normalized = text

    _normalized = re.sub(
        r'(?<=[\s.;:!?,])(\d{1,2})\s*([.)])\s*(?=[A-Z\(])',
        r'\n\1\2 ',
        _normalized,
    )

    numbered = re.split(r'\n\s*(2[0-5]|1\d|[1-9])\s*[.)]\s*(?!\d)', '\n' + _normalized)
    if len(numbered) >= 3:

        _prefix_text = (numbered[0] or '').strip()

        _CLAUSE_BEARING_TAGS_FOR_HEADER = (
            '45A', '45B', '46A', '46B', '47A', '47B', '72', '78', '79',
        )
        _has_substantive_prefix_header = (
            tag in _CLAUSE_BEARING_TAGS_FOR_HEADER
            and _prefix_text
            and len(_prefix_text) >= 20
            and not _prefix_text[0].isdigit()
        )

        if _has_substantive_prefix_header:
            clauses.append(Clause(
                clause_number=0,
                clause_id=f"{tag}-H0",
                text=_prefix_text,
                parent_tag=tag,
                section='',
                is_section_header=True,
            ))

        _by_num = {}
        _order = []
        for i in range(1, len(numbered) - 1, 2):
            try:
                _n = int(numbered[i])
            except (ValueError, TypeError):
                continue
            clause_text = numbered[i + 1].strip()
            if not clause_text:
                continue
            if _n in _by_num:

                _by_num[_n] = (_by_num[_n].rstrip() + '\n' + clause_text).strip()
            else:
                _by_num[_n] = clause_text
                _order.append(_n)

        if (not _has_substantive_prefix_header) and _prefix_text and _order:
            _first_n = _order[0]
            _by_num[_first_n] = f"{_prefix_text}\n{_by_num[_first_n]}".strip()

        for _n in _order:
            clauses.append(Clause(
                clause_number=_n,
                clause_id=f"{tag}-{_n}",
                text=_by_num[_n],
                parent_tag=tag,
            ))
        if clauses:
            return clauses

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

    lines = [ln.strip() for ln in text.split('\n') if ln.strip()]
    if len(lines) >= 2:
        grouped = []
        current = []
        for ln in lines:

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

    clauses.append(Clause(
        clause_number=1,
        clause_id=f"{tag}-1",
        text=text,
        parent_tag=tag,
    ))
    return clauses

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

    t = (t.replace('‘', "'").replace('’', "'")
          .replace('“', '"').replace('”', '"')
          .replace("''", '"'))

    def _clause_targets(scope: str):
        return [m.group(1) for m in
                re.finditer(r'CLAUSE\s+(?:NO\.?\s+)?(\d+)',
                             scope, re.IGNORECASE)]

    for m in re.finditer(r'/DEL(?:ETE)?/([^/]*?)(?=/[A-Z]+/|$)',
                          t, re.IGNORECASE | re.DOTALL):
        scope = m.group(1) or ''
        targets = _clause_targets(scope)
        if targets:
            for tgt in targets:
                ops.append({'op': 'DELETE', 'target': f'Clause {tgt}'})
        else:

            qm = re.search(r"['\"]([^'\"]+)['\"]\s*AS\s+DELETED",
                           scope, re.IGNORECASE)
            if qm:
                ops.append({'op': 'DELETE',
                            'target': f"Words: {qm.group(1).strip()}"})
            else:
                ops.append({'op': 'DELETE'})

    for m in re.finditer(r'/ADD/([^/]*?)(?=/[A-Z]+/|$)',
                          t, re.IGNORECASE | re.DOTALL):
        scope = m.group(1) or ''
        targets = _clause_targets(scope)
        if targets:
            for tgt in targets:
                ops.append({'op': 'ADD', 'target': f'Clause {tgt}'})
        else:
            ops.append({'op': 'ADD'})

    if re.search(r'/REPALL/', t, re.IGNORECASE):
        ops.append({'op': 'REPLACE-ALL'})

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

    seen = set()
    out = []
    for o in ops:
        key = (o.get('op'), o.get('target'))
        if key in seen:
            continue
        seen.add(key)
        out.append(o)
    return out

def _strip_amendment_prefix(line: str) -> str:
    """Remove SWIFT report sub-labels from one amendment line."""
    text = (line or '').strip()
    while True:
        upper = text.upper()
        if upper.startswith('CODE:'):
            text = text[5:].strip()
            continue
        if upper.startswith('NARRATIVE:'):
            text = text[10:].strip()
            continue
        break
    return text

def _is_amendment_line_marker(text: str) -> bool:
    """True for structural markers like 'Line 1' or 'Lines 2-100'."""
    clean = (text or '').strip().rstrip(':')
    if not clean:
        return False
    parts = clean.replace('-', ' ').replace('–', ' ').split()
    if not parts or parts[0].upper() not in ('LINE', 'LINES'):
        return False
    return any(part.isdigit() for part in parts[1:])

def _is_amendment_page_marker(text: str) -> bool:
    s = (text or '').strip()

    if re.fullmatch(r'-{3,}\s*PAGE\b[^-]*-{3,}', s, re.IGNORECASE):
        return True
    parts = s.split()
    if len(parts) != 4:
        return False
    return (
        parts[0].upper() == 'PAGE'
        and parts[1].isdigit()
        and parts[2].upper() == 'OF'
        and parts[3].isdigit()
    )

def _is_operation_footer_start(text: str) -> bool:
    upper = (text or '').strip().upper()
    if not upper:
        return False
    if upper in {'OTHER', 'REPORT FOOTER', 'END OF REPORT'}:
        return True
    return upper.startswith((
        'DELIVERY OVERDUE',
        'NETWORK DELIVERY',
        'PAYMENT CONFIRMATION',
        'CONFIRMED CURRENCY',
        'CONFIRMED AMOUNT',
        'CONFIRMED DATE',
    ))

def _operation_from_line(line: str):
    """Return (operation, rest_of_line) for /ADD/, /DELETE/, /REPALL/ lines."""
    text = _strip_amendment_prefix(line)
    upper = text.upper().lstrip()
    aliases = (
        ('/REPALL/', 'REPALL'),
        ('/DELETE/', 'DELETE'),
        ('/DEL/', 'DELETE'),
        ('/ADD/', 'ADD'),
    )
    for marker, op in aliases:
        if upper.startswith(marker):
            rest = text[len(marker):].strip()
            rest = rest.lstrip('+').lstrip(')').strip()
            return op, rest
    return None, ''

def _clean_operation_lines(lines: List[str], keep_dot_lines: bool = False) -> List[str]:
    cleaned = []
    for raw in lines:
        line = _strip_amendment_prefix(raw)
        if not line:

            if keep_dot_lines and cleaned and cleaned[-1] != '.':
                cleaned.append('.')
            continue
        if _is_operation_footer_start(line):
            break
        if _is_amendment_page_marker(line):
            continue
        if _is_amendment_line_marker(line):
            continue
        op, rest = _operation_from_line(line)
        if op:
            if rest:
                cleaned.append(rest)
            continue
        if not keep_dot_lines and line.strip() == '.':
            continue
        cleaned.append(line)
    return cleaned

def _drop_clause_word(text: str) -> str:
    stripped = (text or '').lstrip()
    upper = stripped.upper()

    for word in ('CLAUSES', 'CALUSES', 'CLAUSE', 'CALUSE'):
        if upper.startswith(word):
            rest = stripped[len(word):].lstrip()
            rest_upper = rest.upper()
            if rest_upper.startswith('NO.'):
                rest = rest[3:].lstrip()
            elif rest_upper.startswith('NO '):
                rest = rest[2:].lstrip()
            return rest
    return stripped

_CLAUSE_KEYWORD_RE = re.compile(r'^(?:CLAUSES?|CALUSES?)\b', re.IGNORECASE)

def _delete_clause_numbers_from_lines(lines: List[str]) -> List[int]:
    nums = []
    seen = set()
    for line in _clean_operation_lines(lines):
        stripped = line.lstrip()

        if not _CLAUSE_KEYWORD_RE.match(stripped):
            continue
        rest = _drop_clause_word(stripped)

        for tok in re.findall(r'\d+', rest):
            num = int(tok)
            if num not in seen:
                nums.append(num)
                seen.add(num)
    return nums

def _line_clause_number(line: str) -> Optional[int]:
    text = (line or '').lstrip()
    if not text:
        return None
    if text.startswith('('):
        pos = 1
        token = ''
        while pos < len(text) and text[pos].isdigit():
            token += text[pos]
            pos += 1
        if token and pos < len(text) and text[pos] == ')':
            return int(token)
        return None
    pos = 0
    token = ''
    while pos < len(text) and text[pos].isdigit():
        token += text[pos]
        pos += 1
    if token and pos < len(text) and text[pos] in {')', '.', ':'}:
        return int(token)
    return None

def _split_numbered_clause_blocks(text: str):
    prefix = []
    blocks = []
    current = None
    for line in (text or '').splitlines():
        clause_num = _line_clause_number(line)
        if clause_num is not None:
            if current is not None:
                blocks.append(current)
            current = {'num': clause_num, 'lines': [line.rstrip()]}
            continue
        if current is None:
            prefix.append(line.rstrip())
        else:
            current['lines'].append(line.rstrip())
    if current is not None:
        blocks.append(current)
    return prefix, blocks

def _render_numbered_clause_blocks(prefix: List[str], blocks: List[dict]) -> str:
    parts = [line for line in prefix if line.strip()]
    for block in blocks:
        block_text = '\n'.join(line.rstrip() for line in block.get('lines', [])).strip()
        if block_text:
            parts.append(block_text)
    return '\n'.join(parts).strip()

def _remove_numbered_clauses(text: str, nums: List[int]) -> str:
    if not nums:
        return text
    prefix, blocks = _split_numbered_clause_blocks(text)
    if not blocks:
        return text
    remove = set(nums)
    kept = [block for block in blocks if block.get('num') not in remove]
    return _render_numbered_clause_blocks(prefix, kept)

def _parse_add_clause_header(line: str):
    text = _drop_clause_word(_strip_amendment_prefix(line))
    pos = 0
    token = ''
    while pos < len(text) and text[pos].isdigit():
        token += text[pos]
        pos += 1
    if not token or pos >= len(text) or text[pos] not in {')', '.', ':'}:
        return None, ''
    return int(token), text[pos + 1:].strip()

def _add_clause_entries_from_lines(lines: List[str]):
    entries = []
    current_num = None
    current_lines = []
    for line in _clean_operation_lines(lines):
        num, rest = _parse_add_clause_header(line)
        if num is not None:
            if current_num is not None:
                entries.append((current_num, '\n'.join(current_lines).strip()))
            current_num = num
            current_lines = [rest] if rest else []
            continue
        if current_num is not None:
            current_lines.append(line)
    if current_num is not None:
        entries.append((current_num, '\n'.join(current_lines).strip()))
    return [(num, text) for num, text in entries if text]

def _insert_or_replace_numbered_clause(text: str, clause_num: int, clause_text: str) -> str:
    content_lines = [line.rstrip() for line in (clause_text or '').splitlines()]
    content_lines = [line for line in content_lines if line.strip()]
    if not content_lines:
        return text
    new_block = {
        'num': clause_num,
        'lines': [f"{clause_num}) {content_lines[0].strip()}"] + content_lines[1:],
    }
    prefix, blocks = _split_numbered_clause_blocks(text)
    if not blocks:
        base = (text or '').rstrip()
        addition = '\n'.join(new_block['lines']).strip()
        return f"{base}\n{addition}".strip() if base else addition

    replaced = False
    for idx, block in enumerate(blocks):
        if block.get('num') == clause_num:
            blocks[idx] = new_block
            replaced = True
            break
    if not replaced:
        insert_at = len(blocks)
        for idx, block in enumerate(blocks):
            if block.get('num', 0) > clause_num:
                insert_at = idx
                break
        blocks.insert(insert_at, new_block)
    return _render_numbered_clause_blocks(prefix, blocks)

def _parse_sub_clause_edit(lines):
    """Return ('WORDS', clause_n, text) | ('SUB', clause_n, letter) | None.

    Detects the BAHL labelled sub-clause edit forms inside an op block's
    payload lines (already stripped of `Code:` / `Narrative:` / `Lines N`
    framing by `_clean_operation_lines`)."""
    cleaned = _clean_operation_lines(lines)
    if not cleaned:
        return None
    first = cleaned[0].strip()

    m = re.match(r'WORDS\s+IN\s+CLAUSE\s+(\d+)\b', first, re.IGNORECASE)
    if m:
        clause_n = int(m.group(1))

        payload = '\n'.join(cleaned[1:]).strip()

        payload = re.sub(r"^['\"]+", '', payload)
        payload = re.sub(r"['\"]+$", '', payload)
        payload = payload.strip()
        if payload:
            return ('WORDS', clause_n, payload)
        return None

    m = re.match(r'SUB\s+CLAUSE\s+(\d+)\s*\(\s*([A-Z])\s*\)+', first, re.IGNORECASE)
    if m:
        clause_n = int(m.group(1))
        letter = m.group(2).upper()
        return ('SUB', clause_n, letter)

    return None

def _apply_sub_clause_word_delete(text, clause_n, payload):
    """Strip `payload` substring from clause N's body. Returns new text
    (unchanged when clause N missing or substring not found — bank
    sometimes targets a wrong clause number)."""
    prefix, blocks = _split_numbered_clause_blocks(text)
    if not blocks:
        return text
    target = next((b for b in blocks if b.get('num') == clause_n), None)
    if target is None:
        return text
    body = '\n'.join(target['lines'])
    if not body:
        return text

    new_body = body
    if payload in body:
        new_body = body.replace(payload, '', 1)
    else:

        try:
            tolerant = re.escape(payload)
            tolerant = tolerant.replace(r'\ ', r'\s+').replace(r'\\\n', r'\s+')
            new_body = re.sub(tolerant, '', body, count=1, flags=re.IGNORECASE)
        except re.error:
            new_body = body

    if new_body == body:
        return text

    new_body = re.sub(r' {2,}', ' ', new_body)
    new_body = re.sub(r'\n[ \t]*\n[ \t]*\n+', '\n\n', new_body)
    target['lines'] = new_body.splitlines()
    return _render_numbered_clause_blocks(prefix, blocks)

def _apply_sub_clause_word_add(text, clause_n, payload):
    """Append `payload` to clause N's body on a new line. Returns new
    text (unchanged when clause N missing)."""
    prefix, blocks = _split_numbered_clause_blocks(text)
    if not blocks:
        return text
    target = next((b for b in blocks if b.get('num') == clause_n), None)
    if target is None:
        return text
    if payload in '\n'.join(target['lines']):

        return text
    target['lines'].extend(payload.splitlines())
    return _render_numbered_clause_blocks(prefix, blocks)

def _apply_sub_bullet_delete(text, clause_n, letter):
    """Remove sub-bullet `(letter)` (or stutter `(letter))`) line(s)
    from clause N's body. Sub-bullet body extends until the next sibling
    sub-bullet `(LETTER)` or end of clause N."""
    prefix, blocks = _split_numbered_clause_blocks(text)
    if not blocks:
        return text
    target = next((b for b in blocks if b.get('num') == clause_n), None)
    if target is None:
        return text

    sub_open_pat = re.compile(r'^\s*\(\s*' + re.escape(letter) + r'\s*\)+', re.IGNORECASE)
    any_sub_pat = re.compile(r'^\s*\(\s*[A-Z]\s*\)+', re.IGNORECASE)

    new_lines = []
    skip = False
    removed = False
    for ln in target['lines']:
        if skip:

            if any_sub_pat.match(ln):
                skip = False
                new_lines.append(ln)

            continue
        if not removed and sub_open_pat.match(ln):
            skip = True
            removed = True
            continue
        new_lines.append(ln)
    if not removed:
        return text
    target['lines'] = new_lines
    return _render_numbered_clause_blocks(prefix, blocks)

def _parse_operation_blocks(amendment_text: str):
    blocks = []
    current_op = None
    current_lines = []
    for raw_line in (amendment_text or '').splitlines():
        op, rest = _operation_from_line(raw_line)
        if op:
            if current_op:
                blocks.append((current_op, current_lines))
            current_op = op
            current_lines = [rest] if rest else []
            continue
        if current_op:
            current_lines.append(raw_line)
    if current_op:
        blocks.append((current_op, current_lines))
    return blocks

def _apply_line_based_operations(base_text: str, amendment_text: str):
    """Apply direct MT707 operation blocks without relying on broad matching.

    P231 — Returns four values: (result_text, fully_handled, clauses_changed,
    clauses_deleted).

    `clauses_changed` is a list of 1-based clause numbers whose CONTENT
    changed in place (only /ADD/ N) entries — these stay at position N
    in the output, with new text). The renderer marks these positions
    AMENDED.

    P244 — `clauses_deleted` is a separate list of 1-based clause numbers
    that were /DELETE/d. After deletion the trailing clauses shift up to
    fill the gap — so position N in the post-renumber output now points
    to a DIFFERENT clause whose content is unchanged. Marking that
    shifted-up position AMENDED was misleading the user (e.g., F46A
    clause 5 wrongly highlighted when only original clause 5 was
    deleted, current display 5 = original clause 6). Tracking deletes
    separately lets the renderer distinguish "delete-only amendment →
    header AMENDED, no per-clause marks" from "REPALL/VLM → whole-field
    AMENDED on every clause" from "explicit ADD/REPLACE → mark just
    those positions".

    REPALL is a whole-field replace and contributes nothing (legacy
    behaviour preserved — both lists stay empty so caller falls back to
    whole-field AMENDED rendering).
    """
    blocks = _parse_operation_blocks(amendment_text)
    if not blocks:
        return base_text, False, [], []

    result = base_text
    handled_any = False
    unsupported = False

    clauses_changed: List[int] = []
    _seen_changed: set = set()
    clauses_deleted: List[int] = []
    _seen_deleted: set = set()

    def _track_changed(nums):
        for _n in nums:
            if _n not in _seen_changed:
                _seen_changed.add(_n)
                clauses_changed.append(_n)

    def _track_deleted(nums):
        for _n in nums:
            if _n not in _seen_deleted:
                _seen_deleted.add(_n)
                clauses_deleted.append(_n)

    for op, lines in blocks:
        if op == 'REPALL':
            clean_lines = _clean_operation_lines(lines, keep_dot_lines=True)
            payload = '\n'.join(clean_lines).strip()
            upper_payload = payload.upper()
            has_nested_instruction = (
                'PLEASE READ' in upper_payload
                or 'UNDER FIELD' in upper_payload
                or ' TO READ AS' in upper_payload
                or 'DELETE WORDING' in upper_payload
            )
            if payload and not has_nested_instruction:
                result = payload
                handled_any = True

            else:
                unsupported = True
        elif op == 'DELETE':
            nums = _delete_clause_numbers_from_lines(lines)
            if nums:
                result = _remove_numbered_clauses(result, nums)
                handled_any = True

                _track_deleted(nums)
            else:

                _sub = _parse_sub_clause_edit(lines)
                if _sub is None:
                    unsupported = True
                else:
                    _kind, _cn, _payload = _sub
                    if _kind == 'WORDS':
                        _new = _apply_sub_clause_word_delete(result, _cn, _payload)
                    else:
                        _new = _apply_sub_bullet_delete(result, _cn, _payload)

                    handled_any = True
                    if _new != result:
                        result = _new

                        _track_changed([_cn])
        elif op == 'ADD':
            entries = _add_clause_entries_from_lines(lines)
            if entries:
                for clause_num, clause_text in entries:
                    result = _insert_or_replace_numbered_clause(result, clause_num, clause_text)
                handled_any = True
                _track_changed([num for num, _ in entries])
            else:

                _sub = _parse_sub_clause_edit(lines)
                if _sub is None or _sub[0] != 'WORDS':

                    unsupported = True
                else:
                    _, _cn, _payload = _sub
                    _new = _apply_sub_clause_word_add(result, _cn, _payload)
                    handled_any = True
                    if _new != result:
                        result = _new
                        _track_changed([_cn])

    return result, handled_any and not unsupported, clauses_changed, clauses_deleted

def _apply_text_amendment(base_text: str, amendment_text: str,
                          _clauses_changed_out: Optional[List[int]] = None,
                          _clauses_deleted_out: Optional[List[int]] = None) -> str:
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

    P231 \u2014 Optional `_clauses_changed_out` list parameter. When provided,
    the function appends 1-based clause numbers that were touched by
    deterministic line-based operations (`/DELETE/ CLAUSE N`, `/ADD/
    CLAUSE N) ...`). The caller (`_apply_amendment`) records this list
    in `record.change_details[tag]['clauses_changed']` so the FLC
    renderer can mark only those specific clauses as AMENDED instead of
    the whole field. When the amendment uses non-clause-numbered patterns
    (full REPALL, narrative DELETE/REPLACE BY, word substitutions) the
    out-list stays empty and the renderer falls back to whole-field
    AMENDED \u2014 preserving the legacy behaviour for those cases.
    """
    if not base_text:
        return base_text

    result = base_text

    amd = amendment_text
    amd = amd.replace('\u2018', "'").replace('\u2019', "'")
    amd = amd.replace('\u201C', '"').replace('\u201D', '"')
    amd = amd.replace("''", '"')

    line_result, line_handled, line_clauses, line_deleted = _apply_line_based_operations(result, amd)
    if line_handled:
        if _clauses_changed_out is not None and line_clauses:
            for _n in line_clauses:
                if _n not in _clauses_changed_out:
                    _clauses_changed_out.append(_n)

        if _clauses_deleted_out is not None and line_deleted:
            for _n in line_deleted:
                if _n not in _clauses_deleted_out:
                    _clauses_deleted_out.append(_n)
        return line_result

    Q = r"""['"]+"""

    _repall_m = re.match(r'\s*/REPALL/\s*(.+)', amd, re.IGNORECASE | re.DOTALL)
    if _repall_m:
        new_content = _repall_m.group(1).strip()

        new_content = re.sub(r'(?:Narrative\d?:\s*)+', '', new_content).strip()
        new_content = re.sub(r'\n\s*Narrative\d?:\s*', '\n', new_content).strip()
        new_content = re.sub(r'^Lines?\s*\d*(?:\s*-\s*\d+)?(?:\s*to\s*\d+)?\s*:?\s*$', '', new_content, flags=re.MULTILINE).strip()
        new_content = re.sub(r'^\s*Line\s+\d+\s*$', '', new_content, flags=re.MULTILINE).strip()
        new_content = re.sub(r'^\s*Code\s*[-:]?\s*(?:/REPALL/)?\s*$', '', new_content, flags=re.MULTILINE).strip()
        new_content = re.sub(r'^\s*Code\s*-\s*Narrative\s*$', '', new_content, flags=re.MULTILINE).strip()

        _has_field_instructions = bool(re.search(
            r'(?:^|\n)\s*(?:UNDER\s+)?FIELD\s+\d{2}[A-Z]?(?:-\d+)?\s+'
            r'(?:TO\s+READ\s+AS|WORD\s+TO\s+READ\s+AS|DELETE\s+WORDING|ADD\s+(?:CLAUSE|LOI|WORDING))',
            new_content, re.IGNORECASE,
        ))

        _read_as_m = re.search(
            r'(?:NOW\s+)?TO\s+(?:BE\s+)?READ\s+AS\s*,?\s*' + Q + r'(.+?)' + Q + r'\s*(?:I/?O\s|$)',
            new_content, re.IGNORECASE | re.DOTALL,
        )
        if not _read_as_m:

            _read_as_m = re.search(
                r'(?:NOW\s+)?TO\s+(?:BE\s+)?READ\s+AS\s*,?\s*' + Q + r'(.+)',
                new_content, re.IGNORECASE | re.DOTALL,
            )
        if _read_as_m and not _has_field_instructions:
            _extracted = _read_as_m.group(1).strip().rstrip("'\"")
            if _extracted:
                result = _extracted

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

            elif not re.search(r'PLEASE\s+READ', new_content, re.IGNORECASE) and not _has_field_instructions:
                if new_content:
                    result = new_content

    for m in re.finditer(
            r'PLEASE\s+READ\s+WORDS\s+IN\s+FIELD\s+\w+-?\d*\s+' + Q + r'(.+?)' + Q + r'\s+AS\s+DELETED',
            amd, re.IGNORECASE | re.DOTALL):
        del_text = m.group(1).strip()
        if del_text in result:
            result = result.replace(del_text, '')

            result = re.sub(r'  +', ' ', result)
            result = re.sub(r'\n\s*\n\s*\n', '\n\n', result)

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

    for m in re.finditer(r'DELETE\s+' + Q + r'(.+?)' + Q + r'\s+REPLACE\s+BY\s+' + Q + r'(.+?)' + Q,
                         amd, re.IGNORECASE | re.DOTALL):
        old_text = m.group(1).strip()
        new_text = m.group(2).strip()
        if old_text in result:
            result = result.replace(old_text, new_text)

    def _replace_old_with_new(base: str, old_text: str, new_text: str) -> str:
        if not old_text:
            return base
        candidates_old = [old_text]
        candidates_new = [new_text]

        _strip_ccy = re.compile(
            r'^(?:[A-Z]{3}|EURO|DOLLAR|POUND|YEN)\s+',
            re.IGNORECASE,
        )
        _bare_old = _strip_ccy.sub('', old_text).strip()
        _bare_new = _strip_ccy.sub('', new_text).strip()
        if _bare_old != old_text:
            candidates_old.append(_bare_old)
            candidates_new.append(_bare_new)

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

    for m in re.finditer(r'C\w{1,5}E\s+NO\.?\s*(\d+)\s+(?:NOW\s+)?TO\s+READ\s+AS\s+' + Q + r'(.+?)' + Q + r'\s+INSTEAD\s+OF\s+' + Q + r'(.+?)' + Q,
                         amd, re.IGNORECASE | re.DOTALL):
        old_text = m.group(3).strip()
        new_text = m.group(2).strip()
        if old_text in result:
            result = result.replace(old_text, new_text)

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

    for m in re.finditer(r'/ADD/\s*\+?\s*\)?\s*(.+?)(?=\n\s*/(?:ADD|DEL(?:ETE)?|REPALL)/|\Z)',
                         amd, re.IGNORECASE | re.DOTALL):
        add_text = m.group(1).strip()

        add_text = re.sub(
            r'^[\s/]*(?:(?:ADD|DELETE|DEL|REPALL)\s*/+\s*)+',
            '', add_text, flags=re.IGNORECASE,
        ).strip()

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

        if re.search(r'WORDS\s+IN\s+CLAUSE', add_text, re.IGNORECASE):
            continue
        if re.search(r'SUB\s+CLAUSE\b', add_text, re.IGNORECASE):
            continue

        if add_text and add_text not in result:
            result = result.rstrip() + '\n' + add_text

    for m in re.finditer(
        r'/DELETE/\s*\n?\s*(.+?)(?=\n\s*/(?:ADD|DEL(?:ETE)?|REPALL)/|\Z)',
        amd, re.IGNORECASE | re.DOTALL,
    ):
        del_text = m.group(1).strip()

        del_text = re.sub(
            r'^[\s/]*(?:(?:ADD|DELETE|DEL|REPALL)\s*/+\s*)+',
            '', del_text, flags=re.IGNORECASE,
        ).strip()

        del_text = re.sub(r'^\s*\.\s*\n', '', del_text)
        del_text = re.sub(r'\n\s*\.\s*$', '', del_text).strip()

        del_text = re.sub(
            r'(?m)^[ \t]*Narrative[ \t]*:[ \t]*', '', del_text,
        )
        del_text = re.sub(
            r'(?m)^[ \t]*Code[ \t]*:[ \t]*', '', del_text,
        )
        del_text = re.sub(
            r'(?m)^[ \t]*Lines?[ \t]+\d+(?:[ \t]*[-–][ \t]*\d+)?[ \t]*$',
            '', del_text,
        )
        del_text = re.sub(
            r'(?m)^[ \t]*Page\s+\d+\s+of\s+\d+[ \t]*$',
            '', del_text, flags=re.IGNORECASE,
        )

        del_text = re.sub(r'\n[ \t]*\n+', '\n', del_text).strip()
        if not del_text or len(del_text) < 20:

            continue

        if re.search(r'^\s*C\w{1,5}E\s+(?:NO\.?\s*)?\d', del_text, re.IGNORECASE):
            continue
        if re.search(r'WORDS\s+IN\s+CLAUSE', del_text, re.IGNORECASE):
            continue
        if re.search(r'SUB\s+CLAUSE\b', del_text, re.IGNORECASE):
            continue
        if re.search(r'PLEASE\s+READ', del_text, re.IGNORECASE):
            continue

        if del_text in result:
            result = result.replace(del_text, '', 1)
        else:
            try:
                tolerant = re.escape(del_text)
                tolerant = tolerant.replace(r'\ ', r'\s+').replace('\\\n', r'\s+')
                new_result = re.sub(
                    tolerant, '', result, count=1,
                    flags=re.IGNORECASE,
                )
                if new_result != result:
                    result = new_result
            except re.error:
                pass

        result = re.sub(r'\n[ \t]*\n[ \t]*\n+', '\n\n', result)

    for m in re.finditer(r'/ADD/\s*\+?\s*\)?\s*TO\s+READ\s+AS\s+' + Q + r'(.+?)' + Q,
                         amd, re.IGNORECASE | re.DOTALL):
        new_text = m.group(1).strip()

        if not re.search(r'DELETE.*REPLACE', amd[max(0,m.start()-50):m.start()], re.IGNORECASE):

            if new_text and new_text not in result:
                result = new_text

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

    for m in re.finditer(r'PLEASE\s+READ\s+FIELD\s+\w+\s+AS\s+' + Q + r'(.+?)' + Q,
                         amd, re.IGNORECASE | re.DOTALL):

        if not re.search(r'FIELD\s+\w+-\d+', m.group(0)):
            new_text = m.group(1).strip()
            if new_text:
                result = new_text

    for m in re.finditer(r'IN\s+FIELD\s+\w+:\s*ADD\s+C\w{1,5}E\s+NO\.?\s*(\d+)',
                         amd, re.IGNORECASE):
        clause_num = int(m.group(1))

        remaining = amd[m.end():].strip()
        if remaining and remaining not in result:
            result = result.rstrip() + f'\n{clause_num}.{remaining}'

    _PRE_CLAUSE_SKIP_RE = re.compile(r'(?:^|\s)(?:WORDS\s+IN|SUB)$')
    for m in re.finditer(r'C\w{1,5}E\s+([\d,\s]+)', amd):
        pre = amd[:m.start()].upper()

        _pre_tail = pre[-20:].rstrip()
        if _PRE_CLAUSE_SKIP_RE.search(_pre_tail):
            continue
        if '/DELETE/' not in pre[max(0,pre.rfind('/')-10):]:
            continue
        nums = [int(n.strip()) for n in m.group(1).split(',') if n.strip().isdigit()]
        for clause_num in nums:

            clause_pat = re.compile(
                r'(\(' + str(clause_num) + r'\)|' + str(clause_num) + r'[\).:])\s*(.+?)(?=\n\s*(?:\(\d+\)|\d+[\).:])\s|\Z)',
                re.DOTALL)
            result = clause_pat.sub('', result)
        result = re.sub(r'\n\s*\n\s*\n', '\n\n', result)

    for m in re.finditer(r'C\w{1,5}E\s+(\d+)\)\s*(.+?)(?=\n\s*C\w{1,5}E\s+\d+\)|\n\s*Code:|\Z)',
                         amd, re.IGNORECASE | re.DOTALL):
        pre = amd[:m.start()].upper()
        if '/ADD/' not in pre[max(0,len(pre)-100):]:
            continue
        clause_num = int(m.group(1))
        new_text = m.group(2).strip()

        new_text = re.sub(r'(?:Narrative\d?:\s*)+', '', new_text).strip()
        new_text = re.sub(r'\n\s*Narrative\d?:\s*', '\n', new_text).strip()
        if new_text and new_text not in result:
            result = result.rstrip() + f'\n{clause_num}){new_text}'

    for m in re.finditer(r'IN\s+FIELD\s+\w+:\s*FOR\s+EXISTING\s+PLEASE\s+READ\s+' + Q + r'(.+?)' + Q,
                         amd, re.IGNORECASE | re.DOTALL):
        new_val = m.group(1).strip()
        if new_val:
            result = new_val

    Q = r"""['"]+"""

    for m in re.finditer(
        r'FIELD\s+\d{2}[A-Z]?(?!\s*-\d|\s+WORD\b)\s+(?:NOW\s+)?TO\s+READ\s+AS\s*\n?\s*'
        + Q + r'?(.+?)'
        r'(?=\n\s*(?:'
        r'FIELD\s+\d{2}[A-Z]'
        r'|UNDER\s+FIELD'
        r'|/(?:REPALL|ADD|DELETE|DEL)/'
        r'|Other\b|Page\s+\d+\s+of'
        r'|Delivery\s+overdue|Network\s+delivery'
        r'|Payment\s+Confirmation|Confirmed\s+(?:Currency|Amount|Date)'
        r'|Block\s+\d|Report\s+(?:Header|Footer|Content)'
        r'|Message\s+(?:Header|Identifier|Details)'
        r')|\Z)',
        amd, re.IGNORECASE | re.DOTALL,
    ):
        new_text = m.group(1).strip()
        new_text = new_text.lstrip("'\"").rstrip("'\"").strip()
        if new_text:
            result = new_text

    for m in re.finditer(
        r'FIELD\s+\d{2}[A-Z]?-(\d+)\s+(?:NOW\s+)?TO\s+READ\s+AS\s*\n?\s*' + Q + r'(.+?)' + Q,
        amd, re.IGNORECASE | re.DOTALL,
    ):
        clause_num = int(m.group(1))
        new_text = m.group(2).strip()

        clause_pat = re.compile(
            r'(' + str(clause_num) + r'[\).:\s])\s*(.+?)(?=\n\s*(?:\d+[\).:])\s|\Z)',
            re.DOTALL,
        )
        cm = clause_pat.search(result)
        if cm:
            result = result[:cm.start(2)] + ' ' + new_text + result[cm.end(2):]

    for m in re.finditer(
        r'FIELD\s+\d{2}[A-Z]?-?(\d*)\s+DELETE\s+WORDING\s+AS\s*\n?\s*' + Q + r'(.+?)' + Q,
        amd, re.IGNORECASE | re.DOTALL,
    ):
        del_text = m.group(2).strip()
        if del_text and del_text in result:
            result = result.replace(del_text, '')
            result = re.sub(r'  +', ' ', result)
            result = re.sub(r'\n\s*\n\s*\n', '\n\n', result)

    for m in re.finditer(
        r'(?:UNDER\s+)?FIELD\s+\d{2}[A-Z]?-?(\d*)\s+WORD\s+TO\s+READ\s+AS\s*\n?\s*'
        + Q + r'([^\n]+?)' + Q + r'\s+I/O\s+'
        + Q + r'([^\n]+?)' + Q + r'\s*(?=$|\n|\Z)',
        amd, re.IGNORECASE | re.MULTILINE,
    ):
        _clause_n_str = m.group(1)
        new_text = m.group(2).strip().strip("'\"").strip()
        old_text = m.group(3).strip().strip("'\"").strip()
        if not (old_text and new_text and old_text != new_text):
            continue

        _parts = []
        _last_q = False
        for _ch in old_text:
            if _ch in "'\"":
                if not _last_q:
                    _parts.append(r"['\"]+")
                    _last_q = True
            else:
                _parts.append(re.escape(_ch))
                _last_q = False
        _tolerant_pat = r"['\"]*" + ''.join(_parts) + r"['\"]*"

        _applied = False
        if re.search(_tolerant_pat, result):
            result = re.sub(_tolerant_pat, new_text, result, count=1)
            _applied = True
        elif old_text in result:
            result = result.replace(old_text, new_text, 1)
            _applied = True

        if (_applied and _clause_n_str and _clause_n_str.isdigit()
                and _clauses_changed_out is not None):
            _n = int(_clause_n_str)
            if _n not in _clauses_changed_out:
                _clauses_changed_out.append(_n)

    for m in re.finditer(
        r'UNDER\s+FIELD\s+\d{2}[A-Z]?(?:-\d+)?\s+ADD\s+(?:LOI\s+)?(?:CLAUSE|WORDING)\s+AS\s*\n?\s*'
        + Q + r'?(.+?)(?=' + Q + r'?\s*$|\n\s*(?:FIELD|UNDER)\s+\d{2}[A-Z])',
        amd, re.IGNORECASE | re.DOTALL,
    ):
        add_text = m.group(1).strip()

        add_text = re.sub(r'["\']$', '', add_text).strip()
        if add_text and add_text not in result:
            result = result.rstrip() + '\n' + add_text

    for m in re.finditer(
        r'UNDER\s+FIELD\s+\d{2}[A-Z]?(?:-\d+)?\s+ADD\s+(.+?)\s+AS\s*\n?\s*'
        + Q + r'(.+?)' + Q,
        amd, re.IGNORECASE | re.DOTALL,
    ):
        add_label = m.group(1).strip()
        add_value = m.group(2).strip()
        if add_value:

            append_text = f"AND ADD {add_label} {add_value}"
            if add_value not in result:
                result = result.rstrip() + '\n' + append_text

    return result

_FIELD_LABEL_STRIP = {
    '20':  r'^(?:Documentary\s+Credit\s+Number|Sender\'?s?\s+Reference|Transaction\s+Reference\s+Number)\s*[\n\r]*',

    '21':  r'^(?:(?:[A-Z]\w*\'?s?\s+){0,3})?Reference\s*[\n\r]*',

    '22A': r'^Purpose\s+of\s+Message\s*[\n\r]*',

    '23':  r'^(?:(?:[A-Z]\w*\'?s?\s+){0,3})?Reference(?:\s+to\s+Pre-?Advice)?\s*[\n\r]*',
    '27':  r'^Sequence\s+of\s+Total\s*[\n\r]*',
    '31C': r'^Date\s+of\s+Issue\s*[\n\r]*',
    '31D': r'^Date\s+and\s+Place\s+of\s+Expiry\s*[\n\r]*',
    '32B': r'^(?:Currency\s+Code,?\s*Amount|Increase\s+of\s+Documentary\s+Credit\s+Amount)\s*[\n\r]*',

    '33B': r'^(?:Additional\s+Amount\s+Covered|Decrease\s+of\s+Documentary\s+Credit\s+Amount|Currency,?\s+Original\s+Ordering\s+Amount)\s*[\n\r]*',
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

    '58A': r'^Requested\s+Confirmation\s+Party.*?(?:Identifier\s+Code)?\s*[\n\r]*',
    '59':  r'^Beneficiary\s*[\n\r]*(?:Name\s+and\s+Address:?\s*[\n\r]*)?',

    '71D': r'^(?:(?:OUR|BEN|SHA|Other)\s+)?Charges\s*[\n\r]*',
    '78':  r'^Instructions\s+to\s+the\s+Paying.*?Bank\s*[\n\r]*',
}

_SUB_BULLET_INLINE_RE = re.compile(
    r'(?<=[\s.:;!?,])'
    r'\(?'
    r'(?:[A-Za-z]|[IVXivx]{1,5})'
    r'\)'
    r'(?=\s+[A-Z\(])'
)

def _restore_inline_sub_bullets(text: str) -> str:
    """When 2+ sub-bullet markers appear inline on a single line within a
    clause-bearing field's text, split the line at each marker so the
    sub-items render as separate lines and the verifier sees them as
    separate conditions. See P210 docstring above for the full background."""
    if not text or '\n' in text and text.count('\n') >= text.count(' ') / 2:

        pass
    new_lines = []
    for line in text.split('\n'):
        matches = list(_SUB_BULLET_INLINE_RE.finditer(line))
        if len(matches) < 2:
            new_lines.append(line)
            continue
        prev_end = 0
        chunks = []
        for m in matches:
            if m.start() > prev_end:
                chunks.append(line[prev_end:m.start()].rstrip())
            prev_end = m.start()
        chunks.append(line[prev_end:].rstrip())
        new_lines.extend(c for c in chunks if c.strip())
    return '\n'.join(new_lines)

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

    # P198h6 — strip OCR-prompt-template leakage that the upstream
    # GLM/VLM occasionally echoed back as content (real failure:
    # 100+ repeated "- Use a table or list format if possible" lines
    # showed up in F57A of the final LC). This pass cleans cached
    # OCR text on regenerate, even if the classifier-side stripper
    # didn't catch it during the original OCR pass.
    _OCR_PROMPT_LEAK_PATTERNS = (
        re.compile(
            r'^\s*[-*•]?\s*Use\s+(?:a\s+)?(?:table|list|bullet)\s+'
            r'(?:or\s+(?:list|table|bullet)\s+)?'
            r'(?:format|structure|layout|representation|style)'
            r'(?:\s+if\s+(?:possible|necessary|appropriate))?\s*$',
            re.IGNORECASE | re.MULTILINE,
        ),
        re.compile(
            r'^\s*[-*•]?\s*Use\s+(?:a\s+)?(?:table|list|bullet[s\-]list|bullet\s+points?)\s+'
            r'to\s+(?:organize|present|display|format|show|list)[^\n]*$',
            re.IGNORECASE | re.MULTILINE,
        ),
        re.compile(
            r'^\s*[-*•]?\s*Use\s+a\s+table\s+structure\s+with\s+headers[^\n]*$',
            re.IGNORECASE | re.MULTILINE,
        ),
        re.compile(
            r'^\s*[-*•]?\s*Use\s+(?:a\s+)?(?:standard|consistent|clear|legible|'
            r'readable|simple|plain)\s+font[^\n]*$',
            re.IGNORECASE | re.MULTILINE,
        ),
        # "Use a clear, legible font for footnotes / document references / ..."
        re.compile(
            r'^\s*[-*•]?\s*Use\s+(?:a\s+)?(?:clear|legible|standard|simple|plain|'
            r'consistent|readable|easy[\s-]to[\s-]read|professional)'
            r'(?:\s*,\s*(?:clear|legible|standard|simple|plain|consistent|'
            r'readable|easy[\s-]to[\s-]read|professional))*'
            r'\s+font\s+for\s+(?:footnotes?|document\s+references?|'
            r'references?|citations?|headings?|subheadings?|titles?|body|'
            r'text|content|lists?|bullets?|tables?|all\s+text)[^\n]*$',
            re.IGNORECASE | re.MULTILINE,
        ),
        re.compile(
            r'^\s*[-*•]?\s*(?:Preserve|Maintain|Keep)\s+(?:line\s+breaks|'
            r'indentation|spacing|formatting|layout)[^\n]*$',
            re.IGNORECASE | re.MULTILINE,
        ),
    )
    for _pat in _OCR_PROMPT_LEAK_PATTERNS:
        v = _pat.sub('', v)
    # P198h6 — Strip markdown-bold wrappers around SWIFT field labels
    # like "**F71D: Charges**" / "**F48: Period for Presentation in Days**"
    # / "**F49: Confirmation Instructions**". These come from the upstream
    # OCR/LLM emitting markdown-styled labels that leak into the LC body.
    # We keep the label text itself (in case it's part of a clause body)
    # but drop the surrounding ** markers entirely.
    v = re.sub(
        r'\*{2,}\s*(F?\d{2}[A-Z]?\s*:\s*[^*\n]+?)\s*\*{2,}',
        r'\1',
        v,
    )
    # Also strip standalone bold-emphasized lines that ONLY contain
    # a SWIFT field label (with no surrounding clause content) —
    # they leak in as decorative headers from the LLM.
    v = re.sub(
        r'^\s*\*{2,}\s*F?\d{2}[A-Z]?\s*:\s*[^*\n]+?\s*\*{2,}\s*$',
        '',
        v,
        flags=re.MULTILINE,
    )
    # Collapse the blank lines the stripping leaves behind
    v = re.sub(r'\n{3,}', '\n\n', v).strip()

    _strip_pat = _FIELD_LABEL_STRIP.get(tag, '')
    if _strip_pat:
        v = re.sub(_strip_pat, '', v, flags=re.IGNORECASE).strip()

    if tag in ('46A', '46B', '47A', '47B', '45A', '45B', '78', '72', '79'):
        v = _restore_inline_sub_bullets(v)

    if tag in ('48', '47A', '46A', '45A', '78', '72'):
        v = re.sub(r'(?:^|\n)\s*Days:?\s*', '\n', v).strip()
        v = re.sub(r'(?:^|\n)\s*Narrative:?\s*/?\s*', '\n', v).strip()

    v = re.sub(r'-?\s*Party\s+Identifier\s*-?\s*Identifier\s*(?:Code)?\s*\n?', '', v, flags=re.IGNORECASE).strip()

    v = re.sub(r'(?:^|\n)\s*Party\s+Identifier:?\s*\n', '\n', v, flags=re.IGNORECASE).strip()
    v = re.sub(r'^-\s+', '', v).strip()
    v = re.sub(r'Identifier\s+Code:?\s*\n?', '', v, flags=re.IGNORECASE).strip()
    v = re.sub(r'^Identifier:?\s*\n?', '', v, flags=re.IGNORECASE).strip()
    v = re.sub(r'^Name\s+and\s+Address:?\s*\n?', '', v, flags=re.IGNORECASE).strip()

    v = re.sub(r'\.{2,}\s*By\s*\.{2,}\s*-?\s*(?:Name\s+and\s+Address\s*-?\s*)*:?\s*[\n\r]*',
               '', v, flags=re.IGNORECASE).strip()
    v = re.sub(r'-\s*Name\s+and\s+Address\s*-?\s*(?:Name\s+and\s+Address)?:?\s*[\n\r]*',
               '', v, flags=re.IGNORECASE).strip()

    v = re.sub(
        r'^(?:UNDER\s+)?FIELD\s+\d{2}[A-Z]?\s*,?\s*(?:NOW\s+)?TO\s+(?:BE\s+)?READ\s+AS\s*,?\s*[\'"]?\s*',
        '', v, flags=re.IGNORECASE).strip()

    v = re.sub(r'[\'"]?\s*I\s*/?\s*O\s+EXISTING\s*', '', v, flags=re.IGNORECASE).strip()
    v = re.sub(r'[\'"]?\s*I/O\s+[\'"].*?[\'"]', '', v, flags=re.IGNORECASE).strip()

    if tag == '32B':
        v = re.sub(r'\b(?:US\s+DOLLAR|EURO|POUND\s+STERLING|JAPANESE\s+YEN)\s*[\n\r]*',
                   '', v, flags=re.IGNORECASE).strip()

    v = re.sub(r'^Applicable\s+Rules:?\s*\n?', '', v, flags=re.IGNORECASE).strip()
    v = re.sub(r'^Available\s+With.*?Code\s*\n?', '', v, flags=re.IGNORECASE).strip()
    v = re.sub(r'^Drawee\s*-?\s*Party.*?Code\s*\n?', '', v, flags=re.IGNORECASE).strip()
    v = re.sub(r'^Applicant\s+Bank\s*-?\s*Party.*?Code\s*\n?', '', v, flags=re.IGNORECASE).strip()
    v = re.sub(r'^Issuing\s+Bank\s*-?\s*Party.*?Code\s*\n?', '', v, flags=re.IGNORECASE).strip()
    v = re.sub(r'^Reimbursing\s+Bank\s*-?\s*Party.*?Code\s*\n?', '', v, flags=re.IGNORECASE).strip()
    v = re.sub(r'^[\'"]?Advise\s+Through[\'"]?\s+Bank\s*-?\s*Party.*?Code\s*\n?',
               '', v, flags=re.IGNORECASE).strip()

    v = re.sub(r'\s*Other\s*\n?\s*Delivery\s+overdue.*$', '', v, flags=re.IGNORECASE | re.DOTALL).strip()
    v = re.sub(r'\s*Delivery\s+overdue\s+warning.*$', '', v, flags=re.IGNORECASE | re.DOTALL).strip()
    v = re.sub(r'\s*Network\s+delivery\s+notif.*$', '', v, flags=re.IGNORECASE | re.DOTALL).strip()
    v = re.sub(r'\s*Payment\s+Confirmation\s+Status.*$', '', v, flags=re.IGNORECASE | re.DOTALL).strip()
    v = re.sub(r'\s*Confirmed\s+(?:Currency|Amount|Date)\s*:?\s*\n?.*$', '', v, flags=re.IGNORECASE | re.DOTALL).strip()

    v = re.sub(r'\s*(?:Confirmed\s+){2,}.*$', '', v, flags=re.IGNORECASE | re.DOTALL).strip()

    v = re.sub(
        r'(?m)^[ \t]*Page\s+\d+\s+of\s+\d+[ \t]*\n?', '', v,
    )
    v = re.sub(r'[ \t]*Page\s+\d+\s+of\s+\d+[ \t]*', '', v).strip()

    v = re.sub(
        r'\n\s*Report\s+Footer\b.*$',
        '', v, flags=re.IGNORECASE | re.DOTALL).strip()
    v = re.sub(
        r'\n\s*Number\s+of\s+Entities\b.*$',
        '', v, flags=re.IGNORECASE | re.DOTALL).strip()
    v = re.sub(
        r'\n\s*End\s+of\s+Report\b.*$',
        '', v, flags=re.IGNORECASE | re.DOTALL).strip()

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

    if tag == '47A':
        v = re.sub(
            r'\n\s*DOCUMENTS\s+PRESENTED\s+\d+\s+DAYS\s+AFTER\s+BILL\s+OF\s+LADING.*$',
            '', v, flags=re.IGNORECASE | re.DOTALL).strip()
        v = re.sub(
            r'\n\s*\d+\s+DAYS\s+FROM\s+SHIPMENT\s+DATE\s+BUT\s+WITHIN.*$',
            '', v, flags=re.IGNORECASE | re.DOTALL).strip()

    v = re.sub(r'\s*/?\(CONT\.?\s*(?:FROM|IN)\s+FIELD\s+\w+\).*', '', v, flags=re.IGNORECASE | re.DOTALL).strip()
    v = re.sub(r'/\(CONT\.?\s*(?:FROM|IN)\s+FIELD\s+\w+\).*', '', v, flags=re.IGNORECASE | re.DOTALL).strip()

    _ftag_merge = re.search(r'F\d{2}[A-Z]?\s*:', v)
    if _ftag_merge:
        v = v[:_ftag_merge.start()].strip()

    v = re.sub(r'\bPage\s+\d+\s+of\s+\d+\b', '', v, flags=re.IGNORECASE).strip()

    v = re.sub(r'There is no visible text.*?(?:clearly visible|another version)[.\s]*',
               '', v, flags=re.IGNORECASE | re.DOTALL).strip()
    v = re.sub(r'The image appears to be blank[.\s]*',
               '', v, flags=re.IGNORECASE).strip()

    v = re.sub(
        r'\s*The (?:provided )?image is (?:entirely |completely )?blank[^.]*\.\s*'
        r'(?:Therefore,?[^.]*\.\s*)?',
        ' ', v, flags=re.IGNORECASE,
    ).strip()

    v = re.sub(r'\bFRM\b', 'FROM', v)
    v = re.sub(r'\bWTHN\b', 'WITHIN', v)
    v = re.sub(r'\bSHPMNT\b', 'SHIPMENT', v)
    v = re.sub(r'\bDOCS?\b(?=\s+(?:PRESENTED|REQUIRED|MUST))', 'DOCUMENTS', v)

    v = re.sub(r'-\s*\n\s*', '-', v)

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

    v = re.sub(r'^-\s+', '', v).strip()

    if tag in ('31C', '31D', '44C'):

        if len(v) >= 7 and v[:6].isdigit() and v[6].isalpha():
            v = v[:6] + ' ' + v[6:]

        _dm = re.search(r'(\d{6})\s+(\d{4})\s+(\w{3})\s+(\d{1,2})', v)
        if _dm:
            _months = {'Jan':'01','Feb':'02','Mar':'03','Apr':'04','May':'05','Jun':'06',
                       'Jul':'07','Aug':'08','Sep':'09','Oct':'10','Nov':'11','Dec':'12'}
            _date_str = f"{_dm.group(2)}-{_months.get(_dm.group(3),'01')}-{int(_dm.group(4)):02d}"
            v = (v[:_dm.start()] + _date_str + v[_dm.end():]).strip()

            v = re.sub(r'\b\d{6}\b\s*', '', v).strip()
        else:

            _raw_date = re.search(r'\b(\d{2})(\d{2})(\d{2})\b', v)
            if _raw_date:
                _yy, _mm, _dd = _raw_date.group(1), _raw_date.group(2), _raw_date.group(3)
                _year = f"20{_yy}" if int(_yy) < 80 else f"19{_yy}"
                _date_str = f"{_year}-{_mm}-{_dd}"
                v = v[:_raw_date.start()] + _date_str + v[_raw_date.end():]
                v = v.strip()

    v = re.sub(r'^[\.,;:]\s*[\n\r]+', '', v).strip()
    v = re.sub(r'^[\.,;:]+\s*(?=[A-Za-z0-9])', '', v).strip()

    if tag in ('32B', '33B'):
        _ccy = re.search(r'([A-Z]{3})(?=\s|\d|$)', v)
        _ccy_str = _ccy.group(1) if _ccy else 'USD'

        _am = re.search(r'#([\d,]+\.\d+)#?', v)
        if _am:
            v = f"{_ccy_str} {_am.group(1)}"
        else:

            _am_bare = re.search(r'#([\d,]+)\.?#', v)
            if _am_bare:
                _raw = _am_bare.group(1).replace(',', '')
                try:
                    v = f"{_ccy_str} {float(_raw):,.2f}"
                except ValueError:
                    pass
            else:

                _am2 = re.search(r'(\d[\d.]*,\d{2})\b', v)
                if _am2:
                    _amt = _am2.group(1).replace('.', '').replace(',', '.')
                    try:
                        v = f"{_ccy_str} {float(_amt):,.2f}"
                    except ValueError:
                        pass
                else:

                    _am3 = re.search(r'([A-Z]{3})\s*([\d,]+(?:\.\d{0,2})?)[,.\s]*$', v)
                    if _am3:
                        _raw = _am3.group(2).rstrip(',.')
                        try:

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

    v = re.sub(r'-?\s*Party\s+Identifier\s*-?\s*Identifier\s*(?:Code)?\s*\n?',
               '', v, flags=re.IGNORECASE).strip()

    v = re.sub(r'(?:^|\n)\s*Party\s+Identifier:?\s*\n',
               '\n', v, flags=re.IGNORECASE).strip()

    v = re.sub(r'^-\s+', '', v).strip()

    v = re.sub(r'Identifier\s+Code:?\s*\n?', '', v, flags=re.IGNORECASE).strip()
    v = re.sub(r'^Identifier:?\s*\n?', '', v, flags=re.IGNORECASE).strip()

    v = re.sub(r'^Name\s+and\s+Address:?\s*\n?', '', v, flags=re.IGNORECASE).strip()
    v = re.sub(r'-\s*Name\s+and\s+Address\s*-?\s*(?:Name\s+and\s+Address)?:?\s*[\n\r]*',
               '', v, flags=re.IGNORECASE).strip()

    v = re.sub(r'\.{2,}\s*By\s*\.{2,}\s*-?\s*(?:Name\s+and\s+Address\s*-?\s*)*:?\s*[\n\r]*',
               '', v, flags=re.IGNORECASE).strip()

    v = re.sub(r'^Drawee\s*-?\s*Party.*?Code\s*\n?', '', v, flags=re.IGNORECASE).strip()
    v = re.sub(r'^Applicant\s+Bank\s*-?\s*Party.*?Code\s*\n?', '', v, flags=re.IGNORECASE).strip()
    v = re.sub(r'^Issuing\s+Bank\s*-?\s*Party.*?Code\s*\n?', '', v, flags=re.IGNORECASE).strip()
    v = re.sub(r'^Reimbursing\s+Bank\s*-?\s*Party.*?Code\s*\n?', '', v, flags=re.IGNORECASE).strip()
    v = re.sub(r'^[\'"]?Advise\s+Through[\'"]?\s+Bank\s*-?\s*Party.*?Code\s*\n?',
               '', v, flags=re.IGNORECASE).strip()

    v = re.sub(r'^Available\s+With.*?Code\s*\n?', '', v, flags=re.IGNORECASE).strip()

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

        if tag in ('21', '23', '26E', '27'):
            if tag == '26E':
                record.amendment_number = _parse_amendment_number(sf.value) or amendment_number
            continue

        if tag == '30':
            record.amendment_date = sf.value
            continue

        _is_amount_field = (tag == '34B' or
                            (tag == '32B' and re.search(r'increase', sf.value, re.IGNORECASE)))
        if _is_amount_field:
            old_val = base_fields.get('32B', '')
            new_val = sf.value

            _is_increase = bool(re.search(r'increase', new_val, re.IGNORECASE))
            new_val = re.sub(r'(?i)^(?:Increase\s+of\s+Documentary\s+Credit\s+Amount|'
                             r'Currency\s+Code,?\s*Amount)\s*[\n\r]*', '', new_val).strip()
            if _is_increase and old_val:

                def _parse_amt_str(s):
                    if not s:
                        return None
                    s = s.replace(' ', '')

                    m = re.search(r'#([\d,]+\.\d+)#?', s)
                    if m:
                        return float(m.group(1).replace(',', ''))

                    m = re.search(r'(\d{1,3}(?:,\d{3})+\.\d+)', s)
                    if m:
                        return float(m.group(1).replace(',', ''))

                    m = re.search(r'(\d+,\d{2})\b', s)
                    if m:
                        return float(m.group(1).replace(',', '.'))

                    m = re.search(r'(\d+\.\d+)', s)
                    if m:
                        return float(m.group(1))

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

        _is_decrease_field = (tag == '33B' and
                              re.search(r'decrease', sf.value, re.IGNORECASE))
        if _is_decrease_field:
            old_val = base_fields.get('32B', '')
            new_val = sf.value
            new_val = re.sub(
                r'(?i)^(?:Decrease\s+of\s+Documentary\s+Credit\s+Amount|'
                r'Additional\s+Amount\s+Covered)\s*[\n\r]*',
                '', new_val).strip()

            def _parse_amt_str_p280(s):
                if not s:
                    return None
                s = s.replace(' ', '')
                m = re.search(r'#([\d,]+\.\d+)#?', s)
                if m:
                    return float(m.group(1).replace(',', ''))
                m = re.search(r'(\d{1,3}(?:,\d{3})+\.\d+)', s)
                if m:
                    return float(m.group(1).replace(',', ''))
                m = re.search(r'(\d+,\d{2})\b', s)
                if m:
                    return float(m.group(1).replace(',', '.'))
                m = re.search(r'(\d+\.\d+)', s)
                if m:
                    return float(m.group(1))
                m = re.search(r'(\d{3,})', s)
                if m:
                    return float(m.group(1))
                return None

            _ccy = (re.search(r'\b([A-Z]{3})\b', old_val)
                    or re.search(r'\b([A-Z]{3})\b', new_val))
            _o = _parse_amt_str_p280(old_val)
            _n = _parse_amt_str_p280(new_val)
            if _o is not None and _n is not None and _ccy:
                _total = _o - _n
                _new_amt = f"{_ccy.group(1)} {_total:,.2f}"
                base_fields['32B'] = _new_amt
                record.fields_changed.append('32B')
                record.change_details['32B'] = {
                    'old': old_val, 'new': _new_amt,
                    'via': '33B', 'operation': 'decrease',
                }

            continue

        actual_tag = tag
        if tag.endswith('B') and tag[:-1] + 'A' in SWIFT_FIELD_LABELS:
            base_tag = tag[:-1] + 'A'
            if base_tag in base_fields or tag in ('45B', '46B', '47B'):
                actual_tag = base_tag

        old_val = base_fields.get(actual_tag, '')

        amd_val = sf.value.strip()

        _label_strip = _FIELD_LABEL_STRIP.get(actual_tag, _FIELD_LABEL_STRIP.get(tag))
        if _label_strip:
            amd_val = re.sub(_label_strip, '', amd_val, flags=re.IGNORECASE).strip()

        amd_val = re.sub(r'(?:^|\n)\s*Lines?\s+\d+(?:\s*[-–]\s*\d+)?\s*(?:\n|$)', '\n', amd_val).strip()

        amd_val = re.sub(r'(?:^|\n)\s*Code\s*:\s*', '\n', amd_val).strip()

        amd_val = re.sub(r'(?:^|\n)\s*Narrative\s*:\s*', '\n', amd_val).strip()

        amd_val = re.sub(r'//(REPALL|ADD|DEL|DELETE)//', r'/\1/', amd_val, flags=re.IGNORECASE)
        amd_val = re.sub(r'//(REPALL|ADD|DEL|DELETE)/', r'/\1/', amd_val, flags=re.IGNORECASE)
        amd_val = re.sub(r'/(REPALL|ADD|DEL|DELETE)//', r'/\1/', amd_val, flags=re.IGNORECASE)

        amd_val = re.sub(
            r'((?:^|\n)\s*)/(REPALL|DELETE|DEL(?!ETE)|ADD)[ \t]*(?=[A-Za-z])',
            r'\1/\2/\n',
            amd_val, flags=re.IGNORECASE,
        )

        _is_to_read_as = bool(re.search(
            r'TO\s+READ\s+AS\b.*?\bINSTEAD\s+OF\b',
            amd_val, re.IGNORECASE | re.DOTALL,
        ))
        if (re.search(r'/ADD/|/DEL/|/DELETE/|/REPALL/|PLEASE\s+READ', amd_val, re.IGNORECASE)
                or re.search(r'(?:^|\n)\+?\s*\)', amd_val)
                or _is_to_read_as
                or re.search(r'UNDER\s+FIELD\s+\d{2}[A-Z]?\s+ADD\b', amd_val, re.IGNORECASE)):

            _clauses_changed: List[int] = []
            _clauses_deleted: List[int] = []
            new_val = _apply_text_amendment(
                old_val, amd_val,
                _clauses_changed_out=_clauses_changed,
                _clauses_deleted_out=_clauses_deleted,
            )
            base_fields[actual_tag] = new_val
            if old_val != new_val:
                record.fields_changed.append(actual_tag)
                _details = {
                    'old': old_val,
                    'new': new_val,
                    'operation': 'text_amendment',
                    'ops': _extract_swift_ops(amd_val),
                }

                if _clauses_changed:
                    _details['clauses_changed'] = list(_clauses_changed)
                if _clauses_deleted:
                    _details['clauses_deleted'] = list(_clauses_deleted)

                if (
                    'clauses_changed' not in _details
                    and 'clauses_deleted' not in _details
                    and old_val != new_val
                ):
                    try:
                        _cc_diff, _cd_diff = _p247_diff_clauses(
                            actual_tag, old_val, new_val,
                        )
                    except Exception:
                        _cc_diff, _cd_diff = [], []
                    if _cc_diff:
                        _details['clauses_changed'] = _cc_diff
                    if _cd_diff:
                        _details['clauses_deleted'] = _cd_diff

                record.change_details[actual_tag] = _details
        else:

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

            clean_val = _strip_field_sub_labels(actual_tag, clean_val)

            base_fields[actual_tag] = clean_val if clean_val else amd_val
            if old_val != base_fields[actual_tag]:
                record.fields_changed.append(actual_tag)
                record.change_details[actual_tag] = {'old': old_val, 'new': base_fields[actual_tag]}

    return record

_P245_PROMPT_ADDENDUM = (
    "- \"<TAG>(N):- Read as 'X'\" (UBL form, no INSTEAD OF) = Apply X to clause N. "
    "If X is much shorter than current clause N AND its words overlap with a "
    "sub-phrase inside clause N (e.g. clause has \"WITHIN (05) DAYS AFTER SHIPMENT\" "
    "and X is \"WITHIN (07) WORKING DAYS AFTER SHIPMENT\"), patch ONLY that matching "
    "sub-phrase and preserve the rest of the clause (FAX numbers, emails, addresses, "
    "references). If X is similar in length to clause N or shares no overlapping "
    "sub-phrase, replace clause N wholesale.\n"
)

_P246_UBL_CLAUSE_READAS_RE = re.compile(
    r'(?:^|\n)\s*\d{2}[A-Z]?\s*\(\s*\d{1,3}\s*\)\s*:?\s*-?\s*Read\s+as',
    re.IGNORECASE,
)

def _p247_diff_clauses(tag: str, old_val: str, new_val: str):
    """P247 — Compute (clauses_changed, clauses_deleted) lists by
    diff-comparing the OLD and NEW value of a clause-bearing field.

    LLM-path amendments populate `change_details[tag]['new']` with the
    full updated field text but never tell us WHICH specific clauses
    changed. The renderer needs that info to mark only the actually-
    changed clauses as AMENDED instead of the entire field.

    Strategy:
      1. Split both old + new into clauses (`_split_into_clauses`).
      2. Build a list of normalized clause-text strings for each side.
      3. Match: every NEW clause that doesn't appear in old → CHANGED.
         Every OLD clause that doesn't appear in new → DELETED.
      4. Return 1-based clause numbers using the NEW field's positions
         for `clauses_changed` (so renderer marks the post-renumber
         positions where new content actually sits).

    Returns ([], []) when the diff isn't meaningful (non-clause field,
    REPALL-style wholesale rewrite where every clause differs slightly,
    etc.) — caller falls back to whole-field AMENDED.

    P247 is universal: every LLM-path amendment on a clause field gets
    accurate per-clause AMENDED tracking, regardless of bank or op type.
    """
    if tag not in ('45A', '45B', '46A', '46B', '47A', '47B', '78', '72', '79'):
        return [], []
    if not old_val or not new_val:
        return [], []

    def _norm(s):
        return re.sub(r'\s+', ' ', (s or '')).strip().upper()

    def _extract_numbered(text):
        """Pull (number, normalized_text) pairs straight from raw text.
        `_split_into_clauses` renumbers positionally; for diffing we
        need the ORIGINAL clause numbers as written in the text so a
        post-deletion gap (e.g. ..., 4, 6, 7, ...) keeps clause 6 = 6,
        not renumbered to 5."""
        if not text:
            return []
        items = []
        cur_num = None
        cur_lines: List[str] = []
        for line in text.split('\n'):
            m = re.match(r'^\s*(\d{1,3})\s*[.\)]\s*(.*)', line)
            if m:
                if cur_num is not None:
                    items.append((cur_num, _norm('\n'.join(cur_lines))))
                cur_num = int(m.group(1))
                cur_lines = [m.group(2)] if m.group(2) else []
            elif cur_num is not None:
                cur_lines.append(line)
        if cur_num is not None:
            items.append((cur_num, _norm('\n'.join(cur_lines))))
        return items

    old_pairs = _extract_numbered(old_val)
    new_pairs = _extract_numbered(new_val)
    if not new_pairs:
        return [], []

    old_by_num = dict(old_pairs)
    new_by_num = dict(new_pairs)

    changed: List[int] = []
    for cn, new_t in new_by_num.items():
        if not new_t:
            continue
        old_t = old_by_num.get(cn)
        if old_t is None:

            changed.append(cn)
        elif old_t != new_t:

            changed.append(cn)

    deleted: List[int] = []
    for cn in old_by_num:
        if cn not in new_by_num:
            deleted.append(cn)

    if changed and len(changed) == len(new_pairs):
        return [], []

    return sorted(changed), sorted(deleted)

_VLM_AMENDMENT_PROMPT = """You are an expert SWIFT MT707 amendment processor for Letters of Credit.

I have a base Letter of Credit with the following field values, and an MT707 amendment that modifies some fields.

BASE LC FIELDS (current values before this amendment):
{base_fields_text}

AMENDMENT TEXT (MT707 message):
{amendment_text}

YOUR TASK: Apply the amendment instructions to produce the UPDATED field values.

AMENDMENT INSTRUCTION TYPES:
- "/REPALL/" followed by "UNDER FIELD XXA, NOW TO BE READ AS 'new text'" = Replace the ENTIRE field with the new quoted text
- "/REPALL/" followed directly by field content under F45B/F46B/F47B = Replace the matching A-field entirely with that content; do not include "/REPALL/", "Line N", "Lines N-M", "Code:", or "Narrative:" in the returned value
- "/DELETE/" followed by "CLAUSE 2,3,5,7" = Delete those numbered clauses from the matching A-field
- "/ADD/" followed by "CLAUSE N) text" = Add or replace that numbered clause in the matching A-field; return clean clause text without the word "CLAUSE" or line markers
- "FIELD XXA-N TO READ AS 'new text'" = Replace clause N of field XXA with the new text
- "FIELD XXA-N WORD TO READ AS 'X' I/O 'Y'" = In clause N, replace word Y with word X
- "FIELD XXA-N DELETE WORDING AS 'text'" = Delete that text from clause N
{p245_addendum}- "UNDER FIELD XXA ADD CLAUSE AS 'text1' 'text2'" = Add new clauses to the end of field XXA
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
12. PRESERVE UNNAMED CLAUSES VERBATIM. A clause-bearing field (45A/46A/47A) usually contains many numbered clauses. The amendment narrative ALWAYS names which clause(s) it touches via "FIELD XXA-N ..." (clause N of field XXA) or "UNDER FIELD XXA ADD ..." (append at end). For EVERY OTHER clause in the field — clauses NOT named in any narrative instruction — copy the text byte-for-byte from the BASE LC FIELDS into your output. Do NOT paraphrase, fix typos, normalise spelling, alter wording, change punctuation, or "improve" any clause that the amendment did not explicitly target.
13. NO CLAUSE LOSS. The number of clauses in your output for a field must equal (base count) + (clauses ADDed by the amendment) - (clauses explicitly DELETED). Never silently drop, merge, or substitute one clause with another. If the amendment did not say "DELETE" or "TO READ AS" for clause N, clause N must appear in the output exactly as it was in the base.
14. NO CONTENT FABRICATION. Do not invent clause text. Every clause in your output must either (a) be present verbatim in the BASE LC FIELDS, or (b) be the verbatim quoted value from the amendment narrative. Do not paraphrase amendment text, do not "blend" clauses, and do not duplicate text from one clause into another.
15. NEVER DEDUPLICATE NUMBERED ITEMS. When a /REPALL/ or amendment payload contains multiple numbered items like "1) X" and "2) X" whose body text is identical, this is INTENTIONAL — banks use it for multi-tranche shipments where two parallel quantities of the same goods are shipped under separate proforma invoices. Preserve every numbered item with its full body, even if items 1 and 2 look like exact duplicates. Repetition is a signal, not a typo. Loss of a numbered item changes the total goods quantity and breaks F32B amount reconciliation downstream.

Example for ADD: If amendment says "UNDER FIELD 47A ADD CLAUSE AS 'CHARTER PARTY B/L ACCEPTABLE'",
return: {{"47A_ADD": "CHARTER PARTY B/L ACCEPTABLE"}}

Example for word change: If amendment says "FIELD 46A-2 WORD TO READ AS 'CLEAN ON BOARD' I/O 'CLEAN ON BOARD'",
return the full 46A with the word changed in clause 2.

Example for preservation (rule 12): If amendment says "FIELD 47A-1 TO READ AS 'NEW TEXT'" and the base has 7 clauses, your output 47A must have 7 clauses where ONLY clause 1's text is the new value; clauses 2-7 are byte-for-byte the base values.

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

    _amd_num_m = re.search(r'(?:F?26E|Number\s+of\s+Amendment)\s*:?\s*(\d+)', amendment_text, re.IGNORECASE)
    if _amd_num_m:
        record.amendment_number = int(_amd_num_m.group(1))
    _date_m = re.search(r'(?:F?30|Date\s+of\s+Amendment)\s*:?\s*(\d{6})\s+(\d{4}\s+\w+\s+\d+)?', amendment_text, re.IGNORECASE)
    if _date_m:
        record.amendment_date = _date_m.group(0).strip()

    _amd_upper = amendment_text.upper()
    _touched_tags = set()
    for _t in ['45A', '45B', '46A', '46B', '47A', '47B', '32B', '34B',
               '31D', '44C', '44E', '44F', '48', '50', '59', '71D', '78']:
        if _t in _amd_upper or f'F{_t}' in _amd_upper or f'FIELD {_t}' in _amd_upper:

            actual = _t[:-1] + 'A' if _t.endswith('B') and _t not in ('32B', '34B', '71B') else _t
            if actual == '34B':
                actual = '32B'
            _touched_tags.add(actual)

    _touched_tags.update(['20', '32B'])

    base_text_parts = []
    for tag in sorted(_touched_tags):
        val = base_fields.get(tag, '')
        if val:

            max_len = 2500 if tag in ('46A', '47A', '45A', '78') else 300
            val_preview = str(val)[:max_len]
            base_text_parts.append(f"F{tag}: {val_preview}")
    base_fields_text = '\n\n'.join(base_text_parts)

    _amd_upper_check = amendment_text.upper()

    _has_add_clause = bool(re.search(
        r'ADD\s+(?:LOI\s+)?CLAUSE\s+AS\b', _amd_upper_check
    ))

    _has_other_ops = any(kw in _amd_upper_check for kw in [
        'WORD TO READ AS', 'DELETE WORDING', 'I/O', 'INCREASE OF DOCUMENTARY',
        'TO READ AS',
    ])

    if _has_add_clause and not _has_other_ops:

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
                            'clauses_changed': [_next_num],
                        }
                        if _progress:
                            _progress(f"      ADD clause to {_add_tag}: {len(_add_text)} chars via LLM")

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

    clean_amd = amendment_text
    clean_amd = re.sub(r'(?:^|\n)\s*Narrative\s*:\s*', '\n', clean_amd).strip()
    clean_amd = re.sub(r'(?:^|\n)\s*Lines?\s+\d+(?:\s*[-–]\s*\d+)?\s*(?:\n|$)', '\n', clean_amd).strip()
    clean_amd = re.sub(r'(?:^|\n)\s*Code\s*:\s*', '\n', clean_amd).strip()
    clean_amd = re.sub(r'\s*Other\s*\n?\s*Delivery\s+overdue.*$', '', clean_amd, flags=re.IGNORECASE | re.DOTALL).strip()
    clean_amd = re.sub(r'\s*Page\s+\d+\s+of\s+\d+\s*', ' ', clean_amd).strip()
    clean_amd = re.sub(r'\{CHK:[A-F0-9]+\}', '', clean_amd).strip()
    clean_amd = re.sub(r'Block\s+5\s*', '', clean_amd).strip()

    clean_amd = re.sub(r'Report\s+(?:Header|Footer|Content).*?(?=\n[A-Z]|\Z)', '', clean_amd, flags=re.IGNORECASE | re.DOTALL).strip()
    clean_amd = re.sub(r'Message\s+(?:Header|Identifier|Details).*?(?=\n[A-Z]|\Z)', '', clean_amd, flags=re.IGNORECASE | re.DOTALL).strip()
    clean_amd = re.sub(r'(?:Delivery\s+overdue|Network\s+delivery|Payment\s+Confirmation|Confirmed\s+(?:Currency|Amount|Date)).*$', '', clean_amd, flags=re.IGNORECASE | re.DOTALL).strip()

    _p245_addendum = (
        _P245_PROMPT_ADDENDUM
        if _P246_UBL_CLAUSE_READAS_RE.search(amendment_text or '')
        else ''
    )

    prompt = (
        _VLM_AMENDMENT_PROMPT
        .replace('{base_fields_text}', base_fields_text)
        .replace('{amendment_text}', clean_amd[:12000])
        .replace('{p245_addendum}', _p245_addendum)
    )

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

            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                changes = json.loads(json_match.group(0))
                if isinstance(changes, dict) and changes:
                    for raw_tag, new_val in changes.items():

                        tag = re.sub(r'^F', '', raw_tag)

                        is_add = tag.endswith('_ADD')
                        if is_add:
                            tag = tag[:-4]

                        if tag in ('26E', '27', '30', '22A', '23', '21'):
                            continue

                        if tag not in EXTRACTION_TAGS:
                            if _progress:
                                _progress(f"      LLM returned non-LC tag {tag!r} — dropping (not in EXTRACTION_TAGS)")
                            continue
                        new_val = str(new_val).strip()
                        if not new_val:
                            continue
                        old_val = base_fields.get(tag, '')
                        if is_add and old_val:

                            new_val = old_val.rstrip() + '\n' + new_val
                        base_fields[tag] = new_val
                        if old_val != new_val:
                            record.fields_changed.append(tag)
                            _det = {
                                'old': old_val,
                                'new': new_val,
                                'operation': 'vlm_amendment',
                            }

                            _cc, _cd = _p247_diff_clauses(tag, old_val, new_val)
                            if _cc:
                                _det['clauses_changed'] = _cc
                            if _cd:
                                _det['clauses_deleted'] = _cd
                            record.change_details[tag] = _det

                    _p249_dw_re = re.compile(
                        r'FIELD\s+(\d{2}[A-Z]?)-?\d*\s+DELETE\s+WORDING\s+AS\s*\n?\s*'
                        r'[\'"]+\s*([^\'"\n\r]+?)\s*[\'"]+',
                        re.IGNORECASE,
                    )

                    for _m in _p249_dw_re.finditer(clean_amd or ''):
                        _dw_raw_tag = _m.group(1).upper()
                        _dw_text = _m.group(2).strip()
                        if not _dw_text:
                            continue

                        _dw_tag = _dw_raw_tag
                        if _dw_tag.endswith('B') and _dw_tag[:-1] + 'A' in SWIFT_FIELD_LABELS:
                            _dw_tag = _dw_tag[:-1] + 'A'
                        _cur = base_fields.get(_dw_tag, '')
                        if not _cur or _dw_text not in _cur:
                            continue
                        _new = _cur.replace(_dw_text, '').rstrip()

                        _new = re.sub(r'  +', ' ', _new)
                        _new = re.sub(r'\n\s*\n\s*\n+', '\n\n', _new)
                        if _new != _cur:
                            base_fields[_dw_tag] = _new
                            if _dw_tag not in record.fields_changed:
                                record.fields_changed.append(_dw_tag)
                            _existing_det = record.change_details.get(_dw_tag, {}) or {}
                            _existing_det['new'] = _new
                            _existing_det.setdefault('old', _cur)
                            _existing_det['operation'] = 'vlm_amendment+p249_delete_wording'
                            record.change_details[_dw_tag] = _existing_det
                            if _progress:
                                _progress(
                                    f"      P249: removed missed DELETE WORDING "
                                    f"'{_dw_text[:40]}' from F{_dw_tag}"
                                )

                    _p251_addclause_augment(base_fields, clean_amd, record, _progress)
                    if _progress:
                        _progress(f"      VLM amendment applied: {record.fields_changed}")
                    return record
        if _progress:
            _err_body = resp.text[:300] if resp.text else ''
            _progress(f"      LLM amendment failed (status {resp.status_code}): {_err_body}, falling back to regex")
    except Exception as e:
        if _progress:
            _progress(f"      VLM amendment error: {e}, falling back to regex")

    return None

def _parse_amendment_number(value: str) -> Optional[int]:
    """Extract amendment number from F26E value."""
    m = re.search(r'(\d+)', value)
    return int(m.group(1)) if m else None

def _sort_amendments(amendment_packets: list) -> list:
    """Sort amendment packets by F26E number or F30 date.

    P267 — tolerance for the SWIFT-export "labelled" form many banks
    (notably BAHL) emit, where the F-tag prefix is followed by the
    spec label name on one line and the actual value on the next:

        F26E: Number of Amendment
         13
        F30: Date of Amendment
         260427 2026 Apr 27

    The pre-P267 regex `(?:F?26E\\s*:?\\s*|:26E:|amendment\\s*(?:no\\.?|
    number)\\s*:?\\s*)(\\d+)` required the digit to follow `F26E:`
    immediately (with only whitespace between). Whitespace `\\s*` is
    GREEDY but will only consume whitespace — it stops at "Number"
    (the label letter), and `(\\d+)` then fails because the next char
    is a letter. Result: every BAHL amendment returned the 9999
    fallback key, Python's stable sort preserved the SWIFT-export's
    newest-first packet order, and amendments were applied in REVERSE
    chronological order (amd 13 first, amd 1 last). amd 1's REPALL of
    F45B then ran LAST and wiped every later /ADD/ append's effect on
    F45A; amd 2's F31D / F44C overwrote amds 4/6/7's later values.

    Fix: insert a lazy `[^\\d]{0,80}?` segment between the F-tag prefix
    and the captured digits, so the regex tolerates an optional label
    name + newline between the tag and the value. The 80-char cap
    prevents pathological matches across unrelated text. Existing
    compact forms (`F26E: 13`, `:26E:13`, `Amendment Number 13`) keep
    working because the lazy segment matches zero chars when the digit
    is already adjacent.
    """
    def _get_sort_key(pkt):

        text = _get_packet_refined_text(pkt)
        date_key = 99999999
        m_f30 = re.search(
            r'(?:F?30\s*:?|:30:)\s*[^\d]{0,80}?(\d{6})\b',
            text, re.IGNORECASE,
        )
        if m_f30:
            yymmdd = m_f30.group(1)
            yy = int(yymmdd[:2])
            century = 2000 if yy < 80 else 1900
            date_key = (century + yy) * 10000 + int(yymmdd[2:])
        else:
            m_ack = re.search(
                r'(?:ACK[/\\]?NAK\s+Reception\s+)?Date\s*[/\-]?\s*Time'
                r'[^\n]*?(\d{4})[/\-](\d{2})[/\-](\d{2})',
                text, re.IGNORECASE,
            )
            if m_ack:
                date_key = (int(m_ack.group(1)) * 10000
                            + int(m_ack.group(2)) * 100
                            + int(m_ack.group(3)))
        sub_key = 9999
        m_f26e = re.search(
            r'(?:F?26E\s*:?|:26E:|amendment\s*(?:no\.?|number)\s*:?)'
            r'\s*[^\d]{0,80}?(\d+)',
            text, re.IGNORECASE,
        )
        if m_f26e:
            sub_key = int(m_f26e.group(1))
        return (date_key, sub_key)

    return sorted(amendment_packets, key=_get_sort_key)

_PAGE_TEXT_LOOKUP = {}

_STEP01_RAW_PAGE_LOOKUP = {}
_STEP02_PAGE_CORRECTIONS = {}

def _recover_f45a_layout_from_step01(consolidated_f45a: str, candidate_pages=None) -> str:
    """P217 — restore F45A multi-paragraph line breaks from step01 raw OCR.

    When step02's VLM-primary path replaces GLM raw text (rule
    'vlm_primary' or 'vlm_full_extraction' in the page's corrections),
    Qwen2.5-VL collapses the natural line breaks inside multi-paragraph
    F45A goods descriptions. step01's GLM raw text preserves them.

    Detection (all must hold for fallback to fire):
      • F45A consolidated value has ≤ 1 newline (clearly inline)
      • F45A consolidated value is substantial (≥ 80 chars)
      • Some candidate page in step01 raw OCR contains an F45A region
        whose content matches the collapsed value (whitespace-normalised
        character compare)
      • That same page's step02 corrections include a VLM replacement
        rule ('vlm_primary' / 'vlm_full_extraction')

    Why we iterate ALL candidate pages: SwiftField.source_page in step06
    is set to the MT700 packet's first page, not the page that actually
    held the F45A value (the MT700 packet may span pages 4–7 with F45A
    on page 5). Iterating the candidate pages and matching by content
    finds the right one regardless.

    If a match is found, return the multi-line region from that step01
    page. Otherwise return the input unchanged. The whitespace-normalised
    match guards against substituting different content (page mismatch /
    OCR drift) — that branch silently keeps the input.

    Verified by inspection across the dataset:
      • LC 5001LC60733 (job c7d454be): step02 page corrections empty —
        VLM did not replace GLM, F45A already multi-line. Gates fail
        (multi-line check + no vlm_primary) → no substitution.
      • LC 0005LC90854 (job d28518f5): step02 page 5 has vlm_primary,
        step01 page 5 contains the matching F45A region. Gates pass →
        multi-line value restored.
    """
    if not consolidated_f45a:
        return consolidated_f45a
    if consolidated_f45a.count('\n') >= 2:
        return consolidated_f45a
    if len(consolidated_f45a) < 80:
        return consolidated_f45a

    def _norm(s: str) -> str:

        s = re.sub(r'\s+', ' ', s).strip().lower()
        s = re.sub(
            r'^description\s+of\s+goods(?:\s+and\s*/?\s*or\s+services)?\s*',
            '', s,
        )
        return s

    target_norm = _norm(consolidated_f45a)

    if candidate_pages:
        pages_to_try = list(candidate_pages)
    else:
        pages_to_try = list(_STEP01_RAW_PAGE_LOOKUP.keys())

    _STEP01_LABEL_RE = re.compile(
        r'(?:^|\n)\s*Description\s+of\s+Goods(?:\s+and/or\s+Services)?\s*\n',
        re.IGNORECASE,
    )
    _F45A_END_RE = re.compile(
        r'(?:^|\n)\s*(?:'
        r'Documents\s+Required'
        r'|Additional\s+Conditions'
        r'|Period\s+for\s+Presentation'
        r'|Confirmation\s+Instructions'
        r'|Applicant\b'
        r'|Beneficiary\b'
        r'|Charges\b'
        r'|Instructions\s+to'
        r'|F\d{2}[A-Z]?\s*:'
        r')',
        re.IGNORECASE,
    )

    for pn in pages_to_try:
        s1_text = _STEP01_RAW_PAGE_LOOKUP.get(pn, '') or ''
        if not s1_text:
            continue
        start_m = _STEP01_LABEL_RE.search(s1_text)
        if not start_m:
            continue
        rest = s1_text[start_m.end():]
        end_m = _F45A_END_RE.search(rest)
        if not end_m:
            continue
        extracted = rest[:end_m.start()].strip()
        if not extracted:
            continue
        if _norm(extracted) != target_norm:
            continue

        page_corr = _STEP02_PAGE_CORRECTIONS.get(pn, []) or []
        if not any(rule in page_corr for rule in ('vlm_primary', 'vlm_full_extraction')):
            continue
        return extracted

    return consolidated_f45a

def _get_packet_refined_text(pkt) -> str:
    """Get concatenated text from a packet using page_numbers -> page_texts lookup."""

    page_nums = pkt.get('page_numbers', []) if isinstance(pkt, dict) else getattr(pkt, 'page_numbers', [])
    if page_nums and _PAGE_TEXT_LOOKUP:
        texts = []
        for pn in page_nums:
            t = _PAGE_TEXT_LOOKUP.get(pn, '')
            if t:
                texts.append(t)
        if texts:
            return '\n'.join(texts)

    pages = pkt.pages if hasattr(pkt, 'pages') else pkt.get('pages', [])
    texts = []
    for p in pages:
        if hasattr(p, 'refined_text'):
            t = p.refined_text
        elif isinstance(p, dict):
            t = p.get('refined_text', p.get('cleaned_text', p.get('raw_text', '')))
        elif isinstance(p, int):

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
- P238 — Preserve cross-field continuation markers verbatim inside the field
  value: "(CONT FROM FIELD XX)", "(CONT. FROM FIELD XX)", "(CONT FORM FIELD
  XX)" (OCR typo of FROM), "/(CONT FROM FIELD XX)", "Narrative: /(CONT FROM
  FIELD XX)". Do NOT drop or summarise these markers — downstream
  cross-reference resolution uses them to route content between fields.
- Return empty object {{}} if no SWIFT fields found

DO NOT EXTRACT (these are NOT LC fields — they belong to other SWIFT messages, not MT700/MT707):
- 71A: Reimbursing Bank's Charges (belongs to MT740/MT747 reimbursement authorisation)
- Any tag from narrative text of MT799 tracers, MT754 payment advices, MT730 acknowledgements,
  MT740 reimbursement authorisations, MT747 reimb amendments, MT940 statements.
- If the page is clearly a reimbursement claim, tracer, or acknowledgement message
  (contains phrases like "A.REIM CLAIM", "TRACER", "TOTAL DUE", "HANDLING COMMISSION",
  "REIMBURSEMENT CLAIM"), return empty object {{}}.
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

    results_dir = None
    for pn in page_nums:

        pages_data = base_pkt.get('pages', []) if isinstance(base_pkt, dict) else getattr(base_pkt, 'pages', [])
        for pd in pages_data:
            img = pd.get('page_image_path', '') if isinstance(pd, dict) else getattr(pd, 'page_image_path', '')
            if img and os.path.exists(img):
                results_dir = os.path.dirname(os.path.dirname(img))
                break
        if results_dir:
            break

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

    _ALLOWED_LC_TAGS = frozenset(EXTRACTION_TAGS)
    with ThreadPoolExecutor(max_workers=min(MAX_CONCURRENT_VLM, len(page_items))) as executor:
        futures = {executor.submit(_extract_fields_vlm_page, pn, img, txt): pn
                   for pn, img, txt in page_items}
        for fut in as_completed(futures):
            pn = futures[fut]
            try:
                page_fields = fut.result()
                for tag, val in page_fields.items():

                    canon = tag.lstrip('F').strip()
                    if canon not in _ALLOWED_LC_TAGS:
                        continue
                    if val and canon not in merged_fields:
                        merged_fields[canon] = val
                    elif val and canon in merged_fields and canon in ('46A', '47A', '45A', '78'):

                        merged_fields[canon] = merged_fields[canon] + '\n' + val
                _progress(f"    Page {pn}: {len(page_fields)} fields")
            except Exception as e:
                _progress(f"    Page {pn}: VLM error: {e}")

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

_P240_MARKER_RE = re.compile(
    r'[/\\]?\s*'
    r'\(\s*CONT\.?\s*(?:INUED|INUATION)?\s+(?:FROM|FORM|IN)\s+FIELD\s+(?P<src>\d{2}[A-Z]?)\s*\)[ \t]*'
    r'(?P<chunk>(?:'
    r'(?!\n\s*F?\d{2}[A-Z]?\s*:)'
    r'(?!\s*[/\\]?\s*\(\s*CONT\.?\s*(?:INUED|INUATION)?\s+(?:FROM|FORM|IN)\s+FIELD)'
    r'.)*)',
    re.IGNORECASE | re.DOTALL,
)

def _p240_norm(s: str) -> str:
    return re.sub(r'\s+', ' ', s).strip().upper()

def _p240_strip_pattern(chunk_text: str):
    words = re.findall(r'\S+', chunk_text)
    if not words:
        return None

    return r'\n\s*[\.,;:]?\s*' + r'\s+'.join(re.escape(w) for w in words) + r'\s*'

def _apply_p240_cross_ref_rescue(_cf: dict, _progress, _phase: str = '') -> None:
    """P240 — rescue orphan continuation chunks across SWIFT fields.

    Reads raw step02 page text from the global `_PAGE_TEXT_LOOKUP`, finds
    `(CONT FROM FIELD XX)` markers, and routes the chunk that follows
    to the correct target field. Idempotent — when the target already
    contains the chunk, it skips silently, so this can be called both
    before and after the MT707 amendment loop without double-appending.
    """
    if not _PAGE_TEXT_LOOKUP:
        return
    _phase_tag = f' [{_phase}]' if _phase else ''
    for _pn, _ptext in _PAGE_TEXT_LOOKUP.items():
        if not isinstance(_ptext, str) or 'CONT' not in _ptext.upper():
            continue
        for _m in _P240_MARKER_RE.finditer(_ptext):
            _src_tag = _m.group('src').upper()
            _chunk = (_m.group('chunk') or '').strip()
            if not _chunk or len(_chunk) < 5:
                continue
            _norm_chunk = _p240_norm(_chunk)
            if not _norm_chunk:
                continue

            if _norm_chunk in _p240_norm(str(_cf.get(_src_tag, '') or '')):
                continue

            _leak_tag = None
            for _other_tag, _other_val in list(_cf.items()):
                if _other_tag.startswith('_') or _other_tag == _src_tag:
                    continue
                if not isinstance(_other_val, str) or not _other_val:
                    continue
                if _norm_chunk in _p240_norm(_other_val):
                    _leak_tag = _other_tag
                    break
            if _leak_tag is None:

                if len(_chunk) < 20:
                    continue
                _existing_tgt = str(_cf.get(_src_tag, '') or '').rstrip()
                _cf[_src_tag] = (
                    f"{_existing_tgt}\n{_chunk}".strip()
                    if _existing_tgt else _chunk
                )
                _progress(
                    f"  P240b{_phase_tag}: appended '{_chunk[:60]}{'...' if len(_chunk) > 60 else ''}' "
                    f"to F{_src_tag} (no leak source — chunk dropped or overwritten)"
                )
                continue
            _strip_pat = _p240_strip_pattern(_chunk)
            if not _strip_pat:
                continue
            _new_leak_val = re.sub(
                _strip_pat, '\n', _cf[_leak_tag],
                count=1, flags=re.IGNORECASE,
            ).rstrip()
            _cf[_leak_tag] = _new_leak_val
            _existing_tgt = str(_cf.get(_src_tag, '') or '').rstrip()
            _cf[_src_tag] = (
                f"{_existing_tgt}\n{_chunk}".strip()
                if _existing_tgt else _chunk
            )
            _progress(
                f"  P240{_phase_tag}: rescued '{_chunk[:60]}{'...' if len(_chunk) > 60 else ''}' "
                f"from F{_leak_tag} → F{_src_tag} (raw page-text marker)"
            )

_P251_ADDCLAUSE_BLOCK_RE = re.compile(
    r'UNDER\s+FIELD\s+(?P<tag>\d{2}[A-Z]?)\s+ADD\s+(?:LOI\s+)?CLAUSE\s+AS\s*\n'
    r"(?P<body>(?:.|\n)*?)"
    r'(?=\n\s*(?:'
    r'UNDER\s+FIELD\s+\d{2}[A-Z]?\s+ADD\s+'
    r'|FIELD\s+\d{2}[A-Z]?-?\d*\s+(?:TO\s+READ\s+AS|WORD\s+TO\s+READ\s+AS|DELETE\s+WORDING)'
    r"|/(?:DELETE|DEL|ADD|REPALL)/"
    r')|\Z)',
    re.IGNORECASE,
)

def _p251_extract_quoted_items(body: str):
    """Split SWIFT `''item1''\\n''item2''...` narrative into a list of
    individual quoted items. The boundary between items is `''<newline>''`.
    Inner SWIFT-escaped quotes (e.g. nested `''X''` for a single quote)
    are left as-is in the item text — the renderer / clause splitter
    handles them downstream."""
    body = (body or '').strip()
    if not body:
        return []
    raw_items = re.split(r"''\s*\n\s*''", body)
    out = []
    for it in raw_items:
        it = it.strip()
        if it.startswith("''"):
            it = it[2:]
        if it.endswith("''"):
            it = it[:-2]
        it = it.strip()
        if it:
            out.append(it)
    return out

def _p251_addclause_augment(base_fields: dict, clean_amd: str, record, _progress) -> None:
    """Re-number ADD CLAUSE AS items the LLM appended without numbering.

    See module-level P251 docstring above for context."""
    if not clean_amd:
        return
    norm = lambda s: re.sub(r'\s+', ' ', s or '').strip().upper()

    for m in _P251_ADDCLAUSE_BLOCK_RE.finditer(clean_amd):
        raw_tag = m.group('tag').upper()
        body = m.group('body').strip()

        tag = raw_tag
        if tag.endswith('B') and tag[:-1] + 'A' in SWIFT_FIELD_LABELS:
            tag = tag[:-1] + 'A'
        items = _p251_extract_quoted_items(body)
        if not items:
            continue
        cur = base_fields.get(tag, '') or ''
        if not cur:
            continue

        clause_blocks = list(re.finditer(
            r'(?:^|\n)\s*(\d+)[\.\)]\s+(.+?)(?=\n\s*\d+[\.\)]\s+|\Z)',
            cur, re.DOTALL,
        ))
        if len(clause_blocks) >= len(items):
            last_n = clause_blocks[-len(items):]
            ok = True
            for it, blk in zip(items, last_n):
                anchor = norm(it)[:80]
                if anchor and anchor not in norm(blk.group(2)):
                    ok = False
                    break
            if ok:
                continue

        earliest_pos = len(cur)
        for it in items:
            words = re.findall(r'\S+', it)[:8]
            if not words:
                continue
            try:
                pat = re.compile(
                    r'\b' + r'\s+'.join(re.escape(w) for w in words) + r'\b',
                    re.IGNORECASE,
                )
            except re.error:
                continue
            mp = pat.search(cur)
            if mp and mp.start() < earliest_pos:
                earliest_pos = mp.start()

        if earliest_pos >= len(cur):

            base = cur.rstrip()
        else:

            line_start = cur.rfind('\n', 0, earliest_pos)
            line_start = line_start + 1 if line_start >= 0 else 0
            base = cur[:line_start].rstrip()

        existing_nums = re.findall(r'(?:^|\n)\s*(\d+)[\.\)]\s+', base)
        next_n = max([int(n) for n in existing_nums] + [0]) + 1
        appended = '\n'.join(f'{next_n + i}. {it}' for i, it in enumerate(items))
        new_val = (base + '\n' + appended).strip() if base else appended

        if new_val == cur:
            continue
        base_fields[tag] = new_val
        if tag not in record.fields_changed:
            record.fields_changed.append(tag)
        _det = record.change_details.get(tag, {}) or {}
        _det.setdefault('old', cur)
        _det['new'] = new_val
        _det['operation'] = (
            _det.get('operation', 'vlm_amendment') + '+p251_addclause_renumber'
        )

        _cc, _cd = _p247_diff_clauses(tag, _det['old'], new_val)
        if _cc:
            _det['clauses_changed'] = _cc
        if _cd:
            _det['clauses_deleted'] = _cd
        record.change_details[tag] = _det
        if _progress:
            _progress(
                f"      P251: numbered {len(items)} ADD CLAUSE AS item(s) "
                f"in F{tag} (clauses {next_n}-{next_n + len(items) - 1})"
            )

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

    global _PAGE_TEXT_LOOKUP, _STEP01_RAW_PAGE_LOOKUP, _STEP02_PAGE_CORRECTIONS
    _page_texts = step5_result.get('page_texts', {})
    if _page_texts:
        _PAGE_TEXT_LOOKUP = {int(k): v for k, v in _page_texts.items() if v}
        _progress(f"Page text lookup: {len(_PAGE_TEXT_LOOKUP)} pages with text")

    _step01_raw = step5_result.get('step01_page_texts', {}) or {}
    if _step01_raw:
        _STEP01_RAW_PAGE_LOOKUP = {int(k): v for k, v in _step01_raw.items() if v}
    else:
        _STEP01_RAW_PAGE_LOOKUP = {}
    _step02_corr = step5_result.get('step02_corrections', {}) or {}
    if _step02_corr:
        _STEP02_PAGE_CORRECTIONS = {int(k): v for k, v in _step02_corr.items() if v}
    else:
        _STEP02_PAGE_CORRECTIONS = {}

    packets_in = step5_result.get('packets', [])
    _progress(f"Consolidating Final LC from {len(packets_in)} packets...")

    mt700_packets = []
    mt707_packets = []
    mt799_packets = []
    other_mt_packets = []
    shipping_packets = []
    other_packets = []

    for pkt in packets_in:
        mt = _get_packet_field(pkt, 'mt_type', '')

        if mt in ('MT700', 'MT701', 'MT705', 'MT710', 'MT711',
                  'MT720', 'MT721', 'MT760'):
            mt700_packets.append(pkt)

        elif mt in ('MT707', 'MT708', 'MT767', 'MT775'):
            mt707_packets.append(pkt)

        elif mt in ('MT799', 'MT999'):
            mt799_packets.append(pkt)

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

        final_lc.swift_format = _detect_format_from_text(base_text)
        _progress(f"  Base LC format: {final_lc.swift_format}")

        base_fields = _extract_swift_fields(base_text, source_page=base_page, source_mt='MT700')
        _progress(f"  Extracted {len(base_fields)} fields from MT700")

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

        _LABEL_STRIP = {
            '20': r'^(?:Documentary\s+Credit\s+Number|Sender\'?s?\s+Reference|Transaction\s+Reference)\s*[\n\r]*',

            '22A': r'^Purpose\s+of\s+Message\s*[\n\r]*',

            '23': r'^(?:(?:[A-Z]\w*\'?s?\s+){0,3})?Reference(?:\s+to\s+Pre-?Advice)?\s*[\n\r]*',
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
            '42D': r'^Drawee(?:\s*-?\s*Party\s+Identifier)?(?:\s*-?\s*Name\s+and\s+Address)?\s*[\n\r]*',
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

            '58A': r'^Requested\s+Confirmation\s+Party.*?(?:Identifier\s+Code)?\s*[\n\r]*',
            '59': r'^Beneficiary\s*[\n\r]*(?:Name\s+and\s+Address:?\s*[\n\r]*)?',

            '71D': r'^(?:(?:OUR|BEN|SHA|Other)\s+)?Charges\s*[\n\r]*',
            '78': r'^Instructions\s+to\s+the\s+Paying.*?Bank\s*[\n\r]*',
        }
        for sf in base_fields:

            sf.value = re.sub(r'```(?:plaintext|text|swift|json)?\s*', '', sf.value, flags=re.IGNORECASE)
            sf.value = re.sub(r'\s*```\s*$', '', sf.value).strip()

            _strip_pat = _LABEL_STRIP.get(sf.tag, '')
            if _strip_pat:
                sf.value = re.sub(_strip_pat, '', sf.value, flags=re.IGNORECASE).strip()

            sf.value = re.sub(r'-?\s*Party\s+Identifier\s*-?\s*Identifier\s*(?:Code)?\s*\n?', '', sf.value, flags=re.IGNORECASE).strip()
            sf.value = re.sub(r'^-\s+', '', sf.value).strip()
            sf.value = re.sub(r'Identifier\s+Code:?\s*\n?', '', sf.value, flags=re.IGNORECASE).strip()
            sf.value = re.sub(r'Identifier:?\s*\n?', '', sf.value, flags=re.IGNORECASE).strip()
            sf.value = re.sub(r'^Name\s+and\s+Address:?\s*\n?', '', sf.value, flags=re.IGNORECASE).strip()

            sf.value = re.sub(r'\.{2,}\s*By\s*\.{2,}\s*-?\s*(?:Name\s+and\s+Address\s*-?\s*)*:?\s*[\n\r]*', '', sf.value, flags=re.IGNORECASE).strip()
            sf.value = re.sub(r'-\s*Name\s+and\s+Address\s*-?\s*(?:Name\s+and\s+Address)?:?\s*[\n\r]*', '', sf.value, flags=re.IGNORECASE).strip()

            if sf.tag == '32B':
                sf.value = re.sub(r'\b(?:US\s+DOLLAR|EURO|POUND\s+STERLING|JAPANESE\s+YEN)\s*[\n\r]*', '', sf.value, flags=re.IGNORECASE).strip()
            sf.value = re.sub(r'^Applicable\s+Rules:?\s*\n?', '', sf.value, flags=re.IGNORECASE).strip()

            sf.value = re.sub(r'^Available\s+With.*?Code\s*\n?', '', sf.value, flags=re.IGNORECASE).strip()
            sf.value = re.sub(r'^Drawee\s*-?\s*Party.*?Code\s*\n?', '', sf.value, flags=re.IGNORECASE).strip()
            sf.value = re.sub(r'^Applicant\s+Bank\s*-?\s*Party.*?Code\s*\n?', '', sf.value, flags=re.IGNORECASE).strip()

            sf.value = re.sub(
                r'\n\s*Other\s*\n\s*(?:Delivery\s+overdue|Network\s+delivery|Payment\s+Confirmation).*$',
                '', sf.value, flags=re.IGNORECASE | re.DOTALL).strip()

            sf.value = re.sub(r'\(CONT\.?\s*(?:FROM|IN)\s+FIELD\s+\w+\)', '', sf.value, flags=re.IGNORECASE).strip()
            sf.value = re.sub(r'/\(CONT\.?\s*(?:FROM|IN)\s+FIELD\s+\w+\)', '', sf.value, flags=re.IGNORECASE).strip()

            _ftag_merge = re.search(r'F\d{2}[A-Z]?\s*:', sf.value)
            if _ftag_merge:
                sf.value = sf.value[:_ftag_merge.start()].strip()

            sf.value = re.sub(r'\bPage\s+\d+\s+of\s+\d+\b', '', sf.value, flags=re.IGNORECASE).strip()

            sf.value = re.sub(
                r'There is no visible text.*?(?:clearly visible|another version)[.\s]*',
                '', sf.value, flags=re.IGNORECASE | re.DOTALL).strip()
            sf.value = re.sub(
                r'The image appears to be blank[.\s]*',
                '', sf.value, flags=re.IGNORECASE).strip()

            if sf.tag == '27':
                continue

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

            _dm = re.search(r'(\d{6})\s+(\d{4})\s+(\w{3})\s+(\d{1,2})', sf.value)
            if _dm and sf.tag in ('31C', '31D', '44C'):
                _months = {'Jan':'01','Feb':'02','Mar':'03','Apr':'04','May':'05','Jun':'06',
                           'Jul':'07','Aug':'08','Sep':'09','Oct':'10','Nov':'11','Dec':'12'}
                _date_str = f"{_dm.group(2)}-{_months.get(_dm.group(3),'01')}-{int(_dm.group(4)):02d}"
                sf.value = sf.value[:_dm.start()] + _date_str + sf.value[_dm.end():]
                sf.value = sf.value.strip()

            if sf.tag == '32B':
                _ccy = re.search(r'\b([A-Z]{3})\b', sf.value)
                _ccy_str = _ccy.group(1) if _ccy else 'USD'

                _am = re.search(r'#([\d,]+\.\d+)#?', sf.value)
                if _am:
                    sf.value = f"{_ccy_str} {_am.group(1)}"
                else:

                    _am2 = re.search(r'(\d[\d.]*,\d{2})\b', sf.value)
                    if _am2:
                        _amt = _am2.group(1).replace('.', '').replace(',', '.')
                        sf.value = f"{_ccy_str} {float(_amt):,.2f}"

            if sf.tag in ('31C', '31D', '44C'):

                if len(sf.value) >= 7 and sf.value[:6].isdigit() and sf.value[6].isalpha():
                    sf.value = sf.value[:6] + ' ' + sf.value[6:]
                _raw_dm = re.search(r'\b(\d{2})(\d{2})(\d{2})\b', sf.value)
                if _raw_dm and not re.search(r'\d{4}-\d{2}-\d{2}', sf.value):
                    _yy, _mm, _dd = _raw_dm.group(1), _raw_dm.group(2), _raw_dm.group(3)
                    _year = f"20{_yy}" if int(_yy) < 80 else f"19{_yy}"
                    sf.value = sf.value[:_raw_dm.start()] + f"{_year}-{_mm}-{_dd}" + sf.value[_raw_dm.end():]
                    sf.value = sf.value.strip()

            sf.value = re.sub(r'^[\.,;:]\s*[\n\r]+', '', sf.value).strip()
            sf.value = re.sub(r'^[\.,;:]+\s*(?=[A-Za-z0-9])', '', sf.value).strip()

            final_lc.consolidated_fields[sf.tag] = sf.value
            final_lc.original_fields[sf.tag] = sf.value
            _progress(f"    F{sf.tag}: {sf.value[:80]}{'...' if len(sf.value) > 80 else ''}")

        _raw_dc = final_lc.consolidated_fields.get('20', '')
        _raw_dc = re.sub(r"(?i)^(?:Sender'?s?\s+Reference|Documentary\s+Credit\s+Number|Transaction\s+Reference\s+Number)\s*[\n\r]*", '', _raw_dc).strip()
        final_lc.dc_number = _raw_dc
        final_lc.consolidated_fields['20'] = _raw_dc

        _raw_f21 = final_lc.consolidated_fields.get('21', '')
        if _raw_f21:
            _raw_f21 = re.sub(r"(?i)^(?:Related\s+Reference|Receiver'?s?\s+Reference)\s*[\n\r]*", '', _raw_f21).strip()

            _f20_has_lc = bool(re.search(r'(?<![A-Z])(?:LC|ILC|ALS|DLC)\d', final_lc.dc_number, re.IGNORECASE))
            _f21_has_lc = bool(re.search(r'(?<![A-Z])(?:LC|ILC|ALS|DLC)\d', _raw_f21, re.IGNORECASE))
            if _raw_f21 and not _f20_has_lc and _f21_has_lc:
                _progress(f"  DC number: using Related Reference (F21) '{_raw_f21}' over Transaction Reference (F20) '{final_lc.dc_number}'")
                final_lc.dc_number = _raw_f21
                final_lc.consolidated_fields['20'] = _raw_f21
        final_lc.source_packets.append(_get_packet_field(base_pkt, 'packet_id', 0))

        _SHORT_ENUM_VALUES = {
            'ALLOWED', 'PROHIBITED', 'NOT ALLOWED', 'PERMITTED',
            'WITHOUT', 'CONFIRM', 'MAY ADD', 'IRREVOCABLE',
        }
        for extra_pkt in mt700_packets[1:]:
            extra_text = _get_packet_refined_text(extra_pkt)
            extra_page = _get_packet_first_page(extra_pkt)

            _first_line = extra_text.strip().split('\n')[0].strip().upper() if extra_text else ''
            if _first_line in _SHORT_ENUM_VALUES:

                _label_only_fields = {'43T', '43P', '49', '40A'}
                for _lof_tag in _label_only_fields:
                    _existing_val = final_lc.consolidated_fields.get(_lof_tag, '')
                    _cleaned = re.sub(r'(?i)^(?:Trans[sh]?ipment|Partial\s+Shipments?|Confirmation\s+Instructions|Form\s+of\s+Documentary\s+Credit)\s*$', '', _existing_val).strip()
                    if not _cleaned and _existing_val:
                        final_lc.consolidated_fields[_lof_tag] = _first_line
                        final_lc.original_fields[_lof_tag] = _first_line
                        _progress(f"    F{_lof_tag}: page-break continuation → '{_first_line}'")
                        break

            _first_ftag_in_extra = re.search(
                r'(?:^|\n)\s*F(\d{2}[A-Z]?)\s*:', extra_text
            )
            if _first_ftag_in_extra:
                _pre_tag = extra_text[:_first_ftag_in_extra.start()].strip()

                _pre_tag = re.sub(
                    r'(?:^|\n)\s*Page\s+\d+\s+of\s+\d+\s*', '\n',
                    _pre_tag, flags=re.IGNORECASE,
                ).strip()
                _pre_tag = re.sub(
                    r'^\s*ORIGINAL\s+COPY\s+NON-NEGOTIABLE\s*', '',
                    _pre_tag, flags=re.IGNORECASE,
                ).strip()

                _looks_like_metadata = bool(re.search(
                    r'(?:Message\s+Details|Message\s+Identifier|Message\s+Preparation|'
                    r'Block\s+\d|Identifier\s*:\s*fin\.|Applic\.?\s+Interface|'
                    r'Sender\s*:|Receiver\s*:|Transaction\s+Reference)',
                    _pre_tag, re.IGNORECASE,
                ))
                if _pre_tag and len(_pre_tag) >= 30 and not _looks_like_metadata:
                    _new_tag_in_extra = _first_ftag_in_extra.group(1).upper()
                    _CLAUSE_BEARING_TAGS = (
                        '45A', '45B', '46A', '46B', '47A', '47B',
                        '78', '72', '79',
                    )
                    try:
                        _new_idx = EXTRACTION_TAGS.index(_new_tag_in_extra)
                    except ValueError:
                        _new_idx = -1
                    _target_tag = None
                    if _new_idx >= 0:
                        for _candidate in reversed(EXTRACTION_TAGS[:_new_idx]):
                            if (_candidate in _CLAUSE_BEARING_TAGS
                                    and _candidate in final_lc.consolidated_fields
                                    and final_lc.consolidated_fields.get(_candidate, '').strip()):
                                _target_tag = _candidate
                                break
                    if _target_tag:
                        _existing = final_lc.consolidated_fields[_target_tag]
                        if _pre_tag not in _existing:
                            final_lc.consolidated_fields[_target_tag] = (
                                _existing.rstrip() + '\n' + _pre_tag
                            )
                            final_lc.original_fields[_target_tag] = (
                                final_lc.consolidated_fields[_target_tag]
                            )
                            _progress(
                                f"    P212: F{_target_tag} continuation rescued from "
                                f"extra MT700 packet (+{len(_pre_tag)} chars before F{_new_tag_in_extra})"
                            )

            extra_fields = _extract_swift_fields(extra_text, source_page=extra_page, source_mt='MT700')
            _clause_tags = {'46A', '47A', '45A', '78', '72', '79'}
            for sf in extra_fields:

                if sf.tag == '27':
                    continue
                if sf.tag not in final_lc.consolidated_fields:
                    final_lc.consolidated_fields[sf.tag] = sf.value
                    final_lc.original_fields[sf.tag] = sf.value
                    _progress(f"    F{sf.tag} (from extra MT700): {sf.value[:60]}...")
                elif sf.tag in _clause_tags and sf.value:

                    existing = final_lc.consolidated_fields[sf.tag]
                    if sf.value not in existing:
                        final_lc.consolidated_fields[sf.tag] = existing.rstrip() + '\n' + sf.value
                        final_lc.original_fields[sf.tag] = final_lc.consolidated_fields[sf.tag]
                        _progress(f"    F{sf.tag} (appended from extra MT700 page): +{len(sf.value)} chars")
            final_lc.source_packets.append(_get_packet_field(extra_pkt, 'packet_id', 0))

        if '45A' in final_lc.consolidated_fields:
            _f45a_value = final_lc.consolidated_fields['45A']

            _candidate_pages = []
            for _pkt in mt700_packets:
                _candidate_pages.extend(
                    _pkt.get('page_numbers', []) if isinstance(_pkt, dict)
                    else (getattr(_pkt, 'page_numbers', []) or [])
                )
            _recovered = _recover_f45a_layout_from_step01(_f45a_value, _candidate_pages)
            if _recovered != _f45a_value:
                final_lc.consolidated_fields['45A'] = _recovered
                final_lc.original_fields['45A'] = _recovered
                _progress(
                    f"  P217: F45A multi-line layout restored from step01 raw OCR "
                    f"(was inline {len(_f45a_value)} chars; now {_recovered.count(chr(10)) + 1} lines)"
                )

    if mt707_packets and final_lc.consolidated_fields:
        _pre_keys = list(final_lc.consolidated_fields.keys())
        _pre_snapshot = {k: final_lc.consolidated_fields.get(k, '') for k in _pre_keys}
        _apply_p240_cross_ref_rescue(
            final_lc.consolidated_fields, _progress, _phase='pre-amendment'
        )

        for _k, _v in final_lc.consolidated_fields.items():
            if _pre_snapshot.get(_k) != _v:
                final_lc.original_fields[_k] = _v

    if mt707_packets:
        sorted_amendments = _sort_amendments(mt707_packets)
        _progress(f"  Applying {len(sorted_amendments)} amendments...")

        _MT707_STRONG_SIGNAL_RE = re.compile(
            r'(?:'
            r'(?:^|\n)\s*F?26E\s*:'
            r'|Number\s+of\s+Amendment'
            r'|(?:^|\n)\s*F?30\s*:'
            r'|Date\s+of\s+Amendment'
            r'|/REPALL/|/ADD/|/DELETE/|/DEL/'
            r'|Identifier\s*:\s*fin\.\s*7(?:0[78]|67|75)'
            r'|fin\.\s*7(?:0[78]|67|75)'
            r')',
            re.IGNORECASE,
        )

        for i, amd_pkt in enumerate(sorted_amendments):
            amd_text = _get_packet_refined_text(amd_pkt)
            amd_page = _get_packet_first_page(amd_pkt)
            pkt_id = _get_packet_field(amd_pkt, 'packet_id', 0)
            is_799 = bool(_get_packet_field(amd_pkt, 'is_799_amendment', False))
            src_mt_label = _get_packet_field(amd_pkt, 'source_mt', '') or 'MT707'

            if not is_799 and not _MT707_STRONG_SIGNAL_RE.search(amd_text):
                _progress(
                    f"    Amendment {i + 1}: P216 rejected packet {pkt_id} "
                    f"(no F26E / F30 / op-code / fin.707 signal — likely a "
                    f"step03 misclassification of a shipping/transmittal "
                    f"document; base MT700 preserved)"
                )
                warnings.append(
                    f"Packet {pkt_id} labelled as MT707 amendment but lacks "
                    f"any genuine MT707 signal (F26E / F30 / op-code / fin.707); "
                    f"skipped to avoid corrupting the base LC fields."
                )
                continue

            if is_799:

                amd_fields = _extract_mt799_amendment_fields(
                    amd_text, source_page=amd_page, source_mt=src_mt_label or 'MT799',
                )

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

                    for _sf in amd_fields:
                        if not _sf.mt799_replace_anchor:
                            continue
                        _cur = final_lc.consolidated_fields.get(_sf.tag, '') or ''
                        if not _cur:
                            continue
                        _stripped = _strip_mt799_anchor_block(
                            _cur, _sf.mt799_replace_anchor,
                        )
                        if _stripped != _cur:
                            final_lc.consolidated_fields[_sf.tag] = _stripped
                            _progress(
                                f"      P273: stripped existing block for "
                                f"{_sf.mt799_replace_anchor!r} from F{_sf.tag} "
                                f"before MT799 /ADD/ append"
                            )
            else:
                amd_fields = _extract_swift_fields(amd_text, source_page=amd_page, source_mt='MT707')

                if len(amd_fields) < 2:
                    try:
                        vlm_amd = _extract_fields_vlm(amd_pkt, amd_text, amd_page, _progress)
                        if len(vlm_amd) > len(amd_fields):
                            amd_fields = vlm_amd
                    except Exception:
                        pass
                _progress(f"    Amendment {i + 1}: {len(amd_fields)} fields from packet {pkt_id}")

            _NESTED_FIELD_INSTR_RE = re.compile(
                r'(?:'
                r'FIELD\s+\d{2}[A-Z]?-?\d*\s+'
                r'(?:(?:NOW\s+)?TO\s+READ\s+AS|WORD\s+TO\s+READ\s+AS'
                r'|DELETE\s+WORDING|ADD\s+(?:LOI\s+)?(?:CLAUSE|WORDING))'
                r'|UNDER\s+FIELD\s+\d{2}[A-Z]?(?:-\d+)?\s+ADD\s+'
                r'(?:LOI\s+)?(?:CLAUSE|WORDING)\s+AS'
                r')',
                re.IGNORECASE,
            )
            _has_structured_clause_ops = any(
                sf.tag in ('45A', '45B', '46A', '46B', '47A', '47B')
                and (
                    any(opcode in (sf.value or '').upper()
                        for opcode in ('/DELETE/', '/DEL/', '/ADD/'))

                    or '/REPALL/' in (sf.value or '').upper()
                )
                for sf in amd_fields
            )

            record = None
            if _has_structured_clause_ops:
                _det_snapshot = dict(final_lc.consolidated_fields)
                det_record = _apply_amendment(
                    final_lc.consolidated_fields,
                    amd_fields,
                    amendment_number=i + 1,
                    source_packet_id=pkt_id,
                )
                if det_record and det_record.fields_changed:
                    record = det_record
                    _progress(
                        f"      Applied via deterministic rules (structured MT707): "
                        f"{record.fields_changed}"
                    )

                    if (
                        'ADD CLAUSE AS' in amd_text.upper()
                        or 'ADD LOI CLAUSE AS' in amd_text.upper()
                    ):
                        _p263_clean_amd = amd_text
                        _p263_clean_amd = re.sub(
                            r'(?:^|\n)\s*Narrative\s*:\s*', '\n',
                            _p263_clean_amd,
                        ).strip()
                        _p263_clean_amd = re.sub(
                            r'(?:^|\n)\s*Lines?\s+\d+(?:\s*[-–]\s*\d+)?\s*(?:\n|$)',
                            '\n', _p263_clean_amd,
                        ).strip()
                        _p263_clean_amd = re.sub(
                            r'(?:^|\n)\s*Code\s*:\s*', '\n',
                            _p263_clean_amd,
                        ).strip()
                        _p263_clean_amd = re.sub(
                            r'\s*Other\s*\n?\s*Delivery\s+overdue.*$',
                            '', _p263_clean_amd,
                            flags=re.IGNORECASE | re.DOTALL,
                        ).strip()
                        _p263_clean_amd = re.sub(
                            r'\s*Page\s+\d+\s+of\s+\d+\s*', ' ',
                            _p263_clean_amd,
                        ).strip()
                        _p263_clean_amd = re.sub(
                            r'\{CHK:[A-F0-9]+\}', '', _p263_clean_amd,
                        ).strip()
                        _p263_clean_amd = re.sub(
                            r'Block\s+5\s*', '', _p263_clean_amd,
                        ).strip()
                        _p263_clean_amd = re.sub(
                            r'Report\s+(?:Header|Footer|Content).*?(?=\n[A-Z]|\Z)',
                            '', _p263_clean_amd,
                            flags=re.IGNORECASE | re.DOTALL,
                        ).strip()
                        _p263_clean_amd = re.sub(
                            r'Message\s+(?:Header|Identifier|Details).*?(?=\n[A-Z]|\Z)',
                            '', _p263_clean_amd,
                            flags=re.IGNORECASE | re.DOTALL,
                        ).strip()
                        _p263_clean_amd = re.sub(
                            r'(?:Delivery\s+overdue|Network\s+delivery'
                            r'|Payment\s+Confirmation'
                            r'|Confirmed\s+(?:Currency|Amount|Date)).*$',
                            '', _p263_clean_amd,
                            flags=re.IGNORECASE | re.DOTALL,
                        ).strip()
                        try:
                            _p251_addclause_augment(
                                final_lc.consolidated_fields,
                                _p263_clean_amd,
                                record,
                                _progress,
                            )
                        except Exception as _e_p263:
                            _progress(
                                f"      P263 ADD-CLAUSE augment failed: "
                                f"{_e_p263} (deterministic result kept)"
                            )
                else:

                    final_lc.consolidated_fields = _det_snapshot

            if record is not None:

                _read_as_targets: List[tuple] = []
                for _m in re.finditer(
                    r'(?:^|\n)\s*(\d{2}[A-Z]?)\s*\(\s*(\d{1,3})\s*\)\s*:?\s*-?\s*'
                    r'Read\s+as\b',
                    amd_text, re.IGNORECASE,
                ):
                    _raw_tag = _m.group(1).upper()
                    _read_as_targets.append((
                        _MT799_TAG_TO_FIELD.get(_raw_tag, _raw_tag),
                        int(_m.group(2)),
                    ))
                if _read_as_targets:

                    _cleaned_amd = re.sub(
                        r'(?:^|\n)\s*\d{2}[A-Z]?\s*'
                        r'(?:\(\s*\d{1,3}\s*\)|\s+\d{1,3})\s*:?\s*-?\s*'
                        r'PLEASE\s+DELETE\s+COMPLETELY[^\n]*',
                        '', amd_text, flags=re.IGNORECASE,
                    )
                    _cleaned_amd = re.sub(
                        r'(?:^|\n)\s*\d{2}[A-Z]?\s*:-\s*[^\n]{0,80}?'
                        r'\bRead\s+as\b[^\n]*(?:\n[^\n]*?(?:instead\s+of\s+existing'
                        r'|i\s*/\s*[oe])[^\n]*)?',
                        '', _cleaned_amd, flags=re.IGNORECASE,
                    )
                    try:
                        aug_record = _apply_amendment_vlm(
                            final_lc.consolidated_fields,
                            _cleaned_amd,
                            amendment_number=i + 1,
                            source_packet_id=pkt_id,
                            _progress=_progress,
                        )
                    except Exception as _e:
                        aug_record = None
                        _progress(f"      P245 VLM augment failed: {_e}")
                    if aug_record and aug_record.fields_changed:

                        for _tag in aug_record.fields_changed:
                            if _tag not in record.fields_changed:
                                record.fields_changed.append(_tag)
                            _aug_det = aug_record.change_details.get(_tag, {})
                            if _tag in record.change_details:
                                _existing = record.change_details[_tag]
                                if 'new' in _aug_det:
                                    _existing['new'] = _aug_det['new']
                            else:
                                record.change_details[_tag] = _aug_det

                        for _tag, _cn in _read_as_targets:
                            if _tag not in record.change_details:
                                continue
                            _det_for_tag = record.change_details[_tag]
                            _existing_changed = _det_for_tag.get('clauses_changed') or []
                            if _cn not in _existing_changed:
                                _existing_changed.append(_cn)
                            _det_for_tag['clauses_changed'] = _existing_changed
                        _progress(
                            f"      P245 VLM augment patched clause-level Read-as: "
                            f"{aug_record.fields_changed} (clauses {_read_as_targets})"
                        )

            if record is None:

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

                    _vlm_handled = set(record.fields_changed)
                    _OP_MARKER_RE = re.compile(
                        r'/(?:ADD|DEL|DELETE|REPALL)/|TO\s+READ\s+AS|'
                        r'PLEASE\s+READ|UNDER\s+FIELD',
                        re.IGNORECASE,
                    )

                    _aug_fields_input = []
                    for _sf in amd_fields:

                        _candidate = _sf.tag
                        if (_sf.tag.endswith('B')
                                and _sf.tag[:-1] + 'A' in SWIFT_FIELD_LABELS):
                            _base_candidate = _sf.tag[:-1] + 'A'
                            if (_base_candidate in final_lc.consolidated_fields
                                    or _sf.tag in ('45B', '46B', '47B')):
                                _candidate = _base_candidate
                        if _candidate in _vlm_handled:
                            continue

                        if _OP_MARKER_RE.search(_sf.value or ''):
                            continue
                        _aug_fields_input.append(_sf)

                    if _aug_fields_input:
                        _aug_record = _apply_amendment(
                            final_lc.consolidated_fields,
                            _aug_fields_input,
                            amendment_number=i + 1,
                            source_packet_id=pkt_id,
                        )

                        _augmented_tags = []
                        for _aug_tag in (_aug_record.fields_changed or []):
                            if _aug_tag in _vlm_handled:
                                continue
                            record.fields_changed.append(_aug_tag)
                            _aug_detail = _aug_record.change_details.get(
                                _aug_tag, {}) or {}
                            _aug_detail['operation'] = 'regex_augment'
                            record.change_details[_aug_tag] = _aug_detail
                            _augmented_tags.append(_aug_tag)
                        if _augmented_tags:
                            _progress(
                                f"      P225 regex augment: added "
                                f"{_augmented_tags} (VLM had "
                                f"{list(_vlm_handled)})"
                            )
                else:

                    record = _apply_amendment(
                        final_lc.consolidated_fields,
                        amd_fields,
                        amendment_number=i + 1,
                        source_packet_id=pkt_id,
                    )
                    _progress(f"      Applied via deterministic rules: {record.fields_changed}")
            final_lc.amendment_log.append(record)
            final_lc.source_packets.append(pkt_id)

            for _ch_tag in record.fields_changed:
                _raw = final_lc.consolidated_fields.get(_ch_tag, '')
                _cleaned = _clean_consolidated_field_value(_ch_tag, _raw)
                if _cleaned != _raw:
                    final_lc.consolidated_fields[_ch_tag] = _cleaned

                    if _ch_tag in record.change_details:
                        record.change_details[_ch_tag]['new'] = _cleaned

            _phantom = []
            for _ch_tag in list(record.fields_changed):
                _det = record.change_details.get(_ch_tag, {}) or {}
                _old_raw = _det.get('old', '')
                _new_raw = _det.get('new', '')
                if not _old_raw or not _new_raw:
                    continue
                _old_clean = _clean_consolidated_field_value(_ch_tag, _old_raw)
                _new_clean = _clean_consolidated_field_value(_ch_tag, _new_raw)
                if (_old_clean.strip()
                        and _old_clean.strip() == _new_clean.strip()):
                    _phantom.append(_ch_tag)
            for _ph in _phantom:
                while _ph in record.fields_changed:
                    record.fields_changed.remove(_ph)
                record.change_details.pop(_ph, None)
            if _phantom:
                _progress(f"      P227: dropped phantom-change tags: {_phantom}")

            if record.fields_changed:
                _progress(f"      Changed: {', '.join(record.fields_changed)}")
            if record.amendment_date:
                _progress(f"      Date: {record.amendment_date}")

        final_lc.amendment_count = len(final_lc.amendment_log)

        if '20' in final_lc.consolidated_fields:
            final_lc.dc_number = final_lc.consolidated_fields['20']

    _cf = final_lc.consolidated_fields

    _amended_fields = set()
    for rec in final_lc.amendment_log:
        _amended_fields.update(rec.fields_changed)

    _ever_had_old: Dict[str, bool] = {}
    for rec in final_lc.amendment_log:
        for _t, _det in (rec.change_details or {}).items():
            if isinstance(_det, dict) and (_det.get('old') or '').strip():
                _ever_had_old[_t] = True
    _added_only = {
        _t for _t in list(_amended_fields)
        if not _ever_had_old.get(_t, False)
    }
    if _added_only:
        _amended_fields -= _added_only
        _progress(
            f"      P274: suppressed AMENDED highlight for empty->non-empty "
            f"add-only tags: {sorted(_added_only)}"
        )
    if _amended_fields:
        _cf['_amended_fields'] = list(_amended_fields)

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

    _cont_marker_re = re.compile(
        r'[/\\]?\s*'
        r'\(\s*CONT\.?\s*(?:INUED|INUATION)?\s+(?:FROM|FORM|IN)\s+FIELD\s+(?P<src>\d{2}[A-Z]?)\s*\)\s*'
        r'(?P<rest>(?:'
        r'(?!\n\s*\d+\s*[.\-\)]\s)'
        r'(?!\n\s*F?\d{2}[A-Z]?\s*:)'
        r'(?!\s*[/\\]?\s*\(\s*CONT\.?\s*(?:INUED|INUATION)?\s+(?:FROM|FORM|IN)\s+FIELD)'
        r'.)*)',
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

        _matches = list(_cont_marker_re.finditer(_val))
        for _m in reversed(_matches):
            _src_tag = _m.group('src').upper()
            _chunk = (_m.group('rest') or '').strip()

            if not _chunk:
                _new_val = (_new_val[:_m.start()] + _new_val[_m.end():])
                _had_match = True
                continue

            if _src_tag == _tag.upper():
                continue
            _cont_pulled.setdefault(_src_tag, []).append(_chunk)
            _new_val = (_new_val[:_m.start()] + _new_val[_m.end():])
            _had_match = True
        if _had_match:
            _cf[_tag] = re.sub(r'\n{3,}', '\n\n', _new_val).strip()

    for _src_tag, _chunks in _cont_pulled.items():
        _existing = _cf.get(_src_tag, '') or ''

        _existing = re.sub(
            r'(?:^|\n)\s*[/\\]?\s*\(\s*CONT\.?\s*(?:INUED|INUATION)?\s+(?:FROM|FORM|IN)\s+FIELD\s+\d{2}[A-Z]?\s*\)\s*',
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

    _apply_p240_cross_ref_rescue(_cf, _progress, _phase='post-amendment')

    _f48_pre = _cf.get('48', '')
    if _f48_pre and isinstance(_f48_pre, str):

        _pre_days_m = re.search(
            r'(?<!\d)(\d{1,3})\s*[/\\\n\r]?\s*'
            r'(?:Narrative\s*:?\s*)?[/\\]?\s*'
            r'(?:PLS\s+)?REFER\b',
            _f48_pre, re.IGNORECASE,
        )
        if _pre_days_m:
            _cf['48'] = _pre_days_m.group(1)
            _progress(
                f"  P211: F48 pre-normalised to day count '{_pre_days_m.group(1)}' "
                f"(stripped trailing 'REFER FIELD ...' to prevent cross-ref overwrite)"
            )

    for _tag, _val in list(_cf.items()):
        if not isinstance(_val, str):
            continue

        _ref_m = re.search(r'\+{3,}SEE\s+FIELD\s+(\d{2}[A-Z]?)\+{3,}', _val, re.IGNORECASE)
        if _ref_m:
            _ref_tag = _ref_m.group(1)
            _ref_val = _cf.get(_ref_tag, '')
            if _ref_val:

                _marker_pat = r'\+{3,}FIELD\s+' + re.escape(_tag) + r'\+{3,}\s*\n?(.*?)(?=\n\+{3,}FIELD|\Z)'
                _marker_m = re.search(_marker_pat, _ref_val, re.IGNORECASE | re.DOTALL)
                if _marker_m:
                    _resolved = _marker_m.group(1).strip()
                    _cf[_tag] = _resolved
                    _progress(f"  F{_tag}: resolved cross-ref (marker) from F{_ref_tag} → {_resolved[:60]}")
                    continue

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

                        if _tag in ('50', '59'):
                            _resolved = _trim_resolved_for_name_address_field(_resolved)
                        _cf[_tag] = _resolved
                        _progress(f"  F{_tag}: resolved clause #{_clause_num} from F{_ref_tag} → {_resolved[:60]}")
                        break
            continue

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

                        if _tag in ('50', '59'):
                            _resolved = _trim_resolved_for_name_address_field(_resolved)
                        _cf[_tag] = _resolved
                        _progress(f"  F{_tag}: resolved ref from F{_ref_tag} clause ({_clause_num}) → {_resolved[:60]}")
                        break
                else:

                    _progress(f"  F{_tag}: clause ({_clause_num}) not found in F{_ref_tag} ({len(_ref_clauses)} clauses)")
            continue

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

        if _tag in ('50', '59'):
            _simple_ref_pattern = (
                r'(?:SEE|REFER(?:\s+TO)?|AS\s+PER)\s+'
                r'(?:FIELD\s+)?(\d{2}[A-Z]?)\b'
            )
        else:
            _simple_ref_pattern = (
                r'(?:SEE|REFER(?:\s+TO)?|AS\s+PER)\s+FIELD\s+(\d{2}[A-Z]?)'
            )
        _simple_ref_m = re.search(
            _simple_ref_pattern, _val, re.IGNORECASE)
        if _simple_ref_m:

            _val_after = (
                _val[:_simple_ref_m.start()]
                + _val[_simple_ref_m.end():]
            )

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

            _has_clause_ref = bool(re.search(
                r'CLAUSE\s+(?:NO\.?\s*)?\d+', _val, re.IGNORECASE))
            if (not _is_multi_clause) and len(_val_residual) <= 30\
                    and not _has_clause_ref:
                _ref_tag = _simple_ref_m.group(1)
                _ref_val = _cf.get(_ref_tag, '')
                if _ref_val:

                    _cf[_tag] = _ref_val
                    _progress(
                        f"  F{_tag}: resolved simple ref from F{_ref_tag} "
                        f"→ {_ref_val[:60]}"
                    )
            else:

                _resolved_via_label = False
                if _tag in ('50', '59'):
                    _ref_tag = _simple_ref_m.group(1)
                    _ref_val = _cf.get(_ref_tag, '')
                    if _ref_val:
                        _ref_clauses = _split_into_clauses(_ref_tag, _ref_val)
                        _party_kw = 'APPLICANT' if _tag == '50' else 'BENEFICIARY'

                        _matched = _find_name_address_clause(
                            _ref_clauses, _party_kw, _tag)
                        if _matched:
                            _resolved = _trim_resolved_for_name_address_field(
                                _matched.text.strip())
                            if _resolved:
                                _cf[_tag] = _resolved
                                _resolved_via_label = True
                                _progress(
                                    f"  F{_tag}: P277 resolved label-matched "
                                    f"clause #{_matched.clause_number} from "
                                    f"F{_ref_tag} → {_resolved[:60]}"
                                )
                if not _resolved_via_label:

                    _progress(
                        f"  F{_tag}: skipped simple-ref replacement "
                        f"(field has multi-clause content; reference "
                        f"is in-clause text only)"
                    )
            continue

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

                        if _tag in ('50', '59'):
                            _resolved = _trim_resolved_for_name_address_field(_resolved)
                        _cf[_tag] = _resolved
                        _progress(f"  F{_tag}: resolved ref from F{_ref_tag} clause #{_clause_num} → {_resolved[:60]}")
                        break

    if _cf.get('21', '').strip().upper() in ('NONREF', 'NON-REF', 'NONE', 'N/A', ''):
        _cf.pop('21', None)

    _f23 = _cf.get('23', '')
    if _f23 and (re.search(r'Message\s+Text|Block\s+\d|Report\s+Content', _f23, re.IGNORECASE)
                 or len(_f23) > 100):
        _cf.pop('23', None)

    _f23_clean = _cf.get('23', '').strip()
    _f20_clean = _cf.get('20', '').strip()
    if _f23_clean and _f20_clean and _f23_clean == _f20_clean:
        _cf.pop('23', None)
        _progress(f"  P223: F23 dropped (redundant — equals F20 LC number)")

    _f30 = _cf.get('30', '')
    if _f30 and re.search(r'Report\s+Content|Message\s+Details|Applic', _f30, re.IGNORECASE):
        _cf.pop('30', None)

    _cf.pop('30', None)

    _cf.pop('26E', None)

    for _b_tag in ('45B', '46B', '47B'):
        _a_tag = _b_tag[:-1] + 'A'
        if _b_tag in _cf and _cf.get(_a_tag):
            _cf.pop(_b_tag, None)

    _f48 = _cf.get('48', '')
    if _f48:
        _days_m = re.match(r'(\d+)\s*/?\s*(?:PLS\s+)?REFER', _f48, re.IGNORECASE)
        if _days_m:
            _cf['48'] = _days_m.group(1)
            _progress(f"  F48: extracted {_days_m.group(1)} days from presentation period")

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

    for _tag in list(_cf.keys()):
        if _tag.startswith('_'):
            continue
        _val = _cf[_tag]
        if isinstance(_val, str):
            _cleaned = _clean_consolidated_field_value(_tag, _val)
            if _cleaned != _val:
                _cf[_tag] = _cleaned

    try:
        _f48_v = str(_cf.get('48', '') or '').strip()
        if _f48_v:
            _norm = _f48_v

            _slash_m = re.match(
                r'^\s*(\d{1,3})\s*/\s*(FROM|FRM)\s+(.+)$',
                _norm,
                flags=re.IGNORECASE,
            )
            if _slash_m:

                _norm = (
                    f"{_slash_m.group(1)} FROM "
                    f"{_slash_m.group(3).strip()}"
                )

            _norm = re.sub(r'\bFRM\b', 'FROM', _norm, flags=re.IGNORECASE)
            _norm = re.sub(r'\bWITH\s+IN\b', 'WITHIN', _norm, flags=re.IGNORECASE)
            if _norm != _f48_v:
                _cf['48'] = _norm
                _progress(
                    f"  P198df: F48 reformatted "
                    f"{_f48_v!r} -> {_norm!r}"
                )
    except Exception as _e:
        try:
            _progress(f"  P198df F48 reformat exception: {_e}")
        except Exception:
            pass

    for _tag in list(_cf.keys()):
        if _tag.startswith('_'):
            continue
        _val = _cf[_tag]
        if isinstance(_val, str):

            _val = re.sub(r'https?://\S+', '', _val)

            _val = re.sub(r"Select\s+'Print'\s+to\s+output.*", '', _val, flags=re.IGNORECASE)

            _val = re.sub(r'\bSWIFT_MT\d+/?\d*', '', _val)
            _val = re.sub(r'\n\s*\d+/\d+\s*$', '', _val)

            _val = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}:\d+\S*', '', _val)

            _val = re.sub(r'\n{3,}', '\n\n', _val)
            _cf[_tag] = _val.strip()

    _amt = _cf.get('32B', '')
    if 'increase' in _amt.lower():
        _amt = re.sub(r'(?i)^Increase\s+of\s+Documentary\s+Credit\s+Amount\s*[\n\r]*', '', _amt).strip()
        _cf['32B'] = _amt

    _amt = _cf.get('32B', '')
    if _amt and isinstance(_amt, str):
        _v = _amt

        _v = re.sub(r'^\s*F?32B\s*[:\-]\s*', '', _v, flags=re.IGNORECASE)

        _next = re.search(r'\bF?\d{2}[A-Z]?\s*:', _v)
        if _next and _next.start() > 0:
            _v = _v[:_next.start()]

        _v = re.sub(
            r'\b(?:US\s*DOLLAR|US\s*DOLLARS|DOLLAR|DOLLARS|EURO|EUROS|POUND\s*STERLING|'
            r'POUNDS|JAPANESE\s*YEN|YEN|FRANC|FRANCS|RUPEE|RUPEES|YUAN|RIYAL|DIRHAM)\b',
            '', _v, flags=re.IGNORECASE,
        )

        _v = _v.replace('#', ' ')

        _ccy_m = re.search(r'\b([A-Z]{3})\b', _v)
        _ccy_str = _ccy_m.group(1) if _ccy_m else 'USD'

        _amt_value = None

        _us = re.search(r'(\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?)', _v)
        if _us:
            try:
                _amt_value = float(_us.group(1).replace(',', ''))
            except ValueError:
                pass

        if _amt_value is None:
            _eu = re.search(r'(\d{1,3}(?:\.\d{3})+,\d{1,2})', _v)
            if _eu:
                try:
                    _amt_value = float(_eu.group(1).replace('.', '').replace(',', '.'))
                except ValueError:
                    pass

        if _amt_value is None:
            _eu2 = re.search(r'(\d+,\d{2})\b', _v)
            if _eu2:
                try:
                    _amt_value = float(_eu2.group(1).replace(',', '.'))
                except ValueError:
                    pass

        if _amt_value is None:
            _us2 = re.search(r'(\d+\.\d{2})\b', _v)
            if _us2:
                try:
                    _amt_value = float(_us2.group(1))
                except ValueError:
                    pass

        if _amt_value is None:
            _int = re.search(r'\b(\d{3,})\b', _v)
            if _int:
                try:
                    _amt_value = float(_int.group(1))
                except ValueError:
                    pass
        if _amt_value is not None and _amt_value > 0:
            _cf['32B'] = f"{_ccy_str} {_amt_value:,.2f}"

    for tag in CLAUSE_TAGS:
        value = final_lc.consolidated_fields.get(tag, '')
        if value:
            clause_list = _split_into_clauses(tag, value, _progress=_progress)
            if clause_list:
                final_lc.clauses[tag] = clause_list
                _progress(f"  F{tag} ({SWIFT_FIELD_LABELS.get(tag, '')}): {len(clause_list)} clauses")

    for pkt in mt799_packets:
        text = _get_packet_refined_text(pkt)
        pkt_id = _get_packet_field(pkt, 'packet_id', 0)

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
