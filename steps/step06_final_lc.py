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
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


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
    Build regex patterns for SWIFT field extraction in both formats.

    Alliance:  :20:0491ILC081972     (colon-wrapped tag)
    Fusion:    F20: 0491ILC081972    (F-prefix tag)

    Each pattern captures the value until the next tag or end of text.
    """
    patterns = []
    for tag in tags:
        # Alliance format: :TAG:VALUE
        # Lookahead stops at next :TAG: pattern or end of text
        patterns.append({
            'tag': tag,
            'format': 'alliance',
            'regex': re.compile(
                r'(?:^|\n)\s*:' + re.escape(tag) + r':\s*(.*?)(?=\n\s*:[A-Z0-9]{2,4}:|\Z)',
                re.DOTALL,
            ),
        })
        # Fusion format: FTAG: VALUE
        # Lookahead stops at next FTAG: pattern or end of text
        patterns.append({
            'tag': tag,
            'format': 'fusion',
            'regex': re.compile(
                r'(?:^|\n)\s*F' + re.escape(tag) + r'\s*:\s*(.*?)(?=\n\s*F[A-Z0-9]{2,4}\s*:|\Z)',
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


def _detect_format_from_text(text: str) -> str:
    """Detect SWIFT format from GLM text content."""
    fusion_count = len(re.findall(r'\bF\d{2}[A-Z]?\s*:', text))
    alliance_count = len(re.findall(r':\d{2}[A-Z]?:', text))
    if fusion_count > alliance_count:
        return 'fusion'
    elif alliance_count > 0:
        return 'alliance'
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

    # F45A (Description of Goods) should NOT be split — it's one continuous description
    if tag in ('45A', 'F45A'):
        return [Clause(clause_number=1, clause_id=f"{tag}-1", text=text, parent_tag=tag)]

    # Try numbered: "1.", "2.", "1)", "2)"
    numbered = re.split(r'\n\s*(\d+)\s*[.)]\s+', '\n' + text)
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


# == Amendment application ====================================================

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

        # New amount replaces old
        if tag == '34B':
            old_val = base_fields.get('32B', '')
            base_fields['32B'] = sf.value
            record.fields_changed.append('32B')
            record.change_details['32B'] = {'old': old_val, 'new': sf.value, 'via': '34B'}
            continue

        # B-suffix -> A-suffix replacement
        actual_tag = tag
        if tag.endswith('B') and tag[:-1] + 'A' in SWIFT_FIELD_LABELS:
            base_tag = tag[:-1] + 'A'
            if base_tag in base_fields or tag in ('45B', '46B', '47B'):
                actual_tag = base_tag

        old_val = base_fields.get(actual_tag, '')
        base_fields[actual_tag] = sf.value
        if old_val != sf.value:
            record.fields_changed.append(actual_tag)
            record.change_details[actual_tag] = {'old': old_val, 'new': sf.value}

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
        if mt == 'MT700':
            mt700_packets.append(pkt)
        elif mt == 'MT707':
            mt707_packets.append(pkt)
        elif mt == 'MT799':
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

        # Clean field values: strip SWIFT label text from values
        # GLM text has: "F20: Documentary Credit Number\n05251LC082463"
        # Regex captures: "Documentary Credit Number\n05251LC082463"
        # We need just: "05251LC082463"
        _LABEL_STRIP = {
            '20': r'^Documentary\s+Credit\s+Number\s*\n?',
            '27': r'^Sequence\s+of\s+Total\s*\n?',
            '31C': r'^Date\s+of\s+Issue\s*\n?',
            '31D': r'^Date\s+and\s+Place\s+of\s+Expiry\s*\n?',
            '32B': r'^Currency\s+Code,?\s*Amount\s*\n?',
            '39A': r'^Percentage\s+Credit\s+Amount\s+Tolerance\s*\n?',
            '40A': r'^Form\s+of\s+Documentary\s+Credit\s*\n?',
            '40E': r'^Applicable\s+Rules\s*\n?',
            '41D': r'^Available\s+With.*?Code\s*\n?',
            '42A': r'^Drawee.*?(?:Identifier\s+Code)?\s*\n?',
            '42C': r'^Drafts\s+at\s*\.{0,3}\s*\n?',
            '43P': r'^Partial\s+Shipments?\s*\n?',
            '43T': r'^Trans[sh]?ipment\s*\n?',
            '44A': r'^Place\s+of\s+Taking.*?\s*\n?',
            '44C': r'^Latest\s+Date\s+of\s+Shipment\s*\n?',
            '44E': r'^Port\s+of\s+Loading.*?Departure\s*\n?',
            '44F': r'^Port\s+of\s+Discharge.*?Destination\s*\n?',
            '45A': r'^Description\s+of\s+Goods.*?Services\s*\n?',
            '46A': r'^Documents\s+Required\s*\n?',
            '47A': r'^Additional\s+Conditions\s*\n?',
            '48': r'^Period\s+for\s+Presentation.*?Days\s*\n?',
            '49': r'^Confirmation\s+Instructions\s*\n?',
            '50': r'^Applicant\s*\n?',
            '51A': r'^Applicant\s+Bank.*?(?:Identifier\s+Code)?\s*\n?',
            '59': r'^Beneficiary\s*\n?(?:Name\s+and\s+Address:?\s*\n?)?',
            '71D': r'^Charges\s*\n?',
            '78': r'^Instructions\s+to\s+the\s+Paying.*?Bank\s*\n?',
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
            # Clean amount format: "516000,00 #516,000.00#" -> "USD 516,000.00"
            _am = re.search(r'#([\d,]+\.\d+)#', sf.value)
            if _am and sf.tag == '32B':
                # Extract currency code (3 uppercase letters)
                _ccy = re.search(r'\b([A-Z]{3})\b', sf.value)
                _ccy_str = _ccy.group(1) if _ccy else ''
                sf.value = f"{_ccy_str} {_am.group(1)}".strip()
            # Remove raw SWIFT date codes like "260131" if already converted
            if sf.tag in ('31C', '31D', '44C'):
                sf.value = re.sub(r'\b\d{6}\b\s*', '', sf.value).strip()

            final_lc.consolidated_fields[sf.tag] = sf.value
            final_lc.original_fields[sf.tag] = sf.value
            _progress(f"    F{sf.tag}: {sf.value[:80]}{'...' if len(sf.value) > 80 else ''}")

        # Set DC number
        final_lc.dc_number = final_lc.consolidated_fields.get('20', '')
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

            amd_fields = _extract_swift_fields(amd_text, source_page=amd_page, source_mt='MT707')
            _progress(f"    Amendment {i + 1}: {len(amd_fields)} fields from packet {pkt_id}")

            record = _apply_amendment(
                final_lc.consolidated_fields,
                amd_fields,
                amendment_number=i + 1,
                source_packet_id=pkt_id,
            )
            final_lc.amendment_log.append(record)
            final_lc.source_packets.append(pkt_id)

            if record.fields_changed:
                _progress(f"      Changed: {', '.join(record.fields_changed)}")
            if record.amendment_date:
                _progress(f"      Date: {record.amendment_date}")

        final_lc.amendment_count = len(final_lc.amendment_log)
        # Update DC number if changed by amendment
        if '20' in final_lc.consolidated_fields:
            final_lc.dc_number = final_lc.consolidated_fields['20']

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
