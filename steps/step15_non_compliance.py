"""
Step 15 — Handling Non-Compliance and Non-Checkable Clauses
=============================================================
Classifies ALL LC fields/clauses into three categories to ensure complete
report coverage.

PURPOSE:
    Not every LC field can or should be verified against shipping documents.
    This step classifies each field so the report accurately represents what
    was checked, what was shown for informational purposes, and what could not
    be checked from documents alone.

THREE CATEGORIES:
    1. checkable      — Document conditions verified in Steps 12-14 (F46A, F47A, F45A)
                        Also includes implicit checks (F31D, F32B, F44C, F48)
    2. informational  — LC metadata shown in report but not checked (F20 = LC Number,
                        F50 = Applicant, F59 = Beneficiary, F40A = Form of Credit, etc.)
    3. non_checkable  — Bank obligations, sanctions clauses, regulatory statements
                        that cannot be verified from documents alone

    This ensures every LC clause appears in the final report for completeness.
    Bank examiners expect to see ALL clauses accounted for, even if some are
    just noted as informational or non-checkable.

INPUTS:
    - Step 7 StructuredLC (all consolidated_fields)

OUTPUTS:
    - clause_status_map: {clause_ref: status} dictionary for Step 16 to use
    - clause_statuses: Full list of ClauseStatus objects with reasons
    - Summary counts: checkable, informational, non_checkable

TRADE FINANCE CONTEXT:
    - F46A/46B = Documents Required — always checkable
    - F47A/47B = Additional Conditions — checkable unless bank-internal
    - F20 = LC Number — informational (shown in report header)
    - F71D = Charges — informational (bank fee allocation, not document condition)
    - "SANCTIONS CLAUSE" = non-checkable (regulatory, not document-based)
    - "BANK TO BANK REIMBURSEMENT UNDER URR 725" = non-checkable (bank obligation)

AI MODEL: None — rule-based classification using regex patterns and field tag sets.
"""

import json
import sys as _sys; _sys.stdout.reconfigure(encoding="utf-8", errors="replace") if hasattr(_sys.stdout, "reconfigure") else None
import time
import re
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Set, Any
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Data Models ──────────────────────────────────────────────────────────

@dataclass
class ClauseStatus:
    """
    Classification of a single LC field/clause.

    Each LC field gets exactly one status classification. For clause fields
    (46A, 47A, etc.) that contain multiple clauses, each clause gets its own
    ClauseStatus with a numbered clause_ref (e.g., "47A-1", "47A-2").
    """
    clause_ref: str = ""             # e.g., "20", "46A-1", "47A-3"
    field_tag: str = ""              # e.g., "20", "46A"
    clause_number: int = 0           # 0 for standalone fields, 1+ for clause items
    status: str = "checkable"       # checkable | informational | non_checkable | skip
    reason: str = ""                 # Why this classification was assigned
    display_label: str = ""          # Human-readable label for report display
    value_preview: str = ""          # First 200 chars of value for report display


# ── Classification Rules ─────────────────────────────────────────────────

# Informational fields — LC metadata that provides context but has no compliance check.
# These are displayed in the report header or Key Terms section.
INFORMATIONAL_FIELDS: Dict[str, str] = {
    '20':   'LC Number',                     # Documentary Credit Number
    '21':   'Related Reference',             # Used for amendments
    '23':   'Reference to Pre-Advice',
    '25':   'Advice Through Bank',
    '27':   'Sequence of Total',             # e.g., "1/1" = page 1 of 1
    '30':   'Date of Amendment',
    '31C':  'Date of Issue',                 # When the LC was issued
    '31D':  'Date and Place of Expiry',      # Checked implicitly in Step 14
    '40A':  'Form of Documentary Credit',    # IRREVOCABLE, REVOCABLE
    '40E':  'Applicable Rules',              # UCP LATEST VERSION
    '41A':  'Available With/By',             # Which bank can negotiate
    '41D':  'Available With/By',
    '42A':  'Drawee',                        # Bank on which drafts are drawn
    '42C':  'Drafts At',                     # e.g., "AT SIGHT", "90 DAYS AFTER SIGHT"
    '42M':  'Mixed Payment Details',
    '42P':  'Deferred Payment Details',
    '43P':  'Partial Shipments',             # ALLOWED or NOT ALLOWED
    '43T':  'Transhipment',                  # ALLOWED or NOT ALLOWED
    '44A':  'Place of Taking in Charge',
    '44B':  'Place of Final Destination',
    '44E':  'Port of Loading',               # Where goods are shipped from
    '44F':  'Port of Discharge',             # Where goods arrive
    '49':   'Confirmation Instructions',
    '50':   'Applicant',                     # The buyer / importer
    '51A':  'Applicant Bank',
    '51D':  'Applicant Bank',
    '52A':  'Issuing Bank',                  # Bank that issued the LC
    '52D':  'Issuing Bank',
    '53A':  'Reimbursing Bank',
    '57A':  'Advising Bank',                 # Bank that advises the LC to beneficiary
    '57D':  'Advising Bank',
    '59':   'Beneficiary',                   # The seller / exporter
    '71D':  'Charges',                       # Who pays banking charges
}

# Fields that are implicitly checked (date/amount logic in Step 14) —
# these should still appear as "checkable" in the status map so Step 16
# does not accidentally mark them as non-checkable
IMPLICIT_CHECK_FIELDS: Set[str] = {'31D', '32B', '44C', '48'}

# Non-checkable clause keywords — regex patterns that identify clauses
# which cannot be verified from shipping documents alone
NON_CHECKABLE_PATTERNS = [
    # Bank obligations — instructions between banks, not document conditions
    r'(?:ADVISING|CONFIRMING|REIMBURSING|NEGOTIATING)\s+BANK\s+(?:IS|SHALL|MUST|WILL)\s+(?:AUTHORISED|AUTHORIZED|OBLIGATED)',
    r'BANK\s+TO\s+BANK\s+REIMBURSEMENT',
    r'REIMBURSEMENT\s+(?:INSTRUCTIONS|CLAIM)',
    r'THIS\s+CREDIT\s+IS\s+SUBJECT\s+TO',
    r'GOVERNED\s+BY\s+(?:UCP|ISBP|ISP|URR)',
    # Sanctions — regulatory compliance, not document-based
    r'SANCTION[S]?\s+(?:CLAUSE|SCREENING|COMPLIANCE)',
    r'OFAC|SDN\s+LIST|EU\s+SANCTION|UN\s+SANCTION',
    # Regulatory
    r'ANTI[- ]?MONEY\s+LAUNDERING',
    r'COMPLIANCE\s+WITH\s+(?:LOCAL|INTERNATIONAL)\s+(?:LAW|REGULATION)',
    r'FORCE\s+MAJEURE',
    # Informational/already handled
    r'THIS\s+(?:LC|CREDIT|DOCUMENTARY\s+CREDIT)\s+(?:NUMBER|NO\.?)\s+MUST\s+BE\s+QUOTED',
    r'ALL\s+BANKING\s+CHARGES\s+OUTSIDE',
    r'DOCUMENTS\s+MUST\s+BE\s+PRESENTED\s+WITHIN',  # handled implicitly in Step 14
    r'PERIOD\s+FOR\s+PRESENTATION',                    # handled implicitly in Step 14
    # P125 — Negotiation policy clauses (bank-to-bank, not document content)
    r'NEGOTIATION\s+UNDER\s+(?:RESERVE|GUARANTEE)',
    r'(?:UNDER\s+)?(?:RESERVE|GUARANTEE)\s+NOT\s+ALLOWED',
    # P125 — Charge-allocation certification (fee policy, not shipping compliance)
    r'NEGOTIATING\s+BANK\s+MUST\s+CERTIFY.{0,200}CHARGES.{0,50}PAID\s+BY\s+(?:THE\s+)?BENEFICIARY',
    r'CHARGES\s+(?:OF|AND)\s+.{0,100}(?:ADVISING|NEGOTIATING)\s+BANK.{0,100}PAID\s+BY\s+(?:THE\s+)?BENEFICIARY',
    # P125 — Courier / dispatch instructions (physical dispatch, not document content)
    r'DOCUMENTS?\s+MUST\s+BE\s+SENT\s+TO.{0,300}(?:DHL|FEDEX|UPS|TNT|ARAMEX|COURIER|IN\s+\d+\s+LOTS?)',
    r'DOCUMENTS?\s+(?:TO\s+BE|MUST\s+BE|SHOULD\s+BE)\s+(?:SENT|FORWARDED|DISPATCHED|COURIERED)\s+(?:TO|BY|VIA).{0,200}(?:DHL|FEDEX|UPS|TNT|ARAMEX|COURIER)',
    r'AT\s+(?:THE\s+)?BENEFICIARY[\'S]{0,2}\s+COST',
    r'IN\s+\d+\s+LOTS?\s+BY\s+(?:DHL|FEDEX|UPS|COURIER)',
]

# Compile patterns for performance (compiled once, used many times)
_NON_CHECKABLE_COMPILED = [re.compile(p, re.IGNORECASE) for p in NON_CHECKABLE_PATTERNS]

# Fields that contain clause lists — these need per-clause decomposition
CLAUSE_FIELDS: Set[str] = {'46A', '46B', '47A', '47B', '45A', '45B', '46C', '77A', '78', '79', '72'}

# Fields to skip entirely — internal metadata that should not appear in the report
SKIP_FIELDS: Set[str] = {
    'swift_format', 'mt_number', 'sender_institution', 'receiver_institution',
    'message_type', 'input_time', 'output_time', 'priority', 'message_reference',
    'session_number', 'sequence_number', 'ack_text', 'direction',
    'amendment_number', 'amendment_date', 'amendment_details',
    '_raw_pages', '_page_images', '_ocr_confidence',
}


def _is_non_checkable(text: str) -> tuple:
    """
    Check if a clause text matches non-checkable patterns.

    Returns (bool, reason) — True if the clause is a bank obligation,
    sanctions clause, or regulatory statement that cannot be verified
    by examining shipping documents.
    """
    text_upper = text.upper().strip()

    for pattern in _NON_CHECKABLE_COMPILED:
        if pattern.search(text_upper):
            return True, f"Matches non-checkable pattern: {pattern.pattern[:50]}"

    return False, ""


def _classify_field(field_tag: str, value: Any, clause_number: int = 0,
                     clause_text: str = '') -> ClauseStatus:
    """
    Classify a single field or clause using a priority-based rule cascade:

    Rule 1: Skip internal metadata fields (starts with '_' or in SKIP_FIELDS)
    Rule 2: Informational fields — LC metadata not requiring compliance check
    Rule 3: Non-checkable patterns — bank obligations, sanctions, regulatory
    Rule 4: Implicit check fields — date/amount fields checked by code
    Rule 5: Clause fields — document conditions verified by VLM
    Default: Informational for unknown standalone fields
    """

    # Determine clause_ref (unique identifier for this clause)
    if clause_number > 0:
        clause_ref = f"{field_tag}-{clause_number}"
    else:
        clause_ref = field_tag

    # Get display label (human-readable name for the report)
    display_label = INFORMATIONAL_FIELDS.get(field_tag, field_tag)

    # Get value preview (first 200 chars for report display)
    if isinstance(value, str):
        preview = value
    elif isinstance(value, list):
        preview = '; '.join(str(v)[:50] for v in value[:3])
    elif isinstance(value, dict):
        preview = str(value)
    else:
        preview = str(value) if value else ''

    if clause_text:
        preview = clause_text[:200]

    # Rule 1: Skip internal fields (never shown in report)
    if field_tag.startswith('_') or field_tag in SKIP_FIELDS:
        return ClauseStatus(
            clause_ref=clause_ref, field_tag=field_tag,
            clause_number=clause_number, status='skip',
            reason='Internal metadata field', display_label=display_label,
            value_preview=preview,
        )

    # Rule 2: Informational fields (shown in report but not checked)
    # Exception: implicit check fields (31D, 32B, 44C, 48) are checkable
    if field_tag in INFORMATIONAL_FIELDS and field_tag not in IMPLICIT_CHECK_FIELDS:
        return ClauseStatus(
            clause_ref=clause_ref, field_tag=field_tag,
            clause_number=clause_number, status='informational',
            reason=f'LC metadata: {display_label}',
            display_label=display_label, value_preview=preview,
        )

    # Rule 3: Check clause text for non-checkable patterns
    # (bank obligations, sanctions, regulatory — cannot verify from documents)
    check_text = clause_text or (value if isinstance(value, str) else '')
    if check_text:
        is_nc, nc_reason = _is_non_checkable(check_text)
        if is_nc:
            return ClauseStatus(
                clause_ref=clause_ref, field_tag=field_tag,
                clause_number=clause_number, status='non_checkable',
                reason=nc_reason, display_label=display_label,
                value_preview=preview,
            )

    # Rule 4: Implicit check fields (verified by Python code in Step 14)
    if field_tag in IMPLICIT_CHECK_FIELDS:
        return ClauseStatus(
            clause_ref=clause_ref, field_tag=field_tag,
            clause_number=clause_number, status='checkable',
            reason='Implicit code-based verification',
            display_label=display_label, value_preview=preview,
        )

    # Rule 5: Clause fields — document conditions verified by VLM
    if field_tag in CLAUSE_FIELDS:
        return ClauseStatus(
            clause_ref=clause_ref, field_tag=field_tag,
            clause_number=clause_number, status='checkable',
            reason='Document condition — VLM verified',
            display_label=display_label, value_preview=preview,
        )

    # Default: informational for unknown standalone fields
    return ClauseStatus(
        clause_ref=clause_ref, field_tag=field_tag,
        clause_number=clause_number, status='informational',
        reason='Standalone field — shown in report',
        display_label=display_label, value_preview=preview,
    )


# We need Any for the value parameter
from typing import Any


# ── Main Run ─────────────────────────────────────────────────────────────

def run(structured_lc: dict, output_dir: str = None, progress_callback=None) -> dict:
    """
    Execute Step 15: Classify all LC clauses as checkable/informational/non-checkable.

    Iterates through ALL fields in the consolidated LC and assigns each one a
    status. This map is used by Step 16 (confidence review) to know which
    clauses should be escalated and which are just informational.

    Args:
        structured_lc: Output from Step 7 (StructuredLC with consolidated_fields)
        output_dir: Directory to save results
        progress_callback: Optional callback(message: str)

    Returns:
        dict with 'clause_status_map', 'clause_statuses', 'summary', 'elapsed_seconds'
    """
    def _progress(msg):
        if progress_callback:
            progress_callback(f"[Step 15] {msg}")
        print(f"[Step 15] {msg}")

    start_time = time.time()

    fields = structured_lc.get('consolidated_fields', structured_lc)
    _progress(f"Classifying {len(fields)} LC fields...")

    statuses: List[ClauseStatus] = []
    clause_status_map: Dict[str, str] = {}  # {clause_ref: status} — compact lookup for Step 16

    for field_tag in sorted(fields.keys()):
        value = fields[field_tag]

        # Skip internal fields silently (they never appear in reports)
        if field_tag.startswith('_') or field_tag in SKIP_FIELDS:
            continue

        # Check if this is a clause field with multiple clauses
        if field_tag in CLAUSE_FIELDS:
            # Extract individual clauses from the list
            if isinstance(value, str):
                clause_texts = [value]
            elif isinstance(value, list):
                clause_texts = []
                for item in value:
                    if isinstance(item, str):
                        clause_texts.append(item)
                    elif isinstance(item, dict):
                        clause_texts.append(item.get('text', item.get('clause_text', str(item))))
                    else:
                        clause_texts.append(str(item))
            else:
                clause_texts = [str(value)]

            # Classify each clause individually (e.g., 47A-1 might be checkable
            # while 47A-3 is non-checkable if it is a sanctions clause)
            for i, text in enumerate(clause_texts, 1):
                if not text or not text.strip():
                    continue
                cs = _classify_field(field_tag, value, clause_number=i, clause_text=text.strip())
                statuses.append(cs)
                clause_status_map[cs.clause_ref] = cs.status

        else:
            # Standalone field (e.g., F20, F50, F32B)
            cs = _classify_field(field_tag, value)
            statuses.append(cs)
            clause_status_map[cs.clause_ref] = cs.status

    # Summary counts
    checkable = sum(1 for s in statuses if s.status == 'checkable')
    informational = sum(1 for s in statuses if s.status == 'informational')
    non_checkable = sum(1 for s in statuses if s.status == 'non_checkable')

    elapsed = time.time() - start_time
    _progress(f"Step 15 complete: {checkable} checkable, {informational} informational, {non_checkable} non-checkable in {elapsed:.1f}s")

    # Log each classification with a short icon for readability
    for cs in statuses:
        status_icon = {'checkable': 'CHK', 'informational': 'INF', 'non_checkable': 'N/C'}.get(cs.status, '???')
        _progress(f"  [{status_icon}] {cs.clause_ref}: {cs.reason[:60]}")

    summary = {
        'total_fields': len(statuses),
        'checkable': checkable,
        'informational': informational,
        'non_checkable': non_checkable,
    }

    # Save results to disk
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        result_file = os.path.join(output_dir, 'step15_result.json')
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({
                'step': 15,
                'step_name': 'Non-Compliance and Non-Checkable Clause Classification',
                'summary': summary,
                'elapsed_seconds': round(elapsed, 2),
                'clause_status_map': clause_status_map,
                'clause_statuses': [asdict(cs) for cs in statuses],
            }, f, indent=2, ensure_ascii=False)

    return {
        'clause_status_map': clause_status_map,
        'clause_statuses': statuses,
        'summary': summary,
        'elapsed_seconds': round(elapsed, 2),
    }


if __name__ == '__main__':
    # CLI test with sample LC data demonstrating all three classification types
    sample_lc = {
        'consolidated_fields': {
            '20': '0491ILC081972',
            '27': '1/1',
            '31C': '260115',
            '31D': '2026-06-30 PAKISTAN',
            '32B': 'USD 490,200.00',
            '40A': 'IRREVOCABLE',
            '40E': 'UCP LATEST VERSION',
            '41A': 'ANY BANK BY NEGOTIATION',
            '43P': 'ALLOWED',
            '43T': 'ALLOWED',
            '44A': 'KARACHI, PAKISTAN',
            '44B': 'ROTTERDAM, NETHERLANDS',
            '44C': '2026-06-15',
            '44E': 'KARACHI PORT',
            '44F': 'ROTTERDAM PORT',
            '48': '21 DAYS AFTER SHIPMENT DATE',
            '50': 'ABC TRADING COMPANY',
            '59': 'XYZ EXPORTS LTD',
            '52A': 'UNITED BANK LIMITED',
            '46A': [
                'SIGNED COMMERCIAL INVOICE IN 3 COPIES SHOWING HS CODE',
                'FULL SET OF CLEAN ON BOARD BILLS OF LADING',
                'INSURANCE POLICY IN DUPLICATE COVERING ALL RISKS',
            ],
            '47A': [
                'ALL DOCUMENTS MUST SHOW LC NUMBER AND DATE',
                'SANCTIONS CLAUSE: THIS LC IS SUBJECT TO APPLICABLE SANCTIONS LAWS',
                'REIMBURSEMENT INSTRUCTIONS: BANK TO BANK REIMBURSEMENT UNDER URR 725',
            ],
            '71D': 'ALL BANKING CHARGES OUTSIDE COUNTRY OF ISSUING BANK ARE FOR ACCOUNT OF BENEFICIARY',
        }
    }
    result = run(sample_lc)
    print(f"\nStatus Map:")
    for ref, status in sorted(result['clause_status_map'].items()):
        print(f"  {ref}: {status}")
