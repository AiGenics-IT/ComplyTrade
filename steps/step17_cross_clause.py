"""
Step 17 — Cross-Clause Dependency Handling
=============================================
Reconciles verification results by checking for F47A overrides or
exemptions that modify F46A document requirements.

PURPOSE:
    In a Letter of Credit, F47A (Additional Conditions) can override or modify
    requirements stated in F46A (Documents Required). This step detects such
    overrides and adjusts verification results accordingly.

    EXAMPLE:
    - F46A-2 requires "OCEAN BILL OF LADING" (direct ocean BL)
    - F47A-3 states "CHARTER PARTY BILL OF LADING ACCEPTABLE"
    - Without this step, a Charter Party BL would FAIL the F46A-2 check
    - This step detects the F47A-3 override and changes the FAIL to PASS

    F47A = Additional Conditions in SWIFT MT700 — these can override, supplement,
    or add conditions to the F46A document requirements. They take precedence
    because they represent later/more specific instructions from the issuing bank.

TRADE FINANCE CONTEXT:
    - "ACCEPTABLE" / "ALLOWED" / "PERMITTED" = permissive language that overrides a stricter F46A requirement
    - "IN LIEU OF" = one document type can substitute for another
    - "PROVIDED THAT" / "SUBJECT TO" = conditional acceptance (override applies only if condition is met)
    - F46A = Documents Required — the primary list of required shipping documents
    - F47A = Additional Conditions — can add, modify, or relax F46A requirements

INPUTS:
    - Verified rows from Steps 14-16 (list of row dicts with compliance results)

OUTPUTS:
    - Reconciled rows with dependency_notes explaining any overrides applied
    - Count of overrides applied

AI MODEL: None — rule-based logic using regex pattern matching on clause text.
"""

import re
import sys as _sys; _sys.stdout.reconfigure(encoding="utf-8", errors="replace") if hasattr(_sys.stdout, "reconfigure") else None
import json
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from pathlib import Path


# ── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class DependencyNote:
    """
    Records a cross-clause dependency resolution.

    When an F47A clause overrides an F46A requirement, this note captures:
    - Which F47A clause provided the override (source)
    - Which F46A row was affected (target)
    - What action was taken (override removed, context added)
    - What the result was before and after reconciliation
    """
    source_clause: str          # e.g. "F47A-3" — the permissive clause
    target_clause: str          # e.g. "F46A-1" — the row that was failing
    action: str                 # "override_removed" | "context_added" | "exemption_applied"
    reason: str                 # human-readable explanation
    original_result: str        # what the row result WAS before reconciliation
    new_result: str             # what the row result IS after reconciliation


@dataclass
class ReconciledRow:
    """
    A verification row after cross-clause dependency resolution.

    If an F47A clause overrides an F46A failure, the row's result and compliance
    are updated, and the dependency_notes list explains why.
    """
    clause_ref: str
    clause_text: str
    condition: str
    findings: str
    document_checked: str
    result: str                 # "PASS" | "FAIL" | "REVIEW"
    compliance: str             # "COMPLIED" | "NOT COMPLIED" | "REVIEW REQUIRED"
    dependency_notes: List[DependencyNote] = field(default_factory=list)
    reconciled: bool = False    # True if result was modified by cross-clause logic


# ── Permissive Pattern Detection ─────────────────────────────────────────────
# These regex patterns detect permissive language in F47A clauses.
# Permissive language indicates that something normally not allowed IS acceptable.

_PERMISSIVE_PATTERNS = [
    re.compile(r'\b(ACCEPTABLE|ACCEPTED)\b', re.IGNORECASE),
    re.compile(r'\b(ALLOWED|ALLOW)\b', re.IGNORECASE),
    re.compile(r'\b(PERMITTED|PERMISSIBLE)\b', re.IGNORECASE),
    re.compile(r'\b(NOT\s+OBJECTIONABLE)\b', re.IGNORECASE),
    re.compile(r'\b(IN\s+LIEU\s+OF)\b', re.IGNORECASE),         # One doc type substitutes for another
    re.compile(r'\b(MAY\s+BE\s+SUBSTITUTED)\b', re.IGNORECASE),
    re.compile(r'\b(ALSO\s+ACCEPTABLE)\b', re.IGNORECASE),
]

# Conditional acceptance patterns — permits something but only if a condition is met.
# Example: "CHARTER PARTY BL ACCEPTABLE PROVIDED THAT IT IS SIGNED BY THE MASTER"
_CONDITIONAL_PATTERNS = [
    re.compile(
        r'\b(ALLOWED|PERMITTED|ACCEPTABLE)\b.*?\b(PROVIDED|SUBJECT\s+TO|ON\s+CONDITION)\b',
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r'\b(PROVIDED|SUBJECT\s+TO)\b.*?\b(ALLOWED|PERMITTED|ACCEPTABLE)\b',
        re.IGNORECASE | re.DOTALL,
    ),
]

# Document-type keywords used to match F47A overrides to F46A requirements.
# When an F47A clause contains "CHARTER PARTY" and "ACCEPTABLE", it overrides
# any F46A failure related to "CHARTER PARTY" bills of lading.
_DOC_KEYWORDS = [
    'CHARTER PARTY', 'CHARTERPARTY', 'HOUSE BILL', 'HOUSE BL',
    'FORWARDER', 'FREIGHT FORWARDER', 'MULTIMODAL', 'COMBINED TRANSPORT',
    'THIRD PARTY', '3RD PARTY', 'STALE', 'LATE SHIPMENT', 'LATE PRESENTATION',
    'TRANSHIPMENT', 'TRANSSHIPMENT', 'PARTIAL SHIPMENT', 'SHORT FORM',
    'COPY', 'PHOTOCOPY', 'FAX', 'TELEFAX', 'COURIER RECEIPT',
    'OCEAN BILL', 'OCEAN BL', 'MARINE BILL', 'AIRWAY BILL', 'AWB',
    'INSURANCE', 'CERTIFICATE', 'INVOICE', 'DRAFT', 'BILL OF EXCHANGE',
    'PACKING LIST', 'WEIGHT LIST', 'BENEFICIARY CERTIFICATE',
    'CERTIFICATE OF ORIGIN', 'COO', 'FUMIGATION', 'PHYTOSANITARY',
    'INSPECTION', 'NON-NEGOTIABLE', 'NEGOTIABLE',
    'CLEAN ON BOARD', 'ON BOARD', 'SHIPPED ON BOARD',
    'BLANK ENDORSED', 'ENDORSEMENT',
]


def _extract_subject_keywords(clause_text: str) -> List[str]:
    """Extract document-type keywords from a permissive F47A clause."""
    text_upper = clause_text.upper()
    found = []
    for kw in _DOC_KEYWORDS:
        if kw in text_upper:
            found.append(kw)
    return found


def _is_permissive(clause_text: str) -> bool:
    """Check if a clause contains permissive language (ACCEPTABLE, ALLOWED, etc.)."""
    for pat in _PERMISSIVE_PATTERNS:
        if pat.search(clause_text):
            return True
    return False


def _is_conditional(clause_text: str) -> Optional[str]:
    """
    Check if a clause has conditional acceptance.

    Returns the condition text (e.g., "SIGNED BY THE MASTER") or None.
    A conditional acceptance means the override applies but needs manual
    verification of the condition — so the result becomes REVIEW, not PASS.
    """
    for pat in _CONDITIONAL_PATTERNS:
        m = pat.search(clause_text)
        if m:
            # Extract the condition portion (everything after PROVIDED/SUBJECT TO)
            cond_match = re.search(
                r'\b(?:PROVIDED|SUBJECT\s+TO|ON\s+CONDITION)\s+(?:THAT\s+)?(.+)',
                clause_text, re.IGNORECASE | re.DOTALL,
            )
            if cond_match:
                return cond_match.group(1).strip()[:200]
            return "conditional acceptance applies"
    return None


def _keyword_overlap(keywords: List[str], target_text: str) -> List[str]:
    """
    Find which keywords from a permissive F47A clause match a target row's text.

    This is how we link an F47A override to the specific F46A failure it resolves.
    For example, if the F47A clause mentions "CHARTER PARTY" and the failing
    F46A row also mentions "CHARTER PARTY", they are related.
    """
    target_upper = target_text.upper()
    return [kw for kw in keywords if kw in target_upper]


# ── Core Reconciliation Logic ────────────────────────────────────────────────

def _collect_f47a_clauses(rows: List[Dict]) -> List[Dict]:
    """Collect all F47A clause rows for cross-reference against F46A failures."""
    f47a = []
    for row in rows:
        ref = row.get('clause_ref', '')
        if ref.upper().startswith('F47A'):
            f47a.append(row)
    return f47a


def _reconcile_rows(
    rows: List[Dict],
    progress_fn=None,
) -> List[ReconciledRow]:
    """
    Main reconciliation: scan F47A for permissive overrides and resolve
    matching F46A/other fails.

    Algorithm:
    1. Collect all F47A clauses
    2. Filter to those with permissive language (ACCEPTABLE, ALLOWED, etc.)
    3. Extract document-type keywords from each permissive F47A clause
    4. For each FAIL row in the results:
       a. Check if any permissive F47A keywords overlap with the failing row's text
       b. If yes: unconditional override → change to PASS
       c. If yes but conditional: change to REVIEW (human must verify the condition)
    """
    # Step 1-3: Build override map from F47A permissive clauses
    f47a_clauses = _collect_f47a_clauses(rows)

    overrides = []
    for clause in f47a_clauses:
        text = clause.get('clause_text', '') or clause.get('condition', '')
        if not _is_permissive(text):
            continue
        keywords = _extract_subject_keywords(text)
        condition_text = _is_conditional(text)
        overrides.append({
            'clause_ref': clause.get('clause_ref', ''),
            'clause_text': text,
            'keywords': keywords,
            'is_conditional': condition_text is not None,
            'condition': condition_text,
        })

    if progress_fn:
        progress_fn(f"Found {len(overrides)} permissive F47A overrides")

    # Step 4: Try to resolve each FAIL using F47A overrides
    reconciled = []
    override_count = 0

    for row in rows:
        rec = ReconciledRow(
            clause_ref=row.get('clause_ref', ''),
            clause_text=row.get('clause_text', ''),
            condition=row.get('condition', ''),
            findings=row.get('findings', row.get('found_text', '')),
            document_checked=row.get('document_checked', ''),
            result=row.get('result', 'REVIEW'),
            compliance=row.get('compliance', 'REVIEW REQUIRED'),
        )

        # Only try to override FAIL results — PASS and REVIEW stay unchanged
        if rec.result != 'FAIL':
            reconciled.append(rec)
            continue

        # Combine all text fields for keyword matching
        row_text = f"{rec.clause_text} {rec.condition} {rec.findings} {rec.document_checked}"

        # Check each permissive F47A override for keyword overlap
        for override in overrides:
            matched_kw = _keyword_overlap(override['keywords'], row_text)
            if not matched_kw:
                continue

            # We have a keyword match — this F47A clause overrides this failure
            note = DependencyNote(
                source_clause=override['clause_ref'],
                target_clause=rec.clause_ref,
                action='override_removed' if not override['is_conditional'] else 'context_added',
                reason=(
                    f"F47A clause \"{override['clause_ref']}\" permits "
                    f"{', '.join(matched_kw)}: \"{override['clause_text'][:120]}\""
                ),
                original_result=rec.result,
                new_result='PASS' if not override['is_conditional'] else 'REVIEW',
            )
            rec.dependency_notes.append(note)

            if override['is_conditional']:
                # Conditional override: change to REVIEW (human must verify the condition)
                rec.result = 'REVIEW'
                rec.compliance = 'REVIEW REQUIRED'
                rec.condition += f" [CONDITIONAL: {override['condition']}]"
            else:
                # Unconditional override: change FAIL to PASS
                rec.result = 'PASS'
                rec.compliance = 'COMPLIED'

            rec.reconciled = True
            override_count += 1
            break  # One override is enough — stop checking further overrides

        reconciled.append(rec)

    if progress_fn:
        progress_fn(f"Reconciled {override_count} fails via F47A overrides")

    return reconciled


# ── Runner ───────────────────────────────────────────────────────────────────

def run(
    verified_rows: List[Dict],
    output_dir: str,
    progress_fn=None,
) -> Dict[str, Any]:
    """
    Execute Step 17: Cross-Clause Dependency Handling.

    Scans all verification rows for F47A permissive overrides that should
    modify F46A failures. This prevents false discrepancies when the LC's
    additional conditions explicitly permit something that F46A would otherwise forbid.

    Args:
        verified_rows: list of verified row dicts from Steps 14-16
        output_dir:    directory for step output
        progress_fn:   callback for progress messages

    Returns:
        dict with step results including reconciled_rows
    """
    t0 = time.time()
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if progress_fn is None:
        def progress_fn(msg): pass

    progress_fn("Step 17: Cross-Clause Dependency Handling")
    progress_fn(f"Processing {len(verified_rows)} verified rows")

    reconciled = _reconcile_rows(verified_rows, progress_fn)

    # Compute summary statistics
    total = len(reconciled)
    overridden = sum(1 for r in reconciled if r.reconciled)
    passes = sum(1 for r in reconciled if r.result == 'PASS')
    fails = sum(1 for r in reconciled if r.result == 'FAIL')
    reviews = sum(1 for r in reconciled if r.result == 'REVIEW')

    result = {
        'step': 17,
        'step_name': 'Cross-Clause Dependency Handling',
        'total_rows': total,
        'overridden_count': overridden,
        'passes': passes,
        'fails': fails,
        'reviews': reviews,
        'reconciled_rows': [asdict(r) for r in reconciled],
        'elapsed_seconds': round(time.time() - t0, 2),
    }

    # Save result to disk
    out_path = Path(output_dir) / 'step17_result.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    progress_fn(
        f"Step 17 complete: {overridden} overrides applied "
        f"({passes}P / {fails}F / {reviews}R)"
    )

    return result
