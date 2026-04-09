"""
Step 13 — Clause Verification Row Construction
=================================================
Takes decomposed conditions from Step 12 and builds the 5-column
verification table structure used in the compliance report.

PURPOSE:
    This step transforms the hierarchical decomposition output (clauses -> conditions)
    into a flat list of verification rows. Each row represents ONE condition to check
    against ONE document. This flat structure is what Steps 14-17 process and what
    Step 20 renders into the PDF report tables.

    Think of it as building the blank examination worksheet that a bank checker would
    use — all the rows are listed, but the "Found Text", "Result", and "Compliance"
    columns are empty (pending verification).

5-Column Structure (matches the final PDF report layout):
    1. Condition(s)      — human-readable condition text from Step 12
    2. Found Text        — exact text found in document (initially "Nil" — filled by Step 14)
    3. Document Checked  — which document type was checked
    4. Result            — short result, 4-5 words max (initially "Pending Verification")
    5. Compliance        — PASS / FAIL / REVIEW (initially "PENDING")

INPUTS:
    - Step 12 decomposed_clauses (list of DecomposedClause objects or dicts)

OUTPUTS:
    - List of VerificationRow objects with all 5 columns initialized
    - Summary counts: VLM rows, implicit rows, informational rows

TRADE FINANCE CONTEXT:
    The verification table mirrors the format used by trade finance operations (TFO)
    departments in banks. Each row is one check item on the document examination
    worksheet. The bank checker goes through each row, examines the relevant document,
    and records their finding.

AI MODEL: None — this is pure data transformation (no model calls).
"""

import json
import sys as _sys; _sys.stdout.reconfigure(encoding="utf-8", errors="replace") if hasattr(_sys.stdout, "reconfigure") else None
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Data Models ──────────────────────────────────────────────────────────

@dataclass
class VerificationRow:
    """
    A single row in the compliance verification table.

    This maps directly to one row in the final PDF report. The 5-column
    fields (condition_text through compliance) are what appear in the table.
    The metadata fields provide context for Steps 14-17 processing.
    """
    row_id: str = ""                   # Unique row identifier (e.g., "R0001")
    clause_ref: str = ""               # Parent clause reference (e.g., "46A-1")
    field_tag: str = ""                # LC field tag (e.g., "46A")
    condition_id: str = ""             # Condition ID from Step 12 (e.g., "46A-1-C1")
    # ── 5-Column Table (these appear in the PDF report) ──
    condition_text: str = ""           # Column 1: What to check (human-readable condition)
    found_text: str = "Nil"            # Column 2: What was found in the document (filled by Step 14)
    document_checked: str = ""         # Column 3: Which document type was examined
    result: str = "Pending Verification"  # Column 4: Short result (4-5 words)
    compliance: str = "PENDING"        # Column 5: PASS / FAIL / REVIEW / PENDING / N/A
    # ── Metadata (used by Steps 14-17, not shown in report) ──
    look_for_value: str = ""           # Expected value from decomposition (search target)
    is_implicit: bool = False          # True = code-based check (dates/amounts), False = VLM check
    implicit_type: str = ""            # Type of implicit check (lc_expiry, late_shipment, etc.)
    confidence: float = 0.0            # Verification confidence score (set by Step 14, reviewed by Step 16)
    original_clause_text: str = ""     # Full original clause for context when reviewing
    verification_notes: str = ""       # Additional notes from verification process


def _build_rows_from_decomposed(decomposed_clauses: list) -> List[VerificationRow]:
    """
    Convert decomposed clauses into flat verification rows.

    Each condition in a DecomposedClause becomes one VerificationRow.
    If a clause has no conditions (purely informational, like a bank instruction),
    it still gets a row marked as "Informational" / "N/A" so it appears in the
    report for completeness — bank checkers expect to see ALL LC clauses accounted for.
    """
    rows = []
    row_counter = 0

    for dc in decomposed_clauses:
        # Handle both dataclass objects and plain dicts (depending on serialization)
        if hasattr(dc, 'clause_ref'):
            clause_ref = dc.clause_ref
            field_tag = dc.field_tag
            original_text = dc.original_text
            conditions = dc.conditions
        else:
            clause_ref = dc.get('clause_ref', '')
            field_tag = dc.get('field_tag', '')
            original_text = dc.get('original_text', '')
            conditions = dc.get('conditions', [])

        if not conditions:
            # Clause with no checkable conditions — create an informational row.
            # This happens for clauses like bank-to-bank instructions (F78) or
            # regulatory statements that cannot be verified from documents.
            row_counter += 1
            rows.append(VerificationRow(
                row_id=f"R{row_counter:04d}",
                clause_ref=clause_ref,
                field_tag=field_tag,
                condition_id=f"{clause_ref}-INFO",
                condition_text=original_text if original_text else "N/A",
                found_text="N/A",
                document_checked="N/A",
                result="Informational",
                compliance="N/A",
                original_clause_text=original_text,
            ))
            continue

        # Create one row per condition — each is independently verifiable
        for cond in conditions:
            # Handle both dataclass and dict format for conditions
            if hasattr(cond, 'condition_text'):
                cond_text = cond.condition_text
                doc_to_check = cond.document_to_check
                look_for = cond.look_for_value
                cond_id = cond.condition_id
                is_implicit = cond.is_implicit
                implicit_type = cond.implicit_type
            else:
                cond_text = cond.get('condition_text', '')
                doc_to_check = cond.get('document_to_check', '')
                look_for = cond.get('look_for_value', '')
                cond_id = cond.get('condition_id', f"{clause_ref}-CX")
                is_implicit = cond.get('is_implicit', False)
                implicit_type = cond.get('implicit_type', '')

            # ── Skip non-documentary conditions per UCP 600 Art 14(h) ──
            # When step 12's decomposition cannot tie a condition to an
            # actual document type, it returns document_to_check='N/A'
            # (or empty / blank). These are non-documentary conditions
            # like "documents to be sent via email", "intimate consignee
            # by fax", "advise applicant by email" — the LC says HOW to
            # do something, not WHAT to present. UCP 600 Art 14(h) says
            # banks disregard non-documentary conditions, so we either:
            #   (a) emit a single informational row noting the condition
            #       was disregarded — preserves visibility for the user
            #       without producing a false REQUIRED DOCUMENT MISSING
            #   (b) skip the row entirely — cleanest for the report
            # We go with (a) so the user can see at a glance that the
            # clause IS recognised, just not enforced.
            _doc_norm = (doc_to_check or '').strip()
            if (not _doc_norm
                    or _doc_norm.upper() == 'N/A'
                    or _doc_norm.upper() == 'NA'
                    or _doc_norm.upper() == 'NONE'):
                row_counter += 1
                rows.append(VerificationRow(
                    row_id=f"R{row_counter:04d}",
                    clause_ref=clause_ref,
                    field_tag=field_tag,
                    condition_id=cond_id,
                    condition_text=cond_text,
                    found_text="Non-documentary condition (UCP 600 Art 14(h))",
                    document_checked="(non-documentary)",
                    result="Disregarded — non-documentary condition per UCP 600 Art 14(h)",
                    compliance="N/A",
                    look_for_value=look_for,
                    is_implicit=is_implicit,
                    implicit_type=implicit_type,
                    original_clause_text=original_text,
                ))
                continue

            row_counter += 1
            rows.append(VerificationRow(
                row_id=f"R{row_counter:04d}",
                clause_ref=clause_ref,
                field_tag=field_tag,
                condition_id=cond_id,
                condition_text=cond_text,
                found_text="Nil",                        # Will be filled by Step 14
                document_checked=doc_to_check,
                result="Pending Verification",           # Will be filled by Step 14
                compliance="PENDING",                    # Will be set by Step 14
                look_for_value=look_for,
                is_implicit=is_implicit,
                implicit_type=implicit_type,
                original_clause_text=original_text,
            ))

    return rows


# ── Main Run ─────────────────────────────────────────────────────────────

def run(step12_result: dict, output_dir: str = None, progress_callback=None) -> dict:
    """
    Execute Step 13: Build verification rows from decomposed clauses.

    This is a fast, purely structural transformation — no model calls or I/O.
    It converts the hierarchical clause->conditions structure into a flat row list.

    Args:
        step12_result: Output from Step 12 (with 'decomposed_clauses')
        output_dir: Directory to save results
        progress_callback: Optional callback(message: str)

    Returns:
        dict with 'rows' (list of VerificationRow), 'total_rows', 'elapsed_seconds'
    """
    def _progress(msg):
        if progress_callback:
            progress_callback(f"[Step 13] {msg}")
        print(f"[Step 13] {msg}")

    start_time = time.time()

    decomposed = step12_result.get('decomposed_clauses', [])
    _progress(f"Building verification rows from {len(decomposed)} decomposed clauses...")

    rows = _build_rows_from_decomposed(decomposed)

    # Summarize row types for progress reporting
    implicit_count = sum(1 for r in rows if r.is_implicit)     # Code-based checks (dates, amounts)
    vlm_count = sum(1 for r in rows if not r.is_implicit and r.compliance == 'PENDING')  # VLM checks
    info_count = sum(1 for r in rows if r.compliance == 'N/A')  # Informational (non-checkable)

    elapsed = time.time() - start_time
    _progress(f"Step 13 complete: {len(rows)} rows ({vlm_count} VLM, {implicit_count} implicit, {info_count} informational) in {elapsed:.1f}s")

    # Group summary by field tag — helps understand distribution of checks across LC sections
    tag_counts: Dict[str, int] = {}
    for r in rows:
        tag_counts[r.field_tag] = tag_counts.get(r.field_tag, 0) + 1
    for tag, count in sorted(tag_counts.items()):
        _progress(f"  {tag}: {count} rows")

    # Save results to disk
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        result_file = os.path.join(output_dir, 'step13_result.json')
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({
                'step': 13,
                'step_name': 'Clause Verification Row Construction',
                'total_rows': len(rows),
                'vlm_rows': vlm_count,
                'implicit_rows': implicit_count,
                'informational_rows': info_count,
                'elapsed_seconds': round(elapsed, 2),
                'tag_summary': tag_counts,
                'rows': [asdict(r) for r in rows],
            }, f, indent=2, ensure_ascii=False)

    return {
        'rows': rows,
        'total_rows': len(rows),
        'vlm_rows': vlm_count,
        'implicit_rows': implicit_count,
        'informational_rows': info_count,
        'elapsed_seconds': round(elapsed, 2),
    }


if __name__ == '__main__':
    # CLI test with sample Step 12 output
    from step12_decomposition import DecomposedClause, Condition

    sample_step12 = {
        'decomposed_clauses': [
            DecomposedClause(
                clause_ref='46A-1', field_tag='46A', clause_number=1,
                original_text='SIGNED COMMERCIAL INVOICE IN 3 COPIES SHOWING HS CODE',
                conditions=[
                    Condition(condition_id='46A-1-C1', condition_text='Invoice must be signed',
                              document_to_check='Commercial Invoice', look_for_value='signature'),
                    Condition(condition_id='46A-1-C2', condition_text='HS Code must appear on invoice',
                              document_to_check='Commercial Invoice', look_for_value='HS Code'),
                ],
            ),
            DecomposedClause(
                clause_ref='31D-1', field_tag='31D', clause_number=1,
                original_text='2026-06-30 PAKISTAN',
                conditions=[
                    Condition(condition_id='31D-1-C1',
                              condition_text='Presentation date must not exceed LC expiry date',
                              document_to_check='Documentary Remittance',
                              look_for_value='2026-06-30', is_implicit=True, implicit_type='lc_expiry'),
                ],
            ),
        ],
    }

    result = run(sample_step12)
    for row in result['rows']:
        print(f"  {row.row_id} | {row.clause_ref} | {row.condition_text[:50]} | {row.document_checked} | {row.compliance}")
