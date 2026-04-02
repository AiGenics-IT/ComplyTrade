"""
Step 16 — Confidence-Based Review Escalation
===============================================
Reviews all verified rows from Step 14 and escalates any with low confidence
to REVIEW status, ensuring uncertain results are flagged for human attention.

PURPOSE:
    The VLM verification in Step 14 assigns a confidence score (0.0 to 1.0) to
    each compliance judgment. If the model is unsure about its answer (e.g., the
    document text was partially illegible, or the condition was ambiguous), the
    confidence will be low.

    This step acts as a safety net: any PASS or FAIL with confidence below the
    threshold (default 98%) is changed to REVIEW, meaning a human must confirm
    the result. This prevents both false positives (incorrectly passing a condition)
    and false negatives (incorrectly failing a condition).

    Additionally, this step checks the result text for uncertainty markers like
    "unclear", "cannot determine", "partially" — these also trigger escalation
    regardless of the numeric confidence score.

INPUTS:
    - Verified rows from Step 14 with confidence scores
    - Clause status map from Step 15 (to preserve informational/non-checkable statuses)

OUTPUTS:
    - ReviewedRow objects with original_compliance preserved and final compliance set
    - Count of escalated rows

TRADE FINANCE CONTEXT:
    In bank document examination, if a checker is uncertain about a condition,
    they escalate it to a senior officer or the trade finance operations manager.
    This step automates that escalation based on confidence thresholds.

AI MODEL: None — rule-based logic only (threshold comparison and text pattern matching).
"""

import json
import sys as _sys; _sys.stdout.reconfigure(encoding="utf-8", errors="replace") if hasattr(_sys.stdout, "reconfigure") else None
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import CONFIDENCE_THRESHOLD


@dataclass
class ReviewedRow:
    """
    A verification row after confidence review.

    Preserves the original compliance judgment (from Step 14) alongside the
    final compliance (which may be changed to 'review' if confidence is low).
    This allows the report to show what the AI thought AND flag uncertainty.
    """
    clause_ref: str
    clause_text: str
    condition_text: str
    document_checked: str
    findings: str
    result: str
    original_compliance: str  # The compliance BEFORE review check (e.g., 'pass' or 'fail')
    compliance: str           # Final compliance — may be changed to 'review' if uncertain
    confidence: float = 0.0
    is_review_escalated: bool = False   # True if this row was escalated by this step
    review_reason: str = ""             # Why it was escalated (low confidence, uncertainty text, etc.)
    source_page: int = 0                # Source page number for traceability
    dependency_note: str = ""           # Cross-clause dependency note from Step 17


def run(verified_rows: list, clause_status_map: dict = None,
        output_dir: str = None, progress_callback=None) -> dict:
    """
    Execute Step 16: Confidence-based review escalation.

    For each verified row:
    1. If the clause is informational/non-checkable (from Step 15), preserve that status
    2. If confidence < threshold AND compliance is pass/fail, escalate to 'review'
    3. If result text contains uncertainty markers, escalate to 'review'
    4. Otherwise, keep the original compliance judgment

    Args:
        verified_rows: List of verified row dicts from Step 14
        clause_status_map: Dict from Step 15 {clause_ref: status} — used to preserve
                           informational/non-checkable classifications
        output_dir: Directory to save results
        progress_callback: Optional callback for progress

    Returns:
        dict with 'rows' (list of ReviewedRow), 'escalated_count', 'elapsed_seconds'
    """
    def _progress(msg):
        if progress_callback:
            progress_callback(f"[Step 16] {msg}")
        print(f"[Step 16] {msg}")

    start_time = time.time()
    threshold = CONFIDENCE_THRESHOLD  # Default: 0.98 (98%)
    _progress(f"Reviewing {len(verified_rows)} rows with confidence threshold {threshold}")

    reviewed = []
    escalated_count = 0

    for row in verified_rows:
        # Normalize to dict format
        if isinstance(row, dict):
            r = row.copy()
        else:
            r = asdict(row) if hasattr(row, '__dataclass_fields__') else {'condition_text': str(row)}

        confidence = float(r.get('confidence', 0.0))
        original_compliance = r.get('compliance', 'review')
        clause_ref = r.get('clause_ref', '')

        # ── Check if this clause is non-checkable (from Step 15) ──
        # Informational and non-checkable clauses keep their status — no escalation
        if clause_status_map:
            clause_status = clause_status_map.get(clause_ref, 'checkable')
            if clause_status in ('informational', 'non_checkable'):
                reviewed.append(ReviewedRow(
                    clause_ref=clause_ref,
                    clause_text=r.get('clause_text', ''),
                    condition_text=r.get('condition_text', ''),
                    document_checked=r.get('document_checked', ''),
                    findings=r.get('findings', r.get('found_text', '')),
                    result=r.get('result', ''),
                    original_compliance=original_compliance,
                    compliance=clause_status,  # Keep as informational/non_checkable
                    confidence=confidence,
                    is_review_escalated=False,
                    review_reason=f"Clause marked as {clause_status}",
                    source_page=r.get('source_page', 0),
                    dependency_note=r.get('dependency_note', ''),
                ))
                continue

        # ── Confidence-based escalation ──
        # If the AI's confidence is below threshold, do not trust the pass/fail judgment
        is_escalated = False
        review_reason = ""
        final_compliance = original_compliance

        if confidence < threshold and original_compliance in ('pass', 'fail'):
            is_escalated = True
            final_compliance = 'review'
            review_reason = f"Low confidence: {confidence:.2f} (threshold: {threshold})"
            escalated_count += 1

        # ── Text-based uncertainty detection ──
        # Even if confidence score is high, certain phrases in the result text
        # indicate the VLM was unsure about its answer
        result_text = (r.get('result', '') or '').lower()
        uncertainty_markers = [
            'unclear', 'cannot determine', 'not sure', 'may not',
            'possibly', 'might be', 'could not verify', 'ambiguous',
            'partially', 'unable to confirm',
        ]
        if any(marker in result_text for marker in uncertainty_markers):
            if not is_escalated:
                is_escalated = True
                final_compliance = 'review'
                review_reason = "Result text indicates uncertainty"
                escalated_count += 1

        reviewed.append(ReviewedRow(
            clause_ref=clause_ref,
            clause_text=r.get('clause_text', ''),
            condition_text=r.get('condition_text', ''),
            document_checked=r.get('document_checked', ''),
            findings=r.get('findings', r.get('found_text', '')),
            result=r.get('result', ''),
            original_compliance=original_compliance,
            compliance=final_compliance,
            confidence=confidence,
            is_review_escalated=is_escalated,
            review_reason=review_reason,
            source_page=r.get('source_page', 0),
            dependency_note=r.get('dependency_note', ''),
        ))

    elapsed = time.time() - start_time
    _progress(f"Step 16 complete: {escalated_count} rows escalated to REVIEW out of {len(reviewed)}")

    # Save results to disk
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        result_file = os.path.join(output_dir, 'step16_result.json')
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({
                'step': 16,
                'step_name': 'Confidence-Based Review Escalation',
                'total_rows': len(reviewed),
                'escalated_count': escalated_count,
                'confidence_threshold': threshold,
                'pass_count': sum(1 for r in reviewed if r.compliance == 'pass'),
                'fail_count': sum(1 for r in reviewed if r.compliance == 'fail'),
                'review_count': sum(1 for r in reviewed if r.compliance == 'review'),
                'informational_count': sum(1 for r in reviewed if r.compliance == 'informational'),
                'elapsed_seconds': round(elapsed, 2),
                'rows': [asdict(r) for r in reviewed],
            }, f, indent=2, ensure_ascii=False)

    return {
        'rows': reviewed,
        'escalated_count': escalated_count,
        'elapsed_seconds': round(elapsed, 2),
    }
