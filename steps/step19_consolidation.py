"""
Step 19 -- Consolidated Clause Verification Output
=====================================================
Merges ALL verified, reconciled rows from Steps 14-17 into one coherent
structure grouped by LC field for final reporting.

PURPOSE:
    After verification (Step 14), confidence review (Step 16), and cross-clause
    reconciliation (Step 17), we have a flat list of rows with mixed clause refs.
    This step organizes them into a hierarchical structure suitable for the PDF
    report and the web UI:

    ConsolidatedOutput
      -> SectionGroup ("Key Terms", "Document Requirements", etc.)
        -> ClauseGroup ("F46A-1", "F46A-2", "F47A-1", etc.)
          -> VerificationRow (individual conditions with PASS/FAIL/REVIEW)

    It also computes the overall compliance decision:
    - COMPLIANT = all checks passed (no fails, no reviews)
    - DISCREPANT = at least one check failed
    - REVIEW REQUIRED = no fails but some checks need human review

REPORT SECTIONS (in order):
    1. Key Terms       -- F20, F31C, F31D, F32B, F39A, F42C, F43P, F43T, F44E, F44F, F44C, etc.
    2. Document Reqs   -- F46A-1, F46A-2, ..., F46A-N (individual document type requirements)
    3. Additional Cond -- F47A-1, F47A-2, ..., F47A-N (additional conditions that cross-cut documents)
    4. Goods Desc      -- F45A (description of goods and services)
    5. Instructions    -- F78, F72, F79, F77A (bank instructions and narratives)
    6. Other           -- any fields not classified into the above sections

INPUTS:
    - Reconciled rows from Step 17 (list of dicts with clause_ref, result, compliance)

OUTPUTS:
    - ConsolidatedOutput with sections, clause groups, overall decision
    - Critical findings list (all FAILs)
    - Review items list (all REVIEWs)

AI MODEL: None -- structural transformation only (grouping, counting, sorting).
"""

import json
import sys as _sys; _sys.stdout.reconfigure(encoding="utf-8", errors="replace") if hasattr(_sys.stdout, "reconfigure") else None
import time
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from pathlib import Path


# ── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class VerificationRow:
    """A single verification check row -- one condition checked against one document."""
    condition: str
    findings: str
    document_checked: str
    result: str                 # "PASS" | "FAIL" | "REVIEW"
    compliance: str             # "COMPLIED" | "NOT COMPLIED" | "REVIEW REQUIRED"
    dependency_notes: List[Dict] = field(default_factory=list)  # Cross-clause notes from Step 17
    reconciled: bool = False    # True if result was modified by cross-clause logic


@dataclass
class ClauseGroup:
    """
    A group of verification rows for one LC clause.

    For example, F46A-1 might say "SIGNED COMMERCIAL INVOICE IN 3 COPIES SHOWING HS CODE"
    and have 2 rows: one for signature check, one for HS Code check.
    The overall_result for the group is COMPLIED only if ALL rows pass.
    """
    clause_ref: str             # e.g. "F46A-1", "F47A-3", "F20"
    clause_text: str            # original LC clause text
    rows: List[VerificationRow] = field(default_factory=list)
    overall_result: str = ""    # "COMPLIED" | "NOT COMPLIED" | "REVIEW REQUIRED"
    row_count: int = 0
    pass_count: int = 0
    fail_count: int = 0
    review_count: int = 0


@dataclass
class SectionGroup:
    """
    A named section containing multiple clause groups.

    Sections organize the report into logical areas that bank checkers expect:
    Key Terms, Document Requirements, Additional Conditions, etc.
    """
    section_name: str           # e.g. "Key Terms", "Document Requirements"
    section_order: int          # Display order in the report
    clauses: List[ClauseGroup] = field(default_factory=list)
    overall_result: str = ""    # Worst result across all clauses in this section
    total_pass: int = 0
    total_fail: int = 0
    total_review: int = 0


@dataclass
class ConsolidatedOutput:
    """
    Master verification dataset -- the final structured output of the compliance check.

    This is the top-level container consumed by Step 20 (PDF report generation)
    and the web UI's compliance view.
    """
    sections: List[SectionGroup] = field(default_factory=list)
    overall_decision: str = ""  # "COMPLIANT" | "DISCREPANT" | "REVIEW REQUIRED"
    total_clauses: int = 0
    total_rows: int = 0
    total_pass: int = 0
    total_fail: int = 0
    total_review: int = 0
    critical_findings: List[Dict] = field(default_factory=list)  # All FAIL rows
    review_items: List[Dict] = field(default_factory=list)        # All REVIEW rows


# ── Section Classification ───────────────────────────────────────────────────
# Maps clause F-tag prefixes to report sections.

# Key Terms tags -- LC metadata and standalone conditions (amounts, dates, parties, ports)
# Include both F-prefixed and non-F-prefixed variants since clause refs may come either way
_KEY_TERMS_TAGS = {
    'F20', 'F31C', 'F31D', 'F32B', 'F39A', 'F42C', 'F43P', 'F43T',
    'F44E', 'F44F', 'F44C', 'F44A', 'F44B', 'F44D',
    'F40A', 'F40E', 'F41A', 'F41D', 'F42A', 'F42M', 'F42P',
    'F48', 'F49', 'F50', 'F51A', 'F52A', 'F53A', 'F57A', 'F59',
    'F71B', 'F71D', 'F77B',
    # Non-F variants
    '20', '31C', '31D', '32B', '39A', '42C', '43P', '43T',
    '44E', '44F', '44C', '44A', '44B', '44D',
    '40A', '40E', '41A', '41D', '42A', '42M', '42P',
    '48', '49', '50', '51A', '52A', '53A', '57A', '59',
    '71B', '71D', '77B',
}

# Ordered section classification -- checked in order, first match wins
_SECTION_MAP = OrderedDict([
    ('Key Terms',             lambda ref: _get_tag(ref) in _KEY_TERMS_TAGS),
    ('Document Requirements', lambda ref: _get_tag(ref) in ('F46A', 'F46B', '46A', '46B')),
    ('Additional Conditions', lambda ref: _get_tag(ref) in ('F47A', '47A')),
    ('Description of Goods',  lambda ref: _get_tag(ref) in ('F45A', 'F45B', '45A', '45B')),
    ('Instructions',          lambda ref: _get_tag(ref) in ('F78', 'F72', 'F79', 'F77A', '78', '72', '79', '77A')),
])


def _get_tag(clause_ref: str) -> str:
    """Extract tag prefix from clause_ref. Example: 'F46A-2' -> 'F46A', '46A-2' -> '46A'."""
    return clause_ref.split('-')[0].upper()


def _classify_section(clause_ref: str) -> str:
    """Determine which report section a clause belongs to."""
    for section_name, test_fn in _SECTION_MAP.items():
        if test_fn(clause_ref):
            return section_name
    return 'Other'


def _compute_overall(pass_c: int, fail_c: int, review_c: int) -> str:
    """
    Compute overall result from pass/fail/review counts.

    Priority: FAIL > REVIEW > PASS
    Any failure means NOT COMPLIED; reviews without failures need human attention.
    """
    if fail_c > 0:
        return 'NOT COMPLIED'
    if review_c > 0:
        return 'REVIEW REQUIRED'
    return 'COMPLIED'


def _sort_clause_ref(ref: str) -> tuple:
    """
    Sort key for clause refs: F20 < F31C < F46A-1 < F46A-2 < F46A-10.

    Sorts first by F-tag prefix alphabetically, then by suffix number.
    This ensures clauses appear in the expected report order.
    """
    tag = _get_tag(ref)
    suffix = ref.split('-', 1)[1] if '-' in ref else '0'
    try:
        suffix_num = int(suffix)
    except ValueError:
        suffix_num = 999
    return (tag, suffix_num)


# ── Consolidation Logic ─────────────────────────────────────────────────────

def _consolidate(rows: List[Dict], progress_fn=None) -> ConsolidatedOutput:
    """
    Merge flat verification rows into consolidated clause-grouped structure.

    Algorithm:
    1. Group rows by clause_ref (e.g., all rows for F46A-1 together)
    2. Build ClauseGroup for each unique clause_ref with pass/fail/review counts
    3. Classify each ClauseGroup into a report section
    4. Compute section-level and overall statistics
    5. Collect critical findings (FAILs) and review items
    """
    if progress_fn is None:
        def progress_fn(msg): pass

    # Step 1: Group rows by clause_ref
    clause_map: Dict[str, List[Dict]] = {}
    clause_text_map: Dict[str, str] = {}  # Store first clause_text seen for each ref

    for row in rows:
        ref = row.get('clause_ref', 'UNKNOWN')
        clause_map.setdefault(ref, []).append(row)
        if not clause_text_map.get(ref):
            clause_text_map[ref] = row.get('clause_text', '')

    progress_fn(f"Grouped into {len(clause_map)} clause refs")

    # Step 2: Build ClauseGroups
    clause_groups: Dict[str, ClauseGroup] = {}
    for ref in sorted(clause_map.keys(), key=_sort_clause_ref):
        group_rows = clause_map[ref]
        vrows = []
        pc = fc = rc = 0

        for r in group_rows:
            result = r.get('result', 'REVIEW').upper()
            # Map result to compliance label
            if result == 'PASS':
                pc += 1
                compliance = 'COMPLIED'
            elif result == 'FAIL':
                fc += 1
                compliance = 'NOT COMPLIED'
            else:
                rc += 1
                compliance = 'REVIEW REQUIRED'

            vrows.append(VerificationRow(
                condition=r.get('condition', '') or r.get('condition_text', '') or r.get('result', ''),
                findings=r.get('findings', '') or r.get('found_text', ''),
                document_checked=r.get('document_checked', ''),
                result=result,
                compliance=compliance,
                dependency_notes=r.get('dependency_notes', []),
                reconciled=r.get('reconciled', False),
            ))

        cg = ClauseGroup(
            clause_ref=ref,
            clause_text=clause_text_map.get(ref, ''),
            rows=vrows,
            overall_result=_compute_overall(pc, fc, rc),
            row_count=len(vrows),
            pass_count=pc,
            fail_count=fc,
            review_count=rc,
        )
        clause_groups[ref] = cg

    # Step 3: Classify into report sections
    section_clauses: Dict[str, List[ClauseGroup]] = {}
    for ref, cg in clause_groups.items():
        section_name = _classify_section(ref)
        section_clauses.setdefault(section_name, []).append(cg)

    # Step 4: Build SectionGroups in defined order
    section_order = list(_SECTION_MAP.keys()) + ['Other']
    sections = []
    for idx, sec_name in enumerate(section_order):
        if sec_name not in section_clauses:
            continue
        clauses = section_clauses[sec_name]
        sp = sum(c.pass_count for c in clauses)
        sf = sum(c.fail_count for c in clauses)
        sr = sum(c.review_count for c in clauses)
        sections.append(SectionGroup(
            section_name=sec_name,
            section_order=idx,
            clauses=clauses,
            overall_result=_compute_overall(sp, sf, sr),
            total_pass=sp,
            total_fail=sf,
            total_review=sr,
        ))

    # Compute grand totals
    tp = sum(s.total_pass for s in sections)
    tf = sum(s.total_fail for s in sections)
    tr = sum(s.total_review for s in sections)

    # Overall decision: any FAIL → DISCREPANT; reviews only → REVIEW REQUIRED; all pass → COMPLIANT
    if tf > 0:
        decision = 'DISCREPANT'
    elif tr > 0:
        decision = 'REVIEW REQUIRED'
    else:
        decision = 'COMPLIANT'

    # Step 5: Collect critical findings (FAILs) and review items for the executive summary
    critical = []
    review_items = []
    for sec in sections:
        for cg in sec.clauses:
            for vr in cg.rows:
                entry = {
                    'clause_ref': cg.clause_ref,
                    'clause_text': cg.clause_text[:200],
                    'condition': vr.condition,
                    'findings': vr.findings,
                    'document_checked': vr.document_checked,
                }
                if vr.result == 'FAIL':
                    critical.append(entry)
                elif vr.result == 'REVIEW':
                    review_items.append(entry)

    # Sort findings by clause_ref for consistent ordering in the report
    critical.sort(key=lambda x: _sort_clause_ref(x['clause_ref']))
    review_items.sort(key=lambda x: _sort_clause_ref(x['clause_ref']))

    output = ConsolidatedOutput(
        sections=sections,
        overall_decision=decision,
        total_clauses=len(clause_groups),
        total_rows=sum(s.total_pass + s.total_fail + s.total_review for s in sections),
        total_pass=tp,
        total_fail=tf,
        total_review=tr,
        critical_findings=critical,
        review_items=review_items,
    )

    progress_fn(
        f"Consolidated: {output.total_clauses} clauses, {output.total_rows} rows -- "
        f"Decision: {decision} ({tp}P / {tf}F / {tr}R)"
    )

    return output


# ── Runner ───────────────────────────────────────────────────────────────────

def run(
    reconciled_rows: List[Dict],
    output_dir: str,
    progress_fn=None,
) -> Dict[str, Any]:
    """
    Execute Step 19: Consolidated Clause Verification Output.

    Takes the flat list of reconciled rows from Step 17 and organizes them
    into the hierarchical structure needed by the PDF report (Step 20).

    Args:
        reconciled_rows: list of reconciled row dicts from Step 17
        output_dir:      directory for step output
        progress_fn:     callback for progress messages

    Returns:
        dict with consolidated verification dataset
    """
    t0 = time.time()
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if progress_fn is None:
        def progress_fn(msg): pass

    progress_fn("Step 19: Consolidated Clause Verification Output")
    progress_fn(f"Processing {len(reconciled_rows)} reconciled rows")

    consolidated = _consolidate(reconciled_rows, progress_fn)

    result = {
        'step': 19,
        'step_name': 'Consolidated Clause Verification Output',
        'overall_decision': consolidated.overall_decision,
        'total_clauses': consolidated.total_clauses,
        'total_rows': consolidated.total_rows,
        'total_pass': consolidated.total_pass,
        'total_fail': consolidated.total_fail,
        'total_review': consolidated.total_review,
        'critical_findings': consolidated.critical_findings,
        'review_items': consolidated.review_items,
        'sections': [asdict(s) for s in consolidated.sections],
        'elapsed_seconds': round(time.time() - t0, 2),
    }

    # Save result to disk
    out_path = Path(output_dir) / 'step19_result.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    progress_fn(
        f"Step 19 complete: {consolidated.overall_decision} -- "
        f"{consolidated.total_clauses} clauses, "
        f"{consolidated.total_pass}P / {consolidated.total_fail}F / {consolidated.total_review}R"
    )

    return result
