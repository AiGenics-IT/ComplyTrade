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
import re
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

def _expand_all_documents_row(row: Dict) -> List[Dict]:
    """P198fc — Expand a single 'All Documents' aggregation row into one
    row per document type.

    Step 14 fans out a universal-quantifier clause ("L/C number must
    appear on all documents") to every doc-class in the submission and
    aggregates the per-doc results into ONE row whose findings carry a
    pipe-separated 'Per-doc: <DocType>: PASS — ... | <DocType>: FAIL — ...'
    block. The combined row was hard to read in the report — the user
    wants one row per document so each doc's PASS/FAIL/REVIEW is
    visible at a glance.

    Returns:
      [row]              if the row is not a Per-doc aggregation
      [row1, row2, ...]  one row per doc-class otherwise (the original
                          row is replaced; aggregation summary becomes
                          a synthetic "All Documents" header row at the
                          top of the group so the overall verdict still
                          shows in the report)

    Each child row inherits the parent's clause_ref, condition, and
    other fields. The findings / result / compliance / document_checked
    are taken from the per-doc segment.
    """
    if not isinstance(row, dict):
        return [row]
    doc_field = str(row.get('document_checked', '') or '').lower().strip()
    findings  = str(row.get('findings', '') or row.get('found_text', '') or '')
    if 'all documents' not in doc_field and 'all docs' not in doc_field:
        return [row]
    if 'per-doc:' not in findings.lower():
        return [row]

    # Locate the Per-doc block — keep everything BEFORE it as the
    # parent's summary findings; everything AFTER is the per-doc list.
    idx = findings.lower().find('per-doc:')
    summary_prefix = findings[:idx].rstrip(' .').strip()
    pd_block = findings[idx + len('per-doc:'):].strip()

    # Each segment is "<DocType>: <PASS/FAIL/REVIEW> — <text>"
    # separated by " | ". The text itself may contain "—" / "|" though,
    # so split conservatively on " | " and parse each segment.
    segments = [s.strip() for s in pd_block.split(' | ') if s.strip()]
    if not segments:
        return [row]

    _verdict_re = re.compile(
        r'^(?P<doc>[^:]+):\s*(?P<verdict>PASS|FAIL|REVIEW|N/?A|INFORMATIONAL)\b'
        r'(?:\s*[—\-:]\s*(?P<text>.*))?$',
        re.IGNORECASE,
    )

    children = []
    parent_keys_to_copy = (
        'clause_ref', 'clause_text', 'original_clause_text',
        'condition', 'condition_text', 'rule_id', 'lc_field',
        'severity', 'reconciled', 'dependency_notes',
        'verification_notes',
    )
    for seg in segments:
        m = _verdict_re.match(seg)
        if not m:
            # Couldn't parse this segment — skip rather than mangle
            continue
        doc_type = m.group('doc').strip()
        verdict  = (m.group('verdict') or '').upper()
        text     = (m.group('text') or '').strip()
        # Normalise N/A
        if verdict in ('N/A', 'NA'):
            verdict = 'N/A'
        child = {k: row.get(k) for k in parent_keys_to_copy if k in row}
        child['document_checked'] = doc_type
        child['compliance']       = verdict
        child['findings']         = text or seg
        child['found_text']       = text or seg
        child['result']           = text or seg
        child['_split_from_all_documents'] = True
        children.append(child)

    if not children:
        return [row]

    # Always keep the parent summary row at the top so the overall
    # verdict (which may be different from the individual children
    # when the universal quantifier weights them differently) stays
    # visible. Trim the parent's findings to only the summary prefix
    # (everything before "Per-doc:") so it doesn't duplicate.
    parent_summary = dict(row)
    parent_summary['findings']   = summary_prefix or row.get('result', '')
    parent_summary['found_text'] = parent_summary['findings']
    parent_summary['_all_documents_summary'] = True
    return [parent_summary] + children


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

    # P198fc — Expand "All Documents" aggregation rows into one row
    # per document. The fan-out happens BEFORE clause grouping so
    # each child row gets counted properly in pass/fail/review tallies.
    _expanded_rows: List[Dict] = []
    _expansion_count = 0
    for _row in rows:
        _children = _expand_all_documents_row(_row)
        if len(_children) > 1:
            _expansion_count += len(_children) - 1
        _expanded_rows.extend(_children)
    if _expansion_count:
        progress_fn(f"P198fc: expanded {_expansion_count} per-doc child rows from 'All Documents' aggregations")
    rows = _expanded_rows

    # Step 1: Group rows by clause_ref
    clause_map: Dict[str, List[Dict]] = {}
    clause_text_map: Dict[str, str] = {}  # Store first clause_text seen for each ref

    for row in rows:
        ref = row.get('clause_ref', 'UNKNOWN')
        clause_map.setdefault(ref, []).append(row)
        if not clause_text_map.get(ref):
            clause_text_map[ref] = (row.get('original_clause_text', '')
                                    or row.get('clause_text', ''))

    progress_fn(f"Grouped into {len(clause_map)} clause refs")

    # P152 — Root-cause fix lives in step 14 (_deterministic_verify and
    # the CORE verification prompt), NOT in step 19. Removed the
    # consolidation-layer overrides (P145, P149, P151) because they were
    # either duplicating logic that belongs in step 14 or relying on the
    # LLM's own finding text as evidence — which is circular when the
    # finding itself mentions the target in a negation phrase.
    #
    # The step-14 deterministic fast-path reads the actual document
    # text and structured facts directly, so it's the only place where
    # the check can be authoritative.

    # P152 REFACTOR — All consolidation-layer overrides (P145, P149,
    # P151) have been removed. They relied on the LLM's finding text as
    # evidence which is CIRCULAR: the finding often negates by quoting
    # the target ("does NOT show BANK AL HABIB") so "BANK AL HABIB"
    # appears in the row's own data even when the document actually
    # lacks it. That's a false-positive risk the user explicitly
    # flagged.
    #
    # The authoritative fix is in step 14 where _deterministic_verify
    # has direct access to the packet's document_text, unified_summary,
    # and bl_subtype. That's the only place the check can trust its
    # evidence.

    # Step 2: Build ClauseGroups
    clause_groups: Dict[str, ClauseGroup] = {}
    for ref in sorted(clause_map.keys(), key=_sort_clause_ref):
        group_rows = clause_map[ref]

        # P131 — Dedupe rows that share the same condition_text within one
        # clause. When step 14 runs the same condition against multiple BL
        # copies (3 originals + 1 copy), each returns its own verdict. If one
        # reads the port clearly (PASS) while another's OCR is fuzzier
        # (REVIEW), the report would show the same row twice with different
        # verdicts — confusing the user. Collapse to the best verdict:
        # PASS > REVIEW > FAIL (any PASS proves compliance; a REVIEW beside
        # a PASS is just an OCR artefact of a copy).
        def _cond_key(r):
            c = (r.get('condition') or r.get('condition_text') or '').strip().lower()
            d = (r.get('document_checked') or '').strip().lower()
            return (c, d)

        def _verdict_rank(r):
            vc = (r.get('compliance') or '').upper().strip()
            if vc in ('PASS', 'COMPLIED', 'COMPLIANT'):
                return 3
            if vc in ('FAIL', 'NOT COMPLIED', 'NON_COMPLIANT', 'DISCREPANT'):
                return 1
            if vc in ('REVIEW', 'REVIEW REQUIRED', 'REVIEW_REQUIRED'):
                return 2
            if vc in ('N/A', 'NA', 'INFORMATIONAL', 'INFO', 'SKIPPED'):
                return 0
            return 2  # unknown → treat as REVIEW

        _by_key = {}
        for _r in group_rows:
            k = _cond_key(_r)
            if not k[0]:  # empty condition → keep as-is (no dedup)
                _by_key.setdefault(('__nokey__', id(_r)), _r)
                continue
            cur = _by_key.get(k)
            if cur is None or _verdict_rank(_r) > _verdict_rank(cur):
                _by_key[k] = _r
        group_rows = list(_by_key.values())

        vrows = []
        pc = fc = rc = 0

        for r in group_rows:
            result_text = r.get('result', 'REVIEW').upper()
            # Check compliance field FIRST (Step 14 puts verdict here),
            # then fall back to result field
            raw_compliance = (r.get('compliance', '') or '').upper().strip()
            if raw_compliance in ('PASS', 'COMPLIED', 'COMPLIANT'):
                pc += 1
                compliance = 'COMPLIED'
            elif raw_compliance in ('FAIL', 'NOT COMPLIED', 'NON_COMPLIANT', 'DISCREPANT'):
                fc += 1
                compliance = 'NOT COMPLIED'
            elif raw_compliance in ('REVIEW', 'REVIEW REQUIRED', 'REVIEW_REQUIRED'):
                rc += 1
                compliance = 'REVIEW REQUIRED'
            elif result_text in ('PASS', 'COMPLIED'):
                pc += 1
                compliance = 'COMPLIED'
            elif result_text in ('FAIL', 'NOT COMPLIED'):
                fc += 1
                compliance = 'NOT COMPLIED'
            elif raw_compliance in ('N/A', 'NA', 'INFORMATIONAL', 'INFO', 'SKIPPED'):
                compliance = 'N/A'
                # Don't count N/A rows in pass/fail/review
            else:
                rc += 1
                compliance = 'REVIEW REQUIRED'

            # condition = the LC requirement text (what we're checking)
            # result = the VLM's short summary of what it found
            _condition = r.get('condition', '') or r.get('condition_text', '')
            _result = result_text
            # If condition is empty, use the clause_text (the original LC clause)
            if not _condition:
                _condition = r.get('clause_text', '') or clause_text_map.get(ref, '')
            # If condition still empty but result has text, keep result as result only
            # DON'T copy result into condition — they must be different
            vrows.append(VerificationRow(
                condition=_condition,
                findings=r.get('findings', '') or r.get('found_text', ''),
                document_checked=r.get('document_checked', ''),
                result=_result,
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

    # Step 4: Build SectionGroups in LC field sequence order
    # Key Terms first (LC header fields), then documents, conditions, goods
    section_order = ['Key Terms', 'Document Requirements', 'Additional Conditions',
                     'Description of Goods', 'Instructions', 'Other']
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
    # Deduplicate: same clause_ref + document_checked + result = show once
    critical = []
    review_items = []
    _seen_critical = set()
    _seen_review = set()
    for sec in sections:
        for cg in sec.clauses:
            for vr in cg.rows:
                # Skip purely informational / N/A rows
                if vr.compliance in ('N/A', 'INFORMATIONAL'):
                    continue
                if vr.result == 'INFORMATIONAL' or (vr.condition or '').strip().upper() == 'INFORMATIONAL':
                    continue
                if not vr.condition and not vr.findings and not vr.document_checked:
                    continue
                # P198dg — store the FULL clause/condition/findings
                # text in the consolidated entries (no truncation).
                # The report layer wraps long text in cells.
                entry = {
                    'clause_ref': cg.clause_ref,
                    'clause_text': cg.clause_text,
                    'condition': vr.condition,
                    'findings': vr.findings,
                    'result': vr.result,
                    'document_checked': vr.document_checked,
                }
                # Dedup key: clause + document + result (prevents "MISSING" x3)
                _dedup_key = f"{cg.clause_ref}|{vr.document_checked}|{vr.result}"
                if vr.compliance == 'NOT COMPLIED':
                    if _dedup_key not in _seen_critical:
                        _seen_critical.add(_dedup_key)
                        critical.append(entry)
                elif vr.compliance == 'REVIEW REQUIRED':
                    if _dedup_key not in _seen_review:
                        _seen_review.add(_dedup_key)
                        review_items.append(entry)

            # For clauses with NO real rows but review_count > 0, add clause-level review entry
            if cg.review_count > 0 and not any(
                vr.compliance == 'REVIEW REQUIRED' and vr.result != 'INFORMATIONAL'
                and (vr.condition or '').strip().upper() != 'INFORMATIONAL'
                for vr in cg.rows
            ):
                # This clause was not decomposed — add as review with clause text
                _clause_text = cg.clause_text or clause_text_map.get(cg.clause_ref, '')
                if _clause_text:
                    _dedup_key = f"{cg.clause_ref}||NOT_DECOMPOSED"
                    if _dedup_key not in _seen_review:
                        _seen_review.add(_dedup_key)
                        review_items.append({
                            'clause_ref': cg.clause_ref,
                            'clause_text': _clause_text,
                            'condition': _clause_text,
                            'findings': 'Clause not decomposed — requires manual review',
                            'result': 'MANUAL REVIEW NEEDED',
                            'document_checked': 'All Documents',
                        })

    # Sort findings by clause_ref for consistent ordering in the report
    critical.sort(key=lambda x: _sort_clause_ref(x['clause_ref']))
    review_items.sort(key=lambda x: _sort_clause_ref(x['clause_ref']))

    # Use deduplicated counts for summary (matches what user sees in executive summary)
    _dedup_fail = len(critical)
    _dedup_review = len(review_items)

    output = ConsolidatedOutput(
        sections=sections,
        overall_decision=decision,
        total_clauses=len(clause_groups),
        total_rows=sum(s.total_pass + s.total_fail + s.total_review for s in sections),
        total_pass=tp,
        total_fail=_dedup_fail,
        total_review=_dedup_review,
        critical_findings=critical,
        review_items=review_items,
    )

    progress_fn(
        f"Consolidated: {output.total_clauses} clauses, {output.total_rows} rows -- "
        f"Decision: {decision} ({tp}P / {tf}F / {tr}R)"
    )

    return output


# ── Runner ───────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────
# P198gd / P198ge — Partial-shipment per-invoice document completeness.
#
# When the LC has F43P=ALLOWED and the bundle contains MULTIPLE
# invoices each representing a separate partial shipment, every
# invoice must be accompanied by its OWN full set of LC-required
# per-invoice documents.
#
# P198ge replaces the original P198gd hardcoded approach with a
# production-grade implementation that:
#   1. Extracts each Commercial Invoice's number using a multi-tier
#      strategy: step03 instrument_references first, then label-based
#      regex ("Invoice No." / "Inv No." / "Invoice Number"), then
#      free-form pattern matching across many real-world formats
#      (Sxxxxxxxxx, MPL/013/INDO/2026, NSAM-2603-B12, INV00190678,
#      etc.). This avoids hardcoding to a single bank's invoice
#      style.
#   2. Reads the LC's actual F46A required-doc list from
#      step07 structured_lc.required_documents and intersects it
#      with a "per-invoice" whitelist — so an LC that requires
#      Cert of Origin + Inspection Cert + Weight Cert (typical for
#      coal) checks each invoice for THOSE docs, while an LC that
#      requires Beneficiary Cert + AWB checks each invoice for those.
#      Filters out F47A rule-style hallucinated entries (using the
#      same patterns as P198fw's step07 filter) and LC-level docs
#      that are not per-invoice (Document Remittance / Charges Cert
#      / Translation Cert / etc.).
#   3. Adds an F43P guard: only fires the doc-completeness section
#      when F43P=ALLOWED. When F43P=NOT ALLOWED but the bundle has
#      ≥2 distinct invoices, that itself is a discrepancy and the
#      section reports "Multiple invoices presented but partial
#      shipments NOT ALLOWED per LC F43P".
#   4. Maps unmatched transport docs (BL/AWB without an invoice
#      ref in their text) to invoices via instrument_references
#      grouping — same physical BL number = same invoice — then
#      page-proximity heuristic for the orphan groups.
# ─────────────────────────────────────────────────────────────────────
import re as _re_p198gd

# Free-form invoice-number patterns. We try these in order until one
# matches. The patterns intentionally allow a wide variety of formats.
_P198GE_INVOICE_PATTERNS = [
    # Compound prefix with letters/digits/slashes/dashes:
    # MPL/013/INDO/2026, NSAM-2603-B12, MCI-786/S-13198,
    # APPK24022BIL-1, MC/0006/25, RE/017/2025, CI-321/2025
    _re_p198gd.compile(
        r'\b[A-Z]{2,6}[\-/_]?(?:\d{1,5}[\-/_])+(?:[A-Z]{1,6}[\-/_]?)?\d{1,6}'
        r'(?:[\-/_][A-Z]?\d{1,6})?\b'
    ),
    # Letter-prefix simple: S26030280, INV00190678, SC553851
    _re_p198gd.compile(r'\b[A-Z]{1,4}\d{6,12}\b'),
    # Numeric-with-slash: 03324/03/2026, 02630/02/2026
    _re_p198gd.compile(r'\b\d{4,6}/\d{2}/\d{4}\b'),
]

# Canonical → aliases. Used to:
#   (a) normalise step09 packet document_types to a canonical form
#   (b) match LC F46A required_document.document_name to the same
#       canonical form so we know what the LC asked for.
_P198GE_DOC_CANONICAL = (
    ('Commercial Invoice',
     ('commercial invoice', 'invoice')),
    ('Packing List',
     ('packing list', 'weight list', 'weight & packing list', 'cargo manifest')),
    ('Bill of Lading',
     ('bill of lading', 'bills of lading', 'b/l', 'bl',
      'ocean bill of lading', 'ocean bills of lading',
      'marine bill of lading', 'marine bills of lading')),
    ('Airway Bill',
     ('airway bill', 'air waybill', 'awb', 'air waybill')),
    ('Shipment Advice',
     ('shipment advice', 'advice of shipment', 'shipping advice')),
    ('Beneficiary Certificate',
     ("beneficiary certificate", "beneficiary cert",
      "beneficiary's declaration", "beneficiary declaration",
      "beneficiary's certificate")),
    ('Certificate of Origin',
     ('certificate of origin', 'cert of origin', 'origin certificate')),
    ('Inspection Certificate',
     ('inspection certificate', 'pre-shipment inspection',
      'sampling and analysis', 'certificate of analysis',
      'cert of analysis', 'analysis certificate', 'sampling certificate',
      'quality certificate')),
    ('Weight Certificate',
     ('weight certificate', 'cert of weight', 'certificate of weight',
      'quantity certificate')),
    # Order matters: 'Health Certificate' MUST be checked before
    # 'Phytosanitary Certificate' because the alias 'sanitary
    # certificate' is a substring of 'phytosanitary certificate' —
    # matching the wrong canonical otherwise.
    ('Health Certificate',
     ('health certificate', 'health cert',
      'sanitary certificate', 'sanitary cert',
      'halal certificate', 'halal cert',
      'kosher certificate', 'kosher cert')),
    ('Phytosanitary Certificate',
     ('phytosanitary certificate', 'phyto certificate',
      'phyto cert', 'phytosanitary')),
    ('Fumigation Certificate',
     ('fumigation certificate', 'fumigation cert', 'fumigation')),
    ('Insurance Certificate',
     ('insurance certificate', 'insurance policy', 'cover note')),
    ('Shipping Company Certificate',
     ('shipping company certificate', "shipping company's certificate",
      'carrier certificate', "carrier's certificate")),
)

# Per-invoice whitelist: documents that are typically PRESENTED ONCE
# PER INVOICE under partial shipments. LC-level documents (presented
# once for the whole presentation, not per invoice) are excluded:
#   • Document Remittance / Covering Schedule   (1 cover per bundle)
#   • Draft Bill of Exchange                    (varies — often 1 per
#                                                bundle, depends on LC)
#   • Charges Certificate / Discrepancy Notice  (1 statement)
#   • Translation Certificate / Language Cert   (1 global statement)
#   • Sanctions Compliance Certificate          (1 global statement)
#   • Confirmation Advice / Document Arrival    (1 LC-level)
_P198GE_PER_INVOICE_CANONICALS = frozenset({
    'Commercial Invoice',
    'Packing List',
    'Bill of Lading',
    'Airway Bill',
    'Shipment Advice',
    'Beneficiary Certificate',
    'Certificate of Origin',
    'Inspection Certificate',
    'Weight Certificate',
    'Phytosanitary Certificate',
    'Fumigation Certificate',
    'Health Certificate',
    'Insurance Certificate',
    'Shipping Company Certificate',
})

# F47A "rule-style" patterns reused from step07 (P198fw). When an LC's
# required_documents entry was extracted from an F47A clause matching
# any of these patterns, it is a phantom (e.g. "Draft Bill of Exchange"
# from "THIRD PARTY DOCUMENTS ARE ACCEPTABLE EXCEPT INVOICE AND DRAFT")
# and must NOT be added to the per-invoice required set.
_P198GE_F47A_RULE_PATTERNS = (
    r'\bARE\s+ACCEPTABLE\b', r'\bIS\s+ACCEPTABLE\b',
    r'\bACCEPTABLE\.\s*$', r'\bNOT\s+ACCEPTABLE\b',
    r'\bDATED\s+PRIOR\s+TO\b',
    r'\bTHIRD\s+PARTY\s+DOCUMENTS\b',
    r'\bDISCREPANC(?:Y|IES)\s+(?:CHARGES|FEE|FEES)\b',
    r'\bCHARGES\s+WILL\s+BE\s+DEDUCTED\b',
    r'\bFEE\s+(?:OF|WILL|MUST|SHALL)\b',
    r'\bPRICE\s+ADJUSTMENT\b',
    r'\bGROSS\s+CALORIFIC\s+VALUE\b',
    r'\bIF\s+THE\s+ACTUAL\b',
    r'\bWITHIN\s+\d+\s+DAYS\s+OF\s+SHIPMENT\b',
    r'\bDOCUMENTS\s+MUST\s+BE\s+SENT\s+TO\b',
    r'\bENDORSED\s+ON\s+THE\s+REVERSE\b',
    r'\bDRAWN\s+AND\s+NEGOTIATED\b',
    r'\bOVERWRITING\b', r'\bALTERATION\b',
    r'\bQUANTITY\s+DIFFERENT\s+FROM\b',
    r'\bSHOWING\s+QUANTITY\s+DIFFERENT\b',
)


def _p198ge_normalize_invoice(inv):
    """Clean an extracted invoice number of OCR-introduced noise.

    Real-data anchors:
      • 'PI2504022DATEDAPR' (53e62015) → 'PI2504022'   (OCR merged the
        " DATED APR..." annotation with the invoice number)
      • 'XPK-TR26030303' (11ec29b8)    → 'XPK-TR260303' (OCR injected
        an extra digit in one packet's reading of the same number)
    """
    if not inv:
        return inv
    s = str(inv).strip(' :.,-')
    # Strip " DATED ..." / "  DATED..." suffix that OCR may have
    # collapsed into the invoice token (with or without surrounding
    # whitespace).
    s = _re_p198gd.sub(r'\s+DATED?\s.*$', '', s, flags=_re_p198gd.IGNORECASE)
    s = _re_p198gd.sub(r'(?i)DATED?[A-Z0-9_/\-\.]*$', '', s)
    # Strip trailing 3-letter month abbreviations + trailing digits
    # (e.g. "...APR2026", "...JAN26", " APR 2026").
    s = _re_p198gd.sub(
        r'(?i)\s*(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s*\d{0,4}\s*$',
        '', s)
    return s.strip(' :.,-') or inv


def _p198ge_dedup_near_duplicates(invoices, packet_counts):
    """Collapse near-duplicate invoice numbers caused by OCR noise.

    `invoices` is a list of distinct invoice strings. `packet_counts`
    is a dict invoice -> count of packets carrying it. We use the
    counts to pick a canonical representative when collapsing.

    Rule (conservative, single-pattern):
      • If invoice A is a strict prefix of B AND B's extra suffix
        looks like a date (digits / month abbrevs) OR is just 1
        character of OCR noise, merge B into A (the shorter, common
        form). We deliberately do NOT do general fuzzy merging
        because real invoice numbers in multi-shipment LCs often
        differ by only one or two characters (e.g. S26030326 vs
        S26030328) and merging them silently would HIDE missing-doc
        discrepancies.
    """
    invoices = sorted(invoices)
    merged = {inv: inv for inv in invoices}   # original → canonical
    for i, a in enumerate(invoices):
        for b in invoices[i + 1:]:
            if merged[a] == merged[b]:
                continue
            ca, cb = merged[a], merged[b]
            if ca == cb:
                continue
            # Strict-prefix pattern, both directions
            for short, long_ in ((ca, cb), (cb, ca)):
                if long_.startswith(short) and len(long_) - len(short) <= 8:
                    tail = long_[len(short):]
                    # Accept tail forms commonly attached by OCR or
                    # bank-internal annotations:
                    #   • digits only (date-like)
                    #   • month-name abbrev with optional digits
                    #   • single delimiter + 1-2 chars (e.g. '-X', '-A1')
                    #   • bare 1-char OCR noise
                    if (_re_p198gd.fullmatch(r'[\-/_]?\d+', tail)
                        or _re_p198gd.fullmatch(r'[\-/_]?(?i:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\d*', tail)
                        or _re_p198gd.fullmatch(r'[\-/_][A-Z]\d{0,2}', tail)
                        or len(tail) <= 1):
                        # Merge long_ into short
                        if long_ == ca:
                            merged[a] = short
                        else:
                            merged[b] = short
                        break
    # Resolve transitive merges
    out_canon = {}
    for inv, c in merged.items():
        seen = set()
        while c != merged.get(c, c) and c not in seen:
            seen.add(c)
            c = merged.get(c, c)
        out_canon[inv] = c
    return out_canon


def _p198ge_canonicalize_doc(name):
    """Map a free-form document name to a canonical type. Returns None
    if no canonical match is found.

    Uses WORD-BOUNDARY matching to avoid spurious sub-string matches
    (the original 'in' check matched 'sanitary certificate' inside
    'phytosanitary certificate' and bound the wrong canonical).
    Tries both directions so short inputs like 'AWB' still match.
    """
    if not name:
        return None
    name_lo = str(name).lower().strip()
    if not name_lo:
        return None
    for canon, aliases in _P198GE_DOC_CANONICAL:
        for a in aliases:
            # Exact match
            if a == name_lo:
                return canon
            # Alias appears as a whole word in the input (most cases)
            if _re_p198gd.search(rf'\b{_re_p198gd.escape(a)}\b', name_lo):
                return canon
            # Input is a whole word inside the alias (short forms like
            # 'AWB' against alias 'awb', or 'B/L' against 'b/l').
            if _re_p198gd.search(rf'\b{_re_p198gd.escape(name_lo)}\b', a):
                return canon
    return None


def _p198ge_extract_invoice_number(pkt):
    """
    Extract this packet's invoice number using a tiered strategy:

      1. step03 instrument_references on any of its pages (the VLM's
         visual extraction — most reliable).
      2. Label-based regex over the packet text: "INVOICE NO." /
         "INV NO." / "INVOICE NUMBER" / "REF NO." / "REFERENCE NO.".
      3. Free-form pattern match over the first 2500 chars of text.

    Returns the FIRST invoice ref found, or None.
    """
    # 1. instrument_references (most reliable)
    for op in pkt.get('original_pages', []) or []:
        if isinstance(op, dict):
            for r in (op.get('instrument_references') or []):
                if r and len(str(r)) >= 3:
                    return str(r).strip()

    txt = (pkt.get('refined_text') or pkt.get('cleaned_text')
           or pkt.get('text') or pkt.get('raw_text') or '')
    if not txt:
        return None
    head = txt[:3000]

    # 2. Label-based regex
    label_patterns = (
        r'INVOICE\s*(?:NO\.?|NUMBER|#)\s*[:\-]?\s*([A-Z0-9][A-Z0-9_/\-\.]{2,30})',
        r'INV(?:\.|OICE)?\s*(?:NO\.?|NUMBER|#)\s*[:\-]?\s*([A-Z0-9][A-Z0-9_/\-\.]{2,30})',
        r'COMM[A-Z]*\s+INVOICE\s+NO\.?\s*[:\-]?\s*([A-Z0-9][A-Z0-9_/\-\.]{2,30})',
        r'(?:REFERENCE|REF)\s*(?:NO\.?|NUMBER|#)\s*[:\-]?\s*([A-Z0-9][A-Z0-9_/\-\.]{2,30})',
    )
    for pat in label_patterns:
        m = _re_p198gd.search(pat, head, _re_p198gd.IGNORECASE)
        if m:
            cand = m.group(1).strip(' :.,-')
            # Reject obviously bad captures (too short / all digits in
            # a section that's clearly not an invoice no — e.g. dates)
            if 3 <= len(cand) <= 40 and not _re_p198gd.match(r'^\d{1,2}$', cand):
                return cand

    # 3. Free-form pattern match (first match wins)
    for rx in _P198GE_INVOICE_PATTERNS:
        m = rx.search(head)
        if m:
            return m.group(0)
    return None


def _p198ge_required_per_invoice_set(structured_lc):
    """
    Build the set of canonical doc-types required PER PARTIAL SHIPMENT
    by THIS LC, derived from F46A required_documents.

    Filters:
      • Drops F47A "rule-style" entries (P198fw logic).
      • Drops LC-level docs not in _P198GE_PER_INVOICE_CANONICALS
        (Doc Remittance, Draft BoE, Charges Cert, etc.).
      • Canonicalises and deduplicates.
      • For BL/AWB, returns a single composite "Transport Document
        (Bill of Lading / Airway Bill)" if BOTH are required (multi-
        modal LCs); otherwise returns the specific transport doc.
    """
    required = []
    seen = set()
    bl_required = False
    awb_required = False
    for rd in (structured_lc.get('required_documents') or []):
        name = rd.get('document_name', '')
        if not name or name == 'None':
            continue
        # F47A rule filter
        src = (rd.get('source_clause_text', '') or '').upper()
        clause_ref = (rd.get('clause_reference', '') or '').upper()
        if 'F47A' in clause_ref or '47A' in clause_ref:
            if any(_re_p198gd.search(p, src) for p in _P198GE_F47A_RULE_PATTERNS):
                continue
        canon = _p198ge_canonicalize_doc(name)
        if not canon:
            continue
        if canon not in _P198GE_PER_INVOICE_CANONICALS:
            continue
        if canon == 'Bill of Lading':
            bl_required = True
            continue
        if canon == 'Airway Bill':
            awb_required = True
            continue
        if canon not in seen:
            seen.add(canon)
            required.append(canon)
    # Combine BL/AWB into a single "Transport Document" requirement.
    if bl_required and awb_required:
        required.append('Transport Document (Bill of Lading / Airway Bill)')
    elif bl_required:
        required.append('Bill of Lading')
    elif awb_required:
        required.append('Airway Bill')
    return required


_P198GE_TRANSPORT_ALIASES = (
    'bill of lading', 'airway bill', 'air waybill', 'awb',
    'b/l', 'bl', 'ocean bill of lading',
)

def _p198gd_pkt_text(p):
    return (p.get('refined_text') or p.get('cleaned_text')
            or p.get('text') or p.get('raw_text') or '')


def _p198gd_partial_shipment_check(job_dir):
    """Return a SectionGroup-style dict to insert at the top of the
    report when partial-shipment per-invoice doc checks find any
    missing documents.

    Returns None if:
      - step09 / step07 / step06 are unreadable
      - fewer than 2 distinct invoice numbers are detected on CIs
      - F43P=NOT ALLOWED with single invoice (nothing to check)
      - LC F46A required docs cannot be parsed
      - no required documents are missing for any invoice (all clean)
    """
    try:
        job_root = Path(job_dir).parent
        s9_path = job_root / 'step09' / 'step09_result.json'
        s7_path = job_root / 'step07' / 'step07_result.json'
        s6_path = job_root / 'step06' / 'step06_result.json'
        if not (s9_path.exists() and s7_path.exists()):
            return None
        with open(s9_path, 'r', encoding='utf-8') as fh:
            d9 = json.load(fh)
        with open(s7_path, 'r', encoding='utf-8') as fh:
            d7 = json.load(fh)
        packets = d9.get('reconciled_packets') or []
        structured_lc = d7.get('structured_lc') or {}
        if not packets or not structured_lc:
            return None
        # F43P value (allows / not allowed)
        f43p = ''
        if s6_path.exists():
            try:
                with open(s6_path, 'r', encoding='utf-8') as fh:
                    d6 = json.load(fh)
                cf = d6.get('consolidated_fields', {}) or {}
                f43p = (cf.get('43P', '') or cf.get('F43P', '') or '').upper()
            except Exception:
                f43p = ''
    except Exception:
        return None

    # Step 1 — extract each Commercial Invoice's invoice number.
    ci_invoice = {}     # packet_id -> invoice_no (normalized)
    ci_packets = []
    for pkt in packets:
        if not isinstance(pkt, dict):
            continue
        dt_lo = (pkt.get('document_type', '') or '').lower()
        if 'invoice' in dt_lo and 'proforma' not in dt_lo and 'tax' not in dt_lo:
            ci_packets.append(pkt)
    for pkt in ci_packets:
        inv_raw = _p198ge_extract_invoice_number(pkt)
        if inv_raw:
            inv = _p198ge_normalize_invoice(inv_raw)
            if inv:
                ci_invoice[pkt.get('packet_id', '?')] = inv

    # Collapse near-duplicate invoice numbers caused by OCR noise
    # ("PI2504022DATEDAPR" → "PI2504022", "XPK-TR26030303" merged
    # into "XPK-TR260303"). Use packet counts to pick the canonical.
    raw_distinct = sorted({v for v in ci_invoice.values() if v})
    packet_counts = {}
    for v in ci_invoice.values():
        packet_counts[v] = packet_counts.get(v, 0) + 1
    canon_map = _p198ge_dedup_near_duplicates(raw_distinct, packet_counts)
    # Apply canonical mapping
    ci_invoice = {pid: canon_map.get(v, v) for pid, v in ci_invoice.items()}
    distinct_invoices = sorted({v for v in ci_invoice.values() if v})

    # Need ≥2 distinct invoice numbers to fire. Single-invoice bundles
    # (even with multiple invoice copies/originals) are not partial-
    # shipment scenarios.
    if len(distinct_invoices) < 2:
        return None

    # F43P guard. If F43P explicitly says NOT ALLOWED but we see
    # multiple invoices, that's itself a discrepancy — flag it
    # specially. Otherwise (ALLOWED / blank / PERMITTED), fall through
    # to the normal per-invoice doc check.
    f43p_not_allowed = ('NOT ALLOWED' in f43p
                        or 'NOT PERMITTED' in f43p
                        or 'PROHIBITED' in f43p)

    # Step 2 — group all packets by invoice. For each non-CI packet,
    # find which invoice number(s) appear in its text or instrument
    # references. Multi-invoice references mean the doc applies to
    # all those invoices (e.g. a Bill of Exchange covering multiple
    # tranches).
    inv_map = {inv: {} for inv in distinct_invoices}
    unmatched_packets = []
    for pkt in packets:
        if not isinstance(pkt, dict):
            continue
        pid = pkt.get('packet_id', '?')
        dt = pkt.get('document_type', '') or ''
        # CIs go to their own invoice only
        if pid in ci_invoice:
            inv_map.setdefault(ci_invoice[pid], {}).setdefault(dt, []).append(pid)
            continue
        # Search packet text + instrument_references for each known invoice
        txt = _p198gd_pkt_text(pkt)
        head = txt[:8000].upper()
        inst_refs_here = set()
        for op in pkt.get('original_pages', []) or []:
            if isinstance(op, dict):
                for r in (op.get('instrument_references') or []):
                    if r:
                        inst_refs_here.add(str(r).upper())
        matched = []
        for inv in distinct_invoices:
            inv_u = inv.upper()
            if inv_u in head or inv_u in inst_refs_here:
                matched.append(inv)
        if matched:
            for inv in matched:
                inv_map.setdefault(inv, {}).setdefault(dt, []).append(pid)
        else:
            unmatched_packets.append(pkt)

    # Step 3 — assign unmatched transport docs (BL/AWB) to invoices
    # via instrument-reference grouping (same BL number = same
    # physical doc) + page-proximity heuristic.
    transport_unmatched = [
        p for p in unmatched_packets
        if any(k in (p.get('document_type', '') or '').lower()
               for k in _P198GE_TRANSPORT_ALIASES)
    ]
    if transport_unmatched:
        # Build invoice -> set of pages it touches (from CI packets)
        inv_pages = {}
        for inv, doc_map in inv_map.items():
            for dt, pids in doc_map.items():
                for pkt in packets:
                    if pkt.get('packet_id') in pids:
                        for op in pkt.get('original_pages', []) or []:
                            if isinstance(op, dict) and op.get('page_number'):
                                inv_pages.setdefault(inv, set()).add(op['page_number'])
        # Group transport packets by BL/AWB number
        groups = {}
        no_num_groups = []
        for pkt in transport_unmatched:
            refs = []
            for op in pkt.get('original_pages', []) or []:
                if isinstance(op, dict):
                    refs.extend(op.get('instrument_references') or [])
            refs = [r for r in refs if r and len(str(r)) >= 4]
            if refs:
                groups.setdefault(refs[0], []).append(pkt)
            else:
                no_num_groups.append([pkt])
        used_invs = set()
        for grp_key, grp_pkts in (list(groups.items())
                                   + [(None, l) for l in no_num_groups]):
            grp_pages = []
            for pkt in grp_pkts:
                for op in pkt.get('original_pages', []) or []:
                    if isinstance(op, dict) and op.get('page_number'):
                        grp_pages.append(op['page_number'])
            if not grp_pages or not inv_pages:
                continue
            grp_center = min(grp_pages)
            best_inv, best_dist = None, 10**9
            for inv, pgs in inv_pages.items():
                if inv in used_invs:
                    continue
                d = min(abs(p - grp_center) for p in pgs)
                if d < best_dist:
                    best_dist, best_inv = d, inv
            if best_inv is not None:
                for pkt in grp_pkts:
                    inv_map.setdefault(best_inv, {}).setdefault(
                        pkt.get('document_type', '') or '?', []).append(
                        pkt.get('packet_id', '?'))
                used_invs.add(best_inv)

    # Step 4 — read the LC's actual per-invoice required-doc list
    # from F46A. If the LC requires nothing per-invoice (rare),
    # nothing to check.
    required_per_invoice = _p198ge_required_per_invoice_set(structured_lc)
    if not required_per_invoice:
        return None

    # Step 5 — for each invoice, check coverage against the required
    # list.
    def _packet_matches_req(pkt_doc_type, req_canonical):
        """Return True if a packet whose document_type is pkt_doc_type
        satisfies a 'req_canonical' requirement."""
        canon = _p198ge_canonicalize_doc(pkt_doc_type)
        if not canon:
            return False
        if req_canonical == 'Transport Document (Bill of Lading / Airway Bill)':
            return canon in ('Bill of Lading', 'Airway Bill')
        return canon == req_canonical

    sec_clauses = []
    n_pass = 0
    n_fail = 0
    for inv in distinct_invoices:
        present = inv_map.get(inv, {})
        rows = []
        missing_names = []
        for req_name in required_per_invoice:
            has = False
            present_pkts = []
            for dt, pkt_ids in present.items():
                if _packet_matches_req(dt, req_name):
                    has = True
                    present_pkts.extend(pkt_ids)
            if has:
                n_pass += 1
                rows.append({
                    'row_id': f'PS-{inv}-{req_name[:25].replace(" ","_")}',
                    'clause_ref': f'Presentation-{inv}',
                    'field_tag': 'F43P',
                    'condition': f'{req_name} required for invoice {inv}',
                    'condition_text': f'{req_name} required for invoice {inv}',
                    'document_checked': req_name,
                    'compliance': 'PASS',
                    'result': (f"PRESENT for invoice {inv}: "
                               f"{', '.join(str(_x) for _x in present_pkts)}"),
                    'findings': (f"PRESENT for invoice {inv}: "
                                 f"{', '.join(str(_x) for _x in present_pkts)}"),
                })
            else:
                n_fail += 1
                missing_names.append(req_name)
                rows.append({
                    'row_id': f'PS-{inv}-MISSING-{req_name[:25].replace(" ","_")}',
                    'clause_ref': f'Presentation-{inv}',
                    'field_tag': 'F43P',
                    'condition': f'{req_name} required for invoice {inv}',
                    'condition_text': f'{req_name} required for invoice {inv}',
                    'document_checked': req_name,
                    'compliance': 'FAIL',
                    'result': (f"MISSING — {req_name} not presented for "
                               f"invoice {inv}."),
                    'findings': (f"MISSING — {req_name} not presented for "
                                 f"invoice {inv}."),
                })
        sec_clauses.append({
            'clause_ref': f'Presentation-{inv}',
            'clause_text': f'Document set for partial-shipment invoice {inv}',
            'overall_result': 'NOT COMPLIED' if missing_names else 'COMPLIED',
            'pass_count': len(rows) - len(missing_names),
            'fail_count': len(missing_names),
            'review_count': 0,
            'rows': rows,
        })

    # F43P=NOT ALLOWED with multiple invoices is itself a discrepancy.
    # Add a top row flagging that, regardless of whether all docs are
    # present per invoice.
    if f43p_not_allowed:
        n_fail += 1
        sec_clauses.insert(0, {
            'clause_ref': 'Presentation-F43P-VIOLATION',
            'clause_text': 'Partial shipments — F43P guard',
            'overall_result': 'NOT COMPLIED',
            'pass_count': 0,
            'fail_count': 1,
            'review_count': 0,
            'rows': [{
                'row_id': 'PS-F43P-VIOLATION',
                'clause_ref': 'Presentation-F43P-VIOLATION',
                'field_tag': 'F43P',
                'condition': 'Multiple invoices presented but F43P forbids partial shipments',
                'condition_text': 'Multiple invoices presented but F43P forbids partial shipments',
                'document_checked': 'All Documents',
                'compliance': 'FAIL',
                'result': (f"DISCREPANCY — LC F43P = '{f43p}' (partial "
                           f"shipments NOT ALLOWED) but the bundle "
                           f"contains {len(distinct_invoices)} distinct "
                           f"invoices: {', '.join(distinct_invoices)}."),
                'findings': (f"DISCREPANCY — LC F43P = '{f43p}' (partial "
                             f"shipments NOT ALLOWED) but the bundle "
                             f"contains {len(distinct_invoices)} distinct "
                             f"invoices: {', '.join(distinct_invoices)}."),
            }],
        })

    # If everything is clean (every invoice fully covered, F43P fine),
    # don't bother adding a section to the report.
    if n_fail == 0:
        return None

    title = 'Partial Shipment — Per-Invoice Document Completeness'
    description = (
        f'LC F43P={f43p or "(blank)"}. Bundle contains '
        f'{len(distinct_invoices)} distinct invoices: '
        f'{", ".join(distinct_invoices)}. Each invoice must carry '
        f'its own LC-required per-invoice docs '
        f'({", ".join(required_per_invoice)}). '
        f'1 transport doc ↔ 1 invoice (strict 1:1).'
    )
    return {
        'section_name': title,
        'section_title': title,
        'section_description': description,
        'total_pass': n_pass,
        'total_fail': n_fail,
        'total_review': 0,
        'clauses': sec_clauses,
    }


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

    sections_out = [asdict(s) for s in consolidated.sections]

    # P198gd — Inject partial-shipment per-invoice doc completeness
    # section at the top of the report when applicable. Only fires
    # for multi-invoice bundles with F43P=ALLOWED.
    overall_pass = consolidated.total_pass
    overall_fail = consolidated.total_fail
    overall_decision = consolidated.overall_decision
    critical_findings_out = list(consolidated.critical_findings)
    try:
        ps_section = _p198gd_partial_shipment_check(output_dir)
        if ps_section is not None:
            sections_out.insert(0, ps_section)
            overall_pass += ps_section['total_pass']
            overall_fail += ps_section['total_fail']
            if ps_section['total_fail'] > 0:
                overall_decision = 'DISCREPANT'
            # Append the missing-doc FAILs to critical_findings so they
            # surface in the executive-summary "Critical Findings"
            # table at the top of the PDF report — not just buried in
            # the per-invoice clause tables further down.
            ps_critical = []
            for cl in ps_section.get('clauses', []):
                for r in cl.get('rows', []):
                    if str(r.get('compliance', '')).upper() in ('FAIL', 'NOT COMPLIED'):
                        ps_critical.append({
                            'clause_ref': r.get('clause_ref') or cl.get('clause_ref'),
                            'clause_text': cl.get('clause_text', ''),
                            'condition': r.get('condition_text') or r.get('condition', ''),
                            'findings': r.get('findings', ''),
                            'result': r.get('result', ''),
                            'document_checked': r.get('document_checked', ''),
                        })
            # Put partial-shipment FAILs at the TOP of the critical
            # findings list so missing-doc discrepancies are seen
            # before less-critical row-level FAILs.
            critical_findings_out = ps_critical + critical_findings_out
            progress_fn(
                f"  [P198gd] Partial-shipment check: "
                f"{ps_section['total_pass']} present, "
                f"{ps_section['total_fail']} missing across "
                f"{len(ps_section['clauses'])} invoice(s)"
            )
    except Exception as _e:
        progress_fn(f"  [P198gd] Partial-shipment check failed: {_e}")

    result = {
        'step': 19,
        'step_name': 'Consolidated Clause Verification Output',
        'overall_decision': overall_decision,
        'total_clauses': consolidated.total_clauses,
        'total_rows': consolidated.total_rows,
        'total_pass': overall_pass,
        'total_fail': overall_fail,
        'total_review': consolidated.total_review,
        'critical_findings': critical_findings_out,
        'review_items': consolidated.review_items,
        'sections': sections_out,
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
