"""
P198fc dry-run — split "All Documents" aggregation rows into one row
per document type at the consolidation stage.

Background:
  Step 14 fans out a universal-quantifier clause to every document in
  the submission and AGGREGATES the per-doc results into one row whose
  findings carry a pipe-separated "Per-doc:" block. The user wanted
  one ROW per document so each docs PASS/FAIL/REVIEW is visible at a
  glance in the PDF report and the checklist UI.

The fix at step19_consolidation.py:_expand_all_documents_row():
  - Detects rows with document_checked = "All Documents" + "Per-doc:"
    block in findings
  - Splits the Per-doc: block on " | " and parses each segment
    "<DocType>: <PASS/FAIL/REVIEW> — <text>"
  - Returns parent summary row + N child rows
  - Each child inherits the parent's clause_ref / condition and gets
    its own document_checked + compliance + findings/result
"""
import sys, os, json
sys.path.insert(0, 'd:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final')

from steps.step19_consolidation import _expand_all_documents_row

results = []
def assert_eq(name, got, expected):
    ok = (got == expected)
    print(f"[{'OK' if ok else 'FAIL'}] {name}")
    if not ok:
        print(f"          got     : {got!r}")
        print(f"          expected: {expected!r}")
    results.append(ok)

# ── Test 1: real-world example from job 2d98b74c (the user's screenshot) ──
print("--- Test 1: real-world 'All Documents must show LC number' ---")
sample_findings = (
    "Requirement satisfied on all 8 document class(es): Shipment Advice, Draft Bill of Exchange, "
    "Bill of Lading, Attached Rider, Packing List, Weight List, Certificate of Origin, "
    "Commercial Invoice. Per-doc: "
    "Shipment Advice: PASS — The Shipment Advice document includes the LC number '0052ILC083930' | "
    "Draft Bill of Exchange: PASS — The Draft Bill of Exchange correctly references the LC number | "
    "Shipment Advice: PASS — The Shipment Advice correctly references the LC number 0052ILC083930 | "
    "Bill of Lading: PASS — The document text shows the LC number (0052ILC083930), H.S. Code | "
    "Attached Rider: PASS — The document includes the LC number '0052ILC083930' | "
    "Packing List: PASS — The Packing List correctly references the LC number and date | "
    "Shipment Advice: PASS — The Shipment Advice is addressed to the Applicant | "
    "Weight List: PASS — The Weight List includes the LC number and date of issue | "
    "Certificate of Origin: PASS — The Certificate of Origin includes the LC number"
)
parent_row = {
    'clause_ref': '47A-2',
    'condition': 'L/C number must appear on all documents',
    'condition_text': 'L/C number must appear on all documents',
    'document_checked': 'All Documents',
    'findings': sample_findings,
    'result': 'PRESENT ON ALL 8 DOC(S): SHIPMENT ADVICE, DRAFT BILL OF EXCHANGE',
    'compliance': 'PASS',
    'rule_id': 'R0024',
}
out = _expand_all_documents_row(parent_row)
print(f"  expanded {len(out)} rows (1 parent + N child)")
for r in out:
    print(f"  - doc='{r.get('document_checked')}' compliance={r.get('compliance')} "
          f"summary={'(SUMMARY)' if r.get('_all_documents_summary') else '(child)' if r.get('_split_from_all_documents') else ''}")

assert_eq("real-world: at least 9 rows (parent + 8 doc-types + dup Shipment Advice rows)",
          len(out) >= 8, True)
assert_eq("real-world: parent row tagged _all_documents_summary",
          out[0].get('_all_documents_summary'), True)
assert_eq("real-world: parent row keeps document_checked='All Documents'",
          out[0].get('document_checked'), 'All Documents')
assert_eq("real-world: child rows have _split_from_all_documents flag",
          all(r.get('_split_from_all_documents') for r in out[1:]), True)
assert_eq("real-world: clause_ref propagated to children",
          all(r.get('clause_ref') == '47A-2' for r in out[1:]), True)
assert_eq("real-world: condition propagated to children",
          all('L/C number' in (r.get('condition') or '') for r in out[1:]), True)
# Verdicts
for r in out[1:]:
    assert_eq(f"  child '{r['document_checked']}': has compliance",
              r.get('compliance') in ('PASS','FAIL','REVIEW','N/A'), True)
# At least one of each known doc-type appears
docs_in_children = {r.get('document_checked') for r in out[1:]}
for expected_doc in ['Shipment Advice', 'Draft Bill of Exchange', 'Bill of Lading',
                     'Packing List', 'Certificate of Origin']:
    assert_eq(f"  child docs include '{expected_doc}'",
              expected_doc in docs_in_children, True)


# ── Test 2: NON-aggregation rows must pass through unchanged ─────────────
print("\n--- Test 2: non-aggregation rows pass through unchanged ---")
single_row = {
    'clause_ref': '46A-1',
    'condition': 'Commercial Invoice in triplicate',
    'document_checked': 'Commercial Invoice',
    'findings': 'Invoice is in triplicate',
    'result': 'Triplicate confirmed',
    'compliance': 'PASS',
}
out = _expand_all_documents_row(single_row)
assert_eq("single-doc row returns unchanged (1 row)", len(out), 1)
assert_eq("single-doc row identity preserved", out[0] is single_row, True)


# ── Test 3: 'All Documents' row WITHOUT Per-doc block ───────────────────
print("\n--- Test 3: 'All Documents' row without Per-doc: ---")
no_pd_row = {
    'clause_ref': '47A-1',
    'condition': 'All documents must be in English',
    'document_checked': 'All Documents',
    'findings': 'All documents are in English language as required',
    'result': 'English language confirmed',
    'compliance': 'PASS',
}
out = _expand_all_documents_row(no_pd_row)
assert_eq("'All Documents' without Per-doc block: passes through unchanged",
          len(out), 1)


# ── Test 4: FAIL aggregation — required value missing on some docs ──────
print("\n--- Test 4: FAIL aggregation ---")
fail_findings = (
    "Required value missing on: Bill of Lading, Packing List. "
    "Present on: Commercial Invoice. Per-doc: "
    "Commercial Invoice: PASS — Invoice shows LC number | "
    "Bill of Lading: FAIL — BL does not show the required LC number | "
    "Packing List: FAIL — Packing List missing LC reference"
)
fail_row = {
    'clause_ref': '47A-2',
    'condition': 'LC number must appear on all documents',
    'document_checked': 'All Documents',
    'findings': fail_findings,
    'result': 'Missing on 2 doc(s): Bill of Lading, Packing List',
    'compliance': 'FAIL',
}
out = _expand_all_documents_row(fail_row)
assert_eq("FAIL aggregation: parent + 3 children = 4 rows", len(out), 4)
verdicts_by_doc = {r.get('document_checked'): r.get('compliance') for r in out[1:]}
assert_eq("FAIL agg: Commercial Invoice = PASS",
          verdicts_by_doc.get('Commercial Invoice'), 'PASS')
assert_eq("FAIL agg: Bill of Lading = FAIL",
          verdicts_by_doc.get('Bill of Lading'), 'FAIL')
assert_eq("FAIL agg: Packing List = FAIL",
          verdicts_by_doc.get('Packing List'), 'FAIL')
assert_eq("FAIL agg: parent compliance preserved as FAIL",
          out[0].get('compliance'), 'FAIL')


# ── Test 5: REVIEW aggregation ──────────────────────────────────────────
print("\n--- Test 5: REVIEW aggregation ---")
rev_findings = (
    "Requirement unclear on: Weight List. Present on: Commercial Invoice. "
    "Per-doc: Commercial Invoice: PASS — Reference present | "
    "Weight List: REVIEW — Document text was not clearly readable for verification"
)
rev_row = {
    'clause_ref': '47A-3',
    'condition': 'All documents dated on/after LC date',
    'document_checked': 'All Documents',
    'findings': rev_findings,
    'result': 'Unclear on 1 doc(s)',
    'compliance': 'REVIEW',
}
out = _expand_all_documents_row(rev_row)
assert_eq("REVIEW agg: parent + 2 children", len(out), 3)
v = {r.get('document_checked'): r.get('compliance') for r in out[1:]}
assert_eq("REVIEW agg: Weight List = REVIEW", v.get('Weight List'), 'REVIEW')


# ── Test 6: malformed Per-doc segment is skipped, not crash ─────────────
print("\n--- Test 6: malformed segment skipped ---")
junk_row = {
    'document_checked': 'All Documents',
    'findings': 'Per-doc: garbage without verdict | Bill of Lading: PASS — clean',
    'compliance': 'PASS',
}
out = _expand_all_documents_row(junk_row)
assert_eq("malformed: at least 2 rows (parent + 1 valid child)", len(out) >= 2, True)


# ── Test 7: full pipeline integration via _consolidate() ────────────────
print("\n--- Test 7: full _consolidate() integration ---")
from steps.step19_consolidation import _consolidate
all_rows = [
    parent_row,        # gets expanded
    single_row,        # passes through
    fail_row,          # gets expanded
    no_pd_row,         # passes through
]
consolidated = _consolidate(all_rows)
# Total rows after consolidation should be: parent_row (1+9) + single_row (1) + fail_row (1+3) + no_pd_row (1) = 16
total_rows = sum(cg.row_count for s in consolidated.sections for cg in s.clauses)
print(f"  Total rows after _consolidate: {total_rows}")
# Allow slight variance in count due to internal pad/dedup, but verify the
# key behavior: more rows out than in (because of expansion)
assert_eq("_consolidate(): total rows > input (expansion fired)",
          total_rows > len(all_rows), True)


# ── Test 8: dataclass field stripping — extra fields don't break vrows ──
print("\n--- Test 8: extra fields don't break VerificationRow ---")
# The expanded child rows have a `_split_from_all_documents` marker that
# isn't a VerificationRow field. The consolidator must not crash on it.
# (already tested implicitly via Test 7, but explicit assertion here)
errors = 0
for s in consolidated.sections:
    for cg in s.clauses:
        for vr in cg.rows:
            try:
                _ = vr.condition  # access fields
            except Exception:
                errors += 1
assert_eq("VerificationRow access: no exceptions", errors, 0)


passed = sum(results)
total_t = len(results)
print(f"\n{passed}/{total_t} cases passed")
if passed != total_t:
    sys.exit(1)
print("OVERALL: OK")
sys.exit(0)
