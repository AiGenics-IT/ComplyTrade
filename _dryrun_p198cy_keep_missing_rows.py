"""
P198cy dry-run — every clause row kept visible even when the
underlying required document is missing from the submission.

Old behavior (P183): when multiple clause sub-conditions targeted
the same required document, only the FIRST was emitted as
"Required document missing" — the rest were silently dropped
from the report via `_drop_from_report=True`. That hid entire
clauses like 46A-5 / 46A-8 / 46A-9 from the checklist when their
document (Beneficiary Certificate, Courier Receipt, Halal, etc.)
was not in the presentation.

P198cy: remove the sibling-dedup. Every clause row emits its own
FAIL with "Required document missing: <doc>" so the report
shows full 46A-X clause coverage regardless of document presence.

This harness simulates the step14 pre-filtered processing of a
set of rows where one document is missing and multiple clauses
reference it. Expected: ALL rows survive with FAIL verdicts, none
dropped.
"""
import sys, os


def simulate_step14_missing(rows, submitted_doc_types):
    """Very small simulation of the relevant step14 pre-filter.
    Returns the list of rows that survive (no _drop_from_report)
    with their final compliance."""
    submitted = set(s.lower() for s in submitted_doc_types)
    seen_missing = set()
    out = []
    for r in rows:
        r = dict(r)
        doc = (r.get('document_checked') or '').strip()
        doc_lo = doc.lower()
        if doc_lo in submitted:
            # Document IS in the submission — pass through with
            # whatever compliance it already has.
            out.append(r); continue
        # Document is missing — emit FAIL per clause sub-condition.
        r['compliance'] = 'FAIL'
        r['findings'] = f"{doc} not found in submission"
        r['result'] = f"Required document missing: {doc}"
        r['verification_notes'] = (
            "P198cy: required document missing from submission"
        )
        r.pop('_drop_from_report', None)
        out.append(r)
        seen_missing.add(doc_lo)
    return out


# ── Scenarios ──
SC = []

# Scenario 1: real-world mix — 46A-4 Shipment Advice missing with
# 7 sub-conditions + 46A-5 SCC present + 46A-8 Beneficiary Cert
# missing with 3 sub-conditions + 46A-9 Beneficiary Cert missing
# with 3 sub-conditions + 46A-11 Halal missing with 2 sub-conds.
SC.append(dict(
    name='Real 0ec5e7c3-like: many missing-doc clauses all stay visible',
    rows=[
        dict(row_id='R0033', clause_ref='46A-4', document_checked='Shipment Advice'),
        dict(row_id='R0034', clause_ref='46A-4', document_checked='Shipment Advice'),
        dict(row_id='R0035', clause_ref='46A-4', document_checked='Shipment Advice'),
        dict(row_id='R0041', clause_ref='46A-5', document_checked='Shipping Company Certificate',
             compliance='PASS'),
        dict(row_id='R0052', clause_ref='46A-8', document_checked='Beneficiary Certificate'),
        dict(row_id='R0053', clause_ref='46A-8', document_checked='Courier Receipt'),
        dict(row_id='R0054', clause_ref='46A-8', document_checked='Fax Confirmation'),
        dict(row_id='R0057', clause_ref='46A-9', document_checked='Beneficiary Certificate'),
        dict(row_id='R0058', clause_ref='46A-9', document_checked='Beneficiary Certificate'),
        dict(row_id='R0013', clause_ref='46A-11', document_checked='Halal Certificate'),
    ],
    submitted=['Shipping Company Certificate', 'Commercial Invoice'],
    # Every row survives; the SCC row keeps its PASS; everything
    # else becomes FAIL "Required document missing".
    expect_count=10,
    expect_no_drops=True,
))

SC.append(dict(
    name='All docs present → no rows changed',
    rows=[
        dict(row_id='R0001', clause_ref='46A-1', document_checked='Commercial Invoice', compliance='PASS'),
        dict(row_id='R0002', clause_ref='46A-2', document_checked='Bill of Lading', compliance='PASS'),
    ],
    submitted=['Commercial Invoice', 'Bill of Lading'],
    expect_count=2,
    expect_no_drops=True,
))

SC.append(dict(
    name='All docs missing → every row shown as FAIL',
    rows=[
        dict(row_id='R0011', clause_ref='46A-10', document_checked='Beneficiary Certificate'),
        dict(row_id='R0052', clause_ref='46A-8', document_checked='Beneficiary Certificate'),
        dict(row_id='R0057', clause_ref='46A-9', document_checked='Beneficiary Certificate'),
    ],
    submitted=[],
    expect_count=3,
    expect_no_drops=True,
))


def main():
    passed = 0; failed = 0
    for i, sc in enumerate(SC, 1):
        out = simulate_step14_missing(sc['rows'], sc['submitted'])
        got_count = len(out)
        no_drops = not any(r.get('_drop_from_report') for r in out)
        ok = (got_count == sc['expect_count'] and no_drops == sc['expect_no_drops'])
        tag = 'OK ' if ok else 'FAIL'
        print(f"[{tag}] #{i:02d}  {sc['name']}")
        print(f"         expect={sc['expect_count']} rows, no_drops={sc['expect_no_drops']}")
        print(f"         got   ={got_count} rows, no_drops={no_drops}")
        if not ok:
            for r in out:
                print(f"           {r.get('row_id')} | {r.get('clause_ref')} | "
                      f"{r.get('document_checked')} | {r.get('compliance')} | "
                      f"drop={r.get('_drop_from_report')}")
        if ok: passed += 1
        else: failed += 1
    print(f"\n{'='*78}\n{passed}/{passed+failed} P198cy scenarios OK\n{'='*78}")
    return failed == 0


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
