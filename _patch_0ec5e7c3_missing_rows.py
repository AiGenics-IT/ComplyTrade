"""Patch job 0ec5e7c3: restore the 17 clause rows that were
silently dropped by the legacy P183 missing-document dedup.

Each restored row becomes a FAIL with "Required document
missing: <doc>" so the UI checklist shows every 46A-X point,
even when the underlying document is absent.

Also updates step19 critical_findings and step20 summary counters.
"""
import os, sys, json, shutil
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

JOB = '0ec5e7c3-996b-4214-9795-99ff239ffdb0'
BASE = os.path.join('results', JOB)


def load(p):
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)


def save(p, d):
    bak = p + '.bak_p198cy'
    if not os.path.exists(bak) and os.path.exists(p):
        shutil.copy2(p, bak)
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(d, f, ensure_ascii=False, indent=2)


def mk_missing_row(step13_row):
    """Turn a step13 row into a step14 FAIL 'Required document missing' row."""
    doc = step13_row.get('document_checked') or 'Required document'
    findings = f"{doc} not found in submission"
    result = f"Required document missing: {doc}"
    r = dict(step13_row)  # copy all step13 fields
    r['found_text'] = findings
    r['document_checked'] = doc
    r['compliance'] = 'FAIL'
    r['confidence'] = 1.0
    r['result'] = result
    r['findings'] = findings
    r['verification_notes'] = (
        "P198cy: required document missing from submission — "
        "clause check reported per-clause (no sibling-dedup)"
    )
    # Ensure _drop_from_report is NOT set
    r.pop('_drop_from_report', None)
    return r


def main():
    s13_path = os.path.join(BASE, 'step13', 'step13_result.json')
    s14_path = os.path.join(BASE, 'step14', 'step14_result.json')
    s19_path = os.path.join(BASE, 'step19', 'step19_result.json')
    s20_path = os.path.join(BASE, 'step20', 'step20_result.json')

    s13 = load(s13_path)
    s14 = load(s14_path)

    s13_rows = s13.get('rows', [])
    s14_rows = s14.get('rows', [])
    s14_ids = {r.get('row_id') for r in s14_rows}

    # Find rows in step13 but missing from step14
    missing = [r for r in s13_rows if r.get('row_id') not in s14_ids]
    if not missing:
        print('No missing rows.')
        return

    print(f'Restoring {len(missing)} dropped rows to step14:')
    added = []
    for sr in missing:
        new_row = mk_missing_row(sr)
        s14_rows.append(new_row)
        added.append(new_row)
        print(f"  {new_row['row_id']} | {new_row.get('clause_ref'):<8} | "
              f"{new_row.get('document_checked'):<35} | {new_row['result']}")

    # Re-sort step14 rows by row_id to keep natural order
    def _row_sort_key(r):
        rid = r.get('row_id', 'R9999')
        try:
            return int(rid.lstrip('R'))
        except Exception:
            return 9999
    s14_rows.sort(key=_row_sort_key)
    s14['rows'] = s14_rows
    save(s14_path, s14)
    print(f'\nstep14 saved with {len(s14_rows)} rows total')

    # step19 — add these to critical_findings
    if os.path.exists(s19_path):
        s19 = load(s19_path)
        cf = s19.get('critical_findings') or []
        existing_keys = set()
        for e in cf:
            if isinstance(e, dict):
                existing_keys.add((e.get('clause_ref'),
                                   e.get('document_checked'),
                                   (e.get('findings') or '')[:80]))
        added_to_cf = 0
        for r in added:
            entry = {
                'clause_ref': r.get('clause_ref'),
                'clause_text': r.get('original_clause_text'),
                'condition': r.get('condition_text'),
                'findings': r['findings'],
                'result': r['result'],
                'document_checked': r.get('document_checked'),
            }
            key = (entry['clause_ref'], entry['document_checked'],
                   (entry['findings'] or '')[:80])
            if key in existing_keys:
                continue
            cf.append(entry)
            existing_keys.add(key)
            added_to_cf += 1
        s19['critical_findings'] = cf
        if isinstance(s19.get('total_fail'), int):
            s19['total_fail'] = s19['total_fail'] + added_to_cf
        if isinstance(s19.get('total_rows'), int):
            s19['total_rows'] = s19['total_rows'] + added_to_cf
        save(s19_path, s19)
        print(f'step19 critical_findings +{added_to_cf} (total_fail={s19.get("total_fail")})')

    # step20 — update counters
    if os.path.exists(s20_path):
        s20 = load(s20_path)
        if isinstance(s20.get('total_fail'), int):
            s20['total_fail'] = s20['total_fail'] + len(added)
        if isinstance(s20.get('total_rows'), int):
            s20['total_rows'] = s20['total_rows'] + len(added)
        if isinstance(s20.get('total_fail'), int) and s20['total_fail'] > 0:
            s20['overall_decision'] = 'DISCREPANT'
        save(s20_path, s20)
        print(f'step20 counters updated (total_fail={s20.get("total_fail")}, '
              f'total_rows={s20.get("total_rows")}, overall={s20.get("overall_decision")})')


if __name__ == '__main__':
    main()
