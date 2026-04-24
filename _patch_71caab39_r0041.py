"""Patch job 71caab39 R0041 (false PI-citation FAIL → PASS).

Applies ONLY to R0041. Leaves every other row, including R0004
(legitimate PI date mismatch), untouched. Updates step14 + step19 +
step20 stored JSON outputs so the checklist and report UI reflect
the fix without re-running verification.
"""
import os, sys, json, shutil
from datetime import datetime
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

JOB = '71caab39-d2a9-453a-8b73-99a2dd106f88'
BASE = os.path.join('results', JOB)
TARGET_ROW_ID = 'R0041'

NEW_FINDINGS = (
    "Proforma Invoice Ref. 786/S-13198-SOYPI-E is cited in the "
    "submission on the Commercial Invoice (structured "
    "proforma_reference: 786/S-13198 SOYPI-E on pkt_20). Under UCP "
    "600 the PI is typically quoted on the Commercial Invoice rather "
    "than submitted as a standalone document — the citation "
    "satisfies the 'must be present in the submission' requirement."
)
NEW_VERIF_NOTES = (
    "P198cl PI-citation-in-submission: structured proforma_reference "
    "on Commercial Invoice (pkt_20). Applied via targeted job-data "
    f"patch on {datetime.now().isoformat(timespec='seconds')}."
)


def _set_row(row):
    row['compliance'] = 'PASS'
    row['findings'] = NEW_FINDINGS
    row['result'] = NEW_FINDINGS[:200]
    row['verification_notes'] = NEW_VERIF_NOTES
    row['found_text'] = NEW_FINDINGS[:200]
    return row


def patch_json(path, row_key='rows'):
    if not os.path.exists(path):
        print(f'  MISS {path}')
        return False
    # Backup
    bak = path + '.bak_p198cl'
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    changed = False
    # Direct rows list
    rows = data.get(row_key)
    if isinstance(rows, list):
        for r in rows:
            if isinstance(r, dict) and r.get('row_id') == TARGET_ROW_ID:
                if r.get('compliance') == 'PASS':
                    print(f'  SKIP {path}: already PASS')
                    return False
                _set_row(r)
                changed = True
    # step19 consolidation: rows may be nested under clauses
    for k in ('clauses', 'consolidated_rows', 'verification_rows', 'non_compliances'):
        v = data.get(k)
        if isinstance(v, list):
            for entry in v:
                if isinstance(entry, dict):
                    for rr in (entry.get('rows') or []):
                        if isinstance(rr, dict) and rr.get('row_id') == TARGET_ROW_ID:
                            if rr.get('compliance') != 'PASS':
                                _set_row(rr); changed = True
                    if entry.get('row_id') == TARGET_ROW_ID:
                        if entry.get('compliance') != 'PASS':
                            _set_row(entry); changed = True
    # Generic recursive pass for any dict with row_id
    def _walk(node):
        nonlocal changed
        if isinstance(node, dict):
            if node.get('row_id') == TARGET_ROW_ID:
                if node.get('compliance') != 'PASS':
                    _set_row(node); changed = True
            for v in node.values():
                _walk(v)
        elif isinstance(node, list):
            for v in node:
                _walk(v)
    _walk(data)

    if changed:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f'  PATCHED {path}')
        return True
    print(f'  NO-CHANGE {path}')
    return False


def patch_step19(path):
    """step19 stores FAIL rows in critical_findings. Remove the R0041
    entry (identified by clause_ref='47A-4' + document_checked=
    'Proforma Invoice' + condition mentions proforma + 'must be
    present') and adjust the totals."""
    if not os.path.exists(path):
        print(f'  MISS {path}')
        return False
    bak = path + '.bak_p198cl'
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    cf = data.get('critical_findings') or []
    keep = []
    removed = 0
    for entry in cf:
        if (isinstance(entry, dict)
            and entry.get('clause_ref') == '47A-4'
            and entry.get('document_checked') == 'Proforma Invoice'
            and 'must be present' in str(entry.get('condition', '')).lower()):
            removed += 1
            continue
        keep.append(entry)
    if removed == 0:
        print(f'  NO-CHANGE {path} (R0041 not in critical_findings)')
        return False
    data['critical_findings'] = keep
    if isinstance(data.get('total_fail'), int):
        data['total_fail'] = max(0, data['total_fail'] - removed)
    if isinstance(data.get('total_pass'), int):
        data['total_pass'] = data['total_pass'] + removed
    # review items should be unaffected
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f'  PATCHED {path} (removed {removed} critical_findings, '
          f'total_fail={data.get("total_fail")}, total_pass={data.get("total_pass")})')
    return True


def patch_step20(path):
    """step20 holds summary counts. Decrement total_fail, increment
    total_pass. Overall decision stays DISCREPANT unless total_fail
    drops to 0."""
    if not os.path.exists(path):
        print(f'  MISS {path}')
        return False
    bak = path + '.bak_p198cl'
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Only adjust once — detect by checking if patch already applied
    # (we stamp a marker key)
    if data.get('_p198cl_patched'):
        print(f'  SKIP {path}: already patched')
        return False
    if isinstance(data.get('total_fail'), int) and data['total_fail'] > 0:
        data['total_fail'] -= 1
    if isinstance(data.get('total_pass'), int):
        data['total_pass'] += 1
    if isinstance(data.get('total_fail'), int) and data['total_fail'] == 0 \
       and isinstance(data.get('total_review'), int) and data['total_review'] == 0:
        data['overall_decision'] = 'COMPLIANT'
    data['_p198cl_patched'] = True
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f'  PATCHED {path} (total_fail={data.get("total_fail")}, '
          f'total_pass={data.get("total_pass")}, '
          f'overall={data.get("overall_decision")})')
    return True


def main():
    print(f'Patching job {JOB}, row {TARGET_ROW_ID}')
    patched = 0
    # step14 row-level patch
    for p in [
        os.path.join(BASE, 'step14', 'step14_result.json'),
        os.path.join(BASE, 'step14b', 'step14b_result.json'),
    ]:
        if patch_json(p):
            patched += 1
    # step19 critical_findings patch
    if patch_step19(os.path.join(BASE, 'step19', 'step19_result.json')):
        patched += 1
    # step20 counter patch
    if patch_step20(os.path.join(BASE, 'step20', 'step20_result.json')):
        patched += 1
    print(f'\nPatched {patched} files')

    candidates = [
        os.path.join(BASE, 'step14', 'step14_result.json'),
        os.path.join(BASE, 'step19', 'step19_result.json'),
        os.path.join(BASE, 'step20', 'step20_result.json'),
    ]

    # Verify final state
    for p in candidates:
        if not os.path.exists(p):
            continue
        with open(p, 'r', encoding='utf-8') as f:
            d = json.load(f)
        def _find(node):
            if isinstance(node, dict):
                if node.get('row_id') == TARGET_ROW_ID:
                    return node.get('compliance'), (node.get('findings') or '')[:80]
                for v in node.values():
                    r = _find(v)
                    if r: return r
            elif isinstance(node, list):
                for v in node:
                    r = _find(v)
                    if r: return r
            return None
        hit = _find(d)
        if hit:
            print(f'  {p}:\n    {TARGET_ROW_ID} = {hit[0]} | findings: {hit[1]}...')


if __name__ == '__main__':
    main()
