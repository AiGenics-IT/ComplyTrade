"""Patch job 56f424de: flip R0004 (45A-1 proforma citation) from
FAIL to PASS.

The row's finding quoted a date mismatch (LC Jan 21, invoice
Feb 18), but the reference DB13612512-... / PI 786/S-13198-SOYPI-E
matches. User confirms the proforma citation on the Commercial
Invoice should PASS — the date-mismatch finding is not a real
documentary discrepancy for this presentation.

This does NOT touch any other row (origin, BL consignee, missing
documents, etc.).
"""
import os, sys, json, shutil
from datetime import datetime
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

JOB = '56f424de-a446-46cf-9d28-72cc22fa7905'
BASE = os.path.join('results', JOB)
TARGET_ROW_ID = 'R0004'
TARGET_CLAUSE = '45A-1'
TARGET_DOC    = 'Commercial Invoice'
TARGET_COND_MARKER = 'PROFORMA'

NEW_FINDINGS = (
    "Commercial Invoice cites Beneficiary's Proforma Invoice "
    "Ref. 786/S-13198-SOYPI-E — the reference is present and "
    "matches the LC requirement. The proforma citation on the "
    "invoice satisfies the 'strictly as per' requirement."
)
NEW_VERIF_NOTES = (
    "Manual patch: proforma citation row flipped PASS per "
    f"user review on {datetime.now().isoformat(timespec='seconds')}."
)


def _save(p, d):
    bak = p + '.bak_r0004'
    if not os.path.exists(bak) and os.path.exists(p):
        shutil.copy2(p, bak)
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(d, f, ensure_ascii=False, indent=2)


def _matches_row(row):
    if not isinstance(row, dict):
        return False
    rid = row.get('row_id')
    cref = row.get('clause_ref')
    doc = row.get('document_checked')
    cond = (row.get('condition_text') or row.get('condition') or '').upper()
    if rid == TARGET_ROW_ID:
        return True
    return (cref == TARGET_CLAUSE and doc == TARGET_DOC
            and TARGET_COND_MARKER in cond
            and str(row.get('compliance', '')).upper() == 'FAIL')


def patch_step14():
    p = os.path.join(BASE, 'step14', 'step14_result.json')
    with open(p, 'r', encoding='utf-8') as f:
        d = json.load(f)
    changed = 0
    for r in d.get('rows', []):
        if _matches_row(r):
            if str(r.get('compliance', '')).upper() == 'PASS':
                continue
            r['compliance'] = 'PASS'
            r['findings'] = NEW_FINDINGS
            r['result'] = NEW_FINDINGS[:200]
            r['found_text'] = NEW_FINDINGS[:200]
            r['verification_notes'] = NEW_VERIF_NOTES
            changed += 1
    if changed:
        _save(p, d)
    print(f'step14: flipped {changed} row(s)')
    return changed


def patch_step19(removed_count):
    p = os.path.join(BASE, 'step19', 'step19_result.json')
    if not os.path.exists(p):
        return 0
    with open(p, 'r', encoding='utf-8') as f:
        d = json.load(f)
    cf = d.get('critical_findings') or []
    keep = []
    removed = 0
    for e in cf:
        if isinstance(e, dict):
            ec = (e.get('condition') or e.get('condition_text') or '').upper()
            dc = e.get('document_checked') or ''
            cr = e.get('clause_ref') or ''
            if (cr == TARGET_CLAUSE and dc == TARGET_DOC
                    and TARGET_COND_MARKER in ec):
                removed += 1
                continue
        keep.append(e)
    if removed == 0:
        return 0
    d['critical_findings'] = keep
    if isinstance(d.get('total_fail'), int):
        d['total_fail'] = max(0, d['total_fail'] - removed)
    if isinstance(d.get('total_pass'), int):
        d['total_pass'] += removed
    _save(p, d)
    print(f'step19: removed {removed} critical_findings')
    return removed


def patch_step20(removed):
    p = os.path.join(BASE, 'step20', 'step20_result.json')
    if removed <= 0 or not os.path.exists(p):
        return 0
    with open(p, 'r', encoding='utf-8') as f:
        d = json.load(f)
    if d.get('_r0004_patched'):
        return 0
    if isinstance(d.get('total_fail'), int):
        d['total_fail'] = max(0, d['total_fail'] - removed)
    if isinstance(d.get('total_pass'), int):
        d['total_pass'] += removed
    if (isinstance(d.get('total_fail'), int) and d['total_fail'] == 0
            and isinstance(d.get('total_review'), int) and d['total_review'] == 0):
        d['overall_decision'] = 'COMPLIANT'
    d['_r0004_patched'] = True
    _save(p, d)
    print(f"step20 updated (total_fail={d.get('total_fail')}, "
          f"total_pass={d.get('total_pass')}, overall={d.get('overall_decision')})")
    return 1


def main():
    n14 = patch_step14()
    n19 = patch_step19(n14)
    patch_step20(n19)

    # Verify
    with open(os.path.join(BASE, 'step14', 'step14_result.json'), encoding='utf-8') as f:
        d = json.load(f)
    for r in d.get('rows', []):
        if _matches_row(r) or r.get('row_id') == TARGET_ROW_ID:
            print(f"  {r.get('row_id')} = {r.get('compliance')} | "
                  f"{str(r.get('findings',''))[:100]}")
            break


if __name__ == '__main__':
    main()
