"""Patch job 73be98d9: flip R0016 (Pakistani Maritime Rules) and
R0017 (ETA at destination) from PASS to FAIL because the
Shipping Company Certificate text does not literally carry the
required statements. The earlier PASSes were LLM hallucinations.

R0015 (Institute Classification Clause) is already FAIL.
R0018 (vessel name) is correctly PASS.
"""
import os, sys, json, shutil
from datetime import datetime
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

JOB = '73be98d9-724f-4500-a08c-79802b4a5794'
BASE = os.path.join('results', JOB)

PATCHES = {
    'R0016': dict(
        compliance='FAIL',
        findings=(
            "Shipping Company Certificate is missing the literal "
            "statement that the carrying vessel is owned by "
            "companies operating in accordance with Pakistani "
            "Maritime Rules and Port Regulations. The earlier PASS "
            "was based on issuer identity (Pacific International "
            "Lines) without the required certification on the doc "
            "face — under UCP 600 Art 14(d) the statement must be "
            "literally present."
        ),
        notes='P198cz SCC-strict: Pakistani Maritime Rules statement absent on doc',
    ),
    'R0017': dict(
        compliance='FAIL',
        findings=(
            "Shipping Company Certificate is missing the "
            "approximate / estimated date of arrival (ETA) at the "
            "port of destination. The doc only shows the "
            "departure / sailing date ('Sailing on: 01 FEB 2025'), "
            "which is NOT the ETA at the destination port. The "
            "earlier PASS conflated the two."
        ),
        notes='P198cz SCC-strict: ETA at destination absent on doc',
    ),
}


def _save(p, d):
    bak = p + '.bak_p198cz'
    if not os.path.exists(bak) and os.path.exists(p):
        shutil.copy2(p, bak)
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(d, f, ensure_ascii=False, indent=2)


def patch_step14():
    p = os.path.join(BASE, 'step14', 'step14_result.json')
    with open(p, 'r', encoding='utf-8') as f:
        d = json.load(f)
    n = 0
    for r in d.get('rows', []):
        rid = r.get('row_id')
        if rid in PATCHES and str(r.get('compliance', '')).upper() != 'FAIL':
            up = PATCHES[rid]
            r['compliance'] = up['compliance']
            r['findings'] = up['findings']
            r['result'] = up['findings'][:200]
            r['found_text'] = up['findings'][:200]
            r['verification_notes'] = up['notes']
            n += 1
    if n:
        _save(p, d)
    print(f'step14: {n} row(s) flipped to FAIL')
    return n


def patch_step19(added_count):
    p = os.path.join(BASE, 'step19', 'step19_result.json')
    if not os.path.exists(p):
        return 0
    with open(p, 'r', encoding='utf-8') as f:
        d = json.load(f)
    s14_path = os.path.join(BASE, 'step14', 'step14_result.json')
    with open(s14_path, encoding='utf-8') as f:
        s14 = json.load(f)
    s14_rows = {r.get('row_id'): r for r in s14.get('rows', [])}

    cf = d.get('critical_findings') or []
    existing_keys = {(e.get('clause_ref'), e.get('document_checked'),
                      (e.get('findings') or '')[:80]) for e in cf if isinstance(e, dict)}

    added = 0
    for rid in PATCHES:
        r = s14_rows.get(rid)
        if not r:
            continue
        entry = {
            'clause_ref': r.get('clause_ref'),
            'clause_text': r.get('original_clause_text'),
            'condition': r.get('condition_text'),
            'findings': r.get('findings'),
            'result': r.get('result'),
            'document_checked': r.get('document_checked'),
        }
        key = (entry['clause_ref'], entry['document_checked'],
               (entry['findings'] or '')[:80])
        if key in existing_keys:
            continue
        cf.append(entry); existing_keys.add(key); added += 1
    if added == 0:
        return 0
    d['critical_findings'] = cf
    if isinstance(d.get('total_fail'), int):
        d['total_fail'] += added
    if isinstance(d.get('total_pass'), int):
        d['total_pass'] = max(0, d['total_pass'] - added)
    _save(p, d)
    print(f'step19: +{added} critical_findings (total_fail={d.get("total_fail")})')
    return added


def patch_step20(added):
    p = os.path.join(BASE, 'step20', 'step20_result.json')
    if added <= 0 or not os.path.exists(p):
        return 0
    with open(p, 'r', encoding='utf-8') as f:
        d = json.load(f)
    if d.get('_p198cz_patched'):
        return 0
    if isinstance(d.get('total_fail'), int):
        d['total_fail'] += added
    if isinstance(d.get('total_pass'), int):
        d['total_pass'] = max(0, d['total_pass'] - added)
    if isinstance(d.get('total_fail'), int) and d['total_fail'] > 0:
        d['overall_decision'] = 'DISCREPANT'
    d['_p198cz_patched'] = True
    _save(p, d)
    print(f'step20 updated (total_fail={d.get("total_fail")}, total_pass={d.get("total_pass")}, overall={d.get("overall_decision")})')
    return 1


def main():
    n = patch_step14()
    if n == 0:
        print('No rows to patch.')
        return
    patch_step19(n)
    patch_step20(n)
    # Verify
    p = os.path.join(BASE, 'step14', 'step14_result.json')
    with open(p, encoding='utf-8') as f:
        d = json.load(f)
    for r in d.get('rows', []):
        if r.get('row_id') in PATCHES:
            print(f"  {r['row_id']} = {r.get('compliance')} | {(r.get('findings') or '')[:120]}")


if __name__ == '__main__':
    main()
