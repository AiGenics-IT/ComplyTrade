"""Patch job 73be98d9: upgrade R0055 (47A-7 charges) and R0057
(47A-9 authenticated SWIFT advice) from N/A to FAIL.

Job state:
  • Documentary Remittance IS present (pkt_3) but does NOT carry
    the literal certification that all charges of negotiating
    bank and advising bank are paid by the beneficiary → R0055
    FAIL.
  • No MT799 / MT999 / authenticated-SWIFT packet in the
    submission → R0057 FAIL.
"""
import os, sys, json, shutil
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

JOB = '73be98d9-724f-4500-a08c-79802b4a5794'
BASE = os.path.join('results', JOB)

PATCHES = {
    'R0055': dict(
        document_checked='Documentary Remittance',
        compliance='FAIL',
        findings=(
            "Documentary Remittance is present in the submission "
            "but does not literally certify that all charges of "
            "the negotiating bank and the advising bank are paid "
            "by the beneficiary. The F47A clause requires this "
            "certification to appear on the documents forwarding "
            "schedule under UCP 600 Art 14(d)."
        ),
        notes='P198da F47A charges-on-forwarding-schedule: DR present but no charges statement',
    ),
    'R0057': dict(
        document_checked='MT799/MT999 SWIFT Advice',
        compliance='FAIL',
        findings=(
            "Authenticated SWIFT advice (MT799 / MT999) from the "
            "negotiating bank is not present in the submission. "
            "The LC F47A clause requires an authenticated SWIFT "
            "message stating amount of negotiation, BL number, "
            "vessel, voyage, ports, container/seal numbers and "
            "date of dispatch — and a copy of that SWIFT message "
            "must accompany the original documents."
        ),
        notes='P198da F47A SWIFT-advice: MT799/MT999 not in submission',
    ),
}


def _save(p, d):
    bak = p + '.bak_p198da'
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
        if rid in PATCHES:
            up = PATCHES[rid]
            r['document_checked'] = up['document_checked']
            r['compliance'] = up['compliance']
            r['findings'] = up['findings']
            r['result'] = up['findings'][:200]
            r['found_text'] = up['findings'][:200]
            r['verification_notes'] = up['notes']
            n += 1
    if n:
        _save(p, d)
    print(f'step14: {n} row(s) updated')
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
    _save(p, d)
    print(f'step19: +{added} critical_findings (total_fail={d.get("total_fail")})')
    return added


def patch_step20(added):
    p = os.path.join(BASE, 'step20', 'step20_result.json')
    if added <= 0 or not os.path.exists(p):
        return 0
    with open(p, 'r', encoding='utf-8') as f:
        d = json.load(f)
    if d.get('_p198da_patched'):
        return 0
    if isinstance(d.get('total_fail'), int):
        d['total_fail'] += added
    if isinstance(d.get('total_fail'), int) and d['total_fail'] > 0:
        d['overall_decision'] = 'DISCREPANT'
    d['_p198da_patched'] = True
    _save(p, d)
    print(f'step20 (total_fail={d.get("total_fail")}, overall={d.get("overall_decision")})')
    return 1


def main():
    n = patch_step14()
    if n == 0:
        print('No rows to patch.')
        return
    patch_step19(n)
    patch_step20(n)
    p = os.path.join(BASE, 'step14', 'step14_result.json')
    with open(p, encoding='utf-8') as f:
        d = json.load(f)
    for r in d.get('rows', []):
        if r.get('row_id') in PATCHES:
            print(f"  {r['row_id']} = {r.get('compliance')} | {r.get('document_checked')} | {(r.get('findings') or '')[:120]}")


if __name__ == '__main__':
    main()
