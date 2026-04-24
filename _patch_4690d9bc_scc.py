"""Patch job 4690d9bc to undo the SCC misclassification.

pkt_22 (page 28) is a Last-3-Cargoes statement from Control Union.
pkt_23 (page 29) is a Shelf Life Certificate from Control Union.
Both were incorrectly matched to "Shipping Company Certificate".

This patch fixes step08 + step09 JSONs and then CLEARS step 12+
for this job so the user can re-run verification against the
corrected classifications.
"""
import os, sys, json, shutil
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

JOB = '4690d9bc-ddc8-4ef1-ba50-ccf0ab2dd273'
BASE = os.path.join('results', JOB)

NEW_ASSIGN = {
    # pkt_22 — Last 3 Cargoes statement
    'pkt_22': dict(
        document_type='Last Cargoes Statement',
        classification_status='alien_document',
        matched_requirement_index=-1,
        matched_requirement_name='',
        reasoning='P198cx post-patch: Last-3-Cargoes statement issued by '
                  'Control Union (independent surveyor) — NOT a Shipping '
                  'Company Certificate (which requires carrier/agent '
                  'issuer attesting Institute Classification Clause).',
    ),
    # pkt_23 — Shelf Life Certificate
    'pkt_23': dict(
        document_type='Shelf Life Certificate',
        classification_status='alien_document',
        matched_requirement_index=-1,
        matched_requirement_name='',
        reasoning='P198cx post-patch: Shelf Life Certificate issued by '
                  'Control Union (independent surveyor) — NOT a Shipping '
                  'Company Certificate. Shelf-life content relates to the '
                  'beneficiary certificate requirement, not SCC.',
    ),
}


def patch_file(path, key):
    if not os.path.exists(path):
        print(f'  MISS {path}')
        return 0
    bak = path + '.bak_p198cx'
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
    with open(path, 'r', encoding='utf-8') as f:
        d = json.load(f)
    packets = d.get(key) or d.get('packets') or d.get('classified_packets') or \
              d.get('reconciled_packets')
    if not isinstance(packets, list):
        print(f'  no packet list in {path}')
        return 0
    changed = 0
    for p in packets:
        pid = p.get('packet_id')
        if pid in NEW_ASSIGN:
            upd = NEW_ASSIGN[pid]
            for k, v in upd.items():
                if p.get(k) != v:
                    p[k] = v
            changed += 1
    if changed:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(d, f, ensure_ascii=False, indent=2)
        print(f'  PATCHED {path}: {changed} packet(s) reclassified')
    else:
        print(f'  NO-CHANGE {path}')
    return changed


def clear_step(step):
    p = os.path.join(BASE, step)
    if os.path.isdir(p):
        bak = p + '.bak_p198cx'
        if not os.path.exists(bak):
            try:
                shutil.copytree(p, bak)
            except Exception as e:
                print(f'  backup failed for {p}: {e}')
        shutil.rmtree(p, ignore_errors=True)
        print(f'  CLEARED {p}')


def main():
    print(f'Patching job {JOB}')
    changed_s8 = patch_file(
        os.path.join(BASE, 'step08', 'step08_result.json'),
        'classified_packets',
    )
    changed_s9 = patch_file(
        os.path.join(BASE, 'step09', 'step09_result.json'),
        'reconciled_packets',
    )
    total = changed_s8 + changed_s9
    if total == 0:
        print('\nNo changes applied.')
        return

    # Clear downstream steps so they re-run with the fixed data.
    for s in ('step12', 'step13', 'step14', 'step14b', 'step19', 'step20'):
        clear_step(s)

    # Verify
    print('\n== verification ==')
    for step, key in [('step08','classified_packets'),
                      ('step09','reconciled_packets')]:
        path = os.path.join(BASE, step, f'{step}_result.json')
        if not os.path.exists(path):
            continue
        with open(path, encoding='utf-8') as f:
            d = json.load(f)
        ps = d.get(key, [])
        for p in ps:
            if p.get('packet_id') in ('pkt_22','pkt_23'):
                print(f"  {step} {p.get('packet_id'):<10} = "
                      f"{p.get('document_type'):<30} "
                      f"[{p.get('classification_status')}]")


if __name__ == '__main__':
    main()
