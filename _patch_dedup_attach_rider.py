"""
Surgical patch — remove duplicate Attach Rider packets from any job
whose step03_result.json has them. Mirrors what the P198eu source fix
does going forward; this is the one-shot cleanup for already-saved
jobs that were written before the fix.

Logic: for each "ATTACHED RIDER / SPECIFICATION OF CARGO / Attach
List" packet whose pages are ALREADY contained in a "Bill of Lading +
Attached List" packet, DROP it.
"""
import os, json, sys, shutil, datetime, glob

ROOT = 'd:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final'
RESULTS = os.path.join(ROOT, 'results')

def is_bl_attach_label(dt):
    dtl = (dt or '').lower().strip()
    if '+' in dtl:
        return False  # already-merged BL+Attached → keep
    return any(k in dtl for k in (
        'attach list', 'attached list', 'attached sheet', 'attached rider',
        'bl attached sheet', 'bl rider', 'rider', 'attached schedule',
        'specification of cargo', 'cargo specification', 'attached specification',
    ))

def patch_job(job_id, dry=False):
    s3f = os.path.join(RESULTS, job_id, 'step03', 'step03_result.json')
    if not os.path.exists(s3f):
        return 0
    with open(s3f, 'r', encoding='utf-8') as f:
        s3 = json.load(f)
    pkts = s3.get('packets') or []

    # Pages already covered by a BL+Attached packet
    covered = set()
    for p in pkts:
        if '+ attached' in (p.get('document_type', '') or '').lower():
            covered.update(p.get('page_numbers') or [])

    # Identify duplicate Rider packets
    drop_indices = []
    for i, p in enumerate(pkts):
        if not is_bl_attach_label(p.get('document_type', '')):
            continue
        pgs = p.get('page_numbers') or []
        if pgs and all(pg in covered for pg in pgs):
            drop_indices.append((i, p.get('packet_id'), pgs, p.get('document_type')))

    if not drop_indices:
        return 0

    print(f"\n=== {job_id} ===")
    for i, pid, pgs, dt in drop_indices:
        print(f"  drop pkt[{i}]={pid}  pages={pgs}  dt={dt}")

    if not dry:
        # Backup
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        shutil.copy2(s3f, s3f + f'.bak_{ts}')
        # Remove
        kept = [p for i, p in enumerate(pkts) if i not in {di[0] for di in drop_indices}]
        s3['packets'] = kept
        s3['total_packets'] = len(kept)
        with open(s3f, 'w', encoding='utf-8') as f:
            json.dump(s3, f, indent=2, ensure_ascii=False)
        print(f"  saved: {s3f}  (was {len(pkts)} pkts -> now {len(kept)})")
    return len(drop_indices)


if __name__ == '__main__':
    DRY = '--dry' in sys.argv
    targets = [a for a in sys.argv[1:] if not a.startswith('--')]
    if not targets:
        # Default: walk ALL jobs
        targets = sorted(os.listdir(RESULTS))
    total = 0
    for jid in targets:
        if not os.path.isdir(os.path.join(RESULTS, jid)):
            continue
        total += patch_job(jid, dry=DRY)
    print(f"\nTotal duplicate Rider packets {'detected' if DRY else 'removed'}: {total}")
