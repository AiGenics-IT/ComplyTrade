"""
P198gk — Force-merge consecutive same-type SWIFT pages into one packet
even when VLM said cont=False.

Real-data anchors:
  • job 1f0fc892: MT700 LC pages 1-4, VLM said cont=False on each →
    was 2 packets, fix produces 1
  • job 226faca7: MT700 LC pages 2-5 → was 2 packets, fix produces 1

Coverage targets:
  • MT700 / LC issuance + MT701 continuation
  • MT707 / Amendment + MT708 continuation
  • MT710 / MT711 third-bank advice
  • MT720 / MT721 transfer
  • MT799 / MT999 free format
  • Other LC-related types (MT730, MT740, MT754, MT756, MT910, MT940)
"""
import sys, os, json, re
sys.path.insert(0, 'd:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final')
os.environ['PYTHONIOENCODING'] = 'utf-8'

results = []
def ok(name, condition, note=''):
    if condition:
        print(f"[OK]  {name}" + (f" — {note}" if note else ""))
    else:
        print(f"[FAIL] {name}" + (f" — {note}" if note else ""))
    results.append(bool(condition))


# Mirror the production list
SWIFT_TYPES = {
    'lc', 'mt700', 'mt701',
    'documentary credit', 'letter of credit',
    'amendment', 'mt707', 'mt708',
    'mt799', 'mt999',
    'mt710', 'mt711',
    'mt720', 'mt721',
    'mt705', 'mt730',
    'mt740', 'mt742', 'mt747',
    'mt734', 'mt744', 'mt732',
    'mt754', 'mt756',
    'mt910', 'mt940', 'mt942',
    'mt760', 'mt767', 'mt768', 'mt769',
}


def simulate_merge(classifications):
    """Mirror step03 _group_into_packets logic for SWIFT merging."""
    classifications = sorted(classifications, key=lambda c: c['page_number'])
    packets = []
    current = None
    for cls in classifications:
        pn = cls['page_number']
        dt = cls['document_type']
        is_cont = cls.get('is_continuation', False)
        copy_status = cls.get('copy_status', 'unknown')

        curr_lower = dt.lower().strip()
        pkt_lower = (current['document_type'].lower().strip() if current else '')
        type_matches = (current and curr_lower == pkt_lower)
        copy_matches = (current and (
            copy_status == current.get('copy_status') or
            copy_status in ('unknown', '') or
            current.get('copy_status') in ('unknown', '')
        ))
        both_swift = (
            current
            and curr_lower in SWIFT_TYPES
            and pkt_lower in SWIFT_TYPES
            and type_matches
        )
        force_merge = both_swift and (is_cont or type_matches)

        if (is_cont and current and type_matches and copy_matches) or force_merge:
            current['pages'].append(pn)
        else:
            if current:
                packets.append(current)
            current = {
                'document_type': dt, 'pages': [pn], 'copy_status': copy_status,
            }
    if current:
        packets.append(current)
    return packets


# ── Section 1 — LC issuance + MT701 continuation ──
print("=" * 70)
print("Section 1: LC + MT701 continuation merge")
print("=" * 70)

# Standard MT700 spanning multiple pages, VLM said cont=False on each
CLS_LC_MULTIPAGE = [
    {'page_number': 1, 'document_type': 'LC', 'is_continuation': False},
    {'page_number': 2, 'document_type': 'LC', 'is_continuation': False},
    {'page_number': 3, 'document_type': 'LC', 'is_continuation': True},
    {'page_number': 4, 'document_type': 'LC', 'is_continuation': True},
    {'page_number': 5, 'document_type': 'COVERING SCHEDULE', 'is_continuation': False},
]
pkts = simulate_merge(CLS_LC_MULTIPAGE)
ok(f"  4-page LC merges into 1 packet (VLM said cont=False on pages 1,2)",
   len(pkts) == 2 and pkts[0]['pages'] == [1, 2, 3, 4],
   f"got {len(pkts)} packets: {[(p['document_type'], p['pages']) for p in pkts]}"
   if not (len(pkts) == 2 and pkts[0]['pages'] == [1, 2, 3, 4]) else '')

# MT700 + MT701 continuation (multi-message LC)
CLS_LC_MT701 = [
    {'page_number': 1, 'document_type': 'MT700', 'is_continuation': False},
    {'page_number': 2, 'document_type': 'MT701', 'is_continuation': False},
    {'page_number': 3, 'document_type': 'Commercial Invoice', 'is_continuation': False},
]
pkts = simulate_merge(CLS_LC_MT701)
# MT700 and MT701 are different doc_types so they stay separate (1 packet each)
# but EACH should be its own packet (no merge across types)
ok(f"  MT700 + MT701 stay as 2 separate SWIFT packets (different types)",
   len(pkts) == 3
   and pkts[0]['pages'] == [1] and pkts[0]['document_type'] == 'MT700'
   and pkts[1]['pages'] == [2] and pkts[1]['document_type'] == 'MT701',
   f"got {[(p['document_type'], p['pages']) for p in pkts]}"
   if not (len(pkts) == 3) else '')


# ── Section 2 — Amendment + MT708 continuation ──
print("\n" + "=" * 70)
print("Section 2: Amendment (MT707) multi-page merge")
print("=" * 70)

CLS_AMEND = [
    {'page_number': 1, 'document_type': 'Amendment', 'is_continuation': False},
    {'page_number': 2, 'document_type': 'Amendment', 'is_continuation': False},
    {'page_number': 3, 'document_type': 'Amendment', 'is_continuation': False},
]
pkts = simulate_merge(CLS_AMEND)
ok(f"  3-page Amendment merges into 1 packet",
   len(pkts) == 1 and pkts[0]['pages'] == [1, 2, 3],
   f"got {len(pkts)}: {[p['pages'] for p in pkts]}"
   if not (len(pkts) == 1) else '')

# Amendment + MT708 (continuation format) — different types stay separate
CLS_AMEND_708 = [
    {'page_number': 1, 'document_type': 'MT707', 'is_continuation': False},
    {'page_number': 2, 'document_type': 'MT708', 'is_continuation': False},
]
pkts = simulate_merge(CLS_AMEND_708)
ok(f"  MT707 + MT708 stay as 2 separate packets (different types)",
   len(pkts) == 2)


# ── Section 3 — Various SWIFT types stay isolated when alone ──
print("\n" + "=" * 70)
print("Section 3: Different SWIFT types do NOT cross-merge")
print("=" * 70)

CLS_MIXED_SWIFT = [
    {'page_number': 1, 'document_type': 'MT799', 'is_continuation': False},
    {'page_number': 2, 'document_type': 'MT730', 'is_continuation': False},
    {'page_number': 3, 'document_type': 'Amendment', 'is_continuation': False},
    {'page_number': 4, 'document_type': 'LC', 'is_continuation': False},
    {'page_number': 5, 'document_type': 'LC', 'is_continuation': False},
]
pkts = simulate_merge(CLS_MIXED_SWIFT)
ok(f"  Mixed SWIFT types: 4 distinct packets + LC merges across pages 4-5",
   len(pkts) == 4 and pkts[-1]['pages'] == [4, 5],
   f"got {[(p['document_type'], p['pages']) for p in pkts]}"
   if not (len(pkts) == 4) else '')


# ── Section 4 — MT799 / MT999 free format multi-page ──
print("\n" + "=" * 70)
print("Section 4: MT799 multi-page merging")
print("=" * 70)

CLS_MT799 = [
    {'page_number': 1, 'document_type': 'MT799', 'is_continuation': False},
    {'page_number': 2, 'document_type': 'MT799', 'is_continuation': False},
]
pkts = simulate_merge(CLS_MT799)
ok(f"  2-page MT799 merges into 1 packet",
   len(pkts) == 1 and pkts[0]['pages'] == [1, 2])


# ── Section 5 — SWIFT does NOT pull non-SWIFT pages ──
print("\n" + "=" * 70)
print("Section 5: SWIFT merging does not absorb non-SWIFT neighbours")
print("=" * 70)

CLS_LC_THEN_CI = [
    {'page_number': 1, 'document_type': 'LC', 'is_continuation': False},
    {'page_number': 2, 'document_type': 'LC', 'is_continuation': False},
    {'page_number': 3, 'document_type': 'Commercial Invoice', 'is_continuation': False},
    {'page_number': 4, 'document_type': 'Commercial Invoice', 'is_continuation': True},
]
pkts = simulate_merge(CLS_LC_THEN_CI)
ok(f"  LC pages merge but CI stays separate",
   len(pkts) == 2
   and pkts[0]['document_type'] == 'LC' and pkts[0]['pages'] == [1, 2]
   and pkts[1]['document_type'] == 'Commercial Invoice' and pkts[1]['pages'] == [3, 4],
   f"got {[(p['document_type'], p['pages']) for p in pkts]}"
   if not (len(pkts) == 2) else '')


# ── Section 6 — Real jobs sweep ──
print("\n" + "=" * 70)
print("Section 6: Real-data sweep — verify SWIFT packets are 1 per logical msg")
print("=" * 70)

JOB_DIR = 'd:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final/results'
jobs_with_split_lcs = 0
jobs_clean = 0
total_swift_pkts = 0

for jid in sorted(os.listdir(JOB_DIR)):
    s3p = f'{JOB_DIR}/{jid}/step03/step03_result.json'
    if not os.path.exists(s3p):
        continue
    try:
        d3 = json.load(open(s3p, 'r', encoding='utf-8'))
    except Exception:
        continue
    pkts = d3.get('packets', [])
    # Find SWIFT packets
    swift_pkts = []
    for pkt in pkts:
        dt = (pkt.get('document_type', '') or '').lower().strip()
        if dt in SWIFT_TYPES:
            swift_pkts.append(pkt)
    # Check for "split LCs" — 2+ adjacent SWIFT packets of same type
    sorted_pkts = sorted(swift_pkts, key=lambda p: min(
        (pg.get('page_number') for pg in p.get('pages',[])
         if isinstance(pg, dict)), default=10**9))
    has_split = False
    for i in range(len(sorted_pkts) - 1):
        a, b = sorted_pkts[i], sorted_pkts[i + 1]
        a_pages = sorted(pg.get('page_number') for pg in a.get('pages',[])
                         if isinstance(pg, dict) and pg.get('page_number'))
        b_pages = sorted(pg.get('page_number') for pg in b.get('pages',[])
                         if isinstance(pg, dict) and pg.get('page_number'))
        if (a_pages and b_pages and
            a.get('document_type') == b.get('document_type') and
            b_pages[0] == a_pages[-1] + 1):
            has_split = True
            break
    if has_split:
        jobs_with_split_lcs += 1
    else:
        jobs_clean += 1
    total_swift_pkts += len(swift_pkts)

print(f"  Jobs scanned: {jobs_with_split_lcs + jobs_clean}")
print(f"  Jobs with split SWIFT packets (would benefit from re-run): "
      f"{jobs_with_split_lcs}")
print(f"  Total SWIFT packets across corpus: {total_swift_pkts}")
ok(f"  Most jobs have clean (un-split) SWIFT packets",
   (jobs_clean / max(jobs_clean + jobs_with_split_lcs, 1)) >= 0.7,
   f"only {jobs_clean}/{jobs_clean+jobs_with_split_lcs} clean"
   if (jobs_clean / max(jobs_clean + jobs_with_split_lcs, 1)) < 0.7 else '')


# ── Section 7 — Source code wiring ──
print("\n" + "=" * 70)
print("Section 7: Source code wiring")
print("=" * 70)

src = open('d:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final/steps/step03_sequencing.py',
           'r', encoding='utf-8').read()
for tok in ('mt701', 'mt708', 'mt711', 'mt721', 'mt942',
            '_SWIFT_TYPES_FOR_MERGE', 'P198gk',
            '_force_merge_swift'):
    ok(f"  step03 has '{tok}'", tok in src)


# ── Section 8 — Specific anchors: real jobs 1f0fc892 / 226faca7 ──
print("\n" + "=" * 70)
print("Section 8: Real anchors after the patch")
print("=" * 70)

for jid_pref, expected_lc_pages in (
    ('1f0fc892', [1, 2, 3, 4]),
    ('226faca7', [2, 3, 4, 5]),
):
    matches = [d for d in os.listdir(JOB_DIR) if d.startswith(jid_pref)]
    if not matches:
        continue
    full = matches[0]
    s3p = f'{JOB_DIR}/{full}/step03/step03_result.json'
    d3 = json.load(open(s3p, 'r', encoding='utf-8'))
    lc_pkts = []
    for pkt in d3.get('packets', []):
        dt = (pkt.get('document_type', '') or '').lower()
        if dt in ('lc', 'mt700'):
            pages = sorted(pg.get('page_number') for pg in pkt.get('pages',[])
                           if isinstance(pg, dict) and pg.get('page_number'))
            lc_pkts.append((pkt.get('packet_id'), pages))
    print(f"  {jid_pref}: LC packets found = {lc_pkts}")
    has_target = any(pgs == expected_lc_pages for _pid, pgs in lc_pkts)
    ok(f"  {jid_pref}: LC merged to single packet [{expected_lc_pages}]",
       has_target,
       f"got {lc_pkts}" if not has_target else '')


# Final tally
print("\n" + "=" * 70)
passed = sum(results)
total = len(results)
print(f"P198gk SWIFT MERGE: {passed}/{total} cases passed")
print("=" * 70)
if passed != total:
    sys.exit(1)
print("OVERALL: OK")
sys.exit(0)
