"""
P198eu dry-run — Rule 1b must REMOVE absorbed Attach-List packets,
not just extend the BL packet's page list.

Bug:
  After P190's Rule 1b "Attach List -> nearest BL" merge, the
  absorbed packet was added to `_consumed` but the list filter that
  was supposed to drop it never ran (the `_consumed = set()` reset
  for Rule 3 happened immediately after). The result was duplicate
  packets — one "Bill of Lading + Attached List" containing pages
  [21, 22] AND a separate "Attached Rider" containing page [22].

Fix: filter out Rule 1b's `_consumed` indices BEFORE the reset.

Real-data anchor:
  Job 2d98b74c-457f-4456-8a85-68841190e4d5
    Without fix: 7 BL+Attached pairs each with a duplicate Rider packet.
    With fix:    7 BL+Attached packets, no duplicate Rider packets.
"""
import sys, os, json
sys.path.insert(0, 'd:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final')

results = []
def assert_eq(name, got, expected):
    ok = (got == expected)
    print(f"[{'OK' if ok else 'FAIL'}] {name}")
    if not ok:
        print(f"          got     : {got!r}")
        print(f"          expected: {expected!r}")
    results.append(ok)

# ── Static check on the source: filter must be present BEFORE the reset ──
src = open('d:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final/steps/step03_sequencing.py',
           'r', encoding='utf-8').read()
# The fix block contains a list-comp filtering merged_packets by _consumed
has_filter = ('merged_packets = [p for i, p in enumerate(merged_packets)'
              in src and 'if i not in _consumed' in src)
assert_eq("source: Rule 1b filter present before _consumed reset", has_filter, True)

# Reset must come AFTER the filter
filter_idx = src.find('merged_packets = [p for i, p in enumerate(merged_packets)')
reset_idx  = src.find('_consumed = set()  # Reset for Rule 3')
assert_eq("source: filter occurs before Rule 3 reset",
          filter_idx > 0 and filter_idx < reset_idx, True)

# ── Real-data sweep — count duplicate Attach Rider packets across saved jobs ──
print()
print("=== Real-data sweep ===")
RESULTS_DIR = 'd:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final/results'
def is_bl_attach_label(dt):
    dtl = (dt or '').lower().strip()
    return any(k in dtl for k in (
        'attach list', 'attached list', 'attached sheet', 'attached rider',
        'bl attached sheet', 'bl rider', 'rider', 'attached schedule',
        'specification of cargo', 'cargo specification', 'attached specification',
    )) and '+' not in dtl  # exclude already-merged "BL + Attached List"

dup_jobs = 0
total_dups = 0
for jid in sorted(os.listdir(RESULTS_DIR)):
    s3f = os.path.join(RESULTS_DIR, jid, 'step03', 'step03_result.json')
    if not os.path.isfile(s3f):
        continue
    try:
        with open(s3f, 'r', encoding='utf-8') as f:
            s3 = json.load(f)
    except Exception:
        continue
    pkts = s3.get('packets') or []
    # Build BL+Attached page sets
    bl_pages = set()
    for p in pkts:
        if '+ attached' in (p.get('document_type','') or '').lower():
            bl_pages.update(p.get('page_numbers') or [])
    # Find Attach Rider packets whose pages are ALREADY in a BL+Attached pkt
    job_dups = []
    for p in pkts:
        if not is_bl_attach_label(p.get('document_type','')):
            continue
        pgs = p.get('page_numbers') or []
        if any(pg in bl_pages for pg in pgs):
            job_dups.append((p.get('packet_id'), pgs, p.get('document_type')))
    if job_dups:
        dup_jobs += 1
        total_dups += len(job_dups)
        print(f"  [DUP] {jid}: {len(job_dups)} duplicate(s)")
        for pid, pgs, dt in job_dups[:3]:
            print(f"        {pid} pages={pgs} dt={dt}")

print(f"\n  Jobs with duplicate-Rider symptom: {dup_jobs}")
print(f"  Total duplicate Rider packets    : {total_dups}")
print(f"  (These will disappear after step03 re-runs with the P198eu fix.)")
print()
# This sweep is informational, not an assertion (the saved JSONs were
# written before the fix; only the source check above asserts the fix
# is in the code).
print(f"\n{sum(results)}/{len(results)} cases passed")
sys.exit(0 if all(results) else 1)
