"""
P198fo + P198fp dry-run.

P198fo — SWIFT direct-pattern continuation:
  When step03 detects multiple consecutive pages as the same SWIFT
  type via direct pattern matching (no BAHL header), the second-and-
  beyond pages now correctly mark is_continuation=True so the packet
  grouper merges them into ONE LC packet instead of N separate
  packets.

P198fp — Step 9 reclassification veto:
  When step09 wants to reclassify a packet from a specific commodity
  / cert name (e.g. 'COAL SPECIFICATIONS AT THE LOADING PORT') to a
  generic bucket (e.g. 'Inspection Certificate'), the veto kicks in
  and KEEPS the original specific name. The trade-specific name is
  the trade-finance truth — collapsing it loses context and breaks
  the coal-quality verifier (P198fb).

Real anchors:
  ff87b18c — pages 1-4 are one MT700 LC, currently split into 2 packets
  ff87b18c — page 41 is "COAL SPECIFICATIONS..." reclassified to
              "Inspection Certificate" by step09
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


# ──────────────────────────────────────────────────────────────────────
# P198fo — SWIFT continuation logic
# ──────────────────────────────────────────────────────────────────────
print("--- P198fo: SWIFT continuation tracking ---")

# Mirror the production logic
def simulate_swift_pages(page_swift_types):
    """Simulates the production loop in step03_sequencing.py.
    page_swift_types is a list of (page_num, swift_type) tuples in
    PDF order. Returns list of (page_num, doc_type, is_continuation)."""
    out = []
    prev_swift_type = None
    SWIFT_TYPES = {'Amendment', 'LC', 'MT799', 'MT999'}
    for pg, st in page_swift_types:
        is_cont = bool(prev_swift_type) and prev_swift_type == st
        if st in SWIFT_TYPES or (st.startswith('MT') and st not in ('MT799', 'MT999')):
            out.append((pg, st, is_cont))
            prev_swift_type = st
    return out

# Test 1: Single LC across 4 pages — should produce ONE packet
res = simulate_swift_pages([(1,'LC'),(2,'LC'),(3,'LC'),(4,'LC')])
assert_eq("4-page LC: page 1 is_continuation=False", res[0][2], False)
assert_eq("4-page LC: page 2 is_continuation=True",  res[1][2], True)
assert_eq("4-page LC: page 3 is_continuation=True",  res[2][2], True)
assert_eq("4-page LC: page 4 is_continuation=True",  res[3][2], True)

# Test 2: LC, then Amendment — first amendment page should NOT be cont
res = simulate_swift_pages([(1,'LC'),(2,'LC'),(3,'Amendment'),(4,'Amendment')])
assert_eq("LC->Amendment: amendment p3 is_continuation=False", res[2][2], False)
assert_eq("LC->Amendment: amendment p4 is_continuation=True",  res[3][2], True)

# Test 3: LC interleaved with MT799
res = simulate_swift_pages([(1,'LC'),(2,'LC'),(3,'MT799'),(4,'LC')])
assert_eq("LC->MT799->LC: MT799 p3 is_continuation=False", res[2][2], False)
assert_eq("LC->MT799->LC: LC p4 is_continuation=False (different prev)",
          res[3][2], False)

# Test 4: Single page LC
res = simulate_swift_pages([(1,'LC')])
assert_eq("Single-page LC: not continuation", res[0][2], False)

# Test 5: BAHL informational MT types
res = simulate_swift_pages([(1,'MT730'),(2,'MT730'),(3,'MT700' if False else 'MT730')])
assert_eq("Two MT730 pages: page 2 is continuation",
          res[1][2], True)


# ──────────────────────────────────────────────────────────────────────
# P198fp — Reclassification veto
# ──────────────────────────────────────────────────────────────────────
print("\n--- P198fp: Reclassification veto ---")
_SPECIFIC_MARKERS = (
    'COAL', 'IRON ORE', 'PETROLEUM', 'CRUDE',
    'SUGAR', 'WHEAT', 'BARLEY', 'CORN', 'RICE',
    'GRAIN', 'OILSEED', 'PALM', 'FERTILIZER',
    'UREA', 'CEMENT', 'CLINKER', 'STEEL',
    'PETCOKE', 'LIGNITE', 'BITUMINOUS',
    'DRAFT SURVEY', 'SAMPLING', 'ANALYSIS',
    'WEIGHT', 'CALORIFIC', 'PROXIMATE',
    'ULTIMATE', 'SPECIFICATION',
)
_GENERIC_BUCKETS = (
    'INSPECTION CERTIFICATE',
    'QUALITY CERTIFICATE',
    'TEST CERTIFICATE',
    'GENERIC CERTIFICATE',
    'CERTIFICATE',
    'SURVEY REPORT',
    'INSPECTION REPORT',
    'QUANTITY CERTIFICATE',
)


def should_veto(orig, new):
    o = orig.upper(); n = new.upper()
    is_specific = any(m in o for m in _SPECIFIC_MARKERS)
    is_generic = any(g == n or n.endswith(' ' + g) or n.startswith(g + ' ') or n == g
                     for g in _GENERIC_BUCKETS)
    new_also_specific = any(m in n for m in _SPECIFIC_MARKERS)
    return is_specific and is_generic and not new_also_specific

# Veto cases — should veto the reclass
VETO_CASES = [
    ("COAL SPECIFICATIONS AT THE LOADING PORT -> Inspection Certificate",
     "COAL SPECIFICATIONS AT THE LOADING PORT", "Inspection Certificate", True),
    ("Coal Sampling and Analysis -> Quality Certificate",
     "Coal Sampling and Analysis", "Quality Certificate", True),
    ("Petroleum Inspection Report -> Inspection Certificate",
     "Petroleum Inspection Report", "Inspection Certificate", True),
    ("Sugar Quality Report -> Test Certificate",
     "Sugar Quality Report", "Test Certificate", True),
    ("Iron Ore Weight Certificate -> Quantity Certificate",
     "Iron Ore Weight Certificate", "Quantity Certificate", True),
    ("Draft Survey Report -> Survey Report",
     "Draft Survey Report (Loading)", "Survey Report", True),
    # NEGATIVE — should NOT veto
    ("Generic 'Quality Cert' -> 'Inspection Cert' (no specific marker in orig)",
     "Quality Certificate", "Inspection Certificate", False),
    ("Coal Spec -> Coal Quality Cert (specific to specific — no veto)",
     "Coal Specifications at Load Port", "Coal Quality Certificate", False),
    ("Banking doc reclass — no veto",
     "Documentary Remittance", "Covering Letter", False),
    ("Empty",
     "", "", False),
]
for name, orig, new, expected_veto in VETO_CASES:
    got = should_veto(orig, new)
    assert_eq(f"veto: {name}", got, expected_veto)


# ──────────────────────────────────────────────────────────────────────
# Source wiring checks
# ──────────────────────────────────────────────────────────────────────
print("\n--- Source wiring ---")
src3 = open('d:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final/steps/step03_sequencing.py',
            'r', encoding='utf-8').read()
assert_eq("P198fo: _is_swift_continuation helper present",
          '_is_swift_continuation' in src3, True)
assert_eq("P198fo: comment marker present",
          'P198fo' in src3, True)

src9 = open('d:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final/steps/step09_shipping_reconciliation.py',
            'r', encoding='utf-8').read()
assert_eq("P198fp: veto block present",
          'P198fp' in src9 and 'reclassify_vetoed' in src9, True)
assert_eq("P198fp: _SPECIFIC_MARKERS list present",
          '_SPECIFIC_MARKERS' in src9, True)
assert_eq("P198fp: _GENERIC_BUCKETS list present",
          '_GENERIC_BUCKETS' in src9, True)


# ──────────────────────────────────────────────────────────────────────
# Real-data check: ff87b18c pages 1-4 SHOULD be one packet now
# (verified by reading the page text and asserting it's all MT700 content)
# ──────────────────────────────────────────────────────────────────────
print("\n--- Real-data check: ff87b18c pages 1-4 are one MT700 ---")
JOB = 'd:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final/results/ff87b18c-473f-4c55-bd78-0545478da00c'
s2p = os.path.join(JOB, 'step02', 'step02_result.json')
if os.path.isfile(s2p):
    with open(s2p, 'r', encoding='utf-8') as f:
        s2 = json.load(f)
    pages_text = {}
    for p in s2.get('cleaned_pages') or s2.get('pages') or []:
        pn = p.get('page_number')
        if pn in (1, 2, 3, 4):
            pages_text[pn] = (p.get('cleaned_text') or p.get('raw_text') or '')
    # Page 1 should have BAHL/Alliance header + Identifier: fin.700
    assert_eq("ff87b18c page 1 has fin.700 marker",
              'fin.700' in (pages_text.get(1) or '').lower(), True)
    # Pages 2-4 should have MT700 F-tag content (not new MT messages)
    for pn in (2, 3, 4):
        txt = pages_text.get(pn) or ''
        has_ftags = any(t in txt for t in ('F42C:', 'F43P:', 'F44E:',
                                            'F45A:', 'F46A:', 'F47A:',
                                            'F48:', 'F49:', 'F71D:',
                                            'F78:', 'F72:', 'F44C:'))
        assert_eq(f"ff87b18c page {pn} has MT700 F-tag continuation content",
                  has_ftags, True)


passed = sum(results)
total = len(results)
print(f"\n{passed}/{total} cases passed")
if passed != total:
    sys.exit(1)
print("OVERALL: OK")
sys.exit(0)
