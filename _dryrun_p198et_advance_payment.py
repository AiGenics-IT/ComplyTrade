"""
P198et dry-run — Split-payment LC (advance + remainder) verification
context.

Validates:
  1. _detect_advance_payment_terms() correctly extracts advance%, net%,
     LC amount, currency, expected_advance, expected_net from F46A /
     F47A text.
  2. The ADVANCE-PAYMENT block is prepended to f47a_context so the
     verification LLM sees it before any per-field rule.
  3. The detector returns None for STANDARD 100%-on-presentation LCs
     (no false positives — previous scenarios must keep working).
  4. Real-job sweep on results/* — no spurious matches across the
     job store.

Real data anchor:
  Job 2d98b74c-457f-4456-8a85-68841190e4d5 (LC 0052ILC083930)
    F32B: USD 10,919.00
    F46A: "A) 80 PERCENT ADVANCE PAYMENT ... B) REMAINING 20 PERCENT ..."
    F47A clause 6: "100 PERCENT L/C VALUE 80 PERCENT ADVANCE PAYMENT
                    AND NET CLAIMING (20 PERCENT)"
  Expected detector output:
    advance_pct=80, net_pct=20, lc_amount=10919.0,
    expected_advance=8735.20, expected_net=2183.80
"""
import sys, os, json
sys.path.insert(0, 'd:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final')

from steps.step14_verification import (
    _detect_advance_payment_terms,
    _format_advance_payment_block,
    _build_f47a_context,
)

results = []
def assert_eq(name, got, expected):
    ok = (got == expected)
    print(f"[{'OK' if ok else 'FAIL'}] {name}")
    if not ok:
        print(f"          got     : {got!r}")
        print(f"          expected: {expected!r}")
    results.append(ok)

def assert_close(name, got, expected, tol=0.01):
    ok = (abs(got - expected) <= tol) if got is not None else False
    print(f"[{'OK' if ok else 'FAIL'}] {name}")
    if not ok:
        print(f"          got={got}  expected={expected}")
    results.append(ok)

# ── Test 1: Real job data (2d98b74c) ─────────────────────────────────────
JOB = 'd:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final/results/2d98b74c-457f-4456-8a85-68841190e4d5'
with open(os.path.join(JOB, 'step06', 'step06_result.json'), 'r',
          encoding='utf-8') as f:
    real_s6 = json.load(f)

info = _detect_advance_payment_terms(real_s6)
print("\n--- Test 1: real job 2d98b74c ---")
print(f"  detector output: {info}")
assert_eq("real job: is_advance_split detected",
          info is not None and info.get('is_advance_split'), True)
assert_eq("real job: advance_pct == 80", info['advance_pct'], 80)
assert_eq("real job: net_pct     == 20", info['net_pct'], 20)
assert_close("real job: lc_amount        ~ 10919.00", info['lc_amount'], 10919.00)
assert_close("real job: expected_advance ~ 8735.20",  info['expected_advance'], 8735.20)
assert_close("real job: expected_net     ~ 2183.80",  info['expected_net'], 2183.80)
assert_eq("real job: currency == USD", info['currency'], 'USD')

# ── Test 2: f47a_context contains the banner ────────────────────────────
ctx = _build_f47a_context(real_s6)
banner_present = 'ADVANCE-PAYMENT / SPLIT-PAYMENT LC' in ctx
print(f"\n--- Test 2: f47a_context banner ---")
print(f"  context starts with banner? {banner_present}")
assert_eq("real job: f47a_context contains advance-payment banner",
          banner_present, True)
assert_eq("real job: banner mentions 80% advance", '80%' in ctx, True)
assert_eq("real job: banner mentions 20% net", '20%' in ctx, True)
assert_eq("real job: banner mentions USD 8,735.20", 'USD 8,735.20' in ctx, True)
assert_eq("real job: banner mentions USD 2,183.80', 'USD 2,183.80' in ctx",
          'USD 2,183.80' in ctx, True)
# F47A original content (clause 1, etc.) must STILL be in the context
assert_eq("real job: original F47A content preserved",
          'STALE, CLAUSED' in ctx.upper() or 'STALE' in ctx.upper(), True)

# ── Test 3: NO advance-payment scenarios (must return None) ─────────────
def make_lc(f46a='', f47a='', f32b='USD 100,000.00'):
    return {
        'consolidated_fields': {
            '32B': f32b,
            '46A': f46a,
            '47A': f47a,
        }
    }

# Standard sight LC, no split
none_cases = [
    ("standard sight LC, no F46A/F47A advance",
     make_lc('FULL SET BL ENDORSED IN BLANK', 'TPND 5/5 LATE SHIPMENT NOT ACCEPTABLE')),
    ("F47A says 100% L/C VALUE — not a split",
     make_lc('', 'INVOICE FOR 100 PERCENT L/C VALUE')),
    ("F46A is just docs, no payment terms",
     make_lc('1. INVOICE 2. BL 3. PACKING LIST', '')),
    ("F47A says 100 PCT covering schedule (no advance)",
     make_lc('', 'COVERING SCHEDULE MUST SHOW 100 PERCENT L/C VALUE')),
    ("empty fields",
     make_lc('', '')),
    ("None step06",
     None),
    ("non-dict step06", "garbage"),
]
print(f"\n--- Test 3: NO false positives on standard LCs ---")
for name, lc in none_cases:
    got = _detect_advance_payment_terms(lc) if isinstance(lc, dict) else \
          _detect_advance_payment_terms(lc)
    ok = (got is None)
    print(f"[{'OK' if ok else 'FAIL'}] {name}: got={got}")
    results.append(ok)

# ── Test 4: synthetic split-payment variants ────────────────────────────
print(f"\n--- Test 4: synthetic advance-payment variants ---")

# Variant A: 70/30 split
sc_a = make_lc(
    'A) 70 PERCENT ADVANCE PAYMENT MADE UPON SWIFT '
    'B) REMAINING 30 PERCENT PAYABLE AGAINST DOCS', '',
    'EUR 50,000.00'
)
ia = _detect_advance_payment_terms(sc_a)
assert_eq("70/30: advance_pct=70", ia['advance_pct'], 70)
assert_eq("70/30: net_pct=30",     ia['net_pct'], 30)
assert_close("70/30: expected_advance=35000", ia['expected_advance'], 35000.00)
assert_close("70/30: expected_net=15000",     ia['expected_net'],     15000.00)
assert_eq("70/30: currency=EUR", ia['currency'], 'EUR')

# Variant B: PCT instead of PERCENT
sc_b = make_lc(
    '50 PCT ADVANCE PAYMENT WILL BE MADE BY SWIFT '
    'REMAINING 50 PCT AGAINST PRESENTATION', '',
    'USD 1,000,000'
)
ib = _detect_advance_payment_terms(sc_b)
assert_eq("PCT 50/50: advance_pct=50", ib['advance_pct'], 50)
assert_eq("PCT 50/50: net_pct=50",     ib['net_pct'], 50)

# Variant C: F47A only (advance terms in F47A, F46A empty)
sc_c = make_lc('', 'COVERING SCHEDULE: 100 PERCENT VALUE LESS '
               '60 PERCENT ADVANCE PAYMENT — NET CLAIMING (40 PERCENT)',
               'PKR 500,000.00')
ic = _detect_advance_payment_terms(sc_c)
assert_eq("F47A-only 60/40: advance_pct=60", ic['advance_pct'], 60)
assert_eq("F47A-only 60/40: net_pct=40",     ic['net_pct'], 40)

# Variant D: only "LESS X PERCENT ADVANCE" phrasing
sc_d = make_lc('SHIPPING INVOICE TO BE DRAWN FOR 100 PERCENT '
               'L/C AMOUNT LESS 90 PERCENT ADVANCE PAYMENT', '',
               'USD 200,000')
id_ = _detect_advance_payment_terms(sc_d)
assert_eq("LESS-only 90/10: advance_pct=90", id_['advance_pct'], 90)
assert_eq("LESS-only 90/10: net_pct=10",     id_['net_pct'], 10)

# Variant E: out-of-range percent (must reject)
sc_e = make_lc('150 PERCENT ADVANCE PAYMENT', '', 'USD 1000')
ie = _detect_advance_payment_terms(sc_e)
assert_eq("150% out-of-range: returns None", ie, None)

# Variant F: 0 LC amount (must reject — can't compute splits)
sc_f = make_lc('80 PERCENT ADVANCE', '', '')
inf = _detect_advance_payment_terms(sc_f)
assert_eq("missing LC amount: returns None", inf, None)

# ── Test 5: real job sweep — count split-payment LCs across results/* ──
print(f"\n--- Test 5: real-data sweep across local results/ ---")
RESULTS_DIR = 'd:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final/results'
total = 0; split_count = 0; standard_count = 0
errors = 0
for jid in sorted(os.listdir(RESULTS_DIR)):
    s6f = os.path.join(RESULTS_DIR, jid, 'step06', 'step06_result.json')
    if not os.path.isfile(s6f):
        continue
    try:
        with open(s6f, 'r', encoding='utf-8') as f:
            s6 = json.load(f)
    except Exception:
        continue
    total += 1
    try:
        det = _detect_advance_payment_terms(s6)
    except Exception as e:
        errors += 1
        continue
    if det:
        split_count += 1
        print(f"  [SPLIT] {jid}: {det['advance_pct']}/{det['net_pct']}, "
              f"LC={det['currency']} {det['lc_amount']:,.2f}, "
              f"net={det['expected_net']:,.2f}")
    else:
        standard_count += 1
print(f"  totals: {total} jobs scanned | "
      f"{split_count} split-payment | {standard_count} standard | {errors} errors")
assert_eq("real-data sweep: no detector exceptions", errors, 0)
assert_eq("real-data sweep: at least 1 split-payment LC found",
          split_count >= 1, True)

passed = sum(results)
total_t = len(results)
print(f"\n{passed}/{total_t} cases passed")
if passed != total_t:
    sys.exit(1)
print("OVERALL: OK")
sys.exit(0)
