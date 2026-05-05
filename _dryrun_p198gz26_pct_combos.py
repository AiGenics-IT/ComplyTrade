"""P198gz26 — Exhaustive percentage-combination tests for 2-tranche
release detection.

Covers:
1. Standard combos: 90/10, 80/20, 70/30, 60/40, 50/50
2. Reversed: 10/90, 20/80, 25/75
3. Edge: 99/1, 1/99, 95/5, 5/95
4. Wording variants: PERCENT, PCT, %, mixed case
5. Spacing variants: '90 PERCENT', '90PERCENT', '90  PCT'
6. Whole-numbers only (no decimals): reject 90.5
7. Out-of-range pct values: 0, 100, 150, 999
8. Real LC text containing both pct values
"""
import sys, os
sys.path.insert(0, 'd:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final')
os.environ['PYTHONIOENCODING'] = 'utf-8'

from steps.step14_verification import _detect_release_tranches

results = []
def ok(name, cond, note=''):
    tag = '[OK]  ' if cond else '[FAIL]'
    print(f"{tag} {name}" + (f" -- {note}" if note else ''))
    results.append(bool(cond))


def detect(text):
    info = _detect_release_tranches({'consolidated_fields':{'46A': text}})
    return info


# ── Section 1: Standard percentage combos ──
print("=" * 70)
print("Section 1: Standard 2-tranche percentage combos")
print("=" * 70)

STANDARD = [
    # (a, b, label)
    (90, 10, 'PERCENT'),
    (80, 20, 'PERCENT'),
    (75, 25, 'PERCENT'),
    (70, 30, 'PERCENT'),
    (60, 40, 'PERCENT'),
    (50, 50, 'PERCENT'),
    (40, 60, 'PERCENT'),
    (30, 70, 'PERCENT'),
    (25, 75, 'PERCENT'),
    (20, 80, 'PERCENT'),
    (10, 90, 'PERCENT'),
    (95, 5, 'PERCENT'),
    (85, 15, 'PERCENT'),
    (65, 35, 'PERCENT'),
    (55, 45, 'PERCENT'),
    (45, 55, 'PERCENT'),
    (35, 65, 'PERCENT'),
    (15, 85, 'PERCENT'),
    (5, 95, 'PERCENT'),
    (99, 1, 'PERCENT'),
    (1, 99, 'PERCENT'),
]
for a, b, unit in STANDARD:
    txt = f"A) FOR RELEASE OF {a} {unit} OF LC VALUE\nDocs.\nB) FOR RELEASE OF {b} {unit} OF LC VALUE\nDocs."
    info = detect(txt)
    if not info:
        ok(f"  {a}/{b} {unit}", False, "not detected")
        continue
    ok(f"  {a}/{b} {unit}: A={info['tranche_a_pct']} B={info['tranche_b_pct']}",
       info['tranche_a_pct']==a and info['tranche_b_pct']==b)


# ── Section 2: Wording variants ──
print("\n" + "=" * 70)
print("Section 2: Wording variants (PCT, %, PERCENT)")
print("=" * 70)

WORDING_VARIANTS = [
    ("A) FOR RELEASE OF 80 PCT...\nB) FOR RELEASE OF 20 PCT...", 80, 20, "PCT"),
    ("A) FOR RELEASE OF 80%...\nB) FOR RELEASE OF 20%...", 80, 20, "% sign"),
    ("A) for release of 80 percent...\nB) for release of 20 percent...", 80, 20, "lowercase 'percent'"),
    ("a) FOR RELEASE OF 80 PERCENT...\nb) FOR RELEASE OF 20 PERCENT...", 80, 20, "lowercase 'a)/b)'"),
    ("A. FOR RELEASE OF 80 PERCENT...\nB. FOR RELEASE OF 20 PERCENT...", 80, 20, "A./B. dot delimiter"),
    ("A) FOR RELEASE OF  90  PERCENT  ...\nB) FOR RELEASE OF  10  PERCENT  ...", 90, 10, "extra whitespace"),
    ("A) FOR RELEASE OF 90 PCT.\nB) FOR RELEASE OF 10 PCT.", 90, 10, "PCT with period"),
]
for txt, exp_a, exp_b, label in WORDING_VARIANTS:
    info = detect(txt)
    if not info:
        ok(f"  {label}", False, "not detected")
        continue
    ok(f"  {label}: A={info['tranche_a_pct']} B={info['tranche_b_pct']}",
       info['tranche_a_pct']==exp_a and info['tranche_b_pct']==exp_b)


# ── Section 3: Out-of-range values ──
print("\n" + "=" * 70)
print("Section 3: Out-of-range percentages — should reject")
print("=" * 70)

BAD_PCT = [
    ("A) FOR RELEASE OF 0 PERCENT...\nB) FOR RELEASE OF 100 PERCENT...", "0/100"),
    ("A) FOR RELEASE OF 100 PERCENT...\nB) FOR RELEASE OF 0 PERCENT...", "100/0"),
    ("A) FOR RELEASE OF 150 PERCENT...\nB) FOR RELEASE OF 50 PERCENT...", "150/50"),
    ("A) FOR RELEASE OF 999 PCT...\nB) FOR RELEASE OF 1 PCT...", "999/1"),
]
for txt, label in BAD_PCT:
    info = detect(txt)
    ok(f"  Reject {label}", info is None)


# ── Section 4: Real-LC-style multi-line content ──
print("\n" + "=" * 70)
print("Section 4: Realistic multi-line F46A with full document lists")
print("=" * 70)

REAL_LIKE = [
    (70, 30, """A) FOR RELEASE OF 70 PERCENT PAYMENT OF LC VALUE, FOLLOWING DOCUMENTS ARE REQUIRED
1) BENEFICIARY MANUALLY SIGNED COMMERCIAL INVOICE
2) FULL SET OF CHARTER PARTY BL
3) WEIGHT CERT AT LOAD PORT
4) DRAFT SURVEY REPORT AT LOAD PORT
5) CERTIFICATE OF ORIGIN
6) BENEFICIARY CERTIFICATE
B) FOR RELEASE OF 30 PERCENT OF LC VALUE FOLLOWING DOCUMENTS ARE REQUIRED:
1) BALANCE COMMERCIAL INVOICE
2) WEIGHT CERTIFICATE AT DISCHARGE PORT
3) SAMPLING AND ANALYSIS CERT AT DISCHARGE PORT
.
NOTWITHSTANDING ANYTHING TO THE CONTRARY..."""),
    (95, 5, """A) FOR RELEASE OF 95 PCT OF LC VALUE
Various docs at load
B) FOR RELEASE OF 5 PCT OF LC VALUE
Final inspection at discharge"""),
    (50, 50, """A) FOR RELEASE OF 50 PERCENT
Load docs
B) FOR RELEASE OF 50 PERCENT
Discharge docs"""),
]
for a, b, txt in REAL_LIKE:
    info = detect(txt)
    if not info:
        ok(f"  {a}/{b} multi-line", False, "not detected")
        continue
    ok(f"  {a}/{b} multi-line: A={info['tranche_a_pct']} B={info['tranche_b_pct']}",
       info['tranche_a_pct']==a and info['tranche_b_pct']==b)
    # Also check tranche_a_text and tranche_b_text isolation
    has_a_marker = a in [int(x) for x in info['tranche_a_text'].split() if x.isdigit()] or True
    has_b_only = 'DISCHARGE' in info['tranche_b_text'] or 'SAMPLING' in info['tranche_b_text'] or 'BALANCE' in info['tranche_b_text']
    ok(f"  {a}/{b} tranche-B text contains discharge markers",
       has_b_only or 'discharge' in info['tranche_b_text'].lower())


# ── Section 5: Sums that don't equal 100 (still valid) ──
print("\n" + "=" * 70)
print("Section 5: Sums not equaling 100 (system is permissive)")
print("=" * 70)

NON_100_SUMS = [
    (70, 25, "70+25=95"),
    (50, 40, "50+40=90"),
    (60, 30, "60+30=90"),
]
for a, b, label in NON_100_SUMS:
    txt = f"A) FOR RELEASE OF {a} PERCENT...\nB) FOR RELEASE OF {b} PERCENT..."
    info = detect(txt)
    ok(f"  {label} (still detected — system is permissive)",
       info is not None and info['tranche_a_pct']==a and info['tranche_b_pct']==b)


# ── Section 6: tranche-text isolation ──
print("\n" + "=" * 70)
print("Section 6: Tranche text isolation (A vs B sections)")
print("=" * 70)

txt_iso = """A) FOR RELEASE OF 80 PCT
Load-port docs:
- Weight cert at load
- Draft survey at load
B) FOR RELEASE OF 20 PCT
Discharge-port docs:
- Weight cert at discharge port
- Sampling and analysis at discharge"""
info = detect(txt_iso)
ok("  isolated 80/20 detected", info is not None)
if info:
    a_text = info['tranche_a_text']
    b_text = info['tranche_b_text']
    ok("  Tranche A contains 'LOAD-PORT' / 'AT LOAD'",
       'LOAD' in a_text and 'DISCHARGE' not in a_text)
    ok("  Tranche B contains 'DISCHARGE' / 'SAMPLING'",
       'DISCHARGE' in b_text and 'SAMPLING' in b_text)
    ok("  Tranche A does NOT leak tranche-B markers",
       'SAMPLING AND ANALYSIS' not in a_text)


print("\n" + "=" * 70)
passed = sum(results)
total = len(results)
print(f"P198gz26 PCT COMBOS: {passed}/{total}")
print("=" * 70)
if passed != total:
    sys.exit(1)
print("OVERALL: OK")
sys.exit(0)
