"""P198gz32 — Aggressive testing of the 2-tranche cover-amount rescue.

Tests _hybrid_amount_check's cover_vs_invoice branch with tranche_info
populated, across many percentage splits, both tranches, edge tolerances,
and adversarial cases that should NOT be rescued.
"""
import sys, os
sys.path.insert(0, 'd:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final')
os.environ['PYTHONIOENCODING'] = 'utf-8'

from steps.step14_implicit import _hybrid_amount_check
from steps.step14_verification import _detect_release_tranches

results = []
def ok(name, cond, note=''):
    tag = '[OK]  ' if cond else '[FAIL]'
    print(f"{tag} {name}" + (f" -- {note}" if note else ''))
    results.append(bool(cond))


def make_cover_pkt(amount, currency='USD'):
    """Mock a Documentary Remittance / cover-schedule packet."""
    return {
        'document_type': 'Documentary Remittance',
        'document_amount': f'{currency} {amount:,.2f}',
        'document_text': f'Documents value: {currency} {amount:,.2f}',
        'unified_summary': {
            'amounts_found': [
                {'role': 'documents_value', 'value': f'{amount:,.2f}',
                 'currency': currency},
            ]
        },
    }


def run_cover_check(cover_amt, inv_total, tranche_a_pct, tranche_b_pct,
                    currency='USD'):
    """Run _hybrid_amount_check for cover_vs_invoice with tranche_info."""
    pkt = make_cover_pkt(cover_amt, currency)
    tranche_info = {
        'is_two_tranche': True,
        'tranche_a_pct': tranche_a_pct,
        'tranche_b_pct': tranche_b_pct,
    }
    inv_str = f'{currency} {inv_total:,.2f}'
    return _hybrid_amount_check(
        lc_amount=inv_total * 1.10, lc_currency=currency,
        tol_plus=10.0, tol_minus=10.0,
        pkt=pkt, check_id='amount_currency',
        check_type='cover_vs_invoice',
        inv_amounts_str=inv_str,
        advance_info=None,
        tranche_info=tranche_info,
    )


# ─────────────────────────────────────────────────────────────────
# Section 1: Standard 90/10 coal LC at various invoice values
# ─────────────────────────────────────────────────────────────────
print("=" * 70)
print("Section 1: Standard 90/10 — varied invoice totals")
print("=" * 70)
INV_VALS = [
    1_000_000.00, 4_667_984.28, 12_345_678.90, 567_890.12,
    99_999_999.99, 50_000.00,
]
for inv in INV_VALS:
    cover_a = round(inv * 0.90, 2)
    r = run_cover_check(cover_a, inv, 90, 10)
    ok(f"  90/10 inv={inv:,.2f} cover={cover_a:,.2f} (90%)",
       r.compliance == 'PASS' and 'tranche-A' in r.result)
    cover_b = round(inv * 0.10, 2)
    r = run_cover_check(cover_b, inv, 90, 10)
    ok(f"  90/10 inv={inv:,.2f} cover={cover_b:,.2f} (10%)",
       r.compliance == 'PASS' and 'tranche-B' in r.result)


# ─────────────────────────────────────────────────────────────────
# Section 2: Various tranche splits
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Section 2: Varied tranche splits")
print("=" * 70)
SPLITS = [(80, 20), (70, 30), (60, 40), (75, 25),
          (95, 5), (85, 15), (65, 35), (55, 45)]
INV = 5_000_000.00
for a_pct, b_pct in SPLITS:
    cover_a = round(INV * a_pct / 100, 2)
    cover_b = round(INV * b_pct / 100, 2)
    r1 = run_cover_check(cover_a, INV, a_pct, b_pct)
    r2 = run_cover_check(cover_b, INV, a_pct, b_pct)
    ok(f"  {a_pct}/{b_pct} tranche-A {cover_a:,.2f}",
       r1.compliance == 'PASS' and 'tranche-A' in r1.result)
    ok(f"  {a_pct}/{b_pct} tranche-B {cover_b:,.2f}",
       r2.compliance == 'PASS' and 'tranche-B' in r2.result)
# Symmetric 50/50: A and B match same amount, code returns tranche-A first
r = run_cover_check(2_500_000.00, INV, 50, 50)
ok("  50/50 symmetric → PASS (matches first matching tranche)",
   r.compliance == 'PASS' and 'tranche-' in r.result)


# ─────────────────────────────────────────────────────────────────
# Section 3: Tolerance edges (within 0.5% should PASS)
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Section 3: Tolerance edge cases")
print("=" * 70)
INV = 1_000_000.00
exp_a = INV * 0.90  # 900_000
TOL_CASES = [
    ('exact', 900000.00, True),
    ('+0.49 (within tol)', 900000.49, True),
    ('-0.49 (within tol)', 899999.51, True),
    ('+5,000 (within 0.5%)', 904500.00, True),  # 0.5% of 900k = 4500
    ('-5,000 (within 0.5%)', 895500.00, True),
    ('+10,000 (above 0.5%)', 910000.00, False),
    ('-10,000 (above 0.5%)', 890000.00, False),
    ('off by 50k (well above tol)', 950000.00, False),
]
for label, amt, should_pass in TOL_CASES:
    r = run_cover_check(amt, INV, 90, 10)
    if should_pass:
        ok(f"  {label} cover={amt:,.2f}",
           r.compliance == 'PASS' and 'tranche' in r.result.lower())
    else:
        ok(f"  {label} cover={amt:,.2f} should NOT match tranche",
           r.compliance != 'PASS' or 'tranche' not in r.result.lower())


# ─────────────────────────────────────────────────────────────────
# Section 4: Adversarial / negative cases — should NOT rescue
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Section 4: Negative cases — must FAIL or skip rescue")
print("=" * 70)
INV = 1_000_000.00

# 4a: cover at 50% under 90/10 split — neither tranche → FAIL
r = run_cover_check(500_000.00, INV, 90, 10)
ok("  4a: 50% cover under 90/10 split → FAIL",
   r.compliance == 'FAIL' and 'mismatch' in r.result.lower())

# 4b: cover at 33% under 90/10 split → FAIL
r = run_cover_check(333_333.33, INV, 90, 10)
ok("  4b: 33% cover under 90/10 split → FAIL",
   r.compliance == 'FAIL')

# 4c: when tranche_info is None → falls back to strict check
pkt = make_cover_pkt(900_000.00)
r = _hybrid_amount_check(
    lc_amount=INV * 1.1, lc_currency='USD',
    tol_plus=10, tol_minus=10, pkt=pkt,
    check_id='amount_currency', check_type='cover_vs_invoice',
    inv_amounts_str=f'USD {INV:,.2f}',
    advance_info=None, tranche_info=None,
)
ok("  4c: tranche_info=None → standard mismatch FAIL",
   r.compliance == 'FAIL')

# 4d: when tranche_info is_two_tranche=False → falls back
r = _hybrid_amount_check(
    lc_amount=INV * 1.1, lc_currency='USD',
    tol_plus=10, tol_minus=10,
    pkt=make_cover_pkt(900_000.00),
    check_id='amount_currency', check_type='cover_vs_invoice',
    inv_amounts_str=f'USD {INV:,.2f}',
    advance_info=None,
    tranche_info={'is_two_tranche': False},
)
ok("  4d: is_two_tranche=False → strict mismatch FAIL",
   r.compliance == 'FAIL')

# 4e: invalid pct 0/100 → not rescued
r = run_cover_check(0.00, INV, 0, 100)
ok("  4e: pct=0 invalid → not rescued (FAIL or mismatch)",
   r.compliance != 'PASS' or 'tranche' not in r.result.lower())


# ─────────────────────────────────────────────────────────────────
# Section 5: Exact match — should still PASS via standard branch
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Section 5: Exact-match cover (no tranche needed) → PASS")
print("=" * 70)
INV = 1_000_000.00
r = run_cover_check(INV, INV, 90, 10)
ok("  Exact match cover==invoice → PASS",
   r.compliance == 'PASS')


# ─────────────────────────────────────────────────────────────────
# Section 6: F46A real-text → _detect_release_tranches integration
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Section 6: F46A → _detect_release_tranches integration")
print("=" * 70)
F46A_REAL = """A) FOR RELEASE OF 90 PERCENT PAYMENT OF LC VALUE, FOLLOWING DOCUMENTS ARE REQUIRED
1) BENEFICIARY MANUALLY SIGNED COMMERCIAL INVOICE
2) FULL SET OF BL
B) FOR RELEASE OF 10 PERCENT
1) BALANCE INVOICE
2) WEIGHT CERT AT DISCHARGE PORT"""
info = _detect_release_tranches({'consolidated_fields': {'46A': F46A_REAL}})
ok("  Detected as 2-tranche", info is not None and info.get('is_two_tranche'))
ok("  A=90, B=10", info['tranche_a_pct'] == 90 and info['tranche_b_pct'] == 10)

# Then run the cover check using detected info
INV = 4_667_984.28
COVER = 4_201_185.85  # exactly 90%
pkt = make_cover_pkt(COVER)
r = _hybrid_amount_check(
    lc_amount=INV * 1.1, lc_currency='USD',
    tol_plus=10, tol_minus=10, pkt=pkt,
    check_id='amount_currency', check_type='cover_vs_invoice',
    inv_amounts_str=f'USD {INV:,.2f}',
    advance_info=None, tranche_info=info,
)
ok("  Real F46A + real numbers → PASS",
   r.compliance == 'PASS' and 'tranche-A' in r.result)
ok("  Result has correct percentage breakdown",
   '90%' in r.result and '4,667,984.28' in r.result and
   '4,201,185.85' in r.result)


# ─────────────────────────────────────────────────────────────────
# Section 7: Non-2-tranche F46A should not trigger rescue
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Section 7: Single-tranche / no-tranche F46A → no rescue")
print("=" * 70)
F46A_PLAIN = """1) COMMERCIAL INVOICE IN 3 ORIGINALS
2) FULL SET OF BL TO ORDER OF BANK
3) PACKING LIST
4) CERTIFICATE OF ORIGIN"""
info = _detect_release_tranches({'consolidated_fields': {'46A': F46A_PLAIN}})
ok("  Plain F46A NOT detected as 2-tranche", info is None)

F46A_THREE = """A) FOR RELEASE OF 50 PERCENT
1) DOCS
B) FOR RELEASE OF 30 PERCENT
1) DOCS
C) FOR RELEASE OF 20 PERCENT
1) DOCS"""
info_3 = _detect_release_tranches({'consolidated_fields': {'46A': F46A_THREE}})
ok("  3-tranche LC detected as just A/B (current limitation acceptable)",
   info_3 is not None,
   f"a={info_3.get('tranche_a_pct') if info_3 else None} "
   f"b={info_3.get('tranche_b_pct') if info_3 else None}")


# ─────────────────────────────────────────────────────────────────
# Section 8: Currency variations
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Section 8: Different currencies")
print("=" * 70)
for ccy in ('USD', 'EUR', 'GBP', 'AED', 'CHF'):
    INV = 1_000_000.00
    r = run_cover_check(INV * 0.90, INV, 90, 10, currency=ccy)
    ok(f"  {ccy} 90% cover → PASS",
       r.compliance == 'PASS' and ccy in r.result)


# ─────────────────────────────────────────────────────────────────
# Section 9: Stress with very small / very large amounts
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Section 9: Stress amounts")
print("=" * 70)
STRESS_CASES = [
    (10_000.00, 9_000.00, 90, 10, True),   # small LC
    (100_000.00, 10_000.00, 90, 10, True), # 10%
    (999_999_999.99, 899_999_999.99, 90, 10, True),  # near-max
    (1_500_000.50, 1_350_000.45, 90, 10, True),  # cents precision
]
for inv, cover, a_pct, b_pct, should in STRESS_CASES:
    r = run_cover_check(cover, inv, a_pct, b_pct)
    if should:
        ok(f"  inv={inv:,.2f} cover={cover:,.2f} → PASS",
           r.compliance == 'PASS')
    else:
        ok(f"  inv={inv:,.2f} cover={cover:,.2f} → FAIL",
           r.compliance == 'FAIL')


# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
passed = sum(results)
total = len(results)
print(f"P198gz32 AGGRESSIVE: {passed}/{total}")
print("=" * 70)
if passed != total:
    sys.exit(1)
print("OVERALL: OK")
sys.exit(0)
