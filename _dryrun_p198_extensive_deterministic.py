"""
P198 EXTENSIVE deterministic dry-run — diverse LC types & edge cases.

Pure-Python (no LLM/VLM/server). Validates that the detectors and
context-builders behave correctly across the full spectrum of LC
patterns we've encountered. Runs in <2 seconds.

Test coverage by category:
  A. Standard sight LCs (no special detector should fire)
  B. Advance-payment LCs (P198et) — various %splits + currencies
  C. Coal LCs (P198fb) — GCV/NAR/petcoke/lignite + edge cases
  D. Discrepancy-whitelist LCs (P198ff)
  E. Late-shipment-with-penalty LCs (P198fg)
  F. Required-surveyor LCs (P198fh)
  G. Banner stacking (multiple detectors firing on same LC)
  H. Negative cases — LCs that look similar but should NOT trigger
"""
import sys, os, json
sys.path.insert(0, 'd:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final')

from steps.step14_verification import (
    _detect_advance_payment_terms,
    _detect_coal_quality_terms,
    _detect_discrepancy_whitelist,
    _detect_late_shipment_penalty,
    _detect_required_surveyors,
    _build_f47a_context,
)

results = []
def assert_eq(name, got, expected):
    ok = (got == expected)
    print(f"[{'OK' if ok else 'FAIL'}] {name}")
    if not ok:
        print(f"          got={got!r}  expected={expected!r}")
    results.append(ok)

def make_lc(f46a='', f47a='', f45a='', f32b='USD 100,000.00', f50='', f59=''):
    return {'consolidated_fields': {
        '32B': f32b, '46A': f46a, '47A': f47a, '45A': f45a,
        '50': f50, '59': f59,
    }}


# ────────────────────────────────────────────────────────────────────
# A. Standard sight LCs — NOTHING should fire
# ────────────────────────────────────────────────────────────────────
print("--- A. Standard sight LCs (no special detector should fire) ---")
A_CASES = [
    ("garments", make_lc(f47a='1) STANDARD CLAUSE\n2) THIRD PARTY DOCUMENTS ACCEPTABLE')),
    ("electronics", make_lc(f47a='1) PARTIAL ALLOWED\n2) FREIGHT FORWARDER B/L NOT ACCEPTABLE')),
    ("machinery", make_lc(f47a='1) USANCE 90 DAYS\n2) DOCUMENTS WITHIN 21 DAYS', f45a='HS 8415.1029 (machinery)')),
    ("chemicals", make_lc(f47a='1) MSDS REQUIRED\n2) DRAFT AT SIGHT')),
    ("rice", make_lc(f47a='1) STANDARD\n2) PHYTOSANITARY REQUIRED', f45a='HS 1006 BASMATI RICE')),
    ("fabric", make_lc(f47a='1) PARTIAL ALLOWED', f45a='HS 5407 POLYESTER FABRIC')),
]
for name, lc in A_CASES:
    assert_eq(f"A.{name}: advance-payment NOT detected",
              _detect_advance_payment_terms(lc) is None, True)
    assert_eq(f"A.{name}: coal NOT detected",
              _detect_coal_quality_terms(lc) is None, True)
    assert_eq(f"A.{name}: whitelist NOT detected",
              _detect_discrepancy_whitelist(lc) is None, True)
    assert_eq(f"A.{name}: late-pen NOT detected",
              _detect_late_shipment_penalty(lc) is None, True)
    assert_eq(f"A.{name}: surveyor NOT detected",
              _detect_required_surveyors(lc) is None, True)


# ────────────────────────────────────────────────────────────────────
# B. Advance-payment LCs — various split percentages
# ────────────────────────────────────────────────────────────────────
print("\n--- B. Advance-payment LCs (P198et) ---")
B_CASES = [
    ("80/20 USD", 80, 20, 'USD',
     make_lc(f46a='A) 80 PERCENT ADVANCE PAYMENT\nB) REMAINING 20 PERCENT', f32b='USD 10,919.00')),
    ("70/30 EUR", 70, 30, 'EUR',
     make_lc(f46a='A) 70 PCT ADVANCE\nB) REMAINING 30 PCT', f32b='EUR 50,000.00')),
    ("50/50 PKR", 50, 50, 'PKR',
     make_lc(f46a='50 PERCENT ADVANCE\nREMAINING 50 PERCENT AGAINST DOCS', f32b='PKR 5,000,000')),
    ("90/10 with LESS phrasing", 90, 10, 'USD',
     make_lc(f46a='SHIPPING INVOICE TO BE DRAWN FOR 100 PERCENT L/C AMOUNT '
                  'LESS 90 PERCENT ADVANCE PAYMENT', f32b='USD 200,000')),
    ("60/40 NET CLAIMING phrasing", 60, 40, 'USD',
     make_lc(f47a='COVERING SCHEDULE: 100 PERCENT VALUE LESS 60 PERCENT '
                  'ADVANCE — NET CLAIMING (40 PERCENT)', f32b='USD 500,000')),
    ("85/15 GCC AED", 85, 15, 'AED',
     make_lc(f46a='85 PCT ADVANCE PAYMENT\n15 PCT BALANCE', f32b='AED 750,000')),
]
for name, exp_adv, exp_net, exp_ccy, lc in B_CASES:
    info = _detect_advance_payment_terms(lc)
    assert_eq(f"B.{name}: detected", info is not None, True)
    if info:
        assert_eq(f"B.{name}: advance_pct={exp_adv}", info['advance_pct'], exp_adv)
        assert_eq(f"B.{name}: net_pct={exp_net}", info['net_pct'], exp_net)
        assert_eq(f"B.{name}: currency={exp_ccy}", info['currency'], exp_ccy)


# ────────────────────────────────────────────────────────────────────
# C. Coal LCs — GCV / petcoke / lignite / NAR / range / various
# ────────────────────────────────────────────────────────────────────
print("\n--- C. Coal LCs (P198fb) — diverse ---")
C_CASES = [
    ("Pakistani thermal 5800/5650",
     make_lc(f47a='17) PRICE ADJUSTMENTS CLAUSE\n'
                  'GROSS CALORIFIC VALUE (ARB) : 5,800 KCAL/KG BELOW 5650 KCAL/KG\n'
                  'TOTAL MOISTURE (ARB) : 11 PCT ABOVE 13 PCT\n'
                  'ASH (ARB) : 15 PCT ABOVE 17 PCT\n'
                  'SULPHUR (ARB) : 0.8 PCT ABOVE 1 PCT\n'
                  'ADJUSTED CFR = (FOB X actualGCV / 5800) + Freight',
             f45a='BITUMINOUS COAL HS 2701.1200', f32b='USD 526,250'),
     5800, 5650),
    ("Bangladesh NAR-based 5500/5300",
     make_lc(f47a='PRICE ADJUSTMENTS\n'
                  'NAR (NET AS RECEIVED) : 5500 KCAL/KG BELOW 5300 KCAL/KG\n'
                  'TM (ARB) : 14 PCT ABOVE 16 PCT\n'
                  'ADJUSTED CFR = (FOB X NAR / 5500) + Freight',
             f45a='COAL', f32b='USD 1,000,000'),
     5500, 5300),
    ("Indian high-rank 6500/6200",
     make_lc(f47a='PRICE ADJUSTMENTS CLAUSE\n'
                  'GCV (ARB) : 6500 KCAL/KG BELOW 6200 KCAL/KG\n'
                  'ADJUSTED PRICE = ACTUAL GCV / 6500 X FOB + FREIGHT',
             f45a='HS 2701.1900 BITUMINOUS', f32b='USD 800,000'),
     6500, 6200),
    ("Petcoke 7800/7500",
     make_lc(f47a='PRICE ADJUSTMENTS\n'
                  'GCV (ARB) : 7800 KCAL/KG BELOW 7500 KCAL/KG\n'
                  'SULPHUR (ARB) : 5 PCT ABOVE 6 PCT\n'
                  'HARDGROVE INDEX 38-42 NO REJECTION\n'
                  'ADJUSTED CFR = (FOB * GCV / 7800) + Freight',
             f45a='PETCOKE HS 2704', f32b='USD 2,000,000'),
     7800, 7500),
    ("Lignite low-rank 4200/3900",
     make_lc(f47a='PRICE ADJUSTMENT FORMULA: ACTUAL GCV / 4200\n'
                  'GCV (ARB) : 4200 KCAL/KG BELOW 3900 KCAL/KG\n'
                  'TM (ARB) : 30 PCT ABOVE 35 PCT',
             f45a='LIGNITE COAL HS 2702', f32b='USD 450,000'),
     4200, 3900),
]
for name, lc, exp_gcv, exp_rej in C_CASES:
    info = _detect_coal_quality_terms(lc)
    assert_eq(f"C.{name}: detected", info and info.get('is_coal_lc'), True)
    if info:
        assert_eq(f"C.{name}: GCV spec={exp_gcv}",
                  info['gcv_spec_kcal'], float(exp_gcv))
        assert_eq(f"C.{name}: GCV reject_below={exp_rej}",
                  info['gcv_reject_below'], float(exp_rej))


# ────────────────────────────────────────────────────────────────────
# D. Discrepancy-whitelist LCs — various phrasings
# ────────────────────────────────────────────────────────────────────
print("\n--- D. Discrepancy-whitelist (P198ff) — varied phrasing ---")
D_CASES = [
    ("standard 7-cat list (Pakistani)",
     make_lc(f47a='14) ALL DISCREPANCIES ARE ACCEPTABLE EXCEPT FOR '
                  'DESCRIPTION OF GOODS, QUANTITY, QUALITY, LATEST DATE OF '
                  'SHIPMENT, PORT OF LOADING AND PORT OF DISCHARGE AND ORIGIN OF GOODS.'),
     {'goods_description', 'quantity', 'quality', 'shipment_date',
      'port_of_loading', 'port_of_discharge', 'origin'}),
    ("compact 3-cat list",
     make_lc(f47a='ALL DISCREPANCIES ACCEPTABLE EXCEPT FOR QUANTITY, QUALITY AND ORIGIN.'),
     {'quantity', 'quality', 'origin'}),
    ("'WAIVED EXCEPT' phrasing",
     make_lc(f47a='Any discrepancy is waived except for the lc number.'),
     {'lc_number'}),
    ("'ACCEPTED EXCEPT' phrasing",
     make_lc(f47a='All discrepancies accepted except for amount and beneficiary.'),
     {'amount', 'beneficiary'}),
    ("multi-line clause",
     make_lc(f47a='17) ALL DISCREPANCIES ARE ACCEPTABLE EXCEPT FOR\n'
                  'GOODS DESCRIPTION,\n'
                  'QUANTITY,\n'
                  'PORT OF LOADING.'),
     {'goods_description', 'quantity', 'port_of_loading'}),
]
for name, lc, exp_cats in D_CASES:
    info = _detect_discrepancy_whitelist(lc)
    assert_eq(f"D.{name}: detected", info is not None, True)
    if info:
        for c in exp_cats:
            assert_eq(f"D.{name}: '{c}' in hard-fail set",
                      c in info['hard_fail_categories'], True)


# ────────────────────────────────────────────────────────────────────
# E. Late-shipment-with-penalty — various penalty structures
# ────────────────────────────────────────────────────────────────────
print("\n--- E. Late-shipment-with-penalty (P198fg) ---")
E_CASES = [
    ("USD per-day", 'USD', 500.0, True,
     make_lc(f47a='LATE SHIPMENT ALLOWED PROVIDED USD 500 PER DAY DEDUCTED.')),
    ("EUR per-day", 'EUR', 200.0, True,
     make_lc(f47a='Late shipment is acceptable with EUR 200 per day penalty.')),
    ("PKR flat", 'PKR', 50000.0, False,
     make_lc(f47a='Late shipment allowed provided PKR 50,000 deducted.')),
    ("USD flat with comma", 'USD', 5000.0, False,
     make_lc(f47a='Late shipment allowed with USD 5,000 penalty.')),
    ("delay form", 'USD', 250.0, True,
     make_lc(f47a='Delay in shipment is accepted with USD 250 per day penalty.')),
    ("subject to wording", 'USD', 1000.0, False,
     make_lc(f47a='LATE SHIPMENT IS ALLOWED SUBJECT TO USD 1000 PENALTY.')),
]
for name, exp_ccy, exp_amt, exp_per_day, lc in E_CASES:
    info = _detect_late_shipment_penalty(lc)
    assert_eq(f"E.{name}: detected", info is not None, True)
    if info:
        assert_eq(f"E.{name}: amount={exp_amt}", info['penalty_amount'], exp_amt)
        assert_eq(f"E.{name}: currency={exp_ccy}", info['penalty_currency'], exp_ccy)
        assert_eq(f"E.{name}: per_day={exp_per_day}", info['per_day'], exp_per_day)


# ────────────────────────────────────────────────────────────────────
# F. Required-surveyor LCs — all 9 known surveyors
# ────────────────────────────────────────────────────────────────────
print("\n--- F. Required-surveyor (P198fh) ---")
F_CASES = [
    ("SGS", 'sgs', make_lc(f46a='CERT BY SGS REQUIRED.')),
    ("Cotecna", 'cotecna', make_lc(f46a='COA BY COTECNA AT LOAD PORT.')),
    ("Intertek (ITS)", 'intertek', make_lc(f46a='CERT BY INTERTEK SERVICES.')),
    ("Bureau Veritas", 'bureau_veritas', make_lc(f46a='CERT BY BUREAU VERITAS.')),
    ("Alfred H Knight", 'alfred_knight', make_lc(f46a='WEIGHT CERT BY ALFRED H KNIGHT.')),
    ("Saybolt", 'saybolt', make_lc(f46a='PETROLEUM CERT BY SAYBOLT.')),
    ("Inspectorate", 'inspectorate', make_lc(f46a='COA BY INSPECTORATE.')),
    ("Geo-Chem", 'geo_chem', make_lc(f46a='CERT BY GEO-CHEM REQUIRED.')),
    ("Control Union", 'control_union', make_lc(f46a='CERT BY CONTROL UNION.')),
    ("multiple OR'd", 'sgs',
     make_lc(f46a='CERT BY SGS, COTECNA OR INTERTEK ACCEPTABLE.')),
]
for name, exp_sv, lc in F_CASES:
    info = _detect_required_surveyors(lc)
    assert_eq(f"F.{name}: detected", info is not None, True)
    if info:
        assert_eq(f"F.{name}: '{exp_sv}' in required",
                  exp_sv in info['required_surveyors'], True)


# ────────────────────────────────────────────────────────────────────
# G. Banner stacking — multiple detectors fire on same LC
# ────────────────────────────────────────────────────────────────────
print("\n--- G. Banner stacking (multiple detectors) ---")
combo1 = make_lc(
    f46a='1) BL\n2) CERT OF ANALYSIS BY SGS\n3) CERT OF WEIGHT BY ALFRED H KNIGHT',
    f47a='14) ALL DISCREPANCIES ACCEPTABLE EXCEPT FOR QUANTITY AND ORIGIN.\n'
         '15) LATE SHIPMENT ALLOWED PROVIDED USD 100 PER DAY DEDUCTED.\n'
         '16) CHARTER PARTY BL ACCEPTABLE.\n'
         '17) PRICE ADJUSTMENTS CLAUSE\n'
         'GCV (ARB) : 5800 KCAL/KG BELOW 5650 KCAL/KG\n'
         'TM (ARB) : 11 PCT ABOVE 13 PCT\n'
         'ADJUSTED CFR = FOB X GCV / 5800 + FREIGHT',
    f45a='BITUMINOUS COAL', f32b='USD 526,250')
ctx = _build_f47a_context(combo1)
assert_eq("G.combo: COAL banner", 'COAL-LC QUALITY' in ctx, True)
assert_eq("G.combo: WHITELIST banner", 'DISCREPANCY WHITELIST' in ctx, True)
assert_eq("G.combo: LATE-PENALTY banner", 'LATE-SHIPMENT-WITH-PENALTY' in ctx, True)
assert_eq("G.combo: SURVEYOR banner", 'REQUIRED INDEPENDENT SURVEYOR' in ctx, True)
assert_eq("G.combo: original F47A clause 16 (CHARTER PARTY) preserved",
          'CHARTER PARTY' in ctx.upper(), True)

# Coal + advance-payment combo (P198et + P198fb both fire)
combo2 = make_lc(
    f46a='A) 80 PERCENT ADVANCE PAYMENT\nB) REMAINING 20 PERCENT',
    f47a='17) PRICE ADJUSTMENTS\n'
         'GCV (ARB) : 6000 KCAL/KG BELOW 5700 KCAL/KG\n'
         'ADJUSTED CFR = FOB X GCV / 6000 + FREIGHT',
    f45a='COAL', f32b='USD 750,000')
ctx2 = _build_f47a_context(combo2)
assert_eq("G.combo2: ADVANCE-PAYMENT banner", 'ADVANCE-PAYMENT' in ctx2, True)
assert_eq("G.combo2: COAL banner", 'COAL-LC QUALITY' in ctx2, True)


# ────────────────────────────────────────────────────────────────────
# H. NEGATIVE / look-alike cases
# ────────────────────────────────────────────────────────────────────
print("\n--- H. Negative / look-alike cases ---")
H_CASES = [
    # Look like coal but aren't
    ("food calorific info", make_lc(f47a='CALORIFIC INFORMATION ON LABEL', f45a='COCOA')),
    ("LNG carrier (GAS not COAL)", make_lc(f47a='GAS CARRIAGE VESSEL CERTIFIED', f45a='LNG')),
    # Look like advance-payment but aren't
    ("100 PCT L/C VALUE only", make_lc(f47a='INVOICE FOR 100 PERCENT L/C VALUE')),
    ("'100 PERCENT' covering schedule (no advance)",
     make_lc(f47a='COVERING SCHEDULE MUST SHOW 100 PERCENT L/C VALUE')),
    # Look like discrepancy-whitelist but aren't
    ("USD 116 discrepancy charge",
     make_lc(f47a='USD 116 DISCREPANCY CHARGES WILL BE DEDUCTED.')),
    ("third-party EXCEPT",
     make_lc(f47a='THIRD PARTY DOCUMENTS ACCEPTABLE EXCEPT INVOICE AND DRAFT.')),
    # Look like late-pen but aren't
    ("late shipment forbidden",
     make_lc(f47a='LATE SHIPMENT NOT ACCEPTABLE.')),
    # SGS substring trap
    ("WGS84 substring",
     make_lc(f46a='WGS84 CO-ORDINATES REQUIRED.')),
]
for name, lc in H_CASES:
    assert_eq(f"H.{name}: coal NOT detected",
              _detect_coal_quality_terms(lc) is None, True)
    assert_eq(f"H.{name}: advance NOT detected",
              _detect_advance_payment_terms(lc) is None, True)
    assert_eq(f"H.{name}: whitelist NOT detected",
              _detect_discrepancy_whitelist(lc) is None, True)
    assert_eq(f"H.{name}: late-pen NOT detected",
              _detect_late_shipment_penalty(lc) is None, True)


# ────────────────────────────────────────────────────────────────────
# I. Real-data sweep — every job in results/
# ────────────────────────────────────────────────────────────────────
print("\n--- I. Real-data sweep ---")
RESULTS = 'd:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final/results'
total = adv = coal = wl = pen = sv = errs = 0
if os.path.isdir(RESULTS):
    for jid in sorted(os.listdir(RESULTS)):
        p = os.path.join(RESULTS, jid, 'step06', 'step06_result.json')
        if not os.path.isfile(p):
            continue
        try:
            with open(p, 'r', encoding='utf-8') as f:
                s6 = json.load(f)
        except Exception:
            continue
        total += 1
        try:
            if _detect_advance_payment_terms(s6):    adv  += 1
            if _detect_coal_quality_terms(s6):       coal += 1
            if _detect_discrepancy_whitelist(s6):    wl   += 1
            if _detect_late_shipment_penalty(s6):    pen  += 1
            if _detect_required_surveyors(s6):       sv   += 1
        except Exception:
            errs += 1
print(f"  {total} jobs scanned | adv={adv} | coal={coal} | wl={wl} | pen={pen} | sv={sv} | errs={errs}")
assert_eq("I.sweep: 0 detector exceptions", errs, 0)
assert_eq("I.sweep: at least 1 advance-payment LC found", adv >= 1, True)
assert_eq("I.sweep: at least 1 coal LC found", coal >= 1, True)
assert_eq("I.sweep: at least 1 whitelist LC found", wl >= 1, True)


passed = sum(results)
total = len(results)
print(f"\n{passed}/{total} cases passed")
if passed != total:
    sys.exit(1)
print("OVERALL: OK")
sys.exit(0)
