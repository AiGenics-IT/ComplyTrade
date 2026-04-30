"""
P198fb dry-run — Coal-LC quality detection + price-adjustment context.

Validates:
  1. _detect_coal_quality_terms() correctly extracts each parameter
     (TM / Ash / Sulphur / VM / GCV / IM / Size / HGI) from real F47A
     text including spec values, rejection thresholds, and basis
     (ARB / ADB).
  2. The COAL-LC QUALITY banner is prepended to f47a_context so the
     verification LLM sees the structured table + pricing formula
     before any per-field rule (CORE rule 7b reads from this banner).
  3. Detector returns None for STANDARD LCs (no false positives on
     non-coal jobs — previous P198eo/ep/eq/er/es/et/eu/ev/ew/ex/ey/ez/fa
     scenarios stay green).
  4. Real-data sweep across results/* — confirm the detector matches
     ONLY genuine coal LCs.

Real-data anchor:
  Job f3ef028e-b879-40d2-9351-39a2aff90175 (LC 0002LC60016/2026)
    Bituminous Steam Coal, 5,000 MT, USD 526,250.00
    F47A clause 17 — full spec table:
      Total Moisture (ARB) 11%, REJECT > 13%
      Inherent Moisture (ADB) 3-5%, no rejection
      Ash (ARB) 15%, REJECT > 17%
      Sulphur (ARB) 0.8%, REJECT > 1%
      Volatile Matter (ARB) 36-40%, no rejection
      GCV (ARB) 5,800 kcal/kg, REJECT < 5,650 kcal/kg
      Size 0-50mm, no rejection
      HGI 40-50, no rejection
    Pricing formula: (FOB x actualGCV / 5800) + Freight
"""
import sys, os, json
sys.path.insert(0, 'd:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final')

from steps.step14_verification import (
    _detect_coal_quality_terms,
    _format_coal_quality_block,
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

# ── Test 1: Real coal job f3ef028e ──────────────────────────────────────
JOB = 'd:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final/results/f3ef028e-b879-40d2-9351-39a2aff90175'
with open(os.path.join(JOB, 'step06', 'step06_result.json'), 'r',
          encoding='utf-8') as f:
    real_s6 = json.load(f)

info = _detect_coal_quality_terms(real_s6)
print("\n--- Test 1: real coal job f3ef028e ---")
print(f"  detector: is_coal_lc={info and info.get('is_coal_lc')}")
print(f"  GCV spec: {info and info.get('gcv_spec_kcal')}")
print(f"  GCV reject: {info and info.get('gcv_reject_below')}")
print(f"  formula: {info and info.get('formula_type')}")

assert_eq("real coal job: is_coal_lc detected",
          info is not None and info.get('is_coal_lc'), True)
assert_close("real coal job: GCV spec ~5800",
             info.get('gcv_spec_kcal'), 5800.0)
assert_close("real coal job: GCV reject_below ~5650",
             info.get('gcv_reject_below'), 5650.0)
assert_eq("real coal job: has GCV-prorated formula",
          info.get('has_price_adjustment') and info.get('formula_type') == 'gcv_prorated', True)
params = info.get('parameters') or {}
assert_eq("real coal job: TM parsed", 'total_moisture' in params, True)
assert_eq("real coal job: TM reject_above ~13",
          params['total_moisture'].get('reject_above'), 13.0)
assert_eq("real coal job: Ash reject_above ~17",
          params['ash'].get('reject_above'), 17.0)
assert_eq("real coal job: Sulphur reject_above ~1",
          params['sulphur'].get('reject_above'), 1.0)
assert_eq("real coal job: Inherent Moisture has 'no rejection' marker",
          'reject' in params.get('inherent_moisture', {}) and
          params['inherent_moisture'].get('reject') is None, True)
assert_eq("real coal job: VM has 'no rejection' marker",
          'reject' in params.get('volatile_matter', {}) and
          params['volatile_matter'].get('reject') is None, True)

# ── Test 2: f47a_context contains the coal banner ───────────────────────
ctx = _build_f47a_context(real_s6)
print(f"\n--- Test 2: f47a_context coal banner ---")
present = 'COAL-LC QUALITY SPECIFICATIONS' in ctx
print(f"  banner present: {present}")
assert_eq("real coal job: f47a_context contains coal banner", present, True)
assert_eq("real coal job: banner shows GCV 5800", '5800' in ctx, True)
assert_eq("real coal job: banner shows REJECT < 5650", '5650' in ctx, True)
assert_eq("real coal job: banner shows TM REJECT > 13", '> 13' in ctx, True)
assert_eq("real coal job: banner shows Ash REJECT > 17", '> 17' in ctx, True)
assert_eq("real coal job: banner shows Sulphur REJECT > 1",
          '> 1' in ctx and 'Sulphur' in ctx, True)
assert_eq("real coal job: banner mentions pricing formula",
          'Adjusted CFR' in ctx and 'FOB' in ctx, True)
# Original F47A content still present
assert_eq("real coal job: original F47A clause 1 preserved",
          'DOCUMENTS DATED PRIOR' in ctx.upper(), True)
assert_eq("real coal job: original F47A clause 16 (CHARTER PARTY) preserved",
          'CHARTER PARTY' in ctx.upper(), True)

# ── Test 3: NO false positives on standard / advance-payment LCs ────────
def make_lc(f46a='', f47a='', f45a='', f32b='USD 100,000.00'):
    return {'consolidated_fields': {
        '32B': f32b, '46A': f46a, '47A': f47a, '45A': f45a,
    }}

print(f"\n--- Test 3: NO false positives ---")
none_cases = [
    ("standard sight LC, no coal mention",
     make_lc('1. INVOICE 2. BL 3. PACKING LIST',
             'TPND 5/5 LATE SHIPMENT NOT ACCEPTABLE')),
    ("advance-payment LC (P198et) — must not collide",
     make_lc('A) 80 PERCENT ADVANCE PAYMENT WILL BE MADE\n'
             'B) REMAINING 20 PERCENT PAYABLE',
             '6) NEGOTIATING BANK COVERING SCHEDULE MUST SHOW 100 PERCENT')),
    ("F47A mentions 'value' but not GCV/coal",
     make_lc('', 'INVOICE FOR 100 PERCENT L/C VALUE')),
    ("standard with HS code 8415 (machinery, not coal)",
     make_lc('CERTIFYING H.S.CODE NO. 8415.1029', '')),
    ("LC mentions 'gas' but no coal markers",
     make_lc('NATURAL GAS', '5/5 TOLERANCE')),
    ("empty fields", make_lc('', '')),
    ("None step06", None),
    ("non-dict step06", "garbage"),
]
for name, lc in none_cases:
    got = _detect_coal_quality_terms(lc)
    ok = (got is None)
    print(f"[{'OK' if ok else 'FAIL'}] {name}: got={got and got.get('is_coal_lc')}")
    results.append(ok)

# ── Test 4: synthetic coal LCs (different specs) ────────────────────────
print(f"\n--- Test 4: synthetic coal LCs ---")
# Variant A: NAR-based (Indian / Bangladesh pattern)
sc_a = make_lc(
    '', '17) PRICE ADJUSTMENTS CLAUSE\n'
    'GROSS CALORIFIC VALUE (ARB) : 6000 KCAL/KG BELOW 5800 KCAL/KG\n'
    'TOTAL MOISTURE (ARB) : 12 PCT ABOVE 14 PCT\n'
    'ASH (ARB) : 10 PCT ABOVE 12 PCT\n'
    'SULPHUR (ARB) : 0.5 PCT ABOVE 0.7 PCT\n'
    'IF GCV ARB IS BELOW 6000, ADJUSTED CFR PRICE = (FOB MULTIPLY BY ACTUAL GCV ARB DIVIDED BY 6000) PLUS FREIGHT',
    'BITUMINOUS COAL HS 2701.1900', 'USD 1,200,000')
ia = _detect_coal_quality_terms(sc_a)
assert_eq("variant A: detected", ia and ia.get('is_coal_lc'), True)
assert_close("variant A: GCV spec=6000", ia['gcv_spec_kcal'], 6000.0)
assert_close("variant A: GCV reject_below=5800", ia['gcv_reject_below'], 5800.0)
assert_eq("variant A: TM reject_above=14",
          ia['parameters']['total_moisture']['reject_above'], 14.0)
assert_eq("variant A: Ash reject_above=12",
          ia['parameters']['ash']['reject_above'], 12.0)
assert_eq("variant A: Sulphur reject_above=0.7",
          ia['parameters']['sulphur']['reject_above'], 0.7)
assert_eq("variant A: formula detected",
          ia['has_price_adjustment'] and ia['formula_type'] == 'gcv_prorated', True)

# Variant B: minimal coal LC (just GCV row, no rejection)
sc_b = make_lc('', 'GROSS CALORIFIC VALUE (ARB): 5500 KCAL/KG\n'
                   'PRICE ADJUSTMENTS CLAUSE\n'
                   'Adjusted price proportional to GCV',
               'COAL HS 2701', 'USD 50,000')
ib = _detect_coal_quality_terms(sc_b)
assert_eq("variant B: detected on GCV-only", ib and ib.get('is_coal_lc'), True)
assert_close("variant B: GCV spec=5500", ib['gcv_spec_kcal'], 5500.0)

# Variant C: no GCV → not detected (degenerate case — coal-related text but no spec)
sc_c = make_lc('', 'COAL OF INDONESIAN ORIGIN — Hardgrove Index 50',
               'BITUMINOUS COAL', 'USD 100,000')
ic = _detect_coal_quality_terms(sc_c)
assert_eq("variant C (no GCV spec): returns None", ic, None)

# ── Test 5: real-data sweep — count coal LCs found across saved jobs ───
print(f"\n--- Test 5: real-data sweep across local results/ ---")
RESULTS_DIR = 'd:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final/results'
total = 0; coal_count = 0; std_count = 0; errors = 0
coal_jobs = []
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
        det = _detect_coal_quality_terms(s6)
    except Exception:
        errors += 1
        continue
    if det:
        coal_count += 1
        coal_jobs.append((jid, det['gcv_spec_kcal'], det.get('gcv_reject_below')))
    else:
        std_count += 1
print(f"  totals: {total} jobs scanned | "
      f"{coal_count} coal LC | {std_count} non-coal | {errors} errors")
for j, gs, gr in coal_jobs[:10]:
    print(f"  [COAL] {j}: GCV spec={gs}, reject<{gr}")
assert_eq("real-data sweep: no detector exceptions", errors, 0)
assert_eq("real-data sweep: at least 1 coal LC found (the f3ef028e job)",
          coal_count >= 1, True)

# ── Test 6: check that advance-payment LCs (P198et) STILL work ──────────
# Importantly, an LC that triggers BOTH advance-payment and coal-quality
# should produce BOTH banners. Verify the existing P198et detector
# wasn't disturbed.
print(f"\n--- Test 6: P198et regression (advance-payment unchanged) ---")
from steps.step14_verification import _detect_advance_payment_terms
real_2d98_p = 'd:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final/results/2d98b74c-457f-4456-8a85-68841190e4d5/step06/step06_result.json'
if os.path.isfile(real_2d98_p):
    with open(real_2d98_p, 'r', encoding='utf-8') as f:
        s6_2d98 = json.load(f)
    apinfo = _detect_advance_payment_terms(s6_2d98)
    assert_eq("P198et regression: real 2d98b74c still detects advance-pay split",
              apinfo and apinfo.get('is_advance_split'), True)
    assert_eq("P198et regression: real 2d98b74c is NOT a coal LC",
              _detect_coal_quality_terms(s6_2d98), None)

# ── Test 7: Coal LC EDGE CASES ─────────────────────────────────────────
# Below are the coal-specific edge cases the system must handle without
# false-firing or missing genuine coal LCs. Each maps to a real-world
# variant we've seen in trade-finance LCs across Pakistan / Bangladesh
# / India coal imports.
print(f"\n--- Test 7: Coal-LC edge cases ---")

EDGE_CASES = [
    # (name, lc_dict, expected_detected, optional checks)
    # 1. Different basis - NAR
    ("NAR-based pricing (Bangladesh / India pattern)",
     make_lc('', '17) PRICE ADJUSTMENT CLAUSE\n'
             'NAR (NET AS RECEIVED) : 5500 KCAL/KG BELOW 5400 KCAL/KG\n'
             'TOTAL MOISTURE (ARB) : 14 PCT ABOVE 16 PCT\n'
             'IF NAR IS BELOW 5500, ADJUSTED CFR = (FOB MULTIPLY ACTUAL NAR DIVIDED BY 5500) PLUS FREIGHT',
             'BITUMINOUS COAL', 'USD 600,000'), True, {}),
    # 2. Different basis - ADB / GAR / DB combo
    ("GAR + ADB mixed basis",
     make_lc('', 'GROSS CALORIFIC VALUE (GAR) : 6300 KCAL/KG BELOW 6000\n'
             'INHERENT MOISTURE (ADB) : 10-12 PCT NO REJECTION\n'
             'TOTAL MOISTURE (GAR) : 18 PCT ABOVE 20 PCT\n'
             'ASH (ADB) : 4 PCT ABOVE 6 PCT\n'
             'PRICE ADJUSTMENTS CLAUSE\n'
             'ADJUSTED PRICE = FOB X ACTUAL GAR / 6300 + FREIGHT',
             'COAL', 'USD 800,000'), True, {'gcv_spec': 6300}),
    # 3. Range-based GCV spec ("5700-5900 KCAL/KG")
    ("Range-based GCV spec (no rejection floor)",
     make_lc('', 'GROSS CALORIFIC VALUE (ARB) : 5700-5900 KCAL/KG NO REJECTION\n'
             'TOTAL MOISTURE (ARB) : 12 PCT ABOVE 14 PCT\n'
             'PRICE ADJUSTMENTS CLAUSE\n'
             'IF GCV ARB BELOW 5700 — REJECTION',
             'COAL', 'USD 200,000'), True, {}),
    # 4. Stepped/penalty pricing (instead of pure proportional)
    ("Stepped pricing — penalty per 100 kcal/kg below spec",
     make_lc('', 'GROSS CALORIFIC VALUE (ARB) : 6500 KCAL/KG BELOW 6200 KCAL/KG\n'
             'TOTAL MOISTURE (ARB) : 8 PCT ABOVE 10 PCT\n'
             'ASH (ARB) : 9 PCT ABOVE 11 PCT\n'
             'PRICE ADJUSTMENT: USD 1.50/MT PENALTY PER 100 KCAL/KG BELOW SPEC',
             'BITUMINOUS COAL', 'USD 1,500,000'), True, {}),
    # 5. Petcoke (HS 2704) with different parameter set
    ("Petcoke pricing — Volatile Combustible Matter dominant",
     make_lc('', 'HARDGROVE INDEX 38-42 NO REJECTION\n'
             'GROSS CALORIFIC VALUE (ARB) : 7800 KCAL/KG BELOW 7500 KCAL/KG\n'
             'SULPHUR (ARB) : 4.5 PCT ABOVE 5 PCT\n'
             'PRICE ADJUSTMENTS CLAUSE\n'
             'ADJUSTED CFR = (FOB * ACTUAL GCV / 7800) + FREIGHT',
             'PETCOKE HS 2704', 'USD 2,000,000'), True, {'gcv_spec': 7800}),
    # 6. Lignite (low-rank, lower spec)
    ("Lignite — lower GCV bands",
     make_lc('', 'GROSS CALORIFIC VALUE (ARB) : 4200 KCAL/KG BELOW 3900 KCAL/KG\n'
             'TOTAL MOISTURE (ARB) : 30 PCT ABOVE 35 PCT\n'
             'PRICE ADJUSTMENT FORMULA: ACTUAL GCV / 4200',
             'LIGNITE COAL', 'USD 450,000'), True, {'gcv_spec': 4200}),
    # 7. Multi-line GCV with explicit "REJECTION" wording
    ("Multi-line GCV with REJECTION word",
     make_lc('', 'PRICE ADJUSTMENTS CLAUSE\n'
             'GROSS CALORIFIC VALUE (ARB) : 5800 KCAL/KG\n'
             'IF BELOW 5650 KCAL/KG REJECTION ACCEPTED\n'
             'TOTAL MOISTURE (ARB) : 11 PCT ABOVE 13 PCT\n'
             'IF GCV ARB IS BELOW 5800, ADJUSTED PRICE = FOB X ACTUAL GCV / 5800 + FREIGHT',
             'COAL', 'USD 500,000'), True, {}),
    # 8. Sentence-form (no table, but valid formula)
    ("Sentence-form coal LC (no table, but formula present)",
     make_lc('', '17) THE COAL SHALL BE BITUMINOUS WITH GROSS CALORIFIC '
             'VALUE OF 5800 KCAL/KG (ARB). IF GCV BELOW 5800, PRICE WILL '
             'BE ADJUSTED PROPORTIONALLY. REJECTION IF GCV BELOW 5500 KCAL/KG. '
             'TOTAL MOISTURE NOT MORE THAN 12 PCT.',
             'COAL', 'USD 300,000'), True, {}),
    # 9. Standard non-coal LC mentioning "calorific" in unrelated context (e.g. food)
    ("Non-coal LC mentioning 'calorific' (food / chocolate) — must NOT match",
     make_lc('', 'CALORIFIC INFORMATION ON LABEL REQUIRED PER FDA STANDARDS',
             'COCOA POWDER HS 1805', 'USD 100,000'), False, {}),
    # 10. LC with 'GCV' acronym in unrelated context — STILL TRIGGERS but no spec → returns None
    ("'GCV' as part of an acronym (Gas Carriage Vessel) — no spec → None",
     make_lc('', 'CARRIER MUST BE GCV-COMPLIANT (GAS CARRIAGE VESSEL)',
             'LNG TRANSPORT', 'USD 500,000'), False, {}),
    # 11. Hardgrove Index alone (not a coal LC by itself, but trigger fires)
    ("Hardgrove mentioned but no GCV spec — None",
     make_lc('', 'HARDGROVE INDEX REQUIRED ON CERTIFICATE',
             'COAL', 'USD 100,000'), False, {}),
    # 12. Coal with stepped rejection (multiple thresholds)
    ("Tiered rejection — different thresholds per parameter",
     make_lc('', 'GROSS CALORIFIC VALUE (ARB) : 6200 KCAL/KG BELOW 5900 KCAL/KG\n'
             'TOTAL MOISTURE (ARB) : 9 PCT ABOVE 11 PCT\n'
             'INHERENT MOISTURE (ADB) : 4-6 PCT NO REJECTION\n'
             'ASH (ARB) : 8 PCT ABOVE 10 PCT\n'
             'SULPHUR (ARB) : 0.6 PCT ABOVE 0.8 PCT\n'
             'VOLATILE MATTER (ARB) : 32-36 PCT NO REJECTION\n'
             'PRICE ADJUSTMENT: ADJUSTED PRICE = (FOB X ACTUAL GCV / 6200) + FREIGHT',
             'BITUMINOUS COAL', 'USD 1,000,000'),
     True, {'gcv_spec': 6200, 'tm_reject': 11, 'ash_reject': 10, 'sulphur_reject': 0.8}),
    # 13. Spec value with decimal point ("5,800.5 KCAL/KG")
    ("Decimal in GCV spec",
     make_lc('', 'GROSS CALORIFIC VALUE (ARB) : 5,800.5 KCAL/KG BELOW 5650 KCAL/KG\n'
             'PRICE ADJUSTMENT: FORMULA AS PER GCV',
             'COAL', 'USD 400,000'), True, {}),
    # 14. Coal + advance-payment combination — BOTH banners should fire
    ("Coal LC WITH advance-payment 70/30 split (both banners fire)",
     make_lc('A) 70 PERCENT ADVANCE PAYMENT VIA SWIFT\n'
             'B) REMAINING 30 PERCENT AGAINST DOCS',
             'GROSS CALORIFIC VALUE (ARB) : 6000 KCAL/KG BELOW 5700 KCAL/KG\n'
             'TOTAL MOISTURE (ARB) : 10 PCT ABOVE 12 PCT\n'
             'ASH (ARB) : 12 PCT ABOVE 14 PCT\n'
             'PRICE ADJUSTMENTS CLAUSE: ADJUSTED CFR = (FOB X ACTUAL GCV / 6000) + FREIGHT',
             'BITUMINOUS COAL', 'USD 750,000'),
     True, {'expect_advance': True}),
    # 15. Charter Party + coal (typical for bulk shipments)
    ("Charter Party BL + coal quality",
     make_lc('FULL SET OF MARINE BL FREIGHT PAYABLE AS PER CHARTER PARTY',
             '16) CHARTER PARTY BILL OF LADING ACCEPTABLE\n'
             '17) GROSS CALORIFIC VALUE (ARB) : 5800 KCAL/KG BELOW 5650 KCAL/KG\n'
             'PRICE ADJUSTMENT: ADJUSTED CFR = FOB * ACTUAL GCV / 5800 + FREIGHT',
             'COAL', 'USD 500,000'), True, {}),
    # 16. Multiple GCV mentions — pick the formal table row (this is the bug we just fixed)
    ("Multiple GCV mentions in one F47A — table row wins",
     make_lc('', 'PRICE ADJUSTMENTS CLAUSE\n'
             'GROSS CALORIFIC VALUE\n'
             'IF ACTUAL GCV IS BELOW 5800 KCAL/KG, ADJUSTMENT APPLIES\n'
             'ADJUSTED CFR = FOB * GCV / 5800 + FREIGHT\n'
             'GROSS CALORIFIC VALUE (ARB) : 5800 KCAL/KG BELOW 5650 KCAL/KG\n'
             'TOTAL MOISTURE (ARB) : 11 PCT ABOVE 13 PCT',
             'COAL', 'USD 500,000'),
     True, {'gcv_spec': 5800, 'gcv_reject_below': 5650}),
    # 17. NAR + GCV both in same LC (uncommon but seen)
    ("Both NAR and GCV mentioned",
     make_lc('', 'PRICE ADJUSTMENTS CLAUSE\n'
             'GROSS CALORIFIC VALUE (ARB) : 5800 KCAL/KG BELOW 5650 KCAL/KG\n'
             'NAR (REPORTED) : 5400 KCAL/KG NO REJECTION\n'
             'TOTAL MOISTURE (ARB) : 11 PCT ABOVE 13 PCT\n'
             'ASH (ARB) : 15 PCT ABOVE 17 PCT\n'
             'ADJUSTED CFR = FOB * GCV / 5800 + FREIGHT',
             'COAL', 'USD 500,000'),
     True, {'gcv_spec': 5800}),
    # 18. Truncated F47A (clause cut off mid-table) — be tolerant
    ("Truncated F47A (clause cut off mid-table)",
     make_lc('', '17) PRICE ADJUSTMENTS CLAUSE\n'
             'GROSS CALORIFIC VALUE (ARB) : 5800 KCAL/KG\n'
             'TOTAL MOISTURE (ARB) : 11 PCT ABO',
             'COAL', 'USD 500,000'), True, {'gcv_spec': 5800}),
    # 19. Sulphur with decimal threshold
    ("Sulphur with decimal threshold (e.g. 0.65)",
     make_lc('', 'PRICE ADJUSTMENTS CLAUSE\n'
             'GROSS CALORIFIC VALUE (ARB) : 5500 KCAL/KG BELOW 5300 KCAL/KG\n'
             'SULPHUR (ARB) : 0.45 PCT ABOVE 0.65 PCT\n'
             'TOTAL MOISTURE (ARB) : 13 PCT ABOVE 15 PCT\n'
             'ADJUSTED PRICE = FOB X GCV / 5500 + FREIGHT',
             'COAL', 'USD 300,000'),
     True, {'sulphur_reject': 0.65}),
    # 20. F47A in list-of-dicts form (some LC parsers wrap clauses this way)
    ("F47A as list-of-dicts (parser variation)",
     {'consolidated_fields': {
         '32B': 'USD 500,000',
         '47A': [
             {'text': '1) STANDARD CLAUSE'},
             {'text': '17) PRICE ADJUSTMENTS CLAUSE\nGROSS CALORIFIC VALUE (ARB) : 5800 KCAL/KG BELOW 5650 KCAL/KG\nADJUSTED CFR = FOB * GCV / 5800 + FREIGHT'},
         ],
     }}, True, {'gcv_spec': 5800}),
]

for case_name, lc, expected_detected, checks in EDGE_CASES:
    got = _detect_coal_quality_terms(lc)
    detected = bool(got and got.get('is_coal_lc'))
    ok = (detected == expected_detected)
    note = ''
    if ok and detected and checks:
        for k, v in checks.items():
            if k == 'gcv_spec':
                if abs((got.get('gcv_spec_kcal') or 0) - v) > 0.01:
                    ok = False; note = f' [gcv_spec={got.get("gcv_spec_kcal")} != {v}]'
            elif k == 'gcv_reject_below':
                if abs((got.get('gcv_reject_below') or 0) - v) > 0.01:
                    ok = False; note = f' [gcv_reject={got.get("gcv_reject_below")} != {v}]'
            elif k == 'tm_reject':
                p = got.get('parameters', {}).get('total_moisture', {})
                if abs((p.get('reject_above') or 0) - v) > 0.01:
                    ok = False; note = f' [tm_reject={p.get("reject_above")} != {v}]'
            elif k == 'ash_reject':
                p = got.get('parameters', {}).get('ash', {})
                if abs((p.get('reject_above') or 0) - v) > 0.01:
                    ok = False; note = f' [ash_reject={p.get("reject_above")} != {v}]'
            elif k == 'sulphur_reject':
                p = got.get('parameters', {}).get('sulphur', {})
                if abs((p.get('reject_above') or 0) - v) > 0.01:
                    ok = False; note = f' [s_reject={p.get("reject_above")} != {v}]'
            elif k == 'expect_advance':
                # Confirm advance-payment STILL detects on the same LC
                ai = _detect_advance_payment_terms(lc)
                if not (ai and ai.get('is_advance_split')):
                    ok = False; note = ' [advance-payment NOT detected on combo LC]'
    print(f"[{'OK' if ok else 'FAIL'}] {case_name}{note}")
    results.append(ok)


passed = sum(results)
total_t = len(results)
print(f"\n{passed}/{total_t} cases passed")
if passed != total_t:
    sys.exit(1)
print("OVERALL: OK")
sys.exit(0)
