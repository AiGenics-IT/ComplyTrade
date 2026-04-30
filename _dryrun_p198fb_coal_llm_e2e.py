"""
P198fb end-to-end LLM dry-run for coal LCs.

Hits the LIVE Qwen Text LLM with the verification CORE prompt for
coal-quality scenarios — verifies the LLM correctly applies the
COAL-LC QUALITY SPECIFICATIONS banner that P198fb injects into
f47a_context.

Anchor: real coal LC f3ef028e — F47A clause 17 has:
  GCV (ARB) spec 5,800 kcal/kg, REJECT if < 5,650
  TM (ARB) spec 11%, REJECT > 13%
  Ash (ARB) spec 15%, REJECT > 17%
  Sulphur (ARB) spec 0.8%, REJECT > 1%
  Pricing formula: Adjusted CFR = (FOB x actualGCV / 5800) + Freight

Test scenarios:
  1. COA shows GCV 5,900 / TM 10% / Ash 14% / S 0.7% — all in spec
     → expect PASS (full price applies, all params within spec)
  2. COA shows GCV 5,700 (between reject 5650 and spec 5800)
     → expect PASS — price is adjusted down per formula, not rejected
  3. COA shows GCV 5,400 (BELOW reject 5,650)
     → expect FAIL — hard rejection threshold breached
  4. COA shows TM 14% (ABOVE reject 13%)
     → expect FAIL — hard rejection threshold breached
  5. COA shows Ash 18% (ABOVE reject 17%)
     → expect FAIL — hard rejection threshold breached
  6. Commercial Invoice priced at USD 517,180 (= 5000 x 103.43 from
     prorated formula when GCV=5700) — should pass under split-pricing
     even though it's BELOW LC face value 526,250
     → expect PASS — adjusted-price recognition

Each scenario hits the LIVE LLM with the same banner the production
verification would produce.
"""
import sys, os, json
sys.path.insert(0, 'd:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final')

from steps.step14_verification import (
    _build_f47a_context,
    _call_vlm,
)

results = []
def case(name, expected, vlm_result):
    got = (vlm_result or {}).get('compliance', '').upper()
    ok = (got == expected)
    print(f"[{'OK' if ok else 'FAIL'}] {name}: got={got}  expected={expected}")
    if not ok:
        print(f"      result: {(vlm_result or {}).get('result','')[:240]}")
    results.append(ok)

JOB = 'd:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final/results/f3ef028e-b879-40d2-9351-39a2aff90175/step06/step06_result.json'
with open(JOB, 'r', encoding='utf-8') as f:
    real_s6 = json.load(f)
f47a_ctx = _build_f47a_context(real_s6)
final_lc_fields = real_s6.get('consolidated_fields', {})

assert 'COAL-LC QUALITY SPECIFICATIONS' in f47a_ctx, "P198fb banner not present"
print("--- f47a_context contains COAL-LC QUALITY SPECIFICATIONS banner ✓ ---\n")

def coa_text(gcv, tm, ash, sulphur, vm=37, im=4, hgi=45):
    return (
        "CERTIFICATE OF SAMPLING AND ANALYSIS\n"
        "ISSUED BY SGS INDONESIA LTD\n"
        "AT LOADING PORT: SAMARINDA, INDONESIA\n"
        "Cargo: Bituminous Steam Coal in Bulk\n"
        "B/L No: BLINDO/2026/001\n"
        "Vessel: MV ABC\n"
        "RESULTS (As Received Basis unless noted):\n"
        f"Gross Calorific Value (ARB): {gcv} kcal/kg\n"
        f"Total Moisture (ARB): {tm}%\n"
        f"Inherent Moisture (ADB): {im}%\n"
        f"Ash (ARB): {ash}%\n"
        f"Total Sulphur (ARB): {sulphur}%\n"
        f"Volatile Matter (ARB): {vm}%\n"
        f"Hardgrove Index: {hgi}\n"
        f"Size: 0-50mm\n"
    )


# ── Scenario 1: all-in-spec COA ─────────────────────────────────────
print("Scenario 1: COA all-in-spec (GCV 5900 / TM 10 / Ash 14 / S 0.7)")
r1 = _call_vlm(
    row_id="coal-1",
    condition_text=("The Certificate of Sampling and Analysis from the load port "
                    "must show coal quality within the LC specifications."),
    clause_ref="46A-4",
    lc_field_value="GCV 5,800 / TM 11% / Ash 15% / S 0.8%",
    f47a_context=f47a_ctx,
    document_type="Certificate of Sampling and Analysis",
    document_text=coa_text(gcv=5900, tm=10, ash=14, sulphur=0.7),
    visual_metadata="(none)",
    final_lc_fields=final_lc_fields,
)
case("Scenario 1: all in spec → PASS", "PASS", r1)
print(f"      reasoning: {r1.get('reasoning','')[:200]}\n")

# ── Scenario 2: GCV 5,700 — adjusted-price territory ────────────────
# 5,700 is BETWEEN reject 5,650 and spec 5,800 — should PASS with the
# pricing formula applied. (5,500 would be below rejection → FAIL.)
print("Scenario 2: GCV 5,700 (between reject 5650 and spec 5800)")
r2 = _call_vlm(
    row_id="coal-2",
    condition_text=("The Certificate of Sampling and Analysis must show coal "
                    "quality within acceptable limits per F47A clause 17."),
    clause_ref="46A-4",
    lc_field_value="GCV 5,800 / TM 11% / Ash 15% / S 0.8%",
    f47a_context=f47a_ctx,
    document_type="Certificate of Sampling and Analysis",
    document_text=coa_text(gcv=5700, tm=10, ash=14, sulphur=0.7),
    visual_metadata="(none)",
    final_lc_fields=final_lc_fields,
)
# This one may legitimately be PASS (above rejection floor) or REVIEW
# (because price needs adjustment) — accept either as long as it's not FAIL.
got2 = (r2 or {}).get('compliance', '').upper()
ok2 = got2 in ('PASS', 'REVIEW')
print(f"[{'OK' if ok2 else 'FAIL'}] Scenario 2: above rejection floor → PASS/REVIEW: got={got2}")
if not ok2:
    print(f"      result: {r2.get('result','')[:240]}")
results.append(ok2)
print(f"      reasoning: {r2.get('reasoning','')[:200]}\n")

# ── Scenario 3: GCV 5,400 — below rejection floor ───────────────────
print("Scenario 3: GCV 5,400 (BELOW reject 5,650)")
r3 = _call_vlm(
    row_id="coal-3",
    condition_text=("The Certificate of Sampling and Analysis must show coal "
                    "quality within acceptable limits per F47A clause 17."),
    clause_ref="46A-4",
    lc_field_value="GCV >= 5,650 kcal/kg required",
    f47a_context=f47a_ctx,
    document_type="Certificate of Sampling and Analysis",
    document_text=coa_text(gcv=5400, tm=10, ash=14, sulphur=0.7),
    visual_metadata="(none)",
    final_lc_fields=final_lc_fields,
)
case("Scenario 3: GCV below rejection → FAIL", "FAIL", r3)
print(f"      reasoning: {r3.get('reasoning','')[:200]}\n")

# ── Scenario 4: TM 14% — above rejection threshold ─────────────────
print("Scenario 4: TM 14% (ABOVE reject 13%)")
r4 = _call_vlm(
    row_id="coal-4",
    condition_text=("The Certificate of Sampling and Analysis must show Total "
                    "Moisture within acceptable limits per F47A clause 17."),
    clause_ref="46A-4",
    lc_field_value="TM (ARB) max 13%",
    f47a_context=f47a_ctx,
    document_type="Certificate of Sampling and Analysis",
    document_text=coa_text(gcv=5800, tm=14, ash=14, sulphur=0.7),
    visual_metadata="(none)",
    final_lc_fields=final_lc_fields,
)
case("Scenario 4: TM above rejection → FAIL", "FAIL", r4)
print(f"      reasoning: {r4.get('reasoning','')[:200]}\n")

# ── Scenario 5: Ash 18% — above rejection threshold ────────────────
print("Scenario 5: Ash 18% (ABOVE reject 17%)")
r5 = _call_vlm(
    row_id="coal-5",
    condition_text=("The Certificate of Sampling and Analysis must show Ash "
                    "within acceptable limits per F47A clause 17."),
    clause_ref="46A-4",
    lc_field_value="Ash (ARB) max 17%",
    f47a_context=f47a_ctx,
    document_type="Certificate of Sampling and Analysis",
    document_text=coa_text(gcv=5800, tm=10, ash=18, sulphur=0.7),
    visual_metadata="(none)",
    final_lc_fields=final_lc_fields,
)
case("Scenario 5: Ash above rejection → FAIL", "FAIL", r5)
print(f"      reasoning: {r5.get('reasoning','')[:200]}\n")

# ── Scenario 6: Commercial Invoice with adjusted price (split-pricing) ─
print("Scenario 6: Invoice priced down via GCV-prorated formula")
r6 = _call_vlm(
    row_id="coal-6",
    condition_text=("The Commercial Invoice amount must not exceed the LC amount."),
    clause_ref="32B",
    lc_field_value="USD 526,250.00",
    f47a_context=f47a_ctx,
    document_type="Commercial Invoice",
    document_text=(
        "COMMERCIAL INVOICE\n"
        "DESCRIPTION: BITUMINOUS STEAM COAL IN BULK\n"
        "QUANTITY: 5,000.00 MT\n"
        "UNIT PRICE: USD 103.43/MT (adjusted per F47A-17 GCV proration: "
        "FOB 105.25 x actual GCV 5,700 / spec GCV 5,800)\n"
        "TOTAL: USD 517,180.00\n"
        "INDONESIA ORIGIN, HS 2701.1200\n"
    ),
    visual_metadata="(none)",
    final_lc_fields=final_lc_fields,
)
case("Scenario 6: invoice priced down via formula → PASS", "PASS", r6)
print(f"      reasoning: {r6.get('reasoning','')[:200]}\n")


passed = sum(results)
total = len(results)
print(f"\n{passed}/{total} live-LLM coal scenarios behaved as expected")
if passed != total:
    sys.exit(1)
print("OVERALL: OK — coal-LC banner correctly steers LLM verdict")
sys.exit(0)
