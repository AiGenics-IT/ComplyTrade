"""
P198ff end-to-end LLM dry-run.

Hits the LIVE Qwen Text LLM with the verification CORE prompt for a
condition that would normally FAIL on a minor mismatch — but the LC's
F47A discrepancy-whitelist clause should auto-PASS it because the
condition's subject is NOT in the hard-fail category list.

Anchor: real coal LC f3ef028e — F47A clause 14 says
  "ALL DISCREPANCIES ARE ACCEPTABLE EXCEPT FOR DESCRIPTION OF GOODS,
   QUANTITY, QUALITY, LATEST DATE OF SHIPMENT, PORT OF LOADING AND
   PORT OF DISCHARGE AND ORIGIN OF GOODS."

Test scenarios:
  1. Address typo on Commercial Invoice (NOT in hard-fail list)
     → LLM should PASS thanks to the whitelist banner
  2. Goods description mismatch (IS in hard-fail list)
     → LLM should still FAIL — whitelist does NOT cover this
  3. Quantity mismatch on certificate (IS hard-fail)
     → LLM should still FAIL

Pass criterion:
  - Scenario 1: PASS
  - Scenarios 2+3: FAIL
  - Confirms the LLM correctly reads the banner and applies the rule
"""
import sys, os, json
sys.path.insert(0, 'd:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final')

from steps.step14_verification import (
    _build_f47a_context,
    _build_verification_prompt_v2,
    _call_vlm,
)

results = []
def case(name, expected_compliance, vlm_result):
    got = (vlm_result or {}).get('compliance', '').upper()
    ok = (got == expected_compliance)
    print(f"[{'OK' if ok else 'FAIL'}] {name}: got={got}  expected={expected_compliance}")
    if not ok:
        print(f"      result: {vlm_result.get('result','')[:200]}")
    results.append(ok)

JOB = 'd:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final/results/f3ef028e-b879-40d2-9351-39a2aff90175/step06/step06_result.json'
with open(JOB, 'r', encoding='utf-8') as f:
    real_s6 = json.load(f)

f47a_ctx = _build_f47a_context(real_s6)
final_lc_fields = real_s6.get('consolidated_fields', {})

assert 'DISCREPANCY WHITELIST' in f47a_ctx, "P198ff banner not present in F47A context"
print("--- f47a_context contains DISCREPANCY WHITELIST banner ✓ ---\n")

# ── Scenario 1: minor address typo on Commercial Invoice ──────────────
# Whitelist hard-fail list: GOODS DESCRIPTION, QUANTITY, QUALITY,
# LATEST SHIPMENT DATE, PORT OF LOADING, PORT OF DISCHARGE, ORIGIN.
# An address typo is NOT in this list → LLM should PASS.
prompt1 = _build_verification_prompt_v2(
    condition_text="The Commercial Invoice is correctly addressed to the Applicant.",
    clause_ref="46A-1",
    lc_field_value="MASTER POWER PVT LIMITED, 3-KM OFF RAIWIND, MANGA MANDI ROAD, DIST LAHORE, PAKISTAN.",
    lc_parties="Applicant: MASTER POWER PVT LIMITED",
    f47a_context=f47a_ctx,
    document_type="Commercial Invoice",
    document_text=(
        "COMMERCIAL INVOICE\n"
        "No. INV/INDO/0001\n"
        "TO: MASTERPOWER PVT LIMITED, 3KM OFF RAIWIND-MANGA MANDI ROAD, "
        "LAHORE, PAKISTAN.\n"  # Note: missing space + "DIST" — minor typo
        "DESCRIPTION: BITUMINOUS STEAM COAL IN BULK\n"
        "QUANTITY: 5,000.00 MT\n"
        "PRICE: USD 105.25/MT\n"
        "TOTAL: USD 526,250.00\n"
        "INDONESIA ORIGIN, HS 2701.1200\n"
    ),
    visual_metadata="(none)",
    unified_summary={},
    bl_subtype={},
    final_lc_fields=final_lc_fields,
)
print("Scenario 1: minor address typo (NOT in hard-fail list)")
r1 = _call_vlm(
    row_id="test-1",
    condition_text="The Commercial Invoice is correctly addressed to the Applicant.",
    clause_ref="46A-1",
    lc_field_value="MASTER POWER PVT LIMITED, 3-KM OFF RAIWIND, MANGA MANDI ROAD, DIST LAHORE, PAKISTAN.",
    f47a_context=f47a_ctx,
    document_type="Commercial Invoice",
    document_text=(
        "COMMERCIAL INVOICE\n"
        "No. INV/INDO/0001\n"
        "TO: MASTERPOWER PVT LIMITED, 3KM OFF RAIWIND-MANGA MANDI ROAD, "
        "LAHORE, PAKISTAN.\n"
        "DESCRIPTION: BITUMINOUS STEAM COAL IN BULK\n"
        "QUANTITY: 5,000.00 MT\n"
        "PRICE: USD 105.25/MT\n"
        "TOTAL: USD 526,250.00\n"
        "INDONESIA ORIGIN, HS 2701.1200\n"
    ),
    visual_metadata="(none)",
    lc_parties="Applicant: MASTER POWER PVT LIMITED",
    final_lc_fields=final_lc_fields,
)
case("Scenario 1: address typo → PASS via whitelist", "PASS", r1)
print(f"      reasoning: {r1.get('reasoning', '')[:200]}\n")

# ── Scenario 2: goods description mismatch (IS in hard-fail list) ─────
prompt2 = _build_verification_prompt_v2(
    condition_text="The Commercial Invoice goods description must match the LC.",
    clause_ref="45A-1",
    lc_field_value="BITUMINOUS STEAM COAL IN BULK",
    lc_parties="",
    f47a_context=f47a_ctx,
    document_type="Commercial Invoice",
    document_text=(
        "COMMERCIAL INVOICE\n"
        "DESCRIPTION: ANTHRACITE COAL IN BULK\n"  # WRONG — different coal type
        "QUANTITY: 5000 MT\n"
        "PRICE: USD 105.25/MT\n"
        "INDONESIA ORIGIN\n"
    ),
    visual_metadata="(none)",
    unified_summary={},
    bl_subtype={},
    final_lc_fields=final_lc_fields,
)
print("Scenario 2: goods description mismatch (IS hard-fail)")
r2 = _call_vlm(
    row_id="test-2",
    condition_text="The Commercial Invoice goods description must match the LC.",
    clause_ref="45A-1",
    lc_field_value="BITUMINOUS STEAM COAL IN BULK",
    f47a_context=f47a_ctx,
    document_type="Commercial Invoice",
    document_text=(
        "COMMERCIAL INVOICE\n"
        "DESCRIPTION: ANTHRACITE COAL IN BULK\n"
        "QUANTITY: 5000 MT\n"
        "PRICE: USD 105.25/MT\n"
        "INDONESIA ORIGIN\n"
    ),
    visual_metadata="(none)",
    final_lc_fields=final_lc_fields,
)
case("Scenario 2: goods description mismatch → FAIL (hard-fail category)", "FAIL", r2)
print(f"      reasoning: {r2.get('reasoning', '')[:200]}\n")

# ── Scenario 3: quantity mismatch (IS hard-fail) ──────────────────────
prompt3 = _build_verification_prompt_v2(
    condition_text="The Commercial Invoice quantity must match the LC.",
    clause_ref="45A-2",
    lc_field_value="5,000.00 M.TONS",
    lc_parties="",
    f47a_context=f47a_ctx,
    document_type="Commercial Invoice",
    document_text=(
        "COMMERCIAL INVOICE\n"
        "DESCRIPTION: BITUMINOUS STEAM COAL IN BULK\n"
        "QUANTITY: 4,200 MT\n"  # WRONG — short by 800 MT
        "PRICE: USD 105.25/MT\n"
        "INDONESIA ORIGIN\n"
    ),
    visual_metadata="(none)",
    unified_summary={},
    bl_subtype={},
    final_lc_fields=final_lc_fields,
)
print("Scenario 3: quantity mismatch (IS hard-fail)")
r3 = _call_vlm(
    row_id="test-3",
    condition_text="The Commercial Invoice quantity must match the LC.",
    clause_ref="45A-2",
    lc_field_value="5,000.00 M.TONS",
    f47a_context=f47a_ctx,
    document_type="Commercial Invoice",
    document_text=(
        "COMMERCIAL INVOICE\n"
        "DESCRIPTION: BITUMINOUS STEAM COAL IN BULK\n"
        "QUANTITY: 4,200 MT\n"
        "PRICE: USD 105.25/MT\n"
        "INDONESIA ORIGIN\n"
    ),
    visual_metadata="(none)",
    final_lc_fields=final_lc_fields,
)
case("Scenario 3: quantity mismatch → FAIL (hard-fail category)", "FAIL", r3)
print(f"      reasoning: {r3.get('reasoning', '')[:200]}\n")

passed = sum(results)
total = len(results)
print(f"\n{passed}/{total} LLM scenarios behaved as expected")
if passed != total:
    sys.exit(1)
print("OVERALL: OK — discrepancy-whitelist banner correctly steers LLM verdict")
sys.exit(0)
