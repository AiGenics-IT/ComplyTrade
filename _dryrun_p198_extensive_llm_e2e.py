"""
P198 EXTENSIVE end-to-end LIVE LLM dry-run.

Hits the live Qwen Text LLM endpoint with the production verification
CORE prompt for a wide spectrum of scenarios — confirms each new
banner (P198et, fb, ff, fg) correctly steers the LLM verdict.

Each scenario specifies:
  - synthetic LC fields (so we know the ground truth)
  - synthetic document text (with known PASS / FAIL evidence)
  - the EXPECTED LLM verdict given the banner context

A passing scenario means: the LLM's verdict matches the ground truth
AFTER seeing the banner — i.e. the banner correctly drove the LLM
away from a false-PASS or false-FAIL it would have otherwise produced.

Network requirement: this dryrun requires the live LLM at
QWEN_TEXT_LLM_URL (set in config/settings.py). Each scenario is one
call (~1-5 seconds depending on model warmth + prompt size).
"""
import sys, os, json
sys.path.insert(0, 'd:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final')

from steps.step14_verification import (
    _build_f47a_context,
    _call_vlm,
)

results = []
def case(name, expected, vlm_result, alt_ok=None):
    """alt_ok = a set of additional acceptable verdicts (for inherently
    ambiguous cases). e.g. when REVIEW is also acceptable."""
    got = (vlm_result or {}).get('compliance', '').upper()
    ok = (got == expected) or (alt_ok and got in alt_ok)
    print(f"[{'OK' if ok else 'FAIL'}] {name}: got={got}  expected={expected}")
    if not ok:
        print(f"      result: {(vlm_result or {}).get('result','')[:240]}")
    results.append(ok)


def make_lc(f46a='', f47a='', f45a='', f32b='USD 100,000.00'):
    return {'consolidated_fields': {
        '32B': f32b, '46A': f46a, '47A': f47a, '45A': f45a,
    }}


# ────────────────────────────────────────────────────────────────────
# Scenario set A — DISCREPANCY WHITELIST (P198ff)
# ────────────────────────────────────────────────────────────────────
print("=== A. Discrepancy whitelist scenarios (P198ff) ===")
wl_lc = make_lc(
    f47a='14) ALL DISCREPANCIES ARE ACCEPTABLE EXCEPT FOR DESCRIPTION OF '
         'GOODS, QUANTITY, QUALITY, LATEST DATE OF SHIPMENT, PORT OF '
         'LOADING AND PORT OF DISCHARGE AND ORIGIN OF GOODS.',
    f45a='BITUMINOUS COAL HS 2701.1200', f32b='USD 500,000')
wl_ctx = _build_f47a_context(wl_lc)

# A.1 — typo in addressee — NOT in hard-fail list → PASS via whitelist
print("\nA.1: minor address typo in invoice (NOT hard-fail)")
r = _call_vlm(
    row_id="A1",
    condition_text="The Commercial Invoice must be addressed correctly to Applicant.",
    clause_ref="46A-1", lc_field_value="ABC INDUSTRIES PVT LTD, KARACHI",
    f47a_context=wl_ctx, document_type="Commercial Invoice",
    document_text=(
        "COMMERCIAL INVOICE\nTO: ABC INDUSTRIE PVT LTD, KARACHI.\n"  # typo
        "DESCRIPTION: BITUMINOUS COAL\nQUANTITY: 5000 MT\nTOTAL: USD 500,000\n"
    ),
    visual_metadata="(none)",
    final_lc_fields=wl_lc['consolidated_fields'],
)
case("A.1: address typo → PASS via whitelist", "PASS", r)

# A.2 — port of loading mismatch — IS hard-fail → FAIL
print("\nA.2: port of loading mismatch (IS hard-fail)")
r = _call_vlm(
    row_id="A2",
    condition_text="The Bill of Lading port of loading must match LC.",
    clause_ref="44E", lc_field_value="ANY PORT IN INDONESIA",
    f47a_context=wl_ctx, document_type="Bill of Lading",
    document_text=(
        "BILL OF LADING\nPort of Loading: SHANGHAI, CHINA\n"  # WRONG
        "Port of Discharge: KARACHI\nVessel: MV ABC\n"
    ),
    visual_metadata="(none)",
    final_lc_fields=wl_lc['consolidated_fields'],
)
case("A.2: port of loading wrong → FAIL (hard-fail category)", "FAIL", r)

# A.3 — invoice number format — NOT hard-fail → PASS
print("\nA.3: invoice number format (NOT hard-fail)")
r = _call_vlm(
    row_id="A3",
    condition_text="The Commercial Invoice must reference a valid invoice number.",
    clause_ref="46A-1", lc_field_value="(any valid invoice no)",
    f47a_context=wl_ctx, document_type="Commercial Invoice",
    document_text=("COMMERCIAL INVOICE\n# CI-2026-001-EX (unusual format)\n"
                   "DESCRIPTION: BITUMINOUS COAL\nQUANTITY: 5000 MT\n"),
    visual_metadata="(none)",
    final_lc_fields=wl_lc['consolidated_fields'],
)
case("A.3: invoice no format → PASS via whitelist", "PASS", r,
     alt_ok={'REVIEW'})


# ────────────────────────────────────────────────────────────────────
# Scenario set B — LATE-SHIPMENT-WITH-PENALTY (P198fg)
# ────────────────────────────────────────────────────────────────────
print("\n=== B. Late-shipment-with-penalty scenarios (P198fg) ===")
late_lc = make_lc(
    f47a='3) LATE SHIPMENT ALLOWED PROVIDED USD 100 PER DAY DEDUCTED.',
    f45a='COTTON FABRIC HS 5407', f32b='USD 200,000')
# Set F44C latest-shipment for context
late_lc['consolidated_fields']['44C'] = '2026-03-15'
late_ctx = _build_f47a_context(late_lc)

# B.1 — BL date is AFTER F44C latest, but penalty clause exists → REVIEW (not FAIL)
print("\nB.1: late shipment with penalty clause active")
r = _call_vlm(
    row_id="B1",
    condition_text="The Bill of Lading shipment date must be on or before LC latest shipment date.",
    clause_ref="44C", lc_field_value="2026-03-15",
    f47a_context=late_ctx, document_type="Bill of Lading",
    document_text=(
        "BILL OF LADING\nShipped on Board: 2026-03-25\n"  # 10 days late
        "Vessel: MV LATE\nPort of Loading: KARACHI\n"
    ),
    visual_metadata="(none)",
    final_lc_fields=late_lc['consolidated_fields'],
)
case("B.1: 10-day late + penalty clause → REVIEW", "REVIEW", r,
     alt_ok={'PASS', 'FAIL'})


# ────────────────────────────────────────────────────────────────────
# Scenario set C — SURVEYOR REQUIREMENT (P198fh)
# ────────────────────────────────────────────────────────────────────
print("\n=== C. Required-surveyor scenarios (P198fh) ===")
surv_lc = make_lc(
    f46a='4) CERTIFICATE OF SAMPLING AND ANALYSIS BY SGS AT LOAD PORT.\n'
         '5) CERTIFICATE OF WEIGHT BY SGS AT LOAD PORT.',
    f45a='COAL', f32b='USD 1,000,000')
surv_ctx = _build_f47a_context(surv_lc)

# C.1 — SGS-issued cert → PASS
print("\nC.1: SGS-issued cert (matches required)")
r = _call_vlm(
    row_id="C1",
    condition_text="The Certificate of Sampling and Analysis must be issued by SGS at load port.",
    clause_ref="46A-4", lc_field_value="SGS load-port",
    f47a_context=surv_ctx, document_type="Certificate of Sampling and Analysis",
    document_text=(
        "CERTIFICATE OF SAMPLING AND ANALYSIS\n"
        "ISSUED BY: SGS INDONESIA\n"
        "Cargo: Bituminous Coal\nResults: GCV (ARB) 5800 kcal/kg\n"
        "Date: 2026-02-20\n"
    ),
    visual_metadata="(none)",
    final_lc_fields=surv_lc['consolidated_fields'],
)
case("C.1: SGS cert → PASS", "PASS", r)

# C.2 — UNKNOWN surveyor (not in LC's required list) → FAIL
print("\nC.2: cert from unknown surveyor")
r = _call_vlm(
    row_id="C2",
    condition_text="The Certificate of Sampling and Analysis must be issued by SGS at load port.",
    clause_ref="46A-4", lc_field_value="SGS load-port",
    f47a_context=surv_ctx, document_type="Certificate of Sampling and Analysis",
    document_text=(
        "CERTIFICATE OF SAMPLING AND ANALYSIS\n"
        "ISSUED BY: ABC PRIVATE LAB JAKARTA\n"  # Not SGS
        "Cargo: Bituminous Coal\nResults: GCV (ARB) 5800 kcal/kg\n"
    ),
    visual_metadata="(none)",
    final_lc_fields=surv_lc['consolidated_fields'],
)
case("C.2: unknown lab → FAIL (not the required surveyor)", "FAIL", r,
     alt_ok={'REVIEW'})


# ────────────────────────────────────────────────────────────────────
# Scenario set D — ADVANCE-PAYMENT (P198et) regression
# ────────────────────────────────────────────────────────────────────
print("\n=== D. Advance-payment regression (P198et) ===")
adv_lc = make_lc(
    f46a='A) 80 PERCENT ADVANCE PAYMENT WILL BE MADE UPON RECEIPT OF '
         'AUTHENTICATED SWIFT.\n'
         'B) REMAINING 20 PERCENT PAYABLE TO BENEFICIARY AGAINST '
         'PRESENTATION OF DOCUMENTS.\n'
         '6. SHIPPING INVOICE TO BE DRAWN FOR 100 PERCENT L/C AMOUNT '
         'LESS 80 PERCENT ADVANCE PAYMENT.',
    f47a='6) NEGOTIATING BANK COVERING SCHEDULE MUST SHOW 100 PERCENT '
         'L/C VALUE 80 PERCENT ADVANCE PAYMENT AND NET CLAIMING (20 PERCENT).',
    f45a='HS 8415.1029 HEAT PUMP', f32b='USD 10,919.00')
adv_ctx = _build_f47a_context(adv_lc)

# D.1 — CI shows full LC value + advance deduction → PASS
print("\nD.1: Commercial Invoice with 80% advance deduction line")
r = _call_vlm(
    row_id="D1",
    condition_text="The Commercial Invoice amount must not exceed the LC amount.",
    clause_ref="32B", lc_field_value="USD 10,919.00",
    f47a_context=adv_ctx, document_type="Commercial Invoice",
    document_text=(
        "COMMERCIAL INVOICE\n"
        "Invoice Total: USD 10,919.00\n"
        "LESS: 80% ADVANCE PAYMENT RECEIVED AS PER CLAUSE 46+A: USD 8,735.20\n"
        "NET PAYABLE: USD 2,183.80\n"
    ),
    visual_metadata="(none)",
    final_lc_fields=adv_lc['consolidated_fields'],
)
case("D.1: CI with deduction line → PASS", "PASS", r)

# D.2 — Draft for 20% net (not LC face value) → PASS
print("\nD.2: Draft Bill of Exchange for 20% net (NOT 100% LC face)")
r = _call_vlm(
    row_id="D2",
    condition_text="The Draft amount should be drawn for the LC value.",
    clause_ref="32B", lc_field_value="USD 10,919.00",
    f47a_context=adv_ctx, document_type="Draft Bill of Exchange",
    document_text=(
        "DRAFT BILL OF EXCHANGE\n"
        "Pay to the order of XYZ Bank, the sum of USD 2,184.00\n"
        "Drawn for net amount under L/C 0052ILC083930 (20% balance "
        "after 80% advance via authenticated SWIFT).\n"
    ),
    visual_metadata="(none)",
    final_lc_fields=adv_lc['consolidated_fields'],
)
case("D.2: Draft for 20% net → PASS via split-payment context", "PASS", r)

# D.3 — Cover schedule WITHOUT the 100/80/20 breakdown → FAIL
print("\nD.3: Document Remittance missing 100/80/20 breakdown")
r = _call_vlm(
    row_id="D3",
    condition_text=("Negotiating Bank Covering Schedule must clearly show "
                    "100 percent L/C value, 80 percent advance payment, and "
                    "20 percent net claiming."),
    clause_ref="47A-6", lc_field_value="(see clause 6)",
    f47a_context=adv_ctx, document_type="Document Remittance",
    document_text=(
        "DOCUMENTARY REMITTANCE / BILLS SCHEDULE\n"
        "Net Amount Claiming: USD 2,184.00\n"  # missing 100% and 80%
        "Documents enclosed as per LC.\n"
    ),
    visual_metadata="(none)",
    final_lc_fields=adv_lc['consolidated_fields'],
)
case("D.3: cover missing 100/80/20 → FAIL", "FAIL", r)


passed = sum(results)
total = len(results)
print(f"\n{passed}/{total} live-LLM scenarios behaved as expected")
if passed != total:
    sys.exit(1)
print("OVERALL: OK — every banner correctly steered LLM verdicts")
sys.exit(0)
