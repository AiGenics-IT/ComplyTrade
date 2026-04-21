"""Test: decompose the F46A BL clause and verify each condition against the BL text."""
import sys, json, re
sys.path.insert(0, 'd:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final')

from steps.step12_decomposition import _call_vlm_decompose as _decompose_clause
from steps.step14_verification import _call_vlm

# ─────────────────────────────────────────────────────────────
# LC F46A clause 2 (the one we're testing)
# ─────────────────────────────────────────────────────────────
clause_text = """FULL SET OF SHIPPED ON BOARD OCEAN ORIGINAL BILLS OF LADING
DRAWN OR ENDORSED TO THE ORDER OF UNITED BANK LTD., CPU (TRADE)
PRINTING AND STATIONARY BUILDING, MAI-KOLACHI ROAD, KARACHI,
PAKISTAN, SHOWING 'FREIGHT PREPAID' AND MARKED NOTIFY APPLICANT
AND OPENING BANK."""

clause_ref = "46A-2"
field_tag = "46A"
lc_ctx = {
    "applicant": "DALDA FOODS LIMITED, F-33, HUB RIVER ROAD, S.I.T.E., KARACHI, PAKISTAN",
    "beneficiary": "BUNGE CANADA INC.",
    "issuing_bank": "UNITED BANK LTD., CPU (TRADE), KARACHI",
    "currency_amount": "USD 950000",
}

# ─────────────────────────────────────────────────────────────
# BL text (as pasted by user)
# ─────────────────────────────────────────────────────────────
bl_text = """CODE NAME:"CONGENBILL" EDITION 1994
SHIPPER
BUNGE CANADA INC.,
2625 VICTORIA AVENUE,
REGINA SASKATCHEWAN
S4T7T9, CANADA
CONSIGNEE
TO ORDER
NOTIFY ADDRESS
DALDA FOODS LIMITED
F-33, HUB RIVER ROAD, S.I
T.E., KARACHI, PAKISTAN AND UNITED BANK LTD., CPU (TRADE)
PRINTING
AND
STATIONARY BUILDING, MAI-KOLACHI ROAD, KARACHI, PAKISTAN
VESSEL
M.V. SCION CHARLOTTE
PORT OF LOADING
VANCOUVER, CANADA
PORT OF DISCHARGE
KARACHI PORT OR PORT QASIM, PAKISTAN
SHIPPER'S DESCRIPTION OF GOODS (SAID TO BE)
CANADIAN GMO CANOLA, IN BULK
STOWED IN HOLD NOS. 1, 2, 3, 4, 5, 6, 7 FULL
FREIGHT PREPAID
CLEAN ON BOARD JANUARY 05, 2026
BILL OF LADING
TO BE USED WITH CHARTER-PARTIES
B/L NO.
VANPAK10
Reference No.
ORIGINAL
GROSS WEIGHT (SAID TO WEIGH)
950.000 MT
(of which NIL on deck...)
Freight payable as per
FREIGHT PREPAID
AS PER CHARTER PARTY DATE 09.04.2025
PACIFIC NORTHWEST SHIP & CARGO SERVICES INC.
157 Chadwick Court, North Vancouver, BC, V7P 3N2 Canada
SHIPPED at the Port of Loading in apparent good order and condition...
IN WITNESS whereof the Master or Agent of the said Vessel has signed three Bills of Lading...
FOR CONDITIONS OF CARRIAGE SEE OVERLEAF.
Place and date of issue: VANCOUVER, CANADA JANUARY 5, 2026
Number of original Bs/L: Three (3)
Signature: PACIFIC NORTHWEST SHIP & CARGO SERVICES INC.
AS AGENTS ONLY FOR AND BY AUTHORITY OF CAPTAIN ZBIGNIEW KLUBA THE
MASTER OF M.V. SCION CHARLOTTE
By authority of BIMCO"""

# Structured facts (Step 3 output for this BL)
unified_summary = {
    "document_identifier": "VANPAK10",
    "issue_date": "2026-01-05",
    "consignee": "TO ORDER",
    "notify_party": "DALDA FOODS LIMITED ... AND UNITED BANK LTD., CPU (TRADE) PRINTING AND STATIONARY BUILDING, MAI-KOLACHI ROAD, KARACHI, PAKISTAN",
    "shipper": "BUNGE CANADA INC.",
    "vessel_name": "M.V. SCION CHARLOTTE",
    "port_of_loading": "VANCOUVER, CANADA",
    "port_of_discharge": "KARACHI PORT OR PORT QASIM, PAKISTAN",
    "goods_description": "CANADIAN GMO CANOLA, IN BULK",
    "quantity": "950.000 MT",
    "freight_terms": "FREIGHT PREPAID",
    "number_of_originals": "Three (3)",
    "parties_found": [
        {"role": "shipper", "name": "BUNGE CANADA INC.",
         "raw": "BUNGE CANADA INC., 2625 VICTORIA AVENUE, REGINA SASKATCHEWAN"},
        {"role": "consignee", "name": "TO ORDER", "raw": "CONSIGNEE\nTO ORDER"},
        {"role": "notify_party", "name": "DALDA FOODS LIMITED",
         "raw": "DALDA FOODS LIMITED F-33, HUB RIVER ROAD, S.I T.E., KARACHI, PAKISTAN"},
        {"role": "second_notify_party", "name": "UNITED BANK LTD., CPU (TRADE)",
         "raw": "UNITED BANK LTD., CPU (TRADE) PRINTING AND STATIONARY BUILDING, MAI-KOLACHI ROAD, KARACHI, PAKISTAN"},
        {"role": "carrier", "name": "PACIFIC NORTHWEST SHIP & CARGO SERVICES INC."},
    ],
    "dates_found": [
        {"role": "issue_date", "value": "2026-01-05", "raw": "JANUARY 5, 2026"},
        {"role": "onboard_date", "value": "2026-01-05", "raw": "CLEAN ON BOARD JANUARY 05, 2026"},
        {"role": "charter_party_date", "value": "2025-04-09", "raw": "AS PER CHARTER PARTY DATE 09.04.2025"},
    ],
    "references_found": [
        {"role": "bl_reference", "value": "VANPAK10", "raw": "B/L NO. VANPAK10"},
    ],
    "other_details_found": [
        {"role": "freight_terms", "value": "FREIGHT PREPAID", "raw": "FREIGHT PREPAID"},
    ],
}

bl_subtype = {
    "form_type": "long_form_printed_overleaf",
    "contract_type": "charter_party",
    "issuer_type": "charter_party_bl",
    "signing_type": "agent_for_master",
    "cleanness": "clean",
    "shipped_on_board_status": "shipped_on_board",
    "consigned_form": "to_order",
    "has_terms_overleaf": True,
    "is_blank_back": False,
    "is_house_bl": False,
    "is_charter_party_bl": True,
    "freight_status": "prepaid",
    "carrier_name": "PACIFIC NORTHWEST SHIP & CARGO SERVICES INC.",
    "bl_type_description": "Charter-party BL (CONGENBILL 1994), agent-for-master, clean-on-board, consigned 'TO ORDER'",
}

lc_parties_str = (
    f"APPLICANT: {lc_ctx['applicant']}\n"
    f"BENEFICIARY: {lc_ctx['beneficiary']}\n"
    f"ISSUING BANK: {lc_ctx['issuing_bank']}"
)

# ─────────────────────────────────────────────────────────────
# Step 12: decompose the clause
# ─────────────────────────────────────────────────────────────
print("=" * 78)
print("STEP 12 — DECOMPOSITION")
print("=" * 78)
print(f"Input clause (F{clause_ref}):")
print(clause_text)
print()

result = _decompose_clause(clause_ref, field_tag, 2, clause_text, lc_ctx)
conditions = result.get("conditions", [])
print(f"→ Decomposed into {len(conditions)} verifiable condition(s):\n")
for i, c in enumerate(conditions, 1):
    print(f"  [{i}] document_to_check = {c.get('document_to_check', '?')!r}")
    _cond = c.get('condition_text') or c.get('condition', '?')
    print(f"      condition_text     = {_cond}")
    if c.get('look_for_value'):
        print(f"      look_for_value     = {c.get('look_for_value')}")
    print()

# ─────────────────────────────────────────────────────────────
# Step 14: verify each decomposed condition against the BL
# ─────────────────────────────────────────────────────────────
print("=" * 78)
print("STEP 14 — VERIFICATION (per condition against the BL)")
print("=" * 78)

for i, c in enumerate(conditions, 1):
    if c.get('document_to_check', '').lower() not in ('bill of lading', 'bl', 'all documents'):
        print(f"  [{i}] SKIP — not a BL condition (doc_to_check={c.get('document_to_check')})")
        continue

    cond_text = c.get('condition_text') or c.get('condition', '')
    print(f"\n  [{i}] CONDITION: {cond_text}")
    print(f"       doc_to_check: {c.get('document_to_check')}")

    result = _call_vlm(
        row_id=f"TEST_{i}",
        condition_text=cond_text,
        clause_ref=clause_ref,
        lc_field_value=clause_text,
        f47a_context="(none)",
        document_type="Bill of Lading",
        document_text=bl_text,
        image_path=None,
        visual_metadata="(No visual metadata)",
        lc_parties=lc_parties_str,
        unified_summary=unified_summary,
        bl_subtype=bl_subtype,
        final_lc_fields={"20": "0525ILC082463", "59": lc_ctx['beneficiary']},
    )
    verdict = result.get("compliance", "?").upper()
    findings = result.get("findings", "")
    quote = result.get("quote", "")
    src = result.get("structured_source", "")
    path = result.get("_verification_path", "")
    post = result.get("_post_check", "")

    print(f"       → verdict     : {verdict}")
    print(f"       → findings    : {findings[:200]}")
    if quote:
        print(f"       → quote       : {quote[:120]}")
    if src:
        print(f"       → source      : {src}")
    if path:
        print(f"       → path        : {path}")
    if post:
        print(f"       → post_check  : {post}")

print("\n" + "=" * 78)
print("TEST COMPLETE")
print("=" * 78)
