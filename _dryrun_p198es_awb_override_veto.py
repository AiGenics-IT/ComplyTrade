"""
P198es dry-run — AWB rule override must NOT fire when the VLM
confirmed the page is a "Bill of Lading".

Background:
  The lexical rule scorer in step08 over-matches AWB / Courier Receipt
  at 0.99 for ANY page with shipping content, including genuine BLs.
  Without a veto, the override suppresses the VLM's correct "Bill of
  Lading" classification and (combined with the post-P198eo matcher
  that no longer maps AWB->BL) turns every BL into an alien Airway
  Bill. This is a regression we must NOT ship.

The fix at step08_shipping_classification.py:1009-1037 adds a guard:
  if VLM said "Bill of Lading" (or any BL alias), the rule override is
  vetoed and the VLM classification stands.

This dry-run feeds synthetic VLM result + rule_matches inputs and
asserts that the override's `rule_override_type` is None when the VLM
said BL, but still fires when the VLM said something else (Courier
Receipt / Shipment Advice / unknown / etc.).
"""
import sys
sys.path.insert(0, 'd:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final')

# We re-implement the relevant override block here — exact mirror of
# the production code at step08_shipping_classification.py:1009-1037.
# The dry-run validates the LOGIC, not the imports (which would pull in
# heavy deps like httpx / VLM SDKs).
def compute_override_type(vlm_result, rule_matches, expected_docs):
    """Mirror of the override block from step08."""
    rule_override_type = None
    _vlm_dt_for_veto = ''
    if vlm_result:
        _vlm_dt_for_veto = (vlm_result.get('document_type') or '').strip().lower()
    _vlm_says_bl = (
        'bill of lading' in _vlm_dt_for_veto
        or _vlm_dt_for_veto in {'b/l', 'bl', 'congenbill',
                                 'master bill of lading', 'house bill of lading',
                                 'ocean bill of lading', 'marine bill of lading',
                                 'multimodal bill of lading',
                                 'combined transport bill of lading'}
    )

    if rule_matches:
        top_score = rule_matches[0].get('score', 0)
        top_names_at_99 = {m['document_name'] for m in rule_matches
                           if m.get('score', 0) >= 0.99}

        if top_score >= 0.99:
            if {'Courier Receipt', 'Airway Bill'} & top_names_at_99 \
                    and not _vlm_says_bl:
                preferred = None
                for ed in expected_docs:
                    en = (ed.get('document_name') or '').upper()
                    if 'COURIER' in en:
                        preferred = 'Courier Receipt'; break
                    if 'AIR' in en and ('WAY' in en or 'BILL' in en):
                        preferred = 'Airway Bill'; break
                rule_override_type = preferred or 'Airway Bill'
            elif rule_matches[0]['document_name'] == 'Documentary Remittance':
                rule_override_type = 'Documentary Remittance'
            elif rule_matches[0]['document_name'] in (
                'Health Certificate', 'Phytosanitary Certificate',
                'Fumigation Certificate',
            ):
                rule_override_type = rule_matches[0]['document_name']
    return rule_override_type


# Common AWB/CR rule_matches scoring (over-matches at 0.99 for any
# shipping content)
AWB_FIRES = [
    {"document_name": "Airway Bill", "score": 0.99},
    {"document_name": "Courier Receipt", "score": 0.99},
    {"document_name": "Bill of Lading", "score": 0.375},
]
LC_BL_ONLY    = [{"document_name": "Bill of Lading"}]
LC_BL_AND_AWB = [{"document_name": "Bill of Lading"},
                 {"document_name": "Airway Bill"}]
LC_BL_AND_CR  = [{"document_name": "Bill of Lading"},
                 {"document_name": "Courier Receipt"}]


def case(name, vlm_dt, rule_matches, expected_docs, expected):
    got = compute_override_type({"document_type": vlm_dt}, rule_matches, expected_docs)
    ok = (got == expected)
    print(f"[{'OK' if ok else 'FAIL'}] {name:60s} vlm={vlm_dt!r:25s} -> override={got!r:20s} expected={expected!r}")
    return ok


results = []

# ── Veto cases: VLM said BL, override must be VETOED ─────────────────────
results.append(case("VLM=BL, AWB rule fires, LC=BL → veto",
    "Bill of Lading", AWB_FIRES, LC_BL_ONLY, None))
results.append(case("VLM=BL upper, AWB rule fires → veto",
    "BILL OF LADING", AWB_FIRES, LC_BL_ONLY, None))
results.append(case("VLM=B/L abbreviated, AWB rule fires → veto",
    "B/L", AWB_FIRES, LC_BL_ONLY, None))
results.append(case("VLM=Master Bill of Lading, AWB rule → veto",
    "Master Bill of Lading", AWB_FIRES, LC_BL_ONLY, None))
results.append(case("VLM=House Bill of Lading, AWB rule → veto",
    "House Bill of Lading", AWB_FIRES, LC_BL_ONLY, None))
results.append(case("VLM=Ocean BL, AWB rule → veto",
    "Ocean Bill of Lading", AWB_FIRES, LC_BL_ONLY, None))
results.append(case("VLM=Marine BL, AWB rule → veto",
    "marine bill of lading", AWB_FIRES, LC_BL_ONLY, None))
results.append(case("VLM=CongenBill, AWB rule → veto",
    "Congenbill", AWB_FIRES, LC_BL_ONLY, None))

# ── Override should still fire when VLM did NOT say BL ──────────────────
results.append(case("VLM=Courier Receipt, AWB rule fires, LC=BL → AWB alien",
    "Courier Receipt", AWB_FIRES, LC_BL_ONLY, "Airway Bill"))
results.append(case("VLM=Airway Bill, AWB rule fires, LC=BL → AWB alien",
    "Airway Bill", AWB_FIRES, LC_BL_ONLY, "Airway Bill"))
results.append(case("VLM=Shipment Advice, AWB rule fires, LC=BL → AWB alien",
    "Shipment Advice", AWB_FIRES, LC_BL_ONLY, "Airway Bill"))
results.append(case("VLM=unknown, AWB rule fires, LC=BL → AWB alien",
    "unknown", AWB_FIRES, LC_BL_ONLY, "Airway Bill"))
results.append(case("VLM=Courier, LC has Courier → preferred=Courier Receipt",
    "Courier Receipt", AWB_FIRES, LC_BL_AND_CR, "Courier Receipt"))
results.append(case("VLM=Airway Bill, LC has AWB → preferred=Airway Bill",
    "Airway Bill", AWB_FIRES, LC_BL_AND_AWB, "Airway Bill"))

# ── Override should NOT fire if rule_matches isn't 0.99 ─────────────────
weak = [{"document_name":"Airway Bill","score":0.7},
        {"document_name":"Bill of Lading","score":0.6}]
results.append(case("AWB rule weak (0.7), VLM=Courier → no override",
    "Courier Receipt", weak, LC_BL_ONLY, None))

# ── Health/Phyto/Fumigation certificate rescue still works ──────────────
hc = [{"document_name":"Health Certificate","score":0.99}]
results.append(case("Health Cert rule fires → override to Health Certificate",
    "Shipping Company Certificate", hc, LC_BL_ONLY, "Health Certificate"))

# ── DR override still works ────────────────────────────────────────────
dr = [{"document_name":"Documentary Remittance","score":0.99}]
results.append(case("DR rule fires → override to Documentary Remittance",
    "Beneficiary Certificate", dr, LC_BL_ONLY, "Documentary Remittance"))

# ── Edge case: VLM empty, AWB rule fires → AWB ─────────────────────────
results.append(case("VLM empty, AWB rule fires → AWB",
    "", AWB_FIRES, LC_BL_ONLY, "Airway Bill"))

passed = sum(results)
total = len(results)
print(f"\n{passed}/{total} cases passed")
sys.exit(0 if passed == total else 1)
