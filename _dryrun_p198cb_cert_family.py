"""P198cb dry-run: certificate-family guard prevents Health / Phyto /
Halal / etc. from being fuzzy-matched to Shipping Company Certificate.

Uses the actual expected_docs list from job e07ce444's step06 result."""
import json
import re
import sys

sys.path.insert(0, '.')
from steps.step08_shipping_classification import _match_type_to_requirement

# Load real expected_docs from the job's LC
with open('results/e07ce444-aa33-4aa7-a380-ba6d182a05a6/step07/step07_result.json',
          encoding='utf-8') as f:
    s7 = json.load(f)

# expected_docs come from step 7 / step 8 inference — simulate a typical
# LC with shipping company cert requirement
expected_docs_from_lc = [
    {'document_name': 'Commercial Invoice'},
    {'document_name': 'Bill of Lading'},
    {'document_name': 'Shipping Company Certificate'},
    {'document_name': 'Certificate from Owner of the Vessel'},
    {'document_name': 'Halal Certificate'},
    {'document_name': 'Health Certificate'},
    {'document_name': 'Beneficiary Certificate'},
    {'document_name': 'Packing List'},
    {'document_name': 'Certificate of Origin'},
    {'document_name': 'Inspection Certificate'},
    {'document_name': 'Quality Certificate'},
    {'document_name': 'Draft Bill of Exchange'},
    {'document_name': 'Shipment Advice'},
]

# LC where SCC is present but Health Cert is NOT in expected — simulates
# the bug scenario where Health Cert would wrongly match SCC.
expected_docs_scc_only = [
    {'document_name': 'Commercial Invoice'},
    {'document_name': 'Bill of Lading'},
    {'document_name': 'Shipping Company Certificate'},
    {'document_name': 'Certificate of Origin'},
]

# LC with BOTH SCC and Vessel Owner Cert (common case)
expected_docs_both_carrier = [
    {'document_name': 'Shipping Company Certificate'},
    {'document_name': 'Certificate from Owner of the Vessel'},
    {'document_name': 'Bill of Lading'},
]

# LC with only vessel-owner cert
expected_docs_vessel_only = [
    {'document_name': 'Bill of Lading'},
    {'document_name': 'Certificate from Owner of the Vessel'},
]

# LC with Carrier Certificate (synonym)
expected_docs_carrier = [
    {'document_name': "Carrier's Certificate"},
    {'document_name': 'Bill of Lading'},
]

# Test cases
cases = [
    # (doc_type, expected_docs, expected_match_name, description)
    ('Health Certificate', expected_docs_from_lc, 'Health Certificate',
     'Health → matches Health when present'),
    ('Health Certificate', expected_docs_scc_only, '',
     'Health → NO match when only SCC in expected (was wrongly matching SCC)'),
    ('Shipping Company Certificate', expected_docs_from_lc,
     'Shipping Company Certificate',
     'SCC → matches SCC'),
    ("Agent's Certificate", expected_docs_from_lc,
     'Shipping Company Certificate',
     'Agent Certificate → SCC family (same carrier-attestation family)'),
    ("Vessel Owner's Certificate", expected_docs_from_lc,
     'Certificate from Owner of the Vessel',
     'Vessel owner cert → matches vessel-owner requirement'),
    ('Halal Certificate', expected_docs_from_lc, 'Halal Certificate',
     'Halal → Halal'),
    ('Halal Certificate', expected_docs_scc_only, '',
     'Halal → NO match when only SCC (was wrongly matching SCC)'),
    ('Phytosanitary Certificate', expected_docs_scc_only, '',
     'Phyto → NO match when only SCC'),
    ('Fumigation Certificate', expected_docs_scc_only, '',
     'Fumigation → NO match when only SCC'),
    ('Beneficiary Certificate', expected_docs_from_lc, 'Beneficiary Certificate',
     'Beneficiary → Beneficiary'),
    ('Beneficiary Certificate', expected_docs_scc_only, '',
     'Beneficiary → NO match when only SCC'),
    ('Certificate of Origin', expected_docs_from_lc, 'Certificate of Origin',
     'Certificate of Origin → itself'),
    ('Inspection Certificate', expected_docs_from_lc, 'Inspection Certificate',
     'Inspection → itself'),
    ('Quality Certificate', expected_docs_from_lc, 'Quality Certificate',
     'Quality → Quality'),
    # Non-cert docs unchanged
    ('Bill of Lading', expected_docs_from_lc, 'Bill of Lading', 'BL → BL'),
    ('Commercial Invoice', expected_docs_from_lc, 'Commercial Invoice',
     'CI → CI'),
    ('Packing List', expected_docs_from_lc, 'Packing List', 'PL → PL'),
    ('Draft Bill of Exchange', expected_docs_from_lc, 'Draft Bill of Exchange',
     'Draft → Draft'),

    # ── Extended carrier-attestation family cases ──
    ("Carrier's Certificate", expected_docs_carrier, "Carrier's Certificate",
     'Carrier cert → carrier cert (exact)'),
    ("Agent's Certificate", expected_docs_carrier, "Carrier's Certificate",
     'Agent cert → Carrier cert (same family)'),
    ('Shipping Company Certificate', expected_docs_carrier, "Carrier's Certificate",
     'SCC → Carrier cert (same family)'),

    # ── Prefer more-specific within same family ──
    ("Vessel Owner's Certificate", expected_docs_both_carrier,
     'Certificate from Owner of the Vessel',
     'Vessel Owner → prefers Vessel-Owner cert over SCC when both in LC'),
    ("Ship Owner's Certificate", expected_docs_both_carrier,
     'Certificate from Owner of the Vessel',
     'Ship Owner → prefers Vessel-Owner cert over SCC'),
    ('Shipping Company Certificate', expected_docs_both_carrier,
     'Shipping Company Certificate',
     'SCC (doc) → SCC (LC) — both present, SCC wins'),

    # ── LC has only Vessel Owner Cert, doc is generic SCC ──
    ('Shipping Company Certificate', expected_docs_vessel_only,
     'Certificate from Owner of the Vessel',
     'SCC doc → Vessel-Owner (only carrier cert in LC, same family)'),

    # ── Exotic cross-family ──
    ('Weight Certificate', expected_docs_scc_only, '',
     'Weight cert → NO match when only SCC in LC'),
    ('Weight Certificate', expected_docs_from_lc, 'Quality Certificate',
     'Weight cert → Quality cert (weight_quality family)'),
]

print("=" * 78)
print("P198cb certificate-family guard — dry-run")
print("=" * 78)

passed = 0
for dt, expected, exp_name, desc in cases:
    idx, name = _match_type_to_requirement(dt, expected)
    ok = 'OK' if name == exp_name else 'FAIL'
    if ok == 'OK':
        passed += 1
    print(f'  [{ok}] {dt!r:34} → matched={name!r:34} (expected {exp_name!r:34}) -- {desc}')

print()
print(f'{passed}/{len(cases)} cases correct')

# ── Confirm against actual 3 mislabeled packets from job e07ce444 ──
print()
print("=" * 78)
print("Against actual job e07ce444 pages 6, 11, 38 (HEALTH CERTIFICATE text)")
print("=" * 78)
# The pages 6/11/38 VLM would probably say "Health Certificate". With SCC
# in the LC's expected_docs (as it is in this job), under OLD logic the
# fuzzy CERTIFICATE overlap pulled it to SCC.
for pg in (6, 11, 38):
    idx, name = _match_type_to_requirement('Health Certificate', expected_docs_from_lc)
    exp_name = 'Health Certificate'
    ok = 'OK' if name == exp_name else 'FAIL'
    print(f'  [{ok}] page {pg}: Health Certificate → matched {name!r}  (expected {exp_name})')
