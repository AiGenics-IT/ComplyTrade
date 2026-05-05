"""P198gz / gz2 / gz3 — classification fixes for job b1479424.

Tests:
- P198gz: bare 'Certificate' must NOT match 'Certificate of Origin' via substring
- P198gz2: step3 'DETAILED MESSAGE' / 'VESSEL ADVICE' overrides packet VLM
- existing P198gj meta-doc guard still fires
"""
import sys, os
sys.path.insert(0, 'd:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final')
os.environ['PYTHONIOENCODING'] = 'utf-8'

from steps.step08_shipping_classification import _match_type_to_requirement

results = []
def ok(name, cond, note=''):
    tag = '[OK]  ' if cond else '[FAIL]'
    print(f"{tag} {name}" + (f" -- {note}" if note else ''))
    results.append(bool(cond))


print("=" * 70)
print("Section 1: Bare-generic substring rejection (P198gz)")
print("=" * 70)

LC_DOCS = [
    {'document_name': 'Certificate of Origin'},
    {'document_name': 'Health Certificate'},
    {'document_name': 'Commercial Invoice'},
    {'document_name': 'Bill of Lading'},
    {'document_name': 'Packing List'},
    {'document_name': 'Beneficiary Certificate'},
    {'document_name': 'Shipment Advice'},
]

CASES = [
    # Bare-generic should NOT match
    ('Certificate',  -1, 'Bare CERTIFICATE -> no match'),
    ('CERTIFICATE',  -1, 'Bare CERTIFICATE upper -> no match'),
    ('Certificates', -1, 'Plural CERTIFICATES -> no match'),
    ('Document',     -1, 'Bare DOCUMENT -> no match'),
    ('Form',         -1, 'Bare FORM -> no match'),
    # Specific should still match
    ('Certificate of Origin', 0, 'Specific CoO matches'),
    ('Health Certificate',    1, 'Health Certificate matches'),
    ('Commercial Invoice',    2, 'CI matches'),
    ('Bill of Lading',        3, 'BL matches'),
    ('OPRINDELSESCERTIFIKAT', -1, 'Foreign-language CoO not matched (alien)'),
    ('Shipment Advice',       6, 'Shipment Advice matches'),
    ('Vessel Advice',         -1, 'Vessel Advice -> alien (P198gz13: distinct from Shipment Advice)'),
]
for dt, expected_idx, label in CASES:
    idx, name = _match_type_to_requirement(dt, LC_DOCS)
    ok(f"  {label}: {dt!r} -> idx={idx}",
       idx == expected_idx,
       f"got idx={idx}, name={name!r}")


print("\n" + "=" * 70)
print("Section 2: Meta-doc guard still works (P198gj)")
print("=" * 70)
META_CASES = [
    ('Certificate of Origin Instructions', -1, 'CoO Instructions blocked'),
    ('Certificate of Origin Guidelines',   -1, 'CoO Guidelines blocked'),
    ('How to Fill Certificate of Origin',  -1, 'How-to blocked'),
    ('Certificate of Origin',               0, 'Real CoO still matches'),
]
for dt, expected_idx, label in META_CASES:
    idx, _ = _match_type_to_requirement(dt, LC_DOCS)
    ok(f"  {label}: {dt!r} -> idx={idx}",
       idx == expected_idx)


print("\n" + "=" * 70)
print("Section 3: Source wiring")
print("=" * 70)
src = open('d:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final/steps/step08_shipping_classification.py',
           'r', encoding='utf-8').read()
ok("  P198gz marker", 'P198gz' in src)
ok("  P198gz2 marker", 'P198gz2' in src)
ok("  _GENERIC_SOLO list", '_GENERIC_SOLO' in src)
ok("  Heading-tag override", 'DETAILED MESSAGE' in src and "VESSEL ADVICE" in src)


print("\n" + "=" * 70)
passed = sum(results)
total = len(results)
print(f"P198gz CLASSIFICATION: {passed}/{total}")
print("=" * 70)
if passed != total:
    sys.exit(1)
print("OVERALL: OK")
sys.exit(0)
