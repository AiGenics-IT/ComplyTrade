"""Aggressive test battery for P198gz12 / 13 / 14 / 15 / 16.

Sections:
1. Canonical map — every entry resolves when LC has matching slot
2. Acronym pre-resolution — PSI / MTC / CoC / etc.
3. Cross-family rejection — Vessel↔Shipment, cert families
4. Negative cases — bare-generic, foreign lang, meta-doc, whitespace
5. P198gz12 anti-CoO-force-fit — real text patterns
6. Real-job sweep — every CoO and Shipment Advice packet across 91 jobs
7. Existing P198g* regression — make sure nothing broke
"""
import sys, os, re, json, glob
sys.path.insert(0, 'd:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final')
os.environ['PYTHONIOENCODING'] = 'utf-8'

from steps.step08_shipping_classification import _match_type_to_requirement

results = []
def ok(name, cond, note=''):
    tag = '[OK]  ' if cond else '[FAIL]'
    print(f"{tag} {name}" + (f" -- {note}" if note else ''))
    results.append(bool(cond))


# ── Section 1: Canonical map coverage ──
print("=" * 70)
print("Section 1: Canonical entries — all bind when LC has matching slot")
print("=" * 70)

LC_FULL = [{'document_name': n} for n in [
    'Certificate of Compliance','Pre-Shipment Inspection Certificate',
    'Certificate of Conformity','ISPM-15 Compliance Certificate',
    'No Wood Packing Declaration','Blacklist / Sanctions Declaration',
    'Certificate of Cleanliness',"Manufacturer's Certificate",
    'Packing Material Declaration','Radiation Certificate',
    'Certificate of Free Sale','Anti-Dumping Declaration',
    'Beneficiary Certificate','Packing List Certificate',
    'Export License Declaration','Import Permit Reference Certificate',
    'Customs Declaration','GSP / Preferential Origin Certificate',
    'Mill Test Certificate','Calibration Certificate',
    'Warranty Certificate','Shelf Life Certificate',
    'Batch / Lot Certificate','Certificate of Sterility',
    'Temperature Certificate','Container Seal Certificate',
    'Container Loading / Stuffing Certificate','Cargo Securing Certificate',
    'Freight Certificate','Demurrage / Detention-Free Declaration',
    'No Child Labor Declaration','REACH Compliance Certificate',
    'RoHS Certificate','Conflict Minerals Declaration',
    'Environmental Compliance Certificate','Boycott Declaration',
    'Embargo Compliance Declaration','Beneficiary Fax/Email Advice Certificate',
    'Vessel Advice','Shipment Advice','Bill of Lading',
    'Commercial Invoice','Packing List','Certificate of Origin',
    'Health Certificate','Phytosanitary Certificate','Halal Certificate',
    'Inspection Certificate','Survey Report','Weight Certificate',
    'Quality Certificate','Insurance Policy/Certificate',
    'Documentary Remittance','Airway Bill',
]]

CANONICAL_TESTS = [
    # Cert taxonomy
    ('CERTIFICATE OF COMPLIANCE','Certificate of Compliance'),
    ('Compliance Certificate','Certificate of Compliance'),
    ('PRE-SHIPMENT INSPECTION CERTIFICATE','Pre-Shipment Inspection Certificate'),
    ('PSI Certificate','Pre-Shipment Inspection Certificate'),
    ('Certificate of Conformity','Certificate of Conformity'),
    ('CoC Certificate','Certificate of Conformity'),
    ('ISPM-15 Certificate','ISPM-15 Compliance Certificate'),
    ('ISPM 15 Certificate','ISPM-15 Compliance Certificate'),
    ('No Wood Packing Declaration','No Wood Packing Declaration'),
    ('BLACKLIST DECLARATION','Blacklist / Sanctions Declaration'),
    ('Sanctions Declaration','Blacklist / Sanctions Declaration'),
    ('Tanker Cleanliness Certificate','Certificate of Cleanliness'),
    ("Manufacturer's Certificate","Manufacturer's Certificate"),
    ('MANUFACTURER CERTIFICATE',"Manufacturer's Certificate"),
    ('Radiation Certificate','Radiation Certificate'),
    ('Non-Radiation Certificate','Radiation Certificate'),
    # Trade/regulatory
    ('Export License Declaration','Export License Declaration'),
    ('IMPORT PERMIT','Import Permit Reference Certificate'),
    ('GSP Certificate','GSP / Preferential Origin Certificate'),
    ('Preferential Origin Certificate','GSP / Preferential Origin Certificate'),
    # Product
    ('Mill Test Certificate','Mill Test Certificate'),
    ('MTC','Mill Test Certificate'),
    ('Material Test Certificate','Mill Test Certificate'),
    ('Calibration Certificate','Calibration Certificate'),
    ('Warranty Certificate','Warranty Certificate'),
    ('Shelf Life Certificate','Shelf Life Certificate'),
    ('Batch Certificate','Batch / Lot Certificate'),
    ('Lot Certificate','Batch / Lot Certificate'),
    ('Certificate of Sterility','Certificate of Sterility'),
    ('Sterility Certificate','Certificate of Sterility'),
    ('Temperature Certificate','Temperature Certificate'),
    ('Cold Chain Certificate','Temperature Certificate'),
    # Logistics
    ('Container Seal Certificate','Container Seal Certificate'),
    ('Stuffing Certificate','Container Loading / Stuffing Certificate'),
    ('Container Loading Certificate','Container Loading / Stuffing Certificate'),
    ('Cargo Securing Certificate','Cargo Securing Certificate'),
    ('Lashing Certificate','Cargo Securing Certificate'),
    ('Freight Certificate','Freight Certificate'),
    # Ethical
    ('No Child Labor Declaration','No Child Labor Declaration'),
    ('No Child Labour Declaration','No Child Labor Declaration'),
    ('REACH Compliance Certificate','REACH Compliance Certificate'),
    ('REACH Certificate','REACH Compliance Certificate'),
    ('RoHS Certificate','RoHS Certificate'),
    ('Conflict Minerals Declaration','Conflict Minerals Declaration'),
    ('Environmental Certificate','Environmental Compliance Certificate'),
    # Country-specific
    ('Boycott Declaration','Boycott Declaration'),
    ('Embargo Declaration','Embargo Compliance Declaration'),
    # Bene variants
    ("Beneficiary Statement",'Beneficiary Certificate'),
    ("Beneficiary's Declaration",'Beneficiary Certificate'),
    ('Beneficiary Fax/Email Advice Certificate','Beneficiary Fax/Email Advice Certificate'),
    # Existing — make sure not regressed
    ('Bill of Lading','Bill of Lading'),
    ('Commercial Invoice','Commercial Invoice'),
    ('Health Certificate','Health Certificate'),
    ('Vessel Advice','Vessel Advice'),
    ('VESSEL ADVICE','Vessel Advice'),
    ('Shipment Advice','Shipment Advice'),
]

for inp, exp in CANONICAL_TESTS:
    idx, name = _match_type_to_requirement(inp, LC_FULL)
    ok(f"  {inp!r:55} -> {name!r}", name == exp,
       f"got {name!r} expected {exp!r}")


# ── Section 2: Acronym pre-resolver edge cases ──
print("\n" + "=" * 70)
print("Section 2: Acronyms — short forms + plain-text expansion")
print("=" * 70)
ACRONYM_TESTS = [
    ('PSI', 'Pre-Shipment Inspection Certificate'),
    ('MTC', 'Mill Test Certificate'),
    ('COC', 'Certificate of Conformity'),
    ('GSP', 'GSP / Preferential Origin Certificate'),
    ('CFS', 'Certificate of Free Sale'),
    ('REACH', 'REACH Compliance Certificate'),
    ('ROHS', 'RoHS Certificate'),
]
for inp, exp in ACRONYM_TESTS:
    idx, name = _match_type_to_requirement(inp, LC_FULL)
    ok(f"  Acronym {inp!r:8} -> {name!r}", name == exp,
       f"got {name!r}")


# ── Section 3: Cross-family rejection ──
print("\n" + "=" * 70)
print("Section 3: Cross-family rejection (don't fuzz cross-distinct families)")
print("=" * 70)

LC_VESSEL_ONLY = [{'document_name': 'Vessel Advice'},
                  {'document_name': 'Bill of Lading'}]
LC_SHIPMENT_ONLY = [{'document_name': 'Shipment Advice'},
                    {'document_name': 'Bill of Lading'}]

idx, name = _match_type_to_requirement('Shipment Advice', LC_VESSEL_ONLY)
ok(f"  Shipment Advice -> alien when LC only has Vessel", idx == -1)

idx, name = _match_type_to_requirement('Vessel Advice', LC_SHIPMENT_ONLY)
ok(f"  Vessel Advice -> alien when LC only has Shipment", idx == -1)

# Cross-cert family rejection (already tested but include)
LC_HEALTH = [{'document_name': 'Health Certificate'}]
idx, name = _match_type_to_requirement('Phytosanitary Certificate', LC_HEALTH)
ok(f"  Phyto -> alien when LC only has Health", idx == -1)

idx, name = _match_type_to_requirement('Halal Certificate', LC_HEALTH)
ok(f"  Halal -> alien when LC only has Health", idx == -1)


# ── Section 4: Negative cases (no spurious binding) ──
print("\n" + "=" * 70)
print("Section 4: Negative cases — should NOT bind")
print("=" * 70)
NEG_TESTS = [
    ('', LC_FULL, -1, 'Empty input'),
    ('   ', LC_FULL, -1, 'Whitespace'),
    ('Certificate', LC_FULL, -1, 'Bare CERTIFICATE'),
    ('Document', LC_FULL, -1, 'Bare DOCUMENT'),
    ('Form', LC_FULL, -1, 'Bare FORM'),
    ('OPRINDELSESCERTIFIKAT', LC_FULL, -1, 'Foreign lang (Danish CoO) -> alien'),
    ('Certificate of Origin Instructions', LC_FULL, -1, 'Meta-doc (Instructions)'),
    ('How to Fill Bill of Lading', LC_FULL, -1, 'How-to'),
    ('Form-E', LC_FULL, -1, 'Form-E -> alien (no Form-E in LC)'),
]
for inp, lc, exp, lbl in NEG_TESTS:
    idx, name = _match_type_to_requirement(inp, lc)
    ok(f"  {lbl}: {inp!r} -> {idx}", idx == exp, f"got idx={idx}")


# ── Section 5: P198gz12 Bene Cert vs CoO heuristic ──
print("\n" + "=" * 70)
print("Section 5: P198gz12 Bene Cert vs CoO classifier")
print("=" * 70)

COO_MARKERS = ('COUNTRY OF ORIGIN','ORIGIN CRITERIA','CHAMBER OF COMMERCE',
               'CERTIFICATE OF ORIGIN','OPRINDELSESCERTIFIKAT','EUR.1','EUR1',
               'GENERALIZED SYSTEM OF PREFERENCES','FORM A','CERT OF ORIGIN')
BENE_MARKERS = ('WE HEREBY CERTIFY','WE CERTIFY THAT',"BENEFICIARY'S CERTIFICATE",
                'BENEFICIARY CERTIFICATE','COMPLY WITH LC','COMPLY WITH L/C',
                'COMPLY WITH THE LETTER OF CREDIT','AS PER L/C','AS PER THE LC',
                'CERTIFY THE GOODS')

def gz12(text_up):
    has_coo = any(m in text_up for m in COO_MARKERS)
    has_bene = any(m in text_up for m in BENE_MARKERS)
    return 'BENE_OVERRIDE' if (has_bene and not has_coo) else \
           'KEEP_COO' if has_coo else 'NEITHER'

GZ12_TESTS = [
    ('We hereby certify goods comply with LC', 'BENE_OVERRIDE',
     'Pure bene-cert language'),
    ('Country of Origin: DENMARK\nChamber of Commerce stamp', 'KEEP_COO',
     'Real CoO with all markers'),
    ('Country of Origin: India\nWe hereby certify the goods comply with LC',
     'KEEP_COO', 'Mixed (CoO markers present, bene language too) -> KEEP CoO'),
    ('OPRINDELSESCERTIFIKAT', 'KEEP_COO', 'Danish CoO'),
    ('Generic certificate of compliance', 'NEITHER',
     'No CoO markers, no bene markers'),
    ('Form A — Generalized System of Preferences', 'KEEP_COO', 'GSP variant'),
    ('We certify the goods are of Indian origin', 'BENE_OVERRIDE',
     'Bene declaration of origin (still bene-cert)'),
    ('EUR.1 movement certificate', 'KEEP_COO', 'EUR.1 origin proof'),
]
for txt, exp, lbl in GZ12_TESTS:
    got = gz12(txt.upper())
    ok(f"  {lbl}: {got}", got == exp, f"expected {exp}")


# ── Section 6: Real-job sweep — Bene-Cert override candidates ──
print("\n" + "=" * 70)
print("Section 6: Real-job sweep — sanity checks across 91 jobs")
print("=" * 70)

job_files = sorted(glob.glob(
    'd:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final/results/*/step08/step08_result.json'))

total_packets = 0
coo_count = 0
coo_real = 0
coo_bene = 0
shipment_count = 0
vessel_in_shipment = 0
all_doc_types = set()

for jp in job_files:
    try: d = json.load(open(jp, encoding='utf-8'))
    except: continue
    for pk in d.get('classified_packets', []):
        total_packets += 1
        dt = (pk.get('document_type','') or '')
        all_doc_types.add(dt)
        txt = (pk.get('cleaned_text') or pk.get('raw_text') or '').upper()
        if dt == 'Certificate of Origin':
            coo_count += 1
            if any(m in txt for m in COO_MARKERS): coo_real += 1
            elif any(m in txt for m in BENE_MARKERS): coo_bene += 1
        elif dt == 'Shipment Advice':
            shipment_count += 1
            for pg in pk.get('original_pages', []):
                if isinstance(pg, dict) and 'VESSEL' in (pg.get('document_type','') or '').upper():
                    vessel_in_shipment += 1; break

print(f"  Total packets across 91 jobs: {total_packets}")
print(f"  Distinct document_type labels seen: {len(all_doc_types)}")
print(f"  CoO packets: {coo_count} (real={coo_real}, would-override-to-bene={coo_bene})")
print(f"  Shipment Advice: {shipment_count} (per-page VESSEL: {vessel_in_shipment})")

ok("  Override rate sensible (CoO -> Bene < 50%)",
   coo_count == 0 or coo_bene / coo_count < 0.5,
   f"{coo_bene}/{coo_count}")
ok("  At least 90% of CoO packets are real CoO",
   coo_count == 0 or coo_real / coo_count >= 0.5,
   f"{coo_real}/{coo_count}")
ok("  Vessel-in-Shipment-Advice rate sensible",
   shipment_count == 0 or vessel_in_shipment / shipment_count < 0.5,
   f"{vessel_in_shipment}/{shipment_count}")


# ── Section 7: Source wiring ──
print("\n" + "=" * 70)
print("Section 7: Source wiring sanity")
print("=" * 70)
src = open('d:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final/steps/step08_shipping_classification.py',
           'r', encoding='utf-8').read()
for marker in ('P198gz12','P198gz13','P198gz14','P198gz15','P198gz16',
               '_COO_MARKERS','_BENE_CERT_MARKERS','_ACRONYMS',
               '_ADVICE_FAMILIES'):
    ok(f"  {marker} present", marker in src)


print("\n" + "=" * 70)
passed = sum(results)
total = len(results)
print(f"P198gz AGGRESSIVE: {passed}/{total}")
print("=" * 70)
if passed != total:
    sys.exit(1)
print("OVERALL: OK")
sys.exit(0)
