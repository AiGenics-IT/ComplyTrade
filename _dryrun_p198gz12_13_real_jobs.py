"""P198gz12 + gz13 — test against real job data.

P198gz12: Beneficiary Cert force-fit to CoO. Identifies packets
currently labelled CoO that are actually Bene Certs (have bene-cert
language but lack CoO markers).

P198gz13: Vessel Advice ↔ Shipment Advice family disambiguation.
Tests the matcher against both labels.
"""
import sys, os, json, glob, re
sys.path.insert(0, 'd:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final')
os.environ['PYTHONIOENCODING'] = 'utf-8'

from steps.step08_shipping_classification import _match_type_to_requirement

results = []
def ok(name, cond, note=''):
    tag = '[OK]  ' if cond else '[FAIL]'
    print(f"{tag} {name}" + (f" -- {note}" if note else ''))
    results.append(bool(cond))


# ── P198gz12 helpers ──
COO_MARKERS = ('COUNTRY OF ORIGIN','ORIGIN CRITERIA','CHAMBER OF COMMERCE',
               'CERTIFICATE OF ORIGIN','OPRINDELSESCERTIFIKAT','EUR.1','EUR1',
               'GENERALIZED SYSTEM OF PREFERENCES','FORM A','CERT OF ORIGIN')
BENE_MARKERS = ('WE HEREBY CERTIFY','WE CERTIFY THAT',"BENEFICIARY'S CERTIFICATE",
                'BENEFICIARY CERTIFICATE','COMPLY WITH LC','COMPLY WITH L/C',
                'COMPLY WITH THE LETTER OF CREDIT','AS PER L/C','AS PER THE LC',
                'CERTIFY THE GOODS')

def gz12_should_override(text_up):
    """Return True if VLM-CoO label should be overridden to Bene Cert."""
    has_coo = any(m in text_up for m in COO_MARKERS)
    has_bene = any(m in text_up for m in BENE_MARKERS)
    return has_bene and not has_coo


# ── Section 1 — sweep all jobs for CoO packets ──
print("=" * 70)
print("Section 1: All packets labelled 'Certificate of Origin' across jobs")
print("=" * 70)

total_coo = 0
real_coo = 0
override_to_bene = 0
suspicious_no_markers = 0
samples = []

for jp in sorted(glob.glob('d:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final/results/*/step08/step08_result.json')):
    job = os.path.basename(os.path.dirname(os.path.dirname(jp)))
    try: d = json.load(open(jp,encoding='utf-8'))
    except: continue
    for pk in d.get('classified_packets', []):
        if (pk.get('document_type','') or '') != 'Certificate of Origin':
            continue
        total_coo += 1
        txt = (pk.get('cleaned_text') or pk.get('raw_text') or '').upper()
        has_coo = any(m in txt for m in COO_MARKERS)
        has_bene = any(m in txt for m in BENE_MARKERS)
        if has_coo:
            real_coo += 1
        elif has_bene:
            override_to_bene += 1
            if len(samples) < 5:
                pages = [p.get('page_number') for p in pk.get('original_pages',[]) if isinstance(p,dict)]
                samples.append((job[:8], pages, has_coo, has_bene))
        else:
            suspicious_no_markers += 1

print(f"  Total CoO packets across all jobs:        {total_coo}")
print(f"  Real CoO (have CoO markers):              {real_coo}")
print(f"  Should override to Bene Cert (P198gz12):  {override_to_bene}")
print(f"  Neither markers (alien-ish):              {suspicious_no_markers}")
print()
print("Sample packets that P198gz12 will override:")
for j,p,c,b in samples:
    print(f"  {j}: pages={p} coo_markers={c} bene_markers={b}")

ok("  Override rate is sensible (<50% of CoO packets)",
   override_to_bene < (total_coo * 0.5),
   f"override={override_to_bene}/{total_coo}")
ok("  At least some packets correctly identified as real CoO",
   real_coo > 0,
   f"real_coo={real_coo}")


# ── Section 2 — Vessel Advice vs Shipment Advice matcher ──
print("\n" + "=" * 70)
print("Section 2: P198gz13 — Vessel Advice vs Shipment Advice matcher")
print("=" * 70)

LC_HAS_BOTH = [
    {'document_name': 'Shipment Advice'},
    {'document_name': 'Vessel Advice'},
    {'document_name': 'Bill of Lading'},
]
LC_HAS_SHIPMENT_ONLY = [
    {'document_name': 'Shipment Advice'},
    {'document_name': 'Bill of Lading'},
]
LC_HAS_VESSEL_ONLY = [
    {'document_name': 'Vessel Advice'},
    {'document_name': 'Bill of Lading'},
]

# When LC has both, each maps to its own slot
idx, name = _match_type_to_requirement('Vessel Advice', LC_HAS_BOTH)
ok(f"  LC has both: 'Vessel Advice' -> {name!r}",
   name == 'Vessel Advice', f'got idx={idx}')
idx, name = _match_type_to_requirement('Shipment Advice', LC_HAS_BOTH)
ok(f"  LC has both: 'Shipment Advice' -> {name!r}",
   name == 'Shipment Advice', f'got idx={idx}')

# When LC has only Shipment Advice and packet is Vessel Advice → alien (don't fuzz to Shipment)
idx, name = _match_type_to_requirement('Vessel Advice', LC_HAS_SHIPMENT_ONLY)
ok(f"  LC has only Shipment: 'Vessel Advice' -> {name!r} (should be alien)",
   idx == -1, f'got idx={idx} name={name}')
idx, name = _match_type_to_requirement('VESSEL ADVICE', LC_HAS_SHIPMENT_ONLY)
ok(f"  LC has only Shipment: 'VESSEL ADVICE' upper -> {name!r} (alien)",
   idx == -1, f'got idx={idx} name={name}')

# When LC has only Vessel Advice and packet is Shipment Advice → alien
idx, name = _match_type_to_requirement('Shipment Advice', LC_HAS_VESSEL_ONLY)
ok(f"  LC has only Vessel: 'Shipment Advice' -> alien",
   idx == -1, f'got idx={idx} name={name}')

# Sanity: BL still matches BL (not affected)
idx, name = _match_type_to_requirement('Bill of Lading', LC_HAS_BOTH)
ok(f"  Bill of Lading still binds correctly",
   name == 'Bill of Lading', f'got idx={idx}')


# ── Section 3 — sweep all jobs for Vessel Advice / Shipment Advice ──
print("\n" + "=" * 70)
print("Section 3: All Shipment Advice packets — check if any are actually Vessel Advice")
print("=" * 70)

shipment_advice_packets = 0
likely_vessel_advice = 0  # has VESSEL or VOYAGE markers but labelled Shipment Advice
real_shipment = 0

for jp in sorted(glob.glob('d:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final/results/*/step08/step08_result.json')):
    job = os.path.basename(os.path.dirname(os.path.dirname(jp)))
    try: d = json.load(open(jp,encoding='utf-8'))
    except: continue
    for pk in d.get('classified_packets', []):
        dt = (pk.get('document_type','') or '').lower()
        if dt != 'shipment advice':
            continue
        shipment_advice_packets += 1
        # Check per-page step3 labels for VESSEL
        per_page_types = []
        for pg in pk.get('original_pages', []):
            if isinstance(pg, dict):
                per_page_types.append((pg.get('document_type','') or '').upper())
        if any('VESSEL' in t for t in per_page_types):
            likely_vessel_advice += 1
        else:
            real_shipment += 1

print(f"  Total 'Shipment Advice' packets:                  {shipment_advice_packets}")
print(f"  Likely should be 'Vessel Advice' (per-page says): {likely_vessel_advice}")
print(f"  Real 'Shipment Advice':                           {real_shipment}")


# ── Section 4 — Source wiring ──
print("\n" + "=" * 70)
print("Section 4: Source wiring")
print("=" * 70)
src = open('d:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final/steps/step08_shipping_classification.py',
           'r', encoding='utf-8').read()
ok("  P198gz12 marker (Bene Cert override)", 'P198gz12' in src)
ok("  P198gz13 marker (Vessel/Shipment family)", 'P198gz13' in src)
ok("  _COO_MARKERS in code", '_COO_MARKERS' in src)
ok("  _BENE_CERT_MARKERS in code", '_BENE_CERT_MARKERS' in src)
ok("  VESSEL ADVICE -> Vessel Advice canonical", "'VESSEL ADVICE': 'Vessel Advice'" in src)
ok("  _ADVICE_FAMILIES in matcher", '_ADVICE_FAMILIES' in src)


print("\n" + "=" * 70)
passed = sum(results)
total = len(results)
print(f"P198gz12+13 REAL JOBS: {passed}/{total}")
print("=" * 70)
if passed != total:
    sys.exit(1)
print("OVERALL: OK")
sys.exit(0)
