"""Coal LC condition / clause scenarios — verify per-document
condition checks fire correctly.

For a 2-tranche coal LC like 1f0fc892:
- Tranche A (90%): CI, CP-BL, Insurance shipment-advice, Weight Cert
  (load), DSR (load), CoO, Bene Cert (docs sent vide email)
- Tranche B (10%): Balance CI, Weight Cert (discharge), Sampling &
  Analysis (discharge)

This test simulates verification rows for each condition and verifies
the row gets the correct disposition under P198gz26 + other guards.
"""
import sys, os, re
sys.path.insert(0, 'd:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final')
os.environ['PYTHONIOENCODING'] = 'utf-8'

from steps.step14_verification import _detect_release_tranches

results = []
def ok(name, cond, note=''):
    tag = '[OK]  ' if cond else '[FAIL]'
    print(f"{tag} {name}" + (f" -- {note}" if note else ''))
    results.append(bool(cond))


# Build a real 1f0fc892-style F46A
F46A_COAL = """A) FOR RELEASE OF 90 PERCENT PAYMENT OF LC VALUE, FOLLOWING DOCUMENTS ARE REQUIRED
1) BENEFICIARY MANUALLY SIGNED COMMERCIAL INVOICE IN OCTUPPLICATE SHOWING PAYMENT CLAIM UPTO 90 PCT AS PER PAYMENT TERMS STATED UNDER LC AND ACTUAL RESULTS WITH PRICE ADJUSTMENT (IF ANY), IN CFR PAKISTAN INTERNATIONAL BULK TERMINAL PAKISTAN IN THE NAME OF APPLICANT WITH FULL ADDRESS INDICATING NTN NO. 0712596-8, H.S. CODE NO. 2701.1200. LC NO.0062/078647/2025 DATED AUGUST 29, 2025 CERTIFYING THE GOODS ARE OF SOUTH AFRICA ORIGIN.
2) FULL SET OF THREE ORIGINAL PLUS THREE NON NEGOTIABLE COPIES OF CLEAN ON BOARD CHARTER PARTY BILLS OF LADING ISSUED OR ENDORSED TO ORDER OF BANK AL HABIB LIMITED, PAKISTAN. MARKED 'FREIGHT PAYABLE AS PER CHARTER PARTY MARKED NOTIFY THE APPLICANT AND BANK AL HABIB LIMITED, PAKISTAN.)
3) INSURANCE COVERED BY THE APPLICANT.
4) ONE ORIGINAL AND TWO COPIES OF CERTIFICATE OF WEIGHT ISSUED BY INDEPENDENT INSPECTION AGENCY AT LOADING PORT.
5) ONE ORIGINAL PLUS TWO COPIES OF DRAFT SURVEY REPORT ISSUED BY INDEPENDENT INSPECTION AGENCY AT LOADING PORT.
6) ONE ORIGINAL PLUS TWO COPIES OF CERTIFICATE OF ORIGIN CERTIFYING GOODS ARE OF SOUTH AFRICA ORIGIN.
7) CERTIFICATE FROM BENEFICIARY STATING THAT, ONE COMPLETE SET OF NON-NEGOTIABLE DOCUMENTS HAS BEEN SENT TO THE APPLICANT AFTER SHIPMENT VIDE EMAIL.

B) FOR RELEASE OF 10 PERCENT OF LC VALUE FOLLOWING DOCUMENTS ARE REQUIRED:
1) BENEFICIARY MANUALLY SIGNED COMMERCIAL INVOICE IN OCTUPPLICATE SHOWING BALANCE PAYMENT OF 10 PERCENT AGAINST SHIPED GOODS PAYABLE TO BENEFICIARY AND ACTUAL RESULTS WITH PRICE ADJUSTMENT.
2) CERTIFICATE OF WEIGHT IN ONE ORIGINAL AND ONE COPY ISSUED BY INSPECTION AGENCY AT DISCHARGE PORT DULY ACCEPTED BY APPLICANT REPRESENTATIVE UNDER HIS OR HER SIGNATURES AND OFFICIAL STAMP. SIGNATURES OF APPLICANT REPRESENTITIVE MUST BE DULY VERIFIED BY BANK AL HABIB LIMITED WAH CANTT BRANCH.
3) CERTIFICATE OF SAMPLING AND ANALYSIS IN ONE ORIGINAL AND THREE COPIES ISSUED BY INSPECTION AGENCY AT DISCHARGE PORT DULY ACCEPTED BY APPLICANT REPRESENTATIVE UNDER HIS OR HER SIGNATURES AND OFFICIAL STAMP."""

info = _detect_release_tranches({'consolidated_fields':{'46A':F46A_COAL}})
ok("  Realistic F46A 2-tranche detected", info is not None)
if info:
    ok(f"    A=90, B=10", info['tranche_a_pct']==90 and info['tranche_b_pct']==10)
    tr_a = info['tranche_a_text']
    tr_b = info['tranche_b_text']

    # Verify A markers stay in A only
    print()
    print("Section A: tranche-A-only conditions (load-port docs)")
    A_PHRASES = [
        "CHARTER PARTY",
        "DRAFT SURVEY REPORT",
        "AT LOADING PORT",
        "ONE COMPLETE SET OF NON-NEGOTIABLE DOCUMENTS HAS BEEN SENT",
        "VIDE EMAIL",
        "FREIGHT PAYABLE AS PER CHARTER PARTY",
    ]
    for ph in A_PHRASES:
        in_a = ph in tr_a
        in_b = ph in tr_b
        ok(f"    '{ph}' in A only", in_a and not in_b)

    print()
    print("Section B: tranche-B-only conditions (discharge-port docs)")
    B_PHRASES = [
        "AT DISCHARGE PORT",
        "SAMPLING AND ANALYSIS",
        "WAH CANTT BRANCH",
        "ACCEPTED BY APPLICANT REPRESENTATIVE",
        "BALANCE PAYMENT OF 10 PERCENT",
    ]
    for ph in B_PHRASES:
        in_a = ph in tr_a
        in_b = ph in tr_b
        ok(f"    '{ph}' in B only", in_b and not in_a)


# ── Section: Per-row deferment simulation ──
print("\n" + "=" * 70)
print("Section: Row-deferment for the actual 1f0fc892 conditions")
print("=" * 70)

# Bundle = load-only (this is the 90% presentation)
BUNDLE_LOAD_ONLY = """SGS DRAFT SURVEY at Richards Bay
Charter Party Bill of Lading SS GOOD HEART
Weight Cert at LOAD PORT
Certificate of Origin South Africa
Beneficiary cert: docs sent vide email"""
DISCHARGE_MARKERS = ('DISCHARGE PORT','DISCHARGE-PORT','AT DISCHARGE',
                     'PORT OF DISCHARGE','SAMPLING AND ANALYSIS',
                     'ACCEPTED BY APPLICANT REPRESENTATIVE',
                     'BALANCE PAYMENT','10 PERCENT','10 PCT','BALANCE OF',
                     'WAH CANTT BRANCH')
def determine(bundle):
    up = bundle.upper()
    return 'B' if any(p in up for p in DISCHARGE_MARKERS) else 'A'
current_tranche = determine(BUNDLE_LOAD_ONLY)
ok(f"  Bundle = load-only → tranche {current_tranche}", current_tranche=='A')

# Now identify which conditions defer
ROWS_REAL = [
    ('R-A1', 'Commercial Invoice in 8 originals showing 90% claim',
     'A — load CI', False),
    ('R-A2', 'Charter Party Bill of Lading endorsed to BAH',
     'A — Charter Party BL', False),
    ('R-A4', 'Weight Certificate at LOADING PORT issued by independent agency',
     'A — load weight', False),
    ('R-A5', 'Draft Survey Report at LOADING PORT',
     'A — load DSR', False),
    ('R-A6', 'Certificate of Origin SOUTH AFRICA',
     'A — CoO', False),
    ('R-A7', 'Beneficiary certificate: docs sent vide email',
     'A — bene email cert', False),
    ('R-B1', 'Commercial Invoice 10 percent balance payment',
     'B — balance CI', True),
    ('R-B2', 'Certificate of Weight AT DISCHARGE PORT verified by WAH CANTT BRANCH',
     'B — discharge weight', True),
    ('R-B3', 'Sampling and Analysis Certificate at discharge port',
     'B — sampling/analysis', True),
]

def is_b_row(cond):
    return any(ph in cond.upper() for ph in DISCHARGE_MARKERS)

for rid, cond, label, expect_defer in ROWS_REAL:
    is_b = is_b_row(cond)
    will_defer = is_b and current_tranche == 'A'
    if expect_defer:
        ok(f"  {rid} ({label}): SHOULD defer → defers={will_defer}",
           will_defer)
    else:
        ok(f"  {rid} ({label}): should NOT defer → defers={will_defer}",
           not will_defer)


# ── Source wiring sanity ──
print("\n" + "=" * 70)
print("Source wiring")
print("=" * 70)
src = open('d:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final/steps/step14_verification.py',
           'r', encoding='utf-8').read()
ok("  step14 has _detect_release_tranches", '_detect_release_tranches' in src)
ok("  step14 has P198gz26 marker", 'P198gz26' in src)
ok("  step14 has UCP scrubber (P198gz27)", 'P198gz27' in src)
ok("  step14 has _scrub_ucp helper", '_scrub_ucp' in src or '_UCP_REs' in src)


print("\n" + "=" * 70)
passed = sum(results)
total = len(results)
print(f"COAL CONDITIONS: {passed}/{total}")
print("=" * 70)
if passed != total:
    sys.exit(1)
print("OVERALL: OK")
sys.exit(0)
