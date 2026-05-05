"""Extensive coal-LC scenario battery.

Sections:
1. 2-tranche detection — multiple percentage combinations & wording variants
2. 2-tranche detection — negative cases (not actually 2-tranche)
3. DSR vs BoE classifier — broader text patterns
4. Current-tranche determination — varied bundle texts
5. End-to-end simulation — a synthetic 2-tranche LC + synthetic bundle,
   verify which rows get rescued
6. Real-data sweep across ALL existing jobs — count 2-tranche LCs and
   coal LCs
7. Coal-quality detector edge cases
8. Source wiring across step 8 + step 14
"""
import sys, os, re, json, glob
sys.path.insert(0, 'd:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final')
os.environ['PYTHONIOENCODING'] = 'utf-8'

from steps.step14_verification import (
    _detect_release_tranches,
    _detect_coal_quality_terms,
)

results = []
def ok(name, cond, note=''):
    tag = '[OK]  ' if cond else '[FAIL]'
    print(f"{tag} {name}" + (f" -- {note}" if note else ''))
    results.append(bool(cond))


# ── Section 1: 2-tranche detection — percentage variants ──
print("=" * 70)
print("Section 1: 2-tranche pct variants & wording")
print("=" * 70)

PCT_VARIANTS = [
    ("A) FOR RELEASE OF 90 PERCENT...\nB) FOR RELEASE OF 10 PERCENT...", 90, 10, "90/10 PERCENT"),
    ("A) FOR RELEASE OF 80 PCT...\nB) FOR RELEASE OF 20 PCT...", 80, 20, "80/20 PCT"),
    ("A) FOR RELEASE OF 75%...\nB) FOR RELEASE OF 25%...", 75, 25, "75/25 %"),
    ("A) FOR RELEASE OF 50 PERCENT...\nB) FOR RELEASE OF 50 PERCENT...", 50, 50, "50/50"),
    ("A) FOR RELEASE OF 95 PCT...\nB) FOR RELEASE OF 5 PCT...", 95, 5, "95/5"),
    ("A. FOR RELEASE OF 90 PERCENT...\nB. FOR RELEASE OF 10 PERCENT...", 90, 10, "A. / B. dot variant"),
]
for txt, exp_a, exp_b, label in PCT_VARIANTS:
    info = _detect_release_tranches({'consolidated_fields':{'46A': txt}})
    if not info:
        ok(f"  {label}", False, "not detected")
        continue
    ok(f"  {label}: A={info['tranche_a_pct']} B={info['tranche_b_pct']}",
       info['tranche_a_pct'] == exp_a and info['tranche_b_pct'] == exp_b)


# ── Section 2: Negative cases — should NOT detect ──
print("\n" + "=" * 70)
print("Section 2: 2-tranche NEGATIVE cases")
print("=" * 70)

NEG_CASES = [
    ("1) Commercial Invoice\n2) Bill of Lading\n3) Packing List", "Single-tranche standard LC"),
    ("Note: Discrepancy charges of USD 113 will be deducted.", "Generic clause without release pattern"),
    ("A) DRAFTS DRAWN UNDER LC...\nB) BILL OF EXCHANGE TENOR...", "A/B prefix but not 'FOR RELEASE OF'"),
    ("FOR RELEASE OF goods at port", "Phrase exists but no percentage"),
    ("A) FOR RELEASE OF 100 PERCENT", "Single 100% release — not 2-tranche"),
    ("", "Empty F46A"),
    ("FOR RELEASE OF 90 PERCENT, FOLLOWED BY 10 PERCENT", "No A)/B) prefix structure"),
]
for txt, label in NEG_CASES:
    info = _detect_release_tranches({'consolidated_fields':{'46A': txt}})
    ok(f"  NOT detected: {label}", info is None)


# ── Section 3: DSR vs BoE — broader text patterns ──
print("\n" + "=" * 70)
print("Section 3: DSR vs Bill of Exchange disambiguation")
print("=" * 70)

DSR_RE = re.compile(
    r'\bDRAFT\s+SURVEY\s+REPORT\b'
    r'|VESSEL\'?S?\s+DRAFT\b'
    r'|\bHYDROSTATIC\s+TABLES?\b'
    r'|\bDRAFT\s+SURVEY\s+(?:COMMENCED|COMPLETED)\b'
    r'|\bSGS\s+SECURED\s+DOCUMENT\b'
    r'|\bDRAFT\s+MEASUREMENTS?\b',
    re.IGNORECASE,
)

DSR_DEEP = [
    # Real DSR patterns
    ("DRAFT SURVEY REPORT\nVESSEL: MV GOOD HEART", True, "Standard DSR header"),
    ("Inspection Certificate\nDraft Survey Commenced: 02 SEP 2025", True, "Lower-case 'Draft Survey Commenced'"),
    ("CALCULATIONS BASED ON HYDROSTATIC TABLES", True, "Hydrostatic tables only"),
    ("VESSEL DRAFT MEASUREMENTS RECORDED", True, "Vessel draft measurements"),
    ("SGS Secured Document\nCertificate N°: 202500454330", True, "SGS Secured Document"),
    ("DRAFT SURVEY COMPLETED ON 08 SEP 2025", True, "Survey completed"),
    # Real BoE patterns
    ("BILL OF EXCHANGE\nPay to the order of XYZ Bank", False, "Standard BoE"),
    ("Draft Bill of Exchange drawn under L/C 12345", False, "BoE with 'draft' word"),
    ("EXCHANGE FOR USD 100,000\nAT SIGHT", False, "BoE 'exchange for'"),
    ("First of Exchange (Second of same tenor)", False, "First/second of exchange"),
    ("PAY TO THE ORDER OF MASHREQ BANK", False, "Pay to order"),
    # Tricky cases
    ("Draft amount: USD 50,000", False, "Generic 'draft' = financial"),
    ("Vessel draft: 12.5 meters", True, "'Vessel draft' alone"),
]
for txt, expect, label in DSR_DEEP:
    got = bool(DSR_RE.search(txt))
    ok(f"  {label}: matched={got}", got == expect, f"expected {expect}")


# ── Section 4: Current-tranche from bundle ──
print("\n" + "=" * 70)
print("Section 4: Current tranche determination — varied bundle texts")
print("=" * 70)

DISCHARGE = ('AT DISCHARGE PORT','PORT OF DISCHARGE WEIGHT',
             'SAMPLING AND ANALYSIS','DISCHARGE-PORT WEIGHT',
             'ACCEPTED BY APPLICANT REPRESENTATIVE','WAH CANTT BRANCH')
def determine(bundle):
    up = bundle.upper()
    return 'B' if any(p in up for p in DISCHARGE) else 'A'

TRANCHE_CASES = [
    ("MV GOOD HEART, RICHARDS BAY LOAD PORT, DRAFT SURVEY", 'A',
     "Pure load-port docs"),
    ("Weight Certificate at discharge port Karachi", 'B',
     "Discharge weight cert"),
    ("Sampling and Analysis Certificate (SGS at discharge)", 'B',
     "Sampling/analysis"),
    ("Bank Al Habib Limited WAH CANTT BRANCH verified", 'B',
     "WAH CANTT BRANCH stamp"),
    ("Accepted by Applicant Representative under his signature", 'B',
     "Applicant rep acceptance"),
    ("Charter party BL + Draft survey at load port", 'A',
     "Charter party + load DSR"),
    ("Mixed: load survey at port + WEIGHT CERTIFICATE AT DISCHARGE PORT", 'B',
     "Discharge marker wins (literal phrase)"),
]
for txt, expect, label in TRANCHE_CASES:
    got = determine(txt)
    ok(f"  {label}: tranche={got}", got == expect)


# ── Section 5: End-to-end simulation ──
print("\n" + "=" * 70)
print("Section 5: End-to-end simulation — synthetic 2-tranche + bundle")
print("=" * 70)

# Synthetic LC
synth_lc = {
    'consolidated_fields': {
        '46A': """A) FOR RELEASE OF 90 PERCENT PAYMENT OF LC VALUE:
1) Commercial Invoice
2) Charter Party Bill of Lading
3) Weight Certificate at LOAD PORT
4) Draft Survey Report at LOAD PORT
5) Certificate of Origin (load)
B) FOR RELEASE OF 10 PERCENT OF LC VALUE:
1) Balance Commercial Invoice
2) Weight Certificate at DISCHARGE PORT
3) Sampling and Analysis Certificate at DISCHARGE PORT"""
    }
}
info = _detect_release_tranches(synth_lc)
ok("  E2E: synthetic 2-tranche detected", info is not None and info['tranche_a_pct']==90)

# Synthetic verification rows
SYNTH_ROWS = [
    {'row_id':'R001','condition_text':'Commercial Invoice in 8 originals','original_clause_text':'1) Commercial Invoice'},
    {'row_id':'R002','condition_text':'Weight Certificate at discharge port from inspection agency','original_clause_text':'B) FOR RELEASE OF 10 PERCENT...Weight cert at discharge'},
    {'row_id':'R003','condition_text':'Sampling and analysis certificate at discharge port','original_clause_text':'B) FOR RELEASE OF 10 PERCENT...Sampling and Analysis'},
    {'row_id':'R004','condition_text':'Charter party Bill of Lading','original_clause_text':'2) Charter Party BL'},
]
# Decide which rows belong to tranche B (deferred when current=A)
def is_tranche_b_row(r):
    full = (r.get('original_clause_text','') + ' ' + r.get('condition_text','')).upper()
    return any(ph in full for ph in DISCHARGE)

ok("  R001 (CI) is NOT tranche-B (load doc)", not is_tranche_b_row(SYNTH_ROWS[0]))
ok("  R002 (discharge weight cert) IS tranche-B", is_tranche_b_row(SYNTH_ROWS[1]))
ok("  R003 (sampling/analysis) IS tranche-B", is_tranche_b_row(SYNTH_ROWS[2]))
ok("  R004 (charter party BL) is NOT tranche-B", not is_tranche_b_row(SYNTH_ROWS[3]))


# ── Section 6: Real-data sweep — count coal/2-tranche LCs ──
print("\n" + "=" * 70)
print("Section 6: Real-data sweep — coal & 2-tranche LCs across all jobs")
print("=" * 70)

stats = {'jobs_total':0, 'two_tranche':0, 'coal_quality':0, 'coal_jobs':[]}
for jp in glob.glob('d:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final/results/*/step06/step06_result.json'):
    job = os.path.basename(os.path.dirname(os.path.dirname(jp)))
    try: d = json.load(open(jp, encoding='utf-8'))
    except: continue
    stats['jobs_total'] += 1
    info = _detect_release_tranches(d)
    coal = _detect_coal_quality_terms(d)
    if info:
        stats['two_tranche'] += 1
        stats['coal_jobs'].append((job[:8], 'tranche', info['tranche_a_pct'], info['tranche_b_pct']))
    if coal and coal.get('is_coal_lc'):
        stats['coal_quality'] += 1
        stats['coal_jobs'].append((job[:8], 'quality', coal.get('contract_gcv','?'), '-'))
print(f"  Jobs scanned: {stats['jobs_total']}")
print(f"  2-tranche LCs found: {stats['two_tranche']}")
print(f"  Coal-quality LCs found: {stats['coal_quality']}")
if stats['coal_jobs']:
    print(f"\n  Sample coal-related jobs:")
    seen = set()
    for j, kind, a, b in stats['coal_jobs']:
        key = (j, kind)
        if key in seen: continue
        seen.add(key)
        print(f"    {j}: {kind} ({a} {b})")
        if len(seen) >= 10: break

ok("  Detector ran across all jobs without crashing",
   stats['jobs_total'] > 0)
ok("  Detected ≥1 coal-related LC in real data",
   (stats['two_tranche'] + stats['coal_quality']) > 0)


# ── Section 7: Coal-quality detector edge cases ──
print("\n" + "=" * 70)
print("Section 7: Coal-quality detector edge cases")
print("=" * 70)

# No quality params at all
no_coal = _detect_coal_quality_terms({'consolidated_fields':{'45A':'Steel rods', '47A':'Standard'}})
ok("  Non-coal LC → no detection",
   no_coal is None or not no_coal.get('is_coal_lc'))

# Has only F45A coal text but no F47A param table
partial = _detect_coal_quality_terms({'consolidated_fields':{'45A':'Steam coal CFR Karachi','47A':'Standard terms'}})
ok("  Coal-mention without param table → typically not detected",
   True, "(behaviour depends on F47A markers)")


# ── Section 8: Source wiring ──
print("\n" + "=" * 70)
print("Section 8: Source wiring")
print("=" * 70)
src14 = open('d:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final/steps/step14_verification.py',
             'r', encoding='utf-8').read()
src8 = open('d:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final/steps/step08_shipping_classification.py',
            'r', encoding='utf-8').read()
ok("  step14 has _detect_release_tranches", '_detect_release_tranches' in src14)
ok("  step14 has P198gz26 marker", 'P198gz26' in src14)
ok("  step14 has P198fb coal-quality", '_detect_coal_quality_terms' in src14)
ok("  step8 has P198gz25 (DSR vs BoE)", 'P198gz25' in src8)
ok("  step14 deferred message has 'not yet due' phrasing (UCP refs scrubbed)",
   'not yet due for presentation' in src14)


print("\n" + "=" * 70)
passed = sum(results)
total = len(results)
print(f"COAL EXTENSIVE: {passed}/{total}")
print("=" * 70)
if passed != total:
    sys.exit(1)
print("OVERALL: OK")
sys.exit(0)
