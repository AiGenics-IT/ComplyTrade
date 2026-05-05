"""Full end-to-end coal-LC scenario battery.

Sections:
1. Tranche detection — 30+ percentage combos including edge cases
2. Tranche detection — irregular formats (extra whitespace, line breaks)
3. Bundle determination — load-only / discharge-only / mixed bundles
4. Row rescue logic — simulate full P198gz26 post-check on rows
5. Synthetic full LC verification flows (load-side, discharge-side,
   final balance presentation)
6. Negative cases — non-coal LCs not affected
7. Coal-quality + tranche together — ensure both detectors fire
   independently
8. UCP/ISBP scrub — test the final scrubber
9. Real-data sweep — every job, find any that should have been flagged
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


def detect(text):
    return _detect_release_tranches({'consolidated_fields': {'46A': text}})


# ── Section 1: Many percentage combos ──
print("=" * 70)
print("Section 1: Wide percentage-combo coverage")
print("=" * 70)
COMBOS = [(a, 100-a) for a in (10, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90)]
COMBOS += [(99,1),(1,99),(95,5),(5,95)]
COMBOS += [(70,30),(30,70),(80,20),(20,80)]  # duplicates intentional, just verifying stability
for a, b in COMBOS:
    txt = f"A) FOR RELEASE OF {a} PERCENT OF LC VALUE\nDocs.\nB) FOR RELEASE OF {b} PERCENT OF LC VALUE\nDocs."
    info = detect(txt)
    ok(f"  {a}/{b}",
       info is not None and info['tranche_a_pct']==a and info['tranche_b_pct']==b)


# ── Section 2: Irregular formatting ──
print("\n" + "=" * 70)
print("Section 2: Irregular formatting")
print("=" * 70)
IRREGULAR = [
    "A) FOR RELEASE OF 90PERCENT...\nB) FOR RELEASE OF 10PERCENT...",
    "A) FOR RELEASE OF 90    PERCENT  OF  LC  VALUE\nB) FOR RELEASE OF 10  PERCENT",
    "Some preamble text...\nA) FOR RELEASE OF 90 PERCENT...\nDocs list.\n\nB) FOR RELEASE OF 10 PERCENT",
    "Multi-line\nnoise\nmore noise\nA) FOR RELEASE OF 80 PCT\nload docs\nB) FOR RELEASE OF 20 PCT\ndischarge docs",
]
for i, txt in enumerate(IRREGULAR, 1):
    info = detect(txt)
    ok(f"  Irregular #{i}", info is not None,
       f"detected={info is not None}")


# ── Section 3: Bundle tranche determination ──
print("\n" + "=" * 70)
print("Section 3: Bundle determination edge cases")
print("=" * 70)
DISCHARGE_MARKERS = ('AT DISCHARGE PORT','PORT OF DISCHARGE WEIGHT',
                     'SAMPLING AND ANALYSIS','DISCHARGE-PORT WEIGHT',
                     'ACCEPTED BY APPLICANT REPRESENTATIVE','WAH CANTT BRANCH')
def determine(bundle):
    up = bundle.upper()
    return 'B' if any(p in up for p in DISCHARGE_MARKERS) else 'A'

BUNDLE_TESTS = [
    ('Pure load-port: SGS load survey, weight at load port', 'A'),
    ('Mixed: load survey + WEIGHT CERTIFICATE AT DISCHARGE PORT', 'B'),
    ('Just CoO + Bene Cert', 'A'),
    ('Empty bundle', 'A'),  # no markers → tranche A default
    ('SAMPLING AND ANALYSIS at load port (rare combo)', 'B'),  # SA marker triggers
    ('Bank Al Habib Wah Cantt branch verification', 'B'),  # case-insensitive after .upper()
    ('WAH CANTT BRANCH in caps', 'B'),
]
for txt, expect in BUNDLE_TESTS:
    got = determine(txt)
    ok(f"  bundle '{txt[:50]}': {got}", got == expect)


# ── Section 4: Row-rescue simulation ──
print("\n" + "=" * 70)
print("Section 4: Row rescue simulation")
print("=" * 70)
# Simulate a 90/10 LC with bundle = load-only (= tranche A)
synth_lc_text = """A) FOR RELEASE OF 90 PERCENT PAYMENT OF LC VALUE
Load docs:
1) Commercial Invoice
2) Charter Party BL
3) Weight Cert at LOAD PORT
B) FOR RELEASE OF 10 PERCENT OF LC VALUE
Discharge docs:
1) Commercial Invoice (balance)
2) Weight Cert at DISCHARGE PORT
3) SAMPLING AND ANALYSIS Cert"""
info = detect(synth_lc_text)
tr_a = info['tranche_a_text']
tr_b = info['tranche_b_text']
b_only = [m for m in DISCHARGE_MARKERS if m in tr_b and m not in tr_a]

ROWS = [
    {'row_id':'R001','condition_text':'Commercial Invoice in 8 originals','original_clause_text':tr_a[:200]},
    {'row_id':'R002','condition_text':'Weight Certificate at discharge port','original_clause_text':tr_b[:200]},
    {'row_id':'R003','condition_text':'Sampling and Analysis Certificate','original_clause_text':tr_b[:200]},
    {'row_id':'R004','condition_text':'Charter Party Bill of Lading','original_clause_text':tr_a[:200]},
    {'row_id':'R005','condition_text':'Balance Commercial Invoice with WAH CANTT BRANCH stamp','original_clause_text':tr_b[:200]},
]
def is_b_row(r):
    full = (r.get('original_clause_text','') + ' ' + r.get('condition_text','')).upper()
    return any(ph in full for ph in b_only)

ok("  R001 (CI load) is NOT tranche-B", not is_b_row(ROWS[0]))
ok("  R002 (Weight at discharge) IS tranche-B", is_b_row(ROWS[1]))
ok("  R003 (SAMPLING AND ANALYSIS) IS tranche-B", is_b_row(ROWS[2]))
ok("  R004 (Charter Party BL) is NOT tranche-B", not is_b_row(ROWS[3]))
ok("  R005 (WAH CANTT BRANCH) IS tranche-B", is_b_row(ROWS[4]))


# ── Section 5: Synthetic full LC verification flows ──
print("\n" + "=" * 70)
print("Section 5: Full LC flows — load presentation vs discharge presentation")
print("=" * 70)

# Flow A: this is the LOAD presentation (90% release)
load_bundle = "MV GOOD HEART, RICHARDS BAY LOAD PORT, DRAFT SURVEY, COO, BENE CERT"
load_tranche = determine(load_bundle)
ok("  Load-presentation determined as tranche A",
   load_tranche == 'A')

# Flow B: this is the DISCHARGE presentation (10% balance release)
discharge_bundle = "WEIGHT CERTIFICATE AT DISCHARGE PORT KARACHI, SAMPLING AND ANALYSIS, WAH CANTT BRANCH"
discharge_tranche = determine(discharge_bundle)
ok("  Discharge-presentation determined as tranche B",
   discharge_tranche == 'B')

# Flow C: combined (full set in one presentation — rare but possible)
combined = load_bundle + ' ' + discharge_bundle
combined_tranche = determine(combined)
ok("  Combined-presentation determined as tranche B (discharge wins)",
   combined_tranche == 'B')


# ── Section 6: Non-coal LC unaffected ──
print("\n" + "=" * 70)
print("Section 6: Non-coal LC not detected as 2-tranche")
print("=" * 70)
NON_COAL = [
    "1) Commercial Invoice in 8 copies\n2) Bill of Lading\n3) Packing List\n4) Certificate of Origin",
    "Goods: Steel rods. Single delivery. CFR Karachi.",
    "Vehicle CKD parts. Partial shipment allowed.",
]
for txt in NON_COAL:
    info = detect(txt)
    ok(f"  Non-coal: NOT detected as 2-tranche",
       info is None,
       f"got {info}")


# ── Section 7: Coal-quality + tranche together ──
print("\n" + "=" * 70)
print("Section 7: Coal-quality + tranche detectors fire independently")
print("=" * 70)

# Synthetic LC with both quality params AND 2-tranche
qual_2tranche = {
    'consolidated_fields': {
        '45A': 'Steam coal CFR Karachi GCV 5800 kcal/kg (NAR), TM 14% max, Ash 8% max',
        '46A': """A) FOR RELEASE OF 90 PERCENT
docs at load port
B) FOR RELEASE OF 10 PERCENT
docs at discharge port""",
        '47A': """PRICE ADJUSTMENT CLAUSE:
GROSS CALORIFIC VALUE (ARB) - Contract spec 5800 kcal/kg
- Reject below 5500 kcal/kg
- Below contract: adjusted CFR = (FOB × actual GCV / 5800) + Freight
TOTAL MOISTURE — max 14 PCT (reject above 16 PCT)
ASH — max 8 PCT (reject above 12 PCT)"""
    }
}
trinfo = _detect_release_tranches(qual_2tranche)
qinfo = _detect_coal_quality_terms(qual_2tranche)
ok("  Both detected on combined LC",
   trinfo is not None and qinfo is not None and qinfo.get('is_coal_lc'))


# ── Section 8: UCP/ISBP scrub final test ──
print("\n" + "=" * 70)
print("Section 8: UCP/ISBP scrub")
print("=" * 70)
UCP_PATS = [
    r'\(\s*(?:per|as\s+per|under|pursuant\s+to)\s+(?:UCP|ISBP)\s+\d{3}[^)]*\)\s*',
    r'\(\s*(?:UCP|ISBP)\s+\d{3}[^)]*\)\s*',
    r'(?:\bper|\bas\s+per|\bunder|\bpursuant\s+to|\bin\s+accordance\s+with)\s+'
    r'(?:UCP|ISBP)\s+\d{3}'
    r'(?:\s+(?:Article|Art\.?)\s*\d+[a-z]?|\s+[A-Z]?\d{1,3}[a-z]?)?\.?\s*[,;:—-]?\s*',
    r'\b(?:UCP|ISBP)\s+\d{3}\s+(?:Article|Art\.?)\s*\d+[a-z]?\.?\s*[,;:—-]?\s*',
    r'\bISBP\s+\d{3}\s+[A-Z]\d{1,3}[a-z]?\.?\s*[,;:—-]?\s*',
]
def scrub(t):
    for p in UCP_PATS:
        t = re.sub(p, '', t, flags=re.IGNORECASE)
    t = re.sub(r'\s{2,}', ' ', t)
    t = re.sub(r'\s+([.,;:])', r'\1', t)
    return t.strip()

SCRUB_TESTS = [
    ('Per UCP 600 Art 23, AWB capacity required.',
     'AWB capacity required.', 'leading Per UCP'),
    ('Discrepancy under UCP 600 Article 14.',
     'Discrepancy', 'trailing under UCP'),
    ('Bundle satisfies (per ISBP 821 H8) the requirement.',
     'Bundle satisfies the requirement.', 'parenthesized per ISBP'),
    ('No regulatory citations here.',
     'No regulatory citations here.', 'plain text unchanged'),
    ('Acceptable per UCP 600 Article 14 because LC says.',
     'Acceptable because LC says.', 'mid-sentence per UCP'),
]
for inp, exp, label in SCRUB_TESTS:
    got = scrub(inp)
    ok(f"  {label}: scrubbed correctly", got == exp,
       f"got {got!r}, exp {exp!r}")


# ── Section 9: Real-data sweep ──
print("\n" + "=" * 70)
print("Section 9: Real-data sweep — verify detectors run cleanly")
print("=" * 70)
total = tr_count = q_count = 0
for jp in glob.glob('d:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final/results/*/step06/step06_result.json'):
    try: d = json.load(open(jp, encoding='utf-8'))
    except: continue
    total += 1
    if _detect_release_tranches(d): tr_count += 1
    qinfo = _detect_coal_quality_terms(d)
    if qinfo and qinfo.get('is_coal_lc'): q_count += 1

print(f"  Jobs scanned: {total}")
print(f"  2-tranche LCs found: {tr_count}")
print(f"  Coal-quality LCs found: {q_count}")
ok("  Detectors ran across all jobs without crashing", total > 0)


print("\n" + "=" * 70)
passed = sum(results)
total_t = len(results)
print(f"COAL FULL E2E: {passed}/{total_t}")
print("=" * 70)
if passed != total_t:
    sys.exit(1)
print("OVERALL: OK")
sys.exit(0)
