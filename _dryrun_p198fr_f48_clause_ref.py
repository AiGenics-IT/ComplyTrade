"""
P198fr dry-run — F48 must NOT collapse to all of F47A when it carries
a "REFER FIELD 47A CLAUSE N" reference.

Bug: the LC F48 = "Period for Presentation in Days\nDays: 21\n
     Narrative: /REFER FIELD 47A CLAUSE 10"  was being replaced
     entirely with the whole F47A "Additional Conditions" block,
     wiping out the actual presentation period.

Real-data anchor:
  Job ff87b18c — F48 saw the BUG (Additional Conditions text bled in).
  After fix, F48 should show the correct value with clause 10 inlined.
"""
import sys, os, re
sys.path.insert(0, 'd:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final')

results = []
def assert_eq(name, got, expected):
    ok = (got == expected)
    print(f"[{'OK' if ok else 'FAIL'}] {name}")
    if not ok:
        print(f"          got     : {got!r}")
        print(f"          expected: {expected!r}")
    results.append(ok)

# ── Test 1: regex matches "CLAUSE N" (no NO.) ───────────────────────
print("--- P198fr: 'REFER FIELD YY CLAUSE N' regex matches (no NO.) ---")
RX = re.compile(
    r'REFER\s+FIELD\s+(\d{2}[A-Z]?)\s+CLAUSE\s+(?:NO\.?\s*)?(\d+)',
    re.IGNORECASE,
)
CASES = [
    ("REFER FIELD 47A CLAUSE 10", '47A', 10),
    ("/REFER FIELD 47A CLAUSE 10", '47A', 10),
    ("REFER FIELD 47A CLAUSE NO. 10", '47A', 10),
    ("REFER FIELD 47A CLAUSE NO.3", '47A', 3),
    ("Refer field 45A clause 5", '45A', 5),
    ("REFER FIELD 47B CLAUSE 7", '47B', 7),
]
for txt, exp_tag, exp_num in CASES:
    m = RX.search(txt)
    assert_eq(f"matches {txt!r}", m is not None, True)
    if m:
        assert_eq(f"  field tag = {exp_tag}", m.group(1), exp_tag)
        assert_eq(f"  clause num = {exp_num}", int(m.group(2)), exp_num)

# Negative — must NOT match these
NEG = [
    "SEE FIELD 47A",                  # no clause number
    "AS PER FIELD 47A",               # no clause number
    "REFER FIELD 45A FOR DETAILS",    # no clause number
    "Refer to clause 10",             # no FIELD
]
for txt in NEG:
    m = RX.search(txt)
    assert_eq(f"NEG: doesn't match {txt!r}", m is None, True)


# ── Test 2: source code has the new pattern BEFORE C2 ─────────────
print("\n--- P198fr: source ordering ---")
src = open('d:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final/steps/step06_final_lc.py',
           'r', encoding='utf-8').read()
p_fr_idx = src.find('P198fr')
c2_idx = src.find('Pattern C2')
assert_eq("P198fr block exists", p_fr_idx > 0, True)
assert_eq("P198fr block precedes Pattern C2", p_fr_idx < c2_idx, True)
assert_eq("P198fr regex tolerates missing 'NO.'",
          'CLAUSE\\s+(?:NO\\.?\\s*)?(\\d+)' in src, True)
assert_eq("Pattern C2 has _has_clause_ref guard",
          '_has_clause_ref' in src, True)


# ── Test 3: simulate the fix on the real ff87b18c F48 value ─────────
print("\n--- P198fr: real-data simulation on ff87b18c F48 ---")
_F48_RAW = ("Period for Presentation in Days\n"
            "Days: 21\n"
            "Narrative: /REFER FIELD 47A CLAUSE 10")
_F47A_RAW = ("1)DOCUMENTS DATED PRIOR TO DATE OF ISSUANCE OF THIS LC ACCEPTABLE.\n"
             "2)DRAFT, INVOICE AND BILL OF LADING MUST SHOW OUR DOCUMENTARY CREDIT NUMBER...\n"
             "3)CHARTER PARTY, SHORT FORM AND BLANK BACK BILL OF LADING ACCEPTABLE.\n"
             "4)FREIGHT FORWARDER AND HOUSE B/L NOT ACCEPTABLE\n"
             "5)ALL DOCUMENTS MUST BE DATED AND MADE OUT IN ENGLISH LANGUAGE\n"
             "6)ANY OVERWRITING, ALTERATION...\n"
             "7)NEGOTIATING BANK MUST CERTIFY...\n"
             "8)USD 116/- DISCREPANCY CHARGES...\n"
             "9)INVOICES EXCEEDING THIS CREDIT AMOUNT NOT ACCEPTABLE.\n"
             "10)DOCUMENTS WITHIN 21 DAYS OF SHIPMENT BUT WITHIN LC EXPIRY.\n"
             "11)DOCUMENTS MUST BE SENT TO BANK AL-HABIB LTD...\n"
             "12)CERTIFICATES SHOWING QUANTITY DIFFERENT...")

# Confirm our regex DOES match on F48 raw
m = RX.search(_F48_RAW)
assert_eq("real F48 raw triggers P198fr regex", m is not None, True)
# And the BARE pattern (Pattern C2) ALSO would match — that's why
# the order matters. Confirm:
BARE = re.compile(
    r'(?:SEE|REFER(?:\s+TO)?|AS\s+PER)\s+FIELD\s+(\d{2}[A-Z]?)',
    re.IGNORECASE,
)
m2 = BARE.search(_F48_RAW)
assert_eq("bare-ref pattern also matches F48 raw (priority test)",
          m2 is not None, True)

# Confirm the residual after our regex match leaves clause-specific text
# (i.e. the fix doesn't accidentally strip too much from F48)
post = _F48_RAW[:m.start()] + 'CLAUSE_10_TEXT' + _F48_RAW[m.end():]
assert_eq("F48 keeps surrounding text after substitution",
          "Period for Presentation" in post and "Days: 21" in post, True)


# ── Test 4: Pattern B (forward order) also accepts CLAUSE without NO. ──
print("\n--- P198fr: Pattern B accepts both 'CLAUSE NO X' and 'CLAUSE X' ---")
PB = re.compile(
    r'(?:PLS\s+)?REFER\s+(?:TO\s+)?CLAUSE\s+(?:NO\.?\s*)?(\d+)\s+OF\s+FIELD\s+(\d{2}[A-Z]?)',
    re.IGNORECASE,
)
B_CASES = [
    ("REFER CLAUSE NO.10 OF FIELD 47A", '10', '47A'),
    ("REFER CLAUSE NO. 10 OF FIELD 47A", '10', '47A'),
    ("REFER CLAUSE 10 OF FIELD 47A", '10', '47A'),
    ("PLS REFER CLAUSE 5 OF FIELD 47A", '5', '47A'),
    ("PLS REFER TO CLAUSE NO.5 OF FIELD 47B", '5', '47B'),
    ("REFER TO CLAUSE 7 OF FIELD 45A", '7', '45A'),
]
for txt, exp_n, exp_t in B_CASES:
    m = PB.search(txt)
    assert_eq(f"Pattern B: {txt!r} matches", m is not None, True)
    if m:
        assert_eq(f"  clause = {exp_n}", m.group(1), exp_n)
        assert_eq(f"  field tag = {exp_t}", m.group(2), exp_t)


passed = sum(results)
total = len(results)
print(f"\n{passed}/{total} cases passed")
if passed != total:
    sys.exit(1)
print("OVERALL: OK")
sys.exit(0)
