"""P198gz18 — SCC Pakistani Maritime Rules / Port Regulations
equivalence. The Maersk-style wording must satisfy the LC's
'OWNED BY COMPANIES OPERATING IN ACCORDANCE WITH PAKISTANI
MARITIME RULES AND PORT REGULATIONS' requirement."""
import sys, re, os
os.environ['PYTHONIOENCODING'] = 'utf-8'

results = []
def ok(name, cond, note=''):
    tag = '[OK]  ' if cond else '[FAIL]'
    print(f"{tag} {name}" + (f" -- {note}" if note else ''))
    results.append(bool(cond))


# Mirror the production regex (P198gz18 expanded)
MARITIME_REQ_RE = re.compile(
    r'\b(?:PAKISTAN(?:I)?\s+MARITIME\s+RULES?|'
    r'OPERATING\s+IN\s+ACCORDANCE\s+WITH\s+PAKISTAN|'
    r'MARITIME\s+RULES?\s+AND\s+PORT\s+REGULATIONS?|'
    r'MARITIME\s+LAWS?\s+AND\s+PORT\s+REGULATIONS?|'
    r'ALLOWED\s+TO\s+ENTER\s+PAKISTAN\s+PORTS|'
    r'ACCORDING\s+TO\s+MARITIME\s+(?:LAWS?|RULES?)\s+'
    r'AND\s+PORT\s+REGULATIONS?)\b',
    re.IGNORECASE,
)


# ── Section 1 — real Maersk SCC text ──
print("=" * 70)
print("Section 1: Real Maersk SCC text variants")
print("=" * 70)

REAL = [
    ("THE CARRYING VESSEL IS ALLOWED TO ENTER PAKISTAN PORTS.",
     True, "Maersk variant 1 (allowed to enter)"),
    ("THAT THE SAID VESSEL IS ALLOWED TO ENTER PAKISTAN PORTS ACCORDING TO MARITIME LAWS AND PORT REGULATIONS.",
     True, "Maersk variant 2 (full sentence)"),
    ("ACCORDING TO MARITIME LAWS AND PORT REGULATIONS",
     True, "Phrase only"),
    ("ACCORDING TO MARITIME RULES AND PORT REGULATIONS",
     True, "Phrase with RULES instead of LAWS"),
    # Original LC wording — should also match
    ("OWNED BY COMPANIES OPERATING IN ACCORDANCE WITH PAKISTANI MARITIME RULES AND PORT REGULATIONS",
     True, "LC literal wording"),
    ("OPERATING IN ACCORDANCE WITH PAKISTAN PORT RULES",
     True, "Operating-in-accordance variant"),
    # Should NOT match
    ("Vessel docked at Karachi port",
     False, "Generic vessel mention"),
    ("Compliance with international rules",
     False, "Generic compliance"),
    ("THE VESSEL IS PAKISTAN FLAGGED",
     False, "Just flag mention"),
]

for txt, expect, label in REAL:
    got = bool(MARITIME_REQ_RE.search(txt))
    ok(f"  {label}: matched={got}", got == expect,
       f"expected {expect}, got {got}")


# ── Section 2 — full SCC text from job b1479424 ──
print("\n" + "=" * 70)
print("Section 2: Full Maersk SCC paragraph (real anchor)")
print("=" * 70)

FULL_MAERSK_SCC = """MAERSK
CERTIFICATE
TO WHOM IT MAY CONCERN
NAME OF VESSEL/VOYAGE : SANTA MARTA EXPRESS / 609S
L/C NO. 1019LC55854/2026
L/C OPENING DATE: 9-JANUARY-2026
L/C OPENING BANK: BANK AL HABIB LIMITED, KARACHI, PAKISTAN
ETA: 27-04-2026 PORT QASIM
-THE UNDERSIGNED DOES HEREBY CERTIFY ON BEHALF OF THE
OWNER, MASTER OR AGENTS OF THE ABOVE NAMED VESSEL THAT
THE SHIPMENT IS EFFECTED BY VESSELS COVERED BY INSTITUTE
CLASSIFICATION CLAUSE.
-THE UNDERSIGNED DOES HEREBY CERTIFY ON BEHALF OF THE
OWNER, MASTER OR AGENTS OF THE ABOVE NAMED VESSEL THAT
THE CARRYING VESSEL IS ALLOWED TO ENTER PAKISTAN PORTS.
-THAT THE SAID VESSEL IS ALLOWED TO ENTER PAKISTAN PORTS
ACCORDING TO MARITIME LAWS AND PORT REGULATIONS.
-That the flag of the above stated vessel is: Denmark
SIGNED FOR THE CARRIER A.P.MOLLER - MAERSK A/S
TRADING AS MAERSK"""

ok("  Maritime requirement satisfied in full SCC",
   bool(MARITIME_REQ_RE.search(FULL_MAERSK_SCC)))
ok("  ICC requirement satisfied in full SCC",
   bool(re.search(r'INSTITUTE\s+CLASSIFICATION\s+CLAUSE',
                  FULL_MAERSK_SCC.upper())))


# ── Section 3 — Source wiring ──
print("\n" + "=" * 70)
print("Section 3: Source wiring")
print("=" * 70)
src = open('d:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final/steps/step14_verification.py',
           'r', encoding='utf-8').read()
ok("  P198gz18 marker", 'P198gz18' in src)
ok("  ALLOWED TO ENTER PAKISTAN PORTS pattern",
   'ALLOWED\\s+TO\\s+ENTER\\s+PAKISTAN\\s+PORTS' in src)
ok("  MARITIME LAWS AND PORT REGULATIONS pattern",
   'MARITIME\\s+LAWS?\\s+AND\\s+PORT\\s+REGULATIONS?' in src)
ok("  SEMANTIC-EQUIVALENCE RULE in prompt",
   'SEMANTIC-EQUIVALENCE RULE' in src)
ok("  ISBP 821 reference in prompt",
   'ISBP 821' in src)


print("\n" + "=" * 70)
passed = sum(results)
total = len(results)
print(f"P198gz18 SCC MARITIME: {passed}/{total}")
print("=" * 70)
if passed != total:
    sys.exit(1)
print("OVERALL: OK")
sys.exit(0)
