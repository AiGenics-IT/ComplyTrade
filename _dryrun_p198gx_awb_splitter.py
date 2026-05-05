"""P198gx — AWB compound-clause deterministic splitter."""
import sys, os, re
sys.path.insert(0, 'd:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final')
os.environ['PYTHONIOENCODING'] = 'utf-8'

results = []
def ok(name, cond, note=''):
    tag = '[OK]  ' if cond else '[FAIL]'
    print(f"{tag} {name}" + (f" -- {note}" if note else ''))
    results.append(bool(cond))


# Mirror the splitter logic
def split_awb(original_text, current_count):
    if current_count != 1:
        return current_count
    up = original_text.upper()
    if not re.search(r'\b(?:AIR\s*WAY\s*BILL|AIRWAYBILL|AWB)\b', up):
        return current_count
    n = 0
    if re.search(r'\bL\s*/\s*C\s+NUMBER\b|\bLC\s+NUMBER\b|\bCREDIT\s+NUMBER\b', up):
        n += 1
    if 'FLIGHT NUMBER' in up or 'FLIGHT NO' in up:
        n += 1
    if 'FREIGHT PREPAID' in up:
        n += 1
    if 'CONSIGNED TO' in up or re.search(r'\bCONSIGNEE\b', up):
        n += 1
    if 'NOTIFY' in up:
        n += 1
    if 'ACCOMPANY' in up:
        n += 1
    return n if n >= 2 else current_count


# ── Real anchor: 46A-2 ─────────────────────────────────────────
print("=" * 70)
print("Section 1: Real anchor (job 94edb6a7 46A-2)")
print("=" * 70)
real = (
    "ORIGINAL FOR CONSIGNOR CLEAN AIRWAY BILL BEARING THIS L/C NUMBER "
    "AND FLIGHT NUMBER EVIDENCING DESPATCH OF GOODS CONSIGNED TO BANK "
    "AL HABIB LIMITED, PAKISTAN SHOWING FREIGHT PREPAID MARKED NOTIFY "
    "THE APPLICANT AND BANK AL HABIB LIMITED, PAKISTAN AIRWAY BILL "
    "MUST EVIDENCE THAT A COPY OF INVOICE AND A COPY OF AIRWAY BILL "
    "ACCOMPANY THE CONSIGNMENT"
)
got = split_awb(real, 1)
ok(f"  Real 46A-2: 1 -> {got} (expect >=5)", got >= 5)

# ── Cases ────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Section 2: Other AWB compound clauses")
print("=" * 70)
CASES = [
    ("Airway bill must bear the LC number and flight number",
     1, 2, "LC# + flight# only"),
    ("Air Waybill marked freight prepaid notify applicant",
     1, 2, "freight prepaid + notify"),
    ("Airway bill consigned to ABC Bank showing freight prepaid",
     1, 2, "consignee + freight"),
    ("Air Waybill bearing LC number, consigned to XYZ, freight prepaid, notify applicant",
     1, 4, "all 4 markers"),
    # Should NOT split
    ("Bill of Lading marked freight prepaid",
     1, 1, "Not AWB - skip"),
    ("Air waybill",
     1, 1, "Trivial - no splittable markers"),
    # LLM already split
    ("Air waybill must bear LC number AND flight number",
     3, 3, "Already 3 - skip"),
]
for txt, current, expected, label in CASES:
    got = split_awb(txt, current)
    ok(f"  {label}: {current} -> {got} (expect {expected})",
       got == expected, f"got {got}")

# ── Source wiring ───────────────────────────────────────────
print("\n" + "=" * 70)
print("Section 3: Source wiring")
print("=" * 70)
src = open('d:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final/steps/step12_decomposition.py',
           'r', encoding='utf-8').read()
ok("  P198gx marker in step12", 'P198gx' in src)
ok("  AWB splitter present", 'AWB compound-clause' in src or 'AWB splitter' in src)

print("\n" + "=" * 70)
passed = sum(results)
total = len(results)
print(f"P198gx: {passed}/{total}")
print("=" * 70)
if passed != total:
    sys.exit(1)
print("OVERALL: OK")
sys.exit(0)
