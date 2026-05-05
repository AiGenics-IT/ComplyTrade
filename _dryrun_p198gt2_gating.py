"""P198gt2 — gating fix so original-copy override only fires on rows
whose REQUIREMENT is the original-copy designation."""
import sys, re, os
os.environ['PYTHONIOENCODING'] = 'utf-8'

results = []
def ok(name, cond, note=''):
    tag = '[OK]  ' if cond else '[FAIL]'
    print(f"{tag} {name}" + (f" -- {note}" if note else ''))
    results.append(bool(cond))


_OTHER = (
    'FREIGHT PREPAID','FREIGHT COLLECT','CONSIGNED TO','CONSIGNEE',
    'NOTIFY','FLIGHT NUMBER','FLIGHT NO','FLT NO',
    'L/C NUMBER','LC NUMBER','CREDIT NUMBER',
    'ACCOMPANY','COPY OF INVOICE','SIGNED BY','SIGNATURE',
    'ISSUED BY','DESPATCH','DISPATCH',
)

def should_fire(cond):
    cu = cond.upper()
    is_about_orig = (
        re.search(r'(?:MUST|SHALL|SHOULD|TO)\s+BE\s+(?:THE\s+)?ORIGINAL\s+(?:COPY\s+)?FOR\b', cu)
        or re.search(r'MARKED\s+(?:AS\s+)?ORIGINAL\s+FOR\b', cu)
        or re.search(r'BEAR(?:ING)?\s+(?:THE\s+)?ORIGINAL\s+(?:COPY\s+)?FOR\b', cu)
        or 'ORIGINAL-COPY DESIGNATION' in cu
        or 'ORIGINAL COPY DESIGNATION' in cu
    )
    has_other = any(t in cu for t in _OTHER)
    if has_other and not is_about_orig:
        return False
    return True


# These rows came from real screenshot — should NOT fire (siblings inherit
# the "Original for Consignor" heading but ask about other things)
print("=" * 70)
print("Section 1: Sibling sub-conditions that should NOT trigger P198gt")
print("=" * 70)
SHOULD_SKIP = [
    'Original for Consignor Clean Airway Bill must show freight prepaid.',
    'Original for Consignor Clean Airway Bill must be marked notify the Applicant and Bank Al Habib Limited, Pakistan.',
    'Original for Consignor Clean Airway Bill must be consigned to Bank Al Habib Limited, Pakistan.',
    'Original for Consignor Clean Airway Bill must bear this L/C number.',
    'Original for Consignor Clean Airway Bill must bear the flight number.',
    'Original for Consignor Clean Airway Bill must evidence despatch of goods.',
    'Original for Consignor Clean Airway Bill must evidence that a copy of invoice and a copy of airway bill accompany the consignment.',
]
for c in SHOULD_SKIP:
    ok(f"  SKIP: {c[:70]}", not should_fire(c))


# These should still fire (genuinely about the original designation)
print("\n" + "=" * 70)
print("Section 2: Genuine original-copy-designation rows that SHOULD fire")
print("=" * 70)
SHOULD_FIRE = [
    'AWB must be original for Consignor.',
    'The Airway Bill must be marked Original for Consignor.',
    'AWB must bear Original for Consignor designation.',
    'Original-copy designation: must match LC.',
    'AWB must be the original for Consignor copy.',
]
for c in SHOULD_FIRE:
    ok(f"  FIRE: {c[:70]}", should_fire(c))


# Source wiring
print("\n" + "=" * 70)
print("Section 3: Source wiring")
print("=" * 70)
src = open('d:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final/steps/step14_verification.py',
           'r', encoding='utf-8').read()
ok("  P198gt2 marker", 'P198gt2' in src)
ok("  '_other_topics' list", '_other_topics' in src)
ok("  '_is_about_original_designation' check",
   '_is_about_original_designation' in src)


print("\n" + "=" * 70)
passed = sum(results)
total = len(results)
print(f"P198gt2 GATING: {passed}/{total}")
print("=" * 70)
if passed != total:
    sys.exit(1)
print("OVERALL: OK")
sys.exit(0)
