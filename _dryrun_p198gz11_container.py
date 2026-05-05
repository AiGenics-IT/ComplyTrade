"""P198gz11 — container-stuffing/physical-packing filter."""
import sys, re, os
os.environ['PYTHONIOENCODING'] = 'utf-8'

results = []
def ok(name, cond, note=''):
    tag = '[OK]  ' if cond else '[FAIL]'
    print(f"{tag} {name}" + (f" -- {note}" if note else ''))
    results.append(bool(cond))


def is_container_stuffing(ct):
    ct_u = ct.upper()
    return bool(
        re.search(
            r'\b(?:ENCLOSED|STUFFED|PLACED|INSERTED|PACKED)\s+(?:WITHIN|INSIDE|IN|INTO)\s+(?:ALL|EACH|EVERY|THE)?\s*(?:THE\s+)?CONTAINER',
            ct_u
        )
        or re.search(
            r'\b(?:MUST\s+BE\s+|TO\s+BE\s+)?ENCLOSED\s+(?:WITHIN|INSIDE|IN|INTO)\s+(?:ALL|EACH|EVERY|THE)?\s*CONTAINER',
            ct_u
        )
        or re.search(
            r'\bACCOMPANY\s+THE\s+(?:GOODS|SHIPMENT|CARGO)\s+(?:INSIDE|WITHIN|IN)\s+(?:THE\s+)?CONTAINER',
            ct_u
        )
    )


print("=" * 70)
print("Section 1: Container-stuffing detection (should fire)")
print("=" * 70)
SHOULD_FIRE = [
    "Invoice must be enclosed within all containers.",
    "Weight and Packing List must be enclosed within all containers.",
    "Documents to be enclosed within the containers.",
    "Copy of invoice to be stuffed in each container.",
    "Packing list must be placed inside all containers.",
    "Documents must accompany the goods inside the container.",
    "Invoice and Packing List to be enclosed within all containers.",
]
for ct in SHOULD_FIRE:
    ok(f"  Should fire: {ct[:60]}", is_container_stuffing(ct))


print("\n" + "=" * 70)
print("Section 2: Should NOT fire (different requirements)")
print("=" * 70)
SHOULD_NOT = [
    "Invoice must be in 3 originals.",
    "AWB must evidence that a copy of invoice and a copy of airway bill accompany the consignment.",  # accompany consignment, NOT inside container
    "Packing list must show net weight and gross weight.",
    "Bill of Lading must show port of discharge as Karachi.",
    "Goods to be shipped in containers.",  # shipping mode, not stuffing docs
    "Container number must appear on BL.",
]
for ct in SHOULD_NOT:
    ok(f"  Should NOT fire: {ct[:65]}", not is_container_stuffing(ct))


print("\n" + "=" * 70)
print("Section 3: Real anchor (job 226faca7 / 47A-13)")
print("=" * 70)
real1 = "Invoice must be enclosed within all containers."
real2 = "Weight and Packing List must be enclosed within all containers."
ok(f"  Real F47A-13 invoice clause", is_container_stuffing(real1))
ok(f"  Real F47A-13 PL clause", is_container_stuffing(real2))


print("\n" + "=" * 70)
print("Section 4: Source wiring")
print("=" * 70)
src = open('d:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final/steps/step12_decomposition.py',
           'r', encoding='utf-8').read()
ok("  P198gz11 marker", 'P198gz11' in src)
ok("  Container-stuffing prompt section", 'container-stuffing' in src.lower() or 'PHYSICAL PACKING' in src or 'physical packing' in src.lower())
ok("  ENCLOSED.*CONTAINER regex pattern", 'ENCLOSED' in src and 'CONTAINER' in src)


print("\n" + "=" * 70)
passed = sum(results)
total = len(results)
print(f"P198gz11 CONTAINER-STUFFING: {passed}/{total}")
print("=" * 70)
if passed != total:
    sys.exit(1)
print("OVERALL: OK")
sys.exit(0)
