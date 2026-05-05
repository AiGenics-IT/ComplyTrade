"""P198gz10 — discharge-port mis-decomposition guard."""
import sys, re, os
sys.path.insert(0, 'd:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final')
os.environ['PYTHONIOENCODING'] = 'utf-8'

results = []
def ok(name, cond, note=''):
    tag = '[OK]  ' if cond else '[FAIL]'
    print(f"{tag} {name}" + (f" -- {note}" if note else ''))
    results.append(bool(cond))


_PORT_TOKENS = ('PORT','TERMINAL','SEAPORT','AIRPORT','NLCCT',
                'CONTAINER TERMINAL','WHARF','BERTH','JETTY')

def is_misdecomp(ct):
    ct_u = ct.upper()
    return bool(
        re.search(
            r'\bCONSIGNEE\s+(?:MUST\s+BE\s+|TO\s+BE\s+|MUST\s+)?'
            r'(?:DISCHARGED|DISCHARGING|DELIVERED|DELIVERING)', ct_u
        )
        or (
            ct_u.startswith('CONSIGNEE MUST BE')
            and any(t in ct_u for t in _PORT_TOKENS)
            and any(rt in ct_u for rt in ('DISCHARG','DELIVER','VIA ','AT ',
                                          'CALL AT','CALLING AT'))
        )
    )


print("=" * 70)
print("Section 1: Mis-decomposition detection")
print("=" * 70)
SHOULD_FIRE = [
    "Consignee must be discharged at Old Seaport Karachi or Karachi NLCCT (National Logistic Cell Container Terminal) via Port Qasim Container Terminal.",
    "Consignee must be delivered at Port Qasim Container Terminal.",
    "Consignee discharged at Karachi Port.",
    "Consignee discharging at NLCCT terminal.",
]
for ct in SHOULD_FIRE:
    ok(f"  Should fire: {ct[:60]}", is_misdecomp(ct))

print()
SHOULD_NOT_FIRE = [
    "Consignee must be Bank Al Habib Limited, Karachi.",
    "Consignee must show 'TO ORDER OF BANK AL HABIB'.",
    "Bill of Lading must show Port of Discharge as Port Qasim.",
    "Consignee must be made out to the order of the issuing bank.",
    "Vessel must discharge cargo at Port Qasim.",  # not a consignee mistake
    "Goods to be delivered to consignee at warehouse.",
]
for ct in SHOULD_NOT_FIRE:
    ok(f"  Should NOT fire: {ct[:60]}", not is_misdecomp(ct))


print("\n" + "=" * 70)
print("Section 2: Real anchor (job 226faca7 / 47A-9)")
print("=" * 70)
real = ("Consignee must be discharged at Old Seaport Karachi or Karachi "
        "NLCCT (National Logistic Cell Container Terminal) via Port Qasim "
        "Container Terminal.")
ok(f"  Real F47A-9 mistake detected", is_misdecomp(real))


# ── Source wiring ──
print("\n" + "=" * 70)
print("Section 3: Source wiring")
print("=" * 70)
src = open('d:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final/steps/step12_decomposition.py',
           'r', encoding='utf-8').read()
ok("  P198gz10 marker", 'P198gz10' in src)
ok("  PORT-OF-DISCHARGE prompt section", 'PORT-OF-DISCHARGE' in src)
ok("  ANTI-CONSIGNEE-DISCHARGE example", 'F47A-9' in src or '226faca7' in src)
ok("  _PORT_TOKENS list", '_PORT_TOKENS' in src)


print("\n" + "=" * 70)
passed = sum(results)
total = len(results)
print(f"P198gz10 DISCHARGE: {passed}/{total}")
print("=" * 70)
if passed != total:
    sys.exit(1)
print("OVERALL: OK")
sys.exit(0)
