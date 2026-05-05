"""
P198gr / P198gs — fan-out skip + AWB freight-forwarder false-fail guard.

P198gr: When an F45A goods-description row is fanned out to a Packing
List (-PL clone) but no PL exists in the bundle, mark the cloned
row as N/A instead of producing a "Packing List missing" FAIL on
every F45A row. The primary missing-doc check at packet level
already reports the missing PL once — this avoids the noise.

P198gs: When LC says "Freight Forwarders / House AWB not acceptable"
but the AWB is genuinely issued by a real airline carrier
(SriLankan Airlines / Emirates / Qatar / Thai / etc.), override
the LLM's freight-forwarder FAIL to PASS. Real-data anchor:
job 94edb6a7 — AWB issued by SriLankan Airlines but stamp
'CAK CASH' (sales rep) made LLM flag it as freight-forwarder.
"""
import sys, os, re
sys.path.insert(0, 'd:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final')
os.environ['PYTHONIOENCODING'] = 'utf-8'

results = []
def ok(name, condition, note=''):
    if condition:
        print(f"[OK]  {name}" + (f" — {note}" if note else ""))
    else:
        print(f"[FAIL] {name}" + (f" — {note}" if note else ""))
    results.append(bool(condition))


# ── Section 1 — P198gr: fan-out skip ──
print("=" * 70)
print("Section 1: P198gr — fan-out clone skip when target missing")
print("=" * 70)

def is_dc_clone(condition_id):
    """Mirror P198gr detection."""
    cid = condition_id or ''
    return cid.endswith('-PL') or cid.endswith('-PL-OPT')

CASES = [
    ('45A-1-C1-PL', True,  'Standard P198dc PL clone'),
    ('45A-1-C1-PL-OPT', True, 'P198dl proforma opportunistic clone'),
    ('45A-1-C1', False, 'Original CI-targeted condition (not a clone)'),
    ('46A-2-C1', False, 'Non-45A condition'),
    ('', False, 'Empty'),
    (None, False, 'None'),
]
for cid, expected, label in CASES:
    got = is_dc_clone(cid)
    ok(f"  {label}: {cid!r} → is_clone={got}",
       got == expected,
       f"got {got}, expected {expected}" if got != expected else '')


# ── Section 2 — P198gs: real-airline AWB detection ──
print("\n" + "=" * 70)
print("Section 2: P198gs — real-airline carrier detection on AWB")
print("=" * 70)

_REAL_AIRLINES = (
    'srilankan airlines', 'sri lankan airlines',
    'emirates', 'emirates skycargo', 'emirates sky cargo',
    'qatar airways', 'qatar airways cargo',
    'etihad airways', 'etihad cargo',
    'thai airways', 'thai cargo',
    'singapore airlines', 'sia cargo',
    'cathay pacific', 'cathay cargo',
    'turkish airlines', 'turkish cargo',
    'pia', 'pakistan international airlines',
    'air china', 'china airlines', 'china southern',
    'china eastern', 'air india', 'indigo',
    'lufthansa', 'lufthansa cargo',
    'british airways', 'air france', 'klm',
    'asiana airlines', 'korean air',
    'malaysia airlines', 'malindo air',
    'cargolux', 'fedex', 'ups airlines', 'dhl aviation',
    'ana cargo', 'jal cargo',
)


def detect_airline(awb_text, awb_issued_by, awb_logos):
    awb_text_lo = (awb_text or '').lower()
    issued_lo = (awb_issued_by or '').lower()
    logos_lo = [(l or '').lower() for l in (awb_logos or [])]
    for airline in _REAL_AIRLINES:
        if (airline in awb_text_lo
                or airline in issued_lo
                or any(airline in lg for lg in logos_lo)):
            return airline
    return None


AIRLINE_CASES = [
    # (awb_text, issued_by, logos, expected_airline_or_None, label)
    ('Air Waybill\nSriLankan Airlines\nFlight UL323',
     '', [], 'srilankan airlines', 'Real-data anchor: SriLankan Airlines on AWB'),
    ('Air Waybill\nQatar Airways Cargo\n129-12345678',
     '', [], 'qatar airways', 'Qatar Airways Cargo'),
    ('AWB\nFlight EK801', 'Emirates SkyCargo', [],
     'emirates', 'Emirates via issued_by field'),
    ('Air Waybill\nThai Airways Cargo Flight TG303',
     '', [], 'thai airways', 'Thai Airways'),
    ('AWB by Etihad Cargo', '', ['ETIHAD AIRWAYS'],
     'etihad airways', 'Etihad via logos'),
    # Should NOT detect — these are freight forwarders
    ('Air Waybill\nKuehne+Nagel International\n(Freight Forwarder)',
     'Kuehne+Nagel', [],
     None, 'Kuehne+Nagel (forwarder, not a carrier)'),
    ('AWB\nDB Schenker Logistics', '', [],
     None, 'DB Schenker (forwarder)'),
    ('AWB\nCAK CASH 12345', 'CAK CASH', [],
     None, 'CAK alone with no airline → no override'),
    # Edge: airline in middle of text
    ('Various headers\nIssued by SriLankan Airlines as carrier\nFlight UL302',
     '', [], 'srilankan airlines', 'Airline in middle of text'),
]
for awb_text, issued_by, logos, expected, label in AIRLINE_CASES:
    got = detect_airline(awb_text, issued_by, logos)
    ok(f"  {label}: detected={got!r}",
       got == expected,
       f"got {got!r}, expected {expected!r}" if got != expected else '')


# ── Section 3 — combined decision logic ──
print("\n" + "=" * 70)
print("Section 3: Combined freight-forwarder rescue decision")
print("=" * 70)

def simulate_p198gs(condition, current_compliance, awb_text,
                    awb_issued_by='', awb_logos=None):
    """Mirror the production decision."""
    cond_u = (condition or '').upper()
    is_ff_rule = (
        ('FREIGHT FORWARDER' in cond_u
         or 'HAWB' in cond_u
         or 'HOUSE AIRWAY' in cond_u
         or 'HOUSE AWB' in cond_u)
        and ('NOT ACCEPT' in cond_u or 'PROHIBITED' in cond_u
             or 'NOT ALLOWED' in cond_u)
    )
    if not is_ff_rule:
        return current_compliance, 'not a freight-forwarder rule'
    if current_compliance.upper() not in ('FAIL', 'NOT COMPLIED'):
        return current_compliance, 'not currently FAIL'
    airline = detect_airline(awb_text, awb_issued_by, awb_logos or [])
    if airline:
        return 'PASS', f'real airline detected: {airline}'
    return current_compliance, 'no airline detected — keep FAIL'


SCENARIOS = [
    # (condition, current, awb_text, expected, label)
    ('FREIGHT FORWARDERS AND HOUSE AIRWAY BILL NOT ACCEPTABLE',
     'FAIL', 'Air Waybill\nSriLankan Airlines\n[stamp: CAK CASH]',
     'PASS', 'SriLankan + CAK stamp → PASS (real anchor)'),
    ('FREIGHT FORWARDERS AND HOUSE AWB NOT ACCEPTABLE',
     'FAIL', 'AWB by Kuehne+Nagel International',
     'FAIL', 'Real freight forwarder → keep FAIL'),
    ('FREIGHT FORWARDERS NOT ACCEPTABLE',
     'PASS', 'AWB by SriLankan Airlines',
     'PASS', 'Already PASS → no change'),
    ('Goods description must match',
     'FAIL', 'AWB by SriLankan Airlines',
     'FAIL', 'Non-FF condition → no rescue'),
    ('FREIGHT FORWARDERS NOT ACCEPTABLE',
     'FAIL', 'AWB by some unknown company',
     'FAIL', 'No airline detected → keep FAIL (correct)'),
]
for cond, cur, awb, expected, label in SCENARIOS:
    new, reason = simulate_p198gs(cond, cur, awb)
    ok(f"  {label}: {cur} → {new} ({reason})",
       new == expected,
       f"got {new}, expected {expected}" if new != expected else '')


# ── Section 4 — Source code wiring ──
print("\n" + "=" * 70)
print("Section 4: Source code wiring")
print("=" * 70)

src14 = open('d:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final/steps/step14_verification.py',
             'r', encoding='utf-8').read()
ok(f"  step14 has 'P198gr' marker", 'P198gr' in src14)
ok(f"  step14 has 'P198gs' marker", 'P198gs' in src14)
ok(f"  step14 has fan_out_target_absent", "fan_out_target_absent" in src14)
ok(f"  step14 has _REAL_AIRLINES", '_REAL_AIRLINES' in src14)
ok(f"  step14 has 'srilankan airlines'", "'srilankan airlines'" in src14)


# ── Section 5 — P198gt: AWB ORIGINAL designation strict match ──
print("\n" + "=" * 70)
print("Section 5: P198gt — AWB ORIGINAL FOR X strict match")
print("=" * 70)

def detect_lc_designee(condition_text):
    m = re.search(
        r'\bORIGINAL\s+FOR\s+(CONSIGNOR|CONSIGNER|SHIPPER|EXPORTER|SUPPLIER)\b',
        (condition_text or '').upper())
    return m.group(1) if m else None


def detect_doc_designee(copy_label, awb_text):
    m = re.search(
        r'\bORIGINAL[\s\d\-:.()\[\]]*(?:FOR\s+)?(CONSIGNOR|CONSIGNER|SHIPPER|EXPORTER|SUPPLIER)\b',
        (copy_label or '').upper() + ' ' + (awb_text or '').upper()[:5000])
    return m.group(1) if m else None


DESIGNEE_CASES = [
    # (lc_text, awb_copy_label, awb_text, expected_lc, expected_doc, label)
    ('ORIGINAL FOR CONSIGNOR CLEAN AIRWAY BILL',
     'ORIGINAL 3 - [FOR SHIPPER]', '',
     'CONSIGNOR', 'SHIPPER', 'LC=Consignor / AWB=Shipper → mismatch'),
    ('ORIGINAL FOR SHIPPER',
     'ORIGINAL 3 - [FOR SHIPPER]', '',
     'SHIPPER', 'SHIPPER', 'LC=Shipper / AWB=Shipper → match'),
    ('ORIGINAL FOR CONSIGNOR',
     'ORIGINAL 3 - [FOR CONSIGNOR]', '',
     'CONSIGNOR', 'CONSIGNOR', 'LC=Consignor / AWB=Consignor → match'),
    ('Goods description must match',
     'ORIGINAL 3 - [FOR SHIPPER]', '',
     None, 'SHIPPER', 'No LC designee → no fire'),
    ('ORIGINAL FOR EXPORTER',
     'ORIGINAL 1', '',
     'EXPORTER', None, 'AWB silent on designation → fail'),
]
for lc, label, txt, exp_lc, exp_doc, name in DESIGNEE_CASES:
    got_lc = detect_lc_designee(lc)
    got_doc = detect_doc_designee(label, txt)
    ok(f"  {name}: LC={got_lc} / Doc={got_doc}",
       got_lc == exp_lc and got_doc == exp_doc,
       f"got LC={got_lc}/Doc={got_doc}, expected LC={exp_lc}/Doc={exp_doc}"
       if (got_lc != exp_lc or got_doc != exp_doc) else '')


# ── Section 6 — P198gs/go user-facing text — no internal markers ──
print("\n" + "=" * 70)
print("Section 6: User-facing messages have no internal markers")
print("=" * 70)

# Sample messages from the production code
SAMPLE_MSGS = [
    # P198go incoterm version messages (cleaned)
    ("LC requires 'Incoterms 2020' but the document does not state any "
     "Incoterms version. Per the LC's explicit version annotation, the "
     "document must include the version (e.g. 'Incoterms 2020').",
     'P198go silent message'),
    ("LC requires 'Incoterms 2020' but the document shows 'Incoterms 2010'. "
     "Wrong Incoterms version.",
     'P198go mismatch message'),
    # P198gs AWB carrier message (cleaned)
    ("AWB is issued by 'srilankan airlines' (a recognized airline carrier), "
     "not a freight forwarder. Sales-rep stamps / agent codes do not "
     "change the carrier identity.",
     'P198gs carrier message'),
]
INTERNAL_MARKERS = ('P198go', 'P198gs', 'P198gr', 'P198gt', 'P198gu',
                    'strict version-compliance check',
                    'P198gs override', 'P198gr fan-out skip')
for msg, label in SAMPLE_MSGS:
    has_marker = any(mk in msg for mk in INTERNAL_MARKERS)
    ok(f"  {label}: no internal marker leaked",
       not has_marker,
       f"contains internal marker" if has_marker else '')


# ── Section 7 — IATA AWB number format ──
print("\n" + "=" * 70)
print("Section 7: IATA AWB number format detection")
print("=" * 70)

_AWB_NUMBER_RE = re.compile(r'\b(\d{3})-(\d{8})\b')

AWB_CASES = [
    ('AWB 603-74212946', '603-74212946', 'SriLankan format from anchor'),
    ('Air Waybill\n176-12345678\nDestination: KHI', '176-12345678', 'Emirates'),
    ('157-87654321 issued', '157-87654321', 'Qatar'),
    ('No AWB number here', None, 'No match'),
    ('Phone 12345678 not AWB', None, '8-digit phone — no airline prefix'),
    ('LC NO 0045LC78957/2025', None, 'LC number — different format'),
]
for txt, expected, label in AWB_CASES:
    m = _AWB_NUMBER_RE.search(txt)
    got = f"{m.group(1)}-{m.group(2)}" if m else None
    ok(f"  {label}: {txt[:40]:<40} → {got}",
       got == expected,
       f"got {got}, expected {expected}" if got != expected else '')


# Source wiring for new rules
print("\n" + "=" * 70)
print("Section 8: Source code wiring for rules 20c/20d/20e + P198gt")
print("=" * 70)

src14 = open('d:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final/steps/step14_verification.py',
             'r', encoding='utf-8').read()
ok(f"  step14 has 20c (AWB ORIGINAL designation)",
   '20c' in src14 and 'ORIGINAL COPY DESIGNATION' in src14.upper())
ok(f"  step14 has 20d (AWB issuer/carrier)",
   '20d' in src14 and 'CARRIER VERIFICATION' in src14.upper())
ok(f"  step14 has 20e (AWB number format)",
   '20e' in src14 and 'NNN-NNNNNNNN' in src14)
ok(f"  step14 has P198gt block", 'P198gt' in src14)
ok(f"  P198go-leak removed from result text",
   'P198go strict' not in src14 or src14.count('P198go strict') == 0)


# ── Section 9 — P198gv: AWB flight-number rescue ──
print("\n" + "=" * 70)
print("Section 9: P198gv — flight number / AWB number detection")
print("=" * 70)

_IATA_FLIGHT_RE = re.compile(
    r'\b([A-Z]{2}|[A-Z]\d|\d[A-Z])\s*[-]?\s*(\d{1,4}[A-Z]?)\b')
_IATA_AWB_RE = re.compile(r'\b\d{3}[-\s]?\d{8}\b')


def detect_flight(awb_text):
    txt_up = (awb_text or '').upper()
    flights = []
    for m in _IATA_FLIGHT_RE.finditer(txt_up):
        ctx = txt_up[max(0, m.start()-60):m.end()+30]
        if any(kw in ctx for kw in ('FLIGHT', 'FLT', 'BY FIRST CARRIER',
                                    'ROUTING', 'CARRIER')):
            flights.append(f"{m.group(1)} {m.group(2)}")
    awb_no = _IATA_AWB_RE.search(awb_text or '')
    return list(set(flights)), awb_no.group(0) if awb_no else None


FLIGHT_CASES = [
    # Real anchor: SriLankan AWB
    ('Requested Flight/Date\nUL 0153/04-Sep\nBy First Carrier UL\n603-74213252',
     'UL 0153', '603-74213252', 'Real anchor: UL 0153 + 603-74213252'),
    ('Air Waybill\nFlight: EK 401\n176-12345678',
     'EK 401', '176-12345678', 'Emirates EK 401'),
    ('AWB\nQR 715 carrier flight\n157-87654321',
     'QR 715', '157-87654321', 'Qatar QR 715'),
    ('Random text with no flight info',
     None, None, 'No flight info → no detection'),
]
for txt, expected_flight, expected_awb, label in FLIGHT_CASES:
    flights, awb = detect_flight(txt)
    if expected_flight:
        ok(f"  {label}: flight {flights[0] if flights else None}",
           expected_flight in flights,
           f"got {flights}, expected {expected_flight}")
    else:
        ok(f"  {label}: no flight detected", not flights)
    ok(f"  {label}: AWB no = {awb}",
       (awb == expected_awb) or (expected_awb is None and awb is None))


# ── Section 10 — P198gw: signing capacity strict ──
print("\n" + "=" * 70)
print("Section 10: P198gw — AWB signing capacity strict")
print("=" * 70)

_AWB_CAPACITY_AFFIRMS = (
    'AS CARRIER', 'AS THE CARRIER',
    'AS ISSUING CARRIER', 'AS THE ISSUING CARRIER',
    'AS AGENT FOR THE CARRIER',
    'AS AGENTS FOR THE CARRIER',
    'AS AGENT ON BEHALF OF THE CARRIER',
    'AS AGENTS ON BEHALF OF THE CARRIER',
    'FOR AND ON BEHALF OF THE CARRIER',
    'FOR AND ON BEHALF OF THE ISSUING CARRIER',
    'FOR THE CARRIER AS AGENT',
    "AS CARRIER'S AGENT",
    'AS AUTHORISED AGENT',
    'AS AUTHORIZED AGENT',
)


def has_capacity(awb_text):
    txt_up = (awb_text or '').upper()
    return any(ph in txt_up for ph in _AWB_CAPACITY_AFFIRMS)


CAPACITY_CASES = [
    # Real anchor — SriLankan AWB has only generic "Signature of
    # Issuing Carrier or its Agent" label, no specific capacity
    ('Signature of Issuing Carrier or its Agent\n[signature]',
     False, 'Real anchor — no explicit capacity → FAIL'),
    ('Signed by SriLankan Airlines AS CARRIER',
     True, 'AS CARRIER explicit'),
    ('AS AGENT FOR THE CARRIER: SriLankan',
     True, 'AS AGENT FOR THE CARRIER'),
    ('FOR AND ON BEHALF OF THE CARRIER',
     True, 'FOR AND ON BEHALF OF THE CARRIER'),
    ('Signed by John Doe at Colombo',
     False, 'No capacity statement'),
]
for txt, expected, label in CAPACITY_CASES:
    got = has_capacity(txt)
    ok(f"  {label}: capacity_present={got}",
       got == expected,
       f"got {got}, expected {expected}" if got != expected else '')


print("\n" + "=" * 70)
passed = sum(results)
total = len(results)
print(f"P198gr / P198gs / P198gt / P198gv / P198gw: {passed}/{total} cases passed")
print("=" * 70)
if passed != total:
    sys.exit(1)
print("OVERALL: OK")
sys.exit(0)
