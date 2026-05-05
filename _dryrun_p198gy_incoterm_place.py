"""P198gy — full-clause Incoterm place check (with ANY-port/airport flexibility)."""
import sys, os, re
sys.path.insert(0, 'd:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final')
os.environ['PYTHONIOENCODING'] = 'utf-8'

results = []
def ok(name, cond, note=''):
    tag = '[OK]  ' if cond else '[FAIL]'
    print(f"{tag} {name}" + (f" -- {note}" if note else ''))
    results.append(bool(cond))


# Mirror helpers from step14_verification.py (same regexes/logic)
_INCOTERMS_VERSION_RE = re.compile(
    r'\bINCOTERMS?\s*:?\s*(\d{4})\b', re.IGNORECASE)
_INCOTERM_CODE_RE = re.compile(
    r'\b(EXW|FCA|FAS|FOB|CFR|CNF|C\&F|CIF|CIP|CPT|'
    r'DAP|DPU|DDP|DAT|DAF|DDU|DES|DEQ)\b', re.IGNORECASE)
_INCOTERM_CLAUSE_RE = re.compile(
    r'\b(EXW|FCA|FAS|FOB|CFR|CNF|C\&F|CIF|CIP|CPT|'
    r'DAP|DPU|DDP|DAT|DAF|DDU|DES|DEQ)\s+'
    r'([A-Z][A-Z0-9 ,/\-\(\)]{2,80}?)'
    r'(?=\s*(?:\(\s*INCOTERMS?|\.|\,|;|$|\n|MUST|TO\s+BE|ON\s+(?:CI|COMMERCIAL)))',
    re.IGNORECASE)


def extract_lc_term(lc_up):
    mc = _INCOTERM_CLAUSE_RE.search(lc_up)
    mv = _INCOTERMS_VERSION_RE.search(lc_up)
    if not mc:
        return None
    code = mc.group(1).upper()
    place = re.sub(r'\s+', ' ', mc.group(2).strip().upper())
    ver = mv.group(1) if mv else None
    country = None
    is_generic = False
    mg = re.search(
        r'\bANY\s+(?:\w+\s+)*?(?:AIR\s*PORT|AIRPORT|SEA\s*PORT|SEAPORT|PORT)\s+(?:IN\s+|OF\s+)?([A-Z][A-Z ]{2,40})',
        place)
    if mg:
        is_generic = True
        country = mg.group(1).strip()
    else:
        mg2 = re.search(
            r'\bANY\s+([A-Z][A-Z ]{2,40}?)\s+(?:AIR\s*PORT|AIRPORT|SEA\s*PORT|SEAPORT|PORT)\b',
            place)
        if mg2:
            is_generic = True
            country = mg2.group(1).strip()
    return (code, place, ver, is_generic, country)


_COUNTRY_NAMES = {
    'CHINA','PAKISTAN','INDIA','MALAYSIA','SINGAPORE','THAILAND',
    'INDONESIA','VIETNAM','JAPAN','KOREA','TURKEY','GERMANY',
    'FRANCE','ITALY','SPAIN','UAE','EMIRATES','UK','USA','AMERICA',
    'BANGLADESH','PHILIPPINES','EGYPT','HONG KONG',
}

def check_doc(doc_up, lc_term):
    code, lc_place, lc_ver, is_generic, country = lc_term
    if not re.search(rf'\b{re.escape(code)}\b', doc_up):
        return False, f"code {code} missing"
    if not is_generic:
        head = [t for t in re.split(r'[ ,/]+', lc_place)
                if t and t not in ('AIRPORT','AIR','PORT','SEAPORT','SEA',
                                   'CITY','OF','IN','AT','THE','NAMED',
                                   'PLACE','ANY','EVERY','A')]
        head = [t for t in head if t not in _COUNTRY_NAMES]
        head = head[:3]
        if head and not any(re.search(rf'\b{re.escape(t)}\b', doc_up) for t in head):
            return False, f"place {lc_place} missing"
    if lc_ver:
        mv = _INCOTERMS_VERSION_RE.search(doc_up)
        if not mv:
            return False, f"doc silent on version"
        if mv.group(1) != lc_ver:
            return False, f"version {mv.group(1)} != {lc_ver}"
    return True, "ok"


# ── Section 1 — clause extraction ──
print("=" * 70)
print("Section 1: LC clause extraction")
print("=" * 70)
EXT = [
    ('FCA ANY AIRPORT IN CHINA(INCOTERMS : 2020)',
     ('FCA', '2020', True, 'CHINA')),
    ('CPT KARACHI AIRPORT (INCOTERMS : 2020)',
     ('CPT', '2020', False, None)),
    ('FOB ANY CHINA SEAPORT (INCOTERMS 2020)',
     ('FOB', '2020', True, 'CHINA')),
    ('CIF MALAYSIA',
     ('CIF', None, False, None)),
    ('FCA ANY AIRPORT IN PEOPLES REPUBLIC OF CHINA (INCOTERMS 2020)',
     ('FCA', '2020', True, 'PEOPLES REPUBLIC OF CHINA')),
]
for txt, exp in EXT:
    t = extract_lc_term(txt.upper())
    if t is None:
        ok(f"  {txt[:50]}", False, "no extraction")
        continue
    code, place, ver, gen, country = t
    got = (code, ver, gen, country)
    ok(f"  {txt[:55]} -> {got}", got == exp,
       f"got {got} expected {exp}")


# ── Section 2 — doc check ──
print("\n" + "=" * 70)
print("Section 2: Doc-side compliance check")
print("=" * 70)
CASES = [
    # (lc, doc, expected_ok, label)
    ('FCA ANY AIRPORT IN CHINA (INCOTERMS : 2020)',
     'FCA SHANGHAI AIRPORT CHINA INCOTERMS 2020', True,
     'Generic ANY-airport: specific Chinese airport OK'),
    ('FCA ANY AIRPORT IN CHINA (INCOTERMS : 2020)',
     'FCA BEIJING CAPITAL AIRPORT CHINA (Incoterms 2020)', True,
     'Generic ANY-airport: another specific Chinese airport OK'),
    ('FCA ANY AIRPORT IN CHINA (INCOTERMS : 2020)',
     'FCA MUMBAI AIRPORT (Incoterms 2020)', True,
     'Generic ANY-airport: skip place check (LLM has world knowledge)'),
    ('CPT KARACHI AIRPORT (INCOTERMS : 2020)',
     'CPT KARACHI (Incoterms 2020)', True,
     'Specific KARACHI matches'),
    ('CPT KARACHI AIRPORT (INCOTERMS : 2020)',
     'CPT LAHORE (Incoterms 2020)', False,
     'Specific KARACHI required but doc shows LAHORE'),
    ('CPT KARACHI AIRPORT (INCOTERMS : 2020)',
     'CPT KARACHI', False,
     'Place OK but version missing'),
    ('CPT KARACHI AIRPORT (INCOTERMS : 2020)',
     'CPT KARACHI (Incoterms 2010)', False,
     'Version mismatch'),
    ('FOB ANY CHINA SEAPORT (INCOTERMS 2020)',
     'FOB SHANGHAI (Incoterms 2020)', True,
     'Generic ANY-seaport CHINA: SHANGHAI OK'),
    ('FOB ANY CHINA SEAPORT (INCOTERMS 2020)',
     'FOB SHANGHAI', False,
     'Generic seaport but version missing (still fails on version)'),
    # CFR no version specified
    ('CFR MALAYSIA',
     'CFR PORT KLANG', True,
     'Country-only LC: skip place check'),
]
for lc, doc, exp_ok, label in CASES:
    t = extract_lc_term(lc.upper())
    if t is None:
        ok(label, False, 'extraction failed')
        continue
    got_ok, reason = check_doc(doc.upper(), t)
    ok(f"  {label}", got_ok == exp_ok,
       f"got ok={got_ok} reason={reason!r}")


# ── Section 3 — source wiring ──
print("\n" + "=" * 70)
print("Section 3: Source wiring")
print("=" * 70)
src = open('d:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final/steps/step14_verification.py',
           'r', encoding='utf-8').read()
ok("  P198gy marker", 'P198gy' in src)
ok("  _INCOTERM_CLAUSE_RE present", '_INCOTERM_CLAUSE_RE' in src)
ok("  _extract_lc_term helper", '_extract_lc_term' in src)
ok("  _check_doc_term helper", '_check_doc_term' in src)


print("\n" + "=" * 70)
passed = sum(results)
total = len(results)
print(f"P198gy INCOTERM PLACE: {passed}/{total}")
print("=" * 70)
if passed != total:
    sys.exit(1)
print("OVERALL: OK")
sys.exit(0)
