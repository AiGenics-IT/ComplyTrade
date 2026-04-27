"""
P198cz dry-run — Shipping Company Certificate strict-content guard.

LC clauses on SCCs commonly demand that the certificate STATE
specific content (Institute Classification Clause coverage,
Pakistani Maritime Rules & Port Regulations, ETA at destination,
etc.). The LLM frequently PASSes these without the literal text
on the certificate by hallucinating from issuer identity ("PIL is
a major shipping line, so it must operate in accordance with
Pakistani Maritime Rules") or confusing departure/sailing dates
with ETA.

The strict-content guard verifies the literal phrase / data point
is on the SCC text. If not, FAIL.
"""
import re, sys


_SCC_REQUIREMENTS = [
    (
        re.compile(
            r'\b(?:PAKISTAN(?:I)?\s+MARITIME\s+RULES?|'
            r'MARITIME\s+RULES?\s+AND\s+PORT\s+REGULATIONS?|'
            r'PORT\s+REGULATIONS?)\b', re.IGNORECASE),
        re.compile(
            r'\b(?:PAKISTAN(?:I)?\s+MARITIME\s+RULES?|'
            r'OPERATING\s+IN\s+ACCORDANCE\s+WITH\s+PAKISTAN|'
            r'MARITIME\s+RULES?\s+AND\s+PORT\s+REGULATIONS?)\b',
            re.IGNORECASE),
        'Pakistani Maritime Rules / Port Regulations',
    ),
    (
        re.compile(r'\bINSTITUTE\s+CLASSIFICATION\s+CLAUSE\b', re.IGNORECASE),
        re.compile(
            r'\bINSTITUTE\s+CLASSIFICATION\s+CLAUSE\b|'
            r'\bICC\s*\(?\s*INSTITUTE\b', re.IGNORECASE),
        'Institute Classification Clause',
    ),
    (
        re.compile(
            r'\b(?:APPROXIMATE\s+DATE\s+OF\s+ARRIVAL|'
            r'ESTIMATED\s+(?:TIME|DATE)\s+OF\s+ARRIVAL|'
            r'\bETA\b|EXPECTED\s+ARRIVAL|'
            r'DATE\s+OF\s+ARRIVAL\s+(?:OF\s+)?(?:THE\s+)?VESSEL|'
            r'ARRIVAL\s+(?:OF\s+)?(?:THE\s+)?VESSEL\s+'
            r'AT\s+(?:THE\s+)?(?:PORT\s+OF\s+)?DESTINATION)\b',
            re.IGNORECASE),
        re.compile(
            r'\b(?:ETA|ESTIMATED\s+(?:TIME|DATE)\s+OF\s+ARRIVAL|'
            r'EXPECTED\s+ARRIVAL|APPROXIMATE\s+DATE\s+OF\s+ARRIVAL|'
            r'DATE\s+OF\s+ARRIVAL\s+AT|ARRIVAL\s+AT\s+'
            r'(?:THE\s+)?(?:PORT\s+OF\s+)?DESTINATION|'
            r'ARRIVING\s+AT\s+DESTINATION|'
            r'EXPECTED\s+(?:TO\s+)?ARRIVE)\b', re.IGNORECASE),
        'Approximate date of arrival (ETA)',
    ),
]


def simulate(cond, doc_text, current='PASS', doc_type='Shipping Company Certificate'):
    if current != 'PASS':
        return current, 'not PASS'
    if 'shipping company' not in doc_type.lower():
        return current, 'not SCC'
    doc_up = doc_text.upper()
    for cond_re, doc_re, label in _SCC_REQUIREMENTS:
        if not cond_re.search(cond):
            continue
        if doc_re.search(doc_up):
            return 'PASS', f'literal evidence on doc for {label}'
        return 'FAIL', f'missing {label} on doc'
    return current, 'no SCC requirement detected'


# Real document text from job 73be98d9
SCC_DOC = """PIL
PACIFIC INTERNATIONAL LINES (PTE) LTD
PIL VIETNAM CO., LTD - HANOI BRANCH
SHIPPING CERTIFICATE
Dated: 1st February 2025
SHIPPER'S DECLARATION:
(B/L: HPH500022000)
Shipper: BRANCH OF VINATEX-NAM DINH SPINNING FACTORY
Port of loading: HAIPHONG PORT, VIETNAM
Port of discharge: KARACHI PORT, PAKISTAN
Vessel: KOTANEKAD0204S
Sailing on: 01FEB 2025
DOCUMENTARY CREDIT NUMBER 0001LC55282/2025, DATE 03.01.2025 AND NAME
OF L/C ISSUING BANK (BANK AL HABIB LTD., PAKISTAN)
CARRIER'S DECLARATION:
TO WHOM IT MAY CONCERN
WE, PACIFIC INTERNATIONAL LINES (PTE) LTD WOULD LIKE TO CERTIFY THAT:
14 DAYS FREE TIME DETENTION ALLOWED AT POD
FOR AND ON BEHALF OF THE CARRIER
PACIFIC INTERNATIONAL LINES (PTE) LTD AS AGENT
"""


SC = []

# Real R0016 — Pakistani Maritime Rules NOT in doc → FAIL
SC.append(dict(name='Real R0016: Pakistani Maritime Rules absent → FAIL',
    cond='Certificate from shipping company or their authorized agents must state '
         'that the carrying vessel is owned by companies operating in accordance '
         'with Pakistani Maritime Rules and Port Regulations.',
    doc=SCC_DOC, expect='FAIL'))

# Real R0017 — ETA absent (only sailing date) → FAIL
SC.append(dict(name='Real R0017: only sailing date, no ETA → FAIL',
    cond='Certificate from shipping company or their authorized agents must show '
         'the approximate date of arrival of the vessel at the port of destination.',
    doc=SCC_DOC, expect='FAIL'))

# R0015 — Institute Classification Clause not in doc → FAIL
SC.append(dict(name='R0015: Institute Classification Clause absent → FAIL',
    cond='Certificate from shipping company or their authorized agents must state '
         'that the carrying vessel is covered under Institute Classification Clause.',
    doc=SCC_DOC, expect='FAIL'))

# Hypothetical: doc has the literal phrase → PASS
SC.append(dict(name='SCC literally states "Pakistani Maritime Rules" → PASS',
    cond='Certificate must state vessel operates in accordance with Pakistani Maritime Rules and Port Regulations.',
    doc='SHIPPING CERTIFICATE\n...vessel is owned by companies operating in accordance '
        'with Pakistani Maritime Rules and Port Regulations...',
    expect='PASS'))

# Hypothetical: doc has Institute Classification Clause → PASS
SC.append(dict(name='SCC literally states "Institute Classification Clause" → PASS',
    cond='Certificate must cover vessel under Institute Classification Clause.',
    doc='SHIPPING CERTIFICATE\nVessel is covered under Institute Classification Clause as required.',
    expect='PASS'))

# Hypothetical: doc has ETA → PASS
SC.append(dict(name='SCC has ETA wording → PASS',
    cond='Certificate must show approximate date of arrival at port of destination.',
    doc='SHIPPING CERTIFICATE\n...ETA Karachi: 15 February 2025\nVessel: KOTA NEKAD',
    expect='PASS'))

# Doc has "Estimated Date of Arrival" → PASS
SC.append(dict(name='Estimated Date of Arrival wording → PASS',
    cond='Certificate must show the approximate date of arrival of the vessel at port of destination.',
    doc='SHIPPING CERTIFICATE\nEstimated Date of Arrival at Karachi Port: 15 Feb 2025',
    expect='PASS'))

# Doc has "Sailing on:" only → FAIL
SC.append(dict(name='Sailing date only → FAIL',
    cond='Certificate must show the approximate date of arrival of the vessel at the port of destination.',
    doc='Sailing on: 01 Feb 2025\nVessel: KOTA NEKAD\nPort of loading: Haiphong',
    expect='FAIL'))

# No SCC requirement in condition → no override
SC.append(dict(name='Non-SCC-requirement condition → no override',
    cond='Certificate must show name of vessel.',
    doc=SCC_DOC, expect='PASS'))

# Already FAIL → no change
SC.append(dict(name='Already FAIL → no change',
    cond='Certificate must show ETA at destination.',
    doc='No relevant content', expect='FAIL', current='FAIL'))


def main():
    passed = 0; failed = 0
    for i, sc in enumerate(SC, 1):
        verdict, note = simulate(sc['cond'], sc['doc'], sc.get('current','PASS'))
        ok = (verdict == sc['expect'])
        tag = 'OK ' if ok else 'FAIL'
        print(f"[{tag}] #{i:02d}  {sc['name']}")
        print(f"         expect={sc['expect']}, got={verdict}")
        print(f"         note: {note}")
        if ok: passed += 1
        else: failed += 1
    print(f"\n{'='*78}\n{passed}/{passed+failed} P198cz SCC-strict scenarios OK\n{'='*78}")
    return failed == 0


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
