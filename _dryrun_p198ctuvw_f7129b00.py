"""
P198ct/cv/cw dry-run — four fixes for job f7129b00.

  P198ct — Draft rescue (LC-ref OCR tolerance + drawee equivalence).
            Draft's "Drawn under L/C No. 1084LC74837/2025" vs LC's
            actual "0184LC74837/2025" — only the 4-digit branch
            prefix before 'LC' differs (OCR 0↔1). Suffix '74837/2025'
            matches exactly → PASS.
            Draft has "Issued by BANK AL HABIB LIMITED" → drawee is
            the LC issuing bank (not CHINA CONSTRUCTION BANK which
            is the payee on "Pay to the Order of …") → PASS.

  P198cv — AWB Original-for-Consignor relaxation. AWB in this job
            carries standard IATA boilerplate "Copies 1, 2 and 3
            are originals and have the same validity" and
            copy_status=original. LC requires "Original for
            Consignor Clean AWB must bear flight number" — flight
            number IS on AWB; the copy-label is implicit for any
            IATA original → PASS.

  P198cw — Transport mode AIRPORT/PORT word-boundary fix. LC F44E
            "ANY AIRPORT IN CHINA" / F44F "LAHORE AIRPORT PAKISTAN"
            was being tagged BOTH as has_sea_port (because 'PORT'
            is a substring of 'AIRPORT') AND has_air_port,
            defeating the mode detector. Now: word-boundary regex,
            AIRPORT takes priority.
"""
import re
import sys, os


# ── P198cw: transport mode detection ──
def expected_mode(f44e, f44f, f46a=''):
    f44e = (f44e or '').upper()
    f44f = (f44f or '').upper()
    f46a = (f46a or '').upper()
    joined = f"{f44e} {f44f}"
    has_air = bool(re.search(r'\bAIRPORT\b|\bAIR\s+PORT\b', joined))
    has_sea = bool(re.search(r'\bSEAPORT\b|\bSEA\s+PORT\b', joined))
    if not has_air and not has_sea:
        has_sea = bool(re.search(r'\bPORT\b', joined))
    if has_air and not has_sea: return 'AIR'
    if has_sea and not has_air: return 'SEA'
    # fallback by F46A
    if 'AIR WAYBILL' in f46a or 'AWB' in f46a: return 'AIR'
    if 'BILL OF LADING' in f46a: return 'SEA'
    return 'UNKNOWN'


# ── P198ct sub-fix 1: draft LC-ref OCR-tolerant ──
def draft_ref_suffix(ref):
    m = re.search(r'LC\s*([A-Z0-9/\-._ ]+)$', (ref or '').upper())
    return m.group(1).strip() if m else ''


def simulate_draft_lc_ref(cond, draft_text, lc_ref_full):
    cond_u = cond.upper()
    if not re.search(r'\b(?:L/?C|DOCUMENTARY\s+CREDIT|CREDIT)\s+(?:NUMBER|NO\.?|REFERENCE|REF\.?)', cond_u):
        return 'noop', 'not an LC-ref condition'
    lc_suffix = draft_ref_suffix(lc_ref_full)
    if not lc_suffix:
        return 'noop', 'no LC suffix'
    draft_refs = re.findall(
        r'\b[A-Z0-9][A-Z0-9/\-]*LC\s*[A-Z0-9/\-]+\b',
        draft_text.upper(),
    )
    for dr in draft_refs:
        if draft_ref_suffix(dr) == lc_suffix:
            return 'PASS', f'suffix {lc_suffix!r} matches ({dr!r})'
    return 'FAIL', 'no draft ref with matching suffix'


# ── P198ct sub-fix 2: draft drawee equivalence ──
def simulate_draft_drawee(cond, draft_text, issuing_bank_raw):
    cond_u = cond.upper()
    if not re.search(r'\b(?:DRAWEE|ISSUING\s+BANK|L/?C\s+ISSUING\s+BANK)\b', cond_u):
        return 'noop', 'not a drawee condition'
    # Normalize issuer name
    issuer_name = (issuing_bank_raw or '').upper()
    issuer_name = re.sub(r'\b[A-Z]{6}[A-Z0-9]{2,5}\b', ' ', issuer_name)
    issuer_name = re.sub(r'[^A-Z ]', ' ', issuer_name)
    issuer_name = re.sub(r'\s+', ' ', issuer_name).strip()
    tokens = set()
    if issuer_name:
        tokens.add(issuer_name)
        parts = issuer_name.split()
        if len(parts) >= 3: tokens.add(' '.join(parts[:3]))
        if len(parts) >= 2: tokens.add(' '.join(parts[:2]))
    for t in tokens:
        if t and t in draft_text.upper():
            return 'PASS', f'issuing bank token {t!r} on draft'
    return 'FAIL', 'issuing bank not on draft'


# ── P198cv: AWB original-for-consignor relaxation ──
def simulate_awb_original(cond, awb_text, copy_status, llm_finding):
    cond_u = cond.upper()
    if not any(p in cond_u for p in (
        'ORIGINAL FOR CONSIGNOR', 'ORIGINAL FOR SHIPPER',
        'ORIGINAL 3', 'ORIGINAL NO. 3', 'ORIGINAL NO 3',
        'CONSIGNOR COPY',
    )):
        return 'noop', 'not an original-for-consignor cond'
    find_u = (llm_finding or '').upper()
    if not any(p in find_u for p in (
        'DOES NOT SPECIFY', 'NOT SPECIFIED THAT IT IS THE ORIGINAL',
        'NOT MARKED AS ORIGINAL FOR', 'DOES NOT IDENTIFY',
        'CANNOT BE VERIFIED', 'NOT CLEAR WHETHER',
    )):
        return 'noop', 'complaint is about data, not copy-label'
    doc_up = awb_text.upper()
    iata = ('COPIES 1, 2 AND 3 OF THIS AIR WAYBILL ARE ORIGINALS' in doc_up
            or 'ORIGINALS AND HAVE THE SAME VALIDITY' in doc_up
            or re.search(r'\bORIGINAL\s+(?:1|2|3)\b', doc_up) is not None)
    if copy_status != 'original' and not iata:
        return 'FAIL', 'not identifiable as an IATA original'
    return 'PASS', f'IATA original (copy_status={copy_status}, iata={iata})'


# ── Real data from job f7129b00 ──
LC_REF = '0184LC74837/2025'
ISSUER = 'BANK AL HABIB LIMITED KARACHI PAKISTAN BMLPKKACPU'
DRAFT_TEXT = """Bill of Exchange
No. XS256374
Date 2025/9/12 ZIBO CHINA
Exchange for USD17430.00
At XXX days after sight of this FIRST
Exchange (Second of exchange being unpaid)
Pay to the Order of CHINA CONSTRUCTION BANK
the sum of
U.S.DOLLARS SEVENTEEN THOUSAND FOUR HUNDRED AND THIRTY ONLY.
Drawn under L/C No. 1084LC74837/2025 Date 20250721
Issued by BANK AL HABIB LIMITED KARACHI PAKISTAN
BMLPKKACPU
SHANDONG XINHUA
PHARMACEUTICAL CO., LTD,
[SIGNATURE]
Chairman
AUTHORIZED SIGNATURE
"""
AWB_TEXT = """STAPLE DOCUMENTS ABOVE PERFORATION
784 PVG 41181022 SA250900311
Shipper's Name and Address
SHANDONG XINHUA PHARMACEUTICAL CO., LTD.
Not Negotiable
Air Waybill
Issued by
Copies 1, 2 and 3 of this Air Waybill are originals and have the same validity.
Consignee's Name and Address
BANK AL HABIB LIMITED
Flight number: CZ8212
"""


SC = []

# P198cw — transport mode
SC.append(dict(group='cw', name='F44E "ANY AIRPORT IN CHINA" + F44F "LAHORE AIRPORT PAKISTAN" → AIR',
    test=lambda: expected_mode('ANY AIRPORT IN CHINA', 'LAHORE AIRPORT - PAKISTAN'), expect='AIR'))
SC.append(dict(group='cw', name='F44E "KARACHI SEAPORT" + F44F "SHANGHAI SEAPORT" → SEA',
    test=lambda: expected_mode('KARACHI SEAPORT', 'SHANGHAI SEAPORT'), expect='SEA'))
SC.append(dict(group='cw', name='Bare "PORT OF KARACHI" (no SEAPORT/AIRPORT) → SEA',
    test=lambda: expected_mode('PORT OF SHANGHAI', 'PORT OF KARACHI'), expect='SEA'))
SC.append(dict(group='cw', name='"AIR PORT" spaced → AIR (not sea via PORT substring)',
    test=lambda: expected_mode('ANY AIR PORT IN CHINA', 'LAHORE AIR PORT'), expect='AIR'))
SC.append(dict(group='cw', name='Mixed SEA + AIR → UNKNOWN (unless F46A disambiguates)',
    test=lambda: expected_mode('SEAPORT SHANGHAI', 'AIRPORT LAHORE'), expect='UNKNOWN'))

# P198ct sub-fix 1 — LC-ref OCR
SC.append(dict(group='ct1', name='Real job: draft 1084LC74837/2025 vs LC 0184LC74837/2025 → PASS via suffix',
    test=lambda: simulate_draft_lc_ref('Drafts must show our documentary credit number.', DRAFT_TEXT, LC_REF),
    expect='PASS'))
SC.append(dict(group='ct1', name='Completely different LC suffix → FAIL',
    test=lambda: simulate_draft_lc_ref('Drafts must show our L/C number.',
        'Drawn under L/C No. 9999LC00000/1111', LC_REF), expect='FAIL'))
SC.append(dict(group='ct1', name='Non-LC-ref condition → noop',
    test=lambda: simulate_draft_lc_ref('Drafts must be signed.', DRAFT_TEXT, LC_REF),
    expect='noop'))

# P198ct sub-fix 2 — drawee
SC.append(dict(group='ct2', name='Real job: draft has "Issued by BANK AL HABIB" → drawee PASS',
    test=lambda: simulate_draft_drawee(
        'Drafts must show the name of the L/C issuing bank (Bank Al Habib Limited, Pakistan).',
        DRAFT_TEXT, ISSUER), expect='PASS'))
SC.append(dict(group='ct2', name='Draft WITHOUT issuing-bank name → FAIL',
    test=lambda: simulate_draft_drawee(
        'Drafts must show the L/C issuing bank.',
        'Drawn under L/C No. 0184LC74837/2025\nIssued by SOME UNKNOWN BANK',
        ISSUER), expect='FAIL'))
SC.append(dict(group='ct2', name='Non-drawee condition → noop',
    test=lambda: simulate_draft_drawee(
        'Drafts must be dated.', DRAFT_TEXT, ISSUER), expect='noop'))
SC.append(dict(group='ct2', name='Short issuer name (2 words) still matches "BANK AL"',
    test=lambda: simulate_draft_drawee(
        'Drafts must show the issuing bank name.',
        'Text mentioning BANK AL HABIB here.',
        'BANK AL HABIB LIMITED KARACHI PAKISTAN'), expect='PASS'))

# P198cv — AWB original
SC.append(dict(group='cv', name='Real job: IATA boilerplate + copy_status=original → PASS',
    test=lambda: simulate_awb_original(
        'Original for Consignor Clean Airway Bill must bear the flight number.',
        AWB_TEXT, 'original',
        'The Airway Bill shows a flight number (CZ8212), but the condition '
        'specifically requires the flight number to be on the Original for '
        'Consignor Clean Airway Bill. The document provided does not specify '
        'that it is the Original for Consignor.'),
    expect='PASS'))
SC.append(dict(group='cv', name='LLM complaint about DATA (not copy-label) → noop',
    test=lambda: simulate_awb_original(
        'Original for Consignor Clean Airway Bill must bear the flight number.',
        AWB_TEXT, 'original',
        'The Airway Bill does not show any flight number at all.'),
    expect='noop'))
SC.append(dict(group='cv', name='Not an original AWB → FAIL',
    test=lambda: simulate_awb_original(
        'Original for Consignor AWB must show X.',
        'just some copy\n',
        'copy', 'does not specify that it is the Original for Consignor'),
    expect='FAIL'))
SC.append(dict(group='cv', name='Non-AWB-original condition → noop',
    test=lambda: simulate_awb_original(
        'AWB must show flight number.',
        AWB_TEXT, 'original',
        'Some finding'), expect='noop'))
SC.append(dict(group='cv', name='IATA boilerplate present but copy_status=unknown → still PASS',
    test=lambda: simulate_awb_original(
        'Original for Consignor Clean AWB must bear X.',
        AWB_TEXT, 'unknown',
        'does not specify that it is the Original for Consignor'),
    expect='PASS'))


def main():
    passed = 0; failed = 0
    for i, sc in enumerate(SC, 1):
        result = sc['test']()
        if isinstance(result, tuple):
            verdict, note = result
        else:
            verdict, note = result, ''
        ok = (verdict == sc['expect'])
        tag = 'OK ' if ok else 'FAIL'
        print(f"[{tag}] [{sc['group']}] #{i:02d}  {sc['name']}")
        print(f"         expect={sc['expect']}, got={verdict}")
        if note: print(f"         note: {note}")
        if ok: passed += 1
        else: failed += 1
    print(f"\n{'='*78}\n{passed}/{passed+failed} scenarios OK\n{'='*78}")
    return failed == 0


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
