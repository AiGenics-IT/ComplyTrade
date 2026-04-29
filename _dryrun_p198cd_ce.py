"""P198cd/ce dry-run against the actual job data + synthetic scenarios.

P198cd — Policy / Cover Note / Open Policy cross-label matching
P198ce — Documentary Remittance: stamp-first; labeled presentation/
         received date; textual "presented within validity" assertion;
         REVIEW (never FAIL) when only an unlabelled bare date is shown.
"""
import json
import re
import sys

sys.path.insert(0, '.')
from steps.step14_implicit import _extract_received_stamp_date, _parse_date
from steps.step14_verification import _normalize_id


# ── P198cd: cross-label policy matching (offline simulation) ──
def simulate_p198cd(cond, doc_text, references_found=None):
    """Return the verdict the P198cd rescue would emit given LLM said FAIL."""
    cond_u = cond.upper()
    label_re = re.compile(
        r'(?:OPEN\s+)?(?:INSURANCE\s+|MARINE\s+)?'
        r'(?:POLICY|COVER\s+NOTE)\s+NO\.?\s*'
        r'([A-Z0-9][A-Z0-9/\-._]{4,}[A-Z0-9])',
        flags=re.IGNORECASE,
    )
    m = label_re.search(cond_u)
    if not m:
        return 'FAIL (P198cd did not fire — no policy needle)'
    ref_num = m.group(1).strip()
    ref_norm = _normalize_id(ref_num)
    if len(ref_norm) < 6 or sum(1 for c in ref_norm if c.isdigit()) < 3:
        return 'FAIL (needle too short / not enough digits)'
    doc_up = doc_text.upper()
    doc_norm = _normalize_id(doc_up)
    found_in_doc = (
        ref_num in doc_up
        or ref_num.replace('-', '') in doc_up.replace('-', '')
        or (len(ref_norm) >= 4 and ref_norm in doc_norm)
    )
    found_in_refs = False
    hit = ''
    for item in (references_found or []):
        v = _normalize_id(item.get('value', ''))
        if v and (v == ref_norm or ref_norm in v or v in ref_norm):
            found_in_refs = True
            hit = f"refs_found[{item.get('role','?')}]={item.get('value')}"
            break
    if found_in_doc or found_in_refs:
        return f'PASS (found_in_doc={found_in_doc} found_in_refs={found_in_refs} {hit})'
    return 'FAIL (neither doc text nor references_found carry the number)'


cd_cases = [
    # (label, cond, doc_text, refs_found, expected_starts_with)
    ('Cover Note label in doc, Policy No in cond (real e07ce444 case)',
     "Shipment Advice must reference Policy No. 11/0000118/1024/0-0.",
     "COVER NOTE NO. 11/0000118/1024/0-0 dated 10 March 2026\nShipment to Pakistan",
     [],
     'PASS'),
    ('Open Policy in doc, Policy in cond',
     "Shipment Advice must reference Policy No. 2023008MIPDO00453.",
     "OPEN POLICY NO. 2023008MIPDO00453 / Applicant: SABIC",
     [],
     'PASS'),
    ('OCR variant: Policy in cond has digit 0, doc has letter O',
     "Shipment Advice must reference Policy No. 2023008MIPD000453.",
     "OPEN POLICY NO. 2023008MIPDO00453",
     [],
     'PASS'),
    ('References_found has cover_note_reference (role differs)',
     "Shipment Advice must reference Policy No. 11/0000118/1024/0-0.",
     "Issued as per cover note",
     [{'role': 'cover_note_reference', 'value': '11/0000118/1024/0-0'}],
     'PASS'),
    ('References_found has open_policy_reference (role differs)',
     "Shipment Advice must reference Policy No. 2023008MIPDO00453.",
     "Generic shipment advice text",
     [{'role': 'open_policy_reference', 'value': '2023008MIPDO00453'}],
     'PASS'),
    ('Different number entirely on doc',
     "Shipment Advice must reference Policy No. 11/0000118/1024/0-0.",
     "COVER NOTE NO. 99/9999999/9999/9-9",
     [],
     'FAIL'),
    ('Policy number missing from doc + refs',
     "Shipment Advice must reference Policy No. 11/0000118/1024/0-0.",
     "Shipment Advice addressed to insurer",
     [],
     'FAIL'),
    ('INSURANCE POLICY NO label variant',
     "Shipment Advice must reference Policy No. ABC-123-456.",
     "INSURANCE POLICY NO. ABC-123-456 issued by AIG",
     [],
     'PASS'),
    ('Marine Policy No label variant',
     "Shipment Advice must reference Policy No. XYZ/9876.",
     "MARINE POLICY NO. XYZ/9876",
     [],
     'PASS'),
]


print("=" * 78)
print("P198cd — cross-label policy / cover-note / open-policy matching")
print("=" * 78)
cd_pass = 0
for label, cond, doc, refs, expected in cd_cases:
    got = simulate_p198cd(cond, doc, refs)
    ok = 'OK' if got.startswith(expected) else 'FAIL'
    if ok == 'OK':
        cd_pass += 1
    print(f'  [{ok}] {label}')
    print(f'       got: {got}')
print(f'  {cd_pass}/{len(cd_cases)} cases correct')


# ── P198ce: labeled date + presented-on-time textual assertion ──
class _MockPkt(dict):
    """Helper to build synthetic packets for the date-extractor tests."""
    pass


def _build_pkt(stamps=None, text=''):
    return {
        'document_type': 'Documentary Remittance',
        'stamps': stamps or [],
        'refined_text': text,
        'cleaned_text': text,
        'raw_text': text,
    }


print()
print("=" * 78)
print("P198ce — presentation-date extraction (stamp / labeled / bare)")
print("=" * 78)

ce_cases = [
    ('RECEIVED stamp 21 FEB 2025 parseable → that date',
     _build_pkt(stamps=[{'text': '21 FEB 2025', 'type': 'rubber_stamp'}]),
     '2025-02-21'),
    ('RECEIVED stamp OCR-mangled "71 CCR 2025" → None (no fallback)',
     _build_pkt(stamps=[{'text': '71 CCR 2025', 'type': 'rubber_stamp'}]),
     None),
    ('Labeled "Presentation Date: 12/03/2026" in doc text',
     _build_pkt(text='Covering schedule\nPresentation Date: 12/03/2026\n'),
     '2026-03-12'),
    ('Labeled "Received Date: 15.02.2026"',
     _build_pkt(text='Received Date: 15.02.2026\nAmount USD 100,000'),
     '2026-02-15'),
    ('Labeled "Presented on 18 Feb 2026"',
     _build_pkt(text='We hereby enclose docs. Presented on 18 Feb 2026.'),
     '2026-02-18'),
    ('UNLABELED bare "DATE: 18/02/25" → None (NOT used as presentation)',
     _build_pkt(text='Covering Schedule\nDATE: 18/02/25\nAmount USD 100,000'),
     None),
    ('Stamp PARSEABLE + labeled date also present → stamp wins',
     _build_pkt(
         stamps=[{'text': 'RECEIVED 19 SEP 2025', 'type': 'rubber_stamp'}],
         text='Presentation Date: 01/01/2030',  # bogus but should be ignored
     ),
     '2025-09-19'),
]

ce_pass = 0
for label, pkt, expected_iso in ce_cases:
    result = _extract_received_stamp_date(pkt)
    if result is None:
        got_iso = None
    else:
        got_iso = result[0].strftime('%Y-%m-%d')
    ok = 'OK' if got_iso == expected_iso else 'FAIL'
    if ok == 'OK':
        ce_pass += 1
    print(f'  [{ok}] {label}')
    print(f'       got={got_iso!r} (expected {expected_iso!r})')
print(f'  {ce_pass}/{len(ce_cases)} cases correct')


# ── P198ce: "presented within validity" text assertion triggers PASS ──
print()
print("=" * 78)
print("P198ce — 'presented within validity' textual assertion → PASS")
print("=" * 78)

_presentation_ok_re = [
    r'DOCUMENTS?\s+(?:HAVE\s+BEEN\s+)?PRESENTED\s+(?:ON\s+TIME|'
    r'WITHIN\s+(?:THE\s+)?(?:L/?C\s+)?(?:VALIDITY|EXPIRY|PERIOD)|'
    r'PRIOR\s+TO\s+EXPIRY|BEFORE\s+(?:THE\s+)?EXPIRY|'
    r'WITHIN\s+(?:THE\s+)?PRESENTATION\s+PERIOD)',
    r'PRESENTATION\s+(?:IS\s+|WAS\s+)?MADE\s+WITHIN\s+(?:L/?C\s+)?'
    r'(?:VALIDITY|EXPIRY\s+DATE)',
    r'DOCUMENTS?\s+ARE\s+(?:BEING\s+)?PRESENTED\s+WITHIN\s+VALIDITY',
    r'WITHIN\s+L/?C\s+VALIDITY\s+PERIOD',
    r'DOCUMENTS?\s+NEGOTIATED\s+WITHIN\s+(?:THE\s+)?VALIDITY',
]


def assertion_triggers(text):
    u = text.upper()
    return any(re.search(p, u, re.IGNORECASE) for p in _presentation_ok_re)


assert_cases = [
    ('"Documents presented within L/C validity"', True),
    ('"Documents have been presented within LC validity period"', True),
    ('"Documents presented on time"', True),
    ('"Documents presented prior to expiry"', True),
    ('"Presentation was made within LC validity"', True),
    ('"Documents are being presented within validity"', True),
    ('"Documents negotiated within the validity"', True),
    ('"Documents presented before the expiry"', True),
    ('Generic: "Covering schedule, amount USD 100,000"', False),
    ('No assertion: "Please process these documents"', False),
    ('Different: "Documents presented at UBL counter"', False),
]
a_pass = 0
for label, expected in assert_cases:
    text_only = label.split(":", 1)[-1].strip(' "')
    got = assertion_triggers(text_only)
    ok = 'OK' if got == expected else 'FAIL'
    if ok == 'OK':
        a_pass += 1
    print(f'  [{ok}] {label:60} → {got} (expected {expected})')
print(f'  {a_pass}/{len(assert_cases)} assertion cases correct')


print()
print("=" * 78)
print(f"P198cd: {cd_pass}/{len(cd_cases)}  |  "
      f"P198ce extractor: {ce_pass}/{len(ce_cases)}  |  "
      f"P198ce assertion: {a_pass}/{len(assert_cases)}")
print("=" * 78)
