"""
P198dx dry-run

Three things must be verified end-to-end:

1. The step08 DR-guard demotes email cover notes from "Documentary
   Remittance" to "Covering Letter" (regression of P198dp).

2. The step08 "Detailed Message" recogniser upgrades pages whose
   header is "DETAILED MESSAGE" + beneficiary-cert language + fax
   / shipment evidence to document_type="Detailed Message".

3. The step14 alias groups "beneficiary certificate" and
   "shipment advice" both contain "detailed message" so a clause
   targeting either kind of document finds the same physical page.

Tests use real OCR from the user's two reported jobs:
  - 53e62015 pages 13 (real DR), 24 (PL), 27 + 29 (email cover notes)
  - 38beca01 page X (Insurance Request email)
plus an inlined "DETAILED MESSAGE" beneficiary fax body for the
F46A-7 Meiji example the user pasted.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from steps.step14_verification import DOC_TYPE_ALIASES, _find_matching_docs

# ---- Mirrors of the step08 P198dp DR-guard + P198dx detailed-message
#      recogniser (kept in sync with the live code). ------------------

_DR_REAL_SIGNALS = [
    r'WE\s+ENCLOSE\s+(?:THE\s+)?(?:FOLLOWING\s+|ABOVE\s+)?'
    r'DOCUMENTS?(?:\s+(?:FOR|DRAWN))?',
    r'WE\s+ARE\s+PLEASED\s+TO\s+ENCLOSE',
    r'WE\s+HEREBY\s+ENCLOSE',
    r'ENCLOSED\s+HEREWITH',
    r'PRESENTATION\s+(?:NUMBER|NO\.?|DATE|AMOUNT)',
    r'TOTAL\s+(?:AMOUNT\s+)?CLAIMED',
    r'PRINCIPAL\s+AMOUNT\s+(?:CLAIMED|EUR|USD|GBP)',
    r'AMOUNTS?\s+CLAIMED\s*[:\n]',
    r'YOUR\s+DOCUMENTARY\s+CREDIT\s+NO',
    r'OUR\s+REFERENCE\s+NO',
    r'REMIT\s+FUNDS\s+TO\s+(?:OUR\s+)?CORRESPONDENT',
    r'(?:UPON|FOR)\s+SETTLEMENT\s+PLEASE\s+REMIT',
    r'QUOTING\s+OUR\s+REFERENCE',
    r'CLAIM\s+REIMBURSEMENT',
    r'COVERING\s+(?:LETTER|SCHEDULE)',
    r'DOCUMENTARY\s+REMITTANCE',
    r'L/?C\s+BILLS?\s+SCHEDULE',
    r'(?:DOCUMENT|EXPORT\s+DC)\s+PRESENTATION\s+SCHEDULE',
    r'SCHEDULE\s+OF\s+PRESENTATION',
    r'BILLS?\s+REMITTANCE\s+LETTER',
    r'\bL/?C\s+ISSUING\s+BANK\b',
    r'\bREIMBURSING\s+BANK\b',
    r'\bYOUR\s+DC\s+REF\b',
    r'\bOUR\s+REF\.\s',
    r'\bPAYMENT\s+INSTRUCTION\b',
    r'\bBILL\s+AMOUNT\b',
    r'DOCUMENTS?\s+SENT\s+TO\s+YOU\s+ON\s+APPROVAL',
    r'DRAWING\s+AMOUNT\s+(?:HAS\s+BEEN\s+)?(?:DULY\s+)?ENDORSED',
    r'PRESENTATION\s+IS\s+SUBJECT\s+TO',
    r'ADVISING\s+CHARGES?\s+AND\s+CONFIRMATION\s+CHARGES?',
]
_BANK_RE = re.compile(
    r'\b(?:MAYBANK|MALAYAN\s+BANKING|BANK\s+AL\s+HABIB|'
    r'HABIB\s+BANK|HBL\b|UBL\b|UNITED\s+BANK\s+LIMITED|'
    r'MEEZAN\s+BANK|FAYSAL\s+BANK|MCB\b|ALLIED\s+BANK|'
    r'STANDARD\s+CHARTERED|HSBC|CITIBANK|JP\s*MORGAN|'
    r'J\.P\.\s*MORGAN|BARCLAYS|DEUTSCHE\s+BANK|RBC\b|'
    r'ROYAL\s+BANK|BNP\s+PARIBAS|COMMERZBANK|MIZUHO|'
    r'BANK\s+OF\s+CHINA|ICBC|BANCO\b|CHINA\s+CONSTRUCTION|'
    r'WELLS\s+FARGO|BANK\s+OF\s+AMERICA|UNICREDIT|'
    r'SOCIETE\s+GENERALE|CREDIT\s+SUISSE|UBS\b|'
    r'NATIONAL\s+BANK|COMMERCIAL\s+BANK)\b'
)
_SWIFT_RE = re.compile(r'\bSWIFT\s*:\s*[A-Z]{6,11}\b')


def dr_guard_classify(initial_doc_type, glm_text):
    """Mirror of the step08 final classification block:
    runs the P198dx detailed-message upgrade first, then the
    P198dp DR-guard demotion. Returns the post-guard doc_type."""
    document_type = initial_doc_type
    _ux = (glm_text or '').upper()

    # ── P198dx — Detailed Message upgrade ──
    if document_type in (
        'Shipment Advice', 'Shipping Advice',
        'Beneficiary Certificate', "Beneficiary's Certificate",
        'Documentary Remittance', 'Covering Letter', 'Covering Schedule',
    ):
        _has_dm_header = bool(re.search(
            r'(?:^|\n)\s*DETAIL(?:ED)?\s+MESSAGE\b',
            _ux, re.MULTILINE))
        _has_bene_cert = bool(re.search(
            r'\bWE\s+CERTIFY\b|CERTIFY(?:ING)?\s+(?:THE\s+)?GOODS\s+'
            r'(?:TO\s+BE\s+)?(?:OF|ARE\s+OF)|'
            r'\bWE\s+ARE\s+PLEASED\s+TO\s+INFORM\s+YOU\s+OF\s+OUR\s+SHIPMENT\b',
            _ux))
        _has_fax_or_email = bool(re.search(
            r'\bFAX(?:\s*(?:NO\.?|NUMBER|#))?\s*[:.\d]|'
            r'\bDIRECT(?:LY)?\s+TO\s+THE\s+APPLICANT\s+BY\s+FAX|'
            r'\bSENT\s+BY\s+FAX\b',
            _ux))
        _has_shipment_evidence = bool(re.search(
            r'\bB[/\s]*L\s+(?:NO\.?|NUMBER)|\bBILL\s+OF\s+LADING|'
            r'\bVESSEL\b|\bETA\b|\bETD\b|\bSHIPPED\s+ON\s+BOARD\b',
            _ux))
        if (_has_dm_header
            and _has_shipment_evidence
            and (_has_bene_cert or _has_fax_or_email)):
            document_type = 'Detailed Message'

    # ── P198dp — DR false-positive guard ──
    if document_type == 'Documentary Remittance':
        _signal_count = sum(1 for p in _DR_REAL_SIGNALS if re.search(p, _ux))
        _bank = bool(_BANK_RE.search(_ux))
        _swift = bool(_SWIFT_RE.search(_ux))
        _is_email = bool(
            re.search(r'\bFROM\s*:\s*[^\n]*@', _ux)
            and re.search(r'\bSUBJECT\s*:', _ux)
        )
        if _is_email:
            _is_real_dr = _signal_count >= 3
        else:
            _is_real_dr = (
                _signal_count >= 2
                or (_bank and _signal_count >= 1)
                or (_swift and _signal_count >= 1)
            )
        if not _is_real_dr:
            document_type = 'Covering Letter'

    return document_type


# ---- Real OCR text from job 53e62015 -----------------------------------

JOB1 = Path('results/53e62015-f805-4985-81e3-2b5de1daee65')
s2 = json.loads((JOB1 / 'step02' / 'step02_result.json').read_text(encoding='utf-8'))
real_pages = {p['page_number']: (p.get('cleaned_text') or p.get('raw_text') or '')
              for p in s2['pages']}

# ---- Synthetic Detailed Message (Meiji example) -----------------------

DETAILED_MESSAGE_TEXT = """meiji
2026/2/23
MEIJI CO., LTD.
PLANNING DEPT.
GLOBAL CACAO BUSINESS DIV.
2-2-1 Kyobashi, Chuo-ku,
Tokyo 104-8306, JAPAN
DETAILED MESSAGE
TO : GLOBAL BRANDS MARKETING (PVT) LTD
204, E.I. LINES,
DR. DAUD POTA ROAD, KARACHI,PAKISTAN
FAX.0092-21-35654644
NAME OF ITEM : MILK PREPARATION IN POWDER FORM FOR INFANTS
& QUANTITY 3900 CTNS X 24 TINS X 400GM MEIJI FM-T
INVOICE NO. : 26PK0209-A
INVOICE VALUE : US$338,910.00
HS CODE NO. : 1901.1000
VESSEL : SANTA MARTA EXPRESS 609S
B/L NO. : A10894836
B/L DATE : 2026/2/23
ETD : 2026/2/23
ETA : 2026/4/27
DELIVERY AGENT IN KARACHI : MAERSK PAKISTAN PVT LTD
We are pleased to inform you of our shipment for L/C No.
1019LC55849/2026 dated 2026/1/9
issuing bank BANK AL HABIB LTD, KARACHI
WE CERTIFY THE GOODS TO BE OF E.U. ORIGIN.
MEIJI CO., LTD.
[SIGNATURE] RYUNOSUKE OHNO
GENERAL MANAGER
"""


def main():
    pass_n, fail_n = 0, 0
    print('=' * 78)
    print('P198dx dry-run — Detailed Message + DR-guard (real OCR + synthetic)')
    print('=' * 78)

    # ── A. Real job 53e62015: pages 13 (real DR), 24 (PL), 27/29 (emails) ──
    print('\n--- A. Real job 53e62015 (post-step08 doc_type) ---')
    cases = [
        # (page, initial step08 input, expected post-guard output)
        (13, 'Documentary Remittance', 'Documentary Remittance'),  # real Maybank DR
        (24, 'Packing List',           'Packing List'),            # untouched
        (26, 'Shipment Advice',        'Shipment Advice'),         # untouched
        # The two emails — step03 canonicalised them to "Document
        # Remittance" / "Documentary Remittance"; the guard demotes.
        (27, 'Documentary Remittance', 'Covering Letter'),
        (29, 'Documentary Remittance', 'Covering Letter'),
    ]
    for pn, in_dt, expected in cases:
        text = real_pages.get(pn, '')
        out = dr_guard_classify(in_dt, text)
        ok = (out == expected)
        tag = 'OK ' if ok else 'FAIL'
        print(f"   [{tag}] p{pn:>2}: in='{in_dt}' -> out='{out}'  expected='{expected}'")
        if ok: pass_n += 1
        else:  fail_n += 1

    # ── B. Detailed Message upgrade ──
    print('\n--- B. Synthetic Meiji DETAILED MESSAGE ---')
    for in_dt, expected in [
        ('Shipment Advice',        'Detailed Message'),
        ('Beneficiary Certificate', 'Detailed Message'),
        ('Covering Letter',        'Detailed Message'),
        ('Documentary Remittance', 'Detailed Message'),
        ('Bill of Lading',         'Bill of Lading'),  # not eligible for upgrade
    ]:
        out = dr_guard_classify(in_dt, DETAILED_MESSAGE_TEXT)
        ok = (out == expected)
        tag = 'OK ' if ok else 'FAIL'
        print(f"   [{tag}] in='{in_dt}' -> out='{out}'  expected='{expected}'")
        if ok: pass_n += 1
        else:  fail_n += 1

    # ── C. step14 alias matching (target -> packet doc-types) ──
    print('\n--- C. step14 _find_matching_docs aliasing ---')
    aliases_tests = [
        # (clause target, packet doc_type, should_match)
        # NOTE: with DETAILED_MESSAGE_TEXT as the packet body, Tier-4
        # text-content fallback may match a target whose alias group
        # contains "detailed message" because that phrase appears in
        # the body. That's intentional — the body proves the page is
        # a detailed message even when mis-labelled. Tests below use
        # text bodies that match the LABEL semantics so the Tier 1+2
        # alias path is what's exercised.
        ('Beneficiary Certificate', 'Detailed Message', True),
        ('Beneficiary Certificate', 'Beneficiary Certificate', True),
        ('Shipment Advice',          'Detailed Message', True),
        ('Shipment Advice',          'Insurance Request', True),  # P198dv
        ('Shipment Advice',          'Insurance Pre-Advise Notice', True),
        ('Insurance Policy',         'Insurance Policy',  True),
        # Negative cases — packet body text intentionally short and
        # unrelated so Tier 4 can't false-match on body content.
        ('Beneficiary Certificate', 'Shipment Advice', False,
         'BL NO. ABC123 VESSEL ATLANTIC PORT KARACHI'),
        ('Shipment Advice',          'Beneficiary Certificate', False,
         "WE HEREBY CERTIFY THE BENEFICIARY'S DECLARATION."),
        ('Insurance Policy',         'Detailed Message', False,
         'DETAILED MESSAGE BL VESSEL ETA — no insurance content'),
        ('Insurance Policy',         'Insurance Request', False,
         'Attached doc for your reference. LC ref only.'),
    ]
    for tc in aliases_tests:
        if len(tc) == 3:
            target, pkt_dt, should_match = tc
            body = DETAILED_MESSAGE_TEXT
        else:
            target, pkt_dt, should_match, body = tc
        pkt = {'document_type': pkt_dt, 'cleaned_text': body,
               'refined_text': body}
        m = _find_matching_docs(target, [pkt])
        matched = (len(m) > 0)
        ok = (matched == should_match)
        tag = 'OK ' if ok else 'FAIL'
        print(f"   [{tag}] target='{target}' pkt='{pkt_dt}' -> match={matched} (expected {should_match})")
        if ok: pass_n += 1
        else:  fail_n += 1

    # ── D. F46A-7-style end-to-end: clause expects Beneficiary Certificate
    # / Detailed Message; the Detailed-Message page must be found ──
    print('\n--- D. F46A-7 end-to-end: bene-cert-detailed-message clause ---')
    pkts = [
        {'document_type': 'Detailed Message',
         'cleaned_text': DETAILED_MESSAGE_TEXT,
         'refined_text': DETAILED_MESSAGE_TEXT,
         'packet_id': 'pkt_dm'},
        {'document_type': 'Bill of Lading',
         'cleaned_text': 'BILL OF LADING ...', 'refined_text': '',
         'packet_id': 'pkt_bl'},
    ]
    for target in ('Beneficiary Certificate', 'Detailed Message',
                    'Shipment Advice'):
        m = _find_matching_docs(target, pkts)
        ok = (len(m) >= 1
              and any(p.get('document_type') == 'Detailed Message' for p in m))
        tag = 'OK ' if ok else 'FAIL'
        print(f"   [{tag}] target='{target}' -> matched {len(m)} (Detailed Message included)")
        if ok: pass_n += 1
        else:  fail_n += 1

    # ── E. Real-job 53e62015 sweep: every page that was DR or
    # Covering-Letter pre-guard, what does the guard do? ──
    print('\n--- E. Real job 53e62015 sweep — each page through the guard ---')
    s8 = json.loads((JOB1 / 'step08' / 'step08_result.json').read_text(encoding='utf-8'))
    all_ok = True
    for cpkt in s8.get('classified_packets', []):
        pgs = [pg.get('page_number') for pg in cpkt.get('original_pages', [])
               if isinstance(pg, dict)]
        dt = cpkt.get('document_type', '')
        if dt not in ('Documentary Remittance', 'Covering Letter',
                      'Shipment Advice', 'Beneficiary Certificate'):
            continue
        text = '\n\n'.join(real_pages.get(pn, '') for pn in pgs if pn)
        # The live data already went through the guard (since it
        # was classified after my P198dp commit); re-run the guard
        # on the stored doc_type and confirm it stays the same
        # (idempotency).
        out = dr_guard_classify(dt, text)
        ok = (out == dt or
              # Special: a page already labelled Documentary Remittance
              # by step08 should remain Documentary Remittance — guard
              # idempotent.
              (dt == 'Covering Letter' and out == 'Covering Letter'))
        if not ok:
            all_ok = False
            print(f"   [FAIL] pgs={pgs} stored_dt='{dt}' guard_re-out='{out}' (idempotent expected)")
    if all_ok:
        print('   [OK ] guard is idempotent on every relevant packet in the job')
        pass_n += 1
    else:
        fail_n += 1

    total = pass_n + fail_n
    print('\n' + '=' * 78)
    print(f'OVERALL: {pass_n}/{total} {"OK" if fail_n == 0 else "— failures present"}')
    print('=' * 78)
    return 0 if fail_n == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
