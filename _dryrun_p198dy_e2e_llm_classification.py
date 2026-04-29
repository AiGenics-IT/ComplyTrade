"""
P198dy end-to-end LLM dry-run.

Runs the deterministic step08 P198dp/P198dy guard AND a live-LLM
classification check on real OCR pages from job 53e62015 +
synthetic Meiji DETAILED MESSAGE. The live LLM is asked
"what type of document is this?" and the answer is checked
against the deterministic verdict.

Hits the live Qwen text LLM endpoint configured in
config/settings.py (QWEN_TEXT_LLM_URL). Skips the LLM step
gracefully if the endpoint is unreachable.
"""
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import requests
from config.settings import QWEN_TEXT_LLM_URL, QWEN_TEXT_LLM_MODEL


# ──────────────────────────────────────────────────────────────────────
# Mirrors of the live step08 guard (kept in sync; pure Python).
# ──────────────────────────────────────────────────────────────────────

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


def deterministic_classify(initial_doc_type, glm_text):
    document_type = initial_doc_type
    _ux = (glm_text or '').upper()
    # Detailed Message upgrade
    if document_type in (
        'Shipment Advice', 'Shipping Advice',
        'Beneficiary Certificate', "Beneficiary's Certificate",
        'Documentary Remittance', 'Covering Letter', 'Covering Schedule',
    ):
        _has_dm_header = bool(re.search(
            r'(?:^|\n)\s*DETAIL(?:ED)?\s+MESSAGE\b', _ux, re.MULTILINE))
        _has_bene_cert = bool(re.search(
            r'\bWE\s+CERTIFY\b|CERTIFY(?:ING)?\s+(?:THE\s+)?GOODS\s+'
            r'(?:TO\s+BE\s+)?(?:OF|ARE\s+OF)|'
            r'\bWE\s+ARE\s+PLEASED\s+TO\s+INFORM\s+YOU\s+OF\s+OUR\s+SHIPMENT\b',
            _ux))
        _has_fax_or_email = bool(re.search(
            r'\bFAX(?:\s*(?:NO\.?|NUMBER|#))?\s*[:.\d]|'
            r'\bDIRECT(?:LY)?\s+TO\s+THE\s+APPLICANT\s+BY\s+FAX|'
            r'\bSENT\s+BY\s+FAX\b', _ux))
        _has_shipment_evidence = bool(re.search(
            r'\bB[/\s]*L\s+(?:NO\.?|NUMBER)|\bBILL\s+OF\s+LADING|'
            r'\bVESSEL\b|\bETA\b|\bETD\b|\bSHIPPED\s+ON\s+BOARD\b', _ux))
        if (_has_dm_header and _has_shipment_evidence
            and (_has_bene_cert or _has_fax_or_email)):
            document_type = 'Detailed Message'
    # DR-guard
    if document_type == 'Documentary Remittance':
        _signals = sum(1 for p in _DR_REAL_SIGNALS if re.search(p, _ux))
        _bank = bool(_BANK_RE.search(_ux))
        _swift = bool(_SWIFT_RE.search(_ux))
        _is_email = bool(
            re.search(r'\bFROM\s*:\s*[^\n]*@', _ux)
            and re.search(r'\bSUBJECT\s*:', _ux))
        if _is_email:
            _is_real_dr = _signals >= 3
        else:
            _is_real_dr = (_signals >= 2 or (_bank and _signals >= 1)
                           or (_swift and _signals >= 1))
        if not _is_real_dr:
            document_type = 'Shipment Advice' if _is_email else 'Covering Letter'
    return document_type


# ──────────────────────────────────────────────────────────────────────
# Live LLM call
# ──────────────────────────────────────────────────────────────────────

LLM_PROMPT = """You are a trade finance documents classifier.
Given the document text below, output ONLY the canonical document
type from this list. Pick the SINGLE best match. Do not explain.
Just output the name verbatim.

CANDIDATES:
- Documentary Remittance   (a bank's covering schedule listing
  enclosed documents for negotiation/payment, usually on bank
  letterhead with "WE ENCLOSE", "TOTAL AMOUNT CLAIMED", "OUR
  REFERENCE NO", "L/C ISSUING BANK", "REIMBURSING BANK",
  "PAYMENT INSTRUCTION" etc.)
- Shipment Advice          (a notification with shipment details:
  vessel, B/L, ETA, port of loading/discharge, OR a forwarder /
  beneficiary email cover note travelling alongside the shipment-
  advice attachment, referencing the LC)
- Detailed Message         (a beneficiary fax titled "DETAILED
  MESSAGE" with shipment details + a "WE CERTIFY" line)
- Beneficiary Certificate  (a short certification by the beneficiary)
- Bill of Lading
- Commercial Invoice
- Packing List
- Insurance Policy
- Insurance Certificate
- Covering Letter           (a non-bank transmittal letter without
  the bank-side payment-claim language)
- Other

DOCUMENT TEXT:
<<<
{text}
>>>

Output (just the category name, nothing else):"""


def llm_classify(text, timeout=60):
    body = {
        'model': QWEN_TEXT_LLM_MODEL,
        'messages': [
            {'role': 'user', 'content': LLM_PROMPT.format(text=text[:6000])},
        ],
        'max_tokens': 32,
        'temperature': 0.1,
    }
    try:
        r = requests.post(QWEN_TEXT_LLM_URL, json=body, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        return (data.get('choices', [{}])[0]
                    .get('message', {}).get('content', '')
                    .strip().splitlines()[0].strip())
    except Exception as e:
        return f'__error__: {e}'


# ──────────────────────────────────────────────────────────────────────
# Test data
# ──────────────────────────────────────────────────────────────────────

JOB = Path('results/53e62015-f805-4985-81e3-2b5de1daee65')
s2 = json.loads((JOB / 'step02' / 'step02_result.json').read_text(encoding='utf-8'))
real_pages = {p['page_number']: (p.get('cleaned_text') or p.get('raw_text') or '')
              for p in s2['pages']}

DETAILED_MESSAGE_TEXT = """meiji
2026/2/23
MEIJI CO., LTD.
DETAILED MESSAGE
TO : GLOBAL BRANDS MARKETING (PVT) LTD
FAX.0092-21-35654644
NAME OF ITEM : MILK PREPARATION
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
"""


def llm_agrees(llm_label, accepted):
    """Loose match: LLM output vs any of the accepted canonical names."""
    if not llm_label or llm_label.startswith('__error__'):
        return None  # skip
    norm = re.sub(r'[^a-z]', '', llm_label.lower())
    for ok in accepted:
        if re.sub(r'[^a-z]', '', ok.lower()) in norm:
            return True
        if norm in re.sub(r'[^a-z]', '', ok.lower()):
            return True
    return False


def main():
    pass_n, fail_n = 0, 0

    print('=' * 78)
    print('P198dy E2E dry-run — deterministic guard + live LLM agreement')
    print(f'LLM endpoint: {QWEN_TEXT_LLM_URL}')
    print('=' * 78)

    cases = [
        # (label, initial step08 input, page text, deterministic expected,
        #  LLM-acceptable answers)
        ('p13 Maybank covering schedule (real DR)',
         'Documentary Remittance', real_pages.get(13, ''),
         'Documentary Remittance',
         ['Documentary Remittance']),
        ('p27 SiekML email cover note',
         'Documentary Remittance', real_pages.get(27, ''),
         'Shipment Advice',
         ['Shipment Advice', 'Covering Letter']),
        ('p29 SiekML email cover note (duplicate of p27)',
         'Documentary Remittance', real_pages.get(29, ''),
         'Shipment Advice',
         ['Shipment Advice', 'Covering Letter']),
        ('p26 real Shipment Advice (Magna-Foremost)',
         'Shipment Advice', real_pages.get(26, ''),
         'Shipment Advice',
         ['Shipment Advice']),
        ('Synthetic Meiji DETAILED MESSAGE',
         'Shipment Advice', DETAILED_MESSAGE_TEXT,
         'Detailed Message',
         ['Detailed Message', 'Beneficiary Certificate', 'Shipment Advice']),
    ]

    for label, in_dt, text, expected, accepted in cases:
        if not text:
            print(f'[SKIP] {label} — no real text on disk')
            continue
        t0 = time.time()
        det = deterministic_classify(in_dt, text)
        det_ok = (det == expected)
        llm_label = llm_classify(text)
        llm_ok = llm_agrees(llm_label, accepted)
        elapsed = time.time() - t0

        det_tag = 'OK ' if det_ok else 'FAIL'
        if llm_ok is None:
            llm_tag = 'SKIP'
        elif llm_ok:
            llm_tag = 'OK '
        else:
            llm_tag = 'FAIL'

        print(f"\n[{det_tag} det / {llm_tag} llm]  {label}  ({elapsed:.1f}s)")
        print(f"        deterministic = '{det}'  (expected '{expected}')")
        if llm_label.startswith('__error__'):
            print(f"        LLM           = (unreachable: {llm_label[12:80]})")
        else:
            print(f"        LLM           = '{llm_label}'  (acceptable: {accepted})")

        if det_ok: pass_n += 1
        else:      fail_n += 1
        if llm_ok is True:  pass_n += 1
        elif llm_ok is False: fail_n += 1
        # llm_ok None = skip, no count

    print('\n' + '=' * 78)
    total = pass_n + fail_n
    print(f'OVERALL: {pass_n}/{total} {"OK" if fail_n == 0 else "— failures present"}')
    print('=' * 78)
    return 0 if fail_n == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
