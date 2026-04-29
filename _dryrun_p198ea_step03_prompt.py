"""
P198ea step03-prompt dry-run.

Sends the ACTUAL CLASSIFY_DOCTYPE_PROMPT (the prompt step03's
per-page VLM uses) to the live Qwen text LLM with real OCR text
from job 53e62015 pages 13/26/27/29 and Meiji synthetic. Verifies
the new rule 7b makes the model pick:
  - "Documentary Remittance" / "Document Remittance" for the
    Maybank covering schedule on page 13
  - "Shipment Advice" for the SiekML email cover notes on
    pages 27 and 29 (the case the user reported)
  - "Shipment Advice" for the real Magna-Foremost shipment
    advice on page 26
  - "Detailed Message" / "Beneficiary Certificate" /
    "Shipment Advice" for the synthetic Meiji DETAILED MESSAGE

This is a vision-less LLM check — step03's real VLM call also
sees the page image, but the OCR text alone is rich enough for
the rule 7b text-pattern teaching to fire.
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
from steps.step03_sequencing import CLASSIFY_DOCTYPE_PROMPT


def llm_classify_doctype(glm_text, timeout=60):
    """Hit the live LLM with the same prompt step03's VLM uses,
    minus the image input. Returns the parsed document_type."""
    prompt = CLASSIFY_DOCTYPE_PROMPT.format(glm_text=glm_text[:4000])
    body = {
        'model': QWEN_TEXT_LLM_MODEL,
        'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': 600,
        'temperature': 0.1,
    }
    try:
        r = requests.post(QWEN_TEXT_LLM_URL, json=body, timeout=timeout)
        r.raise_for_status()
        content = r.json().get('choices', [{}])[0].get('message', {}).get('content', '')
        # Extract JSON
        m = re.search(r'\{[\s\S]*?\}', content)
        if m:
            try:
                obj = json.loads(m.group(0))
                return obj.get('document_type', '').strip(), content
            except Exception:
                pass
        return content.strip().splitlines()[0].strip(), content
    except Exception as e:
        return f'__error__: {e}', ''


def matches(label, accepted):
    if not label or label.startswith('__error__'):
        return None
    norm = re.sub(r'[^a-z0-9]', '', label.lower())
    for ok in accepted:
        ok_norm = re.sub(r'[^a-z0-9]', '', ok.lower())
        if ok_norm and (ok_norm in norm or norm in ok_norm):
            return True
    return False


# Real OCR data
JOB = Path('results/53e62015-f805-4985-81e3-2b5de1daee65')
s2 = json.loads((JOB / 'step02' / 'step02_result.json').read_text(encoding='utf-8'))
real_pages = {p['page_number']: (p.get('cleaned_text')
                                  or p.get('raw_text') or '')
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
We are pleased to inform you of our shipment for L/C No.
1019LC55849/2026 dated 2026/1/9
issuing bank BANK AL HABIB LTD, KARACHI
WE CERTIFY THE GOODS TO BE OF E.U. ORIGIN.
"""

# Synthetic test cases beyond the real-job OCR
SYNTHETIC = [
    ('Insurance Cover Request email (Sajid Hassan / IGI)',
     ['Shipment Advice'],
     """\
Sajid Hassan
Subject: FW:INSURANCE COVER-OPEN POLICY NO.2024/12/HRCMIMOO00189
DOCUMENTARY CREDIT NUMBER :1001LC83147/2025

From: Sajid Hassan
Sent: Tuesday, 30 December 2025 5:01 PM
To: 'FARHAN@HABIBRC.COM'<farhan@habibrc.com>;'ANEES@HABIBRC.COM'
Cc: 'Fatima Furqan'<fatimafurqan@cbl.com.pk>
Subject:INSURANCE COVER-OPEN POLICY NO.2024/12/HRCMIMOO00189

TO IGI INSURANCE COMPANY LTD.
AND APPLICANT: CONTINENTAL BISCUITS LIMITED

We request you to kindly insure goods against OPEN POLICY
NO.2024/12/HRCMIMOO00189 as per below details
DOCUMENTARY CREDIT NUMBER 1001LC83147/2025
INVOICE VALUE: EUR 26,880.00
BILL OF LANDING NO. BLQA66810
VESSEL NAME AND VOYAGE NUMBER: GERHARD SCHULTE/550S
PORT OF LOADING: LA SPEZIA SEAPORT, ITALY
PORT OF DISCHARGE: KARACHI SEAPORT, PAKISTAN
SHIPPED ON BOARD DATE: 11/12/2025
"""),
    ('Bare freight forwarder transmittal email',
     ['Shipment Advice'],
     """\
From: ops@globallogistics.com
Sent: Monday, 12 January 2026 09:15 AM
To: applicant@example.com
Subject: COVER NOTE NO.2026-COV-9981
Attachments: docs.pdf

Dear Sir,
Attached doc for your reference. Thanks!
LC Number: 12345/2026 DATED 251215
L/C ISSUING BANK: BANK XYZ
Regards, John Doe
Logistics Executive
"""),
    ('Real bank-issued covering schedule (no email format)',
     ['Document Remittance', 'Documentary Remittance'],
     """\
HABIB BANK LIMITED
PRESENTATION SCHEDULE
Our Reference No: HBL/EX/2026/4521
Your Documentary Credit No: 0001LC55282/2025
Beneficiary: ABC TRADING
Total Amount Claimed: USD 250,000.00
We enclose the following documents drawn under above LC for negotiation/payment
3 Bill of Lading
3 Commercial Invoice
3 Packing List
"""),
    ('Bank covering schedule sent BY EMAIL (real DR via email)',
     ['Document Remittance', 'Documentary Remittance'],
     """\
From: trade.finance@maybank.com.my
Sent: Wednesday, 15 April 2026 9:30 AM
To: bahltradefin@bankalhabib.com
Subject: Documentary Credit Schedule — LC 1023LC88616/2025

Maybank Trade Finance
Our Reference No: 99190WAM2747705
Your Documentary Credit No: 1023LC88616/2025
Total Amount Claimed: USD 33,203.85
We enclose the following documents drawn under above LC for negotiation/payment:
3 Bill of Lading
8 Commercial Invoice
2 Bill of Exchange
Payment Instruction: Remit funds to our correspondent.
"""),
]


def main():
    pass_n, fail_n = 0, 0
    print('=' * 78)
    print('P198ea step03-prompt dry-run — live LLM with rule 7b')
    print(f'LLM: {QWEN_TEXT_LLM_URL}')
    print('=' * 78)

    # Real-job pages
    real_cases = [
        ('p13 Maybank Documentary Credit Schedule (real DR)',
         ['Document Remittance', 'Documentary Remittance'],
         real_pages.get(13, '')),
        ('p26 real Magna-Foremost Shipment Advice',
         ['Shipment Advice'],
         real_pages.get(26, '')),
        ('p27 SiekML email cover note',
         ['Shipment Advice'],
         real_pages.get(27, '')),
        ('p29 SiekML email cover note (duplicate of p27)',
         ['Shipment Advice'],
         real_pages.get(29, '')),
        ('Synthetic Meiji DETAILED MESSAGE',
         ['Detailed Message', 'Beneficiary Certificate', 'Shipment Advice'],
         DETAILED_MESSAGE_TEXT),
    ]

    print('\n--- A. Real OCR pages (job 53e62015) ---')
    for name, accepted, text in real_cases:
        if not text:
            print(f'[SKIP] {name} — no real text')
            continue
        t0 = time.time()
        label, _raw = llm_classify_doctype(text)
        ok = matches(label, accepted)
        elapsed = time.time() - t0
        if ok is True:
            tag = 'OK  '
            pass_n += 1
        elif ok is False:
            tag = 'FAIL'
            fail_n += 1
        else:
            tag = 'SKIP'
        print(f"[{tag}] {name}  ({elapsed:.1f}s)")
        print(f"        LLM='{label}'  accepted={accepted}")

    print('\n--- B. Synthetic edge cases ---')
    for name, accepted, text in SYNTHETIC:
        t0 = time.time()
        label, _raw = llm_classify_doctype(text)
        ok = matches(label, accepted)
        elapsed = time.time() - t0
        if ok is True:
            tag = 'OK  '
            pass_n += 1
        elif ok is False:
            tag = 'FAIL'
            fail_n += 1
        else:
            tag = 'SKIP'
        print(f"[{tag}] {name}  ({elapsed:.1f}s)")
        print(f"        LLM='{label}'  accepted={accepted}")

    total = pass_n + fail_n
    print('\n' + '=' * 78)
    print(f'OVERALL: {pass_n}/{total} '
          f'{"OK" if fail_n == 0 else "— failures present"}')
    print('=' * 78)
    return 0 if fail_n == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
