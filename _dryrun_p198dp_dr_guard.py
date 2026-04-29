"""
P198dp dry-run — Documentary Remittance false-positive guard.

The guard demotes 'Documentary Remittance' to 'Covering Letter'
unless the page text shows real bank-side / payment-claim signals.

Tests use job 406fec4f-afc1-4e94-8d9d-5868141dca8b's actual OCR
(page 13 = real Maybank covering schedule, pages 27 + 29 = email
cover notes from a logistics company) plus a wide set of synthetic
scenarios covering other realistic banks / shapes / mis-routes.
"""
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')


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


def evaluate(text):
    """Mirror of the P198dp guard. Returns ('keep'|'demote', counts)."""
    u = (text or '').upper()
    signals = sum(1 for p in _DR_REAL_SIGNALS if re.search(p, u))
    bank = bool(_BANK_RE.search(u))
    swift = bool(_SWIFT_RE.search(u))
    is_email = bool(
        re.search(r'\bFROM\s*:\s*[^\n]*@', u)
        and re.search(r'\bSUBJECT\s*:', u)
    )
    if is_email:
        is_real = signals >= 3
    else:
        is_real = (
            signals >= 2
            or (bank and signals >= 1)
            or (swift and signals >= 1)
        )
    return ('keep' if is_real else 'demote'), {
        'signals': signals, 'bank': bank, 'swift': swift, 'email': is_email,
    }


# ── Real job 406fec4f data ────────────────────────────────────────
# P198dx — original test job 406fec4f was cleared from disk;
# fall back to 53e62015 which has the same email-cover-note
# pattern on its pages 27 + 29 (same SiekML / Samling content,
# same Maybank Documentary Credit Schedule on page 13).
_CANDIDATES = [
    'results/53e62015-f805-4985-81e3-2b5de1daee65',
    'results/406fec4f-afc1-4e94-8d9d-5868141dca8b',
]
JOB = next((Path(p) for p in _CANDIDATES if (Path(p) / 'step02').exists()),
           Path(_CANDIDATES[0]))
step02 = json.loads((JOB / 'step02' / 'step02_result.json').read_text(
    encoding='utf-8'))
real_pages = {p['page_number']: (p.get('cleaned_text')
                                  or p.get('raw_text') or '')
              for p in step02['pages']}


# ── Synthetic scenarios ───────────────────────────────────────────
SYNTH = [
    # 1. HBL Pakistan covering schedule
    ('HBL Pakistan covering schedule (real bank DR)',
     'keep',
     """HABIB BANK LIMITED
PRESENTATION SCHEDULE
Our Reference No: HBL/EX/2026/4521
Your Documentary Credit No: 0001LC55282/2025
Beneficiary: ABC TRADING
Total Amount Claimed: USD 250,000.00
We enclose the following documents drawn under above LC
3 Commercial Invoice
3 Bill of Lading
3 Packing List
Please remit funds to our correspondent.
"""),

    # 2. Citibank LC Bills Schedule
    ('Citibank LC Bills Schedule',
     'keep',
     """CITIBANK NA
LC BILLS SCHEDULE
Presentation Number: PRES-2026-78912
Presentation Date: 2026-04-14
Your Documentary Credit No: 12345/2026
We enclose herewith documents for negotiation.
Documents Attached:
- 3 commercial invoice
- 2 bill of lading
"""),

    # 3. Bare email cover note (FedEx pouch slip / cover note)
    ('Bare email cover note from logistics company',
     'demote',
     """From: shipping@globallogistics.com
Sent: Monday, 12 January 2026 09:15 AM
To: applicant@example.com
Subject: COVER NOTE NO. 2025-12-CN-9981
Attachments: docs.pdf

Dear Sir,
Attached doc for your reference. Thanks!
LC: 12345/2026
Regards, John Doe
Logistics Executive
"""),

    # 4. Beneficiary's Certificate that mentions LC reference but no DR claim language
    ('Beneficiary Certificate falsely tagged DR (no real DR signals)',
     'demote',
     """BENEFICIARY'S CERTIFICATE
We hereby certify that the goods shipped under
LC No. 12345/2026 are of Brazilian origin and conform
to the specifications described in our proforma invoice.
Date: 2026-04-12
For ABC Trading Co.
"""),

    # 5. Shipment Advice mentioning LC and "documents attached" once
    ('Shipment Advice with single "documents attached" mention',
     'demote',
     """SHIPMENT ADVICE
Date: 2026-04-10
LC No: 12345/2026
B/L No: TLI-9876
Vessel: MV ATLANTIC
Quantity: 1000 MT
Documents attached: original B/L, packing list.
Best regards,
Beneficiary
"""),

    # 6. Bank cover schedule with only 1 strong signal but bank letterhead
    ('Bank letterhead + 1 signal — kept as DR',
     'keep',
     """MAYBANK
Trade Finance Department
We enclose the following documents drawn under your LC
3 BL
2 CI
"""),

    # 7. SWIFT-only header with one DR signal
    ('SWIFT BIC line + 1 signal — kept as DR',
     'keep',
     """SWIFT: BAHLPKKAXXX
Branch: Karachi
Presentation Number: 2026-001
Documents enclosed for negotiation.
"""),

    # 8. Random invoice with phrase "we enclose" once but no other signals
    ('Invoice with single weak signal — demoted',
     'demote',
     """COMMERCIAL INVOICE
Invoice No.: INV-2026-001
We enclose the following items for the buyer's reference.
Goods: Soybeans 1000 MT
"""),

    # 9. Empty text — demoted (no signals)
    ('Empty text — demoted',
     'demote',
     ""),

    # 10. Real DR with only "Covering Schedule" header (single match)
    ('"Covering Schedule" title alone — single signal, no bank — demoted',
     'demote',
     """COVERING SCHEDULE
Date: 2026-01-15
Some unrelated content.
"""),

    # 11. "Covering Schedule" + payment language (2 signals) — kept
    ('"Covering Schedule" + payment language — kept',
     'keep',
     """COVERING SCHEDULE
Date: 2026-01-15
Total Amount Claimed: USD 100,000
"""),

    # 12. Forwarder's "transmittal letter" email — demoted
    ('Freight forwarder transmittal email — demoted',
     'demote',
     """From: ops@freightforwarder.com
Subject: Transmittal Letter for Shipment No. 12345
Dear Buyer,
Please find attached the documents for your shipment.
Transmittal Letter Reference: TL-2026-998
Best regards
"""),
]


def main():
    print('=' * 78)
    print('P198dp dry-run — DR false-positive guard')
    print('=' * 78)
    pass_n = fail_n = 0

    # ── A. Real-job pages ──
    print('\n--- A. Real OCR from job 406fec4f ---\n')
    real_cases = [
        ('p13', 'keep'),    # Maybank Documentary Credit Schedule
        ('p27', 'demote'),  # Email cover from logistics co.
        ('p29', 'demote'),  # Email cover from logistics co.
    ]
    for label, expect in real_cases:
        pn = int(label[1:])
        verdict, cnt = evaluate(real_pages.get(pn, ''))
        ok = (verdict == expect)
        tag = 'OK ' if ok else 'FAIL'
        print(f"[{tag}] real {label}: expect={expect:6s} got={verdict:6s} "
              f"signals={cnt['signals']} bank={cnt['bank']} "
              f"swift={cnt['swift']} email={cnt['email']}")
        if ok: pass_n += 1
        else: fail_n += 1

    # ── B. Synthetic scenarios ──
    print('\n--- B. Synthetic scenarios ---\n')
    for i, (name, expect, text) in enumerate(SYNTH, 1):
        verdict, cnt = evaluate(text)
        ok = (verdict == expect)
        tag = 'OK ' if ok else 'FAIL'
        print(f"[{tag}] #{i:02d} ({expect:6s}) {name}")
        print(f"      got={verdict:6s} signals={cnt['signals']} "
              f"bank={cnt['bank']} swift={cnt['swift']} "
              f"email={cnt['email']}")
        if ok: pass_n += 1
        else: fail_n += 1

    print('\n' + '=' * 78)
    print(f'Total: {pass_n}/{pass_n+fail_n} OK')
    print('=' * 78)
    return 0 if fail_n == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
