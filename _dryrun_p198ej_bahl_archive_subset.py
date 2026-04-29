"""
P198ej dry-run — BAHL multi-message report grouping must NOT
extend beyond the user's actual PDF when the Alliance footer
references a global page count from a larger archive.

User's failing case (job 7e3699a8): page 5 of a 31-page PDF
contains an MT707 amendment whose footer reads "Page 1931 of
12905" (= page 1931 of the original Alliance archive). The
old math _bahl_max_page = pg_num + (Y - X) = 5 + (12905 - 1931)
= 10979 swept ALL 31 pages into the BAHL Amendment group.

Two fixes verified here:
  1. Cap _bahl_max_page at the actual highest page in the PDF
     (so the math can't extend past page 31 in this example).
  2. Drop pages with NO SWIFT-message structure (no F-tags / no
     narrative / no Block 4-5 / no Message Header) from the
     BAHL group, even within the page-count window.
"""
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')


_BAHL_MSG_DETAIL_RE = re.compile(
    r'Message\s+Details\s+#\s*(\d+)', re.IGNORECASE)
_BAHL_IDENTIFIER_RE = re.compile(
    r'Identifier\s*:\s*fin\.(\d{3})', re.IGNORECASE)
_BAHL_FIN_TO_MT = {
    '700': 'LC', '701': 'LC', '705': 'LC',
    '707': 'Amendment', '708': 'Amendment', '747': 'Amendment',
    '799': 'MT799', '999': 'MT999',
    '754': 'MT754', '940': 'MT940', '730': 'MT730',
    '740': 'MT740', '742': 'MT742',
    '734': 'MT734', '750': 'MT750', '752': 'MT752',
}
_PAGE_XY_RE = re.compile(
    r'Page\s+(\d+)\s*(?:of|/)\s*(\d+)', re.IGNORECASE)
_BARE_XY_RE = re.compile(
    r'(?:^|\n)\s*(\d{1,2})\s+of\s+(\d{1,2})\s*(?=\n|$)',
    re.IGNORECASE | re.MULTILINE)
_BARE_XY_SLASH_RE = re.compile(
    r'(?:^|\n)\s*(\d{1,2})\s*/\s*(\d{1,2})\s*(?=\n|$)',
    re.MULTILINE)
_SWIFT_STRUCT_RE = re.compile(
    r'(?:^|\n)\s*(?:'
    r'(?::?F\d{2}[A-Z]?:|:\d{2}[A-Z]?:)|'
    r'Narrative\s*\d?\s*:|'
    r'Block\s+[1-5]\b|'
    r'Message\s+(?:Header|Identifier|Text)|'
    r'Sender\s*(?:Institution)?\s*:|'
    r'Receiver\s*(?:Institution)?\s*:|'
    r'\bfin\.\d{3}\b|'
    r'Transaction\s+Reference|'
    r'Sequence\s+of\s+Total|'
    r'Status\s*:\s*(?:Modified|Deletable|Read-Only|Acknowledged))',
    re.IGNORECASE | re.MULTILINE)


def detect_xy(text):
    if not text:
        return None
    m = _PAGE_XY_RE.search(text)
    if m:
        x, y = int(m.group(1)), int(m.group(2))
        # Note: NO upper bound on Y here — the caller deals with
        # archive-style "Page 1931 of 12905".
        if 1 <= x <= y:
            return x, y
    tail = text[-400:]
    for r in (_BARE_XY_RE, _BARE_XY_SLASH_RE):
        for _m in r.finditer(tail):
            x, y = int(_m.group(1)), int(_m.group(2))
            if 1 <= x <= y <= 99:
                return x, y
    return None


def simulate_grouping(pages):
    """Mirror the live step03 grouping after P198eg + P198ej.
    pages: list of (page_number, text)."""
    sorted_pages = sorted(pages, key=lambda x: x[0])
    actual_max = max((pn for pn, _ in sorted_pages), default=0)

    # Build msg_detail_pages and page_xy
    msg_detail_pages = {}
    page_xy = {}
    page_has_swift = {}
    for pn, txt in sorted_pages:
        if not txt:
            page_has_swift[pn] = False
            continue
        page_has_swift[pn] = bool(_SWIFT_STRUCT_RE.search(txt))
        for m in _BAHL_MSG_DETAIL_RE.finditer(txt):
            msg_detail_pages.setdefault(pn, []).append(int(m.group(1)))
        xy = detect_xy(txt)
        if xy:
            page_xy[pn] = xy

    if len(msg_detail_pages) < 2:
        return {}, []

    # Compute bahl_max_page (capped at actual_max — P198ej fix)
    bahl_max = 0
    for pn in sorted(msg_detail_pages):
        if pn in page_xy:
            x, y = page_xy[pn]
            bahl_max = max(bahl_max, pn + (y - x))
    if bahl_max > actual_max:
        bahl_max = actual_max

    bahl = {}
    cur_grp = None
    next_id = 0
    bahl_pages = []
    for pn, txt in sorted_pages:
        if bahl_max > 0 and pn > bahl_max:
            break
        if pn in msg_detail_pages:
            msg_num = sorted(msg_detail_pages[pn])[-1]
            id_m = _BAHL_IDENTIFIER_RE.search(txt or '')
            page_fin = id_m.group(1) if id_m else ''
            start_new = (
                cur_grp is None
                or msg_num == 1
                or (page_fin and bahl.get(cur_grp, {}).get('fin')
                    and page_fin != bahl[cur_grp]['fin'])
            )
            if start_new:
                next_id += 1
                bahl[next_id] = {
                    'pages': [], 'fin': page_fin,
                    'mt': _BAHL_FIN_TO_MT.get(page_fin, ''),
                    'msg_num_in_report': msg_num,
                }
                cur_grp = next_id
        # P198ej — drop non-SWIFT pages
        if pn not in msg_detail_pages and not page_has_swift.get(pn):
            cur_grp = None
            continue
        if cur_grp is not None:
            bahl[cur_grp]['pages'].append(pn)
            bahl_pages.append(pn)
    return bahl, bahl_pages


# ── Real job 7e3699a8 — user's exact case ──────────────────────────
JOB = 'results/7e3699a8'
import glob
matches = glob.glob(f'{JOB}*/step02/step02_result.json')
real_pages = []
if matches:
    s2 = json.loads(Path(matches[0]).read_text(encoding='utf-8'))
    for p in s2['pages']:
        real_pages.append((p['page_number'],
                            p.get('cleaned_text') or p.get('raw_text') or ''))


# ── Synthetic scenarios ────────────────────────────────────────────

def syn_swift_msg_page(msg_num, fin, page_x_of_y, body=''):
    return f"""Report Header
Application Alliance Message Management
Message Details #{msg_num}
Identifier: fin.{fin}
Expansion: Amendment to a Documentary Credit
Transaction Reference: ABC-123
Sender: BAHLPKKACPU
Receiver: BPMOIT22XXX
F20: Sender's Reference
F26E: Number of Amendment
{body}
Page {page_x_of_y[0]} of {page_x_of_y[1]}"""


def syn_swift_cont_page(page_x_of_y, body='F45A: GOODS\nNarrative: continuation'):
    return f"""{body}

Page {page_x_of_y[0]} of {page_x_of_y[1]}"""


def syn_shipping_page(label='Bill of Lading', body=''):
    return f"""{label}
{body}
Vessel: ABC SHIP
Consignee: TO ORDER
Notify: APPLICANT"""


# User's exact pasted text from job 7e3699a8 — the MT707 amendment
USER_PAGE_5 = """Message Details #866

Message Identifier

Message Preparation Application: Applic. Interface
Unique Message Identifier: IBPMOIT22XXX 707 0329LC75704/2025 (suffix 25082132199372)

Message Header

Status: Message Modified
Deletable
Format: Swift Sub-Format: Input
Identifier: fin.707 Expansion: Amendment to a
Documentary Credit
Application FIN Nature: Financial
Sender: BAHLPKKACPU LT: X
Receiver: BPMOIT22XXX LT: X
Transaction Reference: 0329LC75704/2025
Related Reference: NON-REF
Priority: Normal
Monitoring: None
MUR: EN4001336
Amount: 3,300. Currency: EUR Value Date:
ACK/NAK Reception Date/Time (GMT): 2025/08/21 15:01:43

Sender / Receiver:

Sender Institution: BAHLPKKACPU Expansion: BANK AL HABIB LIMITED
CENTRAL PROCESSING UNIT
74000 KARACHI
KARACHI
PK
PAKISTAN
Receiver Institution: BPMOIT22XXX Expansion: BPER BANCA S.P.A.
MODENA
IT

Message Text

Block 4
F27: Sequence of Total
Number: 1/
Total: 1
F20: Sender's Reference
0329LC75704/2025
F21: Receiver's Reference
NON-REF
F23: Issuing Bank's Reference
0329LC75704/2025
F52A: Issuing Bank - Party Identifier - Identifier Code
Identifier Code:
BAHLPKKACPU
BANK AL HABIB LIMITED
CENTRAL PROCESSING UNIT
KARACHI PK
F31C: Date of Issue
250731 2025 Jul 31
F26E: Number of Amendment
1
F30: Date of Amendment

Page 1931 of 12905"""

USER_PAGE_6 = """250821
2025 Aug 21
F22A: Purpose of Message
ISSU
F31D: Date and Place of Expiry
Date: 251029 2025 Oct 29
Place: AT ITALY
F32B: Increase of Documentary Credit Amount
Currency: EUR EURO
Amount: 3300, #3,300.#
F43T: Transhipment
ALLOWED
F44C: Latest Date of Shipment
250930 2025 Sep 30
F45B: Description of Goods and/or Services
Line 1
Code: /REPALL/
Lines 2-100
Narrative: 1)USE WORSTED RING SPINNING FRAMES,
Narrative: QUANTITY : 3.00 UNITS
Narrative: AT THE RATE OF EURO 3,300 PER UNIT
Narrative:
Narrative: 2)USED TWISTING TWO FOR ONE
Narrative: QUANTITY : 2.00 UNITS
Narrative: AT THE RATE OF EURO 1,650 PER UNIT
Narrative:
Narrative: SPECIFICATIONS AND FURTHER DETAILS ARE STRICTLY
Narrative: AS PER BENEFICIARY'S PROFORMA INVOICE NO. 05/2025
Narrative: DATED AUG 05, 2025
Narrative:
Narrative: CFR KARACHI SEAPORT - PAKISTAN (INCOTERMS : 2020)
Other
Delivery overdue warning request No
Network delivery notif. request No
Payment Confirmation Status:
Confirmed Currency:
Confirmed Amount:
Confirmed Date:
Page 1932 of 12905"""

# A second BAHL message header is needed for BAHL mode to trigger
# (≥2 message headers). Add a synthetic earlier MT700 to satisfy
# that requirement when testing the user's exact paste.
USER_PAGE_3_MT700 = """Message Details #865
Identifier: fin.700
Expansion: Issue of a Documentary Credit
Sender: BAHLPKKACPU
Receiver: BPMOIT22XXX
Transaction Reference: 0329LC75704/2025
F20: Sender's Reference
F31C: Date of Issue
F45A: GOODS

Page 1929 of 12905"""

USER_PAGE_4_MT700_CONT = """F46A: Documents Required
Narrative: 1) Commercial Invoice
Narrative: 2) Bill of Lading
F47A: Additional Conditions

Page 1930 of 12905"""


DIVERSE_SHIPPING = [
    'BILL OF LADING\nB/L No. ABC123\nVessel: SHIP\nConsignee: TO ORDER\nFreight: PREPAID',
    'BILL OF LADING — Continuation\n(Reverse: Terms and Conditions)\nClause 1: ...',
    'COMMERCIAL INVOICE\nInvoice No.: INV-2025-001\nGoods: WIDGETS 1000 UNITS',
    'COMMERCIAL INVOICE\nPage 2 — additional line items\nTotal: USD 100,000',
    'PACKING LIST\nDate: 16 Aug 2025\nGoods: WIDGETS\nGross: 25,000 KG',
    'INSURANCE CERTIFICATE\nPolicy No.: POL-2025-99\nInsured value: USD 110,000',
    'CERTIFICATE OF ORIGIN\nForm A — generalized scheme of preferences',
    'WEIGHT CERTIFICATE\nGross: 25,000 KG  Net: 23,500 KG',
    'PHYTOSANITARY CERTIFICATE\nIssued by Plant Quarantine Dept',
    'BENEFICIARY CERTIFICATE\nWe hereby certify that one set of docs '
    'has been sent by courier to the applicant',
    'DOCUMENTARY REMITTANCE\nMaybank Trade Finance\nWE ENCLOSE THE FOLLOWING DOCUMENTS\n'
    'TOTAL AMOUNT CLAIMED: USD 100,000',
    'COVER NOTE NO.2025-CN-998\nFrom: shipping@logistics.com\nSubject: COVER NOTE\n'
    'Attached doc for your reference',
    'SHIPMENT ADVICE\nDate: 16 Aug 2025\nB/L No.: ABC123\nVessel: SHIP\n'
    'ETA Karachi: 30 Sep 2025',
    'FUMIGATION CERTIFICATE\nBatch: F-2025-66\nInsecticide: methyl bromide',
    'HEALTH CERTIFICATE\nFit for human consumption',
    'INSPECTION CERTIFICATE\nQuality verified by SGS',
    'SURVEY REPORT\nFull Loading Survey\nVessel arrived 14 Aug 2025',
    'DRAFT BILL OF EXCHANGE\nAt sight\nDrawer: BENEFICIARY  Drawee: BANK',
    'PROFORMA INVOICE\nNo. PI-2025-100  Date: 1 Aug 2025',
    'FAX TRANSMISSION REPORT\nReceived OK\n14 Aug 2025 09:15',
    'EMAIL CONFIRMATION\nSent: 14 Aug 2025\nFrom: ops@beneficiary.com',
    'CHAMBER OF COMMERCE CERTIFICATE\nGoods of Italian origin',
    'AIR WAYBILL\nAWB No.: 020-12345678\nFlight: ABC123\nFROM: Milan TO: Karachi',
    'Q/A LAB REPORT\nSpecifications meet contract',
    'BLANK PAGE',
]


SCENARIOS = [
    # (name, pages, expected_bahl_groups, expected_bahl_pages_set)
    (
        'USER PASTE: 31-page PDF with MT700 (pp 3-4) + MT707 (pp 5-6) '
        'from a "Page X of 12905" Alliance archive — pages 7-31 are '
        'DIVERSE shipping docs (BL/CI/PL/Cert/Insurance/COO/Weight/'
        'Phyto/BeneCert/DocRemit/CoverNote/ShipAdv/Fumig/Health/'
        'Inspect/Survey/Draft/Proforma/Fax/Email/Chamber/AWB/QA/Blank)'
        ' and must NOT be swept into BAHL groups',
        [(1, 'Cover page'),
         (2, 'Continuation'),
         (3, USER_PAGE_3_MT700),
         (4, USER_PAGE_4_MT700_CONT),
         (5, USER_PAGE_5),
         (6, USER_PAGE_6)]
        + [(p, DIVERSE_SHIPPING[(p - 7) % len(DIVERSE_SHIPPING)])
           for p in range(7, 32)],
        2,  # 2 BAHL groups (MT700 + MT707)
        {3, 4, 5, 6},  # only the SWIFT pages
    ),
    (
        'Archive subset: MT707 on pp 5-6 with "Page 1931 of 12905",'
        ' shipping docs on 7-31 — must NOT sweep 7-31 into Amendment',
        # Build a 31-page synthetic mirror of job 7e3699a8
        [(1, syn_swift_msg_page(865, '700', (1929, 12905), body='F31C: 250731'))]
        + [(2, syn_swift_cont_page((1930, 12905)))]
        + [(3, syn_shipping_page('Header Page'))]
        + [(4, syn_shipping_page('Bill of Lading'))]
        + [(5, syn_swift_msg_page(866, '707', (1931, 12905), body='F31C: 250731'))]
        + [(6, syn_swift_cont_page((1932, 12905), body='F45B: GOODS\nNarrative: line'))]
        + [(p, syn_shipping_page(f'BL/Invoice page {p}')) for p in range(7, 32)],
        2,  # 2 BAHL groups (LC msg + Amendment msg)
        # SWIFT pages only: 1, 2, 5, 6
        # (page 4 is shipping → not in BAHL)
        {1, 2, 5, 6},
    ),
    (
        'Normal in-PDF report: "Page 1 of 7" — no archive subset issue',
        [
            (1, syn_swift_msg_page(1, '700', (1, 7), body='F31C: 250731')),
            (2, syn_swift_cont_page((2, 7))),
            (3, syn_swift_cont_page((3, 7))),
            (4, syn_swift_cont_page((4, 7))),
            (5, syn_swift_msg_page(2, '707', (5, 7), body='F31C: 250731')),
            (6, syn_swift_cont_page((6, 7))),
            (7, syn_swift_cont_page((7, 7))),
            (8, syn_shipping_page('Bill of Lading')),
        ],
        2,  # LC group + Amendment group
        {1, 2, 3, 4, 5, 6, 7},  # all SWIFT pages
    ),
    (
        'Archive subset with non-SWIFT page mixed inside the boundary',
        [
            (1, syn_swift_msg_page(100, '700', (500, 9999),
                                    body='F31C: 250731')),
            (2, syn_swift_cont_page((501, 9999))),
            (3, syn_shipping_page('Bill of Lading')),  # non-SWIFT inside boundary
            (4, syn_swift_msg_page(101, '707', (502, 9999),
                                    body='F31C: 250731')),
            (5, syn_swift_cont_page((503, 9999))),
        ],
        2,  # one LC + one Amendment
        {1, 2, 4, 5},  # page 3 should NOT be in BAHL
    ),
    (
        'Single message on first 2 pages, the rest unrelated shipping docs',
        [
            (1, syn_swift_msg_page(1, '700', (1, 2),
                                    body='F31C: 250731')),
            (2, syn_swift_cont_page((2, 2))),
            (3, syn_shipping_page('Commercial Invoice')),
            (4, syn_shipping_page('Packing List')),
        ],
        # Only 1 BAHL message header → BAHL not triggered (need ≥2)
        0,
        set(),
    ),
]


def main():
    pass_n, fail_n = 0, 0
    print('=' * 78)
    print('P198ej dry-run — BAHL grouping cap + SWIFT-content gate')
    print('=' * 78)

    # ── A. Real job (if data present) ──
    print(f'\n--- A. Real job 7e3699a8 (if available) ---')
    if real_pages:
        bahl, bahl_pgs = simulate_grouping(real_pages)
        print(f'   {len(bahl)} BAHL groups detected:')
        for gid, info in bahl.items():
            print(f'     Group {gid}: pages={info["pages"]} fin.{info["fin"]} = {info["mt"]}')
        # Pages 7-31 should NOT be in any BAHL group
        bad = [p for p in bahl_pgs if p > 6]
        if not bad:
            print('   [OK ] no shipping pages (>6) swept into BAHL group')
            pass_n += 1
        else:
            print(f'   [FAIL] {len(bad)} non-SWIFT pages still in BAHL: {bad[:10]}')
            fail_n += 1
    else:
        print('   [SKIP] real job data not on disk')

    # ── B. Synthetic scenarios ──
    print('\n--- B. Synthetic scenarios ---')
    for i, (name, pages, exp_groups, exp_pages) in enumerate(SCENARIOS, 1):
        bahl, bahl_pgs = simulate_grouping(pages)
        actual_pages = set(bahl_pgs)
        ok = (len(bahl) == exp_groups and actual_pages == exp_pages)
        tag = 'OK ' if ok else 'FAIL'
        print(f'\n[{tag}] Scenario {i}: {name}')
        print(f'        groups: {len(bahl)} (expected {exp_groups})')
        print(f'        BAHL pages: {sorted(actual_pages)}')
        print(f'        expected   : {sorted(exp_pages)}')
        if not ok:
            extra = actual_pages - exp_pages
            missing = exp_pages - actual_pages
            if extra:
                print(f'        EXTRA (false positives): {sorted(extra)}')
            if missing:
                print(f'        MISSING (false negatives): {sorted(missing)}')
        if ok: pass_n += 1
        else:  fail_n += 1

    total = pass_n + fail_n
    print('\n' + '=' * 78)
    print(f'OVERALL: {pass_n}/{total} '
          f'{"OK" if fail_n == 0 else "— failures present"}')
    print('=' * 78)
    return 0 if fail_n == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
