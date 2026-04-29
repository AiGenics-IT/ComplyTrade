"""
P198ek + P198el dry-run — covers the two remaining bugs
the user kept seeing:

P198ek — BL consignee Check B (step14_verification.py:7167-7185)
  After the main deterministic consignee check passes, a SECOND
  check verifies the BL document text contains the required bank
  keywords. The keyword-match was substring-only with no
  whitespace normalization, so "ALHABIB" (LC's compact form)
  never matched "AL HABIB" (BL's spaced form). The BL row got
  flipped PASS->FAIL with "Consignee does not show 'BANK ALHABIB
  LTD'" / "BL shows 'TO ORDER' but LC requires 'TO THE ORDER OF
  BANK ALHABIB LTD'". Now uses compact-match (whitespace +
  punctuation stripped) on both the LC keyword and the document
  text, so all three forms (ALHABIB / AL HABIB / AL-HABIB)
  are equivalent.

P198el — UBL / HBL "Document Arrival Notice" detection
  (step08_shipping_classification.py)
  Pages 1-2 of jobs like b54a95c2 / 4e3d783b / 5417141d are
  issuing-bank discrepancy notifications: header "DOCUMENT
  ARRIVAL NOTICE", body "PLEASE BE ADVISED THAT WE HAVE
  RECEIVED THE ORIGINAL DOCUMENTS FROM ...", followed by a
  DISCREPANCIES list. Step08's VLM force-fits these to
  "Documentary Remittance" because that's the closest doc-type
  in the LC's required-documents list. P198el detects the
  structural signals and keeps this as "Document Arrival
  Notice" so downstream charges-on-DR / presentation-period
  checks don't mis-anchor on it.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')


# ─── P198ek ─ Check B compact-match ─────────────────────────────────


def check_b_passes(_cond_up, _doc_up):
    """Mirror of step14_verification.py:7167 Check B with the
    P198ek compact-match fix. Returns True if the row would PASS,
    False if it would FAIL."""
    if 'ORDER OF' not in _cond_up:
        return True  # check skipped
    _order_m = re.search(
        r'ORDER\s+OF\s+([A-Z][A-Z\s,.\-()]+?)(?:\.|$|\n)', _cond_up)
    if not _order_m:
        return True
    _order_party = _order_m.group(1).strip().rstrip('.,')
    _order_keywords = [w for w in _order_party.split()
                       if len(w) >= 3
                       and w not in ('THE', 'AND', 'LTD', 'LIMITED')]
    if not _order_keywords:
        return True
    _doc_compact = re.sub(
        r'[\s\-_.,;:\'"/\\]+', '', (_doc_up or '').upper())
    _found_kw = 0
    for w in _order_keywords:
        w_compact = re.sub(r'[\s\-_.,;:\'"/\\]+', '', w.upper())
        if w in _doc_up or (w_compact and w_compact in _doc_compact):
            _found_kw += 1
    return _found_kw >= len(_order_keywords) * 0.6


# ─── P198el ─ Document Arrival Notice detection ─────────────────────


def is_arrival_notice(text):
    """Mirror of step08 P198el. Returns True if the page should
    be classified as 'Document Arrival Notice'."""
    u = (text or '').upper()
    has_header = bool(re.search(
        r'(?:^|\n)\s*DOCUMENT(?:S)?\s+ARRIVAL\s+NOTICE\b',
        u, re.MULTILINE))
    has_received = bool(re.search(
        r'(?:RECEIVED\s+THE\s+ORIGINAL\s+DOCUMENTS|'
        r'WE\s+HAVE\s+RECEIVED\s+THE\s+(?:ORIGINAL\s+)?DOCUMENTS)', u))
    has_discrepancies = bool(re.search(
        r'\bDISCREPANC(?:Y|IES)\s+(?:NOTED|FOUND|OBSERVED)\b'
        r'|(?:^|\n)\s*DISCREPANCIES?\s*[:\-]?\s*\n',
        u, re.MULTILINE))
    return has_header and (has_received or has_discrepancies)


# ─── Test data ──────────────────────────────────────────────────────

# Real OCR — page 1 of job b54a95c2 (UBL Document Arrival Notice)
REAL_PAGE_1_b54a95c2 = """80%
20%
Advance Payment
Case
UNITED BANK LIMITED
DOCUMENT ARRIVAL NOTICE
ICEBERG INDUSTRIES 51-D COMMERCIAL
AREA A PHASE II DHA
KARACHI PAKISTAN
DATE : 16 APRIL 2026
LC NO. : 0052ILC083930
IB NO. : 0052IBS097771
PLEASE BE ADVISED THAT WE HAVE RECEIVED THE ORIGINAL DOCUMENTS FROM
HABIB BANK AG ZURICH. DETAILS ARE AS FOLLOWS:
AMOUNT : USD 2,184.000
DRAWN BY : LUCKY GULF STAR INDUSTRIES FZC
DATE OF DRAFT : 16-APR-26
VESSEL : SEASPAN BRIGHTESS V 0101W
SHIPMENT :
HEAT PUMP, INDOOR AND OUTDOOR UNIT/SET/PACKAGES
DOCUMENTS
ORIGINAL
DUPLICATE
+BILLS OF EXCHANGE
2
+COMMERCIAL INVOICES
8
+BILLS OF LADING
3
3
+OTHERS
4
DISCREPANCIES NOTED:
1.COMMERCIAL INVOICE NOT SHOWING INCOTERMS AS PER LC
2.COMMERCIAL INVOICE SHOWING PERFORMA INVOICE NO NOT AS PER LC
3.SHIPMENT ADVICE NOT COMPLIED AS PER LC
4.HS CODE NTN NOT NOT ON BL
5.FREIGHT VALUE MISSING
6.CARRIER NOT IDENTIFIED
7.SIGNING CAPACITY NOT AS PER UCP 600
8.STALE BL PRESENTED WHICH IS NOT ACCEPTABLE
9.FORWARDER BL PRESENTED
10.47A CLAUSE 6 NOT COMPLIED AS PER LC
"""

EK_SCENARIOS = [
    # (name, condition, document_text, expected_pass)
    ('USER\'S CASE: LC "BANK ALHABIB LTD" vs BL "BANK AL HABIB" (with space)',
     'Full set of shipped on board marine/ocean Bills of Lading must '
     'be made out to the order of Bank Alhabib Ltd., Pakistan.',
     'Consignee:\nTO THE ORDER OF BANK AL HABIB LTD., PAKISTAN\n'
     'Notify Party:\nS.K. TRADING CO\nFreight: Prepaid',
     True),
    ('LC "BANK ALHABIB" vs BL "BANK AL-HABIB" (with hyphen)',
     'BLs must be made out to the order of Bank Alhabib Ltd.',
     'TO THE ORDER OF BANK AL-HABIB LTD., TECHNO CITY, KARACHI',
     True),
    ('Reverse — LC "AL HABIB" (with space) vs BL "ALHABIB" (no space)',
     'BL consigned to BANK AL HABIB Ltd, Karachi.',
     'TO THE ORDER OF: BANK ALHABIB LTD KARACHI',
     True),
    ('LC "TO ORDER OF" vs BL "TO THE ORDER OF" (same bank, different wording)',
     'BL consignee must be made out to order of Bank Al Habib Ltd.',
     'CONSIGNEE: TO THE ORDER OF BANK AL HABIB LIMITED, PAKISTAN',
     True),
    ('Different bank — LC UBL vs BL Bank Al Habib (must FAIL)',
     'BLs must be made out to the order of UBL Bank Limited.',
     'TO THE ORDER OF BANK AL HABIB LTD',
     False),
    ('Casing variants',
     'BLs must be made out to the order of bank al habib ltd.',
     'CONSIGNEE: TO THE ORDER OF Bank Al Habib Ltd, Pakistan',
     True),
    ('BL has TO ORDER only, no bank name (must FAIL)',
     'BLs must be made out to the order of Bank Alhabib Ltd.',
     'CONSIGNEE: TO ORDER\nVessel: SHIP\nFreight: Prepaid',
     False),
    ('Faysal Bank match',
     'BLs must be made out to the order of Faysal Bank.',
     'TO ORDER OF FAYSAL BANK PAKISTAN',
     True),
]


EL_SCENARIOS = [
    # (name, text, expected_arrival_notice)
    ('USER\'S CASE: real page 1 of b54a95c2 (UBL Document Arrival Notice)',
     REAL_PAGE_1_b54a95c2,
     True),
    ('Synthetic UBL Document Arrival Notice with discrepancies',
     'UNITED BANK LIMITED\nDOCUMENT ARRIVAL NOTICE\n'
     'PLEASE BE ADVISED THAT WE HAVE RECEIVED THE ORIGINAL DOCUMENTS\n'
     'DISCREPANCIES NOTED:\n1. ABC\n2. DEF',
     True),
    ('Real bank covering schedule (Maybank Documentary Credit Schedule)',
     'Maybank\nDOCUMENTARY CREDIT SCHEDULE\n'
     'WE ENCLOSE THE FOLLOWING DOCUMENTS FOR NEGOTIATION/PAYMENT\n'
     'TOTAL AMOUNT CLAIMED: USD 33,203.85',
     False),
    ('Plain Bill of Lading',
     'PIL\nPACIFIC INTERNATIONAL LINES\nPORT-TO-PORT BILL OF LADING\n'
     'Bill of Lading No. ABC123',
     False),
    ('Document Arrival Notice header but NO discrepancy / received signal',
     'DOCUMENT ARRIVAL NOTICE\nThis is a generic notice.',
     False),
    ('Discrepancies list but NO arrival-notice header',
     'BANK COVERING LETTER\nDISCREPANCIES NOTED:\n1. ABC\n2. DEF',
     False),
    ('SiekML email cover note (should NOT match)',
     'From: SiekML <siekml@samling.com.my>\n'
     'Subject: COVER NOTE NO.2025-12-212-M01001DT00001322\n'
     'Attached doc for your reference',
     False),
]


def main():
    pass_n, fail_n = 0, 0
    print('=' * 78)
    print('P198ek + P198el dry-run')
    print('=' * 78)

    print('\n--- A. P198ek — BL consignee Check B compact-match ---')
    for name, cond, doc, expected in EK_SCENARIOS:
        actual = check_b_passes(cond.upper(), doc.upper())
        ok = (actual == expected)
        tag = 'OK ' if ok else 'FAIL'
        print(f'\n[{tag}] {name}')
        print(f'        cond: {cond[:80]}')
        print(f'        doc : {doc[:80]}')
        print(f'        expected_pass={expected}, actual_pass={actual}')
        if ok: pass_n += 1
        else:  fail_n += 1

    print('\n--- B. P198el — Document Arrival Notice recognition ---')
    for name, text, expected in EL_SCENARIOS:
        actual = is_arrival_notice(text)
        ok = (actual == expected)
        tag = 'OK ' if ok else 'FAIL'
        print(f'\n[{tag}] {name}')
        print(f'        text head: {text[:90]!r}')
        print(f'        expected={expected}, actual={actual}')
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
