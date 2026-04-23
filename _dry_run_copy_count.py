"""Comprehensive dry-run for P198as copy-count fast-path.

Goal: ensure the auto-PASS fires ONLY when the condition is purely
about copy count, and falls through to LLM for every compound clause.
"""
import re

_COPY_RE = re.compile(
    r'\b(DUPLICATE|TRIPLICATE|QUADRUPLICATE|OCTUPLICATE|COPIES|'
    r'FULL\s+SET|IN\s+\d+\s+ORIG)',
    re.IGNORECASE,
)

_OTHER_RE = re.compile(
    r'\b('
    r'TO\s+(?:THE\s+)?ORDER\s+OF|CONSIGNED\s+TO|CONSIGNEE|CONSIGNOR|'
    r'IN\s+(?:FAVOUR|FAVOR)\s+OF|ENDORS(?:ED|E|EMENT|ING|ABLE)|'
    r'NOTIFY(?:\s|ING)|NOTIFY\s+PARTY|SHIPPER|'
    r'FREIGHT\s+(?:PREPAID|COLLECT|PAYABLE|TO\s+COLLECT)|'
    r'CLEAN\s+(?:SHIPPED\s+)?ON\s+BOARD|CLEAN\s+ON\s+BOARD|SHIPPED\s+ON\s+BOARD|'
    r'MARKED(?:\s|\')|SHOWING\b|MADE\s+OUT\s+TO|ADDRESSED\s+(?:TO|AT)|'
    r'(?:MUST|SHOULD|SHALL)\s+(?:SHOW|STATE|INDICATE|BEAR|CONTAIN|MENTION|APPEAR|CERTIFY|BE\s+SIGNED|BE\s+MADE|BE\s+MARKED|BE\s+ADDRESSED)|'
    r'BEARING\b|STATING\b|CERTIFYING\b|MENTIONING\b|INDICATING\b|'
    r'SIGNED\s+BY|ISSUED\s+BY|'
    r'CHARTER\s+PARTY|CHARTER-PARTY|SHORT\s+FORM|BLANK\s+BACK|'
    r'HOUSE\s+(?:BL|BILL|B/L)|FIATA|NVOCC|INSTITUTE\s+CLASSIFICATION'
    r')\b',
    re.IGNORECASE,
)

def decide(cond):
    cu = cond.upper()
    if not _COPY_RE.search(cu):
        return 'N/A'  # not even a copy-count condition
    return 'FALL-THROUGH' if _OTHER_RE.search(cu) else 'AUTO-PASS'

# ================================================================
# Scenarios organized by category
# ================================================================
scenarios = [
    # ──────── PURE copy-count → AUTO-PASS ────────
    ('PURE #01', 'Commercial Invoice in triplicate.', 'AUTO-PASS'),
    ('PURE #02', 'Packing List in duplicate.', 'AUTO-PASS'),
    ('PURE #03', 'Invoice in quadruplicate.', 'AUTO-PASS'),
    ('PURE #04', 'Full set of Bills of Lading.', 'AUTO-PASS'),
    ('PURE #05', 'BL in 3 originals.', 'AUTO-PASS'),
    ('PURE #06', 'Certificate of Origin in 1 original and 2 copies.', 'AUTO-PASS'),
    ('PURE #07', 'Inspection Certificate in octuplicate.', 'AUTO-PASS'),
    ('PURE #08', 'Beneficiary certificate in 3 copies.', 'AUTO-PASS'),
    ('PURE #09', 'Insurance Policy/Certificate in full set.', 'AUTO-PASS'),
    ('PURE #10', 'Bill of Lading in 2 originals plus 3 non-negotiable copies.', 'AUTO-PASS'),

    # ──────── copy + CONSIGNEE / ENDORSEMENT → FALL-THROUGH ────────
    ('CONS #01', 'Full set of clean shipped on board Marine/Ocean Charter Party Bills of Lading must be made out to the order of Bank Al Habib Ltd., Pakistan.', 'FALL-THROUGH'),
    ('CONS #02', 'BL in 3 originals must be made out to the order of the issuing bank.', 'FALL-THROUGH'),
    ('CONS #03', 'Full set of BLs consigned to ACME CORP.', 'FALL-THROUGH'),
    ('CONS #04', 'Full set of BLs endorsed to the order of ABC Bank.', 'FALL-THROUGH'),
    ('CONS #05', 'BL in duplicate endorsed in blank by the Shipper.', 'FALL-THROUGH'),
    ('CONS #06', 'Full set of BLs in favour of Bank Al Habib.', 'FALL-THROUGH'),

    # ──────── copy + FREIGHT clause → FALL-THROUGH ────────
    ('FRT  #01', 'BL in triplicate marked freight prepaid.', 'FALL-THROUGH'),
    ('FRT  #02', 'Full set of BLs showing freight payable as per charter party.', 'FALL-THROUGH'),
    ('FRT  #03', 'BL in 3 originals marked freight collect.', 'FALL-THROUGH'),
    ('FRT  #04', 'Full set of clean shipped on board BL with freight prepaid.', 'FALL-THROUGH'),

    # ──────── copy + CLEAN/SHIPPED → FALL-THROUGH ────────
    ('CLN  #01', 'Full set of clean shipped on board BLs.', 'FALL-THROUGH'),
    ('CLN  #02', 'BL in 3 copies, clean on board.', 'FALL-THROUGH'),
    ('CLN  #03', 'Full set of shipped on board marine BLs.', 'FALL-THROUGH'),

    # ──────── copy + NOTIFY clause → FALL-THROUGH ────────
    ('NTF  #01', 'BL in 3 originals and notify the Applicant.', 'FALL-THROUGH'),
    ('NTF  #02', 'Full set of BLs marked notify the Applicant.', 'FALL-THROUGH'),
    ('NTF  #03', 'BL in duplicate notifying ABC Co. and Bank XYZ.', 'FALL-THROUGH'),

    # ──────── copy + SIGNING / ISSUER → FALL-THROUGH ────────
    ('SGN  #01', 'Invoice in triplicate signed by Beneficiary.', 'FALL-THROUGH'),
    ('SGN  #02', 'Certificate in duplicate issued by Chamber of Commerce.', 'FALL-THROUGH'),
    ('SGN  #03', 'BL in 3 originals signed by the master.', 'FALL-THROUGH'),

    # ──────── copy + CERTIFYING content → FALL-THROUGH ────────
    ('CRT  #01', 'Commercial Invoice in triplicate must certify goods of Japan origin.', 'FALL-THROUGH'),
    ('CRT  #02', 'Invoice in duplicate certifying merchandise is free of haram.', 'FALL-THROUGH'),
    ('CRT  #03', "Beneficiary's certificate in 3 copies certifying GOODS ARE BRAND NEW.", 'FALL-THROUGH'),

    # ──────── copy + BL TYPE restrictions → FALL-THROUGH ────────
    ('TYP  #01', 'Full set of marine/ocean BLs (not charter party, not short form, not blank back).', 'FALL-THROUGH'),
    ('TYP  #02', 'BL in 3 originals — charter party BL acceptable.', 'FALL-THROUGH'),
    ('TYP  #03', 'Full set Charter Party BLs.', 'FALL-THROUGH'),
    ('TYP  #04', 'BL in duplicate, short form BL not acceptable.', 'FALL-THROUGH'),
    ('TYP  #05', 'Full set BLs, House BL not acceptable.', 'FALL-THROUGH'),
    ('TYP  #06', 'BL in 3 copies, must not show FIATA reference.', 'FALL-THROUGH'),
    ('TYP  #07', 'Full set BLs — NVOCC BL not acceptable.', 'FALL-THROUGH'),

    # ──────── copy + ADDRESSED TO → FALL-THROUGH ────────
    ('ADR  #01', 'Shipment Advice in 3 copies addressed to the Applicant.', 'FALL-THROUGH'),
    ('ADR  #02', 'Courier Receipt in duplicate addressed to Bank Al Habib.', 'FALL-THROUGH'),

    # ──────── copy + MUST SHOW X → FALL-THROUGH ────────
    ('SHW  #01', 'BL in 3 originals must show LC number.', 'FALL-THROUGH'),
    ('SHW  #02', 'Invoice in triplicate must bear NTN No. 1234.', 'FALL-THROUGH'),
    ('SHW  #03', 'Certificate in duplicate must mention HS Code 1201.9000.', 'FALL-THROUGH'),
    ('SHW  #04', 'BL in duplicate must contain vessel IMO number.', 'FALL-THROUGH'),
    ('SHW  #05', 'BL in full set must indicate port of loading and discharge.', 'FALL-THROUGH'),

    # ──────── Non-copy-count (sanity) → N/A ────────
    ('N/A  #01', 'BL must be signed by master.', 'N/A'),
    ('N/A  #02', 'Invoice must show HS code.', 'N/A'),
    ('N/A  #03', 'Insurance certificate for 110% of invoice value.', 'N/A'),

    # ──────── Edge cases ────────
    ('EDG  #01', 'Beneficiary certificate in triplicate.', 'AUTO-PASS'),   # pure copy
    ('EDG  #02', 'Beneficiary certificate in triplicate stating brand new.', 'FALL-THROUGH'),  # has "STATING"
    ('EDG  #03', 'Invoice in duplicate in English.', 'AUTO-PASS'),  # "IN" inside "IN ENGLISH" doesn't match triggers
    ('EDG  #04', 'Insurance Policy in 2 originals covering 110%.', 'AUTO-PASS'),  # "COVERING" not a trigger unless "MUST ..." -- keep simple
    ('EDG  #05', 'BL in 3 originals bearing IMO number.', 'FALL-THROUGH'),  # "BEAR" verb
    ('EDG  #06', 'BL in triplicate, short form not acceptable.', 'FALL-THROUGH'),
    ('EDG  #07', 'BL in duplicate, short form blank-back not acceptable.', 'FALL-THROUGH'),

    # ──────── CONSIGNEE / NOTIFY PARTY / SHIPPER dedicated scenarios ────────
    ('CONS-NF #01', 'BL in 3 originals, consignee to order of Bank XYZ.', 'FALL-THROUGH'),
    ('CONS-NF #02', 'Full set BL, consignee: Bank ABC.', 'FALL-THROUGH'),
    ('CONS-NF #03', 'BL in duplicate with consignor = Beneficiary.', 'FALL-THROUGH'),
    ('CONS-NF #04', 'BL in triplicate, shipper: M/s XYZ Co.', 'FALL-THROUGH'),
    ('CONS-NF #05', 'BL in 3 copies, notify party: ACME CORP.', 'FALL-THROUGH'),
    ('CONS-NF #06', 'BL in duplicate notifying ABC Co. and Bank XYZ.', 'FALL-THROUGH'),
    ('CONS-NF #07', 'BL in triplicate stating brand new goods.', 'FALL-THROUGH'),
    ('CONS-NF #08', 'BL in 3 originals bearing IMO number.', 'FALL-THROUGH'),
    ('CONS-NF #09', 'BL in 3 originals endorsed in blank.', 'FALL-THROUGH'),
    ('CONS-NF #10', 'Full set BLs mentioning invoice number.', 'FALL-THROUGH'),
    ('CONS-NF #11', 'Full set BLs indicating L/C number.', 'FALL-THROUGH'),
]

passed = 0
failed = 0
print(f'{"ID":<10} {"Expected":<15} {"Got":<15} {"Status"}')
print('-' * 100)
for sid, cond, expected in scenarios:
    got = decide(cond)
    ok = got == expected
    if ok: passed += 1
    else: failed += 1
    status = 'OK' if ok else 'FAIL'
    marker = '' if ok else '  <<'
    print(f'{sid:<10} {expected:<15} {got:<15} {status}{marker}  {cond[:80]}')
print()
print(f'TOTAL: {passed}/{len(scenarios)} passed, {failed} failed')
