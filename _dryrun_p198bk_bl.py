"""Dry-run harness for P198bk/bl/bi fixes. Run with:
    python _dryrun_p198bk_bl.py
"""
import json
import re
import sys

sys.path.insert(0, '.')
from steps.step14_verification import _normalize_id

# Reconstruct the P198bk helpers exactly as they live in step14_verification.py
_ENTITY_WORDS_RE = re.compile(
    r'\b(?:LTD|LIMITED|LLC|PLC|INC|INCORPORATED|CORP|CORPORATION|'
    r'CO|COMPANY|PVT|PRIVATE|S\.?A\.?|S\.?L\.?|B\.?V\.?|N\.?V\.?|'
    r'GMBH|AG|AB|OY)\b\.?',
    flags=re.IGNORECASE,
)


def norm_phrase(s):
    s = str(s or '').upper()
    s = re.sub(r'\b(M/?S\.?|MESSRS\.?|MR\.?|MRS\.?|DR\.?)\s+', '', s)
    s = re.sub(r'\([^)]*\)', ' ', s)
    s = _ENTITY_WORDS_RE.sub(' ', s)
    s = re.sub(
        r',?\s*(?:KARACHI|LAHORE|ISLAMABAD|MUMBAI|DUBAI|RIYADH|DOHA|'
        r'BEIRUT|COLOMBO|HONG\s+KONG|SINGAPORE|LONDON|NEW\s+YORK|'
        r'GULBERG)\b.*$',
        '', s,
    )
    s = re.sub(
        r',?\s*(?:PAKISTAN|INDIA|BANGLADESH|SRI\s+LANKA|UAE|SAUDI\s+ARABIA|'
        r'USA|UNITED\s+STATES|UK|UNITED\s+KINGDOM|CANADA|CHINA)\b.*$',
        '', s,
    )
    s = re.sub(r'[.,;:/\\"\'—–-]+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def phrase_in_doc(name_phrase, doc_up):
    if not name_phrase or not doc_up:
        return False
    _dn = re.sub(r'[^A-Z0-9]+', ' ', doc_up).strip()
    _dn = _ENTITY_WORDS_RE.sub(' ', _dn)
    _dn = re.sub(r'\s+', ' ', _dn)
    words = [w for w in name_phrase.split() if w]
    if not words:
        return False
    if len(words) == 1:
        return bool(re.search(r'\b' + re.escape(name_phrase) + r'\b', _dn))
    gap = r'(?:\s+\S+){0,2}\s+'
    pat = r'\b' + gap.join(re.escape(w) for w in words) + r'\b'
    if re.search(pat, _dn):
        return True
    distinct = [w for w in words if len(w) >= 3]
    if not distinct:
        return False
    hits = sum(1 for w in distinct if re.search(r'\b' + re.escape(w) + r'\b', _dn))
    return (hits / len(distinct)) >= 0.85


# ── Party name scenarios ──
with open('results/48bdb6ee-fa11-4d86-9304-f4cf272674a5/step09/step09_result.json',
          encoding='utf-8') as f:
    d = json.load(f)
sa_text = ''
for pkt in d.get('reconciled_packets', []):
    if pkt.get('document_type', '').lower() == 'shipment advice':
        sa_text = (
            pkt.get('refined_text')
            or pkt.get('cleaned_text')
            or pkt.get('raw_text')
            or ''
        )
        break

doc_with_only_name_variants = """
TO APPLICANT COMPANY LTD
Some other address
"""

doc_with_wrong_party = """
TO UBL INSURERS LIMITED, OFFICE 501, LAHORE PAKISTAN
Some other business
"""

party_cases = [
    # (label, expected_target, doc_text, expected_present)
    ("SHOULD PASS: party present (real data)",
     "H.SHEIKH NOOR-UD-DIN AND SONS (PVT) LTD, 4-KM KAHNA KACHA ROAD, LAHORE PAKISTAN",
     sa_text, True),
    ("SHOULD PASS: minor variant (no PVT, different punct)",
     "H SHEIKH NOOR-UD-DIN AND SONS LTD, 4 KM KAHNA KACHA ROAD",
     sa_text, True),
    ("SHOULD FAIL: different company entirely",
     "M/S ACME TEXTILES MILL PRIVATE LIMITED, KARACHI",
     sa_text, False),
    ("SHOULD FAIL: partial-name hit that shouldn't PASS",
     "H.SHEIKH FAMILY TRUST, 99-B HILL STREET, ISLAMABAD",
     sa_text, False),
    ("SHOULD FAIL: party missing in doc",
     "H.SHEIKH NOOR-UD-DIN AND SONS (PVT) LTD",
     doc_with_wrong_party, False),
    ("SHOULD PASS: SABIC applicant (simpler name)",
     "SAUDI BASIC INDUSTRIES CORPORATION",
     sa_text, True),
    ("SHOULD FAIL: SABIC needle in a doc that doesn't mention it",
     "SAUDI BASIC INDUSTRIES CORPORATION",
     doc_with_wrong_party, False),
]

print("=== P198bk party-name fuzzy match ===")
pass_cnt = 0
for label, target, doc, expected in party_cases:
    phrase = norm_phrase(target)
    actual = phrase_in_doc(phrase, doc.upper())
    ok = 'OK' if actual == expected else 'FAIL'
    if ok == 'OK':
        pass_cnt += 1
    print(f'  [{ok}] {label} -> {actual} (expected {expected})')
    print(f'         phrase={phrase!r}')
print(f'  {pass_cnt}/{len(party_cases)} passed')

# ── Policy / reference number (P198bl) ──
print()
print("=== P198bl OCR-tolerant reference substring ===")
ref_cases = [
    # (cond_text, doc_text, expected PASS)
    ("Shipment Advice must reference Open Policy No. 2023008MIPD000453.",
     sa_text, True),  # OCR variant O<->0
    ("Shipment Advice must reference Open Policy No. 2023008MIPDO00453.",
     sa_text, True),  # exact match
    ("Shipment Advice must reference Open Policy No. 9999999XYZ0000.",
     sa_text, False),  # completely different needle
    ("Shipment Advice must reference Open Policy No. 2023008MIPD555555.",
     sa_text, False),  # wrong tail
    ("Shipment Advice must reference Open Policy No.MIPD000453.",
     sa_text, True),   # partial but OCR-foldable
]
pass_cnt = 0
for cond, doc, expected in ref_cases:
    m = re.search(r'POLICY\s+NO\.?\s*([A-Z0-9\-/]+)', cond.upper())
    ref_num = (m.group(1) if m else '').strip()
    ref_norm = _normalize_id(ref_num)
    doc_up = doc.upper()
    doc_norm = _normalize_id(doc_up)
    ref_plain = ref_num.replace('-', '')
    doc_plain = doc_up.replace('-', '')
    found = (
        ref_num in doc_up
        or ref_plain in doc_plain
        or (len(ref_norm) >= 4 and ref_norm in doc_norm)
    )
    ok = 'OK' if found == expected else 'FAIL'
    if ok == 'OK':
        pass_cnt += 1
    print(f'  [{ok}] cond_ref={ref_num!r} norm={ref_norm!r} found={found} expected={expected}')
print(f'  {pass_cnt}/{len(ref_cases)} passed')

# ── 47A-10 courier instruction multi-line filter (P198bi) ──
print()
print("=== P198bi courier-forwarding filter (multi-line) ===")
patterns = [
    r"DOCUMENTS?\s+(?:MUST|SHALL|WILL|SHOULD|ARE\s+TO|TO)\s+BE\s+SENT\s+TO\s+[\s\S]{5,400}\b(?:BY\s+COURIER|BY\s+DHL|BY\s+FEDEX|BY\s+TNT|BY\s+UPS|BY\s+ARAMEX)",
    r"DOCUMENTS?[\s\S]{0,300}(?:BY\s+COURIER|BY\s+DHL|BY\s+FEDEX|BY\s+TNT|BY\s+UPS|BY\s+ARAMEX)[\s\S]{0,200}AT\s+(?:BENEFICIARY|SELLER|APPLICANT|BUYER)(?:'S|S)?\s+COST",
    r"AT\s+(?:BENEFICIARY|SELLER|APPLICANT|BUYER)(?:'S|S)?\s+COST[\s\S]{0,200}(?:COURIER|DHL|FEDEX)",
]


def matches_filter(text):
    u = text.upper()
    return any(bool(re.search(p, u)) for p in patterns)


courier_cases = [
    # (label, clause_text, expected filter-hit)
    ("SHOULD FILTER: actual 47A-10 multi-line",
     """DOCUMENTS MUST BE SENT TO BANK AL-HABIB LTD. TECHNO CITY, 7TH FLOOR, CORPORATE TOWER, HASRAT
MOHANI ROAD, KARACHI-74000, PAKISTAN
IN 1 LOT, BY COURIER AT BENEFICIARY'S COST""", True),
    ("SHOULD FILTER: single-line named bank",
     "DOCUMENTS MUST BE SENT TO BANK AL-HABIB LTD BY COURIER AT BENEFICIARY'S COST",
     True),
    ("SHOULD FILTER: DHL to issuing bank",
     "ORIGINAL DOCUMENTS TO BE SENT TO THE ISSUING BANK BY DHL AT BENEFICIARYS COST",
     True),
    ("SHOULD NOT FILTER: real requirement on a document",
     "Commercial Invoice must show H.S. CODE 1511.9020",
     False),
    ("SHOULD NOT FILTER: Courier Receipt content check",
     "Courier Receipt must show BL number and applicant name",
     False),
    ("SHOULD NOT FILTER: generic clause about beneficiary",
     "Beneficiary certificate must confirm shelf life 9 months",
     False),
    ("SHOULD FILTER: FEDEX wording, inverted order",
     "AT BENEFICIARY'S COST ALL DOCUMENTS SHALL BE DISPATCHED BY FEDEX",
     True),
]
pass_cnt = 0
for label, clause, expected in courier_cases:
    got = matches_filter(clause)
    ok = 'OK' if got == expected else 'FAIL'
    if ok == 'OK':
        pass_cnt += 1
    print(f'  [{ok}] {label} -> filter_hit={got} (expected {expected})')
print(f'  {pass_cnt}/{len(courier_cases)} passed')
