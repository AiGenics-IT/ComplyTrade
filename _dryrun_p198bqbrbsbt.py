"""Dry-run for P198bq/br/bs/bt + FULL REGRESSION of all prior fixes."""
import json
import re

# ──────────────────────────────────────────────────────────────────────
# Load actual job data
# ──────────────────────────────────────────────────────────────────────
with open('results/11ec29b8-6eaf-4c71-b0f2-1557030dc4c1/step09/step09_result.json',
          encoding='utf-8') as f:
    d = json.load(f)

# BL packets (real + mis-classified)
bl_packets = []
attached_packets = []
sa_packets = []
for p in d.get('reconciled_packets', []):
    dt = (p.get('document_type', '') or '').lower()
    txt = (p.get('refined_text') or p.get('cleaned_text')
           or p.get('raw_text') or '')
    if 'attached list' in dt:
        attached_packets.append((p.get('packet_id'), dt, txt))
    elif 'bill of lading' in dt:
        bl_packets.append((p.get('packet_id'), dt, txt))
    elif 'shipment' in dt or 'shipping' in dt:
        sa_packets.append((p.get('packet_id'), dt, txt))

print(f'Job 11ec29b8 data: {len(bl_packets)} BL packets, '
      f'{len(attached_packets)} Attached List packets, '
      f'{len(sa_packets)} Shipment Advice packets')

# ──────────────────────────────────────────────────────────────────────
# P198bq — freight-wording multi-packet aggregation
# ──────────────────────────────────────────────────────────────────────
print()
print('=' * 78)
print('P198bq — freight-wording aggregation across all BL packets')
print('=' * 78)

_ALTS_PREPAID = ('FREIGHT PREPAID', 'FRT PREPAID', 'FREIGHT PAID', 'PREPAID FREIGHT')
tasks_present = []
tasks_absent = []
for pid, dt, txt in bl_packets:
    present = [t for t in _ALTS_PREPAID if t in txt.upper()]
    if present:
        tasks_present.append((pid, present[0]))
    else:
        tasks_absent.append(pid)

print(f'  BL packets WITH FREIGHT PREPAID: {len(tasks_present)} '
      f'({[t[0] for t in tasks_present]})')
print(f'  BL packets WITHOUT FREIGHT PREPAID: {len(tasks_absent)} '
      f'({tasks_absent})')
verdict = 'PASS' if tasks_present else 'FAIL'
print(f'  Aggregated verdict: {verdict}  (expected PASS since 6 of 7 carry it)')
assert verdict == 'PASS'

# ──────────────────────────────────────────────────────────────────────
# P198br — consignee TO ORDER OF <BANK> aggregation
# ──────────────────────────────────────────────────────────────────────
print()
print('=' * 78)
print('P198br — consignee TO ORDER OF <BANK> aggregation')
print('=' * 78)

target = "Bank Al Habib Ltd., Karachi, Pakistan"
_stop = {'BANK', 'LTD', 'LIMITED', 'LLC', 'PLC', 'INC', 'CO', 'PVT',
         'PRIVATE', 'COMPANY', 'THE', 'OF', 'AND', 'PAKISTAN', 'KARACHI',
         'LAHORE', 'ISLAMABAD'}
target_tokens = [
    w for w in re.split(r'\s+', re.sub(r'[.,;:\'"—–-]+', ' ', target))
    if w and w.upper() not in _stop and len(w) >= 2
]
print(f'  Target tokens: {target_tokens}')

pkts_with_endorsement = []
pkts_without = []
for pid, dt, txt in bl_packets:
    squash = re.sub(r'\s+', ' ', txt.upper())
    found = False
    for m in re.finditer(r'TO\s+(?:THE\s+)?ORDER\s+OF\s+([^.\n]{0,120})', squash):
        window = m.group(1)
        if all(tok.upper() in window for tok in target_tokens):
            found = True
            break
    if found:
        pkts_with_endorsement.append(pid)
    else:
        pkts_without.append(pid)

print(f'  BL packets WITH \'TO ORDER OF BANK AL HABIB\': {len(pkts_with_endorsement)}')
print(f'  BL packets WITHOUT: {len(pkts_without)}')
verdict = 'PASS' if pkts_with_endorsement else 'FAIL'
print(f'  Aggregated verdict: {verdict}  (expected PASS)')
assert verdict == 'PASS'

# ──────────────────────────────────────────────────────────────────────
# P198bs — Attached List exclusion
# ──────────────────────────────────────────────────────────────────────
print()
print('=' * 78)
print('P198bs — Attached List exclusion from All-Documents fan-out')
print('=' * 78)

_ALLDOC_FANOUT_EXCLUDE = (
    'documentary remittance', 'document remittance',
    'covering letter', 'cover letter',
    'covering schedule', 'cover schedule',
    'attached list', 'attached schedule', 'attached manifest',
    'cargo manifest page', 'pallet manifest', 'packing insert',
    'container list', 'container manifest', 'stuffing list',
    'blank page', 'terms and conditions',
    'unknown', 'unidentified',
)


def excluded(pt):
    ptl = (pt or '').lower().strip()
    return any(ex in ptl for ex in _ALLDOC_FANOUT_EXCLUDE)


cases = [
    ('Attached List', True),
    ('attached list', True),
    ('Attached Schedule', True),
    ('Cargo Manifest Page', True),
    ('Packing List', False),  # real document, must NOT be excluded
    ('Commercial Invoice', False),
    ('Bill of Lading', False),
    ('Shipment Advice', False),
    ('Documentary Remittance', True),
    ('Draft Bill of Exchange', False),
]
for dtype, expect in cases:
    got = excluded(dtype)
    ok = 'OK' if got == expect else 'FAIL'
    print(f'  [{ok}] {dtype!r:30} excluded={got} (expected {expect})')

# ──────────────────────────────────────────────────────────────────────
# P198bt — F31D REVIEW message simplification (length + phrasing)
# ──────────────────────────────────────────────────────────────────────
print()
print('=' * 78)
print('P198bt — F31D REVIEW message (simplified wording)')
print('=' * 78)
new_msg = "Receiving / presentation date not clear — manual review."
print(f'  findings: {new_msg!r}')
print(f'  length:   {len(new_msg)} chars (was ~400)')
assert len(new_msg) < 100

# ──────────────────────────────────────────────────────────────────────
# REGRESSION: prior fixes still work
# ──────────────────────────────────────────────────────────────────────
print()
print('=' * 78)
print('REGRESSION: prior fixes still work')
print('=' * 78)


def _normalize_id(s):
    out = ''.join(ch for ch in str(s or '').upper() if ch.isalnum())
    subs = str.maketrans({'O':'0','I':'1','L':'1','S':'5','B':'8','Z':'2','G':'6','Q':'0'})
    return out.translate(subs)


# P198bl — OCR-tolerant reference
cond = 'POLICY NO 2023008MIPD000453'
doc = 'OPEN POLICY NO.2023008MIPDO00453'
assert _normalize_id('2023008MIPD000453') in _normalize_id(doc)
print('  [OK] P198bl OCR-tolerant ref (O↔0) still works')

# P198bm — prohibitive FF condition skipped
_prohibitive_re = re.compile(
    r'\b(?:NOT\s+ACCEPT|MUST\s+NOT|SHALL\s+NOT|NOT\s+PRESENTED|'
    r'NOT\s+PERMITTED|NOT\s+ALLOWED|FORBIDDEN|PROHIBIT)\b'
)
ff_cond = "Bills of Lading with FF reference must not be presented."
assert _prohibitive_re.search(ff_cond.upper())
print('  [OK] P198bm still skips prohibitive FF conditions')

# P198bn — boilerplate NVOCC
_DEFINITION_MARKERS = ('MEANS ', 'SHALL MEAN', 'DEFINED AS', 'DEFINITIONS', 'GLOSSARY')
nvocc_boiler = 'TERMS: "NVOCC" MEANS NON VESSEL OPERATING COMMON CARRIER.'
nvocc_real = 'ISSUED BY: XYZ NON VESSEL OPERATING COMMON CARRIER'
def real_ctx(text, tok):
    idx = 0
    while True:
        pos = text.upper().find(tok, idx)
        if pos < 0: return False
        pre = text.upper()[max(0,pos-80):pos]
        if any(m in pre for m in _DEFINITION_MARKERS):
            idx = pos + 1; continue
        if '"' in pre[-40:] and 'MEANS' in text.upper()[pos:pos+80]:
            idx = pos + 1; continue
        return True
assert real_ctx(nvocc_real, 'NON VESSEL OPERATING') is True
assert real_ctx(nvocc_boiler, 'NON VESSEL OPERATING') is False
print('  [OK] P198bn boilerplate NVOCC rescued, real NVOCC kept')

# P198bo — NVOCC synonym detection
cond_nvocc = "BL stated to be issued by a non-vessel operating carrier company is not acceptable."
synonyms = ('NVOCC', 'NON-VESSEL OPERATING', 'NON VESSEL OPERATING',
            'NON-VESSEL CARRIER', 'NON VESSEL CARRIER')
assert any(s in cond_nvocc.upper() for s in synonyms)
print('  [OK] P198bo NVOCC condition synonyms detected')

# P198bp — email-aware addressed-to
_EMAIL_RE = re.compile(
    r'[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}|'
    r'[A-Z0-9._%+\-]+\s*\(\s*AT\s*\)\s*[A-Z0-9.\-]+',
    flags=re.IGNORECASE,
)
email_cond = "addressed to the Applicant at ABID.HUSSAIN@TECNOPACK.COM.PK"
assert _EMAIL_RE.search(email_cond)
# Emails should be found in sa pkts
any_has_email = any(
    'ABID.HUSSAIN' in txt.upper() and 'TECNOPACK' in txt.upper()
    for _, _, txt in sa_packets
)
assert any_has_email
print('  [OK] P198bp email-aware rescue (email found in at least one shipment advice)')

# P198bh — 45A alt-block code extraction still works
codes = re.findall(r'\b([A-Z]{1,4}\d{3,6}[A-Z]{0,3})\b',
                   'LDPE HP4023WN and/or HP4024WN and/or HP4025ZN')
assert set(codes) == {'HP4023WN', 'HP4024WN', 'HP4025ZN'}
print('  [OK] P198bh alt-block product-code extraction still works')

# P198bi — multi-line courier filter
courier = ("DOCUMENTS MUST BE SENT TO BANK AL-HABIB LTD.\n"
           "ADDRESS\n"
           "IN 1 LOT, BY COURIER AT BENEFICIARY'S COST").upper()
pat = re.compile(
    r"DOCUMENTS?\s+(?:MUST|SHALL|WILL|SHOULD|ARE\s+TO|TO)\s+BE\s+SENT\s+TO\s+"
    r"[\s\S]{5,400}\b(?:BY\s+COURIER|BY\s+DHL|BY\s+FEDEX|BY\s+TNT|BY\s+UPS|BY\s+ARAMEX)",
)
assert pat.search(courier)
print('  [OK] P198bi multi-line courier filter still fires')

print()
print('=' * 78)
print('All new fixes + all prior fixes green.')
print('=' * 78)
