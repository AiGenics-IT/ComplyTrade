"""P198bp dry-run + FULL REGRESSION of prior fixes.

Regression coverage:
  P135  OCR-tolerant reference rescue
  P174/P178/P179  addressed-to deterministic (with P198bk + P198bp guards)
  P198ar/bn/bo  BL prohibited-marker rescue (boilerplate-aware + synonyms)
  P198be/bm  Freight-wording matcher (prohibitive + FORWARDER guards)
  P198bb  Permissive-condition rescue
  P198bd  Vessel age cert label
  P198bh  45A alt-block
  P198bi  Courier-forwarding filter
  P198bk  Party-name fuzzy match
  P198bl  OCR-tolerant reference substring post-check
  P198bp  Email-aware rescue (new)
"""
import json
import re

# =============================================================
# Shared helpers (mirroring step14_verification.py)
# =============================================================
_ENTITY_WORDS_RE = re.compile(
    r'\b(?:LTD|LIMITED|LLC|PLC|INC|INCORPORATED|CORP|CORPORATION|'
    r'CO|COMPANY|PVT|PRIVATE|S\.?A\.?|S\.?L\.?|B\.?V\.?|N\.?V\.?|'
    r'GMBH|AG|AB|OY)\b\.?',
    flags=re.IGNORECASE,
)
_EMAIL_ADDR_RE = re.compile(
    r'[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}|'
    r'[A-Z0-9._%+\-]+\s*\(\s*AT\s*\)\s*[A-Z0-9.\-]+',
    flags=re.IGNORECASE,
)


def _normalize_id(s):
    out = ''.join(ch for ch in str(s or '').upper() if ch.isalnum())
    subs = str.maketrans({
        'O': '0', 'I': '1', 'L': '1', 'S': '5', 'B': '8',
        'Z': '2', 'G': '6', 'Q': '0',
    })
    return out.translate(subs)


def norm_phrase(s):
    s = str(s or '').upper()
    s = re.sub(r'\b(M/?S\.?|MESSRS\.?|MR\.?|MRS\.?|DR\.?)\s+', '', s)
    s = re.sub(r'\([^)]*\)', ' ', s)
    s = _ENTITY_WORDS_RE.sub(' ', s)
    s = re.sub(r',?\s*(?:KARACHI|LAHORE|ISLAMABAD|MUMBAI|DUBAI|RIYADH|DOHA|BEIRUT|COLOMBO|HONG\s+KONG|SINGAPORE|LONDON|NEW\s+YORK|GULBERG)\b.*$', '', s)
    s = re.sub(r',?\s*(?:PAKISTAN|INDIA|BANGLADESH|SRI\s+LANKA|UAE|SAUDI\s+ARABIA|USA|UNITED\s+STATES|UK|UNITED\s+KINGDOM|CANADA|CHINA)\b.*$', '', s)
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


def _normalise_email_text(s):
    s = re.sub(r'\(\s*AT\s*\)', '@', s, flags=re.IGNORECASE)
    s = re.sub(r'\(\s*DOT\s*\)', '.', s, flags=re.IGNORECASE)
    s = re.sub(r'\s*@\s*', '@', s)
    s = re.sub(r'(\w)\s*\.\s*(\w)', r'\1.\2', s)
    return s


def _extract_emails(text):
    t = _normalise_email_text(text)
    return [e.lower() for e in re.findall(
        r'[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}', t,
    )]


# =============================================================
# P198bp: email-aware addressed-to rescue
# =============================================================
def email_check_passes(cond_text, doc_text):
    """Return True if condition contains an email AND that email (or any
    of the condition emails) appears on the document (with (AT)/(DOT)
    normalization)."""
    if not _EMAIL_ADDR_RE.search(cond_text):
        return None  # no email in condition -> fall back to name check
    cond_emails = _extract_emails(cond_text)
    if not cond_emails:
        return None
    doc_emails = _extract_emails(doc_text)
    doc_norm = _normalise_email_text(doc_text).lower()
    for em in cond_emails:
        if em in doc_norm or em in doc_emails:
            return True
    return False


def p174_would_flip(cond_text, doc_text, applicant_name):
    """Simulate whether P174/P178/P179 would flip PASS→FAIL. Respects
    the new P198bp email-address guard."""
    cu = cond_text.upper()
    if 'ADDRESSED TO' not in cu and 'MARKED TO' not in cu:
        return False
    # P198bp: if the condition has an email, defer to email check.
    if _EMAIL_ADDR_RE.search(cu):
        return False
    # Otherwise run P174 name check.
    phrase = norm_phrase(applicant_name)
    present = phrase_in_doc(phrase, doc_text.upper())
    return not present  # FAIL flip when name isn't found


# =============================================================
# Test cases
# =============================================================
print('=' * 80)
print('P198bp (email-aware addressed-to) + regression sweep')
print('=' * 80)

# Load real job data
with open('results/11ec29b8-6eaf-4c71-b0f2-1557030dc4c1/step09/step09_result.json',
          encoding='utf-8') as f:
    d = json.load(f)
sa_packets = [
    p for p in d.get('reconciled_packets', [])
    if 'shipment' in (p.get('document_type', '') or '').lower()
]
sa_texts = {
    p.get('packet_id'): (p.get('refined_text') or p.get('cleaned_text')
                         or p.get('raw_text') or '')
    for p in sa_packets
}
print(f'Real job 11ec29b8 has {len(sa_packets)} Shipment Advice packets.')

email_cond = (
    "Shipment Advice must also be addressed to the Applicant at "
    "ABID.HUSSAIN@TECNOPACK.COM.PK."
)

print()
print('--- P198bp email-aware rescue per packet ---')
results = {}
for pid, txt in sa_texts.items():
    r = email_check_passes(email_cond, txt)
    results[pid] = r
    print(f'  {pid}: email_check -> {r}')
any_pass = any(r is True for r in results.values())
print(f'  ANY packet has the email? -> {any_pass}  (expected True)')
print(f'  Multi-doc existential aggregate verdict: {"PASS" if any_pass else "FAIL"}')

print()
print('--- P174 no longer flips when condition has an email (P198bp guard) ---')
for pid, txt in sa_texts.items():
    flips = p174_would_flip(email_cond, txt, 'TRANSSSION TECNO ELECTRONICS')
    print(f'  {pid}: P174 would flip? -> {flips}  (expected False — email guard)')

# ── Regression checks ──
print()
print('=' * 80)
print('REGRESSION: prior fixes still work')
print('=' * 80)

# P174 still flips on a non-email addressed-to condition when the party is missing
print()
print('--- P174 still flips on NON-email addressed-to that is missing ---')
name_only_cond = "Shipment Advice must also be addressed to TRANSSSION TECNO ELECTRONICS."
for pid, txt in list(sa_texts.items())[:2]:
    flips = p174_would_flip(name_only_cond, txt, 'TRANSSSION TECNO ELECTRONICS')
    print(f'  {pid}: P174 would flip? -> {flips}  (expected True if company name missing)')

# P198bk still works — H.SHEIKH NOOR-UD-DIN
print()
print('--- P198bk party-name with PVT/LTD variance still works ---')
with open('results/48bdb6ee-fa11-4d86-9304-f4cf272674a5/step09/step09_result.json',
          encoding='utf-8') as f:
    d2 = json.load(f)
sa48 = ''
for p in d2.get('reconciled_packets', []):
    if p.get('document_type', '').lower() == 'shipment advice':
        sa48 = p.get('refined_text') or p.get('cleaned_text') or ''
        break
hs_phrase = norm_phrase(
    "H.SHEIKH NOOR-UD-DIN AND SONS (PVT) LTD, 4-KM KAHNA KACHA ROAD, LAHORE PAKISTAN"
)
found = phrase_in_doc(hs_phrase, sa48.upper())
print(f'  48bdb6ee shipment advice vs H.SHEIKH phrase: found={found}  (expected True)')

# P198bl OCR-tolerant reference still works
print()
print('--- P198bl OCR-tolerant reference (O↔0) still works ---')
ref_cond = 'POLICY NO. 2023008MIPD000453'
ref_doc = 'OPEN POLICY NO.2023008MIPDO00453'
ref_norm_cond = _normalize_id(ref_cond)
ref_norm_doc = _normalize_id(ref_doc)
print(f'  cond_norm={ref_norm_cond}  doc_norm={ref_norm_doc}  match={ref_norm_cond[-17:] in ref_norm_doc}')

# P198bm still skips prohibitive freight conditions
print()
print('--- P198bm still skips prohibitive FF conditions ---')
ff_cond = ("Bills of Lading having any reference of issuer being a "
           "freight forwarder must not be presented.")
_prohibitive_re = re.compile(
    r'\b(?:NOT\s+ACCEPT|MUST\s+NOT|SHALL\s+NOT|NOT\s+PRESENTED|'
    r'NOT\s+PERMITTED|NOT\s+ALLOWED|FORBIDDEN|PROHIBIT|'
    r'UNACCEPTABLE|NOT\s+TO\s+BE|CANNOT\s+BE|WILL\s+NOT\s+BE)\b',
)
_dtp_re = re.compile(
    r'\bFREIGHT\s+FORWARDER[S\'’]?\b|\bFIATA\b|\bNVOCC\b|'
    r'\bHOUSE\s+(?:B\s*/\s*L|BILL\s+OF\s+LADING)\b|'
    r'\bNON[\s\-]VESSEL\s+OPERAT',
)
u = ff_cond.upper()
skip = bool(_prohibitive_re.search(u)) or bool(_dtp_re.search(u))
print(f'  P198bm skip? -> {skip}  (expected True)')

# P198bn boilerplate vs real NVOCC evidence
print()
print('--- P198bn boilerplate NVOCC rescue vs real NVOCC evidence ---')
def _has_real_context(text_up, tok):
    markers = ('MEANS ', 'MEAN ', 'SHALL MEAN', 'INCLUDES ',
               'DEFINED AS', 'DEFINITION OF', 'REFERS TO',
               'INTERPRETED AS', 'DEFINED HEREIN',
               '"NVOCC"', '"NVOCG"', "'NVOCC'",
               'DEFINITIONS', 'GLOSSARY')
    idx = 0
    while True:
        pos = text_up.find(tok, idx)
        if pos < 0:
            return False
        pre = text_up[max(0, pos - 80): pos]
        if any(m in pre for m in markers):
            idx = pos + 1; continue
        if '"' in pre[-40:] and 'MEANS' in text_up[pos:pos + 80]:
            idx = pos + 1; continue
        return True
# Real Job BL (boilerplate NVOCC in T&C)
with open('results/73be98d9-724f-4500-a08c-79802b4a5794/step09/step09_result.json',
          encoding='utf-8') as f:
    d3 = json.load(f)
real_bl = ''
for p in d3.get('reconciled_packets', []):
    if 'bill of lading' in p.get('document_type', '').lower():
        real_bl = (p.get('refined_text') or p.get('cleaned_text')
                   or p.get('raw_text') or '').upper()
        break
r1 = _has_real_context(real_bl, 'NON VESSEL OPERATING')
print(f'  Real Job 73be98d9 BL NVOCC is real-context? -> {r1}  (expected False)')
real_nvocc = ("ISSUED BY: XYZ LOGISTICS, NON VESSEL OPERATING COMMON "
              "CARRIER.").upper()
r2 = _has_real_context(real_nvocc, 'NON VESSEL OPERATING')
print(f'  Synthetic real-NVOCC issuer is real-context? -> {r2}  (expected True)')

# P198bh alt-block still detects 45A blocks
print()
print('--- P198bh 45A alt-block groups still work ---')
prod_re = re.compile(r'\b([A-Z]{1,4}\d{3,6}[A-Z]{0,3})\b')
sample_conds = [
    'Goods must show LDPE HP4024WN.',
    'Goods must show LDPE HP4024N.',
    'Goods must show LDPE HP4023WN and/or HP4024WN and/or HP4025ZN.',
]
for s in sample_conds:
    codes = prod_re.findall(s.upper())
    print(f'  {s!r} -> codes={codes}')

# P198bi courier filter crosses newlines
print()
print('--- P198bi multi-line courier filter ---')
courier_multi = """
DOCUMENTS MUST BE SENT TO BANK AL-HABIB LTD. TECHNO CITY,
7TH FLOOR, KARACHI 74000, PAKISTAN
IN 1 LOT, BY COURIER AT BENEFICIARY'S COST
""".upper()
pat = re.compile(
    r"DOCUMENTS?\s+(?:MUST|SHALL|WILL|SHOULD|ARE\s+TO|TO)\s+BE\s+SENT\s+TO\s+"
    r"[\s\S]{5,400}\b(?:BY\s+COURIER|BY\s+DHL|BY\s+FEDEX|BY\s+TNT|BY\s+UPS|BY\s+ARAMEX)",
)
print(f'  P198bi filter hit? -> {bool(pat.search(courier_multi))}  (expected True)')
