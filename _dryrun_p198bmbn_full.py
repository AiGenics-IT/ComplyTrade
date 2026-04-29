"""Full-coverage dry-run for P198bm (freight-wording guard) and P198bn
(boilerplate-aware BL prohibited-marker rescue). Runs both layers on a
large set of real-world-ish scenarios."""
import json
import re

# ── P198bm: freight-wording matcher (step14 post-check) ──
_key_matchers = [
    ('FREIGHT PAYABLE AS PER CHARTER PARTY',
     re.compile(r'\bFREIGHT\s+PAYABLE\s+AS\s+PER\s+CHARTER\s+PART[YI]\b')),
    ('FREIGHT PAYABLE AT DESTINATION',
     re.compile(r'\bFREIGHT\s+PAYABLE\s+AT\s+DESTINATION\b|\bFREIGHT\s+PAYABLE\s+AT\s+(?:THE\s+)?PORT\s+OF\s+DISCHARGE\b')),
    ('FREIGHT PREPAID',
     re.compile(r'\bFREIGHT\s+PREPAID\b|\bPREPAID\s+FREIGHT\b|\bFREIGHT\s+PAID\b')),
    ('FREIGHT COLLECT',
     re.compile(r'\bFREIGHT\s+COLLECT\b|\bCOLLECT\s+FREIGHT\b|\bFREIGHT\s+TO\s+COLLECT\b')),
    ('FREIGHT FORWARD',
     re.compile(r'\bFREIGHT\s+FORWARD\b(?!ER|ERS|ING|ED)|\bFREIGHT\s+TO\s+BE\s+FORWARDED\b')),
    ('FREIGHT PAYABLE',
     re.compile(r'\bFREIGHT\s+PAYABLE\b')),
]
_prohibitive_re = re.compile(
    r'\b(?:NOT\s+ACCEPT|MUST\s+NOT|SHALL\s+NOT|NOT\s+PRESENTED|'
    r'NOT\s+PERMITTED|NOT\s+ALLOWED|FORBIDDEN|PROHIBIT|'
    r'UNACCEPTABLE|NOT\s+TO\s+BE|CANNOT\s+BE|WILL\s+NOT\s+BE)\b',
)
_doc_type_prohibition_re = re.compile(
    r'\bFREIGHT\s+FORWARDER[S\'’]?\b|\bFIATA\b|\bNVOCC\b|'
    r'\bHOUSE\s+(?:B\s*/\s*L|BILL\s+OF\s+LADING)\b|'
    r'\bNON[\s\-]VESSEL\s+OPERAT',
)


def classify_bm(cond):
    u = cond.upper()
    if 'FREIGHT' not in u:
        return ('SKIP', 'no-freight-keyword')
    if _prohibitive_re.search(u):
        return ('SKIP', 'prohibitive')
    if _doc_type_prohibition_re.search(u):
        return ('SKIP', 'doc-type-prohibition')
    for k, p in _key_matchers:
        if p.search(u):
            return ('APPLY', k)
    return ('SKIP', 'no-matching-key')


# ── P198bn: boilerplate-aware prohibited-token check ──
_BL_PROHIB_TOKENS = {
    'FIATA': ('FIATA',),
    'NVOCC': ('NVOCC', 'NON-VESSEL OPERATING',
              'NON VESSEL OPERATING', 'NON-VESSEL CARRIER'),
    'HOUSE': ('HOUSE BILL OF LADING', 'HOUSE B/L',
              'HOUSE BL', 'HBL', 'HAWB'),
    'FORWARDER': (
        "FREIGHT FORWARDER'S BILL",
        'FREIGHT FORWARDER BILL',
        "FORWARDER'S BILL OF LADING",
        'FORWARDER BILL OF LADING',
        'ISSUED BY FREIGHT FORWARDER',
        'AS FREIGHT FORWARDER',
        'IATA/CASS', 'FMC LICENSE',
    ),
}
_DEFINITION_MARKERS = (
    'MEANS ', 'MEAN ', 'SHALL MEAN', 'INCLUDES ',
    'DEFINED AS', 'DEFINITION OF', 'REFERS TO',
    'INTERPRETED AS', 'DEFINED HEREIN',
    '"NVOCC"', '"NVOCG"', "'NVOCC'",
    'DEFINITIONS', 'GLOSSARY',
)


def has_real_context_match(text_up, tok):
    idx = 0
    while True:
        pos = text_up.find(tok, idx)
        if pos < 0:
            return False
        pre = text_up[max(0, pos - 80): pos]
        if any(m in pre for m in _DEFINITION_MARKERS):
            idx = pos + 1
            continue
        if '"' in pre[-40:] and 'MEANS' in text_up[pos:pos + 80]:
            idx = pos + 1
            continue
        return True


def run_p198bn(cond, bl_text, comp='FAIL'):
    """Simulate P198ar+P198bn on a FAIL row. Returns (verdict, reason)."""
    if comp != 'FAIL':
        return (comp, 'not-FAIL-skip')
    u = cond.upper()
    # Must be prohibitive
    if not any(m in u for m in (
        'NOT ACCEPTABLE', 'NOT PERMITTED', 'NOT ALLOWED',
        'MUST NOT', 'UNACCEPTABLE', 'SHALL NOT',
        'WILL NOT', 'NOT BE ACCEPT', 'MUST NOT BE PRESENT',
        'PROHIBIT',
    )):
        return ('FAIL', 'not-prohibitive')
    # Which prohibitions does the condition name? P198bn — match by
    # condition synonyms, not just the BL-token key.
    _COND_SYNONYMS = {
        'FIATA': ('FIATA',),
        'NVOCC': ('NVOCC','NON-VESSEL OPERATING','NON VESSEL OPERATING',
                  'NON-VESSEL CARRIER','NON VESSEL CARRIER'),
        'HOUSE': ('HOUSE B/L','HOUSE BILL OF LADING','HOUSE BL','HBL'),
        'FORWARDER': ('FREIGHT FORWARDER',"FORWARDER'S",'FORWARDERS',
                      'FORWARDER BILL','FORWARDER BL',
                      'ISSUED BY FREIGHT FORWARDER'),
    }
    named = []
    for k, syns in _COND_SYNONYMS.items():
        if any(s in u for s in syns) and k not in named:
            named.append(k)
    if not named:
        return ('FAIL', 'no-named-prohibition')
    bl_up = bl_text.upper()
    tokens_present = []
    tokens_checked = []
    for key in named:
        for tok in _BL_PROHIB_TOKENS.get(key, ()):
            tokens_checked.append(tok)
            if tok in bl_up and has_real_context_match(bl_up, tok):
                tokens_present.append(tok)
    if tokens_present:
        return ('FAIL', f'prohibited-tokens-present={tokens_present}')
    if not tokens_checked:
        return ('FAIL', 'no-tokens-checked')
    return ('PASS', f'all-{len(tokens_checked)}-tokens-absent-or-boilerplate')


# ============================================================
# TEST CASES
# ============================================================
# Each case: (label, condition_text, bl_text, expected_verdict)
# condition_text is the LC clause as step12 would emit it
# bl_text is the actual BL document text

# Real data from job 73be98d9
with open('results/73be98d9-724f-4500-a08c-79802b4a5794/step09/step09_result.json',
          encoding='utf-8') as f:
    d = json.load(f)
real_bl = ''
for pkt in d.get('reconciled_packets', []):
    if 'bill of lading' in pkt.get('document_type', '').lower():
        real_bl = (pkt.get('refined_text') or pkt.get('cleaned_text')
                   or pkt.get('raw_text') or '')
        break

# Sample BL textures
ff_signed_bl = """
BILL OF LADING
SHIPPER: ACME TEXTILES
ISSUER: GLOBAL LOGISTICS LLC
SIGNED AS FREIGHT FORWARDER
FIATA MEMBER
"""

nvocc_real_bl = """
BILL OF LADING ISSUED BY XYZ NON-VESSEL OPERATING COMMON CARRIER.
LICENSED UNDER FMC LICENSE NO. 123.
"""

house_bl = """
HOUSE BILL OF LADING NO. HBL-001
ISSUED BY A LOGISTICS COMPANY
"""

plain_maersk_bl = """
MAERSK LINE
BILL OF LADING NO. MAEU1234567
SIGNED AS AGENT FOR AND ON BEHALF OF THE MASTER
TERMS AND CONDITIONS:
"NVOCC" MEANS NON VESSEL OPERATING COMMON CARRIER.
"FIATA" MEANS THE INTERNATIONAL FEDERATION OF FREIGHT FORWARDERS ASSOCIATIONS.
"""

short_form_bl_only_def = """
TERMS OF CARRIAGE
"HOUSE BILL OF LADING" MEANS A BILL OF LADING ISSUED BY A FREIGHT FORWARDER
ACTING AS A NON-VESSEL OPERATING COMMON CARRIER.
ACTUAL CARRIER: CMA CGM (ASIA PACIFIC) PTE LTD
"""

ff_prohibition_cond = (
    "Presentation of freight forwarder's Bill of Lading (i.e. Bills of "
    "Lading signed in the capacity of freight forwarder) or Bills of "
    "Lading showing words like FIATA or having any reference of issuer "
    "being a freight forwarder or stated to be issued by a non-vessel "
    "operating carrier company is not acceptable."
)
fiata_only_cond = "Bills of Lading showing words like FIATA are not acceptable."
nvocc_only_cond = "BL stated to be issued by a non-vessel operating carrier company is not acceptable."
house_only_cond = "House B/L is not acceptable."
positive_freight_cond = "BL must be marked FREIGHT PREPAID."
positive_forward_cond = "Freight to be forwarded as per charter party."

cases = [
    # ── Real job 73be98d9 BL (Pacific International Lines + glossary NVOCC) ──
    ("[REAL] Job 73be98d9 BL vs FF/FIATA/NVOCC prohibition",
     ff_prohibition_cond, real_bl, 'PASS'),
    ("[REAL] Job 73be98d9 BL vs FIATA-only prohibition",
     fiata_only_cond, real_bl, 'PASS'),
    ("[REAL] Job 73be98d9 BL vs NVOCC-only prohibition",
     nvocc_only_cond, real_bl, 'PASS'),
    ("[REAL] Job 73be98d9 BL vs HOUSE B/L prohibition",
     house_only_cond, real_bl, 'PASS'),

    # ── Synthetic: legit Maersk BL with glossary only ──
    ("[SYN] Maersk BL (glossary only) vs FF prohibition",
     ff_prohibition_cond, plain_maersk_bl, 'PASS'),
    ("[SYN] Maersk BL vs FIATA-only prohibition",
     fiata_only_cond, plain_maersk_bl, 'PASS'),
    ("[SYN] Maersk BL vs NVOCC-only prohibition",
     nvocc_only_cond, plain_maersk_bl, 'PASS'),

    # ── Synthetic: legit CMA CGM BL with T&C definitions ──
    ("[SYN] CMA CGM BL (T&C definitions) vs FF prohibition",
     ff_prohibition_cond, short_form_bl_only_def, 'PASS'),

    # ── Synthetic: BL actually signed as FF (must stay FAIL) ──
    ("[SYN] FF-signed BL vs FF prohibition",
     ff_prohibition_cond, ff_signed_bl, 'FAIL'),
    ("[SYN] FF-signed BL vs FIATA prohibition",
     fiata_only_cond, ff_signed_bl, 'FAIL'),

    # ── Synthetic: real NVOCC BL (must stay FAIL) ──
    ("[SYN] Real NVOCC BL vs NVOCC prohibition",
     nvocc_only_cond, nvocc_real_bl, 'FAIL'),
    ("[SYN] Real NVOCC BL vs FF prohibition",
     ff_prohibition_cond, nvocc_real_bl, 'FAIL'),

    # ── Synthetic: actual House BL (must stay FAIL) ──
    ("[SYN] House BL vs HOUSE prohibition",
     house_only_cond, house_bl, 'FAIL'),
]

print("=" * 80)
print("P198bm + P198bn combined dry-run — BL freight-forwarder prohibition logic")
print("=" * 80)
print()

passed = 0
for label, cond, bl, expected in cases:
    # First — does P198bm correctly skip the freight-wording matcher?
    bm_action, bm_detail = classify_bm(cond)
    bm_skip = (bm_action == 'SKIP')

    # Then — assume the LLM said FAIL (worst case); does P198bn rescue?
    bn_verdict, bn_reason = run_p198bn(cond, bl, comp='FAIL')

    # Combined: if P198bm skips AND P198bn rescues -> PASS
    # if P198bm skips AND P198bn doesn't rescue -> FAIL (real prohibited)
    # if P198bm applies -> it's a positive freight-wording check (not our case here)
    actual = bn_verdict if bm_skip else '(P198be applies, not P198bn path)'
    ok = 'OK' if actual == expected else 'FAIL'
    if ok == 'OK':
        passed += 1
    print(f'[{ok}] {label}')
    print(f'       P198bm: {bm_action}:{bm_detail}')
    print(f'       P198bn: verdict={bn_verdict}  reason={bn_reason}')
    print(f'       combined: {actual} (expected {expected})')
    print()

print(f'{passed}/{len(cases)} scenarios passed')

# ── Positive freight-wording scenarios — ensure P198bm still applies ──
print()
print("=" * 80)
print("Positive freight-wording scenarios (P198bm should APPLY)")
print("=" * 80)
positive_cases = [
    ("BL FREIGHT PREPAID requirement",
     "BL must be marked FREIGHT PREPAID.",
     "MAERSK LINE BL - FREIGHT PREPAID AT ORIGIN",
     ('APPLY', 'FREIGHT PREPAID'), True),
    ("BL FREIGHT PREPAID requirement, BL doesn't have it",
     "BL must be marked FREIGHT PREPAID.",
     "MAERSK LINE BL - FREIGHT COLLECT",
     ('APPLY', 'FREIGHT PREPAID'), False),
    ("BL FREIGHT PAYABLE AS PER CHARTER PARTY",
     "BL must show freight payable as per charter party.",
     "BL - FREIGHT PAYABLE AS PER CHARTER PARTY DATED 28 NOV 2024",
     ('APPLY', 'FREIGHT PAYABLE AS PER CHARTER PARTY'), True),
    ("BL FREIGHT TO BE FORWARDED (payability)",
     "Freight to be forwarded per the charter party.",
     "MT YUNDING - FREIGHT TO BE FORWARDED AT DESTINATION",
     ('APPLY', 'FREIGHT FORWARD'), True),
]
ppass = 0
for label, cond, bl, expected_bm, wording_present in positive_cases:
    bm_action, bm_detail = classify_bm(cond)
    got = (bm_action, bm_detail)
    ok_bm = 'OK' if got == expected_bm else 'FAIL'
    # When APPLY, check wording presence
    if bm_action == 'APPLY':
        # pick first alt from the key
        alts = {
            'FREIGHT PREPAID': ('FREIGHT PREPAID','FRT PREPAID','FREIGHT PAID','PREPAID FREIGHT'),
            'FREIGHT COLLECT': ('FREIGHT COLLECT','FRT COLLECT','COLLECT FREIGHT','FREIGHT TO COLLECT'),
            'FREIGHT FORWARD': ('FREIGHT FORWARD','FREIGHT TO BE FORWARDED','FRT FORWARD'),
            'FREIGHT PAYABLE': ('FREIGHT PAYABLE',),
            'FREIGHT PAYABLE AS PER CHARTER PARTY': (
                'FREIGHT PAYABLE AS PER CHARTER PARTY','FREIGHT AS PER CHARTER PARTY',
                'FREIGHT PER CHARTER PARTY','FREIGHT PAYABLE AS PER C/P',
                'FREIGHT AS PER C/P','FREIGHT PAYABLE'),
            'FREIGHT PAYABLE AT DESTINATION': (
                'FREIGHT PAYABLE AT DESTINATION',
                'FREIGHT PAYABLE AT PORT OF DISCHARGE',
                'FREIGHT PAYABLE AT DESTINATION PORT',
                'FREIGHT COLLECT','COLLECT FREIGHT'),
        }
        present = [t for t in alts.get(bm_detail, ()) if t in bl.upper()]
        has_wording = bool(present)
    else:
        has_wording = None
    ok_wording = 'OK' if has_wording == wording_present else ('n/a' if has_wording is None else 'FAIL')
    ok_all = 'OK' if ok_bm == 'OK' and (has_wording == wording_present) else 'FAIL'
    if ok_all == 'OK':
        ppass += 1
    print(f'[{ok_all}] {label}')
    print(f'       P198bm: {bm_action}:{bm_detail} (expected {expected_bm})')
    print(f'       wording_on_BL: {has_wording} (expected {wording_present})')
    print()
print(f'{ppass}/{len(positive_cases)} positive-wording scenarios passed')
