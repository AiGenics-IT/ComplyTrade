"""
P198cr dry-run — Tightened House B/L rescue guard.

Adds to P198ch/ck:
  1. Capacity-affirm phrases must occur in the last 40% of BL text
     OR within 300 chars of a signature marker (to avoid matching
     boilerplate T&C text that mentions "THE CARRIER" / "FOR AND ON
     BEHALF OF THE CARRIER" in legal clauses).
  2. Forwarder-name block: any BL carrying "LOGISTICS",
     "FORWARDING/FORWARDER", "NVOCC", "FIATA", "CONSOLIDATOR", etc.
     in its text is flagged as a freight-forwarder-issued BL and
     the HOUSE rescue is blocked regardless of other signals.
  3. Dropped "THE CARRIER", "AS CARRIER", "SHIPPING LINE",
     "SHIPPING COMPANY" from _CAPACITY_AFFIRMS — those phrases
     appear in boilerplate T&C of almost every BL and defeated the
     no-capacity-proof check.

Covers the user-reported case: a BL issued by "M.Y Logistics" with
no visible signing capacity was being PASS'd by the P198ar
prohibition-absence rescue because the BL had no literal "HOUSE"
token. It should FAIL.
"""
import re


_BL_PROHIB_TOKENS = {
    'HOUSE': ('HOUSE BILL OF LADING', 'HOUSE B/L',
              'HOUSE BL', 'HBL', 'HAWB'),
}
_COND_SYNONYMS = {
    'HOUSE': ('HOUSE B/L', 'HOUSE BILL OF LADING', 'HOUSE BL', 'HBL'),
}
_DEFINITION_MARKERS = (
    'MEANS ', 'MEAN ', 'SHALL MEAN', 'INCLUDES ',
    'DEFINED AS', 'DEFINITION OF', 'REFERS TO',
    'INTERPRETED AS', 'DEFINED HEREIN',
)
_QUALIFIERS = (
    'MASTER', 'CARRIER', 'OWNER', 'OWNERS',
    'CHARTERER', 'CHARTERERS',
    'SHIPPING LINE', 'SHIPPING COMPANY',
    'THE VESSEL', 'THE SHIP',
)
_CAPACITY_AFFIRMS = (
    'AS MASTER', 'MASTER OF THE VESSEL', 'MASTER OF THE SHIP',
    'AS THE MASTER', 'SIGNED BY THE MASTER',
    'AS AGENT FOR THE MASTER', 'AS AGENTS FOR THE MASTER',
    'AS AGENT FOR MASTER', 'AS AGENTS FOR MASTER',
    'AS AGENT ON BEHALF OF THE MASTER',
    'AS AGENTS ON BEHALF OF THE MASTER',
    'AS AGENTS FOR AND ON BEHALF OF THE MASTER',
    'AS AGENT FOR AND ON BEHALF OF THE MASTER',
    'FOR AND ON BEHALF OF THE MASTER',
    'FOR THE MASTER AS AGENT', 'FOR THE MASTER AS AGENTS',
    'AS AGENTS ONLY FOR AND BY AUTHORITY OF THE MASTER',
    'AS AGENT ONLY FOR AND BY AUTHORITY OF THE MASTER',
    'SIGNED BY THE CARRIER',
    'AS AGENT FOR THE CARRIER', 'AS AGENTS FOR THE CARRIER',
    'AS AGENT FOR AND ON BEHALF OF THE CARRIER',
    'AS AGENTS FOR AND ON BEHALF OF THE CARRIER',
    'FOR AND ON BEHALF OF THE CARRIER',
    'AS OWNER', 'AS OWNERS',
    'AS AGENT FOR THE OWNER', 'AS AGENTS FOR THE OWNER',
    'AS AGENT FOR AND ON BEHALF OF THE OWNER',
    'AS AGENTS FOR AND ON BEHALF OF THE OWNER',
    'FOR AND ON BEHALF OF THE OWNER',
    'AS CHARTERER', 'AS CHARTERERS',
    'FOR AND ON BEHALF OF THE CHARTERER',
)
_SIGN_MARKERS = (
    '[SIGNATURE]', 'SIGNATURE:', 'SIGNED BY',
    'AUTHORIZED SIGNATORY', 'AUTHORISED SIGNATORY',
    'FOR AND ON BEHALF OF', 'STAMP:',
)
_FORWARDER_INDICATORS = (
    r'\bLOGISTICS\b',
    r'\bFREIGHT\s+FORWARD(?:ING|ER|ERS)?\b',
    r'\bFORWARDING\s+AGENT\b',
    r'\bFORWARDER[S\']?\b',
    r'\bNVOCC\b',
    r'\bCONSOLIDAT(?:OR|ION|ED)\b',
    r'\bFIATA\b',
    r'\bMULTIMODAL\s+TRANSPORT\s+OPERATOR\b',
    r'\bMTO\b',
    r'\bEXPRESS\s+CARGO\b',
    r'\bSEA\s*&?\s*AIR\s+FREIGHT\b',
)


def _real_context(text_up, tok):
    idx = 0
    while True:
        pos = text_up.find(tok, idx)
        if pos < 0: return False
        pre = text_up[max(0, pos - 80): pos]
        if any(m in pre for m in _DEFINITION_MARKERS):
            idx = pos + 1; continue
        if '"' in pre[-40:] and 'MEANS' in text_up[pos:pos + 80]:
            idx = pos + 1; continue
        return True


def simulate(cond, doc, bl_subtype, initial='FAIL'):
    if initial != 'FAIL': return initial, 'not FAIL'
    cond_u = cond.upper()
    if not any(m in cond_u for m in (
        'NOT ACCEPTABLE', 'NOT PERMITTED', 'NOT ALLOWED',
        'MUST NOT', 'UNACCEPTABLE', 'SHALL NOT', 'WILL NOT',
        'NOT BE ACCEPT', 'PROHIBIT',
    )):
        return 'FAIL', 'not prohibitive'
    named = []
    for k, syns in _COND_SYNONYMS.items():
        if any(s in cond_u for s in syns):
            if k not in named: named.append(k)
    if not named: return 'FAIL', 'no named prohibitions'
    doc_up = doc.upper()
    tokens_present = []; tokens_checked = []
    for k in named:
        for tok in _BL_PROHIB_TOKENS.get(k, ()):
            tokens_checked.append(tok)
            if tok in doc_up and _real_context(doc_up, tok):
                tokens_present.append(tok)
    if tokens_present:
        return 'FAIL', f'tokens on doc: {tokens_present}'
    if not tokens_checked: return 'FAIL', 'nothing checked'

    if 'HOUSE' in named:
        struct_is_house = False
        if isinstance(bl_subtype, dict):
            if bool(bl_subtype.get('is_house_bl')): struct_is_house = True
            if str(bl_subtype.get('issuer_type', '') or '').lower() == 'house_bl':
                struct_is_house = True
            if str(bl_subtype.get('signing_type', '') or '').lower() == 'forwarder_signed':
                struct_is_house = True

        # Bare AS AGENT
        bare_agent = False
        agent_re = re.compile(r'\bAS\s+AGENTS?\b')
        for m in agent_re.finditer(doc_up):
            end = m.end()
            window = doc_up[end:end + 120]
            pre = doc_up[max(0, m.start() - 40):m.start()]
            if not any(q in window for q in _QUALIFIERS) and \
               not any(q in pre for q in _QUALIFIERS):
                bare_agent = True; break

        # Capacity proof with proximity guard
        doc_len = len(doc_up)
        cap_hit = None
        for ph in _CAPACITY_AFFIRMS:
            p = doc_up.find(ph)
            while p >= 0:
                in_last = (p >= int(doc_len * 0.60))
                near_sig = any(
                    abs(doc_up.find(sm, max(0, p - 300), p + 300 + len(ph)) - p) <= 300
                    for sm in _SIGN_MARKERS
                    if doc_up.find(sm, max(0, p - 300), p + 300 + len(ph)) >= 0
                )
                if in_last or near_sig:
                    cap_hit = (ph, p); break
                p = doc_up.find(ph, p + 1)
            if cap_hit: break
        no_capacity_proof = cap_hit is None

        # Forwarder-name check
        fwd_hit = None
        for pat in _FORWARDER_INDICATORS:
            m = re.search(pat, doc_up)
            if m:
                fwd_hit = (pat, m.group(0)); break

        if struct_is_house or bare_agent or no_capacity_proof or fwd_hit:
            bits = []
            if struct_is_house: bits.append('struct')
            if bare_agent: bits.append('bare-agent')
            if no_capacity_proof and not (struct_is_house or bare_agent or fwd_hit):
                bits.append('no-capacity-proof')
            if fwd_hit: bits.append(f'forwarder:{fwd_hit[1]}')
            return 'FAIL', 'P198ck/cr blocks: ' + '+'.join(bits)

    return 'PASS', f'rescue via P198ar'


COND = 'HOUSE B/L IS NOT ACCEPTABLE.'
SC = []

# User-reported: M.Y Logistics, no master/carrier capacity → BLOCK
SC.append(dict(name='M.Y Logistics issuer, no capacity → BLOCK',
    doc='''BILL OF LADING
Shipper: ABC Exports Ltd
Consignee: TO ORDER
Notify: XYZ Importer
Vessel: MV OCEAN
Port of Loading: Shanghai
Port of Discharge: Karachi

Issued by M.Y Logistics Pte Ltd
[SIGNATURE] [STAMP]
''', subtype={}, expect='FAIL'))

# BL with legit master signature in last 40% → PASS
SC.append(dict(name='Legit master BL — AS MASTER in signature block → PASS',
    doc=('BILL OF LADING\n' + ('Cargo line\n' * 60) +
         '[SIGNATURE]\nCapt J. Doe\nAS MASTER\n'),
    subtype={}, expect='PASS'))

# BL with "THE CARRIER" only in T&C boilerplate (not signature) → BLOCK
SC.append(dict(name='Boilerplate THE CARRIER in T&C only (no sig capacity) → BLOCK',
    doc='''BILL OF LADING
Shipper / Consignee / Notify / Vessel etc.
TERMS AND CONDITIONS:
1. The carrier shall not be liable for any loss or damage.
2. The carrier reserves the right to...
3. Cargo is carried at shipper's risk.
4. The carrier may issue a further set of Bills.
5. On behalf of the carrier all terms apply.

[SIGNATURE]
ABC Global Logistics Pte Ltd.
''', subtype={}, expect='FAIL'))  # logistics forwarder AND no sig capacity

# BL with "FOR AND ON BEHALF OF THE CARRIER" in legit signature
SC.append(dict(name='FOR AND ON BEHALF OF THE CARRIER in signature → PASS',
    doc=('BILL OF LADING\n' + ('Cargo line\n' * 40) +
         'Signed by:\nXYZ Shipping Agency\nFOR AND ON BEHALF OF THE CARRIER\nMAERSK LINE\n'),
    subtype={}, expect='PASS'))

# Forwarder name block — even with legit capacity phrase → still BLOCK
SC.append(dict(name='Legit capacity BUT forwarder issuer name → BLOCK',
    doc=('BILL OF LADING\n' + ('Cargo line\n' * 40) +
         'XYZ FREIGHT FORWARDING\nAS AGENT FOR THE MASTER\n[SIG]\n'),
    subtype={}, expect='FAIL'))

# NVOCC indicator → BLOCK
SC.append(dict(name='NVOCC line in BL → BLOCK',
    doc='''BILL OF LADING
Shipper: ABC
Consignee: TO ORDER
NVOCC: Global NVOCC Lines Inc.
[SIG] AS AGENT FOR THE CARRIER
''', subtype={}, expect='FAIL'))

# FIATA → BLOCK
SC.append(dict(name='FIATA marking → BLOCK',
    doc='''BILL OF LADING (FIATA FBL)
[header]
ABC Co.
[SIG]
''', subtype={}, expect='FAIL'))

# Clean BL with AS MASTER near signature, no forwarder indicators
SC.append(dict(name='Clean master BL, no forwarder patterns → PASS',
    doc='''BILL OF LADING
Shipper: Manufacturer Co.
Consignee: TO ORDER OF BANK
Notify: Applicant
Vessel: MV SEA
Port of Loading: Singapore
Port of Discharge: Karachi

Cargo description...

[SIGNATURE]
Capt. R. Smith
MASTER OF THE VESSEL
''', subtype={}, expect='PASS'))

# Structured bl_subtype says house → BLOCK
SC.append(dict(name='Structured is_house_bl=True → BLOCK',
    doc='any text with AS MASTER in last 40%\n' * 20 + '[SIG] AS MASTER',
    subtype={'is_house_bl': True}, expect='FAIL'))

# Bare AS AGENT → BLOCK
SC.append(dict(name='Bare AS AGENT (no qualifier) → BLOCK',
    doc='BILL OF LADING\n' + ('cargo\n' * 20) + 'Forwarder Co.\nAS AGENT\n[SIG]',
    subtype={}, expect='FAIL'))

# CONSOLIDATOR term → BLOCK
SC.append(dict(name='Cargo consolidator issuer → BLOCK',
    doc='BILL OF LADING\nCARGO CONSOLIDATION SERVICES CO\n[SIG] AS AGENT FOR THE MASTER\n',
    subtype={}, expect='FAIL'))

# Legit carrier like COSCO without "Logistics" → PASS
SC.append(dict(name='Real shipping-line BL (COSCO) with master signature → PASS',
    doc=('COSCO SHIPPING LINES BILL OF LADING\n' + ('Cargo line\n' * 30) +
         '[SIGNATURE]\nCOSCO SHIPPING AGENT\nAS AGENT FOR THE MASTER\n'),
    subtype={}, expect='PASS'))

# Maersk direct — PASS
SC.append(dict(name='Maersk direct master BL → PASS',
    doc=('MAERSK LINE BILL OF LADING\n' + ('Cargo line\n' * 30) +
         'Signed by: A.P. MOLLER-MAERSK\nAS MASTER\n[STAMP]\n'),
    subtype={}, expect='PASS'))

# Empty BL text — BLOCK
SC.append(dict(name='Empty BL text → BLOCK',
    doc='BILL OF LADING\n',
    subtype={}, expect='FAIL'))

# Forwarder name "LOGISTICS" appears but the doc is truly carrier-issued
# e.g., COSCO Logistics is a subsidiary. Edge case — we currently BLOCK
# (conservative).
SC.append(dict(name='"COSCO LOGISTICS" — currently blocked (conservative)',
    doc=('BILL OF LADING — COSCO LOGISTICS DIVISION\n' + ('cargo\n' * 20) +
         '[SIG]\nAS AGENT FOR THE MASTER\n'),
    subtype={}, expect='FAIL'))


def main():
    passed = 0; failed = 0
    for i, sc in enumerate(SC, 1):
        verdict, note = simulate(COND, sc['doc'], sc['subtype'])
        ok = (verdict == sc['expect'])
        tag = 'OK ' if ok else 'FAIL'
        print(f"[{tag}] #{i:02d}  {sc['name']}")
        print(f"         expect={sc['expect']}, got={verdict}")
        print(f"         note: {note}")
        if ok: passed += 1
        else: failed += 1
    print(f"\n{'='*78}\n{passed}/{passed+failed} P198cr house-BL scenarios OK\n{'='*78}")
    return failed == 0


if __name__ == '__main__':
    import sys
    sys.exit(0 if main() else 1)
