"""
P198ck dry-run — Expanded House B/L signing-capacity guard.

P198ch already blocked the HOUSE-rescue when the BL showed bare "AS
AGENT" without a qualifier, or when structured bl_subtype flagged it
as house. P198ck tightens the guard further: if the BL has NO
master / carrier / owner signing-capacity affirmation anywhere in
its text, the HOUSE-prohibition rescue is blocked — the BL cannot
be proved to be carrier-issued, so the prohibition stands.

Scenarios cover:
  • BLs explicitly signed as master / agent-for-master / carrier /
    agent-for-carrier / owner / charterer / shipping line (rescue)
  • Bare "AS AGENT" (block rescue)
  • No signing-capacity text at all (block rescue — new)
  • BL with only "signature" blob and no capacity (block)
  • BL with literal HOUSE token (stays FAIL)
  • Structured bl_subtype.is_house_bl=True (block)
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
    'AS MASTER',
    'MASTER OF THE VESSEL',
    'MASTER OF THE SHIP',
    'AS THE MASTER',
    'SIGNED BY THE MASTER',
    'AS AGENT FOR THE MASTER',
    'AS AGENTS FOR THE MASTER',
    'AS AGENT FOR MASTER',
    'AS AGENTS FOR MASTER',
    'AS AGENT ON BEHALF OF THE MASTER',
    'AS AGENTS ON BEHALF OF THE MASTER',
    'AS AGENTS FOR AND ON BEHALF OF THE MASTER',
    'AS AGENT FOR AND ON BEHALF OF THE MASTER',
    'FOR AND ON BEHALF OF THE MASTER',
    'ON BEHALF OF THE MASTER',
    'FOR THE MASTER AS AGENT',
    'FOR THE MASTER AS AGENTS',
    'AS AGENTS ONLY FOR AND BY AUTHORITY OF THE MASTER',
    'AS AGENT ONLY FOR AND BY AUTHORITY OF THE MASTER',
    'AS CARRIER',
    'THE CARRIER',
    'SIGNED BY THE CARRIER',
    'AS AGENT FOR THE CARRIER',
    'AS AGENTS FOR THE CARRIER',
    'AS AGENT FOR AND ON BEHALF OF THE CARRIER',
    'AS AGENTS FOR AND ON BEHALF OF THE CARRIER',
    'FOR AND ON BEHALF OF THE CARRIER',
    'ON BEHALF OF THE CARRIER',
    'AS OWNER', 'AS OWNERS',
    'AS AGENT FOR THE OWNER',
    'AS AGENTS FOR THE OWNER',
    'AS AGENT FOR AND ON BEHALF OF THE OWNER',
    'AS AGENTS FOR AND ON BEHALF OF THE OWNER',
    'FOR AND ON BEHALF OF THE OWNER',
    'ON BEHALF OF THE OWNER',
    'AS CHARTERER', 'AS CHARTERERS',
    'FOR AND ON BEHALF OF THE CHARTERER',
    'SHIPPING LINE',
    'SHIPPING COMPANY',
)


def _real_context(text_up, tok):
    idx = 0
    while True:
        pos = text_up.find(tok, idx)
        if pos < 0:
            return False
        pre = text_up[max(0, pos - 80): pos]
        if any(m in pre for m in _DEFINITION_MARKERS):
            idx = pos + 1; continue
        if '"' in pre[-40:] and 'MEANS' in text_up[pos:pos + 80]:
            idx = pos + 1; continue
        return True


def simulate(cond, doc, bl_subtype, initial='FAIL'):
    if initial != 'FAIL':
        return initial, 'not FAIL'
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
    if not named:
        return 'FAIL', 'no named prohibitions'
    doc_up = doc.upper()
    tokens_present = []
    tokens_checked = []
    for k in named:
        for tok in _BL_PROHIB_TOKENS.get(k, ()):
            tokens_checked.append(tok)
            if tok in doc_up and _real_context(doc_up, tok):
                tokens_present.append(tok)
    if tokens_present:
        return 'FAIL', f'prohibited tokens on doc: {tokens_present}'
    if not tokens_checked:
        return 'FAIL', 'nothing checked'

    if 'HOUSE' in named:
        struct_is_house = False
        if isinstance(bl_subtype, dict):
            if bool(bl_subtype.get('is_house_bl')):
                struct_is_house = True
            if str(bl_subtype.get('issuer_type', '') or '').lower() == 'house_bl':
                struct_is_house = True
            if str(bl_subtype.get('signing_type', '') or '').lower() == 'forwarder_signed':
                struct_is_house = True

        bare_agent = False
        agent_re = re.compile(r'\bAS\s+AGENTS?\b')
        for m in agent_re.finditer(doc_up):
            end = m.end()
            window = doc_up[end:end + 120]
            pre = doc_up[max(0, m.start() - 40):m.start()]
            if not any(q in window for q in _QUALIFIERS) and \
               not any(q in pre for q in _QUALIFIERS):
                bare_agent = True; break

        no_capacity_proof = not any(ph in doc_up for ph in _CAPACITY_AFFIRMS)

        if struct_is_house or bare_agent or no_capacity_proof:
            bits = []
            if struct_is_house: bits.append('struct')
            if bare_agent: bits.append('bare-agent')
            if no_capacity_proof and not (struct_is_house or bare_agent):
                bits.append('no-capacity-proof')
            return 'FAIL', 'P198ch/ck blocks: ' + '+'.join(bits)

    return 'PASS', f'rescue via P198ar: checked {len(tokens_checked)} tokens'


# Scenarios
SC = []
SC.append(dict(
    name='AS MASTER directly signed — rescue PASS',
    cond='HOUSE B/L IS NOT ACCEPTABLE.',
    doc='BILL OF LADING\nCAPT. J. DOE\nAS MASTER\n[SIGNATURE]\n',
    bl_subtype={}, expect='PASS',
))
SC.append(dict(
    name='AS AGENT FOR MASTER — rescue PASS',
    cond='HOUSE B/L NOT ACCEPTABLE.',
    doc='BILL OF LADING\nXYZ AGENCY\nAS AGENT FOR THE MASTER\n',
    bl_subtype={}, expect='PASS',
))
SC.append(dict(
    name='AS CARRIER direct — rescue PASS',
    cond='HOUSE B/L IS NOT ACCEPTABLE.',
    doc='BILL OF LADING\nMAERSK LINE\nAS CARRIER\n[SIG]\n',
    bl_subtype={}, expect='PASS',
))
SC.append(dict(
    name='AS AGENT FOR THE CARRIER — rescue PASS',
    cond='HOUSE BL NOT ACCEPTABLE.',
    doc='BILL OF LADING\nAGENT CO\nAS AGENT FOR THE CARRIER\n',
    bl_subtype={}, expect='PASS',
))
SC.append(dict(
    name='FOR AND ON BEHALF OF THE MASTER — rescue PASS',
    cond='HOUSE B/L IS NOT ACCEPTABLE.',
    doc='BILL OF LADING\nXYZ\nFOR AND ON BEHALF OF THE MASTER\n',
    bl_subtype={}, expect='PASS',
))
SC.append(dict(
    name='AS OWNER signed — rescue PASS',
    cond='HOUSE BILL OF LADING NOT ACCEPTABLE.',
    doc='BILL OF LADING\nLAMDA SHIPPING\nAS OWNER\n[SIG]\n',
    bl_subtype={}, expect='PASS',
))
SC.append(dict(
    name='Signed by the Shipping Line — rescue PASS',
    cond='HOUSE B/L NOT ACCEPTABLE.',
    doc='BILL OF LADING\n...COSCO SHIPPING LINE\nSIGNED\n',
    bl_subtype={}, expect='PASS',
))
SC.append(dict(
    name='Bare AS AGENT — BLOCK (P198ch)',
    cond='HOUSE B/L IS NOT ACCEPTABLE.',
    doc='BILL OF LADING\nFORWARDER CO\nAS AGENT\n[SIG]\n',
    bl_subtype={}, expect='FAIL',
))
SC.append(dict(
    name='No signing capacity at all — BLOCK (P198ck NEW)',
    cond='HOUSE B/L NOT ACCEPTABLE.',
    doc='BILL OF LADING\nShipper: ABC\nConsignee: XYZ\nVessel: MV OCEAN\n[SIGNATURE]\n',
    bl_subtype={}, expect='FAIL',
))
SC.append(dict(
    name='Only a signature blob with no capacity — BLOCK (P198ck)',
    cond='HOUSE B/L IS NOT ACCEPTABLE.',
    doc='BILL OF LADING\nSignatures:\nXYZ Logistics Pte Ltd\n[stamped]\n',
    bl_subtype={}, expect='FAIL',
))
SC.append(dict(
    name='Structured is_house_bl=True — BLOCK',
    cond='HOUSE B/L NOT ACCEPTABLE.',
    doc='BILL OF LADING\nSIGNED AS CARRIER.\n',  # would otherwise pass
    bl_subtype={'is_house_bl': True, 'issuer_type': 'house_bl',
                'signing_type': 'forwarder_signed'},
    expect='FAIL',
))
SC.append(dict(
    name='BL text literally has HOUSE BILL OF LADING — STAYS FAIL',
    cond='HOUSE BILL OF LADING NOT ACCEPTABLE.',
    doc='HOUSE BILL OF LADING\n[text]\n',
    bl_subtype={}, expect='FAIL',
))
SC.append(dict(
    name='AS CHARTERER (for charter-party BL) — rescue PASS',
    cond='HOUSE B/L IS NOT ACCEPTABLE.',
    doc='CHARTER-PARTY B/L\nSIGNED AS CHARTERER\n',
    bl_subtype={}, expect='PASS',
))
SC.append(dict(
    name='Master of the Vessel — rescue PASS',
    cond='HOUSE B/L IS NOT ACCEPTABLE.',
    doc='BILL OF LADING\nCAPT DOE\nMASTER OF THE VESSEL\n',
    bl_subtype={}, expect='PASS',
))
SC.append(dict(
    name='AS AGENT appears inside boilerplate AND genuine AS AGENT FOR MASTER elsewhere — rescue PASS',
    cond='HOUSE B/L NOT ACCEPTABLE.',
    doc='Definitions: "Agent" means a person acting as agent.\n...\nXYZ\nAS AGENT FOR THE MASTER\n[SIG]',
    bl_subtype={}, expect='PASS',
))
SC.append(dict(
    name='Empty doc text — BLOCK (P198ck)',
    cond='HOUSE B/L NOT ACCEPTABLE.',
    doc='BILL OF LADING\n',
    bl_subtype={}, expect='FAIL',
))
SC.append(dict(
    name='Prohibitive condition absent — no rescue attempted',
    cond='HOUSE B/L IS ACCEPTABLE.',
    doc='BILL OF LADING\nAS AGENT\n',
    bl_subtype={}, expect='FAIL',  # not prohibitive, no rescue
))
SC.append(dict(
    name='FOR AND ON BEHALF OF THE CARRIER — rescue PASS',
    cond='HOUSE B/L NOT PERMITTED.',
    doc='BILL OF LADING\n...FOR AND ON BEHALF OF THE CARRIER\n',
    bl_subtype={}, expect='PASS',
))
SC.append(dict(
    name='FOR THE MASTER AS AGENT (reversed order) — rescue PASS',
    cond='HOUSE B/L MUST NOT BE PRESENT.',
    doc='BILL OF LADING\nXYZ\nFOR THE MASTER AS AGENT\n',
    bl_subtype={}, expect='PASS',
))
SC.append(dict(
    name='Only generic company name with no capacity — BLOCK',
    cond='HOUSE B/L NOT ACCEPTABLE.',
    doc='BILL OF LADING\nPT. GLOBAL FORWARDING\n[stamp] [signature]\n',
    bl_subtype={}, expect='FAIL',
))
SC.append(dict(
    name='SIGNED BY THE CARRIER — rescue PASS',
    cond='HOUSE B/L IS NOT ACCEPTABLE.',
    doc='BILL OF LADING\n...\nSIGNED BY THE CARRIER ON BEHALF OF THE OWNER\n',
    bl_subtype={}, expect='PASS',
))


def main():
    passed = 0
    failed = 0
    for i, sc in enumerate(SC, 1):
        verdict, note = simulate(sc['cond'], sc['doc'], sc['bl_subtype'])
        ok = (verdict == sc['expect'])
        tag = 'OK ' if ok else 'FAIL'
        print(f"[{tag}] #{i:02d}  {sc['name']}")
        print(f"         expect={sc['expect']}, got={verdict}")
        print(f"         note: {note}")
        if ok: passed += 1
        else: failed += 1
    print(f"\n{'='*78}\n{passed}/{passed+failed} P198ck scenarios OK\n{'='*78}")
    return failed == 0


if __name__ == '__main__':
    import sys
    sys.exit(0 if main() else 1)
