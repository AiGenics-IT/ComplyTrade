"""
P198ch dry-run — House B/L signing-capacity detection in P198ar rescue.

Rule:
  • Bare "As Agent" / "As Agents" on a BL with no "for master /
    carrier / owner" qualifier indicates a freight-forwarder-issued
    HOUSE BL. When LC prohibits HOUSE, such a BL should FAIL — the
    P198ar deterministic rescue must NOT flip it to PASS just because
    the literal "HOUSE" token is absent from the BL.
  • "As Agent for Master" / "As Agents for and on behalf of the
    Master" / "As Agents for the Carrier" → NOT house BL → OK to
    rescue.

Simulates the P198ar prohibition-absence rescue with the new
P198ch signing-capacity guard.
"""
import re


# Mirror of P198ar / P198ch logic (test-local copy)
_BL_PROHIB_TOKENS = {
    'HOUSE': ('HOUSE BILL OF LADING', 'HOUSE B/L',
              'HOUSE BL', 'HBL', 'HAWB'),
}
_COND_SYNONYMS = {
    'HOUSE': (
        'HOUSE B/L', 'HOUSE BILL OF LADING',
        'HOUSE BL', 'HBL',
    ),
}
_DEFINITION_MARKERS = (
    'MEANS ', 'MEAN ', 'SHALL MEAN', 'INCLUDES ',
    'DEFINED AS', 'DEFINITION OF', 'REFERS TO',
    'INTERPRETED AS', 'DEFINED HEREIN',
    '"NVOCC"', '"NVOCG"', "'NVOCC'",
    'DEFINITIONS', 'GLOSSARY',
)
_QUALIFIERS = (
    'MASTER', 'CARRIER', 'OWNER', 'OWNERS',
    'CHARTERER', 'CHARTERERS',
    'SHIPPING LINE', 'SHIPPING COMPANY',
    'THE VESSEL', 'THE SHIP',
)


def _has_real_context_match(text_up, tok):
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


def simulate_p198ar_ch(cond_text, doc_text, initial_verdict, bl_subtype):
    """Return (final_verdict, note)."""
    if initial_verdict != 'FAIL':
        return initial_verdict, 'not a FAIL; rescue not attempted'

    cond_u = cond_text.upper()
    prohibitive = any(m in cond_u for m in (
        'NOT ACCEPTABLE', 'NOT PERMITTED', 'NOT ALLOWED',
        'MUST NOT', 'UNACCEPTABLE', 'SHALL NOT',
        'WILL NOT', 'NOT BE ACCEPT', 'MUST NOT BE PRESENT',
        'PROHIBIT',
    ))
    if not prohibitive:
        return 'FAIL', 'condition not prohibitive; no rescue'

    named_prohibitions = []
    for k, syns in _COND_SYNONYMS.items():
        if any(s in cond_u for s in syns):
            if k not in named_prohibitions:
                named_prohibitions.append(k)
    if not named_prohibitions:
        return 'FAIL', 'no named prohibitions; no rescue'

    doc_text_up = doc_text.upper()
    tokens_present = []
    tokens_checked = []
    for k in named_prohibitions:
        for tok in _BL_PROHIB_TOKENS.get(k, ()):
            tokens_checked.append(tok)
            if tok in doc_text_up and _has_real_context_match(doc_text_up, tok):
                tokens_present.append(tok)

    if tokens_present:
        return 'FAIL', f'prohibited tokens present: {tokens_present}'
    if not tokens_checked:
        return 'FAIL', 'no tokens checked'

    # P198ch guard
    if 'HOUSE' in named_prohibitions:
        struct_is_house = False
        if isinstance(bl_subtype, dict):
            if bool(bl_subtype.get('is_house_bl')):
                struct_is_house = True
            if str(bl_subtype.get('issuer_type', '') or '').lower() == 'house_bl':
                struct_is_house = True
            if str(bl_subtype.get('signing_type', '') or '').lower() == 'forwarder_signed':
                struct_is_house = True

        bare_agent = False
        agent_re = re.compile(r'\bAS\s+AGENTS?\b', flags=re.IGNORECASE)
        for m in agent_re.finditer(doc_text_up):
            end = m.end()
            window = doc_text_up[end:end + 120]
            pre = doc_text_up[max(0, m.start() - 40):m.start()]
            if not any(q in window for q in _QUALIFIERS) and \
               not any(q in pre for q in _QUALIFIERS):
                bare_agent = True
                break

        if struct_is_house or bare_agent:
            bits = []
            if struct_is_house:
                bits.append(f'structured={bl_subtype}')
            if bare_agent:
                bits.append("bare 'AS AGENT' signing")
            return 'FAIL', 'P198ch blocks rescue: ' + ' + '.join(bits)

    return 'PASS', f'P198ar rescue: no prohibited tokens; checked {len(tokens_checked)}'


# ────────────── scenarios ──────────────
SCENARIOS = [
    dict(
        name='bare AS AGENT — user case (should STAY FAIL)',
        cond='HOUSE B/L IS NOT ACCEPTABLE.',
        doc=(
            'BILL OF LADING\nSHIPPER: ACME CORP\nCONSIGNEE: XYZ LTD\n'
            'VESSEL MV OCEAN\nSIGNED AT SINGAPORE\n\n'
            'ABC LOGISTICS PTE LTD\nAS AGENT\n[SIGNATURE]\n'
        ),
        bl_subtype={},
        initial='FAIL',
        expect='FAIL',
    ),
    dict(
        name='AS AGENTS alone (should STAY FAIL)',
        cond='HOUSE B/L NOT PERMITTED.',
        doc=(
            'BILL OF LADING\nSHIPPER: ...\n'
            'FREIGHT FORWARDER CO LTD\nAS AGENTS\n[SIGNATURE]\n'
        ),
        bl_subtype={},
        initial='FAIL',
        expect='FAIL',
    ),
    dict(
        name='AS AGENT FOR MASTER (should PASS)',
        cond='HOUSE BILL OF LADING NOT ACCEPTABLE.',
        doc=(
            'BILL OF LADING\nSHIPPER: ...\n'
            'XYZ SHIPPING AGENCY\nAS AGENT FOR THE MASTER CAPT. J. DOE\n'
            '[SIGNATURE]\n'
        ),
        bl_subtype={},
        initial='FAIL',
        expect='PASS',
    ),
    dict(
        name='AS AGENTS FOR AND ON BEHALF OF THE MASTER (should PASS)',
        cond='HOUSE B/L IS NOT ACCEPTABLE.',
        doc=(
            'BILL OF LADING\nSHIPPER: ...\n'
            'XYZ SHIPPING AGENCY\n'
            'AS AGENTS FOR AND ON BEHALF OF THE MASTER\n[SIGNATURE]\n'
        ),
        bl_subtype={},
        initial='FAIL',
        expect='PASS',
    ),
    dict(
        name='AS AGENT FOR THE CARRIER (should PASS)',
        cond='HOUSE BL NOT ACCEPTABLE.',
        doc=(
            'BILL OF LADING\nSHIPPER: ...\n'
            'AGENT CO\nAS AGENT FOR THE CARRIER MAERSK LINE\n[SIGNATURE]\n'
        ),
        bl_subtype={},
        initial='FAIL',
        expect='PASS',
    ),
    dict(
        name='FOR THE MASTER AS AGENT (reversed order, should PASS)',
        cond='HOUSE BILL OF LADING MUST NOT BE PRESENT.',
        doc=(
            'BILL OF LADING\nSHIPPER: ...\n'
            'GLOBAL AGENCY PTE\nFOR THE MASTER AS AGENT\n[SIGNATURE]\n'
        ),
        bl_subtype={},
        initial='FAIL',
        expect='PASS',
    ),
    dict(
        name='structured is_house_bl=True (should STAY FAIL)',
        cond='HOUSE B/L IS NOT ACCEPTABLE.',
        doc=(
            'BILL OF LADING\nSHIPPER: ...\n'
            'FORWARDER CO\nAS CARRIER\n[SIGNATURE]\n'
        ),
        bl_subtype={'is_house_bl': True, 'issuer_type': 'house_bl',
                    'signing_type': 'forwarder_signed'},
        initial='FAIL',
        expect='FAIL',
    ),
    dict(
        name='structured signing_type=forwarder_signed (should STAY FAIL)',
        cond='HOUSE BILL OF LADING NOT ACCEPTABLE.',
        doc=(
            'BILL OF LADING\nSHIPPER: ...\n'
            'NO AGENT TEXT HERE — but structured flags say forwarder.\n'
        ),
        bl_subtype={'signing_type': 'forwarder_signed'},
        initial='FAIL',
        expect='FAIL',
    ),
    dict(
        name='permissive condition (should PASS — no rescue needed; condition non-prohibitive)',
        cond='HOUSE B/L IS ACCEPTABLE.',
        doc='BILL OF LADING\nAS AGENT\n',
        bl_subtype={},
        initial='FAIL',
        expect='FAIL',   # P198ar only rescues prohibitive conditions; P198bb handles permissive.
    ),
    dict(
        name='BL text actually contains HOUSE BILL OF LADING (should STAY FAIL)',
        cond='HOUSE BILL OF LADING NOT ACCEPTABLE.',
        doc='HOUSE BILL OF LADING\nSHIPPER: ...\n',
        bl_subtype={},
        initial='FAIL',
        expect='FAIL',
    ),
    dict(
        name='AS AGENT inside boilerplate (no qualifier nearby), but actual signature is AS AGENT FOR MASTER',
        cond='HOUSE B/L NOT ACCEPTABLE.',
        doc=(
            'BILL OF LADING\n'
            'Definitions: "Agent" means a person acting as agent.\n'
            '...\nXYZ CO\nAS AGENT FOR THE MASTER\n[SIGNATURE]\n'
        ),
        bl_subtype={},
        # Both "AS AGENT" occurrences have MASTER within the 120-char
        # forward window (from the real signature two lines below),
        # so neither is bare — rescue proceeds to PASS. Correct.
        initial='FAIL',
        expect='PASS',
    ),
]


def main():
    passed = 0
    failed = 0
    for sc in SCENARIOS:
        verdict, note = simulate_p198ar_ch(
            sc['cond'], sc['doc'], sc['initial'], sc['bl_subtype'])
        ok = (verdict == sc['expect'])
        tag = 'OK ' if ok else 'FAIL'
        print(f"[{tag}] {sc['name']}")
        print(f"       expect={sc['expect']}, got={verdict}")
        print(f"       note: {note}")
        if ok:
            passed += 1
        else:
            failed += 1
    print(f"\n== {passed}/{passed+failed} scenarios OK ==")
    return failed == 0


if __name__ == '__main__':
    import sys
    sys.exit(0 if main() else 1)
