"""
P198ci dry-run — THCD / Terminal Handling Charges deterministic rescue.

Rule:
  • LC conditions like "BL must show THCD prepaid at origin" refer to
    a SEPARATE charges line on the BL, not the main FREIGHT payability.
  • If the BL carries a THCD / THC / Terminal Handling token WITH a
    prepaid/origin indicator nearby (±120 chars), rescue FAIL→PASS —
    the LLM likely conflated the THCD line with the main FREIGHT
    line (which may legitimately read "FREIGHT COLLECT").
  • If the THCD line itself reads COLLECT without a prepaid
    qualifier nearby, keep the FAIL.
  • Skip boilerplate / tariff clauses ("TO BE BORNE BY", "SHALL BE").
  • Only rescue affirmative conditions; skip prohibitive wording.
"""
import re


# ── Test-local mirror of P198ci logic ──
_THCD_STRONG_TOKENS = (
    'THCD',
    'THC DESTINATION', 'THC DEST',
    'DEST THC', 'DESTINATION THC',
    'DESTINATION TERMINAL HANDLING',
    'TERMINAL HANDLING CHARGES DESTINATION',
    'TERMINAL HANDLING CHARGE DESTINATION',
    'TERMINAL HANDLING DESTINATION',
    'ORIGIN THC', 'THC ORIGIN',
    'ORIGIN TERMINAL HANDLING',
    'TERMINAL HANDLING ORIGIN',
)
_THC_GENERIC_TOKENS = (
    'TERMINAL HANDLING CHARGES',
    'TERMINAL HANDLING CHARGE',
    'TERMINAL HANDLING',
)
_PREPAID_TOKENS = (
    'PREPAID AT ORIGIN',
    'AT ORIGIN PREPAID',
    'ORIGIN PREPAID',
    'PAID AT ORIGIN',
    'ORIGIN CHARGES PREPAID',
    'ORIGIN CHARGES PAID',
    'PRE-PAID', 'PRE PAID',
    'PREPAID', 'PREPAY',
)
_COLLECT_TOKENS = (
    'COLLECT AT DESTINATION',
    'COLLECT',
)
_thc_word_re = re.compile(r'\bTHC\b')


def _thcd_find_match(doc_text_up):
    hits = []
    for tok in _THCD_STRONG_TOKENS:
        idx = 0
        while True:
            p = doc_text_up.find(tok, idx)
            if p < 0:
                break
            hits.append((tok, p, p + len(tok)))
            idx = p + 1
    for tok in _THC_GENERIC_TOKENS:
        idx = 0
        while True:
            p = doc_text_up.find(tok, idx)
            if p < 0:
                break
            hits.append((tok, p, p + len(tok)))
            idx = p + 1
    for m in _thc_word_re.finditer(doc_text_up):
        if any(p <= m.start() < e for (_t, p, e) in hits):
            continue
        hits.append(('THC', m.start(), m.end()))
    for tok, p, e in hits:
        ctx = doc_text_up[max(0, p - 120): e + 120]
        boilerplate = (
            'TO BE BORNE BY', 'BORNE BY', 'PAYABLE BY',
            'SHALL BE', 'IS DEFINED', 'MEANS ',
            'DEFINITION', 'GLOSSARY', 'INCLUDED IN',
        )
        if any(b in ctx for b in boilerplate):
            continue
        has_prepaid = any(pp in ctx for pp in _PREPAID_TOKENS)
        if not has_prepaid:
            continue
        narrow = doc_text_up[max(0, p - 40): e + 40]
        if any(c in narrow for c in _COLLECT_TOKENS) and \
           not any(pp in narrow for pp in _PREPAID_TOKENS):
            continue
        return tok, ctx
    return None


def simulate(cond, doc, initial):
    cond_u = cond.upper()
    if initial != 'FAIL':
        return initial, 'not FAIL'
    cond_has_thcd = (
        'THCD' in cond_u or
        'TERMINAL HANDLING' in cond_u or
        _thc_word_re.search(cond_u) is not None
    )
    if not cond_has_thcd:
        return 'FAIL', 'condition does not mention THCD'
    if re.search(
        r'\b(?:NOT\s+SHOW|MUST\s+NOT|SHALL\s+NOT|'
        r'NOT\s+ACCEPTABLE|NOT\s+PERMITTED|PROHIBIT)\b',
        cond_u,
    ):
        return 'FAIL', 'prohibitive condition'
    m = _thcd_find_match(doc.upper())
    if not m:
        return 'FAIL', 'no THCD+prepaid context on doc'
    return 'PASS', f'THCD match on tok={m[0]!r}'


# ── Scenarios ──
SCENARIOS = [
    dict(
        name='User case — FREIGHT COLLECT + THCD PREPAID AT ORIGIN (PASS)',
        cond='Bill of lading must show THCD prepaid at origin.',
        doc=(
            'BILL OF LADING\n'
            'CHARGES\n'
            'BASIC FREIGHT ............ COLLECT\n'
            'THCD .................... PREPAID AT ORIGIN\n'
            'DOCUMENT FEE ............ PREPAID\n'
        ),
        initial='FAIL', expect='PASS',
    ),
    dict(
        name='Table format — THCD and PREPAID separated by columns (PASS)',
        cond='BL shall show THCD prepaid at origin.',
        doc=(
            'FREIGHT & CHARGES\n'
            'CHARGE   BASIS   PREPAID   COLLECT\n'
            'FREIGHT                      USD 1500.00\n'
            'THCD            USD 40.00\n'
            'The THCD has been prepaid at origin.\n'
        ),
        initial='FAIL', expect='PASS',
    ),
    dict(
        name='Terminal Handling Charges at Destination — prepaid (PASS)',
        cond='Bill of Lading must evidence Terminal Handling Charges Destination prepaid.',
        doc=(
            'FREIGHT COLLECT\n'
            'TERMINAL HANDLING CHARGES DESTINATION PREPAID AT ORIGIN USD 35\n'
        ),
        initial='FAIL', expect='PASS',
    ),
    dict(
        name='THC alone with PREPAID nearby (PASS)',
        cond='BL must show THC prepaid at origin.',
        doc=(
            'FREIGHT COLLECT\n'
            'THC PREPAID AT ORIGIN\n'
        ),
        initial='FAIL', expect='PASS',
    ),
    dict(
        name='THCD marked COLLECT — no PREPAID (STAY FAIL)',
        cond='BL must show THCD prepaid at origin.',
        doc=(
            'FREIGHT COLLECT\n'
            'THCD COLLECT AT DESTINATION\n'
        ),
        initial='FAIL', expect='FAIL',
    ),
    dict(
        name='BL has no THCD at all — pure FREIGHT (STAY FAIL)',
        cond='BL must show THCD prepaid at origin.',
        doc=(
            'FREIGHT COLLECT\n'
            'SHIPPED ON BOARD.\n'
        ),
        initial='FAIL', expect='FAIL',
    ),
    dict(
        name='Only boilerplate tariff reference (STAY FAIL)',
        cond='BL must show THCD prepaid at origin.',
        doc=(
            'FREIGHT COLLECT\n'
            '15. Terminal Handling Charges shall be borne by the '
            'merchant in accordance with the carriers tariff.\n'
        ),
        initial='FAIL', expect='FAIL',
    ),
    dict(
        name='Definition clause (STAY FAIL)',
        cond='BL must show THCD prepaid at origin.',
        doc=(
            'Definitions: "Terminal Handling Charge" means the fee '
            'payable by the merchant at the terminal.\n'
            'FREIGHT COLLECT\n'
        ),
        initial='FAIL', expect='FAIL',
    ),
    dict(
        name='Already PASS — no rescue needed (STAY PASS)',
        cond='BL must show THCD prepaid at origin.',
        doc='FREIGHT COLLECT\nTHCD PREPAID AT ORIGIN\n',
        initial='PASS', expect='PASS',
    ),
    dict(
        name='Prohibitive THCD condition (STAY FAIL even with match)',
        cond='BL must not show any separate THCD charges.',
        doc='THCD PREPAID AT ORIGIN USD 40\n',
        initial='FAIL', expect='FAIL',
    ),
    dict(
        name='ORIGIN THC prepaid variant (PASS)',
        cond='BL must show origin THC prepaid.',
        doc=(
            'FREIGHT COLLECT\n'
            'ORIGIN THC: PREPAID\n'
        ),
        initial='FAIL', expect='PASS',
    ),
    dict(
        name='Pre-paid spelled with hyphen (PASS)',
        cond='BL must show THCD prepaid at origin.',
        doc=(
            'FREIGHT COLLECT\n'
            'THCD PRE-PAID AT ORIGIN\n'
        ),
        initial='FAIL', expect='PASS',
    ),
    dict(
        name='OCR scrambles THCD line into two lines (PASS)',
        cond='BL must show THCD prepaid at origin.',
        doc=(
            'BILL OF LADING #HLC1234\n'
            'FREIGHT: COLLECT\n'
            '\n'
            'THCD\n'
            '  USD 40.00     PREPAID AT ORIGIN\n'
        ),
        initial='FAIL', expect='PASS',
    ),
    dict(
        name='LC asks just THC prepaid (not THCD) — matched (PASS)',
        cond='BL should show THC prepaid at origin.',
        doc='FREIGHT COLLECT\nTHC PREPAID AT ORIGIN USD 35\n',
        initial='FAIL', expect='PASS',
    ),
    dict(
        name='LC condition mentions THC word in bigger context — still matched',
        cond='BL must show THC prepaid at origin in addition to freight terms.',
        doc='FREIGHT COLLECT\nTHCD PREPAID AT ORIGIN\n',
        initial='FAIL', expect='PASS',
    ),
    dict(
        name='Condition unrelated to THCD — ignored',
        cond='BL must show freight prepaid.',
        doc='FREIGHT PREPAID\nSHIPPED ON BOARD\n',
        initial='FAIL', expect='FAIL',
    ),
    dict(
        name='THC appears only in unrelated word like LITHCON (no bare match)',
        cond='BL must show THCD prepaid at origin.',
        doc='FREIGHT COLLECT\nLITHCON SHIPPING CO LTD\n',
        initial='FAIL', expect='FAIL',
    ),
    dict(
        name='Destination Terminal Handling prepaid (PASS)',
        cond='BL shall show THCD prepaid at origin.',
        doc=(
            'FREIGHT COLLECT\n'
            'Destination Terminal Handling: Prepaid at Origin USD 40\n'
        ),
        initial='FAIL', expect='PASS',
    ),
    dict(
        name='THCD PREPAID stated in separate freight section (PASS)',
        cond='Bill of Lading must show THCD prepaid at origin.',
        doc=(
            'FREIGHT DETAILS:\n'
            'OCEAN FREIGHT               COLLECT\n'
            'BL FEE                      PREPAID\n'
            'THCD                        PREPAID\n'
            'SEAL FEE                    PREPAID\n'
        ),
        initial='FAIL', expect='PASS',
    ),
    dict(
        name='THCD prepaid — noise between THCD and PREPAID but <120 chars (PASS)',
        cond='BL must show THCD prepaid at origin.',
        doc=(
            'FREIGHT COLLECT\n'
            'THCD ---- reference 48A/2025 ---- Prepaid at Origin USD 40\n'
        ),
        initial='FAIL', expect='PASS',
    ),
    dict(
        name='Gap between THCD and PREPAID larger than 120 chars (STAY FAIL)',
        cond='BL must show THCD prepaid at origin.',
        doc=(
            'THCD\n' + ' ' * 200 + 'Freight description that is long and boring\n'
            '                                                                   \n'
            'Separately, freight prepaid in full for container TCLU1234567.\n'
        ),
        initial='FAIL', expect='FAIL',
    ),
]


def main():
    passed = 0
    failed = 0
    for i, sc in enumerate(SCENARIOS, 1):
        verdict, note = simulate(sc['cond'], sc['doc'], sc['initial'])
        ok = (verdict == sc['expect'])
        tag = 'OK ' if ok else 'FAIL'
        print(f"[{tag}] #{i:02d}  {sc['name']}")
        print(f"         expect={sc['expect']}, got={verdict}")
        print(f"         note: {note}")
        if ok:
            passed += 1
        else:
            failed += 1
    print(f"\n{'='*78}\n{passed}/{passed+failed} scenarios OK\n{'='*78}")
    return failed == 0


if __name__ == '__main__':
    import sys
    sys.exit(0 if main() else 1)
