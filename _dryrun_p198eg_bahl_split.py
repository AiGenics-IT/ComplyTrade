"""
P198eg dry-run — BAHL multi-message report grouping must split
two concatenated Alliance reports correctly.

Job 4e3d783b-4769-42df-993e-a147c192ad2a (and its mirror 5417141d)
contains TWO separate Alliance Message Management reports
concatenated into one PDF:
    pages 3-7  = Alliance report containing ONE MT700 (LC)
    pages 8-10 = Alliance report containing ONE MT707 (Amendment)
Each report's first SWIFT page carries "Message Details #1" — the
in-report ordinal. The earlier BAHL detector keyed
`_bahl_messages` by the in-report number, so both #1 occurrences
(MT700 page 3 and MT707 page 8) collided into a single group and
pages 3-10 all got labelled "LC". Step08 then never saw the
amendment and the F47A/F45B updates didn't apply.

The P198eg fix tracks each occurrence of "Message Details #N" as
its own sequential group, with three triggers for a NEW group:
    1. First message ever
    2. The in-report number is 1 (report restart)
    3. The fin.NNN on this header page differs from the current
       group's fin (= a different message even if N repeats)

Tests use the actual OCR from job 4e3d783b plus 4 synthetic
multi-report patterns.
"""
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

_BAHL_MSG_DETAIL_RE = re.compile(r'Message\s+Details\s+#\s*(\d+)', re.IGNORECASE)
_BAHL_IDENTIFIER_RE = re.compile(r'Identifier\s*:\s*fin\.(\d{3})', re.IGNORECASE)
_BAHL_FIN_TO_MT = {
    '700': 'LC', '701': 'LC', '705': 'LC',
    '707': 'Amendment', '708': 'Amendment', '747': 'Amendment',
    '799': 'MT799', '999': 'MT999',
    '754': 'MT754', '940': 'MT940', '730': 'MT730',
    '740': 'MT740', '742': 'MT742',
    '734': 'MT734', '750': 'MT750', '752': 'MT752',
}


def simulate_grouping(pages):
    """pages: list of (page_number, text). Returns _bahl_messages dict
    after applying the P198eg fix."""
    sorted_pages = sorted(pages, key=lambda x: x[0])
    msg_detail_pages = {}
    for pn, txt in sorted_pages:
        if not txt:
            continue
        for m in _BAHL_MSG_DETAIL_RE.finditer(txt):
            msg_detail_pages.setdefault(pn, []).append(int(m.group(1)))

    bahl = {}
    if len(msg_detail_pages) < 2:
        return bahl

    cur_grp = None
    next_id = 0
    for pn, txt in sorted_pages:
        if pn in msg_detail_pages:
            msg_num = sorted(msg_detail_pages[pn])[-1]
            id_m = _BAHL_IDENTIFIER_RE.search(txt or '')
            page_fin = id_m.group(1) if id_m else ''
            start_new = (
                cur_grp is None
                or msg_num == 1
                or (page_fin and bahl.get(cur_grp, {}).get('fin')
                    and page_fin != bahl[cur_grp]['fin'])
            )
            if start_new:
                next_id += 1
                bahl[next_id] = {
                    'pages': [], 'fin': page_fin,
                    'mt': _BAHL_FIN_TO_MT.get(page_fin, ''),
                    'msg_num_in_report': msg_num,
                }
                cur_grp = next_id
        if cur_grp is not None:
            bahl[cur_grp]['pages'].append(pn)
    return bahl


# ── Load real job data ───────────────────────────────────────────
def load_pages(job_id, page_range=None):
    s2 = Path(f'results/{job_id}/step02/step02_result.json')
    if not s2.exists():
        return []
    data = json.loads(s2.read_text(encoding='utf-8'))
    pages = []
    for p in data.get('pages', []):
        pn = p.get('page_number')
        if page_range and pn not in page_range:
            continue
        txt = p.get('cleaned_text') or p.get('raw_text') or ''
        pages.append((pn, txt))
    return pages


SYNTH_PAGE_3_MT700 = """Report Header
Application Alliance Message Management
Message Details #1
Identifier: fin.700
Expansion: Issue of a Documentary Credit
F20: TXN-001
F31C: 251215
F45A: GOODS"""

SYNTH_PAGE_8_MT707 = """Report Header
Application Alliance Message Management
Message Details #1
Identifier: fin.707
Expansion: Amendment to a Documentary Credit
F20: TXN-001
F26E: 1
F30: 260209"""

SYNTH_PAGE_3_REPORT2_MT700 = """Report Header
Message Details #1
Identifier: fin.700
F20: TXN-A
F31C: 251201"""

SYNTH_PAGE_4_REPORT2_MT707 = """Continuation of MT700, no new header"""

SYNTH_REPORT_MULTI_MSG = [
    (1, """Message Details #1
Identifier: fin.700
F20: TXN-001"""),
    (2, """Continuation of #1"""),
    (3, """Message Details #2
Identifier: fin.730
F20: TXN-002"""),
    (4, """Message Details #3
Identifier: fin.707
F26E: 1"""),
]


def main():
    pass_n, fail_n = 0, 0
    print('=' * 78)
    print('P198eg dry-run — BAHL multi-message report split')
    print('=' * 78)

    # ── A. Real job 4e3d783b (user's case) ──
    print('\n--- A. Real job 4e3d783b (UBL Iceberg Industries) ---')
    pages = load_pages('4e3d783b-4769-42df-993e-a147c192ad2a')
    bahl = simulate_grouping(pages)
    print(f'   {len(bahl)} BAHL messages detected:')
    for gid, info in bahl.items():
        print(f'     Group {gid}: pages={info["pages"]} fin.{info["fin"]} = {info["mt"]}')
    # Expected: at least one MT700 group containing pages 3-7 only,
    # and one MT707 group containing pages 8-10 only (page 8 must
    # NOT be in the MT700 group).
    mt700 = [g for g in bahl.values() if g['mt'] == 'LC']
    mt707 = [g for g in bahl.values() if g['mt'] == 'Amendment']
    test_a = (
        len(mt700) >= 1 and len(mt707) >= 1
        and 8 not in mt700[0]['pages']
        and 8 in mt707[0]['pages']
        and 7 in mt700[0]['pages']
    )
    if test_a:
        print('   [OK ] MT700 and MT707 split correctly; page 8 in MT707 not MT700')
        pass_n += 1
    else:
        print('   [FAIL] split did NOT happen as expected')
        fail_n += 1

    # ── B. Real job 5417141d (mirror) ──
    print('\n--- B. Real job 5417141d (mirror of 4e3d783b) ---')
    pages = load_pages('5417141d-bbcc-4e73-a3ab-8625d264d66f')
    bahl = simulate_grouping(pages)
    print(f'   {len(bahl)} BAHL messages detected:')
    for gid, info in bahl.items():
        print(f'     Group {gid}: pages={info["pages"]} fin.{info["fin"]} = {info["mt"]}')
    mt700 = [g for g in bahl.values() if g['mt'] == 'LC']
    mt707 = [g for g in bahl.values() if g['mt'] == 'Amendment']
    test_b = (
        len(mt700) >= 1 and len(mt707) >= 1
        and 8 not in mt700[0]['pages']
        and 8 in mt707[0]['pages']
    )
    if test_b:
        print('   [OK ] split correct')
        pass_n += 1
    else:
        print('   [FAIL] split incorrect')
        fail_n += 1

    # ── C. Synthetic — single multi-message report with sequential numbers ──
    print('\n--- C. Synthetic SINGLE report with messages #1, #2, #3 ---')
    bahl = simulate_grouping(SYNTH_REPORT_MULTI_MSG)
    print(f'   {len(bahl)} groups: ' +
          ', '.join(f'g{g}=pages{i["pages"]}({i["mt"]})'
                    for g, i in bahl.items()))
    # Expected 3 groups: MT700, MT730, MT707
    test_c = (len(bahl) == 3
              and any(i['mt'] == 'LC' for i in bahl.values())
              and any(i['mt'] == 'MT730' for i in bahl.values())
              and any(i['mt'] == 'Amendment' for i in bahl.values()))
    if test_c:
        print('   [OK ] 3 messages correctly split (LC + MT730 + Amendment)')
        pass_n += 1
    else:
        print('   [FAIL]')
        fail_n += 1

    # ── D. Synthetic — TWO separate reports each with #1 ──
    print('\n--- D. Synthetic TWO separate reports each starting #1 ---')
    pages = [
        (1, "Header"),
        (2, "Continuation"),
        (3, SYNTH_PAGE_3_MT700),
        (4, "Continuation of MT700"),
        (5, "Continuation of MT700"),
        (8, SYNTH_PAGE_8_MT707),
        (9, "Continuation of MT707"),
    ]
    bahl = simulate_grouping(pages)
    print(f'   {len(bahl)} groups: ' +
          ', '.join(f'g{g}=pages{i["pages"]}({i["mt"]})'
                    for g, i in bahl.items()))
    test_d = (len(bahl) == 2
              and any(i['mt'] == 'LC' for i in bahl.values())
              and any(i['mt'] == 'Amendment' for i in bahl.values()))
    if test_d:
        print('   [OK ] two separate reports split into 2 groups')
        pass_n += 1
    else:
        print('   [FAIL]')
        fail_n += 1

    # ── E. Edge: only ONE message header — no BAHL mode ──
    print('\n--- E. Single message — no BAHL grouping ---')
    pages = [
        (1, """Message Details #1
Identifier: fin.700
F20: TXN-A"""),
        (2, "Continuation"),
    ]
    bahl = simulate_grouping(pages)
    test_e = len(bahl) == 0  # < 2 message headers → no BAHL
    if test_e:
        print('   [OK ] single message — BAHL not triggered')
        pass_n += 1
    else:
        print(f'   [FAIL] BAHL triggered with {len(bahl)} groups')
        fail_n += 1

    total = pass_n + fail_n
    print('\n' + '=' * 78)
    print(f'OVERALL: {pass_n}/{total} '
          f'{"OK" if fail_n == 0 else "— failures present"}')
    print('=' * 78)
    return 0 if fail_n == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
