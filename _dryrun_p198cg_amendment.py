"""P198cg dry-run — MT707 / fin.707 amendment boundary across multiple
pages. Tests that a 2+ page amendment (page 1 = fin.707 header, page 2 =
F45B / F46B continuation body) gets correctly grouped into ONE packet
with page 2 marked as continuation."""
import json
import re
import sys
sys.path.insert(0, '.')

# Mirror production regex (after P198cg)
_SWIFT_LC_PATTERNS = [
    r'Message\s+type:\s*700',
    r'\bfin\.\s*700\b', r'\bfin\.\s*701\b',
    r'\bIssue\s+of\s+a\s+Documentary\s+Credit\b',
    r'(?:^|\n)\s*F40A\s*:', r'(?:^|\n)\s*F31D\s*:',
    r'(?:^|\n)\s*F46A\s*:',
]
_SWIFT_AMEND_PATTERNS = [
    r'Message\s+type:\s*707', r'Message\s+type:\s*708',
    r'SWIFT_MT707', r'SWIFT_MT708',
    r'\bfin\.\s*707\b', r'\bfin\.\s*708\b',
    r'\bIdentifier\s*:\s*fin\.\s*707\b',
    r'\bIdentifier\s*:\s*fin\.\s*708\b',
    r'\bAmendment\s+to\s+a\s+Documentary\s+Credit\b',
    r'(?:^|\n)\s*26E:',
    r'Number\s+of\s+Amendment',
]
_SWIFT_NON_LC_PATTERNS = [
    (r'fin\.\s*730|Message\s+type:\s*730|\bMT[\s_]?730\b', 'MT730'),
    (r'fin\.\s*740|Message\s+type:\s*740|\bMT[\s_]?740\b', 'MT740'),
]
_SWIFT_CONTINUATION_PATTERNS = [
    r'(?:^|\n)\s*(?::|\bF)45[AB][\s:]+',
    r'(?:^|\n)\s*(?::|\bF)46[AB][\s:]+',
    r'(?:^|\n)\s*(?::|\bF)47[AB][\s:]+',
    r'(?:^|\n)\s*(?::|\bF)78[\s:]+',
    r'(?:^|\n)\s*(?::|\bF)72[Z]?[\s:]+',
]


def classify(text):
    if not text or len(text.strip()) < 80:
        return 'blank'
    non_lc = None
    for pat, mt in _SWIFT_NON_LC_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            non_lc = mt
            break
    is_amend = any(re.search(p, text, re.IGNORECASE) for p in _SWIFT_AMEND_PATTERNS)
    is_lc = any(re.search(p, text, re.IGNORECASE) for p in _SWIFT_LC_PATTERNS)
    is_swift_cont = any(re.search(p, text, re.IGNORECASE) for p in _SWIFT_CONTINUATION_PATTERNS)
    # Priority: non_lc → amend → lc → continuation
    if non_lc:
        return non_lc
    if is_amend:
        return 'Amendment'
    if is_lc:
        return 'LC'
    if is_swift_cont:
        return '_swift_continuation'
    return ''


_BAHL_MSG_RE = re.compile(r'Message\s+Details\s+#\s*(\d+)', re.IGNORECASE)
_BAHL_FIN_RE = re.compile(r'Identifier\s*:\s*fin\.(\d{3})', re.IGNORECASE)
_POT_RE = re.compile(r'Page\s+(\d+)\s+of\s+(\d+)', re.IGNORECASE)
_FIN_TO_MT = {'700': 'LC', '701': 'LC', '707': 'Amendment',
              '708': 'Amendment', '730': 'MT730', '754': 'MT754',
              '740': 'MT740', '799': 'MT799'}


def _detect_bahl(pages):
    """Return {pg_num: (msg_num, mt_type, [pages_in_msg])} or {} if not BAHL."""
    msg_headers = {}
    page_of_total = {}
    for pn, t in pages.items():
        for m in _BAHL_MSG_RE.finditer(t):
            msg_headers.setdefault(pn, []).append(int(m.group(1)))
        pm = _POT_RE.search(t)
        if pm:
            page_of_total[pn] = (int(pm.group(1)), int(pm.group(2)))
    if len(msg_headers) < 2:
        return {}
    max_page = 0
    for pn in sorted(msg_headers):
        if pn in page_of_total:
            x, y = page_of_total[pn]
            max_page = max(max_page, pn + (y - x))
    bahl = {}
    curr = None
    for pn in sorted(pages):
        if max_page and pn > max_page:
            break
        if pn in msg_headers:
            curr = msg_headers[pn][-1]
        if curr is not None:
            bahl.setdefault(curr, []).append(pn)
    # Determine mt_type for each message from its first-page fin.XXX
    page_to_msg = {}
    for msg_num, pgs in bahl.items():
        mt_type = ''
        for p in pgs:
            fm = _BAHL_FIN_RE.search(pages.get(p, ''))
            if fm:
                mt_type = _FIN_TO_MT.get(fm.group(1), f"MT{fm.group(1)}")
                break
        for p in pgs:
            page_to_msg[p] = (msg_num, mt_type, pgs)
    return page_to_msg


def resolve_chain(pages):
    """Mirror step03 preclassify-loop AFTER P198cg BAHL group lookup."""
    final = {}
    prev_type = None
    bahl_map = _detect_bahl(pages)
    for pn in sorted(pages):
        t = pages[pn]
        if len(t.strip()) < 80:
            final[pn] = 'Blank Page'
            continue
        # P198cg BAHL group: first page is fresh, others are continuation
        if pn in bahl_map:
            msg_num, mt_type, msg_pgs = bahl_map[pn]
            if mt_type:
                is_first = (pn == msg_pgs[0])
                final[pn] = (mt_type, not is_first)
                prev_type = mt_type
                continue
        st = classify(t)
        if st == 'Amendment':
            final[pn] = ('Amendment', False); prev_type = 'Amendment'
        elif st == 'LC':
            final[pn] = ('LC', False); prev_type = 'LC'
        elif st and st.startswith('MT'):
            final[pn] = (st, False); prev_type = st
        elif st == '_swift_continuation' and prev_type:
            final[pn] = (prev_type, True)
        else:
            final[pn] = ('VLM-decides', False)
    return final


# ──────────────────────────────────────────────────────────────
# Actual job 46660d08 pages 1-6
# ──────────────────────────────────────────────────────────────
with open('results/46660d08-ae1c-44e7-972b-05ea13fc1fe6/step02/step02_result.json',
          encoding='utf-8') as f:
    s2 = json.load(f)
pages = {p.get('page_number'): (p.get('cleaned_text') or p.get('raw_text') or '')
         for p in s2.get('pages', [])}

print("=" * 78)
print("Actual job 46660d08 — MT707 amendment (pages 1-2) + MT700 LC (pages 3-6)")
print("=" * 78)
expected = {
    1: ('Amendment', False),
    2: ('Amendment', True),   # continuation of page 1
    3: ('LC', False),
    4: ('LC', True),
    5: ('LC', True),
    6: ('LC', True),
}
final = resolve_chain({k: pages[k] for k in (1, 2, 3, 4, 5, 6) if k in pages})
passed = 0
for pn in (1, 2, 3, 4, 5, 6):
    got = final.get(pn)
    exp = expected[pn]
    ok = 'OK' if got == exp else 'FAIL'
    if ok == 'OK':
        passed += 1
    regex_st = classify(pages.get(pn, ''))
    print(f'  [{ok}] page {pn}: classify={regex_st!r:28}  final={got}  (expected {exp})')
print(f'  {passed}/6 real-job pages correct')


# ──────────────────────────────────────────────────────────────
# Synthetic MT707 scenarios
# ──────────────────────────────────────────────────────────────
AMEND_P1 = (
    "Message Details #1\n"
    "Unique Message Identifier: IPCBCCNBJSDZ707XXX\n"
    "Format: Swift Sub-Format: Input\n"
    "Identifier: fin.707 Expansion: Amendment to a Documentary Credit\n"
    "Sender: BAHLPKKA\nReceiver: PCBCCNBJ\n"
    "F20: Documentary Credit Number\nREF123\n"
    "F21: Related Reference Number\nORIG-REF-456\n"
    "F26E: Number of Amendment 1\n"
    "F30: Date of Amendment 250905\n"
)
AMEND_P2_F45B = (
    "F45B: Description of Goods (Amended)\n"
    "LEVODOPA\nQUANTITY: 250 KGS\nAT THE RATE OF USD 45/KG\n"
    "CARBIDOPA\nQUANTITY: 30 KGS\nAT THE RATE OF USD 206/KG\n"
    "FCA ANY AIRPORT IN CHINA INCOTERMS 2020\n"
    "F46B: Documents Required (Amended)\n"
    "/DELETE/ CLAUSE 2,3,5,7\n"
    "/ADD/ New clauses...\n"
)
AMEND_P3_F47B = (
    "F47B: Additional Conditions (Amended)\n"
    "Delete existing clause 4 and add new wording.\n"
    "F72Z: Sender to Receiver\nPlease acknowledge.\n"
)
LC_P1 = (
    "Message Details #2\nIdentifier: fin.700 Expansion: Issue of a Documentary Credit\n"
    "F40A: Irrevocable\nF20: REF123\nF31D: 260101 CHINA\n"
    "F45A: Goods\nRBD PALM OLEIN 250 MT\n"
)


synth_cases = [
    dict(
        label="[1] 2-page MT707 amendment (fin.707 + F45B continuation)",
        pages={1: AMEND_P1, 2: AMEND_P2_F45B},
        expected={
            1: ('Amendment', False),
            2: ('Amendment', True),
        },
    ),
    dict(
        label="[2] 3-page MT707 amendment (header + F45B + F47B)",
        pages={1: AMEND_P1, 2: AMEND_P2_F45B, 3: AMEND_P3_F47B},
        expected={
            1: ('Amendment', False),
            2: ('Amendment', True),
            3: ('Amendment', True),
        },
    ),
    dict(
        label="[3] MT707 + MT700 in one PDF",
        pages={1: AMEND_P1, 2: AMEND_P2_F45B, 3: LC_P1},
        expected={
            1: ('Amendment', False),
            2: ('Amendment', True),
            3: ('LC', False),
        },
    ),
    dict(
        label="[4] MT707 with blank page between header and F45B",
        pages={1: AMEND_P1, 2: '', 3: AMEND_P2_F45B},
        expected={
            1: ('Amendment', False),
            2: 'Blank Page',
            3: ('Amendment', True),   # P198cc blank-tolerance preserves prev
        },
    ),
    dict(
        label="[5] Standalone amendment with only 26E body (no fin.707)",
        pages={1: 'F26E: Number of Amendment 2\nF30: Date of Amendment 260101\n'
                  'F31C: Date of Issue 250115\n'
                  'F20: REF999 F23: ORIG-REF-111\n'
                  'F32B: USD 50,000.00\n'
                  'F45B: Goods amended to new description.\n'},
        expected={1: ('Amendment', False)},
    ),
    dict(
        label="[6] MT708 (Amendment to Confirmation) — also in amend family",
        pages={1: (
            'Message Details #1\n'
            'Identifier: fin.708 Expansion: Amendment to a Confirmation\n'
            'Sender: BAHLPKKA\nReceiver: PCBCCNBJ\n'
            'F20: Documentary Credit Number REF999/2026\n'
            'F21: Related Reference ABC-123-456\n'
            'F26E: Number of Amendment 1\n'
            'F30: Date of Amendment 260101\n'
            'F32B: USD 50,000.00\n'
        )},
        expected={1: ('Amendment', False)},
    ),

    dict(
        label="[7] Real-job style: BAHL with MT707 (2 pages) + MT700 (4 pages)",
        pages={
            1: AMEND_P1 + 'Page 1 of 6\n',
            2: AMEND_P2_F45B + 'Page 2 of 6\n',
            3: LC_P1 + 'Page 3 of 6\n',
            4: 'F45A: Description of Goods\nF46A: Documents Required\nPage 4 of 6\n' + ('x' * 100),
            5: 'F47A: Additional Conditions\nPage 5 of 6\n' + ('x' * 100),
            6: 'F72Z: Sender to Receiver\nPage 6 of 6\n' + ('x' * 100),
        },
        expected={
            1: ('Amendment', False),
            2: ('Amendment', True),
            3: ('LC', False),
            4: ('LC', True),
            5: ('LC', True),
            6: ('LC', True),
        },
    ),
]

# Add extra Message Details headers to scenario [7] so BAHL detection kicks in.
scenario_7 = synth_cases[-1]
scenario_7['pages'][1] = 'Message Details #1\n' + scenario_7['pages'][1]
scenario_7['pages'][3] = 'Message Details #2\n' + scenario_7['pages'][3]

print()
print("=" * 78)
print("Synthetic MT707 / amendment scenarios")
print("=" * 78)
s_passed = 0
for case in synth_cases:
    final = resolve_chain(case['pages'])
    ok_all = True
    print(f"\n{case['label']}")
    for pn in sorted(case['pages']):
        got = final.get(pn)
        exp = case['expected'].get(pn)
        ok = got == exp
        if not ok:
            ok_all = False
        mark = '+' if ok else '-'
        regex_st = classify(case['pages'][pn])
        print(f'   [{mark}] page {pn}: classify={regex_st!r:28} final={got}  (expected {exp})')
    if ok_all:
        s_passed += 1
print()
print(f"Synthetic: {s_passed}/{len(synth_cases)} full-scenario matches")

print()
print("=" * 78)
print(f"Real job: {passed}/6 pages | Synthetic: {s_passed}/{len(synth_cases)}")
print("=" * 78)
