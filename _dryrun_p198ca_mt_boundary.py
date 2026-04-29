"""Dry-run P198ca SWIFT preclassification against actual job 6a1da830 data.

Mirrors step03_sequencing.py's regex-based `_page_swift_type` assignment
and confirms that page 1 = MT730 / page 2 = LC / pages 3-5 = _swift_continuation
(which will inherit LC since prev_swift_type was updated to LC on page 2)."""
import json
import re

# ── patterns mirroring step03_sequencing.py (with P198ca addition) ──
_SWIFT_LC_PATTERNS = [
    r'Message\s+type:\s*700',
    r'SWIFT_MT700',
    r'SWIFT_MT\s*700\b',
    # P198ca — Alliance / BAHL report format
    r'\bfin\.\s*700\b',
    r'\bfin\.\s*701\b',
    r'\bIdentifier\s*:\s*fin\.\s*700\b',
    r'\bIdentifier\s*:\s*fin\.\s*701\b',
    r'\bIssue\s+of\s+a\s+Documentary\s+Credit\b',
    r'(?:^|\n)\s*:46A:',
    r'(?:^|\n)\s*F46A\s*:',
    r'(?:^|\n)\s*:40A:',
    r'(?:^|\n)\s*F40A\s*:',
    r'(?:^|\n)\s*:31D:',
    r'(?:^|\n)\s*F31D\s*:',
    r'(?:^|\n)\s*20:\s*Documentary\s+Credit\s+Number',
    r'(?:^|\n)\s*40A:\s*Form\s+of\s+Documentary\s+Credit',
    r'(?:^|\n)\s*31D:\s*Date\s+and\s+Place\s+of\s+Expiry',
    r'(?:^|\n)\s*46A:\s*Documents?\s+Required',
]

_SWIFT_LC_CONT_PATTERNS = [
    r'Message\s+type:\s*701',
    r'SWIFT_MT701',
]

_SWIFT_NON_LC_PATTERNS = [
    (r'Message\s+type:\s*730|fin\.\s*730|SWIFT_MT\s*730|\bMT[\s_]?730\b', 'MT730'),
    (r'Message\s+type:\s*754|fin\.\s*754|SWIFT_MT\s*754|\bMT[\s_]?754\b', 'MT754'),
    (r'Message\s+type:\s*740|fin\.\s*740|SWIFT_MT\s*740|\bMT[\s_]?740\b', 'MT740'),
    (r'Message\s+type:\s*747|fin\.\s*747|SWIFT_MT\s*747|\bMT[\s_]?747\b', 'MT747'),
    (r'Message\s+type:\s*760|fin\.\s*760|SWIFT_MT\s*760|\bMT[\s_]?760\b', 'MT760'),
]

_SWIFT_CONTINUATION_PATTERNS = [
    r'(?:^|\n)\s*(?::|\bF)45A[\s:]+',
    r'(?:^|\n)\s*(?::|\bF)45B[\s:]+',
    r'(?:^|\n)\s*(?::|\bF)46A[\s:]+',
    r'(?:^|\n)\s*(?::|\bF)46B[\s:]+',
    r'(?:^|\n)\s*(?::|\bF)47A[\s:]+',
    r'(?:^|\n)\s*(?::|\bF)47B[\s:]+',
    r'(?:^|\n)\s*(?::|\bF)78[\s:]+',
    r'(?:^|\n)\s*(?::|\bF)72[Z]?[\s:]+',
    r'(?:^|\n)\s*(?::|\bF)71[BD]?[\s:]+',
    r'(?:^|\n)\s*(?::|\bF)49[\s:]+',
]


def classify(text):
    non_lc_mt = None
    for pat, mt in _SWIFT_NON_LC_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            non_lc_mt = mt
            break
    is_lc = any(re.search(p, text, re.IGNORECASE) for p in _SWIFT_LC_PATTERNS)
    is_lc_cont = any(re.search(p, text, re.IGNORECASE) for p in _SWIFT_LC_CONT_PATTERNS)
    is_swift_cont = any(re.search(p, text, re.IGNORECASE) for p in _SWIFT_CONTINUATION_PATTERNS)
    # Priority order (matches production):
    if non_lc_mt:
        return non_lc_mt
    if is_lc_cont:
        return 'LC'
    if is_lc:
        return 'LC'
    if is_swift_cont:
        return '_swift_continuation'
    return ''


# ── Load actual job 6a1da830 text ──
with open('results/6a1da830-022d-497f-aadc-0b5625007611/step02/step02_result.json',
          encoding='utf-8') as f:
    s2 = json.load(f)
pages = {p.get('page_number'): (p.get('cleaned_text') or p.get('raw_text') or '')
         for p in s2.get('pages', [])}

print("=" * 78)
print("P198ca SWIFT MT boundary preclassification — real job 6a1da830 data")
print("=" * 78)

# ── Simulate the production loop: iterate pages 1..5, track prev_swift_type ──
prev_swift_type = None
per_page_final = {}
expected = {1: 'MT730', 2: 'LC', 3: 'LC', 4: 'LC', 5: 'LC'}
for pn in (1, 2, 3, 4, 5):
    text = pages.get(pn, '')
    raw = classify(text)

    # Reproduce the loop logic from step03_sequencing.py:2810-2910
    if raw == 'LC':
        doc_type = 'LC'
        is_cont = False
        prev_swift_type = 'LC'
    elif raw == 'MT730':
        doc_type = 'MT730'
        is_cont = False
        prev_swift_type = 'MT730'
    elif raw.startswith('MT') and raw not in ('MT799', 'MT999'):
        doc_type = raw
        is_cont = False
        prev_swift_type = raw
    elif raw == '_swift_continuation' and prev_swift_type:
        doc_type = prev_swift_type
        is_cont = True
    else:
        doc_type = '(VLM decides)'
        is_cont = '(VLM decides)'
    per_page_final[pn] = (doc_type, is_cont)
    exp = expected.get(pn, '?')
    ok = 'OK' if doc_type == exp else 'FAIL'
    print(f'  [{ok}] page {pn}: regex_st={raw!r:22} → doc_type={doc_type:12} cont={is_cont}  (expected {exp})')

# ── Verify page-2 regex hits: show which pattern matched ──
print()
print("--- Page 2 regex hit details ---")
page2 = pages.get(2, '')
for p in _SWIFT_LC_PATTERNS:
    if re.search(p, page2, re.IGNORECASE):
        m = re.search(p, page2, re.IGNORECASE)
        print(f'  [HIT]   pattern {p!r} matched {m.group(0)!r}')

# ── Negative: ensure page 1 is NOT classified as LC ──
print()
print("--- Page 1 (MT730) — verify it does NOT get classified as LC ---")
page1 = pages.get(1, '')
for p in _SWIFT_LC_PATTERNS:
    m = re.search(p, page1, re.IGNORECASE)
    if m:
        print(f'  [UNEXPECTED HIT] pattern {p!r} matched {m.group(0)!r}')
print("  (no unexpected hits means MT730 regex correctly wins over LC regex)")

print()
print("=" * 78)
print("BAHL multi-message detection (threshold: 2+ Message Details headers)")
print("=" * 78)
_BAHL_MSG_RE = re.compile(r'Message\s+Details\s+#\s*(\d+)', re.IGNORECASE)
_FIN_RE = re.compile(r'Identifier\s*:\s*fin\.(\d{3})', re.IGNORECASE)
_FIN_TO_MT = {'700': 'LC', '701': 'LC', '707': 'Amendment',
              '730': 'MT730', '740': 'MT740', '747': 'Amendment',
              '754': 'MT754', '799': 'MT799', '940': 'MT940'}

msg_detail_pages = {}
for pn in (1, 2, 3, 4, 5):
    t = pages.get(pn, '')
    hits = _BAHL_MSG_RE.findall(t)
    if hits:
        msg_detail_pages[pn] = [int(h) for h in hits]
print(f'  Message Details headers: {msg_detail_pages}')
print(f'  Count: {len(msg_detail_pages)}  — BAHL mode triggers at {len(msg_detail_pages) >= 2}')

# Compute _bahl_max_page from "Page X of Y" on the first message header.
_POT_RE = re.compile(r'Page\s+(\d+)\s+of\s+(\d+)', re.IGNORECASE)
_page_of_total = {}
for pn, txt in pages.items():
    m = _POT_RE.search(txt)
    if m:
        _page_of_total[pn] = (int(m.group(1)), int(m.group(2)))
_bahl_max_page = 0
for pn in sorted(msg_detail_pages.keys()):
    if pn in _page_of_total:
        x, y = _page_of_total[pn]
        _bahl_max_page = max(_bahl_max_page, pn + (y - x))
print(f'  _bahl_max_page: {_bahl_max_page}  (pages beyond this are shipping docs)')

# Simulate BAHL boundary: assign each page to the most recent Message Details
# #N, stopping at _bahl_max_page.
current_msg = None
bahl_pages = {}  # msg_num -> [page_nums]
for pn in sorted(pages.keys()):
    if _bahl_max_page and pn > _bahl_max_page:
        break
    if pn in msg_detail_pages:
        current_msg = msg_detail_pages[pn][-1]
        bahl_pages.setdefault(current_msg, []).append(pn)
    elif current_msg is not None:
        bahl_pages.setdefault(current_msg, []).append(pn)

print()
print("  BAHL message grouping:")
for msg_num, pgs in sorted(bahl_pages.items()):
    # Find fin.XXX on the first page
    first_pg = pgs[0]
    txt = pages.get(first_pg, '')
    fm = _FIN_RE.search(txt)
    fin = fm.group(1) if fm else '?'
    mt = _FIN_TO_MT.get(fin, '?')
    print(f'    Message #{msg_num} → fin.{fin} → {mt}  pages={pgs}')

# ── Final boundary assertion ──
print()
passed = sum(1 for pn, (dt, _) in per_page_final.items() if dt == expected.get(pn))
print("=" * 78)
print(f"Preclassifier boundary: {passed}/{len(expected)} pages correct")
print(f"BAHL mode expected grouping: Msg#1 = MT730 (page 1), "
      f"Msg#2 = LC (pages 2-5)")
got_bahl = {k: v for k, v in bahl_pages.items()}
expected_bahl = {1: [1], 2: [2, 3, 4, 5]}
print(f"BAHL mode actual grouping:   {got_bahl}")
print(f"{'OK' if got_bahl == expected_bahl else 'FAIL'}: "
      f"BAHL boundary = {got_bahl == expected_bahl}")
print("=" * 78)
