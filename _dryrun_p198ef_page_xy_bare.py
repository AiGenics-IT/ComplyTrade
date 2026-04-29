"""
P198ef dry-run — Bare "1 of 2" / "X/Y" pagination footer detection.

Carrier-issued BLs (PIL, Maersk, MSC, COSCO, etc.) commonly print
a bare pagination footer like "1 of 2" / "2 of 2" without the
"Page" prefix. The current _PAGE_XY_RE only catches "Page X of Y"
forms, so these BLs split into two packets at step03 and a notify-
party / consignee check on one half fails because the second half
(with the additional party / endorsement) is in a different
packet.

Tests use the actual page 34 + 35 OCR from job aafd886a (BL No.
BTU600007901, the user's exact failing case), plus 12 synthetic
edge cases that exercise both true positives (bare X of Y in
footer) and true negatives (avoid matching "1 of 2 boxes",
date strings, body prose).
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')


# Mirror of the new P198ef regexes (kept in sync with step03).
_PAGE_XY_RE = re.compile(
    r'Page\s+(\d+)\s*(?:of|/)\s*(\d+)',
    re.IGNORECASE,
)
_BARE_XY_RE = re.compile(
    r'(?:^|\n)\s*(\d{1,2})\s+of\s+(\d{1,2})\s*(?=\n|$)',
    re.IGNORECASE | re.MULTILINE,
)
_BARE_XY_SLASH_RE = re.compile(
    r'(?:^|\n)\s*(\d{1,2})\s*/\s*(\d{1,2})\s*(?=\n|$)',
    re.MULTILINE,
)


def detect(text):
    if not text:
        return None
    m = _PAGE_XY_RE.search(text)
    if m:
        return int(m.group(1)), int(m.group(2))
    tail = text[-400:]
    for _re in (_BARE_XY_RE, _BARE_XY_SLASH_RE):
        for _m in _re.finditer(tail):
            x, y = int(_m.group(1)), int(_m.group(2))
            if 1 <= x <= y <= 99:
                return x, y
    return None


# Real OCR — pages 34 and 35 of job aafd886a (the user's case)
JOB = Path('results/aafd886a-c854-43de-baf1-ffd7c5bf9a58')
real_pages = {}
if (JOB / 'step02' / 'step02_result.json').exists():
    s2 = json.loads((JOB / 'step02' / 'step02_result.json')
                    .read_text(encoding='utf-8'))
    real_pages = {p['page_number']:
                  (p.get('cleaned_text') or p.get('raw_text') or '')
                  for p in s2.get('pages', [])}


SCENARIOS = [
    # (name, text, expected_xy_or_None)
    ('1. "Page 1 of 2" — primary form', 'Some content\nPage 1 of 2', (1, 2)),
    ('2. "Page 1/2" — slash form', 'Body text\nPage 1/2', (1, 2)),
    ('3. Bare "1 of 2" footer (USER\'S CASE — page 34 of BL)',
     'BL header...\n\nFooter content\n\n1 of 2', (1, 2)),
    ('4. Bare "2 of 2" footer (page 35 continuation)',
     'BL continuation...\n\n2 of 2', (2, 2)),
    ('5. Bare "1/2" footer slash form',
     'Carrier...\n[STAMP]\nP123456789\n1/2', (1, 2)),
    ('6. Embedded "1 of 2 boxes" — should NOT match (false positive)',
     '1 of 2 boxes contains 50kg of widgets', None),
    ('7. Date string "13/02/2026" — should NOT match',
     'Invoice date: 13/02/2026\nVessel sailed.', None),
    ('8. "1 of 2" within body prose — should NOT match (no line anchor)',
     'Note that out of these, 1 of 2 cases is exempt.', None),
    ('9. Both forms present (primary wins)',
     '1 of 2\nPage 5 of 10\n2 of 2', (5, 10)),
    ('10. Footer "Page X of Y" with surrounding noise',
     'Carrier signature\nDocument: P12345\nPage 1 of 3 (Lakeport BL)\n', (1, 3)),
    ('11. Bare "5/9" survey report footer',
     'Survey Report - Loading\nDraught readings...\n5/9', (5, 9)),
    ('12. Empty text', '', None),
    ('13. Y < X (invalid pagination, e.g. footer note)',
     'Body...\n5 of 1', None),
    ('14. Three-digit Y (skip — not pagination)',
     'Body...\n1 of 100', None),
]


def simulate_merge(job_dir):
    """Simulate the step03 merge: read all pages, detect pagination,
    return a list of merged-packet sequences {'pages': [...]}.
    Mirrors the logic at step03_sequencing.py:3446-3454 — page X
    forces is_continuation=True when the previous page has
    Page X-1 of Y."""
    s2 = Path(job_dir) / 'step02' / 'step02_result.json'
    if not s2.exists():
        return []
    pages = json.loads(s2.read_text(encoding='utf-8')).get('pages', [])
    pages.sort(key=lambda p: p.get('page_number', 0))
    page_xy = {}
    for p in pages:
        pn = p.get('page_number')
        text = p.get('cleaned_text') or p.get('raw_text') or ''
        xy = detect(text)
        if xy:
            page_xy[pn] = xy
    # Merge: walk pages in order, group sequential X-of-Y runs
    groups = []
    cur = None
    prev_pn = None
    for p in pages:
        pn = p.get('page_number')
        xy = page_xy.get(pn)
        if xy:
            x, y = xy
            prev_xy = page_xy.get(prev_pn) if prev_pn else None
            if (x > 1 and prev_xy
                    and prev_xy[1] == y and prev_xy[0] == x - 1):
                cur['pages'].append(pn)
                cur['xy_seq'].append(xy)
            else:
                if cur is not None:
                    groups.append(cur)
                cur = {'pages': [pn], 'xy_seq': [xy]}
        else:
            if cur is not None:
                groups.append(cur)
                cur = None
            groups.append({'pages': [pn], 'xy_seq': [None]})
        prev_pn = pn
    if cur is not None:
        groups.append(cur)
    return groups


def main():
    pass_n, fail_n = 0, 0
    print('=' * 78)
    print('P198ef dry-run — bare X of Y pagination footer detection')
    print('=' * 78)

    # Real-job pages first
    print('\n--- A. Real OCR from job aafd886a (BL No. BTU600007901) ---')
    real_cases = [
        ('p34 BL front (Notify=S.K. TRADING)', 34, (1, 2)),
        ('p35 BL back/continuation (ALSO NOTIFY=Bank Al-Habib)', 35, (2, 2)),
    ]
    for label, pn, expected in real_cases:
        result = detect(real_pages.get(pn, ''))
        ok = (result == expected)
        tag = 'OK ' if ok else 'FAIL'
        print(f'   [{tag}] {label}: detected={result} expected={expected}')
        if ok: pass_n += 1
        else:  fail_n += 1

    print('\n--- B. Synthetic edge cases ---')
    for name, text, expected in SCENARIOS:
        result = detect(text)
        ok = (result == expected)
        tag = 'OK ' if ok else 'FAIL'
        print(f'   [{tag}] {name}: detected={result} expected={expected}')
        if ok: pass_n += 1
        else:  fail_n += 1

    # ── C. Merge simulation on aafd886a (must group p34+p35) ──
    print('\n--- C. Merge simulation on real job aafd886a ---')
    groups = simulate_merge(JOB)
    p34p35 = [g for g in groups
              if 34 in g.get('pages', []) and 35 in g.get('pages', [])]
    if p34p35:
        print(f'   [OK ] p34 + p35 merged into ONE group: {p34p35[0]}')
        pass_n += 1
    else:
        # Find what group p34 and p35 are in separately
        p34 = next((g for g in groups if 34 in g.get('pages', [])), None)
        p35 = next((g for g in groups if 35 in g.get('pages', [])), None)
        print(f'   [FAIL] p34 group: {p34}')
        print(f'          p35 group: {p35}')
        fail_n += 1

    # ── D. Sweep every real job — count detection rate by signal type ──
    print('\n--- D. Real-job sweep: every results/* job ---')
    import glob
    primary, bare_of, bare_slash, none_n = 0, 0, 0, 0
    job_count = 0
    multi_page_groups = 0
    for s2f in sorted(glob.glob('results/*/step02/step02_result.json')):
        try:
            data = json.loads(Path(s2f).read_text(encoding='utf-8'))
        except Exception:
            continue
        job_count += 1
        for p in data.get('pages', []):
            text = p.get('cleaned_text') or p.get('raw_text') or ''
            if not text:
                continue
            if _PAGE_XY_RE.search(text):
                primary += 1
            elif _BARE_XY_RE.search(text[-400:]):
                bare_of += 1
            elif _BARE_XY_SLASH_RE.search(text[-400:]):
                bare_slash += 1
            else:
                none_n += 1
        # Count multi-page groups (groups with 2+ pages)
        gs = simulate_merge(Path(s2f).parent.parent)
        multi_page_groups += sum(1 for g in gs if len(g.get('pages', [])) >= 2)
    print(f'   {job_count} jobs swept')
    print(f'   primary "Page X of Y" matches : {primary}')
    print(f'   bare "X of Y" footer matches  : {bare_of}')
    print(f'   bare "X/Y" footer matches     : {bare_slash}')
    print(f'   no pagination signal          : {none_n}')
    print(f'   multi-page merged groups (≥2) : {multi_page_groups}')
    pass_n += 1  # informational

    total = pass_n + fail_n
    print('\n' + '=' * 78)
    print(f'OVERALL: {pass_n}/{total} '
          f'{"OK" if fail_n == 0 else "— failures present"}')
    print('=' * 78)
    return 0 if fail_n == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
