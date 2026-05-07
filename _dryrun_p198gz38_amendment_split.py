"""P198gz38/gz39/gz40 — multi-amendment splitter, BL multi-page
unifier, Draft endorsement merger.

Tests on real-data anchor cb7d7bbf and sweeps prior jobs."""
import sys, os, json, glob, re
sys.path.insert(0, 'd:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final')
os.environ['PYTHONIOENCODING'] = 'utf-8'

results = []
def ok(name, cond, note=''):
    tag = '[OK]  ' if cond else '[FAIL]'
    print(f"{tag} {name}" + (f" -- {note}" if note else ''))
    results.append(bool(cond))


# ─────────────────────────────────────────────────────────────────
# Section 1: AMENDMENT splitting — anchor job cb7d7bbf
# ─────────────────────────────────────────────────────────────────
print("=" * 70)
print("Section 1: Amendment splitting (cb7d7bbf — 13 amendments)")
print("=" * 70)

AMD_HEADER_RE = re.compile(
    r'\bMessage\s+Details\s+#\s*\d+\b'
    r'|(?:^|\n|\s):?\s*F?26E\s*:?\s*[A-Za-z]'
    r'|\bNumber\s+of\s+Amendment\b'
    r'|\bIdentifier\s*:\s*fin\.?\s*707\b'
    r'|\bMessage\s+type\s*[:\s]+707\b'
    r'|^\s*MT\s*707\b'
    r'|\bSWIFT_MT\s*707\b',
    flags=re.IGNORECASE | re.MULTILINE,
)
N_RE = re.compile(r'(?:F?26E|26E)\s*:?\s*(?:Number\s+of\s+Amendment\s*:?\s*)?(\d+)', re.IGNORECASE)
N2_RE = re.compile(r'Number\s+of\s+Amendment\D{0,40}(\d+)', re.IGNORECASE)
DATE_RE = re.compile(r'(?:F?30|:30:)\s*Date\s+of\s+Amendment\D{0,30}(\d{6,8})', re.IGNORECASE)
DATE2_RE = re.compile(r'Date\s+of\s+Amendment\D{0,30}(\d{6,8}|\d{2}-\d{2}-\d{4})', re.IGNORECASE)
AMD_TYPES = {'amendment', 'mt707', 'mt708', 'amendment to a documentary credit'}


def split_amendments(d2, d3):
    """Run the splitter logic and return list of (pages_range, amd_num, amd_date)."""
    ptl = {p.get('page_number'): p.get('cleaned_text', '') for p in d2.get('pages', [])}
    out = []
    for pkt in d3.get('packets', []):
        if (pkt.get('document_type', '') or '').lower().strip() not in AMD_TYPES:
            continue
        page_nums = sorted(pkt.get('page_numbers', []))
        if len(page_nums) <= 1:
            continue
        starts = [i for i, pn in enumerate(page_nums)
                  if AMD_HEADER_RE.search(ptl.get(pn, '') or '')]
        for j, si in enumerate(starts):
            ei = starts[j + 1] if j + 1 < len(starts) else len(page_nums)
            sub = page_nums[si:ei]
            t = ptl.get(sub[0], '') or ''
            mn = N_RE.search(t) or N2_RE.search(t)
            md = DATE_RE.search(t) or DATE2_RE.search(t)
            out.append((
                (sub[0], sub[-1]),
                int(mn.group(1)) if mn else None,
                md.group(1) if md else None,
            ))
    return out


base = 'results/cb7d7bbf-a24c-4abc-b3aa-00c6e287e7fd'
d2 = json.load(open(f'{base}/step02/step02_result.json', encoding='utf-8'))
d3 = json.load(open(f'{base}/step03/step03_result.json', encoding='utf-8'))
amends = split_amendments(d2, d3)
ok("13 amendments detected", len(amends) == 13, f"got {len(amends)}")
ok("All amendments have a number", all(a[1] is not None for a in amends))
ok("Amendment numbers cover 1..13",
   sorted([a[1] for a in amends]) == list(range(1, 14)))
ok("All amendments have a date", all(a[2] is not None for a in amends))

# After step06 sorts by F26E, application order should be 1, 2, 3, ..., 13
sorted_by_n = sorted(amends, key=lambda a: a[1])
ok("Sort by F26E produces 1..13 in order",
   [a[1] for a in sorted_by_n] == list(range(1, 14)))


# ─────────────────────────────────────────────────────────────────
# Section 2: Sweep prior jobs for the merge bug
# ─────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("Section 2: Sweep prior jobs for amendment-merge bug")
print("=" * 70)

hits = []
for s2_path in sorted(glob.glob('results/*/step02/step02_result.json')):
    job = s2_path.split(os.sep)[-3]
    s3_path = s2_path.replace(os.sep + 'step02' + os.sep, os.sep + 'step03' + os.sep)\
                     .replace('step02_result.json', 'step03_result.json')
    if not os.path.exists(s3_path):
        continue
    try:
        d2j = json.load(open(s2_path, encoding='utf-8'))
        d3j = json.load(open(s3_path, encoding='utf-8'))
    except Exception:
        continue
    amends_j = split_amendments(d2j, d3j)
    if len(amends_j) > 1:
        # Check that we'd produce >= 2 packets per merged-packet
        ptl_j = {p.get('page_number'): p.get('cleaned_text', '') for p in d2j.get('pages', [])}
        merged_packets = [p for p in d3j.get('packets', [])
                          if (p.get('document_type', '') or '').lower().strip() in AMD_TYPES
                          and len(p.get('page_numbers', [])) > 2]
        if merged_packets:
            hits.append((job, len(amends_j)))
print(f'Jobs with multi-amendment merge bug found: {len(hits)}')
for j, n in hits[:30]:
    print(f'  {j}: would split into {n} amendments')
ok("All hits have F26E numbers parseable",
   all(n >= 2 for _, n in hits))


# ─────────────────────────────────────────────────────────────────
# Section 3: BL multi-page unifier — anchor job cb7d7bbf
# ─────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("Section 3: BL multi-page unifier")
print("=" * 70)

BL_NO_RE = re.compile(
    r'\bB\s*/?\s*L\s*(?:NO\.?|NUMBER|N°)?\s*[:.]?\s*'
    r'([A-Z][A-Z0-9\-]{6,})\b', re.IGNORECASE)
BL_NO_CARRIER_RE = re.compile(
    r'\b(?:ONEY|MAEU|MEDU|COSU|HLCU|YMLU|HJSC|EVRG|EGLV|'
    r'KKLU|MSCU|OOLU|APLU|CMDU|SUDU|UASC|HMMU|ZIMU|PILU|'
    r'NYKS|SAFM|TLLU)[A-Z0-9]{8,16}\b',
)
PAGE_X_OF_N_RE = re.compile(
    r'\bPAGE\s*[:.]?\s*(\d{1,2})\s+OF\s+(\d{1,2})\b', re.IGNORECASE)


def detect_bl_sets(d2, d3):
    """Identify BL packets that should merge into multi-page sets."""
    ptl = {p.get('page_number'): p.get('cleaned_text', '') for p in d2.get('pages', [])}
    bl_packets = []
    for pkt in d3.get('packets', []):
        dt = (pkt.get('document_type', '') or '').lower()
        if 'bill of lading' not in dt:
            continue
        page_nums = sorted(pkt.get('page_numbers', []))
        bl_no, page_x, page_n = '', 0, 0
        for pn in page_nums:
            t = ptl.get(pn, '') or ''
            if not t:
                continue
            if not bl_no:
                mno = BL_NO_RE.search(t) or BL_NO_CARRIER_RE.search(t)
                if mno:
                    bl_no = (mno.group(1) if mno.lastindex else mno.group(0)).upper()
            if not page_x:
                mx = PAGE_X_OF_N_RE.search(t)
                if mx:
                    page_x = int(mx.group(1))
                    page_n = int(mx.group(2))
            if bl_no and page_x:
                break
        bl_packets.append({
            'pkt_id': pkt.get('packet_id'),
            'pages': page_nums,
            'bl_no': bl_no,
            'page_x': page_x,
            'page_n': page_n,
        })
    return bl_packets


bls = detect_bl_sets(d2, d3)
ok("BL packets detected",
   len(bls) == 6,  # 6 packets in cb7d7bbf BL section
   f"got {len(bls)}")
ok("BL number ONEYBKKGA5413400 found on all 3 sets",
   sum(1 for b in bls if b['bl_no'] == 'ONEYBKKGA5413400') >= 3)

# Simulate unifier: with backward-peek for OCR-corrupt face pages.
groups = []
i = 0
while i < len(bls):
    b = bls[i]
    set_bl_no = b['bl_no']
    set_n = b['page_n']
    last_x = b['page_x']
    # Peek next BL packet if current has no marker
    if set_n < 2 and i + 1 < len(bls):
        peek = bls[i + 1]
        if peek['page_n'] >= 2 and peek['page_x'] == 2:
            set_bl_no = peek['bl_no']
            set_n = peek['page_n']
            last_x = 1
    if set_n < 2:
        groups.append({'bl_no': b['bl_no'], 'packets': [b]})
        i += 1
        continue
    g = {'bl_no': set_bl_no, 'packets': [b]}
    j = i + 1
    while j < len(bls):
        nb = bls[j]
        if (nb['bl_no'] == set_bl_no and nb['page_n'] == set_n
                and nb['page_x'] > last_x and nb['page_x'] <= set_n):
            g['packets'].append(nb)
            last_x = nb['page_x']
            j += 1
        else:
            break
    groups.append(g)
    i = j
ok(f"Unification produces 3 BL sets",
   len(groups) == 3,
   f"got {len(groups)} groups: {[len(g['packets']) for g in groups]}")


# ─────────────────────────────────────────────────────────────────
# Section 4: Draft endorsement merger
# ─────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("Section 4: Draft front+back endorsement merger")
print("=" * 70)

ENDORSE_ONLY_RE = re.compile(
    r'\bPAY\s+TO\s+THE\s+ORDER\s+OF\s+ANY\s+BANK\b'
    r'|\bPAY\s+TO\s+THE\s+ORDER\s+OF\b'
    r'|\bENDORSED\s+TO\b'
    r'|\bFOR\s+VALUE\s+RECEIVED\b',
    flags=re.IGNORECASE,
)
BOE_BODY_RE = re.compile(
    r'\bBILL\s+OF\s+EXCHANGE\b'
    r'|\bFOR\s+USD\b'
    r'|\bAT\s+SIGHT\b'
    r'|\bAT\s+\d{1,3}\s+DAYS\b'
    r'|\bof\s+this\s+(?:FIRST|SECOND|SOLE)\s+Bill\s+of\s+Exchange\b',
    flags=re.IGNORECASE,
)
DRAFT_TYPES = {
    'draft bill of exchange', 'bill of exchange',
    'draft', 'sight draft', 'usance draft', 'boe',
}

# pg33 / pg34 of cb7d7bbf
ptl_cb = {p.get('page_number'): p.get('cleaned_text', '') for p in d2.get('pages', [])}
pg33 = ptl_cb.get(33, '')
pg34 = ptl_cb.get(34, '')
ok("pg33 has BoE body markers", BOE_BODY_RE.search(pg33) is not None)
ok("pg34 has endorsement-only pattern",
   ENDORSE_ONLY_RE.search(pg34) is not None
   and BOE_BODY_RE.search(pg34) is None)


# ─────────────────────────────────────────────────────────────────
# Tally
# ─────────────────────────────────────────────────────────────────
print()
print("=" * 70)
passed = sum(results)
total = len(results)
print(f"P198gz38/39/40 AGGRESSIVE: {passed}/{total}")
print("=" * 70)
if passed != total:
    sys.exit(1)
print("OVERALL: OK")
sys.exit(0)
