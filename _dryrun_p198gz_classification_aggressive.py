"""P198gz38-44 — comprehensive classification & merging validator.

Sweeps ALL jobs and simulates each fix. Reports:
- Amendments that need splitting (P198gz38)
- BL multi-page sets that need unifying (P198gz39)
- Draft front+back that need merging (P198gz40)
- Multi-doc CI/PL that need splitting (P198gz41)
- Numbered sub-docs whose suffix was dropped (P198gz42)
- PL continuation marked as CI (P198gz43)
- Same-N Page-of-N continuations needing merge (P198gz44)
"""
import sys, os, json, glob, re
sys.path.insert(0, 'd:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final')
os.environ['PYTHONIOENCODING'] = 'utf-8'

# ─── Detection patterns ──────────────────────────────────────────
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
AMD_TYPES = {'amendment', 'mt707', 'mt708', 'amendment to a documentary credit'}

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

PL_COLS = ('NET-WEIGHT', 'NET WEIGHT', 'GROSS-WEIGHT',
           'GROSS WEIGHT', 'MEASUREMENT')
CI_COLS = ('UNIT PRICE', 'TOTAL AMOUNT', 'AMOUNT IN USD',
           'AMOUNT IN EUR', 'AMOUNT (USD)', 'AMOUNT (EUR)',
           'AMOUNT/USD', 'TOTAL/USD', 'INVOICE VALUE',
           'INVOICE TOTAL', 'TOTAL VALUE',
           'TOTAL CFR', 'TOTAL FOB', 'TOTAL CIF',
           'PRICE PER UNIT')

PXY_RE = re.compile(r'\bPage\s+(\d{1,2})\s*(?:of|/)\s*(\d{1,2})\b',
                    flags=re.IGNORECASE)
PXY_SKIP = {
    'lc', 'amendment', 'mt700', 'mt701', 'mt707', 'mt708',
    'mt799', 'mt999', 'mt705', 'mt710', 'mt711', 'mt720',
    'mt721', 'mt730', 'mt740',
}

SUFFIXABLE_TYPES = {
    'certificate', 'sending report', 'beneficiary certificate',
    'shipping advice', 'shipment advice', 'shipment notice',
    'sales contract', 'proforma invoice', 'commercial invoice',
    'packing list',
}

SPLIT_TYPES = {
    'commercial invoice', 'invoice', 'packing list',
    'weight list', 'weight and packing list',
}


def header_signature(t):
    if not t:
        return ''
    h = t[:300].upper()
    h = re.sub(r'\d+', '', h)
    h = re.sub(r'[^A-Z\s]+', ' ', h)
    h = re.sub(r'\s+', ' ', h).strip()
    return h[:150]


# ─── Per-job analysis ───────────────────────────────────────────
def analyze_job(job_id):
    base = f'results/{job_id}'
    s2_path = f'{base}/step02/step02_result.json'
    s3_path = f'{base}/step03/step03_result.json'
    if not (os.path.exists(s2_path) and os.path.exists(s3_path)):
        return None
    try:
        d2 = json.load(open(s2_path, encoding='utf-8'))
        d3 = json.load(open(s3_path, encoding='utf-8'))
    except Exception:
        return None
    ptl = {p.get('page_number'): p.get('cleaned_text', '') for p in d2.get('pages', [])}
    packets = d3.get('packets', [])

    findings = {
        'amd_split': 0,        # P198gz38
        'bl_unify': 0,         # P198gz39
        'draft_merge': 0,      # P198gz40
        'doc_split': 0,        # P198gz41
        'suffix_lost': 0,      # P198gz42
        'pl_cont_as_ci': 0,    # P198gz43
        'pxy_merge': 0,        # P198gz44
    }

    # P198gz38 — Amendment splits
    for pkt in packets:
        dt = (pkt.get('document_type', '') or '').lower().strip()
        if dt not in AMD_TYPES:
            continue
        page_nums = sorted(pkt.get('page_numbers', []))
        if len(page_nums) <= 1:
            continue
        starts = [i for i, pn in enumerate(page_nums)
                  if AMD_HEADER_RE.search(ptl.get(pn, '') or '')]
        if len(starts) > 1:
            findings['amd_split'] += len(starts) - 1

    # P198gz39 — BL multi-page set unification
    bls = []
    for pkt in packets:
        dt = (pkt.get('document_type', '') or '').lower()
        if 'bill of lading' not in dt:
            continue
        bl_no, x, n = '', 0, 0
        for pn in sorted(pkt.get('page_numbers', [])):
            t = ptl.get(pn, '') or ''
            if not bl_no:
                m = BL_NO_RE.search(t) or BL_NO_CARRIER_RE.search(t)
                if m:
                    bl_no = (m.group(1) if m.lastindex else m.group(0)).upper()
            if not x:
                mx = PAGE_X_OF_N_RE.search(t)
                if mx:
                    x = int(mx.group(1))
                    n = int(mx.group(2))
            if bl_no and x:
                break
        bls.append((bl_no, x, n, sorted(pkt.get('page_numbers', []))))
    # Walk and detect merges (same set_n, sequential x, same bl_no)
    i = 0
    while i < len(bls):
        b = bls[i]
        set_no, set_n, last_x = b[0], b[2], b[1]
        if set_n < 2 and i + 1 < len(bls):
            peek = bls[i + 1]
            if peek[2] >= 2 and peek[1] == 2:
                set_no, set_n, last_x = peek[0], peek[2], 1
        if set_n < 2:
            i += 1; continue
        merged_in_set = 0
        j = i + 1
        while j < len(bls):
            nb = bls[j]
            if (nb[0] == set_no and nb[2] == set_n
                    and nb[1] > last_x and nb[1] <= set_n):
                merged_in_set += 1
                last_x = nb[1]
                j += 1
            else:
                break
        findings['bl_unify'] += merged_in_set
        i = j if merged_in_set else i + 1

    # P198gz40 — Draft front+back endorsement merge
    sorted_idx = sorted(range(len(packets)),
                        key=lambda i: (
                            min(packets[i].get('page_numbers', []))
                            if packets[i].get('page_numbers') else 0))
    for ip in range(len(sorted_idx) - 1):
        i = sorted_idx[ip]; j = sorted_idx[ip + 1]
        pi = packets[i]; pj = packets[j]
        di = (pi.get('document_type', '') or '').lower().strip()
        dj = (pj.get('document_type', '') or '').lower().strip()
        if di not in DRAFT_TYPES or dj not in DRAFT_TYPES:
            continue
        pi_pgs = pi.get('page_numbers', []) or []
        pj_pgs = pj.get('page_numbers', []) or []
        if not pi_pgs or not pj_pgs:
            continue
        if min(pj_pgs) != max(pi_pgs) + 1:
            continue
        pj_first = sorted(pj_pgs)[0]
        pjt = ptl.get(pj_first, '') or ''
        if ENDORSE_ONLY_RE.search(pjt) and not BOE_BODY_RE.search(pjt):
            findings['draft_merge'] += 1

    # P198gz41 — Multi-doc splitter
    for pkt in packets:
        dt = (pkt.get('document_type', '') or '').lower().strip()
        if dt not in SPLIT_TYPES:
            continue
        page_nums = sorted(pkt.get('page_numbers', []))
        if len(page_nums) < 4:
            continue
        first_sig = header_signature(ptl.get(page_nums[0], ''))
        if not first_sig:
            continue
        first_set = set(first_sig.split())
        if not first_set:
            continue
        repeats = 0
        for pn in page_nums[1:]:
            sig = header_signature(ptl.get(pn, ''))
            sig_set = set(sig.split())
            if not sig_set:
                continue
            overlap = len(sig_set & first_set) / max(1, len(first_set))
            if (overlap >= 0.8 and abs(len(sig) - len(first_sig))
                    <= 0.30 * max(len(sig), len(first_sig))):
                repeats += 1
        if repeats > 0:
            findings['doc_split'] += repeats

    # P198gz42 — Numbered sub-doc suffix loss
    for pkt in packets:
        dt = (pkt.get('document_type', '') or '').lower().strip()
        if dt not in SUFFIXABLE_TYPES:
            continue
        # Has the suffix already been preserved? (e.g. "Certificate-1")
        if re.search(r'-\d{1,3}$', pkt.get('document_type', '') or ''):
            continue
        page_nums = pkt.get('page_numbers', [])
        if not page_nums:
            continue
        t = (ptl.get(sorted(page_nums)[0], '') or '')[:400].upper()
        esc = re.escape(dt.upper())
        m = re.search(rf'\??\s*{esc}\s*[-–—]?\s*(\d{{1,3}})\s*\??', t) \
            or re.search(rf'{esc}\s+NO\.?\s+(\d{{1,3}})\b', t)
        if m:
            findings['suffix_lost'] += 1

    # P198gz43 — PL continuation as CI
    for ip in range(len(sorted_idx) - 1):
        i = sorted_idx[ip]; j = sorted_idx[ip + 1]
        pi = packets[i]; pj = packets[j]
        di = (pi.get('document_type', '') or '').lower().strip()
        dj = (pj.get('document_type', '') or '').lower().strip()
        if 'packing list' not in di:
            continue
        if 'commercial invoice' not in dj and dj != 'invoice':
            continue
        pi_pgs = pi.get('page_numbers', []) or []
        pj_pgs = pj.get('page_numbers', []) or []
        if not pi_pgs or not pj_pgs:
            continue
        if min(pj_pgs) != max(pi_pgs) + 1:
            continue
        pj_first = sorted(pj_pgs)[0]
        pjt = (ptl.get(pj_first, '') or '').upper()
        if not pjt or 'COMMERCIAL INVOICE' in pjt[:300]:
            continue
        has_pl = sum(1 for m in PL_COLS if m in pjt) >= 2
        has_ci = any(m in pjt for m in CI_COLS)
        if has_pl and not has_ci:
            findings['pl_cont_as_ci'] += 1

    # P198gz44 — Same-N Page-of-N continuation merger
    def packet_pxy(p):
        min_x = max_x = n = 0
        for pn in sorted(p.get('page_numbers', [])):
            t = ptl.get(pn, '') or ''
            m = PXY_RE.search(t)
            if m:
                x, y = int(m.group(1)), int(m.group(2))
                if y < 2: continue
                if n and y != n: continue
                n = y
                if min_x == 0 or x < min_x: min_x = x
                if x > max_x: max_x = x
        return (min_x, max_x, n)

    consumed = set()
    for ip, i in enumerate(sorted_idx):
        if i in consumed: continue
        pi = packets[i]
        di = (pi.get('document_type', '') or '').lower().strip()
        if di in PXY_SKIP: continue
        minx_i, maxx_i, ni = packet_pxy(pi)
        if ni < 2: continue
        last_x = maxx_i
        for jp in range(ip + 1, len(sorted_idx)):
            j = sorted_idx[jp]
            if j in consumed: continue
            pj = packets[j]
            dj = (pj.get('document_type', '') or '').lower().strip()
            if dj in PXY_SKIP: break
            pi_pgs = pi.get('page_numbers') or []
            pj_pgs = pj.get('page_numbers') or []
            if not pj_pgs: break
            if min(pj_pgs) - max(pi_pgs) > 2: break
            minx_j, maxx_j, nj = packet_pxy(pj)
            if nj != ni: break
            if minx_j == last_x + 1 or minx_j == last_x + 2:
                consumed.add(j)
                findings['pxy_merge'] += 1
                last_x = maxx_j
                if last_x >= ni: break
            else:
                break

    return findings


# ─── Run sweep ──────────────────────────────────────────────────
all_jobs = sorted(set(j.split(os.sep)[-3]
                      for j in glob.glob('results/*/step03/step03_result.json')))
print(f"Sweeping {len(all_jobs)} jobs...\n")

totals = {k: 0 for k in
          ('amd_split', 'bl_unify', 'draft_merge', 'doc_split',
           'suffix_lost', 'pl_cont_as_ci', 'pxy_merge')}
issues_per_job = []
for job in all_jobs:
    f = analyze_job(job)
    if not f:
        continue
    if any(v > 0 for v in f.values()):
        issues_per_job.append((job, f))
        for k, v in f.items():
            totals[k] += v

print("─" * 78)
print("Issues found per job (only jobs with ≥1 issue):")
print("─" * 78)
print(f"{'job':40s} | gz38 | gz39 | gz40 | gz41 | gz42 | gz43 | gz44")
print(f"{'':40s} | amd  | bl   | drft | doc  | sufx | plci | pxy")
print("-" * 78)
for job, f in issues_per_job:
    print(f"{job} | {f['amd_split']:4d} | {f['bl_unify']:4d} | "
          f"{f['draft_merge']:4d} | {f['doc_split']:4d} | "
          f"{f['suffix_lost']:4d} | {f['pl_cont_as_ci']:4d} | "
          f"{f['pxy_merge']:4d}")
print("-" * 78)
print(f"{'TOTAL':40s} | {totals['amd_split']:4d} | {totals['bl_unify']:4d} | "
      f"{totals['draft_merge']:4d} | {totals['doc_split']:4d} | "
      f"{totals['suffix_lost']:4d} | {totals['pl_cont_as_ci']:4d} | "
      f"{totals['pxy_merge']:4d}")
print(f"\nTotal jobs with classification issues: {len(issues_per_job)}/{len(all_jobs)}")
print(f"Total individual fixes that will be applied: {sum(totals.values())}")
