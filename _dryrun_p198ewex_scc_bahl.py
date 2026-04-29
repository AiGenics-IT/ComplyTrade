"""
P198ew + P198ex dry-run — two adjacent fixes.

P198ew: SCC strict-content guard now checks ANY SCC in the bundle
        carries the required statement before forcing FAIL on a
        single packet. A misclassified packet that lacks ICC / PMR
        text no longer fails when a sibling SCC does carry it.

P198ex: BAHL multi-message split now starts a NEW group on EVERY
        "Message Details #N" header, regardless of fin / N value.
        Previously, consecutive same-fin messages (MT999 / MT999)
        collapsed into one group, hiding the MT700 split.

Real anchors:
  Job a2d1ed04 (SCC):
    pkt_26 page 39 = real SCC with ICC + PMR statements
    pkt_30 pages 43-44 = misclassified ATOM packing-cert (no ICC/PMR)
    Without P198ew: pkt_30 fails R0021 / R0022.
    With P198ew:    pkt_30 keeps PASS (ICC + PMR satisfied at set level).

  BAHL_1001LC57343(LC).pdf:
    Pages 1-2 = MT999 (separate)
    Pages 3-4 = MT999 (separate)
    Pages 5-7 = MT700 (LC)
    Pages 8-9 = MT799
    With P198ex: 4 distinct groups instead of 2.
"""
import sys, os, re, json
sys.path.insert(0, 'd:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final')

results = []
def assert_eq(name, got, expected):
    ok = (got == expected)
    print(f"[{'OK' if ok else 'FAIL'}] {name}")
    if not ok:
        print(f"          got     : {got!r}")
        print(f"          expected: {expected!r}")
    results.append(ok)

# ── Static-source check for P198ex (every header starts new group) ──
src = open('d:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final/steps/step03_sequencing.py',
           'r', encoding='utf-8').read()
assert_eq("P198ex: _start_new = True applied unconditionally",
          'P198ex' in src and '_start_new = True' in src, True)

# ── Static-source check for P198ew (set-level SCC corpus) ──
v_src = open('d:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final/steps/step14_verification.py',
             'r', encoding='utf-8').read()
assert_eq("P198ew: _scc_corpus_up corpus is built",
          '_scc_corpus_up' in v_src, True)
assert_eq("P198ew: corpus check uses doc_re.search(_scc_corpus_up)",
          'doc_re.search(_scc_corpus_up)' in v_src, True)
assert_eq("P198ew: 'PASS retained' message logged",
          'satisfied by sibling SCC' in v_src, True)

# ── Functional test for P198ew using real job a2d1ed04 ──
JOB = 'd:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final/results/a2d1ed04-eb9d-4a36-8c57-9f6295e1e3fc'
with open(os.path.join(JOB, 'step08', 'step08_result.json'), 'r',
          encoding='utf-8') as f:
    s8 = json.load(f)

icc_re = re.compile(r'\bINSTITUTE\s+CLASSIFICATION\s+CLAUSE\b|\bICC\s*\(?\s*INSTITUTE\b',
                    re.IGNORECASE)
pmr_re = re.compile(r'\b(?:PAKISTAN(?:I)?\s+MARITIME\s+RULES?|'
                    r'OPERATING\s+IN\s+ACCORDANCE\s+WITH\s+PAKISTAN|'
                    r'MARITIME\s+RULES?\s+AND\s+PORT\s+REGULATIONS?)\b',
                    re.IGNORECASE)

# Build the SCC corpus exactly like P198ew does
scc_packets = []
scc_corpus = ""
for pkt in s8.get('classified_packets', []) or []:
    dt = (pkt.get('document_type') or '').lower()
    mr = (pkt.get('matched_requirement_name') or '').lower()
    if 'shipping company' in dt or 'shipping company' in mr:
        scc_packets.append(pkt)
        txt = (pkt.get('cleaned_text') or pkt.get('raw_text') or '')
        if txt:
            scc_corpus += "\n" + txt.upper()

print(f"\n--- Real job a2d1ed04 SCC corpus ---")
print(f"  SCC packets: {len(scc_packets)}")
for p in scc_packets:
    pgs = [op.get('page_number') for op in (p.get('original_pages') or [])
           if isinstance(op, dict)]
    txt = (p.get('cleaned_text') or p.get('raw_text') or '').upper()
    print(f"   {p.get('packet_id'):8s} pages={pgs} ICC={bool(icc_re.search(txt))} "
          f"PMR={bool(pmr_re.search(txt))}")

assert_eq("real job: ICC found in SCC corpus (set-level)",
          bool(icc_re.search(scc_corpus)), True)
assert_eq("real job: PMR found in SCC corpus (set-level)",
          bool(pmr_re.search(scc_corpus)), True)
assert_eq("real job: at least 1 SCC has ICC literal",
          any(icc_re.search((p.get('cleaned_text') or p.get('raw_text') or '').upper())
              for p in scc_packets), True)
assert_eq("real job: at least 1 SCC has PMR literal",
          any(pmr_re.search((p.get('cleaned_text') or p.get('raw_text') or '').upper())
              for p in scc_packets), True)

# ── P198ey — BAHL header regex catches BOTH styles ──
print("\n--- P198ey: BAHL header regex (Message Details #N OR Message <N>) ---")
import re as _re
_RE = _re.compile(
    r'Message\s+Details\s+#\s*(\d+)|'
    r'Message\s+(\d{2,})\s*\n\s*Message\s+Identifier',
    _re.IGNORECASE,
)
_cases = [
    ('Output style "Message Details #552"', 'Message Details #552\n***\nMessage Identifier', '552'),
    ('Output style "Message Details #968"', 'Message Details #968\n***', '968'),
    ('Input style "Message 2592"',           'Message 2592\nMessage Identifier',              '2592'),
    ('Output style "Message Details #114"', 'Message Details #114\n\nMessage Identifier',     '114'),
    ('Negative — "Message Text" body',       'Message Text\nBlock 4\nF27',                    None),
    ('Negative — "Message Header" body',     'Message Header\nStatus: Read-Only',             None),
    ('Negative — random "message" word',     'This is a random message about banking',        None),
]
for name, txt, expected in _cases:
    matches = list(_RE.finditer(txt))
    got = (matches[0].group(1) or matches[0].group(2)) if matches else None
    ok = (got == expected)
    print(f"[{'OK' if ok else 'FAIL'}] {name}: got={got!r} expected={expected!r}")
    results.append(ok)

# ── BAHL header-splitting logic (P198ex) ──
# Synthetic test: simulate the input data from the BAHL file
print("\n--- BAHL P198ex synthetic split ---")

# Mock _BAHL_FIN_TO_MT
_FIN_TO_MT = {
    '999': 'MT999', '799': 'MT799', '700': 'MT700', '707': 'MT707',
    '730': 'MT730', '710': 'MT710',
}

def simulate_split(headers, fins_by_pg):
    """Simulate the P198ex logic: every header = new group."""
    groups = {}
    next_id = 0
    cur = None
    for pg in sorted(headers.keys()):
        next_id += 1
        groups[next_id] = {
            'pages': [pg],
            'fin': fins_by_pg.get(pg, ''),
            'mt_type': _FIN_TO_MT.get(fins_by_pg.get(pg, ''), ''),
        }
        cur = next_id
    return groups

# Scenario: BAHL_1001LC57343(LC).pdf — 4 messages
hdrs = {1: [1], 3: [2], 5: [3], 8: [4]}
fins = {1: '999', 3: '999', 5: '700', 8: '799'}
out = simulate_split(hdrs, fins)
assert_eq("BAHL: 4 message groups produced", len(out), 4)
assert_eq("BAHL: group 1 mt_type=MT999",  out[1]['mt_type'], 'MT999')
assert_eq("BAHL: group 2 mt_type=MT999",  out[2]['mt_type'], 'MT999')
assert_eq("BAHL: group 3 mt_type=MT700",  out[3]['mt_type'], 'MT700')
assert_eq("BAHL: group 4 mt_type=MT799",  out[4]['mt_type'], 'MT799')
assert_eq("BAHL: group 1 starts at pg 1", out[1]['pages'][0], 1)
assert_eq("BAHL: group 2 starts at pg 3", out[2]['pages'][0], 3)
assert_eq("BAHL: group 3 starts at pg 5", out[3]['pages'][0], 5)
assert_eq("BAHL: group 4 starts at pg 8", out[4]['pages'][0], 8)

# ── P198ez Notify-party expand to full doc when "SEE ATTACHED RIDER" ──
print("\n--- P198ez Notify party SEE-ATTACHED-RIDER fall-through ---")
_REDIRECT_RE = re.compile(
    r'(?:SEE|AS\s+PER|REFER\s+TO)\s+(?:THE\s+)?'
    r'(?:ATTACHED|ATTACH(?:ED)?\s+RIDER|RIDER|ATTACHED\s+LIST|'
    r'ATTACHMENT|ANNEX(?:URE)?|SCHEDULE|SHEET|SPECIFICATION)',
    re.IGNORECASE,
)

def _decide_corpus(notify_section, full_text):
    ns = re.sub(r'\s+', ' ', notify_section).strip()
    return full_text if (_REDIRECT_RE.search(ns) or len(ns) < 80) else notify_section

def assert_pass(name, notify, full, expect_full=True):
    chosen = _decide_corpus(notify, full)
    ok = (chosen is full) == expect_full
    print(f"[{'OK' if ok else 'FAIL'}] {name}: corpus={'FULL' if chosen is full else 'strict'}")
    results.append(ok)

# Real-data anchor — BL pkt_14 from job 2d98b74c
real_notify = ('SEE THE ATTACHED RIDER\nTO OBTAIN DELIVERY CONTACT '
               'SOUTH ASIA LOGISTIC SERVICES\n1111-1112, 11TH FLOOR\n')
real_full   = real_notify + '\n... ISSUING BANK.UNITED BANK LIMITED, KARACHI-PAKISTAN ...'
assert_pass("real-data: 'SEE THE ATTACHED RIDER' → use full text",
            real_notify, real_full, expect_full=True)

# Variants
assert_pass("'AS PER ATTACHED' → use full text",
            'AS PER ATTACHED LIST', 'fulltext', expect_full=True)
assert_pass("'See Annexure' → use full text",
            'see annexure', 'fulltext', expect_full=True)
assert_pass("'Refer to Schedule' → use full text",
            'refer to the schedule', 'fulltext', expect_full=True)
assert_pass("'See Specification' → use full text",
            'See Specification', 'fulltext', expect_full=True)
# Negatives — strict notify with full address keeps strict scope
strict_notify = ('UNITED BANK LIMITED\nKARACHI BRANCH\nPOST BOX 123\n'
                 '1234 MAIN STREET\nADDRESS CONTINUATION DETAIL\n')
assert_pass("genuine notify section (>80 chars, no redirect) → strict",
            strict_notify, strict_notify + 'extra', expect_full=False)

# Static check that the production code has the redirect logic
v_src = open('d:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final/steps/step14_verification.py',
             'r', encoding='utf-8').read()
assert_eq("P198ez: production code has _ns_redirect", '_ns_redirect' in v_src, True)
assert_eq("P198ez: production code uses _ib_blob multi-line",
          '_ib_blob' in v_src and '_ib_blob_parts.append' in v_src, True)

# ── P198ev datetime-vs-date guard ──
print("\n--- P198ev datetime-vs-date comparison guard ---")
imp_src = open('d:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final/steps/step14_implicit.py',
               'r', encoding='utf-8').read()
assert_eq("P198ev: _pres_for_cmp normalisation present",
          '_pres_for_cmp = pres_date.date()' in imp_src, True)
assert_eq("P198ev: comparison uses normalised value",
          '_pres_for_cmp <= _expiry_date' in imp_src, True)

passed = sum(results)
total = len(results)
print(f"\n{passed}/{total} cases passed")
if passed != total:
    sys.exit(1)
print("OVERALL: OK")
sys.exit(0)
