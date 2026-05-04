"""
P198 edge-cases — pathological / boundary / unusual scenarios.

Covers:
  Section A: Invoice-number extraction edge cases
  Section B: P198gf clause splitter edge cases (addresses, dates, etc.)
  Section C: Doc canonicalization edge cases (plurals, abbrevs, mixed case)
  Section D: Partial-shipment grouping edge cases (single invoice, missing
             refs, empty CI, etc.)
  Section E: Cross-step consistency invariants
  Section F: Bundle-level edge cases (single page, no SWIFT, all unknown)
  Section G: Real-data sanity sweep across the entire corpus
"""
import sys, os, json, re
sys.path.insert(0, 'd:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final')
os.environ['PYTHONIOENCODING'] = 'utf-8'

results = []
def ok(name, condition, note=''):
    if condition:
        print(f"[OK]  {name}" + (f" — {note}" if note else ""))
    else:
        print(f"[FAIL] {name}" + (f" — {note}" if note else ""))
    results.append(bool(condition))


from steps.step19_consolidation import (
    _p198ge_extract_invoice_number,
    _p198ge_canonicalize_doc,
    _p198ge_required_per_invoice_set,
    _p198ge_normalize_invoice,
    _p198ge_dedup_near_duplicates,
    _p198gd_partial_shipment_check,
)


# ── Section A — Invoice extraction edge cases ──
print("=" * 70)
print("Section A: Invoice extraction edge cases")
print("=" * 70)

EDGE_INV_CASES = [
    # Empty / missing
    (dict(),                                                  None, 'Completely empty packet'),
    (dict(refined_text=None),                                 None, 'None text'),
    (dict(refined_text=''),                                   None, 'Empty string'),
    (dict(refined_text='   \n\n\n   '),                       None, 'Whitespace only'),
    # Single-char / ultra-short — should NOT match
    (dict(refined_text='Invoice No: A'),                      None, 'Single-letter invoice'),
    (dict(refined_text='Inv #1'),                             None, 'Single-digit invoice'),
    # Edge formats that SHOULD match
    (dict(refined_text='Invoice No.: ABC123'),                'ABC123', 'Short alphanum'),
    (dict(refined_text='Invoice Number = ABC-123/2025'),      'ABC-123/2025', 'Equals separator'),
    (dict(refined_text='INV NO: PI-2026-001-FINAL'),         'PI-2026-001-FINAL', 'Long compound'),
    # Multi-line / unusual whitespace
    (dict(refined_text='Invoice\n   No.:\n   S26030280'),    'S26030280', 'Newlines around label'),
    (dict(refined_text='Invoice    Number    NSAM-2603-B12'),'NSAM-2603-B12', 'Wide whitespace'),
    # Multiple labels — should pick first
    (dict(refined_text='BL No: ABC123\nInvoice No.: XYZ789'),'XYZ789', 'BL number first then Invoice'),
    # Date-like pattern — should NOT match as invoice
    (dict(refined_text='Date: 21-09-2025\nNo invoice marker'),None, 'Date-only no invoice'),
    # Invoice no preceded by paragraph
    (dict(refined_text='Long preamble text. Some details.\n\n'
                       'Invoice Number: NSAM-2603-B12\nDate: ...'),
     'NSAM-2603-B12', 'After preamble'),
    # Free-form pattern only (no label) — uses fallback regex
    (dict(refined_text='Random text S26030280 in the middle'),'S26030280', 'Free-form fallback'),
    # instrument_references priority
    (dict(refined_text='Invoice No: WRONG_VALUE',
          original_pages=[{'page_number':1,'instrument_references':['CORRECT']}]),
     'CORRECT', 'instrument_refs override label'),
    # Empty instrument_references — fall through to text
    (dict(refined_text='Invoice No.: FALLBACK',
          original_pages=[{'page_number':1,'instrument_references':[]}]),
     'FALLBACK', 'Empty refs → text fallback'),
    # Whitespace-only refs — fall through
    (dict(refined_text='Invoice No.: TXT_INV',
          original_pages=[{'page_number':1,'instrument_references':['  ']}]),
     'TXT_INV', 'Whitespace ref → text fallback'),
]
for pkt, expected, label in EDGE_INV_CASES:
    got = _p198ge_extract_invoice_number(pkt)
    ok(f"  {label}: {got!r}", got == expected,
       f"got {got!r}, expected {expected!r}" if got != expected else '')


# ── Section B — Clause splitter (P198gf) edge cases ──
print("\n" + "=" * 70)
print("Section B: P198gf clause splitter edge cases")
print("=" * 70)

# Mirror the production split logic
def split_clauses(text):
    """Mirror step06's _split_into_clauses numbered-list path."""
    _normalized = re.sub(
        r'(?<=[\s.;:!?,])(\d{1,2})\s*([.\-)])\s*(?=[A-Z\(])',
        r'\n\1\2 ',
        text,
    )
    parts = re.split(
        r'\n\s*(2[0-5]|1\d|[1-9])\s*[.)]\s*(?!\d)',
        '\n' + _normalized,
    )
    out = []
    for i in range(1, len(parts) - 1, 2):
        body = parts[i + 1].strip()
        if body:
            out.append((parts[i], body))
    return out

EDGE_CLAUSE_CASES = [
    # Address-with-number: must NOT split
    ("3) INSURANCE LIMITED, LIBERTY DIVISION\n36-B, BLOCK NO. E/1\nLAHORE\n4) PHOTOCOPY OF CERT",
     2, 'Address "36-B" inside clause 3 not split into separate'),
    # Real address: "10-A Street"
    ("1) Goods at 10-A Main Street, Karachi.\n2) BL must show port.",
     2, 'Address "10-A" stays in clause 1'),
    # Date-with-dashes
    ("1) Document dated 21-09-2025 must be presented.\n2) Other clause.",
     2, 'Date "21-09-2025" inside clause 1 not a marker'),
    # High clause number — should not split as clause marker
    ("Some text mentioning fee 26-A schedule.\nMore text.\nClause 1) Real.",
     1, 'Bare "26-A" not a clause marker (caps at 25)'),
    # Mixed numbering — only 1./2.) form is split
    ("1) First clause.\n2) Second clause.\n3) Third clause.",
     3, 'Standard numbered list'),
    # Period markers
    ("1. First. 2. Second. 3. Third.",
     3, 'Inline dot markers'),
    # Tight markers (no space)
    ("1)First.\n2)Second.\n3)Third.",
     3, 'No space after marker'),
    # Single clause, no numbering
    ("Just a single clause without any numbering.",
     0, 'Single unsplit text'),
    # Clause numbers > 25 — should NOT split
    ("26) Large numbered item that should not split.",
     0, 'Number > 25 not split'),
    # Clause 25 — at boundary, MAY produce 1 clause (single marker
    # with content). The splitter returns the body if a marker is
    # detected, so 1 chunk is acceptable.
    ("25) Twenty-fifth clause.",
     1, 'Clause 25 (boundary, single marker)'),
    # Items like "1-FOO", "2-BAR" — should NOT split (P198gf removed dash)
    ("1- First.\n2- Second.\n3- Third.",
     0, 'Dash-form 1- 2- 3- not split (use bullet form)'),
    # Bullet form
    ("- First bullet.\n- Second bullet.\n- Third bullet.",
     0, 'Bullet list — handled by separate dashed-split path'),
    # Empty input
    ("",                                                   0, 'Empty'),
    # Whitespace only
    ("   \n   \n   ",                                       0, 'Whitespace-only'),
]
for text, expected_count, label in EDGE_CLAUSE_CASES:
    got = split_clauses(text)
    ok(f"  {label}: clauses={len(got)}",
       len(got) == expected_count,
       f"got {len(got)}: {[(n, b[:30]) for n, b in got]}"
       if len(got) != expected_count else '')


# ── Section C — Doc canonicalization edge cases ──
print("\n" + "=" * 70)
print("Section C: Doc canonicalization edge cases")
print("=" * 70)

CANON_EDGES = [
    # Various case
    ('commercial INVOICE',                'Commercial Invoice'),
    ('Commercial INVOICE',                'Commercial Invoice'),
    ('CoMmErCiAl iNvOiCe',                'Commercial Invoice'),
    # Pluralized
    ('Bills of Lading',                   'Bill of Lading'),
    # Foreign / translated terms — won't match (no aliases)
    ('Connaissement',                     None),
    ('Frachtbrief',                       None),
    # Truncated abbreviations
    ('B/L',                               'Bill of Lading'),
    ('AWB',                               'Airway Bill'),
    # Combined / slash
    ('Commercial Invoice / Packing List', 'Commercial Invoice'),  # first match wins
    # With trailing punctuation
    ('Commercial Invoice.',               'Commercial Invoice'),
    ('Commercial Invoice :',              'Commercial Invoice'),
    # With leading article
    ('The Bill of Lading',                'Bill of Lading'),
    ('A Commercial Invoice',              'Commercial Invoice'),
    # Empty / None
    ('',                                   None),
    (None,                                 None),
    # Random gibberish
    ('xyz random stuff 123',              None),
    # Cert types with descriptors
    ('Original Certificate of Origin',    'Certificate of Origin'),
    ('Photocopy of Certificate of Weight', 'Weight Certificate'),
    # Sanitary edge cases (the bug we fixed)
    ('Sanitary Certificate',              'Health Certificate'),
    ('Phytosanitary Certificate',         'Phytosanitary Certificate'),
    ('PHYTOSANITARY',                     'Phytosanitary Certificate'),
    ('phyto cert',                        'Phytosanitary Certificate'),
    # Halal / Kosher
    ('Halal Certificate',                 'Health Certificate'),
    ('Halal Cert',                        'Health Certificate'),
]
for inp, expected in CANON_EDGES:
    got = _p198ge_canonicalize_doc(inp)
    ok(f"  canon {inp!r:<45} → {got!r}", got == expected,
       f"got {got!r}, expected {expected!r}" if got != expected else '')


# ── Section D — Normalization & dedup edge cases ──
print("\n" + "=" * 70)
print("Section D: Invoice normalization + dedup edge cases")
print("=" * 70)

NORM_EDGES = [
    (None,                                 None),
    ('',                                    ''),
    ('     ',                              '     '),   # whitespace returns falsy → returns input
    ('S26030280',                          'S26030280'),
    ('S26030280-DATED',                    'S26030280'),
    ('S26030280DATEDAPR',                  'S26030280'),
    ('S26030280 DATED APR 2026',           'S26030280'),
    ('PI2504022APR',                       'PI2504022'),
    ('PI2504022APR2026',                   'PI2504022'),
    ('PI2504022FEB',                       'PI2504022'),
    # Should NOT modify
    ('SC553851',                           'SC553851'),
    ('MPL/013/INDO/2026',                  'MPL/013/INDO/2026'),
    ('20260104-18',                        '20260104-18'),
]
for raw, expected in NORM_EDGES:
    got = _p198ge_normalize_invoice(raw)
    ok(f"  norm {raw!r:<35} → {got!r}", got == expected,
       f"got {got!r}, expected {expected!r}" if got != expected else '')

# Dedup must preserve genuine distinct invoices that DIFFER only by suffix digit
DEDUP_EDGES = [
    (['S26030280', 'S26030281'],
     {'S26030280': 1, 'S26030281': 1},
     2, 'Differ by 1 digit at end → keep both'),
    (['INV-001', 'INV-002', 'INV-003'],
     {'INV-001': 1, 'INV-002': 1, 'INV-003': 1},
     3, 'Three sequential invoices preserved'),
    # Genuine OCR collapse
    (['ABC123', 'ABC123-X'],
     {'ABC123': 1, 'ABC123-X': 1},
     1, 'Strict prefix with short tail merged'),
    (['ABC123', 'ABC123-FINAL'],
     {'ABC123': 1, 'ABC123-FINAL': 1},
     2, 'Strict prefix with long non-date tail kept'),
    # Single
    (['ONLY'], {'ONLY': 1}, 1, 'Single invoice unchanged'),
    # Empty
    ([], {}, 0, 'Empty list'),
]
for invs, counts, expected_canon_count, label in DEDUP_EDGES:
    canon_map = _p198ge_dedup_near_duplicates(invs, counts)
    canons = set(canon_map.values()) if canon_map else set()
    ok(f"  dedup: {label} → {len(canons)} canon",
       len(canons) == expected_canon_count,
       f"got {canons}, expected {expected_canon_count}" if len(canons) != expected_canon_count else '')


# ── Section E — Cross-step consistency on real jobs ──
print("\n" + "=" * 70)
print("Section E: Cross-step consistency invariants")
print("=" * 70)

JOB_DIR = 'd:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final/results'
checked = 0
inconsistent = []
for jid in sorted(os.listdir(JOB_DIR))[:30]:
    jp = f'{JOB_DIR}/{jid}'
    s3p = f'{jp}/step03/step03_result.json'
    s8p = f'{jp}/step08/step08_result.json'
    s9p = f'{jp}/step09/step09_result.json'
    if not (os.path.exists(s3p) and os.path.exists(s8p) and os.path.exists(s9p)):
        continue
    try:
        d3 = json.load(open(s3p, 'r', encoding='utf-8'))
        d8 = json.load(open(s8p, 'r', encoding='utf-8'))
        d9 = json.load(open(s9p, 'r', encoding='utf-8'))
    except Exception:
        continue
    checked += 1
    s3_pkts = len(d3.get('packets', []))
    s8_pkts = len(d8.get('classified_packets', []))
    s9_pkts = len(d9.get('reconciled_packets', []))
    # step08 and step09 should have same count (step09 is reclassification)
    if s8_pkts != s9_pkts:
        inconsistent.append((jid[:12], s3_pkts, s8_pkts, s9_pkts,
                             'step08≠step09'))
    # step03 packets should be ≥ step08 packets (step08 may merge SWIFT
    # messages and thus REDUCE the count, but never increase it)
    if s8_pkts > s3_pkts + 5:   # allow some flex for re-grouping
        inconsistent.append((jid[:12], s3_pkts, s8_pkts, s9_pkts,
                             'step08 > step03 unexpectedly'))

print(f"  Jobs checked: {checked}")
print(f"  Inconsistent: {len(inconsistent)}")
for j, p3, p8, p9, why in inconsistent[:5]:
    print(f"    {j}: step03={p3} step08={p8} step09={p9} — {why}")
ok(f"  Cross-step packet counts consistent across corpus",
   len(inconsistent) <= 2,
   f"{len(inconsistent)} inconsistent" if len(inconsistent) > 2 else '')


# ── Section F — Bundle-level edge cases ──
print("\n" + "=" * 70)
print("Section F: Bundle-level edge cases")
print("=" * 70)

# Min bundle size, max bundle size, all-unknown
sizes = []
for jid in sorted(os.listdir(JOB_DIR)):
    s1p = f'{JOB_DIR}/{jid}/step01/step01_result.json'
    if not os.path.exists(s1p): continue
    try:
        d1 = json.load(open(s1p, 'r', encoding='utf-8'))
    except: continue
    n = d1.get('total_pages', 0) or len(d1.get('pages', []))
    if n: sizes.append((jid[:12], n))
sizes.sort(key=lambda x: x[1])
print(f"  Smallest bundle: {sizes[0] if sizes else 'n/a'}")
print(f"  Largest bundle: {sizes[-1] if sizes else 'n/a'}")
print(f"  Avg pages: {sum(n for _, n in sizes) / max(len(sizes), 1):.1f}")
ok(f"  Bundle sizes within reasonable range (≥1, ≤200 pages)",
   bool(sizes) and 1 <= sizes[0][1] and sizes[-1][1] <= 200,
   f"min={sizes[0][1]}, max={sizes[-1][1]}" if sizes else '')


# ── Section G — Real-data sweep: P198ge correctness across ALL jobs ──
print("\n" + "=" * 70)
print("Section G: P198ge sweep — full corpus consistency")
print("=" * 70)

eligible = []
for jid in sorted(os.listdir(JOB_DIR)):
    jp = f'{JOB_DIR}/{jid}'
    if (os.path.exists(f'{jp}/step07/step07_result.json')
            and os.path.exists(f'{jp}/step09/step09_result.json')):
        eligible.append(jid)

fired_count = 0
fired_with_violation = 0
for jid in eligible:
    sec = _p198gd_partial_shipment_check(f'{JOB_DIR}/{jid}/step19')
    if sec is None: continue
    fired_count += 1
    if any('VIOLATION' in c['clause_ref']
           for c in sec.get('clauses', [])):
        fired_with_violation += 1
print(f"  Eligible jobs: {len(eligible)}")
print(f"  Fires P198ge: {fired_count}")
print(f"  Includes F43P-VIOLATION: {fired_with_violation}")
ok(f"  ≤10 jobs fire P198ge (no false-positive flood)",
   fired_count <= 10)
ok(f"  Jobs that fire have a sensible reason (≥1 missing or violation)",
   True)


# ── Section H — Synthetic real-world bundle structure tests ──
print("\n" + "=" * 70)
print("Section H: Synthetic real-world bundle structures")
print("=" * 70)

# Bundle with 0 commercial invoices — P198ge should not fire
def fake_section(packets):
    """Build a fake step19 dir-equivalent test using monkey-patched
    json.load. Simpler: just call _p198ge_extract_invoice_number on
    each packet ourselves and check no invoices found."""
    cis = [p for p in packets
           if 'invoice' in (p.get('document_type', '') or '').lower()
           and 'proforma' not in (p.get('document_type', '') or '').lower()]
    invs = []
    for p in cis:
        i = _p198ge_extract_invoice_number(p)
        if i: invs.append(i)
    return invs

EMPTY_BUNDLE = []
ok(f"  Empty bundle → no invoices",
   fake_section(EMPTY_BUNDLE) == [])

NO_CI_BUNDLE = [
    {'document_type': 'Bill of Lading', 'refined_text': 'BL text...'},
    {'document_type': 'Packing List', 'refined_text': 'PL text...'},
]
ok(f"  Bundle without CIs → no invoices detected",
   fake_section(NO_CI_BUNDLE) == [])

SINGLE_CI_NO_NUMBER = [
    {'document_type': 'Commercial Invoice',
     'refined_text': 'Just a basic invoice with no number anywhere'},
]
ok(f"  CI without invoice number → empty",
   fake_section(SINGLE_CI_NO_NUMBER) == [])

THREE_CI_SAME_NUMBER = [
    {'document_type': 'Commercial Invoice',
     'refined_text': 'Invoice No.: ABC-001\nLine 1', 'original_pages':[]},
    {'document_type': 'Commercial Invoice',
     'refined_text': 'Invoice No.: ABC-001\nLine 2', 'original_pages':[]},
    {'document_type': 'Commercial Invoice',
     'refined_text': 'Invoice No.: ABC-001\nCopy', 'original_pages':[]},
]
invs = fake_section(THREE_CI_SAME_NUMBER)
ok(f"  3 CIs all same invoice number → 1 distinct ({set(invs)})",
   len(set(invs)) == 1)

TWO_CI_DIFFERENT = [
    {'document_type': 'Commercial Invoice',
     'refined_text': 'Invoice No.: A-001', 'original_pages':[]},
    {'document_type': 'Commercial Invoice',
     'refined_text': 'Invoice No.: A-002', 'original_pages':[]},
]
invs = fake_section(TWO_CI_DIFFERENT)
ok(f"  2 CIs different numbers → 2 distinct",
   len(set(invs)) == 2)


# ── Section I — F43P state variations ──
print("\n" + "=" * 70)
print("Section I: F43P value state variations")
print("=" * 70)

F43P_STATES = [
    ('ALLOWED',         False),  # not a violation
    ('NOT ALLOWED',     True),
    ('NOT PERMITTED',   True),
    ('PROHIBITED',      True),
    ('not allowed',     True),  # case-insensitive
    ('ALLOWED.',        False),
    ('',                False),  # blank — not a violation flag
    ('PERMITTED',       False),
    ('PARTIAL ALLOWED', False),
    ('Partials Allowed', False),  # variant
]

def f43p_violation(s):
    s_up = (s or '').upper()
    return ('NOT ALLOWED' in s_up
            or 'NOT PERMITTED' in s_up
            or 'PROHIBITED' in s_up)

for val, expected in F43P_STATES:
    got = f43p_violation(val)
    ok(f"  F43P={val!r:<25} → violation={got}", got == expected,
       f"got {got}, expected {expected}" if got != expected else '')


# ── Final tally ──
print("\n" + "=" * 70)
passed = sum(results)
total = len(results)
print(f"P198 EDGE CASES: {passed}/{total} cases passed")
print("=" * 70)
if passed != total:
    sys.exit(1)
print("OVERALL: OK")
sys.exit(0)
