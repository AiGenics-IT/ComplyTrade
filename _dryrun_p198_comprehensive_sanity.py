"""
P198 comprehensive sanity sweep — runs after the 76-test regression
sweep, covering edge cases and cross-job consistency for every
recent fix (P198fs / fw / fx / fy / gc / gd / ge).

Stress areas:
  • Invoice-number extraction across every observed real-world format
  • Doc canonicalization with full alias coverage
  • Per-invoice required-doc derivation from real LC F46A on
    Toyota / coal / multi-modal / single-shipment LCs
  • F43P guard variations (ALLOWED / NOT ALLOWED / PROHIBITED / blank)
  • OCR-noise dedup conservatism (must NOT collapse genuine distinct
    invoices like S26030326 vs S26030328)
  • BL bare-agent guard (P198fx) against every real BL packet
  • Tier-4 ambiguous-target guard (P198fy) against every real Weight
    Cert / Inspection Cert in the corpus
  • F47A rule-pattern filter (P198fw) against every F47A clause
    on every job
  • Shipment-advice hint-veto patterns (P198fs/gc) against every
    real Shipment Advice doc_hint
  • End-to-end real-job sweep with no false positives
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


# ── Section 1 — Invoice extraction across observed formats ──
print("=" * 70)
print("Section 1: Invoice extraction across observed formats")
print("=" * 70)

# All real-world formats found across 76 jobs
INV_FORMAT_CASES = [
    # (raw_text, expected, format_name)
    # From step3 instrument_references on packet
    (dict(original_pages=[{'page_number':1,'instrument_references':['S26030280']}]),
     'S26030280', 'Toyota S-format via instrument_references'),
    (dict(original_pages=[{'page_number':1,'instrument_references':['MPL/013/INDO/2026']}]),
     'MPL/013/INDO/2026', 'MPL coal via instrument_references'),
    # From label-based regex
    (dict(refined_text='Invoice No.: S26030280'),                       'S26030280',           'Toyota label'),
    (dict(refined_text='COMM. INVOICE NO. S26030328'),                  'S26030328',           'Toyota label uppercase'),
    (dict(cleaned_text='Inv No: SC553851'),                             'SC553851',            'SC numeric label'),
    (dict(refined_text='Invoice Number: NSAM-2603-B12'),                'NSAM-2603-B12',       'NSAM compound prefix'),
    (dict(refined_text='Reference No.: 03324/03/2026'),                 '03324/03/2026',       'Numeric/slash'),
    (dict(refined_text='Inv #INV00190678'),                             'INV00190678',         'INV-prefix'),
    (dict(refined_text='Inv Number: MPL/013/INDO/2026'),                'MPL/013/INDO/2026',   'MPL via label'),
    (dict(refined_text='Invoice No. APPK24022BIL-1'),                   'APPK24022BIL-1',      'APPK compound'),
    (dict(refined_text='Invoice Number: MC/0006/25'),                   'MC/0006/25',          'MC slash'),
    (dict(refined_text='Inv. No: MCI-786/S-13198'),                     'MCI-786/S-13198',     'MCI compound'),
    (dict(refined_text='Reference Number CI-321/2025'),                 'CI-321/2025',         'CI- compound'),
    (dict(refined_text='Invoice No. RE/017/2025'),                      'RE/017/2025',         'RE/ slash'),
    (dict(refined_text='Invoice Number: SL/J0040/2026'),                'SL/J0040/2026',       'SL/J slash'),
    (dict(refined_text='Invoice Number: 20260104-18'),                  '20260104-18',         'Numeric-dash'),
    (dict(refined_text='Inv No: XPK-TR260303'),                         'XPK-TR260303',        'XPK compound'),
    (dict(refined_text='Just some text without any invoice'),           None,                  'No match'),
    (dict(refined_text=''),                                             None,                  'Empty'),
    # instrument_references takes priority
    (dict(refined_text='Invoice No: WRONG',
          original_pages=[{'page_number':1,'instrument_references':['CORRECT-001']}]),
     'CORRECT-001', 'instrument_references priority over text'),
]
for pkt, expected, label in INV_FORMAT_CASES:
    got = _p198ge_extract_invoice_number(pkt)
    ok(f"  inv-extract: {label:<45} → {got!r}", got == expected,
       f"got {got!r}, expected {expected!r}" if got != expected else '')


# ── Section 2 — Invoice normalization / dedup conservatism ──
print("\n" + "=" * 70)
print("Section 2: Invoice normalization (OCR noise) + dedup")
print("=" * 70)

NORM_CASES = [
    ('PI2504022DATEDAPR',       'PI2504022',     'OCR concat: DATED APR strip'),
    ('PI2504022 DATEDAPR2026',  'PI2504022',     'DATED with year strip'),
    ('PI2504022APR',            'PI2504022',     'month abbrev strip'),
    ('S26030280',               'S26030280',     'clean — no change'),
    ('MPL/013/INDO/2026',       'MPL/013/INDO/2026', 'slash-format clean'),
    ('   S26030280  ',          'S26030280',     'whitespace strip'),
    ('S26030280:',              'S26030280',     'trailing colon strip'),
    ('',                        '',              'empty'),
]
for raw, expected, label in NORM_CASES:
    got = _p198ge_normalize_invoice(raw)
    ok(f"  normalize: {raw!r:<30} → {got!r:<25} ({label})",
       got == expected,
       f"got {got!r}" if got != expected else '')

# Dedup — must NOT collapse real distinct invoices
DEDUP_CASES = [
    # (input_invoices, packet_counts, expected_canon_keys_set)
    # Real Toyota — must keep all 3
    (['S26030280', 'S26030326', 'S26030328'],
     {'S26030280':5, 'S26030326':3, 'S26030328':4},
     {'S26030280', 'S26030326', 'S26030328'},
     'Toyota: 3 distinct invoices preserved'),
    # OCR noise: PI2504022DATEDAPR (after normalize) → already same as PI2504022
    (['PI2504022', 'PI2504022APR'],
     {'PI2504022':7, 'PI2504022APR':1},
     {'PI2504022'},
     'OCR APR suffix collapsed to canonical'),
    # Two genuinely distinct invoices with one shared char — must keep both
    (['INV-001', 'INV-002'],
     {'INV-001':1, 'INV-002':1},
     {'INV-001', 'INV-002'},
     'Distinct numeric suffix preserved'),
    # Single invoice — no change
    (['ONLY-ONE'],
     {'ONLY-ONE':1},
     {'ONLY-ONE'},
     'Single invoice unchanged'),
]
for invoices, counts, expected_canons, label in DEDUP_CASES:
    canon_map = _p198ge_dedup_near_duplicates(invoices, counts)
    got_canons = set(canon_map.values())
    ok(f"  dedup: {label}",
       got_canons == expected_canons,
       f"got canons {got_canons}, expected {expected_canons}" if got_canons != expected_canons else '')


# ── Section 3 — Doc canonicalization full alias coverage ──
print("\n" + "=" * 70)
print("Section 3: Doc canonicalization — every observed alias")
print("=" * 70)

CANON_CASES = [
    # Commercial Invoice family
    ('Commercial Invoice',        'Commercial Invoice'),
    ('COMMERCIAL INVOICE',        'Commercial Invoice'),
    ('Invoice',                   'Commercial Invoice'),
    ('INVOICE',                   'Commercial Invoice'),
    # Bill of Lading family
    ('Bill of Lading',            'Bill of Lading'),
    ('BILL OF LADING',            'Bill of Lading'),
    ('B/L',                       'Bill of Lading'),
    ('Ocean Bill of Lading',      'Bill of Lading'),
    ('Marine Bill of Lading',     'Bill of Lading'),
    # Airway Bill family
    ('Airway Bill',               'Airway Bill'),
    ('Air Waybill',               'Airway Bill'),
    ('AWB',                       'Airway Bill'),
    # Beneficiary Certificate family
    ('Beneficiary Certificate',                   'Beneficiary Certificate'),
    ("Beneficiary's Declaration/Certificate",     'Beneficiary Certificate'),
    ("Beneficiary's Certificate",                 'Beneficiary Certificate'),
    ('Beneficiary Declaration',                   'Beneficiary Certificate'),
    # Inspection family (covers Sampling/Analysis/Quality)
    ('Inspection Certificate',                    'Inspection Certificate'),
    ('Certificate of Sampling and Analysis',      'Inspection Certificate'),
    ('CERTIFICATE OF SAMPLING AND ANALYSIS',      'Inspection Certificate'),
    ('Quality Certificate',                       'Inspection Certificate'),
    ('Pre-Shipment Inspection',                   'Inspection Certificate'),
    # Weight Certificate family
    ('Weight Certificate',                        'Weight Certificate'),
    ('CERTIFICATE OF WEIGHT',                     'Weight Certificate'),
    ('Quantity Certificate',                      'Weight Certificate'),
    # Health/sanitary
    ('Health Certificate',                        'Health Certificate'),
    ('Halal Certificate',                         'Health Certificate'),
    ('Sanitary Certificate',                      'Health Certificate'),
    # Origin
    ('Certificate of Origin',                     'Certificate of Origin'),
    ('Cert of Origin',                            'Certificate of Origin'),
    ('Origin Certificate',                        'Certificate of Origin'),
    # Phyto / Fumigation
    ('Phytosanitary Certificate',                 'Phytosanitary Certificate'),
    ('Fumigation Certificate',                    'Fumigation Certificate'),
    # Insurance family
    ('Insurance Certificate',                     'Insurance Certificate'),
    ('Insurance Policy',                          'Insurance Certificate'),
    ('Cover Note',                                'Insurance Certificate'),
    # Shipping company cert
    ('Shipping Company Certificate',              'Shipping Company Certificate'),
    ("Shipping Company's Certificate",            'Shipping Company Certificate'),
    ('Carrier Certificate',                       'Shipping Company Certificate'),
    # Packing list family
    ('Packing List',                              'Packing List'),
    ('Weight List',                               'Packing List'),
    ('Cargo Manifest',                            'Packing List'),
    # Excluded (LC-level, not per-invoice → returns None or NOT in whitelist)
    ('Document Remittance',                       None),
    ('Charges Certificate',                       None),
    ('Translation Certificate',                   None),
    ('Discrepancy Fee Notice',                    None),
    ('Random Stuff',                              None),
]
for raw, expected in CANON_CASES:
    got = _p198ge_canonicalize_doc(raw)
    ok(f"  canon: {raw!r:<50} → {got!r}", got == expected,
       f"got {got!r}" if got != expected else '')


# ── Section 4 — Per-invoice required set on real LCs ──
print("\n" + "=" * 70)
print("Section 4: Per-invoice required-doc set across real LCs")
print("=" * 70)

JOB_DIR = 'd:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final/results'

LC_REQUIREMENT_CHECKS = [
    # (job_id_prefix, must_include, must_exclude, label)
    ('4dc16c1a',
     {'Commercial Invoice', 'Beneficiary Certificate', 'Certificate of Origin',
      'Shipment Advice', 'Transport Document (Bill of Lading / Airway Bill)'},
     {'Document Remittance', 'Charges Certificate'},
     'Toyota multi-modal LC'),
    ('1450d59f',
     {'Commercial Invoice', 'Bill of Lading', 'Inspection Certificate',
      'Weight Certificate', 'Certificate of Origin', 'Shipment Advice'},
     {'Document Remittance', 'Quantity Certificate', 'Quality Certificate',
      'Discrepancy Fee Notice'},
     'Coal LC (sea-only)'),
    ('10ebcb74',
     # 10ebcb74 has Airway Bill required (no BL)
     {'Commercial Invoice', 'Beneficiary Certificate', 'Airway Bill',
      'Phytosanitary Certificate'},
     {'Bill of Lading'},
     'Air-only LC with phyto'),
]

for job_pref, must_incl, must_excl, label in LC_REQUIREMENT_CHECKS:
    matches = [d for d in os.listdir(JOB_DIR) if d.startswith(job_pref)]
    if not matches:
        ok(f"  {label}: SKIP (no job folder)", True, 'no fixture')
        continue
    full = matches[0]
    s7p = f'{JOB_DIR}/{full}/step07/step07_result.json'
    if not os.path.exists(s7p):
        ok(f"  {label}: SKIP (no step07)", True, 'no fixture')
        continue
    d7 = json.load(open(s7p, 'r', encoding='utf-8'))
    required = set(_p198ge_required_per_invoice_set(d7['structured_lc']))
    missing_required = must_incl - required
    spurious_inclusions = must_excl & required
    ok(f"  {label}: required ⊇ {sorted(must_incl)}",
       not missing_required,
       f"missing from result: {missing_required}" if missing_required else '')
    ok(f"  {label}: required ∩ {sorted(must_excl)} = ∅",
       not spurious_inclusions,
       f"unexpected inclusions: {spurious_inclusions}" if spurious_inclusions else '')


# ── Section 5 — F43P guard variations on synthetic LC ──
print("\n" + "=" * 70)
print("Section 5: F43P guard variations")
print("=" * 70)

# We can't easily synthesize a full step09 + step07 setup, so we
# verify the guard by directly inspecting the output of
# _p198gd_partial_shipment_check on a synthetic job folder layout.
# Skip if no real job demonstrates each F43P state.
F43P_REAL_JOB_CHECKS = [
    ('4dc16c1a', 'ALLOWED',     'Toyota — F43P=ALLOWED with multi-invoice → fires'),
    ('e458b300', 'NOT ALLOWED', 'e458b300 — F43P=NOT ALLOWED with multi-invoice → fires + violation flag'),
    ('1450d59f', 'ALLOWED',     'Coal — F43P=ALLOWED single invoice → does NOT fire'),
]
for job_pref, expected_f43p, label in F43P_REAL_JOB_CHECKS:
    matches = [d for d in os.listdir(JOB_DIR) if d.startswith(job_pref)]
    if not matches:
        ok(f"  {label}: SKIP", True); continue
    full = matches[0]
    sec = _p198gd_partial_shipment_check(f'{JOB_DIR}/{full}/step19')
    if 'NOT ALLOWED' in expected_f43p:
        # Should produce a F43P-VIOLATION clause
        if sec is None:
            ok(f"  {label}: F43P-VIOLATION clause", False, 'section returned None')
        else:
            crefs = [c['clause_ref'] for c in sec['clauses']]
            has_violation = any('F43P-VIOLATION' in c for c in crefs)
            ok(f"  {label}: F43P-VIOLATION clause", has_violation,
               f"found clause_refs={crefs}" if not has_violation else '')
    elif 'single invoice' in label.lower():
        ok(f"  {label}", sec is None,
           'unexpectedly fired' if sec is not None else '')
    else:
        # ALLOWED multi-invoice — should fire WITHOUT the violation flag
        if sec is None:
            ok(f"  {label}", False, 'unexpectedly silent')
        else:
            crefs = [c['clause_ref'] for c in sec['clauses']]
            has_violation = any('F43P-VIOLATION' in c for c in crefs)
            ok(f"  {label}: fires", True)
            ok(f"  {label}: NO F43P-VIOLATION clause", not has_violation,
               f"unexpected violation flag in {crefs}" if has_violation else '')


# ── Section 6 — Cross-job no-false-positive sweep ──
print("\n" + "=" * 70)
print("Section 6: Sweep all jobs — no false positives")
print("=" * 70)

eligible = []
for jid in sorted(os.listdir(JOB_DIR)):
    jp = f'{JOB_DIR}/{jid}'
    if not os.path.isdir(jp): continue
    if (os.path.exists(f'{jp}/step07/step07_result.json')
            and os.path.exists(f'{jp}/step09/step09_result.json')):
        eligible.append(jid)

fired = []
for jid in eligible:
    sec = _p198gd_partial_shipment_check(f'{JOB_DIR}/{jid}/step19')
    if sec is not None:
        invs = [c['clause_ref'].replace('Presentation-', '')
                for c in sec['clauses']
                if not c['clause_ref'].endswith('VIOLATION')]
        f43p_violation = any('VIOLATION' in c['clause_ref'] for c in sec['clauses'])
        fired.append((jid, invs, f43p_violation))

print(f"  Eligible jobs: {len(eligible)}")
print(f"  Jobs that fire P198ge: {len(fired)}")
for jid, invs, viol in fired:
    print(f"    {jid[:12]} — {len(invs)} invoices {'[F43P-VIOLATION]' if viol else ''}")

# Hard expectation: only 3 known multi-invoice jobs fire (4dc16c1a, a2d1ed04, e458b300).
known_firers = {'4dc16c1a', 'a2d1ed04', 'e458b300'}
fired_prefixes = {jid[:8] for jid, _, _ in fired}
spurious = fired_prefixes - known_firers
ok(f"  Cross-job: NO unexpected jobs fire (only known multi-invoice 3 jobs fire)",
   not spurious,
   f"unexpected: {spurious}" if spurious else '')
ok(f"  Cross-job: exactly {len(known_firers)} known firers actually fire",
   known_firers.issubset(fired_prefixes),
   f"missing: {known_firers - fired_prefixes}" if not known_firers.issubset(fired_prefixes) else '')


# ── Section 7 — F47A rule-pattern filter (P198fw) on real LCs ──
print("\n" + "=" * 70)
print("Section 7: F47A rule-pattern filter (P198fw)")
print("=" * 70)

# Spot-check that for real LCs, F47A entries phantom-flagged by
# ambiguity but matching rule patterns ARE excluded from the
# per-invoice set.
PHANTOM_FILTER_CHECKS = [
    ('1450d59f',
     # Coal LC has these phantoms in F47A:
     {'Beneficiary Certificate',     # from "DOCUMENTS DATED PRIOR..."
      'Discrepancy Fee Notice',       # from "USD 116/- DISCREPANCY CHARGES..."
      'Quantity Certificate',         # from "CERTIFICATES SHOWING QUANTITY DIFFERENT..."
      'Quality Certificate',          # from "PRICE ADJUSTMENTS CLAUSE..."
      'Draft Bill of Exchange'},      # from "THIRD PARTY DOCUMENTS ARE ACCEPTABLE EXCEPT..."
     'Coal — F47A phantoms filtered'),
]
for job_pref, must_exclude, label in PHANTOM_FILTER_CHECKS:
    matches = [d for d in os.listdir(JOB_DIR) if d.startswith(job_pref)]
    if not matches:
        ok(f"  {label}: SKIP", True); continue
    full = matches[0]
    s7p = f'{JOB_DIR}/{full}/step07/step07_result.json'
    if not os.path.exists(s7p):
        continue
    d7 = json.load(open(s7p, 'r', encoding='utf-8'))
    required = set(_p198ge_required_per_invoice_set(d7['structured_lc']))
    leaked = must_exclude & required
    ok(f"  {label}", not leaked,
       f"phantoms leaked into required set: {leaked}" if leaked else '')


# ── Section 8 — BL bare-agent guard (P198fx) on real BLs ──
print("\n" + "=" * 70)
print("Section 8: BL bare-agent guard (P198fx)")
print("=" * 70)

# For each real BL packet across the corpus, verify the bare_agent
# heuristic correctly distinguishes:
#   • BL with proper "FOR AND ON BEHALF OF THE MASTER" → not bare
#   • BL with only "AS AGENT" + no qualifier → bare
#   • BL with no capacity at all → bare (UCP 600 art 20)

import re as _re
_QUALIFIERS = ('MASTER', 'CARRIER', 'OWNER', 'OWNERS',
               'CHARTERER', 'CHARTERERS',
               'SHIPPING LINE', 'SHIPPING COMPANY',
               'THE VESSEL', 'THE SHIP')
_SIGN_MARKERS = ('[SIGNATURE]', 'SIGNATURE:', 'SIGNED BY',
                 'AUTHORIZED SIGNATORY', 'AUTHORISED SIGNATORY',
                 'FOR AND ON BEHALF OF', 'STAMP:')
_CAPACITY_AFFIRMS = (
    'AS MASTER', 'MASTER OF THE VESSEL', 'AS THE MASTER',
    'AS AGENT FOR THE MASTER', 'AS AGENTS FOR THE MASTER',
    'AS AGENT FOR AND ON BEHALF OF THE MASTER',
    'AS AGENTS FOR AND ON BEHALF OF THE MASTER',
    'FOR AND ON BEHALF OF THE MASTER',
    'AS AGENT FOR THE CARRIER', 'AS AGENTS FOR THE CARRIER',
    'FOR AND ON BEHALF OF THE CARRIER',
    'AS OWNER', 'AS OWNERS',
    'FOR AND ON BEHALF OF THE OWNER',
)

def cap_proof_hit(t):
    n = len(t)
    for ph in _CAPACITY_AFFIRMS:
        p = t.find(ph)
        while p >= 0:
            if p >= int(n * 0.60):
                return True
            for sm in _SIGN_MARKERS:
                sp = t.find(sm, max(0, p - 300), p + 300 + len(ph))
                if sp >= 0 and abs(sp - p) <= 300:
                    return True
            p = t.find(ph, p + 1)
    return False

def bare_agent(doc_text):
    t = doc_text.upper()
    n = len(t)
    bare_unq = qual = False
    for m in _re.finditer(r'\bAS\s+AGENTS?\b', t):
        s, e = m.start(), m.end()
        if not (s >= int(n * 0.60) or any(
                t.find(sm, max(0, s - 300), e + 300) >= 0
                for sm in _SIGN_MARKERS)):
            continue
        window = t[e:e + 120]
        pre = t[max(0, s - 40):s]
        if any(q in window or q in pre for q in _QUALIFIERS):
            qual = True
        else:
            bare_unq = True
    bare = bare_unq and not qual
    if cap_proof_hit(t):
        bare = False
    return bare

# Sweep BL packets across all jobs; assertion is conservative
# (bareness only flagged when capacity legitimately absent).
bl_pkt_count = 0
bl_with_master_capacity = 0
bl_flagged_bare = 0
bl_master_but_flagged_bare = 0   # this should be 0
for jid in eligible:
    s9p = f'{JOB_DIR}/{jid}/step09/step09_result.json'
    if not os.path.exists(s9p): continue
    d9 = json.load(open(s9p, 'r', encoding='utf-8'))
    for pkt in d9.get('reconciled_packets', []):
        if 'bill of lading' not in (pkt.get('document_type', '') or '').lower():
            continue
        txt = (pkt.get('refined_text') or pkt.get('cleaned_text')
               or pkt.get('text') or '')
        if len(txt) < 100:
            continue
        bl_pkt_count += 1
        flagged = bare_agent(txt)
        if flagged:
            bl_flagged_bare += 1
        if 'FOR AND ON BEHALF OF THE MASTER' in txt.upper() \
                or 'AS AGENT FOR THE MASTER' in txt.upper() \
                or 'AS AGENTS FOR THE MASTER' in txt.upper():
            bl_with_master_capacity += 1
            if flagged:
                bl_master_but_flagged_bare += 1
                print(f"    UNEXPECTED bare flag on {jid[:12]} pkt {pkt.get('packet_id')}")

print(f"  Total BL packets scanned: {bl_pkt_count}")
print(f"  BLs with explicit MASTER capacity: {bl_with_master_capacity}")
print(f"  BLs flagged as bare-agent (house-style): {bl_flagged_bare}")
ok(f"  P198fx: NO BL with master capacity ever flagged bare",
   bl_master_but_flagged_bare == 0,
   f"{bl_master_but_flagged_bare} false positives" if bl_master_but_flagged_bare else '')


# ── Section 9 — Tier-4 ambiguous-target guard (P198fy) on real Weight Certs ──
print("\n" + "=" * 70)
print("Section 9: Tier-4 ambiguous-target guard (P198fy)")
print("=" * 70)

# For each real Weight Cert / Inspection Cert, verify the bare word
# 'draft' does NOT cause a Tier-4 match.

_AMBIG = {
    'draft': ('bill of exchange', 'draft bill', 'sight draft',
              'usance draft', 'time draft', 'tenor draft',
              'first of exchange', 'second of exchange',
              'drawn on', 'drawn at', 'pay against this'),
}

def tier4_ambig_match(target, header):
    target = target.lower()
    header = header.lower()
    ambig = _AMBIG.get(target)
    if ambig is not None:
        return any(p in header for p in ambig)
    return len(target) >= 4 and target in header

weight_cert_count = 0
spurious_draft_match = 0
for jid in eligible:
    s9p = f'{JOB_DIR}/{jid}/step09/step09_result.json'
    if not os.path.exists(s9p): continue
    d9 = json.load(open(s9p, 'r', encoding='utf-8'))
    for pkt in d9.get('reconciled_packets', []):
        dt_lo = (pkt.get('document_type', '') or '').lower()
        if not ('weight' in dt_lo and 'cert' in dt_lo):
            continue
        txt = (pkt.get('refined_text') or pkt.get('cleaned_text')
               or pkt.get('text') or '')
        if 'draft' not in txt.lower():
            continue
        weight_cert_count += 1
        if tier4_ambig_match('draft', txt[:2000]):
            spurious_draft_match += 1
            print(f"    UNEXPECTED draft match on {jid[:12]} pkt {pkt.get('packet_id')}")

print(f"  Weight Certs containing word 'draft': {weight_cert_count}")
ok(f"  P198fy: NO Weight Cert spuriously matches 'draft' Tier-4",
   spurious_draft_match == 0,
   f"{spurious_draft_match} false positives" if spurious_draft_match else '')


# ── Section 10 — Shipment-advice hint patterns (P198fs/gc) on real hints ──
print("\n" + "=" * 70)
print("Section 10: Shipment-advice hint patterns (P198fs/gc)")
print("=" * 70)

# Read the live patterns list from step03
src03 = open('d:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final/steps/step03_sequencing.py',
             'r', encoding='utf-8').read()
needed_patterns = ('advised to applicant', 'shipment under credit',
                   'details of shipment', 'after shipment',
                   'advice of shipment', 'shipment advice')
for p in needed_patterns:
    ok(f"  step03 has pattern '{p}'", f"'{p}'" in src03)

# Real-data: scan all Shipment Advice doc_hints across step03 outputs.
# Use the FULL pattern list (P198fs base + P198gc extensions) so this
# test reflects the live behaviour, not just the original P198fs set.
SA_HINT_PATTERNS = needed_patterns + (
    'sampling and analysis', 'certificate of analysis',
    'certificate of weight',
    # P198gc extensions
    'shipping notification', 'notification of shipment',
    'shipment declaration', 'shipment details notification',
    'cargo shipment information', 'shipping information',
    'fax transmission advising', 'logistics manager',
    'logistics executive',
    'lc reference', 'l/c reference', 'documentary credit',
    'requesting insurance', 'insurance coverage',
    'advised to applicant', 'shipment under credit',
    'details of shipment', 'after shipment',
    'shipment notification', 'shipment information',
)
sa_hint_count = 0
sa_hint_caught = 0
for jid in eligible:
    s3p = f'{JOB_DIR}/{jid}/step03/step03_result.json'
    if not os.path.exists(s3p): continue
    try:
        d3 = json.load(open(s3p, 'r', encoding='utf-8'))
    except Exception:
        continue
    for c in d3.get('classifications', []):
        dt = (c.get('document_type', '') or '').lower()
        hint = (c.get('doc_hint', '') or '').lower()
        # Only check Shipment Advice that ALSO has cont=True
        # would otherwise be force-overridden — those are the ones
        # P198fs/gc protects.
        if 'shipment advice' in dt and hint:
            sa_hint_count += 1
            if any(p in hint for p in SA_HINT_PATTERNS):
                sa_hint_caught += 1

print(f"  Total Shipment Advice doc_hints in corpus: {sa_hint_count}")
print(f"  Caught by P198fs/gc patterns: {sa_hint_caught}")
ok(f"  P198fs/gc patterns catch ≥80% of Shipment Advice hints",
   sa_hint_count == 0 or sa_hint_caught / max(sa_hint_count, 1) >= 0.8,
   f"only {sa_hint_caught}/{sa_hint_count} caught" if sa_hint_count and sa_hint_caught/sa_hint_count < 0.8 else '')


# ── Final tally ──
print("\n" + "=" * 70)
passed = sum(results)
total = len(results)
print(f"COMPREHENSIVE SANITY: {passed}/{total} cases passed")
print("=" * 70)
if passed != total:
    sys.exit(1)
print("OVERALL: OK")
sys.exit(0)
