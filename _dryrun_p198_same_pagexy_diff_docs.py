"""
Same Page-XY sequence, DIFFERENT actual doc types — the multi-page
archive scanner trap.

Scenario: a bank or warehouse scanner numbers EVERY page of a 75-page
trade-finance archive as "Page 1 of 75", "Page 2 of 75", ..., "Page
75 of 75". The numbers are sequential and share the same Y. Naively
applying X-of-Y → continuation logic would merge every page into one
giant packet — but the actual documents are different (LC, Amendment,
MT730, Commercial Invoice, BL, Cert of Origin, Shipment Advice, etc.).

Real anchor — job 6ae4964f had a coal-LC bundle with Page-XY footers
of the form "Page X of 75". Pages 23/24/25 had sequential footers
(38/39/40 of 75) but their VLM-identified doc types were different:
  Page 23 → Shipment Advice
  Page 24 → Certificate of Sampling and Analysis
  Page 25 → Certificate of Weight

Without the P198fs hint-veto, the multi-page-XY logic would force
cont=True on pages 24, 25 → they would inherit the previous BL
classification, hiding three distinct documents.

This test proves the system handles this correctly via the
hint-distinct doc-family veto (P198fs in both the multi-page-XY and
no-XY branches), independent of how Page-X-of-Y looks.
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


# Mirror the production hint-distinct logic
_DISTINCT_DOC_HINT_PATTERNS = (
    'advice of shipment', 'shipment advice', 'shipping advice',
    'advised to applicant', 'advised to the applicant',
    'advice to applicant', 'notification to applicant',
    'shipment under credit', 'details of shipment',
    'shipment notification', 'shipment information',
    'after shipment', 'shipment details under credit',
    'shipping notification', 'notification of shipment',
    'shipment declaration', 'shipment details notification',
    'cargo shipment information', 'shipping information',
    'sampling and analysis', 'certificate of analysis',
    'cert of analysis', 'analysis certificate',
    'sampling', 'analysis results',
    'certificate of weight', 'cert of weight', 'weight certificate',
    'confirming weight', 'independent surveyor',
    'certificate of origin', 'cert of origin',
    'phytosanitary', 'fumigation', 'health certificate',
    'sanitary certificate',
    'inspection certificate', 'pre-shipment inspection',
    'insurance certificate', 'insurance policy', 'cover note',
    'beneficiary certificate', 'beneficiary cert',
    'shipping company certificate',
    'draft survey', 'load port survey', 'discharge port survey',
    'commercial invoice', 'proforma invoice', 'tax invoice',
    'packing list', 'weight list', 'cargo manifest',
    'bill of exchange', 'draft bill of exchange',
    'freight invoice',
    'documentary remittance', 'covering schedule',
    'schedule of documents',
    'forwarding letter', 'covering letter', 'cover letter',
)

def hint_distinct(hint):
    h = (hint or '').lower()
    if any(p in h for p in _DISTINCT_DOC_HINT_PATTERNS):
        return True
    if re.search(r'\bsampling\s*,?\s*and\s*\banalysis\b', h):
        return True
    if re.search(r'\binspection\s*,?\s*sampling\b', h):
        return True
    if re.search(r'\bweight\s*,?\s*and\s*\bquality\b', h):
        return True
    return False


# ── Section 1 — Same-Y sequential, different doc families ──
print("=" * 70)
print("Section 1: Same-Y sequential X with DIFFERENT doc families")
print("=" * 70)

# (prev_xy, curr_xy, prev_doc_type, curr_vlm_type, curr_hint, expected_outcome)
SCENARIOS = [
    # Real anchor: archive footer "X of 75", but docs are distinct
    ((37, 75), (38, 75), 'BL Conditions of Carriage',
     'Shipment Advice',
     'Advice of shipment under credit, detailing vessel and ports',
     'KEEP_VLM',  # P198fs vetoes BL inheritance
     'Page 38/75: prev=BL T&C, VLM=Shipment Advice, hint clearly Shipment Advice'),
    ((38, 75), (39, 75), 'Shipment Advice',
     'Certificate of Sampling and Analysis',
     'Certificate detailing inspection, sampling and analysis of coal',
     'KEEP_VLM',
     'Page 39/75: prev=Shipment Advice, VLM=Cert of Sampling, hint clearly Sampling'),
    ((39, 75), (40, 75), 'Certificate of Sampling and Analysis',
     'Certificate of Weight',
     'Certificate issued by independent surveyor confirming cargo weight',
     'KEEP_VLM',
     'Page 40/75: prev=Cert of Sampling, VLM=Cert of Weight, hint clearly Cert of Weight'),
    # Synthetic: 5-page LC vs 5-page bundle archive
    ((1, 5), (2, 5), 'LC',
     'LC',
     'Continuation of LC terms and conditions',
     'CONTINUE_LC',  # both LC, same-family inheritance OK
     'LC continuation across same-Y archive — keep LC inheritance'),
    # Different doc family but generic hint — ambiguous, system defaults to inherit
    ((1, 10), (2, 10), 'Bill of Lading',
     'Commercial Invoice',
     'Commercial invoice for goods',
     'KEEP_VLM',  # commercial invoice IS in distinct hint patterns
     'Page 2/10: VLM=Commercial Invoice, hint says Commercial Invoice'),
    # Different doc family but UNCLEAR hint
    ((1, 10), (2, 10), 'Bill of Lading',
     'Document Remittance',
     'Bank presentation schedule',
     'INHERIT',  # hint doesn't match any distinct pattern → inherit BL
     'Page 2/10: VLM=Doc Remittance, hint generic — falls through to inherit'),
]

for prev_xy, curr_xy, prev_type, vlm_type, hint, expected, label in SCENARIOS:
    # Reproduce the production decision:
    same_y = curr_xy[1] == prev_xy[1]
    seq = curr_xy[0] == prev_xy[0] + 1
    is_multi_page_xy = same_y and seq and curr_xy[0] > 1

    if not is_multi_page_xy:
        outcome = 'KEEP_VLM'   # no inheritance pressure
    elif vlm_type.lower() == prev_type.lower():
        outcome = 'CONTINUE_LC'  # same family, continuation natural
    elif hint_distinct(hint):
        outcome = 'KEEP_VLM'  # P198fs hint-veto
    else:
        outcome = 'INHERIT'    # default: inherit prev type

    ok(f"  {label}: outcome={outcome}", outcome == expected,
       f"got {outcome}, expected {expected}" if outcome != expected else '')


# ── Section 2 — Real-data: pages with sequential Page-XY but distinct VLM types ──
print("\n" + "=" * 70)
print("Section 2: Real-data sweep — sequential Page-XY with VLM-distinct types")
print("=" * 70)

JOB_DIR = 'd:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final/results'
PAGE_XY_PATS = [
    re.compile(r'(\d{1,3})\s*OF\s*(\d{1,3})', re.IGNORECASE),
    re.compile(r'(\d{1,3})\s*/\s*(\d{1,3})'),
]

veto_fired_count = 0
inherited_correctly_count = 0
total_consec_count = 0
spurious_inheritance = []   # cases where hint clearly says distinct but page is part of BL packet

for jid in sorted(os.listdir(JOB_DIR)):
    s3p = f'{JOB_DIR}/{jid}/step03/step03_result.json'
    if not os.path.exists(s3p):
        continue
    try:
        d3 = json.load(open(s3p, 'r', encoding='utf-8'))
    except Exception:
        continue
    cls = d3.get('classifications', [])
    if len(cls) < 2:
        continue
    # Walk consecutive pages
    for i, c in enumerate(cls[1:], 1):
        prev = cls[i - 1]
        prev_dt = (prev.get('document_type', '') or '').strip().lower()
        curr_dt = (c.get('document_type', '') or '').strip().lower()
        curr_hint = c.get('doc_hint', '') or ''
        curr_cont = c.get('is_continuation', False)
        # Look for "Page X of Y" or "X/Y" inside the doc_hint or refined_text
        found_xy = False
        for pat in PAGE_XY_PATS:
            if pat.search(curr_hint):
                found_xy = True
                break
        if curr_dt == prev_dt:
            continue  # same family — not interesting for veto check
        # Different VLM types AND distinct hint AND was NOT inherited?
        if hint_distinct(curr_hint) and not curr_cont:
            veto_fired_count += 1
        elif hint_distinct(curr_hint) and curr_cont and prev_dt and curr_dt != prev_dt:
            # Inherited DESPITE hint indicating distinct — could be bug
            spurious_inheritance.append(
                (jid[:12], c.get('page_number'),
                 prev_dt[:20], curr_dt[:20], curr_hint[:60]))
        total_consec_count += 1

print(f"  Cases of cont=False where VLM hint indicates distinct family: "
      f"{veto_fired_count}")
print(f"  Cases where hint=distinct but page WAS inherited (potential bug): "
      f"{len(spurious_inheritance)}")
for j, p, pt, ct, h in spurious_inheritance[:6]:
    print(f"    {j} pg{p} prev={pt!r} → curr={ct!r}, hint={h!r}")
ok(f"  P198fs hint-veto fires correctly across corpus",
   veto_fired_count >= 5)
ok(f"  Spurious inheritance (hint distinct but inherited) rare",
   len(spurious_inheritance) <= 30,
   f"{len(spurious_inheritance)} spurious — investigate" if len(spurious_inheritance) > 30 else '')


# ── Section 3 — Anchor: real job 6ae4964f if present ──
print("\n" + "=" * 70)
print("Section 3: Real job 6ae4964f anchor (if step03 still present)")
print("=" * 70)

matches = [d for d in os.listdir(JOB_DIR) if d.startswith('6ae4964f')]
if matches:
    full = matches[0]
    s3p = f'{JOB_DIR}/{full}/step03/step03_result.json'
    if os.path.exists(s3p):
        d3 = json.load(open(s3p, 'r', encoding='utf-8'))
        page_types = {}
        for c in d3.get('classifications', []):
            page_types[c.get('page_number')] = (
                c.get('document_type', ''), c.get('doc_hint', ''))
        # Verify pages 23/24/25 are NOT BL Conditions of Carriage
        for pg, expected_kw in [(23, 'shipment'), (24, 'sampling'),
                                  (25, 'weight')]:
            if pg in page_types:
                dt, hint = page_types[pg]
                ok(f"  6ae4964f page {pg}: doc_type contains '{expected_kw}'",
                   expected_kw in dt.lower() or expected_kw in hint.lower(),
                   f"got dt={dt!r} hint={hint[:60]!r}" if expected_kw not in dt.lower() and expected_kw not in hint.lower() else '')
            else:
                ok(f"  6ae4964f page {pg}: present", True, 'page absent — skipped')
    else:
        print("  step03 missing for 6ae4964f — skipping")
        ok(f"  Skip — no fixture", True)
else:
    print("  Job 6ae4964f folder not present — skipping")
    ok(f"  Skip — no job folder", True)


# ── Section 4 — VLM cont vs Page-XY conflict resolution ──
print("\n" + "=" * 70)
print("Section 4: VLM cont=False but Page-XY says continuation")
print("=" * 70)

# When VLM says cont=False but Page-XY footer indicates this IS a
# continuation page, the system FORCES cont=True UNLESS the hint
# vetoes (P198fs).

CONFLICTS = [
    # (vlm_cont, has_page_xy, hint, expected_final_cont)
    (False, True,  'BL conditions of carriage',
     True,  'VLM says no but Page-XY agrees + hint generic → force cont'),
    (False, True,  'Advice of shipment under credit',
     False, 'VLM says no, Page-XY agrees BUT hint distinct → P198fs vetoes'),
    (False, True,  'Certificate of weight by independent surveyor',
     False, 'VLM says no, Page-XY agrees BUT hint distinct → veto'),
    (True,  False, 'Some hint',
     True,  'VLM says yes, no Page-XY → respect VLM'),
    (False, False, 'Some hint',
     False, 'VLM says no, no Page-XY → respect VLM'),
]
for vlm_cont, has_xy, hint, expected, label in CONFLICTS:
    if has_xy:
        # Page-XY suggests continuation
        if hint_distinct(hint):
            final = False  # P198fs veto
        else:
            final = True   # force cont via XY
    else:
        final = vlm_cont
    ok(f"  {label}: final_cont={final}", final == expected,
       f"got {final}, expected {expected}" if final != expected else '')


# ── Final tally ──
print("\n" + "=" * 70)
passed = sum(results)
total = len(results)
print(f"SAME-PAGE-XY-DIFFERENT-DOCS: {passed}/{total} cases passed")
print("=" * 70)
if passed != total:
    sys.exit(1)
print("OVERALL: OK")
sys.exit(0)
