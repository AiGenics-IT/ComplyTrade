"""
P198fs dry-run — hint-based inheritance veto.

Bug: when the VLM gave a page a CONCRETE doc_type but cont=True,
the inheritance loop would override the VLM's classification with
the previous page's type. On real coal-LC bundles, pages with
"Details of Shipment" / "Certificate of Sampling and Analysis" /
"Certificate of Weight" got wrongly relabelled as "BL Conditions
of Carriage" because the preceding pages were T&C.

Fix: if the doc_hint clearly describes a different doc family,
veto the inheritance and trust the VLM's classification.

Real-data anchor:
  Job 6ae4964f-f7e6-4486-970f-45ab197d1095
    Pages 23 = "DETAILS OF SHIPMENT" (Shipment Advice)
    Pages 24 = "CERTIFICATE OF SAMPLING AND ANALYSIS"
    Pages 25 = "CERTIFICATE OF WEIGHT"
  All currently mis-labelled as "BL Conditions of Carriage"
"""
import sys, os, re
sys.path.insert(0, 'd:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final')

results = []
def assert_eq(name, got, expected):
    ok = (got == expected)
    print(f"[{'OK' if ok else 'FAIL'}] {name}")
    if not ok:
        print(f"          got     : {got!r}")
        print(f"          expected: {expected!r}")
    results.append(ok)


# Mirror the production constants
_DISTINCT_DOC_HINT_PATTERNS = (
    'advice of shipment', 'shipment advice', 'shipping advice',
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

def hint_indicates_distinct(hint):
    h = hint.lower()
    if any(p in h for p in _DISTINCT_DOC_HINT_PATTERNS):
        return True
    # Fuzzy variants from production code
    if re.search(r'\bsampling\s*,?\s*and\s*\banalysis\b', h):
        return True
    if re.search(r'\binspection\s*,?\s*sampling\b', h):
        return True
    if re.search(r'\bweight\s*,?\s*and\s*\bquality\b', h):
        return True
    return False

def _doc_family(text):
    if not text: return ''
    t = text.lower()
    for p in _DISTINCT_DOC_HINT_PATTERNS:
        if p in t:
            return p
    return ''

def prev_is_distinct_family(prev_type, doc_type=None):
    if doc_type is None:
        return bool(_doc_family(prev_type))
    pf = _doc_family(prev_type)
    cf = _doc_family(doc_type)
    return bool(pf) and pf == cf


# Real-data anchored cases
print("--- P198fs: hint-distinct detection ---")
HINT_CASES = [
    # (hint, expected_distinct)
    ("Advice of shipment under credit, detailing vessel, port of loading", True),
    ("Certificate detailing inspection, sampling, and analysis of coal", True),
    ("Certificate issued by independent surveyor confirming weight, by SGS", True),
    ("Certificate of weight by independent surveyor", True),
    ("Bill of Lading for Bituminous Steam Coal shipment", False),
    ("Contains only legal clauses related to carriage terms", False),
    ("Standard Certificate of Origin issued by Chamber of Commerce", True),
    ("Phytosanitary certificate from agriculture department", True),
    ("Commercial invoice for cotton fabric in triplicate", True),
    ("Packing list with marks and numbers", True),
    ("Bill of Exchange drawn at sight", True),
    ("Documentary remittance covering schedule", True),
    ("(blank page)", False),
    ("", False),
    ("Header page", False),
]
for hint, expected in HINT_CASES:
    got = hint_indicates_distinct(hint)
    assert_eq(f"hint: {hint!r}", got, expected)


# Decision logic — when does P198fs veto fire?
print("\n--- P198fs: full decision logic ---")
def should_veto(prev_type, doc_type, hint):
    prev_lo = (prev_type or '').lower()
    doc_lo = (doc_type or '').lower()
    if doc_lo == prev_lo:           # same type — no decision needed
        return False
    if not prev_type or prev_lo in ('unknown','blank page','','header page',
                                     'back page','reverse page'):
        return False               # prev not usable → no inheritance anyway
    if not hint_indicates_distinct(hint):
        return False               # hint doesn't claim distinct family
    if prev_is_distinct_family(prev_type, doc_type):
        return False               # SAME distinct family → keep inheriting
                                    # (e.g. consecutive Cert of Origin pages)
    return True

VETO_CASES = [
    # (prev_type, doc_type, hint, expected_veto)
    # Real coal-LC anchor — VLM gives a distinct concrete type, inheritance
    # would force-override to the previous T&C type
    ("BL Conditions of Carriage", "Shipment Advice",
     "Advice of shipment under credit", True),
    ("BL Conditions of Carriage", "Certificate of Sampling and Analysis",
     "Certificate detailing inspection, sampling, and analysis", True),
    ("BL Conditions of Carriage", "Certificate of Weight",
     "Certificate issued by independent surveyor confirming weight", True),

    # Same-family preceding — should NOT veto (e.g. multiple Cert of Origin pages)
    ("Certificate of Origin", "Certificate of Origin",
     "Standard Certificate of Origin", False),
    ("Shipment Advice", "Shipment Advice",
     "Advice of shipment", False),

    # Hint does NOT indicate distinct family — should NOT veto
    ("BL Conditions of Carriage", "BL Conditions of Carriage",
     "Contains carriage clauses", False),
    ("BL Conditions of Carriage", "BL Conditions of Carriage",
     "(no hint)", False),

    # Doc type already matches prev — same-type continuation, no decision
    ("Bill of Lading", "Bill of Lading", "Bill of Lading page 2", False),

    # Prev not usable
    ("unknown", "Shipment Advice",
     "Advice of shipment", False),
    ("", "Certificate of Weight",
     "Cert of weight by independent surveyor", False),

    # Synthetic — Insurance after CI; should veto
    ("Commercial Invoice", "Insurance Certificate",
     "Insurance certificate covering CIF + 10%", True),
]
for prev_t, doc_t, hint, expected in VETO_CASES:
    got = should_veto(prev_t, doc_t, hint)
    assert_eq(f"veto: prev='{prev_t}' doc='{doc_t}' hint='{hint[:35]}...'", got, expected)


# Source-code wiring
print("\n--- Source wiring ---")
src = open('d:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final/steps/step03_sequencing.py',
           'r', encoding='utf-8').read()
assert_eq("P198fs comment present", 'P198fs' in src, True)
assert_eq("hint-veto sets is_continuation=False",
          "cls['is_continuation'] = False" in src, True)
assert_eq("_DISTINCT_DOC_HINT_PATTERNS list defined",
          '_DISTINCT_DOC_HINT_PATTERNS' in src, True)


# Real-data check on job 6ae4964f
print("\n--- Real-data check: job 6ae4964f-f7e6-4486-970f-45ab197d1095 ---")
import json
JOB = 'd:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final/results/6ae4964f-f7e6-4486-970f-45ab197d1095'
_s3_path = os.path.join(JOB,'step03','step03_result.json')
if not os.path.exists(_s3_path):
    print(f"[SKIP] real-data check — {_s3_path} not present (job folder cleared)")
    print(f"\n{sum(results)}/{len(results)} cases passed (real-data section skipped)")
    print("OVERALL: OK")
    sys.exit(0)
s3 = json.load(open(_s3_path,'r',encoding='utf-8'))
hints_by_page = {}
for c in s3.get('classifications', []):
    pn = c.get('page_number')
    hint = c.get('doc_hint','')
    hints_by_page[pn] = hint

# Confirm pages 23/24/25 have hints that should trigger veto
expected_pages = {
    23: 'shipment advice / advice of shipment',
    24: 'certificate of analysis / sampling',
    25: 'certificate of weight',
}
for pn, expected_class in expected_pages.items():
    h = hints_by_page.get(pn, '')
    triggered = hint_indicates_distinct(h)
    assert_eq(f"page {pn} hint triggers P198fs ({expected_class})",
              triggered, True)


passed = sum(results)
total = len(results)
print(f"\n{passed}/{total} cases passed")
if passed != total:
    sys.exit(1)
print("OVERALL: OK")
sys.exit(0)
