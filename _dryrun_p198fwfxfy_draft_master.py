"""
P198fw / P198fx / P198fy dry-run.

P198fw: Step 7 must NOT extract phantom required documents from F47A
        clauses that are RULES (third-party rules, fee rules,
        price-adjustment formulas) rather than document requirements.

P198fx: Step 14 _bare_agent flag must NOT trip on stray "AS AGENT"
        occurrences in BL Terms & Conditions boilerplate when a
        properly-qualified "AS AGENT FOR THE MASTER" exists in the
        signing block. Real-data anchor: job 1450d59f BL signed
        "FOR AND ON BEHALF OF THE MASTER" was being flagged as
        house BL.

P198fy: Step 14 _find_matching_docs Tier 4 must NOT match a Weight
        Certificate (which contains "draft survey") to a "Draft" /
        "Bill of Exchange" requirement. The bare word "draft" is
        ambiguous; the multi-word phrase "bill of exchange" or
        "first of exchange" must be required.
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

# ── P198fw — F47A rule patterns ─────────────────────────────────
print("--- P198fw: F47A rule-pattern filter ---")
_F47A_RULE_PATTERNS = (
    r'\bARE\s+ACCEPTABLE\b', r'\bIS\s+ACCEPTABLE\b',
    r'\bACCEPTABLE\.\s*$', r'\bNOT\s+ACCEPTABLE\b',
    r'\bDATED\s+PRIOR\s+TO\b',
    r'\bTHIRD\s+PARTY\s+DOCUMENTS\b',
    r'\bDISCREPANC(?:Y|IES)\s+(?:CHARGES|FEE|FEES)\b',
    r'\bCHARGES\s+WILL\s+BE\s+DEDUCTED\b',
    r'\bFEE\s+(?:OF|WILL|MUST|SHALL)\b',
    r'\bPRICE\s+ADJUSTMENT\b',
    r'\bGROSS\s+CALORIFIC\s+VALUE\b',
    r'\bIF\s+THE\s+ACTUAL\b',
    r'\bWITHIN\s+\d+\s+DAYS\s+OF\s+SHIPMENT\b',
    r'\bDOCUMENTS\s+MUST\s+BE\s+SENT\s+TO\b',
    r'\bENDORSED\s+ON\s+THE\s+REVERSE\b',
    r'\bDRAWN\s+AND\s+NEGOTIATED\b',
    r'\bFREIGHT\s+FORWARDER\b.*\b(?:NOT\s+ACCEPT|ACCEPTABLE)',
    r'\bCHARTER\s+PARTY\s+(?:BILL|DATE)\b',
    r'\bOVERWRITING\b', r'\bALTERATION\b',
    r'\bQUANTITY\s+DIFFERENT\s+FROM\b',
    r'\bSHOWING\s+QUANTITY\s+DIFFERENT\b',
)
def is_rule(text):
    t = text.upper()
    return any(re.search(p, t) for p in _F47A_RULE_PATTERNS)

# Real F47A clauses from job 1450d59f — should ALL be rules
F47A_RULES = [
    "DOCUMENTS DATED PRIOR TO DATE OF ISSUANCE OF THIS LC ACCEPTABLE.",
    "THIRD PARTY DOCUMENTS ARE ACCEPTABLE EXCEPT INVOICE AND DRAFT",
    "USD 116/- DISCREPANCY CHARGES WILL BE DEDUCTED INCASE OF DOCUMENTS CONTAIN DISCREPANCY.",
    "PRICE ADJUSTMENTS CLAUSE:\nGROSS CALORIFIC VALUE\nIF THE ACTUAL GROSS CALORIFIC VALUE IS LESS THAN...",
    "DOCUMENTS WITHIN 21 DAYS OF SHIPMENT BUT WITHIN LC EXPIRY.",
    "DOCUMENTS MUST BE SENT TO BANK AL-HABIB LTD. TECHNO CITY...",
    "CERTIFICATES SHOWING QUANTITY DIFFERENT FROM BL AND INVOICE ARE ACCEPTABLE.",
    "ANY OVERWRITING, ALTERATION ON DOCUMENTS NOT ACCEPTABLE",
    "CHARTER PARTY BILL OF LADING ACCEPTABLE. CHARTER PARTY DATE IS DIFFERENT FROM BL DATE...",
    "FREIGHT FORWARDER AND HOUSE B/L NOT ACCEPTABLE",
]
for txt in F47A_RULES:
    assert_eq(f"rule-detected: {txt[:50]}...", is_rule(txt), True)

# F47A clauses that GENUINELY require new documents — should NOT be rules
F47A_REQUIREMENTS = [
    "BENEFICIARY MUST PROVIDE A CERTIFICATE OF ANALYSIS FROM XYZ LABORATORY",
    "ADDITIONAL FUMIGATION CERTIFICATE FROM AGRICULTURE DEPARTMENT REQUIRED",
    "PHYTOSANITARY CERTIFICATE TO BE INCLUDED IN THE DOCUMENTS",
    "BENEFICIARY'S CERTIFICATE CONFIRMING THAT GOODS ARE NEW",
]
for txt in F47A_REQUIREMENTS:
    assert_eq(f"NOT-rule: {txt[:50]}...", is_rule(txt), False)


# ── P198fx — bare_agent + capacity-affirm proximity ──────────────
print("\n--- P198fx: bare_agent guard with capacity-affirm proximity ---")
_QUALIFIERS = ('MASTER', 'CARRIER', 'OWNER', 'OWNERS',
               'CHARTERER', 'CHARTERERS',
               'SHIPPING LINE', 'SHIPPING COMPANY',
               'THE VESSEL', 'THE SHIP')
_SIGN_MARKERS = ('[SIGNATURE]', 'SIGNATURE:', 'SIGNED BY',
                 'AUTHORIZED SIGNATORY', 'AUTHORISED SIGNATORY',
                 'FOR AND ON BEHALF OF', 'STAMP:')

_CAPACITY_AFFIRMS = (
    'AS MASTER', 'MASTER OF THE VESSEL', 'MASTER OF THE SHIP',
    'AS THE MASTER', 'SIGNED BY THE MASTER',
    'AS AGENT FOR THE MASTER', 'AS AGENTS FOR THE MASTER',
    'AS AGENT FOR MASTER', 'AS AGENTS FOR MASTER',
    'AS AGENT ON BEHALF OF THE MASTER',
    'AS AGENTS ON BEHALF OF THE MASTER',
    'AS AGENTS FOR AND ON BEHALF OF THE MASTER',
    'AS AGENT FOR AND ON BEHALF OF THE MASTER',
    'FOR AND ON BEHALF OF THE MASTER',
    'FOR THE MASTER AS AGENT', 'FOR THE MASTER AS AGENTS',
    'AS AGENTS ONLY FOR AND BY AUTHORITY OF THE MASTER',
    'AS AGENT ONLY FOR AND BY AUTHORITY OF THE MASTER',
    'SIGNED BY THE CARRIER',
    'AS AGENT FOR THE CARRIER', 'AS AGENTS FOR THE CARRIER',
    'AS AGENT FOR AND ON BEHALF OF THE CARRIER',
    'AS AGENTS FOR AND ON BEHALF OF THE CARRIER',
    'FOR AND ON BEHALF OF THE CARRIER',
    'AS OWNER', 'AS OWNERS',
    'AS AGENT FOR THE OWNER', 'AS AGENTS FOR THE OWNER',
    'AS AGENT FOR AND ON BEHALF OF THE OWNER',
    'AS AGENTS FOR AND ON BEHALF OF THE OWNER',
    'FOR AND ON BEHALF OF THE OWNER',
    'AS CHARTERER', 'AS CHARTERERS',
    'FOR AND ON BEHALF OF THE CHARTERER',
)

def cap_proof_hit(doc_text):
    """Find any capacity-affirm phrase near a signature block."""
    t = doc_text.upper()
    n = len(t)
    for ph in _CAPACITY_AFFIRMS:
        p = t.find(ph)
        while p >= 0:
            in_last = (p >= int(n * 0.60))
            near_sig = False
            for sm in _SIGN_MARKERS:
                sp = t.find(sm, max(0, p - 300), p + 300 + len(ph))
                if sp >= 0 and abs(sp - p) <= 300:
                    near_sig = True
                    break
            if in_last or near_sig:
                return (ph, p)
            p = t.find(ph, p + 1)
    return None

def bare_agent_check(doc_text):
    """Mirrors the production logic exactly."""
    t = doc_text.upper()
    n = len(t)
    bare_unq = False
    qual = False
    agent_re = re.compile(r'\bAS\s+AGENTS?\b', flags=re.IGNORECASE)
    for m in agent_re.finditer(t):
        s, e = m.start(), m.end()
        in_last = (s >= int(n * 0.60))
        near_sig = False
        for sm in _SIGN_MARKERS:
            sp = t.find(sm, max(0, s - 300), e + 300)
            if sp >= 0:
                near_sig = True
                break
        if not (in_last or near_sig):
            continue
        window = t[e:e + 120]
        pre = t[max(0, s - 40):s]
        if any(q in window for q in _QUALIFIERS) or \
           any(q in pre for q in _QUALIFIERS):
            qual = True
        else:
            bare_unq = True
    bare = bare_unq and not qual
    # P198fx — capacity-affirm override
    if cap_proof_hit(doc_text) is not None:
        bare = False
    return bare

# Cases
# 1. Real BL with proper "AS AGENT FOR AND ON BEHALF OF THE MASTER" in signing block + boilerplate "as agent" in T&C
BL_GOOD_WITH_BOILERPLATE = (
    "BILL OF LADING\n"
    "Shipper: International Energy Resources FZC\n"
    "Consignee: TO ORDER\n"
    "[CARGO DETAILS]\n"
    "TERMS AND CONDITIONS\n"
    "1. The carrier as agent of the shipper shall not be liable...\n"
    "2. As agent acting on instructions, the carrier may...\n"
    "3. Routine clauses about the carrier as agent during transit.\n"
    + "Filler clauses... " * 50
    + "\nFOR AND ON BEHALF OF THE MASTER\n"
    "M/V VESSEL\n"
    "AUTHORIZED SIGNATORY\n"
    "STAMP: PT. RIANDY FIESTA SAMUDERA"
)
assert_eq("Good BL: 'FOR AND ON BEHALF OF THE MASTER' near signature → not bare_agent",
          bare_agent_check(BL_GOOD_WITH_BOILERPLATE), False)

# 2. House BL with only bare "AS AGENT" in signing block
BL_HOUSE = (
    "BILL OF LADING\n"
    "Shipper: ABC Co.\n"
    + "Routine clauses... " * 60
    + "\nSIGNED BY M.Y LOGISTICS\n"
    "AS AGENT\n"
    "AUTHORIZED SIGNATORY"
)
assert_eq("House BL: only bare 'AS AGENT' near signature → bare_agent",
          bare_agent_check(BL_HOUSE), True)

# 3. BL with qualified agent in signing area
BL_AGENT_FOR_CARRIER = (
    "BILL OF LADING\n"
    + "Filler... " * 60
    + "\nAS AGENT FOR AND ON BEHALF OF THE CARRIER\n"
    "AUTHORIZED SIGNATORY"
)
assert_eq("BL: 'AS AGENT FOR... CARRIER' → not bare_agent",
          bare_agent_check(BL_AGENT_FOR_CARRIER), False)

# 4. BL with stray bare AS AGENT in T&C only (no signing-block agent at all)
# Should NOT be flagged as bare_agent because the T&C occurrences are NOT
# in the signing zone.
BL_TC_ONLY = (
    "Beginning of BL...\n"
    "Clause: as agent for the shipper, the forwarder may handle...\n"
    + "filler... " * 200
    + "\nNo signing block visible.\n"
)
# The "as agent" appears at the very top — NOT in last 40% AND not near sig markers.
# So it should be ignored entirely.
assert_eq("Boilerplate-only 'as agent' at top of doc → not bare_agent (skipped, out of signing zone)",
          bare_agent_check(BL_TC_ONLY), False)

# 5. UCP 600 art 20 — BL signed but NO capacity stated (no master/carrier/agent)
# Per user feedback: "if the signing capacity is not mentioned in the BL
# that is also a discrepancy even if it's signed". Confirm _cap_proof_hit
# is None so _no_capacity_proof = True → FAIL stands.
BL_NO_CAPACITY = (
    "BILL OF LADING\n"
    "Shipper: ABC Co.\n"
    + "Routine clauses... " * 60
    + "\nSIGNED BY:\n"
    "[signature]\n"
    "John Smith\n"
    "Authorized Signatory\n"
    # No "AS MASTER", no "AS AGENT FOR THE MASTER", no "FOR AND ON BEHALF OF THE..."
)
assert_eq("BL with NO capacity stated → cap_proof_hit is None (FAIL stands)",
          cap_proof_hit(BL_NO_CAPACITY), None)


# ── P198fy — Tier 4 ambiguous-target guard ─────────────────────
print("\n--- P198fy: Tier 4 'draft' must NOT match Weight Cert ---")

# Mock _find_matching_docs Tier 4 guard logic
_AMBIGUOUS_BARE_TARGETS = {
    'draft': ('bill of exchange', 'draft bill', 'sight draft',
              'usance draft', 'time draft', 'tenor draft',
              'first of exchange', 'second of exchange',
              'drawn on', 'drawn at', 'pay against this'),
    'bill': ('bill of exchange', 'bill of lading',),
    'note': ('debit note', 'credit note', 'cover note',
             'promissory note',),
    'letter': ('forwarding letter', 'cover letter', 'covering letter',
               'letter of indemnity', 'letter of guarantee',),
}

def tier4_match(target, header):
    """Return True if the packet's header should match the target."""
    target = target.lower()
    header = header.lower()
    ambig = None
    for bare, phrases in _AMBIGUOUS_BARE_TARGETS.items():
        if target == bare or target.startswith(bare + ' '):
            ambig = phrases
            break
    if ambig is not None:
        return any(p in header for p in ambig)
    # Non-ambiguous: bare word match suffices
    return len(target) >= 4 and target in header

# Weight certificate text mentions "draft survey"
WEIGHT_CERT_TEXT = (
    "CERTIFICATE OF WEIGHT\n"
    "Issued by PT. ANINDYA WIRAPUTRA KONSULT\n"
    "Total quantity confirmed by vessel's draft survey.\n"
    "Initial draft: 8.50m, final draft: 9.20m.\n"
    "Quantity: 5,000.00 MT"
)
assert_eq("'draft' target should NOT match Weight Cert with 'draft survey'",
          tier4_match('draft', WEIGHT_CERT_TEXT), False)

# Real draft document
REAL_DRAFT_TEXT = (
    "BILL OF EXCHANGE\n"
    "First of Exchange (Second unpaid)\n"
    "Drawn on Bank Al Habib Ltd.\n"
    "Pay against this draft the sum of USD 500,000.00\n"
)
assert_eq("'draft' target SHOULD match real BoE with 'bill of exchange' phrase",
          tier4_match('draft', REAL_DRAFT_TEXT), True)

# Inspection cert with "draft survey" mention
INSPECTION_CERT = (
    "CERTIFICATE OF SAMPLING AND ANALYSIS\n"
    "Issued by PT. ANINDYA WIRAPUTRA KONSULT\n"
    "Sampling per ASTM standards. Reference draft survey results.\n"
)
assert_eq("'draft' target should NOT match Inspection Cert with 'draft survey'",
          tier4_match('draft', INSPECTION_CERT), False)

# Bill of Lading text mentions "bill" in header
BL_TEXT = (
    "BILL OF LADING\n"
    "Shipper: ABC Trading Co.\n"
    "Consignee: TO ORDER\n"
)
assert_eq("'bill' target SHOULD match BL with 'bill of lading'",
          tier4_match('bill', BL_TEXT), True)
assert_eq("'note' target should NOT match BL (no debit/credit/cover note)",
          tier4_match('note', BL_TEXT), False)


# ── Source wiring checks ───────────────────────────────────────
print("\n--- Source wiring ---")
src7 = open('d:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final/steps/step07_clause_extraction.py',
            'r', encoding='utf-8').read()
src14 = open('d:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final/steps/step14_verification.py',
             'r', encoding='utf-8').read()
assert_eq("step07 has P198fw block", 'P198fw' in src7, True)
assert_eq("step07 has _F47A_RULE_PATTERNS", '_F47A_RULE_PATTERNS' in src7, True)
assert_eq("step14 has P198fx block", 'P198fx' in src14, True)
assert_eq("step14 has P198fy block", 'P198fy' in src14, True)
assert_eq("step14 has _AMBIGUOUS_BARE_TARGETS", '_AMBIGUOUS_BARE_TARGETS' in src14, True)
assert_eq("step14 has _qualified_in_sig_zone", '_qualified_in_sig_zone' in src14, True)


# ── Real-data check on job 1450d59f ─────────────────────────────
print("\n--- Real-data check: job 1450d59f F47A clauses ---")
import json
JOB = 'd:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final/results/1450d59f-220e-4536-a5ce-c1dc76dee05e'
s7 = json.load(open(os.path.join(JOB, 'step07', 'step07_result.json'), 'r', encoding='utf-8'))
slc = s7.get('structured_lc', {})
f47a_clauses = [c for c in slc.get('all_clauses', [])
                if (c.get('field_tag') or '').upper() in ('F47A', '47A')]
phantom_count = 0
real_req_count = 0
for c in f47a_clauses:
    txt = c.get('clause_text', '')
    if is_rule(txt):
        phantom_count += 1
    else:
        real_req_count += 1
print(f"  F47A clauses: {len(f47a_clauses)} | phantom-rules: {phantom_count} | real-requirements: {real_req_count}")
assert_eq("Most F47A clauses on job 1450d59f are rules (phantom-doc filter would activate)",
          phantom_count > real_req_count, True)

# Confirm the SPECIFIC phantom docs from job 1450d59f would be filtered
print("\n--- Real F47A clause-by-clause filter verification ---")
EXPECTED_PHANTOMS = (
    ("DOCUMENTS DATED PRIOR TO DATE OF ISSUANCE", "Beneficiary Certificate"),
    ("THIRD PARTY DOCUMENTS ARE ACCEPTABLE EXCEPT INVOICE AND DRAFT", "Draft Bill of Exchange"),
    ("DISCREPANCY CHARGES WILL BE DEDUCTED", "Discrepancy Fee Notice"),
    ("CERTIFICATES SHOWING QUANTITY DIFFERENT FROM", "Quantity Certificate"),
    ("PRICE ADJUSTMENTS CLAUSE", "Quality Certificate"),
)
for clause_substring, phantom_doc in EXPECTED_PHANTOMS:
    found = False
    for c in f47a_clauses:
        if clause_substring in (c.get('clause_text', '') or '').upper():
            found = True
            assert_eq(f"  '{phantom_doc}' phantom would be filtered",
                      is_rule(c.get('clause_text', '')), True)
            break
    if not found:
        print(f"  [SKIP] clause '{clause_substring}' not in this job")


# ── Real packet test: Tier 4 ─────────────────────────────────────
print("\n--- Real-data Tier 4 test on job f3ef028e (had the bug) ---")
import importlib.util
spec = importlib.util.spec_from_file_location(
    "step14_verification",
    "d:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final/steps/step14_verification.py",
)
# Don't actually import — just hand-roll the Tier 4 against real packets
PRIOR_JOB = 'd:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final/results/f3ef028e-b879-40d2-9351-39a2aff90175'
s9 = json.load(open(os.path.join(PRIOR_JOB, 'step09', 'step09_result.json'), 'r', encoding='utf-8'))
packets = s9.get('reconciled_packets', [])
print(f"  Loaded {len(packets)} packets from f3ef028e")

# Find the Weight Certificate packet
wc_pkt = None
for pkt in packets:
    pt = (pkt.get('document_type', '') or '').lower()
    if 'weight' in pt and 'cert' in pt:
        wc_pkt = pkt; break
if wc_pkt:
    wc_text = (wc_pkt.get('refined_text', '') or wc_pkt.get('cleaned_text', '')
               or wc_pkt.get('text', '') or '').lower()[:2000]
    has_draft = 'draft' in wc_text
    has_boe_phrase = any(p in wc_text for p in (
        'bill of exchange', 'first of exchange', 'second of exchange',
        'sight draft', 'usance draft', 'pay against this'))
    print(f"  Weight Cert text length: {len(wc_text)}, contains 'draft': {has_draft}, "
          f"contains BoE phrase: {has_boe_phrase}")
    assert_eq("Real Weight Cert text contains 'draft' (the original bug source)",
              has_draft, True)
    assert_eq("Real Weight Cert text does NOT contain BoE phrase (so P198fy correctly skips)",
              has_boe_phrase, False)
    assert_eq("P198fy: Weight Cert would NOT match 'draft' target",
              tier4_match('draft', wc_text), False)

# Also verify Inspection Cert wouldn't match
ins_pkt = None
for pkt in packets:
    pt = (pkt.get('document_type', '') or '').lower()
    if 'sampling' in pt or 'inspection' in pt or 'analysis' in pt:
        ins_pkt = pkt; break
if ins_pkt:
    ins_text = (ins_pkt.get('refined_text', '') or ins_pkt.get('cleaned_text', '')
                or ins_pkt.get('text', '') or '').lower()[:2000]
    assert_eq("P198fy: Inspection Cert would NOT match 'draft' target",
              tier4_match('draft', ins_text), False)


# ── Real BL signing-block test ──────────────────────────────────
print("\n--- Real BL test: 'FOR AND ON BEHALF OF THE MASTER' must not flag bare_agent ---")
# Find BL packets in job f3ef028e
bl_pkts = [p for p in packets if 'bill of lading' in (p.get('document_type', '') or '').lower()
           or 'bl conditions' in (p.get('document_type', '') or '').lower()]
print(f"  Found {len(bl_pkts)} BL/T&C packets")
checked = 0
for pkt in bl_pkts:
    bl_text = (pkt.get('refined_text', '') or pkt.get('cleaned_text', '')
               or pkt.get('text', '') or '')
    if not bl_text or len(bl_text) < 100:
        continue
    bl_up = bl_text.upper()
    has_master_qualifier = ('FOR AND ON BEHALF OF THE MASTER' in bl_up
                            or 'AS AGENT FOR THE MASTER' in bl_up
                            or 'AS AGENTS FOR THE MASTER' in bl_up
                            or 'AS AGENT FOR AND ON BEHALF OF THE MASTER' in bl_up)
    if has_master_qualifier:
        result = bare_agent_check(bl_text)
        assert_eq(f"  BL with 'FOR/AS AGENT FOR THE MASTER' (pkt={pkt.get('packet_id')}): not bare_agent",
                  result, False)
        checked += 1
print(f"  Checked {checked} BL packet(s) with master-qualifier")


passed = sum(results)
total = len(results)
print(f"\n{passed}/{total} cases passed")
if passed != total:
    sys.exit(1)
print("OVERALL: OK")
sys.exit(0)
