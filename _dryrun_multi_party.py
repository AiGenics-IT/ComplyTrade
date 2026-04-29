"""Multi-party addressing dry-run.

LC clauses commonly say "Shipment Advice must be addressed to (a) the
insurance company X AND (b) the applicant Y". The document may carry:
  1. Both A and B on the same page
  2. Only A (pkts 34/35 style)
  3. Only B
  4. Neither
  5. Spread across multiple packets (one has A, another has B)

Tests:
  (a) P198bk fuzzy name match for each party
  (b) P198bp email-aware rescue for each party's email
  (c) Multi-doc existential aggregation — ANY packet carries the required
      party → PASS for that condition (from step14:6021+)
  (d) Cross-condition independence — condition for A and condition for B
      are scored separately
"""
import json
import re
import sys
sys.path.insert(0, '.')

_ENTITY_WORDS_RE = re.compile(
    r'\b(?:LTD|LIMITED|LLC|PLC|INC|INCORPORATED|CORP|CORPORATION|'
    r'CO|COMPANY|PVT|PRIVATE|S\.?A\.?|S\.?L\.?|B\.?V\.?|N\.?V\.?|'
    r'GMBH|AG|AB|OY)\b\.?',
    flags=re.IGNORECASE,
)


def norm_phrase(s):
    s = str(s or '').upper()
    s = re.sub(r'\b(M/?S\.?|MESSRS\.?|MR\.?|MRS\.?|DR\.?)\s+', '', s)
    s = re.sub(r'\([^)]*\)', ' ', s)
    s = _ENTITY_WORDS_RE.sub(' ', s)
    s = re.sub(r',?\s*(?:KARACHI|LAHORE|ISLAMABAD|MUMBAI|DUBAI|RIYADH|DOHA|'
               r'BEIRUT|COLOMBO|HONG\s+KONG|SINGAPORE|LONDON|NEW\s+YORK|GULBERG)\b.*$', '', s)
    s = re.sub(r',?\s*(?:PAKISTAN|INDIA|BANGLADESH|SRI\s+LANKA|UAE|SAUDI\s+ARABIA|'
               r'USA|UNITED\s+STATES|UK|UNITED\s+KINGDOM|CANADA|CHINA)\b.*$', '', s)
    s = re.sub(r'[.,;:/\\"\'—–-]+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def phrase_in_doc(name_phrase, doc_up):
    """P198bk fuzzy match: contiguous with up-to-2-word gap, or ≥85% token coverage."""
    if not name_phrase or not doc_up:
        return False
    _dn = re.sub(r'[^A-Z0-9]+', ' ', doc_up).strip()
    _dn = _ENTITY_WORDS_RE.sub(' ', _dn)
    _dn = re.sub(r'\s+', ' ', _dn)
    words = [w for w in name_phrase.split() if w]
    if not words:
        return False
    if len(words) == 1:
        return bool(re.search(r'\b' + re.escape(name_phrase) + r'\b', _dn))
    gap = r'(?:\s+\S+){0,2}\s+'
    pat = r'\b' + gap.join(re.escape(w) for w in words) + r'\b'
    if re.search(pat, _dn):
        return True
    distinct = [w for w in words if len(w) >= 3]
    if not distinct:
        return False
    hits = sum(1 for w in distinct if re.search(r'\b' + re.escape(w) + r'\b', _dn))
    return (hits / len(distinct)) >= 0.85


def check_party(party_name, doc_text):
    """Return True if party is addressed on this packet."""
    return phrase_in_doc(norm_phrase(party_name), doc_text.upper())


def check_email(email, doc_text):
    """Return True if email is on doc (with (AT) normalisation)."""
    t = doc_text.upper()
    t = re.sub(r'\(\s*AT\s*\)', '@', t)
    t = re.sub(r'\(\s*DOT\s*\)', '.', t)
    t = re.sub(r'[\[\{<]\s*AT\s*[\]\}>]', '@', t)
    t = re.sub(r'\s*@\s*', '@', t)
    return email.upper() in t


def aggregate_existential(per_packet_results):
    """Multi-doc existential: ANY PASS → PASS, else FAIL."""
    if any(r for r in per_packet_results):
        return 'PASS'
    return 'FAIL'


# ──────────────────────────────────────────────────────────────
# Scenarios — LC requires TWO parties (A AND B); various doc states
# ──────────────────────────────────────────────────────────────

INSURER = "Century Insurance Company Limited Window Takaful Operations"
APPLICANT = "H.SHEIKH NOOR-UD-DIN AND SONS (PVT) LTD"
APPLICANT_EMAIL = "ABID.HUSSAIN@TECNOPACK.COM.PK"
INSURER_EMAIL = "INFO@CICL.COM.PK"

# Document snippets
DOC_BOTH_NAMES = (
    "Shipment Advice dated 16.02.2025\n"
    "TO CENTURY INSURANCE COMPANY LIMITED WINDOW TAKAFUL OPERATIONS\n"
    "2ND FLOOR, EBRAHIM ESTATES, D/1 UNION COMMERCIAL AREA, KARACHI\n"
    "AND TO H.SHEIKH NOOR-UD-DIN AND SONS (PVT) LTD., 4-KM KAHNA KACHA ROAD, LAHORE\n"
    "L/C No: 0007LC55189/2025\n"
    "Cargo: 250 MT RBD Palm Olein\n"
)

DOC_INSURER_ONLY = (
    "Shipment Advice dated 16.02.2025\n"
    "TO CENTURY INSURANCE COMPANY LIMITED WINDOW TAKAFUL OPERATIONS\n"
    "2ND FLOOR, EBRAHIM ESTATES, D/1 UNION COMMERCIAL AREA, KARACHI\n"
    "L/C No: 0007LC55189/2025\n"
    "Cargo: 250 MT RBD Palm Olein\n"
)

DOC_APPLICANT_ONLY = (
    "Shipment Advice dated 16.02.2025\n"
    "TO H.SHEIKH NOOR-UD-DIN AND SONS (PVT) LTD., 4-KM KAHNA KACHA ROAD, LAHORE\n"
    "L/C No: 0007LC55189/2025\n"
    "Cargo: 250 MT RBD Palm Olein\n"
)

DOC_NEITHER = (
    "Shipment Advice dated 16.02.2025\n"
    "TO SOME UNRELATED PARTY CORPORATION\n"
    "L/C No: 0007LC55189/2025\n"
    "Cargo: 250 MT RBD Palm Olein\n"
)

DOC_BOTH_EMAILS = (
    "Shipment Advice dated 16.02.2025\n"
    "TO CENTURY INSURANCE COMPANY LIMITED\n"
    "EMAIL INFO(AT)CICL.COM.PK\n"
    "AND TO H.SHEIKH NOOR-UD-DIN AND SONS\n"
    "EMAIL: ABID.HUSSAIN(AT)TECNOPACK.COM.PK\n"
)

DOC_INSURER_EMAIL_ONLY = (
    "Shipment Advice dated 16.02.2025\n"
    "TO CENTURY INSURANCE COMPANY LIMITED\n"
    "EMAIL INFO(AT)CICL.COM.PK\n"
)


print("=" * 78)
print("Single-packet: LC requires TWO parties, various doc states")
print("=" * 78)

single_packet_cases = [
    # (label, doc, A_present_expected, B_present_expected)
    ("Doc has BOTH named parties",
     DOC_BOTH_NAMES, True, True),
    ("Doc has ONLY insurer (applicant missing)",
     DOC_INSURER_ONLY, True, False),
    ("Doc has ONLY applicant (insurer missing)",
     DOC_APPLICANT_ONLY, False, True),
    ("Doc has NEITHER party",
     DOC_NEITHER, False, False),
]

sp_pass = 0
for label, doc, a_exp, b_exp in single_packet_cases:
    a_got = check_party(INSURER, doc)
    b_got = check_party(APPLICANT, doc)
    ok_a = 'OK' if a_got == a_exp else 'FAIL'
    ok_b = 'OK' if b_got == b_exp else 'FAIL'
    overall = 'OK' if (ok_a == 'OK' and ok_b == 'OK') else 'FAIL'
    if overall == 'OK':
        sp_pass += 1
    print(f'  [{overall}] {label}')
    print(f'       insurer={a_got} (exp {a_exp})  applicant={b_got} (exp {b_exp})')
print(f'  {sp_pass}/{len(single_packet_cases)} single-packet cases correct')


# ──────────────────────────────────────────────────────────────
# Multi-packet scenarios — one packet has A, another has B
# ──────────────────────────────────────────────────────────────
print()
print("=" * 78)
print("Multi-packet: LC requires A AND B; parties split across packets")
print("=" * 78)

mp_cases = [
    dict(
        label="All 4 packets have BOTH → both conditions PASS",
        packets=[DOC_BOTH_NAMES] * 4,
        a_expected='PASS',
        b_expected='PASS',
    ),
    dict(
        label="Packets 1,2 have insurer only; 3,4 have applicant only",
        packets=[DOC_INSURER_ONLY, DOC_INSURER_ONLY,
                 DOC_APPLICANT_ONLY, DOC_APPLICANT_ONLY],
        a_expected='PASS',  # insurer on 1,2 (any pass → PASS)
        b_expected='PASS',  # applicant on 3,4 (any pass → PASS)
    ),
    dict(
        label="Only 1 packet has insurer; none has applicant",
        packets=[DOC_INSURER_ONLY, DOC_NEITHER, DOC_NEITHER],
        a_expected='PASS',
        b_expected='FAIL',
    ),
    dict(
        label="Only 1 packet has applicant; none has insurer",
        packets=[DOC_APPLICANT_ONLY, DOC_NEITHER, DOC_NEITHER],
        a_expected='FAIL',
        b_expected='PASS',
    ),
    dict(
        label="No packet has either party",
        packets=[DOC_NEITHER, DOC_NEITHER],
        a_expected='FAIL',
        b_expected='FAIL',
    ),
    dict(
        label="Real 11ec29b8 style: 4 shipment advices, 2 with both emails, 2 with insurer only",
        packets=[DOC_INSURER_ONLY, DOC_INSURER_ONLY,
                 DOC_BOTH_NAMES, DOC_BOTH_NAMES],
        a_expected='PASS',
        b_expected='PASS',
    ),
]

mp_pass = 0
for case in mp_cases:
    a_per_pkt = [check_party(INSURER, d) for d in case['packets']]
    b_per_pkt = [check_party(APPLICANT, d) for d in case['packets']]
    a_verdict = aggregate_existential(a_per_pkt)
    b_verdict = aggregate_existential(b_per_pkt)
    ok_a = 'OK' if a_verdict == case['a_expected'] else 'FAIL'
    ok_b = 'OK' if b_verdict == case['b_expected'] else 'FAIL'
    overall = 'OK' if (ok_a == 'OK' and ok_b == 'OK') else 'FAIL'
    if overall == 'OK':
        mp_pass += 1
    print(f"  [{overall}] {case['label']}")
    print(f"       insurer per-pkt={a_per_pkt} agg={a_verdict} (exp {case['a_expected']})")
    print(f"       applicant per-pkt={b_per_pkt} agg={b_verdict} (exp {case['b_expected']})")
print(f"  {mp_pass}/{len(mp_cases)} multi-packet cases correct")


# ──────────────────────────────────────────────────────────────
# Email-based multi-party (P198bp)
# ──────────────────────────────────────────────────────────────
print()
print("=" * 78)
print("Multi-packet email — insurer email + applicant email across packets")
print("=" * 78)

em_cases = [
    dict(
        label="All 4 packets have BOTH emails",
        packets=[DOC_BOTH_EMAILS] * 4,
        insurer_expected='PASS',
        applicant_expected='PASS',
    ),
    dict(
        label="pkts 34/35 style: insurer email only on 2 pkts; applicant email on other 2",
        packets=[DOC_INSURER_EMAIL_ONLY, DOC_INSURER_EMAIL_ONLY,
                 DOC_BOTH_EMAILS, DOC_BOTH_EMAILS],
        insurer_expected='PASS',
        applicant_expected='PASS',  # pkts 3,4 carry applicant email
    ),
    dict(
        label="Only insurer email anywhere — applicant email missing",
        packets=[DOC_INSURER_EMAIL_ONLY, DOC_INSURER_EMAIL_ONLY,
                 DOC_INSURER_EMAIL_ONLY],
        insurer_expected='PASS',
        applicant_expected='FAIL',
    ),
]

em_pass = 0
for case in em_cases:
    i_per = [check_email(INSURER_EMAIL, d) for d in case['packets']]
    a_per = [check_email(APPLICANT_EMAIL, d) for d in case['packets']]
    i_verdict = aggregate_existential(i_per)
    a_verdict = aggregate_existential(a_per)
    ok_i = 'OK' if i_verdict == case['insurer_expected'] else 'FAIL'
    ok_a = 'OK' if a_verdict == case['applicant_expected'] else 'FAIL'
    overall = 'OK' if (ok_i == 'OK' and ok_a == 'OK') else 'FAIL'
    if overall == 'OK':
        em_pass += 1
    print(f"  [{overall}] {case['label']}")
    print(f"       insurer-email per-pkt={i_per} agg={i_verdict} (exp {case['insurer_expected']})")
    print(f"       applicant-email per-pkt={a_per} agg={a_verdict} (exp {case['applicant_expected']})")
print(f"  {em_pass}/{len(em_cases)} email-aggregation cases correct")


# ──────────────────────────────────────────────────────────────
# Tricky name variants — corporate suffix / punctuation / case
# ──────────────────────────────────────────────────────────────
print()
print("=" * 78)
print("Name-variant tolerance — PVT/LTD, punctuation, case")
print("=" * 78)

variant_cases = [
    ("LC name with PVT LTD, doc without PVT LTD",
     "H.SHEIKH NOOR-UD-DIN AND SONS (PVT) LTD",
     "TO H SHEIKH NOOR UD DIN AND SONS in Lahore",
     True),
    ("LC name without PVT LTD, doc with PVT LTD",
     "H.SHEIKH NOOR-UD-DIN AND SONS",
     "TO H.SHEIKH NOOR-UD-DIN AND SONS (PVT) LIMITED, LAHORE",
     True),
    ("LC name has corporate tokens anywhere, doc rearranged",
     "CENTURY INSURANCE COMPANY LIMITED",
     "TO CENTURY INSURANCE CO LIMITED, Karachi",
     True),
    ("Different company with similar name (NEGATIVE)",
     "CENTURY INSURANCE COMPANY LIMITED",
     "TO ACME INSURANCE COMPANY LIMITED, Karachi",
     False),
    ("Completely different (NEGATIVE)",
     "H.SHEIKH NOOR-UD-DIN AND SONS",
     "TO XYZ TRADERS LLC",
     False),
    ("Case mismatch only",
     "Apical Middle East FZCO",
     "TO APICAL MIDDLE EAST FZCO",
     True),
]
v_pass = 0
for label, name, doc, expected in variant_cases:
    got = check_party(name, doc)
    ok = 'OK' if got == expected else 'FAIL'
    if ok == 'OK':
        v_pass += 1
    print(f'  [{ok}] {label}')
    print(f'       name={name!r}')
    print(f'       doc ={doc[:80]!r}')
    print(f'       match={got} (expected {expected})')
print(f'  {v_pass}/{len(variant_cases)} variant cases correct')


print()
print("=" * 78)
print(f"Single-packet two-party: {sp_pass}/{len(single_packet_cases)}  | "
      f"Multi-packet name: {mp_pass}/{len(mp_cases)}  | "
      f"Multi-packet email: {em_pass}/{len(em_cases)}  | "
      f"Variants: {v_pass}/{len(variant_cases)}")
total = sp_pass + mp_pass + em_pass + v_pass
grand = len(single_packet_cases) + len(mp_cases) + len(em_cases) + len(variant_cases)
print(f"OVERALL: {total}/{grand}")
print("=" * 78)
