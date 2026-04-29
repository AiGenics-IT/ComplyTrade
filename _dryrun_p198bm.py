"""Dry-run for P198bm freight-wording refinement."""
import re

_key_matchers = [
    ('FREIGHT PAYABLE AS PER CHARTER PARTY',
     re.compile(r'\bFREIGHT\s+PAYABLE\s+AS\s+PER\s+CHARTER\s+PART[YI]\b')),
    ('FREIGHT PAYABLE AT DESTINATION',
     re.compile(r'\bFREIGHT\s+PAYABLE\s+AT\s+DESTINATION\b|\bFREIGHT\s+PAYABLE\s+AT\s+(?:THE\s+)?PORT\s+OF\s+DISCHARGE\b')),
    ('FREIGHT PREPAID',
     re.compile(r'\bFREIGHT\s+PREPAID\b|\bPREPAID\s+FREIGHT\b|\bFREIGHT\s+PAID\b')),
    ('FREIGHT COLLECT',
     re.compile(r'\bFREIGHT\s+COLLECT\b|\bCOLLECT\s+FREIGHT\b|\bFREIGHT\s+TO\s+COLLECT\b')),
    ('FREIGHT FORWARD',
     re.compile(r'\bFREIGHT\s+FORWARD\b(?!ER|ERS|ING|ED)|\bFREIGHT\s+TO\s+BE\s+FORWARDED\b')),
    ('FREIGHT PAYABLE',
     re.compile(r'\bFREIGHT\s+PAYABLE\b')),
]

_prohibitive_re = re.compile(
    r'\b(?:NOT\s+ACCEPT|MUST\s+NOT|SHALL\s+NOT|NOT\s+PRESENTED|'
    r'NOT\s+PERMITTED|NOT\s+ALLOWED|FORBIDDEN|PROHIBIT|'
    r'UNACCEPTABLE|NOT\s+TO\s+BE|CANNOT\s+BE|WILL\s+NOT\s+BE)\b',
)

_doc_type_prohibition_re = re.compile(
    r'\bFREIGHT\s+FORWARDER[S\'’]?\b|\bFIATA\b|\bNVOCC\b|'
    r'\bHOUSE\s+(?:B\s*/\s*L|BILL\s+OF\s+LADING)\b|'
    r'\bNON[\s\-]VESSEL\s+OPERAT',
)


def classify(cond):
    u = cond.upper()
    if 'FREIGHT' not in u:
        return ('SKIP', 'no-freight-keyword')
    if _prohibitive_re.search(u):
        return ('SKIP', 'prohibitive')
    if _doc_type_prohibition_re.search(u):
        return ('SKIP', 'doc-type-prohibition')
    for k, p in _key_matchers:
        if p.search(u):
            return ('APPLY', k)
    return ('SKIP', 'no-matching-key')


# (label, condition, expected_action, expected_key_or_skip_reason)
cases = [
    # ── User's specific false-FAIL case ──
    ("User case: FF BL must not be presented (prohibitive)",
     "Bills of Lading having any reference of issuer being a freight forwarder must not be presented.",
     'SKIP', None),

    ("FIATA prohibition",
     "Bills of Lading showing words like FIATA are not acceptable.",
     'SKIP', None),

    ("NVOCC prohibition",
     "BL stated to be issued by a non-vessel operating carrier company is not acceptable.",
     'SKIP', None),

    ("Freight forwarder's BL not acceptable (another wording)",
     "Freight forwarder's Bill of Lading not acceptable.",
     'SKIP', None),

    ("House B/L not acceptable",
     "House B/L is not acceptable.",
     'SKIP', None),

    # ── Positive payability checks (should APPLY) ──
    ("FREIGHT PREPAID requirement",
     "The Bill of Lading must be marked FREIGHT PREPAID.",
     'APPLY', 'FREIGHT PREPAID'),

    ("FREIGHT PAYABLE AS PER CHARTER PARTY",
     "The B/L must show freight payable as per charter party.",
     'APPLY', 'FREIGHT PAYABLE AS PER CHARTER PARTY'),

    ("FREIGHT COLLECT",
     "Freight collect at destination.",
     'APPLY', 'FREIGHT COLLECT'),

    ("FREIGHT FORWARD (payability)",
     "The BL must indicate FREIGHT FORWARD.",
     'APPLY', 'FREIGHT FORWARD'),

    ("FREIGHT TO BE FORWARDED",
     "Freight to be forwarded as per charter party.",
     'APPLY', 'FREIGHT FORWARD'),

    ("Generic FREIGHT PAYABLE",
     "The Bill of Lading must show freight payable.",
     'APPLY', 'FREIGHT PAYABLE'),

    ("FREIGHT PAYABLE AT DESTINATION",
     "BL must mark freight payable at destination.",
     'APPLY', 'FREIGHT PAYABLE AT DESTINATION'),

    ("FREIGHT PAYABLE AT PORT OF DISCHARGE",
     "The BL must show freight payable at port of discharge.",
     'APPLY', 'FREIGHT PAYABLE AT DESTINATION'),

    # ── Tricky cases: FORWARDER vs FORWARD ──
    ("FORWARDER in condition should NOT match FREIGHT FORWARD key",
     "Freight forwarder name must appear.",
     'SKIP', None),  # missing prohibitive, but doc-type-prohibition regex catches "FREIGHT FORWARDER"

    ("FORWARDER (positive — extremely unusual, treat as doc-type)",
     "The freight forwarder shall sign.",
     'SKIP', None),  # doc-type-prohibition catches it

    # ── Unrelated conditions ──
    ("No FREIGHT keyword",
     "Commercial Invoice must show HS code.",
     'SKIP', None),

    ("FREIGHT in totally different context",
     "Certificate must reference freight calculation sheet.",
     'SKIP', None),  # no matching key, skip

    # ── Combined: permissive carve-out mentioning FF ──
    ("Charter Party BL permissive (not prohibitive, but has CP keywords)",
     "Charter Party B/L is acceptable.",
     'SKIP', None),  # doesn't match any freight payability key
]

print("=== P198bm freight-wording guard dry-run ===")
passed = 0
for label, cond, exp_action, exp_key in cases:
    action, detail = classify(cond)
    ok = 'OK' if action == exp_action and (exp_key is None or detail == exp_key) else 'FAIL'
    if ok == 'OK':
        passed += 1
    print(f'  [{ok}] {label}')
    print(f'         -> {action}:{detail}  (expected {exp_action}:{exp_key})')
print(f'  {passed}/{len(cases)} passed')
