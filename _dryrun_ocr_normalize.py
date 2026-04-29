"""OCR normalization dry-run — cover every common letter↔digit confusion
in reference-number matching (policy, cover note, LC number, NTN, BL
number, container number, etc.). Tests that the same needle matches
regardless of which side (condition or doc) has the OCR error.

Mapping (one-way letter→digit, applied identically to BOTH sides):
    O→0, Q→0
    I→1, L→1 (also lowercase 'l' via upper)
    S→5
    B→8
    Z→2
    G→6
"""
import sys
import re
sys.path.insert(0, '.')
from steps.step14_verification import _normalize_id


cases = [
    # (label, needle_in_cond, needle_in_doc, expected_match)

    # ── O ↔ 0 (both directions) ──
    ("O↔0: cond has digit 0, doc has letter O",
     "MIPD000453", "MIPDO00453", True),
    ("O↔0: cond has letter O, doc has digit 0",
     "MIPDO00453", "MIPD000453", True),
    ("O↔0 multiple: both letters and digits mixed on each side",
     "POLICY-2023O08", "POLICY-2023008", True),
    ("Q↔0: cond has digit 0, doc has letter Q",
     "PQL1CY000", "PQL1CYQQQ", True),

    # ── I ↔ 1 (both directions) ──
    ("I↔1: cond has digit 1, doc has letter I",
     "LC-1001-59573", "LC-I00I-59573", True),
    ("I↔1: cond has letter I, doc has digit 1",
     "LC-I00I-59573", "LC-1001-59573", True),
    ("I↔1 with dashes",
     "1-I-1-I", "I-1-I-1", True),
    ("L↔1 (lowercase l uppercase L→1)",
     "CERTIFICATE-L1234", "CERTIFICATE-11234", True),

    # ── S ↔ 5 ──
    ("S↔5: cond has 5, doc has S",
     "ABC-555-XYZ", "ABC-SSS-XYZ", True),
    ("S↔5: mixed variants",
     "POLICY-S50-5S0", "POLICY-550-SS0", True),

    # ── B ↔ 8 ──
    ("B↔8: cond has 8, doc has B",
     "LC-8888-XYZ", "LC-BBBB-XYZ", True),
    ("B↔8 mixed",
     "NTN-B8B8", "NTN-8B8B", True),

    # ── Z ↔ 2 ──
    ("Z↔2: cond has 2, doc has Z",
     "REF-2222", "REF-ZZZZ", True),

    # ── G ↔ 6 ──
    ("G↔6: cond has 6, doc has G",
     "SEAL-666666", "SEAL-GGGGGG", True),

    # ── Mixed (the actual real-world 11/0000118/1024/0-0 style) ──
    ("Real policy number: same on both sides",
     "11/0000118/1024/0-0", "11/0000118/1024/0-0", True),
    ("Real policy number: OCR variant with O in doc",
     "11/0000118/1024/0-0", "11/OOOO118/1024/0-0", True),
    ("Real policy number: OCR variant with O in cond",
     "11/OOOO118/I024/O-O", "11/0000118/1024/0-0", True),
    ("Real policy number: I/1 confusion on doc",
     "11/0000118/1024/0-0", "II/0000II8/I024/0-0", True),

    # ── Complex with multiple error types ──
    ("Multi-error: I↔1 + O↔0 + S↔5 simultaneously",
     "1O5-S01-I23", "105-501-123", True),
    ("Multi-error: all letters on doc",
     "8IZG50-1Q00", "8IZG50-1Q00", True),

    # ── Negative cases — genuinely different numbers ──
    ("Different policy (negative)",
     "POLICY-12345", "POLICY-99999", False),
    # Short needle IS a substring of a longer one — documented behavior.
    # Production relies on this for partial-reference tolerance; if a
    # longer genuine reference happens to contain the shorter one, we
    # accept. Would only bite if needle is unreasonably short.
    ("Substring match (short needle inside longer number — accepted)",
     "ABC-123", "ABC-1234567", True),
    ("Totally different numbers (negative)",
     "11/0000118/1024/0-0", "99/9999999/9999/9-9", False),

    # ── Container number scenarios ──
    ("Container: same chars both sides",
     "YMLU8681239", "YMLU8681239", True),
    # OCR O↔0 within container number (common confusion — Maersk tracking
    # has owner letters + 7 digits; OCR sometimes reads 0 as O)
    ("Container: O↔0 OCR within digits",
     "MAEU1234560", "MAEU123456O", True),

    # ── Additional real-world OCR confusions ──
    ("Lowercase l ↔ 1 (after upper → L → 1)",
     "REF-111", "REF-lll", True),
    ("Mixed: I vs 1 vs l inside same number",
     "101-I01-101", "1O1-101-l0l", True),
    ("LC number with L/1 ambiguity",
     "1001LC59573", "I00lLC59573", True),

    # ── Edge cases ──
    ("Short needle (3 chars) inside exact same",
     "123", "Ref123", True),
    ("Empty needle (should not match anything meaningful)",
     "", "Any doc text", True),  # empty IS substring of anything (Python default)
]


def match(cond_ref, doc_text):
    """Mirror step14 P135 / P198cd check: cond_norm in doc_norm."""
    cond_norm = _normalize_id(cond_ref)
    doc_norm = _normalize_id(doc_text)
    return cond_norm in doc_norm


print("=" * 78)
print("OCR character-confusion normalization — bidirectional matching")
print("=" * 78)
passed = 0
for label, cond, doc, expected in cases:
    cn = _normalize_id(cond)
    dn = _normalize_id(doc)
    got = match(cond, doc)
    ok = 'OK' if got == expected else 'FAIL'
    if ok == 'OK':
        passed += 1
    print(f'  [{ok}] {label}')
    print(f'       cond={cond!r:40} → norm={cn!r}')
    print(f'       doc ={doc!r:40} → norm={dn!r}')
    print(f'       match={got}  (expected {expected})')
print()
print(f"{passed}/{len(cases)} cases correct")

# ── Real-world examples from actual jobs ──
print()
print("=" * 78)
print("Real-world examples (from actual jobs we've verified)")
print("=" * 78)

real_cases = [
    # Every variant seen in production
    ("Job e07ce444 — Policy No in LC vs Cover Note No in doc",
     "11/0000118/1024/0-0",
     "COVER NOTE NO. 11/0000118/1024/0-0 / Century Insurance",
     True),
    ("Job 48bdb6ee — Open Policy with OCR O instead of 0",
     "2023008MIPD000453",
     "OPEN POLICY NO.2023008MIPDO00453",
     True),
    ("Job 48bdb6ee — LC number 0007LC55189/2025",
     "0007LC55189/2025",
     "L/C No:0007LC55189/2025DD;250103",
     True),
    ("BL number YMJAS226041311",
     "YMJAS226041311",
     "B/L NO.\nYMJAS226041311",
     True),
    # Negative: truly different invoice vs policy
    ("Different LC numbers — MUST not match",
     "0007LC55189/2025",
     "9999XX99999/9999",
     False),
    ("Partially similar but different",
     "ABC12345",
     "ABC12346",
     False),
]

r_passed = 0
for label, cond, doc, expected in real_cases:
    got = match(cond, doc)
    ok = 'OK' if got == expected else 'FAIL'
    if ok == 'OK':
        r_passed += 1
    print(f'  [{ok}] {label}')
    print(f'       cond={cond!r}')
    print(f'       doc ={doc[:70]!r}')
    print(f'       match={got}  (expected {expected})')
print()
print(f"{r_passed}/{len(real_cases)} real-world cases correct")

print()
print("=" * 78)
print(f"OVERALL: {passed + r_passed}/{len(cases) + len(real_cases)} scenarios correct")
print("=" * 78)
