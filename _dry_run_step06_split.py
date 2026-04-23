"""Dry-run comprehensive battery for step06 _split_into_clauses."""
import sys
sys.path.insert(0, 'd:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final')
if 'steps.step06_final_lc' in sys.modules:
    del sys.modules['steps.step06_final_lc']
from steps.step06_final_lc import _split_into_clauses

cases = [
    # ── Classic newline-separated ──
    ('Classic newline',
     "1) CLAUSE ONE.\n2) CLAUSE TWO.\n3) CLAUSE THREE.", 3),
    # ── All inline (no newlines) — the main bug ──
    ('All inline',
     "1) CLAUSE ONE TEXT 2) CLAUSE TWO TEXT 3) CLAUSE THREE TEXT", 3),
    # ── Mixed ──
    ('Mixed newline + inline',
     "1) FIRST\n2) SECOND 3) THIRD\n4) FOURTH", 4),
    # ── No space after marker ──
    ('No space after marker',
     "1)FIRST\n2)SECOND\n3)THIRD", 3),
    # ── Period separators ──
    ('Period separators',
     "1. FIRST.\n2. SECOND.\n3. THIRD.", 3),
    # ── 13 clauses (like user's LC) ──
    ('13 clauses — inline',
     "1) A 2) B 3) C 4) D 5) E\n6) F 7) G 8)H 9)I 10)J 11)K 12) L 13) M", 13),
    # ── Single clause ──
    ('Single clause',
     "JUST ONE CLAUSE HERE WITH SOMETHING ELSE.", 1),
    # ── Numeric IDs must NOT be split ──
    ('NTN number safety',
     "Invoice must mention NTN No. 3075811-4 and HS 1511.9030.", 1),
    ('Date safety',
     "DATED 07-01-2025 AND EXPIRING 12-03-2025.", 1),
    ('Phone number safety',
     "FAX NO. 0092-52-4580312 AND TEL 4400000.", 1),
    # ── Ordinal numbers inside text (must NOT split) ──
    ('Ordinals safety',
     "WITHIN 4 WORKING DAYS. Beneficiary 2 copies required.", 1),
    # ── Inline after sentence end ──
    ('Period + number',
     "FINAL. 2) SECOND CLAUSE. 3) THIRD.", 3),
    # ── Real-world: numbered marker after NTN number ──
    ('NTN + inline clause',
     "...NTN No. 3075811-4 2) FULL SET OF BLS 3) CERTIFICATE", 3),
    # ── Clauses with sub-items (I, II, III) must stay together ──
    ('Clauses with roman sub-items',
     "1) CERT (I) COVERED UNDER ICC (II) OWNED BY CO\n2) NEXT CLAUSE", 2),
    # ── Double-digit numbers ──
    ('Double-digit 10+11',
     "10) TENTH CLAUSE 11) ELEVENTH 12) TWELFTH", 3),
    # ── Empty / trivial ──
    ('Empty', "", 0),
    ('Whitespace only', "   \n   \n  ", 0),
    # ── Letter split ──
    ('Letter split',
     "A. APPLE.\nB. BANANA.\nC. CHERRY.", 3),
]

passed, failed = 0, 0
for label, text, expected in cases:
    tag = '46A' if '46A' in label else '47A'
    clauses = _split_into_clauses(tag, text)
    got = len(clauses)
    ok = got == expected
    if ok:
        passed += 1
        print(f'  [OK] {label}: got={got}, expected={expected}')
    else:
        failed += 1
        print(f'  [FAIL] {label}: got={got}, expected={expected}')
        for c in clauses:
            print(f'       {c.clause_id}: {c.text[:80]!r}')
print()
print(f'TOTAL: {passed}/{len(cases)} passed, {failed} failed')
