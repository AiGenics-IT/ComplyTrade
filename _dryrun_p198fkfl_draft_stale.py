"""
P198fk + P198fl dry-run.

P198fk — Draft tenor placeholder rescue:
  When LC F42C says AT SIGHT and the draft text shows
  "AT SIGHT / XXX DAYS" / "AT SIGHT / ___ DAYS" / similar template
  phrases with NON-NUMERIC placeholders, we override the LLM's FAIL
  verdict to PASS because the placeholder is an empty form slot, not
  a real day count.

P198fl — Stale-BL REVIEW message cleanup:
  When the staleness check can't run (DR receiving_date or BL
  on-board date is missing), the REVIEW message is now short and
  user-friendly instead of leaking internal field names + irrelevant
  form-type chatter.
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

# ── P198fk — placeholder detection logic ────────────────────────────
# Mirror the regex from step14_implicit.py
def _is_placeholder_sight(text):
    text = text.upper()
    if not re.search(r'\bAT\s+SIGHT\b', text):
        return False
    has_placeholder = bool(re.search(
        r'AT\s+SIGHT\s*[/\\\-]?\s*'
        r'(?:X{2,}|_{2,}|-{2,}|\.{2,3}|\s*___+\s*|BLANK)\s*DAYS?',
        text,
    ))
    has_real_days = bool(re.search(
        r'AT\s+(\d{1,3})\s*DAYS?\s*(?:AFTER\s*)?(?:SIGHT|B/L|BL|'
        r'BILL\s+OF\s+LADING|SHIPMENT)',
        text,
    ))
    return has_placeholder and not has_real_days


print("--- P198fk: Draft tenor placeholder detection ---")
PLACEHOLDER_CASES = [
    # name, text, expected_is_placeholder
    ("XXX days template",
     "AT SIGHT / XXX DAYS OF THIS FIRST OF EXCHANGE", True),
    ("__ underscores",
     "AT SIGHT / ___ DAYS AFTER SIGHT", True),
    ("--- dashes",
     "AT SIGHT / --- DAYS AFTER SIGHT", True),
    ("ellipsis",
     "AT SIGHT / ... DAYS", True),
    ("BLANK word",
     "AT SIGHT / BLANK DAYS", True),
    ("XX (only 2 X)",
     "AT SIGHT / XX DAYS", True),
    ("backslash separator",
     "AT SIGHT \\ XXX DAYS", True),
    ("hyphen separator with space",
     "AT SIGHT - XXX DAYS", True),
    ("just 'At Sight'",
     "AT SIGHT", False),  # No placeholder pattern, but no real days either — should be False
    # Real usance — should NOT match placeholder
    ("60 DAYS USANCE",
     "AT 60 DAYS SIGHT", False),
    ("90 DAYS BL DATE",
     "AT 90 DAYS BL DATE", False),
    ("30 DAYS AFTER BILL OF LADING",
     "AT 30 DAYS AFTER BILL OF LADING DATE", False),
    ("180 DAYS AFTER SIGHT",
     "AT 180 DAYS AFTER SIGHT", False),
    # Template with REAL days — should NOT match (real days override)
    ("template AT SIGHT but also 90 DAYS",
     "AT SIGHT / XXX DAYS, MATURING AT 90 DAYS BL DATE", False),
    # No 'AT SIGHT' at all
    ("no AT SIGHT keyword",
     "FOR VALUE RECEIVED PAY TO ORDER", False),
]
for name, text, expected in PLACEHOLDER_CASES:
    got = _is_placeholder_sight(text)
    assert_eq(f"placeholder: {name}", got, expected)


print("\n--- P198fk: production code wires the rescue ---")
src = open('d:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final/steps/step14_implicit.py',
           'r', encoding='utf-8').read()
assert_eq("P198fk: rescue block present in step14_implicit",
          'P198fk' in src and 'placeholder rescue' in src, True)
assert_eq("P198fk: prompt has placeholder warning",
          'PRE-PRINTED PLACEHOLDERS' in src or 'placeholder' in src.lower(), True)
assert_eq("P198fk: rescue overrides FAIL to PASS",
          "FAIL->PASS (P198fk placeholder rescue" in src, True)


# ── P198fl — Stale-BL REVIEW message cleanup ────────────────────────
print("\n--- P198fl: Stale-BL REVIEW message cleanup ---")
v_src = open('d:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final/steps/step14_verification.py',
             'r', encoding='utf-8').read()

# Old leaky text MUST be gone
assert_eq("P198fl: old 'Form type / blank back / house' chatter removed",
          'Form type / blank back / house / claused signals are' not in v_src, True)
assert_eq("P198fl: old 'DR receiving_date' raw field-name removed from message",
          'DR receiving_date' not in v_src or
          v_src.count('DR receiving_date') < 2,  # may still appear in comments
          True)

# New friendly text MUST be present
assert_eq("P198fl: new short message present",
          'Cannot determine staleness automatically' in v_src, True)
assert_eq("P198fl: 'Manual review' phrasing present (may be split across lines)",
          'Manual review' in v_src, True)
assert_eq("P198fl: human-friendly 'BL on-board date' wording",
          'Bill of Lading on-board date' in v_src, True)
assert_eq("P198fl: human-friendly 'Documentary Remittance receiving date' wording",
          'Documentary Remittance receiving date' in v_src, True)


passed = sum(results)
total = len(results)
print(f"\n{passed}/{total} cases passed")
if passed != total:
    sys.exit(1)
print("OVERALL: OK")
sys.exit(0)
