"""
P198dh dry-run — accept slash form 'Page X/Y' in addition to
'Page X of Y'.

Job e65ba874 had pages 12-15 and 18-21 each footed with
'Page 1/4', 'Page 2/4', 'Page 3/4', 'Page 4/4'. The previous
regex `Page\\s+(\\d+)\\s+of\\s+(\\d+)` only matched the 'of' form,
so the multi-page Survey Reports broke at page 4 — page 15 (and
21) was getting classified as a separate Tanker Cleanliness
Certificate even though it was page 4 of the same survey set.

The new regex `Page\\s+(\\d+)\\s*(?:of|/)\\s*(\\d+)` accepts:
  • Page 1 of 4
  • Page 1of4
  • Page 1/4
  • Page 1 / 4
  • PAGE 1 OF 4 (case-insensitive)
"""
import re, sys


PAGE_XY_RE = re.compile(r'Page\s+(\d+)\s*(?:of|/)\s*(\d+)', re.IGNORECASE)


SC = [
    # User-reported real form
    ('Page 1/4', (1, 4)),
    ('Page 2/4', (2, 4)),
    ('Page 3/4', (3, 4)),
    ('Page 4/4', (4, 4)),
    ('Page 1 / 4', (1, 4)),
    ('Page  1  /  4 ', (1, 4)),
    ('PAGE 1/4', (1, 4)),
    ('page 1/4', (1, 4)),

    # Existing 'of' form still works
    ('Page 1 of 4', (1, 4)),
    ('Page 1 OF 4', (1, 4)),
    ('PAGE 1 OF 4', (1, 4)),
    ('page 2 of 5', (2, 5)),

    # Mixed form on same line
    ('Surveyor report — Page 3/4 — Singapore', (3, 4)),
    ('See footer: Page 2 of 7. End.', (2, 7)),

    # No spaces around 'of' — does NOT match (intentional, ambiguous)
    ('Page1of4', None),

    # No 'of' / '/' — should NOT match
    ('Page 1', None),
    ('1/4', None),

    # Multi-digit
    ('Page 12 of 15', (12, 15)),
    ('Page 12/15', (12, 15)),
    ('Page 100 / 200', (100, 200)),
]


def main():
    p = f = 0
    for i, (text, expected) in enumerate(SC, 1):
        m = PAGE_XY_RE.search(text)
        got = (int(m.group(1)), int(m.group(2))) if m else None
        ok = got == expected
        tag = 'OK ' if ok else 'FAIL'
        print(f"[{tag}] #{i:02d}  {text!r:<40}  expect={expected}  got={got}")
        if ok: p += 1
        else: f += 1
    print(f"\n{'='*78}\n{p}/{p+f} P198dh page-xy scenarios OK\n{'='*78}")
    return f == 0


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
