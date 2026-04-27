"""
P198de dry-run — F48 day-count parser.

The hybrid presentation_period check was using a regex that only
matched "<N> DAYS", missing BAHL's slash notation
"15/FRM SHIPMENT DATE BUT WITH IN EXPIRY". Result: every BAHL
job silently fell back to the UCP 600 default of 21 days even
when F48 explicitly said 15.

The fix accepts the classic "<N> DAYS" form, the BAHL slash form,
and a leading bare number followed by FROM/FRM/DAYS/SHIPMENT-
token.
"""
import re, sys


def parse_period(f48):
    f48_u = (f48 or '').upper()
    period_days = 21
    pd = (
        re.search(r'\b(\d{1,3})\s*DAYS?\b', f48_u)
        or re.search(r'\b(\d{1,3})\s*/\s*(?:FROM|FRM)\b', f48_u)
        or re.search(
            r'\b(\d{1,3})\s*(?:FROM|FRM)\s+'
            r'(?:SHIPMENT|B/?L|BL\b|NEGOTIATION|PRESENTATION)',
            f48_u,
        )
        or re.search(
            r'^\s*(\d{1,3})\s*[/\-:]?\s*(?:FROM|FRM|DAYS|DAY)\b',
            f48_u,
        )
        or re.search(r'^\s*(\d{1,3})\b', f48_u)
    )
    if pd:
        try:
            return int(pd.group(1))
        except ValueError:
            return period_days
    return period_days


SC = [
    # Real BAHL forms
    ('15/FROM SHIPMENT DATE BUT WITHIN EXPIRY', 15),
    ('15/FRM SHIPMENT DATE BUT WITH IN EXPIRY', 15),
    ('21/FROM SHIPMENT DATE BUT WITHIN EXPIRY', 21),
    ('21/FRM SHIPMENT DATE BUT WITH IN EXPIRY', 21),
    ('21 / FRM SHIPMENT DATE BUT WITH IN EXPIRY', 21),

    # Classic UCP wording
    ('21 DAYS FROM SHIPMENT DATE', 21),
    ('15 DAYS BUT WITHIN EXPIRY', 15),
    ('within 30 days of shipment', 30),
    ('Documents presented within 21 days', 21),

    # Mixed
    ('15 FRM B/L DATE', 15),
    ('15/FRM B/L DATE', 15),
    ('15 FROM SHIPMENT', 15),
    ('21/PRESENTATION FROM SHIPMENT', 21),

    # Empty / missing → default 21
    ('', 21),
    (None, 21),
    ('Not specified', 21),

    # Unusual numbers
    ('7 DAYS', 7),
    ('45 / FROM SHIPMENT', 45),
    ('120 DAYS', 120),
    ('120/FRM SHIPMENT', 120),
]


def main():
    p = f = 0
    for i, (f48, expected) in enumerate(SC, 1):
        got = parse_period(f48)
        ok = got == expected
        tag = 'OK ' if ok else 'FAIL'
        print(f"[{tag}] #{i:02d}  F48={f48!r}  expect={expected}  got={got}")
        if ok: p += 1
        else: f += 1
    print(f"\n{'='*78}\n{p}/{p+f} P198de F48-parser scenarios OK\n{'='*78}")
    return f == 0


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
