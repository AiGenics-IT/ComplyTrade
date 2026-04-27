"""
P198df dry-run — F48 display normalisation in the final LC.

The BAHL-style SWIFT export writes F48 as
  "15/FRM SHIPMENT DATE BUT WITH IN EXPIRY"
which is correct semantically but reads awkwardly in the final
LC report. Rewrite it to a clean English form
  "15 days from shipment date but within expiry"
without losing the day-count or the within-expiry intent.

Already-clean wording (e.g. "21 days from shipment date") is
left untouched.
"""
import re, sys


def normalise(f48):
    if not f48:
        return f48
    s = str(f48).strip()
    if not s:
        return s
    out = s
    m = re.match(r'^\s*(\d{1,3})\s*/\s*(FROM|FRM)\s+(.+)$', out, re.IGNORECASE)
    if m:
        out = f"{m.group(1)} days from {m.group(3).strip()}"
    out = re.sub(r'\bFRM\b', 'from', out, flags=re.IGNORECASE)
    out = re.sub(r'\bWITH\s+IN\b', 'within', out, flags=re.IGNORECASE)
    return out.lower() if out != s else s


SC = [
    # BAHL slash form (real cases)
    ('15/FROM SHIPMENT DATE BUT WITHIN EXPIRY',
     '15 days from shipment date but within expiry'),
    ('15/FRM SHIPMENT DATE BUT WITH IN EXPIRY',
     '15 days from shipment date but within expiry'),
    ('21/FROM SHIPMENT DATE BUT WITHIN EXPIRY',
     '21 days from shipment date but within expiry'),
    ('21/FRM SHIPMENT DATE BUT WITH IN EXPIRY',
     '21 days from shipment date but within expiry'),
    ('21 / FRM SHIPMENT DATE BUT WITH IN EXPIRY',
     '21 days from shipment date but within expiry'),

    # Already clean → leave alone
    ('21 DAYS FROM SHIPMENT DATE BUT WITHIN EXPIRY',
     '21 DAYS FROM SHIPMENT DATE BUT WITHIN EXPIRY'),
    ('within 21 days of shipment',
     'within 21 days of shipment'),
    ('Period for Presentation 15 days', 'Period for Presentation 15 days'),

    # Variants
    ('15/FROM B/L DATE BUT WITHIN EXPIRY',
     '15 days from b/l date but within expiry'),
    ('15 FRM SHIPMENT DATE',
     '15 from shipment date'),  # FRM-only abbrev still gets cleaned

    # Edge: empty / None
    ('', ''),
    (None, None),

    # Numbers other than 15/21
    ('30/FROM SHIPMENT DATE BUT WITHIN EXPIRY',
     '30 days from shipment date but within expiry'),
    ('7/FRM SHIPMENT DATE',
     '7 days from shipment date'),
    ('120/FROM SHIPMENT',
     '120 days from shipment'),
]


def main():
    p = f = 0
    for i, (inp, expected) in enumerate(SC, 1):
        got = normalise(inp)
        ok = got == expected
        tag = 'OK ' if ok else 'FAIL'
        print(f"[{tag}] #{i:02d}  in={inp!r}")
        print(f"        expect={expected!r}")
        print(f"        got   ={got!r}")
        if ok: p += 1
        else: f += 1
    print(f"\n{'='*78}\n{p}/{p+f} P198df F48 display scenarios OK\n{'='*78}")
    return f == 0


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
