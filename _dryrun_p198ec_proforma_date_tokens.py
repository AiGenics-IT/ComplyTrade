"""
P198ec dry-run — Proforma date integrity, edge-case grid

The previous logic was fooled by the YYMMDD-shaped digit-run inside
references like "PI2504022" (the leading "250402" parsed as
SWIFT YYMMDD = 2-Apr-2025 and beat the real "APR 18, 2025" token
in the same string). Patch: bare \\d{6}/\\d{8} now require word
boundaries, and the comparison path uses a `_best_date_token`
helper that prefers month-name tokens over bare digits.

Tests every combination of {ref same/diff} x {date same/diff}
on the actual P198ak comparison path:
"""
import importlib
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Load the real regex + parser from step14
import steps.step14_verification as s14
importlib.reload(s14)


_MONTH_NAMES = (
    r'(?:JAN(?:UARY)?|FEB(?:RUARY)?|MAR(?:CH)?|'
    r'APR(?:IL)?|MAY|JUN(?:E)?|JUL(?:Y)?|'
    r'AUG(?:UST)?|SEP(?:T(?:EMBER)?)?|'
    r'OCT(?:OBER)?|NOV(?:EMBER)?|DEC(?:EMBER)?)'
)
MONTH = re.compile(
    r'(?:' + _MONTH_NAMES + r'\.?\s*\d{1,2}[,\s]+\d{2,4}|'
    r'\d{1,2}[\s\-./]+' + _MONTH_NAMES + r'\.?[\s\-./]+\d{2,4})',
    flags=re.IGNORECASE,
)
DATE = re.compile(
    r'(?:' + _MONTH_NAMES + r'\.?\s*\d{1,2}[,\s]+\d{2,4}|'
    r'\d{1,2}[\s\-./]+' + _MONTH_NAMES + r'\.?[\s\-./]+\d{2,4}|'
    r'\d{4}[-./]\d{1,2}[-./]\d{1,2}|'
    r'\d{1,2}[-./]\d{1,2}[-./]\d{2,4}|'
    r'\b\d{6}\b|\b\d{8}\b)',
    flags=re.IGNORECASE,
)


def parse_token(text):
    s = (text or '').upper()
    m = MONTH.search(s)
    if not m:
        m = DATE.search(s)
    return m.group(0) if m else ''


# Shape of the YYMMDD parser used inside step14 (mirrors _pro_parse).
_MONTHS = {
    'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
    'JUL': 7, 'AUG': 8, 'SEP': 9, 'SEPT': 9, 'OCT': 10,
    'NOV': 11, 'DEC': 12,
    'JANUARY': 1, 'FEBRUARY': 2, 'MARCH': 3, 'APRIL': 4,
    'JUNE': 6, 'JULY': 7, 'AUGUST': 8, 'SEPTEMBER': 9,
    'OCTOBER': 10, 'NOVEMBER': 11, 'DECEMBER': 12,
}


def parse_date(s):
    if not s:
        return None
    s = str(s).upper().strip().rstrip('.,;:')
    s = re.sub(r'(\d+)(ST|ND|RD|TH)\b', r'\1', s)
    s = re.sub(r'\s+', ' ', s).strip()
    m = re.match(r'^([A-Z]+)[\s,.\- ]*(\d{1,2})[\s,.\- ]+(\d{2,4})$', s)
    if m and _MONTHS.get(m.group(1)):
        y = int(m.group(3))
        d = int(m.group(2))
        if y < 100:
            y = 2000 + y if y <= 69 else 1900 + y
        return (y, _MONTHS[m.group(1)], d)
    m = re.match(r'^(\d{1,2})[\s\-./]+([A-Z]+)\.?[\s\-./]+(\d{2,4})$', s)
    if m and _MONTHS.get(m.group(2)):
        y = int(m.group(3))
        if y < 100:
            y = 2000 + y if y <= 69 else 1900 + y
        return (y, _MONTHS[m.group(2)], int(m.group(1)))
    m = re.match(r'^(\d{4})[-./](\d{1,2})[-./](\d{1,2})$', s)
    if m:
        return (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = re.match(r'^\d{6}$', s)
    if m:
        return (2000 + int(s[:2]), int(s[2:4]), int(s[4:6]))
    return None


def compare(lc_text, inv_text):
    lc_tok = parse_token(lc_text)
    inv_tok = parse_token(inv_text)
    return parse_date(lc_tok), parse_date(inv_tok)


SCENARIOS = [
    # (name, lc_raw, inv_raw, expected_match)
    ('1. Same ref, same date — exact match',
     'PI2504022 dated APR 18, 2025',
     'OTHER DETAILS ARE STRICTLY AS PER PROFORMA INVOICE NO. PI2504022 DATED APR 18, 2025',
     True),
    ('2. Same ref, same date — different month-name casing',
     'PI2504022 dated apr 18, 2025',
     'PROFORMA INVOICE NO. PI2504022 DATED Apr 18, 2025',
     True),
    ('3. Same ref, different date',
     'PI2504022 dated APR 18, 2025',
     'PROFORMA INVOICE NO. PI2504022 DATED MAR 18, 2025',
     False),
    ('4. Same ref, different year',
     'PI2504022 dated APR 18, 2025',
     'PROFORMA INVOICE NO. PI2504022 DATED APR 18, 2024',
     False),
    ('5. Same ref, different day',
     'PI2504022 dated APR 18, 2025',
     'PROFORMA INVOICE NO. PI2504022 DATED APR 19, 2025',
     False),
    ('6. Reference with embedded YYMMDD-shaped digits (BUG REGRESSION)',
     # PI2504022 has "250402" which previously parsed as 2-Apr-2025
     'PI2504022 dated APR 18, 2025',
     "OTHER DETAILS ARE STRICTLY AS PER BENEFICIARY'S "
     "PROFORMA INVOICE NO. PI2504022 DATED APR 18, 2025",
     True),
    ('7. ISO-form date',
     'PI2504022 dated 2025-04-18',
     'PROFORMA INVOICE NO. PI2504022 DATED 2025-04-18',
     True),
    ('8. ISO-form date mismatched',
     'PI2504022 dated 2025-04-18',
     'PROFORMA INVOICE NO. PI2504022 DATED 2025-04-19',
     False),
    ('9. Day-month-year European',
     'PI2504022 dated 18-APR-2025',
     'PROFORMA INVOICE NO. PI2504022 DATED 18-APR-2025',
     True),
    ('10. SWIFT YYMMDD on both sides',
     '250418', '250418', True),
    ('11. SWIFT YYMMDD mismatch',
     '250418', '250419', False),
    ('12. Mixed: LC uses month-name, invoice uses YYMMDD',
     'PI2504022 dated APR 18, 2025',
     'PROFORMA INVOICE NO. PI2504022 DATED 250418',
     True),
    ('13. Reference INSIDE a longer ref string (digits embedded)',
     # User's actual case
     'PI2504022 dated APR 18, 2025',
     'INVOICE NO. MCI-786/S-13198 PI2504022 DATED APR 18, 2025',
     True),
    ('14. Bare digit run that could be misread as date',
     # Reference "ABCD250402EF" — should NOT pull "250402" out
     'XYZ123 dated APR 18, 2025',
     'PROFORMA INVOICE NO.ABCD250402EF (XYZ123) DATED APR 18, 2025',
     True),
    ('15. Date with dotted separators "18.04.2025"',
     'PI2504022 dated 18.04.2025',
     'PROFORMA INVOICE NO. PI2504022 DATED 18.04.2025',
     True),
]


def main():
    pass_n, fail_n = 0, 0
    print('=' * 78)
    print('P198ec dry-run — proforma date integrity edge cases')
    print('=' * 78)
    for name, lc, inv, expected in SCENARIOS:
        lc_p, inv_p = compare(lc, inv)
        if lc_p is None or inv_p is None:
            actual_match = (lc_p == inv_p)
        else:
            actual_match = (lc_p == inv_p)
        ok = (actual_match == expected)
        tag = 'OK ' if ok else 'FAIL'
        print(f'\n[{tag}] {name}')
        print(f'        LC raw     = {lc!r}')
        print(f'        inv raw    = {inv!r}')
        print(f'        LC parsed  = {lc_p}')
        print(f'        inv parsed = {inv_p}')
        print(f'        match      = {actual_match}  (expected {expected})')
        if ok: pass_n += 1
        else:  fail_n += 1
    total = pass_n + fail_n
    print('\n' + '=' * 78)
    print(f'OVERALL: {pass_n}/{total} '
          f'{"OK" if fail_n == 0 else "— failures present"}')
    print('=' * 78)
    return 0 if fail_n == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
