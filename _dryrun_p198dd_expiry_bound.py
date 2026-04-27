"""
P198dd dry-run — F48 'BUT WITHIN EXPIRY' relaxation + removal of
the auto cross-link from late-presentation to LC-expired.

Issues fixed:

  1. The auto cross-link in step14_implicit was creating a second
     lc_expiry FAIL row reading "LC EXPIRED" purely because the
     21-day rule was breached. That produced false FAILs even
     when the actual presentation date was BEFORE the actual LC
     expiry date.

  2. F48 phrasing like "15/FROM SHIPMENT DATE BUT WITHIN EXPIRY"
     means the period is bounded by LC expiry — the X-day count
     is a soft target, not a hard limit. Presentations on or
     before LC expiry are PASS even when they exceed the X days.

  3. The condition / result strings used "{period_days:03d}"
     which produced "021 days" / "015 days" — fixed to plain
     "{period_days}" so the user sees "21 days" / "15 days".
"""
import re
import sys
from datetime import date


def parse_iso_date(s):
    if not s: return None
    m = re.search(r'(\d{4})[-./](\d{1,2})[-./](\d{1,2})', s)
    if not m: return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except Exception:
        return None


def evaluate(shipment, presentation, f48, lc_expiry):
    """Mirror of the relaxed presentation_period check."""
    ship_d = parse_iso_date(shipment)
    pres_d = parse_iso_date(presentation)
    exp_d = parse_iso_date(lc_expiry)
    if ship_d is None or pres_d is None:
        return ('REVIEW', 'date parse failed')
    period_days = 21
    pd = re.search(r'\b(\d{1,3})\s*DAYS?\b', (f48 or '').upper())
    if pd:
        try:
            period_days = int(pd.group(1))
        except ValueError:
            pass
    days_elapsed = (pres_d - ship_d).days
    within = 0 <= days_elapsed <= period_days
    expiry_bound = bool(re.search(
        r'BUT\s+WITHIN\s+EXPIRY|WITHIN\s+(?:LC\s+|L/?C\s+)?EXPIRY|'
        r'WITHIN\s+(?:LC\s+|L/?C\s+)?VALIDITY|'
        r'OR\s+(?:LC\s+|L/?C\s+)?EXPIRY|'
        r'AS\s+PER\s+(?:LC\s+|L/?C\s+)?(?:VALIDITY|EXPIRY)',
        (f48 or '').upper(),
    ))
    if (not within) and expiry_bound and exp_d and pres_d <= exp_d:
        return ('PASS',
                f'Presented {days_elapsed}d > {period_days}-day soft target '
                f'but ≤ LC expiry {exp_d} (F48 expiry-bound)')
    return ('PASS' if within else 'FAIL',
            f'days_elapsed={days_elapsed} period_days={period_days} '
            f'expiry_bound={expiry_bound}')


SC = []

# Real job 436a3369 case
SC.append(dict(
    name='Real 436a3369: 23d > 15d but BEFORE LC expiry → PASS via expiry-bound',
    shipment='2025-02-01', presentation='2025-02-24',
    f48='15/FROM SHIPMENT DATE BUT WITHIN EXPIRY',
    expiry='2025-02-28', expect='PASS',
))

# Within the X-day count → PASS regardless
SC.append(dict(
    name='Within 21 days, no expiry qualifier → PASS',
    shipment='2025-01-01', presentation='2025-01-15',
    f48='21 DAYS FROM SHIPMENT', expiry='2025-03-01', expect='PASS',
))

# Exceeds X-day, no expiry qualifier → FAIL even if before expiry
SC.append(dict(
    name='Exceeds 21d, NO "WITHIN EXPIRY" qualifier → FAIL',
    shipment='2025-01-01', presentation='2025-01-25',
    f48='21 DAYS FROM SHIPMENT', expiry='2025-03-01', expect='FAIL',
))

# Exceeds X-day, expiry qualifier, presentation > expiry → FAIL
SC.append(dict(
    name='Expiry-bound but presentation AFTER expiry → FAIL',
    shipment='2025-01-01', presentation='2025-03-05',
    f48='15 DAYS BUT WITHIN EXPIRY', expiry='2025-03-01', expect='FAIL',
))

# "WITHIN VALIDITY" qualifier
SC.append(dict(
    name='Phrasing "WITHIN LC VALIDITY" → expiry-bound PASS',
    shipment='2025-01-01', presentation='2025-01-25',
    f48='15 DAYS WITHIN LC VALIDITY', expiry='2025-02-15', expect='PASS',
))

# "AS PER LC VALIDITY"
SC.append(dict(
    name='Phrasing "AS PER LC VALIDITY" → expiry-bound PASS',
    shipment='2025-01-01', presentation='2025-01-25',
    f48='15 DAYS AS PER LC VALIDITY', expiry='2025-02-15', expect='PASS',
))

# Presentation BEFORE shipment (negative days) → FAIL (defensive)
SC.append(dict(
    name='Presentation before shipment → FAIL',
    shipment='2025-02-15', presentation='2025-02-10',
    f48='21 DAYS', expiry='2025-03-15', expect='FAIL',
))

# F48 missing, default 21
SC.append(dict(
    name='No F48 (default 21d), exceeds → FAIL',
    shipment='2025-01-01', presentation='2025-01-25',
    f48='', expiry='2025-03-01', expect='FAIL',
))

# F48 missing, default 21, within → PASS
SC.append(dict(
    name='No F48 (default 21d), within → PASS',
    shipment='2025-01-01', presentation='2025-01-15',
    f48='', expiry='2025-03-01', expect='PASS',
))


def main():
    p = f = 0
    for i, sc in enumerate(SC, 1):
        v, r = evaluate(sc['shipment'], sc['presentation'], sc['f48'], sc['expiry'])
        ok = (v == sc['expect'])
        tag = 'OK ' if ok else 'FAIL'
        print(f"[{tag}] #{i:02d}  {sc['name']}")
        print(f"         expect={sc['expect']}, got={v}  ({r})")
        if ok: p += 1
        else: f += 1
    print(f"\n{'='*78}\n{p}/{p+f} P198dd scenarios OK\n{'='*78}")
    return f == 0


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
