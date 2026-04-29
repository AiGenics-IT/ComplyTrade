"""P198cf dry-run — presentation_period as hybrid check. Uses actual job
7b5db83e data plus synthetic scenarios covering every mode:

  • No stamp, no labeled date, unlabelled bare "DATE: X" present → REVIEW
  • Textual "presented within validity" assertion present → PASS
  • Labeled "Presentation Date: X" → use it (PASS/FAIL by days calc)
  • RECEIVED stamp date → use it (PASS/FAIL by days calc)
  • Actual job 7b5db83e: Documentary Remittance with only issue date
    → should REVIEW (NOT FAIL)
"""
import json
import re
import sys
from datetime import datetime

sys.path.insert(0, '.')
from steps.step14_implicit import _extract_received_stamp_date, _parse_date


_presentation_ok_patterns = (
    r'DOCUMENTS?\s+(?:HAVE\s+BEEN\s+)?PRESENTED\s+(?:ON\s+TIME|'
    r'WITHIN\s+(?:THE\s+)?(?:L/?C\s+)?(?:VALIDITY|EXPIRY|PERIOD)|'
    r'PRIOR\s+TO\s+EXPIRY|BEFORE\s+(?:THE\s+)?EXPIRY|'
    r'WITHIN\s+(?:THE\s+)?PRESENTATION\s+PERIOD)',
    r'PRESENTATION\s+(?:IS\s+|WAS\s+)?MADE\s+WITHIN\s+(?:L/?C\s+)?'
    r'(?:VALIDITY|EXPIRY\s+DATE)',
    r'DOCUMENTS?\s+ARE\s+(?:BEING\s+)?PRESENTED\s+WITHIN\s+VALIDITY',
    r'WITHIN\s+L/?C\s+VALIDITY\s+PERIOD',
    r'DOCUMENTS?\s+NEGOTIATED\s+WITHIN\s+(?:THE\s+)?VALIDITY',
)


def simulate_presentation_period(cover_pkt, shipment_date_iso, period_days=21):
    """Mirror the P198cf hybrid branch."""
    doc_text_up = (cover_pkt.get('refined_text') or cover_pkt.get('cleaned_text')
                   or cover_pkt.get('raw_text') or '').upper()
    # Textual assertion
    if any(re.search(p, doc_text_up, re.IGNORECASE) for p in _presentation_ok_patterns):
        return 'PASS', 'textual assertion'
    # Labeled / stamp date
    stamp = _extract_received_stamp_date(cover_pkt)
    if stamp is None:
        return 'REVIEW', 'no stamp / no labeled date'
    pres_date, raw = stamp
    ship_date = datetime.fromisoformat(shipment_date_iso)
    days_elapsed = (pres_date - ship_date).days
    within = 0 <= days_elapsed <= period_days
    if within:
        return 'PASS', f'{days_elapsed}d ≤ {period_days} (pres={pres_date:%Y-%m-%d} ship={ship_date:%Y-%m-%d})'
    return 'FAIL', f'{days_elapsed}d > {period_days} (pres={pres_date:%Y-%m-%d})'


# ──────────────────────────────────────────────────────────────
# Actual 7b5db83e remittance
# ──────────────────────────────────────────────────────────────
try:
    with open('results/7b5db83e-c441-4531-b3a4-9c5523e68e34/step09/step09_result.json',
              encoding='utf-8') as f:
        s9 = json.load(f)
    remittance_pkt = None
    for p in s9.get('reconciled_packets', []):
        dt = (p.get('document_type', '') or '').lower()
        if 'remittance' in dt:
            remittance_pkt = p
            break
    if remittance_pkt:
        print("=" * 78)
        print("Actual 7b5db83e Documentary Remittance")
        print("=" * 78)
        txt = (remittance_pkt.get('refined_text') or remittance_pkt.get('cleaned_text')
               or remittance_pkt.get('raw_text') or '')
        print(f"text len: {len(txt)}")
        stamps = remittance_pkt.get('stamps') or []
        print(f"stamps: {stamps}")
        print(f"document_date: {remittance_pkt.get('document_date')!r}")
        print(f"issue_date in summary: {(remittance_pkt.get('unified_summary') or {}).get('issue_date')!r}")
        # Verdict with shipment date 2026-02-23 (per user: "ON 23RD FEBRUARY 2026")
        verdict, reason = simulate_presentation_period(
            remittance_pkt, '2026-02-23', period_days=21)
        print(f"\n[Hybrid verdict] → {verdict}  reason: {reason}")
        print(f"  Expected: REVIEW (only issue/sending date 16MAR26; no stamp; no labeled presentation date)")
        ok_real = (verdict == 'REVIEW')
except Exception as e:
    print(f"[warn] could not load 7b5db83e: {e}")
    ok_real = None


# ──────────────────────────────────────────────────────────────
# Synthetic cases
# ──────────────────────────────────────────────────────────────
def _pkt(text='', stamps=None, dt='Documentary Remittance'):
    return {
        'document_type': dt,
        'refined_text': text,
        'cleaned_text': text,
        'raw_text': text,
        'stamps': stamps or [],
    }


print()
print("=" * 78)
print("Synthetic scenarios")
print("=" * 78)
cases = [
    ('Bare "DATE: 16MAR26" only — the real-job case',
     _pkt(text='Covering Schedule\nDATE: 16MAR26\nAmount USD 250,000'),
     '2026-02-23', 21, 'REVIEW'),
    ('Stamp 10 MAR 2026 + shipment 23 FEB 2026 (15 days)',
     _pkt(stamps=[{'text': '10 MAR 2026', 'type': 'rubber_stamp'}]),
     '2026-02-23', 21, 'PASS'),
    ('Stamp 20 MAR 2026 + shipment 23 FEB 2026 (25 days > 21) → FAIL',
     _pkt(stamps=[{'text': '20 MAR 2026', 'type': 'rubber_stamp'}]),
     '2026-02-23', 21, 'FAIL'),
    ('Labeled "Presentation Date: 12/03/2026" (17 days, ≤21) → PASS',
     _pkt(text='Presentation Date: 12/03/2026\nAmount USD 250,000'),
     '2026-02-23', 21, 'PASS'),
    ('Labeled "Received Date: 25.03.2026" (30 days > 21) → FAIL',
     _pkt(text='Received Date: 25.03.2026\nAmount USD 250,000'),
     '2026-02-23', 21, 'FAIL'),
    ('Textual assertion "documents presented within LC validity"',
     _pkt(text=('Covering Schedule\n'
                'We hereby certify documents have been presented within LC validity period.\n'
                'DATE: 16MAR26')),
     '2026-02-23', 21, 'PASS'),
    ('Custom period 30 days from F48, presentation 25 days after → PASS',
     _pkt(stamps=[{'text': '20 MAR 2026', 'type': 'rubber_stamp'}]),
     '2026-02-23', 30, 'PASS'),
    ('OCR-mangled stamp "71 CCR 2025" + bare date → REVIEW',
     _pkt(
         stamps=[{'text': '71 CCR 2025', 'type': 'rubber_stamp'}],
         text='DATE: 16MAR26',
     ),
     '2026-02-23', 21, 'REVIEW'),
    ('No stamp, no assertion, no labeled date, no bare date → REVIEW',
     _pkt(text='Covering Schedule\nAmount USD 250,000\nLC No: XYZ'),
     '2026-02-23', 21, 'REVIEW'),

    # ── Edge: exactly at the boundary ──
    ('Exactly on day 21 (boundary) → PASS',
     _pkt(stamps=[{'text': '16 MAR 2026', 'type': 'rubber_stamp'}]),
     '2026-02-23', 21, 'PASS'),  # 21 days exact
    ('One day past (day 22) → FAIL',
     _pkt(stamps=[{'text': '17 MAR 2026', 'type': 'rubber_stamp'}]),
     '2026-02-23', 21, 'FAIL'),
    ('Same day shipment + presentation (day 0) → PASS',
     _pkt(stamps=[{'text': '23 FEB 2026', 'type': 'rubber_stamp'}]),
     '2026-02-23', 21, 'PASS'),

    # ── Edge: presentation BEFORE shipment (impossible / anomaly) ──
    ('Presentation BEFORE shipment — days_elapsed<0 → FAIL',
     _pkt(stamps=[{'text': '20 FEB 2026', 'type': 'rubber_stamp'}]),
     '2026-02-23', 21, 'FAIL'),  # negative days → outside [0, 21]

    # ── Different period_days from F48 ──
    ('F48 period=15, presentation 10 days after → PASS',
     _pkt(stamps=[{'text': '05 MAR 2026', 'type': 'rubber_stamp'}]),
     '2026-02-23', 15, 'PASS'),
    ('F48 period=15, presentation 20 days after → FAIL',
     _pkt(stamps=[{'text': '15 MAR 2026', 'type': 'rubber_stamp'}]),
     '2026-02-23', 15, 'FAIL'),
    ('F48 period=45 (very generous), 40 days → PASS',
     _pkt(stamps=[{'text': '04 APR 2026', 'type': 'rubber_stamp'}]),
     '2026-02-23', 45, 'PASS'),

    # ── Combined: stamp + assertion both present — assertion wins (earlier return) ──
    ('Stamp 30 MAR 2026 (35 days > 21) + assertion text → PASS (assertion wins)',
     _pkt(
         stamps=[{'text': '30 MAR 2026', 'type': 'rubber_stamp'}],
         text='Documents presented within LC validity period',
     ),
     '2026-02-23', 21, 'PASS'),

    # ── Labeled date formats ──
    ('Labeled "Date of Presentation: 10-MAR-2026"',
     _pkt(text='Date of Presentation: 10-MAR-2026\nLC No: XYZ'),
     '2026-02-23', 21, 'PASS'),
    ('Labeled "Received: 10 March 2026"',
     _pkt(text='Received: 10 March 2026\nAmount USD 250,000'),
     '2026-02-23', 21, 'PASS'),
    ('Labeled "Date of Receipt: 15/03/2026"',
     _pkt(text='Date of Receipt: 15/03/2026'),
     '2026-02-23', 21, 'PASS'),
    ('Labeled "Presented on 25.03.2026" (30 days) → FAIL',
     _pkt(text='Presented on 25.03.2026'),
     '2026-02-23', 21, 'FAIL'),

    # ── Different textual assertion phrasings ──
    ('Assertion "documents presented on time"',
     _pkt(text='We hereby confirm that documents presented on time.'),
     '2026-02-23', 21, 'PASS'),
    ('Assertion "presented prior to expiry"',
     _pkt(text='Documents have been presented prior to expiry.'),
     '2026-02-23', 21, 'PASS'),
    ('Assertion "documents negotiated within the validity"',
     _pkt(text='Documents negotiated within the validity.'),
     '2026-02-23', 21, 'PASS'),

    # ── Negative: similar wording but not an assertion ──
    ('"Documents presented at UBL counter" (place, not timing) → REVIEW',
     _pkt(text='Documents presented at UBL counter'),
     '2026-02-23', 21, 'REVIEW'),
    ('"Present documents to opening bank" (instruction) → REVIEW',
     _pkt(text='Please present documents to opening bank'),
     '2026-02-23', 21, 'REVIEW'),

    # ── Stamp has multiple dates (noisy OCR) ──
    ('Stamp contains noise but a parseable date → use first parseable',
     _pkt(stamps=[
         {'text': 'xxx', 'type': 'rubber_stamp'},
         {'text': '10 MAR 2026', 'type': 'rubber_stamp'},
     ]),
     '2026-02-23', 21, 'PASS'),

    # ── Only stamp data, no labeled date, no assertion ──
    ('Stamp date only → use it (no confusion with issue date)',
     _pkt(
         stamps=[{'text': 'RECEIVED 15 MAR 2026', 'type': 'rubber_stamp'}],
         text='DATE: 16MAR26',  # bare date must NOT be used
     ),
     '2026-02-23', 21, 'PASS'),
]

passed = 0
for label, pkt, ship_iso, period, expected in cases:
    v, reason = simulate_presentation_period(pkt, ship_iso, period)
    ok = 'OK' if v == expected else 'FAIL'
    if ok == 'OK':
        passed += 1
    print(f'  [{ok}] {label}')
    print(f'       got={v}  reason={reason}  (expected {expected})')

print()
print("=" * 78)
print(f"Real-job check: {'OK (REVIEW)' if ok_real else ('FAIL' if ok_real is False else 'SKIPPED')}")
print(f"Synthetic: {passed}/{len(cases)} cases correct")
print("=" * 78)
