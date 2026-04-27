"""
P198dk dry-run — F47A SWIFT-advice rescue must

  1) Load MT799/MT999 packets from step03 + step02 cleaned_text when
     they were never routed to step09 (older jobs).
  2) Inspect the MT799/MT999 content against F47A-14 required fields.
     PASS only when shipment evidence is present (vessel, voyage, IMO,
     BL no., container, seal, ports, negotiation amount). Fail with a
     content-mismatch finding when the message reads as an issuance-
     time advising-bank notification.

Tests use job 08345848-0e35-4c02-9f26-7287f60b2028's real OCR (pages 1
and 2 — both CIBC issuance-time MT799s) plus a synthetic post-
negotiation MT799 to verify the PASS branch.
"""
import importlib.util
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')


# Load the helpers from step14_verification by re-creating them
# in isolation (avoids importing the whole module's heavy deps).
import re

_ISSUANCE = (
    r'WE\s+HAVE\s+(?:TODAY\s+)?ADVISED\s+THE\s+(?:LC|L/?C|CREDIT)',
    r'(?:WE\s+HAVE\s+)?ADVISED\s+THE\s+(?:ABOVE\s+)?'
    r'(?:MENTIONED\s+)?(?:LETTERS?\s+OF\s+CREDIT|LCS?|L/?CS?)',
    r'WITHOUT\s+ADDING\s+OUR\s+CONFIRMATION',
    r'KINDLY\s+UPDATE\s+YOUR\s+RECORDS',
    r'KINDLY\s+MARK\s+YOUR\s+RECORDS',
)
_EVID = {
    'vessel': r'\bVESSEL\b|\bM\.?\s*V\.?\s+[A-Z]|\bMV\s+[A-Z]',
    'voyage': r'\bVOYAGE\b',
    'IMO': r'\bIMO\b',
    'BL number': r'\bB\s*/\s*L\b|\bBILL\s+OF\s+LADING\b',
    'container': r'\bCONTAINER\b',
    'seal': r'\bSEAL\b',
    'port of loading': r'\bPORT\s+OF\s+LOADING\b',
    'port of discharge': r'\bPORT\s+OF\s+DISCHARGE\b',
    'negotiation amount': (r'\bAMOUNT\s+OF\s+NEGOTIATION\b|'
                           r'\bNEGOTIATION\s+AMOUNT\b|'
                           r'\bNEGOTIATED\s+AMOUNT\b'),
}


def evaluate(pkt):
    txt = pkt.get('document_text') or ''
    if not txt:
        return False, 'no readable text in MT799/MT999'
    u = txt.upper()
    for pat in _ISSUANCE:
        if re.search(pat, u):
            return False, ('MT799/MT999 reads as an LC-issuance / '
                           'advising-bank notification, not a '
                           'post-negotiation advice')
    found = [k for k, r in _EVID.items() if re.search(r, u)]
    if len(found) >= 2:
        return True, 'contains negotiation evidence: ' + ', '.join(found)
    return False, ('MT799/MT999 lacks the negotiation-time fields '
                   'required by F47A — found '
                   + (', '.join(found) if found else 'none')
                   + ' of {vessel, voyage, IMO, BL number, container, '
                   'seal, ports, negotiation amount}')


def load_mt_side(out_dir):
    """Mirror of _load_mt_side_swift_advice."""
    job = Path(out_dir).parent
    s3p = job / 'step03' / 'step03_result.json'
    s2p = job / 'step02' / 'step02_result.json'
    if not s3p.exists():
        return []
    s3 = json.loads(s3p.read_text(encoding='utf-8'))
    pt = {}
    if s2p.exists():
        s2 = json.loads(s2p.read_text(encoding='utf-8'))
        for pp in s2.get('pages', []):
            pn = pp.get('page_number')
            if pn is not None:
                pt[pn] = (pp.get('cleaned_text')
                          or pp.get('raw_text') or '')
    out = []
    for pk in s3.get('packets', []):
        dtu = (pk.get('document_type') or '').upper()
        if not any(k in dtu for k in ('MT799', 'MT 799',
                                       'MT999', 'MT 999',
                                       'FIN.799', 'FIN.999')):
            continue
        pages = (pk.get('page_numbers')
                 or [p.get('page_number')
                     for p in (pk.get('pages') or [])])
        txt = '\n\n'.join(pt.get(pn, '') for pn in pages
                          if pn is not None)
        out.append({
            'packet_id': pk.get('packet_id'),
            'document_type': pk.get('document_type'),
            'page_numbers': pages,
            'document_text': txt,
        })
    return out


# ─── Test data ─────────────────────────────────────────────────────

JOB_DIR = Path('results/08345848-0e35-4c02-9f26-7287f60b2028/step14')

POST_NEGOTIATION_MT799 = {
    'document_type': 'MT799',
    'page_numbers': [99],
    'document_text': """Message Details
Sender: HABCANTOXXX  Receiver: BAHLPKKACPU
Format: Swift   Identifier: fin.799

F20: NEG-LC-1089LC59947
F79: Narrative
ATTN: TRADE FINANCE DEPT.
WE HAVE TODAY NEGOTIATED THE ABOVE LC FOR USD 490,200.00.
AMOUNT OF NEGOTIATION: USD 490,200.00
BILL OF LADING NO: TLI-SOY-0014E DATED 14-APR-2026
NAME OF CARRIER: THE LOGISTICS INTERNATIONAL
NAMES OF THE VESSEL: MV ATLANTIC SPIRIT
IMO NUMBER: 9876543
VOYAGE NUMBER: 23W
DATE OF SHIPMENT: 14-APR-2026
PORT OF LOADING: SANTOS, BRAZIL
PORT OF DISCHARGE: KARACHI, PAKISTAN
CONTAINER NUMBER: ABCU1234567
SEAL NUMBER: SL00098765
COURIER COMPANY: DHL  RECEIPT NO: DHL-789456
DATE OF DISPATCH: 16-APR-2026
""",
}

EMPTY_MT799 = {
    'document_type': 'MT799',
    'page_numbers': [50],
    'document_text': '',
}


def main():
    print('=' * 78)
    print('P198dk dry-run — F47A SWIFT-advice content check')
    print('=' * 78)

    # ── Test 1: Load MT-side packets from job 08345848 ──
    print('\n[Test 1] Load step03 MT799 packets for job 08345848 '
          '(real OCR via step02)')
    mt_pkts = load_mt_side(str(JOB_DIR))
    print(f'  found {len(mt_pkts)} MT799/MT999 packet(s)')
    for p in mt_pkts:
        print(f"    {p['packet_id']}: {p['document_type']} "
              f"pages={p['page_numbers']} text_len="
              f"{len(p['document_text'])}")
    t1_ok = len(mt_pkts) == 2 and all(
        len(p['document_text']) > 50 for p in mt_pkts)
    print(f"  RESULT: {'OK' if t1_ok else 'FAIL'}")

    # ── Test 2: Both packets must be identified as issuance-time ──
    print('\n[Test 2] Each real MT799 must FAIL content-check '
          '(issuance-time)')
    t2_ok = True
    for p in mt_pkts:
        ok, why = evaluate(p)
        print(f"  pkt {p['packet_id']} (page {p['page_numbers']}): "
              f"valid={ok}  reason={why[:90]}")
        if ok:
            t2_ok = False
    print(f"  RESULT: {'OK' if t2_ok else 'FAIL'}")

    # ── Test 3: Synthetic post-negotiation MT799 must PASS ──
    print('\n[Test 3] Synthetic post-negotiation MT799 must PASS')
    ok, why = evaluate(POST_NEGOTIATION_MT799)
    print(f"  valid={ok}  reason={why[:120]}")
    t3_ok = ok
    print(f"  RESULT: {'OK' if t3_ok else 'FAIL'}")

    # ── Test 4: Empty-text MT799 must FAIL ──
    print('\n[Test 4] MT799 with empty text must FAIL')
    ok, why = evaluate(EMPTY_MT799)
    print(f"  valid={ok}  reason={why[:90]}")
    t4_ok = (not ok)
    print(f"  RESULT: {'OK' if t4_ok else 'FAIL'}")

    # ── Test 5: Synthetic MT799 with only one evidence field ──
    print('\n[Test 5] MT799 with only ONE evidence field must FAIL')
    only_one = {
        'document_type': 'MT799',
        'document_text': 'F79: AMOUNT OF NEGOTIATION USD 100. Regards.',
    }
    ok, why = evaluate(only_one)
    print(f"  valid={ok}  reason={why[:120]}")
    t5_ok = (not ok)
    print(f"  RESULT: {'OK' if t5_ok else 'FAIL'}")

    # ── Test 6: Mixed MT799 (issuance phrase but with shipment evidence)
    # — issuance markers RULE OUT validity. ──
    print('\n[Test 6] MT799 with issuance phrase + shipment evidence: '
          'issuance phrase wins (FAIL)')
    mixed = {
        'document_type': 'MT799',
        'document_text': ('WE HAVE TODAY ADVISED THE LC. '
                          'BILL OF LADING NO: X. VESSEL: MV TEST.'),
    }
    ok, why = evaluate(mixed)
    print(f"  valid={ok}  reason={why[:120]}")
    t6_ok = (not ok)
    print(f"  RESULT: {'OK' if t6_ok else 'FAIL'}")

    # ── Summary ──
    all_ok = t1_ok and t2_ok and t3_ok and t4_ok and t5_ok and t6_ok
    print('\n' + '=' * 78)
    print(f'OVERALL: {"OK — all 6 scenarios pass" if all_ok else "FAIL"}')
    print('=' * 78)
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
