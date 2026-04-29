"""
P198dv dry-run — F46A clauses that ask for a "shipment advice"
(typically an email from beneficiary to insurance + applicant) must
match pages classified as "Insurance Request" / "Insurance Cover
Request" / "Insurance Pre-Advise Notice" — these are the same
physical document, and the VLM frequently labels them by their
subject line instead of their LC role.

Tests use the user's real F46A-4 clause + the actual email body
from job 38beca01 (BAHL/CBL/IGI), plus 14 synthetic scenarios that
cover other realistic LC-doc combinations and edge cases (so we do
not accidentally match an insurance POLICY clause to a shipment
advice page or vice-versa).
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Import the live alias table from step14 so we exercise the same
# data the running classifier uses.
from steps.step14_verification import DOC_TYPE_ALIASES, _find_matching_docs

# Real text the user pasted (page that step08 labels "Insurance Request")
INS_REQUEST_TEXT = (
    "Sajid Hassan\n"
    "Subject: FW:INSURANCE COVER-OPEN POLICY NO.2024/12/HRCMIMOO00189||\n"
    "DOCUMENTARY CREDIT NUMBER :1001LC83147/2025\n\n"
    "From: Sajid Hassan\n"
    "Sent: Tuesday, 30 December 2025 5:01 PM\n"
    "To: 'FARHAN@HABIBRC.COM'<farhan@habibrc.com>;'ANEES@HABIBRC.COM'\n"
    "Cc: 'Fatima Furqan'<fatimafurqan@cbl.com.pk>\n"
    "Subject: INSURANCE COVER-OPEN POLICY NO.2024/12/HRCMIMOO00189\n\n"
    "TO IGI INSURANCE COMPANY LTD.,\n"
    "AND APPLICANT: CONTINENTAL BISCUITS LIMITED\n\n"
    "We request you to kindly insure goods against OPEN POLICY\n"
    "NO.2024/12/HRCMIMOO00189 as per below details\n\n"
    "DOCUMENTARY CREDIT NUMBER 1001LC83147/2025, DATE 17-10-2025\n"
    "L/C ISSUING BANK-BANK AL HABIB LIMITED, KARACHI\n"
    "INVOICE NUMBER: SL/J1376/2025 DATED.18/12/2025\n"
    "INVOICE VALUE: EUR 26,880.00\n"
    "BILL OF LANDING NO. BLQA66810\n"
    "VESSEL NAME AND VOYAGE NUMBER: GERHARD SCHULTE/550S\n"
    "PORT OF LOADING: LA SPEZIA SEAPORT, ITALY\n"
    "PORT OF DISCHARGE: KARACHI SEAPORT, PAKISTAN\n"
    "SHIPPED ON BOARD DATE: 11/12/2025\n\n"
    "Best regards\nSajid Hassan\nLogistics Manager\n"
)

# What step08/VLM labels these emails as (real-world variations)
INS_REQUEST_LABELS = [
    'Insurance Request',
    'Insurance Cover Request',
    'Insurance Cover Note Request',
    'Insurance Pre-Advise Notice',
    'Insurance Pre-Advice',
    'Insurance Notification',
    'Insurance Advice',
    'Request for Insurance Cover',
    'Shipment Advice',                       # canonical name (control)
    'Beneficiary Shipment Advice',           # alternate canonical
    'Shipping Advice',
]

# Other doc-types that should NOT match "Shipment Advice" target
NEGATIVE_LABELS = [
    'Insurance Policy',
    'Insurance Certificate',
    'Marine Insurance Policy',
    'Cargo Insurance Certificate',
    'Bill of Lading',
    'Commercial Invoice',
    'Packing List',
    'Documentary Remittance',
    'Covering Letter',
    'Beneficiary Certificate',
    'Certificate of Origin',
]


def make_packet(doc_type, text=''):
    return {
        'document_type': doc_type,
        'cleaned_text': text or INS_REQUEST_TEXT,
        'refined_text': text or INS_REQUEST_TEXT,
        'packet_id': f'pkt_{doc_type.replace(" ", "_").lower()}',
    }


def main():
    print('=' * 78)
    print('P198dv dry-run — Insurance-Request email matches Shipment Advice')
    print('=' * 78)

    pass_n, fail_n = 0, 0

    # ── Test 1: every variant matches "Shipment Advice" target ──
    print('\n[Test 1] Each Insurance-Request label → matches "Shipment Advice"')
    for label in INS_REQUEST_LABELS:
        pkts = [make_packet(label)]
        m = _find_matching_docs('Shipment Advice', pkts)
        ok = (len(m) == 1)
        tag = 'OK ' if ok else 'FAIL'
        print(f"   [{tag}] '{label}' → matched={len(m)}")
        if ok: pass_n += 1
        else:  fail_n += 1

    # ── Test 2: negative — none of these should match "Shipment Advice" ──
    print('\n[Test 2] Other doc-types do NOT match "Shipment Advice"')
    for label in NEGATIVE_LABELS:
        pkts = [make_packet(label)]
        m = _find_matching_docs('Shipment Advice', pkts)
        # Allow at most a fuzzy keyword match; the strict check
        # is on the alias group only, so canonical mismatches
        # should yield 0.
        ok = (len(m) == 0)
        tag = 'OK ' if ok else 'FAIL'
        print(f"   [{tag}] '{label}' → matched={len(m)}")
        if ok: pass_n += 1
        else:  fail_n += 1

    # ── Test 3: F46A-4-style clause target — Insurance Request page
    # is found AND its text contains the LC reference + BL + vessel
    # so any downstream content checks can validate compliance.
    print('\n[Test 3] Real email content carries the LC F46A-4 evidence '
          '(LC ref, BL, vessel, ports, invoice #/value)')
    text = INS_REQUEST_TEXT.upper()
    indicators = {
        'documentary credit / LC reference':
            r'DOCUMENTARY\s+CREDIT|LC\s+(?:NO\.?|REFERENCE)|1001LC83147/2025',
        'BL number':         r'BILL\s+OF\s+LA[ND]+ING|B/?L\s+NO\.?\s*[A-Z0-9]',
        'vessel':            r'VESSEL\s+NAME|GERHARD\s+SCHULTE',
        'voyage':            r'VOYAGE',
        'port of loading':   r'PORT\s+OF\s+LOADING|LA\s+SPEZIA',
        'port of discharge': r'PORT\s+OF\s+DISCHARGE|KARACHI',
        'invoice number':    r'INVOICE\s+(?:NO|NUMBER)',
        'invoice value':     r'INVOICE\s+VALUE|EUR\s+\d',
        'shipped-on-board':  r'SHIPPED\s+ON\s+BOARD',
        'open policy ref':   r'OPEN\s+POLICY\s+NO',
        'IGI insurer':       r'IGI\s+(?:GENERAL\s+)?INSURANCE',
        'applicant CCed':    r'FATIMAFURQAN@CBL|CONTINENTAL\s+BISCUITS',
    }
    misses = []
    for k, pat in indicators.items():
        if not re.search(pat, text):
            misses.append(k)
    if misses:
        print(f'   FAIL — missing evidence: {misses}')
        fail_n += 1
    else:
        print('   OK — all required F46A-4 indicators present in email body')
        pass_n += 1

    # ── Test 4: Insurance Policy clause target should NOT match an
    # Insurance Request page (an "insurance request" is not the
    # actual policy and must not auto-pass an insurance-policy LC).
    print('\n[Test 4] LC asking for "Insurance Policy" must NOT match a '
          'shipment-advice-style "Insurance Request" page')
    pkts_mixed = [
        make_packet('Insurance Request', INS_REQUEST_TEXT),
        make_packet('Insurance Policy', 'INSURANCE POLICY NUMBER AB/12345'
                                         '\nINSURED VALUE EUR 30,000\n'
                                         'INSURER: IGI GENERAL INSURANCE'),
    ]
    m = _find_matching_docs('Insurance Policy', pkts_mixed)
    matched_types = [(p.get('document_type') or '').lower() for p in m]
    ok = ('insurance request' not in matched_types
          and 'insurance policy' in matched_types)
    print(f"   matched: {matched_types}")
    print(f"   {'OK ' if ok else 'FAIL'}")
    if ok: pass_n += 1
    else:  fail_n += 1

    # ── Test 5: BOTH an Insurance Request AND a Shipment Advice
    # are present — the F46A "Shipment Advice" target should pick
    # up both (or at least the Shipment Advice). Aggregation will
    # de-duplicate downstream.
    print('\n[Test 5] Both Insurance Request + Shipment Advice present '
          '→ both match "Shipment Advice" target')
    pkts_both = [
        make_packet('Insurance Request', INS_REQUEST_TEXT),
        make_packet('Shipment Advice', INS_REQUEST_TEXT),
    ]
    m = _find_matching_docs('Shipment Advice', pkts_both)
    ok = (len(m) == 2)
    print(f"   matched={len(m)}  {'OK ' if ok else 'FAIL'}")
    if ok: pass_n += 1
    else:  fail_n += 1

    # ── Test 6: empty packet list → no match
    print('\n[Test 6] No packets → no match (sanity)')
    m = _find_matching_docs('Shipment Advice', [])
    ok = (len(m) == 0)
    print(f"   matched={len(m)}  {'OK ' if ok else 'FAIL'}")
    if ok: pass_n += 1
    else:  fail_n += 1

    # ── Summary ──
    total = pass_n + fail_n
    print('\n' + '=' * 78)
    print(f'OVERALL: {pass_n}/{total} {"OK" if fail_n == 0 else "— failures present"}')
    print('=' * 78)
    return 0 if fail_n == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
