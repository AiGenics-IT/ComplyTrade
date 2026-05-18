"""Smoke tests for the 8083 -> 8082 adapter.

Verifies:
  • _doc_type_to_mt maps doc_type+kind to the right MT enum
  • _parse_amendment_number pulls the N from 'Amendment N'
  • _build_page_texts pulls cleaned text per page
  • _build_packets produces packets in the shape step06 expects
  • adapt_8083_to_step_results returns the right top-level keys
  • Real-job parity: run against an existing 8083 classification.json
    (when one is on disk) and check the output is well-formed.
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bridge.adapter import (
    adapt_8083_to_step_results,
    _doc_type_to_mt,
    _parse_amendment_number,
    _build_page_texts,
    _build_packets,
    _flatten_stamps_and_signatures,
    _ucp17_apparent_originals,
)


fails = []
def expect(label, actual, expected):
    if actual == expected:
        print(f'  PASS  {label}')
    else:
        print(f'  FAIL  {label}')
        print(f'        expected: {expected!r}')
        print(f'        actual:   {actual!r}')
        fails.append(label)


# ── _doc_type_to_mt ───────────────────────────────────────────────
print('\n== _doc_type_to_mt ==')
expect('LC                  -> MT700', _doc_type_to_mt('LC', 'swift'), 'MT700')
expect('Amendment 1         -> MT707', _doc_type_to_mt('Amendment 1', 'swift'), 'MT707')
expect('Amendment 13        -> MT707', _doc_type_to_mt('Amendment 13', 'swift'), 'MT707')
expect('MT799               -> MT799', _doc_type_to_mt('MT799', 'swift'), 'MT799')
expect('MT999               -> MT999', _doc_type_to_mt('MT999', 'swift'), 'MT999')
expect('Refusal Notice      -> MT734', _doc_type_to_mt('Refusal Notice', 'swift'), 'MT734')
expect('Acknowledgment      -> MT730', _doc_type_to_mt('Acknowledgment Advice', 'swift'), 'MT730')
expect('Bill of Lading      -> shipping', _doc_type_to_mt('Bill of Lading', 'lc_required'), 'shipping')
expect('Commercial Invoice  -> shipping', _doc_type_to_mt('Commercial Invoice', 'extra'), 'shipping')
expect('empty / non-SWIFT   -> shipping', _doc_type_to_mt('', 'extra'), 'shipping')


# ── _parse_amendment_number ───────────────────────────────────────
print('\n== _parse_amendment_number ==')
expect('Amendment 1      -> 1', _parse_amendment_number('Amendment 1'), 1)
expect('Amendment 13     -> 13', _parse_amendment_number('Amendment 13'), 13)
expect('Amendment 5 Cont -> 5', _parse_amendment_number('Amendment 5 Continuation'), 5)
expect('LC               -> None', _parse_amendment_number('LC'), None)
expect('empty            -> None', _parse_amendment_number(''), None)


# ── _build_page_texts ─────────────────────────────────────────────
print('\n== _build_page_texts ==')
fake8083 = {
    'pages': [
        {'page_number': 1, 'text': 'page one body'},
        {'page_number': 2, 'text': ''},   # empty text -> skip
        {'page_number': 3, 'text': 'page three body'},
    ]
}
pt = _build_page_texts(fake8083)
expect('len',  len(pt), 2)
expect('pg1',  pt.get(1), 'page one body')
expect('pg3',  pt.get(3), 'page three body')
expect('pg2 missing', 2 in pt, False)


# ── _flatten_stamps_and_signatures ────────────────────────────────
print('\n== _flatten_stamps_and_signatures ==')
fields = {
    'Signature': {
        'shipper_signature': 'FOR ANGKANA',
        'carrier_signature': 'OCEAN NETWORK EXPRESS',
        'signing_capacity': 'As Agent for the Carrier',
        'stamps': ['SECOND ORIGINAL', 'SHIPPED ON BOARD'],
    },
    'Stamps': ['LCBN000395A', 'CLEAN ON BOARD'],
    'Bank Stamps': [
        {'stamp_text': 'RECEIVED 23 APR 2026 BANK AL HABIB',
         'stamp_date': '2026-04-23',
         'stamp_role': 'receipt',
         'bank_name': 'BANK AL HABIB LIMITED'},
    ],
    'Endorsements': [
        {'sequence': '1st',
         'endorser_name': 'PAK SUZUKI MOTOR',
         'endorsement_date': '2026-04-25',
         'endorsement_text': 'FOR PAK SUZUKI MOTOR ...',
         'is_blank_endorsement': False},
    ],
}
stamps, sigs = _flatten_stamps_and_signatures(fields)
expect('stamps count (2 sig + 2 std + 1 bank)', len(stamps), 5)
expect('signatures count (1 sig + 1 endorsement)', len(sigs), 2)
expect('bank stamp date carried',
       any(s.get('date') == '2026-04-23' for s in stamps), True)
expect('endorsement type is endorsement',
       any(s.get('type') == 'endorsement' for s in sigs), True)


# ── _ucp17_apparent_originals (Shipment Advice only) ──────────────
print('\n== _ucp17_apparent_originals (Shipment Advice only) ==')

# ── Promotion ONLY when document_type is Shipment Advice ─────────
# Case A: Shipment Advice, 1 unmarked + signature → promote
pkt_a = {'document_type': 'Shipment Advice',
         'originals_count': 0, 'copies_count': 0, 'unknown_marker_count': 1,
         'signatures': [{'description': 'Jessie Zhang'}], 'stamps': []}
expect('SA: unmarked + signature -> (1, 1)', _ucp17_apparent_originals(pkt_a), (1, 1))

# Case B: Shipment Advice, 1 unmarked + stamp only → promote
pkt_b = {'document_type': 'Shipment Advice',
         'originals_count': 0, 'copies_count': 0, 'unknown_marker_count': 1,
         'signatures': [], 'stamps': [{'text': 'BANK STAMP'}]}
expect('SA: unmarked + stamp -> (1, 1)', _ucp17_apparent_originals(pkt_b), (1, 1))

# Case C: Shipment Advice, no signature AND no stamp → no promotion
pkt_c = {'document_type': 'Shipment Advice',
         'originals_count': 0, 'copies_count': 0, 'unknown_marker_count': 1,
         'signatures': [], 'stamps': []}
expect('SA: unmarked + no sig/stamp -> (0, 0)', _ucp17_apparent_originals(pkt_c), (0, 0))

# ── All other doc types must return raw counts, no matter what ───
# Case D: Bill of Lading with unmarked + signature → NO promotion
pkt_d = {'document_type': 'Bill of Lading',
         'originals_count': 0, 'copies_count': 0, 'unknown_marker_count': 1,
         'signatures': [{'description': 'x'}], 'stamps': []}
expect('BL: untouched even when signed -> (0, 0)', _ucp17_apparent_originals(pkt_d), (0, 0))

# Case E: Commercial Invoice with unmarked + signature → NO promotion
pkt_e = {'document_type': 'Commercial Invoice',
         'originals_count': 0, 'copies_count': 0, 'unknown_marker_count': 1,
         'signatures': [{'description': 'x'}], 'stamps': []}
expect('CI: untouched even when signed -> (0, 0)', _ucp17_apparent_originals(pkt_e), (0, 0))

# Case F: SA with already-counted ORIGINAL — passes raw through, no change
pkt_f = {'document_type': 'Shipment Advice',
         'originals_count': 1, 'copies_count': 0, 'unknown_marker_count': 0,
         'signatures': [{'description': 'x'}], 'stamps': []}
expect('SA: explicit ORIGINAL untouched -> (1, 0)', _ucp17_apparent_originals(pkt_f), (1, 0))

# Case G: missing/None values are safe on the SA path
pkt_g = {'document_type': 'Shipment Advice',
         'originals_count': None, 'unknown_marker_count': None,
         'signatures': [{'description': 'x'}]}
expect('SA: None counts -> (0, 0)', _ucp17_apparent_originals(pkt_g), (0, 0))


# ── _build_packets ────────────────────────────────────────────────
print('\n== _build_packets ==')
fake8083 = {
    'pages': [
        {'page_number': 1, 'text': 'LC text 1'},
        {'page_number': 2, 'text': 'LC text 2'},
        {'page_number': 3, 'text': 'Amendment text 1'},
        {'page_number': 10, 'text': 'BL text'},
    ],
    'logical_documents': [
        {'logical_doc_id': 'sw_1', 'kind': 'swift',
         'document_type': 'LC', 'all_pages': [1, 2],
         'lc_reference': '1003LC55989/2026'},
        {'logical_doc_id': 'sw_2', 'kind': 'swift',
         'document_type': 'Amendment 1', 'all_pages': [3]},
        {'logical_doc_id': 'ld_1', 'kind': 'lc_required',
         'document_type': 'Bill of Lading', 'all_pages': [10],
         'fields': {'Stamps': ['ORIGINAL']}},
    ],
}
pt = _build_page_texts(fake8083)
packets = _build_packets(fake8083, pt)
expect('packet count', len(packets), 3)
expect('packet 1 mt_type', packets[0]['mt_type'], 'MT700')
expect('packet 1 page_numbers', packets[0]['page_numbers'], [1, 2])
expect('packet 1 lc_reference', packets[0]['lc_reference'], '1003LC55989/2026')
expect('packet 2 mt_type', packets[1]['mt_type'], 'MT707')
expect('packet 2 amendment_number', packets[1]['amendment_number'], 1)
expect('packet 3 mt_type', packets[2]['mt_type'], 'shipping')
expect('packet 3 has page text',
       'BL text' in packets[2]['pages'][0]['refined_text'], True)


# ── adapt_8083_to_step_results ────────────────────────────────────
print('\n== adapt_8083_to_step_results ==')
out = adapt_8083_to_step_results(fake8083)
expect('top-level keys',
       sorted(out.keys()),
       ['step01', 'step02', 'step03', 'step04', 'step05', 'step08', 'step09'])
expect('step05 has packets', 'packets' in out['step05'], True)
expect('step05 has page_texts', 'page_texts' in out['step05'], True)
expect('step05 packets count', len(out['step05']['packets']), 3)
expect('step08 only shipping',
       len(out['step08']['classified_packets']), 1)
expect('step03 packets count', len(out['step03']['packets']), 3)
expect('step04 packets carry mt_type',
       all('mt_type' in p for p in out['step04']['packets']), True)


# ── Real-job test: run against an existing 8083 classification.json
print('\n== Real-job adapt ==')
RESULTS = 'D:/COMPLYTRADE/V7/FINAL/classifier_server/results'
target = None
if os.path.isdir(RESULTS):
    for name in sorted(os.listdir(RESULTS), reverse=True):
        p = os.path.join(RESULTS, name, 'classification.json')
        if os.path.exists(p):
            target = p
            break
if target:
    with open(target, 'r', encoding='utf-8') as f:
        c = json.load(f)
    out = adapt_8083_to_step_results(c)
    print(f'  source job: {target}')
    print(f'  total_pages:          {c.get("total_pages")}')
    print(f'  logical_documents:    {len(c.get("logical_documents") or [])}')
    print(f'  step05 packets:       {len(out["step05"]["packets"])}')
    print(f'  step05 page_texts:    {len(out["step05"]["page_texts"])}')
    print(f'  step08 shipping pkts: {len(out["step08"]["classified_packets"])}')
    # Print packet mt_type breakdown
    from collections import Counter
    mt_counts = Counter(p['mt_type'] for p in out['step05']['packets'])
    print(f'  packet mt_types:      {dict(mt_counts)}')
    # Sample first shipping packet
    sp = next((p for p in out['step05']['packets'] if p['mt_type'] == 'shipping'), None)
    if sp:
        print(f'  sample shipping pkt:  document_type={sp["document_type"]!r}, '
              f'pages={sp["page_numbers"][:5]}, '
              f'stamps={len(sp["stamps"])}, signatures={len(sp["signatures"])}')
    expect('real-job: page_texts non-empty', len(out['step05']['page_texts']) > 0, True)
    expect('real-job: packets non-empty', len(out['step05']['packets']) > 0, True)
    expect('real-job: at least one SWIFT packet',
           any(p['mt_type'].startswith('MT') for p in out['step05']['packets']),
           True)
else:
    print('  (no existing 8083 classification.json on disk — skipping)')


print('\n' + '=' * 60)
if fails:
    print(f'FAILED: {len(fails)} test(s)')
    for f in fails:
        print(f'  - {f}')
    sys.exit(1)
else:
    print('ALL ADAPTER TESTS PASS')
    sys.exit(0)
