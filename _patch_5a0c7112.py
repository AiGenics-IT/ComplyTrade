"""One-off patch for 5a0c7112 — apply P198gz53 (Ship Agent Cert pg11)
+ P198gz54 (Halal Lampiran merge) to step03/step08/step09 cached data."""
import json, os
base = 'results/5a0c7112-1bd9-4c05-8863-93fe0696717c'

# Reconstruct pg11 data from step03 classifications
p3 = base + '/step03/step03_result.json'
d3 = json.load(open(p3, encoding='utf-8'))
pg11_cls = None
for c in d3.get('classifications', []):
    if isinstance(c, dict) and c.get('page_number') == 11:
        pg11_cls = c
        break
print(f'pg11 classification entry: {pg11_cls is not None}')

# === step03: add pkt_5b for pg11 ===
new_packets = []
for pkt in d3['packets']:
    new_packets.append(pkt)
    if pkt.get('packet_id') == 'pkt_5':
        new_pkt = {
            'packet_id': 'pkt_5b',
            'document_type': 'Ship Agent Certificate',
            'pages': [pg11_cls] if pg11_cls else [{'page_number': 11}],
            'page_numbers': [11],
            'boundary_confidence': 0.99,
            'copy_status': 'unknown',
            'copy_label': '',
            'marking_status': (pg11_cls.get('marking_status', '') if pg11_cls else ''),
            'stamps': (pg11_cls.get('stamps', []) if pg11_cls else []),
            'signatures': (pg11_cls.get('signatures', []) if pg11_cls else []),
            'seals': (pg11_cls.get('seals', []) if pg11_cls else []),
            'logos': (pg11_cls.get('logos', []) if pg11_cls else []),
            'doc_hint': 'Ship Agent Certificate from APM PT. ADHIGANA PRATAMA MULYA, vessel MT. CARTAGENA',
        }
        new_packets.append(new_pkt)
d3['packets'] = new_packets
d3['total_packets'] = len(new_packets)
json.dump(d3, open(p3, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
print('step03 patched')


def split_pkt5_and_merge_halal(packets, key_id_for_save):
    """Apply same logic to step08/step09 packet list. Returns updated list."""
    pkt_5 = None
    pkt_7 = None
    for pkt in packets:
        if not isinstance(pkt, dict):
            continue
        if pkt.get('packet_id') == 'pkt_5':
            pkt_5 = pkt
        elif pkt.get('packet_id') == 'pkt_7':
            pkt_7 = pkt

    new_pkt_5b = None
    if pkt_5:
        op = pkt_5.get('original_pages', []) or []
        pg11_op = None
        for o in op:
            if isinstance(o, dict) and o.get('page_number') == 11:
                o['document_type'] = 'Ship Agent Certificate'
                o['is_continuation'] = False
                pg11_op = o
                break
        # Strip pg11 from pkt_5
        pkt_5['original_pages'] = [
            o for o in op
            if not (isinstance(o, dict) and o.get('page_number') == 11)
        ]
        # Filter image paths
        if 'page_image_paths' in pkt_5:
            pkt_5['page_image_paths'] = [
                ip for ip in pkt_5['page_image_paths']
                if 'page_011' not in str(ip) and 'page_11.' not in str(ip)
            ]
        if 'page_numbers' in pkt_5:
            pkt_5['page_numbers'] = [p for p in pkt_5['page_numbers'] if p != 11]
        # Build pkt_5b
        new_pkt_5b = {
            'packet_id': 'pkt_5b',
            'original_pages': [pg11_op] if pg11_op else [{'page_number': 11}],
            'page_numbers': [11],
            'document_type': 'Ship Agent Certificate',
            'classification_status': 'extra_document',
            'match_confidence': 0.95,
            'matched_requirement_index': -1,
            'matched_requirement_name': '',
            'document_summary': 'Ship Agent Certificate from APM PT. ADHIGANA PRATAMA MULYA, vessel MT. CARTAGENA',
            'document_number': '',
            'document_date': '',
            'document_amount': '',
            'stamps': (pg11_op.get('stamps', []) if pg11_op else []),
            'signatures': (pg11_op.get('signatures', []) if pg11_op else []),
            'seals': [],
            'logos': [],
            'copy_status': 'unknown',
            'copy_label': '',
            'marking_status': (pg11_op.get('marking_status', '') if pg11_op else ''),
            'issued_by': 'APM PT. ADHIGANA PRATAMA MULYA',
            'lc_reference': '',
            'unified_summary': {},
        }

    consumed = []
    if pkt_7:
        for pkt in packets:
            if not isinstance(pkt, dict):
                continue
            if pkt.get('packet_id') == pkt_7.get('packet_id'):
                continue
            op = pkt.get('original_pages', []) or []
            pg_nums = [(x.get('page_number') if isinstance(x, dict) else x) for x in op]
            pg_nums = [p for p in pg_nums if isinstance(p, int)]
            if any(p in (14, 15, 16) for p in pg_nums):
                for o in op:
                    if isinstance(o, dict) and o.get('page_number') in (14, 15, 16):
                        o['document_type'] = 'Halal Certificate'
                        o['is_continuation'] = True
                        pkt_7.setdefault('original_pages', []).append(o)
                # Move image paths too
                pkt_7.setdefault('page_image_paths', []).extend(pkt.get('page_image_paths', []) or [])
                consumed.append(pkt.get('packet_id'))
        if consumed:
            if 'Lampiran' not in (pkt_7.get('document_type') or ''):
                pkt_7['document_type'] = 'Halal Certificate + Lampiran (Attachment)'
            # Update page_numbers field if present
            if 'page_numbers' in pkt_7:
                op = pkt_7.get('original_pages', []) or []
                pkt_7['page_numbers'] = sorted({
                    o.get('page_number') for o in op
                    if isinstance(o, dict) and isinstance(o.get('page_number'), int)
                })

    out = [p for p in packets if p.get('packet_id') not in consumed]
    if new_pkt_5b:
        # Insert pkt_5b after pkt_5
        idx = next((i for i, p in enumerate(out)
                    if p.get('packet_id') == 'pkt_5'), -1)
        if idx >= 0:
            out.insert(idx + 1, new_pkt_5b)
        else:
            out.append(new_pkt_5b)
    return out


# step08
p8 = base + '/step08/step08_result.json'
if os.path.exists(p8):
    d8 = json.load(open(p8, encoding='utf-8'))
    d8['classified_packets'] = split_pkt5_and_merge_halal(
        d8.get('classified_packets', []), 'classified_packets')
    d8['total_packets'] = len(d8['classified_packets'])
    json.dump(d8, open(p8, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print('step08 patched')

# step09
p9 = base + '/step09/step09_result.json'
if os.path.exists(p9):
    d9 = json.load(open(p9, encoding='utf-8'))
    key = 'packets' if 'packets' in d9 else 'reconciled_packets'
    d9[key] = split_pkt5_and_merge_halal(d9.get(key, []), key)
    json.dump(d9, open(p9, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print('step09 patched')


# Verification — step08 (UI source)
print('\n=== step08 final state pg 8-18 ===')
d8 = json.load(open(p8, encoding='utf-8'))
for cp in d8.get('classified_packets', []):
    op = cp.get('original_pages', []) or []
    pg_nums = sorted({
        (x.get('page_number') if isinstance(x, dict) else x) for x in op
        if (isinstance(x, dict) and isinstance(x.get('page_number'), int))
        or isinstance(x, int)
    })
    if any(8 <= p <= 18 for p in pg_nums):
        print(f"  {cp.get('packet_id'):<10} {cp.get('document_type'):<55} pages={pg_nums}")
