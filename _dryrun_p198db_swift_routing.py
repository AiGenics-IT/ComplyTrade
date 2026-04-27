"""
P198db dry-run — MT799 / MT999 SWIFT advice routed to shipping
packets so step 14's F47A-9 check (P198da) can find them.

Server-side: when an MT799/MT999 packet is detected on the LC
side AND it is NOT an amendment, a copy is also placed in
shipping_packets with mt_type='shipping',
is_swift_advice_copy=True, source_mt='MT799'.

Step 14 P198da `_has_swift_advice_packet` now picks these up by:
  • document_type containing mt799/mt999/fin.799/fin.999/free format
  • source_mt = MT799 / MT999
  • is_swift_advice_copy = True
"""
import re
import sys


# Test-local mirror of the routing logic in server.py:3036+
def route_packet_pipeline(s3_packet):
    """Mirror the server.py pipeline routing for one s3 packet.
    Returns (mt_packets_list, shipping_packets_list)."""
    mt_packets = []
    shipping_packets = []
    pkt_copy = dict(s3_packet)
    dt = (s3_packet.get('document_type', '') or '').lower()
    text = (s3_packet.get('document_text') or '') or ''
    text_up = text.upper()

    # Free-format MT799/MT999 detection
    is_ff = (
        'mt799' in dt or 'mt 799' in dt or 'fin.799' in dt
        or 'mt999' in dt or 'mt 999' in dt or 'fin.999' in dt
        or 'free format' in dt or 'free-format' in dt
        or 'F79:' in text or ':79:' in text or 'fin.799' in text.lower()
        or 'fin.999' in text.lower()
    )
    if is_ff:
        # Amendment marker (F26E)
        is_amendment = bool(re.search(r'\bF26E\b|\b:26E:\b', text))
        if is_amendment:
            pkt_copy['mt_type'] = 'MT707'
            pkt_copy['source_mt'] = 'MT799'
            pkt_copy['is_799_amendment'] = True
            mt_packets.append(pkt_copy)
        else:
            pkt_copy['mt_type'] = 'MT799'
            pkt_copy['source_mt'] = 'MT799'
            pkt_copy['is_799_amendment'] = False
            mt_packets.append(pkt_copy)
            # P198db — copy to shipping
            ship_copy = dict(pkt_copy)
            ship_copy['mt_type'] = 'shipping'
            ship_copy['document_type'] = pkt_copy.get('document_type') or 'MT799'
            ship_copy['source_mt'] = 'MT799'
            ship_copy['is_swift_advice_copy'] = True
            shipping_packets.append(ship_copy)
        return mt_packets, shipping_packets
    if 'lc' == dt or 'letter of credit' in dt or 'amendment' in dt:
        pkt_copy['mt_type'] = 'MT707' if 'amend' in dt else 'MT700'
        mt_packets.append(pkt_copy)
        return mt_packets, shipping_packets
    pkt_copy['mt_type'] = 'shipping'
    shipping_packets.append(pkt_copy)
    return mt_packets, shipping_packets


# Step 14 P198da has-swift-advice check (mirror)
def has_swift_advice_packet(pkts):
    for p in pkts or []:
        if not isinstance(p, dict):
            continue
        dt = (p.get('document_type') or '').lower()
        if any(k in dt for k in ('mt799', 'mt 799', 'mt999',
                                  'mt 999', 'fin.799', 'fin.999',
                                  'free format message',
                                  'free-format message',
                                  'authenticated swift',
                                  'swift advice',
                                  'swift message')):
            return p
        if str(p.get('source_mt') or '').upper() in ('MT799', 'MT999'):
            return p
        if p.get('is_swift_advice_copy'):
            return p
    return None


# Scenarios
SC = []

# Scenario 1: Non-amendment MT799 → also copied to shipping
SC.append(dict(
    name='Non-amendment MT799 → routed to BOTH mt_packets and shipping_packets',
    s3_pkt=dict(packet_id='pkt_5', document_type='MT799',
                document_text='F20: NEG-001\nF79:\nWE HEREBY ADVISE...'),
    expect_mt=1, expect_ship=1,
    expect_swift_in_ship=True))

# Scenario 2: MT799 amendment (F26E) → ONLY mt_packets
SC.append(dict(
    name='MT799 amendment (F26E) → only mt_packets, NOT in shipping',
    s3_pkt=dict(packet_id='pkt_2', document_type='MT799',
                document_text='F20: AMD-001\nF26E: 1\nF79:\nUNDER FIELD 45A...'),
    expect_mt=1, expect_ship=0,
    expect_swift_in_ship=False))

# Scenario 3: MT999 free-format → routed to both
SC.append(dict(
    name='MT999 free-format → both lists',
    s3_pkt=dict(packet_id='pkt_7', document_type='MT999',
                document_text='F20: 12345\nF79: SOME NARRATIVE'),
    expect_mt=1, expect_ship=1,
    expect_swift_in_ship=True))

# Scenario 4: Plain shipping doc → only shipping
SC.append(dict(
    name='Bill of Lading → only shipping_packets',
    s3_pkt=dict(packet_id='pkt_10', document_type='Bill of Lading',
                document_text='B/L No. 12345...'),
    expect_mt=0, expect_ship=1,
    expect_swift_in_ship=False))

# Scenario 5: LC → only mt_packets
SC.append(dict(
    name='LC (MT700) → only mt_packets',
    s3_pkt=dict(packet_id='pkt_1', document_type='LC',
                document_text='F20: 0001LC55282/2025...'),
    expect_mt=1, expect_ship=0,
    expect_swift_in_ship=False))


def main():
    passed = 0; failed = 0
    for i, sc in enumerate(SC, 1):
        mts, ships = route_packet_pipeline(sc['s3_pkt'])
        swift_in_ship = has_swift_advice_packet(ships) is not None
        ok = (
            len(mts) == sc['expect_mt']
            and len(ships) == sc['expect_ship']
            and swift_in_ship == sc['expect_swift_in_ship']
        )
        tag = 'OK ' if ok else 'FAIL'
        print(f"[{tag}] #{i:02d}  {sc['name']}")
        print(f"         expect: mt={sc['expect_mt']} ship={sc['expect_ship']} "
              f"swift_in_ship={sc['expect_swift_in_ship']}")
        print(f"         got:    mt={len(mts)} ship={len(ships)} "
              f"swift_in_ship={swift_in_ship}")
        if ok: passed += 1
        else: failed += 1
    print(f"\n{'='*78}\n{passed}/{passed+failed} P198db routing scenarios OK\n{'='*78}")
    return failed == 0


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
