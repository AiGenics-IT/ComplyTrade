"""
P198cj dry-run — BAHL multi-page report-grouping guard.

The multi-page "Page X of Y" clustering in step03_sequencing was
merging DIFFERENT SWIFT messages into a single packet whenever a
BAHL bundle prints unified pagination (e.g. "Page 1 of 5" for the
MT730 ack + "Page 2 of 5 ... Page 5 of 5" for the following MT700
LC). The fix: skip any packet whose pages came from the BAHL
splitter when building the Page-X-of-Y clusters.

This harness exercises the specific logic without spinning up the
VLM. It reproduces (1) the user's real job dccea6dd scenario and
(2) synthetic scenarios ensuring we DON'T block merging for
legitimate multi-page non-SWIFT packets.
"""
import re


_SKIP_MERGE = {'blank page', 'blank_page', 'header page'}
_BL_TYPES = {'bill of lading', 'copy non-negotiable bill of lading'}
_BL_TC_TYPES = {'bl conditions of carriage', 'conditions of carriage'}


class Packet:
    def __init__(self, packet_id, document_type, pages, page_numbers):
        self.packet_id = packet_id
        self.document_type = document_type
        self.pages = pages  # list of dicts
        self.page_numbers = page_numbers[:]


def simulate_clustering(merged_packets, page_text_map, bahl_page_set):
    """Mirror of the Page-X-of-Y clustering (with P198cj fix)."""
    report_groups = {}
    for i, pkt in enumerate(merged_packets):
        dt = pkt.document_type.lower().strip()
        if dt in _SKIP_MERGE or dt in _BL_TYPES or dt in _BL_TC_TYPES:
            continue
        # P198cj — skip packets that contain BAHL-pre-classified pages.
        if bahl_page_set and any(
            (pg.get('page_number', 0) if isinstance(pg, dict) else 0) in bahl_page_set
            for pg in pkt.pages
        ):
            continue
        for pg_cls in pkt.pages:
            pg_num = pg_cls.get('page_number', 0) if isinstance(pg_cls, dict) else 0
            pg_text = page_text_map.get(pg_num, '') if pg_num else ''
            pxy = re.search(r'Page\s+(\d+)\s+of\s+(\d+)', pg_text, re.IGNORECASE)
            if pxy and int(pxy.group(2)) > 1:
                page_x = int(pxy.group(1))
                page_y = int(pxy.group(2))
                if page_y > 5 and dt in ('lc', 'amendment', 'mt799', 'mt999',
                                         'mt730', 'mt754', 'mt940', 'mt740', 'mt747'):
                    break
                report_groups.setdefault(page_y, []).append(
                    (i, pkt.page_numbers[0] if pkt.page_numbers else 0, page_x)
                )
                break

    clusters = []
    for _Y, entries in report_groups.items():
        entries = sorted(entries, key=lambda e: e[1])
        cur = []
        _dir = None
        for ent in entries:
            if not cur:
                cur = [ent]; _dir = None; continue
            last = cur[-1]
            gap = ent[1] - last[1]
            if gap <= 0 or gap > (_Y + 2):
                if len(cur) > 1: clusters.append(cur)
                cur = [ent]; _dir = None; continue
            step = ent[2] - last[2]
            if _dir is None:
                if step in (+1, -1): _dir = step; cur.append(ent)
                elif step in (+2, -2): _dir = 1 if step > 0 else -1; cur.append(ent)
                else:
                    if len(cur) > 1: clusters.append(cur)
                    cur = [ent]; _dir = None
            else:
                if step == _dir or step == 2 * _dir: cur.append(ent)
                else:
                    if len(cur) > 1: clusters.append(cur)
                    cur = [ent]; _dir = None
        if len(cur) > 1:
            clusters.append(cur)

    # Apply merges (primary = smallest X in cluster)
    consumed = set()
    for cluster in clusters:
        if len(cluster) <= 1: continue
        sorted_by_x = sorted(cluster, key=lambda e: e[2])
        primary_idx = sorted_by_x[0][0]
        primary = merged_packets[primary_idx]
        for ent in sorted_by_x[1:]:
            oi = ent[0]
            if oi == primary_idx or oi in consumed: continue
            other = merged_packets[oi]
            primary.page_numbers.extend(other.page_numbers)
            primary.pages.extend(other.pages)
            consumed.add(oi)
    return [p for i, p in enumerate(merged_packets) if i not in consumed]


# ── Scenarios ──

def mkp(pid, dt, pgs):
    return Packet(pid, dt, [{'page_number': p} for p in pgs], pgs)


SCENARIOS = []

# Scenario 1 — dccea6dd real case: BAHL MT730 p1 + MT700 LC p2-5,
# bundle pagination "Page 1-5 of 5". Must stay as 2 packets.
SCENARIOS.append(dict(
    name='Real dccea6dd: BAHL MT730 p1 + MT700 LC p2-5, bundle pagination "of 5"',
    packets=[
        mkp('pkt_1', 'MT730', [1]),
        mkp('pkt_2', 'LC', [2, 3, 4, 5]),
        mkp('pkt_3', 'Document Remittance', [6]),
    ],
    page_texts={
        1: 'Identifier: fin.730. Page 1 of 5',
        2: 'Identifier: fin.700. Page 2 of 5',
        3: 'F45A continues. Page 3 of 5',
        4: 'F46A continues. Page 4 of 5',
        5: 'F47A ends. Page 5 of 5',
        6: 'Document Remittance page (no Page X of Y).',
    },
    bahl_page_set={1, 2, 3, 4, 5},
    expect_packets=['pkt_1', 'pkt_2', 'pkt_3'],
    expect_pages={'pkt_1': [1], 'pkt_2': [2, 3, 4, 5], 'pkt_3': [6]},
))

# Scenario 2 — BAHL 4-message bundle (MT799+MT799+MT730+MT700 pages 1-7,
# bundle pagination "of 7"). Must stay as 4 distinct packets.
SCENARIOS.append(dict(
    name='BAHL 4-message bundle (MT799/MT799/MT730/MT700), bundle pagination "of 7"',
    packets=[
        mkp('pkt_1', 'MT799', [1]),
        mkp('pkt_2', 'MT799', [2]),
        mkp('pkt_3', 'MT730', [3]),
        mkp('pkt_4', 'LC', [4, 5, 6, 7]),
    ],
    page_texts={pg: f'fin.xxx Page {pg} of 7' for pg in range(1, 8)},
    bahl_page_set={1, 2, 3, 4, 5, 6, 7},
    expect_packets=['pkt_1', 'pkt_2', 'pkt_3', 'pkt_4'],
    expect_pages={'pkt_1': [1], 'pkt_2': [2], 'pkt_3': [3], 'pkt_4': [4, 5, 6, 7]},
))

# Scenario 3 — legitimate multi-page NON-BAHL report (e.g. Quality
# Cert split into 3 separate packets by VLM but with "Page 1/2/3 of 3"
# pagination). Should STILL merge.
SCENARIOS.append(dict(
    name='Multi-page Quality Certificate (non-BAHL), "Page X of 3" — should merge',
    packets=[
        mkp('pkt_1', 'Quality Certificate', [10]),
        mkp('pkt_2', 'Quality Certificate', [11]),
        mkp('pkt_3', 'Quality Certificate', [12]),
    ],
    page_texts={
        10: 'QC header. Page 1 of 3',
        11: 'QC continued. Page 2 of 3',
        12: 'QC ends. Page 3 of 3',
    },
    bahl_page_set=set(),
    expect_packets=['pkt_1'],
    expect_pages={'pkt_1': [10, 11, 12]},
))

# Scenario 4 — mixed: BAHL pages 1-5 (LC+MT730) + non-BAHL multi-page
# Survey Report on pages 10-12. BAHL must NOT merge, Survey Report must.
SCENARIOS.append(dict(
    name='Mixed: BAHL LC+MT730 + non-BAHL multi-page Survey Report',
    packets=[
        mkp('pkt_1', 'MT730', [1]),
        mkp('pkt_2', 'LC', [2, 3, 4, 5]),
        mkp('pkt_3', 'Survey Report', [10]),
        mkp('pkt_4', 'Survey Report', [11]),
        mkp('pkt_5', 'Survey Report', [12]),
    ],
    page_texts={
        1: 'fin.730 Page 1 of 5',
        2: 'fin.700 Page 2 of 5',
        3: 'Page 3 of 5',
        4: 'Page 4 of 5',
        5: 'Page 5 of 5',
        10: 'SR. Page 1 of 3',
        11: 'SR. Page 2 of 3',
        12: 'SR. Page 3 of 3',
    },
    bahl_page_set={1, 2, 3, 4, 5},
    expect_packets=['pkt_1', 'pkt_2', 'pkt_3'],
    expect_pages={'pkt_1': [1], 'pkt_2': [2, 3, 4, 5], 'pkt_3': [10, 11, 12]},
))

# Scenario 5 — non-BAHL multi-page LC with its OWN pagination (standalone
# MT700 spanning 4 pages, "Page X of 4"). Merging ok. bahl_page_set empty.
SCENARIOS.append(dict(
    name='Non-BAHL standalone MT700 with "Page X of 4" — merge ok',
    packets=[
        mkp('pkt_1', 'LC', [1]),
        mkp('pkt_2', 'LC', [2]),
        mkp('pkt_3', 'LC', [3]),
        mkp('pkt_4', 'LC', [4]),
    ],
    page_texts={
        1: 'MT700 Page 1 of 4',
        2: 'Page 2 of 4',
        3: 'Page 3 of 4',
        4: 'Page 4 of 4',
    },
    bahl_page_set=set(),
    expect_packets=['pkt_1'],
    expect_pages={'pkt_1': [1, 2, 3, 4]},
))

# Scenario 6 — BAHL amendment (MT707 p1-2) + LC (MT700 p3-6) in one
# bundle, "Page X of 6". Must stay 2 packets.
SCENARIOS.append(dict(
    name='BAHL MT707 amendment (p1-2) + MT700 LC (p3-6), bundle "of 6"',
    packets=[
        mkp('pkt_1', 'Amendment', [1, 2]),
        mkp('pkt_2', 'LC', [3, 4, 5, 6]),
    ],
    page_texts={pg: f'Page {pg} of 6' for pg in range(1, 7)},
    bahl_page_set={1, 2, 3, 4, 5, 6},
    expect_packets=['pkt_1', 'pkt_2'],
    expect_pages={'pkt_1': [1, 2], 'pkt_2': [3, 4, 5, 6]},
))


def main():
    passed = 0
    failed = 0
    for i, sc in enumerate(SCENARIOS, 1):
        packets = [Packet(p.packet_id, p.document_type, [dict(pg) for pg in p.pages], list(p.page_numbers))
                   for p in sc['packets']]
        out = simulate_clustering(packets, sc['page_texts'], sc['bahl_page_set'])
        got_ids = [p.packet_id for p in out]
        got_pages = {p.packet_id: sorted(p.page_numbers) for p in out}
        ok = (got_ids == sc['expect_packets'] and got_pages == sc['expect_pages'])
        tag = 'OK ' if ok else 'FAIL'
        print(f"[{tag}] #{i:02d}  {sc['name']}")
        print(f"         expect packets = {sc['expect_packets']}")
        print(f"         got    packets = {got_ids}")
        print(f"         expect pages   = {sc['expect_pages']}")
        print(f"         got    pages   = {got_pages}")
        if ok: passed += 1
        else: failed += 1
    print(f"\n{'='*78}\n{passed}/{passed+failed} BAHL-merge scenarios OK\n{'='*78}")
    return failed == 0


if __name__ == '__main__':
    import sys
    sys.exit(0 if main() else 1)
