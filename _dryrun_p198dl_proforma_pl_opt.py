"""
P198dl dry-run — F45A proforma OPPORTUNISTIC check on Packing List.

Step 13 clones an F45A proforma invoice number/date condition to a
Packing List row with condition_id ending in '-PL-OPT'. Step 14's
pre-processor must:

  • silently skip (compliance='N/A') when the Packing List does NOT
    carry any proforma reference;
  • leave the row unchanged when the PL DOES carry one (LLM verifies
    the match normally).

Tests use:
  • job 08345848-0e35-4c02-9f26-7287f60b2028's real PL packets
    (no proforma reference) → must skip silently
  • a synthetic PL with proforma → must NOT pre-decide (LLM path)
"""
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def pl_has_proforma(text):
    u = (text or '').upper()
    return bool(
        re.search(r'\bPRO\s*[\-]?\s*FORMA\b', u)
        or re.search(r'\bP/?\s*INVOICE\b', u)
        or re.search(r'\bPI\s+(?:NO\.?|#|NUMBER)', u)
    )


def simulate_p198dl(rows, packets):
    """Mirror of the step14 P198dl pre-processor."""
    pl_pkt = None
    for p in packets:
        dt = (p.get('document_type') or '').lower()
        if 'packing list' in dt:
            pl_pkt = p
            break
    pl_text = ''
    if pl_pkt:
        pl_text = (pl_pkt.get('refined_text')
                   or pl_pkt.get('cleaned_text')
                   or pl_pkt.get('document_text') or '')
    has_pf = pl_has_proforma(pl_text)
    skipped = []
    left_for_llm = []
    for row in rows:
        cid = row.get('condition_id') or ''
        if not cid.endswith('-PL-OPT'):
            continue
        if has_pf:
            left_for_llm.append(row['row_id'])
            continue
        row['compliance'] = 'N/A'
        row['result'] = ('Not applicable — Packing List does not carry '
                         'proforma invoice number/date')
        row['_p198da_handled'] = True
        skipped.append(row['row_id'])
    return skipped, left_for_llm, has_pf


# Real PL packets from job 08345848
JOB = Path('results/08345848-0e35-4c02-9f26-7287f60b2028')
real_step09 = json.loads(
    (JOB / 'step09' / 'step09_result.json').read_text(encoding='utf-8'))
real_packets = (real_step09.get('reconciled_packets')
                or real_step09.get('packets') or [])

# Real F45A condition example: proforma reference in F45A
real_proforma_row = {
    'row_id': 'R0099',
    'condition_id': '45A-1-C3-PL-OPT',
    'condition_text': ('STRICTLY AS PER BENEFICIARY\'S PROFORMA INVOICE '
                       'NO. MCI-786/S-13198-SOY-E DATED FEBRUARY 1, 2026'),
    'document_checked': 'Packing List',
    'compliance': 'PENDING',
}

# Goods/qty PL row (existing P198dc), should NOT be touched
goods_row = {
    'row_id': 'R0050',
    'condition_id': '45A-1-C1-PL',
    'condition_text': 'GOODS DESCRIPTION: SOYBEANS',
    'document_checked': 'Packing List',
    'compliance': 'PENDING',
}


def main():
    print('=' * 78)
    print('P198dl dry-run — F45A proforma opportunistic on Packing List')
    print('=' * 78)

    # ── Test 1: real job's PL DOES carry a proforma reference
    # ("STRICTLY AS PER BENEFICIARY'S PROFORMA INVOICE REF.NO.786/...")
    # → must leave for LLM (no skip). Goods row must stay untouched. ──
    print('\n[Test 1] Job 08345848 real PL → leave for LLM (PL carries '
          'proforma ref)')
    rows = [dict(real_proforma_row), dict(goods_row)]
    skipped, llm, has_pf = simulate_p198dl(rows, real_packets)
    print(f"  PL has proforma ref: {has_pf}")
    print(f"  Skipped (N/A): {skipped}")
    print(f"  Left for LLM:  {llm}")
    pf_row = next(r for r in rows if r['row_id'] == 'R0099')
    g_row = next(r for r in rows if r['row_id'] == 'R0050')
    t1_ok = (
        has_pf
        and pf_row['compliance'] == 'PENDING'  # left for LLM
        and g_row['compliance'] == 'PENDING'   # untouched
        and llm == ['R0099']
        and skipped == []
    )
    print(f"  RESULT: {'OK' if t1_ok else 'FAIL'}")

    # ── Test 1b: clean PL with NO proforma reference at all → skip ──
    print('\n[Test 1b] Synthetic PL with NO proforma reference '
          '→ silent skip (N/A)')
    pl_clean = [{
        'document_type': 'Packing List',
        'refined_text': ('PACKING LIST\nDate: 01-FEB-2026\n'
                         'Invoice No.: MCI-786/S-13198-SOY-E\n'
                         'Goods: SOYBEANS  Qty: 1000 MT\n'
                         'Net Weight: 1000 MT  Gross Weight: 1010 MT\n'
                         'Marks: as per LC'),
    }]
    rows = [dict(real_proforma_row)]
    skipped, llm, has_pf = simulate_p198dl(rows, pl_clean)
    print(f"  PL has proforma ref: {has_pf}")
    print(f"  Skipped: {skipped}, Left for LLM: {llm}")
    t1b_ok = (not has_pf and skipped == ['R0099']
              and rows[0]['compliance'] == 'N/A'
              and rows[0].get('_p198da_handled') is True)
    print(f"  RESULT: {'OK' if t1b_ok else 'FAIL'}")

    # ── Test 2: synthetic PL with proforma reference → leave for LLM
    print('\n[Test 2] Synthetic PL with proforma ref → leave for LLM')
    pl_with_pf = [{
        'document_type': 'Packing List',
        'refined_text': ('PACKING LIST\nProforma Invoice No. '
                         'MCI-786/S-13198-SOY-E dated 01-Feb-2026\n'
                         'Goods: Soybeans 1000 MT'),
    }]
    rows = [dict(real_proforma_row)]
    skipped, llm, has_pf = simulate_p198dl(rows, pl_with_pf)
    print(f"  PL has proforma ref: {has_pf}")
    print(f"  Skipped: {skipped}, Left for LLM: {llm}")
    t2_ok = (has_pf and not skipped and llm == ['R0099']
             and rows[0]['compliance'] == 'PENDING')
    print(f"  RESULT: {'OK' if t2_ok else 'FAIL'}")

    # ── Test 3: PL with "PI No." short form → recognised
    print('\n[Test 3] PL using short form "PI No. ..." → recognised')
    pl_pi = [{
        'document_type': 'Packing List',
        'refined_text': 'PI No. MCI-786 DATE 01-FEB-2026',
    }]
    rows = [dict(real_proforma_row)]
    skipped, llm, has_pf = simulate_p198dl(rows, pl_pi)
    print(f"  PL has proforma ref: {has_pf}")
    t3_ok = has_pf and not skipped
    print(f"  RESULT: {'OK' if t3_ok else 'FAIL'}")

    # ── Test 4: empty packet list → silent skip
    print('\n[Test 4] No PL packet at all → silent skip')
    rows = [dict(real_proforma_row)]
    skipped, llm, has_pf = simulate_p198dl(rows, [])
    print(f"  PL has proforma ref: {has_pf}")
    print(f"  Skipped: {skipped}")
    t4_ok = (not has_pf and skipped == ['R0099']
             and rows[0]['compliance'] == 'N/A')
    print(f"  RESULT: {'OK' if t4_ok else 'FAIL'}")

    # ── Test 5: row without -PL-OPT marker is NEVER touched
    print('\n[Test 5] Row without -PL-OPT marker is never touched')
    rows = [dict(goods_row), {
        'row_id': 'R0001',
        'condition_id': '45A-1-C1',
        'condition_text': 'Goods description',
        'document_checked': 'Commercial Invoice',
        'compliance': 'PENDING',
    }]
    skipped, llm, has_pf = simulate_p198dl(rows, [])
    print(f"  Skipped: {skipped}, Left for LLM: {llm}")
    t5_ok = (skipped == [] and llm == [])
    print(f"  RESULT: {'OK' if t5_ok else 'FAIL'}")

    all_ok = t1_ok and t1b_ok and t2_ok and t3_ok and t4_ok and t5_ok
    print('\n' + '=' * 78)
    print(f'OVERALL: {"OK — all 6 scenarios pass" if all_ok else "FAIL"}')
    print('=' * 78)
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
