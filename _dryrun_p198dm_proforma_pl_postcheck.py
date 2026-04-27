"""
P198dm dry-run — extend P198ak proforma date integrity post-check
to also cover Packing List rows (cloned by P198dl).

Job 08345848 LC F45A:  PROFORMA INVOICE REF.NO. 786/S-13198-SOYPI-E
                       DATED JAN 21, 2026
Job 08345848 CI body:  PROFORMA INVOICE REF.NO. 786/S-13198 SOYPI-E
                       DATED FEB 18, 2026   ← mismatch
Job 08345848 PL body:  PROFORMA INVOICE REF.NO. 786/S-13198-SOYPI-E
                       DATED FEB 18, 2026   ← mismatch

Verifies:
  1. CI row gets flipped PASS->FAIL (regression of P198ak — must
     keep working).
  2. PL row gets flipped PASS->FAIL (the new P198dm scope).
  3. CI row is NOT flipped by a PL-only mismatch and vice versa
     (citation scope per document type).
  4. PL row that genuinely matches LC date stays PASS.
  5. Row with non-matching ref is left alone.

Mirrors only the comparison logic from step14_verification.py
(re-implemented in isolation — no LLM, no module imports).
"""
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')


_PRO_INV_REGEX = re.compile(
    r'(?:P(?:RO)?\.?\s*)?FORMA\s*(?:INV(?:OICE)?\.?)?\s*'
    r'(?:REF\.?|#)?\s*(?:NO\.?|NUMBER)?\s*[:\.]?\s*'
    r'([A-Z0-9][A-Z0-9/\- .\n]*?)\s*'
    r'(?:DATED|DT\.?|DATE|DT)\s*[:\.]?\s*'
    r'([A-Z]+\.?\s*\d{1,2}[,\s]+\d{2,4}|'
    r'\d{1,2}[\s\-./]+[A-Z]+\.?[\s\-./]+\d{2,4}|'
    r'\d{4}[-./]\d{1,2}[-./]\d{1,2}|'
    r'\d{1,2}[-./]\d{1,2}[-./]\d{2,4})',
    re.DOTALL,
)


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
    return None


def norm_ref(s):
    return re.sub(r'[\s\-/.]', '', (s or '').upper())


def find_proforma_in_text(text):
    if not text:
        return None, None
    u = text.upper()
    if 'PROFORMA' not in u and 'PRO-FORMA' not in u and 'PRO FORMA' not in u:
        return None, None
    m = _PRO_INV_REGEX.search(u)
    if not m:
        return None, None
    return re.sub(r'\s+', ' ', m.group(1).strip()), m.group(2).strip()


def simulate_p198dm(rows, packets, lc_f45a):
    """Mirror of P198ak (with P198dm extension). Flip CI/PL rows
    PASS->FAIL when LC ref matches but date differs."""
    # 1. Extract LC proforma ref + date
    lc_m = _PRO_INV_REGEX.search((lc_f45a or '').upper())
    if not lc_m:
        return []
    lc_ref = re.sub(r'\s+', ' ', lc_m.group(1).strip())
    lc_date_raw = lc_m.group(2).strip()
    lc_date = parse_date(lc_date_raw)
    lc_ref_n = norm_ref(lc_ref)

    # 2. Build citations from CI + PL packets
    citations = []
    for p in packets:
        dt = (p.get('document_type') or '').lower()
        is_inv = 'invoice' in dt and 'proforma' not in dt
        is_pl = 'packing list' in dt
        if not (is_inv or is_pl):
            continue
        label = p.get('document_type', 'CI')
        text = (p.get('refined_text') or p.get('cleaned_text') or '')
        ref, date_raw = find_proforma_in_text(text)
        if ref and date_raw:
            citations.append((label, ref, date_raw, parse_date(date_raw)))

    flips = []
    for row in rows:
        cond = (row.get('condition_text') or '').upper()
        if 'PROFORMA' not in cond:
            continue
        doc = (row.get('document_checked') or '').lower()
        if doc and 'invoice' not in doc and 'packing list' not in doc:
            continue
        cur = (row.get('compliance') or '').upper()
        if cur not in ('PASS', 'REVIEW'):
            continue
        # Scope citations to this row's document type
        row_is_pl = 'packing list' in doc
        row_is_inv = 'invoice' in doc and 'proforma' not in doc
        for lab, ref, date_raw, date in citations:
            llab = lab.lower()
            pkt_is_pl = 'packing list' in llab
            pkt_is_inv = 'invoice' in llab and 'proforma' not in llab
            if row_is_pl and not pkt_is_pl:
                continue
            if row_is_inv and not pkt_is_inv:
                continue
            ref_n = norm_ref(ref)
            if not (lc_ref_n == ref_n or lc_ref_n in ref_n or ref_n in lc_ref_n):
                continue
            if lc_date and date and lc_date != date:
                row['compliance'] = 'FAIL'
                row['_flipped_by'] = lab
                flips.append((row['row_id'], lab, lc_date_raw, date_raw))
                break
    return flips


# ── Real job 08345848 data ────────────────────────────────────────
JOB = Path('results/08345848-0e35-4c02-9f26-7287f60b2028')
step09 = json.loads((JOB / 'step09' / 'step09_result.json').read_text(
    encoding='utf-8'))
real_packets = (step09.get('reconciled_packets')
                or step09.get('packets') or [])

LC_F45A_REAL = (
    "1000 M/TONS (+/- 10 PCT) BRAZIL ORIGIN SOYBEANS\n"
    "FURTHER DETAILS AND SPECIFICATIONS MUST STRICTLY FOLLOW\n"
    "BENEFICIARY'S PROFORMA INVOICE REF.NO. 786/S-13198-SOYPI-E\n"
    "DATED JAN 21, 2026 ON THE COMMERCIAL INVOICE.\n"
)


def main():
    print('=' * 78)
    print('P198dm dry-run — proforma date integrity on CI + PL')
    print('=' * 78)

    # ── Test 1: CI row + PL row, both with date mismatch → both flip
    print('\n[Test 1] Real job 08345848: CI row R0006 and PL row R0007 '
          'both must flip PASS->FAIL')
    rows = [
        {'row_id': 'R0006',
         'condition_text': ("Further details and specifications must "
                            "strictly follow Beneficiary's Proforma "
                            "Invoice Ref.No. 786/S-13198-SOYPI-E "
                            "dated Jan 21, 2026"),
         'document_checked': 'Commercial Invoice',
         'compliance': 'PASS'},
        {'row_id': 'R0007',
         'condition_text': ("Further details and specifications must "
                            "strictly follow Beneficiary's Proforma "
                            "Invoice Ref.No. 786/S-13198-SOYPI-E "
                            "dated Jan 21, 2026"),
         'document_checked': 'Packing List',
         'compliance': 'PASS'},
    ]
    flips = simulate_p198dm(rows, real_packets, LC_F45A_REAL)
    flipped_ids = {f[0] for f in flips}
    print(f"  flips: {flips}")
    t1_ok = ('R0006' in flipped_ids and 'R0007' in flipped_ids
             and rows[0]['compliance'] == 'FAIL'
             and rows[1]['compliance'] == 'FAIL')
    print(f"  RESULT: {'OK' if t1_ok else 'FAIL'}")

    # ── Test 2: PL matches LC date → stays PASS, CI mismatch still flipped
    print('\n[Test 2] PL date matches LC; CI date mismatched. PL stays '
          'PASS, CI still flips.')
    pkts = [
        {'document_type': 'Commercial Invoice',
         'refined_text': ("BENEFICIARY'S PROFORMA INVOICE REF.NO. "
                          "786/S-13198-SOYPI-E DATED FEB 18, 2026")},
        {'document_type': 'Packing List',
         'refined_text': ("BENEFICIARY'S PROFORMA INVOICE REF.NO. "
                          "786/S-13198-SOYPI-E DATED JAN 21, 2026")},
    ]
    rows = [
        {'row_id': 'R6', 'condition_text': 'STRICTLY AS PER PROFORMA',
         'document_checked': 'Commercial Invoice', 'compliance': 'PASS'},
        {'row_id': 'R7', 'condition_text': 'STRICTLY AS PER PROFORMA',
         'document_checked': 'Packing List', 'compliance': 'PASS'},
    ]
    flips = simulate_p198dm(rows, pkts, LC_F45A_REAL)
    flipped_ids = {f[0] for f in flips}
    t2_ok = (flipped_ids == {'R6'}
             and rows[0]['compliance'] == 'FAIL'
             and rows[1]['compliance'] == 'PASS')
    print(f"  flips: {flips}")
    print(f"  RESULT: {'OK' if t2_ok else 'FAIL'}")

    # ── Test 3: CI date matches, PL mismatched → only PL flips
    print('\n[Test 3] CI matches, PL mismatched → only PL flips')
    pkts = [
        {'document_type': 'Commercial Invoice',
         'refined_text': ("PROFORMA INVOICE REF.NO. 786/S-13198-SOYPI-E "
                          "DATED JAN 21, 2026")},
        {'document_type': 'Packing List',
         'refined_text': ("PROFORMA INVOICE REF.NO. 786/S-13198-SOYPI-E "
                          "DATED FEB 18, 2026")},
    ]
    rows = [
        {'row_id': 'R6', 'condition_text': 'PROFORMA',
         'document_checked': 'Commercial Invoice', 'compliance': 'PASS'},
        {'row_id': 'R7', 'condition_text': 'PROFORMA',
         'document_checked': 'Packing List', 'compliance': 'PASS'},
    ]
    flips = simulate_p198dm(rows, pkts, LC_F45A_REAL)
    flipped_ids = {f[0] for f in flips}
    t3_ok = (flipped_ids == {'R7'}
             and rows[0]['compliance'] == 'PASS'
             and rows[1]['compliance'] == 'FAIL')
    print(f"  flips: {flips}")
    print(f"  RESULT: {'OK' if t3_ok else 'FAIL'}")

    # ── Test 4: PL has no proforma reference → no citation, no flip
    print('\n[Test 4] PL has no proforma reference → row left alone')
    pkts = [
        {'document_type': 'Packing List',
         'refined_text': "PACKING LIST  Goods: Soybeans  Qty: 1000 MT"},
    ]
    rows = [{'row_id': 'R7', 'condition_text': 'PROFORMA',
             'document_checked': 'Packing List', 'compliance': 'PASS'}]
    flips = simulate_p198dm(rows, pkts, LC_F45A_REAL)
    t4_ok = (flips == [] and rows[0]['compliance'] == 'PASS')
    print(f"  flips: {flips}")
    print(f"  RESULT: {'OK' if t4_ok else 'FAIL'}")

    # ── Test 5: Both CI + PL match LC date exactly → no flips
    print('\n[Test 5] Both CI and PL match LC date → no flips')
    pkts = [
        {'document_type': 'Commercial Invoice',
         'refined_text': ("PROFORMA INVOICE REF.NO. 786/S-13198-SOYPI-E "
                          "DATED JAN 21, 2026")},
        {'document_type': 'Packing List',
         'refined_text': ("PROFORMA INVOICE REF.NO. 786/S-13198-SOYPI-E "
                          "DATED JAN 21, 2026")},
    ]
    rows = [
        {'row_id': 'R6', 'condition_text': 'PROFORMA',
         'document_checked': 'Commercial Invoice', 'compliance': 'PASS'},
        {'row_id': 'R7', 'condition_text': 'PROFORMA',
         'document_checked': 'Packing List', 'compliance': 'PASS'},
    ]
    flips = simulate_p198dm(rows, pkts, LC_F45A_REAL)
    t5_ok = (flips == []
             and rows[0]['compliance'] == 'PASS'
             and rows[1]['compliance'] == 'PASS')
    print(f"  flips: {flips}")
    print(f"  RESULT: {'OK' if t5_ok else 'FAIL'}")

    # ── Test 6: Different proforma ref on PL → ref doesn't match, no flip
    print('\n[Test 6] PL has different proforma ref → ref does not match, '
          'no flip')
    pkts = [
        {'document_type': 'Packing List',
         'refined_text': ("PROFORMA INVOICE REF.NO. UNRELATED-XYZ "
                          "DATED MAR 10, 2026")},
    ]
    rows = [{'row_id': 'R7', 'condition_text': 'PROFORMA',
             'document_checked': 'Packing List', 'compliance': 'PASS'}]
    flips = simulate_p198dm(rows, pkts, LC_F45A_REAL)
    t6_ok = (flips == [] and rows[0]['compliance'] == 'PASS')
    print(f"  flips: {flips}")
    print(f"  RESULT: {'OK' if t6_ok else 'FAIL'}")

    # ── Test 7: Already-FAIL row should NOT be touched
    print('\n[Test 7] Row already FAIL → not re-touched')
    rows = [{'row_id': 'R7', 'condition_text': 'PROFORMA',
             'document_checked': 'Packing List', 'compliance': 'FAIL'}]
    flips = simulate_p198dm(rows, real_packets, LC_F45A_REAL)
    t7_ok = (flips == [] and rows[0]['compliance'] == 'FAIL')
    print(f"  flips: {flips}")
    print(f"  RESULT: {'OK' if t7_ok else 'FAIL'}")

    all_ok = all([t1_ok, t2_ok, t3_ok, t4_ok, t5_ok, t6_ok, t7_ok])
    print('\n' + '=' * 78)
    print(f'OVERALL: {"OK — all 7 scenarios pass" if all_ok else "FAIL"}')
    print('=' * 78)
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
