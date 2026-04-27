"""
Multi-job real-data sweep for the recent P198 fixes:

  • P198cy — every clause sub-row visible when document missing
  • P198cz — strict-content guard on Shipping Company Certificate
  • P198da — F47A "needs evidence" recognizer (charges + SWIFT)
  • P198db — MT799 / MT999 routed to shipping packets
  • P198dc — F45A goods/quantity fan-out to Packing List
  • P198dd — F48 "BUT WITHIN EXPIRY" relaxation + no auto LC-expired
             cross-link, "021"->"21" formatting fix

Walks every job under results/ that has step09 + step13 + step14
results, applies the deterministic recognizers / guards in dry-run
mode, and reports.
"""
import os, sys, json, re
from datetime import date

ROOT = 'results'
sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def load(p):
    try:
        with open(p, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def is_iata_awb_original(text):
    if not text: return False
    up = text.upper()
    return ('COPIES 1, 2 AND 3 OF THIS AIR WAYBILL ARE ORIGINALS' in up
            or 'ORIGINALS AND HAVE THE SAME VALIDITY' in up
            or re.search(r'\bORIGINAL\s+(?:1|2|3)\b', up) is not None)


def has_swift(packets):
    for p in packets or []:
        if not isinstance(p, dict): continue
        dt = (p.get('document_type') or '').lower()
        if any(k in dt for k in ('mt799', 'mt 799', 'mt999', 'mt 999',
                                  'fin.799', 'fin.999', 'free format',
                                  'authenticated swift', 'swift advice',
                                  'swift message')):
            return p
        if str(p.get('source_mt') or '').upper() in ('MT799', 'MT999'):
            return p
        if p.get('is_swift_advice_copy'):
            return p
    return None


def has_dr(packets):
    for p in packets or []:
        if not isinstance(p, dict): continue
        dt = (p.get('document_type') or '').lower()
        if any(k in dt for k in ('document remittance', 'documentary remittance',
                                  'covering schedule', 'covering letter',
                                  'cover letter', 'cover schedule',
                                  'bills schedule', 'forwarding schedule')):
            return p
    return None


_DR_CHARGES = re.compile(
    r'(?:ALL\s+)?(?:OUR\s+)?CHARGES?\s+'
    r'(?:AND\s+(?:ALL\s+)?(?:OUR\s+)?CHARGES?\s+OF\s+'
    r'(?:THE\s+)?ADVISING\s+BANK\s+)?'
    r'(?:ARE\s+|TO\s+BE\s+)?'
    r'(?:PAID|BORNE|FOR\s+(?:THE\s+)?ACCOUNT)\s+'
    r'(?:OF|BY)\s+(?:THE\s+)?BENEFICIARY',
    re.IGNORECASE | re.DOTALL,
)


def _parse_iso(s):
    if not s: return None
    m = re.search(r'(\d{4})[-./](\d{1,2})[-./](\d{1,2})', str(s))
    if not m: return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except Exception:
        return None


# Counters
checks = {
    'cy_kept_rows': 0, 'cy_dropped_rows': 0,
    'cz_scc_passes': 0, 'cz_scc_strict_fail': 0,
    'da_47a_charges': 0, 'da_47a_swift': 0,
    'db_swift_in_ship': 0, 'db_swift_in_mt': 0,
    'dc_pl_clones': 0,
    'dd_expiry_bound_pass': 0, 'dd_021_format_bug': 0,
    'jobs_scanned': 0,
}


def scan_job(job_dir):
    s9 = load(os.path.join(job_dir, 'step09', 'step09_result.json'))
    s13 = load(os.path.join(job_dir, 'step13', 'step13_result.json'))
    s14 = load(os.path.join(job_dir, 'step14', 'step14_result.json'))
    s14b = load(os.path.join(job_dir, 'step14b', 'step14b_result.json'))
    s6 = load(os.path.join(job_dir, 'step06', 'step06_result.json'))
    if not (s13 and s14):
        return None
    pkts = (s9 or {}).get('reconciled_packets') or (s9 or {}).get('packets') or \
           (s9 or {}).get('classified_packets') or []
    s13_rows = s13.get('rows') or []
    s14_rows = s14.get('rows') or []
    s13_ids = {r.get('row_id') for r in s13_rows}
    s14_ids = {r.get('row_id') for r in s14_rows}
    dropped = s13_ids - s14_ids
    # P198cy: track dropped vs kept counts
    checks['cy_dropped_rows'] += len(dropped)
    checks['cy_kept_rows'] += len(s14_ids & s13_ids)

    # P198cz: any SCC PASS that lacks the literal phrase but condition demands it
    for r in s14_rows:
        if (r.get('document_checked') or '').lower() != 'shipping company certificate':
            continue
        comp = (r.get('compliance') or '').upper()
        cond = (r.get('condition_text') or '')
        if not cond: continue
        if comp != 'PASS': continue
        # Pull the corresponding SCC packet text
        scc_pkt = next((p for p in pkts
                        if (p.get('document_type') or '').lower() == 'shipping company certificate'),
                       None)
        scc_text = (scc_pkt or {}).get('document_text', '') if scc_pkt else ''
        scc_up = scc_text.upper()
        # Look for the three demand patterns
        if 'PAKISTANI MARITIME RULES' in cond.upper() or 'PORT REGULATIONS' in cond.upper():
            present = bool(re.search(r'PAKISTAN(?:I)?\s+MARITIME\s+RULES?'
                                      r'|MARITIME\s+RULES?\s+AND\s+PORT\s+REGULATIONS?',
                                      scc_up))
            if not present:
                checks['cz_scc_strict_fail'] += 1
            else:
                checks['cz_scc_passes'] += 1
        if 'INSTITUTE CLASSIFICATION CLAUSE' in cond.upper():
            present = bool(re.search(r'INSTITUTE\s+CLASSIFICATION\s+CLAUSE', scc_up))
            if not present:
                checks['cz_scc_strict_fail'] += 1
            else:
                checks['cz_scc_passes'] += 1
        if any(k in cond.upper() for k in ('APPROXIMATE DATE OF ARRIVAL',
                                             'ESTIMATED TIME OF ARRIVAL',
                                             'EXPECTED ARRIVAL', 'ETA')):
            present = bool(re.search(r'\b(?:ETA|ESTIMATED\s+(?:TIME|DATE)\s+OF\s+ARRIVAL'
                                      r'|EXPECTED\s+ARRIVAL|APPROXIMATE\s+DATE\s+OF\s+ARRIVAL'
                                      r'|DATE\s+OF\s+ARRIVAL\s+AT|ARRIVAL\s+AT\s+'
                                      r'(?:THE\s+)?(?:PORT\s+OF\s+)?DESTINATION)\b',
                                      scc_up))
            if not present:
                checks['cz_scc_strict_fail'] += 1
            else:
                checks['cz_scc_passes'] += 1

    # P198da: F47A charges + SWIFT detection
    for r in s13_rows:
        cref = (r.get('clause_ref','') or '').upper()
        if not (cref.startswith('47A') or cref.startswith('47B')):
            continue
        cond = r.get('condition_text','') or ''
        cond_u = cond.upper()
        if 'CHARGES' in cond_u and 'BENEFICIARY' in cond_u and (
                'CERTIFY' in cond_u or 'SCHEDULE' in cond_u
                or 'NEGOTIATING BANK' in cond_u):
            checks['da_47a_charges'] += 1
        if any(k in cond_u for k in ('AUTHENTICATED SWIFT', 'VIA SWIFT',
                                       'BY SWIFT', 'MT 799', 'MT799',
                                       'MT 999', 'MT999',
                                       'FREE FORMAT MESSAGE',
                                       'SWIFT MESSAGE MUST ACCOMPANY')) \
                and ('NEGOTIATING' in cond_u or 'ADVISE' in cond_u
                     or 'ACCOMPANY' in cond_u or 'ADVICE' in cond_u):
            checks['da_47a_swift'] += 1

    # P198db: SWIFT packets present
    if has_swift(pkts):
        checks['db_swift_in_ship'] += 1

    # P198dc: F45A clones to Packing List
    for r in s13_rows:
        if (r.get('document_checked') or '').lower() != 'packing list':
            continue
        if '45A' in (r.get('field_tag','') or '').upper() \
           or '45A' in (r.get('clause_ref','') or '').upper():
            checks['dc_pl_clones'] += 1

    # P198dd: 021-format bug + expiry-bound check from step14b
    if s14b:
        for c in s14b.get('checks', []):
            cond = (c.get('condition') or '')
            res = (c.get('result') or '')
            if re.search(r'\b021\s*days\b', cond + ' ' + res, re.I) \
               or re.search(r'\bDocuments within\s+0\d{2}\s+days\b', cond):
                checks['dd_021_format_bug'] += 1
            f48_v = ((s6 or {}).get('consolidated_fields', {}) or {}).get('48','') or ''
            if (c.get('check_id') == 'presentation_period'
                and c.get('compliance') == 'FAIL'
                and re.search(r'BUT\s+WITHIN\s+EXPIRY|WITHIN\s+(?:LC\s+|L/?C\s+)?(?:EXPIRY|VALIDITY)',
                              f48_v.upper())):
                expiry = ((s6 or {}).get('consolidated_fields', {}) or {}).get('31D','')
                exp_d = _parse_iso(expiry)
                m_pres = re.search(r'(\d{4}-\d{2}-\d{2})', c.get('findings', '') or '')
                pres_d = _parse_iso(m_pres.group(1)) if m_pres else None
                if exp_d and pres_d and pres_d <= exp_d:
                    checks['dd_expiry_bound_pass'] += 1


def main():
    if not os.path.isdir(ROOT):
        print(f'No {ROOT}/ directory')
        return False
    for entry in sorted(os.listdir(ROOT)):
        full = os.path.join(ROOT, entry)
        if not os.path.isdir(full):
            continue
        scan_job(full)
        checks['jobs_scanned'] += 1

    print('\n=== Multi-job sweep ===')
    print(f"Jobs scanned: {checks['jobs_scanned']}\n")
    print('--- P198cy (every clause row visible) ---')
    print(f"  Rows kept in step14: {checks['cy_kept_rows']}")
    print(f"  Rows dropped (s13 -> s14): {checks['cy_dropped_rows']}")
    print(f"  -> dropped count should be small / mostly N/A artifacts")
    print()
    print('--- P198cz (SCC strict-content) ---')
    print(f"  SCC PASS rows where literal evidence present: {checks['cz_scc_passes']}")
    print(f"  SCC PASS rows where evidence MISSING (would be flipped to FAIL): {checks['cz_scc_strict_fail']}")
    print()
    print('--- P198da (F47A needs-evidence) ---')
    print(f"  Charges rows detected in step13: {checks['da_47a_charges']}")
    print(f"  SWIFT-advice rows detected in step13: {checks['da_47a_swift']}")
    print()
    print('--- P198db (SWIFT packets visible to verifier) ---')
    print(f"  Jobs with at least one SWIFT packet on shipping side: {checks['db_swift_in_ship']}")
    print()
    print('--- P198dc (F45A -> Packing List clones) ---')
    print(f"  Packing List rows with field_tag=45A in step13: {checks['dc_pl_clones']}")
    print('  (existing jobs were built before P198dc, so most show 0 unless re-run)')
    print()
    print('--- P198dd (expiry-bound + 021 format) ---')
    print(f"  Step14b rows still showing the 021/015 zero-padded format: {checks['dd_021_format_bug']}")
    print(f"  -> existing jobs predate the fix; new runs use plain '21'/'15'")
    print(f"  Late-presentation FAIL rows that SHOULD become PASS via expiry-bound: {checks['dd_expiry_bound_pass']}")
    print()
    print('=' * 78)
    print('Multi-job sweep complete — counters above quantify pre-fix state.')
    print('Per-job patches applied separately for affected jobs.')
    print('=' * 78)
    return True


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
