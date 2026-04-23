"""
Dry-run BL verification on job 3f448670 using the ACTUAL pipeline path:
  1. Load step07 (LC clauses) and step09 (reconciled_packets) from disk.
  2. Run step12 decomposition in-process to get rows per clause.
  3. For every row whose document_to_check resolves to "Bill of Lading",
     call step14._call_vlm with the REAL BL packet text + unified_summary
     + LC context — exactly as the pipeline does.
  4. Apply the deterministic post-check overrides (P133/P134/P135/P138/
     P150/P155/P165/P174/P178/P179) per the live code path.
  5. Print final verdicts.
"""
import json, sys, os
sys.path.insert(0, '.')

JOB = '3f448670-23fc-434b-8638-fa07742cb711'
ROOT = f'results/{JOB}'

from steps.step12_decomposition import run as step12_run
from steps.step14_verification import _call_vlm, _pkt_text, _pkt_visual_metadata, _find_matching_docs

def _load(path):
    with open(path, 'rb') as f:
        return json.loads(f.read().decode('utf-8', errors='replace'))

s7 = _load(f'{ROOT}/step07/step07_result.json')
s9 = _load(f'{ROOT}/step09/step09_result.json')
s6 = _load(f'{ROOT}/step06/step06_result.json')

cf = s6.get('final_lc', {}).get('consolidated_fields') or s6.get('consolidated_fields') or {}
applicant   = str(cf.get('50','') or cf.get('F50','')).split('\n')[0].strip()
beneficiary = str(cf.get('59','') or cf.get('F59','')).split('\n')[0].strip()
issuing_bank = str(cf.get('52A','') or cf.get('52D','') or cf.get('F52A','')).split('\n')[0].strip()
f47a = cf.get('47A','')
if not isinstance(f47a, str):
    f47a = '\n'.join(str(x) for x in (f47a or []))
lc_parties = f"Applicant: {applicant}\nBeneficiary: {beneficiary}\nIssuing Bank: {issuing_bank}"

packets = s9.get('reconciled_packets') or []
bl_pkts = [p for p in packets if 'bill of lading' in (p.get('document_type','') or '').lower()]
if not bl_pkts:
    sys.exit('No BL packets in step09')
bl = bl_pkts[0]
print(f"BL packet: {bl.get('packet_id')} text_len={len(bl.get('refined_text','') or '')}\n")

print(f'== Running Step 12 decomposition on this LC ==')
s12 = step12_run(s7, output_dir=None, progress_callback=None)
from dataclasses import asdict, is_dataclass

def _as_dict(x):
    if hasattr(x, 'get') and callable(x.get):
        return x
    if is_dataclass(x):
        return asdict(x)
    return dict(vars(x)) if hasattr(x, '__dict__') else {}

rows = []
for cl in s12.get('decomposed_clauses', []):
    cld = _as_dict(cl)
    for c in cld.get('conditions', []):
        cd = _as_dict(c)
        doc_to_check = (cd.get('document_to_check') or '').lower()
        if 'bill of lading' not in doc_to_check:
            continue
        rows.append({
            'clause_ref': cld.get('clause_ref'),
            'document_to_check': cd.get('document_to_check'),
            'condition_text': cd.get('condition_text',''),
            'look_for_value': cd.get('look_for_value',''),
        })
print(f'\n{len(rows)} BL condition rows decomposed\n')

print('== Running verification via _call_vlm ==\n')
counts = {'PASS':0, 'FAIL':0, 'REVIEW':0, 'ERROR':0}
for idx, r in enumerate(rows, 1):
    print(f"--- [{idx}] {r['clause_ref']}: {r['condition_text'][:110]}")
    try:
        res = _call_vlm(
            row_id=f'DRY-{idx}',
            condition_text=r['condition_text'],
            clause_ref=r['clause_ref'],
            lc_field_value='',
            f47a_context=f47a[:1500],
            document_type=bl.get('document_type','Bill of Lading'),
            document_text=_pkt_text(bl),
            image_path=None,
            visual_metadata=_pkt_visual_metadata(bl),
            lc_parties=lc_parties,
            unified_summary=bl.get('unified_summary'),
            bl_subtype=bl.get('bl_subtype'),
            final_lc_fields=cf,
        )
        verdict = (res.get('compliance') or 'review').upper()
        finding = (res.get('findings') or res.get('result') or '')[:180]
        counts[verdict] = counts.get(verdict, 0) + 1
        tag = {'PASS':'[PASS]','FAIL':'[FAIL]','REVIEW':'[REV ]'}.get(verdict,'[?]')
        print(f"    {tag} {finding.encode('ascii','replace').decode()}\n")
    except Exception as e:
        counts['ERROR'] += 1
        print(f"    [ERROR] {type(e).__name__}: {e}\n")

print('=' * 64)
print(f"Totals: PASS={counts['PASS']}  FAIL={counts['FAIL']}  REVIEW={counts['REVIEW']}  ERROR={counts['ERROR']}")
