"""
P198ep real-data sweep — walk every job's step03 result and report
"SPECIFICATION OF CARGO" pages and where they end up.

This shows which jobs WOULD benefit from the P198ep fix (i.e. jobs
where SPEC pages got absorbed into a non-BL document type by the
old inheritance rule). The fix preserves the SPEC label and routes
the page to the nearest BL via Rule 1b smart-merge.

The script also reports any jobs where SPEC pages successfully landed
in a BL packet (no harm there) and any where they ended up in a wrong
doc-type packet (Commercial Invoice / Covering Letter / Shipping
Company Cert / etc.) — those are the cases the fix targets.
"""
import os, json, sys

ROOT = 'd:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final'
RESULTS = os.path.join(ROOT, 'results')

if not os.path.isdir(RESULTS):
    print('No results/ directory'); sys.exit(0)

# Phrases that the VLM emits for BL cargo-spec rider sheets
SPEC_PHRASES = [
    'specification of cargo', 'cargo specification',
    'attached specification', 'specification sheet',
    'cargo specification sheet',
]

def is_spec(dt):
    dtl = (dt or '').lower().strip()
    return any(p in dtl for p in SPEC_PHRASES)

def is_bl(dt):
    dtl = (dt or '').lower().strip()
    return 'bill of lading' in dtl or dtl in {'b/l','bl'}

total_jobs = 0
jobs_with_spec_pages = 0
total_spec_pages = 0
spec_in_bl   = 0
spec_in_ci   = 0
spec_in_cl   = 0
spec_in_scc  = 0
spec_in_other = 0
problem_jobs = []

for job_id in sorted(os.listdir(RESULTS))[:200]:
    job_dir = os.path.join(RESULTS, job_id)
    if not os.path.isdir(job_dir):
        continue
    s3_file = os.path.join(job_dir, 'step03', 'step03_result.json')
    if not os.path.exists(s3_file):
        continue
    total_jobs += 1
    try:
        with open(s3_file, 'r', encoding='utf-8') as f:
            s3 = json.load(f)
    except Exception:
        continue

    # Look at the per-page classifications: even though step03 may have
    # already overwritten the doc_type via the OLD CONTINUATION fix, we
    # can detect the issue by looking at original_pages inside packets
    # OR at the doc_hint / multiple_instruments fields the VLM emitted.
    # The most reliable signal is the doc_hint string which the VLM
    # writes BEFORE the inheritance pass.
    spec_pages_in_job = []
    cls_list = s3.get('classifications', [])
    for cls in cls_list:
        pn = cls.get('page_number')
        dt = cls.get('document_type', '')
        hint = (cls.get('doc_hint') or '').lower()
        # Direct hit on doc_type
        if is_spec(dt):
            spec_pages_in_job.append((pn, dt, 'direct'))
            continue
        # Inherited cases — the doc_hint may still mention specification
        if 'specification' in hint and 'cargo' in hint:
            spec_pages_in_job.append((pn, dt, f'hint:{hint[:60]}'))

    if not spec_pages_in_job:
        continue
    jobs_with_spec_pages += 1
    total_spec_pages += len(spec_pages_in_job)

    # Classify how each SPEC page ended up
    job_problem = False
    for (pn, dt, src) in spec_pages_in_job:
        # Where did the page land in packets?
        end_dt = ''
        for pkt in s3.get('packets', []):
            if pn in pkt.get('page_numbers', []):
                end_dt = pkt.get('document_type', '')
                break
        end_low = (end_dt or '').lower()
        if 'bill of lading' in end_low:
            spec_in_bl += 1
        elif 'commercial invoice' in end_low or 'invoice' in end_low:
            spec_in_ci += 1
            job_problem = True
        elif 'covering letter' in end_low or 'cover letter' in end_low or \
             'covering schedule' in end_low or 'document remittance' in end_low:
            spec_in_cl += 1
            job_problem = True
        elif 'shipping company' in end_low:
            spec_in_scc += 1
            job_problem = True
        else:
            spec_in_other += 1
            if end_low and not is_spec(end_dt) and not is_bl(end_dt):
                job_problem = True
    if job_problem:
        problem_jobs.append((job_id, spec_pages_in_job))

print('=' * 78)
print(f"  Jobs scanned                      : {total_jobs}")
print(f"  Jobs with SPECIFICATION OF CARGO  : {jobs_with_spec_pages}")
print(f"  Total SPEC pages found            : {total_spec_pages}")
print()
print(f"  SPEC ended up in BL packet         : {spec_in_bl:4d}  (correct — fix harmless)")
print(f"  SPEC ended up in Commercial Invoice: {spec_in_ci:4d}  (BUG — fix corrects)")
print(f"  SPEC ended up in Covering Letter   : {spec_in_cl:4d}  (BUG — fix corrects)")
print(f"  SPEC ended up in Shipping Co Cert  : {spec_in_scc:4d}  (BUG — fix corrects)")
print(f"  SPEC ended up in OTHER doc type    : {spec_in_other:4d}  (review case-by-case)")
print()
print(f"  Jobs with BUGGY SPEC attribution   : {len(problem_jobs)}")
print('=' * 78)
if problem_jobs:
    print()
    print('Sample jobs (first 10):')
    for job_id, pgs in problem_jobs[:10]:
        print(f"  {job_id}:")
        for (pn, dt, src) in pgs[:5]:
            print(f"    page {pn}: dt={dt!r}  ({src})")

print()
print('OVERALL: real-data sweep completed — {0} pages would route to nearest BL'.format(spec_in_ci + spec_in_cl + spec_in_scc + spec_in_other))
sys.exit(0)
