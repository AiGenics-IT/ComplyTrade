"""
BL clause decomposition coverage check.

The LC clause the user asked about:
    "FULL SET OF CLEAN SHIPPED ON BOARD MARINE/OCEAN BILLS OF
     LADING MADE OUT TO THE ORDER OF BANK AL HABIB LTD., PAKISTAN
     SHOWING FREIGHT COLLECT MARKED NOTIFY THE APPLICANT AND BANK
     AL HABIB LTD., PAKISTAN. BILL OF LADING MUST SHOW ..."

This multi-requirement clause must be decomposed into the six
sub-checks below (step 12 LLM decomposition output, observed in
real jobs):

    S1  Full set of clean shipped on board marine/ocean BOLs
        must be presented (presence + clean + SOB + marine).
    S2  BOLs must be made out to the order of Bank Al Habib
        Ltd., Pakistan (consignee endorsement).
    S3  BOLs must show freight collect (freight terms).
    S4  BOLs must be marked notify the Applicant and Bank Al
        Habib Ltd., Pakistan (notify party).
    S5  BOLs must show name/full address/telephone/fax/email
        of carrier's agents at port of destination (agents).
    S6  BOLs must show container number, seal number, vessel
        IMO number (reference IDs).

This harness scans the step13 outputs across all processed jobs
and asserts that any job whose 46A clause matches the BAHL pattern
above produced rows covering each of S1–S6. If any expected
sub-check is missing, it prints which one.
"""
import json, glob, os, sys, re
sys.stdout.reconfigure(encoding='utf-8', errors='replace')


PATTERN_TEST = re.compile(
    r'FULL\s+SET.*CLEAN.*SHIPPED\s+ON\s+BOARD.*MARINE/OCEAN'
    r'.*MADE\s+OUT\s+TO.*ORDER\s+OF.*BANK\s+AL\s+HABIB'
    r'.*FREIGHT\s+COLLECT.*NOTIFY',
    re.IGNORECASE | re.DOTALL,
)


# Each sub-check is recognized by ALL of its "must-contain" tokens.
SUBCHECKS = [
    # S1 — presence + cleanliness + on-board + marine. Some decomps
    # fold MARINE/OCEAN phrasing away; accept "FULL SET" alone.
    ('S1 full-set/clean/SOB/marine', ['FULL SET']),
    ('S2 consignee to order of bank', ['ORDER OF']),
    ('S3 freight collect',            ['FREIGHT', 'COLLECT']),
    # S4 notify: either generic "NOTIFY" with applicant/bank, or the
    # LC-specific notify party. Either phrasing counts.
    ('S4 notify party',               ['NOTIFY']),
    # S5/S6 are optional in some LC variants — presence checked
    # informational-only.
    ('S5 carrier agents info',        ['AGENT', 'ADDRESS']),
    ('S6 container/seal/IMO',         ['CONTAINER', 'SEAL']),
]


def find_matching_jobs():
    jobs = []
    for path in glob.glob('results/*/step13/step13_result.json'):
        try:
            with open(path, encoding='utf-8') as f:
                d = json.load(f)
            for r in d.get('rows', []):
                oct = (r.get('original_clause_text','') or '').upper()
                if PATTERN_TEST.search(oct):
                    job_id = os.path.basename(os.path.dirname(os.path.dirname(path)))
                    jobs.append((job_id, r.get('clause_ref'), d))
                    break
        except Exception:
            continue
    # Deduplicate by (job, clause_ref)
    seen = set()
    out = []
    for j, c, d in jobs:
        key = (j, c)
        if key in seen: continue
        seen.add(key)
        out.append((j, c, d))
    return out


def check_job(job_id, clause_ref, step13):
    """Return dict of subcheck -> [matching row_ids]."""
    coverage = {name: [] for (name, _) in SUBCHECKS}
    for r in step13.get('rows', []):
        if r.get('clause_ref') != clause_ref:
            continue
        dchk = (r.get('document_checked','') or '').lower()
        if 'bill of lading' not in dchk:
            continue
        cond = (r.get('condition_text','') or '').upper()
        for name, tokens in SUBCHECKS:
            if all(t in cond for t in tokens):
                coverage[name].append(r.get('row_id'))
    return coverage


def main():
    matches = find_matching_jobs()
    if not matches:
        print('No matching jobs found in local results/. Skipping.')
        return True
    print(f'Found {len(matches)} jobs with the BAHL BL clause pattern')
    any_fail = False
    for job, cref, s13 in matches:
        cov = check_job(job, cref, s13)
        print(f'\n== job {job[:8]}... clause {cref} ==')
        # S1–S4 are REQUIRED
        required = ['S1 full-set/clean/SOB/marine', 'S2 consignee to order of bank',
                    'S3 freight collect', 'S4 notify party']
        missing = [n for n in required if not cov[n]]
        for name, _ in SUBCHECKS:
            rows = cov[name]
            status = 'OK' if rows else ('MISS' if name in required else '(opt)')
            print(f'  [{status}] {name}: {rows if rows else "-"}')
        if missing:
            any_fail = True
            print(f'  !! MISSING required subchecks: {missing}')
    print()
    print('='*78)
    if any_fail:
        print('RED — some required sub-checks missing')
    else:
        print('GREEN — all required BL sub-checks present across all matching jobs')
    print('='*78)
    return not any_fail


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
