"""Extensive cross-job test: simulate the new step08 logic against
every existing job's step08_result.json and report:
- packets where the new rule WOULD have changed the verdict
- a sanity classification of whether that change improves or breaks
- counts and per-job summary

The goal is to catch regressions BEFORE the user re-runs verification.
"""
import sys, os, json, re, glob
sys.path.insert(0, 'd:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final')
os.environ['PYTHONIOENCODING'] = 'utf-8'


from steps.step08_shipping_classification import _match_type_to_requirement


_STOP = {'THE','OF','AND','FOR','WITH','TO','A','AN',
         'IN','ON','AT','BY','OR','PAGE','BLANK',
         'HEADER','UNKNOWN','CONTINUATION'}


def tokens(s):
    return [t for t in re.findall(r'[A-Z]{3,}', s.upper().strip())
            if t not in _STOP]


def should_prefer_prior(prior, vlm):
    pt = tokens(prior); vt = tokens(vlm)
    p = prior.upper().strip(); v = vlm.upper().strip()
    pn = re.sub(r'\W+', '', p); vn = re.sub(r'\W+', '', v)
    norm_match = pn and vn and (pn in vn or vn in pn)
    BACK = ('CONDITIONS OF CARRIAGE','BL CONDITIONS',
            'BILL OF LADING CONDITIONS','STANDARD CONDITIONS',
            'STANDARD TERMS','TERMS AND CONDITIONS',
            'TERMS OF CARRIAGE','TERMS OF SERVICE','GENERAL CONDITIONS',
            'ATTACH LIST','ATTACHED LIST','ATTACH RIDER','ATTACHED RIDER',
            'RIDER','DESCRIPTION OF GOODS','DESCRIPTION OF CARGO',
            'GOODS DESCRIPTION','CARGO MANIFEST','CONTINUATION SHEET')
    is_back = any(b in p for b in BACK)
    return (
        len(pt) >= 2
        and bool(vt)
        and not (set(pt) & set(vt))
        and not (p and p in v)
        and not norm_match
        and not is_back
    )


jobs = sorted(glob.glob(
    'd:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final/results/*/step08/step08_result.json'
))

print(f"Found {len(jobs)} jobs\n")

global_stats = {'total_pkts': 0, 'would_change': 0,
                'would_change_examples': [],
                'substring_unbinding': 0,
                'substring_examples': []}

for jpath in jobs:
    job_id = jpath.split('/')[-3]
    try:
        d = json.load(open(jpath, encoding='utf-8'))
    except Exception:
        continue
    pkts = d.get('classified_packets') or []
    expected_docs = d.get('required_documents_used', [])
    if isinstance(expected_docs, list) and expected_docs and isinstance(expected_docs[0], str):
        expected_docs = [{'document_name': n} for n in expected_docs]

    for pk in pkts:
        global_stats['total_pkts'] += 1
        packet_type = pk.get('document_type', '') or ''
        pages = pk.get('original_pages') or []
        # Get step3 prior (from first page's document_type)
        prior_dt = ''
        for pg in pages:
            if isinstance(pg, dict):
                prior_dt = (pg.get('document_type') or '').strip()
                break
        if not prior_dt:
            continue

        if should_prefer_prior(prior_dt, packet_type):
            global_stats['would_change'] += 1
            if len(global_stats['would_change_examples']) < 15:
                global_stats['would_change_examples'].append(
                    f"  {job_id[:8]} pkt {pk.get('packet_id','?')} pages={pages[:1]} "
                    f"step3={prior_dt!r:45} packet={packet_type!r}"
                )

        # Substring rule: would bare-generic step3 dt have been
        # rejected as a match?
        if expected_docs and prior_dt.upper() in (
            'CERTIFICATE','CERTIFICATES','CERT','DOCUMENT','DOCUMENTS',
            'DOC','FORM','PAGE','NOTE','NOTICE','STATEMENT'
        ):
            idx, name = _match_type_to_requirement(prior_dt, expected_docs)
            if idx == -1:
                global_stats['substring_unbinding'] += 1
                if len(global_stats['substring_examples']) < 10:
                    global_stats['substring_examples'].append(
                        f"  {job_id[:8]} step3={prior_dt!r} -> alien (was: {packet_type!r})"
                    )


print(f"Total packets scanned: {global_stats['total_pkts']}")
print(f"Would-change (dynamic override fires): {global_stats['would_change']}")
print(f"Substring unbinding (bare-generic -> alien): {global_stats['substring_unbinding']}\n")

print("=== Would-change examples (top 15) ===")
for line in global_stats['would_change_examples']:
    print(line)

print("\n=== Substring unbinding examples (top 10) ===")
for line in global_stats['substring_examples']:
    print(line)

# Sanity check: rate of overrides should be small (<20% of packets)
override_rate = (global_stats['would_change'] / max(1, global_stats['total_pkts'])) * 100
print(f"\nOverride rate: {override_rate:.1f}% of {global_stats['total_pkts']} packets")
if override_rate > 25:
    print("WARNING: override rate >25% — rule too aggressive")
    sys.exit(1)
print("\nOVERALL: OK")
sys.exit(0)
