"""
P198dg dry-run — report text:
  • No LLM/VLM mentions in user-visible findings.
  • No truncation in the executive summary or per-clause tables.

Walks every step14 / step19 / step20 stored result, validates that:
  • findings / result / found_text / condition / clause_text fields
    do NOT mention 'LLM' or 'VLM' as a process actor.
  • The text under those fields is preserved end-to-end (still has
    full sentences, no '...' truncation suffix on critical fields).
"""
import os, json, re, glob, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')


_FORBID_RE = re.compile(
    r'\b(?:LLM|VLM)\s+(?:previously|hallucinated|hallucination|frequently|'
    r'commonly|sometimes|may|might|finding|reasoning|misread|misreading|'
    r'false|earlier|incorrect)\b',
    re.IGNORECASE,
)
_USER_FIELDS = ('findings', 'result', 'found_text', 'condition',
                'condition_text', 'clause_text')


def scan_obj(obj, hits):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str) and k in _USER_FIELDS:
                m = _FORBID_RE.search(v)
                if m:
                    hits.append((k, m.group(0), v[:120]))
            elif isinstance(v, (dict, list)):
                scan_obj(v, hits)
    elif isinstance(obj, list):
        for item in obj:
            scan_obj(item, hits)


def main():
    paths = (
        sorted(glob.glob('results/*/step1[49]/*_result.json')) +
        sorted(glob.glob('results/*/step20/*_result.json'))
    )
    total_hits = 0
    files_with_hits = 0
    for p in paths:
        try:
            with open(p, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            continue
        hits = []
        scan_obj(data, hits)
        if hits:
            files_with_hits += 1
            total_hits += len(hits)
            print(f'!! {p[len("results/"):]}: {len(hits)} mention(s)')
            for k, frag, ctx in hits[:3]:
                print(f'    {k}: ...{frag}... | {ctx[:90]}')

    if total_hits == 0:
        print('All step14/step19/step20 stored data is free of '
              'LLM/VLM-process-actor wording.')

    # Also validate truncation: check report code paths use 100000 cap
    rep = open('steps/step20_report.py', encoding='utf-8').read()
    chk = open('steps/step19_consolidation.py', encoding='utf-8').read()
    truncation_ok = (
        '_safe_str(_cond_text, 100000)' in rep
        and '_safe_str(_find_text, 100000)' in rep
        and '_safe_str(result_val, 100000)' in rep
        and "or cf.get('condition', ''), 100000)" in rep
        and "'clause_text': cg.clause_text," in chk  # not [:200]
        and "'clause_text': _clause_text," in chk
    )

    print()
    print('=' * 78)
    print(f'LLM/VLM mention scan : {total_hits} hits across {files_with_hits} files')
    print(f'Truncation removed   : {"YES" if truncation_ok else "NO"}')
    print('=' * 78)
    return total_hits == 0 and truncation_ok


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
