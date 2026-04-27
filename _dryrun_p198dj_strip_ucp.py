"""
P198dj dry-run — UCP / ISBP citations stripped from user-facing
findings.

Scans:
  • In-code finding strings in step14_verification, step14_implicit,
    step06_final_lc, step08_shipping_classification — should not
    contain "UCP 600 Art X" / "ISBP 745" / "Under UCP 600..."
    inside f-string content destined for user-visible report.
  • All stored step14 / step19 / step20 result JSONs — same fields.
"""
import os, re, json, glob, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')


_FORBID = re.compile(r"\b(?:UCP\s*600|ISBP\s*\d+|UCP\s*Art|ISBP\s*[¶§])",
                     re.IGNORECASE)
USER_FIELDS = ('findings', 'result', 'found_text', 'condition',
               'condition_text', 'clause_text')


def scan_obj(obj, hits):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str) and k in USER_FIELDS:
                m = _FORBID.search(v)
                if m:
                    hits.append((k, m.group(0), v[:120]))
            elif isinstance(v, (dict, list)):
                scan_obj(v, hits)
    elif isinstance(obj, list):
        for item in obj:
            scan_obj(item, hits)


def main():
    # Stored data check
    stored_hits = 0
    stored_files = 0
    for path in (sorted(glob.glob('results/*/step1[49]/*_result.json')) +
                  sorted(glob.glob('results/*/step20/*_result.json'))):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            continue
        hits = []
        scan_obj(data, hits)
        if hits:
            stored_files += 1
            stored_hits += len(hits)
            print(f'!! {path}: {len(hits)} hit(s)')
            for k, frag, ctx in hits[:3]:
                print(f'    {k}: ...{frag}... | {ctx[:80]}')

    # In-code check — scan finding strings in source files
    src_hits = 0
    src_files_with_hits = 0
    for fp in ('steps/step14_verification.py',
                'steps/step14_implicit.py',
                'steps/step06_final_lc.py',
                'steps/step08_shipping_classification.py'):
        if not os.path.exists(fp): continue
        with open(fp, 'r', encoding='utf-8') as f: src = f.read()
        # Find f-string content lines that mention UCP/ISBP and look
        # like user-facing strings (typical: f"...UCP..." inside
        # row['findings'] / row.set("findings", ...) / msg = f"...")
        in_code_hits = []
        for i, line in enumerate(src.split('\n'), 1):
            # Skip pure comment lines
            stripped = line.lstrip()
            if stripped.startswith('#'): continue
            # Skip docstrings — heuristic: triple-quoted lines we won't
            # recognise reliably; let it slip and scan anyway.
            if _FORBID.search(line):
                # Differentiate code structures (e.g. UCP_VARIABLE) from
                # English strings: only report if appearance is in a
                # string-looking context (quotes nearby).
                if '"' in line or "'" in line:
                    in_code_hits.append((i, line.strip()[:120]))
        if in_code_hits:
            src_files_with_hits += 1
            src_hits += len(in_code_hits)
            print(f'\nIN-CODE {fp}: {len(in_code_hits)} hit(s)')
            for ln, ctx in in_code_hits[:6]:
                print(f'    L{ln}: {ctx}')

    print('\n' + '=' * 78)
    print(f'Stored data: {stored_hits} hit(s) in {stored_files} file(s)')
    print(f'In-code   : {src_hits} hit(s) in {src_files_with_hits} file(s)')
    print('=' * 78)
    return stored_hits == 0 and src_hits == 0


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
