"""Strip UCP / ISBP citations from user-facing fields in all
stored job results. Mirror of the in-code line-by-line rules.
"""
import os, json, re, shutil, glob, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

USER_FIELDS = ('findings', 'result', 'found_text', 'condition',
               'condition_text', 'clause_text')

RULES = [
    (re.compile(r"Under\s+UCP\s+600\s+Art\s+\d+(?:\([a-z]\))?"
                r"(?:\s*/\s*\d+(?:\([a-z]\))?)?"
                r"(?:\s+and\s+ISBP\s+\d+\s*[¶§]?[A-Z\d]*)?,?\s*",
                re.IGNORECASE), ""),
    (re.compile(r"\(\s*UCP\s+600\s+Art\s+\d+(?:\([a-z]\))?"
                r"(?:\s*/\s*\d+(?:\([a-z]\))?)?\s*\)\s*",
                re.IGNORECASE), ""),
    (re.compile(r"per\s+UCP\s+600\s+Art\s+\d+(?:\([a-z]\))?"
                r"(?:\s*/\s*\d+(?:\([a-z]\))?)?\.?",
                re.IGNORECASE), ""),
    (re.compile(r"\bpermitted\s+under\s+UCP\s+600",
                re.IGNORECASE), "permitted"),
    (re.compile(r"Under\s+UCP\s+600\s*/?\s*ISBP\s+\d+\s*[¶§]?[A-Z\d]*\s*",
                re.IGNORECASE), ""),
    (re.compile(r"\bISBP\s+\d+\s*[¶§]?[A-Z\d]*", re.IGNORECASE), ""),
    (re.compile(r"\bUCP\s+600\s+Art\s+\d+(?:\([a-z]\))?",
                re.IGNORECASE), ""),
    (re.compile(r"\bUCP\s+600\b", re.IGNORECASE), ""),
    (re.compile(r"\(\s+\)"), ""),
    (re.compile(r"\s+\."), "."),
    (re.compile(r"\.\s*\."), "."),
    (re.compile(r"  +"), " "),
]


def clean(value):
    if not isinstance(value, str) or not value:
        return value, False
    new = value
    for pat, rep in RULES:
        new = pat.sub(rep, new)
    new = new.strip()
    return new, new != value


def patch_obj(obj):
    changed = 0
    if isinstance(obj, dict):
        for k, v in list(obj.items()):
            if isinstance(v, str) and k in USER_FIELDS:
                new, ch = clean(v)
                if ch:
                    obj[k] = new
                    changed += 1
            elif isinstance(v, (dict, list)):
                changed += patch_obj(v)
    elif isinstance(obj, list):
        for item in obj:
            changed += patch_obj(item)
    return changed


def main():
    files = 0
    fields = 0
    for path in sorted(glob.glob('results/*/step1[49]*/*_result.json') +
                        glob.glob('results/*/step20/*_result.json')):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            continue
        n = patch_obj(data)
        if n:
            bak = path + '.bak_strip_ucp'
            if not os.path.exists(bak):
                shutil.copy2(path, bak)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f'{path[len("results/"):]}: {n} field(s)')
            files += 1; fields += n
    print(f'\n{files} files patched, {fields} fields cleaned')


if __name__ == '__main__':
    main()
