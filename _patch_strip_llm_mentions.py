"""Strip LLM / VLM mentions from user-facing findings / result /
condition / clause_text / found_text fields across all stored
job results. This is a one-time cleanup so the report download
no longer leaks technical jargon to the customer.

Replaces phrases like:
  "The LLM previously PASSed this check ..." → ""
  "The earlier LLM finding compared ..."     → "The earlier finding compared ..."
  "VLM false FAIL"                            → "false FAIL"
  "LLM's 'no date found' was incorrect."     → ""
  "Original LLM finding said not found"      → ""
  "(P135 override)"  / "(P138 override)"     → ""

Walks step14, step14b, step19, step20 results in every
results/* directory. Backups are written next to each file.
"""
import os, json, re, shutil, glob
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Per-occurrence rewrite rules. Each entry is (pattern, replacement).
RULES = [
    # P198cz "The LLM previously PASSed this check ..."
    (re.compile(
        r'\.?\s*The\s+LLM\s+previously\s+PASSed\s+this\s+check\s+'
        r'without\s+literal\s+evidence\s+on\s+the\s+document\s*;\s*',
        re.IGNORECASE), '. '),
    # "Original LLM finding said not found, but..."
    (re.compile(
        r'\s*Original\s+LLM\s+finding\s+said\s+not\s+found[,.]?\s+but\s+',
        re.IGNORECASE), ' '),
    # "LLM's 'no date found' was incorrect. (P138 override)"
    (re.compile(
        r"\s*LLM'?s?\s+['\"]?no\s+date\s+found['\"]?\s+was\s+incorrect\.?\s*",
        re.IGNORECASE), ' '),
    # "(P135 override)" / "(P138 override)" / "(P170 override)"
    (re.compile(r'\s*\(\s*P\d+\s+override\s*\)\s*', re.IGNORECASE), ' '),
    # "the earlier LLM finding compared" → "the earlier finding compared"
    (re.compile(r'\bthe\s+earlier\s+LLM\s+finding\b', re.IGNORECASE),
     'the earlier finding'),
    # "VLM false FAIL corrected" → "false FAIL corrected"
    (re.compile(r'\bVLM\s+false\s+FAIL\b', re.IGNORECASE), 'false FAIL'),
    # "earlier LLM reasoning citing form type ..."
    (re.compile(r'\(\s*P\d+\s*[—\-]\s*earlier\s+LLM\s+reasoning\s+citing\s+'
                r'form\s+type\s*/\s*blank\s+back\s*/\s*house\s*/\s*claused\s+'
                r'is\s+irrelevant\s+for\s+staleness\.?\s*\)\s*',
                re.IGNORECASE),
     'Form type / blank back / house / claused signals are irrelevant for staleness.'),
    # Generic "the LLM" → "earlier check"
    (re.compile(r'\bthe\s+LLM\s+previously\b', re.IGNORECASE), 'previously'),
    (re.compile(r'\bThe\s+LLM\b', re.IGNORECASE), 'The earlier check'),
    (re.compile(r'\bthe\s+LLM\b', re.IGNORECASE), 'the earlier check'),
    (re.compile(r'\bThe\s+VLM\b', re.IGNORECASE), 'The earlier check'),
    (re.compile(r'\bthe\s+VLM\b', re.IGNORECASE), 'the earlier check'),
    # Standalone "LLM" / "VLM" in narrative text
    (re.compile(r'\bLLM\s+(?:hallucinated|hallucination|misread|misreading|'
                r'frequently|commonly|sometimes|may|might)\s+',
                re.IGNORECASE), ''),
    (re.compile(r'\bVLM\s+(?:hallucinated|hallucination|misread|misreading|'
                r'frequently|commonly|sometimes|may|might)\s+',
                re.IGNORECASE), ''),
    # Tidy double spaces / orphan punctuation
    (re.compile(r'\s+\.'), '.'),
    (re.compile(r'\s{2,}'), ' '),
]

USER_FIELDS = ('findings', 'result', 'found_text', 'condition',
               'condition_text', 'clause_text')


def clean(value):
    if not isinstance(value, str) or not value:
        return value, False
    new = value
    for pat, rep in RULES:
        new = pat.sub(rep, new)
    new = new.strip()
    return new, new != value


def patch_obj(obj):
    """Recursively strip LLM/VLM mentions from any string field."""
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
    total_files = 0
    total_changes = 0
    for path in sorted(glob.glob('results/*/step1[49]*/*_result.json') +
                        glob.glob('results/*/step20/*_result.json')):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            continue
        n = patch_obj(data)
        if n:
            bak = path + '.bak_strip_llm'
            if not os.path.exists(bak):
                shutil.copy2(path, bak)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f'{path[len("results/"):]}: {n} field(s) cleaned')
            total_files += 1
            total_changes += n
    print(f'\n{total_files} files patched, {total_changes} field changes')


if __name__ == '__main__':
    main()
