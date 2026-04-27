"""P198dn — Remove [:200] caps on row['result'] / row['found_text']
in step14 rescue paths. The Result column in the per-clause tables
shows the full finding text (no truncation)."""
import re
from pathlib import Path

TARGET = Path('steps/step14_verification.py')
src = TARGET.read_text(encoding='utf-8')
lines = src.split('\n')

PATTERNS = [
    # row['result'] = msg[:200]   |   row["result"] = msg[:200]
    re.compile(r"^(\s*row\[['\"](?:result|found_text)['\"]\]\s*=\s*[^\n]*?)\[:200\]\s*$"),
    # _set(row, 'result', _msg[:200])  |  _set(row, "result", X[:200])
    re.compile(r"^(\s*_set\(\s*row\s*,\s*['\"](?:result|found_text)['\"]\s*,\s*[^\n]*?)\[:200\]\s*\)\s*$"),
]

changes = 0
new_lines = []
for ln in lines:
    new = ln
    # Try _set(row, 'X', Y[:200]) pattern first (multi-arg call)
    m = re.match(r"^(\s*_set\(\s*row\s*,\s*['\"](?:result|found_text)['\"]\s*,\s*)(.*?)\[:200\]\s*(\)\s*)$", ln)
    if m:
        new = f"{m.group(1)}{m.group(2)}{m.group(3)}"
    else:
        # Try row['X'] = Y[:200] pattern
        m = re.match(r"^(\s*row\[['\"](?:result|found_text)['\"]\]\s*=\s*)(.*?)\[:200\]\s*$", ln)
        if m:
            new = f"{m.group(1)}{m.group(2)}"
    if new != ln:
        changes += 1
    new_lines.append(new)

TARGET.write_text('\n'.join(new_lines), encoding='utf-8')
print(f'P198dn: {changes} lines patched in {TARGET}')
