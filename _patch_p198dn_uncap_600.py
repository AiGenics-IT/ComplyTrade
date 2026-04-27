"""P198dn (cont) — strip [:600] caps on parsed['result'] / parsed['findings']."""
import re
from pathlib import Path

TARGET = Path('steps/step14_verification.py')
src = TARGET.read_text(encoding='utf-8')
lines = src.split('\n')

changes = 0
new_lines = []
for ln in lines:
    new = ln
    # parsed["result"] = X[:600]   |  parsed['result'] = X[:600]
    m = re.match(r"^(\s*parsed\[['\"](?:result|findings|found_text)['\"]\]\s*=\s*)(.*?)\[:600\](\s*(?:if\b.*)?$)", ln)
    if m:
        new = f"{m.group(1)}{m.group(2)}{m.group(3)}"
    if new != ln:
        changes += 1
    new_lines.append(new)

TARGET.write_text('\n'.join(new_lines), encoding='utf-8')
print(f'P198dn-cont: {changes} lines patched')
