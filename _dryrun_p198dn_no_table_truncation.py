"""
P198dn dry-run — verify there are no remaining truncations of
row['result'] / row['found_text'] / row['findings'] in step14
rescue paths, and that step20 table cells are uncapped.
"""
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

problems = []

# 1. step14_verification.py — no [:200], [:300], [:500], [:600]
#    truncations on row['result'] / row['found_text'] / parsed['result'].
s14 = Path('steps/step14_verification.py').read_text(encoding='utf-8')
for cap in (200, 300, 500, 600):
    pat = re.compile(
        rf"(?:row|parsed)\[['\"](?:result|found_text|findings)['\"]\][^=\n]*?=[^\n]*?\[:{cap}\]"
        rf"|_set\(\s*[A-Za-z_]+\s*,\s*['\"](?:result|found_text|findings)['\"]\s*,\s*[^\n]*?\[:{cap}\]",
    )
    for ln_no, line in enumerate(s14.split('\n'), 1):
        if pat.search(line):
            problems.append(f"step14_verification.py:{ln_no} — still has [:{cap}] cap on result/found_text/findings: {line.strip()[:120]}")

# 2. step20_report.py — main per-clause table cells must use 100000 cap.
s20 = Path('steps/step20_report.py').read_text(encoding='utf-8')
# Verify the four key cells are at 100000.
for needle, name in [
    ("_safe_str(_cond_text, 100000)", "Condition cell"),
    ("_safe_str(_find_text, 100000)", "Findings cell"),
    ('_safe_str(row.get("document_checked", ""), 100000)', "Document Checked cell"),
    ("_safe_str(result_val, 100000)", "Result cell"),
    ("_safe_str(clause.get('clause_text', ''), 100000)", "Clause-text bar"),
]:
    if needle not in s20:
        problems.append(f"step20_report.py — missing 100000 cap on {name} ('{needle}')")

# Manual review detail cap should be 100000 (not 500)
if "_safe_str(\n                    ri.get('condition', '') or ri.get('result', '') or ri.get('findings', ''), 500)" in s20:
    problems.append("step20_report.py — Manual Review detail still capped at 500")

print('=' * 78)
print('P198dn dry-run — table-cell truncation check')
print('=' * 78)
if problems:
    print(f'\n{len(problems)} problem(s) found:\n')
    for p in problems:
        print(f'  ! {p}')
    print('\nFAIL')
    sys.exit(1)
else:
    print('\nNo truncations remaining on row[result] / row[found_text] /')
    print('parsed[result] / clause_text / Manual Review detail.')
    print('Per-clause table cells (Condition, Findings, Document Checked,')
    print('Result, Clause-text bar) are all uncapped (max_len=100000).')
    print('\nOK')
