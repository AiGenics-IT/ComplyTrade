"""
Patch 3e286ed4 step14 data with P198bb permissive rescue + P198bd
vessel-age-cert label, then regenerate Step 19 (consolidation) and
Step 20 (final PDF report). Run with:

    python _patch_3e286ed4_bb_report.py
"""
import json
import os
import re
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

JOB_ID = "3e286ed4-41f2-4959-9d84-ec1d846ac285"
LC     = "1001LC59573/2026"
RESULTS = os.path.join(HERE, "results", JOB_ID)

step06_path = os.path.join(RESULTS, "step06", "step06_result.json")
step07_path = os.path.join(RESULTS, "step07", "step07_result.json")
step09_path = os.path.join(RESULTS, "step09", "step09_result.json")
step12_path = os.path.join(RESULTS, "step12", "step12_result.json")
step13_path = os.path.join(RESULTS, "step13", "step13_result.json")
step14_path = os.path.join(RESULTS, "step14", "step14_result.json")
step14b_path = os.path.join(RESULTS, "step14b", "step14b_result.json")

with open(step14_path) as f:
    s14 = json.load(f)

rows = s14.get("rows", [])
print(f"Loaded {len(rows)} step14 rows")

# ── Patch 1: P198bb — permissive condition cannot FAIL ──
_PERMISSIVE_RE = re.compile(
    r'\b(?:is|are|to\s+be)\s+'
    r'(?:acceptable|permissible|permitted|allowed|allowable)\b',
    flags=re.IGNORECASE,
)
_PERMISSIVE_NEG_RE = re.compile(
    r'\b(?:not|no|never|must\s+not|shall\s+not|cannot)\s+'
    r'(?:be\s+)?(?:acceptable|permissible|permitted|allowed)\b',
    flags=re.IGNORECASE,
)

flips = 0
for r in rows:
    if (r.get("compliance") or "").upper() != "FAIL":
        continue
    cond = (r.get("condition_text") or "").strip()
    if not cond:
        continue
    if not _PERMISSIVE_RE.search(cond):
        continue
    if _PERMISSIVE_NEG_RE.search(cond):
        continue
    msg = (
        f"Permissive LC allowance — the condition states "
        f"'{cond[:120]}'. A permissive carve-out grants an "
        f"allowance and cannot produce a discrepancy. "
        f"Treated as PASS."
    )
    r["compliance"] = "PASS"
    r["findings"] = msg
    r["result"] = msg[:200]
    r["verification_notes"] = (
        "P198bb permissive-cant-fail: "
        "acceptable/permitted/allowed statement cannot FAIL"
    )
    print(f"  [P198bb] {r.get('row_id')} FAIL->PASS: {cond[:100]}")
    flips += 1

# ── Patch 2: P198bd — vessel age cert distinct label ──
labeled = 0
for r in rows:
    if (r.get("compliance") or "").upper() != "FAIL":
        continue
    cond_label = (
        (r.get("condition_text") or "")
        + " "
        + (r.get("look_for_value") or "")
    ).upper()
    findings = (r.get("findings") or "")
    result = (r.get("result") or "")
    # Only relabel when it currently says "not found in submission"
    if "NOT FOUND IN SUBMISSION" not in (findings + result).upper():
        continue
    new_label = None
    if ('OWNER OF THE VESSEL' in cond_label
            or 'VESSEL OWNER' in cond_label
            or 'VESSELS AGE' in cond_label
            or "VESSEL'S AGE" in cond_label
            or '15 YEARS' in cond_label):
        new_label = "Vessel Age Certificate"
    elif 'SHELF LIFE' in cond_label and 'CERTIFICATE' in cond_label:
        new_label = "Shelf Life Certificate"
    elif ('DIRECT SAILING' in cond_label
          or 'SAIL DIRECT' in cond_label):
        new_label = "Direct Sailing Certificate"
    if new_label is None:
        continue
    r["document_checked"] = new_label
    r["findings"] = f"{new_label} not found in submission"
    r["found_text"] = r["findings"]
    r["result"] = f"Required document missing: {new_label}"
    print(f"  [P198bd] {r.get('row_id')} relabeled to: {new_label}")
    labeled += 1

# Recompute summary counts
summary = s14.get("summary", {})
pass_n = sum(1 for r in rows if (r.get("compliance") or "").upper() == "PASS")
fail_n = sum(1 for r in rows if (r.get("compliance") or "").upper() == "FAIL")
review_n = sum(1 for r in rows if (r.get("compliance") or "").upper() == "REVIEW")
info_n = sum(1 for r in rows if (r.get("compliance") or "").upper() in ("N/A", "INFO"))
summary["pass"] = pass_n
summary["fail"] = fail_n
summary["review"] = review_n
summary["informational"] = info_n
summary["total_rows"] = len(rows)
s14["summary"] = summary
s14["pass"] = pass_n
s14["fail"] = fail_n
s14["review"] = review_n

with open(step14_path, "w") as f:
    json.dump(s14, f, indent=2)
print(f"[OK] Patched step14 — {flips} permissive flips, {labeled} cert relabels")
print(f"     new counts: {pass_n}P / {fail_n}F / {review_n}R / {info_n}I")

# ── Regen Step 19 (consolidation) ──
from steps import step19_consolidation  # type: ignore

with open(step06_path) as f: s6 = json.load(f)
with open(step07_path) as f: s7 = json.load(f)
with open(step09_path) as f: s9 = json.load(f)
with open(step13_path) as f: s13 = json.load(f)
try:
    with open(step14b_path) as f: s14b = json.load(f)
except Exception:
    s14b = {}

out_dir19 = os.path.join(RESULTS, "step19")
os.makedirs(out_dir19, exist_ok=True)
print("\n[Step 19] Regenerating consolidation...")

def _log(msg): print("  ", msg)

# step19.run(reconciled_rows, output_dir, progress_fn)
# Build reconciled_rows = step14 rows + step14b rows
all_rows = list(s14.get("rows", []))
try:
    for c in (s14b.get("checks") or []):
        # step14b checks already have compliance/findings fields; shim to row shape
        row = dict(c)
        row.setdefault("row_id", c.get("check_id", ""))
        row.setdefault("condition_text", c.get("condition", ""))
        row.setdefault("document_type", c.get("document_checked", ""))
        all_rows.append(row)
except Exception as _e:
    print(f"[warn] step14b merge: {_e}")
s19 = step19_consolidation.run(all_rows, out_dir19, _log)

if hasattr(s19, 'model_dump'):
    s19d = s19.model_dump()
elif hasattr(s19, 'dict'):
    s19d = s19.dict()
else:
    s19d = s19

print(f"[Step 19] decision={s19d.get('overall_decision')} "
      f"{s19d.get('total_pass')}P/{s19d.get('total_fail')}F/"
      f"{s19d.get('total_review')}R")

# ── Regen Step 20 (report) ──
from steps import step20_report  # type: ignore
out_dir20 = os.path.join(RESULTS, "step20")
os.makedirs(out_dir20, exist_ok=True)
print("\n[Step 20] Regenerating PDF report...")
# step20.run(consolidated, lc_fields, output_dir, progress_fn)
s20 = step20_report.run(s19d, s6, out_dir20, _log)

if hasattr(s20, 'model_dump'):
    s20d = s20.model_dump()
elif hasattr(s20, 'dict'):
    s20d = s20.dict()
else:
    s20d = s20

pdf_path = s20d.get('pdf_path') or s20d.get('report_path') or s20d.get('output_path')
print(f"[Step 20] PDF: {pdf_path}")
print("\nDone.")
