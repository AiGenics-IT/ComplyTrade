"""Re-run Step 3 only for job 46660d08 to pick up P198cg fix."""
import os, sys, json, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from steps import step03_sequencing
from dataclasses import asdict

JOB_ID = "46660d08-ae1c-44e7-972b-05ea13fc1fe6"
RESULTS = os.path.join("results", JOB_ID)


def _p(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def _to_dict(o):
    if hasattr(o, '__dataclass_fields__'):
        return asdict(o)
    if isinstance(o, dict):
        return {k: _to_dict(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_to_dict(i) for i in o]
    return o


s2_path = os.path.join(RESULTS, "step02", "step02_result.json")
with open(s2_path, 'r', encoding='utf-8') as f:
    s2 = json.load(f)
_p(f"Step 2 loaded from {s2_path}")

s3_dir = os.path.join(RESULTS, "step03")
os.makedirs(s3_dir, exist_ok=True)

_p("Running Step 3 (P198cg build)...")
s3 = _to_dict(step03_sequencing.run(s2, s3_dir, _p))

with open(os.path.join(s3_dir, "step03_result.json"), 'w', encoding='utf-8') as f:
    json.dump(s3, f, ensure_ascii=False, indent=2)

packets = s3.get('packets', [])
_p(f"Step 3 done: {len(packets)} packets")
for p in packets:
    pgs = p.get('page_numbers', [])
    dt = p.get('document_type', '?')
    fmt = p.get('swift_format') or p.get('format') or ''
    _p(f"  {p.get('packet_id')}: {dt}  pages={pgs}  format={fmt}")

# Focused report on pages 1-6 grouping
_p("")
_p("=== PAGES 1-6 GROUPING ===")
for p in packets:
    pgs = p.get('page_numbers', [])
    if any(pn in (1,2,3,4,5,6) for pn in pgs):
        _p(f"  {p.get('packet_id')}: {p.get('document_type')}  pages={pgs}")
