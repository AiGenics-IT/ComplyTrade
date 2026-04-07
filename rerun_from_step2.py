"""Re-run pipeline from Step 2 onwards using existing Step 1 data."""
import os, sys, json, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from steps import step02_ocr_cleaning, step03_sequencing

JOB_ID = "dd974dc6-e24c-45a6-8434-6e06f78bba9f"
RESULTS = os.path.join("results", JOB_ID)

def _p(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

# Load Step 1 result
with open(os.path.join(RESULTS, "step01", "step01_result.json"), 'r', encoding='utf-8') as f:
    s1 = json.load(f)
_p(f"Step 1 loaded: {s1['total_pages']} pages, {s1.get('total_chars', 0)} chars")

# Step 2
_p("Running Step 2...")
s2_dir = os.path.join(RESULTS, "step02")
os.makedirs(s2_dir, exist_ok=True)
s2 = step02_ocr_cleaning.run(s1, s2_dir, _p)
# Convert dataclasses
def _to_dict(obj):
    if hasattr(obj, '__dataclass_fields__'):
        from dataclasses import asdict
        return asdict(obj)
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_dict(i) for i in obj]
    return obj
s2 = _to_dict(s2)
with open(os.path.join(s2_dir, "step02_result.json"), 'w', encoding='utf-8') as f:
    json.dump(s2, f, ensure_ascii=False, indent=2)
_p(f"Step 2 done")

# Step 3
_p("Running Step 3...")
s3_dir = os.path.join(RESULTS, "step03")
os.makedirs(s3_dir, exist_ok=True)
s3 = _to_dict(step03_sequencing.run(s2, s3_dir, _p))
with open(os.path.join(s3_dir, "step03_result.json"), 'w', encoding='utf-8') as f:
    json.dump(s3, f, ensure_ascii=False, indent=2)
packets = s3.get('packets', [])
_p(f"Step 3 done: {len(packets)} packets")
for p in packets:
    pgs = p.get('page_numbers', [])
    dt = p.get('document_type', '?')
    _p(f"  {p.get('packet_id')}: {dt} pages={pgs}")

_p("Classification complete. Check results above.")
