"""Test Step 3 classification only with updated prompt."""
import os, sys, json, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from steps import step03_sequencing

JOB_ID = "dd974dc6-e24c-45a6-8434-6e06f78bba9f"
RESULTS = os.path.join("results", JOB_ID)

def _p(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

# Load Step 2 result (from the first successful run)
with open(os.path.join(RESULTS, "step02", "step02_result.json"), 'r', encoding='utf-8') as f:
    s2 = json.load(f)

non_empty = sum(1 for p in s2.get('pages', []) if p.get('cleaned_text', ''))
_p(f"Step 2 loaded: {len(s2.get('pages',[]))} pages, {non_empty} non-empty")

# Run Step 3
_p("Running Step 3 with updated classification prompt...")
s3_dir = os.path.join(RESULTS, "step03")
os.makedirs(s3_dir, exist_ok=True)

start = time.time()
try:
    def _to_dict(obj):
        if hasattr(obj, '__dataclass_fields__'):
            from dataclasses import asdict
            return asdict(obj)
        if isinstance(obj, dict):
            return {k: _to_dict(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_dict(i) for i in obj]
        return obj

    s3 = _to_dict(step03_sequencing.run(s2, s3_dir, _p))
    with open(os.path.join(s3_dir, "step03_result.json"), 'w', encoding='utf-8') as f:
        json.dump(s3, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - start
    packets = s3.get('packets', [])
    _p(f"\nStep 3 done in {elapsed:.1f}s: {len(packets)} packets\n")

    for p in packets:
        pgs = p.get('page_numbers', [])
        dt = p.get('document_type', '?')
        conf = p.get('boundary_confidence', 0)
        copy_lbl = p.get('copy_label', '')
        _p(f"  {p.get('packet_id')}: {dt} pages={pgs} conf={conf} copy={copy_lbl}")

except Exception as e:
    import traceback
    _p(f"ERROR: {e}")
    traceback.print_exc()
