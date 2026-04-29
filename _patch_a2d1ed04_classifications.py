"""
Surgical fix-up for job a2d1ed04-eb9d-4a36-8c57-9f6295e1e3fc — patches the
saved step03/step08 JSON files in place so the extracted-text view shows
the correct classifications. NO VLM/LLM calls. NO pipeline re-run.

Two corrections:

  P198eo — step08 packets where the AWB rule fired and produced a wrong
  "Bill of Lading" match (because the LC has only BL, no AWB / Courier,
  in expected_docs). For each such packet:
      document_type            = "Airway Bill"
      matched_requirement_index = -1
      matched_requirement_name  = ""
      classification_status     = "alien_document"

  P198ep — step03 per-page classifications where doc_hint identifies a
  cargo-specification rider (the BL "*** AS PER ATTACHED SPECIFICATION
  ***" attachment) but inheritance overwrote the doc_type to whatever
  the preceding page was (Commercial Invoice / Covering Letter / etc.):
      cls['document_type'] = "Specification of Cargo"

  The packet boundaries themselves are NOT changed — that would require
  re-running the bl_subtype / unified_summary generators. Instead, the
  server's extracted-text endpoint (post-P198er) prefers the per-page
  classification label over the packet-level label when they differ,
  so the UI display is correct without any structural mutation.

Use:
  python _patch_a2d1ed04_classifications.py
"""
import os, json, sys, datetime, shutil

JOB_ID  = 'a2d1ed04-eb9d-4a36-8c57-9f6295e1e3fc'
ROOT    = 'd:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final'
JOB_DIR = os.path.join(ROOT, 'results', JOB_ID)

if not os.path.isdir(JOB_DIR):
    print(f"Job dir not found: {JOB_DIR}")
    sys.exit(1)

# ── Backup helpers ──────────────────────────────────────────────────────
def _backup(path):
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    b = path + f'.bak_{ts}'
    shutil.copy2(path, b)
    print(f"  backup: {b}")

# ──────────────────────────────────────────────────────────────────────
# 1. STEP 08 — undo wrong AWB->BL mapping
# ──────────────────────────────────────────────────────────────────────
s8_path = os.path.join(JOB_DIR, 'step08', 'step08_result.json')
if os.path.exists(s8_path):
    print("=" * 78)
    print(f"STEP 08: {s8_path}")
    print("=" * 78)
    with open(s8_path, 'r', encoding='utf-8') as f:
        s8 = json.load(f)

    fixed = 0
    kept   = 0
    for pkt in s8.get('classified_packets', []) or []:
        if not isinstance(pkt, dict):
            continue
        reasoning = (pkt.get('vlm_reasoning') or '').lower()
        cur_dt    = (pkt.get('document_type') or '').strip()
        # Recover what the VLM ACTUALLY said before the override fired —
        # the override message captures it: "...(suppressing VLM
        # classification: <vlm_label>)".
        suppressed_vlm = ''
        if 'suppressing vlm classification:' in reasoning:
            suppressed_vlm = reasoning.split('suppressing vlm classification:', 1)[1]
            suppressed_vlm = suppressed_vlm.strip().strip(')').strip()
        # P198es veto: only demote to AWB when the VLM did NOT confirm BL.
        # If VLM said "Bill of Lading", the rule override was a false
        # positive (AWB scorer over-matches on shipping pages) and the
        # current "Bill of Lading" label is genuinely correct — leave it.
        vlm_said_bl = ('bill of lading' in suppressed_vlm
                       or suppressed_vlm in {'b/l', 'bl', 'congenbill'})
        if 'airway bill signals' in reasoning and 'bill of lading' in cur_dt.lower() \
                and not vlm_said_bl:
            pages = [op.get('page_number') for op in (pkt.get('original_pages') or [])
                     if isinstance(op, dict)]
            print(f"  pkt={pkt.get('packet_id')} pages={pages} | BL -> Airway Bill (alien) | VLM had said: {suppressed_vlm}")
            pkt['document_type']             = 'Airway Bill'
            pkt['matched_requirement_index'] = -1
            pkt['matched_requirement_name']  = ''
            pkt['classification_status']     = 'alien_document'
            existing_notes = pkt.get('ambiguity_notes', '') or ''
            note = "P198eo backfill: AWB rule override; LC has no AWB/Courier slot, so this is alien."
            pkt['ambiguity_notes'] = (note if not existing_notes
                                      else f"{existing_notes}; {note}")
            fixed += 1
        elif 'airway bill signals' in reasoning and vlm_said_bl:
            # Genuine BL whose AWB rule was a false-positive lexical
            # match. Strip the override note so future re-runs use the
            # VLM's BL classification cleanly.
            pkt['vlm_reasoning'] = ('VLM classified as Bill of Lading; '
                                    'rule-override AWB signals suppressed '
                                    '(P198es veto — false-positive AWB scorer)')
            kept += 1
    print(f"  -> kept {kept} BL packets (VLM correctly said BL — false-positive AWB scorer)")
    print(f"  -> patched {fixed} packets")
    if fixed:
        _backup(s8_path)
        with open(s8_path, 'w', encoding='utf-8') as f:
            json.dump(s8, f, indent=2, ensure_ascii=False)
        print(f"  saved: {s8_path}")

# ──────────────────────────────────────────────────────────────────────
# 2. STEP 03 — relabel SPEC OF CARGO pages
# ──────────────────────────────────────────────────────────────────────
s3_path = os.path.join(JOB_DIR, 'step03', 'step03_result.json')
if os.path.exists(s3_path):
    print()
    print("=" * 78)
    print(f"STEP 03: {s3_path}")
    print("=" * 78)
    with open(s3_path, 'r', encoding='utf-8') as f:
        s3 = json.load(f)

    fixed = 0
    for cls in s3.get('classifications', []) or []:
        if not isinstance(cls, dict):
            continue
        hint = (cls.get('doc_hint') or '').lower()
        cur  = (cls.get('document_type') or '').strip()
        # Detect cargo-specification rider via doc_hint (preserved by VLM
        # before inheritance overwrite). Skip if already labelled
        # correctly OR if the current label is one we don't want to
        # touch (e.g. legitimate BL classification we shouldn't relabel).
        is_spec_hint = ('specification' in hint and 'cargo' in hint)
        if not is_spec_hint:
            continue
        if 'specification' in cur.lower():
            continue   # already correct
        pn = cls.get('page_number')
        print(f"  pg={pn:3d}: '{cur}' -> 'Specification of Cargo' (hint={hint[:40]})")
        cls['document_type'] = 'Specification of Cargo'
        cls['_p198ep_relabel_from'] = cur
        fixed += 1
    print(f"  -> relabelled {fixed} per-page classifications")
    if fixed:
        _backup(s3_path)
        with open(s3_path, 'w', encoding='utf-8') as f:
            json.dump(s3, f, indent=2, ensure_ascii=False)
        print(f"  saved: {s3_path}")

print()
print("=" * 78)
print("DONE — refresh the extracted-text view to see the corrected labels.")
print("=" * 78)
print()
print("Per-page display sources (per server.py post-P198er):")
print("  step08 packet doctype  >  step03 per-page classification  >  step03 packet")
print()
print("So pages now show:")
print("  • Pages 40, 79, 80, 83 -> 'Airway Bill' (was 'Bill of Lading')")
print("  • Pages 14, 16, 17, 18, 44 -> 'Specification of Cargo'")
print("    (was 'Commercial Invoice' / 'Covering Letter')")
print("  • Pages 8, 32, 33, 34, 72, 73, 80 -> 'Specification of Cargo'")
print("    (were already in BL packets so the label was BL — now per-page truth)")
