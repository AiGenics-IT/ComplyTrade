"""Patch job e07ce444 data so pkt_3, pkt_6, pkt_28 (pages 6, 11, 38) are
correctly labelled 'Health Certificate' instead of 'Shipping Company
Certificate'. The LC's expected_docs list does NOT include Health
Certificate, so these packets are properly marked as 'alien_document'.
"""
import json
import os
import shutil

JOB = "e07ce444-aa33-4aa7-a380-ba6d182a05a6"
ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results", JOB)

# Packet IDs known to be Health Certs mislabeled as SCC
TARGET_PACKETS = {"pkt_3", "pkt_6", "pkt_28"}

# ── Step 9: reconciled_packets ──
s9_path = os.path.join(RESULTS, "step09", "step09_result.json")
shutil.copy(s9_path, s9_path + ".bak")
with open(s9_path, encoding="utf-8") as f:
    s9 = json.load(f)

patched = 0
for pkt in s9.get("reconciled_packets", []):
    pid = pkt.get("packet_id")
    if pid not in TARGET_PACKETS:
        continue
    txt_head = (pkt.get("refined_text") or pkt.get("cleaned_text")
                or pkt.get("raw_text") or "")[:120]
    assert "HEALTH CERTIFICATE" in txt_head.upper(), (
        f"Sanity check failed — {pid} text does not contain "
        f"'HEALTH CERTIFICATE': {txt_head!r}"
    )
    old_dt = pkt.get("document_type")
    old_mr = pkt.get("matched_requirement_name", "")
    pkt["document_type"] = "Health Certificate"
    pkt["matched_requirement_name"] = "Health Certificate"
    pkt["matched_requirement_index"] = -1
    pkt["classification_status"] = "alien_document"
    pkt["was_reclassified"] = True
    pkt["previous_document_type"] = old_dt or ""
    print(f"  [OK] {pid} (page {pkt.get('original_pages',[{}])[0].get('page_number','?')}): "
          f"{old_dt!r} → 'Health Certificate' "
          f"(matched_requirement: {old_mr!r} → 'Health Certificate')")
    patched += 1

with open(s9_path, "w", encoding="utf-8") as f:
    json.dump(s9, f, indent=2, ensure_ascii=False)
print(f"[OK] Patched {patched} packets in step09_result.json (backup: .bak)")

# ── Step 8: apply same update to classifications list for consistency ──
s8_path = os.path.join(RESULTS, "step08", "step08_result.json")
if os.path.exists(s8_path):
    shutil.copy(s8_path, s8_path + ".bak")
    with open(s8_path, encoding="utf-8") as f:
        s8 = json.load(f)
    s8_patched = 0
    for pkt in s8.get("classifications", []) + s8.get("classified_packets", []):
        pid = pkt.get("packet_id")
        if pid not in TARGET_PACKETS:
            continue
        pkt["document_type"] = "Health Certificate"
        pkt["matched_requirement_name"] = "Health Certificate"
        pkt["matched_requirement_index"] = -1
        pkt["classification_status"] = "alien_document"
        s8_patched += 1
    if s8_patched:
        with open(s8_path, "w", encoding="utf-8") as f:
            json.dump(s8, f, indent=2, ensure_ascii=False)
        print(f"[OK] Also patched {s8_patched} entries in step08_result.json")

# ── Sanity: re-read and confirm ──
with open(s9_path, encoding="utf-8") as f:
    s9_check = json.load(f)
print("\nFinal verification:")
for pkt in s9_check.get("reconciled_packets", []):
    pid = pkt.get("packet_id")
    if pid in TARGET_PACKETS:
        print(f"  {pid}: document_type={pkt.get('document_type')!r}  "
              f"matched_requirement_name={pkt.get('matched_requirement_name')!r}  "
              f"classification_status={pkt.get('classification_status')!r}")
