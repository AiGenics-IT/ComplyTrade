"""
P198eo dry-run — verify that AWB does NOT fuzzy-match to BL when LC requires
only BL (no AWB / no courier).

Setup:
  - doc_type = "Airway Bill"
  - LC expected_docs = [{"document_name": "Bill of Lading"}, ...]
  - Old behavior: fuzzy match found "BILL" overlap → mapped AWB → BL
  - New behavior: AWB family with no AWB/Courier requirement → return -1
                  (alien_document)

Also exercises:
  - LC has Courier Receipt → AWB doc_type maps to Courier Receipt requirement
  - LC has Airway Bill → AWB doc_type maps to Airway Bill requirement
  - LC has only BL → AWB doc_type returns -1 (NEW)
  - VLM says Courier Receipt, LC has only BL → Courier Receipt returns -1 (NEW)
"""
import sys
sys.path.insert(0, 'd:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final')
from steps.step08_shipping_classification import _match_type_to_requirement

def case(name, doc_type, expected, expect_idx, expect_name):
    idx, n = _match_type_to_requirement(doc_type, expected)
    status = "OK" if (idx == expect_idx and n == expect_name) else "FAIL"
    print(f"[{status}] {name:60s} dt={doc_type!r:25s} -> idx={idx} name={n!r}")
    if status == "FAIL":
        print(f"          expected idx={expect_idx} name={expect_name!r}")
    return status == "OK"

results = []

# 1. AWB doc_type vs LC with BL only → must NOT match (alien)
results.append(case("AWB doc vs BL-only LC (alien)",
    "Airway Bill",
    [{"document_name": "Bill of Lading"}, {"document_name": "Commercial Invoice"}],
    -1, ""))

# 2. AWB doc_type vs LC with AWB → match
results.append(case("AWB doc vs AWB LC",
    "Airway Bill",
    [{"document_name": "Airway Bill"}, {"document_name": "Commercial Invoice"}],
    0, "Airway Bill"))

# 3. AWB doc_type vs LC with Courier Receipt → match (cross-label same family)
results.append(case("AWB doc vs Courier-Receipt LC",
    "Airway Bill",
    [{"document_name": "Courier Receipt"}, {"document_name": "Commercial Invoice"}],
    0, "Courier Receipt"))

# 4. Courier Receipt doc_type vs LC with BL only → must NOT match (alien)
results.append(case("CR doc vs BL-only LC (alien)",
    "Courier Receipt",
    [{"document_name": "Bill of Lading"}, {"document_name": "Packing List"}],
    -1, ""))

# 5. Courier Receipt doc_type vs LC with AWB → match
results.append(case("CR doc vs AWB LC",
    "Courier Receipt",
    [{"document_name": "Air Waybill"}, {"document_name": "Bill of Lading"}],
    0, "Air Waybill"))

# 6. Sanity: BL doc_type vs LC with BL → match (regression check)
results.append(case("BL doc vs BL LC (regression)",
    "Bill of Lading",
    [{"document_name": "Bill of Lading"}, {"document_name": "Commercial Invoice"}],
    0, "Bill of Lading"))

# 7. Sanity: Commercial Invoice should still match
results.append(case("CI doc vs CI LC (regression)",
    "Commercial Invoice",
    [{"document_name": "Bill of Lading"}, {"document_name": "Commercial Invoice"}],
    1, "Commercial Invoice"))

# 8. Sanity: Packing List should still match via fuzzy
results.append(case("PL doc vs PL LC (regression)",
    "Packing List",
    [{"document_name": "Bill of Lading"}, {"document_name": "Packing List"}],
    1, "Packing List"))

# 9. AWB doc_type with NO expected_docs → -1
results.append(case("AWB doc vs empty LC",
    "Airway Bill",
    [],
    -1, ""))

# 10. HAWB / MAWB variants vs BL-only → -1
results.append(case("HAWB doc vs BL-only LC (alien)",
    "HAWB",
    [{"document_name": "Bill of Lading"}],
    -1, ""))

results.append(case("MAWB doc vs BL-only LC (alien)",
    "MAWB",
    [{"document_name": "Bill of Lading"}],
    -1, ""))

# 11. DHL Express variants
results.append(case("DHL Express Waybill vs BL-only LC (alien)",
    "DHL Express Waybill",
    [{"document_name": "Bill of Lading"}],
    -1, ""))

passed = sum(results)
total = len(results)
print(f"\n{passed}/{total} cases passed")
sys.exit(0 if passed == total else 1)
