"""LLM-backed end-to-end dry-run.

For each scenario, builds a compact verification prompt (same shape as
step14's split-prompt core — condition, document text excerpt, structured
facts) and posts to the actual LLM. Parses the verdict and compares to
the expected verdict. Reports mismatches so we know where the LLM itself
is unreliable and a post-check rescue is carrying the load.

Usage:
    python _dryrun_llm_end_to_end.py

Requires: LAN reachable (10.20.10.4 LLM).
"""
import json
import os
import re
import sys
import time
import requests


LLM_URL = "http://10.20.10.4/llm/v1/chat/completions"
LLM_MODEL = "Qwen2.5-72B-Instruct-GPTQ-Int8"
TIMEOUT = 90

BL_FAMILY_PACK = """BILL OF LADING — additional rules:

NVOCC / FIATA / FREIGHT FORWARDER — DISTINGUISH DEFINITION FROM EVIDENCE:
BL T&C pages carry glossary definitions like
    "NVOCC" MEANS NON VESSEL OPERATING COMMON CARRIER.
    "FIATA" MEANS THE INTERNATIONAL FEDERATION OF FREIGHT FORWARDERS.
These are boilerplate, NOT evidence. A BL is actually an NVOCC/FIATA/
House/FF BL ONLY when the term appears in:
  - ISSUER / CARRIER letterhead or identification block
  - SIGNATURE block ("SIGNED AS FREIGHT FORWARDER", "AS NVOCC")
  - STAMP / SEAL identifying the BL type on the face
  - Explicit BL-class title ("HOUSE BILL OF LADING" as the printed title)

For a prohibitive condition "NVOCC/FF/FIATA/House BL NOT ACCEPTABLE":
  - Token ONLY in \"<TERM>\" MEANS ... / DEFINITIONS block → PASS (boilerplate)
  - Token in issuer line or signature block or title → FAIL
  - Issuer is a real ocean carrier (Maersk / MSC / CMA CGM / COSCO / OOCL /
    Hapag-Lloyd / ONE / Evergreen / PIL / Pacific International Lines /
    Yang Ming / ZIM / HMM / APL) AND token only in T&C definitions → PASS

Do NOT write "BL is issued by NVOCC" unless the issuer letterhead /
signature block actually says so.

CONSIGNEE "TO ORDER" WITHOUT NAMED BANK:
If LC requires the BL made out "TO THE ORDER OF <BANK>" and the consignee
field shows just "TO ORDER" with no named party and no explicit endorsement
line to <BANK> on the BL face, verdict is FAIL — NOT REVIEW. A bare "TO
ORDER" without endorsement is a strict UCP 600 Art 14(e) discrepancy.
"""

PROMPT_TEMPLATE = """You are a trade finance document examiner under UCP 600.
Evaluate whether the document satisfies the LC condition.

LC CONDITION:
{cond}

DOCUMENT TYPE: {doc_type}

DOCUMENT TEXT (excerpt, may be truncated):
---
{doc_text}
---

STRUCTURED FACTS (parsed by OCR pipeline):
{structured}

{family_pack}

RULES:
- Return strict JSON: {{"verdict": "PASS"|"FAIL"|"REVIEW", "findings": "<=150 chars"}}
- PASS when the document clearly satisfies the condition
- FAIL when the document clearly violates the condition
- REVIEW only when the evidence is ambiguous
- Do NOT add any text outside the JSON.
"""


def ask_llm(prompt):
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 300,
        "temperature": 0.0,
    }
    t0 = time.time()
    resp = requests.post(LLM_URL, json=payload, timeout=TIMEOUT)
    elapsed = time.time() - t0
    if resp.status_code != 200:
        body = (resp.text or "")[:200].replace("\n", " ")
        return {
            "verdict": "ERROR",
            "findings": f"HTTP {resp.status_code}: {body}",
            "elapsed": elapsed,
            "raw": "",
        }
    raw = resp.json()["choices"][0]["message"]["content"].strip()
    # Extract JSON
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        return {"verdict": "PARSE_FAIL", "findings": "no JSON found",
                "elapsed": elapsed, "raw": raw}
    try:
        parsed = json.loads(m.group(0))
    except Exception as e:
        return {"verdict": "PARSE_FAIL", "findings": f"JSON error: {e}",
                "elapsed": elapsed, "raw": raw}
    parsed["elapsed"] = elapsed
    parsed["raw"] = raw
    return parsed


# ────────────────────────────────────────────────────────────
# Load real job data
# ────────────────────────────────────────────────────────────
DATA_DIR = "results"
REAL_BLS = []
SHIPMENT_ADVICES = []
INVOICES = []


def _load_packets(job_id):
    with open(f"{DATA_DIR}/{job_id}/step09/step09_result.json",
              encoding="utf-8") as f:
        d = json.load(f)
    return d.get("reconciled_packets", [])


# Job 11ec29b8 — LDPE Infinix (BL has FREIGHT PREPAID + BANK AL HABIB)
try:
    for p in _load_packets("11ec29b8-6eaf-4c71-b0f2-1557030dc4c1"):
        dt = (p.get("document_type", "") or "").lower()
        txt = (p.get("refined_text") or p.get("cleaned_text") or "")
        if "bill of lading" in dt and len(txt) > 5000:
            REAL_BLS.append(("11ec29b8-BL", p.get("document_type"), txt))
        if "shipment" in dt:
            SHIPMENT_ADVICES.append(("11ec29b8-SA", p.get("document_type"), txt,
                                     p.get("packet_id")))
        if "commercial invoice" in dt:
            INVOICES.append(("11ec29b8-INV", p.get("document_type"), txt))
except Exception as e:
    print(f"[warn] could not load 11ec29b8 data: {e}")

# Job 73be98d9 — Pacific International Lines BL with boilerplate NVOCC
try:
    for p in _load_packets("73be98d9-724f-4500-a08c-79802b4a5794"):
        dt = (p.get("document_type", "") or "").lower()
        txt = (p.get("refined_text") or p.get("cleaned_text") or "")
        if "bill of lading" in dt and len(txt) > 10000:
            REAL_BLS.append(("73be98d9-BL", p.get("document_type"), txt))
            break
except Exception as e:
    print(f"[warn] could not load 73be98d9: {e}")


# Pick one real BL and one invoice with enough text
def _trim(text, n=4000):
    return text[:n] if len(text) > n else text


# ────────────────────────────────────────────────────────────
# Scenario definitions
# ────────────────────────────────────────────────────────────
def sa_with_email():
    """Find the shipment advice that DOES carry ABID.HUSSAIN email."""
    for (_, _, txt, _) in SHIPMENT_ADVICES:
        if "ABID.HUSSAIN" in txt.upper() and "TECNOPACK" in txt.upper():
            return txt
    return None


def sa_without_email():
    """Find a shipment advice that does NOT have the applicant email."""
    for (_, _, txt, _) in SHIPMENT_ADVICES:
        if "ABID.HUSSAIN" not in txt.upper():
            return txt
    return None


# Real BLs
real_bl_11 = REAL_BLS[0][2] if REAL_BLS else ""
real_bl_73 = None
for (lbl, _, txt) in REAL_BLS:
    if "73be98d9" in lbl:
        real_bl_73 = txt
        break

scenarios = [
    {
        "label": "FREIGHT PREPAID present on BL",
        "cond": "Bills of Lading must show freight prepaid.",
        "doc_type": "Bill of Lading",
        "doc_text": _trim(real_bl_11),
        "structured": "bl_subtype.is_freight_prepaid=True",
        "expected": "PASS",
    },
    {
        "label": "FREIGHT PREPAID absent on BL (hypothetical)",
        "cond": "Bills of Lading must show freight prepaid.",
        "doc_type": "Bill of Lading",
        "doc_text": "Shipper: ACME. Consignee: TO ORDER. Notify: ABC. "
                    "Vessel MT X. FREIGHT COLLECT. Port of loading...",
        "structured": "(none extracted)",
        "expected": "FAIL",
    },
    {
        "label": "Consignee TO THE ORDER OF BANK AL HABIB LTD",
        "cond": "Bills of Lading must be made out to the order of Bank Al Habib Ltd., Karachi, Pakistan.",
        "doc_type": "Bill of Lading",
        "doc_text": _trim(real_bl_11),
        "structured": 'unified_summary.consignee="TO THE ORDER OF BANK AL HABIB LTD., KARACHI, PAKISTAN"',
        "expected": "PASS",
    },
    {
        "label": "Consignee is just TO ORDER (no bank) — should REVIEW or FAIL",
        "cond": "Bills of Lading must be made out to the order of Bank Al Habib Ltd., Karachi, Pakistan.",
        "doc_type": "Bill of Lading",
        "doc_text": "CONSIGNEE: TO ORDER. Notify: IMPORTER PAKISTAN. "
                    "Vessel MT X. Port of loading...",
        "structured": 'unified_summary.consignee="TO ORDER"',
        "expected": "FAIL",
    },
    {
        "label": "Boilerplate NVOCC definition on BL T&C — should PASS",
        "cond": "Presentation of Bills of Lading stated to be issued by a non-vessel operating carrier company is not acceptable.",
        "doc_type": "Bill of Lading",
        "doc_text": _trim(real_bl_73) if real_bl_73 else (
            'PACIFIC INTERNATIONAL LINES (PRIVATE) LIMITED\n'
            'BL NO. ABC123\nSIGNED AS AGENT FOR AND ON BEHALF OF THE MASTER\n'
            '"NVOCC" MEANS NON VESSEL OPERATING COMMON CARRIER.\n'),
        "structured": "bl_subtype.signing_type=agent_for_master; issuer=PACIFIC INTERNATIONAL LINES",
        "expected": "PASS",
    },
    {
        "label": "Real FIATA signature block — should FAIL",
        "cond": "Presentation of Bills of Lading showing words like FIATA is not acceptable.",
        "doc_type": "Bill of Lading",
        "doc_text": "BILL OF LADING\nISSUED BY GLOBAL LOGISTICS LLC\n"
                    "FIATA MEMBER\nSIGNED AS FREIGHT FORWARDER",
        "structured": "issuer=GLOBAL LOGISTICS LLC (FIATA member)",
        "expected": "FAIL",
    },
    {
        "label": "Policy number with OCR O/0 variant",
        "cond": "Shipment Advice must reference Open Policy No. 2023008MIPD000453.",
        "doc_type": "Shipment Advice",
        "doc_text": ("Shipment Advice DD:16.02.2025\nShipment No: 9246193\n"
                     "OPEN POLICY NO.2023008MIPDO00453\nL/C No: 0007LC55189/2025"),
        "structured": "references_found[role=open_policy_reference].value=2023008MIPDO00453",
        "expected": "PASS",
    },
    {
        "label": "Applicant email present on shipment advice (pkt_36/37 style)",
        "cond": "Shipment Advice must also be addressed to the Applicant at ABID.HUSSAIN@TECNOPACK.COM.PK.",
        "doc_type": "Shipment Advice",
        "doc_text": _trim(sa_with_email() or "") or (
            "TO: UBL INSURERS\nTO: TRANSSSION TECNO ELECTRONICS\n"
            "EMAIL: ABID.HUSSAIN(AT)TECNOPACK.COM.PK\n"
            "POLICY NO.2023008MIPDO00453"),
        "structured": "(extracted emails not available)",
        "expected": "PASS",
    },
    {
        "label": "Applicant email MISSING on shipment advice (pkt_34/35 style)",
        "cond": "Shipment Advice must also be addressed to the Applicant at ABID.HUSSAIN@TECNOPACK.COM.PK.",
        "doc_type": "Shipment Advice",
        "doc_text": _trim(sa_without_email() or "") or (
            "TO: UBL INSURERS, LAHORE PAKISTAN\n"
            "Shipment date 2026-03-11\nNO applicant email on page."),
        "structured": "(extracted emails show only info@cicl.com.pk)",
        "expected": "FAIL",
    },
    {
        "label": "HS code match on Commercial Invoice",
        "cond": "Commercial Invoice must show H.S. Code 3901.1000.",
        "doc_type": "Commercial Invoice",
        "doc_text": "Beneficiary: SABIC\nProduct: LDPE HP4024WN\n"
                    "H.S. CODE 3901.1000\nQTY 25.500 MT @ USD 1,140/MT\n"
                    "Total USD 29,070.00",
        "structured": "references_found[role=hs_code].value=3901.1000",
        "expected": "PASS",
    },
    {
        "label": "HS code mismatch (wrong code)",
        "cond": "Commercial Invoice must show H.S. Code 3901.1000.",
        "doc_type": "Commercial Invoice",
        "doc_text": "Beneficiary: SABIC\nH.S. CODE 1511.9020\nQTY 25.500 MT",
        "structured": "references_found[role=hs_code].value=1511.9020",
        "expected": "FAIL",
    },
    {
        "label": "45A AND/OR alternates — invoice ships first block",
        "cond": "Goods description must show LDPE HP4024N at rate USD 1,190 per M.Ton.",
        "doc_type": "Commercial Invoice",
        "doc_text": "Beneficiary: SABIC\nProduct: LDPE HP4024WN\n"
                    "QTY 25.500 MT @ USD 1,140/MT\n"
                    "LC 45A lists HP4023WN AND/OR HP4024WN AND/OR HP4025ZN @1,140 OR HP4024N @1,190",
        "structured": "invoice line: HP4024WN @ 1,140",
        "expected": "FAIL",  # LLM doesn't know about alt-block; P198bh rescues downstream
    },
    {
        "label": "Permissive: Charter Party B/L is acceptable",
        "cond": "Charter Party B/L is acceptable.",
        "doc_type": "Bill of Lading",
        "doc_text": "BILL OF LADING - CHARTER PARTY\nSIGNED AS AGENT FOR THE MASTER",
        "structured": "bl_subtype.is_charter_party=True",
        "expected": "PASS",
    },
]


def run():
    print("=" * 80)
    print(f"LLM-backed dry-run: {len(scenarios)} scenarios against {LLM_URL}")
    print("=" * 80)
    match = 0
    issues = []
    for i, s in enumerate(scenarios, 1):
        family = BL_FAMILY_PACK if 'bill of lading' in s["doc_type"].lower() else ""
        prompt = PROMPT_TEMPLATE.format(
            cond=s["cond"],
            doc_type=s["doc_type"],
            doc_text=s["doc_text"][:3500],
            structured=s["structured"],
            family_pack=family,
        )
        print(f"\n[{i:02d}] {s['label']}")
        print(f"     expected: {s['expected']}")
        r = ask_llm(prompt)
        verdict = (r.get("verdict") or "").upper()
        findings = (r.get("findings") or "")[:160]
        elapsed = r.get("elapsed", 0)
        if verdict == s["expected"]:
            match += 1
            print(f"     got:      {verdict}  ({elapsed:.1f}s)  ✓")
        else:
            issues.append((i, s["label"], s["expected"], verdict, findings))
            print(f"     got:      {verdict}  ({elapsed:.1f}s)  ✗")
        print(f"     LLM says: {findings}")
    print()
    print("=" * 80)
    print(f"LLM verdict match: {match}/{len(scenarios)}")
    print("=" * 80)
    if issues:
        print()
        print("Mismatches (where post-check rescues must do the work):")
        for i, label, exp, got, fnd in issues:
            print(f"  [{i:02d}] {label}: expected {exp}, LLM said {got}")
            print(f"        findings: {fnd}")


if __name__ == "__main__":
    run()
