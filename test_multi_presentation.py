"""Probe how the LLM + P195 prompt handle multi-presentation scenarios
the pipeline doesn't explicitly support at the system level. These tests
send realistic condition + document text directly to the Text LLM and
see whether it reasons about them correctly from the prompt alone.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(r"d:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final")
sys.path.insert(0, str(ROOT))

from config.settings import QWEN_TEXT_LLM_URL, QWEN_TEXT_LLM_MODEL  # noqa
from steps.step14_verification import CORE_VERIFICATION_PROMPT  # noqa
import requests  # noqa


CASES = [
    {
        "name": "Short shipment within F43P partial-allowed tolerance (expect PASS)",
        "condition_text": (
            "This is a 2nd presentation. Total LC amount is USD 255,000.00. "
            "1st presentation drew USD 150,000.00; remaining available is "
            "USD 105,000.00. Commercial Invoice of 2nd presentation must "
            "not exceed the remaining available amount."
        ),
        "clause_ref": "F32B",
        "lc_field_value": "USD 255,000.00 total; remaining USD 105,000.00",
        "lc_parties": "",
        "f47a_context": "PARTIAL SHIPMENTS ALLOWED.",
        "document_type": "Commercial Invoice",
        "structured_facts": '{"amount": "USD 90000.00"}',
        "document_text": "2nd Presentation — Commercial Invoice for USD 90,000.00 (under remaining 105,000.00)",
        "visual_metadata": "",
        "expected_verdict": "PASS",
    },
    {
        "name": "2nd presentation OVERDRAWING remaining balance (expect FAIL)",
        "condition_text": (
            "This is a 2nd presentation. LC total: USD 255,000. Prior "
            "presentation drew USD 200,000. Remaining available: USD 55,000. "
            "Commercial Invoice must not overdraw the remaining LC balance."
        ),
        "clause_ref": "F32B",
        "lc_field_value": "Remaining USD 55,000.00",
        "lc_parties": "",
        "f47a_context": "",
        "document_type": "Commercial Invoice",
        "structured_facts": '{"amount": "USD 70000.00"}',
        "document_text": "Commercial Invoice for USD 70,000.00 (2nd presentation — overdraws remaining 55,000)",
        "visual_metadata": "",
        "expected_verdict": "FAIL",
    },
    {
        "name": "Amendment between 1st and 2nd — increased LC amount (expect PASS)",
        "condition_text": (
            "Between 1st and 2nd presentation an MT707 amendment increased "
            "LC amount from USD 255,000 to USD 350,000. 2nd presentation "
            "Commercial Invoice must fit within the AMENDED LC amount, "
            "not the original."
        ),
        "clause_ref": "F32B",
        "lc_field_value": "Amended LC amount: USD 350,000.00",
        "lc_parties": "",
        "f47a_context": "MT707 AMENDMENT 001 DATED 2026-03-15: F32B AMOUNT INCREASED FROM USD 255,000 TO USD 350,000.",
        "document_type": "Commercial Invoice",
        "structured_facts": '{"amount": "USD 120000.00"}',
        "document_text": "2nd Presentation CI: USD 120,000.00 (within amended ceiling of USD 350,000)",
        "visual_metadata": "",
        "expected_verdict": "PASS",
    },
    {
        "name": "Incomplete shipping docs — cover letter says 'to follow in 2nd lot' (expect REVIEW or PASS)",
        "condition_text": (
            "Presentation must include full set of shipping documents: "
            "Commercial Invoice, Packing List, Bill of Lading, Certificate "
            "of Origin. However, the cover letter explicitly states that "
            "remaining documents will follow in the 2nd lot. Under this "
            "condition, incomplete-set is NOT a discrepancy on this lot."
        ),
        "clause_ref": "46A-1",
        "lc_field_value": "full set of shipping documents",
        "lc_parties": "",
        "f47a_context": "",
        "document_type": "Documentary Remittance",
        "structured_facts": "",
        "document_text": (
            "COVERING SCHEDULE\n"
            "THE FOLLOWING ARE BEING SENT IN THIS LOT:\n"
            " - Commercial Invoice\n - Packing List\n"
            "CERTIFICATE OF ORIGIN AND BILL OF LADING WILL FOLLOW IN THE 2ND LOT."
        ),
        "visual_metadata": "",
        "expected_verdict": "PASS",
    },
    {
        "name": "Conflict between BL and Commercial Invoice weights (expect FAIL)",
        "condition_text": (
            "Details across shipping documents must be consistent. Bill of "
            "Lading net weight must match Commercial Invoice net weight."
        ),
        "clause_ref": "47A-5",
        "lc_field_value": "Consistent weight across docs",
        "lc_parties": "",
        "f47a_context": "",
        "document_type": "Commercial Invoice",
        "structured_facts": '{"net_weight_ci": "249,500 KG", "net_weight_bl_from_set": "248,100 KG"}',
        "document_text": (
            "Commercial Invoice: Net Weight 249,500 KG\n"
            "Bill of Lading (other document in set): Net Weight 248,100 KG\n"
            "These two documents in the same presentation show conflicting net weights."
        ),
        "visual_metadata": "",
        "expected_verdict": "FAIL",
    },
    {
        "name": "Landed weight basis — final amount deferred to discharge weight (expect REVIEW)",
        "condition_text": (
            "Per F47A, payment is subject to landed weight and quality basis. "
            "Commercial Invoice amount is provisional and subject to "
            "adjustment after landing survey."
        ),
        "clause_ref": "47A-16",
        "lc_field_value": "Landed weight & quality basis",
        "lc_parties": "",
        "f47a_context": "PAYMENT OF THIS L/C IS SUBJECT TO LANDED WEIGHT AND QUALITY BASIS.",
        "document_type": "Commercial Invoice",
        "structured_facts": '{"amount": "USD 255000.00 (provisional, subject to landing)"}',
        "document_text": (
            "Provisional Commercial Invoice: USD 255,000.00\n"
            "Final amount subject to landed weight and quality per LC clause 47A-16. "
            "Landing survey pending at discharge port."
        ),
        "visual_metadata": "",
        "expected_verdict": "REVIEW",
    },
    {
        "name": "2nd presentation with 'Full and Final' cover letter (expect PASS)",
        "condition_text": (
            "Secondary presentation must include a cover letter statement "
            "confirming this is the final lot of shipping documents."
        ),
        "clause_ref": "47A-18",
        "lc_field_value": "Full and Final shipping documents statement",
        "lc_parties": "",
        "f47a_context": "",
        "document_type": "Documentary Remittance",
        "structured_facts": "",
        "document_text": (
            "COVERING SCHEDULE — 2ND PRESENTATION\n"
            "WE HEREBY CONFIRM THAT THESE ARE THE FULL AND FINAL SHIPPING "
            "DOCUMENTS FOR THIS LETTER OF CREDIT. NO FURTHER LOTS WILL FOLLOW."
        ),
        "visual_metadata": "",
        "expected_verdict": "PASS",
    },
]


def run_case(c):
    prompt = CORE_VERIFICATION_PROMPT.format(
        condition_text=c["condition_text"],
        clause_ref=c["clause_ref"],
        lc_field_value=c["lc_field_value"],
        lc_parties=c["lc_parties"],
        f47a_context=c["f47a_context"],
        document_type=c["document_type"],
        structured_facts=c["structured_facts"],
        document_text=c["document_text"],
        visual_metadata=c["visual_metadata"],
        family_pack="",
    )
    body = {
        "model": QWEN_TEXT_LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 1000,
    }
    t0 = time.time()
    try:
        r = requests.post(QWEN_TEXT_LLM_URL, json=body, timeout=300)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        dt = time.time() - t0
    except Exception as e:
        return f"REQUEST FAILED: {e}", None, 0.0
    verdict = "?"
    import re
    m = re.search(r'"verdict"\s*:\s*"([A-Za-z_]+)"', content)
    if not m:
        m = re.search(r'"compliance"\s*:\s*"([A-Za-z_]+)"', content)
    if m:
        verdict = m.group(1).upper()
    return verdict, content[:500], dt


def main():
    print(f"LLM: {QWEN_TEXT_LLM_URL}")
    print(f"Model: {QWEN_TEXT_LLM_MODEL}")
    print("=" * 70)
    results = []
    for c in CASES:
        verdict, snippet, dt = run_case(c)
        ok = verdict == c["expected_verdict"]
        mark = "OK " if ok else "XX "
        results.append(ok)
        print(f"\n{mark} {c['name']}")
        print(f"     expected={c['expected_verdict']}, got={verdict}, time={dt:.1f}s")
        if snippet:
            print(f"     response[:300]: {snippet[:300]}")
    print("\n" + "=" * 70)
    ok_n = sum(1 for x in results if x)
    print(f"MULTI-PRESENTATION RESULT: {ok_n}/{len(results)} pass")


if __name__ == "__main__":
    main()
