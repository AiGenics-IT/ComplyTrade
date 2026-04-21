"""Test the P195-strengthened CORE_VERIFICATION_PROMPT against the
live text LLM using each of the 6 worked-example scenarios.

For every scenario we send the real condition + document text and
check that the LLM returns the expected verdict (PASS or FAIL).
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
        "name": "Consignee TO ORDER without endorsement (expect FAIL)",
        "condition_text": "Bill of lading must be made out to the order of Bank Al Habib Ltd., Pakistan.",
        "clause_ref": "46A-2",
        "lc_field_value": "BANK AL HABIB LTD., PAKISTAN",
        "lc_parties": "Issuing Bank: BANK AL HABIB LTD.\nBeneficiary: OLAM GLOBAL AGRI PTE LTD",
        "f47a_context": "",
        "document_type": "Bill of Lading",
        "structured_facts": '{"consignee": "TO ORDER", "notify_party": "TRANSSSION TECNO ELECTRONICS (PVT) LTD. AND BANK AL HABIB LTD., PAKISTAN"}',
        "document_text": "CONSIGNEE: TO ORDER\nNOTIFY PARTY: TRANSSSION TECNO ELECTRONICS (PVT) LTD. AND BANK AL HABIB LTD., PAKISTAN\nSHIPPER: OLAM GLOBAL AGRI PTE LTD",
        "visual_metadata": "",
        "expected_verdict": "FAIL",
    },
    {
        "name": "BL shipper is different entity from beneficiary (expect FAIL)",
        "condition_text": "Bill of Lading must show the Beneficiary as the Shipper.",
        "clause_ref": "47A-3",
        "lc_field_value": "OLAM GLOBAL AGRI PTE LTD",
        "lc_parties": "Beneficiary: OLAM GLOBAL AGRI PTE LTD, 7 STRAITS VIEW MARINA ONE, SINGAPORE",
        "f47a_context": "",
        "document_type": "Bill of Lading",
        "structured_facts": '{"shipper": "PT CITRA BORNEO UTAMA TBK"}',
        "document_text": "SHIPPER: PT CITRA BORNEO UTAMA TBK\nJALAN PELABUHAN ASDP, KALIMANTAN INDONESIA",
        "visual_metadata": "",
        "expected_verdict": "FAIL",
    },
    {
        "name": "HS code last-digit difference (expect FAIL)",
        "condition_text": "H.S. Code No. 9018.9050 must appear on the Bill of Lading.",
        "clause_ref": "47A-2",
        "lc_field_value": "9018.9050",
        "lc_parties": "",
        "f47a_context": "",
        "document_type": "Bill of Lading",
        "structured_facts": '{"hs_codes": ["9018909000"]}',
        "document_text": "HS CODE: 9018909000",
        "visual_metadata": "",
        "expected_verdict": "FAIL",
    },
    {
        "name": "NTN present but GST missing (expect FAIL)",
        "condition_text": "GST No. 03-00-3075811-17 must appear on the Bill of Lading.",
        "clause_ref": "46A-2",
        "lc_field_value": "03-00-3075811-17",
        "lc_parties": "",
        "f47a_context": "",
        "document_type": "Bill of Lading",
        "structured_facts": '{"ntn_number": "3075811-4"}',
        "document_text": "NTN NO. 3075811-4\nShipper: OLAM GLOBAL AGRI PTE LTD",
        "visual_metadata": "",
        "expected_verdict": "FAIL",
    },
    {
        "name": "CFR invoice without separate freight line (expect FAIL)",
        "condition_text": "Freight value should be mentioned on Commercial Invoice separately.",
        "clause_ref": "47A-8",
        "lc_field_value": "Freight separately",
        "lc_parties": "",
        "f47a_context": "",
        "document_type": "Commercial Invoice",
        "structured_facts": '{"amount": "USD 97,216.00", "incoterms": "CFR KARACHI"}',
        "document_text": "TOTAL CFR KARACHI: USD 97,216.00\nInvoice Number: XPK-TRZ26030703",
        "visual_metadata": "",
        "expected_verdict": "FAIL",
    },
    {
        "name": "Legitimate name prefix (LC truncated) (expect PASS)",
        "condition_text": "Shipment Advice must be addressed to the Applicant.",
        "clause_ref": "46A-3",
        "lc_field_value": "SINDH INSTITUTE OF UROLOGY AND",
        "lc_parties": "Applicant: SINDH INSTITUTE OF UROLOGY AND TRANSPLANTATION (SIUT), KARACHI, PAKISTAN",
        "f47a_context": "",
        "document_type": "Shipment Advice",
        "structured_facts": "",
        "document_text": "TO: SINDH INSTITUTE OF UROLOGY AND TRANSPLANTATION (SIUT), KARACHI, PAKISTAN\nSubject: Shipment Advice for L/C 0401ILC083248",
        "visual_metadata": "",
        "expected_verdict": "PASS",
    },
    {
        "name": "BL notify includes issuing bank (expect PASS)",
        "condition_text": "Bill of Lading must be marked notify Bank Al Habib Ltd.",
        "clause_ref": "46A-2",
        "lc_field_value": "BANK AL HABIB LTD.",
        "lc_parties": "Issuing Bank: BANK AL HABIB LTD., PAKISTAN",
        "f47a_context": "",
        "document_type": "Bill of Lading",
        "structured_facts": '{"notify_party": "TRANSSSION TECNO ELECTRONICS AND BANK AL HABIB LTD., KARACHI, PAKISTAN"}',
        "document_text": "NOTIFY PARTY: TRANSSSION TECNO ELECTRONICS (PVT) LTD. AND BANK AL HABIB LTD., KARACHI, PAKISTAN",
        "visual_metadata": "",
        "expected_verdict": "PASS",
    },
    {
        "name": "BL notify does NOT include issuing bank (expect FAIL)",
        "condition_text": "Bill of Lading must be marked notify Bank Al Habib Ltd.",
        "clause_ref": "46A-2",
        "lc_field_value": "BANK AL HABIB LTD.",
        "lc_parties": "Issuing Bank: BANK AL HABIB LTD.",
        "f47a_context": "",
        "document_type": "Bill of Lading",
        "structured_facts": '{"notify_party": "TRANSSSION TECNO ELECTRONICS (PVT) LTD."}',
        "document_text": "NOTIFY PARTY: TRANSSSION TECNO ELECTRONICS (PVT) LTD.",
        "visual_metadata": "",
        "expected_verdict": "FAIL",
    },
    {
        "name": "BL marked FREIGHT PREPAID (expect PASS)",
        "condition_text": "Bill of Lading must be marked 'Freight Prepaid'.",
        "clause_ref": "46A-2",
        "lc_field_value": "Freight Prepaid",
        "lc_parties": "",
        "f47a_context": "",
        "document_type": "Bill of Lading",
        "structured_facts": '{"freight_terms": "FREIGHT PREPAID"}',
        "document_text": "FREIGHT TERMS: FREIGHT PREPAID\nShipper has paid all freight.",
        "visual_metadata": "",
        "expected_verdict": "PASS",
    },
    {
        "name": "BL with Freight COLLECT when LC wants PREPAID (expect FAIL)",
        "condition_text": "Bill of Lading must be marked 'Freight Prepaid'.",
        "clause_ref": "46A-2",
        "lc_field_value": "Freight Prepaid",
        "lc_parties": "",
        "f47a_context": "",
        "document_type": "Bill of Lading",
        "structured_facts": '{"freight_terms": "FREIGHT COLLECT"}',
        "document_text": "FREIGHT TERMS: FREIGHT COLLECT",
        "visual_metadata": "",
        "expected_verdict": "FAIL",
    },
    {
        "name": "HS code exact match with trailing zeros (expect PASS)",
        "condition_text": "H.S. Code No. 9018.9050 must appear on the Commercial Invoice.",
        "clause_ref": "47A-2",
        "lc_field_value": "9018.9050",
        "lc_parties": "",
        "f47a_context": "",
        "document_type": "Commercial Invoice",
        "structured_facts": '{"hs_codes": ["9018905000"]}',
        "document_text": "HS CODE: 9018905000",
        "visual_metadata": "",
        "expected_verdict": "PASS",
    },
    {
        "name": "Invoice origin single country when multi required (expect FAIL)",
        "condition_text": "Commercial Invoice must certify Malaysia and Indonesia origin.",
        "clause_ref": "46A-1",
        "lc_field_value": "Malaysia and Indonesia origin",
        "lc_parties": "",
        "f47a_context": "",
        "document_type": "Commercial Invoice",
        "structured_facts": '{"key_clauses": ["WE CERTIFY THAT MERCHANDISE ARE OF INDONESIA ORIGIN"]}',
        "document_text": "WE CERTIFY THAT MERCHANDISE ARE OF INDONESIA ORIGIN\nHS CODE 1511.9030",
        "visual_metadata": "",
        "expected_verdict": "FAIL",
    },
    {
        "name": "Invoice covers both required origins (expect PASS)",
        "condition_text": "Commercial Invoice must certify Malaysia and Indonesia origin.",
        "clause_ref": "46A-1",
        "lc_field_value": "Malaysia and Indonesia origin",
        "lc_parties": "",
        "f47a_context": "",
        "document_type": "Commercial Invoice",
        "structured_facts": '{"key_clauses": ["MERCHANDISE ARE OF MALAYSIA AND INDONESIA ORIGIN"]}',
        "document_text": "WE CERTIFY THAT MERCHANDISE ARE OF MALAYSIA AND INDONESIA ORIGIN",
        "visual_metadata": "",
        "expected_verdict": "PASS",
    },
    {
        "name": "Blank-back BL forbidden and BL IS blank back (expect FAIL)",
        "condition_text": "Bill of Lading must not be blank back.",
        "clause_ref": "47A-1",
        "lc_field_value": "NOT BLANK BACK",
        "lc_parties": "",
        "f47a_context": "STALE, CLAUSED, BLANK BACK, SHORT FORM AND/OR CHARTER PARTY BILL OF LADING NOT ACCEPTABLE.",
        "document_type": "Bill of Lading",
        "structured_facts": '{"bl_subtype": {"has_terms_overleaf": false, "is_blank_back": true, "is_short_form": false, "form_type": "blank_back"}}',
        "document_text": "(no T&C page attached; front only)",
        "visual_metadata": "",
        "expected_verdict": "FAIL",
    },
    {
        "name": "Short-form BL forbidden but BL has T&C overleaf (expect PASS)",
        "condition_text": "Bill of Lading must not be short form.",
        "clause_ref": "47A-1",
        "lc_field_value": "NOT SHORT FORM",
        "lc_parties": "",
        "f47a_context": "",
        "document_type": "Bill of Lading",
        "structured_facts": '{"bl_subtype": {"has_terms_overleaf": true, "is_blank_back": false, "is_short_form": false, "form_type": "long_form_printed_overleaf"}}',
        "document_text": "(BL with carriage T&Cs on separate overleaf page)",
        "visual_metadata": "",
        "expected_verdict": "PASS",
    },
    {
        "name": "Consignee explicitly 'TO ORDER OF BANK' (expect PASS)",
        "condition_text": "Bill of Lading must be made out to the order of Bank Al Habib Ltd., Pakistan.",
        "clause_ref": "46A-2",
        "lc_field_value": "BANK AL HABIB LTD., PAKISTAN",
        "lc_parties": "Issuing Bank: BANK AL HABIB LTD., PAKISTAN",
        "f47a_context": "",
        "document_type": "Bill of Lading",
        "structured_facts": '{"consignee": "TO THE ORDER OF BANK AL HABIB LTD., KARACHI, PAKISTAN"}',
        "document_text": "CONSIGNEE: TO THE ORDER OF BANK AL HABIB LTD., KARACHI, PAKISTAN",
        "visual_metadata": "",
        "expected_verdict": "PASS",
    },
    {
        "name": "Date reference mismatch (policy number) (expect FAIL)",
        "condition_text": "Commercial Invoice must reference Open Policy No. 2023008MIPD000453.",
        "clause_ref": "46A-3",
        "lc_field_value": "2023008MIPD000453",
        "lc_parties": "",
        "f47a_context": "",
        "document_type": "Commercial Invoice",
        "structured_facts": '{"references_found": [{"role": "invoice_reference", "value": "XPK-TR26030303"}]}',
        "document_text": "INVOICE NO. XPK-TR26030303\nDate: 03-03-2026\n(no open policy reference anywhere)",
        "visual_metadata": "",
        "expected_verdict": "FAIL",
    },
    {
        "name": "Policy reference present with OCR O↔0 (expect PASS)",
        "condition_text": "Commercial Invoice must reference Open Policy No. 2023008MIPD000453.",
        "clause_ref": "46A-3",
        "lc_field_value": "2023008MIPD000453",
        "lc_parties": "",
        "f47a_context": "",
        "document_type": "Commercial Invoice",
        "structured_facts": '{"references_found": [{"role": "open_policy_reference", "value": "2023008MIPDO00453"}]}',
        "document_text": "OPEN POLICY NO. 2023008MIPDO00453 (OCR misread O for 0)",
        "visual_metadata": "",
        "expected_verdict": "PASS",
    },
    {
        "name": "Applicant address variation (expect PASS)",
        "condition_text": "Bill of Lading must be marked notify the Applicant.",
        "clause_ref": "46A-2",
        "lc_field_value": "TRANSSSION TECNO ELECTRONICS (PRIVATE) LIMITED. PLOT NO. 259/E, BLOCK-6, P.E.C.H.S OFF SHAHRAH-E-FAISAL KARACHI, PAKISTAN",
        "lc_parties": "Applicant: TRANSSSION TECNO ELECTRONICS (PRIVATE) LIMITED. PLOT NO. 259/E, BLOCK-6",
        "f47a_context": "",
        "document_type": "Bill of Lading",
        "structured_facts": '{"notify_party": "TRANS SION TECNO ELECTRONICS (PRIVATE) LIMITED, PLOT NO.259/E, BLOCK-6, SHAHRAH-E-FAISAL KARACHI, PAKISTAN"}',
        "document_text": "NOTIFY PARTY: TRANS SION TECNO ELECTRONICS (PRIVATE) LIMITED, PLOT NO.259/E, BLOCK-6, SHAHRAH-E-FAISAL KARACHI, PAKISTAN",
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
        "max_tokens": 800,
    }
    t0 = time.time()
    try:
        r = requests.post(QWEN_TEXT_LLM_URL, json=body, timeout=300)
        r.raise_for_status()
        resp = r.json()
        content = resp["choices"][0]["message"]["content"]
        dt = time.time() - t0
    except Exception as e:
        return f"REQUEST FAILED: {e}", None, 0.0

    # Parse verdict from response (it returns JSON with a verdict key)
    verdict = "?"
    try:
        import re
        m = re.search(r'"verdict"\s*:\s*"([A-Za-z_]+)"', content)
        if not m:
            m = re.search(r'"compliance"\s*:\s*"([A-Za-z_]+)"', content)
        if m:
            verdict = m.group(1).upper()
    except Exception:
        pass
    return verdict, content[:400], dt


def main():
    print(f"LLM: {QWEN_TEXT_LLM_URL}")
    print(f"Model: {QWEN_TEXT_LLM_MODEL}")
    print("=" * 60)
    results = []
    for c in CASES:
        verdict, snippet, dt = run_case(c)
        ok = verdict == c["expected_verdict"]
        mark = "OK " if ok else "XX "
        results.append(ok)
        print(f"\n{mark} {c['name']}")
        print(f"     expected={c['expected_verdict']}, got={verdict}, time={dt:.1f}s")
        if snippet:
            # show just the verdict + findings line
            print(f"     response[:200]: {snippet[:200]}")
    print("\n" + "=" * 60)
    ok_n = sum(1 for x in results if x)
    print(f"RESULT: {ok_n}/{len(results)} pass")


if __name__ == "__main__":
    main()
