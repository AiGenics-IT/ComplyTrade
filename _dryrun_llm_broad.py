"""Broad LLM-backed dry-run across all major LC check categories.

30+ scenarios covering: documentary requirements, prohibitions, permissive
carve-outs, amounts, dates, parties, transport, insurance, OCR variants,
negations, tricky wording. Each scenario carries an expected verdict and
the actual LLM verdict is compared.

Usage: python _dryrun_llm_broad.py
"""
import json
import re
import time
import requests

LLM_URL = "http://10.20.10.4/llm/v1/chat/completions"
LLM_MODEL = "Qwen2.5-72B-Instruct-GPTQ-Int8"
TIMEOUT = 90

SHIPMENT_ADVICE_FAMILY_PACK = """SHIPMENT ADVICE — additional rules:

POLICY / COVER NOTE / OPEN POLICY NUMBER — INTERCHANGEABLE LABELS:
When the LC requires the Shipment Advice to reference an insurance identifier,
that same reference number may appear on the document under ANY of these
interchangeable labels: Policy No. / Open Policy No. / Cover Note No. /
Insurance Policy No. / Marine Policy No.
If the NUMBER matches (character-for-character, ignoring O↔0 OCR variance),
the document references the required policy → PASS, regardless of which label
precedes it. Do NOT FAIL because the condition says "Policy No." and the
document uses "Cover Note No." — the label is interchangeable; the number binds.
"""

BL_FAMILY_PACK = """BILL OF LADING — additional rules:

NVOCC / FIATA / FREIGHT FORWARDER — DISTINGUISH DEFINITION FROM EVIDENCE:
BL T&C pages carry glossary definitions like
    "NVOCC" MEANS NON VESSEL OPERATING COMMON CARRIER.
These are boilerplate, NOT evidence. A BL is actually an NVOCC/FIATA/
House/FF BL ONLY when the term appears in:
  - ISSUER / CARRIER letterhead or identification block
  - SIGNATURE block ("SIGNED AS FREIGHT FORWARDER", "AS NVOCC")
  - STAMP / SEAL identifying the BL type on the face
  - Explicit BL-class title ("HOUSE BILL OF LADING" as the printed title)

For a prohibitive condition "NVOCC/FF/FIATA/House BL NOT ACCEPTABLE":
  - Token ONLY in \"<TERM>\" MEANS ... / DEFINITIONS block → PASS
  - Token in issuer line or signature block or title → FAIL
  - Real ocean carrier as issuer (Maersk / MSC / CMA CGM / COSCO / OOCL /
    Hapag-Lloyd / ONE / Evergreen / PIL / Yang Ming / ZIM / HMM / APL)
    AND token only in T&C → PASS

CONSIGNEE "TO ORDER" WITHOUT NAMED BANK:
If LC requires BL made out "TO THE ORDER OF <BANK>" and the consignee
shows just "TO ORDER" with no named party and no explicit endorsement
line to <BANK> on the face, verdict is FAIL — NOT REVIEW.

CONTAINER / SEAL NUMBERS (ISO 6346):
Container numbers on ocean BLs follow ISO 6346 (4 uppercase letters + 6 or
7 digits, e.g. YMLU8681239, YMAV443317). They often appear in the running
particulars text ("YMLU8681239 40'HQ FCL") rather than under a CONTAINER
NO: label. If at least one ISO 6346 code is present anywhere on the BL,
the container-number requirement is satisfied.
"""

DRAFT_FAMILY_PACK = """DRAFT / BILL OF EXCHANGE — additional rules:

THIRD-PARTY EXCEPTION — DRAFT MUST BE BENEFICIARY-DRAWN:
Clauses like "Third party documents are acceptable EXCEPT Draft and
Invoice" do NOT mean Drafts are inadmissible. They mean that while
most docs may be third-party, the DRAFT (and INVOICE) must be issued
by the BENEFICIARY — not a third party.
Verdict logic:
  - Draft drawer / issuer matches LC beneficiary (F59) → PASS
  - Draft drawer / issuer is a different entity       → FAIL
Look at "FOR AND ON BEHALF OF <X>", the drawer signature line, and
parties_found[role=drawer]. Never write "Draft is not acceptable
according to the condition" solely because the condition mentions
Draft in the exception list.
"""

PROMPT_TEMPLATE = """You are a trade finance document examiner under UCP 600.
Evaluate whether the document satisfies the LC condition.

LC CONDITION:
{cond}

DOCUMENT TYPE: {doc_type}

DOCUMENT TEXT (excerpt):
---
{doc_text}
---

STRUCTURED FACTS:
{structured}

{family_pack}

RULES:
- Return strict JSON: {{"verdict": "PASS"|"FAIL"|"REVIEW", "findings": "<=150 chars"}}
- PASS when the document clearly satisfies the condition
- FAIL when the document clearly violates the condition
- REVIEW only when the evidence is genuinely ambiguous
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
    try:
        resp = requests.post(LLM_URL, json=payload, timeout=TIMEOUT)
    except Exception as e:
        return {"verdict": "ERROR", "findings": str(e)[:200],
                "elapsed": time.time() - t0, "raw": ""}
    elapsed = time.time() - t0
    if resp.status_code != 200:
        return {"verdict": "ERROR", "findings": f"HTTP {resp.status_code}",
                "elapsed": elapsed, "raw": resp.text[:200]}
    raw = resp.json()["choices"][0]["message"]["content"].strip()
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        return {"verdict": "PARSE_FAIL", "findings": "no JSON",
                "elapsed": elapsed, "raw": raw}
    try:
        parsed = json.loads(m.group(0))
    except Exception as e:
        return {"verdict": "PARSE_FAIL", "findings": str(e),
                "elapsed": elapsed, "raw": raw}
    parsed["elapsed"] = elapsed
    parsed["raw"] = raw
    return parsed


scenarios = [
    # ── 1. Documentary: LC number must appear on all documents ──
    dict(
        label="LC number present on Draft",
        doc_type="Draft Bill of Exchange",
        cond="Draft must show the Documentary Credit number 0007LC55189/2025.",
        doc_text=("Bill of Exchange\nDrawn under LC No. 0007LC55189/2025\n"
                  "Beneficiary: SABIC\nDrawn on: BANK AL HABIB LIMITED"),
        structured="references_found[role=lc_reference]=0007LC55189/2025",
        expected="PASS",
    ),
    dict(
        label="LC number missing on Packing List",
        doc_type="Packing List",
        cond="Packing List must show the Documentary Credit number 0007LC55189/2025.",
        doc_text=("Packing List No. PL-001\nBeneficiary: SABIC\n"
                  "Cargo: 25.500 MT LDPE\n100 pallets"),
        structured="references_found: none for lc_reference",
        expected="FAIL",
    ),

    # ── 2. HS code ──
    dict(
        label="HS code 1511.9020 matches on invoice",
        doc_type="Commercial Invoice",
        cond="Commercial Invoice must state H.S. Code 1511.9020.",
        doc_text="Invoice No. 001\nGoods: RBD Palm Olein\nH.S. CODE 1511.9020\nQty: 25 MT",
        structured="references_found[role=hs_code]=1511.9020",
        expected="PASS",
    ),
    dict(
        label="HS code differs (1511.9030 vs 1511.9020)",
        doc_type="Commercial Invoice",
        cond="Commercial Invoice must state H.S. Code 1511.9020.",
        doc_text="Invoice No. 001\nH.S. CODE 1511.9030\nGoods: RBD Palm Olein",
        structured="references_found[role=hs_code]=1511.9030",
        expected="FAIL",
    ),

    # ── 3. Charter Party permissive vs prohibitive ──
    dict(
        label="47A says CPBL acceptable, BL is CPBL → PASS",
        doc_type="Bill of Lading",
        cond="Charter Party B/L is acceptable.",
        doc_text=("BILL OF LADING — CHARTER PARTY B/L\n"
                  "MV SEA QUEEN\nSIGNED AS AGENT FOR THE MASTER"),
        structured="bl_subtype.is_charter_party=True",
        expected="PASS",
    ),
    dict(
        label="47A prohibits CPBL, BL is CPBL → FAIL",
        doc_type="Bill of Lading",
        cond="Charter Party Bills of Lading are not acceptable.",
        doc_text=("BILL OF LADING — CHARTER PARTY B/L\n"
                  "MV SEA QUEEN\nSIGNED AS AGENT FOR THE MASTER"),
        structured="bl_subtype.is_charter_party=True",
        expected="FAIL",
    ),

    # ── 4. NVOCC boilerplate vs real ──
    dict(
        label="NVOCC only in T&C glossary (real carrier issuer) → PASS",
        doc_type="Bill of Lading",
        cond="Bills of Lading stated to be issued by a non-vessel operating carrier company are not acceptable.",
        doc_text=(
            "MAERSK LINE\nBILL OF LADING NO. MAEU1234567\n"
            "SIGNED AS AGENT FOR AND ON BEHALF OF THE MASTER\n"
            "TERMS AND CONDITIONS:\n"
            "\"NVOCC\" MEANS NON VESSEL OPERATING COMMON CARRIER."
        ),
        structured="issuer=MAERSK LINE; signing_type=agent_for_master",
        expected="PASS",
    ),
    dict(
        label="Real NVOCC issuer → FAIL",
        doc_type="Bill of Lading",
        cond="Bills of Lading stated to be issued by a non-vessel operating carrier company are not acceptable.",
        doc_text=(
            "XYZ LOGISTICS LLC — NON VESSEL OPERATING COMMON CARRIER\n"
            "BILL OF LADING NO. XYZ-001\nLicensed NVOCC FMC No. 12345\n"
        ),
        structured="issuer=XYZ LOGISTICS (NVOCC)",
        expected="FAIL",
    ),

    # ── 5. House BL prohibition ──
    dict(
        label="House BL title → FAIL",
        doc_type="Bill of Lading",
        cond="House B/L is not acceptable.",
        doc_text="HOUSE BILL OF LADING NO. HBL-001\nIssued by ACME LOGISTICS",
        structured="bl_subtype.issuer_type=house_bl",
        expected="FAIL",
    ),

    # ── 6. Freight wording ──
    dict(
        label="Freight PREPAID required + present",
        doc_type="Bill of Lading",
        cond="BL must show FREIGHT PREPAID.",
        doc_text="MAERSK LINE\nBL No. 123\nFREIGHT PREPAID AT ORIGIN",
        structured="bl_subtype.is_freight_prepaid=True",
        expected="PASS",
    ),
    dict(
        label="Freight PREPAID required + only COLLECT present",
        doc_type="Bill of Lading",
        cond="BL must show FREIGHT PREPAID.",
        doc_text="MAERSK LINE\nBL No. 123\nFREIGHT COLLECT AT DESTINATION",
        structured="bl_subtype.is_freight_prepaid=False",
        expected="FAIL",
    ),
    dict(
        label="Generic FREIGHT PAYABLE satisfied by AS PER CHARTER PARTY",
        doc_type="Bill of Lading",
        cond="BL must show freight payable.",
        doc_text=("MV SEA QUEEN\nFREIGHT PAYABLE AS PER CHARTER PARTY "
                  "DATED 28 NOV 2024"),
        structured="bl_subtype.contract_type=charter_party",
        expected="PASS",
    ),

    # ── 7. Consignee TO ORDER OF ──
    dict(
        label="Consignee: TO THE ORDER OF BANK AL HABIB (named) → PASS",
        doc_type="Bill of Lading",
        cond="BL must be consigned to the order of Bank Al Habib Ltd., Karachi, Pakistan.",
        doc_text="CONSIGNEE: TO THE ORDER OF BANK AL HABIB LTD., KARACHI, PAKISTAN",
        structured='unified_summary.consignee="TO THE ORDER OF BANK AL HABIB LTD., KARACHI, PAKISTAN"',
        expected="PASS",
    ),
    dict(
        label="Consignee: TO ORDER only, bank in NOTIFY → FAIL",
        doc_type="Bill of Lading",
        cond="BL must be consigned to the order of Bank Al Habib Ltd., Karachi, Pakistan.",
        doc_text=("CONSIGNEE: TO ORDER\n"
                  "NOTIFY PARTY: BANK AL HABIB LTD., KARACHI\n"
                  "SHIPPER: ABC TEXTILES"),
        structured='unified_summary.consignee="TO ORDER" notify_party="BANK AL HABIB"',
        expected="FAIL",
    ),

    # ── 8. Shipper / Beneficiary match ──
    dict(
        label="Shipper matches beneficiary on BL",
        doc_type="Bill of Lading",
        cond="Shipper must match LC beneficiary SAUDI BASIC INDUSTRIES CORPORATION (SABIC).",
        doc_text=("SHIPPER: SAUDI BASIC INDUSTRIES CORPORATION (SABIC)\n"
                  "P.O. Box 59090 Riyadh 11525"),
        structured='parties_found[role=shipper].name="SAUDI BASIC INDUSTRIES CORPORATION"',
        expected="PASS",
    ),
    dict(
        label="Shipper differs from beneficiary → FAIL",
        doc_type="Bill of Lading",
        cond="Shipper must match LC beneficiary SAUDI BASIC INDUSTRIES CORPORATION.",
        doc_text="SHIPPER: ACME TRADING CO., DUBAI",
        structured='parties_found[role=shipper].name="ACME TRADING CO"',
        expected="FAIL",
    ),

    # ── 9. Port checks ──
    dict(
        label="Port of loading matches LC (Indonesia)",
        doc_type="Bill of Lading",
        cond="Port of Loading must be any Indonesian seaport.",
        doc_text=("PORT OF LOADING: BALIKPAPAN, INDONESIA\n"
                  "PORT OF DISCHARGE: PORT QASIM, PAKISTAN"),
        structured="references_found[role=port_of_loading]=BALIKPAPAN, INDONESIA",
        expected="PASS",
    ),
    dict(
        label="Port of loading does NOT match",
        doc_type="Bill of Lading",
        cond="Port of Loading must be any Indonesian seaport.",
        doc_text="PORT OF LOADING: JEBEL ALI, UAE\nPORT OF DISCHARGE: PORT QASIM",
        structured="references_found[role=port_of_loading]=JEBEL ALI, UAE",
        expected="FAIL",
    ),

    # ── 10. Quantity + tolerance ──
    dict(
        label="Quantity within +2% tolerance",
        doc_type="Commercial Invoice",
        cond="Quantity must be 250 MT (±2% tolerance allowed per F39A/F47A).",
        doc_text="QTY: 253.00 MT\nRate: USD 1,140/MT\nTotal USD 288,420.00",
        structured="quantity=253 MT (within +2% of 250)",
        expected="PASS",
    ),
    dict(
        label="Quantity exceeds +2% tolerance",
        doc_type="Commercial Invoice",
        cond="Quantity must be 250 MT (±2% tolerance).",
        doc_text="QTY: 270.00 MT\nRate: USD 1,140/MT",
        structured="quantity=270 MT (exceeds +2% = 255 MT)",
        expected="FAIL",
    ),

    # ── 11. Date — issue date before LC expiry ──
    dict(
        label="Draft dated within LC validity",
        doc_type="Draft Bill of Exchange",
        cond="Draft must be dated on or before LC expiry date (2025-04-01).",
        doc_text="Bill of Exchange\nDated: 18 February 2025\nAmount USD 29,070.00",
        structured="dates_found[role=issue_date]=2025-02-18",
        expected="PASS",
    ),
    dict(
        label="Document dated AFTER LC expiry",
        doc_type="Commercial Invoice",
        cond="Documents must be presented on or before LC expiry date (2025-04-01).",
        doc_text="Commercial Invoice dated 15 April 2025\nAmount USD 29,070.00",
        structured="dates_found[role=issue_date]=2025-04-15",
        expected="FAIL",
    ),

    # ── 12. Currency check ──
    dict(
        label="Invoice currency USD matches LC",
        doc_type="Commercial Invoice",
        cond="Invoice currency must be USD.",
        doc_text="Invoice Total: USD 29,070.00\nCurrency: US Dollar",
        structured="amounts_found[role=invoice_total].currency=USD",
        expected="PASS",
    ),
    dict(
        label="Invoice currency EUR ≠ LC USD → FAIL",
        doc_type="Commercial Invoice",
        cond="Invoice currency must be USD.",
        doc_text="Invoice Total: EUR 27,500.00\nCurrency: Euro",
        structured="amounts_found[role=invoice_total].currency=EUR",
        expected="FAIL",
    ),

    # ── 13. Third-party docs acceptable except draft/invoice ──
    dict(
        label="Third party Certificate of Origin (acceptable)",
        doc_type="Certificate of Origin",
        cond="Third party documents are acceptable except for Invoice, Draft, and Bill of Lading.",
        doc_text="CERTIFICATE OF ORIGIN\nIssued by: CHAMBER OF COMMERCE LAHORE\nBeneficiary's country",
        structured="issuer=Chamber of Commerce (third party, not beneficiary)",
        expected="PASS",
    ),

    # ── 14. Permissive that cannot FAIL ──
    dict(
        label="Permissive: Blank Back BL is acceptable (self-referential)",
        doc_type="Bill of Lading",
        cond="Blank Back B/L is acceptable.",
        doc_text="BILL OF LADING\nShort-form / Blank Back printed",
        structured="bl_subtype.is_blank_back=True",
        expected="PASS",
    ),

    # ── 15. Transshipment permissive ──
    dict(
        label="43T ALLOWED + no transshipment on BL → PASS",
        doc_type="Bill of Lading",
        cond="LC Transshipment Condition (F43T): ALLOWED. Check BL for any transshipment indication.",
        doc_text=("BILL OF LADING\nVESSEL: MV SEA QUEEN\n"
                  "PORT OF LOADING: BALIKPAPAN\nPORT OF DISCHARGE: PORT QASIM\n"
                  "No transshipment."),
        structured="bl_subtype.transshipment=False",
        expected="PASS",
    ),
    dict(
        label="43T PROHIBITED + explicit transshipment evidence → FAIL",
        doc_type="Bill of Lading",
        cond="LC Transshipment (F43T): NOT ALLOWED. Check BL for transshipment.",
        doc_text=("BILL OF LADING\nTRANSSHIPPED AT COLOMBO BY MV CONNECT\n"
                  "From MV SEA QUEEN"),
        structured="bl_subtype.transshipment=True (2 vessels named)",
        expected="FAIL",
    ),

    # ── 16. OCR variants ──
    dict(
        label="LC number OCR: O vs 0 variant matches",
        doc_type="Draft Bill of Exchange",
        cond="Draft must show LC No. 0007LC55189/2025.",
        doc_text="Drawn under Credit No. O007LC55189/2025 dated 03.01.2025",
        structured="references_found[role=lc_reference].raw='O007LC55189/2025'",
        expected="PASS",
    ),

    # ── 17. Insurance clause prohibition (not an LC compliance check) ──
    dict(
        label="Insurance covered by applicant → informational, not a doc check",
        doc_type="Commercial Invoice",
        cond="Insurance is covered by the applicant.",
        doc_text="Commercial Invoice\nCFR ANY SEAPORT IN PAKISTAN (INCOTERMS 2020)",
        structured="INCOTERMS=CFR (insurance covered by buyer)",
        expected="PASS",
    ),

    # ── 18. Notify party match ──
    dict(
        label="Notify party matches both applicant + issuing bank",
        doc_type="Bill of Lading",
        cond="BL Notify Party must be both the Applicant and Bank Al Habib Ltd., Pakistan.",
        doc_text=("NOTIFY PARTY:\n"
                  "H.SHEIKH NOOR-UD-DIN AND SONS (PVT) LTD.\n"
                  "AND BANK AL HABIB LTD., KARACHI, PAKISTAN"),
        structured='parties_found[role=notify_party].raw="H.SHEIKH NOOR-UD-DIN AND SONS (PVT) LTD. AND BANK AL HABIB LTD"',
        expected="PASS",
    ),

    # ── 19. Third-party Draft (prohibited) ──
    dict(
        label="Draft issued by beneficiary → PASS (must be beneficiary-drawn)",
        doc_type="Draft Bill of Exchange",
        cond="Third-party Drafts not acceptable. Draft must be drawn by beneficiary.",
        doc_text="Bill of Exchange\nDrawn by: SAUDI BASIC INDUSTRIES CORPORATION\n(the LC beneficiary)",
        structured='parties_found[role=drawer]="SABIC"',
        expected="PASS",
    ),
    dict(
        label="Draft drawn by a third party → FAIL",
        doc_type="Draft Bill of Exchange",
        cond="Third-party Drafts not acceptable. Draft must be drawn by the beneficiary SABIC.",
        doc_text="Bill of Exchange\nDrawn by: ACME TRADING CO., LONDON (not the beneficiary)",
        structured='parties_found[role=drawer]="ACME TRADING CO"',
        expected="FAIL",
    ),

    # ── 20. Stamp / signature required ──
    dict(
        label="Invoice signed by beneficiary → PASS",
        doc_type="Commercial Invoice",
        cond="Invoice must be signed by the beneficiary.",
        doc_text=("Commercial Invoice No. 001\n...\nFor and on behalf of SABIC\n"
                  "[SIGNATURE]\nAuthorized signatory"),
        structured="signatures=[handwritten, beneficiary]",
        expected="PASS",
    ),
    dict(
        label="Invoice unsigned → FAIL",
        doc_type="Commercial Invoice",
        cond="Invoice must be signed by the beneficiary.",
        doc_text="Commercial Invoice No. 001\nGoods: LDPE\nAmount USD 29,070\n(no signature)",
        structured="signatures=[]",
        expected="FAIL",
    ),

    # ── 21. Language check ──
    dict(
        label="Invoice in English → PASS",
        doc_type="Commercial Invoice",
        cond="All documents must be in English.",
        doc_text="COMMERCIAL INVOICE\nBeneficiary: SABIC\nGoods description: LDPE HP4024WN",
        structured="language=English",
        expected="PASS",
    ),

    # ── 22. Onboard notation ──
    dict(
        label="BL marked 'SHIPPED ON BOARD' → PASS",
        doc_type="Bill of Lading",
        cond="BL must carry an on-board notation.",
        doc_text=("BILL OF LADING\nSHIPPED ON BOARD 16 FEB 2025\n"
                  "VESSEL MV SEA QUEEN"),
        structured="dates_found[role=onboard_date]=2025-02-16",
        expected="PASS",
    ),

    # ── 23. Container number embedded in running text ──
    dict(
        label="BL container numbers in particulars (ISO 6346)",
        doc_type="Bill of Lading",
        cond="BL must show container number and seal number.",
        doc_text=(
            "BILL OF LADING\nMARKS/CONTAINER NOS\n"
            "YMLU8681239 40'HQ FCL/ FCL YMAV443317 17 PACKAGES 4137.110KGS\n"
            "SEAL NO: SL123456"),
        structured="references_found[role=container_number]=YMLU8681239, YMAV443317; seal_number=SL123456",
        expected="PASS",
    ),

    # ── 24. Container number NOT present on BL → FAIL ──
    dict(
        label="BL missing container numbers → FAIL",
        doc_type="Bill of Lading",
        cond="BL must show container number and seal number.",
        doc_text=("BILL OF LADING\nSHIPPER: ACME\nCONSIGNEE: TO ORDER\n"
                  "No container/seal details provided"),
        structured="references_found[role=container_number]=none",
        expected="FAIL",
    ),

    # ── 25. Draft third-party exception: beneficiary-drawn PASS ──
    dict(
        label="Third-party except Draft: beneficiary-drawn Draft → PASS",
        doc_type="Draft Bill of Exchange",
        cond="Third Party Documents are acceptable except Draft and Invoice.",
        doc_text=(
            "BILL OF EXCHANGE\n"
            "THIS FIRST OF EXCHANGE (SECOND OF EXCHANGE BEING UNPAID)\n"
            "PAY TO THE ORDER OF INFINIX MOBILITY LIMITED\n"
            "TO BANK AL HABIB LIMITED ISLAMIC BANKING PAKISTAN\n"
            "FOR AND ON BEHALF OF INFINIX MOBILITY LIMITED"),
        structured='parties_found[role=drawer]="INFINIX MOBILITY LIMITED" (=F59 beneficiary)',
        expected="PASS",
    ),

    # ── 26. Draft third-party exception: third-party drawn → FAIL ──
    dict(
        label="Third-party except Draft: third-party-drawn Draft → FAIL",
        doc_type="Draft Bill of Exchange",
        cond="Third Party Documents are acceptable except Draft and Invoice.",
        doc_text=(
            "BILL OF EXCHANGE\n"
            "PAY TO THE ORDER OF OVERSEAS MIDDLEMAN LLC\n"
            "FOR AND ON BEHALF OF THIRD PARTY TRADERS"),
        structured='parties_found[role=drawer]="THIRD PARTY TRADERS" (beneficiary=INFINIX MOBILITY LIMITED)',
        expected="FAIL",
    ),

    # ── 27. Applicant email variant with (AT) ──
    dict(
        label="Shipment Advice applicant email in (AT) form",
        doc_type="Shipment Advice",
        cond="Shipment Advice must also be addressed to the Applicant at ABID.HUSSAIN@TECNOPACK.COM.PK.",
        doc_text=("Shipment Advice\n"
                  "TO UBL INSURERS\n"
                  "TO H.SHEIKH NOOR-UD-DIN AND SONS (PVT) LTD\n"
                  "EMAIL: ABID.HUSSAIN(AT)TECNOPACK.COM.PK\n"),
        structured="references_found[role=applicant_email]=ABID.HUSSAIN(AT)TECNOPACK.COM.PK",
        expected="PASS",
    ),

    # ── 28. Policy No in LC → Cover Note No on doc (P198cd) ──
    dict(
        label="Cross-label: LC says Policy No, doc says COVER NOTE NO (same number)",
        doc_type="Shipment Advice",
        cond="Shipment Advice must reference Policy No. 11/0000118/1024/0-0.",
        doc_text=("Shipment Advice\nDated: 10 March 2026\n"
                  "Carrier: MAERSK LINE\nVessel: MT NCC SAMA\n"
                  "COVER NOTE NO. 11/0000118/1024/0-0\n"
                  "Insurer: CENTURY INSURANCE COMPANY LIMITED"),
        structured="references_found[role=cover_note_reference]=11/0000118/1024/0-0",
        expected="PASS",
    ),

    # ── 29. Policy No in LC → Open Policy No in doc ──
    dict(
        label="Cross-label: LC says Policy No, doc says OPEN POLICY NO",
        doc_type="Shipment Advice",
        cond="Shipment Advice must reference Policy No. 2023008MIPD000453.",
        doc_text=("Shipment Advice DD:16.02.2025\n"
                  "OPEN POLICY NO.2023008MIPDO00453\n"  # OCR O for 0
                  "L/C No: 0007LC55189/2025"),
        structured="references_found[role=open_policy_reference]=2023008MIPDO00453",
        expected="PASS",
    ),

    # ── 30. Policy No MISSING from doc entirely → FAIL ──
    dict(
        label="Policy No genuinely missing from doc → FAIL",
        doc_type="Shipment Advice",
        cond="Shipment Advice must reference Policy No. 11/0000118/1024/0-0.",
        doc_text=("Shipment Advice\nCarrier: MAERSK\nVessel: MT SEA QUEEN\n"
                  "Port of loading: Kumai\nNo policy reference."),
        structured="references_found: no policy/cover-note entries",
        expected="FAIL",
    ),

    # ── 31. Documentary Remittance textual assertion "presented on time" ──
    dict(
        label="Remittance asserts 'documents presented within LC validity' → PASS",
        doc_type="Documentary Remittance",
        cond="Documents must be presented before LC expiry (2026-06-01 AT PAKISTAN).",
        doc_text=("Documentary Remittance / Covering Schedule\n"
                  "L/C No: 0007LC55189/2025\n"
                  "We hereby confirm that documents have been presented within "
                  "the LC validity period. Amount: USD 250,000."),
        structured="stamps=[]; no received date parseable",
        expected="PASS",
    ),
]


def run():
    print("=" * 78)
    print(f"Broad LLM dry-run: {len(scenarios)} scenarios against {LLM_MODEL}")
    print("=" * 78)
    correct = 0
    mismatches = []
    errors = []
    per_expected = {"PASS": [0, 0], "FAIL": [0, 0], "REVIEW": [0, 0]}  # [correct, total]
    total_time = 0.0
    for i, s in enumerate(scenarios, 1):
        dt = s["doc_type"].lower()
        if 'bill of lading' in dt:
            family = BL_FAMILY_PACK
        elif 'draft' in dt or 'bill of exchange' in dt:
            family = DRAFT_FAMILY_PACK
        elif 'shipment' in dt or 'shipping' in dt:
            family = SHIPMENT_ADVICE_FAMILY_PACK
        else:
            family = ""
        prompt = PROMPT_TEMPLATE.format(
            cond=s["cond"],
            doc_type=s["doc_type"],
            doc_text=s["doc_text"][:3500],
            structured=s["structured"],
            family_pack=family,
        )
        r = ask_llm(prompt)
        verdict = (r.get("verdict") or "").upper()
        findings = (r.get("findings") or "")[:140]
        elapsed = r.get("elapsed", 0)
        total_time += elapsed
        expected = s["expected"]
        per_expected[expected][1] += 1
        status = "ok"
        if verdict == "ERROR":
            errors.append((i, s["label"], findings))
            status = "ERR"
        elif verdict == expected:
            correct += 1
            per_expected[expected][0] += 1
            status = "ok"
        else:
            mismatches.append((i, s["label"], expected, verdict, findings))
            status = "miss"
        mark = {'ok': '+', 'miss': '-', 'ERR': '!'}[status]
        print(f"[{i:02d}] [{mark}] expected={expected:6}  got={verdict:6}  "
              f"({elapsed:.1f}s)  {s['label'][:60]}")
        if status != 'ok':
            print(f"       -> {findings}")
    print()
    print("=" * 78)
    print(f"SUMMARY: {correct}/{len(scenarios)} correct  "
          f"({100*correct/len(scenarios):.0f}%)  total_time={total_time:.1f}s  "
          f"avg={total_time/max(len(scenarios),1):.1f}s/call")
    for k in ("PASS", "FAIL", "REVIEW"):
        c, t = per_expected[k]
        if t:
            print(f"  expected {k}: {c}/{t}")
    print("=" * 78)
    if mismatches:
        print("\nMismatches:")
        for i, lbl, exp, got, fnd in mismatches:
            print(f"  [{i:02d}] {lbl}: expected {exp}, LLM said {got}")
            print(f"        findings: {fnd}")
    if errors:
        print("\nErrors:")
        for i, lbl, fnd in errors:
            print(f"  [{i:02d}] {lbl}: {fnd}")


if __name__ == "__main__":
    run()
