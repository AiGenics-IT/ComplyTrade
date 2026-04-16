"""
Step 12 — Clause-by-Clause Condition Decomposition
=====================================================
Takes each LC clause from Step 7's StructuredLC and sends it to the Qwen VLM
to decompose into individual checkable conditions.

PURPOSE:
    An LC clause like "FULL SET OF CLEAN ON BOARD BILLS OF LADING MADE OUT TO
    ORDER OF ISSUING BANK MARKED FREIGHT PREPAID NOTIFY APPLICANT" contains
    MULTIPLE independent conditions:
      1. Full set (3/3 originals) of Bills of Lading
      2. Clean (no adverse clauses on the BL)
      3. On Board (goods loaded, not just received for shipment)
      4. Made out to order of Issuing Bank (consignee field)
      5. Marked Freight Prepaid
      6. Notify party = Applicant

    This step breaks each clause into its individual conditions so that each
    can be independently verified against the actual shipping documents.

    If one condition applies to multiple documents (e.g., "HS CODE MUST APPEAR
    ON BL AND INVOICES"), separate rows are created for each document.

INPUTS:
    - Step 7 StructuredLC (consolidated_fields with clause data)
    - Clause fields processed: 46A, 46B, 47A, 47B, 45A, 45B, 46C, 77A, 78, 79, 72

OUTPUTS:
    - List of DecomposedClause objects, each containing:
      - clause_ref (e.g., "46A-1", "47A-3")
      - original clause text
      - List of Condition objects with: condition_text, document_to_check, look_for_value

TRADE FINANCE CONTEXT:
    - F46A = Documents Required — lists all shipping documents the beneficiary must present
    - F47A = Additional Conditions — extra requirements that may override or supplement F46A
    - F45A = Description of Goods — must match across Invoice, BL, and other documents
    - F78 = Instructions to Paying/Accepting/Negotiating Bank
    - Some fields (31D, 44C, 48, 32B) are checked by code logic, not VLM (implicit checks)

AI MODEL: Qwen VLM at QWEN_VLM_URL (resolved from config/settings.py
          via the VLM_MODEL_SIZE switch — 7B or 72B). Used for decomposing
          clause text into individual conditions. Implicit checks (dates,
          amounts) bypass the VLM entirely.
"""

import json
import sys as _sys; _sys.stdout.reconfigure(encoding="utf-8", errors="replace") if hasattr(_sys.stdout, "reconfigure") else None
import time
import re
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import QWEN_TEXT_LLM_URL, QWEN_TEXT_LLM_MODEL, MAX_CONCURRENT_VLM, VLM_TIMEOUT


# ── Data Models ──────────────────────────────────────────────────────────

@dataclass
class Condition:
    """
    A single checkable condition extracted from an LC clause.

    Each condition is independently verifiable against ONE specific document.
    For example, "Invoice must show HS Code" can be checked by looking at the
    Commercial Invoice text for an HS Code pattern.
    """
    condition_id: str = ""
    condition_text: str = ""             # Human-readable description of what to verify
    document_to_check: str = ""          # Target document type (e.g., "Bill of Lading")
    look_for_value: str = ""             # Specific value or pattern to search for
    is_implicit: bool = False            # True for date/amount checks done by Python code, not VLM
    implicit_type: str = ""              # lc_expiry, late_shipment, late_presentation, amount_overdrawn
    confidence: float = 1.0


@dataclass
class DecomposedClause:
    """
    A single LC clause broken into individual checkable conditions.

    For example, F46A clause #1 might decompose into 5 separate conditions,
    each targeting a specific document and looking for specific evidence.
    """
    clause_ref: str = ""             # e.g. "46A-1", "47A-3" — unique identifier
    field_tag: str = ""              # e.g. "46A", "47A", "45A" — SWIFT field tag
    clause_number: int = 0           # Position within the field (1-based)
    original_text: str = ""          # Full original clause text from the LC
    conditions: List[Condition] = field(default_factory=list)
    elapsed_seconds: float = 0.0     # Time taken for VLM decomposition


# ── VLM Prompt ───────────────────────────────────────────────────────────
# This system prompt instructs the Qwen VLM on how to decompose LC clauses.
# It includes trade finance domain knowledge to improve accuracy.

DECOMPOSITION_SYSTEM_PROMPT = """You are a senior trade finance examiner specializing in Letter of Credit (LC) compliance under UCP 600 and ISBP 821.

Your task: decompose an LC clause into individual, checkable conditions that a document checker can verify against the presented shipping documents.

═══════════════════════════════════════════════════════════════
CRITICAL: UNDERSTAND THE CLAUSE BEFORE DECOMPOSING
═══════════════════════════════════════════════════════════════

READ the clause carefully and understand its MEANING in trade finance context BEFORE creating conditions. Many clauses use specific trade finance language that must be interpreted correctly:

PERMISSIVE vs PROHIBITIVE CLAUSES:
  • "THIRD PARTY DOCUMENTS ARE ACCEPTABLE EXCEPT INVOICE AND DRAFT"
    → This ALLOWS third-party docs for everything EXCEPT invoice/draft.
    → Invoice and Draft MUST be issued by the beneficiary (not third party).
    → Do NOT create "Invoice is not acceptable" or "Draft is not acceptable" — that is WRONG.
    → Correct: One condition noting invoice/draft must be beneficiary-issued (not third party).

  • "LARGER QUANTITIES ACCEPTABLE" / "CHARTER PARTY BL ACCEPTABLE"
    → These ALLOW something — they are permissive relaxations, not requirements.
    → Return EMPTY ARRAY [] — nothing to check.

  • "SHORT FORM / BLANK BACK / HOUSE / STALE / FORWARDER AGENT BL NOT ACCEPTABLE"
    → This PROHIBITS certain BL types. Create conditions that BL must NOT be these types.

  • "GOODS DESCRIPTION SHOWING [X] TO BE ACCEPTABLE"
    → This means showing [X] (e.g., "CANOLA" instead of full "CANADIAN CANOLA SEED NO.1") is acceptable on documents.
    → This is permissive — it relaxes goods description matching. Return EMPTY ARRAY [].

  • "ALL DOCUMENTS EXCEPT BL SHOWING APPLICANT AS NOTIFY TO BE ACCEPTABLE OR AS CONSIGNEE ACCEPTABLE"
    → This says non-BL documents MAY show applicant as notify OR consignee — both are acceptable.
    → This is permissive — it does NOT require anything. Return EMPTY ARRAY [].
    → ANY clause that says "[something] TO BE ACCEPTABLE" or "[something] ACCEPTABLE" at the end is permissive.

TOLERANCE CLAUSES:
  • "PLUS ZERO/MINUS TEN PERCENT TOLERANCE IN QUANTITY AND AMOUNT TO BE ACCEPTABLE"
    → This sets tolerance parameters (+0%/-10%) — it does NOT require checking on documents.
    → Return EMPTY ARRAY [] — tolerance is applied by code in amount/quantity checks.
  • "1000 MT LESS 10 PCT" or "1000 MT PLUS/MINUS 5 PCT"
    → Tolerance in goods description — handled by code, not document checking.

═══════════════════════════════════════════════════════════════
RETURN EMPTY ARRAY [] FOR THESE CLAUSE TYPES:
═══════════════════════════════════════════════════════════════
  • Bank-to-bank obligations: "NEGOTIATING BANK MUST...", "ADVISING BANK MUST..."
  • Fee/charge clauses: "DISCREPANCY HANDLING CHARGES", "BANK CHARGES WILL BE DEDUCTED"
  • Document forwarding: "ALL DOCUMENTS TO BE FORWARDED TO US BY COURIER"
  • Regulatory/compliance statements: "TRANSPORTATION SHOULD BE EFFECTED BY COMPANIES OPERATING IN ACCORDANCE WITH...LAWS AND REGULATIONS", "SANCTIONED BY...", "BOUND BY SANCTIONS" (these cannot be verified from documents)
  • Physical packing/shipping instructions: "COPY SHOULD BE PLACED IN CARTON/DRUM", "INTIMATE DETAILS TO CONSIGNEE ON E-MAIL", "DOCUMENTS TO BE SENT BY COURIER" (these are physical actions, not document requirements — cannot be verified from document content)

ACCOMPANIMENT CLAUSES:
  • "COPY OF X MUST ACCOMPANY ORIGINAL DOCUMENTS" means a copy of document X must be present in the submission. The document_to_check should be the document itself (e.g., "Shipment Advice"), NOT "Original Documents". Check if that document exists in the submission.
  • Do NOT use "Original Documents" as a document_to_check — that is not a document type.
  • Pure permissive clauses: "LARGER QUANTITIES ACCEPTABLE", "CHARTER PARTY BL ACCEPTABLE", "GOODS DESCRIPTION SHOWING [X] TO BE ACCEPTABLE"
  • Tolerance specifications: "PLUS/MINUS X PERCENT TOLERANCE" (handled by code)
  • Date validity: "DOCUMENTS DATED PRIOR TO... NOT ACCEPTABLE" (handled separately)

═══════════════════════════════════════════════════════════════
TRANSMISSION CLAUSES — ONE SET / VIA EMAIL / VIA COURIER (CRITICAL)
═══════════════════════════════════════════════════════════════
A "transmission clause" describes HOW a bundle of documents was
delivered (typically a courier copy or an email copy sent to the
applicant or to a third party). It is NOT a list of separate
documents to verify. The whole clause maps to ONE compliance row
that asks whether the transmission evidence is present in the
submission set — usually a covering email screenshot, a courier
airway bill, or a remittance schedule that lists what was sent.

Recognise these wordings as transmission clauses:
  • "ONE SET OF NON-NEGOTIABLE DOCUMENTS COMPRISING OF ... TO BE
     SENT TO THE APPLICANT VIA EMAIL"
  • "A COPY OF THE FOLLOWING DOCUMENTS ... TO BE DISPATCHED TO
     THE APPLICANT BY EMAIL / COURIER / FAX"
  • "PHOTOCOPIES OF ... TO BE FORWARDED DIRECTLY TO THE APPLICANT"
  • "ONE SET OF SHIPPING DOCUMENTS TO BE SENT BY COURIER TO ..."
  • "DOCUMENTS LISTED ABOVE TO BE EMAILED TO ..."

WHEN YOU SEE A TRANSMISSION CLAUSE:
  → Emit EXACTLY ONE condition for the whole clause.
  → document_to_check = "Documentary Remittance"
    (the covering email / fax / courier evidence is grouped under
    Documentary Remittance — see the synonym list below)
  → condition_text = a single human-readable summary of what must
    be evidenced, e.g.
       "One set of non-negotiable documents (signed Invoice,
        Packing List, Form 7, Form 3, Certificate of Analysis,
        Air Waybill copy) must have been sent to the applicant
        via email — verify the covering email / remittance
        schedule shows the bundle was dispatched."
  → Do NOT create separate rows for each document listed inside
    the bundle. Those documents are ALREADY checked by their own
    F46A clauses elsewhere in this LC. Re-checking them here
    duplicates work and produces false fails when the verifier
    looks for an email header on each individual PDF.
  → Do NOT create rows with document_to_check = "All Documents"
    or with the individual document names ("Commercial Invoice",
    "Packing List", "Form 7", etc.). One row, one document type,
    one piece of evidence — that is the rule.

WORKED COUNTER-EXAMPLE 1 — "ONE SET ... VIA EMAIL":
   Clause: "ONE SET OF NON-NEGOTIABLE DOCUMENTS COMPRISING OF
            MANUALLY SIGNED INVOICE, PACKING LIST, COPIES OF
            FORM 7 (BATCH CERTIFICATE), FORM 3, CERTIFICATE OF
            ANALYSIS, COPY OF AWB, TO BE SENT TO THE APPLICANT
            VIA EMAIL."

   WRONG decomposition (do NOT do this):
     ❌ 7 separate rows, each asking the verifier to find email
        evidence on the corresponding document. Result: 7 false
        FAILs because individual documents do not carry email
        headers.

   CORRECT decomposition (one row only):
     1. document_to_check = "Documentary Remittance"
        condition_text   = "One set of non-negotiable documents
            (signed Commercial Invoice, Packing List, Form 7
            Batch Certificate, Form 3, Certificate of Analysis,
            and a copy of the Air Waybill) must have been sent
            to the applicant via email. Verify a covering email,
            remittance schedule, or transmission record exists
            in the submission showing this bundle was dispatched
            to the applicant."
        is_implicit      = false

═══════════════════════════════════════════════════════════════
DISPATCH + BENEFICIARY-CERTIFICATE-TO-THIS-EFFECT CLAUSES
═══════════════════════════════════════════════════════════════
A second very common pattern is a clause that says:
  • Some original document (FTA Certificate, COO, Quality Cert,
    Insurance Policy, etc.) must be dispatched / couriered /
    sent / handed over to a named party (the applicant or a
    third party), AND
  • A "BENEFICIARY CERTIFICATE TO THIS EFFECT" must accompany
    the presented documents stating that the dispatch happened.

WHAT TO GENERATE (read all rules before decomposing):

  RULE D-1 — Document conditions:
    Decompose ALL conditions that relate to the named original
    document (FTA Certificate, COO, Insurance Policy, etc.).
    Each becomes a separate row with document_to_check = the
    named document type. This includes:
      • document must be present in the submission
      • document must be dispatched to <recipient>
      • document must be sent to <postal address>
      • document must be sent along with non-negotiable documents
    Even though some of these conditions CANNOT be verified when
    the document is missing (they will appear as N/A in the
    report), they must still be generated so the checker sees the
    COMPLETE list of requirements for that document.

  RULE D-2 — Beneficiary-certificate accompaniment row:
    If the clause says "BENEFICIARY CERTIFICATE TO THIS EFFECT TO
    ACCOMPANY ORIGINAL DOCUMENTS", create a separate row:
    → document_to_check = "Beneficiary Certificate"
    → condition_text = "Beneficiary Certificate must accompany
      the original documents."
    This is a physical presence check confirming the certificate
    is in the submission set.

  RULE D-4 — Any other genuinely separate conditions:
    If the clause contains OTHER requirements beyond the dispatch
    and beneficiary certificate (e.g. a time limit like "WITHIN 7
    DAYS FROM SHIPMENT DATE", or "A COPY OF FTA MUST BE ATTACHED
    TO COMMERCIAL INVOICE"), decompose those too — each becomes
    its own row with the appropriate document_to_check.

  RULE D-3 — What NOT to generate:
    ❌ "Beneficiary Certificate must STATE that the original
       <doc> was dispatched to <recipient>..." — do NOT create
       a content-check row asking the verifier to find a specific
       dispatch statement inside the Beneficiary Certificate.
       The Beneficiary Certificate just needs to EXIST and
       ACCOMPANY the original documents (Rule D-2). The wording
       "TO THIS EFFECT" in the clause means the certificate's
       existence IS the evidence — it does not mean the verifier
       must parse the certificate for a specific sentence about
       the dispatch. Creating such a row produces a false FAIL
       because the certificate's exact wording never matches the
       decomposer's invented condition text.
    ❌ Multiple rows on the Beneficiary Certificate splitting
       the dispatch details into sub-phrases.
    ❌ Duplicate conditions that say the same thing in different
       words (e.g. "FTA must accompany original documents" AND
       "FTA must be sent with non-negotiable documents" — these
       are the same requirement, keep only one).

WORKED COUNTER-EXAMPLE 2 — FTA dispatch + Beneficiary Certificate:
   Clause: "FTA CERTIFICATE REQUIRED. ORIGINAL FTA SHOULD BE
            DISPATCHED IMMEDIATELY TO GETZ PHARMA (PVT) LTD.
            POSTAL ADDRESS ALONG WITH NON-NEGOTIABLE DOCUMENTS.
            BENEFICIARY CERTIFICATE TO THIS EFFECT TO ACCOMPANY
            ORIGINAL DOCUMENTS. DAYS FROM SHIPMENT DATE BUT
            WITHIN THE VALIDITY OF LC."

   WRONG decomposition (do NOT do this):
     ❌ 7 rows like
        - "Original FTA Certificate must be dispatched immediately
           to Getz Pharma"  (FTA Certificate)
        - "FTA Certificate must be sent to the postal address of
           Getz Pharma"  (FTA Certificate)
        - "FTA Certificate must be sent along with non-negotiable
           documents"  (FTA Certificate)
        - "Beneficiary Certificate must state that the original
           FTA Certificate was dispatched immediately to Getz
           Pharma"  (Beneficiary Certificate)
        - "Beneficiary Certificate must state that the FTA
           Certificate was sent to the postal address of Getz
           Pharma"  (Beneficiary Certificate)
        - "Beneficiary Certificate must state that the FTA
           Certificate was sent along with non-negotiable
           documents"  (Beneficiary Certificate)
        - "Beneficiary Certificate must accompany the original
           documents"  (Beneficiary Certificate)
        These produce six false fails because the FTA Certificate
        cannot self-report being dispatched, and the Beneficiary
        Certificate carries one combined sentence, not three
        separate statements.

   CORRECT decomposition (applying rules D-1 and D-2):
     1. (Rule D-1) document_to_check = "FTA Certificate"
        condition_text   = "An FTA Certificate must be present
            in the submission."

     2. (Rule D-1) document_to_check = "FTA Certificate"
        condition_text   = "Original FTA Certificate must be
            dispatched immediately to Getz Pharma (Pvt) Ltd."

     3. (Rule D-1) document_to_check = "FTA Certificate"
        condition_text   = "FTA Certificate must be sent to the
            postal address of Getz Pharma (Pvt) Ltd."

     4. (Rule D-1) document_to_check = "FTA Certificate"
        condition_text   = "FTA Certificate must be sent along
            with non-negotiable documents."

     5. (Rule D-2) document_to_check = "Beneficiary Certificate"
        condition_text   = "Beneficiary Certificate must
            accompany the original documents."

═══════════════════════════════════════════════════════════════
TRADE FINANCE KNOWLEDGE
═══════════════════════════════════════════════════════════════
  • "FULL SET" = 3/3 originals (e.g., "FULL SET OF BILLS OF LADING" = 3 original BLs)
  • "ISSUING BANK" = the actual bank that issued the LC — use the bank name if provided
  • "IN DUPLICATE" = 2 copies, "IN TRIPLICATE" = 3 copies
  • Bill of Lading does NOT show dollar amounts — never create "BL must show amount" conditions
  • Bill of Lading does NOT typically show LC/credit number unless F47A specifically requires it
  • "INSURANCE COVERED BY APPLICANT" = applicant handles insurance, beneficiary must send SHIPMENT ADVICE (not insurance doc)
  • "ENDORSED TO THE ORDER OF [BANK]" on BL = consignee field shows "TO THE ORDER OF [BANK]"

═══════════════════════════════════════════════════════════════
INSURANCE — HARD PROHIBITION (READ THIS BEFORE EVERY F47A CLAUSE)
═══════════════════════════════════════════════════════════════
The mere appearance of the word "INSURANCE" in a clause does NOT mean
the document_to_check is "Insurance Policy/Certificate". Most F47A
mentions of insurance describe HOW INSURANCE IS HANDLED, not what
document the beneficiary must present. You MUST follow these rules:

RULE I-1 (the most important):
If a clause contains ANY of the following phrases — anywhere in the
clause text — the beneficiary will NOT submit any insurance document.
You MUST NEVER create a condition with document_to_check =
"Insurance Policy" or "Insurance Certificate" or "Insurance Policy/
Certificate" for ANY part of such a clause:
  • "INSURANCE COVERED BY APPLICANT"
  • "INSURANCE TO BE COVERED BY APPLICANT"
  • "INSURANCE TO BE EFFECTED BY APPLICANT"
  • "INSURANCE ARRANGED BY APPLICANT"
  • "INSURANCE ON BUYER'S ACCOUNT"
  • "INSURANCE NOT REQUIRED"
  • "INSURANCE COVERED BY BUYER"
  • "INSURANCE BY APPLICANT"

RULE I-2 (Incoterms):
For LCs whose Incoterms are CFR / CNF / FOB / EXW / FCA / FAS, the
beneficiary does NOT supply insurance. Only CIF / CIP terms require
insurance from the beneficiary. If you can see the LC Incoterms in the
context and they are non-insurance Incoterms, you MUST NOT create
"Insurance Policy/Certificate" conditions for ANY clause in this LC.

RULE I-3 (real intent of insurance-mentioning clauses):
When a clause contains "INSURANCE COVERED BY APPLICANT" together with
instructions to ADVISE / NOTIFY / SEND DETAILS to the insurer or
applicant within N days of shipment, the SUBJECT of the clause is the
SHIPMENT ADVICE the beneficiary must produce — NOT an insurance
document. Use document_to_check = "Shipment Advice" for CONTENT
conditions (address, policy number, shipment details, timing).

RULE I-3b: PROOF OF DISPATCH (CRITICAL)
When the clause specifies HOW the shipment advice must be sent:
  • "BY COURIER" → create a separate row with
    document_to_check = "Courier Receipt" to verify that a courier
    receipt/airway bill exists proving the dispatch.
  • "VIA EMAIL" / "BY EMAIL" → create a separate row with
    document_to_check = "Email Evidence" to verify email screenshot
    or transmission record exists.
  • "BY FAX" → create a separate row with
    document_to_check = "Fax Confirmation" to verify fax transmission
    report exists.
The Shipment Advice itself does NOT prove how it was sent — only
the courier receipt / email screenshot / fax report proves that.
Always create BOTH:
  - Content checks against "Shipment Advice" (what does it say?)
  - Dispatch proof against "Courier Receipt" / "Email Evidence" (was it sent?)

RULE I-4 ("ALL DOCUMENTS" fan-out):
When a clause says "L/C NUMBER MUST APPEAR ON ALL DOCUMENTS" (or any
"ALL DOCUMENTS" fan-out), use document_to_check = "All Documents"
(literal string). Do NOT enumerate doc types yourself. The verifier
will fan out at run time across documents that are ACTUALLY present in
this submission. NEVER include "Insurance Policy/Certificate" in such
a fan-out unless RULE I-1 and RULE I-2 both confirm that insurance is
required from the beneficiary in this LC.

WORKED EXAMPLE A — "BY COURIER" (STUDY THIS):
Clause: "INSURANCE COVERED BY THE APPLICANT. ALL SHIPMENT UNDER
THIS CREDIT MUST BE ADVISED BY THE BENEFICIARY WITHIN FOUR
WORKING DAYS AFTER SHIPMENT DIRECT TO UBL INSURERS LIMITED,
OFFICE NO.501, 5TH FLOOR, SIDIQUE TRADE CENTRE, MAIN BOULEVARD
GULBERG-II, LAHORE PAKISTAN AND TO THE APPLICANT
BY COURIER REFERRING TO THEIR OPEN POLICY NO.
2023008MIPDO00453
MENTIONING THE DETAIL OF SHIPMENT AND VALUE OF INVOICE
COPIES OF SUCH SHIPMENT ADVICES
MUST ACCOMPANY THE ORIGINAL DOCUMENTS."

CORRECT decomposition:
  1. document_to_check="Shipment Advice", condition="Shipment Advice
     must be dated within 4 working days after shipment date"
  2. document_to_check="Shipment Advice", condition="Shipment Advice
     must be addressed to UBL Insurers Limited, Office No.501, 5th
     Floor, Sidique Trade Centre, Main Boulevard Gulberg-II, Lahore
     Pakistan"
  3. document_to_check="Shipment Advice", condition="Shipment Advice
     must also be addressed to the Applicant"
  4. document_to_check="Shipment Advice", condition="Shipment Advice
     must reference Open Policy No. 2023008MIPDO00453"
  5. document_to_check="Shipment Advice", condition="Shipment Advice
     must mention details of shipment (vessel, port, BL date)"
  6. document_to_check="Shipment Advice", condition="Shipment Advice
     must mention value of invoice"
  7. document_to_check="Courier Receipt", condition="Courier receipt
     must exist proving shipment advice was dispatched by courier
     to UBL Insurers Limited and to the Applicant"
  8. document_to_check="Shipment Advice", condition="Copies of the
     Shipment Advice must accompany the original documents"

WORKED EXAMPLE B — "VIA EMAIL":
Clause: "INSURANCE COVERED BY APPLICANT. ALL SHIPMENT(S) UNDER THIS
CREDIT MUST BE ADVISED BY THE BENEFICIARY WITHIN THREE WORKING DAYS
AFTER SHIPMENT TO M/S. SINDH INSURANCE, KARACHI, PAKISTAN AND TO THE
APPLICANT, REFERRING TO THEIR COVER NOTE NO. SIL/D/T001/0000001192/
0925/001 DATED 04-09-2025 VIA EMAIL INFO(AT)SIUT.ORG GIVING FULL
DETAILS OF SHIPMENT, SUCH AS L/C NO., INVOICE VALUE, PORT OF SHIPMENT,
VESSEL NAME, SHIPPED ON BOARD DATE. A COPY OF THIS SHIPMENT ADVICE TO
ACCOMPANY EACH SET OF DOCUMENT(S)"

CORRECT decomposition:
  1. document_to_check="Shipment Advice", condition="Beneficiary must
     send a Shipment Advice within 3 working days after shipment"
  2. document_to_check="Shipment Advice", condition="Shipment Advice
     must be addressed to M/s. Sindh Insurance, Karachi, Pakistan AND
     to the Applicant"
  3. document_to_check="Shipment Advice", condition="Shipment Advice
     must reference Cover Note No. SIL/D/T001/0000001192/0925/001
     dated 04-09-2025"
  4. document_to_check="Email Evidence", condition="Email evidence must
     exist showing shipment advice was sent via email to
     info(at)siut.org"
  5. document_to_check="Shipment Advice", condition="Shipment Advice
     must show L/C number"
  6. document_to_check="Shipment Advice", condition="Shipment Advice
     must show invoice value"
  7. document_to_check="Shipment Advice", condition="Shipment Advice
     must show port of shipment"
  8. document_to_check="Shipment Advice", condition="Shipment Advice
     must show vessel name"
  9. document_to_check="Shipment Advice", condition="Shipment Advice
     must show shipped on board date"
 10. document_to_check="Shipment Advice", condition="A copy of the
     Shipment Advice must accompany each set of presented documents"

INCORRECT decomposition (DO NOT DO THIS):
  ❌ document_to_check="Insurance Policy/Certificate", condition="..."
  ❌ Any row asking the verifier to find anything on an Insurance
     Policy or Insurance Certificate.
  ❌ Checking "Shipment Advice" for courier/email dispatch proof —
     the Shipment Advice does NOT prove how it was sent.
The reason is simple: there IS no insurance document in this
presentation — the applicant arranged the cover separately. Creating
an Insurance row guarantees a false "REQUIRED DOCUMENT MISSING"
discrepancy on every run.

DOCUMENT IDENTIFICATION:
  • "SHIPMENT ADVICE" or "BENEFICIARY SHIPMENT ADVICE" = document type "Shipment Advice"
  • "CERTIFICATE FROM SHIPPING COMPANY OR THEIR AUTHORIZED AGENTS" = "Shipping Company Certificate"
  • "INSURANCE POLICY" or "INSURANCE CERTIFICATE" = "Insurance Policy/Certificate"
  • SYNONYMS — these all mean the SAME document; always use "Shipping
    Company Certificate" as document_to_check, never "Agent's Certificate":
      - "Shipping Company Certificate"
      - "Shipping Certificate"
      - "Agent's Certificate" / "Agents Certificate" / "Agent Certificate"
      - "Shipping Agent's Certificate" / "Shipping Agent Certificate"
      - "Carrier's Certificate" / "Carrier Certificate"
      - "Certificate from Shipping Company"
      - "Certificate from Shipping Company or Their Authorized Agents"
  • SYNONYMS — these all mean the SAME document; always use "Documentary
    Remittance" as document_to_check:
      - "Documentary Remittance"
      - "Covering Letter" / "Covering Schedule" / "Cover Schedule"
      - "Export DC Document Presentation Schedule"
      - "Export DC Presentation Schedule"
      - "Document Presentation Schedule" / "Presentation Schedule"
      - "Schedule of Documents" / "Letter of Transmittal"

SHIPMENT ADVICE DECOMPOSITION — CRITICAL:
When a clause describes a SHIPMENT ADVICE requirement (typically
under "INSURANCE COVERED BY APPLICANT"), decompose it into SEPARATE
conditions for each distinct requirement:

  1. TIMELINESS: "within N working days after shipment" →
     document_to_check = "Shipment Advice"
     condition = "Beneficiary must send a Shipment Advice within
                  N working days after shipment."

  2. EACH ADDRESSEE SEPARATELY: If the clause says "TO [Party A]
     AND TO [Party B]", create TWO separate conditions:
     a) document_to_check = "Shipment Advice"
        condition = "Shipment Advice must be addressed to
                     [Party A full name and address]."
     b) document_to_check = "Shipment Advice"
        condition = "Shipment Advice must also be addressed to
                     the Applicant [ACTUAL APPLICANT NAME from F50]."
     IMPORTANT: When the clause says "AND TO THE APPLICANT", you
     MUST replace "THE APPLICANT" with the actual applicant name
     from the LC context (F50). For example if F50 is "GETZ PHARMA
     (PVT) LTD", the condition should say "must also be addressed
     to GETZ PHARMA (PVT) LTD" — NOT just "to the Applicant".
     This allows the verifier to check the actual name on the
     Shipment Advice.
     Do NOT combine them into one row — the VLM will check the
     first addressee and skip the second.

  3. POLICY/REFERENCE: "referring to their Policy No. XXX" →
     document_to_check = "Shipment Advice"
     condition = "Shipment Advice must reference Policy No. XXX"

  4. EACH REQUIRED DETAIL: "giving full details such as L/C No.,
     Invoice Value, Port of Shipment, Vessel Name, Shipped On Board
     Date, Expected Arrival Date" → create ONE condition listing
     ALL required details that must appear on the Shipment Advice.

  5. ACCOMPANIMENT: "a copy must accompany each set of documents" →
     document_to_check = "Shipment Advice"
     condition = "A copy of the Shipment Advice must accompany
                  each set of presented documents."

  6. EMAIL/FAX: If the clause specifies an email address or fax for
     the notification, create a condition checking the email address.

CLAUSE STRUCTURE — READ CAREFULLY:
  • "INSURANCE COVERED BY APPLICANT, BENEFICIARY SHIPMENT ADVICE QUOTING [items] SHOULD BE SENT TO [address]"
    → This clause is about the SHIPMENT ADVICE, not insurance or BL.
    → "QUOTING NAME OF VESSEL, B/L NO., DATE OF SHIPMENT, AMOUNT, QUANTITY AND CREDIT NUMBER" — ALL these items must appear on the SHIPMENT ADVICE (not the BL, not the invoice).
    → Create conditions checking the SHIPMENT ADVICE for: vessel name, BL number, shipment date, amount, quantity, credit number.
    → Also create conditions for: sent to correct addressee, within time limit, copy accompanies original docs.
    → Do NOT create conditions checking the Bill of Lading for amount or credit number from this clause.

═══════════════════════════════════════════════════════════════
RULES FOR CREATING CONDITIONS
═══════════════════════════════════════════════════════════════
1. Each condition must be independently verifiable against a SINGLE document
2. If a condition applies to MULTIPLE documents, create SEPARATE conditions for each document
   CRITICAL: "X SHOULD APPEAR ON BILL OF LADING AND INVOICES" = TWO conditions:
   - One checking X on Bill of Lading
   - One checking X on Commercial Invoice
   NEVER skip a document. If the clause says "AND", create conditions for EVERY document listed.

═══════════════════════════════════════════════════════════════
DISTRIBUTIVE CONJUNCTION RULE — READ THIS BEFORE GENERATING CONDITIONS
═══════════════════════════════════════════════════════════════
When a clause has the form:
    "[Subject1] AND [Subject2] should appear on [Target1] AND [Target2]"
you MUST generate ALL CROSS-PRODUCT combinations — Subjects × Targets.
Do NOT generate fewer just because some combinations seem unusual in real-world
practice (e.g. "H.S. codes don't normally go on a BL"). The CLAUSE is the
source of truth, not your real-world prior knowledge.

WORKED EXAMPLE — STUDY THIS:

Clause: "L/C NUMBER SHOULD APPEAR ON INVOICES AND DRAFT AND H.S. CODE NO.
         1205.1000 AND IMPORTER'S NTN NO. 2128133-5 SHOULD APPEAR ON
         BILL OF LADING AND INVOICES."

This is TWO sub-sentences joined by AND:
  Sub-sentence 1:  L/C number → [Invoices, Draft]               (1 subject × 2 targets = 2 conditions)
  Sub-sentence 2:  [H.S. Code, NTN] → [Bill of Lading, Invoices] (2 subjects × 2 targets = 4 conditions)

You MUST produce 2 + 4 = 6 conditions:
  1. L/C number             → Commercial Invoice
  2. L/C number             → Draft
  3. H.S. Code 1205.1000    → Bill of Lading      ← do NOT skip this even though
                                                     H.S. codes are unusual on a BL.
                                                     The clause requires it.
  4. H.S. Code 1205.1000    → Commercial Invoice
  5. NTN 2128133-5          → Bill of Lading
  6. NTN 2128133-5          → Commercial Invoice

ANOTHER EXAMPLE:
Clause: "Beneficiary name and address must appear on the BL, Invoice and Packing List."
  Subjects: [Beneficiary name, Beneficiary address]
  Targets:  [Bill of Lading, Commercial Invoice, Packing List]
You MUST produce 2 × 3 = 6 conditions, one per (subject, target) pair.

SELF-CHECK BEFORE RETURNING:
- For each sub-sentence, count len(subjects) × len(targets) and call this K.
- The total number of conditions you emit MUST equal the sum of K across
  all sub-sentences.
- If your count doesn't match, you missed a combination — go back and add it.
- NEVER return fewer conditions than the math demands.

═══════════════════════════════════════════════════════════════
ROUTING RULES — WHICH DOCUMENT TO CHECK
═══════════════════════════════════════════════════════════════
The "document_to_check" must be the document where the EVIDENCE for the
check lives — NOT necessarily the document the clause is "about".

  • For "STALE BL NOT ACCEPTABLE" / "BL must not be stale" / "DOCUMENTS
    MUST NOT BE STALE":
    → document_to_check = "Documentary Remittance"
    → Reason: "stale" means presentation_date - shipment_date > F48 days.
      The presentation date lives ONLY on the Documentary Remittance.
      The BL has the shipment date but cannot tell you whether the BL is
      stale on its own. Routing this check to the BL forces the verifier
      to hallucinate a comparison against today's date, which is wrong.
    → condition_text: "Documents must be presented within F48 days of BL
      shipment date — if presentation is later, the BL is stale (per UCP
      600 Art 14(c))"
    → look_for_value: "presentation date on Documentary Remittance"
    → SYNONYMS — these all refer to the SAME document, always use
      "Documentary Remittance" as document_to_check:
        - "Documentary Remittance"
        - "Covering Letter" / "Covering Schedule" / "Cover Schedule"
        - "Export DC Document Presentation Schedule"
        - "Export DC Presentation Schedule"
        - "Document Presentation Schedule"
        - "Schedule of Documents" / "Letter of Transmittal"

  • For "DOCUMENTS PRESENTED LATER THAN N DAYS NOT ACCEPTABLE" /
    "DOCUMENTS PRESENTED AFTER LC EXPIRY":
    → document_to_check = "Documentary Remittance"
    → Same reason: presentation date lives there.
    → Same synonym list as above applies.

  • For "SHIPPED ON/BEFORE [date]" / "LATEST SHIPMENT":
    → document_to_check = "Bill of Lading"
    → Reason: shipment date lives on the BL (CLEAN ON BOARD / SHIPPED
      ON BOARD notation).

  • For "DOCUMENTS DATED ON OR AFTER LC ISSUE DATE":
    → document_to_check = the document being checked (Commercial Invoice,
      BL, etc.) — the document's own issue date is on the document itself.

  • For "[NUMBER/CODE] MUST APPEAR ON [DOCUMENT]":
    → document_to_check = the named target document (one row per target).

GENERAL PRINCIPLE:
"Where does the evidence live?" — that's the document to check, even if
the clause is grammatically about a different document.

═══════════════════════════════════════════════════════════════

3. condition_text: clear, human-readable description of what to verify
4. document_to_check: the exact document type (e.g., "Bill of Lading", "Commercial Invoice")
5. look_for_value: the specific value, text, or pattern to search for

FIELD-SPECIFIC GUIDANCE:
  • F45A (Description of Goods): Check goods description, quantity, unit price, trade terms (CFR/CIF/FOB), port ONLY on Commercial Invoice. Do NOT create conditions for Bill of Lading from F45A — BL has its own requirements in F46A. BL shows weight/packages in its own format, not the LC goods description verbatim.
  • F46A (Documents Required): Extract every document requirement — type, copies, signature, addressee, content. ALWAYS decompose F46A — never return empty array.
  • F47A (Additional Conditions): Understand meaning first. Only create conditions for things that need CHECKING. Do not create conditions for permissions or relaxations.

Respond ONLY with a JSON array. Each element:
{
  "condition_text": "...",
  "document_to_check": "...",
  "look_for_value": "..."
}"""


# User prompt template — fills in the specific clause and LC context
DECOMPOSITION_USER_TEMPLATE = """Decompose this LC clause into individual checkable conditions.

LC Field: {field_tag}
Clause #{clause_number}:
{clause_text}

Additional LC context:
- Applicant: {applicant}
- Beneficiary: {beneficiary}
- Issuing Bank: {issuing_bank}
- Currency/Amount: {currency_amount}

Return a JSON array of conditions."""


# ── Implicit Checks ─────────────────────────────────────────────────────
# These LC fields are verified by deterministic Python code rather than VLM,
# because they involve date arithmetic or amount comparison — tasks where
# code is more reliable than an AI model.

# All 13 LC Key Terms that require code-based verification
# These are checked by deterministic Python logic, not VLM
IMPLICIT_CHECK_FIELDS = {
    '31C': 'date_of_issue',       # 1. All docs must be issued on/after LC date
    '31D': 'lc_expiry',           # 2. Documents presented within LC expiry
    '51D': 'applicant_bank',      # 3. BL endorsed to order of Applicant Bank
    '50':  'applicant_check',     # 4. Notify/Consignee in BL matches Applicant
    '59':  'beneficiary_check',   # 5. Shipper in BL matches Beneficiary
    '32B': 'amount_overdrawn',    # 6. Amount check: invoice vs LC, tolerance, partial
    '41D': 'available_with',      # 7. Negotiating bank in same country as expiry place
    '42C': 'draft_at_sight',      # 8. Draft reflects "At Sight" terms
    '42A': 'drawee_check',        # 9. Drawee in Draft matches LC (issuing bank)
    '43T': 'transshipment',       # 10. Transshipment allowed/not per BL
    '44E': 'port_of_loading',     # 11. Port of Loading in BL matches LC
    '44F': 'port_of_discharge',   # 12. Port of Discharge in BL matches LC
    '44C': 'late_shipment',       # 13. Shipment date on BL vs latest shipment date
    '48':  'late_presentation',   # Presentation period (derived from 31D + 44C)
}


def _build_implicit_conditions(field_tag: str, clause_text: str) -> List[Condition]:
    """
    Build code-verifiable conditions for ALL 13 LC Key Terms.

    These conditions are flagged as is_implicit=True so Step 14 knows to use
    Python logic instead of sending them to the VLM. Each check maps to a
    specific verification rule from trade finance practice.
    """
    implicit_type = IMPLICIT_CHECK_FIELDS.get(field_tag, '')
    conditions = []

    # 1. F31C - Date of Issue
    if implicit_type == 'date_of_issue':
        conditions.append(Condition(
            condition_text="All shipping documents must be issued on or after LC issuance date. Documents dated prior to LC date are discrepancies unless LC allows it.",
            document_to_check="All Documents",
            look_for_value=clause_text.strip(),
            is_implicit=True,
            implicit_type='date_of_issue',
        ))

    # 2. F31D - Date and Place of Expiry
    elif implicit_type == 'lc_expiry':
        conditions.append(Condition(
            condition_text="Documents must be presented within LC expiry date. Check presentation date from Covering Schedule against LC expiry.",
            document_to_check="Documentary Remittance / Covering Letter",
            look_for_value=clause_text.strip(),
            is_implicit=True,
            implicit_type='lc_expiry',
        ))

    # 3. F51D - Applicant Bank
    elif implicit_type == 'applicant_bank':
        conditions.append(Condition(
            condition_text="Bill of Lading must be issued or endorsed to the order of the Applicant Bank. Verify proper endorsement is available.",
            document_to_check="Bill of Lading",
            look_for_value=clause_text.strip(),
            is_implicit=True,
            implicit_type='applicant_bank',
        ))

    # 4. F50 - Applicant
    elif implicit_type == 'applicant_check':
        conditions.append(Condition(
            condition_text="Notify Party or Consignee in Bill of Lading must match the Applicant name as per LC.",
            document_to_check="Bill of Lading",
            look_for_value=clause_text.strip(),
            is_implicit=True,
            implicit_type='applicant_check',
        ))

    # 5. F59 - Beneficiary
    elif implicit_type == 'beneficiary_check':
        conditions.append(Condition(
            condition_text="Shipper/Exporter in Bill of Lading must match the Beneficiary mentioned in LC.",
            document_to_check="Bill of Lading",
            look_for_value=clause_text.strip(),
            is_implicit=True,
            implicit_type='beneficiary_check',
        ))

    # 6. F32B - Currency & Amount
    elif implicit_type == 'amount_overdrawn':
        conditions.append(Condition(
            condition_text="Verify LC amount against Covering Schedule, Commercial Invoice, and Bill of Exchange. Check for overdrawn/short shipment, tolerance, partial shipment conditions.",
            document_to_check="Commercial Invoice",
            look_for_value=clause_text.strip(),
            is_implicit=True,
            implicit_type='amount_overdrawn',
        ))
        conditions.append(Condition(
            condition_text="Draft/Bill of Exchange amount must match Invoice amount.",
            document_to_check="Draft Bill of Exchange",
            look_for_value=clause_text.strip(),
            is_implicit=True,
            implicit_type='draft_vs_invoice',
        ))

    # 7. F41D - Available With / By
    elif implicit_type == 'available_with':
        conditions.append(Condition(
            condition_text="If 'Any Bank' is specified, the negotiating/presenting bank must be in the same country as the place of LC expiry.",
            document_to_check="Documentary Remittance / Covering Letter",
            look_for_value=clause_text.strip(),
            is_implicit=True,
            implicit_type='available_with',
        ))

    # 8. F42C - Draft at Sight
    elif implicit_type == 'draft_at_sight':
        conditions.append(Condition(
            condition_text="Bill of Exchange (Draft) must reflect 'At Sight' terms as per LC.",
            document_to_check="Draft Bill of Exchange",
            look_for_value=clause_text.strip(),
            is_implicit=True,
            implicit_type='draft_at_sight',
        ))

    # 9. F42A - Drawee
    elif implicit_type == 'drawee_check':
        conditions.append(Condition(
            condition_text="Drawee mentioned in Bill of Exchange must match LC requirement (typically the Issuing Bank or Applicant Bank).",
            document_to_check="Draft Bill of Exchange",
            look_for_value=clause_text.strip(),
            is_implicit=True,
            implicit_type='drawee_check',
        ))

    # 10. F43T - Transshipment
    elif implicit_type == 'transshipment':
        conditions.append(Condition(
            condition_text="Verify transshipment condition in LC against Bill of Lading. Check vessel details and LC terms (allowed/not allowed).",
            document_to_check="Bill of Lading",
            look_for_value=clause_text.strip(),
            is_implicit=True,
            implicit_type='transshipment',
        ))

    # 11. F44E - Port of Loading
    elif implicit_type == 'port_of_loading':
        conditions.append(Condition(
            condition_text="Port of Loading in Bill of Lading must match LC requirement. If LC says 'Any Port in [Country]', verify port belongs to that country.",
            document_to_check="Bill of Lading",
            look_for_value=clause_text.strip(),
            is_implicit=True,
            implicit_type='port_of_loading',
        ))

    # 12. F44F - Port of Discharge
    elif implicit_type == 'port_of_discharge':
        conditions.append(Condition(
            condition_text="Port of Discharge in Bill of Lading must match LC requirement.",
            document_to_check="Bill of Lading",
            look_for_value=clause_text.strip(),
            is_implicit=True,
            implicit_type='port_of_discharge',
        ))

    # 13. F44C - Latest Date of Shipment
    elif implicit_type == 'late_shipment':
        conditions.append(Condition(
            condition_text="Shipment date on BL must be on or before the latest shipment date in LC. If exceeded, mark as Late Shipment Discrepancy.",
            document_to_check="Bill of Lading",
            look_for_value=clause_text.strip(),
            is_implicit=True,
            implicit_type='late_shipment',
        ))

    # F48 - Presentation Period (derived check)
    elif implicit_type == 'late_presentation':
        conditions.append(Condition(
            condition_text="Documents must be presented within the stipulated period after shipment date.",
            document_to_check="Documentary Remittance / Covering Letter",
            look_for_value=clause_text.strip(),
            is_implicit=True,
            implicit_type='late_presentation',
        ))
        conditions.append(Condition(
            condition_text="Documents must not be stale at presentation.",
            document_to_check="Documentary Remittance / Covering Letter",
            look_for_value=clause_text.strip(),
            is_implicit=True,
            implicit_type='stale_documents',
        ))

    return conditions


# ── VLM Call ─────────────────────────────────────────────────────────────

def _call_vlm_decompose(clause_ref: str, field_tag: str, clause_number: int,
                         clause_text: str, lc_context: dict) -> dict:
    """
    Send a single clause to the Qwen VLM for decomposition into conditions.

    The VLM receives the clause text along with LC context (applicant, beneficiary,
    bank, amount) so it can resolve references like "ISSUING BANK" to the actual
    bank name, or understand what "THE GOODS" refers to.

    Returns a dict with clause_ref, conditions list, and elapsed time.
    On error, returns the error message and an empty conditions list.
    """
    start = time.time()
    try:
        # Fill in the user prompt template with clause-specific data
        user_msg = DECOMPOSITION_USER_TEMPLATE.format(
            field_tag=field_tag,
            clause_number=clause_number,
            clause_text=clause_text,
            applicant=lc_context.get('applicant', 'N/A'),
            beneficiary=lc_context.get('beneficiary', 'N/A'),
            issuing_bank=lc_context.get('issuing_bank', 'N/A'),
            currency_amount=lc_context.get('currency_amount', 'N/A'),
        )

        payload = {
            "model": QWEN_TEXT_LLM_MODEL,
            "messages": [
                {"role": "system", "content": DECOMPOSITION_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "temperature": 0.0,       # Zero temperature for fully deterministic output
            "max_tokens": 2048,
        }

        resp = requests.post(QWEN_TEXT_LLM_URL, json=payload, timeout=VLM_TIMEOUT)
        elapsed = time.time() - start

        if resp.status_code != 200:
            return {
                'clause_ref': clause_ref,
                'conditions': [],
                'error': f'HTTP {resp.status_code}',
                'elapsed': elapsed,
            }

        result = resp.json()
        content = result['choices'][0]['message']['content'].strip()

        # Extract JSON array from response — VLM may wrap it in markdown code fences
        json_match = re.search(r'\[.*\]', content, re.DOTALL)
        if json_match:
            conditions_raw = json.loads(json_match.group(0))
        else:
            conditions_raw = []

        return {
            'clause_ref': clause_ref,
            'conditions': conditions_raw,
            'elapsed': elapsed,
        }

    except json.JSONDecodeError:
        return {
            'clause_ref': clause_ref,
            'conditions': [],
            'error': 'JSON parse error from VLM',
            'elapsed': time.time() - start,
        }
    except Exception as e:
        return {
            'clause_ref': clause_ref,
            'conditions': [],
            'error': str(e),
            'elapsed': time.time() - start,
        }


# ── Clause Fields ────────────────────────────────────────────────────────
# These SWIFT field tags contain clause lists that need decomposition.
# Other fields (like F20 = LC Number) are standalone values, not clauses.

CLAUSE_FIELDS = {'46A', '46B', '47A', '47B', '45A', '45B', '46C'}
# Note: 78 (Instructions to Bank), 79 (Narrative), 72 (Sender to Receiver), 77A (Regulatory)
# are bank-to-bank instructions — NOT checkable against shipping documents, shown as informational only.


def _extract_lc_context(structured_lc: dict) -> dict:
    """
    Pull key LC context fields for VLM prompt enrichment.

    The VLM needs to know who the applicant, beneficiary, and issuing bank are
    so it can resolve references in clauses like "NOTIFY APPLICANT" or
    "TO ORDER OF ISSUING BANK".
    """
    fields = structured_lc.get('consolidated_fields', structured_lc)

    def _get(keys):
        """Try multiple possible key names — field naming varies between SWIFT formats."""
        for k in keys:
            v = fields.get(k)
            if v:
                return v if isinstance(v, str) else str(v)
        return 'N/A'

    return {
        'applicant': _get(['Applicant', 'Applicant_Name', '50']),
        'beneficiary': _get(['Beneficiary', 'Beneficiary_Name', '59']),
        'issuing_bank': _get(['Issuing_Bank', 'Issuing_Bank_Details', 'Sending_Institution', '52A']),
        'currency_amount': _get(['Amount', 'Currency_Amount', 'LC_Amount', '32B']),
    }


def _extract_clauses(structured_lc: dict) -> List[dict]:
    """
    Extract individual clauses from StructuredLC consolidated_fields.

    Handles three value formats:
    1. String — a single clause (treated as clause #1)
    2. List of strings — multiple clauses numbered 1, 2, 3...
    3. List of dicts — clauses with metadata (text + clause_number)
    """
    fields = structured_lc.get('consolidated_fields', structured_lc)
    clauses = []

    for field_tag in sorted(fields.keys()):
        if field_tag not in CLAUSE_FIELDS:
            continue

        value = fields[field_tag]

        # Normalize value into a list of text strings
        if isinstance(value, str):
            clause_texts = [value]
        elif isinstance(value, list):
            clause_texts = []
            for item in value:
                if isinstance(item, str):
                    clause_texts.append(item)
                elif isinstance(item, dict):
                    clause_texts.append(item.get('text', item.get('clause_text', str(item))))
                else:
                    clause_texts.append(str(item))
        else:
            clause_texts = [str(value)]

        # Create one entry per clause, numbered sequentially
        for i, text in enumerate(clause_texts, 1):
            if not text or not text.strip():
                continue
            clauses.append({
                'field_tag': field_tag,
                'clause_number': i,
                'clause_ref': f"{field_tag}-{i}",
                'text': text.strip(),
            })

    return clauses


# ── Main Run ─────────────────────────────────────────────────────────────

def run(structured_lc: dict, output_dir: str = None, progress_callback=None) -> dict:
    """
    Execute Step 12: Decompose LC clauses into individual checkable conditions.

    The function:
    1. Extracts all clause-type fields from the StructuredLC
    2. Separates implicit (code-checkable) from VLM-decomposed clauses
    3. Sends VLM clauses concurrently to the Qwen model
    4. Builds Condition objects from VLM responses
    5. Returns the full list of DecomposedClause objects

    Args:
        structured_lc: Output from Step 7 (StructuredLC with consolidated_fields)
        output_dir: Directory to save results
        progress_callback: Optional callback(message: str)

    Returns:
        dict with 'decomposed_clauses', 'total_conditions', 'elapsed_seconds'
    """
    def _progress(msg):
        if progress_callback:
            progress_callback(f"[Step 12] {msg}")
        print(f"[Step 12] {msg}")

    start_time = time.time()

    # Handle Step 7 output format: may have 'structured_lc' wrapper or 'all_clauses' directly
    _inner = structured_lc.get('structured_lc', structured_lc)
    # If Step 7 already pre-built clause objects, use them directly
    _prebuilt_clauses = _inner.get('all_clauses', [])

    # Extract LC context for VLM prompt enrichment (applicant, beneficiary, bank, amount)
    lc_context = _extract_lc_context(_inner)
    _progress(f"LC context: Applicant={lc_context['applicant'][:30]}, Bank={lc_context['issuing_bank'][:30]}")

    # Extract all clauses — use prebuilt if available from Step 7
    if _prebuilt_clauses:
        # Convert Step 7's clause format to Step 12's expected format
        clauses = []
        for _pc in _prebuilt_clauses:
            _ft = _pc.get('field_tag', '')
            if _ft in CLAUSE_FIELDS:
                clauses.append({
                    'clause_ref': f"{_ft}-{_pc.get('clause_number', 1)}",
                    'field_tag': _ft,
                    'clause_number': _pc.get('clause_number', 1),
                    'text': _pc.get('clause_text', _pc.get('text', '')),
                })
    else:
        clauses = _extract_clauses(_inner)
    _progress(f"Found {len(clauses)} clauses to decompose")

    decomposed: List[DecomposedClause] = []
    vlm_tasks = []
    errors = []

    # ── Separate implicit vs VLM-decomposed clauses ──
    # Implicit: date/amount fields checked by Python code (faster, more reliable)
    # VLM: document requirement clauses decomposed by AI (handles natural language)
    for clause in clauses:
        ft = clause['field_tag']

        # Check if this is an implicit (code-based) check
        if ft in IMPLICIT_CHECK_FIELDS:
            implicit_conds = _build_implicit_conditions(ft, clause['text'])
            dc = DecomposedClause(
                clause_ref=clause['clause_ref'],
                field_tag=ft,
                clause_number=clause['clause_number'],
                original_text=clause['text'],
                conditions=implicit_conds,
                elapsed_seconds=0.0,
            )
            decomposed.append(dc)
            _progress(f"  {clause['clause_ref']}: {len(implicit_conds)} implicit conditions")
        else:
            vlm_tasks.append(clause)

    # ── Send VLM tasks concurrently ──
    # Multiple clauses are processed in parallel to reduce total latency.
    # MAX_CONCURRENT_VLM controls how many simultaneous requests hit the model server.
    if vlm_tasks:
        _progress(f"Sending {len(vlm_tasks)} clauses to VLM for decomposition...")
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_VLM) as executor:
            futures = {}
            for clause in vlm_tasks:
                future = executor.submit(
                    _call_vlm_decompose,
                    clause['clause_ref'],
                    clause['field_tag'],
                    clause['clause_number'],
                    clause['text'],
                    lc_context,
                )
                futures[future] = clause

            for future in as_completed(futures):
                clause = futures[future]
                result = future.result()

                if 'error' in result:
                    errors.append(f"{clause['clause_ref']}: {result['error']}")
                    _progress(f"  {clause['clause_ref']}: ERROR - {result['error']}")

                # Build Condition objects from VLM response JSON
                conditions = []
                for j, cond_raw in enumerate(result.get('conditions', []), 1):
                    conditions.append(Condition(
                        condition_id=f"{clause['clause_ref']}-C{j}",
                        condition_text=cond_raw.get('condition_text', ''),
                        document_to_check=cond_raw.get('document_to_check', ''),
                        look_for_value=cond_raw.get('look_for_value', ''),
                    ))

                dc = DecomposedClause(
                    clause_ref=clause['clause_ref'],
                    field_tag=clause['field_tag'],
                    clause_number=clause['clause_number'],
                    original_text=clause['text'],
                    conditions=conditions,
                    elapsed_seconds=result.get('elapsed', 0),
                )
                decomposed.append(dc)
                _progress(f"  {clause['clause_ref']}: {len(conditions)} conditions ({result.get('elapsed', 0):.1f}s)")

    # ── Post-processing: fix common VLM misinterpretations ──
    for dc in decomposed:
        original_upper = dc.original_text.upper()

        # Filter out false conditions from permissive clauses
        filtered = []
        for cond in dc.conditions:
            ct_upper = cond.condition_text.upper()
            doc_upper = cond.document_to_check.upper()

            # "THIRD PARTY DOCUMENTS ACCEPTABLE EXCEPT X AND Y"
            # VLM sometimes creates "X is not acceptable" — wrong interpretation
            if re.search(r'THIRD\s+PARTY\s+DOCUMENTS?\s+(ARE\s+)?ACCEPTABLE', original_upper):
                if re.search(r'(IS\s+)?NOT\s+ACCEPTABLE', ct_upper):
                    _progress(f"  {dc.clause_ref}: FILTERED false condition: {cond.condition_text[:80]}")
                    continue

            # BL should never check for dollar amount
            if 'BILL OF LADING' in doc_upper and re.search(
                    r'\b(AMOUNT|DOLLAR|PRICE|USD|EUR|GBP)\b', ct_upper):
                # Unless the original clause explicitly says BL must show amount
                if 'BILL OF LADING' not in original_upper or 'AMOUNT' not in original_upper:
                    _progress(f"  {dc.clause_ref}: FILTERED BL amount check: {cond.condition_text[:80]}")
                    continue

            # Tolerance clauses should not create verification conditions
            if re.search(r'(PLUS|MINUS|TOLERANCE).*(ACCEPTABLE|ALLOWED)', original_upper):
                if re.search(r'TOLERANCE|PERCENT', ct_upper):
                    _progress(f"  {dc.clause_ref}: FILTERED tolerance (handled by code): {cond.condition_text[:80]}")
                    continue

            # "LARGER QUANTITIES ACCEPTABLE" is permissive — no check needed
            if re.search(r'LARGER\s+QUANTIT.*(ACCEPTABLE|ALLOWED)', original_upper):
                _progress(f"  {dc.clause_ref}: FILTERED permissive larger-qty clause: {cond.condition_text[:80]}")
                continue

            # "GOODS DESCRIPTION SHOWING X TO BE ACCEPTABLE" is permissive
            if re.search(r'GOODS\s+DESCRIPTION\s+SHOWING.*ACCEPTABLE', original_upper):
                _progress(f"  {dc.clause_ref}: FILTERED permissive goods-desc clause: {cond.condition_text[:80]}")
                continue

            # General permissive pattern: clause ends with "TO BE ACCEPTABLE" or "ACCEPTABLE"
            # and the entire clause is about acceptability, not about requirements
            if (dc.field_tag == '47A'
                    and re.search(r'(TO\s+BE\s+)?ACCEPTABLE\.?\s*$', original_upper)
                    and 'NOT ACCEPTABLE' not in original_upper
                    and not re.search(r'MUST|SHALL|REQUIRED|SHOULD APPEAR', original_upper)):
                _progress(f"  {dc.clause_ref}: FILTERED permissive F47A clause: {cond.condition_text[:80]}")
                continue

            filtered.append(cond)

        dc.conditions = filtered

    # Post-processing: multi-document auto-add REMOVED — temperature 0 + prompt is sufficient

    # Sort by clause_ref for consistent ordering (e.g., 45A-1, 46A-1, 46A-2, 47A-1)
    decomposed.sort(key=lambda d: d.clause_ref)

    # Assign condition IDs if not already set (ensures every condition has a unique ID)
    for dc in decomposed:
        for j, cond in enumerate(dc.conditions, 1):
            if not cond.condition_id:
                cond.condition_id = f"{dc.clause_ref}-C{j}"

    total_conditions = sum(len(dc.conditions) for dc in decomposed)
    elapsed = time.time() - start_time
    _progress(f"Step 12 complete: {len(decomposed)} clauses -> {total_conditions} conditions in {elapsed:.1f}s")

    # Save results to disk
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        result_file = os.path.join(output_dir, 'step12_result.json')
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({
                'step': 12,
                'step_name': 'Clause-by-Clause Condition Decomposition',
                'total_clauses': len(decomposed),
                'total_conditions': total_conditions,
                'elapsed_seconds': round(elapsed, 2),
                'errors': errors,
                'decomposed_clauses': [asdict(dc) for dc in decomposed],
            }, f, indent=2, ensure_ascii=False)

    return {
        'decomposed_clauses': decomposed,
        'total_clauses': len(decomposed),
        'total_conditions': total_conditions,
        'elapsed_seconds': round(elapsed, 2),
        'errors': errors,
    }


if __name__ == '__main__':
    # CLI test with sample LC data — demonstrates decomposition of typical clauses
    sample_lc = {
        'consolidated_fields': {
            '46A': [
                "SIGNED COMMERCIAL INVOICE IN 3 COPIES SHOWING HS CODE",
                "FULL SET OF CLEAN ON BOARD BILLS OF LADING MADE OUT TO ORDER OF ISSUING BANK MARKED FREIGHT PREPAID NOTIFY APPLICANT",
                "INSURANCE POLICY OR CERTIFICATE IN DUPLICATE FOR 110 PCT OF INVOICE VALUE COVERING ALL RISKS",
            ],
            '47A': [
                "ALL DOCUMENTS MUST SHOW LC NUMBER AND DATE",
                "HS CODE MUST APPEAR ON BL AND INVOICES",
            ],
            '31D': "2026-06-30 PAKISTAN",
            '44C': "2026-06-15",
            '32B': "USD 490,200.00",
            'Applicant': 'ABC Trading Co.',
            'Beneficiary': 'XYZ Exports Ltd.',
            'Issuing_Bank': 'United Bank Limited',
        }
    }
    result = run(sample_lc)
    for dc in result['decomposed_clauses']:
        print(f"\n{dc.clause_ref}: {dc.original_text[:80]}...")
        for c in dc.conditions:
            print(f"  -> [{c.document_to_check}] {c.condition_text}")
