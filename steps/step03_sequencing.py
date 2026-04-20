"""
Step 3 -- Page Sequencing and Document Packet Formation
========================================================
Groups consecutive PDF pages into logical "document packets". A single PDF
contains multiple trade-finance documents (LC pages, Bills of Lading,
Invoices, Certificates, etc.) concatenated with no explicit separators.

HOW IT WORKS:
    Phase 1 -- Qwen classifies EVERY page:
        Sends page image + GLM text to Qwen VLM for each page.
        Qwen returns: document_type, is_continuation, confidence, stamps, signatures.
        This is the PRIMARY classification — Qwen sees the actual image.

    Phase 2 -- Group pages into packets:
        Pages with same document_type in sequence are grouped.
        Continuation pages merge into the previous packet.
        Copy detection: if same doc_type appears again after a different doc, it's a new copy.

    Phase 3 -- Context re-check (optional):
        For low-confidence pages, re-send with context of surrounding pages.

WHY QWEN FOR EVERY PAGE:
    - Text-based boundary detection is unreliable (many docs lack clear headers)
    - Copies of same document have identical text — only visual differences (stamps, markings)
    - Endorsement pages, blank backs, stamp-only pages need visual understanding
    - The old system (posss3.py) used VLM for every page and it worked well

INPUT:  Step 2 output -- list of PageCleaned (cleaned_text + raw_text + page_image_path)
OUTPUT: List of DocumentPacket objects with pages[], doc_type, boundary_confidence

MODEL:  Qwen VLM at QWEN_VLM_URL (7B or 72B per VLM_MODEL_SIZE switch in
        config/settings.py — classifies every page).
        GLM text included in every prompt (Qwen reviews, never rewrites)
"""

import os
import sys as _sys; _sys.stdout.reconfigure(encoding="utf-8", errors="replace") if hasattr(_sys.stdout, "reconfigure") else None
import re
import json
import time
import base64
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import QWEN_VLM_URL, QWEN_VLM_MODEL, MAX_CONCURRENT_VLM, VLM_TIMEOUT
try:
    from config.settings import QWEN_TEXT_LLM_URL, QWEN_TEXT_LLM_MODEL
except ImportError:
    QWEN_TEXT_LLM_URL = os.getenv("QWEN_TEXT_LLM_URL", "")
    QWEN_TEXT_LLM_MODEL = os.getenv("QWEN_TEXT_LLM_MODEL", "Qwen2.5-72B-Instruct")


# ── Data Models ──

@dataclass
class PageClassification:
    """Qwen's classification of a single page."""
    page_number: int
    document_type: str = "unknown"        # Bill of Lading, Commercial Invoice, LC, etc.
    is_continuation: bool = False          # True = continuation of previous document
    confidence: float = 0.0
    stamps: List[dict] = field(default_factory=list)
    signatures: List[dict] = field(default_factory=list)
    seals: List[dict] = field(default_factory=list)
    logos: List[dict] = field(default_factory=list)
    copy_status: str = "unknown"           # original, copy, non_negotiable
    copy_label: str = ""                   # ORIGINAL, COPY, NON-NEGOTIABLE, FIRST ORIGINAL
    marking_status: str = "unknown"        # stamped_and_signed, signed, stamped, unsigned
    doc_hint: str = ""                     # Additional context from Qwen
    raw_text: str = ""
    cleaned_text: str = ""
    page_image_path: str = ""


@dataclass
class DocumentPacket:
    """A group of pages forming one logical document."""
    packet_id: str = ""
    document_type: str = "unknown"
    pages: List[dict] = field(default_factory=list)
    page_numbers: List[int] = field(default_factory=list)
    boundary_confidence: float = 0.0
    copy_status: str = "original"
    copy_label: str = ""
    marking_status: str = "unsigned"
    stamps: List[dict] = field(default_factory=list)
    signatures: List[dict] = field(default_factory=list)
    seals: List[dict] = field(default_factory=list)
    logos: List[dict] = field(default_factory=list)
    doc_hint: str = ""
    # ── New additive fields (backward compatible) ──
    # bl_subtype: populated only for Bill of Lading packets (Tier 7)
    #   form_type, contract_type, issuer_type, signing_type,
    #   has_terms_overleaf, is_blank_back, carrier_name, forwarder_name
    bl_subtype: Optional[dict] = None
    # unified_summary: structured fields extracted from ALL pages of the
    # packet (Tier 8). Used by step09/step13/step14 to avoid page-1-only bias.
    unified_summary: Optional[dict] = None
    # validation_status: valid | re_checked | low_confidence (Tier 3)
    validation_status: str = "valid"


# ──────────────────────────────────────────────────────────────────────── #
# PROMPTS — 5 specialized sub-prompts (editable in settings page)           #
#                                                                          #
# Each prompt does ONE job. Sub-call flow per page:                        #
#   Step 3a → CLASSIFY_DOCTYPE_PROMPT   (text-heavy: doc type + continuation) #
#   Step 3b → EXTRACT_MARKINGS_PROMPT   (visual: stamps, sigs, seals, logos)  #
#   Step 3c → COPY_STATUS_PROMPT        (narrow: ORIGINAL/COPY/NON-NEGOTIABLE)#
# Packet-level calls (run once per packet, not per page):                  #
#   Step 3d → BL_SUBTYPE_PROMPT         (BL sub-type — 8 yes/no fields)    #
#   Step 3e → PACKET_SUMMARY_PROMPT     (structured summary across pages)  #
# ──────────────────────────────────────────────────────────────────────── #


# ── Step 3a — Document Type Classification (text-dominant) ──
CLASSIFY_DOCTYPE_PROMPT = """You are a trade finance document classifier. Classify ONE page.

GLM OCR TEXT (trusted):
{glm_text}

Return ONLY JSON:
{{
  "document_type": "exact doc type or heading visible on page",
  "is_continuation": false,
  "confidence": 0.95,
  "doc_hint": "one-line description"
}}

COMMON TYPES (use exact heading when visible):
  LC, Amendment, MT799, MT999, Bill of Lading, Commercial Invoice,
  Draft Bill of Exchange, Packing List, Certificate of Origin,
  Insurance Certificate, Insurance Policy, Weight Certificate,
  Quality Certificate, Quantity Certificate, Shipment Advice,
  Document Remittance, Beneficiary Certificate, Fumigation Certificate,
  Phytosanitary Certificate, Health Certificate, Inspection Certificate,
  Notice of Readiness, Port Clearance Certificate, Tanker Cleanliness Certificate,
  Shore Tank Measurements, Time Sheet, Vessel Experience Factor,
  Master Receipt for Sealed Samples, Letter of Authority, Letter of Indemnity,
  Certificate of Receipted Quantity, Products Quality Certificate,
  Products Quantity Certificate, Loading Inspection Report, Survey Report,
  Ullage Report, Cargo Manifest, Mate Receipt, Debit Note, Credit Note,
  Proforma Invoice, Air Waybill, Railway Bill, CMR, Courier Receipt,
  Truck Receipt, Delivery Order, Warehouse Receipt, Halal Certificate,
  Radiation Certificate, Non-GMO Certificate, Age Certificate,
  Pre-Shipment Inspection Certificate, GSP Form A, Export License,
  Import License, Chamber of Commerce Certificate, Mill Certificate,
  BL Conditions of Carriage, Endorsement Page, Covering Letter, Header Page.

DECISIVE RULES (apply in order):

1. SWIFT / LC family:
   - F-tags visible (F20:, F26E:, F31C:, F46A:, F47A:, :20:, :31C:, :46A:, :47A:)
     → "LC" (default). If F26E or "Date of Amendment" present → "Amendment".
   - "fin.799" / "Free Format Message" / "F79:" / "MT 799" / "Bank-to-Bank Message"
     → "MT799" (NOT an LC, even if body mentions 45A/46A/47A — those are references
     inside the free-format narrative, not real fields).
   - Same signals with 999 → "MT999".
   - Page with ONLY a bank letterhead/logo/SWIFT BIC and NO F-tags → "Header Page"
     or "Covering Letter" (NOT an LC). "FUSION TRADE INNOVATION" header +
     "Select 'Print' to output" with no SWIFT body → "Header Page".

2. Draft / Bill of Exchange:
   - "BILL OF EXCHANGE" OR any TWO of: "PAY THIS FIRST/SECOND/THIRD",
     "AT SIGHT" / "AT XX DAYS SIGHT", "DRAWN ON", "FOR VALUE RECEIVED",
     "DRAWER/DRAWEE", "TO THE ORDER OF" → "Draft Bill of Exchange".
   - Drafts are short (~300-500 chars). FIRST/SECOND/THIRD copies are
     each separate Draft packets — do NOT merge them via is_continuation.
   - A page with "AT Sight" + "DRAWN ON" + L/C number but no BL fields is a
     DRAFT, not a BL.

3. Bill of Lading family:
   - "BILL OF LADING" / "B/L NO" / "B/L NO." / "TANKER BILL OF LADING"
     / "CONGENBILL" / "GENCON" / SHIPPER+CONSIGNEE+VESSEL+PORTS together
     → "Bill of Lading".
   - "Details As Per Attached Sheet(s)" / "See Attached" / "As Per Rider" in a
     BL → the next page(s) are BL continuation (is_continuation=true).
   - Full-set BL = 3 originals + N non-negotiables. Each copy is its OWN packet;
     do NOT merge them via is_continuation. Different copy_status ⇒ different packet.
   - Back of BL with endorsement stamps only (e.g. "TO THE ORDER OF..." stamps,
     no shipper/consignee/vessel) → "Endorsement Page" with is_continuation=true.

4. BL Conditions of Carriage (CRITICAL — NOT a BL):
   - Page titled "Conditions of Carriage" / "Conditions of Contract" /
     "Terms and Conditions" containing ONLY numbered legal clauses (Hague Rules,
     Paramount, Jason, BIMCO, General Average, Arbitration, Liberty and Deviation)
     and NO shipper/consignee/vessel/port fields → "BL Conditions of Carriage".
   - This is the REVERSE side of a BL, NOT a BL itself.

5. Commercial Invoice family:
   - "COMMERCIAL INVOICE" / "Invoice No." / "Invoice Number" → "Commercial Invoice".
   - "PROFORMA INVOICE" → "Proforma Invoice".
   - "DEBIT NOTE" / "CREDIT NOTE" → those exact types.
   - Page 2+ of an invoice with additional line items / totals / bank details and
     no new title → "Commercial Invoice" with is_continuation=true.

6. Packing List: "PACKING LIST" / "PACKING SLIP" → "Packing List".

7. Bank presentation schedules (easy to confuse with LC):
   - "L/C BILLS SCHEDULE" / "COVERING SCHEDULE" / "DOCUMENT PRESENTATION" /
     "EXPORT DC DOCUMENT PRESENTATION SCHEDULE" with document list + CBC +
     amount + maturity → "Document Remittance" (NOT an LC).

8. Surveyor / weight / quality reports — USE EXACT TITLE:
   - "WEIGHT / QUALITY CERTIFICATE" → "Weight / Quality Certificate".
   - "QUALITY / ANALYSIS" (only test results, no weight) → "Quality / Analysis"
     (NOT "Inspection Certificate" — use the actual heading).
   - "FULL LOADING SURVEY REPORT" (loading timeline, tank cleanliness) →
     "Full Loading Survey Report".
   - "HEALTH CERTIFICATE" → "Health Certificate".
   - Always use the EXACT document heading. Do NOT force-fit generic categories.

9. Continuation / multi-page (CRITICAL — read carefully):
   - "Page X of Y" with X > 1 → is_continuation=true ALWAYS, even if the
     page has its own prominent heading / title block / letterhead. A
     surveyor, bank, or lab commonly repeats their letterhead on EVERY
     page of a multi-page report. The HEADER does NOT define the doc
     boundary — the "Page N of M" footer does.
     Example:
       Page 8 has heading "CERTIFICATE OF QUALITY AND WEIGHT..." + "Page 1 of 2"
         → new doc, cont=false
       Page 9 has heading "REPORT" or even repeats the same letterhead +
         "Page 2 of 2" → is_continuation=true (SAME doc as page 8).
   - If the page shows "Page X of Y" with X > 1 AND your instinct is
     "this looks like a new doc" because of the heading, STILL tag
     is_continuation=true. The doc_type should match the previous page's
     (the one with "Page 1 of Y").
   - Letterhead-or-footer-only page following a classified doc → SAME type as
     previous page with is_continuation=true. NEVER return "unknown" here —
     either continue the previous type or use "Header Page".
   - A continuation page mentions amounts/quantities/goods matching the previous
     doc → same type, is_continuation=true. Do NOT relabel as a different type.

10. Fallback: use the EXACT heading visible on the page. "unknown" is a last resort.

CRITICAL DON'TS:
- Do NOT classify a BL copy (same text, different stamp) as Shipment Advice.
- Do NOT confuse REFERENCES (an "Invoice No." inside a BL body) with DOCUMENT
  TYPE — look at HEADERS + FIELD STRUCTURE, not keywords in body text.
- Do NOT classify an MT799/MT999 as "LC" or "Amendment" even if its body
  references F45A / F46A / F47A — F79 narrative field or fin.799 header is
  decisive.
- Do NOT relabel a BL copy/original as something else because stamps differ.

Return JSON only.
"""


# ── Step 3b — Visual Markings Extraction (visual-only, no doc-type rules) ──
EXTRACT_MARKINGS_PROMPT = """Look at this page image. List EVERY visual marking you see.

Return ONLY JSON:
{
  "stamps": [{"text": "exact text inside stamp", "type": "rubber_stamp|embossed|printed", "position": "top-right|bottom-left|center|etc"}],
  "signatures": [{"description": "handwritten signature shape/style", "type": "handwritten|digital", "signatory": "name if readable, else empty"}],
  "seals": [{"description": "round/oval company seal", "position": "where on page"}],
  "logos": [{"company_name": "name in logo", "position": "top-left|header|etc"}]
}

Rules:
- Include EVERY visible marking, even faint ones. Err on the side of including.
- For stamps: read the exact text inside the stamp bounding box.
- For signatures: describe the shape if name is illegible (e.g. "looping cursive initials").
- Return empty arrays [] for categories with nothing found.
- Do NOT classify the document type. Do NOT describe page content outside markings.
- Only output JSON. No commentary.
"""


# ── Step 3c — Copy / Original Status (narrow targeted question) ──
COPY_STATUS_PROMPT = """Look at this page image. Find the ORIGINAL / COPY / NON-NEGOTIABLE marker
(usually a stamp, sometimes printed text) and report the document's copy status.

Return ONLY JSON:
{
  "copy_status": "original|copy|non_negotiable|unknown",
  "copy_label": "exact text of the marker (e.g. 'ORIGINAL', 'COPY', 'NON-NEGOTIABLE', 'FIRST ORIGINAL', 'SECOND ORIGINAL', 'THIRD ORIGINAL')",
  "marking_status": "stamped_and_signed|signed|stamped|unsigned"
}

Rules:
- "FIRST ORIGINAL" / "SECOND ORIGINAL" / "THIRD ORIGINAL" → copy_status = original (record ordinal in copy_label)
- "NON-NEGOTIABLE" / "NON NEGOTIABLE" → copy_status = non_negotiable
- "COPY" (without "ORIGINAL") → copy_status = copy
- No marker visible → copy_status = "unknown"
- marking_status: does the page show BOTH stamps AND signatures, just one, or neither?

Only output JSON.
"""


# ── Step 3d — BL Sub-type Classification (packet-level, runs once per BL packet) ──
BL_SUBTYPE_PROMPT = """You are classifying a Bill of Lading that may span multiple pages. Below
is the full concatenated OCR text from all pages of the BL packet (front + reverse).
Determine the BL's full set of sub-type attributes.

PACKET TEXT (all pages):
{packet_text}

Return ONLY JSON:
{{
  "form_type": "short_form_blank_back|long_form_printed_overleaf|condensed|unknown",
  "contract_type": "standard|charter_party|combined_transport|through|tanker|multimodal|unknown",
  "issuer_type": "master_bl|house_bl|charter_party_bl|unknown",
  "signing_type": "master_signed|agent_for_master|carrier_signed|forwarder_signed|unknown",
  "cleanness": "clean|claused|unknown",
  "shipped_on_board_status": "shipped_on_board|received_for_shipment|unknown",
  "negotiability": "negotiable|non_negotiable|straight|unknown",
  "consigned_form": "to_order|to_order_of_bank|to_order_of_shipper|straight_consignee|bearer|unknown",
  "has_terms_overleaf": false,
  "is_blank_back": false,
  "is_short_form": false,
  "is_house_bl": false,
  "is_charter_party_bl": false,
  "is_claused_bl": false,
  "freight_status": "prepaid|collect|payable_at_destination|unknown",
  "carrier_name": "name of carrier/shipping line, empty if not identifiable",
  "forwarder_name": "name of forwarder if house_bl, else empty",
  "clausing_notes": "if claused, the exact text of any damage/defect clause (e.g. '2 BAGS TORN') — else empty",
  "bl_type_description": "one-sentence human summary, e.g. 'Charter-party, long-form, agent-for-master, clean-on-board, to-order-of-bank, non-negotiable copy'"
}}

DETECTION RULES:

Contract type:
  - "CONGENBILL" / "GENCON" / "AS PER CHARTER PARTY" / "CHARTER PARTY DATED" → charter_party
  - "COMBINED TRANSPORT" → combined_transport
  - "MULTIMODAL TRANSPORT" → multimodal
  - "THROUGH BILL OF LADING" → through
  - "TANKER BILL OF LADING" → tanker
  - else → standard

Signing (look in signature block):
  - "AS MASTER" or "MASTER OF THE VESSEL" → master_signed
  - "AS AGENTS ONLY FOR AND BY AUTHORITY OF" + CAPTAIN/MASTER → agent_for_master
  - "AS CARRIER" / "THE CARRIER" → carrier_signed
  - "AS FREIGHT FORWARDER" / "NVOCC" / "NON-VESSEL OPERATING" → forwarder_signed

Issuer:
  - Carrier/shipping line in letterhead → master_bl (also set is_house_bl=false)
  - Freight forwarder / NVOCC in letterhead → house_bl (also set is_house_bl=true)
  - CONGENBILL/GENCON + charter party → charter_party_bl (also set is_charter_party_bl=true)

CLEAN vs CLAUSED (CRITICAL — UCP 600 Art 27):
  - A CLEAN BL has NO NOTATION of damage, defect, or shortage on the cargo
    (e.g. "CLEAN ON BOARD" / "LADEN ON BOARD" / just "SHIPPED ON BOARD" with
    no damage remarks) → cleanness = "clean", is_claused_bl = false.
  - A CLAUSED BL (also called "dirty" / "foul" BL) has explicit damage/defect
    notations. Look for clauses like:
      • "2 BAGS TORN" / "X BAGS BROKEN" / "BROKEN PACKAGING"
      • "CARGO DAMAGED" / "IN DAMAGED CONDITION"
      • "LEAKING DRUMS" / "STAINED" / "WET DAMAGE"
      • "SHORT SHIPPED" / "SHORTAGE OF X UNITS"
      • "RUSTY" / "DENTED" / "TORN"
      • "WITH EXCEPTIONS" / "SUBJECT TO CLAUSE X"
      • Pre-printed "said to contain" + damage remarks
    → cleanness = "claused", is_claused_bl = true,
      clausing_notes = the exact offending clause text.
  - "SAID TO CONTAIN" / "SHIPPER'S LOAD AND COUNT" by themselves are STANDARD
    disclaimers and do NOT make the BL claused.

Shipped-on-board:
  - "SHIPPED ON BOARD" / "CLEAN ON BOARD" / "LADEN ON BOARD" with a date
    → shipped_on_board_status = "shipped_on_board"
  - "RECEIVED FOR SHIPMENT" without on-board notation
    → shipped_on_board_status = "received_for_shipment"

Negotiability:
  - "NON-NEGOTIABLE" / "NON NEGOTIABLE" stamp → non_negotiable
  - "NEGOTIABLE" or ORIGINAL stamp (and no NON-NEGOTIABLE) → negotiable
  - Consigned to a specific named party with no "TO ORDER" → straight

Consignee form (look at CONSIGNEE field):
  - "TO ORDER" (blank) → to_order
  - "TO ORDER OF [BANK NAME]" → to_order_of_bank
  - "TO ORDER OF [SHIPPER/beneficiary]" → to_order_of_shipper
  - A specific named company with no "ORDER OF" → straight_consignee
  - "TO BEARER" → bearer

Freight:
  - "FREIGHT PREPAID" → prepaid
  - "FREIGHT COLLECT" / "FREIGHT PAYABLE AT DESTINATION" → collect
  - "FREIGHT PAYABLE AT [X]" → payable_at_destination

Overleaf / blank back / short form — DECISIVE RULE:
  Every BL has ONE Terms & Conditions (T&C) page on its reverse (overleaf).
  The PACKET TEXT above is the COMBINED text of all pages of this BL packet,
  including any T&C page that was merged in during grouping.

  Check the PACKET TEXT for T&C content — look for legal-clause signals like:
    • "CONDITIONS OF CARRIAGE" / "TERMS AND CONDITIONS OF CARRIAGE"
    • "PARAMOUNT CLAUSE" / "HAGUE RULES" / "HAGUE-VISBY"
    • "JASON CLAUSE" / "NEW JASON CLAUSE"
    • "BOTH-TO-BLAME COLLISION CLAUSE"
    • "BIMCO" / "YORK-ANTWERP RULES"
    • "GENERAL AVERAGE" (in full legal framing, not just a mention)
    • "LIBERTY AND DEVIATION CLAUSE"
    • Numbered legal clauses (1. Paramount Clause, 2. ..., 3. ...)

  DECISION:
  - PACKET TEXT contains T&C legal clauses (detectable by above signals)
    → has_terms_overleaf = true
    → form_type = long_form_printed_overleaf
    → is_blank_back = false
    → is_short_form = false

  - "SEE OVERLEAF" / "CONDITIONS OF CARRIAGE SEE OVERLEAF" / "TERMS ON
    REVERSE" appears on the BL face but NO T&C clauses in packet text
    → has_terms_overleaf = true (text is on a separate page not in packet)
    → form_type = long_form_printed_overleaf
    → is_blank_back = false

  - NO T&C clauses detected in packet AND no "SEE OVERLEAF" reference on
    the BL face → is_blank_back = true
    → form_type = short_form_blank_back
    → is_short_form = true
    → has_terms_overleaf = false

  - Short form with external T&C URL ("see www.carrier.com/terms") →
    form_type = short_form_blank_back, is_short_form = true,
    has_terms_overleaf = false (terms are NOT physically attached).

  A BL WITHOUT ANY T&C (neither overleaf clauses nor a URL reference) is
  BLANK BACK — this is the core UCP 600 Art 20(c) consideration.

bl_type_description: Write ONE sentence combining the detected attributes,
e.g. "Charter-party, long-form, agent-for-master, clean on board,
to-order-of-bank, non-negotiable copy."

Return JSON only. Use "unknown" only when the signal is genuinely absent.
"""


# ── Step 3e — Packet Summary (structured extraction across all pages) ──
PACKET_SUMMARY_PROMPT = """You are creating a structured summary of a trade finance document
that may span multiple pages. Merge information from ALL pages into one object so
downstream verification checks work on the whole document, not page-1 only.

DOCUMENT TYPE: {doc_type}
PACKET TEXT (concatenated from all pages):
{packet_text}

Return ONLY JSON. INCLUDE ONLY fields that have real values. OMIT empty fields.

========================================================================
CAPTURE COMPLETE TEXT — NEVER TRUNCATE OR SUMMARIZE.
========================================================================

The PACKET TEXT below is the FULL concatenated OCR of every page of this
document (separated by "--- PAGE BREAK ---" markers). This is ONE logical
document even if each page has its own letterhead / heading / "Page N of M"
footer — that repetition is normal for multi-page docs. Treat the whole
packet as a single document and produce ONE unified summary covering ALL
pages.

For ALL text fields, capture the COMPLETE text as printed on the document.
Do NOT summarize, paraphrase, or keep only the first line.

CRITICAL for multi-page docs:
 - A single dates_found entry (e.g. certificate_issue_date) must reflect
   the date shown on the doc, NOT be multiplied per page.
 - A single amount_total applies to the whole doc — don't sum page totals
   if the same figure repeats on each page.
 - BUT line items, batch/lot rows, individual weights, and per-batch
   MFG/EXP dates MUST be captured EACH — if 16 line items span across
   pages 1-2, return 16 separate quantities_found entries.
 - goods_description must contain the COMPLETE text across all pages —
   don't truncate at a page break.
 - parties on the document appear once (usually on page 1) — don't
   duplicate per page.

CRITICAL — fields that commonly span multiple lines and MUST be captured in full:
  - goods_description: include EVERY line of the goods description block.
    LC / invoice / BL / packing list goods descriptions often run 5-20 lines
    (product + grade + variety + origin + specification + incoterm + port +
    proforma reference + etc.). Concatenate with spaces or keep line breaks.
  - marks_and_numbers: the FULL shipping marks block (container marks,
    package marks, L/C ref stamp, all lines).
  - key_clauses: EVERY clause, each as its own array element (do NOT merge).
  - notes: the FULL body of any remarks/notes block.
  - clausing_notes (for BL): exact text of every damage/defect clause.
  - any address: FULL multi-line address (street, city, postcode, country).

If a field has multiple values (e.g. two notify parties, three HS codes,
two quantities — ordered AND shipped), return them as SEPARATE entries in
the relevant array, NEVER collapsed into one.

========================================================================
EXTRACT EVERY DATE, AMOUNT, QUANTITY, REFERENCE, AND PARTY YOU CAN SEE.
========================================================================

Beyond the typed fields below, return FOUR structured arrays that capture
EVERY date / amount / reference number / named party that appears on the
document — each tagged with its role. These arrays are the authoritative
source used by downstream verification; don't skip any.

1. dates_found[] — every date-like string on the page.
   CAPTURE EVERY DATE. A single document can legitimately carry 10+ distinct
   dates (e.g. LC issue + LC expiry + latest shipment + invoice date +
   proforma date + BL issue + on-board + ETA + signing + bank received stamp +
   cert issue + test date + each batch's MFG + each batch's EXP). Each goes
   as its OWN entry.
   role vocabulary (use exactly one — or invent a descriptive snake_case
   role when nothing below fits, rather than "other"):
     — Document's own dates —
       bl_issue_date, invoice_date, draft_date, certificate_issue_date,
       issue_date (generic fallback)
     — LC / credit dates —
       lc_issue_date, expiry_date (LC F31D validity), latest_shipment_date,
       amendment_date, presentation_date
     — Shipment / vessel dates —
       onboard_date, shipment_date, loading_date, discharge_date,
       delivery_date, eta_date, etd_date, arrival_date, departure_date,
       sailing_date, nor_date, nor_tender_date, vessel_clearance_date,
       port_clearance_date, customs_clearance_date
     — Inspection / testing / surveying —
       inspection_date, test_date, sampling_date, survey_date,
       fumigation_date, treatment_date, sample_collection_date,
       test_report_date
     — Insurance / validity —
       insurance_effective_date, insurance_expiry_date,
       validity_start_date, validity_end_date
     — Product / batch / lot —
       manufacturing_date, production_date, packed_date,
       product_expiry_date, best_before_date, use_by_date,
       batch_date, lot_date
     — Contractual —
       charter_party_date, contract_date, maturity_date, due_date,
       payment_date
     — Signature / receipt stamps —
       signature_date, stamp_date, received_date, acceptance_date,
       transmittal_date
     — Fallback —
       other (last resort only — ALWAYS include raw text)
   Tagging hints:
     - "ETA ..." / "Estimated Time of Arrival" / "ETA DISPORT" → eta_date
     - "ETD" / "Estimated Time of Departure" → etd_date
     - "Actual arrival" / "Arrived on" → arrival_date
     - "SHIPPED ON BOARD" / "LADEN ON BOARD" → onboard_date
     - "NOTICE OF READINESS" tender → nor_tender_date
     - "Bank RECEIVED stamp" dated → stamp_date OR received_date
     - "Cert issued on" → certificate_issue_date
     - "L/C DATE" / "L/C ISSUE DATE" / "DC DATE" / "CREDIT DATE" anywhere on
       ANY document (even if the doc itself is a draft/invoice/cert) →
       lc_issue_date (NOT the doc's own issue_date). The referenced LC's
       issuance date is a separate role from the document's own issue date.
     - "MFG" / "MFG DATE" / "MANUFACTURING DATE" / "DATE OF MFG" /
       "PRODUCTION DATE" / "PACKED DATE" → manufacturing_date
     - "EXP" / "EXP DATE" / "EXPIRY DATE" / "EXPIRATION DATE" / "USE BY" /
       "USE BEFORE" (on goods/products — NOT on LC) → product_expiry_date
     - "BEST BEFORE" / "BBD" / "BEST BY" → best_before_date
     - "BATCH DATE" / "LOT DATE" (when separate from MFG date) → batch_date
     - Per-batch / per-lot dates on a packing list: one entry PER batch, tagged appropriately.
     - LC expiry (F31D) / certificate validity → expiry_date (distinct from product_expiry_date).
     - Use "other" ONLY when nothing above fits — include reason in raw.
   Format: {{"role": "...", "value": "YYYY-MM-DD", "raw": "text as printed"}}

2. amounts_found[] — every monetary amount on the page.
   role vocabulary (preferred):
     invoice_total, invoice_subtotal, line_item_amount, unit_price,
     draft_amount, lc_amount, freight_amount, freight_prepaid_amount,
     insurance_premium, insurance_amount, sum_insured,
     penalty, discount, tax, duty, demurrage, commission,
     bank_charges, handling_fee, storage_fee, advance_payment,
     deposit, balance_due, exchange_rate_amount, net_amount, gross_amount,
     loi_amount, refund, total_weight_value.
   Format: {{"role": "...", "currency": "USD", "value": "123456.78",
     "in_words": "amount in words if present", "raw": "USD 123,456.78"}}

3. references_found[] — every reference number / document number / code.
   Tagging hints:
     - "L/C NO" / "LC NUMBER" / "DC NUMBER" / "CREDIT NO" / "CREDIT NUMBER"
       → lc_reference (tag this on ANY document type — BLs, drafts, invoices,
       covering schedules, and shipment advices all commonly reference the LC).
     - "B/L NO" / "BL NUMBER" → bl_reference.
     - "INVOICE NO" / "INVOICE NUMBER" / "INV NO" → invoice_reference.
     - "CONTRACT NO" / "CONTRACT REF" → contract_reference.
     - "PROFORMA INV" → proforma_reference.
   role vocabulary (preferred):
     lc_reference, invoice_reference, bl_reference, draft_reference,
     contract_reference, proforma_reference, purchase_order,
     booking_reference, voyage_number, container_number, seal_number,
     vessel_imo, vessel_mmsi,
     hs_code, tariff_code, ncm_code, goods_code,
     ntn_number, tin_number, vat_number, eori_number, aeo_number,
     sales_tax_reg_no, sro_number, customs_declaration_number,
     cover_note_reference, policy_reference, certificate_reference,
     phytosanitary_certificate_number, health_certificate_number,
     weight_certificate_number, quality_certificate_number,
     coo_reference, gsp_form_a_number, chamber_reference,
     mill_certificate_number, test_report_number,
     batch_number, lot_number, shipping_marks, marks_and_numbers,
     warehouse_receipt_number, delivery_order_number, mate_receipt_number,
     export_license_number, import_license_number, export_registration,
     swift_bic, iban, account_number,
     msds_reference, tally_sheet_reference.
   Format: {{"role": "...", "value": "VANPAK10", "raw": "BL NO: VANPAK10"}}

4. parties_found[] — every named company / person / bank on the document.

   CRITICAL — MULTIPLE PARTIES IN THE SAME FIELD (notify, consignee,
   shipper, etc.) are VERY COMMON and MUST each be captured as a
   SEPARATE entry. Do NOT collapse, pick only the first, or treat the
   second as continuation text.

   Signals that a field carries multiple parties:
     • The word "and" / "And" / "AND" inside the block on its own line
     • A second full company name with its own address after the first
     • Multiple numbered entries ("1.", "2.")
     • Bullets / dashes between names
     • Bank names following a company (common: notify = company + bank)
     • "ALSO TO:" / "AND TO:" / "CC:" / "with a copy to" headers

   Example (MUST return BOTH as separate entries):
     Document text:
       "Notify Party (see clause 22)
        Global Brands Marketing (PVT) Ltd.
        204, E.I.Lines
        Karachi, Pakistan
        and
        BANK AL-HABIB LTD"
     CORRECT extraction:
       {{"role":"notify_party","name":"Global Brands Marketing (PVT) Ltd.",
         "address":"204, E.I.Lines, Karachi, Pakistan",
         "raw":"Global Brands Marketing (PVT) Ltd., 204, E.I.Lines, Karachi, Pakistan"}}
       {{"role":"second_notify_party","name":"BANK AL-HABIB LTD",
         "raw":"and BANK AL-HABIB LTD"}}
     WRONG extraction (do NOT do this):
       only one notify_party entry for "Global Brands Marketing".

   Use role=second_notify_party / second_consignee / co_shipper etc. when
   a SECOND party is present in the same field. If more than two, invent
   role=third_notify_party / ... to preserve all of them.

   role vocabulary (preferred):
     applicant, beneficiary, issuer, shipper, exporter, manufacturer, producer,
     consignee, ultimate_consignee, notify_party, second_notify_party,
     drawer, drawee, payee,
     issuing_bank, advising_bank, confirming_bank, reimbursing_bank,
     negotiating_bank, paying_bank, collecting_bank,
     carrier, vessel_owner, vessel_charterer, charterer,
     forwarder, freight_forwarder, nvocc, broker, agent, signing_agent,
     master, captain,
     insurer, insurance_broker,
     surveyor, inspector, testing_laboratory,
     certifying_authority, chamber_of_commerce,
     health_authority, agriculture_authority, port_authority,
     customs_authority, stevedore, tallyman, receiver.
   Format: {{"role": "...", "name": "UNITED BANK LTD", "raw": "to the order of UNITED BANK LTD",
     "address": "optional"}}

5. quantities_found[] — every QUANTITY/WEIGHT/COUNT on the document.
   These are NOT monetary amounts — they are physical units (MT, BAGS,
   CARTONS, CBM, M3, PCS, DRUMS, PIECES, TONS, KG, LBS, etc.).
   Capture EACH separately — a Packing List may show quantity_ordered AND
   quantity_shipped AND quantity_loaded AND quantity_discharged, all
   different. Do NOT collapse them into one.
   role vocabulary (preferred, but invent new snake_case roles when needed):
     quantity_ordered, quantity_shipped, quantity_loaded, quantity_discharged,
     quantity_declared, quantity_invoiced, quantity_allowed,
     gross_weight, net_weight, tare_weight, dead_weight,
     weight_per_package, weight_per_unit,
     measurement, volume, cubic_measurement,
     number_of_packages, number_of_containers, number_of_bags,
     number_of_drums, number_of_cartons, number_of_pallets, number_of_units,
     minimum_quantity, maximum_quantity, tolerance_percent.
   Format: {{"role": "...", "value": "65,052.890", "unit": "MT",
     "raw": "GROSS WEIGHT: 65,052.890 METRIC TONS"}}

6. other_details_found[] — ANY other factual detail on the document that
   doesn't fit the five arrays above. Examples: vessel dimensions, stowage
   holds, cargo grade, tariff notes, INCOTERMS, UCP/ICC clause references,
   Institute Classification Clause, validity clauses, governing law, special
   remarks, loading instructions, hold numbers, cargo quality specs, etc.
   Format: {{"role": "short_snake_case_label", "value": "the detail",
     "raw": "text as printed"}}

========================================================================
RULES FOR ALL SIX ARRAYS:
========================================================================
- Be EXHAUSTIVE. If a date / amount / quantity / reference / party / fact
  appears on the document, capture it. Do NOT skip anything as "minor".
- If multiple items share a role (e.g. TWO notify parties, THREE HS codes,
  FIVE dates all of different roles), return each as a SEPARATE entry.
- If a specific value fits no preferred role, INVENT a descriptive
  snake_case role (e.g. "vessel_imo", "institute_classification_clause",
  "arbitration_clause_reference") rather than using "other".
- Use role="other" ONLY as a last resort, AND include why in the raw text.
- Omit arrays entirely if the document truly has none of that kind of item.

========================================================================
EXPLICIT EXTRACTION CHECKLIST — DO NOT SKIP (CRITICAL)
========================================================================
Many documents carry data in SECTIONS the LLM may gloss over. Explicitly
LOOK FOR and capture these if they appear ANYWHERE on any page:

0. BL-SPECIFIC IDENTIFIERS (CRITICAL — often on Bill of Lading face):
   Modern BLs commonly carry an "L/C BACK REFERENCE" block listing:
     • L/C no. (Documentary Credit number) → references_found[role=lc_reference]
     • L/C opening date / L/C date / DC issue date →
       dates_found[role=lc_issue_date]
     • L/C opening bank / Issuing Bank / Credit-opening bank →
       parties_found[role=issuing_bank]
       DO NOT mix this into consignee even if consignee is "TO ORDER OF
       [the same bank]" — the issuing_bank role is SEPARATE from consignee
       role. If the consignee is "TO ORDER OF BANK XYZ" AND BANK XYZ is
       also named as "L/C opening bank", create BOTH entries:
       - parties_found[role=consignee, name="To The Order of Bank XYZ"]
       - parties_found[role=issuing_bank, name="Bank XYZ"]
     • Exporter's bank / Advising bank / Negotiating bank if shown →
       parties_found with the specific role.

1. IDENTIFIERS ALWAYS EXTRACTED TO references_found + typed fields:
   - Every HS Code / HTS code / commodity code → references_found[role=hs_code]
     AND hs_codes[] typed array.
   - NTN / TIN / VAT / EORI / AEO / SRO number → references_found with its
     specific role AND the matching typed field (ntn_number, tin_number, etc.).
   - Every container number, seal number → references_found[role=...]
     AND container_numbers[]/seal_numbers[] typed arrays.
   - Every LC / DC / proforma / purchase order / contract number →
     references_found[role=lc_reference / proforma_reference / ...]
     AND matching typed field.
   - SWIFT/BIC codes, IBAN, bank account numbers →
     references_found[role=swift_bic / iban / account_number].

2. DATES ALWAYS EXTRACTED TO dates_found (multiple distinct dates possible):
   - Doc's own issue date → certificate_issue_date / invoice_date / bl_issue_date / draft_date.
   - LC issue date referenced (e.g. "DC Date of Issue: 2-Jan-2026") →
     lc_issue_date (distinct from the doc's own issue_date).
   - Proforma invoice date referenced (e.g. "DATED: 28-Nov-2025") → invoice_date with clear raw.
   - Shipment / on-board / ETA / ETD / loading / discharge — all separate entries.
   - PRODUCT / BATCH / LOT level dates (CRITICAL — common on pharmaceuticals,
     food, chemicals, perishables, medical devices, packing lists):
       • "MFG" / "MFG DATE" / "MANUFACTURING DATE" → manufacturing_date
       • "EXP" / "EXP DATE" / "EXPIRY DATE" / "USE BY" → product_expiry_date
       • "BEST BEFORE" / "BBD" → best_before_date
       • "PRODUCTION DATE" / "PACKED" → production_date / packed_date
       • "BATCH DATE" / per-batch dating → batch_date
     If a packing list or invoice has SEVERAL batches each with its own
     MFG/EXP → return one entry per batch, each correctly tagged. Do NOT
     collapse them.

3. LINE ITEMS ON INVOICES / PACKING LISTS:
   - Capture EVERY line item. If an invoice has 16 rows, return 16 separate
     quantities_found entries (one per row) — do NOT skip duplicates or
     summarize "same product appears 10 times". Each Lot number, Qty, and
     Unit Price is a DISTINCT quantity entry.
   - In goods_description, include EVERY line item's full text (product code,
     lot, qty, HS, ECCN, COO, etc.). Do NOT truncate for brevity.

4. BANK / PAYMENT DETAILS — capture in references_found:
   - "Remit To" bank name → parties_found[role=paying_bank or collecting_bank]
   - SWIFT address / BIC → references_found[role=swift_bic]
   - Account numbers (Export A/C, Bank A/C) → references_found[role=account_number]

5. EXPORT CONTROL / ORIGIN NOTES:
   - Country of Origin → other_details_found[role=country_of_origin].
   - Export License Type / Number → references_found[role=export_license_number].
   - Destination Control Statements → other_details_found[role=destination_control_statement].

5b. INCOTERMS (CRITICAL — often in table cells under "Incoterms" column):
   - Any of: CFR, CIF, FOB, EXW, DAP, DDP, DAT, DPU, CPT, FCA, C&F, CNF, FAS
     → other_details_found[role=incoterms] with value = the 2-3-letter code.
   - Include the NAMED PLACE if present (e.g. "CFR KARACHI SEAPORT" or
     "FOB ANY CHINESE SEAPORT"). Raw keeps full text.
   - These often appear in a small table alongside "Sales Rep", "Haemo No",
     "Payment Term", "Currency" — capture them even if the column header is
     in a different row than the value.
   - Payment terms like "LC at Sight" / "TT" / "Cash" →
     other_details_found[role=payment_terms].

6. WEIGHTS / MEASUREMENTS:
   - Gross weight, net weight, tare weight, measurement (CBM) — all as
     SEPARATE quantities_found entries with proper units.

7. SHIPMENT ADVICE / NOTIFICATION EMAILS (P128 — CRITICAL for Shipment Advice):
   A Shipment Advice is typically an email/fax sent BY the beneficiary to
   MULTIPLE recipients (insurer, applicant, consignee, bank). The email
   header lists ALL recipients — capture EVERY ONE, not just the first.
   Watch for these header patterns:
     • "Sent: <date>"         → dates_found[role=advice_sent_date]
     • "To: a@x.com ; b@y.com ; c@z.com"  (top-of-document email header)
     • "TO: Company A"  ...  "TO: Company B"  (each "TO:" is a separate recipient)
     • "CC:" / "BCC:"         → also separate recipients
     • "E-Mail: foo(at)bar.com" or "foo@bar.com" under a recipient block
     • "Fax: +92-21-1234567"  under a recipient block
   EXTRACTION RULES:
   (a) EVERY distinct "TO:" recipient or email address in the "To:" header
       MUST become a parties_found entry. Use roles: notify_party,
       second_notify_party, third_notify_party, fourth_notify_party,
       insurer, applicant, consignee, advising_bank (pick the role that
       matches what the document says about the recipient's purpose).
   (b) EVERY email address on the document MUST be captured in
       other_details_found with role=notification_email, one entry per
       address. Normalize "(at)" / "(AT)" back to "@" in the value field,
       keep the raw verbatim. Even if the same address is in both the
       header and a per-recipient block, emit ONE entry per DISTINCT
       address but list all raw occurrences.
   (c) Fax numbers → other_details_found[role=notification_fax], one entry
       per distinct number.
   (d) The SUBJECT line → other_details_found[role=subject].
   (e) Open Policy / Cover Note / Insurance reference → references_found
       with role=open_policy_reference / cover_note_reference.
   (f) CRITICAL: do NOT drop a notify party just because the same company
       name also appears elsewhere on the document. If the email header
       lists three distinct companies/addresses, emit THREE parties_found
       entries. Missing a recipient makes the Shipment Advice fail the
       "notify all parties" check.
   (g) For the typed top-level field "notify_party", put the FIRST notify
       party only (it is a single string) — the full list lives in
       parties_found[].

If a value is on the document but you didn't extract it to a structured
array, you have FAILED this extraction. Re-check your output before
returning.

========================================================================
TYPED TOP-LEVEL FIELDS (convenience copies — pull best value from arrays above):
========================================================================

CRITICAL — TYPED FIELDS MUST BE SIMPLE STRINGS OR NUMBERS, NEVER OBJECTS:
  ✅ CORRECT:  "amount": "USD 97,216.00"
  ❌ WRONG:    "amount": {{"role": "draft_amount", "currency": "USD", "value": "97216.00"}}
  ✅ CORRECT:  "lc_reference": "0401ILC083248"
  ❌ WRONG:    "lc_reference": {{"role": "lc_reference", "value": "0401ILC083248"}}
Use structured objects ONLY inside the five structured arrays
(dates_found[], amounts_found[], quantities_found[], references_found[],
parties_found[], other_details_found[]) — NEVER as top-level field values.


{{
  "document_identifier": "the document's OWN reference/serial/tracking number (e.g. invoice no, BL no, cert no, LC no, draft no) — NEVER the document TYPE name. If there is no specific number, leave this empty rather than putting the doc type.",
  "issue_date": "YYYY-MM-DD",
  "issuer": "name of issuing party",
  "beneficiary": "beneficiary / recipient name",
  "shipper": "for BL/Invoice",
  "consignee": "for BL",
  "notify_party": "for BL",
  "drawer": "for Draft",
  "drawee": "for Draft",
  "payee": "for Draft",
  "goods_description": "full description — include all pages if it spans",
  "quantity": "with unit",
  "weight": "with unit",
  "amount": "with currency",
  "vessel_name": "for BL",
  "voyage_number": "for BL",
  "port_of_loading": "",
  "port_of_discharge": "",
  "place_of_receipt": "",
  "place_of_delivery": "",
  "shipment_date": "YYYY-MM-DD — on-board date for BL",
  "lc_reference": "any LC/DC number referenced",
  "invoice_reference": "any invoice number referenced",
  "contract_reference": "",
  "key_clauses": ["charter party dated X", "LC 47A clause", etc],
  "cross_references": ["references to other documents in the same packet set"],
  "notes": "anything unusual (name change, stamps-only page, etc.)",

  // Bill of Lading specific fields (include when doc is a BL — else omit)
  "bl_number": "BL / B/L number",
  "bl_date": "YYYY-MM-DD — date on the BL itself",
  "ntn_number": "NTN / National Tax Number (usually on Pakistan BLs)",
  "hs_codes": ["HS / HTS / commodity codes found (e.g. 1201.00.00)"],
  "freight_terms": "FREIGHT PREPAID / FREIGHT COLLECT / FREIGHT PAYABLE AT DESTINATION",
  "freight_amount": "amount + currency if shown on BL",
  "number_of_originals": "e.g. 3/THREE",
  "container_numbers": ["MSCU1234567", "etc"],
  "seal_numbers": ["seal IDs"],
  "marks_and_numbers": "shipping marks block (can be multi-line)",
  "gross_weight": "with unit",
  "net_weight": "with unit",
  "measurement": "CBM / volume with unit",
  "number_of_packages": "e.g. 500 BAGS",
  "package_type": "BAGS / BULK / DRUMS / CTNS",
  "onboard_date": "YYYY-MM-DD — 'SHIPPED ON BOARD' date",
  "signed_by": "signing party name (e.g. 'AS AGENT FOR CAPTAIN ...')",
  "charter_party_reference": "if CP BL: 'AS PER CHARTER PARTY DATED YYYY-MM-DD'"
}}

Rules:
- If a field value spans multiple pages (e.g. goods description on page 1 + continuation on page 2), merge them.
- For BL: include vessel + both ports + shipment date even if on different pages.
- For Invoice: include total amount (may be on last page) + all goods lines.
- For Draft: include drawer/drawee/payee + amount + tenor.
- If the document references other documents (LC no, invoice no), record them in lc_reference / invoice_reference / cross_references.
- Do NOT invent values. Omit fields with no actual data.

Only output JSON.
"""


# ── Back-compat alias — kept so existing imports (server.py) keep working ──
# Legacy single-prompt path. New flow uses the 5 prompts above.
CLASSIFY_PROMPT = """You are a trade finance document classifier. Look at this page image and the OCR text below.

GLM OCR TEXT (trusted — extracted from this page):
{glm_text}

CLASSIFY this page. Return ONLY valid JSON:
{{
    "document_type": "exact type from the list below",
    "is_continuation": false,
    "confidence": 0.95,
    "stamps": [{{"text": "stamp text if readable", "type": "rubber_stamp/embossed/printed", "position": "top-right"}}],
    "signatures": [{{"description": "handwritten signature", "type": "handwritten/digital", "signatory": "name if readable"}}],
    "seals": [{{"description": "round company seal"}}],
    "logos": [{{"company_name": "company name", "position": "top-left"}}],
    "copy_status": "original or copy or non_negotiable",
    "copy_label": "exact text of marking: ORIGINAL, COPY, NON-NEGOTIABLE, FIRST ORIGINAL, SECOND ORIGINAL, THIRD ORIGINAL, etc.",
    "marking_status": "stamped_and_signed or signed or stamped or unsigned",
    "doc_hint": "brief 1-line description of what this page contains"
}}

DOCUMENT TYPES — Use the EXACT document title/heading visible on the page. Common types include but are NOT limited to:
  LC, Amendment, MT799, MT999, Bill of Lading, Commercial Invoice, Draft Bill of Exchange,
  Packing List, Certificate of Origin, Insurance Certificate, Insurance Policy,
  Weight Certificate, Quality Certificate, Quantity Certificate,
  Shipment Advice, Document Remittance, Beneficiary Certificate,
  Fumigation Certificate, Phytosanitary Certificate, Inspection Certificate,
  Notice of Readiness, Port Clearance Certificate, Tanker Cleanliness Certificate,
  Shore Tank Measurements, Time Sheet, Vessel Experience Factor,
  Master Receipt for Sealed Samples, Letter of Authority,
  Certificate of Receipted Quantity, Products Quality Certificate,
  Products Quantity Certificate, Loading Inspection Report,
  Survey Report, Ullage Report, Cargo Manifest, Mate Receipt,
  Debit Note, Credit Note, Proforma Invoice, Health Certificate,
  Full Loading Survey Report, Quality / Analysis, Weight / Quality Certificate,
  BL Conditions of Carriage, Letter of Indemnity,
  Endorsement Page, Blank Page, Covering Letter, Header Page

  Air Waybill, Railway Bill, CMR Consignment Note, Inland Waterway Bill,
  Courier Receipt, Truck Receipt, Delivery Order, Warehouse Receipt,
  Combined Transport Bill, Through Bill of Lading,
  Halal Certificate, Radiation Certificate, Non-GMO Certificate,
  Age Certificate, Pre-Shipment Inspection Certificate, GSP Form A,
  CITES Certificate, Dangerous Goods Declaration, Customs Declaration,
  Export License, Import License, Chamber of Commerce Certificate,
  Mill Certificate, Test Report, Legalized Document, Consularized Document,
  Agents Certificate, Vessel Classification Certificate,
  Draught Survey Report, Loading Report, Discharge Report,
  Collection Instruction, Reimbursement Schedule

  If the document does NOT match any of the above, use the ACTUAL title/heading
  visible on the document. NEVER force-fit a document into an incorrect category.

CLASSIFICATION RULES:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HIGH-PRIORITY RULES — READ FIRST. DO NOT RETURN "unknown" if ANY of these match.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

━━━ DRAFT / BILL OF EXCHANGE ━━━
A page is a "Draft Bill of Exchange" if it contains ANY TWO of:
  • "PAY THIS FIRST" / "PAY THIS SECOND" / "PAY THIS BILL OF EXCHANGE"
  • "AT SIGHT" / "AT XX DAYS SIGHT" / "AT XX DAYS AFTER SIGHT"
  • "TO THE ORDER OF"
  • "DRAWN ON" (followed by a bank name or BIC)
  • "FOR VALUE RECEIVED"
  • "BILL OF EXCHANGE"
  • "DRAWER" / "DRAWEE"
  • "FIRST ORIGINAL" / "SECOND ORIGINAL" alongside "PAY"
  • A typewritten "Pay this First / Second of exchange" block
Examples that MUST classify as "Draft Bill of Exchange":
  - "AT Sight PAY THIS Second First unpaid TO THE ORDER OF Ourselves... DRAWN ON: UNILPKKA"
  - "Pay this First Bill of Exchange (Second of same tenor and date being unpaid) to the order of..."
  - Any document showing currency amount + "AT Sight" + "DRAWN ON" + "L/C Number"
A draft is usually only ~300-500 chars. Multiple copies (First / Second / Third) are common — each copy is its own "Draft Bill of Exchange" packet.

━━━ CONGENBILL / GENCON / Bill of Lading ━━━
A page is a "Bill of Lading" if it contains ANY of:
  • "CONGENBILL" (BIMCO short-form BL — code name)
  • "GENCON" (BIMCO general charter BL)
  • "BILL OF LADING" / "B/L NO" / "BL NO"
  • A header showing SHIPPER + CONSIGNEE + VESSEL + PORT OF LOADING + PORT OF DISCHARGE together
  • "SHIPPED on board" / "CLEAN ON BOARD" / "LADEN ON BOARD"
  • "FREIGHT PREPAID" / "FREIGHT COLLECT" with shipping context
  • "AS PER CHARTER PARTY" / "CHARTER PARTY DATED"
  • "Number of original Bs/L"
A page that begins with "CODE NAME: CONGENBILL EDITION ____" is ALWAYS a Bill of Lading.

━━━ COMMERCIAL INVOICE LETTERHEAD-ONLY PAGE ━━━
If a page has ONLY the company letterhead / footer (name, address, phone, fax,
email, website) and NO actual document content (no items, no amounts, no
header words like "INVOICE" / "BILL OF LADING" / "CERTIFICATE"), classify it
as the SAME type as the PREVIOUS document with is_continuation=true.
For example: a page with only "Viterra B.V. | P.O. Box 1120, Rotterdam | info@bunge.com"
following a Commercial Invoice page is the back/footer of that invoice — classify
as "Commercial Invoice" with is_continuation=true. NEVER return "unknown" for
a letterhead-only page; either continue the previous type, or use "Header Page".

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- SWIFT F-tags (F20:, F31C:, F42A:, F46A:, F47A:, :20:, :31C:, :46A:) or "Message type: 700/707" -> "LC" or "Amendment"
- "26E: Number of Amendment" or "Date of Amendment" -> "Amendment"
- ── MT799 / MT999 FREE FORMAT MESSAGES (CRITICAL — read carefully) ──
  An MT799 or MT999 is a SWIFT free-format bank-to-bank message. It is NOT a Letter of Credit and NOT an MT707 amendment, even though it OFTEN references the underlying LC ("WITH REFERENCE TO OUR MT 700 DATED ...", "OUR LC NO. ...", "ABOVE CAPTIONED CREDIT").
  Identify an MT799/MT999 by ANY of these signals on the page:
    • "Identifier: fin.799" / "fin.799" / "fin.999"
    • "Expansion: Free Format Message"
    • "Free Format Message" anywhere on the page
    • "Bank-to-Bank Message" / "Bank to Bank Message"
    • "Message type: 799" / "Message Type: 799" / "MT 799" / "MT799"
    • A narrative field labelled "F79: Narrative" or ":79:" (this is the unique field of MT799 — it does NOT exist in MT700/707)
    • A SWIFT report header that shows "799" instead of "700"/"707"
  When you see ANY of those, classify the page as "MT799" (or "MT999"), regardless of what fields appear in the message body.
  In particular: an MT799 may carry F20 (Transaction Reference), F21 (Related Reference), F79 (Narrative). It may even MENTION F45A / F46A / F47A in the narrative body when the bank is sending an amendment via free format (e.g. "UNDER FIELD 45A RATE SHOULD READ AS 'EUR 141,396.00' I/O 'EUR 141,396.56'"). The presence of those field references in the BODY does NOT make it an LC or an MT707 — only F26E (Number of Amendment) makes a message a real MT707 amendment.
  If the page is an MT799 carrying free-format amendment instructions, STILL classify it as "MT799" — the downstream pipeline will detect the amendment intent and apply it.
- "Page X of Y" continuation or no own header -> is_continuation = true
- Page with "FUSION TRADE INNOVATION" header + "Select 'Print' to output" but NO SWIFT content -> "Header Page"
- Page mostly blank with only "TO THE ORDER OF" endorsement stamps/signatures -> "Endorsement Page"
- Back side of Bill of Lading with endorsement stamps only -> "Endorsement Page" with is_continuation = true
- "TANKER BILL OF LADING" or "BILL OF LADING" or "CONGENBILL" or "B/L NO." or "B / L NO." or "BL/MTD Number" -> "Bill of Lading"
- Bill of Lading has fields: SHIPPER, CONSIGNEE, NOTIFY ADDRESS, VESSEL, PORT OF LOADING, PORT OF DISCHARGE, GOODS DESCRIPTION, FREIGHT, "SHIPPED on board", ORIGINAL/NON-NEGOTIABLE stamp. If you see these fields, it is a Bill of Lading — NOT a Shipment Advice.
- MULTIPLE COPIES: A "Full Set" of BL means 3 originals (marked ORIGINAL) + non-negotiable copies (marked NON-NEGOTIABLE). Each copy is a SEPARATE Bill of Lading page with the SAME content but different stamp. Classify each as "Bill of Lading" — do NOT classify BL copies as "Shipment Advice" or any other type.
- Shipment Advice is a LETTER (usually 1 page) from beneficiary to insurance company listing vessel, B/L no, date, amount. It does NOT have CONSIGNEE, SHIPPER, PORT fields in a structured format like a BL.
- "PORT CLEARANCE" or Chinese port clearance form (国际航行船舶出口岸许可证) -> "Port Clearance Certificate"
- "NOTICE OF READINESS" -> "Notice of Readiness"
- "TANKER CLEANLINESS CERTIFICATE" -> "Tanker Cleanliness Certificate"
- "SHORE TANK MEASUREMENTS" -> "Shore Tank Measurements"
- "TIME SHEET" with loading events table -> "Time Sheet"
- "VESSEL'S EXPERIENCE FACTOR" or "MEASUREMENTS OF QUANTITY RECEIVED ON VESSEL" -> "Vessel Experience Factor"
- "MASTER'S RECEIPT FOR SEALED SAMPLES" -> "Master Receipt for Sealed Samples"
- "LETTER OF AUTHORITY" for signing BL -> "Letter of Authority"
- "CERTIFICATE OF RECEIPTED QUANTITY" -> "Certificate of Receipted Quantity"
- "PRODUCTS QUALITY CERTIFICATE" -> "Products Quality Certificate"
- "PRODUCTS QUANTITY CERTIFICATE" -> "Products Quantity Certificate"
- "COMMERCIAL INVOICE" or "Invoice number:" -> "Commercial Invoice"
- "PACKING LIST" or "PACKING SLIP" -> "Packing List" (Packing Slip is a synonym of Packing List, always classify as "Packing List")
- Email with attachment list / "SEDNA" / covering letter -> "Document Remittance" or "Covering Letter"
- ORIGINAL/COPY/NON-NEGOTIABLE stamps -> record in copy_status and copy_label
- FIRST ORIGINAL/SECOND ORIGINAL/THIRD ORIGINAL -> separate copies of same document

MULTI-PAGE DOCUMENTS:
- If a page has NO clear title/header of its own but contains continuation data (line items, amounts, table rows, etc.), it is likely a CONTINUATION of the PREVIOUS document.
- Commercial Invoice page 2: may show additional line items, totals, bank details, or certification text without repeating "Commercial Invoice" title → classify as "Commercial Invoice" with is_continuation=true
- Bill of Lading page 2: may show additional cargo details, terms & conditions → classify as "Bill of Lading" with is_continuation=true
- Look at the OCR text: if it contains amounts, quantities, goods descriptions, or reference numbers matching the previous page's document type, it is a continuation.
- Do NOT classify a continuation page as a completely different document type (e.g., don't call Invoice page 2 a "Certificate").

PAGE NUMBERING:
- If the page shows "Page X of Y" (e.g., "Page 5 of 29"), this is page X of a Y-page document. Set is_continuation=true if X > 1.
- Use the ACTUAL document title/heading from the page for document_type, not a generic category.
- Example: A page titled "PRE-SHIPMENT INSPECTION REPORT" with "Page 18 of 29" should be classified as "Pre-Shipment Inspection Report" with is_continuation=true — NOT as "Inspection Certificate".
- Example: A page with "CERTIFICATE OF CONFORMANCE" heading should be "Certificate of Conformance" — NOT "Inspection Certificate".
- Always use the SPECIFIC title visible on the document. "Inspection Certificate" is too generic — use the actual heading.

IMPORTANT — DO NOT CONFUSE REFERENCES WITH DOCUMENT TYPE:
- A Bill of Lading cargo description page may MENTION "Commercial Invoice No." or "L/C Number" — these are REFERENCES, not the document type.
- If the page has "H.B/L No." or "B/L No." or "Marks & Nos." or "Description of Goods" column headers, it is a BILL OF LADING — even if it mentions invoice numbers in the cargo text.
- Look at the PAGE HEADER and STRUCTURE (column headers, form fields) to determine document type — NOT keywords in the body text.

━━━ BL CONDITIONS OF CARRIAGE (CRITICAL — DO NOT CONFUSE WITH BILL OF LADING) ━━━
A page titled "Conditions of carriage" or "Conditions of Contract" or "Terms and Conditions"
that contains ONLY legal clauses (Hague Rules, Paramount clause, BIMCO, General Average,
Arbitration, Jason Clause, Liberty and Deviation) is a "BL Conditions of Carriage" — it is
the reverse/back side of a Bill of Lading, NOT a Bill of Lading itself.
Key signals: numbered legal clauses (1. Paramount clause, 2. Both-to-Blame, 3. New Jason, etc.),
no shipper/consignee/vessel/port fields, no "SHIPPED ON BOARD" clause.
ALWAYS classify as "BL Conditions of Carriage", NEVER as "Bill of Lading".

━━━ WEIGHT/QUALITY CERTIFICATE vs QUALITY/ANALYSIS vs FULL LOADING SURVEY REPORT ━━━
These are DIFFERENT documents issued by the same surveyor (e.g. Alfred H Knight):
- "WEIGHT / QUALITY CERTIFICATE": Has BOTH weight establishment data AND quality test results
  (specification table with TESTED RESULTS). Title says "WEIGHT / QUALITY CERTIFICATE".
- "QUALITY / ANALYSIS": Has ONLY quality test results (specification table). Title says
  "QUALITY / ANALYSIS" or just "QUALITY". This is NOT an "Inspection Certificate".
- "HEALTH CERTIFICATE": Certifies goods are fit for human consumption, free from haram elements.
  Title says "HEALTH CERTIFICATE".
- "FULL LOADING SURVEY REPORT": Has loading timeline (vessel arrived, cargo hose connected, etc.),
  fitness/cleanliness of tanks, last 3 cargoes table. Title says "FULL LOADING SURVEY REPORT".
USE THE EXACT TITLE from the document. Do NOT rename "QUALITY / ANALYSIS" to "Inspection Certificate".

━━━ L/C BILLS SCHEDULE / COVERING SCHEDULE / DOCUMENT PRESENTATION ━━━
A page from a negotiating bank (e.g. Standard Chartered, BIDV, BNP Paribas) showing
"L/C BILLS SCHEDULE" or "COVERING SCHEDULE" or "DOCUMENT PRESENTATION" or
"EXPORT DC DOCUMENT PRESENTATION SCHEDULE" with document list, CBC number, amount,
maturity, and instructions is a "Documentary Remittance" (also known as Covering
Schedule, L/C Bills Schedule, or Document Presentation).

P130 — DOCUMENT PRESENTATION vs MT799 (CRITICAL — common misclassification):
A cover letter whose TITLE says "Document Presentation" and which lists attached
documents in a TABLE is ALWAYS a "Document Remittance" (aka Documentary
Remittance / Document Presentation / Covering Schedule / Bill Remittance Letter)
— NEVER "MT799" / "MT999" / "MT740" even if those names appear as ROWS inside
the attachment table.

WORKED EXAMPLE (actual BIDV presenting-bank cover, do not misclassify):

    BIDV  NGÂN HÀNG TMCP ĐẦU TƯ VÀ PHÁT TRIỂN VIỆT NAM
    Bank for Investment and Development of Vietnam JSC
    HA NOI BRANCH ... Swift code: BIDVVNVX
    20 February 2025
    Mail to: BANK AL-HABIB LTD., ... KARACHI-74000, PAKISTAN

    Document Presentation                           ← PAGE TITLE

    Letter of credit no: 0001LC55282/2025
    Document set no:     BE25B00114-001
    Documents value:     USD 56,661.00
    Applicant:           AHMAD HASSAN TEXTILE MILLS LTD

    Please acknowledge receipt for the documents enclosed herewith:

    1st mail  2nd mail   Document Description       ← ATTACHMENT TABLE
    2                    Bill of Exchange
    3+1C                 Bill of Lading
    8                    Commercial Invoice
    1+1C                 Cert. of Origin
    3                    Packing list
    3                    shipping cert (14 DAYS)
    1                    shipment advice
    1                    SHIPPING CERT
    1C                   MT799                       ← row naming an enclosed item
    ...
    Settlement instructions for this document set:
    Please remit proceeds by SWIFT to our Hanoi Head Office's account ...
    Subject to the Rules mentioned in the underlying Letter of Credit ...

→ document_type = "Document Remittance" (NOT "MT799").
   The words "MT799" / "SHIPPING CERT" inside the table are attachment
   descriptions, not the page's classification. Confidence HIGH.

DISTINGUISHING RULE — be strict:
- If the page has ANY of these signals, it is a Document Remittance:
    • title line containing "Document Presentation" / "Bill Remittance Letter"
      / "Covering Schedule" / "L/C Bills Schedule" / "Export DC Document
      Presentation Schedule"
    • header block with "Letter of credit no:" + "Document set no:" +
      "Documents value:" + "Applicant:" on consecutive lines
    • "Please acknowledge receipt for the documents enclosed herewith"
    • a table with columns "1st mail" / "2nd mail" / "Document Description"
      (or "Originals" / "Copies" / "Description")
    • "Settlement instructions for this document set"
    • the page lists 3+ distinct document types as rows
- An actual MT799 page looks NOTHING like the above. It has:
    • page header "FIN.799" / "Free Format Message" / "Message type: 799"
    • SWIFT F-tags (F20 Transaction Reference, F21 Related Reference,
      F79 Narrative) as labelled blocks
    • a single free-text narrative body — NO document-enclosed table
    • no "1st mail / 2nd mail" columns

ATTACHED SHEETS:
- If a Bill of Lading says "Details As Per Attached Sheet(s)" or "See Attached" or "As Per Rider", the NEXT page(s) are continuation sheets of that BL — they should be classified as "Bill of Lading" with is_continuation=true, even if they have their own header.

BANK HEADER / COVERING PAGES:
- A page showing only a bank's letterhead, logo, address, and SWIFT codes (like OCBC Bank, HSBC, Citibank) WITHOUT any SWIFT F-tag fields (F20:, F31C:, F46A:, :20:, :31C:) is a "Covering Letter" or "Header Page" — NOT an LC.
- An LC page MUST contain SWIFT field tags like F20/F31C/F46A/F47A (Fusion) or :20:/:31C:/:46A: (Alliance) or bare tags like "20: Documentary Credit Number". Just having a bank name and SWIFT code on a page does NOT make it an LC.
- An LC page MUST be type 700 or 701 — if the page header / report identifier shows "fin.799", "fin.999", "Free Format Message", "Message type: 799", or "Bank-to-Bank Message", it is NOT an LC, classify as "MT799" or "MT999" instead. F20 alone is not enough — both LC and MT799 carry F20 (Transaction Reference Number).
- If a page has a bank logo at the top and a table/form below with transaction details but NO SWIFT field tags, it is likely a "Covering Letter", "Export DC Document Presentation Schedule", or "Document Remittance" — NOT an LC.
"""


def _classify_page_vlm(page_num: int, image_path: str, glm_text: str, _max_retries: int = 3) -> dict:
    """Send one page to Qwen for classification with retry logic."""
    if not os.path.exists(image_path):
        return {'page_number': page_num, 'document_type': 'unknown', 'confidence': 0.0,
                'error': 'Image not found'}

    # Resize image to max 1280px to fit within 16K context window.
    # Full-res scans (2560x3600) use 8-10K tokens for the image alone,
    # leaving no room for text+prompt. 1280px is enough for classification.
    try:
        from PIL import Image
        import io
        img = Image.open(image_path)
        _max_dim = 1280
        if img.width > _max_dim or img.height > _max_dim:
            _scale = min(_max_dim / img.width, _max_dim / img.height)
            _new_size = (int(img.width * _scale), int(img.height * _scale))
            img = img.resize(_new_size, Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            img_b64 = base64.b64encode(buf.getvalue()).decode()
        else:
            img_b64 = base64.b64encode(open(image_path, 'rb').read()).decode()
    except Exception:
        img_b64 = base64.b64encode(open(image_path, 'rb').read()).decode()
    # Keep ALL OCR text — never truncate below 4000
    _max_text = 4000
    _truncated_text = glm_text[:_max_text] if len(glm_text) > _max_text else glm_text
    prompt = CLASSIFY_PROMPT.format(glm_text=_truncated_text)
    _current_img_b64 = img_b64  # May be replaced with further resized version on retry
    payload = {
        "model": QWEN_VLM_MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_current_img_b64}"}},
            {"type": "text", "text": prompt}
        ]}],
        "max_tokens": 1500, "temperature": 0.1
    }

    last_err = None
    for attempt in range(_max_retries):
        try:
            resp = requests.post(QWEN_VLM_URL, json=payload, timeout=None)

            if resp.status_code != 200:
                last_err = f'VLM HTTP {resp.status_code}: {resp.text[:200]}'
                print(f"[Step 3] Page {page_num} attempt {attempt+1}: HTTP {resp.status_code}: {resp.text[:200]}")

                # If context overflow or max_tokens error, resize image only
                if ('max_tokens' in resp.text or 'context length' in resp.text or
                    'Input length' in resp.text):
                    try:
                        from PIL import Image
                        import io
                        img = Image.open(image_path)
                        # Reduce to 50% on first retry, 30% on second
                        scale = 0.5 if attempt == 0 else 0.3
                        new_size = (int(img.width * scale), int(img.height * scale))
                        img = img.resize(new_size, Image.LANCZOS)
                        buf = io.BytesIO()
                        img.save(buf, format='PNG')
                        _current_img_b64 = base64.b64encode(buf.getvalue()).decode()
                        payload['messages'][0]['content'][0]['image_url']['url'] = f"data:image/png;base64,{_current_img_b64}"
                        print(f"[Step 3] Page {page_num}: Resized image to {new_size[0]}x{new_size[1]} for retry {attempt+1}")
                    except Exception as _resize_err:
                        print(f"[Step 3] Page {page_num}: Resize failed: {_resize_err}")

                time.sleep(2 * (attempt + 1))
                continue

            result = resp.json()
            content = result.get('choices', [{}])[0].get('message', {}).get('content', '')

            # Strip markdown fences if present
            content = content.strip()
            if content.startswith('```'):
                content = content.split('\n', 1)[1] if '\n' in content else content[3:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()

            # Find JSON in response
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                try:
                    parsed = json.loads(content[json_start:json_end])
                    parsed['page_number'] = page_num
                    return parsed
                except json.JSONDecodeError:
                    pass  # Fall through to truncated JSON handler

            # Truncated JSON fallback — VLM response was cut off but
            # contains valid field values. Extract key fields with regex.
            _dt_m = re.search(r'"document_type"\s*:\s*"([^"]+)"', content)
            _cont_m = re.search(r'"is_continuation"\s*:\s*(true|false)', content, re.IGNORECASE)
            _conf_m = re.search(r'"confidence"\s*:\s*([\d.]+)', content)
            _copy_m = re.search(r'"copy_status"\s*:\s*"([^"]+)"', content)
            _copy_lbl_m = re.search(r'"copy_label"\s*:\s*"([^"]*)"', content)
            _mark_m = re.search(r'"marking_status"\s*:\s*"([^"]+)"', content)
            if _dt_m:
                # Extract stamps and signatures arrays if possible
                _stamps = []
                _sigs = []
                try:
                    _stamps_m = re.search(r'"stamps"\s*:\s*(\[.*?\])', content, re.DOTALL)
                    if _stamps_m:
                        _stamps = json.loads(_stamps_m.group(1))
                except Exception:
                    pass
                try:
                    _sigs_m = re.search(r'"signatures"\s*:\s*(\[.*?\])', content, re.DOTALL)
                    if _sigs_m:
                        _sigs = json.loads(_sigs_m.group(1))
                except Exception:
                    pass
                print(f"[Step 3] Page {page_num}: Recovered from truncated JSON: {_dt_m.group(1)}")
                return {
                    'page_number': page_num,
                    'document_type': _dt_m.group(1),
                    'is_continuation': _cont_m.group(1).lower() == 'true' if _cont_m else False,
                    'confidence': float(_conf_m.group(1)) if _conf_m else 0.90,
                    'copy_status': _copy_m.group(1) if _copy_m else 'unknown',
                    'copy_label': _copy_lbl_m.group(1) if _copy_lbl_m else '',
                    'marking_status': _mark_m.group(1) if _mark_m else 'unknown',
                    'stamps': _stamps,
                    'signatures': _sigs,
                    'seals': [], 'logos': [],
                    'doc_hint': content[:300],
                }
            else:
                return {'page_number': page_num, 'document_type': 'unknown', 'confidence': 0.3,
                        'doc_hint': content[:500]}
        except requests.exceptions.Timeout:
            last_err = 'VLM timeout'
            time.sleep(2 * (attempt + 1))
        except (requests.exceptions.ConnectionError, ConnectionResetError) as e:
            last_err = str(e)
            time.sleep(3 * (attempt + 1))
        except Exception as e:
            last_err = str(e)
            time.sleep(2 * (attempt + 1))

    return {'page_number': page_num, 'document_type': 'unknown', 'confidence': 0.0,
            'error': f'Failed after {_max_retries} retries: {last_err}'}


# ──────────────────────────────────────────────────────────────────────── #
# NEW — per-page sub-call VLM helpers (Tier 1 split)                        #
# Each function sends ONE focused prompt per page. Callers parallelize.     #
# ──────────────────────────────────────────────────────────────────────── #

def _prepare_image_b64(image_path: str, max_dim: int = 1280) -> Optional[str]:
    """Resize image to at most max_dim on either axis; return base64 PNG."""
    if not os.path.exists(image_path):
        return None
    try:
        from PIL import Image
        import io
        img = Image.open(image_path)
        if img.width > max_dim or img.height > max_dim:
            scale = min(max_dim / img.width, max_dim / img.height)
            new_size = (int(img.width * scale), int(img.height * scale))
            img = img.resize(new_size, Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            return base64.b64encode(buf.getvalue()).decode()
        return base64.b64encode(open(image_path, 'rb').read()).decode()
    except Exception:
        try:
            return base64.b64encode(open(image_path, 'rb').read()).decode()
        except Exception:
            return None


def _extract_json_from_response(text: str) -> Optional[dict]:
    """Robust JSON extraction from LLM/VLM response:
    1. Strip markdown code fences (```json ... ```).
    2. Try a direct parse on the stripped body.
    3. Fall back to greedy-brace extraction.
    4. Fall back to 'balance-braces-on-truncation' for responses cut off mid-JSON.
    Returns parsed dict or None.
    """
    if not text:
        return None
    # Strip common markdown code-fence wrappers
    cleaned = text.strip()
    for fence_re in (r'^```(?:json|JSON)?\s*', r'\s*```$'):
        cleaned = re.sub(fence_re, '', cleaned, flags=re.DOTALL).strip()
    # Direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Greedy brace match
    m = re.search(r'\{[\s\S]*\}', cleaned)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    # Balance-braces-on-truncation: response ran out before closing {
    # Count unclosed braces and append them. Also close any unclosed string.
    start = cleaned.find('{')
    if start >= 0:
        partial = cleaned[start:]
        # If we're in the middle of a string, close it
        if partial.count('"') % 2 == 1:
            partial += '"'
        # Trim trailing incomplete value: if last meaningful char is comma, drop it
        partial_stripped = partial.rstrip()
        if partial_stripped.endswith(','):
            partial = partial_stripped[:-1]
        # Close unbalanced braces
        open_braces = partial.count('{') - partial.count('}')
        open_brackets = partial.count('[') - partial.count(']')
        if open_brackets > 0:
            partial += ']' * open_brackets
        if open_braces > 0:
            partial += '}' * open_braces
        try:
            return json.loads(partial)
        except json.JSONDecodeError:
            pass
    return None


def _vlm_call_json(prompt: str, image_b64: Optional[str],
                   max_tokens: int = 1500, temperature: float = 0.1,
                   max_retries: int = 3) -> dict:
    """Send a prompt (+ optional image) to Qwen VLM and parse JSON out of the response."""
    content = [{"type": "text", "text": prompt}]
    if image_b64:
        content.insert(0, {"type": "image_url",
                           "image_url": {"url": f"data:image/png;base64,{image_b64}"}})
    payload = {
        "model": QWEN_VLM_MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(QWEN_VLM_URL, json=payload, timeout=None)
            if resp.status_code != 200:
                last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
                time.sleep(2 * (attempt + 1))
                continue
            text = resp.json().get('choices', [{}])[0].get('message', {}).get('content', '')
            parsed = _extract_json_from_response(text)
            if parsed is not None:
                return parsed
            last_err = f"could not parse JSON: {text[:200]}"
        except requests.exceptions.Timeout:
            last_err = 'VLM timeout'
            time.sleep(2 * (attempt + 1))
        except Exception as e:
            last_err = str(e)
            time.sleep(2 * (attempt + 1))
    return {"_error": last_err or "unknown"}


def _llm_text_call(prompt: str, max_tokens: int = 2500, temperature: float = 0.1,
                   max_retries: int = 3) -> dict:
    """Send a text-only prompt to the Qwen text LLM (used for large-packet summaries)."""
    if not QWEN_TEXT_LLM_URL:
        return {"_error": "QWEN_TEXT_LLM_URL not configured"}
    payload = {
        "model": QWEN_TEXT_LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(QWEN_TEXT_LLM_URL, json=payload, timeout=None)
            if resp.status_code != 200:
                last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
                time.sleep(2 * (attempt + 1))
                continue
            text = resp.json().get('choices', [{}])[0].get('message', {}).get('content', '')
            parsed = _extract_json_from_response(text)
            if parsed is not None:
                return parsed
            # If no JSON found or unparseable, return raw text so caller can inspect
            return {"_raw": text}
        except requests.exceptions.Timeout:
            last_err = 'LLM timeout'
            time.sleep(2 * (attempt + 1))
        except Exception as e:
            last_err = str(e)
            time.sleep(2 * (attempt + 1))
    return {"_error": last_err or "unknown"}


def _classify_doctype_vlm(page_num: int, image_path: str, glm_text: str) -> dict:
    """Step 3a — classify ONE page's document_type + is_continuation.
    Text-dominant task. Sends image at 1280px (classification doesn't need detail)."""
    img_b64 = _prepare_image_b64(image_path, max_dim=1280)
    _text = glm_text[:4000] if len(glm_text) > 4000 else glm_text
    prompt = CLASSIFY_DOCTYPE_PROMPT.format(glm_text=_text)
    result = _vlm_call_json(prompt, img_b64, max_tokens=800)
    # Normalize
    return {
        "page_number": page_num,
        "document_type": result.get("document_type", "unknown"),
        "is_continuation": bool(result.get("is_continuation", False)),
        "confidence": float(result.get("confidence", 0.0) or 0.0),
        "doc_hint": result.get("doc_hint", "") or "",
        "_error": result.get("_error"),
    }


def _extract_markings_vlm(page_num: int, image_path: str) -> dict:
    """Step 3b — extract stamps/signatures/seals/logos from ONE page.
    Visual-only. Uses HIGHER resolution (1920px) so small stamps stay legible."""
    img_b64 = _prepare_image_b64(image_path, max_dim=1920)
    result = _vlm_call_json(EXTRACT_MARKINGS_PROMPT, img_b64, max_tokens=1200)
    _as_list = lambda v: v if isinstance(v, list) else []
    return {
        "page_number": page_num,
        "stamps": _as_list(result.get("stamps")),
        "signatures": _as_list(result.get("signatures")),
        "seals": _as_list(result.get("seals")),
        "logos": _as_list(result.get("logos")),
        "_error": result.get("_error"),
    }


def _detect_copy_status_vlm(page_num: int, image_path: str) -> dict:
    """Step 3c — find ORIGINAL / COPY / NON-NEGOTIABLE stamp and report copy status.
    Narrow visual question. 1600px — balance between legibility and speed."""
    img_b64 = _prepare_image_b64(image_path, max_dim=1600)
    result = _vlm_call_json(COPY_STATUS_PROMPT, img_b64, max_tokens=400)
    return {
        "page_number": page_num,
        "copy_status": result.get("copy_status", "unknown"),
        "copy_label": result.get("copy_label", "") or "",
        "marking_status": result.get("marking_status", "unknown"),
        "_error": result.get("_error"),
    }


# ──────────────────────────────────────────────────────────────────────── #
# NEW — packet-level VLM helpers (Tier 7 + Tier 8)                          #
# Called ONCE per packet after grouping.                                    #
# ──────────────────────────────────────────────────────────────────────── #

def _classify_bl_subtype(packet_text: str, front_image_path: Optional[str] = None,
                         reverse_image_path: Optional[str] = None) -> dict:
    """Step 3d — classify BL sub-type (form/contract/issuer/signing + overleaf/blank-back).
    Sends up to 2 page images (front + reverse) for stamp/signature disambiguation."""
    _text = packet_text[:12000] if len(packet_text) > 12000 else packet_text
    prompt = BL_SUBTYPE_PROMPT.format(packet_text=_text)
    # BL packets are small (1-3 pages typically) → send front + reverse images
    img_b64 = _prepare_image_b64(front_image_path, max_dim=1600) if front_image_path else None
    # For now pass only the front image; reverse-specific signals come from text
    result = _vlm_call_json(prompt, img_b64, max_tokens=800)
    # Normalize shape so caller can trust the fields
    _bool = lambda v: bool(v) if isinstance(v, bool) else (str(v).lower() in ('true', 'yes', '1'))
    return {
        "form_type": result.get("form_type", "unknown"),
        "contract_type": result.get("contract_type", "unknown"),
        "issuer_type": result.get("issuer_type", "unknown"),
        "signing_type": result.get("signing_type", "unknown"),
        "has_terms_overleaf": _bool(result.get("has_terms_overleaf", False)),
        "is_blank_back": _bool(result.get("is_blank_back", False)),
        "carrier_name": result.get("carrier_name", "") or "",
        "forwarder_name": result.get("forwarder_name", "") or "",
        "_error": result.get("_error"),
    }


def _summarize_packet(doc_type: str, page_texts: List[str],
                      page_images: Optional[List[str]] = None) -> dict:
    """Step 3e — structured summary across all pages of a packet.

    Tiered chunking by packet size:
      1-4 pages   → VLM, send ALL images + full text
      5-20 pages  → LLM text-only, full concatenated OCR
      21+ pages   → LLM text-only, chunk by 10 (or 15 for very large),
                    per-chunk summary, then merge summaries with one LLM call
    """
    n = len(page_texts)
    combined = "\n\n--- PAGE BREAK ---\n\n".join(page_texts)

    # Packet summaries can be large: 5 structured arrays + typed fields +
    # BL-specific fields + invented roles. Raised max_tokens to prevent
    # mid-JSON truncation (bug seen with max_tokens=2000: response cut off
    # mid-word on a rich Commercial Invoice, leaving summary unparseable).
    _SUMMARY_MAX_TOKENS = 5000

    # Path A: small packet (≤4 pages) — VLM with images
    if n <= 4:
        _text = combined[:15000] if len(combined) > 15000 else combined
        prompt = PACKET_SUMMARY_PROMPT.format(doc_type=doc_type, packet_text=_text)
        # Use first page image only; text carries the rest
        img_b64 = None
        if page_images:
            for p in page_images:
                if p and os.path.exists(p):
                    img_b64 = _prepare_image_b64(p, max_dim=1280)
                    if img_b64:
                        break
        result = _vlm_call_json(prompt, img_b64, max_tokens=_SUMMARY_MAX_TOKENS)
        return _clean_summary(result)

    # Path B: medium packet (5-20 pages) — text-only LLM
    if n <= 20:
        _text = combined[:40000] if len(combined) > 40000 else combined
        prompt = PACKET_SUMMARY_PROMPT.format(doc_type=doc_type, packet_text=_text)
        result = _llm_text_call(prompt, max_tokens=_SUMMARY_MAX_TOKENS)
        return _clean_summary(result)

    # Path C: large packet (21+ pages) — chunk, summarize each, then merge
    chunk_size = 10 if n <= 40 else 15
    chunk_summaries = []
    for i in range(0, n, chunk_size):
        chunk_pages = page_texts[i:i + chunk_size]
        chunk_text = "\n\n--- PAGE BREAK ---\n\n".join(chunk_pages)
        chunk_text = chunk_text[:30000]
        chunk_prompt = PACKET_SUMMARY_PROMPT.format(
            doc_type=f"{doc_type} (pages {i+1}-{i+len(chunk_pages)} of {n})",
            packet_text=chunk_text,
        )
        chunk_result = _llm_text_call(chunk_prompt, max_tokens=3500)
        chunk_summaries.append(_clean_summary(chunk_result))

    # Merge chunk summaries into a final summary
    merge_prompt = (
        f"You have {len(chunk_summaries)} partial summaries from a single {doc_type} "
        f"document that spans {n} pages. Merge them into ONE unified summary. "
        f"Preserve all identifiers, parties, amounts, and dates. Concatenate goods "
        f"descriptions without duplication. Use the same JSON schema as the partials.\n\n"
        f"PARTIAL SUMMARIES:\n{json.dumps(chunk_summaries, ensure_ascii=False)[:30000]}\n\n"
        f"Return ONLY the merged JSON summary."
    )
    merged = _llm_text_call(merge_prompt, max_tokens=_SUMMARY_MAX_TOKENS)
    return _clean_summary(merged)


def _clean_summary(raw: dict) -> dict:
    """Strip internal error/raw keys and empty values from a summary."""
    if not isinstance(raw, dict):
        return {"_error": f"non-dict summary: {type(raw).__name__}"}
    cleaned = {}
    for k, v in raw.items():
        if k.startswith('_'):
            continue
        if v in (None, "", [], {}):
            continue
        cleaned[k] = v
    # Preserve error if present
    if raw.get("_error"):
        cleaned["_error"] = raw["_error"]
    return cleaned


# ──────────────────────────────────────────────────────────────────────── #
# NEW — Tier 3: LLM-based packet validator                                  #
# Validates packet's COMBINED text against its claimed document_type        #
# using a short LLM text call. NO regex — LLM handles noisy OCR gracefully. #
# ──────────────────────────────────────────────────────────────────────── #

_PACKET_VALIDATOR_PROMPT = """You are validating a trade-finance document classification.

CLAIMED DOCUMENT TYPE: {doc_type}

FULL PACKET TEXT (all pages combined):
{packet_text}

Does the text support the claimed document type?
- YES if content matches (e.g. an LC packet contains SWIFT field tags; a BL contains shipper+consignee+vessel/port; an invoice contains items+amounts).
- NO if the text clearly describes a different document (e.g. claimed LC but only letterhead; claimed BL but only legal clauses from a T&C page).
- UNSURE if the text is too short or ambiguous to tell.

Return ONLY JSON:
{{
  "verdict": "YES|NO|UNSURE",
  "confidence": 0.0-1.0,
  "suggested_type": "if NO, what type would fit better; else empty",
  "reason": "one short sentence"
}}
"""


def _validate_packet_llm(doc_type: str, combined_text: str,
                         max_chars: int = 8000) -> dict:
    """Ask the text LLM whether the combined packet text supports the
    claimed document type. Returns a verdict dict — NOT regex-based.
    Callers decide how strictly to act on 'NO' / 'UNSURE'.
    """
    if not combined_text or not combined_text.strip():
        return {"verdict": "UNSURE", "confidence": 0.0,
                "suggested_type": "", "reason": "empty text"}
    _text = combined_text[:max_chars] if len(combined_text) > max_chars else combined_text
    prompt = _PACKET_VALIDATOR_PROMPT.format(doc_type=doc_type, packet_text=_text)
    result = _llm_text_call(prompt, max_tokens=300)
    # Normalize
    verdict = str(result.get("verdict", "UNSURE")).upper()
    if verdict not in ("YES", "NO", "UNSURE"):
        verdict = "UNSURE"
    return {
        "verdict": verdict,
        "confidence": float(result.get("confidence", 0.0) or 0.0),
        "suggested_type": result.get("suggested_type", "") or "",
        "reason": result.get("reason", "") or "",
        "_error": result.get("_error"),
    }


# ──────────────────────────────────────────────────────────────────────── #
# NEW — Drop-in split classifier: runs the 3 sub-calls in parallel for     #
# one page and merges results into the same shape as legacy classifier.    #
# Callers (run() orchestrator) can swap _classify_page_vlm for this.       #
# ──────────────────────────────────────────────────────────────────────── #

def _classify_page_vlm_split(page_num: int, image_path: str, glm_text: str) -> dict:
    """Run 3 specialized VLM sub-calls in parallel for ONE page and merge."""
    if not os.path.exists(image_path):
        return {'page_number': page_num, 'document_type': 'unknown', 'confidence': 0.0,
                'error': 'Image not found'}

    doctype_result = {}
    markings_result = {}
    copy_result = {}

    with ThreadPoolExecutor(max_workers=3) as pool:
        f_doctype = pool.submit(_classify_doctype_vlm, page_num, image_path, glm_text)
        f_markings = pool.submit(_extract_markings_vlm, page_num, image_path)
        f_copy = pool.submit(_detect_copy_status_vlm, page_num, image_path)
        try:
            doctype_result = f_doctype.result()
        except Exception as e:
            doctype_result = {"_error": f"doctype sub-call failed: {e}"}
        try:
            markings_result = f_markings.result()
        except Exception as e:
            markings_result = {"_error": f"markings sub-call failed: {e}"}
        try:
            copy_result = f_copy.result()
        except Exception as e:
            copy_result = {"_error": f"copy sub-call failed: {e}"}

    # Merge into the legacy dict shape so _group_into_packets keeps working
    merged = {
        "page_number": page_num,
        "document_type": doctype_result.get("document_type", "unknown"),
        "is_continuation": bool(doctype_result.get("is_continuation", False)),
        "confidence": float(doctype_result.get("confidence", 0.0) or 0.0),
        "doc_hint": doctype_result.get("doc_hint", "") or "",
        "stamps": markings_result.get("stamps") or [],
        "signatures": markings_result.get("signatures") or [],
        "seals": markings_result.get("seals") or [],
        "logos": markings_result.get("logos") or [],
        "copy_status": copy_result.get("copy_status", "unknown"),
        "copy_label": copy_result.get("copy_label", "") or "",
        "marking_status": copy_result.get("marking_status", "unknown"),
    }
    # Roll up sub-call errors (if any) into a single error field
    errors = [e for e in (doctype_result.get("_error"),
                          markings_result.get("_error"),
                          copy_result.get("_error")) if e]
    if errors:
        merged["_sub_errors"] = errors
    return merged


# ──────────────────────────────────────────────────────────────────────── #
# NEW — neighbour-context re-classifier (for validation failures)           #
# Sends a page back to the VLM with context from prev + next pages so it   #
# can make a better decision when the first-pass classification was wrong. #
# ──────────────────────────────────────────────────────────────────────── #

_RECHECK_PROMPT = """You are re-classifying a page that likely got the wrong document_type
on the first pass. Use the neighbour-page context to decide correctly.

PAGE IMAGE is attached. Its OCR text is below.

CURRENT PAGE OCR:
{glm_text}

PREVIOUS PAGE TYPE: {prev_type}
NEXT PAGE TYPE:     {next_type}
FIRST-PASS GUESS:   {first_guess}
VALIDATOR VERDICT:  {validator_verdict} — reason: {validator_reason}

Consider: does this page continue the previous doc? Is it a different doc? A
letterhead/cover page? An endorsement page (back of a BL)? A BL Conditions of
Carriage (back of a BL with legal clauses only)?

Return ONLY JSON:
{{
  "document_type": "final decision",
  "is_continuation": false,
  "confidence": 0.0-1.0,
  "doc_hint": "short reason"
}}
"""


def _recheck_page_with_context(page_num: int, image_path: str, glm_text: str,
                               prev_type: str, next_type: str,
                               first_guess: str, validator_verdict: str,
                               validator_reason: str) -> dict:
    """Re-classify a page using neighbour context. Returns same shape as doctype sub-call."""
    img_b64 = _prepare_image_b64(image_path, max_dim=1280)
    _text = glm_text[:3500] if len(glm_text) > 3500 else glm_text
    prompt = _RECHECK_PROMPT.format(
        glm_text=_text, prev_type=prev_type or "(none)",
        next_type=next_type or "(none)", first_guess=first_guess or "unknown",
        validator_verdict=validator_verdict or "UNSURE",
        validator_reason=(validator_reason or "")[:200],
    )
    result = _vlm_call_json(prompt, img_b64, max_tokens=500)
    return {
        "page_number": page_num,
        "document_type": result.get("document_type", first_guess),
        "is_continuation": bool(result.get("is_continuation", False)),
        "confidence": float(result.get("confidence", 0.0) or 0.0),
        "doc_hint": result.get("doc_hint", "") or "",
        "_error": result.get("_error"),
    }


# ──────────────────────────────────────────────────────────────────────── #
# DOC-TYPE SYNONYM TABLE                                                   #
#                                                                          #
# Canonicalizes VLM classification output so synonymous doc types don't    #
# split one physical document into multiple packets. This happens when     #
# the VLM uses different title words for front vs. back of the same doc   #
# (e.g. "Packing Slip" on p1 and "Packing List" on p2), or when a BL is   #
# classified variably as "Bill of Lading" / "Tanker BL" / "CONGENBILL".    #
#                                                                          #
# Keys = canonical name, values = list of synonymous lowercased strings.   #
# Extend this table when new synonym pairs surface in real data.           #
# ──────────────────────────────────────────────────────────────────────── #
_DOC_TYPE_SYNONYMS = {
    'LC': [
        'lc', 'l/c', 'letter of credit', 'documentary credit', 'credit',
        'swift message', 'mt700', 'mt 700', 'fin.700', 'irrevocable letter of credit',
        'irrevocable documentary credit',
    ],
    'Amendment': [
        'amendment', 'lc amendment', 'credit amendment', 'documentary credit amendment',
        'mt707', 'mt 707', 'fin.707',
    ],
    'MT799': [
        'mt799', 'mt 799', 'fin.799',
        'free format message', 'bank-to-bank message', 'bank to bank message',
    ],
    'MT999': ['mt999', 'mt 999', 'fin.999'],
    'Bill of Lading': [
        'bill of lading', 'b/l', 'bl', 'bl no', 'b/l no',
        'tanker bill of lading', 'tanker b/l',
        'combined transport bill of lading', 'combined transport b/l', 'ctbl',
        'through bill of lading', 'through b/l',
        'house bill of lading', 'house b/l', 'hbl',
        'master bill of lading', 'master b/l', 'mbl',
        'charter party bill of lading', 'charter party b/l', 'cpbl',
        'congenbill', 'gencon',
        'ocean bill of lading', 'marine bill of lading',
        'multimodal bill of lading',
    ],
    'BL Conditions of Carriage': [
        'bl conditions of carriage', 'conditions of carriage',
        'bill of lading conditions of carriage', 'terms and conditions of carriage',
    ],
    'Packing List': [
        'packing list', 'packing slip', 'packing memo', 'packing note',
        'list of packages', 'packaging list',
    ],
    'Commercial Invoice': [
        'commercial invoice', 'tax invoice', 'trade invoice',
        'final invoice', 'invoice',
    ],
    'Proforma Invoice': ['proforma invoice', 'pro-forma invoice', 'pro forma invoice'],
    'Draft Bill of Exchange': [
        'draft bill of exchange', 'bill of exchange', 'draft', 'boe', 'draft boe',
        'bills of exchange',
    ],
    'Certificate of Origin': [
        'certificate of origin', 'coo', 'c/o', 'origin certificate',
        'chamber of commerce certificate of origin', 'gsp form a',
    ],
    'Phytosanitary Certificate': [
        'phytosanitary certificate', 'plant health certificate', 'phyto certificate',
        'phytosanitary',
    ],
    'Health Certificate': [
        'health certificate', 'veterinary health certificate',
        'food safety certificate', 'sanitary certificate',
    ],
    'Halal Certificate': ['halal certificate', 'halal', 'halal certification'],
    'Fumigation Certificate': ['fumigation certificate', 'fumigation'],
    'Weight Certificate': [
        'weight certificate', 'certificate of weight', 'weighing certificate',
        'weighbridge certificate',
    ],
    'Weight / Quality Certificate': [
        'weight / quality certificate', 'weight/quality certificate',
        'weight and quality certificate',
    ],
    'Quality Certificate': [
        'quality certificate', 'quality analysis', 'quality / analysis',
        'quality and analysis', 'products quality certificate',
        'certificate of quality', 'quality report',
    ],
    'Quantity Certificate': [
        'quantity certificate', 'certificate of quantity',
        'products quantity certificate', 'certificate of receipted quantity',
    ],
    'Inspection Certificate': [
        'inspection certificate', 'pre-shipment inspection certificate',
        'pre shipment inspection certificate', 'psi certificate',
    ],
    'Insurance Certificate': [
        'insurance certificate', 'marine insurance certificate',
        'cargo insurance certificate',
    ],
    'Insurance Policy': [
        'insurance policy', 'marine insurance policy', 'cargo insurance policy',
    ],
    'Shipment Advice': [
        'shipment advice', 'advice of shipment', 'cargo advice',
        'shipping advice',
    ],
    'Document Remittance': [
        'document remittance', 'documentary remittance',
        'l/c bills schedule', 'lc bills schedule', 'covering schedule',
        'document presentation', 'export dc document presentation schedule',
        'export documentary credit document presentation schedule',
    ],
    'Covering Letter': ['covering letter', 'cover letter', 'transmittal letter'],
    'Letter of Indemnity': ['letter of indemnity', 'loi', 'indemnity letter'],
    'Letter of Authority': ['letter of authority', 'authority letter'],
    'Notice of Readiness': ['notice of readiness', 'nor'],
    'Endorsement Page': ['endorsement page', 'endorsement', 'bl endorsement'],
    'Header Page': ['header page', 'header'],
    'Beneficiary Certificate': ['beneficiary certificate', 'beneficiarys certificate',
                                 "beneficiary's certificate"],
    'Port Clearance Certificate': ['port clearance certificate', 'port clearance'],
    'Tanker Cleanliness Certificate': ['tanker cleanliness certificate',
                                        'tank cleanliness certificate'],
    'Survey Report': [
        'survey report', 'draught survey report', 'loading survey report',
        'discharge survey report', 'full loading survey report',
    ],
    'Agents Certificate': ['agents certificate', "agent's certificate",
                           'shipping agents certificate', 'ships agents certificate'],
}


def _canonical_doc_type(doc_type: str) -> str:
    """Map a VLM-returned doc_type string to its canonical name.
    If no synonym matches, returns the original doc_type unchanged (so
    unknown/new types are preserved verbatim).
    """
    if not doc_type:
        return doc_type
    key = doc_type.lower().strip()
    # Strip parenthesized qualifiers like "(Original)" that sometimes slip in
    if '(' in key:
        key = key.split('(')[0].strip()
    for canonical, synonyms in _DOC_TYPE_SYNONYMS.items():
        if key == canonical.lower():
            return canonical
        if key in synonyms:
            return canonical
    return doc_type


def _group_into_packets(classifications: List[dict]) -> List[DocumentPacket]:
    """Group classified pages into document packets."""
    if not classifications:
        return []

    packets = []
    current_packet = None

    for cls in sorted(classifications, key=lambda c: c.get('page_number', 0)):
        pg_num = cls.get('page_number', 0)
        doc_type = cls.get('document_type', 'unknown')
        is_cont = cls.get('is_continuation', False)
        confidence = cls.get('confidence', 0.0)

        # Normalize doc type using the comprehensive synonym table.
        # Keeps alternating VLM outputs ("Packing Slip"/"Packing List",
        # "CONGENBILL"/"Bill of Lading", etc.) from splitting one physical
        # document into multiple packets.
        doc_type = _canonical_doc_type(doc_type)
        doc_type_lower = doc_type.lower().strip()

        # Continuation: only merge if document type AND copy status match
        # (prevents different BL copies from being merged into one packet)
        _copy_status = cls.get('copy_status', 'unknown')
        # Type matching: exact match OR both types are survey report aliases
        _SURVEY_TYPE_ALIASES = {
            'quality / analysis', 'quality/analysis', 'quality analysis',
            'quality certificate', 'products quality certificate',
            'loading inspection report', 'full loading survey report',
            'survey report', 'inspection report', 'inspection certificate',
            'pre-shipment inspection report', 'draught survey report',
            'loading report', 'discharge report',
        }
        _curr_lower = doc_type.lower().strip()
        _pkt_lower = current_packet.document_type.lower().strip() if current_packet else ''
        _type_matches = (current_packet and (
            _curr_lower == _pkt_lower or
            (_curr_lower in _SURVEY_TYPE_ALIASES and _pkt_lower in _SURVEY_TYPE_ALIASES)
        ))
        _copy_matches = (current_packet and
                         (_copy_status == current_packet.copy_status or
                          _copy_status in ('unknown', '') or
                          current_packet.copy_status in ('unknown', '')))
        if is_cont and current_packet and _type_matches and _copy_matches:
            # Same type + same copy status — merge into current packet
            current_packet.page_numbers.append(pg_num)
            current_packet.pages.append(cls)
            current_packet.stamps.extend(cls.get('stamps', []))
            current_packet.signatures.extend(cls.get('signatures', []))
            current_packet.seals.extend(cls.get('seals', []))
            current_packet.logos.extend(cls.get('logos', []))
        else:
            # New document — start new packet
            if current_packet:
                packets.append(current_packet)

            pkt_id = f"pkt_{len(packets)+1}"
            copy_status = cls.get('copy_status', 'unknown')
            if copy_status == 'unknown':
                copy_status = 'original'

            current_packet = DocumentPacket(
                packet_id=pkt_id,
                document_type=doc_type,
                pages=[cls],
                page_numbers=[pg_num],
                boundary_confidence=confidence,
                copy_status=copy_status,
                copy_label=cls.get('copy_label', ''),
                marking_status=cls.get('marking_status', 'unsigned'),
                stamps=cls.get('stamps', []) if isinstance(cls.get('stamps'), list) else [],
                signatures=cls.get('signatures', []) if isinstance(cls.get('signatures'), list) else [],
                seals=cls.get('seals', []) if isinstance(cls.get('seals'), list) else [],
                logos=cls.get('logos', []) if isinstance(cls.get('logos'), list) else [],
                doc_hint=cls.get('doc_hint', ''),
            )

    # Don't forget the last packet
    if current_packet:
        packets.append(current_packet)

    return packets


def run(step2_result: dict, output_dir: str = None, progress_callback=None) -> dict:
    """
    Execute Step 3: Classify every page with Qwen, then group into packets.

    Args:
        step2_result: Output from Step 2 (pages with cleaned_text + page_image_path)
        output_dir: Directory to save results
        progress_callback: Optional callback for progress

    Returns:
        dict with 'packets' (list of DocumentPacket), 'classifications', 'elapsed_seconds'
    """
    def _progress(msg):
        if progress_callback:
            progress_callback(f"[Step 3] {msg}")
        print(f"[Step 3] {msg}")

    start_time = time.time()
    pages = step2_result.get('pages', [])
    _progress(f"Classifying {len(pages)} pages with Qwen VLM...")

    # ── Phase 0: SWIFT pre-classification (code-based, 100% accurate) ──
    # Detect LC/Amendment/MT799/MT999 pages from OCR text BEFORE sending to VLM.
    # Also detect "Page X of Y" for Fusion multi-page document grouping.
    _swift_preclassified = {}  # page_number -> classification dict

    _SWIFT_LC_PATTERNS = [
        r'Message\s+type:\s*700',
        r'SWIFT_MT700',
        r'(?:^|\n)\s*:20:',              # Alliance F20
        r'(?:^|\n)\s*F20\s*:',           # Fusion F20
        r'(?:^|\n)\s*:46A:',             # Alliance F46A
        r'(?:^|\n)\s*F46A\s*:',          # Fusion F46A
        r'(?:^|\n)\s*:31C:',             # Alliance date of issue
        r'(?:^|\n)\s*F31C\s*:',          # Fusion date of issue
        r'(?:^|\n)\s*20:\s*Documentary\s+Credit\s+Number',  # Fusion long form
        r'(?:^|\n)\s*40A:\s*Form\s+of\s+Documentary\s+Credit',
        r'Sender\'?s?\s+Reference\s*\n\s*[A-Z0-9]{5,}',  # Fusion Sender's Reference with value
    ]
    _SWIFT_LC_CONT_PATTERNS = [
        # MT701 is continuation of MT700 (additional LC pages)
        r'Message\s+type:\s*701',
        r'SWIFT_MT701',
    ]
    _SWIFT_AMEND_PATTERNS = [
        r'Message\s+type:\s*707',
        r'SWIFT_MT707',
        r'(?:^|\n)\s*26E:\s*Number\s+of\s+Amendment',
        r'(?:^|\n)\s*26E:',              # Alliance amendment number
        r'26E:\s*\d+',                   # 26E with number
        r'Number\s+of\s+Amendment',
        r'Date\s+of\s+Amendment',
        r'Increase\s+of\s+Documentary\s+Credit',
        r'Decrease\s+of\s+Documentary\s+Credit',
    ]
    # MT799 is a free-format SWIFT message. Alliance Message Management
    # reports show it as `Identifier: fin.799` / `Expansion: Free Format
    # Message` — NOT as "Message type: 799". The narrative body field is
    # F79 / :79:. None of these were caught by the original two-pattern
    # list, so 799 pages with F20 (Transaction Reference) were getting
    # misclassified as LC because is_lc fired before is_799.
    _SWIFT_799_PATTERNS = [
        r'Message\s+type:\s*7?99',
        r'\bSWIFT[_ ]?MT[_ ]?7?99\b',
        r'\bMT\s*7?99\b',
        r'\bfin\.\s*7?99\b',
        r'\bIdentifier\s*:\s*fin\.\s*7?99\b',
        r'\bFREE\s+FORMAT\s+MESSAGE\b',
        r'\bBANK[- ]TO[- ]BANK\s+MESSAGE\b',
        r'(?:^|\n)\s*F79\s*:',          # MT799 narrative field (Fusion)
        r'(?:^|\n)\s*:79:',             # MT799 narrative field (Alliance)
    ]
    _SWIFT_999_PATTERNS = [
        r'Message\s+type:\s*999',
        r'\bSWIFT[_ ]?MT[_ ]?999\b',
        r'\bMT\s*999\b',
        r'\bfin\.\s*999\b',
        r'\bIdentifier\s*:\s*fin\.\s*999\b',
    ]
    _SWIFT_CONTINUATION_PATTERNS = [
        r'(?:^|\n)\s*(?::|\bF)45A[\s:]+',  # Description of goods
        r'(?:^|\n)\s*(?::|\bF)45B[\s:]+',  # Description of goods contd
        r'(?:^|\n)\s*(?::|\bF)46A[\s:]+',  # Documents Required
        r'(?:^|\n)\s*(?::|\bF)46B[\s:]+',  # Documents Required contd
        r'(?:^|\n)\s*(?::|\bF)47A[\s:]+',  # Additional Conditions
        r'(?:^|\n)\s*(?::|\bF)47B[\s:]+',  # Additional Conditions contd
        r'(?:^|\n)\s*(?::|\bF)78[\s:]+',   # Instructions
        r'(?:^|\n)\s*(?::|\bF)72[\s:]+',   # Sender to Receiver
        r'(?:^|\n)\s*(?::|\bF)49[\s:]+',   # Confirmation Instructions
        r'(?:^|\n)\s*(?::|\bF)71[BD][\s:]+',  # Charges
    ]
    _FUSION_HEADER_PATTERNS = [
        r'FUSION\s+TRADE\s+INNOVATION',
        r'Formatted\s+Outward\s+SWIFT\s+message',
        r'Select\s+.?Print.?\s+to\s+output',
        r'SwiftOutViewWP\.jsf',
    ]

    all_page_data = []
    for page in pages:
        if hasattr(page, 'page_number'):
            pg_num = page.page_number
            text = page.cleaned_text or page.raw_text
            img_path = page.page_image_path
        else:
            pg_num = page.get('page_number', 0)
            text = page.get('cleaned_text', page.get('raw_text', ''))
            img_path = page.get('page_image_path', '')
        all_page_data.append((pg_num, img_path, text))

    # Sort by page number
    all_page_data.sort(key=lambda x: x[0])

    # ── Step 0a: Detect "Page X of Y" on each page for Fusion grouping ──
    _page_of_total = {}  # page_number -> (x, y) where "Page X of Y"
    for pg_num, _, text in all_page_data:
        m = re.search(r'Page\s+(\d+)\s+of\s+(\d+)', text or '', re.IGNORECASE)
        if m:
            _page_of_total[pg_num] = (int(m.group(1)), int(m.group(2)))

    # ── Step 0a-bis: BAHL multi-message report detection ──
    # BAHL (Bank Al Habib) PDFs bundle multiple SWIFT messages (MT700, MT707,
    # MT799, MT999, MT754, MT940, MT730, MT747, MT740) into a single PDF with
    # "Page X of 43" report pagination. Each message is separated by
    # "Message Details #N" headers with its own "Identifier: fin.XXX".
    # If we detect this format, we split by message boundary instead of
    # "Page X of Y" grouping (which would lump all 43 pages as one MT799).
    _BAHL_MSG_DETAIL_RE = re.compile(r'Message\s+Details\s+#\s*(\d+)', re.IGNORECASE)
    _BAHL_IDENTIFIER_RE = re.compile(r'Identifier\s*:\s*fin\.(\d{3})', re.IGNORECASE)
    _BAHL_FIN_TO_MT = {
        '700': 'LC', '701': 'LC', '705': 'LC',
        '707': 'Amendment', '708': 'Amendment',
        '747': 'Amendment',  # Amendment to Auth to Reimburse
        '799': 'MT799', '999': 'MT999',
        '754': 'MT754', '940': 'MT940', '730': 'MT730',
        '740': 'MT740', '742': 'MT742',
        '734': 'MT734', '750': 'MT750', '752': 'MT752',
    }

    _is_bahl = False
    _bahl_messages = {}  # msg_number -> {'pages': [int], 'mt_type': str, 'fin': str}

    # Scan for "Message Details #N" on each page
    _msg_detail_pages = {}  # page_number -> [msg_numbers found on this page]
    for pg_num, _, text in all_page_data:
        if not text:
            continue
        for m in _BAHL_MSG_DETAIL_RE.finditer(text):
            msg_num = int(m.group(1))
            if pg_num not in _msg_detail_pages:
                _msg_detail_pages[pg_num] = []
            _msg_detail_pages[pg_num].append(msg_num)

    # If 3+ pages have "Message Details #N", this is a BAHL multi-message report
    if len(_msg_detail_pages) >= 3:
        _is_bahl = True
        _progress(f"  BAHL multi-message report detected: {len(_msg_detail_pages)} message headers found")

        # Build message boundaries: assign each page to a message
        # Sort pages, then assign: page belongs to the most recent Message Details #N
        sorted_pages_list = sorted(all_page_data, key=lambda x: x[0])
        # Find the max page number that has a "Page X of Y" matching the
        # BAHL report pagination. Pages beyond this are shipping docs.
        _bahl_max_page = 0
        for pg_num in sorted(_msg_detail_pages.keys()):
            if pg_num in _page_of_total:
                _x, _y = _page_of_total[pg_num]
                # The report total (e.g., "Page 1 of 7") tells us the last page
                _bahl_max_page = max(_bahl_max_page, pg_num + (_y - _x))

        current_msg_num = None
        for pg_num, _, text in sorted_pages_list:
            # Stop assigning to BAHL messages once we pass the report boundary
            if _bahl_max_page > 0 and pg_num > _bahl_max_page:
                break

            # Check if this page starts a new message
            if pg_num in _msg_detail_pages:
                new_msgs = sorted(_msg_detail_pages[pg_num])
                for nm in new_msgs:
                    if nm not in _bahl_messages:
                        _bahl_messages[nm] = {'pages': [], 'mt_type': '', 'fin': ''}
                current_msg_num = new_msgs[-1]

            if current_msg_num is not None:
                if current_msg_num in _bahl_messages:
                    _bahl_messages[current_msg_num]['pages'].append(pg_num)

        # Extract fin.XXX identifier for each message
        for msg_num, msg_info in _bahl_messages.items():
            for pg_num in msg_info['pages']:
                text = next((t for pn, _, t in all_page_data if pn == pg_num), '')
                if text:
                    m = _BAHL_IDENTIFIER_RE.search(text)
                    if m:
                        fin_num = m.group(1)
                        msg_info['fin'] = fin_num
                        msg_info['mt_type'] = _BAHL_FIN_TO_MT.get(fin_num, f'MT{fin_num}')
                        break

        # Log BAHL messages
        for msg_num in sorted(_bahl_messages.keys()):
            mi = _bahl_messages[msg_num]
            _progress(f"    Message #{msg_num}: fin.{mi['fin'] or '?'} = {mi['mt_type'] or '?'}, pages {mi['pages']}")

        # SUPPRESS normal "Page X of Y" grouping for BAHL pages
        # The report-level "Page X of 43" would incorrectly group everything
        _bahl_page_set = set()
        for mi in _bahl_messages.values():
            _bahl_page_set.update(mi['pages'])
        _page_of_total = {pg: xy for pg, xy in _page_of_total.items()
                          if pg not in _bahl_page_set}

    # ── Step 0b: First pass — detect SWIFT message starts and Fusion headers ──
    _page_swift_type = {}  # page_number -> 'LC'|'Amendment'|'MT799'|'MT999'|'fusion_header'

    # If BAHL detected, pre-populate _page_swift_type from message boundaries
    if _is_bahl:
        for msg_num, mi in _bahl_messages.items():
            mt = mi['mt_type']
            if mt:
                for i, pg in enumerate(mi['pages']):
                    _page_swift_type[pg] = mt

    for pg_num, _, text in all_page_data:
        if not text:
            continue

        # Skip pages already classified by BAHL multi-message splitter
        if _is_bahl and pg_num in _page_swift_type:
            continue

        is_amendment = any(re.search(p, text, re.IGNORECASE) for p in _SWIFT_AMEND_PATTERNS)
        is_lc = any(re.search(p, text, re.IGNORECASE) for p in _SWIFT_LC_PATTERNS)
        is_lc_cont = any(re.search(p, text, re.IGNORECASE) for p in _SWIFT_LC_CONT_PATTERNS)
        is_799 = any(re.search(p, text, re.IGNORECASE) for p in _SWIFT_799_PATTERNS)
        is_999 = any(re.search(p, text, re.IGNORECASE) for p in _SWIFT_999_PATTERNS)

        # P130 — If page looks like a presenting-bank cover letter
        # (Documentary Remittance / Document Presentation), cancel the
        # MT799/MT999 detection — "MT799" appearing as a ROW in the
        # attachment table is not the page's type.
        _is_cover_letter = (
            re.search(r'\bDocument\s+Presentation\b', text, re.IGNORECASE) or
            re.search(r'\bBill\s+Remittance\s+Letter\b', text, re.IGNORECASE) or
            re.search(r'\bCovering\s+Schedule\b', text, re.IGNORECASE) or
            re.search(r'\bL/?C\s+Bills\s+Schedule\b', text, re.IGNORECASE) or
            re.search(r'\bExport\s+DC\s+Document\s+Presentation\s+Schedule\b', text, re.IGNORECASE) or
            re.search(r'\bPlease\s+acknowledge\s+receipt\s+for\s+the\s+documents?\s+enclosed\b', text, re.IGNORECASE) or
            re.search(r'\b1st\s+mail\b.{0,20}\b2nd\s+mail\b', text, re.IGNORECASE | re.DOTALL) or
            re.search(r'\bSettlement\s+instructions\s+for\s+this\s+document\s+set\b', text, re.IGNORECASE)
        )
        if _is_cover_letter:
            # Force: this is a Document Remittance, not MT799/MT999.
            is_799 = False
            is_999 = False
            _progress(f"  Page {pg_num}: overriding SWIFT MT detection — cover letter signals present (Document Remittance)")
        is_swift_cont = any(re.search(p, text, re.IGNORECASE) for p in _SWIFT_CONTINUATION_PATTERNS)
        is_fusion_header = any(re.search(p, text, re.IGNORECASE) for p in _FUSION_HEADER_PATTERNS)

        # PRIORITY ORDER MATTERS: a free-format MT799 page often references
        # field tags (F20, F45A, etc.) in its body, which would otherwise
        # match _SWIFT_LC_PATTERNS and get the page misclassified as LC.
        # So MT799/MT999 must be checked BEFORE the LC patterns.
        # Same for Amendment vs LC: an MT707 amendment also has F20.
        if is_799:
            _page_swift_type[pg_num] = 'MT799'
        elif is_999:
            _page_swift_type[pg_num] = 'MT999'
        # MT701 is LC continuation (not a new LC)
        elif is_lc_cont:
            _page_swift_type[pg_num] = 'LC'  # Treat as LC, will be marked continuation later
        elif is_amendment:
            _page_swift_type[pg_num] = 'Amendment'
        elif is_lc:
            _page_swift_type[pg_num] = 'LC'
        elif is_swift_cont:
            _page_swift_type[pg_num] = '_swift_continuation'
        elif is_fusion_header:
            _page_swift_type[pg_num] = '_fusion_header'

    # ── Step 0c: Second pass — resolve continuations and Fusion headers using "Page X of Y" ──
    # Group pages that share the same "of Y" total into document groups
    # Find the SWIFT type of each group from the first page that has a Message type
    _doc_groups = {}  # group_key -> {'type': str, 'pages': [int], 'start': int}

    # Build groups from Page X of Y
    if _page_of_total:
        # Group consecutive pages with same total
        sorted_pages = sorted(all_page_data, key=lambda x: x[0])
        current_group = None
        for pg_num, _, text in sorted_pages:
            if pg_num in _page_of_total:
                x, y = _page_of_total[pg_num]
                if x == 1:
                    # Start of a new document group
                    current_group = {'pages': [pg_num], 'total': y, 'type': None, 'start_pg': pg_num}
                    _doc_groups[pg_num] = current_group
                elif current_group and current_group['total'] == y:
                    current_group['pages'].append(pg_num)
                else:
                    # Orphan page — create its own group
                    current_group = {'pages': [pg_num], 'total': y, 'type': None, 'start_pg': pg_num}
                    _doc_groups[pg_num] = current_group

        # Assign types to groups from any page that has a SWIFT type
        for start_pg, group in _doc_groups.items():
            for pg in group['pages']:
                st = _page_swift_type.get(pg, '')
                if st and not st.startswith('_'):
                    group['type'] = st
                    break

    # ── Step 0d: Build final pre-classification ──
    prev_swift_type = None
    for pg_num, img_path, text in all_page_data:
        # Check if this page belongs to a Fusion document group
        _in_group = None
        for start_pg, group in _doc_groups.items():
            if pg_num in group['pages'] and group['type']:
                _in_group = group
                break

        if _in_group:
            # This page belongs to a known Fusion document group
            doc_type = _in_group['type']
            is_first = (pg_num == _in_group['pages'][0])
            is_cont = not is_first
            _swift_preclassified[pg_num] = {
                'page_number': pg_num, 'document_type': doc_type,
                'is_continuation': is_cont, 'confidence': 0.99,
                'stamps': [], 'signatures': [], 'seals': [], 'logos': [],
                'copy_status': 'original', 'copy_label': '', 'marking_status': 'unsigned',
                'doc_hint': f'Fusion {doc_type} (Page {_page_of_total.get(pg_num, ("?","?"))[0]} of {_page_of_total.get(pg_num, ("?","?"))[1]})',
            }
            prev_swift_type = doc_type
            _progress(f"  Page {pg_num}: PRE-CLASSIFIED as {doc_type}{' (cont)' if is_cont else ''} [Page {_page_of_total.get(pg_num, ('?','?'))[0]} of {_page_of_total.get(pg_num, ('?','?'))[1]}]")
            continue

        # Not in a Fusion group — use direct SWIFT pattern detection
        st = _page_swift_type.get(pg_num, '')

        if st == 'Amendment':
            _swift_preclassified[pg_num] = {
                'page_number': pg_num, 'document_type': 'Amendment',
                'is_continuation': False, 'confidence': 0.99,
                'stamps': [], 'signatures': [], 'seals': [], 'logos': [],
                'copy_status': 'original', 'copy_label': '', 'marking_status': 'unsigned',
                'doc_hint': 'SWIFT MT707 Amendment detected from text patterns',
            }
            prev_swift_type = 'Amendment'
            _progress(f"  Page {pg_num}: PRE-CLASSIFIED as Amendment (SWIFT pattern)")
        elif st == 'LC':
            _swift_preclassified[pg_num] = {
                'page_number': pg_num, 'document_type': 'LC',
                'is_continuation': False, 'confidence': 0.99,
                'stamps': [], 'signatures': [], 'seals': [], 'logos': [],
                'copy_status': 'original', 'copy_label': '', 'marking_status': 'unsigned',
                'doc_hint': 'SWIFT MT700 LC detected from text patterns',
            }
            prev_swift_type = 'LC'
            _progress(f"  Page {pg_num}: PRE-CLASSIFIED as LC (SWIFT pattern)")
        elif st == 'MT799':
            _swift_preclassified[pg_num] = {
                'page_number': pg_num, 'document_type': 'MT799',
                'is_continuation': False, 'confidence': 0.99,
                'stamps': [], 'signatures': [], 'seals': [], 'logos': [],
                'copy_status': 'original', 'copy_label': '', 'marking_status': 'unsigned',
                'doc_hint': 'SWIFT MT799 Free Format Message',
            }
            prev_swift_type = 'MT799'
            _progress(f"  Page {pg_num}: PRE-CLASSIFIED as MT799")
        elif st == 'MT999':
            _swift_preclassified[pg_num] = {
                'page_number': pg_num, 'document_type': 'MT999',
                'is_continuation': False, 'confidence': 0.99,
                'stamps': [], 'signatures': [], 'seals': [], 'logos': [],
                'copy_status': 'original', 'copy_label': '', 'marking_status': 'unsigned',
                'doc_hint': 'SWIFT MT999 Free Format Message',
            }
            prev_swift_type = 'MT999'
            _progress(f"  Page {pg_num}: PRE-CLASSIFIED as MT999")
        elif st.startswith('MT') and st not in ('MT799', 'MT999'):
            # BAHL informational MT types: MT730, MT754, MT940, MT740, MT747, etc.
            _swift_preclassified[pg_num] = {
                'page_number': pg_num, 'document_type': st,
                'is_continuation': False, 'confidence': 0.99,
                'stamps': [], 'signatures': [], 'seals': [], 'logos': [],
                'copy_status': 'original', 'copy_label': '', 'marking_status': 'unsigned',
                'doc_hint': f'SWIFT {st} (BAHL multi-message)',
            }
            prev_swift_type = st
            _progress(f"  Page {pg_num}: PRE-CLASSIFIED as {st}")
        elif st == '_swift_continuation' and prev_swift_type:
            _swift_preclassified[pg_num] = {
                'page_number': pg_num, 'document_type': prev_swift_type,
                'is_continuation': True, 'confidence': 0.95,
                'stamps': [], 'signatures': [], 'seals': [], 'logos': [],
                'copy_status': 'original', 'copy_label': '', 'marking_status': 'unsigned',
                'doc_hint': f'SWIFT continuation of {prev_swift_type} (F-tags detected)',
            }
            _progress(f"  Page {pg_num}: PRE-CLASSIFIED as {prev_swift_type} continuation")
        elif st == '_fusion_header' and prev_swift_type:
            # Fusion header page between SWIFT content — belongs to the same document
            _swift_preclassified[pg_num] = {
                'page_number': pg_num, 'document_type': prev_swift_type,
                'is_continuation': True, 'confidence': 0.90,
                'stamps': [], 'signatures': [], 'seals': [], 'logos': [],
                'copy_status': 'original', 'copy_label': '', 'marking_status': 'unsigned',
                'doc_hint': f'Fusion header page (part of {prev_swift_type})',
            }
            _progress(f"  Page {pg_num}: PRE-CLASSIFIED as {prev_swift_type} header page")
        else:
            prev_swift_type = None  # Reset — not SWIFT content

    _progress(f"  Pre-classified {len(_swift_preclassified)} pages as SWIFT (LC/Amendment/MT799/MT999)")

    # ── Phase 1: ALL pages go to VLM for classification + stamp/signature detection ──
    # SWIFT pre-classification is used to OVERRIDE VLM's document_type if VLM gets it wrong,
    # but VLM still runs on every page to extract stamps, signatures, seals, logos, copy status.
    classifications = []
    vlm_tasks = list(all_page_data)

    _progress(f"[3a+3b+3c] Sending ALL {len(vlm_tasks)} pages to VLM for 3-way per-page split...")
    _progress(f"  3a=Doc Type | 3b=Markings & Seals | 3c=Copy/Original Status (parallel per page)")

    # Run VLM classification concurrently. Each page triggers 3 sub-calls
    # (doc-type, markings, copy-status) in its own mini-pool, so we cap the
    # outer pool lower to avoid overwhelming the VLM server.
    _outer_workers = max(1, MAX_CONCURRENT_VLM // 2)
    with ThreadPoolExecutor(max_workers=_outer_workers) as executor:
        futures = {}
        for pg_num, img_path, text in vlm_tasks:
            future = executor.submit(_classify_page_vlm_split, pg_num, img_path, text)
            futures[future] = pg_num

        done_count = 0
        for future in as_completed(futures):
            pg_num = futures[future]
            try:
                result = future.result()

                # If this page was pre-classified as SWIFT, override VLM's document_type
                # but KEEP VLM's stamps, signatures, seals, logos, copy_status
                if pg_num in _swift_preclassified:
                    pre = _swift_preclassified[pg_num]
                    vlm_doc_type = result.get('document_type', 'unknown')
                    # Override document_type and is_continuation from SWIFT detection
                    result['document_type'] = pre['document_type']
                    result['is_continuation'] = pre['is_continuation']
                    result['confidence'] = max(result.get('confidence', 0), pre['confidence'])
                    if vlm_doc_type.lower() in ('blank page', 'blank_page', 'unknown'):
                        result['doc_hint'] = f"SWIFT {pre['document_type']} (VLM said '{vlm_doc_type}' — overridden)"
                        _progress(f"  Page {pg_num}: OVERRIDE {vlm_doc_type}→{pre['document_type']} (SWIFT pattern)")

                classifications.append(result)
                doc_type = result.get('document_type', '?')
                conf = result.get('confidence', 0)
                is_cont = result.get('is_continuation', False)
                stamps_count = len(result.get('stamps', []))
                sigs_count = len(result.get('signatures', []))
                copy_st = result.get('copy_status', '?')

                done_count += 1
                _progress(f"  Page {pg_num}: {doc_type} (conf={conf:.2f}, cont={is_cont}, "
                          f"stamps={stamps_count}, sigs={sigs_count}, copy={copy_st}) "
                          f"[{done_count}/{len(vlm_tasks)}]")

            except Exception as e:
                _progress(f"  Page {pg_num}: ERROR - {e}")
                classifications.append({
                    'page_number': pg_num, 'document_type': 'unknown',
                    'confidence': 0.0, 'error': str(e)
                })

    # Sort by page number
    classifications.sort(key=lambda c: c.get('page_number', 0))

    # ── Phase 1a: Copy status detection from OCR text ──
    # The VLM is the SOLE classifier for document type — we trust it completely.
    # We only use OCR text to detect copy status (ORIGINAL/NON-NEGOTIABLE/COPY)
    # when the VLM missed the stamp marking.

    import re as _re_cls

    for cls in classifications:
        pg_num = cls.get('page_number', 0)
        if pg_num in _swift_preclassified:
            continue

        # Get OCR text for this page
        _pg_text = ''
        for _pn, _ip, _tx in all_page_data:
            if _pn == pg_num:
                _pg_text = (_tx or '').upper()
                break
        if not _pg_text:
            continue

        # Copy status from OCR text (if VLM missed it)
        _copy = cls.get('copy_status', '')
        if not _copy or _copy in ('unknown', ''):
            if 'NON-NEGOTIABLE' in _pg_text or 'NON NEGOTIABLE' in _pg_text:
                cls['copy_status'] = 'non_negotiable'
                cls['copy_label'] = 'NON-NEGOTIABLE'
            elif _re_cls.search(r'(?<!\w)(FIRST|SECOND|THIRD)\s+ORIGINAL(?!\w)', _pg_text):
                cls['copy_status'] = 'original'
                _fm = _re_cls.search(r'(FIRST|SECOND|THIRD)\s+ORIGINAL', _pg_text)
                cls['copy_label'] = _fm.group(0) if _fm else 'ORIGINAL'
            elif _re_cls.search(r'(?<!\w)ORIGINAL(?!\w)', _pg_text) and _pg_text.count('ORIGINAL') <= 4:
                cls['copy_status'] = 'original'
                cls['copy_label'] = 'ORIGINAL'
            elif _re_cls.search(r'(?<!\w)COPY(?!\s+OF)(?!\s+NO)', _pg_text):
                cls['copy_status'] = 'copy'
                cls['copy_label'] = 'COPY'

    # ── Phase 1b: Multi-page document detection via content similarity ──
    # If two consecutive pages have the SAME document type but DIFFERENT content,
    # the second is a continuation (e.g., BL front + cargo details).
    # If they have SAME content, they're separate copies (e.g., 3 original BLs).
    for i in range(1, len(classifications)):
        curr = classifications[i]
        prev = classifications[i - 1]
        curr_type = (curr.get('document_type', '') or '').lower()
        prev_type = (prev.get('document_type', '') or '').lower()

        # Only check same document types, skip SWIFT pages
        if curr_type != prev_type or not curr_type:
            continue
        if curr.get('page_number', 0) in _swift_preclassified:
            continue
        if curr.get('is_continuation', False):
            continue  # Already marked as continuation

        # Get OCR text for both pages
        prev_text = ''
        curr_text = ''
        for _pn, _ip, _tx in all_page_data:
            if _pn == prev.get('page_number', 0):
                prev_text = (_tx or '').upper()[:500]
            if _pn == curr.get('page_number', 0):
                curr_text = (_tx or '').upper()[:500]

        if not prev_text or not curr_text:
            continue

        # Calculate word overlap (similarity)
        prev_words = set(prev_text.split())
        curr_words = set(curr_text.split())
        if not prev_words or not curr_words:
            continue
        overlap = len(prev_words & curr_words)
        total = max(len(prev_words), len(curr_words))
        similarity = overlap / total if total > 0 else 0

        # High similarity (>70%) = separate copies (identical BLs) → keep separate
        # Low similarity (<40%) = different content (BL front + cargo) → continuation
        if similarity < 0.40:
            curr['is_continuation'] = True
            curr['copy_status'] = prev.get('copy_status', curr.get('copy_status', ''))
            _progress(f"  Page {curr.get('page_number', '?')}: CONTINUATION of page {prev.get('page_number', '?')} (similarity={similarity:.0%})")

    # ── Phase 1c: Continuation type inheritance ──
    # Handles THREE scenarios:
    #
    # A) NORMAL SEQUENCING (adjacent pages, no Page X of Y):
    #    Page 19 = "Quality / Analysis", page 20 = continuation "Inspection Cert"
    #    → page 20 inherits "Quality / Analysis" from page 19
    #
    # B) REVERSE SEQUENCING (pages have "Page X of Y" out of PDF order):
    #    Page 19 = "Page 3 of 3", page 20 = "Page 2 of 3", page 22 = "Page 1 of 3"
    #    → DON'T inherit here — let multi-page grouping (Rule 3) handle via
    #      type normalization + "Page X of Y" matching
    #
    # C) SINGLE-PAGE documents between multi-page reports:
    #    Page 18 = COO, page 19 = report "Page 2 of 3" (continuation)
    #    → DON'T inherit COO into the report page
    #
    # RULE: Only inherit previous type when the continuation page does NOT have
    #        its own "Page X of Y" marker. Pages with "Page X of Y" are handled
    #        by the multi-page grouping in Rule 3 below.

    # Build type-normalization aliases for similarity checking
    _SURVEY_ALIASES = {
        'quality / analysis', 'quality/analysis', 'quality analysis',
        'quality certificate', 'products quality certificate',
        'loading inspection report', 'full loading survey report',
        'survey report', 'inspection report', 'inspection certificate',
        'pre-shipment inspection report', 'pre-shipment inspection certificate',
        'draught survey report', 'loading report', 'discharge report',
    }

    _prev_type = None
    for cls in classifications:
        pg_num = cls.get('page_number', 0)
        doc_type = (cls.get('document_type', '') or '').strip()
        doc_type_lower = doc_type.lower()
        is_cont = cls.get('is_continuation', False)

        # Skip blank pages — don't let them break the chain
        if doc_type_lower in ('blank page', 'blank_page', ''):
            continue

        # Skip SWIFT pre-classified pages — those are already correct
        if pg_num in _swift_preclassified:
            _prev_type = doc_type
            continue

        # Track the previous page number for multi-page-marker inheritance
        _prev_pg_num = locals().get('_prev_pg_num', 0)

        # OVERRIDE: "Page X of Y" with X>1 is authoritative for continuation.
        # The VLM sometimes says cont=false for pages 2/3/etc. because the
        # page has its own heading — ignore that, the Page-of-Y footer wins.
        if pg_num in _page_of_total:
            _x, _y = _page_of_total[pg_num]
            if _x > 1:
                _prev_xy = _page_of_total.get(_prev_pg_num)
                if _prev_xy and _prev_xy[1] == _y and _prev_xy[0] == _x - 1:
                    # Sequential continuation — force is_cont=True
                    if not is_cont:
                        _progress(f"  Page {pg_num}: FORCING cont=true (has Page {_x} of {_y}, prev has Page {_x-1} of {_y}) — VLM said cont=false")
                    is_cont = True
                    cls['is_continuation'] = True

        if is_cont and _prev_type:
            # Check if this page has its own "Page X of Y" marker
            _has_page_xy = pg_num in _page_of_total

            if _has_page_xy:
                _x, _y = _page_of_total[pg_num]
                # Multi-page signal is authoritative for continuation decisions.
                # If X > 1 AND the immediately preceding PDF page had
                # "Page X-1 of Y" (same Y), this IS the continuation of that
                # doc — inherit the previous doc_type even if the VLM gave it
                # a different name (common when each page has its own heading).
                _prev_xy = _page_of_total.get(_prev_pg_num)
                if _x > 1 and _prev_xy and _prev_xy[1] == _y and _prev_xy[0] == _x - 1:
                    if doc_type_lower != _prev_type.lower():
                        _progress(f"  Page {pg_num}: MULTI-PAGE CONTINUATION (Page {_x} of {_y}): '{doc_type}' → '{_prev_type}' (inherits from Page {_x-1} of {_y})")
                        cls['document_type'] = _prev_type
                    # Keep _prev_type unchanged so subsequent pages of the
                    # same multi-page doc also inherit correctly.
                else:
                    # Not a direct sequential continuation — let multi-page
                    # grouping (Rule 3) handle it and update _prev_type.
                    _prev_type = doc_type
                    _progress(f"  Page {pg_num}: has Page {_x} of {_y} — skipping inheritance, will use multi-page grouping")
            else:
                # No "Page X of Y" — safe to inherit previous page's type
                if doc_type_lower != _prev_type.lower():
                    _progress(f"  Page {pg_num}: CONTINUATION type fix: '{doc_type}' → '{_prev_type}' (inherits from previous page)")
                    cls['document_type'] = _prev_type
                # Keep _prev_type unchanged (the inherited type)
        else:
            # Not a continuation — this becomes the new "previous type"
            _prev_type = doc_type
        # Update previous-page tracker for next iteration
        _prev_pg_num = pg_num

    # ── Phase 2: Group pages into document packets ──
    _progress("Grouping pages into document packets...")
    packets = _group_into_packets(classifications)

    # ── Smart Document Merging ──
    # After initial packet building, merge related documents that are on
    # different (non-adjacent) pages:
    # 1. BL + BL Conditions of Carriage → merge T&C into nearest BL packet
    # 2. Multi-page reports (Page X of Y) → group by issuer + report type
    # 3. Bill of Exchange front + endorsement back → merge
    # 4. Any document split across pages with same issuer/BL number

    _bl_types = {'bill of lading'}
    _bl_tc_types = {'bl conditions of carriage', 'conditions of carriage', 'bl terms and conditions'}
    _bl_attach_types = {'attach list', 'attached sheet', 'attached list', 'rider',
                        'bl attached sheet', 'bl rider', 'attached schedule'}
    _boe_types = {'draft bill of exchange', 'bill of exchange'}
    _boe_back_types = {'endorsement page'}
    _skip_merge = {'blank page', 'blank_page', 'header page'}

    merged_packets = []
    _consumed = set()  # packet indices that were merged into another

    for i, pkt in enumerate(packets):
        if i in _consumed:
            continue
        dt = pkt.document_type.lower().strip()

        # Rule 1: BL — absorb ONE T&C page, picking the CLOSEST by page distance.
        # Trade-finance convention: each BL has exactly ONE T&C (overleaf/back).
        # - If a T&C page is within 3 pages of the BL (before or after), merge.
        # - If no T&C is adjacent, leave the BL untouched so Step 3d can
        #   correctly set bl_subtype.is_blank_back = true.
        if dt in _bl_types:
            _bl_max = max(pkt.page_numbers) if pkt.page_numbers else 0
            _bl_min = min(pkt.page_numbers) if pkt.page_numbers else 9999
            _best_tc = None
            _best_dist = 999
            for j, other in enumerate(packets):
                if j in _consumed or j == i:
                    continue
                odt = other.document_type.lower().strip()
                if odt not in _bl_tc_types:
                    continue
                _tc_min = min(other.page_numbers) if other.page_numbers else 9999
                _tc_max = max(other.page_numbers) if other.page_numbers else 0
                # Prefer T&C that comes AFTER the BL (overleaf is typically
                # the next page). Accept BEFORE only when nothing follows.
                if _tc_min > _bl_max:
                    _dist = _tc_min - _bl_max
                elif _tc_max < _bl_min:
                    _dist = (_bl_min - _tc_max) + 100   # penalise "before BL"
                else:
                    _dist = 0                             # interleaved
                if _dist < _best_dist:
                    _best_tc = j
                    _best_dist = _dist
            # Only merge if within 3-page window — beyond that it likely
            # belongs to a DIFFERENT BL packet.
            if _best_tc is not None and _best_dist <= 3:
                other = packets[_best_tc]
                pkt.page_numbers.extend(other.page_numbers)
                pkt.pages.extend(other.pages)
                pkt.stamps.extend(other.stamps)
                pkt.signatures.extend(other.signatures)
                pkt.seals.extend(other.seals)
                _consumed.add(_best_tc)
                _progress(f"  Merged {other.packet_id} (BL T&C pg {other.page_numbers}) into {pkt.packet_id} (BL pg {pkt.page_numbers[:3]}) — distance {_best_dist}")

        # Rule 2: Bill of Exchange — absorb endorsement pages
        elif dt in _boe_types:
            for j, other in enumerate(packets):
                if j in _consumed or j == i:
                    continue
                odt = other.document_type.lower().strip()
                if odt in _boe_back_types:
                    pkt.page_numbers.extend(other.page_numbers)
                    pkt.pages.extend(other.pages)
                    pkt.stamps.extend(other.stamps)
                    pkt.signatures.extend(other.signatures)
                    _consumed.add(j)
                    _progress(f"  Merged {other.packet_id} (Endorsement pg {other.page_numbers}) into {pkt.packet_id} (BoE pg {pkt.page_numbers[:2]})")
                    break

        merged_packets.append(pkt)

    # Rule 1b: Attach List → merge into nearest preceding BL
    # "Attach List" / "Attached Sheet" is a rider page of a Bill of Lading
    # containing cargo details that didn't fit on the main BL page.
    # Merge each Attach List into the nearest BL that comes BEFORE it.
    for i, pkt in enumerate(merged_packets):
        if i in _consumed:
            continue
        dt = pkt.document_type.lower().strip()
        if dt not in _bl_attach_types:
            continue
        # Find nearest preceding BL by page number
        _attach_pg = min(pkt.page_numbers) if pkt.page_numbers else 999
        _best_bl = None
        _best_dist = 999
        for j, other in enumerate(merged_packets):
            if j in _consumed or j == i:
                continue
            odt = other.document_type.lower().strip()
            if odt in _bl_types:
                _bl_max_pg = max(other.page_numbers) if other.page_numbers else 0
                _dist = _attach_pg - _bl_max_pg
                if 0 < _dist < _best_dist:
                    _best_bl = j
                    _best_dist = _dist
        if _best_bl is not None:
            _bl_pkt = merged_packets[_best_bl]
            _bl_pkt.page_numbers.extend(pkt.page_numbers)
            _bl_pkt.pages.extend(pkt.pages)
            _bl_pkt.stamps.extend(pkt.stamps)
            _bl_pkt.signatures.extend(pkt.signatures)
            _consumed.add(i)
            _progress(f"  Merged {pkt.packet_id} (Attach List pg {pkt.page_numbers}) into {_bl_pkt.packet_id} (BL pg {_bl_pkt.page_numbers[:3]})")

    # Rule 1 already skipped consumed packets (via `if i in _consumed: continue`
    # at the top of the loop) so they were never appended to merged_packets.
    # The previous `merged_packets = [p for i, p in enumerate(merged_packets) if i not in _consumed]`
    # was BUGGED: _consumed holds indices into the original `packets` list but
    # the filter used them as indices into `merged_packets` (which has fewer
    # entries and different positions). That silently dropped an unrelated
    # packet (e.g. Document Remittance page 8 on job c4384df6) because its
    # position in merged_packets happened to collide with a _consumed index.
    # Just reset _consumed for Rule 3; merged_packets already excludes
    # Rule 1 victims.
    _consumed = set()  # Reset for Rule 3

    # Build page_text_map for Rule 3 (needed for "Page X of Y" detection)
    page_text_map = {}
    for page in pages:
        if hasattr(page, 'page_number'):
            page_text_map[page.page_number] = {
                'raw_text': page.raw_text,
                'cleaned_text': page.cleaned_text,
                'page_image_path': page.page_image_path,
            }
        else:
            page_text_map[page.get('page_number', 0)] = {
                'raw_text': page.get('raw_text', ''),
                'cleaned_text': page.get('cleaned_text', ''),
                'page_image_path': page.get('page_image_path', ''),
            }

    # Rule 3: Group multi-page reports by "Page X of Y" within their text
    # Documents like Alfred H Knight reports have "Page 1 of 3", "Page 2 of 3"
    # in their footer. Group packets from the same report type + same issuer.
    _report_groups = {}  # key: (doc_type_normalized, issuer) -> list of packet indices
    for i, pkt in enumerate(merged_packets):
        dt = pkt.document_type.lower().strip()
        if dt in _skip_merge or dt in _bl_types or dt in _bl_tc_types:
            continue
        # Check if any page has "Page X of Y" with Y > 1
        for pg_cls in pkt.pages:
            hint = pg_cls.get('doc_hint', '') if isinstance(pg_cls, dict) else ''
            # Look for multi-page indicator in the page text
            pg_num = pg_cls.get('page_number', 0) if isinstance(pg_cls, dict) else 0
            pg_text = page_text_map.get(pg_num, {}).get('cleaned_text', '') if pg_num else ''
            _pxy = re.search(r'Page\s+(\d+)\s+of\s+(\d+)', pg_text, re.IGNORECASE)
            if _pxy and int(_pxy.group(2)) > 1:
                _page_x = int(_pxy.group(1))
                _page_y = int(_pxy.group(2))

                # Skip if this is a BAHL report pagination (large Y like "Page 1 of 7"
                # or "Page 1 of 43") AND the document is a SWIFT type — those are
                # report-level pagination, not document-level.
                if _page_y > 5 and dt in ('lc', 'amendment', 'mt799', 'mt999', 'mt730',
                                            'mt754', 'mt940', 'mt740', 'mt747'):
                    break  # Don't use report-level pagination for SWIFT merging

                # Extract issuer from first 500 chars (covers letterhead + subheading)
                _issuer = ''
                _header = pg_text[:500].upper()
                if 'ALFRED H KNIGHT' in _header or 'AHK' in _header:
                    _issuer = 'AHK'
                elif 'INTERTEK' in _header:
                    _issuer = 'INTERTEK'
                elif 'SGS' in _header:
                    _issuer = 'SGS'
                elif 'BUREAU VERITAS' in _header:
                    _issuer = 'BV'
                elif 'COTECNA' in _header:
                    _issuer = 'COTECNA'
                elif 'CALEB BRETT' in _header:
                    _issuer = 'CB'
                elif 'SAYBOLT' in _header:
                    _issuer = 'SAYBOLT'
                elif 'OLAM' in _header:
                    _issuer = 'OLAM'
                elif 'NATIONAL CHEMICAL' in _header or 'NCC' in _header:
                    _issuer = 'NCC'
                elif 'STANDARD CHARTERED' in _header:
                    _issuer = 'SCB'
                elif 'VITOL' in _header:
                    _issuer = 'VITOL'
                elif 'TRAFIGURA' in _header:
                    _issuer = 'TRAFIGURA'
                elif 'PETRONAS' in _header:
                    _issuer = 'PETRONAS'
                elif 'GUNVOR' in _header:
                    _issuer = 'GUNVOR'
                else:
                    # Fallback: use first 4+ letter word as issuer identifier
                    _words = re.findall(r'[A-Z]{4,}', _header)
                    _issuer = _words[0] if _words else 'UNK'

                _norm_type = re.sub(r'\s*\(.*?\)', '', dt).strip()

                # Normalize surveyor report variants to a common type
                # VLM may classify different pages of the same report with
                # different names (e.g., "Quality / Analysis" vs "Loading
                # Inspection Report" vs "Full Loading Survey Report" — all
                # from Alfred H Knight). Group them by normalized type.
                _SURVEYOR_REPORT_ALIASES = {
                    'quality / analysis': 'survey_report',
                    'quality/analysis': 'survey_report',
                    'quality analysis': 'survey_report',
                    'quality certificate': 'survey_report',
                    'products quality certificate': 'survey_report',
                    'loading inspection report': 'survey_report',
                    'full loading survey report': 'survey_report',
                    'survey report': 'survey_report',
                    'survey report certificate': 'survey_report',
                    'inspection report': 'survey_report',
                    'certificate of quality': 'survey_report',
                    'certificate of weight': 'survey_report',
                    'certificate of quality and weight': 'survey_report',
                    'certificate of quality and weight ascertained': 'survey_report',
                    'certificate of quality and weight ascertained at port of loading': 'survey_report',
                    'certificate of quality and weight ascertained at port of discharge': 'survey_report',
                    'certificate of analysis': 'survey_report',
                    'certificate of inspection': 'survey_report',
                    'inspection certificate': 'survey_report',
                    'pre-shipment inspection report': 'survey_report',
                    'pre-shipment inspection certificate': 'survey_report',
                    'draught survey report': 'survey_report',
                    'loading report': 'survey_report',
                    'discharge report': 'survey_report',
                    'quantity certificate': 'quantity_report',
                    'products quantity certificate': 'quantity_report',
                    'certificate of quantity': 'quantity_report',
                    'certificate of receipted quantity': 'quantity_report',
                    'weight certificate': 'weight_report',
                    'weight / quality certificate': 'weight_report',
                    'weight/quality certificate': 'weight_report',
                }
                _norm_type = _SURVEYOR_REPORT_ALIASES.get(_norm_type, _norm_type)

                _key = (_norm_type, _issuer, str(_page_y))

                if _key not in _report_groups:
                    _report_groups[_key] = []
                # Store (packet_index, page_x, first_page_number) for ordering
                _report_groups[_key].append((i, _page_x, pkt.page_numbers[0]))
                break

    # Merge multi-page report groups
    for _key, pkt_entries in _report_groups.items():
        if len(pkt_entries) <= 1:
            continue

        # Check: number of packets should match or be close to total pages
        _total_y = int(_key[2])
        # Detect multiple copies: if any page_x appears more than once,
        # we have separate copies (e.g., two packets both claiming "Page 1 of 3")
        _page_x_values = [e[1] for e in pkt_entries]
        _has_duplicate_pages = len(_page_x_values) != len(set(_page_x_values))
        if len(pkt_entries) > _total_y or _has_duplicate_pages:
            # Multiple copies of the same N-page report. Assign each orphan
            # (Page 2, Page 3, ...) to the NEAREST PRECEDING Page-1 anchor
            # that doesn't already own that page number.
            # Previous algorithm iterated Page-1 anchors in order and greedily
            # grabbed far-away pages — it would give an anchor at page 18 the
            # Page-3 at page 26 (distance 8) instead of the anchor at page 24
            # (distance 2). Fixed: for each orphan, pick the closest preceding
            # anchor that still needs that page number.
            _sorted = sorted(pkt_entries, key=lambda x: x[2])  # sort by first page number
            anchors = {}  # anchor_first_pg -> [entries]
            _orphan_queue = []
            for _entry in _sorted:
                if _entry[1] == 1:
                    anchors[_entry[2]] = [_entry]
                else:
                    _orphan_queue.append(_entry)

            # Assign each orphan to its best-matching anchor (nearest preceding
            # anchor that hasn't already absorbed this page number). If none
            # fit, fall back to the nearest anchor regardless of direction.
            for _orphan in _orphan_queue:
                _o_idx, _o_x, _o_pg = _orphan
                _best_anchor_pg = None
                _best_dist = 999
                # Prefer anchors that come BEFORE the orphan in page order
                for _a_pg, _a_group in anchors.items():
                    _a_xs = {e[1] for e in _a_group}
                    if _o_x in _a_xs:
                        continue  # this anchor already has a Page _o_x
                    _dist = _o_pg - _a_pg
                    if _dist < 0:
                        continue  # orphan must be after the anchor
                    if _dist <= _total_y * 3 and _dist < _best_dist:
                        _best_anchor_pg = _a_pg
                        _best_dist = _dist
                # Fallback: accept anchor AFTER the orphan if no preceding one
                if _best_anchor_pg is None:
                    for _a_pg, _a_group in anchors.items():
                        _a_xs = {e[1] for e in _a_group}
                        if _o_x in _a_xs:
                            continue
                        _dist = abs(_o_pg - _a_pg)
                        if _dist <= _total_y * 3 and _dist < _best_dist:
                            _best_anchor_pg = _a_pg
                            _best_dist = _dist
                if _best_anchor_pg is not None:
                    anchors[_best_anchor_pg].append(_orphan)

            _groups = [g for g in anchors.values() if len(g) > 1]
            # Track entries that found a home so we don't double-merge
            _used_ids = set()
            for g in _groups:
                for e in g:
                    _used_ids.add(id(e))
            # Orphans without any preceding anchor stay as their own packets

            # Merge each proximity group
            for _group in _groups:
                if len(_group) <= 1:
                    continue
                _sorted_group = sorted(_group, key=lambda x: x[1])  # sort by page_x
                primary_idx = _sorted_group[0][0]
                primary = merged_packets[primary_idx]
                for _entry in _sorted_group[1:]:
                    other_idx = _entry[0]
                    other = merged_packets[other_idx]
                    if other_idx not in _consumed:
                        primary.page_numbers.extend(other.page_numbers)
                        primary.pages.extend(other.pages)
                        primary.stamps.extend(other.stamps)
                        primary.signatures.extend(other.signatures)
                        primary.seals.extend(other.seals)
                        _consumed.add(other_idx)
                        _progress(f"  Merged {other.packet_id} (pg {other.page_numbers}) into {primary.packet_id} ({_key[0]}, copy group)")
        else:
            # Exact match — merge all into one
            _sorted = sorted(pkt_entries, key=lambda x: x[1])  # sort by page_x
            primary_idx = _sorted[0][0]
            primary = merged_packets[primary_idx]
            for _entry in _sorted[1:]:
                other_idx = _entry[0]
                other = merged_packets[other_idx]
                if other_idx not in _consumed:
                    primary.page_numbers.extend(other.page_numbers)
                    primary.pages.extend(other.pages)
                    primary.stamps.extend(other.stamps)
                    primary.signatures.extend(other.signatures)
                    primary.seals.extend(other.seals)
                    _consumed.add(other_idx)
                    _progress(f"  Merged {other.packet_id} (pg {other.page_numbers}) into {primary.packet_id} ({_key[0]}, {_key[1]}, {_key[2]} pages)")

    # Remove consumed packets
    final_packets = [p for i, p in enumerate(merged_packets) if i not in _consumed]

    # Sort page_numbers within each packet
    for pkt in final_packets:
        pkt.page_numbers.sort()

    if len(packets) != len(final_packets):
        _progress(f"  Smart merging: {len(packets)} → {len(final_packets)} packets ({len(packets) - len(final_packets)} merged)")

    packets = final_packets

    # ──────────────────────────────────────────────────────────────────── #
    # NEW POST-GROUPING PHASES                                              #
    #   Phase V1 — Build per-packet combined text (used by V2/V3/V4)       #
    #   Phase V2 — LLM validation on combined packet text                  #
    #   Phase V3 — BL sub-type classification (BL packets only)            #
    #   Phase V4 — Packet summary (tiered chunking: VLM / LLM / chunked)   #
    # ──────────────────────────────────────────────────────────────────── #

    # Phase V1: combined text + image paths per packet
    _packet_texts: Dict[str, str] = {}
    _packet_images: Dict[str, List[str]] = {}
    for pkt in packets:
        _texts: List[str] = []
        _imgs: List[str] = []
        for pg_num in pkt.page_numbers:
            pg_data = page_text_map.get(pg_num, {})
            _texts.append(pg_data.get('cleaned_text', pg_data.get('raw_text', '')) or '')
            _img = pg_data.get('page_image_path') or ''
            if _img:
                _imgs.append(_img)
        _packet_texts[pkt.packet_id] = "\n\n--- PAGE BREAK ---\n\n".join(_texts)
        _packet_images[pkt.packet_id] = _imgs
        pkt.doc_hint = pkt.doc_hint or pkt.document_type

    # Phase V2: validate each packet's classification against its combined text
    _progress(f"[3f] Validating {len(packets)} packets via LLM (no regex)...")
    _validation_failures: List[tuple] = []  # (packet, verdict_dict)
    if QWEN_TEXT_LLM_URL:
        with ThreadPoolExecutor(max_workers=max(1, MAX_CONCURRENT_VLM // 2)) as executor:
            _vfutures = {}
            for pkt in packets:
                _vfutures[executor.submit(
                    _validate_packet_llm, pkt.document_type, _packet_texts[pkt.packet_id]
                )] = pkt
            for fut in as_completed(_vfutures):
                pkt = _vfutures[fut]
                try:
                    verdict = fut.result()
                except Exception as e:
                    verdict = {"verdict": "UNSURE", "confidence": 0.0,
                               "suggested_type": "", "reason": f"validator error: {e}"}
                if verdict.get("verdict") == "NO":
                    _validation_failures.append((pkt, verdict))
                    pkt.validation_status = "low_confidence"
                    _progress(f"  {pkt.packet_id} FAILED validation: claimed={pkt.document_type}, "
                              f"suggested={verdict.get('suggested_type')!r} — {verdict.get('reason')}")
    else:
        _progress(f"  Skipped validation (QWEN_TEXT_LLM_URL not set)")

    # Phase V2b: Neighbour-context re-check for pages in failed packets
    if _validation_failures:
        _progress(f"[3g] Re-checking {sum(len(p.page_numbers) for p, _ in _validation_failures)} "
                  f"pages from {len(_validation_failures)} failed packets (neighbour context)...")
        # Build a page_num → (prev_type, next_type) map
        _pages_sorted = sorted(page_text_map.keys())
        _neighbours = {}
        # Use current packet assignment to look up prev/next doc types
        _page_to_pkt_type = {}
        for pkt in packets:
            for pg in pkt.page_numbers:
                _page_to_pkt_type[pg] = pkt.document_type
        for idx, pg in enumerate(_pages_sorted):
            prev_t = _page_to_pkt_type.get(_pages_sorted[idx - 1], "") if idx > 0 else ""
            next_t = _page_to_pkt_type.get(_pages_sorted[idx + 1], "") if idx + 1 < len(_pages_sorted) else ""
            _neighbours[pg] = (prev_t, next_t)

        with ThreadPoolExecutor(max_workers=max(1, MAX_CONCURRENT_VLM // 2)) as executor:
            _rfutures = {}
            for pkt, verdict in _validation_failures:
                for pg_num in pkt.page_numbers:
                    pg_data = page_text_map.get(pg_num, {})
                    _text = pg_data.get('cleaned_text', pg_data.get('raw_text', '')) or ''
                    _img = pg_data.get('page_image_path') or ''
                    if not _img:
                        continue
                    prev_t, next_t = _neighbours.get(pg_num, ("", ""))
                    _rfutures[executor.submit(
                        _recheck_page_with_context, pg_num, _img, _text,
                        prev_t, next_t, pkt.document_type,
                        verdict.get("verdict", "UNSURE"), verdict.get("reason", "")
                    )] = (pkt, pg_num)
            for fut in as_completed(_rfutures):
                pkt, pg_num = _rfutures[fut]
                try:
                    rc = fut.result()
                except Exception as e:
                    rc = {"_error": str(e)}
                if not rc.get("_error") and rc.get("document_type"):
                    _progress(f"  re-check p{pg_num}: {pkt.document_type} -> {rc['document_type']} "
                              f"(conf {rc.get('confidence', 0):.2f})")
                    pkt.validation_status = "re_checked"
                    # Note: we record the re-check but DON'T regroup automatically —
                    # regrouping would risk breaking SWIFT/BAHL logic. Downstream
                    # consumers can read pkt.validation_status and re-check values.

    # Phase V3: BL sub-type classification (only for Bill of Lading packets).
    # Broad match — covers all BL variants (CONGENBILL, Combined Transport BL,
    # Tanker BL, House BL, Master BL, Charter Party BL, etc.) that may slip
    # past canonicalization (e.g. on jobs that ran before the canonicalizer).
    def _is_bl(dt: str) -> bool:
        if not dt:
            return False
        dt_lower = dt.lower()
        if 'conditions of carriage' in dt_lower:
            return False  # back side of BL, not a BL itself
        return (
            'bill of lading' in dt_lower
            or 'b/l' in dt_lower
            or dt_lower in ('bl', 'congenbill', 'gencon')
        )
    _bl_packets = [p for p in packets if _is_bl(p.document_type)]
    if _bl_packets:
        _progress(f"[3d] Classifying BL sub-type for {len(_bl_packets)} Bill of Lading packet(s)...")
        with ThreadPoolExecutor(max_workers=max(1, MAX_CONCURRENT_VLM // 2)) as executor:
            _blfutures = {}
            for pkt in _bl_packets:
                _front = _packet_images[pkt.packet_id][0] if _packet_images[pkt.packet_id] else None
                _rev = _packet_images[pkt.packet_id][1] if len(_packet_images[pkt.packet_id]) > 1 else None
                _blfutures[executor.submit(
                    _classify_bl_subtype, _packet_texts[pkt.packet_id], _front, _rev
                )] = pkt
            for fut in as_completed(_blfutures):
                pkt = _blfutures[fut]
                try:
                    pkt.bl_subtype = fut.result()
                    _progress(f"  {pkt.packet_id}: form={pkt.bl_subtype.get('form_type')}, "
                              f"contract={pkt.bl_subtype.get('contract_type')}, "
                              f"signing={pkt.bl_subtype.get('signing_type')}")
                except Exception as e:
                    pkt.bl_subtype = {"_error": str(e)}

    # Phase V4: Packet summary for every packet (tiered chunking)
    _progress(f"[3e] Generating packet summaries for {len(packets)} packets (tiered by page count)...")
    _size_bucket = {"small": 0, "medium": 0, "large": 0}
    with ThreadPoolExecutor(max_workers=max(1, MAX_CONCURRENT_VLM // 2)) as executor:
        _sfutures = {}
        for pkt in packets:
            _pg_texts = _packet_texts[pkt.packet_id].split("\n\n--- PAGE BREAK ---\n\n")
            _pg_imgs = _packet_images[pkt.packet_id]
            n = len(pkt.page_numbers)
            if n <= 4:
                _size_bucket["small"] += 1
            elif n <= 20:
                _size_bucket["medium"] += 1
            else:
                _size_bucket["large"] += 1
            _sfutures[executor.submit(
                _summarize_packet, pkt.document_type, _pg_texts, _pg_imgs
            )] = pkt
        for fut in as_completed(_sfutures):
            pkt = _sfutures[fut]
            try:
                pkt.unified_summary = fut.result()
            except Exception as e:
                pkt.unified_summary = {"_error": str(e)}
    _progress(f"  summaries: {_size_bucket['small']} small (≤4pg VLM), "
              f"{_size_bucket['medium']} medium (5-20pg LLM), "
              f"{_size_bucket['large']} large (21+pg chunked LLM)")

    elapsed = time.time() - start_time

    _progress(f"Step 3 complete: {len(packets)} packets from {len(pages)} pages in {elapsed:.1f}s")
    for pkt in packets:
        _progress(f"  {pkt.packet_id}: {pkt.document_type} (pages {pkt.page_numbers}, "
                  f"copy={pkt.copy_status}, stamps={len(pkt.stamps)}, sigs={len(pkt.signatures)})")

    # ── Save results ──
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        result_file = os.path.join(output_dir, 'step03_result.json')
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({
                'step': 3,
                'step_name': 'Page Sequencing and Document Packet Formation',
                'total_pages': len(pages),
                'total_packets': len(packets),
                'classifications': classifications,
                'packets': [asdict(p) for p in packets],
                'elapsed_seconds': round(elapsed, 2),
            }, f, indent=2, ensure_ascii=False)

    return {
        'packets': [asdict(p) for p in packets],
        'classifications': classifications,
        'total_pages': len(pages),
        'elapsed_seconds': round(elapsed, 2),
    }


if __name__ == '__main__':
    import sys as _sys2
    if len(_sys2.argv) < 2:
        print("Usage: python step03_sequencing.py <step02_result.json>")
        _sys2.exit(1)
    with open(_sys2.argv[1], 'r', encoding='utf-8') as f:
        s2 = json.load(f)
    result = run(s2, 'test_step03')
    print(f"Result: {result['total_pages']} pages -> {len(result['packets'])} packets")
