"""
Step 8 -- Shipping Document Classification Against Expected List
================================================================
For each shipping packet, sends page image + GLM text to Qwen VLM
which classifies the document against the LC's required document list.

KEY PRINCIPLE: GLM OCR already extracted trusted text (Step 1).
Qwen VLM gets the page image + GLM text together. It reviews,
classifies, detects stamps/signatures, and extracts metadata.
Qwen NEVER rewrites GLM text -- only adds missing things.

INPUT:
    - Shipping packets from Step 3 (with GLM text + page images)
    - Required documents from Step 7 (expected document list from LC)

OUTPUT:
    - classified_packets[]: each with document_type, stamps, signatures,
      copy_status, marking_status, issued_by, etc.
"""

import json
import sys as _sys
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import os
import re
import time
import base64
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import QWEN_VLM_URL, QWEN_VLM_MODEL, MAX_CONCURRENT_VLM, VLM_TIMEOUT


# ── Dataclasses ──

@dataclass
class StampInfo:
    """A stamp detected on a document."""
    text: str = ""
    type: str = ""  # rubber_stamp / embossed / printed


@dataclass
class SignatureInfo:
    """A signature detected on a document."""
    description: str = ""
    type: str = ""  # handwritten / digital


@dataclass
class SealInfo:
    """A seal detected on a document."""
    description: str = ""


@dataclass
class LogoInfo:
    """A logo detected on a document."""
    company_name: str = ""


@dataclass
class ClassifiedPacket:
    """A shipping packet after classification against the LC's expected document list."""
    packet_id: str
    original_pages: List[int] = field(default_factory=list)
    page_image_paths: List[str] = field(default_factory=list)
    raw_text: str = ""
    cleaned_text: str = ""

    # Classification results
    document_type: str = ""
    classification_status: str = "unknown"  # matched_document | alien_document | extra_document | unknown
    match_confidence: float = 0.0
    matched_requirement_index: int = -1
    matched_requirement_name: str = ""

    # VLM classification details
    vlm_top_matches: List[dict] = field(default_factory=list)
    vlm_reasoning: str = ""

    # Document summary from VLM
    document_summary: str = ""
    document_number: str = ""
    document_date: str = ""
    document_amount: str = ""

    # Visual elements detected by VLM
    stamps: List[dict] = field(default_factory=list)
    signatures: List[dict] = field(default_factory=list)
    seals: List[dict] = field(default_factory=list)
    logos: List[dict] = field(default_factory=list)

    # Copy/Original status
    copy_status: str = ""        # "original" or "copy"
    copy_label: str = ""         # exact text of copy marking
    marking_status: str = ""     # stamped_and_signed / signed / stamped / unsigned

    # Issuer and references
    issued_by: str = ""
    lc_reference: str = ""

    # Metadata
    source_step: int = 8
    confidence: float = 0.0
    ambiguity_flag: bool = False
    ambiguity_notes: str = ""
    elapsed_seconds: float = 0.0

    # ── Bill of Lading: blank-back / short-form detection ──
    # `is_bl_terms_page` is True when the packet IS the carriage T&Cs sheet
    # (no shipper/consignee/route info, just the printed conditions of carriage).
    # `has_bl_terms_pages_in_set` is True when ANY other packet in the same
    # document set carries those T&Cs — meaning a BL otherwise looking like
    # blank-back is actually a full-form BL whose terms are on a separate sheet.
    is_bl_terms_page: bool = False
    has_bl_terms_pages_in_set: bool = False
    bl_short_form_status: str = ""  # "" | "full_form" | "short_form" | "blank_back"

    # ── NEW: Preserved from Step 3 (do NOT regenerate — read-through only) ──
    # These fields MUST be carried forward so Step 14's new split-prompt path
    # sees the structured facts Step 3 extracted. Without this, Step 14's
    # _build_structured_facts receives empty dicts and the whole refactor is
    # neutered. Always copy these fields from the source Step 3 packet.
    unified_summary: Optional[dict] = None
    bl_subtype: Optional[dict] = None
    validation_status: str = "valid"


# ── Classification Prompt ──

_CLASSIFICATION_PROMPT = """You are a trade finance SHIPPING DOCUMENT classifier.

CRITICAL RULES (read first, follow strictly):
- The Letter of Credit (LC / MT700 / MT707 / MT799) has ALREADY been classified
  upstream. The document you are looking at NOW is a SHIPPING DOCUMENT (e.g.
  invoice, bill of lading, packing list, certificate, etc.).
- You MUST NEVER return "Letter of Credit", "LC", "MT700", "MT707", or
  "Amendment" as the document_type. Those values are FORBIDDEN. If the
  page genuinely looks like an LC by mistake, classify it as the actual
  shipping document type that best fits, or "Header Page" / "Covering
  Letter" / "Unknown" if it has no shipping content.
- Look at the PAGE IMAGE and the OCR TEXT FROM THIS PAGE ONLY. Do NOT
  classify based on what the LC requires — that list is only a HINT for
  resolving the spelling of the document type if there is a match.

==================== CANDIDATE DOCUMENT TYPES (HINT ONLY) ====================
The LC's required-document list is shown below. This is a HINT for choosing
the EXACT spelling when the document on the page matches one of these. It is
NOT a description of what the page contains. Do NOT assume the page is one of
these types — verify against the actual page image and OCR text.

{required_docs_list}
==============================================================================

==================== PAGE TO CLASSIFY (FROM THE IMAGE) =======================
Document text extracted by GLM-OCR:

{glm_text}
==============================================================================

Based on the image AND the document text above, classify this page:
1. document_type: the ACTUAL document title/heading visible on the PAGE. If
   the page matches one of the LC's required types, use that EXACT spelling.
   If no match, use the real title visible on the page (e.g., "Port Clearance
   Certificate", "Time Sheet", "Tanker Cleanliness Certificate", "Shore Tank
   Measurements", "Vessel Experience Factor", "Master Receipt for Sealed
   Samples", "Letter of Authority", etc.).
   FORBIDDEN VALUES: "Letter of Credit", "LC", "MT700", "MT707", "MT799",
   "Amendment". If the page truly has no recognisable shipping content,
   return "Unknown" or "Header Page" or "Blank Page".
   IMPORTANT DISAMBIGUATION RULES:
   • A DHL / FedEx / UPS / TNT / Aramex / EXPRESS ENVELOPE / WAYBILL / AWB / HAWB / MAWB / "Air Waybill" label is a "Courier Receipt" or "Airway Bill" — these two are TREATED AS THE SAME CATEGORY. Use whichever name appears in the LC's required-documents list; if the LC asked for "Courier Receipt" use that, if it asked for "Airway Bill" use that. It is NEVER a "Documentary Remittance" or "Beneficiary Certificate", even though it carries documents.
   • A bank-letterhead page that says "We enclose documents related to above referenced letter of credit", "Total Amount Claimed", "Presentation Number", "Our Reference No.", "Your Documentary Credit No.", or "remit funds to our correspondent" is a "Documentary Remittance" (covering schedule), NEVER a "Beneficiary Certificate".
   • A "Beneficiary Certificate" is a short certificate ISSUED BY THE BENEFICIARY (the seller/exporter), titled "BENEFICIARY'S CERTIFICATE" or similar, certifying a single fact (e.g. "we hereby certify that one set of documents has been sent by courier"). It is on the beneficiary's letterhead, not a bank's.
   • A "Short Form Bill of Lading" / "Blank Back Bill of Lading" / "Liner Bill of Lading" / "Charter Party Bill of Lading" / "Combined Transport Bill of Lading" / "Multimodal Bill of Lading" is STILL a "Bill of Lading" ((v)). It is NOT an "Airway Bill" and NOT a "Courier Receipt" — it just lacks the full carriage terms on the reverse side.
   • CRITICAL — "COVER NOTE NO." / "COVER NOTE NUMBER" / "OPEN POLICY NO." in a subject line or body is an INSURANCE REFERENCE NUMBER (the insurance company's reference for the open-policy cover note for this shipment). It is NOT a description of the email's nature. An email whose subject is "COVER NOTE NO. <reference>" + LC reference is a forwarder / beneficiary / logistics-company transmittal email that travels alongside the actual shipment-advice attachment. Classify it as "Shipment Advice", NOT "Covering Letter" and NOT "Documentary Remittance".
   • Forwarder / logistics-company emails (sender title "Logistics Executive" / "Logistics Manager", domain like samling.com.my / globallogistics.com / TREEONE / etc.) that reference an LC number and either an "OPEN POLICY NO" / "COVER NOTE NO" / "Attached doc for your reference" line are "Shipment Advice", NOT "Covering Letter".
   • A page is ONLY a "Documentary Remittance" / "Covering Schedule" when it carries the BANK-side payment-claim language ("We enclose documents for negotiation/payment", "Total Amount Claimed", "Presentation Number", "Our Reference No.", "L/C Issuing Bank", "Reimbursing Bank", "Payment Instruction"). A bare email without these signals — even when the email itself is from a bank — is NOT a Documentary Remittance.
2. confidence: 0.0 to 1.0
3. summary: Detailed description including ALL key details: drawer/drawee, at sight/usance, to the order of, amount in figures AND words, tenor, maturity, endorsements, and any other relevant terms visible in the document
4. document_number: if visible
5. date: if visible
6. amount: if visible (with currency)
7. stamps: list of stamp text detected [{{"text": "...", "type": "rubber_stamp/embossed/printed"}}]
8. signatures: list [{{"description": "...", "type": "handwritten/digital"}}]
9. seals: list [{{"description": "..."}}]
10. logos: list [{{"company_name": "..."}}]
11. copy_status: "original" or "copy" (look for ORIGINAL/COPY/NON-NEGOTIABLE stamps)
12. copy_label: exact text of copy marking if visible
13. marking_status: "stamped_and_signed" / "signed" / "stamped" / "unsigned"
14. issued_by: who issued this document
15. lc_reference: LC number if visible on document

Return ONLY valid JSON:
{{
    "document_type": "...",
    "confidence": 0.0,
    "summary": "...",
    "document_number": "",
    "date": "",
    "amount": "",
    "stamps": [],
    "signatures": [],
    "seals": [],
    "logos": [],
    "copy_status": "",
    "copy_label": "",
    "marking_status": "",
    "issued_by": "",
    "lc_reference": "",
    "reasoning": ""
}}"""


# ── Rule-Based Pre-Classification ──

_DOC_INDICATORS = {
    "Commercial Invoice": [
        r'COMMERCIAL\s+INVOICE', r'INVOICE\s+NO', r'INVOICE\s+DATE',
        r'UNIT\s+PRICE', r'TOTAL\s+AMOUNT', r'SOLD\s+TO',
    ],
    "Bill of Lading": [
        r'BILL\s+OF\s+LADING', r'B/?L\s+NO', r'SHIPPER', r'CONSIGNEE',
        r'PORT\s+OF\s+LOADING', r'PORT\s+OF\s+DISCHARGE', r'ON\s+BOARD',
        r'VESSEL\s+NAME', r'OCEAN\s+BILL',
        # Short form / blank back / charter party / liner B/L variants —
        # all still classified as "Bill of Lading" ((v)).
        r'SHORT\s+FORM\s+BILL\s+OF\s+LADING',
        r'BLANK\s+BACK\s+BILL\s+OF\s+LADING',
        r'LINER\s+BILL\s+OF\s+LADING',
        r'CHARTER\s+PARTY\s+BILL\s+OF\s+LADING',
        r'COMBINED\s+TRANSPORT\s+BILL\s+OF\s+LADING',
        r'MULTIMODAL\s+(?:TRANSPORT\s+)?BILL\s+OF\s+LADING',
        r'(?:CONDITIONS|TERMS)\s+OF\s+CARRIAGE\s+(?:ARE\s+)?(?:REFERRED|AVAILABLE)',
    ],
    # ── Airway Bill / Courier Receipt are treated as the SAME category ──
    # Both are single-piece air/courier transport documents ( /
    # Art 25). Patterns for both kinds of waybill live here so either signal
    # — an airline AWB or a DHL/FedEx/UPS/TNT/Aramex express envelope —
    # produces the same classification. The companion entry "Courier Receipt"
    # below is an ALIAS that mirrors the same patterns so the matcher in
    # _match_type_to_requirement() will resolve to whichever label the LC
    # expected list uses.
    "Airway Bill": [
        # Classic air waybill signals
        r'AIR\s*WAY\s*BILL', r'\bAWB\b', r'AIRLINE', r'FLIGHT\s+NO',
        r'AIRPORT\s+OF\s+(?:DEPARTURE|DESTINATION)',
        r'HOUSE\s+AIR\s*WAYBILL', r'MASTER\s+AIR\s*WAYBILL', r'\bHAWB\b', r'\bMAWB\b',
        # Courier / express waybill signals (DHL, FedEx, UPS, TNT, Aramex, ...)
        r'\bDHL\b', r'\bFEDEX\b', r'\bFED\s*EX\b', r'\bUPS\b', r'\bTNT\b',
        r'\bARAMEX\b', r'\bSF\s+EXPRESS\b', r'\bBLUE\s+DART\b', r'\bSKYNET\b',
        r'EXPRESS\s+ENVELOPE', r'EXPRESS\s+(?:WAYBILL|COURIER|DELIVERY)',
        r'COURIER\s+(?:RECEIPT|WAYBILL|SERVICE)',
        r'\bWAYBILL\b\s*(?:NO|NUMBER|#)?',
        r'(?:PIECES|PCS)\s*(?:WEIGHT|/?WGT)',
        r'CONTENTS?\s*:\s*DOCUMENTS',
        r'TRACKING\s+(?:NUMBER|NO\.?)',
        r'SHIPPER\s+REFERENCE',
        r'\bXPD\b',          # DHL XPD service
        r'\bWPX\b',          # DHL Worldwide Package Express
    ],
    # Alias category — same patterns as Airway Bill above. The rule-based
    # classifier therefore produces a tied score for both names, and the
    # downstream matcher resolves whichever the LC's expected document list
    # actually requested ("Airway Bill", "Air Waybill", "Courier Receipt",
    # "Courier Service Receipt", etc.).
    "Courier Receipt": [
        r'\bDHL\b', r'\bFEDEX\b', r'\bFED\s*EX\b', r'\bUPS\b', r'\bTNT\b',
        r'\bARAMEX\b', r'\bSF\s+EXPRESS\b', r'\bBLUE\s+DART\b', r'\bSKYNET\b',
        r'EXPRESS\s+ENVELOPE', r'EXPRESS\s+(?:WAYBILL|COURIER|DELIVERY)',
        r'COURIER\s+(?:RECEIPT|WAYBILL|SERVICE)',
        r'\bWAYBILL\b\s*(?:NO|NUMBER|#)?',
        r'(?:PIECES|PCS)\s*(?:WEIGHT|/?WGT)',
        r'CONTENTS?\s*:\s*DOCUMENTS',
        r'TRACKING\s+(?:NUMBER|NO\.?)',
        r'SHIPPER\s+REFERENCE',
        r'\bXPD\b', r'\bWPX\b',
        r'AIR\s*WAY\s*BILL', r'\bAWB\b', r'\bHAWB\b', r'\bMAWB\b',
        r'AIRLINE', r'FLIGHT\s+NO', r'AIRPORT\s+OF\s+(?:DEPARTURE|DESTINATION)',
    ],
    "Insurance Policy/Certificate": [
        r'INSURANCE\s+(?:POLICY|CERTIFICATE)', r'INSURED\s+VALUE',
        r'MARINE\s+CARGO', r'PREMIUM', r'SUM\s+INSURED', r'UNDERWRITER',
    ],
    "Certificate of Origin": [
        r'CERTIFICATE\s+OF\s+ORIGIN', r'COUNTRY\s+OF\s+ORIGIN',
        r'ORIGIN\s+CRITERIA', r'CHAMBER\s+OF\s+COMMERCE',
    ],
    "Packing List": [
        r'PACKING\s+LIST', r'PACKING\s+SLIP',
        r'NET\s+WEIGHT', r'GROSS\s+WEIGHT',
        r'CARTONS?', r'PACKAGES?', r'DIMENSIONS',
    ],
    "Weight List": [
        r'WEIGHT\s+LIST', r'WEIGHT\s+CERTIFICATE', r'TARE\s+WEIGHT',
    ],
    "Draft": [
        r'DRAFT', r'BILL\s+OF\s+EXCHANGE', r'PAY\s+TO\s+THE\s+ORDER',
        r'AT\s+SIGHT', r'TENOR', r'DRAWEE', r'DRAWER',
    ],
    "Beneficiary Certificate": [
        r'BENEFICIARY\s*(?:\'S)?\s*CERTIFICATE', r'WE\s+HEREBY\s+CERTIFY',
    ],
    "Inspection Certificate": [
        r'INSPECTION\s+(?:CERTIFICATE|REPORT)', r'SURVEYOR',
    ],
    "Shipping Advice": [
        r'SHIPPING\s+ADVICE', r'SHIPMENT\s+ADVICE',
    ],
    "Fumigation Certificate": [
        r'FUMIGATION\s+CERTIFICATE', r'FUMIGAT',
    ],
    "Phytosanitary Certificate": [
        r'PHYTOSANITARY', r'PLANT\s+(?:HEALTH|PROTECTION)',
    ],
    "Health Certificate": [
        r'HEALTH\s+CERTIFICATE', r'FIT\s+FOR\s+HUMAN\s+CONSUMPTION',
    ],
    "Documentary Remittance": [
        r'DOCUMENTARY\s+REMITTANCE',
        r'COVERING\s+(?:LETTER|SCHEDULE)',
        r'L/?C\s+BILLS?\s+SCHEDULE',
        r'BILLS?\s+SCHEDULE',
        # P64: "Export DC Document Presentation Schedule" and its variants
        # are the same document as Documentary Remittance — the bank's
        # covering schedule sent with presented documents under an LC.
        r'EXPORT\s+DC\s+DOCUMENT\s+PRESENTATION\s+SCHEDULE',
        r'EXPORT\s+DC\s+PRESENTATION\s+SCHEDULE',
        r'DOCUMENT\s+PRESENTATION\s+SCHEDULE',
        r'(?<!NO\s)PRESENTATION\s+SCHEDULE',
        r'DC\s+PRESENTATION\s+SCHEDULE',
        r'EXPORT\s+(?:LC|L/C|DC|D/C)\s+PRESENTATION',
        r'SCHEDULE\s+OF\s+PRESENTATION',
        r'ENCLOSED\s+HEREWITH',
        r'WE\s+(?:ARE\s+)?ENCLOS(?:E|ING|URE)',
        r'DOCUMENTS?\s+ATTACHED',
        r'PLEASE\s+(?:FIND|ACCEPT)\s+(?:ENCLOSED|ATTACHED|HEREWITH)',
        r'WE\s+ENCLOSE\s+DOCUMENTS?\s+(?:RELATED|RELATING|UNDER|PERTAINING)',
        # Bank covering-schedule structural fields
        r'PRESENTATION\s+(?:NUMBER|NO\.?|DATE|AMOUNT)',
        r'TOTAL\s+(?:AMOUNT\s+)?CLAIMED',
        r'PRINCIPAL\s+AMOUNT\s+(?:CLAIMED|EUR|USD|GBP)',
        r'AMOUNTS?\s+CLAIMED\s*[:\n]',
        r'OUR\s+REFERENCE\s+NO',
        r'YOUR\s+DOCUMENTARY\s+CREDIT\s+NO',
        # "REMIT FUNDS / SETTLEMENT" — bank-to-bank reimbursement claim language
        r'REMIT\s+FUNDS\s+TO\s+(?:OUR\s+)?CORRESPONDENT',
        r'(?:UPON|FOR)\s+SETTLEMENT\s+PLEASE\s+REMIT',
        r'CLAIM\s+REIMBURSEMENT',
        r'QUOTING\s+OUR\s+REFERENCE',
    ],
    # P81: Form 7 = Batch Certificate (Pakistan Drug Act Rule 14(d)(i))
    # Form 3 = Drug Registration / Import Certificate
    "Form 7 (Batch Certificate)": [
        r'FORM\s+7', r'BATCH\s+CERTIFIC(?:ATE|ATION)',
        r'RULE\s+14\s*\(\s*d\s*\)', r'BATCH\s+NO',
        r'DATE\s+OF\s+MANUFACT', r'DATE\s+OF\s+EXPIRY',
        r'DRUG\s+ACT\s+1976',
    ],
    "Form 3": [
        r'FORM\s+3\b', r'DRUG\s+REGISTRATION\s+CERTIFICATE',
        r'IMPORT\s+CERTIFICATE\s+FOR\s+DRUG',
        r'FORM\s+OF\s+UNDERTAKING',
    ],
    # P90: Certificate of Analysis — pharmaceutical / chemical analysis reports
    "Certificate of Analysis": [
        r'CERTIFICATE\s+OF\s+ANALYSIS',
        r'ANALYTICAL?\s+CERTIFICATE',
        r'COA\b',
        r'ANALYSIS\s+REPORT',
        r'ANALYSIS\s+CERTIFICATE',
        r'TEST\s+REPORT',
        r'ASSAY\s+RESULT',
        r'SPECIFICATION\s+AND\s+(?:TEST|ANALYSIS)\s+RESULT',
    ],
    "Letter of Indemnity": [
        r'LETTER\s+OF\s+INDEMNITY',
        r'\bLOI\b',
        r'INDEMNITY\s+LETTER',
        r'WE\s+HEREBY\s+EXPRESSLY\s+WARRANT',
        r'INDEMNIFY\s+AND\s+HOLD.*?HARMLESS',
        r'LOCATE\s+AND\s+SURRENDER.*?BILLS?\s+OF\s+LADING',
        r'UNABLE\s+TO\s+PROVIDE.*?BILLS?\s+OF\s+LADING',
    ],
    "Notice of Readiness": [
        r'NOTICE\s+OF\s+READINESS', r'NOR\s+(?:RE-?)?TENDER', r'VESSEL\s+HAS\s+OFFICIALLY\s+ARRIVED',
    ],
    "Port Clearance Certificate": [
        r'PORT\s+CLEARANCE', r'国际航行船舶出口岸许可证', r'PORT\s+CLEARANCE\s+CERTIFICATE',
    ],
    "Tanker Cleanliness Certificate": [
        r'TANKER\s+CLEANLINESS\s+CERTIFICATE', r'TANK\s+INSPECTION\s+COMPLETED',
        r'NOMINATED\s+TANKS', r'TANK\s+COATINGS',
    ],
    "Shore Tank Measurements": [
        r'SHORE\s+TANK\s+MEASUREMENTS', r'SHORE\s+TANK.*BEFORE\s+AND\s+AFTER',
        r'TANK\s+GAUGED\s+VOLUME', r'TOTAL\s+QUANTITY\s+RECEIVED',
    ],
    "Time Sheet": [
        r'TIME\s+SHEET', r'VESSEL\s+ARRIVED', r'COMMENCED\s+LOADING',
        r'COMPLETED\s+LOADING', r'PILOT\s+SCHEDULED',
    ],
    "Vessel Experience Factor": [
        r'VESSEL\'?S?\s+EXPERIENCE\s+FACTOR', r'MEASUREMENTS?\s+OF\s+QUANTITY\s+RECEIVED\s+ON\s+VESSEL',
        r'AVERAGE\s+TCV\s+RATIO',
    ],
    "Master Receipt for Sealed Samples": [
        r'MASTER\'?S?\s+RECEIPT\s+FOR\s+SEALED\s+SAMPLES',
        r'SEALED\s+SAMPLES', r'SEAL\s+NUMBER',
    ],
    "Letter of Authority": [
        r'LETTER\s+OF\s+AUTHORITY', r'AUTHORISE\s+.*AGENT',
        r'AUTHORITY\s+FOR\s+SIGNING\s+BILL\s+OF\s+LADING',
    ],
    "Certificate of Receipted Quantity": [
        r'CERTIFICATE\s+OF\s+RECEIPTED\s+QUANTITY', r'RECEIPTED\s+QUANTITY',
        r'SHORE\s+RECEIPTED\s+QUANTITIES',
    ],
    "Products Quality Certificate": [
        r'PRODUCTS?\s+QUALITY\s+CERTIFICATE', r'TEST\s+ITEM.*METHOD.*RESULT',
    ],
    "Products Quantity Certificate": [
        r'PRODUCTS?\s+QUANTITY\s+CERTIFICATE',
    ],
    "Loading Inspection Report": [
        r'LOADING\s+INSPECTION', r'LOAD(?:ING)?\s+PORT\s+INDEPENDENT',
    ],
    "Survey Report": [
        r'SURVEY\s+REPORT', r'SURVEYOR.*REPORT',
    ],
    "Mate Receipt": [
        r'MATE\'?S?\s+RECEIPT', r'MATE\s+RECEIPT',
    ],
    "Statement of Facts": [
        r'STATEMENT\s+OF\s+FACTS',
    ],
    "Stowage Plan": [
        r'STOWAGE\s+PLAN', r'LOADING\s+PLAN',
    ],
    "Insurance Pre-Advise Notice": [
        r'INSURANCE\s+PRE[- ]?ADVI[SC]E', r'PRE[- ]?ADVISE\s+NOTICE',
        r'ESTIMATED\s+SHIPMENT\s+DETAILS', r'NOMINEE\s+OPEN\s+POLICY',
    ],
}


# ── Bill of Lading "Terms and Conditions of Carriage" page detector ──
#
# A blank-back / short-form BL is one whose REVERSE side does not carry the
# detailed carriage terms. (v) accepts these unconditionally
# UNLESS the LC explicitly forbids them ("SHORT FORM / BLANK BACK / HOUSE /
# STALE / FORWARDER AGENT BL NOT ACCEPTABLE").
#
# However, when the LC DOES forbid blank-back BLs, we must not raise the
# discrepancy if the document set ALSO contains a separate page printing
# the carriage terms — that page IS the "reverse side", just supplied on a
# separate sheet, and the BL is therefore a full-form BL.
#
# This helper recognises a T&C page by counting BL legal-clause keywords.
_BL_TERMS_KEYWORDS = [
    r'CONDITIONS?\s+OF\s+CARRIAGE',
    r'TERMS\s+AND\s+CONDITIONS\s+OF\s+(?:CARRIAGE|TRANSPORT)',
    r'CARRIER\'?S?\s+(?:LIABILITY|RESPONSIBILITY|OBLIGATIONS?)',
    r'HAGUE\s+(?:RULES|VISBY|VISBY\s+RULES)',
    r'HAMBURG\s+RULES',
    r'ROTTERDAM\s+RULES',
    r'COGSA\b',                                       # Carriage of Goods by Sea Act
    r'GENERAL\s+AVERAGE',
    r'YORK[/\-]ANTWERP\s+RULES',
    r'PARAMOUNT\s+CLAUSE',
    r'JURISDICTION\s+(?:AND\s+)?(?:LAW|CLAUSE)',
    r'LAW\s+AND\s+JURISDICTION',
    r'NOTICE\s+OF\s+(?:CLAIM|LOSS\s+OR\s+DAMAGE)',
    r'PERIOD\s+OF\s+RESPONSIBILITY',
    r'DECK\s+CARGO',
    r'LIVE\s+ANIMALS',
    r'DANGEROUS\s+(?:GOODS|CARGO)',
    r'FREIGHT\s+(?:PREPAID|COLLECT|PAYABLE)',
    r'DEMURRAGE',
    r'LIEN\s+ON\s+(?:CARGO|GOODS)',
    r'SUB[- ]?CONTRACTING',
    r'HIMALAYA\s+CLAUSE',
    r'BOTH[- ]TO[- ]BLAME\s+COLLISION',
    r'NEW\s+JASON\s+CLAUSE',
    r'CLAUSE\s+PARAMOUNT',
    r'MERCHANT\s+SHALL',
    r'CARRIER\s+SHALL\s+NOT\s+BE\s+LIABLE',
    r'INDEMNIFY\s+THE\s+CARRIER',
]


def _looks_like_bl_terms_page(text: str) -> bool:
    """
    True if `text` looks like a Bill of Lading "Terms and Conditions of
    Carriage" / reverse-side page.

    Detection rule: at least 3 distinct BL legal-clause keywords AND the
    text contains the word CARRIER at least twice (so a BL front side that
    happens to mention "Hague Rules" once doesn't get tagged).
    """
    if not text or len(text) < 200:
        return False
    upper = text.upper()
    if upper.count('CARRIER') < 2:
        return False
    hits = sum(1 for p in _BL_TERMS_KEYWORDS if re.search(p, upper))
    return hits >= 3


def _rule_based_classify(text: str) -> List[dict]:
    """Pre-classify document using keyword matching. Returns sorted matches."""
    upper = text.upper()
    scores = {}
    for doc_type, patterns in _DOC_INDICATORS.items():
        hits = sum(1 for p in patterns if re.search(p, upper))
        if hits > 0:
            score = hits / len(patterns)
            scores[doc_type] = round(score, 3)

    # ── Documentary Remittance / Covering Schedule override ──
    # Bank covering schedules (e.g. "We enclose documents related to above
    # referenced letter of credit ... Total Amount Claimed ... Presentation
    # Number ... Our Reference No.") are routinely misread as Beneficiary
    # Certificate because they sit on bank letterhead and contain
    # "We hereby ...". When ≥3 high-specificity covering-schedule signals
    # are present, force Documentary Remittance to the top so the VLM
    # prompt and downstream matching see the correct candidate first.
    dr_strong_signals = [
        r'WE\s+ENCLOSE\s+DOCUMENTS?',
        r'ENCLOSED\s+HEREWITH',
        r'DOCUMENTS?\s+ATTACHED',
        r'PRESENTATION\s+(?:NUMBER|NO\.?|DATE|AMOUNT)',
        r'TOTAL\s+(?:AMOUNT\s+)?CLAIMED',
        r'PRINCIPAL\s+AMOUNT\s+(?:CLAIMED|EUR|USD|GBP)',
        r'AMOUNTS?\s+CLAIMED\s*[:\n]',
        r'YOUR\s+DOCUMENTARY\s+CREDIT\s+NO',
        r'OUR\s+REFERENCE\s+NO',
        r'REMIT\s+FUNDS\s+TO\s+(?:OUR\s+)?CORRESPONDENT',
        r'QUOTING\s+OUR\s+REFERENCE',
        r'CLAIM\s+REIMBURSEMENT',
        r'COVERING\s+(?:LETTER|SCHEDULE)',
        r'DOCUMENTARY\s+REMITTANCE',
    ]
    dr_hits = sum(1 for p in dr_strong_signals if re.search(p, upper))

    # ── Courier / Express Waybill override ──
    # A DHL / FedEx / UPS / TNT / Aramex express envelope is a Courier
    # Receipt (or Air Waybill), NEVER a Documentary Remittance, even
    # though it travels alongside the document set. Detect courier
    # signals and force Courier Receipt to win — also suppressing the
    # DR override below to avoid the same misroute.
    courier_signals = [
        r'\bDHL\b', r'\bFEDEX\b', r'\bFED\s*EX\b', r'\bUPS\b',
        r'\bTNT\b', r'\bARAMEX\b', r'\bSF\s+EXPRESS\b',
        r'EXPRESS\s+ENVELOPE', r'\bWAYBILL\b',
        r'\bXPD\b', r'\bWPX\b',
        r'COURIER\s+(?:RECEIPT|WAYBILL|SERVICE)',
        r'TRACKING\s+(?:NUMBER|NO\.?)',
    ]
    # Also accept generic AWB signals as part of the same family
    awb_signals = [
        r'AIR\s*WAY\s*BILL', r'\bAWB\b', r'\bHAWB\b', r'\bMAWB\b',
        r'HOUSE\s+AIR\s*WAYBILL', r'MASTER\s+AIR\s*WAYBILL',
        r'AIRLINE', r'FLIGHT\s+NO',
        r'AIRPORT\s+OF\s+(?:DEPARTURE|DESTINATION)',
    ]
    awb_hits = sum(1 for p in awb_signals if re.search(p, upper))
    courier_hits = sum(1 for p in courier_signals if re.search(p, upper))

    # Air waybill OR courier waybill — same family. DHL/FedEx/UPS/TNT
    # are airline carriers that issue air waybills, not just couriers.
    # A FedEx Express shipping label IS an AWB — it serves as the
    # official shipping contract and tracking document. Both labels
    # are boosted equally; the VLM or LC requirement determines which
    # name appears. They are interchangeable for verification purposes
    # via DOC_TYPE_ALIASES in step14.
    is_courier_or_awb = (courier_hits >= 2) or (awb_hits >= 2) or (courier_hits + awb_hits >= 2)
    if is_courier_or_awb:
        scores['Courier Receipt'] = max(scores.get('Courier Receipt', 0), 0.99)
        scores['Airway Bill']     = max(scores.get('Airway Bill', 0), 0.99)
        # Suppress DR — courier/AWB labels are not covering schedules.
        if 'Documentary Remittance' in scores:
            scores['Documentary Remittance'] = min(scores['Documentary Remittance'], 0.10)
        # Don't run the DR override either.
        dr_hits = 0

    if dr_hits >= 3:
        # Override: this is a covering schedule. Boost score above all others.
        scores['Documentary Remittance'] = max(scores.get('Documentary Remittance', 0), 0.99)
        # Demote Beneficiary Certificate when DR signals dominate — a bank
        # covering schedule is NEVER a beneficiary's certificate even if
        # it contains "WE HEREBY" type language.
        if 'Beneficiary Certificate' in scores:
            scores['Beneficiary Certificate'] = min(scores['Beneficiary Certificate'], 0.10)

    return [
        {"document_name": name, "score": score}
        for name, score in sorted(scores.items(), key=lambda x: -x[1])
    ]


def _is_courier_or_awb_label(name: str) -> bool:
    """True if the document name refers to either an air waybill or a
    courier receipt — they are treated as one family."""
    if not name:
        return False
    n = name.upper()
    return (
        'COURIER' in n
        or 'AWB' in n
        or 'HAWB' in n
        or 'MAWB' in n
        or ('AIR' in n and ('WAY' in n or 'BILL' in n))
        or 'EXPRESS' in n and ('WAYBILL' in n or 'ENVELOPE' in n or 'DELIVERY' in n)
    )


def _match_type_to_requirement(doc_type: str, expected_docs: List[dict]) -> tuple:
    """Match a document type string to the expected docs list. Returns (index, name).

    Uses multiple matching strategies:
    1. Exact containment (either direction)
    2. Normalized containment (strip plurals, strip 'OF', etc.)
    3. Fuzzy word overlap (1+ significant words match)
    4. Air waybill ⇄ courier receipt equivalence
    """
    if not doc_type:
        return -1, ""
    dt_upper = doc_type.upper().strip()

    # AWB / Courier Receipt are the same family — match either label to
    # whichever the LC required.
    if _is_courier_or_awb_label(dt_upper):
        for i, ed in enumerate(expected_docs):
            if _is_courier_or_awb_label(ed.get('document_name', '')):
                return i, ed.get('document_name', '')

    # Normalize: strip plurals and common words for better matching
    def _normalize(s):
        s = s.upper()
        s = re.sub(r'\bBILLS?\b', 'BILL', s)
        s = re.sub(r'\bCERTIFICATES?\b', 'CERTIFICATE', s)
        s = re.sub(r'\bINVOICES?\b', 'INVOICE', s)
        s = re.sub(r'\bADVICES?\b', 'ADVICE', s)
        s = re.sub(r'\bEXCHANGES?\b', 'EXCHANGE', s)
        return s

    dt_norm = _normalize(dt_upper)

    for i, ed in enumerate(expected_docs):
        ed_name = ed.get('document_name', '').upper()
        ed_norm = _normalize(ed_name)
        # Exact containment
        if dt_upper in ed_name or ed_name in dt_upper:
            return i, ed.get('document_name', '')
        # Normalized containment
        if dt_norm in ed_norm or ed_norm in dt_norm:
            return i, ed.get('document_name', '')

    # ── Pharma regulatory form alias matching ──
    # Pakistani LCs reference "Form 7" or "Batch Certificate" interchangeably.
    # Similarly "Form 3" = "Form of Undertaking". Match either name to the
    # LC's required document regardless of which label was used.
    _PHARMA_ALIASES = {
        'FORM 7': {'BATCH CERTIFICATE', 'BATCH CERTIFICATION', 'FORM 7 (BATCH CERTIFICATE)'},
        'BATCH CERTIFICATE': {'FORM 7', 'FORM 7 (BATCH CERTIFICATE)', 'BATCH CERTIFICATION'},
        'FORM 3': {'FORM OF UNDERTAKING', 'DRUG REGISTRATION CERTIFICATE', 'IMPORT CERTIFICATE',
                   'FORM 3 (FORM OF UNDERTAKING)'},
        'FORM OF UNDERTAKING': {'FORM 3', 'FORM 3 (FORM OF UNDERTAKING)',
                                'DRUG REGISTRATION CERTIFICATE'},
        'CERTIFICATE OF ANALYSIS': {'ANALYSIS CERTIFICATE', 'ANALYTICAL CERTIFICATE',
                                    'TEST REPORT', 'TEST CERTIFICATE'},
    }
    # Check both directions: doc_type → expected and expected → doc_type
    for _alias_key, _alias_set in _PHARMA_ALIASES.items():
        if _alias_key in dt_upper or dt_upper in _alias_key:
            # doc_type matches an alias key — look for any alias value in expected
            for i, ed in enumerate(expected_docs):
                ed_upper2 = ed.get('document_name', '').upper()
                if ed_upper2 in _alias_set or _alias_key in ed_upper2 or ed_upper2 in _alias_key:
                    return i, ed.get('document_name', '')
                for av in _alias_set:
                    if av in ed_upper2 or ed_upper2 in av:
                        return i, ed.get('document_name', '')

    # Fuzzy: check if any significant words overlap (1+ is enough)
    dt_words = set(re.findall(r'[A-Z]{3,}', _normalize(dt_upper)))
    # Remove common filler words
    _FILLER = {'THE', 'AND', 'FOR', 'FROM', 'WITH', 'THEIR', 'MUST', 'SHOULD',
               'FULL', 'SET', 'ORIGINAL', 'COPY', 'DUPLICATE', 'ORDER'}
    dt_words -= _FILLER

    # P198cb — certificate-family guard. Certificates belong to mutually
    # exclusive semantic families. A Health Certificate (regulatory /
    # food-safety) MUST NOT match a Shipping Company Certificate (carrier
    # attestation) just because both contain the word "CERTIFICATE". The
    # marker tokens below identify the family of a cert name; if doc_type
    # names one family and expected names a different family, reject the
    # match regardless of any other key-word overlap.
    _CERT_FAMILIES = {
        'health':          {'HEALTH'},
        'phytosanitary':   {'PHYTOSANITARY', 'PHYTO'},
        'fumigation':      {'FUMIGATION', 'FUMIGATED', 'FUMIGATOR'},
        'halal':           {'HALAL'},
        'shipping_agent':  {'SHIPPING', 'AGENT', "AGENT'S", 'AGENTS',
                            'CARRIER', "CARRIER'S", 'CARRIERS',
                            'VESSEL', "VESSEL'S", 'SHIPOWNER',
                            "SHIPOWNER'S", 'NVOCC'},
        'beneficiary':     {'BENEFICIARY', "BENEFICIARY'S"},
        'origin':          {'ORIGIN'},
        'analysis':        {'ANALYSIS', 'ANALYTICAL'},
        'inspection':      {'INSPECTION', 'INSPECTED'},
        'survey':          {'SURVEY', 'SURVEYED'},
        'weight_quality':  {'WEIGHT', 'QUALITY'},
        'insurance':       {'INSURANCE', 'POLICY', 'COVER NOTE'},
    }

    def _cert_family(words):
        found = set()
        for fam, tokens in _CERT_FAMILIES.items():
            if words & tokens:
                found.add(fam)
        return found

    _is_cert_dt = 'CERTIFICATE' in dt_words
    dt_families = _cert_family(dt_words) if _is_cert_dt else set()

    _key_words = {'BILL', 'LADING', 'INVOICE', 'DRAFT', 'EXCHANGE',
                  'ADVICE', 'PACKING', 'WEIGHT', 'QUALITY', 'INSURANCE',
                  'FUMIGATION', 'PHYTOSANITARY'}
    # Collect all candidate matches with a score; pick the best at the end.
    # Score heuristic: +10 per shared non-filler non-CERTIFICATE token,
    # +5 per shared key_word, +3 for same-family cert pairing. Negative
    # for cross-family certs (→ skip).
    _best = (-1, "", -1)  # (score, name, index)
    for i, ed in enumerate(expected_docs):
        ed_upper = ed.get('document_name', '').upper()
        ed_words = set(re.findall(r'[A-Z]{3,}', _normalize(ed_upper))) - _FILLER
        overlap = dt_words & ed_words
        _is_cert_ed = 'CERTIFICATE' in ed_words

        if _is_cert_dt and _is_cert_ed:
            ed_families = _cert_family(ed_words)
            if dt_families and ed_families and not (dt_families & ed_families):
                continue  # cross-family — reject

        # Non-CERTIFICATE overlap tokens count most
        _non_cert_overlap = overlap - {'CERTIFICATE'}
        score = 10 * len(_non_cert_overlap)
        key_hits = overlap & _key_words
        score += 5 * len(key_hits)
        if _is_cert_dt and _is_cert_ed:
            ed_families = _cert_family(ed_words)
            if dt_families and ed_families and (dt_families & ed_families):
                score += 3
            # Bonus for family-specific tokens both present
            score += 2 * len(_non_cert_overlap)

        # Minimum floor: at least one non-filler overlap besides CERTIFICATE,
        # OR a same-family cert pair, OR a key-word match.
        _has_floor = (
            bool(_non_cert_overlap)
            or bool(key_hits)
            or (_is_cert_dt and _is_cert_ed
                and dt_families and _cert_family(ed_words)
                and (dt_families & _cert_family(ed_words)))
        )
        if not _has_floor:
            continue

        if score > _best[0]:
            _best = (score, ed.get('document_name', ''), i)

    if _best[2] >= 0:
        return _best[2], _best[1]
    return -1, ""


def _classify_single_packet(packet: dict, expected_docs: List[dict], packet_index: int) -> dict:
    """Classify a single shipping packet by sending image + GLM text to Qwen VLM."""
    start = time.time()

    # Get text content (GLM OCR text)
    glm_text = packet.get('cleaned_text', packet.get('raw_text', ''))
    if not glm_text:
        texts = packet.get('page_texts', [])
        glm_text = '\n'.join(str(t) for t in texts) if texts else ''

    pages = packet.get('pages', packet.get('original_pages', []))
    image_paths = packet.get('page_image_paths', packet.get('image_paths', []))

    # ── Respect prior structural classification from Step 3 ──
    # If Step 3 already classified this packet as a structural / non-document
    # page (Header Page, Blank Page, Endorsement Page, Covering Letter / Cover
    # Letter, Fusion Header, Endorsement, Back Page) then we MUST NOT let the
    # VLM force-fit it into one of the LC's required document slots. These
    # pages are bank letterheads, blank reverses, or endorsement stamps with
    # almost no content — when sent to the VLM with the LC's required-docs
    # list, the model hallucinates a match (e.g. promoting an OCBC header
    # page to "Draft Bill of Exchange") and pollutes the verification with
    # false REVIEW rows for amount / currency / drawee checks against an
    # empty page.
    #
    # Detection: read step 3's document_type from the packet itself OR from
    # the first page's classification (step 3 stores it on each page object).
    _STRUCTURAL_TYPES = {
        'header page', 'blank page', 'endorsement page', 'endorsement',
        'covering letter', 'cover letter', 'fusion header', 'back page',
    }
    _prior_dt = (packet.get('document_type') or '').strip().lower()
    if not _prior_dt and pages:
        _first = pages[0] if isinstance(pages[0], dict) else {}
        _prior_dt = (_first.get('document_type') or '').strip().lower()
    if _prior_dt in _STRUCTURAL_TYPES:
        # Keep the structural classification, skip VLM, do NOT match to any
        # LC requirement. The downstream verifier will see no packets in
        # the relevant document buckets and will report each genuinely
        # missing document ONCE via the F46A presence check.
        elapsed = time.time() - start
        keep_dt = (packet.get('document_type') or
                   (pages[0].get('document_type') if pages and isinstance(pages[0], dict) else '')
                   or 'Header Page')
        return asdict(ClassifiedPacket(
            packet_id=packet.get('packet_id', "packet_%03d" % packet_index),
            original_pages=pages if isinstance(pages, list) else [pages],
            page_image_paths=image_paths if isinstance(image_paths, list) else [image_paths],
            raw_text=packet.get('raw_text', ''),
            cleaned_text=packet.get('cleaned_text', glm_text),
            document_type=keep_dt,
            classification_status='informational',
            match_confidence=0.99,
            matched_requirement_index=-1,
            matched_requirement_name='',
            vlm_top_matches=[],
            vlm_reasoning=f'Structural page from Step 3 ({keep_dt}) — not matched to any LC requirement',
            document_summary='',
            document_number='',
            document_date='',
            document_amount='',
            stamps=[], signatures=[], seals=[], logos=[],
            copy_status='', copy_label='', marking_status='',
            issued_by='', lc_reference='',
            confidence=0.99,
            ambiguity_flag=False,
            ambiguity_notes='',
            elapsed_seconds=round(elapsed, 2),
            is_bl_terms_page=False,
            # Preserve Step 3 structured data through Step 8 so Step 14 can read it
            unified_summary=packet.get('unified_summary'),
            bl_subtype=packet.get('bl_subtype'),
            validation_status=packet.get('validation_status', 'valid'),
        ))

    # Build required docs list for prompt
    req_lines = []
    for i, d in enumerate(expected_docs):
        req_lines.append("%d. %s (%d originals, %d copies)" % (
            i + 1,
            d.get('document_name', 'Unknown'),
            d.get('originals_count', 0),
            d.get('copies_count', 0),
        ))
    required_docs_list = "\n".join(req_lines)

    # Rule-based pre-classification (fallback)
    rule_matches = _rule_based_classify(glm_text)

    # VLM classification
    vlm_result = None
    prompt_text = _CLASSIFICATION_PROMPT.format(
        glm_text=glm_text[:5000],
        required_docs_list=required_docs_list,
    )

    try:
        content_parts = []

        # Include first page image
        if image_paths:
            first_img = str(image_paths[0])
            if os.path.exists(first_img):
                try:
                    with open(first_img, 'rb') as f:
                        img_b64 = base64.b64encode(f.read()).decode('utf-8')
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,%s" % img_b64}
                    })
                except Exception:
                    pass

        content_parts.append({"type": "text", "text": prompt_text})

        resp = requests.post(QWEN_VLM_URL, json={
            "model": QWEN_VLM_MODEL,
            "messages": [{"role": "user", "content": content_parts}],
            "max_tokens": 2000,
            "temperature": 0.1,
        }, timeout=None)

        if resp.status_code == 200:
            result = resp.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                vlm_result = json.loads(json_match.group(0))

    except Exception as e:
        print("[Step 8] VLM classification failed for packet %d: %s" % (packet_index, e))

    # ── Forbid the VLM from returning Letter of Credit / MT-side types ──
    # Step 8 only ever sees SHIPPING documents (the MT-side packets are
    # split out upstream by server.py / the inline step04 logic). Smaller
    # VLMs (notably the 7B) sometimes hallucinate "Letter of Credit" as
    # the document type because they bleed the prompt text ("The LC
    # requires these documents:") into the answer instead of looking at
    # the actual page. When that happens, drop the VLM document_type and
    # fall back to step 3's prior classification (which is much more
    # reliable because step 3 doesn't have the LC required-docs list in
    # its prompt).
    _forbidden = {
        'letter of credit', 'lc', 'mt700', 'mt 700', 'mt707', 'mt 707',
        'mt799', 'mt 799', 'amendment', 'documentary credit',
    }
    if vlm_result:
        _vlm_dt = (vlm_result.get('document_type') or '').strip().lower()
        if _vlm_dt in _forbidden:
            # Strip the bad document_type but KEEP the metadata (number,
            # date, amount, stamps, signatures, etc.) — those are still
            # useful even when the type was hallucinated.
            print(f"[Step 8] VLM returned forbidden type {_vlm_dt!r} for packet {packet_index} — using prior step3 classification, keeping VLM metadata")
            vlm_result['document_type'] = ''

    # ── TRUST STEP 3'S CLASSIFICATION when it produced a meaningful type ──
    # Step 3 looks at each page IN ISOLATION (no LC required-docs context)
    # and classifies it from the title/heading visible on the image. That
    # makes it MUCH more reliable than step 8's VLM, which sees the LC
    # required-docs list and is prone to prompt-bleed on small models.
    #
    # If step 3 already produced a meaningful document_type (anything that
    # is not 'Unknown', 'LC', 'Amendment', or one of the structural types
    # already handled above), we use that as the canonical answer and only
    # let the step 8 VLM CONFIRM / REFINE it. The VLM result is then used
    # primarily for metadata extraction (number, date, amount, stamps,
    # signatures, copy_status, marking_status).
    _prior_dt_for_match = (packet.get('document_type') or '').strip()
    if not _prior_dt_for_match and pages:
        _first = pages[0] if isinstance(pages[0], dict) else {}
        _prior_dt_for_match = (_first.get('document_type') or '').strip()
    _prior_lower = _prior_dt_for_match.lower()
    _trust_prior = bool(
        _prior_dt_for_match
        and _prior_lower not in _forbidden
        and _prior_lower not in {'unknown', '', 'header page', 'blank page',
                                  'endorsement page', 'covering letter',
                                  'cover letter', 'fusion header', 'back page'}
    )

    # ── Combine results ──
    document_type = ""
    match_confidence = 0.0
    matched_index = -1
    matched_name = ""
    top_matches = rule_matches[:5]
    reasoning = ""
    classification_status = "unknown"

    # VLM-extracted visual elements
    stamps = []
    signatures = []
    seals = []
    logos = []
    copy_status = ""
    copy_label = ""
    marking_status = ""
    issued_by = ""
    lc_reference = ""
    document_summary = ""
    document_number = ""
    document_date = ""
    document_amount = ""

    # If the rule-based classifier triggered a high-confidence override
    # (Courier Receipt / Airway Bill, or Documentary Remittance via the
    # dedicated signal checks above), trust it over the VLM — these are
    # exactly the cases where the VLM tends to misroute (DHL labels →
    # Documentary Remittance, bank covering schedules → Beneficiary
    # Certificate).
    #
    # Air Waybill and Courier Receipt are operationally the same family;
    # both names get the 0.99 boost together when courier/AWB signals fire.
    # When that happens we resolve to whichever label the LC's expected
    # document list actually requested, falling back to "Airway Bill".
    rule_override_type = None
    if rule_matches:
        top_score = rule_matches[0].get('score', 0)
        top_names_at_99 = {m['document_name'] for m in rule_matches
                           if m.get('score', 0) >= 0.99}

        if top_score >= 0.99:
            if {'Courier Receipt', 'Airway Bill'} & top_names_at_99:
                # Resolve to whichever the LC asked for
                preferred = None
                for ed in expected_docs:
                    en = (ed.get('document_name') or '').upper()
                    if 'COURIER' in en:
                        preferred = 'Courier Receipt'; break
                    if 'AIR' in en and ('WAY' in en or 'BILL' in en):
                        preferred = 'Airway Bill'; break
                rule_override_type = preferred or 'Airway Bill'
            elif rule_matches[0]['document_name'] == 'Documentary Remittance':
                rule_override_type = 'Documentary Remittance'
            # P198aq — Strong-lexical-marker certificate rescue.
            # Some certificates have VERY distinctive heading /
            # content wording that's rarely used by any other doc
            # family: a page with "HEALTH CERTIFICATE" + "FIT FOR
            # HUMAN CONSUMPTION" is a Health Certificate, NOT a
            # Shipping Company Certificate (which the VLM has been
            # observed to call it when the issuer's letterhead
            # mentions ships/vessels). When the rule classifier
            # scores ≥ 0.99 for one of these high-specificity
            # certificate types, override the VLM label.
            elif rule_matches[0]['document_name'] in (
                'Health Certificate',
                'Phytosanitary Certificate',
                'Fumigation Certificate',
            ):
                rule_override_type = rule_matches[0]['document_name']

    if rule_override_type:
        document_type = rule_override_type
        match_confidence = 0.95
        reasoning = (
            f"Rule-based override: strong {rule_override_type} signals "
            f"(suppressing VLM classification: {vlm_result.get('document_type', '?') if vlm_result else 'n/a'})"
        )
        if vlm_result:
            # Still keep the visual elements VLM extracted
            document_summary = vlm_result.get('summary', '')
            document_number = vlm_result.get('document_number', '')
            document_date = vlm_result.get('date', '')
            document_amount = vlm_result.get('amount', '')
            raw_stamps = vlm_result.get('stamps', [])
            if isinstance(raw_stamps, list):
                stamps = [{"text": s.get("text", ""), "type": s.get("type", "rubber_stamp")}
                          for s in raw_stamps if isinstance(s, dict)]
            raw_sigs = vlm_result.get('signatures', [])
            if isinstance(raw_sigs, list):
                signatures = [{"description": s.get("description", ""), "type": s.get("type", "handwritten")}
                              for s in raw_sigs if isinstance(s, dict)]
            raw_seals = vlm_result.get('seals', [])
            if isinstance(raw_seals, list):
                seals = [{"description": s.get("description", "")}
                         for s in raw_seals if isinstance(s, dict)]
            raw_logos = vlm_result.get('logos', [])
            if isinstance(raw_logos, list):
                logos = [{"company_name": s.get("company_name", "")}
                         for s in raw_logos if isinstance(s, dict)]
            copy_status = vlm_result.get('copy_status', '')
            copy_label = vlm_result.get('copy_label', '')
            marking_status = vlm_result.get('marking_status', '')
            issued_by = vlm_result.get('issued_by', '')
            lc_reference = vlm_result.get('lc_reference', '')
        matched_index, matched_name = _match_type_to_requirement(document_type, expected_docs)
        classification_status = "matched_document" if matched_index >= 0 else "alien_document"
    elif vlm_result:
        # Default: take whatever the VLM said
        document_type = vlm_result.get('document_type', '')
        match_confidence = float(vlm_result.get('confidence', 0.0))
        reasoning = vlm_result.get('reasoning', '')
        # ── PRIOR-CLASSIFICATION OVERRIDE ──
        # If step 3 already produced a meaningful document_type for this
        # packet AND the VLM either (a) returned the forbidden Letter of
        # Credit value (now stripped to '') or (b) returned something
        # generic/empty, prefer step 3's classification. Step 3 is more
        # reliable because it sees the page in isolation without the LC
        # required-documents list bleeding into the prompt.
        if _trust_prior:
            _vlm_dt_lower = (document_type or '').strip().lower()
            if (not document_type
                    or _vlm_dt_lower in _forbidden
                    or _vlm_dt_lower in {'unknown', 'other', 'document'}):
                document_type = _prior_dt_for_match
                reasoning = f"Step 3 prior classification (preferred): {_prior_dt_for_match}"
                match_confidence = max(match_confidence, 0.90)

        # ── P198cx — Block "Shipping Company Certificate" force-fit ──
        # An LC's "Shipping Company Certificate" requirement is a
        # carrier-issued / agent-issued attestation about the VESSEL
        # (Institute Classification Clause coverage, vessel ownership,
        # port regulations, etc.). When the VLM sees this slot in the
        # LC's expected-docs list and a page it can't confidently
        # classify, it commonly force-fits independent-SURVEYOR
        # certificates (Last 3 Cargoes statements, Shelf Life
        # Certificates, Certificates of Analysis, Load Port Survey
        # Reports, etc.) into the SCC slot — they happen to mention
        # the vessel name but are issued by Control Union / SGS / Alfred
        # H Knight, not by the shipping company.
        #
        # Rule: if VLM returned "Shipping Company Certificate" but the
        # document body carries high-specificity markers for a
        # different certificate family — or step 3 already labelled
        # it with one of those specific names — override the VLM and
        # keep the specific type (which will then resolve as
        # alien_document via the standard matcher). These markers
        # virtually never appear on a genuine SCC.
        _vlm_dt_u = (document_type or '').strip().upper()
        if _vlm_dt_u == 'SHIPPING COMPANY CERTIFICATE':
            _glm_up = (glm_text or '').upper()
            _prior_up = (_prior_dt_for_match or '').upper()
            _SPECIFIC_CERT_MARKERS = (
                # Shelf life
                (r'\bSHELF\s+LIFE\s+CERTIFICATE\b', 'Shelf Life Certificate'),
                (r'\bSHELF\s+LIFE\b.*\b(?:EXPIRY|PRODUCTION)\s+DATE\b',
                 'Shelf Life Certificate'),
                # Last 3 / previous cargoes
                (r'\bLAST\s+\d\s+CARGOES\b', 'Last Cargoes Statement'),
                (r'\bLAST\s+THREE\s+CARGOES\b', 'Last Cargoes Statement'),
                (r'\bPREVIOUS\s+CARGOES?\b', 'Last Cargoes Statement'),
                (r'\bFOSFA\s+INTERNATIONAL\s+LIST\s+OF\s+BANNED\s+PREVIOUS\s+CARGOES\b',
                 'Last Cargoes Statement'),
                # Analysis / quality / survey (surveyor-issued — not SCC)
                (r'\bCERTIFICATE\s+OF\s+ANALYSIS\b', 'Certificate of Analysis'),
                (r'\bCERTIFICATE\s+OF\s+QUALITY\s+AND\s+WEIGHT\b',
                 'Certificate of Quality and Weight'),
                (r'\bLOAD\s+PORT\s+SURVEY\s+REPORT\b', 'Load Port Survey Report'),
                (r'\bDISCHARGE\s+SURVEY\s+REPORT\b', 'Discharge Survey Report'),
                (r'\bDRAUGHT\s+SURVEY\s+REPORT\b', 'Draught Survey Report'),
                # Other high-spec types
                (r'\bPHYTOSANITARY\b', 'Phytosanitary Certificate'),
                (r'\bHEALTH\s+CERTIFICATE\b', 'Health Certificate'),
                (r'\bFUMIGATION\s+CERTIFICATE\b', 'Fumigation Certificate'),
                (r'\bHALAL\s+CERTIFICATE\b', 'Halal Certificate'),
            )
            _override_to = None
            for _pat, _name in _SPECIFIC_CERT_MARKERS:
                if re.search(_pat, _glm_up) or re.search(_pat, _prior_up):
                    _override_to = _name
                    break
            # Also: if step 3's label directly names one of these
            # specific types (e.g. "SHELF LIFE CERTIFICATE"), keep it.
            if not _override_to and _prior_up:
                _PRIOR_SPECIFIC_NAMES = {
                    'SHELF LIFE CERTIFICATE': 'Shelf Life Certificate',
                    'CERTIFICATE OF ANALYSIS': 'Certificate of Analysis',
                    'CERTIFICATE OF QUALITY AND WEIGHT':
                        'Certificate of Quality and Weight',
                    'CERTIFICATE OF QUALITY': 'Certificate of Quality',
                    'LOAD PORT SURVEY REPORT': 'Load Port Survey Report',
                    'DISCHARGE SURVEY REPORT': 'Discharge Survey Report',
                    'DRAUGHT SURVEY REPORT': 'Draught Survey Report',
                    'LAST 3 CARGOES': 'Last Cargoes Statement',
                    'LAST CARGOES': 'Last Cargoes Statement',
                    'PHYTOSANITARY CERTIFICATE': 'Phytosanitary Certificate',
                    'HEALTH CERTIFICATE': 'Health Certificate',
                    'FUMIGATION CERTIFICATE': 'Fumigation Certificate',
                    'HALAL CERTIFICATE': 'Halal Certificate',
                }
                for _key, _name in _PRIOR_SPECIFIC_NAMES.items():
                    if _key in _prior_up:
                        _override_to = _name
                        break
            # Additional guard: if issued_by / letterhead names an
            # independent surveyor (Control Union, SGS, Alfred H
            # Knight, Intertek, Bureau Veritas, etc.) — definitely
            # NOT a shipping company certificate.
            if not _override_to:
                _SURVEYOR_NAMES = (
                    'CONTROL UNION', 'SGS', 'ALFRED H KNIGHT',
                    'INTERTEK', 'BUREAU VERITAS', 'SAYBOLT',
                    'CORNELDER', 'COTECNA', 'OMIC',
                )
                if any(s in _glm_up for s in _SURVEYOR_NAMES):
                    # Prefer step 3's label if informative; else
                    # fall back to a generic "Certificate".
                    _override_to = _prior_dt_for_match or 'Certificate'
            if _override_to:
                print(
                    f"[Step 8] P198cx: overriding VLM "
                    f"'Shipping Company Certificate' → '{_override_to}' "
                    f"for packet {packet_index} (surveyor-issued / "
                    f"high-specificity certificate, not SCC)"
                )
                document_type = _override_to
                reasoning = (
                    f"P198cx: VLM labelled 'Shipping Company "
                    f"Certificate' but document is {_override_to} "
                    f"(surveyor-issued, not a carrier/agent "
                    f"attestation). SCC false-match blocked."
                )
                match_confidence = max(match_confidence, 0.90)
        document_summary = vlm_result.get('summary', '')
        document_number = vlm_result.get('document_number', '')
        document_date = vlm_result.get('date', '')
        document_amount = vlm_result.get('amount', '')

        # Visual elements
        raw_stamps = vlm_result.get('stamps', [])
        if isinstance(raw_stamps, list):
            stamps = [{"text": s.get("text", ""), "type": s.get("type", "rubber_stamp")}
                      for s in raw_stamps if isinstance(s, dict)]

        raw_sigs = vlm_result.get('signatures', [])
        if isinstance(raw_sigs, list):
            signatures = [{"description": s.get("description", ""), "type": s.get("type", "handwritten")}
                          for s in raw_sigs if isinstance(s, dict)]

        raw_seals = vlm_result.get('seals', [])
        if isinstance(raw_seals, list):
            seals = [{"description": s.get("description", "")}
                     for s in raw_seals if isinstance(s, dict)]

        raw_logos = vlm_result.get('logos', [])
        if isinstance(raw_logos, list):
            logos = [{"company_name": s.get("company_name", "")}
                     for s in raw_logos if isinstance(s, dict)]

        copy_status = vlm_result.get('copy_status', '')
        copy_label = vlm_result.get('copy_label', '')
        marking_status = vlm_result.get('marking_status', '')
        issued_by = vlm_result.get('issued_by', '')
        lc_reference = vlm_result.get('lc_reference', '')

        # Match to expected docs
        matched_index, matched_name = _match_type_to_requirement(document_type, expected_docs)
        if matched_index >= 0:
            classification_status = "matched_document"
        else:
            classification_status = "alien_document"

    elif _trust_prior:
        # No VLM result and no rule override, but step 3 already produced
        # a meaningful document_type. Use that directly.
        document_type = _prior_dt_for_match
        match_confidence = 0.85
        reasoning = f"Step 3 prior classification (VLM unavailable): {_prior_dt_for_match}"
        matched_index, matched_name = _match_type_to_requirement(document_type, expected_docs)
        classification_status = "matched_document" if matched_index >= 0 else "alien_document"

    elif rule_matches:
        # Fallback to rule-based
        best = rule_matches[0]
        document_type = best['document_name']
        match_confidence = best['score'] * 0.8

        matched_index, matched_name = _match_type_to_requirement(document_type, expected_docs)
        if matched_index >= 0:
            classification_status = "matched_document"
        else:
            classification_status = "alien_document" if match_confidence > 0.3 else "unknown"

        reasoning = "Rule-based classification (VLM unavailable): top match %s (%.2f)" % (
            document_type, match_confidence)

    # Determine ambiguity
    ambiguity = False
    ambiguity_notes = ""
    if 0 < match_confidence < 0.6:
        ambiguity = True
        ambiguity_notes = "Classification confidence %.2f is low" % match_confidence
    if len(top_matches) >= 2:
        scores = [m.get('score', 0) for m in top_matches[:2]]
        if len(scores) == 2 and scores[0] > 0 and scores[1] > 0:
            if scores[0] - scores[1] < 0.2:
                ambiguity = True
                if ambiguity_notes:
                    ambiguity_notes += "; "
                ambiguity_notes += "Close scores: %s=%.2f vs %s=%.2f" % (
                    top_matches[0]['document_name'], scores[0],
                    top_matches[1]['document_name'], scores[1])

    # ── Detect BL terms-and-conditions-of-carriage page ──
    # If this packet's text looks like the BL reverse-side legal clauses
    # (Hague Rules, Carrier's liability, Paramount Clause, etc.), mark it
    # so the run()-level pass can decide whether other BLs in the set are
    # blank-back or full-form.
    is_bl_terms_page = _looks_like_bl_terms_page(glm_text)

    # ── Normalize pharma regulatory forms ──
    # Pakistani pharma LCs reference documents by their Drug Act form
    # number (Form 3, Form 7) but the VLM / Step 3 may classify them by
    # their descriptive title. Normalize here so that downstream matching
    # finds them regardless of which name the LC clause used.
    _dt_check = (document_type or '').upper()
    _pharma_norm = {
        # Form 7 = Batch Certificate (Drug Act Rule 14(d)(i))
        'BATCH CERTIFICATE': 'Form 7 (Batch Certificate)',
        'BATCH CERTIFICATION': 'Form 7 (Batch Certificate)',
        # Form 3 = Form of Undertaking / Drug Import Certificate
        'FORM OF UNDERTAKING': 'Form 3 (Form of Undertaking)',
        'DRUG REGISTRATION CERTIFICATE': 'Form 3 (Form of Undertaking)',
        'IMPORT CERTIFICATE': 'Form 3 (Form of Undertaking)',
        'DRUG IMPORT CERTIFICATE': 'Form 3 (Form of Undertaking)',
    }
    for _pn_key, _pn_val in _pharma_norm.items():
        if _pn_key in _dt_check:
            document_type = _pn_val
            break
    # Also catch "FORM 7" / "FORM 3" directly from VLM
    if re.match(r'^FORM\s*7\b', _dt_check):
        document_type = 'Form 7 (Batch Certificate)'
    elif re.match(r'^FORM\s*3\b', _dt_check):
        document_type = 'Form 3 (Form of Undertaking)'

    # ── Canonicalise document_type ──
    # The VLM returns inconsistent casing for the same logical document
    # (e.g. "Shipment Advice" on one copy and "SHIPMENT ADVICE" on the
    # next). When the packet is matched to an LC requirement, use the
    # LC's exact spelling as the canonical name so all copies of the
    # same document collapse to one type. Downstream consumers (step 9
    # reconciliation, step 13 row construction, step 14 verifier) then
    # don't double-count or duplicate rows for casing-only differences.
    if matched_index >= 0 and matched_name:
        document_type = matched_name
    else:
        # Title-case fallback when there's no LC match — at least
        # collapses casing variants of the same string.
        document_type = (document_type or '').strip()
        # P198az — Canonicalize to known Title-Case forms so downstream
        # code (step 12 targeting, step 14 matching, aggregation) sees
        # one consistent label instead of case variants like
        # "HEALTH CERTIFICATE" vs "Health Certificate" vs "Certificate
        # of Analysis" vs "CERTIFICATE OF ANALYSIS".
        _DOC_TYPE_CANON = {
            'HEALTH CERTIFICATE': 'Health Certificate',
            'PHYTOSANITARY CERTIFICATE': 'Phytosanitary Certificate',
            'FUMIGATION CERTIFICATE': 'Fumigation Certificate',
            'HALAL CERTIFICATE': 'Halal Certificate',
            'CERTIFICATE OF ORIGIN': 'Certificate of Origin',
            'WEIGHT CERTIFICATE': 'Weight Certificate',
            'QUALITY CERTIFICATE': 'Quality Certificate',
            'WEIGHT/QUALITY CERTIFICATE': 'Quality Certificate',
            'WEIGHT AND QUALITY CERTIFICATE': 'Quality Certificate',
            'CERTIFICATE OF WEIGHT AND QUALITY': 'Quality Certificate',
            'CERTIFICATE OF QUALITY AND WEIGHT': 'Quality Certificate',
            'CERTIFICATE OF ANALYSIS': 'Quality Certificate',
            'BENEFICIARY CERTIFICATE': 'Beneficiary Certificate',
            "BENEFICIARY'S CERTIFICATE": 'Beneficiary Certificate',
            'INSPECTION CERTIFICATE': 'Inspection Certificate',
            'INSPECTION REPORT': 'Inspection Certificate',
            'SHIPPING COMPANY CERTIFICATE': 'Shipping Company Certificate',
            'SURVEY REPORT': 'Survey Report',
            'FULL LOADING SURVEY REPORT': 'Full Loading Survey Report',
            'COMMERCIAL INVOICE': 'Commercial Invoice',
            'BILL OF LADING': 'Bill of Lading',
            'DRAFT BILL OF EXCHANGE': 'Draft Bill of Exchange',
            'BILL OF EXCHANGE': 'Draft Bill of Exchange',
            'DOCUMENTARY REMITTANCE': 'Documentary Remittance',
            'COVERING LETTER': 'Documentary Remittance',
            'COVERING SCHEDULE': 'Documentary Remittance',
            'PACKING LIST': 'Packing List',
            'PACKING NOTE': 'Packing List',
            'PACKING SLIP': 'Packing List',
            'AIR WAYBILL': 'Airway Bill',
            'AIRWAY BILL': 'Airway Bill',
            'COURIER RECEIPT': 'Courier Receipt',
            'SHIPMENT ADVICE': 'Shipment Advice',
            'SHIPPING ADVICE': 'Shipment Advice',
        }
        if document_type:
            _key = document_type.upper().strip()
            if _key in _DOC_TYPE_CANON:
                document_type = _DOC_TYPE_CANON[_key]
            elif document_type.isupper():
                # Unknown all-uppercase label — fall back to Title Case
                document_type = document_type.title()

    # ── P198dx — Preserve "L/C Issuing Bank" label in summary ──
    # When the source text uses the labelled form "L/C Issuing Bank"
    # (or "LC Issuing Bank"), the VLM-generated document_summary
    # often drops the "L/C" qualifier and just writes "Issuing
    # Bank: ...", which loses information bank checkers rely on
    # (the issuing bank in an L/C context is specifically the
    # CREDIT-issuing bank, not any bank that issued the document
    # in question). Restore the qualifier in the summary when the
    # source text supports it.
    if document_summary and glm_text:
        _src_has_lc_issuing = bool(re.search(
            r'\bL\s*/?\s*C\s+ISSUING\s+BANK\b', glm_text, re.IGNORECASE))
        if _src_has_lc_issuing:
            _new_sum = re.sub(
                r'(?<!L/C\s)(?<!L/C-\s)(?<!LC\s)(?<![A-Z])'
                r'\bIssuing\s+Bank\b',
                'L/C Issuing Bank',
                document_summary,
            )
            # Avoid double-prefixing if a substitution already had L/C
            _new_sum = re.sub(r'\bL/C\s+L/C\s+Issuing\s+Bank\b',
                              'L/C Issuing Bank', _new_sum)
            if _new_sum != document_summary:
                document_summary = _new_sum

    # ── P198dx — "Detailed Message" recognition ──
    # F46A clauses such as "BENEFICIARY CERTIFICATE CERTIFYING THAT
    # THEY HAVE SENT DETAILED MESSAGE DIRECTLY TO THE APPLICANT BY
    # FAX..." are satisfied by a beneficiary-issued fax / email titled
    # "DETAILED MESSAGE" that carries vessel / B/L / ETA / value /
    # delivery-agent details together with a "WE CERTIFY" line.
    # Step03 / VLM often lands on "Shipment Advice" or "Beneficiary
    # Certificate" for these pages — both correct in spirit but
    # neither matches the user-visible label the LC clause expects.
    # When the page carries the explicit "DETAILED MESSAGE" header
    # AND beneficiary-certification language AND fax / shipment
    # evidence, upgrade the doc-type so step14 routes it to either
    # a Beneficiary Certificate clause or a Shipment Advice clause
    # via the P198dx aliases.
    if document_type in (
        'Shipment Advice', 'Shipping Advice',
        'Beneficiary Certificate', 'Beneficiary\'s Certificate',
        'Documentary Remittance', 'Covering Letter', 'Covering Schedule',
    ):
        _ux = (glm_text or '').upper()
        _has_dm_header = bool(re.search(
            r'(?:^|\n)\s*DETAIL(?:ED)?\s+MESSAGE\b',
            _ux, re.MULTILINE))
        _has_bene_cert = bool(re.search(
            r'\bWE\s+CERTIFY\b|CERTIFY(?:ING)?\s+(?:THE\s+)?GOODS\s+'
            r'(?:TO\s+BE\s+)?(?:OF|ARE\s+OF)|'
            r'\bWE\s+ARE\s+PLEASED\s+TO\s+INFORM\s+YOU\s+OF\s+OUR\s+SHIPMENT\b',
            _ux))
        _has_fax_or_email = bool(re.search(
            r'\bFAX(?:\s*(?:NO\.?|NUMBER|#))?\s*[:.\d]|'
            r'\bDIRECT(?:LY)?\s+TO\s+THE\s+APPLICANT\s+BY\s+FAX|'
            r'\bSENT\s+BY\s+FAX\b',
            _ux))
        _has_shipment_evidence = bool(re.search(
            r'\bB[/\s]*L\s+(?:NO\.?|NUMBER)|\bBILL\s+OF\s+LADING|'
            r'\bVESSEL\b|\bETA\b|\bETD\b|\bSHIPPED\s+ON\s+BOARD\b',
            _ux))
        if (_has_dm_header
            and _has_shipment_evidence
            and (_has_bene_cert or _has_fax_or_email)):
            try:
                print(
                    f"  [P198dx detailed-message] "
                    f"{packet.get('packet_id','?')} "
                    f"({document_type} -> Detailed Message): "
                    f"header={_has_dm_header}, bene_cert={_has_bene_cert}, "
                    f"fax={_has_fax_or_email}, shipment_evidence="
                    f"{_has_shipment_evidence}"
                )
            except Exception:
                pass
            document_type = 'Detailed Message'

    # ── P198el — "Document Arrival Notice" recognition ──
    # UBL / HBL / other issuing-bank notifications to the applicant
    # that say "DOCUMENT ARRIVAL NOTICE" + "PLEASE BE ADVISED THAT
    # WE HAVE RECEIVED THE ORIGINAL DOCUMENTS FROM ..." + a
    # DISCREPANCIES list are NOT Documentary Remittances. They are
    # the issuing bank's discrepancy / arrival notification to the
    # applicant. The VLM at step08 force-fits them to "Documentary
    # Remittance" because that's the closest doc-type in the LC's
    # required-documents list. Detect the structural signals and
    # keep this as its own canonical type so downstream verification
    # doesn't mis-anchor charges-on-DR / presentation-period checks
    # on a non-DR page.
    if document_type in (
        'Documentary Remittance', 'Document Remittance',
        'Covering Letter', 'Covering Schedule',
        'Document Arrival Notice',
    ):
        _ux2 = (glm_text or '').upper()
        _has_arrival_header = bool(re.search(
            r'(?:^|\n)\s*DOCUMENT(?:S)?\s+ARRIVAL\s+NOTICE\b',
            _ux2, re.MULTILINE,
        ))
        _has_received_from = bool(re.search(
            r'(?:RECEIVED\s+THE\s+ORIGINAL\s+DOCUMENTS|'
            r'WE\s+HAVE\s+RECEIVED\s+THE\s+(?:ORIGINAL\s+)?DOCUMENTS)',
            _ux2,
        ))
        _has_discrepancies_block = bool(re.search(
            r'\bDISCREPANC(?:Y|IES)\s+(?:NOTED|FOUND|OBSERVED)\b'
            r'|(?:^|\n)\s*DISCREPANCIES?\s*[:\-]?\s*\n',
            _ux2, re.MULTILINE,
        ))
        if (_has_arrival_header
            and (_has_received_from or _has_discrepancies_block)):
            try:
                print(
                    f"  [P198el arrival-notice] "
                    f"{packet.get('packet_id','?')} "
                    f"({document_type} -> Document Arrival Notice): "
                    f"header={_has_arrival_header}, "
                    f"received_from={_has_received_from}, "
                    f"discrepancies={_has_discrepancies_block}"
                )
            except Exception:
                pass
            document_type = 'Document Arrival Notice'

    # ── P198dp — Documentary Remittance false-positive guard ──
    # A genuine bank covering schedule / Documentary Remittance shows
    # bank letterhead AND payment-claim language ("WE ENCLOSE FOR
    # NEGOTIATION/PAYMENT", "TOTAL AMOUNT CLAIMED", "PRESENTATION
    # NUMBER", "OUR/YOUR REFERENCE NO", "REMIT FUNDS", "CLAIM
    # REIMBURSEMENT", etc.). Pages that arrive at "Documentary
    # Remittance" via aggressive canonicalization (step03 maps any
    # "Covering Letter" → "Document Remittance"; step08 maps
    # "COVERING LETTER" → "Documentary Remittance") — including
    # seller/freight-forwarder email cover notes that just say
    # "Attached doc for your reference. Thanks!" — must NOT keep
    # the DR label since they are not the negotiating bank's
    # covering schedule. Without this guard, downstream checks
    # (charges-on-DR, presentation-period-on-DR, LC-expiry-on-DR)
    # mis-anchor on the wrong page and produce false PASS/FAIL.
    if document_type == 'Documentary Remittance':
        _u = (glm_text or '').upper()
        _dr_real_signals = [
            r'WE\s+ENCLOSE\s+(?:THE\s+)?(?:FOLLOWING\s+|ABOVE\s+)?'
            r'DOCUMENTS?(?:\s+(?:FOR|DRAWN))?',
            r'WE\s+ARE\s+PLEASED\s+TO\s+ENCLOSE',
            r'WE\s+HEREBY\s+ENCLOSE',
            r'ENCLOSED\s+HEREWITH',
            r'PRESENTATION\s+(?:NUMBER|NO\.?|DATE|AMOUNT)',
            r'TOTAL\s+(?:AMOUNT\s+)?CLAIMED',
            r'PRINCIPAL\s+AMOUNT\s+(?:CLAIMED|EUR|USD|GBP)',
            r'AMOUNTS?\s+CLAIMED\s*[:\n]',
            r'YOUR\s+DOCUMENTARY\s+CREDIT\s+NO',
            r'OUR\s+REFERENCE\s+NO',
            r'REMIT\s+FUNDS\s+TO\s+(?:OUR\s+)?CORRESPONDENT',
            r'(?:UPON|FOR)\s+SETTLEMENT\s+PLEASE\s+REMIT',
            r'QUOTING\s+OUR\s+REFERENCE',
            r'CLAIM\s+REIMBURSEMENT',
            r'COVERING\s+(?:LETTER|SCHEDULE)',
            r'DOCUMENTARY\s+REMITTANCE',
            r'L/?C\s+BILLS?\s+SCHEDULE',
            r'(?:DOCUMENT|EXPORT\s+DC)\s+PRESENTATION\s+SCHEDULE',
            r'SCHEDULE\s+OF\s+PRESENTATION',
            r'BILLS?\s+REMITTANCE\s+LETTER',
            # Bank-presentation structural / short-form signals (Habib
            # Canadian, Maybank etc. write 'Our Ref.', 'Your DC Ref.',
            # 'L/C Issuing Bank', 'Reimbursing Bank', 'Mail To' as
            # labelled fields of the covering schedule).
            r'\bL/?C\s+ISSUING\s+BANK\b',
            r'\bREIMBURSING\s+BANK\b',
            r'\bYOUR\s+DC\s+REF\b',
            r'\bOUR\s+REF\.\s',
            r'\bPAYMENT\s+INSTRUCTION\b',
            r'\bBILL\s+AMOUNT\b',
            r'DOCUMENTS?\s+SENT\s+TO\s+YOU\s+ON\s+APPROVAL',
            r'DRAWING\s+AMOUNT\s+(?:HAS\s+BEEN\s+)?(?:DULY\s+)?ENDORSED',
            r'PRESENTATION\s+IS\s+SUBJECT\s+TO',
            r'ADVISING\s+CHARGES?\s+AND\s+CONFIRMATION\s+CHARGES?',
        ]
        _signal_count = sum(1 for _p in _dr_real_signals
                            if re.search(_p, _u))
        # Bank letterhead — broad pattern: any well-known bank name OR
        # a SWIFT BIC line at the top of the page.
        _bank_letterhead = bool(re.search(
            r'\b(?:MAYBANK|MALAYAN\s+BANKING|BANK\s+AL\s+HABIB|'
            r'HABIB\s+BANK|HBL\b|UBL\b|UNITED\s+BANK\s+LIMITED|'
            r'MEEZAN\s+BANK|FAYSAL\s+BANK|MCB\b|ALLIED\s+BANK|'
            r'STANDARD\s+CHARTERED|HSBC|CITIBANK|JP\s*MORGAN|'
            r'J\.P\.\s*MORGAN|BARCLAYS|DEUTSCHE\s+BANK|RBC\b|'
            r'ROYAL\s+BANK|BNP\s+PARIBAS|COMMERZBANK|MIZUHO|'
            r'BANK\s+OF\s+CHINA|ICBC|BANCO\b|CHINA\s+CONSTRUCTION|'
            r'WELLS\s+FARGO|BANK\s+OF\s+AMERICA|UNICREDIT|'
            r'SOCIETE\s+GENERALE|CREDIT\s+SUISSE|UBS\b|'
            r'NATIONAL\s+BANK|COMMERCIAL\s+BANK)\b', _u))
        _swift_header = bool(re.search(
            r'\bSWIFT\s*:\s*[A-Z]{6,11}\b', _u))
        # Email cover note signal: From:<email> + Subject: together
        _is_email = bool(
            re.search(r'\bFROM\s*:\s*[^\n]*@', _u)
            and re.search(r'\bSUBJECT\s*:', _u)
        )

        # Demote unless real bank covering schedule shape:
        #   - non-email: ≥2 strong signals, OR bank letterhead + ≥1
        #     signal, OR SWIFT header + ≥1 signal.
        #   - email cover note (From:<email> + Subject:): require
        #     ≥3 strong signals. Emails routinely mention banks /
        #     L/C issuing bank as reference text, so 1 signal +
        #     bank-name match is not enough to call them DR.
        if _is_email:
            _is_real_dr = _signal_count >= 3
        else:
            _is_real_dr = (
                _signal_count >= 2
                or (_bank_letterhead and _signal_count >= 1)
                or (_swift_header and _signal_count >= 1)
            )
        if not _is_real_dr:
            _was = document_type
            # P198dy — Demote target depends on shape:
            #   - Email cover note (From:<email> + Subject:) referencing
            #     an LC → "Shipment Advice". Bank workflow treats these
            #     forwarder / logistics-company emails as the shipment
            #     advice for that LC, even when the email body carries
            #     just "Attached doc for your reference" + LC ref +
            #     L/C Issuing Bank line. The actual shipment evidence
            #     lives on the adjacent (attachment) page which is
            #     ALREADY classified as Shipment Advice; this label
            #     keeps the email together with that flow so F46A
            #     shipment-advice checks don't ignore it.
            #   - Anything else (plain transmittal letters, mis-routed
            #     non-email pages) → "Covering Letter".
            if _is_email:
                document_type = 'Shipment Advice'
            else:
                document_type = 'Covering Letter'
            try:
                print(
                    f"  [P198dy DR-guard] {packet.get('packet_id','?')} "
                    f"({_was} -> {document_type}): signals={_signal_count}, "
                    f"bank_letterhead={_bank_letterhead}, "
                    f"swift_header={_swift_header}, email={_is_email}"
                )
            except Exception:
                pass

    elapsed = time.time() - start

    classified = ClassifiedPacket(
        packet_id=packet.get('packet_id', "packet_%03d" % packet_index),
        original_pages=pages if isinstance(pages, list) else [pages],
        page_image_paths=image_paths if isinstance(image_paths, list) else [image_paths],
        raw_text=packet.get('raw_text', ''),
        cleaned_text=packet.get('cleaned_text', glm_text),
        document_type=document_type,
        classification_status=classification_status,
        match_confidence=match_confidence,
        matched_requirement_index=matched_index,
        matched_requirement_name=matched_name,
        vlm_top_matches=top_matches,
        vlm_reasoning=reasoning,
        document_summary=document_summary,
        document_number=document_number,
        document_date=document_date,
        document_amount=document_amount,
        stamps=stamps,
        signatures=signatures,
        seals=seals,
        logos=logos,
        copy_status=copy_status,
        copy_label=copy_label,
        marking_status=marking_status,
        issued_by=issued_by,
        lc_reference=lc_reference,
        confidence=match_confidence,
        ambiguity_flag=ambiguity,
        ambiguity_notes=ambiguity_notes,
        elapsed_seconds=round(elapsed, 2),
        is_bl_terms_page=is_bl_terms_page,
        # Preserve Step 3 structured data so Step 14 can read it unchanged
        unified_summary=packet.get('unified_summary'),
        bl_subtype=packet.get('bl_subtype'),
        validation_status=packet.get('validation_status', 'valid'),
    )

    return asdict(classified)


# ── Main Run Function ──

def run(step3_result: dict, step7_result: dict, output_dir: str = None, progress_callback=None) -> dict:
    """
    Execute Step 8: Classify shipping packets against expected document list.

    Args:
        step3_result: Output from Step 3 with 'shipping_packets' list
        step7_result: Output from Step 7 with 'required_documents' list
        output_dir: Directory to save results
        progress_callback: Optional callback for progress updates

    Returns:
        dict with 'classified_packets', 'summary', 'required_documents', 'elapsed_seconds'
    """
    def _progress(msg):
        if progress_callback:
            progress_callback("[Step 8] %s" % msg)
        print("[Step 8] %s" % msg)

    start_time = time.time()

    packets = step3_result.get('shipping_packets', step3_result.get('packets', []))
    required_docs = step7_result.get('required_documents', [])

    _progress("Classifying %d shipping packets against %d expected documents..." % (
        len(packets), len(required_docs)))

    if not packets:
        return {
            'classified_packets': [],
            'summary': {'total': 0, 'matched': 0, 'alien': 0, 'extra': 0, 'unknown': 0},
            'required_documents': required_docs,
            'elapsed_seconds': 0,
        }

    # Classify packets concurrently
    classified_packets = [None] * len(packets)

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_VLM) as executor:
        futures = {}
        for idx, packet in enumerate(packets):
            future = executor.submit(_classify_single_packet, packet, required_docs, idx)
            futures[future] = idx

        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                classified_packets[idx] = result
                status = result.get('classification_status', 'unknown')
                doc_type = result.get('document_type', '?')
                conf = result.get('match_confidence', 0)
                copy = result.get('copy_status', '')
                marking = result.get('marking_status', '')
                _progress("  Packet %d: %s [%s] (%.2f) copy=%s marking=%s" % (
                    idx, doc_type, status, conf, copy, marking))
            except Exception as e:
                _progress("  Packet %d: ERROR - %s" % (idx, e))
                classified_packets[idx] = asdict(ClassifiedPacket(
                    packet_id="packet_%03d" % idx,
                    classification_status="unknown",
                    ambiguity_flag=True,
                    ambiguity_notes="Classification error: %s" % e,
                ))

    # Post-process: detect extra documents
    type_counts = {}
    for cp in classified_packets:
        if cp and cp.get('classification_status') == 'matched_document':
            mi = cp.get('matched_requirement_index', -1)
            if mi >= 0:
                type_counts[mi] = type_counts.get(mi, 0) + 1

    for cp in classified_packets:
        if cp and cp.get('classification_status') == 'matched_document':
            mi = cp.get('matched_requirement_index', -1)
            if mi >= 0 and mi < len(required_docs):
                expected_total = (required_docs[mi].get('originals_count', 0) +
                                  required_docs[mi].get('copies_count', 0))
                if expected_total > 0 and type_counts.get(mi, 0) > expected_total:
                    cp['classification_status'] = 'extra_document'
                    cp['ambiguity_flag'] = True
                    notes = cp.get('ambiguity_notes', '')
                    extra_note = "More instances (%d) than expected (%d)" % (
                        type_counts[mi], expected_total)
                    cp['ambiguity_notes'] = ("%s; %s" % (notes, extra_note)).strip('; ')

    # ── Bill of Lading: blank-back vs full-form resolution ──
    #
    # Rule:
    #   • A BL packet whose text contains the carriage T&Cs is a full-form BL.
    #   • A BL packet whose text does NOT contain the T&Cs is treated as
    #     short-form / blank-back UNLESS some OTHER packet in the same
    #     classification set carries those T&Cs — in which case the BL is
    #     a full-form BL whose terms were printed on a separate sheet.
    #
    # The downstream audit (step12 / step15) only raises a discrepancy when
    # the LC explicitly forbids short-form / blank-back BLs; this flag tells
    # it whether to fire.
    has_terms_page_in_set = any(
        p and p.get('is_bl_terms_page') for p in classified_packets
    )

    def _is_bl(cp: dict) -> bool:
        if not cp:
            return False
        dt = (cp.get('document_type') or '').upper()
        mr = (cp.get('matched_requirement_name') or '').upper()
        return ('BILL OF LADING' in dt or 'BILL OF LADING' in mr
                or dt == 'BILL OF LADING' or mr == 'BILL OF LADING')

    for cp in classified_packets:
        if not _is_bl(cp):
            continue
        if cp.get('is_bl_terms_page'):
            # This packet IS the T&C sheet — not a real BL by itself.
            cp['bl_short_form_status'] = 'full_form'
            cp['has_bl_terms_pages_in_set'] = True
            continue

        cp['has_bl_terms_pages_in_set'] = has_terms_page_in_set

        own_text = (cp.get('cleaned_text') or cp.get('raw_text') or '')
        own_has_terms = _looks_like_bl_terms_page(own_text)

        if own_has_terms:
            cp['bl_short_form_status'] = 'full_form'
        elif has_terms_page_in_set:
            # Carriage terms supplied on a separate page in the same set →
            # the BL is effectively a full-form BL.
            cp['bl_short_form_status'] = 'full_form'
        else:
            # No T&Cs anywhere in the document set — treat as short-form /
            # blank-back. ((v) still ACCEPTS this unless
            # the LC explicitly forbids it; the discrepancy is raised
            # later by the cross-clause audit when that LC clause exists.)
            cp['bl_short_form_status'] = 'short_form'

    # Summary
    summary = {
        'total': len(classified_packets),
        'matched': sum(1 for p in classified_packets if p and p.get('classification_status') == 'matched_document'),
        'alien': sum(1 for p in classified_packets if p and p.get('classification_status') == 'alien_document'),
        'extra': sum(1 for p in classified_packets if p and p.get('classification_status') == 'extra_document'),
        'unknown': sum(1 for p in classified_packets if p and p.get('classification_status') == 'unknown'),
        'ambiguous': sum(1 for p in classified_packets if p and p.get('ambiguity_flag')),
        'bl_full_form': sum(1 for p in classified_packets
                            if p and p.get('bl_short_form_status') == 'full_form'),
        'bl_short_form': sum(1 for p in classified_packets
                             if p and p.get('bl_short_form_status') == 'short_form'),
    }

    elapsed = time.time() - start_time

    # Save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        result_file = os.path.join(output_dir, 'step08_result.json')
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({
                'step': 8,
                'step_name': 'Shipping Document Classification',
                'total_packets': len(classified_packets),
                'summary': summary,
                'elapsed_seconds': round(elapsed, 2),
                'classified_packets': classified_packets,
                'required_documents_used': required_docs,
            }, f, indent=2, ensure_ascii=False)

    _progress("Step 8 complete: %d matched, %d alien, %d extra, %d unknown in %.1fs" % (
        summary['matched'], summary['alien'], summary['extra'], summary['unknown'], elapsed))

    return {
        'classified_packets': classified_packets,
        'summary': summary,
        'required_documents': required_docs,
        'elapsed_seconds': round(elapsed, 2),
    }


if __name__ == '__main__':
    import sys as _main_sys
    if len(_main_sys.argv) < 3:
        print("Usage: python step08_shipping_classification.py <step03_result.json> <step07_result.json>")
        _main_sys.exit(1)
    with open(_main_sys.argv[1], 'r', encoding='utf-8') as f:
        step3 = json.load(f)
    with open(_main_sys.argv[2], 'r', encoding='utf-8') as f:
        step7 = json.load(f)
    result = run(step3, step7, output_dir=os.path.dirname(_main_sys.argv[1]))
    print("\nResult: %s" % result['summary'])
