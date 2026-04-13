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
   • A "Short Form Bill of Lading" / "Blank Back Bill of Lading" / "Liner Bill of Lading" / "Charter Party Bill of Lading" / "Combined Transport Bill of Lading" / "Multimodal Bill of Lading" is STILL a "Bill of Lading" (per UCP 600 Art 20(a)(v)). It is NOT an "Airway Bill" and NOT a "Courier Receipt" — it just lacks the full carriage terms on the reverse side.
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
        # all still classified as "Bill of Lading" (UCP 600 Art 20(a)(v)).
        r'SHORT\s+FORM\s+BILL\s+OF\s+LADING',
        r'BLANK\s+BACK\s+BILL\s+OF\s+LADING',
        r'LINER\s+BILL\s+OF\s+LADING',
        r'CHARTER\s+PARTY\s+BILL\s+OF\s+LADING',
        r'COMBINED\s+TRANSPORT\s+BILL\s+OF\s+LADING',
        r'MULTIMODAL\s+(?:TRANSPORT\s+)?BILL\s+OF\s+LADING',
        r'(?:CONDITIONS|TERMS)\s+OF\s+CARRIAGE\s+(?:ARE\s+)?(?:REFERRED|AVAILABLE)',
    ],
    # ── Airway Bill / Courier Receipt are treated as the SAME category ──
    # Both are single-piece air/courier transport documents (UCP 600 Art 23 /
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
# detailed carriage terms. UCP 600 Art 20(a)(v) accepts these unconditionally
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
    for i, ed in enumerate(expected_docs):
        ed_upper = ed.get('document_name', '').upper()
        ed_words = set(re.findall(r'[A-Z]{3,}', _normalize(ed_upper))) - _FILLER
        overlap = dt_words & ed_words
        # Key document words: BILL, LADING, INVOICE, CERTIFICATE, DRAFT, ADVICE, PACKING, WEIGHT, QUALITY
        _key_words = {'BILL', 'LADING', 'INVOICE', 'CERTIFICATE', 'DRAFT', 'EXCHANGE',
                      'ADVICE', 'PACKING', 'WEIGHT', 'QUALITY', 'INSURANCE', 'FUMIGATION',
                      'PHYTOSANITARY', 'SHIPPING', 'AGENT'}
        key_overlap = overlap & _key_words
        if key_overlap:
            return i, ed.get('document_name', '')
        if len(overlap) >= 2:
            return i, ed.get('document_name', '')
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
        }, timeout=VLM_TIMEOUT)

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
        if document_type and document_type.isupper():
            document_type = document_type.title()

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
            # blank-back. (UCP 600 Art 20(a)(v) still ACCEPTS this unless
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
