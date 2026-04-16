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


# ── VLM Classification Prompt ──

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
A page from a negotiating bank (e.g. Standard Chartered) showing "L/C BILLS SCHEDULE" or
"COVERING SCHEDULE" or "DOCUMENT PRESENTATION" or "EXPORT DC DOCUMENT PRESENTATION SCHEDULE"
with document list, CBC number, amount, maturity, and instructions is
a "Documentary Remittance" (also known as Covering Schedule, L/C Bills Schedule, or Document Presentation).

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
            resp = requests.post(QWEN_VLM_URL, json=payload, timeout=VLM_TIMEOUT)

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

        # Normalize doc type
        doc_type_lower = doc_type.lower().strip()
        if doc_type_lower in ('lc', 'letter of credit', 'swift message', 'mt700'):
            doc_type = 'LC'
        elif doc_type_lower in ('amendment', 'mt707', 'lc amendment'):
            doc_type = 'Amendment'

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

    _progress(f"Sending ALL {len(vlm_tasks)} pages to VLM for classification + visual detection...")

    # Run VLM classification concurrently
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_VLM) as executor:
        futures = {}
        for pg_num, img_path, text in vlm_tasks:
            future = executor.submit(_classify_page_vlm, pg_num, img_path, text)
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

        if is_cont and _prev_type:
            # Check if this page has its own "Page X of Y" marker
            _has_page_xy = pg_num in _page_of_total

            if _has_page_xy:
                # This page has "Page X of Y" — let multi-page grouping handle it.
                # DON'T inherit from previous PDF page (could be a different document).
                # But DO update _prev_type so NEXT continuation pages can inherit.
                _prev_type = doc_type
                _progress(f"  Page {pg_num}: has Page {_page_of_total[pg_num][0]} of {_page_of_total[pg_num][1]} — skipping inheritance, will use multi-page grouping")
            else:
                # No "Page X of Y" — safe to inherit previous page's type
                if doc_type_lower != _prev_type.lower():
                    _progress(f"  Page {pg_num}: CONTINUATION type fix: '{doc_type}' → '{_prev_type}' (inherits from previous page)")
                    cls['document_type'] = _prev_type
                # Keep _prev_type unchanged (the inherited type)
        else:
            # Not a continuation — this becomes the new "previous type"
            _prev_type = doc_type

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

        # Rule 1: BL — absorb nearby BL T&C and Attach List pages
        if dt in _bl_types:
            # 1a: Absorb ONE nearest T&C
            for j, other in enumerate(packets):
                if j in _consumed or j == i:
                    continue
                odt = other.document_type.lower().strip()
                if odt in _bl_tc_types:
                    pkt.page_numbers.extend(other.page_numbers)
                    pkt.pages.extend(other.pages)
                    pkt.stamps.extend(other.stamps)
                    pkt.signatures.extend(other.signatures)
                    pkt.seals.extend(other.seals)
                    _consumed.add(j)
                    _progress(f"  Merged {other.packet_id} (BL T&C pg {other.page_numbers}) into {pkt.packet_id} (BL pg {pkt.page_numbers[:2]})")
                    break  # Only merge ONE T&C per BL

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

    # Remove consumed packets from merged list
    merged_packets = [p for i, p in enumerate(merged_packets) if i not in _consumed]
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
                    'inspection report': 'survey_report',
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
            # More packets than total pages — likely multiple copies of same report
            # Group by proximity: packets within a small window are one copy
            # Use total_pages * 2 as the max gap (e.g., 3-page report → max 6 page gap)
            # Group by matching Page X numbers — each complete set (1,2,3) is one copy.
            # Build groups greedily: for each "Page 1", find the nearest "Page 2", "Page 3" etc.
            _sorted = sorted(pkt_entries, key=lambda x: x[2])  # sort by first page number
            _used = set()
            _groups = []
            for _entry in _sorted:
                if id(_entry) in _used:
                    continue
                if _entry[1] != 1:
                    continue  # Start groups from "Page 1"
                _group = [_entry]
                _used.add(id(_entry))
                _last_pg = _entry[2]
                # Find Page 2, Page 3, etc. — nearest unused match
                for _need_x in range(2, _total_y + 1):
                    _best = None
                    _best_dist = 999
                    for _cand in _sorted:
                        if id(_cand) in _used:
                            continue
                        if _cand[1] == _need_x:
                            _dist = abs(_cand[2] - _last_pg)
                            if _dist < _best_dist:
                                _best = _cand
                                _best_dist = _dist
                    if _best and _best_dist <= _total_y * 3:
                        _group.append(_best)
                        _used.add(id(_best))
                        _last_pg = _best[2]
                if len(_group) > 1:
                    _groups.append(_group)
            # Also group orphan pages (Page 2 or 3 without a Page 1 nearby)
            _remaining = [e for e in _sorted if id(e) not in _used]
            if _remaining:
                _groups.append(_remaining)

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

    for pkt in packets:
        # Concatenate text from all pages in packet
        all_text = []
        for pg_num in pkt.page_numbers:
            pg_data = page_text_map.get(pg_num, {})
            all_text.append(pg_data.get('cleaned_text', pg_data.get('raw_text', '')))
        pkt.doc_hint = pkt.doc_hint or pkt.document_type

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
