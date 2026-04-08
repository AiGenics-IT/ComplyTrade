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

MODEL:  Qwen VLM @ http://10.20.10.2:8085 (classifies every page)
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
  LC, Amendment, Bill of Lading, Commercial Invoice, Draft Bill of Exchange,
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
- SWIFT F-tags (F20:, F31C:, F42A:, F46A:, F47A:, :20:, :31C:, :46A:) or "Message type: 700/707" -> "LC" or "Amendment"
- "26E: Number of Amendment" or "Date of Amendment" -> "Amendment"
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

BANK HEADER / COVERING PAGES:
- A page showing only a bank's letterhead, logo, address, and SWIFT codes (like OCBC Bank, HSBC, Citibank) WITHOUT any SWIFT F-tag fields (F20:, F31C:, F46A:, :20:, :31C:) is a "Covering Letter" or "Header Page" — NOT an LC.
- An LC page MUST contain SWIFT field tags like F20/F31C/F46A/F47A (Fusion) or :20:/:31C:/:46A: (Alliance) or bare tags like "20: Documentary Credit Number". Just having a bank name and SWIFT code on a page does NOT make it an LC.
- If a page has a bank logo at the top and a table/form below with transaction details but NO SWIFT field tags, it is likely a "Covering Letter", "Export DC Document Presentation Schedule", or "Document Remittance" — NOT an LC.
"""


def _classify_page_vlm(page_num: int, image_path: str, glm_text: str, _max_retries: int = 3) -> dict:
    """Send one page to Qwen for classification with retry logic."""
    if not os.path.exists(image_path):
        return {'page_number': page_num, 'document_type': 'unknown', 'confidence': 0.0,
                'error': 'Image not found'}

    img_b64 = base64.b64encode(open(image_path, 'rb').read()).decode()
    prompt = CLASSIFY_PROMPT.format(glm_text=glm_text)
    _current_img_b64 = img_b64  # May be replaced with resized version on retry
    payload = {
        "model": QWEN_VLM_MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_current_img_b64}"}},
            {"type": "text", "text": prompt}
        ]}],
        "max_tokens": 1000, "temperature": 0.1
    }

    last_err = None
    for attempt in range(_max_retries):
        try:
            resp = requests.post(QWEN_VLM_URL, json=payload, timeout=VLM_TIMEOUT)

            if resp.status_code != 200:
                last_err = f'VLM HTTP {resp.status_code}: {resp.text[:200]}'
                print(f"[Step 3] Page {page_num} attempt {attempt+1}: HTTP {resp.status_code}: {resp.text[:200]}")

                # If max_tokens error (image too large), resize and retry
                if 'max_tokens' in resp.text and 'got -' in resp.text:
                    try:
                        from PIL import Image
                        import io
                        img = Image.open(image_path)
                        # Reduce to 50% each retry
                        scale = 0.5 if attempt == 0 else 0.3
                        new_size = (int(img.width * scale), int(img.height * scale))
                        img = img.resize(new_size, Image.LANCZOS)
                        buf = io.BytesIO()
                        img.save(buf, format='PNG')
                        _current_img_b64 = base64.b64encode(buf.getvalue()).decode()
                        payload['messages'][0]['content'][0]['image_url']['url'] = f"data:image/png;base64,{_current_img_b64}"
                        print(f"[Step 3] Page {page_num}: Resized image to {new_size[0]}x{new_size[1]} for retry")
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
                parsed = json.loads(content[json_start:json_end])
                parsed['page_number'] = page_num
                return parsed
            else:
                return {'page_number': page_num, 'document_type': 'unknown', 'confidence': 0.3,
                        'doc_hint': content[:500]}

        except json.JSONDecodeError:
            return {'page_number': page_num, 'document_type': 'unknown', 'confidence': 0.3,
                    'doc_hint': content[:500] if 'content' in dir() else ''}
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
        _type_matches = (current_packet and
                         doc_type.lower().strip() == current_packet.document_type.lower().strip())
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
    _SWIFT_799_PATTERNS = [
        r'Message\s+type:\s*799',
        r'SWIFT_MT799',
    ]
    _SWIFT_999_PATTERNS = [
        r'Message\s+type:\s*999',
        r'SWIFT_MT999',
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

    # ── Step 0b: First pass — detect SWIFT message starts and Fusion headers ──
    _page_swift_type = {}  # page_number -> 'LC'|'Amendment'|'MT799'|'MT999'|'fusion_header'

    for pg_num, _, text in all_page_data:
        if not text:
            continue

        is_amendment = any(re.search(p, text, re.IGNORECASE) for p in _SWIFT_AMEND_PATTERNS)
        is_lc = any(re.search(p, text, re.IGNORECASE) for p in _SWIFT_LC_PATTERNS)
        is_lc_cont = any(re.search(p, text, re.IGNORECASE) for p in _SWIFT_LC_CONT_PATTERNS)
        is_799 = any(re.search(p, text, re.IGNORECASE) for p in _SWIFT_799_PATTERNS)
        is_999 = any(re.search(p, text, re.IGNORECASE) for p in _SWIFT_999_PATTERNS)
        is_swift_cont = any(re.search(p, text, re.IGNORECASE) for p in _SWIFT_CONTINUATION_PATTERNS)
        is_fusion_header = any(re.search(p, text, re.IGNORECASE) for p in _FUSION_HEADER_PATTERNS)

        # MT701 is LC continuation (not a new LC)
        if is_lc_cont:
            _page_swift_type[pg_num] = 'LC'  # Treat as LC, will be marked continuation later
        elif is_amendment:
            _page_swift_type[pg_num] = 'Amendment'
        elif is_lc:
            _page_swift_type[pg_num] = 'LC'
        elif is_799:
            _page_swift_type[pg_num] = 'MT799'
        elif is_999:
            _page_swift_type[pg_num] = 'MT999'
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

    # Phase 1b: Sandwich heuristic REMOVED — VLM is the sole classifier.
    # The heuristic was merging correctly classified pages (e.g., Agents Certificate
    # between BL pages) into the wrong document type.

    # ── Phase 2: Group pages into document packets ──
    _progress("Grouping pages into document packets...")
    packets = _group_into_packets(classifications)

    # Add text data to packets
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
