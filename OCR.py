#!/usr/bin/env python3
"""
Trade Finance AI Parser v8.0 — PRODUCTION
==========================================
Handles TWO major SWIFT formats:
  1. HBL/Fusion format: "Message type: 707", fields "27: Sequence of Total"
  2. Alliance/UBL format: "fin.700", fields "F27: Sequence of Total"

Features:
  - OCR via Qwen2.5-VL-7B-Instruct (for scanned docs)
  - pdfplumber text extraction (for digital/text PDFs)
  - Hybrid: tries text first, falls back to VLM OCR if text is poor
  - Multi-strategy classification (SWIFT detection + keywords + VLM fallback)
  - Stamp/seal detection per page via VLM
  - Structured field extraction for LC, Amendments, and all shipping doc types
  - Smart merging: only LC/Amendment continuation pages merge
  - Amendment splitter for merged amendments
"""

import re
import os
import gc
import json
import tempfile
import traceback
from typing import List, Dict, Tuple, Optional
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

# ================= CONDITIONAL IMPORTS =================

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False
    print("⚠️  pdfplumber not installed. Install with: pip install pdfplumber")

try:
    import pytesseract
    from pdf2image import convert_from_bytes
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False
    print("⚠️  pytesseract/pdf2image not installed. Tesseract OCR disabled.")

try:
    import torch
    from PIL import Image
    if not HAS_TESSERACT:
        from pdf2image import convert_from_bytes
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    HAS_VLM = True
except ImportError:
    HAS_VLM = False
    print("⚠️  VLM dependencies not installed. Running in text-only mode.")

# ================= MODEL SETUP =================

MODEL_PATH = "/home/aigenics/AI_MODELS/Qwen2.5-VL-7B-Instruct"
model = None
processor = None

def load_vlm():
    global model, processor
    if not HAS_VLM:
        print("❌ Cannot load VLM: dependencies missing")
        return False
    if model is not None:
        return True
    try:
        print(f"Loading Qwen2.5-VL-7B-Instruct from {MODEL_PATH}...")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        ).eval()
        processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
        print("✅ VLM loaded successfully")
        return True
    except Exception as e:
        print(f"❌ VLM load failed: {e}")
        return False


# ================= VLM INFERENCE =================

def vlm_inference(image: "Image.Image", prompt: str, max_tokens: int = 4096) -> str:
    """Run Qwen2.5-VL inference on a single image."""
    if model is None or processor is None:
        return ""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Extract images from messages
    image_inputs = [image]

    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    del inputs, generated_ids, generated_ids_trimmed
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return output_text


# ================= VLM PROMPTS =================

STAMP_DETECT_PROMPT = """Look at this document image carefully.
Answer in JSON format only (no markdown, no backticks):
{"has_stamps": true/false, "stamp_details": "description of each stamp/seal visible", "is_stamp_only": true/false}

A stamp is any rubber stamp, ink stamp, embossed seal, official seal, chop mark, or round/rectangular approval mark.
is_stamp_only = true means the page has ONLY stamps/seals with no other meaningful text content."""

VLM_CLASSIFY_PROMPT = """Classify this trade finance document image as exactly ONE of:
"LC", "AMENDMENT", "Commercial Invoice", "Bill of Lading", 
"Endorsement Page", "Certificate of Origin", "Certificate of Insurance", 
"Beneficiary Certificate", "Quantity Certificate", "Quality Certificate",
"Certificate of Receipted Quantity",
"Survey Report", "Notice of Readiness", "Time Sheet", "Draft Bill of Exchange",
"Tanker Cleanliness Certificate", "Vessel Experience Factor",
"Shore Tank Measurement", "Master's Receipt", "Letter of Authority",
"Port Clearance", "Email Correspondence", "Stamp Only Page",
"Blank Page", "Other"

Answer in JSON only (no markdown): {"document_type": "...", "confidence": 0.0}"""

VLM_OCR_PROMPT = """Extract ALL text from this document image exactly as it appears.
Preserve field numbers, labels, values, line breaks. Include every character visible."""


# ================= TRADE TAG MAP =================

TRADE_TAG_MAP = {
    "20": "DC_Number", "21": "Related_DC", "23": "Reference", "27": "Sequence",
    "26E": "Amendment_Number", "30": "Amendment_Date", "31C": "Date_of_Issue",
    "31D": "Date_Place_Expiry", "31B": "Date_of_Availability",
    "32B": "Amount", "33B": "Decrease_Amount", "34B": "Increase_Amount",
    "39A": "Tolerance",
    "40A": "Form_of_Credit", "40E": "Applicable_Rules",
    "41A": "Available_With_By", "41D": "Available_With_By_Details",
    "42C": "Drafts_at", "42A": "Drawee_Accepting_Bank", "42D": "Drawee_Contact",
    "42P": "Negotiation_Deferred_Payment",
    "43P": "Partial_Shipments", "43T": "Transshipment",
    "44A": "Place_of_Expiry", "44B": "Place_of_Receipt",
    "44C": "Latest_Shipment_Date", "44D": "Dispatch_From",
    "44E": "Port_of_Loading", "44F": "Port_of_Discharge",
    "44H": "Latest_Date_of_Shipment",
    "45A": "Description_of_Goods", "45B": "Description_of_Goods_Additional",
    "46A": "Documents_Required", "46B": "Documents_Required_Amendment",
    "46C": "Other_Documents",
    "47A": "Additional_Conditions", "47B": "Additional_Conditions_Amendment",
    "48": "Presentation_Period", "49": "Confirmation_Instructions",
    "50": "Applicant", "51D": "Applicant_Bank",
    "52A": "Issuing_Bank_Code", "52D": "Issuing_Bank_Details",
    "53A": "Reimbursing_Bank", "53D": "Sender_Bank_Account_Details",
    "57A": "Advise_Bank", "57D": "Advise_Bank_Details",
    "59": "Beneficiary",
    "71B": "Charges_Bank_to_Bank", "71D": "Charges",
    "72": "Sender_to_Receiver_Information", "72Z": "Sender_to_Receiver_Info",
    "77A": "Instructions_to_Bank_Free", "77B": "Discrepancy_Procedures",
    "77C": "Forwarding_Bank_Details",
    "78": "Instructions_to_Bank",
    "79": "Amendment_Details", "79A": "Amendment_Text",
    "22A": "Purpose_of_Message", "23L": "LC_Type_Code",
}


# ================= SHIPPING DOC KEYWORDS =================

SHIPPING_DOC_KEYWORDS = {
    "Commercial Invoice": [
        "COMMERCIAL INVOICE", "PROFORMA INVOICE", "TAX INVOICE",
        "INVOICE NO", "INVOICE NUMBER", "INVOICE DATE", "INVOICE AMOUNT",
        "SELLER'S INVOICE", "C/I NO", "INV NO"
    ],
    "Bill of Lading": [
        "BILL OF LADING", "TANKER BILL OF LADING", "OCEAN BILL OF LADING",
        "MULTIMODAL BILL", "COMBINED TRANSPORT BILL", "B/L NO", "BL NO",
        "SHIPPED ON BOARD", "CLEAN ON BOARD", "ON BOARD DATE",
        "FREIGHT PREPAID", "FREIGHT COLLECT", "LADEN ON BOARD",
        "MASTER BILL OF LADING", "HOUSE BILL OF LADING",
        "THROUGH BILL OF LADING", "NEGOTIABLE BILL"
    ],
    "Air Waybill": [
        "AIR WAYBILL", "AIRWAY BILL", "AWB NO", "HOUSE AIR WAYBILL",
        "MASTER AIR WAYBILL", "MAWB", "HAWB"
    ],
    "Packing List": [
        "PACKING LIST", "PACKING DETAILS", "PACKING SPECIFICATION",
        "PACKING SLIP", "WEIGHT LIST", "P/L NO"
    ],
    "Certificate of Origin": [
        "CERTIFICATE OF ORIGIN", "ORIGIN CERTIFICATE", "COUNTRY OF ORIGIN",
        "CHAMBER OF COMMERCE", "PREFERENTIAL ORIGIN", "GSP CERTIFICATE"
    ],
    "Certificate of Insurance": [
        "CERTIFICATE OF INSURANCE", "INSURANCE CERTIFICATE", "INSURANCE POLICY",
        "MARINE INSURANCE", "CARGO INSURANCE", "COVER NOTE", "ALL RISKS",
        "INSTITUTE CARGO CLAUSES", "INSURED AMOUNT"
    ],
    "Beneficiary Certificate": [
        "BENEFICIARY CERTIFICATE", "BENEFICIARY'S CERTIFICATE",
        "WE HEREBY CERTIFY", "BENEFICIARY'S STATEMENT"
    ],
    "Quantity Certificate": [
        "QUANTITY CERTIFICATE", "PRODUCTS QUANTITY CERTIFICATE",
        "PRODUCTS QUANTITY CERTIFIC",  # OCR truncation
        "QUANTITY SURVEYOR", "WEIGHT CERTIFICATE", "CERTIFICATE OF WEIGHT",
        "WEIGHMENT CERTIFICATE", "CERTIFICATES OF WEIGHTS"
    ],
    "Certificate of Receipted Quantity": [
        "CERTIFICATE OF RECEIPTED QUANTITY", "RECEIPTED QUANTITY",
        "RECEIPTED QUANTITIES", "SHORE RECEIPTED",
    ],
    "Quality Certificate": [
        "QUALITY CERTIFICATE", "PRODUCT QUALITY", "CERTIFICATE OF ANALYSIS",
        "TEST CERTIFICATE", "INSPECTION CERTIFICATE", "I&C CERTIFICATE",
        "LAB CERTIFICATE", "LABORATORY REPORT", "TEST REPORT",
        "ANALYSIS REPORT", "COA NO", "ASSAY REPORT",
        "CERTIFICATE OF COMPLIANCE", "CONFORMITY CERTIFICATE",
        "PRODUCTS QUALITY CERTIFICA",  # OCR truncation
    ],
    "Phytosanitary Certificate": [
        "PHYTOSANITARY CERTIFICATE", "PHYTOSANITARY", "PLANT HEALTH"
    ],
    "Health Certificate": [
        "HEALTH CERTIFICATE", "SANITARY CERTIFICATE", "VETERINARY CERTIFICATE",
        "HALAL CERTIFICATE", "FUMIGATION CERTIFICATE"
    ],
    "Survey Report": [
        "SURVEY REPORT", "SURVEYOR'S REPORT", "INDEPENDENT SURVEY",
        "PRE-SHIPMENT SURVEY", "LOADING SURVEY", "DISCHARGE SURVEY",
        "DRAFT SURVEY", "ULLAGE REPORT", "SURVEYOR CERTIFICATE"
    ],
    "Tanker Cleanliness Certificate": [
        "TANKER CLEANLINESS", "CLEANLINESS CERTIFICATE", "TANK INSPECTION",
        "TANK CLEANLINESS", "VESSEL CLEANLINESS"
    ],
    "Notice of Readiness": [
        "NOTICE OF READINESS", "N.O.R", "NOR TENDERED", "VESSEL IS READY",
        "TENDER NOTICE", "READY TO LOAD", "READY TO DISCHARGE"
    ],
    "Time Sheet": [
        "TIME SHEET", "STATEMENT OF FACTS", "SERVICE CERTIFICATE",
        "TIME LOG", "LAYTIME CALCULATION", "DEMURRAGE CALCULATION",
        "PILOT ON BOARD", "COMMENCED ULLAGING",
    ],
    "Letter of Authority": [
        "LETTER OF AUTHORITY", "AUTHORITY TO SIGN", "SIGNING AUTHORITY",
        "POWER OF ATTORNEY", "DO HEREBY AUTHORISE", "HEREBY AUTHORIZE",
    ],
    "Master's Receipt": [
        "MASTER'S RECEIPT", "RECEIPT FOR SEALED SAMPLES", "MASTER RECEIPT",
        "SEALED SAMPLES", "CAPTAIN'S RECEIPT"
    ],
    "Vessel Experience Factor": [
        "VESSEL'S EXPERIENCE FACTOR", "VESSEL EXPERIENCE FACTOR",
        "VEF CERTIFICATE", "EXPERIENCE FACTOR",
        "LAST FIVE PORTS", "PORTS OF CALL",
    ],
    "Shore Tank Measurement": [
        "SHORE TANK MEASUREMENT", "SHORE TANK VOLUME", "TANK MEASUREMENT",
        "SHORE TANK", "TANK GAUGING", "TANK DIP",
        "TANK NO", "BEFORE AND AFTER LOADING",
    ],
    "Temperature Density Certificate": [
        "TEMPERATURE CERTIFICATE", "DENSITY CERTIFICATE", "T/D CERTIFICATE"
    ],
    "Draft Bill of Exchange": [
        "BILL OF EXCHANGE", "DRAFT AT SIGHT", "USANCE DRAFT",
        "PAY TO THE ORDER OF", "EXCHANGE FOR", "AT SIGHT OF THIS"
    ],
    "Warehouse Receipt": ["WAREHOUSE RECEIPT", "W/R"],
    "Delivery Order": ["DELIVERY ORDER", "D/O NO", "RELEASE ORDER"],
    "Shipping Advice": [
        "SHIPPING ADVICE", "SHIPMENT ADVICE", "ADVICE OF SHIPMENT"
    ],
    "Covering Schedule": [
        "COVERING SCHEDULE", "SCHEDULE OF DOCUMENTS", "DOCUMENTS ENCLOSED"
    ],
    "Forwarding Letter": [
        "FORWARDING LETTER", "LETTER OF TRANSMITTAL", "WE ENCLOSE HEREWITH",
        "DOCUMENTS FORWARDED", "REMITTING BANK"
    ],
    "Email Correspondence": ["FROM:", "SENT:", "SUBJECT:"],
    "Debit Note": ["DEBIT NOTE", "DEBIT MEMO"],
    "Credit Note": ["CREDIT NOTE", "CREDIT MEMO"],
    "Port Clearance": [
        "PORT CLEARANCE", "PORT CLEARANCE CERTIFICATE", "CLEARANCE CERTIFICATE",
        "TIME OF DEPARTURE", "NEXT PORT", "MSA",
    ],
    "Endorsement Page": [
        "TO THE ORDER OF", "ENDORSED TO", "ENDORSEMENT",
    ],
}


# ================= SWIFT FORMAT DETECTION =================

def detect_swift_format(text: str) -> Optional[str]:
    """
    Detect which SWIFT format this page uses.
    Returns: 'fusion', 'alliance', or None
    """
    if not text:
        return None

    # Format 1: HBL/Fusion — "Message type: 700" or "Messagetype:707"
    if re.search(r'Message\s*type\s*:\s*\d{3}', text, re.IGNORECASE):
        return 'fusion'

    # Format 2: Alliance/UBL — "fin.700" or "fin.707" or "F27:" "F20:"
    if re.search(r'(?:Identifier|fin)\s*[:.]\s*(?:fin\.)?\d{3}', text, re.IGNORECASE):
        return 'alliance'
    if re.search(r'\nF\d{2}[A-Z]*:', text):
        return 'alliance'

    # Fusion format with no-space text: "Messagetype:707"
    if re.search(r'Messagetype\s*:\s*\d{3}', text, re.IGNORECASE):
        return 'fusion'

    return None


def detect_mt_number(text: str) -> Optional[str]:
    """Extract the SWIFT MT number (700, 707, etc.)"""
    if not text:
        return None

    # Fusion: "Message type: 707"
    m = re.search(r'Message\s*type\s*:\s*(\d{3})', text, re.IGNORECASE)
    if m:
        return m.group(1)

    # Fusion no-space: "Messagetype:707"
    m = re.search(r'Messagetype\s*:\s*(\d{3})', text, re.IGNORECASE)
    if m:
        return m.group(1)

    # Alliance: "fin.700" or "Identifier: fin.707"
    m = re.search(r'(?:Identifier|fin)\s*[:.]\s*(?:fin\.)?(\d{3})', text, re.IGNORECASE)
    if m:
        return m.group(1)

    # SWIFT_MT marker: "SWIFT_MT700" or "SWIFT_MT701"
    m = re.search(r'SWIFT_MT(\d{3})', text)
    if m:
        return m.group(1)

    return None


# ================= FIELD EXTRACTION =================

def extract_fusion_fields(text: str) -> Dict[str, str]:
    """
    Extract fields from HBL/Fusion format.
    Format: "TAG: Label\\nValue" or "TAG: Label\\n Value"
    Also handles no-space variants: "27:SequenceofTotal\\n1/1"
    """
    fields = {}

    # Normalize crushed text (add space after tag colon if missing)
    # e.g. "27:SequenceofTotal" → "27: SequenceofTotal"
    normalized = re.sub(r'(\d{2}[A-Z]*):(\S)', r'\1: \2', text)

    # Pattern: TAG: Label text followed by value lines until next TAG or end
    # We match "27:" or "52A:" etc followed by content until the next tag
    pattern = r'(?:^|\n)\s*(\d{2}[A-Z]*):\s*(.*?)(?=\n\s*\d{2}[A-Z]*:\s|\Z)'
    matches = re.findall(pattern, normalized, re.DOTALL)

    for tag, raw_value in matches:
        label = TRADE_TAG_MAP.get(tag, f"Field_{tag}")

        # Clean: remove the "label" line if it matches known SWIFT labels
        lines = raw_value.strip().split('\n')
        value_lines = []

        # Known label patterns to skip
        label_patterns = [
            r"^Sequence\s*of\s*Total$", r"^Sender'?s?\s*Reference$",
            r"^Receiver'?s?\s*Reference$", r"^Issuing\s*Bank'?s?\s*Reference$",
            r"^Issuing\s*Bank$", r"^Date\s*of\s*Issue$",
            r"^Number\s*of\s*Amendment$", r"^Date\s*of\s*Amendment$",
            r"^Purpose\s*of\s*Message$", r"^Form\s*of\s*Documentary\s*Credit$",
            r"^Applicable\s*Rules$", r"^Date\s*and\s*Place\s*of\s*Expiry$",
            r"^Currency\s*Code,?\s*Amount$", r"^Percentage\s*Credit.*Tolerance$",
            r"^Available\s*With.*By.*$", r"^Negotiation.*Payment\s*Details$",
            r"^Partial\s*Shipments$", r"^Transhipment$",
            r"^Port\s*of\s*Loading.*$", r"^Port\s*of\s*Discharge.*$",
            r"^Latest\s*Date\s*of\s*Shipment$", r"^Description\s*of\s*Goods.*$",
            r"^Documents\s*Required$", r"^Additional\s*Conditions$",
            r"^Period\s*for\s*Presentation.*$", r"^Conf[ri]mation\s*Instructions$",
            r"^Charges$", r"^Instructions\s*to.*Bank$",
            r"^Reimbursing\s*Bank$", r"^'?Advise\s*Through'?\s*Bank$",
            r"^Applicant$", r"^Benef[ic]+iary$", r"^Documentary\s*Credit\s*Number$",
            r"^Decrease\s*of\s*Documentary.*$",
        ]

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            # Check if this line is just a label (skip it)
            is_label = False
            for lp in label_patterns:
                if re.match(lp, stripped, re.IGNORECASE):
                    is_label = True
                    break
            # Also skip "Sequenceoftotal" (no-space variant)
            if re.match(r'^[A-Z][a-z]+(?:[A-Z][a-z]+){2,}$', stripped):
                # CamelCase run-together label — skip
                is_label = True
            if not is_label:
                value_lines.append(stripped)

        value = ' '.join(value_lines).strip()
        if value:
            if label in fields:
                fields[label] += ' ' + value
            else:
                fields[label] = value

    return fields


def extract_alliance_fields(text: str) -> Dict[str, str]:
    """
    Extract fields from Alliance/UBL format.
    Format: "FTAG: Label\\n Sub-fields with Name and Address:, Currency:, etc."
    """
    fields = {}

    # Match F-prefixed tags: F27:, F20:, F40A:, etc.
    pattern = r'(?:^|\n)\s*F(\d{2}[A-Z]*):\s*(.*?)(?=\n\s*F\d{2}[A-Z]*:|\Z)'
    matches = re.findall(pattern, text, re.DOTALL)

    for tag, raw_value in matches:
        label = TRADE_TAG_MAP.get(tag, f"Field_{tag}")

        # Parse the Alliance sub-structure to extract actual values
        lines = raw_value.strip().split('\n')
        value_parts = []

        skip_patterns = [
            r'^\s*(?:Name and Address|Party Identifier|Identifier Code)\s*:?\s*$',
            r'^\s*Lines?\d+(?:to\d+)?:\s*(?:Line\s*\d+)?\s*$',
            r'^\s*Lines2to100:\s*Lines\s*2-100(?::\s*(?:Narrative|Code))?\s*$',
            r'^\s*(?:Number|Total|Days|Date|Place|Currency|Amount|Code|Narrative\d*)\s*:\s*$',
        ]

        # Known Alliance label lines to strip completely
        alliance_labels = [
            r'^Sequence\s+of\s+Total$',
            r'^Sender.s\s+Reference$', r'^Receiver.s\s+Reference$',
            r'^Issuing\s+Bank.s\s+Reference$',
            r'^Documentary\s+Credit\s+Number$',
            r'^(?:Date\s+of\s+Issue|Date\s+and\s+Place\s+of\s+Expiry)$',
            r'^Number\s+of\s+Amendment$', r'^Date\s+of\s+Amendment$',
            r'^Purpose\s+of\s+Message$',
            r'^Form\s+of\s+Documentary\s+Credit$',
            r'^(?:Partial\s+Shipments|Transhipment)$',
            r'^(?:Port\s+of\s+Loading|Port\s+of\s+Discharge).*$',
            r'^(?:Latest\s+Date\s+of\s+Shipment)$',
            r'^Description\s+of\s+Goods.*$', r'^Documents\s+Required$',
            r'^Additional\s+Conditions$', r'^Confirmation\s+Instructions$',
            r'^Period\s+for\s+Presentation.*$',
            r'^Instructions\s+to\s+the\s+Pay.*$',
            r'^Charges$', r'^Beneficiary$', r'^Applicant$',
            r'^(?:Applicant|Issuing|Reimbursing|Advise)\s+Bank.*$',
            r'^Available\s+With\s+.*$', r'^Negotiation.*Details$',
            r'^Currency\s+Code.*Amount$', r'^Applicable\s+Rules$',
            r'^Decrease\s+of\s+Documentary.*$',
            r'^Page\s+\d+\s+of\s+\d+$',
        ]

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # Skip structural labels
            should_skip = False
            for sp in skip_patterns:
                if re.match(sp, stripped, re.IGNORECASE):
                    should_skip = True
                    break
            if should_skip:
                continue

            # Skip Alliance label lines
            for al in alliance_labels:
                if re.match(al, stripped, re.IGNORECASE):
                    should_skip = True
                    break
            if should_skip:
                continue

            # Remove "Page X of Y" from middle of text
            stripped = re.sub(r'\s*Page\s+\d+\s+of\s+\d+\s*', ' ', stripped).strip()

            # Extract value from "Key: Value" sub-fields
            sub_match = re.match(
                r'\s*(?:Number|Total|Days|Date|Place|Currency|Amount|Code|'
                r'Narrative\d*|Identifier Code|Applicable Rules|Name and Address)'
                r'\s*:\s*(.+)',
                stripped, re.IGNORECASE
            )
            if sub_match:
                val = sub_match.group(1).strip()
                # Remove Alliance display artifacts like "#52,069,920.00#"
                val = re.sub(r'#[\d,. ]+#', '', val).strip()
                # Remove expanded date "210906 2021 Sep 06" → "210906"
                val = re.sub(r'(\d{6})\s+\d{4}\s+\w{3}\s+\d{2}', r'\1', val)
                if val:
                    value_parts.append(val)
            else:
                # Check if it's a value line (not a metadata line)
                if not re.match(r'^(?:Expansion|Sender|Receiver|Message|Block|Status|Format|Application|Deletable|Priority|Monitoring|MUR|Other|Delivery|Network)\b', stripped, re.IGNORECASE):
                    cleaned = re.sub(r'(\d{6})\s+\d{4}\s+\w{3}\s+\d{2}', r'\1', stripped)
                    value_parts.append(cleaned)

        value = ' '.join(value_parts).strip()
        if value:
            if label in fields:
                fields[label] += ' ' + value
            else:
                fields[label] = value

    return fields


def extract_swift_fields(text: str) -> Dict[str, str]:
    """Auto-detect format and extract SWIFT fields."""
    fmt = detect_swift_format(text)

    if fmt == 'alliance':
        return extract_alliance_fields(text)
    elif fmt == 'fusion':
        return extract_fusion_fields(text)
    else:
        # Try both and pick whichever yields more results
        fusion_fields = extract_fusion_fields(text)
        alliance_fields = extract_alliance_fields(text)
        if len(alliance_fields) > len(fusion_fields):
            return alliance_fields
        return fusion_fields


def extract_alliance_metadata(text: str) -> Dict[str, str]:
    """Extract metadata from Alliance format header (Sender/Receiver Institution)."""
    meta = {}

    # Sender Institution
    m = re.search(r'Sender\s+Institution:\s*(\S+)\s+Expansion:\s*(.+?)(?:\n|$)', text)
    if m:
        meta['sender_swift_code'] = m.group(1).strip()
        meta['sender_institution'] = m.group(2).strip()

    # Receiver Institution
    m = re.search(r'Receiver\s+Institution:\s*(\S+)\s+Expansion:\s*(.+?)(?:\n|$)', text)
    if m:
        meta['receiver_swift_code'] = m.group(1).strip()
        meta['receiver_institution'] = m.group(2).strip()

    # Transaction Reference
    m = re.search(r'Transaction\s+Reference:\s*(\S+)', text)
    if m:
        meta['transaction_reference'] = m.group(1).strip()

    return meta


# ================= CLASSIFICATION =================

MT_TYPE_MAP = {
    "700": "lc", "701": "lc", "710": "lc", "720": "lc",
    "707": "amendment", "708": "amendment",
    "730": "acknowledgement", "740": "reimbursement",
    "799": "free_format_message",
}


def classify_page(text: str, page_num: int = 0) -> Dict:
    """
    Classify a single page. Returns:
    {
        "doc_type": str,
        "confidence": float,
        "swift_format": str or None,  # 'fusion', 'alliance', None
        "mt_number": str or None,     # '700', '707', etc.
        "is_continuation": bool,       # looks like continuation of previous SWIFT
        "is_header_only": bool,        # page with just header/footer, no content
    }
    """
    if not text or len(text.strip()) < 20:
        return {
            "doc_type": "blank_page", "confidence": 0.1,
            "swift_format": None, "mt_number": None,
            "is_continuation": False, "is_header_only": False,
        }

    text_up = text.upper()
    clean_text = re.sub(r'\s+', '', text)

    # ── Check for header-only pages (Fusion "Select Print to output...") ──
    if re.search(r'10\.200\.144\.101', text) and len(clean_text) < 250:
        # This is just a browser header page, not real content
        return {
            "doc_type": "header_page", "confidence": 0.95,
            "swift_format": "fusion", "mt_number": None,
            "is_continuation": False, "is_header_only": True,
        }

    # ── SWIFT Detection ──
    swift_fmt = detect_swift_format(text)
    mt_num = detect_mt_number(text)

    if mt_num and mt_num in MT_TYPE_MAP:
        return {
            "doc_type": MT_TYPE_MAP[mt_num],
            "confidence": 1.0,
            "swift_format": swift_fmt or "unknown",
            "mt_number": mt_num,
            "is_continuation": False,
            "is_header_only": False,
        }

    # ── Check for SWIFT continuation page (has field tags but no MT header) ──
    has_fusion_tags = bool(re.findall(r'(?:^|\n)\s*\d{2}[A-Z]*:\s', text))
    has_alliance_tags = bool(re.findall(r'(?:^|\n)\s*F\d{2}[A-Z]*:', text))
    has_mt_markers = bool(re.search(r'SWIFT_MT\d{3}|Message\s*type|fin\.\d{3}', text, re.IGNORECASE))

    # Fusion no-space tags: "27:SequenceofTotal"
    has_nospace_tags = bool(re.findall(r'(?:^|\n)\d{2}[A-Z]*:[A-Z]', text))

    if (has_fusion_tags or has_alliance_tags or has_nospace_tags) and not has_mt_markers:
        # Count how many SWIFT tags
        tag_count = len(re.findall(r'(?:^|\n)\s*(?:F)?\d{2}[A-Z]*:', text))
        if tag_count >= 2:
            # Determine if LC or amendment based on tags present
            if re.search(r'(?:26E|79|79A|22A|33B|47B|46B|45B)\s*:', text):
                return {
                    "doc_type": "amendment", "confidence": 0.9,
                    "swift_format": swift_fmt, "mt_number": None,
                    "is_continuation": True, "is_header_only": False,
                }
            return {
                "doc_type": "lc", "confidence": 0.9,
                "swift_format": swift_fmt, "mt_number": None,
                "is_continuation": True, "is_header_only": False,
            }

    # ── Endorsement page detection (back of BL) ──
    if re.search(r'TO\s+THE\s+ORDER\s+OF', text_up) and len(clean_text) < 600:
        return {
            "doc_type": "Endorsement Page", "confidence": 0.85,
            "swift_format": None, "mt_number": None,
            "is_continuation": False, "is_header_only": False,
        }

    # ── Keyword-based classification for shipping docs ──
    # Check BEFORE lc_continuation to avoid misclassifying 
    # Beneficiary Certificates, Commercial Invoices, etc.
    best_match = None
    best_score = 0
    first_300 = text_up[:300]  # Title area — keywords here get boosted

    for doc_type, keywords in SHIPPING_DOC_KEYWORDS.items():
        matches = sum(1 for kw in keywords if kw in text_up)
        title_matches = sum(1 for kw in keywords if kw in first_300)
        if matches > 0:
            score = min(0.95, 0.55 + (matches * 0.1))
            # Boost if keyword appears in title area (likely document header)
            if title_matches > 0:
                score = min(0.95, score + 0.15)
            # Extra boost: find earliest keyword position — earlier = more likely the title
            earliest_pos = len(text_up)
            for kw in keywords:
                pos = text_up.find(kw)
                if pos >= 0 and pos < earliest_pos:
                    earliest_pos = pos
            if earliest_pos < 80:  # Very first line(s)
                score = min(0.95, score + 0.10)
            if score > best_score:
                best_score = score
                best_match = doc_type

    if best_match and best_score >= 0.55:
        return {
            "doc_type": best_match, "confidence": best_score,
            "swift_format": None, "mt_number": None,
            "is_continuation": False, "is_header_only": False,
        }

    # ── Page that looks like LC/Amendment content continuation ──
    # These pages have no field tag at the START but contain LC content
    # (e.g., continuation of 45A, 46A, 47A fields spanning multiple pages)
    lc_continuation_indicators = [
        r'(?:CLAUSE\s*NO|FIELD\s+\d{2}[A-Z]|L/C\s+CLAUSE)',
        r'(?:BILL\s+OF\s+LADING|B/L\s+(?:DATE|QUANTITY|NO))',
        r'(?:BENEFICIARY|APPLICANT|NEGOTIAT|REIMBURSE)',
        r'(?:DISCHARGE\s+PORT|LOADING\s+PORT|PORT\s+KEAMARI|PORT\s+QASIM)',
        r'(?:DOCUMENTS?\s+(?:REQUIRED|PRESENT|SHOULD|MUST))',
        r'(?:CERTIFICATE\s+OF|INSURANCE\s+ADVICE)',
        r'(?:VESSEL|SHIPMENT|CARGO|CHARTER\s+PARTY)',
        r'(?:CONT\'?D\s+(?:IN|FROM)\s+FIELD)',
        r'(?:SWIFT_MT\d{3})',
        r'(?:\(\d+\)\s+[A-Z])',  # Numbered clauses like (1) PHOTOCOPIES...
    ]
    lc_indicator_count = sum(
        1 for p in lc_continuation_indicators
        if re.search(p, text_up)
    )
    if lc_indicator_count >= 3:  # Require 3+ indicators (not 2) to be more selective
        return {
            "doc_type": "lc_continuation", "confidence": 0.8,
            "swift_format": swift_fmt, "mt_number": None,
            "is_continuation": True, "is_header_only": False,
        }

    # Short page with bank header → likely SWIFT continuation
    if len(clean_text) < 400:
        has_bank_header = bool(re.search(
            r'(?:Jubilee\s*Insurance|HABB\s*PK|HABIB\s*BANK|HBL|'
            r'Alliance\s*Message|Page\s*\d+\s*of\s*\d+)',
            text, re.IGNORECASE
        ))
        if has_bank_header:
            return {
                "doc_type": "swift_continuation", "confidence": 0.7,
                "swift_format": swift_fmt, "mt_number": None,
                "is_continuation": True, "is_header_only": False,
            }

    # ── Heuristic fallbacks ──
    has_currency = bool(re.search(r'(?:USD|EUR|GBP|JPY|CNY|INR|PKR|AED|SAR)\s*[\d.,]+', text_up))
    if has_currency and re.search(r'(?:TOTAL|AMOUNT|UNIT PRICE|PRICE PER)', text_up):
        return {
            "doc_type": "Commercial Invoice", "confidence": 0.45,
            "swift_format": None, "mt_number": None,
            "is_continuation": False, "is_header_only": False,
        }

    letter_count = sum(1 for p in [
        r'(?i)dear\s+(?:sir|madam|mr|ms|sirs)',
        r'(?i)yours?\s+(?:faithfully|sincerely|truly)',
        r'(?i)regards?\s*,',
    ] if re.search(p, text))
    if letter_count >= 2:
        return {
            "doc_type": "Email Correspondence", "confidence": 0.5,
            "swift_format": None, "mt_number": None,
            "is_continuation": False, "is_header_only": False,
        }

    return {
        "doc_type": "unidentified", "confidence": 0.0,
        "swift_format": None, "mt_number": None,
        "is_continuation": False, "is_header_only": False,
    }


# ================= SEGMENTATION =================

def _page_has_doc_header(text: str) -> bool:
    """Check if a page starts with a document header/title (not a continuation)."""
    first_lines = text.strip()[:300].upper()
    # Document headers typically start with titles, company names, form headers
    header_patterns = [
        r'^(?:TANKER\s+)?BILL\s+OF\s+LADING',
        r'^COMMERCIAL\s+INVOICE', r'^CERTIFICATE\s+OF',
        r'^NOTICE\s+OF\s+READINESS', r'^TIME\s+SHEET',
        r'^(?:PRODUCT|QUALITY)\s+(?:CERTIFICATE|QUALITY)',
        r'^SHORE\s+TANK\s+MEASUREMENT', r'^TANKER\s+CLEANLINESS',
        r'^MASTER.S\s+RECEIPT', r'^LETTER\s+OF\s+AUTHORITY',
        r'^PORT\s+CLEARANCE', r'^BENEFICIARY\s+CERTIFICATE',
        r'^VESSEL.S?\s+EXPERIENCE\s+FACTOR',
        r'^(?:CHINA\s+CERTIFICATION|CCIC|HAFNIA|SAWANT)',
        r'^(?:SAHARA|PETROCHINA|PAKISTAN\s+STATE)',
        r'^(?:THE\s+PEOPLE.S\s+REPUBLIC)',
        r'^(?:REPUBLIC\s+OF\s+SINGAPORE)',
        r'^(?:FROM:|SENT:|RE:|FWD:)',
    ]
    for p in header_patterns:
        if re.search(p, first_lines):
            return True
    return False


def segment_pages(pages_text: List[str], progress_callback=None) -> List[Dict]:
    """
    Classify and segment pages into document objects.
    Merging rules:
      - LC pages (700/701) merge together
      - Amendment pages (707) with same reference merge together
      - SWIFT continuation pages merge into the preceding SWIFT object
      - Header-only pages are skipped
      - Everything else stays separate
    """
    def _seg_progress(msg):
        print(msg)
        if progress_callback:
            try: progress_callback("classification", msg)
            except: pass

    # First: classify every page
    page_classifications = []
    for i, text in enumerate(pages_text):
        cls = classify_page(text, i + 1)
        page_classifications.append(cls)
        _seg_progress(f"  Page {i+1}: {cls['doc_type']} (conf={cls['confidence']:.2f}, "
              f"fmt={cls['swift_format']}, mt={cls['mt_number']}, "
              f"cont={cls['is_continuation']}, hdr={cls['is_header_only']})")

    # Second: segment with merging
    objects = []

    for i, (text, cls) in enumerate(zip(pages_text, page_classifications)):
        page_num = i + 1

        # Skip header-only pages
        if cls['is_header_only']:
            continue

        should_merge = False

        if objects:
            prev = objects[-1]
            prev_type = prev['type']

            # Rule 1: SWIFT continuation pages merge into previous SWIFT doc
            if cls['is_continuation'] and prev_type in ('lc', 'amendment', 'lc_continuation'):
                should_merge = True

            # Rule 2: LC 2/2 (MT701) merges into LC 1/2 (MT700)
            elif cls['mt_number'] in ('701',) and prev_type == 'lc':
                should_merge = True

            # Rule 3: Same MT type with same reference → merge
            elif (cls['doc_type'] in ('lc', 'amendment')
                  and cls['doc_type'] == prev_type
                  and cls['mt_number'] == prev.get('mt_number')
                  and not cls.get('is_new_document', False)):
                should_merge = True

            # Rule 4: Page marked as swift_continuation or lc_continuation
            elif cls['doc_type'] in ('swift_continuation', 'lc_continuation') and prev_type in ('lc', 'amendment'):
                should_merge = True

            # Rule 5: Unidentified page right after SWIFT → likely continuation
            elif (cls['doc_type'] == 'unidentified'
                  and prev_type in ('lc', 'amendment')
                  and cls['confidence'] == 0.0
                  and len(text.strip()) < 500):
                should_merge = True

            # Rule 6: Endorsement/back page merges into previous Bill of Lading
            elif cls['doc_type'] == 'Endorsement Page' and prev_type in ('Bill of Lading',):
                should_merge = True

            # Rule 7: Same document type on consecutive pages → merge
            # ONLY for document types that are commonly multi-page
            # AND only if the page doesn't have its own document header
            elif (cls['doc_type'] == prev_type
                  and cls['doc_type'] in ('Shore Tank Measurement', 'Tanker Cleanliness Certificate',
                                          'Port Clearance', 'Master\'s Receipt')
                  and not cls['is_header_only']
                  and not _page_has_doc_header(text)):
                should_merge = True

            # Rule 8: Blank page after non-blank doc shouldn't break a sequence
            # (skip blank pages entirely)

            # Rule 9: Commercial Invoice continuation (page 31 after page 30)
            elif (cls['doc_type'] == 'lc_continuation' 
                  and prev_type in ('Commercial Invoice',)):
                should_merge = True

            # Rule 10: Pages classified as shipping docs but actually LC text continuation
            # If previous is LC and this page starts mid-sentence (no doc header)
            elif (prev_type == 'lc'
                  and cls['doc_type'] in ('Bill of Lading', 'Shore Tank Measurement', 
                                          'Quality Certificate', 'Quantity Certificate',
                                          'Certificate of Receipted Quantity',
                                          'Beneficiary Certificate')
                  and not _page_has_doc_header(text)):
                should_merge = True

            # Rule 11: Page after Commercial Invoice that continues it
            elif (prev_type == 'Commercial Invoice'
                  and cls['doc_type'] in ('Bill of Lading', 'lc_continuation')
                  and not _page_has_doc_header(text)):
                should_merge = True

        if should_merge:
            objects[-1]['pages'].append(page_num)
            objects[-1]['texts'].append(text)
            objects[-1]['classifications'].append(cls)
        else:
            # For continuation types that weren't merged, assign a proper type
            obj_type = cls['doc_type']
            if obj_type in ('swift_continuation', 'lc_continuation'):
                obj_type = 'lc'  # default to LC if standalone continuation

            objects.append({
                'type': obj_type,
                'pages': [page_num],
                'texts': [text],
                'classifications': [cls],
                'swift_format': cls['swift_format'],
                'mt_number': cls['mt_number'],
            })

    return objects


# ================= STAMP DETECTION =================

def detect_stamps_vlm(image) -> Dict:
    """Use VLM to detect stamps on a page image."""
    if model is None:
        return {"has_stamps": False, "stamp_details": "", "is_stamp_only": False}

    try:
        raw = vlm_inference(image, STAMP_DETECT_PROMPT, max_tokens=200)
        raw = raw.strip()
        raw = re.sub(r'^```(?:json)?\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
        result = json.loads(raw)

        # Normalize stamp_details to string (VLM may return list of dicts)
        details = result.get("stamp_details", "")
        if isinstance(details, list):
            parts = []
            for item in details:
                if isinstance(item, dict):
                    parts.append(item.get('description', item.get('text_content', str(item))))
                else:
                    parts.append(str(item))
            details = "; ".join(parts)
        elif isinstance(details, dict):
            details = details.get('description', details.get('text_content', str(details)))

        return {
            "has_stamps": result.get("has_stamps", False),
            "stamp_details": str(details),
            "is_stamp_only": result.get("is_stamp_only", False),
        }
    except:
        return {"has_stamps": False, "stamp_details": "", "is_stamp_only": False}


def classify_with_vlm(image) -> Dict:
    """Use VLM to classify a page when text-based classification fails."""
    if model is None:
        return {"document_type": "unidentified", "confidence": 0.3}

    try:
        raw = vlm_inference(image, VLM_CLASSIFY_PROMPT, max_tokens=200)
        raw = raw.strip()
        raw = re.sub(r'^```(?:json)?\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
        result = json.loads(raw)
        return result
    except:
        return {"document_type": "unidentified", "confidence": 0.3}


# ================= AMENDMENT SPLITTER =================

class AmendmentSplitter:
    @staticmethod
    def needs_splitting(data: dict) -> bool:
        """Check if a merged amendment object contains multiple amendments."""
        amend_nums = data.get('Amendment_Number', '')
        # Alliance format: "Number of Amendment 01 Number of Amendment 02"
        alliance_count = len(re.findall(r'(?:Number\s*of\s*Amendment|NumberofAmendment)\s*(\d+)', amend_nums))
        if alliance_count > 1:
            return True
        # Fusion format: "01 02 03 04" (space-separated)
        nums = amend_nums.strip().split()
        if len(nums) > 1 and all(n.isdigit() for n in nums):
            return True
        return False

    @staticmethod
    def split(obj: dict) -> List[dict]:
        data = obj.get('data', {})
        amend_nums = data.get('Amendment_Number', '')
        amend_dates = data.get('Amendment_Date', '')

        # Try Alliance format first
        numbers = re.findall(r'(?:Number\s*of\s*Amendment|NumberofAmendment)\s*(\d+)', amend_nums)
        dates = re.findall(r'(?:Date\s*of\s*Amendment|DateofAmendment)\s*(\d{6})', amend_dates)

        # Fall back to Fusion format (space-separated)
        if not numbers:
            nums_list = amend_nums.strip().split()
            if len(nums_list) > 1 and all(n.isdigit() for n in nums_list):
                numbers = nums_list
            else:
                return [obj]

        if not dates:
            dates_list = amend_dates.strip().split()
            if all(len(d) == 6 and d.isdigit() for d in dates_list):
                dates = dates_list

        if len(numbers) <= 1:
            return [obj]

        # For multi-value shared fields, split them too
        def _split_field(val, n):
            """Split a repeated field value into n parts."""
            if not val:
                return [''] * n
            # Try splitting by known repeated patterns
            parts = val.strip().split()
            if len(parts) == n:
                return parts
            # If field has repeated values (e.g., "ISSU ISSU ISSU ISSU")
            # Try splitting evenly
            return [val] * n  # Fallback: same for all

        results = []
        page_ref = obj.get('page_reference', '?')
        pages_str = page_ref.split('-')
        start_page = int(pages_str[0]) if pages_str[0].isdigit() else 0

        for i, num in enumerate(numbers):
            new_obj = {
                'object_type': 'amendment',
                'page_reference': str(start_page + i) if start_page > 0 else page_ref,
                'page_count': 1,
                'data': {
                    'document_category': 'amendment',
                    'document_type': 'AMENDMENT',
                    'classification_confidence': data.get('classification_confidence', 0.9),
                    'has_stamps': data.get('has_stamps', False),
                    'Amendment_Number': num,
                    'Amendment_Date': dates[i] if i < len(dates) else '',
                    'swift_format': data.get('swift_format', 'unknown'),
                }
            }

            # Copy shared single-value fields
            for f in ['DC_Number', 'Related_DC', 'Reference', 'Issuing_Bank_Code',
                      'Date_of_Issue', 'Purpose_of_Message']:
                if f in data:
                    val = data[f]
                    # Try to extract the i-th repeated value
                    parts = _split_field(val, len(numbers))
                    if len(parts) == len(numbers):
                        new_obj['data'][f] = parts[i]
                    else:
                        new_obj['data'][f] = val

            # Fix DC_Number — it's the same for all amendments
            if 'DC_Number' in data:
                # Extract unique DC numbers
                dc_parts = data['DC_Number'].strip().split()
                unique_dcs = list(dict.fromkeys(dc_parts))
                if len(unique_dcs) == 1:
                    new_obj['data']['DC_Number'] = unique_dcs[0]

            # Fix Date_of_Issue — same for all
            if 'Date_of_Issue' in data:
                doi_parts = data['Date_of_Issue'].strip().split()
                unique_dois = list(dict.fromkeys(doi_parts))
                if len(unique_dois) == 1:
                    new_obj['data']['Date_of_Issue'] = unique_dois[0]

            # Fix Purpose_of_Message
            if 'Purpose_of_Message' in data:
                pom_parts = data['Purpose_of_Message'].strip().split()
                unique_poms = list(dict.fromkeys(pom_parts))
                if len(unique_poms) == 1:
                    new_obj['data']['Purpose_of_Message'] = unique_poms[0]

            # Split amendment-specific content fields by /ADD/ markers
            for block_field in ['Additional_Conditions_Amendment',
                               'Documents_Required_Amendment',
                               'Description_of_Goods_Additional']:
                if block_field in data:
                    blocks = re.split(r'/ADD/\+\)', data[block_field])
                    blocks = [b.strip() for b in blocks if b.strip()]
                    if i < len(blocks):
                        new_obj['data'][block_field] = '/ADD/+)' + blocks[i]

            # Amount fields — only first amendment typically has these
            if 'Decrease_Amount' in data and i == 0:
                new_obj['data']['Decrease_Amount'] = data['Decrease_Amount']
            if 'Increase_Amount' in data and i == 0:
                new_obj['data']['Increase_Amount'] = data['Increase_Amount']

            # Presentation_Period
            if 'Presentation_Period' in data and i == 0:
                new_obj['data']['Presentation_Period'] = data['Presentation_Period']

            results.append(new_obj)

        return results


# ================= FINAL LC CONSOLIDATOR =================

# SWIFT field code mapping for LC fields
SWIFT_FIELD_CODES = {
    'Form_of_Credit': '40A',
    'DC_Number': '20',
    'Date_of_Issue': '31C',
    'Applicable_Rules': '40E',
    'Date_Place_Expiry': '31D',
    'Applicant': '50',
    'Beneficiary': '59',
    'Amount': '32B',
    'Tolerance': '39A',
    'Available_With_By': '41A',
    'Negotiation_Deferred_Payment': '42C',
    'Partial_Shipments': '43P',
    'Transshipment': '43T',
    'Port_of_Loading': '44E',
    'Port_of_Discharge': '44F',
    'Latest_Shipment_Date': '44C',
    'Description_of_Goods': '45A',
    'Documents_Required': '46A',
    'Additional_Conditions': '47A',
    'Charges': '71D',
    'Presentation_Period': '48',
    'Confirmation_Instructions': '49',
    'Reimbursing_Bank': '53A',
    'Instructions_to_Bank': '78',
    'Advise_Bank': '57A',
    'Sequence': '27',
    'Sender_Reference': '20',
    'Related_DC': '21',
    'Reference': '20',
    'Issuing_Bank_Code': '52A',
    'Purpose_of_Message': '22A',
    'Amendment_Number': '26E',
    'Amendment_Date': '30',
    'Decrease_Amount': '33B',
    'Increase_Amount': '34B',
}


def _fix_nospace_text(text: str) -> str:
    """
    Fix text that has no spaces between words (common in Fusion SWIFT extractions).
    E.g., 'CLAUSENO.10TOREADAS' → 'CLAUSE NO. 10 TO READ AS'
    """
    if not text:
        return text

    # Common no-space patterns in trade finance amendments
    replacements = [
        (r'CLAUSE\s*NO\.?\s*', 'CLAUSE NO. '),
        (r'CLAU(?:E)?NO\.?\s*', 'CLAUSE NO. '),
        (r'TOREADAS', 'TO READ AS '),
        (r'TOREAD\s*AS', 'TO READ AS '),
        (r'NOWTOREADAS', 'NOW TO READ AS '),
        (r'NOWTO\s*READ\s*AS', 'NOW TO READ AS '),
        (r'INSTEADOF', 'INSTEAD OF '),
        (r'INSTEAD\s*OF', 'INSTEAD OF '),
        (r'REPLACEBY', 'REPLACE BY '),
        (r'REPLACE\s*BY', 'REPLACE BY '),
        (r'DELETEBY', 'DELETE BY '),
        (r'BILLOFLADINGISACCEPTABLE', 'BILL OF LADING IS ACCEPTABLE '),
        (r'BILLOFLADING', 'BILL OF LADING '),
        (r'LETTEROFAUTHORITYFORSIGNING', 'LETTER OF AUTHORITY FOR SIGNING '),
        (r'LETTEROFAUTHORITY', 'LETTER OF AUTHORITY '),
        (r'FURTHERDOCUMENTARYEVIDENCEWILLBEREQUIREDAGAINST', 'FURTHER DOCUMENTARY EVIDENCE WILL BE REQUIRED AGAINST '),
        (r'FURTHERDOCUMENTARY', 'FURTHER DOCUMENTARY '),
        (r'EVIDENCEWILLBE', 'EVIDENCE WILL BE '),
        (r'REQUIREDAGAINST', 'REQUIRED AGAINST '),
        (r'ORIGINALDOCUMENTS', 'ORIGINAL DOCUMENTS '),
        (r'OI\.?IR\.?ICGhIuNnAdLrigDaOrCRUoMadENTS', 'ORIGINAL DOCUMENTS'),
        (r'BENEFICIARYMUSTACCOMPANY', 'BENEFICIARY MUST ACCOMPANY '),
        (r'SHIPMENTTOBEEFFECTEDONLY', 'SHIPMENT TO BE EFFECTED ONLY '),
        (r'SHIPMENTTOBE', 'SHIPMENT TO BE '),
        (r'EFFECTEDONLY', 'EFFECTED ONLY '),
        (r'ONPANAMAFLAGGEDVESSEL', 'ON PANAMA FLAGGED VESSEL '),
        (r'PANAMAFLAGGED', 'PANAMA FLAGGED '),
        (r'ANDACERTIFICATETOTHISEFFECTFROM', 'AND A CERTIFICATE TO THIS EFFECT FROM '),
        (r'CERTIFICATETOTHISEFFECT', 'CERTIFICATE TO THIS EFFECT '),
        (r'COMPLYL/CCLAUSE', 'COMPLY L/C CLAUSE '),
        (r'TOCOMPLYL', 'TO COMPLY L'),
        (r'UNDERL/CCLAUSE', 'UNDER L/C CLAUSE '),
        (r'CLAUSEFIELD', 'CLAUSE FIELD '),
        (r'M/STOBENOMINATED', 'M/S TO BE NOMINATED'),
        (r'M/S\.SAWANTANDCO\.PRIVATELTD', 'M/S. SAWANT AND CO. PRIVATE LTD'),
        (r'SAWANTANDCO', 'SAWANT AND CO'),
        (r'PRIVATELTD', 'PRIVATE LTD'),
        (r'HABIBBANKLIMITED', 'HABIB BANK LIMITED '),
        (r'HBLPLAZABRANCH', 'HBL PLAZA BRANCH '),
        (r'I\.I\.CHUNDRIGARROAD', 'I.I. CHUNDRIGAR ROAD '),
        (r'I\.I\.ChundrigarRoad', 'I.I. Chundrigar Road '),
        (r'KARACHI\.PAKISTAN', 'KARACHI. PAKISTAN'),
        (r'Messagetype:', 'Message type: '),
        (r'ToInstitution:', 'To Institution: '),
        (r'Priority:Normal', 'Priority: Normal'),
        (r'QATARNATIONALBANK', 'QATAR NATIONAL BANK '),
        (r'JubileeInsuranceHouse', 'Jubilee Insurance House '),
        (r'Karachi-Pakistan', 'Karachi - Pakistan '),
        (r'6thFloor,', '6th Floor, '),
    ]

    result = text
    for pattern, replacement in replacements:
        result = re.sub(pattern, replacement, result)

    # Clean up multiple spaces
    result = re.sub(r'\s{2,}', ' ', result).strip()

    # Remove common noise from Fusion printouts
    noise_patterns = [
        r"Select 'Print' to output\.\.\.",
        r"Page \d+ of \d+",
        r"Formatted Outward SWIFT message details",
        r"http://\d+\.\d+\.\d+\.\d+:\s*\d+/\S+",
        r"10\.200\.\d+\.\d+:\s*\d+/\S+",
        r"\d+/\d+/\d{4}$",
    ]
    for np in noise_patterns:
        result = re.sub(np, '', result, flags=re.IGNORECASE)

    return re.sub(r'\s{2,}', ' ', result).strip()

class FinalLCConsolidator:
    """
    Consolidates the original LC with all amendments to produce a 'Final LC'.
    Each field/clause is tracked — if amended, it shows which amendment changed it.
    """

    @staticmethod
    def consolidate(lc_obj: dict, amendment_objs: List[dict]) -> dict:
        """
        Build a Final LC from the original LC and ordered amendments.
        Returns a new object with type 'final_lc'.
        """
        lc_data = lc_obj.get('data', {})

        # Start with a copy of LC fields
        final_fields = {}
        field_history = {}  # track changes per field

        # Fields to process
        skip_keys = {'document_category', 'document_type', 'classification_confidence',
                     'has_stamps', 'stamp_details', 'is_stamp_only', 'swift_format',
                     'classification_method', 'text_preview'}

        for k, v in lc_data.items():
            if k in skip_keys:
                continue
            final_fields[k] = v
            field_history[k] = {
                'original_value': v,
                'current_value': v,
                'amendments': [],
            }

        # Sort amendments by number
        sorted_amendments = sorted(
            amendment_objs,
            key=lambda a: int(a.get('data', {}).get('Amendment_Number', '0') or '0')
        )

        # Apply each amendment
        for amend_obj in sorted_amendments:
            amend_data = amend_obj.get('data', {})
            amend_num = amend_data.get('Amendment_Number', '?')
            amend_date = amend_data.get('Amendment_Date', '?')
            amend_page = amend_obj.get('page_reference', '?')

            amend_info = {
                'amendment_number': amend_num,
                'amendment_date': amend_date,
                'amendment_page': amend_page,
            }

            # Process Decrease/Increase amounts
            if 'Decrease_Amount' in amend_data:
                _apply_amount_change(final_fields, field_history, 'Amount',
                                     'decrease', amend_data['Decrease_Amount'], amend_info)
            if 'Increase_Amount' in amend_data:
                _apply_amount_change(final_fields, field_history, 'Amount',
                                     'increase', amend_data['Increase_Amount'], amend_info)

            # Process Description_of_Goods_Additional (changes to field 45A)
            if 'Description_of_Goods_Additional' in amend_data:
                _apply_text_amendment(final_fields, field_history,
                                      'Description_of_Goods',
                                      amend_data['Description_of_Goods_Additional'],
                                      amend_info)

            # Process Additional_Conditions_Amendment (changes to field 47A)
            if 'Additional_Conditions_Amendment' in amend_data:
                _apply_text_amendment(final_fields, field_history,
                                      'Additional_Conditions',
                                      amend_data['Additional_Conditions_Amendment'],
                                      amend_info)

            # Process Documents_Required_Amendment (changes to field 46A/46B)
            if 'Documents_Required_Amendment' in amend_data:
                _apply_text_amendment(final_fields, field_history,
                                      'Documents_Required',
                                      amend_data['Documents_Required_Amendment'],
                                      amend_info)

            # Process Presentation_Period changes
            if 'Presentation_Period' in amend_data and amend_data['Presentation_Period']:
                pp = amend_data['Presentation_Period']
                # Clean the value — remove Fusion printout noise
                pp_clean = _fix_nospace_text(pp)
                # Extract just the meaningful part (e.g., "45/PLS REFER CLAUSE NO.5 OF FIELD 47A")
                pp_match = re.match(r'(\d+/PLS\s+REFER\s+CLAUSE\s+NO\.?\s*\d+\s+OF\s+FIELD\s+\d+[A-Z]?)', pp_clean)
                if pp_match:
                    pp_clean = pp_match.group(1).strip()
                else:
                    pp_match = re.match(r'(\d+/[A-Z\s.]+FIELD\s*\d+[A-Z]?)', pp_clean)
                    if pp_match:
                        pp_clean = pp_match.group(1).strip()
                if pp_clean and not pp_clean.startswith('10.200'):
                    old_val = final_fields.get('Presentation_Period', '')
                    old_clean = _fix_nospace_text(old_val)
                    old_match = re.match(r'(\d+/PLS\s+REFER\s+CLAUSE\s+NO\.?\s*\d+\s+OF\s+FIELD\s+\d+[A-Z]?)', old_clean)
                    if old_match:
                        old_clean = old_match.group(1).strip()
                    else:
                        old_match = re.match(r'(\d+/[A-Z\s.]+FIELD\s*\d+[A-Z]?)', old_clean)
                        if old_match:
                            old_clean = old_match.group(1).strip()
                    final_fields['Presentation_Period'] = pp_clean
                    if 'Presentation_Period' not in field_history:
                        field_history['Presentation_Period'] = {
                            'original_value': old_clean,
                            'current_value': pp_clean,
                            'amendments': [],
                        }
                    field_history['Presentation_Period']['current_value'] = pp_clean
                    field_history['Presentation_Period']['amendments'].append({
                        **amend_info,
                        'change': f'Changed from "{old_clean}" to "{pp_clean}"',
                    })

        # Now build the structured Final LC with clauses broken out
        structured_fields = {}
        amendment_tracker = []

        for field_name, value in final_fields.items():
            history = field_history.get(field_name, {})
            amendments_applied = history.get('amendments', [])

            # Get SWIFT field code
            swift_code = SWIFT_FIELD_CODES.get(field_name, '')

            # Clean change descriptions (fix no-space text, remove noise)
            for amend in amendments_applied:
                if 'change' in amend:
                    amend['change'] = _fix_nospace_text(amend['change'])
                if 'change_description' in amend:
                    amend['change_description'] = _fix_nospace_text(amend['change_description'])

            # Break multi-clause fields into separate items
            if field_name in ('Additional_Conditions', 'Documents_Required',
                              'Description_of_Goods'):
                clauses = _split_into_clauses(value)
                clause_list = []
                for clause in clauses:
                    clause_entry = {
                        'swift_field': swift_code,
                        'text': clause['text'],
                    }
                    if clause.get('clause_number'):
                        clause_entry['clause_number'] = clause['clause_number']
                        clause_entry['swift_field_ref'] = f"{swift_code}-{clause['clause_number']}"

                    # Check if this clause was amended
                    clause_amendments = _find_clause_amendments(
                        clause, amendments_applied, field_name
                    )
                    if clause_amendments:
                        clause_entry['amended'] = True
                        clause_entry['amended_by'] = clause_amendments
                    else:
                        clause_entry['amended'] = False

                    clause_list.append(clause_entry)

                structured_fields[field_name] = clause_list
            else:
                entry = {
                    'swift_field': swift_code,
                    'value': value,
                }
                if amendments_applied:
                    entry['amended'] = True
                    entry['amended_by'] = amendments_applied
                    amendment_tracker.extend(amendments_applied)
                else:
                    entry['amended'] = False
                structured_fields[field_name] = entry

        # Build summary of all amendments applied
        amendments_summary = []
        for amend_obj in sorted_amendments:
            ad = amend_obj.get('data', {})
            amendments_summary.append({
                'amendment_number': ad.get('Amendment_Number', '?'),
                'amendment_date': ad.get('Amendment_Date', '?'),
                'page_reference': amend_obj.get('page_reference', '?'),
                'fields_changed': _get_fields_changed(ad),
            })

        return {
            'object_type': 'final_lc',
            'page_reference': f"{lc_obj.get('page_reference', '?')} (consolidated)",
            'page_count': 0,
            'data': {
                'document_category': 'final_lc',
                'document_type': 'FINAL_LC',
                'DC_Number': _clean_repeated(lc_data.get('DC_Number', '')),
                'Date_of_Issue': lc_data.get('Date_of_Issue', ''),
                'total_amendments_applied': len(sorted_amendments),
                'amendments_summary': amendments_summary,
                'consolidated_fields': structured_fields,
            }
        }


def _clean_repeated(val: str) -> str:
    """Remove repeated values like 'ILC123 ILC123' → 'ILC123'."""
    parts = val.strip().split()
    unique = list(dict.fromkeys(parts))
    if len(unique) == 1:
        return unique[0]
    return val


def _apply_amount_change(fields, history, field_name, direction, amount_str, amend_info):
    """Apply a decrease/increase to the Amount field."""
    old_val = fields.get(field_name, '')
    change_desc = f"{'Decreased' if direction == 'decrease' else 'Increased'} by {amount_str}"

    # Try to calculate new amount
    try:
        # Parse current: "USD20455337,00"
        curr_match = re.search(r'([A-Z]{3})\s*([\d.,]+)', old_val)
        change_match = re.search(r'([A-Z]{3})\s*([\d.,]+)', amount_str)
        if curr_match and change_match:
            currency = curr_match.group(1)
            curr_amt = float(curr_match.group(2).replace(',', '.').replace('..', '.'))
            change_amt = float(change_match.group(2).replace(',', '.').replace('..', '.'))
            if direction == 'decrease':
                new_amt = curr_amt - change_amt
            else:
                new_amt = curr_amt + change_amt
            new_val = f"{currency}{new_amt:,.2f}".replace(',', '_').replace('.', ',').replace('_', '')
            fields[field_name] = new_val
            change_desc += f" → New amount: {new_val}"
    except:
        pass

    if field_name not in history:
        history[field_name] = {'original_value': old_val, 'current_value': old_val, 'amendments': []}
    history[field_name]['current_value'] = fields.get(field_name, old_val)
    history[field_name]['amendments'].append({**amend_info, 'change': change_desc})


def _apply_text_amendment(fields, history, target_field, amendment_text, amend_info):
    """Apply DELETE/REPLACE and /ADD/ instructions to a text field."""
    if target_field not in fields:
        return

    current_val = fields[target_field]
    changes = []

    # First, clean no-space text for better pattern matching
    cleaned_amendment = _fix_nospace_text(amendment_text)

    # Parse amendment instructions (use cleaned text for matching)
    # Pattern 1: CLAUSE NO.X: DELETE ''old'' REPLACE BY ''new''
    delete_replace = re.findall(
        r"(?:CLAUSE\s*NO\.?\s*(\d+)[^']*)?DELETE\s+''([^']+)''\s*REPLACE\s+(?:BY\s+)?''([^']+)''",
        cleaned_amendment, re.IGNORECASE
    )
    for clause_num, old_text, new_text in delete_replace:
        old_clean = old_text.strip()
        new_clean = new_text.strip()
        if old_clean in current_val:
            current_val = current_val.replace(old_clean, new_clean, 1)
            change = f"Clause {clause_num}: " if clause_num else ""
            change += f'Replaced "{old_clean}" with "{new_clean}"'
            changes.append(change)
        elif old_clean.replace(' ', '') in current_val.replace(' ', ''):
            changes.append(f"Clause {clause_num}: Replaced \"{old_clean}\" with \"{new_clean}\"")

    # Pattern 2: CLAUSE NO.X NOW TO READ AS ''new text''
    read_as = re.findall(
        r"CLAUSE\s*(?:NO\.?\s*)?(\d+)\s*(?:NOW\s+)?TO\s+READ\s+AS\s+''([^']+)''",
        cleaned_amendment, re.IGNORECASE
    )
    for clause_num, new_text in read_as:
        changes.append(f"Clause {clause_num}: Now reads as \"{new_text.strip()[:200]}\"")

    # Pattern 3: TO READ AS ''new'' INSTEAD OF ''old''
    instead_of = re.findall(
        r"TO\s+READ\s+AS\s+''([^']+)''\s*INSTEAD\s*OF\s*''([^']+)''",
        cleaned_amendment, re.IGNORECASE
    )
    for new_text, old_text in instead_of:
        old_clean = old_text.strip()
        new_clean = new_text.strip()
        if old_clean in current_val:
            current_val = current_val.replace(old_clean, new_clean, 1)
        changes.append(f'Replaced "{old_clean}" with "{new_clean}"')

    fields[target_field] = current_val

    if target_field not in history:
        history[target_field] = {'original_value': current_val, 'current_value': current_val, 'amendments': []}
    history[target_field]['current_value'] = current_val
    history[target_field]['amendments'].append({
        **amend_info,
        'change': '; '.join(changes) if changes else _fix_nospace_text(amendment_text[:300]),
    })


def _split_into_clauses(text: str) -> List[dict]:
    """Split a multi-clause field into separate numbered clauses."""
    clauses = []

    # Try splitting by numbered patterns: (1), (2), ... 
    parts = re.split(r'\((\d+)\)\s+', text)
    if len(parts) > 2:
        if parts[0].strip():
            clauses.append({'clause_number': None, 'text': parts[0].strip()})
        for i in range(1, len(parts), 2):
            num = parts[i]
            clause_text = parts[i + 1].strip() if i + 1 < len(parts) else ''
            if clause_text:
                clauses.append({'clause_number': num, 'text': clause_text})
        return clauses

    # Try splitting by "N." pattern — Documents_Required uses "1.TEXT 2.TEXT"
    # Match "1." at start, or " 1." or newline+"1." 
    parts = re.split(r'(?:^|(?<=\s))(\d{1,2})\.\s*(?=[A-Z])', text)
    if len(parts) > 2:
        if parts[0].strip():
            clauses.append({'clause_number': None, 'text': parts[0].strip()})
        for i in range(1, len(parts), 2):
            num = parts[i]
            clause_text = parts[i + 1].strip() if i + 1 < len(parts) else ''
            if clause_text:
                clauses.append({'clause_number': num, 'text': clause_text})
        return clauses

    # Try splitting by "+)" pattern used in some SWIFT messages
    parts = re.split(r'\(\+\)\s+', text)
    if len(parts) > 1:
        for i, part in enumerate(parts):
            if part.strip():
                clauses.append({'clause_number': str(i + 1) if i > 0 else None, 'text': part.strip()})
        return clauses

    # No splitting possible — return as single clause
    clauses.append({'clause_number': None, 'text': text})
    return clauses


def _find_clause_amendments(clause: dict, amendments: list, field_name: str) -> list:
    """Check if a specific clause was affected by any amendment."""
    matches = []
    clause_num = clause.get('clause_number')
    clause_text = clause.get('text', '')

    for amend in amendments:
        change_text = amend.get('change', '')

        # Check if amendment explicitly mentions this clause number
        if clause_num:
            # Look for "Clause 5:", "Clause NO.5", etc.
            clause_refs = re.findall(r'Clause\s*(?:NO\.?\s*)?(\d+)', change_text, re.IGNORECASE)
            if clause_num in clause_refs:
                matches.append({
                    'amendment_number': amend['amendment_number'],
                    'amendment_date': amend['amendment_date'],
                    'amendment_page': amend['amendment_page'],
                    'change_description': change_text,
                })
                continue

        # For preamble/non-numbered clauses, check for direct text overlap
        if not clause_num and clause_text:
            # Check if amendment changes something that appears in this clause text
            # Look for specific replaced strings
            replaced_strs = re.findall(r'Replaced\s+"([^"]+)"', change_text)
            for rs in replaced_strs:
                if rs in clause_text or rs.replace(' ', '') in clause_text.replace(' ', ''):
                    matches.append({
                        'amendment_number': amend['amendment_number'],
                        'amendment_date': amend['amendment_date'],
                        'amendment_page': amend['amendment_page'],
                        'change_description': change_text,
                    })
                    break

    # Also check raw amendment text for CLAUSE NO.X references not caught above
    # Only match if the FIRST clause reference in the text matches this clause
    if clause_num and not matches:
        for amend in amendments:
            raw_change = amend.get('change', '')
            # Find all clause references
            raw_refs = re.findall(r'CLAUSE\s*(?:NO\.?\s*)?(\d+)', raw_change, re.IGNORECASE)
            # Only match if clause_num is the FIRST (primary) reference
            if raw_refs and raw_refs[0] == clause_num:
                matches.append({
                    'amendment_number': amend['amendment_number'],
                    'amendment_date': amend['amendment_date'],
                    'amendment_page': amend['amendment_page'],
                    'change_description': raw_change,
                })

    return matches


def _get_fields_changed(amend_data: dict) -> list:
    """List which LC fields this amendment changed."""
    changed = []
    if amend_data.get('Description_of_Goods_Additional'):
        changed.append('Description_of_Goods (45A)')
    if amend_data.get('Additional_Conditions_Amendment'):
        changed.append('Additional_Conditions (47A)')
    if amend_data.get('Documents_Required_Amendment'):
        changed.append('Documents_Required (46A)')
    if amend_data.get('Decrease_Amount'):
        changed.append(f"Amount decreased by {amend_data['Decrease_Amount']}")
    if amend_data.get('Increase_Amount'):
        changed.append(f"Amount increased by {amend_data['Increase_Amount']}")
    if amend_data.get('Presentation_Period'):
        changed.append('Presentation_Period (48)')
    return changed


# ================= MAIN PIPELINE =================

def process_pdf(content: bytes, use_vlm_ocr: bool = False, detect_stamps: bool = True,
                progress_callback=None) -> Dict:
    """
    Main processing pipeline.
    1. Extract text from PDF (pdfplumber for digital, VLM OCR for scanned)
    2. Classify each page
    3. Segment into document objects
    4. Extract structured fields
    5. Optionally detect stamps via VLM
    6. Split merged amendments

    progress_callback: optional callable(stage, message) for live progress reporting
    """

    def _progress(stage: str, message: str):
        """Report progress both to console and callback."""
        print(message)
        if progress_callback:
            try:
                progress_callback(stage, message)
            except Exception:
                pass

    images = None

    # Step 1: Text extraction
    pages_text = []

    if HAS_PDFPLUMBER:
        import io as _io
        with pdfplumber.open(_io.BytesIO(content)) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                pages_text.append(t if t else "")
        _progress("extraction", f"✓ pdfplumber extracted text from {len(pages_text)} pages")

        # Check quality: if most pages are very short, we may need OCR
        avg_len = sum(len(t) for t in pages_text) / max(len(pages_text), 1)
        if avg_len < 50 and (HAS_TESSERACT or HAS_VLM):
            print(f"  ⚠️ Average text length {avg_len:.0f} chars — pages may be scanned images")
            use_vlm_ocr = True

        # Also check per-page: count how many pages have thin text
        thin_pages = sum(1 for t in pages_text if len(t.strip()) < 50)
        if thin_pages > 0 and (HAS_TESSERACT or HAS_VLM):
            _progress("extraction", f"  ℹ️ {thin_pages}/{len(pages_text)} pages have < 50 chars — will OCR those")
            use_vlm_ocr = True

    # Step 1b: OCR for pages with insufficient text
    # Priority: Tesseract (fast, ~3s/page) > VLM OCR (slow, ~30s/page)
    if use_vlm_ocr:
        if HAS_TESSERACT:
            import time as _time
            _ocr_start = _time.time()
            _progress("ocr", "🔍 Running Tesseract OCR on pages with insufficient text...")
            ocr_count = 0
            for i in range(len(pages_text)):
                if len(pages_text[i].strip()) < 50:
                    # Convert one page at a time to avoid OOM
                    page_imgs = convert_from_bytes(
                        content, dpi=200, first_page=i+1, last_page=i+1
                    )
                    ocr_text = pytesseract.image_to_string(page_imgs[0])
                    pages_text[i] = ocr_text
                    ocr_count += 1
                    del page_imgs  # free memory immediately
                    if ocr_count % 10 == 0:
                        _progress("ocr", f"   ... OCR'd {ocr_count} pages so far")
            _ocr_elapsed = _time.time() - _ocr_start
            _progress("ocr", f"   ✓ Tesseract OCR: {ocr_count} pages in {_ocr_elapsed:.1f}s "
                  f"({_ocr_elapsed/max(ocr_count,1):.1f}s/page)")

        elif HAS_VLM:
            if not load_vlm():
                if not pages_text:
                    raise RuntimeError("No text extraction method available")
            else:
                _progress("ocr", "🔍 Running VLM OCR on pages with insufficient text...")
                ocr_count = 0
                for i in range(len(pages_text)):
                    if len(pages_text[i].strip()) < 50:
                        page_imgs = convert_from_bytes(
                            content, dpi=300, first_page=i+1, last_page=i+1
                        )
                        _progress("ocr", f"   Page {i+1}/{len(pages_text)}: VLM OCR")
                        ocr_text = vlm_inference(page_imgs[0], VLM_OCR_PROMPT, max_tokens=4096)
                        pages_text[i] = ocr_text
                        ocr_count += 1
                        del page_imgs
                _progress("ocr", f"   ✓ VLM OCR completed for {ocr_count} pages")
        else:
            print("  ⚠️ No OCR engine available — scanned pages will have no text!")

    total_pages = len(pages_text)

    # Step 2 & 3: Classify and segment
    _progress("classification", f"\n📄 Classifying {total_pages} pages...")
    segments = segment_pages(pages_text, progress_callback=progress_callback)

    _progress("classification", f"📑 {len(segments)} document objects found")

    # Step 4: Detect stamps (if VLM available)
    stamp_info_per_page = {}
    images = None  # Will be populated per-page for VLM classification fallback
    if detect_stamps and HAS_VLM and load_vlm():
        _progress("stamps", "\n🔍 Detecting stamps...")
        for i in range(total_pages):
            page_imgs = convert_from_bytes(content, dpi=200, first_page=i+1, last_page=i+1)
            stamp_info_per_page[i + 1] = detect_stamps_vlm(page_imgs[0])
            si = stamp_info_per_page[i + 1]
            if si['has_stamps']:
                _progress("stamps", f"   Page {i+1}: 🔴 STAMPS — {si['stamp_details']}")
                if si['is_stamp_only']:
                    _progress("stamps", f"            ⚠️ STAMP-ONLY PAGE")
            del page_imgs

    # Step 5: Extract fields and build output
    _progress("extraction_fields", "\n📋 Extracting structured fields...")
    final_output = []

    for obj in segments:
        page_ref = (
            f"{obj['pages'][0]}-{obj['pages'][-1]}"
            if len(obj['pages']) > 1
            else str(obj['pages'][0])
        )
        full_text = "\n".join(obj['texts'])
        doc_type = obj['type']

        # Stamp aggregation
        any_stamps = any(
            stamp_info_per_page.get(p, {}).get('has_stamps', False)
            for p in obj['pages']
        )

        def _stringify_stamp_details(details):
            """Convert stamp_details to string regardless of type."""
            if isinstance(details, str):
                return details
            if isinstance(details, list):
                parts = []
                for item in details:
                    if isinstance(item, dict):
                        parts.append(item.get('description', item.get('text_content', str(item))))
                    else:
                        parts.append(str(item))
                return "; ".join(parts)
            if isinstance(details, dict):
                return details.get('description', details.get('text_content', str(details)))
            return str(details) if details else ""

        stamp_details = "; ".join(
            _stringify_stamp_details(stamp_info_per_page.get(p, {}).get('stamp_details', ''))
            for p in obj['pages']
            if stamp_info_per_page.get(p, {}).get('stamp_details')
        )
        all_stamp_only = all(
            stamp_info_per_page.get(p, {}).get('is_stamp_only', False)
            for p in obj['pages']
        ) if stamp_info_per_page else False

        # If ALL pages in this object are stamp-only, reclassify
        if all_stamp_only and stamp_info_per_page:
            doc_type = "stamp_only_page"

        avg_conf = sum(
            c.get('confidence', 0) for c in obj['classifications']
        ) / max(len(obj['classifications']), 1)

        extracted = {
            "document_category": doc_type,
            "document_type": doc_type.upper().replace(" ", "_"),
            "classification_confidence": round(avg_conf, 2),
            "has_stamps": any_stamps,
        }
        if stamp_details:
            extracted["stamp_details"] = stamp_details
        if all_stamp_only and stamp_info_per_page:
            extracted["is_stamp_only"] = True

        # Field extraction based on type
        if doc_type in ('lc', 'amendment'):
            # Detect format
            fmt = obj.get('swift_format')
            if not fmt:
                for cls in obj['classifications']:
                    if cls.get('swift_format'):
                        fmt = cls['swift_format']
                        break

            swift_fields = extract_swift_fields(full_text)
            extracted.update(swift_fields)

            # Also extract Alliance metadata if applicable
            if fmt == 'alliance':
                meta = extract_alliance_metadata(full_text)
                if meta:
                    extracted.update(meta)

            extracted['swift_format'] = fmt or 'unknown'

        elif doc_type not in ('blank_page', 'header_page', 'stamp_only_page'):
            # Text preview for non-SWIFT docs
            preview = full_text[:500].replace('\n', ' ')
            if len(full_text) > 500:
                preview += "..."
            extracted["text_preview"] = preview

        # VLM fallback classification for unidentified or blank pages
        if doc_type in ('unidentified', 'blank_page') and HAS_VLM and load_vlm():
            page_idx = obj['pages'][0]  # 1-based
            page_imgs = convert_from_bytes(content, dpi=200, first_page=page_idx, last_page=page_idx)
            vlm_cls = classify_with_vlm(page_imgs[0])
            vlm_type = vlm_cls.get('document_type', 'Other')
            vlm_conf = vlm_cls.get('confidence', 0.3)
            if vlm_type not in ('Blank Page', 'Other', 'Blank_Page'):
                extracted['document_category'] = vlm_type
                extracted['document_type'] = vlm_type.upper().replace(' ', '_')
                extracted['classification_confidence'] = vlm_conf
                extracted['classification_method'] = 'vlm_fallback'
                doc_type = vlm_type
                obj['type'] = vlm_type  # update the object type
            del page_imgs

        final_output.append({
            "object_type": doc_type,
            "page_reference": page_ref,
            "page_count": len(obj['pages']),
            "data": extracted,
        })

    # Step 6: Split merged amendments
    processed_output = []
    for obj in final_output:
        if obj['object_type'] == 'amendment' and AmendmentSplitter.needs_splitting(obj.get('data', {})):
            processed_output.extend(AmendmentSplitter.split(obj))
        else:
            processed_output.append(obj)

    # Step 7: Generate Final LC (consolidated LC + all amendments applied)
    lc_objs = [o for o in processed_output if o['object_type'] == 'lc']
    amend_objs = [o for o in processed_output if o['object_type'] == 'amendment']
    if lc_objs and amend_objs:
        _progress("extraction_fields", f"\n📋 Generating Final LC (LC + {len(amend_objs)} amendments)...")
        final_lc = FinalLCConsolidator.consolidate(lc_objs[0], amend_objs)
        processed_output.append(final_lc)
        _progress("extraction_fields", f"   ✓ Final LC consolidated")
    elif lc_objs:
        _progress("extraction_fields", f"\n📋 No amendments found — LC is already final")
        # Still add a final_lc entry that mirrors the LC
        final_lc = {
            'object_type': 'final_lc',
            'page_reference': lc_objs[0].get('page_reference', '?'),
            'page_count': 0,
            'data': {
                'document_category': 'final_lc',
                'document_type': 'FINAL_LC',
                'DC_Number': _clean_repeated(lc_objs[0].get('data', {}).get('DC_Number', '')),
                'total_amendments_applied': 0,
                'amendments_summary': [],
                'consolidated_fields': {
                    k: {'value': v, 'amended': False}
                    for k, v in lc_objs[0].get('data', {}).items()
                    if k not in ('document_category', 'document_type', 'classification_confidence',
                                 'has_stamps', 'stamp_details', 'swift_format')
                },
            }
        }
        processed_output.append(final_lc)

    # Summary
    type_counts = {}
    for obj in processed_output:
        t = obj['object_type']
        type_counts[t] = type_counts.get(t, 0) + 1

    skip_types = {'blank_page', 'header_page', 'unidentified', 'Other'}
    identified = sum(c for t, c in type_counts.items() if t not in skip_types)
    total = len(processed_output)
    rate = (identified / total * 100) if total > 0 else 0

    _progress("summary", f"\n📊 Results:")
    for t, c in sorted(type_counts.items()):
        _progress("summary", f"   {t}: {c}")
    _progress("summary", f"   Classification rate: {rate:.1f}%")

    return {
        "total_pages": total_pages,
        "total_objects": total,
        "classification_rate": f"{rate:.1f}%",
        "type_summary": type_counts,
        "identified_objects": processed_output,
    }


# ================= FASTAPI APPLICATION =================

import uuid
import shutil
import threading
from pathlib import Path
from datetime import datetime
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Any

app = FastAPI(title="Trade Finance AI Parser v8.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
UPLOAD_DIR = Path("uploads/lc_uploads")
RESULTS_DIR = Path("uploads/lc_results")
VIEW_DIR = Path("view")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# In-memory job store
processing_jobs: Dict[str, Any] = {}


def _get_job_dir(job_id: str) -> Path:
    d = UPLOAD_DIR / job_id; d.mkdir(exist_ok=True); return d

def _get_results_dir(job_id: str) -> Path:
    d = RESULTS_DIR / job_id; d.mkdir(exist_ok=True); return d


def _run_processing(job_id: str, file_paths: list):
    """Background thread: process each file through the parser pipeline."""
    try:
        processing_jobs[job_id]["status"] = "processing"
        processing_jobs[job_id]["message"] = "Processing documents..."
        processing_jobs[job_id]["progress_log"] = []

        def _on_progress(stage: str, message: str):
            """Callback from process_pdf to capture live progress."""
            entry = {"stage": stage, "message": message, "time": datetime.now().isoformat()}
            processing_jobs[job_id]["progress_log"].append(entry)
            processing_jobs[job_id]["current_stage"] = stage
            processing_jobs[job_id]["message"] = message

        results_dir = _get_results_dir(job_id)
        all_objects = []
        merged_type_summary = {}
        total_pages = 0
        errors = []

        for file_path in file_paths:
            try:
                _on_progress("upload", f"📂 Processing {file_path.name}...")

                with open(file_path, "rb") as f:
                    content = f.read()

                result = process_pdf(content, use_vlm_ocr=False, detect_stamps=True,
                                     progress_callback=_on_progress)

                objects = result.get("identified_objects", [])
                for obj in objects:
                    obj["_source_file"] = file_path.name
                all_objects.extend(objects)

                for doc_type, count in result.get("type_summary", {}).items():
                    merged_type_summary[doc_type] = merged_type_summary.get(doc_type, 0) + count

                total_pages += result.get("total_pages", 0)

                # Save per-file raw response
                raw_file = results_dir / f"{file_path.stem}_api_response.json"
                with open(raw_file, "w", encoding="utf-8") as f:
                    json.dump({
                        "filename": file_path.name,
                        "status": "success",
                        **result,
                    }, f, indent=2, ensure_ascii=False)

                processing_jobs[job_id]["files_processed"] += 1

            except Exception as e:
                errors.append(f"{file_path.name}: {str(e)}")

        # Build document manifest
        document_manifest = []
        for obj in all_objects:
            obj_type = obj.get("object_type", "unidentified")
            data = obj.get("data", {})

            if obj_type == "lc":
                category = "lc"
            elif obj_type == "amendment":
                category = "amendment"
            elif obj_type == "final_lc":
                category = "final_lc"
            elif obj_type in ("blank_page", "header_page", "unidentified"):
                category = "unclassified"
            else:
                category = "supporting"

            document_manifest.append({
                "source_file": obj.get("_source_file", ""),
                "page_reference": obj.get("page_reference", ""),
                "page_count": obj.get("page_count", 1),
                "object_type": obj_type,
                "category": category,
                "classification_confidence": data.get("classification_confidence", 0),
                "has_stamps": data.get("has_stamps", False),
                "stamp_details": data.get("stamp_details", ""),
                "data": data,
            })

        # Build consolidated_lcs for checklist compatibility
        final_lc_obj = next((o for o in all_objects if o.get("object_type") == "final_lc"), None)
        lc_objs = [o for o in all_objects if o.get("object_type") == "lc"]
        consolidated_lcs = []

        if final_lc_obj:
            flc_data = final_lc_obj.get("data", {})
            cf = flc_data.get("consolidated_fields", {})

            docs_req = cf.get("Documents_Required", [])
            add_cond = cf.get("Additional_Conditions", [])
            if isinstance(docs_req, dict): docs_req = [docs_req]
            if isinstance(add_cond, dict): add_cond = [add_cond]

            dc_num = flc_data.get("DC_Number", "")
            if not dc_num and lc_objs:
                dc_num = lc_objs[0].get("data", {}).get("DC_Number", "UNKNOWN")
            dc_parts = dc_num.strip().split()
            if len(set(dc_parts)) == 1 and dc_parts:
                dc_num = dc_parts[0]

            consolidated_lcs.append({
                "lc_number": dc_num,
                "original_issue_date": flc_data.get("Date_of_Issue", ""),
                "amendments_applied": flc_data.get("total_amendments_applied", 0),
                "amendments_summary": flc_data.get("amendments_summary", []),
                "consolidated_fields": cf,
                "documents_required": docs_req,
                "additional_conditions": add_cond,
                "download_url": f"/api/download/{job_id}/{dc_num}",
            })

            lc_file = results_dir / f"{dc_num}_consolidated.json"
            with open(lc_file, "w", encoding="utf-8") as f:
                json.dump(consolidated_lcs[0], f, indent=2, ensure_ascii=False)

        # Write master results
        results = {
            "job_id": job_id,
            "processing_date": datetime.now().isoformat(),
            "total_pages": total_pages,
            "files_processed": processing_jobs[job_id]["files_processed"],
            "lcs_found": merged_type_summary.get("lc", 0),
            "amendments_found": merged_type_summary.get("amendment", 0),
            "supporting_docs_found": sum(
                c for t, c in merged_type_summary.items()
                if t not in ("lc", "amendment", "final_lc", "blank_page", "header_page", "unidentified")
            ),
            "unclassified_docs": sum(
                c for t, c in merged_type_summary.items()
                if t in ("blank_page", "header_page", "unidentified")
            ),
            "type_summary": merged_type_summary,
            "documents": document_manifest,
            "consolidated_lcs": consolidated_lcs,
            "errors": errors,
        }

        with open(results_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        processing_jobs[job_id]["status"] = "completed"
        processing_jobs[job_id]["message"] = "Processing completed successfully"
        processing_jobs[job_id]["completed_at"] = datetime.now().isoformat()
        processing_jobs[job_id]["lcs_found"] = merged_type_summary.get("lc", 0)
        processing_jobs[job_id]["amendments_found"] = merged_type_summary.get("amendment", 0)
        processing_jobs[job_id]["errors"] = errors

    except Exception as e:
        processing_jobs[job_id]["status"] = "failed"
        processing_jobs[job_id]["message"] = f"Processing failed: {str(e)}"
        processing_jobs[job_id]["errors"].append(str(e))


# ─────────────── RAW OCR ENDPOINTS (unchanged) ───────────────

@app.post("/ocr/")
async def analyze_document(file: UploadFile = File(...)):
    try:
        content = await file.read()
        print(f"\n{'=' * 80}")
        print(f"📥 Processing: {file.filename} ({len(content)} bytes)")
        print(f"{'=' * 80}")

        result = process_pdf(content, use_vlm_ocr=False, detect_stamps=True)

        return {
            "filename": file.filename,
            "status": "success",
            "extraction_method": ("pdfplumber + Tesseract" if HAS_TESSERACT else "pdfplumber") + (" + Qwen2.5-VL-7B" if HAS_VLM else ""),
            **result,
        }

    except Exception as e:
        return JSONResponse({
            "error": str(e),
            "traceback": traceback.format_exc()
        }, status_code=500)


@app.post("/ocr/scan/")
async def analyze_scanned_document(file: UploadFile = File(...)):
    """Endpoint specifically for scanned/image PDFs — forces VLM OCR."""
    try:
        content = await file.read()
        print(f"\n{'=' * 80}")
        print(f"📥 Processing (SCAN MODE): {file.filename}")
        print(f"{'=' * 80}")

        result = process_pdf(content, use_vlm_ocr=True, detect_stamps=True)

        return {
            "filename": file.filename,
            "status": "success",
            "extraction_method": "Qwen2.5-VL-7B OCR + VLM",
            **result,
        }

    except Exception as e:
        return JSONResponse({
            "error": str(e),
            "traceback": traceback.format_exc()
        }, status_code=500)


# ─────────────── WEB UI ENDPOINTS ───────────────

@app.get("/")
async def root():
    return {
        "service": "Trade Finance AI Parser",
        "version": "8.0.0",
        "status": "online",
        "vlm_available": HAS_VLM and model is not None,
        "tesseract_available": HAS_TESSERACT,
        "pdfplumber_available": HAS_PDFPLUMBER,
        "ui": "/interface",
        "endpoints": {
            "/ocr/": "Direct OCR processing (single file, returns full JSON)",
            "/ocr/scan/": "Force VLM OCR mode",
            "/interface": "Web interface",
            "/checklist": "LC verification checklist",
            "/api/upload": "Upload files for background processing",
            "/api/status/{job_id}": "Check job status",
            "/api/result/{job_id}": "Get processing results",
        },
    }


@app.get("/interface", response_class=HTMLResponse)
async def serve_interface():
    html_path = VIEW_DIR / "web_interface.html"
    if not html_path.exists():
        return HTMLResponse("<h1>view/web_interface.html not found</h1><p>Create a 'view' folder next to the server script.</p>", status_code=404)
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/checklist", response_class=HTMLResponse)
async def serve_checklist():
    html_path = VIEW_DIR / "checklist.html"
    if not html_path.exists():
        return HTMLResponse("<h1>view/checklist.html not found</h1>", status_code=404)
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


@app.post("/api/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload one or more files for background processing."""
    if not files:
        return JSONResponse({"error": "No files provided"}, status_code=400)

    job_id = str(uuid.uuid4())
    job_dir = _get_job_dir(job_id)

    processing_jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "message": "Files uploaded, queued for processing",
        "created_at": datetime.now().isoformat(),
        "completed_at": None,
        "files_processed": 0,
        "lcs_found": 0,
        "amendments_found": 0,
        "errors": [],
    }

    file_paths = []
    for upload_file in files:
        fp = job_dir / upload_file.filename
        with open(fp, "wb") as buf:
            shutil.copyfileobj(upload_file.file, buf)
        file_paths.append(fp)

    # Run in background thread
    t = threading.Thread(target=_run_processing, args=(job_id, file_paths), daemon=True)
    t.start()

    return {
        "job_id": job_id,
        "status": "pending",
        "message": f"Processing {len(files)} file(s)",
        "files": [f.filename for f in files],
        "status_url": f"/api/status/{job_id}",
        "result_url": f"/api/result/{job_id}",
    }


@app.get("/api/status/{job_id}")
async def get_job_status(job_id: str):
    if job_id not in processing_jobs:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    job = processing_jobs[job_id]
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "message": job.get("message"),
        "current_stage": job.get("current_stage", ""),
        "created_at": job["created_at"],
        "completed_at": job.get("completed_at"),
        "files_processed": job["files_processed"],
        "lcs_found": job["lcs_found"],
        "amendments_found": job["amendments_found"],
        "errors": job["errors"],
        "progress_log": job.get("progress_log", []),
    }


@app.get("/api/result/{job_id}")
async def get_job_result(job_id: str):
    if job_id not in processing_jobs:
        return JSONResponse({"error": "Job not found"}, status_code=404)

    job = processing_jobs[job_id]

    if job["status"] in ("pending", "processing"):
        return {"job_id": job_id, "status": job["status"], "message": job.get("message", "Still processing.")}

    if job["status"] == "failed":
        return {"job_id": job_id, "status": "failed", "message": job["message"], "errors": job["errors"]}

    results_file = _get_results_dir(job_id) / "results.json"
    if not results_file.exists():
        return JSONResponse({"error": "Results file not found"}, status_code=500)

    with open(results_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    for lc_data in results.get("consolidated_lcs", []):
        lc_num = lc_data.get("lc_number", "")
        lc_data["download_url"] = f"/api/download/{job_id}/{lc_num}"

    return results


@app.get("/api/download/{job_id}/{lc_number}")
async def download_consolidated_lc(job_id: str, lc_number: str):
    if job_id not in processing_jobs:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    lc_file = _get_results_dir(job_id) / f"{lc_number}_consolidated.json"
    if not lc_file.exists():
        return JSONResponse({"error": "LC file not found"}, status_code=404)
    return FileResponse(lc_file, media_type="application/json", filename=f"{lc_number}_consolidated.json")


@app.get("/api/download-original/{job_id}/{filename}")
async def download_original_file(job_id: str, filename: str):
    if job_id not in processing_jobs:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    original = _get_job_dir(job_id) / filename
    if not original.exists():
        return JSONResponse({"error": "File not found"}, status_code=404)
    mime_map = {".pdf": "application/pdf", ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
    return FileResponse(original, media_type=mime_map.get(original.suffix.lower(), "application/octet-stream"), filename=filename)


@app.get("/api/jobs")
async def list_jobs():
    jobs_list = [
        {
            "job_id": jid,
            "status": jd["status"],
            "created_at": jd["created_at"],
            "files_processed": jd["files_processed"],
            "lcs_found": jd["lcs_found"],
            "amendments_found": jd["amendments_found"],
        }
        for jid, jd in processing_jobs.items()
    ]
    jobs_list.sort(key=lambda x: x["created_at"], reverse=True)
    return {"total_jobs": len(jobs_list), "jobs": jobs_list}


@app.delete("/api/job/{job_id}")
async def delete_job(job_id: str):
    if job_id not in processing_jobs:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    job_dir = _get_job_dir(job_id)
    if job_dir.exists(): shutil.rmtree(job_dir)
    results_dir = _get_results_dir(job_id)
    if results_dir.exists(): shutil.rmtree(results_dir)
    del processing_jobs[job_id]
    return {"message": f"Job {job_id} deleted"}


@app.get("/api/lc/{job_id}/{lc_number}")
async def get_specific_lc(job_id: str, lc_number: str):
    if job_id not in processing_jobs:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    lc_file = _get_results_dir(job_id) / f"{lc_number}_consolidated.json"
    if not lc_file.exists():
        return JSONResponse({"error": "LC not found"}, status_code=404)
    with open(lc_file, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    import uvicorn

    # Try to load VLM at startup
    if HAS_VLM:
        load_vlm()

    print("\n" + "=" * 80)
    print("Trade Finance AI Parser v8.0")
    print(f"VLM: {'Loaded' if model is not None else 'Not available'}")
    print(f"pdfplumber: {'Available' if HAS_PDFPLUMBER else 'Not available'}")
    print(f"Web UI: http://0.0.0.0:8082/interface")
    print(f"Checklist: http://0.0.0.0:8082/checklist")
    print(f"API Docs: http://0.0.0.0:8082/docs")
    print("Port: 8082")
    print("=" * 80 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8082)