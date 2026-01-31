# """
# Universal SWIFT MT Message Extraction System - FINAL PRODUCTION VERSION
# ✅ Extracts COMPLETE amendment text (fixes "CLAUSE NO." truncation bug)
# ✅ Handles ALL text formats: "47B: Label\n text" and ":47B:text" and "F47B:text"
# ✅ Properly extracts all numbered points (1), (2), (3)... separately
# ✅ 100% BACKWARD COMPATIBLE with lc_api.py
# """

# import re
# import json
# from typing import Dict, List, Optional, Tuple, Any
# from dataclasses import dataclass, asdict, field
# from datetime import datetime
# from pathlib import Path
# from services.ai_auditor import OfflineLCAuditor 
# from services.ai_postProcessor import AIOCRPostProcessor 


# # ============================================================================
# # SWIFT MT MESSAGE TYPES
# # ============================================================================

# SWIFT_MESSAGE_TYPES = {
#     '700': {'name': 'Issue of a Documentary Credit', 'category': 'Documentary Credits'},
#     '701': {'name': 'Issue of a Documentary Credit', 'category': 'Documentary Credits'},
#     '705': {'name': 'Pre-Advice of a Documentary Credit', 'category': 'Documentary Credits'},
#     '707': {'name': 'Amendment to a Documentary Credit', 'category': 'Documentary Credits'},
#     '710': {'name': 'Advice of Third Bank LC', 'category': 'Documentary Credits'},
#     '720': {'name': 'Transfer of a Documentary Credit', 'category': 'Documentary Credits'},
#     '730': {'name': 'Acknowledgement', 'category': 'Documentary Credits'},
#     '740': {'name': 'Authorization to Reimburse', 'category': 'Documentary Credits'},
#     '747': {'name': 'Amendment to Authorization to Reimburse', 'category': 'Documentary Credits'},
#     '750': {'name': 'Advice of Discrepancy', 'category': 'Documentary Credits'},
#     '760': {'name': 'Issue of a Guarantee', 'category': 'Guarantees'},
#     '767': {'name': 'Amendment to a Guarantee', 'category': 'Guarantees'},
#     '780': {'name': 'Claim under a Guarantee', 'category': 'Guarantees'},
#     '790': {'name': 'Advice of Charges/Interest', 'category': 'Documentary Credits'},
# }


# COMPREHENSIVE_FIELD_MAPPINGS = {
#     ':20:': 'Transaction Reference Number',
#     ':21:': 'Related Reference',
#     ':23:': 'Issuing Bank\'s Reference',
#     ':26E:': 'Number of Amendment',
#     ':27:': 'Sequence of Total',
#     ':30:': 'Date of Amendment',
#     ':31C:': 'Date of Issue',
#     ':31D:': 'Date and Place of Expiry',
#     ':32A:': 'Value Date/Currency/Amount',
#     ':32B:': 'Currency/Amount',
#     ':39A:': 'Percentage Credit Amount Tolerance',
#     ':40A:': 'Form of Documentary Credit',
#     ':40E:': 'Applicable Rules',
#     ':41A:': 'Available With...By...',
#     ':42P:': 'Deferred Payment Details',
#     ':43P:': 'Partial Shipments',
#     ':43T:': 'Transhipment',
#     ':44C:': 'Latest Date of Shipment',
#     ':44E:': 'Port of Loading',
#     ':44F:': 'Port of Discharge',
#     ':45A:': 'Description of Goods',
#     ':45B:': 'Description of Goods (Amendment)',
#     ':46A:': 'Documents Required',
#     ':46B:': 'Documents Required (Amendment)',
#     ':47A:': 'Additional Conditions',
#     ':47B:': 'Additional Conditions (Amendment)',
#     ':48:': 'Period for Presentation',
#     ':49:': 'Confirmation Instructions',
#     ':50:': 'Applicant',
#     ':51D:': 'Applicant Bank',
#     ':52A:': 'Issuing Bank',
#     ':52D:': 'Issuing Bank',
#     ':53A:': 'Reimbursing Bank',
#     ':57A:': 'Advise Through Bank',
#     ':59:': 'Beneficiary',
#     ':71D:': 'Details of Charges',
#     ':72Z:': 'Sender to Receiver Information',
#     ':78:': 'Instructions to Paying/Accepting/Negotiating Bank',
#     ':22A:': 'Purpose of Message',
# }


# SUPPORTING_DOCUMENT_TYPES = {

#     # =========================
#     # FINANCIAL / COMMERCIAL
#     # =========================
#     "COMMERCIAL_INVOICE": [
#         "COMMERCIAL INVOICE",
#         "FINAL INVOICE",
#         "INVOICE NO",
#         "TAX INVOICE",
#         "PROFORMA INVOICE"
#     ],

#     "PROFORMA_INVOICE": [
#         "PROFORMA INVOICE"
#     ],

#     "CREDIT_NOTE": [
#         "CREDIT NOTE"
#     ],

#     "DEBIT_NOTE": [
#         "DEBIT NOTE"
#     ],

#     "STATEMENT_OF_ACCOUNT": [
#         "STATEMENT OF ACCOUNT"
#     ],

#     # =========================
#     # TRANSPORT DOCUMENTS
#     # =========================
#     "BILL_OF_LADING": [
#         "BILL OF LADING",
#         "B/L",
#         "OCEAN BILL OF LADING",
#         "SHIPPED ON BOARD BILL OF LADING",
#         "CLEAN ON BOARD"
#     ],

#     "SEA_WAYBILL": [
#         "SEA WAYBILL",
#         "SEAWAY BILL"
#     ],

#     "AIR_WAYBILL": [
#         "AIR WAYBILL",
#         "AWB",
#         "MASTER AIR WAYBILL",
#         "HOUSE AIR WAYBILL"
#     ],

#     "ROAD_TRANSPORT_DOCUMENT": [
#         "CMR",
#         "ROAD CONSIGNMENT NOTE"
#     ],

#     "RAIL_TRANSPORT_DOCUMENT": [
#         "RAILWAY RECEIPT",
#         "RAIL CONSIGNMENT NOTE"
#     ],

#     "MULTIMODAL_TRANSPORT_DOCUMENT": [
#         "MULTIMODAL TRANSPORT DOCUMENT",
#         "COMBINED TRANSPORT DOCUMENT"
#     ],

#     "CHARTER_PARTY_BILL": [
#         "CHARTER PARTY BILL OF LADING"
#     ],

#     # =========================
#     # PACKING / QUANTITY
#     # =========================
#     "PACKING_LIST": [
#         "PACKING LIST",
#         "PACKING DETAILS"
#     ],

#     "WEIGHT_CERTIFICATE": [
#         "WEIGHT CERTIFICATE",
#         "CERTIFICATE OF WEIGHT"
#     ],

#     "MEASUREMENT_CERTIFICATE": [
#         "MEASUREMENT CERTIFICATE",
#         "ULLAGE REPORT",
#         "DRAFT SURVEY"
#     ],

#     "QUANTITY_CERTIFICATE": [
#         "QUANTITY CERTIFICATE"
#     ],

#     # =========================
#     # INSURANCE
#     # =========================
#     "INSURANCE_POLICY": [
#         "INSURANCE POLICY"
#     ],

#     "INSURANCE_CERTIFICATE": [
#         "INSURANCE CERTIFICATE"
#     ],

#     "COVER_NOTE": [
#         "COVER NOTE"
#     ],

#     # =========================
#     # ORIGIN / TRADE COMPLIANCE
#     # =========================
#     "CERTIFICATE_OF_ORIGIN": [
#         "CERTIFICATE OF ORIGIN",
#         "COO",
#         "FORM A",
#         "FORM E"
#     ],

#     "EUR1_CERTIFICATE": [
#         "EUR.1",
#         "EUR1 CERTIFICATE"
#     ],

#     "ATR_CERTIFICATE": [
#         "ATR CERTIFICATE"
#     ],

#     "GSP_CERTIFICATE": [
#         "GSP CERTIFICATE"
#     ],

#     # =========================
#     # INSPECTION / QUALITY
#     # =========================
#     "INSPECTION_CERTIFICATE": [
#         "INSPECTION CERTIFICATE",
#         "CERTIFICATE OF INSPECTION"
#     ],

#     "QUALITY_CERTIFICATE": [
#         "CERTIFICATE OF QUALITY",
#         "CERTIFICATE OF ANALYSIS",
#         "ANALYSIS CERTIFICATE"
#     ],

#     "SURVEYOR_REPORT": [
#         "SURVEYOR REPORT",
#         "SURVEY REPORT"
#     ],

#     "PRE_SHIPMENT_INSPECTION": [
#         "PRE-SHIPMENT INSPECTION",
#         "PSI CERTIFICATE"
#     ],

#     # =========================
#     # AGRICULTURE / FOOD / HEALTH
#     # =========================
#     "PHYTOSANITARY_CERTIFICATE": [
#         "PHYTOSANITARY CERTIFICATE"
#     ],

#     "VETERINARY_CERTIFICATE": [
#         "VETERINARY CERTIFICATE"
#     ],

#     "HEALTH_CERTIFICATE": [
#         "HEALTH CERTIFICATE"
#     ],

#     "FUMIGATION_CERTIFICATE": [
#         "FUMIGATION CERTIFICATE"
#     ],

#     # =========================
#     # BENEFICIARY / APPLICANT DECLARATIONS
#     # =========================
#     "BENEFICIARY_CERTIFICATE": [
#         "BENEFICIARY CERTIFICATE"
#     ],

#     "APPLICANT_CERTIFICATE": [
#         "APPLICANT CERTIFICATE"
#     ],

#     "CERTIFICATE_OF_COMPLIANCE": [
#         "CERTIFICATE OF COMPLIANCE"
#     ],

#     "NON_MANIPULATION_CERTIFICATE": [
#         "NON MANIPULATION CERTIFICATE"
#     ],

#     "NON_BLACKLIST_CERTIFICATE": [
#         "NON BLACKLIST CERTIFICATE"
#     ],

#     # =========================
#     # CUSTOMS / REGULATORY
#     # =========================
#     "CUSTOMS_DECLARATION": [
#         "CUSTOMS DECLARATION",
#         "EXPORT DECLARATION",
#         "IMPORT DECLARATION"
#     ],

#     "EXPORT_LICENSE": [
#         "EXPORT LICENSE"
#     ],

#     "IMPORT_LICENSE": [
#         "IMPORT LICENSE"
#     ],

#     # =========================
#     # PAYMENT / BANKING
#     # =========================
#     "BENEFICIARY_STATEMENT": [
#         "BENEFICIARY STATEMENT"
#     ],

#     "DRAFT": [
#         "DRAFT",
#         "BILL OF EXCHANGE"
#     ],

#     "REMITTANCE_ADVICE": [
#         "REMITTANCE ADVICE"
#     ],

#     # =========================
#     # MISC / FALLBACK
#     # =========================
#     "LETTER_OF_INDEMNITY": [
#         "LETTER OF INDEMNITY",
#         "LOI"
#     ],

#     "DELIVERY_ORDER": [
#         "DELIVERY ORDER"
#     ],

#     "WAREHOUSE_RECEIPT": [
#         "WAREHOUSE RECEIPT"
#     ],

#     "UNKNOWN_SUPPORTING": []
# }


# # ============================================================================
# # DATA STRUCTURES
# # ============================================================================

# @dataclass
# class LCField:
#     """Represents a single field"""
#     field_code: str
#     field_name: str
#     value: str
#     raw_text: str


# @dataclass
# class LCDocument:
#     """Represents a complete LC/MT document"""
#     document_type: str
#     lc_number: str
#     message_type: str
#     issue_date: Optional[str] = None
#     amendment_number: Optional[str] = None
#     amendment_date: Optional[str] = None
#     sender: Optional[str] = None
#     receiver: Optional[str] = None
#     fields: Dict[str, LCField] = None
#     additional_conditions: List[Dict] = None
#     documents_required: List[Dict] = None
#     raw_text: str = ""
    
#     def __post_init__(self):
#         if self.fields is None:
#             self.fields = {}
#         if self.additional_conditions is None:
#             self.additional_conditions = []
#         if self.documents_required is None:
#             self.documents_required = []


# # ============================================================================
# # UNIVERSAL MT EXTRACTOR - FINAL VERSION
# # ============================================================================


# def normalize_lc_number(value: str) -> str:
#     if not value:
#         return ""
#     return re.sub(r'[^A-Z0-9]', '', value.upper())
# class LCExtractor:
#     """Bulletproof Universal SWIFT MT Extractor with COMPLETE text extraction"""
    
#     FIELD_MAPPINGS = COMPREHENSIVE_FIELD_MAPPINGS
    
#     def __init__(self):
#         self.current_doc = None
#         self.message_types = SWIFT_MESSAGE_TYPES
#         self.ocr_cleaner = AIOCRPostProcessor()

#     def _sanitize_text(self, text: str) -> str:
#         """Removes non-mappable characters that break the console."""
#         if not text:
#             return ""
#         # Only keep printable characters, tabs, and newlines
#         return "".join(c for c in text if c.isprintable() or c in "\n\r\t")
        


#     def extract_from_text(self, text: str) -> Any:
#         """Extract data or categorize supporting documents for later validation."""

#         print("\n[LCExtractor] Starting extraction process... before Prostprocessing\n",text)
#         # text = self.ocr_cleaner.clean_text(text)
#         print("\n[LCExtractor] Starting extraction process... after Prostprocessing\n",text)

#         # 1. Clean the incoming OCR text immediately (Fixes your charmap error)
#         text = self._sanitize_text(text)
        
#         # 2. Detect if it's a SWIFT message
#         mt_type = self._detect_message_type(text)
        
#         # 3. CLASSIFICATION LOGIC
#         if mt_type:
#             # It's a SWIFT message (LC or Amendment)
#             is_amendment = self._is_amendment(text, mt_type)
#             if is_amendment:
#                 return self._extract_amendment(text, mt_type)
#             else:
#                 return self._extract_lc(text, mt_type)
        
#         # 4. SUPPORTING DOCUMENT LOGIC (The "Skip but Keep" part)
#         # If no MT type is found, it's likely an Invoice, BL, or Certificate
#         return self._categorize_supporting_doc(text)

#     def _categorize_supporting_doc(self, text: str, file_name: str = "Unknown File") -> LCDocument:
#         """
#         Categorize non-SWIFT supporting documents (Invoice, BL, Certificates, etc.)
#         """
#         text_upper = text.upper()

#         detected_type = "UNKNOWN_SUPPORTING"
#         matched_keywords = []
#         confidence = 0

#         # 1️⃣ Document type scoring
#         for doc_type, keywords in SUPPORTING_DOCUMENT_TYPES.items():
#             hits = [kw for kw in keywords if kw in text_upper]
#             if hits and len(hits) > confidence:
#                 detected_type = doc_type
#                 matched_keywords = hits
#                 confidence = len(hits)

#         # 🔍 CONSOLE LOG: Classification Result
#         print(f"\n[CLASSIFICATION] File: {file_name}")
#         print(f"               Detected Category: {detected_type}")
#         print(f"               Confidence Score:  {confidence} keywords matched")

#         # 2️⃣ Create a unified LCDocument (NO LC NUMBER YET)
#         doc = LCDocument(
#             document_type=detected_type,
#             lc_number="PENDING",              # ⬅️ Assigned later
#             message_type="NON_SWIFT",
#             raw_text=text,
#             fields={},                        # Supporting docs don't have SWIFT fields
#             additional_conditions=[],
#             documents_required=[]
#         )

#         # 3️⃣ Attach classification metadata
#         doc.is_supporting = True
#         doc.file_name = file_name             # Store filename for reference
#         doc.classification = {
#             "confidence": confidence,
#             "matched_keywords": matched_keywords
#         }
#         doc.status = "stored_for_validation"

#         return doc

#     def _find_lc_reference_in_supporting_doc(self, text: str) -> str:
#         """Finds LC Number in non-SWIFT documents like Invoices."""
#         patterns = [
#             r'L/?C\s*(?:NO\.?|NUMBER)?\s*:?\s*([A-Z0-9/]{5,})',
#             r'DOCUMENTARY\s*CREDIT\s*(?:NO\.?|NUMBER)?\s*:?\s*([A-Z0-9/]{5,})',
#             r'CREDIT\s*NUMBER\s*:?\s*([A-Z0-9/]{5,})'
#         ]
#         for p in patterns:
#             match = re.search(p, text, re.IGNORECASE)
#             if match:
#                 # Use your existing normalize function to clean it
#                 return normalize_lc_number(match.group(1))
#         return "UNKNOWN"
    



#     def _strip_ocr_labels(self, text: str) -> str:
#         # List of common labels found in OCR
#         labels = [
#             r'Documentary Credit Number', r"Sender's Reference", 
#             r'Additional Conditions', r'Documents Required',
#             r'Description of Goods', r'Date of Issue'
#         ]
#         clean_text = text
#         for label in labels:
#             # Case-insensitive removal of the label if it's at the start
#             clean_text = re.sub(rf'^\s*{label}\s*', '', clean_text, flags=re.IGNORECASE)
#         return clean_text.strip()
        
#     def _detect_message_type(self, text: str) -> Optional[str]:
#         # Use a pattern that doesn't rely on word boundaries or spaces
#         patterns = [
#             # This matches "Message type", "Messagetype", "Message-type", etc.
#             r'Message\s*t?y?p?e?\s*:?\s*(\d{3})', 
#             r'MT\s*(\d{3})',
#             r'fin\.?\s*(\d{3})'
#         ]
        
#         # Pre-clean the text JUST for detection to handle the "squash"
#         search_text = text.replace(" ", "").replace("\n", "")

#         # print(search_text[:500],'THe search text her')  # Debug: print the cleaned text snippet
#         for pattern in patterns:
#             # Search in the squashed text for "Messagetype:707"
#             match = re.search(r'Messagetype:?(\d{3})', search_text, re.IGNORECASE)
#             # print(match,'The match here')  # Debug: print the match object
#             if match:
#                 return match.group(1)
#             # Fallback to standard search in original text
#             match = re.search(pattern, text, re.IGNORECASE)
#             if match:
#                 return match.group(1)
                
#         return None


#     def _is_amendment(self, text: str, mt_type: Optional[str]) -> bool:
#         # 1️⃣ Check Message Type (detected by the squashed-friendly regex above)
#         if mt_type in {'707', '747', '767'}:
#             return True

#         # 2️⃣ Mandatory amendment fields (Strong signals)
#         # We look for the tag (26E) or labels without forcing spaces
#         mandatory_indicators = [
#             r'26E',                       
#             r'Number\s*of\s*Amendment',   
#             r'Date\s*of\s*Amendment',     
#         ]

#         for pattern in mandatory_indicators:
#             if re.search(pattern, text, re.IGNORECASE):
#                 return True

#         # 3️⃣ Amendment-only operations and field tags (4.2 and 4.3 specifically)
#         amendment_signals = [
#             r'45B', r'46B', r'47B',       # Amendment-only tags [cite: 1, 4]
#             r'/ADD/', r'/DELETE/', r'/REPALL/' # Operation codes [cite: 1, 4]
#         ]

#         for pattern in amendment_signals:
#             if re.search(pattern, text, re.IGNORECASE):
#                 return True

#         return False

#     def _extract_lc(self, text: str, mt_type: Optional[str] = None) -> LCDocument:
#         """Extract LC from ANY text format"""
#         doc = LCDocument(
#             document_type="LC",
#             lc_number="",
#             message_type=f"MT{mt_type}" if mt_type else "MT700",
#             raw_text=text
#         )
        
#         # Extract LC number - THREE FORMAT PATTERNS
#         lc_patterns = [
#             # Pattern 1: "20: Documentary Credit Number\n  ILC..."
#             r'20:\s*Documentary Credit Number\s+([A-Z0-9]+)',
#             # Pattern 2: ":20:ILC..."
#             r':20:\s*([A-Z0-9]+)',
#             # Pattern 3: "Documentary Credit Number\n  ILC..."
#             r'Documentary Credit Number\s+([A-Z0-9]+)',
#         ]
#         for pattern in lc_patterns:
#             match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
#             if match:
#                 doc.lc_number = normalize_lc_number(match.group(1))
#                 # doc.lc_number = match.group(1).strip()
#                 break
        
#         # Extract issue date - THREE FORMAT PATTERNS
#         date_patterns = [
#             # Pattern 1: "31C: Date of Issue\n  230509"
#             r'31C:\s*Date of Issue\s+(\d{6})',
#             # Pattern 2: ":31C:230509"
#             r':31C:\s*(\d{6})',
#         ]
#         for pattern in date_patterns:
#             match = re.search(pattern, text, re.IGNORECASE)
#             if match:
#                 doc.issue_date = match.group(1).strip()
#                 break
        
#         # Extract sender
#         sender_patterns = [
#             r'To Institution:\s*([A-Z0-9]+)',
#             r'52A:\s*Issuing Bank\s+([A-Z0-9]+)',
#         ]
#         for pattern in sender_patterns:
#             match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
#             if match:
#                 doc.sender = match.group(1).strip()
#                 break
        
#         # Extract receiver
#         receiver_patterns = [
#             r'To Institution:\s*([A-Z0-9]+)',
#         ]
#         for pattern in receiver_patterns:
#             match = re.search(pattern, text, re.IGNORECASE)
#             if match:
#                 doc.receiver = match.group(1).strip()
#                 break
        
#         # Extract all fields
#         doc.fields = self._extract_all_fields(text)
        
#         # Extract additional conditions (47A)
#         doc.additional_conditions = self._extract_numbered_points_robust(text, ['47A:', ':47A:'])
        
#         # Extract documents required (46A)
#         doc.documents_required = self._extract_numbered_points_robust(text, ['46A:', ':46A:'])
        
#         return doc
    
#     def _extract_amendment(self, text: str, mt_type: Optional[str] = None) -> LCDocument:
#         """Extract amendment with 'Squash-Proof' regex for messy OCR"""
#         doc = LCDocument(
#             document_type="AMENDMENT",
#             lc_number="",
#             message_type=f"MT{mt_type}" if mt_type else "MT707",
#             raw_text=text
#         )
        
#         # 1. LC Number (Tag 20)
#         # Flexible: matches ":20:", "20:", "Sender'sReference", or "20:Sender'sReference"
#         lc_patterns = [
#             r'20:?(?:\s*Sender\'s\s*Reference\s*)?([A-Z0-9]{10,})', # Capture long alphanumeric strings
#             r'Sender\'s\s*Reference\s*([A-Z0-9]{10,})'
#         ]
#         for pattern in lc_patterns:
#             match = re.search(pattern, text, re.IGNORECASE)
#             if match:
#                 doc.lc_number = normalize_lc_number(match.group(1))
#                 break
                
#         # 2. Amendment Number (Tag 26E)
#         # OCR Example: "26E:NumberofAmendment02"
#         amend_patterns = [
#             r'26E:?(?:\s*Number\s*of\s*Amendment\s*)?(\d+)',
#             r'Number\s*of\s*Amendment\s*(\d+)'
#         ]
#         for pattern in amend_patterns:
#             match = re.search(pattern, text, re.IGNORECASE)
#             if match:
#                 doc.amendment_number = match.group(1).strip()
#                 break

#         # 3. Amendment Date (Tag 30)
#         # OCR Example: "30:DateofAmendment230525"
#         amend_date_patterns = [
#             r'30:?(?:\s*Date\s*of\s*Amendment\s*)?(\d{6})',
#             r'Date\s*of\s*Amendment\s*(\d{6})'
#         ]
#         for pattern in amend_date_patterns:
#             match = re.search(pattern, text, re.IGNORECASE)
#             if match:
#                 doc.amendment_date = match.group(1).strip()
#                 break

#         # 4. Date of Issue (Tag 31C)
#         issue_patterns = [
#             r'31C:?(?:\s*Date\s*of\s*Issue\s*)?(\d{6})'
#         ]
#         for pattern in issue_patterns:
#             match = re.search(pattern, text, re.IGNORECASE)
#             if match:
#                 doc.issue_date = match.group(1).strip()
#                 break

#         # 5. Sender (Tag 52A)
#         # OCR Example: "52A:IssuingBankHABBPKKA786"
#         sender_patterns = [
#             r'52A:?(?:\s*Issuing\s*Bank\s*)?([A-Z0-9]{8,11})'
#         ]
#         for pattern in sender_patterns:
#             match = re.search(pattern, text, re.IGNORECASE)
#             if match:
#                 doc.sender = match.group(1).strip()
#                 break

#         # 6. Receiver (To Institution)
#         # OCR Example: "ToInstitution:QNBAQAQAXXX"
#         receiver_patterns = [
#             r'To\s*Institution\s*:?\s*([A-Z0-9]{8,11})'
#         ]
#         for pattern in receiver_patterns:
#             match = re.search(pattern, text, re.IGNORECASE)
#             if match:
#                 doc.receiver = match.group(1).strip()
#                 break

#         # 7. Field-by-Field and Clause Extraction
#         doc.fields = self._extract_all_fields(text)
#         doc.additional_conditions = self._extract_amendment_changes_complete(text, ['47B'])
#         doc.documents_required = self._extract_amendment_changes_complete(text, ['46B'])
        
#         desc_changes = self._extract_amendment_changes_complete(text, ['45B'])
#         if desc_changes:
#             for change in desc_changes:
#                 change['field_code'] = '45B'
#             doc.additional_conditions.extend(desc_changes)
        
#         return doc


#     def _extract_all_fields(self, text: str) -> Dict[str, LCField]:
#         fields = {}
        
#         # IMPROVED REGEX: 
#         # 1. Matches tags at start of line or after a newline
#         # 2. Matches tags even if preceded by a letter (e.g., F20: or Reference20:)
#         # 3. Requires a colon after the 2-3 digit code
#         # flexible_pattern = r'(?:^|\n|[a-zA-Z])\s*:?(\d{2,3}[A-Z]?):'
#         flexible_pattern = r'(?:^|\n|(?<=[a-zA-Z]))\s*:?(\d{2,3}[A-Z]?):'
#         matches = list(re.finditer(flexible_pattern, text))
        
#         for i, match in enumerate(matches):
#             field_num = match.group(1)
#             start_pos = match.end()
#             # The field ends where the next tag begins, or at the end of the document
#             end_pos = matches[i+1].start() if i < len(matches) - 1 else len(text)
            
#             raw_content = text[start_pos:end_pos].strip()
            
#             # --- NOISY OCR CLEANING LOGIC ---
#             # In your HBL/UBL docs, labels like "Documentary Credit Number" 
#             # often appear right after the tag before the actual value.
#             content_lines = raw_content.split('\n')
            
#             if content_lines:
#                 first_line = content_lines[0].strip()
#                 # List of labels to ignore if they appear at the very start of a field
#                 noise_labels = [
#                     "Documentary Credit Number", "Sender's Reference", "Receiver's Reference",
#                     "Issuing Bank's Reference", "Number of Amendment", "Date of Amendment",
#                     "Date of Issue", "Date and Place of Expiry", "Applicable Rules",
#                     "Form of Documentary Credit", "Additional Conditions", "Documents Required",
#                     "Description of Goods", "Narrative", "Lines 2-100", "Lines2to100"
#                 ]
                
#                 # Check if the first line is just one of these labels
#                 is_label = any(label.lower() in first_line.lower() for label in noise_labels)
                
#                 # If the first line is a known label and contains NO digits, 
#                 # or if it's explicitly a "Narrative" marker, skip it.
#                 if is_label and (not any(char.isdigit() for char in first_line) or "Narrative" in first_line):
#                     field_value = "\n".join(content_lines[1:]).strip()
#                 else:
#                     field_value = raw_content
#             else:
#                 field_value = raw_content

#             field_code = f':{field_num}:'
            
#             # Store the field
#             fields[field_code] = LCField(
#                 field_code=field_code,
#                 field_name=self.FIELD_MAPPINGS.get(field_code, f"Field {field_num}"),
#                 value=self._clean_field_value(field_value),
#                 raw_text=raw_content
#             )
            
#         return fields

#     def _extract_numbered_points_robust(self, text: str, field_codes: List[str]) -> List[Dict]:
#         """
#         PROPERLY extract ALL numbered points as INDIVIDUAL items.
#         Handles (1), 1., and OCR-squashed numbers like '1.TEXT'
#         """
#         points = []
        
#         # We use the already-extracted fields to ensure consistency
#         if not hasattr(self, 'current_doc_fields') or not self.current_doc_fields:
#             self.current_doc_fields = self._extract_all_fields(text)
            
#         for code in field_codes:
#             # Normalize the field code for lookup (e.g., '47A' -> ':47A:')
#             lookup = code if code.startswith(':') else f':{code.rstrip(":")}:'
            
#             if lookup not in self.current_doc_fields:
#                 continue
                
#             content = self.current_doc_fields[lookup].raw_text
            
#             # --- ROBUST POINT SPLITTING ---
#             # This regex matches:
#             # 1. (1) or (12)
#             # 2. 1. or 12.
#             # 3. Numbers at the start of a new line followed by a space
#             split_pattern = r'(?:\n|^)\s*(?:\(?(\d+)\)?[\.\s]|(\d+)\.)'
            
#             # Find all numbering positions
#             matches = list(re.finditer(split_pattern, content))
            
#             if not matches:
#                 # If no numbered points found, treat the whole block as Point 1
#                 clean_text = self._clean_field_value(content)
#                 if clean_text:
#                     points.append({
#                         'point_number': 1,
#                         'text': clean_text,
#                         'field_code': lookup.strip(':')
#                     })
#                 continue

#             for i, match in enumerate(matches):
#                 point_num = match.group(1) or match.group(2)
#                 start_pos = match.end()
#                 # Point ends where the next number starts
#                 end_pos = matches[i+1].start() if i < len(matches) - 1 else len(content)
                
#                 point_text = content[start_pos:end_pos]
                
#                 # Clean OCR noise (Narrative, Lines 2-100, etc.) from the point text
#                 clean_point_text = self._clean_field_value(point_text)
                
#                 if clean_point_text:
#                     points.append({
#                         'point_number': int(point_num),
#                         'text': clean_point_text,
#                         'field_code': lookup.strip(':')
#                     })
        
#         # Remove duplicates and sort by point number
#         seen_points = set()
#         unique_points = []
#         for p in sorted(points, key=lambda x: x['point_number']):
#             if p['point_number'] not in seen_points:
#                 unique_points.append(p)
#                 seen_points.add(p['point_number'])
                
#         return unique_points
  

#     def _extract_amendment_changes_complete(self, text: str, field_codes: List[str]) -> List[Dict]:
#         """
#         Extracts ALL 5 changes from the provided PDF by iterating through every 
#         /ADD/, /DELETE/, or /REPALL/ tag found in each field.
#         """
#         changes = []
#         fields = self._extract_all_fields(text) 

#         for code in field_codes:
#             lookup = f':{code.strip(":")}:'
#             if lookup not in fields:
#                 continue
                
#             content = fields[lookup].raw_text
            
#             # 1. Find ALL SWIFT operation tags in the field
#             op_pattern = r'/(ADD|DELETE|REPALL)/'
#             op_matches = list(re.finditer(op_pattern, content, re.IGNORECASE))

#             # 2. Extract and clean the segment for EACH operation found
#             for i, match in enumerate(op_matches):
#                 op_type = match.group(1).upper()
#                 start_pos = match.end()
#                 # Segment ends at the next operation tag or the end of the field
#                 end_pos = op_matches[i+1].start() if i < len(op_matches) - 1 else len(content)
                
#                 segment_text = content[start_pos:end_pos]
                
#                 # --- AGGRESSIVE CLEANING ---
#                 # Remove repeating OCR noise and squashed "Lines2t" fragments
#                 segment_text = re.sub(r'Lines\s?\d?\s?to\s?\d+:?', '', segment_text, flags=re.IGNORECASE)
#                 segment_text = re.sub(r'Lines\s?\d?-\d+:?', '', segment_text, flags=re.IGNORECASE)
#                 segment_text = re.sub(r'Lines\d+[a-z]*', '', segment_text, flags=re.IGNORECASE) 
#                 segment_text = re.sub(r'Narrativel?:?', '', segment_text, flags=re.IGNORECASE)
#                 segment_text = re.sub(r'Code\s?:?', '', segment_text, flags=re.IGNORECASE)
                
#                 # Standardize whitespace and remove leading/trailing noise symbols
#                 segment_text = re.sub(r'\s+', ' ', segment_text).strip()
#                 segment_text = re.sub(r'^[+)\s/:\-]+', '', segment_text) 

#                 if segment_text:
#                     change = {
#                         'operation': op_type,
#                         'field_code': code.strip(":"),
#                         'narrative': segment_text,
#                         'change_text': segment_text
#                     }

#                     # 3. Detect Point Number (e.g., 1, 11, 19, 20, 21)
#                     point_match = re.search(
#                         r'(?:CLAU[S|E]*\s+)?NO\.?\s*(\d+)|FIELD\s+\d+[A-Z]?[-]?(\d+)', 
#                         segment_text, 
#                         re.IGNORECASE
#                     )
                    
#                     if point_match:
#                         p_num = point_match.group(1) or point_match.group(2)
#                         change['point_number'] = int(p_num)
                    
#                     changes.append(change)
        
#         return changes
#     def _clean_field_value(self, value: str) -> str:
#         """Enhanced cleaning for MT700 and Supporting Docs to prevent label-bleeding."""
#         if not value:
#             return ""

#         # 1. Remove OCR Label noise globally
#         noise_patterns = [
#             r'Lines\s?\d?\s?to\s?\d+:?', 
#             r'Lines\s?\d?-\d+:?',
#             r'Lines\d+[a-z]*',   # Catches squashed 'Lines2t'
#             r'Narrativel?:?', 
#             r'Code\s?:?'
#         ]
        
#         for pattern in noise_patterns:
#             value = re.sub(pattern, '', value, flags=re.IGNORECASE)

#         # 2. Normalize whitespace
#         value = re.sub(r'\s+', ' ', value)

#         # 3. Strip common labels that appear at the start of values
#         prefixes = [
#             'Name and Address:', 'Currency:', 'Date:', 'Place:', 
#             'Number:', 'Total:', 'Amount:', 'Days:', 'Party Identifier:',
#             'Account:', 'Settlement Amount:', 'Narrative:', 'Beneficiary:'
#         ]
        
#         for prefix in prefixes:
#             value = re.sub(f'^{re.escape(prefix)}\s*', '', value, flags=re.IGNORECASE)

#         # 4. Final Polish
#         value = value.strip()
#         # Remove any leading artifacts common in PDF-to-Text conversion
#         value = re.sub(r'^[+)\s/:\-]+', '', value) 
        
#         return value.strip()


# # ============================================================================
# # CONSOLIDATOR
# # ============================================================================

# # import re
# # from typing import Dict, Any, List
# # from dataclasses import asdict
# class LCConsolidator:
#     """
#     Consolidate LC with amendments using multi-dialect semantic patching and AI Auditing.
#     Handles squashed text, complex replacements, and multiple operational scenarios.
#     """
    
#     def __init__(self, use_ai=True):
#         self.lcs: Dict[str, Any] = {}
#         self.amendments: Dict[str, List[Any]] = {}
#         self.use_ai = use_ai
#         if self.use_ai:
#             # Import the improved auditor
#             # from improved_auditor import OfflineLCAuditor
#             self.auditor = OfflineLCAuditor()
    
#     def add_document(self, doc: Any):
#         """Categorizes documents into LCs or Amendments for consolidation."""
#         ln = doc.lc_number.strip() if hasattr(doc, 'lc_number') else str(doc)
#         if doc.document_type == "LC":
#             self.lcs[ln] = doc
#         else:
#             if ln not in self.amendments:
#                 self.amendments[ln] = []
#             self.amendments[ln].append(doc)

#     def _clean_instruction_text(self, text: str) -> str:
#         """Removes 'CLAUSE NO. X TO READ AS' prefixes to isolate the pure value."""
#         text = re.sub(r'^(?:CLAUSE|ITEM)\s*(?:NO\.)?\s*\d+\s*', '', text, flags=re.IGNORECASE)
#         text = re.sub(r'^(?:NOW\s+)?TO\s+READ\s+AS\s*', '', text, flags=re.IGNORECASE)
#         return text.strip()

#     def _extract_point_number(self, instruction: str) -> int:
#         """
#         Extracts the clause/point number from the instruction.
#         Example: "CLAUSE NO.5" -> 5, "CLAUSENO.27" -> 27
#         """
#         match = re.search(r'(?:CLAUSE|ITEM)?\s*(?:NO\.?)?\s*(\d+)', instruction, re.IGNORECASE)
#         if match:
#             return int(match.group(1))
#         return None

#     def _apply_change(self, points_list: List[Dict], change: Dict, field_code: str):
#         """
#         Intelligently applies an amendment change to the existing points list.
#         Prevents 'half-text' errors by ensuring the original 200-word clause is 
#         preserved unless a total replacement is explicitly required.
#         """
#         narrative = change.get('narrative', '')
#         # Clean up common OCR artifacts and quote doubling
#         raw_text = change.get('change_text', narrative).replace("''", "'").replace('"', "'")
        
#         # 1. IDENTIFY TARGET POINT
#         # Use the extractor's point number or parse it from the text (e.g., "CLAUSE 10")
#         target_point = change.get('point_number')
#         if not target_point:
#             target_point = self._extract_point_number(raw_text)
        
#         # 2. FIND THE EXISTING CLAUSE IN THE LC
#         existing_point = None
#         if target_point:
#             existing_point = next((p for p in points_list if p.get('point_number') == target_point), None)

#         # 3. OPERATION: DELETE
#         operation = change.get('operation', '').upper()
#         if operation == 'DELETE' and target_point:
#             # Use a list comprehension to remove the point without index errors
#             points_list[:] = [p for p in points_list if p.get('point_number') != target_point]
#             print(f"✓ Deleted point {target_point}")
#             return

#         # 4. OPERATION: MODIFY EXISTING POINT (The "Half-Text" Fix)
#         if existing_point:
#             original_text = existing_point.get('text', '')
            
#             # Scenario A: AI Semantic Merging (Best for name changes inside long clauses)
#             if self.use_ai:
#                 try:
#                     merged_text = self.auditor.generate_merged_text(
#                         original_text=original_text,
#                         instruction=raw_text
#                     )
                    
#                     if merged_text and len(merged_text) > (len(raw_text) * 0.5):
#                         existing_point.update({
#                             'text': merged_text,
#                             'modified_by_amendment': True,
#                             'ai_processed': True,
#                             'original_instruction': raw_text
#                         })
#                         print(f"✓ AI semantic merge successful for point {target_point}")
#                         return
#                 except Exception as e:
#                     print(f"⚠ AI merge failed for point {target_point}: {e}")

#             # Scenario B: Regex Fallback (Handles "X INSTEAD OF Y")
#             result = self._regex_fallback(original_text, raw_text)
#             if result:
#                 existing_point.update({
#                     'text': result,
#                     'modified_by_amendment': True,
#                     'fallback_method': 'regex'
#                 })
#                 print(f"✓ Regex patched point {target_point}")
#                 return

#             # Scenario C: Total Replacement (If it says "TO READ AS" but merging failed)
#             if "TO READ AS" in raw_text.upper() or operation == 'REPALL':
#                 clean_text = self._clean_instruction_text(raw_text)
#                 # Validation: Don't replace a long text with a tiny "instruction-like" string
#                 if len(clean_text) > 5:
#                     existing_point['text'] = clean_text
#                     existing_point['modified_by_amendment'] = True
#                     print(f"✓ Point {target_point} replaced with new text block")
#                 return

#         # 5. OPERATION: ADD NEW POINT
#         else:
#             new_text = self._clean_instruction_text(raw_text)
#             # Ensure we don't add "empty" instructions or pure "DELETE" markers
#             if new_text and "/DELETE/" not in raw_text.upper():
#                 new_point = {
#                     'point_number': target_point or (max([p.get('point_number', 0) for p in points_list], default=0) + 1),
#                     'text': new_text,
#                     'field_code': field_code,
#                     'added_by_amendment': True,
#                     'original_instruction': raw_text
#                 }
#                 points_list.append(new_point)
#                 print(f"✓ Added new point {new_point['point_number']}")
   
   
#     def _regex_fallback(self, original_text: str, instruction: str) -> str:
#         """
#         Attempts regex-based replacement when AI fails.
#         """
#         # Pattern 1: "X INSTEAD OF Y"
#         match = re.search(r"['\"](.+?)['\"]\s+INSTEAD\s+OF\s+['\"](.+?)['\"]", instruction, re.I | re.S)
#         if match:
#             new_val, old_val = match.group(1).strip(), match.group(2).strip()
#             if old_val.upper() in original_text.upper():
#                 return re.sub(re.escape(old_val), new_val, original_text, flags=re.IGNORECASE)
        
#         # Pattern 2: "DELETE X REPLACE BY Y"
#         match = re.search(r"DELETE\s+['\"](.+?)['\"]\s+REPLACE\s+(?:BY|WITH)\s+['\"](.+?)['\"]", instruction, re.I | re.S)
#         if match:
#             old_val, new_val = match.group(1).strip(), match.group(2).strip()
#             if old_val.upper() in original_text.upper():
#                 return re.sub(re.escape(old_val), new_val, original_text, flags=re.IGNORECASE)
        
#         # Pattern 3: "TO READ AS X" (complete replacement)
#         match = re.search(r"TO\s+READ\s+AS\s+['\"](.+?)['\"]", instruction, re.I | re.S)
#         if match:
#             return match.group(1).strip()
        
#         return None

#     def consolidate(self, lc_number: str) -> Dict:
#         """Merges all amendments into the base LC and returns a consolidated document."""
#         if lc_number not in self.lcs:
#             return None
            
#         print(f"\n{'='*80}")
#         print(f"CONSOLIDATING LC: {lc_number}")
#         print(f"{'='*80}\n")
        
#         original_lc = self.lcs[lc_number]
#         amendments = self.amendments.get(lc_number, [])
        
#         # Sort amendments by number to ensure chronological patching
#         amendments.sort(key=lambda x: int(x.amendment_number) if x.amendment_number else 0)
        
#         consolidated = {
#             'lc_number': lc_number,
#             'original_issue_date': original_lc.issue_date,
#             'sender': original_lc.sender,
#             'receiver': original_lc.receiver,
#             'message_type': 'MT700_CONSOLIDATED',
#             'amendments_applied': len(amendments),
#             'last_amendment_date': amendments[-1].amendment_date if amendments else None,
#             'fields': {k: asdict(v) if hasattr(v, '__dataclass_fields__') else v 
#                       for k, v in original_lc.fields.items()},
#             'additional_conditions': [dict(p) for p in original_lc.additional_conditions],
#             'documents_required': [dict(p) for p in original_lc.documents_required],
#             'amendment_history': []
#         }
        
#         for amendment in amendments:
#             print(f"\n--- Processing Amendment {amendment.amendment_number} ---")
#             am_rec = {'amendment_number': amendment.amendment_number, 'changes': []}
            
#             # Update Field 47A (Additional Conditions) using Amendment Field 47B
#             for change in amendment.additional_conditions:
#                 print(f"  Applying change to 47A: {change.get('narrative', '')[:80]}...")
#                 self._apply_change(consolidated['additional_conditions'], change, '47A')
#                 am_rec['changes'].append(change)
            
#             # Update Field 46A (Documents Required) using Amendment Field 46B
#             for change in amendment.documents_required:
#                 print(f"  Applying change to 46A: {change.get('narrative', '')[:80]}...")
#                 self._apply_change(consolidated['documents_required'], change, '46A')
#                 am_rec['changes'].append(change)
            
#             consolidated['amendment_history'].append(am_rec)
        
#         # Final cleanup: Ensure points are in numerical order
#         consolidated['additional_conditions'].sort(key=lambda x: x.get('point_number', 999))
#         consolidated['documents_required'].sort(key=lambda x: x.get('point_number', 999))
        
#         print(f"\n{'='*80}")
#         print(f"CONSOLIDATION COMPLETE")
#         print(f"{'='*80}\n")
        
#         return consolidated

#     def get_all_consolidated(self) -> List[Dict]:
#         """Iterates through all loaded LCs and returns their consolidated state."""
#         return [self.consolidate(ln) for ln in self.lcs.keys() if self.consolidate(ln)]

#     def print_summary(self, consolidated: Dict):
#         """Prints a human-readable summary of the consolidated LC."""
#         print(f"\n{'='*80}")
#         print(f"CONSOLIDATED LC SUMMARY: {consolidated['lc_number']}")
#         print(f"{'='*80}")
#         print(f"Amendments Applied: {consolidated['amendments_applied']}")
#         print(f"Last Amendment: {consolidated['last_amendment_date']}")
        
#         print(f"\n--- Additional Conditions (Field 47A) ---")
#         for condition in consolidated['additional_conditions']:
#             status = ""
#             if condition.get('modified_by_amendment'):
#                 status = "[MODIFIED]"
#             elif condition.get('added_by_amendment'):
#                 status = "[NEW]"
            
#             print(f"{condition['point_number']:3d}. {status:12s} {condition['text'][:100]}...")
        
#         print(f"\n--- Documents Required (Field 46A) ---")
#         for doc in consolidated['documents_required']:
#             status = ""
#             if doc.get('modified_by_amendment'):
#                 status = "[MODIFIED]"
#             elif doc.get('added_by_amendment'):
#                 status = "[NEW]"
            
#             print(f"{doc['point_number']:3d}. {status:12s} {doc['text'][:100]}...")
        
#         print(f"\n{'='*80}\n")


# # ============================================================================
# # UTILITY FUNCTIONS
# # ============================================================================

# import json
# from datetime import datetime
# from typing import List, Dict
# from dataclasses import asdict

# def process_lc_documents(file_paths: List[str], output_path: str = None) -> Dict:
#     """
#     Process multiple LC documents with:
#     1. SWIFT Consolidation (MT700 + MT707)
#     2. Supporting Document Categorization (Invoice/BL)
#     3. AI Audit of consolidated clauses against supporting docs
#     """
#     # Import here to avoid circular dependencies if any
#     from services.lc_extractor import LCExtractor, LCConsolidator
    
#     extractor = LCExtractor()
#     consolidator = LCConsolidator()
    
#     # Text accumulator to give the AI context from Invoices/BLs
#     all_supporting_text = ""
    
#     results = {
#         'processing_date': datetime.now().isoformat(),
#         'total_documents_processed': 0,
#         'lcs_found': 0,
#         'amendments_found': 0,
#         'supporting_docs_found': 0,
#         'documents': [],
#         'consolidated_lcs': []
#     }

#     # --- PHASE 1: EXTRACTION & CATEGORIZATION ---
#     for file_path in file_paths:
#         try:
#             print(f"\n{'='*40}")
#             print(f"ANALYZING: {file_path}")
#             print(f"{'='*40}") # Fixed syntax error here
            
#             with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
#                 text = f.read()
            
#             # Extract doc (Returns LCDocument for both SWIFT and Supporting)
#             doc = extractor.extract_from_text(text)
            
#             # Logic Branching based on your new _categorize_supporting_doc method
#             is_supporting = getattr(doc, 'is_supporting', False) or doc.message_type == "NON_SWIFT"

#             if is_supporting:
#                 print(f"[TYPE] Supporting Document Detected: {doc.document_type}")
#                 results['supporting_docs_found'] += 1
#                 # Accumulate text for AI context
#                 all_supporting_text += f"\n--- DATA FROM {doc.document_type} ({file_path}) ---\n"
#                 all_supporting_text += doc.raw_text + "\n"
#             else:
#                 print(f"[TYPE] SWIFT Message Detected: {doc.document_type}")
#                 consolidator.add_document(doc)
#                 if doc.document_type == 'LC':
#                     results['lcs_found'] += 1
#                 else:
#                     results['amendments_found'] += 1

#             results['total_documents_processed'] += 1
#             results['documents'].append(asdict(doc))

#         except Exception as e:
#             print(f"ERROR processing {file_path}: {str(e)}")
#             continue
    
#     # --- PHASE 2: CONSOLIDATION ---
#     print(f"\n{'-'*40}")
#     print("CONSOLIDATING AMENDMENTS...")
#     consolidated_data = consolidator.get_all_consolidated()
    
#     # --- PHASE 3: AI AUDIT ---
#     # Only run if we have a Master LC AND text from shipping docs (Invoice/BL)
#     if consolidated_data and all_supporting_text.strip():
#         print("RUNNING AI AUDIT AGAINST SUPPORTING DOCUMENTS...")
#         try:
#             # Local import of Auditor
#             from services.ai_auditor import OfflineLCAuditor
#             auditor = OfflineLCAuditor()
            
#             for lc in consolidated_data:
#                 # Audit Additional Conditions (47A) and Documents Required (46A)
#                 # These keys match your LCConsolidator.consolidate output
#                 for field_key in ['additional_conditions', 'documents_required']:
#                     if field_key in lc:
#                         for point in lc[field_key]:
#                             clause = point.get('text', '')
#                             if len(clause) > 15:
#                                 # AI Verification
#                                 verdict = auditor.verify_clause(clause, all_supporting_text[:4000])
                                
#                                 point['ai_audit'] = {
#                                     "status": "COMPLIANT" if "yes" in verdict.lower()[:10] else "DISCREPANCY",
#                                     "explanation": verdict.strip()
#                                 }
#         except Exception as ai_err:
#             print(f"AI Audit encountered an issue: {str(ai_err)}")

#     results['consolidated_lcs'] = consolidated_data

#     # --- PHASE 4: EXPORT ---
#     if output_path:
#         with open(output_path, 'w', encoding='utf-8') as f:
#             json.dump(results, f, indent=2, ensure_ascii=False)
#         print(f"\nSUCCESS: Results saved to {output_path}")

#     return results
    
# if __name__ == "__main__":
#     import sys
    
#     if len(sys.argv) > 1:
#         files = sys.argv[1:]
#         output = "lc_consolidated_output.json"
#         results = process_lc_documents(files, output)
#         print(f"\n{'='*70}")
#         print("Processing Complete")
#         print(f"{'='*70}")
#         print(f"Documents processed: {results['total_documents_processed']}")
#         print(f"LCs found: {results['lcs_found']}")
#         print(f"Amendments found: {results['amendments_found']}")
#         print(f"Consolidated: {len(results['consolidated_lcs'])} LCs")
#     else:
#         print("Usage: python lc_extractor.py <file1> <file2> ...")