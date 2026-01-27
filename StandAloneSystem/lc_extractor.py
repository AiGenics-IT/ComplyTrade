"""
Universal SWIFT MT Message Extraction System - FINAL PRODUCTION VERSION
✅ Extracts COMPLETE amendment text (fixes "CLAUSE NO." truncation bug)
✅ Handles ALL text formats: "47B: Label\n text" and ":47B:text" and "F47B:text"
✅ Properly extracts all numbered points (1), (2), (3)... separately
✅ 100% BACKWARD COMPATIBLE with lc_api.py
"""

import re
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path


# ============================================================================
# SWIFT MT MESSAGE TYPES
# ============================================================================

SWIFT_MESSAGE_TYPES = {
    '700': {'name': 'Issue of a Documentary Credit', 'category': 'Documentary Credits'},
    '701': {'name': 'Issue of a Documentary Credit', 'category': 'Documentary Credits'},
    '705': {'name': 'Pre-Advice of a Documentary Credit', 'category': 'Documentary Credits'},
    '707': {'name': 'Amendment to a Documentary Credit', 'category': 'Documentary Credits'},
    '710': {'name': 'Advice of Third Bank LC', 'category': 'Documentary Credits'},
    '720': {'name': 'Transfer of a Documentary Credit', 'category': 'Documentary Credits'},
    '730': {'name': 'Acknowledgement', 'category': 'Documentary Credits'},
    '740': {'name': 'Authorization to Reimburse', 'category': 'Documentary Credits'},
    '747': {'name': 'Amendment to Authorization to Reimburse', 'category': 'Documentary Credits'},
    '750': {'name': 'Advice of Discrepancy', 'category': 'Documentary Credits'},
    '760': {'name': 'Issue of a Guarantee', 'category': 'Guarantees'},
    '767': {'name': 'Amendment to a Guarantee', 'category': 'Guarantees'},
    '780': {'name': 'Claim under a Guarantee', 'category': 'Guarantees'},
    '790': {'name': 'Advice of Charges/Interest', 'category': 'Documentary Credits'},
}


COMPREHENSIVE_FIELD_MAPPINGS = {
    ':20:': 'Transaction Reference Number',
    ':21:': 'Related Reference',
    ':23:': 'Issuing Bank\'s Reference',
    ':26E:': 'Number of Amendment',
    ':27:': 'Sequence of Total',
    ':30:': 'Date of Amendment',
    ':31C:': 'Date of Issue',
    ':31D:': 'Date and Place of Expiry',
    ':32A:': 'Value Date/Currency/Amount',
    ':32B:': 'Currency/Amount',
    ':39A:': 'Percentage Credit Amount Tolerance',
    ':40A:': 'Form of Documentary Credit',
    ':40E:': 'Applicable Rules',
    ':41A:': 'Available With...By...',
    ':42P:': 'Deferred Payment Details',
    ':43P:': 'Partial Shipments',
    ':43T:': 'Transhipment',
    ':44C:': 'Latest Date of Shipment',
    ':44E:': 'Port of Loading',
    ':44F:': 'Port of Discharge',
    ':45A:': 'Description of Goods',
    ':45B:': 'Description of Goods (Amendment)',
    ':46A:': 'Documents Required',
    ':46B:': 'Documents Required (Amendment)',
    ':47A:': 'Additional Conditions',
    ':47B:': 'Additional Conditions (Amendment)',
    ':48:': 'Period for Presentation',
    ':49:': 'Confirmation Instructions',
    ':50:': 'Applicant',
    ':51D:': 'Applicant Bank',
    ':52A:': 'Issuing Bank',
    ':52D:': 'Issuing Bank',
    ':53A:': 'Reimbursing Bank',
    ':57A:': 'Advise Through Bank',
    ':59:': 'Beneficiary',
    ':71D:': 'Details of Charges',
    ':72Z:': 'Sender to Receiver Information',
    ':78:': 'Instructions to Paying/Accepting/Negotiating Bank',
    ':22A:': 'Purpose of Message',
}


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class LCField:
    """Represents a single field"""
    field_code: str
    field_name: str
    value: str
    raw_text: str


@dataclass
class LCDocument:
    """Represents a complete LC/MT document"""
    document_type: str
    lc_number: str
    message_type: str
    issue_date: Optional[str] = None
    amendment_number: Optional[str] = None
    amendment_date: Optional[str] = None
    sender: Optional[str] = None
    receiver: Optional[str] = None
    fields: Dict[str, LCField] = None
    additional_conditions: List[Dict] = None
    documents_required: List[Dict] = None
    raw_text: str = ""
    
    def __post_init__(self):
        if self.fields is None:
            self.fields = {}
        if self.additional_conditions is None:
            self.additional_conditions = []
        if self.documents_required is None:
            self.documents_required = []


# ============================================================================
# UNIVERSAL MT EXTRACTOR - FINAL VERSION
# ============================================================================


def normalize_lc_number(value: str) -> str:
    if not value:
        return ""
    return re.sub(r'[^A-Z0-9]', '', value.upper())
class LCExtractor:
    """Bulletproof Universal SWIFT MT Extractor with COMPLETE text extraction"""
    
    FIELD_MAPPINGS = COMPREHENSIVE_FIELD_MAPPINGS
    
    def __init__(self):
        self.current_doc = None
        self.message_types = SWIFT_MESSAGE_TYPES
    
    def extract_from_text(self, text: str) -> LCDocument:
        """Extract MT message data from ANY text format with Console Debugging"""
        
        # --- NEW DEBUG CONSOLE LOGS ---
        print("\n" + "="*50)
        print("EXTRACTOR DEBUG: NEW DOCUMENT DETECTED")
        print("-" * 50)
        # Print the first 500 characters to see if tags are there
        print(f"RAW TEXT START:\n{text[:500]}") 
        print("="*50 + "\n")
        # ------------------------------

        # Detect message type
        mt_type = self._detect_message_type(text)
        
        print(f"DETECTED MESSAGE TYPE: {mt_type}")  # Debug: print detected message type
        # Determine if LC or Amendment
        is_amendment = self._is_amendment(text, mt_type)

        print(f"IS AMENDMENT: {is_amendment}")  # Debug: print if it's an amendment
        if is_amendment:
            return self._extract_amendment(text, mt_type)
        else:
            return self._extract_lc(text, mt_type)
    



    def _strip_ocr_labels(self, text: str) -> str:
        # List of common labels found in OCR
        labels = [
            r'Documentary Credit Number', r"Sender's Reference", 
            r'Additional Conditions', r'Documents Required',
            r'Description of Goods', r'Date of Issue'
        ]
        clean_text = text
        for label in labels:
            # Case-insensitive removal of the label if it's at the start
            clean_text = re.sub(rf'^\s*{label}\s*', '', clean_text, flags=re.IGNORECASE)
        return clean_text.strip()
        
    def _detect_message_type(self, text: str) -> Optional[str]:
        # Use a pattern that doesn't rely on word boundaries or spaces
        patterns = [
            # This matches "Message type", "Messagetype", "Message-type", etc.
            r'Message\s*t?y?p?e?\s*:?\s*(\d{3})', 
            r'MT\s*(\d{3})',
            r'fin\.?\s*(\d{3})'
        ]
        
        # Pre-clean the text JUST for detection to handle the "squash"
        search_text = text.replace(" ", "").replace("\n", "")

        print(search_text[:500],'THe search text her')  # Debug: print the cleaned text snippet
        for pattern in patterns:
            # Search in the squashed text for "Messagetype:707"
            match = re.search(r'Messagetype:?(\d{3})', search_text, re.IGNORECASE)
            print(match,'The match here')  # Debug: print the match object
            if match:
                return match.group(1)
                
            # Fallback to standard search in original text
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
                
        return None


    def _is_amendment(self, text: str, mt_type: Optional[str]) -> bool:
        # 1️⃣ Check Message Type (detected by the squashed-friendly regex above)
        if mt_type in {'707', '747', '767'}:
            return True

        # 2️⃣ Mandatory amendment fields (Strong signals)
        # We look for the tag (26E) or labels without forcing spaces
        mandatory_indicators = [
            r'26E',                       
            r'Number\s*of\s*Amendment',   
            r'Date\s*of\s*Amendment',     
        ]

        for pattern in mandatory_indicators:
            if re.search(pattern, text, re.IGNORECASE):
                return True

        # 3️⃣ Amendment-only operations and field tags (4.2 and 4.3 specifically)
        amendment_signals = [
            r'45B', r'46B', r'47B',       # Amendment-only tags [cite: 1, 4]
            r'/ADD/', r'/DELETE/', r'/REPALL/' # Operation codes [cite: 1, 4]
        ]

        for pattern in amendment_signals:
            if re.search(pattern, text, re.IGNORECASE):
                return True

        return False

    def _extract_lc(self, text: str, mt_type: Optional[str] = None) -> LCDocument:
        """Extract LC from ANY text format"""
        doc = LCDocument(
            document_type="LC",
            lc_number="",
            message_type=f"MT{mt_type}" if mt_type else "MT700",
            raw_text=text
        )
        
        # Extract LC number - THREE FORMAT PATTERNS
        lc_patterns = [
            # Pattern 1: "20: Documentary Credit Number\n  ILC..."
            r'20:\s*Documentary Credit Number\s+([A-Z0-9]+)',
            # Pattern 2: ":20:ILC..."
            r':20:\s*([A-Z0-9]+)',
            # Pattern 3: "Documentary Credit Number\n  ILC..."
            r'Documentary Credit Number\s+([A-Z0-9]+)',
        ]
        for pattern in lc_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                doc.lc_number = normalize_lc_number(match.group(1))
                # doc.lc_number = match.group(1).strip()
                break
        
        # Extract issue date - THREE FORMAT PATTERNS
        date_patterns = [
            # Pattern 1: "31C: Date of Issue\n  230509"
            r'31C:\s*Date of Issue\s+(\d{6})',
            # Pattern 2: ":31C:230509"
            r':31C:\s*(\d{6})',
        ]
        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                doc.issue_date = match.group(1).strip()
                break
        
        # Extract sender
        sender_patterns = [
            r'To Institution:\s*([A-Z0-9]+)',
            r'52A:\s*Issuing Bank\s+([A-Z0-9]+)',
        ]
        for pattern in sender_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                doc.sender = match.group(1).strip()
                break
        
        # Extract receiver
        receiver_patterns = [
            r'To Institution:\s*([A-Z0-9]+)',
        ]
        for pattern in receiver_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                doc.receiver = match.group(1).strip()
                break
        
        # Extract all fields
        doc.fields = self._extract_all_fields(text)
        
        # Extract additional conditions (47A)
        doc.additional_conditions = self._extract_numbered_points_robust(text, ['47A:', ':47A:'])
        
        # Extract documents required (46A)
        doc.documents_required = self._extract_numbered_points_robust(text, ['46A:', ':46A:'])
        
        return doc
    
    def _extract_amendment(self, text: str, mt_type: Optional[str] = None) -> LCDocument:
        """Extract amendment with 'Squash-Proof' regex for messy OCR"""
        doc = LCDocument(
            document_type="AMENDMENT",
            lc_number="",
            message_type=f"MT{mt_type}" if mt_type else "MT707",
            raw_text=text
        )
        
        # 1. LC Number (Tag 20)
        # Flexible: matches ":20:", "20:", "Sender'sReference", or "20:Sender'sReference"
        lc_patterns = [
            r'20:?(?:\s*Sender\'s\s*Reference\s*)?([A-Z0-9]{10,})', # Capture long alphanumeric strings
            r'Sender\'s\s*Reference\s*([A-Z0-9]{10,})'
        ]
        for pattern in lc_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                doc.lc_number = normalize_lc_number(match.group(1))
                break
                
        # 2. Amendment Number (Tag 26E)
        # OCR Example: "26E:NumberofAmendment02"
        amend_patterns = [
            r'26E:?(?:\s*Number\s*of\s*Amendment\s*)?(\d+)',
            r'Number\s*of\s*Amendment\s*(\d+)'
        ]
        for pattern in amend_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                doc.amendment_number = match.group(1).strip()
                break

        # 3. Amendment Date (Tag 30)
        # OCR Example: "30:DateofAmendment230525"
        amend_date_patterns = [
            r'30:?(?:\s*Date\s*of\s*Amendment\s*)?(\d{6})',
            r'Date\s*of\s*Amendment\s*(\d{6})'
        ]
        for pattern in amend_date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                doc.amendment_date = match.group(1).strip()
                break

        # 4. Date of Issue (Tag 31C)
        issue_patterns = [
            r'31C:?(?:\s*Date\s*of\s*Issue\s*)?(\d{6})'
        ]
        for pattern in issue_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                doc.issue_date = match.group(1).strip()
                break

        # 5. Sender (Tag 52A)
        # OCR Example: "52A:IssuingBankHABBPKKA786"
        sender_patterns = [
            r'52A:?(?:\s*Issuing\s*Bank\s*)?([A-Z0-9]{8,11})'
        ]
        for pattern in sender_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                doc.sender = match.group(1).strip()
                break

        # 6. Receiver (To Institution)
        # OCR Example: "ToInstitution:QNBAQAQAXXX"
        receiver_patterns = [
            r'To\s*Institution\s*:?\s*([A-Z0-9]{8,11})'
        ]
        for pattern in receiver_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                doc.receiver = match.group(1).strip()
                break

        # 7. Field-by-Field and Clause Extraction
        doc.fields = self._extract_all_fields(text)
        doc.additional_conditions = self._extract_amendment_changes_complete(text, ['47B'])
        doc.documents_required = self._extract_amendment_changes_complete(text, ['46B'])
        
        desc_changes = self._extract_amendment_changes_complete(text, ['45B'])
        if desc_changes:
            for change in desc_changes:
                change['field_code'] = '45B'
            doc.additional_conditions.extend(desc_changes)
        
        return doc


    def _extract_all_fields(self, text: str) -> Dict[str, LCField]:
        fields = {}
        
        # IMPROVED REGEX: 
        # 1. Matches tags at start of line or after a newline
        # 2. Matches tags even if preceded by a letter (e.g., F20: or Reference20:)
        # 3. Requires a colon after the 2-3 digit code
        flexible_pattern = r'(?:^|\n|[a-zA-Z])\s*:?(\d{2,3}[A-Z]?):'
        
        matches = list(re.finditer(flexible_pattern, text))
        
        for i, match in enumerate(matches):
            field_num = match.group(1)
            start_pos = match.end()
            # The field ends where the next tag begins, or at the end of the document
            end_pos = matches[i+1].start() if i < len(matches) - 1 else len(text)
            
            raw_content = text[start_pos:end_pos].strip()
            
            # --- NOISY OCR CLEANING LOGIC ---
            # In your HBL/UBL docs, labels like "Documentary Credit Number" 
            # often appear right after the tag before the actual value.
            content_lines = raw_content.split('\n')
            
            if content_lines:
                first_line = content_lines[0].strip()
                # List of labels to ignore if they appear at the very start of a field
                noise_labels = [
                    "Documentary Credit Number", "Sender's Reference", "Receiver's Reference",
                    "Issuing Bank's Reference", "Number of Amendment", "Date of Amendment",
                    "Date of Issue", "Date and Place of Expiry", "Applicable Rules",
                    "Form of Documentary Credit", "Additional Conditions", "Documents Required",
                    "Description of Goods", "Narrative", "Lines 2-100", "Lines2to100"
                ]
                
                # Check if the first line is just one of these labels
                is_label = any(label.lower() in first_line.lower() for label in noise_labels)
                
                # If the first line is a known label and contains NO digits, 
                # or if it's explicitly a "Narrative" marker, skip it.
                if is_label and (not any(char.isdigit() for char in first_line) or "Narrative" in first_line):
                    field_value = "\n".join(content_lines[1:]).strip()
                else:
                    field_value = raw_content
            else:
                field_value = raw_content

            field_code = f':{field_num}:'
            
            # Store the field
            fields[field_code] = LCField(
                field_code=field_code,
                field_name=self.FIELD_MAPPINGS.get(field_code, f"Field {field_num}"),
                value=self._clean_field_value(field_value),
                raw_text=raw_content
            )
            
        return fields

    def _extract_numbered_points_robust(self, text: str, field_codes: List[str]) -> List[Dict]:
        """
        PROPERLY extract ALL numbered points as INDIVIDUAL items.
        Handles (1), 1., and OCR-squashed numbers like '1.TEXT'
        """
        points = []
        
        # We use the already-extracted fields to ensure consistency
        if not hasattr(self, 'current_doc_fields') or not self.current_doc_fields:
            self.current_doc_fields = self._extract_all_fields(text)
            
        for code in field_codes:
            # Normalize the field code for lookup (e.g., '47A' -> ':47A:')
            lookup = code if code.startswith(':') else f':{code.rstrip(":")}:'
            
            if lookup not in self.current_doc_fields:
                continue
                
            content = self.current_doc_fields[lookup].raw_text
            
            # --- ROBUST POINT SPLITTING ---
            # This regex matches:
            # 1. (1) or (12)
            # 2. 1. or 12.
            # 3. Numbers at the start of a new line followed by a space
            split_pattern = r'(?:\n|^)\s*(?:\(?(\d+)\)?[\.\s]|(\d+)\.)'
            
            # Find all numbering positions
            matches = list(re.finditer(split_pattern, content))
            
            if not matches:
                # If no numbered points found, treat the whole block as Point 1
                clean_text = self._clean_field_value(content)
                if clean_text:
                    points.append({
                        'point_number': 1,
                        'text': clean_text,
                        'field_code': lookup.strip(':')
                    })
                continue

            for i, match in enumerate(matches):
                point_num = match.group(1) or match.group(2)
                start_pos = match.end()
                # Point ends where the next number starts
                end_pos = matches[i+1].start() if i < len(matches) - 1 else len(content)
                
                point_text = content[start_pos:end_pos]
                
                # Clean OCR noise (Narrative, Lines 2-100, etc.) from the point text
                clean_point_text = self._clean_field_value(point_text)
                
                if clean_point_text:
                    points.append({
                        'point_number': int(point_num),
                        'text': clean_point_text,
                        'field_code': lookup.strip(':')
                    })
        
        # Remove duplicates and sort by point number
        seen_points = set()
        unique_points = []
        for p in sorted(points, key=lambda x: x['point_number']):
            if p['point_number'] not in seen_points:
                unique_points.append(p)
                seen_points.add(p['point_number'])
                
        return unique_points
  

    def _extract_amendment_changes_complete(self, text: str, field_codes: List[str]) -> List[Dict]:
        """Extracts complete amendment narrative, fixing truncation and OCR noise."""
        changes = []
        # Uses the flexible field extractor we just updated
        fields = self._extract_all_fields(text) 

        for code in field_codes:
            lookup = code if code.startswith(':') else f':{code.rstrip(":")}:'
            if lookup not in fields:
                continue
                
            # Use raw_text to ensure we see the /ADD/ or /REPALL/ tags
            content = fields[lookup].raw_text
            
            # 1. Identify all SWIFT amendment operations in the field
            operations = []
            for op_match in re.finditer(r'/(ADD|DELETE|REPALL)/', content, re.IGNORECASE):
                operations.append({
                    'type': op_match.group(1).upper(),
                    'start': op_match.end(),
                    'token': op_match.group(0)
                })

            # 2. Extract the text between operations
            for i, op in enumerate(operations):
                # End is either the start of the next operation or the end of the field
                end_pos = operations[i + 1]['start'] - len(operations[i + 1]['token']) if i < len(operations) - 1 else len(content)
                op_text = content[op['start']:end_pos].strip()
                
                # --- NOISY OCR CLEANING ---
                # Remove artifacts like "Lines 2-100", "Narrative:", etc., that appear mid-amendment
                op_text = re.sub(r'Lines\s?\d?\s?to\s?\d+:?', '', op_text, flags=re.IGNORECASE)
                op_text = re.sub(r'Lines\s\d+-\d+:?', '', op_text, flags=re.IGNORECASE)
                op_text = re.sub(r'Narrativel?:?', '', op_text, flags=re.IGNORECASE)
                
                # Remove HBL/UBL specific leading noise like '+)' or '/'
                op_text = re.sub(r'^[+)\s/]+', '', op_text) 
                
                # Normalize spaces
                op_text = re.sub(r'\s+', ' ', op_text).strip()

                if op_text:
                    change = {
                        'operation': op['type'],
                        'field_code': lookup.strip(':'),
                        'narrative': op_text,
                        'change_text': op_text
                    }

                    # 3. ADVANCED POINT NUMBER DETECTION
                    # Matches "CLAUSE NO.10", "NO. 21", "FIELD 47A-11", or "POINT 5"
                    # Updated to handle OCR errors like "CLAUE"
                    point_match = re.search(
                        r'(?:CLAU[S|E]*\s+)?NO\.?\s*(\d+)|FIELD\s+\d+[A-Z]?[-]?(\d+)', 
                        op_text, 
                        re.IGNORECASE
                    )
                    
                    if point_match:
                        # Take whichever group (1 or 2) matched the number
                        p_num = point_match.group(1) or point_match.group(2)
                        change['point_number'] = int(p_num)
                    
                    changes.append(change)
        
        return changes
   
    def _clean_field_value(self, value: str) -> str:
        """Clean field value from OCR artifacts and SWIFT labels"""
        if not value:
            return ""

        # 1. Handle OCR Artifacts globally within the text
        # Removes "Lines2to100:", "Lines 2-100:", "Narrative:", etc.
        noise_patterns = [
            r'Lines\s?\d?\s?to\s?\d+:?', 
            r'Lines\s\d+-\d+:?',
            r'Narrativel?:?', # Catches 'Narrative:' and the OCR error 'Narrativel:'
            r'Code:?'
        ]
        
        for pattern in noise_patterns:
            value = re.sub(pattern, '', value, flags=re.IGNORECASE)

        # 2. Normalize whitespace (Convert all newlines/tabs to single spaces)
        value = re.sub(r'\s+', ' ', value)

        # 3. Strip specific SWIFT field prefixes from the START of the value
        # These are labels that often get merged into the data field
        prefixes = [
            'Name and Address:', 'Currency:', 'Date:', 'Place:', 
            'Number:', 'Total:', 'Amount:', 'Days:', 'Party Identifier:',
            'Account:', 'Settlement Amount:'
        ]
        
        for prefix in prefixes:
            value = re.sub(f'^{re.escape(prefix)}\s*', '', value, flags=re.IGNORECASE)

        # 4. Final Polish
        # Remove any leading/trailing weird symbols like +) or / common in HBL
        value = value.strip()
        value = re.sub(r'^[+)\s/]+', '', value) 
        
        return value.strip()


# ============================================================================
# CONSOLIDATOR
# ============================================================================

class LCConsolidator:
    """Consolidate LC with amendments"""
    
    def __init__(self):
        self.lcs: Dict[str, LCDocument] = {}
        self.amendments: Dict[str, List[LCDocument]] = {}
    
    def add_document(self, doc: LCDocument):
        """Add document"""
        doc.lc_number = normalize_lc_number(doc.lc_number)
        if doc.document_type == "LC":
            self.lcs[doc.lc_number] = doc
        else:
            if doc.lc_number not in self.amendments:
                self.amendments[doc.lc_number] = []
            self.amendments[doc.lc_number].append(doc)
    



    def consolidate(self, lc_number: str) -> Dict:
        """Consolidate LC with amendments"""
        if lc_number not in self.lcs:
            return None
        
        original_lc = self.lcs[lc_number]
        amendments = self.amendments.get(lc_number, [])
        amendments.sort(key=lambda x: int(x.amendment_number) if x.amendment_number else 0)
        
        consolidated = {
            'lc_number': lc_number,
            'original_issue_date': original_lc.issue_date,
            'sender': original_lc.sender,
            'receiver': original_lc.receiver,
            'message_type': 'MT700_CONSOLIDATED',
            'amendments_applied': len(amendments),
            'last_amendment_date': amendments[-1].amendment_date if amendments else None,
            'fields': {k: asdict(v) for k, v in original_lc.fields.items()},
            'additional_conditions': original_lc.additional_conditions.copy(),
            'documents_required': original_lc.documents_required.copy(),
            'amendment_history': []
        }
        
        for amendment in amendments:
            amendment_record = {
                'amendment_number': amendment.amendment_number,
                'amendment_date': amendment.amendment_date,
                'changes': []
            }
            
            for change in amendment.additional_conditions:
                self._apply_change(consolidated['additional_conditions'], change, '47A')
                amendment_record['changes'].append(change)
            
            for change in amendment.documents_required:
                self._apply_change(consolidated['documents_required'], change, '46A')
                amendment_record['changes'].append(change)
            
            consolidated['amendment_history'].append(amendment_record)
        
        if consolidated['additional_conditions']:
            consolidated['additional_conditions'].sort(key=lambda x: x.get('point_number', 999))
        if consolidated['documents_required']:
            consolidated['documents_required'].sort(key=lambda x: x.get('point_number', 999))
        
        return consolidated
    
    def _apply_change(self, points_list: List[Dict], change: Dict, field_code: str):
        """
        Applies amendment logic to the original LC points.
        This is where the 'Update' actually happens.
        """
        operation = change.get('operation', 'ADD')
        target_point = change.get('point_number')
        new_text = change.get('change_text', change.get('narrative', ''))

        # 1. DELETE OPERATION
        if operation == 'DELETE':
            if target_point:
                # Remove the specific numbered clause
                points_list[:] = [p for p in points_list if p.get('point_number') != target_point]
            else:
                # If no point number, we look for a text match (fallback)
                points_list[:] = [p for p in points_list if new_text.lower() not in p.get('text', '').lower()]

        # 2. REPLACE ALL / MODIFY OPERATION (Crucial for your "REPALL" tags)
        elif operation == 'REPALL':
            if target_point:
                found = False
                for i, point in enumerate(points_list):
                    if point.get('point_number') == target_point:
                        points_list[i]['text'] = new_text
                        points_list[i]['modified_by_amendment'] = True
                        found = True
                        break
                # If the point didn't exist in original, treat REPALL as an ADD
                if not found:
                    operation = 'ADD' 
            else:
                # If REPALL is used without a point number, it usually means 
                # replace the WHOLE field.
                points_list.clear()
                operation = 'ADD'

        # 3. ADD OPERATION
        if operation == 'ADD':
            # Check if this point number already exists to avoid duplicates
            if target_point and any(p.get('point_number') == target_point for p in points_list):
                # If it exists, we update it (common in messy amendments)
                for point in points_list:
                    if point.get('point_number') == target_point:
                        point['text'] = new_text
                        point['modified_by_amendment'] = True
            else:
                # Add as a brand new point
                points_list.append({
                    'point_number': target_point if target_point else (max([p.get('point_number', 0) for p in points_list], default=0) + 1),
                    'text': new_text,
                    'field_code': field_code,
                    'added_by_amendment': True
                })
    def get_all_consolidated(self) -> List[Dict]:
        """Get all consolidated LCs"""
        results = []
        for lc_number in self.lcs.keys():
            consolidated = self.consolidate(lc_number)
            if consolidated:
                results.append(consolidated)
        return results


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def process_lc_documents(file_paths: List[str], output_path: str = None) -> Dict:
    """Process multiple LC documents with granular text extraction logs"""
    extractor = LCExtractor()
    consolidator = LCConsolidator()
    
    results = {
        'processing_date': datetime.now().isoformat(),
        'total_documents_processed': 0,
        'lcs_found': 0,
        'amendments_found': 0,
        'documents': [],
        'consolidated_lcs': []
    }

    for file_path in file_paths:
        try:
            print(f"\n{'='*30}")
            print(f"DEBUGGING FILE: {file_path}")
            print(f"{'='*30}")
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            
            # 1. PRINT RAW SOURCE TEXT
            print(f"\n[STEP 1: RAW TEXT FROM OCR]")
            print("-" * 20)
            # Printing first 1000 chars to see headers and field starts
            print(text[:1000]) 
            print("-" * 20)

            doc = extractor.extract_from_text(text)
            
            # 2. PRINT EXTRACTION RESULTS
            print(f"\n[STEP 2: EXTRACTED METADATA]")
            print(f"Document Type: {doc.document_type}")
            print(f"LC Number:     {doc.lc_number}")
            print(f"Message Type:  {doc.message_type}")
            print(f"Amend Number:  {doc.amendment_number}")
            
            # 3. PRINT FIELD-BY-FIELD ANALYSIS
            print(f"\n[STEP 3: EXTRACTED FIELDS CONTENT]")
            for code, field in doc.fields.items():
                # This helps see if the value contains 'Sender's Reference' or extra noise
                print(f"Field {code:6} | Value: {field.value[:100]}...")

            consolidator.add_document(doc)
            
            results['total_documents_processed'] += 1
            if doc.document_type == 'LC':
                results['lcs_found'] += 1
            else:
                results['amendments_found'] += 1
            
            results['documents'].append(asdict(doc))
            
        except Exception as e:
            print(f"ERROR processing {file_path}: {str(e)}")
            continue
    
    results['consolidated_lcs'] = consolidator.get_all_consolidated()
    return results
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        files = sys.argv[1:]
        output = "lc_consolidated_output.json"
        results = process_lc_documents(files, output)
        print(f"\n{'='*70}")
        print("Processing Complete")
        print(f"{'='*70}")
        print(f"Documents processed: {results['total_documents_processed']}")
        print(f"LCs found: {results['lcs_found']}")
        print(f"Amendments found: {results['amendments_found']}")
        print(f"Consolidated: {len(results['consolidated_lcs'])} LCs")
    else:
        print("Usage: python lc_extractor.py <file1> <file2> ...")