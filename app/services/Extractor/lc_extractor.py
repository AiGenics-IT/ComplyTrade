"""
LC Extractor
Main extraction logic for LC and Amendment documents
"""

import re
from typing import Optional, Any
from .models import LCDocument
from .constants import COMPREHENSIVE_FIELD_MAPPINGS, SWIFT_MESSAGE_TYPES
from .utils import normalize_lc_number, sanitize_text
from .field_extractor import FieldExtractor
from .document_detector import DocumentDetector
from services.ai_postProcessor import AIOCRPostProcessor


class LCExtractor:
    """Bulletproof Universal SWIFT MT Extractor with COMPLETE text extraction"""
    
    def __init__(self):
        self.current_doc = None
        self.current_doc_fields = None
        self.field_mappings = COMPREHENSIVE_FIELD_MAPPINGS
        self.message_types = SWIFT_MESSAGE_TYPES
        self.ocr_cleaner = AIOCRPostProcessor()
        self.field_extractor = FieldExtractor()
        self.detector = DocumentDetector()

    def extract_from_text(self, text: str) -> Any:
        """Extract data or categorize supporting documents for later validation."""

        print("\n[LCExtractor] Starting extraction process... before Postprocessing\n", text)
        # text = self.ocr_cleaner.clean_text(text)
        print("\n[LCExtractor] Starting extraction process... after Postprocessing\n", text)

        # 1. Clean the incoming OCR text immediately (Fixes charmap error)
        text = sanitize_text(text)
        
        # 2. Detect if it's a SWIFT message
        mt_type = self.detector.detect_message_type(text)
        
        # 3. CLASSIFICATION LOGIC
        if mt_type:
            # It's a SWIFT message (LC or Amendment)
            is_amendment = self.detector.is_amendment(text, mt_type)
            if is_amendment:
                return self._extract_amendment(text, mt_type)
            else:
                return self._extract_lc(text, mt_type)
        
        # 4. SUPPORTING DOCUMENT LOGIC (The "Skip but Keep" part)
        # If no MT type is found, it's likely an Invoice, BL, or Certificate
        return self._categorize_supporting_doc(text)

    def _categorize_supporting_doc(self, text: str, file_name: str = "Unknown File") -> LCDocument:
        """
        Categorize non-SWIFT supporting documents (Invoice, BL, Certificates, etc.)
        """
        classification = self.detector.categorize_supporting_doc(text, file_name)
        
        # Create a unified LCDocument (NO LC NUMBER YET)
        doc = LCDocument(
            document_type=classification['type'],
            lc_number="PENDING",              # ⬅️ Assigned later
            message_type="NON_SWIFT",
            raw_text=text,
            fields={},                        # Supporting docs don't have SWIFT fields
            additional_conditions=[],
            documents_required=[]
        )

        # Attach classification metadata
        doc.is_supporting = True
        doc.file_name = file_name             # Store filename for reference
        doc.classification = {
            "confidence": classification['confidence'],
            "matched_keywords": classification['matched_keywords']
        }
        doc.status = "stored_for_validation"

        return doc

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
        doc.fields = self.field_extractor.extract_all_fields(text)
        self.current_doc_fields = doc.fields
        
        # Extract additional conditions (47A)
        doc.additional_conditions = self.field_extractor.extract_numbered_points_robust(
            text, ['47A:', ':47A:'], doc.fields
        )
        
        # Extract documents required (46A)
        doc.documents_required = self.field_extractor.extract_numbered_points_robust(
            text, ['46A:', ':46A:'], doc.fields
        )
        
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
        doc.fields = self.field_extractor.extract_all_fields(text)
        doc.additional_conditions = self.field_extractor.extract_amendment_changes_complete(text, ['47B'])
        doc.documents_required = self.field_extractor.extract_amendment_changes_complete(text, ['46B'])
        
        desc_changes = self.field_extractor.extract_amendment_changes_complete(text, ['45B'])
        if desc_changes:
            for change in desc_changes:
                change['field_code'] = '45B'
            doc.additional_conditions.extend(desc_changes)
        
        return doc