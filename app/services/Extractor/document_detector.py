"""
Document Detector
Detects message types and categorizes supporting documents
"""

import re
from typing import Optional
from .constants import SWIFT_MESSAGE_TYPES, SUPPORTING_DOCUMENT_TYPES


class DocumentDetector:
    """Detects and categorizes LC documents"""
    
    def __init__(self):
        self.message_types = SWIFT_MESSAGE_TYPES
        self.supporting_types = SUPPORTING_DOCUMENT_TYPES
    
    def detect_message_type(self, text: str) -> Optional[str]:
        """Detect SWIFT message type from text"""
        # Use a pattern that doesn't rely on word boundaries or spaces
        patterns = [
            # This matches "Message type", "Messagetype", "Message-type", etc.
            r'Message\s*t?y?p?e?\s*:?\s*(\d{3})', 
            r'MT\s*(\d{3})',
            r'fin\.?\s*(\d{3})'
        ]
        
        # Pre-clean the text JUST for detection to handle the "squash"
        search_text = text.replace(" ", "").replace("\n", "")

        for pattern in patterns:
            # Search in the squashed text for "Messagetype:707"
            match = re.search(r'Messagetype:?(\d{3})', search_text, re.IGNORECASE)
            if match:
                return match.group(1)
            # Fallback to standard search in original text
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
                
        return None
    
    def is_amendment(self, text: str, mt_type: Optional[str]) -> bool:
        """Determine if document is an amendment"""
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
            r'45B', r'46B', r'47B',       # Amendment-only tags
            r'/ADD/', r'/DELETE/', r'/REPALL/' # Operation codes
        ]

        for pattern in amendment_signals:
            if re.search(pattern, text, re.IGNORECASE):
                return True

        return False
    
    def categorize_supporting_doc(self, text: str, file_name: str = "Unknown File") -> dict:
        """
        Categorize non-SWIFT supporting documents (Invoice, BL, Certificates, etc.)
        Returns a dictionary with classification information
        """
        text_upper = text.upper()

        detected_type = "UNKNOWN_SUPPORTING"
        matched_keywords = []
        confidence = 0

        # 1️⃣ Document type scoring
        for doc_type, keywords in self.supporting_types.items():
            hits = [kw for kw in keywords if kw in text_upper]
            if hits and len(hits) > confidence:
                detected_type = doc_type
                matched_keywords = hits
                confidence = len(hits)

        # 🔍 CONSOLE LOG: Classification Result
        print(f"\n[CLASSIFICATION] File: {file_name}")
        print(f"               Detected Category: {detected_type}")
        print(f"               Confidence Score:  {confidence} keywords matched")

        return {
            'type': detected_type,
            'confidence': confidence,
            'matched_keywords': matched_keywords,
            'is_supporting': True,
            'file_name': file_name
        }
    
    def find_lc_reference_in_supporting_doc(self, text: str) -> str:
        """Finds LC Number in non-SWIFT documents like Invoices."""
        from utils import normalize_lc_number
        
        patterns = [
            r'L/?C\s*(?:NO\.?|NUMBER)?\s*:?\s*([A-Z0-9/]{5,})',
            r'DOCUMENTARY\s*CREDIT\s*(?:NO\.?|NUMBER)?\s*:?\s*([A-Z0-9/]{5,})',
            r'CREDIT\s*NUMBER\s*:?\s*([A-Z0-9/]{5,})'
        ]
        for p in patterns:
            match = re.search(p, text, re.IGNORECASE)
            if match:
                # Use normalize function to clean it
                return normalize_lc_number(match.group(1))
        return "UNKNOWN"