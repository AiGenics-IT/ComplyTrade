"""
Data Models
Data structures for LC fields, documents, and related entities
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


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