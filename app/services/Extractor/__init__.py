"""
Universal SWIFT MT Message Extraction System - FINAL PRODUCTION VERSION
✅ Extracts COMPLETE amendment text (fixes "CLAUSE NO." truncation bug)
✅ Handles ALL text formats: "47B: Label\n text" and ":47B:text" and "F47B:text"
✅ Properly extracts all numbered points (1), (2), (3)... separately
✅ 100% BACKWARD COMPATIBLE with lc_api.py

Modular architecture for better maintainability and testing.
"""

from .lc_extractor import LCExtractor
from .lc_consolidator import LCConsolidator
from .lc_document_processor import process_lc_documents
from .models import LCDocument, LCField

from .constants import (
    SWIFT_MESSAGE_TYPES,
    COMPREHENSIVE_FIELD_MAPPINGS,
    SUPPORTING_DOCUMENT_TYPES
)

__all__ = [
    'LCExtractor',
    'LCConsolidator',
    'process_lc_documents',
    'LCDocument',
    'LCField',
    'SWIFT_MESSAGE_TYPES',
    'COMPREHENSIVE_FIELD_MAPPINGS',
    'SUPPORTING_DOCUMENT_TYPES'
]

__version__ = '1.0.0'