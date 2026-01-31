"""
Enhanced OCR Module with Multi-Document Support
Automatically splits and processes multiple documents from a single PDF
"""

from .image_preprocessor import ImagePreprocessor
from .ocr_engine import EnhancedOCRProcessor
from .pdf_processor import PDFProcessorEnhanced
from .document_processor import DocumentProcessor

__all__ = [
    'ImagePreprocessor',
    'EnhancedOCRProcessor',
    'PDFProcessorEnhanced',
    'DocumentProcessor'
]

__version__ = '1.0.0'

if __name__ == "__main__":
    print("Enhanced OCR Processor with Multi-Document Support")
    print("Automatically splits combined PDFs into individual documents")   