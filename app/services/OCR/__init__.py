"""
Enhanced OCR Module with AI Hybrid Layer
Automatically uses AI for poor quality documents and provides clean text output
"""

from .image_preprocessor import ImagePreprocessor
from .ocr_engine import EnhancedOCRProcessor
from .pdf_processor import PDFProcessorEnhanced
from .document_processor import DocumentProcessor
from .ai_hybrid_layer import AIHybridProcessor, create_hybrid_processor

__all__ = [
    'ImagePreprocessor',
    'EnhancedOCRProcessor',
    'PDFProcessorEnhanced',
    'DocumentProcessor',
    'AIHybridProcessor',
    'create_hybrid_processor'
]

__version__ = '2.0.0'

if __name__ == "__main__":
    print("Enhanced OCR Processor with AI Hybrid Layer")
    print("Automatically uses AI for poor quality documents")
    print("Version: 2.0.0")