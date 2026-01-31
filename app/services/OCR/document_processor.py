"""
Document Processor Module
Universal document processor with multi-document support
"""

from pathlib import Path
from typing import List, Dict
from .pdf_processor import PDFProcessorEnhanced
from .ocr_engine import EnhancedOCRProcessor
from services.document_splitter import MultiDocumentSplitter


class DocumentProcessor:
    """Universal document processor with multi-document support"""
    
    def __init__(self, ocr_backend='tesseract', use_preprocessing=True, use_postprocessing=True):
        self.pdf_processor = PDFProcessorEnhanced(
            ocr_backend=ocr_backend,
            use_preprocessing=use_preprocessing,
            use_postprocessing=use_postprocessing
        )
        self.ocr_processor = EnhancedOCRProcessor(
            backend=ocr_backend,
            use_preprocessing=use_preprocessing,
            use_postprocessing=use_postprocessing
        )
        self.splitter = MultiDocumentSplitter()
    
    def process_document(self, file_path: str, force_ocr=False, aggressive_ocr=False,
                        aggressive_postprocessing=False, split_multi=True) -> str:
        """
        Process document with optional splitting
        
        Args:
            file_path: Path to file
            force_ocr: Force OCR
            aggressive_ocr: Aggressive preprocessing
            aggressive_postprocessing: Aggressive cleanup
            split_multi: Return split documents for multi-doc files
        
        Returns:
            Extracted text
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        print(f"Processing: {file_path.name}")
        
        if extension == '.pdf':
            return self.pdf_processor.extract_text_from_pdf(
                str(file_path), 
                use_ocr=force_ocr, 
                aggressive_ocr=aggressive_ocr,
                aggressive_postprocessing=aggressive_postprocessing
            )
        
        elif extension in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif']:
            return self.ocr_processor.extract_text_from_image(
                str(file_path),
                aggressive_preprocessing=aggressive_ocr,
                aggressive_postprocessing=aggressive_postprocessing
            )
        
        elif extension in ['.txt', '.text']:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        
        else:
            print(f"⚠ Unsupported format: {extension}")
            return ""
    
    def process_and_split(self, file_path: str, force_ocr=False) -> List[Dict[str, str]]:
        """
        Process file and split into multiple documents
        
        Returns:
            List of {text, type, index} dictionaries
        """
        # Get combined text
        combined_text = self.process_document(file_path, force_ocr=force_ocr)
        
        # Split into documents
        split_docs = self.splitter.split_documents(combined_text)
        
        # Format results
        results = []
        for i, (doc_text, doc_type) in enumerate(split_docs):
            results.append({
                'index': i + 1,
                'type': doc_type,
                'text': doc_text,
                'length': len(doc_text)
            })
        
        print(f"\n✓ Extracted {len(results)} document(s) from {Path(file_path).name}")
        for doc in results:
            print(f"  {doc['index']}. {doc['type']} ({doc['length']} chars)")
        
        return results