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
    
    def __init__(self, ocr_backend='tesseract', use_preprocessing=True, use_postprocessing=True, use_api=True):
        """
        Initialize document processor
        
        Args:
            ocr_backend: OCR engine to use ('tesseract', 'easyocr', 'paddleocr')
            use_preprocessing: Enable image preprocessing
            use_postprocessing: Enable text post-processing
            use_api: If True, ALL files are sent to OCR API instead of local processing
        """
        print(f"[DocumentProcessor] Initializing with use_api={use_api}")
        
        self.use_api = use_api
        
        # Initialize OCR processor with API flag
        self.ocr_processor = EnhancedOCRProcessor(
            backend=ocr_backend,
            use_preprocessing=use_preprocessing,
            use_postprocessing=use_postprocessing,
            use_api=use_api
        )
        
        # Only initialize PDF processor if NOT using API
        if not use_api:
            self.pdf_processor = PDFProcessorEnhanced(
                ocr_backend=ocr_backend,
                use_preprocessing=use_preprocessing,
                use_postprocessing=use_postprocessing
            )
            print("[DocumentProcessor] PDF processor initialized")
        else:
            self.pdf_processor = None
            print("[DocumentProcessor] Skipping PDF processor - using API for all files")
        
        # Initialize document splitter
        try:
            self.splitter = MultiDocumentSplitter()
        except Exception as e:
            print(f"⚠ Document splitter not available: {e}")
            self.splitter = None
    
    def process_document(self, file_path: str, force_ocr=False, aggressive_ocr=False,
                         aggressive_postprocessing=False, split_multi=True) -> str:
        """
        Process document with optional splitting.
        If `use_api=True`, ALL files are sent to OCR API regardless of type.
        
        Args:
            file_path: Path to file
            force_ocr: Force OCR for PDFs (ignored if use_api=True)
            aggressive_ocr: Aggressive preprocessing (ignored if use_api=True)
            aggressive_postprocessing: Aggressive cleanup (ignored if use_api=True)
            split_multi: Return split documents for multi-doc files
        
        Returns:
            Extracted text
        """
        file_path = Path(file_path)
        ext = file_path.suffix.lower()
        
        print(f"\n[DocumentProcessor] Processing: {file_path.name}")
        print(f"[DocumentProcessor] File type: {ext}, use_api={self.use_api}")
        
        # ====== API MODE: Route ALL file types to API ======
        if self.use_api:
            print(f"[DocumentProcessor] API mode active - routing to API OCR")
            return self.ocr_processor._ocr_via_api(str(file_path))
        
        # ====== LOCAL MODE: Process based on file type ======
        print(f"[DocumentProcessor] Local mode active - processing with local engines")
        
        if ext == '.pdf':
            if self.pdf_processor is None:
                raise RuntimeError("PDF processor not initialized. Cannot process PDFs in local mode.")
            
            return self.pdf_processor.extract_text_from_pdf(
                str(file_path), 
                use_ocr=force_ocr, 
                aggressive_ocr=aggressive_ocr,
                aggressive_postprocessing=aggressive_postprocessing
            )
        
        elif ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif']:
            return self.ocr_processor.extract_text_from_image(
                str(file_path),
                aggressive_preprocessing=aggressive_ocr,
                aggressive_postprocessing=aggressive_postprocessing
            )
        
        elif ext in ['.txt', '.text']:
            try:
                return file_path.read_text(encoding='utf-8', errors='ignore')
            except Exception as e:
                print(f"⚠ Error reading text file: {e}")
                return ""
        
        else:
            print(f"⚠ Unsupported format: {ext}")
            return ""
    
    def process_and_split(self, file_path: str, force_ocr=False) -> List[Dict[str, str]]:
        """
        Process file and split into multiple documents
        
        Args:
            file_path: Path to file
            force_ocr: Force OCR for PDFs
        
        Returns:
            List of {text, type, index} dictionaries
        """
        if self.splitter is None:
            print("⚠ Document splitter not available")
            # Return single document
            text = self.process_document(file_path, force_ocr=force_ocr)
            return [{
                'index': 1,
                'type': 'unknown',
                'text': text,
                'length': len(text)
            }]
        
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