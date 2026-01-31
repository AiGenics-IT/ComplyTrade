"""
PDF Processor Module
Enhanced PDF processing with OCR and multi-document support
"""

import os
import tempfile
from typing import List, Dict
from .ocr_engine import EnhancedOCRProcessor
from services.document_splitter import MultiDocumentSplitter


class PDFProcessorEnhanced:
    """Enhanced PDF processor with multi-document support"""
    
    def __init__(self, ocr_backend='tesseract', use_preprocessing=True, use_postprocessing=True):
        self.ocr_processor = EnhancedOCRProcessor(
            backend=ocr_backend, 
            use_preprocessing=use_preprocessing,
            use_postprocessing=use_postprocessing
        )
        self.splitter = MultiDocumentSplitter()
    
    def extract_text_from_pdf(self, pdf_path: str, use_ocr=False, aggressive_ocr=False,
                             aggressive_postprocessing=False, split_documents=True) -> str:
        """
        Extract text from PDF with optional document splitting
        
        Args:
            pdf_path: Path to PDF
            use_ocr: Force OCR
            aggressive_ocr: Aggressive preprocessing
            aggressive_postprocessing: Aggressive text cleanup
            split_documents: Return split documents (for multi-doc PDFs)
        
        Returns:
            Extracted text (combined or split)
        """
        text = ""
        
        try:
            if not use_ocr:
                text = self._extract_text_layer(pdf_path)
            
            if not text.strip() or use_ocr:
                print(f"Using OCR with post-processing for: {pdf_path}")
                text = self._extract_with_enhanced_ocr(pdf_path, aggressive_ocr, aggressive_postprocessing)
            
        except Exception as e:
            print(f"Error processing PDF: {str(e)}")
        
        return text
    
    def extract_and_split(self, pdf_path: str, use_ocr=False) -> List[Dict[str, str]]:
        """
        Extract and split into multiple documents
        
        Returns:
            List of {text, type, index} dictionaries
        """
        # Get combined text
        combined_text = self.extract_text_from_pdf(pdf_path, use_ocr=use_ocr)
        
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
        
        return results
    
    def _extract_text_layer(self, pdf_path: str) -> str:
        """Extract text layer from digital PDF"""
        try:
            import pdfplumber
            
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            
            return text
        except ImportError:
            print("⚠ pdfplumber not available")
            return ""
    
    def _extract_with_enhanced_ocr(self, pdf_path: str, aggressive_preprocess: bool = False,
                                   aggressive_postprocess: bool = False) -> str:
        """Extract with OCR and post-processing"""
        try:
            from pdf2image import convert_from_path
            
            images = convert_from_path(pdf_path, dpi=300, fmt='png')
            
            text = ""
            for i, img in enumerate(images):
                print(f"  Processing page {i+1}/{len(images)}...")
                
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    temp_img_path = tmp.name
                
                try:
                    img.save(temp_img_path, 'PNG', quality=100)
                    
                    page_text = self.ocr_processor.extract_text_from_image(
                        temp_img_path, 
                        aggressive_preprocessing=aggressive_preprocess,
                        aggressive_postprocessing=aggressive_postprocess
                    )
                    
                    page_text = "".join(c for c in page_text if c.isprintable() or c in "\n\r\t")
                    
                    text += f"\n--- Page {i+1} ---\n{page_text}\n"
                    
                finally:
                    if os.path.exists(temp_img_path):
                        os.remove(temp_img_path)
            
            return text
        
        except ImportError:
            print("⚠ pdf2image not available")
            return ""
        except Exception as e:
            print(f"⚠ OCR Failed: {str(e)}")
            return ""