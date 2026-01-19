"""
OCR Module for LC Document Processing
Supports multiple OCR backends: Tesseract, EasyOCR, PaddleOCR
Handles PDFs, images, and scanned documents
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import json
from PIL import Image
import numpy as np


class OCRProcessor:
    """Unified OCR processor with multiple backend support"""
    
    def __init__(self, backend='tesseract', language='eng'):
        """
        Initialize OCR processor
        
        Args:
            backend: 'tesseract', 'easyocr', or 'paddleocr'
            language: Language code (default: 'eng' for English)
        """
        self.backend = backend
        self.language = language
        self.ocr_engine = None
        
        self._initialize_backend()
    
    def _initialize_backend(self):
        """Initialize the selected OCR backend"""
        if self.backend == 'tesseract':
            try:
                import pytesseract
                self.ocr_engine = pytesseract
                print(f"✓ Tesseract OCR initialized")
            except ImportError:
                print("⚠ Tesseract not available. Install: pip install pytesseract")
                self.ocr_engine = None
        
        elif self.backend == 'easyocr':
            try:
                import easyocr
                self.ocr_engine = easyocr.Reader([self.language])
                print(f"✓ EasyOCR initialized")
            except ImportError:
                print("⚠ EasyOCR not available. Install: pip install easyocr")
                self.ocr_engine = None
        
        elif self.backend == 'paddleocr':
            try:
                from paddleocr import PaddleOCR
                self.ocr_engine = PaddleOCR(use_angle_cls=True, lang='en')
                print(f"✓ PaddleOCR initialized")
            except ImportError:
                print("⚠ PaddleOCR not available. Install: pip install paddleocr")
                self.ocr_engine = None
    
    def extract_text_from_image(self, image_path: str) -> str:
        """Extract text from an image file"""
        if self.ocr_engine is None:
            raise RuntimeError(f"OCR backend '{self.backend}' not available")
        
        try:
            if self.backend == 'tesseract':
                return self._tesseract_extract(image_path)
            elif self.backend == 'easyocr':
                return self._easyocr_extract(image_path)
            elif self.backend == 'paddleocr':
                return self._paddleocr_extract(image_path)
        except Exception as e:
            print(f"Error extracting text from {image_path}: {str(e)}")
            return ""
    
    def _tesseract_extract(self, image_path: str) -> str:
        """Extract text using Tesseract"""
        import pytesseract
        from PIL import Image
        
        img = Image.open(image_path)
        
        # Preprocess image for better OCR
        img = self._preprocess_image(img)
        
        # Extract text
        text = pytesseract.image_to_string(img, lang=self.language)
        return text
    
    def _easyocr_extract(self, image_path: str) -> str:
        """Extract text using EasyOCR"""
        results = self.ocr_engine.readtext(image_path)
        
        # Combine all text
        text = '\n'.join([result[1] for result in results])
        return text
    
    def _paddleocr_extract(self, image_path: str) -> str:
        """Extract text using PaddleOCR"""
        result = self.ocr_engine.ocr(image_path, cls=True)
        
        # Extract text from results
        text_lines = []
        for line in result[0]:
            if line:
                text_lines.append(line[1][0])
        
        return '\n'.join(text_lines)
    
    def _preprocess_image(self, img: Image.Image) -> Image.Image:
        """Preprocess image for better OCR accuracy"""
        # Convert to grayscale
        img = img.convert('L')
        
        # Increase contrast
        import numpy as np
        img_array = np.array(img)
        
        # Simple thresholding
        threshold = 128
        img_array = np.where(img_array > threshold, 255, 0).astype(np.uint8)
        
        return Image.fromarray(img_array)


class PDFProcessor:
    """Process PDF documents with text extraction and OCR"""
    
    def __init__(self, ocr_backend='tesseract'):
        """
        Initialize PDF processor
        
        Args:
            ocr_backend: OCR backend to use for image-based PDFs
        """
        self.ocr_processor = OCRProcessor(backend=ocr_backend)
    
    def extract_text_from_pdf(self, pdf_path: str, use_ocr=False) -> str:
        """
        Extract text from PDF
        
        Args:
            pdf_path: Path to PDF file
            use_ocr: Force OCR even if text is extractable
        
        Returns:
            Extracted text
        """
        text = ""
        
        try:
            # Try text extraction first (for digital PDFs)
            if not use_ocr:
                text = self._extract_text_layer(pdf_path)
            
            # If no text found or OCR forced, use OCR
            if not text.strip() or use_ocr:
                print(f"Using OCR for: {pdf_path}")
                text = self._extract_with_ocr(pdf_path)
            
        except Exception as e:
            print(f"Error processing PDF {pdf_path}: {str(e)}")
        
        return text
    
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
            print("⚠ pdfplumber not available. Install: pip install pdfplumber")
            return ""
    
    def _extract_with_ocr(self, pdf_path: str) -> str:
        """Extract text from PDF using OCR (for scanned documents)"""
        try:
            from pdf2image import convert_from_path
            
            # Convert PDF to images
            images = convert_from_path(pdf_path, dpi=300)
            
            text = ""
            for i, img in enumerate(images):
                print(f"  Processing page {i+1}/{len(images)}...")
                
                # Save temp image
                temp_img_path = f"/tmp/page_{i}.png"
                img.save(temp_img_path)
                
                # Extract text from image
                page_text = self.ocr_processor.extract_text_from_image(temp_img_path)
                text += f"\n--- Page {i+1} ---\n{page_text}\n"
                
                # Clean up
                os.remove(temp_img_path)
            
            return text
        
        except ImportError:
            print("⚠ pdf2image not available. Install: pip install pdf2image")
            print("  Also requires poppler-utils: apt-get install poppler-utils")
            return ""


class DocumentProcessor:
    """Universal document processor for various formats"""
    
    def __init__(self, ocr_backend='tesseract'):
        self.pdf_processor = PDFProcessor(ocr_backend=ocr_backend)
        self.ocr_processor = OCRProcessor(backend=ocr_backend)
    
    def process_document(self, file_path: str, force_ocr=False) -> str:
        """
        Process any document type and extract text
        
        Args:
            file_path: Path to document
            force_ocr: Force OCR even for digital documents
        
        Returns:
            Extracted text
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        print(f"Processing: {file_path.name}")
        
        if extension == '.pdf':
            return self.pdf_processor.extract_text_from_pdf(str(file_path), use_ocr=force_ocr)
        
        elif extension in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
            return self.ocr_processor.extract_text_from_image(str(file_path))
        
        elif extension in ['.txt', '.text']:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        
        elif extension in ['.docx', '.doc']:
            return self._extract_from_word(str(file_path))
        
        else:
            print(f"⚠ Unsupported file format: {extension}")
            return ""
    
    def _extract_from_word(self, file_path: str) -> str:
        """Extract text from Word documents"""
        try:
            import docx2txt
            return docx2txt.process(file_path)
        except ImportError:
            print("⚠ docx2txt not available. Install: pip install docx2txt")
            return ""
    
    def process_batch(self, file_paths: List[str], output_dir: str = None) -> Dict[str, str]:
        """
        Process multiple documents
        
        Args:
            file_paths: List of document paths
            output_dir: Optional directory to save extracted text
        
        Returns:
            Dictionary mapping file paths to extracted text
        """
        results = {}
        
        for file_path in file_paths:
            text = self.process_document(file_path)
            results[file_path] = text
            
            # Save to output directory if specified
            if output_dir:
                output_path = Path(output_dir) / f"{Path(file_path).stem}_extracted.txt"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                
                print(f"  Saved to: {output_path}")
        
        return results


def setup_ocr_environment():
    """Check and install required OCR dependencies"""
    print("Checking OCR environment...")
    
    # Check Tesseract
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        print("✓ Tesseract OCR is installed")
    except:
        print("✗ Tesseract OCR not found")
        print("  Install: sudo apt-get install tesseract-ocr")
        print("  Python package: pip install pytesseract")
    
    # Check pdf2image dependencies
    try:
        from pdf2image import convert_from_path
        print("✓ pdf2image is installed")
    except:
        print("✗ pdf2image not found")
        print("  Install: pip install pdf2image")
        print("  System dependency: sudo apt-get install poppler-utils")
    
    # Check pdfplumber
    try:
        import pdfplumber
        print("✓ pdfplumber is installed")
    except:
        print("✗ pdfplumber not found")
        print("  Install: pip install pdfplumber")
    
    print("\nRecommended installation commands:")
    print("  System: sudo apt-get install tesseract-ocr poppler-utils")
    print("  Python: pip install pytesseract pdf2image pdfplumber Pillow numpy")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == '--check':
            setup_ocr_environment()
        else:
            # Process files
            processor = DocumentProcessor(ocr_backend='tesseract')
            files = sys.argv[1:]
            
            results = processor.process_batch(files, output_dir='/home/claude/extracted_texts')
            
            print(f"\n✓ Processed {len(results)} documents")
    else:
        print("Usage:")
        print("  python lc_ocr.py --check              # Check OCR environment")
        print("  python lc_ocr.py <file1> <file2> ... # Process documents")
