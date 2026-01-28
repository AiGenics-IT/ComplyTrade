"""
Enhanced OCR Module with Integrated Post-Processing
Automatically fixes squashed text and spacing issues in OCR output
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import json
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import tempfile
import re


class OCRPostProcessor:
    """Integrated post-processor for OCR text cleanup"""
    
    def __init__(self):
        # Banking-specific squashed patterns - order matters (longer patterns first)
        self.squashed_patterns = {
            # Multi-word patterns (must come first)
            'LETTEROFAUTHORITYFORSIGNING': 'LETTER OF AUTHORITY FOR SIGNING',
            'BILLOFLADINGISACCEPTABLETO': 'BILL OF LADING IS ACCEPTABLE TO',
            'FURTHERDOCUMENTARYEVIDENCE': 'FURTHER DOCUMENTARY EVIDENCE',
            'DOCUMENTARYEVIDENCEWILL': 'DOCUMENTARY EVIDENCE WILL',
            'ACCEPTABLETOCOMPLYL/C': 'ACCEPTABLE TO COMPLY L/C',
            'ACCEPTABLETOCOMPLY': 'ACCEPTABLE TO COMPLY',
            'REQUIREDAGAINST': 'REQUIRED AGAINST',
            'UNDERL/CCLAUSE': 'UNDER L/C CLAUSE',
            'COMPLYL/CCLAUSE': 'COMPLY L/C CLAUSE',
            
            # Clause/Field patterns
            'CLAUSENO': 'CLAUSE NO',
            'CLAUENO': 'CLAUSE NO',
            'CLAUSENUMBER': 'CLAUSE NUMBER',
            'FIELDNO': 'FIELD NO',
            'ITEMNO': 'ITEM NO',
            
            # Action patterns
            'TOREADAS': 'TO READ AS',
            'TOREAD': 'TO READ',
            'NOWTOREAD': 'NOW TO READ',
            'INSTEADOF': 'INSTEAD OF',
            'REPLACEBY': 'REPLACE BY',
            'REPLACEWITH': 'REPLACE WITH',
            'DELETEREPLACE': 'DELETE REPLACE',
            
            # Banking terms
            'BILLOFLADING': 'BILL OF LADING',
            'BILLOF': 'BILL OF',
            'LETTEROFCREDIT': 'LETTER OF CREDIT',
            'LETTEROFAUTHORITY': 'LETTER OF AUTHORITY',
            'LETTEROF': 'LETTER OF',
            'L/CCLAUSE': 'L/C CLAUSE',
            'LCCLAUSE': 'LC CLAUSE',
            
            # Common combinations
            'ACCEPTABLETO': 'ACCEPTABLE TO',
            'ISACCEPTABLE': 'IS ACCEPTABLE',
            'TOCOMPLY': 'TO COMPLY',
            'FORSIGNING': 'FOR SIGNING',
            'EVIDENCEWILL': 'EVIDENCE WILL',
            'WILLBE': 'WILL BE',
            'MUSTBE': 'MUST BE',
            'SHALLBE': 'SHALL BE',
            'ANDOR': 'AND/OR',
            'AND/ORTHE': 'AND/OR THE',
            'NOTREQUIRED': 'NOT REQUIRED',
            'ISREQUIRED': 'IS REQUIRED',
        }
        
        self.banking_keywords = [
            'CLAUSE', 'FIELD', 'DOCUMENT', 'LETTER', 'AUTHORITY',
            'INSTEAD', 'REPLACE', 'DELETE', 'READ', 'ACCEPTABLE',
            'REQUIRED', 'AGAINST', 'UNDER', 'BILL', 'LADING',
            'CERTIFICATE', 'INVOICE', 'COMPLY', 'EVIDENCE',
            'SIGNING', 'DOCUMENTARY', 'FURTHER', 'SHIPMENT',
        ]
    
    def clean_text(self, text: str, aggressive: bool = False) -> str:
        """
        Main cleaning pipeline with comprehensive text fixing
        
        Args:
            text: Raw OCR text
            aggressive: More aggressive spacing
        
        Returns:
            Cleaned text
        """
        if not text:
            return text
        
        # Step 1: Fix squashed patterns (longest first to avoid partial replacements)
        for squashed, proper in self.squashed_patterns.items():
            text = re.sub(re.escape(squashed), proper, text, flags=re.IGNORECASE)
        
        # Step 2: Fix common OCR character mistakes
        text = self._fix_ocr_errors(text)
        
        # Step 3: Separate stuck keywords
        text = self._separate_keywords(text)
        
        # Step 4: Separate numbers and letters
        text = re.sub(r'([A-Z]{2,})(\d+)', r'\1 \2', text)  # CLAUSE27 -> CLAUSE 27
        text = re.sub(r'(\d+)([A-Z]{2,})', r'\1 \2', text)  # 27CLAUSE -> 27 CLAUSE
        text = re.sub(r'(NO\.?)(\d+)', r'\1 \2', text, flags=re.IGNORECASE)  # NO.27 -> NO. 27
        
        # Step 5: Fix punctuation spacing
        text = re.sub(r'([.,;:])([A-Z])', r'\1 \2', text)  # Add space after punctuation
        text = re.sub(r"([A-Z])([\'\"])", r"\1 \2", text)  # Add space before quotes
        
        # Step 6: Fix FIELD46A pattern
        text = re.sub(r'FIELD(\d+[A-Z])', r'FIELD \1', text, flags=re.IGNORECASE)
        
        # Step 7: Fix L/C patterns
        text = re.sub(r'L/C([A-Z])', r'L/C \1', text)
        text = re.sub(r'([A-Z])L/C', r'\1 L/C', text)
        
        # Step 8: Normalize whitespace
        text = re.sub(r' +', ' ', text)  # Multiple spaces -> single space
        text = re.sub(r'\s+([.,;:])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r"\s+([\'\"])", r'\1', text)  # Remove space before quotes
        
        # Step 9: Fix over-splitting issues
        text = self._fix_oversplitting(text)
        
        return text.strip()
    
    def _fix_ocr_errors(self, text: str) -> str:
        """Fix common OCR character recognition errors"""
        fixes = {
            'l/C': 'L/C',
            'l/c': 'L/C',
            '0F': 'OF',
            'TH E': 'THE',
            'AN D': 'AND',
            'W ITH': 'WITH',
            'REQU IRED': 'REQUIRED',
            'DOCU MENT': 'DOCUMENT',
            'EVID ENCE': 'EVIDENCE',
        }
        
        for wrong, correct in fixes.items():
            text = text.replace(wrong, correct)
        
        return text
    
    def _separate_keywords(self, text: str) -> str:
        """
        Separate stuck banking keywords from surrounding text
        """
        for keyword in self.banking_keywords:
            # Pattern 1: Keyword followed by uppercase letters (CLAUSEFIELD -> CLAUSE FIELD)
            text = re.sub(f'({keyword})([A-Z]{{2,}})', r'\1 \2', text, flags=re.IGNORECASE)
            
            # Pattern 2: Uppercase letters followed by keyword (FIELDCLAUSE -> FIELD CLAUSE)
            text = re.sub(f'([A-Z]{{2,}})({keyword})', r'\1 \2', text, flags=re.IGNORECASE)
            
            # Pattern 3: Keyword followed by lowercase (for camelCase)
            text = re.sub(f'({keyword})([a-z]+)', r'\1 \2', text, flags=re.IGNORECASE)
            
            # Pattern 4: Lowercase followed by keyword (for camelCase)
            text = re.sub(f'([a-z]+)({keyword})', r'\1 \2', text, flags=re.IGNORECASE)
        
        return text
    
    def _fix_oversplitting(self, text: str) -> str:
        """
        Fix cases where words were incorrectly split
        """
        # Common oversplitting issues
        fixes = {
            'DOCU MENTARY': 'DOCUMENTARY',
            'DOCU MENT': 'DOCUMENT',
            'EVID ENCE': 'EVIDENCE',
            'AUTH ORITY': 'AUTHORITY',
            'CERTIF ICATE': 'CERTIFICATE',
            'SHIPME NT': 'SHIPMENT',
            'PAYME NT': 'PAYMENT',
            'REQU IRED': 'REQUIRED',
        }
        
        for wrong, correct in fixes.items():
            text = text.replace(wrong, correct)
        
        return text


class ImagePreprocessor:
    """Advanced image preprocessing for OCR accuracy improvement"""
    
    def __init__(self):
        self.default_dpi = 300
    
    def preprocess_for_ocr(self, image: Image.Image, aggressive: bool = False) -> Image.Image:
        """Comprehensive preprocessing pipeline for OCR"""
        img_array = np.array(image)
        
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        img_array = self._denoise(img_array, aggressive)
        img_array = self._enhance_contrast(img_array)
        img_array = self._sharpen(img_array, aggressive)
        img_array = self._binarize(img_array, aggressive)
        img_array = self._deskew(img_array)
        img_array = self._remove_borders(img_array)
        
        return Image.fromarray(img_array)
    
    def _denoise(self, img: np.ndarray, aggressive: bool = False) -> np.ndarray:
        if aggressive:
            img = cv2.GaussianBlur(img, (5, 5), 0)
        else:
            img = cv2.GaussianBlur(img, (3, 3), 0)
        
        img = cv2.fastNlMeansDenoising(img, None, h=10, templateWindowSize=7, searchWindowSize=21)
        img = cv2.bilateralFilter(img, 9, 75, 75)
        
        return img
    
    def _enhance_contrast(self, img: np.ndarray) -> np.ndarray:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(img)
    
    def _sharpen(self, img: np.ndarray, aggressive: bool = False) -> np.ndarray:
        if aggressive:
            kernel = np.array([[-1, -1, -1], [-1,  9, -1], [-1, -1, -1]])
        else:
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        
        return cv2.filter2D(img, -1, kernel)
    
    def _binarize(self, img: np.ndarray, aggressive: bool = False) -> np.ndarray:
        if aggressive:
            _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            img = cv2.adaptiveThreshold(
                img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
        return img
    
    def _deskew(self, img: np.ndarray) -> np.ndarray:
        edges = cv2.Canny(img, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
        
        if lines is not None and len(lines) > 0:
            angles = []
            for rho, theta in lines[:, 0]:
                angle = np.degrees(theta) - 90
                if -45 < angle < 45:
                    angles.append(angle)
            
            if angles:
                median_angle = np.median(angles)
                if abs(median_angle) > 0.5:
                    (h, w) = img.shape
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                    img = cv2.warpAffine(img, M, (w, h), 
                                        flags=cv2.INTER_CUBIC, 
                                        borderMode=cv2.BORDER_REPLICATE)
        return img
    
    def _remove_borders(self, img: np.ndarray) -> np.ndarray:
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            margin = 10
            y_start = max(0, y - margin)
            y_end = min(img.shape[0], y + h + margin)
            x_start = max(0, x - margin)
            x_end = min(img.shape[1], x + w + margin)
            
            img = img[y_start:y_end, x_start:x_end]
        
        return img
    
    def upscale_image(self, image: Image.Image, target_dpi: int = 300) -> Image.Image:
        width, height = image.size
        current_dpi = image.info.get('dpi', (72, 72))[0]
        scale_factor = target_dpi / current_dpi
        
        if scale_factor > 1:
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        return image


class EnhancedOCRProcessor:
    """Enhanced OCR processor with automatic post-processing"""
    
    def __init__(self, backend='tesseract', language='eng', use_preprocessing=True, use_postprocessing=True):
        """
        Initialize Enhanced OCR processor with post-processing
        
        Args:
            backend: 'tesseract', 'easyocr', or 'paddleocr'
            language: Language code
            use_preprocessing: Enable image preprocessing
            use_postprocessing: Enable text post-processing
        """
        self.backend = backend
        self.language = language
        self.use_preprocessing = use_preprocessing
        self.use_postprocessing = use_postprocessing
        self.preprocessor = ImagePreprocessor()
        self.postprocessor = OCRPostProcessor()
        self.ocr_engine = None
        
        self._initialize_backend()
    
    def _initialize_backend(self):
        """Initialize the selected OCR backend"""
        if self.backend == 'tesseract':
            try:
                import pytesseract
                self.ocr_engine = pytesseract
                postprocessing_status = "with" if self.use_postprocessing else "without"
                print(f"✓ Tesseract OCR initialized {postprocessing_status} post-processing")
            except ImportError:
                print("⚠ Tesseract not available. Install: pip install pytesseract")
                self.ocr_engine = None
        
        elif self.backend == 'easyocr':
            try:
                import easyocr
                self.ocr_engine = easyocr.Reader([self.language], gpu=False)
                postprocessing_status = "with" if self.use_postprocessing else "without"
                print(f"✓ EasyOCR initialized {postprocessing_status} post-processing")
            except ImportError:
                print("⚠ EasyOCR not available. Install: pip install easyocr")
                self.ocr_engine = None
        
        elif self.backend == 'paddleocr':
            try:
                from paddleocr import PaddleOCR
                self.ocr_engine = PaddleOCR(use_angle_cls=True, lang='en')
                postprocessing_status = "with" if self.use_postprocessing else "without"
                print(f"✓ PaddleOCR initialized {postprocessing_status} post-processing")
            except ImportError:
                print("⚠ PaddleOCR not available. Install: pip install paddleocr")
                self.ocr_engine = None
    
    def extract_text_from_image(self, image_path: str, aggressive_preprocessing: bool = False, 
                               aggressive_postprocessing: bool = False) -> str:
        """
        Extract text from image with automatic post-processing
        
        Args:
            image_path: Path to image
            aggressive_preprocessing: Aggressive image preprocessing
            aggressive_postprocessing: Aggressive text cleanup
        
        Returns:
            Cleaned and properly spaced text
        """
        if self.ocr_engine is None:
            raise RuntimeError(f"OCR backend '{self.backend}' not available")
        
        try:
            img = Image.open(image_path)
            
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            img = self.preprocessor.upscale_image(img, target_dpi=300)
            
            if self.use_preprocessing:
                img = self.preprocessor.preprocess_for_ocr(img, aggressive=aggressive_preprocessing)
            
            # Extract raw text
            if self.backend == 'tesseract':
                text = self._tesseract_extract(img)
            elif self.backend == 'easyocr':
                text = self._easyocr_extract(img)
            elif self.backend == 'paddleocr':
                text = self._paddleocr_extract(img)
            else:
                text = ""
            
            # Apply post-processing to clean up text
            if self.use_postprocessing and text:
                text = self.postprocessor.clean_text(text, aggressive=aggressive_postprocessing)
            
            return text
            
        except Exception as e:
            print(f"Error extracting text from {image_path}: {str(e)}")
            return ""
    
    def _tesseract_extract(self, img: Image.Image) -> str:
        """Extract text using Tesseract"""
        import pytesseract
        
        psm_modes = [6, 3, 4, 11]
        best_text = ""
        best_confidence = 0
        
        for psm in psm_modes:
            config = f'--oem 3 --psm {psm}'
            
            try:
                data = pytesseract.image_to_data(img, config=config, output_type=pytesseract.Output.DICT)
                text = pytesseract.image_to_string(img, config=config)
                
                confidences = [int(conf) for conf in data['conf'] if conf != '-1']
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                
                if avg_confidence > best_confidence:
                    best_confidence = avg_confidence
                    best_text = text
                    
            except Exception as e:
                continue
        
        return best_text
    
    def _easyocr_extract(self, img: Image.Image) -> str:
        """Extract text using EasyOCR"""
        img_array = np.array(img)
        results = self.ocr_engine.readtext(img_array, detail=0)
        return '\n'.join(results)
    
    def _paddleocr_extract(self, img: Image.Image) -> str:
        """Extract text using PaddleOCR"""
        img_array = np.array(img)
        result = self.ocr_engine.ocr(img_array, cls=True)
        
        text_lines = []
        if result and result[0]:
            for line in result[0]:
                if line:
                    text_lines.append(line[1][0])
        
        return '\n'.join(text_lines)


class PDFProcessorEnhanced:
    """Enhanced PDF processor with automatic text post-processing"""
    
    def __init__(self, ocr_backend='tesseract', use_preprocessing=True, use_postprocessing=True):
        """
        Initialize with post-processing enabled
        
        Args:
            ocr_backend: OCR backend to use
            use_preprocessing: Enable image preprocessing
            use_postprocessing: Enable automatic text cleanup
        """
        self.ocr_processor = EnhancedOCRProcessor(
            backend=ocr_backend, 
            use_preprocessing=use_preprocessing,
            use_postprocessing=use_postprocessing
        )
    
    def extract_text_from_pdf(self, pdf_path: str, use_ocr=False, aggressive_ocr=False,
                             aggressive_postprocessing=False) -> str:
        """
        Extract text from PDF with post-processing
        
        Args:
            pdf_path: Path to PDF file
            use_ocr: Force OCR even if text layer exists
            aggressive_ocr: Use aggressive image preprocessing
            aggressive_postprocessing: Use aggressive text cleanup
        
        Returns:
            Extracted and cleaned text
        """
        text = ""
        
        try:
            if not use_ocr:
                text = self._extract_text_layer(pdf_path)
            
            if not text.strip() or use_ocr:
                print(f"Using enhanced OCR with post-processing for: {pdf_path}")
                text = self._extract_with_enhanced_ocr(pdf_path, aggressive_ocr, aggressive_postprocessing)
            
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
    
    def _extract_with_enhanced_ocr(self, pdf_path: str, aggressive_preprocess: bool = False,
                                   aggressive_postprocess: bool = False) -> str:
        """
        Extract with OCR and post-processing
        """
        try:
            from pdf2image import convert_from_path
            
            images = convert_from_path(pdf_path, dpi=300, fmt='png')
            
            text = ""
            for i, img in enumerate(images):
                print(f"  Processing page {i+1}/{len(images)} with enhanced OCR...")
                
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    temp_img_path = tmp.name
                
                try:
                    img.save(temp_img_path, 'PNG', quality=100)
                    
                    page_text = self.ocr_processor.extract_text_from_image(
                        temp_img_path, 
                        aggressive_preprocessing=aggressive_preprocess,
                        aggressive_postprocessing=aggressive_postprocess
                    )
                    
                    # Sanitize for Windows
                    page_text = "".join(c for c in page_text if c.isprintable() or c in "\n\r\t")
                    
                    text += f"\n--- Page {i+1} ---\n{page_text}\n"
                    
                finally:
                    if os.path.exists(temp_img_path):
                        os.remove(temp_img_path)
            
            return text
        
        except ImportError:
            print("⚠ pdf2image not available. Install: pip install pdf2image")
            return ""
        except Exception as e:
            print(f"⚠ Enhanced OCR Failed: {str(e)}")
            return ""


class DocumentProcessor:
    """Enhanced universal document processor with post-processing"""
    
    def __init__(self, ocr_backend='tesseract', use_preprocessing=True, use_postprocessing=True):
        """
        Initialize with post-processing enabled
        
        Args:
            ocr_backend: OCR backend to use ('tesseract', 'easyocr', 'paddleocr')
            use_preprocessing: Enable image preprocessing
            use_postprocessing: Enable automatic text cleanup
        """
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
    
    def process_document(self, file_path: str, force_ocr=False, aggressive_ocr=False,
                        aggressive_postprocessing=False) -> str:
        """
        Process document with automatic text cleanup
        
        Args:
            file_path: Path to document file
            force_ocr: Force OCR even for digital documents
            aggressive_ocr: Use aggressive image preprocessing
            aggressive_postprocessing: Use aggressive text cleanup
        
        Returns:
            Extracted and cleaned text
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
            print(f"⚠ Unsupported file format: {extension}")
            return ""


if __name__ == "__main__":
    print("Enhanced OCR Processor with Automatic Post-Processing")
    print("Fixes squashed text and spacing issues automatically")