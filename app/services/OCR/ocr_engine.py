"""
OCR Engines Module
Implementations for different OCR backends (Tesseract, EasyOCR, PaddleOCR)
"""

import numpy as np
from PIL import Image
from typing import Optional
from .image_preprocessor import ImagePreprocessor
from services.ai_postProcessor import AIOCRPostProcessor


class EnhancedOCRProcessor:
    """Enhanced OCR processor with automatic post-processing and document splitting"""
    
    def __init__(self, backend='tesseract', language='eng', use_preprocessing=True, use_postprocessing=True):
        self.backend = backend
        self.language = language
        self.use_preprocessing = use_preprocessing
        self.use_postprocessing = use_postprocessing
        self.preprocessor = ImagePreprocessor()
        self.postprocessor = AIOCRPostProcessor()
        self.ocr_engine = None
        
        self._initialize_backend()
    
    def _initialize_backend(self):
        """Initialize the selected OCR backend"""
        if self.backend == 'tesseract':
            try:
                import pytesseract
                self.ocr_engine = pytesseract
                print(f"✓ Tesseract OCR initialized with post-processing")
            except ImportError:
                print("⚠ Tesseract not available")
                self.ocr_engine = None
        
        elif self.backend == 'easyocr':
            try:
                import easyocr
                self.ocr_engine = easyocr.Reader([self.language], gpu=False)
                print(f"✓ EasyOCR initialized with post-processing")
            except ImportError:
                print("⚠ EasyOCR not available")
                self.ocr_engine = None
        
        elif self.backend == 'paddleocr':
            try:
                from paddleocr import PaddleOCR
                self.ocr_engine = PaddleOCR(use_angle_cls=True, lang='en')
                print(f"✓ PaddleOCR initialized with post-processing")
            except ImportError:
                print("⚠ PaddleOCR not available")
                self.ocr_engine = None
    
    def extract_text_from_image(self, image_path: str, aggressive_preprocessing: bool = False, 
                               aggressive_postprocessing: bool = False) -> str:
        """Extract text from image with automatic post-processing"""
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
            
            # Apply post-processing
            if self.use_postprocessing and text:
                text = self.postprocessor.clean_text(text)
            
            return text
            
        except Exception as e:
            print(f"Error extracting text: {str(e)}")
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
                    
            except Exception:
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