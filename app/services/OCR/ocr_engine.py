"""
OCR Engine with Timeout Protection
Prevents hanging on difficult images
"""

import numpy as np
from PIL import Image
from typing import Optional
import signal
from contextlib import contextmanager


class TimeoutException(Exception):
    """Timeout exception"""
    pass


@contextmanager
def timeout(seconds):
    """Timeout context manager"""
    def timeout_handler(signum, frame):
        raise TimeoutException(f"Operation timed out after {seconds} seconds")
    
    original_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, original_handler)


class EnhancedOCRProcessor:
    """OCR processor with timeout protection and fallback modes"""
    
    def __init__(self, backend='tesseract', language='eng', use_preprocessing=True, use_postprocessing=True):
        self.backend = backend
        self.language = language
        self.use_preprocessing = use_preprocessing
        self.use_postprocessing = use_postprocessing
        self.ocr_engine = None
        self.preprocessing_timeout = 30  # 30 seconds max for preprocessing
        self.ocr_timeout = 60  # 60 seconds max for OCR
        
        # Import preprocessor and postprocessor
        from services.OCR.image_preprocessor import ImagePreprocessor
        from services.ai_postProcessor import AIOCRPostProcessor
        
        self.preprocessor = ImagePreprocessor()
        self.postprocessor = AIOCRPostProcessor()
        
        self._initialize_backend()
    
    def _initialize_backend(self):
        """Initialize OCR backend"""
        if self.backend == 'tesseract':
            try:
                import pytesseract
                self.ocr_engine = pytesseract
                print(f"✓ Tesseract OCR initialized")
            except ImportError:
                print("⚠ Tesseract not available")
                self.ocr_engine = None
        
        elif self.backend == 'easyocr':
            try:
                import easyocr
                self.ocr_engine = easyocr.Reader([self.language], gpu=False)
                print(f"✓ EasyOCR initialized")
            except ImportError:
                print("⚠ EasyOCR not available")
                self.ocr_engine = None
        
        elif self.backend == 'paddleocr':
            try:
                from paddleocr import PaddleOCR
                self.ocr_engine = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
                print(f"✓ PaddleOCR initialized")
            except ImportError:
                print("⚠ PaddleOCR not available")
                self.ocr_engine = None
    
    def extract_text_from_image(self, image_path: str, aggressive_preprocessing: bool = False, 
                               aggressive_postprocessing: bool = False) -> str:
        """
        Extract text with timeout protection and fallback
        """
        if self.ocr_engine is None:
            raise RuntimeError(f"OCR backend '{self.backend}' not available")
        
        try:
            img = Image.open(image_path)
            
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Upscale
            img = self.preprocessor.upscale_image(img, target_dpi=300)
            
            # Try preprocessing with timeout
            if self.use_preprocessing:
                try:
                    print(f"  → Preprocessing image...")
                    img = self._preprocess_with_timeout(img, aggressive_preprocessing)
                    print(f"  ✓ Preprocessing complete")
                except TimeoutException:
                    print(f"  ⚠ Preprocessing timeout, using quick mode")
                    img = self.preprocessor.quick_preprocess(img)
                except Exception as e:
                    print(f"  ⚠ Preprocessing failed: {str(e)}, using original")
            
            # Extract text with timeout
            try:
                print(f"  → Extracting text...")
                text = self._extract_with_timeout(img)
                print(f"  ✓ Text extracted ({len(text)} chars)")
            except TimeoutException:
                print(f"  ⚠ OCR timeout, trying simple mode")
                text = self._simple_extract(img)
            
            # Post-process
            if self.use_postprocessing and text:
                print(f"  → Post-processing...")
                text = self.postprocessor.clean_text(text)
                print(f"  ✓ Post-processing complete")
            
            return text
            
        except Exception as e:
            print(f"Error extracting text: {str(e)}")
            return ""
    
    def _preprocess_with_timeout(self, img: Image.Image, aggressive: bool) -> Image.Image:
        """Preprocess with timeout protection"""
        with timeout(self.preprocessing_timeout):
            return self.preprocessor.preprocess_for_ocr(img, aggressive=aggressive)
    
    def _extract_with_timeout(self, img: Image.Image) -> str:
        """Extract text with timeout protection"""
        with timeout(self.ocr_timeout):
            if self.backend == 'tesseract':
                return self._tesseract_extract(img)
            elif self.backend == 'easyocr':
                return self._easyocr_extract(img)
            elif self.backend == 'paddleocr':
                return self._paddleocr_extract(img)
        return ""
    
    def _simple_extract(self, img: Image.Image) -> str:
        """Simple fast extraction without multiple PSM modes"""
        if self.backend == 'tesseract':
            import pytesseract
            return pytesseract.image_to_string(img, config='--oem 3 --psm 6')
        elif self.backend == 'easyocr':
            img_array = np.array(img)
            results = self.ocr_engine.readtext(img_array, detail=0)
            return '\n'.join(results)
        elif self.backend == 'paddleocr':
            return self._paddleocr_extract(img)
        return ""
    
    def _tesseract_extract(self, img: Image.Image) -> str:
        """Extract using Tesseract with multiple PSM modes"""
        import pytesseract
        
        psm_modes = [6, 3, 4]  # Reduced from 4 modes to 3 for speed
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
        
        return best_text if best_text else self._simple_extract(img)
    
    def _easyocr_extract(self, img: Image.Image) -> str:
        """Extract using EasyOCR"""
        img_array = np.array(img)
        results = self.ocr_engine.readtext(img_array, detail=0)
        return '\n'.join(results)
    
    def _paddleocr_extract(self, img: Image.Image) -> str:
        """Extract using PaddleOCR"""
        img_array = np.array(img)
        result = self.ocr_engine.ocr(img_array, cls=True)
        
        text_lines = []
        if result and result[0]:
            for line in result[0]:
                if line:
                    text_lines.append(line[1][0])
        
        return '\n'.join(text_lines)