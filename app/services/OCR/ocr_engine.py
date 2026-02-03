"""
Windows-Compatible OCR Engine
No Unix signals - pure exception handling
"""

import numpy as np
from PIL import Image
from typing import Optional
from .olm_ocr_processor import OlmOCRProcessor 
                
class EnhancedOCRProcessor:
    """OCR processor optimized for Windows"""
    
    def __init__(self, backend='tesseract', language='eng', use_preprocessing=True, use_postprocessing=True):
        self.backend = backend
        self.language = language
        self.use_preprocessing = use_preprocessing
        self.use_postprocessing = use_postprocessing
        self.ocr_engine = None
        
        # Import preprocessor and postprocessor
        try:
            from services.OCR.image_preprocessor import ImagePreprocessor
            self.preprocessor = ImagePreprocessor()
        except ImportError:
            from .image_preprocessor import ImagePreprocessor
            self.preprocessor = ImagePreprocessor()
        
        try:
            from services.ai_postProcessor import AIOCRPostProcessor
            self.postprocessor = AIOCRPostProcessor()
        except ImportError:
            self.postprocessor = None
        
        self._initialize_backend()
    
    def _initialize_backend(self):
        """Initialize OCR backend (Tesseract, EasyOCR, or PaddleOCR)"""
        # Mapping Tesseract codes to EasyOCR codes
        TESSERACT_TO_EASYOCR_LANG = {
            'eng': 'en',
            'fra': 'fr',
            'deu': 'de',
            'spa': 'es',
            'ita': 'it',
            # add more as needed
        }

        if self.backend == 'olmocr':
            try:
                # We import our specialized class from the previous step
                self.ocr_engine = OlmOCRProcessor()
                print("✓ olmOCR-2 (Qwen2-VL) backend initialized")
            except Exception as e:
                print(f"⚠ olmOCR-2 failed to load: {e}")
                self.ocr_engine = None
        elif self.backend.lower() == 'tesseract':
            try:
                import pytesseract
                self.ocr_engine = pytesseract
                print(f"✓ Tesseract OCR initialized with language '{self.language}'")
            except ImportError:
                print("⚠ Tesseract not available")
                self.ocr_engine = None

        elif self.backend.lower() == 'easyocr':
            try:
                import easyocr
                # Convert Tesseract code to EasyOCR code if needed
                eo_lang = TESSERACT_TO_EASYOCR_LANG.get(self.language, self.language)
                self.ocr_engine = easyocr.Reader([eo_lang], gpu=False)
                print(f"✓ EasyOCR initialized with language '{eo_lang}'")
            except ImportError:
                print("⚠ EasyOCR not available")
                self.ocr_engine = None

        elif self.backend.lower() == 'paddleocr':
            try:
                from paddleocr import PaddleOCR
                # PaddleOCR uses 'en', 'ch', 'japan', etc.
                # Map Tesseract codes to PaddleOCR if needed
                TESSERACT_TO_PADDLE_LANG = {
                    'eng': 'en',
                    'fra': 'en',  # PaddleOCR may need 'en' for Latin languages
                    'deu': 'en',
                    'spa': 'en',
                    'ita': 'en',
                }
                paddle_lang = TESSERACT_TO_PADDLE_LANG.get(self.language, 'en')
                self.ocr_engine = PaddleOCR(use_angle_cls=True, lang=paddle_lang)
                print(f"✓ PaddleOCR initialized with language '{paddle_lang}'")
            except ImportError:
                print("⚠ PaddleOCR not available")
                self.ocr_engine = None

        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    
    def extract_text_from_image(self, image_path: str, aggressive_preprocessing: bool = False, 
                               aggressive_postprocessing: bool = False) -> str:
        """
        Extract text from image with robust error handling
        """
        if self.ocr_engine is None:
            raise RuntimeError(f"OCR backend '{self.backend}' not available")
        
        try:
            # Load image
            img = Image.open(image_path)
            
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Upscale
            print(f"  → Upscaling image...")
            img = self.preprocessor.upscale_image(img, target_dpi=300)
            
            # Preprocessing
            if self.use_preprocessing:
                try:
                    print(f"  → Preprocessing image...")
                    img = self.preprocessor.preprocess_for_ocr(img, aggressive=aggressive_preprocessing)
                    print(f"  ✓ Preprocessing complete")
                except Exception as e:
                    print(f"  ⚠ Preprocessing failed: {str(e)}, using quick mode")
                    try:
                        img = self.preprocessor.quick_preprocess(img)
                    except Exception:
                        print(f"  ⚠ Quick mode failed, using original")
            
            # Extract text
            try:
                print(f"  → Extracting text with {self.backend}...")
                
                if self.backend == 'tesseract':
                    text = self._tesseract_extract(img)
                elif self.backend == 'easyocr':
                    text = self._easyocr_extract(img)
                elif self.backend == 'paddleocr':
                    text = self._paddleocr_extract(img)
                else:
                    text = ""
                
                print(f"  ✓ Extracted {len(text)} characters")
            
            except Exception as e:
                print(f"  ⚠ OCR failed: {str(e)}, trying simple mode")
                text = self._simple_extract(img)
            
            # Post-process
            if self.use_postprocessing and text and self.postprocessor:
                try:
                    print(f"  → Post-processing...")
                    text = self.postprocessor.clean_text(text)
                    print(f"  ✓ Post-processing complete")
                except Exception as e:
                    print(f"  ⚠ Post-processing failed: {str(e)}")
            
            return text
            
        except Exception as e:
            print(f"Error extracting text: {str(e)}")
            return ""
    
    def _simple_extract(self, img: Image.Image) -> str:
        """Simple fast extraction"""
        try:
            if self.backend == 'tesseract':
                import pytesseract
                return pytesseract.image_to_string(img, config='--oem 3 --psm 6')
            elif self.backend == 'easyocr':
                img_array = np.array(img)
                results = self.ocr_engine.readtext(img_array, detail=0)
                return '\n'.join(results)
            elif self.backend == 'paddleocr':
                return self._paddleocr_extract(img)
        except Exception:
            pass
        return ""
    
    def _tesseract_extract(self, img: Image.Image) -> str:
        """Extract using Tesseract with multiple PSM modes"""
        import pytesseract
        
        # Try different page segmentation modes
        psm_modes = [6, 3, 4]  # Reduced for speed
        best_text = ""
        best_confidence = 0
        
        for psm in psm_modes:
            try:
                config = f'--oem 3 --psm {psm}'
                
                # Get data for confidence
                data = pytesseract.image_to_data(img, config=config, output_type=pytesseract.Output.DICT)
                text = pytesseract.image_to_string(img, config=config)
                
                # Calculate confidence
                confidences = [int(conf) for conf in data['conf'] if conf != '-1']
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                
                if avg_confidence > best_confidence:
                    best_confidence = avg_confidence
                    best_text = text
                
                # If we got good confidence, stop trying
                if avg_confidence > 70:
                    break
                    
            except Exception as e:
                print(f"    ⚠ PSM {psm} failed: {str(e)}")
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
        result = self.ocr_engine.ocr(img_array)
        
        text_lines = []
        if result and result[0]:
            for line in result[0]:
                if line:
                    text_lines.append(line[1][0])
        
        return '\n'.join(text_lines)