"""
Windows-Compatible OCR Engine
No Unix signals - pure exception handling
"""

import numpy as np
from PIL import Image
from pathlib import Path
import requests
from typing import Optional


class EnhancedOCRProcessor:
    """OCR processor optimized for Windows, supports local backends and OCR API"""

    # Set your OCR API URL here
    OCR_API_URL: str = "http://10.20.10.2:8082/ocr/"

    def __init__(self, backend='tesseract', language='eng', use_preprocessing=True,
                 use_postprocessing=True, use_api=True):
        self.backend = backend
        self.language = language
        self.use_preprocessing = use_preprocessing
        self.use_postprocessing = use_postprocessing
        self.use_api = use_api
        self.ocr_engine = None

        print(f"[OCR Engine] Initializing with use_api={self.use_api}")

        # Import preprocessor
        try:
            from services.OCR.image_preprocessor import ImagePreprocessor
            self.preprocessor = ImagePreprocessor()
        except ImportError:
            try:
                from .image_preprocessor import ImagePreprocessor
                self.preprocessor = ImagePreprocessor()
            except ImportError:
                self.preprocessor = None
                print("⚠ ImagePreprocessor not available")

        # Import postprocessor
        try:
            from services.ai_postProcessor import AIOCRPostProcessor
            self.postprocessor = AIOCRPostProcessor()
        except ImportError:
            self.postprocessor = None
            print("⚠ AIOCRPostProcessor not available")

        # Only initialize backend if NOT using API
        if not self.use_api:
            print("[OCR Engine] Initializing local backend...")
            self._initialize_backend()
        else:
            print("[OCR Engine] Skipping local backend initialization - using API")

    def _initialize_backend(self):
        """Initialize OCR backend (Tesseract, EasyOCR, or PaddleOCR)"""
        TESSERACT_TO_EASYOCR_LANG = {
            'eng': 'en', 'fra': 'fr', 'deu': 'de', 'spa': 'es', 'ita': 'it'
        }

        if self.backend.lower() == 'tesseract':
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
                eo_lang = TESSERACT_TO_EASYOCR_LANG.get(self.language, self.language)
                self.ocr_engine = easyocr.Reader([eo_lang], gpu=False)
                print(f"✓ EasyOCR initialized with language '{eo_lang}'")
            except ImportError:
                print("⚠ EasyOCR not available")
                self.ocr_engine = None

        elif self.backend.lower() == 'paddleocr':
            try:
                from paddleocr import PaddleOCR
                TESSERACT_TO_PADDLE_LANG = {
                    'eng': 'en', 'fra': 'en', 'deu': 'en', 'spa': 'en', 'ita': 'en'
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
        Extract text from image with robust error handling.
        If `use_api=True`, the OCR API is used instead of local backends.
        """
        print(f"[OCR Engine] extract_text_from_image called with use_api={self.use_api}")
        
        # If API mode is enabled, use API regardless
        if self.use_api:
            print(f"[OCR Engine] Routing to API OCR")
            return self._ocr_via_api(image_path)

        # Local processing mode
        print(f"[OCR Engine] Using local backend: {self.backend}")
        
        if self.ocr_engine is None:
            raise RuntimeError(f"OCR backend '{self.backend}' not available")

        try:
            img = Image.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')

            if self.preprocessor:
                print(f"  → Upscaling image...")
                img = self.preprocessor.upscale_image(img, target_dpi=300)

            if self.use_preprocessing and self.preprocessor:
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
        import pytesseract
        psm_modes = [6, 3, 4]
        best_text = ""
        best_confidence = 0

        for psm in psm_modes:
            try:
                config = f'--oem 3 --psm {psm}'
                data = pytesseract.image_to_data(img, config=config, output_type=pytesseract.Output.DICT)
                text = pytesseract.image_to_string(img, config=config)
                confidences = [int(conf) for conf in data['conf'] if conf != '-1']
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                if avg_confidence > best_confidence:
                    best_confidence = avg_confidence
                    best_text = text
                if avg_confidence > 70:
                    break
            except Exception as e:
                print(f"    ⚠ PSM {psm} failed: {str(e)}")
                continue

        return best_text if best_text else self._simple_extract(img)

    def _easyocr_extract(self, img: Image.Image) -> str:
        img_array = np.array(img)
        results = self.ocr_engine.readtext(img_array, detail=0)
        return '\n'.join(results)

    def _paddleocr_extract(self, img: Image.Image) -> str:
        img_array = np.array(img)
        result = self.ocr_engine.ocr(img_array)
        text_lines = []
        if result and result[0]:
            for line in result[0]:
                if line:
                    text_lines.append(line[1][0])
        return '\n'.join(text_lines)

    def _ocr_via_api(self, file_path: str) -> str:
        """
        Send file to OCR API and return extracted text
        """
        try:
            file_name = Path(file_path).name
            print(f"  → Sending to OCR API: {file_name}")
            
            with open(file_path, "rb") as f:
                response = requests.post(
                    self.OCR_API_URL,
                    files={"file": (file_name, f)},
                    timeout=180
                )
                response.raise_for_status()
                
                result = response.json()
                text = result.get("text", "")
                
                print(f"  ✓ OCR API returned {len(text)} characters for {file_name}")
                return text
                
        except requests.exceptions.Timeout:
            print(f"  ✗ OCR API timeout for {Path(file_path).name}")
            return ""
        except requests.exceptions.RequestException as e:
            print(f"  ✗ OCR API request failed: {e}")
            return ""
        except Exception as e:
            print(f"  ✗ OCR API error: {e}")
            return ""