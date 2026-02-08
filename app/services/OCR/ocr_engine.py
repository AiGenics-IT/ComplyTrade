"""
Windows-Compatible OCR Engine
Enhanced with GOT-OCR API support and NO TIMEOUT for large files
"""

import numpy as np
from PIL import Image
from typing import Optional
import os
import requests
from pathlib import Path


class EnhancedOCRProcessor:
    """OCR processor optimized for Windows with API support"""
    
    # Default API URL (can be overridden by environment variable)
    OCR_API_URL = "http://10.20.10.2:8082/ocr/"
    
    def __init__(self, backend='tesseract', language='eng', use_preprocessing=True, 
                 use_postprocessing=True, use_api=False):
        self.backend = backend
        self.language = language
        self.use_preprocessing = use_preprocessing
        self.use_postprocessing = use_postprocessing
        self.use_api = use_api  # NEW: Flag to use API
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
        
        if not self.use_api:
            self._initialize_backend()
        else:
            print(f"✓ Using GOT-OCR API at {os.getenv('OCR_API_URL', self.OCR_API_URL)}")
    
    def _initialize_backend(self):
        """Initialize OCR backend (Tesseract, EasyOCR, or PaddleOCR)"""
        # Mapping Tesseract codes to EasyOCR codes
        TESSERACT_TO_EASYOCR_LANG = {
            'eng': 'en',
            'fra': 'fr',
            'deu': 'de',
            'spa': 'es',
            'ita': 'it',
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
                    'eng': 'en',
                    'fra': 'en',
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
            image = Image.open(image_path)
            
            # Preprocessing
            if self.use_preprocessing:
                if aggressive_preprocessing:
                    image = self.preprocessor.aggressive_preprocess(image)
                else:
                    image = self.preprocessor.preprocess(image)
            
            # Extract text based on backend
            if self.backend.lower() == 'tesseract':
                text = self._tesseract_extract(image)
            elif self.backend.lower() == 'easyocr':
                text = self._easyocr_extract(image)
            elif self.backend.lower() == 'paddleocr':
                text = self._paddleocr_extract(image)
            else:
                text = self._simple_extract(image)
            
            # Postprocessing
            if self.use_postprocessing and self.postprocessor:
                if aggressive_postprocessing:
                    text = self.postprocessor.aggressive_clean(text)
                else:
                    text = self.postprocessor.clean(text)
            
            return text
            
        except Exception as e:
            print(f"    ⚠ Error extracting text: {str(e)}")
            return ""
    
    def _ocr_via_api(self, file_path: str) -> str:
        """
        Send the ENTIRE document (PDF or Image) to the GOT-OCR API.
        NO TIMEOUT - will wait indefinitely for large documents to process.
        
        This is CRITICAL for handling large multi-page PDFs that may take
        10+ minutes to process on the server.
        """
        try:
            # Get URL from environment variable or use default
            target_url = os.getenv("OCR_API_URL", self.OCR_API_URL)
            
            file_path_obj = Path(file_path)
            file_name = file_path_obj.name
            file_size_mb = file_path_obj.stat().st_size / (1024 * 1024)
            
            print(f"  → [API] Preparing full document upload: {file_name}")
            print(f"  → [API] File size: {file_size_mb:.2f} MB")
            print(f"  → [API] Target Endpoint: {target_url}")
            print(f"  → [API] ⏳ NO TIMEOUT - Will wait until processing completes (may take several minutes)...")
            
            with open(file_path, "rb") as f:
                files = {"file": (file_name, f)}
                
                # CRITICAL: timeout=None means NO TIMEOUT
                # The request will wait indefinitely for the server to respond
                # This is essential for large documents that may take 10-30 minutes
                response = requests.post(
                    target_url,
                    files=files,
                    timeout=None  # Wait indefinitely - no matter how long it takes!
                )
                
                # Check for server-side errors
                if response.status_code == 500:
                    error_msg = "Server returned 500 Internal Error. Check GOT-OCR server logs."
                    print(f"  ✗ {error_msg}")
                    return '{"status": "error", "error": "' + error_msg + '", "identified_objects": []}'
                
                # Check for other HTTP errors
                if response.status_code != 200:
                    error_msg = f"Server returned {response.status_code}: {response.text[:200]}"
                    print(f"  ✗ {error_msg}")
                    return '{"status": "error", "error": "' + error_msg + '", "identified_objects": []}'
                
                # Success!
                server_json_string = response.text
                response_size_kb = len(server_json_string) / 1024
                print(f"  ✓ OCR API successfully processed the entire document ({response_size_kb:.1f} KB response)")
                
                return server_json_string
                
        except requests.exceptions.Timeout:
            # This should NEVER happen now since we set timeout=None
            # But keeping the handler just in case
            error_msg = "Request timed out unexpectedly (this should not happen with timeout=None)"
            print(f"  ✗ {error_msg}")
            return '{"status": "error", "error": "' + error_msg + '", "identified_objects": []}'
        
        except requests.exceptions.ConnectionError as e:
            error_msg = f"Cannot connect to OCR server at {target_url}. Is the server running?"
            print(f"  ✗ {error_msg}")
            return '{"status": "error", "error": "' + error_msg + '", "identified_objects": []}'
        
        except requests.exceptions.RequestException as e:
            error_msg = f"Network error: {str(e)}"
            print(f"  ✗ {error_msg}")
            return '{"status": "error", "error": "' + error_msg + '", "identified_objects": []}'
        
        except FileNotFoundError:
            error_msg = f"File not found: {file_path}"
            print(f"  ✗ {error_msg}")
            return '{"status": "error", "error": "' + error_msg + '", "identified_objects": []}'
        
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            print(f"  ✗ {error_msg}")
            return '{"status": "error", "error": "' + error_msg + '", "identified_objects": []}'
    
    def _simple_extract(self, img: Image.Image) -> str:
        """Simple fallback extraction"""
        try:
            if self.backend.lower() == 'tesseract':
                import pytesseract
                return pytesseract.image_to_string(img)
        except Exception:
            pass
        return ""
    
    def _tesseract_extract(self, img: Image.Image) -> str:
        """Extract using Tesseract with multiple PSM modes"""
        import pytesseract
        
        # Try different page segmentation modes
        psm_modes = [6, 3, 4]
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