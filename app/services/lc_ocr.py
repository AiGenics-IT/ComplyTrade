# """
# Enhanced OCR Module with Multi-Document Support
# Automatically splits and processes multiple documents from a single PDF
# """

# import os
# import sys
# from pathlib import Path
# from typing import List, Optional, Tuple, Dict
# import json
# import numpy as np
# from PIL import Image, ImageEnhance, ImageFilter
# import cv2
# import tempfile
# import re
# from services.ai_postProcessor import AIOCRPostProcessor
# from services.document_splitter import MultiDocumentSplitter

# # class MultiDocumentSplitter:
# #     """Intelligently splits combined text into individual documents"""
    
# #     def __init__(self):
# #         self.swift_markers = [
# #             r'Message\s*type:\s*(\d{3})',
# #             r'MT\s*(\d{3})',
# #             r'Formatted\s+(Outward|Inward)\s+SWIFT',
# #             r'----------SWIFT_MT\d+------',
# #         ]
        
# #         self.page_markers = [
# #             r'---\s*Page\s+\d+\s*---',
# #             r'Page\s+\d+\s+of\s+\d+',
# #         ]
    
# #     def split_documents(self, combined_text: str) -> List[Tuple[str, str]]:
# #         """Split combined text into individual documents"""
        
# #         # Strategy 1: Split by SWIFT message type markers
# #         swift_splits = self._split_by_swift_markers(combined_text)
        
# #         if len(swift_splits) > 1:
# #             print(f"✓ Found {len(swift_splits)} documents using SWIFT markers")
# #             return [(doc, self._detect_doc_type(doc)) for doc in swift_splits]
        
# #         # Strategy 2: Split by page markers
# #         page_splits = self._split_by_page_markers(combined_text)
        
# #         if len(page_splits) > 1:
# #             print(f"✓ Found {len(page_splits)} documents using page markers")
# #             return [(doc, self._detect_doc_type(doc)) for doc in page_splits]
        
# #         # Strategy 3: Single document
# #         print(f"✓ Single document detected")
# #         return [(combined_text, self._detect_doc_type(combined_text))]
    
# #     def _split_by_swift_markers(self, text: str) -> List[str]:
# #         """Split by SWIFT message type markers"""
# #         split_positions = []
        
# #         for pattern in self.swift_markers:
# #             for match in re.finditer(pattern, text, re.IGNORECASE):
# #                 split_positions.append(match.start())
        
# #         if not split_positions:
# #             return [text]
        
# #         split_positions = sorted(set(split_positions))
        
# #         documents = []
# #         for i, pos in enumerate(split_positions):
# #             if i < len(split_positions) - 1:
# #                 doc_text = text[pos:split_positions[i + 1]]
# #             else:
# #                 doc_text = text[pos:]
            
# #             if len(doc_text.strip()) > 100:
# #                 documents.append(doc_text.strip())
        
# #         return documents
    
# #     def _split_by_page_markers(self, text: str) -> List[str]:
# #         """Split by page markers"""
# #         split_positions = []
        
# #         for pattern in self.page_markers:
# #             for match in re.finditer(pattern, text, re.IGNORECASE):
# #                 split_positions.append(match.start())
        
# #         if not split_positions:
# #             return [text]
        
# #         split_positions = sorted(set(split_positions))
        
# #         documents = []
# #         for i, pos in enumerate(split_positions):
# #             if i < len(split_positions) - 1:
# #                 doc_text = text[pos:split_positions[i + 1]]
# #             else:
# #                 doc_text = text[pos:]
            
# #             if len(doc_text.strip()) > 100:
# #                 documents.append(doc_text.strip())
        
# #         return documents
    
# #     def _detect_doc_type(self, text: str) -> str:
# #         """Detect document type from content"""
# #         text_upper = text.upper()
        
# #         mt_match = re.search(r'MESSAGE\s*TYPE:\s*(\d{3})|MT\s*(\d{3})', text_upper)
# #         if mt_match:
# #             mt_type = mt_match.group(1) or mt_match.group(2)
# #             if mt_type == '700':
# #                 return 'LC'
# #             elif mt_type in ['707', '747', '767']:
# #                 return 'AMENDMENT'
# #             else:
# #                 return 'SWIFT_OTHER'
        
# #         if any(x in text_upper for x in ['26E:', 'NUMBER OF AMENDMENT']):
# #             return 'AMENDMENT'
        
# #         if 'DOCUMENTARY CREDIT NUMBER' in text_upper:
# #             return 'LC'
        
# #         if 'INVOICE' in text_upper:
# #             return 'INVOICE'
# #         elif 'BILL OF LADING' in text_upper or 'B/L' in text_upper:
# #             return 'BILL_OF_LADING'
# #         elif 'CERTIFICATE' in text_upper:
# #             return 'CERTIFICATE'
        
# #         return 'UNKNOWN'


# class ImagePreprocessor:
#     """Advanced image preprocessing for OCR accuracy improvement"""
    
#     def __init__(self):
#         self.default_dpi = 300
    
#     def preprocess_for_ocr(self, image: Image.Image, aggressive: bool = False) -> Image.Image:
#         """Comprehensive preprocessing pipeline for OCR"""
#         img_array = np.array(image)
        
#         if len(img_array.shape) == 3:
#             img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
#         img_array = self._denoise(img_array, aggressive)
#         img_array = self._enhance_contrast(img_array)
#         img_array = self._sharpen(img_array, aggressive)
#         img_array = self._binarize(img_array, aggressive)
#         img_array = self._deskew(img_array)
#         img_array = self._remove_borders(img_array)
        
#         return Image.fromarray(img_array)
    
#     def _denoise(self, img: np.ndarray, aggressive: bool = False) -> np.ndarray:
#         if aggressive:
#             img = cv2.GaussianBlur(img, (5, 5), 0)
#         else:
#             img = cv2.GaussianBlur(img, (3, 3), 0)
        
#         img = cv2.fastNlMeansDenoising(img, None, h=10, templateWindowSize=7, searchWindowSize=21)
#         img = cv2.bilateralFilter(img, 9, 75, 75)
        
#         return img
    
#     def _enhance_contrast(self, img: np.ndarray) -> np.ndarray:
#         clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#         return clahe.apply(img)
    
#     def _sharpen(self, img: np.ndarray, aggressive: bool = False) -> np.ndarray:
#         if aggressive:
#             kernel = np.array([[-1, -1, -1], [-1,  9, -1], [-1, -1, -1]])
#         else:
#             kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        
#         return cv2.filter2D(img, -1, kernel)
    
#     def _binarize(self, img: np.ndarray, aggressive: bool = False) -> np.ndarray:
#         if aggressive:
#             _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#         else:
#             img = cv2.adaptiveThreshold(
#                 img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
#                 cv2.THRESH_BINARY, 11, 2
#             )
#         return img
    
#     def _deskew(self, img: np.ndarray) -> np.ndarray:
#         edges = cv2.Canny(img, 50, 150, apertureSize=3)
#         lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
        
#         if lines is not None and len(lines) > 0:
#             angles = []
#             for rho, theta in lines[:, 0]:
#                 angle = np.degrees(theta) - 90
#                 if -45 < angle < 45:
#                     angles.append(angle)
            
#             if angles:
#                 median_angle = np.median(angles)
#                 if abs(median_angle) > 0.5:
#                     (h, w) = img.shape
#                     center = (w // 2, h // 2)
#                     M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
#                     img = cv2.warpAffine(img, M, (w, h), 
#                                         flags=cv2.INTER_CUBIC, 
#                                         borderMode=cv2.BORDER_REPLICATE)
#         return img
    
#     def _remove_borders(self, img: np.ndarray) -> np.ndarray:
#         contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
#         if contours:
#             largest_contour = max(contours, key=cv2.contourArea)
#             x, y, w, h = cv2.boundingRect(largest_contour)
            
#             margin = 10
#             y_start = max(0, y - margin)
#             y_end = min(img.shape[0], y + h + margin)
#             x_start = max(0, x - margin)
#             x_end = min(img.shape[1], x + w + margin)
            
#             img = img[y_start:y_end, x_start:x_end]
        
#         return img
    
#     def upscale_image(self, image: Image.Image, target_dpi: int = 300) -> Image.Image:
#         width, height = image.size
#         current_dpi = image.info.get('dpi', (72, 72))[0]
#         scale_factor = target_dpi / current_dpi
        
#         if scale_factor > 1:
#             new_width = int(width * scale_factor)
#             new_height = int(height * scale_factor)
#             image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
#         return image


# class EnhancedOCRProcessor:
#     """Enhanced OCR processor with automatic post-processing and document splitting"""
    
#     def __init__(self, backend='tesseract', language='eng', use_preprocessing=True, use_postprocessing=True):
#         self.backend = backend
#         self.language = language
#         self.use_preprocessing = use_preprocessing
#         self.use_postprocessing = use_postprocessing
#         self.preprocessor = ImagePreprocessor()
#         self.postprocessor = AIOCRPostProcessor()
#         self.splitter = MultiDocumentSplitter()
#         self.ocr_engine = None
        
#         self._initialize_backend()
    
#     def _initialize_backend(self):
#         """Initialize the selected OCR backend"""
#         if self.backend == 'tesseract':
#             try:
#                 import pytesseract
#                 self.ocr_engine = pytesseract
#                 print(f"✓ Tesseract OCR initialized with post-processing")
#             except ImportError:
#                 print("⚠ Tesseract not available")
#                 self.ocr_engine = None
        
#         elif self.backend == 'easyocr':
#             try:
#                 import easyocr
#                 self.ocr_engine = easyocr.Reader([self.language], gpu=False)
#                 print(f"✓ EasyOCR initialized with post-processing")
#             except ImportError:
#                 print("⚠ EasyOCR not available")
#                 self.ocr_engine = None
        
#         elif self.backend == 'paddleocr':
#             try:
#                 from paddleocr import PaddleOCR
#                 self.ocr_engine = PaddleOCR(use_angle_cls=True, lang='en')
#                 print(f"✓ PaddleOCR initialized with post-processing")
#             except ImportError:
#                 print("⚠ PaddleOCR not available")
#                 self.ocr_engine = None
    
#     def extract_text_from_image(self, image_path: str, aggressive_preprocessing: bool = False, 
#                                aggressive_postprocessing: bool = False) -> str:
#         """Extract text from image with automatic post-processing"""
#         if self.ocr_engine is None:
#             raise RuntimeError(f"OCR backend '{self.backend}' not available")
        
#         try:
#             img = Image.open(image_path)
            
#             if img.mode != 'RGB':
#                 img = img.convert('RGB')
            
#             img = self.preprocessor.upscale_image(img, target_dpi=300)
            
#             if self.use_preprocessing:
#                 img = self.preprocessor.preprocess_for_ocr(img, aggressive=aggressive_preprocessing)
            
#             # Extract raw text
#             if self.backend == 'tesseract':
#                 text = self._tesseract_extract(img)
#             elif self.backend == 'easyocr':
#                 text = self._easyocr_extract(img)
#             elif self.backend == 'paddleocr':
#                 text = self._paddleocr_extract(img)
#             else:
#                 text = ""
            
#             # Apply post-processing
#             if self.use_postprocessing and text:
#                 text = self.postprocessor.clean_text(text)
            
#             return text
            
#         except Exception as e:
#             print(f"Error extracting text: {str(e)}")
#             return ""
    
#     def _tesseract_extract(self, img: Image.Image) -> str:
#         """Extract text using Tesseract"""
#         import pytesseract
        
#         psm_modes = [6, 3, 4, 11]
#         best_text = ""
#         best_confidence = 0
        
#         for psm in psm_modes:
#             config = f'--oem 3 --psm {psm}'
            
#             try:
#                 data = pytesseract.image_to_data(img, config=config, output_type=pytesseract.Output.DICT)
#                 text = pytesseract.image_to_string(img, config=config)
                
#                 confidences = [int(conf) for conf in data['conf'] if conf != '-1']
#                 avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                
#                 if avg_confidence > best_confidence:
#                     best_confidence = avg_confidence
#                     best_text = text
                    
#             except Exception:
#                 continue
        
#         return best_text
    
#     def _easyocr_extract(self, img: Image.Image) -> str:
#         """Extract text using EasyOCR"""
#         img_array = np.array(img)
#         results = self.ocr_engine.readtext(img_array, detail=0)
#         return '\n'.join(results)
    
#     def _paddleocr_extract(self, img: Image.Image) -> str:
#         """Extract text using PaddleOCR"""
#         img_array = np.array(img)
#         result = self.ocr_engine.ocr(img_array, cls=True)
        
#         text_lines = []
#         if result and result[0]:
#             for line in result[0]:
#                 if line:
#                     text_lines.append(line[1][0])
        
#         return '\n'.join(text_lines)


# class PDFProcessorEnhanced:
#     """Enhanced PDF processor with multi-document support"""
    
#     def __init__(self, ocr_backend='tesseract', use_preprocessing=True, use_postprocessing=True):
#         self.ocr_processor = EnhancedOCRProcessor(
#             backend=ocr_backend, 
#             use_preprocessing=use_preprocessing,
#             use_postprocessing=use_postprocessing
#         )
#         self.splitter = MultiDocumentSplitter()
    
#     def extract_text_from_pdf(self, pdf_path: str, use_ocr=False, aggressive_ocr=False,
#                              aggressive_postprocessing=False, split_documents=True) -> str:
#         """
#         Extract text from PDF with optional document splitting
        
#         Args:
#             pdf_path: Path to PDF
#             use_ocr: Force OCR
#             aggressive_ocr: Aggressive preprocessing
#             aggressive_postprocessing: Aggressive text cleanup
#             split_documents: Return split documents (for multi-doc PDFs)
        
#         Returns:
#             Extracted text (combined or split)
#         """
#         text = ""
        
#         try:
#             if not use_ocr:
#                 text = self._extract_text_layer(pdf_path)
            
#             if not text.strip() or use_ocr:
#                 print(f"Using OCR with post-processing for: {pdf_path}")
#                 text = self._extract_with_enhanced_ocr(pdf_path, aggressive_ocr, aggressive_postprocessing)
            
#         except Exception as e:
#             print(f"Error processing PDF: {str(e)}")
        
#         return text
    
#     def extract_and_split(self, pdf_path: str, use_ocr=False) -> List[Dict[str, str]]:
#         """
#         Extract and split into multiple documents
        
#         Returns:
#             List of {text, type, index} dictionaries
#         """
#         # Get combined text
#         combined_text = self.extract_text_from_pdf(pdf_path, use_ocr=use_ocr)
        
#         # Split into documents
#         split_docs = self.splitter.split_documents(combined_text)
        
#         # Format results
#         results = []
#         for i, (doc_text, doc_type) in enumerate(split_docs):
#             results.append({
#                 'index': i + 1,
#                 'type': doc_type,
#                 'text': doc_text,
#                 'length': len(doc_text)
#             })
        
#         return results
    
#     def _extract_text_layer(self, pdf_path: str) -> str:
#         """Extract text layer from digital PDF"""
#         try:
#             import pdfplumber
            
#             text = ""
#             with pdfplumber.open(pdf_path) as pdf:
#                 for page in pdf.pages:
#                     page_text = page.extract_text()
#                     if page_text:
#                         text += page_text + "\n"
            
#             return text
#         except ImportError:
#             print("⚠ pdfplumber not available")
#             return ""
    
#     def _extract_with_enhanced_ocr(self, pdf_path: str, aggressive_preprocess: bool = False,
#                                    aggressive_postprocess: bool = False) -> str:
#         """Extract with OCR and post-processing"""
#         try:
#             from pdf2image import convert_from_path
            
#             images = convert_from_path(pdf_path, dpi=300, fmt='png')
            
#             text = ""
#             for i, img in enumerate(images):
#                 print(f"  Processing page {i+1}/{len(images)}...")
                
#                 with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
#                     temp_img_path = tmp.name
                
#                 try:
#                     img.save(temp_img_path, 'PNG', quality=100)
                    
#                     page_text = self.ocr_processor.extract_text_from_image(
#                         temp_img_path, 
#                         aggressive_preprocessing=aggressive_preprocess,
#                         aggressive_postprocessing=aggressive_postprocess
#                     )
                    
#                     page_text = "".join(c for c in page_text if c.isprintable() or c in "\n\r\t")
                    
#                     text += f"\n--- Page {i+1} ---\n{page_text}\n"
                    
#                 finally:
#                     if os.path.exists(temp_img_path):
#                         os.remove(temp_img_path)
            
#             return text
        
#         except ImportError:
#             print("⚠ pdf2image not available")
#             return ""
#         except Exception as e:
#             print(f"⚠ OCR Failed: {str(e)}")
#             return ""


# class DocumentProcessor:
#     """Universal document processor with multi-document support"""
    
#     def __init__(self, ocr_backend='tesseract', use_preprocessing=True, use_postprocessing=True):
#         self.pdf_processor = PDFProcessorEnhanced(
#             ocr_backend=ocr_backend,
#             use_preprocessing=use_preprocessing,
#             use_postprocessing=use_postprocessing
#         )
#         self.ocr_processor = EnhancedOCRProcessor(
#             backend=ocr_backend,
#             use_preprocessing=use_preprocessing,
#             use_postprocessing=use_postprocessing
#         )
#         self.splitter = MultiDocumentSplitter()
    
#     def process_document(self, file_path: str, force_ocr=False, aggressive_ocr=False,
#                         aggressive_postprocessing=False, split_multi=True) -> str:
#         """
#         Process document with optional splitting
        
#         Args:
#             file_path: Path to file
#             force_ocr: Force OCR
#             aggressive_ocr: Aggressive preprocessing
#             aggressive_postprocessing: Aggressive cleanup
#             split_multi: Return split documents for multi-doc files
        
#         Returns:
#             Extracted text
#         """
#         file_path = Path(file_path)
#         extension = file_path.suffix.lower()
        
#         print(f"Processing: {file_path.name}")
        
#         if extension == '.pdf':
#             return self.pdf_processor.extract_text_from_pdf(
#                 str(file_path), 
#                 use_ocr=force_ocr, 
#                 aggressive_ocr=aggressive_ocr,
#                 aggressive_postprocessing=aggressive_postprocessing
#             )
        
#         elif extension in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif']:
#             return self.ocr_processor.extract_text_from_image(
#                 str(file_path),
#                 aggressive_preprocessing=aggressive_ocr,
#                 aggressive_postprocessing=aggressive_postprocessing
#             )
        
#         elif extension in ['.txt', '.text']:
#             with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
#                 return f.read()
        
#         else:
#             print(f"⚠ Unsupported format: {extension}")
#             return ""
    
#     def process_and_split(self, file_path: str, force_ocr=False) -> List[Dict[str, str]]:
#         """
#         Process file and split into multiple documents
        
#         Returns:
#             List of {text, type, index} dictionaries
#         """
#         # Get combined text
#         combined_text = self.process_document(file_path, force_ocr=force_ocr)
        
#         # Split into documents
#         split_docs = self.splitter.split_documents(combined_text)
        
#         # Format results
#         results = []
#         for i, (doc_text, doc_type) in enumerate(split_docs):
#             results.append({
#                 'index': i + 1,
#                 'type': doc_type,
#                 'text': doc_text,
#                 'length': len(doc_text)
#             })
        
#         print(f"\n✓ Extracted {len(results)} document(s) from {Path(file_path).name}")
#         for doc in results:
#             print(f"  {doc['index']}. {doc['type']} ({doc['length']} chars)")
        
#         return results


# if __name__ == "__main__":
#     print("Enhanced OCR Processor with Multi-Document Support")
#     print("Automatically splits combined PDFs into individual documents")