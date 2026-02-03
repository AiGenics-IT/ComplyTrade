"""
AI Hybrid Document Processor
AI layer that works with existing OCR system to produce clean, accurate text
Processes ALL document types: PDFs, images, and text files
"""

import torch
import os
from PIL import Image
from pathlib import Path
from typing import Dict, Optional
import numpy as np
from transformers import AutoProcessor, AutoModelForCausalLM, Florence2Processor


class AIHybridProcessor:
    """
    AI-powered hybrid processor that enhances existing OCR system
    Uses Florence-2 vision model for intelligent text extraction and cleaning
    """
    
    def __init__(self, existing_processor=None):
        """
        Args:
            existing_processor: Your existing DocumentProcessor instance
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.hf_token = os.getenv('HF_TOKEN')
        self.model = None
        self.processor = None
        self.use_ai = True
        
        # Store reference to existing processor
        self.existing_processor = existing_processor
        
        print(f"\n{'='*80}")
        print("AI HYBRID PROCESSOR - Initializing")
        print(f"{'='*80}")
        print(f"  → Device: {self.device}")
        print(f"  → AI Mode: Enabled")
        
        # Lazy load AI model
        self._model_loaded = False
    
    def _load_ai_model(self):
        """Lazy load Donut model: Stable, OCR-free document understanding."""
        if getattr(self, "_model_loaded", False):
            return True

        try:
            from transformers import DonutProcessor, VisionEncoderDecoderModel
            import torch
            import os
            from pathlib import Path

            model_name = "naver-clova-ix/donut-base-finetuned-docvqa"
            model_root = os.getenv("MODEL_PATH")
            model_root = Path(model_root) if (model_root and str(model_root).strip()) else None
            
            print(f"[OCR-AI] Initializing Donut: {model_name}...")

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Standard load without memory-saving 'meta' device tricks
            self.processor = DonutProcessor.from_pretrained(
                model_name, 
                cache_dir=str(model_root) if model_root else None
            )
            self.model = VisionEncoderDecoderModel.from_pretrained(
                model_name,
                cache_dir=str(model_root) if model_root else None,
                low_cpu_mem_usage=False,
                device_map=None
            )
            
            self.model.to(self.device)
            self.model.eval()

            print(f"[OCR-AI] Donut loaded successfully on {self.device}")
            self._model_loaded = True
            return True

        except Exception as e:
            print(f"[OCR-AI] Failed to load Donut: {e}")
            self.use_ai = False
            return False
    
    
    
    def process_document(self, file_path: str, force_ocr: bool = False, 
                        aggressive_ocr: bool = False, 
                        aggressive_postprocessing: bool = False) -> Dict:
        """
        Process any document type with AI enhancement
        
        Args:
            file_path: Path to document (PDF, image, or text)
            force_ocr: Force OCR for PDFs
            aggressive_ocr: Use aggressive preprocessing
            aggressive_postprocessing: Use aggressive text cleaning
        
        Returns:
            Dict with:
                - 'text': Clean extracted text
                - 'method': Extraction method used
                - 'confidence': AI confidence score (if applicable)
                - 'quality': Detected quality
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        print(f"\n{'='*80}")
        print(f"AI HYBRID PROCESSING: {file_path.name}")
        print(f"{'='*80}")
        
        result = {
            'text': '',
            'method': 'unknown',
            'confidence': 0.0,
            'quality': 'unknown',
            'source_file': str(file_path)
        }
        
        # Route to appropriate processor
        if extension == '.pdf':
            result = self._process_pdf(file_path, force_ocr, aggressive_ocr, aggressive_postprocessing)
        elif extension in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif']:
            result = self._process_image(file_path, aggressive_ocr, aggressive_postprocessing)
        elif extension in ['.txt', '.text']:
            result = self._process_text(file_path)
        else:
            print(f"  ⚠ Unsupported format: {extension}")
            result['method'] = 'unsupported'
        
        # Final AI cleaning pass if text looks dirty
        if result['text'] and self._needs_cleaning(result['text']):
            print(f"  → AI cleaning pass...")
            result['text'] = self._ai_clean_text(result['text'])
        
        print(f"\n{'='*80}")
        print(f"EXTRACTION COMPLETE")
        print(f"{'='*80}")
        print(f"  Method: {result['method']}")
        print(f"  Quality: {result['quality']}")
        print(f"  Characters: {len(result['text'])}")
        print(f"  Confidence: {result['confidence']:.2f}")
        print(f"{'='*80}\n")
        
        return result
    
    def _process_pdf(self, pdf_path: Path, force_ocr: bool, 
                    aggressive_ocr: bool, aggressive_postprocessing: bool) -> Dict:
        """
        Process PDF with hybrid approach
        1. Try digital text extraction
        2. If poor quality or force_ocr, use AI-enhanced OCR
        """
        result = {
            'text': '',
            'method': 'pdf_hybrid',
            'confidence': 1.0,
            'quality': 'unknown'
        }
        
        # Step 1: Try traditional extraction first
        if not force_ocr and self.existing_processor:
            print("  → Step 1: Trying digital text extraction...")
            try:
                from .pdf_processor import PDFProcessorEnhanced
                pdf_proc = PDFProcessorEnhanced()
                text = pdf_proc._extract_text_layer(str(pdf_path))
                
                if text and len(text.strip()) > 100:
                    print(f"  ✓ Extracted {len(text)} chars from text layer")
                    result['text'] = text
                    result['method'] = 'pdf_digital'
                    result['quality'] = 'good'
                    return result
            except Exception as e:
                print(f"  ⚠ Digital extraction failed: {str(e)}")
        
        # Step 2: Use AI-enhanced OCR
        print("  → Step 2: Using AI-enhanced OCR...")
        return self._process_pdf_with_ai_ocr(pdf_path, aggressive_ocr, aggressive_postprocessing)
    
    def _process_pdf_with_ai_ocr(self, pdf_path: Path, aggressive: bool, 
                                aggressive_postprocessing: bool) -> Dict:
        """Process PDF pages with AI-enhanced OCR"""
        try:
            from pdf2image import convert_from_path
            import tempfile
            
            # Convert PDF to images
            print(f"  → Converting PDF to images...")
            images = convert_from_path(str(pdf_path), dpi=300, fmt='png')
            
            all_text = []
            total_confidence = 0.0
            quality_scores = []
            
            for i, img in enumerate(images):
                print(f"\n  → Processing page {i+1}/{len(images)}...")
                
                # Save temp image
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    temp_path = tmp.name
                    img.save(temp_path, 'PNG')
                
                try:
                    # Process with AI hybrid
                    page_result = self._process_image(Path(temp_path), aggressive, aggressive_postprocessing)
                    
                    if page_result['text']:
                        all_text.append(f"--- Page {i+1} ---\n{page_result['text']}")
                        total_confidence += page_result['confidence']
                        quality_scores.append(page_result['quality'])
                
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
            
            # Combine results
            combined_text = '\n\n'.join(all_text)
            avg_confidence = total_confidence / len(images) if images else 0.0
            
            # Determine overall quality
            if 'poor' in quality_scores or 'very_poor' in quality_scores:
                overall_quality = 'poor'
            elif 'medium' in quality_scores:
                overall_quality = 'medium'
            else:
                overall_quality = 'good'
            
            return {
                'text': combined_text,
                'method': 'pdf_ai_ocr',
                'confidence': avg_confidence,
                'quality': overall_quality
            }
        
        except Exception as e:
            print(f"  ✗ PDF AI OCR failed: {str(e)}")
            return {
                'text': '',
                'method': 'pdf_failed',
                'confidence': 0.0,
                'quality': 'failed'
            }
    
    def _process_image(self, image_path: Path, aggressive: bool, 
                      aggressive_postprocessing: bool) -> Dict:
        """
        Process image with hybrid AI + traditional OCR
        """
        result = {
            'text': '',
            'method': 'image_hybrid',
            'confidence': 0.0,
            'quality': 'unknown'
        }
        
        try:
            img = Image.open(image_path)
            
            # Detect image quality
            quality = self._detect_image_quality(img)
            # quality = 'very_poor'  # Force poor for testing
            result['quality'] = quality
            
            print(f"  → Detected quality: {quality}")
            
            # Strategy based on quality
            if quality in ['poor', 'very_poor'] or aggressive:
                # Use AI for poor quality
                if self.use_ai:
                    print(f"  → Using AI OCR for poor quality image...")
                    ai_result = self._ai_extract_text(img)
                    
                    if ai_result and len(ai_result) > 50:
                        result['text'] = ai_result
                        result['method'] = 'ai_vision'
                        result['confidence'] = 0.9
                        return result
            
            # Use traditional OCR
            print(f"  → Using traditional OCR...")
            if self.existing_processor:
                traditional_text = self.existing_processor.ocr_processor.extract_text_from_image(
                    str(image_path),
                    aggressive_preprocessing=aggressive,
                    aggressive_postprocessing=aggressive_postprocessing
                )
                
                # Check if result is garbage
                if self._is_garbage(traditional_text):
                    print(f"  ⚠ Traditional OCR produced garbage")
                    
                    # Try AI as fallback
                    if self.use_ai:
                        print(f"  → Trying AI OCR as fallback...")
                        ai_result = self._ai_extract_text(img)
                        
                        if ai_result and len(ai_result) > 50:
                            result['text'] = ai_result
                            result['method'] = 'ai_vision_fallback'
                            result['confidence'] = 0.85
                            return result
                
                result['text'] = traditional_text
                result['method'] = 'traditional_ocr'
                result['confidence'] = 0.7
            
            return result
        
        except Exception as e:
            print(f"  ✗ Image processing failed: {str(e)}")
            result['method'] = 'failed'
            return result
    
    def _process_text(self, text_path: Path) -> Dict:
        """Process plain text file"""
        try:
            with open(text_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            
            return {
                'text': text,
                'method': 'text_file',
                'confidence': 1.0,
                'quality': 'excellent'
            }
        except Exception as e:
            print(f"  ✗ Text file processing failed: {str(e)}")
            return {
                'text': '',
                'method': 'text_failed',
                'confidence': 0.0,
                'quality': 'failed'
            }
    
    def _detect_image_quality(self, img: Image.Image) -> str:
        """Detect image quality using AI or traditional metrics"""
        try:
            img_array = np.array(img)
            
            # Convert to grayscale
            if len(img_array.shape) == 3:
                import cv2
                img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                img_gray = img_array
            
            # Calculate metrics
            std_dev = img_gray.std()
            max_val = img_gray.max()
            min_val = img_gray.min()
            contrast_range = max_val - min_val
            
            # Sharpness
            import cv2
            laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
            sharpness = laplacian.var()
            
            # Score
            score = 0
            if contrast_range > 200: score += 2
            elif contrast_range < 150: score -= 3
            
            if std_dev > 60: score += 2
            elif std_dev < 40: score -= 3
            
            if max_val > 240: score += 2
            elif max_val < 220: score -= 2
            
            if sharpness > 500: score += 2
            elif sharpness < 100: score -= 2
            
            # Return quality
            if score >= 4: return 'excellent'
            elif score >= 1: return 'good'
            elif score >= -2: return 'medium'
            elif score >= -4: return 'poor'
            else: return 'very_poor'
        
        except:
            return 'unknown'
    
    def _ai_extract_text(self, img: Image.Image) -> str:
        """Extract text using Donut by prompting it to parse the document."""
        if not self.use_ai:
            return ""
        
        if not getattr(self, "_model_loaded", False):
            if not self._load_ai_model():
                return ""
        
        try:
            import torch
            import re

            # Donut requires RGB
            if img.mode != "RGB":
                img = img.convert("RGB")

            # 1. Prepare Inputs
            # Prompt Donut to extract text (using a standard parsing prompt)
            task_prompt = "<s_docvqa><s_question>Extract all the text from this document.</s_question><s_answer>"
            decoder_input_ids = self.processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
            pixel_values = self.processor(img, return_tensors="pt").pixel_values

            # Move everything to the correct device
            pixel_values = pixel_values.to(self.device)
            decoder_input_ids = decoder_input_ids.to(self.device)

            # 2. Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    pixel_values,
                    decoder_input_ids=decoder_input_ids,
                    max_length=self.model.config.decoder.max_position_embeddings,
                    early_stopping=True,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    use_cache=True,
                    num_beams=1, # Donut is already quite accurate with greedy search
                    bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                    return_dict_in_generate=True,
                )

            # 3. Process Output
            sequence = self.processor.batch_decode(outputs.sequences)[0]
            sequence = sequence.replace(self.processor.tokenizer.eos_token, "").replace(self.processor.tokenizer.pad_token, "")
            
            # Extract the content inside the answer tag
            extracted_text = re.search(r"<s_answer>(.*?)$", sequence, re.DOTALL)
            extracted_text = extracted_text.group(1).strip() if extracted_text else sequence
            
            # Clean up any leftover XML-like tags Donut might produce
            extracted_text = re.sub(r"<.*?>", "", extracted_text)

            print(f"  ✓ Donut extracted {len(extracted_text)} characters")
            return extracted_text
        
        except Exception as e:
            print(f"  ⚠ Donut extraction failed: {str(e)}")
            return ""
    
    
    
    
    
    def _is_garbage(self, text: str) -> bool:
        """Detect if OCR output is garbage"""
        if not text or len(text) < 50:
            return True
        
        # Calculate alphanumeric ratio
        alphanum_count = sum(c.isalnum() for c in text)
        ratio = alphanum_count / len(text) if len(text) > 0 else 0
        
        # If less than 30% alphanumeric, probably garbage
        if ratio < 0.3:
            return True
        
        # Check for excessive repeated characters
        unique_chars = len(set(text.replace(' ', '').replace('\n', '')))
        if unique_chars < 10:
            return True
        
        return False
    
    def _needs_cleaning(self, text: str) -> bool:
        """Check if text needs AI cleaning"""
        # Check for common OCR errors
        error_indicators = [
            text.count('â€¢') > 5,  # Unicode errors
            text.count('Â') > 5,
            text.count('â€') > 5,
            len([c for c in text if not c.isprintable()]) / len(text) > 0.05,  # Non-printable chars
        ]
        
        return any(error_indicators)
    
    def _ai_clean_text(self, text: str) -> str:
        """
        Use AI to clean and correct OCR text
        (Optional: can use GPT-like model for text correction)
        """
        # For now, just basic cleaning
        # You can enhance this with a language model later
        
        # Remove excessive whitespace
        import re
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        
        # Fix common OCR errors
        replacements = {
            'â€¢': '•',
            'â€™': "'",
            'â€œ': '"',
            'â€': '"',
            'Â': '',
            'Ã©': 'é',
            'Ã¨': 'è',
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text


# ============================================================================
# WRAPPER FOR EXISTING SYSTEM
# ============================================================================

def create_hybrid_processor(existing_processor=None):
    """
    Factory function to create hybrid processor
    
    Usage:
        from services.OCR.document_processor import DocumentProcessor
        from services.OCR.ai_hybrid_layer import create_hybrid_processor
        
        # Create existing processor
        doc_processor = DocumentProcessor()
        
        # Wrap with AI layer
        ai_processor = create_hybrid_processor(doc_processor)
        
        # Use it
        result = ai_processor.process_document('document.pdf')
        clean_text = result['text']
    """
    return AIHybridProcessor(existing_processor)