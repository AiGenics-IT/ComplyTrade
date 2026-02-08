"""
LC Extractor
Main extraction logic for LC and Amendment documents
Supports both regex-based and LLM-based (Flan-T5) extraction
"""

import os
import re
import json
import torch
from pathlib import Path
from typing import Optional, Any, Dict, Literal
from .models import LCDocument
from .constants import COMPREHENSIVE_FIELD_MAPPINGS, SWIFT_MESSAGE_TYPES
from .utils import normalize_lc_number, sanitize_text
from .field_extractor import FieldExtractor
from .document_detector import DocumentDetector
from services.ai_postProcessor import AIOCRPostProcessor



class SWIFTField:
    """Helper class to allow .attribute access while remaining JSON serializable"""
    def __init__(self, field_code, field_name, value, raw_text):
        self.field_code = field_code
        self.field_name = field_name
        self.value = value
        self.raw_text = raw_text

    def to_dict(self):
        return {
            "field_code": self.field_code,
            "field_name": self.field_name,
            "value": self.value,
            "raw_text": self.raw_text
        }
        
        
class LCExtractor:
    """
    Bulletproof Universal SWIFT MT Extractor
    Supports both regex-based and LLM-based (Flan-T5) extraction
    """
    
    def __init__(
        self,
        use_llm: bool = False,  # Default to FALSE
        model_name: Optional[str] = None,
        model_root: Optional[str] = None,
        mode: Literal["llm", "regex", "auto"] = "regex"  # Default to REGEX
    ):
        """
        Initialize LC Extractor
        
        Args:
            use_llm: Enable LLM-based extraction (default: False)
            model_name: Hugging Face model name (default: from MODEL_NAME env or "google/flan-t5-large")
            model_root: Local model path (default: from MODEL_PATH env)
            mode: Extraction mode - "llm", "regex", or "auto" (default: regex)
        """
        # FORCE REGEX MODE - LLM is unreliable
        if use_llm and mode in ["llm", "auto"]:
            print(f"[LCExtractor] ⚠️  WARNING: LLM mode requested but unreliable. Forcing REGEX mode.")
            self.mode = "regex"
        else:
            self.mode = "regex"  # Always use regex
        
        self.current_doc = None
        self.current_doc_fields = None
        self.field_mappings = COMPREHENSIVE_FIELD_MAPPINGS
        self.message_types = SWIFT_MESSAGE_TYPES
        
        # Initialize regex-based components (always available)
        print(f"[LCExtractor] ✓ Initializing in {self.mode.upper()} mode (PRODUCTION)")
        self.ocr_cleaner = AIOCRPostProcessor()
        self.field_extractor = FieldExtractor()
        self.detector = DocumentDetector()
        
        # DO NOT Initialize LLM - it's unreliable for SWIFT extraction
        self.llm_model = None
        self.llm_tokenizer = None
        self.device = None
        self.dtype = None
        
        # Only initialize LLM if explicitly requested AND mode is llm
        # (This code path should never execute in production)
        if use_llm and mode == "llm":
            print("[LCExtractor] ⚠️  LLM mode explicitly requested (not recommended)")
            self._initialize_llm(model_name, model_root)

    def _initialize_llm(self, model_name: Optional[str] = None, model_root: Optional[str] = None):
        """
        Initialize Flan-T5 model with GPU support and environment-based configuration.
        Downloads model to cache_dir if not already present.
        """
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, logging
            logging.set_verbosity_error()  # suppress unnecessary HF warnings
            
            # 1️⃣ Resolve Model Name (always use HuggingFace model name, not path)
            self.model_name = os.getenv("MODEL_NAME", model_name or "google/flan-t5-large")
            print(f"[LCExtractor] Resolved model name: {self.model_name}")

            # 2️⃣ Resolve Cache Directory (where to download/store the model)
            env_root = os.getenv("MODEL_PATH", model_root)
            if env_root and str(env_root).strip():
                self.model_root = Path(env_root)
                self.model_root.mkdir(parents=True, exist_ok=True)
                cache_dir = str(self.model_root)
                print(f"[LCExtractor] Using model cache directory: {cache_dir}")
            else:
                cache_dir = None
                print(f"[LCExtractor] Using default HuggingFace cache")

            # 3️⃣ Detect Device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            print(f"[LCExtractor] Device: {self.device} | dtype: {self.dtype}")
            
            # 4️⃣ Load Tokenizer (downloads to cache_dir if not present)
            print(f"[LCExtractor] Loading tokenizer for {self.model_name}...")
            self.llm_tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,  # ← Use model name, not path!
                use_fast=False,
                legacy=False,
                cache_dir=cache_dir  # ← Downloads here if needed
            )
            
            # 5️⃣ Load Model (downloads to cache_dir if not present)
            print(f"[LCExtractor] Loading model for {self.model_name}...")
            self.llm_model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,  # ← Use model name, not path!
                torch_dtype=self.dtype,
                device_map="auto" if torch.cuda.is_available() else None,
                cache_dir=cache_dir  # ← Downloads here if needed
            )
            
            # 6️⃣ Move to device if needed
            if not torch.cuda.is_available():
                self.llm_model.to(self.device)
            
            print(f"[LCExtractor] ✓ LLM loaded successfully on {self.device}")
            
        except ImportError as e:
            print(f"[LCExtractor] ⚠ transformers not installed: {e}")
            if self.mode == "llm":
                raise RuntimeError("LLM mode requires 'transformers' and 'torch'. Install: pip install transformers torch")
            else:
                print(f"[LCExtractor] Falling back to regex mode")
                self.mode = "regex"
                
        except Exception as e:
            print(f"[LCExtractor] ⚠ Failed to load LLM: {e}")
            if self.mode == "llm":
                raise
            else:
                print(f"[LCExtractor] Falling back to regex mode")
                self.mode = "regex"

    def extract_from_text(self, text: str) -> Any:
        """
        Extract data or categorize supporting documents for later validation.
        Handles LC, Amendments, and Supporting documents with robust OCR text processing.
        Uses LLM if available, otherwise falls back to regex.
        """
        # Log current mode
        print(f"\n{'='*70}")
        print(f"[LCExtractor] CURRENT MODE: {self.mode.upper()}")
        print(f"{'='*70}")
        
        # Route based on mode
        if self.mode == "llm":
            print("[LCExtractor] → Using LLM extraction")
            return self._extract_with_llm(text)
        elif self.mode == "regex":
            print("[LCExtractor] → Using REGEX extraction")
            return self._extract_with_regex(text)
        else:  # auto mode
            print("[LCExtractor] → Using AUTO mode (LLM with regex fallback)")
            try:
                result = self._extract_with_llm(text)
                # Validate result
                if self._is_extraction_complete(result):
                    print("[LCExtractor] ✓ LLM extraction successful")
                    return result
                else:
                    print("[LCExtractor] ⚠ LLM extraction incomplete, trying regex...")
                    return self._extract_with_regex(text)
            except Exception as e:
                print(f"[LCExtractor] ⚠ LLM extraction failed: {e}, trying regex...")
                return self._extract_with_regex(text)
    
    def _is_extraction_complete(self, doc: Any) -> bool:
        """Check if extraction result has minimum required fields"""
        if not doc:
            return False
        if doc.document_type in ["LC", "AMENDMENT"]:
            if not doc.lc_number or doc.lc_number == "":
                return False
            if doc.document_type == "AMENDMENT":
                if not hasattr(doc, 'amendment_number') or not doc.amendment_number:
                    return False
        return True
    
    def _extract_with_llm(self, text: str) -> Any:
        """
        Extract using Flan-T5 LLM
        """
        if self.llm_model is None:
            raise RuntimeError("LLM not initialized")
        
        print("\n[LCExtractor-LLM] Starting LLM-based extraction...")
        
        # Clean text
        text_clean = sanitize_text(text)
        
        # Detect document type with STRICT rules
        doc_type = self._detect_document_type_llm(text_clean)
        print(f"[LCExtractor-LLM] Detected document type: {doc_type}")
        
        # Route to appropriate extractor
        if doc_type == "LC":
            return self._extract_lc_llm(text_clean)
        elif doc_type == "AMENDMENT":
            return self._extract_amendment_llm(text_clean)
        else:
            return self._categorize_supporting_doc(text_clean)
    
    def _extract_with_regex(self, text: str) -> Any:
        """
        Extract using traditional regex patterns.
        Modified to support Alliance 'fin.xxx' and 'Unique Message Identifier' formats.
        """
        print("\n[LCExtractor-Regex] Starting regex-based extraction...")

        # --- Step 1: Minimal OCR cleaning ---
        text_clean = sanitize_text(text)

        # --- Step 2: Fix squashed tags, spacing, and preserve line breaks ---
        text_clean = self._prepare_text_for_extraction(text_clean)

        # --- Step 3: Normalize OCR text further for detection ---
        text_norm = self._normalize_ocr_text_for_detection(text_clean)

        # --- Step 4: Detect SWIFT MT message type (ROBUST VERSION) ---
        mt_type = None

        # Strategy A: Standard "Message type: 700"
        mt_type_match = re.search(r'Message\s*type[:]*\s*(\d{3})', text_norm, re.IGNORECASE)

        print(f"[LCExtractor-Regex] MT type detection asjkfas: {mt_type}")
        # Strategy B: Alliance "Identifier: fin.700"
        alliance_match = re.search(r'Identifier[:]*\s*fin\.?\s*(\d{3})', text_norm, re.IGNORECASE)
        
        # Strategy C: Alliance "Unique Message Identifier: ... 700 ..."
        unique_match = re.search(r'Unique\s*Message\s*Identifier:.*?\s(\d{3})\s', text_norm, re.IGNORECASE | re.DOTALL)

        if mt_type_match:
            mt_type = mt_type_match.group(1)
        elif alliance_match:
            mt_type = alliance_match.group(1)
            print(f"[LCExtractor] ✓ Detected MT{mt_type} via Alliance Identifier")
        elif unique_match:
            mt_type = unique_match.group(1)
            print(f"[LCExtractor] ✓ Detected MT{mt_type} via Unique Message ID")

        # Strategy D: Expansion Title Fallback
        if not mt_type:
            if "Issue of a Documentary Credit" in text_norm:
                mt_type = "700"
            elif "Amendment to a Documentary Credit" in text_norm:
                mt_type = "707"

        # --- Step 5: STRICT Document Type Detection ---
        # RULE: LC = MT700 or MT701 (no 26E tag)
        #       AMENDMENT = MT707 AND has 26E tag
        is_amendment = False
        
        if mt_type == "707":
            # MT707 - Check for amendment marker (support optional F prefix for Alliance)
            if re.search(r'F?26E[:|\s]|Number\s*of\s*Amendment', text_norm, re.IGNORECASE):
                is_amendment = True
                print(f"[LCExtractor] ✓ Detected AMENDMENT MT707 (has 26E tag)")
            else:
                print(f"[LCExtractor] ⚠ WARNING: MT707 without 26E tag, treating as LC")
                is_amendment = False
        
        elif mt_type in ["700", "701"]:
            # MT700/701 - ALWAYS LC, never amendment
            is_amendment = False
            print(f"[LCExtractor] ✓ Detected LC MT{mt_type}")
        
        else:
            # Unknown or missing MT type
            if mt_type:
                print(f"[LCExtractor] ⚠ Unknown MT type: {mt_type}")
            else:
                print(f"[LCExtractor] ⚠ No MT type found")
            is_amendment = False

        # --- Step 6: Route extraction ---
        if mt_type:
            if is_amendment:
                print(f"[LCExtractor] → Routing to Amendment extractor (MT{mt_type})")
                return self._extract_amendment(text_norm, mt_type)
            else:
                print(f"[LCExtractor] → Routing to LC extractor (MT{mt_type})")
                return self._extract_lc(text_norm, mt_type)

        # --- Step 7: Fallback for supporting documents ---
        print(f"[LCExtractor] → Treating as supporting document")
        return self._categorize_supporting_doc(text_norm)
    
    
    
    
    
    
    # ========================================================================
    # LLM-BASED EXTRACTION METHODS
    # ========================================================================
    
    def _detect_document_type_llm(self, text: str) -> str:
        """
        Detect document type using STRICT SWIFT MT validation
        
        RULES (NON-NEGOTIABLE):
        - LC: MT700 or MT701 (does NOT have 26E tag)
        - AMENDMENT: MT707 AND has 26E tag (Number of Amendment)
        - SUPPORTING: Everything else
        """
        # First, check for SWIFT message type
        mt_type_match = re.search(r'Message\s*type[:]*\s*(\d{3})', text, re.IGNORECASE)
        mt_type = mt_type_match.group(1) if mt_type_match else None
        
        print(f"[LCExtractor-LLM] MT Type found: {mt_type}")
        
        # STRICT VALIDATION
        if mt_type == "707":
            # MT707 - check for amendment marker (26E)
            has_26E = bool(re.search(r'26E[:|\s]|Number\s*of\s*Amendment', text, re.IGNORECASE))
            print(f"[LCExtractor-LLM] MT707 has 26E tag: {has_26E}")
            
            if has_26E:
                return "AMENDMENT"
            else:
                # MT707 without 26E - treat as LC (unusual but possible)
                print("[LCExtractor-LLM] ⚠ WARNING: MT707 without 26E tag, treating as LC")
                return "LC"
        
        elif mt_type in ["700", "701"]:
            # MT700/701 - these are ALWAYS LC, never amendments
            # Even if they have confusing text, trust the message type
            print(f"[LCExtractor-LLM] ✓ MT{mt_type} = LC (guaranteed)")
            return "LC"
        
        else:
            # No valid MT type - use LLM to classify
            print(f"[LCExtractor-LLM] No valid MT type, using LLM classification")
            prompt = f"""Analyze this document and identify its type.

Document text:
{text[:1000]}

Question: Is this document a:
1. Letter of Credit (LC) - SWIFT MT700 or MT701
2. Amendment to LC - SWIFT MT707 with tag 26E (Number of Amendment)
3. Supporting document (Invoice, Bill of Lading, Certificate, etc.)

Answer with only one word: LC, AMENDMENT, or SUPPORTING"""

            response = self._generate_llm(prompt, max_length=10)
            response = response.strip().upper()
            
            if "AMENDMENT" in response:
                return "AMENDMENT"
            elif "LC" in response:
                return "LC"
            else:
                return "SUPPORTING"
    
    def _extract_lc_llm(self, text: str) -> LCDocument:
        """Extract LC fields using LLM"""
        print(f"[LCExtractor-LLM] Extracting LC fields...")
        
        doc = LCDocument(
            document_type="LC",
            lc_number="",
            message_type="MT700",
            raw_text=text
        )
        
        # Extract key fields using targeted prompts
        doc.lc_number = self._extract_field_llm(
            text,
            "What is the Documentary Credit Number or LC Number? Look for tag 20: or 'Sender's Reference'. Extract only the alphanumeric code (e.g., ILC07860544623PK). Remove all spaces."
        )
        
        doc.issue_date = self._extract_field_llm(
            text,
            "What is the Date of Issue? Look for tag 31C:. Extract only the 6-digit date in YYMMDD format."
        )
        
        doc.sender = self._extract_field_llm(
            text,
            "What is the Issuing Bank SWIFT code? Look for tag 52A:. Extract only the 8-11 character SWIFT/BIC code (e.g., HABBPKKA786). Remove all spaces."
        )
        
        doc.receiver = self._extract_field_llm(
            text,
            "What is the Advising Bank? Look for 'To Institution:'. Extract only the 8-11 character SWIFT/BIC code. Remove all spaces."
        )
        
        # Extract structured fields
        doc.fields = self._extract_all_fields_llm(text, "LC")
        
        # Extract clauses
        doc.additional_conditions = self._extract_clauses_llm(
            text,
            "Extract all additional conditions from tag 47A. List each condition as a numbered point."
        )
        
        doc.documents_required = self._extract_clauses_llm(
            text,
            "Extract all required documents from tag 46A. List each document as a numbered point."
        )
        
        print(f"[LCExtractor-LLM] LC extraction complete:")
        print(f"  - LC Number: {doc.lc_number}")
        print(f"  - Issue Date: {doc.issue_date}")
        print(f"  - Sender: {doc.sender}")
        print(f"  - Receiver: {doc.receiver}")
        
        return doc
    
    def _extract_amendment_llm(self, text: str) -> LCDocument:
        """Extract Amendment fields using LLM"""
        print(f"[LCExtractor-LLM] Extracting Amendment fields...")
        
        doc = LCDocument(
            document_type="AMENDMENT",
            lc_number="",
            message_type="MT707",
            raw_text=text
        )
        
        # Extract key fields
        doc.lc_number = self._extract_field_llm(
            text,
            "What is the LC Number being amended? Look for tag 20: or 'Sender's Reference'. The format is typically ILC followed by 11 digits and 2 letters. Remove all spaces and extract the complete code."
        )
        
        doc.amendment_number = self._extract_field_llm(
            text,
            "What is the Amendment Number? Look for tag 26E: or 'Number of Amendment'. Extract only the number."
        )
        
        doc.amendment_date = self._extract_field_llm(
            text,
            "What is the Date of Amendment? Look for tag 30:. Extract only the 6-digit date in YYMMDD format."
        )
        
        doc.issue_date = self._extract_field_llm(
            text,
            "What is the original Date of Issue? Look for tag 31C:. Extract only the 6-digit date in YYMMDD format."
        )
        
        doc.sender = self._extract_field_llm(
            text,
            "What is the Issuing Bank SWIFT code? Look for tag 52A: or 'Issuing Bank'. Extract only the 8-11 character code, removing any spaces."
        )
        
        doc.receiver = self._extract_field_llm(
            text,
            "What is the receiving bank? Look for 'To Institution:'. Extract only the 8-11 character SWIFT/BIC code, removing any spaces."
        )
        
        # Extract amendment changes
        doc.additional_conditions = self._extract_amendment_changes_llm(
            text,
            "Extract all changes to Additional Conditions from tag 47B. For each change, specify if it's ADD, DELETE, or REPLACE, and what clause number is affected."
        )
        
        doc.documents_required = self._extract_amendment_changes_llm(
            text,
            "Extract all changes to Documents Required from tag 46B. For each change, specify if it's ADD, DELETE, or REPLACE, and what clause number is affected."
        )
        
        # Extract all fields
        doc.fields = self._extract_all_fields_llm(text, "AMENDMENT")
        
        print(f"[LCExtractor-LLM] Amendment extraction complete:")
        print(f"  - LC Number: {doc.lc_number}")
        print(f"  - Amendment Number: {doc.amendment_number}")
        print(f"  - Amendment Date: {doc.amendment_date}")
        print(f"  - Sender: {doc.sender}")
        print(f"  - Receiver: {doc.receiver}")
        print(f"  - Changes: {len(doc.additional_conditions)} + {len(doc.documents_required)}")
        

        return doc
    
    def _extract_field_llm(self, text: str, instruction: str, max_length: int = 50) -> str:
        """Extract a single field using LLM"""
        text_truncated = text[:2000] if len(text) > 2000 else text
        
        prompt = f"""Extract information from this SWIFT message:

{text_truncated}

{instruction}

Answer with only the extracted value, no explanation:"""
        
        result = self._generate_llm(prompt, max_length=max_length)
        result = result.strip()
        
        # Post-process: remove spaces from codes
        if "SWIFT" in instruction or "BIC" in instruction or "LC Number" in instruction:
            result = re.sub(r'\s+', '', result)
        
        return result
    
    def _extract_clauses_llm(self, text: str, instruction: str) -> list:
        """Extract numbered clauses using LLM"""
        prompt = f"""Extract information from this SWIFT message:

{text[:1500]}

{instruction}

Format your answer as a JSON array of objects with 'number' and 'text' fields.
Example: [{{"number": "1", "text": "Clause text"}}, {{"number": "2", "text": "Another clause"}}]

Answer:"""
        
        result = self._generate_llm(prompt, max_length=500)
        
        try:
            clauses = json.loads(result)
            return clauses
        except:
            # Fallback: parse numbered list
            lines = result.split('\n')
            clauses = []
            for line in lines:
                match = re.match(r'(\d+)[.)\s]+(.+)', line.strip())
                if match:
                    clauses.append({'number': match.group(1), 'text': match.group(2).strip()})
            return clauses
    
    def _extract_amendment_changes_llm(self, text: str, instruction: str) -> list:
        """Extract amendment changes using LLM"""
        prompt = f"""Extract amendment changes from this SWIFT message:

{text[:1500]}

{instruction}

Format as JSON array with 'operation' (ADD/DELETE/REPLACE), 'clause_number', and 'content' fields.

Answer:"""
        
        result = self._generate_llm(prompt, max_length=500)
        
        try:
            changes = json.loads(result)
            return changes
        except:
            # Fallback regex
            changes = []
            for match in re.finditer(r'/(ADD|DELETE|REPLACE)/.*?CLAUSE\s*NO[.:]?\s*(\d+)', text, re.IGNORECASE):
                operation = match.group(1).upper()
                clause_num = match.group(2)
                start = match.end()
                content = text[start:start+200].strip()
                changes.append({'operation': operation, 'clause_number': clause_num, 'content': content})
            return changes
    
    def _extract_all_fields_llm(self, text: str, doc_type: str) -> Dict[str, str]:
        """Extract all SWIFT fields using LLM"""
        prompt = f"""Extract all SWIFT message fields from this {doc_type} document.

Document:
{text[:1500]}

List each field in format "TAG: VALUE"

Fields:"""
        
        result = self._generate_llm(prompt, max_length=300)
        
        fields = {}
        for line in result.split('\n'):
            match = re.match(r'([A-Z0-9]+):\s*(.+)', line.strip())
            if match:
                fields[match.group(1)] = match.group(2).strip()
        
        return fields
    
    def _generate_llm(self, prompt: str, max_length: int = 100) -> str:
        """Generate response using Flan-T5"""
        inputs = self.llm_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.llm_model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                temperature=0.3,
                do_sample=False,
                early_stopping=True
            )
        
        response = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    
    # ========================================================================
    # REGEX-BASED EXTRACTION METHODS (LEGACY)
    # ========================================================================
    
    def _prepare_text_for_extraction(self, text: str) -> str:
        """Clean OCR text while preserving tags and line breaks"""
        if not text:
            return ""

        # Fix OCR splits
        text = re.sub(r'\bI\s+LC\b', 'ILC', text, flags=re.IGNORECASE)
        text = re.sub(r'sw\s*if\s*t?', 'swift', text, flags=re.IGNORECASE)
        text = re.sub(r'NON\s+REF', 'NONREF', text, flags=re.IGNORECASE)
        
        # Fix spaced LC numbers
        text = re.sub(r'(ILC)\s+(\d{11})\s+([A-Z]{2})', r'\1\2\3', text, flags=re.IGNORECASE)
        text = re.sub(r'(ILC)\s+(\d{10,}[A-Z]{0,2})', r'\1\2', text, flags=re.IGNORECASE)
        text = re.sub(r'(ILC\s*\d+)\s*\n\s*([A-Z0-9]+)', r'\1\2', text, flags=re.IGNORECASE)
        
        # Fix dates split across lines
        text = re.sub(r'(\d)\s*\n\s*(\d{5})', r'\1\2', text)
        
        # Fix spaced SWIFT codes
        text = re.sub(r'([A-Z]{3,4})\s+([A-Z]{2})\s+([A-Z]{2})\s+([A-Z0-9]{3})', r'\1\2\3\4', text)
        text = re.sub(r'([A-Z]{2})\s+([A-Z]{2})\s+([A-Z]{2})\s+([A-Z]{2})\s+([A-Z]{3})', r'\1\2\3\4\5', text)

        # Ensure SWIFT tags start on new lines
        tags = ['Message type', 'To Institution', '20:', '21:', '23:', '26E:', '30:', 
                '31C:', '22A:', '46B:', '47B:', '52A:', 'Priority:', '27:', '45B:']
        for tag in tags:
            text = re.sub(rf'(?<!\n)({re.escape(tag)})', r'\n\1', text, flags=re.IGNORECASE)

        # Clean up spaces
        lines = text.splitlines()
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            line = re.sub(r'\s+', ' ', line)
            if line:
                cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)

    def _normalize_ocr_text_for_detection(self, text: str) -> str:
        """Normalize OCR text for detection"""
        normalized = text
        normalized = re.sub(r'(\d)\s+([A-Z])\s*:', r'\1\2:', normalized, flags=re.IGNORECASE)
        normalized = re.sub(r'([0-9A-Z]+)\s*:\s*', r'\1: ', normalized)
        normalized = re.sub(r'[ \t]{2,}', ' ', normalized)
        return normalized

    def _categorize_supporting_doc(self, text: str, file_name: str = "Unknown File") -> LCDocument:
        """Categorize non-SWIFT supporting documents"""
        classification = self.detector.categorize_supporting_doc(text, file_name)
        
        doc = LCDocument(
            document_type=classification['type'],
            lc_number="PENDING",
            message_type="NON_SWIFT",
            raw_text=text,
            fields={},
            additional_conditions=[],
            documents_required=[]
        )

        doc.is_supporting = True
        doc.file_name = file_name
        doc.classification = {
            "confidence": classification['confidence'],
            "matched_keywords": classification['matched_keywords']
        }
        doc.status = "stored_for_validation"

        return doc

    
    
    
    def _extract_lc(self, text: str, mt_type: Optional[str] = None) -> LCDocument:
        """Extract LC from ANY text format, including Alliance Header and standard SWIFT."""
        print(f"\n[LC Extract] ========== STARTING LC EXTRACTION ==========")
        print(f"[LC Extract] Text length: {len(text)} characters")
        print(f"[LC Extract] MT Type: {mt_type}")
        print(f"[LC Extract] Text preview (first 1000 chars):\n{text[:1000]}\n")
        
        doc = LCDocument(
            document_type="LC",
            lc_number="",
            message_type=f"MT{mt_type}" if mt_type else "MT700",
            raw_text=text
        )

        # --- Step 1: Metadata Extraction (Alliance Header & Patterns) ---
        print(f"[LC Extract] Attempting to extract LC metadata...")
        
        header_lc = re.search(r'Transaction Reference[:\s]+([A-Z0-9]+)', text, re.I)
        tag_lc = re.search(r'F?20[:\s]+(?:Documentary Credit Number\s+)?([A-Z0-9]+)', text, re.I)
        
        if header_lc:
            doc.lc_number = normalize_lc_number(header_lc.group(1).strip())
            print(f"[LC Extract] ✓ Found LC Number in Header: {doc.lc_number}")
        elif tag_lc:
            doc.lc_number = normalize_lc_number(tag_lc.group(1).strip())
            print(f"[LC Extract] ✓ Found LC Number in Tags: {doc.lc_number}")

        sender_match = re.search(r'Sender[:\s]+([A-Z0-9]{8,11})', text, re.I)
        if sender_match:
            doc.sender = sender_match.group(1).strip()

        receiver_match = re.search(r'Receiver[:\s]+([A-Z0-9]{8,11})', text, re.I)
        if receiver_match:
            doc.receiver = receiver_match.group(1).strip()

        date_match = re.search(r'F?31C[:\s]+(?:Date of Issue\s*)?(\d{6,})', text, re.I)
        if date_match:
            doc.issue_date = date_match.group(1).strip()

        # --- Step 2: Robust Field Extraction ---
        print(f"\n[LC Extract] Extracting all SWIFT fields...")
        
        # Capture tags (optional F) and content until next tag or footer
        field_pattern = r"(F?\d{2}[A-Z]?):\s*(.*?)(?=\s*F?\d{2}[A-Z]?:|Page\s*\d+|Unique Message|$)"
        all_matches = re.findall(field_pattern, text, re.DOTALL)
        
        field_names = {
            "20": "Documentary Credit Number", "27": "Sequence of Total",
            "31C": "Date of Issue", "31D": "Date and Place of Expiry",
            "32B": "Currency Code, Amount", "40A": "Form of Documentary Credit",
            "45A": "Description of Goods", "46A": "Documents Required",
            "47A": "Additional Conditions", "50": "Applicant", "59": "Beneficiary"
        }

        extracted_fields = {}
        for tag, content in all_matches:
            clean_tag = re.sub(r'^F', '', tag).strip()
            
            # Clean content and artifacts
            val = content.strip()
            val = re.sub(r'^(Narrative|Number|Amount|Currency)[:\s]+', '', val, flags=re.I)
            val = re.sub(r'\s+F$', '', val) 
            
            field_key = f":{clean_tag}:"
            
            # FIX: Use SWIFTField class here instead of LCExtractor
            extracted_fields[field_key] = SWIFTField(
                field_code=field_key,
                field_name=field_names.get(clean_tag, f"Field {clean_tag}"),
                value=val.replace('\n', ' ').strip(),
                raw_text=content.strip()
            )
        
        doc.fields = extracted_fields
        self.current_doc_fields = doc.fields
        print(f"[LC Extract] Extracted {len(doc.fields)} fields")

        # --- Step 3: Extract Clauses ---
        # Clauses extraction logic works here because doc.fields contains objects with .raw_text
        print(f"\n[LC Extract] Extracting clauses...")
        
        doc.additional_conditions = self.field_extractor.extract_numbered_points_robust(
            text, ['47A:', ':47A:', 'F47A:'], doc.fields
        )
        print(f"[LC Extract] Additional conditions: {len(doc.additional_conditions)} items")
        
        doc.documents_required = self.field_extractor.extract_numbered_points_robust(
            text, ['46A:', ':46A:', 'F46A:'], doc.fields
        )
        print(f"[LC Extract] Documents required: {len(doc.documents_required)} items")

        # --- Step 4: JSON Serialization Conversion ---
        # Convert SWIFTField objects back into dictionaries so json.dump() doesn't fail
        final_serialized_fields = {}
        for key, field_obj in doc.fields.items():
            if hasattr(field_obj, 'to_dict'):
                final_serialized_fields[key] = field_obj.to_dict()
            else:
                final_serialized_fields[key] = field_obj

        doc.fields = final_serialized_fields
        
        print(f"\n[LC Extract] ========== EXTRACTION COMPLETE ==========")
        return doc
    
    
    
    
    
    
    def _extract_amendment(self, text: str, mt_type: Optional[str] = None) -> LCDocument:
        """
        Extract amendment supporting both standard SWIFT and Alliance 'F-prefixed' formats.
        Robust against system labels like 'Narrative:', 'Code:', and 'Lines2to100:'.
        """
        print(f"[Amendment Extract] Starting robust amendment extraction...")
        
        doc = LCDocument(
            document_type="AMENDMENT",
            lc_number="",
            message_type=f"MT{mt_type}" if mt_type else "MT707",
            raw_text=text
        )   

        # --- Helper for Cleaning OCR Artifacts & System Labels ---
        def sanitize_content(val: str) -> str:
            # 1. Basic OCR character cleaning
            val = val.replace('’', "'").replace('‘', "'").replace('“', '"').replace('”', '"')
            
            # 2. Remove Alliance system labels (e.g., 'Narrative:', 'Expansion:')
            # This prevents system noise from being saved as part of the LC clause
            system_labels = [
                "Additional Conditions", "Documents Required", "Description of Goods",
                "Number:", "Expansion:", "Name and Address:", "Code:", "Narrative:", 
                "Lines2to100:", "Lines 2-100:", "Total:"
            ]
            for label in system_labels:
                val = re.sub(rf"{label}", "", val, flags=re.IGNORECASE)
            
            # 3. Strip non-ASCII blobs/emojis
            val = val.encode("ascii", "ignore").decode("ascii")
            return val.strip()

        # =========================================================
        # Fuzzy Metadata Extraction (F-Prefix & System ID Support)
        # =========================================================
        
        # 1. LC Number (Field 20 / F20)
        # Support for standard 'ILC...' and system IDs like '0239ILU012702'
        lc_match = re.search(r'F?20:.*?(?:Number)?\s*([A-Z0-9\s-]{8,25})', text, re.DOTALL | re.IGNORECASE)
        if lc_match:
            raw_lc = re.sub(r'\s+', '', lc_match.group(1))
            doc.lc_number = normalize_lc_number(raw_lc)

        # 2. Amendment Number (Field 26E / F26E)
        am_match = re.search(r'F?26\s*E:.*?(?:Number.*?Amendment)?\s*(\d+)', text, re.IGNORECASE | re.DOTALL)
        if am_match:
            doc.amendment_number = am_match.group(1).strip().zfill(2)

        # 3. Dates (Field 31C or 30 / F31C or F30)
        # Supports '210906' and '2021 Sep 06'
        issue_date_match = re.search(r'F?31\s*C:.*?(?:Date)?\s*([A-Z0-9\s-]{6,15})', text, re.IGNORECASE | re.DOTALL)
        if issue_date_match:
            doc.issue_date = re.sub(r'\s+', '', issue_date_match.group(1))

        amend_date_match = re.search(r'F?30:.*?(?:Date)?\s*([A-Z0-9\s-]{6,15})', text, re.IGNORECASE | re.DOTALL)
        if amend_date_match:
            doc.amendment_date = re.sub(r'\s+', '', amend_date_match.group(1))

        # 4. Sender/Receiver (BIC Extraction)
        bic_matches = re.findall(r'([A-Z]{6}[A-Z0-9]{2,5})', text)
        if len(bic_matches) >= 2:
            doc.sender = bic_matches[0]
            doc.receiver = bic_matches[1]

        # =============================
        # Extract Changes (The Body)
        # =============================
        # Updated Regex: Handles optional 'F' prefix. 
        # Boundaries: Stops at next Tag, URL, Page marker, or Alliance System Header.
        swift_field_pattern = r"(F?\d{2}[A-Z]?):\s*(.*?)(?=\s*F?\d{2}[A-Z]?:|http:|Page\s*\d+|Unique Message|Alliance Message|$)"
        extracted_fields = re.findall(swift_field_pattern, text, re.DOTALL)
        
        doc.additional_conditions = []
        doc.documents_required = []
        
        FOOTER_STOPPERS = ["http", "Page ", "Select 'Print'", "Fusion Trade", "Unique Message", "Delivery overdue"]

        for tag, content in extracted_fields:
            clean_val = sanitize_content(content)
            
            # Standardize Tag: Remove 'F' and whitespace (F 47 A -> 47A)
            clean_tag = re.sub(r'^F', '', tag.strip().upper())
            clean_tag = re.sub(r'\s+', '', clean_tag)

            # Apply footer stoppers
            for stopper in FOOTER_STOPPERS:
                if stopper.lower() in clean_val.lower():
                    clean_val = clean_val.split(stopper)[0].strip()

            if not clean_val or len(clean_val) < 5: 
                continue
                
            change_obj = {
                "field_code": clean_tag,
                "content": clean_val,
                "operation": "ADD" if "/ADD/" in clean_val.upper() else "CHANGE"
            }

            # Map Alliance tags (47A/46A) and Standard tags (47B/46B) to unified lists
            if clean_tag in ['47A', '47B']:
                doc.additional_conditions.append(change_obj)
            elif clean_tag in ['46A', '46B']:
                doc.documents_required.append(change_obj)
            elif clean_tag in ['45A', '45B', '79']:
                doc.additional_conditions.append(change_obj)

        total_changes = len(doc.additional_conditions) + len(doc.documents_required)
        print(f"[Amendment Extract] Complete: LC {doc.lc_number} | AM {doc.amendment_number} | Changes: {total_changes}")
        
        return doc