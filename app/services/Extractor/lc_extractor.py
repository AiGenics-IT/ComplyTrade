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


class LCExtractor:
    """
    Bulletproof Universal SWIFT MT Extractor
    Supports both regex-based and LLM-based (Flan-T5) extraction
    """
    
    def __init__(
        self,
        use_llm: bool = True,
        model_name: Optional[str] = None,
        model_root: Optional[str] = None,
        mode: Literal["llm", "regex", "auto"] = "auto"
    ):
        """
        Initialize LC Extractor
        
        Args:
            use_llm: Enable LLM-based extraction (default: True)
            model_name: Hugging Face model name (default: from MODEL_NAME env or "google/flan-t5-large")
            model_root: Local model path (default: from MODEL_PATH env)
            mode: Extraction mode - "llm", "regex", or "auto" (try LLM, fallback to regex)
        """
        self.mode = mode if use_llm else "regex"
        self.current_doc = None
        self.current_doc_fields = None
        self.field_mappings = COMPREHENSIVE_FIELD_MAPPINGS
        self.message_types = SWIFT_MESSAGE_TYPES
        
        # Initialize regex-based components (always available)
        print(f"[LCExtractor] Initializing in {self.mode} mode...")
        self.ocr_cleaner = AIOCRPostProcessor()
        self.field_extractor = FieldExtractor()
        self.detector = DocumentDetector()
        
        # Initialize LLM components if enabled
        self.llm_model = None
        self.llm_tokenizer = None
        self.device = None
        self.dtype = None
        
        if self.mode in ["llm", "auto"]:
            self._initialize_llm(model_name, model_root)

    def _initialize_llm(self, model_name: Optional[str] = None, model_root: Optional[str] = None):
        """
        Initialize Flan-T5 model with GPU support and environment-based configuration
        """
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            
            # 1️⃣ Resolve Model Name
            self.model_name = os.getenv("MODEL_NAME", model_name or "google/flan-t5-large")
            print(f"[LCExtractor] Resolved model name: {self.model_name}")

            # 2️⃣ Resolve Model Root (optional local folder)
            env_root = os.getenv("MODEL_PATH", model_root)
            if env_root and str(env_root).strip():
                self.model_root = Path(env_root)
                print(f"[LCExtractor] Using local model root: {self.model_root}")
                model_path = str(self.model_root)
            else:
                self.model_root = None
                model_path = self.model_name
                print(f"[LCExtractor] No MODEL_PATH specified — using default Hugging Face cache")

            # 3️⃣ Detect Device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            print(f"[LCExtractor] Device: {self.device} | dtype: {self.dtype}")
            
            # 4️⃣ Load Model and Tokenizer
            print(f"[LCExtractor] Loading tokenizer from {model_path}...")
            self.llm_tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            print(f"[LCExtractor] Loading model from {model_path}...")
            self.llm_model = AutoModelForSeq2SeqLM.from_pretrained(
                model_path,
                torch_dtype=self.dtype,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
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
        # Route based on mode
        if self.mode == "llm":
            return self._extract_with_llm(text)
        elif self.mode == "regex":
            return self._extract_with_regex(text)
        else:  # auto mode
            try:
                result = self._extract_with_llm(text)
                # Validate result
                if self._is_extraction_complete(result):
                    return result
                else:
                    print("[LCExtractor] LLM extraction incomplete, trying regex...")
                    return self._extract_with_regex(text)
            except Exception as e:
                print(f"[LCExtractor] LLM extraction failed: {e}, trying regex...")
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
        
        # Detect document type
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
        Extract using traditional regex patterns
        """
        print("\n[LCExtractor-Regex] Starting regex-based extraction...")

        # --- Step 0: Log raw input ---
        print("\n[LCExtractor] Starting extraction process... before PostProcessing\n", text[:500])

        # --- Step 1: Minimal OCR cleaning ---
        text_clean = sanitize_text(text)

        # --- Step 2: Fix squashed tags, spacing, and preserve line breaks ---
        text_clean = self._prepare_text_for_extraction(text_clean)

        # --- Step 3: Normalize OCR text further for detection ---
        text_norm = self._normalize_ocr_text_for_detection(text_clean)

        # --- Step 4: Log cleaned text ---
        print("\n[LCExtractor] Starting extraction process... after PostProcessing\n", text_norm[:500])

        # --- Step 5: Detect SWIFT MT message type ---
        mt_type_match = re.search(r'Message\s*type[:]*\s*(\d{3})', text_norm, re.IGNORECASE)
        mt_type = mt_type_match.group(1) if mt_type_match else None

        # --- Step 6: Detect amendment tags (26E or Number of Amendment) ---
        is_amendment = False
        if mt_type in ["707", "MT707"]:
            if re.search(r'26E|Number\s*of\s*Amendment', text_norm, re.IGNORECASE):
                is_amendment = True
                print(f"[LCExtractor] Amendment detected via 26E tag")

        # --- Step 7: Route extraction ---
        if mt_type:
            if is_amendment:
                print(f"[LCExtractor] Routing to Amendment extractor (MT{mt_type})")
                return self._extract_amendment(text_norm, mt_type)
            else:
                print(f"[LCExtractor] Routing to LC extractor (MT{mt_type})")
                return self._extract_lc(text_norm, mt_type)

        # --- Step 8: Fallback for supporting documents ---
        print(f"[LCExtractor] No MT type found - treating as supporting document")
        return self._categorize_supporting_doc(text_norm)
    
    # ========================================================================
    # LLM-BASED EXTRACTION METHODS
    # ========================================================================
    
    def _detect_document_type_llm(self, text: str) -> str:
        """Detect document type using LLM"""
        prompt = f"""Analyze this document and identify its type.

Document text:
{text[:1000]}

Question: Is this document a:
1. Letter of Credit (LC) - SWIFT MT700
2. Amendment to LC - SWIFT MT707 (look for "26E" or "Number of Amendment")
3. Supporting document (Invoice, Bill of Lading, Certificate, etc.)

Answer with only one word: LC, AMENDMENT, or SUPPORTING"""

        response = self._generate_llm(prompt, max_length=10)
        response = response.strip().upper()
        
        # Fallback regex detection
        if "AMENDMENT" in response or re.search(r'26E|Number\s*of\s*Amendment', text, re.IGNORECASE):
            return "AMENDMENT"
        elif "LC" in response or re.search(r'Message\s*type:\s*700', text, re.IGNORECASE):
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

        # --- Step 0: Log raw input ---
        print("\n[LCExtractor] Starting extraction process... before PostProcessing\n", text)

        # --- Step 1: Minimal OCR cleaning ---
        text_clean = sanitize_text(text)

        # --- Step 2: Fix squashed tags, spacing, and preserve line breaks ---
        text_clean = self._prepare_text_for_extraction(text_clean)

        # --- Step 3: Normalize OCR text further for detection ---
        text_norm = self._normalize_ocr_text_for_detection(text_clean)

        # --- Step 4: Log cleaned text ---
        print("\n[LCExtractor] Starting extraction process... after PostProcessing\n", text_norm)

        # --- Step 5: Detect SWIFT MT message type ---
        mt_type_match = re.search(r'Message\s*type[:]*\s*(\d{3})', text_norm, re.IGNORECASE)
        mt_type = mt_type_match.group(1) if mt_type_match else None

        # --- Step 6: Detect amendment tags (26E or Number of Amendment) ---
        is_amendment = False
        if mt_type in ["707", "MT707"]:
            if re.search(r'26E|Number\s*of\s*Amendment', text_norm, re.IGNORECASE):
                is_amendment = True
                print(f"[LCExtractor] Amendment detected via 26E tag")

        # --- Step 7: Route extraction ---
        if mt_type:
            if is_amendment:
                print(f"[LCExtractor] Routing to Amendment extractor (MT{mt_type})")
                return self._extract_amendment(text_norm, mt_type)
            else:
                print(f"[LCExtractor] Routing to LC extractor (MT{mt_type})")
                return self._extract_lc(text_norm, mt_type)

        # --- Step 8: Fallback for supporting documents ---
        print(f"[LCExtractor] No MT type found - treating as supporting document")
        return self._categorize_supporting_doc(text_norm)

    
    def _prepare_text_for_extraction(self, text: str) -> str:
        """
        Clean OCR text while preserving tags and line breaks for reliable amendment detection.

        Steps:
        1. Normalize common OCR spacing issues (remove extra spaces inside tags)
        2. Ensure each SWIFT tag starts on a new line
        3. Fix squashed references (e.g., "I LC 07860544623 PK" → "ILC07860544623PK")
        4. Preserve multi-line addresses and long field values
        5. Fix split LC numbers across lines
        """
        if not text:
            return ""

        # Step 1: Fix OCR splits inside known tags and references
        text = re.sub(r'\bI\s+LC\b', 'ILC', text, flags=re.IGNORECASE)
        text = re.sub(r'sw\s*if\s*t?', 'swift', text, flags=re.IGNORECASE)
        text = re.sub(r'NON\s+REF', 'NONREF', text, flags=re.IGNORECASE)
        
        # Step 1.5: Fix spaced LC numbers BEFORE line processing
        # Pattern: "ILC 07860544623 PK" or "I LC 07860544623 PK"
        # This captures: ILC + (spaces + digits/letters)+ and removes all spaces
        text = re.sub(r'(ILC)\s+(\d{11})\s+([A-Z]{2})', r'\1\2\3', text, flags=re.IGNORECASE)
        text = re.sub(r'(ILC)\s+(\d{10,}[A-Z]{0,2})', r'\1\2', text, flags=re.IGNORECASE)
        
        # Fix split LC numbers that span lines (e.g., "ILC 078605446\n23PK")
        text = re.sub(r'(ILC\s*\d+)\s*\n\s*([A-Z0-9]+)', r'\1\2', text, flags=re.IGNORECASE)
        
        # Fix dates split across lines (e.g., "2\n30509" -> "230509")
        text = re.sub(r'(\d)\s*\n\s*(\d{5})', r'\1\2', text)
        
        # Fix spaced SWIFT codes (e.g., "HAB B PK KA 786" -> "HABBPKKA786")
        text = re.sub(r'([A-Z]{3,4})\s+([A-Z]{2})\s+([A-Z]{2})\s+([A-Z0-9]{3})', r'\1\2\3\4', text)
        text = re.sub(r'([A-Z]{4})\s+([A-Z]{2})\s+([A-Z]{2})\s+([A-Z]{3})', r'\1\2\3\4', text)
        
        # Fix spaced BIC codes (e.g., "QN B AQ A QAX XX" -> "QNBAQAQAXXX")
        text = re.sub(r'([A-Z]{2})\s+([A-Z]{2})\s+([A-Z]{2})\s+([A-Z]{2})\s+([A-Z]{3})', r'\1\2\3\4\5', text)

        # Step 2: Ensure each SWIFT tag starts on a new line
        tags = ['Message type', 'To Institution', '20:', '21:', '23:', '26E:', '30:', 
                '31C:', '22A:', '46B:', '47B:', '52A:', 'Priority:', '27:', '45B:']
        for tag in tags:
            # Add newline before tag if not already at start of line
            text = re.sub(rf'(?<!\n)({re.escape(tag)})', r'\n\1', text, flags=re.IGNORECASE)

        # Step 3: Remove excessive spaces but preserve line breaks
        lines = text.splitlines()
        cleaned_lines = []
        for line in lines:
            # Remove leading/trailing spaces
            line = line.strip()
            # Collapse multiple spaces inside the line
            line = re.sub(r'\s+', ' ', line)
            if line:
                cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)


    def _normalize_ocr_text_for_detection(self, text: str) -> str:
        """
        Normalize OCR text for amendment detection WITHOUT collapsing important spaces.
        - Fixes extra spaces in numeric/letter tags (e.g., '2 6 E' -> '26E')
        - Keeps line breaks to separate tags
        - Keeps spaces after colons intact
        """
        normalized = text

        # Fix OCR-split tags: 2 6 E -> 26E, 4 7 B -> 47B, 3 1 C -> 31C
        normalized = re.sub(r'(\d)\s+([A-Z])\s*:', r'\1\2:', normalized, flags=re.IGNORECASE)

        # Fix OCR-split tags with extra spaces before colon: 26E : -> 26E:
        normalized = re.sub(r'([0-9A-Z]+)\s*:\s*', r'\1: ', normalized)

        # Remove multiple spaces, but keep line breaks
        normalized = re.sub(r'[ \t]{2,}', ' ', normalized)

        return normalized

    def _categorize_supporting_doc(self, text: str, file_name: str = "Unknown File") -> LCDocument:
        """
        Categorize non-SWIFT supporting documents (Invoice, BL, Certificates, etc.)
        """
        classification = self.detector.categorize_supporting_doc(text, file_name)
        
        # Create a unified LCDocument (NO LC NUMBER YET)
        doc = LCDocument(
            document_type=classification['type'],
            lc_number="PENDING",
            message_type="NON_SWIFT",
            raw_text=text,
            fields={},
            additional_conditions=[],
            documents_required=[]
        )

        # Attach classification metadata
        doc.is_supporting = True
        doc.file_name = file_name
        doc.classification = {
            "confidence": classification['confidence'],
            "matched_keywords": classification['matched_keywords']
        }
        doc.status = "stored_for_validation"

        return doc

    def _extract_lc(self, text: str, mt_type: Optional[str] = None) -> LCDocument:
        """Extract LC from ANY text format"""
        doc = LCDocument(
            document_type="LC",
            lc_number="",
            message_type=f"MT{mt_type}" if mt_type else "MT700",
            raw_text=text
        )
        
        # Extract LC number - MULTIPLE FORMAT PATTERNS
        lc_patterns = [
            r'20:\s*Sender\'?s\s*Reference\s*([A-Z0-9]+)',
            r'20:\s*Documentary Credit Number\s+([A-Z0-9]+)',
            r':20:\s*([A-Z0-9]+)',
            r'Documentary Credit Number\s+([A-Z0-9]+)',
            r'20:\s*([A-Z0-9]{10,})',  # Catch any long alphanumeric after 20:
        ]
        for pattern in lc_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                raw_lc = match.group(1)
                doc.lc_number = normalize_lc_number(raw_lc)
                print(f"[LC Extract] Found LC number: {doc.lc_number}")
                break
        
        # Extract issue date
        date_patterns = [
            r'31C:\s*Date of Issue\s*(\d{6})',
            r':31C:\s*(\d{6})',
            r'31C:\s*(\d{6})',
        ]
        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                doc.issue_date = match.group(1).strip()
                break
        
        # Extract sender
        sender_patterns = [
            r'52A:\s*Issuing Bank\s*([A-Z0-9]{8,11})',
            r':52A:\s*([A-Z0-9]{8,11})',
            r'52A:\s*([A-Z0-9]{8,11})',
        ]
        for pattern in sender_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                doc.sender = match.group(1).strip()
                break
        
        # Extract receiver
        receiver_patterns = [
            r'To Institution:\s*([A-Z0-9]{8,11})',
        ]
        for pattern in receiver_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                doc.receiver = match.group(1).strip()
                break
        
        # Extract all fields
        doc.fields = self.field_extractor.extract_all_fields(text)
        self.current_doc_fields = doc.fields
        
        # Extract additional conditions (47A)
        doc.additional_conditions = self.field_extractor.extract_numbered_points_robust(
            text, ['47A:', ':47A:'], doc.fields
        )
        
        # Extract documents required (46A)
        doc.documents_required = self.field_extractor.extract_numbered_points_robust(
            text, ['46A:', ':46A:'], doc.fields
        )
        
        return doc
    
    def _extract_amendment(self, text: str, mt_type: Optional[str] = None) -> LCDocument:
        """Extract amendment with 'Squash-Proof' regex for messy OCR and multi-line splits"""
        print(f"[Amendment Extract] Starting amendment extraction...")
        
        doc = LCDocument(
            document_type="AMENDMENT",
            lc_number="",
            message_type=f"MT{mt_type}" if mt_type else "MT707",
            raw_text=text
        )
        
        # 1. LC Number (Tag 20) - Handle multi-line splits AND spaces
        # First, let's do one more pass to clean up the LC number area
        lc_section = re.search(r'20:\s*Sender\'?s\s*Reference(.{0,50})', text, re.IGNORECASE | re.DOTALL)
        if lc_section:
            lc_text = lc_section.group(1)
            print(f"[Amendment Extract] LC section found: '{lc_text}'")
            # Remove ALL spaces from the LC number
            lc_cleaned = re.sub(r'\s+', '', lc_text)
            # Extract clean LC number
            lc_match = re.search(r'(ILC\d{11}[A-Z]{2})', lc_cleaned, re.IGNORECASE)
            if lc_match:
                doc.lc_number = normalize_lc_number(lc_match.group(1))
                print(f"[Amendment Extract] Found LC number: {doc.lc_number}")
        
        # Fallback patterns if the above didn't work
        if not doc.lc_number:
            lc_patterns = [
                # Try to find ILC followed by 11 digits and 2 letters (with or without spaces)
                r'20:.*?(ILC\s*\d{11}\s*[A-Z]{2})',
                r'(ILC\s*\d{11}\s*[A-Z]{2})',
                # Standard format
                r'20:\s*Sender\'?s\s*Reference\s*([A-Z0-9]+)',
                # Compact format
                r'20:\s*([A-Z]{3}\d{11,}[A-Z]{2})',
                # Flexible fallback
                r'20:.*?([A-Z]{3}\d{10,}[A-Z]*)',
            ]
            
            for pattern in lc_patterns:
                match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                if match:
                    raw_lc = match.group(1).strip()
                    # Remove all spaces from the captured LC number
                    raw_lc = re.sub(r'\s+', '', raw_lc)
                    doc.lc_number = normalize_lc_number(raw_lc)
                    print(f"[Amendment Extract] Found LC number (fallback): {doc.lc_number} (raw: {raw_lc})")
                    break
        
        if not doc.lc_number:
            print(f"[Amendment Extract] WARNING: LC number not found!")
                
        # 2. Amendment Number (Tag 26E)
        amend_patterns = [
            r'26E:\s*Number\s*of\s*Amendment\s*(\d+)',
            r'26E:\s*(\d+)',
            r'Number\s*of\s*Amendment\s*(\d+)',
        ]
        for pattern in amend_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                doc.amendment_number = match.group(1).strip()
                print(f"[Amendment Extract] Found amendment number: {doc.amendment_number}")
                break

        # 3. Amendment Date (Tag 30)
        amend_date_patterns = [
            r'30:\s*Date\s*of\s*Amendment\s*(\d{6})',
            r'30:\s*(\d{6})',
            r'Date\s*of\s*Amendment\s*(\d{6})',
        ]
        for pattern in amend_date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                doc.amendment_date = match.group(1).strip()
                print(f"[Amendment Extract] Found amendment date: {doc.amendment_date}")
                break

        # 4. Date of Issue (Tag 31C)
        issue_patterns = [
            r'31C:\s*Date\s*of\s*Issue\s*(\d{6})',
            r'31C:\s*(\d{6})',
        ]
        for pattern in issue_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                doc.issue_date = match.group(1).strip()
                print(f"[Amendment Extract] Found issue date: {doc.issue_date}")
                break

        # 5. Sender (Tag 52A) - Handle spaces in SWIFT codes
        sender_patterns = [
            r'52A:\s*Issuing\s*Bank\s*([A-Z0-9\s]{8,15})',
            r'52A:\s*([A-Z0-9\s]{8,15})',
        ]
        for pattern in sender_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                sender_raw = match.group(1).strip()
                # Remove spaces from SWIFT code
                doc.sender = re.sub(r'\s+', '', sender_raw)
                print(f"[Amendment Extract] Found sender: {doc.sender} (raw: {sender_raw})")
                break

        # 6. Receiver (To Institution) - Handle spaces in BIC codes
        receiver_patterns = [
            r'To\s*Institution:\s*([A-Z0-9\s]{8,15})',
        ]
        for pattern in receiver_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                receiver_raw = match.group(1).strip()
                # Remove spaces from BIC code
                doc.receiver = re.sub(r'\s+', '', receiver_raw)
                print(f"[Amendment Extract] Found receiver: {doc.receiver} (raw: {receiver_raw})")
                break

        # 7. Field-by-Field and Clause Extraction
        print(f"[Amendment Extract] Extracting all fields...")
        doc.fields = self.field_extractor.extract_all_fields(text)
        
        print(f"[Amendment Extract] Extracting amendment changes...")
        doc.additional_conditions = self.field_extractor.extract_amendment_changes_complete(text, ['47B'])
        doc.documents_required = self.field_extractor.extract_amendment_changes_complete(text, ['46B'])
        
        desc_changes = self.field_extractor.extract_amendment_changes_complete(text, ['45B'])
        if desc_changes:
            for change in desc_changes:
                change['field_code'] = '45B'
            doc.additional_conditions.extend(desc_changes)
        
        print(f"[Amendment Extract] Extraction complete:")
        print(f"  - LC Number: {doc.lc_number}")
        print(f"  - Amendment Number: {doc.amendment_number}")
        print(f"  - Amendment Date: {doc.amendment_date}")
        print(f"  - Sender: {doc.sender}")
        print(f"  - Receiver: {doc.receiver}")
        print(f"  - Fields extracted: {len(doc.fields)}")
        print(f"  - Additional conditions: {len(doc.additional_conditions)}")
        print(f"  - Documents required: {len(doc.documents_required)}")
        
        return doc















# """
# LC Extractor
# Main extraction logic for LC and Amendment documents
# """

# import re
# from typing import Optional, Any
# from .models import LCDocument
# from .constants import COMPREHENSIVE_FIELD_MAPPINGS, SWIFT_MESSAGE_TYPES
# from .utils import normalize_lc_number, sanitize_text
# from .field_extractor import FieldExtractor
# from .document_detector import DocumentDetector
# from services.ai_postProcessor import AIOCRPostProcessor


# class LCExtractor:
#     """Bulletproof Universal SWIFT MT Extractor with COMPLETE text extraction"""
    
#     def __init__(self):
#         self.current_doc = None
#         self.current_doc_fields = None
#         self.field_mappings = COMPREHENSIVE_FIELD_MAPPINGS
#         self.message_types = SWIFT_MESSAGE_TYPES
#         self.ocr_cleaner = AIOCRPostProcessor()
#         self.field_extractor = FieldExtractor()
#         self.detector = DocumentDetector()

#     def extract_from_text(self, text: str) -> Any:
#         """
#         Extract data or categorize supporting documents for later validation.
#         Handles LC, Amendments, and Supporting documents with robust OCR text processing.
#         """

#         # --- Step 0: Log raw input ---
#         print("\n[LCExtractor] Starting extraction process... before PostProcessing\n", text)

#         # --- Step 1: Minimal OCR cleaning ---
#         text_clean = sanitize_text(text)

#         # --- Step 2: Fix squashed tags, spacing, and preserve line breaks ---
#         text_clean = self._prepare_text_for_extraction(text_clean)

#         # --- Step 3: Normalize OCR text further for detection ---
#         text_norm = self._normalize_ocr_text_for_detection(text_clean)

#         # --- Step 4: Log cleaned text ---
#         print("\n[LCExtractor] Starting extraction process... after PostProcessing\n", text_norm)

#         # --- Step 5: Detect SWIFT MT message type ---
#         mt_type_match = re.search(r'Message\s*type[:]*\s*(\d{3})', text_norm, re.IGNORECASE)
#         mt_type = mt_type_match.group(1) if mt_type_match else None

#         # --- Step 6: Detect amendment tags (26E or Number of Amendment) ---
#         is_amendment = False
#         if mt_type in ["707", "MT707"]:
#             if re.search(r'26E|Number\s*of\s*Amendment', text_norm, re.IGNORECASE):
#                 is_amendment = True
#                 print(f"[LCExtractor] Amendment detected via 26E tag")

#         # --- Step 7: Route extraction ---
#         if mt_type:
#             if is_amendment:
#                 print(f"[LCExtractor] Routing to Amendment extractor (MT{mt_type})")
#                 return self._extract_amendment(text_norm, mt_type)
#             else:
#                 print(f"[LCExtractor] Routing to LC extractor (MT{mt_type})")
#                 return self._extract_lc(text_norm, mt_type)

#         # --- Step 8: Fallback for supporting documents ---
#         print(f"[LCExtractor] No MT type found - treating as supporting document")
#         return self._categorize_supporting_doc(text_norm)

    
#     def _prepare_text_for_extraction(self, text: str) -> str:
#         """
#         Clean OCR text while preserving tags and line breaks for reliable amendment detection.

#         Steps:
#         1. Normalize common OCR spacing issues (remove extra spaces inside tags)
#         2. Ensure each SWIFT tag starts on a new line
#         3. Fix squashed references (e.g., "I LC 07860544623 PK" → "ILC07860544623PK")
#         4. Preserve multi-line addresses and long field values
#         5. Fix split LC numbers across lines
#         """
#         if not text:
#             return ""

#         # Step 1: Fix OCR splits inside known tags and references
#         text = re.sub(r'\bI\s*LC\b', 'ILC', text, flags=re.IGNORECASE)
#         text = re.sub(r'sw\s*if\s*t?', 'swift', text, flags=re.IGNORECASE)
#         text = re.sub(r'NON\s*REF', 'NONREF', text, flags=re.IGNORECASE)
        
#         # Fix split LC numbers that span lines (e.g., "ILC 078605446\n23PK")
#         # This pattern looks for ILC followed by digits, then captures trailing digits/letters on next line
#         text = re.sub(r'(ILC\s*\d+)\s*\n\s*([A-Z0-9]+)', r'\1\2', text, flags=re.IGNORECASE)
        
#         # Fix dates split across lines (e.g., "2\n30509" -> "230509")
#         text = re.sub(r'(\d)\s*\n\s*(\d{5})', r'\1\2', text)

#         # Step 2: Ensure each SWIFT tag starts on a new line
#         tags = ['Message type', 'To Institution', '20:', '21:', '23:', '26E:', '30:', 
#                 '31C:', '22A:', '46B:', '47B:', '52A:', 'Priority:', '27:', '45B:']
#         for tag in tags:
#             # Add newline before tag if not already at start of line
#             text = re.sub(rf'(?<!\n)({re.escape(tag)})', r'\n\1', text, flags=re.IGNORECASE)

#         # Step 3: Remove excessive spaces but preserve line breaks
#         lines = text.splitlines()
#         cleaned_lines = []
#         for line in lines:
#             # Remove leading/trailing spaces
#             line = line.strip()
#             # Collapse multiple spaces inside the line
#             line = re.sub(r'\s+', ' ', line)
#             if line:
#                 cleaned_lines.append(line)

#         return '\n'.join(cleaned_lines)


#     def _normalize_ocr_text_for_detection(self, text: str) -> str:
#         """
#         Normalize OCR text for amendment detection WITHOUT collapsing important spaces.
#         - Fixes extra spaces in numeric/letter tags (e.g., '2 6 E' -> '26E')
#         - Keeps line breaks to separate tags
#         - Keeps spaces after colons intact
#         """
#         normalized = text

#         # Fix OCR-split tags: 2 6 E -> 26E, 4 7 B -> 47B, 3 1 C -> 31C
#         normalized = re.sub(r'(\d)\s+([A-Z])\s*:', r'\1\2:', normalized, flags=re.IGNORECASE)

#         # Fix OCR-split tags with extra spaces before colon: 26E : -> 26E:
#         normalized = re.sub(r'([0-9A-Z]+)\s*:\s*', r'\1: ', normalized)

#         # Remove multiple spaces, but keep line breaks
#         normalized = re.sub(r'[ \t]{2,}', ' ', normalized)

#         return normalized

#     def _categorize_supporting_doc(self, text: str, file_name: str = "Unknown File") -> LCDocument:
#         """
#         Categorize non-SWIFT supporting documents (Invoice, BL, Certificates, etc.)
#         """
#         classification = self.detector.categorize_supporting_doc(text, file_name)
        
#         # Create a unified LCDocument (NO LC NUMBER YET)
#         doc = LCDocument(
#             document_type=classification['type'],
#             lc_number="PENDING",
#             message_type="NON_SWIFT",
#             raw_text=text,
#             fields={},
#             additional_conditions=[],
#             documents_required=[]
#         )

#         # Attach classification metadata
#         doc.is_supporting = True
#         doc.file_name = file_name
#         doc.classification = {
#             "confidence": classification['confidence'],
#             "matched_keywords": classification['matched_keywords']
#         }
#         doc.status = "stored_for_validation"

#         return doc

#     def _extract_lc(self, text: str, mt_type: Optional[str] = None) -> LCDocument:
#         """Extract LC from ANY text format"""
#         doc = LCDocument(
#             document_type="LC",
#             lc_number="",
#             message_type=f"MT{mt_type}" if mt_type else "MT700",
#             raw_text=text
#         )
        
#         # Extract LC number - MULTIPLE FORMAT PATTERNS
#         lc_patterns = [
#             r'20:\s*Sender\'?s\s*Reference\s*([A-Z0-9]+)',
#             r'20:\s*Documentary Credit Number\s+([A-Z0-9]+)',
#             r':20:\s*([A-Z0-9]+)',
#             r'Documentary Credit Number\s+([A-Z0-9]+)',
#             r'20:\s*([A-Z0-9]{10,})',  # Catch any long alphanumeric after 20:
#         ]
#         for pattern in lc_patterns:
#             match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
#             if match:
#                 raw_lc = match.group(1)
#                 doc.lc_number = normalize_lc_number(raw_lc)
#                 print(f"[LC Extract] Found LC number: {doc.lc_number}")
#                 break
        
#         # Extract issue date
#         date_patterns = [
#             r'31C:\s*Date of Issue\s*(\d{6})',
#             r':31C:\s*(\d{6})',
#             r'31C:\s*(\d{6})',
#         ]
#         for pattern in date_patterns:
#             match = re.search(pattern, text, re.IGNORECASE)
#             if match:
#                 doc.issue_date = match.group(1).strip()
#                 break
        
#         # Extract sender
#         sender_patterns = [
#             r'52A:\s*Issuing Bank\s*([A-Z0-9]{8,11})',
#             r':52A:\s*([A-Z0-9]{8,11})',
#             r'52A:\s*([A-Z0-9]{8,11})',
#         ]
#         for pattern in sender_patterns:
#             match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
#             if match:
#                 doc.sender = match.group(1).strip()
#                 break
        
#         # Extract receiver
#         receiver_patterns = [
#             r'To Institution:\s*([A-Z0-9]{8,11})',
#         ]
#         for pattern in receiver_patterns:
#             match = re.search(pattern, text, re.IGNORECASE)
#             if match:
#                 doc.receiver = match.group(1).strip()
#                 break
        
#         # Extract all fields
#         doc.fields = self.field_extractor.extract_all_fields(text)
#         self.current_doc_fields = doc.fields
        
#         # Extract additional conditions (47A)
#         doc.additional_conditions = self.field_extractor.extract_numbered_points_robust(
#             text, ['47A:', ':47A:'], doc.fields
#         )
        
#         # Extract documents required (46A)
#         doc.documents_required = self.field_extractor.extract_numbered_points_robust(
#             text, ['46A:', ':46A:'], doc.fields
#         )
        
#         return doc
    
#     def _extract_amendment(self, text: str, mt_type: Optional[str] = None) -> LCDocument:
#         """Extract amendment with 'Squash-Proof' regex for messy OCR and multi-line splits"""
#         print(f"[Amendment Extract] Starting amendment extraction...")
        
#         doc = LCDocument(
#             document_type="AMENDMENT",
#             lc_number="",
#             message_type=f"MT{mt_type}" if mt_type else "MT707",
#             raw_text=text
#         )
        
#         # 1. LC Number (Tag 20) - Handle multi-line splits
#         # Pattern now captures LC numbers that might be split across lines
#         lc_patterns = [
#             # Standard format
#             r'20:\s*Sender\'?s\s*Reference\s*([A-Z0-9]+)',
#             # Compact format
#             r'20:\s*([A-Z]{3}\d{11,}[A-Z]{2})',  # ILC followed by 11+ digits then 2 letters
#             # Flexible fallback
#             r'20:.*?([A-Z]{3}\d{10,}[A-Z]*)',
#         ]
        
#         for pattern in lc_patterns:
#             match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
#             if match:
#                 raw_lc = match.group(1).strip()
#                 doc.lc_number = normalize_lc_number(raw_lc)
#                 print(f"[Amendment Extract] Found LC number: {doc.lc_number} (raw: {raw_lc})")
#                 break
        
#         if not doc.lc_number:
#             print(f"[Amendment Extract] WARNING: LC number not found!")
                
#         # 2. Amendment Number (Tag 26E)
#         amend_patterns = [
#             r'26E:\s*Number\s*of\s*Amendment\s*(\d+)',
#             r'26E:\s*(\d+)',
#             r'Number\s*of\s*Amendment\s*(\d+)',
#         ]
#         for pattern in amend_patterns:
#             match = re.search(pattern, text, re.IGNORECASE)
#             if match:
#                 doc.amendment_number = match.group(1).strip()
#                 print(f"[Amendment Extract] Found amendment number: {doc.amendment_number}")
#                 break

#         # 3. Amendment Date (Tag 30)
#         amend_date_patterns = [
#             r'30:\s*Date\s*of\s*Amendment\s*(\d{6})',
#             r'30:\s*(\d{6})',
#             r'Date\s*of\s*Amendment\s*(\d{6})',
#         ]
#         for pattern in amend_date_patterns:
#             match = re.search(pattern, text, re.IGNORECASE)
#             if match:
#                 doc.amendment_date = match.group(1).strip()
#                 print(f"[Amendment Extract] Found amendment date: {doc.amendment_date}")
#                 break

#         # 4. Date of Issue (Tag 31C)
#         issue_patterns = [
#             r'31C:\s*Date\s*of\s*Issue\s*(\d{6})',
#             r'31C:\s*(\d{6})',
#         ]
#         for pattern in issue_patterns:
#             match = re.search(pattern, text, re.IGNORECASE)
#             if match:
#                 doc.issue_date = match.group(1).strip()
#                 print(f"[Amendment Extract] Found issue date: {doc.issue_date}")
#                 break

#         # 5. Sender (Tag 52A)
#         sender_patterns = [
#             r'52A:\s*Issuing\s*Bank\s*([A-Z0-9]{8,11})',
#             r'52A:\s*([A-Z0-9]{8,11})',
#         ]
#         for pattern in sender_patterns:
#             match = re.search(pattern, text, re.IGNORECASE)
#             if match:
#                 doc.sender = match.group(1).strip()
#                 print(f"[Amendment Extract] Found sender: {doc.sender}")
#                 break

#         # 6. Receiver (To Institution)
#         receiver_patterns = [
#             r'To\s*Institution:\s*([A-Z0-9]{8,11})',
#         ]
#         for pattern in receiver_patterns:
#             match = re.search(pattern, text, re.IGNORECASE)
#             if match:
#                 doc.receiver = match.group(1).strip()
#                 print(f"[Amendment Extract] Found receiver: {doc.receiver}")
#                 break

#         # 7. Field-by-Field and Clause Extraction
#         print(f"[Amendment Extract] Extracting all fields...")
#         doc.fields = self.field_extractor.extract_all_fields(text)
        
#         print(f"[Amendment Extract] Extracting amendment changes...")
#         doc.additional_conditions = self.field_extractor.extract_amendment_changes_complete(text, ['47B'])
#         doc.documents_required = self.field_extractor.extract_amendment_changes_complete(text, ['46B'])
        
#         desc_changes = self.field_extractor.extract_amendment_changes_complete(text, ['45B'])
#         if desc_changes:
#             for change in desc_changes:
#                 change['field_code'] = '45B'
#             doc.additional_conditions.extend(desc_changes)
        
#         print(f"[Amendment Extract] Extraction complete:")
#         print(f"  - LC Number: {doc.lc_number}")
#         print(f"  - Amendment Number: {doc.amendment_number}")
#         print(f"  - Amendment Date: {doc.amendment_date}")
#         print(f"  - Sender: {doc.sender}")
#         print(f"  - Receiver: {doc.receiver}")
#         print(f"  - Fields extracted: {len(doc.fields)}")
#         print(f"  - Additional conditions: {len(doc.additional_conditions)}")
#         print(f"  - Documents required: {len(doc.documents_required)}")
        
#         return doc