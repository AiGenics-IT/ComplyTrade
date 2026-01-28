"""
OCR Post-Processor for Banking Documents
Fixes squashed text, missing spaces, and improves readability
Specialized for LC (Letter of Credit) and SWIFT messages
"""

import re
from typing import List, Tuple, Dict, Optional
import string


class OCRPostProcessor:
    """
    Advanced post-processing for OCR text to fix common issues:
    - Squashed keywords (CLAUSENO.27TOREADAS)
    - Missing spaces between words
    - Merged numeric/alpha characters
    - Banking-specific terminology
    """
    
    def __init__(self):
        # Banking and LC-specific keywords that need spacing
        self.banking_keywords = [
            'CLAUSE', 'FIELD', 'ITEM', 'DOCUMENT', 'LETTER', 'AUTHORITY',
            'INSTEAD', 'REPLACE', 'DELETE', 'READ', 'NOW', 'ACCEPTABLE',
            'REQUIRED', 'AGAINST', 'UNDER', 'BILL', 'LADING', 'SIGNED',
            'ORIGINAL', 'COPY', 'CERTIFICATE', 'INVOICE', 'COMPLY',
            'EVIDENCE', 'SHIPMENT', 'PAYMENT', 'PRESENTATION', 'DAYS',
            'INSURANCE', 'VESSEL', 'PORT', 'DATE', 'MASTER', 'AGENT'
        ]
        
        # Common squashed patterns in banking documents
        self.squashed_patterns = {
            'CLAUSENO': 'CLAUSE NO',
            'CLAUENO': 'CLAUSE NO',
            'CLAUSENUMBER': 'CLAUSE NUMBER',
            'FIELDNO': 'FIELD NO',
            'ITEMNO': 'ITEM NO',
            'TOREADAS': 'TO READ AS',
            'TOREAD': 'TO READ',
            'NOWTOREAD': 'NOW TO READ',
            'INSTEADOF': 'INSTEAD OF',
            'REPLACEBY': 'REPLACE BY',
            'REPLACEWITH': 'REPLACE WITH',
            'DELETEREPLACE': 'DELETE REPLACE',
            'BILLOF': 'BILL OF',
            'BILLOFLADING': 'BILL OF LADING',
            'LETTEROF': 'LETTER OF',
            'LETTEROFCREDIT': 'LETTER OF CREDIT',
            'L/CCLAUSE': 'L/C CLAUSE',
            'LCCLAUSE': 'LC CLAUSE',
            'ACCEPTABLETO': 'ACCEPTABLE TO',
            'REQUIREDAGAINST': 'REQUIRED AGAINST',
            'EVIDENCEWILL': 'EVIDENCE WILL',
            'WILLBE': 'WILL BE',
            'MUSTBE': 'MUST BE',
            'SHALLBE': 'SHALL BE',
            'ANDOR': 'AND/OR',
            'AND/ORTHE': 'AND/OR THE',
            'FORSIGNING': 'FOR SIGNING',
            'ISACCEPTABLE': 'IS ACCEPTABLE',
            'TOCOMPLY': 'TO COMPLY',
            'COMPLYL/C': 'COMPLY L/C',
            'UNDERL/C': 'UNDER L/C',
            'DOCUMENT ARY': 'DOCUMENTARY',  # Fix over-splitting
        }
        
        # Prefixes that should be separated
        self.separable_prefixes = ['UN', 'NON', 'PRE', 'POST', 'SUB', 'INTER']
        
    def process_text(self, text: str, aggressive: bool = False) -> str:
        """
        Main post-processing pipeline
        
        Args:
            text: Raw OCR text
            aggressive: Use more aggressive spacing (may over-correct)
        
        Returns:
            Cleaned and properly spaced text
        """
        if not text:
            return text
        
        # Step 1: Fix known squashed patterns
        text = self._fix_squashed_patterns(text)
        
        # Step 2: Add spaces between keywords
        text = self._separate_keywords(text)
        
        # Step 3: Fix number-letter boundaries
        text = self._separate_numbers_letters(text)
        
        # Step 4: Fix punctuation spacing
        text = self._fix_punctuation_spacing(text)
        
        # Step 5: Handle camelCase splits (if aggressive)
        if aggressive:
            text = self._split_camel_case(text)
        
        # Step 6: Clean up extra spaces
        text = self._normalize_whitespace(text)
        
        # Step 7: Fix common OCR mistakes
        text = self._fix_common_ocr_errors(text)
        
        return text
    
    def _fix_squashed_patterns(self, text: str) -> str:
        """Replace known squashed patterns with properly spaced versions"""
        for squashed, proper in self.squashed_patterns.items():
            # Case-insensitive replacement
            text = re.sub(
                re.escape(squashed), 
                proper, 
                text, 
                flags=re.IGNORECASE
            )
        return text
    
    def _separate_keywords(self, text: str) -> str:
        """
        Add spaces around banking keywords that are stuck to other text
        Example: "CLAUSEFIELD" -> "CLAUSE FIELD"
        """
        for keyword in self.banking_keywords:
            # Add space after keyword if followed by uppercase letter or lowercase
            pattern = f'({keyword})([A-Z]{{2,}}|[a-z]+)'
            text = re.sub(pattern, r'\1 \2', text, flags=re.IGNORECASE)
            
            # Add space before keyword if preceded by uppercase letter or lowercase
            pattern = f'([A-Z]{{2,}}|[a-z]+)({keyword})'
            text = re.sub(pattern, r'\1 \2', text, flags=re.IGNORECASE)
            
            # Special case: separate within camelCase
            pattern = f'([a-z])({keyword})'
            text = re.sub(pattern, r'\1 \2', text, flags=re.IGNORECASE)
        
        return text
    
    def _separate_numbers_letters(self, text: str) -> str:
        """
        Add spaces between numbers and letters
        Example: "CLAUSE27" -> "CLAUSE 27", "46A" -> "46A" (keep)
        """
        # Separate letter followed by number (CLAUSE27 -> CLAUSE 27)
        text = re.sub(r'([A-Z]{2,})(\d+)', r'\1 \2', text)
        
        # Separate number followed by multiple letters (27CLAUSE -> 27 CLAUSE)
        # But keep single letter suffixes (46A stays 46A)
        text = re.sub(r'(\d+)([A-Z]{2,})', r'\1 \2', text)
        
        # Separate if there's a period or dot
        text = re.sub(r'(NO\.)(\d+)', r'\1 \2', text, flags=re.IGNORECASE)
        
        return text
    
    def _fix_punctuation_spacing(self, text: str) -> str:
        """
        Fix spacing around punctuation
        """
        # Add space after commas, periods, colons if not followed by space
        text = re.sub(r'([.,;:])([A-Z])', r'\1 \2', text)
        
        # Fix "NO." followed by digit
        text = re.sub(r'(NO\.)(\d)', r'\1 \2', text, flags=re.IGNORECASE)
        
        # Fix apostrophes/quotes stuck to words
        text = re.sub(r"([A-Z])'([A-Z])", r"\1' \2", text)
        
        return text
    
    def _split_camel_case(self, text: str) -> str:
        """
        Split camelCase words (aggressive mode only)
        Example: "BillOfLading" -> "Bill Of Lading"
        """
        # Insert space before uppercase letters that follow lowercase
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """
        Clean up multiple spaces and normalize whitespace
        """
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        
        # Fix spaces around punctuation
        text = re.sub(r'\s+([.,;:])', r'\1', text)
        
        # Ensure single space after punctuation
        text = re.sub(r'([.,;:])([^\s\d])', r'\1 \2', text)
        
        # Clean up line breaks
        text = re.sub(r'\n\s+\n', '\n\n', text)
        
        return text.strip()
    
    def _fix_common_ocr_errors(self, text: str) -> str:
        """
        Fix common OCR character recognition errors
        """
        ocr_fixes = {
            'l/C': 'L/C',
            'l/c': 'L/C',
            '0F': 'OF',
            'TH E': 'THE',
            'AN D': 'AND',
            'W ITH': 'WITH',
            'REQU IRED': 'REQUIRED',
        }
        
        for wrong, correct in ocr_fixes.items():
            text = text.replace(wrong, correct)
        
        return text
    
    def process_clause_text(self, text: str) -> str:
        """
        Specialized processing for LC clause text
        Handles patterns like: CLAUSENO.27TOREADAS'...'
        """
        # Fix the specific pattern in your example
        # CLAUSENO.27TOREADAS'X' -> CLAUSE NO.27 TO READ AS 'X'
        
        # Pattern 1: CLAUSENO.XX... -> CLAUSE NO.XX ...
        text = re.sub(
            r'(CLAU[SE]*NO[\.]?)(\d+)([A-Z])',
            r'CLAUSE NO.\2 \3',
            text,
            flags=re.IGNORECASE
        )
        
        # Pattern 2: Add space before 'TO READ AS'
        text = re.sub(
            r'([A-Z\d])TOREADAS',
            r'\1 TO READ AS ',
            text,
            flags=re.IGNORECASE
        )
        
        # Pattern 3: Add space before quotes
        text = re.sub(r"([A-Z])'", r"\1 '", text)
        
        # Pattern 4: Separate FIELD46A -> FIELD 46A
        text = re.sub(r'FIELD(\d+[A-Z])', r'FIELD \1', text, flags=re.IGNORECASE)
        
        return self.process_text(text)
    
    def extract_and_clean_clauses(self, text: str) -> List[Dict[str, str]]:
        """
        Extract individual clauses from text and clean each one
        Returns list of cleaned clauses with numbers
        """
        clauses = []
        
        # Pattern to find clauses: CLAUSE NO.XX or CLAUSENO.XX
        pattern = r'(?:CLAUSE|ITEM)\s*(?:NO\.?)?\s*(\d+)[\s:.]*(.*?)(?=(?:CLAUSE|ITEM)\s*(?:NO\.?)?\s*\d+|$)'
        
        matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
        
        for match in matches:
            clause_num = match.group(1)
            clause_text = match.group(2).strip()
            
            # Clean the clause text
            cleaned_text = self.process_clause_text(clause_text)
            
            clauses.append({
                'number': int(clause_num),
                'text': cleaned_text,
                'original': clause_text
            })
        
        return clauses


class BankingDocumentCleaner:
    """
    High-level cleaner for banking documents combining OCR and post-processing
    """
    
    def __init__(self):
        self.postprocessor = OCRPostProcessor()
    
    def clean_ocr_output(self, raw_text: str, document_type: str = 'LC') -> str:
        """
        Clean OCR output based on document type
        
        Args:
            raw_text: Raw OCR text
            document_type: 'LC', 'SWIFT', 'AMENDMENT', 'INVOICE', etc.
        
        Returns:
            Cleaned text
        """
        if document_type in ['LC', 'AMENDMENT']:
            # Use clause-specific processing
            return self.postprocessor.process_clause_text(raw_text)
        else:
            # Use general processing
            return self.postprocessor.process_text(raw_text)
    
    def extract_structured_data(self, raw_text: str) -> Dict:
        """
        Extract structured data from banking document
        """
        cleaned_text = self.postprocessor.process_text(raw_text)
        
        # Extract clauses
        clauses = self.postprocessor.extract_and_clean_clauses(cleaned_text)
        
        return {
            'raw_text': raw_text,
            'cleaned_text': cleaned_text,
            'clauses': clauses,
            'clause_count': len(clauses)
        }


def test_postprocessor():
    """Test the post-processor with your example"""
    
    # Your problematic example
    raw_text = "CLAUENO.27TOREADAS'LETTEROFAUTHORITYFORSIGNING BILLOFLADINGISACCEPTABLETOCOMPLYL/CCLAUSEFIELD46A-3.NO FURTHERDOCUMENTARYEVIDENCEWILLBEREQUIREDAGAINST'LETTEROF AUTHORITYFORSIGNINGBILLOFLADING'UNDERL/CCLAUSEFIELD 46A-3'"
    
    print("="*80)
    print("OCR POST-PROCESSOR TEST")
    print("="*80)
    
    print("\n--- ORIGINAL (Raw OCR) ---")
    print(raw_text)
    
    processor = OCRPostProcessor()
    
    # Test 1: General processing
    cleaned = processor.process_text(raw_text)
    print("\n--- CLEANED (General) ---")
    print(cleaned)
    
    # Test 2: Clause-specific processing
    clause_cleaned = processor.process_clause_text(raw_text)
    print("\n--- CLEANED (Clause-Specific) ---")
    print(clause_cleaned)
    
    # Test 3: Extract clauses
    clauses = processor.extract_and_clean_clauses(raw_text)
    print("\n--- EXTRACTED CLAUSES ---")
    for clause in clauses:
        print(f"Clause {clause['number']}: {clause['text'][:100]}...")
    
    print("\n" + "="*80)
    
    # More examples
    test_cases = [
        "CLAUSENO.5DELETE'30DAYS'REPLACEBY'45DAYS'",
        "FIELDNO.46AINSTEADOF'TOBENOMINATED'",
        "BILLOFLADINGISREQUIREDAGAINSTPAYMENT",
        "LETTEROFAUTHORITYFORSIGNING",
    ]
    
    print("\n--- ADDITIONAL TEST CASES ---")
    for test in test_cases:
        cleaned = processor.process_clause_text(test)
        print(f"\nOriginal: {test}")
        print(f"Cleaned:  {cleaned}")


if __name__ == "__main__":
    test_postprocessor()