"""
Robust LC Consolidator for GOT-OCR Integration
Handles misclassified documents and validates LC/Amendment data
"""

import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of document validation"""
    is_valid: bool
    reason: str = ""
    extracted_lc_number: Optional[str] = None


class LCConsolidatorGOT:
    """
    Robust LC Consolidator that validates and filters documents
    before processing
    """
    
    # Required fields for LC validation
    LC_REQUIRED_FIELDS = {
        'DC_Number', 'Reference', 'Issuing_Bank', 'Date_of_Issue', 'Expiry_Date'
    }

    # Required fields for Amendment validation
    AMENDMENT_REQUIRED_FIELDS = {
        'DC_Number', 'Reference', 'Amendment_Number', 'Amendment_Date', 'Issuing_Bank'
    }

    # Fields that indicate actual LC content
    LC_INDICATOR_FIELDS = {
        'Form_of_Credit', 'Applicant', 'Beneficiary', 'Amount',
        'Documents_Required', 'Additional_Conditions',
        'Payment_Terms', 'Shipment_Terms', 'Partial_Shipment_Allowed'
    }

    # Fields that indicate actual Amendment content
    AMENDMENT_INDICATOR_FIELDS = {
        'Additional_Conditions_2', 'Field_46B', 'Description_of_Goods_Additional',
        'Amount', 'Expiry_Date', 'Shipment_Terms'
    }
    
    def __init__(self, use_ai=True):
        self.lcs: Dict[str, Dict] = {}
        self.amendments: Dict[str, List[Dict]] = {}
        self.unidentified: List[Dict] = []
        self.validation_log: List[str] = []
        self.use_ai = use_ai
        
        if use_ai:
            try:
                from services.ai_auditor import OfflineLCAuditor
                self.auditor = OfflineLCAuditor()
                print("✓ AI Auditor initialized")
            except:
                self.use_ai = False
                print("⚠ AI Auditor not available")
    
    def validate_lc(self, data: Dict) -> ValidationResult:
        """
        Validate that a document marked as LC is actually an LC
        
        Returns:
            ValidationResult with is_valid flag and reason
        """
        # Check if data is empty or just noise
        if not data or len(data) < 3:
            return ValidationResult(
                is_valid=False,
                reason="Insufficient data fields"
            )
        
        # Extract potential LC number first
        lc_number = self._extract_lc_number(data)
        
        # Check for required fields
        has_required = any(field in data for field in self.LC_REQUIRED_FIELDS)
        if not has_required:
            return ValidationResult(
                is_valid=False,
                reason="Missing required LC fields (DC_Number, Reference, etc.)"
            )
        
        # Check for LC indicator fields (content that makes it an LC)
        has_indicators = any(field in data for field in self.LC_INDICATOR_FIELDS)
        if not has_indicators:
            return ValidationResult(
                is_valid=False,
                reason="Missing LC content indicators (Applicant, Beneficiary, Amount, etc.)"
            )
        
        # Check if LC number is valid
        if not lc_number or lc_number == "UNKNOWN":
            return ValidationResult(
                is_valid=False,
                reason="Could not extract valid LC number"
            )
        
        # Check if data looks like shore tank measurements or other non-LC content
        if self._looks_like_non_lc_content(data):
            return ValidationResult(
                is_valid=False,
                reason="Content appears to be non-LC data (measurements, tables, etc.)"
            )
        
        return ValidationResult(
            is_valid=True,
            extracted_lc_number=lc_number
        )
    
    def validate_amendment(self, data: Dict) -> ValidationResult:
        """
        Validate that a document marked as Amendment is actually an Amendment
        """
        # Check if data is empty or just noise
        if not data or len(data) < 3:
            return ValidationResult(
                is_valid=False,
                reason="Insufficient data fields"
            )
        
        # Extract potential LC number
        lc_number = self._extract_lc_number(data)
        
        # Check for required amendment fields
        has_required = any(field in data for field in self.AMENDMENT_REQUIRED_FIELDS)
        if not has_required:
            return ValidationResult(
                is_valid=False,
                reason="Missing required Amendment fields"
            )
        
        # Check for amendment indicator fields (actual changes/instructions)
        has_indicators = any(field in data for field in self.AMENDMENT_INDICATOR_FIELDS)
        if not has_indicators:
            return ValidationResult(
                is_valid=False,
                reason="Missing Amendment change instructions"
            )
        
        # Check amendment number
        amend_num = data.get('Amendment_Number', '')
        if not amend_num or not re.search(r'\d+', amend_num):
            return ValidationResult(
                is_valid=False,
                reason="Missing or invalid Amendment Number"
            )
        
        # Check if LC number is valid
        if not lc_number or lc_number == "UNKNOWN":
            return ValidationResult(
                is_valid=False,
                reason="Could not extract valid LC number from amendment"
            )
        
        return ValidationResult(
            is_valid=True,
            extracted_lc_number=lc_number
        )
    
    def _looks_like_non_lc_content(self, data: Dict) -> bool:
        """
        Check if the data looks like non-LC content
        (e.g., shore tank measurements, tables, etc.)
        """
        # Check if most fields are technical/measurement related
        technical_keywords = [
            'tank', 'volume', 'metric', 'bbs', 'temperature',
            'gauge', 'shore', 'measurement', 'calibration',
            'mm', 'ft-ins', 'gross observed', 'vcf', 'wcf'
        ]
        
        # Count how many fields contain technical keywords
        technical_count = 0
        total_fields = 0
        
        for key, value in data.items():
            total_fields += 1
            key_lower = key.lower()
            value_lower = str(value).lower()
            
            if any(kw in key_lower or kw in value_lower for kw in technical_keywords):
                technical_count += 1
        
        # If more than 50% of fields are technical, it's probably not an LC
        if total_fields > 0 and (technical_count / total_fields) > 0.5:
            return True
        
        # Check if Reference field contains only numbers/table data
        reference = str(data.get('Reference', ''))
        if reference and re.match(r'^[\d\s\.\,\(\)]+$', reference):
            return True
        
        return False
    
    def _extract_lc_number(self, data: Dict) -> str:
        """Extract LC number from various fields with improved pattern matching"""
        # Try different field names
        possible_fields = [
            'DC_Number',
            'Reference',
            "Sender's Reference",
            "Issuing Bank's Reference"
        ]
        
        for field in possible_fields:
            value = data.get(field, '')
            if value:
                # Pattern: ILC followed by digits and optional country code
                # Example: ILC07860544623PK
                match = re.search(r'\b(ILC\d{10,}[A-Z]{0,2})\b', value)
                if match:
                    return match.group(1)
                
                # Alternative patterns
                match = re.search(r'\b([A-Z]{2,4}\d{10,}[A-Z]{0,2})\b', value)
                if match:
                    lc_num = match.group(1)
                    # Verify it's not just random alphanumeric
                    if len(lc_num) >= 12:
                        return lc_num
        
        return "UNKNOWN"
    
    def _extract_amendment_number(self, text: str) -> str:
        """Extract amendment number from text"""
        # Handle various formats including symbols like ∅
        text = text.replace('∅', '0')
        text = text.replace('\\varnothing', '0')
        
        # Extract number
        match = re.search(r'(\d+)', text)
        if match:
            return match.group(1).zfill(2)
        return "00"
    
    def add_document(self, doc_entry: Dict[str, Any]):
        """
        Add document with validation
        
        doc_entry structure:
        {
            'object_type': 'lc' | 'amendment',
            'page_reference': '1' | '2-4',
            'page_count': 1,
            'data': {
                'DC_Number': ...,
                ... other fields ...
            }
        }
        """
        obj_type = doc_entry.get('object_type', '').lower()
        data = doc_entry.get('data', {})
        page_ref = doc_entry.get('page_reference', '?')
        
        print(f"\n--- Processing {obj_type.upper()} on page {page_ref} ---")
        
        # Validate based on type
        if obj_type == 'lc':
            validation = self.validate_lc(data)
            
            if not validation.is_valid:
                log_msg = f"⚠ LC validation failed on page {page_ref}: {validation.reason}"
                print(f"  {log_msg}")
                self.validation_log.append(log_msg)
                
                # Reclassify as unidentified
                doc_entry['object_type'] = 'unidentified'
                doc_entry['original_type'] = 'lc'
                doc_entry['validation_failure'] = validation.reason
                self.unidentified.append(doc_entry)
                return
            
            lc_number = validation.extracted_lc_number
            self.lcs[lc_number] = self._map_lc_data(lc_number, data, page_ref)
            print(f"  ✓ Valid LC: {lc_number}")
        
        elif obj_type == 'amendment':
            validation = self.validate_amendment(data)
            
            if not validation.is_valid:
                log_msg = f"⚠ Amendment validation failed on page {page_ref}: {validation.reason}"
                print(f"  {log_msg}")
                self.validation_log.append(log_msg)
                
                # Reclassify as unidentified
                doc_entry['object_type'] = 'unidentified'
                doc_entry['original_type'] = 'amendment'
                doc_entry['validation_failure'] = validation.reason
                self.unidentified.append(doc_entry)
                return
            
            lc_number = validation.extracted_lc_number
            if lc_number not in self.amendments:
                self.amendments[lc_number] = []
            
            amendment = self._map_amendment_data(lc_number, data, page_ref)
            self.amendments[lc_number].append(amendment)
            amend_num = amendment.get('amendment_number', '?')
            print(f"  ✓ Valid Amendment {amend_num} for LC: {lc_number}")
        
        else:
            # Already marked as unidentified or other type
            self.unidentified.append(doc_entry)
            print(f"  ℹ Stored as {obj_type}")
    
    def _map_lc_data(self, lc_number: str, data: Dict, page_ref: str) -> Dict:
        """Map GOT-OCR LC data to internal format with robust parsing"""
        print(f"  [DEBUG] Mapping LC data for {lc_number}")
        
        # Parse conditions and documents with fallback
        additional_conditions = self._parse_conditions(
            data.get('Additional_Conditions', '')
        ) or []
        
        documents_required = self._parse_documents(
            data.get('Documents_Required', '')
        ) or []
        
        print(f"    ✓ Parsed: {len(additional_conditions)} conditions, {len(documents_required)} documents")
        
        return {
            'lc_number': lc_number,
            'page_reference': page_ref,
            'issue_date': self._extract_date(data.get('Date_of_Issue', '')),
            'sender': data.get('Issuing_Bank', data.get('Field_52A', '')),
            'receiver': data.get('Applicant', ''),
            'beneficiary': data.get('Beneficiary', ''),
            'amount': data.get('Amount', ''),
            'fields': data,
            'additional_conditions': additional_conditions,
            'documents_required': documents_required
        }
    
    def _map_amendment_data(self, lc_number: str, data: Dict, page_ref: str) -> Dict:
        """Map GOT-OCR amendment data to internal format"""
        amend_num_raw = data.get('Amendment_Number', '0')
        amend_num = self._extract_amendment_number(amend_num_raw)
        
        return {
            'lc_number': lc_number,
            'page_reference': page_ref,
            'amendment_number': amend_num,
            'amendment_date': self._extract_date(data.get('Amendment_Date', '')),
            'changes': {
                'additional_conditions': self._parse_amendment_changes(
                    data.get('Additional_Conditions_2', '')
                ),
                'documents_required': self._parse_amendment_changes(
                    data.get('Field_46B', '')
                ),
                'description_of_goods': self._parse_amendment_changes(
                    data.get('Description_of_Goods_Additional', '')
                )
            },
            'raw_data': data
        }
    
    def _extract_date(self, text: str) -> str:
        """Extract date from text (format: YYMMDD)"""
        match = re.search(r'(\d{6})', text)
        return match.group(1) if match else ''
    
    def _parse_conditions(self, text: str) -> List[Dict]:
        """Parse conditions into numbered points with robust handling"""
        if not text or len(text.strip()) < 10:
            return []
        
        # Clean the text
        text = text.strip()
        text = re.sub(r'^Additional Conditions\s*', '', text, flags=re.IGNORECASE)
        
        points = []
        
        # Try to split by numbered points
        # Patterns: (1), 1., 1), [1]
        parts = re.split(r'\s*[\(\[]?(\d+)[\)\]\.]\s+', text)
        
        if len(parts) > 2:  # Successfully split
            for i in range(1, len(parts), 2):
                if i + 1 < len(parts):
                    point_num = int(parts[i])
                    point_text = parts[i + 1].strip()
                    
                    if point_text and len(point_text) > 5:
                        points.append({
                            'point_number': point_num,
                            'text': point_text,
                            'field_code': '47A'
                        })
        else:
            # No numbered points, treat as single block
            if len(text) > 15:
                points.append({
                    'point_number': 1,
                    'text': text,
                    'field_code': '47A'
                })
        
        return points
    
    def _parse_documents(self, text: str) -> List[Dict]:
        """Parse documents required into numbered points"""
        if not text or len(text.strip()) < 10:
            return []
        
        # Clean the text
        text = text.strip()
        text = re.sub(r'^Documents Required\s*', '', text, flags=re.IGNORECASE)
        
        points = []
        
        # Split by numbered points
        parts = re.split(r'\s*[\(\[]?(\d+)[\)\]\.]\s+', text)
        
        if len(parts) > 2:
            for i in range(1, len(parts), 2):
                if i + 1 < len(parts):
                    point_num = int(parts[i])
                    point_text = parts[i + 1].strip()
                    
                    if point_text and len(point_text) > 5:
                        points.append({
                            'point_number': point_num,
                            'text': point_text,
                            'field_code': '46A'
                        })
        else:
            if len(text) > 15:
                points.append({
                    'point_number': 1,
                    'text': text,
                    'field_code': '46A'
                })
        
        return points
    
    def _parse_amendment_changes(self, text: str) -> List[Dict]:
        """Parse amendment instructions"""
        if not text or len(text.strip()) < 5:
            return []
        
        changes = []
        
        # Split by common amendment patterns
        instructions = re.split(
            r'(?=(?:/ADD/|/DELETE/|/REPALL/|CLAUSE\s+NO\.?\s*\d+))',
            text,
            flags=re.IGNORECASE
        )
        
        for instruction in instructions:
            instruction = instruction.strip()
            if len(instruction) < 5:
                continue
            
            # Extract clause number
            clause_match = re.search(
                r'CLAUSE\s+NO\.?\s*(\d+)',
                instruction,
                re.IGNORECASE
            )
            point_number = int(clause_match.group(1)) if clause_match else None
            
            # Determine operation
            operation = self._detect_operation(instruction)
            
            changes.append({
                'narrative': instruction,
                'change_text': instruction,
                'point_number': point_number,
                'operation': operation
            })
        
        return changes
    
    def _detect_operation(self, text: str) -> Optional[str]:
        """Detect the type of amendment operation"""
        text_upper = text.upper()
        
        if '/ADD/' in text_upper:
            return 'ADD'
        elif '/DELETE/' in text_upper:
            return 'DELETE'
        elif '/REPALL/' in text_upper:
            return 'REPALL'
        elif 'DELETE' in text_upper and 'REPLACE' in text_upper:
            return 'REPLACE'
        elif 'TO READ AS' in text_upper or 'NOW TO READ AS' in text_upper:
            return 'REPLACE'
        elif 'INSTEAD OF' in text_upper:
            return 'REPLACE'
        
        return None
    
    def consolidate(self, lc_number: str) -> Optional[Dict]:
        """Consolidate LC with all its amendments"""
        if lc_number not in self.lcs:
            return None
        
        print(f"\n{'='*80}")
        print(f"CONSOLIDATING LC: {lc_number}")
        print(f"{'='*80}")
        
        base_lc = self.lcs[lc_number]
        amendments = self.amendments.get(lc_number, [])
        
        # Sort amendments by number
        amendments.sort(key=lambda x: int(x['amendment_number']))
        
        # Build consolidated structure
        consolidated = {
            'lc_number': lc_number,
            'original_page_reference': base_lc['page_reference'],
            'issue_date': base_lc['issue_date'],
            'sender': base_lc['sender'],
            'receiver': base_lc['receiver'],
            'beneficiary': base_lc.get('beneficiary', ''),
            'amount': base_lc.get('amount', ''),
            'amendments_applied': len(amendments),
            'last_amendment_date': amendments[-1]['amendment_date'] if amendments else None,
            'fields': base_lc['fields'].copy(),
            'additional_conditions': [p.copy() for p in base_lc['additional_conditions']],
            'documents_required': [p.copy() for p in base_lc['documents_required']],
            'amendment_history': []
        }
        
        # Apply each amendment
        for amendment in amendments:
            amend_num = amendment['amendment_number']
            print(f"\n--- Applying Amendment {amend_num} ---")
            
            # Apply changes
            for change in amendment['changes']['additional_conditions']:
                self._apply_change(
                    consolidated['additional_conditions'],
                    change,
                    '47A'
                )
            
            for change in amendment['changes']['documents_required']:
                self._apply_change(
                    consolidated['documents_required'],
                    change,
                    '46A'
                )
            
            # Record in history (map keys to match frontend expectations)
            consolidated['amendment_history'].append({
                'amendment_number': amend_num,
                'amendment_date': amendment['amendment_date'],
                'page_reference': amendment['page_reference'],
                'changes': {
                    'conditions': amendment['changes']['additional_conditions'],
                    'documents': amendment['changes']['documents_required'],
                    'description_of_goods': amendment['changes'].get('description_of_goods', [])
                }
            })
        
        # Sort points
        consolidated['additional_conditions'].sort(
            key=lambda x: x.get('point_number', 999)
        )
        consolidated['documents_required'].sort(
            key=lambda x: x.get('point_number', 999)
        )
        
        print(f"\n{'='*80}")
        print(f"CONSOLIDATION COMPLETE")
        print(f"  Amendments: {len(amendments)}")
        print(f"  Conditions: {len(consolidated['additional_conditions'])}")
        print(f"  Documents: {len(consolidated['documents_required'])}")
        print(f"{'='*80}\n")
        
        return consolidated
    
    def _apply_change(self, points_list: List[Dict], change: Dict, field_code: str):
        """Apply amendment change to points list"""
        narrative = change.get('narrative', '')
        point_number = change.get('point_number')
        operation = change.get('operation')
        
        if not narrative:
            return
        
        # Find target point
        existing_point = None
        if point_number:
            existing_point = next(
                (p for p in points_list if p.get('point_number') == point_number),
                None
            )
        
        # DELETE operation
        if operation == 'DELETE' and point_number:
            points_list[:] = [
                p for p in points_list 
                if p.get('point_number') != point_number
            ]
            print(f"    ✓ Deleted point {point_number}")
            return
        
        # MODIFY existing point
        if existing_point:
            # Try to extract the replacement text
            new_text = self._extract_replacement_text(narrative)
            
            if new_text:
                existing_point['text'] = new_text
                existing_point['modified_by_amendment'] = True
                print(f"    ✓ Modified point {point_number}")
            else:
                print(f"    ⚠ Could not extract replacement for point {point_number}")
        
        # ADD new point
        elif operation == 'ADD':
            clean_text = self._clean_amendment_text(narrative)
            
            if clean_text and len(clean_text) > 15:
                if not point_number:
                    point_number = max(
                        [p.get('point_number', 0) for p in points_list],
                        default=0
                    ) + 1
                
                points_list.append({
                    'point_number': point_number,
                    'text': clean_text,
                    'field_code': field_code,
                    'added_by_amendment': True
                })
                print(f"    ✓ Added point {point_number}")
    
    def _extract_replacement_text(self, narrative: str) -> Optional[str]:
        """Extract the replacement text from amendment narrative"""
        # Pattern: TO READ AS "..."
        match = re.search(
            r'(?:TO\s+READ\s+AS|NOW\s+TO\s+READ\s+AS)\s+["\'](.+?)["\']',
            narrative,
            re.I | re.S
        )
        if match:
            return match.group(1).strip()
        
        # Pattern: DELETE "..." REPLACE BY "..."
        match = re.search(
            r'DELETE\s+["\'].*?["\']\s+REPLACE\s+(?:BY|WITH)\s+["\'](.+?)["\']',
            narrative,
            re.I | re.S
        )
        if match:
            return match.group(1).strip()
        
        return None
    
    def _clean_amendment_text(self, text: str) -> str:
        """Clean amendment text for addition"""
        # Remove amendment prefixes
        text = re.sub(r'^/ADD/\s*\+?\)?\s*', '', text, flags=re.I)
        text = re.sub(r'^CLAUSE\s+NO\.?\s*\d+:?\s*', '', text, flags=re.I)
        
        # Clean LaTeX artifacts
        text = re.sub(r'\\title\{[^}]*\}', '', text)
        text = re.sub(r'\\section\*?\{[^}]*\}', '', text)
        text = re.sub(r'\\[a-zA-Z]+', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def get_all_consolidated(self) -> List[Dict]:
        """Get all consolidated LCs"""
        results = []
        
        for lc_number in self.lcs.keys():
            consolidated = self.consolidate(lc_number)
            if consolidated:
                results.append(consolidated)
        
        return results
    
    def get_summary(self) -> Dict:
        """Get processing summary"""
        return {
            'total_lcs': len(self.lcs),
            'total_amendments': sum(len(amends) for amends in self.amendments.values()),
            'unidentified_documents': len(self.unidentified),
            'validation_issues': len(self.validation_log),
            'lc_numbers': list(self.lcs.keys()),
            'validation_log': self.validation_log
        }