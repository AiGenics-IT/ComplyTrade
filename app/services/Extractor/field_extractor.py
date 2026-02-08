"""
Field Extractor
Extracts SWIFT fields and numbered points from LC documents
"""

import re
from typing import Dict, List
from .models import LCField
from .constants import COMPREHENSIVE_FIELD_MAPPINGS
from .utils import clean_field_value


class FieldExtractor:
    """Extracts fields and numbered points from SWIFT messages"""
    
    def __init__(self):
        self.field_mappings = COMPREHENSIVE_FIELD_MAPPINGS
    
    def extract_all_fields(self, text: str) -> Dict[str, LCField]:
        """Extract all SWIFT fields from text"""
        fields = {}
        
        # IMPROVED REGEX: 
        # 1. Matches tags at start of line or after a newline
        # 2. Matches tags even if preceded by a letter (e.g., F20: or Reference20:)
        # 3. Requires a colon after the 2-3 digit code
        flexible_pattern = r'(?:^|\n|(?<=[a-zA-Z]))\s*:?(\d{2,3}[A-Z]?):'
        matches = list(re.finditer(flexible_pattern, text))
        
        for i, match in enumerate(matches):
            field_num = match.group(1)
            start_pos = match.end()
            # The field ends where the next tag begins, or at the end of the document
            end_pos = matches[i+1].start() if i < len(matches) - 1 else len(text)
            
            raw_content = text[start_pos:end_pos].strip()
            
            # --- NOISY OCR CLEANING LOGIC ---
            # In your HBL/UBL docs, labels like "Documentary Credit Number" 
            # often appear right after the tag before the actual value.
            content_lines = raw_content.split('\n')
            
            if content_lines:
                first_line = content_lines[0].strip()
                # List of labels to ignore if they appear at the very start of a field
                noise_labels = [
                    "Documentary Credit Number", "Sender's Reference", "Receiver's Reference",
                    "Issuing Bank's Reference", "Number of Amendment", "Date of Amendment",
                    "Date of Issue", "Date and Place of Expiry", "Applicable Rules",
                    "Form of Documentary Credit", "Additional Conditions", "Documents Required",
                    "Description of Goods", "Narrative", "Lines 2-100", "Lines2to100"
                ]
                
                # Check if the first line is just one of these labels
                is_label = any(label.lower() in first_line.lower() for label in noise_labels)
                
                # If the first line is a known label and contains NO digits, 
                # or if it's explicitly a "Narrative" marker, skip it.
                if is_label and (not any(char.isdigit() for char in first_line) or "Narrative" in first_line):
                    field_value = "\n".join(content_lines[1:]).strip()
                else:
                    field_value = raw_content
            else:
                field_value = raw_content

            field_code = f':{field_num}:'
            
            # Store the field
            fields[field_code] = LCField(
                field_code=field_code,
                field_name=self.field_mappings.get(field_code, f"Field {field_num}"),
                value=clean_field_value(field_value),
                raw_text=raw_content
            )
            
        return fields
    
    def extract_numbered_points_robust(self, text: str, field_codes: List[str], 
                                      existing_fields: Dict[str, LCField] = None) -> List[Dict]:
        """
        PROPERLY extract ALL numbered points as INDIVIDUAL items.
        Handles (1), 1., and OCR-squashed numbers like '1.TEXT'
        """
        points = []
        
        # Use existing fields if provided, otherwise extract them
        if existing_fields is None:
            existing_fields = self.extract_all_fields(text)
            
        for code in field_codes:
            # Normalize the field code for lookup (e.g., '47A' -> ':47A:')
            lookup = code if code.startswith(':') else f':{code.rstrip(":")}:'
            
            if lookup not in existing_fields:
                continue
                
            content = existing_fields[lookup].raw_text
            
            # CRITICAL: Clean page markers FIRST
            content = re.sub(r'---\s*Page\s+\d+\s*---', ' ', content, flags=re.IGNORECASE)
            
            # --- ROBUST POINT SPLITTING ---
            # This regex matches:
            # 1. (1) or (12)
            # 2. 1. or 12.
            # 3. Numbers at the start of a new line followed by a space
            split_pattern = r'(?:\n|^)\s*(?:\(?(\d+)\)?[\.\s]|(\d+)\.)'
            
            # Find all numbering positions
            matches = list(re.finditer(split_pattern, content))
            
            if not matches:
                # If no numbered points found, treat the whole block as Point 1
                clean_text = clean_field_value(content)
                if clean_text:
                    points.append({
                        'point_number': 1,
                        'text': clean_text,
                        'field_code': lookup.strip(':')
                    })
                continue

            for i, match in enumerate(matches):
                point_num = match.group(1) or match.group(2)
                start_pos = match.end()
                # Point ends where the next number starts
                end_pos = matches[i+1].start() if i < len(matches) - 1 else len(content)
                
                point_text = content[start_pos:end_pos]
                
                # Clean OCR noise (Narrative, Lines 2-100, etc.) from the point text
                clean_point_text = clean_field_value(point_text)
                
                if clean_point_text:
                    points.append({
                        'point_number': int(point_num),
                        'text': clean_point_text,
                        'field_code': lookup.strip(':')
                    })
        
        # Remove duplicates and sort by point number
        seen_points = set()
        unique_points = []
        for p in sorted(points, key=lambda x: x['point_number']):
            if p['point_number'] not in seen_points:
                unique_points.append(p)
                seen_points.add(p['point_number'])
                
        return unique_points
    
    def extract_amendment_changes_complete(self, text: str, field_codes: List[str]) -> List[Dict]:
        """
        Enhanced extraction of amendment changes from tags like 46B, 47B
        Handles ALL formats including typos like CLAUE vs CLAUSE
        """
        changes = []
        fields = self.extract_all_fields(text)

        for code in field_codes:
            lookup = f':{code.strip(":")}:'
            if lookup not in fields:
                continue
                
            content = fields[lookup].raw_text
            
            # CRITICAL: Remove page markers FIRST
            content = re.sub(r'---\s*Page\s+\d+\s*---', '', content, flags=re.IGNORECASE)
            
            # Remove URLs and junk
            content = re.sub(r'http[s]?://[^\s]+', '', content)
            content = re.sub(r'Select\s+[\'"]Print[\'"].*', '', content, flags=re.IGNORECASE)
            content = re.sub(r'Formatted\s+Outward.*', '', content, flags=re.IGNORECASE)
            
            # Pattern 1: Try to find SWIFT operation tags (/ADD/, /DELETE/, /REPALL/)
            op_pattern = r'/(ADD|DELETE|REPALL)/'
            op_matches = list(re.finditer(op_pattern, content, re.IGNORECASE))

            if op_matches:
                # Extract changes with operation markers
                for i, match in enumerate(op_matches):
                    op_type = match.group(1).upper()
                    start_pos = match.end()
                    # Segment ends at the next operation tag or the end of the field
                    end_pos = op_matches[i+1].start() if i < len(op_matches) - 1 else len(content)
                    
                    segment_text = content[start_pos:end_pos]
                    
                    # Clean noise
                    segment_text = self._clean_amendment_text(segment_text)
                    
                    if segment_text:
                        change = {
                            'operation': op_type,
                            'field_code': code.strip(":"),
                            'content': segment_text
                        }

                        # Extract clause/point number - handle typos like CLAUE
                        point_match = re.search(
                            r'(?:CLAU[ES]+\s+)?NO\.?\s*(\d+)|FIELD\s+\d+[A-Z]?[-]?(\d+)', 
                            segment_text, 
                            re.IGNORECASE
                        )
                        
                        if point_match:
                            p_num = point_match.group(1) or point_match.group(2)
                            change['clause_number'] = p_num
                        
                        changes.append(change)
            
            else:
                # Pattern 2: No operation markers found
                cleaned_content = self._clean_amendment_text(content)
                
                if cleaned_content:
                    # Try to detect clause numbers (handle typos)
                    clause_pattern = r'(?:CLAU[ES]+\s+NO\.?\s*(\d+)|POINT\s+(\d+))'
                    clause_matches = list(re.finditer(clause_pattern, cleaned_content, re.IGNORECASE))
                    
                    if clause_matches:
                        # Split by clause numbers
                        for i, match in enumerate(clause_matches):
                            clause_num = match.group(1) or match.group(2)
                            start_pos = match.end()
                            end_pos = clause_matches[i+1].start() if i < len(clause_matches) - 1 else len(cleaned_content)
                            
                            clause_text = cleaned_content[start_pos:end_pos].strip()
                            clause_text = self._clean_amendment_text(clause_text)
                            
                            if clause_text:
                                changes.append({
                                    'operation': 'MODIFY',
                                    'field_code': code.strip(":"),
                                    'clause_number': clause_num,
                                    'content': clause_text
                                })
                    else:
                        # No clause numbers - treat as single change
                        changes.append({
                            'operation': 'MODIFY',
                            'field_code': code.strip(":"),
                            'content': cleaned_content
                        })
        
        return changes
    
    def _clean_amendment_text(self, text: str) -> str:
        """
        Clean OCR noise from amendment text - AGGRESSIVE cleaning
        """
        # Remove page markers
        text = re.sub(r'---\s*Page\s+\d+\s*---', ' ', text, flags=re.IGNORECASE)
        
        # Remove URLs
        text = re.sub(r'http[s]?://[^\s]+', '', text)
        
        # Remove print/header junk
        text = re.sub(r'Select\s+[\'"]Print[\'"].*?output[\.]*', '', text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r'Formatted\s+Outward\s+SWIFT.*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'FUSION\s+TRADE\s+INNOVATION', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Page\s+\d+\s+of\s+\d+', '', text, flags=re.IGNORECASE)
        
        # Remove bank letterhead fragments
        text = re.sub(r'H\s*BL\s+HABIB\s+BANK', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\d+\s*th\s+Floor,?\s+Jubilee\s+Insurance\s+House', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Karachi\s*-\s*Pakistan', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Tel:\s*\d{4}-\d{3}-\d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'swift:\s*[A-Z\s]+\d+', '', text, flags=re.IGNORECASE)
        
        # Remove "Lines X to Y" OCR noise
        text = re.sub(r'Lines\s?\d?\s?to\s?\d+:?', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Lines\s?\d?-\d+:?', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Lines\d+[a-z]*', '', text, flags=re.IGNORECASE) 
        text = re.sub(r'Narrativel?:?', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Code\s?:?', '', text, flags=re.IGNORECASE)
        
        # Standardize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove leading/trailing noise symbols
        text = re.sub(r'^[+)\s/:\-]+', '', text)
        text = re.sub(r'[+)\s/:\-]+$', '', text)
        
        return text