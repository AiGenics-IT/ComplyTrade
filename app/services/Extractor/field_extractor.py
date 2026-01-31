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
        Extracts ALL 5 changes from the provided PDF by iterating through every 
        /ADD/, /DELETE/, or /REPALL/ tag found in each field.
        """
        changes = []
        fields = self.extract_all_fields(text)

        for code in field_codes:
            lookup = f':{code.strip(":")}:'
            if lookup not in fields:
                continue
                
            content = fields[lookup].raw_text
            
            # 1. Find ALL SWIFT operation tags in the field
            op_pattern = r'/(ADD|DELETE|REPALL)/'
            op_matches = list(re.finditer(op_pattern, content, re.IGNORECASE))

            # 2. Extract and clean the segment for EACH operation found
            for i, match in enumerate(op_matches):
                op_type = match.group(1).upper()
                start_pos = match.end()
                # Segment ends at the next operation tag or the end of the field
                end_pos = op_matches[i+1].start() if i < len(op_matches) - 1 else len(content)
                
                segment_text = content[start_pos:end_pos]
                
                # --- AGGRESSIVE CLEANING ---
                # Remove repeating OCR noise and squashed "Lines2t" fragments
                segment_text = re.sub(r'Lines\s?\d?\s?to\s?\d+:?', '', segment_text, flags=re.IGNORECASE)
                segment_text = re.sub(r'Lines\s?\d?-\d+:?', '', segment_text, flags=re.IGNORECASE)
                segment_text = re.sub(r'Lines\d+[a-z]*', '', segment_text, flags=re.IGNORECASE) 
                segment_text = re.sub(r'Narrativel?:?', '', segment_text, flags=re.IGNORECASE)
                segment_text = re.sub(r'Code\s?:?', '', segment_text, flags=re.IGNORECASE)
                
                # Standardize whitespace and remove leading/trailing noise symbols
                segment_text = re.sub(r'\s+', ' ', segment_text).strip()
                segment_text = re.sub(r'^[+)\s/:\-]+', '', segment_text) 

                if segment_text:
                    change = {
                        'operation': op_type,
                        'field_code': code.strip(":"),
                        'narrative': segment_text,
                        'change_text': segment_text
                    }

                    # 3. Detect Point Number (e.g., 1, 11, 19, 20, 21)
                    point_match = re.search(
                        r'(?:CLAU[S|E]*\s+)?NO\.?\s*(\d+)|FIELD\s+\d+[A-Z]?[-]?(\d+)', 
                        segment_text, 
                        re.IGNORECASE
                    )
                    
                    if point_match:
                        p_num = point_match.group(1) or point_match.group(2)
                        change['point_number'] = int(p_num)
                    
                    changes.append(change)
        
        return changes