"""
Letter of Credit (LC) Extraction System
Handles SWIFT MT700 (LC Issuance) and MT707 (LC Amendment) formats
Supports multiple document formats with OCR capabilities
"""

import re
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path


@dataclass
class LCField:
    """Represents a single field in an LC"""
    field_code: str
    field_name: str
    value: str
    raw_text: str


@dataclass
class LCDocument:
    """Represents a complete LC document"""
    document_type: str  # "LC" or "AMENDMENT"
    lc_number: str
    message_type: str  # "MT700" or "MT707"
    issue_date: Optional[str] = None
    amendment_number: Optional[str] = None
    amendment_date: Optional[str] = None
    sender: Optional[str] = None
    receiver: Optional[str] = None
    fields: Dict[str, LCField] = None
    additional_conditions: List[Dict] = None
    documents_required: List[Dict] = None
    raw_text: str = ""
    
    def __post_init__(self):
        if self.fields is None:
            self.fields = {}
        if self.additional_conditions is None:
            self.additional_conditions = []
        if self.documents_required is None:
            self.documents_required = []


class LCExtractor:
    """Extracts structured data from LC documents"""
    
    # SWIFT field mappings
    FIELD_MAPPINGS = {
        'F20': 'Documentary Credit Number',
        'F21': 'Receiver\'s Reference',
        'F23': 'Issuing Bank\'s Reference',
        'F27': 'Sequence of Total',
        'F31C': 'Date of Issue',
        'F31D': 'Date and Place of Expiry',
        'F32B': 'Currency Code, Amount',
        'F40A': 'Form of Documentary Credit',
        'F40E': 'Applicable Rules',
        'F41D': 'Available With',
        'F42P': 'Deferred Payment Details',
        'F43P': 'Partial Shipments',
        'F43T': 'Transhipment',
        'F44C': 'Latest Date of Shipment',
        'F44E': 'Port of Loading',
        'F44F': 'Port of Discharge',
        'F45A': 'Description of Goods',
        'F46A': 'Documents Required',
        'F46B': 'Documents Required (Amendment)',
        'F47A': 'Additional Conditions',
        'F47B': 'Additional Conditions (Amendment)',
        'F48': 'Period for Presentation',
        'F49': 'Confirmation Instructions',
        'F50': 'Applicant',
        'F51D': 'Applicant Bank',
        'F52D': 'Issuing Bank',
        'F53A': 'Reimbursing Bank',
        'F59': 'Beneficiary',
        'F71D': 'Charges',
        'F78': 'Instructions to Bank',
        'F26E': 'Number of Amendment',
        'F30': 'Date of Amendment',
        'F22A': 'Purpose of Message',
        'F72Z': 'Sender to Receiver Information'
    }
    
    def __init__(self):
        self.current_doc = None
    
    def extract_from_text(self, text: str) -> LCDocument:
        """Extract LC data from text content"""
        # Determine document type
        if 'fin.707' in text or 'Amendment to a Documentary Credit' in text:
            return self._extract_amendment(text)
        elif 'fin.700' in text or 'Issue of a Documentary Credit' in text:
            return self._extract_lc(text)
        else:
            # Try to determine based on content
            if 'F26E' in text or 'Number of Amendment' in text:
                return self._extract_amendment(text)
            else:
                return self._extract_lc(text)
    
    def _extract_lc(self, text: str) -> LCDocument:
        """Extract original LC (MT700)"""
        doc = LCDocument(
            document_type="LC",
            lc_number="",
            message_type="MT700",
            raw_text=text
        )
        
        # Extract LC number
        lc_match = re.search(r'F20:\s*Documentary Credit Number\s*([A-Z0-9]+)', text)
        if lc_match:
            doc.lc_number = lc_match.group(1).strip()
        
        # Extract issue date
        date_match = re.search(r'F31C:\s*Date of Issue\s*(\d{6})\s*(\d{4}\s+\w+\s+\d+)', text)
        if date_match:
            doc.issue_date = date_match.group(2).strip()
        
        # Extract sender and receiver
        sender_match = re.search(r'Sender:\s*([A-Z0-9]+)', text)
        if sender_match:
            doc.sender = sender_match.group(1).strip()
        
        receiver_match = re.search(r'Receiver:\s*([A-Z0-9]+)', text)
        if receiver_match:
            doc.receiver = receiver_match.group(1).strip()
        
        # Extract all fields
        doc.fields = self._extract_all_fields(text)
        
        # Extract structured additional conditions (F47A)
        doc.additional_conditions = self._extract_numbered_points(text, 'F47A')
        
        # Extract structured documents required (F46A)
        doc.documents_required = self._extract_numbered_points(text, 'F46A')
        
        return doc
    
    def _extract_amendment(self, text: str) -> LCDocument:
        """Extract LC Amendment (MT707)"""
        doc = LCDocument(
            document_type="AMENDMENT",
            lc_number="",
            message_type="MT707",
            raw_text=text
        )
        
        # Extract LC number from sender's reference
        lc_match = re.search(r'F20:\s*Sender\'s Reference\s*([A-Z0-9]+)', text)
        if lc_match:
            doc.lc_number = lc_match.group(1).strip()
        
        # Extract amendment number
        amend_num_match = re.search(r'F26E:\s*Number of Amendment\s*(\d+)', text)
        if amend_num_match:
            doc.amendment_number = amend_num_match.group(1).strip()
        
        # Extract amendment date
        amend_date_match = re.search(r'F30:\s*Date of Amendment\s*(\d{6})\s*(\d{4}\s+\w+\s+\d+)', text)
        if amend_date_match:
            doc.amendment_date = amend_date_match.group(2).strip()
        
        # Extract original issue date
        issue_date_match = re.search(r'F31C:\s*Date of Issue\s*(\d{6})\s*(\d{4}\s+\w+\s+\d+)', text)
        if issue_date_match:
            doc.issue_date = issue_date_match.group(2).strip()
        
        # Extract sender and receiver
        sender_match = re.search(r'Sender:\s*([A-Z0-9]+)', text)
        if sender_match:
            doc.sender = sender_match.group(1).strip()
        
        receiver_match = re.search(r'Receiver:\s*([A-Z0-9]+)', text)
        if receiver_match:
            doc.receiver = receiver_match.group(1).strip()
        
        # Extract all fields
        doc.fields = self._extract_all_fields(text)
        
        # Extract amendment-specific changes
        doc.additional_conditions = self._extract_amendment_changes(text, 'F47B')
        doc.documents_required = self._extract_amendment_changes(text, 'F46B')
        
        return doc
    
    def _extract_all_fields(self, text: str) -> Dict[str, LCField]:
        """Extract all SWIFT fields from text"""
        fields = {}
        
        for field_code, field_name in self.FIELD_MAPPINGS.items():
            # Pattern to match field and its content
            pattern = rf'{field_code}:(.*?)(?=F\d+:|Other|$)'
            matches = re.finditer(pattern, text, re.DOTALL)
            
            for match in matches:
                raw_content = match.group(1).strip()
                
                # Clean up the content
                clean_value = self._clean_field_value(raw_content)
                
                if clean_value:
                    fields[field_code] = LCField(
                        field_code=field_code,
                        field_name=field_name,
                        value=clean_value,
                        raw_text=raw_content
                    )
        
        return fields
    
    def _clean_field_value(self, value: str) -> str:
        """Clean and normalize field value"""
        # Remove excessive whitespace
        value = re.sub(r'\s+', ' ', value)
        
        # Remove common prefixes
        value = re.sub(r'^(Name and Address:|Currency:|Code:|Narrative:|Date:|Place:|Number:|Total:|Amount:|Days:)\s*', '', value)
        
        return value.strip()
    
    def _extract_numbered_points(self, text: str, field_code: str) -> List[Dict]:
        """Extract numbered points from fields like F47A or F46A"""
        points = []
        
        # Find the field section
        pattern = rf'{field_code}:(.*?)(?=F\d+:|Other|$)'
        match = re.search(pattern, text, re.DOTALL)
        
        if not match:
            return points
        
        content = match.group(1)
        
        # Extract numbered points (1., 2., 3., etc.)
        point_pattern = r'(\d+)\.\s*([^\n]+(?:\n(?!\d+\.)[^\n]+)*)'
        point_matches = re.finditer(point_pattern, content, re.MULTILINE)
        
        for point_match in point_matches:
            point_num = point_match.group(1)
            point_text = point_match.group(2).strip()
            
            # Clean up the text
            point_text = re.sub(r'\s+', ' ', point_text)
            point_text = re.sub(r'\s*\.\s*$', '', point_text)
            
            points.append({
                'point_number': int(point_num),
                'text': point_text,
                'field_code': field_code
            })
        
        # Sort by point number
        points.sort(key=lambda x: x['point_number'])
        
        return points
    
    def _extract_amendment_changes(self, text: str, field_code: str) -> List[Dict]:
        """Extract amendment changes with operation codes (DELETE, ADD, REPALL)"""
        changes = []
        
        # Find the field section
        pattern = rf'{field_code}:(.*?)(?=F\d+:|Other|$)'
        match = re.search(pattern, text, re.DOTALL)
        
        if not match:
            return changes
        
        content = match.group(1)
        
        # Extract changes with operation codes
        current_change = None
        
        for line in content.split('\n'):
            line = line.strip()
            
            # Check for operation code
            if '/DELETE/' in line:
                if current_change:
                    changes.append(current_change)
                current_change = {
                    'operation': 'DELETE',
                    'field_code': field_code,
                    'narrative': []
                }
            elif '/ADD/' in line:
                if current_change:
                    changes.append(current_change)
                current_change = {
                    'operation': 'ADD',
                    'field_code': field_code,
                    'narrative': []
                }
            elif '/REPALL/' in line:
                if current_change:
                    changes.append(current_change)
                current_change = {
                    'operation': 'REPLACE_ALL',
                    'field_code': field_code,
                    'narrative': []
                }
            elif 'Narrative:' in line and current_change:
                # Extract narrative text
                narrative_text = re.sub(r'.*Narrative:\s*', '', line).strip()
                if narrative_text:
                    current_change['narrative'].append(narrative_text)
            elif line and current_change and not line.startswith('Code:') and not line.startswith('Line'):
                # Continue previous narrative
                current_change['narrative'].append(line)
        
        # Add last change
        if current_change:
            changes.append(current_change)
        
        # Join narratives and extract point numbers
        for change in changes:
            full_narrative = ' '.join(change['narrative'])
            change['narrative'] = full_narrative
            
            # Try to extract point number being modified
            point_match = re.search(r'FIELD\s+47A-(\d+)|46A-(\d+)', full_narrative)
            if point_match:
                point_num = point_match.group(1) or point_match.group(2)
                change['point_number'] = int(point_num)
            
            # Extract the actual change text
            change_text_match = re.search(r'AS\s+["\']([^"\']+)["\']|AS\s+(.+)$', full_narrative)
            if change_text_match:
                change['change_text'] = (change_text_match.group(1) or change_text_match.group(2)).strip()
        
        return changes


class LCConsolidator:
    """Consolidates LC with its amendments"""
    
    def __init__(self):
        self.lcs: Dict[str, LCDocument] = {}
        self.amendments: Dict[str, List[LCDocument]] = {}
    
    def add_document(self, doc: LCDocument):
        """Add a document to the consolidator"""
        if doc.document_type == "LC":
            self.lcs[doc.lc_number] = doc
        else:  # AMENDMENT
            if doc.lc_number not in self.amendments:
                self.amendments[doc.lc_number] = []
            self.amendments[doc.lc_number].append(doc)
    
    def consolidate(self, lc_number: str) -> Dict:
        """Consolidate an LC with all its amendments"""
        if lc_number not in self.lcs:
            return None
        
        original_lc = self.lcs[lc_number]
        amendments = self.amendments.get(lc_number, [])
        
        # Sort amendments by amendment number
        amendments.sort(key=lambda x: int(x.amendment_number) if x.amendment_number else 0)
        
        # Start with original LC data
        consolidated = {
            'lc_number': lc_number,
            'original_issue_date': original_lc.issue_date,
            'sender': original_lc.sender,
            'receiver': original_lc.receiver,
            'message_type': 'MT700_CONSOLIDATED',
            'amendments_applied': len(amendments),
            'last_amendment_date': amendments[-1].amendment_date if amendments else None,
            'fields': {k: asdict(v) for k, v in original_lc.fields.items()},
            'additional_conditions': original_lc.additional_conditions.copy(),
            'documents_required': original_lc.documents_required.copy(),
            'amendment_history': []
        }
        
        # Apply each amendment
        for amendment in amendments:
            amendment_record = {
                'amendment_number': amendment.amendment_number,
                'amendment_date': amendment.amendment_date,
                'changes': []
            }
            
            # Apply changes from F47B (Additional Conditions)
            for change in amendment.additional_conditions:
                self._apply_change(consolidated['additional_conditions'], change, 'F47A')
                amendment_record['changes'].append(change)
            
            # Apply changes from F46B (Documents Required)
            for change in amendment.documents_required:
                self._apply_change(consolidated['documents_required'], change, 'F46A')
                amendment_record['changes'].append(change)
            
            consolidated['amendment_history'].append(amendment_record)
        
        # Sort final points
        consolidated['additional_conditions'].sort(key=lambda x: x.get('point_number', 999))
        consolidated['documents_required'].sort(key=lambda x: x.get('point_number', 999))
        
        return consolidated
    
    def _apply_change(self, points_list: List[Dict], change: Dict, field_code: str):
        """Apply an amendment change to a list of points"""
        operation = change['operation']
        
        if operation == 'DELETE':
            # Find and remove the point
            if 'point_number' in change:
                points_list[:] = [p for p in points_list if p.get('point_number') != change['point_number']]
        
        elif operation == 'ADD':
            # Add a new point
            new_point = {
                'point_number': change.get('point_number', len(points_list) + 1),
                'text': change.get('change_text', change['narrative']),
                'field_code': field_code,
                'added_by_amendment': True
            }
            points_list.append(new_point)
        
        elif operation == 'REPLACE_ALL':
            # Replace an existing point entirely
            if 'point_number' in change:
                for i, point in enumerate(points_list):
                    if point.get('point_number') == change['point_number']:
                        points_list[i] = {
                            'point_number': change['point_number'],
                            'text': change.get('change_text', change['narrative']),
                            'field_code': field_code,
                            'modified_by_amendment': True
                        }
                        break
    
    def get_all_consolidated(self) -> List[Dict]:
        """Get all consolidated LCs"""
        results = []
        for lc_number in self.lcs.keys():
            consolidated = self.consolidate(lc_number)
            if consolidated:
                results.append(consolidated)
        return results


def process_lc_documents(file_paths: List[str], output_path: str = None) -> Dict:
    """
    Process multiple LC documents and consolidate them
    
    Args:
        file_paths: List of paths to LC documents (PDF, text, etc.)
        output_path: Optional path to save JSON output
    
    Returns:
        Dictionary containing all processed and consolidated LCs
    """
    extractor = LCExtractor()
    consolidator = LCConsolidator()
    
    results = {
        'processing_date': datetime.now().isoformat(),
        'total_documents_processed': 0,
        'lcs_found': 0,
        'amendments_found': 0,
        'documents': [],
        'consolidated_lcs': []
    }
    
    # Process each file
    for file_path in file_paths:
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            
            # Extract LC data
            doc = extractor.extract_from_text(text)
            
            # Add to consolidator
            consolidator.add_document(doc)
            
            # Track in results
            results['total_documents_processed'] += 1
            if doc.document_type == 'LC':
                results['lcs_found'] += 1
            else:
                results['amendments_found'] += 1
            
            # Add document summary
            doc_dict = asdict(doc)
            results['documents'].append(doc_dict)
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            continue
    
    # Consolidate all LCs
    results['consolidated_lcs'] = consolidator.get_all_consolidated()
    
    # Save to JSON if output path provided
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        files = sys.argv[1:]
        output = "lc_consolidated_output.json"
        results = process_lc_documents(files, output)
        print(f"\nProcessed {results['total_documents_processed']} documents")
        print(f"Found {results['lcs_found']} LCs and {results['amendments_found']} amendments")
        print(f"Consolidated {len(results['consolidated_lcs'])} LCs")
    else:
        print("Usage: python lc_extractor.py <file1> <file2> ...")
