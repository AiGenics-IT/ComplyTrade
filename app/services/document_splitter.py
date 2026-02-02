"""
Complete Multi-Document Processing Workflow
Splits combined text BEFORE passing to extractor
"""

import re
from typing import List, Dict, Any
from dataclasses import asdict
import json
from datetime import datetime


class MultiDocumentSplitter:
    """Intelligently splits combined text into individual documents"""
    
    def __init__(self):
        # Primary split patterns (SWIFT message boundaries)
        self.swift_markers = [
            r'(?:^|\n)Message\s*type:\s*(\d{3})',
            r'(?:^|\n)MT\s*(\d{3})',
            r'Formatted\s+(Outward|Inward)\s+SWIFT\s+message\s+details',
        ]
        
    def split_documents(self, combined_text: str) -> List[Dict[str, str]]:
        """
        Split combined text into individual documents
        
        Returns:
            List of {'text': str, 'type': str, 'index': int} dicts
        """
        # Find all SWIFT message type positions
        split_positions = []
        
        for pattern in self.swift_markers:
            for match in re.finditer(pattern, combined_text, re.IGNORECASE | re.MULTILINE):
                # Get context to determine if this is a real document boundary
                start = max(0, match.start() - 100)
                context = combined_text[start:match.start()]
                
                # Only split if this looks like a document start
                # (not embedded in middle of content)
                lines_before = context.split('\n')
                if len(lines_before[-1].strip()) < 30:  # Less than 30 chars on same line
                    split_positions.append({
                        'pos': match.start(),
                        'match': match.group(0),
                        'mt_type': match.group(1) if match.lastindex else None
                    })
        
        if not split_positions:
            print("⚠ No SWIFT markers found - returning as single document")
            return [{'index': 1, 'text': combined_text, 'type': 'UNKNOWN'}]
        
        # Remove duplicates at same position
        seen_pos = set()
        unique_splits = []
        for sp in split_positions:
            if sp['pos'] not in seen_pos:
                unique_splits.append(sp)
                seen_pos.add(sp['pos'])
        
        split_positions = sorted(unique_splits, key=lambda x: x['pos'])
        
        print(f"\n{'='*80}")
        print(f"DOCUMENT SPLITTING")
        print(f"{'='*80}")
        print(f"Found {len(split_positions)} document boundaries:")
        for i, sp in enumerate(split_positions, 1):
            print(f"  {i}. {sp['match'][:50]}... at position {sp['pos']}")
        print(f"{'='*80}\n")
        
        # Split text at these positions
        documents = []
        for i, split_info in enumerate(split_positions):
            pos = split_info['pos']
            
            # Get text from this position to next position (or end)
            if i < len(split_positions) - 1:
                end_pos = split_positions[i + 1]['pos']
            else:
                end_pos = len(combined_text)
            
            doc_text = combined_text[pos:end_pos].strip()
            
            # Only include if substantial content
            if len(doc_text) > 100:
                doc_type = self._detect_doc_type(doc_text, split_info['mt_type'])
                documents.append({
                    'index': i + 1,
                    'text': doc_text,
                    'type': doc_type,
                    'length': len(doc_text)
                })
        
        return documents
    
    def _detect_doc_type(self, text: str, mt_type: str = None) -> str:
        """Detect document type"""
        text_upper = text.upper()
        
        # Check MT type first
        if mt_type:
            if mt_type in ['707', '747', '767']:
                return 'AMENDMENT'
            elif mt_type == '700':
                return 'LC'
        
        # Check for amendment indicators
        amendment_indicators = [
            '26E:', 'NUMBER OF AMENDMENT', 'DATE OF AMENDMENT',
            '47B:', '46B:', '/ADD/', '/DELETE/', '/REPALL/'
        ]
        amendment_count = sum(1 for ind in amendment_indicators if ind in text_upper)
        
        if amendment_count >= 2:
            return 'AMENDMENT'
        
        # Check for LC indicators
        if 'DOCUMENTARY CREDIT NUMBER' in text_upper or 'IRREVOCABLE' in text_upper:
            return 'LC'
        
        # Supporting documents
        if 'INVOICE' in text_upper:
            return 'INVOICE'
        elif 'BILL OF LADING' in text_upper:
            return 'BILL_OF_LADING'
        elif 'CERTIFICATE' in text_upper:
            return 'CERTIFICATE'
        
        return 'UNKNOWN'


def process_multi_document_text(combined_text: str, extractor, consolidator) -> Dict[str, Any]:
    """
    Process combined text containing multiple documents
    
    Args:
        combined_text: Raw text from OCR (contains multiple documents)
        extractor: LCExtractor instance
        consolidator: LCConsolidator instance
    
    Returns:
        Processing results dictionary
    """
    # Step 1: Split into individual documents
    splitter = ImprovedMultiDocumentSplitter()
    split_docs = splitter.split_documents(combined_text)
    
    print(f"\n{'='*80}")
    print(f"PROCESSING {len(split_docs)} DOCUMENTS")
    print(f"{'='*80}\n")
    
    results = {
        'total_documents': len(split_docs),
        'lcs_found': 0,
        'amendments_found': 0,
        'supporting_docs_found': 0,
        'documents': [],
        'errors': []
    }
    
    # Step 2: Extract each document individually
    for doc_info in split_docs:
        try:
            print(f"\n→ Processing Document {doc_info['index']} ({doc_info['type']})")
            print(f"  Length: {doc_info['length']} chars")
            
            # Extract using LC extractor
            structured_doc = extractor.extract_from_text(doc_info['text'])
            
            # Categorize
            is_supporting = getattr(structured_doc, 'is_supporting', False) or \
                          structured_doc.message_type == "NON_SWIFT"
            
            if is_supporting:
                print(f"  ✓ Supporting Document: {structured_doc.document_type}")
                results['supporting_docs_found'] += 1
            else:
                print(f"  ✓ SWIFT Message: {structured_doc.document_type}")
                print(f"    LC Number: {structured_doc.lc_number}")
                print(f"    Amendment #: {structured_doc.amendment_number if hasattr(structured_doc, 'amendment_number') else 'N/A'}")
                
                # Add to consolidator
                consolidator.add_document(structured_doc)
                
                if structured_doc.document_type == 'LC':
                    results['lcs_found'] += 1
                elif structured_doc.document_type == 'AMENDMENT':
                    results['amendments_found'] += 1
            
            results['documents'].append({
                'index': doc_info['index'],
                'type': structured_doc.document_type,
                'lc_number': structured_doc.lc_number,
                'extracted': asdict(structured_doc)
            })
            
        except Exception as e:
            error_msg = f"Error processing document {doc_info['index']}: {str(e)}"
            print(f"  ✗ {error_msg}")
            results['errors'].append({
                'index': doc_info['index'],
                'error': error_msg
            })
    
    return results


def process_pdf_with_multiple_documents(pdf_path: str, output_path: str = None) -> Dict:
    """
    Complete workflow: OCR → Split → Extract → Consolidate
    
    Args:
        pdf_path: Path to PDF file
        output_path: Optional path to save JSON results
    
    Returns:
        Complete processing results
    """
    from services.OCR.document_processor import DocumentProcessor
    from services.lc_extractor import LCExtractor, LCConsolidator

    from services.Extractor.lc_consolidator import LCConsolidator

    
    print(f"\n{'='*80}")
    print(f"PROCESSING MULTI-DOCUMENT PDF")
    print(f"{'='*80}")
    print(f"File: {pdf_path}\n")
    
    # Step 1: Extract text with OCR
    print("Step 1: Extracting text from PDF...")
    doc_processor = DocumentProcessor(use_postprocessing=True)
    combined_text = doc_processor.process_document(pdf_path, force_ocr=False)
    
    if not combined_text or len(combined_text.strip()) < 50:
        print("⚠ No text extracted from PDF")
        return {'error': 'No text extracted'}
    
    print(f"✓ Extracted {len(combined_text)} characters\n")
    
    # Step 2: Initialize extractors
    print("Step 2: Initializing extractors...")
    extractor = LCExtractor()
    consolidator = LCConsolidator(use_ai=True)
    print("✓ Ready\n")
    
    # Step 3: Split and extract
    print("Step 3: Splitting and extracting documents...")
    extraction_results = process_multi_document_text(combined_text, extractor, consolidator)
    
    # Step 4: Consolidate
    print(f"\n{'='*80}")
    print("Step 4: Consolidating LCs with amendments...")
    print(f"{'='*80}\n")
    
    consolidated_lcs = consolidator.get_all_consolidated()
    
    # Step 5: Build final results
    final_results = {
        'processing_date': datetime.now().isoformat(),
        'source_file': pdf_path,
        'extraction_summary': {
            'total_documents': extraction_results['total_documents'],
            'lcs_found': extraction_results['lcs_found'],
            'amendments_found': extraction_results['amendments_found'],
            'supporting_docs_found': extraction_results['supporting_docs_found'],
            'errors': len(extraction_results['errors'])
        },
        'consolidated_lcs': consolidated_lcs,
        'all_documents': extraction_results['documents'],
        'errors': extraction_results['errors']
    }
    
    # Step 6: Save results
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Results saved to: {output_path}")
    
    # Print summary
    print(f"\n{'='*80}")
    print("PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"Documents processed: {extraction_results['total_documents']}")
    print(f"  - LCs: {extraction_results['lcs_found']}")
    print(f"  - Amendments: {extraction_results['amendments_found']}")
    print(f"  - Supporting Docs: {extraction_results['supporting_docs_found']}")
    print(f"Consolidated LCs: {len(consolidated_lcs)}")
    if extraction_results['errors']:
        print(f"Errors: {len(extraction_results['errors'])}")
    print(f"{'='*80}\n")
    
    return final_results


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        pdf_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else "consolidated_output.json"
        
        results = process_pdf_with_multiple_documents(pdf_file, output_file)
        
        # Print detailed summary
        if results.get('consolidated_lcs'):
            for lc in results['consolidated_lcs']:
                print(f"\n{'='*80}")
                print(f"LC: {lc['lc_number']}")
                print(f"{'='*80}")
                print(f"Amendments applied: {lc['amendments_applied']}")
                print(f"Additional conditions: {len(lc['additional_conditions'])}")
                print(f"Documents required: {len(lc['documents_required'])}")
    else:
        print("Usage: python multi_doc_processor.py <pdf_file> [output_file]")
        print("\nExample:")
        print("  python multi_doc_processor.py combined_docs.pdf results.json")