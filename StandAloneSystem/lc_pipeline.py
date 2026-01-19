"""
Complete LC Processing Pipeline
Combines OCR, text extraction, and LC consolidation
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

# Import our modules
from lc_ocr import DocumentProcessor
from lc_extractor import LCExtractor, LCConsolidator


class LCProcessingPipeline:
    """Complete pipeline for processing LC documents"""
    
    def __init__(self, ocr_backend='tesseract', output_dir='./lc_output'):
        """
        Initialize LC processing pipeline
        
        Args:
            ocr_backend: OCR backend ('tesseract', 'easyocr', 'paddleocr')
            output_dir: Directory for output files
        """
        self.doc_processor = DocumentProcessor(ocr_backend=ocr_backend)
        self.lc_extractor = LCExtractor()
        self.consolidator = LCConsolidator()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"LC Processing Pipeline initialized")
        print(f"OCR Backend: {ocr_backend}")
        print(f"Output Directory: {self.output_dir}")
    
    def process_files(self, file_paths: List[str], force_ocr=False) -> Dict:
        """
        Process multiple LC documents
        
        Args:
            file_paths: List of paths to LC documents
            force_ocr: Force OCR even for digital PDFs
        
        Returns:
            Complete processing results
        """
        print(f"\n{'='*70}")
        print(f"PROCESSING {len(file_paths)} LC DOCUMENTS")
        print(f"{'='*70}\n")
        
        results = {
            'processing_date': datetime.now().isoformat(),
            'total_files': len(file_paths),
            'ocr_backend': self.doc_processor.ocr_processor.backend,
            'files_processed': [],
            'lcs_found': {},
            'amendments_found': {},
            'consolidated_lcs': {},
            'errors': []
        }
        
        # Step 1: Extract text from all documents
        print("STEP 1: Text Extraction")
        print("-" * 70)
        
        extracted_texts = {}
        for i, file_path in enumerate(file_paths, 1):
            try:
                print(f"\n[{i}/{len(file_paths)}] {Path(file_path).name}")
                
                # Extract text
                text = self.doc_processor.process_document(file_path, force_ocr=force_ocr)
                
                if text.strip():
                    extracted_texts[file_path] = text
                    
                    # Save extracted text
                    text_output = self.output_dir / 'extracted_texts' / f"{Path(file_path).stem}.txt"
                    text_output.parent.mkdir(parents=True, exist_ok=True)
                    with open(text_output, 'w', encoding='utf-8') as f:
                        f.write(text)
                    
                    print(f"  ✓ Text extracted ({len(text)} chars)")
                    print(f"  Saved to: {text_output}")
                    
                    results['files_processed'].append({
                        'file': str(file_path),
                        'status': 'success',
                        'text_length': len(text)
                    })
                else:
                    print(f"  ⚠ No text extracted")
                    results['files_processed'].append({
                        'file': str(file_path),
                        'status': 'no_text',
                        'text_length': 0
                    })
                    
            except Exception as e:
                error_msg = f"Error processing {file_path}: {str(e)}"
                print(f"  ✗ {error_msg}")
                results['errors'].append(error_msg)
                results['files_processed'].append({
                    'file': str(file_path),
                    'status': 'error',
                    'error': str(e)
                })
        
        # Step 2: Extract LC structure
        print(f"\n\nSTEP 2: LC Structure Extraction")
        print("-" * 70)
        
        for file_path, text in extracted_texts.items():
            try:
                print(f"\nAnalyzing: {Path(file_path).name}")
                
                # Extract LC document
                lc_doc = self.lc_extractor.extract_from_text(text)
                
                if lc_doc.lc_number:
                    print(f"  ✓ {lc_doc.document_type}: {lc_doc.lc_number}")
                    
                    if lc_doc.document_type == 'LC':
                        print(f"    Issue Date: {lc_doc.issue_date}")
                        print(f"    Fields Extracted: {len(lc_doc.fields)}")
                        print(f"    Additional Conditions: {len(lc_doc.additional_conditions)}")
                        print(f"    Documents Required: {len(lc_doc.documents_required)}")
                        
                        results['lcs_found'][lc_doc.lc_number] = {
                            'file': str(file_path),
                            'issue_date': lc_doc.issue_date,
                            'fields_count': len(lc_doc.fields),
                            'conditions_count': len(lc_doc.additional_conditions),
                            'documents_count': len(lc_doc.documents_required)
                        }
                    else:
                        print(f"    Amendment #{lc_doc.amendment_number}")
                        print(f"    Amendment Date: {lc_doc.amendment_date}")
                        print(f"    Changes: {len(lc_doc.additional_conditions) + len(lc_doc.documents_required)}")
                        
                        if lc_doc.lc_number not in results['amendments_found']:
                            results['amendments_found'][lc_doc.lc_number] = []
                        
                        results['amendments_found'][lc_doc.lc_number].append({
                            'file': str(file_path),
                            'amendment_number': lc_doc.amendment_number,
                            'amendment_date': lc_doc.amendment_date,
                            'changes_count': len(lc_doc.additional_conditions) + len(lc_doc.documents_required)
                        })
                    
                    # Add to consolidator
                    self.consolidator.add_document(lc_doc)
                    
                    # Save individual document JSON
                    doc_output = self.output_dir / 'individual_docs' / f"{Path(file_path).stem}.json"
                    doc_output.parent.mkdir(parents=True, exist_ok=True)
                    
                    from dataclasses import asdict
                    with open(doc_output, 'w', encoding='utf-8') as f:
                        json.dump(asdict(lc_doc), f, indent=2, ensure_ascii=False)
                    
                    print(f"  Saved to: {doc_output}")
                else:
                    print(f"  ⚠ Could not identify LC number")
                    
            except Exception as e:
                error_msg = f"Error extracting LC from {file_path}: {str(e)}"
                print(f"  ✗ {error_msg}")
                results['errors'].append(error_msg)
        
        # Step 3: Consolidate LCs with amendments
        print(f"\n\nSTEP 3: Consolidation")
        print("-" * 70)
        
        consolidated_output = self.output_dir / 'consolidated'
        consolidated_output.mkdir(parents=True, exist_ok=True)
        
        all_consolidated = self.consolidator.get_all_consolidated()
        
        for consolidated_lc in all_consolidated:
            lc_number = consolidated_lc['lc_number']
            print(f"\nConsolidating LC: {lc_number}")
            print(f"  Amendments Applied: {consolidated_lc['amendments_applied']}")
            print(f"  Final Conditions: {len(consolidated_lc['additional_conditions'])}")
            print(f"  Final Documents Required: {len(consolidated_lc['documents_required'])}")
            
            # Save consolidated LC
            output_file = consolidated_output / f"{lc_number}_consolidated.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(consolidated_lc, f, indent=2, ensure_ascii=False)
            
            print(f"  Saved to: {output_file}")
            
            results['consolidated_lcs'][lc_number] = {
                'amendments_applied': consolidated_lc['amendments_applied'],
                'conditions_count': len(consolidated_lc['additional_conditions']),
                'documents_count': len(consolidated_lc['documents_required']),
                'output_file': str(output_file)
            }
        
        # Step 4: Generate summary report
        print(f"\n\nSTEP 4: Summary Report")
        print("-" * 70)
        
        summary_file = self.output_dir / 'processing_summary.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Processing Complete")
        print(f"\nSummary:")
        print(f"  Total Files: {results['total_files']}")
        print(f"  LCs Found: {len(results['lcs_found'])}")
        print(f"  Amendments Found: {sum(len(v) for v in results['amendments_found'].values())}")
        print(f"  Consolidated LCs: {len(results['consolidated_lcs'])}")
        print(f"  Errors: {len(results['errors'])}")
        print(f"\nOutput Directory: {self.output_dir}")
        print(f"Summary Report: {summary_file}")
        
        return results
    
    def generate_human_readable_report(self, lc_number: str = None):
        """Generate a human-readable report of consolidated LC"""
        if lc_number:
            lc_numbers = [lc_number]
        else:
            lc_numbers = list(self.consolidator.lcs.keys())
        
        for lc_num in lc_numbers:
            consolidated = self.consolidator.consolidate(lc_num)
            if not consolidated:
                continue
            
            report_file = self.output_dir / 'reports' / f"{lc_num}_report.txt"
            report_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write(f"LETTER OF CREDIT - CONSOLIDATED REPORT\n")
                f.write("=" * 80 + "\n\n")
                
                f.write(f"LC Number: {consolidated['lc_number']}\n")
                f.write(f"Original Issue Date: {consolidated['original_issue_date']}\n")
                f.write(f"Amendments Applied: {consolidated['amendments_applied']}\n")
                if consolidated['last_amendment_date']:
                    f.write(f"Last Amendment Date: {consolidated['last_amendment_date']}\n")
                f.write(f"Sender: {consolidated['sender']}\n")
                f.write(f"Receiver: {consolidated['receiver']}\n")
                f.write("\n")
                
                # Additional Conditions
                f.write("-" * 80 + "\n")
                f.write("ADDITIONAL CONDITIONS (Field 47A)\n")
                f.write("-" * 80 + "\n\n")
                
                for condition in consolidated['additional_conditions']:
                    f.write(f"{condition['point_number']}. {condition['text']}\n")
                    if condition.get('added_by_amendment'):
                        f.write("   [Added by Amendment]\n")
                    elif condition.get('modified_by_amendment'):
                        f.write("   [Modified by Amendment]\n")
                    f.write("\n")
                
                # Documents Required
                f.write("\n" + "-" * 80 + "\n")
                f.write("DOCUMENTS REQUIRED (Field 46A)\n")
                f.write("-" * 80 + "\n\n")
                
                for doc in consolidated['documents_required']:
                    f.write(f"{doc['point_number']}. {doc['text']}\n")
                    if doc.get('added_by_amendment'):
                        f.write("   [Added by Amendment]\n")
                    elif doc.get('modified_by_amendment'):
                        f.write("   [Modified by Amendment]\n")
                    f.write("\n")
                
                # Amendment History
                if consolidated['amendment_history']:
                    f.write("\n" + "-" * 80 + "\n")
                    f.write("AMENDMENT HISTORY\n")
                    f.write("-" * 80 + "\n\n")
                    
                    for amendment in consolidated['amendment_history']:
                        f.write(f"Amendment #{amendment['amendment_number']}\n")
                        f.write(f"Date: {amendment['amendment_date']}\n")
                        f.write(f"Changes Applied: {len(amendment['changes'])}\n\n")
                        
                        for i, change in enumerate(amendment['changes'], 1):
                            f.write(f"  Change {i}:\n")
                            f.write(f"    Operation: {change['operation']}\n")
                            f.write(f"    Field: {change['field_code']}\n")
                            if 'point_number' in change:
                                f.write(f"    Point: {change['point_number']}\n")
                            f.write(f"    Details: {change['narrative']}\n\n")
            
            print(f"Human-readable report saved to: {report_file}")


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("LC Processing Pipeline")
        print("=" * 70)
        print("\nUsage:")
        print("  python lc_pipeline.py <file1> <file2> ... [options]")
        print("\nOptions:")
        print("  --ocr-backend <backend>  OCR backend: tesseract, easyocr, paddleocr")
        print("  --force-ocr              Force OCR even for digital PDFs")
        print("  --output-dir <dir>       Output directory (default: ./lc_output)")
        print("\nExample:")
        print("  python lc_pipeline.py LC1.pdf Amendment1.pdf --output-dir results")
        return
    
    # Parse arguments
    files = []
    ocr_backend = 'tesseract'
    force_ocr = False
    output_dir = './lc_output'
    
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        
        if arg == '--ocr-backend':
            ocr_backend = sys.argv[i + 1]
            i += 2
        elif arg == '--force-ocr':
            force_ocr = True
            i += 1
        elif arg == '--output-dir':
            output_dir = sys.argv[i + 1]
            i += 2
        else:
            files.append(arg)
            i += 1
    
    # Initialize pipeline
    pipeline = LCProcessingPipeline(ocr_backend=ocr_backend, output_dir=output_dir)
    
    # Process files
    results = pipeline.process_files(files, force_ocr=force_ocr)
    
    # Generate human-readable reports
    print("\nGenerating human-readable reports...")
    pipeline.generate_human_readable_report()
    
    print("\n" + "=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
