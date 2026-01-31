"""
Document Processor
Main processing pipeline for LC documents with extraction, consolidation, and AI auditing
"""

import json
from datetime import datetime
from typing import List, Dict
from dataclasses import asdict
from pathlib import Path
from .lc_extractor import LCExtractor
from .lc_consolidator import LCConsolidator
from services.ai_auditor import OfflineLCAuditor


def process_lc_documents(file_paths: List[str], output_path: str = None) -> Dict:
    """
    Process multiple LC documents with:
    1. SWIFT Consolidation (MT700 + MT707)
    2. Supporting Document Categorization (Invoice/BL)
    3. AI Audit of consolidated clauses against supporting docs
    """
    extractor = LCExtractor()
    consolidator = LCConsolidator()
    
    # Text accumulator to give the AI context from Invoices/BLs
    all_supporting_text = ""
    
    results = {
        'processing_date': datetime.now().isoformat(),
        'total_documents_processed': 0,
        'lcs_found': 0,
        'amendments_found': 0,
        'supporting_docs_found': 0,
        'documents': [],
        'consolidated_lcs': []
    }

    # --- PHASE 1: EXTRACTION & CATEGORIZATION ---
    for file_path in file_paths:
        try:
            print(f"\n{'='*40}")
            print(f"ANALYZING: {file_path}")
            print(f"{'='*40}")
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            
            # Extract doc (Returns LCDocument for both SWIFT and Supporting)
            doc = extractor.extract_from_text(text)
            
            # Logic Branching based on categorization
            is_supporting = getattr(doc, 'is_supporting', False) or doc.message_type == "NON_SWIFT"

            if is_supporting:
                print(f"[TYPE] Supporting Document Detected: {doc.document_type}")
                results['supporting_docs_found'] += 1
                # Accumulate text for AI context
                all_supporting_text += f"\n--- DATA FROM {doc.document_type} ({file_path}) ---\n"
                all_supporting_text += doc.raw_text + "\n"
            else:
                print(f"[TYPE] SWIFT Message Detected: {doc.document_type}")
                consolidator.add_document(doc)
                if doc.document_type == 'LC':
                    results['lcs_found'] += 1
                else:
                    results['amendments_found'] += 1

            results['total_documents_processed'] += 1
            results['documents'].append(asdict(doc))

        except Exception as e:
            print(f"ERROR processing {file_path}: {str(e)}")
            continue
    
    # --- PHASE 2: CONSOLIDATION ---
    print(f"\n{'-'*40}")
    print("CONSOLIDATING AMENDMENTS...")
    consolidated_data = consolidator.get_all_consolidated()
    
    # --- PHASE 3: AI AUDIT ---
    # Only run if we have a Master LC AND text from shipping docs (Invoice/BL)
    if consolidated_data and all_supporting_text.strip():
        print("RUNNING AI AUDIT AGAINST SUPPORTING DOCUMENTS...")
        try:
            auditor = OfflineLCAuditor()
            
            for lc in consolidated_data:
                # Audit Additional Conditions (47A) and Documents Required (46A)
                for field_key in ['additional_conditions', 'documents_required']:
                    if field_key in lc:
                        for point in lc[field_key]:
                            clause = point.get('text', '')
                            if len(clause) > 15:
                                # AI Verification
                                verdict = auditor.verify_clause(clause, all_supporting_text[:4000])
                                
                                point['ai_audit'] = {
                                    "status": "COMPLIANT" if "yes" in verdict.lower()[:10] else "DISCREPANCY",
                                    "explanation": verdict.strip()
                                }
        except Exception as ai_err:
            print(f"AI Audit encountered an issue: {str(ai_err)}")

    results['consolidated_lcs'] = consolidated_data

    # --- PHASE 4: EXPORT ---
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nSUCCESS: Results saved to {output_path}")

    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        files = sys.argv[1:]
        output = "lc_consolidated_output.json"
        results = process_lc_documents(files, output)
        print(f"\n{'='*70}")
        print("Processing Complete")
        print(f"{'='*70}")
        print(f"Documents processed: {results['total_documents_processed']}")
        print(f"LCs found: {results['lcs_found']}")
        print(f"Amendments found: {results['amendments_found']}")
        print(f"Consolidated: {len(results['consolidated_lcs'])} LCs")
    else:
        print("Usage: python document_processor.py <file1> <file2> ...")