"""
Document Processor for GOT-OCR API Integration
Parses structured JSON response from GOT-OCR server
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from .ocr_engine import EnhancedOCRProcessor


class DocumentProcessor:
    """
    Document processor that integrates with GOT-OCR API
    Handles structured JSON responses with identified_objects array
    """
    
    def __init__(self, ocr_backend='tesseract', use_preprocessing=True, 
                 use_postprocessing=True, use_api=True):
        print(f"[DocumentProcessor] Initializing with use_api={use_api}")
        self.use_api = use_api
        
        self.ocr_processor = EnhancedOCRProcessor(
            backend=ocr_backend,
            use_preprocessing=use_preprocessing,
            use_postprocessing=use_postprocessing,
            use_api=use_api
        )
    
    def process_document(self, file_path: str, job_id: str = None, 
                        processing_jobs: dict = None, **kwargs) -> Dict[str, Any]:
        """
        Process document and return structured data
        
        Returns:
            {
                'filename': str,
                'status': str,
                'lcs': List[Dict],
                'amendments': List[Dict],
                'shipping_docs': List[Dict],
                'unidentified': List[Dict],
                'raw_response': str
            }
        """
        path_obj = Path(file_path)
        
        if not path_obj.exists():
            return {
                'filename': path_obj.name,
                'status': 'error',
                'error': 'File not found',
                'lcs': [],
                'amendments': [],
                'shipping_docs': [],
                'unidentified': []
            }
        
        # Send to OCR API if enabled
        if self.use_api:
            print(f"[DocumentProcessor] Sending {path_obj.name} to OCR API...")
            
            # Update job message if provided
            if job_id and processing_jobs and job_id in processing_jobs:
                processing_jobs[job_id]["message"] = f"OCR processing {path_obj.name}..."
            
            # Call OCR API (without extra parameters)
            raw_response = self.ocr_processor._ocr_via_api(str(path_obj))
            
            # Parse the structured JSON response
            return self._parse_got_ocr_response(raw_response, path_obj.name)
        
        # Fallback to local processing (not recommended)
        return {
            'filename': path_obj.name,
            'status': 'error',
            'error': 'Local processing not implemented',
            'lcs': [],
            'amendments': [],
            'shipping_docs': [],
            'unidentified': []
        }
    
    def _parse_got_ocr_response(self, raw_json: str, filename: str) -> Dict[str, Any]:
        """
        Parse GOT-OCR server response with identified_objects array
        
        Response structure:
        {
            "filename": "file.pdf",
            "status": "success",
            "identified_objects": [
                {
                    "object_type": "lc" | "amendment" | "Bill of Lading" | "unidentified",
                    "page_reference": "1" | "2-4",
                    "page_count": 1,
                    "data": {...}
                }
            ]
        }
        """
        try:
            data = json.loads(raw_json)
            
            if not isinstance(data, dict):
                raise ValueError("Invalid response format")
            
            result = {
                'filename': data.get('filename', filename),
                'status': data.get('status', 'unknown'),
                'lcs': [],
                'amendments': [],
                'shipping_docs': [],
                'unidentified': [],
                'raw_response': raw_json
            }
            
            # Parse identified_objects array
            objects = data.get('identified_objects', [])
            
            print(f"[DocumentProcessor] Found {len(objects)} objects in {filename}")
            
            for obj in objects:
                object_type = obj.get('object_type', '').lower()
                page_ref = obj.get('page_reference', '?')
                page_count = obj.get('page_count', 1)
                obj_data = obj.get('data', {})
                
                # Create document entry
                doc_entry = {
                    'object_type': object_type,
                    'page_reference': page_ref,
                    'page_count': page_count,
                    'data': obj_data,
                    'document_type': obj_data.get('document_type', 'UNKNOWN'),
                    'document_category': obj_data.get('document_category', 'unknown')
                }
                
                # Categorize based on object_type
                if object_type == 'lc':
                    result['lcs'].append(doc_entry)
                    print(f"  ✓ LC found on page(s) {page_ref}")
                
                elif object_type == 'amendment':
                    result['amendments'].append(doc_entry)
                    amend_num = obj_data.get('Amendment_Number', 'Unknown')
                    print(f"  ✓ Amendment {amend_num} found on page(s) {page_ref}")
                
                elif object_type == 'unidentified':
                    result['unidentified'].append(doc_entry)
                    print(f"  ⚠ Unidentified document on page(s) {page_ref}")
                
                else:
                    # All other types are shipping documents
                    result['shipping_docs'].append(doc_entry)
                    doc_type = obj_data.get('document_type', object_type.upper())
                    print(f"  ✓ {doc_type} found on page(s) {page_ref}")
            
            # Log summary
            print(f"\n[DocumentProcessor] Summary for {filename}:")
            print(f"  LCs: {len(result['lcs'])}")
            print(f"  Amendments: {len(result['amendments'])}")
            print(f"  Shipping Docs: {len(result['shipping_docs'])}")
            print(f"  Unidentified: {len(result['unidentified'])}")
            
            return result
        
        except json.JSONDecodeError as e:
            print(f"✗ Error: Failed to parse JSON response: {e}")
            return {
                'filename': filename,
                'status': 'error',
                'error': f'JSON parse error: {str(e)}',
                'lcs': [],
                'amendments': [],
                'shipping_docs': [],
                'unidentified': [],
                'raw_response': raw_json
            }
        
        except Exception as e:
            print(f"✗ Error processing {filename}: {e}")
            return {
                'filename': filename,
                'status': 'error',
                'error': str(e),
                'lcs': [],
                'amendments': [],
                'shipping_docs': [],
                'unidentified': [],
                'raw_response': raw_json
            }
    
    def get_page_count(self, file_path: str) -> int:
        """Return number of pages (for compatibility)"""
        try:
            import fitz
            doc = fitz.open(file_path)
            count = doc.page_count
            doc.close()
            return count
        except:
            return 1