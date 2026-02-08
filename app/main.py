"""
LC Processing REST API
FastAPI-based service for processing Letter of Credit documents
Integrated with GOT-OCR server for superior accuracy
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from pydantic import BaseModel
import uvicorn
import os
import json
import uuid
import re
from pathlib import Path
from datetime import datetime
import shutil

# Import our LC processing modules
from services.OCR.document_processor import DocumentProcessor
# IMPORTANT: Use the fixed consolidator with validation
from services.Extractor.lc_consolidator import LCConsolidatorGOT

# Initialize FastAPI app
app = FastAPI(
    title="LC Processing API",
    description="API for processing Letter of Credit documents and amendments with GOT-OCR",
    version="2.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
UPLOAD_DIR = Path("uploads/lc_uploads")
RESULTS_DIR = Path("uploads/lc_results")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# In-memory storage for processing status
processing_jobs = {}


# Helper function to sanitize filenames
def sanitize_filename(text: str, max_length: int = 100) -> str:
    """
    Sanitize text for use as a filename.
    Removes invalid characters and limits length.
    """
    if not text:
        return "unnamed"
    
    # Remove invalid Windows filename characters: < > : " / \ | ? *
    text = re.sub(r'[<>:"/\\|?*]', '', text)
    
    # Replace spaces and special chars with underscores
    text = re.sub(r'[\s\+\-]+', '_', text)
    
    # Remove any remaining non-alphanumeric chars except underscores and dots
    text = re.sub(r'[^a-zA-Z0-9_.]', '', text)
    
    # Remove multiple consecutive underscores
    text = re.sub(r'_+', '_', text)
    
    # Remove leading/trailing underscores
    text = text.strip('_')
    
    # Limit length
    if len(text) > max_length:
        text = text[:max_length]
    
    return text or "unnamed"


# Pydantic models for API
class JobStatus(BaseModel):
    job_id: str
    status: str  # "pending", "processing", "completed", "failed"
    message: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None
    files_processed: int = 0
    lcs_found: int = 0
    amendments_found: int = 0
    shipping_docs_found: int = 0
    unidentified_found: int = 0
    errors: List[str] = []


class DocumentInfo(BaseModel):
    source_file: str
    page_reference: str
    page_count: int
    document_type: str
    category: str


class ConsolidatedLCInfo(BaseModel):
    lc_number: str
    original_page_reference: str
    amendments_applied: int
    last_amendment_date: Optional[str]
    conditions_count: int
    documents_count: int
    download_url: str


# Helper functions
def create_job_id() -> str:
    """Generate unique job ID"""
    return str(uuid.uuid4())


def get_job_dir(job_id: str) -> Path:
    """Get job-specific directory"""
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(exist_ok=True)
    return job_dir


def get_results_dir(job_id: str) -> Path:
    """Get job-specific results directory"""
    results_dir = RESULTS_DIR / job_id
    results_dir.mkdir(exist_ok=True)
    return results_dir


async def save_upload_file(upload_file: UploadFile, destination: Path):
    """Save uploaded file to destination"""
    with open(destination, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)


def _extract_lc_number_from_data(data: dict) -> str:
    """Extract LC number from GOT-OCR data"""
    for field in ['DC_Number', 'Reference', 'Issuing_Bank_Reference']:
        value = data.get(field, '')
        if value:
            match = re.search(r'([A-Z]{2,}[0-9]{10,}[A-Z]{0,})', value)
            if match:
                return match.group(1)
    return "UNKNOWN"


def _extract_amendment_number(text: str) -> str:
    """Extract amendment number from text"""
    match = re.search(r'(\d+)', text)
    return match.group(1).zfill(2) if match else '00'


def _extract_date(text: str) -> str:
    """Extract date from text"""
    match = re.search(r'(\d{6})', text)
    return match.group(1) if match else ''


def process_lc_job(job_id: str, file_paths: List[Path], ocr_backend: str = "tesseract"):
    """
    Background task to process LC documents using GOT-OCR server
    Maintains original function name for compatibility
    
    Workflow:
    1. Send files to GOT-OCR server
    2. Parse JSON response with identified_objects
    3. Categorize documents (LC, Amendment, Shipping, Unidentified)
    4. Consolidate LCs with amendments (with validation)
    5. Generate results and save to disk
    """
    try:
        # Initialize Status
        processing_jobs[job_id]["status"] = "processing"
        processing_jobs[job_id]["message"] = "Initializing GOT-OCR processor..."
        
        print(f"\n{'='*80}")
        print(f"JOB STARTED: {job_id}")
        print(f"{'='*80}\n")
        
        # 1. Initialize Processor and Consolidator
        doc_processor = DocumentProcessor(use_api=True)
        consolidator = LCConsolidatorGOT(use_ai=True)  # Uses fixed version with validation
        results_dir = get_results_dir(job_id)
        
        # Track all documents
        all_lcs = []
        all_amendments = []
        all_shipping_docs = []
        all_unidentified = []
        
        # 2. Process Each File
        for file_path in file_paths:
            try:
                processing_jobs[job_id]["message"] = f"Analyzing {file_path.name} with GOT-OCR..."
                print(f"\n{'='*80}")
                print(f"Processing: {file_path.name}")
                print(f"{'='*80}")
                
                # Send to GOT-OCR API and get structured response
                result = doc_processor.process_document(
                    str(file_path),
                    job_id=job_id,
                    processing_jobs=processing_jobs
                )
                
                if result['status'] != 'success':
                    error = result.get('error', 'Unknown error')
                    print(f"  ✗ Error: {error}")
                    processing_jobs[job_id]["errors"].append(f"{file_path.name}: {error}")
                    continue
                
                # Extract categorized documents
                lcs = result.get('lcs', [])
                amendments = result.get('amendments', [])
                shipping_docs = result.get('shipping_docs', [])
                unidentified = result.get('unidentified', [])
                
                # Log summary
                print(f"\n  Summary for {file_path.name}:")
                print(f"    LCs: {len(lcs)}")
                print(f"    Amendments: {len(amendments)}")
                print(f"    Shipping Docs: {len(shipping_docs)}")
                print(f"    Unidentified: {len(unidentified)}")
                
                # Add source filename to each document
                for doc in lcs + amendments + shipping_docs + unidentified:
                    doc['source_file'] = file_path.name
                
                # Accumulate documents
                all_lcs.extend(lcs)
                all_amendments.extend(amendments)
                all_shipping_docs.extend(shipping_docs)
                all_unidentified.extend(unidentified)
                
                # Add LCs and Amendments to consolidator (with validation)
                # The consolidator will validate and possibly reclassify documents
                for lc in lcs:
                    consolidator.add_document(lc)
                
                for amendment in amendments:
                    consolidator.add_document(amendment)
                
                # Update counts AFTER validation
                # Note: Some LCs/amendments might be reclassified as unidentified
                processing_jobs[job_id]["lcs_found"] = len(consolidator.lcs)
                processing_jobs[job_id]["amendments_found"] = sum(
                    len(amends) for amends in consolidator.amendments.values()
                )
                processing_jobs[job_id]["shipping_docs_found"] += len(shipping_docs)
                processing_jobs[job_id]["unidentified_found"] = (
                    len(unidentified) + len(consolidator.unidentified)
                )
                processing_jobs[job_id]["files_processed"] += 1
                
            except Exception as e:
                error_msg = f"Error processing {file_path.name}: {str(e)}"
                print(f"  ✗ {error_msg}")
                processing_jobs[job_id]["errors"].append(error_msg)
        
        # 2.5. Log validation results
        validation_summary = consolidator.get_summary()
        if validation_summary['validation_issues'] > 0:
            print(f"\n{'='*80}")
            print(f"VALIDATION SUMMARY")
            print(f"{'='*80}")
            print(f"  Valid LCs: {validation_summary['total_lcs']}")
            print(f"  Valid Amendments: {validation_summary['total_amendments']}")
            print(f"  Validation Issues: {validation_summary['validation_issues']}")
            print(f"  Reclassified as Unidentified: {len(consolidator.unidentified)}")
            
            for issue in validation_summary['validation_log']:
                print(f"    {issue}")
            print(f"{'='*80}\n")
        
        # 3. Consolidate LCs with Amendments
        processing_jobs[job_id]["message"] = "Consolidating LCs with amendments..."
        print(f"\n{'='*80}")
        print(f"CONSOLIDATION PHASE")
        print(f"{'='*80}")
        
        consolidated_lcs = consolidator.get_all_consolidated()
        
        # 4. Save Consolidated LCs
        consolidated_dir = results_dir / "consolidated"
        consolidated_dir.mkdir(exist_ok=True)
        
        for consolidated_lc in consolidated_lcs:
            lc_num = consolidated_lc['lc_number']
            safe_name = sanitize_filename(lc_num)
            
            # Verify data integrity before saving
            if not isinstance(consolidated_lc.get('additional_conditions'), list):
                print(f"  ⚠ WARNING: additional_conditions is not a list for {lc_num}")
                consolidated_lc['additional_conditions'] = []
            
            if not isinstance(consolidated_lc.get('documents_required'), list):
                print(f"  ⚠ WARNING: documents_required is not a list for {lc_num}")
                consolidated_lc['documents_required'] = []
            
            consolidated_path = consolidated_dir / f"{safe_name}_consolidated.json"
            with open(consolidated_path, "w", encoding="utf-8") as f:
                json.dump(consolidated_lc, f, indent=2, ensure_ascii=False)
            
            print(f"  ✓ Saved consolidated LC: {lc_num}")
        
        # 5. Add reclassified documents to unidentified list
        all_unidentified.extend(consolidator.unidentified)
        
        # 6. Format Document Lists for API Response
        lcs_list = _format_document_list(all_lcs, job_id, 'LC')
        amendments_list = _format_document_list(all_amendments, job_id, 'AMENDMENT')
        shipping_list = _format_document_list(all_shipping_docs, job_id, 'SHIPPING')
        unidentified_list = _format_document_list(all_unidentified, job_id, 'UNIDENTIFIED')
        
        # 7. Build Final Results
        final_output = {
            "job_id": job_id,
            "processing_date": datetime.now().isoformat(),
            
            # Summary counts
            "summary": {
                "files_analyzed": len(file_paths),
                "lcs_found": len(consolidator.lcs),  # Only valid LCs
                "amendments_found": sum(len(amends) for amends in consolidator.amendments.values()),
                "shipping_docs_found": len(all_shipping_docs),
                "unidentified_found": len(all_unidentified),
                "consolidated_lcs": len(consolidated_lcs),
                "validation_issues": validation_summary['validation_issues']
            },
            
            # Document counts
            "counts": {
                "lcs": len(consolidator.lcs),
                "amendments": sum(len(amends) for amends in consolidator.amendments.values()),
                "shipping_docs": len(all_shipping_docs),
                "unidentified": len(all_unidentified),
                "consolidated_lcs": len(consolidated_lcs)
            },
            
            # Document lists with page references
            "lcs": lcs_list,
            "amendments": amendments_list,
            "shipping_docs": shipping_list,
            "unidentified": unidentified_list,
            
            # Consolidated LCs with full data
            "consolidated_lcs": [
                {
                    "lc_number": lc.get('lc_number', 'UNKNOWN'),
                    "original_page_reference": lc.get('original_page_reference', '?'),
                    "issue_date": lc.get('issue_date', ''),
                    "sender": lc.get('sender', '')[:100] if lc.get('sender') else '',
                    "receiver": lc.get('receiver', '')[:100] if lc.get('receiver') else '',
                    "amendments_applied": lc.get('amendments_applied', 0),
                    "last_amendment_date": lc.get('last_amendment_date', ''),
                    "conditions_count": len(lc.get('additional_conditions', [])),
                    "documents_count": len(lc.get('documents_required', [])),
                    "amendment_history": lc.get('amendment_history', []),
                    
                    # FULL DATA - Add complete arrays
                    "additional_conditions": lc.get('additional_conditions', []),
                    "documents_required": lc.get('documents_required', []),
                    "fields": lc.get('fields', {}),  # All raw OCR fields
                    
                    "download_url": f"/api/download/{job_id}/{sanitize_filename(lc.get('lc_number', 'unknown'))}"
                }
                for lc in consolidated_lcs
            ],
            
            # Validation info
            "validation": {
                "total_issues": validation_summary['validation_issues'],
                "validation_log": validation_summary['validation_log'],
                "valid_lcs": validation_summary['lc_numbers']
            },
            
            # Errors
            "errors": processing_jobs[job_id]["errors"]
        }
        
        # 8. Save Results to Disk
        results_path = results_dir / "results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(final_output, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*80}")
        print(f"JOB COMPLETED: {job_id}")
        print(f"{'='*80}")
        print(f"  Files Processed: {len(file_paths)}")
        print(f"  Valid LCs: {len(consolidator.lcs)}")
        print(f"  Valid Amendments: {sum(len(a) for a in consolidator.amendments.values())}")
        print(f"  Shipping Docs: {len(all_shipping_docs)}")
        print(f"  Unidentified: {len(all_unidentified)}")
        print(f"  Consolidated LCs: {len(consolidated_lcs)}")
        print(f"  Validation Issues: {validation_summary['validation_issues']}")
        print(f"{'='*80}\n")
        
        # 9. Mark Job as Completed
        processing_jobs[job_id]["status"] = "completed"
        processing_jobs[job_id]["message"] = "Processing completed successfully"
        processing_jobs[job_id]["completed_at"] = datetime.now().isoformat()
        processing_jobs[job_id]["results"] = final_output
        
    except Exception as e:
        error_msg = f"Critical error: {str(e)}"
        print(f"\n{'='*80}")
        print(f"JOB FAILED: {job_id}")
        print(f"Error: {error_msg}")
        print(f"{'='*80}\n")
        
        processing_jobs[job_id]["status"] = "failed"
        processing_jobs[job_id]["message"] = error_msg
        processing_jobs[job_id]["completed_at"] = datetime.now().isoformat()
        processing_jobs[job_id]["errors"].append(error_msg)


def _format_document_list(documents: List[dict], job_id: str, category: str) -> List[dict]:
    """Format document list with metadata and download links"""
    formatted = []
    
    for doc in documents:
        entry = {
            'source_file': doc.get('source_file', 'Unknown'),
            'page_reference': doc.get('page_reference', '?'),
            'page_count': doc.get('page_count', 1),
            'document_type': doc.get('document_type', 'UNKNOWN'),
            'category': category
        }
        
        # Add category-specific fields
        data = doc.get('data', {})
        
        if category == 'LC':
            entry['lc_number'] = _extract_lc_number_from_data(data)
            entry['issue_date'] = _extract_date(data.get('Date_of_Issue', ''))
            entry['applicant'] = data.get('Applicant', '')[:100] if data.get('Applicant') else ''
            entry['beneficiary'] = data.get('Beneficiary', '')[:100] if data.get('Beneficiary') else ''
        
        elif category == 'AMENDMENT':
            entry['lc_number'] = _extract_lc_number_from_data(data)
            entry['amendment_number'] = _extract_amendment_number(data.get('Amendment_Number', ''))
            entry['amendment_date'] = _extract_date(data.get('Amendment_Date', ''))
        
        elif category == 'UNIDENTIFIED':
            # Add validation failure reason if available
            if 'validation_failure' in doc:
                entry['validation_failure'] = doc['validation_failure']
            if 'original_type' in doc:
                entry['original_type'] = doc['original_type']
        
        # Download link (points to original file)
        filename = doc.get('source_file', 'unknown.pdf')
        entry['download_url'] = f"/api/download/original/{job_id}/{filename}"
        
        formatted.append(entry)
    
    return formatted


# API Endpoints

@app.get("/")
async def root():
    """API health check"""
    return {
        "service": "LC Processing API",
        "version": "2.0.0 (GOT-OCR Integrated + Validation)",
        "status": "online",
        "ocr_backend": "GOT-OCR2.0 (Remote)",
        "features": [
            "Multi-stage document classification",
            "Document validation and filtering",
            "Page validation and boundary detection",
            "LC-Amendment consolidation",
            "AI-powered text merging",
            "Superior OCR accuracy",
            "Misclassification detection"
        ],
        "endpoints": {
            "upload": "/api/upload",
            "status": "/api/status/{job_id}",
            "result": "/api/result/{job_id}",
            "download": "/api/download/{job_id}/{lc_number}",
            "list_jobs": "/api/jobs",
            "interface": "/interface"
        }
    }


@app.post("/api/upload")
async def upload_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    ocr_backend: str = "tesseract",  # Kept for compatibility but not used (GOT-OCR is used)
    force_ocr: bool = False
):
    """
    Upload LC documents for processing with GOT-OCR
    
    - **files**: List of LC/Amendment documents (PDF, images, etc.)
    - **ocr_backend**: (Legacy parameter - GOT-OCR is always used)
    - **force_ocr**: Force OCR even for digital PDFs
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    # Create job
    job_id = create_job_id()
    job_dir = get_job_dir(job_id)
    
    # Initialize job status
    processing_jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "message": "Files uploaded, queued for processing",
        "created_at": datetime.now().isoformat(),
        "completed_at": None,
        "files_processed": 0,
        "lcs_found": 0,
        "amendments_found": 0,
        "shipping_docs_found": 0,
        "unidentified_found": 0,
        "errors": []
    }
    
    # Save uploaded files
    file_paths = []
    for upload_file in files:
        file_path = job_dir / upload_file.filename
        await save_upload_file(upload_file, file_path)
        file_paths.append(file_path)
    
    # Start background processing
    background_tasks.add_task(process_lc_job, job_id, file_paths, ocr_backend)
    
    return {
        "job_id": job_id,
        "status": "pending",
        "message": f"Processing {len(files)} file(s) with GOT-OCR",
        "files": [f.filename for f in files],
        "status_url": f"/api/status/{job_id}",
        "result_url": f"/api/result/{job_id}"
    }


@app.get("/api/status/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get processing status for a job"""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JobStatus(**processing_jobs[job_id])


@app.get("/api/result/{job_id}")
async def get_job_result(job_id: str):
    """Get processing results for a completed job"""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = processing_jobs[job_id]
    
    if job["status"] == "pending" or job["status"] == "processing":
        return {
            "job_id": job_id,
            "status": job["status"],
            "message": job.get("message", "Job is still processing"),
            "progress": {
                "files_processed": job.get("files_processed", 0),
                "lcs_found": job.get("lcs_found", 0),
                "amendments_found": job.get("amendments_found", 0)
            }
        }
    
    if job["status"] == "failed":
        return {
            "job_id": job_id,
            "status": "failed",
            "message": job["message"],
            "errors": job["errors"]
        }
    
    # Return results
    results_file = get_results_dir(job_id) / "results.json"
    
    if not results_file.exists():
        raise HTTPException(status_code=500, detail="Results file not found")
    
    with open(results_file, "r", encoding="utf-8") as f:
        results = json.load(f)
    
    return results


@app.get("/api/download/{job_id}/{lc_number}")
async def download_consolidated_lc(job_id: str, lc_number: str):
    """Download consolidated LC JSON file"""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Sanitize LC number for filename
    safe_lc_number = sanitize_filename(lc_number)
    
    lc_file = get_results_dir(job_id) / "consolidated" / f"{safe_lc_number}_consolidated.json"
    
    if not lc_file.exists():
        raise HTTPException(status_code=404, detail=f"LC file not found: {safe_lc_number}")
    
    return FileResponse(
        lc_file,
        media_type="application/json",
        filename=f"{safe_lc_number}_consolidated.json"
    )


@app.get("/api/download/original/{job_id}/{filename}")
async def download_original_file(job_id: str, filename: str):
    """Download original uploaded file"""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    file_path = get_job_dir(job_id) / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        file_path,
        media_type="application/pdf",
        filename=filename
    )


@app.get("/api/jobs")
async def list_jobs():
    """List all processing jobs"""
    jobs_list = []
    for job_id, job_data in processing_jobs.items():
        jobs_list.append({
            "job_id": job_id,
            "status": job_data["status"],
            "created_at": job_data["created_at"],
            "completed_at": job_data.get("completed_at"),
            "files_processed": job_data["files_processed"],
            "lcs_found": job_data["lcs_found"],
            "amendments_found": job_data["amendments_found"],
            "shipping_docs_found": job_data.get("shipping_docs_found", 0),
            "unidentified_found": job_data.get("unidentified_found", 0)
        })
    
    # Sort by creation time (newest first)
    jobs_list.sort(key=lambda x: x["created_at"], reverse=True)
    
    return {
        "total_jobs": len(jobs_list),
        "jobs": jobs_list
    }


@app.delete("/api/job/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its associated files"""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Delete job files
    job_dir = get_job_dir(job_id)
    if job_dir.exists():
        shutil.rmtree(job_dir)
    
    # Delete results
    results_dir = get_results_dir(job_id)
    if results_dir.exists():
        shutil.rmtree(results_dir)
    
    # Remove from memory
    del processing_jobs[job_id]
    
    return {"message": f"Job {job_id} deleted successfully"}


@app.get("/api/lc/{job_id}/{lc_number}")
async def get_specific_lc(job_id: str, lc_number: str):
    """Get specific consolidated LC data"""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    safe_lc_number = sanitize_filename(lc_number)
    lc_file = get_results_dir(job_id) / "consolidated" / f"{safe_lc_number}_consolidated.json"
    
    if not lc_file.exists():
        raise HTTPException(status_code=404, detail="LC not found")
    
    with open(lc_file, "r", encoding="utf-8") as f:
        lc_data = json.load(f)
    
    return lc_data


@app.get("/interface", response_class=HTMLResponse)
async def get_ui():
    """
    Serves the web interface for the LC Processing System
    """
    html_path = Path("view/web_interface.html")
    
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Interface file not found in 'view' folder")
        
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    print("=" * 70)
    print("LC Processing API Server (GOT-OCR + Validation)")
    print("=" * 70)
    print("\n🚀 Features:")
    print("  • GOT-OCR2.0 for superior accuracy")
    print("  • Document validation and filtering")
    print("  • Multi-stage document classification")
    print("  • Automatic LC-Amendment consolidation")
    print("  • AI-powered text merging")
    print("  • Misclassification detection")
    print("\n🌐 Starting server on http://0.0.0.0:8000")
    print("\n📚 API Documentation:")
    print("  • Swagger UI: http://0.0.0.0:8000/docs")
    print("  • ReDoc: http://0.0.0.0:8000/redoc")
    print("  • Web Interface: http://0.0.0.0:8000/interface")
    print("\n⌨️  Press CTRL+C to stop")
    print("=" * 70)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)