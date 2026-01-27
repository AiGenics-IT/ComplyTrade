"""
LC Processing REST API
FastAPI-based service for processing Letter of Credit documents
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from pydantic import BaseModel
import uvicorn
import os
import json
import uuid
from pathlib import Path
from datetime import datetime
import shutil

# Import our LC processing modules
# from lc_ocr import DocumentProcessor
# from lc_extractor import LCExtractor, LCConsolidator
from StandAloneSystem.lc_ocr import DocumentProcessor
from StandAloneSystem.lc_extractor import LCExtractor, LCConsolidator


# Initialize FastAPI app
app = FastAPI(
    title="LC Processing API",
    description="API for processing Letter of Credit documents and amendments",
    version="1.0.0"
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
UPLOAD_DIR = Path("/uploads/lc_uploads")
RESULTS_DIR = Path("/uploads/lc_results")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# In-memory storage for processing status
processing_jobs = {}


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
    errors: List[str] = []


class LCInfo(BaseModel):
    lc_number: str
    original_issue_date: Optional[str]
    sender: Optional[str]
    receiver: Optional[str]
    amendments_applied: int
    last_amendment_date: Optional[str]


class ConsolidatedLCResponse(BaseModel):
    lc_number: str
    original_issue_date: Optional[str]
    sender: Optional[str]
    receiver: Optional[str]
    amendments_applied: int
    additional_conditions_count: int
    documents_required_count: int
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


def process_lc_job(job_id: str, file_paths: List[Path], ocr_backend: str = "tesseract"):
    """Background task to process LC documents"""
    try:
        processing_jobs[job_id]["status"] = "processing"
        processing_jobs[job_id]["message"] = "Processing documents..."
        
        # Initialize processors
        doc_processor = DocumentProcessor(ocr_backend=ocr_backend)
        lc_extractor = LCExtractor()
        consolidator = LCConsolidator()
        
        results_dir = get_results_dir(job_id)
        
        # Process each file
        extracted_docs = []
        for file_path in file_paths:
            try:
                # Extract text
                text = doc_processor.process_document(str(file_path))
                
                if text.strip():
                    # Extract LC structure
                    lc_doc = lc_extractor.extract_from_text(text)
                    
                    if lc_doc.lc_number:
                        consolidator.add_document(lc_doc)
                        extracted_docs.append(lc_doc)
                        
                        processing_jobs[job_id]["files_processed"] += 1
                        
                        if lc_doc.document_type == "LC":
                            processing_jobs[job_id]["lcs_found"] += 1
                        else:
                            processing_jobs[job_id]["amendments_found"] += 1
                    
            except Exception as e:
                error_msg = f"Error processing {file_path.name}: {str(e)}"
                processing_jobs[job_id]["errors"].append(error_msg)
        
        # Consolidate all LCs
        consolidated_lcs = consolidator.get_all_consolidated()
        
        # Save results
        results = {
            "job_id": job_id,
            "processing_date": datetime.now().isoformat(),
            "files_processed": processing_jobs[job_id]["files_processed"],
            "lcs_found": processing_jobs[job_id]["lcs_found"],
            "amendments_found": processing_jobs[job_id]["amendments_found"],
            "consolidated_lcs": consolidated_lcs,
            "errors": processing_jobs[job_id]["errors"]
        }
        
        # Save consolidated results
        results_file = results_dir / "results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save individual consolidated LCs
        for lc_data in consolidated_lcs:
            lc_file = results_dir / f"{lc_data['lc_number']}_consolidated.json"
            with open(lc_file, "w") as f:
                json.dump(lc_data, f, indent=2, ensure_ascii=False)
        
        # Update job status
        processing_jobs[job_id]["status"] = "completed"
        processing_jobs[job_id]["message"] = "Processing completed successfully"
        processing_jobs[job_id]["completed_at"] = datetime.now().isoformat()
        processing_jobs[job_id]["results"] = results
        
    except Exception as e:
        processing_jobs[job_id]["status"] = "failed"
        processing_jobs[job_id]["message"] = f"Processing failed: {str(e)}"
        processing_jobs[job_id]["errors"].append(str(e))


# API Endpoints

@app.get("/")
async def root():
    """API health check"""
    return {
        "service": "LC Processing API",
        "version": "1.0.0",
        "status": "online",
        "endpoints": {
            "upload": "/api/upload",
            "status": "/api/status/{job_id}",
            "result": "/api/result/{job_id}",
            "download": "/api/download/{job_id}/{lc_number}",
            "list_jobs": "/api/jobs"
        }
    }


@app.post("/api/upload")
async def upload_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    ocr_backend: str = "tesseract",
    force_ocr: bool = True
):
    """
    Upload LC documents for processing
    
    - **files**: List of LC/Amendment documents (PDF, images, etc.)
    - **ocr_backend**: OCR engine to use (tesseract, easyocr, paddleocr)
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
        "message": "Files uploaded, waiting to process",
        "created_at": datetime.now().isoformat(),
        "completed_at": None,
        "files_processed": 0,
        "lcs_found": 0,
        "amendments_found": 0,
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
        "message": f"Processing {len(files)} files",
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
            "message": "Job is still processing. Please check status."
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
    
    with open(results_file, "r") as f:
        results = json.load(f)
    
    # Add download URLs
    for lc_data in results["consolidated_lcs"]:
        lc_number = lc_data["lc_number"]
        lc_data["download_url"] = f"/api/download/{job_id}/{lc_number}"
    
    return results


@app.get("/api/download/{job_id}/{lc_number}")
async def download_consolidated_lc(job_id: str, lc_number: str):
    """Download consolidated LC JSON file"""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    lc_file = get_results_dir(job_id) / f"{lc_number}_consolidated.json"
    
    if not lc_file.exists():
        raise HTTPException(status_code=404, detail="LC file not found")
    
    return FileResponse(
        lc_file,
        media_type="application/json",
        filename=f"{lc_number}_consolidated.json"
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
            "files_processed": job_data["files_processed"],
            "lcs_found": job_data["lcs_found"],
            "amendments_found": job_data["amendments_found"]
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
    
    lc_file = get_results_dir(job_id) / f"{lc_number}_consolidated.json"
    
    if not lc_file.exists():
        raise HTTPException(status_code=404, detail="LC not found")
    
    with open(lc_file, "r") as f:
        lc_data = json.load(f)
    
    return lc_data


if __name__ == "__main__":
    print("=" * 70)
    print("LC Processing API Server")
    print("=" * 70)
    print("\nStarting server on http://0.0.0.0:8000")
    print("\nAPI Documentation: http://0.0.0.0:8000/docs")
    print("Alternative Docs: http://0.0.0.0:8000/redoc")
    print("\nPress CTRL+C to stop")
    print("=" * 70)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
