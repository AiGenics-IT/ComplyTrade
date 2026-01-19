"""
Python Client for LC Processing API
Example usage of the LC Processing REST API
"""

import requests
import time
import json
from pathlib import Path
from typing import List, Dict, Optional


class LCProcessingClient:
    """Client for interacting with LC Processing API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize API client
        
        Args:
            base_url: Base URL of the API server
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def upload_documents(
        self, 
        file_paths: List[str], 
        ocr_backend: str = "tesseract",
        force_ocr: bool = False
    ) -> Dict:
        """
        Upload documents for processing
        
        Args:
            file_paths: List of file paths to upload
            ocr_backend: OCR engine to use (tesseract, easyocr, paddleocr)
            force_ocr: Force OCR even for digital PDFs
        
        Returns:
            Job information including job_id
        """
        files = []
        for file_path in file_paths:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            files.append(('files', (path.name, open(path, 'rb'))))
        
        params = {
            'ocr_backend': ocr_backend,
            'force_ocr': force_ocr
        }
        
        response = self.session.post(
            f"{self.base_url}/api/upload",
            files=files,
            params=params
        )
        
        # Close file handles
        for _, (_, file_handle) in files:
            file_handle.close()
        
        response.raise_for_status()
        return response.json()
    
    def get_status(self, job_id: str) -> Dict:
        """
        Get processing status for a job
        
        Args:
            job_id: Job ID returned from upload
        
        Returns:
            Job status information
        """
        response = self.session.get(f"{self.base_url}/api/status/{job_id}")
        response.raise_for_status()
        return response.json()
    
    def get_result(self, job_id: str) -> Dict:
        """
        Get processing results for a completed job
        
        Args:
            job_id: Job ID
        
        Returns:
            Processing results including consolidated LCs
        """
        response = self.session.get(f"{self.base_url}/api/result/{job_id}")
        response.raise_for_status()
        return response.json()
    
    def download_lc(self, job_id: str, lc_number: str, output_path: str) -> None:
        """
        Download consolidated LC JSON file
        
        Args:
            job_id: Job ID
            lc_number: LC number
            output_path: Path to save the file
        """
        response = self.session.get(
            f"{self.base_url}/api/download/{job_id}/{lc_number}"
        )
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            f.write(response.content)
    
    def get_lc_data(self, job_id: str, lc_number: str) -> Dict:
        """
        Get specific LC data
        
        Args:
            job_id: Job ID
            lc_number: LC number
        
        Returns:
            LC data as dictionary
        """
        response = self.session.get(f"{self.base_url}/api/lc/{job_id}/{lc_number}")
        response.raise_for_status()
        return response.json()
    
    def list_jobs(self) -> Dict:
        """
        List all processing jobs
        
        Returns:
            List of jobs with their status
        """
        response = self.session.get(f"{self.base_url}/api/jobs")
        response.raise_for_status()
        return response.json()
    
    def delete_job(self, job_id: str) -> Dict:
        """
        Delete a job and its files
        
        Args:
            job_id: Job ID to delete
        
        Returns:
            Deletion confirmation
        """
        response = self.session.delete(f"{self.base_url}/api/job/{job_id}")
        response.raise_for_status()
        return response.json()
    
    def process_and_wait(
        self, 
        file_paths: List[str],
        ocr_backend: str = "tesseract",
        force_ocr: bool = False,
        poll_interval: int = 2,
        timeout: int = 300
    ) -> Dict:
        """
        Upload documents and wait for processing to complete
        
        Args:
            file_paths: List of file paths to process
            ocr_backend: OCR engine to use
            force_ocr: Force OCR
            poll_interval: Seconds between status checks
            timeout: Maximum time to wait in seconds
        
        Returns:
            Processing results
        """
        # Upload
        print(f"Uploading {len(file_paths)} files...")
        upload_result = self.upload_documents(file_paths, ocr_backend, force_ocr)
        job_id = upload_result['job_id']
        print(f"Job ID: {job_id}")
        print(f"Status: {upload_result['status']}")
        
        # Wait for completion
        start_time = time.time()
        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Processing timeout after {timeout} seconds")
            
            status = self.get_status(job_id)
            print(f"Status: {status['status']} - {status['message']}")
            
            if status['status'] == 'completed':
                print("✓ Processing completed!")
                break
            elif status['status'] == 'failed':
                raise Exception(f"Processing failed: {status['message']}")
            
            time.sleep(poll_interval)
        
        # Get results
        results = self.get_result(job_id)
        return results


# Example usage
def example_usage():
    """Example of using the LC Processing API client"""
    
    # Initialize client
    client = LCProcessingClient(base_url="http://localhost:8000")
    
    # Example 1: Upload and process documents
    print("=" * 70)
    print("Example 1: Upload and Process Documents")
    print("=" * 70)
    
    files = [
        "LC_Swift.pdf",
        "LC_Amendment_1.pdf"
    ]
    
    try:
        results = client.process_and_wait(
            files,
            ocr_backend="tesseract",
            force_ocr=False
        )
        
        print(f"\nResults:")
        print(f"  Files Processed: {results['files_processed']}")
        print(f"  LCs Found: {results['lcs_found']}")
        print(f"  Amendments Found: {results['amendments_found']}")
        
        # Print consolidated LCs
        for lc_data in results['consolidated_lcs']:
            print(f"\n  LC: {lc_data['lc_number']}")
            print(f"    Original Issue Date: {lc_data['original_issue_date']}")
            print(f"    Amendments Applied: {lc_data['amendments_applied']}")
            print(f"    Additional Conditions: {len(lc_data['additional_conditions'])}")
            print(f"    Documents Required: {len(lc_data['documents_required'])}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "=" * 70)
    print("Example 2: Manual Upload and Status Checking")
    print("=" * 70)
    
    try:
        # Upload
        upload_result = client.upload_documents(files)
        job_id = upload_result['job_id']
        print(f"Job ID: {job_id}")
        
        # Check status periodically
        while True:
            status = client.get_status(job_id)
            print(f"Status: {status['status']}")
            
            if status['status'] in ['completed', 'failed']:
                break
            
            time.sleep(2)
        
        # Get results
        if status['status'] == 'completed':
            results = client.get_result(job_id)
            
            # Download specific LC
            for lc_data in results['consolidated_lcs']:
                lc_number = lc_data['lc_number']
                output_file = f"{lc_number}_consolidated.json"
                
                client.download_lc(job_id, lc_number, output_file)
                print(f"Downloaded: {output_file}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "=" * 70)
    print("Example 3: List All Jobs")
    print("=" * 70)
    
    try:
        jobs = client.list_jobs()
        print(f"Total Jobs: {jobs['total_jobs']}")
        
        for job in jobs['jobs'][:5]:  # Show first 5 jobs
            print(f"\nJob ID: {job['job_id']}")
            print(f"  Status: {job['status']}")
            print(f"  Created: {job['created_at']}")
            print(f"  LCs Found: {job['lcs_found']}")
            print(f"  Amendments Found: {job['amendments_found']}")
        
    except Exception as e:
        print(f"Error: {e}")


# Command-line interface
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("LC Processing API Client")
        print("=" * 70)
        print("\nUsage:")
        print("  python lc_api_client.py <file1> <file2> ... [options]")
        print("\nOptions:")
        print("  --api-url <url>       API base URL (default: http://localhost:8000)")
        print("  --ocr-backend <name>  OCR backend: tesseract, easyocr, paddleocr")
        print("  --force-ocr           Force OCR even for digital PDFs")
        print("  --output-dir <dir>    Directory to save results")
        print("\nExamples:")
        print("  python lc_api_client.py LC.pdf Amendment.pdf")
        print("  python lc_api_client.py *.pdf --ocr-backend easyocr")
        print("  python lc_api_client.py LC.pdf --force-ocr --output-dir ./results")
        sys.exit(0)
    
    # Parse arguments
    files = []
    api_url = "http://localhost:8000"
    ocr_backend = "tesseract"
    force_ocr = False
    output_dir = "./lc_results"
    
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        
        if arg == '--api-url':
            api_url = sys.argv[i + 1]
            i += 2
        elif arg == '--ocr-backend':
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
    
    if not files:
        print("Error: No files specified")
        sys.exit(1)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Process files
    print("=" * 70)
    print("LC Processing API Client")
    print("=" * 70)
    print(f"API URL: {api_url}")
    print(f"Files: {len(files)}")
    print(f"OCR Backend: {ocr_backend}")
    print(f"Output Directory: {output_dir}")
    print("=" * 70)
    print()
    
    client = LCProcessingClient(base_url=api_url)
    
    try:
        results = client.process_and_wait(
            files,
            ocr_backend=ocr_backend,
            force_ocr=force_ocr
        )
        
        job_id = results['job_id']
        
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"Files Processed: {results['files_processed']}")
        print(f"LCs Found: {results['lcs_found']}")
        print(f"Amendments Found: {results['amendments_found']}")
        print()
        
        # Download all consolidated LCs
        for lc_data in results['consolidated_lcs']:
            lc_number = lc_data['lc_number']
            output_file = Path(output_dir) / f"{lc_number}_consolidated.json"
            
            client.download_lc(job_id, lc_number, str(output_file))
            
            print(f"LC: {lc_number}")
            print(f"  Issue Date: {lc_data['original_issue_date']}")
            print(f"  Amendments: {lc_data['amendments_applied']}")
            print(f"  Conditions: {len(lc_data['additional_conditions'])}")
            print(f"  Saved to: {output_file}")
            print()
        
        if results.get('errors'):
            print("Errors:")
            for error in results['errors']:
                print(f"  - {error}")
        
        print("=" * 70)
        print("✓ Processing Complete")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)
