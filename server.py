"""
ComplyTrade Pilot V2 — Main Server
====================================
FastAPI server orchestrating the 20-step pipeline for documentary credit
(Letter of Credit) compliance verification.

PURPOSE:
    This is the main entry point for the ComplyTrade system. It:
    1. Serves the web interface (HTML views for upload, checklist, compliance)
    2. Provides REST API endpoints for file upload, job status, and results
    3. Orchestrates the 20-step pipeline in the background

PIPELINE OVERVIEW (20 Steps in Two Phases):

    Phase 1 — Extraction & Classification (Steps 1-11):
      Step 1:  Page-Level Raw OCR — extract text from each PDF page using GLM-OCR
      Step 2:  OCR Text Cleaning — fix common OCR errors and normalize text
      Step 3:  Page Sequencing — group pages into logical document packets
      Step 4:  MT Identification — identify SWIFT message types (MT700, MT710, etc.)
      Step 5:  MT OCR Reconciliation — reconcile OCR with SWIFT field extraction
      Step 6:  Final LC Consolidation — merge amendments into a single Final LC
      Step 7:  Clause & Requirement Extraction — parse LC clauses and required documents
      Step 8:  Shipping Document Classification — classify non-LC docs (BL, Invoice, etc.)
      Step 9:  Shipping OCR Reconciliation — reconcile shipping doc OCR and fields
      Step 10: Traceability — build confidence metrics and data lineage
      Step 11: Human Review Gate — present results for human approval

    Phase 2 — Verification & Reporting (Steps 12-20):
      Step 12: Clause Decomposition — break LC clauses into individual conditions (VLM)
      Step 13: Row Construction — build 5-column verification table structure
      Step 14: Verification — check each condition against actual documents (VLM + code)
      Step 15: Non-Compliance Classification — classify non-checkable clauses
      Step 16: Confidence Review — escalate low-confidence results to REVIEW
      Step 17: Cross-Clause Dependencies — resolve F47A overrides on F46A requirements
      Step 18: Threading — parallel clause processing (currently inline)
      Step 19: Consolidation — merge all results into final structure
      Step 20: Report Generation — produce the PDF compliance report

AI MODELS:
    - GLM-OCR @ http://10.20.10.3:8001 — Raw text extraction from PDF page images (Step 1)
    - Qwen 2.5-VL-7B @ http://10.20.10.3:8000 — Classification, decomposition, verification,
      and cross-document checks (Steps 4, 8, 12, 14)

TRADE FINANCE CONTEXT:
    A Letter of Credit (LC) is a bank guarantee used in international trade. The
    issuing bank promises to pay the beneficiary (exporter) if they present
    shipping documents that comply with all LC conditions. This system automates
    the document examination process that bank trade finance operations (TFO)
    departments perform manually.

    Key terms:
    - MT700 = SWIFT message format for issuing an LC
    - MT710 = Advice of a Third Bank's LC
    - F46A = Documents Required field (lists all required shipping documents)
    - F47A = Additional Conditions field (extra requirements)
    - UCP 600 = Uniform Customs and Practice for Documentary Credits (ICC rules)
    - ISBP 821 = International Standard Banking Practice (ICC guidelines)
"""

import os
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
import json
import uuid
import time
import traceback
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# Add project root to Python path so step imports work
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import (
    SERVER_HOST, SERVER_PORT, BUILD_TAG,
    UPLOAD_DIR, RESULTS_DIR, VIEW_DIR,
)
from config.database import init_database

# ── Step module imports ──
# Each step is a separate module in the steps/ directory with a run() function.
from steps import step01_raw_ocr
from steps import step02_ocr_cleaning
from steps import step03_sequencing
from steps import step04_mt_identification
from steps import step05_mt_reconciliation
from steps import step06_final_lc
from steps import step07_clause_extraction
from steps import step08_shipping_classification
from steps import step09_shipping_reconciliation
from steps import step10_traceability
from steps import step11_human_review
from steps import step12_decomposition
from steps import step13_row_construction
from steps import step14_verification
from steps import step15_non_compliance
from steps import step16_confidence_review
from steps import step17_cross_clause
from steps import step18_threading
from steps import step19_consolidation
from steps import step20_report

app = FastAPI(title="ComplyTrade Pilot V2", version="2.0.0")

# Ensure required directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(VIEW_DIR, exist_ok=True)

# In-memory job store — tracks pipeline status, progress messages, and results.
# In production, this would be backed by a database for durability.
_jobs = {}


# ══════════════════════════════════════════════════════════════
# API ROUTES
# ══════════════════════════════════════════════════════════════

@app.on_event("startup")
def startup():
    """Initialize database on server startup (non-critical — pipeline works without it)."""
    try:
        init_database()
    except Exception as e:
        print(f"[WARN] Database init failed: {e}")


@app.get("/version")
def version():
    """Return server version and build tag for health checks."""
    return {"build_tag": BUILD_TAG, "version": "2.0.0",
            "startup_time": datetime.now().isoformat()}


# ── Web Interface Routes ──
# These serve the HTML views for the browser-based UI.

@app.get("/")
@app.get("/interface")
def interface():
    """Serve the main web interface (document upload and processing view)."""
    html_path = os.path.join(VIEW_DIR, "web_interface.html")
    if os.path.exists(html_path):
        return HTMLResponse(open(html_path, 'r', encoding='utf-8').read())
    return HTMLResponse("<h1>ComplyTrade Pilot V2</h1><p>Web interface not found.</p>")


@app.get("/logo.png")
def serve_logo():
    """Serve the ComplyTrade logo for the web interface."""
    logo_path = os.path.join(VIEW_DIR, "logo.png")
    if os.path.exists(logo_path):
        return FileResponse(logo_path, media_type="image/png")
    raise HTTPException(404, "Logo not found")


@app.get("/extracted-text")
def extracted_text_page():
    html_path = os.path.join(VIEW_DIR, "extracted_text.html")
    if os.path.exists(html_path):
        return HTMLResponse(open(html_path, 'r', encoding='utf-8').read())
    raise HTTPException(404, "Extracted text viewer not found")


@app.get("/api/extracted-text/{job_id}")
def get_extracted_text(job_id: str):
    """Return all extracted text for every page — GLM raw, cleaned, VLM additions, final."""
    # Check both in-memory and disk — job may have been processed before server restart
    _results_dir = os.path.join(RESULTS_DIR, job_id)
    if job_id not in _jobs and not os.path.isdir(_results_dir):
        raise HTTPException(404, "Job not found")
    sr = _jobs.get(job_id, {}).get('step_results', {})
    s1 = sr.get('step01', {})
    s2 = sr.get('step02', {})
    s3 = sr.get('step03', {})

    # Fall back to disk if in-memory data is incomplete (old jobs)
    _results_dir = os.path.join(RESULTS_DIR, job_id)
    if not s1.get('pages'):
        _s1_file = os.path.join(_results_dir, 'step01', 'step01_result.json')
        if os.path.exists(_s1_file):
            with open(_s1_file, 'r', encoding='utf-8') as _f:
                s1 = json.load(_f)
    if not s2.get('pages'):
        _s2_file = os.path.join(_results_dir, 'step02', 'step02_result.json')
        if os.path.exists(_s2_file):
            with open(_s2_file, 'r', encoding='utf-8') as _f:
                s2 = json.load(_f)
    if not s3.get('packets'):
        _s3_file = os.path.join(_results_dir, 'step03', 'step03_result.json')
        if os.path.exists(_s3_file):
            with open(_s3_file, 'r', encoding='utf-8') as _f:
                s3 = json.load(_f)

    # Build page type lookup from Step 3
    _page_types = {}
    _page_copy = {}
    for pkt in s3.get('packets', []):
        if isinstance(pkt, dict):
            for pn in pkt.get('page_numbers', []):
                _page_types[pn] = pkt.get('document_type', 'unknown')
                _page_copy[pn] = pkt.get('copy_status', '')

    pages = []
    s1_pages = s1.get('pages', [])
    s2_pages = s2.get('pages', [])

    # Build s2 lookup
    _s2_map = {}
    for p in s2_pages:
        if isinstance(p, dict):
            _s2_map[p.get('page_number', 0)] = p

    total_glm = 0
    total_final = 0
    vlm_count = 0

    for p1 in s1_pages:
        if isinstance(p1, dict):
            pn = p1.get('page_number', 0)
            raw = p1.get('raw_text', '')
            p2 = _s2_map.get(pn, {})
            cleaned = p2.get('cleaned_text', raw)
            # VLM additions/replacements are now part of Step 2's cleaned_text.
            # Extract them from the corrections log.
            vlm_added = ''
            for _corr in p2.get('corrections', []):
                if isinstance(_corr, dict) and _corr.get('rule') in ('vlm_missing_text', 'vlm_full_extraction'):
                    vlm_added = _corr.get('corrected', '')
                    break
            # Final text = cleaned_text (already includes VLM additions from Step 2)
            final = cleaned

            total_glm += len(raw)
            total_final += len(str(final))
            if vlm_added:
                vlm_count += 1

            pages.append({
                'page_number': pn,
                'document_type': _page_types.get(pn, 'unknown'),
                'copy_status': _page_copy.get(pn, ''),
                'raw_text': raw,
                'glm_chars': len(raw),
                'cleaned_text': cleaned,
                'cleaned_chars': len(cleaned),
                'vlm_additions': vlm_added,
                'vlm_added': bool(vlm_added),
                'final_text': str(final),
                'final_chars': len(str(final)),
            })

    return {
        'total_pages': len(pages),
        'total_glm_chars': total_glm,
        'total_final_chars': total_final,
        'vlm_additions': vlm_count,
        'pages': pages,
    }


@app.get("/final-lc-viewer")
def final_lc_viewer():
    html_path = os.path.join(VIEW_DIR, "final_lc_viewer.html")
    if os.path.exists(html_path):
        return HTMLResponse(open(html_path, 'r', encoding='utf-8').read())
    raise HTTPException(404, "Final LC viewer not found")


@app.get("/checklist")
def checklist():
    """Serve the compliance checklist view (shows Final LC fields and required documents)."""
    html_path = os.path.join(VIEW_DIR, "checklist.html")
    if os.path.exists(html_path):
        return HTMLResponse(open(html_path, 'r', encoding='utf-8').read())
    raise HTTPException(404, "Checklist view not found")


@app.get("/compliance")
def compliance():
    """Serve the compliance rules view (shows verification rules and results)."""
    html_path = os.path.join(VIEW_DIR, "compliance_rules.html")
    if os.path.exists(html_path):
        return HTMLResponse(open(html_path, 'r', encoding='utf-8').read())
    raise HTTPException(404, "Compliance rules view not found")


# ── File Upload and Job Management ──

@app.post("/api/upload")
async def upload(background_tasks: BackgroundTasks, request: Request):
    """
    Upload a PDF document and start the processing pipeline.

    Accepts both 'file' and 'files' field names for backward compatibility
    with different UI versions. Creates a unique job ID, saves the PDF,
    and starts Phase 1 (Steps 1-11) as a background task.

    Returns the job_id for subsequent status/result queries.
    """
    # Accept both 'file' and 'files' field names, single or multiple
    form = await request.form()
    upload_file = None
    extra_files = []

    # Try 'files' first (old UI), then 'file' (new UI)
    files_field = form.getlist('files')
    if files_field:
        upload_file = files_field[0]
        extra_files = files_field[1:]
    elif 'file' in form:
        upload_file = form['file']

    if not upload_file:
        raise HTTPException(400, "No file uploaded")

    # Create unique job directory and save the uploaded PDF
    job_id = str(uuid.uuid4())
    filename = upload_file.filename or "document.pdf"
    job_dir = os.path.join(UPLOAD_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    pdf_path = os.path.join(job_dir, filename)
    with open(pdf_path, 'wb') as f:
        content = await upload_file.read()
        f.write(content)

    # Save additional files if multiple uploaded (e.g., LC + shipping docs in separate PDFs)
    for extra in extra_files:
        extra_path = os.path.join(job_dir, extra.filename or "extra.pdf")
        with open(extra_path, 'wb') as ef:
            ef.write(await extra.read())

    # Initialize job tracking
    _jobs[job_id] = {
        'status': 'uploaded', 'filename': filename, 'pdf_path': pdf_path,
        'progress': [], 'current_step': 0, 'total_steps': 20,
        'result': None, 'created_at': datetime.now().isoformat(),
        'step_results': {},  # Stores output from each step for inter-step data flow
    }

    # Start Phase 1 pipeline in background (non-blocking)
    background_tasks.add_task(_process_pipeline, job_id)
    return {"job_id": job_id, "filename": filename, "status": "processing"}


@app.post("/api/upload/{job_id}")
async def upload_additional(job_id: str, request: Request):
    """Upload additional files to an existing job (e.g., adding shipping documents later)."""
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found")
    form = await request.form()
    files_field = form.getlist('files')
    if not files_field and 'file' in form:
        files_field = [form['file']]
    job_dir = os.path.join(UPLOAD_DIR, job_id)
    for f in files_field:
        path = os.path.join(job_dir, f.filename or "extra.pdf")
        with open(path, 'wb') as out:
            out.write(await f.read())
    return {"status": "ok", "files_added": len(files_field)}


@app.get("/api/status/{job_id}")
def get_status(job_id: str):
    """
    Get current processing status for a job.

    Returns the current step number, recent progress messages, and overall status.
    The web UI polls this endpoint to show real-time progress.
    """
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found")
    job = _jobs[job_id]
    import re as _re
    # Map step numbers to UI stage names
    _step_stages = {
        1: 'ocr', 2: 'ocr', 3: 'classification', 4: 'classification',
        5: 'classification', 6: 'extraction_fields', 7: 'extraction_fields',
        8: 'classification', 9: 'classification', 10: 'validation',
        11: 'checklist', 12: 'validation', 13: 'validation', 14: 'validation',
        15: 'validation', 16: 'validation', 17: 'validation', 18: 'validation',
        19: 'summary', 20: 'saving',
    }
    # Build progress_log in format the old UI expects: [{message, stage}]
    progress_log = []
    for p in job['progress'][-200:]:
        _sm = _re.search(r'Step (\d+)', p)
        _step = int(_sm.group(1)) if _sm else job['current_step']
        progress_log.append({'message': p, 'stage': _step_stages.get(_step, 'processing')})

    return {
        'status': job['status'],
        'current_step': job['current_step'],
        'total_steps': job['total_steps'],
        'progress': job['progress'][-200:],
        'progress_log': progress_log,
        'filename': job['filename'],
        'message': job['progress'][-1] if job['progress'] else '',
    }


@app.get("/api/result/{job_id}")
def get_result(job_id: str):
    """
    Return results in format compatible with the old web UI.

    This endpoint builds a response that the legacy checklist.html and
    web_interface.html can consume. It transforms the internal step_results
    into the identified_objects / consolidated_lcs format the old UI expects.
    """
    _results_dir = os.path.join(RESULTS_DIR, job_id)
    if job_id not in _jobs and not os.path.isdir(_results_dir):
        raise HTTPException(404, "Job not found")
    job = _jobs.get(job_id, {'status': 'completed', 'step_results': {}})
    # Allow results even if verification (Phase 2) failed — Phase 1 data is still valid
    if job['status'] not in ('completed', 'review_pending', 'failed'):
        return {"status": job['status'], "message": "Processing not yet complete"}

    sr = job.get('step_results', {})

    # Load step results from disk if not in memory (server restarted)
    def _load_step(step_name):
        cached = sr.get(step_name, {})
        if cached and (cached.get('pages') or cached.get('packets') or cached.get('final_lc')
                       or cached.get('consolidated_fields') or cached.get('total_pages')):
            return cached
        _f = os.path.join(_results_dir, step_name, f'{step_name}_result.json')
        if os.path.exists(_f):
            with open(_f, 'r', encoding='utf-8') as fh:
                return json.load(fh)
        return cached

    s1 = _load_step('step01')
    s2 = _load_step('step02')
    s3 = _load_step('step03')
    s4 = sr.get('step04', {})
    s5 = sr.get('step05', {})
    s6 = _load_step('step06')
    s7 = _load_step('step07')
    s8 = _load_step('step08')
    s9 = _load_step('step09')

    # ── Build page text lookup from Step 2 ──
    s2_page_texts = {}
    for _pg in s2.get('pages', []):
        if isinstance(_pg, dict):
            _pn = _pg.get('page_number', 0)
            _txt = _pg.get('cleaned_text', _pg.get('raw_text', ''))
            if _pn and _txt: s2_page_texts[_pn] = _txt

    # ── Pre-compute LC data (needed by identified_objects below) ──
    flc = s6.get('final_lc', s6)
    cf = flc.get('consolidated_fields', {})
    import re as _re
    _FIELD_LABELS = {
        '20': r'^Documentary\s+Credit\s+Number\s*',
        '31C': r'^Date\s+of\s+Issue\s*',
        '31D': r'^Date\s+and\s+Place\s+of\s+Expiry\s*',
        '32B': r'^Currency\s+Code,?\s+Amount\s*',
        '39A': r'^Percentage\s+Credit\s+Amount\s+Tolerance\s*',
        '40A': r'^Form\s+of\s+Documentary\s+Credit\s*',
        '40E': r'^Applicable\s+Rules\s*',
        '42C': r'^Drafts\s+at\s*\.{0,3}\s*',
        '43P': r'^Partial\s+Shipments?\s*',
        '43T': r'^Trans[sh]?ipment\s*',
        '44C': r'^Latest\s+Date\s+of\s+Shipment\s*',
        '44E': r'^Port\s+of\s+Loading.*?Departure\s*',
        '44F': r'^Port\s+of\s+Discharge.*?Destination\s*',
        '46A': r'^Documents\s+Required\s*',
        '47A': r'^Additional\s+Conditions\s*',
        '48': r'^Period\s+for\s+Presentation.*?Days\s*',
        '50': r'^Applicant\s*',
        '59': r'^Beneficiary\s*(?:Name\s+and\s+Address:?\s*)?',
    }
    def _clean_field_value(tag, raw_val):
        if not raw_val or not isinstance(raw_val, str): return str(raw_val) if raw_val else ''
        val = raw_val.strip()
        lp = _FIELD_LABELS.get(tag, '')
        if lp: val = _re.sub(r'^' + lp, '', val, flags=_re.IGNORECASE).strip()
        val = _re.sub(r'^Identifier\s+Code:?\s*', '', val, flags=_re.IGNORECASE).strip()
        val = _re.sub(r'^Name\s+and\s+Address:?\s*', '', val, flags=_re.IGNORECASE).strip()
        return val
    _clean_cf = {t: _clean_field_value(t, v) if isinstance(v, str) else v for t, v in cf.items()}
    dc_number = _clean_field_value('20', flc.get('dc_number', cf.get('20', '')))
    if isinstance(dc_number, dict): dc_number = dc_number.get('value', str(dc_number))
    dc_number = str(dc_number).replace('\n', ' ').strip()
    _amendments = flc.get('amendment_count', 0)

    # ── Build identified_objects ──
    # LC/MT docs from Step 3, shipping docs from Step 8 (has VLM-extracted fields)
    identified_objects = []
    _lc_types = {'lc', 'amendment', 'mt700', 'mt707'}
    _skip_types = {'blank page', 'blank_page', 'endorsement page'}

    # 1. Add LC/MT packets from Step 3 (no VLM field extraction needed)
    for pkt in s3.get('packets', []):
        if not isinstance(pkt, dict): continue
        doc_type = (pkt.get('document_type', '') or '').lower()
        if doc_type in _skip_types: continue
        if doc_type not in _lc_types: continue  # shipping handled below
        pg_nums = pkt.get('page_numbers', [])
        pg_ref = f"{pg_nums[0]}-{pg_nums[-1]}" if len(pg_nums) > 1 else str(pg_nums[0]) if pg_nums else '?'
        text = '\n'.join(s2_page_texts.get(pn, '') for pn in pg_nums)
        stamps = pkt.get('stamps', [])
        signatures = pkt.get('signatures', [])
        # Use lowercase 'lc' / 'amendment' for UI badge compatibility
        _ot = pkt.get('document_type', 'LC').lower()
        if _ot == 'amendment': _ot = 'amendment'
        elif _ot in ('lc', 'mt700'): _ot = 'lc'
        identified_objects.append({
            'object_type': _ot,
            'category': _ot,  # 'lc' or 'amendment' — for UI tab placement
            'page_reference': pg_ref, 'pages': pg_nums,
            'data': {
                'document_type': _ot.upper(),
                'document_category': _ot,
                'classification_confidence': pkt.get('boundary_confidence', 0.95),
                'text_preview': text,
                'copy_status': pkt.get('copy_status', 'original'),
                'copy_label': pkt.get('copy_label', 'ORIGINAL'),
                'marking_status': pkt.get('marking_status', 'unsigned'),
                'has_stamps': bool(stamps), 'has_signatures': bool(signatures),
                'stamps': stamps, 'signatures': signatures,
                'seals': pkt.get('seals', []), 'logos': pkt.get('logos', []),
            }
        })

    # 1b. Add Final LC as identified document (for the Final LC tab in UI)
    if dc_number and _clean_cf:
        _lc_summary_parts = [
            f"DC Number: {dc_number}",
            f"Date of Issue: {_date_of_issue}" if '_date_of_issue' in dir() else '',
            f"Amount: {_clean_field_value('32B', cf.get('32B', ''))}",
            f"Applicant: {_clean_field_value('50', cf.get('50', ''))}",
            f"Beneficiary: {_clean_field_value('59', cf.get('59', ''))}",
            f"Expiry: {_clean_field_value('31D', cf.get('31D', ''))}",
            f"Port of Loading: {_clean_field_value('44E', cf.get('44E', ''))}",
            f"Port of Discharge: {_clean_field_value('44F', cf.get('44F', ''))}",
        ]
        _lc_summary = ' | '.join(p for p in _lc_summary_parts if p and ': ' in p and not p.endswith(': '))
        # Find LC packet pages
        _lc_pages = []
        for _p3 in s3.get('packets', []):
            if isinstance(_p3, dict) and (_p3.get('document_type', '').lower() in _lc_types):
                _lc_pages.extend(_p3.get('page_numbers', []))
        identified_objects.append({
            'object_type': 'final_lc',
            'category': 'final_lc',
            'page_reference': f"{_lc_pages[0]}-{_lc_pages[-1]}" if _lc_pages else '?',
            'pages': _lc_pages,
            'data': {
                'document_type': 'FINAL_LC',
                'document_category': 'final_lc',
                'DC_Number': dc_number,
                'classification_confidence': 0.99,
                'text_preview': '\n'.join(s2_page_texts.get(pn, '') for pn in _lc_pages),
                'copy_status': 'original',
                'copy_label': 'ORIGINAL',
                'marking_status': 'unsigned',
                'has_stamps': False, 'has_signatures': False,
                'stamps': [], 'signatures': [], 'seals': [], 'logos': [],
                'document_summary': _lc_summary,
                '_vlm_summary': {'dc_number': dc_number},
                'consolidated_fields': _clean_cf,
            }
        })

    # 2. Add shipping docs from Step 8 (has VLM-extracted summary, fields, stamps)
    # Use Step 3's document_type (Qwen classified correctly) instead of Step 8's VLM rename
    # Build Step 3 page->doc_type lookup
    _s3_page_type = {}
    for _p3 in s3.get('packets', []):
        if isinstance(_p3, dict):
            for _pn in _p3.get('page_numbers', []):
                _s3_page_type[_pn] = _p3.get('document_type', 'unknown')

    # Prefer Step 9 packets (has document-specific fields) over Step 8
    _shipping_pkts = s9.get('packets', s9.get('reconciled_packets', s8.get('packets', s8.get('classified_packets', []))))
    for pkt in _shipping_pkts:
        if not isinstance(pkt, dict): continue
        # Get Step 3's classification for this packet's first page
        _pages_field = pkt.get('original_pages', pkt.get('pages', []))
        _first_page = 0
        for _pg in (_pages_field if isinstance(_pages_field, list) else []):
            if isinstance(_pg, dict): _first_page = _pg.get('page_number', 0); break
            elif isinstance(_pg, int): _first_page = _pg; break
        # Use Step 3 name (correct) over Step 8 VLM name (sometimes wrong)
        doc_type = _s3_page_type.get(_first_page, pkt.get('document_type', 'unknown'))
        # Extract page numbers from classification dicts inside 'pages'
        _pages_field = pkt.get('original_pages', pkt.get('pages', []))
        pg_nums = []
        for _pg in (_pages_field if isinstance(_pages_field, list) else []):
            if isinstance(_pg, dict): pg_nums.append(_pg.get('page_number', 0))
            elif isinstance(_pg, int): pg_nums.append(_pg)
        if not pg_nums: pg_nums = pkt.get('page_numbers', [])
        pg_ref = f"{pg_nums[0]}-{pg_nums[-1]}" if len(pg_nums) > 1 else str(pg_nums[0]) if pg_nums else '?'
        # Get text from Step 2
        text = '\n'.join(s2_page_texts.get(pn, '') for pn in pg_nums)
        if not text: text = pkt.get('raw_text', pkt.get('cleaned_text', ''))
        # Get stamps from Step 8 VLM + Step 3 classifications
        stamps = list(pkt.get('stamps', []))
        signatures = list(pkt.get('signatures', []))
        seals = list(pkt.get('seals', []))
        logos = list(pkt.get('logos', []))
        for _pg in (_pages_field if isinstance(_pages_field, list) else []):
            if isinstance(_pg, dict):
                stamps.extend(_pg.get('stamps', []))
                signatures.extend(_pg.get('signatures', []))
                seals.extend(_pg.get('seals', []))
                logos.extend(_pg.get('logos', []))
        has_stamps = bool(stamps)
        has_sigs = bool(signatures)
        copy_status = pkt.get('copy_status', 'original')
        copy_label = pkt.get('copy_label', '')
        # Fix partial labels
        if copy_label.upper() in ('COP', 'COP.'): copy_label = 'COPY'; copy_status = 'copy'
        elif copy_label.upper().startswith('NON'): copy_label = 'NON-NEGOTIABLE'; copy_status = 'copy'
        elif copy_label.upper() in ('ORIG', 'ORIGINAL'): copy_label = 'ORIGINAL'
        marking = pkt.get('marking_status', 'unsigned')
        if has_stamps and has_sigs: marking = 'stamped_and_signed'
        elif has_stamps: marking = 'stamped'
        elif has_sigs: marking = 'signed'
        # Get document fields
        doc_fields = pkt.get('document_fields', pkt.get('extracted_fields', {}))
        if not isinstance(doc_fields, dict): doc_fields = {}
        # Also include top-level fields from Step 9
        for _tk in ('document_number', 'document_date', 'document_amount', 'issued_by', 'lc_reference',
                     'copy_status', 'copy_label', 'marking_status', 'document_summary'):
            _tv = pkt.get(_tk, '')
            if _tv and _tk not in doc_fields:
                doc_fields[_tk] = _tv
        # Build pipe-separated summary for UI "Show Document Info" button
        # Include ALL extracted fields — nothing should be hidden
        _sm = {
            'Description': doc_fields.get('description', ''),
            'Summary': pkt.get('document_summary', ''),
            'Doc No': pkt.get('document_number', doc_fields.get('document_number', doc_fields.get('invoice_number', doc_fields.get('bl_number', '')))),
            'Date': pkt.get('document_date', doc_fields.get('date', doc_fields.get('invoice_date', doc_fields.get('bl_date', '')))),
            'Amount': doc_fields.get('amount', doc_fields.get('invoice_amount', doc_fields.get('draft_amount', ''))),
            'Currency': doc_fields.get('currency', ''),
            'LC Ref': doc_fields.get('lc_reference', pkt.get('lc_reference', doc_fields.get('lc_number', ''))),
            'Consignee': doc_fields.get('consignee', ''),
            'Notify Party': doc_fields.get('notify_party', ''),
            'Shipper': doc_fields.get('shipper', doc_fields.get('exporter', '')),
            'Vessel': doc_fields.get('vessel_name', doc_fields.get('vessel', doc_fields.get('ocean_vessel', ''))),
            'Voyage': doc_fields.get('voyage_number', doc_fields.get('voyage', '')),
            'Port of Loading': doc_fields.get('port_of_loading', ''),
            'Port of Discharge': doc_fields.get('port_of_discharge', doc_fields.get('port_of_destination', '')),
            'Place of Receipt': doc_fields.get('place_of_receipt', ''),
            'Place of Delivery': doc_fields.get('place_of_delivery', doc_fields.get('final_destination', '')),
            'BL Date': doc_fields.get('bl_date', doc_fields.get('shipped_on_board_date', '')),
            'Shipped On Board': doc_fields.get('shipped_on_board', ''),
            'Freight': doc_fields.get('freight_status', doc_fields.get('freight', '')),
            'Number of Originals': doc_fields.get('number_of_originals', ''),
            'Drawee': doc_fields.get('drawee', ''),
            'Drawer': doc_fields.get('drawer', ''),
            'Tenor': doc_fields.get('tenor', ''),
            'Signed By': doc_fields.get('signed_by', ''),
            'Issued By': doc_fields.get('issued_by', pkt.get('issued_by', '')),
            'Beneficiary': doc_fields.get('beneficiary', ''),
            'Applicant': doc_fields.get('applicant', ''),
            'HS Code': doc_fields.get('hs_code', ''),
            'NTN': doc_fields.get('ntn_number', doc_fields.get('ntn', '')),
            'Incoterms': doc_fields.get('incoterms', ''),
            'Goods': doc_fields.get('goods_description', doc_fields.get('cargo_description', '')),
            'Quantity': doc_fields.get('quantity', ''),
            'Gross Weight': doc_fields.get('gross_weight', doc_fields.get('weight', '')),
            'Net Weight': doc_fields.get('net_weight', ''),
            'Packages': doc_fields.get('packages', doc_fields.get('number_of_packages', '')),
            'Country of Origin': doc_fields.get('country_of_origin', ''),
            'Insurance Amount': doc_fields.get('insurance_amount', ''),
            'Policy Number': doc_fields.get('policy_number', doc_fields.get('cover_note_number', '')),
            'Risks Covered': doc_fields.get('risks_covered', ''),
            'Certificate No': doc_fields.get('certificate_number', doc_fields.get('certificate_no', '')),
            'Inspection Result': doc_fields.get('inspection_result', doc_fields.get('test_results', '')),
        }
        # Also add ANY remaining doc_fields not already covered
        # Track which raw keys are already used to avoid duplicates
        _used_keys = set()
        for _label, _val in _sm.items():
            if _val and str(_val).strip():
                _used_keys.add(str(_val).strip().lower()[:50])
        for _fk, _fv in doc_fields.items():
            if not _fv or not str(_fv).strip(): continue
            # Skip if value already shown under a different label
            if str(_fv).strip().lower()[:50] in _used_keys: continue
            # Skip internal/meta keys
            if _fk in ('document_summary', 'copy_status', 'copy_label', 'marking_status', 'confidence',
                        'source_step', 'ambiguity_flag', 'ambiguity_notes'): continue
            _label = _fk.replace('_', ' ').title()
            _sm[_label] = _fv
            _used_keys.add(str(_fv).strip().lower()[:50])
        _parts = [f"{k}: {str(v).strip()}" for k, v in _sm.items()
                   if v and str(v).strip() and str(v).strip().lower() not in ('none','n/a','unknown','','nil')]
        data_dict = {
            'document_type': doc_type.upper().replace(' ', '_'),
            'document_category': doc_type, 'document_title': doc_type,
            'classification_confidence': pkt.get('match_confidence', 0.85),
            'text_preview': text,
            'copy_status': copy_status,
            'copy_label': copy_label or ('ORIGINAL' if copy_status == 'original' else 'COPY'),
            'copy_reason': f'{"Text contains " + copy_label + " marker" if copy_label else "Inferred"}',
            'marking_status': marking,
            'has_stamps': has_stamps, 'has_signatures': has_sigs,
            'has_seals': bool(seals), 'has_logos': bool(logos),
            'stamps': stamps, 'signatures': signatures, 'seals': seals, 'logos': logos,
            'stamp_details': '; '.join(s.get('text', s.get('description', '')) if isinstance(s, dict) else str(s) for s in stamps),
            'stamps_per_page': [{'page': pg_nums[0] if pg_nums else 0, 'stamps': stamps, 'signatures': signatures, 'seals': seals}] if (stamps or signatures or seals) else [],
            'copy_indicators': [{'text': copy_label, 'format': 'stamp', 'position': 'top-right'}] if copy_label else [],
            'document_summary': ' | '.join(_parts) if _parts else '',
            '_vlm_summary': {k.lower().replace(' ', '_'): v for k, v in _sm.items() if v and str(v).strip()},
        }
        for fk, fv in doc_fields.items():
            if fk not in data_dict and fv: data_dict[fk] = fv
        identified_objects.append({
            'object_type': doc_type,
            'category': 'supporting',
            'page_reference': pg_ref,
            'pages': pg_nums, 'data': data_dict,
        })

    # ── Build Final LC data (already computed above) ──
    # dc_number, _clean_cf, flc, cf all defined at the top of this function

    # Build consolidated_lcs list for the UI
    consolidated_lcs = []
    if dc_number:
        _clauses = flc.get('clauses', {})
        _46a_count = len(_clauses.get('46A', _clauses.get('F46A', [])))
        _47a_count = len(_clauses.get('47A', _clauses.get('F47A', [])))
        _date_raw = cf.get('31C', cf.get('F31C', ''))
        if isinstance(_date_raw, dict): _date_raw = _date_raw.get('value', str(_date_raw))
        _date_of_issue = _clean_field_value('31C', str(_date_raw))
        # Parse date: "251212 2025 Dec 12" -> "2025-12-12"
        _dm = _re.search(r'(\d{4})\s+(\w{3})\s+(\d{1,2})', str(_date_of_issue))
        if _dm:
            _months = {'Jan':'01','Feb':'02','Mar':'03','Apr':'04','May':'05','Jun':'06',
                       'Jul':'07','Aug':'08','Sep':'09','Oct':'10','Nov':'11','Dec':'12'}
            _date_of_issue = f"{_dm.group(1)}-{_months.get(_dm.group(2), '01')}-{int(_dm.group(3)):02d}"

        consolidated_lcs.append({
            'object_type': 'final_lc',
            # Top-level fields for UI compatibility
            'lc_number': dc_number,
            'dc_number': dc_number,
            'issue_date': str(_date_of_issue).replace('\n', ' ').strip(),
            'original_issue_date': str(_date_of_issue).replace('\n', ' ').strip(),
            'amendments_applied': _amendments,
            'documents_required': list(range(_46a_count)),  # UI checks array length
            'additional_conditions': list(range(_47a_count)),  # UI checks array length
            'download_url': f'/api/result/{job_id}',  # JSON download URL
            'view_final_lc_url': f'/final-lc-viewer?job_id={job_id}&lc={dc_number}',
            'data': {
                'DC_Number': dc_number,
                'Date_of_Issue': str(_date_of_issue).replace('\n', ' ').strip(),
                'total_amendments_applied': _amendments,
                'Documents_Required_count': _46a_count,
                'Additional_Conditions_count': _47a_count,
                'consolidated_fields': _clean_cf,
                'clauses': {k: [c if isinstance(c,dict) else {'text':str(c)} for c in v] if isinstance(v,list) else v for k,v in _clauses.items()},
            }
        })

    # Build type_summary — count of each document type for the UI dashboard
    # UI expects lowercase keys: ts.lc, ts.amendment, ts.final_lc
    type_summary = {}
    for obj in identified_objects:
        ot = obj.get('object_type', 'unknown')
        type_summary[ot] = type_summary.get(ot, 0) + 1

    # Count pages from Step 1
    total_pages = s1.get('total_pages', 0)

    return {
        'status': 'completed',
        'job_id': job_id,
        'total_pages': total_pages,
        'total_files': 1,
        'identified_objects': identified_objects,
        'consolidated_lcs': consolidated_lcs,
        'type_summary': type_summary,
        'documents': identified_objects,  # UI expects array, not count
        'lcs': len(consolidated_lcs),
        'amendments': flc.get('amendment_count', 0),
        'errors': [],
        'warnings': [],
        'elapsed_seconds': s1.get('elapsed', 0),
    }


@app.get("/api/report/{job_id}")
def get_report(job_id: str):
    """Download the PDF compliance report for a completed job."""
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found")
    results_dir = os.path.join(RESULTS_DIR, job_id)
    # Search for the generated PDF report file
    for f in Path(results_dir).rglob("*_compliance_report.pdf"):
        return FileResponse(str(f), media_type="application/pdf",
                            filename=f.name)
    raise HTTPException(404, "Report not generated yet")


@app.get("/api/final-lc/{job_id}")
def get_final_lc(job_id: str):
    """Get the Final LC data (consolidated from all amendments) for a job."""
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found")
    step_results = _jobs[job_id].get('step_results', {})
    return step_results.get('step06', {"error": "Final LC not yet generated"})


# ── Compatibility endpoints (for old views) ──
# These endpoints maintain backward compatibility with earlier UI versions.

@app.get("/api/final-lc-pdf/{job_id}/{lc_number}")
def get_final_lc_pdf(job_id: str, lc_number: str):
    """Generate/return Final LC PDF (compatibility endpoint)."""
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found")
    results_dir = os.path.join(RESULTS_DIR, job_id)
    # Check if already generated
    for f in Path(results_dir).rglob("*final_lc*.pdf"):
        return FileResponse(str(f), media_type="application/pdf", filename=f.name)
    # Generate on-demand from Step 6 data
    sr = _jobs[job_id].get('step_results', {})
    s6 = sr.get('step06', {})
    flc = s6.get('final_lc', s6)
    cf = flc.get('consolidated_fields', {})
    clauses = flc.get('clauses', {})
    if cf:
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.platypus import (
                SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, HRFlowable,
            )
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib import colors
            from reportlab.lib.colors import HexColor, white
            from reportlab.lib.units import mm
            from reportlab.lib.enums import TA_CENTER
            import re as _re

            dc_num = cf.get('20', flc.get('dc_number', 'unknown'))
            if isinstance(dc_num, dict): dc_num = dc_num.get('value', str(dc_num))
            dc_num = str(dc_num).replace('\n', ' ').strip()[:30]

            pdf_dir = os.path.join(results_dir, 'final_lc')
            os.makedirs(pdf_dir, exist_ok=True)
            pdf_path = os.path.join(pdf_dir, f"{dc_num}_final_lc.pdf")

            # ── Colors (matching old system) ──
            C_NAVY       = HexColor('#1a3a5c')
            C_BLUE       = HexColor('#2563eb')
            C_AMENDED_BG = HexColor('#fefce8')
            C_AMENDED_BD = HexColor('#ca8a04')
            C_AMENDED_TX = HexColor('#92400e')
            C_ROW_ALT    = colors.Color(0.97, 0.98, 0.99, alpha=0.3)  # semi-transparent for watermark visibility
            C_ROW_NORM   = colors.Color(1, 1, 1, alpha=0.3)            # semi-transparent white
            C_BORDER     = HexColor('#cbd5e1')
            C_HDR_FG     = HexColor('#ffffff')
            C_LABEL      = HexColor('#334155')
            C_VALUE      = HexColor('#0f172a')
            C_META       = HexColor('#64748b')
            C_DIVIDER    = HexColor('#e2e8f0')

            doc = SimpleDocTemplate(
                pdf_path, pagesize=A4,
                leftMargin=22*mm, rightMargin=22*mm,
                topMargin=22*mm, bottomMargin=22*mm,
            )
            page_width = A4[0] - 44*mm

            styles = getSampleStyleSheet()
            def _add_st(name, **kw):
                if name not in styles:
                    styles.add(ParagraphStyle(name, parent=styles['Normal'], **kw))
                return styles[name]

            _add_st('FTitle', fontSize=20, leading=26, textColor=C_BLUE, alignment=TA_CENTER,
                     fontName='Helvetica-Bold', spaceAfter=3)
            _add_st('FDCNum', fontSize=14, leading=18, textColor=C_NAVY, alignment=TA_CENTER,
                     fontName='Helvetica-Bold', spaceAfter=3)
            _add_st('FMeta', fontSize=9, leading=13, textColor=C_META, alignment=TA_CENTER,
                     fontName='Helvetica', spaceAfter=14)
            _add_st('FSect', fontSize=11, leading=15, textColor=C_HDR_FG, fontName='Helvetica-Bold',
                     spaceBefore=0, spaceAfter=0)
            _add_st('FLabel', fontSize=9, leading=13, textColor=C_LABEL, fontName='Helvetica-Bold')
            _add_st('FValue', fontSize=9, leading=13, textColor=C_VALUE, fontName='Helvetica',
                     wordWrap='CJK')
            _add_st('FAmBadge', fontSize=7, leading=10, textColor=C_AMENDED_TX, fontName='Helvetica-Bold',
                     alignment=TA_CENTER)
            _add_st('FFooter', fontSize=8, leading=11, textColor=C_META, alignment=TA_CENTER,
                     fontName='Helvetica', spaceAfter=0)
            _add_st('FClNum', fontSize=9, leading=13, textColor=C_META, fontName='Helvetica-Bold',
                     alignment=TA_CENTER)
            _add_st('FHdr', fontSize=9, leading=13, textColor=C_HDR_FG, fontName='Helvetica-Bold')

            elements = []

            # ── Footer/watermark callback ──
            def _flc_footer(canvas, doc_obj):
                canvas.saveState()
                pw, ph = A4
                # Watermark
                canvas.setFont('Helvetica-Bold', 60)
                canvas.setFillColor(colors.Color(0.85, 0.85, 0.85, alpha=0.35))
                canvas.translate(pw/2, ph/2)
                canvas.rotate(45)
                canvas.drawCentredString(0, 0, 'AiGenics')
                canvas.rotate(-45)
                canvas.translate(-pw/2, -ph/2)
                # Footer
                canvas.setFont('Helvetica', 7)
                canvas.setFillColor(C_META)
                ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                canvas.drawCentredString(pw/2, 12*mm,
                    f'ComplyTrade | AiGenics  |  Generated: {ts}  |  Page {doc_obj.page}')
                # Logo
                try:
                    logo_path = os.path.join(VIEW_DIR, 'logo.png')
                    if os.path.exists(logo_path):
                        canvas.drawImage(logo_path, pw - 45*mm, ph - 15*mm,
                                         width=30*mm, height=10*mm, preserveAspectRatio=True, mask='auto')
                except Exception:
                    pass
                canvas.restoreState()

            # ── Section header band ──
            def _section_header(label):
                tbl = Table([[Paragraph(label, styles['FSect'])]], colWidths=[page_width])
                tbl.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,-1), C_NAVY),
                    ('TOPPADDING', (0,0), (-1,-1), 7),
                    ('BOTTOMPADDING', (0,0), (-1,-1), 7),
                    ('LEFTPADDING', (0,0), (-1,-1), 10),
                    ('RIGHTPADDING', (0,0), (-1,-1), 10),
                ]))
                return tbl

            def _safe(text, limit=800):
                text = str(text)
                text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                text = text.replace('\n', '<br/>')
                return text

            # ── Title block ──
            elements.append(Paragraph("Final Letter of Credit", styles['FTitle']))
            elements.append(Paragraph(f"Documentary Credit No.: {dc_num or chr(8212)}", styles['FDCNum']))
            meta_parts = [f"Generated: {datetime.now().strftime('%d %b %Y  %H:%M')}"]
            elements.append(Paragraph("   |   ".join(meta_parts), styles['FMeta']))
            elements.append(HRFlowable(width="100%", thickness=1.5, color=C_BLUE, spaceAfter=16))

            # ── Tag label map ──
            _TAG_NAMES = {
                '20':'Documentary Credit Number','23':'Reference to Pre-Advice',
                '27':'Sequence of Total','31C':'Date of Issue',
                '31D':'Date and Place of Expiry','32B':'Currency Code, Amount',
                '39A':'Percentage Credit Amount Tolerance','40A':'Form of Documentary Credit',
                '40E':'Applicable Rules','41A':'Available With...By...',
                '41D':'Available With...By...','42C':'Drafts at...',
                '42A':'Drawee','42D':'Drawee','43P':'Partial Shipments',
                '43T':'Transshipment','44A':'Place of Taking in Charge',
                '44E':'Port of Loading','44F':'Port of Discharge',
                '44B':'Place of Final Destination','44C':'Latest Date of Shipment',
                '44D':'Shipment Period','48':'Period for Presentation',
                '49':'Confirmation Instructions','50':'Applicant',
                '51A':'Applicant Bank','52A':'Issuing Bank','52D':'Issuing Bank',
                '53A':'Reimbursing Bank','57A':'Advise Through Bank',
                '57D':'Advise Through Bank','59':'Beneficiary',
                '71D':'Charges','72':'Sender to Receiver Information',
            }
            _CLAUSE_TAGS = {'45A','45B','46A','46B','46C','47A','47B','78','72','79','77A','77B'}
            _CLAUSE_LABELS = {
                '45A':'Description of Goods and/or Services','45B':'Description of Goods (Additional)',
                '46A':'Documents Required','46B':'Documents Required (contd.)',
                '46C':'Other Documents','47A':'Additional Conditions',
                '47B':'Additional Conditions (contd.)','78':'Instructions to Banks',
                '72':'Sender to Receiver Information','79':'Narrative / Amendment Details',
                '77A':'Free Format Instructions','77B':'Discrepancy Procedures',
            }

            # ── Section 1: LC Details (single-value fields) ──
            elements.append(_section_header("LC Details"))
            elements.append(Spacer(1, 1))

            TAG_COL   = 14*mm
            FIELD_COL = 82*mm
            BADGE_COL = 18*mm
            VALUE_COL = page_width - FIELD_COL - TAG_COL - BADGE_COL

            detail_rows = []
            for tag in sorted(cf.keys()):
                if tag in _CLAUSE_TAGS:
                    continue
                val = cf[tag]
                amended = False
                if isinstance(val, dict):
                    amended = bool(val.get('amended', False))
                    val = val.get('value', str(val))
                # Strip SWIFT field labels from value
                val = str(val)
                import re as _re2
                for _lbl in ['Documentary Credit Number', 'Date of Issue', 'Date and Place of Expiry',
                             'Currency Code, Amount', 'Form of Documentary Credit', 'Applicable Rules',
                             'Partial Shipments', 'Transhipment', 'Transshipment', 'Latest Date of Shipment',
                             'Port of Loading/Airport of Departure', 'Port of Discharge/Airport of Destination',
                             'Description of Goods and/or Services', 'Documents Required', 'Additional Conditions',
                             'Period for Presentation in Days', 'Confirmation Instructions', 'Applicant',
                             'Beneficiary', 'Charges', 'Instructions to the Paying/Accepting/Negotiating Bank',
                             'Percentage Credit Amount Tolerance', 'Drafts at',
                             'Identifier Code:', 'Name and Address:', 'Date:', 'Place:', 'Currency:', 'Amount:',
                             'Days:', 'Narrative:', 'Number:', 'Total:', 'Code:']:
                    val = val.replace(_lbl, '').replace(_lbl.lower(), '')
                val = _re2.sub(r'Tolerance\s+\d:?\s*', '', val)
                val = val.replace('\n', ' ').strip()
                if not val.strip():
                    continue
                name = _TAG_NAMES.get(tag, tag.replace('_', ' ').title())
                badge = Paragraph('AMENDED', styles['FAmBadge']) if amended else Paragraph('', styles['FAmBadge'])
                detail_rows.append((
                    tag, name, val, badge, amended,
                ))

            if detail_rows:
                tbl_data = [[
                    Paragraph('Tag', styles['FHdr']),
                    Paragraph('Field', styles['FHdr']),
                    Paragraph('Value', styles['FHdr']),
                    Paragraph('', styles['FHdr']),
                ]]
                row_styles = [
                    ('BACKGROUND', (0,0), (-1,0), C_NAVY),
                    ('FONTSIZE', (0,0), (-1,-1), 9),
                    ('TOPPADDING', (0,0), (-1,-1), 6),
                    ('BOTTOMPADDING', (0,0), (-1,-1), 6),
                    ('LEFTPADDING', (0,0), (-1,-1), 8),
                    ('RIGHTPADDING', (0,0), (-1,-1), 6),
                    ('VALIGN', (0,0), (-1,-1), 'TOP'),
                    ('LINEBELOW', (0,0), (-1,-2), 0.4, C_DIVIDER),
                ]
                for i, (tag, name, val, badge, amended) in enumerate(detail_rows):
                    tbl_data.append([
                        Paragraph(tag, styles['FClNum']),
                        Paragraph(_safe(name), styles['FLabel']),
                        Paragraph(_safe(val), styles['FValue']),
                        badge,
                    ])
                    ri = i + 1
                    bg = C_AMENDED_BG if amended else (C_ROW_ALT if i % 2 else C_ROW_NORM)
                    row_styles.append(('BACKGROUND', (0, ri), (-1, ri), bg))
                    if amended:
                        row_styles.append(('LINEAFTER', (0, ri), (0, ri), 2.5, C_AMENDED_BD))

                t = Table(tbl_data, colWidths=[TAG_COL, FIELD_COL, VALUE_COL, BADGE_COL])
                t.setStyle(TableStyle(row_styles))
                elements.append(t)
            elements.append(Spacer(1, 16))

            # ── Section 2: Clause fields ──
            for tag in sorted(clauses.keys()):
                cls_list = clauses[tag]
                if not isinstance(cls_list, list):
                    continue
                label = _CLAUSE_LABELS.get(tag, tag.replace('_', ' ').title())
                elements.append(_section_header(f"Field {tag}: {label}"))
                elements.append(Spacer(1, 1))

                NUM_COL  = 14*mm
                TEXT_COL = page_width - NUM_COL - BADGE_COL
                cl_data = [[
                    Paragraph('<b>No.</b>', ParagraphStyle('_clhdr', parent=styles['FClNum'], textColor=C_HDR_FG)),
                    Paragraph('<b>Clause Text</b>', ParagraphStyle('_clhdr2', parent=styles['FValue'], textColor=C_HDR_FG, fontName='Helvetica-Bold')),
                    Paragraph('', styles['FAmBadge']),
                ]]
                cl_styles = [
                    ('BACKGROUND', (0,0), (-1,0), C_NAVY),
                    ('FONTSIZE', (0,0), (-1,-1), 9),
                    ('TOPPADDING', (0,0), (-1,-1), 7),
                    ('BOTTOMPADDING', (0,0), (-1,-1), 7),
                    ('LEFTPADDING', (0,0), (-1,-1), 7),
                    ('RIGHTPADDING', (0,0), (-1,-1), 6),
                    ('VALIGN', (0,0), (-1,-1), 'TOP'),
                    ('LINEBELOW', (0,0), (-1,-2), 0.4, C_DIVIDER),
                ]
                for i, cl in enumerate(cls_list):
                    if isinstance(cl, dict):
                        cnum = str(cl.get('clause_number', i+1))
                        ctext = cl.get('text', str(cl))
                        amended = bool(cl.get('amended', False))
                    else:
                        cnum = str(i+1)
                        ctext = str(cl)
                        amended = False
                    if not ctext.strip():
                        continue
                    badge = Paragraph('AMENDED', styles['FAmBadge']) if amended else Paragraph('', styles['FAmBadge'])
                    cl_data.append([
                        Paragraph(cnum, styles['FClNum']),
                        Paragraph(_safe(ctext, 1200), styles['FValue']),
                        badge,
                    ])
                    ri = i + 1
                    bg = C_AMENDED_BG if amended else (C_ROW_ALT if i % 2 else C_ROW_NORM)
                    cl_styles.append(('BACKGROUND', (0, ri), (-1, ri), bg))
                    if amended:
                        cl_styles.append(('LINEAFTER', (1, ri), (1, ri), 2.5, C_AMENDED_BD))

                if len(cl_data) > 1:
                    t = Table(cl_data, colWidths=[NUM_COL, TEXT_COL, BADGE_COL])
                    t.setStyle(TableStyle(cl_styles))
                    elements.append(t)
                elements.append(Spacer(1, 16))

            # ── Footer ──
            elements.append(HRFlowable(width="100%", thickness=0.8, color=C_DIVIDER, spaceAfter=6))
            elements.append(Paragraph(
                f"Generated by ComplyTrade AI  \u00b7  {datetime.now().strftime('%d %b %Y %H:%M')}  \u00b7  "
                "Consolidated Final LC incorporating all accepted amendments.",
                styles['FFooter']))

            doc.build(elements, onFirstPage=_flc_footer, onLaterPages=_flc_footer)
            return FileResponse(pdf_path, media_type="application/pdf", filename=f"{dc_num}_final_lc.pdf")
        except Exception as e:
            raise HTTPException(500, f"Failed to generate Final LC PDF: {e}")
    raise HTTPException(404, "No Final LC data available")


@app.get("/api/select-lc/{job_id}")
def select_lc(job_id: str):
    """Return list of LCs found in the document (usually just one)."""
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found")
    sr = _jobs[job_id].get('step_results', {})
    s6 = sr.get('step06', {})
    dc_num = ''
    if isinstance(s6, dict):
        cf = s6.get('consolidated_fields', {})
        dc_num = cf.get('DC_Number', {}).get('value', '') if isinstance(cf.get('DC_Number'), dict) else cf.get('DC_Number', '')
    return {"lcs": [dc_num] if dc_num else [], "selected": dc_num}


@app.post("/api/verify/{job_id}/{lc_number}")
def start_verification(job_id: str, lc_number: str, background_tasks: BackgroundTasks):
    """
    Start verification (Steps 12-20) for a specific LC.

    Called after the user reviews Phase 1 results and clicks "Verify" to
    begin the compliance checking phase.
    """
    _results_dir = os.path.join(RESULTS_DIR, job_id)
    if job_id not in _jobs:
        # Rebuild job from disk if results exist (server was restarted)
        if os.path.isdir(_results_dir):
            _jobs[job_id] = {
                'status': 'completed', 'filename': '', 'pdf_path': '',
                'progress': [], 'current_step': 11, 'total_steps': 20,
                'result': None, 'created_at': '', 'step_results': {},
            }
            # Load step results from disk
            for _sn in ['step01','step02','step03','step06','step07','step08','step09','step10','step11']:
                _sf = os.path.join(_results_dir, _sn, f'{_sn}_result.json')
                if os.path.exists(_sf):
                    with open(_sf, 'r', encoding='utf-8') as _fh:
                        _jobs[job_id]['step_results'][_sn] = json.load(_fh)
        else:
            raise HTTPException(404, "Job not found")
    job = _jobs[job_id]
    # Check if verification is currently running (not failed/completed)
    if job.get('review_approved') and job.get('status') == 'processing':
        return {"status": "already_running", "message": "Verification is currently in progress"}
    job['review_approved'] = True
    job['status'] = 'processing'
    verification_id = str(uuid.uuid4())
    job['verification_id'] = verification_id
    # Start Phase 2 pipeline in background
    background_tasks.add_task(_continue_verification, job_id)
    return {"verification_id": verification_id, "status": "processing"}


@app.get("/api/verify/status/{verification_id}")
def verify_status(verification_id: str):
    """Get verification status by verification_id."""
    for jid, job in _jobs.items():
        if job.get('verification_id') == verification_id:
            import re as _re
            _step_stages = {
                12: 'validation', 13: 'validation', 14: 'validation',
                15: 'validation', 16: 'validation', 17: 'validation',
                18: 'validation', 19: 'summary', 20: 'saving',
            }
            progress_log = []
            for p in job['progress'][-100:]:
                _sm = _re.search(r'Step (\d+)', p)
                _step = int(_sm.group(1)) if _sm else job['current_step']
                progress_log.append({'message': p, 'stage': _step_stages.get(_step, 'validation')})
            return {
                'status': 'completed' if job['status'] == 'completed' else 'processing',
                'current_step': job['current_step'],
                'progress': job['progress'][-100:],
                'progress_log': progress_log,
                'message': job['progress'][-1] if job['progress'] else '',
            }
    raise HTTPException(404, "Verification not found")


@app.get("/api/verify/result/{verification_id}")
def verify_result(verification_id: str):
    """Get verification result by verification_id."""
    for jid, job in _jobs.items():
        if job.get('verification_id') == verification_id:
            sr = job.get('step_results', {})
            s19 = sr.get('step19', {})
            # Build checklist-compatible response
            verifications = []
            for section in s19.get('sections', []):
                for row in section.get('rows', []):
                    verifications.append({
                        'check': row.get('condition_text', row.get('clause_ref', '')),
                        'status': row.get('compliance', 'review'),
                        'severity': 'critical' if row.get('compliance') == 'fail' else 'minor',
                        'detail': row.get('result', ''),
                        'swiftTag': row.get('clause_ref', '').split('-')[0] if row.get('clause_ref') else '',
                        'rule_reference': row.get('clause_ref', ''),
                        'document_checked': row.get('document_checked', ''),
                        'findings': row.get('findings', row.get('found_text', '')),
                    })
            return {
                "status": job['status'],
                "verification_id": verification_id,
                "result": job.get('result', {}),
                "verifications": verifications,
                "overall_decision": s19.get('overall_decision', 'REVIEW REQUIRED'),
                "total_pass": s19.get('total_pass', s19.get('pass_count', 0)),
                "total_fail": s19.get('total_fail', s19.get('fail_count', 0)),
                "total_review": s19.get('total_review', s19.get('review_count', 0)),
            }
    raise HTTPException(404, "Verification not found")


@app.get("/api/verify/history/{job_id}/{lc_number}")
def verify_history(job_id: str, lc_number: str):
    """Get verification history (placeholder for future audit trail feature)."""
    return {"history": []}


@app.post("/api/verify/update/{verification_id}")
def verify_update(verification_id: str):
    """Update verification (placeholder for future re-verification feature)."""
    return {"status": "ok"}


@app.get("/api/report/{job_id}/{lc_number}")
def get_report_by_lc(job_id: str, lc_number: str):
    """Get compliance report for a specific LC number."""
    results_dir = os.path.join(RESULTS_DIR, job_id)
    for f in Path(results_dir).rglob("*compliance_report*.pdf"):
        return FileResponse(str(f), media_type="application/pdf", filename=f.name)
    raise HTTPException(404, "Report not generated yet")


@app.get("/api/compliance/rulesets")
def get_rulesets():
    """
    Get available compliance rulesets.

    Currently supports UCP 600 (primary). ISBP 821 and ISP98 are listed
    but not yet fully implemented.

    TRADE FINANCE CONTEXT:
    - UCP 600 = Uniform Customs and Practice for Documentary Credits (ICC Publication 600)
    - ISBP 821 = International Standard Banking Practice for Examination of Documents (ICC)
    - ISP98 = International Standby Practices (for standby LCs)
    """
    return {"rulesets": [
        {"id": "ucp600", "name": "UCP 600", "active": True},
        {"id": "isbp821", "name": "ISBP 821", "active": False},
        {"id": "isp98", "name": "ISP98", "active": False},
    ]}


@app.get("/api/vessel/status/{job_id}")
def vessel_status(job_id: str):
    """Vessel tracking status (placeholder — not yet implemented)."""
    return {"status": "not_available", "message": "Vessel tracking not configured"}


@app.post("/api/vessel/track")
def vessel_track():
    """Vessel tracking request (placeholder — not yet implemented)."""
    return {"status": "not_available"}


@app.post("/api/cancel/{job_id}")
def cancel_job(job_id: str):
    """Cancel a running job. Only works if the job is currently processing."""
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found")
    job = _jobs[job_id]
    if job['status'] == 'processing':
        job['status'] = 'cancelled'
        job['progress'].append(f"[{datetime.now().strftime('%H:%M:%S')}] Job cancelled by user")
        return {"status": "cancelled", "message": "Job cancelled"}
    return {"status": job['status'], "message": "Job not in cancellable state"}


# ── Human Review endpoints (Step 11) ──
# These endpoints support the human review gate between Phase 1 and Phase 2.

@app.get("/api/review/{job_id}")
def get_review_data(job_id: str):
    """
    Get all data needed for human review: Final LC, shipping documents, traceability.

    The web UI uses this to display the document set for human inspection
    before the user approves and starts verification.
    """
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found")
    step_results = _jobs[job_id].get('step_results', {})
    return {
        'final_lc': step_results.get('step06', {}),
        'shipping_docs': step_results.get('step09', {}),
        'traceability': step_results.get('step10', {}),
    }


@app.post("/api/review/{job_id}/approve")
def approve_review(job_id: str):
    """
    Approve the document set and start Phase 2 (verification).

    Once approved, the pipeline continues from Step 12 in a background thread.
    This is the gate between Phase 1 (extraction) and Phase 2 (verification).
    """
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found")
    job = _jobs[job_id]
    if job['status'] != 'review_pending':
        return {"error": "Job not in review state"}
    job['review_approved'] = True
    job['status'] = 'processing'
    # Continue pipeline from Step 12 in a background thread
    import threading
    threading.Thread(target=_continue_verification, args=(job_id,), daemon=True).start()
    return {"status": "approved", "message": "Verification phase starting"}


# ══════════════════════════════════════════════════════════════
# PIPELINE ORCHESTRATOR
# ══════════════════════════════════════════════════════════════

def _to_dict(obj):
    """
    Recursively convert dataclass or object to dict for JSON serialization.

    Step modules may return dataclass objects, but inter-step data passing
    and JSON persistence require plain dicts. This function handles the conversion.
    """
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_dict(v) for v in obj]
    if hasattr(obj, '__dataclass_fields__'):
        from dataclasses import asdict
        return asdict(obj)
    return obj


def _process_pipeline(job_id: str):
    """
    Run Phase 1: Steps 1-11 (extraction + classification).

    This is the first phase of the pipeline, covering:
    - PDF OCR and text extraction
    - SWIFT message identification and Final LC assembly
    - Shipping document classification
    - Traceability and confidence scoring
    - Human review gate

    After Step 11, the pipeline pauses and waits for the user to review
    and approve the results before continuing to Phase 2.
    """
    job = _jobs[job_id]
    pdf_path = job['pdf_path']
    results_dir = os.path.join(RESULTS_DIR, job_id)
    os.makedirs(results_dir, exist_ok=True)

    def _p(msg):
        """Append timestamped progress message to job progress log."""
        job['progress'].append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

    try:
        job['status'] = 'processing'

        # ── Step 1: Raw OCR (GLM) ──
        # Extract raw text from each PDF page using the GLM-OCR model.
        # Each page image is sent to the model server for text extraction.
        job['current_step'] = 1
        _p("Step 1: Page-Level Raw OCR Extraction...")
        s1 = step01_raw_ocr.run(pdf_path, os.path.join(results_dir, 'step01'), _p)
        if s1.get('error'):
            raise Exception(f"Step 1 failed: {s1['error']}")
        job['step_results']['step01'] = _to_dict(s1)  # Full data — convert dataclasses to dicts
        _p(f"Step 1 done: {s1['total_pages']} pages in {s1['elapsed_seconds']}s")

        # ── Step 2: OCR Cleaning ──
        # Fix common OCR errors: B->8, G->6, misread characters, normalize whitespace.
        job['current_step'] = 2
        _p("Step 2: OCR Text Cleaning...")
        s2 = step02_ocr_cleaning.run(s1, os.path.join(results_dir, 'step02'), _p)
        job['step_results']['step02'] = _to_dict(s2)  # Full data — convert dataclasses to dicts

        # ── Step 3: Page Sequencing ──
        # Group pages into logical document packets (e.g., pages 3-5 = one BL).
        # Uses document boundary detection (headers, stamps, content changes).
        job['current_step'] = 3
        _p("Step 3: Page Sequencing & Document Packet Formation...")
        s3 = _to_dict(step03_sequencing.run(s2, os.path.join(results_dir, 'step03'), _p))
        job['step_results']['step03'] = s3

        # ── Step 4: MT Identification (uses Step 3 classification as base) ──
        # Step 3 already classified every page. Step 4 separates MT from shipping
        # and verifies the MT type using text patterns (F-tags, MT headers)
        job['current_step'] = 4
        _p("Step 4: MT Document Identification (from Step 3 classification)...")

        _lc_types = {'lc', 'amendment', 'mt700', 'mt707'}
        # Endorsement pages and blank pages are BACK SIDES of the previous document
        # They carry stamps, endorsements, signatures — merge into previous packet
        _back_page_types = {'blank page', 'blank_page', 'endorsement page'}
        _s3_packets = s3.get('packets', [])

        _mt_packets = []
        _shipping_packets = []
        _prev_packet = None
        for _pkt in _s3_packets:
            _dt = (_pkt.get('document_type', '') or '').lower()
            if _dt in _back_page_types:
                # This is the back side of the previous document — merge
                if _prev_packet:
                    _prev_packet['page_numbers'] = _prev_packet.get('page_numbers', []) + _pkt.get('page_numbers', [])
                    # Merge stamps/signatures from back page into previous doc
                    for _field in ('stamps', 'signatures', 'seals', 'logos'):
                        _prev_list = _prev_packet.get(_field, [])
                        _back_list = _pkt.get(_field, [])
                        if isinstance(_back_list, list):
                            _prev_list.extend(_back_list)
                            _prev_packet[_field] = _prev_list
                    _p(f"  Merged {_dt} (pg {_pkt.get('page_numbers', [])}) into previous {_prev_packet.get('document_type', '?')}")
                continue
            _pkt_copy = dict(_pkt)
            if _dt in _lc_types:
                _pkt_copy['mt_type'] = 'MT707' if 'amend' in _dt else 'MT700'
                _mt_packets.append(_pkt_copy)
                _prev_packet = _pkt_copy
            else:
                _pkt_copy['mt_type'] = 'shipping'
                _shipping_packets.append(_pkt_copy)
                _prev_packet = _pkt_copy

        s4 = {'packets': _mt_packets + _shipping_packets}
        job['step_results']['step04'] = s4
        _p(f"  MT/LC: {len(_mt_packets)}, Shipping: {len(_shipping_packets)}")

        # ── Step 5: Passthrough ──
        # VLM text completion for ALL pages (including MT/LC) is now handled in
        # Step 2 (OCR Cleaning). Step 2's cleaned_text is the single source of
        # truth for all downstream steps. Step 5 just passes through.
        job['current_step'] = 5
        _p("Step 5: Passthrough (VLM text review handled in Step 2 for all pages)")
        s5 = s4
        job['step_results']['step05'] = s5

        # ── Step 6: Final LC Consolidation ──
        # Build Final LC from MT packets using GLM text extracted by Step 1
        job['current_step'] = 6
        _p("Step 6: Final LC Consolidation...")
        # Build page text lookup from Step 2 (cleaned GLM text)
        _s6_input = dict(s5)
        _s6_input['page_texts'] = {}
        _s2_pages = s2.get('pages', [])
        _p(f"  Building page_texts from Step 2: {len(_s2_pages)} pages")
        for _pg in _s2_pages:
            if isinstance(_pg, dict):
                _pgn = _pg.get('page_number', 0)
                _txt = _pg.get('cleaned_text', _pg.get('raw_text', ''))
            elif hasattr(_pg, 'page_number'):
                _pgn = _pg.page_number
                _txt = _pg.cleaned_text or _pg.raw_text
            else:
                _pgn = 0
                _txt = ''
            if _pgn and _txt:
                _s6_input['page_texts'][_pgn] = _txt
        _p(f"  page_texts populated: {len(_s6_input['page_texts'])} pages with text")
        # Show first 100 chars of LC pages for debugging
        for _lpkt in _mt_packets:
            for _lpn in _lpkt.get('page_numbers', [])[:2]:
                _ltxt = _s6_input['page_texts'].get(_lpn, '')
                _p(f"  LC page {_lpn}: {len(_ltxt)} chars, starts with: {_ltxt[:80]}")
        s6 = _to_dict(step06_final_lc.run(_s6_input, os.path.join(results_dir, 'step06'), _p))
        job['step_results']['step06'] = s6

        # ── Step 7: Clause & Requirement Extraction ──
        # Parse F46A into required documents list, F47A into additional conditions.
        # This defines what the beneficiary must present to get paid.
        job['current_step'] = 7
        _p("Step 7: Final LC Clause & Requirement Extraction...")
        s7 = _to_dict(step07_clause_extraction.run(s6, os.path.join(results_dir, 'step07'), _p))
        job['step_results']['step07'] = s7  # Store full data — Step 12 needs the clause list

        # ── Step 8: Shipping Document Classification ──
        # Classify each non-LC packet: Bill of Lading, Commercial Invoice,
        # Insurance Policy, Certificate of Origin, Packing List, etc.
        job['current_step'] = 8
        _p("Step 8: Shipping Document Classification...")
        # Pass non-LC packets from Step 3 as shipping docs
        # Enrich each packet with page image paths and GLM text
        _img_dir = os.path.join(results_dir, 'step01', 'images')
        _shipping_from_s3 = []
        for _spkt in s3.get('packets', []):
            if not isinstance(_spkt, dict):
                continue
            _dt = (_spkt.get('document_type', '') or '').lower()
            if _dt in ('lc', 'amendment', 'blank page', 'blank_page', 'endorsement page'):
                continue
            _spkt_copy = dict(_spkt)
            # Add image paths
            _pg_nums = _spkt_copy.get('page_numbers', [])
            _img_paths = []
            for _pn in _pg_nums:
                _ip = os.path.join(_img_dir, f"page_{_pn:03d}.png")
                if os.path.exists(_ip):
                    _img_paths.append(_ip)
            _spkt_copy['page_image_paths'] = _img_paths
            # Add GLM text
            _texts = []
            for _pn in _pg_nums:
                _t = _s6_input.get('page_texts', {}).get(_pn, '')
                if _t:
                    _texts.append(_t)
            _spkt_copy['text'] = '\n'.join(_texts)
            _spkt_copy['cleaned_text'] = _spkt_copy['text']
            _shipping_from_s3.append(_spkt_copy)
        _s6_with_shipping = dict(s6)
        _s6_with_shipping['shipping_packets'] = _shipping_from_s3
        s8 = _to_dict(step08_shipping_classification.run(_s6_with_shipping, s7, os.path.join(results_dir, 'step08'), _p))
        job['step_results']['step08'] = s8  # Full data needed by Step 9

        # ── Step 9: Shipping OCR Reconciliation ──
        # Extract structured fields from shipping documents (amounts, dates,
        # consignee, notify party, etc.) and match to LC requirements.
        job['current_step'] = 9
        _p("Step 9: Shipping OCR Reconciliation...")
        s9 = _to_dict(step09_shipping_reconciliation.run(s8, s7, os.path.join(results_dir, 'step09'), _p))
        job['step_results']['step09'] = s9  # Full data needed by Step 14

        # ── Step 10: Traceability ──
        # Build confidence metrics and data lineage across all previous steps.
        # Identifies potential issues (low OCR confidence, ambiguous classifications).
        job['current_step'] = 10
        _p("Step 10: Traceability & Confidence Preservation...")
        s10 = _to_dict(step10_traceability.run(
            {'step01': s1, 'step02': s2, 'step03': s3, 'step04': s4,
             'step05': s5, 'step06': s6, 'step07': s7, 'step08': s8, 'step09': s9},
            os.path.join(results_dir, 'step10'), _p
        ))
        job['step_results']['step10'] = {'flags': s10.get('total_flags', 0)}

        # ── Step 11: Human Review Gate ──
        # Create a review session with all document packets for human inspection.
        # The pipeline pauses here until the user approves.
        job['current_step'] = 11
        _p("Step 11: Ready for Human Review...")
        s11 = _to_dict(step11_human_review.run(
            step7_result=s7, step9_result=s9, step10_result=s10,
            job_id=job_id, output_dir=os.path.join(results_dir, 'step11'),
            progress_callback=_p
        ))
        job['step_results']['step11'] = s11

        # Phase 1 complete — wait for user to review and start verification
        job['status'] = 'completed'
        _p("Phase 1 complete: Documents extracted, classified, and ready for review.")
        _p("Click 'Verify' on the checklist to start compliance verification (Steps 12-20).")

    except Exception as e:
        job['status'] = 'failed'
        _p(f"Pipeline FAILED: {str(e)}")
        _p(traceback.format_exc()[:500])


def _continue_verification(job_id: str):
    """
    Run Phase 2: Steps 12-20 (verification + report generation).

    This phase runs after the user approves the Phase 1 results. It:
    - Decomposes LC clauses into individual conditions (Step 12)
    - Verifies each condition against actual document content (Steps 13-14)
    - Handles non-checkable clauses and confidence review (Steps 15-16)
    - Resolves cross-clause dependencies like F47A overrides (Step 17)
    - Consolidates results into final structure (Step 19)
    - Generates the PDF compliance report (Step 20)
    """
    job = _jobs[job_id]
    results_dir = os.path.join(RESULTS_DIR, job_id)
    sr = job['step_results']

    def _p(msg):
        """Append timestamped progress message to job progress log."""
        job['progress'].append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

    try:
        # ── Step 12: Clause Decomposition ──
        # Send each LC clause to the Qwen VLM to break it into individual
        # checkable conditions (e.g., "signed" + "HS Code" + "3 copies").
        job['current_step'] = 12
        _p("Step 12: Clause-by-Clause Condition Decomposition...")
        s12 = _to_dict(step12_decomposition.run(sr.get('step07', {}), os.path.join(results_dir, 'step12'), _p))
        sr['step12'] = {'conditions': sum(len(c.get('conditions', [])) for c in s12.get('decomposed_clauses', []))}

        # ── Step 13: Row Construction ──
        # Build the 5-column verification table (blank worksheet).
        # Each condition becomes one row to be filled by Step 14.
        job['current_step'] = 13
        _p("Step 13: Verification Row Construction...")
        s13 = _to_dict(step13_row_construction.run(s12, os.path.join(results_dir, 'step13'), _p))
        sr['step13'] = {'rows': len(s13.get('rows', []))}

        # ── Step 14: Verification ──
        # Check each condition against the actual shipping documents.
        # Uses VLM for natural language conditions, Python code for dates/amounts.
        job['current_step'] = 14
        _p("Step 14: VLM Clause Verification...")
        # Pass Step 9 documents + Step 6 LC data + Step 2 page texts + Step 1 image paths
        _s14_docs = sr.get('step09', {})
        _s14_lc = sr.get('step06', {})
        # Build page_texts from Step 2 cleaned text (includes VLM additions)
        _s14_page_texts = {}
        for _pg in sr.get('step02', {}).get('pages', []):
            if isinstance(_pg, dict):
                _pgn = _pg.get('page_number', 0)
                _txt = _pg.get('cleaned_text', _pg.get('raw_text', ''))
            elif hasattr(_pg, 'page_number'):
                _pgn = _pg.page_number
                _txt = _pg.cleaned_text or _pg.raw_text
            else:
                continue
            if _pgn and _txt:
                _s14_page_texts[_pgn] = _txt
        _s14_lc['page_texts'] = _s14_page_texts
        _img_dir = os.path.join(results_dir, 'step01', 'images')
        _s14_lc['image_dir'] = _img_dir
        s14 = _to_dict(step14_verification.run(
            s13, _s14_docs, _s14_lc,
            os.path.join(results_dir, 'step14'), _p
        ))
        sr['step14'] = s14

        # ── Step 15: Non-Compliance Handling ──
        # Classify all LC clauses as checkable/informational/non-checkable.
        # Ensures bank obligations and sanctions clauses are not falsely failed.
        job['current_step'] = 15
        _p("Step 15: Non-Compliance & Non-Checkable Clauses...")
        s15 = _to_dict(step15_non_compliance.run(sr.get('step07', {}), os.path.join(results_dir, 'step15'), _p))
        sr['step15'] = s15

        # ── Step 16: Confidence Review ──
        # Escalate any low-confidence results to REVIEW status.
        # Prevents false passes/fails when the AI model is uncertain.
        job['current_step'] = 16
        _p("Step 16: Confidence-Based Review Escalation...")
        s16 = _to_dict(step16_confidence_review.run(
            s14.get('rows', []), s15.get('clause_status_map', {}),
            os.path.join(results_dir, 'step16'), _p
        ))
        sr['step16'] = {'escalated': s16.get('escalated_count', 0)}

        # ── Step 17: Cross-Clause Dependencies ──
        # Resolve F47A overrides that modify F46A requirements.
        # Example: F47A says "Charter Party BL acceptable" -> remove F46A BL type fail.
        job['current_step'] = 17
        _p("Step 17: Cross-Clause Dependency Handling...")
        s17 = _to_dict(step17_cross_clause.run(
            s16.get('rows', []),
            os.path.join(results_dir, 'step17'), _p
        ))
        sr['step17'] = {'overrides': s17.get('overrides_applied', 0)}

        # ── Step 18: Threading (already handled inline) ──
        # Parallel processing was handled within Steps 12 and 14 via ThreadPoolExecutor.
        # This step is a placeholder for future dedicated parallel processing.
        job['current_step'] = 18
        _p("Step 18: Multi-threaded processing complete (inline)")
        sr['step18'] = {'status': 'complete'}

        # ── Step 19: Consolidation ──
        # Merge all verified rows into sections (Key Terms, Document Requirements,
        # Additional Conditions, etc.) for the final report structure.
        job['current_step'] = 19
        _p("Step 19: Consolidating Verification Output...")
        s19 = _to_dict(step19_consolidation.run(
            s17.get('reconciled_rows', s17.get('rows', [])),
            os.path.join(results_dir, 'step19'), _p
        ))
        sr['step19'] = {
            'decision': s19.get('overall_decision', ''),
            'pass': s19.get('pass_count', 0),
            'fail': s19.get('fail_count', 0),
            'review': s19.get('review_count', 0),
        }

        # ── Step 20: Report Generation ──
        # Generate the final PDF compliance report with cover page,
        # executive summary, and clause-by-clause verification tables.
        job['current_step'] = 20
        _p("Step 20: Generating Final Compliance Report...")
        s20 = step20_report.run(
            s19, sr.get('step06', {}),
            os.path.join(results_dir, 'step20'), _p
        )
        sr['step20'] = {'report_path': s20.get('report_path', '')}

        # Pipeline complete — set final status and overall result
        job['status'] = 'completed'
        job['result'] = sr.get('step19', {})
        _p(f"Pipeline complete! Decision: {sr.get('step19', {}).get('decision', 'N/A')}")

    except Exception as e:
        job['status'] = 'failed'
        _p(f"Verification FAILED at Step {job['current_step']}: {str(e)}")
        _p(traceback.format_exc()[:500])


# ══════════════════════════════════════════════════════════════
# MAIN — Server Entry Point
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print(f"""
================================================================================
  ComplyTrade Pilot V2
  Build: {BUILD_TAG}
  GLM-OCR: http://10.20.10.3:8001
  Qwen VLM: http://10.20.10.3:8000
  Database: trade_finance_pilot @ localhost:5432
  Web UI: http://{SERVER_HOST}:{SERVER_PORT}/interface
  Checklist: http://{SERVER_HOST}:{SERVER_PORT}/checklist
  Compliance: http://{SERVER_HOST}:{SERVER_PORT}/compliance
  API Docs: http://{SERVER_HOST}:{SERVER_PORT}/docs
  Port: {SERVER_PORT}
================================================================================
""")
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
