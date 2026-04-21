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
    - GLM-OCR @ http://10.20.10.2:8001 — Raw text extraction from PDF page
      images (Step 1). Always used regardless of VLM_MODEL_SIZE.
    - Qwen VLM at QWEN_VLM_URL — Classification, decomposition, verification,
      and cross-document checks (Steps 2–14). Switch the entire VLM pipeline
      between 7B and 72B by setting VLM_MODEL_SIZE in config/settings.py:
        • "72B" → http://10.20.10.2:8085 (Qwen2.5-VL-72B-Instruct-AWQ)
        • "7B"  → http://10.20.10.3:8000 (Qwen2.5-VL-7B-Instruct)

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
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# Add project root to Python path so step imports work
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import (
    SERVER_HOST, SERVER_PORT, BUILD_TAG,
    UPLOAD_DIR, RESULTS_DIR, VIEW_DIR,
    GLM_OCR_URL, QWEN_VLM_URL, STEP_ENABLED,
    AUTH_ENABLED, AUTH_USERNAME, AUTH_PASSWORD,
)
from config.database import init_database
import base64 as _b64

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
from steps import step14_implicit

app = FastAPI(title="ComplyTrade Pilot V2", version="2.1.0")

# ── HTTP Basic Auth middleware ──
_auth_enabled = AUTH_ENABLED

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response as StarletteResponse

class BasicAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if not _auth_enabled:
            return await call_next(request)
        # Skip auth for favicon and static assets
        path = request.url.path
        if path in ('/favicon.ico', '/logo.png'):
            return await call_next(request)
        # Check Authorization header
        auth = request.headers.get('Authorization', '')
        if auth.startswith('Basic '):
            try:
                decoded = _b64.b64decode(auth[6:]).decode('utf-8')
                user, pwd = decoded.split(':', 1)
                if user == AUTH_USERNAME and pwd == AUTH_PASSWORD:
                    return await call_next(request)
            except Exception:
                pass
        # Return 401 with WWW-Authenticate header (browser shows login popup)
        return StarletteResponse(
            content='Unauthorized',
            status_code=401,
            headers={'WWW-Authenticate': 'Basic realm="ComplyTrade"'},
        )

app.add_middleware(BasicAuthMiddleware)

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
    return {"build_tag": BUILD_TAG, "version": "2.1.0",
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
        return HTMLResponse(
            open(html_path, 'r', encoding='utf-8').read(),
            headers={
                'Cache-Control': 'no-store, no-cache, must-revalidate, max-age=0',
                'Pragma': 'no-cache',
                'Expires': '0',
            },
        )
    raise HTTPException(404, "Extracted text viewer not found")


@app.get("/api/page-image/{job_id}/{page_num}")
def get_page_image(job_id: str, page_num: int, w: int = 0):
    """Serve a page image, optionally resized for faster loading."""
    img_path = os.path.join(RESULTS_DIR, job_id, 'step01', 'images', f'page_{page_num:03d}.png')
    if not os.path.exists(img_path):
        raise HTTPException(404, "Page image not found")
    if w and w < 2000:
        try:
            from PIL import Image as _PILImage
            import io as _io
            img = _PILImage.open(img_path)
            ratio = w / img.width
            new_h = int(img.height * ratio)
            img = img.resize((w, new_h), _PILImage.LANCZOS)
            buf = _io.BytesIO()
            img.save(buf, format='JPEG', quality=85)
            buf.seek(0)
            return StreamingResponse(buf, media_type="image/jpeg")
        except Exception:
            pass
    return FileResponse(img_path, media_type="image/png")


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

    # Load Step 8 for document summaries (page -> summary lookup)
    # Step 8 classification produces document_type, document_summary,
    # document_number, document_date, document_amount, issued_by, etc.
    _s8 = sr.get('step08', {})
    if not _s8.get('classified_packets'):
        _s8_file = os.path.join(_results_dir, 'step08', 'step08_result.json')
        if os.path.exists(_s8_file):
            try:
                with open(_s8_file, 'r', encoding='utf-8') as _f:
                    _s8 = json.load(_f)
            except Exception:
                _s8 = {}
    _page_summary = {}
    _page_vlm_summary = {}
    _page_stamps = {}  # page_number -> {stamps, signatures, seals, logos}
    for _cpkt in _s8.get('classified_packets', []):
        if not isinstance(_cpkt, dict):
            continue
        _doc_type = _cpkt.get('document_type', '')
        _doc_summary = _cpkt.get('document_summary', '')
        # NOTE: do NOT include 'document_type' here — step8's classification
        # is sometimes a WRONG LC-requirement match (e.g. it tags a Draft BoE
        # as 'Bill of Lading' because that was the required doc it tried to
        # fulfil). The CORRECT doc_type is already shown at the page level
        # (sourced from step3 → step9 reclassification chain). Showing step8's
        # type in the vlm_summary block causes a visible mismatch.
        _vlm_sum = {
            'document_number': _cpkt.get('document_number', ''),
            'document_date': _cpkt.get('document_date', ''),
            'document_amount': _cpkt.get('document_amount', ''),
            'issued_by': _cpkt.get('issued_by', ''),
            'copy_status': _cpkt.get('copy_status', ''),
            'lc_reference': _cpkt.get('lc_reference', ''),
            'match_confidence': _cpkt.get('match_confidence', ''),
            'description': _cpkt.get('document_summary', ''),
        }
        # Remove empty values
        _vlm_sum = {k: v for k, v in _vlm_sum.items() if v}
        _pkt_pages = _cpkt.get('page_numbers',
                               [p.get('page_number', 0) for p in _cpkt.get('original_pages', [])
                                if isinstance(p, dict)])
        if not _pkt_pages and _cpkt.get('page_image_paths'):
            # Try to extract page numbers from image paths
            import re as _re_pg
            for _ip in _cpkt.get('page_image_paths', []):
                _pm = _re_pg.search(r'page_(\d+)', str(_ip))
                if _pm:
                    _pkt_pages.append(int(_pm.group(1)))
        # Build stamps/signatures lookup per page
        _stamps = _cpkt.get('stamps', [])
        _signatures = _cpkt.get('signatures', [])
        _seals = _cpkt.get('seals', [])
        _logos = _cpkt.get('logos', [])
        # Also check original_pages for step3-level stamps
        for _op in _cpkt.get('original_pages', []):
            if isinstance(_op, dict):
                _stamps.extend(_op.get('stamps', []))
                _signatures.extend(_op.get('signatures', []))
                _seals.extend(_op.get('seals', []))

        for _pn in _pkt_pages:
            _summary_text = f"{_doc_type}"
            if _doc_summary:
                _summary_text += f" | {_doc_summary}"
            _page_summary[_pn] = _summary_text
            if _vlm_sum:
                _page_vlm_summary[_pn] = _vlm_sum
            if _stamps or _signatures or _seals or _logos:
                _page_stamps[_pn] = {
                    'stamps': _stamps,
                    'signatures': _signatures,
                    'seals': _seals,
                    'logos': _logos,
                }

    # Build page type lookup from Step 3 (including new bl_subtype + unified_summary)
    _page_types = {}
    _page_copy = {}
    _page_bl_subtype = {}      # page -> bl_subtype dict (BL packets only)
    _page_unified_summary = {} # page -> unified_summary dict
    _page_copy_label = {}
    _page_packet_pages = {}    # page -> list of ALL pages in the same packet
    _page_packet_id = {}       # page -> packet_id
    _page_original_type = {}   # page -> step3's ORIGINAL doc_type (before any step9 reclass)
    for pkt in s3.get('packets', []):
        if isinstance(pkt, dict):
            _bl_st = pkt.get('bl_subtype')
            _unified = pkt.get('unified_summary')
            _pkt_page_list = sorted(pkt.get('page_numbers', []))
            _pkt_id = pkt.get('packet_id', '')
            for pn in pkt.get('page_numbers', []):
                _s3_dt = pkt.get('document_type', 'unknown')
                _page_types[pn] = _s3_dt
                _page_original_type[pn] = _s3_dt
                _page_copy[pn] = pkt.get('copy_status', '')
                _page_copy_label[pn] = pkt.get('copy_label', '')
                _page_packet_pages[pn] = _pkt_page_list
                _page_packet_id[pn] = _pkt_id
                if _bl_st:
                    _page_bl_subtype[pn] = _bl_st
                if _unified:
                    _page_unified_summary[pn] = _unified

    # Step 9 may genuinely reclassify a packet (sets was_reclassified=True)
    # — e.g. Quality Certificate → Agents Certificate when the doc is really
    # an agent's cert the VLM misread. In that case we want the updated type
    # shown on the UI.
    # HOWEVER Step 8/9 also renames packets to the LC-required document name
    # for matching purposes (matched_requirement_name), which does NOT
    # indicate an actual classification change — it just means this packet
    # is being checked against that LC requirement. If we blindly used step9's
    # doc_type, ALL shipping packets can end up labelled with one LC-required
    # type (e.g. "Shipping Company Certificate") overwriting Step 3's rich,
    # correct classifications (Survey Report Cert, Cert of Quality, etc.).
    # RULE: ONLY override step3's type when step9 explicitly marks
    # was_reclassified=True AND previous_document_type differs from the new
    # type. Otherwise keep Step 3's authoritative classification.
    _s9 = sr.get('step09', {})
    if not _s9.get('packets') and not _s9.get('reconciled_packets'):
        _s9_file = os.path.join(_results_dir, 'step09', 'step09_result.json')
        if os.path.exists(_s9_file):
            try:
                with open(_s9_file, 'r', encoding='utf-8') as _f:
                    _s9 = json.load(_f)
            except Exception:
                pass
    _s9_pkts = _s9.get('packets', _s9.get('reconciled_packets', []))
    for _s9_pkt in _s9_pkts:
        if not isinstance(_s9_pkt, dict):
            continue
        if not _s9_pkt.get('was_reclassified'):
            continue  # not a real reclass — keep Step 3's type
        _s9_dt = _s9_pkt.get('document_type', '')
        _prev_dt = _s9_pkt.get('previous_document_type', '')
        if not _s9_dt or not _prev_dt or _s9_dt == _prev_dt:
            continue  # nothing actionable
        _pg_nums = _s9_pkt.get('page_numbers', [])
        if not _pg_nums:
            for _op in _s9_pkt.get('original_pages', []) or []:
                if isinstance(_op, dict):
                    _pn = _op.get('page_number')
                    if _pn:
                        _pg_nums.append(_pn)
        for _pn in _pg_nums:
            if _pn in _page_types:
                _page_types[_pn] = _s9_dt

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
                'copy_label': _page_copy_label.get(pn, ''),
                'raw_text': raw,
                'glm_chars': len(raw),
                'cleaned_text': cleaned,
                'cleaned_chars': len(cleaned),
                'vlm_additions': vlm_added,
                'vlm_added': bool(vlm_added),
                'final_text': str(final),
                'final_chars': len(str(final)),
                'document_summary': _page_summary.get(pn, ''),
                'vlm_summary': _page_vlm_summary.get(pn, {}),
                'stamps_info': _page_stamps.get(pn, {}),
                # NEW: Step 3 sub-call outputs (bl_subtype for BL packets, unified_summary for all)
                'bl_subtype': _page_bl_subtype.get(pn),
                'unified_summary': _page_unified_summary.get(pn, {}),
                # NEW: which packet this page belongs to + all pages of that packet
                # (so the UI can show "Summary based on N pages: [X, Y, Z]")
                'packet_id': _page_packet_id.get(pn, ''),
                'packet_pages': _page_packet_pages.get(pn, [pn]),
                # NEW: step3's original classification, exposed so the UI can
                # display "currently Agents Certificate (originally Quality Certificate)"
                # when Step 9 reclassified. document_type above reflects CURRENT type.
                'original_document_type': _page_original_type.get(pn, ''),
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


@app.get("/dashboard")
def dashboard_page():
    """Serve the analytics dashboard page."""
    html_path = os.path.join(VIEW_DIR, "dashboard.html")
    if os.path.exists(html_path):
        return HTMLResponse(open(html_path, 'r', encoding='utf-8').read())
    raise HTTPException(404, "Dashboard not found")


@app.get("/api/dashboard")
def get_dashboard():
    """Get dashboard analytics — total jobs, pass/fail rates, avg time, top discrepancies."""
    total = 0; compliant = 0; discrepant = 0; review = 0
    total_time = 0; recent = []; all_discs = {}

    # Scan all jobs from disk
    if os.path.isdir(RESULTS_DIR):
        for d in sorted(os.listdir(RESULTS_DIR), reverse=True):
            dpath = os.path.join(RESULTS_DIR, d)
            if not os.path.isdir(dpath) or d.startswith('_'):
                continue
            # Get LC number
            lc_num = ''; filename = ''; elapsed = 0; decision = ''; date_str = ''
            s06 = os.path.join(dpath, 'step06', 'step06_result.json')
            if os.path.exists(s06):
                try:
                    with open(s06, 'r', encoding='utf-8') as f:
                        s6d = json.load(f)
                    lc_num = s6d.get('dc_number', '')
                    elapsed = round(s6d.get('elapsed_seconds', 0))
                except Exception:
                    pass
            # Get verification result
            s19 = os.path.join(dpath, 'step19', 'step19_result.json')
            if os.path.exists(s19):
                try:
                    with open(s19, 'r', encoding='utf-8') as f:
                        s19d = json.load(f)
                    decision = s19d.get('overall_decision', '')
                    elapsed = round(s19d.get('elapsed_seconds', elapsed))
                    total += 1
                    if 'COMPLIANT' in decision.upper() and 'NON' not in decision.upper():
                        compliant += 1
                    elif 'DISCREPANT' in decision.upper() or 'NON' in decision.upper():
                        discrepant += 1
                    else:
                        review += 1
                    total_time += elapsed
                    # Collect discrepancies
                    for cf in s19d.get('critical_findings', []):
                        dt = cf.get('result', '')[:40]
                        if dt:
                            all_discs[dt] = all_discs.get(dt, 0) + 1
                except Exception:
                    pass
            # Get file info
            s01 = os.path.join(dpath, 'step01')
            if os.path.isdir(s01):
                s01f = os.path.join(s01, 'step01_result.json')
                if os.path.exists(s01f):
                    try:
                        with open(s01f, 'r', encoding='utf-8') as f:
                            s1d = json.load(f)
                        filename = s1d.get('filename', '')
                        date_str = s1d.get('timestamp', '')[:10]
                    except Exception:
                        pass
            if len(recent) < 10:
                recent.append({
                    'job_id': d, 'lc_number': lc_num, 'filename': filename,
                    'decision': decision or 'Processing', 'elapsed': elapsed,
                    'date': date_str, 'status': 'completed' if decision else 'processing',
                })

    top_discs = sorted(all_discs.items(), key=lambda x: -x[1])[:8]
    avg_time = round(total_time / max(total, 1) / 60, 1)  # in minutes

    return {
        'total_jobs': total,
        'compliant': compliant,
        'discrepant': discrepant,
        'review': review,
        'avg_time': avg_time,
        'top_discrepancies': [{'type': t, 'count': c} for t, c in top_discs],
        'recent_jobs': recent,
    }


@app.get("/vessel-tracking")
def vessel_tracking_page():
    """Serve the vessel tracking page."""
    html_path = os.path.join(VIEW_DIR, "vessel_tracking.html")
    if os.path.exists(html_path):
        return HTMLResponse(open(html_path, 'r', encoding='utf-8').read())
    raise HTTPException(404, "Vessel tracking page not found")


@app.get("/api/vessel/search")
async def vessel_search(name: str):
    """Search for a vessel by name using MyShipTracking API."""
    import httpx, re as _re
    _VESSEL_API_KEY = "f9NkQT*J!717@Vpj14A*yQJc9bWooMlSNq"
    _VESSEL_API_BASE = "https://api.myshiptracking.com/api/v2"
    clean = _re.sub(r'^(?:M/?V\.?\s+|MT\s+|SS\s+)', '', name.strip(), flags=_re.IGNORECASE).strip()
    if len(clean) < 3:
        return {"results": [], "message": "Name too short (min 3 chars)"}
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(f"{_VESSEL_API_BASE}/vessel/search",
                                    headers={"Authorization": f"Bearer {_VESSEL_API_KEY}"},
                                    params={"name": clean})
            data = resp.json()
            if data.get("status") == "success":
                return {"results": data.get("data", [])}
            return {"results": [], "message": data.get("message", "No results")}
    except Exception as e:
        return {"results": [], "error": str(e)}


@app.get("/api/vessel/details")
async def vessel_details(mmsi: int):
    """Get vessel position and details by MMSI."""
    import httpx
    _VESSEL_API_KEY = "f9NkQT*J!717@Vpj14A*yQJc9bWooMlSNq"
    _VESSEL_API_BASE = "https://api.myshiptracking.com/api/v2"
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(f"{_VESSEL_API_BASE}/vessel",
                                    headers={"Authorization": f"Bearer {_VESSEL_API_KEY}"},
                                    params={"mmsi": mmsi, "response": "extended"})
            data = resp.json()
            if data.get("status") == "success":
                return {"vessel": data.get("data", {})}
            return {"vessel": {}, "message": data.get("message", "Not found")}
    except Exception as e:
        return {"vessel": {}, "error": str(e)}


@app.get("/api/vessel/portcalls")
async def vessel_portcalls(mmsi: int, days: int = 30):
    """Get port call history for a vessel."""
    import httpx
    _VESSEL_API_KEY = "f9NkQT*J!717@Vpj14A*yQJc9bWooMlSNq"
    _VESSEL_API_BASE = "https://api.myshiptracking.com/api/v2"
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(f"{_VESSEL_API_BASE}/port/calls",
                                    headers={"Authorization": f"Bearer {_VESSEL_API_KEY}"},
                                    params={"mmsi": mmsi, "days": days})
            data = resp.json()
            if data.get("status") == "success":
                return {"port_calls": data.get("data", [])}
            return {"port_calls": [], "message": data.get("message", "No data")}
    except Exception as e:
        return {"port_calls": [], "error": str(e)}


@app.get("/document-compare")
def document_compare_page():
    """Serve the document comparison page."""
    html_path = os.path.join(VIEW_DIR, "document_compare.html")
    if os.path.exists(html_path):
        return HTMLResponse(open(html_path, 'r', encoding='utf-8').read())
    raise HTTPException(404, "Document comparison page not found")


@app.get("/report-viewer")
def report_viewer_page():
    """Serve the interactive compliance report viewer."""
    html_path = os.path.join(VIEW_DIR, "report_viewer.html")
    if os.path.exists(html_path):
        return HTMLResponse(open(html_path, 'r', encoding='utf-8').read())
    raise HTTPException(404, "Report viewer not found")


@app.get("/sanctions")
def sanctions_page():
    """Serve the sanctions list management page."""
    html_path = os.path.join(VIEW_DIR, "sanctions.html")
    if os.path.exists(html_path):
        return HTMLResponse(open(html_path, 'r', encoding='utf-8').read())
    raise HTTPException(404, "Sanctions page not found")


@app.get("/api/sanctions")
def get_sanctions():
    """Get all sanctions lists (banks, countries, goods) from database."""
    try:
        from config.database import get_connection
        conn = get_connection()
        cur = conn.cursor()
        result = {"banks": [], "countries": [], "goods": []}
        for cat in result.keys():
            cur.execute("SELECT value FROM sanctions WHERE category = %s ORDER BY id", (cat,))
            result[cat] = [row[0] for row in cur.fetchall()]
        cur.close()
        conn.close()
        return result
    except Exception as e:
        print(f"[WARN] Sanctions DB read failed: {e}")
        return {"banks": [], "countries": [], "goods": []}


@app.post("/api/sanctions")
async def save_sanctions(request: Request):
    """Save sanctions lists to database. Replaces all items per category."""
    body = await request.json()
    try:
        from config.database import get_connection
        conn = get_connection()
        conn.autocommit = True
        cur = conn.cursor()
        for cat in ['banks', 'countries', 'goods']:
            items = body.get(cat, [])
            # Clear existing items for this category
            cur.execute("DELETE FROM sanctions WHERE category = %s", (cat,))
            # Insert new items
            for item in items:
                if item and isinstance(item, str):
                    cur.execute("INSERT INTO sanctions (category, value) VALUES (%s, %s)", (cat, item))
        cur.close()
        conn.close()
        return {"status": "ok", "message": "Sanctions list saved to database"}
    except Exception as e:
        print(f"[ERROR] Sanctions DB write failed: {e}")
        raise HTTPException(500, f"Database error: {str(e)}")


@app.get("/compliance")
def compliance():
    """Serve the compliance rules view (shows verification rules and results)."""
    html_path = os.path.join(VIEW_DIR, "compliance_rules.html")
    if os.path.exists(html_path):
        return HTMLResponse(open(html_path, 'r', encoding='utf-8').read())
    raise HTTPException(404, "Compliance rules view not found")


@app.get("/jobs")
def jobs_page():
    """Serve the jobs dashboard page."""
    html_path = os.path.join(VIEW_DIR, "jobs.html")
    if os.path.exists(html_path):
        return HTMLResponse(open(html_path, 'r', encoding='utf-8').read())
    raise HTTPException(404, "Jobs dashboard not found")


@app.get("/api/jobs")
def list_jobs():
    """List all jobs with LC numbers and status."""
    jobs_list = []
    # From in-memory jobs
    for jid, job in _jobs.items():
        lc_numbers = []
        s06 = job.get('step_results', {}).get('step06', {})
        if isinstance(s06, dict):
            fl = s06.get('final_lc', s06)
            if isinstance(fl, dict):
                dc = fl.get('dc_number', '') or fl.get('consolidated_fields', {}).get('20', '')
                if dc:
                    lc_numbers.append(dc)
        # Sum elapsed from step results
        _elapsed = 0
        for _sk, _sv in job.get('step_results', {}).items():
            if isinstance(_sv, dict):
                _elapsed += _sv.get('elapsed_seconds', 0)
        jobs_list.append({
            'job_id': jid,
            'filename': job.get('filename', ''),
            'status': job.get('status', 'unknown'),
            'current_step': job.get('current_step', 0),
            'total_pages': job.get('total_pages', 0),
            'lc_numbers': lc_numbers,
            'created_at': job.get('created_at', ''),
            'elapsed_seconds': round(_elapsed, 1),
            'notes': job.get('notes', ''),
        })

    # Also scan results directory for jobs not in memory
    if os.path.isdir(RESULTS_DIR):
        for d in os.listdir(RESULTS_DIR):
            if d in _jobs:
                continue
            dpath = os.path.join(RESULTS_DIR, d)
            if not os.path.isdir(dpath):
                continue
            lc_numbers = []
            filename = ''
            total_pages = 0
            current_step = 0
            # Try to read step06 for LC number
            s06_path = os.path.join(dpath, 'step06', 'step06_result.json')
            if os.path.exists(s06_path):
                try:
                    with open(s06_path, 'r', encoding='utf-8') as f:
                        s06_data = json.load(f)
                    dc = s06_data.get('dc_number', '')
                    if dc:
                        lc_numbers.append(dc)
                except Exception:
                    pass
                current_step = max(current_step, 6)
            # Check what steps exist
            for sn in range(20, 0, -1):
                sp = os.path.join(dpath, f'step{sn:02d}')
                if not os.path.isdir(sp):
                    sp = os.path.join(dpath, f'step{sn}')
                if os.path.isdir(sp):
                    current_step = max(current_step, sn)
                    break
            # Try to get filename from uploads
            upload_dir = os.path.join(UPLOAD_DIR, d)
            if os.path.isdir(upload_dir):
                files = os.listdir(upload_dir)
                if files:
                    filename = files[0]
            # Get page count and timing from step01
            created_at = ''
            elapsed_seconds = 0
            s01_path = os.path.join(dpath, 'step01', 'step01_result.json')
            if os.path.exists(s01_path):
                try:
                    with open(s01_path, 'r', encoding='utf-8') as f:
                        s01_data = json.load(f)
                    total_pages = s01_data.get('total_pages', 0)
                    elapsed_seconds += s01_data.get('elapsed_seconds', 0)
                except Exception:
                    pass
            # Get created_at from directory modification time
            try:
                created_at = datetime.fromtimestamp(os.path.getctime(dpath)).isoformat()
            except Exception:
                pass
            # Sum elapsed from all steps
            for sn in range(2, 21):
                for sp_name in (f'step{sn:02d}', f'step{sn}'):
                    sp_file = os.path.join(dpath, sp_name, f'{sp_name}_result.json')
                    if os.path.exists(sp_file):
                        try:
                            with open(sp_file, 'r', encoding='utf-8') as f:
                                elapsed_seconds += json.load(f).get('elapsed_seconds', 0)
                        except Exception:
                            pass
                        break
            status = 'completed' if current_step >= 11 else 'processing' if current_step > 0 else 'uploaded'
            # Load notes from disk
            _notes = ''
            _notes_path = os.path.join(dpath, '_notes.txt')
            if os.path.exists(_notes_path):
                try:
                    with open(_notes_path, 'r', encoding='utf-8') as _nf:
                        _notes = _nf.read().strip()
                except Exception:
                    pass
            jobs_list.append({
                'job_id': d,
                'filename': filename,
                'status': status,
                'current_step': current_step,
                'total_pages': total_pages,
                'lc_numbers': lc_numbers,
                'created_at': created_at,
                'elapsed_seconds': round(elapsed_seconds, 1),
                'notes': _notes,
            })

    # Sort by most recent first (jobs with higher step counts first)
    jobs_list.sort(key=lambda j: j.get('current_step', 0), reverse=True)
    return {"jobs": jobs_list, "total": len(jobs_list)}


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str, request: Request):
    """Delete a job and all its results. Requires a reason."""
    reason = ''
    try:
        body = await request.json()
        reason = body.get('reason', '')
    except Exception:
        pass

    # Log deletion with reason
    deletion_log = os.path.join(RESULTS_DIR, 'deletion_log.jsonl')
    log_entry = json.dumps({
        'job_id': job_id,
        'reason': reason,
        'deleted_at': datetime.now().isoformat(),
        'had_results': os.path.isdir(os.path.join(RESULTS_DIR, job_id)),
    })
    with open(deletion_log, 'a', encoding='utf-8') as f:
        f.write(log_entry + '\n')

    # Remove from memory
    if job_id in _jobs:
        del _jobs[job_id]

    # Remove results directory
    import shutil
    results_path = os.path.join(RESULTS_DIR, job_id)
    if os.path.isdir(results_path):
        shutil.rmtree(results_path, ignore_errors=True)

    # Remove uploads directory
    upload_path = os.path.join(UPLOAD_DIR, job_id)
    if os.path.isdir(upload_path):
        shutil.rmtree(upload_path, ignore_errors=True)

    return {"status": "deleted", "job_id": job_id, "reason": reason}


@app.post("/api/jobs/{job_id}/clear-verification")
def clear_verification(job_id: str):
    """
    Clear ONLY the verification-stage outputs for a job (step12-step20).
    Pre-verification stages (step01-step09) are kept intact so re-running
    verification is fast (no OCR / classification re-run).

    Used by the Jobs page "Clear Verification" button to allow the user
    to re-trigger verification after fixing prompt / logic bugs without
    paying the OCR cost again.
    """
    import shutil
    job_dir = os.path.join(RESULTS_DIR, job_id)
    if not os.path.isdir(job_dir):
        raise HTTPException(404, f"Job results directory not found: {job_id}")

    removed = []
    # Verification + post-verification stages
    _stages = ['step12', 'step13', 'step14', 'step14b', 'step15',
               'step16', 'step17', 'step18', 'step19', 'step20']
    for stage in _stages:
        stage_path = os.path.join(job_dir, stage)
        if os.path.isdir(stage_path):
            shutil.rmtree(stage_path, ignore_errors=True)
            removed.append(stage)

    # Also clear any in-memory verification state for this job
    if job_id in _jobs:
        sr = _jobs[job_id].get('step_results', {})
        for stage in _stages:
            sr.pop(stage, None)

    return {
        "status": "cleared",
        "job_id": job_id,
        "removed_stages": removed,
        "kept_stages": ["step01", "step02", "step03", "step06", "step07", "step08", "step09"],
    }


@app.get("/checks")
def checks_config_page():
    """Serve the checks configuration page."""
    html_path = os.path.join(VIEW_DIR, "checks_config.html")
    if os.path.exists(html_path):
        return HTMLResponse(open(html_path, 'r', encoding='utf-8').read())
    raise HTTPException(404, "Checks config view not found")


@app.get("/api/checks/config")
def get_checks_config():
    """Get current checks configuration."""
    return step14_implicit.load_checks_config()


@app.put("/api/checks/config")
async def update_checks_config(request: Request):
    """Update checks configuration."""
    body = await request.json()
    step14_implicit.save_checks_config(body)
    return {"status": "saved", "checks": sum(1 for v in body.values() if v.get('enabled'))}


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
    # Always return Phase 1 data if available (steps 1-11).
    # Verification (Phase 2) runs in background — don't block the UI from showing LC data.
    # Only block if Phase 1 itself hasn't completed (step < 6 means no LC data yet).
    if job['status'] == 'processing' and job.get('current_step', 0) < 6:
        return {"status": job['status'], "current_step": job.get('current_step', 0),
                "message": "Phase 1 processing — LC data not yet available"}

    sr = job.get('step_results', {})

    # Load step results from disk if not in memory (server restarted)
    _has_overrides = job.get('overrides', {}) if job_id in _jobs else {}

    def _load_step(step_name):
        # If overrides exist, always reload from disk (overrides are saved to disk)
        if _has_overrides and step_name in ('step03', 'step08', 'step09'):
            _f = os.path.join(_results_dir, step_name, f'{step_name}_result.json')
            if os.path.exists(_f):
                with open(_f, 'r', encoding='utf-8') as fh:
                    return json.load(fh)
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
        '20': r'^(?:Documentary\s+Credit\s+Number|Sender\'?s?\s+Reference)\s*',
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
        # Truncate if another F-tag got merged in (e.g. "...F39A: Percentage...")
        _ftm = _re.search(r'F\d{2}[A-Z]?\s*:', val)
        if _ftm:
            val = val[:_ftm.start()].strip()
        # Strip "Page X of Y"
        val = _re.sub(r'\bPage\s+\d+\s+of\s+\d+\b\s*', '', val, flags=_re.IGNORECASE).strip()
        # Strip OCR garbage from blank pages
        val = _re.sub(r'There is no visible text.*?(?:clearly visible|another version)[.\s]*', '', val, flags=_re.IGNORECASE|_re.DOTALL).strip()
        val = _re.sub(r'The image appears to be blank[.\s]*', '', val, flags=_re.IGNORECASE).strip()
        # Strip CRITICAL RULES prompt leakage
        val = _re.sub(r'CRITICAL RULES:.*$', '', val, flags=_re.DOTALL).strip()
        return val
    _clean_cf = {t: _clean_field_value(t, v) if isinstance(v, str) else v for t, v in cf.items()}
    dc_number = _clean_field_value('20', flc.get('dc_number', cf.get('20', '')))
    if isinstance(dc_number, dict): dc_number = dc_number.get('value', str(dc_number))
    dc_number = str(dc_number).replace('\n', ' ').strip()
    _amendments = flc.get('amendment_count', 0)

    # ── Build identified_objects ──
    # Helper: format page numbers smartly
    # Continuous ranges use dash (12-14), non-continuous use comma (12, 10)
    def _format_page_ref(pg_nums: list) -> str:
        if not pg_nums:
            return '?'
        if len(pg_nums) == 1:
            return str(pg_nums[0])
        pgs = sorted(pg_nums)
        # Build ranges: [12,13,14] → "12-14", [10,12,14] → "10, 12, 14"
        ranges = []
        start = pgs[0]
        prev = pgs[0]
        for p in pgs[1:]:
            if p == prev + 1:
                prev = p
            else:
                ranges.append(f"{start}-{prev}" if prev > start else str(start))
                start = prev = p
        ranges.append(f"{start}-{prev}" if prev > start else str(start))
        return ', '.join(ranges)

    # LC/MT docs from Step 3, shipping docs from Step 8 (has VLM-extracted fields)
    identified_objects = []
    _lc_types = {'lc', 'amendment', 'mt700', 'mt707'}
    _swift_info_types = {'mt799', 'mt999', 'mt730', 'mt754', 'mt940', 'mt740', 'mt747', 'mt734'}
    _skip_types = {'blank page', 'blank_page', 'endorsement page'}

    # 1. Add LC/MT packets from Step 3 (no VLM field extraction needed)
    for pkt in s3.get('packets', []):
        if not isinstance(pkt, dict): continue
        doc_type = (pkt.get('document_type', '') or '').lower()
        if doc_type in _skip_types: continue
        if doc_type not in _lc_types and doc_type not in _swift_info_types: continue  # shipping handled below
        pg_nums = pkt.get('page_numbers', [])
        pg_ref = _format_page_ref(pg_nums)
        text = '\n'.join(s2_page_texts.get(pn, '') for pn in pg_nums)
        stamps = pkt.get('stamps', [])
        signatures = pkt.get('signatures', [])
        # Use lowercase 'lc' / 'amendment' for UI badge compatibility
        _ot = pkt.get('document_type', 'LC').lower()
        if _ot == 'amendment': _ot = 'amendment'
        elif _ot in ('lc', 'mt700'): _ot = 'lc'
        elif _ot in _swift_info_types: _ot = _ot  # keep type name, category set below
        # Set category for UI tab placement
        _cat = _ot
        if _ot in _swift_info_types:
            _cat = 'swift_messages'  # Must match JS cats key in web_interface.html
        identified_objects.append({
            'object_type': _ot,
            'category': _cat,
            'page_reference': pg_ref, 'pages': pg_nums,
            'data': {
                'document_type': _ot.upper(),
                'document_category': _cat,
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
            'page_reference': _format_page_ref(_lc_pages),
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
    _s3_page_bl_subtype = {}     # page -> bl_subtype dict (BL packets only)
    _s3_page_unified = {}        # page -> unified_summary dict
    _s3_page_copy_status = {}    # page -> copy_status (original / copy / non_negotiable)
    _s3_page_copy_label = {}     # page -> copy_label (ORIGINAL / COPY / NON-NEGOTIABLE / ...)
    for _p3 in s3.get('packets', []):
        if isinstance(_p3, dict):
            _bl = _p3.get('bl_subtype')
            _un = _p3.get('unified_summary')
            _cs = _p3.get('copy_status', '')
            _cl = _p3.get('copy_label', '')
            for _pn in _p3.get('page_numbers', []):
                _s3_page_type[_pn] = _p3.get('document_type', 'unknown')
                if _bl:
                    _s3_page_bl_subtype[_pn] = _bl
                if _un:
                    _s3_page_unified[_pn] = _un
                if _cs:
                    _s3_page_copy_status[_pn] = _cs
                if _cl:
                    _s3_page_copy_label[_pn] = _cl

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
        pg_ref = _format_page_ref(pg_nums)
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
        # Step 3c is the AUTHORITATIVE source for copy_status (VLM call dedicated
        # to stamp reading). Step 9 sometimes drops or defaults copy_status to
        # 'original', causing mismatch between Extracted Text page (step3) and
        # Identified Docs modal (step9). Prefer step3 value when step9 is
        # missing/default.
        _pkt_copy_status = pkt.get('copy_status', '')
        _pkt_copy_label = pkt.get('copy_label', '')
        # Step 3 copy_status for this packet's first page (authoritative)
        _s3_cs = _s3_page_copy_status.get(_first_page, '')
        _s3_cl = _s3_page_copy_label.get(_first_page, '')
        # Prefer step3 when step9 value is missing, empty, or default-only
        copy_status = _pkt_copy_status or _s3_cs or 'original'
        copy_label = _pkt_copy_label or _s3_cl or ''
        # If step3 has more specific info (non_negotiable/copy) but step9
        # says just 'original', trust step3
        if _s3_cs in ('non_negotiable', 'copy') and _pkt_copy_status in ('', 'original'):
            copy_status = _s3_cs
            copy_label = _s3_cl or copy_label
        # P164 — STEP 3 ALWAYS WINS for non_negotiable. If Step 3 detected
        # a NON-NEGOTIABLE stamp on the page, do NOT let downstream label
        # normalization flip it back to 'original'. Previously, a BL with
        # copy_status='non_negotiable' + copy_label='ORIGINAL' (the word
        # ORIGINAL sometimes printed alongside the NON-NEGOTIABLE stamp)
        # was being overwritten to copy_status='original' by the ORIG
        # branch below.
        _step3_is_nonneg = (_s3_cs == 'non_negotiable')
        # Normalize copy_label — preserve ordinal for FIRST/SECOND/THIRD ORIGINAL
        _cl_up = copy_label.upper()
        if _cl_up in ('COP', 'COP.'):
            copy_label = 'COPY'
            copy_status = 'copy'
        elif _cl_up.startswith('NON'):
            copy_label = 'NON-NEGOTIABLE'
            copy_status = 'non_negotiable'
        elif _step3_is_nonneg:
            # P164 — Step 3 says non_negotiable, keep it regardless of label
            copy_status = 'non_negotiable'
            if not copy_label or _cl_up in ('ORIG', 'ORIGINAL'):
                copy_label = 'NON-NEGOTIABLE'
        elif 'FIRST ORIGINAL' in _cl_up or '1ST ORIGINAL' in _cl_up:
            copy_label = 'FIRST ORIGINAL'
            copy_status = 'original'
        elif 'SECOND ORIGINAL' in _cl_up or '2ND ORIGINAL' in _cl_up:
            copy_label = 'SECOND ORIGINAL'
            copy_status = 'original'
        elif 'THIRD ORIGINAL' in _cl_up or '3RD ORIGINAL' in _cl_up:
            copy_label = 'THIRD ORIGINAL'
            copy_status = 'original'
        elif _cl_up in ('ORIG', 'ORIGINAL'):
            copy_label = 'ORIGINAL'
            copy_status = 'original'
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

        # Pull Step 3 sub-call results (bl_subtype, unified_summary) for this packet
        # (defined here so the merge block below AND the data_dict below can use them)
        _bl_subtype_s3 = _s3_page_bl_subtype.get(_first_page) or None
        _unified_summary_s3 = _s3_page_unified.get(_first_page) or {}

        # NEW — merge Step 3e unified_summary into the card so BL-specific and
        # cross-doc fields (NTN, HS, freight, charter party, dates_found,
        # amounts_found, references_found, parties_found) show on the result page.
        _us = _unified_summary_s3 or {}
        if isinstance(_us, dict) and _us:
            for _uk, _uv in _us.items():
                if _uk.startswith('_'):
                    continue
                if not _uv:
                    continue
                # Structured arrays get rendered as a compact multi-line string
                if isinstance(_uv, list) and _uv and isinstance(_uv[0], dict):
                    _rows = []
                    for _it in _uv:
                        if not isinstance(_it, dict):
                            continue
                        _role = str(_it.get('role', 'other')).replace('_', ' ')
                        _v = _it.get('value') or _it.get('name') or _it.get('raw') or ''
                        _cur = _it.get('currency', '')
                        if _cur:
                            _rows.append(f"{_role}: {_cur} {_v}".strip())
                        else:
                            _rows.append(f"{_role}: {_v}")
                    if _rows:
                        _label_struct = _uk.replace('_found', '').replace('_', ' ').title()
                        _sm[_label_struct] = " | ".join(_rows)
                    continue
                if isinstance(_uv, list):
                    _uv = ", ".join(str(x) for x in _uv if x)
                # Unwrap dicts — LLM sometimes returns typed fields as objects
                # (e.g. amount={"role":"draft_amount","currency":"USD","value":"..."})
                # rather than plain strings. Pick the primary value field.
                if isinstance(_uv, dict):
                    _inner = None
                    for _pref in ('value', 'name', 'text', 'raw', 'in_words', 'amount'):
                        if _uv.get(_pref):
                            _inner = _uv[_pref]
                            _cur = _uv.get('currency', '')
                            _uv = (f"{_cur} {_inner}".strip() if _cur else str(_inner))
                            break
                    if _inner is None:
                        # No recognized value key — skip rather than JSON-stringify
                        continue
                if not _uv or not str(_uv).strip():
                    continue
                _uv_key = str(_uv).strip().lower()[:50]
                if _uv_key in _used_keys:
                    continue
                _label_u = _uk.replace('_', ' ').title()
                if _label_u not in _sm:
                    _sm[_label_u] = _uv
                    _used_keys.add(_uv_key)
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
            # NEW — Step 3 sub-call outputs (bl_subtype only for BL; unified_summary for all)
            'bl_subtype': _bl_subtype_s3,
            'unified_summary': _unified_summary_s3,
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

    # Build type_summary — count of each document type for the UI dashboard.
    # Keys are LOWERCASED to collapse case-variants (e.g. "Certificate of
    # Quality and Weight..." and "CERTIFICATE OF QUALITY AND WEIGHT..." would
    # otherwise appear as separate buckets). UI expects lowercase keys anyway
    # (ts.lc / ts.amendment / ts.final_lc).
    type_summary = {}
    for obj in identified_objects:
        ot = obj.get('object_type', 'unknown') or 'unknown'
        key = ot.lower().strip()
        type_summary[key] = type_summary.get(key, 0) + 1

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
    # Search for the generated PDF report file — return newest
    all_reports = []
    for pattern in ("*_compliance_report.pdf", "ComplyTrade_Report*.pdf"):
        all_reports.extend(Path(results_dir).rglob(pattern))
    if all_reports:
        newest = max(all_reports, key=lambda f: f.stat().st_mtime)
        return FileResponse(str(newest), media_type="application/pdf", filename=newest.name)
    raise HTTPException(404, "Report not generated yet")


@app.get("/api/final-lc/{job_id}")
def get_final_lc(job_id: str):
    """Get the Final LC data (consolidated from all amendments) for a job."""
    # Try in-memory first
    if job_id in _jobs:
        step_results = _jobs[job_id].get('step_results', {})
        s06 = step_results.get('step06')
        if s06:
            return s06
    # Fall back to disk
    s06_path = os.path.join(RESULTS_DIR, job_id, 'step06', 'step06_result.json')
    if os.path.exists(s06_path):
        with open(s06_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    raise HTTPException(404, "Final LC not yet generated")


import queue as _queue_mod
_regen_logs = {}  # job_id -> queue.Queue of log messages

@app.get("/api/regenerate-logs/{job_id}")
async def regenerate_logs_sse(job_id: str):
    """SSE endpoint for streaming regeneration logs."""
    from starlette.responses import StreamingResponse
    import asyncio

    async def _stream():
        # Wait for queue to appear (POST creates it)
        _waited = 0
        while job_id not in _regen_logs and _waited < 30:
            await asyncio.sleep(0.5)
            _waited += 1
        q = _regen_logs.get(job_id)
        if not q:
            yield f"data: {json.dumps({'log': 'Waiting for server...'})}\n\n"
            return

        _idle = 0
        while _idle < 600:  # 10 min max
            try:
                msg = q.get_nowait()
                _idle = 0
                if msg == '__DONE__':
                    yield f"data: {json.dumps({'done': True})}\n\n"
                    return
                yield f"data: {json.dumps({'log': msg})}\n\n"
            except Exception:
                _idle += 1
                await asyncio.sleep(0.3)

    return StreamingResponse(_stream(), media_type="text/event-stream")


_regen_results = {}  # job_id -> result dict or None (in progress)

@app.post("/api/regenerate-final-lc/{job_id}")
def regenerate_final_lc(job_id: str):
    """
    Start Final LC regeneration in background. Returns immediately.
    Frontend watches /api/regenerate-logs/{job_id} SSE for progress,
    then polls /api/regenerate-result/{job_id} for the final result.
    """
    results_dir = os.path.join(RESULTS_DIR, job_id)
    if not os.path.isdir(results_dir):
        raise HTTPException(404, "Job results not found")

    s2_path = os.path.join(results_dir, 'step02', 'step02_result.json')
    s3_path = os.path.join(results_dir, 'step03', 'step03_result.json')
    if not os.path.exists(s2_path) or not os.path.exists(s3_path):
        raise HTTPException(400, "Steps 1-3 must be completed first")

    # Start in background thread
    import threading
    _regen_results[job_id] = None  # Mark as in-progress
    threading.Thread(target=_do_regenerate, args=(job_id,), daemon=True).start()
    return {"status": "started", "message": "Regeneration started. Watch logs via SSE."}


@app.get("/api/regenerate-result/{job_id}")
def get_regenerate_result(job_id: str):
    """Poll for regeneration result."""
    if job_id not in _regen_results:
        return {"status": "not_found"}
    result = _regen_results.get(job_id)
    if result is None:
        return {"status": "in_progress"}
    _regen_results.pop(job_id, None)
    return result


def _do_regenerate(job_id: str):
    """Background thread for FLC regeneration."""
    results_dir = os.path.join(RESULTS_DIR, job_id)
    s2_path = os.path.join(results_dir, 'step02', 'step02_result.json')
    s3_path = os.path.join(results_dir, 'step03', 'step03_result.json')

    with open(s2_path, 'r', encoding='utf-8') as f:
        s2 = json.load(f)
    with open(s3_path, 'r', encoding='utf-8') as f:
        s3 = json.load(f)

    # Build Step 4/5 input (same logic as pipeline)
    mt_packets = []
    shipping_packets = []
    for pkt in s3.get('packets', []):
        pkt_copy = dict(pkt)
        dt = (pkt.get('document_type', '') or '').lower()
        if any(x in dt for x in ['lc', 'letter of credit', 'amendment', 'mt7']):
            pkt_copy['mt_type'] = 'MT707' if 'amend' in dt else 'MT700'
            mt_packets.append(pkt_copy)
        else:
            pkt_copy['mt_type'] = 'shipping'
            shipping_packets.append(pkt_copy)

    s5_input = {'packets': mt_packets + shipping_packets, 'page_texts': {}}
    for p in s2.get('pages', []):
        pn = p.get('page_number', 0)
        text = p.get('cleaned_text', p.get('raw_text', ''))
        if pn and text:
            s5_input['page_texts'][pn] = text

    # Delete old step06 result
    s6_dir = os.path.join(results_dir, 'step06')
    if os.path.isdir(s6_dir):
        import shutil
        shutil.rmtree(s6_dir)
    os.makedirs(s6_dir, exist_ok=True)

    _regen_logs[job_id] = _queue_mod.Queue()
    def _log(msg):
        print(f"[Regenerate] {msg}")
        try:
            _regen_logs[job_id].put(msg)
        except Exception:
            pass

    # Clear steps 6-20
    import shutil as _shutil
    for _sn in ['step06', 'step07', 'step08', 'step09', 'step12', 'step13',
                'step14', 'step14b', 'step15', 'step16', 'step17',
                'step18', 'step19', 'step20']:
        _sp = os.path.join(results_dir, _sn)
        if os.path.isdir(_sp):
            _shutil.rmtree(_sp, ignore_errors=True)
        if job_id in _jobs:
            _jobs[job_id].get('step_results', {}).pop(_sn, None)

    # Re-run Step 6 (Final LC)
    try:
        os.makedirs(s6_dir, exist_ok=True)
        s6 = _to_dict(step06_final_lc.run(s5_input, s6_dir, _log))
    except Exception as e:
        raise HTTPException(500, f"Step 6 failed: {str(e)}")
    if job_id in _jobs:
        _jobs[job_id]['step_results']['step06'] = s6

    # Re-run Step 7 (Clause Extraction)
    try:
        s7_dir = os.path.join(results_dir, 'step07')
        os.makedirs(s7_dir, exist_ok=True)
        s7 = _to_dict(step07_clause_extraction.run(s6, s7_dir, _log))
        if job_id in _jobs:
            _jobs[job_id]['step_results']['step07'] = s7
    except Exception as e:
        _log(f"Step 7 failed: {e}")
        s7 = {}

    # Re-run Step 8 (Shipping Classification) — needs step03 packets + step07 required docs
    try:
        s8_dir = os.path.join(results_dir, 'step08')
        os.makedirs(s8_dir, exist_ok=True)
        # Build shipping packets from step03
        _ship_pkts = [dict(p) for p in s3.get('packets', [])
                      if (p.get('document_type', '') or '').lower() not in
                      ('lc', 'letter of credit', 'amendment', 'mt799', 'mt999',
                       'mt754', 'mt940', 'mt730', 'mt740', 'mt747')]
        s8 = _to_dict(step08_shipping_classification.run(
            {'packets': _ship_pkts}, s7, s8_dir, _log))
        if job_id in _jobs:
            _jobs[job_id]['step_results']['step08'] = s8
    except Exception as e:
        _log(f"Step 8 failed: {e}")
        s8 = {}

    # Re-run Step 9 (Shipping Reconciliation)
    try:
        s9_dir = os.path.join(results_dir, 'step09')
        os.makedirs(s9_dir, exist_ok=True)
        s9 = _to_dict(step09_shipping_reconciliation.run(s8, s7, s9_dir, _log))
        if job_id in _jobs:
            _jobs[job_id]['step_results']['step09'] = s9
    except Exception as e:
        _log(f"Step 9 failed: {e}")

    # Read back from saved file for accurate data
    _s6_saved = {}
    _s6_file = os.path.join(s6_dir, 'step06_result.json')
    if os.path.exists(_s6_file):
        with open(_s6_file, 'r', encoding='utf-8') as f:
            _s6_saved = json.load(f)
    cf = _s6_saved.get('consolidated_fields', s6.get('consolidated_fields', {}))

    result = {
        "status": "ok",
        "dc_number": _s6_saved.get('dc_number', s6.get('dc_number', '')),
        "total_fields": _s6_saved.get('total_fields', len(cf)),
        "amendment_count": _s6_saved.get('amendment_count', 0),
        "message": "Final LC regenerated. Steps 6-9 complete. Ready for verification.",
    }
    _regen_results[job_id] = result

    # Signal completion to SSE
    _log('Regeneration complete!')
    try:
        _regen_logs[job_id].put('__DONE__')
    except Exception:
        pass
    _regen_logs.pop(job_id, None)


@app.put("/api/final-lc/{job_id}")
async def save_final_lc(job_id: str, request: Request):
    """Save edited Final LC fields back to disk and memory."""
    body = await request.json()
    changes = body.get('changes', {})
    if not changes:
        return {"status": "no_changes"}

    # Load current step06 data
    s06_path = os.path.join(RESULTS_DIR, job_id, 'step06', 'step06_result.json')
    if not os.path.exists(s06_path):
        raise HTTPException(404, "Final LC not found for this job")

    with open(s06_path, 'r', encoding='utf-8') as f:
        s06_data = json.load(f)

    cf = s06_data.get('consolidated_fields', {})

    # Apply changes
    applied = []
    for field_id, change in changes.items():
        tag = change.get('tag', '')
        clause_idx = change.get('clauseIdx')
        new_value = change.get('value', '')

        if clause_idx is not None and clause_idx != '' and clause_idx != 'null':
            # Clause field edit — update in clauses dict
            clauses = s06_data.get('clauses', {})
            if tag in clauses and isinstance(clauses[tag], list):
                idx = int(clause_idx)
                if 0 <= idx < len(clauses[tag]):
                    old_val = clauses[tag][idx].get('text', '')
                    clauses[tag][idx]['text'] = new_value
                    applied.append({'tag': tag, 'clause': idx, 'old': old_val[:80], 'new': new_value[:80]})
        else:
            # Standalone field edit
            old_val = cf.get(tag, '')
            cf[tag] = new_value
            # Also update original_fields if present
            if 'original_fields' in s06_data:
                s06_data['original_fields'][tag] = new_value
            applied.append({'tag': tag, 'old': str(old_val)[:80], 'new': new_value[:80]})

            # Update dc_number if tag 20 changed
            if tag == '20':
                s06_data['dc_number'] = new_value
                # Also update in _vlm_summary if present
                if '_vlm_summary' in s06_data:
                    s06_data['_vlm_summary']['dc_number'] = new_value

    # Save back to disk
    with open(s06_path, 'w', encoding='utf-8') as f:
        json.dump(s06_data, f, indent=2, ensure_ascii=False)

    # Update in-memory if loaded
    if job_id in _jobs:
        _jobs[job_id]['step_results']['step06'] = s06_data

    return {"status": "saved", "changes_applied": len(applied), "details": applied}


# ── Compatibility endpoints (for old views) ──
# These endpoints maintain backward compatibility with earlier UI versions.

@app.get("/api/final-lc-pdf/{job_id}/{lc_number:path}")
def get_final_lc_pdf(job_id: str, lc_number: str):
    """Generate/return Final LC PDF (compatibility endpoint)."""
    results_dir = os.path.join(RESULTS_DIR, job_id)
    if not os.path.isdir(results_dir):
        raise HTTPException(404, "Job not found")
    # Check if already generated
    for f in Path(results_dir).rglob("*final_lc*.pdf"):
        return FileResponse(str(f), media_type="application/pdf", filename=f.name)
    # Generate on-demand from Step 6 data (in-memory or disk)
    sr = _jobs.get(job_id, {}).get('step_results', {})
    s6 = sr.get('step06', {})
    if not s6:
        # Load from disk
        _s6_path = os.path.join(results_dir, 'step06', 'step06_result.json')
        if os.path.exists(_s6_path):
            with open(_s6_path, 'r', encoding='utf-8') as _f:
                s6 = json.load(_f)
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
            dc_num = str(dc_num).replace('\n', ' ').replace('/', '_').replace('\\', '_').strip()[:30]

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


@app.post("/api/verify/{job_id}/{lc_number:path}")
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
    # Check if verification is currently running — block double runs
    if job.get('status') == 'processing':
        return {"verification_id": job.get('verification_id', ''), "status": "already_running", "message": "Verification is currently in progress"}
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
                'status': job['status'] if job['status'] in ('completed', 'failed') else 'processing',
                'current_step': job['current_step'],
                'progress': job['progress'][-100:],
                'progress_log': progress_log,
                'message': job['progress'][-1] if job['progress'] else '',
            }
    raise HTTPException(404, "Verification not found")


@app.get("/api/verify/result/{verification_id}")
def verify_result(verification_id: str):
    """Get verification result by verification_id.

    Returns data in the format expected by checklist.html's mapV() function.
    Key: results[] array with clauseNumber, type, status, requirement, checks[], etc.
    """
    for jid, job in _jobs.items():
        if job.get('verification_id') == verification_id:
            sr = job.get('step_results', {})

            # Load full step19 result from disk (sr['step19'] only has summary)
            results_dir = os.path.join(RESULTS_DIR, jid)
            s19_full = {}
            s19_file = os.path.join(results_dir, 'step19', 'step19_result.json')
            if os.path.exists(s19_file):
                with open(s19_file, 'r', encoding='utf-8') as f:
                    s19_full = json.load(f)

            s19 = s19_full or sr.get('step19', {})

            # Build checklist-compatible results array
            # mapV() matches by clauseNumber (e.g. "1", "2") or type (e.g. "43P")
            results = []
            for section in s19.get('sections', []):
                for clause in section.get('clauses', []):
                    clause_ref = clause.get('clause_ref', '')
                    clause_text = clause.get('clause_text', '')
                    overall = clause.get('overall_result', 'REVIEW REQUIRED').upper()

                    # Map overall_result to status for checklist
                    if overall in ('COMPLIED', 'PASS', 'COMPLIANT'):
                        status = 'compliant'
                    elif overall in ('NOT COMPLIED', 'FAIL', 'DISCREPANT'):
                        status = 'non_compliant'
                    else:
                        status = 'review_required'

                    # Extract clause number from ref (e.g. "46A-1" -> "1", "47A-3" -> "3")
                    clause_number = None
                    tag = clause_ref.split('-')[0].upper() if clause_ref else ''
                    if '-' in clause_ref:
                        clause_number = clause_ref.split('-', 1)[1]

                    # Determine type for LC field matching
                    field_type = tag.lstrip('F') if tag.startswith('F') else tag

                    # Build checks array from rows
                    checks = []
                    for row in clause.get('rows', []):
                        row_compliance = str(row.get('compliance', '')).upper()
                        if row_compliance in ('COMPLIED', 'PASS'):
                            row_status = 'pass'
                        elif row_compliance in ('NOT COMPLIED', 'FAIL'):
                            row_status = 'fail'
                        else:
                            row_status = 'review'
                        checks.append({
                            'check': row.get('condition', ''),
                            'status': row_status,
                            'detail': row.get('result', ''),
                            'document_checked': row.get('document_checked', ''),
                            'findings': row.get('findings', row.get('found_text', '')),
                        })

                    # Determine if this is an LC field (standalone, not 46A/47A/45A clause)
                    is_lc_field = field_type not in ('46A', '46B', '47A', '47B', '45A', '45B', '78', '72', '79')

                    results.append({
                        'clauseNumber': clause_number,
                        'clause_ref': clause_ref,
                        'type': field_type,
                        'is_lc_field': is_lc_field,
                        'lc_field_label': clause_text[:80] if is_lc_field else '',
                        'status': status,
                        'requirement': clause_text,
                        'summary': f"{clause.get('pass_count', 0)}P / {clause.get('fail_count', 0)}F / {clause.get('review_count', 0)}R",
                        'checks': checks,
                        'rule_checks': checks,
                        'matched_documents': [],
                    })

            return {
                "status": job.get('status', 'completed'),
                "verification_id": verification_id,
                "results": results,
                "summary": {
                    "overall_decision": s19.get('overall_decision', 'REVIEW REQUIRED'),
                    "total_clauses": s19.get('total_clauses', 0),
                    "total_pass": s19.get('total_pass', 0),
                    "total_fail": s19.get('total_fail', 0),
                    "total_review": s19.get('total_review', 0),
                },
                "overall_decision": s19.get('overall_decision', 'REVIEW REQUIRED'),
                "total_pass": s19.get('total_pass', 0),
                "total_fail": s19.get('total_fail', 0),
                "total_review": s19.get('total_review', 0),
                "elapsed_seconds": s19.get('elapsed_seconds', 0),
            }

    # Fallback: try verification_id as job_id (for disk-based results)
    s19_path = os.path.join(RESULTS_DIR, verification_id, 'step19', 'step19_result.json')
    if os.path.exists(s19_path):
        with open(s19_path, 'r', encoding='utf-8') as f:
            s19 = json.load(f)
        results = []
        for section in s19.get('sections', []):
            for clause in section.get('clauses', []):
                clause_ref = clause.get('clause_ref', '')
                tag = clause_ref.split('-')[0].upper() if clause_ref else ''
                clause_number = clause_ref.split('-', 1)[1] if '-' in clause_ref else None
                field_type = tag.lstrip('F') if tag.startswith('F') else tag
                overall = clause.get('overall_result', 'REVIEW REQUIRED').upper()
                status = 'compliant' if overall in ('COMPLIED','PASS') else 'non_compliant' if overall in ('NOT COMPLIED','FAIL') else 'review_required'
                is_lc_field = field_type not in ('46A','46B','47A','47B','45A','45B','78','72','79')
                checks = [{'check': r.get('condition',''), 'status': 'pass' if r.get('compliance','').upper() in ('COMPLIED','PASS') else 'fail' if r.get('compliance','').upper() in ('NOT COMPLIED','FAIL') else 'review', 'detail': r.get('result',''), 'document_checked': r.get('document_checked',''), 'findings': r.get('findings','')} for r in clause.get('rows',[])]
                results.append({'clauseNumber': clause_number, 'clause_ref': clause_ref, 'type': field_type, 'is_lc_field': is_lc_field, 'lc_field_label': clause.get('clause_text','')[:80] if is_lc_field else '', 'status': status, 'requirement': clause.get('clause_text',''), 'summary': f"{clause.get('pass_count',0)}P / {clause.get('fail_count',0)}F / {clause.get('review_count',0)}R", 'checks': checks, 'rule_checks': checks, 'matched_documents': []})
        return {
            "status": "completed", "verification_id": verification_id, "results": results,
            "summary": {"overall_decision": s19.get('overall_decision',''), "total_pass": s19.get('total_pass',0), "total_fail": s19.get('total_fail',0), "total_review": s19.get('total_review',0)},
            "overall_decision": s19.get('overall_decision',''), "total_pass": s19.get('total_pass',0), "total_fail": s19.get('total_fail',0), "total_review": s19.get('total_review',0),
            "elapsed_seconds": s19.get('elapsed_seconds', 0),
        }

    raise HTTPException(404, "Verification not found")


@app.get("/api/verify/history/{job_id}/{lc_number:path}")
def verify_history(job_id: str, lc_number: str):
    """Get verification history for a job/LC."""
    if job_id in _jobs:
        job = _jobs[job_id]
        vid = job.get('verification_id', '')
        if vid and job.get('status') == 'completed':
            return {"history": [{"verification_id": vid, "status": "completed", "lc_number": lc_number}]}
    # Check disk for step19 result
    s19_path = os.path.join(RESULTS_DIR, job_id, 'step19', 'step19_result.json')
    if os.path.exists(s19_path):
        return {"history": [{"verification_id": job_id, "status": "completed", "lc_number": lc_number}]}
    return {"history": []}


@app.post("/api/verify/cancel/{verification_id}")
def verify_cancel(verification_id: str):
    """Cancel a running verification."""
    for jid, job in _jobs.items():
        if job.get('verification_id') == verification_id:
            if job['status'] == 'processing':
                job['status'] = 'failed'
                job['progress'].append(f"[{datetime.now().strftime('%H:%M:%S')}] Verification cancelled by user")
            return {"status": "cancelled", "verification_id": verification_id}
    raise HTTPException(404, "Verification not found")


@app.get("/api/step19/{job_id}")
def get_step19_result(job_id: str):
    """Return the raw step19 consolidation result (sections, critical_findings, review_items).

    Used by the interactive report viewer to render clause-by-clause tables.
    """
    s19_path = os.path.join(RESULTS_DIR, job_id, 'step19', 'step19_result.json')
    if os.path.exists(s19_path):
        with open(s19_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    # Also check in-memory
    if job_id in _jobs:
        sr = _jobs[job_id].get('step_results', {})
        if 'step19' in sr:
            return sr['step19']
    raise HTTPException(404, "Step 19 result not found for this job")


@app.post("/api/verify/update/{verification_id}")
async def verify_update(verification_id: str, request: Request):
    """Save compliance overrides from the interactive report viewer.

    Accepts JSON body with:
        overrides: list of {section_ref, clause_ref, row_index, original_compliance, new_compliance, reason}
        job_id: the job ID
        lc_number: the LC number
    """
    body = await request.json()
    overrides = body.get('overrides', [])
    job_id = body.get('job_id', verification_id)

    # Save overrides to disk alongside step19
    overrides_dir = os.path.join(RESULTS_DIR, job_id, 'step19')
    os.makedirs(overrides_dir, exist_ok=True)
    overrides_path = os.path.join(overrides_dir, 'compliance_overrides.json')

    # Merge with existing overrides if any
    existing = []
    if os.path.exists(overrides_path):
        with open(overrides_path, 'r', encoding='utf-8') as f:
            existing = json.load(f).get('overrides', [])

    # Replace existing overrides for same clause_ref+row_index, add new ones
    merged = {f"{o['clause_ref']}|{o['row_index']}": o for o in existing}
    for o in overrides:
        merged[f"{o['clause_ref']}|{o['row_index']}"] = o

    save_data = {
        'job_id': job_id,
        'lc_number': body.get('lc_number', ''),
        'overrides': list(merged.values()),
        'saved_at': datetime.now().isoformat(),
    }
    with open(overrides_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    return {"status": "ok", "saved": len(merged), "path": overrides_path}


@app.get("/api/report/{job_id}/{lc_number:path}")
def get_report_by_lc(job_id: str, lc_number: str):
    """Get compliance report for a specific LC number. Returns the most recent report."""
    results_dir = os.path.join(RESULTS_DIR, job_id)
    all_reports = []
    for pattern in ("*compliance_report*.pdf", "ComplyTrade_Report*.pdf"):
        all_reports.extend(Path(results_dir).rglob(pattern))
    if all_reports:
        # Return the newest file
        newest = max(all_reports, key=lambda f: f.stat().st_mtime)
        return FileResponse(str(newest), media_type="application/pdf", filename=newest.name)
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


@app.post("/api/jobs/{job_id}/notes")
async def save_job_notes(job_id: str, request: Request):
    """Save notes for a job."""
    body = await request.json()
    notes = body.get('notes', '')
    # Save to disk
    results_dir = os.path.join(RESULTS_DIR, job_id)
    if os.path.isdir(results_dir):
        notes_path = os.path.join(results_dir, '_notes.txt')
        with open(notes_path, 'w', encoding='utf-8') as f:
            f.write(notes)
    # Save in memory
    if job_id in _jobs:
        _jobs[job_id]['notes'] = notes
    return {"status": "ok"}


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
        s4 = {}
        if _is_step_enabled(4):
            job['current_step'] = 4
            _p("Step 4: MT Document Identification (from Step 3 classification)...")

            _lc_types = {'lc', 'amendment', 'mt700', 'mt701', 'mt705', 'mt707', 'mt708',
                         'mt710', 'mt711', 'mt720', 'mt721'}
            # MT799 / MT999 = SWIFT free-format messages. These are NOT LCs but
            # they often carry amendment instructions in the narrative body
            # ("UNDER FIELD 45A SHOULD READ AS X I/O Y"). When that happens we
            # promote the packet to MT707 so step06 will apply the amendment
            # via _extract_mt799_amendment_fields().
            _free_format_types = {'mt799', 'mt999', 'free format message',
                                  'free_format_message', 'bank-to-bank message'}
            # Endorsement pages and blank pages are BACK SIDES of the previous document
            # They carry stamps, endorsements, signatures — merge into previous packet
            _back_page_types = {'blank page', 'blank_page', 'endorsement page'}
            _s3_packets = s3.get('packets', [])

            # Import the MT799 amendment detector from step04 so we can apply
            # the same promotion logic the standalone runner uses.
            from steps.step04_mt_identification import (
                _is_mt799_amendment as _s4_is_mt799_amendment,
                _looks_like_mt799_free_format as _s4_looks_like_mt799_ff,
            )

            # Build a {page_number: cleaned_text} index from Step 2's output.
            # Step 3 stores packets with only page-number references, not the
            # actual cleaned text — so a `_pkt['pages'][i]['cleaned_text']`
            # lookup returns empty. We must source text from Step 2 directly.
            _s2_page_text = {}
            for _pg in s2.get('pages', []) or []:
                if isinstance(_pg, dict):
                    _pn = _pg.get('page_number')
                    _txt = _pg.get('cleaned_text') or _pg.get('raw_text') or ''
                else:
                    _pn = getattr(_pg, 'page_number', None)
                    _txt = getattr(_pg, 'cleaned_text', '') or getattr(_pg, 'raw_text', '')
                if _pn is not None and _txt:
                    _s2_page_text[int(_pn)] = _txt

            def _packet_full_text(_pkt):
                """Concatenate cleaned text for every page in the packet,
                pulling from the Step-2 page-text index (Step-3 packets do
                not carry the cleaned text inline)."""
                _parts = []
                # Try inline pages first (covers the case where step3 did
                # carry cleaned_text in some configurations)
                for _pg in _pkt.get('pages', []) or []:
                    if isinstance(_pg, dict):
                        _t = _pg.get('cleaned_text') or _pg.get('raw_text') or ''
                    else:
                        _t = getattr(_pg, 'cleaned_text', '') or getattr(_pg, 'raw_text', '')
                    if _t:
                        _parts.append(_t)
                # Fall back to the Step-2 index by page number
                if not _parts:
                    for _pn in _pkt.get('page_numbers', []) or []:
                        try:
                            _t = _s2_page_text.get(int(_pn), '')
                        except (TypeError, ValueError):
                            _t = ''
                        if _t:
                            _parts.append(_t)
                return '\n'.join(_parts)

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

                # ── MT799 / MT999 free-format ──
                # Check this BEFORE the LC branch because step03 may have
                # mislabelled an MT799 as "LC" if its body references F-tags.
                # We re-inspect the packet text for free-format markers and
                # amendment instructions and route accordingly.
                _pkt_text = _packet_full_text(_pkt)
                _is_ff = _dt in _free_format_types or _s4_looks_like_mt799_ff(_pkt_text)
                if _is_ff:
                    if _s4_is_mt799_amendment(_pkt_text):
                        _pkt_copy['mt_type'] = 'MT707'
                        _pkt_copy['source_mt'] = 'MT799'
                        _pkt_copy['is_799_amendment'] = True
                        _p(f"  pkt {_pkt.get('packet_id','?')} pages={_pkt.get('page_numbers',[])} → MT799 amendment (promoted to MT707)")
                    else:
                        _pkt_copy['mt_type'] = 'MT799'
                        _pkt_copy['source_mt'] = 'MT799'
                        _pkt_copy['is_799_amendment'] = False
                        _p(f"  pkt {_pkt.get('packet_id','?')} pages={_pkt.get('page_numbers',[])} → MT799 free format")
                    _mt_packets.append(_pkt_copy)
                    _prev_packet = _pkt_copy
                    continue

                if _dt in _lc_types:
                    _pkt_copy['mt_type'] = 'MT707' if 'amend' in _dt else 'MT700'
                    _mt_packets.append(_pkt_copy)
                    _prev_packet = _pkt_copy
                # P103: BAHL informational MT types — not LC, not shipping
                elif _dt.upper().startswith('MT') and _dt.upper() in (
                    'MT754', 'MT940', 'MT730', 'MT740', 'MT742',
                    'MT734', 'MT750', 'MT752', 'MT747',
                ):
                    _pkt_copy['mt_type'] = _dt.upper()
                    _mt_packets.append(_pkt_copy)
                    _prev_packet = _pkt_copy
                    _p(f"  pkt {_pkt.get('packet_id','?')} pages={_pkt.get('page_numbers',[])} → {_dt.upper()} (informational)")
                else:
                    _pkt_copy['mt_type'] = 'shipping'
                    _shipping_packets.append(_pkt_copy)
                    _prev_packet = _pkt_copy

            s4 = {'packets': _mt_packets + _shipping_packets}
            job['step_results']['step04'] = s4
            _p(f"  MT/LC: {len(_mt_packets)}, Shipping: {len(_shipping_packets)}")
        else:
            _p("Step 4: SKIPPED (disabled in settings)")
            s4 = s3

        # ── Step 5: Passthrough ──
        # VLM text completion for ALL pages (including MT/LC) is now handled in
        # Step 2 (OCR Cleaning). Step 2's cleaned_text is the single source of
        # truth for all downstream steps. Step 5 just passes through.
        s5 = {}
        if _is_step_enabled(5):
            job['current_step'] = 5
            _p("Step 5: Passthrough (VLM text review handled in Step 2 for all pages)")
            s5 = s4
            job['step_results']['step05'] = s5
        else:
            _p("Step 5: SKIPPED (disabled in settings)")
            s5 = s4

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
            if _dt in ('lc', 'amendment', 'blank page', 'blank_page', 'endorsement page',
                       'mt799', 'mt999', 'mt730', 'mt754', 'mt940', 'mt740', 'mt747', 'mt734',
                       'header page'):
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
        s10 = {}
        if _is_step_enabled(10):
            job['current_step'] = 10
            _p("Step 10: Traceability & Confidence Preservation...")
            s10 = _to_dict(step10_traceability.run(
                {'step01': s1, 'step02': s2, 'step03': s3, 'step04': s4,
                 'step05': s5, 'step06': s6, 'step07': s7, 'step08': s8, 'step09': s9},
                os.path.join(results_dir, 'step10'), _p
            ))
            job['step_results']['step10'] = {'flags': s10.get('total_flags', 0)}
        else:
            _p("Step 10: SKIPPED (disabled in settings)")

        # ── Step 11: Human Review Gate ──
        s11 = {}
        if _is_step_enabled(11):
            job['current_step'] = 11
            _p("Step 11: Ready for Human Review...")
            s11 = _to_dict(step11_human_review.run(
                step7_result=s7, step9_result=s9, step10_result=s10,
                job_id=job_id, output_dir=os.path.join(results_dir, 'step11'),
                progress_callback=_p
            ))
            job['step_results']['step11'] = s11
        else:
            _p("Step 11: SKIPPED (disabled in settings)")

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
        ts_msg = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        job['progress'].append(ts_msg)
        print(ts_msg, flush=True)

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

        # ── Step 14b: Implicit LC Key Term Checks (VLM-based) ──
        # Verify LC key term fields (dates, amounts, ports, shipment) using
        # specialized VLM prompts with exact trade finance rules.
        _p("Step 14b: Implicit LC Key Term Verification...")
        try:
            # Get LC consolidated fields
            _s06 = sr.get('step06', {})
            _lc_cf = {}
            if 'final_lc' in _s06 and isinstance(_s06['final_lc'], dict):
                _lc_cf = _s06['final_lc'].get('consolidated_fields', {})
            elif 'consolidated_fields' in _s06:
                _lc_cf = _s06['consolidated_fields']

            # Get shipping document packets
            _s09 = sr.get('step09', {})
            _packets = _s09.get('reconciled_packets', [])

            s14b = _to_dict(step14_implicit.run(
                _lc_cf, _packets,
                config_dir='config',
                output_dir=os.path.join(results_dir, 'step14b'),
                progress_fn=_p,
            ))
            sr['step14b'] = s14b

            # Merge 14b check results into s14 rows for consolidation
            for chk in s14b.get('checks', []):
                s14['rows'] = s14.get('rows', [])
                s14['rows'].append({
                    'clause_ref': chk.get('clause_ref', ''),
                    'clause_text': chk.get('condition', ''),
                    'condition_text': chk.get('condition', ''),
                    'document_checked': chk.get('document_checked', ''),
                    'findings': chk.get('findings', ''),
                    'result': chk.get('result', ''),
                    'compliance': chk.get('compliance', 'REVIEW'),
                    'confidence': chk.get('confidence', 1.0),
                    'is_implicit': True,
                    'source_step': '14b',
                })
            _p(f"Step 14b: {s14b.get('summary', {}).get('total', 0)} checks merged into pipeline")
        except Exception as _e14b:
            _p(f"Step 14b WARNING: {_e14b}")
            print(f"[WARN] Step 14b: {_e14b}\n{traceback.format_exc()}", flush=True)

        # ── Step 15: Non-Compliance Handling ──
        s15 = {}
        if _is_step_enabled(15):
            job['current_step'] = 15
            _p("Step 15: Non-Compliance & Non-Checkable Clauses...")
            s15 = _to_dict(step15_non_compliance.run(sr.get('step07', {}), os.path.join(results_dir, 'step15'), _p))
            sr['step15'] = s15
        else:
            _p("Step 15: SKIPPED (disabled in settings)")

        # ── Step 16: Confidence Review ──
        s16 = {}
        if _is_step_enabled(16):
            job['current_step'] = 16
            _p("Step 16: Confidence-Based Review Escalation...")
            s16 = _to_dict(step16_confidence_review.run(
                s14.get('rows', []), s15.get('clause_status_map', {}),
                os.path.join(results_dir, 'step16'), _p
            ))
            sr['step16'] = {'escalated': s16.get('escalated_count', 0)}
        else:
            _p("Step 16: SKIPPED (disabled in settings)")

        # ── Step 17: Cross-Clause Dependencies ──
        s17 = {}
        if _is_step_enabled(17):
            job['current_step'] = 17
            _p("Step 17: Cross-Clause Dependency Handling...")
            try:
                _s17_input = s16.get('rows', []) if s16 else s14.get('rows', [])
                s17 = _to_dict(step17_cross_clause.run(
                    _s17_input,
                    os.path.join(results_dir, 'step17'), _p
                ))
                sr['step17'] = {'overrides': s17.get('overrides_applied', 0)}
            except Exception as _e17:
                _p(f"Step 17 FAILED: {_e17}")
                print(f"[ERROR] Step 17: {_e17}\n{traceback.format_exc()}", flush=True)
                raise
        else:
            _p("Step 17: SKIPPED (disabled in settings)")

        # ── Step 18: Threading ──
        if _is_step_enabled(18):
            job['current_step'] = 18
            _p("Step 18: Multi-threaded processing complete (inline)")
            sr['step18'] = {'status': 'complete'}
        else:
            _p("Step 18: SKIPPED (disabled in settings)")

        # ── Step 19: Consolidation ──
        job['current_step'] = 19
        _p("Step 19: Consolidating Verification Output...")
        try:
            # Use the latest available rows — from step17, step16, or step14
            _s19_rows = (s17.get('reconciled_rows', s17.get('rows', []))
                         if s17 else s16.get('rows', [])
                         if s16 else s14.get('rows', []))
            s19 = _to_dict(step19_consolidation.run(
                _s19_rows,
                os.path.join(results_dir, 'step19'), _p
            ))
            sr['step19'] = {
                'decision': s19.get('overall_decision', ''),
                'pass': s19.get('total_pass', 0),
                'fail': s19.get('total_fail', 0),
                'review': s19.get('total_review', 0),
            }
        except Exception as _e19:
            _p(f"Step 19 FAILED: {_e19}")
            print(f"[ERROR] Step 19: {_e19}\n{traceback.format_exc()}", flush=True)
            raise

        # ── Step 20: Report Generation ──
        job['current_step'] = 20
        _p("Step 20: Generating Final Compliance Report...")
        try:
            s20 = step20_report.run(
                s19, sr.get('step06', {}),
                os.path.join(results_dir, 'step20'), _p
            )
            sr['step20'] = {'report_path': s20.get('report_path', s20.get('pdf_path', ''))}
        except Exception as _e20:
            _p(f"Step 20 FAILED: {_e20}")
            print(f"[ERROR] Step 20: {_e20}\n{traceback.format_exc()}", flush=True)
            raise

        # Pipeline complete
        job['status'] = 'completed'
        job['result'] = sr.get('step19', {})
        _p(f"Pipeline complete! Decision: {sr.get('step19', {}).get('decision', 'N/A')}")

    except Exception as e:
        job['status'] = 'failed'
        _err_msg = f"Verification FAILED at Step {job['current_step']}: {str(e)}"
        _err_tb = traceback.format_exc()
        _p(_err_msg)
        _p(_err_tb[:500])
        print(f"\n[ERROR] {_err_msg}", flush=True)
        print(_err_tb, flush=True)


# ══════════════════════════════════════════════════════════════
# MAIN — Server Entry Point
# ══════════════════════════════════════════════════════════════

# ── Settings API ──────────────────────────────────────────────────────────────

_STEP_NAMES = {
    1: "Page-Level Raw OCR", 2: "OCR Text Cleaning", 3: "Page Sequencing & Classification",
    4: "MT Identification", 5: "MT Reconciliation", 6: "Final LC Extraction",
    7: "Clause & Requirement Extraction", 8: "Shipping Doc Classification",
    9: "Shipping OCR Reconciliation", 10: "Traceability Flags",
    11: "Human Review Gate", 12: "Clause Decomposition",
    13: "Row Construction", 14: "VLM Verification",
    15: "Non-Compliance Summary", 16: "Confidence Review",
    17: "Cross-Clause Checks", 18: "Threading",
    19: "Consolidation", 20: "Report Generation",
}
_CORE_STEPS = {1, 2, 3, 6, 7, 8, 9, 12, 13, 14, 19, 20}

# Runtime copy of step toggles (so we don't modify the imported config)
_step_enabled = dict(STEP_ENABLED)


@app.get("/settings")
def settings_page():
    """Serve the settings management page."""
    settings_path = os.path.join(VIEW_DIR, 'settings.html')
    if os.path.exists(settings_path):
        return HTMLResponse(open(settings_path, encoding='utf-8').read())
    # Inline fallback if file doesn't exist
    return HTMLResponse("<html><body><h1>Settings page not found</h1></body></html>")


@app.get("/api/prompts")
def get_prompts():
    """Get ALL system prompts used in the pipeline (view-only).

    Settings-page numbering (preserves user's '3a = Step 4' alignment):
      Step 1  — GLM OCR                                (backend step01)
      Step 3  — Page Classification (Legacy Combined)  (backend step03 legacy)
      Step 4  — Document Type        (Step 3a)         (backend step03 sub)
      Step 5  — Markings & Seals     (Step 3b)         (backend step03 sub)
      Step 6  — Copy / Original      (Step 3c)         (backend step03 sub)
      Step 7  — BL Sub-type          (Step 3d)         (backend step03 sub)
      Step 8  — Packet Summary       (Step 3e)         (backend step03 sub)
      Step 9  — Packet Validator     (Step 3f)         (backend step03 sub)
      Step 10 — Page Re-check        (Step 3g)         (backend step03 sub)
      Step 11 — MT Identification                      (backend step04)
      Step 12 — MT Reconciliation                      (backend step05)
      Step 13 — LC Field Extraction                    (backend step06)
      Step 14 — LC Amendment Processing                (backend step06)
      Step 15 — Shipping Classification                (backend step08)
      Step 16 — Shipping Reconciliation                (backend step09)
      Step 17 — Decomposition (System)                 (backend step12)
      Step 18 — Decomposition (User)                   (backend step12)
      Step 19 — Verification                           (backend step14)
    """
    def _safe_import(module_path: str, attr: str) -> str:
        try:
            mod = __import__(module_path, fromlist=[attr])
            return getattr(mod, attr, '') or ''
        except Exception:
            return ''

    from config.settings import GLM_OCR_PROMPT
    from steps.step03_sequencing import (
        CLASSIFY_PROMPT,
        CLASSIFY_DOCTYPE_PROMPT,
        EXTRACT_MARKINGS_PROMPT,
        COPY_STATUS_PROMPT,
        BL_SUBTYPE_PROMPT,
        PACKET_SUMMARY_PROMPT,
        _PACKET_VALIDATOR_PROMPT,
        _RECHECK_PROMPT,
    )
    from steps.step12_decomposition import DECOMPOSITION_SYSTEM_PROMPT
    from steps.step14_verification import _VLM_PROMPT_TEMPLATE

    # Step 2 combines main + fallback OCR cleaning prompts
    _s02_main = _safe_import('steps.step02_ocr_cleaning', '_VLM_EXTRACT_PROMPT')
    _s02_fb = _safe_import('steps.step02_ocr_cleaning', '_VLM_FALLBACK_PROMPT')
    _s02_combined = (
        "=== MAIN EXTRACTION PROMPT (used for every page) ===\n"
        f"{_s02_main}\n\n"
        "=== FALLBACK PROMPT (used when GLM returns garbage) ===\n"
        f"{_s02_fb}"
    ) if (_s02_main or _s02_fb) else ''

    return {
        "step1":  GLM_OCR_PROMPT,
        "step2":  _s02_combined,
        "step3":  CLASSIFY_PROMPT,
        "step4":  CLASSIFY_DOCTYPE_PROMPT,
        "step5":  EXTRACT_MARKINGS_PROMPT,
        "step6":  COPY_STATUS_PROMPT,
        "step7":  BL_SUBTYPE_PROMPT,
        "step8":  PACKET_SUMMARY_PROMPT,
        "step9":  _PACKET_VALIDATOR_PROMPT,
        "step10": _RECHECK_PROMPT,
        "step11": _safe_import('steps.step04_mt_identification', '_VLM_CLASSIFY_PROMPT'),
        "step12": _safe_import('steps.step05_mt_reconciliation', '_VLM_RECONCILE_PROMPT'),
        "step13": _safe_import('steps.step06_final_lc', '_VLM_EXTRACT_PROMPT'),
        "step14": _safe_import('steps.step06_final_lc', '_VLM_AMENDMENT_PROMPT'),
        "step15": _safe_import('steps.step08_shipping_classification', '_CLASSIFICATION_PROMPT'),
        "step16": _safe_import('steps.step09_shipping_reconciliation', '_RECONCILIATION_PROMPT'),
        "step17": DECOMPOSITION_SYSTEM_PROMPT,
        "step18": _safe_import('steps.step12_decomposition', 'DECOMPOSITION_USER_TEMPLATE'),
        "step19": _VLM_PROMPT_TEMPLATE,
    }


@app.get("/api/settings")
def get_settings():
    """Get current pipeline settings."""
    from config import settings as _cfg
    steps = []
    for i in range(1, 21):
        steps.append({
            "step": i,
            "name": _STEP_NAMES.get(i, f"Step {i}"),
            "enabled": _step_enabled.get(i, True),
            "core": i in _CORE_STEPS,
        })
    return {
        "steps": steps,
        "concurrency": {
            "max_concurrent_ocr": _cfg.MAX_CONCURRENT_OCR,
            "max_concurrent_vlm": _cfg.MAX_CONCURRENT_VLM,
        },
        "models": {
            "glm_ocr_url": _cfg.GLM_OCR_URL,
            "qwen_vlm_url": _cfg.QWEN_VLM_URL,
            "qwen_vlm_model": _cfg.QWEN_VLM_MODEL,
            "text_llm_url": getattr(_cfg, 'QWEN_TEXT_LLM_URL', '') or '',
            "text_llm_model": getattr(_cfg, 'QWEN_TEXT_LLM_MODEL', '') or '',
        },
        "timeouts": {
            "ocr_timeout": _cfg.OCR_TIMEOUT,
            "vlm_timeout": _cfg.VLM_TIMEOUT,
        },
        "confidence_threshold": _cfg.CONFIDENCE_THRESHOLD,
        "auth": {
            "enabled": _auth_enabled,
            "username": _cfg.AUTH_USERNAME,
        },
    }


@app.post("/api/settings")
async def update_settings(request: Request):
    """Update pipeline settings (step toggles, concurrency, etc.)."""
    from config import settings as _cfg
    body = await request.json()

    # Update step toggles
    if 'steps' in body:
        for step_update in body['steps']:
            step_num = step_update.get('step')
            enabled = step_update.get('enabled')
            if step_num is not None and enabled is not None:
                _step_enabled[step_num] = bool(enabled)

    # Update concurrency
    if 'concurrency' in body:
        c = body['concurrency']
        if 'max_concurrent_ocr' in c:
            _cfg.MAX_CONCURRENT_OCR = int(c['max_concurrent_ocr'])
        if 'max_concurrent_vlm' in c:
            _cfg.MAX_CONCURRENT_VLM = int(c['max_concurrent_vlm'])

    # Update confidence threshold
    if 'confidence_threshold' in body:
        _cfg.CONFIDENCE_THRESHOLD = float(body['confidence_threshold'])

    # Update model selection
    if 'models' in body:
        m = body['models']
        if 'qwen_vlm_url' in m:
            _cfg.QWEN_VLM_URL = m['qwen_vlm_url']
        if 'qwen_vlm_model' in m:
            _cfg.QWEN_VLM_MODEL = m['qwen_vlm_model']
        if 'glm_ocr_url' in m:
            _cfg.GLM_OCR_URL = m['glm_ocr_url']
        if 'text_llm_url' in m:
            _cfg.QWEN_TEXT_LLM_URL = m['text_llm_url']
        if 'text_llm_model' in m:
            _cfg.QWEN_TEXT_LLM_MODEL = m['text_llm_model']

    # Update timeouts
    if 'timeouts' in body:
        t = body['timeouts']
        if 'ocr_timeout' in t:
            _cfg.OCR_TIMEOUT = int(t['ocr_timeout'])
        if 'vlm_timeout' in t:
            _cfg.VLM_TIMEOUT = int(t['vlm_timeout'])

    # Update auth
    if 'auth' in body:
        global _auth_enabled
        a = body['auth']
        if 'enabled' in a:
            _auth_enabled = bool(a['enabled'])
        if 'username' in a and a['username']:
            _cfg.AUTH_USERNAME = a['username']
        if 'password' in a and a['password']:
            _cfg.AUTH_PASSWORD = a['password']

    return {"status": "ok", "message": "Settings updated"}


@app.post("/api/override/{job_id}/classification")
async def override_classification(job_id: str, request: Request):
    """
    Override the classification of a document packet.
    Updates step03, step08, and step09 so verification picks up the change.
    """
    # Allow job from disk even if not in memory
    results_dir = os.path.join(RESULTS_DIR, job_id)
    if job_id not in _jobs and not os.path.isdir(results_dir):
        raise HTTPException(404, "Job not found")

    body = await request.json()
    packet_id = body.get('packet_id', '')
    page_number = body.get('page_number')
    new_type = body.get('document_type')
    new_copy_status = body.get('copy_status')
    notes = body.get('notes', '')

    if not packet_id and not page_number:
        raise HTTPException(400, "packet_id or page_number required")

    # Helper: match packet by packet_id or by page_number
    def _matches(pkt):
        if packet_id and str(pkt.get('packet_id', '')) == str(packet_id):
            return True
        if page_number:
            pn = int(page_number)
            pkt_pages = pkt.get('page_numbers', pkt.get('pages', []))
            if pn in pkt_pages:
                return True
        return False

    def _update_packet(pkt):
        if new_type:
            pkt['document_type'] = new_type
            pkt['override'] = True
        if new_copy_status:
            pkt['copy_status'] = new_copy_status
        if notes:
            pkt['notes'] = notes

    # Update step03 (page classification)
    step03_path = os.path.join(results_dir, 'step03', 'step03_result.json')
    if os.path.exists(step03_path):
        with open(step03_path, 'r', encoding='utf-8') as f:
            s3 = json.load(f)
        # Update classifications (per-page)
        if page_number:
            pn = int(page_number)
            for cls in s3.get('classifications', []):
                if cls.get('page_number') == pn:
                    if new_type: cls['document_type'] = new_type
                    if new_copy_status: cls['copy_status'] = new_copy_status
        # Update packets
        for pkt in s3.get('packets', []):
            if _matches(pkt):
                _update_packet(pkt)
        with open(step03_path, 'w', encoding='utf-8') as f:
            json.dump(s3, f, indent=2, ensure_ascii=False)

    # Update step08 (shipping classification)
    step08_path = os.path.join(results_dir, 'step08', 'step08_result.json')
    if os.path.exists(step08_path):
        with open(step08_path, 'r', encoding='utf-8') as f:
            s8 = json.load(f)
        for pkt in s8.get('classified_packets', s8.get('packets', [])):
            if _matches(pkt):
                _update_packet(pkt)
        with open(step08_path, 'w', encoding='utf-8') as f:
            json.dump(s8, f, indent=2, ensure_ascii=False)

    # Update step09 (reconciled packets — this is what verification reads)
    step09_path = os.path.join(results_dir, 'step09', 'step09_result.json')
    if os.path.exists(step09_path):
        with open(step09_path, 'r', encoding='utf-8') as f:
            s9 = json.load(f)
        for pkt in s9.get('reconciled_packets', s9.get('packets', [])):
            if _matches(pkt):
                _update_packet(pkt)
        with open(step09_path, 'w', encoding='utf-8') as f:
            json.dump(s9, f, indent=2, ensure_ascii=False)

    # Store in memory too
    if job_id in _jobs:
        if 'overrides' not in _jobs[job_id]:
            _jobs[job_id]['overrides'] = {}
        _jobs[job_id]['overrides'][packet_id or f'pg_{page_number}'] = {
            'document_type': new_type, 'copy_status': new_copy_status,
            'notes': notes, 'timestamp': datetime.now().isoformat(),
        }

    return {"status": "ok", "message": f"Override applied — verification will use updated classification"}


@app.post("/api/override/{job_id}/final-lc")
async def override_final_lc(job_id: str, request: Request):
    """
    Override Final LC field values.
    Changes propagate to verification on re-run.
    """
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found")
    body = await request.json()
    field_tag = body.get('field_tag')
    new_value = body.get('value')

    if not field_tag or new_value is None:
        raise HTTPException(400, "field_tag and value required")

    # Update step06 result
    results_dir = os.path.join(RESULTS_DIR, job_id)
    step06_path = os.path.join(results_dir, 'step06', 'step06_result.json')
    if os.path.exists(step06_path):
        with open(step06_path, 'r', encoding='utf-8') as f:
            s6 = json.load(f)
        s6.get('consolidated_fields', {})[field_tag] = new_value
        # If updating field 20 (DC Number), also update dc_number
        if field_tag == '20':
            s6['dc_number'] = new_value
        with open(step06_path, 'w', encoding='utf-8') as f:
            json.dump(s6, f, indent=2, ensure_ascii=False)
        return {"status": "ok", "message": f"LC field {field_tag} updated"}
    else:
        raise HTTPException(404, "Step 6 results not found")


def _is_step_enabled(step_num: int) -> bool:
    """Check if a pipeline step is enabled."""
    return _step_enabled.get(step_num, True)


if __name__ == '__main__':
    print(f"""
================================================================================
  ComplyTrade Pilot V2
  Build: {BUILD_TAG}
  GLM-OCR: {GLM_OCR_URL}
  Qwen VLM: {QWEN_VLM_URL}
  Database: trade_finance_pilot @ localhost:5432
  Web UI: http://{SERVER_HOST}:{SERVER_PORT}/interface
  Checklist: http://{SERVER_HOST}:{SERVER_PORT}/checklist
  Compliance: http://{SERVER_HOST}:{SERVER_PORT}/compliance
  API Docs: http://{SERVER_HOST}:{SERVER_PORT}/docs
  Port: {SERVER_PORT}
================================================================================
""")
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
