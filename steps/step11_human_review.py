"""
Step 11 — Human Review and Approval Gate
==========================================
This step acts as a mandatory checkpoint between the document extraction/classification
phase (Steps 1-10) and the compliance verification phase (Steps 12-20).

PURPOSE:
    Before the system automatically verifies LC conditions against shipping documents,
    a human reviewer must inspect and approve the AI's work product:
    - Is the Final LC correctly assembled from SWIFT messages?
    - Are shipping documents correctly classified (BL, Invoice, Insurance, etc.)?
    - Are any documents incorrectly flagged as "alien" (not required by the LC)?
    - Do low-confidence classifications need human correction?

    This prevents verification errors caused by mis-classification from propagating
    into the compliance report. In trade finance, a wrong document type assignment
    (e.g., labeling a Packing List as an Invoice) would produce meaningless results.

INPUTS:
    - Step 7 result: StructuredLC with consolidated_fields and required_documents
      (the parsed Letter of Credit with its document requirements from field 46A)
    - Step 9 result: Reconciled shipping document packets with classification
    - Step 10 result (optional): Traceability report with pipeline confidence

OUTPUTS:
    - ReviewSession object containing all packets for human inspection
    - API endpoints for get/approve/reclassify/mark_alien operations
    - Attention items list highlighting documents that need human review

TRADE FINANCE CONTEXT:
    In documentary credit (LC) processing, the bank checker must first verify that
    all required documents are present and correctly identified before examining
    their content. This step mirrors that real-world workflow.

AI MODEL: None — this is a human-in-the-loop step with no AI model calls.
"""

import json
import sys as _sys; _sys.stdout.reconfigure(encoding="utf-8", errors="replace") if hasattr(_sys.stdout, "reconfigure") else None
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from pathlib import Path
from datetime import datetime


# ── Dataclasses ──
# These define the structure for tracking human review actions and document packets.

@dataclass
class AuditEntry:
    """
    A single human action in the review process.

    Every action the reviewer takes (approve, reclassify, mark alien) is logged
    here for audit trail purposes. In trade finance, maintaining a complete audit
    trail is a regulatory requirement — every decision must be traceable.
    """
    action: str                         # "approve" | "reclassify" | "mark_alien" | "unmark_alien" | "edit_field" | "comment"
    target_packet_id: str = ""          # which packet was affected
    old_value: str = ""
    new_value: str = ""
    reason: str = ""
    reviewer: str = ""
    timestamp: str = ""

    def __post_init__(self):
        # Auto-set timestamp to current UTC time if not provided
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat() + "Z"


@dataclass
class ReviewPacket:
    """
    A document packet presented for review.

    Each packet represents one logical document (e.g., a Bill of Lading that
    spans pages 3-5 of the uploaded PDF). The reviewer sees the AI's classification
    and confidence, and can override if incorrect.

    TRADE FINANCE TERMS:
    - matched_document: AI found a matching LC requirement for this document
    - alien_document: document not required by the LC (e.g., internal memo included by mistake)
    - extra_document: additional copy beyond what LC requires
    - ambiguity_flag: AI is unsure about classification (e.g., could be Invoice or Proforma)
    """
    packet_id: str
    document_type: str = ""
    classification_status: str = ""     # matched_document | alien_document | extra_document | unknown
    match_confidence: float = 0.0
    matched_requirement_name: str = ""
    original_pages: List[int] = field(default_factory=list)
    page_image_paths: List[str] = field(default_factory=list)
    refined_text: str = ""
    extracted_fields: Dict[str, Any] = field(default_factory=dict)
    ambiguity_flag: bool = False
    ambiguity_notes: str = ""

    # Human review state — these fields are populated by the reviewer's actions
    human_approved: bool = False
    human_document_type: str = ""       # human-corrected type (empty = accept AI type)
    human_notes: str = ""


@dataclass
class ReviewSession:
    """
    Complete review session for a job.

    This is the top-level container for all review data. It tracks the overall
    session status, stores all document packets, and accumulates the audit log
    of human actions. The pipeline will not proceed to Step 12 until the
    session status changes to "approved".
    """
    job_id: str
    status: str = "pending_review"      # pending_review | in_review | approved | rejected
    created_at: str = ""
    updated_at: str = ""
    reviewer: str = ""

    # Data for review — these come from earlier pipeline steps
    final_lc_summary: Dict[str, Any] = field(default_factory=dict)
    review_packets: List[dict] = field(default_factory=list)
    traceability_warnings: List[str] = field(default_factory=list)
    pipeline_confidence: float = 0.0

    # Counts — quick summary for the review UI
    total_packets: int = 0
    matched_count: int = 0
    alien_count: int = 0
    ambiguous_count: int = 0

    # Human actions — audit trail of everything the reviewer did
    audit_log: List[dict] = field(default_factory=list)
    human_changes_count: int = 0

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat() + "Z"
        if not self.updated_at:
            self.updated_at = self.created_at


# ── In-Memory Session Store ──
# In production, this would be a database (PostgreSQL). For the pipeline prototype,
# we use a dictionary for fast lookups and JSON files for persistence across restarts.

_sessions: Dict[str, dict] = {}


def _save_session(session_data: dict, output_dir: str):
    """Persist session to disk as a JSON file for durability across server restarts."""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fpath = os.path.join(output_dir, f"review_session_{session_data['job_id']}.json")
        with open(fpath, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)


def _load_session(job_id: str, output_dir: str) -> Optional[dict]:
    """
    Load session from memory first, then fall back to disk.
    This two-tier lookup ensures fast access while supporting persistence.
    """
    # Check in-memory cache first (fast path)
    if job_id in _sessions:
        return _sessions[job_id]
    # Fall back to disk (slow path — used after server restart)
    if output_dir:
        fpath = os.path.join(output_dir, f"review_session_{job_id}.json")
        if os.path.exists(fpath):
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            _sessions[job_id] = data
            return data
    return None


# ── API-Compatible Functions ──
# These functions mirror REST API endpoints. The server.py file calls them directly,
# or they can be registered as FastAPI routes via register_routes().

def get_review(job_id: str, output_dir: str = None) -> dict:
    """
    GET /api/review/{job_id}

    Returns the review session data for a given job.
    The frontend uses this to display document packets for human inspection.
    """
    session = _load_session(job_id, output_dir or '')
    if not session:
        return {'error': f'Review session not found for job {job_id}', 'status': 404}

    return {
        'status': 200,
        'data': session,
    }


def approve_review(job_id: str, reviewer: str = "", notes: str = "", output_dir: str = None) -> dict:
    """
    POST /api/review/{job_id}/approve

    Approve the document set. This is the gate that allows the pipeline to proceed.
    Once approved, Steps 12-20 (verification phase) can begin.

    All packets are marked as human_approved=True, meaning the reviewer accepts
    the AI's classification (or has already corrected any mistakes via reclassify).
    """
    session = _load_session(job_id, output_dir or '')
    if not session:
        return {'error': f'Review session not found for job {job_id}', 'status': 404}

    # Prevent double-approval
    if session.get('status') == 'approved':
        return {'error': 'Session already approved', 'status': 400}

    # Record approval in the audit log for traceability
    entry = AuditEntry(
        action="approve",
        reason=notes or "Human reviewer approved document set",
        reviewer=reviewer,
    )
    session.setdefault('audit_log', []).append(asdict(entry))
    session['status'] = 'approved'
    session['reviewer'] = reviewer
    session['updated_at'] = datetime.utcnow().isoformat() + "Z"

    # Mark all packets as approved — the reviewer has accepted the full set
    for pkt in session.get('review_packets', []):
        pkt['human_approved'] = True

    _sessions[job_id] = session
    _save_session(session, output_dir or '')

    return {
        'status': 200,
        'message': f'Review session {job_id} approved',
        'data': session,
    }


def reclassify_document(job_id: str, packet_id: str, new_document_type: str,
                         reason: str = "", reviewer: str = "", output_dir: str = None) -> dict:
    """
    POST /api/review/{job_id}/reclassify

    Reclassify a specific document packet to a different document type.

    USE CASE: The AI classified a document as "Packing List" but the reviewer
    sees it is actually a "Weight List". This correction ensures Step 14
    checks the right conditions against the right document.
    """
    session = _load_session(job_id, output_dir or '')
    if not session:
        return {'error': f'Review session not found for job {job_id}', 'status': 404}

    # Cannot modify after approval — prevents accidental changes during verification
    if session.get('status') == 'approved':
        return {'error': 'Cannot modify approved session', 'status': 400}

    # Find the packet by its unique ID
    target = None
    for pkt in session.get('review_packets', []):
        if pkt.get('packet_id') == packet_id:
            target = pkt
            break

    if not target:
        return {'error': f'Packet {packet_id} not found', 'status': 404}

    old_type = target.get('document_type', '')

    # Record change in audit log — captures what changed and why
    entry = AuditEntry(
        action="reclassify",
        target_packet_id=packet_id,
        old_value=old_type,
        new_value=new_document_type,
        reason=reason,
        reviewer=reviewer,
    )
    session.setdefault('audit_log', []).append(asdict(entry))

    # Apply the reclassification
    target['human_document_type'] = new_document_type
    target['document_type'] = new_document_type
    target['classification_status'] = 'matched_document'  # Human says it's a match
    session['human_changes_count'] = session.get('human_changes_count', 0) + 1
    session['updated_at'] = datetime.utcnow().isoformat() + "Z"

    _sessions[job_id] = session
    _save_session(session, output_dir or '')

    return {
        'status': 200,
        'message': f'Packet {packet_id} reclassified from "{old_type}" to "{new_document_type}"',
        'data': target,
    }


def mark_alien(job_id: str, packet_id: str, is_alien: bool = True,
               reason: str = "", reviewer: str = "", output_dir: str = None) -> dict:
    """
    POST /api/review/{job_id}/mark_alien

    Mark or unmark a document packet as alien (not required by the LC).

    TRADE FINANCE CONTEXT:
    An "alien document" is one that was included in the document package but is
    not required by the Letter of Credit. Examples: internal memos, duplicate
    copies, or documents from a different transaction. Alien documents are
    excluded from compliance verification — they would produce false failures
    if checked against LC conditions they were never meant to satisfy.
    """
    session = _load_session(job_id, output_dir or '')
    if not session:
        return {'error': f'Review session not found for job {job_id}', 'status': 404}

    if session.get('status') == 'approved':
        return {'error': 'Cannot modify approved session', 'status': 400}

    # Find the packet
    target = None
    for pkt in session.get('review_packets', []):
        if pkt.get('packet_id') == packet_id:
            target = pkt
            break

    if not target:
        return {'error': f'Packet {packet_id} not found', 'status': 404}

    action = "mark_alien" if is_alien else "unmark_alien"
    old_status = target.get('classification_status', '')
    new_status = 'alien_document' if is_alien else 'matched_document'

    # Record change in audit log
    entry = AuditEntry(
        action=action,
        target_packet_id=packet_id,
        old_value=old_status,
        new_value=new_status,
        reason=reason,
        reviewer=reviewer,
    )
    session.setdefault('audit_log', []).append(asdict(entry))

    # Apply change
    target['classification_status'] = new_status
    session['human_changes_count'] = session.get('human_changes_count', 0) + 1
    session['updated_at'] = datetime.utcnow().isoformat() + "Z"

    # Recount alien and matched documents after the change
    session['alien_count'] = sum(
        1 for p in session.get('review_packets', [])
        if p.get('classification_status') == 'alien_document'
    )
    session['matched_count'] = sum(
        1 for p in session.get('review_packets', [])
        if p.get('classification_status') == 'matched_document'
    )

    _sessions[job_id] = session
    _save_session(session, output_dir or '')

    return {
        'status': 200,
        'message': f'Packet {packet_id} {"marked as alien" if is_alien else "unmarked as alien"}',
        'data': target,
    }


# ── Main Run Function ──

def run(step7_result: dict, step9_result: dict, step10_result: dict = None,
        job_id: str = None, output_dir: str = None, progress_callback=None) -> dict:
    """
    Execute Step 11: Create a human review session.

    This function assembles all the data a human reviewer needs to inspect:
    1. Final LC summary (key fields like LC number, amount, expiry)
    2. Document packets with AI classification and confidence scores
    3. Traceability warnings from the pipeline (e.g., OCR quality issues)
    4. Items needing special attention (low confidence, ambiguous, alien)

    The session is persisted to disk and can be retrieved via the API.

    Args:
        step7_result: Output from Step 7 with 'structured_lc', 'required_documents'
        step9_result: Output from Step 9 with 'reconciled_packets'
        step10_result: Output from Step 10 with 'traceability_report' (optional)
        job_id: Unique job identifier (auto-generated if not provided)
        output_dir: Directory to save results
        progress_callback: Optional callback for progress updates

    Returns:
        dict with 'review_session', 'job_id', 'elapsed_seconds'
    """
    def _progress(msg):
        if progress_callback:
            progress_callback(f"[Step 11] {msg}")
        print(f"[Step 11] {msg}")

    start_time = time.time()

    if not job_id:
        job_id = str(uuid.uuid4())[:8]

    _progress(f"Creating review session {job_id}...")

    # ── Build Final LC summary for the review UI ──
    # Extract key LC fields so the reviewer can see the LC context at a glance
    structured_lc = step7_result.get('structured_lc', {})
    required_docs = step7_result.get('required_documents', [])
    standalone = step7_result.get('standalone_fields', {})

    lc_summary = {
        'total_clauses': structured_lc.get('total_clauses', 0),
        'total_required_docs': len(required_docs),
        'required_document_names': [d.get('document_name', '?') for d in required_docs],
        'key_fields': {},
    }

    # Extract key LC fields for summary display
    # F20 = LC Number, F31D = Expiry, F32B = Amount, F59 = Beneficiary,
    # F50 = Applicant, F44E = Port of Loading, F44F = Port of Discharge,
    # F44C = Latest Shipment Date
    for tag in ['F20', 'F31D', 'F32B', 'F59', 'F50', 'F44E', 'F44F', 'F44C']:
        if tag in standalone:
            lc_summary['key_fields'][tag] = standalone[tag]

    # ── Build review packets from reconciled shipping documents ──
    # Each reconciled packet becomes a ReviewPacket for human inspection
    reconciled = step9_result.get('reconciled_packets', [])
    review_packets = []

    for pkt in reconciled:
        if not isinstance(pkt, dict):
            continue

        rpkt = ReviewPacket(
            packet_id=pkt.get('packet_id', ''),
            document_type=pkt.get('document_type', ''),
            classification_status=pkt.get('classification_status', 'unknown'),
            match_confidence=pkt.get('match_confidence', pkt.get('confidence', 0.0)),
            matched_requirement_name=pkt.get('matched_requirement_name', ''),
            original_pages=pkt.get('original_pages', []),
            page_image_paths=pkt.get('page_image_paths', []),
            # Truncate text to 2000 chars — enough for review UI display without overloading it
            refined_text=pkt.get('refined_text', pkt.get('cleaned_text', '')),
            extracted_fields=pkt.get('extracted_fields', {}),
            ambiguity_flag=pkt.get('ambiguity_flag', False),
            ambiguity_notes=pkt.get('ambiguity_notes', ''),
        )
        review_packets.append(asdict(rpkt))

    # ── Get traceability warnings from Step 10 ──
    # These warn about potential issues in the pipeline (e.g., low OCR confidence)
    trace_warnings = []
    pipeline_confidence = 0.0
    if step10_result:
        trace_warnings = step10_result.get('warnings', [])
        pipeline_confidence = step10_result.get('pipeline_confidence', 0.0)

    # ── Compute summary counts for the review dashboard ──
    matched = sum(1 for p in review_packets if p.get('classification_status') == 'matched_document')
    alien = sum(1 for p in review_packets if p.get('classification_status') == 'alien_document')
    ambiguous = sum(1 for p in review_packets if p.get('ambiguity_flag'))

    # ── Create the review session ──
    session = ReviewSession(
        job_id=job_id,
        status="pending_review",
        final_lc_summary=lc_summary,
        review_packets=review_packets,
        traceability_warnings=trace_warnings,
        pipeline_confidence=pipeline_confidence,
        total_packets=len(review_packets),
        matched_count=matched,
        alien_count=alien,
        ambiguous_count=ambiguous,
    )

    session_data = asdict(session)

    # Store in memory (for fast API access) and on disk (for persistence)
    _sessions[job_id] = session_data
    if output_dir:
        _save_session(session_data, output_dir)

    elapsed = time.time() - start_time

    _progress(f"Review session created:")
    _progress(f"  Job ID: {job_id}")
    _progress(f"  Packets: {len(review_packets)} total")
    _progress(f"  Matched: {matched}, Alien: {alien}, Ambiguous: {ambiguous}")
    _progress(f"  Pipeline confidence: {pipeline_confidence:.3f}")
    _progress(f"  Traceability warnings: {len(trace_warnings)}")

    # ── Identify items that need special human attention ──
    # These are flagged in the review UI so the reviewer knows where to focus
    attention_items = []
    for pkt in review_packets:
        needs_review = False
        reasons = []
        if pkt.get('ambiguity_flag'):
            needs_review = True
            reasons.append("ambiguous classification")
        if pkt.get('classification_status') == 'alien_document':
            needs_review = True
            reasons.append("alien document")
        if pkt.get('classification_status') == 'unknown':
            needs_review = True
            reasons.append("unclassified")
        # Confidence below 0.7 (70%) is considered unreliable
        if pkt.get('match_confidence', 1.0) < 0.7:
            needs_review = True
            reasons.append(f"low confidence ({pkt.get('match_confidence', 0):.2f})")

        if needs_review:
            attention_items.append({
                'packet_id': pkt.get('packet_id'),
                'document_type': pkt.get('document_type', 'Unknown'),
                'reasons': reasons,
            })

    if attention_items:
        _progress(f"  Items needing attention: {len(attention_items)}")
        for item in attention_items:
            _progress(f"    - {item['packet_id']}: {', '.join(item['reasons'])}")

    # ── Save step result to disk ──
    # Includes API endpoint references so the frontend knows where to call
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        result_file = os.path.join(output_dir, 'step11_result.json')
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({
                'step': 11,
                'step_name': 'Human Review and Approval Gate',
                'job_id': job_id,
                'status': 'pending_review',
                'total_packets': len(review_packets),
                'matched_count': matched,
                'alien_count': alien,
                'ambiguous_count': ambiguous,
                'attention_items': attention_items,
                'pipeline_confidence': pipeline_confidence,
                'elapsed_seconds': round(elapsed, 2),
                'api_endpoints': {
                    'get_review': f'/api/review/{job_id}',
                    'approve': f'/api/review/{job_id}/approve',
                    'reclassify': f'/api/review/{job_id}/reclassify',
                    'mark_alien': f'/api/review/{job_id}/mark_alien',
                },
                'review_session': session_data,
            }, f, indent=2, ensure_ascii=False)

    _progress(f"Step 11 complete in {elapsed:.1f}s")

    return {
        'review_session': session_data,
        'job_id': job_id,
        'status': 'pending_review',
        'attention_items': attention_items,
        'elapsed_seconds': round(elapsed, 2),
    }


# ── FastAPI Route Registration Helper ──

def register_routes(app):
    """
    Register review API routes with a FastAPI app.

    This allows the review endpoints to be mounted on an existing FastAPI app
    so the frontend can interact with the review session via REST calls.

    Usage:
        from steps.step11_human_review import register_routes
        register_routes(app)
    """
    from fastapi import Request
    from fastapi.responses import JSONResponse

    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')

    @app.get("/api/review/{job_id}")
    async def api_get_review(job_id: str):
        result = get_review(job_id, output_dir=results_dir)
        return JSONResponse(content=result, status_code=result.get('status', 200))

    @app.post("/api/review/{job_id}/approve")
    async def api_approve(job_id: str, request: Request):
        body = await request.json() if request.headers.get('content-type') == 'application/json' else {}
        result = approve_review(
            job_id,
            reviewer=body.get('reviewer', ''),
            notes=body.get('notes', ''),
            output_dir=results_dir,
        )
        return JSONResponse(content=result, status_code=result.get('status', 200))

    @app.post("/api/review/{job_id}/reclassify")
    async def api_reclassify(job_id: str, request: Request):
        body = await request.json()
        result = reclassify_document(
            job_id,
            packet_id=body.get('packet_id', ''),
            new_document_type=body.get('new_document_type', ''),
            reason=body.get('reason', ''),
            reviewer=body.get('reviewer', ''),
            output_dir=results_dir,
        )
        return JSONResponse(content=result, status_code=result.get('status', 200))

    @app.post("/api/review/{job_id}/mark_alien")
    async def api_mark_alien(job_id: str, request: Request):
        body = await request.json()
        result = mark_alien(
            job_id,
            packet_id=body.get('packet_id', ''),
            is_alien=body.get('is_alien', True),
            reason=body.get('reason', ''),
            reviewer=body.get('reviewer', ''),
            output_dir=results_dir,
        )
        return JSONResponse(content=result, status_code=result.get('status', 200))


if __name__ == '__main__':
    # CLI test mode — run with Step 7 and Step 9 result JSON files
    import sys as _sys
    if len(_sys.argv) < 3:
        print("Usage: python step11_human_review.py <step07_result.json> <step09_result.json> [step10_result.json]")
        _sys.exit(1)

    with open(_sys.argv[1], 'r', encoding='utf-8') as f:
        step7 = json.load(f)
    with open(_sys.argv[2], 'r', encoding='utf-8') as f:
        step9 = json.load(f)
    step10 = None
    if len(_sys.argv) >= 4:
        with open(_sys.argv[3], 'r', encoding='utf-8') as f:
            step10 = json.load(f)

    result = run(step7, step9, step10, output_dir=os.path.dirname(_sys.argv[1]))
    print(f"\nJob ID: {result['job_id']}")
    print(f"Status: {result['status']}")
    print(f"Attention items: {len(result['attention_items'])}")
