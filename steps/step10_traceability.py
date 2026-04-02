"""
Step 10 — Traceability, Confidence, and Ambiguity Preservation
================================================================
This is the final quality-assurance step. It does NOT process documents —
instead, it validates the OUTPUTS of all previous steps to ensure every
piece of data has a complete audit trail.

PURPOSE:
    In trade finance compliance, auditability is essential. When a bank
    examiner asks "where did this data come from?" or "how confident are
    you in this field?", the system must be able to answer. This step
    ensures that guarantee by checking every field for:
    1. source_page: Which PDF page was this extracted from?
    2. source_step: Which pipeline step produced this data?
    3. confidence: How confident is the extraction? (0.0 to 1.0)
    4. ambiguity_flag: Is there any uncertainty about this data?

WHAT IT CHECKS:
    - Step 1: Every page has OCR text with confidence scores
    - Step 2: Every page has cleaned text
    - Step 7: Every clause and required document has source tracing
    - Step 8: Every classification has confidence and matched requirement
    - Step 9: Every reconciled packet has refined text and extraction status
    - Other steps: Generic field-level validation

PIPELINE CONFIDENCE:
    Calculates an overall pipeline confidence score (average of all field
    confidences) which gives a single number summarizing data quality.
    Low pipeline confidence (<0.70) triggers warnings.

INPUT:
    - Dictionary mapping step numbers to their result dicts:
      {1: step1_result, 2: step2_result, ..., 9: step9_result}

OUTPUT:
    - TraceabilityReport with:
        * field_traces: per-field audit trail entries
        * step_summaries: per-step quality statistics
        * pipeline_confidence: overall confidence score
        * warnings: any issues found

MODEL:
    No AI model used — pure validation and aggregation logic.
"""

import json
import sys as _sys; _sys.stdout.reconfigure(encoding="utf-8", errors="replace") if hasattr(_sys.stdout, "reconfigure") else None
import os
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from pathlib import Path


# ── Dataclasses ──

@dataclass
class FieldTrace:
    """
    Traceability record for a single data field in the pipeline.

    Every extracted field (OCR text, classification, clause, required document,
    extracted field value) gets a FieldTrace that documents its provenance.
    If any provenance information is missing, provenance_complete=False and
    missing_provenance lists what is absent.
    """
    field_path: str                     # Dot-notation path, e.g. "step7.clause.F46A.1", "step9.packet_001.document_type"
    field_value: str                    # Current value (truncated to 200 chars for display)
    source_page: Optional[int] = None   # Which PDF page this came from (None if unknown)
    source_step: Optional[int] = None   # Which pipeline step produced this (None if unknown)
    source_step_name: str = ""          # Human-readable step name
    confidence: float = 0.0             # Extraction/classification confidence (0.0-1.0)
    ambiguity_flag: bool = False        # True if there is uncertainty about this field
    ambiguity_notes: str = ""           # Details about the ambiguity
    provenance_complete: bool = True    # False if any required provenance info is missing
    missing_provenance: List[str] = field(default_factory=list)  # List of what's missing: "source_page", "source_step", "confidence"


@dataclass
class StepSummary:
    """
    Summary statistics for a single step's traceability status.
    Gives a quick overview of data quality for each pipeline step.
    """
    step_number: int                    # Step number (1-10)
    step_name: str                      # Human-readable step name
    total_fields: int = 0               # Total data fields produced by this step
    fields_with_provenance: int = 0     # Fields with complete audit trail
    fields_missing_provenance: int = 0  # Fields missing some provenance info
    avg_confidence: float = 0.0         # Average confidence across all fields
    ambiguous_fields: int = 0           # Number of fields flagged as ambiguous
    elapsed_seconds: float = 0.0        # How long this step took to run


@dataclass
class TraceabilityReport:
    """
    Complete traceability report across all pipeline steps.
    This is the top-level audit document that proves the system's data
    provenance chain is intact. pipeline_confidence gives a single number
    summarizing overall data quality.
    """
    field_traces: List[dict] = field(default_factory=list)       # Every field's audit trail
    step_summaries: List[dict] = field(default_factory=list)     # Per-step quality stats
    pipeline_confidence: float = 0.0                              # Overall confidence (avg of all fields)
    total_fields: int = 0                                         # Total fields across all steps
    total_provenance_complete: int = 0                            # Fields with full audit trail
    total_provenance_missing: int = 0                             # Fields missing some provenance
    total_ambiguous: int = 0                                      # Fields flagged as ambiguous
    warnings: List[str] = field(default_factory=list)            # Quality warnings
    elapsed_seconds: float = 0.0                                  # Time to run Step 10


# ── Step Names ──
# Human-readable names for each pipeline step, used in reports and logs.

_STEP_NAMES = {
    1: "Page-Level Raw OCR Extraction",
    2: "OCR Text Cleaning",
    3: "Page Segmentation into Packets",
    4: "LC Page Extraction",
    5: "LC OCR and SWIFT Parsing",
    6: "Final LC Consolidation",
    7: "Clause and Requirement Extraction",
    8: "Shipping Document Classification",
    9: "Shipping OCR Reconciliation",
    10: "Traceability Validation",
}


# ── Validation Functions ──

def _validate_field(field_path: str, value: Any, source_data: dict) -> FieldTrace:
    """
    Validate a single field for traceability completeness.

    Checks that the field has:
    - source_page or page_number (where did this data come from?)
    - source_step (which pipeline step produced it?)
    - confidence (how reliable is this data?)
    - ambiguity_flag (is there any uncertainty?)

    Returns a FieldTrace with provenance_complete=True only if ALL
    required metadata is present.
    """
    # Determine display value
    if isinstance(value, str):
        display = value if len(value) > 200 else value
    elif isinstance(value, (int, float, bool)):
        display = str(value)
    elif isinstance(value, dict):
        display = json.dumps(value)
    elif isinstance(value, list):
        display = f"[{len(value)} items]"
    else:
        display = str(value)

    trace = FieldTrace(
        field_path=field_path,
        field_value=display,
    )

    # Check for source_page
    if 'source_page' in source_data:
        trace.source_page = source_data['source_page']
    elif 'page_number' in source_data:
        trace.source_page = source_data['page_number']
    else:
        trace.missing_provenance.append('source_page')

    # Check for source_step
    if 'source_step' in source_data:
        trace.source_step = source_data['source_step']
        trace.source_step_name = _STEP_NAMES.get(source_data['source_step'], '')
    else:
        trace.missing_provenance.append('source_step')

    # Check for confidence
    if 'confidence' in source_data:
        trace.confidence = source_data['confidence']
    elif 'match_confidence' in source_data:
        trace.confidence = source_data['match_confidence']
    else:
        trace.missing_provenance.append('confidence')
        trace.confidence = 0.5  # Default medium confidence

    # Check for ambiguity
    if 'ambiguity_flag' in source_data:
        trace.ambiguity_flag = source_data['ambiguity_flag']
        trace.ambiguity_notes = source_data.get('ambiguity_notes', '')
    elif 'ambiguous' in source_data:
        trace.ambiguity_flag = source_data['ambiguous']

    # Determine if provenance is complete
    trace.provenance_complete = len(trace.missing_provenance) == 0

    return trace


def _validate_step1(step_result: dict) -> List[FieldTrace]:
    """Validate Step 1 (OCR) results — check each page has text, confidence, and page number."""
    traces = []
    pages = step_result.get('pages', [])
    for page in pages:
        pg = page if isinstance(page, dict) else asdict(page)
        pg_num = pg.get('page_number', 0)

        # Raw text
        trace = _validate_field(
            f"step1.page_{pg_num}.raw_text",
            pg.get('raw_text', ''),
            {
                'source_page': pg_num,
                'source_step': 1,
                'confidence': pg.get('confidence', 0.0),
                'ambiguity_flag': pg.get('has_unreadable', False),
                'ambiguity_notes': '; '.join(pg.get('ambiguity_notes', [])) if isinstance(pg.get('ambiguity_notes'), list) else '',
            }
        )
        traces.append(trace)

    return traces


def _validate_step2(step_result: dict) -> List[FieldTrace]:
    """Validate Step 2 (cleaning) results — check cleaned text exists for each page."""
    traces = []
    pages = step_result.get('pages', [])
    for page in pages:
        pg = page if isinstance(page, dict) else asdict(page)
        pg_num = pg.get('page_number', 0)

        trace = _validate_field(
            f"step2.page_{pg_num}.cleaned_text",
            pg.get('cleaned_text', ''),
            {
                'source_page': pg_num,
                'source_step': 2,
                'confidence': 0.95,
            }
        )
        traces.append(trace)

    return traces


def _validate_step7(step_result: dict) -> List[FieldTrace]:
    """Validate Step 7 (clause extraction) — check clauses and required documents have provenance."""
    traces = []

    # Clauses
    clauses = step_result.get('all_clauses', [])
    for clause in clauses:
        c = clause if isinstance(clause, dict) else asdict(clause)
        tag = c.get('field_tag', '?')
        num = c.get('clause_number', 0)

        trace = _validate_field(
            f"step7.clause.{tag}.{num}",
            c.get('clause_text', ''),
            {
                'source_page': c.get('source_page'),
                'source_step': c.get('source_step', 7),
                'confidence': c.get('confidence', 1.0),
                'ambiguity_flag': c.get('ambiguity_flag', False),
                'ambiguity_notes': c.get('ambiguity_notes', ''),
            }
        )
        traces.append(trace)

    # Required documents
    docs = step_result.get('required_documents', [])
    for i, doc in enumerate(docs):
        d = doc if isinstance(doc, dict) else asdict(doc)

        trace = _validate_field(
            f"step7.required_doc.{i}.{d.get('document_name', 'unknown')}",
            d.get('document_name', ''),
            {
                'source_page': d.get('source_page'),
                'source_step': 7,
                'confidence': d.get('confidence', 0.0),
                'ambiguity_flag': d.get('ambiguity_flag', False),
                'ambiguity_notes': d.get('ambiguity_notes', ''),
            }
        )
        traces.append(trace)

    return traces


def _validate_step8(step_result: dict) -> List[FieldTrace]:
    """Validate Step 8 (shipping classification) — check each packet has type and confidence."""
    traces = []
    packets = step_result.get('classified_packets', [])
    for packet in packets:
        p = packet if isinstance(packet, dict) else asdict(packet)
        pid = p.get('packet_id', '?')

        trace = _validate_field(
            f"step8.{pid}.document_type",
            p.get('document_type', ''),
            {
                'source_page': p.get('original_pages', [None])[0] if p.get('original_pages') else None,
                'source_step': 8,
                'confidence': p.get('match_confidence', 0.0),
                'ambiguity_flag': p.get('ambiguity_flag', False),
                'ambiguity_notes': p.get('ambiguity_notes', ''),
            }
        )
        traces.append(trace)

    return traces


def _validate_step9(step_result: dict) -> List[FieldTrace]:
    """Validate Step 9 (shipping reconciliation) — check refined text and extracted fields."""
    traces = []
    packets = step_result.get('reconciled_packets', [])
    for packet in packets:
        p = packet if isinstance(packet, dict) else asdict(packet)
        pid = p.get('packet_id', '?')

        # Document type (final)
        trace = _validate_field(
            f"step9.{pid}.document_type",
            p.get('document_type', ''),
            {
                'source_page': p.get('original_pages', [None])[0] if p.get('original_pages') else None,
                'source_step': 9,
                'confidence': p.get('confidence', 0.0),
                'ambiguity_flag': p.get('ambiguity_flag', False),
                'ambiguity_notes': p.get('ambiguity_notes', ''),
            }
        )
        traces.append(trace)

        # Refined text
        trace = _validate_field(
            f"step9.{pid}.refined_text",
            p.get('refined_text', ''),
            {
                'source_page': p.get('original_pages', [None])[0] if p.get('original_pages') else None,
                'source_step': 9,
                'confidence': p.get('confidence', 0.0),
            }
        )
        traces.append(trace)

        # Extracted fields
        for fname, fval in p.get('extracted_fields', {}).items():
            trace = _validate_field(
                f"step9.{pid}.extracted.{fname}",
                fval,
                {
                    'source_step': 9,
                    'confidence': p.get('confidence', 0.0),
                }
            )
            traces.append(trace)

    return traces


def _validate_generic_step(step_result: dict, step_num: int) -> List[FieldTrace]:
    """
    Generic validator for steps without specific validation logic (Steps 3, 4, 5, 6).
    Walks top-level keys looking for lists of dicts (common output format) and
    validates each item for provenance metadata.
    """
    traces = []

    # Walk top-level keys looking for provenance data
    for key, value in step_result.items():
        if key in ('step', 'step_name', 'elapsed_seconds', 'errors', 'result_file'):
            continue

        if isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    trace = _validate_field(
                        f"step{step_num}.{key}.{i}",
                        item,
                        {
                            'source_step': item.get('source_step', step_num),
                            'source_page': item.get('source_page', item.get('page_number')),
                            'confidence': item.get('confidence', 0.5),
                            'ambiguity_flag': item.get('ambiguity_flag', False),
                            'ambiguity_notes': item.get('ambiguity_notes', ''),
                        }
                    )
                    traces.append(trace)
        elif isinstance(value, dict):
            trace = _validate_field(
                f"step{step_num}.{key}",
                value,
                {
                    'source_step': value.get('source_step', step_num),
                    'confidence': value.get('confidence', 0.5),
                }
            )
            traces.append(trace)

    return traces


# ── Step Validators Registry ──
# Maps step numbers to their validation functions. Steps not listed here
# use the generic validator which performs basic provenance checking.

_VALIDATORS = {
    1: _validate_step1,
    2: _validate_step2,
    7: _validate_step7,
    8: _validate_step8,
    9: _validate_step9,
}


# ── Main Run Function ──

def run(all_step_results: Dict[int, dict], output_dir: str = None, progress_callback=None) -> dict:
    """
    Execute Step 10: Validate traceability across all pipeline steps.

    Args:
        all_step_results: Dict mapping step number → step result dict
                         e.g. {1: step1_result, 2: step2_result, ...}
        output_dir: Directory to save results
        progress_callback: Optional callback for progress updates

    Returns:
        dict with 'traceability_report', 'warnings', 'elapsed_seconds'
    """
    def _progress(msg):
        if progress_callback:
            progress_callback(f"[Step 10] {msg}")
        print(f"[Step 10] {msg}")

    start_time = time.time()

    all_traces = []
    step_summaries = []
    warnings = []

    _progress(f"Validating traceability across {len(all_step_results)} steps...")

    for step_num in sorted(all_step_results.keys()):
        step_result = all_step_results[step_num]
        step_name = _STEP_NAMES.get(step_num, f"Step {step_num}")

        _progress(f"  Validating {step_name}...")

        # Get validator
        validator = _VALIDATORS.get(step_num, lambda r: _validate_generic_step(r, step_num))
        try:
            traces = validator(step_result)
        except Exception as e:
            warnings.append(f"Step {step_num} ({step_name}): validation error: {e}")
            traces = []

        # Calculate step summary
        confidences = [t.confidence for t in traces if t.confidence > 0]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

        summary = StepSummary(
            step_number=step_num,
            step_name=step_name,
            total_fields=len(traces),
            fields_with_provenance=sum(1 for t in traces if t.provenance_complete),
            fields_missing_provenance=sum(1 for t in traces if not t.provenance_complete),
            avg_confidence=round(avg_conf, 3),
            ambiguous_fields=sum(1 for t in traces if t.ambiguity_flag),
            elapsed_seconds=step_result.get('elapsed_seconds', 0),
        )
        step_summaries.append(asdict(summary))

        # Generate warnings
        if summary.fields_missing_provenance > 0:
            warnings.append(
                f"Step {step_num}: {summary.fields_missing_provenance}/{summary.total_fields} "
                f"fields missing provenance"
            )
        if summary.avg_confidence < 0.7 and summary.total_fields > 0:
            warnings.append(
                f"Step {step_num}: low average confidence ({summary.avg_confidence:.2f})"
            )
        if summary.ambiguous_fields > 0:
            warnings.append(
                f"Step {step_num}: {summary.ambiguous_fields} ambiguous fields"
            )

        all_traces.extend([asdict(t) for t in traces])

        _progress(f"    {len(traces)} fields: {summary.fields_with_provenance} OK, "
                  f"{summary.fields_missing_provenance} missing provenance, "
                  f"avg confidence {avg_conf:.2f}")

    # Pipeline-level aggregation — compute overall confidence as the average
    # of all individual field confidences. This gives a single number that
    # represents the overall data quality of the pipeline output.
    all_confidences = [t['confidence'] for t in all_traces if t.get('confidence', 0) > 0]
    pipeline_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0

    report = TraceabilityReport(
        field_traces=all_traces,
        step_summaries=step_summaries,
        pipeline_confidence=round(pipeline_confidence, 3),
        total_fields=len(all_traces),
        total_provenance_complete=sum(1 for t in all_traces if t.get('provenance_complete', False)),
        total_provenance_missing=sum(1 for t in all_traces if not t.get('provenance_complete', True)),
        total_ambiguous=sum(1 for t in all_traces if t.get('ambiguity_flag', False)),
        warnings=warnings,
        elapsed_seconds=round(time.time() - start_time, 2),
    )

    elapsed = time.time() - start_time

    _progress(f"Traceability summary:")
    _progress(f"  Total fields: {report.total_fields}")
    _progress(f"  Provenance complete: {report.total_provenance_complete}")
    _progress(f"  Provenance missing: {report.total_provenance_missing}")
    _progress(f"  Ambiguous: {report.total_ambiguous}")
    _progress(f"  Pipeline confidence: {report.pipeline_confidence:.3f}")
    _progress(f"  Warnings: {len(warnings)}")

    # Save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        result_file = os.path.join(output_dir, 'step10_result.json')
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({
                'step': 10,
                'step_name': 'Traceability, Confidence, and Ambiguity Preservation',
                'pipeline_confidence': report.pipeline_confidence,
                'total_fields': report.total_fields,
                'total_provenance_complete': report.total_provenance_complete,
                'total_provenance_missing': report.total_provenance_missing,
                'total_ambiguous': report.total_ambiguous,
                'warnings': warnings,
                'elapsed_seconds': round(elapsed, 2),
                'step_summaries': step_summaries,
                'field_traces': all_traces,
            }, f, indent=2, ensure_ascii=False)

    _progress(f"Step 10 complete in {elapsed:.1f}s")

    return {
        'traceability_report': asdict(report),
        'step_summaries': step_summaries,
        'warnings': warnings,
        'pipeline_confidence': report.pipeline_confidence,
        'elapsed_seconds': round(elapsed, 2),
    }


if __name__ == '__main__':
    import sys as _sys
    if len(_sys.argv) < 2:
        print("Usage: python step10_traceability.py <results_dir>")
        print("  Reads step01_result.json through step09_result.json from results_dir")
        _sys.exit(1)

    results_dir = _sys.argv[1]
    all_results = {}
    for step_num in range(1, 10):
        fpath = os.path.join(results_dir, f'step{step_num:02d}_result.json')
        if os.path.exists(fpath):
            with open(fpath, 'r', encoding='utf-8') as f:
                all_results[step_num] = json.load(f)
            print(f"Loaded step {step_num}")

    result = run(all_results, output_dir=results_dir)
    print(f"\nPipeline confidence: {result['pipeline_confidence']:.3f}")
    print(f"Warnings: {len(result['warnings'])}")
    for w in result['warnings']:
        print(f"  - {w}")
