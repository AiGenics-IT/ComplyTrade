"""
Step 9 -- Shipping OCR Reconciliation
=======================================
For each classified shipping packet, sends image + GLM text + classification
to Qwen VLM. The VLM:
  1. Reviews GLM text against the page image
  2. ADDS any text GLM missed (stamps, handwritten notes, small print)
  3. Does NOT change or rewrite existing GLM text
  4. Extracts document-type-specific fields

KEY PRINCIPLE: GLM OCR already extracted trusted text (Step 1).
Qwen VLM only adds missing things -- never rewrites GLM text.

TEXT PROVENANCE (after this step, shipping docs have 3 layers):
    raw_text    -- GLM OCR output (Step 1)
    refined_text -- GLM text + VLM additions (this step)
    change_log  -- audit trail of what was added and why

INPUT:  Classified packets from Step 8 + required docs from Step 7
OUTPUT: Reconciled packets with refined text, extracted fields, change_log
"""

import json
import sys as _sys
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import os
import re
import time
import base64
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import QWEN_VLM_URL, QWEN_VLM_MODEL, MAX_CONCURRENT_VLM, VLM_TIMEOUT


# ── Dataclasses ──

@dataclass
class ChangeLogEntry:
    """Records a single change made during reconciliation."""
    field: str                         # "text_addition", "document_type", "field_extraction"
    old_value: str = ""
    new_value: str = ""
    reason: str = ""
    source: str = "vlm_reconciliation"
    timestamp: float = 0.0


@dataclass
class ReconciledPacket:
    """A shipping packet after VLM-based OCR reconciliation."""
    packet_id: str
    original_pages: List[int] = field(default_factory=list)
    page_image_paths: List[str] = field(default_factory=list)

    # Text provenance chain
    raw_text: str = ""                      # GLM OCR output (Step 1)
    cleaned_text: str = ""                  # After deterministic cleaning (Step 2)
    refined_text: str = ""                  # GLM text + VLM additions (this step)

    # Classification (carried from Step 8, may be updated)
    document_type: str = ""
    classification_status: str = "unknown"
    match_confidence: float = 0.0
    matched_requirement_index: int = -1
    matched_requirement_name: str = ""
    was_reclassified: bool = False
    previous_document_type: str = ""

    # Document summary and visual elements (carried from Step 8)
    document_summary: str = ""
    document_number: str = ""
    document_date: str = ""
    document_amount: str = ""
    stamps: List[dict] = field(default_factory=list)
    signatures: List[dict] = field(default_factory=list)
    seals: List[dict] = field(default_factory=list)
    logos: List[dict] = field(default_factory=list)
    copy_status: str = ""
    copy_label: str = ""
    marking_status: str = ""
    issued_by: str = ""
    lc_reference: str = ""

    # Document-type-specific extracted fields
    extracted_fields: Dict[str, Any] = field(default_factory=dict)

    # P129 — Carry forward Step 3 structured facts for deterministic Step 14
    unified_summary: Dict[str, Any] = field(default_factory=dict)
    bl_subtype: Dict[str, Any] = field(default_factory=dict)
    validation_status: str = ""

    # Change tracking
    change_log: List[dict] = field(default_factory=list)

    # Metadata
    source_step: int = 9
    confidence: float = 0.0
    ambiguity_flag: bool = False
    ambiguity_notes: str = ""
    elapsed_seconds: float = 0.0


# ── Document-Type-Specific Field Lists ──

_DOC_TYPE_FIELDS = {
    "Bill of Lading": (
        "shipper, consignee, notify_party, vessel, voyage, "
        "port_of_loading, port_of_discharge, bl_number, bl_date, "
        "shipped_on_board, shipped_on_board_date, freight_status, "
        "number_of_originals, goods_description, gross_weight, "
        "net_weight, number_of_packages, marks_and_numbers, "
        "place_of_receipt, place_of_delivery"
    ),
    "Commercial Invoice": (
        "invoice_number, invoice_date, amount, currency, "
        "goods_description, hs_code, ntn_number, "
        "incoterms (FULL term including suffixes like FO/FI e.g. 'CFR FO' not just 'CFR'), "
        "unit_price, quantity, total_amount, amount_in_words, "
        "buyer_name, seller_name, lc_reference"
    ),
    "Draft": (
        "drawee, drawer, amount, amount_in_words, currency, "
        "tenor, at_sight, lc_reference, draft_date, draft_number"
    ),
    "Insurance Policy/Certificate": (
        "policy_number, insured_party, sum_insured, currency, "
        "coverage_type, voyage_from, voyage_to, goods_description, "
        "claims_agent, premium, effective_date"
    ),
    "Certificate of Origin": (
        "certificate_number, country_of_origin, exporter_name, "
        "importer_name, goods_description, issuing_authority, "
        "certification_date, hs_code"
    ),
    "Packing List": (
        "total_packages, net_weight, gross_weight, dimensions, "
        "marks_and_numbers, goods_description, packing_details"
    ),
    "Weight List": (
        "total_net_weight, total_gross_weight, tare_weight, "
        "weight_per_package, number_of_packages"
    ),
    "Beneficiary Certificate": (
        "certificate_text, certifying_party, certification_date, "
        "lc_reference"
    ),
    "Inspection Certificate": (
        "certificate_number, inspector_name, inspection_date, "
        "goods_description, inspection_result, surveyor_company"
    ),
    "Shipping Advice": (
        "vessel_name, voyage_number, bl_number, bl_date, "
        "port_of_loading, port_of_discharge, goods_description, "
        "estimated_arrival"
    ),
    "Fumigation Certificate": (
        "certificate_number, fumigation_date, chemicals_used, "
        "treatment_duration, goods_description"
    ),
    "Phytosanitary Certificate": (
        "certificate_number, issuing_authority, inspection_date, "
        "place_of_origin, goods_description"
    ),
    "Health Certificate": (
        "certificate_number, issuing_authority, inspection_date, "
        "goods_description, fitness_declaration"
    ),
}

_DEFAULT_FIELDS = (
    "document_number, document_date, issuer, reference_numbers, "
    "key_text, any_amounts"
)


# ── Reconciliation Prompt ──

_RECONCILIATION_PROMPT = """This document was classified as: {document_type}

GLM OCR extracted this text:
{glm_text}

Review the image against the GLM text.
- Only ADD text that GLM missed (e.g., text in stamps, handwritten notes, small print)
- Do NOT change or rewrite existing GLM text
- Report what you added and why

Also extract these fields specific to {document_type}:
[{field_list}]

Return ONLY valid JSON:
{{
    "refined_text": "the GLM text with any additions appended at the end",
    "additions": [
        {{"text": "added text", "location": "where on the page", "reason": "why GLM missed it"}}
    ],
    "document_fields": {{
        "field_name": "field_value"
    }},
    "reclassify": false,
    "new_document_type": "",
    "reclassify_reason": "",
    "reclassify_confidence": 0.0
}}"""


def _reconcile_single_packet(packet: dict, expected_docs: List[dict], packet_index: int) -> dict:
    """
    Reconcile a single shipping packet: send image + GLM text + classification
    to Qwen VLM for text additions and field extraction.
    """
    start = time.time()

    doc_type = packet.get('document_type', '')
    glm_text = packet.get('cleaned_text', packet.get('raw_text', ''))
    image_paths = packet.get('page_image_paths', [])
    original_status = packet.get('classification_status', 'unknown')

    change_log = []

    # Get document-type-specific field list
    field_list = _DOC_TYPE_FIELDS.get(doc_type, _DEFAULT_FIELDS)

    # Build prompt
    prompt = _RECONCILIATION_PROMPT.format(
        document_type=doc_type if doc_type else "Unknown",
        glm_text=glm_text[:5000],
        field_list=field_list,
    )

    # Build VLM request with images
    content_parts = []
    if image_paths:
        for img_path in image_paths[:2]:  # Max 2 pages
            img_str = str(img_path)
            if os.path.exists(img_str):
                try:
                    with open(img_str, 'rb') as f:
                        img_b64 = base64.b64encode(f.read()).decode('utf-8')
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,%s" % img_b64}
                    })
                except Exception:
                    pass

    content_parts.append({"type": "text", "text": prompt})

    # Call VLM
    refined_text = glm_text  # Default: keep GLM text unchanged
    extracted_fields = {}
    was_reclassified = False
    previous_type = ""
    additions = []

    try:
        resp = requests.post(QWEN_VLM_URL, json={
            "model": QWEN_VLM_MODEL,
            "messages": [{"role": "user", "content": content_parts}],
            "max_tokens": 4000,
            "temperature": 0.1,
        }, timeout=None)

        if resp.status_code == 200:
            result = resp.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")

            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                vlm_data = json.loads(json_match.group(0))

                # Refined text (GLM text + additions)
                new_text = vlm_data.get('refined_text', '')
                if new_text and len(new_text) >= len(glm_text) * 0.9:
                    refined_text = new_text
                    change_log.append(asdict(ChangeLogEntry(
                        field="text_addition",
                        old_value="[%d chars]" % len(glm_text),
                        new_value="[%d chars]" % len(refined_text),
                        reason="VLM added missing text",
                        timestamp=time.time(),
                    )))

                # Log individual additions
                additions = vlm_data.get('additions', [])
                if isinstance(additions, list):
                    for add in additions:
                        if isinstance(add, dict) and add.get('text'):
                            change_log.append(asdict(ChangeLogEntry(
                                field="text_addition",
                                old_value="",
                                new_value=str(add.get('text', '')),
                                reason="VLM found: %s (%s)" % (
                                    add.get('location', 'unknown'),
                                    add.get('reason', 'missed by GLM')),
                                timestamp=time.time(),
                            )))

                # Reclassification check
                if vlm_data.get('reclassify', False):
                    new_type = vlm_data.get('new_document_type', '')
                    reclass_conf = float(vlm_data.get('reclassify_confidence', 0.0))
                    reclass_reason = vlm_data.get('reclassify_reason', '')

                    if new_type and reclass_conf > 0.7 and new_type != doc_type:
                        previous_type = doc_type
                        doc_type = new_type
                        was_reclassified = True

                        change_log.append(asdict(ChangeLogEntry(
                            field="document_type",
                            old_value=previous_type,
                            new_value=doc_type,
                            reason="VLM reclassification (%.2f): %s" % (
                                reclass_conf, reclass_reason),
                            timestamp=time.time(),
                        )))

                        # Try to match new type to expected docs
                        matched_idx = -1
                        for i, ed in enumerate(expected_docs):
                            ed_name = ed.get('document_name', '').upper()
                            dt_upper = doc_type.upper()
                            if dt_upper in ed_name or ed_name in dt_upper:
                                matched_idx = i
                                break

                        if matched_idx >= 0:
                            packet['matched_requirement_index'] = matched_idx
                            packet['matched_requirement_name'] = expected_docs[matched_idx].get('document_name', '')
                            packet['classification_status'] = 'matched_document'
                        else:
                            packet['classification_status'] = 'alien_document'

                # Extracted fields
                extracted_fields = vlm_data.get('document_fields', {})
                if not isinstance(extracted_fields, dict):
                    extracted_fields = {}

    except Exception as e:
        print("[Step 9] VLM reconciliation failed for packet %d: %s" % (packet_index, e))
        change_log.append(asdict(ChangeLogEntry(
            field="reconciliation",
            old_value="",
            new_value="",
            reason="VLM call failed: %s" % e,
            timestamp=time.time(),
        )))

    elapsed = time.time() - start

    # Confidence adjustment
    confidence = float(packet.get('match_confidence', 0.0))
    if refined_text != glm_text:
        confidence = min(confidence + 0.05, 1.0)
    if was_reclassified:
        confidence = max(confidence - 0.1, 0.0)

    # Ambiguity
    ambiguity = packet.get('ambiguity_flag', False)
    ambiguity_notes = packet.get('ambiguity_notes', '')
    if was_reclassified:
        ambiguity = True
        if ambiguity_notes:
            ambiguity_notes += "; "
        ambiguity_notes += "Reclassified from %s to %s" % (previous_type, doc_type)

    # Carry forward Step 8 visual elements
    reconciled = ReconciledPacket(
        packet_id=packet.get('packet_id', "packet_%03d" % packet_index),
        original_pages=packet.get('original_pages', []),
        page_image_paths=packet.get('page_image_paths', []),
        raw_text=packet.get('raw_text', ''),
        cleaned_text=packet.get('cleaned_text', glm_text),
        refined_text=refined_text,
        document_type=doc_type,
        classification_status=packet.get('classification_status', original_status),
        match_confidence=confidence,
        matched_requirement_index=packet.get('matched_requirement_index', -1),
        matched_requirement_name=packet.get('matched_requirement_name', ''),
        was_reclassified=was_reclassified,
        previous_document_type=previous_type,
        # Carry forward Step 8 visual elements
        document_summary=packet.get('document_summary', ''),
        document_number=packet.get('document_number', ''),
        document_date=packet.get('document_date', ''),
        document_amount=packet.get('document_amount', ''),
        stamps=packet.get('stamps', []),
        signatures=packet.get('signatures', []),
        seals=packet.get('seals', []),
        logos=packet.get('logos', []),
        copy_status=packet.get('copy_status', ''),
        copy_label=packet.get('copy_label', ''),
        marking_status=packet.get('marking_status', ''),
        issued_by=packet.get('issued_by', ''),
        lc_reference=packet.get('lc_reference', ''),
        # New from this step
        extracted_fields=extracted_fields,
        change_log=change_log,
        confidence=confidence,
        ambiguity_flag=ambiguity,
        ambiguity_notes=ambiguity_notes,
        elapsed_seconds=round(elapsed, 2),
        # P129 — carry forward Step 3 structured facts
        unified_summary=packet.get('unified_summary', {}) or {},
        bl_subtype=packet.get('bl_subtype', {}) or {},
        validation_status=packet.get('validation_status', '') or '',
    )

    return asdict(reconciled)


# ── Main Run Function ──

def run(step8_result: dict, step7_result: dict = None, output_dir: str = None, progress_callback=None) -> dict:
    """
    Execute Step 9: Reconcile OCR text and extract document-specific fields.

    Args:
        step8_result: Output from Step 8 with 'classified_packets'
        step7_result: Output from Step 7 with 'required_documents' (optional)
        output_dir: Directory to save results
        progress_callback: Optional callback for progress updates

    Returns:
        dict with 'reconciled_packets', 'summary', 'elapsed_seconds'
    """
    def _progress(msg):
        if progress_callback:
            progress_callback("[Step 9] %s" % msg)
        print("[Step 9] %s" % msg)

    start_time = time.time()

    packets = step8_result.get('classified_packets', [])
    required_docs = (step7_result or step8_result).get('required_documents', [])

    _progress("Reconciling %d classified packets..." % len(packets))

    if not packets:
        return {
            'reconciled_packets': [],
            'summary': {'total': 0, 'refined': 0, 'reclassified': 0},
            'elapsed_seconds': 0,
        }

    # Process packets concurrently
    reconciled_packets = [None] * len(packets)

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_VLM) as executor:
        futures = {}
        for idx, packet in enumerate(packets):
            future = executor.submit(_reconcile_single_packet, packet, required_docs, idx)
            futures[future] = idx

        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                reconciled_packets[idx] = result
                doc_type = result.get('document_type', '?')
                reclass = " [RECLASSIFIED]" if result.get('was_reclassified') else ""
                changes = len(result.get('change_log', []))
                n_fields = len(result.get('extracted_fields', {}))
                _progress("  Packet %d: %s%s (%d changes, %d fields)" % (
                    idx, doc_type, reclass, changes, n_fields))
            except Exception as e:
                _progress("  Packet %d: ERROR - %s" % (idx, e))
                reconciled_packets[idx] = asdict(ReconciledPacket(
                    packet_id="packet_%03d" % idx,
                    ambiguity_flag=True,
                    ambiguity_notes="Reconciliation error: %s" % e,
                ))

    # Summary
    total_refined = sum(
        1 for p in reconciled_packets
        if p and p.get('refined_text', '') != p.get('cleaned_text', '')
    )
    total_reclassified = sum(
        1 for p in reconciled_packets
        if p and p.get('was_reclassified', False)
    )

    summary = {
        'total': len(reconciled_packets),
        'refined': total_refined,
        'reclassified': total_reclassified,
        'matched': sum(1 for p in reconciled_packets if p and p.get('classification_status') == 'matched_document'),
        'alien': sum(1 for p in reconciled_packets if p and p.get('classification_status') == 'alien_document'),
        'extra': sum(1 for p in reconciled_packets if p and p.get('classification_status') == 'extra_document'),
        'unknown': sum(1 for p in reconciled_packets if p and p.get('classification_status') == 'unknown'),
        'ambiguous': sum(1 for p in reconciled_packets if p and p.get('ambiguity_flag')),
        'fields_extracted': sum(
            len(p.get('extracted_fields', {}))
            for p in reconciled_packets if p
        ),
    }

    elapsed = time.time() - start_time

    # Save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        result_file = os.path.join(output_dir, 'step09_result.json')
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({
                'step': 9,
                'step_name': 'Shipping OCR Reconciliation',
                'total_packets': len(reconciled_packets),
                'summary': summary,
                'elapsed_seconds': round(elapsed, 2),
                'reconciled_packets': reconciled_packets,
            }, f, indent=2, ensure_ascii=False)

    _progress("Step 9 complete: %d refined, %d reclassified, %d fields in %.1fs" % (
        total_refined, total_reclassified, summary['fields_extracted'], elapsed))

    return {
        'reconciled_packets': reconciled_packets,
        'summary': summary,
        'elapsed_seconds': round(elapsed, 2),
    }


if __name__ == '__main__':
    import sys as _main_sys
    if len(_main_sys.argv) < 2:
        print("Usage: python step09_shipping_reconciliation.py <step08_result.json> [step07_result.json]")
        _main_sys.exit(1)
    with open(_main_sys.argv[1], 'r', encoding='utf-8') as f:
        step8 = json.load(f)
    step7 = None
    if len(_main_sys.argv) >= 3:
        with open(_main_sys.argv[2], 'r', encoding='utf-8') as f:
            step7 = json.load(f)
    result = run(step8, step7, output_dir=os.path.dirname(_main_sys.argv[1]))
    print("\nResult: %s" % result['summary'])
