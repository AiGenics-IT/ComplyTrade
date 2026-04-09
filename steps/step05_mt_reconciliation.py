"""
Step 5 -- MT OCR Reconciliation
=================================
Uses Qwen VLM as a "second pair of eyes" to verify and supplement the
GLM OCR text of MT-side documents (Letters of Credit, amendments, SWIFT messages).

KEY PRINCIPLE:
    GLM OCR (Step 1) already extracted the raw text -- this is the TRUSTED source.
    Qwen gets BOTH the page image AND the GLM text. It does NOT re-extract text.
    It only ADDS missing things (F-tags, values) -- never changes existing text.

TEXT PROVENANCE CHAIN:
    After this step, each page has three text layers:
    1. raw_text     -- GLM OCR output (Step 1, never modified)
    2. cleaned_text -- deterministic cleaning (Step 2)
    3. refined_text -- GLM text + Qwen additions (this step)
    Downstream steps use refined_text when available.

SELECTIVE PROCESSING:
    - MT-side packets (MT700, MT707, MT799, unknown, covering_letter): sent to VLM
    - Shipping packets with confidence >= 0.80: SKIP VLM (passthrough)
      They get their own VLM reconciliation in Step 9.

INPUT:  Step 4 output -- list of ClassifiedPacket with mt_type
OUTPUT: List of ReconciledPacket with refined_text, change_log

MODEL:  Qwen VLM at QWEN_VLM_URL (7B or 72B per VLM_MODEL_SIZE switch in
        config/settings.py)
"""

import os
import sys as _sys; _sys.stdout.reconfigure(encoding="utf-8", errors="replace") if hasattr(_sys.stdout, "reconfigure") else None
import re
import json
import time
import base64
import requests
from dataclasses import dataclass, field, asdict
from typing import List, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import QWEN_VLM_URL, QWEN_VLM_MODEL, VLM_TIMEOUT, MAX_CONCURRENT_VLM


# == Dataclasses ==============================================================

@dataclass
class ChangeLogEntry:
    """A single change made during reconciliation -- part of audit trail."""
    field: str               # "text" or "classification"
    original: str
    refined: str
    reason: str
    confidence: float = 0.0


@dataclass
class ReconciledPage:
    """A page with all three text layers -- full provenance chain."""
    page_number: int
    raw_text: str              # GLM OCR output (Step 1, never modified)
    cleaned_text: str          # After deterministic cleaning (Step 2)
    refined_text: str          # GLM text + Qwen additions (this step)
    page_image_path: str
    text_changed: bool = False
    change_summary: str = ""


@dataclass
class ReconciledPacket:
    """Packet after OCR reconciliation and potential reclassification."""
    packet_id: int
    pages: List[ReconciledPage] = field(default_factory=list)
    page_numbers: List[int] = field(default_factory=list)
    page_count: int = 0
    boundary_confidence: float = 0.0
    # Original classification (from Step 4)
    original_mt_type: str = ""
    original_mt_confidence: float = 0.0
    # Refined classification
    mt_type: str = ""
    mt_confidence: float = 0.0
    swift_format: str = ""
    doc_hint: str = ""
    # Reconciliation metadata
    reconciliation_method: str = ""    # "vlm" | "passthrough"
    change_log: List[ChangeLogEntry] = field(default_factory=list)
    was_reclassified: bool = False
    text_was_corrected: bool = False
    # MT799 amendment promotion (carried through from Step 4)
    source_mt: str = ""
    is_799_amendment: bool = False


# == VLM reconciliation prompt ================================================
# The prompt explicitly tells VLM to ONLY ADD missing text, never change existing.

_VLM_RECONCILE_PROMPT = """You are verifying the GLM OCR extraction of a trade finance SWIFT message page.

Current classification: {mt_type} (confidence: {confidence:.2f})
SWIFT format: {swift_format}

CRITICAL RULE: The GLM OCR text below is the TRUSTED text source. Do NOT change any existing text.
You may ONLY ADD text that is visible in the image but MISSING from the OCR text.

GLM OCR Text:
---
{ocr_text}
---

Tasks:
1. REVIEW the GLM text against the page image. Look for:
   - Missing SWIFT F-tags (F20:, F31C:, F46A: or :20:, :31C:, :46A:)
   - Missing field values after tags
   - Missing stamps, endorsements, or annotations visible in the image
   Report ONLY what is MISSING -- do NOT rewrite or change existing text.

2. CLASSIFICATION CHECK: Is "{mt_type}" correct?
   - MT700 = Original Letter of Credit
   - MT707 = Amendment to LC (contains F26E/26E amendment number)
   - MT799 = Free format SWIFT message
   - shipping = Commercial/shipping document
   - covering_letter = Documentary Remittance / Covering Letter

3. VISUAL ELEMENTS: Note any stamps, signatures, seals, or logos visible in the image.

Respond in EXACTLY this JSON format:
{{
  "text_complete": true/false,
  "missing_text": "ONLY the text that is missing from OCR (leave empty if nothing missing)",
  "missing_fields": ["list of missing F-tags or field names"],
  "stamps_detected": ["list of stamp text: ORIGINAL, COPY, etc."],
  "signatures_detected": true/false,
  "classification_correct": true/false,
  "correct_mt_type": "{mt_type}",
  "confidence": 0.0-1.0,
  "reason": "brief explanation"
}}"""


def _vlm_reconcile_page(
    image_path: str,
    ocr_text: str,
    mt_type: str,
    mt_confidence: float,
    swift_format: str,
) -> dict:
    """
    Send page image + GLM text to Qwen VLM for additive-only reconciliation.
    VLM reviews the image against GLM text and reports ONLY missing content.
    """
    prompt = _VLM_RECONCILE_PROMPT.format(
        mt_type=mt_type,
        confidence=mt_confidence,
        swift_format=swift_format or 'unknown',
        ocr_text=ocr_text,  # send more text for better review
    )

    messages = [{"role": "user", "content": []}]

    # Add image
    if image_path and os.path.exists(image_path):
        try:
            with open(image_path, 'rb') as f:
                img_b64 = base64.b64encode(f.read()).decode('utf-8')
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_b64}"},
            })
        except Exception:
            pass  # proceed without image

    messages[0]["content"].append({"type": "text", "text": prompt})

    try:
        resp = requests.post(
            QWEN_VLM_URL,
            json={
                "model": QWEN_VLM_MODEL,
                "messages": messages,
                "max_tokens": 2000,
                "temperature": 0.1,
            },
            timeout=VLM_TIMEOUT,
        )
        if resp.status_code != 200:
            return {'error': f'HTTP {resp.status_code}'}

        content = resp.json()['choices'][0]['message']['content']

        # Extract JSON -- handle nested objects
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Try markdown code block
        code_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
        if code_match:
            try:
                return json.loads(code_match.group(1))
            except json.JSONDecodeError:
                pass

        return {'error': 'no valid JSON', 'raw': content}

    except requests.exceptions.Timeout:
        return {'error': 'vlm_timeout'}
    except Exception as e:
        return {'error': str(e)}


def _merge_missing_text(cleaned_text: str, missing_text: str, missing_fields: list) -> str:
    """
    Append VLM-identified missing text to the GLM cleaned text.

    ADDITIVE ONLY -- never changes existing text. Appends missing content
    in a clearly marked section so the audit trail is clear.
    """
    if not missing_text or len(missing_text.strip()) < 10:
        return cleaned_text

    missing_text = missing_text.strip()

    # Build the addition section
    additions = []
    if missing_fields:
        additions.append(f"Missing fields: {', '.join(missing_fields[:10])}")
    additions.append(missing_text)

    return cleaned_text + "\n\n[VLM ADDITIONS]\n" + "\n".join(additions)


# == Main run function ========================================================

def run(step4_result: dict, output_dir: str = None, progress_callback=None) -> dict:
    """
    Execute Step 5: MT OCR Reconciliation.

    For MT packets: sends page image + GLM text to Qwen.
    For shipping packets: passthrough (GLM text is sufficient).

    Args:
        step4_result: Output from Step 4 (with 'packets' list of ClassifiedPacket)
        output_dir: Directory to save results
        progress_callback: Optional callback(message: str)

    Returns:
        dict with 'packets', 'reclassified', 'text_corrected', 'elapsed_seconds'
    """
    def _progress(msg):
        if progress_callback:
            progress_callback(f"[Step 5] {msg}")
        print(f"[Step 5] {msg}")

    start_time = time.time()
    packets_in = step4_result.get('packets', [])
    _progress(f"Reconciling {len(packets_in)} packets...")

    reconciled: List[ReconciledPacket] = []
    total_reclassified = 0
    total_text_corrected = 0
    vlm_calls = 0

    # Separate packets into VLM-needed and passthrough
    vlm_tasks = []   # (packet_index, page_index, image_path, ocr_text, mt_type, mt_conf, swift_fmt)
    packet_data = []  # preprocessed packet info

    for pkt_idx, pkt_in in enumerate(packets_in):
        if hasattr(pkt_in, 'packet_id'):
            pkt_id = pkt_in.packet_id
            pages_data = pkt_in.pages
            page_numbers = pkt_in.page_numbers
            page_count = pkt_in.page_count
            b_conf = pkt_in.boundary_confidence
            orig_mt = pkt_in.mt_type
            orig_conf = pkt_in.mt_confidence
            swift_fmt = pkt_in.swift_format
            doc_hint = pkt_in.doc_hint
            src_mt = getattr(pkt_in, 'source_mt', '')
            is_799_amd = bool(getattr(pkt_in, 'is_799_amendment', False))
        else:
            pkt_id = pkt_in.get('packet_id', 0)
            pages_data = pkt_in.get('pages', [])
            page_numbers = pkt_in.get('page_numbers', [])
            page_count = pkt_in.get('page_count', 0)
            b_conf = pkt_in.get('boundary_confidence', 0)
            orig_mt = pkt_in.get('mt_type', '')
            orig_conf = pkt_in.get('mt_confidence', 0)
            swift_fmt = pkt_in.get('swift_format', '')
            doc_hint = pkt_in.get('doc_hint', '')
            src_mt = pkt_in.get('source_mt', '')
            is_799_amd = bool(pkt_in.get('is_799_amendment', False))

        # Decide if VLM is needed:
        # MT-side, unknown, covering_letter, or low-confidence shipping
        needs_vlm = (
            orig_mt.startswith('MT')
            or orig_mt in ('unknown', 'covering_letter')
            or orig_conf < 0.80
        )

        page_infos = []
        for p in pages_data:
            if hasattr(p, 'page_number'):
                pg_num = p.page_number
                raw = p.raw_text
                cleaned = p.cleaned_text
                img_path = p.page_image_path
            else:
                pg_num = p.get('page_number', 0)
                raw = p.get('raw_text', '')
                cleaned = p.get('cleaned_text', '')
                img_path = p.get('page_image_path', '')

            page_infos.append({
                'page_number': pg_num,
                'raw_text': raw,
                'cleaned_text': cleaned,
                'page_image_path': img_path,
            })

            if needs_vlm:
                vlm_tasks.append((pkt_idx, len(page_infos) - 1, img_path, cleaned, orig_mt, orig_conf, swift_fmt))

        packet_data.append({
            'pkt_id': pkt_id,
            'page_infos': page_infos,
            'page_numbers': page_numbers if page_numbers else [pi['page_number'] for pi in page_infos],
            'page_count': page_count if page_count else len(page_infos),
            'b_conf': b_conf,
            'orig_mt': orig_mt,
            'orig_conf': orig_conf,
            'swift_fmt': swift_fmt,
            'doc_hint': doc_hint,
            'src_mt': src_mt,
            'is_799_amd': is_799_amd,
            'needs_vlm': needs_vlm,
            'vlm_results': {},  # page_index -> vlm_result
        })

    # -- Execute VLM calls concurrently --
    _progress(f"  {len(vlm_tasks)} pages need VLM reconciliation...")

    if vlm_tasks:
        def _do_vlm(task):
            pkt_idx, pg_idx, img_path, ocr_text, mt_type, mt_conf, swift_fmt = task
            result = _vlm_reconcile_page(img_path, ocr_text, mt_type, mt_conf, swift_fmt)
            return (pkt_idx, pg_idx, result)

        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_VLM) as executor:
            futures = {executor.submit(_do_vlm, task): task for task in vlm_tasks}
            for future in as_completed(futures):
                try:
                    pkt_idx, pg_idx, vlm_result = future.result()
                    vlm_calls += 1
                    packet_data[pkt_idx]['vlm_results'][pg_idx] = vlm_result

                    pd = packet_data[pkt_idx]
                    pg_num = pd['page_infos'][pg_idx]['page_number']
                    if 'error' in vlm_result:
                        _progress(f"    VLM error packet {pd['pkt_id']} page {pg_num}: {vlm_result['error']}")
                    else:
                        if not vlm_result.get('text_complete', True):
                            _progress(f"    VLM packet {pd['pkt_id']} page {pg_num}: missing text found")
                        else:
                            _progress(f"    VLM packet {pd['pkt_id']} page {pg_num}: text complete")
                except Exception as e:
                    _progress(f"    VLM future error: {e}")

    # -- Build reconciled packets --
    for pd in packet_data:
        change_log: List[ChangeLogEntry] = []
        recon_pages: List[ReconciledPage] = []
        packet_reclassified = False
        packet_text_corrected = False
        refined_mt = pd['orig_mt']
        refined_conf = pd['orig_conf']

        for pg_idx, pi in enumerate(pd['page_infos']):
            refined = pi['cleaned_text']  # default: no change
            text_changed = False
            change_summary = ""

            vlm_result = pd['vlm_results'].get(pg_idx)
            if vlm_result and 'error' not in vlm_result:
                # Text additions (additive only)
                if not vlm_result.get('text_complete', True):
                    missing_text = vlm_result.get('missing_text', '')
                    missing_fields = vlm_result.get('missing_fields', [])
                    if missing_text and len(missing_text.strip()) >= 10:
                        refined = _merge_missing_text(pi['cleaned_text'], missing_text, missing_fields)
                        text_changed = True
                        packet_text_corrected = True
                        change_summary = f"Added {len(missing_fields)} missing fields"
                        change_log.append(ChangeLogEntry(
                            field='text',
                            original=f"page {pi['page_number']}: {len(pi['cleaned_text'])} chars",
                            refined=f"page {pi['page_number']}: {len(refined)} chars, +{len(missing_fields)} fields",
                            reason=f"Missing: {', '.join(missing_fields[:5])}",
                            confidence=float(vlm_result.get('confidence', 0.7)),
                        ))

                # Stamps detected by VLM
                stamps = vlm_result.get('stamps_detected', [])
                if stamps:
                    stamp_text = ', '.join(stamps)
                    if stamp_text not in refined:
                        refined = refined + f"\n[VLM STAMPS: {stamp_text}]"

                # Classification check (first page only for packet-level)
                if pg_idx == 0 and not vlm_result.get('classification_correct', True):
                    new_mt = vlm_result.get('correct_mt_type', pd['orig_mt'])
                    new_conf = float(vlm_result.get('confidence', pd['orig_conf']))
                    # Don't let VLM downgrade an MT799-promoted-to-MT707 back to MT799
                    # — Step 4 already established it carries amendment instructions.
                    if pd.get('is_799_amd') and new_mt in ('MT799', 'MT999'):
                        new_mt = pd['orig_mt']
                    # Don't let VLM upgrade an MT799 to MT700/MT707/MT710/MT720
                    # just because the body references field tags. Step 4 has
                    # multiple safety nets to detect free-format MT799 — if it
                    # decided MT799, the VLM should not override that without
                    # an extremely high confidence (>= 0.90). MT799 is the
                    # most-misclassified type because its body LOOKS like an
                    # LC by referencing fields F45A/F46A/F47A.
                    if pd['orig_mt'] in ('MT799', 'MT999') and new_mt in (
                        'MT700', 'MT701', 'MT705', 'MT707', 'MT708',
                        'MT710', 'MT711', 'MT720', 'MT721',
                    ):
                        if new_conf < 0.90:
                            new_mt = pd['orig_mt']
                    if new_mt != pd['orig_mt'] and new_conf > 0.60:
                        refined_mt = new_mt
                        refined_conf = new_conf
                        packet_reclassified = True
                        change_log.append(ChangeLogEntry(
                            field='classification',
                            original=f"{pd['orig_mt']} ({pd['orig_conf']:.2f})",
                            refined=f"{refined_mt} ({refined_conf:.2f})",
                            reason=vlm_result.get('reason', 'vlm reclassification'),
                            confidence=new_conf,
                        ))

            recon_pages.append(ReconciledPage(
                page_number=pi['page_number'],
                raw_text=pi['raw_text'],
                cleaned_text=pi['cleaned_text'],
                refined_text=refined,
                page_image_path=pi['page_image_path'],
                text_changed=text_changed,
                change_summary=change_summary,
            ))

        if packet_reclassified:
            total_reclassified += 1
        if packet_text_corrected:
            total_text_corrected += 1

        rpkt = ReconciledPacket(
            packet_id=pd['pkt_id'],
            pages=recon_pages,
            page_numbers=pd['page_numbers'],
            page_count=pd['page_count'],
            boundary_confidence=pd['b_conf'],
            original_mt_type=pd['orig_mt'],
            original_mt_confidence=pd['orig_conf'],
            mt_type=refined_mt,
            mt_confidence=refined_conf,
            swift_format=pd['swift_fmt'],
            doc_hint=pd['doc_hint'],
            reconciliation_method='vlm' if pd['needs_vlm'] else 'passthrough',
            change_log=change_log,
            was_reclassified=packet_reclassified,
            text_was_corrected=packet_text_corrected,
            source_mt=pd.get('src_mt', ''),
            is_799_amendment=pd.get('is_799_amd', False),
        )
        reconciled.append(rpkt)

        status = []
        if packet_reclassified:
            status.append(f"RECLASSIFIED {pd['orig_mt']}->{refined_mt}")
        if packet_text_corrected:
            status.append("TEXT ADDED")
        if not status:
            status.append("OK" if pd['needs_vlm'] else "PASSTHROUGH")
        _progress(f"  Packet {pd['pkt_id']}: {refined_mt} -- {', '.join(status)}")

    elapsed = time.time() - start_time
    _progress(f"Step 5 complete: {total_reclassified} reclassified, {total_text_corrected} text-added, "
              f"{vlm_calls} VLM calls in {elapsed:.1f}s")

    # Save results
    result_file = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        result_file = os.path.join(output_dir, 'step05_result.json')
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({
                'step': 5,
                'step_name': 'MT OCR Reconciliation',
                'total_packets': len(reconciled),
                'reclassified': total_reclassified,
                'text_corrected': total_text_corrected,
                'vlm_calls': vlm_calls,
                'elapsed_seconds': round(elapsed, 2),
                'packets': [asdict(p) for p in reconciled],
            }, f, indent=2, ensure_ascii=False)

    return {
        'packets': reconciled,
        'reclassified': total_reclassified,
        'text_corrected': total_text_corrected,
        'elapsed_seconds': round(elapsed, 2),
        'errors': [],
        'result_file': result_file,
    }


if __name__ == '__main__':
    import sys as _sys
    if len(_sys.argv) < 2:
        print("Usage: python step05_mt_reconciliation.py <step04_result.json>")
        _sys.exit(1)
    with open(_sys.argv[1], 'r', encoding='utf-8') as f:
        step4 = json.load(f)
    result = run(step4, output_dir=os.path.dirname(_sys.argv[1]))
    print(f"\nResult: {result['reclassified']} reclassified, {result['text_corrected']} corrected, {result['elapsed_seconds']}s")
