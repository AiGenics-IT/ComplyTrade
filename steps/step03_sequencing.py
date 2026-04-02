"""
Step 3 -- Page Sequencing and Document Packet Formation
========================================================
Groups consecutive PDF pages into logical "document packets". A single PDF
contains multiple trade-finance documents (LC pages, Bills of Lading,
Invoices, Certificates, etc.) concatenated with no explicit separators.

HOW IT WORKS:
    Phase 1 -- Qwen classifies EVERY page:
        Sends page image + GLM text to Qwen VLM for each page.
        Qwen returns: document_type, is_continuation, confidence, stamps, signatures.
        This is the PRIMARY classification — Qwen sees the actual image.

    Phase 2 -- Group pages into packets:
        Pages with same document_type in sequence are grouped.
        Continuation pages merge into the previous packet.
        Copy detection: if same doc_type appears again after a different doc, it's a new copy.

    Phase 3 -- Context re-check (optional):
        For low-confidence pages, re-send with context of surrounding pages.

WHY QWEN FOR EVERY PAGE:
    - Text-based boundary detection is unreliable (many docs lack clear headers)
    - Copies of same document have identical text — only visual differences (stamps, markings)
    - Endorsement pages, blank backs, stamp-only pages need visual understanding
    - The old system (posss3.py) used VLM for every page and it worked well

INPUT:  Step 2 output -- list of PageCleaned (cleaned_text + raw_text + page_image_path)
OUTPUT: List of DocumentPacket objects with pages[], doc_type, boundary_confidence

MODEL:  Qwen VLM @ http://10.20.10.2:8085 (classifies every page)
        GLM text included in every prompt (Qwen reviews, never rewrites)
"""

import os
import sys as _sys; _sys.stdout.reconfigure(encoding="utf-8", errors="replace") if hasattr(_sys.stdout, "reconfigure") else None
import re
import json
import time
import base64
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import QWEN_VLM_URL, QWEN_VLM_MODEL, MAX_CONCURRENT_VLM, VLM_TIMEOUT


# ── Data Models ──

@dataclass
class PageClassification:
    """Qwen's classification of a single page."""
    page_number: int
    document_type: str = "unknown"        # Bill of Lading, Commercial Invoice, LC, etc.
    is_continuation: bool = False          # True = continuation of previous document
    confidence: float = 0.0
    stamps: List[dict] = field(default_factory=list)
    signatures: List[dict] = field(default_factory=list)
    seals: List[dict] = field(default_factory=list)
    logos: List[dict] = field(default_factory=list)
    copy_status: str = "unknown"           # original, copy, non_negotiable
    copy_label: str = ""                   # ORIGINAL, COPY, NON-NEGOTIABLE, FIRST ORIGINAL
    marking_status: str = "unknown"        # stamped_and_signed, signed, stamped, unsigned
    doc_hint: str = ""                     # Additional context from Qwen
    raw_text: str = ""
    cleaned_text: str = ""
    page_image_path: str = ""


@dataclass
class DocumentPacket:
    """A group of pages forming one logical document."""
    packet_id: str = ""
    document_type: str = "unknown"
    pages: List[dict] = field(default_factory=list)
    page_numbers: List[int] = field(default_factory=list)
    boundary_confidence: float = 0.0
    copy_status: str = "original"
    copy_label: str = ""
    marking_status: str = "unsigned"
    stamps: List[dict] = field(default_factory=list)
    signatures: List[dict] = field(default_factory=list)
    seals: List[dict] = field(default_factory=list)
    logos: List[dict] = field(default_factory=list)
    doc_hint: str = ""


# ── VLM Classification Prompt ──

CLASSIFY_PROMPT = """You are a trade finance document classifier. Look at this page image and the OCR text below.

GLM OCR TEXT (trusted — extracted from this page):
{glm_text}

CLASSIFY this page. Return ONLY valid JSON:
{{
    "document_type": "exact type from this list: LC, Amendment, Bill of Lading, Commercial Invoice, Draft Bill of Exchange, Packing List, Certificate of Origin, Insurance Certificate, Weight Certificate, Quality Certificate, Quantity Certificate, Shipment Advice, Document Remittance, Agents Certificate, Shipping Company Certificate, Beneficiary Certificate, Fumigation Certificate, Phytosanitary Certificate, Inspection Certificate, Accreditation Certificate, Pre-Shipment Inspection Report, Intertek Report, Letter of Clarification, Endorsement Page, Blank Page, or describe if none match",
    "is_continuation": false,
    "confidence": 0.95,
    "stamps": [{{"text": "stamp text if readable", "type": "rubber_stamp/embossed/printed", "position": "top-right"}}],
    "signatures": [{{"description": "handwritten signature", "type": "handwritten/digital", "signatory": "name if readable"}}],
    "seals": [{{"description": "round company seal"}}],
    "logos": [{{"company_name": "company name", "position": "top-left"}}],
    "copy_status": "original or copy or non_negotiable",
    "copy_label": "exact text of marking: ORIGINAL, COPY, NON-NEGOTIABLE, FIRST ORIGINAL, etc.",
    "marking_status": "stamped_and_signed or signed or stamped or unsigned",
    "doc_hint": "brief 1-line description of what this page contains"
}}

RULES:
- If page has SWIFT F-tags (F20:, F31C:, F42A:, F46A:, F47A:) -> type is "LC" or "Amendment"
- If page says "Page 2 of 3" or continues from previous without its own header -> is_continuation = true
- Look for ORIGINAL/COPY/NON-NEGOTIABLE stamps in the image corners
- Detect handwritten signatures, rubber stamps, embossed seals, company logos
- If page is mostly blank or just has stamps/endorsements -> "Endorsement Page" or "Blank Page"
"""


def _classify_page_vlm(page_num: int, image_path: str, glm_text: str) -> dict:
    """Send one page to Qwen for classification."""
    try:
        if not os.path.exists(image_path):
            return {'page_number': page_num, 'document_type': 'unknown', 'confidence': 0.0,
                    'error': 'Image not found'}

        img_b64 = base64.b64encode(open(image_path, 'rb').read()).decode()
        prompt = CLASSIFY_PROMPT.format(glm_text=glm_text)

        resp = requests.post(QWEN_VLM_URL, json={
            "model": QWEN_VLM_MODEL,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                {"type": "text", "text": prompt}
            ]}],
            "max_tokens": 1000, "temperature": 0.1
        }, timeout=VLM_TIMEOUT)

        if resp.status_code != 200:
            return {'page_number': page_num, 'document_type': 'unknown', 'confidence': 0.0,
                    'error': f'VLM HTTP {resp.status_code}'}

        result = resp.json()
        content = result.get('choices', [{}])[0].get('message', {}).get('content', '')

        # Parse JSON from response
        # Strip markdown fences if present
        content = content.strip()
        if content.startswith('```'):
            content = content.split('\n', 1)[1] if '\n' in content else content[3:]
        if content.endswith('```'):
            content = content[:-3]
        content = content.strip()

        # Find JSON in response
        json_start = content.find('{')
        json_end = content.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            parsed = json.loads(content[json_start:json_end])
            parsed['page_number'] = page_num
            return parsed
        else:
            return {'page_number': page_num, 'document_type': 'unknown', 'confidence': 0.3,
                    'doc_hint': content[:500]}

    except json.JSONDecodeError:
        return {'page_number': page_num, 'document_type': 'unknown', 'confidence': 0.3,
                'doc_hint': content[:500] if 'content' in dir() else ''}
    except requests.exceptions.Timeout:
        return {'page_number': page_num, 'document_type': 'unknown', 'confidence': 0.0,
                'error': 'VLM timeout'}
    except Exception as e:
        return {'page_number': page_num, 'document_type': 'unknown', 'confidence': 0.0,
                'error': str(e)}


def _group_into_packets(classifications: List[dict]) -> List[DocumentPacket]:
    """Group classified pages into document packets."""
    if not classifications:
        return []

    packets = []
    current_packet = None

    for cls in sorted(classifications, key=lambda c: c.get('page_number', 0)):
        pg_num = cls.get('page_number', 0)
        doc_type = cls.get('document_type', 'unknown')
        is_cont = cls.get('is_continuation', False)
        confidence = cls.get('confidence', 0.0)

        # Normalize doc type
        doc_type_lower = doc_type.lower().strip()
        if doc_type_lower in ('lc', 'letter of credit', 'swift message', 'mt700'):
            doc_type = 'LC'
        elif doc_type_lower in ('amendment', 'mt707', 'lc amendment'):
            doc_type = 'Amendment'

        if is_cont and current_packet:
            # Continuation — merge into current packet
            current_packet.page_numbers.append(pg_num)
            current_packet.pages.append(cls)
            # Merge stamps/signatures
            current_packet.stamps.extend(cls.get('stamps', []))
            current_packet.signatures.extend(cls.get('signatures', []))
            current_packet.seals.extend(cls.get('seals', []))
            current_packet.logos.extend(cls.get('logos', []))
        else:
            # New document — start new packet
            if current_packet:
                packets.append(current_packet)

            pkt_id = f"pkt_{len(packets)+1}"
            copy_status = cls.get('copy_status', 'unknown')
            if copy_status == 'unknown':
                copy_status = 'original'

            current_packet = DocumentPacket(
                packet_id=pkt_id,
                document_type=doc_type,
                pages=[cls],
                page_numbers=[pg_num],
                boundary_confidence=confidence,
                copy_status=copy_status,
                copy_label=cls.get('copy_label', ''),
                marking_status=cls.get('marking_status', 'unsigned'),
                stamps=cls.get('stamps', []) if isinstance(cls.get('stamps'), list) else [],
                signatures=cls.get('signatures', []) if isinstance(cls.get('signatures'), list) else [],
                seals=cls.get('seals', []) if isinstance(cls.get('seals'), list) else [],
                logos=cls.get('logos', []) if isinstance(cls.get('logos'), list) else [],
                doc_hint=cls.get('doc_hint', ''),
            )

    # Don't forget the last packet
    if current_packet:
        packets.append(current_packet)

    return packets


def run(step2_result: dict, output_dir: str = None, progress_callback=None) -> dict:
    """
    Execute Step 3: Classify every page with Qwen, then group into packets.

    Args:
        step2_result: Output from Step 2 (pages with cleaned_text + page_image_path)
        output_dir: Directory to save results
        progress_callback: Optional callback for progress

    Returns:
        dict with 'packets' (list of DocumentPacket), 'classifications', 'elapsed_seconds'
    """
    def _progress(msg):
        if progress_callback:
            progress_callback(f"[Step 3] {msg}")
        print(f"[Step 3] {msg}")

    start_time = time.time()
    pages = step2_result.get('pages', [])
    _progress(f"Classifying {len(pages)} pages with Qwen VLM...")

    # ── Phase 1: Classify every page with Qwen ──
    classifications = []

    # Build tasks for concurrent VLM calls
    tasks = []
    for page in pages:
        if hasattr(page, 'page_number'):
            pg_num = page.page_number
            text = page.cleaned_text or page.raw_text
            img_path = page.page_image_path
        else:
            pg_num = page.get('page_number', 0)
            text = page.get('cleaned_text', page.get('raw_text', ''))
            img_path = page.get('page_image_path', '')

        tasks.append((pg_num, img_path, text))

    # Run VLM classification concurrently
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_VLM) as executor:
        futures = {}
        for pg_num, img_path, text in tasks:
            future = executor.submit(_classify_page_vlm, pg_num, img_path, text)
            futures[future] = pg_num

        done_count = 0
        for future in as_completed(futures):
            pg_num = futures[future]
            try:
                result = future.result()
                classifications.append(result)
                doc_type = result.get('document_type', '?')
                conf = result.get('confidence', 0)
                is_cont = result.get('is_continuation', False)
                stamps_count = len(result.get('stamps', []))
                sigs_count = len(result.get('signatures', []))
                copy_st = result.get('copy_status', '?')

                done_count += 1
                _progress(f"  Page {pg_num}: {doc_type} (conf={conf:.2f}, cont={is_cont}, "
                          f"stamps={stamps_count}, sigs={sigs_count}, copy={copy_st}) "
                          f"[{done_count}/{len(tasks)}]")

            except Exception as e:
                _progress(f"  Page {pg_num}: ERROR - {e}")
                classifications.append({
                    'page_number': pg_num, 'document_type': 'unknown',
                    'confidence': 0.0, 'error': str(e)
                })

    # Sort by page number
    classifications.sort(key=lambda c: c.get('page_number', 0))

    # ── Phase 2: Group pages into document packets ──
    _progress("Grouping pages into document packets...")
    packets = _group_into_packets(classifications)

    # Add text data to packets
    page_text_map = {}
    for page in pages:
        if hasattr(page, 'page_number'):
            page_text_map[page.page_number] = {
                'raw_text': page.raw_text,
                'cleaned_text': page.cleaned_text,
                'page_image_path': page.page_image_path,
            }
        else:
            page_text_map[page.get('page_number', 0)] = {
                'raw_text': page.get('raw_text', ''),
                'cleaned_text': page.get('cleaned_text', ''),
                'page_image_path': page.get('page_image_path', ''),
            }

    for pkt in packets:
        # Concatenate text from all pages in packet
        all_text = []
        for pg_num in pkt.page_numbers:
            pg_data = page_text_map.get(pg_num, {})
            all_text.append(pg_data.get('cleaned_text', pg_data.get('raw_text', '')))
        pkt.doc_hint = pkt.doc_hint or pkt.document_type

    elapsed = time.time() - start_time

    _progress(f"Step 3 complete: {len(packets)} packets from {len(pages)} pages in {elapsed:.1f}s")
    for pkt in packets:
        _progress(f"  {pkt.packet_id}: {pkt.document_type} (pages {pkt.page_numbers}, "
                  f"copy={pkt.copy_status}, stamps={len(pkt.stamps)}, sigs={len(pkt.signatures)})")

    # ── Save results ──
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        result_file = os.path.join(output_dir, 'step03_result.json')
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({
                'step': 3,
                'step_name': 'Page Sequencing and Document Packet Formation',
                'total_pages': len(pages),
                'total_packets': len(packets),
                'classifications': classifications,
                'packets': [asdict(p) for p in packets],
                'elapsed_seconds': round(elapsed, 2),
            }, f, indent=2, ensure_ascii=False)

    return {
        'packets': [asdict(p) for p in packets],
        'classifications': classifications,
        'total_pages': len(pages),
        'elapsed_seconds': round(elapsed, 2),
    }


if __name__ == '__main__':
    import sys as _sys2
    if len(_sys2.argv) < 2:
        print("Usage: python step03_sequencing.py <step02_result.json>")
        _sys2.exit(1)
    with open(_sys2.argv[1], 'r', encoding='utf-8') as f:
        s2 = json.load(f)
    result = run(s2, 'test_step03')
    print(f"Result: {result['total_pages']} pages -> {len(result['packets'])} packets")
