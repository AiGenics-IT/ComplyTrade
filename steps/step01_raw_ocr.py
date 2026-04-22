"""
Step 1 — Page-Level Raw OCR Extraction
========================================
This is the very first step in the ComplyTrade document processing pipeline.
It takes a PDF file (which could be a scanned trade finance document bundle
containing Letters of Credit, Bills of Lading, Invoices, Certificates, etc.)
and extracts raw text from every page using an AI-powered OCR model.

PURPOSE:
    Convert scanned/digital PDF pages into machine-readable text so that
    downstream steps can classify, parse, and verify the documents.

INPUT:
    - A single PDF file path (the uploaded trade finance document bundle).
      In trade finance, banks receive bundles of documents — the LC (Letter of
      Credit) pages, shipping documents, certificates — all in one PDF.

OUTPUT:
    - A list of PageOCR objects, one per page, each containing:
        * raw_text: the OCR-extracted text (no cleaning applied yet)
        * page_number: 1-based page index
        * confidence: OCR engine's confidence score (0.0 to 1.0)
        * flags for stamps, signatures, images, tables, handwriting
        * page_image_path: saved PNG of the rendered page (for VLM steps later)

MODEL:
    GLM-OCR @ http://10.20.10.2:8001/api/ocr
    A specialized OCR model that handles scanned documents, stamps, signatures,
    and mixed-layout pages common in trade finance paperwork.
    Step 1 ALWAYS uses GLM regardless of the VLM_MODEL_SIZE switch in
    config/settings.py — that switch only affects Qwen VLM steps (2–14).

STRATEGY:
    1. Split the PDF into N single-page PDFs in memory (preserves text layers).
    2. Submit every page in parallel to GLM (ThreadPoolExecutor, MAX_CONCURRENT_OCR).
    3. Each page gets up to 3 attempts with escalating per-request timeouts
       [60s, 180s, 300s] — a hung GLM request cannot block the job forever.
    4. Each page is also rendered to a 300 DPI PNG image for use by the
       Vision Language Model (VLM) in later steps (Steps 3, 5, 8, 9).
"""

import os
# Ensure UTF-8 output on Windows to handle international characters in trade docs
# (e.g., Arabic names, Chinese port names, accented European city names)
import sys as _sys; _sys.stdout.reconfigure(encoding="utf-8", errors="replace") if hasattr(_sys.stdout, "reconfigure") else None
import io
import json
import time
import requests
import fitz  # PyMuPDF — used to render PDF pages into PNG images for OCR and VLM
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from typing import List, Optional
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
# Configuration constants: OCR endpoint URL, model name, prompt template,
# concurrency limits, and timeout settings — all centralized in config/settings.py
from config.settings import GLM_OCR_URL, GLM_OCR_MODEL, GLM_OCR_PROMPT, MAX_CONCURRENT_OCR, OCR_TIMEOUT


@dataclass
class PageOCR:
    """
    Output of Step 1 for a single page.

    Holds the raw OCR text plus metadata flags that help downstream steps.
    For example, has_stamps=True tells the verification engine to check for
    ORIGINAL/COPY markings (important in trade finance where banks require
    a specific number of original documents).
    """
    page_number: int               # 1-based page number within the PDF
    raw_text: str                  # Unmodified OCR output text
    char_count: int = 0            # Length of raw_text (auto-calculated)
    confidence: float = 0.0        # OCR engine's confidence (0.0-1.0)
    ocr_method: str = "glm-ocr"   # Which OCR engine was used
    # --- Visual element flags ---
    # These flags are auto-detected from OCR output markers.
    # Trade finance documents commonly have stamps (ORIGINAL, COPY),
    # signatures (for endorsements), and tables (for goods descriptions).
    has_stamps: bool = False       # ORIGINAL/COPY/DUPLICATE stamps detected
    has_signatures: bool = False   # Signature blocks detected
    has_images: bool = False       # Logos or embedded images detected
    has_tables: bool = False       # Tabular data detected (common in invoices, packing lists)
    has_handwritten: bool = False  # Handwritten annotations detected
    has_unreadable: bool = False   # Parts marked as unclear/unreadable by OCR
    ambiguity_notes: List[str] = field(default_factory=list)  # Any OCR ambiguity notes
    page_image_path: str = ""      # Path to saved PNG image of this page
    elapsed_seconds: float = 0.0   # Time taken to OCR this page

    def __post_init__(self):
        """Auto-detect visual element flags from OCR marker tokens in the text."""
        self.char_count = len(self.raw_text)
        # The GLM-OCR model inserts bracketed markers like [STAMP:...], [SIGNATURE],
        # [IMAGE:...] when it detects these visual elements on the page
        t = self.raw_text
        # Stamps: look for GLM markers OR common trade finance stamp text
        # (ORIGINAL, COPY, DUPLICATE, NON-NEGOTIABLE are critical in LC compliance)
        self.has_stamps = '[STAMP:' in t or any(kw in t.upper() for kw in ['ORIGINAL', 'COPY', 'DUPLICATE', 'NON-NEGOTIABLE'])
        self.has_signatures = '[SIGNATURE' in t
        self.has_images = '[IMAGE:' in t or '[LOGO:' in t
        # Tables detected by pipe characters (common in invoice line items)
        self.has_tables = self.has_tables or ('|' in t and t.count('|') > 3)
        self.has_handwritten = '[HANDWRITTEN:' in t
        self.has_unreadable = '[unclear]' in t


def _pdf_to_images(pdf_path: str, dpi: int = 300) -> List[bytes]:
    """
    Convert each PDF page to a PNG image in memory.

    These images serve two purposes:
    1. Saved to disk for the VLM (Vision Language Model) in later steps
       (Steps 3, 5, 8, 9) which need to "see" the page layout, stamps, etc.
    2. Used as a visual reference alongside OCR text for boundary detection
       and document classification.

    DPI of 300 is chosen as a balance between readability and file size —
    trade finance documents often have small print (LC terms, fine print clauses).
    """
    doc = fitz.open(pdf_path)
    images = []
    # PyMuPDF uses 72 DPI as base; zoom factor scales to target DPI
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    for page in doc:
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        images.append(img_bytes)
    doc.close()
    return images


def _save_page_image(img_bytes: bytes, output_dir: str, page_num: int) -> str:
    """
    Save a page image to disk for use by VLM steps downstream.
    Returns the file path. Filenames are zero-padded (page_001.png, page_002.png)
    for correct alphabetical sorting.
    """
    path = os.path.join(output_dir, f"page_{page_num:03d}.png")
    with open(path, 'wb') as f:
        f.write(img_bytes)
    return path


def _extract_single_page_pdf(pdf_path: str, page_num: int) -> bytes:
    """
    Extract a single page (1-based) as a one-page PDF in memory.

    Uses PyMuPDF's insert_pdf which preserves the original text layer — critical
    because GLM reads embedded text directly on native-digital pages (Bills of
    Lading, invoices, etc.) and falls back to image OCR only when no text layer
    exists. Rasterizing to PNG loses that text layer and drops content on
    digitally-native pages.
    """
    src = fitz.open(pdf_path)
    out = fitz.open()
    out.insert_pdf(src, from_page=page_num - 1, to_page=page_num - 1)
    data = out.write()
    out.close()
    src.close()
    return data


# Per-page retry ladder: each failed timeout is its own backoff, no extra sleep.
# Max wall time per page = sum = 900s (15 min), after which the page is reported
# empty and the pipeline continues — step02 VLM fallback handles empty text.
_OCR_RETRY_TIMEOUTS = [180, 300, 420]


def _ocr_single_page_glm(pdf_bytes: bytes, page_num: int, prompt: str, timeout_s: int) -> dict:
    """
    Send a single-page PDF blob to the GLM-OCR API and return the extracted text.

    The PDF contains exactly one page so 'pages' is always '1'. page_num is
    only used for logging and response tagging. timeout_s is the hard ceiling
    on this one request — caller decides retry policy.

    Returns a dict with 'page', 'text', 'confidence', 'elapsed', and optionally
    'error' if something went wrong.
    """
    try:
        start = time.time()
        files = {'file': (f'page_{page_num:03d}.pdf', io.BytesIO(pdf_bytes), 'application/pdf')}
        data = {
            'model': GLM_OCR_MODEL,
            'mode': 'per-page',
            'format_output': 'false',
            'pages': '1',
            'prompt': prompt,
        }
        resp = requests.post(GLM_OCR_URL, files=files, data=data, timeout=timeout_s)
        elapsed = time.time() - start

        if resp.status_code != 200:
            return {'page': page_num, 'text': '', 'confidence': 0.0, 'error': f'HTTP {resp.status_code}', 'elapsed': elapsed}

        result = resp.json()
        # Handle different GLM-OCR response formats — the API may return
        # a list of pages, a dict with 'pages' key, or a single dict
        if isinstance(result, list):
            page_data = result[0] if result else {}
        elif isinstance(result, dict):
            pages = result.get('pages', result.get('results', [result]))
            page_data = pages[0] if pages else result
        else:
            page_data = {}

        text = ''
        if isinstance(page_data, dict):
            text = page_data.get('text', page_data.get('raw_ocr_text', page_data.get('content', '')))
        elif isinstance(page_data, str):
            text = page_data

        # Strip markdown code fences if present — some LLM-based OCR models
        # wrap their output in ```markdown``` blocks
        if text.startswith('```'):
            text = text.split('\n', 1)[1] if '\n' in text else text[3:]
        if text.endswith('```'):
            text = text[:-3]
        text = text.strip()

        confidence = page_data.get('confidence', 0.95) if isinstance(page_data, dict) else 0.95

        return {'page': page_num, 'text': text, 'confidence': confidence, 'elapsed': elapsed}

    except requests.exceptions.Timeout:
        return {'page': page_num, 'text': '', 'confidence': 0.0, 'error': f'timeout@{timeout_s}s', 'elapsed': timeout_s}
    except Exception as e:
        return {'page': page_num, 'text': '', 'confidence': 0.0, 'error': str(e), 'elapsed': 0}


def _ocr_page_with_retries(pdf_bytes: bytes, page_num: int, prompt: str, progress) -> dict:
    """
    Call GLM for one page up to 3 times with escalating timeouts [60s, 180s, 300s].

    The first success short-circuits. If all three attempts fail, returns the
    final result dict (with 'error' field populated). Total max wall time per
    page is the sum of the timeouts (540s).
    """
    result = None
    for attempt_idx, tmo in enumerate(_OCR_RETRY_TIMEOUTS, start=1):
        result = _ocr_single_page_glm(pdf_bytes, page_num, prompt, timeout_s=tmo)
        if not result.get('error'):
            if attempt_idx > 1:
                progress(f"  Page {page_num}: recovered on attempt {attempt_idx} ({tmo}s budget)")
            return result
        more = attempt_idx < len(_OCR_RETRY_TIMEOUTS)
        progress(f"  Page {page_num}: attempt {attempt_idx}/{len(_OCR_RETRY_TIMEOUTS)} "
                 f"failed ({result['error']}) — {'retrying' if more else 'giving up'}")
    return result


def run(pdf_path: str, output_dir: str = None, progress_callback=None) -> dict:
    """
    Execute Step 1: Raw OCR extraction for all pages.

    Args:
        pdf_path: Path to the uploaded PDF
        output_dir: Directory to save page images and results
        progress_callback: Optional callback(message: str) for progress updates

    Returns:
        dict with 'pages' (list of PageOCR), 'total_pages', 'elapsed_seconds', 'errors'
    """
    def _progress(msg):
        if progress_callback:
            progress_callback(f"[Step 1] {msg}")
        print(f"[Step 1] {msg}")

    if not os.path.exists(pdf_path):
        return {'error': f'File not found: {pdf_path}', 'pages': [], 'total_pages': 0}

    # Create output directory
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(pdf_path), 'step01_ocr')
    os.makedirs(output_dir, exist_ok=True)

    start_time = time.time()

    # Get page count
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    doc.close()
    _progress(f"PDF has {total_pages} pages. Starting GLM-OCR extraction...")

    # --- OCR Strategy ---
    # Split the PDF into per-page blobs (preserves text layer), then submit each
    # page in parallel with per-request timeouts. A hung GLM request on any
    # single page cannot block the job forever — it aborts at its timeout and
    # retries with a larger budget up to 3 times, then gives up for that page.
    _progress(f"Splitting {total_pages} single-page PDFs and submitting to GLM-OCR "
              f"(concurrency={MAX_CONCURRENT_OCR}, retries={_OCR_RETRY_TIMEOUTS}s)...")
    page_pdfs = {pg: _extract_single_page_pdf(pdf_path, pg) for pg in range(1, total_pages + 1)}

    ocr_start = time.time()
    raw_pages_by_num: dict = {}
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_OCR) as executor:
        futures = {
            executor.submit(_ocr_page_with_retries, page_pdfs[pg], pg, GLM_OCR_PROMPT, _progress): pg
            for pg in range(1, total_pages + 1)
        }
        for future in as_completed(futures):
            pg = futures[future]
            try:
                raw_pages_by_num[pg] = future.result()
            except Exception as e:
                raw_pages_by_num[pg] = {'page': pg, 'text': '', 'confidence': 0.0, 'error': str(e), 'elapsed': 0}
            r = raw_pages_by_num[pg]
            if r.get('error'):
                _progress(f"  Page {pg}: FAILED ({r['error']})")
            else:
                _progress(f"  Page {pg}: {len(r.get('text',''))} chars in {r.get('elapsed',0):.1f}s")

    ocr_elapsed = time.time() - ocr_start
    raw_pages = [raw_pages_by_num[pg] for pg in range(1, total_pages + 1)]
    ok = sum(1 for r in raw_pages if not r.get('error'))
    _progress(f"GLM-OCR complete: {ok}/{total_pages} pages OK in {ocr_elapsed:.1f}s "
              f"({ocr_elapsed/max(total_pages,1):.1f}s/page wall)")

    if not raw_pages:
        _progress("GLM-OCR failed completely — no pages extracted")
        return {'error': 'GLM-OCR returned no pages', 'pages': [], 'total_pages': total_pages}

    # Convert PDF pages to 300 DPI PNG images — these are needed by the
    # Vision Language Model (VLM) in Steps 3, 5, 8, 9 for visual analysis
    _progress("Converting PDF pages to images...")
    page_images = _pdf_to_images(pdf_path, dpi=300)
    img_dir = os.path.join(output_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)

    # Build PageOCR objects — one per page, combining OCR text with the saved image path
    pages = []
    errors = []
    for i, raw_pg in enumerate(raw_pages):
        pg_num = i + 1
        if isinstance(raw_pg, dict):
            text = raw_pg.get('text', raw_pg.get('raw_ocr_text', raw_pg.get('content', '')))
            conf = raw_pg.get('confidence', 0.95)
        elif isinstance(raw_pg, str):
            text = raw_pg
            conf = 0.90
        else:
            text = ''
            conf = 0.0

        # Strip markdown fences
        if text.startswith('```'):
            lines = text.split('\n')
            text = '\n'.join(lines[1:] if len(lines) > 1 else lines)
        if text.endswith('```'):
            text = text[:-3]
        text = text.strip()

        # Save page image
        img_path = ''
        if i < len(page_images):
            img_path = _save_page_image(page_images[i], img_dir, pg_num)

        page = PageOCR(
            page_number=pg_num,
            raw_text=text,
            confidence=conf,
            page_image_path=img_path,
            elapsed_seconds=ocr_elapsed / max(len(raw_pages), 1),
        )

        if page.char_count == 0:
            errors.append(f"Page {pg_num}: empty OCR result")
            _progress(f"  Page {pg_num}: EMPTY")
        elif page.char_count < 30:
            _progress(f"  Page {pg_num}: {page.char_count} chars (very short)")
        else:
            _progress(f"  Page {pg_num}: {page.char_count} chars OK")

        pages.append(page)

    total_elapsed = time.time() - start_time

    # Save results
    result_file = os.path.join(output_dir, 'step01_result.json')
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump({
            'step': 1,
            'step_name': 'Page-Level Raw OCR Extraction',
            'source_file': os.path.basename(pdf_path),
            'total_pages': total_pages,
            'pages_extracted': len(pages),
            'empty_pages': sum(1 for p in pages if p.char_count == 0),
            'total_chars': sum(p.char_count for p in pages),
            'elapsed_seconds': round(total_elapsed, 2),
            'ocr_elapsed_seconds': round(ocr_elapsed, 2),
            'per_page_seconds': round(ocr_elapsed / max(len(pages), 1), 2),
            'errors': errors,
            'pages': [asdict(p) for p in pages],
        }, f, indent=2, ensure_ascii=False)

    _progress(f"Step 1 complete: {len(pages)} pages, {sum(p.char_count for p in pages)} chars, {total_elapsed:.1f}s")

    return {
        'pages': pages,
        'total_pages': total_pages,
        'elapsed_seconds': round(total_elapsed, 2),
        'errors': errors,
        'result_file': result_file,
    }


if __name__ == '__main__':
    # CLI test
    import sys
    if len(sys.argv) < 2:
        print("Usage: python step01_raw_ocr.py <pdf_path>")
        sys.exit(1)
    result = run(sys.argv[1])
    print(f"\nResult: {result['total_pages']} pages, {result['elapsed_seconds']}s")
    for p in result['pages']:
        print(f"  Page {p.page_number}: {p.char_count} chars")
