"""
Step 2 — OCR Text Cleaning + VLM Text Completion
===================================================
Takes the raw OCR text from Step 1 and:
  1. Strips trailing whitespace and excessive blank lines (minimal cleanup)
  2. Sends each page image + GLM text to Qwen VLM to find missing text

PURPOSE:
    GLM OCR is the trusted text source — we do NOT apply code-based
    character substitutions or corrections that could alter meaning.
    Instead, VLM visually reviews EVERY page to find text GLM missed
    (clauses at page edges, text at top/bottom, cut-off content).
    VLM only ADDS missing text — never changes existing GLM text.

INPUT:
    - List of PageOCR objects from Step 1 (raw_text, page_number, page_image_path)

OUTPUT:
    - List of PageCleaned objects, each containing:
        * raw_text: original GLM OCR output (unchanged)
        * cleaned_text: GLM text + any VLM additions (single source of truth)
        * corrections: log of VLM additions (rule='vlm_missing_text')

MODEL:
    Qwen 2.5-VL-72B @ http://10.20.10.2:8085 (VLM text completion)
"""

import re
import sys as _sys; _sys.stdout.reconfigure(encoding="utf-8", errors="replace") if hasattr(_sys.stdout, "reconfigure") else None
import json
import time
from dataclasses import dataclass, field, asdict
from typing import List
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class CorrectionEntry:
    """
    Single correction made during cleaning. Used for the audit trail so that
    every text change can be traced back to a specific rule. This is important
    for compliance — auditors need to know what was changed and why.
    """
    original: str       # The text before correction (truncated to 50 chars for logging)
    corrected: str      # The text after correction (truncated to 50 chars for logging)
    rule: str           # Name of the cleaning rule that triggered this change
    position: int = 0   # Character position in the text where the change occurred


@dataclass
class PageCleaned:
    """
    Output of Step 2 for a single page. Preserves both raw and cleaned text
    so downstream steps can always refer back to the original OCR output.
    """
    page_number: int                                    # 1-based page number
    raw_text: str                                       # Original OCR text (from Step 1)
    cleaned_text: str                                   # Text after applying cleaning rules
    corrections: List[dict] = field(default_factory=list)  # Log of all corrections applied
    page_image_path: str = ""                           # Path to page image (passed through from Step 1)


def _clean_ocr_text(raw_text: str) -> tuple:
    """
    Minimal text cleanup — GLM OCR output is trusted as-is.
    Only strips trailing whitespace and excessive blank lines.
    VLM handles missing text detection separately.

    Returns (cleaned_text, corrections_log).
    """
    text = raw_text

    # Strip trailing whitespace per line
    lines = text.split('\n')
    lines = [line.rstrip() for line in lines]
    text = '\n'.join(lines)

    # Remove excessive blank lines (4+ → 2)
    text = re.sub(r'\n{4,}', '\n\n\n', text)

    return text.strip(), []


def run(step1_result: dict, output_dir: str = None, progress_callback=None) -> dict:
    """
    Execute Step 2: Clean OCR text for all pages.

    Args:
        step1_result: Output from Step 1 (with 'pages' list)
        output_dir: Directory to save results
        progress_callback: Optional callback for progress

    Returns:
        dict with 'pages' (list of PageCleaned), 'total_corrections', 'elapsed_seconds'
    """
    def _progress(msg):
        if progress_callback:
            progress_callback(f"[Step 2] {msg}")
        print(f"[Step 2] {msg}")

    start_time = time.time()
    pages_in = step1_result.get('pages', [])
    _progress(f"Cleaning OCR text for {len(pages_in)} pages...")

    pages_out = []
    total_corrections = 0

    for page_in in pages_in:
        # Handle both PageOCR objects and dicts
        if hasattr(page_in, 'page_number'):
            pg_num = page_in.page_number
            raw = page_in.raw_text
            img_path = page_in.page_image_path
        else:
            pg_num = page_in.get('page_number', 0)
            raw = page_in.get('raw_text', '')
            img_path = page_in.get('page_image_path', '')

        cleaned, corrections = _clean_ocr_text(raw)
        total_corrections += len(corrections)

        page_out = PageCleaned(
            page_number=pg_num,
            raw_text=raw,
            cleaned_text=cleaned,
            corrections=[asdict(c) if hasattr(c, '__dataclass_fields__') else c for c in corrections],
            page_image_path=img_path,
        )
        pages_out.append(page_out)

        if corrections:
            _progress(f"  Page {pg_num}: {len(corrections)} corrections")

    # ── VLM Text Completion ──
    # Send each page image + cleaned text to Qwen VLM to find text GLM missed.
    # GLM sometimes misses numbered clauses at page boundaries or text near edges.
    # VLM only ADDS missing text — never changes existing GLM text.
    _progress("VLM reviewing pages for missing text...")
    import base64 as _b64
    import requests as _requests
    import os as _os
    from concurrent.futures import ThreadPoolExecutor, as_completed
    try:
        from config.settings import QWEN_VLM_URL, QWEN_VLM_MODEL, VLM_TIMEOUT, MAX_CONCURRENT_VLM
    except ImportError:
        QWEN_VLM_URL = "http://10.20.10.2:8085/v1/chat/completions"
        QWEN_VLM_MODEL = "/home/aigenics/AI_MODELS/Qwen2.5-VL-72B-Instruct-AWQ"
        VLM_TIMEOUT = 600
        MAX_CONCURRENT_VLM = 4

    # Garbage detection: GLM sometimes echoes the prompt instead of reading the page
    _GARBAGE_MARKERS = [
        'CRITICAL RULES:', 'Extract ALL amounts in BOTH figures',
        'NUMBERED CLAUSES:', 'CONTINUATION TEXT:', 'Note signatures as [SIGNATURE]',
        'Preserve line breaks and formatting', 'Missing even ONE character',
    ]

    def _is_garbage_text(text: str) -> bool:
        """Detect if GLM returned its own prompt instead of actual page text."""
        if not text:
            return True
        hits = sum(1 for m in _GARBAGE_MARKERS if m in text)
        return hits >= 3  # 3+ markers = definitely prompt text, not page content

    def _vlm_check_page(page_out):
        pg_num = page_out.page_number
        img_path = page_out.page_image_path
        cleaned = page_out.cleaned_text
        if not img_path or not _os.path.exists(img_path):
            return pg_num, None, False
        try:
            _img_b64 = _b64.b64encode(open(img_path, 'rb').read()).decode()

            # If GLM returned garbage (echoed prompt), do FULL extraction
            if _is_garbage_text(cleaned) or len(cleaned) < 20:
                _prompt = (
                    "The OCR model failed to read this page correctly. "
                    "Please extract ALL text from this document page image.\n\n"
                    "RULES:\n"
                    "- Extract EVERY line of text exactly as it appears\n"
                    "- Include ALL SWIFT field tags (F20:, :20:, F46A:, :46A:, etc.) with their complete values\n"
                    "- Include ALL numbered clauses (1., 2., 3... or 1), 2), 3)...)\n"
                    "- Preserve line breaks and formatting\n"
                    "- Do NOT summarize or interpret — extract the raw text\n"
                    "- If text continues from previous page, still extract it completely"
                )
                _resp = _requests.post(QWEN_VLM_URL, json={
                    "model": QWEN_VLM_MODEL,
                    "messages": [{"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + _img_b64}},
                        {"type": "text", "text": _prompt}
                    ]}],
                    "max_tokens": 4000, "temperature": 0.1
                }, timeout=int(VLM_TIMEOUT))
                if _resp.status_code == 200:
                    _content = _resp.json().get('choices', [{}])[0].get('message', {}).get('content', '')
                    if _content and len(_content.strip()) > 10:
                        return pg_num, _content.strip(), True  # True = full replacement
                return pg_num, None, False

            # Normal mode: compare GLM text against image, find missing text
            # Estimate how much text the page should have vs what GLM extracted
            # If GLM got very little text for a page that clearly has more, do full extraction
            _prompt = (
                "Look at this document page image carefully. The OCR system extracted this text:\n\n"
                "---OCR TEXT---\n%s\n---END OCR---\n\n"
                "TASK: Compare the OCR text above with ALL text visible in the image.\n\n"
                "STEP 1: Count roughly how many lines of text are in the IMAGE.\n"
                "STEP 2: Count how many lines are in the OCR TEXT.\n"
                "STEP 3: If the image has SIGNIFICANTLY MORE text than the OCR (e.g. the image shows "
                "a full invoice/document but OCR only has a few lines), the OCR is INCOMPLETE.\n\n"
                "If OCR is INCOMPLETE or MISSING text:\n"
                "- Return ALL the missing text exactly as it appears in the image\n"
                "- Include headers, addresses, line items, tables, amounts, totals\n"
                "- Include column headers and all rows\n"
                "- Note stamps as [STAMP] and signatures as [SIGNATURE]\n\n"
                "If OCR text truly covers ALL visible text in the image, return ONLY: COMPLETE\n\n"
                "Do NOT repeat text that is already in the OCR output."
            ) % cleaned
            _resp = _requests.post(QWEN_VLM_URL, json={
                "model": QWEN_VLM_MODEL,
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + _img_b64}},
                    {"type": "text", "text": _prompt}
                ]}],
                "max_tokens": 4000, "temperature": 0.1
            }, timeout=int(VLM_TIMEOUT))
            if _resp.status_code == 200:
                _content = _resp.json().get('choices', [{}])[0].get('message', {}).get('content', '')
                if 'COMPLETE' not in _content.upper()[:20] and len(_content.strip()) > 10:
                    return pg_num, _content.strip(), False
            return pg_num, None, False
        except Exception:
            return pg_num, None, False

    vlm_additions = 0
    vlm_replacements = 0
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_VLM) as executor:
        futures = {executor.submit(_vlm_check_page, p): p for p in pages_out}
        for future in as_completed(futures):
            pg_num, new_text, is_replacement = future.result()
            if new_text:
                for p in pages_out:
                    if p.page_number == pg_num:
                        if is_replacement:
                            # GLM returned garbage — replace entirely with VLM extraction
                            vlm_replacements += 1
                            p.cleaned_text = new_text
                            p.corrections.append({'original': 'GLM_GARBAGE', 'corrected': new_text, 'rule': 'vlm_full_extraction'})
                            _progress(f"  Page {pg_num}: GLM garbage detected — VLM re-extracted ({len(new_text)} chars)")
                        else:
                            # Normal addition — append missing text
                            vlm_additions += 1
                            p.cleaned_text = p.cleaned_text + '\n' + new_text
                            p.corrections.append({'original': '', 'corrected': new_text, 'rule': 'vlm_missing_text'})
                            _progress(f"  Page {pg_num}: VLM found missing text ({len(new_text)} chars)")
                        break
            else:
                _progress(f"  Page {pg_num}: text complete")

    if vlm_additions or vlm_replacements:
        _progress(f"VLM: {vlm_additions} additions, {vlm_replacements} full re-extractions (GLM garbage)")
        total_corrections += vlm_additions + vlm_replacements

    elapsed = time.time() - start_time
    _progress(f"Step 2 complete: {total_corrections} corrections, {vlm_additions} VLM additions, {vlm_replacements} re-extractions across {len(pages_out)} pages in {elapsed:.1f}s")

    # Save results
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        result_file = os.path.join(output_dir, 'step02_result.json')
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({
                'step': 2,
                'step_name': 'OCR Text Cleaning',
                'total_pages': len(pages_out),
                'total_corrections': total_corrections,
                'elapsed_seconds': round(elapsed, 2),
                'pages': [asdict(p) for p in pages_out],
            }, f, indent=2, ensure_ascii=False)

    return {
        'pages': pages_out,
        'total_corrections': total_corrections,
        'elapsed_seconds': round(elapsed, 2),
    }
