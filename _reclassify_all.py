"""
One-shot reclassifier — re-runs ONLY step08 (shipping classification)
and step09 (shipping reconciliation) for every existing job, using the
already-on-disk step03 (sequencing) and step07 (clause extraction)
outputs as input. Skips OCR (step01/02) entirely so it's fast.

Used after a step08 logic change (e.g. the structural-page guard) when
we want to re-process all historical jobs without re-doing GLM OCR.
"""
import json
import os
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, str(Path(__file__).parent))

from steps import step08_shipping_classification as s08
from steps import step09_shipping_reconciliation as s09

RESULTS_DIR = 'results'


def _load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def reclassify_job(job_id):
    base = os.path.join(RESULTS_DIR, job_id)
    print(f'\n{"=" * 70}\n>>> Job: {job_id}\n{"=" * 70}')

    s3 = _load_json(os.path.join(base, 'step03', 'step03_result.json'))
    if not s3:
        print('  SKIP — no step03 result on disk')
        return False
    s7 = _load_json(os.path.join(base, 'step07', 'step07_result.json'))
    if not s7:
        print('  SKIP — no step07 result on disk')
        return False
    s2 = _load_json(os.path.join(base, 'step02', 'step02_result.json'))
    if not s2:
        print('  SKIP — no step02 result on disk')
        return False

    # Build a {page_number: cleaned_text} index from Step 2.
    # CRITICAL: step03 packets store ONLY page-number references, the actual
    # cleaned text lives in step02 indexed by page number. Without this
    # population, step08 + step09 + verification all see EMPTY documents
    # and the VLM hallucinates everything (wrong invoice numbers, wrong
    # amounts, missing fields that are clearly on the page, etc.).
    _s2_page_text = {}
    _s2_page_imgs = {}
    for pg in s2.get('pages', []) or []:
        if not isinstance(pg, dict):
            continue
        pn = pg.get('page_number')
        txt = pg.get('cleaned_text') or pg.get('raw_text') or ''
        img = pg.get('page_image_path') or ''
        if pn is not None:
            try:
                _s2_page_text[int(pn)] = txt
                if img:
                    _s2_page_imgs[int(pn)] = img
            except (TypeError, ValueError):
                pass

    print(f'  step02 page texts indexed: {len(_s2_page_text)}')
    print(f'  step03 packets: {len(s3.get("packets", []))}')
    print(f'  step07 required_documents: {len(s7.get("required_documents", []))}')

    # ──────────────────────────────────────────────────────────
    # POST-PROCESSING ORDER:
    #   1. Populate text on each packet (from step 2 page index)
    #   2. Detect emails and re-classify any 'unknown' packet whose
    #      text looks like an email message
    #   3. Merge adjacent same-type fragments (LC/LC) and merge any
    #      remaining unknowns into the previous real document as a
    #      continuation
    # ──────────────────────────────────────────────────────────

    def _pkt_dt(p):
        return (p.get('document_type') or '').strip().lower()

    def _pkt_text(p):
        return p.get('cleaned_text') or p.get('raw_text') or ''

    def _are_adjacent(prev_pages, next_pages):
        if not prev_pages or not next_pages:
            return False
        try:
            return int(min(next_pages)) == int(max(prev_pages)) + 1
        except (TypeError, ValueError):
            return False

    def _merge_into(prev, curr):
        """Merge curr packet into prev packet (in-place on prev)."""
        prev_pages = list(prev.get('page_numbers') or [])
        for pn in (curr.get('page_numbers') or []):
            if pn not in prev_pages:
                prev_pages.append(pn)
        prev['page_numbers'] = sorted(prev_pages)
        prev_pgs = list(prev.get('pages') or [])
        for pg in (curr.get('pages') or []):
            if pg not in prev_pgs:
                prev_pgs.append(pg)
        prev['pages'] = prev_pgs
        # Concatenate text
        ptxt = _pkt_text(prev)
        ctxt = _pkt_text(curr)
        if ctxt and ctxt not in ptxt:
            joined = (ptxt + '\n' + ctxt).strip() if ptxt else ctxt
            prev['cleaned_text'] = joined
            prev['raw_text'] = joined
        # Merge image paths
        prev_imgs = list(prev.get('page_image_paths') or [])
        for img in (curr.get('page_image_paths') or []):
            if img not in prev_imgs:
                prev_imgs.append(img)
        prev['page_image_paths'] = prev_imgs
        # Merge stamps/signatures/seals/logos
        for fld in ('stamps', 'signatures', 'seals', 'logos'):
            base = list(prev.get(fld) or [])
            for x in (curr.get(fld) or []):
                if x not in base:
                    base.append(x)
            prev[fld] = base

    # ── STEP 1: POPULATE TEXT (must run before email detection / merge) ──
    # CRITICAL: step03 packets store ONLY page-number references, the actual
    # cleaned text lives in step02 indexed by page number. Without this
    # population, step08 + step09 + verification all see EMPTY documents
    # and the VLM hallucinates everything.
    for pkt in s3.get('packets', []):
        page_nums = pkt.get('page_numbers') or []
        if not page_nums:
            for pg in pkt.get('pages', []) or []:
                if isinstance(pg, dict) and pg.get('page_number') is not None:
                    try:
                        page_nums.append(int(pg['page_number']))
                    except (TypeError, ValueError):
                        pass
        parts = []
        imgs = []
        for pn in page_nums:
            try:
                pn_int = int(pn)
            except (TypeError, ValueError):
                continue
            t = _s2_page_text.get(pn_int, '')
            if t:
                parts.append(t)
            i = _s2_page_imgs.get(pn_int, '')
            if i:
                imgs.append(i)
        if parts and not pkt.get('cleaned_text'):
            pkt['cleaned_text'] = '\n'.join(parts)
        if parts and not pkt.get('raw_text'):
            pkt['raw_text'] = '\n'.join(parts)
        if imgs and not pkt.get('page_image_paths'):
            pkt['page_image_paths'] = imgs

    # ── STEP 2: EMAIL DETECTION ──
    # Step 3 does not have an "Email / Cover Email" type in its prompt, so
    # email pages (covering emails sent alongside the documents) often come
    # back as 'unknown'. Promote any unknown packet whose text looks like
    # an email into 'Covering Email'. This stops the merge pass from
    # mistakenly absorbing the email into the previous shipping document.
    import re as _re_email
    _EMAIL_SIGNALS = [
        r'(?:^|\n)\s*From\s*:\s*\S',
        r'(?:^|\n)\s*To\s*:\s*\S',
        r'(?:^|\n)\s*Subject\s*:\s*\S',
        r'(?:^|\n)\s*Sent\s*:\s*\S',
        r'(?:^|\n)\s*Cc\s*:\s*\S',
        r'(?:^|\n)\s*Bcc\s*:\s*\S',
        r'\bRe\s*:\s*',                       # reply prefix
        r'\bFwd?\s*:\s*',                     # forward prefix
        r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}',  # email address
        r'\bB[\.\s]?regards\b',
        r'\bBest\s+regards\b',
        r'\bKind\s+regards\b',
        r'\bRegards\s*[,\.]\s*\n',
        r'发给',                               # Chinese "sent to"
        r'详细信息',                            # Chinese "details"
    ]
    for pkt in s3.get('packets', []):
        if _pkt_dt(pkt) != 'unknown':
            continue
        txt = _pkt_text(pkt)
        if not txt or len(txt) < 30:
            continue
        hits = sum(1 for p in _EMAIL_SIGNALS if _re_email.search(p, txt, _re_email.IGNORECASE))
        # 2+ signals = confident email (an email address alone isn't enough)
        if hits >= 2:
            pkt['document_type'] = 'Covering Email'
            print(f'  [email] {pkt.get("packet_id","?")} pages={pkt.get("page_numbers")} reclassified unknown -> Covering Email ({hits} signals)')

    # ── STEP 3: MERGE PASS ──
    # Step 3 sometimes splits a multi-page document into separate packets
    # (LC pages 1-4 split into pages [1] + [2,3,4], or commercial invoice
    # pages 7-10 split into 4 single-page packets). Merge adjacent same-
    # type packets so downstream steps see one document with all its pages.
    #
    # Also: any 'unknown' packet that survived the email detection above
    # is treated as a true continuation of the previous real document.
    _packets = list(s3.get('packets', []) or [])
    merged = []
    for pkt in _packets:
        if not merged:
            merged.append(pkt)
            continue
        prev = merged[-1]
        prev_dt = _pkt_dt(prev)
        curr_dt = _pkt_dt(pkt)
        prev_pages = prev.get('page_numbers') or []
        curr_pages = pkt.get('page_numbers') or []

        # Rule A — adjacent same-type merge (LC/LC, Commercial Invoice/CI, etc.)
        if (prev_dt and curr_dt and prev_dt == curr_dt
                and _are_adjacent(prev_pages, curr_pages)):
            _merge_into(prev, pkt)
            print(f'  [merge A] {pkt.get("packet_id","?")} ({curr_dt}, pages {curr_pages}) -> {prev.get("packet_id","?")} (now pages {prev["page_numbers"]})')
            continue

        # Rule B — adjacent unknown after a real shipping doc → continuation
        _STRUCTURAL_OR_NULL = {'', 'unknown', 'header page', 'blank page',
                               'endorsement page', 'fusion header',
                               'covering email', 'cover email'}
        if (curr_dt == 'unknown'
                and prev_dt not in _STRUCTURAL_OR_NULL
                and _are_adjacent(prev_pages, curr_pages)):
            _merge_into(prev, pkt)
            print(f'  [merge B] {pkt.get("packet_id","?")} (unknown, pages {curr_pages}) -> {prev.get("packet_id","?")} ({prev_dt}, now pages {prev["page_numbers"]})')
            continue

        merged.append(pkt)

    s3['packets'] = merged
    print(f'  packets after merge pass: {len(merged)} (was {len(_packets)})')

    # ── Split LC / amendment / MT799 / structural packets out of the list ──
    # Step 3 stores ALL packets under a single 'packets' key. Server.py's
    # inline step04 logic separates LC-side (MT700/MT707/MT799) from
    # shipping-side packets before sending shipping ones to step 8.
    # We replicate that split here so step 8 only sees actual shipping
    # documents — otherwise every LC page gets force-classified as some
    # shipping document type.
    _LC_TYPES = {
        'lc', 'amendment', 'mt700', 'mt701', 'mt705', 'mt707', 'mt708',
        'mt710', 'mt711', 'mt720', 'mt721',
    }
    _FREE_FORMAT_TYPES = {
        'mt799', 'mt999', 'free format message',
        'free_format_message', 'bank-to-bank message',
    }
    _BACK_PAGE_TYPES = {'blank page', 'blank_page', 'endorsement page'}
    shipping_packets = []
    for pkt in s3.get('packets', []):
        dt = (pkt.get('document_type', '') or '').strip().lower()
        if dt in _LC_TYPES or dt in _FREE_FORMAT_TYPES or dt in _BACK_PAGE_TYPES:
            continue
        shipping_packets.append(pkt)

    print(f'  shipping packets after LC/MT799 split: {len(shipping_packets)}')

    # Step 8 reads from 'shipping_packets' first, falls back to 'packets'
    s3_for_s8 = dict(s3)
    s3_for_s8['shipping_packets'] = shipping_packets

    # ── Step 8: Shipping Classification ──
    print('  [Step 8] Re-classifying shipping packets...')
    t0 = time.time()
    try:
        s8_result = s08.run(s3_for_s8, s7, output_dir=os.path.join(base, 'step08'),
                            progress_callback=lambda m: print(f'    {m}'))
    except Exception as e:
        print(f'  [Step 8] FAILED: {e}')
        return False
    print(f'  [Step 8] done in {time.time() - t0:.1f}s')

    # ── Step 9: Shipping Reconciliation ──
    print('  [Step 9] Reconciling shipping packets...')
    t0 = time.time()
    try:
        s9_result = s09.run(s8_result, output_dir=os.path.join(base, 'step09'),
                            progress_callback=lambda m: print(f'    {m}'))
    except Exception as e:
        print(f'  [Step 9] FAILED: {e}')
        return False
    print(f'  [Step 9] done in {time.time() - t0:.1f}s')

    return True


def main():
    if not os.path.isdir(RESULTS_DIR):
        print(f'No {RESULTS_DIR} directory')
        return

    jobs = sorted(d for d in os.listdir(RESULTS_DIR)
                  if os.path.isdir(os.path.join(RESULTS_DIR, d)))
    if len(sys.argv) > 1:
        only = sys.argv[1]
        jobs = [j for j in jobs if j == only]

    print(f'Found {len(jobs)} jobs to reclassify')
    ok = 0
    failed = 0
    for j in jobs:
        if reclassify_job(j):
            ok += 1
        else:
            failed += 1

    print(f'\n{"=" * 70}')
    print(f'TOTAL: {ok} succeeded, {failed} failed/skipped')


if __name__ == '__main__':
    main()
