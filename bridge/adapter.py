"""Maps 8083 classification.json → 8082 step result dicts.

8083 returns one big dict with:
  • pages              — per-page OCR + classification
  • logical_documents  — SWIFT docs (kind=swift) + LC-required (kind=lc_required) + extras (kind=extra)
  • physical_packets   — physical doc clusters
  • final_lc           — 8083's FLC (ignored; we let 8082's step06 build the better FLC)
  • lc_requirements    — 8083's F46A parse (ignored unless step06 wants a hint)
  • timings            — per-step seconds

8082 expects (the bits step06 onward consume):
  step5_result = {
    'page_texts': {page_num: cleaned_text},
    'packets':    [list of packet dicts],
  }
  Each packet dict needs: packet_id, mt_type, page_numbers, pages (list of
  {page_number, refined_text}), is_799_amendment, source_mt.

For downstream verification + UI that reads step01..step05 JSONs directly
from disk, we also write minimal stub JSONs so old code paths keep working.
"""
from __future__ import annotations

import os
import json
from typing import Dict, List, Optional, Tuple


# ── doc_type → mt_type mapping for SWIFT logical docs ──────────────
# 8083's `document_type` values map to 8082's `mt_type` enum.
_SWIFT_DOCTYPE_TO_MT = {
    'lc':                                      'MT700',
    'lc continuation':                         'MT700',
    'mt700':                                   'MT700',
    'mt701':                                   'MT700',
    'authorisation to reimburse':              'MT740',
    'authorisation to reimburse amendment':    'MT747',
    'reimbursement claim':                     'MT742',
    'bank guarantee':                          'MT760',
    'bank guarantee amendment':                'MT767',
    'mt799':                                   'MT799',
    'mt999':                                   'MT999',
    'acknowledgment advice':                   'MT730',
    'discharge advice':                        'MT732',
    'refusal advice':                          'MT734',
    'refusal notice':                          'MT734',
    'trade finance query':                     'MT759',
    'third-bank lc advice':                    'MT710',
    'transferred lc':                          'MT720',
}


def _doc_type_to_mt(doc_type: str, kind: str = '') -> str:
    """Map an 8083 `document_type` (e.g. 'Amendment 3', 'LC', 'MT799') and
    `kind` (swift/lc_required/extra) to 8082's `mt_type` string.

    Returns 'shipping' for non-SWIFT docs."""
    if not doc_type:
        return 'shipping' if kind != 'swift' else ''
    d = doc_type.strip()
    d_lower = d.lower()
    # "Amendment N" or "Amendment N Continuation" → MT707
    if d_lower.startswith('amendment'):
        return 'MT707'
    # Exact lookup
    if d_lower in _SWIFT_DOCTYPE_TO_MT:
        return _SWIFT_DOCTYPE_TO_MT[d_lower]
    # Non-SWIFT logical doc kinds (lc_required, extra) → shipping
    if kind in ('lc_required', 'extra'):
        return 'shipping'
    # Anything else with kind=swift but unknown family → empty (step06
    # will route to other_mt_packets).
    return ''


def _parse_amendment_number(doc_type: str) -> Optional[int]:
    """Extract N from 'Amendment N' / 'Amendment N Continuation'.
    Returns None when the doc type is not an amendment."""
    import re
    m = re.search(r'^amendment\s+(\d+)', (doc_type or '').strip(), re.IGNORECASE)
    return int(m.group(1)) if m else None


def _build_page_texts(c8083: Dict) -> Dict[int, str]:
    """Page-number → cleaned text dict. 8083 stores GLM-OCR cleaned text
    (with hallucination guard + prompt-line stripping already applied) in
    `pages[*].text`."""
    out: Dict[int, str] = {}
    for p in c8083.get('pages', []) or []:
        pn = p.get('page_number')
        text = p.get('text') or ''
        if isinstance(pn, int) and pn > 0 and text:
            out[pn] = text
    return out


def _build_packets(c8083: Dict, page_texts: Dict[int, str]) -> List[Dict]:
    """Map 8083 `logical_documents` → 8082 packet list.

    Each 8083 logical doc becomes one packet. SWIFT docs get a proper
    mt_type; matched LC-required docs and extras get mt_type='shipping'.
    The packet's `pages` list contains lightweight page dicts with
    page_number + refined_text so step06's _get_packet_refined_text
    works whether or not _PAGE_TEXT_LOOKUP is set.
    """
    packets: List[Dict] = []
    next_id = 1
    for ld in c8083.get('logical_documents', []) or []:
        all_pages = ld.get('all_pages') or []
        kind = ld.get('kind') or ''
        doc_type = ld.get('document_type') or ''
        mt = _doc_type_to_mt(doc_type, kind)
        amd_no = _parse_amendment_number(doc_type)
        # Page dicts with the text step06 needs
        page_dicts = []
        for pn in all_pages:
            t = page_texts.get(pn, '')
            page_dicts.append({
                'page_number': pn,
                'raw_text': t,
                'cleaned_text': t,
                'refined_text': t,
            })
        # Detect MT799-promoted-amendment hint (8083 marks via document_type
        # 'Amendment N' even when sourced from MT799). 8083 also keeps the
        # original mt_number for SWIFT pattern detections.
        kf_mt = ld.get('mt_number') or ''
        is_799_amd = (
            mt == 'MT707' and (
                kf_mt.upper() in ('MT799', 'MT999')
                or 'mt799' in (ld.get('logical_doc_id') or '').lower()
            )
        )
        packets.append({
            'packet_id':       next_id,
            'mt_type':         mt,
            'mt_confidence':   0.95,
            'page_numbers':    list(all_pages),
            'page_count':      len(all_pages),
            'pages':           page_dicts,
            'amendment_number': amd_no,
            'lc_reference':    ld.get('lc_reference') or '',
            'sender_reference': ld.get('sender_reference') or '',
            'related_reference': ld.get('related_reference') or '',
            'reconciliation_method': 'passthrough',
            'was_reclassified': False,
            'text_was_corrected': False,
            'is_799_amendment': is_799_amd,
            'source_mt':       kf_mt or ('MT799' if is_799_amd else ''),
            # Carry through 8083's extracted fields so downstream
            # verification can read stamps/signatures/endorsements
            # without re-extracting.
            'extracted_fields': ld.get('fields') or {},
            'document_type':   doc_type,
            'kind_from_8083':  kind,
            'bl_subtype':      ld.get('bl_subtype', ''),
            'bl_status_flags': ld.get('bl_status_flags', []),
            'awb_subtype':     ld.get('awb_subtype', ''),
            'bl_signing_capacity': ld.get('bl_signing_capacity'),
            'awb_signing_capacity': ld.get('awb_signing_capacity'),
            'is_house_bl':     ld.get('is_house_bl'),
            'is_carrier_signed': ld.get('is_carrier_signed'),
        })
        next_id += 1
    return packets


def _flatten_stamps_and_signatures(fields: Dict) -> Tuple[List[Dict], List[Dict]]:
    """Pull stamps + signatures out of 8083's extracted_fields into 8082's
    flat per-packet shape. step14_verification reads:
        pkt.get('stamps', [])      → [{'text', 'type'}, ...]
        pkt.get('signatures', [])  → [{'description', 'type'}, ...]
    """
    stamps: List[Dict] = []
    sigs: List[Dict] = []
    if not isinstance(fields, dict):
        return stamps, sigs

    # ── Stamps: from Signature.stamps[], Stamps[], Bank Stamps[]
    sig_obj = fields.get('Signature') or {}
    if isinstance(sig_obj, dict):
        for s in sig_obj.get('stamps') or []:
            if s:
                stamps.append({'text': str(s), 'type': 'rubber_stamp'})
    for s in fields.get('Stamps') or []:
        if s:
            stamps.append({'text': str(s), 'type': 'rubber_stamp'})
    for bs in fields.get('Bank Stamps') or []:
        if isinstance(bs, dict):
            txt = bs.get('stamp_text') or ''
            if txt:
                stamps.append({
                    'text': txt,
                    'type': 'bank_stamp',
                    'date': bs.get('stamp_date'),
                    'role': bs.get('stamp_role'),
                    'bank': bs.get('bank_name'),
                    'branch': bs.get('branch'),
                })

    # ── Signatures: from Signature object (BL/AWB schema)
    if isinstance(sig_obj, dict):
        parts = []
        for k in ('shipper_signature', 'carrier_signature',
                  'signatory_name', 'signatory_company',
                  'signatory_role', 'signing_capacity',
                  'capacity_verbatim'):
            v = sig_obj.get(k)
            if v:
                parts.append(f'{k}: {v}')
        if parts:
            sigs.append({
                'description': ' / '.join(parts),
                'type': 'handwritten',
            })

    # ── Endorsements (BL) treated as signatures with date
    for end in fields.get('Endorsements') or []:
        if isinstance(end, dict):
            parts = []
            for k in ('endorser_name', 'endorser_capacity', 'endorsed_to',
                      'endorsement_text', 'sequence'):
                v = end.get(k)
                if v:
                    parts.append(f'{k}: {v}')
            sigs.append({
                'description': ' / '.join(parts),
                'type': 'endorsement',
                'date': end.get('endorsement_date'),
                'is_blank': end.get('is_blank_endorsement'),
            })

    return stamps, sigs


def adapt_8083_to_step_results(c8083: Dict,
                                 results_dir: str = None
                                 ) -> Dict[str, Dict]:
    """Convert 8083's classification.json into the per-step result dicts
    that 8082's step06+ pipeline expects.

    If `results_dir` is provided, also write minimal stub JSONs to disk
    at results_dir/stepXX/stepXX_result.json so downstream code that
    reads from disk (step10/11/14/20, server.py UI loaders) keeps
    working. The stubs only carry the fields those readers actually use.

    Returns a dict keyed by step name with:
      'step01': raw_ocr stub (pages with text + image paths)
      'step02': cleaned text stub
      'step03': page sequencing stub (packets formed)
      'step04': MT identification stub
      'step05': MT reconciliation — the REAL input step06 consumes
      'step08': shipping classification stub (carries extracted fields)
      'step09': shipping reconciliation stub
    """
    page_texts = _build_page_texts(c8083)
    packets = _build_packets(c8083, page_texts)

    # Augment each packet with flattened stamps/signatures so the
    # step08/09 readers (which expect pkt['stamps']/['signatures'])
    # work without changes.
    for pkt in packets:
        st, sg = _flatten_stamps_and_signatures(pkt.get('extracted_fields', {}))
        pkt['stamps'] = st
        pkt['signatures'] = sg

    # ── step01: raw OCR ─────────────────────────────────────────────
    step01 = {
        'pages': [
            {
                'page_number': p.get('page_number'),
                'text':         p.get('text', ''),
                'raw_text':     p.get('text', ''),
                'image_path':   '',  # 8083 hosts images via /api/page-image
                'ocr_method':   p.get('method', 'glm_ocr'),
            }
            for p in (c8083.get('pages') or [])
        ],
        'page_count': c8083.get('total_pages', 0),
        'source':     '8083_classifier',
    }

    # ── step02: cleaned text ───────────────────────────────────────
    step02 = {
        'pages': [
            {
                'page_number':  pn,
                'cleaned_text': txt,
                'changes':      [],
            }
            for pn, txt in sorted(page_texts.items())
        ],
        'source': '8083_classifier',
    }

    # ── step03: page sequencing → packets formed ───────────────────
    # The classifier already grouped pages into packets; expose that
    # mapping so any downstream UI showing "packet boundaries" works.
    step03 = {
        'packets': [
            {
                'packet_id':    pkt['packet_id'],
                'page_numbers': pkt['page_numbers'],
                'page_count':   pkt['page_count'],
                'doc_type_hint': pkt.get('document_type', ''),
            }
            for pkt in packets
        ],
        'classifications': [
            {
                'page_number':  p.get('page_number'),
                'doc_type':     p.get('doc_type', ''),
                'method':       p.get('method', ''),
                'confidence':   p.get('confidence', 0.0),
            }
            for p in (c8083.get('pages') or [])
        ],
        'source': '8083_classifier',
    }

    # ── step04: MT identification ──────────────────────────────────
    step04 = {
        'packets': [
            {
                'packet_id':    pkt['packet_id'],
                'mt_type':      pkt['mt_type'],
                'mt_confidence': pkt['mt_confidence'],
                'is_799_amendment': pkt['is_799_amendment'],
                'source_mt':    pkt.get('source_mt', ''),
            }
            for pkt in packets
        ],
        'source': '8083_classifier',
    }

    # ── step05: MT reconciliation — the REAL input step06 needs ───
    step05 = {
        'packets':    packets,
        'page_texts': page_texts,
        'source':     '8083_classifier',
    }

    # ── step08: shipping classification ────────────────────────────
    # Carry extracted_fields so verification step14 + report step20
    # can render structured tables (Containers, Parties, etc.).
    step08 = {
        'classified_packets': [
            {
                'packet_id':     pkt['packet_id'],
                'document_type': pkt.get('document_type', ''),
                'page_numbers':  pkt['page_numbers'],
                'mt_type':       pkt['mt_type'],
                'stamps':        pkt['stamps'],
                'signatures':    pkt['signatures'],
                'extracted_fields': pkt.get('extracted_fields', {}),
                'bl_subtype':    pkt.get('bl_subtype', ''),
                'bl_status_flags': pkt.get('bl_status_flags', []),
                'awb_subtype':   pkt.get('awb_subtype', ''),
            }
            for pkt in packets
            if pkt['mt_type'] == 'shipping'
        ],
        'source': '8083_classifier',
    }

    # ── step09: shipping reconciliation ────────────────────────────
    step09 = {
        'reconciled_packets': step08['classified_packets'],
        'source': '8083_classifier',
    }

    out = {
        'step01': step01,
        'step02': step02,
        'step03': step03,
        'step04': step04,
        'step05': step05,
        'step08': step08,
        'step09': step09,
    }

    # Persist stubs if a results_dir was given
    if results_dir:
        for step_name, payload in out.items():
            step_dir = os.path.join(results_dir, step_name)
            os.makedirs(step_dir, exist_ok=True)
            stub_path = os.path.join(step_dir, f'{step_name}_result.json')
            with open(stub_path, 'w', encoding='utf-8') as f:
                # 8082's step05 dataclasses serialize fine via default=str
                json.dump(payload, f, ensure_ascii=False, indent=2,
                          default=str)
        # Also stash 8083's full classification.json for forensics
        full_path = os.path.join(results_dir, 'classification_8083.json')
        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(c8083, f, ensure_ascii=False, indent=2, default=str)

    return out
