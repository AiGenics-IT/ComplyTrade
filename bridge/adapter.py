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
import re as _re_adapter
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


def _legacy_document_type(doc_type_8083: str, mt_type: str) -> str:
    """Convert 8083's canonical doc_type ('LC', 'Amendment 3', 'Bill of
    Lading', 'MT799') into the legacy step03 `document_type` strings
    that downstream regenerate / rerun endpoints check via substring
    match (`'lc' in dt`, `'amend' in dt`, `'mt7' in dt`).

    Mapping:
      LC / MT700 family       -> 'LC'
      Amendment N             -> 'Amendment'  (so `'amend' in dt` is True)
      MT799 / MT999           -> 'MT799' / 'MT999'
      Acknowledgment Advice   -> 'MT730'
      Refusal Notice          -> 'MT734'
      Authorisation Reimburse -> 'MT740'
      Bank Guarantee          -> 'MT760'
      Bank Guarantee Amend    -> 'MT767'
      shipping                -> the original 8083 doc_type (so the UI
                                 can show 'Bill of Lading' / 'Commercial
                                 Invoice' / etc. unchanged).
    """
    if not doc_type_8083:
        return ''
    dt_lower = doc_type_8083.lower()
    if dt_lower.startswith('amendment'):
        return 'Amendment'
    if mt_type in ('MT700', 'MT701'):
        return 'LC'
    if mt_type.startswith('MT') and mt_type not in ('MT700', 'MT701'):
        return mt_type
    return doc_type_8083


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


def _build_page_doctypes(c8083: Dict) -> Dict[int, str]:
    """Page-number → 8083-assigned doc_type (lowercase). Used to filter
    out T&C / overleaf / Conditions-of-Carriage pages from the body we
    feed to step14 verification — those add nothing to clause checks
    but blow up prompt size + LLM latency."""
    out: Dict[int, str] = {}
    for p in c8083.get('pages', []) or []:
        pn = p.get('page_number')
        dt = (p.get('doc_type') or '').strip().lower()
        if isinstance(pn, int) and pn > 0:
            out[pn] = dt
    return out


# Page doc_types whose body is pure boilerplate for verification
# purposes — strip them from packet text. The boolean status flags
# (bl_terms_on_back / has_overleaf / blank_back) still travel on
# the packet so the verifier knows T&C was attached.
_VERIF_SKIP_DOCTYPE_SUBSTRINGS = (
    't&c',
    'terms and conditions',
    'terms & conditions',
    'conditions of carriage',
    'conditions of contract',
    'overleaf',
    'blank back',
    'blank page',
    'rear of bill of lading',
    'back of bill of lading',
)


def _is_skippable_verification_page(doc_type: str) -> bool:
    if not doc_type:
        return False
    dt = doc_type.lower()
    for needle in _VERIF_SKIP_DOCTYPE_SUBSTRINGS:
        if needle in dt:
            return True
    return False


def _compute_sections(ld: Dict, pages_by_num: Dict[int, Dict]) -> List[Dict]:
    """Compute the sub-sections inside a logical document by walking
    its all_pages list and grouping consecutive same-doc-type runs.

    Mirrors the client-side JS in classifier_server/server.py:2012-2026
    so the 8082 Summary panel can show e.g.:
       "Sections inside this document (2)
        1. Bill of Lading       page 7  · 1 pg
        2. Attached Rider       page 15 · 1 pg"

    Returns a list of {type, pages: [int, ...]} dicts.
    """
    all_pages = ld.get('all_pages') or []
    if not all_pages:
        return []
    parent_dt = ld.get('document_type') or ''
    sections: List[Dict] = []
    cur: Dict = {}
    for pn in all_pages:
        p = pages_by_num.get(pn, {}) or {}
        sec_type = p.get('doc_type') or parent_dt
        if (cur and cur.get('type') == sec_type
                and cur['pages']
                and pn == cur['pages'][-1] + 1):
            cur['pages'].append(pn)
        else:
            if cur:
                sections.append(cur)
            cur = {'type': sec_type, 'pages': [pn]}
    if cur:
        sections.append(cur)
    return sections


# Status-flag codes 8083 emits → human-readable labels (same wording
# the 8083 UI uses so 8082 stays consistent).
_BL_STATUS_FLAG_LABELS = {
    'has_overleaf':     'T&C overleaf',
    'blank_back':       'Blank back',
    'short_form':       'Short form',
    'shipped_on_board': 'Shipped on board',
    'clean_on_board':   'Clean on board',
}


def _build_bl_subtype_dict(ld: Dict) -> Dict:
    """Convert 8083's BL flags + signing capacity into the dict shape
    step14_verification expects under pkt['bl_subtype'].

    The legacy 8082 step03 emitted `bl_subtype` as a DICT with keys like
    is_house_bl, is_short_form, is_blank_back, has_terms_overleaf,
    contract_type, signing_type, issuer_type, cleanness, is_claused_bl,
    clausing_notes — and step14 does direct `bl_subtype.get('is_house_bl')`
    lookups (see step14_verification.py:3604-3771 + :11699).

    8083 emits the same information across several flat fields:
      • ld['bl_subtype']            — string title (e.g. "Multimodal …")
      • ld['bl_status_flags']       — ['has_overleaf','blank_back',…]
      • ld['bl_signing_capacity']   — {label, capacity, evidence}
      • ld['is_house_bl'], ld['is_carrier_signed']  — bools
      • ld['bl_terms_on_back'], ld['bl_blank_back'] — bools
      • ld['signing_capacity_code'] — enum

    We re-assemble those into the dict step14 wants so the legacy
    BL signing/contract/cleanness verification logic keeps working
    end-to-end. Without this, ALL BL signing/clean/short-form
    checks silently fall back to 'unknown' because the dict lookups
    return None on the string value.
    """
    flags = set(ld.get('bl_status_flags') or [])
    sig = ld.get('bl_signing_capacity') or {}
    sc_code = (ld.get('signing_capacity_code') or sig.get('capacity') or '').lower()
    # Map 8083's capacity enum → step14's signing_type lowercase string
    signing_type = ''
    if sc_code in ('carrier_signed', 'master_signed', 'carrier'):
        signing_type = 'carrier'
    elif sc_code in ('agent_for_carrier', 'agent_for_master', 'agent_for_owner'):
        signing_type = 'agent'
    elif sc_code in ('forwarder_signed', 'forwarder'):
        signing_type = 'forwarder'
    # issuer_type — derive from house vs carrier
    issuer_type = ''
    if ld.get('is_house_bl') is True:
        issuer_type = 'house'
    elif ld.get('is_carrier_signed') is True:
        issuer_type = 'carrier'
    # cleanness — 8083 doesn't have explicit cleanness, but
    # clean_on_board flag is the proxy
    cleanness = ''
    if 'clean_on_board' in flags:
        cleanness = 'clean'
    out: Dict = {
        'bl_type_description': ld.get('bl_subtype') or '',
        'is_house_bl':         ld.get('is_house_bl'),
        'is_charter_party_bl': None,
        'is_short_form':       ('short_form' in flags) or (ld.get('bl_blank_back') is True),
        'is_blank_back':       ('blank_back' in flags) or (ld.get('bl_blank_back') is True),
        'has_terms_overleaf':  ('has_overleaf' in flags) or (ld.get('bl_terms_on_back') is True),
        'shipped_on_board_status': 'shipped_on_board' if 'shipped_on_board' in flags else '',
        'signing_type':        signing_type,
        'issuer_type':         issuer_type,
        'cleanness':           cleanness,
        'is_claused_bl':       None,
        'clausing_notes':      '',
        'carrier_name':        (sig.get('evidence') or '') if signing_type == 'carrier' else '',
        'forwarder_name':      (sig.get('evidence') or '') if signing_type == 'forwarder' else '',
        'contract_type':       '',  # 8083 doesn't track this
        'form_type':           '',
    }
    return out


def _bl_status_flag_labels(ld: Dict) -> List[str]:
    """Pretty-print bl_status_flags + signing capacity + printout/copy
    info into a flat list of human-readable badge labels. Used by the
    extracted-text Summary panel + verification (step14 searches the
    list for things like 'agent for carrier').
    """
    out: List[str] = []
    bl_st = ld.get('bl_subtype')
    if bl_st: out.append(str(bl_st))
    awb_st = ld.get('awb_subtype')
    if awb_st: out.append(str(awb_st))
    for f in (ld.get('bl_status_flags') or []):
        out.append(_BL_STATUS_FLAG_LABELS.get(f, f))
    sig = ld.get('bl_signing_capacity') or ld.get('awb_signing_capacity')
    if sig and isinstance(sig, dict) and sig.get('label'):
        out.append(f"✍ {sig['label']}")
    if ld.get('is_house_bl') is True:
        out.append('House BL (forwarder)')
    elif ld.get('is_carrier_signed') is True:
        out.append('Carrier-signed (UCP art. 20)')
    pi, pt = ld.get('printout_index'), ld.get('printout_total')
    if pi and pt and pt > 1:
        out.append(f'Printout {pi}/{pt}')
    return out


def _build_packets(c8083: Dict, page_texts: Dict[int, str]) -> List[Dict]:
    """Map 8083 `logical_documents` → 8082 packet list.

    Each 8083 logical doc becomes one packet. SWIFT docs get a proper
    mt_type; matched LC-required docs and extras get mt_type='shipping'.
    The packet's `pages` list contains lightweight page dicts with
    page_number + refined_text so step06's _get_packet_refined_text
    works whether or not _PAGE_TEXT_LOOKUP is set.
    """
    # Build a page-number → 8083-page lookup so we can compute the
    # in-doc sections for each logical document (consecutive same-
    # doc-type page runs).
    pages_by_num: Dict[int, Dict] = {}
    for _p in c8083.get('pages', []) or []:
        _pn = _p.get('page_number')
        if isinstance(_pn, int) and _pn > 0:
            pages_by_num[_pn] = _p

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
        # Promote a handful of flat keys so step14_verification's
        # `pkt.get('document_number')`, `pkt.get('issued_by')` etc.
        # have data to read.
        _flat = _promote_flat_keys(doc_type, ld.get('fields') or {})
        packets.append({
            'packet_id':       next_id,
            'mt_type':         mt,
            'mt_confidence':   0.95,
            'page_numbers':    list(all_pages),
            'page_count':      len(all_pages),
            'pages':           page_dicts,
            'amendment_number': amd_no,
            'lc_reference':    _flat.get('lc_reference') or ld.get('lc_reference') or '',
            'sender_reference': ld.get('sender_reference') or '',
            'related_reference': ld.get('related_reference') or '',
            'reconciliation_method': 'passthrough',
            'was_reclassified': False,
            'text_was_corrected': False,
            'is_799_amendment': is_799_amd,
            'source_mt':       kf_mt or ('MT799' if is_799_amd else ''),
            # Flat verification keys promoted from extracted_fields
            'document_number': _flat.get('document_number', ''),
            'document_date':   _flat.get('document_date', ''),
            'document_amount': _flat.get('document_amount', ''),
            'currency':        _flat.get('currency', ''),
            'issued_by':       _flat.get('issued_by', ''),
            # Carry through 8083's extracted fields so downstream
            # verification can read stamps/signatures/endorsements
            # without re-extracting.
            'extracted_fields': ld.get('fields') or {},
            # Human-readable one-line summary the 8082 UI displays in
            # the identified-documents view + extracted-text page.
            # Built from 8083's deep_extract structured fields — single
            # source of truth, no second extraction needed.
            'document_summary': _build_doc_summary(doc_type, ld.get('fields') or {}),
            'document_type':   doc_type,
            'kind_from_8083':  kind,
            # step14 expects bl_subtype as a DICT (legacy step03 shape);
            # keep the 8083 title string under bl_subtype_label for the
            # UI badge.
            'bl_subtype':       _build_bl_subtype_dict(ld),
            'bl_subtype_label': ld.get('bl_subtype', ''),
            'bl_status_flags':  ld.get('bl_status_flags', []),
            'awb_subtype':      ld.get('awb_subtype', ''),
            'bl_signing_capacity':  ld.get('bl_signing_capacity'),
            'awb_signing_capacity': ld.get('awb_signing_capacity'),
            'is_house_bl':       ld.get('is_house_bl'),
            'is_carrier_signed': ld.get('is_carrier_signed'),
            # ── Multi-printout / copy-status / sub-section data ──
            # 8083's logical document tracks every printout of the
            # same logical doc (each copy / non-negotiable / original)
            # and the inferred sub-sections inside (e.g. BL + Attached
            # Rider). Pass them all through so the extracted-text
            # Summary panel can show the same details the 8083 UI
            # shows, AND so verification can reason about copy counts,
            # signing capacity, etc.
            'printout_index':   ld.get('printout_index'),
            'printout_total':   ld.get('printout_total'),
            'copies':           ld.get('copies') or [],
            'originals_count':  ld.get('originals_count', 0),
            'copies_count':     ld.get('copies_count', 0),
            'unknown_marker_count': ld.get('unknown_marker_count', 0),
            # Compute sub-sections inside this logical doc (same way
            # the 8083 client JS does at server.py:2012-2026) so the
            # 8082 Summary panel can show e.g. "1. Bill of Lading p7 ·
            # 2. Attached Rider p15".
            'sections':         _compute_sections(ld, pages_by_num),
            'logical_doc_id':   ld.get('logical_doc_id', ''),
            'attachments_required': ld.get('attachments_required') or [],
            'attachments_found':    ld.get('attachments_found') or [],
            # ── LC matching audit-trail — verification reads these to
            # answer "do the originals + copies counts match what the
            # LC asked for in 46A?" without re-extracting.
            'doc_ref':                ld.get('doc_ref', ''),
            'lc_requirement_index':   ld.get('lc_requirement_index', -1),
            'lc_required_originals':  ld.get('lc_required_originals'),
            'lc_required_copies':     ld.get('lc_required_copies'),
            'matched_originals_count': ld.get('matched_originals_count'),
            'matched_copies_count':   ld.get('matched_copies_count'),
            'total_instances':        ld.get('total_instances'),
            'must_show':              ld.get('must_show') or [],
            'match_reason':           ld.get('reason', ''),
            'methods':                ld.get('methods') or [],
            # BL T&C flags — also surfaced as a status_flag_labels
            # entry, but kept raw here so verification can check
            # directly (e.g. for short-form BL warnings).
            'bl_terms_on_back':       ld.get('bl_terms_on_back'),
            'bl_blank_back':          ld.get('bl_blank_back'),
            'signing_capacity_code':  ld.get('signing_capacity_code', ''),
            # Hand-y derived field: a flat human-readable list of
            # status badges (e.g. ["Blank back", "Carrier-signed (UCP
            # art. 20)", "Printout 1/3"]). step14 can search this for
            # things like 'shipped on board' / 'agent for carrier'.
            'status_flag_labels': _bl_status_flag_labels(ld),
        })
        next_id += 1
    return packets


def _build_doc_summary(doc_type: str, fields: Dict) -> str:
    """Build a one-line human-readable summary from 8083's deep_extract
    structured fields. Used by the extracted-text page and the identified-
    documents view (which both look for `document_summary` on each
    packet).

    Format: 'Label1: value1 | Label2: value2 | ...'
    Each doc type has its own preferred field set so summaries are
    consistent across docs.
    """
    if not isinstance(fields, dict) or not fields:
        return ''
    dt = (doc_type or '').lower()
    # Per-doc-type field preference. First match wins. Multiple fields
    # are joined with ' | '.
    def _v(key: str) -> str:
        v = fields.get(key)
        if isinstance(v, dict):
            # Pick the most meaningful value from a nested object
            for k in ('name', 'value', 'date', 'amount', 'number'):
                if v.get(k):
                    return str(v[k])[:80]
            return ''
        if isinstance(v, list):
            return ', '.join(str(x)[:40] for x in v[:3] if x)[:80]
        if v is None:
            return ''
        s = str(v).strip()
        return s.split('\n')[0][:120] if s else ''

    pairs = []
    def _add(label: str, *candidates: str) -> None:
        for key in candidates:
            val = _v(key)
            if val:
                pairs.append(f'{label}: {val}')
                return

    if 'bill of lading' in dt or 'sea waybill' in dt:
        _add('BL No', 'BL Number', 'B/L Number', 'Document Number')
        _add('Carrier', 'Carrier', 'Issuing Carrier')
        _add('Vessel', 'Vessel', 'Vessel Name')
        _add('From', 'Port of Loading', 'Place of Receipt')
        _add('To', 'Port of Discharge', 'Place of Delivery')
        _add('On Board', 'Shipped On Board Date', 'On Board Date')
        _add('Issued', 'Date of Issue', 'Document Date')
    elif 'airway' in dt or 'air waybill' in dt or 'courier' in dt:
        _add('AWB No', 'AWB Number', 'Document Number')
        _add('Carrier', 'Issuing Carrier', 'Carrier')
        _add('From', 'Airport of Departure')
        _add('To', 'Airport of Destination')
        _add('Issued', 'Executed On Date', 'Document Date')
    elif 'commercial invoice' in dt or 'proforma' in dt:
        _add('Invoice No', 'Invoice Number', 'Document Number')
        _add('Date', 'Invoice Date', 'Document Date')
        _add('Total', 'Invoice Total', 'Total Amount')
        _add('Currency', 'Currency')
        _add('Incoterms', 'Incoterms', 'Incoterms with Named Place')
    elif 'packing list' in dt or 'weight list' in dt:
        _add('PL No', 'Packing List Number', 'Document Number')
        _add('Date', 'Document Date', 'Issue Date')
        _add('Packages', 'Total Number of Packages', 'Total Packages')
        _add('Gross Wt', 'Total Gross Weight')
        _add('Net Wt', 'Total Net Weight')
    elif 'certificate of origin' in dt:
        _add('Cert No', 'Certificate Number', 'Document Number')
        _add('Date', 'Document Date')
        _add('Country', 'Country of Origin')
        _add('Issuer', 'Issuing Authority', 'Issuer')
    elif 'beneficiary' in dt:
        _add('Cert No', 'Certificate Number')
        _add('Date', 'Document Date', 'Issue Date')
        _add('Issuer', 'Issuer', 'Beneficiary Name')
    elif 'shipping company' in dt:
        _add('Cert No', 'Certificate Number')
        _add('Date', 'Document Date')
        _add('Vessel', 'Vessel Name', 'Vessel')
        _add('Carrier', 'Carrier', 'Issuing Carrier')
    elif 'draft' in dt or 'bill of exchange' in dt:
        _add('Draft No', 'Draft Number', 'Document Number')
        _add('Date', 'Date of Drawing', 'Document Date')
        _add('Amount', 'Total Amount in Figures', 'Amount')
        _add('Tenor', 'Tenor')
    elif 'shipment advice' in dt or 'vessel advice' in dt:
        _add('Advice No', 'Advice Number', 'Document Number')
        _add('Date', 'Advice Date', 'Document Date')
        _add('Vessel', 'Vessel Name', 'Vessel')
        _add('BL No', 'BL Number')
    elif 'documentary remittance' in dt or 'covering' in dt or 'bills schedule' in dt:
        _add('Ref', 'Letter Reference', 'Our Reference', 'Document Number')
        _add('Date', 'Date', 'Document Date')
        _add('LC No', 'LC Reference', 'Letter of Credit Number')
        _add('Amount', 'Total Amount', 'Documents Amount')
        _add('From', 'From')
        _add('To', 'To')
    else:
        # Generic fallback for other cert types
        _add('No', 'Certificate Number', 'Document Number', 'Reference Number')
        _add('Date', 'Document Date', 'Issue Date')
        _add('Issuer', 'Issuer', 'Issuing Authority')

    return ' | '.join(pairs)


def _promote_flat_keys(doc_type: str, fields: Dict) -> Dict:
    """Pick out a few flat keys (document_number, document_date,
    document_amount, currency, lc_reference, issued_by) from the
    structured extracted_fields. step14_verification + the UI both read
    these directly off the packet (`pkt.get('document_number')`), so
    they need to exist at the top level, not just nested under
    `extracted_fields`.

    Doc-type agnostic — works for any document the classifier emits,
    including new types added later. Strategy: each flat key has a
    priority list of CANDIDATE FIELD NAMES (specific → generic). We
    try them in order and use the first scalar match. Then we add a
    REGEX FALLBACK that scans every remaining field name (e.g. 'AWB
    Number', 'B/L No.', 'Certificate Ref.') so an unknown doc type
    with reasonable field naming still surfaces a number/date.
    """
    if not isinstance(fields, dict):
        return {}

    def _scalar(v) -> str:
        if v is None or isinstance(v, (list, dict)):
            return ''
        s = str(v).strip()
        return s.split('\n')[0][:200] if s else ''

    def _pick_by_name(*names: str) -> str:
        # Case-insensitive lookup against the field name AS IT APPEARS
        # in extracted_fields (preserves the LLM's chosen label).
        lower_map = {k.strip().lower(): k for k in fields.keys()}
        for n in names:
            k = lower_map.get(n.strip().lower())
            if k is not None:
                s = _scalar(fields[k])
                if s:
                    return s
        return ''

    def _pick_by_regex(*patterns: str) -> str:
        # Scan every key looking for a match. First match wins.
        compiled = [_re_adapter.compile(p, _re_adapter.IGNORECASE) for p in patterns]
        for k, v in fields.items():
            if isinstance(v, (list, dict)):
                continue
            for pat in compiled:
                if pat.search(k):
                    s = _scalar(v)
                    if s:
                        return s
        return ''

    out: Dict[str, str] = {}
    # document_number — try canonical names, then a regex over any
    # field name containing "number"/"no"/"ref"/"id" (but not "lc no"
    # which we surface separately as lc_reference).
    n = (_pick_by_name(
            'BL Number', 'B/L Number', 'BL No', 'B/L No',
            'AWB Number', 'AWB No',
            'Invoice Number', 'Invoice No',
            'Packing List Number',
            'Certificate Number', 'Cert Number',
            'Draft Number',
            'Advice Number',
            'Letter Reference', 'Our Reference', 'Sender Reference',
            'Document Number', 'Reference Number',
         )
         or _pick_by_regex(
            r'^(?!lc\s+).*\b(number|no\.?|reference|ref\.?|id)\b',
         ))
    if n:
        out['document_number'] = n
    # document_date — any field with 'date' / 'issued on' / 'drawn on'
    d = (_pick_by_name(
            'Document Date', 'Date of Issue', 'Issue Date', 'Issued Date',
            'Invoice Date', 'Date', 'Executed On Date',
            'Date of Drawing', 'Advice Date',
         )
         or _pick_by_regex(r'\b(date|issued|executed)\b'))
    if d:
        out['document_date'] = d
    # document_amount
    a = (_pick_by_name(
            'Invoice Total', 'Total Amount', 'Total Amount in Figures',
            'Amount', 'Documents Amount', 'Total',
         )
         or _pick_by_regex(r'^(total|amount|invoice\s+total)\b'))
    if a:
        out['document_amount'] = a
    cur = _pick_by_name('Currency') or _pick_by_regex(r'\bcurrency\b')
    if cur:
        out['currency'] = cur
    lc = (_pick_by_name(
            'LC Reference', 'Letter of Credit Number', 'LC Number',
            'Documentary Credit Number', 'LC No',
         )
         or _pick_by_regex(r'\b(letter\s+of\s+credit|documentary\s+credit|\blc\s+(no|number|ref))\b'))
    if lc:
        out['lc_reference'] = lc
    # issued_by — try doc-aware canonical names, then anything that
    # looks like an issuer/issuing authority/carrier/etc.
    iss = (_pick_by_name(
            'Carrier', 'Issuing Carrier',
            'Issuing Authority', 'Certifying Authority',
            'Insurance Company',
            'Inspector / Surveyor / Lab Name', 'Inspection Company',
            'Surveyor', 'Lab Name', 'Testing Laboratory',
            'Pest-Control Operator Name & Address',
            'Seller', 'Exporter', 'Shipper',
            'Issuer', 'Issuing Bank', 'Beneficiary Name', 'Beneficiary',
         )
         or _pick_by_regex(r'\b(issuer|issuing|carrier|certifier|surveyor|inspector|laboratory)\b'))
    if iss:
        out['issued_by'] = iss
    return out


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
    page_doctypes = _build_page_doctypes(c8083)
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
        # /api/result/{job_id} and several other consumers read
        # `total_pages` from step01 to populate the result-screen stat
        # box; keep both keys in sync so the legacy UI shows the real
        # count instead of 0.
        'total_pages': c8083.get('total_pages', 0) or len(c8083.get('pages') or []),
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
    # Downstream code (server.py:302, 377, 413, 457, 1155, 1213, 1629,
    # 1638, 1746, 1771, 2222-2242, etc.) reads packets via:
    #   pkt.get('document_type') → routes to MT700/MT707/shipping
    #   pkt.get('page_numbers')
    #   pkt.get('stamps')   / pkt.get('signatures')  → verification
    #   pkt.get('cleaned_text') / pkt.get('text')    → context display
    # We surface all of these from 8083's logical doc + page data so the
    # /api/regenerate-final-lc + /api/rerun-classification endpoints
    # work against the adapter output exactly like they would against
    # the legacy step03 output.
    step03_packets = []
    for pkt in packets:
        full_text = '\n'.join(
            (page_texts.get(pn, '') for pn in pkt['page_numbers'])
        )
        step03_packets.append({
            'packet_id':       pkt['packet_id'],
            'page_numbers':    pkt['page_numbers'],
            'page_count':      pkt['page_count'],
            'pages':           pkt['pages'],
            # Lowercase form so the existing `dt in ('lc','amendment',...)`
            # checks match. Use 'amendment' for any "Amendment N" so
            # `'amend' in dt` resolves true (step03 legacy convention).
            'document_type':   _legacy_document_type(pkt.get('document_type', ''),
                                                      pkt.get('mt_type', '')),
            'mt_type':         pkt['mt_type'],
            'boundary_confidence': 0.95,
            'stamps':          pkt['stamps'],
            'signatures':      pkt['signatures'],
            'text':            full_text,
            'cleaned_text':    full_text,
            'refined_text':    full_text,
            'source_mt':       pkt.get('source_mt', ''),
            'is_799_amendment': pkt.get('is_799_amendment', False),
            'extracted_fields': pkt.get('extracted_fields', {}),
            'document_summary': pkt.get('document_summary', ''),
            'bl_subtype':      pkt.get('bl_subtype', ''),
            'bl_status_flags': pkt.get('bl_status_flags', []),
            'awb_subtype':     pkt.get('awb_subtype', ''),
        })
    step03 = {
        'packets': step03_packets,
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
    # `document_summary` powers the identified-documents view + the
    # extracted-text page summary panel.
    #
    # CRITICAL — verification (step14) reads the document body via
    # `_pkt_text(pkt)` which checks refined_text → cleaned_text →
    # text → raw_text. If NONE of those keys are present the
    # verifier has no evidence and every clause comes back as
    # REVIEW or FAIL ("doc not found / cannot verify"). The old
    # legacy step08 path always emitted these — we must too.
    _shipping_packets = []
    for pkt in packets:
        if pkt['mt_type'] != 'shipping':
            continue
        # Concatenate the per-page cleaned text into a single body for
        # the packet, in page order. This mirrors what _build_packets
        # does for step03_packets at lines 580-582 above.
        #
        # IMPORTANT — filter out T&C / overleaf / blank-back / Conditions-
        # of-Carriage pages. Those carry boilerplate that step14
        # verification doesn't need but that costs us thousands of
        # tokens per LLM call (e.g. a BL packet of 22K chars where 18K
        # is reverse-side terms). The presence of T&C/overleaf is
        # still flagged on the packet (bl_terms_on_back, bl_blank_back,
        # bl_status_flags has_overleaf/blank_back) so the verifier
        # knows it was attached, just without re-sending the boilerplate.
        _kept_pages: List[int] = []
        _skipped_tc_pages: List[int] = []
        for pn in pkt['page_numbers']:
            _dt = page_doctypes.get(pn, '')
            if _is_skippable_verification_page(_dt):
                _skipped_tc_pages.append(pn)
            else:
                _kept_pages.append(pn)
        _doc_body = '\n'.join(
            (page_texts.get(pn, '') for pn in _kept_pages)
        )
        if _skipped_tc_pages:
            _doc_body += (
                f"\n\n[Note: {len(_skipped_tc_pages)} terms-and-conditions / "
                f"overleaf page(s) attached on pages "
                f"{', '.join(str(p) for p in _skipped_tc_pages)} — "
                f"boilerplate not included in body. Status flags still "
                f"apply: see audit header above.]"
            )
        # ── Audit header injection ───────────────────────────────
        # step14 reads the packet body as the entire evidence for
        # each verification clause. The new 8083 fields (status
        # flags, signing capacity, originals/copies counts, sections,
        # attachments, must_show LC phrases, match reason) are NOT
        # in the OCR text itself — they're metadata that the
        # classifier computed. Prepending them as a small audit
        # header makes them visible to the LLM verifier without
        # any step14 code changes. The verifier treats this as
        # extra context, then the doc text follows.
        _audit_lines = []
        _audit_lines.append(f"DOCUMENT TYPE: {pkt.get('document_type', '')}")
        _bl_lbl = pkt.get('bl_subtype_label', '')
        if _bl_lbl:
            _audit_lines.append(f"DOCUMENT SUBTYPE: {_bl_lbl}")
        _flags = pkt.get('status_flag_labels') or []
        if _flags:
            _audit_lines.append(f"STATUS FLAGS: {' · '.join(_flags)}")
        _sig = pkt.get('bl_signing_capacity') or pkt.get('awb_signing_capacity') or {}
        if isinstance(_sig, dict) and _sig.get('label'):
            _ev = _sig.get('evidence') or ''
            _audit_lines.append(
                f"SIGNED AS: {_sig['label']}" + (f" (evidence: {_ev[:120]})" if _ev else '')
            )
        _orig_c = pkt.get('originals_count', 0)
        _copy_c = pkt.get('copies_count', 0)
        _lc_o   = pkt.get('lc_required_originals')
        _lc_c   = pkt.get('lc_required_copies')
        if _orig_c or _copy_c or _lc_o is not None or _lc_c is not None:
            _req = f"LC requires: {_lc_o if _lc_o is not None else '?'} originals + {_lc_c if _lc_c is not None else '?'} copies"
            _sub = f"Submitted: {_orig_c} originals + {_copy_c} copies"
            _audit_lines.append(f"COPY COUNTS: {_sub} ({_req})")
        _copies = pkt.get('copies') or []
        if _copies:
            _audit_lines.append("PRINTOUTS:")
            for i, _cp in enumerate(_copies, 1):
                _lbl = _cp.get('copy_label') or 'UNMARKED'
                _cat = _cp.get('copy_category') or ''
                _pgs = ','.join(str(_p) for _p in (_cp.get('pages') or []))
                _audit_lines.append(f"  {i}. {_lbl} [{_cat}] — pages {_pgs}")
        _secs = pkt.get('sections') or []
        if len(_secs) > 1:
            _audit_lines.append("SECTIONS:")
            for i, _s in enumerate(_secs, 1):
                _t = _s.get('type', 'Section')
                _pgs = ','.join(str(_p) for _p in (_s.get('pages') or []))
                _audit_lines.append(f"  {i}. {_t} — page(s) {_pgs}")
        _att_req = pkt.get('attachments_required') or []
        _att_fnd = pkt.get('attachments_found') or []
        if _att_req:
            _audit_lines.append(f"ATTACHMENTS REQUIRED: {' · '.join(str(a) for a in _att_req)}")
        if _att_fnd:
            _audit_lines.append(f"ATTACHMENTS FOUND: {' · '.join(str(a) for a in _att_fnd)}")
        # P198h6 — REMOVED 'LC MUST-SHOW PHRASES' audit-header
        # injection. The phrases (e.g. "Importer's NTN No. 1550365-8
        # should appear" or "House bill of lading acceptable") were
        # being misread by the verifier as proof of presence on the
        # document — causing both deterministic detectors (P198gz36)
        # and the LLM verifier to PASS conditions when the required
        # value only appeared in the audit header, not in the doc
        # body. The same requirement reaches the verifier via
        # condition_text on each row, so removing this section
        # eliminates the false positives without losing context.
        # _must = pkt.get('must_show') or []
        # if _must:
        #     _audit_lines.append("LC MUST-SHOW PHRASES (each must appear in this document):")
        #     for i, _m in enumerate(_must, 1):
        #         _audit_lines.append(f"  {i}. {_m}")
        _reason = pkt.get('match_reason', '')
        if _reason:
            _audit_lines.append(f"CLASSIFIER MATCH REASON: {_reason}")
        _doc_ref = pkt.get('doc_ref', '')
        if _doc_ref:
            _audit_lines.append(f"DOC REF: {_doc_ref}")

        # P198gz — Surface structured extracted_fields (scalars, nested
        # dicts, tables) into the audit header so the LLM verifier sees
        # ALL data from EVERY merged page of the document — not just
        # the flat OCR text. The extracted_fields are merged across all
        # member pages by 8083's deep_extract, so this gives the
        # verifier the full set of containers / parties / marks / etc.
        try:
            _xf = pkt.get('extracted_fields') or {}
            if isinstance(_xf, dict) and _xf:
                _XF_SKIP = {
                    '_extract_source', 'Extract Source', 'extract_source',
                    '_deep_extract_error',
                    '_doc_type_override_from_fields',
                    '_promoted_from_extras_after_reconcile',
                }

                def _scalar_str(v):
                    if v is None:
                        return ''
                    s = str(v).strip()
                    return s.split('\n')[0][:240] if s else ''

                _scalars = []
                _objects = []
                _arrays  = []
                for _k, _v in _xf.items():
                    if not _k or _k.startswith('_') or _k in _XF_SKIP:
                        continue
                    if isinstance(_v, list):
                        if _v:
                            _arrays.append((_k, _v))
                    elif isinstance(_v, dict):
                        if _v:
                            _objects.append((_k, _v))
                    elif _v is not None and str(_v).strip() != '':
                        _scalars.append((_k, _v))

                if _scalars or _objects or _arrays:
                    _audit_lines.append("EXTRACTED FIELDS (deep_extract — merged across ALL member pages):")
                if _scalars:
                    for _k, _v in _scalars:
                        _audit_lines.append(f"  • {_k}: {_scalar_str(_v)}")
                for _k, _v in _objects:
                    _rows = []
                    for _kk, _vv in _v.items():
                        if not _kk or str(_kk).startswith('_'):
                            continue
                        _s = _scalar_str(_vv) if not isinstance(_vv, (list, dict)) else json.dumps(_vv, ensure_ascii=False)[:240]
                        if _s:
                            _rows.append(f"      - {_kk}: {_s}")
                    if _rows:
                        _audit_lines.append(f"  • {_k}:")
                        _audit_lines.extend(_rows)
                for _k, _arr in _arrays:
                    _first = _arr[0]
                    if isinstance(_first, dict):
                        _cols = []
                        for _row in _arr:
                            for _ck in _row.keys():
                                if _ck and not str(_ck).startswith('_') and _ck not in _cols:
                                    _cols.append(_ck)
                        _audit_lines.append(f"  • {_k} ({len(_arr)} rows; columns: {', '.join(_cols)}):")
                        for _i, _row in enumerate(_arr, 1):
                            _cells = []
                            for _c in _cols:
                                _val = _row.get(_c)
                                if _val is None or (isinstance(_val, str) and not _val.strip()):
                                    continue
                                if isinstance(_val, (list, dict)):
                                    _cells.append(f"{_c}={json.dumps(_val, ensure_ascii=False)[:120]}")
                                else:
                                    _cells.append(f"{_c}={_scalar_str(_val)}")
                            if _cells:
                                _audit_lines.append(f"      [{_i}] " + ' | '.join(_cells))
                    else:
                        _items = []
                        for _v in _arr[:30]:
                            if isinstance(_v, (list, dict)):
                                _items.append(json.dumps(_v, ensure_ascii=False)[:120])
                            else:
                                _items.append(_scalar_str(_v))
                        _audit_lines.append(f"  • {_k}: {', '.join(_x for _x in _items if _x)}")
        except Exception:
            # Never let structured-field formatting crash verification.
            pass

        if _audit_lines:
            _full_text = (
                "=== DOCUMENT AUDIT (from classifier) ===\n"
                + '\n'.join(_audit_lines)
                + "\n=== END AUDIT — DOCUMENT TEXT FOLLOWS ===\n\n"
                + _doc_body
            )
        else:
            _full_text = _doc_body
        _shipping_packets.append({
            'packet_id':     pkt['packet_id'],
            'document_type': pkt.get('document_type', ''),
            'page_numbers':  pkt['page_numbers'],
            'mt_type':       pkt['mt_type'],
            'stamps':        pkt['stamps'],
            'signatures':    pkt['signatures'],
            'extracted_fields': pkt.get('extracted_fields', {}),
            'document_summary': pkt.get('document_summary', ''),
            'bl_subtype':       pkt.get('bl_subtype', {}),
            'bl_subtype_label': pkt.get('bl_subtype_label', ''),
            'bl_status_flags':  pkt.get('bl_status_flags', []),
            'awb_subtype':      pkt.get('awb_subtype', ''),
            # Signing capacity, house BL, carrier-signed, printout
            # index/total, copies list, sections, attachments — all
            # the things the 8083 Summary panel shows. Verification
            # (step14) can also reason about copy counts vs LC's
            # "in N originals + M copies" requirement.
            'bl_signing_capacity':  pkt.get('bl_signing_capacity'),
            'awb_signing_capacity': pkt.get('awb_signing_capacity'),
            'is_house_bl':       pkt.get('is_house_bl'),
            'is_carrier_signed': pkt.get('is_carrier_signed'),
            'printout_index':    pkt.get('printout_index'),
            'printout_total':    pkt.get('printout_total'),
            'copies':            pkt.get('copies') or [],
            'originals_count':   pkt.get('originals_count', 0),
            'copies_count':      pkt.get('copies_count', 0),
            'unknown_marker_count': pkt.get('unknown_marker_count', 0),
            'sections':          pkt.get('sections') or [],
            'logical_doc_id':    pkt.get('logical_doc_id', ''),
            'attachments_required': pkt.get('attachments_required') or [],
            'attachments_found':    pkt.get('attachments_found') or [],
            'status_flag_labels':   pkt.get('status_flag_labels') or [],
            # LC-matching audit trail (originals + copies vs required)
            'doc_ref':                pkt.get('doc_ref', ''),
            'lc_requirement_index':   pkt.get('lc_requirement_index', -1),
            'lc_required_originals':  pkt.get('lc_required_originals'),
            'lc_required_copies':     pkt.get('lc_required_copies'),
            'matched_originals_count': pkt.get('matched_originals_count'),
            'matched_copies_count':   pkt.get('matched_copies_count'),
            'total_instances':        pkt.get('total_instances'),
            'must_show':              pkt.get('must_show') or [],
            'match_reason':           pkt.get('match_reason', ''),
            'methods':                pkt.get('methods') or [],
            'bl_terms_on_back':       pkt.get('bl_terms_on_back'),
            'bl_blank_back':          pkt.get('bl_blank_back'),
            'signing_capacity_code':  pkt.get('signing_capacity_code', ''),
            # ── Verification text (REQUIRED by step14._pkt_text) ──
            'text':          _full_text,
            'cleaned_text':  _full_text,
            'refined_text':  _full_text,
            'raw_text':      _full_text,
            # step14._pkt_images reads this; we keep it empty because
            # the LLM call deliberately skips the image payload
            # (see step14:5205-5207) — text-only verification is
            # both faster and stays under the 72B's 16k token limit.
            'page_image_paths': [],
            # Flat verification keys (step14 reads these directly)
            'document_number': pkt.get('document_number', ''),
            'document_date':   pkt.get('document_date', ''),
            'document_amount': pkt.get('document_amount', ''),
            'currency':        pkt.get('currency', ''),
            'issued_by':       pkt.get('issued_by', ''),
            'lc_reference':    pkt.get('lc_reference', ''),
        })
    step08 = {
        'classified_packets': _shipping_packets,
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
