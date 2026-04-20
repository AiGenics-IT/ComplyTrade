"""Dry-run test of the post-check overrides for the two stubborn FAIL cases."""
import re
import sys

sys.path.insert(0, 'd:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final')

from steps.step14_verification import _normalize_id


# Simulate the post-check logic inline so we don't need the full _call_vlm path

def _simulate_post_checks(condition_text, parsed_findings, document_text,
                          unified_summary, bl_subtype):
    """Replicates the P133-P144 post-check cascade on the LLM's returned verdict."""
    parsed = {
        "compliance": "fail",
        "verdict": "FAIL",
        "findings": parsed_findings,
        "result": parsed_findings[:200],
    }

    # --- P141 / P143 / P144 universal override (the critical one) ---
    _comp_final = str(parsed.get("compliance", "")).lower().strip()
    _fin_final = str(parsed.get("findings", "")).upper()
    _NEG_PHRASES = (
        'NOT FOUND', 'NOT PRESENT', 'DOES NOT CONTAIN',
        'DOES NOT SHOW', 'DOES NOT INCLUDE', 'DOESN\'T SHOW',
        'NOT APPEAR', 'NOT DISPLAYED', 'NOT INCLUDED',
        'NOT REFERENCED', 'NOT QUOTED', 'NOT MENTIONED',
        'NOT STATED', 'IS MISSING', 'CANNOT FIND', 'CAN NOT FIND',
        'IS NOT SHOWN', 'NOT LISTED', 'NOT PRESENTED',
        'NO MATCH', 'DOES NOT MATCH',
    )
    if not (
        _comp_final in ('fail', 'not_complied', 'non_compliant', 'discrepant')
        and any(p in _fin_final for p in _NEG_PHRASES)
    ):
        return parsed, "no trigger"

    _cond_tokens = []
    for _src in (condition_text or '', parsed.get('findings', '') or ''):
        for _m in re.finditer(r"[\'\"“”‘’]([^\'\"“”‘’]{3,120})[\'\"“”‘’]", _src):
            _cond_tokens.append(_m.group(1))
    for _m in re.finditer(
        r'[A-Z0-9][A-Z0-9/\-._]{5,}[A-Z0-9]',
        condition_text or '', flags=re.IGNORECASE,
    ):
        _tok = _m.group(0)
        if re.search(r'\d', _tok):
            _cond_tokens.append(_tok)
    for _m in re.finditer(
        r'\b(?:[A-Z][A-Za-z\-]+(?:\s+(?:AND|&|OF|THE|AL|DE|DU|LA|EL))?\s+){1,6}[A-Z][A-Za-z\-]+(?:\s+(?:LTD|LIMITED|PLC|INC|CO|PVT|CORP|LLC|LLP))?',
        condition_text or '',
    ):
        _chunk = _m.group(0).strip()
        if len(_chunk) >= 6 and _chunk.upper() != _chunk:
            _cond_tokens.append(_chunk)

    _blob_raw = (document_text or '') + ' ' + str(unified_summary or '') + ' ' + str(bl_subtype or '')
    _blob_norm = _normalize_id(_blob_raw)
    _blob_upper = _blob_raw.upper()

    def _token_in_blob(tok):
        tok = tok.strip(' .,:\'""')
        if not tok or len(tok) < 3:
            return False
        if re.search(r'\d', tok):
            _n = _normalize_id(tok)
            return len(_n) >= 4 and _n in _blob_norm
        _flat = re.sub(r'\s+', ' ', tok.upper()).strip()
        _blob_flat = re.sub(r'\s+', ' ', _blob_upper)
        if _flat in _blob_flat:
            return True
        _words = [w for w in _flat.split() if len(w) >= 3]
        return _words and all(w in _blob_flat for w in _words)

    _hit_tok = None
    for _t in _cond_tokens:
        if _token_in_blob(_t):
            _hit_tok = _t
            break

    if not _hit_tok:
        _finding_caps = re.findall(
            r'\b([A-Z][A-Z\-]{2,}(?:\s+[A-Z][A-Z\-0-9]*){1,5})\b',
            parsed.get('findings', '') or '',
        )
        _STOPWORDS = {'NOT FOUND', 'NOT PRESENT', 'DOES NOT', 'DOES NOT SHOW',
                       'DOES NOT CONTAIN', 'DOES NOT INCLUDE', 'NOT APPEAR',
                       'NOT DISPLAYED', 'NOT INCLUDED', 'IS MISSING',
                       'NOT QUOTED', 'CANNOT FIND', 'NO MATCH',
                       'THE DOCUMENT', 'THE BL', 'THE INVOICE', 'THE CONSIGNEE',
                       'THE POLICY', 'THE REQUIRED'}
        for _phrase in _finding_caps:
            _ph = _phrase.strip()
            if any(sw in _ph for sw in _STOPWORDS):
                continue
            if len(_ph) < 6:
                continue
            if _token_in_blob(_ph):
                _hit_tok = _ph
                break

    if not _hit_tok:
        _BANK_KEYWORDS = [
            'AL HABIB', 'ALHABIB', 'AL-HABIB',
            'ALFALAH', 'AL FALAH', 'AL-FALAH',
            'MEEZAN', 'FAYSAL', 'ASKARI',
            'UBL', 'HBL', 'MCB', 'NBP', 'ABL', 'BAF', 'JS BANK',
            'STANDARD CHARTERED', 'CITIBANK', 'HSBC',
            'MIZUHO', 'MUFG',
        ]
        _cond_upper = (condition_text or '').upper()
        for _bkw in _BANK_KEYWORDS:
            if _bkw in _cond_upper:
                _bkw_flat = re.sub(r'\s+', ' ', _bkw)
                _blob_flat = re.sub(r'\s+', ' ', _blob_upper)
                if _bkw_flat in _blob_flat:
                    _hit_tok = _bkw
                    break

    if _hit_tok:
        parsed["compliance"] = "pass"
        parsed["verdict"] = "PASS"
        parsed["findings"] = f"Matched '{_hit_tok}' in evidence. (P141 override)"
        return parsed, f"override fired on '{_hit_tok}'"
    return parsed, "override did not fire"


# ═══════════════════════════════════════════════════════════════════════
# TEST CASE 1: Consignee
# ═══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("CASE 1 — Consignee 'TO ORDER OF BANK AL HABIB'")
print("=" * 70)

case1_condition = "Bill of lading must be made out to the order of Bank Al Habib Ltd., Karachi, Pakistan."
case1_finding = "CONSIGNEE DOES NOT SHOW 'BANK AL HABIB LTD'"
case1_doc_text = """
MAERSK
BILL OF LADING FOR OCEAN TRANSPORT
B/L No. 720629555
Shipper: SAUDI BASIC INDUSTRIES CORPORATION (SABIC)
Consignee (Negotiable only if consigned "to order" ...)
TO ORDER OF:
BANK AL HABIB LTD.,
KARACHI
Notify Party: H.SHEIKH NOOR-UD-DIN AND SONS (PVT) LTD.
"""
case1_summary = {
    "consignee": "TO ORDER OF: BANK AL HABIB LTD., KARACHI",
    "parties_found": [
        {"role": "consignee", "name": "TO ORDER OF BANK AL HABIB LTD", "raw": "TO ORDER OF: BANK AL HABIB LTD., KARACHI"},
    ],
}

result1, reason1 = _simulate_post_checks(
    case1_condition, case1_finding, case1_doc_text, case1_summary, {}
)
print(f"  Initial verdict: FAIL")
print(f"  Finding: {case1_finding}")
print(f"  After post-check: {result1['compliance'].upper()}")
print(f"  Reason: {reason1}")
print(f"  {'✓ PASS — override fired correctly' if result1['compliance'] == 'pass' else '✗ STILL FAIL — bug!'}")

# ═══════════════════════════════════════════════════════════════════════
# TEST CASE 2: Policy number (OCR O↔0)
# ═══════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("CASE 2 — Policy 2023008MIPD000453 (OCR letter-O vs digit-0)")
print("=" * 70)

case2_condition = "Shipment Advice must reference Open Policy No. 2023008MIPD000453."
case2_finding = "POLICY 2023008MIPD000453 NOT FOUND IN DOCUMENT"
case2_doc_text = """
Shipment Advice DD:16.02.2025
Shipment No: 9246193
TO UBL INSURERS LIMITED, LAHORE PAKISTAN
L/C No:0007LC55189/2025DD;250103
TO ORDER OF: BANK AL HABIB LTD., KARACHI
OPEN POLICY NO.2023008MIPDO00453
NAME OF L/C ISSUING BANK (BANK AL HABIB LIMITED, PAKISTAN).
"""
case2_summary = {
    "open_policy_reference": "2023008MIPDO00453",
    "references_found": [
        {"role": "open_policy_reference", "value": "2023008MIPDO00453", "raw": "OPEN POLICY NO.2023008MIPDO00453"},
    ],
}

result2, reason2 = _simulate_post_checks(
    case2_condition, case2_finding, case2_doc_text, case2_summary, {}
)
print(f"  Initial verdict: FAIL")
print(f"  Finding: {case2_finding}")
print(f"  After post-check: {result2['compliance'].upper()}")
print(f"  Reason: {reason2}")
print(f"  {'✓ PASS — override fired correctly' if result2['compliance'] == 'pass' else '✗ STILL FAIL — bug!'}")

# ═══════════════════════════════════════════════════════════════════════
# TEST CASE 3: Policy number with other phrasing (cover note ↔ open policy)
# ═══════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("CASE 3 — Policy with variant 11/0000118/1024/0-0")
print("=" * 70)

case3_condition = "Shipment Advice must reference Policy No. 11/0000118/1024/0-0."
case3_finding = "Policy number 11/0000118/1024/0-0 not found in document text"
case3_doc_text = "cover note reference: NO.11/0000118/1024/0-0"
case3_summary = {
    "cover_note_reference": "NO.11/0000118/1024/0-0",
}

result3, reason3 = _simulate_post_checks(
    case3_condition, case3_finding, case3_doc_text, case3_summary, {}
)
print(f"  Initial verdict: FAIL")
print(f"  Finding: {case3_finding}")
print(f"  After post-check: {result3['compliance'].upper()}")
print(f"  Reason: {reason3}")
print(f"  {'✓ PASS — override fired correctly' if result3['compliance'] == 'pass' else '✗ STILL FAIL — bug!'}")

# ═══════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
_all_pass = all(r['compliance'] == 'pass' for r in [result1, result2, result3])
print(f"OVERALL: {'✓ ALL OVERRIDES FIRING CORRECTLY' if _all_pass else '✗ BUG — at least one override failed'}")
print("=" * 70)
