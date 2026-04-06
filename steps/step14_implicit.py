"""
Step 14b — VLM-Based LC Key Term Verification
===============================================
Verifies LC key term fields against shipping documents using the Qwen VLM.
Each check has a detailed, rule-specific prompt that encodes the exact
trade finance logic the VLM must follow.

Unlike Step 14 (which verifies decomposed F46A/F47A conditions), this module
handles the 15 LC key term fields (F31C, F31D, F32B, etc.) with specialized
prompts for each check type.

CHECKS IMPLEMENTED (all configurable via checks config):
    1. Date of Issue (F31C) — docs dated on/after LC issuance
    2. Applicable Rules (F40E) — informational, no check
    3. Expiry Date (F31D) — presentation before LC expiry
    4. Applicant (F50) — reserved
    5. Beneficiary (F59) — reserved
    6. Amount & Currency (F32B) — overdrawn, short, partial, draft, currency
    7. Bank (F52A) — reserved
    8. Draft/Tenor (F42C) — sight vs usance
    9. Drawee (F42A) — reserved
    10. Partial Shipment (F43P) — enforced via amount checks
    11. Transshipment (F43T) — BL transshipment check
    12. Port of Loading (F44E) — BL port match
    13. Port of Discharge (F44F) — BL port match
    14. Latest Shipment Date (F44C) — BL on-board vs deadline
    15. Presentation Period (F48) — presentation date vs shipment + period

DOCUMENT DEDUPLICATION:
    Only ONE original per unique (doc_type, doc_number) pair is checked.

AI MODEL: Qwen VLM at QWEN_VLM_URL (from config/settings.py)
"""

import re
import json
import os
import time
import base64
import traceback
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from config.settings import QWEN_VLM_URL, QWEN_VLM_MODEL, MAX_CONCURRENT_VLM, VLM_TIMEOUT
except ImportError:
    QWEN_VLM_URL = "http://10.20.10.3:8000/v1/chat/completions"
    QWEN_VLM_MODEL = ""
    MAX_CONCURRENT_VLM = 4
    VLM_TIMEOUT = 600

try:
    import httpx
except ImportError:
    httpx = None


# ══════════════════════════════════════════════════════════════
# CHECKS CONFIGURATION
# ══════════════════════════════════════════════════════════════

DEFAULT_CHECKS_CONFIG = {
    "date_of_issue": {
        "enabled": True,
        "name": "Date of Issue (F31C)",
        "description": "All documents must be dated on or after LC issuance date. Exceptions: analysis/inspection certificates, COO, packing lists, beneficiary draft/invoice.",
        "category": "dates",
        "severity": "hard",
    },
    "applicable_rules": {
        "enabled": False,
        "name": "Applicable Rules (F40E)",
        "description": "UCP version — informational only, no document check required.",
        "category": "informational",
        "severity": "info",
    },
    "lc_expiry": {
        "enabled": True,
        "name": "Date/Place of Expiry (F31D)",
        "description": "Documents must be presented before LC expiry. Check covering schedule presentation date.",
        "category": "dates",
        "severity": "hard",
    },
    "applicant": {
        "enabled": False,
        "name": "Applicant (F50)",
        "description": "Reserved for future checks.",
        "category": "parties",
        "severity": "info",
    },
    "beneficiary": {
        "enabled": False,
        "name": "Beneficiary (F59)",
        "description": "Reserved for future checks.",
        "category": "parties",
        "severity": "info",
    },
    "amount_currency": {
        "enabled": True,
        "name": "Amount & Currency (F32B)",
        "description": "Check overdrawn, short shipment, partial shipment, cover schedule, invoice totals, draft match, currency consistency.",
        "category": "financial",
        "severity": "hard",
    },
    "bank": {
        "enabled": False,
        "name": "Issuing Bank (F52A)",
        "description": "Reserved for future checks.",
        "category": "parties",
        "severity": "info",
    },
    "draft_tenor": {
        "enabled": True,
        "name": "Draft / Tenor (F42C)",
        "description": "Verify draft says 'At Sight' for sight LC, or correct tenor for usance. Draft amount must match invoice total.",
        "category": "financial",
        "severity": "hard",
    },
    "drawee": {
        "enabled": False,
        "name": "Drawee (F42A)",
        "description": "Reserved for future checks.",
        "category": "parties",
        "severity": "info",
    },
    "partial_shipment": {
        "enabled": True,
        "name": "Partial Shipment (F43P)",
        "description": "If prohibited, all invoices must relate to same shipment and total must match LC quantity.",
        "category": "shipping",
        "severity": "hard",
    },
    "transshipment": {
        "enabled": True,
        "name": "Transshipment (F43T)",
        "description": "If prohibited, BL must not indicate transshipment.",
        "category": "shipping",
        "severity": "hard",
    },
    "port_of_loading": {
        "enabled": True,
        "name": "Port of Loading (F44E)",
        "description": "BL port of loading must match LC requirement.",
        "category": "shipping",
        "severity": "hard",
    },
    "port_of_discharge": {
        "enabled": True,
        "name": "Port of Discharge (F44F)",
        "description": "BL port of discharge must match LC requirement.",
        "category": "shipping",
        "severity": "hard",
    },
    "latest_shipment": {
        "enabled": True,
        "name": "Latest Date of Shipment (F44C)",
        "description": "BL shipped-on-board date must be on or before latest shipment date.",
        "category": "dates",
        "severity": "hard",
    },
    "presentation_period": {
        "enabled": True,
        "name": "Period for Presentation (F48)",
        "description": "Documents must be presented within stipulated period after shipment. Default 21 days per UCP 600.",
        "category": "dates",
        "severity": "hard",
    },

    # ── Document-Level Checks (F46A) ──
    "doc_bill_of_lading": {
        "enabled": True,
        "name": "Bill of Lading Verification",
        "description": "Full set, clean on board, consignee, notify party, freight status, shipped on board notation, carrier/agent signing, port marks. UCP 600 Art 20.",
        "category": "documents",
        "severity": "hard",
    },
    "doc_commercial_invoice": {
        "enabled": True,
        "name": "Commercial Invoice Verification",
        "description": "Goods description match, quantity, unit price, total amount, HS code, NTN, beneficiary name, applicant name, LC number. UCP 600 Art 18.",
        "category": "documents",
        "severity": "hard",
    },
    "doc_insurance": {
        "enabled": True,
        "name": "Insurance Document Verification",
        "description": "Coverage amount (min 110% CIF/CIP), currency match, risks covered, effective date on/before shipment, Institute Cargo Clauses. UCP 600 Art 28.",
        "category": "documents",
        "severity": "hard",
    },
    "doc_draft": {
        "enabled": True,
        "name": "Draft / Bill of Exchange Verification",
        "description": "Amount, drawee, tenor (at sight / usance), drawn on correct bank, signed by beneficiary. UCP 600 Art 14.",
        "category": "documents",
        "severity": "hard",
    },
    "doc_certificate_of_origin": {
        "enabled": True,
        "name": "Certificate of Origin Verification",
        "description": "Country of origin match, goods description, beneficiary/exporter name, certification authority.",
        "category": "documents",
        "severity": "hard",
    },
    "doc_packing_list": {
        "enabled": True,
        "name": "Packing / Weight List Verification",
        "description": "Gross/net weight, number of packages, marks and numbers consistency with BL and invoice.",
        "category": "documents",
        "severity": "hard",
    },
    "doc_inspection_cert": {
        "enabled": True,
        "name": "Inspection / Quality Certificate Verification",
        "description": "Issuing authority matches LC requirement (e.g., SGS, Intertek), goods match, date, signed.",
        "category": "documents",
        "severity": "hard",
    },
    "doc_shipping_advice": {
        "enabled": True,
        "name": "Shipping Advice Verification",
        "description": "Vessel name, shipment date, port of loading/discharge, addressed to correct party (insurer/applicant).",
        "category": "documents",
        "severity": "soft",
    },
    "doc_fumigation_cert": {
        "enabled": True,
        "name": "Fumigation Certificate Verification",
        "description": "Fumigation performed, chemicals used, date, certification by approved authority.",
        "category": "documents",
        "severity": "hard",
    },
    "doc_phytosanitary": {
        "enabled": True,
        "name": "Phytosanitary Certificate Verification",
        "description": "Issued by government authority, plant health certification, importer name, country of origin.",
        "category": "documents",
        "severity": "hard",
    },
    "doc_health_cert": {
        "enabled": True,
        "name": "Health / Veterinary Certificate Verification",
        "description": "Issued by government authority, health compliance, product safety certification.",
        "category": "documents",
        "severity": "hard",
    },
    "doc_beneficiary_cert": {
        "enabled": True,
        "name": "Beneficiary Certificate Verification",
        "description": "Signed by beneficiary, certifying specific conditions as required by LC (e.g., fax/email notification, document dispatch).",
        "category": "documents",
        "severity": "soft",
    },

    # ── Cross-Document & General Checks ──
    "cross_doc_consistency": {
        "enabled": True,
        "name": "Cross-Document Consistency",
        "description": "Verify consistency of goods description, quantity, weight, marks/numbers, vessel name, ports across BL, Invoice, Packing List, and other documents.",
        "category": "cross_check",
        "severity": "hard",
    },
    "doc_lc_number_all": {
        "enabled": True,
        "name": "LC Number on All Documents",
        "description": "LC number / documentary credit number must appear on all presented documents.",
        "category": "cross_check",
        "severity": "soft",
    },
    "doc_signatures": {
        "enabled": True,
        "name": "Signatures & Authentication",
        "description": "All documents requiring signatures must be properly signed. Stamps, seals, and authentication marks where required.",
        "category": "cross_check",
        "severity": "hard",
    },
    "doc_originals_copies": {
        "enabled": True,
        "name": "Originals & Copies Count",
        "description": "Verify correct number of originals and copies as required by LC (e.g., 'Full set 3/3 originals', '2 copies').",
        "category": "cross_check",
        "severity": "hard",
    },
    "doc_document_completeness": {
        "enabled": True,
        "name": "Document Completeness",
        "description": "All documents listed in F46A must be present in the submission. Missing documents are a discrepancy.",
        "category": "cross_check",
        "severity": "hard",
    },
}


def load_checks_config(config_dir: str = "config") -> Dict:
    """Load checks config from disk, or return defaults."""
    config_path = os.path.join(config_dir, "checks_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        merged = dict(DEFAULT_CHECKS_CONFIG)
        for k, v in saved.items():
            if k in merged:
                merged[k]["enabled"] = v.get("enabled", merged[k]["enabled"])
        return merged
    return dict(DEFAULT_CHECKS_CONFIG)


def save_checks_config(config: Dict, config_dir: str = "config") -> None:
    """Save checks config to disk."""
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, "checks_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


# ══════════════════════════════════════════════════════════════
# DOCUMENT DEDUPLICATION
# ══════════════════════════════════════════════════════════════

def _deduplicate_documents(packets: List[Dict]) -> List[Dict]:
    """Select one representative document per unique (doc_type, doc_number) pair."""
    groups: Dict[str, List[Dict]] = {}
    for pkt in packets:
        doc_type = pkt.get('document_type', 'Unknown')
        doc_num = pkt.get('document_number', '') or pkt.get('packet_id', '')
        key = f"{doc_type}|{doc_num}"
        groups.setdefault(key, []).append(pkt)

    selected = []
    for key, docs in groups.items():
        if len(docs) == 1:
            selected.append(docs[0])
        else:
            originals = [d for d in docs if 'original' in str(d.get('copy_status', '')).lower()
                         or 'original' in str(d.get('copy_label', '')).lower()]
            pool = originals if originals else docs
            best = max(pool, key=lambda d: len(d.get('refined_text', d.get('cleaned_text', ''))))
            selected.append(best)
    return selected


def _get_docs_by_type(packets: List[Dict], *doc_types: str) -> List[Dict]:
    """Get deduplicated documents matching any of the given types."""
    matched = []
    for pkt in packets:
        dt = pkt.get('document_type', '').lower()
        for t in doc_types:
            if t.lower() in dt:
                matched.append(pkt)
                break
    return _deduplicate_documents(matched)


def _get_doc_text(pkt: Dict) -> str:
    return pkt.get('refined_text', '') or pkt.get('cleaned_text', '') or pkt.get('raw_text', '')


# ══════════════════════════════════════════════════════════════
# VLM CALL
# ══════════════════════════════════════════════════════════════

def _call_vlm(prompt: str, doc_text: str, image_path: str = None) -> Dict:
    """Send a check prompt + document text to VLM and get structured result."""
    if httpx is None:
        return {"result": "REVIEW", "findings": "httpx not installed", "confidence": 0.0}

    messages = [{"role": "user", "content": []}]

    # Add image if available
    if image_path and os.path.exists(image_path):
        try:
            with open(image_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_b64}"}
            })
        except Exception:
            pass

    full_prompt = f"""{prompt}

--- DOCUMENT TEXT ---
{doc_text[:6000]}
--- END DOCUMENT TEXT ---

Respond in this exact JSON format:
{{"result": "PASS" or "FAIL" or "REVIEW", "findings": "<exact text/value you found in the document>", "detail": "<2-3 word summary>", "confidence": 0.0 to 1.0}}

IMPORTANT:
- PASS = condition is clearly satisfied
- FAIL = condition is clearly NOT satisfied (discrepancy found)
- REVIEW = cannot determine from document text (unclear/missing info)
- findings must contain the ACTUAL text or value you found, not a description
- Respond with JSON only, no other text."""

    messages[0]["content"].append({"type": "text", "text": full_prompt})

    try:
        with httpx.Client(timeout=VLM_TIMEOUT) as client:
            resp = client.post(QWEN_VLM_URL, json={
                "model": QWEN_VLM_MODEL,
                "messages": messages,
                "max_tokens": 300,
                "temperature": 0.1,
            })
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"].strip()

        # Parse JSON from response
        jm = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if jm:
            return json.loads(jm.group(0))
        return {"result": "REVIEW", "findings": text[:200], "confidence": 0.5}
    except Exception as e:
        return {"result": "REVIEW", "findings": f"VLM error: {str(e)[:100]}", "confidence": 0.0}


# ══════════════════════════════════════════════════════════════
# CHECK RESULT
# ══════════════════════════════════════════════════════════════

@dataclass
class CheckResult:
    check_id: str
    clause_ref: str
    condition: str
    document_checked: str
    findings: str
    result: str
    compliance: str             # PASS | FAIL | REVIEW
    confidence: float = 1.0
    severity: str = "hard"
    details: Dict = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════
# VLM-BASED CHECK PROMPTS
# ══════════════════════════════════════════════════════════════

def _build_check_tasks(check_id: str, lc_fields: Dict, packets: List[Dict]) -> List[Dict]:
    """
    Build VLM tasks for a given check. Each task contains a prompt, document text,
    and metadata for result construction.

    Returns list of dicts: {prompt, doc_text, image_path, clause_ref, condition, doc_type, check_id, severity}
    """
    tasks = []

    if check_id == "date_of_issue":
        lc_date = lc_fields.get('31C', lc_fields.get('F31C', ''))
        if not lc_date:
            return tasks
        lc_text_all = ' '.join(str(v) for v in lc_fields.values()).upper()
        prior_accepted = bool(re.search(r'DOCUMENT.{0,30}DATED\s+PRIOR.{0,30}(?:ISSUANCE|ISSUE|CREDIT).{0,20}(?:ACCEPT|PERMIT|ALLOW)', lc_text_all))

        exempt_types = {'analysis', 'inspection', 'certificate of origin', 'coo', 'packing list', 'packing', 'weight list', 'draft', 'bill of exchange', 'invoice', 'inception', 'beneficiary'}
        all_docs = _deduplicate_documents(packets)
        for pkt in all_docs:
            doc_type = pkt.get('document_type', 'Unknown')
            if any(ex in doc_type.lower() for ex in exempt_types):
                continue
            if prior_accepted:
                continue
            tasks.append({
                "prompt": f"""You are a trade finance document examiner. Check if this document is dated ON or AFTER the LC issuance date.

LC Date of Issue: {lc_date}
Document Type: {doc_type}

RULES:
- Find the document's date (issue date, date of document, dated, etc.)
- The document date must be ON or AFTER {lc_date}
- If the document is dated BEFORE {lc_date}, it is a FAIL (discrepancy)
- If you cannot find a date on the document, result is REVIEW""",
                "doc_text": _get_doc_text(pkt),
                "image_path": (pkt.get('page_image_paths', [None]) or [None])[0],
                "clause_ref": "F31C", "condition": f"Document must be dated on/after LC issue date ({lc_date})",
                "doc_type": doc_type, "check_id": check_id, "severity": "hard",
            })

    elif check_id == "lc_expiry":
        expiry = lc_fields.get('31D', lc_fields.get('F31D', ''))
        if not expiry:
            return tasks
        covers = _get_docs_by_type(packets, 'remittance', 'covering', 'cover letter', 'schedule')
        for cover in covers:
            tasks.append({
                "prompt": f"""You are a trade finance document examiner. Check if the documents were presented BEFORE the LC expiry date.

LC Expiry Date and Place: {expiry}
Document Type: {cover.get('document_type', 'Covering Schedule')}

RULES:
- Find the PRESENTATION DATE or DATE on the covering schedule / remittance letter
- The covering schedule confirms when documents were sent to the bank
- The presentation date must be ON or BEFORE the LC expiry date
- If presented AFTER the expiry date, it is a FAIL — "LC EXPIRED / Late Presentation"
- Extract the exact date you find on the document""",
                "doc_text": _get_doc_text(cover),
                "image_path": (cover.get('page_image_paths', [None]) or [None])[0],
                "clause_ref": "F31D", "condition": f"Documents must be presented before LC expiry ({expiry})",
                "doc_type": cover.get('document_type', 'Covering Schedule'), "check_id": check_id, "severity": "hard",
            })

    elif check_id == "amount_currency":
        amount_str = lc_fields.get('32B', lc_fields.get('F32B', ''))
        tol_str = lc_fields.get('39A', lc_fields.get('F39A', ''))
        partial = lc_fields.get('43P', lc_fields.get('F43P', ''))
        goods = lc_fields.get('45A', lc_fields.get('F45A', ''))

        # Check on all invoices
        invoices = _get_docs_by_type(packets, 'commercial invoice', 'invoice')
        for inv in invoices:
            tasks.append({
                "prompt": f"""You are a trade finance document examiner. Verify the Commercial Invoice amount against the LC.

LC Amount & Currency: {amount_str}
LC Tolerance (F39A): {tol_str if tol_str else 'Not specified'}
Partial Shipment (F43P): {partial if partial else 'Not specified'}
LC Goods Description: {str(goods)[:500]}

RULES:
1. Find the TOTAL AMOUNT and CURRENCY on this invoice
2. The currency must match the LC currency exactly (e.g., USD, EUR, PKR)
3. Check (Quantity × Unit Price) + any charges = Invoice Total (mathematical integrity)
4. If the invoice total EXCEEDS the LC amount + positive tolerance, it is OVERDRAWN → FAIL
5. If the invoice total is significantly LESS than LC amount - negative tolerance, it may be SHORT SHIPMENT
6. If tolerance says "ABOUT" or "APPROXIMATELY", apply ±10% buffer
7. Extract: currency, total amount, unit price, quantity from the invoice""",
                "doc_text": _get_doc_text(inv),
                "image_path": (inv.get('page_image_paths', [None]) or [None])[0],
                "clause_ref": "F32B", "condition": f"Invoice amount within LC amount ({amount_str})",
                "doc_type": inv.get('document_type', 'Commercial Invoice'), "check_id": check_id, "severity": "hard",
            })

        # Check on draft
        drafts = _get_docs_by_type(packets, 'draft', 'bill of exchange')
        for draft in drafts:
            tasks.append({
                "prompt": f"""You are a trade finance document examiner. Verify the Draft/Bill of Exchange amount.

LC Amount & Currency: {amount_str}
Number of Commercial Invoices presented: {len(invoices)}

RULES:
1. Find the AMOUNT and CURRENCY on the Draft/Bill of Exchange
2. Usually there is only ONE Draft for the total amount of the entire presentation
3. The Draft amount must EXACTLY match the sum of all Commercial Invoices
4. If the Draft is even $0.01 higher than the invoice total, it is a DISCREPANCY → FAIL
5. The Draft amount must NOT exceed the LC amount + tolerance
6. The currency must match the LC currency
7. Extract: amount, currency, drawee name from the draft""",
                "doc_text": _get_doc_text(draft),
                "image_path": (draft.get('page_image_paths', [None]) or [None])[0],
                "clause_ref": "F32B", "condition": f"Draft amount must match invoice total and not exceed LC ({amount_str})",
                "doc_type": draft.get('document_type', 'Draft'), "check_id": check_id, "severity": "hard",
            })

        # Check on cover schedule
        covers = _get_docs_by_type(packets, 'remittance', 'covering', 'cover', 'schedule')
        for cover in covers:
            tasks.append({
                "prompt": f"""You are a trade finance document examiner. Verify the amount on the Covering Schedule.

LC Amount & Currency: {amount_str}

RULES:
1. Find the TOTAL VALUE / AMOUNT declared on the covering schedule
2. This amount must match the sum of all presented Commercial Invoices
3. Any variance between cover schedule amount and invoice total is an Ambiguity Discrepancy
4. The currency must match the LC currency
5. Extract: amount, currency from the covering schedule""",
                "doc_text": _get_doc_text(cover),
                "image_path": (cover.get('page_image_paths', [None]) or [None])[0],
                "clause_ref": "F32B", "condition": f"Cover schedule amount must match invoice total",
                "doc_type": cover.get('document_type', 'Cover Schedule'), "check_id": check_id, "severity": "soft",
            })

    elif check_id == "draft_tenor":
        tenor = lc_fields.get('42C', lc_fields.get('F42C', ''))
        if not tenor:
            return tasks
        drafts = _get_docs_by_type(packets, 'draft', 'bill of exchange')
        for draft in drafts:
            tasks.append({
                "prompt": f"""You are a trade finance document examiner. Verify the Draft tenor matches the LC.

LC Draft Terms (F42C): {tenor}

RULES:
1. If LC says "AT SIGHT" → the Draft must clearly state "At Sight" — payment is due immediately
2. If LC says usance terms (e.g., "90 Days after Bill of Lading Date") → the Draft must state the same tenor
3. The tenor wording on the Draft must be IDENTICAL to the LC requirement
4. If usance, check that the maturity date calculation is correct based on the trigger event
5. Extract: the exact tenor/payment terms stated on the Draft""",
                "doc_text": _get_doc_text(draft),
                "image_path": (draft.get('page_image_paths', [None]) or [None])[0],
                "clause_ref": "F42C", "condition": f"Draft tenor must match LC: {tenor}",
                "doc_type": draft.get('document_type', 'Draft'), "check_id": check_id, "severity": "hard",
            })

    elif check_id == "transshipment":
        ts = lc_fields.get('43T', lc_fields.get('F43T', ''))
        if not ts:
            return tasks
        bls = _get_docs_by_type(packets, 'bill of lading', 'b/l')
        for bl in bls:
            tasks.append({
                "prompt": f"""You are a trade finance document examiner. Check for transshipment on the Bill of Lading.

LC Transshipment Condition (F43T): {ts}

RULES:
1. If LC says "NOT ALLOWED" or "PROHIBITED":
   - Check if the BL indicates transshipment (goods moved through multiple vessels)
   - If BL shows transshipment → FAIL
   - If BL shows direct shipment (no transshipment) → PASS
2. If LC says "ALLOWED" or "PERMITTED":
   - Transshipment is acceptable → PASS regardless
3. Look for: transshipment notations, multiple vessel names, "via" routing, feeder vessel mentions
4. Extract: any transshipment indication found on the BL""",
                "doc_text": _get_doc_text(bl),
                "image_path": (bl.get('page_image_paths', [None]) or [None])[0],
                "clause_ref": "F43T", "condition": f"Transshipment: {ts}",
                "doc_type": bl.get('document_type', 'Bill of Lading'), "check_id": check_id, "severity": "hard",
            })

    elif check_id == "port_of_loading":
        port = lc_fields.get('44E', lc_fields.get('F44E', ''))
        if not port:
            return tasks
        bls = _get_docs_by_type(packets, 'bill of lading', 'b/l')
        for bl in bls:
            tasks.append({
                "prompt": f"""You are a trade finance document examiner. Verify Port of Loading on the Bill of Lading.

LC Port of Loading (F44E): {port}

RULES:
1. Find the PORT OF LOADING on the Bill of Lading
2. It must match the LC requirement: "{port}"
3. Minor spelling differences are acceptable (e.g., "KARACHI PORT" vs "PORT OF KARACHI")
4. If LC says "ANY PORT IN [COUNTRY]", verify the port belongs to that country
5. If the port is completely different, it is a FAIL
6. Extract: the exact port of loading stated on the BL""",
                "doc_text": _get_doc_text(bl),
                "image_path": (bl.get('page_image_paths', [None]) or [None])[0],
                "clause_ref": "F44E", "condition": f"Port of Loading must match: {port}",
                "doc_type": bl.get('document_type', 'Bill of Lading'), "check_id": check_id, "severity": "hard",
            })

    elif check_id == "port_of_discharge":
        port = lc_fields.get('44F', lc_fields.get('F44F', ''))
        if not port:
            return tasks
        bls = _get_docs_by_type(packets, 'bill of lading', 'b/l')
        for bl in bls:
            tasks.append({
                "prompt": f"""You are a trade finance document examiner. Verify Port of Discharge on the Bill of Lading.

LC Port of Discharge (F44F): {port}

RULES:
1. Find the PORT OF DISCHARGE / DESTINATION on the Bill of Lading
2. It must match the LC requirement: "{port}"
3. Minor spelling differences are acceptable
4. If the port is completely different, it is a FAIL
5. Extract: the exact port of discharge stated on the BL""",
                "doc_text": _get_doc_text(bl),
                "image_path": (bl.get('page_image_paths', [None]) or [None])[0],
                "clause_ref": "F44F", "condition": f"Port of Discharge must match: {port}",
                "doc_type": bl.get('document_type', 'Bill of Lading'), "check_id": check_id, "severity": "hard",
            })

    elif check_id == "latest_shipment":
        latest = lc_fields.get('44C', lc_fields.get('F44C', ''))
        if not latest:
            return tasks
        bls = _get_docs_by_type(packets, 'bill of lading', 'b/l')
        for bl in bls:
            tasks.append({
                "prompt": f"""You are a trade finance document examiner. Check if shipment was made on time.

LC Latest Date of Shipment (F44C): {latest}
LC Expiry Date (F31D): {lc_fields.get('31D', lc_fields.get('F31D', 'N/A'))}

RULES:
1. Find the SHIPMENT DATE on the Bill of Lading:
   - First look for "SHIPPED ON BOARD" notation with a date — this is the definitive shipment date
   - If no on-board notation, the BL ISSUE DATE is considered the shipment date
2. The shipment date must be ON or BEFORE: {latest}
3. If the shipment date is AFTER {latest}, it is "LATE SHIPMENT" → FAIL (hard discrepancy)
4. The shipment date must also be on or before the LC Expiry Date
5. If there are multiple BLs (partial shipment), EVERY BL must have a shipment date on or before the deadline
6. Extract: the exact shipped-on-board date or BL issue date""",
                "doc_text": _get_doc_text(bl),
                "image_path": (bl.get('page_image_paths', [None]) or [None])[0],
                "clause_ref": "F44C", "condition": f"Shipment must be on/before {latest}",
                "doc_type": bl.get('document_type', 'Bill of Lading'), "check_id": check_id, "severity": "hard",
            })

    elif check_id == "presentation_period":
        period_str = lc_fields.get('48', lc_fields.get('F48', ''))
        period_days = '21'
        if period_str:
            pm = re.search(r'(\d+)', str(period_str))
            if pm:
                period_days = pm.group(1)
        covers = _get_docs_by_type(packets, 'remittance', 'covering', 'cover letter', 'schedule')
        bls = _get_docs_by_type(packets, 'bill of lading', 'b/l')
        bl_text_snippet = ''
        for bl in bls[:1]:
            bl_t = _get_doc_text(bl)
            bl_text_snippet = bl_t[:1000] if bl_t else ''

        for cover in covers:
            tasks.append({
                "prompt": f"""You are a trade finance document examiner. Check if documents were presented within the required period after shipment.

LC Period for Presentation (F48): {period_str if period_str else 'Not specified — default 21 days per UCP 600'}
Presentation period: {period_days} days after shipment date

BILL OF LADING TEXT (for shipment date reference):
{bl_text_snippet}

RULES:
1. Find the PRESENTATION DATE from the covering schedule (the date documents were sent to the bank)
2. Find the SHIPMENT DATE from the Bill of Lading (shipped on board date or BL issue date)
3. Calculate: Presentation_Date must be <= (Shipment_Date + {period_days} days)
4. If documents were presented MORE than {period_days} days after shipment → "LATE PRESENTATION" → FAIL
5. If documents are stale at presentation → FAIL
6. If F48 is blank, UCP 600 defaults to 21 days
7. Extract: presentation date from cover, shipment date from BL, and the day count""",
                "doc_text": _get_doc_text(cover),
                "image_path": (cover.get('page_image_paths', [None]) or [None])[0],
                "clause_ref": "F48", "condition": f"Documents within {period_days} days of shipment",
                "doc_type": cover.get('document_type', 'Covering Schedule'), "check_id": check_id, "severity": "hard",
            })

    return tasks


# ══════════════════════════════════════════════════════════════
# MAIN RUNNER
# ══════════════════════════════════════════════════════════════

_ENABLED_CHECK_IDS = [
    'date_of_issue', 'lc_expiry', 'amount_currency', 'draft_tenor',
    'transshipment', 'port_of_loading', 'port_of_discharge',
    'latest_shipment', 'presentation_period',
]


def run(
    lc_fields: Dict,
    packets: List[Dict],
    config_dir: str = "config",
    output_dir: str = None,
    progress_fn=None,
) -> Dict[str, Any]:
    """Run all enabled VLM-based implicit checks."""
    t0 = time.time()
    if progress_fn is None:
        def progress_fn(msg): pass

    config = load_checks_config(config_dir)
    enabled_count = sum(1 for c in config.values() if c.get('enabled'))
    progress_fn(f"Step 14b: Implicit LC key term checks ({enabled_count} enabled)")

    # Build all VLM tasks
    all_tasks = []
    for check_id in _ENABLED_CHECK_IDS:
        if not config.get(check_id, {}).get('enabled', False):
            progress_fn(f"  [{check_id}] SKIPPED (disabled)")
            continue
        tasks = _build_check_tasks(check_id, lc_fields, packets)
        all_tasks.extend(tasks)
        progress_fn(f"  [{check_id}] {len(tasks)} VLM tasks queued")

    progress_fn(f"  Total: {len(all_tasks)} VLM tasks to execute")

    # Execute VLM tasks concurrently
    all_results: List[CheckResult] = []

    def _execute_task(task: Dict) -> CheckResult:
        t1 = time.time()
        vlm_resp = _call_vlm(task['prompt'], task['doc_text'], task.get('image_path'))
        elapsed = round(time.time() - t1, 1)

        compliance = vlm_resp.get('result', 'REVIEW').upper()
        if compliance not in ('PASS', 'FAIL', 'REVIEW'):
            compliance = 'REVIEW'

        findings = vlm_resp.get('findings', '')
        detail = vlm_resp.get('detail', '')
        confidence = float(vlm_resp.get('confidence', 0.8))

        progress_fn(f"  [{task['check_id']}] [{task['doc_type']}]: {compliance} - {detail or findings[:50]} ({elapsed}s)")

        return CheckResult(
            check_id=task['check_id'],
            clause_ref=task['clause_ref'],
            condition=task['condition'],
            document_checked=task['doc_type'],
            findings=findings,
            result=detail or compliance,
            compliance=compliance,
            confidence=confidence,
            severity=task.get('severity', 'hard'),
        )

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_VLM) as executor:
        futures = {executor.submit(_execute_task, t): t for t in all_tasks}
        for future in as_completed(futures):
            try:
                result = future.result()
                all_results.append(result)
            except Exception as e:
                task = futures[future]
                progress_fn(f"  [{task['check_id']}] ERROR: {e}")
                all_results.append(CheckResult(
                    check_id=task['check_id'], clause_ref=task['clause_ref'],
                    condition=task['condition'], document_checked=task['doc_type'],
                    findings=f"Error: {str(e)[:100]}", result="Check failed",
                    compliance="REVIEW",
                ))

    # Summary
    total_pass = sum(1 for r in all_results if r.compliance == 'PASS')
    total_fail = sum(1 for r in all_results if r.compliance == 'FAIL')
    total_review = sum(1 for r in all_results if r.compliance == 'REVIEW')
    elapsed = round(time.time() - t0, 2)

    progress_fn(f"Step 14b complete: {total_pass}P / {total_fail}F / {total_review}R in {elapsed}s")

    result = {
        'step': '14b',
        'step_name': 'Implicit LC Key Term Verification',
        'checks': [asdict(r) for r in all_results],
        'summary': {
            'total': len(all_results),
            'pass': total_pass,
            'fail': total_fail,
            'review': total_review,
        },
        'config_used': {k: v.get('enabled', False) for k, v in config.items()},
        'elapsed_seconds': elapsed,
    }

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        out_path = Path(output_dir) / 'step14b_result.json'
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    return result
