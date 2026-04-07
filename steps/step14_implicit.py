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

    # ── Document & Clause Checks (VLM-based, from Step 12→14) ──
    "f46a_verification": {
        "enabled": True,
        "name": "F46A — Documents Required Verification",
        "description": "Verify all documents listed in F46A against the actual shipping documents. Each clause is decomposed into individual conditions and checked by VLM. Conditions are dynamic per LC.",
        "category": "clauses",
        "severity": "hard",
    },
    "f47a_verification": {
        "enabled": True,
        "name": "F47A — Additional Conditions Verification",
        "description": "Verify all additional conditions in F47A. These can override, supplement, or add requirements to F46A. Checked by VLM against all relevant documents.",
        "category": "clauses",
        "severity": "hard",
    },
    "f45a_verification": {
        "enabled": True,
        "name": "F45A — Description of Goods Verification",
        "description": "Verify goods description matches across Invoice, BL, and other documents. Checked by VLM.",
        "category": "clauses",
        "severity": "hard",
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

def _parse_date(date_str: str) -> Optional[datetime]:
    """Parse various date formats to datetime."""
    if not date_str:
        return None
    date_str = str(date_str).strip()
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d", "%d %b %Y", "%d %B %Y",
                "%b %d, %Y", "%B %d, %Y", "%Y%m%d", "%d.%m.%Y", "%m/%d/%Y", "%d-%b-%Y"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    m = re.search(r'(\d{4})-(\d{2})-(\d{2})', date_str)
    if m:
        try: return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except: pass
    m = re.search(r'(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+(\d{4})', date_str, re.IGNORECASE)
    if m:
        months = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}
        try: return datetime(int(m.group(3)), months[m.group(2).lower()[:3]], int(m.group(1)))
        except: pass
    # Try "Month DD, YYYY" or "MONTH DD YYYY"
    m = re.search(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+(\d{1,2}),?\s+(\d{4})', date_str, re.IGNORECASE)
    if m:
        months = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}
        try: return datetime(int(m.group(3)), months[m.group(1).lower()[:3]], int(m.group(2)))
        except: pass
    return None


def _parse_amount(amount_str: str) -> Optional[float]:
    """Parse amount string to float. Handles multiple formats:
    - USD 516,000.00
    - #516,000.00#  or  #516,000.00
    - 516000,00 (European)
    - USD\nUS DOLLAR\n516000,00\n#516,000.00
    """
    if not amount_str:
        return None
    s = str(amount_str)
    # Priority 1: formatted amount in #...# markers
    m = re.search(r'#([\d,]+\.\d+)#?', s)
    if m:
        return float(m.group(1).replace(',', ''))
    # Priority 2: comma-formatted "516,000.00" or "1,234,567.89" (MUST have comma)
    m = re.search(r'(\d{1,3}(?:,\d{3})+\.\d{2})\b', s)  # + not * — requires at least one comma
    if m:
        return float(m.group(1).replace(',', ''))
    # Priority 3: plain number with decimals "516000.00" or "23552.20"
    m = re.search(r'(\d{2,}\.\d{2})\b', s)
    if m:
        return float(m.group(1))
    # Priority 4: European format "516000,00" (comma as decimal)
    m = re.search(r'(\d+),(\d{2})\b', s)
    if m:
        return float(f"{m.group(1)}.{m.group(2)}")
    # Priority 5: plain integer
    m = re.search(r'(\d{4,})', s)
    if m:
        return float(m.group(1))
    return None


def _extract_currency(s: str) -> str:
    m = re.search(r'\b([A-Z]{3})\b', str(s))
    return m.group(1) if m else ''


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
{{"result": "PASS" or "FAIL" or "REVIEW", "findings": "<exact text/value you found in the document>", "detail": "<specific 5-10 word description of what you found or what is wrong>", "confidence": 0.0 to 1.0}}

IMPORTANT:
- PASS = condition is clearly satisfied
- FAIL = condition is clearly NOT satisfied (discrepancy found)
- REVIEW = cannot determine from document text (unclear/missing info)
- findings must contain the ACTUAL text or value you found, not a description
- detail must be SPECIFIC — never write just "DISCREPANCY FOUND" or "FAIL". Instead write what the discrepancy IS, e.g. "Cover amount USD 490,200 vs invoice USD 516,000" or "Port Vancouver does not match LC requirement Karachi"
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


def _call_vlm_extract(prompt: str, doc_text: str, image_path: str = None) -> Dict:
    """VLM extraction only — ask VLM to find and return specific values from document."""
    extract_prompt = f"""{prompt}

--- DOCUMENT TEXT ---
{doc_text[:6000]}
--- END DOCUMENT TEXT ---

Return ONLY valid JSON with the extracted values. Do not analyze or judge — just extract what you find."""

    if httpx is None:
        return {}
    messages = [{"role": "user", "content": []}]
    if image_path and os.path.exists(image_path):
        try:
            with open(image_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()
            messages[0]["content"].append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}})
        except Exception:
            pass
    messages[0]["content"].append({"type": "text", "text": extract_prompt})
    try:
        with httpx.Client(timeout=VLM_TIMEOUT) as client:
            resp = client.post(QWEN_VLM_URL, json={
                "model": QWEN_VLM_MODEL, "messages": messages,
                "max_tokens": 300, "temperature": 0.1,
            })
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"].strip()
        jm = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if jm:
            return json.loads(jm.group(0))
        return {}
    except Exception:
        return {}


# ══════════════════════════════════════════════════════════════
# HYBRID CHECKS — VLM extracts, Python compares
# ══════════════════════════════════════════════════════════════

def _hybrid_date_check(check_id: str, clause_ref: str, lc_date_str: str,
                       pkt: Dict, condition: str, check_type: str) -> CheckResult:
    """VLM extracts document date, Python compares to LC date."""
    doc_type = pkt.get('document_type', 'Unknown')
    doc_text = _get_doc_text(pkt)
    image_path = (pkt.get('page_image_paths', [None]) or [None])[0]

    # First try extracted fields from Step 9
    doc_date_str = pkt.get('document_date', '') or (pkt.get('extracted_fields', {}) or {}).get('date', '')

    # If no date from Step 9, ask VLM to extract
    if not doc_date_str or not _parse_date(doc_date_str):
        extracted = _call_vlm_extract(
            f"Find the DATE on this {doc_type}. Look for: issue date, document date, dated, presentation date, shipped on board date.\nReturn: {{\"date\": \"the date you found\"}}",
            doc_text, image_path)
        doc_date_str = extracted.get('date', '')

    lc_date = _parse_date(lc_date_str)
    doc_date = _parse_date(doc_date_str)

    if not doc_date:
        return CheckResult(check_id=check_id, clause_ref=clause_ref, condition=condition,
            document_checked=doc_type, findings=f"No date found on document",
            result="Date not found", compliance="REVIEW", severity="hard")

    if not lc_date:
        return CheckResult(check_id=check_id, clause_ref=clause_ref, condition=condition,
            document_checked=doc_type, findings=f"Cannot parse LC date: {lc_date_str}",
            result="LC date unclear", compliance="REVIEW", severity="hard")

    if check_type == 'after':
        # Document must be dated ON or AFTER lc_date
        if doc_date >= lc_date:
            return CheckResult(check_id=check_id, clause_ref=clause_ref, condition=condition,
                document_checked=doc_type, findings=f"Document date: {doc_date_str}",
                result=f"Document dated {doc_date_str} — after LC date {lc_date_str}", compliance="PASS", severity="hard")
        else:
            return CheckResult(check_id=check_id, clause_ref=clause_ref, condition=condition,
                document_checked=doc_type, findings=f"Document date: {doc_date_str}",
                result=f"Document dated {doc_date_str} — BEFORE LC date {lc_date_str}", compliance="FAIL", severity="hard")
    elif check_type == 'before':
        # Document/presentation must be ON or BEFORE lc_date
        if doc_date <= lc_date:
            return CheckResult(check_id=check_id, clause_ref=clause_ref, condition=condition,
                document_checked=doc_type, findings=f"Date: {doc_date_str}",
                result=f"Date {doc_date_str} — within deadline {lc_date_str}", compliance="PASS", severity="hard")
        else:
            return CheckResult(check_id=check_id, clause_ref=clause_ref, condition=condition,
                document_checked=doc_type, findings=f"Date: {doc_date_str}",
                result=f"Date {doc_date_str} — AFTER deadline {lc_date_str}", compliance="FAIL", severity="hard")

    return CheckResult(check_id=check_id, clause_ref=clause_ref, condition=condition,
        document_checked=doc_type, findings=doc_date_str, result="Date check", compliance="REVIEW", severity="hard")


def _hybrid_amount_check(lc_amount: float, lc_currency: str, tol_plus: float, tol_minus: float,
                          pkt: Dict, check_id: str, check_type: str, inv_amounts_str: str) -> CheckResult:
    """VLM extracts amount, Python compares."""
    doc_type = pkt.get('document_type', 'Unknown')
    doc_text = _get_doc_text(pkt)
    image_path = (pkt.get('page_image_paths', [None]) or [None])[0]

    # Try Step 9 extracted amount first
    doc_amt_str = pkt.get('document_amount', '') or (pkt.get('extracted_fields', {}) or {}).get('amount', '')

    if not doc_amt_str or not _parse_amount(doc_amt_str):
        extracted = _call_vlm_extract(
            f"Find the TOTAL AMOUNT and CURRENCY on this {doc_type}.\nReturn: {{\"amount\": \"the total amount with currency\", \"currency\": \"3-letter ISO code\"}}",
            doc_text, image_path)
        doc_amt_str = extracted.get('amount', '')

    doc_amount = _parse_amount(doc_amt_str)
    doc_currency = _extract_currency(doc_amt_str)

    if doc_amount is None:
        return CheckResult(check_id=check_id, clause_ref="F32B",
            condition=f"Amount check on {doc_type}", document_checked=doc_type,
            findings="Amount not found", result="Amount not extractable", compliance="REVIEW", severity="hard")

    # Currency check
    if lc_currency and doc_currency and doc_currency != lc_currency:
        return CheckResult(check_id=check_id, clause_ref="F32B",
            condition=f"Currency must be {lc_currency}", document_checked=doc_type,
            findings=f"{doc_currency} {doc_amount:,.2f}",
            result=f"Currency mismatch: {doc_currency} vs LC {lc_currency}", compliance="FAIL", severity="hard")

    max_amount = lc_amount * (1 + tol_plus / 100)
    min_amount = lc_amount * (1 - tol_minus / 100)

    if check_type == 'invoice_vs_lc':
        if doc_amount > max_amount:
            return CheckResult(check_id=check_id, clause_ref="F32B",
                condition=f"Invoice must not exceed LC {lc_currency} {lc_amount:,.2f} +{tol_plus}%",
                document_checked=doc_type, findings=f"{lc_currency} {doc_amount:,.2f}",
                result=f"OVERDRAWN: {lc_currency} {doc_amount:,.2f} exceeds max {lc_currency} {max_amount:,.2f}", compliance="FAIL", severity="hard")
        else:
            return CheckResult(check_id=check_id, clause_ref="F32B",
                condition=f"Invoice within LC amount ({lc_currency} {lc_amount:,.2f} ±{tol_plus}%/{tol_minus}%)",
                document_checked=doc_type, findings=f"{lc_currency} {doc_amount:,.2f}",
                result=f"Amount OK: {lc_currency} {doc_amount:,.2f} within limit {lc_currency} {max_amount:,.2f}", compliance="PASS", severity="hard")

    elif check_type == 'draft_vs_invoice':
        # Compare draft to invoice total — NOT LC amount
        inv_total = _parse_amount(inv_amounts_str)
        if inv_total and abs(doc_amount - inv_total) <= 0.01:
            return CheckResult(check_id=check_id, clause_ref="F32B",
                condition="Draft must match invoice total", document_checked=doc_type,
                findings=f"Draft: {lc_currency} {doc_amount:,.2f}",
                result=f"Draft matches invoice total {lc_currency} {inv_total:,.2f}", compliance="PASS", severity="hard")
        elif inv_total:
            return CheckResult(check_id=check_id, clause_ref="F32B",
                condition="Draft must match invoice total", document_checked=doc_type,
                findings=f"Draft: {lc_currency} {doc_amount:,.2f}",
                result=f"Draft {lc_currency} {doc_amount:,.2f} vs invoices {lc_currency} {inv_total:,.2f}", compliance="FAIL", severity="hard")
        elif doc_amount > max_amount:
            return CheckResult(check_id=check_id, clause_ref="F32B",
                condition="Draft must not exceed LC amount", document_checked=doc_type,
                findings=f"Draft: {lc_currency} {doc_amount:,.2f}",
                result=f"Draft exceeds LC max {lc_currency} {max_amount:,.2f}", compliance="FAIL", severity="hard")
        else:
            return CheckResult(check_id=check_id, clause_ref="F32B",
                condition="Draft amount check", document_checked=doc_type,
                findings=f"Draft: {lc_currency} {doc_amount:,.2f}",
                result=f"Draft within LC limit", compliance="PASS", severity="hard")

    elif check_type == 'cover_vs_invoice':
        inv_total = _parse_amount(inv_amounts_str)
        if inv_total and abs(doc_amount - inv_total) <= 0.01:
            return CheckResult(check_id=check_id, clause_ref="F32B",
                condition="Cover schedule must match invoice total", document_checked=doc_type,
                findings=f"Cover: {lc_currency} {doc_amount:,.2f}",
                result=f"Cover matches invoices", compliance="PASS", severity="soft")
        elif inv_total:
            return CheckResult(check_id=check_id, clause_ref="F32B",
                condition="Cover schedule must match invoice total", document_checked=doc_type,
                findings=f"Cover: {lc_currency} {doc_amount:,.2f} vs invoices: {lc_currency} {inv_total:,.2f}",
                result=f"Cover/invoice mismatch: {lc_currency} {doc_amount:,.2f} vs {lc_currency} {inv_total:,.2f}", compliance="FAIL", severity="soft")
        else:
            return CheckResult(check_id=check_id, clause_ref="F32B",
                condition="Cover schedule amount check", document_checked=doc_type,
                findings=f"Cover: {lc_currency} {doc_amount:,.2f}",
                result="Cannot determine invoice total for comparison", compliance="REVIEW", severity="soft")

    return CheckResult(check_id=check_id, clause_ref="F32B", condition="Amount check",
        document_checked=doc_type, findings=f"{doc_amount:,.2f}", result="Amount check", compliance="REVIEW", severity="hard")


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
- If presented AFTER the expiry date, it is a FAIL
- If FAIL, the detail MUST say "LC EXPIRED - presented [date found] after expiry [expiry date]"
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

        # Get invoice amounts from extracted fields for context
        invoices = _get_docs_by_type(packets, 'commercial invoice', 'invoice')
        _inv_amounts = []
        for inv in invoices:
            _ia = inv.get('document_amount', '') or (inv.get('extracted_fields', {}) or {}).get('amount', '')
            if _ia:
                _inv_amounts.append(_ia)
        _inv_amounts_str = ', '.join(_inv_amounts) if _inv_amounts else 'Not yet determined'

        for inv in invoices:
            tasks.append({
                "prompt": f"""You are a trade finance document examiner. Verify the Commercial Invoice amount against the LC.

LC Amount & Currency: {amount_str}
LC Tolerance (F39A): {tol_str if tol_str else 'Not specified — standard 5% under UCP 600 Art 30b'}
Partial Shipment (F43P): {partial if partial else 'Not specified'}

RULES:
1. Find the TOTAL AMOUNT and CURRENCY on this invoice
2. The currency must match the LC currency exactly (e.g., USD, EUR, PKR)
3. Check mathematical integrity: (Quantity × Unit Price) + charges = Invoice Total
4. The invoice total must NOT EXCEED the LC amount + positive tolerance → if it does, OVERDRAWN → FAIL
5. If the invoice total is LESS than the LC amount, that is ACCEPTABLE (it could be a partial shipment or short shipment within tolerance)
6. A lower invoice amount is NOT a discrepancy by itself — only exceeding the LC ceiling is
7. If tolerance says "ABOUT" or "APPROXIMATELY", apply ±10% buffer
8. Extract: currency, total amount, unit price, quantity""",
                "doc_text": _get_doc_text(inv),
                "image_path": (inv.get('page_image_paths', [None]) or [None])[0],
                "clause_ref": "F32B", "condition": f"Invoice amount must not exceed LC amount ({amount_str})",
                "doc_type": inv.get('document_type', 'Commercial Invoice'), "check_id": check_id, "severity": "hard",
            })

        # Check on draft — compare to INVOICE TOTAL, NOT LC amount
        drafts = _get_docs_by_type(packets, 'draft', 'bill of exchange')
        for draft in drafts:
            tasks.append({
                "prompt": f"""You are a trade finance document examiner. Verify the Draft/Bill of Exchange amount.

LC Amount & Currency: {amount_str}
Invoice amounts found: {_inv_amounts_str}
Number of Commercial Invoices: {len(invoices)}

CRITICAL RULES:
1. Find the AMOUNT and CURRENCY on the Draft/Bill of Exchange
2. The Draft amount must match the SUM OF ALL COMMERCIAL INVOICES — NOT the LC amount
3. DO NOT compare the Draft amount to the LC amount directly — that is WRONG
4. The LC amount ({amount_str}) is the MAXIMUM CEILING, not what the draft should equal
5. If invoices total USD 490,200.00 and the draft says USD 490,200.00 → PASS (they match)
6. If invoices total USD 490,200.00 but draft says USD 516,000.00 → FAIL (draft exceeds invoices)
7. The draft amount must also not exceed the LC amount + tolerance, but this is secondary
8. Currency must match the LC currency
9. Extract: draft amount, currency, drawee name""",
                "doc_text": _get_doc_text(draft),
                "image_path": (draft.get('page_image_paths', [None]) or [None])[0],
                "clause_ref": "F32B", "condition": f"Draft amount must match invoice total ({_inv_amounts_str}), not exceed LC ({amount_str})",
                "doc_type": draft.get('document_type', 'Draft'), "check_id": check_id, "severity": "hard",
            })

        # Check on cover schedule
        covers = _get_docs_by_type(packets, 'remittance', 'covering', 'cover', 'schedule')
        for cover in covers:
            tasks.append({
                "prompt": f"""You are a trade finance document examiner. Verify the amount on the Covering Schedule.

LC Amount & Currency: {amount_str}
Invoice amounts found: {_inv_amounts_str}

RULES:
1. Find the TOTAL VALUE / AMOUNT declared on the covering schedule
2. This amount must match the sum of all presented Commercial Invoices
3. DO NOT compare the cover amount to the LC amount — compare to invoice total
4. Any variance between cover schedule amount and invoice total is a discrepancy
5. The currency must match the LC currency
6. Extract: amount, currency from the covering schedule""",
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
3. CRITICAL: If the LC says "ANY [COUNTRY] PORT" or "ANY PORT IN [COUNTRY]" (e.g., "ANY CANADA PORT"), then ANY port in that country is acceptable = PASS. For example, Vancouver is a Canada port, so "ANY CANADA PORT" matches Vancouver = PASS.
4. Minor spelling differences are acceptable (e.g., "KARACHI PORT" vs "PORT OF KARACHI")
5. Only mark FAIL if the port is in a completely DIFFERENT country than required
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

    # ── Run HYBRID checks first (VLM extracts, Python compares) ──
    all_results: List[CheckResult] = []
    _HYBRID_CHECKS = {'date_of_issue', 'lc_expiry', 'amount_currency', 'latest_shipment'}

    for check_id in _HYBRID_CHECKS:
        if not config.get(check_id, {}).get('enabled', False):
            progress_fn(f"  [{check_id}] SKIPPED (disabled)")
            continue

        if check_id == 'date_of_issue':
            lc_date = lc_fields.get('31C', lc_fields.get('F31C', ''))
            if lc_date:
                lc_text_all = ' '.join(str(v) for v in lc_fields.values()).upper()
                prior_accepted = bool(re.search(r'DOCUMENT.{0,30}DATED\s+PRIOR.{0,30}(?:ISSUANCE|ISSUE|CREDIT).{0,20}(?:ACCEPT|PERMIT|ALLOW)', lc_text_all))
                if not prior_accepted:
                    exempt_types = {'analysis', 'inspection', 'certificate of origin', 'coo', 'packing list', 'packing', 'weight list', 'draft', 'bill of exchange', 'invoice', 'inception', 'beneficiary'}
                    for pkt in _deduplicate_documents(packets):
                        dt = pkt.get('document_type', 'Unknown')
                        if any(ex in dt.lower() for ex in exempt_types):
                            continue
                        r = _hybrid_date_check('date_of_issue', 'F31C', lc_date, pkt,
                            f"Document must be dated on/after LC issue date ({lc_date})", 'after')
                        all_results.append(r)
                        progress_fn(f"  [date_of_issue] [{dt}]: {r.compliance} - {r.result[:50]}")

        elif check_id == 'lc_expiry':
            expiry = lc_fields.get('31D', lc_fields.get('F31D', ''))
            if expiry:
                for cover in _get_docs_by_type(packets, 'remittance', 'covering', 'cover letter', 'schedule'):
                    r = _hybrid_date_check('lc_expiry', 'F31D', expiry, cover,
                        f"Documents must be presented before LC expiry ({expiry})", 'before')
                    if r.compliance == 'FAIL':
                        r.result = f"LC EXPIRED - {r.result}"
                    all_results.append(r)
                    progress_fn(f"  [lc_expiry] [{cover.get('document_type','')}]: {r.compliance} - {r.result[:50]}")

        elif check_id == 'latest_shipment':
            latest = lc_fields.get('44C', lc_fields.get('F44C', ''))
            if latest:
                # Check if F47A allows late shipment (e.g., with penalty)
                f47a = str(lc_fields.get('47A', lc_fields.get('F47A', '')))
                late_shipment_allowed = bool(re.search(
                    r'LATE\s+SHIPMENT\s+(IS\s+)?(ALLOWED|ACCEPTABLE|PERMITTED)',
                    f47a, re.IGNORECASE))
                for bl in _get_docs_by_type(packets, 'bill of lading', 'b/l'):
                    # Use shipped_on_board_date if available
                    sob = (bl.get('extracted_fields', {}) or {}).get('shipped_on_board_date', '')
                    if sob:
                        bl['document_date'] = sob  # Override for hybrid check
                    r = _hybrid_date_check('latest_shipment', 'F44C', latest, bl,
                        f"Shipment must be on/before {latest}", 'before')
                    if r.compliance == 'FAIL':
                        r.result = f"LATE SHIPMENT - {r.result}"
                        if late_shipment_allowed:
                            r.result += " (Note: F47A allows late shipment with penalty conditions)"
                    all_results.append(r)
                    progress_fn(f"  [latest_shipment] [{bl.get('document_type','')}]: {r.compliance} - {r.result[:50]}")

        elif check_id == 'amount_currency':
            amount_str = lc_fields.get('32B', lc_fields.get('F32B', ''))
            tol_str = lc_fields.get('39A', lc_fields.get('F39A', ''))
            lc_amount = _parse_amount(amount_str)
            lc_currency = _extract_currency(amount_str)
            tol_plus, tol_minus = 0.0, 0.0
            if tol_str:
                tm = re.search(r'(\d+)\s*/\s*(\d+)', str(tol_str))
                if tm:
                    tol_plus, tol_minus = float(tm.group(1)), float(tm.group(2))
                else:
                    tm = re.search(r'(\d+)', str(tol_str))
                    if tm: tol_plus = tol_minus = float(tm.group(1))
            goods = str(lc_fields.get('45A', lc_fields.get('F45A', ''))).upper()
            if re.search(r'\b(ABOUT|APPROXIMATELY|APPROX)\b', amount_str.upper() + ' ' + goods):
                tol_plus = max(tol_plus, 10.0)
                tol_minus = max(tol_minus, 10.0)

            if lc_amount:
                # Get invoice amounts — track numeric total for draft/cover comparison
                invoices = _get_docs_by_type(packets, 'commercial invoice', 'invoice')
                _inv_total_numeric = 0.0
                for inv in invoices:
                    r = _hybrid_amount_check(lc_amount, lc_currency, tol_plus, tol_minus, inv,
                        'amount_currency', 'invoice_vs_lc', '')
                    all_results.append(r)
                    # Extract numeric amount from findings
                    _inv_amt = _parse_amount(r.findings)
                    if _inv_amt:
                        _inv_total_numeric = max(_inv_total_numeric, _inv_amt)  # Use largest invoice (they're usually the same or one is the total)
                    progress_fn(f"  [amount_currency] [{inv.get('document_type','')}]: {r.compliance} - {r.result[:50]}")

                _inv_amounts_str = f"{lc_currency} {_inv_total_numeric:,.2f}" if _inv_total_numeric else ''
                progress_fn(f"  [amount_currency] Invoice total for draft/cover comparison: {_inv_amounts_str}")

                # Draft vs invoice total
                for draft in _get_docs_by_type(packets, 'draft', 'bill of exchange'):
                    r = _hybrid_amount_check(lc_amount, lc_currency, tol_plus, tol_minus, draft,
                        'amount_currency', 'draft_vs_invoice', _inv_amounts_str)
                    all_results.append(r)
                    progress_fn(f"  [amount_currency] [{draft.get('document_type','')}]: {r.compliance} - {r.result[:50]}")

                # Cover vs invoice total
                for cover in _get_docs_by_type(packets, 'remittance', 'covering', 'cover', 'schedule'):
                    r = _hybrid_amount_check(lc_amount, lc_currency, tol_plus, tol_minus, cover,
                        'amount_currency', 'cover_vs_invoice', _inv_amounts_str)
                    all_results.append(r)
                    progress_fn(f"  [amount_currency] [{cover.get('document_type','')}]: {r.compliance} - {r.result[:50]}")

    # ── Build VLM-only tasks for remaining checks ──
    all_tasks = []
    _VLM_ONLY_CHECKS = [c for c in _ENABLED_CHECK_IDS if c not in _HYBRID_CHECKS]
    for check_id in _VLM_ONLY_CHECKS:
        if not config.get(check_id, {}).get('enabled', False):
            progress_fn(f"  [{check_id}] SKIPPED (disabled)")
            continue
        tasks = _build_check_tasks(check_id, lc_fields, packets)
        all_tasks.extend(tasks)
        progress_fn(f"  [{check_id}] {len(tasks)} VLM tasks queued")

    progress_fn(f"  Total: {len(all_tasks)} VLM tasks to execute")

    # Execute VLM tasks concurrently (append to all_results from hybrid checks)

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

    # ── Cross-link: Late Presentation → LC Expired ──
    # If presentation period check found FAIL (late presentation), also flag LC expired
    has_late_presentation = any(r.check_id == 'presentation_period' and r.compliance == 'FAIL' for r in all_results)
    has_lc_expiry_fail = any(r.check_id == 'lc_expiry' and r.compliance == 'FAIL' for r in all_results)
    if has_late_presentation and not has_lc_expiry_fail:
        # Find the late presentation result for details
        late_r = next((r for r in all_results if r.check_id == 'presentation_period' and r.compliance == 'FAIL'), None)
        if late_r:
            all_results.append(CheckResult(
                check_id='lc_expiry', clause_ref='F31D',
                condition='LC has expired — documents presented after expiry date',
                document_checked=late_r.document_checked,
                findings=late_r.findings,
                result='LC EXPIRED',
                compliance='FAIL',
                confidence=1.0,
                severity='hard',
            ))
            progress_fn(f"  [lc_expiry] Auto-flagged: LC EXPIRED (linked from late presentation)")

    # Reverse: LC Expired → Late Presentation
    # If LC expired, presentation is ALWAYS late (regardless of what the period check says)
    has_late_presentation_fail = any(r.check_id == 'presentation_period' and r.compliance == 'FAIL' for r in all_results)
    if has_lc_expiry_fail and not has_late_presentation_fail:
        expiry_r = next((r for r in all_results if r.check_id == 'lc_expiry' and r.compliance == 'FAIL'), None)
        if expiry_r:
            # Remove any PASS presentation_period result (it's wrong if LC expired)
            all_results[:] = [r for r in all_results if not (r.check_id == 'presentation_period' and r.compliance == 'PASS')]
            all_results.append(CheckResult(
                check_id='presentation_period', clause_ref='F48',
                condition='Documents presented after LC expiry — late presentation',
                document_checked=expiry_r.document_checked,
                findings=expiry_r.findings,
                result='LATE PRESENTATION — documents presented after LC expiry date',
                compliance='FAIL',
                confidence=1.0,
                severity='hard',
            ))
            progress_fn(f"  [presentation_period] Auto-flagged: LATE PRESENTATION (linked from LC expiry)")

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
