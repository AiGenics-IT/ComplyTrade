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

# Single source of truth: config/settings.py decides which Qwen VLM this
# step talks to (7B vs 72B) via VLM_MODEL_SIZE. No silent fallback — if
# the import fails, fail loudly instead of using the wrong model.
import sys as _sys_imp
_sys_imp.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import QWEN_VLM_URL, QWEN_VLM_MODEL, MAX_CONCURRENT_VLM, VLM_TIMEOUT

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
    "transport_mode_match": {
        "enabled": True,
        "name": "Transport Mode Match (F44E/F44F/F46A)",
        "description": "The mode of transport implied by the LC (sea / air / courier / road / rail) must match the actual transport document presented. e.g. an LC asking for an Air Waybill cannot be satisfied with a Bill of Lading, and vice versa. Deterministic check — no VLM call.",
        "category": "documents",
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
    """
    Select one representative document per unique (doc_type, doc_number) pair.

    KEY RULE: when document_number is empty (which happens when step 9 had
    no text to extract from, e.g. on packets that came from empty step02
    pages), DO NOT fall back to packet_id — that would make every packet
    unique and defeat dedup. Instead, treat all packets of the same
    document_type as copies of the same logical document.

    Result:
      • 3 'Bill of Lading' packets all with doc_number 'KYKHIG2600024'
        → collapses to 1 representative (the Original / longest text).
      • 3 'Bill of Lading' packets all with empty doc_number
        → still collapses to 1 representative (treats them as copies
        of the same logical doc, since we have no other signal).
      • 2 'Bill of Lading' packets with DIFFERENT doc_numbers
        → kept as 2 separate documents (genuinely different BLs).
    """
    groups: Dict[str, List[Dict]] = {}
    for pkt in packets:
        doc_type = (pkt.get('document_type', 'Unknown') or 'Unknown').strip()
        doc_num = (pkt.get('document_number', '') or '').strip()
        # Normalise the type so 'Bill of Lading' / 'BILL OF LADING' /
        # 'bill of lading' all collapse to one bucket.
        type_key = doc_type.lower()
        if doc_num:
            key = f"{type_key}|{doc_num.lower()}"
        else:
            # No document_number — collapse all copies of this type
            # into one bucket. NEVER fall back to packet_id (that
            # would defeat the entire purpose of the dedup).
            key = f"{type_key}|"
        groups.setdefault(key, []).append(pkt)

    selected = []
    for key, docs in groups.items():
        if len(docs) == 1:
            selected.append(docs[0])
        else:
            originals = [d for d in docs if 'original' in str(d.get('copy_status', '')).lower()
                         or 'original' in str(d.get('copy_label', '')).lower()]
            pool = originals if originals else docs
            best = max(pool, key=lambda d: len(d.get('refined_text', d.get('cleaned_text', '')) or ''))
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

_MONTH_NAMES = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12,
    'january': 1, 'february': 2, 'march': 3, 'april': 4,
    'june': 6, 'july': 7, 'august': 8, 'september': 9,
    'october': 10, 'november': 11, 'december': 12,
    # SWIFT 3-letter abbreviations (already in the short form above)
    'sept': 9,
}

# python-dateutil is the primary parser — handles ~80% of real-world formats
# out of the box. Imported lazily so the module still loads on minimal installs.
try:
    from dateutil import parser as _du_parser
    _HAS_DATEUTIL = True
except ImportError:  # pragma: no cover
    _HAS_DATEUTIL = False


def _parse_date(date_str: str) -> Optional[datetime]:
    """
    Parse virtually any human-written or machine-written date string to a
    datetime. The trade-finance pipeline sees an enormous variety of date
    formats coming from OCR + VLM extraction — this needs to handle all of
    them OR return None gracefully.

    Supported (non-exhaustive):
      ISO            2026-01-20  2026/01/20  2026.01.20  20260120
      DMY            20-01-2026  20/01/2026  20.01.2026  20 01 2026
      MDY            01-20-2026  01/20/2026  01.20.2026
      Long          "Jan 20, 2026"  "January 20, 2026"  "20 January 2026"
      Glued         "Jan20, 2026"  "Jan.20, 2026"  "Jan. 20 2026"
                    "20-Jan-2026"  "20.Jan.2026"  "20Jan2026"  "20-JAN-26"
      Ordinal       "1st March 2026"  "20th January, 2026"  "the 3rd of May 2026"
      SWIFT         260120 (YYMMDD — only when context is a SWIFT date field)
                    20260120 (YYYYMMDD)
      With time     "2026-01-20T00:00:00"  "2026-01-20 14:30"  "Jan 20 2026 14:30"
      Mixed         "Date: Jan.20, 2026"  "Issued on 2026-01-20"  "(20.01.2026)"

    Returns None if the string cannot be confidently parsed.
    """
    if not date_str:
        return None
    raw = str(date_str).strip()
    if not raw:
        return None

    # Pre-normalisation
    s = raw
    # Strip enclosing punctuation: "(2026-01-20)" -> "2026-01-20"
    s = s.strip("()[]{}<>'\"")
    # Fix OCR-misread month abbreviations (common stamp/rubber stamp errors)
    _OCR_MONTH_FIXES = {
        'SFP': 'SEP', 'SBP': 'SEP', 'SEF': 'SEP', 'S3P': 'SEP',
        'OCI': 'OCT', 'OCL': 'OCT', 'OC1': 'OCT', 'QCT': 'OCT',
        'NOV': 'NOV', 'N0V': 'NOV',
        'DFC': 'DEC', 'DBC': 'DEC', 'D3C': 'DEC',
        'JAN': 'JAN', 'J4N': 'JAN',
        'FEB': 'FEB', 'FFB': 'FEB', 'F3B': 'FEB',
        'MAR': 'MAR', 'M4R': 'MAR',
        'APR': 'APR', 'A9R': 'APR', 'AFR': 'APR',
        'MAY': 'MAY', 'M4Y': 'MAY',
        'JUN': 'JUN', 'JUH': 'JUN',
        'JUL': 'JUL', 'JU1': 'JUL',
        'AUG': 'AUG', 'AUC': 'AUG', 'AU6': 'AUG',
    }
    for _ocr_bad, _correct in _OCR_MONTH_FIXES.items():
        if _ocr_bad != _correct:
            s = re.sub(r'\b' + _ocr_bad + r'\b', _correct, s, flags=re.IGNORECASE)
    # Strip noise prefixes that often sit in front of the date in OCR text
    s = re.sub(
        r'^(?:Date|Dated|Issued\s+on|Issue\s+date|Issuance\s+date|'
        r'Document\s+date|D/?O/?I|DOI)\s*[:\-]?\s*',
        '', s, flags=re.IGNORECASE,
    ).strip()
    # Strip "the" before ordinals: "the 3rd of May 2026"
    s = re.sub(r'\bthe\s+', '', s, flags=re.IGNORECASE)
    s = re.sub(r'\s+of\s+', ' ', s, flags=re.IGNORECASE)
    # Strip ordinal suffixes: 29th -> 29, 1st -> 1, 2nd -> 2, 3rd -> 3
    s = re.sub(r'(\d+)(st|nd|rd|th)\b', r'\1', s, flags=re.IGNORECASE)
    # Collapse repeated whitespace
    s = re.sub(r'\s+', ' ', s).strip()

    # ── Pass 1: pure-digit SWIFT-style dates (no separators) ──
    # YYYYMMDD or YYMMDD — only when the entire string is digits and
    # the length is unambiguous. We check this BEFORE dateutil because
    # dateutil treats "260120" as a year (year 260120-something).
    if re.fullmatch(r'\d{8}', s):
        try:
            return datetime(int(s[0:4]), int(s[4:6]), int(s[6:8]))
        except ValueError:
            pass
    if re.fullmatch(r'\d{6}', s):
        try:
            yy = int(s[0:2])
            mm = int(s[2:4])
            dd = int(s[4:6])
            # Pivot at 50: 00-49 → 2000s, 50-99 → 1900s
            yyyy = 2000 + yy if yy < 50 else 1900 + yy
            return datetime(yyyy, mm, dd)
        except ValueError:
            pass

    # ── Pass 1b (P198g): ISO format YYYY-MM-DD must be parsed
    # year-first with an EXPLICIT year-month-day order. Handing it to
    # dateutil with dayfirst=True causes "2025-01-02" to be read as
    # "01 day / 02 month / 2025 year" → Feb 1, which produces catastrophic
    # F31C comparisons (LC date "2025-01-02" parsed as Feb 1 makes every
    # January-dated document look like it was issued before the LC).
    # Handle ISO separators (-, /, .) explicitly here.
    _iso = re.match(r'^(\d{4})[\-./](\d{1,2})[\-./](\d{1,2})(?:[T ]\d.*)?$', s)
    if _iso:
        try:
            return datetime(int(_iso.group(1)), int(_iso.group(2)), int(_iso.group(3)))
        except ValueError:
            pass

    # ── Pass 2: dateutil (handles the majority of common formats) ──
    if _HAS_DATEUTIL:
        # Trade finance LCs are international — most use DD/MM/YYYY, so
        # dayfirst=True is the safer default for formats where the year
        # is LAST. When the year is first (ISO-like), Pass 1b above has
        # already returned; anything reaching here is year-last.
        try:
            d = _du_parser.parse(s, dayfirst=True, fuzzy=True)
            return datetime(d.year, d.month, d.day)
        except (ValueError, OverflowError, _du_parser.ParserError):
            pass
        # If DMY parsing failed AND the string looks unambiguous in MDY,
        # try once more with dayfirst=False
        try:
            d = _du_parser.parse(s, dayfirst=False, fuzzy=True)
            return datetime(d.year, d.month, d.day)
        except (ValueError, OverflowError, _du_parser.ParserError):
            pass

    # ── Pass 3: hand-rolled regex fallbacks for the cases dateutil
    # struggles with (glued formats like "Jan.20, 2026" or "Jan20, 2026") ──
    # Try strptime with a wide format list first
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d",
                "%d %b %Y", "%d %B %Y", "%b %d, %Y", "%B %d, %Y",
                "%Y%m%d", "%d.%m.%Y", "%m/%d/%Y", "%d-%b-%Y",
                "%d.%b.%Y", "%d %b. %Y", "%b %d %Y", "%B %d %Y",
                "%d-%b-%y", "%d/%b/%Y", "%d %b, %Y"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue

    # ISO-ish "YYYY-MM-DD" anywhere in the string
    m = re.search(r'(\d{4})[\-./](\d{1,2})[\-./](\d{1,2})', s)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except Exception:
            pass

    # "DD <month> YYYY" — date first, with any separator
    m = re.search(
        r'(\d{1,2})[\s\.\-/]+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?[\s\.\-/]*(\d{2,4})',
        s, re.IGNORECASE,
    )
    if m:
        try:
            yyyy = int(m.group(3))
            if yyyy < 100:
                yyyy = 2000 + yyyy if yyyy < 50 else 1900 + yyyy
            return datetime(yyyy, _MONTH_NAMES[m.group(2).lower()[:4]
                                                 if m.group(2).lower() == 'sept' else m.group(2).lower()[:3]],
                            int(m.group(1)))
        except Exception:
            pass

    # "<month> DD YYYY" / "<month> DD, YYYY" — month first, ANY separator (or none)
    m = re.search(
        r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?'
        r'\s*[\.\-/,]?\s*'                  # optional separator: space/period/dash/slash/comma, or none
        r'(\d{1,2})\s*[,\.\-/]?\s*(\d{2,4})',
        s, re.IGNORECASE,
    )
    if m:
        try:
            yyyy = int(m.group(3))
            if yyyy < 100:
                yyyy = 2000 + yyyy if yyyy < 50 else 1900 + yyyy
            mname = m.group(1).lower()
            mkey = mname[:4] if mname == 'sept' else mname[:3]
            return datetime(yyyy, _MONTH_NAMES[mkey], int(m.group(2)))
        except Exception:
            pass

    # "DD/MM/YY" / "DD-MM-YY" — last resort 2-digit year
    m = re.search(r'(\d{1,2})[\-./](\d{1,2})[\-./](\d{2})\b', s)
    if m:
        try:
            yy = int(m.group(3))
            yyyy = 2000 + yy if yy < 50 else 1900 + yy
            return datetime(yyyy, int(m.group(2)), int(m.group(1)))
        except Exception:
            pass

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
    # P198ab: comma-formatted integer "48,182,508" or "192,730,032"
    # Common for JPY/YEN which have no sub-units. Must have ≥ 2 comma
    # groups so we don't collide with Priority 4 (European "1,23").
    m = re.search(r'(\d{1,3}(?:,\d{3}){2,})(?!\d)', s)
    if m:
        return float(m.group(1).replace(',', ''))
    # Priority 4: European format "516000,00" (comma as decimal)
    m = re.search(r'(\d+),(\d{2})\b', s)
    if m:
        return float(f"{m.group(1)}.{m.group(2)}")
    # Priority 5: plain integer
    m = re.search(r'(\d{4,})', s)
    if m:
        return float(m.group(1))
    # P198ab: short comma-formatted integer "4,500" (1 comma group)
    m = re.search(r'(\d{1,3}(?:,\d{3})+)(?!\d)', s)
    if m:
        return float(m.group(1).replace(',', ''))
    return None


_CURRENCY_SYNONYMS = {
    # Japanese yen — docs often say YEN but SWIFT F32B uses JPY
    'YEN': 'JPY', 'JPY': 'JPY', 'JAPANESE YEN': 'JPY',
    # US dollar variants
    'USD': 'USD', 'US': 'USD', 'US DOLLAR': 'USD', 'US DOLLARS': 'USD',
    'DOLLARS': 'USD',
    # Euro
    'EUR': 'EUR', 'EURO': 'EUR', 'EUROS': 'EUR',
    # Pound / Pakistani rupee
    'GBP': 'GBP', 'POUND': 'GBP', 'POUNDS': 'GBP',
    'PKR': 'PKR', 'RUPEES': 'PKR', 'RS': 'PKR',
    # CNY / CNH
    'CNY': 'CNY', 'RMB': 'CNY', 'RENMINBI': 'CNY', 'YUAN': 'CNY',
}


def _normalize_currency(code: str) -> str:
    """Map document currency variants to SWIFT 3-letter ISO codes
    (e.g. YEN→JPY, RMB→CNY, US DOLLAR→USD). Returns the input
    unchanged if it's already a 3-letter code not in the synonym map."""
    if not code:
        return ''
    c = str(code).upper().strip()
    return _CURRENCY_SYNONYMS.get(c, c)


def _extract_currency(s: str) -> str:
    s_up = str(s).upper()
    # Check long-form names first (e.g. "US DOLLAR", "YEN")
    for _k in ('JAPANESE YEN', 'US DOLLAR', 'US DOLLARS',
               'RENMINBI', 'EURO', 'EUROS', 'POUND', 'POUNDS',
               'YEN', 'DOLLARS', 'RUPEES', 'YUAN', 'RMB'):
        if re.search(r'\b' + re.escape(_k) + r'\b', s_up):
            return _normalize_currency(_k)
    # Then standard 3-letter codes
    m = re.search(r'\b([A-Z]{3})\b', s_up)
    if m:
        return _normalize_currency(m.group(1))
    return ''


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
        with httpx.Client(timeout=None) as client:
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
        with httpx.Client(timeout=None) as client:
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

def _extract_received_stamp_date(pkt: Dict):
    """For covering schedules / documentary remittance: look at stamps
    AND labeled date fields on the page. Returns (datetime, raw text)
    on success, or None if no identifiable presentation/received date
    is parseable.

    Priority order:
      1. Stamps with parseable dates (RECEIVED stamp at issuing bank —
         strongest evidence of actual receipt date).
      2. Explicitly labeled dates in the doc text: "Presentation Date",
         "Received Date", "Received:", "Date of Presentation",
         "Presented on" — these carry the same semantic weight.
      3. None — no identifiable presentation/received date.

    P198ce — unlabelled bare dates (e.g. "DATE: 18/02/25" at the top
    of a covering schedule) are NEVER used — that is the SENDING /
    issue date, not the receiving date.
    """
    # 1. Stamps
    for stamp in (pkt.get('stamps') or []):
        if not isinstance(stamp, dict):
            continue
        text = str(stamp.get('text', '') or '').strip()
        if not text:
            continue
        dt = _parse_date(text)
        if dt:
            return dt, text

    # 2. Labeled date fields in doc text
    doc_text = _get_doc_text(pkt) or ''
    _labeled_patterns = [
        r'RECEIVED\s+DATE\s*[:\-]\s*([^\n]{4,30})',
        r'DATE\s+OF\s+RECEIPT\s*[:\-]\s*([^\n]{4,30})',
        r'DATE\s+RECEIVED\s*[:\-]\s*([^\n]{4,30})',
        r'PRESENTATION\s+DATE\s*[:\-]\s*([^\n]{4,30})',
        r'DATE\s+OF\s+PRESENTATION\s*[:\-]\s*([^\n]{4,30})',
        r'PRESENTED\s+ON\s*[:\-]?\s*([^\n]{4,30})',
        r'RECEIVED\s*[:\-]\s*([0-9][^\n]{3,25})',
        r'RECEIVING\s+DATE\s*[:\-]\s*([^\n]{4,30})',
    ]
    for _pat in _labeled_patterns:
        m = re.search(_pat, doc_text, re.IGNORECASE)
        if not m:
            continue
        raw = m.group(1).strip()
        dt = _parse_date(raw)
        if dt:
            return dt, m.group(0).strip()
    return None


def _hybrid_date_check(check_id: str, clause_ref: str, lc_date_str: str,
                       pkt: Dict, condition: str, check_type: str) -> CheckResult:
    """VLM extracts document date, Python compares to LC date."""
    doc_type = pkt.get('document_type', 'Unknown')
    doc_text = _get_doc_text(pkt)
    image_path = (pkt.get('page_image_paths', [None]) or [None])[0]

    # P198bg — For lc_expiry and presentation_period on covering /
    # remittance / schedule docs, the RECEIVED stamp is the only
    # trustworthy presentation date. The document_date on such covers is
    # the SENDING date (when the negotiating bank dispatched), NOT the
    # receiving date at the issuing bank. Using the sending date causes
    # false FAILs ("LC expired") or false PASSes on genuinely late
    # presentations. If the stamp is OCR-mangled and unparseable, we
    # return REVIEW — we must NOT fall back to the issue date.
    _doc_type_lo = (doc_type or '').lower()
    _is_cover_doc = any(k in _doc_type_lo for k in (
        'remittance', 'cover letter', 'covering', 'schedule',
        'cover schedule',
    ))
    if check_id in ('lc_expiry', 'presentation_period') and _is_cover_doc:
        # P198ce — If the covering schedule / remittance EXPLICITLY
        # certifies that documents were presented on time / within LC
        # validity, that assertion IS the presentation-timing evidence
        # — PASS without needing a stamp date. Look for common phrases
        # bank covering letters use to confirm on-time presentation.
        _doc_text_up = _get_doc_text(pkt).upper()
        _presentation_ok_patterns = (
            r'DOCUMENTS?\s+(?:HAVE\s+BEEN\s+)?PRESENTED\s+(?:ON\s+TIME|'
            r'WITHIN\s+(?:THE\s+)?(?:L/?C\s+)?(?:VALIDITY|EXPIRY|PERIOD)|'
            r'PRIOR\s+TO\s+EXPIRY|BEFORE\s+(?:THE\s+)?EXPIRY|'
            r'WITHIN\s+(?:THE\s+)?PRESENTATION\s+PERIOD)',
            r'PRESENTATION\s+(?:IS\s+|WAS\s+)?MADE\s+WITHIN\s+(?:L/?C\s+)?'
            r'(?:VALIDITY|EXPIRY\s+DATE)',
            r'DOCUMENTS?\s+ARE\s+(?:BEING\s+)?PRESENTED\s+WITHIN\s+VALIDITY',
            r'WITHIN\s+L/?C\s+VALIDITY\s+PERIOD',
            r'DOCUMENTS?\s+NEGOTIATED\s+WITHIN\s+(?:THE\s+)?VALIDITY',
        )
        if any(re.search(p, _doc_text_up, re.IGNORECASE)
               for p in _presentation_ok_patterns):
            return CheckResult(
                check_id=check_id, clause_ref=clause_ref, condition=condition,
                document_checked=doc_type,
                findings=(
                    "Covering schedule explicitly certifies documents were "
                    "presented within LC validity / on time."
                ),
                result="Presented within LC validity per covering schedule",
                compliance="PASS", severity="hard",
            )
        _stamp = _extract_received_stamp_date(pkt)
        _stamps_raw = pkt.get('stamps') or []
        if _stamp is not None:
            doc_date = _stamp[0]
            doc_date_str = _stamp[1]
            lc_date = _parse_date(lc_date_str)
            if lc_date is None:
                return CheckResult(
                    check_id=check_id, clause_ref=clause_ref, condition=condition,
                    document_checked=doc_type,
                    findings=f"Cannot parse LC date: {lc_date_str}",
                    result="LC date unclear", compliance="REVIEW", severity="hard",
                )
            if check_type == 'before':
                if doc_date <= lc_date:
                    return CheckResult(
                        check_id=check_id, clause_ref=clause_ref, condition=condition,
                        document_checked=doc_type,
                        findings=(
                            f"Presentation date from RECEIVED stamp: "
                            f"'{doc_date_str}' (parsed {doc_date:%Y-%m-%d})"
                        ),
                        result=(
                            f"Received {doc_date:%Y-%m-%d} — within deadline "
                            f"{lc_date_str}"
                        ),
                        compliance="PASS", severity="hard",
                    )
                return CheckResult(
                    check_id=check_id, clause_ref=clause_ref, condition=condition,
                    document_checked=doc_type,
                    findings=(
                        f"Presentation date from RECEIVED stamp: "
                        f"'{doc_date_str}' (parsed {doc_date:%Y-%m-%d})"
                    ),
                    result=(
                        f"Received {doc_date:%Y-%m-%d} — AFTER deadline "
                        f"{lc_date_str}"
                    ),
                    compliance="FAIL", severity="hard",
                )
        # No stamp OR stamp could not be parsed → REVIEW (never fall
        # back to the issue date on a covering document, since the
        # issue date is when the negotiating bank SENT, not when the
        # issuing bank RECEIVED).
        # P198bt — Short, user-facing finding. Prior wording was too
        # long (~100 words) and repeated the system's internal logic in
        # the report. Keep it one line.
        return CheckResult(
            check_id=check_id, clause_ref=clause_ref, condition=condition,
            document_checked=doc_type,
            findings="Receiving / presentation date not clear — manual review.",
            result="Receiving / presentation date not clear — manual review",
            compliance="REVIEW", severity="hard",
        )

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

    # P198ac: for cover / schedule / remittance docs we prefer
    # unified_summary.amounts_found[role=documents_value|documents_amount]
    # over pkt.document_amount. The packet-level document_amount is
    # frequently the LC face amount (F32B) copied from the cover
    # header instead of the actual documents-value line, which would
    # cause false "cover/invoice mismatch" FAILs.
    _dt_lo_pre = str(doc_type).lower()
    _is_cover_pre = (
        'remittance' in _dt_lo_pre or
        'covering' in _dt_lo_pre or
        'schedule' in _dt_lo_pre or
        'cover' in _dt_lo_pre
    )
    doc_amt_str = ''
    if _is_cover_pre:
        _us_pre = pkt.get('unified_summary') or {}
        if isinstance(_us_pre, dict):
            for _role in ('documents_value', 'documents_amount',
                          'total_payable'):
                for _item in (_us_pre.get('amounts_found') or []):
                    if not isinstance(_item, dict):
                        continue
                    if str(_item.get('role', '') or '').lower() != _role:
                        continue
                    _v = _item.get('value') or _item.get('raw') or ''
                    if _v and _parse_amount(str(_v).strip()):
                        _ccy = str(_item.get('currency', '') or '').strip()
                        doc_amt_str = (
                            f"{_ccy} {_v}".strip() if _ccy else str(_v).strip()
                        )
                        break
                if doc_amt_str:
                    break

    # Try Step 9 extracted amount next
    if not doc_amt_str:
        doc_amt_str = pkt.get('document_amount', '') or (pkt.get('extracted_fields', {}) or {}).get('amount', '')

    # P198ab: fall back to unified_summary before hitting the VLM.
    # Step 3 already extracts amounts_found[] with role labels; use
    # those so we don't pay for an LLM round-trip (and so JPY-style
    # integer amounts that live only in amounts_found get picked up).
    if not doc_amt_str or not _parse_amount(doc_amt_str):
        _us = pkt.get('unified_summary') or {}
        if isinstance(_us, dict):
            _dt_lo = str(doc_type).lower()
            _is_inv = 'invoice' in _dt_lo
            _is_draft_doc = (
                'draft' in _dt_lo or
                'bill of exchange' in _dt_lo or
                _dt_lo.strip() == 'boe'
            )
            _is_cover = (
                'remittance' in _dt_lo or
                'covering' in _dt_lo or
                'schedule' in _dt_lo or
                'cover' in _dt_lo
            )
            if _is_inv:
                _preferred_roles = (
                    'invoice_total', 'total_amount', 'total',
                    'invoice_amount', 'document_amount', 'amount',
                    'line_item_amount',
                )
            elif _is_draft_doc:
                _preferred_roles = (
                    'draft_amount', 'face_value', 'amount',
                    'document_amount',
                )
            elif _is_cover:
                _preferred_roles = (
                    'documents_value', 'documents_amount',
                    'total_payable', 'total_amount', 'total',
                    'lc_amount', 'amount',
                )
            else:
                _preferred_roles = (
                    'total', 'total_amount', 'invoice_total',
                    'document_amount', 'amount',
                )
            _found = ''
            for _role in _preferred_roles:
                for _item in (_us.get('amounts_found') or []):
                    if not isinstance(_item, dict):
                        continue
                    _r = str(_item.get('role', '') or '').lower()
                    if _r != _role:
                        continue
                    _v = (
                        _item.get('value') or _item.get('raw') or
                        _item.get('amount') or ''
                    )
                    if _v and _parse_amount(str(_v).strip()):
                        _ccy = str(_item.get('currency', '') or '').strip()
                        _found = (
                            f"{_ccy} {_v}".strip() if _ccy else str(_v).strip()
                        )
                        break
                if _found:
                    break
            # Typed top-level keys as a secondary fallback
            if not _found:
                for _k in (
                    'document_amount', 'draft_amount',
                    'invoice_amount', 'total_amount', 'amount',
                ):
                    _v = _us.get(_k)
                    if _v and _parse_amount(str(_v).strip()):
                        _found = str(_v).strip()
                        break
            if _found:
                doc_amt_str = _found

    if not doc_amt_str or not _parse_amount(doc_amt_str):
        extracted = _call_vlm_extract(
            f"Find the TOTAL AMOUNT and CURRENCY on this {doc_type}.\nReturn: {{\"amount\": \"the total amount with currency\", \"currency\": \"3-letter ISO code\"}}",
            doc_text, image_path)
        doc_amt_str = extracted.get('amount', '')

    doc_amount = _parse_amount(doc_amt_str)
    doc_currency = _extract_currency(doc_amt_str)
    # P198ac — Normalise LC currency for comparison (YEN vs JPY, etc.)
    lc_currency = _normalize_currency(lc_currency) if lc_currency else lc_currency

    # P184 — DRAFT TENOR DUPLICATE correction.
    # If the document is a Draft / Bill of Exchange and the extracted
    # amount is exactly 2× (or 3×) a repeating face value, this is a
    # tenor-duplicate BoE (First + Second of Exchange on the same page,
    # same face value) that got doubled by the summariser. Use the
    # face value once.
    _is_draft = (
        'draft' in str(doc_type).lower() or
        'bill of exchange' in str(doc_type).lower() or
        'boe' == str(doc_type).lower().strip()
    )
    if _is_draft and doc_amount is not None and doc_text:
        # Count distinct face-value occurrences in the doc text.
        _amt_re = re.compile(
            r'(?:USD|EUR|GBP|JPY|CNY|PKR|AED|SAR)\s*([\d,]+(?:\.\d{1,2})?)',
            re.IGNORECASE,
        )
        _found_amts = []
        for _m in _amt_re.finditer(doc_text or ''):
            try:
                _v = float(_m.group(1).replace(',', ''))
                if _v > 0:
                    _found_amts.append(_v)
            except (ValueError, IndexError):
                pass
        if _found_amts:
            # If the first occurrence repeats (2+ times) and sums to the
            # current doc_amount, use the single face value.
            _first = _found_amts[0]
            _count_first = sum(1 for _v in _found_amts if abs(_v - _first) < 0.01)
            if (_count_first >= 2 and _first > 0 and
                    abs(doc_amount - _first * _count_first) < 0.02):
                doc_amount = _first
                doc_amt_str = f"{doc_currency or lc_currency} {_first:,.2f}"
            # Alternative: if doc_text has explicit
            # "First of Exchange" + "Second of Exchange" with the SAME
            # reference number, always take the single face value.
            _txt_up = (doc_text or '').upper()
            if ('FIRST OF EXCHANGE' in _txt_up and
                    ('SECOND OF EXCHANGE' in _txt_up or
                     'SECOND OF THE SAME TENOR' in _txt_up)):
                if _found_amts and _first > 0:
                    doc_amount = _first
                    doc_amt_str = f"{doc_currency or lc_currency} {_first:,.2f}"

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

        # P187 — Tenor-duplicate correction. When a Draft / Bill of
        # Exchange page carries BOTH "First of Exchange" and "Second of
        # Exchange" for the same underlying draft, the summariser
        # commonly sums the two face values and returns 2× the real
        # amount. If we see the tenor-duplicate markers in the text OR
        # the extracted draft amount is effectively 2× (or 3×) the
        # invoice total, divide the draft amount back down to the
        # single face value before comparing.
        if _is_draft and inv_total and doc_amount:
            _txt_up = (doc_text or '').upper()
            _has_first_second = (
                ('FIRST OF EXCHANGE' in _txt_up or 'SOLE OF EXCHANGE' in _txt_up) and
                ('SECOND OF EXCHANGE' in _txt_up or
                 'SECOND OF THE SAME TENOR' in _txt_up or
                 'OF THE SAME TENOR AND DATE' in _txt_up)
            )
            for _mult in (2, 3):
                if abs(doc_amount - inv_total * _mult) <= max(0.02, inv_total * 0.001):
                    # Amount is n× invoice total — strong tenor-dup signal.
                    if _has_first_second or _mult == 2:
                        doc_amount = doc_amount / _mult
                        doc_amt_str = f"{doc_currency or lc_currency} {doc_amount:,.2f}"
                    break
            # Even if the ratio check didn't fire (rounding edge), the
            # explicit First+Second markers plus a doc_amount >
            # inv_total is enough signal. Halve once.
            if (_has_first_second and doc_amount > inv_total * 1.8 and
                    doc_amount <= inv_total * 2.2):
                doc_amount = doc_amount / 2.0
                doc_amt_str = f"{doc_currency or lc_currency} {doc_amount:,.2f}"

        if inv_total and abs(doc_amount - inv_total) <= 0.01:
            return CheckResult(check_id=check_id, clause_ref="F32B",
                condition="Draft must match invoice total", document_checked=doc_type,
                findings=f"Draft: {lc_currency} {doc_amount:,.2f}",
                result=f"Draft matches invoice total {lc_currency} {inv_total:,.2f}", compliance="PASS", severity="hard")
        elif inv_total:
            # P198cp — Draft amount may LEGITIMATELY differ from the
            # sum of commercial invoices when the presentation is a
            # partial shipment / tranche, when the draft is drawn for
            # one specific invoice in a multi-invoice presentation, or
            # when the LC permits partial drawings. Comparing draft to
            # total-of-invoices and hard-FAILing on mismatch produces
            # false discrepancies on valid partial-shipment flows.
            # Demote to informational REVIEW so the difference is
            # visible in the review queue without blocking compliance.
            return CheckResult(check_id=check_id, clause_ref="F32B",
                condition="Draft and invoice total (informational)",
                document_checked=doc_type,
                findings=f"Draft: {lc_currency} {doc_amount:,.2f}",
                result=(f"Draft {lc_currency} {doc_amount:,.2f} vs "
                        f"invoices total {lc_currency} {inv_total:,.2f} "
                        f"— not cross-checked (partial shipments and "
                        f"tranche drafts are permitted under UCP 600)."),
                compliance="REVIEW", severity="soft")
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
            _cover_stamps_exp = cover.get('stamps', [])
            _stamp_info = ''
            if _cover_stamps_exp:
                _stamp_info = '\nSTAMPS ON DOCUMENT: ' + ', '.join(
                    s.get('text', '') for s in _cover_stamps_exp if s.get('text')
                )
            tasks.append({
                "prompt": f"""You are a trade finance document examiner. Check if the documents were presented BEFORE the LC expiry date.

LC Expiry Date and Place: {expiry}
Document Type: {cover.get('document_type', 'Covering Schedule')}{_stamp_info}

RULES:
- The PRESENTATION DATE is when the ISSUING BANK RECEIVED the documents,
  NOT when the negotiating bank sent them.
- PRIORITY ORDER for finding the presentation date:
  1. RECEIVED stamp (e.g., "RECEIVED 19 SEP 2025") — this is the BEST
     evidence of when the issuing bank received the documents. USE THIS.
  2. "Presented on:", "Date of Presentation:", "Received Date:" fields
  3. "DATE:" on the covering schedule — this is the SENDING date (when
     the negotiating bank dispatched), NOT the receiving date. Only use
     this as FALLBACK if no RECEIVED stamp exists.
- Check the STAMPS section in the visual metadata for RECEIVED stamps.
- The presentation date must be ON or BEFORE the LC expiry date
- If presented AFTER the expiry date, it is a FAIL
- If FAIL, the detail MUST say "LC EXPIRED - presented [date found] after expiry [expiry date]"
- Extract the exact date you find (preferably from RECEIVED stamp)""",
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

━━━ UCP 600 DEFINITION OF TRANSSHIPMENT (CRITICAL — READ FIRST) ━━━
UCP 600 Art 19(b) / 20(b) define transshipment as:
   "Unloading from one means of conveyance and reloading to ANOTHER
    means of conveyance during the carriage from the place of dispatch
    / port of loading to the place of final destination / port of
    discharge named in the credit."

Transshipment REQUIRES: cargo taken off the first vessel AND reloaded
onto a DIFFERENT vessel mid-journey. A single vessel calling at
multiple ports on one voyage is NOT transshipment.

━━━ WHAT IS TRANSSHIPMENT (FAIL) ━━━
Transshipment is indicated ONLY by EXPLICIT evidence of cargo moving
between DIFFERENT vessels:
  • "TRANSHIPPED AT <PORT> BY <SECOND VESSEL>" or similar
  • "TRANSSHIPMENT AT <PORT>" with a second vessel named
  • Two separate vessels named as the first-leg carrier and the
    ocean carrier (feeder vessel + mother vessel)
  • "ON-CARRIAGE FROM <PORT> BY <VESSEL>" or "CONNECTING VESSEL"
  • Explicit "T/S" / "TRANSHIP" / "VIA <PORT> BY <OTHER VESSEL>"

━━━ WHAT IS NOT TRANSSHIPMENT (PASS) ━━━
The following patterns ALONE are NOT transshipment:
  • "AS AT <PORT>" / "AS AGENTS AT <PORT>" / "AS AGENT AT <PORT>"
    — indicates the DECLARED / CONTRACTUAL loading port; the ship
    recorded the loading at this port even if the vessel also
    touched another port. This is NOT transshipment.
  • Carrier's office address in a different country (boilerplate
    on the BL — just the issuer's HQ, not a routing leg)
  • The same vessel calling at multiple ports on one voyage
  • "ROUTING VIA <PORT>" on the SAME vessel with NO second vessel
  • "FROM <PORT A> TO <PORT B>" listed on the same vessel
  • A discharge at an intermediate port followed by cabotage/
    local delivery — unless a DIFFERENT ocean vessel is named
  • Port Klang, Singapore, Colombo, or Jebel Ali appearing as
    the CARRIER'S OFFICE, NOT as an on-carriage port

Common false-positive pattern (DO NOT fail this):
    "Port of Loading: BALIKPANG, INDONESIA"
    "PORT KLANG, MALAYSIA AS AT BALIKPANG, INDONESIA"
   → "AS AT" means the BL RECORDS loading at Balikpang, even though
     the carrier's office / handling agent is at Port Klang. Only
     ONE vessel, ONE loading. NOT transshipment → PASS.

━━━ DECISION RULES ━━━
1. If LC says "ALLOWED" / "PERMITTED":
   - Transshipment (if any) is acceptable → PASS regardless.
2. If LC says "NOT ALLOWED" / "PROHIBITED":
   - Apply the UCP 600 transshipment definition above.
   - FAIL only when you can identify EXPLICIT evidence of TWO
     DIFFERENT vessels + a mid-journey reload.
   - If the BL shows only "AS AT" / carrier-office / same-vessel
     multi-port routing → PASS (no transshipment).
   - When in doubt and no second vessel is identified → PASS.
3. Quote the exact BL line you are relying on in your response.
   Do NOT flip to FAIL based on the word "via" alone or on a
   carrier office being in a different country.""",
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
1. Find the PORT OF LOADING on the Bill of Lading.
2. It must match the LC requirement: "{port}".
3. ━━━ "ANY [COUNTRY] PORT" RULE — READ CAREFULLY ━━━
   If the LC says "ANY [COUNTRY] PORT/SEAPORT", "ANY PORT IN [COUNTRY]", or "ANY [COUNTRY ADJECTIVE] PORT/SEAPORT", then ANY port located IN THAT COUNTRY is acceptable → PASS.
   The country adjective form ALWAYS refers to the same country as the noun form:
       CHINESE = CHINA           CANADIAN = CANADA          PAKISTANI = PAKISTAN
       INDIAN = INDIA            MALAYSIAN = MALAYSIA       INDONESIAN = INDONESIA
       SINGAPOREAN = SINGAPORE   THAI = THAILAND            VIETNAMESE = VIETNAM
       JAPANESE = JAPAN          KOREAN = (SOUTH) KOREA     BANGLADESHI = BANGLADESH
       FILIPINO = PHILIPPINES    EMIRATI = UAE              SAUDI = SAUDI ARABIA
       TURKISH = TURKEY          GERMAN = GERMANY           ITALIAN = ITALY
       SPANISH = SPAIN           BRITISH/ENGLISH = UK       AMERICAN = USA
       BRAZILIAN = BRAZIL        AUSTRALIAN = AUSTRALIA     KENYAN = KENYA
       SOUTH AFRICAN = SOUTH AFRICA
   So "ANY CHINESE SEAPORT" = "ANY CHINA PORT" — both mean any port in China.
4. ━━━ COUNTRY MEMBERSHIP — READ CAREFULLY ━━━
   The following cities/regions are PART OF CHINA for the purpose of "any Chinese port":
       Hong Kong / HONGKONG / HK / HKG / KWAI CHUNG, Macau / MACAO,
       Shanghai, Shenzhen, Ningbo, Qingdao, Tianjin, Guangzhou, Xiamen,
       Dalian, Yantian, Huangpu, Hong Kong International, Kaohsiung
   Other useful country memberships:
       SINGAPORE = Singapore (city-state)
       MALAYSIA: Port Klang, Penang, Johor, Tanjung Pelepas
       INDONESIA: Jakarta, Tanjung Priok, Surabaya, Belawan
       UAE: Dubai (Jebel Ali), Abu Dhabi, Sharjah
       UK: Felixstowe, Southampton, London Gateway, Liverpool
       USA: Long Beach, Los Angeles, NY/NJ, Houston, Charleston, Savannah
       INDIA: Mumbai (Nhava Sheva), Chennai, Kolkata, Cochin, Kandla
       PAKISTAN: Karachi, Port Qasim, Bin Qasim
   So "HONGKONG SEAPORT, CHINA" matches "ANY CHINESE SEAPORT" → PASS, because Hong Kong is in China.
5. EXAMPLES THAT MUST PASS (do not mark these as FAIL):
   • LC: "ANY CHINESE SEAPORT"     BL: "HONGKONG SEAPORT, CHINA"   → PASS (Hong Kong is in China)
   • LC: "ANY CHINESE SEAPORT"     BL: "SHANGHAI, CHINA"           → PASS
   • LC: "ANY CHINESE SEAPORT"     BL: "MACAU, CHINA"              → PASS
   • LC: "ANY CHINA PORT"          BL: "HUANGPU CHINA"             → PASS
   • LC: "ANY CANADIAN PORT"       BL: "VANCOUVER, CANADA"         → PASS
   • LC: "ANY MALAYSIA PORT"       BL: "PENANG PORT, MALAYSIA"     → PASS
   • LC: "KARACHI PORT, PAKISTAN"  BL: "PORT OF KARACHI"           → PASS (same city, qualifier varies)
   • LC: "SHANGHAI"                BL: "SHANGHAI SEAPORT"          → PASS (qualifier varies)
6. EXAMPLES THAT MUST FAIL:
   • LC: "ANY CHINESE SEAPORT"     BL: "PORT OF KARACHI, PAKISTAN" → FAIL (Pakistan is not China)
   • LC: "KARACHI"                 BL: "MUMBAI"                    → FAIL (different cities, different countries)
7. Minor spelling differences are acceptable (e.g., "KARACHI PORT" vs "PORT OF KARACHI", "HONGKONG" vs "HONG KONG").
8. Only mark FAIL if the port is in a COMPLETELY DIFFERENT country than what the LC requires.
9. Extract: the exact port of loading stated on the BL.""",
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
1. Find the PORT OF DISCHARGE / DESTINATION on the Bill of Lading.
2. It must match the LC requirement: "{port}".
3. ━━━ "ANY [COUNTRY] PORT" RULE — same as Port of Loading ━━━
   "ANY [COUNTRY] PORT/SEAPORT" or "ANY [COUNTRY ADJECTIVE] PORT/SEAPORT" matches ANY port in that country.
   Country adjectives ALWAYS map to the country noun:
       CHINESE = CHINA, CANADIAN = CANADA, PAKISTANI = PAKISTAN, INDIAN = INDIA,
       MALAYSIAN = MALAYSIA, INDONESIAN = INDONESIA, JAPANESE = JAPAN,
       KOREAN = KOREA, BANGLADESHI = BANGLADESH, EMIRATI = UAE,
       SAUDI = SAUDI ARABIA, AMERICAN = USA, BRITISH = UK, etc.
4. CHINA includes: Hong Kong / HONGKONG / HK / HKG, Macau / MACAO, Shanghai,
   Shenzhen, Ningbo, Qingdao, Tianjin, Guangzhou, Xiamen, Dalian, Yantian, Huangpu, Kaohsiung.
   So "HONGKONG, CHINA" satisfies "ANY CHINESE PORT" → PASS.
5. EXAMPLES THAT MUST PASS:
   • LC "ANY CHINESE SEAPORT"  vs BL "HONGKONG SEAPORT, CHINA"  → PASS
   • LC "KARACHI PORT, PAKISTAN" vs BL "PORT OF KARACHI"        → PASS
   • LC "ANY UK PORT"          vs BL "FELIXSTOWE, UK"           → PASS
6. Minor spelling / qualifier differences are acceptable
   ("KARACHI PORT" = "PORT OF KARACHI" = "KARACHI SEAPORT").
7. Only mark FAIL if the port is in a COMPLETELY DIFFERENT country than required.
8. Extract: the exact port of discharge stated on the BL.""",
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

    elif check_id == "transport_mode_match":
        # ── DETERMINISTIC: no VLM call ──
        # Detect the transport mode the LC expects from F44E (Port of
        # Loading / Airport of Departure), F44F (Port of Discharge /
        # Airport of Destination), and the F46A document requirements.
        # Compare against what was actually presented (Bill of Lading vs
        # Air Waybill vs Courier Receipt). Mismatch = hard discrepancy.
        f44e = (lc_fields.get('44E') or lc_fields.get('F44E') or '').upper()
        f44f = (lc_fields.get('44F') or lc_fields.get('F44F') or '').upper()
        f46a_full = (lc_fields.get('46A') or lc_fields.get('F46A') or '').upper()
        # Combine all 46A clauses if stored as a list
        f46a_clauses = lc_fields.get('46A_clauses') or []
        if isinstance(f46a_clauses, list):
            f46a_full = (f46a_full + ' ' + ' '.join(str(c).upper() for c in f46a_clauses)).strip()
        f43p = (lc_fields.get('43P') or lc_fields.get('F43P') or '').upper()  # Partial Shipments

        def _expected_mode():
            # PRIORITY 1: F44E/F44F port names — strongest signal.
            # Use word-boundary matching so "AIRPORT" does NOT also
            # satisfy a bare "PORT" check (P198cw fix — "ANY AIRPORT
            # IN CHINA" was previously flagged BOTH as has_sea_port
            # (because "PORT" appears as a substring) AND
            # has_air_port, defeating the mode detection and falling
            # through to the F46A priority).
            _joined = f"{f44e} {f44f}"
            _has_air_port = bool(re.search(r'\bAIRPORT\b|\bAIR\s+PORT\b', _joined))
            _has_sea_port = bool(re.search(r'\bSEAPORT\b|\bSEA\s+PORT\b', _joined))
            if not _has_air_port and not _has_sea_port:
                # Generic "PORT" (bare token, NOT inside AIRPORT) implies SEA.
                _has_sea_port = bool(re.search(r'\bPORT\b', _joined))
            if _has_air_port and not _has_sea_port:
                return 'AIR'
            if _has_sea_port and not _has_air_port:
                return 'SEA'

            # PRIORITY 2: F46A document requirements — secondary signal
            # Check what transport document the LC requires
            _f46a_has_bl = any(k in f46a_full for k in (
                'BILL OF LADING', 'OCEAN BILL OF LADING', 'MARINE BILL OF LADING',
                'SHIPPED ON BOARD', 'CHARTER PARTY BILL', 'COMBINED TRANSPORT BILL',
                'MULTIMODAL TRANSPORT', 'CLEAN ON BOARD',
            ))
            _f46a_has_awb = any(k in f46a_full for k in (
                'AIR WAYBILL', 'AIRWAY BILL', 'AIRWAYBILL', 'AWB', 'HAWB', 'MAWB',
                'HOUSE AIR WAYBILL', 'MASTER AIR WAYBILL',
            ))
            # COURIER in F46A is usually about sending documents BY courier
            # (e.g., "send shipment advice by courier"), NOT about the
            # transport mode. Only treat as COURIER if NO sea/air signals.
            _f46a_has_courier = any(k in f46a_full for k in (
                'COURIER RECEIPT', 'COURIER WAYBILL',
            )) and not _f46a_has_bl and not _f46a_has_awb

            if _f46a_has_bl:
                return 'SEA'
            if _f46a_has_awb:
                return 'AIR'
            if _f46a_has_courier:
                return 'COURIER'
            return 'UNKNOWN'

        def _detect_actual_mode_for_packet(p):
            dt = (p.get('document_type', '') or '').lower()
            if any(k in dt for k in ('air waybill', 'airway bill', 'airwaybill',
                                      'awb', 'hawb', 'mawb',
                                      'house air waybill', 'master air waybill')):
                return 'AIR'
            if any(k in dt for k in ('courier receipt', 'courier waybill',
                                      'courier service', 'express envelope',
                                      'express waybill', 'express delivery')):
                return 'COURIER'
            if 'bill of lading' in dt or dt.startswith('bl ') or dt == 'bl':
                return 'SEA'
            if 'truck receipt' in dt or 'cmr' in dt or 'road waybill' in dt:
                return 'ROAD'
            if 'rail consignment' in dt or 'railway bill' in dt:
                return 'RAIL'
            return None

        expected = _expected_mode()
        actual_modes = []
        actual_packets_by_mode = {}
        for p in packets:
            m = _detect_actual_mode_for_packet(p)
            if m:
                actual_modes.append(m)
                actual_packets_by_mode.setdefault(m, []).append(p)

        # Skip the check entirely when we can't determine LC expectation
        if expected == 'UNKNOWN':
            return tasks

        # Skip when no transport document was presented at all (the missing-
        # document check will catch this elsewhere via the F46A presence pass)
        if not actual_modes:
            return tasks

        # Determine the dominant actual mode
        from collections import Counter as _Cnt
        mode_counts = _Cnt(actual_modes)
        dominant_mode = mode_counts.most_common(1)[0][0]

        # Mismatch — emit a deterministic task that step 14's runner will
        # flush straight through as a CheckResult (no VLM call).
        if dominant_mode != expected:
            example_pkt = actual_packets_by_mode[dominant_mode][0]
            example_dt = example_pkt.get('document_type', dominant_mode)
            tasks.append({
                "prompt": "__DETERMINISTIC__",
                "doc_text": "",
                "image_path": None,
                "clause_ref": "F44E/F46A",
                "condition": f"Transport mode must be {expected} per LC",
                "doc_type": example_dt,
                "check_id": check_id,
                "severity": "hard",
                "_deterministic_result": {
                    "result": "FAIL",
                    "findings": (f"LC expects {expected} transport "
                                  f"(F44E/F44F/F46A signals); documents presented "
                                  f"are {dominant_mode} ({example_dt})"),
                    "detail": f"Transport mode mismatch: LC={expected}, document={dominant_mode}",
                    "confidence": 1.0,
                },
            })
        else:
            # PASS row so the report shows we checked
            example_pkt = actual_packets_by_mode[dominant_mode][0]
            example_dt = example_pkt.get('document_type', dominant_mode)
            tasks.append({
                "prompt": "__DETERMINISTIC__",
                "doc_text": "",
                "image_path": None,
                "clause_ref": "F44E/F46A",
                "condition": f"Transport mode must be {expected} per LC",
                "doc_type": example_dt,
                "check_id": check_id,
                "severity": "hard",
                "_deterministic_result": {
                    "result": "PASS",
                    "findings": f"{dominant_mode} transport — {example_dt}",
                    "detail": f"Transport mode matches: LC={expected}, document={dominant_mode}",
                    "confidence": 1.0,
                },
            })

    elif check_id == "presentation_period":
        period_str = lc_fields.get('48', lc_fields.get('F48', ''))
        period_days = '21'
        if period_str:
            pm = re.search(r'(\d+)', str(period_str))
            if pm:
                period_days = pm.group(1)
        covers = _get_docs_by_type(packets, 'remittance', 'covering', 'cover letter', 'schedule', 'presentation')
        bls = _get_docs_by_type(packets, 'bill of lading', 'b/l', 'transport')
        bl_full_text = ''
        for bl in bls[:1]:
            bl_t = _get_doc_text(bl)
            bl_full_text = bl_t[:3000] if bl_t else ''

        for cover in covers:
            # Include stamp data — RECEIVED stamps show the actual presentation date
            _cover_stamps = cover.get('stamps', [])
            _stamp_text = ''
            if _cover_stamps:
                _stamp_text = '\n[STAMPS ON THIS DOCUMENT: ' + ', '.join(
                    s.get('text', '') for s in _cover_stamps if s.get('text')
                ) + ']'
            # Combine cover + stamps + BL text so LLM can find both dates
            combined_text = f"=== COVERING SCHEDULE / PRESENTATION DOCUMENT ===\n{_get_doc_text(cover)}{_stamp_text}\n\n=== BILL OF LADING (for shipment date) ===\n{bl_full_text}"
            tasks.append({
                "prompt": f"""You are a trade finance document examiner. Check if documents were presented within the required period after shipment.

LC Period for Presentation (F48): {period_str if period_str else 'Not specified — default 21 days per UCP 600'}
Presentation period: {period_days} days after shipment date

TWO DOCUMENTS ARE PROVIDED BELOW:
1. COVERING SCHEDULE — find the PRESENTATION DATE (date documents were sent/presented to the bank)
2. BILL OF LADING — find the SHIPMENT DATE (shipped on board date or BL issue date)

RULES:
1. The PRESENTATION DATE is when the ISSUING BANK RECEIVED the documents:
   - BEST: Look for RECEIVED stamp (e.g., "RECEIVED 19 SEP 2025") in stamps metadata
   - FALLBACK: "Presented on:", "Date of Presentation:", "Received Date:" fields
   - LAST RESORT: "DATE:" on covering schedule (this is SENDING date, not receiving)
2. The SHIPMENT DATE is on the Bill of Lading (look for "SHIPPED ON BOARD" date, or "Date of Issue" at bottom of BL)
3. Calculate: Presentation_Date must be <= (Shipment_Date + {period_days} days)
4. If documents were presented MORE than {period_days} days after shipment → "LATE PRESENTATION" → FAIL
5. If F48 is blank, UCP 600 defaults to 21 days
6. IMPORTANT: Do NOT say "shipment date missing" if the BL text below shows a date. Look carefully for dates like "30/01/2026", "JANUARY 30, 2026", "SHIPPED ON BOARD 30/01/2026" etc.
7. Extract: presentation date from cover (preferably RECEIVED stamp), shipment date from BL, and the day count""",
                "doc_text": combined_text,
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
    'latest_shipment', 'presentation_period', 'transport_mode_match',
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
    # P198cf — presentation_period is now hybrid so it uses the same
    # stamp-first / labeled-date / textual-assertion logic as lc_expiry.
    # Without this, the VLM-only path would take the Documentary
    # Remittance's unlabelled "DATE: 16MAR26" (which is the SENDING
    # date) and treat it as the PRESENTATION date, then FAIL the
    # 30-day rule — and that FAIL cross-links to an auto-FAIL on
    # lc_expiry (step14_implicit.py:2175+).
    _HYBRID_CHECKS = {'date_of_issue', 'lc_expiry', 'amount_currency',
                      'latest_shipment', 'presentation_period'}

    for check_id in _HYBRID_CHECKS:
        if not config.get(check_id, {}).get('enabled', False):
            progress_fn(f"  [{check_id}] SKIPPED (disabled)")
            continue

        if check_id == 'date_of_issue':
            lc_date = lc_fields.get('31C', lc_fields.get('F31C', ''))
            if lc_date:
                # P86: Only run the date-of-issue check if the LC
                # EXPLICITLY says documents dated prior to the credit
                # are NOT acceptable. Without that clause, there is no
                # requirement to check document dates against LC issue
                # date — documents can be dated before the LC is opened
                # (which is normal for manufacturing certificates,
                # proforma invoices, quality reports, etc.).
                lc_text_all = ' '.join(str(v) for v in lc_fields.values()).upper()
                prior_prohibited = bool(re.search(
                    r'DOCUMENT.{0,40}DATED?\s+(?:PRIOR|BEFORE|EARLIER).{0,40}'
                    r'(?:ISSUANCE|ISSUE|CREDIT|DATE\s+OF\s+(?:THIS\s+)?CREDIT).{0,30}'
                    r'(?:NOT\s+ACCEPT|NOT\s+PERMIT|NOT\s+ALLOW|UNACCEPT|PROHIBIT|REJECT)',
                    lc_text_all,
                ))
                if prior_prohibited:
                    # P198bs — Attached List / attached schedule / pallet
                    # manifest / etc. are ancillary pages of the Bill of
                    # Lading, not independently presented documents. F31C
                    # date-of-issue applies to standalone presented docs
                    # only; skipping these avoids spurious "no date" REVIEWs.
                    _F31C_EXCLUDE_TYPES = (
                        'attached list', 'attached schedule',
                        'attached manifest', 'cargo manifest page',
                        'pallet manifest', 'packing insert',
                        'container list', 'container manifest',
                        'stuffing list',
                        'documentary remittance', 'document remittance',
                        'covering letter', 'cover letter',
                        'covering schedule', 'cover schedule',
                        'blank page', 'back page', 'back cover',
                        'terms and conditions', 'terms overleaf',
                        'conditions of carriage',
                        'unknown', 'unidentified',
                    )
                    for pkt in _deduplicate_documents(packets):
                        dt = pkt.get('document_type', 'Unknown')
                        _dt_lo = (dt or '').lower().strip()
                        if any(ex in _dt_lo for ex in _F31C_EXCLUDE_TYPES):
                            progress_fn(f"  [date_of_issue] [{dt}]: SKIPPED (ancillary/non-standalone)")
                            continue
                        r = _hybrid_date_check('date_of_issue', 'F31C', lc_date, pkt,
                            f"Document must be dated on/after LC issue date ({lc_date})", 'after')
                        all_results.append(r)
                        progress_fn(f"  [date_of_issue] [{dt}]: {r.compliance} - {r.result[:50]}")
                else:
                    progress_fn(f"  [date_of_issue] SKIPPED — LC does not prohibit documents dated prior to credit")

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

        elif check_id == 'presentation_period':
            # P198cf — Hybrid presentation-period check. The period
            # rule (F48 or default 21 days) says docs must be presented
            # within N days AFTER shipment. Compute:
            #     days_elapsed = presentation_date − shipment_date
            # Presentation date MUST come from (a) a RECEIVED stamp, OR
            # (b) a LABELED presentation/received date on the cover, OR
            # (c) a textual "presented within validity" assertion. An
            # unlabelled "DATE: X" on the Documentary Remittance is the
            # SENDING date (when the negotiating bank dispatched), not
            # the receipt date — so we must NEVER use it for this check.
            f48 = str(lc_fields.get('48', lc_fields.get('F48', '')) or '')
            # Default presentation period per UCP 600 Art 14(c) = 21 days
            period_days = 21
            _pd_m = re.search(r'\b(\d{1,3})\s*DAYS?\b', f48.upper())
            if _pd_m:
                try:
                    period_days = int(_pd_m.group(1))
                except ValueError:
                    pass
            latest = lc_fields.get('44C', lc_fields.get('F44C', '')) or ''
            for cover in _get_docs_by_type(packets, 'remittance', 'covering',
                                           'cover letter', 'schedule'):
                # First try the existing helper — it handles textual
                # "presented within validity" assertion and labeled /
                # stamp dates, and returns REVIEW for unlabelled dates.
                # We reuse its decision machinery by asking a synthetic
                # "must be presented before lc_expiry" question; if that
                # comes back REVIEW (no reliable date), the period check
                # also REVIEWs.
                _pres_date_tuple = _extract_received_stamp_date(cover)
                _doc_text_up = _get_doc_text(cover).upper()
                _presentation_ok_patterns = (
                    r'DOCUMENTS?\s+(?:HAVE\s+BEEN\s+)?PRESENTED\s+(?:ON\s+TIME|'
                    r'WITHIN\s+(?:THE\s+)?(?:L/?C\s+)?(?:VALIDITY|EXPIRY|PERIOD)|'
                    r'PRIOR\s+TO\s+EXPIRY|BEFORE\s+(?:THE\s+)?EXPIRY|'
                    r'WITHIN\s+(?:THE\s+)?PRESENTATION\s+PERIOD)',
                    r'PRESENTATION\s+(?:IS\s+|WAS\s+)?MADE\s+WITHIN\s+(?:L/?C\s+)?'
                    r'(?:VALIDITY|EXPIRY\s+DATE)',
                    r'DOCUMENTS?\s+ARE\s+(?:BEING\s+)?PRESENTED\s+WITHIN\s+VALIDITY',
                    r'WITHIN\s+L/?C\s+VALIDITY\s+PERIOD',
                    r'DOCUMENTS?\s+NEGOTIATED\s+WITHIN\s+(?:THE\s+)?VALIDITY',
                )
                _doc = cover.get('document_type', 'Documentary Remittance')
                if any(re.search(p, _doc_text_up, re.IGNORECASE)
                       for p in _presentation_ok_patterns):
                    all_results.append(CheckResult(
                        check_id='presentation_period', clause_ref='F48',
                        condition=f"Documents within {period_days:03d} days of shipment",
                        document_checked=_doc,
                        findings=(
                            "Covering schedule explicitly certifies documents "
                            "were presented within LC validity / on time."
                        ),
                        result="Presented within LC validity per covering schedule",
                        compliance='PASS', severity='hard',
                    ))
                    progress_fn(f"  [presentation_period] [{_doc}]: PASS (textual assertion)")
                    continue
                if _pres_date_tuple is None:
                    # No stamp, no labeled date → REVIEW. NEVER use the
                    # sending / issue date for this check.
                    all_results.append(CheckResult(
                        check_id='presentation_period', clause_ref='F48',
                        condition=f"Documents within {period_days:03d} days of shipment",
                        document_checked=_doc,
                        findings=(
                            "Receiving / presentation date not clear — "
                            "manual review."
                        ),
                        result="Receiving / presentation date not clear — manual review",
                        compliance='REVIEW', severity='hard',
                    ))
                    progress_fn(f"  [presentation_period] [{_doc}]: REVIEW (no stamp / labeled date)")
                    continue
                # We have a trustworthy presentation date. Compute period.
                pres_date, pres_date_raw = _pres_date_tuple
                # Shipment date: first BL on-board date we can find
                ship_date = None
                for bl in _get_docs_by_type(packets, 'bill of lading', 'b/l'):
                    _sob = (bl.get('extracted_fields', {}) or {}).get('shipped_on_board_date', '')
                    if _sob:
                        ship_date = _parse_date(_sob)
                    if not ship_date:
                        ship_date = _parse_date(bl.get('document_date', ''))
                    if ship_date:
                        break
                if ship_date is None:
                    all_results.append(CheckResult(
                        check_id='presentation_period', clause_ref='F48',
                        condition=f"Documents within {period_days:03d} days of shipment",
                        document_checked=_doc,
                        findings=(
                            f"Presentation date {pres_date:%Y-%m-%d} known but "
                            f"shipment date could not be parsed from BL — "
                            f"manual review."
                        ),
                        result="Shipment date unclear — manual review",
                        compliance='REVIEW', severity='hard',
                    ))
                    continue
                days_elapsed = (pres_date - ship_date).days
                within = 0 <= days_elapsed <= period_days
                if within:
                    all_results.append(CheckResult(
                        check_id='presentation_period', clause_ref='F48',
                        condition=f"Documents within {period_days:03d} days of shipment",
                        document_checked=_doc,
                        findings=(
                            f"Presented {pres_date:%Y-%m-%d} "
                            f"({days_elapsed} day(s) after shipment "
                            f"{ship_date:%Y-%m-%d}) — within {period_days}-day limit."
                        ),
                        result=(
                            f"Presented {days_elapsed} days after shipment — "
                            f"within {period_days}-day limit"
                        ),
                        compliance='PASS', severity='hard',
                    ))
                    progress_fn(f"  [presentation_period] [{_doc}]: PASS ({days_elapsed}d ≤ {period_days})")
                else:
                    all_results.append(CheckResult(
                        check_id='presentation_period', clause_ref='F48',
                        condition=f"Documents within {period_days:03d} days of shipment",
                        document_checked=_doc,
                        findings=(
                            f"Presented {pres_date:%Y-%m-%d} "
                            f"({days_elapsed} day(s) after shipment "
                            f"{ship_date:%Y-%m-%d}) — exceeds {period_days}-day limit."
                        ),
                        result=(
                            f"Presentation {days_elapsed} days after shipment "
                            f"— exceeds {period_days}-day limit"
                        ),
                        compliance='FAIL', severity='hard',
                    ))
                    progress_fn(f"  [presentation_period] [{_doc}]: FAIL ({days_elapsed}d > {period_days})")

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
                # P69: Helper — run the amount check across all matched
                # packets of a given doc-class, but emit ONLY ONE row to
                # the report. Avoids the duplicate-row bug where two
                # cover-schedule packets (or two drafts) produced both a
                # PASS row and a "Amount not extractable" REVIEW row for
                # what the user sees as the same logical document.
                #
                # Selection priority (highest first):
                #   1. PASS  — fully verified, amount matches
                #   2. FAIL  — fully verified, amount differs
                #   3. REVIEW — fallback when nothing better is available
                # Within the same priority bucket, the first packet wins.
                def _emit_one(matched_pkts, check_type, label):
                    if not matched_pkts:
                        return None
                    _best = None
                    _best_rank = 99
                    _rank = {'PASS': 0, 'FAIL': 1, 'REVIEW': 2}
                    for _p in matched_pkts:
                        _r = _hybrid_amount_check(
                            lc_amount, lc_currency, tol_plus, tol_minus, _p,
                            'amount_currency', check_type, _inv_amounts_str_local,
                        )
                        _rk = _rank.get(str(_r.compliance).upper(), 3)
                        if _rk < _best_rank:
                            _best = _r
                            _best_rank = _rk
                            if _best_rank == 0:
                                break  # PASS — stop searching
                    if _best is not None:
                        all_results.append(_best)
                        progress_fn(f"  [amount_currency] [{label}]: {_best.compliance} - {_best.result[:50]}")
                    return _best

                # Get invoice amounts — track numeric total for draft/cover comparison
                invoices = _get_docs_by_type(packets, 'commercial invoice', 'invoice')
                _inv_total_numeric = 0.0
                _inv_amounts_str_local = ''  # available for the helper above
                for inv in invoices:
                    r = _hybrid_amount_check(lc_amount, lc_currency, tol_plus, tol_minus, inv,
                        'amount_currency', 'invoice_vs_lc', '')
                    all_results.append(r)
                    # Extract numeric amount from findings
                    _inv_amt = _parse_amount(r.findings)
                    if _inv_amt:
                        _inv_total_numeric = max(_inv_total_numeric, _inv_amt)  # Use largest invoice (they're usually the same or one is the total)
                    progress_fn(f"  [amount_currency] [{inv.get('document_type','')}]: {r.compliance} - {r.result[:50]}")

                _inv_amounts_str_local = f"{lc_currency} {_inv_total_numeric:,.2f}" if _inv_total_numeric else ''
                progress_fn(f"  [amount_currency] Invoice total for draft/cover comparison: {_inv_amounts_str_local}")

                # Draft vs invoice total — emit ONE row
                _emit_one(
                    _get_docs_by_type(packets, 'draft', 'bill of exchange'),
                    'draft_vs_invoice', 'Draft',
                )

                # Cover vs invoice total — emit ONE row
                _emit_one(
                    _get_docs_by_type(packets, 'remittance', 'covering', 'cover', 'schedule'),
                    'cover_vs_invoice', 'Cover Schedule',
                )

    # ── P91: BL Originals Check ──
    # If LC requires original BL ("FULL SET", "ORIGINAL", or just "BILL OF
    # LADING" without "COPY") and ALL submitted BLs are marked NON-NEGOTIABLE
    # or COPY, flag as a hard discrepancy.
    if config.get('doc_originals_copies', {}).get('enabled', True):
        bl_packets = _get_docs_by_type(packets, 'bill of lading')
        if bl_packets:
            # Check if any BL is an original
            has_original_bl = False
            all_bl_copy_statuses = []
            for bl in bl_packets:
                copy_status = str(bl.get('copy_status', bl.get('copy_label', ''))).upper()
                doc_type_upper = str(bl.get('document_type', '')).upper()
                text_upper = str(bl.get('refined_text', bl.get('cleaned_text', bl.get('text', '')))).upper()[:500]

                all_bl_copy_statuses.append(copy_status or doc_type_upper)

                # Check if this BL is an original
                is_non_negotiable = ('NON-NEGOTIABLE' in copy_status or 'NON NEGOTIABLE' in copy_status or
                                     'NON-NEGOTIABLE' in doc_type_upper or 'NON NEGOTIABLE' in doc_type_upper or
                                     'NON-NEGOTIABLE' in text_upper or 'COPY NON-NEGOTIABLE' in text_upper or
                                     'COPY NOT NEGOTIABLE' in text_upper)
                is_copy = ('COPY' in copy_status and 'ORIGINAL' not in copy_status)

                if not is_non_negotiable and not is_copy:
                    has_original_bl = True

                # Also check if it's explicitly marked ORIGINAL
                if 'ORIGINAL' in copy_status and 'NON' not in copy_status:
                    has_original_bl = True

            # Check if LC requires originals (almost always does for BL)
            f46a = str(lc_fields.get('46A', lc_fields.get('F46A', ''))).upper()
            lc_requires_original = ('FULL SET' in f46a or 'ORIGINAL' in f46a or
                                    'BILL OF LADING' in f46a)
            # "COPY OF B/L" or "COPY OF BILL OF LADING" means copy is acceptable
            copy_acceptable = bool(re.search(r'COPY\s+OF\s+(?:B/?L|BILL\s+OF\s+LADING)', f46a))

            # Find the specific F46A clause number that mentions BL
            _bl_clause_ref = 'F46A'
            _f46a_clauses = str(lc_fields.get('46A', lc_fields.get('F46A', ''))).split('\n')
            for _ci, _cl in enumerate(_f46a_clauses, 1):
                if 'BILL OF LADING' in _cl.upper() or 'B/L' in _cl.upper():
                    _bl_clause_ref = f'46A-{_ci}'
                    break

            if not has_original_bl and lc_requires_original and not copy_acceptable:
                all_results.append(CheckResult(
                    check_id='doc_originals_copies',
                    clause_ref=_bl_clause_ref,
                    condition='Full set of original Bill of Lading must be presented',
                    document_checked='Bill of Lading',
                    findings=f"All BLs are NON-NEGOTIABLE copies: {', '.join(all_bl_copy_statuses[:3])}",
                    result='BL IS COPY NON-NEGOTIABLE. No original Bill of Lading presented. LC requires original Bill of Lading.',
                    compliance='FAIL',
                    severity='hard',
                ))
                progress_fn(f"  [doc_originals_copies] FAIL - No original BL found ({len(bl_packets)} copies)")
            elif has_original_bl:
                all_results.append(CheckResult(
                    check_id='doc_originals_copies',
                    clause_ref=_bl_clause_ref,
                    condition='Original Bill of Lading must be presented',
                    document_checked='Bill of Lading',
                    findings=f"Original BL found in submission ({len(bl_packets)} BL packets)",
                    result='Original BL present',
                    compliance='PASS',
                    severity='hard',
                ))
                progress_fn(f"  [doc_originals_copies] PASS - Original BL found")

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
        # ── Short-circuit for deterministic checks ──
        # Some builders (e.g. transport_mode_match) compute the result in
        # pure Python and embed it on the task as `_deterministic_result`.
        # The runner skips the VLM call entirely and emits the precomputed
        # result. This makes those checks instant and immune to model
        # hallucinations.
        if task.get('prompt') == '__DETERMINISTIC__' and task.get('_deterministic_result'):
            det = task['_deterministic_result']
            compliance = (det.get('result') or 'REVIEW').upper()
            if compliance not in ('PASS', 'FAIL', 'REVIEW'):
                compliance = 'REVIEW'
            findings = det.get('findings', '')
            detail = det.get('detail', '')
            confidence = float(det.get('confidence', 1.0))
            progress_fn(f"  [{task['check_id']}] [{task['doc_type']}]: {compliance} (deterministic) - {detail or findings[:60]}")
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
