"""
Step 7 -- Final LC Clause and Requirement Extraction
=====================================================
Takes the consolidated Final LC from Step 6 and extracts:
  1. ALL clauses from every LC field (consolidated_fields + clauses dict)
  2. REQUIRED DOCUMENTS from F46A/F46B with detailed attributes
  3. Standalone fields (DC number, dates, amounts, etc.)

Uses rule-based regex extraction only (no VLM needed -- clauses are text).

INPUT:  Step 6 output (final_lc with consolidated_fields and clauses)
OUTPUT: all_clauses[], required_documents[], standalone_fields{}
"""

import json
import sys as _sys
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import os
import re
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from pathlib import Path


# ── Dataclasses ──

@dataclass
class LCClause:
    """A single clause extracted from the Final LC."""
    field_tag: str                      # e.g. "F46A", "F47A", "F31D"
    field_name: str                     # e.g. "Documents Required"
    clause_number: Optional[int] = None # 1-based position within multi-clause fields
    clause_text: str = ""
    source_page: Optional[int] = None
    source_step: int = 7
    confidence: float = 1.0
    ambiguity_flag: bool = False
    ambiguity_notes: str = ""


@dataclass
class RequiredDocument:
    """A document required by the LC (extracted from F46A/F46B clauses)."""
    document_name: str
    clause_reference: str = ""          # e.g. "F46A clause 3"
    required_or_conditional: str = "required"  # "required" | "conditional"
    originals_count: int = 0
    copies_count: int = 0
    signature_required: bool = False
    issuer_requirements: str = ""
    content_requirements: List[str] = field(default_factory=list)
    source_clause_text: str = ""
    source_page: Optional[int] = None
    confidence: float = 1.0
    ambiguity_flag: bool = False
    ambiguity_notes: str = ""


@dataclass
class StructuredLC:
    """Complete structured LC output from Step 7."""
    all_clauses: List[dict] = field(default_factory=list)
    required_documents: List[dict] = field(default_factory=list)
    standalone_fields: Dict[str, Any] = field(default_factory=dict)
    total_clauses: int = 0
    total_required_docs: int = 0


# ── SWIFT Field Tag Reference ──

_FIELD_NAMES = {
    "F20": "Documentary Credit Number",
    "F23": "Issuing Bank Reference",
    "F27": "Sequence of Total",
    "F31C": "Date of Issue",
    "F31D": "Date and Place of Expiry",
    "F32B": "Currency Code, Amount",
    "F39A": "Percentage Credit Amount Tolerance",
    "F39B": "Maximum Credit Amount",
    "F40A": "Form of Documentary Credit",
    "F40E": "Applicable Rules",
    "F41A": "Available With ... By ...",
    "F41D": "Available With ... By ... (Name/Address)",
    "F42A": "Drawee",
    "F42C": "Drafts at ...",
    "F42M": "Mixed Payment Details",
    "F42P": "Deferred Payment Details",
    "F43P": "Partial Shipments",
    "F43T": "Transshipment",
    "F44A": "Place of Taking in Charge / Dispatch",
    "F44B": "Place of Final Destination / Delivery",
    "F44C": "Latest Date of Shipment",
    "F44D": "Shipment Period",
    "F44E": "Port of Loading / Airport of Departure",
    "F44F": "Port of Discharge / Airport of Destination",
    "F45A": "Description of Goods and/or Services",
    "F45B": "Description of Goods (Continuation)",
    "F46A": "Documents Required",
    "F46B": "Documents Required (Continuation)",
    "F47A": "Additional Conditions",
    "F47B": "Additional Conditions (Continuation)",
    "F48": "Period for Presentation",
    "F49": "Confirmation Instructions",
    "F50": "Applicant",
    "F51A": "Applicant Bank",
    "F52A": "Issuing Bank",
    "F53A": "Reimbursing Bank",
    "F57A": "Advising Through Bank",
    "F59": "Beneficiary",
    "F71B": "Charges",
    "F71D": "Charges (Details)",
    "F72": "Sender to Receiver Information",
    "F77A": "Narrative",
    "F78": "Instructions to Paying/Accepting/Negotiating Bank",
    "F79": "Narrative for Amendments",
}

# Fields that contain clause lists -- split into individual clauses
# Both F-prefixed (old format) and bare tags (new format) are accepted
_CLAUSE_FIELDS = {"F46A", "F46B", "F47A", "F47B", "F45A", "F45B", "F77A", "F78", "F79", "F72",
                  "46A", "46B", "47A", "47B", "45A", "45B", "77A", "78", "79", "72"}

# Internal metadata fields to skip
_SKIP_FIELDS = {
    "swift_format", "mt_number", "source_file", "processing_date",
    "ocr_method", "extraction_method", "page_count",
}

# ── Copy/Original Count Patterns ──

_COPY_PATTERNS = [
    # "FULL SET" = 3 originals (trade finance standard: 3/3)
    (r'FULL\s+SET', lambda m: (3, 0)),
    # "3/3 ORIGINAL"
    (r'(\d+)\s*/\s*(\d+)\s*ORIGINAL', lambda m: (int(m.group(1)), 0)),
    # "IN 3 ORIGINALS"
    (r'IN\s+(\d+)\s+ORIGINAL', lambda m: (int(m.group(1)), 0)),
    # "3 ORIGINALS"
    (r'(\d+)\s+ORIGINAL', lambda m: (int(m.group(1)), 0)),
    # "IN OCTUPLICATE" = 8 copies
    (r'IN\s+OCTUPLICATE', lambda m: (0, 8)),
    # "IN TRIPLICATE" = 3 copies
    (r'IN\s+TRIPLICATE', lambda m: (0, 3)),
    # "IN DUPLICATE" = 2 copies
    (r'IN\s+DUPLICATE', lambda m: (0, 2)),
    # "IN QUADRUPLICATE" = 4 copies
    (r'IN\s+QUADRUPLICATE', lambda m: (0, 4)),
    # "IN QUINTUPLICATE" = 5 copies
    (r'IN\s+QUINTUPLICATE', lambda m: (0, 5)),
    # "IN SEXTUPLICATE" = 6 copies
    (r'IN\s+SEXTUPLICATE', lambda m: (0, 6)),
    # "IN SEPTUPLICATE" = 7 copies
    (r'IN\s+SEPTUPLICATE', lambda m: (0, 7)),
    # Bare "ORIGINAL" = 1
    (r'\bORIGINAL\b', lambda m: (1, 0)),
    # "3 COPIES"
    (r'(\d+)\s+COP(?:Y|IES)', lambda m: (0, int(m.group(1)))),
    # Bare "COPY" / "COPIES"
    (r'\bCOP(?:Y|IES)\b', lambda m: (0, 1)),
]


# ── Known Document Type Patterns ──

_DOC_TYPE_PATTERNS = [
    (r'COMMERCIAL\s+INVOICE', "Commercial Invoice"),
    (r'PROFORMA\s+INVOICE', "Proforma Invoice"),
    (r'BILL\s+OF\s+LADING', "Bill of Lading"),
    (r'AIRWAY\s*BILL', "Airway Bill"),
    (r'INSURANCE\s+(?:POLICY|CERTIFICATE)', "Insurance Policy/Certificate"),
    (r'CERTIFICATE\s+OF\s+ORIGIN', "Certificate of Origin"),
    (r'PACKING\s+LIST', "Packing List"),
    (r'WEIGHT\s+(?:LIST|CERTIFICATE|NOTE)', "Weight List"),
    (r'INSPECTION\s+CERTIFICATE', "Inspection Certificate"),
    (r'BENEFICIARY\s*(?:\'S)?\s*CERTIFICATE', "Beneficiary Certificate"),
    (r'(?:DRAFT|BILL\s+OF\s+EXCHANGE)', "Draft"),
    (r'SHIPPING\s+ADVICE', "Shipping Advice"),
    (r'FUMIGATION\s+CERTIFICATE', "Fumigation Certificate"),
    (r'PHYTOSANITARY\s+CERTIFICATE', "Phytosanitary Certificate"),
    (r'HEALTH\s+CERTIFICATE', "Health Certificate"),
    (r'QUALITY\s+CERTIFICATE', "Quality Certificate"),
    (r'ANALYSIS\s+CERTIFICATE', "Analysis Certificate"),
    (r'SURVEY\s+(?:REPORT|CERTIFICATE)', "Survey Report"),
    (r'CLEAN\s+ON\s+BOARD', "Bill of Lading"),
    (r'MARINE\s+CARGO\s+INSURANCE', "Insurance Policy/Certificate"),
]


# ── Helper Functions ──

def _split_into_clauses(text: str) -> List[str]:
    """Split a multi-clause field into individual clauses."""
    if not text or not isinstance(text, str):
        return [str(text)] if text else []

    # Handle JSON-encoded lists
    if text.startswith('[') and text.endswith(']'):
        try:
            items = json.loads(text)
            if isinstance(items, list):
                return [str(i) for i in items if str(i).strip()]
        except (json.JSONDecodeError, ValueError):
            pass

    # P198at — Numbered splitting with BOTH newline-separated AND
    # inline-separated markers. Many LCs come through as one long
    # line where the numbered markers "1)", "2)", ... appear INLINE
    # separated only by spaces (after step06 text consolidation
    # reflows newlines). Others have a partial mix. Strategy:
    #   1) Split on newline-preceded markers first.
    #   2) For each resulting chunk, also split on INLINE markers.
    #   3) Flatten and validate that chunks begin with "N)" / "N."
    #      and the numbers are loosely monotonic increasing.
    # Guarded against splitting inside numeric IDs like
    # "NTN 3075811-4" or dates like "2025/07/11" by requiring the
    # marker to be preceded by whitespace or period AND followed by
    # whitespace + uppercase letter / "(".

    def _inline_split(chunk):
        # Split on "N)" or "N." preceded by whitespace/period/comma,
        # followed EITHER by whitespace+uppercase OR directly by an
        # uppercase letter / "(". Real LCs sometimes come through
        # with NO space between marker and content ("8)BENEFICIARY").
        inline = re.split(
            r'(?<=[\s\.\,])(?=\d{1,2}[\)\.]\s*[A-Z\(])',
            chunk,
        )
        parts = [c.strip() for c in inline if c.strip()]
        if len(parts) <= 1:
            return [chunk.strip()] if chunk.strip() else []
        _is_marker = lambda c: bool(re.match(r'^\d{1,2}[\)\.]', c))
        if not all(_is_marker(c) for c in parts[1:]):
            return [chunk.strip()] if chunk.strip() else []
        def _num(c):
            m = re.match(r'^(\d{1,2})', c)
            return int(m.group(1)) if m else -1
        nums = [_num(c) for c in parts if _is_marker(c)]
        if (nums and len(nums) >= 2 and all(n > 0 for n in nums) and
                all(nums[i+1] >= nums[i] for i in range(len(nums)-1))):
            return parts
        return [chunk.strip()] if chunk.strip() else []

    # Pass 1: newline-separated numbered markers (with or without
    # space after the marker — "10)" / "10) ")
    numbered = re.split(r'\n(?=\d+[\.\)])', text)
    numbered = [c.strip() for c in numbered if c.strip()]

    # Pass 2: expand each chunk via inline-marker splitting. This
    # catches LCs where clauses "1)", "2)", ..., "13)" are all on
    # one line separated by spaces, as well as partially reflowed
    # LCs where only some clauses have a preceding newline.
    expanded = []
    for ch in numbered:
        expanded.extend(_inline_split(ch))

    # Validate combined: if we got more than 1 chunk and they look
    # like a proper numbered clause list, use it.
    if len(expanded) > 1:
        _is_marker = lambda c: bool(re.match(r'^\d{1,2}[\)\.]', c))
        if all(_is_marker(c) for c in expanded):
            return expanded
        # Accept even if the very first chunk is a preamble (no
        # leading marker) as long as the rest are markers.
        if all(_is_marker(c) for c in expanded[1:]):
            return expanded

    # Try letter splitting: "A.", "B.", etc.
    lettered = re.split(r'\n(?=[A-Z][\.\)]\s)', text)
    if len(lettered) > 1:
        return [c.strip() for c in lettered if c.strip()]

    # Try dash splitting
    dashed = re.split(r'\n(?=-\s)', text)
    if len(dashed) > 1:
        return [c.strip() for c in dashed if c.strip()]

    # Try splitting on lines that start with +
    plus_split = re.split(r'\n(?=\+\s)', text)
    if len(plus_split) > 1:
        return [c.strip() for c in plus_split if c.strip()]

    # Single clause
    if text.strip():
        return [text.strip()]
    return []


def _extract_copy_counts(clause_text: str) -> tuple:
    """Extract (originals, copies) from clause text."""
    upper = clause_text.upper()
    originals = 0
    copies = 0

    for pattern, extractor in _COPY_PATTERNS:
        m = re.search(pattern, upper)
        if m:
            o, c = extractor(m)
            originals = max(originals, o)
            copies = max(copies, c)

    # "3/3" pattern with ORIGINAL
    m = re.search(r'(\d+)\s*/\s*(\d+)', upper)
    if m and 'ORIGINAL' in upper:
        originals = max(originals, int(m.group(1)))

    return originals, copies


def _detect_signature_required(clause_text: str) -> bool:
    """Check if the clause requires a signature or authentication."""
    upper = clause_text.upper()
    sig_patterns = [
        'MANUALLY SIGNED', 'SIGNED', 'SIGN', 'SIGNATURE', 'COUNTERSIGNED',
        'ENDORSED', 'AUTHENTICATED', 'CERTIFIED BY',
        'STAMPED AND SIGNED', 'DULY SIGNED',
    ]
    return any(p in upper for p in sig_patterns)


def _extract_issuer(clause_text: str) -> str:
    """Extract who must issue the document."""
    patterns = [
        r'(?:TO\s+BE\s+)?ISSUED\s+BY\s+(.+?)(?:\.|,|\n|$)',
        r'SIGNED\s+BY\s+(.+?)(?:\.|,|\n|$)',
        r'CERTIFIED\s+BY\s+(.+?)(?:\.|,|\n|$)',
        r'FROM\s+(.+?)(?:\.|,|\n|$)',
        r'BY\s+THE\s+(.+?)(?:\.|,|\n|$)',
    ]
    for pat in patterns:
        m = re.search(pat, clause_text, re.IGNORECASE)
        if m:
            issuer = m.group(1).strip()
            if 5 < len(issuer) < 200:
                return issuer
    return ""


def _extract_content_requirements(clause_text: str) -> List[str]:
    """Extract content requirements (INDICATING, SHOWING, CERTIFYING, etc.)."""
    reqs = []
    patterns = [
        r'(?:INDICATING|SHOWING|STATING|EVIDENCING|CERTIFYING|CONFIRMING|BEARING)\s+(.+?)(?:\.|$)',
        r'MUST\s+(?:SHOW|STATE|INDICATE|CONTAIN|BEAR|MENTION)\s+(.+?)(?:\.|$)',
        r'SHALL\s+(?:SHOW|STATE|INDICATE|CONTAIN|BEAR|MENTION)\s+(.+?)(?:\.|$)',
        r'MARKED\s+(?:AS|WITH)\s+(.+?)(?:\.|$)',
        r'NOTIFY(?:ING)?\s+PARTY[:\s]+(.+?)(?:\.|$)',
        r'CONSIGNED\s+TO\s+(.+?)(?:\.|$)',
    ]
    for pat in patterns:
        for m in re.finditer(pat, clause_text, re.IGNORECASE):
            req = m.group(1).strip()
            if len(req) > 3:
                reqs.append(req)
    return reqs


def _detect_conditional(clause_text: str) -> str:
    """Detect if a document requirement is conditional vs always required."""
    upper = clause_text.upper()
    cond_keywords = [
        'IF APPLICABLE', 'IF REQUIRED', 'IF ANY', 'WHEN APPLICABLE',
        'IN CASE OF', 'PROVIDED THAT', 'UNLESS', 'WHERE APPLICABLE',
    ]
    for kw in cond_keywords:
        if kw in upper:
            return "conditional"
    return "required"


def _identify_document_from_clause(clause_text: str) -> Optional[str]:
    """Identify document type name from clause text using VLM.

    Sends the clause to Qwen VLM and asks for the standard trade finance
    document name. VLM understands that:
    'FULL SET OF SHIPPED ON BOARD OCEAN ORIGINAL BILLS OF LADING...' = Bill of Lading
    'INSURANCE COVERED BY APPLICANT, BENEFICIARY SHIPMENT ADVICE...' = Shipment Advice
    'CERTIFICATE FROM SHIPPING COMPANY OR THEIR AUTHORIZED AGENTS...' = Shipping Company Certificate
    """
    try:
        import requests
        from config.settings import QWEN_VLM_URL, QWEN_VLM_MODEL
        prompt = (
            "This is a clause from an LC (Letter of Credit) field F46A (Documents Required). "
            "It describes ONE specific document that the beneficiary must present.\n\n"
            "Clause: %s\n\n"
            "What is the SHORT standard trade finance document name for this clause?\n"
            "Examples: Bill of Lading, Commercial Invoice, Shipment Advice, Weight Certificate, "
            "Quality Certificate, Packing List, Certificate of Origin, Insurance Certificate, "
            "Draft Bill of Exchange, Phytosanitary Certificate, Fumigation Certificate, "
            "Agents Certificate, Shipping Company Certificate, Beneficiary Certificate, "
            "Inspection Certificate, Document Remittance\n\n"
            "Return ONLY the document name. Nothing else. No explanation."
        ) % clause_text[:800]
        resp = requests.post(QWEN_VLM_URL, json={
            "model": QWEN_VLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 30, "temperature": 0.1
        }, timeout=None)
        if resp.status_code == 200:
            result = resp.json()
            name = result.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
            # Clean: remove quotes, markdown, periods
            name = name.strip('"\'`*. ')
            if name and len(name) < 60 and not name.startswith('{'):
                return name
    except Exception:
        pass
    return None


# ── Main Run Function ──

def run(step6_result: dict, output_dir: str = None, progress_callback=None) -> dict:
    """
    Execute Step 7: Extract clauses and required documents from Final LC.

    Args:
        step6_result: Output from Step 6 containing 'final_lc' dict
        output_dir: Directory to save results
        progress_callback: Optional callback for progress updates

    Returns:
        dict with 'structured_lc', 'all_clauses', 'required_documents',
        'standalone_fields', 'elapsed_seconds'
    """
    def _progress(msg):
        if progress_callback:
            progress_callback("[Step 7] %s" % msg)
        print("[Step 7] %s" % msg)

    start_time = time.time()

    # ── Get Final LC data ──
    _flc_obj = step6_result.get('final_lc', step6_result)
    if isinstance(_flc_obj, dict) and 'consolidated_fields' in _flc_obj:
        final_lc = dict(_flc_obj.get('consolidated_fields', {}))
        for _tag, _cls_list in _flc_obj.get('clauses', {}).items():
            if isinstance(_cls_list, list) and _cls_list:
                final_lc[_tag] = _cls_list
    elif isinstance(_flc_obj, dict):
        final_lc = _flc_obj
    else:
        final_lc = step6_result.get('consolidated_fields', {})

    if not final_lc:
        return {
            'error': 'No Final LC data found in Step 6 output',
            'structured_lc': asdict(StructuredLC()),
            'all_clauses': [],
            'required_documents': [],
            'standalone_fields': {},
            'elapsed_seconds': 0,
        }

    _progress("Processing Final LC with %d fields..." % len(final_lc))

    all_clauses = []
    standalone_fields = {}
    doc_clause_texts = []  # clauses from F46A/F46B for document extraction

    # ── Extract all clauses and standalone fields ──
    for field_tag, field_value in final_lc.items():
        if field_tag in _SKIP_FIELDS:
            continue

        field_name = _FIELD_NAMES.get(field_tag, field_tag.replace('_', ' ').title())

        # Handle list values (already split into clauses by Step 6)
        if isinstance(field_value, list):
            for idx, item in enumerate(field_value):
                if isinstance(item, dict):
                    clause_text = item.get('text', item.get('clause_text', json.dumps(item)))
                    source_page = item.get('source_page', None)
                else:
                    clause_text = str(item)
                    source_page = None

                clause = LCClause(
                    field_tag=field_tag,
                    field_name=field_name,
                    clause_number=idx + 1,
                    clause_text=clause_text,
                    source_page=source_page,
                )
                all_clauses.append(asdict(clause))

                if field_tag in ("F46A", "F46B", "46A", "46B"):
                    doc_clause_texts.append(clause_text)
            continue

        # Convert to string
        text_value = str(field_value) if field_value is not None else ""

        if field_tag in _CLAUSE_FIELDS and len(text_value) > 50:
            sub_clauses = _split_into_clauses(text_value)
            for idx, clause_text in enumerate(sub_clauses):
                clause = LCClause(
                    field_tag=field_tag,
                    field_name=field_name,
                    clause_number=idx + 1,
                    clause_text=clause_text,
                )
                all_clauses.append(asdict(clause))

                if field_tag in ("F46A", "F46B", "46A", "46B"):
                    doc_clause_texts.append(clause_text)
        else:
            # Standalone field
            standalone_fields[field_tag] = {
                'field_name': field_name,
                'value': text_value,
                'source_step': 7,
            }
            clause = LCClause(
                field_tag=field_tag,
                field_name=field_name,
                clause_number=1,
                clause_text=text_value,
            )
            all_clauses.append(asdict(clause))

            # Short F46A/F46B still counts
            if field_tag in ("F46A", "F46B", "46A", "46B") and text_value.strip():
                doc_clause_texts.append(text_value)

    _progress("Extracted %d clauses, %d standalone fields" % (len(all_clauses), len(standalone_fields)))

    # ── Extract Required Documents from F46A/F46B ──
    required_documents = []

    if doc_clause_texts:
        _progress("Extracting required documents from %d F46A/F46B clauses..." % len(doc_clause_texts))

        for idx, clause_text in enumerate(doc_clause_texts):
            doc_name = _identify_document_from_clause(clause_text)
            if not doc_name:
                first_line = clause_text.split('\n')[0].strip()
                if len(first_line) > 5:
                    doc_name = first_line
                else:
                    doc_name = "Document (Clause %d)" % (idx + 1)

            originals, copies = _extract_copy_counts(clause_text)
            sig_required = _detect_signature_required(clause_text)
            issuer = _extract_issuer(clause_text)
            content_reqs = _extract_content_requirements(clause_text)
            req_or_cond = _detect_conditional(clause_text)

            doc = RequiredDocument(
                document_name=doc_name,
                clause_reference="F46A clause %d" % (idx + 1),
                required_or_conditional=req_or_cond,
                originals_count=originals,
                copies_count=copies,
                signature_required=sig_required,
                issuer_requirements=issuer,
                content_requirements=content_reqs,
                source_clause_text=clause_text,
                confidence=0.90,
            )
            required_documents.append(asdict(doc))

    _progress("Total required documents from F46A: %d" % len(required_documents))
    for _rd in required_documents:
        _progress("  Required: %s (orig=%d, copies=%d, sig=%s)" % (
            _rd.get('document_name', '?'), _rd.get('originals_count', 0),
            _rd.get('copies_count', 0), _rd.get('signature_required', False)))

    # ── Check F47A for additional document requirements ──
    f47a_clauses = [c for c in all_clauses if c.get('field_tag') in ('F47A', 'F47B', '47A', '47B')]
    for clause in f47a_clauses:
        clause_text = clause.get('clause_text', '')
        doc_name = _identify_document_from_clause(clause_text)
        if doc_name:
            originals, copies = _extract_copy_counts(clause_text)
            doc = RequiredDocument(
                document_name=doc_name,
                clause_reference="F47A clause %s" % clause.get('clause_number', '?'),
                required_or_conditional=_detect_conditional(clause_text),
                originals_count=originals,
                copies_count=copies,
                signature_required=_detect_signature_required(clause_text),
                issuer_requirements=_extract_issuer(clause_text),
                content_requirements=_extract_content_requirements(clause_text),
                source_clause_text=clause_text,
                confidence=0.80,
                ambiguity_flag=True,
                ambiguity_notes="Extracted from F47A (Additional Conditions), not F46A",
            )
            existing_names = {rd['document_name'].upper() for rd in required_documents}
            if doc_name.upper() not in existing_names:
                required_documents.append(asdict(doc))

    _progress("Total required documents (incl. F47A): %d" % len(required_documents))
    _progress("Required document list:")
    for _idx, _rd in enumerate(required_documents):
        _progress("  %d. %s (orig=%d, copies=%d)" % (
            _idx + 1, _rd.get('document_name', '?'),
            _rd.get('originals_count', 0), _rd.get('copies_count', 0)))

    # ── Build StructuredLC ──
    structured_lc = StructuredLC(
        all_clauses=all_clauses,
        required_documents=required_documents,
        standalone_fields=standalone_fields,
        total_clauses=len(all_clauses),
        total_required_docs=len(required_documents),
    )

    elapsed = time.time() - start_time

    # Save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        result_file = os.path.join(output_dir, 'step07_result.json')
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({
                'step': 7,
                'step_name': 'Final LC Clause and Requirement Extraction',
                'total_clauses': len(all_clauses),
                'total_required_docs': len(required_documents),
                'total_standalone_fields': len(standalone_fields),
                'elapsed_seconds': round(elapsed, 2),
                'structured_lc': asdict(structured_lc),
            }, f, indent=2, ensure_ascii=False)

    _progress("Step 7 complete: %d clauses, %d required docs in %.1fs" % (
        len(all_clauses), len(required_documents), elapsed))

    return {
        'structured_lc': asdict(structured_lc),
        'all_clauses': all_clauses,
        'required_documents': required_documents,
        'standalone_fields': standalone_fields,
        'elapsed_seconds': round(elapsed, 2),
    }


if __name__ == '__main__':
    import sys as _main_sys
    if len(_main_sys.argv) < 2:
        print("Usage: python step07_clause_extraction.py <step06_result.json>")
        _main_sys.exit(1)
    with open(_main_sys.argv[1], 'r', encoding='utf-8') as f:
        step6 = json.load(f)
    result = run(step6, output_dir=os.path.dirname(_main_sys.argv[1]))
    print("\nResult: %ss" % result['elapsed_seconds'])
    print("  Clauses: %d" % len(result['all_clauses']))
    print("  Required docs: %d" % len(result['required_documents']))
    for rd in result['required_documents']:
        print("    - %s (%dO/%dC)" % (rd['document_name'], rd['originals_count'], rd['copies_count']))
