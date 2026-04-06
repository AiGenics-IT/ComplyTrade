"""
Step 12 — Clause-by-Clause Condition Decomposition
=====================================================
Takes each LC clause from Step 7's StructuredLC and sends it to the Qwen VLM
to decompose into individual checkable conditions.

PURPOSE:
    An LC clause like "FULL SET OF CLEAN ON BOARD BILLS OF LADING MADE OUT TO
    ORDER OF ISSUING BANK MARKED FREIGHT PREPAID NOTIFY APPLICANT" contains
    MULTIPLE independent conditions:
      1. Full set (3/3 originals) of Bills of Lading
      2. Clean (no adverse clauses on the BL)
      3. On Board (goods loaded, not just received for shipment)
      4. Made out to order of Issuing Bank (consignee field)
      5. Marked Freight Prepaid
      6. Notify party = Applicant

    This step breaks each clause into its individual conditions so that each
    can be independently verified against the actual shipping documents.

    If one condition applies to multiple documents (e.g., "HS CODE MUST APPEAR
    ON BL AND INVOICES"), separate rows are created for each document.

INPUTS:
    - Step 7 StructuredLC (consolidated_fields with clause data)
    - Clause fields processed: 46A, 46B, 47A, 47B, 45A, 45B, 46C, 77A, 78, 79, 72

OUTPUTS:
    - List of DecomposedClause objects, each containing:
      - clause_ref (e.g., "46A-1", "47A-3")
      - original clause text
      - List of Condition objects with: condition_text, document_to_check, look_for_value

TRADE FINANCE CONTEXT:
    - F46A = Documents Required — lists all shipping documents the beneficiary must present
    - F47A = Additional Conditions — extra requirements that may override or supplement F46A
    - F45A = Description of Goods — must match across Invoice, BL, and other documents
    - F78 = Instructions to Paying/Accepting/Negotiating Bank
    - Some fields (31D, 44C, 48, 32B) are checked by code logic, not VLM (implicit checks)

AI MODEL: Qwen 2.5-VL-7B @ http://10.20.10.3:8000/v1/chat/completions
    Used for decomposing clause text into individual conditions.
    Implicit checks (dates, amounts) bypass the VLM entirely.
"""

import json
import sys as _sys; _sys.stdout.reconfigure(encoding="utf-8", errors="replace") if hasattr(_sys.stdout, "reconfigure") else None
import time
import re
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import QWEN_VLM_URL, QWEN_VLM_MODEL, MAX_CONCURRENT_VLM, VLM_TIMEOUT


# ── Data Models ──────────────────────────────────────────────────────────

@dataclass
class Condition:
    """
    A single checkable condition extracted from an LC clause.

    Each condition is independently verifiable against ONE specific document.
    For example, "Invoice must show HS Code" can be checked by looking at the
    Commercial Invoice text for an HS Code pattern.
    """
    condition_id: str = ""
    condition_text: str = ""             # Human-readable description of what to verify
    document_to_check: str = ""          # Target document type (e.g., "Bill of Lading")
    look_for_value: str = ""             # Specific value or pattern to search for
    is_implicit: bool = False            # True for date/amount checks done by Python code, not VLM
    implicit_type: str = ""              # lc_expiry, late_shipment, late_presentation, amount_overdrawn
    confidence: float = 1.0


@dataclass
class DecomposedClause:
    """
    A single LC clause broken into individual checkable conditions.

    For example, F46A clause #1 might decompose into 5 separate conditions,
    each targeting a specific document and looking for specific evidence.
    """
    clause_ref: str = ""             # e.g. "46A-1", "47A-3" — unique identifier
    field_tag: str = ""              # e.g. "46A", "47A", "45A" — SWIFT field tag
    clause_number: int = 0           # Position within the field (1-based)
    original_text: str = ""          # Full original clause text from the LC
    conditions: List[Condition] = field(default_factory=list)
    elapsed_seconds: float = 0.0     # Time taken for VLM decomposition


# ── VLM Prompt ───────────────────────────────────────────────────────────
# This system prompt instructs the Qwen VLM on how to decompose LC clauses.
# It includes trade finance domain knowledge to improve accuracy.

DECOMPOSITION_SYSTEM_PROMPT = """You are a trade finance expert specializing in Letter of Credit (LC) compliance under UCP 600 and ISBP 821.

Your task: decompose an LC clause into individual, checkable conditions.

TRADE FINANCE KNOWLEDGE:
- "FULL SET" means 3/3 originals (e.g., "FULL SET OF BILLS OF LADING" = 3 original BLs required)
- "ISSUING BANK" refers to the actual bank that issued the LC — use the bank name if provided
- "ACCEPTABLE" or "MAY" = optional/permissive — do NOT create mandatory conditions for these
- "IN DUPLICATE" = 2 copies, "IN TRIPLICATE" = 3 copies

RETURN EMPTY ARRAY [] ONLY for these specific clause types:
  * Bank-to-bank obligations: clauses that start with "NEGOTIATING BANK MUST...", "ADVISING BANK MUST..." (these are bank duties, not document requirements)
  * Fee/charge clauses: "DISCREPANCY HANDLING CHARGES", "BANK CHARGES WILL BE DEDUCTED"
  * Document forwarding: "ALL DOCUMENTS TO BE FORWARDED TO US BY COURIER"
  * Permissive overrides: "THIRD PARTY DOCUMENTS ARE ACCEPTABLE", "LARGER QUANTITIES ACCEPTABLE", "CHARTER PARTY BL ACCEPTABLE" (these ALLOW something, they don't require anything)
  * Date validity: "DOCUMENTS DATED PRIOR TO... NOT ACCEPTABLE" (handled separately)

IMPORTANT: If the clause is about a DOCUMENT REQUIREMENT (F46A), ALWAYS decompose it — never return empty array for F46A clauses.

DOCUMENT IDENTIFICATION:
  * "SHIPMENT ADVICE" or "BENEFICIARY SHIPMENT ADVICE" = check on document type "Shipment Advice"
  * "CERTIFICATE FROM SHIPPING COMPANY OR THEIR AUTHORIZED AGENTS" = check on "Shipping Company Certificate" or "Agent Certificate"
  * "INSURANCE" only when it says "INSURANCE POLICY" or "INSURANCE CERTIFICATE" — NOT when it says "INSURANCE COVERED BY APPLICANT" (that means applicant handles insurance, beneficiary sends advice)
  * When clause says "INSURANCE COVERED BY APPLICANT", the checkable part is the SHIPMENT ADVICE, not insurance

RULES:
1. Each condition must be independently verifiable against a SINGLE document
2. If a condition applies to MULTIPLE documents, create SEPARATE conditions for each document
   Example: "HS CODE MUST APPEAR ON BL AND INVOICES" -> 2 conditions (one for BL, one for Invoice)
3. condition_text: human-readable description of what to verify
4. document_to_check: the document type to check (e.g., "Bill of Lading", "Commercial Invoice", "Insurance Policy")
5. look_for_value: the specific value, text, or pattern to search for in the document

FIELD-SPECIFIC GUIDANCE:
- F45A (Description of Goods): Extract conditions for quantity, goods name, unit price, trade terms (CFR/CIF/FOB), port of destination, proforma invoice reference. Check these on BOTH Commercial Invoice AND Bill of Lading.
- F46A (Documents Required): Extract every requirement — signature, copies, addressee, content requirements, certifications, specific clauses to include.
- F47A (Additional Conditions): Extract each condition. If it says something is "ACCEPTABLE" or "ALLOWED", note it as a permissive condition.
- For certificates: extract issuer requirement, language, content, and addressee conditions.

IMPORTANT: ALWAYS return at least one condition for every clause. Never return an empty array unless the clause is truly a bank-internal instruction (like reimbursement terms). Every document requirement and goods description has checkable conditions.

Respond ONLY with a JSON array. Each element:
{
  "condition_text": "...",
  "document_to_check": "...",
  "look_for_value": "..."
}"""


# User prompt template — fills in the specific clause and LC context
DECOMPOSITION_USER_TEMPLATE = """Decompose this LC clause into individual checkable conditions.

LC Field: {field_tag}
Clause #{clause_number}:
{clause_text}

Additional LC context:
- Applicant: {applicant}
- Beneficiary: {beneficiary}
- Issuing Bank: {issuing_bank}
- Currency/Amount: {currency_amount}

Return a JSON array of conditions."""


# ── Implicit Checks ─────────────────────────────────────────────────────
# These LC fields are verified by deterministic Python code rather than VLM,
# because they involve date arithmetic or amount comparison — tasks where
# code is more reliable than an AI model.

# All 13 LC Key Terms that require code-based verification
# These are checked by deterministic Python logic, not VLM
IMPLICIT_CHECK_FIELDS = {
    '31C': 'date_of_issue',       # 1. All docs must be issued on/after LC date
    '31D': 'lc_expiry',           # 2. Documents presented within LC expiry
    '51D': 'applicant_bank',      # 3. BL endorsed to order of Applicant Bank
    '50':  'applicant_check',     # 4. Notify/Consignee in BL matches Applicant
    '59':  'beneficiary_check',   # 5. Shipper in BL matches Beneficiary
    '32B': 'amount_overdrawn',    # 6. Amount check: invoice vs LC, tolerance, partial
    '41D': 'available_with',      # 7. Negotiating bank in same country as expiry place
    '42C': 'draft_at_sight',      # 8. Draft reflects "At Sight" terms
    '42A': 'drawee_check',        # 9. Drawee in Draft matches LC (issuing bank)
    '43T': 'transshipment',       # 10. Transshipment allowed/not per BL
    '44E': 'port_of_loading',     # 11. Port of Loading in BL matches LC
    '44F': 'port_of_discharge',   # 12. Port of Discharge in BL matches LC
    '44C': 'late_shipment',       # 13. Shipment date on BL vs latest shipment date
    '48':  'late_presentation',   # Presentation period (derived from 31D + 44C)
}


def _build_implicit_conditions(field_tag: str, clause_text: str) -> List[Condition]:
    """
    Build code-verifiable conditions for ALL 13 LC Key Terms.

    These conditions are flagged as is_implicit=True so Step 14 knows to use
    Python logic instead of sending them to the VLM. Each check maps to a
    specific verification rule from trade finance practice.
    """
    implicit_type = IMPLICIT_CHECK_FIELDS.get(field_tag, '')
    conditions = []

    # 1. F31C - Date of Issue
    if implicit_type == 'date_of_issue':
        conditions.append(Condition(
            condition_text="All shipping documents must be issued on or after LC issuance date. Documents dated prior to LC date are discrepancies unless LC allows it.",
            document_to_check="All Documents",
            look_for_value=clause_text.strip(),
            is_implicit=True,
            implicit_type='date_of_issue',
        ))

    # 2. F31D - Date and Place of Expiry
    elif implicit_type == 'lc_expiry':
        conditions.append(Condition(
            condition_text="Documents must be presented within LC expiry date. Check presentation date from Covering Schedule against LC expiry.",
            document_to_check="Documentary Remittance / Covering Letter",
            look_for_value=clause_text.strip(),
            is_implicit=True,
            implicit_type='lc_expiry',
        ))

    # 3. F51D - Applicant Bank
    elif implicit_type == 'applicant_bank':
        conditions.append(Condition(
            condition_text="Bill of Lading must be issued or endorsed to the order of the Applicant Bank. Verify proper endorsement is available.",
            document_to_check="Bill of Lading",
            look_for_value=clause_text.strip(),
            is_implicit=True,
            implicit_type='applicant_bank',
        ))

    # 4. F50 - Applicant
    elif implicit_type == 'applicant_check':
        conditions.append(Condition(
            condition_text="Notify Party or Consignee in Bill of Lading must match the Applicant name as per LC.",
            document_to_check="Bill of Lading",
            look_for_value=clause_text.strip(),
            is_implicit=True,
            implicit_type='applicant_check',
        ))

    # 5. F59 - Beneficiary
    elif implicit_type == 'beneficiary_check':
        conditions.append(Condition(
            condition_text="Shipper/Exporter in Bill of Lading must match the Beneficiary mentioned in LC.",
            document_to_check="Bill of Lading",
            look_for_value=clause_text.strip(),
            is_implicit=True,
            implicit_type='beneficiary_check',
        ))

    # 6. F32B - Currency & Amount
    elif implicit_type == 'amount_overdrawn':
        conditions.append(Condition(
            condition_text="Verify LC amount against Covering Schedule, Commercial Invoice, and Bill of Exchange. Check for overdrawn/short shipment, tolerance, partial shipment conditions.",
            document_to_check="Commercial Invoice",
            look_for_value=clause_text.strip(),
            is_implicit=True,
            implicit_type='amount_overdrawn',
        ))
        conditions.append(Condition(
            condition_text="Draft/Bill of Exchange amount must match Invoice amount.",
            document_to_check="Draft Bill of Exchange",
            look_for_value=clause_text.strip(),
            is_implicit=True,
            implicit_type='draft_vs_invoice',
        ))

    # 7. F41D - Available With / By
    elif implicit_type == 'available_with':
        conditions.append(Condition(
            condition_text="If 'Any Bank' is specified, the negotiating/presenting bank must be in the same country as the place of LC expiry.",
            document_to_check="Documentary Remittance / Covering Letter",
            look_for_value=clause_text.strip(),
            is_implicit=True,
            implicit_type='available_with',
        ))

    # 8. F42C - Draft at Sight
    elif implicit_type == 'draft_at_sight':
        conditions.append(Condition(
            condition_text="Bill of Exchange (Draft) must reflect 'At Sight' terms as per LC.",
            document_to_check="Draft Bill of Exchange",
            look_for_value=clause_text.strip(),
            is_implicit=True,
            implicit_type='draft_at_sight',
        ))

    # 9. F42A - Drawee
    elif implicit_type == 'drawee_check':
        conditions.append(Condition(
            condition_text="Drawee mentioned in Bill of Exchange must match LC requirement (typically the Issuing Bank or Applicant Bank).",
            document_to_check="Draft Bill of Exchange",
            look_for_value=clause_text.strip(),
            is_implicit=True,
            implicit_type='drawee_check',
        ))

    # 10. F43T - Transshipment
    elif implicit_type == 'transshipment':
        conditions.append(Condition(
            condition_text="Verify transshipment condition in LC against Bill of Lading. Check vessel details and LC terms (allowed/not allowed).",
            document_to_check="Bill of Lading",
            look_for_value=clause_text.strip(),
            is_implicit=True,
            implicit_type='transshipment',
        ))

    # 11. F44E - Port of Loading
    elif implicit_type == 'port_of_loading':
        conditions.append(Condition(
            condition_text="Port of Loading in Bill of Lading must match LC requirement. If LC says 'Any Port in [Country]', verify port belongs to that country.",
            document_to_check="Bill of Lading",
            look_for_value=clause_text.strip(),
            is_implicit=True,
            implicit_type='port_of_loading',
        ))

    # 12. F44F - Port of Discharge
    elif implicit_type == 'port_of_discharge':
        conditions.append(Condition(
            condition_text="Port of Discharge in Bill of Lading must match LC requirement.",
            document_to_check="Bill of Lading",
            look_for_value=clause_text.strip(),
            is_implicit=True,
            implicit_type='port_of_discharge',
        ))

    # 13. F44C - Latest Date of Shipment
    elif implicit_type == 'late_shipment':
        conditions.append(Condition(
            condition_text="Shipment date on BL must be on or before the latest shipment date in LC. If exceeded, mark as Late Shipment Discrepancy.",
            document_to_check="Bill of Lading",
            look_for_value=clause_text.strip(),
            is_implicit=True,
            implicit_type='late_shipment',
        ))

    # F48 - Presentation Period (derived check)
    elif implicit_type == 'late_presentation':
        conditions.append(Condition(
            condition_text="Documents must be presented within the stipulated period after shipment date.",
            document_to_check="Documentary Remittance / Covering Letter",
            look_for_value=clause_text.strip(),
            is_implicit=True,
            implicit_type='late_presentation',
        ))
        conditions.append(Condition(
            condition_text="Documents must not be stale at presentation.",
            document_to_check="Documentary Remittance / Covering Letter",
            look_for_value=clause_text.strip(),
            is_implicit=True,
            implicit_type='stale_documents',
        ))

    return conditions


# ── VLM Call ─────────────────────────────────────────────────────────────

def _call_vlm_decompose(clause_ref: str, field_tag: str, clause_number: int,
                         clause_text: str, lc_context: dict) -> dict:
    """
    Send a single clause to the Qwen VLM for decomposition into conditions.

    The VLM receives the clause text along with LC context (applicant, beneficiary,
    bank, amount) so it can resolve references like "ISSUING BANK" to the actual
    bank name, or understand what "THE GOODS" refers to.

    Returns a dict with clause_ref, conditions list, and elapsed time.
    On error, returns the error message and an empty conditions list.
    """
    start = time.time()
    try:
        # Fill in the user prompt template with clause-specific data
        user_msg = DECOMPOSITION_USER_TEMPLATE.format(
            field_tag=field_tag,
            clause_number=clause_number,
            clause_text=clause_text,
            applicant=lc_context.get('applicant', 'N/A'),
            beneficiary=lc_context.get('beneficiary', 'N/A'),
            issuing_bank=lc_context.get('issuing_bank', 'N/A'),
            currency_amount=lc_context.get('currency_amount', 'N/A'),
        )

        payload = {
            "model": QWEN_VLM_MODEL,
            "messages": [
                {"role": "system", "content": DECOMPOSITION_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "temperature": 0.1,       # Low temperature for deterministic output
            "max_tokens": 2048,
        }

        resp = requests.post(QWEN_VLM_URL, json=payload, timeout=VLM_TIMEOUT)
        elapsed = time.time() - start

        if resp.status_code != 200:
            return {
                'clause_ref': clause_ref,
                'conditions': [],
                'error': f'HTTP {resp.status_code}',
                'elapsed': elapsed,
            }

        result = resp.json()
        content = result['choices'][0]['message']['content'].strip()

        # Extract JSON array from response — VLM may wrap it in markdown code fences
        json_match = re.search(r'\[.*\]', content, re.DOTALL)
        if json_match:
            conditions_raw = json.loads(json_match.group(0))
        else:
            conditions_raw = []

        return {
            'clause_ref': clause_ref,
            'conditions': conditions_raw,
            'elapsed': elapsed,
        }

    except json.JSONDecodeError:
        return {
            'clause_ref': clause_ref,
            'conditions': [],
            'error': 'JSON parse error from VLM',
            'elapsed': time.time() - start,
        }
    except Exception as e:
        return {
            'clause_ref': clause_ref,
            'conditions': [],
            'error': str(e),
            'elapsed': time.time() - start,
        }


# ── Clause Fields ────────────────────────────────────────────────────────
# These SWIFT field tags contain clause lists that need decomposition.
# Other fields (like F20 = LC Number) are standalone values, not clauses.

CLAUSE_FIELDS = {'46A', '46B', '47A', '47B', '45A', '45B', '46C'}
# Note: 78 (Instructions to Bank), 79 (Narrative), 72 (Sender to Receiver), 77A (Regulatory)
# are bank-to-bank instructions — NOT checkable against shipping documents, shown as informational only.


def _extract_lc_context(structured_lc: dict) -> dict:
    """
    Pull key LC context fields for VLM prompt enrichment.

    The VLM needs to know who the applicant, beneficiary, and issuing bank are
    so it can resolve references in clauses like "NOTIFY APPLICANT" or
    "TO ORDER OF ISSUING BANK".
    """
    fields = structured_lc.get('consolidated_fields', structured_lc)

    def _get(keys):
        """Try multiple possible key names — field naming varies between SWIFT formats."""
        for k in keys:
            v = fields.get(k)
            if v:
                return v if isinstance(v, str) else str(v)
        return 'N/A'

    return {
        'applicant': _get(['Applicant', 'Applicant_Name', '50']),
        'beneficiary': _get(['Beneficiary', 'Beneficiary_Name', '59']),
        'issuing_bank': _get(['Issuing_Bank', 'Issuing_Bank_Details', 'Sending_Institution', '52A']),
        'currency_amount': _get(['Amount', 'Currency_Amount', 'LC_Amount', '32B']),
    }


def _extract_clauses(structured_lc: dict) -> List[dict]:
    """
    Extract individual clauses from StructuredLC consolidated_fields.

    Handles three value formats:
    1. String — a single clause (treated as clause #1)
    2. List of strings — multiple clauses numbered 1, 2, 3...
    3. List of dicts — clauses with metadata (text + clause_number)
    """
    fields = structured_lc.get('consolidated_fields', structured_lc)
    clauses = []

    for field_tag in sorted(fields.keys()):
        if field_tag not in CLAUSE_FIELDS:
            continue

        value = fields[field_tag]

        # Normalize value into a list of text strings
        if isinstance(value, str):
            clause_texts = [value]
        elif isinstance(value, list):
            clause_texts = []
            for item in value:
                if isinstance(item, str):
                    clause_texts.append(item)
                elif isinstance(item, dict):
                    clause_texts.append(item.get('text', item.get('clause_text', str(item))))
                else:
                    clause_texts.append(str(item))
        else:
            clause_texts = [str(value)]

        # Create one entry per clause, numbered sequentially
        for i, text in enumerate(clause_texts, 1):
            if not text or not text.strip():
                continue
            clauses.append({
                'field_tag': field_tag,
                'clause_number': i,
                'clause_ref': f"{field_tag}-{i}",
                'text': text.strip(),
            })

    return clauses


# ── Main Run ─────────────────────────────────────────────────────────────

def run(structured_lc: dict, output_dir: str = None, progress_callback=None) -> dict:
    """
    Execute Step 12: Decompose LC clauses into individual checkable conditions.

    The function:
    1. Extracts all clause-type fields from the StructuredLC
    2. Separates implicit (code-checkable) from VLM-decomposed clauses
    3. Sends VLM clauses concurrently to the Qwen model
    4. Builds Condition objects from VLM responses
    5. Returns the full list of DecomposedClause objects

    Args:
        structured_lc: Output from Step 7 (StructuredLC with consolidated_fields)
        output_dir: Directory to save results
        progress_callback: Optional callback(message: str)

    Returns:
        dict with 'decomposed_clauses', 'total_conditions', 'elapsed_seconds'
    """
    def _progress(msg):
        if progress_callback:
            progress_callback(f"[Step 12] {msg}")
        print(f"[Step 12] {msg}")

    start_time = time.time()

    # Handle Step 7 output format: may have 'structured_lc' wrapper or 'all_clauses' directly
    _inner = structured_lc.get('structured_lc', structured_lc)
    # If Step 7 already pre-built clause objects, use them directly
    _prebuilt_clauses = _inner.get('all_clauses', [])

    # Extract LC context for VLM prompt enrichment (applicant, beneficiary, bank, amount)
    lc_context = _extract_lc_context(_inner)
    _progress(f"LC context: Applicant={lc_context['applicant'][:30]}, Bank={lc_context['issuing_bank'][:30]}")

    # Extract all clauses — use prebuilt if available from Step 7
    if _prebuilt_clauses:
        # Convert Step 7's clause format to Step 12's expected format
        clauses = []
        for _pc in _prebuilt_clauses:
            _ft = _pc.get('field_tag', '')
            if _ft in CLAUSE_FIELDS:
                clauses.append({
                    'clause_ref': f"{_ft}-{_pc.get('clause_number', 1)}",
                    'field_tag': _ft,
                    'clause_number': _pc.get('clause_number', 1),
                    'text': _pc.get('clause_text', _pc.get('text', '')),
                })
    else:
        clauses = _extract_clauses(_inner)
    _progress(f"Found {len(clauses)} clauses to decompose")

    decomposed: List[DecomposedClause] = []
    vlm_tasks = []
    errors = []

    # ── Separate implicit vs VLM-decomposed clauses ──
    # Implicit: date/amount fields checked by Python code (faster, more reliable)
    # VLM: document requirement clauses decomposed by AI (handles natural language)
    for clause in clauses:
        ft = clause['field_tag']

        # Check if this is an implicit (code-based) check
        if ft in IMPLICIT_CHECK_FIELDS:
            implicit_conds = _build_implicit_conditions(ft, clause['text'])
            dc = DecomposedClause(
                clause_ref=clause['clause_ref'],
                field_tag=ft,
                clause_number=clause['clause_number'],
                original_text=clause['text'],
                conditions=implicit_conds,
                elapsed_seconds=0.0,
            )
            decomposed.append(dc)
            _progress(f"  {clause['clause_ref']}: {len(implicit_conds)} implicit conditions")
        else:
            vlm_tasks.append(clause)

    # ── Send VLM tasks concurrently ──
    # Multiple clauses are processed in parallel to reduce total latency.
    # MAX_CONCURRENT_VLM controls how many simultaneous requests hit the model server.
    if vlm_tasks:
        _progress(f"Sending {len(vlm_tasks)} clauses to VLM for decomposition...")
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_VLM) as executor:
            futures = {}
            for clause in vlm_tasks:
                future = executor.submit(
                    _call_vlm_decompose,
                    clause['clause_ref'],
                    clause['field_tag'],
                    clause['clause_number'],
                    clause['text'],
                    lc_context,
                )
                futures[future] = clause

            for future in as_completed(futures):
                clause = futures[future]
                result = future.result()

                if 'error' in result:
                    errors.append(f"{clause['clause_ref']}: {result['error']}")
                    _progress(f"  {clause['clause_ref']}: ERROR - {result['error']}")

                # Build Condition objects from VLM response JSON
                conditions = []
                for j, cond_raw in enumerate(result.get('conditions', []), 1):
                    conditions.append(Condition(
                        condition_id=f"{clause['clause_ref']}-C{j}",
                        condition_text=cond_raw.get('condition_text', ''),
                        document_to_check=cond_raw.get('document_to_check', ''),
                        look_for_value=cond_raw.get('look_for_value', ''),
                    ))

                dc = DecomposedClause(
                    clause_ref=clause['clause_ref'],
                    field_tag=clause['field_tag'],
                    clause_number=clause['clause_number'],
                    original_text=clause['text'],
                    conditions=conditions,
                    elapsed_seconds=result.get('elapsed', 0),
                )
                decomposed.append(dc)
                _progress(f"  {clause['clause_ref']}: {len(conditions)} conditions ({result.get('elapsed', 0):.1f}s)")

    # Sort by clause_ref for consistent ordering (e.g., 45A-1, 46A-1, 46A-2, 47A-1)
    decomposed.sort(key=lambda d: d.clause_ref)

    # Assign condition IDs if not already set (ensures every condition has a unique ID)
    for dc in decomposed:
        for j, cond in enumerate(dc.conditions, 1):
            if not cond.condition_id:
                cond.condition_id = f"{dc.clause_ref}-C{j}"

    total_conditions = sum(len(dc.conditions) for dc in decomposed)
    elapsed = time.time() - start_time
    _progress(f"Step 12 complete: {len(decomposed)} clauses -> {total_conditions} conditions in {elapsed:.1f}s")

    # Save results to disk
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        result_file = os.path.join(output_dir, 'step12_result.json')
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({
                'step': 12,
                'step_name': 'Clause-by-Clause Condition Decomposition',
                'total_clauses': len(decomposed),
                'total_conditions': total_conditions,
                'elapsed_seconds': round(elapsed, 2),
                'errors': errors,
                'decomposed_clauses': [asdict(dc) for dc in decomposed],
            }, f, indent=2, ensure_ascii=False)

    return {
        'decomposed_clauses': decomposed,
        'total_clauses': len(decomposed),
        'total_conditions': total_conditions,
        'elapsed_seconds': round(elapsed, 2),
        'errors': errors,
    }


if __name__ == '__main__':
    # CLI test with sample LC data — demonstrates decomposition of typical clauses
    sample_lc = {
        'consolidated_fields': {
            '46A': [
                "SIGNED COMMERCIAL INVOICE IN 3 COPIES SHOWING HS CODE",
                "FULL SET OF CLEAN ON BOARD BILLS OF LADING MADE OUT TO ORDER OF ISSUING BANK MARKED FREIGHT PREPAID NOTIFY APPLICANT",
                "INSURANCE POLICY OR CERTIFICATE IN DUPLICATE FOR 110 PCT OF INVOICE VALUE COVERING ALL RISKS",
            ],
            '47A': [
                "ALL DOCUMENTS MUST SHOW LC NUMBER AND DATE",
                "HS CODE MUST APPEAR ON BL AND INVOICES",
            ],
            '31D': "2026-06-30 PAKISTAN",
            '44C': "2026-06-15",
            '32B': "USD 490,200.00",
            'Applicant': 'ABC Trading Co.',
            'Beneficiary': 'XYZ Exports Ltd.',
            'Issuing_Bank': 'United Bank Limited',
        }
    }
    result = run(sample_lc)
    for dc in result['decomposed_clauses']:
        print(f"\n{dc.clause_ref}: {dc.original_text[:80]}...")
        for c in dc.conditions:
            print(f"  -> [{c.document_to_check}] {c.condition_text}")
