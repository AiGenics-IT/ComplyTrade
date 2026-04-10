"""
ComplyTrade Pilot V2 — Configuration
"""

import os as _os

# ── Model Endpoints ──
#
# Pipeline overview:
#   • Step 1 (Raw OCR)        → ALWAYS uses GLM-OCR (specialised OCR model)
#   • Steps 2–14 (everything  → Use Qwen VLM, switchable between 7B and 72B
#     that needs vision/      via VLM_MODEL_SIZE below
#     reasoning)
#
# Three physical inference hosts on the internal LAN:
#   • 10.20.10.2:8001  → GLM-OCR (always used by Step 1)
#   • 10.20.10.2:8085  → Qwen 2.5-VL-72B-Instruct-AWQ
#   • 10.20.10.3:8000  → Qwen 2.5-VL-7B-Instruct

# Step 1: GLM-OCR — always used regardless of VLM_MODEL_SIZE
GLM_OCR_URL = "http://10.20.10.2:8001/api/ocr"
GLM_OCR_MODEL = "glm-ocr"

# Static endpoint constants for each Qwen model.
# Steps 2–14 do not import these directly — they import the resolved
# QWEN_VLM_URL / QWEN_VLM_MODEL below, which switch on VLM_MODEL_SIZE.
QWEN_7B_URL = "http://10.20.10.3:8000/v1/chat/completions"
QWEN_7B_MODEL = "/home/aigenics/AI_MODELS/Qwen2.5-VL-7B-Instruct"

QWEN_72B_URL = "http://10.20.10.2:8085/v1/chat/completions"
QWEN_72B_MODEL = "/home/aigenics/AI_MODELS/Qwen2.5-VL-72B-Instruct-AWQ"

# ── Active VLM selection ──
# Single source of truth for the entire pipeline (Steps 2–14).
# Switch the whole system between 7B and 72B by changing this one value
# (or by setting the VLM_MODEL_SIZE environment variable to "7B" / "72B").
# Step 1 (GLM-OCR) is unaffected — it always uses GLM regardless.
VLM_MODEL_SIZE = _os.environ.get("VLM_MODEL_SIZE", "72B").upper()

if VLM_MODEL_SIZE == "7B":
    QWEN_VLM_URL = QWEN_7B_URL
    QWEN_VLM_MODEL = QWEN_7B_MODEL
else:
    # Default / "72B"
    VLM_MODEL_SIZE = "72B"
    QWEN_VLM_URL = QWEN_72B_URL
    QWEN_VLM_MODEL = QWEN_72B_MODEL

# ── Server ──
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8082
BUILD_TAG = "2026-04-10-P77"

# ── Processing ──
MAX_CONCURRENT_OCR = 8
MAX_CONCURRENT_VLM = 8
OCR_TIMEOUT = 600       # seconds per page
VLM_TIMEOUT = 1200      # seconds per VLM call (20 min — large docs need time)
CONFIDENCE_THRESHOLD = 0.70  # Below this → REVIEW status. Same threshold for both 7B and 72B for now.

# ── Database ──
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "trade_finance_pilot"
DB_USER = "postgres"
DB_PASS = "123"

# ── Authentication ──
AUTH_ENABLED = True
AUTH_USERNAME = "admin"
AUTH_PASSWORD = "complytrade2026"

# ── Branding ──
LOGO_PATH = "view/logo.png"
WATERMARK_TEXT = "AiGenics"
COMPANY_NAME = "AiGenics"

# ── Paths ──
UPLOAD_DIR = "uploads"
RESULTS_DIR = "results"
VIEW_DIR = "view"

# ── Pipeline Steps Toggle ──
# Set to False to skip a step. Core steps (1,2,3,6,7,8,9,12,13,14,19,20) should stay enabled.
# Optional steps can be disabled to speed up processing.
STEP_ENABLED = {
    1:  True,   # Page-Level Raw OCR Extraction (CORE — required)
    2:  True,   # OCR Text Cleaning (CORE — improves quality)
    3:  True,   # Page Sequencing & Classification (CORE — required)
    4:  True,   # MT Identification
    5:  True,   # MT Reconciliation
    6:  True,   # Final LC Extraction (CORE — required)
    7:  True,   # Clause & Requirement Extraction (CORE — required)
    8:  True,   # Shipping Document Classification (CORE — required)
    9:  True,   # Shipping OCR Reconciliation (CORE — required)
    10: False,  # Traceability Flags (OPTIONAL — can skip)
    11: False,  # Human Review Flags (OPTIONAL — can skip)
    12: True,   # Clause Decomposition (CORE — required for verification)
    13: True,   # Row Construction (CORE — required for verification)
    14: True,   # VLM Verification (CORE — main verification)
    15: False,  # Non-Compliance Summary (OPTIONAL — duplicate of step14)
    16: False,  # Confidence Review (OPTIONAL — can skip)
    17: False,  # Cross-Clause Checks (OPTIONAL — covered by step14b)
    18: False,  # Threading (OPTIONAL — just groups results)
    19: True,   # Consolidation (CORE — required for report)
    20: True,   # Report Generation (CORE — required)
}

# ── GLM OCR Prompt ──
GLM_OCR_PROMPT = """Extract ALL text from this page. Every word, every number, every symbol. Missing even ONE character is a critical failure.

CRITICAL RULES:
- Extract ALL amounts in BOTH figures (USD 490,200.00) AND words (Four hundred ninety thousand)
- Preserve @ in emails exactly
- Extract stamp text: ORIGINAL, COPY, NON-NEGOTIABLE
- Extract ALL table rows and columns completely
- Extract SWIFT F-tags (F20:, F31C:, F46A:, F47A:) with COMPLETE values
- NUMBERED CLAUSES: If you see numbered items (1., 2., 3... or 1), 2), 3)...), extract EVERY SINGLE ONE. Do NOT skip any numbered clause even if it continues from a previous page or seems like a continuation.
- CONTINUATION TEXT: If text at the top of the page continues from the previous page (no number prefix), still extract it completely.
- Note signatures as [SIGNATURE] and logos as [LOGO: name]
- Note images as [IMAGE: description]
- If text is unclear, mark as [unclear]
- Do NOT skip any text, no matter how small or how close to the page edge
- Do NOT summarize or interpret
- Preserve line breaks and formatting"""
