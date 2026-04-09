"""
ComplyTrade Pilot V2 — Configuration
"""

# ── Model Endpoints ──
GLM_OCR_URL = "http://72.255.9.104:8001/api/ocr"
GLM_OCR_MODEL = "glm-ocr"

# Qwen 7B for lightweight tasks (text cleaning, simple checks)
QWEN_7B_URL = "http://72.255.9.104:8000/v1/chat/completions"
QWEN_7B_MODEL = "/home/aigenics/AI_MODELS/Qwen2.5-VL-7B-Instruct"

# Qwen VLM for classification, decomposition, verification
QWEN_VLM_URL = "http://72.255.9.104:8085/v1/chat/completions"
QWEN_VLM_MODEL = "/home/aigenics/AI_MODELS/Qwen2.5-VL-72B-Instruct-AWQ"

# ── Server ──
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8082
BUILD_TAG = "2026-04-09-P54"

# ── Processing ──
MAX_CONCURRENT_OCR = 8
MAX_CONCURRENT_VLM = 8
OCR_TIMEOUT = 600       # seconds per page
VLM_TIMEOUT = 1200      # seconds per VLM call (20 min — large docs need time)
CONFIDENCE_THRESHOLD = 0.70  # Below this → REVIEW status (0.70 for 7B, 0.85 for 72B)

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
