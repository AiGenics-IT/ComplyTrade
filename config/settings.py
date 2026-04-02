"""
ComplyTrade Pilot V2 — Configuration
"""

# ── Model Endpoints ──
GLM_OCR_URL = "http://10.20.10.3:8001/api/ocr"
GLM_OCR_MODEL = "glm-ocr"

# Qwen 7B for lightweight tasks (text cleaning, simple checks)
QWEN_7B_URL = "http://10.20.10.3:8000/v1/chat/completions"
QWEN_7B_MODEL = "/home/aigenics/AI_MODELS/Qwen2.5-VL-7B-Instruct"

# Qwen VLM for classification, decomposition, verification
QWEN_VLM_URL = "http://10.20.10.2:8085/v1/chat/completions"
QWEN_VLM_MODEL = "/home/aigenics/AI_MODELS/Qwen2.5-VL-72B-Instruct-AWQ"

# ── Server ──
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8090
BUILD_TAG = "2026-04-02-P24"

# ── Processing ──
MAX_CONCURRENT_OCR = 4
MAX_CONCURRENT_VLM = 4
OCR_TIMEOUT = 120       # seconds per page
VLM_TIMEOUT = 600       # seconds per VLM call (96GB GPU handles large images)
CONFIDENCE_THRESHOLD = 0.98  # Below this → REVIEW status

# ── Database ──
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "trade_finance_pilot"
DB_USER = "postgres"
DB_PASS = "123"

# ── Branding ──
LOGO_PATH = "view/logo.png"
WATERMARK_TEXT = "AiGenics"
COMPANY_NAME = "AiGenics"

# ── Paths ──
UPLOAD_DIR = "uploads"
RESULTS_DIR = "results"
VIEW_DIR = "view"

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
