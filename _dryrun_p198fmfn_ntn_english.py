"""
P198fm + P198fn dry-run.

P198fm — NTN deterministic rescue:
  When the LC says "BL/Invoice must show NTN <num>" and the LLM FAILed
  because it didn't pick the NTN out of OCR noise, we scan
  unified_summary.references_found AND raw doc text for the NTN. If
  found anywhere → override FAIL to PASS.

P198fn — English-language deterministic rescue:
  When the LC says "All documents in English" and the LLM FAILs a doc
  that is clearly English (>= 70% ASCII letters + >= 5 stop-words),
  override the FAIL to PASS.
"""
import sys, os, re
sys.path.insert(0, 'd:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final')

results = []
def assert_eq(name, got, expected):
    ok = (got == expected)
    print(f"[{'OK' if ok else 'FAIL'}] {name}")
    if not ok:
        print(f"          got     : {got!r}")
        print(f"          expected: {expected!r}")
    results.append(ok)

# ── P198fm — NTN extraction logic ───────────────────────────────────
print("--- P198fm: NTN extraction patterns ---")

def _norm_ntn(s):
    return re.sub(r'[^0-9A-Z]', '', str(s or '').upper())

# Match the regex used in production
def _extract_ntn_from_condition(cond_u):
    m = re.search(r'NTN\s*(?:NO\.?|NUMBER)?\s*[:.]?\s*([0-9][0-9A-Z\-]{4,15})', cond_u)
    if not m:
        m = re.search(r'\b(\d{6,8}-?\d?)\b', cond_u)
    return m.group(1) if m else None

NTN_CASES = [
    ("Standard NTN with hyphen",
     "BILLS OF LADING MUST SHOW NTN NO. 0710106-6.",
     "0710106-6"),
    ("NTN without 'NO.' ",
     "Importer's NTN 2232692-8 must appear.",
     "2232692-8"),
    ("compact NTN",
     "NTN: 0710106-6", "0710106-6"),
    ("8 digits no hyphen",
     "NTN 12345678", "12345678"),
    ("with parenthetical",
     "NTN (0710106-6) on all docs.", "0710106-6"),
]
for name, cond, expected in NTN_CASES:
    got = _extract_ntn_from_condition(cond.upper())
    assert_eq(f"NTN extract: {name}", got, expected)

# Match: required vs document text
print("\n--- P198fm: NTN match against doc text/refs ---")
MATCH_CASES = [
    ("references_found has NTN exactly",
     "0710106-6", [{'role': 'ntn_number', 'value': '0710106-6'}], "doc text", True),
    ("ref has compact form (no hyphen)",
     "0710106-6", [{'role': 'ntn', 'value': '07101066'}], "doc text", True),
    ("ref has different NTN",
     "0710106-6", [{'role': 'ntn_number', 'value': '2232692-8'}], "doc text", False),
    ("doc text has NTN with prefix",
     "0710106-6", [], "Importer NTN: 0710106-6 issued 2025", True),
    ("doc text has glued NTN",
     "0710106-6", [], "DOCUMENTARYCREDITNTN0710106-6DATED", True),
    ("doc text has NO NTN",
     "0710106-6", [], "Random invoice text without any NTN.", False),
]
for name, req, refs, txt, expected in MATCH_CASES:
    req_norm = _norm_ntn(req)
    found = False
    for item in refs:
        r = (item.get('role') or '').lower()
        if any(k in r for k in ('ntn', 'national_tax')):
            v = _norm_ntn(item.get('value'))
            if v and req_norm in v:
                found = True; break
    if not found:
        txt_norm = _norm_ntn(txt)
        if req_norm in txt_norm or req_norm.replace('-', '') in txt_norm:
            found = True
    assert_eq(f"NTN match: {name}", found, expected)


# ── P198fn — English-language detection ─────────────────────────────
print("\n--- P198fn: English-language heuristic ---")
_STOPWORDS = {
    'the','and','of','to','in','for','is','on','by','with',
    'this','that','as','from','be','are','at','we','or','an',
    'will','all','have','has','no','not','a','shall','any','date',
}

def is_english(text):
    if len(text) < 50:
        return False
    letters = sum(1 for c in text if c.isalpha() and ord(c) < 128)
    nonspace = sum(1 for c in text if not c.isspace())
    if nonspace == 0:
        return False
    ratio = letters / nonspace
    words = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
    stops = sum(1 for w in words if w in _STOPWORDS)
    return ratio >= 0.70 and stops >= 5


# Real ICIL-style credit report (the user's exact case)
icil_text = (
    "ICIL\nPAKISTAN\nInternational Credit Information Ltd.\n"
    "Risk Solutions for Credit Decisions\n"
    "GlobalCIS BASISNET Ydtb RiME TRELIS BUSINESS coface\n"
    "Source Of Credit Reports : COFACE\n"
    "Ultimate Beneficiary\nMR. ATIF M. KAWAJA\nMS. NIDA SHAHZAD\n"
    "Shareholding\n50%\nPercentage\n50%\n"
    "Involvement In Other Companies\nNot disclosed\n"
    "Directors And Supervisors\nMR. ATIF M. KAWAJA, MS. NIDA SHAHZAD, MR. NASIR YAR KHAN\n"
    "ICIL\nPAKISTAN\nFor Exclusive Use of\nBANK AL HABIB LIMITED\n"
    "THIS REPORT MAY NOT BE REPRODUCED IN WHOLE OR IN PART IN ANY FROM OR MANNER WHATSOEVER\n"
)

ENG_CASES = [
    ("ICIL credit report (the user's case)", icil_text, True),
    ("simple English invoice",
     "Commercial Invoice. Description of goods: Cotton fabric. Quantity: 5000 yards. "
     "Total amount payable on this date is USD 50,000.", True),
    ("English BL with technical jargon",
     "Bill of Lading. Shipper: ABC. Consignee: To order of Bank XYZ. "
     "Vessel: MV CARGO. Port of Loading: Karachi. Port of Discharge: Hamburg. "
     "Marks and numbers as per attached list.", True),
    ("Mostly Arabic text",
     "هذا نص عربي بالكامل لا يحتوي على إنجليزية إطلاقًا "
     "ويجب أن يفشل في فحص اللغة الإنجليزية حتى لو احتوى على بعض الأرقام 12345 "
     "ولا يجب أن يمر هذا الفحص لأنه ليس باللغة المطلوبة بأي شكل من الأشكال.", False),
    ("Heavy non-Latin (Chinese)",
     "这是中文文本完全没有英文应该被识别为非英文文档以验证启发式正常工作 "
     "并且不应该错误地通过英文语言检查 这只是一个测试用的简单文本", False),
    ("Empty short text",
     "OK", False),
    ("Numeric-only invoice line",
     "5000 50000 200000 100", False),  # No stopwords
    ("Mixed but predominantly English",
     "Beneficiary's signed Commercial Invoice in triplicate certifying that "
     "the merchandise is of South Korea origin mentioning HS Code 8415.1029 "
     "and importer's NTN 2232692-8.", True),
]
for name, text, expected in ENG_CASES:
    got = is_english(text)
    assert_eq(f"English heuristic: {name}", got, expected)


# ── Source-code wiring check ────────────────────────────────────────
print("\n--- Source wiring checks ---")
v_src = open('d:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final/steps/step14_verification.py',
             'r', encoding='utf-8').read()
assert_eq("P198fm: NTN rescue block present",
          'P198fm' in v_src and 'NTN deterministic rescue' in v_src, True)
assert_eq("P198fm: scans references_found AND document_text",
          'references_found' in v_src and 'document_text' in v_src, True)
assert_eq("P198fn: English rescue block present",
          'P198fn' in v_src and 'English deterministic rescue' in v_src, True)
assert_eq("P198fn: stop-word + ASCII-ratio heuristic",
          '_STOPWORDS' in v_src and 'ascii_ratio' in v_src.lower() or
          'ascii letter ratio' in v_src.lower(), True)


passed = sum(results)
total = len(results)
print(f"\n{passed}/{total} cases passed")
if passed != total:
    sys.exit(1)
print("OVERALL: OK")
sys.exit(0)
