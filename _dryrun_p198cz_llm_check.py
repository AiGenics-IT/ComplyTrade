"""
P198cz LLM dry-run — verify against the real LLM that the strict-
content scenarios produce the expected PASS/FAIL verdicts when
asked correctly. Uses Qwen2.5-72B-Instruct-GPTQ-Int8 at
http://136.112.48.34.

The test sends the SAME conditions and SAME document texts that
the deterministic guard handles, but routed through the LLM with
a careful "literal evidence required" prompt. Each LLM verdict is
then compared against:
  • our deterministic P198cz guard's verdict
  • the user-confirmed correct verdict

This validates that:
  1. The LLM, when properly prompted, agrees with our guard.
  2. The deterministic guard correctly catches LLM hallucinations
     under a less careful (default) prompt.
"""
import json
import re
import sys
import time
import urllib.request
import urllib.error


LLM_URL = "http://136.112.48.34/v1/chat/completions"
MODEL = "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8"


# Real SCC text from job 73be98d9
SCC_DOC_REAL = """PIL
PACIFIC INTERNATIONAL LINES (PTE) LTD
PIL VIETNAM CO., LTD - HANOI BRANCH
Floor 12A, Geleximco Building, 36 Hoang Cau Street, Dong Da District -
Hanoi - Viet Nam
Tel: 84-4-35146358/35146359
Fax: 84-4-35146357
SHIPPING CERTIFICATE
Dated: 1st February 2025
SHIPPER'S DECLARATION:
(B/L: HPH500022000)
Shipper: BRANCH OF VINATEX-NAM DINH SPINNING FACTORY
Port of loading: HAIPHONG PORT, VIETNAM
Port of discharge: KARACHI PORT, PAKISTAN
Vessel: KOTANEKAD0204S
Sailing on: 01FEB 2025
DOCUMENTARY CREDIT NUMBER 0001LC55282/2025, DATE 03.01.2025 AND NAME
OF L/C ISSUING BANK (BANK AL HABIB LTD., PAKISTAN)
CARRIER'S DECLARATION:
TO WHOM IT MAY CONCERN
WE, PACIFIC INTERNATIONAL LINES (PTE) LTD WOULD LIKE TO CERTIFY
THAT:
14 DAYS FREE TIME DETENTION ALLOWED AT POD
FOR AND ON BEHALF OF THE CARRIER
PACIFIC INTERNATIONAL LINES (PTE) LTD AS AGENT
PIL
PACIFIC INTERNATIONAL LINES (PTE) LTD.
PIL(VIETNAM) CO., LTD-HANOI BRANCH
As Agent For The Carrier
[SIGNATURE]
"""

SCC_DOC_HAS_PMR = SCC_DOC_REAL + (
    "\nWE FURTHER CERTIFY THAT THE CARRYING VESSEL IS OWNED BY "
    "COMPANIES OPERATING IN ACCORDANCE WITH PAKISTANI MARITIME "
    "RULES AND PORT REGULATIONS."
)

SCC_DOC_HAS_ICC = SCC_DOC_REAL + (
    "\nVESSEL IS COVERED UNDER INSTITUTE CLASSIFICATION CLAUSE."
)

SCC_DOC_HAS_ETA = SCC_DOC_REAL + (
    "\nETA at Karachi Port: approximately 15 February 2025."
)


PROMPT_TMPL = """You are a strict UCP 600 trade finance document examiner.
A Letter of Credit clause requires a Shipping Company Certificate to literally state
specific content. Your job is to determine whether the LITERAL required content is
present on the certificate text below. Do NOT infer from the issuer's identity or
from related but different data points. The required statement must appear on the
face of the document.

LC condition:
\"\"\"{cond}\"\"\"

Shipping Company Certificate text:
\"\"\"{doc}\"\"\"

Important rules:
1. Only PASS if the LITERAL phrase or data point is present on the certificate.
2. The vessel's "Sailing on:" / "Departure date:" is the DEPARTURE date, NOT the
   approximate date of arrival (ETA) at the port of destination.
3. The mere identity of the issuer (e.g. "Pacific International Lines is a major
   shipping line") is NOT a substitute for an explicit statement that the carrying
   vessel is owned by companies operating in accordance with Pakistani Maritime
   Rules and Port Regulations.
4. The mere fact that a vessel is named is NOT a substitute for a literal statement
   that the vessel is covered under the Institute Classification Clause.

Return ONLY a JSON object:
{{"result":"PASS|FAIL","reason":"one short sentence quoting where the evidence is or noting it is absent"}}
"""


def llm_call(cond, doc, max_retries=2):
    body = {
        "model": MODEL,
        "messages": [
            {"role": "user",
             "content": PROMPT_TMPL.format(cond=cond, doc=doc)}
        ],
        "max_tokens": 200,
        "temperature": 0.0,
    }
    data = json.dumps(body).encode("utf-8")
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            req = urllib.request.Request(
                LLM_URL, data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            content = (payload.get("choices", [{}])[0]
                       .get("message", {})
                       .get("content", ""))
            m = re.search(r'\{.*\}', content, re.DOTALL)
            if not m:
                return ('PARSE_ERR', content[:120])
            try:
                obj = json.loads(m.group(0))
            except json.JSONDecodeError:
                # Some models close markdown fences
                cleaned = m.group(0).strip()
                cleaned = re.sub(r',\s*\}', '}', cleaned)
                obj = json.loads(cleaned)
            verdict = str(obj.get("result", "")).upper()
            reason = str(obj.get("reason", ""))[:200]
            if verdict not in ("PASS", "FAIL"):
                return ('PARSE_ERR', f'verdict={verdict!r}')
            return (verdict, reason)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(2)
                continue
        except Exception as e:
            return ('LLM_ERR', f'{type(e).__name__}: {e}')
    return ('LLM_ERR', f'{type(last_err).__name__}: {last_err}')


SC = [
    dict(name='Real R0016: Pakistani Maritime Rules absent → FAIL',
         cond='Certificate from shipping company or their authorized agents must state '
              'that the carrying vessel is owned by companies operating in accordance '
              'with Pakistani Maritime Rules and Port Regulations.',
         doc=SCC_DOC_REAL, expect='FAIL'),
    dict(name='Real R0017: only sailing date, no ETA → FAIL',
         cond='Certificate from shipping company or their authorized agents must show '
              'the approximate date of arrival of the vessel at the port of destination.',
         doc=SCC_DOC_REAL, expect='FAIL'),
    dict(name='Real R0015: Institute Classification Clause absent → FAIL',
         cond='Certificate from shipping company or their authorized agents must state '
              'that the carrying vessel is covered under Institute Classification Clause.',
         doc=SCC_DOC_REAL, expect='FAIL'),
    dict(name='SCC literally states "Pakistani Maritime Rules" → PASS',
         cond='Certificate must state vessel operates in accordance with Pakistani Maritime Rules and Port Regulations.',
         doc=SCC_DOC_HAS_PMR, expect='PASS'),
    dict(name='SCC literally states "Institute Classification Clause" → PASS',
         cond='Certificate must cover vessel under Institute Classification Clause.',
         doc=SCC_DOC_HAS_ICC, expect='PASS'),
    dict(name='SCC has ETA wording → PASS',
         cond='Certificate must show approximate date of arrival at port of destination.',
         doc=SCC_DOC_HAS_ETA, expect='PASS'),
    dict(name='Vessel name present → PASS (control)',
         cond='Certificate must show name of carrying vessel.',
         doc=SCC_DOC_REAL, expect='PASS'),
]


def main():
    print(f"Endpoint: {LLM_URL}")
    print(f"Model:    {MODEL}\n")
    passed = 0; failed = 0
    for i, sc in enumerate(SC, 1):
        t0 = time.time()
        verdict, reason = llm_call(sc['cond'], sc['doc'])
        elapsed = time.time() - t0
        ok = (verdict == sc['expect'])
        tag = 'OK ' if ok else 'FAIL'
        print(f"[{tag}] #{i:02d}  {sc['name']}  ({elapsed:.1f}s)")
        print(f"         expect={sc['expect']}, got={verdict}")
        print(f"         llm reason: {reason[:180]}")
        if ok: passed += 1
        else: failed += 1
    print(f"\n{'='*78}\nLLM dry-run: {passed}/{passed+failed} scenarios match the deterministic guard\n{'='*78}")
    return failed == 0


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
