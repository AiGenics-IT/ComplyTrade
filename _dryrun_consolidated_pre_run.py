"""
Consolidated pre-run stress test: live LLM verification of the
recent fixes (P198cy / cz / da / db / dc / dd / de) before the
user kicks off a fresh end-to-end pipeline.

Each scenario is sent to the LIVE Qwen LLM at
http://136.112.48.34/v1/chat/completions and the verdict is
compared against the deterministic guard's expected verdict.
Scenarios are picked to stress real-world ambiguities that the
fixes need to handle correctly.
"""
import json, re, sys, time, urllib.request, urllib.error


LLM_URL = "http://136.112.48.34/v1/chat/completions"
MODEL = "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8"


PROMPT_DOC = """You are a strict UCP 600 trade finance document examiner.

LC condition (verbatim):
\"\"\"{cond}\"\"\"

Document text submitted (one document; identify any literal
evidence on the body):
\"\"\"{doc}\"\"\"

Rules:
1. Only PASS if literal evidence is present on the document.
   Do NOT infer from issuer identity, sailing date, or unrelated
   lines.
2. The vessel's "Sailing on" / "Departure date" is NOT the ETA
   at the destination port.
3. The mere identity of the issuer (e.g. Pacific International
   Lines, COSCO, etc.) does NOT substitute for an explicit
   statement that the vessel is owned by companies operating in
   accordance with Pakistani Maritime Rules and Port Regulations.
4. The mere fact that a vessel is named is NOT a substitute for
   a literal statement that the vessel is covered under the
   Institute Classification Clause.
5. For "must be present in the submission" presence checks: the
   reference must literally appear on at least one document
   text.
6. For F48-style date conditions ("X DAYS BUT WITHIN EXPIRY"):
   the LC expiry is the binding deadline when the F48 phrasing
   includes that "WITHIN EXPIRY" qualifier. Otherwise the X-day
   count is strict.

Return ONLY a JSON object:
{{"result":"PASS|FAIL","reason":"one short sentence quoting evidence or noting absence"}}
"""


def llm_call(cond, doc, max_retries=2):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT_DOC.format(cond=cond, doc=doc)}],
        "max_tokens": 250, "temperature": 0.0,
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
                       .get("message", {}).get("content", ""))
            m = re.search(r'\{.*\}', content, re.DOTALL)
            if not m:
                return ('PARSE_ERR', content[:120])
            obj = json.loads(re.sub(r',\s*\}', '}', m.group(0).strip()))
            v = str(obj.get("result", "")).upper()
            r = str(obj.get("reason", ""))[:200]
            if v not in ("PASS", "FAIL"):
                return ('PARSE_ERR', f'verdict={v!r}')
            return (v, r)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(2)
        except Exception as e:
            return ('LLM_ERR', f'{type(e).__name__}: {e}')
    return ('LLM_ERR', f'{type(last_err).__name__}: {last_err}')


# Real-document fixtures (truncated for prompt economy)
SCC_NO_PMR_NO_ETA = """PIL PACIFIC INTERNATIONAL LINES (PTE) LTD
SHIPPING CERTIFICATE
Vessel: KOTANEKAD0204S
Sailing on: 01FEB 2025
Port of loading: HAIPHONG PORT, VIETNAM
Port of discharge: KARACHI PORT, PAKISTAN
WE, PACIFIC INTERNATIONAL LINES (PTE) LTD WOULD LIKE TO CERTIFY THAT:
14 DAYS FREE TIME DETENTION ALLOWED AT POD
FOR AND ON BEHALF OF THE CARRIER
PACIFIC INTERNATIONAL LINES (PTE) LTD AS AGENT
"""
SCC_HAS_PMR = SCC_NO_PMR_NO_ETA + (
    "\nWE FURTHER CERTIFY THAT THE CARRYING VESSEL IS OWNED BY "
    "COMPANIES OPERATING IN ACCORDANCE WITH PAKISTANI MARITIME "
    "RULES AND PORT REGULATIONS."
)
SCC_HAS_ICC = SCC_NO_PMR_NO_ETA + (
    "\nVESSEL IS COVERED UNDER INSTITUTE CLASSIFICATION CLAUSE."
)
SCC_HAS_ETA = SCC_NO_PMR_NO_ETA + (
    "\nETA at Karachi Port: approximately 15 February 2025."
)

DR_NO_CHARGES = """We enclose documents related to LC 0001LC55282/2025.
Total Amount Claimed: USD 100,000.
Presentation Number: 12345.
Bank Al Habib Limited
"""
DR_WITH_CHARGES = DR_NO_CHARGES + (
    "\nALL OUR CHARGES AND ALL CHARGES OF THE ADVISING BANK ARE "
    "PAID BY THE BENEFICIARY."
)

CI_BRAZIL = """COMMERCIAL INVOICE
Invoice No: MCI-786/S-13198-SOY-E
Date: 18 FEB 2026
Origin: Brazil
Soybeans 1000 MT
USD 481,580.00
"""
CI_BRAZIL_PROFORMA = CI_BRAZIL + (
    "\nProforma Invoice Reference: 786/S-13198-SOYPI-E dated Jan 21, 2026"
)


SC = [
    # P198cz — SCC strict-content guard
    dict(group='cz', name='SCC missing Pakistani Maritime Rules',
         cond='Certificate from shipping company must state vessel is owned by '
              'companies operating in accordance with Pakistani Maritime Rules and Port Regulations.',
         doc=SCC_NO_PMR_NO_ETA, expect='FAIL'),
    dict(group='cz', name='SCC literally states Pakistani Maritime Rules',
         cond='Certificate must state vessel operates in accordance with Pakistani Maritime Rules.',
         doc=SCC_HAS_PMR, expect='PASS'),
    dict(group='cz', name='SCC missing ICC',
         cond='Certificate must state vessel is covered under Institute Classification Clause.',
         doc=SCC_NO_PMR_NO_ETA, expect='FAIL'),
    dict(group='cz', name='SCC has ICC',
         cond='Certificate must state vessel is covered under Institute Classification Clause.',
         doc=SCC_HAS_ICC, expect='PASS'),
    dict(group='cz', name='SCC missing ETA (only sailing date)',
         cond='Certificate must show approximate date of arrival at port of destination.',
         doc=SCC_NO_PMR_NO_ETA, expect='FAIL'),
    dict(group='cz', name='SCC has ETA',
         cond='Certificate must show approximate date of arrival at port of destination.',
         doc=SCC_HAS_ETA, expect='PASS'),

    # P198da — F47A charges on DR
    dict(group='da', name='DR missing charges-paid-by-beneficiary',
         cond="Negotiating bank must certify on the documents forwarding "
              "schedule that all their charges and all charges of the advising "
              "bank are paid by the beneficiary.",
         doc=DR_NO_CHARGES, expect='FAIL'),
    dict(group='da', name='DR has charges-paid-by-beneficiary',
         cond="Negotiating bank must certify on the documents forwarding "
              "schedule that all their charges and all charges of the advising "
              "bank are paid by the beneficiary.",
         doc=DR_WITH_CHARGES, expect='PASS'),

    # P198cl — Proforma citation in submission
    dict(group='cl', name='CI cites PI 786/S-13198-SOYPI-E (PASS)',
         cond="Beneficiary's Proforma Invoice Ref. No. 786/S-13198-SOYPI-E "
              "dated Jan 21, 2026 must be present in the submission.",
         doc=CI_BRAZIL_PROFORMA, expect='PASS'),
    dict(group='cl', name='CI without PI citation (FAIL)',
         cond="Beneficiary's Proforma Invoice Ref. No. 786/S-13198-SOYPI-E "
              "must be present in the submission.",
         doc=CI_BRAZIL, expect='FAIL'),

    # P198co — CoO Chamber-of-Commerce equivalence
    dict(group='co', name='CoO issued by Wuhan Customs (equivalent issuer)',
         cond='Certificate of Origin must be issued / certified by Chamber '
              'of Commerce in the country of exporter.',
         doc='CERTIFICATE OF ORIGIN\nIssued in THE PEOPLE\'S REPUBLIC OF CHINA\n'
             'Stamp: Wuhan Customs\n', expect='PASS'),
    dict(group='co', name='CoO from random Trading Co (not equivalent)',
         cond='Certificate of Origin must be issued / certified by Chamber '
              'of Commerce in the country of exporter.',
         doc='CERTIFICATE OF ORIGIN\nIssued by ABC Trading Co. Ltd.', expect='FAIL'),

    # P198cs — strict freight-wording
    dict(group='cs', name='BL has "FREIGHT COLLECT" adjacent (PASS)',
         cond='Bill of Lading must show freight collect.',
         doc='BILL OF LADING\nFREIGHT COLLECT\n', expect='PASS'),
    dict(group='cs', name='BL has "COLLECT" alone (FAIL)',
         cond='Bill of Lading must show freight collect.',
         doc='BILL OF LADING\nAMOUNT COLLECT: USD 1500\n', expect='FAIL'),
]


def main():
    print(f"LLM endpoint: {LLM_URL}\n")
    pass_count = fail_count = 0
    by_group = {}
    for i, sc in enumerate(SC, 1):
        t0 = time.time()
        v, r = llm_call(sc['cond'], sc['doc'])
        elapsed = time.time() - t0
        ok = (v == sc['expect'])
        tag = 'OK ' if ok else 'FAIL'
        bg = sc['group']
        by_group.setdefault(bg, [0, 0])
        if ok:
            pass_count += 1; by_group[bg][0] += 1
        else:
            fail_count += 1; by_group[bg][1] += 1
        print(f"[{tag}] [{bg}] #{i:02d}  {sc['name']}  ({elapsed:.1f}s)")
        print(f"        expect={sc['expect']}  got={v}  | {r[:140]}")
    print()
    print('='*78)
    print(f'LLM agreement: {pass_count}/{pass_count+fail_count} scenarios OK')
    for g, (p, f) in sorted(by_group.items()):
        print(f'  group {g}: {p}/{p+f}')
    print('='*78)
    return fail_count == 0


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
