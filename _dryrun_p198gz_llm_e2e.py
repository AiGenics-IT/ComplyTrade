"""End-to-end LLM probe — call the actual Qwen LLM with the new prompt
on the disputed conditions from job 104ac15f and 94edb6a7. Verify the
LLM gets each right."""
import sys, os, json, requests
sys.path.insert(0, 'd:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final')
os.environ['PYTHONIOENCODING'] = 'utf-8'

from dotenv import load_dotenv
load_dotenv('d:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final/.env')
URL = os.getenv('QWEN_TEXT_LLM_URL')
MODEL = os.getenv('QWEN_TEXT_LLM_MODEL')
print(f"LLM endpoint: {URL}\nModel: {MODEL}\n")


def call(prompt, max_tok=400):
    r = requests.post(URL, json={
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tok,
        "temperature": 0.0,
    }, timeout=120)
    return r.json()['choices'][0]['message']['content']


# Load job data
JOB = 'd:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final/results/104ac15f-56ca-4499-badf-aaf3b92f401c'
d8 = json.load(open(f'{JOB}/step08/step08_result.json', encoding='utf-8'))
draft = next(p for p in d8['classified_packets']
             if 'draft' in (p.get('document_type','') or '').lower()
             or 'bill of exchange' in (p.get('document_type','') or '').lower())
awb = next(p for p in d8['classified_packets']
           if 'airway' in (p.get('document_type','') or '').lower())
pl = next(p for p in d8['classified_packets']
          if 'packing' in (p.get('document_type','') or '').lower())

draft_text = (draft.get('cleaned_text') or draft.get('raw_text') or '')[:2500]
awb_text = (awb.get('cleaned_text') or awb.get('raw_text') or '')[:2500]
pl_text = (pl.get('cleaned_text') or pl.get('raw_text') or '')[:2500]


# ── Probe 1 — Draft drawee (LLM had been confusing PAYEE with DRAWEE) ──
print("=" * 70)
print("PROBE 1: Draft drawee (PAYEE-vs-DRAWEE rule)")
print("=" * 70)
prompt1 = f"""You are a trade finance examiner. Read CAREFULLY.

ANTI-CONFUSION RULE — bill of exchange has THREE roles:
- DRAWER: who issues the draft (the beneficiary/exporter)
- DRAWEE: party on whom the draft is drawn = LC ISSUING BANK,
  identified by "Drawn under L/C No. X — Issued by <Bank>" or
  a "DRAWEE: <Bank>" label.
- PAYEE: identified ONLY by "Pay to the Order of <X>" — this is
  the COLLECTING / NEGOTIATING bank, NOT the drawee.

DO NOT confuse "Pay to the Order of X" with the drawee. X is the
PAYEE, not the drawee.

LC required drawee: BANK AL HABIB LIMITED, KARACHI, PAKISTAN

Draft text:
\"\"\"
{draft_text}
\"\"\"

Q: Is the draft drawn on Bank Al Habib? Answer with JSON:
{{"verdict": "PASS" or "FAIL",
  "drawee_name": "...",
  "payee_name": "...",
  "reasoning": "..."}}"""
try:
    out = call(prompt1)
    print(out[:1000])
except Exception as e:
    print(f"ERR: {e}")
print()

# ── Probe 2 — AWB flight number (was hallucinating "no flight number") ──
print("=" * 70)
print("PROBE 2: AWB flight number ID")
print("=" * 70)
prompt2 = f"""You are a trade finance examiner.

AWB FLIGHT-IDENTIFICATION RULE — ALL of the following count as the
flight number on an AWB (do not require the literal label "Flight Number:"):
- IATA code "UL 0153", "EK 401", "CZ8212"
- Carrier reference like "SA250900311" (Sinotech Air)
- IATA AWB number "NNN-NNNNNNNN" or "NNN ORIG NNNNNNNN"
  (e.g. "603-74213252", "784 PVG 41181022")
- "Requested Flight/Date" field

LC requires: "AWB must bear the flight number".

AWB text:
\"\"\"
{awb_text}
\"\"\"

Q: Does the AWB bear a flight identification per the rule above?
Answer JSON:
{{"verdict": "PASS" or "FAIL",
  "flight_evidence_quoted": "exact text from AWB",
  "reasoning": "..."}}"""
try:
    out = call(prompt2)
    print(out[:1000])
except Exception as e:
    print(f"ERR: {e}")
print()

# ── Probe 3 — Incoterm version on Packing List (should NOT fail) ──
print("=" * 70)
print("PROBE 3: Incoterm version on Packing List (should not flag)")
print("=" * 70)
prompt3 = f"""You are a trade finance examiner.

INCOTERM SCOPE RULE — Incoterm version compliance ONLY applies to
documents that normally carry the trade-term: Commercial Invoice
(always), Bill of Lading / Airway Bill (sometimes), Insurance.

DO NOT raise "missing Incoterm version" on a Packing List, Weight
List, Health Cert, Bene Cert, etc. — these don't carry the trade-term.
Only fail them if they DO state an Incoterm AND the version is wrong.

LC condition: "All documents must show Incoterms 2020"
Document being checked: Packing List

Packing List text:
\"\"\"
{pl_text}
\"\"\"

Q: Is missing Incoterm version on this Packing List a discrepancy?
Answer JSON:
{{"verdict": "PASS" or "FAIL" or "N/A",
  "reasoning": "..."}}"""
try:
    out = call(prompt3)
    print(out[:800])
except Exception as e:
    print(f"ERR: {e}")
print()


# ── Probe 4 — AWB notify party (was hallucinating PASS) ──
print("=" * 70)
print("PROBE 4: AWB notify party (anti-hallucination)")
print("=" * 70)
prompt4 = f"""You are a trade finance examiner.

ANTI-HALLUCINATION RULE: A "Notify Party" requirement is only
satisfied when the named party LITERALLY appears under a
"NOTIFY PARTY" field on the AWB. If there is NO Notify Party block
on the AWB, the requirement CANNOT be met (regardless of what other
roles the party plays). Do NOT pass based on inference.

LC condition: "AWB must be marked notify the Applicant and Bank Al Habib Limited, Pakistan."

AWB text:
\"\"\"
{awb_text}
\"\"\"

Q: Does the AWB have a Notify Party block listing the Applicant and
Bank Al Habib? Answer JSON:
{{"verdict": "PASS" or "FAIL",
  "notify_block_present": true or false,
  "quote_from_awb": "exact text or empty",
  "reasoning": "..."}}"""
try:
    out = call(prompt4)
    print(out[:800])
except Exception as e:
    print(f"ERR: {e}")
print()
