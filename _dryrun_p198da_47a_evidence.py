"""
P198da dry-run — F47A "needs evidence" recognizer.

Two F47A patterns currently dropped as informational by step 12
should be verified instead:

  • Charges-on-Forwarding-Schedule — the negotiating bank must
    certify on the documents forwarding schedule that charges
    are paid by the beneficiary. Verifiable on the Documentary
    Remittance / Covering Schedule.

  • Authenticated SWIFT advice (MT799 / MT999) — the negotiating
    bank must advise the issuing bank via authenticated SWIFT,
    and a copy of the SWIFT message must accompany the original
    documents. Verifiable by looking for an MT799 / MT999
    packet in the submission.

This harness exercises both the deterministic recognizer and
sends the same scenarios to the live LLM at
http://136.112.48.34/v1/chat/completions to confirm agreement.
"""
import json
import re
import sys
import time
import urllib.request


LLM_URL = "http://136.112.48.34/v1/chat/completions"
MODEL = "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8"


# ── Deterministic logic (mirror of P198da in step14_verification) ──
_NEG_BANK_CHARGES_RE = re.compile(
    r'(?:NEGOTIAT(?:ING|ION)|FORWARDING)\s+(?:BANK|SCHEDULE)[^.]{0,200}?'
    r'(?:CHARGES?|FEE|EXPENSE)[^.]{0,200}?'
    r'(?:PAID|BORNE|PAYABLE|ACCOUNT)\s+(?:BY|OF|AT)?\s*'
    r'(?:THE\s+)?(?:BENEFICIARY|APPLICANT)',
    re.IGNORECASE | re.DOTALL,
)
_NEG_BANK_CHARGES_ALT_RE = re.compile(
    r'CERTIF(?:Y|ICATION)[^.]{0,200}?CHARGES?[^.]{0,200}?'
    r'(?:PAID|BORNE)[^.]{0,40}?BENEFICIARY',
    re.IGNORECASE | re.DOTALL,
)
_SWIFT_ADVICE_RE = re.compile(
    r'(?:AUTHENT(?:ICATED|IC)\s+SWIFT|VIA\s+SWIFT|BY\s+SWIFT|'
    r'MT\s*799|MT\s*999|FREE\s+FORMAT\s+MESSAGE|'
    r'SWIFT\s+MESSAGE\s+MUST\s+ACCOMPANY)',
    re.IGNORECASE,
)
_DR_CHARGES_DOC_RE = re.compile(
    r'(?:ALL\s+)?(?:OUR\s+)?CHARGES?\s+'
    r'(?:AND\s+(?:ALL\s+)?(?:OUR\s+)?CHARGES?\s+OF\s+'
    r'(?:THE\s+)?ADVISING\s+BANK\s+)?'
    r'(?:ARE\s+|TO\s+BE\s+)?'
    r'(?:PAID|BORNE|FOR\s+(?:THE\s+)?ACCOUNT)\s+'
    r'(?:OF|BY)\s+(?:THE\s+)?BENEFICIARY',
    re.IGNORECASE | re.DOTALL,
)


def has_dr(packets):
    for p in packets or []:
        dt = (p.get('document_type') or '').lower()
        if any(k in dt for k in ('document remittance', 'documentary remittance',
                                  'covering schedule', 'covering letter',
                                  'cover letter', 'cover schedule',
                                  'bills schedule', 'forwarding schedule')):
            return p
    return None


def has_swift(packets):
    for p in packets or []:
        dt = (p.get('document_type') or '').lower()
        if any(k in dt for k in ('mt799', 'mt 799', 'mt999', 'mt 999',
                                  'fin.799', 'fin.999',
                                  'free format message', 'free-format message',
                                  'authenticated swift', 'swift advice',
                                  'swift message')):
            return p
    return None


def deterministic(cond, packets, current='N/A'):
    if (current or '').upper() not in ('N/A', 'NA', 'PENDING', ''):
        return current, 'not N/A'
    cu = cond.upper()
    is_charges = bool(
        _NEG_BANK_CHARGES_RE.search(cond) or _NEG_BANK_CHARGES_ALT_RE.search(cond)
    ) and 'FORWARDING' in cu or (
        'CHARGES' in cu and 'BENEFICIARY' in cu
        and ('CERTIFY' in cu or 'SCHEDULE' in cu or 'NEGOTIATING BANK' in cu)
    )
    is_swift = bool(_SWIFT_ADVICE_RE.search(cu)) and (
        'NEGOTIATING' in cu or 'ADVISE' in cu or 'ACCOMPANY' in cu or 'ADVICE' in cu
    )
    if is_charges:
        dr = has_dr(packets)
        if not dr:
            return 'FAIL', 'DR missing'
        txt = dr.get('document_text', '') or dr.get('cleaned_text', '')
        if _DR_CHARGES_DOC_RE.search(txt):
            return 'PASS', 'DR carries charges-paid-by-beneficiary'
        return 'FAIL', 'DR present, no charges statement'
    if is_swift:
        sw = has_swift(packets)
        return ('PASS', f'SWIFT advice ({sw.get("document_type")})') if sw else ('FAIL', 'no MT799/MT999')
    return current, 'no F47A pattern matched'


# ── LLM logic ──
LLM_PROMPT = """You are a strict UCP 600 trade finance document examiner.

LC F47A clause:
\"\"\"{cond}\"\"\"

Documents that were submitted in this presentation (one entry per packet, with
document type and the first 600 characters of its body text):
{pkt_listing}

Decide whether the LC condition is satisfied by the submitted presentation.

Important rules:
1. The condition must be met by some document IN THE SUBMISSION above. Identity
   of the issuing bank or LC is NOT a substitute.
2. If the condition demands an authenticated SWIFT message (MT799 / MT999), look
   for a packet whose document_type clearly identifies one (e.g. "MT799",
   "MT999", "Free-Format Message", "Authenticated SWIFT Advice"). The mere
   mention of "SWIFT" inside another document does NOT satisfy this.
3. If the condition demands certification on a Forwarding Schedule / Documentary
   Remittance / Covering Schedule, the literal certification must appear on
   that document.

Return ONLY a JSON object:
{{"result":"PASS|FAIL","reason":"one short sentence quoting evidence or noting absence"}}
"""


def _pkt_listing(packets):
    out = []
    for p in (packets or [])[:8]:
        dt = p.get('document_type', '?')
        txt = (p.get('document_text') or p.get('cleaned_text') or '')[:600]
        out.append(f"- type={dt}\n  body: {txt}")
    return "\n".join(out) if out else "(no packets)"


def llm_call(cond, packets, max_retries=2):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": LLM_PROMPT.format(
            cond=cond, pkt_listing=_pkt_listing(packets))}],
        "max_tokens": 250,
        "temperature": 0.0,
    }
    data = json.dumps(body).encode("utf-8")
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
        except Exception as e:
            if attempt < max_retries:
                time.sleep(2)
                continue
            return ('LLM_ERR', f'{type(e).__name__}: {e}')


# Real condition text from job 73be98d9
COND_47A_7 = (
    "NEGOTIATING BANK MUST CERTIFY ON THEIR DOCUMENTS FORWARDING\n"
    "SCHEDULE THAT ALL THEIR CHARGES AND ALL CHARGES OF THE ADVISING\n"
    "BANK ARE PAID BY THE BENEFICIARY."
)
COND_47A_9 = (
    "ON THE DATE OF NEGOTIATION, THE NEGOTIATING\n"
    "BANK MUST ADVISE US VIA AUTHENTICATED SWIFT\n"
    "ON BAHLPKKACPU, STATING THE AMOUNT OF NEGOTIATION,\n"
    "BILL OF LADING NUMBER, NAME OF CARRIER, COURIER\n"
    "COMPANY NAME AND RECEIPT NO, NAMES OF THE VESSEL,\n"
    "VOYAGE NUMBER, PORT OF SHIPMENT, PORT OF LOADING AND\n"
    "PORT OF DISCHARGE, CONTAINER NUMBER, SEAL NUMBER AND\n"
    "DATE OF DISPATCH OF DOCUMENTS.\n"
    "COPY OF SUCH SWIFT MESSAGE MUST ACCOMPANY WITH ORIGINAL\n"
    "SET OF DOCUMENTS."
)


# Packet-set fixtures
PKTS_NO_DR_NO_SWIFT = [
    dict(document_type='Commercial Invoice', document_text='INV NO 123\nGOODS...'),
    dict(document_type='Bill of Lading', document_text='B/L 456...'),
]
PKTS_DR_NO_STATEMENT = [
    dict(document_type='Documentary Remittance',
         document_text='WE ENCLOSE DOCUMENTS RELATED TO L/C ABOVE.\n'
                       'TOTAL AMOUNT CLAIMED: USD 100,000.\nPRESENTATION DATE: 01 MAR.'),
    dict(document_type='Commercial Invoice', document_text='INV...'),
]
PKTS_DR_WITH_STATEMENT = [
    dict(document_type='Documentary Remittance',
         document_text='WE ENCLOSE DOCUMENTS RELATED TO L/C.\n'
                       'ALL OUR CHARGES AND ALL CHARGES OF THE ADVISING BANK '
                       'ARE PAID BY THE BENEFICIARY.\nPresentation date: 01 MAR.'),
    dict(document_type='Commercial Invoice', document_text='INV...'),
]
PKTS_NO_SWIFT_HAS_DR = [
    dict(document_type='Documentary Remittance',
         document_text='Standard cover schedule.'),
    dict(document_type='Commercial Invoice', document_text='INV...'),
]
PKTS_HAS_MT799 = [
    dict(document_type='MT799',
         document_text='MT799 Free Format Message\nAmount of negotiation USD 100,000\n'
                       'BL No: 12345\nVessel: KOTA NEKAD\nVoyage: 0204S\n...'),
    dict(document_type='Commercial Invoice', document_text='INV...'),
]


SC = [
    # ---- F47A-7 charges scenarios ----
    dict(group='charges', name='Real R0055: DR missing → FAIL',
         cond=COND_47A_7, packets=PKTS_NO_DR_NO_SWIFT, expect='FAIL'),
    dict(group='charges', name='DR present, no statement → FAIL',
         cond=COND_47A_7, packets=PKTS_DR_NO_STATEMENT, expect='FAIL'),
    dict(group='charges', name='DR with charges-paid-by-beneficiary statement → PASS',
         cond=COND_47A_7, packets=PKTS_DR_WITH_STATEMENT, expect='PASS'),
    # ---- F47A-9 SWIFT scenarios ----
    dict(group='swift', name='Real R0057: no MT799/MT999 → FAIL',
         cond=COND_47A_9, packets=PKTS_NO_SWIFT_HAS_DR, expect='FAIL'),
    dict(group='swift', name='MT799 in submission → PASS',
         cond=COND_47A_9, packets=PKTS_HAS_MT799, expect='PASS'),
]


def main():
    print("Deterministic + LLM dual dry-run\n")
    print(f"LLM endpoint: {LLM_URL}\n")

    det_pass = 0; llm_pass = 0; both_pass = 0; total = len(SC)
    for i, sc in enumerate(SC, 1):
        det_v, det_r = deterministic(sc['cond'], sc['packets'])
        t0 = time.time()
        llm_v, llm_r = llm_call(sc['cond'], sc['packets'])
        elapsed = time.time() - t0

        det_ok = (det_v == sc['expect'])
        llm_ok = (llm_v == sc['expect'])
        both = det_ok and llm_ok

        if det_ok: det_pass += 1
        if llm_ok: llm_pass += 1
        if both: both_pass += 1

        tag_d = 'OK' if det_ok else 'NO'
        tag_l = 'OK' if llm_ok else 'NO'
        print(f"#{i:02d} [{sc['group']}]  {sc['name']}  ({elapsed:.1f}s)")
        print(f"     expect = {sc['expect']}")
        print(f"     [det {tag_d}] = {det_v} | {det_r}")
        print(f"     [llm {tag_l}] = {llm_v} | {llm_r}")
        print()

    print("="*78)
    print(f"Deterministic guard: {det_pass}/{total}  |  "
          f"LLM agreement: {llm_pass}/{total}  |  "
          f"Both agree: {both_pass}/{total}")
    print("="*78)
    return det_pass == total and llm_pass == total


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
