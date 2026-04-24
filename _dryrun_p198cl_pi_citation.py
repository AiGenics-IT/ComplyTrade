"""
P198cl dry-run — Proforma Invoice "must be present in submission"
citation rescue.

When the LC says:
    "Beneficiary's Proforma Invoice Ref.No. 786/S-13198-SOYPI-E
     dated Jan 21, 2026 must be present in the submission"
and NO separate Proforma Invoice document is submitted, the
previous verification path fell back to the Commercial Invoice
and mis-read the CI's own invoice_reference (MCI-786/S-13198-SOY-E)
as the "Proforma Invoice reference number" — producing a false
FAIL. This rescue scans all packets' structured refs and body
text for the literal PI reference and rescues FAIL→PASS when
found anywhere in the submission.

Uses real packet data from job 71caab39 (LC 1089LC59947/2026):
  pkt_20 CI: proforma_reference="786/S-13198 SOYPI-E", body has
             "PROFORMA INVOICE REF.NO.786/S-13198 SOYPI-E"
  pkt_21 CI: body has the PI reference (no structured role)
  pkt_22 CI: proforma_reference="786/S-13198 SOYP I-E" (space scatter)
  pkt_23 CI: ONLY invoice_reference="MCI-786/S-13198-SOY-E"
             (the CI packet that triggered the false FAIL)

Also synthetic scenarios for OCR scatter, separator variation,
absence of the PI ref (genuine missing PI), and interaction with
R0004-style "strictly follow" rows (must NOT rescue those).
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import re
from steps.step14_verification import _normalize_id


# Test-local mirror of P198cl logic.
_PRESENCE_PATTERNS = re.compile(
    r'\b(?:must|shall|should|is\s+to|to)\s+'
    r'(?:be\s+)?'
    r'(?:present|shown|cited|referenced|included|'
    r'indicated|mentioned|attached|submitted|appear|'
    r'quoted|noted)\b',
    flags=re.IGNORECASE,
)
_STRICT_PATTERNS = re.compile(
    r'\b(?:strictly\s+(?:as\s+per|follow)|'
    r'binding\s+(?:both|on)|'
    r'date\s+must|must\s+match\s+the\s+date|'
    r'shall\s+match\s+the\s+date)\b',
    flags=re.IGNORECASE,
)


def simulate(cond, doc_type, compliance, all_packets):
    if compliance != 'FAIL':
        return compliance, 'not FAIL; no rescue attempted'
    if 'proforma' not in doc_type.lower():
        return 'FAIL', 'document_checked != Proforma Invoice'
    if 'PROFORMA' not in cond.upper():
        return 'FAIL', 'condition does not mention proforma'
    if _STRICT_PATTERNS.search(cond):
        return 'FAIL', 'strict-match row; handled by P198ak date check'
    if not _PRESENCE_PATTERNS.search(cond):
        return 'FAIL', 'condition does not ask for presence'

    # Extract PI tokens
    ids_raw = re.findall(
        r'[A-Z0-9][A-Z0-9/\-._]{5,}[A-Z0-9]',
        cond, flags=re.IGNORECASE,
    )
    ids = [t for t in ids_raw if re.search(r'\d', t)]
    if not ids:
        return 'FAIL', 'no identifier tokens in condition'

    # Flatten packets
    refs = []
    texts = []
    for p in all_packets:
        dt = p.get('document_type', '?')
        us = p.get('unified_summary') or {}
        if isinstance(us, dict):
            for r in (us.get('references_found') or []):
                if isinstance(r, dict):
                    v = str(r.get('value', '') or '')
                    if v:
                        refs.append((dt, _normalize_id(v), v,
                                     str(r.get('role', '') or '')))
            for k in ('proforma_reference', 'invoice_reference',
                     'contract_reference'):
                tv = str(us.get(k, '') or '')
                if tv:
                    refs.append((dt, _normalize_id(tv), tv, k))
        txt = p.get('document_text', '') or p.get('cleaned_text', '') or ''
        if txt:
            texts.append((dt, txt.upper()))

    best = None
    for needle in ids:
        n_norm = _normalize_id(needle)
        if len(n_norm) < 6:
            continue
        # P1: proforma_reference role
        for dt, v_norm, raw, role in refs:
            if role.lower() not in ('proforma_reference', 'pi_reference'):
                continue
            if v_norm and (v_norm == n_norm or n_norm in v_norm or v_norm in n_norm):
                best = (dt, raw, f'structured proforma_reference on {dt}')
                break
        if best: break
        # P2: exact match any role
        for dt, v_norm, raw, role in refs:
            if v_norm == n_norm:
                best = (dt, raw, f'structured {role} on {dt} (exact)')
                break
        if best: break
        # P3: body text
        for dt, txt_up in texts:
            txt_norm = _normalize_id(txt_up)
            if n_norm and n_norm in txt_norm:
                pos = txt_up.find(needle.upper())
                quote = needle if pos < 0 else txt_up[max(0, pos-30):pos+len(needle)+30].strip()
                best = (dt, quote, f'body text on {dt}')
                break
        if best: break

    if not best:
        return 'FAIL', 'PI ref not found in any packet'
    return 'PASS', f'P198cl rescue: {best[2]}'


# ── Real job data ──
REAL_PACKETS = [
    # pkt_20
    dict(document_type='Commercial Invoice',
         document_text='...PROFORMA INVOICE REF.NO.786/S-13198 SOYPI-E DATED FEB 18, 2026 CFR KARACHI PORT...',
         unified_summary=dict(
             invoice_reference='786/S-13198-SOY',
             references_found=[
                 dict(role='lc_reference', value='1089LC59947/2026'),
                 dict(role='invoice_reference', value='786/S-13198-SOY'),
                 dict(role='proforma_reference', value='786/S-13198 SOYPI-E'),
                 dict(role='hs_code', value='1201.9000'),
             ]
         )),
    # pkt_21
    dict(document_type='Commercial Invoice',
         document_text='... beneficiary commercial invoice ...',
         unified_summary=dict(
             invoice_reference='786/S-13198-SOY',
             references_found=[
                 dict(role='lc_reference', value='1089LC59947/2026'),
                 dict(role='invoice_reference', value='786/S-13198-SOY'),
                 dict(role='hs_code', value='1201.9000'),
             ]
         )),
    # pkt_22
    dict(document_type='Commercial Invoice',
         document_text='... proforma ref 786/S-13198 SOYP I-E ...',
         unified_summary=dict(
             invoice_reference='786/S-13198-SOY',
             references_found=[
                 dict(role='lc_reference', value='1089LC59947/2026'),
                 dict(role='invoice_reference', value='786/S-13198-SOY'),
                 dict(role='proforma_reference', value='786/S-13198 SOYP I-E'),
             ]
         )),
    # pkt_23 — the misleading packet with the CI's own identity
    dict(document_type='Commercial Invoice',
         document_text='COMMERCIAL INVOICE NO. MCI-786/S-13198-SOY-E\nDATE: FEB 18 2026...',
         unified_summary=dict(
             invoice_reference='MCI-786/S-13198-SOY-E',
             references_found=[
                 dict(role='lc_reference', value='1089LC59947/2026'),
                 dict(role='invoice_reference', value='MCI-786/S-13198-SOY-E'),
             ]
         )),
]


SC = []

# Scenario 1 — Real job case: R0041 with full packet set should PASS
SC.append(dict(
    name='Real job 71caab39 R0041: full packet set',
    cond="Beneficiary's Proforma Invoice Ref. No. 786/S-13198-SOYPI-E dated Jan 21, 2026 must be present in the submission.",
    doc='Proforma Invoice',
    compliance='FAIL',
    packets=REAL_PACKETS,
    expect='PASS',
))

# Scenario 2 — Only pkt_23 exists (the misleading CI) — should FAIL
# (PI ref genuinely NOT in any submitted doc)
SC.append(dict(
    name='Only misleading CI pkt_23 (no PI citation anywhere) → STAY FAIL',
    cond="Beneficiary's Proforma Invoice Ref. No. 786/S-13198-SOYPI-E dated Jan 21, 2026 must be present in the submission.",
    doc='Proforma Invoice',
    compliance='FAIL',
    packets=[REAL_PACKETS[3]],
    expect='FAIL',
))

# Scenario 3 — Only pkt_20 (has proforma_reference structured) → PASS
SC.append(dict(
    name='Only pkt_20 with structured proforma_reference → PASS',
    cond="Beneficiary's Proforma Invoice Ref. No. 786/S-13198-SOYPI-E dated Jan 21, 2026 must be present in the submission.",
    doc='Proforma Invoice',
    compliance='FAIL',
    packets=[REAL_PACKETS[0]],
    expect='PASS',
))

# Scenario 4 — Only pkt_22 with OCR-scattered proforma_reference → PASS
SC.append(dict(
    name='Only pkt_22 with OCR-scattered proforma_reference → PASS',
    cond="Beneficiary's Proforma Invoice Ref. No. 786/S-13198-SOYPI-E dated Jan 21, 2026 must be present in the submission.",
    doc='Proforma Invoice',
    compliance='FAIL',
    packets=[REAL_PACKETS[2]],
    expect='PASS',
))

# Scenario 5 — R0004 "strictly follow" date-binding row should NOT
# be rescued (real date mismatch must stay FAIL).
SC.append(dict(
    name='R0004 strictly-follow row → STAY FAIL (handled by P198ak)',
    cond='Further details and specifications must strictly follow Beneficiary\'s Proforma Invoice Ref.No. 786/S-13198-SOYPI-E dated Jan 21, 2026 on the Commercial Invoice.',
    doc='Commercial Invoice',
    compliance='FAIL',
    packets=REAL_PACKETS,
    expect='FAIL',
))

# Scenario 6 — Already PASS (no rescue needed)
SC.append(dict(
    name='Already PASS — no change',
    cond="Beneficiary's Proforma Invoice Ref. No. 786/S-13198-SOYPI-E must be present in the submission.",
    doc='Proforma Invoice',
    compliance='PASS',
    packets=REAL_PACKETS,
    expect='PASS',
))

# Scenario 7 — Completely different PI ref (genuinely missing) → STAY FAIL
SC.append(dict(
    name='Different PI ref not cited anywhere → STAY FAIL',
    cond="Beneficiary's Proforma Invoice Ref. No. 9999/XYZ-99999-DIFF-Z must be present in the submission.",
    doc='Proforma Invoice',
    compliance='FAIL',
    packets=REAL_PACKETS,
    expect='FAIL',
))

# Scenario 8 — Condition uses "must be shown on the Commercial Invoice"
SC.append(dict(
    name='Condition says "must be shown on the invoice" → PASS via body text',
    cond="Proforma Invoice Ref.No. 786/S-13198-SOYPI-E dated Jan 21, 2026 must be shown on the commercial invoice.",
    doc='Proforma Invoice',
    compliance='FAIL',
    packets=REAL_PACKETS,
    expect='PASS',
))

# Scenario 9 — OCR hyphen/space variation: condition has hyphens, BL body has spaces
SC.append(dict(
    name='Condition uses hyphens, body uses spaces → PASS via OCR normalization',
    cond="Proforma Invoice Ref. 786-S-13198-SOYPI-E must be present in the submission.",
    doc='Proforma Invoice',
    compliance='FAIL',
    packets=[REAL_PACKETS[0]],
    expect='PASS',
))

# Scenario 10 — condition has lowercase
SC.append(dict(
    name='Condition lowercase → PASS',
    cond="beneficiary's proforma invoice ref no 786/S-13198-SOYPI-E must be present in the submission.",
    doc='Proforma Invoice',
    compliance='FAIL',
    packets=REAL_PACKETS,
    expect='PASS',
))

# Scenario 11 — document_checked not Proforma Invoice → should not run
SC.append(dict(
    name='Different doc type (e.g. BL) → no rescue',
    cond="Bill of Lading must reference Proforma Invoice 786/S-13198-SOYPI-E and be present in submission.",
    doc='Bill of Lading',
    compliance='FAIL',
    packets=REAL_PACKETS,
    expect='FAIL',
))

# Scenario 12 — Condition is purely about date and not presence → no rescue
SC.append(dict(
    name='Condition about date without "present" verb → no rescue',
    cond="Proforma Invoice Ref. No. 786/S-13198-SOYPI-E dated Jan 21, 2026.",
    doc='Proforma Invoice',
    compliance='FAIL',
    packets=REAL_PACKETS,
    expect='FAIL',
))

# Scenario 13 — Empty packet list
SC.append(dict(
    name='Empty packet list → STAY FAIL',
    cond="Beneficiary's Proforma Invoice Ref. No. 786/S-13198-SOYPI-E must be present in the submission.",
    doc='Proforma Invoice',
    compliance='FAIL',
    packets=[],
    expect='FAIL',
))

# Scenario 14 — Multiple PI refs in condition; only one present
SC.append(dict(
    name='Multiple PI refs in condition; any hit → PASS',
    cond="Proforma Invoice Ref.No. 786/S-13198-SOYPI-E or 786/S-13198-MAIZE-Z must be present.",
    doc='Proforma Invoice',
    compliance='FAIL',
    packets=REAL_PACKETS,
    expect='PASS',
))

# Scenario 15 — PI ref spelled with O/0 confusion (OCR)
SC.append(dict(
    name='PI ref OCR O↔0 fold → PASS',
    cond="Proforma Invoice Ref. 786/S-13198-S0YPI-E must be present.",  # "0" instead of "O"
    doc='Proforma Invoice',
    compliance='FAIL',
    packets=REAL_PACKETS,
    expect='PASS',
))


def main():
    passed = 0; failed = 0
    for i, sc in enumerate(SC, 1):
        verdict, note = simulate(sc['cond'], sc['doc'], sc['compliance'], sc['packets'])
        ok = (verdict == sc['expect'])
        tag = 'OK ' if ok else 'FAIL'
        print(f"[{tag}] #{i:02d}  {sc['name']}")
        print(f"         expect={sc['expect']}, got={verdict}")
        print(f"         note: {note}")
        if ok: passed += 1
        else: failed += 1
    print(f"\n{'='*78}\n{passed}/{passed+failed} P198cl scenarios OK\n{'='*78}")
    return failed == 0


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
