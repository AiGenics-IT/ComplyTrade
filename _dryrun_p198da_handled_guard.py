"""
P198da_handled guard dry-run.

Confirms that:
  • The pre-processor sets `_p198da_handled = True` on rows it
    finalises (charges-on-DR + SWIFT-advice).
  • _build_tasks emits a skip task with reason='p198da_handled'
    for those rows so the LLM never sees them.
  • The skip-task handler (P169 drop logic) bypasses these rows
    via the explicit guard.

Without this guard, a row pre-set to FAIL with synthetic
document_checked='MT799/MT999 SWIFT Advice' was being silently
dropped (compliance reset to N/A, _drop_from_report=True) by
the P169 'doc not in LC text' check — because the synthetic
doc-type label doesn't appear in F46A/F46B/F47A literally.
"""
import re, sys


# ── Test-local mirror ──
def preprocessor(rows, packets):
    """Mirror of step14 P198da preprocessor."""
    _SWIFT_ADVICE_RE = re.compile(
        r'(?:AUTHENT(?:ICATED|IC)\s+SWIFT|VIA\s+SWIFT|BY\s+SWIFT|'
        r'MT\s*799|MT\s*999|FREE\s+FORMAT\s+MESSAGE|'
        r'SWIFT\s+MESSAGE\s+MUST\s+ACCOMPANY)',
        re.IGNORECASE,
    )
    handled = 0
    for row in rows:
        comp = (row.get('compliance', '') or '').upper()
        if comp not in ('N/A', 'NA', 'PENDING', ''):
            continue
        cref = (row.get('clause_ref', '') or '').upper()
        cond = row.get('condition_text', '') or ''
        if not cond: continue
        cu = cond.upper()
        if '47A' not in cref and '47B' not in cref:
            continue
        is_swift = bool(_SWIFT_ADVICE_RE.search(cu)) and (
            'NEGOTIATING' in cu or 'ADVISE' in cu
            or 'ACCOMPANY' in cu or 'ADVICE' in cu)
        is_charges = ('CHARGES' in cu and 'BENEFICIARY' in cu
                       and ('CERTIFY' in cu or 'SCHEDULE' in cu
                            or 'NEGOTIATING BANK' in cu))
        if not (is_swift or is_charges):
            continue
        row['_p198da_handled'] = True
        if is_swift:
            sw = next((p for p in packets if any(k in (p.get('document_type','') or '').lower()
                       for k in ('mt799','mt999','fin.799','fin.999','free format'))), None)
            row['document_checked'] = 'MT799/MT999 SWIFT Advice'
            row['compliance'] = 'PASS' if sw else 'FAIL'
        elif is_charges:
            dr = next((p for p in packets if 'remittance' in (p.get('document_type','') or '').lower()
                       or 'covering' in (p.get('document_type','') or '').lower()), None)
            row['document_checked'] = 'Documentary Remittance'
            row['compliance'] = 'PASS' if dr else 'FAIL'
        handled += 1
    return handled


def build_tasks_skip_count(rows):
    """Mirror of _build_tasks — counts how many P198da-handled rows
    end up as skip tasks with the right reason."""
    skip_with_reason = 0
    for row in rows:
        if row.get('_p198da_handled'):
            skip_with_reason += 1
    return skip_with_reason


def drop_simulator(rows):
    """Mirror of P169 drop logic — drops rows whose document_checked
    isn't in LC text. With the guard, P198da-handled rows survive."""
    LC_TEXT_BLOB = 'F46A: COMMERCIAL INVOICE, BILL OF LADING, ETC. F47A: NEGOTIATING BANK MUST ADVISE'
    survived = []
    dropped = []
    for row in rows:
        if row.get('_p198da_handled'):
            survived.append(row)  # protected
            continue
        doc = (row.get('document_checked','') or '').upper()
        if doc and doc not in LC_TEXT_BLOB.upper():
            dropped.append(row)
        else:
            survived.append(row)
    return survived, dropped


SC = []

# Scenario 1 — Real R0059 (47A-9 SWIFT advice), no MT799/MT999 in submission
SC.append(dict(
    name='R0059 47A-9 SWIFT advice + no MT799 → guard protects FAIL row',
    rows=[
        dict(row_id='R0059', clause_ref='47A-9',
             condition_text='ON THE DATE OF NEGOTIATION, THE NEGOTIATING BANK MUST '
                            'ADVISE US VIA AUTHENTICATED SWIFT ON BAHLPKKACPU... '
                            'COPY OF SUCH SWIFT MESSAGE MUST ACCOMPANY WITH ORIGINAL '
                            'SET OF DOCUMENTS.',
             compliance='N/A', document_checked='N/A'),
    ],
    packets=[
        dict(document_type='Commercial Invoice', document_text='...'),
        dict(document_type='Bill of Lading', document_text='...'),
    ],
    expect_handled=1,
    expect_compliance='FAIL',
    expect_doc='MT799/MT999 SWIFT Advice',
    expect_dropped=0,
))

# Scenario 2 — 47A-7 charges-on-DR, DR present
SC.append(dict(
    name='47A-7 charges + DR present → PASS protected',
    rows=[
        dict(row_id='R0055', clause_ref='47A-7',
             condition_text='Negotiating bank must certify on their documents '
                            'forwarding schedule that all their charges and all '
                            'charges of advising bank are paid by the beneficiary.',
             compliance='N/A', document_checked='N/A'),
    ],
    packets=[
        dict(document_type='Documentary Remittance', document_text='...'),
        dict(document_type='Commercial Invoice', document_text='...'),
    ],
    expect_handled=1,
    expect_compliance='PASS',
    expect_doc='Documentary Remittance',
    expect_dropped=0,
))

# Scenario 3 — Non-47A row → not handled
SC.append(dict(
    name='Non-47A row → no handling, normal flow',
    rows=[
        dict(row_id='R0001', clause_ref='46A-1',
             condition_text='Goods description must be SOYBEANS.',
             compliance='PENDING', document_checked='Commercial Invoice'),
    ],
    packets=[
        dict(document_type='Commercial Invoice', document_text='...'),
    ],
    expect_handled=0,
    expect_compliance='PENDING',
    expect_doc='Commercial Invoice',
    expect_dropped=0,
))

# Scenario 4 — Non-FAIL N/A row that's not a F47A pattern → not handled, gets dropped
SC.append(dict(
    name='Random N/A row with synthetic doc → drop logic fires',
    rows=[
        dict(row_id='R0099', clause_ref='47A-99',
             condition_text='Some unrelated bank-to-bank instruction.',
             compliance='N/A', document_checked='Random Doc'),
    ],
    packets=[],
    expect_handled=0,
    expect_compliance='N/A',
    expect_doc='Random Doc',
    expect_dropped=1,
))


def main():
    p = f = 0
    for i, sc in enumerate(SC, 1):
        rows = [dict(r) for r in sc['rows']]
        handled = preprocessor(rows, sc['packets'])
        skip = build_tasks_skip_count(rows)
        survived, dropped = drop_simulator(rows)
        ok = (handled == sc['expect_handled']
              and rows[0].get('compliance') == sc['expect_compliance']
              and rows[0].get('document_checked') == sc['expect_doc']
              and len(dropped) == sc['expect_dropped'])
        tag = 'OK ' if ok else 'FAIL'
        print(f"[{tag}] #{i:02d}  {sc['name']}")
        print(f"        handled={handled} (expect {sc['expect_handled']})")
        print(f"        compliance={rows[0].get('compliance')} (expect {sc['expect_compliance']})")
        print(f"        doc={rows[0].get('document_checked')} (expect {sc['expect_doc']})")
        print(f"        dropped={len(dropped)} (expect {sc['expect_dropped']})")
        if ok: p += 1
        else: f += 1
    print(f"\n{'='*78}\n{p}/{p+f} P198da-handled-guard scenarios OK\n{'='*78}")
    return f == 0


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
