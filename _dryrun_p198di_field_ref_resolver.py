"""
P198di dry-run — guard the 'AS PER FIELD X' resolver against
overwriting multi-clause fields.

Job 08345848's F47A contained 17 numbered conditions. Clause 10
read 'ALL DOCUMENTS SHOWING DESCRIPTION OF GOODS AS SOYBEANS
ACCEPTABLE ONLY INVOICE TO SHOW FULL DESCRIPTION AS PER FIELD
45A'. The pre-existing simple-ref resolver matched the literal
'AS PER FIELD 45A' inside that one clause and replaced the
ENTIRE F47A value with F45A's content (SOYBEANS, QUANTITY, etc.),
wiping out the other 16 conditions.

The resolver now only fires when the field's content is
essentially a bare reference (short, single-line, no numbered
list items). Long multi-clause fields keep their content even
when one of the clauses contains an 'AS PER FIELD X' phrase.
"""
import re, sys


def simulate(field_val, field_map):
    """Mirror of the P198di-guarded resolver."""
    _val = field_val
    _simple_ref_m = re.search(
        r'(?:SEE|REFER(?:\s+TO)?|AS\s+PER)\s+FIELD\s+(\d{2}[A-Z]?)',
        _val, re.IGNORECASE)
    if not _simple_ref_m:
        return _val, 'no reference matched'
    _val_after = _val[:_simple_ref_m.start()] + _val[_simple_ref_m.end():]
    _val_residual = re.sub(
        r'(?:Period\s+for\s+Presentation(?:\s+in\s+Days)?|'
        r'Additional\s+Conditions|Documents\s+Required|'
        r'Description\s+of\s+Goods(?:\s+and/or\s+Services)?|'
        r'Days?\s*:|Narrative\s*:|/|\\|\.|\s)+',
        ' ', _val_after, flags=re.IGNORECASE,
    ).strip()
    _is_multi = bool(
        re.search(r'(?m)^\s*\d+\s*[\)\.]\s', _val)
        or len(_val) > 200
    )
    if (not _is_multi) and len(_val_residual) <= 30:
        ref_tag = _simple_ref_m.group(1)
        ref_val = field_map.get(ref_tag, '')
        return ref_val if ref_val else _val, f'replaced from F{ref_tag}'
    return _val, ('multi-clause field — not replaced'
                  if _is_multi else f'residual={_val_residual!r}')


SC = []

# Real F47A from job 08345848 — 17 numbered conditions, one of
# which mentions "AS PER FIELD 45A". Should NOT be replaced.
F47A_REAL = (
    "1) DOCUMENTS DATED PRIOR TO DATE OF ISSUANCE OF THIS LC ARE\n"
    "ACCEPTABLE.\n"
    "2) DRAFTS AND INVOICE MUST SHOW OUR DOCUMENTARY CREDIT NUMBER,\n"
    "DATE AND NAME OF L/C ISSUING BANK (BANK AL HABIB LIMITED,\n"
    "PAKISTAN).\n"
    "3) SHORT FORM, BLANK BACK ... (etc.)\n"
    "10) ALL DOCUMENTS SHOWING DESCRIPTION OF GOODS AS SOYBEANS "
    "ACCEPTABLE ONLY INVOICE TO SHOW FULL DESCRIPTION AS PER "
    "FIELD 45A\n"
    "11) THIRD PARTY DOCUMENTS ARE ACCEPTABLE...\n"
    "17) IN CASE OF MULTIPLE SHIPMENTS..."
)
F45A_REAL = "SOYBEANS\nQUANTITY 1000 M/TONS (+/-10 PCT)\n"

SC.append(dict(
    name='Real 08345848: F47A with "AS PER FIELD 45A" inside clause 10 → not replaced',
    val=F47A_REAL,
    map={'45A': F45A_REAL, '47A': F47A_REAL},
    expect=F47A_REAL,
))

# Bare reference (typical F48 form): SHOULD be replaced.
SC.append(dict(
    name='F48 = "PLS REFER FIELD 47A" (bare reference) → replace with F47A',
    val='Period for Presentation in Days\nDays: 30\nNarrative: REFER FIELD 47A',
    map={'47A': '15 days from shipment but within expiry'},
    expect='15 days from shipment but within expiry',
))

# Bare "AS PER FIELD 47A." → replace
SC.append(dict(
    name='Bare "AS PER FIELD 47A." → replace',
    val='AS PER FIELD 47A.',
    map={'47A': 'Some F47A content'},
    expect='Some F47A content',
))

# Multi-clause field with numbered items, mentioning F45A in one clause → NOT replaced
SC.append(dict(
    name='F46A clause containing "AS PER FIELD 45A" → not replaced',
    val=(
        '1) Commercial invoice in the name of applicant.\n'
        '2) Bill of Lading shipped on board.\n'
        '3) Goods description AS PER FIELD 45A.\n'
        '4) Other condition.'
    ),
    map={'45A': 'SOYBEANS QUANTITY 1000 MT'},
    expect=(
        '1) Commercial invoice in the name of applicant.\n'
        '2) Bill of Lading shipped on board.\n'
        '3) Goods description AS PER FIELD 45A.\n'
        '4) Other condition.'
    ),
))

# Long single-paragraph (no numbered items) but > 200 chars → NOT replaced
SC.append(dict(
    name='Long paragraph (>200 chars) with reference inside → not replaced',
    val=('All documents must be carefully prepared and presented within ' * 5
         + ' AS PER FIELD 47A.'),
    map={'47A': 'X'},
    expect=None,  # use special check
))

# Field that just says "PLS REFER 47A" with no other content → replace
SC.append(dict(
    name='"PLS REFER FIELD 47A" + nothing else → replace',
    val='PLS REFER FIELD 47A',
    map={'47A': 'See conditions in F47A'},
    expect='See conditions in F47A',
))

# F48 with a number prefix — common form
SC.append(dict(
    name='F48 = "30 PLS REFER FIELD 47A" → replace (mostly bare ref)',
    val='30 PLS REFER FIELD 47A',
    map={'47A': '30 days from shipment but within expiry'},
    expect='30 days from shipment but within expiry',
))

# No matching reference field
SC.append(dict(
    name='Reference matched but referenced field is empty → no change',
    val='SEE FIELD 99',
    map={'99': ''},
    expect='SEE FIELD 99',
))


def main():
    p = f = 0
    for i, sc in enumerate(SC, 1):
        out, note = simulate(sc['val'], sc['map'])
        if sc['expect'] is None:
            # Special: long paragraph case — check that out preserves the original
            ok = out == sc['val']
        else:
            ok = out == sc['expect']
        tag = 'OK ' if ok else 'FAIL'
        print(f"[{tag}] #{i:02d}  {sc['name']}")
        print(f"        note: {note}")
        if not ok:
            print(f"        got    : {out[:120]!r}")
            print(f"        expect : {sc['expect'][:120] if sc['expect'] else sc['val'][:120]!r}")
        if ok: p += 1
        else: f += 1
    print(f"\n{'='*78}\n{p}/{p+f} P198di field-ref-resolver scenarios OK\n{'='*78}")
    return f == 0


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
