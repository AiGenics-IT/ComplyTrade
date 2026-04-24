"""
P198cs dry-run — Strict freight-wording adjacency override.

Rule (user-requested):
  • "FREIGHT PREPAID" or "FREIGHT COLLECT" must be written TOGETHER
    on the BL — both tokens, adjacent (whitespace/newline OK).
  • Bare "PREPAID" alone MUST NOT PASS.
  • Bare "COLLECT" alone MUST NOT PASS.
  • Only applies to MANDATORY conditions ("must show" / "shall
    show" / "must indicate" / etc.). Permissive ("freight collect
    is acceptable") and prohibitive ("freight collect not
    permitted") conditions handled by other layers.
"""
import re


_FREIGHT_ADJ_RE = {
    'FREIGHT PREPAID': re.compile(
        r'\b(?:FREIGHT|FRT\.?)\s+PREPAID\b'
        r'|\bPREPAID\s+(?:FREIGHT|FRT\.?)\b'
        r'|\b(?:FREIGHT|FRT\.?)\s+PAID\b'
    ),
    'FREIGHT COLLECT': re.compile(
        r'\b(?:FREIGHT|FRT\.?)\s+COLLECT\b'
        r'|\bCOLLECT\s+(?:FREIGHT|FRT\.?)\b'
        r'|\b(?:FREIGHT|FRT\.?)\s+TO\s+COLLECT\b'
    ),
}
_REQ_FREIGHT_KEY_RE = [
    ('FREIGHT PREPAID',
     re.compile(r'\b(?:FREIGHT|FRT\.?)\s+PREPAID\b|\bPREPAID\s+(?:FREIGHT|FRT\.?)\b|\b(?:FREIGHT|FRT\.?)\s+PAID\b')),
    ('FREIGHT COLLECT',
     re.compile(r'\b(?:FREIGHT|FRT\.?)\s+COLLECT\b|\bCOLLECT\s+(?:FREIGHT|FRT\.?)\b')),
]
_MANDATORY_VERB_RE = re.compile(
    r'\b(?:MUST|SHALL|HAS\s+TO|HAVE\s+TO|TO\s+BE|MUST\s+BE)\b'
    r'[^.]{0,120}?\b(?:SHOW|SHOWING|INDICATE|STATE|READ|MARKED|CARRY|BEAR|EVIDENCE|PRESENT)\b',
    flags=re.IGNORECASE,
)
_PERMISSIVE_VERB_RE = re.compile(
    r'\b(?:is|are|to\s+be)\s+(?:acceptable|permitted|allowed|permissible|allowable)\b',
    flags=re.IGNORECASE,
)
_PROHIBITIVE_RE = re.compile(
    r'\b(?:NOT\s+ACCEPTABLE|NOT\s+PERMITTED|NOT\s+ALLOWED|'
    r'MUST\s+NOT|SHALL\s+NOT|UNACCEPTABLE|PROHIBIT|'
    r'NOT\s+BE\s+ACCEPT)\b',
    flags=re.IGNORECASE,
)


def simulate(cond, bl_text):
    cu = cond.upper()
    if 'FREIGHT' not in cu:
        return 'skip', 'no FREIGHT keyword'
    if _PERMISSIVE_VERB_RE.search(cond):
        return 'skip', 'permissive — no strict override'
    if _PROHIBITIVE_RE.search(cu):
        return 'skip', 'prohibitive — no strict override'
    if re.search(r'\bFREIGHT\s+FORWARDER', cu) or 'FIATA' in cu or 'NVOCC' in cu:
        return 'skip', 'forwarder / NVOCC condition, not freight payability'
    req = None
    for k, p in _REQ_FREIGHT_KEY_RE:
        if p.search(cu):
            req = k; break
    if not req:
        return 'skip', 'no specific freight key in condition'
    if not _MANDATORY_VERB_RE.search(cond):
        return 'skip', 'not mandatory verb'
    adj = _FREIGHT_ADJ_RE[req]
    hit = adj.search(bl_text.upper())
    return ('PASS' if hit else 'FAIL',
            f'required={req}, adjacent-hit={bool(hit)}')


SC = []

# --- Passing adjacency ---
SC.append(dict(name='FREIGHT PREPAID adjacent → PASS',
    cond='Bill of Lading must show freight prepaid.',
    bl='BILL OF LADING\n...FREIGHT PREPAID...\n', expect='PASS'))
SC.append(dict(name='FREIGHT COLLECT adjacent → PASS',
    cond='Bill of Lading must show freight collect.',
    bl='BILL OF LADING\n...FREIGHT COLLECT...\n', expect='PASS'))
SC.append(dict(name='OCR newline between FREIGHT and COLLECT → PASS (same as user asked)',
    cond='Bill of Lading must show freight collect.',
    bl='BILL OF LADING\nFREIGHT\nCOLLECT\n', expect='PASS'))
SC.append(dict(name='PREPAID FREIGHT (reverse) → PASS',
    cond='Bill of Lading must show freight prepaid.',
    bl='BILL OF LADING\nPREPAID FREIGHT\n', expect='PASS'))
SC.append(dict(name='COLLECT FREIGHT (reverse) → PASS',
    cond='Bill of Lading must show freight collect.',
    bl='BILL OF LADING\nCOLLECT FREIGHT\n', expect='PASS'))
SC.append(dict(name='FRT COLLECT (abbreviated) → PASS',
    cond='Bill of Lading must show freight collect.',
    bl='BILL OF LADING\nFRT COLLECT\n', expect='PASS'))
SC.append(dict(name='FREIGHT PAID (as prepaid alias) → PASS for prepaid',
    cond='Bill of Lading must show freight prepaid.',
    bl='BILL OF LADING\nFREIGHT PAID\n', expect='PASS'))

# --- User-reported failures: bare tokens must NOT pass ---
SC.append(dict(name='Bare "PREPAID" alone (no FREIGHT) → FAIL',
    cond='Bill of Lading must show freight prepaid.',
    bl='BILL OF LADING\nPREPAID\nOTHER STUFF\n', expect='FAIL'))
SC.append(dict(name='Bare "COLLECT" alone (no FREIGHT) → FAIL',
    cond='Bill of Lading must show freight collect.',
    bl='BILL OF LADING\nCOLLECT\nOTHER STUFF\n', expect='FAIL'))
SC.append(dict(name='BL shows "COLLECT: USD 1500" without FREIGHT nearby → FAIL',
    cond='Bill of Lading must show freight collect.',
    bl='BILL OF LADING\nAMOUNT COLLECT: USD 1500\nSHIPPED ON BOARD\n',
    expect='FAIL'))
SC.append(dict(name='BL shows PREPAID in a stamp (no FREIGHT adjacency) → FAIL',
    cond='Bill of Lading must show freight prepaid.',
    bl='BILL OF LADING\n[STAMP: PREPAID]\nCargo description\n',
    expect='FAIL'))

# --- Opposite wording: BL shows wrong freight terms ---
SC.append(dict(name='BL shows FREIGHT PREPAID but LC requires COLLECT → FAIL',
    cond='Bill of Lading must show freight collect.',
    bl='BILL OF LADING\nFREIGHT PREPAID\n', expect='FAIL'))
SC.append(dict(name='BL shows FREIGHT COLLECT but LC requires PREPAID → FAIL',
    cond='Bill of Lading must show freight prepaid.',
    bl='BILL OF LADING\nFREIGHT COLLECT\n', expect='FAIL'))

# --- Conditions that should be SKIPPED (not strictly overridden) ---
SC.append(dict(name='Permissive condition "freight collect is acceptable" → skip',
    cond='Freight collect is acceptable.',
    bl='BILL OF LADING\n(no freight wording)\n', expect='skip'))
SC.append(dict(name='Prohibitive condition "freight collect not permitted" → skip',
    cond='Freight collect is not permitted.',
    bl='BILL OF LADING\nFREIGHT COLLECT\n', expect='skip'))
SC.append(dict(name='Forwarder-type condition → skip',
    cond='Freight forwarder\'s BL not acceptable.',
    bl='BILL OF LADING\nABC Freight Forwarders\n', expect='skip'))
SC.append(dict(name='Non-freight condition → skip',
    cond='Bill of Lading must show consignee to order of bank.',
    bl='BILL OF LADING\n...', expect='skip'))
SC.append(dict(name='Non-mandatory wording "freight collect shown" → skip',
    cond='Freight collect shown on BL.',
    bl='BILL OF LADING\nFREIGHT COLLECT\n', expect='skip'))

# --- OCR-tolerant adjacency ---
SC.append(dict(name='Multiple spaces between FREIGHT and COLLECT → PASS',
    cond='Bill of Lading must show freight collect.',
    bl='BILL OF LADING\nFREIGHT   COLLECT\n', expect='PASS'))
SC.append(dict(name='FREIGHT + newline + tab + COLLECT → PASS',
    cond='Bill of Lading must show freight collect.',
    bl='BILL OF LADING\nFREIGHT\n\t COLLECT\n', expect='PASS'))

# --- Negative: FREIGHT with unrelated word before COLLECT far away → FAIL ---
SC.append(dict(name='FREIGHT followed by far-away COLLECT via lots of text → FAIL',
    cond='Bill of Lading must show freight collect.',
    bl='BILL OF LADING\nFREIGHT DETAILS:\n' + ('cargo description line\n'*5) + 'COLLECT OR REMIT\n',
    expect='FAIL'))


def main():
    passed = 0; failed = 0
    for i, sc in enumerate(SC, 1):
        verdict, note = simulate(sc['cond'], sc['bl'])
        ok = (verdict == sc['expect'])
        tag = 'OK ' if ok else 'FAIL'
        print(f"[{tag}] #{i:02d}  {sc['name']}")
        print(f"         expect={sc['expect']}, got={verdict}")
        print(f"         note: {note}")
        if ok: passed += 1
        else: failed += 1
    print(f"\n{'='*78}\n{passed}/{passed+failed} P198cs freight-strict scenarios OK\n{'='*78}")
    return failed == 0


if __name__ == '__main__':
    import sys
    sys.exit(0 if main() else 1)
