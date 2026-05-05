"""P198gz5b — notify-party detection across BOTH formal block AND
free-text mentions. Use real AWB data + synthetic edge cases."""
import sys, os, re, json
sys.path.insert(0, 'd:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final')
os.environ['PYTHONIOENCODING'] = 'utf-8'

results = []
def ok(name, cond, note=''):
    tag = '[OK]  ' if cond else '[FAIL]'
    print(f"{tag} {name}" + (f" -- {note}" if note else ''))
    results.append(bool(cond))


def gz5b(awb_text_up, cond_u):
    """Mirror P198gz5b from production."""
    has_notify_anywhere = bool(re.search(r'\bNOTIFY\b', awb_text_up))
    lc_pats = []
    for pat in (r'\bBANK\s+AL\s+HABIB\b', r'\bAL\s+HABIB\b',
                r'\bHABIB\s+BANK\b', r'\bAPPLICANT\b'):
        if re.search(pat, cond_u): lc_pats.append(pat)
    if not lc_pats:
        return ('SKIP', 'no notify name in cond')
    missing = []; missing_in_ctx = []
    for pat in lc_pats:
        if not re.search(pat, awb_text_up):
            missing.append(pat); continue
        notify_ctx_hit = False
        for nm in re.finditer(r'\bNOTIFY\b', awb_text_up):
            ctx_end = min(len(awb_text_up), nm.end() + 400)
            if re.search(pat, awb_text_up[nm.start():ctx_end]):
                notify_ctx_hit = True; break
        if not notify_ctx_hit:
            missing_in_ctx.append(pat)
    if missing and not has_notify_anywhere:
        return ('FAIL', f'no notify keyword, missing: {missing}')
    if missing_in_ctx and not missing:
        return ('FAIL', f'parties present but not in notify ctx: {missing_in_ctx}')
    if missing:
        return ('FAIL', f'missing: {missing}')
    return ('PASS', 'all parties found in notify context')


# ── Section 1 — synthetic scenarios ──
print("=" * 70)
print("Section 1: Synthetic scenarios")
print("=" * 70)
COND = "AWB must be marked notify the Applicant and Bank Al Habib Limited, Pakistan."
COND_U = COND.upper()

CASES = [
    # Formal Notify Party block with both parties
    ("CONSIGNEE: ALI ENT\nNOTIFY PARTY: APPLICANT AND BANK AL HABIB LIMITED, LAHORE",
     'PASS', 'Formal Notify block with both parties'),
    # Free-text "Notify:" mention — also valid
    ("HANDLING INFO: NOTIFY APPLICANT AND BANK AL HABIB LIMITED, LAHORE",
     'PASS', 'Free-text Notify mention with both parties'),
    # OSI/SCI line with notify
    ("OSI: NOTIFY APPLICANT AND BANK AL HABIB ON ARRIVAL",
     'PASS', 'OSI line with notify keyword'),
    # No notify field but parties appear in description
    ("DESCRIPTION: APPLICANT AND BANK AL HABIB ARE THE PARTIES",
     'FAIL', 'Parties present but no NOTIFY keyword'),
    # No notify field, no parties
    ("CONSIGNEE: ALI ENT\nSHIPPER: RAJAPAKSHE ENT",
     'FAIL', 'No notify keyword + parties absent'),
    # Notify field but only ONE of the two parties
    ("NOTIFY PARTY: APPLICANT ONLY",
     'FAIL', 'Notify block missing Bank Al Habib'),
    # Notify with applicant nearby + bank al habib elsewhere on doc
    ("NOTIFY APPLICANT\n" + ("FILLER " * 80) + "\nBANK AL HABIB LIMITED ON ARRIVAL",
     'FAIL', 'Bank Al Habib >400 chars from NOTIFY → not in context'),
    # Both within notify context (one block)
    ("NOTIFY PARTY: APPLICANT AND BANK AL HABIB LIMITED",
     'PASS', 'Both parties within single Notify context'),
    # Multiple NOTIFY mentions (e.g., "Notify1: APPLICANT" + "Notify2: AL HABIB")
    ("NOTIFY 1: APPLICANT\nNOTIFY 2: BANK AL HABIB LIMITED",
     'PASS', 'Two separate Notify lines, each with one party'),
]
for txt, expect, label in CASES:
    v, why = gz5b(txt.upper(), COND_U)
    ok(f"  {label}: {v}", v == expect, f'expected {expect}; reason={why}')


# ── Section 2 — Real AWB from job 94edb6a7 (no notify on doc) ──
print("\n" + "=" * 70)
print("Section 2: Real AWB job 94edb6a7 (SriLankan, no notify)")
print("=" * 70)
d8 = json.load(open('results/94edb6a7-6179-4f2a-b1f7-f2cd3cee64bf/step08/step08_result.json', encoding='utf-8'))
awb = next(p for p in d8['classified_packets'] if 'airway' in (p.get('document_type','') or '').lower())
awb_text = (awb.get('cleaned_text') or awb.get('raw_text') or '').upper()
v, why = gz5b(awb_text, COND_U)
ok(f"  Real AWB no-notify case: {v}", v == 'FAIL', f'reason={why}')


# ── Section 3 — Real AWB from job 104ac15f (has notify=Genetics) ──
print("\n" + "=" * 70)
print("Section 3: Real AWB job 104ac15f (Sinotech, notify=Genetics)")
print("=" * 70)
d8 = json.load(open('results/104ac15f-56ca-4499-badf-aaf3b92f401c/step08/step08_result.json', encoding='utf-8'))
awb = next(p for p in d8['classified_packets'] if 'airway' in (p.get('document_type','') or '').lower())
awb_text = (awb.get('cleaned_text') or awb.get('raw_text') or '').upper()
v, why = gz5b(awb_text, COND_U)
# This AWB notifies Genetics, NOT Bank Al Habib / Applicant — should FAIL
ok(f"  Real AWB Sinotech (notify=Genetics, not Al Habib): {v}",
   v == 'FAIL', f'reason={why}')


# ── Section 4 — Source wiring ──
print("\n" + "=" * 70)
print("Section 4: Source wiring")
print("=" * 70)
src = open('d:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final/steps/step14_verification.py',
           'r', encoding='utf-8').read()
ok("  P198gz5b marker", 'P198gz5b' in src)
ok("  has_notify_anywhere check", '_has_notify_anywhere' in src)
ok("  notify_ctx 400-char window", "_ctx_end + 400" not in src and "+ 400" in src)
ok("  missing_in_notify_ctx tracking", '_missing_in_notify_ctx' in src)


print("\n" + "=" * 70)
passed = sum(results)
total = len(results)
print(f"P198gz5b NOTIFY: {passed}/{total}")
print("=" * 70)
if passed != total:
    sys.exit(1)
print("OVERALL: OK")
sys.exit(0)
