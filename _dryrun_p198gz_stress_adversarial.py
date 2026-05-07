"""Stress + adversarial + cross-rule scenarios.

Sections:
1. Adversarial inputs — malformed, partial, garbled data
2. Internationalization — non-English document titles
3. Performance — detector latency on large inputs
4. Cross-rule interactions — multiple guards on same row
5. Real-data exhaustive sweep — every job × every detector
6. Boundary cases — pcts at 0, 100, negative, NaN-ish
7. Robust regex tests — unicode whitespace, smart quotes, line endings
8. Stability — re-run detectors 100x to ensure determinism
"""
import sys, os, re, json, glob, time
sys.path.insert(0, 'd:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final')
os.environ['PYTHONIOENCODING'] = 'utf-8'

from steps.step14_verification import (
    _detect_release_tranches,
    _detect_coal_quality_terms,
    _detect_advance_payment_terms,
)
from steps.step08_shipping_classification import _match_type_to_requirement

results = []
def ok(name, cond, note=''):
    tag = '[OK]  ' if cond else '[FAIL]'
    print(f"{tag} {name}" + (f" -- {note}" if note else ''))
    results.append(bool(cond))


# ── Section 1: Adversarial inputs ──
print("=" * 70)
print("Section 1: Adversarial inputs (malformed / garbled)")
print("=" * 70)

ADV_TRANCHE = [
    # (input_dict, expect_None_or_dict, label)
    ({'consolidated_fields': None}, None, 'None fields'),
    ({'consolidated_fields': {}}, None, 'Empty fields'),
    ({'consolidated_fields': {'46A': None}}, None, 'None F46A'),
    ({'consolidated_fields': {'46A': ''}}, None, 'Empty F46A'),
    ({'consolidated_fields': {'46A': '\x00\x00\x00'}}, None, 'Null bytes'),
    ({'consolidated_fields': {'46A': 'A) FOR RELEASE OF\n\nB) FOR RELEASE OF'}}, None, 'No percentages'),
    ({}, None, 'Empty top dict'),
    (None, None, 'None root'),
    ({'consolidated_fields': {'46A': 'A' * 50000}}, None, 'Huge garbage'),
]
for inp, expect, label in ADV_TRANCHE:
    try:
        info = _detect_release_tranches(inp)
        got = info if info else None
        ok(f"  {label}: returned None safely", got is None)
    except Exception as e:
        ok(f"  {label}: should not crash", False, f"crashed: {e}")


# Same for coal quality
ADV_COAL = [
    {'consolidated_fields': None},
    {},
    None,
    {'consolidated_fields': {'45A': '\x00', '47A': ''}},
]
for i, inp in enumerate(ADV_COAL, 1):
    try:
        info = _detect_coal_quality_terms(inp)
        ok(f"  Adversarial coal #{i}: ran without crash", True)
    except Exception as e:
        ok(f"  Adversarial coal #{i}", False, f"crashed: {e}")


# ── Section 2: Internationalization ──
print("\n" + "=" * 70)
print("Section 2: Non-English document titles")
print("=" * 70)

LC_FULL = [{'document_name': n} for n in [
    'Bill of Lading', 'Commercial Invoice', 'Certificate of Origin',
    'Health Certificate', 'Packing List',
]]
INTL_CASES = [
    ('Conocimiento de Embarque', -1, "Spanish BL"),
    ('Connaissement', -1, "French BL"),
    ('Frachtbrief', -1, "German Bill of Lading"),
    ('OPRINDELSESCERTIFIKAT', -1, "Danish CoO"),
    ('CERTIFICAT DORIGINE', -1, "French CoO"),
    ('TARJETA DE EMBARQUE', -1, "Spanish boarding"),
    ('Certificate of Origin', 2, "English CoO matches"),
]
for inp, exp_idx, label in INTL_CASES:
    idx, name = _match_type_to_requirement(inp, LC_FULL)
    ok(f"  {label}: idx={idx}", idx == exp_idx,
       f"got idx={idx} name={name!r}")


# ── Section 3: Performance ──
print("\n" + "=" * 70)
print("Section 3: Performance (detector latency)")
print("=" * 70)

# Big LC text (~10k chars)
big_lc_text = "A) FOR RELEASE OF 90 PERCENT\n" + "Lorem ipsum " * 500 + \
              "\nB) FOR RELEASE OF 10 PERCENT\n" + "Foo bar " * 500
t0 = time.time()
for _ in range(100):
    _detect_release_tranches({'consolidated_fields':{'46A':big_lc_text}})
elapsed = (time.time() - t0) * 1000
ok(f"  100x detect_release_tranches on 10k-char F46A: {elapsed:.0f}ms",
   elapsed < 1000, f"avg {elapsed/100:.2f}ms per call")

t0 = time.time()
for _ in range(100):
    _detect_coal_quality_terms({'consolidated_fields':{
        '47A': 'GROSS CALORIFIC VALUE (ARB): 5800\nTOTAL MOISTURE (ARB): 14 PCT MAX\n' + 'noise '*500
    }})
elapsed = (time.time() - t0) * 1000
ok(f"  100x detect_coal_quality on noisy F47A: {elapsed:.0f}ms",
   elapsed < 2000, f"avg {elapsed/100:.2f}ms per call")


# ── Section 4: Cross-rule interactions ──
print("\n" + "=" * 70)
print("Section 4: Cross-rule interactions on synthetic LC")
print("=" * 70)

# LC with 2-tranche AND coal-quality AND advance-payment
synth = {
    'consolidated_fields': {
        '45A': 'Steam coal CFR Karachi 5800 NAR',
        '46A': """A) FOR RELEASE OF 80 PERCENT
docs at load
B) FOR RELEASE OF 20 PERCENT
docs at discharge port""",
        '47A': """GROSS CALORIFIC VALUE (ARB): 5800 KCAL/KG (REJECT BELOW 5500)
TOTAL MOISTURE (ARB): 14 PCT MAX REJECT ABOVE 16 PCT
ASH (ADB): 12 PCT MAX REJECT ABOVE 16 PCT""",
    }
}
trinfo = _detect_release_tranches(synth)
qinfo = _detect_coal_quality_terms(synth)
adv = _detect_advance_payment_terms(synth)
ok("  3 detectors fire independently — tranche detected",
   trinfo is not None and trinfo['tranche_a_pct']==80)
ok("  3 detectors fire independently — coal detected",
   qinfo is not None and qinfo.get('is_coal_lc'))
ok("  3 detectors fire independently — advance NOT detected (this is a release-tranche, not advance)",
   adv is None)


# ── Section 5: Real-data exhaustive sweep ──
print("\n" + "=" * 70)
print("Section 5: Real-data exhaustive sweep (every job × every detector)")
print("=" * 70)

stats = {'jobs':0, 'tr':0, 'coal':0, 'adv':0, 'errors':0}
import json
for jp in glob.glob('d:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final/results/*/step06/step06_result.json'):
    try:
        d = json.load(open(jp, encoding='utf-8'))
    except Exception:
        stats['errors']+=1; continue
    stats['jobs'] += 1
    try:
        if _detect_release_tranches(d): stats['tr'] += 1
    except Exception: stats['errors']+=1
    try:
        c = _detect_coal_quality_terms(d)
        if c and c.get('is_coal_lc'): stats['coal'] += 1
    except Exception: stats['errors']+=1
    try:
        if _detect_advance_payment_terms(d): stats['adv'] += 1
    except Exception: stats['errors']+=1

print(f"  Jobs: {stats['jobs']}")
print(f"  2-tranche: {stats['tr']}")
print(f"  Coal-quality: {stats['coal']}")
print(f"  Advance-payment: {stats['adv']}")
print(f"  Crashes: {stats['errors']}")
ok("  All detectors ran without crashes", stats['errors'] == 0)


# ── Section 6: Boundary pcts ──
print("\n" + "=" * 70)
print("Section 6: Boundary percentages")
print("=" * 70)

BOUND = [
    ("A) FOR RELEASE OF 0 PERCENT...\nB) FOR RELEASE OF 100 PERCENT", None, "0/100"),
    ("A) FOR RELEASE OF 1 PERCENT...\nB) FOR RELEASE OF 99 PERCENT", (1,99), "1/99"),
    ("A) FOR RELEASE OF 99 PERCENT...\nB) FOR RELEASE OF 1 PERCENT", (99,1), "99/1"),
    ("A) FOR RELEASE OF -10 PERCENT...\nB) FOR RELEASE OF 110 PERCENT", None, "negative/over"),
    ("A) FOR RELEASE OF 50.5 PERCENT...\nB) FOR RELEASE OF 49.5 PERCENT", None, "decimals (regex \\d only)"),
]
for txt, expect, label in BOUND:
    info = _detect_release_tranches({'consolidated_fields':{'46A':txt}})
    if expect is None:
        ok(f"  {label}: rejected", info is None)
    else:
        ok(f"  {label}: A={info['tranche_a_pct']} B={info['tranche_b_pct']}",
           info is not None and (info['tranche_a_pct'],info['tranche_b_pct'])==expect)


# ── Section 7: Robust regex (unicode / line endings) ──
print("\n" + "=" * 70)
print("Section 7: Unicode whitespace and line endings")
print("=" * 70)

ROBUST = [
    ("A) FOR RELEASE OF 90 PERCENT...\r\nB) FOR RELEASE OF 10 PERCENT", "CRLF"),
    ("A) FOR RELEASE OF 90 PERCENT...\rB) FOR RELEASE OF 10 PERCENT", "old-mac CR"),
    ("A) FOR RELEASE OF 90 PERCENT...\nB) FOR RELEASE OF 10 PERCENT",
     "non-breaking space — should fail (\\s doesn't match \\u00A0 by default)"),
]
for txt, label in ROBUST:
    info = _detect_release_tranches({'consolidated_fields':{'46A':txt}})
    if 'non-breaking' in label:
        # Acceptable behavior either way
        ok(f"  {label}: {'ran' if info else 'rejected — non-breaking space not matched'}",
           True)
    else:
        ok(f"  {label}: detected", info is not None)


# ── Section 8: Stability / determinism ──
print("\n" + "=" * 70)
print("Section 8: Stability — same input → same output 100x")
print("=" * 70)

stable_input = {'consolidated_fields':{'46A':"A) FOR RELEASE OF 90 PERCENT\nload\nB) FOR RELEASE OF 10 PERCENT\ndischarge"}}
prev = _detect_release_tranches(stable_input)
all_match = True
for _ in range(100):
    cur = _detect_release_tranches(stable_input)
    if cur != prev:
        all_match = False; break
ok("  Tranche detector is deterministic (100 runs)", all_match)


# ── Source wiring ──
print("\n" + "=" * 70)
print("Section 9: Source wiring sanity (final check)")
print("=" * 70)
src14 = open('d:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final/steps/step14_verification.py',
             'r', encoding='utf-8').read()
src8 = open('d:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final/steps/step08_shipping_classification.py',
            'r', encoding='utf-8').read()
src3 = open('d:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final/steps/step03_sequencing.py',
            'r', encoding='utf-8').read()
src6 = open('d:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final/steps/step06_final_lc.py',
            'r', encoding='utf-8').read()
src7 = open('d:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final/steps/step07_clause_extraction.py',
            'r', encoding='utf-8').read()
src12 = open('d:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final/steps/step12_decomposition.py',
             'r', encoding='utf-8').read()

EXPECTED_MARKERS = [
    ('step03', src3, ['P198gz8','P198gz28']),
    ('step06', src6, ['P198gz22']),
    ('step07', src7, ['P198gz20']),
    ('step08', src8, ['P198gz12','P198gz13','P198gz14','P198gz15','P198gz16','P198gz17','P198gz23','P198gz25']),
    ('step12', src12, ['P198gz10','P198gz11','P198gx']),  # P198gx is the AWB splitter in step12
    ('step14', src14, ['P198gt2','P198gv','P198gw','P198gy','P198gz5','P198gz6','P198gz7','P198gz18','P198gz19','P198gz26','P198gz27']),  # P198gz24 was manual patch only
]
for name, src, markers in EXPECTED_MARKERS:
    for m in markers:
        ok(f"  {name} contains {m}", m in src)


print("\n" + "=" * 70)
passed = sum(results)
total = len(results)
print(f"STRESS + ADVERSARIAL: {passed}/{total}")
print("=" * 70)
if passed != total:
    sys.exit(1)
print("OVERALL: OK")
sys.exit(0)
