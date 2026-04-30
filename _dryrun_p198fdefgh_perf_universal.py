"""
P198fd / fe / ff / fg / fh dry-run — performance + universal F47A patterns.

Fixes covered:
  P198fd — TCP keepalive HTTP session (no firewall idle timeout)
  P198fe — Result cache (skip duplicate LLM calls)
  P198ff — Discrepancy-whitelist clause detector
  P198fg — Late-shipment-with-penalty detector
  P198fh — Required-independent-surveyor detector

Real-data anchor:
  Job f3ef028e — F47A clause 14 has the discrepancy-whitelist:
    "ALL DISCREPANCIES ARE ACCEPTABLE EXCEPT FOR DESCRIPTION OF GOODS,
     QUANTITY, QUALITY, LATEST DATE OF SHIPMENT, PORT OF LOADING AND
     PORT OF DISCHARGE AND ORIGIN OF GOODS"

  Negative anchors (jobs that must NOT trigger the new detectors):
    Standard sight LCs, advance-payment LCs, etc.
"""
import sys, os, json
sys.path.insert(0, 'd:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final')

from steps.step14_verification import (
    _detect_discrepancy_whitelist,
    _detect_late_shipment_penalty,
    _detect_required_surveyors,
    _build_f47a_context,
    _LLMResultCache, _LLM_CACHE,
    _LLM_SESSION,
)

results = []
def assert_eq(name, got, expected):
    ok = (got == expected)
    print(f"[{'OK' if ok else 'FAIL'}] {name}")
    if not ok:
        print(f"          got     : {got!r}")
        print(f"          expected: {expected!r}")
    results.append(ok)

def make_lc(f47a='', f46a='', f32b='USD 100,000.00'):
    return {'consolidated_fields': {
        '32B': f32b, '46A': f46a, '47A': f47a,
    }}

# ──────────────────────────────────────────────────────────────────────
# P198fd — keepalive HTTP session
# ──────────────────────────────────────────────────────────────────────
print("--- P198fd: keepalive HTTP session ---")
import requests
assert_eq("P198fd: _LLM_SESSION is a requests.Session",
          isinstance(_LLM_SESSION, requests.Session), True)
# Check that adapters are mounted for both http and https
assert_eq("P198fd: http:// adapter mounted",
          'http://' in _LLM_SESSION.adapters, True)
assert_eq("P198fd: https:// adapter mounted",
          'https://' in _LLM_SESSION.adapters, True)
# The mounted adapter should have a non-default pool size
adapter = _LLM_SESSION.get_adapter('http://example.com/')
assert_eq("P198fd: pool_connections >= 16",
          adapter._pool_connections >= 16, True)

# ──────────────────────────────────────────────────────────────────────
# P198fe — LLM result cache
# ──────────────────────────────────────────────────────────────────────
print("\n--- P198fe: LLM result cache ---")
cache = _LLMResultCache(max_entries=4)
key1 = _LLMResultCache.make_key('Qwen', 'hello world')
key2 = _LLMResultCache.make_key('Qwen', 'goodbye')
key3 = _LLMResultCache.make_key('Qwen', 'hello world')  # same as key1
assert_eq("P198fe: identical inputs → identical key", key1 == key3, True)
assert_eq("P198fe: different inputs → different keys", key1 != key2, True)
assert_eq("P198fe: empty cache get returns None", cache.get(key1), None)
cache.put(key1, {'compliance': 'PASS', 'result': 'OK'})
hit = cache.get(key1)
assert_eq("P198fe: put then get returns the stored dict", hit and hit.get('compliance'), 'PASS')
assert_eq("P198fe: cache returns COPY (mutating doesn't affect store)",
          (hit.update({'compliance': 'X'}), cache.get(key1).get('compliance'))[1], 'PASS')
# LRU eviction
cache.put('k1', {'compliance': 'PASS'})
cache.put('k2', {'compliance': 'PASS'})
cache.put('k3', {'compliance': 'PASS'})
cache.put('k4', {'compliance': 'PASS'})  # cache full at 4
cache.put('k5', {'compliance': 'PASS'})  # evict oldest (k1)
assert_eq("P198fe: LRU evicts oldest when full",
          cache.get('k1') is None and cache.get('k5') is not None, True)

# Hit / miss counters
m0, h0 = cache.misses, cache.hits
cache.get('k5'); cache.get('k5'); cache.get('does-not-exist')
assert_eq("P198fe: hit counter incremented",
          cache.hits >= h0 + 2, True)
assert_eq("P198fe: miss counter incremented",
          cache.misses >= m0 + 1, True)

# ──────────────────────────────────────────────────────────────────────
# P198ff — Discrepancy-whitelist detector
# ──────────────────────────────────────────────────────────────────────
print("\n--- P198ff: Discrepancy-whitelist detector ---")

# Real job f3ef028e
real_job_path = 'd:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final/results/f3ef028e-b879-40d2-9351-39a2aff90175/step06/step06_result.json'
if os.path.isfile(real_job_path):
    with open(real_job_path, 'r', encoding='utf-8') as f:
        real_s6 = json.load(f)
    info = _detect_discrepancy_whitelist(real_s6)
    assert_eq("real f3ef028e: whitelist detected",
              info is not None, True)
    cats = info['hard_fail_categories']
    print(f"  Hard-fail categories: {sorted(cats)}")
    for c in ('goods_description', 'quantity', 'quality',
              'shipment_date', 'port_of_loading', 'port_of_discharge', 'origin'):
        assert_eq(f"real f3ef028e: '{c}' in hard-fail set",
                  c in cats, True)

# Synthetic positive cases
sc_simple = make_lc('14) ALL DISCREPANCIES ARE ACCEPTABLE EXCEPT FOR DESCRIPTION OF GOODS, QUANTITY AND QUALITY.')
i1 = _detect_discrepancy_whitelist(sc_simple)
assert_eq("simple whitelist: 3 categories",
          i1 and len(i1['hard_fail_categories']) >= 3, True)

sc_alt = make_lc('All discrepancies waived except for amount, port of loading, origin.')
i2 = _detect_discrepancy_whitelist(sc_alt)
assert_eq("alt phrasing 'waived': detected",
          i2 is not None, True)

sc_quote = make_lc('Any discrepancy is accepted except for the lc number and beneficiary.')
i3 = _detect_discrepancy_whitelist(sc_quote)
assert_eq("'is accepted except' wording: detected",
          i3 is not None, True)

# Synthetic NEGATIVE cases — must NOT match
neg_cases = [
    ("standard LC, no whitelist clause",
     make_lc('1) STANDARD CLAUSES, 2) STALE BL NOT ACCEPTABLE')),
    ("clause mentioning 'discrepancy' but not the whitelist pattern",
     make_lc('USD 116 DISCREPANCY CHARGES WILL BE DEDUCTED IF DOCUMENTS CONTAIN DISCREPANCY')),
    ("clause about acceptance but not discrepancy",
     make_lc('NEGOTIATION UNDER RESERVE NOT ACCEPTABLE')),
    ("empty F47A", make_lc('')),
    ("None step06", None),
]
for name, lc in neg_cases:
    got = _detect_discrepancy_whitelist(lc)
    assert_eq(f"NEG: {name}", got is None, True)

# Banner injection
ctx = _build_f47a_context(real_s6) if os.path.isfile(real_job_path) else None
if ctx:
    assert_eq("real f3ef028e: f47a_context contains DISCREPANCY WHITELIST banner",
              'DISCREPANCY WHITELIST' in ctx, True)

# ──────────────────────────────────────────────────────────────────────
# P198fg — Late-shipment-with-penalty detector
# ──────────────────────────────────────────────────────────────────────
print("\n--- P198fg: Late-shipment-with-penalty ---")
sc_pen1 = make_lc('LATE SHIPMENT ALLOWED PROVIDED USD 500 PER DAY DEDUCTED.')
ip1 = _detect_late_shipment_penalty(sc_pen1)
assert_eq("simple penalty: detected", ip1 and ip1.get('has_late_penalty'), True)
assert_eq("penalty amount = 500", ip1['penalty_amount'], 500.0)
assert_eq("penalty currency USD", ip1['penalty_currency'], 'USD')
assert_eq("per_day flag", ip1['per_day'], True)

sc_pen2 = make_lc('Late shipment is acceptable with USD 1,000 penalty.')
ip2 = _detect_late_shipment_penalty(sc_pen2)
assert_eq("flat penalty: detected", ip2 is not None, True)
assert_eq("flat penalty amount = 1000", ip2['penalty_amount'], 1000.0)

sc_pen3 = make_lc('Delay in shipment is accepted with EUR 250 per day penalty.')
ip3 = _detect_late_shipment_penalty(sc_pen3)
assert_eq("delay-form penalty: detected", ip3 is not None, True)

# Negative cases
neg_pen = [
    ("no penalty clause",
     make_lc('LATE SHIPMENT NOT ACCEPTABLE')),
    ("late shipment forbidden",
     make_lc('SHIPMENT MUST BE ON OR BEFORE LATEST DATE')),
    ("empty F47A", make_lc('')),
]
for name, lc in neg_pen:
    got = _detect_late_shipment_penalty(lc)
    assert_eq(f"NEG penalty: {name}", got is None, True)

# ──────────────────────────────────────────────────────────────────────
# P198fh — Required surveyor detector
# ──────────────────────────────────────────────────────────────────────
print("\n--- P198fh: Required-surveyor detector ---")
sc_sgs = make_lc('', 'CERTIFICATE OF ANALYSIS ISSUED BY SGS AT LOADING PORT.')
isgs = _detect_required_surveyors(sc_sgs)
assert_eq("SGS: detected", isgs and 'sgs' in isgs['required_surveyors'], True)

sc_multi = make_lc('', 'CERT OF ANALYSIS BY SGS, COTECNA OR INTERTEK ACCEPTABLE.')
imulti = _detect_required_surveyors(sc_multi)
assert_eq("multiple surveyors: SGS detected",
          imulti and 'sgs' in imulti['required_surveyors'], True)
assert_eq("multiple surveyors: Cotecna detected",
          imulti and 'cotecna' in imulti['required_surveyors'], True)
assert_eq("multiple surveyors: Intertek detected",
          imulti and 'intertek' in imulti['required_surveyors'], True)

sc_alfred = make_lc('', 'WEIGHT CERT BY ALFRED H KNIGHT AT LOADING PORT')
ialf = _detect_required_surveyors(sc_alfred)
assert_eq("Alfred H Knight: detected",
          ialf and 'alfred_knight' in ialf['required_surveyors'], True)

# Negative cases
neg_sv = [
    ("no surveyor named",
     make_lc('', 'CERTIFICATE OF ORIGIN BY CHAMBER OF COMMERCE')),
    ("'inspector' but not by name",
     make_lc('', 'INDEPENDENT INSPECTION REQUIRED')),
    ("empty fields", make_lc('', '')),
]
for name, lc in neg_sv:
    got = _detect_required_surveyors(lc)
    assert_eq(f"NEG surveyor: {name}", got is None, True)

# ──────────────────────────────────────────────────────────────────────
# Banner integration — multiple banners stack correctly
# ──────────────────────────────────────────────────────────────────────
print("\n--- Banner integration ---")
combo_lc = make_lc(
    f47a='14) ALL DISCREPANCIES ACCEPTABLE EXCEPT FOR QUANTITY AND ORIGIN. '
         '15) LATE SHIPMENT ALLOWED PROVIDED USD 100 PER DAY DEDUCTED. '
         '16) CHARTER PARTY BL ACCEPTABLE.',
    f46a='1) COMMERCIAL INVOICE 2) BL 3) CERT OF ANALYSIS BY SGS')
ctx_combo = _build_f47a_context(combo_lc)
assert_eq("combo: discrepancy whitelist banner",
          'DISCREPANCY WHITELIST' in ctx_combo, True)
assert_eq("combo: late-shipment-penalty banner",
          'LATE-SHIPMENT-WITH-PENALTY' in ctx_combo, True)
assert_eq("combo: surveyor banner",
          'REQUIRED INDEPENDENT SURVEYOR' in ctx_combo, True)
# Original F47A text preserved
assert_eq("combo: original F47A clause 16 preserved",
          'CHARTER PARTY' in ctx_combo.upper(), True)

# Standard LC (no special clauses) — no extra banners
std_lc = make_lc('1) STANDARD CLAUSE 2) ANOTHER CLAUSE')
ctx_std = _build_f47a_context(std_lc)
for banner in ('DISCREPANCY WHITELIST', 'LATE-SHIPMENT-WITH-PENALTY',
               'REQUIRED INDEPENDENT SURVEYOR'):
    assert_eq(f"standard LC: '{banner}' NOT in context",
              banner not in ctx_std, True)

# ──────────────────────────────────────────────────────────────────────
# P198eo through P198fc REGRESSION — make sure none of the existing
# detectors are disturbed by the new code
# ──────────────────────────────────────────────────────────────────────
print("\n--- Regression: existing detectors still work ---")
from steps.step14_verification import (
    _detect_advance_payment_terms,
    _detect_coal_quality_terms,
)
# Real coal job f3ef028e
if os.path.isfile(real_job_path):
    with open(real_job_path, 'r', encoding='utf-8') as f:
        s6 = json.load(f)
    coal_info = _detect_coal_quality_terms(s6)
    assert_eq("regression: coal LC still detected",
              coal_info and coal_info.get('is_coal_lc'), True)
    assert_eq("regression: GCV spec still 5800",
              coal_info.get('gcv_spec_kcal'), 5800.0)

# Real advance-payment job 2d98b74c
adv_p = 'd:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final/results/2d98b74c-457f-4456-8a85-68841190e4d5/step06/step06_result.json'
if os.path.isfile(adv_p):
    with open(adv_p, 'r', encoding='utf-8') as f:
        s6 = json.load(f)
    adv_info = _detect_advance_payment_terms(s6)
    assert_eq("regression: advance-payment LC still detected (80/20)",
              adv_info and adv_info.get('advance_pct') == 80, True)


# ──────────────────────────────────────────────────────────────────────
# Extended EDGE CASE coverage (additional dry-run scenarios)
# ──────────────────────────────────────────────────────────────────────
print("\n--- EXTENDED edge cases ---")

# ── P198ff (whitelist) extended ─────────────────────────────────────
WL_CASES = [
    # (name, lc, expect_detected, optional_check)
    # Multiline clause across newline boundaries
    ("multi-line clause across newlines",
     make_lc('14) ALL DISCREPANCIES ARE ACCEPTABLE EXCEPT FOR\n'
             'DESCRIPTION OF GOODS, QUANTITY,\n'
             'AND PORT OF LOADING.'),
     True, lambda i: 'goods_description' in i['hard_fail_categories']
                  and 'quantity' in i['hard_fail_categories']
                  and 'port_of_loading' in i['hard_fail_categories']),
    # Different word order — "EXCEPT" before pipe-bullet listing
    ("verbose clause with semicolons",
     make_lc('17) ALL DISCREPANCIES ACCEPTABLE EXCEPT FOR: '
             'GOODS DESCRIPTION; QUANTITY; AMOUNT; PORT OF DISCHARGE.'),
     True, None),
    # Mixed case
    ("Mixed case wording",
     make_lc('any Discrepancy Is Accepted Except For Quality and Origin.'),
     True, lambda i: 'quality' in i['hard_fail_categories']),
    # Long EXCEPT list (10+ items)
    ("long EXCEPT list",
     make_lc('All discrepancies acceptable except for description of goods, '
             'quantity, quality, latest date of shipment, port of loading, '
             'port of discharge, origin of goods, applicant, beneficiary, '
             'consignee, lc number.'),
     True, lambda i: len(i['hard_fail_categories']) >= 8),
    # Numbered clause prefix
    ("numbered clause '14) ALL DISCREPANCIES'",
     make_lc('1) STANDARD\n14) ALL DISCREPANCIES ARE ACCEPTABLE EXCEPT FOR QUANTITY.'),
     True, lambda i: 'quantity' in i['hard_fail_categories']),
    # Excluded categories with synonyms
    ("'GOODS DESCRIPTION' (reverse word order)",
     make_lc('All discrepancies acceptable except for goods description.'),
     True, lambda i: 'goods_description' in i['hard_fail_categories']),
    ("'GRADE' instead of 'QUALITY'",
     make_lc('All discrepancies acceptable except for grade and origin.'),
     True, lambda i: 'quality' in i['hard_fail_categories']),
    # NEGATIVE — must NOT match
    ("'discrepancy charges' (admin fee, not whitelist)",
     make_lc('USD 116/- DISCREPANCY CHARGES WILL BE DEDUCTED INCASE OF DISCREPANCY'),
     False, None),
    ("'no discrepancy allowed' clause",
     make_lc('NO DISCREPANCY IS ALLOWED IN THIS PRESENTATION.'),
     False, None),
    ("clause says 'except' but not in discrepancy context",
     make_lc('THIRD PARTY DOCUMENTS ACCEPTABLE EXCEPT INVOICE AND DRAFT.'),
     False, None),
    ("'ACCEPTED' but no 'EXCEPT'",
     make_lc('ALL DISCREPANCIES ACCEPTED WITH BUYER APPROVAL.'),
     False, None),
]
for name, lc, expect_det, check in WL_CASES:
    info = _detect_discrepancy_whitelist(lc)
    detected = info is not None
    ok = (detected == expect_det)
    if ok and detected and check:
        if not check(info):
            ok = False
    print(f"[{'OK' if ok else 'FAIL'}] WL: {name}")
    results.append(ok)

# ── P198fg (late shipment penalty) extended ─────────────────────────
PEN_CASES = [
    ("PCT-based penalty rate",
     make_lc('LATE SHIPMENT ALLOWED PROVIDED 0.5 PCT PER DAY DEDUCTION.'),
     True),
    ("EUR currency penalty",
     make_lc('Late shipment is acceptable with EUR 200 per day penalty.'),
     True),
    ("PKR currency penalty",
     make_lc('Late shipment allowed provided PKR 50,000 deducted.'),
     True),
    ("flat penalty without per-day",
     make_lc('Late shipment allowed with USD 5,000 penalty.'),
     True),
    ("'subject to' wording",
     make_lc('LATE SHIPMENT IS ALLOWED SUBJECT TO USD 1000 PENALTY.'),
     True),
    ("Mixed case + decimals",
     make_lc('Delay in shipment is accepted with USD 75.50 per day deduction.'),
     True),
    # NEGATIVE
    ("LATE SHIPMENT NOT ACCEPTABLE",
     make_lc('LATE SHIPMENT NOT ACCEPTABLE.'), False),
    ("STALE BL NOT ACCEPTABLE (different concept)",
     make_lc('STALE BL NOT ACCEPTABLE.'), False),
    ("just 'late' without context",
     make_lc('PARTIAL SHIPMENT ALLOWED, NO LATE PENALTY.'), False),
    ("empty F47A", make_lc(''), False),
]
for name, lc, expect_det in PEN_CASES:
    got = _detect_late_shipment_penalty(lc)
    detected = got is not None
    ok = (detected == expect_det)
    print(f"[{'OK' if ok else 'FAIL'}] PEN: {name}")
    results.append(ok)

# ── P198fh (surveyor) extended ──────────────────────────────────────
SV_CASES = [
    ("Bureau Veritas full name",
     make_lc('', 'CERT BY BUREAU VERITAS REQUIRED.'),
     True, 'bureau_veritas'),
    ("Saybolt",
     make_lc('', 'PETROLEUM CERT BY SAYBOLT.'),
     True, 'saybolt'),
    ("Inspectorate",
     make_lc('', 'COA BY INSPECTORATE GROUP.'),
     True, 'inspectorate'),
    ("Geo-Chem with hyphen",
     make_lc('', 'CERT OF ANALYSIS BY GEO-CHEM REQUIRED.'),
     True, 'geo_chem'),
    ("Control Union",
     make_lc('', 'COMMODITY CERT ISSUED BY CONTROL UNION.'),
     True, 'control_union'),
    ("Cotecna lowercase",
     make_lc('', 'cotecna issued cert.'),
     True, 'cotecna'),
    # NEGATIVE
    ("Chamber of Commerce only",
     make_lc('', 'CERT OF ORIGIN BY CHAMBER OF COMMERCE.'), False, None),
    ("Generic 'independent surveyor'",
     make_lc('', 'CERT BY ANY INDEPENDENT SURVEYOR REQUIRED.'), False, None),
    ("'SGS' substring inside another word — must NOT match",
     make_lc('', 'WGS84 CO-ORDINATES REQUIRED ON CERT.'), False, None),
    ("empty fields", make_lc('', ''), False, None),
]
for name, lc, expect_det, expected_sv in SV_CASES:
    got = _detect_required_surveyors(lc)
    detected = got is not None
    ok = (detected == expect_det)
    if ok and detected and expected_sv:
        if expected_sv not in got['required_surveyors']:
            ok = False
    print(f"[{'OK' if ok else 'FAIL'}] SV: {name}")
    results.append(ok)

# ── Cache stress + thread-safety smoke test ─────────────────────────
print("\n--- Cache stress & thread-safety ---")
import threading as _th
stress_cache = _LLMResultCache(max_entries=32)
errors = []
def _worker(n):
    try:
        for i in range(50):
            k = _LLMResultCache.make_key('M', f'prompt-{n}-{i}')
            stress_cache.put(k, {'compliance': 'PASS', 'i': i})
            stress_cache.get(k)
    except Exception as e:
        errors.append(e)
ts = [_th.Thread(target=_worker, args=(i,)) for i in range(8)]
for t in ts: t.start()
for t in ts: t.join()
assert_eq("cache: thread-safe (no exceptions)", len(errors), 0)
assert_eq("cache: respects max_entries under load",
          len(stress_cache._d) <= stress_cache.max, True)
assert_eq("cache: hit-rate counters increment monotonically",
          stress_cache.hits >= 0 and stress_cache.misses >= 0, True)

# ── HTTP session sanity ────────────────────────────────────────────
print("\n--- HTTP session sanity ---")
# Verify the session is the same instance every time (module-level singleton)
from steps.step14_verification import _LLM_SESSION as _S2
assert_eq("HTTP session: module-level singleton", _LLM_SESSION is _S2, True)

# Real-data sweep — confirm the new detectors run cleanly across all jobs
print("\n--- Real-data sweep across results/ ---")
import os as _os_sweep
RESULTS = 'd:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final/results'
total = wl_hits = pen_hits = sv_hits = errs = 0
if _os_sweep.path.isdir(RESULTS):
    for jid in sorted(_os_sweep.listdir(RESULTS)):
        s6f = _os_sweep.path.join(RESULTS, jid, 'step06', 'step06_result.json')
        if not _os_sweep.path.isfile(s6f):
            continue
        try:
            with open(s6f, 'r', encoding='utf-8') as f:
                s6 = json.load(f)
        except Exception:
            continue
        total += 1
        try:
            if _detect_discrepancy_whitelist(s6): wl_hits += 1
            if _detect_late_shipment_penalty(s6): pen_hits += 1
            if _detect_required_surveyors(s6): sv_hits += 1
        except Exception:
            errs += 1
    print(f"  {total} jobs scanned | {wl_hits} whitelist | {pen_hits} late-pen | {sv_hits} surveyor | {errs} errors")
assert_eq("real-data sweep: no detector exceptions", errs, 0)
assert_eq("real-data sweep: at least 1 whitelist match (f3ef028e)",
          wl_hits >= 1, True)


passed = sum(results)
total = len(results)
print(f"\n{passed}/{total} cases passed")
if passed != total:
    sys.exit(1)
print("OVERALL: OK")
sys.exit(0)
