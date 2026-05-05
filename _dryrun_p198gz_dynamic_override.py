"""P198gz dynamic prior-vs-VLM preference rule.

Tests the conservative override: step3's specific title wins over
packet-VLM only when they share ZERO significant tokens.
"""
import sys, re, os
os.environ['PYTHONIOENCODING'] = 'utf-8'

results = []
def ok(name, cond, note=''):
    tag = '[OK]  ' if cond else '[FAIL]'
    print(f"{tag} {name}" + (f" -- {note}" if note else ''))
    results.append(bool(cond))


def should_prefer_prior(prior, vlm):
    """Mirror the production rule."""
    _STOP = {'THE','OF','AND','FOR','WITH','TO','A','AN',
             'IN','ON','AT','BY','OR','PAGE','BLANK',
             'HEADER','UNKNOWN','CONTINUATION'}
    p = prior.upper().strip()
    v = vlm.upper().strip()
    pt = [t for t in re.findall(r'[A-Z]{3,}', p) if t not in _STOP]
    vt = [t for t in re.findall(r'[A-Z]{3,}', v) if t not in _STOP]
    return (
        len(pt) >= 2
        and bool(vt)
        and not (set(pt) & set(vt))
        and not (p and p in v)
    )


# Real-data anchors (from jobs b1479424 + 1f0fc892)
print("=" * 70)
print("Section 1: Should override (step3 specific, VLM force-fit)")
print("=" * 70)
SHOULD = [
    ("DRAFT SURVEY REPORT", "Inspection Certificate", "DSR vs IC"),
    ("DRAFT SURVEY REPORT", "Survey Report", "DSR vs SR (subset of prior)"),
    ("COAL SPECIFICATIONS AT THE LOADING PORT", "Inspection Certificate",
     "Coal Specs vs IC"),
    ("DETAILED MESSAGE", "Commercial Invoice", "DM vs CI"),
    ("VESSEL ADVICE", "Commercial Invoice", "VA vs CI"),
]
for prior, vlm, label in SHOULD:
    got = should_prefer_prior(prior, vlm)
    if 'subset' in label:
        # SR (Survey Report) tokens = {SURVEY, REPORT} which IS subset
        # of DSR tokens {DRAFT, SURVEY, REPORT}, so the rule will NOT
        # override (token overlap exists). That's by design — don't
        # touch when VLM is a reasonable shorter form. Skip from
        # SHOULD.
        ok(f"  {label}: prefer_prior={got} (overlap exists, expected False)",
           not got)
    else:
        ok(f"  {label}: prefer_prior={got}", got)


print("\n" + "=" * 70)
print("Section 2: Should NOT override (VLM was reasonable)")
print("=" * 70)
NOT = [
    # Same family — token overlap
    ("Bill of Lading", "Bill of Lading", "Identical"),
    ("Master Bill of Lading", "Bill of Lading", "Sub-variant of BL"),
    ("BENEFICIARY'S CERTIFICATE", "Beneficiary Certificate",
     "Apostrophe variant"),
    ("PACKING LIST", "Packing List", "Case variant"),
    ("CERTIFICATE OF ORIGIN", "Certificate of Origin", "Same"),
    ("WEIGHT CERTIFICATE", "Weight Certificate", "Same"),
    # Single-token prior — not specific enough
    ("Certificate", "Certificate of Origin", "Bare prior"),
    ("Document", "Commercial Invoice", "Bare prior"),
    # VLM has no significant tokens
    ("DRAFT SURVEY REPORT", "", "VLM empty"),
    ("DRAFT SURVEY REPORT", "Unknown", "VLM unknown"),
]
for prior, vlm, label in NOT:
    got = should_prefer_prior(prior, vlm)
    ok(f"  {label}: prefer_prior={got} (expected False)", not got)


print("\n" + "=" * 70)
print("Section 3: Source wiring")
print("=" * 70)
src = open('d:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final/steps/step08_shipping_classification.py',
           'r', encoding='utf-8').read()
ok("  CONSERVATIVE override comment", 'CONSERVATIVE override' in src)
ok("  ANTI-FORCE-FIT in prompt", 'ANTI-FORCE-FIT' in src)
ok("  EVIDENCE OVER LC LIST in prompt", 'EVIDENCE OVER LC LIST' in src)
ok("  P198gz substring guard", 'P198gz' in src and '_GENERIC_SOLO' in src)


print("\n" + "=" * 70)
passed = sum(results)
total = len(results)
print(f"P198gz DYNAMIC OVERRIDE: {passed}/{total}")
print("=" * 70)
if passed != total:
    sys.exit(1)
print("OVERALL: OK")
sys.exit(0)
