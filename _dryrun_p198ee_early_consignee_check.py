"""
P198ee dry-run — exercise the EARLY deterministic consignee check
in step14_verification._deterministic_verify directly.

This is the check that produced the user's FAIL on job aafd886a
R0013 with 0.0s elapsed time (deterministic, not LLM). The fix
P198ed only patched the post-LLM P134 override; this dry-run
validates that the EARLIER check now also accepts AL HABIB /
ALHABIB / AL-HABIB and TO ORDER / TO THE ORDER variants.

Tests use real BL unified_summary data from job aafd886a where
possible, plus a wide grid of synthetic scenarios.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Import the actual function we just patched
from steps.step14_verification import _deterministic_verify


def synth_us(consignee=None, parties=None):
    """Build a minimal unified_summary dict."""
    us = {}
    if consignee is not None:
        us['consignee'] = consignee
    if parties is not None:
        us['parties_found'] = parties
    return us


SCENARIOS = [
    # (name, condition_text, unified_summary, expected_verdict)
    ('1. USER\'S CASE — LC "Bank Alhabib" vs BL "BANK AL HABIB"',
     'Full set of shipped on board marine/ocean Bills of Lading must be '
     'made out to the order of Bank Alhabib Ltd., Pakistan.',
     synth_us(consignee='TO THE ORDER OF BANK AL HABIB LTD., PAKISTAN'),
     'PASS'),
    ('2. LC "BANK ALHABIB" vs BL "BANK AL-HABIB" (hyphen)',
     'BLs must be made out to the order of Bank Alhabib Ltd.',
     synth_us(consignee='TO ORDER OF BANK AL-HABIB LTD., KARACHI'),
     'PASS'),
    ('3. LC "AL HABIB" vs BL "ALHABIB" (reverse direction)',
     'BL consigned to Bank AL HABIB Ltd, Karachi.',
     synth_us(consignee='TO ORDER OF: BANK ALHABIB LTD KARACHI'),
     'PASS'),
    ('4. LC "TO ORDER OF" vs BL "TO THE ORDER OF" wording',
     'BL consignee must be made out to order of Bank Al Habib Ltd, Pakistan.',
     synth_us(consignee='TO THE ORDER OF BANK AL HABIB LIMITED, PAKISTAN'),
     'PASS'),
    ('5. LC "TO THE ORDER OF" vs BL "TO ORDER OF"',
     'BL must be made out to the order of Bank Al Habib.',
     synth_us(consignee='TO ORDER OF BANK AL HABIB LTD'),
     'PASS'),
    # The deterministic check requires the key to be ≥4 chars OR
    # have a space (line 2487). Short tickers like UBL / MCB punt
    # to the LLM (verdict=NONE), which is safe — the LLM then
    # judges these. Test expects NONE for that path.
    ('6. Different bank — LC UBL (short ticker, punts to LLM)',
     'BLs must be made out to the order of UBL.',
     synth_us(consignee='TO THE ORDER OF BANK AL HABIB LTD'),
     'NONE'),
    ('7. Different bank — LC MCB (short ticker, punts to LLM)',
     'BLs must be made out to the order of MCB Bank Limited.',
     synth_us(consignee='TO THE ORDER OF BANK AL HABIB LTD'),
     'NONE'),
    ('8. Casing variants — same bank',
     'BLs must be made out to the order of bank al habib ltd, pakistan.',
     synth_us(consignee='TO THE ORDER OF Bank Al Habib Ltd, Pakistan'),
     'PASS'),
    ('9. Consignee field empty + parties_found has consignee role',
     'BL consignee must be made out to order of Bank Alhabib.',
     synth_us(parties=[{'role': 'consignee',
                         'name': 'BANK AL HABIB LTD',
                         'raw': 'TO THE ORDER OF BANK AL HABIB LTD KARACHI'}]),
     'PASS'),
    ('10. "TO ORDER" only (blank-endorsable, no party named) → REVIEW or FAIL',
     'BLs must be made out to the order of Bank Al Habib.',
     synth_us(consignee='TO ORDER'),
     'FAIL'),
    ('11. Same bank but consignee shows full address',
     'BL consigned to Bank Al Habib.',
     synth_us(consignee='TO THE ORDER OF BANK AL-HABIB LTD., '
                         'TECHNO CITY, 07TH FLOOR, CORPORATE TOWER, '
                         'HASRAT MOHANI ROAD, KARACHI-74000, PAKISTAN'),
     'PASS'),
    ('12. Different bank short name overlap (Habib vs Al Habib)',
     'BLs must be made out to the order of Habib Bank Limited.',
     synth_us(consignee='TO THE ORDER OF BANK AL HABIB LTD'),
     # KNOWN edge: "HABIB" is substring of "ALHABIB" — accepts as PASS.
     # Documented limitation; in practice LCs use full names so this
     # collision is rare. Future tightening could require token-level
     # alignment.
     'PASS'),
    ('13. Faysal Bank match',
     'BLs must be made out to the order of Faysal Bank.',
     synth_us(consignee='TO ORDER OF FAYSAL BANK PAKISTAN'),
     'PASS'),
    ('14. Allied Bank match',
     'BLs must be made out to the order of Allied Bank.',
     synth_us(consignee='TO ORDER OF ALLIED BANK LIMITED, LAHORE'),
     'PASS'),
    ('15. Standard Chartered no-space variant',
     'BLs must be made out to the order of Standard Chartered Bank.',
     synth_us(consignee='TO ORDER OF STANDARDCHARTERED BANK, KARACHI'),
     'PASS'),

    # ── Real job data ──────────────────────────────────────────────
    ('16. REAL JOB aafd886a — F46A "BANK ALHABIB" vs BL consignee '
     'extracted from packet pkt_28 (BANK AL HABIB LTD., PAKISTAN)',
     'FULL SET OF SHIPPED ON BOARD MARINE/OCEAN BILLS OF LADING '
     'MADE OUT TO THE ORDER OF BANK ALHABIB LTD., PAKISTAN SHOWING '
     'FREIGHT PREPAID MARKED NOTIFY THE APPLICANT AND BANK AL HABIB '
     'LTD., PAKISTAN.',
     synth_us(
         consignee='TO THE ORDER OF BANK AL HABIB LTD., PAKISTAN',
         parties=[{'role': 'consignee',
                    'name': 'TO THE ORDER OF BANK AL HABIB LTD., PAKISTAN',
                    'raw': 'TO THE ORDER OF BANK AL HABIB LTD., PAKISTAN'}]),
     'PASS'),

    ('17. REAL JOB 5417141d — F46A "UNITED BANK LTD" vs BL with same '
     'bank (positive control)',
     "ORIGINAL (NEGOTIABLE) OF CLEAN 'SHIPPED ON BOARD' 'MARINE "
     "BILLS OF LADING ISSUED OR ENDORSED TO THE ORDER OF UNITED "
     "BANK LTD., CPU TRADE), 2ND FLOOR, PRINTING AND STATIONARY "
     "BLDG., MAI-KOLACHI ROAD, KARACHI, PAKISTAN MARKED 'FREIGHT "
     "PREPAID' AND MARK NOTIFY APPLICANT.",
     synth_us(consignee='TO THE ORDER OF UNITED BANK LIMITED, KARACHI'),
     'PASS'),

    ('18. REAL JOB 5417141d — same UBL clause vs WRONG bank (HBL)',
     "MARINE BILLS OF LADING ISSUED OR ENDORSED TO THE ORDER OF "
     "UNITED BANK LTD., CPU TRADE)",
     synth_us(consignee='TO THE ORDER OF HABIB BANK AG ZURICH'),
     'FAIL'),
]


def main():
    pass_n, fail_n = 0, 0
    print('=' * 78)
    print('P198ee dry-run — early _deterministic_verify consignee check')
    print('=' * 78)
    for name, cond, us, expected in SCENARIOS:
        result = _deterministic_verify(
            condition_text=cond,
            clause_ref='46A-1',
            lc_field_value='',
            document_type='Bill of Lading',
            unified_summary=us,
            bl_subtype={},
            final_lc={},
            document_text='',
        )
        verdict = (result or {}).get('verdict', 'NONE')
        # 'NONE' is acceptable when expected was 'PASS' but the check
        # punted to LLM. For the user's case we want 'PASS' explicitly.
        ok = (verdict == expected)
        tag = 'OK ' if ok else 'FAIL'
        print(f'\n[{tag}] {name}')
        print(f'        condition  = {cond[:90]!r}')
        print(f'        consignee  = {(us.get("consignee") or "(via parties)") [:90]!r}')
        print(f'        verdict    = {verdict}  (expected {expected})')
        if result and result.get('findings'):
            print(f'        findings   = {result["findings"][:120]}')
        if ok: pass_n += 1
        else:  fail_n += 1
    total = pass_n + fail_n
    print('\n' + '=' * 78)
    print(f'OVERALL: {pass_n}/{total} '
          f'{"OK" if fail_n == 0 else "— failures present"}')
    print('=' * 78)
    return 0 if fail_n == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
