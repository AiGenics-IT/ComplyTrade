"""
P198 dry run — re-decompose the problem clauses from job
1287cfe0-4973-4460-9a52-14a8d74e2993 against the live LLM with the
updated prompt + full LC context, and print how the decomposition
turns out. This does NOT touch the server — it calls the decomposer
module directly.

Checks we expect to pass:
  (A) 47A-11 override propagates: BL/BC conditions NEVER include
      "must be issued by beneficiary" / "shipper must be beneficiary"
      when F47A has "THIRD PARTY DOCUMENTS ARE ACCEPTABLE EXCEPT
      DRAFT AND INVOICE".
  (B) 46A-9 sub-conditions: all 3 sub-conditions under a
      Beneficiary's Certificate parent route to document_to_check =
      "Beneficiary Certificate" (no sub-condition drifts to
      Documentary Remittance).
  (C) 46A-12 countries-of-origin: no redundant pair like "must
      mention multiple" + "must not state single".
"""
import json
from steps.step12_decomposition import _call_vlm_decompose

# Real LC context from job 1287cfe0 (paraphrased from the step06 / step07 output).
LC_CONTEXT = {
    'applicant': 'M/S ALEXER PAKISTAN LIMITED',
    'beneficiary': 'OLAM GLOBAL AGRI PTE. LTD.',
    'issuing_bank': 'BANK AL HABIB LIMITED',
    'currency_amount': 'USD',
    'f47a_additional_conditions': """
1) DOCUMENTS ISSUED PRIOR TO LC ISSUANCE DATE ARE NOT ACCEPTABLE.
2) ALL DOCUMENTS MUST BEAR LC/DC NUMBER.
...
10) BL MUST NOT BE STATED TO BE ISSUED BY A NON-VESSEL OPERATING CARRIER COMPANY.
11) THIRD PARTY DOCUMENTS ARE ACCEPTABLE EXCEPT DRAFT AND INVOICE.
""".strip(),
    'f45a_goods_description': 'CRUDE DEGUMMED SOYABEAN OIL OF INDONESIAN/MALAYSIAN ORIGIN',
    'f43t_transshipment': 'NOT ALLOWED',
    'f43p_partial_shipments': 'NOT ALLOWED',
    'f44c_latest_shipment': '2025-12-31',
}

CASES = [
    {
        'ref': '46A-9',
        'field_tag': '46A',
        'num': 9,
        'text': """BENEFICIARY'S CERTIFICATE CERTIFYING THAT CERTIFICATE OF ORIGIN ISSUED BY
GOVERNING AUTHORITY AS PER MALAYSIAN-PAKISTAN CLOSER ECONOMIC PARTNERSHIP
AGREEMENT (MPCEPA) RULES. AND CERTIFICATE OF ORIGIN ISSUED BY GOVERNING
AUTHORITY AS PER INDONESIA-PAKISTAN PREFERENTIAL TRADE AGREEMENT RULES OF
ORIGIN HAS BEEN SENT TO APPLICANT DIRECTLY. EVIDENCE TO THIS EFFECT MUST
ACCOMPANY WITH THE DOCUMENTS.""",
    },
    {
        'ref': '46A-12',
        'field_tag': '46A',
        'num': 12,
        'text': """IN CASE OF A SINGLE PRESENTATION/SHIPMENT UNDER LC COVERING THE FULL LC
VALUE, INVOICES ARE REQUIRED TO MENTION THE MULTIPLE COUNTRIES OF ORIGIN
COVERING THE SHIPMENT. HOWEVER, INVOICES STATING ALTERNATE OR SINGLE
COUNTRY OF ORIGIN ARE NOT ACCEPTABLE.""",
    },
    {
        'ref': '47A-10',
        'field_tag': '47A',
        'num': 10,
        'text': """BL MUST NOT BE STATED TO BE ISSUED BY A NON-VESSEL OPERATING CARRIER
COMPANY.""",
    },
    {
        # Sanity: a bare "BL shipper must be beneficiary" type clause
        # would normally invent a beneficiary-shipper rule. With F47A-11
        # in context it should NOT.
        'ref': '47A-11',
        'field_tag': '47A',
        'num': 11,
        'text': "THIRD PARTY DOCUMENTS ARE ACCEPTABLE EXCEPT DRAFT AND INVOICE.",
    },
]


def _short(s, n=120):
    s = (s or '').replace('\n', ' ').strip()
    return s if len(s) <= n else s[:n] + '…'


def check(conds, expect):
    """expect: dict of label -> callable(list[cond]) -> bool"""
    for label, fn in expect.items():
        ok = fn(conds)
        print(f'      [{"PASS" if ok else "FAIL"}] {label}')


def main():
    for case in CASES:
        print('=' * 78)
        print(f"Case {case['ref']}: {_short(case['text'], 180)}")
        r = _call_vlm_decompose(
            clause_ref=case['ref'],
            field_tag=case['field_tag'],
            clause_number=case['num'],
            clause_text=case['text'],
            lc_context=LC_CONTEXT,
        )
        conds = r.get('conditions', [])
        print(f"  → {len(conds)} condition(s), elapsed {r.get('elapsed', 0):.1f}s")
        for c in conds:
            print(f"    - doc={c.get('document_to_check','?'):<32} cond={_short(c.get('condition_text',''), 180)}")

        if case['ref'] == '46A-9':
            check(conds, {
                'all sub-conditions route to Beneficiary Certificate':
                    lambda cs: all('beneficiary' in (c.get('document_to_check','') or '').lower() for c in cs),
                'no sub-condition routes to Documentary Remittance':
                    lambda cs: not any('remittance' in (c.get('document_to_check','') or '').lower() for c in cs),
            })
        elif case['ref'] == '46A-12':
            texts = [(c.get('condition_text','') or '').lower() for c in conds]
            check(conds, {
                'no "single country" sub-condition (redundant with multiple)':
                    lambda cs: not any('single country' in t or 'only one country' in t for t in texts),
                'at least one "multiple countries" positive requirement':
                    lambda cs: any('multiple countries' in t for t in texts),
            })
        elif case['ref'] == '47A-11':
            check(conds, {
                '47A-11 is permissive → produces 0 rows OR only draft/invoice rows':
                    lambda cs: len(cs) == 0 or all(
                        'draft' in (c.get('document_to_check','') or '').lower()
                        or 'invoice' in (c.get('document_to_check','') or '').lower()
                        for c in cs
                    ),
            })


if __name__ == '__main__':
    main()
