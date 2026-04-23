"""
Dry-run for job 3f448670:
 1. Date check (F31C pre-dated-docs) against Commercial Invoice, Packing List, BL.
 2. Shipment Advice content checks (addressed-to, policy ref, details).
 3. Beneficiary Certificate presence / content.
Uses the latest in-memory step14_verification code (P198p) via importlib.reload,
so fixes to prompt rules (13y date comparison, T&C marker, etc.) are exercised
even though the running server hasn't been restarted yet.
"""
import json, importlib, sys
import steps.step14_verification as s14v
import steps.step12_decomposition as s12
importlib.reload(s14v)
importlib.reload(s12)

JOB = '3f448670-23fc-434b-8638-fa07742cb711'

def _load(p):
    with open(p, 'rb') as f:
        return json.loads(f.read().decode('utf-8', errors='replace'))

s7 = _load(f'results/{JOB}/step07/step07_result.json')
s9 = _load(f'results/{JOB}/step09/step09_result.json')

pkts = s9.get('reconciled_packets') or []
def _find(doc_substr):
    return [p for p in pkts if doc_substr.lower() in (p.get('document_type','') or '').lower()]

ci = _find('commercial invoice')[0] if _find('commercial invoice') else None
pl = _find('packing list')[0] if _find('packing list') else None
bl = _find('bill of lading')[0] if _find('bill of lading') else None
sa = _find('shipment advice')[0] if _find('shipment advice') else None
bc = _find('beneficiary')[0] if _find('beneficiary') else None

_inner = s7.get('structured_lc', s7)
ctx = s12._extract_lc_context(_inner)
lc_parties = f"Applicant: {ctx['applicant'][:80]}\nBeneficiary: {ctx['beneficiary'][:80]}\nIssuing Bank: {ctx['issuing_bank']}"
cf_sf = s7.get('structured_lc',{}).get('standalone_fields',{}) or {}

def _run(row_id, cond, clref, pkt, label):
    if not pkt:
        print(f"\n--- {label}: PACKET NOT FOUND ---")
        return
    res = s14v._call_vlm(
        row_id=row_id, condition_text=cond, clause_ref=clref,
        lc_field_value='',
        f47a_context=ctx['f47a_additional_conditions'][:1500],
        document_type=pkt.get('document_type','?'),
        document_text=s14v._pkt_text(pkt), image_path=None,
        visual_metadata=s14v._pkt_visual_metadata(pkt),
        lc_parties=lc_parties,
        unified_summary=pkt.get('unified_summary'),
        bl_subtype=pkt.get('bl_subtype'),
        final_lc_fields=cf_sf,
    )
    v = (res.get('compliance') or 'review').upper()
    f = (res.get('findings') or res.get('result') or '')[:320]
    print(f"\n--- {label} ---")
    print(f"CONDITION: {cond}")
    print(f"VERDICT:   {v}")
    print(f"FINDING:   {f.encode('ascii','replace').decode()}")

print("="*90)
print("1. F31C DATE CHECKS (LC issue date 2025-01-02)")
print("="*90)
DATE_COND = ("All shipping documents must be issued on or after LC issuance date "
             "(F31C: 2025-01-02). Documents dated prior to LC date are discrepancies "
             "per UCP 600 Art 14(i).")
_run('DRY-D1', DATE_COND, '47A-1', ci, 'Commercial Invoice')
_run('DRY-D2', DATE_COND, '47A-1', pl, 'Packing List')
_run('DRY-D3', DATE_COND, '47A-1', bl, 'Bill of Lading')

print("\n" + "="*90)
print("2. SHIPMENT ADVICE checks")
print("="*90)
SA_COND1 = ("Shipment Advice must be addressed to Century Insurance Company Limited, "
            "Office 504 and 505, 5th Floor, Marine Point, DC-1, Block-9, Clifton, "
            "Karachi, Pakistan.")
SA_COND2 = ("Shipment Advice must also be addressed to the Applicant "
            "(BIKIYA INDUSTRIES (PVT) LIMITED).")
SA_COND3 = "Shipment Advice must reference Cover Note No. C/08/MN/00037802/21."
SA_COND4 = ("Shipment Advice must mention vessel name, BL number, shipment date, "
            "invoice value, and credit number.")
_run('DRY-S1', SA_COND1, '46A-4', sa, 'Shipment Advice — addressed to Century Insurance')
_run('DRY-S2', SA_COND2, '46A-4', sa, 'Shipment Advice — also addressed to Applicant')
_run('DRY-S3', SA_COND3, '46A-4', sa, 'Shipment Advice — references Cover Note')
_run('DRY-S4', SA_COND4, '46A-4', sa, 'Shipment Advice — mentions shipment details')

print("\n" + "="*90)
print("3. BENEFICIARY CERTIFICATE checks")
print("="*90)
BC_COND1 = "Beneficiary Certificate must accompany the original documents."
BC_COND2 = ("Beneficiary Certificate must state / evidence that the Shipment Advice "
            "was sent to the named parties by email.")
_run('DRY-B1', BC_COND1, '46A-4', bc, 'Beneficiary Certificate — presence / accompanying')
_run('DRY-B2', BC_COND2, '46A-4', bc, 'Beneficiary Certificate — content')
