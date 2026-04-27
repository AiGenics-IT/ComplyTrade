"""
P198dc dry-run — F45A goods-description / quantity fan-out to
Packing List.

When step 12 emits an F45A condition targeting the Commercial
Invoice for goods description or quantity, step 13 now clones
the row with document_checked='Packing List'. Both docs are
verified against the same condition, matching banking practice.

Exceptions:
  • Unit price / per-piece rate / Incoterms (CFR/CIF/FOB/EXW/DDP)
    rows are CI-only — Packing List doesn't carry pricing.
  • Proforma-invoice citation rows are CI-only — UCP 600 Art 18(c).
  • Total value / invoice value rows are CI-only.
"""
import re
import sys


def should_clone(field_tag, doc_to_check, cond_text):
    """Mirror of P198dc clone-decision logic in step13."""
    ftag_u = (field_tag or '').upper()
    doc_u = (doc_to_check or '').upper()
    cond_u = (cond_text or '').upper()
    if '45A' not in ftag_u or 'COMMERCIAL INVOICE' not in doc_u:
        return False, 'not 45A->CI'
    is_goods = any(k in cond_u for k in (
        'GOODS DESCRIPTION', 'DESCRIPTION OF GOODS',
        'PRODUCT NAME', 'PRODUCT DESCRIPTION',
        'MERCHANDISE', 'COMMODITY',
        'GOODS MUST', 'DESCRIPTION MUST',
    ))
    is_qty = any(k in cond_u for k in (
        'QUANTITY', 'NUMBER OF UNITS', 'NUMBER OF PIECES',
        'TOTAL QUANTITY', 'TOTAL UNITS', 'NET WEIGHT',
        'GROSS WEIGHT', 'TOTAL WEIGHT', 'METRIC TONS',
        'NO. OF PCS', 'NO. OF UNITS',
        'NO OF PCS', 'NO OF UNITS',
    ))
    is_price_or_other = any(k in cond_u for k in (
        'UNIT PRICE', 'PER PC', 'PER PIECE', 'PER UNIT',
        'PER MT', 'PER KG', 'AT THE RATE',
        'INCOTERMS', 'CFR', 'CIF', 'FOB', 'EXW', 'DDP',
        'PROFORMA', 'PRO-FORMA', 'PRO FORMA',
        'TOTAL VALUE', 'TOTAL AMOUNT', 'INVOICE VALUE',
        'STRICTLY AS PER', 'STRICTLY AS PER BENEFICIARY',
    ))
    if (is_goods or is_qty) and not is_price_or_other:
        return True, ('goods+' if is_goods else '') + ('qty' if is_qty else '')
    return False, ('price/other' if is_price_or_other else 'no goods/qty match')


SC = []

# Goods description rows → CLONE
SC.append(dict(name='F45A goods description on CI → CLONE',
    ftag='45A', doc='Commercial Invoice',
    cond='Goods description on Commercial Invoice must read SOYBEANS.',
    expect=True))
SC.append(dict(name='F45A description of goods on CI → CLONE',
    ftag='45A', doc='Commercial Invoice',
    cond='Description of goods must match LC F45A.',
    expect=True))
SC.append(dict(name='F45A product name on CI → CLONE',
    ftag='45A', doc='Commercial Invoice',
    cond='Product name on Commercial Invoice must be COMPRESSOR DONPER L68WU1.',
    expect=True))
SC.append(dict(name='F45A merchandise certification → CLONE',
    ftag='45A', doc='Commercial Invoice',
    cond='Merchandise must be certified as Brazil origin.',
    expect=True))

# Quantity rows → CLONE
SC.append(dict(name='F45A quantity on CI → CLONE',
    ftag='45A', doc='Commercial Invoice',
    cond='Quantity must be 5000 MT (+/- 10%).',
    expect=True))
SC.append(dict(name='F45A net weight on CI → CLONE',
    ftag='45A', doc='Commercial Invoice',
    cond='Net weight must match LC requirement.',
    expect=True))
SC.append(dict(name='F45A number of units on CI → CLONE',
    ftag='45A', doc='Commercial Invoice',
    cond='Number of units must be 1152 PCS.',
    expect=True))

# Price / Incoterms / citation rows → DO NOT clone
SC.append(dict(name='F45A unit price → NO CLONE',
    ftag='45A', doc='Commercial Invoice',
    cond='Unit price must be USD 19.60 per PC for COMPRESSOR.',
    expect=False))
SC.append(dict(name='F45A Incoterms (CFR) → NO CLONE',
    ftag='45A', doc='Commercial Invoice',
    cond='Trade terms must be CFR Karachi Port (Incoterms 2020).',
    expect=False))
SC.append(dict(name='F45A proforma citation → NO CLONE',
    ftag='45A', doc='Commercial Invoice',
    cond='Specifications must strictly follow Beneficiary\'s Proforma Invoice 786/S-13198-SOYPI-E.',
    expect=False))
SC.append(dict(name='F45A total value → NO CLONE',
    ftag='45A', doc='Commercial Invoice',
    cond='Total invoice value must match LC amount USD 481,580.',
    expect=False))

# Non-45A rows → NO CLONE
SC.append(dict(name='F46A consignee on BL → NO CLONE',
    ftag='46A', doc='Bill of Lading',
    cond='Consignee must be TO ORDER OF BANK AL HABIB.',
    expect=False))
SC.append(dict(name='F45A condition but doc != CI → NO CLONE',
    ftag='45A', doc='Bill of Lading',
    cond='Goods description must match LC.',
    expect=False))


def main():
    passed = 0; failed = 0
    for i, sc in enumerate(SC, 1):
        clone, why = should_clone(sc['ftag'], sc['doc'], sc['cond'])
        ok = (clone == sc['expect'])
        tag = 'OK ' if ok else 'FAIL'
        print(f"[{tag}] #{i:02d}  {sc['name']}")
        print(f"         expect_clone={sc['expect']}, got={clone}  ({why})")
        if ok: passed += 1
        else: failed += 1
    print(f"\n{'='*78}\n{passed}/{passed+failed} P198dc F45A->Packing List scenarios OK\n{'='*78}")
    return failed == 0


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
