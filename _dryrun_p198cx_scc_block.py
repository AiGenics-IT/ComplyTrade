"""
P198cx dry-run — Block Shipping Company Certificate force-fit.

VLM was misclassifying surveyor-issued certificates as "Shipping
Company Certificate" just because the LC required one and the VLM
saw vessel-related content on the page. Fix: when VLM says SCC
but the document carries high-specificity markers for a different
certificate family (Shelf Life, Last 3 Cargoes, Analysis, Survey
Report, Phyto, Health, Fumigation, Halal) OR step 3 already
labelled it with one of those names OR the document is surveyor-
issued (Control Union, SGS, Alfred H Knight, Intertek, etc.),
override the VLM and keep the specific type.

Real-job scenarios from 4690d9bc:
  pkt_22 — "Certificate" (Last 3 Cargoes statement from Control
           Union) → overridden to "Last Cargoes Statement"
  pkt_23 — "SHELF LIFE CERTIFICATE" from step 3 → overridden to
           "Shelf Life Certificate"
"""
import re, sys, os


_SPECIFIC_CERT_MARKERS = (
    (r'\bSHELF\s+LIFE\s+CERTIFICATE\b', 'Shelf Life Certificate'),
    (r'\bSHELF\s+LIFE\b.*\b(?:EXPIRY|PRODUCTION)\s+DATE\b',
     'Shelf Life Certificate'),
    (r'\bLAST\s+\d\s+CARGOES\b', 'Last Cargoes Statement'),
    (r'\bLAST\s+THREE\s+CARGOES\b', 'Last Cargoes Statement'),
    (r'\bPREVIOUS\s+CARGOES?\b', 'Last Cargoes Statement'),
    (r'\bFOSFA\s+INTERNATIONAL\s+LIST\s+OF\s+BANNED\s+PREVIOUS\s+CARGOES\b',
     'Last Cargoes Statement'),
    (r'\bCERTIFICATE\s+OF\s+ANALYSIS\b', 'Certificate of Analysis'),
    (r'\bCERTIFICATE\s+OF\s+QUALITY\s+AND\s+WEIGHT\b',
     'Certificate of Quality and Weight'),
    (r'\bLOAD\s+PORT\s+SURVEY\s+REPORT\b', 'Load Port Survey Report'),
    (r'\bDISCHARGE\s+SURVEY\s+REPORT\b', 'Discharge Survey Report'),
    (r'\bDRAUGHT\s+SURVEY\s+REPORT\b', 'Draught Survey Report'),
    (r'\bPHYTOSANITARY\b', 'Phytosanitary Certificate'),
    (r'\bHEALTH\s+CERTIFICATE\b', 'Health Certificate'),
    (r'\bFUMIGATION\s+CERTIFICATE\b', 'Fumigation Certificate'),
    (r'\bHALAL\s+CERTIFICATE\b', 'Halal Certificate'),
)
_PRIOR_SPECIFIC_NAMES = {
    'SHELF LIFE CERTIFICATE': 'Shelf Life Certificate',
    'CERTIFICATE OF ANALYSIS': 'Certificate of Analysis',
    'CERTIFICATE OF QUALITY AND WEIGHT': 'Certificate of Quality and Weight',
    'CERTIFICATE OF QUALITY': 'Certificate of Quality',
    'LOAD PORT SURVEY REPORT': 'Load Port Survey Report',
    'DISCHARGE SURVEY REPORT': 'Discharge Survey Report',
    'DRAUGHT SURVEY REPORT': 'Draught Survey Report',
    'LAST 3 CARGOES': 'Last Cargoes Statement',
    'LAST CARGOES': 'Last Cargoes Statement',
    'PHYTOSANITARY CERTIFICATE': 'Phytosanitary Certificate',
    'HEALTH CERTIFICATE': 'Health Certificate',
    'FUMIGATION CERTIFICATE': 'Fumigation Certificate',
    'HALAL CERTIFICATE': 'Halal Certificate',
}
_SURVEYOR_NAMES = (
    'CONTROL UNION', 'SGS', 'ALFRED H KNIGHT',
    'INTERTEK', 'BUREAU VERITAS', 'SAYBOLT',
    'CORNELDER', 'COTECNA', 'OMIC',
)


def simulate(vlm_dt, glm_text, prior_step3_dt):
    if (vlm_dt or '').strip().upper() != 'SHIPPING COMPANY CERTIFICATE':
        return vlm_dt, 'VLM did not say SCC; no override'
    glm_up = (glm_text or '').upper()
    prior_up = (prior_step3_dt or '').upper()
    for pat, name in _SPECIFIC_CERT_MARKERS:
        if re.search(pat, glm_up) or re.search(pat, prior_up):
            return name, f'marker {pat!r} matched'
    for key, name in _PRIOR_SPECIFIC_NAMES.items():
        if key in prior_up:
            return name, f'step3 label {key!r}'
    if any(s in glm_up for s in _SURVEYOR_NAMES):
        return prior_step3_dt or 'Certificate', 'surveyor issuer detected'
    return vlm_dt, 'no markers; keeping VLM SCC'


# Real pkt_22 body (Last 3 Cargoes from Control Union)
PKT_22_TEXT = """Certificate
LAST 3 CARGOES
Certificate No: RQ/306163
Vessel : M/T SEA LEGEND
Port of Loading : SAN LORENZO, ARGENTINA
Port of Discharge : KHI PORT / PORT QASIM
Quantity : 250.000 MT
Description of goods : CRUDE DEGUMMED SOYABEAN OIL
We received from ship's authorities a statement that the below mentioned ships tanks
have not contained any leaded petroleum or other leaded product on at least the
last three cargoes in the vessel's tanks which received the goods.
The immediate previous cargoes in tanks receiving the goods were not products appearing
on the FOSFA International List of Banned Previous Cargoes in force at the date of the bill.
Issued by CONTROL UNION ARGENTINA S.A.
INDEPENDENT SUREVEYOR
"""
# Real pkt_23 body (Shelf Life from Control Union)
PKT_23_TEXT = """Certificate
SHELF LIFE CERTIFICATE
Certificate No: RQ/306164
Vessel : M/T SEA LEGEND
Description of goods : CRUDE DEGUMMED SOYABEAN OIL
Based on shipper's declaration, we hereby certify the following:
- Production date: November 24th, 26th, 2024
- Expiry date: 1 year.
Manufacturer: Molinos Agro S.A.
Certificate issued at: BUENOS AIRES, ARGENTINA
Issued by CONTROL UNION ARGENTINA S.A.
INDEPENDENT SUREVEYOR
"""

# Genuine SCC text (for negative control)
SCC_GENUINE = """SHIPPING COMPANY CERTIFICATE
This is to certify that the carrying vessel M/T SEA LEGEND:
(i) is covered under the Institute Classification Clause
(ii) is owned by a company operating in accordance with Pakistani
Maritime rules and port regulations.
Issued by COSCO SHIPPING LINES — Authorized Agent
"""


SC = []
SC.append(dict(name='Real pkt_22: Last 3 Cargoes / Control Union → override to Last Cargoes Statement',
    vlm='Shipping Company Certificate', glm=PKT_22_TEXT, prior='Certificate',
    expect='Last Cargoes Statement'))
SC.append(dict(name='Real pkt_23: SHELF LIFE CERTIFICATE / Control Union → Shelf Life Certificate',
    vlm='Shipping Company Certificate', glm=PKT_23_TEXT, prior='SHELF LIFE CERTIFICATE',
    expect='Shelf Life Certificate'))
SC.append(dict(name='Certificate of Analysis body → override',
    vlm='Shipping Company Certificate',
    glm='CERTIFICATE OF ANALYSIS\nParameters...', prior='CERTIFICATE OF ANALYSIS',
    expect='Certificate of Analysis'))
SC.append(dict(name='Load Port Survey Report → override',
    vlm='Shipping Company Certificate',
    glm='LOAD PORT SURVEY REPORT\n...', prior='LOAD PORT SURVEY REPORT',
    expect='Load Port Survey Report'))
SC.append(dict(name='Health Certificate wrongly labelled SCC → override',
    vlm='Shipping Company Certificate',
    glm='HEALTH CERTIFICATE\nFIT FOR HUMAN CONSUMPTION', prior='Health Certificate',
    expect='Health Certificate'))
SC.append(dict(name='Phytosanitary wrongly labelled SCC → override',
    vlm='Shipping Company Certificate',
    glm='PHYTOSANITARY CERTIFICATE\nPlant health...', prior='Phytosanitary Certificate',
    expect='Phytosanitary Certificate'))
SC.append(dict(name='SGS-issued survey report (no marker match, surveyor fallback)',
    vlm='Shipping Company Certificate',
    glm='Report\nIssued by SGS Argentina S.A.\nAt load port...',
    prior='Survey Report', expect='Survey Report'))
SC.append(dict(name='Genuine SCC with Institute Classification Clause → stays SCC',
    vlm='Shipping Company Certificate', glm=SCC_GENUINE, prior='Shipping Company Certificate',
    expect='Shipping Company Certificate'))
SC.append(dict(name='VLM returned something else (not SCC) → no override',
    vlm='Certificate of Origin', glm=PKT_22_TEXT, prior='Certificate',
    expect='Certificate of Origin'))
SC.append(dict(name='No prior and no markers, no surveyor → stays SCC',
    vlm='Shipping Company Certificate',
    glm='CERTIFICATE\nThis certifies that vessel is ok.', prior='Certificate',
    expect='Shipping Company Certificate'))
SC.append(dict(name='Alfred H Knight surveyor → override to step3 or Certificate',
    vlm='Shipping Company Certificate',
    glm='Certificate\nIssued by ALFRED H KNIGHT\nCargo inspected.',
    prior='Quality Inspection Report', expect='Quality Inspection Report'))
SC.append(dict(name='FOSFA list reference → Last Cargoes Statement',
    vlm='Shipping Company Certificate',
    glm='FOSFA INTERNATIONAL LIST OF BANNED PREVIOUS CARGOES', prior='Certificate',
    expect='Last Cargoes Statement'))


def main():
    passed = 0; failed = 0
    for i, sc in enumerate(SC, 1):
        got, note = simulate(sc['vlm'], sc['glm'], sc['prior'])
        ok = (got == sc['expect'])
        tag = 'OK ' if ok else 'FAIL'
        print(f"[{tag}] #{i:02d}  {sc['name']}")
        print(f"         expect={sc['expect']!r}, got={got!r}")
        print(f"         note: {note}")
        if ok: passed += 1
        else: failed += 1
    print(f"\n{'='*78}\n{passed}/{passed+failed} P198cx SCC-block scenarios OK\n{'='*78}")
    return failed == 0


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
