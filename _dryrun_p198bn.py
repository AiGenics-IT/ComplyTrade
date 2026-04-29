"""P198bn dry-run: boilerplate-aware prohibited-marker rescue against
the actual 73be98d9 BL text."""
import json
import re

_DEFINITION_MARKERS = (
    'MEANS ', 'MEAN ', 'SHALL MEAN', 'INCLUDES ',
    'DEFINED AS', 'DEFINITION OF', 'REFERS TO',
    'INTERPRETED AS', 'DEFINED HEREIN',
    '"NVOCC"', '"NVOCG"', "'NVOCC'",
    'DEFINITIONS', 'GLOSSARY',
)


def has_real_context_match(text_up, tok):
    idx = 0
    while True:
        pos = text_up.find(tok, idx)
        if pos < 0:
            return False
        pre = text_up[max(0, pos - 80): pos]
        if any(m in pre for m in _DEFINITION_MARKERS):
            idx = pos + 1
            continue
        if '"' in pre[-40:] and 'MEANS' in text_up[pos:pos + 80]:
            idx = pos + 1
            continue
        return True


# ── Load the actual job's BL text ──
with open('results/73be98d9-724f-4500-a08c-79802b4a5794/step09/step09_result.json',
          encoding='utf-8') as f:
    d = json.load(f)
bl_text = ''
for pkt in d.get('reconciled_packets', []):
    if 'bill of lading' in pkt.get('document_type', '').lower():
        bl_text = (
            pkt.get('refined_text')
            or pkt.get('cleaned_text')
            or pkt.get('raw_text')
            or ''
        ).upper()
        break

print('=== Job 73be98d9 BL ===')
print(f'BL text length: {len(bl_text)}')
print(f'Occurrences of NON VESSEL OPERATING: {bl_text.count("NON VESSEL OPERATING")}')
print(f'Real-context NVOCC match (NON VESSEL OPERATING): {has_real_context_match(bl_text, "NON VESSEL OPERATING")}')
print(f'Real-context FIATA match: {has_real_context_match(bl_text, "FIATA")}')
print(f'Real-context FORWARDER BILL match: {has_real_context_match(bl_text, "FORWARDER BILL OF LADING")}')
print()

# ── Synthetic cases: BL that IS actually NVOCC ──
print('=== Synthetic: BL where NVOCC is REAL (should keep FAIL) ===')
real_nvocc = """
ISSUED BY: XYZ LOGISTICS, NON VESSEL OPERATING COMMON CARRIER.
SIGNED AS AGENT FOR NVOCC BY JOHN DOE.
""".upper()
print(f'Real-context NVOCC match: {has_real_context_match(real_nvocc, "NON VESSEL OPERATING")}')

print()
print('=== Synthetic: BL where FIATA is in signature/issuer (real) ===')
real_fiata = """
BILL OF LADING NO. 123
ISSUER: ACME FREIGHT FIATA MEMBER
SIGNED AS FREIGHT FORWARDER
""".upper()
print(f'Real-context FIATA match: {has_real_context_match(real_fiata, "FIATA")}')

print()
print('=== Synthetic: Pure definition block (should ignore) ===')
def_only = """
"FIATA" MEANS THE INTERNATIONAL FEDERATION OF FREIGHT FORWARDERS ASSOCIATIONS.
"NVOCC" MEANS NON VESSEL OPERATING COMMON CARRIER AS DEFINED HEREIN.
""".upper()
print(f'Real-context FIATA match (should be False): {has_real_context_match(def_only, "FIATA")}')
print(f'Real-context NVOCC match (should be False): {has_real_context_match(def_only, "NON VESSEL OPERATING")}')
