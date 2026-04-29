"""
P198dp dry-run #2 — Run the DR guard against EVERY real job in
`results/` that currently has a Documentary Remittance packet.

For each DR packet:
  - Load its OCR text from step02 (concatenated cleaned_text of its
    page numbers).
  - Run the guard.
  - If the guard demotes a real bank covering schedule to
    'Covering Letter', that is a regression — flag it.

This catches false-NEGATIVE behaviour (over-aggressive demotion) on
the broad zoo of real bank schedules in the local results store.
"""
import json
import re
import sys
import os
import glob
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')


_DR_REAL_SIGNALS = [
    r'WE\s+ENCLOSE\s+(?:THE\s+)?(?:FOLLOWING\s+|ABOVE\s+)?'
    r'DOCUMENTS?(?:\s+(?:FOR|DRAWN))?',
    r'WE\s+ARE\s+PLEASED\s+TO\s+ENCLOSE',
    r'WE\s+HEREBY\s+ENCLOSE',
    r'ENCLOSED\s+HEREWITH',
    r'PRESENTATION\s+(?:NUMBER|NO\.?|DATE|AMOUNT)',
    r'TOTAL\s+(?:AMOUNT\s+)?CLAIMED',
    r'PRINCIPAL\s+AMOUNT\s+(?:CLAIMED|EUR|USD|GBP)',
    r'AMOUNTS?\s+CLAIMED\s*[:\n]',
    r'YOUR\s+DOCUMENTARY\s+CREDIT\s+NO',
    r'OUR\s+REFERENCE\s+NO',
    r'REMIT\s+FUNDS\s+TO\s+(?:OUR\s+)?CORRESPONDENT',
    r'(?:UPON|FOR)\s+SETTLEMENT\s+PLEASE\s+REMIT',
    r'QUOTING\s+OUR\s+REFERENCE',
    r'CLAIM\s+REIMBURSEMENT',
    r'COVERING\s+(?:LETTER|SCHEDULE)',
    r'DOCUMENTARY\s+REMITTANCE',
    r'L/?C\s+BILLS?\s+SCHEDULE',
    r'(?:DOCUMENT|EXPORT\s+DC)\s+PRESENTATION\s+SCHEDULE',
    r'SCHEDULE\s+OF\s+PRESENTATION',
    r'BILLS?\s+REMITTANCE\s+LETTER',
    r'\bL/?C\s+ISSUING\s+BANK\b',
    r'\bREIMBURSING\s+BANK\b',
    r'\bYOUR\s+DC\s+REF\b',
    r'\bOUR\s+REF\.\s',
    r'\bPAYMENT\s+INSTRUCTION\b',
    r'\bBILL\s+AMOUNT\b',
    r'DOCUMENTS?\s+SENT\s+TO\s+YOU\s+ON\s+APPROVAL',
    r'DRAWING\s+AMOUNT\s+(?:HAS\s+BEEN\s+)?(?:DULY\s+)?ENDORSED',
    r'PRESENTATION\s+IS\s+SUBJECT\s+TO',
    r'ADVISING\s+CHARGES?\s+AND\s+CONFIRMATION\s+CHARGES?',
]
_BANK_RE = re.compile(
    r'\b(?:MAYBANK|MALAYAN\s+BANKING|BANK\s+AL\s+HABIB|'
    r'HABIB\s+BANK|HBL\b|UBL\b|UNITED\s+BANK\s+LIMITED|'
    r'MEEZAN\s+BANK|FAYSAL\s+BANK|MCB\b|ALLIED\s+BANK|'
    r'STANDARD\s+CHARTERED|HSBC|CITIBANK|JP\s*MORGAN|'
    r'J\.P\.\s*MORGAN|BARCLAYS|DEUTSCHE\s+BANK|RBC\b|'
    r'ROYAL\s+BANK|BNP\s+PARIBAS|COMMERZBANK|MIZUHO|'
    r'BANK\s+OF\s+CHINA|ICBC|BANCO\b|CHINA\s+CONSTRUCTION|'
    r'WELLS\s+FARGO|BANK\s+OF\s+AMERICA|UNICREDIT|'
    r'SOCIETE\s+GENERALE|CREDIT\s+SUISSE|UBS\b|'
    r'NATIONAL\s+BANK|COMMERCIAL\s+BANK)\b'
)
_SWIFT_RE = re.compile(r'\bSWIFT\s*:\s*[A-Z]{6,11}\b')


def evaluate(text):
    u = (text or '').upper()
    signals = sum(1 for p in _DR_REAL_SIGNALS if re.search(p, u))
    bank = bool(_BANK_RE.search(u))
    swift = bool(_SWIFT_RE.search(u))
    is_email = bool(
        re.search(r'\bFROM\s*:\s*[^\n]*@', u)
        and re.search(r'\bSUBJECT\s*:', u)
    )
    if is_email:
        is_real = signals >= 3
    else:
        is_real = (
            signals >= 2
            or (bank and signals >= 1)
            or (swift and signals >= 1)
        )
    return ('keep' if is_real else 'demote'), signals, bank, swift


def page_text_for(job_id, page_nums):
    s2_path = Path('results') / job_id / 'step02' / 'step02_result.json'
    if not s2_path.exists():
        return ''
    s2 = json.loads(s2_path.read_text(encoding='utf-8'))
    pmap = {p['page_number']: (p.get('cleaned_text')
                                or p.get('raw_text') or '')
            for p in s2.get('pages', [])}
    return '\n\n'.join(pmap.get(pn, '') for pn in (page_nums or []))


def main():
    results = []
    for jf in sorted(glob.glob('results/*/step08/step08_result.json')):
        job_id = os.path.basename(os.path.dirname(os.path.dirname(jf)))
        try:
            d = json.loads(Path(jf).read_text(encoding='utf-8'))
        except Exception:
            continue
        for p in (d.get('classified_packets') or []):
            if (p.get('document_type') or '').lower() != 'documentary remittance':
                continue
            pgs = [pg.get('page_number')
                   for pg in (p.get('original_pages') or [])
                   if isinstance(pg, dict)]
            text = page_text_for(job_id, pgs)
            if not text or len(text) < 50:
                continue
            verdict, sig, bank, swift = evaluate(text)
            results.append((job_id, p.get('packet_id'), pgs, verdict,
                            sig, bank, swift))

    # Tally
    keep = [r for r in results if r[3] == 'keep']
    demote = [r for r in results if r[3] == 'demote']
    print(f'Real-job DR audit ({len(results)} packets currently labelled '
          f'Documentary Remittance):')
    print(f'  Guard verdict KEEP  : {len(keep)}')
    print(f'  Guard verdict DEMOTE: {len(demote)}')

    if demote:
        print(f'\nThese {len(demote)} previously-labelled DR packets would '
              f'be DEMOTED by the guard:')
        for j, pkt, pgs, _, sig, bank, swift in demote[:30]:
            print(f'  {j[:8]} {pkt} pgs={pgs} sig={sig} bank={bank} '
                  f'swift={swift}')
        if len(demote) > 30:
            print(f'  ... and {len(demote) - 30} more')

    print('\n(For each demoted packet, manual review recommended — but the '
          'guard is conservative: it keeps when ≥2 signals OR bank+1 OR '
          'swift+1 are present, which is the same shape as a real bank '
          'covering schedule.)')


if __name__ == '__main__':
    main()
