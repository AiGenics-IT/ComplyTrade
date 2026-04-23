"""Comprehensive dry-run battery for P198ak/am/an/ao/ap."""
import re

# ================================================================
# Battery 1: P198ak proforma-date parser — date format variety
# ================================================================
print("=" * 72)
print("Battery 1: P198ak proforma-date parser - 23 date formats")
print("=" * 72)

_MONTHS = {'JAN':1,'FEB':2,'MAR':3,'APR':4,'MAY':5,'JUN':6,'JUL':7,'AUG':8,'SEP':9,'SEPT':9,
           'OCT':10,'NOV':11,'DEC':12,'JANUARY':1,'FEBRUARY':2,'MARCH':3,'APRIL':4,'JUNE':6,
           'JULY':7,'AUGUST':8,'SEPTEMBER':9,'OCTOBER':10,'NOVEMBER':11,'DECEMBER':12}

def pd(s):
    if not s: return None
    s = re.sub(r'(\d+)(ST|ND|RD|TH)\b', r'\1', str(s).upper().strip().rstrip('.,;:'))
    s = re.sub(r'\s+', ' ', s).strip()
    m = re.match(r'^([A-Z]+)\.?\s*[- ]?\s*(\d{1,2})\s*[,.\- ]\s*(\d{2,4})$', s)
    if m and _MONTHS.get(m.group(1)):
        y = int(m.group(3)); y = 2000+y if y<100 and y<=69 else (1900+y if y<100 else y)
        return (y, _MONTHS[m.group(1)], int(m.group(2)))
    m = re.match(r'^(\d{1,2})[\s\-./]+([A-Z]+)\.?[\s\-./]+(\d{2,4})$', s)
    if m and _MONTHS.get(m.group(2)):
        y = int(m.group(3)); y = 2000+y if y<100 and y<=69 else (1900+y if y<100 else y)
        return (y, _MONTHS[m.group(2)], int(m.group(1)))
    m = re.match(r'^(\d{4})[-./](\d{1,2})[-./](\d{1,2})$', s)
    if m: return (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = re.match(r'^(\d{1,2})[-./](\d{1,2})[-./](\d{2,4})$', s)
    if m:
        a, b = int(m.group(1)), int(m.group(2)); y = int(m.group(3))
        if y < 100: y = 2000+y if y<=69 else 1900+y
        if a > 12 and 1 <= b <= 12: return (y, b, a)
        if b > 12 and 1 <= a <= 12: return (y, a, b)
        return None
    m = re.match(r'^(\d{2})(\d{2})(\d{2})$', s)
    if m:
        y = 2000+int(m.group(1)) if int(m.group(1))<=69 else 1900+int(m.group(1))
        mo, d = int(m.group(2)), int(m.group(3))
        if 1 <= mo <= 12 and 1 <= d <= 31: return (y, mo, d)
    m = re.match(r'^(\d{4})(\d{2})(\d{2})$', s)
    if m:
        mo, d = int(m.group(2)), int(m.group(3))
        if 1 <= mo <= 12 and 1 <= d <= 31: return (int(m.group(1)), mo, d)
    return None

tests_dates = [
    ('JAN 21, 2026', (2026,1,21)), ('Jan 21, 2026', (2026,1,21)),
    ('JANUARY 21, 2026', (2026,1,21)), ('JAN.21.2026', (2026,1,21)),
    ('JAN 21 2026', (2026,1,21)), ('21 JAN 2026', (2026,1,21)),
    ('21-JAN-2026', (2026,1,21)), ('21/JAN/2026', (2026,1,21)),
    ('21 JANUARY 2026', (2026,1,21)), ('2026-01-21', (2026,1,21)),
    ('2026/01/21', (2026,1,21)), ('2026.01.21', (2026,1,21)),
    ('21-01-2026', (2026,1,21)), ('01-21-2026', (2026,1,21)),
    ('05-06-2026', None), ('21-01-26', (2026,1,21)),
    ('260121', (2026,1,21)), ('20260121', (2026,1,21)),
    ('JAN 21st 2026', (2026,1,21)), ('21ST JAN 2026', (2026,1,21)),
    ('SEPT 5, 2024', (2024,9,5)), ('', None), ('abc', None), ('13-13-2026', None),
]
passed = 0
for s, expected in tests_dates:
    got = pd(s)
    if got == expected: passed += 1
    else: print(f"  FAIL: pd({s!r}) = {got}, expected {expected}")
print(f"  Battery 1: {passed}/{len(tests_dates)} passed")

# ================================================================
# Battery 2: P198ap F45A -> Packing List fan-out + PASS/FAIL simulation
# ================================================================
print()
print("=" * 72)
print("Battery 2: P198ap F45A -> PL fan-out + correct/incorrect values")
print("=" * 72)

def build_tasks(rows, packets, flag_on):
    tasks = []
    for r in rows:
        matches = [p for p in packets if p['document_type'].lower() == r['document_to_check'].lower()]
        if not matches:
            tasks.append({'row_id': r['row_id'], 'skip': True, 'reason': 'doc_not_found'}); continue
        for m in matches:
            tasks.append({'row_id': r['row_id'], 'skip': False,
                          'document_type': m['document_type'],
                          'clause_ref': r['clause_ref'],
                          'condition_text': r['condition_text'],
                          'look_for': r.get('look_for', ''),
                          'document_text': m['text']})
    if flag_on:
        pl = [p for p in packets if p['document_type'].lower() == 'packing list']
        if pl:
            srcs = [t for t in tasks if not t.get('skip') and 'invoice' in t['document_type'].lower() and '45A' in t['clause_ref'].upper()]
            seen = set()
            for s in srcs:
                if s['row_id'] in seen: continue
                seen.add(s['row_id'])
                for p in pl:
                    tasks.append({'row_id': s['row_id'], 'skip': False,
                                  'document_type': p['document_type'],
                                  'clause_ref': s['clause_ref'],
                                  'condition_text': s['condition_text'],
                                  'look_for': s['look_for'],
                                  'document_text': p['text']})
    return tasks

def simulate_verdict(task):
    """Simulate what the verifier would conclude based on look_for in doc_text."""
    if task.get('skip'):
        return 'MISSING'
    lf = (task.get('look_for', '') or '').upper()
    txt = (task.get('document_text', '') or '').upper()
    if not lf:
        return 'N/A'
    return 'PASS' if lf in txt else 'FAIL'

def aggregate_existential(tasks, row_id):
    """ANY PASS -> row PASS, else FAIL."""
    row_tasks = [t for t in tasks if t.get('row_id') == row_id]
    results = [simulate_verdict(t) for t in row_tasks]
    if 'PASS' in results: return 'PASS'
    if 'MISSING' in results and 'PASS' not in results: return 'FAIL(missing)'
    return 'FAIL'

# Scenario A: flag OFF - only CI checked
rows = [
    {'row_id':'R01','clause_ref':'45A-1','document_to_check':'Commercial Invoice',
     'condition_text':"Goods desc 'SOYBEANS'",'look_for':'SOYBEANS'},
    {'row_id':'R02','clause_ref':'45A-1','document_to_check':'Commercial Invoice',
     'condition_text':"Qty 1000 MT",'look_for':'1000 MT'},
]

scenarios = [
    ('A: flag OFF + CI has correct + PL absent', False,
     [
         {'document_type':'Commercial Invoice','text':'Invoice: SOYBEANS 1000 MT USD 481.58/MT'},
     ],
     {'R01':'PASS','R02':'PASS'}),
    ('B: flag OFF + CI has correct + PL present-correct', False,
     [
         {'document_type':'Commercial Invoice','text':'Invoice: SOYBEANS 1000 MT'},
         {'document_type':'Packing List','text':'PL: SOYBEANS 1000 MT'},
     ],
     {'R01':'PASS','R02':'PASS'}),
    ('C: flag ON + CI correct + PL correct', True,
     [
         {'document_type':'Commercial Invoice','text':'Invoice: SOYBEANS 1000 MT'},
         {'document_type':'Packing List','text':'PL: SOYBEANS 1000 MT net weight'},
     ],
     {'R01':'PASS','R02':'PASS'}),
    ('D: flag ON + CI correct + PL INCORRECT (different goods)', True,
     [
         {'document_type':'Commercial Invoice','text':'Invoice: SOYBEANS 1000 MT'},
         {'document_type':'Packing List','text':'PL: CORN 500 MT (wrong data)'},
     ],
     {'R01':'PASS','R02':'PASS'}),  # existential: CI passes, so row PASS
    ('E: flag ON + CI INCORRECT + PL correct', True,
     [
         {'document_type':'Commercial Invoice','text':'Invoice: CORN 500 MT (wrong)'},
         {'document_type':'Packing List','text':'PL: SOYBEANS 1000 MT'},
     ],
     {'R01':'PASS','R02':'PASS'}),  # existential: PL passes
    ('F: flag ON + CI INCORRECT + PL INCORRECT', True,
     [
         {'document_type':'Commercial Invoice','text':'Invoice: CORN 500 MT'},
         {'document_type':'Packing List','text':'PL: RICE 2000 MT'},
     ],
     {'R01':'FAIL','R02':'FAIL'}),  # both wrong -> FAIL
    ('G: flag ON + PL absent + CI correct', True,
     [
         {'document_type':'Commercial Invoice','text':'Invoice: SOYBEANS 1000 MT'},
     ],
     {'R01':'PASS','R02':'PASS'}),  # no PL clone, CI passes
    ('H: flag ON + PL absent + CI INCORRECT', True,
     [
         {'document_type':'Commercial Invoice','text':'Invoice: CORN 500 MT'},
     ],
     {'R01':'FAIL','R02':'FAIL'}),  # CI only, fails
]

for label, flag, packets, expected in scenarios:
    tasks = build_tasks(rows, packets, flag)
    got = {rid: aggregate_existential(tasks, rid) for rid in expected}
    ok = got == expected
    status = 'OK' if ok else 'FAIL'
    print(f"  [{status}] {label}")
    if not ok:
        print(f"      expected: {expected}")
        print(f"      got:      {got}")
    # Show per-packet breakdown
    r01_tasks = [t for t in tasks if t.get('row_id') == 'R01']
    breakdown = [(t.get('document_type'), simulate_verdict(t)) for t in r01_tasks]
    print(f"      R01 per-packet: {breakdown}")

# ================================================================
# Battery 3: P198ao BL master-agency vs LC charter party rules
# ================================================================
print()
print("=" * 72)
print("Battery 3: P198ao BL master-agency + LC CPBL rule combinations")
print("=" * 72)
_MASTER_AGENCY = (
    'AS AGENTS FOR AND ON BEHALF OF THE MASTER','AS AGENT FOR AND ON BEHALF OF THE MASTER',
    'AS AGENTS FOR THE MASTER','AS AGENT FOR THE MASTER',
    'AS AGENT ON BEHALF OF THE MASTER','AS AGENTS ON BEHALF OF THE MASTER',
    'ON BEHALF OF THE MASTER AS AGENT','ON BEHALF OF THE MASTER AS AGENTS',
    'AS AGENTS ONLY FOR AND BY AUTHORITY OF THE MASTER','AS AGENT ONLY FOR AND BY AUTHORITY OF THE MASTER',
    'FOR THE MASTER AS AGENT','FOR THE MASTER AS AGENTS',
    'AGENT FOR MASTER','AGENTS FOR MASTER',
    'AS AGENT FOR THE CARRIER','AS AGENTS FOR THE CARRIER',
    'FOR AND ON BEHALF OF THE CARRIER',
    'AS AGENT FOR AND ON BEHALF OF THE OWNER','AS AGENTS FOR AND ON BEHALF OF THE OWNER',
)

def p198ao(cond, doc_text, current, signing=''):
    cu = cond.upper(); du = doc_text.upper()
    if current != 'FAIL': return current
    rel = ('CHARTER PARTY','CHARTER-PARTY','FORWARDER','HOUSE BL','HOUSE BILL','SIGNED BY','SIGNATORY','AGENT','MASTER','CARRIER')
    if not any(m in cu for m in rel): return current
    prohib = any(m in cu for m in ('NOT ACCEPTABLE','NOT PERMITTED','NOT ALLOWED','MUST NOT BE','UNACCEPTABLE','SHALL NOT','WILL NOT','NOT BE ACCEPT'))
    permiss = (not prohib) and any(m in cu for m in ('ACCEPTABLE','PERMITTED','ALLOWED','MAY BE','CAN BE'))
    has_text = any(p in du for p in _MASTER_AGENCY)
    is_struct = signing.lower() in ('agent_for_master','master_signed','carrier_signed','agent_for_carrier','agent_for_owner')
    if not (has_text or is_struct): return current
    if prohib and 'CHARTER PARTY' in cu:
        if 'CHARTER PARTY' in du: return current
    elif prohib and ('FORWARDER' in cu or 'HOUSE' in cu):
        pass
    elif permiss:
        pass
    else:
        return current
    return 'PASS'

cases_bl = [
    ('CPBL acceptable + agent signing', 'Charter party BL acceptable',
     'AS AGENTS FOR AND ON BEHALF OF THE MASTER CAPT LIN', 'FAIL', 'agent_for_master', 'PASS'),
    ('CPBL prohibited + NO CP text', 'Charter party BL not acceptable',
     'AS AGENT FOR THE MASTER', 'FAIL', 'agent_for_master', 'PASS'),
    ('CPBL prohibited + HAS CP text', 'Charter party BL not acceptable',
     'CHARTER PARTY BILL\nAS AGENT FOR MASTER', 'FAIL', 'agent_for_master', 'FAIL'),
    ('Forwarder prohibited + agent signing', 'Must not be forwarder BL',
     'AS AGENT ONLY FOR AND BY AUTHORITY OF THE MASTER', 'FAIL', 'agent_for_master', 'PASS'),
    ('House BL prohibited + agent signing', 'House BL not acceptable',
     'FOR AND ON BEHALF OF THE CARRIER', 'FAIL', 'carrier_signed', 'PASS'),
    ('Already PASS - dont touch', 'Charter party acceptable',
     'AS AGENT FOR MASTER', 'PASS', 'agent_for_master', 'PASS'),
    ('No master-agency evidence', 'Charter party acceptable',
     'Signed by XYZ Line', 'FAIL', '', 'FAIL'),
    ('Non-BL condition', 'Invoice must show total',
     'AS AGENT FOR MASTER', 'FAIL', 'agent_for_master', 'FAIL'),
]
for label, cond, doc, cur, sig, expected in cases_bl:
    got = p198ao(cond, doc, cur, sig)
    ok = got == expected
    print(f"  [{'OK' if ok else 'FAIL'}] {label}: {cur}->{got}")

print()
print("=" * 72)
print("DONE")
print("=" * 72)
