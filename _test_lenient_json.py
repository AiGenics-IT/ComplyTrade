import json

def _lenient_json_loads(s):
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    out = []
    in_str = False
    escape = False
    for ch in s:
        if escape:
            out.append(ch)
            escape = False
            continue
        if ch == "\\" and in_str:
            out.append(ch)
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            out.append(ch)
            continue
        if in_str and ch == '\n':
            out.append('\\n')
            continue
        if in_str and ch == '\r':
            out.append('\\r')
            continue
        if in_str and ch == '\t':
            out.append('\\t')
            continue
        out.append(ch)
    try:
        return json.loads(''.join(out))
    except json.JSONDecodeError:
        return None

bad = '''{ "verdict": "PASS", "quote": "SHIPMENT ADVISE
DATE: 02.FEB, 2025
TO: CENTURY INSURANCE COMPANY LIMITED,
OFFICE 504 AND 5TH FLOOR, MARINE POINT,
DC-1, BLOCK-9, CLIFTON KARACHI, PAKISTAN", "findings": "doc present" }'''

try:
    json.loads(bad)
    print('strict PASS (unexpected)')
except Exception as e:
    print('strict fails:', e)

p = _lenient_json_loads(bad)
print('lenient verdict:', p.get('verdict') if p else None)
print('lenient findings:', (p.get('findings') if p else None))
print('lenient quote len:', len(p.get('quote','')) if p else 0)
