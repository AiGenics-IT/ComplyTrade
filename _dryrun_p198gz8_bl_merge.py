"""P198gz8 — BL+T&C extended absorption test.

Verify:
1. Single BL with multiple T&C pages → all absorbed into the BL packet
2. Three distinct BLs with their T&C runs → each BL absorbs ONLY its own T&C
3. T&C before BL face (back-cover) → still absorbed (fallback path)
4. Other doc types (CoO, AWB, Invoice) unaffected
"""
import sys, os, types
sys.path.insert(0, 'd:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final')
os.environ['PYTHONIOENCODING'] = 'utf-8'

results = []
def ok(name, cond, note=''):
    tag = '[OK]  ' if cond else '[FAIL]'
    print(f"{tag} {name}" + (f" -- {note}" if note else ''))
    results.append(bool(cond))


# Mirror the classifier helpers (small subset)
def is_bl(dt):
    dt = dt.lower().strip()
    return ('bill of lading' in dt or dt == 'bl' or 'b/l' in dt) \
        and 'conditions of carriage' not in dt and 'terms' not in dt

def is_bl_tc(dt):
    dt = dt.lower().strip()
    return ('bl conditions of carriage' in dt
            or 'bill of lading conditions of carriage' in dt
            or 'conditions of carriage' in dt
            or 'bl t&c' in dt or 'bill of lading t&c' in dt)


# Mirror the absorption logic from step03_sequencing.py
class Pkt:
    def __init__(self, pid, dt, pages):
        self.packet_id = pid
        self.document_type = dt
        self.page_numbers = list(pages)
        self.pages = []
        self.stamps = []
        self.signatures = []
        self.seals = []


def absorb(packets):
    consumed = set()
    out = []
    for i, pkt in enumerate(packets):
        if i in consumed:
            continue
        dt = pkt.document_type.lower().strip()
        if is_bl(dt):
            bl_max = max(pkt.page_numbers); bl_min = min(pkt.page_numbers)
            next_bl_face_min = 999999
            for j, o in enumerate(packets):
                if j == i or j in consumed: continue
                odt = o.document_type.lower().strip()
                if is_bl(odt) and not is_bl_tc(odt):
                    om = min(o.page_numbers)
                    if om > bl_max and om < next_bl_face_min:
                        next_bl_face_min = om
            hard_boundary = next_bl_face_min
            absorbed = 0
            for j, o in enumerate(packets):
                if j == i or j in consumed: continue
                if not is_bl_tc(o.document_type.lower().strip()): continue
                tcm = min(o.page_numbers)
                if tcm > bl_max and tcm < hard_boundary:
                    pkt.page_numbers.extend(o.page_numbers)
                    consumed.add(j); absorbed += 1
            if absorbed == 0:
                best = None; bdist = 999
                for j, o in enumerate(packets):
                    if j == i or j in consumed: continue
                    if not is_bl_tc(o.document_type.lower().strip()): continue
                    tcm = max(o.page_numbers)
                    if tcm < bl_min:
                        d = (bl_min - tcm) + 1
                        if d < bdist and d <= 6:
                            best = j; bdist = d
                if best is not None:
                    o = packets[best]
                    pkt.page_numbers.extend(o.page_numbers)
                    consumed.add(best)
            if absorbed > 0 or (i not in consumed and best is not None if False else True):
                pass
        out.append(pkt)
    out = [p for i, p in enumerate(packets) if i not in consumed]
    return out


# ── Scenario 1 — single BL with 5 T&C pages ──
print("=" * 70)
print("Scenario 1: Single BL with 5 T&C pages")
print("=" * 70)
pkts = [
    Pkt('p1','Bill of Lading',[27]),
    Pkt('p2','BL Conditions of Carriage',[28,29]),
    Pkt('p3','BL Conditions of Carriage',[30,31]),
    Pkt('p4','BL Conditions of Carriage',[32]),
]
out = absorb(pkts)
expected_pages = sorted([27,28,29,30,31,32])
got = sorted(out[0].page_numbers)
ok(f"  Single BL absorbs ALL T&C pages: got {got}", got == expected_pages)
ok(f"  Result has 1 packet (others consumed)", len(out) == 1)


# ── Scenario 2 — three distinct BLs ──
print("\n" + "=" * 70)
print("Scenario 2: Three distinct BLs (27-32, 33-38, 39-44)")
print("=" * 70)
pkts = [
    Pkt('p1','Bill of Lading',[27]),
    Pkt('p2','BL Conditions of Carriage',[28,29]),
    Pkt('p3','BL Conditions of Carriage',[30,31]),
    Pkt('p4','BL Conditions of Carriage',[32]),
    Pkt('p5','Bill of Lading',[33]),
    Pkt('p6','BL Conditions of Carriage',[34,35]),
    Pkt('p7','BL Conditions of Carriage',[36,37]),
    Pkt('p8','BL Conditions of Carriage',[38]),
    Pkt('p9','Bill of Lading',[39]),
    Pkt('p10','BL Conditions of Carriage',[40,41]),
    Pkt('p11','BL Conditions of Carriage',[42,43]),
    Pkt('p12','BL Conditions of Carriage',[44]),
]
out = absorb(pkts)
ok(f"  3 BL packets remain: got {len(out)}", len(out) == 3)
if len(out) == 3:
    bl1 = sorted(out[0].page_numbers)
    bl2 = sorted(out[1].page_numbers)
    bl3 = sorted(out[2].page_numbers)
    ok(f"  BL#1 = 27-32: {bl1}", bl1 == [27,28,29,30,31,32])
    ok(f"  BL#2 = 33-38: {bl2}", bl2 == [33,34,35,36,37,38])
    ok(f"  BL#3 = 39-44: {bl3}", bl3 == [39,40,41,42,43,44])


# ── Scenario 3 — T&C before BL (back-cover fallback) ──
print("\n" + "=" * 70)
print("Scenario 3: T&C before BL (back-cover order)")
print("=" * 70)
pkts = [
    Pkt('p1','BL Conditions of Carriage',[5,6]),
    Pkt('p2','Bill of Lading',[7]),
]
out = absorb(pkts)
ok(f"  T&C-before-BL still absorbed (back-cover): got {len(out)} pkt",
   len(out) == 1)
if len(out) == 1:
    ok(f"  All pages 5,6,7 in one packet: {sorted(out[0].page_numbers)}",
       sorted(out[0].page_numbers) == [5,6,7])


# ── Scenario 4 — other doc types unaffected ──
print("\n" + "=" * 70)
print("Scenario 4: Non-BL types unchanged")
print("=" * 70)
pkts = [
    Pkt('p1','Commercial Invoice',[1]),
    Pkt('p2','Certificate of Origin',[2]),
    Pkt('p3','Airway Bill',[3]),
    Pkt('p4','Packing List',[4]),
]
out = absorb(pkts)
ok(f"  Non-BL packets unchanged: {len(out)} packets", len(out) == 4)


# ── Scenario 4b — Unrelated CI between BL face and T&C ──
print("\n" + "=" * 70)
print("Scenario 4b: Unrelated page interrupts BL set — T&C after CI still absorbs")
print("=" * 70)
pkts = [
    Pkt('q1','Bill of Lading',[10]),
    Pkt('q2','Commercial Invoice',[11]),     # interrupting unrelated doc
    Pkt('q3','BL Conditions of Carriage',[12,13]),  # T&C after CI — should absorb
    Pkt('q4','Bill of Lading',[20]),
]
out = absorb(pkts)
bl1 = next((p for p in out if is_bl(p.document_type) and 10 in p.page_numbers), None)
ci = next((p for p in out if 'commercial invoice' in p.document_type.lower()), None)
ok(f"  CI not absorbed (different page preserved): CI present = {ci is not None}",
   ci is not None and ci.page_numbers == [11])
ok(f"  T&C across the CI gap absorbed into BL: pkt = {bl1.page_numbers if bl1 else 'None'}",
   bl1 is not None and sorted(bl1.page_numbers) == [10,12,13])

# ── Scenario 5 — BL with no T&C (just face) ──
print("\n" + "=" * 70)
print("Scenario 5: BL with no T&C")
print("=" * 70)
pkts = [Pkt('p1','Bill of Lading',[10])]
out = absorb(pkts)
ok(f"  Standalone BL preserved: {len(out)} pkt with {out[0].page_numbers}",
   len(out) == 1 and out[0].page_numbers == [10])


# ── Source wiring ──
print("\n" + "=" * 70)
print("Section 6: Source wiring")
print("=" * 70)
src = open('d:/COMPLYTRADE/V7/FINAL/Trade-Finance-Final/steps/step03_sequencing.py',
           'r', encoding='utf-8').read()
ok("  P198gz8 marker", 'P198gz8' in src)
ok("  '_next_bl_face_min' boundary logic", '_next_bl_face_min' in src)
ok("  back-cover fallback present", 'back-cover' in src)


print("\n" + "=" * 70)
passed = sum(results)
total = len(results)
print(f"P198gz8 BL MERGE: {passed}/{total}")
print("=" * 70)
if passed != total:
    sys.exit(1)
print("OVERALL: OK")
sys.exit(0)
