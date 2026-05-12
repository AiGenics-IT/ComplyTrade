"""Generate ComplyTrade architecture PDF (boxes + arrows on page 1)."""
from reportlab.lib.pagesizes import landscape, A4, letter
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor, white, black
from reportlab.pdfgen import canvas as _canvas
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
    Flowable,
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER
import os

OUT = os.path.join(os.path.dirname(__file__), 'ARCHITECTURE.pdf')

# Color palette
C_BG       = HexColor('#f8fafc')
C_USER     = HexColor('#0ea5e9')   # sky blue
C_USER_BG  = HexColor('#e0f2fe')
C_M2       = HexColor('#1554ca')   # primary blue
C_M2_BG    = HexColor('#dbeafe')
C_M1       = HexColor('#7c3aed')   # purple
C_M1_BG    = HexColor('#ede9fe')
C_8082     = HexColor('#1e40af')
C_8083     = HexColor('#0d9488')   # teal
C_GLM      = HexColor('#d97706')   # amber
C_VLM      = HexColor('#9333ea')
C_LLM      = HexColor('#be123c')   # rose
C_BORDER   = HexColor('#cbd5e1')
C_TEXT     = HexColor('#0f172a')
C_MUTED    = HexColor('#475569')
C_ARROW    = HexColor('#334155')
C_ARROW_X  = HexColor('#9333ea')   # cross-machine arrows
C_ARROW_L  = HexColor('#16a34a')   # loopback arrows
C_LIGHT    = HexColor('#94a3b8')


def _box(c, x, y, w, h, fill, stroke, radius=8, line_w=1.4):
    c.setFillColor(fill)
    c.setStrokeColor(stroke)
    c.setLineWidth(line_w)
    c.roundRect(x, y, w, h, radius, stroke=1, fill=1)


def _text(c, x, y, txt, font='Helvetica', size=9, color=C_TEXT, anchor='start'):
    c.setFillColor(color)
    c.setFont(font, size)
    if anchor == 'middle':
        c.drawCentredString(x, y, txt)
    elif anchor == 'end':
        c.drawRightString(x, y, txt)
    else:
        c.drawString(x, y, txt)


def _arrow(c, x1, y1, x2, y2, color=C_ARROW, w=1.6, label=None,
           label_color=None, dashed=False, head=6, label_offset=4):
    c.setStrokeColor(color)
    c.setLineWidth(w)
    if dashed:
        c.setDash(4, 3)
    else:
        c.setDash([])
    c.line(x1, y1, x2, y2)
    c.setDash([])
    # arrowhead
    import math
    ang = math.atan2(y2 - y1, x2 - x1)
    hx1 = x2 - head * math.cos(ang - math.pi / 6)
    hy1 = y2 - head * math.sin(ang - math.pi / 6)
    hx2 = x2 - head * math.cos(ang + math.pi / 6)
    hy2 = y2 - head * math.sin(ang + math.pi / 6)
    c.setFillColor(color)
    p = c.beginPath()
    p.moveTo(x2, y2); p.lineTo(hx1, hy1); p.lineTo(hx2, hy2); p.close()
    c.drawPath(p, stroke=0, fill=1)
    if label:
        # midpoint, slightly offset perpendicular to the arrow
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2
        c.setFillColor(label_color or color)
        c.setFont('Helvetica-Bold', 7.5)
        # tiny white-ish background pad for readability
        tw = c.stringWidth(label, 'Helvetica-Bold', 7.5)
        c.setFillColor(white)
        c.rect(mx - tw/2 - 3, my + label_offset - 2,
               tw + 6, 11, stroke=0, fill=1)
        c.setFillColor(label_color or color)
        c.drawCentredString(mx, my + label_offset, label)


def draw_diagram_page(c, W, H):
    """Page 1 — architecture diagram with boxes and arrows."""
    # background
    c.setFillColor(C_BG)
    c.rect(0, 0, W, H, stroke=0, fill=1)

    # ── Title bar ──────────────────────────────────────────────
    c.setFillColor(C_M2)
    c.rect(0, H - 50, W, 50, stroke=0, fill=1)
    _text(c, W/2, H - 22, 'ComplyTrade — System Architecture',
          font='Helvetica-Bold', size=18, color=white, anchor='middle')
    _text(c, W/2, H - 40,
          'Two-machine deployment · loopback + cross-machine HTTP',
          font='Helvetica', size=10, color=HexColor('#cfdcf3'), anchor='middle')

    # ── User browser box (top center, below title) ────────────
    ux, uy, uw, uh = W/2 - 100, H - 105, 200, 40
    _box(c, ux, uy, uw, uh, C_USER_BG, C_USER, radius=10, line_w=1.6)
    _text(c, W/2, uy + 23, 'USER BROWSER',
          font='Helvetica-Bold', size=11, color=C_USER, anchor='middle')
    _text(c, W/2, uy + 9, 'Upload · Checklist · Verify · Report',
          font='Helvetica', size=8, color=C_MUTED, anchor='middle')

    # ── Machine 2 outer container (left side, larger) ─────────
    m2x, m2y = 30, 50
    m2w, m2h = 510, H - 165
    _box(c, m2x, m2y, m2w, m2h, white, C_M2, radius=12, line_w=2.2)
    # Machine 2 header band
    c.setFillColor(C_M2_BG)
    c.roundRect(m2x, m2y + m2h - 36, m2w, 36, 12, stroke=0, fill=1)
    # cover bottom corners
    c.rect(m2x, m2y + m2h - 36, m2w, 6, stroke=0, fill=1)
    _text(c, m2x + 14, m2y + m2h - 16, 'MACHINE 2',
          font='Helvetica-Bold', size=12, color=C_M2)
    _text(c, m2x + 14, m2y + m2h - 28,
          '2× RTX 5090 · 32 GB each · application + light OCR',
          font='Helvetica', size=8.5, color=C_MUTED)

    # ── 8082 box (top inside M2) ──────────────────────────────
    pad = 18
    box_w = m2w - 2 * pad
    box_h = 110
    s82_x = m2x + pad
    s82_y = m2y + m2h - 36 - pad - box_h
    _box(c, s82_x, s82_y, box_w, box_h, white, C_8082, radius=8, line_w=1.6)
    # title strip
    c.setFillColor(C_8082)
    c.roundRect(s82_x, s82_y + box_h - 26, box_w, 26, 8, stroke=0, fill=1)
    c.rect(s82_x, s82_y + box_h - 26, box_w, 8, stroke=0, fill=1)
    _text(c, s82_x + 12, s82_y + box_h - 16, '8082 — MAIN SERVER',
          font='Helvetica-Bold', size=11, color=white)
    _text(c, s82_x + box_w - 12, s82_y + box_h - 16,
          'FastAPI · Python', font='Helvetica', size=8,
          color=HexColor('#dbeafe'), anchor='end')
    # body bullets
    body = [
        '• Web UI (HTML views) · /api/upload · /api/result',
        '• step06 Final LC consolidation · step07 LC clauses',
        '• step12 clause decomposition · step14 verification',
        '• step15-19 consolidation · step20 report PDF',
    ]
    for i, line in enumerate(body):
        _text(c, s82_x + 14, s82_y + box_h - 42 - i * 13, line,
              font='Helvetica', size=8.5, color=C_TEXT)

    # ── 8083 box (middle) ─────────────────────────────────────
    s83_y = s82_y - 16 - 110
    _box(c, s82_x, s83_y, box_w, 110, white, C_8083, radius=8, line_w=1.6)
    c.setFillColor(C_8083)
    c.roundRect(s82_x, s83_y + 110 - 26, box_w, 26, 8, stroke=0, fill=1)
    c.rect(s82_x, s83_y + 110 - 26, box_w, 8, stroke=0, fill=1)
    _text(c, s82_x + 12, s83_y + 110 - 16, '8083 — CLASSIFIER SERVER',
          font='Helvetica-Bold', size=11, color=white)
    _text(c, s82_x + box_w - 12, s83_y + 110 - 16,
          'FastAPI · Python', font='Helvetica', size=8,
          color=HexColor('#ccfbf1'), anchor='end')
    body = [
        '• /classify — entry point (shared job_id with 8082)',
        '• Step 1: GLM-OCR (raw text)  +  Step 2: VLM rescue',
        '• Page classify · SWIFT pre-classify · LC requirements',
        '• Deep field extract · stamps/signatures · doc matching',
    ]
    for i, line in enumerate(body):
        _text(c, s82_x + 14, s83_y + 110 - 42 - i * 13, line,
              font='Helvetica', size=8.5, color=C_TEXT)

    # ── GLM-OCR box (bottom) ──────────────────────────────────
    glm_h = 70
    glm_y = s83_y - 16 - glm_h
    _box(c, s82_x, glm_y, box_w, glm_h, white, C_GLM, radius=8, line_w=1.6)
    c.setFillColor(C_GLM)
    c.roundRect(s82_x, glm_y + glm_h - 24, box_w, 24, 8, stroke=0, fill=1)
    c.rect(s82_x, glm_y + glm_h - 24, box_w, 6, stroke=0, fill=1)
    _text(c, s82_x + 12, glm_y + glm_h - 15, 'GLM-OCR  (port 8001)',
          font='Helvetica-Bold', size=11, color=white)
    _text(c, s82_x + box_w - 12, glm_y + glm_h - 15,
          'vLLM endpoint', font='Helvetica', size=8,
          color=HexColor('#fef3c7'), anchor='end')
    _text(c, s82_x + 14, glm_y + 28,
          '• Per-page raw OCR text', font='Helvetica',
          size=8.5, color=C_TEXT)
    _text(c, s82_x + 14, glm_y + 14,
          '• Hallucination-prone on faint scans → Qwen-VL rescues it',
          font='Helvetica', size=8.5, color=C_MUTED)

    # ── Machine 1 outer container (right side) ────────────────
    m1x = 565
    m1y = 50
    m1w = W - m1x - 30
    m1h = H - 165
    _box(c, m1x, m1y, m1w, m1h, white, C_M1, radius=12, line_w=2.2)
    c.setFillColor(C_M1_BG)
    c.roundRect(m1x, m1y + m1h - 36, m1w, 36, 12, stroke=0, fill=1)
    c.rect(m1x, m1y + m1h - 36, m1w, 6, stroke=0, fill=1)
    _text(c, m1x + 14, m1y + m1h - 16, 'MACHINE 1',
          font='Helvetica-Bold', size=12, color=C_M1)
    _text(c, m1x + 14, m1y + m1h - 28,
          '2× RTX Pro 6000 · 96 GB each · heavy 72B inference',
          font='Helvetica', size=8.5, color=C_MUTED)

    # ── Qwen-VL box (top inside M1) ───────────────────────────
    pad1 = 16
    vl_w = m1w - 2 * pad1
    vl_h = 150
    vl_x = m1x + pad1
    vl_y = m1y + m1h - 36 - pad1 - vl_h
    _box(c, vl_x, vl_y, vl_w, vl_h, white, C_VLM, radius=8, line_w=1.6)
    c.setFillColor(C_VLM)
    c.roundRect(vl_x, vl_y + vl_h - 26, vl_w, 26, 8, stroke=0, fill=1)
    c.rect(vl_x, vl_y + vl_h - 26, vl_w, 8, stroke=0, fill=1)
    _text(c, vl_x + 12, vl_y + vl_h - 16, 'Qwen2.5-VL-72B  (AWQ)',
          font='Helvetica-Bold', size=11, color=white)
    _text(c, vl_x + vl_w - 12, vl_y + vl_h - 16,
          'image + text', font='Helvetica', size=8,
          color=HexColor('#ede9fe'), anchor='end')
    _text(c, vl_x + 12, vl_y + vl_h - 42,
          'Used by 8083 for:', font='Helvetica-Bold',
          size=8.5, color=C_VLM)
    body = [
        '• Step 2 OCR rescue (faint scans)',
        '• Per-page doc-type classify',
        '• Deep field extract',
        '• Stamps / signatures detect',
        '• Doc-to-LC matching',
        '• Positioned-text bbox',
    ]
    for i, line in enumerate(body):
        _text(c, vl_x + 18, vl_y + vl_h - 56 - i * 13, line,
              font='Helvetica', size=8.5, color=C_TEXT)

    # ── Qwen-LLM box (bottom inside M1) ───────────────────────
    ll_h = 130
    ll_y = vl_y - 16 - ll_h
    _box(c, vl_x, ll_y, vl_w, ll_h, white, C_LLM, radius=8, line_w=1.6)
    c.setFillColor(C_LLM)
    c.roundRect(vl_x, ll_y + ll_h - 26, vl_w, 26, 8, stroke=0, fill=1)
    c.rect(vl_x, ll_y + ll_h - 26, vl_w, 8, stroke=0, fill=1)
    _text(c, vl_x + 12, ll_y + ll_h - 16, 'Qwen2.5-72B  (GPTQ-Int8)',
          font='Helvetica-Bold', size=11, color=white)
    _text(c, vl_x + vl_w - 12, ll_y + ll_h - 16,
          'text only', font='Helvetica', size=8,
          color=HexColor('#ffe4e6'), anchor='end')
    _text(c, vl_x + 12, ll_y + ll_h - 42,
          'Used by 8082 for:', font='Helvetica-Bold',
          size=8.5, color=C_LLM)
    body = [
        '• step12 clause decompose',
        '• step14 verification',
        '   (24 parallel workers)',
        '• step06 amendment merge',
    ]
    for i, line in enumerate(body):
        _text(c, vl_x + 18, ll_y + ll_h - 56 - i * 13, line,
              font='Helvetica', size=8.5, color=C_TEXT)

    # ────────────── ARROWS ──────────────────────────────────────
    # User -> 8082 (HTTPS)
    _arrow(c, W/2, uy, W/2, s82_y + box_h,
           color=C_USER, w=2.0, label='HTTPS · port 8082',
           label_color=C_USER, head=8)

    # 8082 -> 8083 (loopback)
    _arrow(c, s82_x + box_w/2, s82_y,
           s82_x + box_w/2, s83_y + 110,
           color=C_ARROW_L, w=2.0,
           label='HTTP loopback  /classify  (shared job_id)',
           label_color=C_ARROW_L, head=8)

    # 8083 -> GLM (loopback)
    _arrow(c, s82_x + box_w/2, s83_y,
           s82_x + box_w/2, glm_y + glm_h,
           color=C_ARROW_L, w=2.0,
           label='HTTP loopback  /api/ocr (per page)',
           label_color=C_ARROW_L, head=8)

    # 8083 -> VLM (cross-machine)
    _arrow(c, s82_x + box_w, s83_y + 110/2,
           vl_x, vl_y + vl_h/2,
           color=C_ARROW_X, w=2.0,
           label='HTTP · LAN · image+text',
           label_color=C_ARROW_X, head=8)

    # 8082 -> LLM (cross-machine)
    _arrow(c, s82_x + box_w, s82_y + box_h/2,
           vl_x, ll_y + ll_h/2,
           color=C_ARROW_X, w=2.0,
           label='HTTP · LAN · text',
           label_color=C_ARROW_X, head=8)

    # ── Legend ────────────────────────────────────────────────
    lx, ly = 30, 22
    _text(c, lx, ly, 'Legend:',
          font='Helvetica-Bold', size=8.5, color=C_MUTED)
    # green = loopback
    c.setStrokeColor(C_ARROW_L); c.setLineWidth(2.0)
    c.line(lx + 50, ly + 3, lx + 80, ly + 3)
    _text(c, lx + 84, ly, 'loopback (same machine)',
          font='Helvetica', size=8, color=C_TEXT)
    # purple = cross-machine
    c.setStrokeColor(C_ARROW_X); c.setLineWidth(2.0)
    c.line(lx + 230, ly + 3, lx + 260, ly + 3)
    _text(c, lx + 264, ly, 'cross-machine (LAN)',
          font='Helvetica', size=8, color=C_TEXT)
    # blue = user
    c.setStrokeColor(C_USER); c.setLineWidth(2.0)
    c.line(lx + 400, ly + 3, lx + 430, ly + 3)
    _text(c, lx + 434, ly, 'user',
          font='Helvetica', size=8, color=C_TEXT)
    # footer note
    _text(c, W - 30, ly,
          'page 1/3 · diagram · pages 2-3: services, endpoints, flow',
          font='Helvetica-Oblique', size=7.5,
          color=C_MUTED, anchor='end')


# ──────────────────────────────────────────────────────────────
# Page 2-3: tables + flow as Platypus flowables, but on a separate
# document. We mix custom-canvas (page 1) with Platypus (pages 2+)
# by drawing both onto the same canvas manually.
# ──────────────────────────────────────────────────────────────

def draw_table_page(c, W, H):
    """Page 2 — service responsibility table + endpoints."""
    c.setFillColor(C_BG); c.rect(0, 0, W, H, stroke=0, fill=1)
    # header
    c.setFillColor(C_M2); c.rect(0, H - 40, W, 40, stroke=0, fill=1)
    _text(c, 30, H - 25, 'Service Responsibilities & Endpoints',
          font='Helvetica-Bold', size=14, color=white)
    _text(c, W - 30, H - 25, 'page 2/3',
          font='Helvetica', size=9, color=HexColor('#cfdcf3'), anchor='end')

    # ── Responsibility table ──────────────────────────────────
    rows = [
        ['Machine', 'Service', 'Owns'],
        ['Machine 2', '8082 main server',
         'Upload UI · job state · step06 Final LC · step07 LC clauses · '
         'step12-14 verification fan-out · step15-20 report'],
        ['Machine 2', '8083 classifier',
         'Per-page OCR (GLM + VLM rescue) · doc-type classify · SWIFT '
         'pre-classify · LC requirements parse · deep field extract · '
         'stamps · positioned-text'],
        ['Machine 2', 'GLM-OCR (vLLM :8001)',
         'Step 1 — raw per-page text. Lightweight, fits on 5090s.'],
        ['Machine 1', 'Qwen2.5-VL-72B (AWQ)',
         'Image+text reasoning. Step 2 OCR rescue · classification · '
         'deep_extract · matching · bbox.'],
        ['Machine 1', 'Qwen2.5-72B (GPTQ-Int8)',
         'Text-only LLM. Clause decomposition (step12) · verification '
         '(step14, 24 parallel) · amendment merge (step06).'],
    ]
    styles = getSampleStyleSheet()
    cell_style = ParagraphStyle('cell', parent=styles['BodyText'],
                                fontSize=8.5, leading=11,
                                textColor=C_TEXT)
    head_style = ParagraphStyle('head', parent=styles['BodyText'],
                                fontSize=9, leading=12,
                                textColor=white,
                                fontName='Helvetica-Bold')
    data = []
    for i, r in enumerate(rows):
        s = head_style if i == 0 else cell_style
        data.append([Paragraph(c0, s) for c0 in r])
    t = Table(data, colWidths=[60*mm, 40*mm, 150*mm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), C_M2),
        ('GRID',       (0, 0), (-1, -1), 0.5, C_BORDER),
        ('VALIGN',     (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING',  (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING',   (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING',(0, 0), (-1, -1), 5),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, HexColor('#f1f5f9')]),
    ]))
    tw, th = t.wrapOn(c, W - 60, H - 100)
    t.drawOn(c, 30, H - 60 - th)

    # ── Endpoints block ───────────────────────────────────────
    ey = H - 80 - th
    _text(c, 30, ey, 'Network endpoints',
          font='Helvetica-Bold', size=12, color=C_M2)
    ey -= 18
    end_lines = [
        ('Machine 2 (apps)', None),
        ('  8082 main server', 'http://M2:8082'),
        ('  8083 classifier',  'http://localhost:8083  (8082 → 8083 loopback)'),
        ('  GLM-OCR',          'http://10.20.10.2:8001/api/ocr   (VPN local)'),
        ('  GLM-OCR (fallback)', 'http://34.171.200.116/api/ocr   (GCP)'),
        ('', ''),
        ('Machine 1 (heavy LLMs)', None),
        ('  Qwen-VL 72B',  'http://M1:.../vllm/v1/chat/completions'),
        ('  Qwen-LLM 72B', 'http://M1:.../v1/chat/completions'),
    ]
    for lbl, val in end_lines:
        if val is None:
            _text(c, 30, ey, lbl,
                  font='Helvetica-Bold', size=9.5, color=C_M2)
        else:
            _text(c, 30, ey, lbl,
                  font='Helvetica', size=9, color=C_TEXT)
            _text(c, 170, ey, val,
                  font='Courier', size=8.5, color=C_MUTED)
        ey -= 14

    # ── OCR log tags box ──────────────────────────────────────
    ey -= 10
    _text(c, 30, ey, 'Per-page OCR log tags (8082 live processing panel)',
          font='Helvetica-Bold', size=12, color=C_M2)
    ey -= 16
    tag_rows = [
        ('(no tag)',                              'GLM was clean — used as-is'),
        ('[text reader lines stripped]',          'GLM had prompt-template lines mixed in; stripped, rest kept'),
        ('[text reader rescue ✓]',                'GLM hallucinated → Qwen-VL re-OCRed from image; recovered text'),
        ('[text reader blank]',                   'GLM produced minimal output; page treated as blank'),
        ('[text reader+trade expert blank]',      'Both GLM and Qwen-VL agree the page is blank'),
    ]
    for tag, meaning in tag_rows:
        _text(c, 40, ey, tag,
              font='Courier-Bold', size=8.5, color=C_8083)
        _text(c, 240, ey, meaning,
              font='Helvetica', size=8.5, color=C_TEXT)
        ey -= 13


def draw_flow_page(c, W, H):
    """Page 3 — end-to-end request flow as a numbered list."""
    c.setFillColor(C_BG); c.rect(0, 0, W, H, stroke=0, fill=1)
    c.setFillColor(C_M2); c.rect(0, H - 40, W, 40, stroke=0, fill=1)
    _text(c, 30, H - 25, 'End-to-End Request Flow',
          font='Helvetica-Bold', size=14, color=white)
    _text(c, W - 30, H - 25, 'page 3/3',
          font='Helvetica', size=9, color=HexColor('#cfdcf3'), anchor='end')

    y = H - 65
    sections = [
        ('Phase 1 — Upload & Classification', C_M2, [
            ('1', 'User uploads PDF', 'Browser → 8082  (HTTPS, POST /api/upload)'),
            ('2', '8082 generates job_id', 'stores source.pdf in results/<job_id>/'),
            ('3', '8082 calls classifier', '8082 → 8083  (HTTP loopback, /classify, same job_id)'),
            ('4', '8083 runs OCR in parallel', '8083 → GLM-OCR  (HTTP loopback, /api/ocr per page)'),
            ('5', 'Rescue when GLM fails', '8083 → Qwen-VL 72B  (LAN, image + transcribe prompt)'),
            ('6', '8083 classifies each page', '8083 → Qwen-VL 72B  (LAN, image + GLM text)'),
            ('7', 'SWIFT + LC requirements', '8083 parses :20:/:31C:/:46A:/:47A: regex on cleaned text'),
            ('8', 'Deep field extract', '8083 → Qwen-VL 72B  (LAN, per packet, fields + stamps)'),
            ('9', 'Match shipping docs ↔ LC', '8083 → Qwen-VL 72B  (LAN, doc page vs. LC requirement)'),
            ('10', 'Classifier returns to main', '8083 → 8082  (classification.json: pages, logical_docs, fields)'),
            ('11', 'Adapter → step01..step09', '8082 in-process: bridge/adapter.py converts to legacy shape'),
            ('12', 'Final LC consolidation', '8082 step06 (regex on OCR text → consolidated_fields, clauses)'),
            ('13', 'Clause + requirement extract', '8082 step07 (LC requirements + conditions parsed from F46A/F47A)'),
        ]),
        ('Phase 2 — Verification (on user request)', C_LLM, [
            ('14', 'User clicks Verify', 'Browser → 8082  (POST /api/verify/{job_id}/{lc_number})'),
            ('15', 'Decompose clauses', '8082 step12 → Qwen-LLM 72B  (LAN, ~24 parallel workers)'),
            ('16', 'Build verification rows', '8082 step13 in-process (rows = clause × target doc)'),
            ('17', 'Verify each row', '8082 step14 → Qwen-LLM 72B  (LAN, parallel, doc text + audit header)'),
            ('18', 'Confidence + cross-clause', '8082 step15-17 in-process (overrides, escalations, dependencies)'),
            ('19', 'Consolidate', '8082 step18-19 in-process (final result rows)'),
            ('20', 'Report PDF', '8082 step20 generates compliance_report.pdf'),
            ('21', 'Browser displays result', '8082 → Browser  (JSON results + report download URL)'),
        ]),
    ]
    for title, color, steps in sections:
        _text(c, 30, y, title,
              font='Helvetica-Bold', size=12, color=color)
        y -= 18
        for n, what, how in steps:
            # number bubble
            c.setFillColor(color)
            c.circle(42, y + 4, 9, stroke=0, fill=1)
            _text(c, 42, y + 1, n, font='Helvetica-Bold',
                  size=8.5, color=white, anchor='middle')
            _text(c, 60, y + 4, what,
                  font='Helvetica-Bold', size=9, color=C_TEXT)
            _text(c, 60, y - 7, how,
                  font='Helvetica', size=8.5, color=C_MUTED)
            y -= 24
        y -= 6


def main():
    c = _canvas.Canvas(OUT, pagesize=landscape(A4))
    W, H = landscape(A4)
    # Page 1
    draw_diagram_page(c, W, H)
    c.showPage()
    # Page 2
    draw_table_page(c, W, H)
    c.showPage()
    # Page 3
    draw_flow_page(c, W, H)
    c.showPage()
    c.save()
    print(f'Wrote {OUT}  ({os.path.getsize(OUT)} bytes)')


if __name__ == '__main__':
    main()
