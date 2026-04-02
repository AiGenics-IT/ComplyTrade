"""
Step 20 -- Final Compliance Report Generation (PDF)
=====================================================
Takes consolidated output from Step 19, Final LC data, and generates
a professional PDF compliance report using ReportLab.

PURPOSE:
    This is the final deliverable of the pipeline -- a professional PDF report
    that bank trade finance operations (TFO) can use to make payment decisions.
    The report format follows industry standards used by Pakistani banks for
    documentary credit examination.

REPORT STRUCTURE:
    1. Cover Page -- LC number, date, applicant, beneficiary, amount, overall decision
       with color-coded decision banner (green=COMPLIANT, red=DISCREPANT, amber=REVIEW)
    2. Executive Summary -- Critical findings (FAILs) + review items in tables
    3. Section Tables -- Clause-by-clause verification organized by section:
       - Key Terms Verification
       - Document Requirements (F46A)
       - Additional Conditions (F47A)
       - Description of Goods (F45A)
       - Instructions (F78)
    4. Footer with generation timestamp, page numbers, and AiGenics watermark

REPORT TABLE FORMAT (5 columns, matching industry standard):
    | Condition(s) | Found Text | Document Checked | Result | Compliance |

    Each clause gets its own table with a sub-header showing clause ref and overall result.
    A result row at the bottom shows "Result: COMPLIED" (green) or "Result: NOT COMPLIED" (red).

INPUTS:
    - Step 19 consolidated output (sections, clause groups, critical findings)
    - Final LC consolidated fields (for cover page LC details)

OUTPUTS:
    - PDF file path (e.g., ComplyTrade_Report_0491ILC081972_20260331_143000.pdf)

AI MODEL: None -- PDF generation only using ReportLab library.
"""

import os
import sys as _sys; _sys.stdout.reconfigure(encoding="utf-8", errors="replace") if hasattr(_sys.stdout, "reconfigure") else None
import json
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from pathlib import Path
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm, inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer,
    PageBreak, KeepTogether, HRFlowable,
)
from reportlab.platypus.flowables import Flowable


# ── Brand Colors ─────────────────────────────────────────────────────────────
# ComplyTrade brand palette -- matches the old verification.py ComplianceReportPDF.

BRAND_BLUE  = colors.HexColor('#1554ca')   # Primary brand blue
NAVY        = colors.HexColor('#1a3a5c')   # C_NAVY -- section headers, borders
DARK_NAVY   = colors.HexColor('#0d3b8a')   # Dark blue -- emphasis
LIGHT_BLUE  = colors.HexColor('#E8EDF5')   # Result row background
ACCENT_BLUE = colors.HexColor('#3B6CB4')   # Review items header
GREEN       = colors.HexColor('#16a34a')   # PASS / COMPLIED indicators
RED         = colors.HexColor('#dc2626')   # FAIL / NOT COMPLIED / DISCREPANT indicators
AMBER       = colors.HexColor('#d97706')   # REVIEW / REVIEW REQUIRED indicators (orange)
WHITE       = colors.white
LIGHT_GRAY  = colors.HexColor('#f3f4f6')   # Alternating row background
LIGHT_GREEN = colors.HexColor('#dcfce7')   # Pass background tint
LIGHT_RED   = colors.HexColor('#fee2e2')   # Fail background tint
LIGHT_ORANGE= colors.HexColor('#fef3c7')   # Warning background tint
WHITE_SMOKE = colors.HexColor('#f8f9fa')   # Subtle alt row background
MID_GRAY    = colors.HexColor('#dee2e6')   # Grid lines / border gray
DARK_GRAY   = colors.HexColor('#6b7280')   # Footer text, secondary text
C_ROW_ALT   = colors.HexColor('#f8fafc')   # Alternating row (very light blue-gray)


# ── Custom Styles ────────────────────────────────────────────────────────────

def _build_styles():
    """
    Build paragraph styles for the report.

    Defines styles for cover page elements, section headers, cell text,
    and footer -- all using the ComplyTrade brand colors matching the old system.
    """
    styles = getSampleStyleSheet()

    # Cover page title -- large, centered, brand blue
    styles.add(ParagraphStyle(
        'CoverTitle',
        parent=styles['Title'],
        fontSize=38, leading=46,
        textColor=BRAND_BLUE,
        spaceAfter=4,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold',
    ))
    # Cover page subtitle
    styles.add(ParagraphStyle(
        'CoverSubtitle',
        parent=styles['Title'],
        fontSize=20, leading=26,
        textColor=DARK_NAVY,
        alignment=TA_CENTER,
        spaceAfter=4,
        fontName='Helvetica',
    ))
    # Cover page decision banner (COMPLIANT / DISCREPANT / REVIEW REQUIRED)
    styles.add(ParagraphStyle(
        'CoverDecision',
        parent=styles['Normal'],
        fontSize=20,
        alignment=TA_CENTER,
        spaceAfter=8 * mm,
        spaceBefore=8 * mm,
    ))
    # Section headers -- brand blue, bold
    styles.add(ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading1'],
        fontSize=17, leading=22,
        textColor=BRAND_BLUE,
        fontName='Helvetica-Bold',
        spaceBefore=16, spaceAfter=10,
        borderPadding=(0, 0, 2, 0),
    ))
    # Clause sub-headers
    styles.add(ParagraphStyle(
        'SubHeader',
        parent=styles['Heading2'],
        fontSize=13, leading=17,
        textColor=DARK_NAVY,
        fontName='Helvetica-Bold',
        spaceBefore=12, spaceAfter=7,
    ))
    # Table cell text -- standard size
    styles.add(ParagraphStyle(
        'CellText',
        parent=styles['Normal'],
        fontSize=9, leading=12,
        fontName='Helvetica',
        wordWrap='CJK',
    ))
    # Table cell text -- bold variant
    styles.add(ParagraphStyle(
        'CellTextBold',
        parent=styles['Normal'],
        fontSize=9, leading=12,
        fontName='Helvetica-Bold',
    ))
    # Table cell header -- white text for navy header rows
    styles.add(ParagraphStyle(
        'CellTextHeader',
        parent=styles['Normal'],
        fontSize=9, leading=12,
        fontName='Helvetica-Bold',
        textColor=WHITE,
    ))
    # Small text header -- white, for tighter tables
    styles.add(ParagraphStyle(
        'SmallTextHeader',
        parent=styles['Normal'],
        fontSize=8, leading=10.5,
        fontName='Helvetica-Bold',
        textColor=WHITE,
    ))
    # Table cell text -- smaller variant for dense data
    styles.add(ParagraphStyle(
        'CellTextSmall',
        parent=styles['Normal'],
        fontSize=8, leading=10.5,
        textColor=DARK_GRAY,
        fontName='Helvetica',
        wordWrap='CJK',
    ))
    # Body text
    styles.add(ParagraphStyle(
        'BodyText14',
        parent=styles['Normal'],
        fontSize=10, leading=14,
        fontName='Helvetica',
    ))
    # Finding items in executive summary
    styles.add(ParagraphStyle(
        'FindingItem',
        parent=styles['Normal'],
        fontSize=9.5, leading=14,
        fontName='Helvetica',
        leftIndent=8, spaceBefore=3, spaceAfter=3,
    ))
    # Footer text
    styles.add(ParagraphStyle(
        'FooterText',
        parent=styles['Normal'],
        fontSize=7,
        textColor=DARK_GRAY,
        alignment=TA_CENTER,
    ))

    return styles


# ── Helper Functions ─────────────────────────────────────────────────────────

def _safe_str(val, max_len=300) -> str:
    """Safely convert a value to string, truncating if needed to prevent table overflow."""
    if val is None:
        return ''
    s = str(val).strip()
    if len(s) > max_len:
        return s[:max_len] + '...'
    return s


def _result_color(result: str) -> colors.Color:
    """Get the appropriate color for a result value (green/red/amber)."""
    r = result.upper().strip()
    if r in ('PASS', 'COMPLIED'):
        return GREEN
    elif r in ('FAIL', 'NOT COMPLIED'):
        return RED
    else:
        return AMBER   # REVIEW, REVIEW REQUIRED, etc.


def _result_badge(result: str, compliance: str, styles) -> Paragraph:
    """Create a colored compliance badge as a Paragraph for table cells."""
    c = _result_color(compliance)
    hex_color = c.hexval() if hasattr(c, 'hexval') else '#616161'
    return Paragraph(
        f'<font color="{hex_color}"><b>{_safe_str(compliance, 20)}</b></font>',
        styles['CellText'],
    )


def _decision_color(decision: str) -> colors.Color:
    """Get color for the overall compliance decision on the cover page."""
    d = decision.upper()
    if 'COMPLIANT' in d and 'DISCREPANT' not in d:
        return GREEN    # Fully compliant
    elif 'DISCREPANT' in d:
        return RED      # Has failures -- do not pay without resolution
    else:
        return AMBER    # Needs review -- escalate to senior officer


# ── Cover Page ───────────────────────────────────────────────────────────────

def _build_cover(lc_fields: Dict, decision: str, stats: Dict, styles) -> List:
    """
    Build cover page flowables matching the old ComplianceReportPDF styling.

    The cover page shows:
    - ComplyTrade branding with decorative lines
    - LETTER OF CREDIT COMPLIANCE REPORT badge
    - LC Number display with underline
    - Overall compliance decision (large, color-coded banner)
    - Key LC details table
    - Verification summary counts
    """
    elements = []
    page_width = A4[0] - 44 * mm

    # Top decorative line
    elements.append(Spacer(1, 50 * mm))
    elements.append(HRFlowable(
        width='80%', thickness=3,
        color=BRAND_BLUE, spaceAfter=12,
    ))
    elements.append(Spacer(1, 6 * mm))

    elements.append(Paragraph('ComplyTrade', styles['CoverTitle']))
    elements.append(Spacer(1, 3 * mm))
    elements.append(Paragraph('Trade Finance Compliance Platform', ParagraphStyle(
        'CoverTagline', parent=styles['Normal'],
        fontSize=12, leading=16, textColor=DARK_GRAY,
        alignment=TA_CENTER, fontName='Helvetica-Oblique',
    )))
    elements.append(Spacer(1, 14 * mm))

    # Report type badge -- navy background, white text
    badge_data = [[Paragraph(
        '<b>LETTER OF CREDIT COMPLIANCE REPORT</b>',
        ParagraphStyle('BadgeText', parent=styles['Normal'],
                       fontSize=13, leading=18, textColor=WHITE,
                       alignment=TA_CENTER, fontName='Helvetica-Bold'),
    )]]
    badge_table = Table(badge_data, colWidths=[130 * mm], rowHeights=[14 * mm])
    badge_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), BRAND_BLUE),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(badge_table)
    elements.append(Spacer(1, 14 * mm))

    # LC Number display
    lc_number = _safe_str(
        lc_fields.get('DC_Number') or lc_fields.get('F20') or lc_fields.get('LC_Number', 'N/A')
    )
    lc_info_data = [
        [Paragraph('<font color="#6b7280">LC Number</font>', ParagraphStyle(
            'LCLabel', parent=styles['Normal'], fontSize=10, leading=14,
            alignment=TA_CENTER, fontName='Helvetica',
        ))],
        [Paragraph(f'<b>{lc_number}</b>', ParagraphStyle(
            'LCValue', parent=styles['Normal'], fontSize=22, leading=28,
            alignment=TA_CENTER, fontName='Helvetica-Bold',
            textColor=DARK_NAVY,
        ))],
    ]
    lc_info_table = Table(lc_info_data, colWidths=[130 * mm])
    lc_info_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LINEBELOW', (0, 1), (0, 1), 1.5, BRAND_BLUE),
    ]))
    elements.append(lc_info_table)
    elements.append(Spacer(1, 10 * mm))

    # Report date
    report_date = datetime.now().strftime('%B %d, %Y')
    elements.append(Paragraph(
        f'<font size="11" color="#6b7280">Report Date: {report_date}</font>',
        ParagraphStyle('DateText', alignment=TA_CENTER, parent=styles['Normal']),
    ))

    # Decision banner -- green COMPLIANT or red DISCREPANT
    elements.append(Spacer(1, 10 * mm))
    dec_color = _decision_color(decision)
    _dec_row = [[Paragraph(
        f'<b><font size="14" color="white">DECISION: {decision}</font></b>',
        ParagraphStyle('DecisionText', parent=styles['Normal'], alignment=TA_CENTER),
    )]]
    _dec_tbl = Table(_dec_row, colWidths=[page_width])
    _dec_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), dec_color),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('TOPPADDING', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
    ]))
    elements.append(_dec_tbl)

    elements.append(Spacer(1, 20 * mm))
    elements.append(HRFlowable(
        width='80%', thickness=1,
        color=MID_GRAY, spaceAfter=8,
    ))
    elements.append(Paragraph(
        '<font size="8" color="#9ca3af">Confidential \u2014 For authorized recipients only</font>',
        ParagraphStyle('ConfText', alignment=TA_CENTER, parent=styles['Normal']),
    ))

    elements.append(PageBreak())
    return elements


# ── Executive Summary ────────────────────────────────────────────────────────

def _build_executive_summary(
    critical_findings: List[Dict],
    review_items: List[Dict],
    styles,
) -> List:
    """
    Build executive summary section matching the old ComplianceReportPDF styling.

    Shows:
    1. Decision box (green/red full-width banner)
    2. Critical Findings (red) -- numbered list with LC clause references
    3. Review Required (orange) -- numbered list
    4. Summary counts at bottom
    """
    elements = []
    page_width = A4[0] - 44 * mm

    elements.append(Spacer(1, 6 * mm))
    elements.append(Paragraph("Executive Summary", styles['SectionHeader']))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=BRAND_BLUE, spaceAfter=8))

    # Decision box -- full-width colored banner
    _disc_count = len(critical_findings)
    _status_text = 'DOCUMENTS DISCREPANT' if _disc_count > 0 else 'DOCUMENTS COMPLIANT'
    _status_color = RED if _disc_count > 0 else GREEN

    _dec_row = [[Paragraph(
        f'<b><font size="14" color="white">DECISION: {_status_text}</font></b>',
        ParagraphStyle('DecisionText', parent=styles['Normal'], alignment=TA_CENTER),
    )]]
    _dec_tbl = Table(_dec_row, colWidths=[page_width])
    _dec_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), _status_color),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('TOPPADDING', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
    ]))
    elements.append(_dec_tbl)
    elements.append(Spacer(1, 6 * mm))

    def _esc(t):
        return str(t).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;') if t else ''

    # ── Critical Findings (red) ──
    if critical_findings:
        elements.append(Paragraph(
            '<b><font color="#C00000">Critical Findings:</font></b>', styles['Normal']))
        elements.append(Spacer(1, 2 * mm))
        for i, cf_item in enumerate(critical_findings[:30], 1):
            detail = _esc(_safe_str(cf_item.get('condition', '') or cf_item.get('findings', cf_item.get('found_text', '')), 200))
            ref = cf_item.get('clause_ref', '')
            if ref:
                detail += f' <font color="#666666">[{_esc(ref)}]</font>'
            elements.append(Paragraph(
                f'<font size="9">{i}. {detail}</font>',
                styles['CellTextSmall']))
        elements.append(Spacer(1, 5 * mm))
    else:
        elements.append(Paragraph(
            '<font color="#16a34a"><b>No critical findings.</b></font>',
            styles['Normal'],
        ))
        elements.append(Spacer(1, 5 * mm))

    # ── Review Required (orange) ──
    if review_items:
        elements.append(Paragraph(
            '<b><font color="#C55A11">Review Required:</font></b>', styles['Normal']))
        elements.append(Spacer(1, 2 * mm))
        for i, ri in enumerate(review_items[:20], 1):
            detail = _esc(_safe_str(ri.get('condition', ''), 200))
            ref = ri.get('clause_ref', '')
            if ref:
                detail += f' <font color="#666666">[{_esc(ref)}]</font>'
            elements.append(Paragraph(
                f'<font size="9" color="#C55A11">{i}. {detail}</font>',
                styles['CellTextSmall']))
        elements.append(Spacer(1, 5 * mm))

    # Summary line
    elements.append(Spacer(1, 4 * mm))
    elements.append(Paragraph(
        f'<font size="8" color="#666666">Critical Findings: {len(critical_findings)}'
        f', Review Required: {len(review_items)}</font>',
        styles['CellTextSmall']))

    elements.append(PageBreak())
    return elements


# ── Clause-by-Clause Tables ─────────────────────────────────────────────────

def _build_section_tables(sections: List[Dict], styles) -> List:
    """
    Build clause-by-clause verification tables matching the old ComplianceReportPDF.

    Each clause gets:
    - LAYER 1: Navy field label bar (LC Field + Field Description)
    - LAYER 2: Light blue field value bar
    - LAYER 3: 5-column table (Condition | Found Text | Document Checked | Result | Compliance)
    - LAYER 4: Result bar (green "Result: Complied" or red "Result: Not Complied")
    """
    elements = []
    page_width = A4[0] - 44 * mm

    # Colors matching the old system's clause-by-clause section
    _NAVY_C     = colors.HexColor('#1F3864')
    _TEAL_C     = colors.HexColor('#1F4E79')
    _FIELD_BG_C = colors.HexColor('#EBF5FD')
    _ODD_BG_C   = colors.HexColor('#EBF5EB')
    _DISC_BG_C  = colors.HexColor('#FCE4D6')
    _RESULT_OK_BG = colors.HexColor('#E2EFDA')
    _RESULT_NG_BG = colors.HexColor('#FCE4D6')
    _RESULT_OK_FG = colors.HexColor('#375623')
    _RESULT_NG_FG = colors.HexColor('#C00000')
    _AMBER_BG   = colors.HexColor('#FFF9E6')

    _cw = [page_width * 0.28, page_width * 0.14, page_width * 0.14, page_width * 0.30, page_width * 0.14]

    def _esc(t):
        return str(t).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;') if t else ''

    def _sym_text(comp):
        c = str(comp).lower().strip()
        if c in ('pass', 'complied'):
            return 'Pass', '#375623'
        if c in ('fail', 'not complied', 'false', 'non_compliant'):
            return 'Fail', '#C00000'
        if c in ('review', 'warning', 'review_required', 'review required'):
            return 'Review', '#C55A11'
        return 'N/A', '#777777'

    for section in sections:
        section_name = section.get('section_name', 'Unknown')

        # Section header
        elements.append(Spacer(1, 4 * mm))
        elements.append(Paragraph(
            f'<b><font color="#1F3864" size="11">{_esc(section_name)}</font></b>',
            styles['Normal']))
        elements.append(Spacer(1, 2 * mm))

        clauses = section.get('clauses', [])
        for clause in clauses:
            clause_ref = clause.get('clause_ref', '')
            clause_text = _safe_str(clause.get('clause_text', ''), 500)
            overall = clause.get('overall_result', 'REVIEW REQUIRED')
            overall_lower = overall.lower().strip()

            # LAYER 1: Field label bar (navy)
            _l1 = [[
                Paragraph(f'<b><font color="white">LC Field: {_esc(clause_ref)}</font></b>',
                          styles['CellTextSmall']),
                Paragraph(f'<font color="white">Field Description: {_esc(clause_text[:200])}</font>',
                          styles['CellTextSmall']),
            ]]
            _t1 = Table(_l1, colWidths=[page_width * 0.25, page_width * 0.75])
            _t1.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), _NAVY_C),
                ('GRID', (0, 0), (-1, 0), 0.3, colors.HexColor('#888888')),
                ('TOPPADDING', (0, 0), (-1, 0), 5),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 5),
                ('LEFTPADDING', (0, 0), (-1, 0), 6),
                ('VALIGN', (0, 0), (-1, 0), 'MIDDLE'),
            ]))
            elements.append(_t1)

            # LAYER 2: Field value bar (light blue)
            if clause_text:
                _l2 = [[Paragraph(
                    f'<font size="8"><b>Field Value:</b> {_esc(clause_text)}</font>',
                    styles['CellTextSmall'])]]
                _t2 = Table(_l2, colWidths=[page_width])
                _t2.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), _FIELD_BG_C),
                    ('TEXTCOLOR', (0, 0), (-1, 0), _NAVY_C),
                    ('GRID', (0, 0), (-1, 0), 0.3, colors.HexColor('#AABBCC')),
                    ('TOPPADDING', (0, 0), (-1, 0), 3),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 3),
                    ('LEFTPADDING', (0, 0), (-1, 0), 6),
                ]))
                elements.append(_t2)

            # LAYER 3: 5-column verification table
            rows_data = clause.get('rows', [])
            if not rows_data:
                elements.append(Spacer(1, 4 * mm))
                continue

            # Column headers (teal background, white text)
            _hdr = [Paragraph(f'<b><font color="white">{h}</font></b>', styles['CellTextSmall'])
                    for h in ['Condition(s)', 'Findings', 'Document Checked', 'Result', 'Compliance']]
            _tbl_data = [_hdr]
            _tbl_styles = [
                ('BACKGROUND', (0, 0), (-1, 0), _TEAL_C),
                ('GRID', (0, 0), (-1, -1), 0.3, colors.HexColor('#AAAAAA')),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ('LEFTPADDING', (0, 0), (-1, -1), 4),
                ('RIGHTPADDING', (0, 0), (-1, -1), 3),
                ('ALIGN', (4, 0), (4, -1), 'CENTER'),
            ]

            for ri, row in enumerate(rows_data):
                compliance_val = str(row.get('compliance', '')).lower()
                result_val = _safe_str(row.get('result', ''), 200)
                _is_fail = compliance_val in ('fail', 'false', 'not complied', 'non_compliant')
                _is_rev = compliance_val in ('review', 'warning', 'review_required', 'review required')
                _row_bg = (_DISC_BG_C if _is_fail else
                           _AMBER_BG if _is_rev else
                           (_ODD_BG_C if ri % 2 == 0 else WHITE))
                _s, _sc = _sym_text(row.get('compliance', 'na'))

                _tbl_data.append([
                    Paragraph(f'<font size="7.5">{_esc(_safe_str(row.get("condition", ""), 250))}</font>',
                              styles['CellTextSmall']),
                    Paragraph(f'<font size="7"><i>{_esc(_safe_str(row.get("findings", row.get("found_text", "")), 120))}</i></font>',
                              styles['CellTextSmall']),
                    Paragraph(f'<font size="7">{_esc(_safe_str(row.get("document_checked", ""), 80))}</font>',
                              styles['CellTextSmall']),
                    Paragraph(f'<font size="7.5">{_esc(result_val)}</font>',
                              styles['CellTextSmall']),
                    Paragraph(f'<b><font size="8" color="{_sc}">{_s}</font></b>',
                              styles['CellTextSmall']),
                ])
                _ri_tbl = len(_tbl_data) - 1
                _tbl_styles.append(('BACKGROUND', (0, _ri_tbl), (-1, _ri_tbl), _row_bg))

            _ct = Table(_tbl_data, colWidths=_cw, repeatRows=1)
            _ct.setStyle(TableStyle(_tbl_styles))
            elements.append(_ct)

            # LAYER 4: Result bar
            _is_complied = overall_lower in ('complied', 'pass', 'compliant')
            _is_review = overall_lower in ('review', 'review required', 'review_required')
            _r_bg = _RESULT_OK_BG if _is_complied else _RESULT_NG_BG
            _r_fg = _RESULT_OK_FG if _is_complied else _RESULT_NG_FG
            _r_label = ('Result: Complied' if _is_complied else
                        'Result: Review Required' if _is_review else
                        'Result: Not Complied')
            _r_sym = 'PASS' if _is_complied else ('REVIEW' if _is_review else 'FAIL')
            _r_fg_h = f"#{_r_fg.hexval()[2:]}" if hasattr(_r_fg, 'hexval') else '#375623'

            _rr = [[
                Paragraph(f'<b><font color="{_r_fg_h}">{_r_label}</font></b>',
                          styles['CellTextSmall']),
                '', '', '',
                Paragraph(f'<b><font color="{_r_fg_h}" size="10">{_r_sym}</font></b>',
                          styles['CellTextSmall']),
            ]]
            _rt = Table(_rr, colWidths=_cw)
            _rt.setStyle(TableStyle([
                ('SPAN', (0, 0), (3, 0)),
                ('BACKGROUND', (0, 0), (-1, 0), _r_bg),
                ('ALIGN', (4, 0), (4, 0), 'CENTER'),
                ('VALIGN', (0, 0), (-1, 0), 'MIDDLE'),
                ('TOPPADDING', (0, 0), (-1, 0), 5),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 5),
                ('LEFTPADDING', (0, 0), (-1, 0), 6),
                ('GRID', (0, 0), (-1, 0), 0.4, colors.HexColor('#AAAAAA')),
            ]))
            elements.append(_rt)
            elements.append(Spacer(1, 4 * mm))

    return elements


# ── Footer ───────────────────────────────────────────────────────────────────

def _footer_callback(canvas, doc):
    """
    Draw footer, AiGenics watermark, and logo on each page.

    Called by ReportLab for every page during PDF generation.
    Matches the old ComplianceReportPDF styling:
    - Diagonal "AiGenics" watermark (30% opacity)
    - Footer bar: "ComplyTrade | AiGenics | Generated: timestamp | Page N"
    - AiGenics logo in top-right corner
    """
    canvas.saveState()
    page_w, page_h = A4

    # ── Diagonal watermark (30% opacity) ──
    canvas.setFont('Helvetica-Bold', 60)
    canvas.setFillColor(colors.Color(0.85, 0.85, 0.85, alpha=0.3))
    canvas.translate(page_w / 2, page_h / 2)
    canvas.rotate(45)
    canvas.drawCentredString(0, 0, 'AiGenics')
    canvas.rotate(-45)
    canvas.translate(-page_w / 2, -page_h / 2)

    # ── Footer ──
    canvas.setFont('Helvetica', 7)
    canvas.setFillColor(DARK_GRAY)
    y = 12 * mm
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    canvas.drawCentredString(
        page_w / 2, y,
        f'ComplyTrade | AiGenics  |  Generated: {timestamp}  |  Page {doc.page}',
    )

    # ── AiGenics logo in top-right corner ──
    try:
        logo_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'view', 'logo.png')
        if os.path.exists(logo_path):
            canvas.drawImage(logo_path, page_w - 45 * mm, page_h - 15 * mm,
                             width=30 * mm, height=10 * mm, preserveAspectRatio=True, mask='auto')
    except Exception:
        pass  # Logo is optional -- continue without it

    canvas.restoreState()


# ── PDF Builder ──────────────────────────────────────────────────────────────

def generate_pdf(
    consolidated: Dict,
    lc_fields: Dict,
    output_path: str,
    progress_fn=None,
) -> str:
    """
    Generate the final compliance report PDF.

    Assembles all report sections (cover, executive summary, clause tables)
    into a ReportLab document and builds the PDF file.

    Args:
        consolidated: Step 19 output dict (sections, findings, decision)
        lc_fields:    Final LC consolidated fields (for cover page)
        output_path:  path for the output PDF file
        progress_fn:  callback for progress messages

    Returns:
        absolute path to generated PDF
    """
    if progress_fn is None:
        def progress_fn(msg): pass

    styles = _build_styles()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Configure the PDF document with A4 page size and margins (matching old system)
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        topMargin=22 * mm,
        bottomMargin=22 * mm,
        leftMargin=22 * mm,
        rightMargin=22 * mm,
        title='ComplyTrade Compliance Report',
        author='ComplyTrade',
    )

    decision = consolidated.get('overall_decision', 'REVIEW REQUIRED')
    stats = {
        'total_pass': consolidated.get('total_pass', 0),
        'total_fail': consolidated.get('total_fail', 0),
        'total_review': consolidated.get('total_review', 0),
    }

    elements = []

    # 1. Cover page -- LC details and overall decision
    progress_fn("Building cover page...")
    elements.extend(_build_cover(lc_fields, decision, stats, styles))

    # 2. Executive Summary -- critical findings and review items
    progress_fn("Building executive summary...")
    elements.extend(_build_executive_summary(
        consolidated.get('critical_findings', []),
        consolidated.get('review_items', []),
        styles,
    ))

    # 3-4. Section tables -- clause-by-clause verification details
    progress_fn("Building clause-by-clause tables...")
    sections = consolidated.get('sections', [])
    elements.extend(_build_section_tables(sections, styles))

    # Build the PDF -- _footer_callback draws watermark and footer on every page
    progress_fn("Rendering PDF...")
    doc.build(elements, onFirstPage=_footer_callback, onLaterPages=_footer_callback)

    abs_path = str(Path(output_path).resolve())
    progress_fn(f"PDF generated: {abs_path}")

    return abs_path


# ── Runner ───────────────────────────────────────────────────────────────────

def run(
    consolidated: Dict,
    lc_fields: Dict,
    output_dir: str,
    progress_fn=None,
) -> Dict[str, Any]:
    """
    Execute Step 20: Final Compliance Report Generation.

    Generates a professional PDF report from the consolidated verification data.
    The filename includes the LC number and timestamp for easy identification.

    Args:
        consolidated: Step 19 output dict
        lc_fields:    Final LC consolidated fields dict
        output_dir:   directory for step output
        progress_fn:  callback for progress messages

    Returns:
        dict with step results including pdf_path
    """
    t0 = time.time()
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if progress_fn is None:
        def progress_fn(msg): pass

    progress_fn("Step 20: Final Compliance Report Generation")

    # Build filename from LC number -- sanitized for filesystem safety
    lc_number = _safe_str(
        lc_fields.get('DC_Number') or lc_fields.get('F20') or lc_fields.get('LC_Number', 'report'),
        50,
    )
    # Replace non-alphanumeric characters with underscores
    safe_name = ''.join(c if c.isalnum() or c in '-_' else '_' for c in lc_number)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    pdf_filename = f'ComplyTrade_Report_{safe_name}_{timestamp}.pdf'
    pdf_path = os.path.join(output_dir, pdf_filename)

    try:
        abs_path = generate_pdf(consolidated, lc_fields, pdf_path, progress_fn)

        result = {
            'step': 20,
            'step_name': 'Final Compliance Report Generation',
            'pdf_path': abs_path,
            'pdf_filename': pdf_filename,
            'overall_decision': consolidated.get('overall_decision', 'REVIEW REQUIRED'),
            'total_clauses': consolidated.get('total_clauses', 0),
            'total_rows': consolidated.get('total_rows', 0),
            'total_pass': consolidated.get('total_pass', 0),
            'total_fail': consolidated.get('total_fail', 0),
            'total_review': consolidated.get('total_review', 0),
            'elapsed_seconds': round(time.time() - t0, 2),
        }

    except Exception as e:
        import traceback
        result = {
            'step': 20,
            'step_name': 'Final Compliance Report Generation',
            'error': str(e),
            'traceback': traceback.format_exc()[:500],
            'elapsed_seconds': round(time.time() - t0, 2),
        }
        progress_fn(f"Step 20 FAILED: {str(e)}")

    # Save result metadata (separate from the PDF itself)
    meta_path = Path(output_dir) / 'step20_result.json'
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    if 'error' not in result:
        progress_fn(
            f"Step 20 complete: {pdf_filename} -- "
            f"{result['overall_decision']} in {result['elapsed_seconds']:.1f}s"
        )

    return result
