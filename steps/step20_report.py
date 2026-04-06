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
    """Build paragraph styles for the report."""
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        'CoverTitle', parent=styles['Title'],
        fontSize=38, leading=46, textColor=BRAND_BLUE,
        spaceAfter=4, alignment=TA_CENTER, fontName='Helvetica-Bold',
    ))
    styles.add(ParagraphStyle(
        'CoverSubtitle', parent=styles['Title'],
        fontSize=20, leading=26, textColor=DARK_NAVY,
        alignment=TA_CENTER, spaceAfter=4, fontName='Helvetica',
    ))
    styles.add(ParagraphStyle(
        'CoverDecision', parent=styles['Normal'],
        fontSize=20, alignment=TA_CENTER,
        spaceAfter=8 * mm, spaceBefore=8 * mm,
    ))
    styles.add(ParagraphStyle(
        'SectionHeader', parent=styles['Heading1'],
        fontSize=17, leading=22, textColor=BRAND_BLUE,
        fontName='Helvetica-Bold', spaceBefore=16, spaceAfter=10,
        borderPadding=(0, 0, 2, 0),
    ))
    styles.add(ParagraphStyle(
        'SubHeader', parent=styles['Heading2'],
        fontSize=13, leading=17, textColor=DARK_NAVY,
        fontName='Helvetica-Bold', spaceBefore=12, spaceAfter=7,
    ))
    styles.add(ParagraphStyle(
        'CellText', parent=styles['Normal'],
        fontSize=9, leading=12, fontName='Helvetica', wordWrap='CJK',
    ))
    styles.add(ParagraphStyle(
        'CellTextBold', parent=styles['Normal'],
        fontSize=9, leading=12, fontName='Helvetica-Bold',
    ))
    styles.add(ParagraphStyle(
        'CellTextHeader', parent=styles['Normal'],
        fontSize=9, leading=12, fontName='Helvetica-Bold', textColor=WHITE,
    ))
    styles.add(ParagraphStyle(
        'SmallTextHeader', parent=styles['Normal'],
        fontSize=8, leading=10.5, fontName='Helvetica-Bold', textColor=WHITE,
    ))
    styles.add(ParagraphStyle(
        'CellTextSmall', parent=styles['Normal'],
        fontSize=8, leading=10.5, textColor=DARK_GRAY,
        fontName='Helvetica', wordWrap='CJK',
    ))
    styles.add(ParagraphStyle(
        'BodyText14', parent=styles['Normal'],
        fontSize=10, leading=14, fontName='Helvetica',
    ))
    styles.add(ParagraphStyle(
        'FindingItem', parent=styles['Normal'],
        fontSize=9.5, leading=14, fontName='Helvetica',
        leftIndent=8, spaceBefore=3, spaceAfter=3,
    ))
    styles.add(ParagraphStyle(
        'FooterText', parent=styles['Normal'],
        fontSize=7, textColor=DARK_GRAY, alignment=TA_CENTER,
    ))

    return styles


# ── Helper Functions ─────────────────────────────────────────────────────────

def _safe_str(val, max_len=300) -> str:
    """Safely convert a value to string, truncating if needed."""
    if val is None:
        return ''
    s = str(val).strip()
    if len(s) > max_len:
        return s[:max_len] + '...'
    return s


def _esc(t):
    """Escape HTML entities for ReportLab Paragraph."""
    return str(t).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;') if t else ''


def _result_color(result: str) -> colors.Color:
    r = result.upper().strip()
    if r in ('PASS', 'COMPLIED'):
        return GREEN
    elif r in ('FAIL', 'NOT COMPLIED'):
        return RED
    else:
        return AMBER


def _decision_color(decision: str) -> colors.Color:
    d = decision.upper()
    if 'COMPLIANT' in d and 'DISCREPANT' not in d and 'REVIEW' not in d:
        return GREEN
    elif 'DISCREPANT' in d or 'NOT COMPLIED' in d:
        return RED
    else:
        return AMBER


def _extract_lc_fields(step06_data: Dict) -> Dict:
    """
    Extract LC fields from step06 result structure.

    step06 returns: {
        'final_lc': {'dc_number': '...', 'consolidated_fields': {'20': '...', '31C': '...'}, ...},
        ...
    }
    OR it may already be the consolidated_fields dict directly.
    OR it may be from the saved JSON with top-level keys.

    This function normalizes all formats into a flat dict with useful keys.
    """
    lc = {}

    # Try to get consolidated_fields from nested structure
    cf = {}
    if 'final_lc' in step06_data:
        fl = step06_data['final_lc']
        if isinstance(fl, dict):
            cf = fl.get('consolidated_fields', {})
            lc['dc_number'] = fl.get('dc_number', '')
            lc['swift_format'] = fl.get('swift_format', '')
        elif hasattr(fl, 'consolidated_fields'):
            cf = fl.consolidated_fields
            lc['dc_number'] = getattr(fl, 'dc_number', '')
            lc['swift_format'] = getattr(fl, 'swift_format', '')
    elif 'consolidated_fields' in step06_data:
        cf = step06_data['consolidated_fields']
        lc['dc_number'] = step06_data.get('dc_number', '')
        lc['swift_format'] = step06_data.get('swift_format', '')
    else:
        # Might already be a flat dict with DC_Number, F20 etc
        cf = step06_data

    # Map SWIFT tags to readable fields
    lc['lc_number'] = (
        lc.get('dc_number', '') or
        cf.get('20', '') or cf.get('F20', '') or
        cf.get('DC_Number', '') or cf.get('LC_Number', '') or
        step06_data.get('dc_number', '') or 'N/A'
    )
    lc['date_of_issue'] = cf.get('31C', cf.get('F31C', cf.get('Date_of_Issue', '')))
    lc['expiry_date'] = cf.get('31D', cf.get('F31D', cf.get('Date_Place_Expiry', '')))
    lc['amount'] = cf.get('32B', cf.get('F32B', cf.get('Amount', '')))
    lc['applicant'] = cf.get('50', cf.get('F50', cf.get('Applicant', '')))
    lc['beneficiary'] = cf.get('59', cf.get('F59', cf.get('Beneficiary', '')))
    lc['issuing_bank'] = cf.get('52A', cf.get('F52A', cf.get('Issuing_Bank', '')))
    lc['advising_bank'] = cf.get('57A', cf.get('F57A', cf.get('Advising_Bank', '')))
    lc['available_with'] = cf.get('41A', cf.get('F41A', cf.get('Available_With', '')))
    lc['drafts_at'] = cf.get('42C', cf.get('F42C', cf.get('Drafts_At', '')))
    lc['partial_shipment'] = cf.get('43P', cf.get('F43P', cf.get('Partial_Shipments', '')))
    lc['transhipment'] = cf.get('43T', cf.get('F43T', cf.get('Transhipment', '')))
    lc['port_loading'] = cf.get('44E', cf.get('F44E', cf.get('Port_of_Loading', '')))
    lc['port_discharge'] = cf.get('44F', cf.get('F44F', cf.get('Port_of_Discharge', '')))
    lc['latest_shipment'] = cf.get('44C', cf.get('F44C', cf.get('Latest_Shipment_Date', '')))
    lc['form_of_credit'] = cf.get('40A', cf.get('F40A', cf.get('Form_of_Credit', '')))
    lc['applicable_rules'] = cf.get('40E', cf.get('F40E', cf.get('Applicable_Rules', '')))
    lc['tolerance'] = cf.get('39A', cf.get('F39A', cf.get('Tolerance', '')))
    lc['presentation_period'] = cf.get('48', cf.get('F48', cf.get('Period_for_Presentation', '')))

    return lc


# ── SWIFT Tag Label Map ──────────────────────────────────────────────────────

_TAG_LABELS = {
    '20': 'Documentary Credit Number', 'F20': 'Documentary Credit Number',
    '31C': 'Date of Issue', 'F31C': 'Date of Issue',
    '31D': 'Date and Place of Expiry', 'F31D': 'Date and Place of Expiry',
    '32B': 'Currency Code, Amount', 'F32B': 'Currency Code, Amount',
    '39A': 'Percentage Credit Amount Tolerance', 'F39A': 'Percentage Credit Amount Tolerance',
    '40A': 'Form of Documentary Credit', 'F40A': 'Form of Documentary Credit',
    '40E': 'Applicable Rules', 'F40E': 'Applicable Rules',
    '41A': 'Available With ... By ...', 'F41A': 'Available With ... By ...',
    '42C': 'Drafts At', 'F42C': 'Drafts At',
    '43P': 'Partial Shipments', 'F43P': 'Partial Shipments',
    '43T': 'Transhipment', 'F43T': 'Transhipment',
    '44A': 'Place of Taking in Charge', 'F44A': 'Place of Taking in Charge',
    '44B': 'Place of Final Destination', 'F44B': 'Place of Final Destination',
    '44C': 'Latest Date of Shipment', 'F44C': 'Latest Date of Shipment',
    '44E': 'Port of Loading', 'F44E': 'Port of Loading',
    '44F': 'Port of Discharge', 'F44F': 'Port of Discharge',
    '48': 'Period for Presentation', 'F48': 'Period for Presentation',
    '49': 'Confirmation Instructions', 'F49': 'Confirmation Instructions',
    '50': 'Applicant', 'F50': 'Applicant',
    '51A': 'Applicant Bank', 'F51A': 'Applicant Bank',
    '52A': 'Issuing Bank', 'F52A': 'Issuing Bank',
    '53A': 'Reimbursing Bank', 'F53A': 'Reimbursing Bank',
    '57A': 'Advise Through Bank', 'F57A': 'Advise Through Bank',
    '59': 'Beneficiary', 'F59': 'Beneficiary',
    '71B': 'Charges', 'F71B': 'Charges',
    '71D': 'Charges Details', 'F71D': 'Charges Details',
    '77B': 'Regulatory Reporting', 'F77B': 'Regulatory Reporting',
    '45A': 'Description of Goods and/or Services', 'F45A': 'Description of Goods and/or Services',
    '46A': 'Documents Required', 'F46A': 'Documents Required',
    '47A': 'Additional Conditions', 'F47A': 'Additional Conditions',
    '78': 'Instructions to Paying/Accepting/Negotiating Bank', 'F78': 'Instructions to Paying/Accepting/Negotiating Bank',
    '72': 'Sender to Receiver Information', 'F72': 'Sender to Receiver Information',
    '79': 'Narrative', 'F79': 'Narrative',
}

def _get_field_description(clause_ref: str) -> str:
    """Get human-readable field description from clause ref like '46A-1' or 'F46A-1'."""
    tag = clause_ref.split('-')[0].upper()
    # Remove leading F if present for lookup
    tag_no_f = tag[1:] if tag.startswith('F') else tag
    tag_with_f = 'F' + tag_no_f
    return _TAG_LABELS.get(tag, _TAG_LABELS.get(tag_no_f, _TAG_LABELS.get(tag_with_f, '')))


# ── Cover Page ───────────────────────────────────────────────────────────────

def _build_cover(lc: Dict, decision: str, stats: Dict, styles) -> List:
    """Build cover page with LC details and overall compliance decision."""
    elements = []
    page_width = A4[0] - 44 * mm

    # Top decorative line
    elements.append(Spacer(1, 50 * mm))
    elements.append(HRFlowable(width='80%', thickness=3, color=BRAND_BLUE, spaceAfter=12))
    elements.append(Spacer(1, 6 * mm))

    elements.append(Paragraph('ComplyTrade', styles['CoverTitle']))
    elements.append(Spacer(1, 3 * mm))
    elements.append(Paragraph('Trade Finance Compliance Platform', ParagraphStyle(
        'CoverTagline', parent=styles['Normal'],
        fontSize=12, leading=16, textColor=DARK_GRAY,
        alignment=TA_CENTER, fontName='Helvetica-Oblique',
    )))
    elements.append(Spacer(1, 14 * mm))

    # Report type badge
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
    elements.append(Spacer(1, 10 * mm))

    # LC Number display
    lc_number = lc.get('lc_number', 'N/A') or 'N/A'
    lc_info_data = [
        [Paragraph('<font color="#6b7280">LC Number</font>', ParagraphStyle(
            'LCLabel', parent=styles['Normal'], fontSize=10, leading=14,
            alignment=TA_CENTER, fontName='Helvetica',
        ))],
        [Paragraph(f'<b>{_esc(lc_number)}</b>', ParagraphStyle(
            'LCValue', parent=styles['Normal'], fontSize=22, leading=28,
            alignment=TA_CENTER, fontName='Helvetica-Bold', textColor=DARK_NAVY,
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
    elements.append(Spacer(1, 8 * mm))

    # LC Key Details table
    _detail_rows = []
    _detail_fields = [
        ('Applicant', lc.get('applicant', '')),
        ('Beneficiary', lc.get('beneficiary', '')),
        ('Amount', lc.get('amount', '')),
        ('Date of Issue', lc.get('date_of_issue', '')),
        ('Date / Place of Expiry', lc.get('expiry_date', '')),
        ('Latest Shipment', lc.get('latest_shipment', '')),
        ('Port of Loading', lc.get('port_loading', '')),
        ('Port of Discharge', lc.get('port_discharge', '')),
        ('Applicable Rules', lc.get('applicable_rules', '')),
    ]
    for label, val in _detail_fields:
        val_str = _safe_str(val, 120)
        if val_str:
            _detail_rows.append([
                Paragraph(f'<b><font size="8" color="#444444">{_esc(label)}</font></b>',
                          styles['CellTextSmall']),
                Paragraph(f'<font size="8">{_esc(val_str)}</font>',
                          styles['CellTextSmall']),
            ])

    if _detail_rows:
        _dt = Table(_detail_rows, colWidths=[page_width * 0.35, page_width * 0.65])
        _dt.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 0.3, MID_GRAY),
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#F0F4FA')),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ]))
        elements.append(_dt)
        elements.append(Spacer(1, 6 * mm))

    # Report date
    report_date = datetime.now().strftime('%B %d, %Y')
    elements.append(Paragraph(
        f'<font size="11" color="#6b7280">Report Date: {report_date}</font>',
        ParagraphStyle('DateText', alignment=TA_CENTER, parent=styles['Normal']),
    ))

    # Decision banner
    elements.append(Spacer(1, 8 * mm))
    dec_color = _decision_color(decision)
    _dec_row = [[Paragraph(
        f'<b><font size="14" color="white">DECISION: {_esc(decision)}</font></b>',
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

    # Verification stats
    elements.append(Spacer(1, 6 * mm))
    tp = stats.get('total_pass', 0)
    tf = stats.get('total_fail', 0)
    tr = stats.get('total_review', 0)
    elements.append(Paragraph(
        f'<font size="9" color="#6b7280">'
        f'<font color="#16a34a"><b>{tp}</b></font> Passed  |  '
        f'<font color="#dc2626"><b>{tf}</b></font> Failed  |  '
        f'<font color="#d97706"><b>{tr}</b></font> Review Required'
        f'</font>',
        ParagraphStyle('StatsText', alignment=TA_CENTER, parent=styles['Normal']),
    ))

    elements.append(Spacer(1, 12 * mm))
    elements.append(HRFlowable(width='80%', thickness=1, color=MID_GRAY, spaceAfter=8))
    elements.append(Paragraph(
        '<font size="8" color="#9ca3af">Confidential \u2014 For authorized recipients only</font>',
        ParagraphStyle('ConfText', alignment=TA_CENTER, parent=styles['Normal']),
    ))

    elements.append(PageBreak())
    return elements


# ── Executive Summary ────────────────────────────────────────────────────────

def _build_executive_summary(
    decision: str,
    critical_findings: List[Dict],
    review_items: List[Dict],
    stats: Dict,
    styles,
) -> List:
    """Build executive summary with consistent decision and findings."""
    elements = []
    page_width = A4[0] - 44 * mm

    elements.append(Spacer(1, 6 * mm))
    elements.append(Paragraph("Executive Summary", styles['SectionHeader']))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=BRAND_BLUE, spaceAfter=8))

    # Decision box -- uses the SAME decision as cover page
    _status_color = _decision_color(decision)
    _dec_row = [[Paragraph(
        f'<b><font size="14" color="white">DECISION: {_esc(decision)}</font></b>',
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

    # Stats summary
    tp = stats.get('total_pass', 0)
    tf = stats.get('total_fail', 0)
    tr = stats.get('total_review', 0)
    total = tp + tf + tr
    elements.append(Paragraph(
        f'<font size="10">Total Checks: <b>{total}</b>  |  '
        f'<font color="#16a34a"><b>{tp} Passed</b></font>  |  '
        f'<font color="#dc2626"><b>{tf} Failed</b></font>  |  '
        f'<font color="#d97706"><b>{tr} Review</b></font></font>',
        styles['BodyText14'],
    ))
    elements.append(Spacer(1, 6 * mm))

    # Critical Findings (red)
    if critical_findings:
        elements.append(Paragraph(
            '<b><font color="#C00000">Critical Findings (Discrepancies):</font></b>', styles['Normal']))
        elements.append(Spacer(1, 2 * mm))

        # Build a table for critical findings
        _cf_hdr = [
            Paragraph('<b><font color="white">#</font></b>', styles['SmallTextHeader']),
            Paragraph('<b><font color="white">Clause</font></b>', styles['SmallTextHeader']),
            Paragraph('<b><font color="white">Document</font></b>', styles['SmallTextHeader']),
            Paragraph('<b><font color="white">Finding</font></b>', styles['SmallTextHeader']),
        ]
        _cf_rows = [_cf_hdr]
        for i, cf in enumerate(critical_findings[:30], 1):
            finding = _esc(_safe_str(
                cf.get('result', '') or cf.get('condition', '') or cf.get('findings', ''), 150))
            _cf_rows.append([
                Paragraph(f'<font size="8">{i}</font>', styles['CellTextSmall']),
                Paragraph(f'<font size="8"><b>{_esc(cf.get("clause_ref", ""))}</b></font>', styles['CellTextSmall']),
                Paragraph(f'<font size="8">{_esc(cf.get("document_checked", ""))}</font>', styles['CellTextSmall']),
                Paragraph(f'<font size="8" color="#C00000">{finding}</font>', styles['CellTextSmall']),
            ])

        _cf_tbl = Table(_cf_rows, colWidths=[page_width * 0.06, page_width * 0.14, page_width * 0.20, page_width * 0.60])
        _cf_styles = [
            ('BACKGROUND', (0, 0), (-1, 0), RED),
            ('GRID', (0, 0), (-1, -1), 0.3, MID_GRAY),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ]
        for ri in range(1, len(_cf_rows)):
            _cf_styles.append(('BACKGROUND', (0, ri), (-1, ri),
                               LIGHT_RED if ri % 2 == 1 else WHITE))
        _cf_tbl.setStyle(TableStyle(_cf_styles))
        elements.append(_cf_tbl)
        elements.append(Spacer(1, 5 * mm))
    else:
        if tf == 0 and tr == 0:
            elements.append(Paragraph(
                '<font color="#16a34a"><b>All checks passed. No discrepancies found.</b></font>',
                styles['Normal'],
            ))
        elif tf == 0:
            elements.append(Paragraph(
                '<b>No critical failures.</b> Some items require manual review (see details below).',
                styles['Normal'],
            ))
        elements.append(Spacer(1, 5 * mm))

    # Review Required (orange) -- show items that have real data
    if review_items:
        # Filter out informational-only review items
        _review_display = []
        for r in review_items:
            _c = (r.get('condition', '') or '').strip()
            _f = (r.get('findings', '') or '').strip()
            _r = (r.get('result', '') or '').strip()
            _d = (r.get('document_checked', '') or '').strip()
            # Skip if all fields are empty/N/A/INFORMATIONAL
            if _c.upper() in ('', 'N/A', 'INFORMATIONAL') and _f.upper() in ('', 'N/A') and _r.upper() in ('', 'N/A', 'INFORMATIONAL'):
                continue
            _review_display.append(r)
            if len(_review_display) >= 25:
                break

        if _review_display:
            # Build a table for review items (same format as critical findings)
            elements.append(Paragraph(
                f'<b><font color="#C55A11">Items Requiring Manual Review ({len(_review_display)}):</font></b>', styles['Normal']))
            elements.append(Spacer(1, 2 * mm))

            _rv_hdr = [
                Paragraph('<b><font color="white">#</font></b>', styles['SmallTextHeader']),
                Paragraph('<b><font color="white">Clause</font></b>', styles['SmallTextHeader']),
                Paragraph('<b><font color="white">Document</font></b>', styles['SmallTextHeader']),
                Paragraph('<b><font color="white">Finding</font></b>', styles['SmallTextHeader']),
            ]
            _rv_rows = [_rv_hdr]
            for i, ri in enumerate(_review_display, 1):
                detail = _esc(_safe_str(
                    ri.get('condition', '') or ri.get('result', '') or ri.get('findings', ''), 150))
                _rv_rows.append([
                    Paragraph(f'<font size="8">{i}</font>', styles['CellTextSmall']),
                    Paragraph(f'<font size="8"><b>{_esc(ri.get("clause_ref", ""))}</b></font>', styles['CellTextSmall']),
                    Paragraph(f'<font size="8">{_esc(ri.get("document_checked", ""))}</font>', styles['CellTextSmall']),
                    Paragraph(f'<font size="8" color="#C55A11">{detail}</font>', styles['CellTextSmall']),
                ])

            _rv_tbl = Table(_rv_rows, colWidths=[page_width * 0.06, page_width * 0.14, page_width * 0.20, page_width * 0.60])
            _rv_styles = [
                ('BACKGROUND', (0, 0), (-1, 0), AMBER),
                ('GRID', (0, 0), (-1, -1), 0.3, MID_GRAY),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('TOPPADDING', (0, 0), (-1, -1), 3),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
                ('LEFTPADDING', (0, 0), (-1, -1), 4),
            ]
            for ri in range(1, len(_rv_rows)):
                _rv_styles.append(('BACKGROUND', (0, ri), (-1, ri),
                                   LIGHT_ORANGE if ri % 2 == 1 else WHITE))
            _rv_tbl.setStyle(TableStyle(_rv_styles))
            elements.append(_rv_tbl)
            elements.append(Spacer(1, 5 * mm))

    elements.append(PageBreak())
    return elements


# ── Clause-by-Clause Tables ─────────────────────────────────────────────────

def _is_informational_only(clause: Dict) -> bool:
    """Check if a clause has only N/A / INFORMATIONAL rows (no real verification)."""
    rows = clause.get('rows', [])
    if not rows:
        return True
    for r in rows:
        cond = str(r.get('condition', '')).strip().upper()
        result = str(r.get('result', '')).strip().upper()
        findings = str(r.get('findings', r.get('found_text', ''))).strip()
        doc = str(r.get('document_checked', '')).strip()

        # If result is explicitly INFORMATIONAL → informational
        if result == 'INFORMATIONAL' or cond == 'INFORMATIONAL':
            continue
        # If condition and findings are both empty/N/A → informational
        if (not cond or cond == 'N/A') and (not findings or findings == 'N/A') and (not doc or doc == 'N/A'):
            continue
        # Has real verification data
        return False
    return True


def _build_section_tables(sections: List[Dict], styles) -> List:
    """Build clause-by-clause verification tables."""
    elements = []
    page_width = A4[0] - 44 * mm

    _NAVY_C     = colors.HexColor('#1F3864')
    _TEAL_C     = colors.HexColor('#1F4E79')
    _FIELD_BG_C = colors.HexColor('#EBF5FD')
    _ODD_BG_C   = colors.HexColor('#EBF5EB')
    _DISC_BG_C  = colors.HexColor('#FCE4D6')
    _RESULT_OK_BG = colors.HexColor('#E2EFDA')
    _RESULT_NG_BG = colors.HexColor('#FCE4D6')
    _RESULT_RV_BG = colors.HexColor('#FFF9E6')
    _RESULT_OK_FG = colors.HexColor('#375623')
    _RESULT_NG_FG = colors.HexColor('#C00000')
    _RESULT_RV_FG = colors.HexColor('#C55A11')
    _AMBER_BG   = colors.HexColor('#FFF9E6')

    _cw = [page_width * 0.28, page_width * 0.14, page_width * 0.14, page_width * 0.30, page_width * 0.14]

    def _sym_text(comp):
        c = str(comp).lower().strip()
        if c in ('pass', 'complied'):
            return 'Pass', '#375623'
        if c in ('fail', 'not complied', 'false', 'non_compliant'):
            return 'Fail', '#C00000'
        if c in ('review', 'warning', 'review_required', 'review required'):
            return 'Review', '#C55A11'
        if c in ('info', 'informational'):
            return 'Info', '#777777'
        return 'N/A', '#777777'

    for section in sections:
        section_name = section.get('section_name', 'Unknown')

        if not section.get('clauses', []):
            continue

        # Section header with stats
        sp = section.get('total_pass', 0)
        sf = section.get('total_fail', 0)
        sr_count = section.get('total_review', 0)
        elements.append(Spacer(1, 4 * mm))
        elements.append(Paragraph(
            f'<b><font color="#1F3864" size="13">{_esc(section_name)}</font></b>'
            f'  <font color="#888888" size="8">({sp}P / {sf}F / {sr_count}R)</font>',
            styles['Normal']))
        elements.append(HRFlowable(width="100%", thickness=1, color=BRAND_BLUE, spaceAfter=4))

        for clause in section.get('clauses', []):

            clause_ref = clause.get('clause_ref', '')
            clause_text = _safe_str(clause.get('clause_text', ''), 500)
            overall = clause.get('overall_result', 'REVIEW REQUIRED')
            overall_lower = overall.lower().strip()

            # Get field description — show clause text if available, else generic label
            field_desc = _get_field_description(clause_ref) or ''
            # Use actual clause text for the header description
            header_desc = clause_text[:200] if clause_text else field_desc

            # LAYER 1: Field label bar (navy)
            _l1 = [[
                Paragraph(f'<b><font color="white">LC Field: {_esc(clause_ref)}</font></b>',
                          styles['CellTextSmall']),
                Paragraph(f'<font color="white">{_esc(header_desc)}</font>',
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

            # LAYER 2: Field value bar (light blue) -- show the actual LC clause text
            if clause_text:
                _l2 = [[Paragraph(
                    f'<font size="8"><b>LC Clause:</b> {_esc(clause_text[:400])}</font>',
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
            real_rows = rows_data

            if not real_rows:
                elements.append(Spacer(1, 4 * mm))
                continue

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

            for ri, row in enumerate(real_rows):
                compliance_val = str(row.get('compliance', '')).lower()
                result_val = _safe_str(row.get('result', ''), 200)
                _is_fail = compliance_val in ('fail', 'false', 'not complied', 'non_compliant')
                _is_pass = compliance_val in ('pass', 'complied')
                _is_rev = compliance_val in ('review', 'warning', 'review_required', 'review required')
                _row_bg = (_DISC_BG_C if _is_fail else
                           _ODD_BG_C if _is_pass else
                           _AMBER_BG if _is_rev else
                           (WHITE if ri % 2 == 0 else colors.HexColor('#F8F9FA')))
                _s, _sc = _sym_text(row.get('compliance', 'na'))

                _cond_text = row.get('condition', '') or row.get('condition_text', '') or row.get('result', '')
                _find_text = row.get('findings', '') or row.get('found_text', '')
                _tbl_data.append([
                    Paragraph(f'<font size="7.5">{_esc(_safe_str(_cond_text, 250))}</font>',
                              styles['CellTextSmall']),
                    Paragraph(f'<font size="7"><i>{_esc(_safe_str(_find_text, 120))}</i></font>',
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
            if _is_complied:
                _r_bg, _r_fg = _RESULT_OK_BG, _RESULT_OK_FG
                _r_label, _r_sym = 'Result: Complied', 'PASS'
            elif _is_review:
                _r_bg, _r_fg = _RESULT_RV_BG, _RESULT_RV_FG
                _r_label, _r_sym = 'Result: Review Required', 'REVIEW'
            else:
                _r_bg, _r_fg = _RESULT_NG_BG, _RESULT_NG_FG
                _r_label, _r_sym = 'Result: Not Complied', 'FAIL'

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
    """Draw footer, AiGenics watermark, and logo on each page."""
    canvas.saveState()
    page_w, page_h = A4

    # Diagonal watermark (30% opacity)
    canvas.setFont('Helvetica-Bold', 60)
    canvas.setFillColor(colors.Color(0.85, 0.85, 0.85, alpha=0.3))
    canvas.translate(page_w / 2, page_h / 2)
    canvas.rotate(45)
    canvas.drawCentredString(0, 0, 'AiGenics')
    canvas.rotate(-45)
    canvas.translate(-page_w / 2, -page_h / 2)

    # Footer
    canvas.setFont('Helvetica', 7)
    canvas.setFillColor(DARK_GRAY)
    y = 12 * mm
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    canvas.drawCentredString(
        page_w / 2, y,
        f'ComplyTrade | AiGenics  |  Generated: {timestamp}  |  Page {doc.page}',
    )

    # AiGenics logo in top-right corner
    try:
        logo_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'view', 'logo.png')
        if os.path.exists(logo_path):
            canvas.drawImage(logo_path, page_w - 45 * mm, page_h - 15 * mm,
                             width=30 * mm, height=10 * mm, preserveAspectRatio=True, mask='auto')
    except Exception:
        pass

    canvas.restoreState()


# ── PDF Builder ──────────────────────────────────────────────────────────────

def generate_pdf(
    consolidated: Dict,
    lc_fields_raw: Dict,
    output_path: str,
    progress_fn=None,
) -> str:
    """Generate the final compliance report PDF."""
    if progress_fn is None:
        def progress_fn(msg): pass

    styles = _build_styles()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        topMargin=22 * mm, bottomMargin=22 * mm,
        leftMargin=22 * mm, rightMargin=22 * mm,
        title='ComplyTrade Compliance Report', author='ComplyTrade',
    )

    # Extract LC fields from step06 structure
    lc = _extract_lc_fields(lc_fields_raw)

    decision = consolidated.get('overall_decision', 'REVIEW REQUIRED')
    stats = {
        'total_pass': consolidated.get('total_pass', 0),
        'total_fail': consolidated.get('total_fail', 0),
        'total_review': consolidated.get('total_review', 0),
    }

    elements = []

    # 1. Cover page
    progress_fn("Building cover page...")
    elements.extend(_build_cover(lc, decision, stats, styles))

    # 2. Executive Summary
    progress_fn("Building executive summary...")
    elements.extend(_build_executive_summary(
        decision,
        consolidated.get('critical_findings', []),
        consolidated.get('review_items', []),
        stats,
        styles,
    ))

    # 3. Section tables
    progress_fn("Building clause-by-clause tables...")
    sections = consolidated.get('sections', [])
    elements.extend(_build_section_tables(sections, styles))

    # Build PDF
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

    Args:
        consolidated: Step 19 output dict
        lc_fields:    Step 06 result dict (contains final_lc with consolidated_fields)
        output_dir:   directory for step output
        progress_fn:  callback for progress messages
    """
    t0 = time.time()
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if progress_fn is None:
        def progress_fn(msg): pass

    progress_fn("Step 20: Final Compliance Report Generation")

    # Extract LC number for filename
    lc = _extract_lc_fields(lc_fields)
    lc_number = lc.get('lc_number', 'report') or 'report'
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
            'report_path': abs_path,
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

    # Save result metadata
    meta_path = Path(output_dir) / 'step20_result.json'
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    if 'error' not in result:
        progress_fn(
            f"Step 20 complete: {pdf_filename} -- "
            f"{result['overall_decision']} in {result['elapsed_seconds']:.1f}s"
        )

    return result
