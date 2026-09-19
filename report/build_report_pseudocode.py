"""Builds BronchoTrack_V3_Fusion_Pseudocode.pdf -- pseudocode for every file
in the bronchotrack.fusion package (plus scalar_kalman.py and
tests/test_fusion_smoke.py), per the user's request for "just the
pseudocode for each file".
"""
import datetime

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Preformatted, PageBreak,
    Table, TableStyle, HRFlowable, KeepTogether,
)

from pseudocode_content import SECTIONS

OUT_PATH = "/home/claude/bronchotrack_pipeline/report/BronchoTrack_V3_Fusion_Pseudocode.pdf"

INK = colors.HexColor("#1c2b33")
MUTED = colors.HexColor("#54646d")
ACCENT = colors.HexColor("#1f6f78")
CODE_BG = colors.HexColor("#f4f2ec")
CODE_EDGE = colors.HexColor("#d8d3c4")

MARGIN = 0.75 * inch

doc = SimpleDocTemplate(
    OUT_PATH,
    pagesize=letter,
    leftMargin=MARGIN,
    rightMargin=MARGIN,
    topMargin=0.85 * inch,
    bottomMargin=0.75 * inch,
    title="BronchoTrack_V3 -- Fusion Package Pseudocode",
    author="Aveesha Nishendra",
)

base = getSampleStyleSheet()

title_style = ParagraphStyle(
    "TitleCustom", parent=base["Title"], fontName="Helvetica-Bold",
    fontSize=21, leading=25, textColor=INK, spaceAfter=4,
)
subtitle_style = ParagraphStyle(
    "SubtitleCustom", parent=base["Normal"], fontName="Helvetica-Oblique",
    fontSize=11, leading=14, textColor=MUTED, spaceAfter=2,
)
meta_style = ParagraphStyle(
    "MetaCustom", parent=base["Normal"], fontName="Helvetica",
    fontSize=9.5, leading=13, textColor=MUTED,
)
intro_style = ParagraphStyle(
    "IntroCustom", parent=base["Normal"], fontName="Helvetica",
    fontSize=10.3, leading=15, textColor=INK, spaceAfter=6, alignment=TA_LEFT,
)
filename_style = ParagraphStyle(
    "FileName", parent=base["Normal"], fontName="Courier-Bold",
    fontSize=13, leading=16, textColor=ACCENT, spaceBefore=2, spaceAfter=3,
)
filedesc_style = ParagraphStyle(
    "FileDesc", parent=base["Normal"], fontName="Helvetica-Oblique",
    fontSize=9.3, leading=13, textColor=MUTED, spaceAfter=8,
)
code_style = ParagraphStyle(
    "Code", parent=base["Code"], fontName="Courier", fontSize=8.0,
    leading=10.6, textColor=INK, backColor=CODE_BG, borderColor=CODE_EDGE,
    borderWidth=0.75, borderPadding=(8, 8, 8, 8), leftIndent=0,
)

story = []

# ---- Title block ----
story.append(Paragraph("BronchoTrack_V3", title_style))
story.append(Paragraph("Fusion Package -- Pseudocode Reference", subtitle_style))
story.append(Spacer(1, 10))

meta_table = Table(
    [
        ["Prepared by", "Aveesha Nishendra"],
        ["Date", datetime.date.today().strftime("%B %d, %Y")],
        ["Scope", "bronchotrack/fusion/ (9 files), bronchotrack/scalar_kalman.py,\ntests/test_fusion_smoke.py"],
    ],
    colWidths=[1.2 * inch, 4.6 * inch],
)
meta_table.setStyle(TableStyle([
    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
    ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
    ("FONTSIZE", (0, 0), (-1, -1), 9.5),
    ("TEXTCOLOR", (0, 0), (-1, -1), INK),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("TOPPADDING", (0, 0), (-1, -1), 2),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ("LEFTPADDING", (0, 0), (-1, -1), 0),
]))
story.append(meta_table)
story.append(Spacer(1, 12))
story.append(HRFlowable(width="100%", thickness=1, color=CODE_EDGE, spaceAfter=12))

story.append(Paragraph(
    "This document lays out pseudocode for every file in the "
    "<b>bronchotrack.fusion</b> package -- the three-model fused "
    "localization pipeline (diameter-growth motion model, point-based "
    "bearing matching, and whole-mask ratio method, combined through a "
    "Kalman filter) -- plus the shared <b>scalar_kalman.py</b> filter it "
    "depends on and its smoke-test suite. Each entry gives the file's "
    "role in one line, followed by a functional walkthrough of its "
    "classes and functions in the order they execute.",
    intro_style,
))
story.append(Spacer(1, 4))

# ---- Per-file sections ----
for i, (fname, desc, code) in enumerate(SECTIONS):
    # Keep the filename + description glued to the start of its code block,
    # but don't force pathologically long code blocks to stay whole.
    header_block = KeepTogether([
        Paragraph(fname, filename_style),
        Paragraph(desc, filedesc_style),
    ])
    story.append(header_block)
    story.append(Preformatted(code, code_style))
    story.append(Spacer(1, 14))
    if i < len(SECTIONS) - 1:
        story.append(HRFlowable(width="100%", thickness=0.5, color=CODE_EDGE, spaceAfter=14))


def _footer(canvas, doc_):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(MARGIN, 0.5 * inch, "BronchoTrack_V3 -- Fusion Package Pseudocode")
    canvas.drawRightString(letter[0] - MARGIN, 0.5 * inch, f"Page {doc_.page}")
    canvas.restoreState()


doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
print(f"saved: {OUT_PATH}")
