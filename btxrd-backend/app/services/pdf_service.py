"""
PDF Report Generation Service
Converts markdown report text + images into a professional downloadable PDF.
Uses reportlab for reliable cross-platform PDF generation.
"""

from __future__ import annotations
import io
import os
import re
import logging
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm, inch
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, HRFlowable, KeepTogether,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

logger = logging.getLogger(__name__)

# ── Colour Palette ─────────────────────────────────────────────────────────
NAVY = colors.HexColor("#0f172a")
DARK_BLUE = colors.HexColor("#1e3a5f")
BLUE = colors.HexColor("#2563eb")
LIGHT_BLUE = colors.HexColor("#dbeafe")
SLATE = colors.HexColor("#475569")
LIGHT_SLATE = colors.HexColor("#94a3b8")
WHITE = colors.white
AMBER = colors.HexColor("#d97706")
AMBER_BG = colors.HexColor("#fef3c7")
RED = colors.HexColor("#dc2626")
TABLE_HEADER_BG = colors.HexColor("#1e3a5f")
TABLE_ALT_ROW = colors.HexColor("#f1f5f9")
BORDER_COLOR = colors.HexColor("#cbd5e1")


def _build_styles() -> dict:
    """Build all paragraph styles for the report."""
    styles = getSampleStyleSheet()

    custom = {
        "title": ParagraphStyle(
            "ReportTitle", parent=styles["Title"],
            fontSize=22, leading=28, textColor=NAVY,
            spaceAfter=4, alignment=TA_CENTER,
            fontName="Helvetica-Bold",
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle", parent=styles["Normal"],
            fontSize=10, leading=14, textColor=LIGHT_SLATE,
            alignment=TA_CENTER, spaceAfter=20,
        ),
        "section_heading": ParagraphStyle(
            "SectionHeading", parent=styles["Heading2"],
            fontSize=13, leading=18, textColor=DARK_BLUE,
            spaceBefore=16, spaceAfter=6,
            fontName="Helvetica-Bold",
            borderWidth=0, borderPadding=0,
        ),
        "body": ParagraphStyle(
            "ReportBody", parent=styles["Normal"],
            fontSize=10, leading=15, textColor=NAVY,
            alignment=TA_JUSTIFY, spaceAfter=6,
        ),
        "body_bold": ParagraphStyle(
            "ReportBodyBold", parent=styles["Normal"],
            fontSize=10, leading=15, textColor=NAVY,
            fontName="Helvetica-Bold", spaceAfter=4,
        ),
        "bullet": ParagraphStyle(
            "ReportBullet", parent=styles["Normal"],
            fontSize=10, leading=15, textColor=NAVY,
            leftIndent=18, spaceAfter=3,
            bulletIndent=6, bulletFontSize=10,
        ),
        "numbered": ParagraphStyle(
            "ReportNumbered", parent=styles["Normal"],
            fontSize=10, leading=15, textColor=NAVY,
            leftIndent=18, spaceAfter=3,
        ),
        "disclaimer": ParagraphStyle(
            "Disclaimer", parent=styles["Normal"],
            fontSize=9, leading=13, textColor=RED,
            alignment=TA_JUSTIFY, spaceAfter=6,
            fontName="Helvetica-BoldOblique",
        ),
        "footer": ParagraphStyle(
            "Footer", parent=styles["Normal"],
            fontSize=7, leading=10, textColor=LIGHT_SLATE,
            alignment=TA_CENTER,
        ),
        "image_caption": ParagraphStyle(
            "ImageCaption", parent=styles["Normal"],
            fontSize=9, leading=12, textColor=SLATE,
            alignment=TA_CENTER, spaceAfter=12,
        ),
        "table_header": ParagraphStyle(
            "TableHeader", parent=styles["Normal"],
            fontSize=9, leading=12, textColor=WHITE,
            fontName="Helvetica-Bold",
        ),
        "table_cell": ParagraphStyle(
            "TableCell", parent=styles["Normal"],
            fontSize=9, leading=12, textColor=NAVY,
        ),
    }
    return custom


def _parse_markdown_to_flowables(md_text: str, styles: dict) -> list:
    """Parse the markdown report text into reportlab flowables."""
    flowables = []
    lines = md_text.strip().split("\n")
    i = 0

    while i < len(lines):
        line = lines[i].rstrip()

        # Skip empty lines
        if not line:
            flowables.append(Spacer(1, 4))
            i += 1
            continue

        # Section heading (## HEADING)
        if line.startswith("## "):
            heading_text = line[3:].strip()
            # Add a thin horizontal rule before each section
            flowables.append(Spacer(1, 6))
            flowables.append(HRFlowable(
                width="100%", thickness=0.5, color=BORDER_COLOR,
                spaceBefore=2, spaceAfter=6,
            ))
            flowables.append(Paragraph(heading_text, styles["section_heading"]))
            i += 1
            continue

        # H1 heading (# HEADING) — skip, we have our own title
        if line.startswith("# "):
            i += 1
            continue

        # Table detection (| ... | ... |)
        if line.startswith("|") and "|" in line[1:]:
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                stripped = lines[i].strip()
                # Skip separator rows (|---|---|)
                if re.match(r"^\|[\s\-:|]+\|$", stripped):
                    i += 1
                    continue
                cells = [c.strip() for c in stripped.split("|")[1:-1]]
                table_lines.append(cells)
                i += 1

            if table_lines:
                flowables.append(_build_table(table_lines, styles))
                flowables.append(Spacer(1, 8))
            continue

        # Numbered list (1. item, 2. item)
        if re.match(r"^\d+\.\s", line):
            text = re.sub(r"^\d+\.\s*", "", line)
            text = _clean_markdown_inline(text)
            flowables.append(Paragraph(f"• {text}", styles["numbered"]))
            i += 1
            continue

        # Bullet list (- item or * item)
        if line.startswith("- ") or line.startswith("* "):
            text = line[2:].strip()
            text = _clean_markdown_inline(text)
            flowables.append(Paragraph(f"• {text}", styles["bullet"]))
            i += 1
            continue

        # Disclaimer line (⚠️)
        if "⚠️" in line or "DISCLAIMER" in line.upper():
            text = _clean_markdown_inline(line)
            flowables.append(Paragraph(text, styles["disclaimer"]))
            i += 1
            continue

        # Regular paragraph
        text = _clean_markdown_inline(line)
        if text:
            flowables.append(Paragraph(text, styles["body"]))
        i += 1

    return flowables


def _clean_markdown_inline(text: str) -> str:
    """Convert inline markdown (bold, italic) to reportlab XML tags."""
    # Bold: **text** → <b>text</b>
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    # Italic: *text* → <i>text</i>
    text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
    # Remove any remaining markdown artifacts
    text = text.replace("---", "")
    return text.strip()


def _build_table(rows: list[list[str]], styles: dict) -> Table:
    """Build a styled reportlab Table from parsed rows."""
    if not rows:
        return Spacer(1, 1)

    # First row is header
    header = rows[0]
    data_rows = rows[1:] if len(rows) > 1 else []

    table_data = []
    # Header row
    table_data.append([
        Paragraph(_clean_markdown_inline(cell), styles["table_header"])
        for cell in header
    ])
    # Data rows
    for row in data_rows:
        table_data.append([
            Paragraph(_clean_markdown_inline(cell), styles["table_cell"])
            for cell in row
        ])

    num_cols = len(header)
    col_widths = [None] * num_cols  # Auto-size

    table = Table(table_data, colWidths=col_widths, repeatRows=1)

    style_commands = [
        ("BACKGROUND", (0, 0), (-1, 0), TABLE_HEADER_BG),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]
    # Alternate row shading
    for row_idx in range(1, len(table_data)):
        if row_idx % 2 == 0:
            style_commands.append(("BACKGROUND", (0, row_idx), (-1, row_idx), TABLE_ALT_ROW))

    table.setStyle(TableStyle(style_commands))
    return table


def _add_header_footer(canvas, doc):
    """Draw header and footer on every page."""
    canvas.saveState()

    # Header line
    canvas.setStrokeColor(BLUE)
    canvas.setLineWidth(1.5)
    canvas.line(25 * mm, A4[1] - 18 * mm, A4[0] - 25 * mm, A4[1] - 18 * mm)

    # Header text
    canvas.setFont("Helvetica-Bold", 8)
    canvas.setFillColor(DARK_BLUE)
    canvas.drawString(25 * mm, A4[1] - 16 * mm, "BTXRD — AI-Assisted Bone Tumor X-Ray Analysis")

    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(LIGHT_SLATE)
    date_str = datetime.now().strftime("%B %d, %Y  |  %H:%M")
    canvas.drawRightString(A4[0] - 25 * mm, A4[1] - 16 * mm, date_str)

    # Footer
    canvas.setStrokeColor(BORDER_COLOR)
    canvas.setLineWidth(0.5)
    canvas.line(25 * mm, 18 * mm, A4[0] - 25 * mm, 18 * mm)

    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(LIGHT_SLATE)
    canvas.drawString(25 * mm, 13 * mm, "BTXRD v1.0  |  For Research & Educational Use Only  |  NOT a Clinical Diagnosis")
    canvas.drawRightString(A4[0] - 25 * mm, 13 * mm, f"Page {doc.page}")

    canvas.restoreState()


def generate_pdf(
    report_text: str,
    analysis: dict,
    upload_dir: str,
    case_id: str = "N/A",
) -> bytes:
    """
    Generate a professional PDF report.

    Args:
        report_text: Markdown-formatted report from LLM
        analysis: Full inference result dict
        upload_dir: Path to uploads directory (for resolving image paths)
        case_id: Image/case identifier

    Returns:
        PDF file content as bytes
    """
    buffer = io.BytesIO()
    styles = _build_styles()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        topMargin=25 * mm,
        bottomMargin=25 * mm,
        leftMargin=25 * mm,
        rightMargin=25 * mm,
        title="BTXRD AI Radiology Report",
        author="BTXRD System",
    )

    story = []

    # ── Title Page Content ─────────────────────────────────────────────────
    story.append(Spacer(1, 30))

    # Title
    story.append(Paragraph("AI-Assisted Radiology Report", styles["title"]))
    story.append(Paragraph("Bone Tumor X-Ray Detection & Analysis", styles["subtitle"]))
    story.append(Spacer(1, 8))

    # Meta info table
    now = datetime.now()
    meta_data = [
        [Paragraph("<b>Case ID:</b>", styles["table_cell"]),
         Paragraph(case_id, styles["table_cell"])],
        [Paragraph("<b>Report Date:</b>", styles["table_cell"]),
         Paragraph(now.strftime("%B %d, %Y at %H:%M"), styles["table_cell"])],
        [Paragraph("<b>System:</b>", styles["table_cell"]),
         Paragraph("BTXRD v1.0 (ConvNeXt-Tiny + SegFormer-B2)", styles["table_cell"])],
        [Paragraph("<b>Report Type:</b>", styles["table_cell"]),
         Paragraph("AI-Generated Screening Report", styles["table_cell"])],
    ]
    meta_table = Table(meta_data, colWidths=[120, 320])
    meta_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("BACKGROUND", (0, 0), (0, -1), TABLE_ALT_ROW),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 20))

    # ── Images ─────────────────────────────────────────────────────────────
    original_url = analysis.get("original_url", "")
    cls_gradcam_url = analysis.get("cls_gradcam_url", "")
    seg = analysis.get("segmentation") or {}
    overlay_url = seg.get("overlay_url", "")
    mask_url = seg.get("mask_url", "")

    image_paths = {
        "Original X-Ray": _resolve_image_path(original_url, upload_dir),
        "Grad-CAM Attention Map": _resolve_image_path(cls_gradcam_url, upload_dir),
        "Segmentation Overlay": _resolve_image_path(overlay_url, upload_dir),
    }

    # Add images in pairs
    valid_images = [(label, path) for label, path in image_paths.items() if path and os.path.exists(path)]

    if valid_images:
        story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER_COLOR, spaceAfter=8))
        story.append(Paragraph("IMAGING", styles["section_heading"]))
        story.append(Spacer(1, 6))

        # Build image row (up to 3 images side by side)
        img_cells = []
        cap_cells = []
        for label, path in valid_images:
            try:
                img = Image(path, width=2.0 * inch, height=2.0 * inch, kind="proportional")
                img_cells.append(img)
                cap_cells.append(Paragraph(f"<i>{label}</i>", styles["image_caption"]))
            except Exception as e:
                logger.warning("Failed to embed image %s: %s", path, e)

        if img_cells:
            # Images row
            img_table = Table([img_cells, cap_cells], colWidths=[2.2 * inch] * len(img_cells))
            img_table.setStyle(TableStyle([
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(img_table)

    story.append(Spacer(1, 12))

    # ── Report Body ────────────────────────────────────────────────────────
    body_flowables = _parse_markdown_to_flowables(report_text, styles)
    story.extend(body_flowables)

    # ── Build PDF ──────────────────────────────────────────────────────────
    doc.build(story, onFirstPage=_add_header_footer, onLaterPages=_add_header_footer)
    return buffer.getvalue()


def _resolve_image_path(url: str, upload_dir: str) -> str | None:
    """Resolve a URL like /files/xxx.png to an absolute filesystem path."""
    if not url:
        return None

    # Strip /files/ prefix
    if url.startswith("/files/"):
        relative = url[len("/files/"):]
        abs_path = os.path.join(upload_dir, relative)
        if os.path.exists(abs_path):
            return abs_path

    # Try as absolute path
    if os.path.exists(url):
        return url

    return None
