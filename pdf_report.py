"""Turn the markdown brief into a clean PDF (uses reportlab)."""
import re
from datetime import date
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from tools import clean_text

ACCENT = colors.HexColor("#1F4E79")


def _safe(text: str) -> str:
    """Built-in PDF fonts only cover Windows-1252; replace anything else instead of drawing black boxes."""
    return clean_text(text).encode("cp1252", "replace").decode("cp1252")


def _inline(text: str) -> str:
    """Escape for reportlab, then apply light markdown (bold, code, links)."""
    text = escape(_safe(text))
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`(.+?)`", r"<font face='Courier'>\1</font>", text)
    text = re.sub(r"(https?://[^\s)]+)", r'<link href="\1" color="#1F4E79">\1</link>', text)
    return text


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.grey)
    canvas.drawCentredString(A4[0] / 2, 0.45 * inch, f"Page {doc.page}")
    canvas.restoreState()


def markdown_to_pdf(topic: str, brief: str, path: str) -> None:
    base = getSampleStyleSheet()
    title = ParagraphStyle("T", parent=base["Title"], fontSize=20, leading=24,
                           textColor=ACCENT, alignment=0, spaceAfter=4)
    sub = ParagraphStyle("Sub", parent=base["Normal"], fontSize=9, textColor=colors.grey, spaceAfter=10)
    h2 = ParagraphStyle("H2", parent=base["Heading2"], fontSize=13.5, leading=17, textColor=ACCENT,
                        spaceBefore=14, spaceAfter=5, keepWithNext=1)
    body = ParagraphStyle("B", parent=base["Normal"], fontSize=10.5, leading=15, spaceAfter=4)
    small = ParagraphStyle("S", parent=body, fontSize=9, leading=12.5)
    bullet = ParagraphStyle("Bu", parent=body, leftIndent=16, bulletIndent=4)
    cell = ParagraphStyle("C", parent=body, fontSize=8.5, leading=11, spaceAfter=0)
    cell_head = ParagraphStyle("CH", parent=cell, textColor=colors.white, fontName="Helvetica-Bold")

    doc = SimpleDocTemplate(path, pagesize=A4, leftMargin=0.8 * inch, rightMargin=0.8 * inch,
                            topMargin=0.8 * inch, bottomMargin=0.8 * inch,
                            title=_safe(topic), author="Research Agent")
    story = [Paragraph(_inline(topic[:1].upper() + topic[1:]), title),
             Paragraph(f"Research brief - generated {date.today():%d %B %Y}", sub),
             HRFlowable(width="100%", thickness=1, color=ACCENT, spaceAfter=6)]

    rows, section = [], ""

    def flush_table():
        if not rows:
            return
        n = len(rows[0])
        fractions = [0.14, 0.54, 0.32] if n == 3 else [1 / n] * n
        data = [[Paragraph(_inline(c), cell_head if r == 0 else cell) for c in row]
                for r, row in enumerate(rows)]
        t = Table(data, colWidths=[doc.width * f for f in fractions], repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B0B7BF")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F5F8")]),
            ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.extend([t, Spacer(1, 6)])
        rows.clear()

    for raw in brief.splitlines():
        line = raw.rstrip()
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):   # skip the |---|---| row
                rows.append(cells)
            continue
        flush_table()
        if not line.strip():
            continue
        if line.startswith("#"):
            section = line.lstrip("# ").strip()
            story.append(Paragraph(_inline(section), h2))
        elif re.match(r"^[-*]\s+", line):
            story.append(Paragraph(_inline(re.sub(r"^[-*]\s+", "", line)), bullet, bulletText="\u2022"))
        else:
            story.append(Paragraph(_inline(line), small if section == "Sources" else body))
    flush_table()

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)