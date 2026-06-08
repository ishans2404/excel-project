import json
import os
import base64
import re
import warnings
from datetime import datetime
from langchain_core.tools import tool

warnings.filterwarnings("ignore")


@tool
def report_generator_tool(
    title: str,
    sections_json: str,
    summary: str = "",
    kpis_json: str = "[]",
    session_id: str = "report",
    metadata_json: str = "{}",
) -> str:
    """Generate a professional PDF report. Analyze data FIRST, then call this tool.

    Args:
        title: Report title
        sections_json: JSON array of sections (REQUIRED — must contain real analysis data):
            [{"heading": "Title", "content": "Real findings with numbers",
              "chart_b64": "base64_png_optional",
              "table": [["Col1","Col2"],["val1","val2"]] }]
            Content: supports **bold**, ## subheading, - bullets, 1. numbered items
        summary: Executive summary text (optional)
        kpis_json: JSON array: [{"label": "Total Cases", "value": "1,234"}]
        session_id: used in output filename
        metadata_json: {"author":"...","period":"...","source":"..."} (optional)
    """
    try:
        sections = json.loads(sections_json) if sections_json and sections_json.strip() not in ("","[]") else []
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"sections_json parse error: {e}"})
    try:
        kpis = json.loads(kpis_json) if kpis_json and kpis_json.strip() not in ("","[]") else []
    except json.JSONDecodeError:
        kpis = []
    try:
        metadata = json.loads(metadata_json) if metadata_json and metadata_json.strip() not in ("","{}") else {}
    except json.JSONDecodeError:
        metadata = {}

    if not sections and not summary:
        return json.dumps({"error": "No content. Analyze data first, collect findings, then pass as sections_json."})

    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib.colors import HexColor, white, black
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, Image as RLImage, KeepTogether,
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY, TA_RIGHT

    PRIMARY = HexColor("#1A237E")
    PRIMARY_LIGHT = HexColor("#3949AB")
    ACCENT = HexColor("#E3F2FD")
    BORDER = HexColor("#9FA8DA")
    TEXT = HexColor("#212121")
    MUTED = HexColor("#757575")

    # Styles
    title_st = ParagraphStyle("T", fontSize=24, textColor=white, fontName="Helvetica-Bold",
                               alignment=TA_CENTER, spaceAfter=6)
    sub_st = ParagraphStyle("S", fontSize=10, textColor=HexColor("#C5CAE9"),
                              alignment=TA_CENTER, spaceAfter=4)
    meta_st = ParagraphStyle("M", fontSize=8, textColor=HexColor("#9FA8DA"),
                               alignment=TA_CENTER)
    h1_st = ParagraphStyle("H1", fontSize=14, textColor=PRIMARY, fontName="Helvetica-Bold",
                             spaceBefore=14, spaceAfter=6)
    h2_st = ParagraphStyle("H2", fontSize=11, textColor=PRIMARY_LIGHT, fontName="Helvetica-Bold",
                             spaceBefore=8, spaceAfter=4)
    body_st = ParagraphStyle("B", fontSize=9, leading=14, spaceAfter=4,
                               alignment=TA_JUSTIFY, textColor=TEXT)
    bullet_st = ParagraphStyle("BU", fontSize=9, leading=13, spaceAfter=2,
                                 leftIndent=16, bulletIndent=8, textColor=TEXT)
    num_st = ParagraphStyle("N", fontSize=9, leading=13, spaceAfter=2,
                               leftIndent=16, textColor=TEXT)
    caption_st = ParagraphStyle("C", fontSize=8, textColor=MUTED, alignment=TA_CENTER, spaceAfter=4)
    kpi_val_st = ParagraphStyle("KV", fontSize=18, textColor=PRIMARY, fontName="Helvetica-Bold",
                                  alignment=TA_CENTER)
    kpi_lbl_st = ParagraphStyle("KL", fontSize=8, textColor=MUTED, alignment=TA_CENTER)

    story = []

    # ── Cover ──────────────────────────────────────────────────────────────
    meta_lines = []
    if metadata.get("period"):
        meta_lines.append(f"Period: {metadata['period']}")
    if metadata.get("source"):
        meta_lines.append(f"Source: {metadata['source']}")
    if metadata.get("author"):
        meta_lines.append(f"Prepared by: {metadata['author']}")
    meta_lines.append(f"Generated: {datetime.now().strftime('%B %d, %Y  %H:%M')}")

    cover_data = [[Paragraph(title, title_st)]]
    cover_data.append([Paragraph(" · ".join(meta_lines[:2]), sub_st)])
    if len(meta_lines) > 2:
        cover_data.append([Paragraph(" · ".join(meta_lines[2:]), meta_st)])

    story.append(Table(
        cover_data, colWidths=[17 * cm],
        style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
            ("BACKGROUND", (0, 1), (-1, -1), PRIMARY_LIGHT),
            ("TOPPADDING", (0, 0), (-1, -1), 18),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
            ("LEFTPADDING", (0, 0), (-1, -1), 20),
            ("RIGHTPADDING", (0, 0), (-1, -1), 20),
        ])
    ))
    story.append(Spacer(1, 16))

    # ── KPI Cards ──────────────────────────────────────────────────────────
    if kpis:
        story.append(Paragraph("Key Metrics", h1_st))
        story.append(HRFlowable(width="100%", thickness=1.5, color=PRIMARY))
        story.append(Spacer(1, 8))

        cards_per_row = min(4, len(kpis))
        card_w = 17 * cm / cards_per_row

        rows_chunks = [kpis[i:i + cards_per_row] for i in range(0, len(kpis), cards_per_row)]
        for chunk in rows_chunks:
            cells = []
            for kpi in chunk:
                val_text = str(kpi.get("value", "—"))
                if kpi.get("unit"):
                    val_text += f" {kpi['unit']}"
                cell = Table(
                    [[Paragraph(val_text, kpi_val_st)],
                     [Paragraph(kpi.get("label", ""), kpi_lbl_st)]],
                    colWidths=[card_w - 8],
                    style=TableStyle([
                        ("BACKGROUND", (0, 0), (-1, -1), ACCENT),
                        ("BOX", (0, 0), (-1, -1), 0.5, BORDER),
                        ("TOPPADDING", (0, 0), (-1, -1), 10),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                    ])
                )
                cells.append(cell)
            while len(cells) < cards_per_row:
                cells.append(Spacer(1, 1))
            story.append(Table(
                [cells], colWidths=[card_w] * cards_per_row,
                style=TableStyle([
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ])
            ))
            story.append(Spacer(1, 6))
        story.append(Spacer(1, 10))

    # ── Executive Summary ──────────────────────────────────────────────────
    if summary:
        story.append(Paragraph("Executive Summary", h1_st))
        story.append(HRFlowable(width="100%", thickness=1.5, color=PRIMARY))
        story.append(Spacer(1, 6))
        for line in summary.split("\n"):
            line = line.strip()
            if line:
                line_fmt = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", line)
                story.append(Paragraph(line_fmt[:800], body_st))
        story.append(Spacer(1, 12))

    # ── Sections ───────────────────────────────────────────────────────────
    for sec in sections:
        heading = sec.get("heading", "")
        content = sec.get("content", "")
        chart_b64 = sec.get("chart_b64", "")
        table_data = sec.get("table", [])

        section_elements = []

        if heading:
            section_elements.append(Paragraph(heading, h1_st))
            section_elements.append(HRFlowable(width="100%", thickness=0.8, color=BORDER))
            section_elements.append(Spacer(1, 6))

        if content:
            counter = [0]
            for line in content.split("\n"):
                line = line.strip()
                if not line:
                    section_elements.append(Spacer(1, 4))
                    continue

                line_fmt = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", line)

                if line.startswith("## "):
                    section_elements.append(Paragraph(line[3:], h2_st))
                elif line.startswith("# "):
                    section_elements.append(Paragraph(line[2:], h1_st))
                elif line.startswith("- ") or line.startswith("• "):
                    section_elements.append(Paragraph(f"• {line_fmt[2:]}", bullet_st))
                elif re.match(r"^\d+\.", line):
                    # Numbered list item
                    section_elements.append(Paragraph(line_fmt[:600], num_st))
                elif line.startswith("|") and "|" in line[1:]:
                    # Skip markdown table separator lines
                    if re.match(r"^[\|\-\s:]+$", line):
                        continue
                    # Parse markdown table row
                    cells_raw = [c.strip() for c in line.strip("|").split("|")]
                    # Will be handled as a group below (simple inline table)
                    section_elements.append(Paragraph(line_fmt[:600], body_st))
                else:
                    section_elements.append(Paragraph(line_fmt[:600], body_st))

        # Structured table (list of lists)
        if table_data and len(table_data) >= 2:
            try:
                t_data = []
                for i, row in enumerate(table_data[:60]):
                    t_data.append([Paragraph(str(cell), ParagraphStyle(
                        "TC", fontSize=8,
                        fontName="Helvetica-Bold" if i == 0 else "Helvetica",
                        textColor=white if i == 0 else TEXT,
                        alignment=TA_CENTER if i == 0 else TA_LEFT,
                    )) for cell in row])

                n_cols = len(table_data[0])
                col_w = 17 * cm / n_cols

                t_style = [
                    ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, ACCENT]),
                    ("GRID", (0, 0), (-1, -1), 0.4, BORDER),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ]
                section_elements.append(Table(t_data, colWidths=[col_w] * n_cols, style=TableStyle(t_style)))
                section_elements.append(Spacer(1, 6))
            except Exception:
                pass

        # Chart image
        if chart_b64:
            try:
                import io as _io
                from PIL import Image as PILImage
                img_data = base64.b64decode(chart_b64)
                img_buf = _io.BytesIO(img_data)
                pil_img = PILImage.open(img_buf)
                w, h = pil_img.size
                max_w = 16 * cm
                scale = min(max_w / w, 11 * cm / h)
                img_buf.seek(0)
                rl_img = RLImage(img_buf, width=w * scale, height=h * scale)
                section_elements.append(rl_img)
                section_elements.append(Paragraph(heading, caption_st))
            except Exception as e:
                section_elements.append(Paragraph(f"[Chart could not render: {e}]", caption_st))

        section_elements.append(Spacer(1, 14))

        # Keep heading + first few elements together
        try:
            story.append(KeepTogether(section_elements[:4]))
            story.extend(section_elements[4:])
        except Exception:
            story.extend(section_elements)

    # ── Footer ─────────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=HexColor("#BDBDBD")))
    story.append(Paragraph("Enterprise Analytics AI Agent", caption_st))

    # ── Build PDF ──────────────────────────────────────────────────────────
    from config import settings
    os.makedirs(settings.REPORTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"report_{session_id[:8]}_{timestamp}.pdf"
    report_path = os.path.join(settings.REPORTS_DIR, filename)

    try:
        doc = SimpleDocTemplate(
            report_path, pagesize=A4,
            rightMargin=2 * cm, leftMargin=2 * cm,
            topMargin=1.5 * cm, bottomMargin=2 * cm,
        )
        doc.build(story)
        size_kb = round(os.path.getsize(report_path) / 1024, 1)
        return json.dumps({
            "status": "success",
            "report_path": report_path,
            "filename": filename,
            "size_kb": size_kb,
            "sections_count": len(sections),
            "kpis_count": len(kpis),
            "message": f"PDF report ready for download ({size_kb} KB, {len(sections)} sections).",
        })
    except Exception as e:
        import traceback
        return json.dumps({"error": str(e), "trace": traceback.format_exc()})