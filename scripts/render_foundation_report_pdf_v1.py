#!/usr/bin/env python3
"""Render the bilingual dependency-mask Markdown reports as polished PDFs."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
    Table, TableStyle,
)
from pypdf import PdfReader


NAVY = colors.HexColor("#17365D")
MID_BLUE = colors.HexColor("#4F81BD")
LIGHT_BLUE = colors.HexColor("#DCE6F1")
PALE = colors.HexColor("#F5F7FA")
TEXT = colors.HexColor("#3D3D3D")


def register_fonts():
    pdfmetrics.registerFont(TTFont("Body", r"C:\Windows\Fonts\msyh.ttc", subfontIndex=0))
    pdfmetrics.registerFont(TTFont("BodyBold", r"C:\Windows\Fonts\msyhbd.ttc", subfontIndex=0))
    pdfmetrics.registerFont(TTFont("Mono", r"C:\Windows\Fonts\consola.ttf"))


def inline(text):
    value = html.escape(text.strip())
    value = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", value)
    value = re.sub(r"`(.+?)`", r'<font name="Mono" size="8">\1</font>', value)
    value = re.sub(r"\[(.+?)\]\((https?://[^)]+)\)", r'<a href="\2" color="#17365D">\1</a>', value)
    return value


def parse_report(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    title = next(line[2:].strip() for line in lines if line.startswith("# "))
    metadata, headings = [], []
    for line in lines:
        if line.startswith("**") and line.count("**") >= 2:
            metadata.append(line.replace("**", "").strip())
        if line.startswith("## "):
            headings.append(line[3:].strip())
    return lines, title, metadata[:3], headings


def styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Title", parent=base["Title"], fontName="BodyBold", fontSize=22,
            leading=30, textColor=NAVY, alignment=TA_CENTER, spaceAfter=14,
        ),
        "meta": ParagraphStyle(
            "Meta", parent=base["BodyText"], fontName="Body", fontSize=10,
            leading=16, textColor=colors.HexColor("#666666"), alignment=TA_CENTER,
        ),
        "toc_title": ParagraphStyle(
            "TocTitle", parent=base["Heading2"], fontName="BodyBold", fontSize=14,
            leading=18, textColor=NAVY, spaceBefore=12, spaceAfter=8,
        ),
        "toc": ParagraphStyle(
            "Toc", parent=base["BodyText"], fontName="Body", fontSize=10,
            leading=18, textColor=NAVY, leftIndent=8,
        ),
        "h2": ParagraphStyle(
            "H2", parent=base["Heading1"], fontName="BodyBold", fontSize=16,
            leading=21, textColor=NAVY, spaceBefore=13, spaceAfter=8,
            keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "H3", parent=base["Heading2"], fontName="BodyBold", fontSize=11.5,
            leading=16, textColor=MID_BLUE, spaceBefore=9, spaceAfter=5,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "Body", parent=base["BodyText"], fontName="Body", fontSize=8.8,
            leading=14.2, textColor=TEXT, spaceAfter=6, alignment=TA_LEFT,
            wordWrap="CJK",
        ),
        "bullet": ParagraphStyle(
            "Bullet", parent=base["BodyText"], fontName="Body", fontSize=8.7,
            leading=13.8, textColor=TEXT, leftIndent=14, firstLineIndent=-8,
            spaceAfter=3, wordWrap="CJK",
        ),
        "quote": ParagraphStyle(
            "Quote", parent=base["BodyText"], fontName="Body", fontSize=9.2,
            leading=15, textColor=NAVY, leftIndent=16, rightIndent=12,
            borderColor=MID_BLUE, borderWidth=1.2, borderPadding=8,
            backColor=colors.HexColor("#F2F6FA"), spaceBefore=5, spaceAfter=9,
            wordWrap="CJK",
        ),
        "caption": ParagraphStyle(
            "Caption", parent=base["BodyText"], fontName="Body", fontSize=7.5,
            leading=10.5, textColor=colors.HexColor("#666666"),
            alignment=TA_CENTER, spaceAfter=7, wordWrap="CJK",
        ),
        "table": ParagraphStyle(
            "TableCell", parent=base["BodyText"], fontName="Body", fontSize=6.7,
            leading=8.8, textColor=TEXT, wordWrap="CJK",
        ),
        "table_head": ParagraphStyle(
            "TableHead", parent=base["BodyText"], fontName="BodyBold", fontSize=6.8,
            leading=9, textColor=colors.white, wordWrap="CJK",
        ),
    }


def make_table(rows, available_width, style):
    data = []
    for row_index, row in enumerate(rows):
        cell_style = style["table_head"] if row_index == 0 else style["table"]
        data.append([Paragraph(inline(cell), cell_style) for cell in row])
    columns = max(len(row) for row in data)
    widths = [available_width / columns] * columns
    result = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CCD5DF")),
    ]
    for row_index in range(1, len(data)):
        if row_index % 2 == 0:
            commands.append(("BACKGROUND", (0, row_index), (-1, row_index), PALE))
    result.setStyle(TableStyle(commands))
    return result


def markdown_story(lines, report_dir, style, available_width):
    story = []
    index = 0
    started = False
    while index < len(lines):
        line = lines[index].rstrip()
        if line.startswith("# "):
            index += 1
            continue
        if not started and not line.startswith("## "):
            index += 1
            continue
        if line.startswith("## "):
            started = True
            story.append(Paragraph(inline(line[3:]), style["h2"]))
            index += 1
            continue
        if line.startswith("### "):
            story.append(Paragraph(inline(line[4:]), style["h3"]))
            index += 1
            continue
        image_match = re.fullmatch(r"!\[(.*?)\]\((.*?)\)", line.strip())
        if image_match:
            path = report_dir / image_match.group(2)
            image = Image(str(path))
            width, height = image.imageWidth, image.imageHeight
            scale = min(available_width / width, 96 * mm / height)
            image.drawWidth = width * scale
            image.drawHeight = height * scale
            story.append(Spacer(1, 3))
            story.append(image)
            story.append(Paragraph(inline(image_match.group(1)), style["caption"]))
            index += 1
            continue
        if line.startswith("|"):
            raw_rows = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                raw_rows.append([cell.strip() for cell in lines[index].strip().strip("|").split("|")])
                index += 1
            if len(raw_rows) >= 2 and all(re.fullmatch(r":?-+:?", cell) for cell in raw_rows[1]):
                raw_rows.pop(1)
            story.append(Spacer(1, 3))
            story.append(make_table(raw_rows, available_width, style))
            story.append(Spacer(1, 7))
            continue
        if line.startswith("> "):
            story.append(Paragraph(inline(line[2:]), style["quote"]))
            index += 1
            continue
        if re.match(r"^[-*] ", line):
            story.append(Paragraph("• " + inline(line[2:]), style["bullet"]))
            index += 1
            continue
        numbered = re.match(r"^(\d+)\. (.*)", line)
        if numbered:
            story.append(Paragraph(
                f"{numbered.group(1)}. " + inline(numbered.group(2)), style["bullet"]
            ))
            index += 1
            continue
        if not line.strip():
            index += 1
            continue
        paragraph = [line.strip()]
        index += 1
        while index < len(lines):
            candidate = lines[index].rstrip()
            if (
                not candidate.strip() or candidate.startswith("#")
                or candidate.startswith("|") or candidate.startswith("![")
                or candidate.startswith("> ") or re.match(r"^[-*] ", candidate)
                or re.match(r"^\d+\. ", candidate)
            ):
                break
            paragraph.append(candidate.strip())
            index += 1
        story.append(Paragraph(inline(" ".join(paragraph)), style["body"]))
    return story


def render(source, output, language):
    register_fonts()
    lines, title, metadata, headings = parse_report(source)
    style = styles()
    width = A4[0] - 38 * mm
    doc = SimpleDocTemplate(
        str(output), pagesize=A4, rightMargin=19 * mm, leftMargin=19 * mm,
        topMargin=21 * mm, bottomMargin=18 * mm,
        title=title, author="Representation Model Study",
    )
    story = [Spacer(1, 34 * mm), Paragraph(inline(title), style["title"]), Spacer(1, 4 * mm)]
    for item in metadata:
        story.append(Paragraph(inline(item), style["meta"]))
    story += [Spacer(1, 17 * mm), Paragraph("目录" if language == "zh" else "Contents", style["toc_title"])]
    for heading in headings:
        story.append(Paragraph(inline(heading), style["toc"]))
    story.append(PageBreak())
    story.extend(markdown_story(lines, Path(source).parent, style, width))

    def page(canvas, document):
        canvas.saveState()
        canvas.setFont("Body", 7.2)
        canvas.setFillColor(colors.HexColor("#666666"))
        header = title if len(title) <= 48 else "Food-Composition Foundation Evidence"
        canvas.drawString(19 * mm, A4[1] - 11 * mm, header)
        canvas.setStrokeColor(colors.HexColor("#B8C7D9"))
        canvas.setLineWidth(0.45)
        canvas.line(19 * mm, A4[1] - 14 * mm, A4[0] - 19 * mm, A4[1] - 14 * mm)
        canvas.drawRightString(A4[0] - 19 * mm, 10 * mm, str(document.page))
        canvas.restoreState()

    doc.build(story, onFirstPage=page, onLaterPages=page)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    outputs = [
        args.report_dir / "food_foundation_evidence_dependency_mask_zh.pdf",
        args.report_dir / "food_foundation_evidence_dependency_mask_en.pdf",
    ]
    render(
        args.report_dir / "report_zh.md",
        outputs[0],
        "zh",
    )
    render(
        args.report_dir / "report_en.md",
        outputs[1],
        "en",
    )
    manifest_path = args.report_dir / "report_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "complete"
    manifest["pdf_outputs"] = {
        path.name: {
            "pages": len(PdfReader(path).pages),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in outputs
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
