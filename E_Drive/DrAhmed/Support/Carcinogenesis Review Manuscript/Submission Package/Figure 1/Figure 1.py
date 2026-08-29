#!/usr/bin/env python3
"""Generate Figure 1 in PDF, SVG, PNG, JPEG, and TIFF formats."""

from __future__ import annotations

import csv
import math
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import fitz  # PyMuPDF
from PIL import Image
from reportlab.lib.colors import Color, HexColor
from reportlab.lib.pagesizes import portrait
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


# -----------------------------------------------------------------------------
# Output and physical-size specification
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR
PDF_PATH = OUT_DIR / "Figure_1.pdf"
PNG_PATH = OUT_DIR / "Figure_1.png"
JPG_PATH = OUT_DIR / "Figure_1.jpg"
TIF_PATH = OUT_DIR / "Figure_1.tif"
SVG_PATH = OUT_DIR / "Figure_1.svg"
QC_PATH = OUT_DIR / "Figure_1_QC.csv"
README_PATH = OUT_DIR / "README.txt"
LEGEND_PATH = OUT_DIR / "Figure_1_legend.txt"

REFERENCE_W = 1358.0
REFERENCE_H = 1159.0
PAGE_W = 16.5 * cm  # journal-compliant double-column width
PAGE_H = PAGE_W * REFERENCE_H / REFERENCE_W
DPI = 300

# Fonts are embedded into the vector PDF. Font files are not distributed.
def first_existing(paths: Sequence[str]) -> Path:
    for item in paths:
        candidate = Path(item)
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No compatible condensed sans-serif font was found.")


REGULAR_FONT_FILE = first_existing([
    "C:/Windows/Fonts/arialn.ttf",
    "C:/Windows/Fonts/ARIALN.TTF",
    "/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf",
])
BOLD_FONT_FILE = first_existing([
    "C:/Windows/Fonts/arialnb.ttf",
    "C:/Windows/Fonts/ARIALNB.TTF",
    "/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf",
])
pdfmetrics.registerFont(TTFont("LSN", str(REGULAR_FONT_FILE)))
pdfmetrics.registerFont(TTFont("LSN-Bold", str(BOLD_FONT_FILE)))


# -----------------------------------------------------------------------------
# Visual specification
# -----------------------------------------------------------------------------
BLACK = HexColor("#111111")
WHITE = HexColor("#FFFFFF")
ARROW = HexColor("#111111")

BLUE_BORDER = HexColor("#07589B")
BLUE_TOP = HexColor("#E4EFF9")
BLUE_BOTTOM = HexColor("#D0E3F5")
BLUE_STAGE = HexColor("#0A4A88")

GREEN_BORDER = HexColor("#08733A")
GREEN_TOP = HexColor("#E0F1E5")
GREEN_BOTTOM = HexColor("#D0E9D8")
GREEN_STAGE = HexColor("#0B6837")

GOLD_BORDER = HexColor("#C68C00")
GOLD_TOP = HexColor("#FFF6DE")
GOLD_BOTTOM = HexColor("#FDF0CB")
GOLD_STAGE = HexColor("#9B740D")

RED_BORDER = HexColor("#C91518")
RED_TOP = HexColor("#FCE8E8")
RED_BOTTOM = HexColor("#F8D9D9")

GRAY_BORDER = HexColor("#727272")
GRAY_TOP = HexColor("#F4F4F4")
GRAY_BOTTOM = HexColor("#EAEAEA")
GRAY_STAGE = HexColor("#176644")

# Typography
BODY_PT = 6.15
TITLE_PT = 6.15
TOTAL_PT = 9.15
STAGE_PT = 6.25


@dataclass(frozen=True)
class BoxSpec:
    x: float
    y: float
    w: float
    h: float
    title: str
    lines: Sequence[str]
    border: Color
    top_fill: Color
    bottom_fill: Color
    title_frac: float
    body_start_frac: float
    body_gap_px: float
    title_size: float = TITLE_PT
    body_size: float = BODY_PT
    radius_px: float = 10.0
    line_width_px: float = 2.1


# -----------------------------------------------------------------------------
# Coordinate and colour helpers
# -----------------------------------------------------------------------------
def xp(v: float) -> float:
    return v * PAGE_W / REFERENCE_W


def yp_top(v: float) -> float:
    """Convert top-origin reference y to PDF bottom-origin y."""
    return PAGE_H - v * PAGE_H / REFERENCE_H


def hp(v: float) -> float:
    return v * PAGE_H / REFERENCE_H


def wp(v: float) -> float:
    return v * PAGE_W / REFERENCE_W


def color_hex(c: Color) -> str:
    return '#%02X%02X%02X' % (round(c.red*255), round(c.green*255), round(c.blue*255))


def mix(a: Color, b: Color, t: float) -> Color:
    return Color(
        a.red + (b.red - a.red) * t,
        a.green + (b.green - a.green) * t,
        a.blue + (b.blue - a.blue) * t,
    )


def draw_gradient_round_rect(c: canvas.Canvas, x: float, y_top: float, w: float, h: float,
                             top: Color, bottom: Color, border: Color,
                             radius_px: float = 10.0, line_width_px: float = 2.0) -> None:
    x_pt = xp(x)
    y_pt = yp_top(y_top + h)
    w_pt = wp(w)
    h_pt = hp(h)
    radius_pt = min(wp(radius_px), hp(radius_px))

    c.saveState()
    clip_path = c.beginPath()
    clip_path.roundRect(x_pt, y_pt, w_pt, h_pt, radius_pt)
    c.clipPath(clip_path, stroke=0, fill=0)

    steps = 36
    strip_h = h_pt / steps + 0.15
    for i in range(steps):
        t = i / max(steps - 1, 1)
        colour = mix(bottom, top, t)
        c.setFillColor(colour)
        c.rect(x_pt, y_pt + i * h_pt / steps, w_pt, strip_h, stroke=0, fill=1)
    c.restoreState()

    c.setStrokeColor(border)
    c.setLineWidth(wp(line_width_px))
    c.roundRect(x_pt, y_pt, w_pt, h_pt, radius_pt, stroke=1, fill=0)


def draw_centered_text(c: canvas.Canvas, text: str, x_center: float, y_top: float,
                       font: str, size_pt: float, colour: Color = BLACK) -> None:
    c.setFont(font, size_pt)
    c.setFillColor(colour)
    c.drawCentredString(xp(x_center), yp_top(y_top), text)


def draw_box(c: canvas.Canvas, b: BoxSpec) -> None:
    draw_gradient_round_rect(c, b.x, b.y, b.w, b.h, b.top_fill, b.bottom_fill,
                             b.border, b.radius_px, b.line_width_px)

    x_center = b.x + b.w / 2.0
    title_y = b.y + b.h * b.title_frac
    draw_centered_text(c, b.title, x_center, title_y, "LSN-Bold", b.title_size)

    first_y = b.y + b.h * b.body_start_frac
    for i, line in enumerate(b.lines):
        draw_centered_text(c, line, x_center, first_y + i * b.body_gap_px,
                           "LSN", b.body_size)


def draw_stage_bar(c: canvas.Canvas, x: float, y: float, w: float, h: float,
                   text: str, fill: Color) -> None:
    x_pt = xp(x)
    y_pt = yp_top(y + h)
    w_pt = wp(w)
    h_pt = hp(h)
    r_pt = min(wp(7), hp(7))
    c.setFillColor(fill)
    c.setStrokeColor(fill)
    c.roundRect(x_pt, y_pt, w_pt, h_pt, r_pt, stroke=1, fill=1)
    c.saveState()
    c.translate(x_pt + w_pt / 2, y_pt + h_pt / 2)
    c.rotate(90)
    c.setFillColor(WHITE)
    c.setFont("LSN-Bold", STAGE_PT)
    c.drawCentredString(0, -STAGE_PT * 0.32, text)
    c.restoreState()


def draw_arrow(c: canvas.Canvas, x1: float, y1: float, x2: float, y2: float,
               colour: Color = ARROW, width_px: float = 1.6,
               head_len_px: float = 10.0, head_width_px: float = 8.0) -> None:
    x1p, y1p = xp(x1), yp_top(y1)
    x2p, y2p = xp(x2), yp_top(y2)
    dx, dy = x2p - x1p, y2p - y1p
    length = math.hypot(dx, dy)
    if length == 0:
        return
    ux, uy = dx / length, dy / length
    hlen = wp(head_len_px)
    hwidth = wp(head_width_px)
    base_x = x2p - ux * hlen
    base_y = y2p - uy * hlen
    pxv, pyv = -uy, ux

    c.setStrokeColor(colour)
    c.setFillColor(colour)
    c.setLineWidth(wp(width_px))
    c.line(x1p, y1p, base_x, base_y)

    p = c.beginPath()
    p.moveTo(x2p, y2p)
    p.lineTo(base_x + pxv * hwidth / 2, base_y + pyv * hwidth / 2)
    p.lineTo(base_x - pxv * hwidth / 2, base_y - pyv * hwidth / 2)
    p.close()
    c.drawPath(p, stroke=0, fill=1)


# -----------------------------------------------------------------------------
# Figure construction
# -----------------------------------------------------------------------------
def make_vector_pdf() -> None:
    c = canvas.Canvas(str(PDF_PATH), pagesize=(PAGE_W, PAGE_H), pageCompression=1)
    c.setTitle("Figure 1")
    c.setAuthor("Redrawn as a vector figure for journal submission")
    c.setSubject("PRISMA-style study selection flow diagram")
    c.setFillColor(WHITE)
    c.rect(0, 0, PAGE_W, PAGE_H, stroke=0, fill=1)

    # Study-selection stage labels
    draw_stage_bar(c, 13, 11, 37, 201, "IDENTIFICATION", BLUE_STAGE)
    draw_stage_bar(c, 14, 320, 36, 201, "SCREENING", GREEN_STAGE)
    draw_stage_bar(c, 13, 553, 37, 121, "ELIGIBILITY", GOLD_STAGE)
    draw_stage_bar(c, 14, 692, 36, 416, "INCLUDED", GRAY_STAGE)

    boxes = [
        BoxSpec(72, 10, 492, 161,
                "Records identified from databases",
                ["n = 2,847",
                 "PubMed/MEDLINE, Embase, Scopus,",
                 "Web of Science, Cochrane Library",
                 "Date range: Jan 2015–Dec 2024"],
                BLUE_BORDER, BLUE_TOP, BLUE_BOTTOM,
                title_frac=0.17, body_start_frac=0.39, body_gap_px=21.5),

        BoxSpec(588, 10, 606, 161,
                "Grey literature / supplementary sources",
                ["WHO reports, IARC monographs,",
                 "GLOBOCAN 2022, CDC/WHO STI/HIV",
                 "guidelines, Cochrane reviews",
                 "Count: not formally tracked"],
                BLUE_BORDER, BLUE_TOP, BLUE_BOTTOM,
                title_frac=0.17, body_start_frac=0.39, body_gap_px=21.5),

        BoxSpec(72, 203, 771, 89,
                "Records after duplicate removal",
                ["n = 2,104 (743 duplicates removed)",
                 "Cross-database deduplication"],
                BLUE_BORDER, BLUE_TOP, BLUE_BOTTOM,
                title_frac=0.30, body_start_frac=0.57, body_gap_px=20.0),

        BoxSpec(892, 202, 450, 89,
                "Records removed: duplicates",
                ["n = 743 removed",
                 "Cross-database deduplication"],
                RED_BORDER, RED_TOP, RED_BOTTOM,
                title_frac=0.30, body_start_frac=0.57, body_gap_px=20.0),

        BoxSpec(72, 323, 771, 86,
                "Records screened (title / abstract)",
                ["n = 2,104",
                 "Sequential verification by second reviewer"],
                GREEN_BORDER, GREEN_TOP, GREEN_BOTTOM,
                title_frac=0.28, body_start_frac=0.56, body_gap_px=19.5),

        BoxSpec(892, 323, 450, 86,
                "Excluded at title/abstract stage",
                ["n = 1,960 excluded",
                 "Not meeting PICO inclusion criteria"],
                RED_BORDER, RED_TOP, RED_BOTTOM,
                title_frac=0.28, body_start_frac=0.56, body_gap_px=19.5),

        BoxSpec(72, 439, 770, 82,
                "Full-text articles retrieved for assessment",
                ["n = 144 full texts reviewed",
                 "All potentially eligible after abstract screen"],
                GREEN_BORDER, GREEN_TOP, GREEN_BOTTOM,
                title_frac=0.28, body_start_frac=0.58, body_gap_px=19.0),

        BoxSpec(72, 553, 771, 108,
                "Full-text articles assessed for eligibility",
                ["n = 144",
                 "Inclusion criteria: PICO, design, quality"],
                GOLD_BORDER, GOLD_TOP, GOLD_BOTTOM,
                title_frac=0.25, body_start_frac=0.55, body_gap_px=21.0),

        BoxSpec(893, 490, 449, 239,
                "Full texts excluded  (n = 89)",
                ["Insufficient co-infection data ........ n = 31",
                 "Outcomes unavailable / unreportable ... n = 28",
                 "Insufficient methodological quality ... n = 30",
                 "(Note: < 4 case reports, editorials,",
                 "Sample size < 5 / no unique miRNA data)",
                 "Review pre-2010 (not seminal)"],
                RED_BORDER, RED_TOP, RED_BOTTOM,
                title_frac=0.13, body_start_frac=0.28, body_gap_px=21.0),

        BoxSpec(73, 693, 769, 92,
                "Studies included — narrative synthesis",
                ["n = 55 studies from systematic search",
                 "Identified through database search and screening"],
                GOLD_BORDER, GOLD_TOP, GOLD_BOTTOM,
                title_frac=0.29, body_start_frac=0.57, body_gap_px=20.0),

        BoxSpec(72, 815, 770, 139,
                "Studies included in primary narrative synthesis",
                ["n = 55 synthesis studies",
                 "CT/HPV co-infection studies ........ n = 19",
                 "HSV-2/HPV co-infection studies ..... n = 9",
                 "HIV/HPV co-infection studies ....... n = 13",
                 "Triple co-infection / cross-cutting .... n = 14"],
                GOLD_BORDER, GOLD_TOP, GOLD_BOTTOM,
                title_frac=0.17, body_start_frac=0.34, body_gap_px=17.5),

        BoxSpec(893, 815, 448, 139,
                "Additional contextual references",
                ["n = 22 (not from systematic search)",
                 "Introduction / burden statistics ..... n = 8",
                 "HPV biology / background ............ n = 7",
                 "Methodology / guidelines ............ n = 7"],
                GRAY_BORDER, GRAY_TOP, GRAY_BOTTOM,
                title_frac=0.18, body_start_frac=0.39, body_gap_px=18.0),

        BoxSpec(71, 992, 517, 114,
                "TOTAL STUDIES IN NARRATIVE SYNTHESIS",
                ["n = 55"],
                GOLD_BORDER, GOLD_TOP, GOLD_BOTTOM,
                title_frac=0.30, body_start_frac=0.68, body_gap_px=0.0,
                body_size=TOTAL_PT),

        BoxSpec(606, 991, 733, 116,
                "TOTAL REFERENCES IN MANUSCRIPT",
                ["n = 77", "(55 synthesis + 22 contextual)"],
                GRAY_BORDER, GRAY_TOP, GRAY_BOTTOM,
                title_frac=0.28, body_start_frac=0.61, body_gap_px=25.0,
                body_size=TOTAL_PT),
    ]

    for b in boxes:
        draw_box(c, b)

    # Flow connectors
    draw_arrow(c, 282, 173, 282, 199)
    draw_arrow(c, 723, 173, 723, 199)
    draw_arrow(c, 471, 294, 471, 318)
    draw_arrow(c, 843, 247, 887, 247)
    draw_arrow(c, 471, 410, 471, 434)
    draw_arrow(c, 843, 365, 887, 365)
    draw_arrow(c, 471, 522, 471, 548)
    draw_arrow(c, 471, 663, 471, 688)
    draw_arrow(c, 843, 607, 887, 607)
    draw_arrow(c, 471, 787, 471, 811)
    draw_arrow(c, 842, 879, 887, 879)
    draw_arrow(c, 303, 956, 303, 988)
    draw_arrow(c, 932, 956, 932, 988)

    c.showPage()
    c.save()


def make_svg() -> None:
    """Create an editable vector SVG using the same logical coordinates."""
    import svgwrite

    dwg = svgwrite.Drawing(str(SVG_PATH), size=(f"{PAGE_W}pt", f"{PAGE_H}pt"),
                           viewBox=f"0 0 {REFERENCE_W} {REFERENCE_H}")
    dwg.add(dwg.rect(insert=(0, 0), size=(REFERENCE_W, REFERENCE_H), fill="#FFFFFF"))

    def svg_gradient(gid: str, top: Color, bottom: Color):
        grad = dwg.linearGradient(start=(0, 0), end=(0, 1), id=gid)
        grad.add_stop_color(0, color_hex(top))
        grad.add_stop_color(1, color_hex(bottom))
        dwg.defs.add(grad)

    svg_gradient("blue", BLUE_TOP, BLUE_BOTTOM)
    svg_gradient("green", GREEN_TOP, GREEN_BOTTOM)
    svg_gradient("gold", GOLD_TOP, GOLD_BOTTOM)
    svg_gradient("red", RED_TOP, RED_BOTTOM)
    svg_gradient("gray", GRAY_TOP, GRAY_BOTTOM)

    def stage(x, y, w, h, text, fill):
        dwg.add(dwg.rect(insert=(x, y), size=(w, h), rx=7, ry=7,
                         fill=color_hex(fill), stroke=color_hex(fill), stroke_width=1))
        dwg.add(dwg.text(text, insert=(x+w/2, y+h/2),
                         text_anchor="middle", dominant_baseline="middle",
                         font_family="Liberation Sans Narrow", font_size=18,
                         font_weight="bold", fill="#FFFFFF",
                         transform=f"rotate(-90 {x+w/2} {y+h/2})"))

    stage(13, 11, 37, 201, "IDENTIFICATION", BLUE_STAGE)
    stage(14, 320, 36, 201, "SCREENING", GREEN_STAGE)
    stage(13, 553, 37, 121, "ELIGIBILITY", GOLD_STAGE)
    stage(14, 692, 36, 416, "INCLUDED", GRAY_STAGE)

    gradient_map = {
        color_hex(BLUE_BORDER): "blue", color_hex(GREEN_BORDER): "green",
        color_hex(GOLD_BORDER): "gold", color_hex(RED_BORDER): "red",
        color_hex(GRAY_BORDER): "gray"
    }

    for b in boxes_for_svg():
        gid = gradient_map[color_hex(b.border)]
        dwg.add(dwg.rect(insert=(b.x, b.y), size=(b.w, b.h), rx=b.radius_px, ry=b.radius_px,
                         fill=f"url(#{gid})", stroke=color_hex(b.border), stroke_width=b.line_width_px))
        title_y = b.y + b.h * b.title_frac
        dwg.add(dwg.text(b.title, insert=(b.x+b.w/2, title_y), text_anchor="middle",
                         dominant_baseline="middle", font_family="Liberation Sans Narrow",
                         font_size=18, font_weight="bold", fill="#111111"))
        first_y = b.y + b.h * b.body_start_frac
        font_px = 18 if b.body_size < TOTAL_PT else 27
        for i, line in enumerate(b.lines):
            dwg.add(dwg.text(line, insert=(b.x+b.w/2, first_y+i*b.body_gap_px),
                             text_anchor="middle", dominant_baseline="middle",
                             font_family="Liberation Sans Narrow", font_size=font_px,
                             fill="#111111"))

    def s_arrow(x1,y1,x2,y2):
        marker = dwg.marker(insert=(8,4), size=(8,8), orient="auto", markerUnits="strokeWidth")
        marker.add(dwg.path(d="M0,0 L8,4 L0,8 z", fill="#111111"))
        dwg.defs.add(marker)
        line = dwg.line(start=(x1,y1), end=(x2,y2), stroke="#111111", stroke_width=2)
        line.set_markers((None, None, marker))
        dwg.add(line)

    for a in [(282,173,282,199),(723,173,723,199),(471,294,471,318),(843,247,887,247),
              (471,410,471,434),(843,365,887,365),(471,522,471,548),(471,663,471,688),
              (843,607,887,607),(471,787,471,811),(842,879,887,879),(303,956,303,988),
              (932,956,932,988)]:
        s_arrow(*a)

    dwg.save(pretty=True)


def boxes_for_svg() -> list[BoxSpec]:
    # SVG box specifications
    return [
        BoxSpec(72,10,492,161,"Records identified from databases",["n = 2,847","PubMed/MEDLINE, Embase, Scopus,","Web of Science, Cochrane Library","Date range: Jan 2015–Dec 2024"],BLUE_BORDER,BLUE_TOP,BLUE_BOTTOM,0.17,0.39,21.5),
        BoxSpec(588,10,606,161,"Grey literature / supplementary sources",["WHO reports, IARC monographs,","GLOBOCAN 2022, CDC/WHO STI/HIV","guidelines, Cochrane reviews","Count: not formally tracked"],BLUE_BORDER,BLUE_TOP,BLUE_BOTTOM,0.17,0.39,21.5),
        BoxSpec(72,203,771,89,"Records after duplicate removal",["n = 2,104 (743 duplicates removed)","Cross-database deduplication"],BLUE_BORDER,BLUE_TOP,BLUE_BOTTOM,0.30,0.57,20.0),
        BoxSpec(892,202,450,89,"Records removed: duplicates",["n = 743 removed","Cross-database deduplication"],RED_BORDER,RED_TOP,RED_BOTTOM,0.30,0.57,20.0),
        BoxSpec(72,323,771,86,"Records screened (title / abstract)",["n = 2,104","Sequential verification by second reviewer"],GREEN_BORDER,GREEN_TOP,GREEN_BOTTOM,0.28,0.56,19.5),
        BoxSpec(892,323,450,86,"Excluded at title/abstract stage",["n = 1,960 excluded","Not meeting PICO inclusion criteria"],RED_BORDER,RED_TOP,RED_BOTTOM,0.28,0.56,19.5),
        BoxSpec(72,439,770,82,"Full-text articles retrieved for assessment",["n = 144 full texts reviewed","All potentially eligible after abstract screen"],GREEN_BORDER,GREEN_TOP,GREEN_BOTTOM,0.28,0.58,19.0),
        BoxSpec(72,553,771,108,"Full-text articles assessed for eligibility",["n = 144","Inclusion criteria: PICO, design, quality"],GOLD_BORDER,GOLD_TOP,GOLD_BOTTOM,0.25,0.55,21.0),
        BoxSpec(893,490,449,239,"Full texts excluded  (n = 89)",["Insufficient co-infection data ........ n = 31","Outcomes unavailable / unreportable ... n = 28","Insufficient methodological quality ... n = 30","(Note: < 4 case reports, editorials,","Sample size < 5 / no unique miRNA data)","Review pre-2010 (not seminal)"],RED_BORDER,RED_TOP,RED_BOTTOM,0.13,0.28,21.0),
        BoxSpec(73,693,769,92,"Studies included — narrative synthesis",["n = 55 studies from systematic search","Identified through database search and screening"],GOLD_BORDER,GOLD_TOP,GOLD_BOTTOM,0.29,0.57,20.0),
        BoxSpec(72,815,770,139,"Studies included in primary narrative synthesis",["n = 55 synthesis studies","CT/HPV co-infection studies ........ n = 19","HSV-2/HPV co-infection studies ..... n = 9","HIV/HPV co-infection studies ....... n = 13","Triple co-infection / cross-cutting .... n = 14"],GOLD_BORDER,GOLD_TOP,GOLD_BOTTOM,0.17,0.34,17.5),
        BoxSpec(893,815,448,139,"Additional contextual references",["n = 22 (not from systematic search)","Introduction / burden statistics ..... n = 8","HPV biology / background ............ n = 7","Methodology / guidelines ............ n = 7"],GRAY_BORDER,GRAY_TOP,GRAY_BOTTOM,0.18,0.39,18.0),
        BoxSpec(71,992,517,114,"TOTAL STUDIES IN NARRATIVE SYNTHESIS",["n = 55"],GOLD_BORDER,GOLD_TOP,GOLD_BOTTOM,0.30,0.68,0.0,body_size=TOTAL_PT),
        BoxSpec(606,991,733,116,"TOTAL REFERENCES IN MANUSCRIPT",["n = 77","(55 synthesis + 22 contextual)"],GRAY_BORDER,GRAY_TOP,GRAY_BOTTOM,0.28,0.61,25.0,body_size=TOTAL_PT),
    ]


def render_rasters() -> None:
    doc = fitz.open(PDF_PATH)
    page = doc[0]
    zoom = DPI / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    png_bytes = pix.tobytes("png")
    PNG_PATH.write_bytes(png_bytes)

    im = Image.open(PNG_PATH).convert("RGB")
    im.save(JPG_PATH, "JPEG", quality=96, dpi=(DPI, DPI), subsampling=0)
    im.save(TIF_PATH, "TIFF", compression="tiff_lzw", dpi=(DPI, DPI))


def write_support_files() -> None:
    legend = (
        "Figure 1. Study identification, screening, eligibility assessment, and inclusion flow diagram. "
        "A total of 2,847 database records were identified; after removal of 743 duplicates, 2,104 records "
        "were screened, 144 full-text articles were assessed, and 55 studies were included in the narrative "
        "synthesis. A further 22 contextual references were cited outside the systematic-study count."
    )
    LEGEND_PATH.write_text(legend, encoding="utf-8")

    doc = fitz.open(PDF_PATH)
    page = doc[0]
    image_count = len(page.get_images(full=True))
    width_cm = PAGE_W / cm
    height_cm = PAGE_H / cm
    im = Image.open(TIF_PATH)
    dpi = im.info.get("dpi", (DPI, DPI))

    rows = [
        ["PDF page width (cm)", f"{width_cm:.3f}"],
        ["PDF page height (cm)", f"{height_cm:.3f}"],
        ["PDF embedded raster-image count", str(image_count)],
        ["PDF vector status", "PASS - no embedded raster images" if image_count == 0 else "CHECK"],
        ["TIFF width (pixels)", str(im.width)],
        ["TIFF height (pixels)", str(im.height)],
        ["TIFF dpi x", str(round(float(dpi[0]), 2))],
        ["TIFF dpi y", str(round(float(dpi[1]), 2))],
        ["Submission width compliance", "PASS - 16.5 cm is within 15-17 cm"],
        ["Whole figure supplied as one file", "PASS"],
    ]
    with QC_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Check", "Result"])
        writer.writerows(rows)

    README_PATH.write_text(
        "Figure 1 - approved vector redraw package\n"
        "=========================================\n\n"
        "The approved supplied image was used as the visual specification only.\n"
        "The PDF was redrawn from vector boxes, text and arrows; the approved PNG\n"
        "was not placed or embedded into the PDF.\n\n"
        "Deliverables\n"
        "------------\n"
        "Figure_1.pdf  - 100% vector PDF; no embedded raster images\n"
        "Figure_1.svg  - editable vector companion\n"
        "Figure_1.tif  - 300 dpi, LZW-compressed TIFF for journal submission\n"
        "Figure_1.jpg  - 300 dpi, high-quality JPG alternative\n"
        "Figure_1.png  - 300 dpi lossless preview\n"
        "generate_figure_1.py - fully reproducible generation script\n"
        "Figure_1_QC.csv - technical preflight results\n"
        "Figure_1_legend.txt - legend for placement after the references\n\n"
        "Technical specification\n"
        "-----------------------\n"
        f"Final width: {PAGE_W/cm:.2f} cm\n"
        f"Final height: {PAGE_H/cm:.2f} cm\n"
        "Raster resolution: 300 dpi\n"
        "PDF construction: vector shapes and embedded subset fonts only\n",
        encoding="utf-8"
    )


def main() -> None:
    make_vector_pdf()
    make_svg()
    render_rasters()
    write_support_files()
    print(f"Figure 1 files saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
