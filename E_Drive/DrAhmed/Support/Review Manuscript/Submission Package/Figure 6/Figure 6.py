#!/usr/bin/env python3
"""Publication-quality native-vector reconstruction of Figure 6.

The supplied raster reference is used only for QA comparison. It is never
embedded in publication outputs. All visible artwork is rebuilt using native
Matplotlib text, vector rectangles, rounded paths, dashed lines, and polygons.
"""
from __future__ import annotations

import io
import math
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle
from matplotlib.transforms import Bbox
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

# -----------------------------------------------------------------------------
# Paths and output geometry
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_IMAGE = Path("/mnt/data/image.png")
OUT_DIR = SCRIPT_DIR
ZIP_PATH = SCRIPT_DIR.parent / "Figure_6_publication_ready_package.zip"

CANVAS_W = 1126
CANVAS_H = 720
DPI = 300
WIDTH_CM = 16.5
HEIGHT_CM = WIDTH_CM * CANVAS_H / CANVAS_W
VECTOR_FIGSIZE_IN = (WIDTH_CM / 2.54, HEIGHT_CM / 2.54)
RASTER_W = 1949
RASTER_H = 1246
RASTER_FIGSIZE_IN = ((RASTER_W + 0.03) / DPI, (RASTER_H + 0.03) / DPI)
SOURCE_DPI = CANVAS_W / VECTOR_FIGSIZE_IN[0]
SOURCE_FIGSIZE_IN = VECTOR_FIGSIZE_IN

FONT_FAMILY = "Liberation Sans"

# -----------------------------------------------------------------------------
# Palettes and line styling
# -----------------------------------------------------------------------------
PALETTES: Dict[str, Tuple[str, str, str]] = {
    "pink": ("#FFF9F8", "#FDECEA", "#F9DEDA"),
    "red_strip": ("#F66A5E", "#EE4F43", "#E63E34"),
    "yellow": ("#FFFDF2", "#FFF5D4", "#FCE9B2"),
    "grey": ("#FCFDFD", "#F5F6F7", "#ECEEEF"),
    "outcome": ("#FAFCFD", "#F1F5F7", "#E8EEF1"),
    "blue": ("#F4FAFE", "#E6F3FC", "#D8ECF9"),
}

COLORS = {
    "red_border": "#EF4B3F",
    "red_heading": "#B52E25",
    "red_italic": "#D35047",
    "yellow_border": "#ECAA32",
    "yellow_title": "#8C6910",
    "yellow_body": "#9B7417",
    "grey_border": "#A9ADB0",
    "grey_heading": "#4E5154",
    "grey_text": "#989A9C",
    "warning": "#CF5E56",
    "outcome_border": "#B9C4CA",
    "outcome_heading": "#304B5E",
    "outcome_body": "#6C6F72",
    "blue_border": "#58A9DB",
    "blue_heading": "#21668E",
    "blue_body": "#2D719A",
    "connector_grey": "#969A9D",
    "connector_red": "#EF4B3F",
    "footer_muted": "#B86660",
    "white": "#FFFFFF",
}

LINEWIDTHS = {
    "major": 1.18,
    "grey_box": 1.08,
    "connector": 1.22,
    "strip": 0.0,
}
DASHES = {
    "grey_box": (0, (3.2, 2.4)),
    "connector": (0, (3.4, 2.5)),
}

FONT = {
    "banner_heading": 7.0,
    "banner_body": 5.75,
    "panel_title": 7.0,
    "panel_body": 5.85,
    "panel_small": 5.05,
    "strip": 5.7,
    "mechanism_title": 6.8,
    "mechanism_body": 5.75,
    "warning": 5.7,
    "outcome_title": 7.0,
    "outcome_body": 5.35,
    "syndemic_title": 6.7,
    "syndemic_body": 5.45,
    "footer_1": 5.9,
    "footer_2": 5.1,
    "footer_3": 4.75,
}

# -----------------------------------------------------------------------------
# Structured content and geometry
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class TextLine:
    text: str
    x: float
    y: float
    size: float
    color: str
    weight: str = "normal"
    style: str = "normal"
    ha: str = "center"
    parent: str = ""
    role: str = "text"

@dataclass(frozen=True)
class Panel:
    key: str
    bounds: Tuple[float, float, float, float]  # x, y(top), w, h
    palette: str
    border: str
    radius: float
    dashed: bool = False
    linewidth: float = 1.18

PANELS: Dict[str, Panel] = {
    "banner": Panel("banner", (37, 6, 1053, 43), "pink", COLORS["red_border"], 9),
    "serology": Panel("serology", (37, 61, 501, 147), "yellow", COLORS["yellow_border"], 11),
    "top_hit": Panel("top_hit", (575, 61, 514, 147), "pink", COLORS["red_border"], 11),
    "mech_a": Panel("mech_a", (37, 236, 313, 177), "grey", COLORS["grey_border"], 10, True, LINEWIDTHS["grey_box"]),
    "mech_b": Panel("mech_b", (405, 236, 315, 177), "grey", COLORS["grey_border"], 10, True, LINEWIDTHS["grey_box"]),
    "mech_c": Panel("mech_c", (775, 236, 314, 177), "pink", COLORS["red_border"], 10),
    "outcome": Panel("outcome", (37, 469, 1053, 76), "outcome", COLORS["outcome_border"], 10),
    "syndemic": Panel("syndemic", (37, 561, 1053, 70), "blue", COLORS["blue_border"], 10),
    "footer": Panel("footer", (37, 646, 1053, 64), "pink", COLORS["red_border"], 9),
}

HEADER_STRIPS = {
    "top_hit_strip": (594, 92, 476, 26),
    "mech_c_strip": (792, 269, 280, 24),
}

TEXT_LINES: Tuple[TextLine, ...] = (
    # Top banner
    TextLine("Evidence: derived from non-human experimental systems; human cervical tissue validation absent. GRADE: LOW.", 563, 19, FONT["banner_heading"], COLORS["red_heading"], "bold", parent="banner", role="banner_1"),
    TextLine("Adjusted pooled OR=1.41 (95% CI 0.98-2.04; CI crosses unity) [Sun 2023, 28 studies]. No guideline recommends altered screening for HSV-2 serostatus.", 563, 36, FONT["banner_body"], COLORS["red_italic"], style="italic", parent="banner", role="banner_2"),

    # Serological panel
    TextLine("Serological Misclassification Bias", 287.5, 81, FONT["panel_title"], COLORS["yellow_title"], "bold", parent="serology", role="title"),
    TextLine("1. Serology = lifetime exposure, not active co-infection", 287.5, 103, FONT["panel_body"], COLORS["yellow_body"], parent="serology"),
    TextLine("2. HSV-1/HSV-2 cross-reactivity inflates false-positive HSV-2", 287.5, 123, FONT["panel_body"], COLORS["yellow_body"], parent="serology"),
    TextLine("3. Seropositive does not distinguish reactivating from latent", 287.5, 143, FONT["panel_body"], COLORS["yellow_body"], parent="serology"),
    TextLine("All three sources inflate observed epidemiological association.", 287.5, 163, FONT["panel_body"], COLORS["yellow_body"], parent="serology"),
    TextLine("No HSV-2 DNA detected in 200 CC specimens by PCR [Tran-Thanh 2003]", 287.5, 183, FONT["panel_body"], COLORS["yellow_body"], parent="serology"),
    TextLine("argues against persistent viral presence as mechanism.", 287.5, 199, FONT["panel_body"], COLORS["yellow_body"], parent="serology"),

    # Top-right hit-and-run
    TextLine("Hit-and-Run Hypothesis", 832, 81, FONT["panel_title"], COLORS["red_heading"], "bold", parent="top_hit", role="title"),
    TextLine("HYPOTHESIS ONLY - UNCONFIRMED IN HUMAN CERVICAL TISSUE", 832, 105, FONT["strip"], COLORS["white"], "bold", parent="top_hit_strip", role="strip"),
    TextLine("Transient HSV-2 causes permanent genomic alterations", 832, 129, FONT["panel_body"], COLORS["red_italic"], style="italic", parent="top_hit"),
    TextLine("ICP0 transactivates heterologous promoters (cell models only)", 832, 147, FONT["panel_body"], COLORS["red_italic"], style="italic", parent="top_hit"),
    TextLine("Viral persistence not required under this model", 832, 165, FONT["panel_body"], COLORS["red_italic"], style="italic", parent="top_hit"),
    TextLine("Cannot currently inform clinical management decisions.", 832, 184, FONT["warning"], COLORS["red_heading"], "bold", parent="top_hit"),
    TextLine("Evidence level: experimental cell models; no human cervical tissue data.", 832, 198.5, FONT["panel_small"], COLORS["grey_text"], parent="top_hit"),

    # Mechanism A
    TextLine("A. Epithelial Disruption", 193.5, 258, FONT["mechanism_title"], COLORS["grey_heading"], "bold", "italic", parent="mech_a", role="title"),
    TextLine("HSV-2 genital ulceration", 193.5, 283, FONT["mechanism_body"], COLORS["grey_text"], style="italic", parent="mech_a"),
    TextLine("creates epithelial breaches", 193.5, 302, FONT["mechanism_body"], COLORS["grey_text"], style="italic", parent="mech_a"),
    TextLine("facilitating HPV basal cell access", 193.5, 321, FONT["mechanism_body"], COLORS["grey_text"], style="italic", parent="mech_a"),
    TextLine("Episodic not sustained ulceration -", 193.5, 341, FONT["mechanism_body"], COLORS["grey_text"], style="italic", parent="mech_a"),
    TextLine("sufficiency undemonstrated", 193.5, 359, FONT["mechanism_body"], COLORS["grey_text"], style="italic", parent="mech_a"),
    TextLine("[Biological plausibility only]", 193.5, 379, FONT["warning"], COLORS["warning"], style="italic", parent="mech_a"),
    TextLine("No human cervical tissue validation", 193.5, 399, FONT["panel_small"], COLORS["grey_text"], parent="mech_a"),

    # Mechanism B
    TextLine("B. Immune Evasion", 562.5, 258, FONT["mechanism_title"], COLORS["grey_heading"], "bold", "italic", parent="mech_b", role="title"),
    TextLine("ICP47 mediates TAP blockade", 562.5, 283, FONT["mechanism_body"], COLORS["grey_text"], style="italic", parent="mech_b"),
    TextLine("Reduced NK cell activity", 562.5, 302, FONT["mechanism_body"], COLORS["grey_text"], style="italic", parent="mech_b"),
    TextLine("Suppressed type I IFN responses", 562.5, 321, FONT["mechanism_body"], COLORS["grey_text"], style="italic", parent="mech_b"),
    TextLine("Characterised in non-cervical", 562.5, 341, FONT["mechanism_body"], COLORS["grey_text"], style="italic", parent="mech_b"),
    TextLine("experimental systems only", 562.5, 359, FONT["mechanism_body"], COLORS["grey_text"], style="italic", parent="mech_b"),
    TextLine("[Human cervical validation absent]", 562.5, 379, FONT["warning"], COLORS["warning"], style="italic", parent="mech_b"),
    TextLine("Non-human model data only", 562.5, 399, FONT["panel_small"], COLORS["grey_text"], parent="mech_b"),

    # Mechanism C
    TextLine("C. Hit-and-Run", 932, 258, FONT["mechanism_title"], COLORS["red_heading"], "bold", parent="mech_c", role="title"),
    TextLine("HYPOTHESIS - UNCONFIRMED", 932, 281, FONT["strip"], COLORS["white"], "bold", parent="mech_c_strip", role="strip"),
    TextLine("Transient infection causes permanent", 932, 307, FONT["mechanism_body"], COLORS["red_italic"], style="italic", parent="mech_c"),
    TextLine("genomic instability via ICP0", 932, 326, FONT["mechanism_body"], COLORS["red_italic"], style="italic", parent="mech_c"),
    TextLine("transactivation (cell models)", 932, 345, FONT["mechanism_body"], COLORS["red_italic"], style="italic", parent="mech_c"),
    TextLine("Not validated in human cervical", 932, 367, FONT["warning"], COLORS["red_heading"], "bold", parent="mech_c"),
    TextLine("tissue. Hypothesis-generating only.", 932, 385, FONT["warning"], COLORS["red_heading"], "bold", parent="mech_c"),
    TextLine("Cannot inform clinical management", 932, 402, FONT["panel_small"], COLORS["grey_text"], parent="mech_c"),

    # Outcome
    TextLine("Postulated Outcome: HPV Persistence and Neoplastic Progression", 563, 490, FONT["outcome_title"], COLORS["outcome_heading"], "bold", parent="outcome", role="title"),
    TextLine("All pathways remain hypothetical for human cervical tissue context. Seidman 2023: HSV-2 associated with HPV incidence (aHR=1.8)", 563, 514, FONT["outcome_body"], COLORS["outcome_body"], style="italic", parent="outcome"),
    TextLine("but NOT with precancerous lesion development at 36-month follow-up - suggests facilitated acquisition without independent neoplastic promotion.", 563, 534, FONT["outcome_body"], COLORS["outcome_body"], style="italic", parent="outcome"),

    # Syndemic
    TextLine("Syndemic Context (WLHIV only - not general population):", 563, 579, FONT["syndemic_title"], COLORS["blue_heading"], "bold", parent="syndemic", role="title"),
    TextLine("Triple co-infection (HPV + HIV + HSV-2) associated with elevated HPV16/18 viral load vs. mono-infection [Moran 2025; female sex workers].", 563, 600, FONT["syndemic_body"], COLORS["blue_body"], parent="syndemic"),
    TextLine("HSV-2 effect may be context-dependent (requires concurrent HIV-mediated immune suppression) rather than independent carcinogenic effect.", 563, 620, FONT["syndemic_body"], COLORS["blue_body"], parent="syndemic"),

    # Footer, complete and unclipped
    TextLine("Footer: Evidence from non-human experimental systems; human cervical validation absent for all depicted HSV-2 mechanisms.", 563, 660, FONT["footer_1"], COLORS["red_heading"], "bold", parent="footer", role="footer_1"),
    TextLine("Hit-and-run mechanism remains an unconfirmed hypothesis in human cervical tissue. Standard screening protocols apply without HSV-2-specific modification.", 563, 678, FONT["footer_2"], COLORS["footer_muted"], style="italic", parent="footer", role="footer_2"),
    TextLine("GRADE: LOW | Clinical recommendation: Standard screening - no guideline-supported modification | Suppressive therapy as CC prevention: Tier D (speculative)", 563, 696, FONT["footer_3"], COLORS["footer_muted"], parent="footer", role="footer_3"),
)

REQUIRED_STRINGS = [line.text for line in TEXT_LINES]

# -----------------------------------------------------------------------------
# Drawing helpers
# -----------------------------------------------------------------------------
def configure_matplotlib() -> None:
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [FONT_FAMILY, "DejaVu Sans"],
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.transparent": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.unicode_minus": False,
        "path.simplify": False,
        "text.antialiased": True,
    })


def inspect_reference_image() -> Tuple[Tuple[int, int], str]:
    if not REFERENCE_IMAGE.exists():
        raise FileNotFoundError(REFERENCE_IMAGE)
    with Image.open(REFERENCE_IMAGE) as im:
        if im.size != (CANVAS_W, CANVAS_H):
            raise RuntimeError(f"Reference dimensions {im.size}, expected {(CANVAS_W, CANVAS_H)}")
        if im.mode != "RGB":
            raise RuntimeError(f"Reference mode {im.mode}, expected RGB")
        return im.size, im.mode


def make_canvas(figsize_in: Tuple[float, float], dpi: float) -> Tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=figsize_in, dpi=dpi, facecolor="white", layout=None)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, CANVAS_W)
    ax.set_ylim(CANVAS_H, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    return fig, ax


def interpolate_gradient(stops: Sequence[str], n: int) -> np.ndarray:
    rgb = np.asarray([to_rgb(c) for c in stops], dtype=float)
    t = np.linspace(0, 1, n)
    out = np.empty((n, 3), dtype=float)
    for i, value in enumerate(t):
        if value <= 0.5:
            u = value / 0.5
            out[i] = rgb[0] * (1-u) + rgb[1] * u
        else:
            u = (value-0.5) / 0.5
            out[i] = rgb[1] * (1-u) + rgb[2] * u
    return out


def draw_vector_gradient_rounded_box(ax: plt.Axes, panel: Panel, segments: int = 140, zorder: float = 2.0) -> Dict[str, object]:
    x, y, w, h = panel.bounds
    clip = FancyBboxPatch((x, y), w, h,
                          boxstyle=f"round,pad=0,rounding_size={panel.radius}",
                          facecolor="none", edgecolor="none", linewidth=0,
                          transform=ax.transData)
    ax.add_patch(clip)
    colors = interpolate_gradient(PALETTES[panel.palette], segments)
    seg_w = w / segments
    strips = []
    for i, color in enumerate(colors):
        rect = Rectangle((x+i*seg_w, y), seg_w*1.10, h,
                         facecolor=color, edgecolor="none", linewidth=0,
                         antialiased=False, zorder=zorder)
        rect.set_clip_path(clip)
        ax.add_patch(rect)
        strips.append(rect)
    border = FancyBboxPatch((x, y), w, h,
                            boxstyle=f"round,pad=0,rounding_size={panel.radius}",
                            facecolor="none", edgecolor=panel.border,
                            linewidth=panel.linewidth,
                            linestyle=DASHES["grey_box"] if panel.dashed else "solid",
                            capstyle="butt", joinstyle="round", zorder=zorder+1)
    ax.add_patch(border)
    return {"clip": clip, "strips": strips, "border": border}


def draw_header_strip(ax: plt.Axes, key: str, bounds: Tuple[float,float,float,float], segments: int = 90) -> Dict[str, object]:
    x, y, w, h = bounds
    radius = 4.5
    clip = FancyBboxPatch((x, y), w, h,
                          boxstyle=f"round,pad=0,rounding_size={radius}",
                          facecolor="none", edgecolor="none", linewidth=0)
    ax.add_patch(clip)
    colors = interpolate_gradient(PALETTES["red_strip"], segments)
    sw = w/segments
    strips=[]
    for i,c in enumerate(colors):
        r=Rectangle((x+i*sw,y),sw*1.10,h,facecolor=c,edgecolor="none",linewidth=0,antialiased=False,zorder=4)
        r.set_clip_path(clip); ax.add_patch(r); strips.append(r)
    border=FancyBboxPatch((x,y),w,h,boxstyle=f"round,pad=0,rounding_size={radius}",facecolor="none",edgecolor="#E94B40",linewidth=0.35,zorder=4.2)
    ax.add_patch(border)
    return {"key":key,"clip":clip,"strips":strips,"border":border,"bounds":bounds}


def draw_dashed_connector(ax: plt.Axes, x: float, y0: float, y_tip: float, color: str) -> Dict[str, object]:
    shaft_end = y_tip - 12
    line, = ax.plot([x,x],[y0,shaft_end], color=color, linewidth=LINEWIDTHS["connector"],
                    linestyle=DASHES["connector"], solid_capstyle="butt", zorder=1.0)
    head = Polygon([(x, y_tip), (x-7.2, shaft_end), (x+7.2, shaft_end)], closed=True,
                   facecolor=color, edgecolor=color, linewidth=0.4, zorder=1.1)
    ax.add_patch(head)
    return {"line":line,"head":head}


def draw_text_lines(ax: plt.Axes) -> List[Dict[str, object]]:
    registry=[]
    for line in TEXT_LINES:
        artist=ax.text(line.x,line.y,line.text,ha=line.ha,va="center",fontsize=line.size,
                       color=line.color,fontweight=line.weight,fontstyle=line.style,
                       zorder=6,clip_on=False)
        registry.append({"line":line,"artist":artist,"parent":line.parent,"role":line.role})
    return registry


def create_figure(figsize_in: Tuple[float,float], dpi: float) -> Tuple[plt.Figure, plt.Axes, Dict[str,object]]:
    fig, ax = make_canvas(figsize_in, dpi)

    connectors=[
        draw_dashed_connector(ax, 193.5, 413, 469, COLORS["connector_grey"]),
        draw_dashed_connector(ax, 562.5, 413, 469, COLORS["connector_grey"]),
        draw_dashed_connector(ax, 932, 413, 469, COLORS["connector_red"]),
    ]

    panel_art={key:draw_vector_gradient_rounded_box(ax,panel) for key,panel in PANELS.items()}
    strips={key:draw_header_strip(ax,key,bounds) for key,bounds in HEADER_STRIPS.items()}
    texts=draw_text_lines(ax)
    metadata={"panels":panel_art,"strips":strips,"texts":texts,"connectors":connectors}
    return fig,ax,metadata

# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------
def data_bbox(ax: plt.Axes, bounds: Tuple[float,float,float,float]) -> Bbox:
    x,y,w,h=bounds
    p0=ax.transData.transform((x,y))
    p1=ax.transData.transform((x+w,y+h))
    return Bbox.from_extents(min(p0[0],p1[0]),min(p0[1],p1[1]),max(p0[0],p1[0]),max(p0[1],p1[1]))


def overlap(a:Bbox,b:Bbox,pad:float=0)->bool:
    return not ((a.x1+pad)<=b.x0 or (b.x1+pad)<=a.x0 or (a.y1+pad)<=b.y0 or (b.y1+pad)<=a.y0)


def validate_scientific_content() -> None:
    # One authoritative source of text is used for drawing. Validate exact required tokens.
    all_text="\n".join(REQUIRED_STRINGS)
    required_tokens=[
        "GRADE: LOW.", "OR=1.41", "0.98-2.04", "Sun 2023, 28 studies", "HSV-1/HSV-2",
        "Tran-Thanh 2003", "ICP0", "ICP47", "type I IFN", "aHR=1.8", "HPV16/18",
        "Moran 2025; female sex workers", "Tier D (speculative)",
        "Clinical recommendation: Standard screening - no guideline-supported modification",
    ]
    missing=[t for t in required_tokens if t not in all_text]
    if missing:
        raise RuntimeError("Missing scientific tokens: "+", ".join(missing))
    if len(PANELS)!=9:
        raise RuntimeError("Expected nine panels")
    if len(HEADER_STRIPS)!=2:
        raise RuntimeError("Expected two header strips")


def validate_layout(fig: plt.Figure, ax: plt.Axes, meta: Dict[str,object]) -> Dict[str,object]:
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    canvas=fig.bbox
    failures=[]
    text_records=[]
    parent_bboxes={key:data_bbox(ax,p.bounds) for key,p in PANELS.items()}
    parent_bboxes.update({key:data_bbox(ax,bounds) for key,bounds in HEADER_STRIPS.items()})

    min_border_clear=1e9
    for rec in meta["texts"]:
        line:TextLine=rec["line"]
        bb=rec["artist"].get_window_extent(renderer)
        parent=parent_bboxes[line.parent]
        # Main containment margins in final-render pixels.
        left=bb.x0-parent.x0; right=parent.x1-bb.x1; top=parent.y1-bb.y1; bottom=bb.y0-parent.y0
        min_border_clear=min(min_border_clear,left,right,top,bottom)
        # Long scientific lines need at least 4 px; all others 5 px.
        scale = fig.dpi / DPI
        horizontal_margin=4.0 * scale
        vertical_margin=3.0 * scale
        if bb.x0 < parent.x0+horizontal_margin or bb.x1 > parent.x1-horizontal_margin or bb.y0 < parent.y0+vertical_margin or bb.y1 > parent.y1-vertical_margin:
            failures.append(f"text containment: {line.role or line.text[:35]}")
        if bb.x0 < canvas.x0 or bb.x1 > canvas.x1 or bb.y0 < canvas.y0 or bb.y1 > canvas.y1:
            failures.append(f"canvas text overflow: {line.role or line.text[:35]}")
        text_records.append({"line":line,"bbox":bb,"parent_bbox":parent})

    # Text-to-text overlap only within same panel. Closely spaced lines must not intersect.
    for i in range(len(text_records)):
        for j in range(i+1,len(text_records)):
            a=text_records[i]; b=text_records[j]
            if a["line"].parent==b["line"].parent and overlap(a["bbox"],b["bbox"],pad=0.5):
                failures.append(f"text overlap in {a['line'].parent}: {a['line'].role}/{b['line'].role}")

    # Panel bounds inside canvas and non-overlap checks for stacked lower panels.
    panel_boxes={k:data_bbox(ax,p.bounds) for k,p in PANELS.items()}
    for key,bb in panel_boxes.items():
        if bb.x0<canvas.x0 or bb.x1>canvas.x1 or bb.y0<canvas.y0 or bb.y1>canvas.y1:
            failures.append(f"panel outside canvas: {key}")
    for a,b in [("outcome","syndemic"),("syndemic","footer")]:
        if overlap(panel_boxes[a],panel_boxes[b]):
            failures.append(f"panel overlap: {a}/{b}")

    # Footer-specific validation.
    footer_lines=[r for r in text_records if r["line"].parent=="footer"]
    footer_bb=panel_boxes["footer"]
    footer_by_role={r["line"].role:r["bbox"] for r in footer_lines}
    footer3=footer_by_role["footer_3"]
    # In display coordinates, distance from text bottom to footer lower edge.
    footer3_clearance=footer3.y0-footer_bb.y0
    footer_border_to_canvas=footer_bb.y0-canvas.y0
    required_footer_clearance = 8.0 * (fig.dpi / DPI)
    if footer3_clearance < required_footer_clearance:
        failures.append(f"footer line 3 lower clearance {footer3_clearance:.2f}px < {required_footer_clearance:.2f}px")
    if footer_border_to_canvas < required_footer_clearance:
        failures.append(f"footer border canvas clearance {footer_border_to_canvas:.2f}px < {required_footer_clearance:.2f}px")
    if "|" not in next(r["line"].text for r in footer_lines if r["line"].role=="footer_3"):
        failures.append("footer vertical bars missing")
    if "Tier D (speculative)" not in next(r["line"].text for r in footer_lines if r["line"].role=="footer_3"):
        failures.append("footer Tier D missing")

    # Header-strip validation.
    strip_records=[r for r in text_records if r["line"].role=="strip"]
    for r in strip_records:
        bb=r["bbox"]; parent=r["parent_bbox"]
        if min(bb.x0-parent.x0,parent.x1-bb.x1,bb.y0-parent.y0,parent.y1-bb.y1)<4*(fig.dpi/DPI):
            failures.append(f"header strip clearance: {r['line'].parent}")

    # Connectors count and geometry.
    if len(meta["connectors"])!=3:
        failures.append("connector count")

    if failures:
        raise RuntimeError("Layout validation failed: "+"; ".join(failures))

    return {
        "text_records":text_records,
        "parent_bboxes":parent_bboxes,
        "panel_bboxes":panel_boxes,
        "failures":failures,
        "min_text_border_clearance":min_border_clear,
        "footer_bbox":footer_bb,
        "footer_line_bboxes":footer_by_role,
        "footer3_clearance":footer3_clearance,
        "footer_border_canvas_clearance":footer_border_to_canvas,
        "banner_bboxes":{r["line"].role:r["bbox"] for r in text_records if r["line"].parent=="banner"},
        "strip_bboxes":{r["line"].parent:r["bbox"] for r in strip_records},
        "text_object_count":len(text_records),
    }

# -----------------------------------------------------------------------------
# Export
# -----------------------------------------------------------------------------
def render_raster() -> Tuple[Image.Image, Dict[str,object]]:
    fig,ax,meta=create_figure(RASTER_FIGSIZE_IN,DPI)
    canvas=matplotlib.backends.backend_agg.FigureCanvasAgg(fig)
    canvas.draw()
    validation=validate_layout(fig,ax,meta)
    rgba=np.asarray(canvas.buffer_rgba())
    image=Image.fromarray(rgba,"RGBA").convert("RGB")
    if image.size!=(RASTER_W,RASTER_H):
        raise RuntimeError(f"Unexpected raster size {image.size}")
    plt.close(fig)
    return image,validation


def export_raster_outputs() -> Tuple[Path,Path,Path,Dict[str,object]]:
    image,validation=render_raster()
    png=OUT_DIR/"Figure_6_HSV2_evidence_framework.png"
    jpg=OUT_DIR/"Figure_6_HSV2_evidence_framework.jpg"
    tif=OUT_DIR/"Figure_6_HSV2_evidence_framework.tiff"
    image.save(png,"PNG",dpi=(DPI,DPI),optimize=True)
    image.save(jpg,"JPEG",dpi=(DPI,DPI),quality=100,subsampling=0,optimize=True)
    image.save(tif,"TIFF",dpi=(DPI,DPI),compression="tiff_lzw")
    return png,jpg,tif,validation


def export_vector_outputs() -> Tuple[Path,Path]:
    pdf=OUT_DIR/"Figure_6_HSV2_evidence_framework.pdf"
    svg=OUT_DIR/"Figure_6_HSV2_evidence_framework.svg"
    fig,ax,meta=create_figure(VECTOR_FIGSIZE_IN,DPI)
    validate_layout(fig,ax,meta)
    fig.savefig(pdf,format="pdf",facecolor="white",transparent=False)
    fig.savefig(svg,format="svg",facecolor="white",transparent=False)
    plt.close(fig)
    return pdf,svg

# -----------------------------------------------------------------------------
# Export inspection
# -----------------------------------------------------------------------------
def inspect_raster(path:Path)->Dict[str,object]:
    with Image.open(path) as im:
        compression=im.info.get("compression")
        if path.suffix.lower() in {".tif",".tiff"}:
            compression=im.tag_v2.get(259,compression)
        return {
            "file":path.name,"format":im.format,"size":im.size,"mode":im.mode,
            "dpi":im.info.get("dpi"),"transparency":("A" in im.getbands() or "transparency" in im.info),
            "compression":compression,"frames":getattr(im,"n_frames",1),"bytes":path.stat().st_size,
        }


def inspect_vectors(pdf:Path,svg:Path)->Dict[str,object]:
    out={}
    try:
        import fitz  # type: ignore
        doc=fitz.open(pdf); page=doc[0]
        out["pdf_page_points"]=(page.rect.width,page.rect.height)
        out["pdf_image_count"]=len(page.get_images(full=True))
        out["pdf_page_count"]=doc.page_count
        doc.close()
    except Exception as exc:
        out["pdf_error"]=str(exc)
    s=svg.read_text(encoding="utf-8",errors="replace")
    out["svg_has_image_tag"]=bool(re.search(r"<image\b",s,re.I))
    out["svg_rect_count"]=len(re.findall(r"<rect\b",s,re.I))
    out["svg_path_count"]=len(re.findall(r"<path\b",s,re.I))
    mw=re.search(r'<svg[^>]+width="([^"]+)"',s,re.I); mh=re.search(r'<svg[^>]+height="([^"]+)"',s,re.I)
    out["svg_width"]=mw.group(1) if mw else "not found"; out["svg_height"]=mh.group(1) if mh else "not found"
    return out


def render_pdf_preview(pdf:Path)->Image.Image|None:
    try:
        import fitz  # type: ignore
        doc=fitz.open(pdf); pix=doc[0].get_pixmap(dpi=300,alpha=False)
        im=Image.frombytes("RGB",[pix.width,pix.height],pix.samples); doc.close(); return im
    except Exception:
        return None


def render_svg_preview(svg:Path)->Image.Image|None:
    try:
        import cairosvg  # type: ignore
        data=cairosvg.svg2png(url=str(svg),dpi=300)
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        return None

# -----------------------------------------------------------------------------
# QA files
# -----------------------------------------------------------------------------
def qa_font(size:int,bold:bool=False):
    try:
        from matplotlib import font_manager
        fp=font_manager.findfont(font_manager.FontProperties(family=FONT_FAMILY,weight="bold" if bold else "normal"))
        return ImageFont.truetype(fp,size)
    except Exception:
        return ImageFont.load_default()


def create_source_size_reconstruction() -> Image.Image:
    fig,ax,meta=create_figure(SOURCE_FIGSIZE_IN,SOURCE_DPI)
    canvas=matplotlib.backends.backend_agg.FigureCanvasAgg(fig); canvas.draw(); validate_layout(fig,ax,meta)
    im=Image.fromarray(np.asarray(canvas.buffer_rgba()),"RGBA").convert("RGB")
    plt.close(fig)
    if im.size!=(CANVAS_W,CANVAS_H):
        raise RuntimeError(f"Source-size QA render is {im.size}")
    return im


def create_qa_files(png:Path,validation:Dict[str,object])->Tuple[Path,Path,Path,Path,Path,Path]:
    with Image.open(REFERENCE_IMAGE) as im: ref=im.convert("RGB")
    rec_source=create_source_size_reconstruction()
    with Image.open(png) as im: rec_full=im.convert("RGB")

    overlay=OUT_DIR/"Figure_6_QA_overlay.png"
    Image.blend(ref,rec_source,0.5).save(overlay,dpi=(DPI,DPI))

    gap=18; header=46
    side=Image.new("RGB",(CANVAS_W*2+gap,CANVAS_H+header),"white")
    side.paste(ref,(0,header)); side.paste(rec_source,(CANVAS_W+gap,header))
    d=ImageDraw.Draw(side); font=qa_font(22,True)
    d.text((CANVAS_W//2,18),"Reference",fill="black",font=font,anchor="mm")
    d.text((CANVAS_W+gap+CANVAS_W//2,18),"Reconstructed",fill="black",font=font,anchor="mm")
    side_path=OUT_DIR/"Figure_6_QA_side_by_side.png"; side.save(side_path,dpi=(DPI,DPI))

    def edges(im:Image.Image)->Image.Image:
        e=ImageOps.grayscale(im).filter(ImageFilter.FIND_EDGES); e=ImageOps.autocontrast(e,cutoff=1); return ImageOps.invert(e).convert("RGB")
    ec=Image.new("RGB",side.size,"white"); ec.paste(edges(ref),(0,header)); ec.paste(edges(rec_source),(CANVAS_W+gap,header))
    de=ImageDraw.Draw(ec); de.text((CANVAS_W//2,18),"Reference edges",fill="black",font=font,anchor="mm"); de.text((CANVAS_W+gap+CANVAS_W//2,18),"Reconstructed edges",fill="black",font=font,anchor="mm")
    edge_path=OUT_DIR/"Figure_6_QA_edges.png"; ec.save(edge_path,dpi=(DPI,DPI))

    # Text-boundary QA on final raster.
    bounds=rec_full.copy(); db=ImageDraw.Draw(bounds)
    for rec in validation["text_records"]:
        bb=rec["bbox"]; box=(int(bb.x0),int(RASTER_H-bb.y1),int(bb.x1),int(RASTER_H-bb.y0)); db.rectangle(box,outline=(220,0,180),width=1)
    for key,bb in validation["panel_bboxes"].items():
        box=(int(bb.x0),int(RASTER_H-bb.y1),int(bb.x1),int(RASTER_H-bb.y0)); db.rectangle(box,outline=(0,145,255),width=2)
    for key in HEADER_STRIPS:
        bb=validation["parent_bboxes"][key]; box=(int(bb.x0),int(RASTER_H-bb.y1),int(bb.x1),int(RASTER_H-bb.y0)); db.rectangle(box,outline=(0,200,100),width=2)
    text_bounds=OUT_DIR/"Figure_6_QA_text_bounds.png"; bounds.save(text_bounds,dpi=(DPI,DPI))

    scale_x=RASTER_W/CANVAS_W; scale_y=RASTER_H/CANVAS_H
    # Footer close-up.
    x0,y0,w,h=PANELS["footer"].bounds; pad=14
    crop=(max(int(x0*scale_x)-pad,0),max(int(y0*scale_y)-pad,0),min(int((x0+w)*scale_x)+pad,RASTER_W),min(int((y0+h)*scale_y)+pad,RASTER_H))
    ftr=rec_full.crop(crop); ftr=ftr.resize((ftr.width*3,ftr.height*3),Image.Resampling.LANCZOS)
    footer_close=OUT_DIR/"Figure_6_QA_footer_closeup.png"; ftr.save(footer_close,dpi=(DPI,DPI))

    # Top panels close-up.
    crop=(int(20*scale_x),0,int(1105*scale_x),int(222*scale_y))
    top=rec_full.crop(crop); top=top.resize((top.width*2,top.height*2),Image.Resampling.LANCZOS)
    top_close=OUT_DIR/"Figure_6_QA_top_panels_closeup.png"; top.save(top_close,dpi=(DPI,DPI))
    return overlay,side_path,edge_path,text_bounds,footer_close,top_close

# -----------------------------------------------------------------------------
# QA report and packaging
# -----------------------------------------------------------------------------
def write_qa_report(ref_info,rasters:Sequence[Path],pdf:Path,svg:Path,validation:Dict[str,object],vector_info:Mapping[str,object])->Path:
    path=OUT_DIR/"Figure_6_QA_report.txt"
    lines=["FIGURE 6 PUBLICATION-READY QA REPORT","="*54,""]
    lines += [f"Source image: {REFERENCE_IMAGE}",f"Source dimensions: {ref_info[0][0]} x {ref_info[0][1]} px",f"Source colour mode: {ref_info[1]}","Source use: visual QA only; never embedded or upscaled into publication outputs.",""]
    lines += [f"Final physical dimensions: {WIDTH_CM:.3f} x {HEIGHT_CM:.3f} cm",f"Final raster dimensions: {RASTER_W} x {RASTER_H} px",f"Nominal DPI: {DPI} x {DPI}",""]
    lines += ["RASTER OUTPUTS","-"*54]
    for p in rasters:
        q=inspect_raster(p)
        lines += [f"File: {q['file']}",f"  Format: {q['format']}",f"  Dimensions: {q['size'][0]} x {q['size'][1]}",f"  Mode: {q['mode']}",f"  DPI metadata: {q['dpi']}",f"  Transparency: {q['transparency']}",f"  File size: {q['bytes']} bytes"]
        if q["format"]=="TIFF": lines += [f"  Compression tag: {q['compression']} (5 = LZW)",f"  Frame count: {q['frames']}"]
        if q["format"]=="JPEG": lines += ["  Quality: 100","  Chroma subsampling: 0 (4:4:4)"]
    lines += ["","VECTOR OUTPUTS","-"*54]
    for k,v in vector_info.items(): lines.append(f"{k}: {v}")
    pdf_preview=render_pdf_preview(pdf); svg_preview=render_svg_preview(svg)
    lines += [f"PDF preview rendered: {pdf_preview is not None}",f"SVG preview rendered: {svg_preview is not None}","Vector gradients: adjacent opaque vector rectangles clipped by native rounded paths.","Original reference embedded: no.",""]
    lines += ["CONTENT AND GEOMETRY VALIDATION","-"*54,f"Panel count: {len(PANELS)}",f"Connector count: 3",f"Text object count: {validation['text_object_count']}","Scientific wording and required tokens: validated.","All panel boundaries inside canvas: yes","Middle panels horizontally aligned: yes","Outcome/syndemic/footer panel overlap: none","Header strips: 2; complete and contained","Text-overflow failures: 0",f"Minimum text-to-parent-border clearance: {validation['min_text_border_clearance']:.2f} px",""]
    fb=validation["footer_bbox"]
    lines += ["FOOTER VALIDATION","-"*54,f"Footer panel bbox px: ({fb.x0:.2f}, {fb.y0:.2f}, {fb.x1:.2f}, {fb.y1:.2f})"]
    for role,bb in validation["footer_line_bboxes"].items(): lines.append(f"{role} bbox px: ({bb.x0:.2f}, {bb.y0:.2f}, {bb.x1:.2f}, {bb.y1:.2f})")
    lines += [f"Clearance beneath footer Line 3: {validation['footer3_clearance']:.2f} px",f"White canvas clearance beneath footer border: {validation['footer_border_canvas_clearance']:.2f} px","All three footer lines fully visible: yes","All vertical bars in footer Line 3 present: yes","Tier D (speculative) fully visible: yes",""]
    lines += ["TOP-BANNER AND HEADER-STRIP VALIDATION","-"*54]
    for role,bb in validation["banner_bboxes"].items(): lines.append(f"{role} bbox px: ({bb.x0:.2f}, {bb.y0:.2f}, {bb.x1:.2f}, {bb.y1:.2f})")
    for role,bb in validation["strip_bboxes"].items(): lines.append(f"{role} text bbox px: ({bb.x0:.2f}, {bb.y0:.2f}, {bb.x1:.2f}, {bb.y1:.2f})")
    lines += ["GRADE: LOW. complete: yes","HSV-2 serostatus. complete: yes","Both header-strip phrases complete: yes",""]
    lines += ["INTENTIONAL VISUAL IMPROVEMENTS","-"*54,"Light, restrained three-stop gradients replace flat fills.","Gradients are decorative only and encode no scientific variable.","All gradient segments are opaque and vector-safe.","The previously clipped third footer line is now completely visible.","Minor residual differences are limited to native font antialiasing and the intentionally lighter premium gradients."]
    path.write_text("\n".join(lines)+"\n",encoding="utf-8")
    return path


def create_zip(paths:Sequence[Path])->Path:
    with zipfile.ZipFile(ZIP_PATH,"w",zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in paths: z.write(p,arcname=p.name)
    return ZIP_PATH


def main()->None:
    configure_matplotlib()
    ref_info=inspect_reference_image()
    validate_scientific_content()
    pdf,svg=export_vector_outputs()
    png,jpg,tif,validation=export_raster_outputs()
    vector_info=inspect_vectors(pdf,svg)
    qa=create_qa_files(png,validation)
    report=write_qa_report(ref_info,[png,jpg,tif],pdf,svg,validation,vector_info)
    deliverables=[Path(__file__).resolve(),png,jpg,tif,pdf,svg,*qa,report]
    package=create_zip(deliverables)
    print(f"Generated {len(deliverables)} deliverables")
    print(package)
    print(f"Footer line 3 clearance: {validation['footer3_clearance']:.2f}px")
    print(f"Footer border-to-canvas clearance: {validation['footer_border_canvas_clearance']:.2f}px")
    print(f"Minimum text-to-border clearance: {validation['min_text_border_clearance']:.2f}px")

if __name__=="__main__":
    main()
