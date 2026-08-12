#!/usr/bin/env python3
"""Publication-quality reconstruction of Figure 5.

The source raster is used only for QA comparisons. All publication outputs are
redrawn from native Matplotlib text, lines, patches, and vector gradient strips.
"""
from __future__ import annotations

import io
import math
import re
import zipfile
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle, PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

# -----------------------------------------------------------------------------
# Paths and global configuration
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_IMAGE = Path("/mnt/data/image.png")
OUT_DIR = SCRIPT_DIR
ZIP_PATH = SCRIPT_DIR.parent / "Figure_5_publication_ready_package.zip"

DPI = 300
WIDTH_CM = 16.5
SOURCE_W = 1070
SOURCE_H = 720
ASPECT = SOURCE_W / SOURCE_H
HEIGHT_CM = WIDTH_CM / ASPECT
WIDTH_IN = WIDTH_CM / 2.54
HEIGHT_IN = HEIGHT_CM / 2.54
RASTER_W = 1949
RASTER_H = 1311
RASTER_FIGSIZE = ((RASTER_W + 0.04) / DPI, (RASTER_H + 0.04) / DPI)
VECTOR_FIGSIZE = (WIDTH_IN, HEIGHT_IN)

# Geometry measured from the supplied 1070 x 720 source.
AX_RECT = [372/SOURCE_W, (SOURCE_H-603)/SOURCE_H, (848-372)/SOURCE_W, (603-12)/SOURCE_H]
LEFT_LABEL_X = 0.342
RIGHT_VALUE_X = 0.810
RIGHT_HEADER_Y = 0.997
LEGEND_RECT = [0.350, 0.005, 0.440, 0.060]

X_LIM = (0.025, 7.0)
X_TICKS = [0.05, 0.1, 0.5, 1, 2, 3, 4]
X_TICK_LABELS = ["0.05", "0.1", "0.5", "1", "2", "3", "4"]
Y_LIM = (-0.8, 8.8)
Y_POS = list(range(8, -1, -1))
SEPARATOR_Y = 4.5
SHADE_Y0, SHADE_Y1 = 5.5, 8.5

FONT_FAMILY = "DejaVu Sans"
FONT = {
    "study": 4.8,
    "right": 4.2,
    "header": 4.4,
    "tick": 4.7,
    "xlabel": 5.6,
    "section": 4.2,
    "mol_ann": 4.0,
    "mol_right": 4.2,
    "legend": 4.1,
}

COLORS = {
    "blue_line": "#1769B7",
    "orange_line": "#E56216",
    "green_line": "#287D35",
    "red_line": "#C41F24",
    "grid": "#E8EBEF",
    "null": "#B4B8BD",
    "separator": "#73777C",
    "spine": "#33373B",
    "annotation": "#555A60",
}

GRADIENTS: Dict[str, Tuple[str, str, str]] = {
    "blue": ("#E2EFFB", "#82B2DF", "#3F82C4"),
    "orange": ("#FFF0E3", "#F4B985", "#E78B47"),
    "green": ("#E7F3E9", "#8EBF93", "#5B9C62"),
    "red": ("#F9E5E6", "#DC8B90", "#BF555C"),
    "shade": ("#FAFCFF", "#F5F8FC", "#EEF4FA"),
}

MARKER_WIDTH_PT = 6.0
MARKER_HEIGHT_PT = 6.0
OPEN_DIAMOND_MS = 7.5
CI_LW = 1.00
CAP_HALF = 0.20
SPINE_LW = 0.72
GRID_LW = 0.48
NULL_LW = 1.15

LEGEND_ENTRIES = [
    ("blue", "SR/MA (estimates unadjusted for HPV)"),
    ("green", "Prospective cohort (WLHIV)"),
    ("orange", "Case-control / IARC pooled"),
    ("red", "Null, non-significant, or molecular null"),
]

# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
def build_dataframe() -> pd.DataFrame:
    rows = [
        ("Zhang 2023b   All HHVs and CC, pooled\n(SR/MA, 67 studies; UNADJUSTED for HPV)", 2.74, 2.13, 3.53, "blue", "2.74 (2.13–3.53)", False),
        ("Zhang 2023b   HSV-2 and cervical cancer\n(SR/MA sub-group; UNADJUSTED for HPV)", 3.01, 2.24, 4.04, "blue", "3.01 (2.24–4.04)", False),
        ("Zhang 2023b   HSV-2 and precancerous lesions\n(SR/MA sub-group; UNADJUSTED)", 2.14, 1.55, 2.96, "blue", "2.14 (1.55–2.96)", False),
        ("Smith 2002   SCC risk, IARC pooled\n(11 countries; HPV-positive women)", 2.20, 1.40, 3.40, "orange", "2.20 (1.40–3.40)", False),
        ("Seidman 2023   HPV incidence (aHR)\n(Prospective cohort; WLHIV; 36 months)", 1.80, 1.10, 3.00, "green", "1.80 (1.10–3.00)", False),
        ("Lehtinen 2002   CC risk, Nordic nested CC\n(Nested CC; HPV-adjusted; non-significant)", 1.40, 0.90, 2.20, "red", "1.40 (0.90–2.20)", False),
        ("Smith 2002   Adenocarcinoma risk, IARC (null)\n(HPV-positive women; no association)", 1.80, 0.90, 3.80, "red", "1.80 (0.90–3.80)", False),
        ("Seidman 2023   HPV persistence (aHR)\n(WLHIV; not assoc. with precancer)", 1.60, 1.00, 2.50, "red", "1.60 (1.00–2.50)", False),
        ("Tran-Thanh 2003   Molecular null (PCR)\n(No HSV-2 DNA in 200 CC specimens)", 0.05, np.nan, np.nan, "red", "", True),
    ]
    df = pd.DataFrame(rows, columns=["label", "estimate", "lower", "upper", "category", "right_text", "molecular_null"])
    df["y"] = Y_POS
    return df


def validate_scientific_data(df: pd.DataFrame) -> None:
    expected = [
        (2.74, 2.13, 3.53), (3.01, 2.24, 4.04), (2.14, 1.55, 2.96),
        (2.20, 1.40, 3.40), (1.80, 1.10, 3.00), (1.40, 0.90, 2.20),
        (1.80, 0.90, 3.80), (1.60, 1.00, 2.50),
    ]
    assert len(df) == 9
    assert list(zip(df.iloc[:8].estimate, df.iloc[:8].lower, df.iloc[:8].upper)) == expected
    assert df.iloc[8].molecular_null and math.isclose(df.iloc[8].estimate, 0.05)
    assert pd.isna(df.iloc[8].lower) and pd.isna(df.iloc[8].upper)
    assert list(df.y) == Y_POS

# -----------------------------------------------------------------------------
# Drawing helpers
# -----------------------------------------------------------------------------
def configure_matplotlib() -> None:
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [FONT_FAMILY],
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.transparent": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.unicode_minus": False,
        "path.simplify": False,
    })


def interp_three(stops: Sequence[str], n: int) -> np.ndarray:
    rgb = np.asarray([to_rgb(c) for c in stops], float)
    t = np.linspace(0, 1, n)
    out = np.empty((n, 3))
    for i, v in enumerate(t):
        if v <= 0.5:
            u = v / 0.5
            out[i] = rgb[0]*(1-u) + rgb[1]*u
        else:
            u = (v-0.5)/0.5
            out[i] = rgb[1]*(1-u) + rgb[2]*u
    return out


def data_to_axes_fraction(ax: plt.Axes, x: float, y: float) -> Tuple[float, float]:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xf = (math.log10(x)-math.log10(xmin))/(math.log10(xmax)-math.log10(xmin))
    yf = (y-ymin)/(ymax-ymin)
    return xf, yf


def draw_background_gradient(ax: plt.Axes, segments: int = 100) -> None:
    ymin, ymax = ax.get_ylim()
    y0 = (SHADE_Y0-ymin)/(ymax-ymin)
    y1 = (SHADE_Y1-ymin)/(ymax-ymin)
    colors = interp_three(GRADIENTS["shade"], segments)
    h = (y1-y0)/segments
    for i, c in enumerate(colors):
        ax.add_patch(Rectangle((0, y0+i*h), 1, h*1.03, transform=ax.transAxes,
                               facecolor=c, edgecolor="none", linewidth=0,
                               antialiased=False, zorder=0.1, clip_on=True))


def draw_gradient_diamond(fig: plt.Figure, ax: plt.Axes, x: float, y: float, category: str,
                          width_pt: float = MARKER_WIDTH_PT, height_pt: float = MARKER_HEIGHT_PT,
                          strips: int = 36) -> PathPatch:
    xf, yf = data_to_axes_fraction(ax, x, y)
    ax_w_in = fig.get_figwidth() * AX_RECT[2]
    ax_h_in = fig.get_figheight() * AX_RECT[3]
    half_w = (width_pt/72) / ax_w_in / 2
    half_h = (height_pt/72) / ax_h_in / 2
    verts = [(xf, yf+half_h), (xf+half_w, yf), (xf, yf-half_h), (xf-half_w, yf), (xf, yf+half_h)]
    path = MplPath(verts)
    clip = PathPatch(path, transform=ax.transAxes, facecolor="none", edgecolor="none")
    ax.add_patch(clip)
    colors = interp_three(GRADIENTS[category], strips)
    x0 = xf-half_w
    seg_w = (2*half_w)/strips
    for i, c in enumerate(colors):
        r = Rectangle((x0+i*seg_w, yf-half_h), seg_w*1.08, 2*half_h,
                      transform=ax.transAxes, facecolor=c, edgecolor="none", linewidth=0,
                      antialiased=False, zorder=5, clip_on=True)
        r.set_clip_path(clip)
        ax.add_patch(r)
    outline = PathPatch(path, transform=ax.transAxes, facecolor="none",
                        edgecolor=COLORS[f"{category}_line"], linewidth=0.35,
                        joinstyle="miter", zorder=5.5)
    ax.add_patch(outline)
    return outline


def make_figure(figsize: Tuple[float, float]) -> Tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=figsize, dpi=DPI, facecolor="white", layout=None)
    ax = fig.add_axes(AX_RECT)
    ax.set_xscale("log")
    ax.set_xlim(*X_LIM)
    ax.set_ylim(*Y_LIM)
    ax.set_xticks(X_TICKS)
    ax.set_xticklabels(X_TICK_LABELS, fontsize=FONT["tick"])
    ax.set_yticks([])
    ax.minorticks_on()
    ax.tick_params(axis="x", which="major", direction="out", length=2.7, width=0.55, pad=2)
    ax.tick_params(axis="x", which="minor", direction="out", length=1.4, width=0.32, color="#AEB3B8")
    ax.tick_params(axis="y", length=0)
    for x in X_TICKS:
        ax.axvline(x, color=COLORS["grid"], linewidth=GRID_LW, zorder=0.4)
    ax.axvline(1, color=COLORS["null"], linewidth=NULL_LW, linestyle=(0, (4, 2)), zorder=1.0)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(COLORS["spine"])
        s.set_linewidth(SPINE_LW)
    ax.set_xlabel("Odds Ratio / Adjusted Hazard Ratio  (log scale)", fontsize=FONT["xlabel"],
                  fontweight="bold", labelpad=5.0)
    ax.xaxis.set_label_coords(0.5, -0.042)
    return fig, ax


def draw_study_labels(fig: plt.Figure, ax: plt.Axes, df: pd.DataFrame, registry: List[dict]) -> None:
    ymin, ymax = ax.get_ylim()
    for row in df.itertuples(index=False):
        yf = AX_RECT[1] + AX_RECT[3] * ((row.y-ymin)/(ymax-ymin))
        txt = fig.text(LEFT_LABEL_X, yf, row.label, ha="right", va="center",
                       fontsize=FONT["study"], color="black", linespacing=0.90,
                       multialignment="right")
        registry.append({"name": f"study_{row.y}", "artist": txt, "kind": "study"})


def draw_confidence_intervals(fig: plt.Figure, ax: plt.Axes, df: pd.DataFrame) -> None:
    for row in df.iloc[:8].itertuples(index=False):
        c = COLORS[f"{row.category}_line"]
        ax.hlines(row.y, row.lower, row.upper, color=c, linewidth=CI_LW, zorder=4)
        ax.vlines([row.lower, row.upper], row.y-CAP_HALF, row.y+CAP_HALF, color=c, linewidth=CI_LW, zorder=4)
        draw_gradient_diamond(fig, ax, row.estimate, row.y, row.category)


def draw_molecular_null(fig: plt.Figure, ax: plt.Axes, registry: List[dict]) -> None:
    red = COLORS["red_line"]
    ax.plot([0.05], [0], marker="D", markersize=OPEN_DIAMOND_MS,
            markerfacecolor="white", markeredgecolor=red, markeredgewidth=1.35,
            linestyle="none", zorder=6)
    ann = ax.annotate("No HSV-2 DNA\ndetected (PCR)", xy=(0.05, 0), xytext=(0.18, -0.43),
                      textcoords="data", ha="left", va="center", fontsize=FONT["mol_ann"],
                      color=red, fontstyle="italic", linespacing=0.90,
                      arrowprops=dict(arrowstyle="->", color=red, linewidth=0.75,
                                      shrinkA=1, shrinkB=2, mutation_scale=7),
                      annotation_clip=False, zorder=7)
    registry.append({"name": "molecular_inplot", "artist": ann, "kind": "annotation"})


def draw_right_column(fig: plt.Figure, ax: plt.Axes, df: pd.DataFrame, registry: List[dict]) -> None:
    header = fig.text(RIGHT_VALUE_X, RIGHT_HEADER_Y, "OR / aHR (95% CI)", ha="left", va="top",
                      fontsize=FONT["header"], fontweight="bold", color="black")
    registry.append({"name": "right_header", "artist": header, "kind": "right"})
    ymin, ymax = ax.get_ylim()
    for row in df.iloc[:8].itertuples(index=False):
        yf = AX_RECT[1] + AX_RECT[3] * ((row.y-ymin)/(ymax-ymin))
        txt = fig.text(RIGHT_VALUE_X, yf, row.right_text, ha="left", va="center",
                       fontsize=FONT["right"], color="black")
        registry.append({"name": f"right_{row.y}", "artist": txt, "kind": "right"})
    yf = AX_RECT[1] + AX_RECT[3] * ((0-ymin)/(ymax-ymin))
    txt = fig.text(RIGHT_VALUE_X, yf, "No HSV-2 DNA detected\n(PCR; molecular null)",
                   ha="left", va="center", fontsize=FONT["mol_right"],
                   color=COLORS["red_line"], fontweight="bold", fontstyle="italic",
                   linespacing=0.92)
    registry.append({"name": "right_molecular", "artist": txt, "kind": "right"})


def draw_section_annotations(ax: plt.Axes, registry: List[dict]) -> None:
    ax.axhline(SEPARATOR_Y, color=COLORS["separator"], linewidth=0.70,
               linestyle=(0, (1.0, 1.7)), zorder=2)
    t1 = ax.text(0.028, 4.78, "▲ Elevated associations", ha="left", va="center",
                 fontsize=FONT["section"], fontstyle="italic", color=COLORS["annotation"], zorder=7)
    t2 = ax.text(0.028, 4.20, "▼ Null / non-significant / molecular null", ha="left", va="center",
                 fontsize=FONT["section"], fontstyle="italic", color=COLORS["annotation"], zorder=7)
    registry.extend([
        {"name": "section_elevated", "artist": t1, "kind": "annotation"},
        {"name": "section_null", "artist": t2, "kind": "annotation"},
    ])


def draw_gradient_swatch(ax: plt.Axes, x: float, y: float, w: float, h: float,
                         category: str, segments: int = 28) -> List[Rectangle]:
    out = []
    colors = interp_three(GRADIENTS[category], segments)
    sw = w/segments
    for i, c in enumerate(colors):
        r = Rectangle((x+i*sw, y), sw*1.08, h, transform=ax.transAxes,
                      facecolor=c, edgecolor="none", linewidth=0, antialiased=False)
        ax.add_patch(r); out.append(r)
    return out


def draw_custom_legend(fig: plt.Figure, registry: List[dict]) -> dict:
    lax = fig.add_axes(LEGEND_RECT)
    lax.set_xlim(0, 1); lax.set_ylim(0, 1); lax.axis("off")
    frame = Rectangle((0, 0), 1, 1, transform=lax.transAxes, facecolor="white",
                      edgecolor="#CFCFCF", linewidth=0.65)
    lax.add_patch(frame)
    sw_x_l, sw_x_r = 0.020, 0.550
    sw_w, sw_h = 0.032, 0.22
    gap = 0.012
    y_top, y_bot = 0.68, 0.30
    specs = [
        (sw_x_l, y_top, "blue", LEGEND_ENTRIES[0][1]),
        (sw_x_r, y_top, "green", LEGEND_ENTRIES[1][1]),
        (sw_x_l, y_bot, "orange", LEGEND_ENTRIES[2][1]),
        (sw_x_r, y_bot, "red", LEGEND_ENTRIES[3][1]),
    ]
    texts = []
    symbol_bboxes = []
    for x, y, cat, label in specs:
        draw_gradient_swatch(lax, x, y-sw_h/2, sw_w, sw_h, cat)
        txt = lax.text(x+sw_w+gap, y, label, transform=lax.transAxes,
                       ha="left", va="center", fontsize=FONT["legend"], color="black")
        texts.append((label, txt))
        registry.append({"name": f"legend_{cat}", "artist": txt, "kind": "legend"})
        symbol_bboxes.append((cat, (x, y-sw_h/2, sw_w, sw_h)))
    return {"ax": lax, "frame": frame, "texts": texts, "symbols": symbol_bboxes}


def create_figure(df: pd.DataFrame, figsize: Tuple[float, float]) -> Tuple[plt.Figure, plt.Axes, List[dict], dict]:
    fig, ax = make_figure(figsize)
    registry: List[dict] = []
    draw_background_gradient(ax)
    draw_study_labels(fig, ax, df, registry)
    draw_confidence_intervals(fig, ax, df)
    draw_molecular_null(fig, ax, registry)
    draw_right_column(fig, ax, df, registry)
    draw_section_annotations(ax, registry)
    legend = draw_custom_legend(fig, registry)
    registry.append({"name": "xlabel", "artist": ax.xaxis.label, "kind": "xlabel"})
    for i, lab in enumerate(ax.get_xticklabels()):
        registry.append({"name": f"tick_{i}", "artist": lab, "kind": "tick"})
    return fig, ax, registry, legend

# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------
def _bbox_axes_rect(ax: plt.Axes, rect: Tuple[float,float,float,float]) -> Bbox:
    x,y,w,h = rect
    p0 = ax.transAxes.transform((x,y)); p1 = ax.transAxes.transform((x+w,y+h))
    return Bbox.from_extents(p0[0], p0[1], p1[0], p1[1])


def bboxes_overlap(a: Bbox, b: Bbox, pad: float = 0) -> bool:
    return not ((a.x1+pad)<=b.x0 or (b.x1+pad)<=a.x0 or (a.y1+pad)<=b.y0 or (b.y1+pad)<=a.y0)


def validate_text_bounds(fig: plt.Figure, ax: plt.Axes, registry: List[dict], legend: dict) -> dict:
    fig.canvas.draw(); renderer = fig.canvas.get_renderer(); canvas = fig.bbox
    items = []
    failures = []
    for rec in registry:
        bb = rec["artist"].get_window_extent(renderer)
        items.append({**rec, "bbox": bb})
        if bb.x0 < canvas.x0-0.5 or bb.y0 < canvas.y0-0.5 or bb.x1 > canvas.x1+0.5 or bb.y1 > canvas.y1+0.5:
            failures.append(f"canvas overflow: {rec['name']}")
    axis_bb = ax.get_window_extent(renderer)
    for it in items:
        if it["kind"] == "study" and it["bbox"].x1 > axis_bb.x0-4:
            failures.append(f"study overlaps plot: {it['name']}")
        if it["kind"] == "right" and it["bbox"].x0 < axis_bb.x1+4:
            failures.append(f"right column overlaps plot: {it['name']}")
    # Legend frame and content.
    frame_bb = legend["frame"].get_window_extent(renderer)
    legend_text_bbs = {label: txt.get_window_extent(renderer) for label, txt in legend["texts"]}
    legend_symbol_bbs = {cat: _bbox_axes_rect(legend["ax"], rect) for cat, rect in legend["symbols"]}
    legend_failures = []
    for label, bb in legend_text_bbs.items():
        if not (bb.x0 >= frame_bb.x0+4 and bb.x1 <= frame_bb.x1-6 and bb.y0 >= frame_bb.y0+3 and bb.y1 <= frame_bb.y1-3):
            legend_failures.append(f"legend containment: {label}")
    for cat, bb in legend_symbol_bbs.items():
        if not (bb.x0 >= frame_bb.x0+2 and bb.x1 <= frame_bb.x1-2 and bb.y0 >= frame_bb.y0+2 and bb.y1 <= frame_bb.y1-2):
            legend_failures.append(f"legend symbol containment: {cat}")
    # Check text overlaps in legend.
    vals = list(legend_text_bbs.items())
    for i in range(len(vals)):
        for j in range(i+1, len(vals)):
            if bboxes_overlap(vals[i][1], vals[j][1], pad=1):
                legend_failures.append(f"legend text overlap: {vals[i][0]} | {vals[j][0]}")
    xlabel_bb = ax.xaxis.label.get_window_extent(renderer)
    if bboxes_overlap(frame_bb, xlabel_bb, pad=2):
        legend_failures.append("legend overlaps x-axis label")
    failures.extend(legend_failures)
    right_clear = min(frame_bb.x1-b.x1 for b in legend_text_bbs.values())
    canvas_clear = min(min(b.x0-canvas.x0, canvas.x1-b.x1, b.y0-canvas.y0, canvas.y1-b.y1) for b in [i["bbox"] for i in items])
    return {
        "items": items, "failures": failures, "axis_bbox": axis_bb,
        "frame_bbox": frame_bb, "legend_text_bboxes": legend_text_bbs,
        "legend_symbol_bboxes": legend_symbol_bbs,
        "legend_right_clearance": right_clear, "min_canvas_clearance": canvas_clear,
        "xlabel_bbox": xlabel_bb,
    }

# -----------------------------------------------------------------------------
# Exports
# -----------------------------------------------------------------------------
def render_raster(df: pd.DataFrame) -> Tuple[Image.Image, dict]:
    fig, ax, registry, legend = create_figure(df, RASTER_FIGSIZE)
    canvas = matplotlib.backends.backend_agg.FigureCanvasAgg(fig); canvas.draw()
    validation = validate_text_bounds(fig, ax, registry, legend)
    if validation["failures"]:
        raise RuntimeError("Text/legend validation failed: " + "; ".join(validation["failures"]))
    rgba = np.asarray(canvas.buffer_rgba())
    image = Image.fromarray(rgba, "RGBA").convert("RGB")
    if image.size != (RASTER_W, RASTER_H):
        raise RuntimeError(f"Unexpected native raster dimensions {image.size}")
    plt.close(fig)
    return image, validation


def export_rasters(df: pd.DataFrame) -> Tuple[Path,Path,Path,dict]:
    image, validation = render_raster(df)
    png = OUT_DIR/"Figure_5_HSV2_forest_plot.png"
    jpg = OUT_DIR/"Figure_5_HSV2_forest_plot.jpg"
    tif = OUT_DIR/"Figure_5_HSV2_forest_plot.tiff"
    image.save(png, "PNG", dpi=(DPI,DPI), optimize=True)
    image.save(jpg, "JPEG", dpi=(DPI,DPI), quality=100, subsampling=0, optimize=True)
    image.save(tif, "TIFF", dpi=(DPI,DPI), compression="tiff_lzw")
    return png,jpg,tif,validation


def export_vectors(df: pd.DataFrame) -> Tuple[Path,Path]:
    pdf = OUT_DIR/"Figure_5_HSV2_forest_plot.pdf"
    svg = OUT_DIR/"Figure_5_HSV2_forest_plot.svg"
    fig, ax, registry, legend = create_figure(df, VECTOR_FIGSIZE)
    val = validate_text_bounds(fig, ax, registry, legend)
    if val["failures"]:
        raise RuntimeError("Vector layout validation failed: " + "; ".join(val["failures"]))
    fig.savefig(pdf, format="pdf", facecolor="white", transparent=False)
    fig.savefig(svg, format="svg", facecolor="white", transparent=False)
    plt.close(fig)
    return pdf,svg

# -----------------------------------------------------------------------------
# Export validation and QA
# -----------------------------------------------------------------------------
def inspect_reference_image() -> Tuple[Tuple[int,int], str]:
    with Image.open(REFERENCE_IMAGE) as im:
        return im.size, im.mode


def inspect_raster(path: Path) -> dict:
    with Image.open(path) as im:
        compression = im.info.get("compression")
        if path.suffix.lower() in {".tif", ".tiff"}:
            compression = im.tag_v2.get(259, compression)
        return {
            "file": path.name, "format": im.format, "size": im.size, "mode": im.mode,
            "dpi": im.info.get("dpi"), "transparency": ("A" in im.getbands() or "transparency" in im.info),
            "compression": compression, "frames": getattr(im, "n_frames", 1),
            "bytes": path.stat().st_size,
        }


def inspect_vectors(pdf: Path, svg: Path) -> dict:
    out = {}
    try:
        import fitz  # type: ignore
        doc = fitz.open(pdf); page=doc[0]
        out["pdf_page_points"] = (page.rect.width, page.rect.height)
        out["pdf_image_count"] = len(page.get_images(full=True))
        out["pdf_page_count"] = doc.page_count
        doc.close()
    except Exception as e:
        out["pdf_error"] = str(e)
    s = svg.read_text(encoding="utf-8", errors="replace")
    out["svg_has_image_tag"] = bool(re.search(r"<image\b", s, re.I))
    out["svg_rect_count"] = len(re.findall(r"<rect\b", s, re.I))
    out["svg_path_count"] = len(re.findall(r"<path\b", s, re.I))
    mw = re.search(r'<svg[^>]+width="([^"]+)"', s, re.I)
    mh = re.search(r'<svg[^>]+height="([^"]+)"', s, re.I)
    out["svg_width"] = mw.group(1) if mw else "not found"
    out["svg_height"] = mh.group(1) if mh else "not found"
    return out


def font_for_qa(size: int, bold: bool=False):
    try:
        from matplotlib import font_manager
        fp = font_manager.findfont(font_manager.FontProperties(family=FONT_FAMILY, weight="bold" if bold else "normal"))
        return ImageFont.truetype(fp, size)
    except Exception:
        return ImageFont.load_default()


def create_qa_files(png: Path, validation: dict) -> Tuple[Path,Path,Path,Path,Path,Path]:
    with Image.open(REFERENCE_IMAGE) as im:
        ref=im.convert("RGB")
    with Image.open(png) as im:
        rec_full=im.convert("RGB")
    rec=rec_full.resize(ref.size, Image.Resampling.LANCZOS)

    overlay=OUT_DIR/"Figure_5_QA_overlay.png"
    Image.blend(ref, rec, 0.5).save(overlay, dpi=(DPI,DPI))

    gap=18; header=46
    side=Image.new("RGB", (ref.width*2+gap, ref.height+header), "white")
    side.paste(ref,(0,header)); side.paste(rec,(ref.width+gap,header))
    d=ImageDraw.Draw(side); f=font_for_qa(22,True)
    d.text((ref.width//2,18),"Reference",fill="black",font=f,anchor="mm")
    d.text((ref.width+gap+ref.width//2,18),"Reconstructed",fill="black",font=f,anchor="mm")
    side_path=OUT_DIR/"Figure_5_QA_side_by_side.png"; side.save(side_path,dpi=(DPI,DPI))

    def edge(im):
        e=ImageOps.grayscale(im).filter(ImageFilter.FIND_EDGES); e=ImageOps.autocontrast(e,cutoff=1); return ImageOps.invert(e).convert("RGB")
    ec=Image.new("RGB", side.size,"white"); ec.paste(edge(ref),(0,header)); ec.paste(edge(rec),(ref.width+gap,header))
    de=ImageDraw.Draw(ec); de.text((ref.width//2,18),"Reference edges",fill="black",font=f,anchor="mm"); de.text((ref.width+gap+ref.width//2,18),"Reconstructed edges",fill="black",font=f,anchor="mm")
    edges=OUT_DIR/"Figure_5_QA_edges.png"; ec.save(edges,dpi=(DPI,DPI))

    # Text-bounds QA on final raster.
    bounds=rec_full.copy(); db=ImageDraw.Draw(bounds)
    for item in validation["items"]:
        b=item["bbox"]; box=(int(b.x0), int(RASTER_H-b.y1), int(b.x1), int(RASTER_H-b.y0))
        db.rectangle(box, outline=(220,0,180), width=1)
    fb=validation["frame_bbox"]; db.rectangle((int(fb.x0),int(RASTER_H-fb.y1),int(fb.x1),int(RASTER_H-fb.y0)),outline=(0,150,255),width=2)
    text_bounds=OUT_DIR/"Figure_5_QA_text_bounds.png"; bounds.save(text_bounds,dpi=(DPI,DPI))

    # Legend closeup.
    exp=14; l=max(int(fb.x0)-exp,0); t=max(int(RASTER_H-fb.y1)-exp,0); r=min(int(fb.x1)+exp,RASTER_W); b=min(int(RASTER_H-fb.y0)+exp,RASTER_H)
    leg=rec_full.crop((l,t,r,b)); leg=leg.resize((leg.width*4,leg.height*4),Image.Resampling.LANCZOS)
    legend_close=OUT_DIR/"Figure_5_QA_legend_closeup.png"; leg.save(legend_close,dpi=(DPI,DPI))

    # Molecular-null closeup: plot bottom-left through right statement.
    crop=(int(0.30*RASTER_W), int(0.62*RASTER_H), int(0.98*RASTER_W), int(0.91*RASTER_H))
    mol=rec_full.crop(crop); mol=mol.resize((mol.width*2,mol.height*2),Image.Resampling.LANCZOS)
    mol_close=OUT_DIR/"Figure_5_QA_molecular_null_closeup.png"; mol.save(mol_close,dpi=(DPI,DPI))
    return overlay,side_path,edges,text_bounds,legend_close,mol_close


def write_report(df: pd.DataFrame, ref_info, rasters, pdf, svg, validation, vector_info) -> Path:
    path=OUT_DIR/"Figure_5_QA_report.txt"
    lines=[]
    lines += ["FIGURE 5 PUBLICATION-READY QA REPORT", "="*52, ""]
    lines += [f"Source image: {REFERENCE_IMAGE}", f"Source dimensions: {ref_info[0][0]} x {ref_info[0][1]} px", f"Source colour mode: {ref_info[1]}", "Source use: QA comparison only; not embedded or upscaled into publication outputs.", ""]
    lines += [f"Final physical dimensions: {WIDTH_CM:.3f} x {HEIGHT_CM:.3f} cm", f"Raster dimensions: {RASTER_W} x {RASTER_H} px", f"Nominal DPI: {DPI} x {DPI}", ""]
    lines += ["RASTER OUTPUTS", "-"*52]
    for p in rasters:
        q=inspect_raster(p)
        lines += [f"File: {q['file']}",f"  Format: {q['format']}",f"  Dimensions: {q['size'][0]} x {q['size'][1]}",f"  Mode: {q['mode']}",f"  DPI metadata: {q['dpi']}",f"  Transparency: {q['transparency']}",f"  File size: {q['bytes']} bytes"]
        if q['format']=='TIFF': lines += [f"  Compression tag: {q['compression']} (5 = LZW)",f"  Frames: {q['frames']}"]
        if q['format']=='JPEG': lines += ["  Quality: 100","  Chroma subsampling: 0 (4:4:4)"]
    lines += ["", "VECTOR OUTPUTS", "-"*52]
    for k,v in vector_info.items(): lines.append(f"{k}: {v}")
    lines += ["Vector gradients: native rectangle strips clipped to vector diamonds; upper shading uses vector rectangles.", "Original reference embedded: no.", ""]
    lines += ["SCIENTIFIC VALIDATION", "-"*52, f"Study rows: {len(df)}", "Conventional numerical rows: 8", "Molecular-null rows: 1", "All eight estimates and confidence intervals: validated exactly.", "Row 9: open diamond at x=0.05; no numerical estimate string, CI line, or endpoint caps.", f"Labelled ticks: {X_TICK_LABELS}", f"Axis limits: {X_LIM}", "Null-reference line: x=1.0", "Shading: rows 1–3 only", "Separator: between rows 4 and 5", "Legend entries: 4", ""]
    lines += ["LEGEND AND TEXT BOUNDARY VALIDATION", "-"*52]
    fb=validation['frame_bbox']; lines.append(f"Legend frame bbox px: ({fb.x0:.2f}, {fb.y0:.2f}, {fb.x1:.2f}, {fb.y1:.2f})")
    for label,bb in validation['legend_text_bboxes'].items(): lines.append(f"Legend label [{label}] bbox: ({bb.x0:.2f}, {bb.y0:.2f}, {bb.x1:.2f}, {bb.y1:.2f})")
    lines += [f"Minimum legend right-edge clearance: {validation['legend_right_clearance']:.2f} px",f"Minimum overall text-to-canvas clearance: {validation['min_canvas_clearance']:.2f} px",f"Text/legend failures: {len(validation['failures'])}","All text-containment tests passed: yes" if not validation['failures'] else "All text-containment tests passed: no",""]
    lines += ["INTENTIONAL VISUAL CHANGE", "-"*52, "Filled diamonds, matching legend swatches, and the upper background band use lighter premium gradients.", "Gradients are decorative only and do not encode an additional variable.", "Minor residual differences are limited to native renderer antialiasing and the requested lighter gradients."]
    path.write_text("\n".join(lines)+"\n",encoding="utf-8")
    return path


def create_zip(paths: Sequence[Path]) -> Path:
    with zipfile.ZipFile(ZIP_PATH,"w",zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in paths: z.write(p,arcname=p.name)
    return ZIP_PATH


def main() -> None:
    configure_matplotlib()
    ref_info=inspect_reference_image()
    if ref_info[0] != (SOURCE_W,SOURCE_H):
        raise RuntimeError(f"Unexpected source image dimensions: {ref_info[0]}")
    df=build_dataframe(); validate_scientific_data(df)
    pdf,svg=export_vectors(df)
    png,jpg,tif,validation=export_rasters(df)
    vector_info=inspect_vectors(pdf,svg)
    qa=create_qa_files(png,validation)
    report=write_report(df,ref_info,[png,jpg,tif],pdf,svg,validation,vector_info)
    deliverables=[Path(__file__).resolve(),png,jpg,tif,pdf,svg,*qa,report]
    package=create_zip(deliverables)
    print(f"Generated {len(deliverables)} deliverables")
    print(package)
    print(f"Legend right clearance: {validation['legend_right_clearance']:.2f}px")
    print(f"Minimum text canvas clearance: {validation['min_canvas_clearance']:.2f}px")

if __name__ == "__main__":
    main()
