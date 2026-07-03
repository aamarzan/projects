#!/usr/bin/env python3
"""
create_s1_figure_premium_v6.py

Final premium Figure S1 builder with *safe* gel handling.

Main fixes in v6:
- no aggressive manual cropping by default
- safe border trimming with built-in retention padding
- preserves full gel wording/labels as much as possible
- keeps the same premium multi-panel layout
- keeps the center K417N interpretation panel
- exports PNG, PDF, and TIFF

Recommended use:
    python create_s1_figure_premium_v6.py \
      --input-dir "E:\\DrAhmed\\Submitted\\RFLP\\Image_Extract\\Thesis Figures\\Gel Raw Figure" \
      --output-prefix "E:\\DrAhmed\\Submitted\\RFLP\\S1 figure\\Figure_S1_premium"

Dependencies:
    pip install pillow matplotlib numpy
"""

from __future__ import annotations

import argparse
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.gridspec import GridSpec
from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ImageOps


# =========================
# Configuration
# =========================

@dataclass(frozen=True)
class GelPanel:
    key: str
    panel_letter: str
    title: str
    chip: str
    filename: str
    # Keep the option for rare gentle manual crop, but default is no crop.
    crop_lrtb: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)


PANELS: Tuple[GelPanel, ...] = (
    GelPanel("T19I",  "A", "T19I",  "Gain-of-site locus",  "f7.png"),
    GelPanel("N440K", "B", "N440K", "Additional spike locus", "f10.png"),
    GelPanel("Q677P", "C", "Q677P", "Loss-of-site locus", "f19.png"),
    GelPanel("K417N", "D", "K417N", "Conditional locus", "f61.png"),
    GelPanel("S413R", "E", "S413R", "Non-spike locus", "f79.png"),
)

DEFAULT_INPUT_DIR = r"E:\DrAhmed\Submitted\RFLP\Image_Extract\Thesis Figures\Gel Raw Figure"
DEFAULT_OUTPUT_PREFIX = r"E:\DrAhmed\Submitted\RFLP\Final Figure\Fig_S1\Figure_S1_premium"

FIGURE_TITLE = "Representative PCR-RFLP gels and K417N conditional digestion"
FIGURE_FOOTER = (
    "Representative PCR-RFLP gels for T19I, N440K, Q677P, K417N, and S413R. "
    "K417N is shown with a conditional digestion schematic because the mutant state can yield either 52 + 74 bp "
    "or remain 126 bp depending on nucleotide route; therefore, a single 126 bp band is not exclusive to wild type."
)


# =========================
# Image utilities
# =========================


def read_image(path: Path) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(f"Required image not found: {path}")
    return Image.open(path).convert("L")



def safe_trim_uniform_border(
    img: Image.Image,
    tolerance: int = 6,
    keep_margin_frac: float = 0.03,
    min_keep_px: int = 20,
) -> Image.Image:
    """
    Safely trims only obvious uniform outer border while preserving nearby text.

    Unlike aggressive autocrop, this function expands the detected content box
    by a safety margin so edge labels/wording are not clipped.
    """
    arr = np.asarray(img)
    h, w = arr.shape[:2]

    # Estimate background from the four corners.
    corner = np.concatenate([
        arr[:12, :12].ravel(),
        arr[:12, -12:].ravel(),
        arr[-12:, :12].ravel(),
        arr[-12:, -12:].ravel(),
    ])
    bg_val = int(np.median(corner))

    bg = Image.new(img.mode, img.size, bg_val)
    diff = ImageChops.difference(img, bg)
    diff = ImageChops.add(diff, diff, 2.0, -tolerance)
    bbox = diff.getbbox()
    if bbox is None:
        return img

    x0, y0, x1, y1 = bbox

    # Expand the box to retain border-adjacent wording.
    mx = max(min_keep_px, int(round(w * keep_margin_frac)))
    my = max(min_keep_px, int(round(h * keep_margin_frac)))

    x0 = max(0, x0 - mx)
    y0 = max(0, y0 - my)
    x1 = min(w, x1 + mx)
    y1 = min(h, y1 + my)

    return img.crop((x0, y0, x1, y1))



def crop_fractional(img: Image.Image, crop_lrtb: Tuple[float, float, float, float]) -> Image.Image:
    l, r, t, b = crop_lrtb
    if max(crop_lrtb) <= 0:
        return img
    w, h = img.size
    x0 = int(round(w * l))
    x1 = w - int(round(w * r))
    y0 = int(round(h * t))
    y1 = h - int(round(h * b))
    if x1 <= x0 or y1 <= y0:
        return img
    return img.crop((x0, y0, x1, y1))



def presentational_tune(img: Image.Image, enhance: bool) -> Image.Image:
    if not enhance:
        return img
    img = ImageOps.autocontrast(img, cutoff=0.25)
    img = ImageEnhance.Contrast(img).enhance(1.08)
    img = ImageEnhance.Sharpness(img).enhance(1.05)
    img = img.filter(ImageFilter.UnsharpMask(radius=1.0, percent=110, threshold=2))
    return img



def fit_on_black_mat(
    img: Image.Image,
    canvas_size: Tuple[int, int] = (1800, 1200),
    padding: int = 65,
    add_outer_border: int = 14,
    border_gray: int = 245,
) -> np.ndarray:
    """
    Fit gel image onto a black presentation mat, preserving aspect ratio.
    A small outer border is added first so edge wording stays visually separated
    from the mat and does not appear clipped.
    """
    if add_outer_border > 0:
        img = ImageOps.expand(img, border=add_outer_border, fill=border_gray)

    canvas_w, canvas_h = canvas_size
    inner_w = canvas_w - 2 * padding
    inner_h = canvas_h - 2 * padding

    src_w, src_h = img.size
    scale = min(inner_w / src_w, inner_h / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))

    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("L", canvas_size, 8)
    x = (canvas_w - new_w) // 2
    y = (canvas_h - new_h) // 2
    canvas.paste(resized, (x, y))
    return np.asarray(canvas)



def prepare_gel_image(path: Path, panel: GelPanel, enhance: bool, no_autocrop: bool) -> np.ndarray:
    img = read_image(path)

    # Safe trim only; no aggressive fractional trim.
    if not no_autocrop:
        img = safe_trim_uniform_border(img, tolerance=6, keep_margin_frac=0.035, min_keep_px=22)

    # Manual crop remains available but defaults to zero for all panels.
    img = crop_fractional(img, panel.crop_lrtb)

    img = presentational_tune(img, enhance=enhance)
    return fit_on_black_mat(img)


# =========================
# Drawing utilities
# =========================


def rounded_card(ax, fc: str = "white", ec: str = "#d4dbe4", lw: float = 1.4, radius: float = 0.03):
    rect = patches.FancyBboxPatch(
        (0, 0), 1, 1,
        transform=ax.transAxes,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        zorder=-5,
    )
    ax.add_patch(rect)



def panel_badge(ax, text: str):
    ax.text(
        0.025, 0.985, text,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=13.0, fontweight="bold", color="#111111",
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#151515", lw=1.0),
        zorder=50,
    )



def chip(ax, x: float, y: float, text: str, fc: str = "#eef4fb", ec: str = "#c8d8ea", color: str = "#24415d"):
    ax.text(
        x, y, text,
        transform=ax.transAxes,
        ha="left", va="center",
        fontsize=8.9, fontweight="bold", color=color,
        bbox=dict(boxstyle="round,pad=0.28", fc=fc, ec=ec, lw=0.9),
        zorder=60,
    )



def draw_gel_panel(ax, image: np.ndarray, panel: GelPanel):
    ax.set_axis_off()
    rounded_card(ax, fc="#ffffff", ec="#d7dde6", lw=1.45, radius=0.035)
    panel_badge(ax, panel.panel_letter)

    ax.text(
        0.12, 0.972, panel.title,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=13.4, fontweight="bold", color="#111827",
    )
    chip(ax, 0.12, 0.90, panel.chip)

    # Slightly taller gel window for better whole-image visibility.
    gel_ax = ax.inset_axes([0.045, 0.085, 0.91, 0.75])
    gel_ax.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    gel_ax.set_axis_off()
    for spine in gel_ax.spines.values():
        spine.set_visible(False)

    ax.add_patch(
        patches.FancyBboxPatch(
            (0.045, 0.085), 0.91, 0.75,
            transform=ax.transAxes,
            boxstyle="round,pad=0.003,rounding_size=0.02",
            linewidth=1.0, edgecolor="#111111", facecolor="none",
        )
    )



def flow_box(ax, x, y, w, h, title, body_lines, fc, ec, title_color="#111827", body_color="#374151"):
    ax.add_patch(
        patches.FancyBboxPatch(
            (x, y), w, h,
            transform=ax.transAxes,
            boxstyle="round,pad=0.014,rounding_size=0.03",
            linewidth=1.28,
            edgecolor=ec,
            facecolor=fc,
        )
    )
    ax.text(
        x + w / 2, y + h * 0.72, title,
        transform=ax.transAxes,
        ha="center", va="center",
        fontsize=10.2, fontweight="bold", color=title_color,
    )
    if isinstance(body_lines, str):
        body_lines = [body_lines]
    body_text = "\n".join(body_lines)
    ax.text(
        x + w / 2, y + h * 0.34, body_text,
        transform=ax.transAxes,
        ha="center", va="center",
        fontsize=8.2, color=body_color,
        linespacing=1.18,
    )



def metadata_chip(ax, x, y, text, fc="#eef4fb", ec="#d8e4f3", color="#1f3b57"):
    ax.text(
        x, y, text,
        transform=ax.transAxes,
        ha="center", va="center",
        fontsize=9.0, fontweight="bold", color=color,
        linespacing=1.08,
        bbox=dict(boxstyle="round,pad=0.32", fc=fc, ec=ec, lw=1.0),
    )



def straight_arrow(ax, x0, y0, x1, y1, color="#4b6074", lw=1.25):
    ax.annotate(
        "",
        xy=(x1, y1), xytext=(x0, y0),
        xycoords=ax.transAxes, textcoords=ax.transAxes,
        arrowprops=dict(arrowstyle="->", lw=lw, color=color, shrinkA=4, shrinkB=4),
    )



def lane_icon(ax, xc: float, y0: float, fragments: Iterable[int], height: float = 0.10):
    ax.add_patch(
        patches.FancyBboxPatch(
            (xc - 0.03, y0), 0.06, height,
            transform=ax.transAxes,
            boxstyle="round,pad=0.004,rounding_size=0.008",
            linewidth=0.8, edgecolor="#9ca3af", facecolor="#f9fafb",
        )
    )
    max_bp = 130
    for bp in fragments:
        yy = y0 + height * (1 - min(bp, max_bp) / max_bp)
        ax.plot([xc - 0.019, xc + 0.019], [yy, yy], transform=ax.transAxes,
                color="#111827", lw=2.0, solid_capstyle="round")



def draw_k417n_logic(ax):
    ax.set_axis_off()
    rounded_card(ax, fc="#ffffff", ec="#d7dde6", lw=1.55, radius=0.035)

    ax.text(
        0.055, 0.965, "K417N conditional digestion",
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=13.8, fontweight="bold", color="#111827",
    )
    chip(ax, 0.055, 0.895, "Interpretation panel", fc="#fff4e8", ec="#f0d3a8", color="#8a4b00")

    metadata_chip(ax, 0.23, 0.77, "Target locus\nK417N")
    metadata_chip(ax, 0.50, 0.77, "Amplicon\n126 bp")
    metadata_chip(ax, 0.77, 0.77, "Enzyme\nSspI-HF")

    flow_box(
        ax, 0.05, 0.46, 0.27, 0.20,
        title="Wild type",
        body_lines=["AAG", "undigested", "126 bp"],
        fc="#f5fbf4", ec="#b7d5b3",
        title_color="#214d22", body_color="#2f4f2f",
    )
    flow_box(
        ax, 0.365, 0.46, 0.27, 0.20,
        title="Mutant route 1",
        body_lines=["AAT", "site gained", "52 + 74 bp"],
        fc="#fff8eb", ec="#f0d4a5",
        title_color="#8a4b00", body_color="#7c5a1f",
    )
    flow_box(
        ax, 0.68, 0.46, 0.27, 0.20,
        title="Mutant route 2",
        body_lines=["AAC", "site not gained", "126 bp"],
        fc="#fff8eb", ec="#f0d4a5",
        title_color="#8a4b00", body_color="#7c5a1f",
    )

    straight_arrow(ax, 0.23, 0.73, 0.185, 0.66)
    straight_arrow(ax, 0.50, 0.73, 0.50, 0.66)
    straight_arrow(ax, 0.77, 0.73, 0.815, 0.66)

    lane_icon(ax, 0.185, 0.28, [126])
    lane_icon(ax, 0.50, 0.28, [52, 74])
    lane_icon(ax, 0.815, 0.28, [126])

    ax.text(0.185, 0.252, "WT", transform=ax.transAxes, ha="center", va="top", fontsize=8.4, color="#374151")
    ax.text(0.50, 0.252, "Mut AAT", transform=ax.transAxes, ha="center", va="top", fontsize=8.4, color="#374151")
    ax.text(0.815, 0.252, "Mut AAC", transform=ax.transAxes, ha="center", va="top", fontsize=8.4, color="#374151")

    ax.add_patch(
        patches.FancyBboxPatch(
            (0.05, 0.06), 0.90, 0.11,
            transform=ax.transAxes,
            boxstyle="round,pad=0.016,rounding_size=0.03",
            linewidth=1.15, edgecolor="#d8bfd8", facecolor="#faf5fb",
        )
    )
    ax.text(
        0.50, 0.115,
        "A single 126 bp band is not exclusive to wild type;\n"
        "for K417N, ambiguous single-band results should be sequence-confirmed.",
        transform=ax.transAxes,
        ha="center", va="center",
        fontsize=8.7, color="#5b3a5e",
        linespacing=1.18,
    )



def add_footer(fig):
    wrapped = textwrap.fill(FIGURE_FOOTER, width=160)
    fig.text(0.5, 0.028, wrapped, ha="center", va="bottom", fontsize=10.6, color="#374151")


# =========================
# Figure builder
# =========================


def build_figure(input_dir: Path, output_prefix: Path, dpi: int, enhance: bool, no_autocrop: bool) -> Dict[str, Path]:
    prepared: Dict[str, np.ndarray] = {}
    for panel in PANELS:
        prepared[panel.key] = prepare_gel_image(
            input_dir / panel.filename,
            panel=panel,
            enhance=enhance,
            no_autocrop=no_autocrop,
        )

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "svg.fonttype": "none",
    })

    fig = plt.figure(figsize=(15.5, 10.6), constrained_layout=False)
    gs = GridSpec(
        2, 3, figure=fig,
        width_ratios=[1.0, 1.12, 1.0],
        height_ratios=[1.0, 1.0],
        left=0.035, right=0.985, top=0.915, bottom=0.09,
        wspace=0.12, hspace=0.14,
    )

    fig.suptitle(FIGURE_TITLE, fontsize=17.6, fontweight="bold", y=0.962)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_f = fig.add_subplot(gs[0, 1])
    ax_b = fig.add_subplot(gs[0, 2])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[1, 2])

    panel_map = {p.key: p for p in PANELS}
    draw_gel_panel(ax_a, prepared["T19I"], panel_map["T19I"])
    draw_k417n_logic(ax_f)
    draw_gel_panel(ax_b, prepared["N440K"], panel_map["N440K"])
    draw_gel_panel(ax_c, prepared["Q677P"], panel_map["Q677P"])
    draw_gel_panel(ax_d, prepared["K417N"], panel_map["K417N"])
    draw_gel_panel(ax_e, prepared["S413R"], panel_map["S413R"])

    add_footer(fig)

    output_prefix = output_prefix.resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    saved: Dict[str, Path] = {}
    for suffix in (".png", ".pdf", ".svg", ".tiff"):
        out = output_prefix.with_suffix(suffix)
        fig.savefig(out, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
        saved[suffix] = out

    plt.close(fig)
    return saved


# =========================
# CLI
# =========================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create a premium Figure S1 from representative gel images with safe, non-aggressive gel handling.")
    p.add_argument("--input-dir", type=Path, default=Path(DEFAULT_INPUT_DIR), help="Directory containing the gel PNG files")
    p.add_argument("--output-prefix", type=Path, default=Path(DEFAULT_OUTPUT_PREFIX), help="Output prefix without extension")
    p.add_argument("--dpi", type=int, default=600, help="Export DPI for PNG/TIFF/PDF rendering")
    p.add_argument("--no-enhance", action="store_true", help="Disable gentle autocontrast/sharpening")
    p.add_argument("--no-autocrop", action="store_true", help="Disable safe border trimming and keep the original gel image extent")
    return p.parse_args()



def main() -> None:
    args = parse_args()
    saved = build_figure(
        input_dir=args.input_dir,
        output_prefix=args.output_prefix,
        dpi=args.dpi,
        enhance=not args.no_enhance,
        no_autocrop=args.no_autocrop,
    )

    print("Saved files:")
    for _, path in saved.items():
        print(f"  {path}")


if __name__ == "__main__":
    main()
