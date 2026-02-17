#!/usr/bin/env python3
"""
Figure 1 (Premium v5.2): Clean workflow schematic (no overflow, premium gradient headers).
- EPS-safe (no transparency): gradients drawn with solid strips; soft shadow uses solid light patch.
- Cross-lane arrows are routed around boxes (journal-friendly; no overlaps).

Option A implemented:
- Restriction mapping -> RFLP digestion + gel (routed, endpoint shifted to avoid label overlap)
- RFLP digestion + gel -> Consensus mutation call (routed, no overlaps)
- No Expected pattern set -> Consensus arrow

Exports: PNG (600 dpi), PDF, EPS
"""

import argparse
from pathlib import Path
import textwrap

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
from matplotlib.colors import to_rgb
from matplotlib.path import Path as MplPath


# ------------------------- STYLE ------------------------- #
def apply_style():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",   # if Arial installed: change to "Arial"
        "font.size": 9.0,
        "axes.linewidth": 0.8,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06,
    })


# ------------------------- HELPERS ------------------------- #
def row_positions(n, left=0.04, right=0.02, gap=0.03):
    usable = 1 - left - right - gap * (n - 1)
    w = usable / n
    xs = [left + i * (w + gap) for i in range(n)]
    return xs, w


def lerp(a, b, t):
    return a + (b - a) * t


def draw_vertical_gradient(ax, x, y, w, h, c_top, c_bottom, steps=90, z=3):
    """EPS-safe gradient: many thin rectangles."""
    rt, gt, bt = to_rgb(c_top)
    rb, gb, bb = to_rgb(c_bottom)
    dh = h / steps
    for i in range(steps):
        t = i / (steps - 1)
        r = lerp(rt, rb, t)
        g = lerp(gt, gb, t)
        b = lerp(bt, bb, t)
        ax.add_patch(Rectangle(
            (x, y + i * dh), w, dh,
            facecolor=(r, g, b), edgecolor="none", zorder=z
        ))


def wrap_for_box(text, w_axes):
    """Heuristic wrap based on box width in axes coordinates."""
    chars = max(16, int(w_axes * 120))
    return "\n".join(textwrap.fill(line, width=chars) for line in text.split("\n"))


def add_step(ax, x, y, w, h, title, body,
             header_h_ratio=0.34,
             border="#111827",
             grad_top="#EAFFFF",
             grad_bottom="#81D1F3",
             rounding=0.020,
             wrap_title=True,
             title_fs=9.2,
             shadow=True):
    """Draw a premium rounded box with EPS-safe gradient header and clipped text."""

    # Subtle shadow (EPS-safe: solid light patch, no alpha)
    if shadow:
        ax.add_patch(FancyBboxPatch(
            (x + 0.004, y - 0.004), w, h,
            boxstyle=f"round,pad=0.012,rounding_size={rounding}",
            linewidth=0.0, edgecolor="none", facecolor="#E5E7EB",
            zorder=1
        ))

    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.012,rounding_size={rounding}",
        linewidth=1.1, edgecolor=border, facecolor="white",
        zorder=2
    )
    ax.add_patch(box)

    header_h = h * header_h_ratio
    header_y = y + h - header_h
    draw_vertical_gradient(ax, x, header_y, w, header_h, grad_top, grad_bottom, steps=95, z=3)

    ax.plot([x, x + w], [header_y, header_y], color=border, lw=0.9, zorder=4)

    # Thin highlight at very top of header (premium sheen; EPS-safe)
    ax.plot([x + 0.002, x + w - 0.002], [y + h - 0.002, y + h - 0.002],
            color="#FFFFFF", lw=0.7, zorder=4)

    t = wrap_for_box(title, w_axes=w) if wrap_title else title
    title_text = ax.text(
        x + 0.014, header_y + header_h / 2,
        t, ha="left", va="center",
        fontsize=title_fs, weight="bold", color="#0B1220",
        clip_on=True, zorder=5
    )
    title_text.set_clip_path(box)

    b = wrap_for_box(body, w_axes=w)
    body_text = ax.text(
        x + 0.014, y + (header_y - y) / 2,
        b, ha="left", va="center",
        fontsize=8.4, color="#111827",
        clip_on=True, zorder=5
    )
    body_text.set_clip_path(box)

    return box


def arrow(ax, p1, p2, color="#111827", lw=1.05, ms=12, z=10, shrinkA=10, shrinkB=10):
    ax.add_patch(FancyArrowPatch(
        p1, p2,
        arrowstyle="-|>", mutation_scale=ms,
        linewidth=lw, color=color,
        shrinkA=shrinkA, shrinkB=shrinkB,
        capstyle="round", joinstyle="round",
        zorder=z
    ))


def routed_arrow(ax, points, color="#111827", lw=1.10, ms=12, z=10, shrinkA=12, shrinkB=12):
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1)
    path = MplPath(points, codes)
    patch = FancyArrowPatch(
        path=path,
        arrowstyle="-|>", mutation_scale=ms,
        linewidth=lw, color=color,
        shrinkA=shrinkA, shrinkB=shrinkB,
        capstyle="round", joinstyle="round",
        zorder=z
    )
    ax.add_patch(patch)
    return patch


def mid_right(box):  return (box.get_x() + box.get_width(), box.get_y() + box.get_height() / 2)
def mid_left(box):   return (box.get_x(), box.get_y() + box.get_height() / 2)
def mid_top(box):    return (box.get_x() + box.get_width() / 2, box.get_y() + box.get_height())
def mid_bottom(box): return (box.get_x() + box.get_width() / 2, box.get_y())


# ------------------------- FIGURE ------------------------- #
def build(out_dir: Path, stem="Figure_1_Workflow"):
    apply_style()

    fig = plt.figure(figsize=(10.6, 4.9))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(
        0.50, 0.985,
        "End-to-end workflow for PCR–RFLP verification of SARS-CoV-2 mutations",
        ha="center", va="top",
        fontsize=16, weight="bold", color="#0B1220",
        zorder=20
    )

    # Lane backgrounds (keep your exact colors)
    ax.add_patch(Rectangle((0.02, 0.64), 0.96, 0.20, facecolor="#F0FFFD", edgecolor="none", zorder=0))
    ax.add_patch(Rectangle((0.02, 0.36), 0.96, 0.20, facecolor="#F1FDFF", edgecolor="none", zorder=0))
    ax.add_patch(Rectangle((0.02, 0.08), 0.96, 0.20, facecolor="#EFF5FF", edgecolor="none", zorder=0))

    ax.plot([0.02, 0.98], [0.64, 0.64], color="#CBD5E1", lw=0.7, zorder=0)
    ax.plot([0.02, 0.98], [0.36, 0.36], color="#CBD5E1", lw=0.7, zorder=0)

    ax.text(0.02, 0.84, "Wet-lab (Sample → PCR–RFLP)", ha="left", va="bottom",
            fontsize=12, weight="bold", color="#0B1220", zorder=20)
    ax.text(0.02, 0.56, "In-silico (Assay design & expected patterns)", ha="left", va="bottom",
            fontsize=12, weight="bold", color="#0B1220", zorder=20)
    ax.text(0.02, 0.28, "Verification & Reporting", ha="left", va="bottom",
            fontsize=12, weight="bold", color="#0B1220", zorder=20)

    # ---- Premium: per-box light gradients (cool, formal, distinct) ----
    wet_grads = [
        ("#F1FFFF", "#8EDAF7"),
        ("#F2FBFF", "#93CCFF"),
        ("#F3F7FF", "#A1B8FF"),
        ("#F4F3FF", "#B7AEFF"),
        ("#F0FFFA", "#89E7D4"),
    ]
    insilico_grads = [
        ("#F1FFFD", "#86E3D3"),
        ("#F0FAFF", "#86D7FF"),
        ("#F2F7FF", "#99C4FF"),
        ("#F5F7FF", "#AEBBFF"),
    ]
    verify_grads = [
        ("#FFF9F1", "#FFCFA3"),
        ("#FFF5F7", "#FFB7C8"),
        ("#F3F2FF", "#BDB0FF"),
    ]

    # -------- Row 1 --------
    y1, h1 = 0.67, 0.15
    xs1, w1 = row_positions(5, left=0.04, right=0.02, gap=0.03)

    s1 = add_step(ax, xs1[0], y1, w1, h1, "Clinical specimen",
                  "Nasopharyngeal swab (or equivalent)",
                  grad_top=wet_grads[0][0], grad_bottom=wet_grads[0][1])
    s2 = add_step(ax, xs1[1], y1, w1, h1, "RNA extraction + QC",
                  "Yield/quality check; accept or repeat",
                  grad_top=wet_grads[1][0], grad_bottom=wet_grads[1][1])
    s3 = add_step(ax, xs1[2], y1, w1, h1, "RT-qPCR screening",
                  "Ct thresholding; select positives",
                  grad_top=wet_grads[2][0], grad_bottom=wet_grads[2][1])
    s4 = add_step(ax, xs1[3], y1, w1, h1, "cDNA synthesis → PCR",
                "Amplify mutation locus panel (25 loci)",
                grad_top=wet_grads[3][0], grad_bottom=wet_grads[3][1],
                wrap_title=False, title_fs=9.0)
    s5 = add_step(ax, xs1[4], y1, w1, h1, "RFLP digestion + gel",
                  "Banding pattern comparison (mutant vs wild)",
                  grad_top=wet_grads[4][0], grad_bottom=wet_grads[4][1],
                  wrap_title=False, title_fs=9.0)   # <-- keep in ONE line

    for a, b in [(s1, s2), (s2, s3), (s3, s4), (s4, s5)]:
        arrow(ax, mid_right(a), mid_left(b), color="#0B1220")

    # -------- Row 2 --------
    y2, h2 = 0.39, 0.15
    xs2, w2 = row_positions(4, left=0.04, right=0.02, gap=0.04)

    i1 = add_step(ax, xs2[0], y2, w2, h2, "Reference & target selection",
                "Select mutation loci (Spike + non-Spike)",
                grad_top=insilico_grads[0][0], grad_bottom=insilico_grads[0][1],
                wrap_title=False, title_fs=9.0)
    i2 = add_step(ax, xs2[1], y2, w2, h2, "Primer design (in-silico)",
                  "Amplicon size + specificity checks",
                  wrap_title=False, title_fs=9.0,
                  grad_top=insilico_grads[1][0], grad_bottom=insilico_grads[1][1])
    i3 = add_step(ax, xs2[2], y2, w2, h2, "Restriction mapping",
                  "Enzyme selection + expected fragments",
                  grad_top=insilico_grads[2][0], grad_bottom=insilico_grads[2][1])
    i4 = add_step(ax, xs2[3], y2, w2, h2, "Expected pattern set",
                  "Interpretation key (wild vs mutant)",
                  grad_top=insilico_grads[3][0], grad_bottom=insilico_grads[3][1])

    for a, b in [(i1, i2), (i2, i3), (i3, i4)]:
        arrow(ax, mid_right(a), mid_left(b), color="#0B3B2E")

    # -------- Row 3 --------
    y3, h3 = 0.11, 0.15
    xs3, w3 = row_positions(3, left=0.04, right=0.02, gap=0.04)

    v1 = add_step(ax, xs3[0], y3, w3, h3, "Representative Sanger confirmation",
                  "Confirm selected loci (trace + alignment)",
                  grad_top=verify_grads[0][0], grad_bottom=verify_grads[0][1])
    v2 = add_step(ax, xs3[1], y3, w3, h3, "Consensus mutation call",
                  "PCR–RFLP + Sanger consistency check",
                  grad_top=verify_grads[1][0], grad_bottom=verify_grads[1][1])
    v3 = add_step(ax, xs3[2], y3, w3, h3, "Deliverables for manuscript",
                  "Panel map; signature matrix; design QC plots; representative gels",
                  grad_top=verify_grads[2][0], grad_bottom=verify_grads[2][1])

    arrow(ax, mid_right(v1), mid_left(v2), color="#7C2D12")
    arrow(ax, mid_right(v2), mid_left(v3), color="#7C2D12")

    # -------- Cross-lane arrows (ROUTED; no overlaps) --------

    # (1) Restriction mapping -> RFLP digestion + gel
    # Shift endpoint to RIGHT-bottom of s5 so the vertical segment does NOT cross the label text.
    y_corridor_12 = (y2 + h2 + y1) / 2
    x_i3 = mid_top(i3)[0]

    # NEW: endpoint shifted right along the bottom edge of s5
    p_end_s5 = (s5.get_x() + s5.get_width() * 0.82, s5.get_y())  # <-- key fix (no overlap)
    x_end = p_end_s5[0]

    routed_arrow(
        ax,
        [mid_top(i3), (x_i3, y_corridor_12), (x_end, y_corridor_12), p_end_s5],
        color="#0F766E",
        lw=1.15, ms=12, z=12,
        shrinkA=12, shrinkB=12
    )

    # Label (with solid white bbox so nothing visually clashes; EPS-safe)
    ax.text(
        (x_i3 + mid_bottom(s5)[0]) / 2,
        y_corridor_12 + 0.022,
        "Expected fragments guide gel interpretation",
        ha="center", va="center",
        fontsize=10, color="#0F766E",
        zorder=15,
        bbox=dict(facecolor="white", edgecolor="none", pad=0.15)
    )

    # (2) GEL -> Consensus mutation call (start from RIGHT side of RFLP box)
    y_corridor_23 = (y3 + h3 + y2) / 2
    x_v2t = mid_top(v2)[0]
    x_margin = 0.975

    p_start = mid_right(s5)
    y_start = p_start[1]

    routed_arrow(
        ax,
        [p_start,
         (x_margin, y_start),
         (x_margin, y_corridor_23),
         (x_v2t, y_corridor_23),
         mid_top(v2)],
        color="#7C2D12",
        lw=1.15, ms=12, z=12,
        shrinkA=12, shrinkB=14
    )

    ax.text(
        0.02, 0.03,
        "Abbreviations: RT-qPCR, reverse transcription quantitative PCR; RFLP, restriction fragment length polymorphism.",
        ha="left", va="bottom",
        fontsize=9.2, color="#374151",
        zorder=20
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / stem
    fig.savefig(base.with_suffix(".png"), dpi=600)
    fig.savefig(base.with_suffix(".pdf"))
    fig.savefig(base.with_suffix(".eps"))
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out_figures")
    ap.add_argument("--name", default="Figure_1_Workflow")
    args = ap.parse_args()
    build(Path(args.out), args.name)
