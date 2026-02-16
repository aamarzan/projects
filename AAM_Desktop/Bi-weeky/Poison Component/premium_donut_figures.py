# premium_donut_triptych.py
# Creates 3 premium donut charts with outside, non-overlapping labels + clean leader lines
# Outputs: PNG (600 dpi) + PDF + SVG for each figure

from __future__ import annotations

import math
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe


# -----------------------------
# Premium palettes (non-gradient)
# -----------------------------
PALETTE_BLUE = [
    "#0B1F3B", "#143B63", "#1F5F8B", "#2E7FA5", "#48A3C8",
    "#74C3D8", "#A5D8E6", "#D7EEF5", "#4B5563", "#9CA3AF",
]
PALETTE_GREEN = [
    "#0B3B2E", "#0F5A42", "#177A57", "#1F9A6E", "#31B487",
    "#5CC7A2", "#90D9C1", "#C8EEE1", "#4B5563", "#9CA3AF",
]
PALETTE_PURPLE = [
    "#2A0A3D", "#3E145C", "#56207A", "#6D2F97", "#8647B0",
    "#A069C6", "#BE9AD9", "#E6D6F1", "#4B5563", "#9CA3AF",
]


def spread_positions(
    y: List[float],
    min_sep: float = 0.24,
    y_min: float = -1.05,
    y_max: float = 1.05,
) -> List[float]:
    """
    Enforces minimum vertical separation between label y-positions.
    Returns y positions in the same order as input.
    """
    if len(y) == 0:
        return []

    order = np.argsort(y)
    ys = np.array(y, dtype=float)[order]
    out = ys.copy()

    # forward pass
    for i in range(1, len(out)):
        if out[i] - out[i - 1] < min_sep:
            out[i] = out[i - 1] + min_sep

    # shift down if overflow
    overflow = out[-1] - y_max
    if overflow > 0:
        out -= overflow

    # backward pass (respect min_sep and y_min)
    for i in range(len(out) - 2, -1, -1):
        if out[i + 1] - out[i] < min_sep:
            out[i] = out[i + 1] - min_sep

    under = y_min - out[0]
    if under > 0:
        out += under
        # re-forward pass if needed
        for i in range(1, len(out)):
            if out[i] - out[i - 1] < min_sep:
                out[i] = out[i - 1] + min_sep
        overflow = out[-1] - y_max
        if overflow > 0:
            out -= overflow

    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    return out[inv].tolist()


def donut_with_callouts(
    ax,
    labels: List[str],
    values: List[int],
    title: str,
    palette: List[str],
    startangle: float = 90,
    donut_width: float = 0.36,
    label_font: int = 12,
    pct_decimals: int = 1,
    # geometry
    r_anchor: float = 1.02,
    r_radial: float = 1.22,
    x_elbow: float = 1.32,
    x_text: float = 1.62,
    min_sep: float = 0.26,
    # styling
    line_color: str = "#6e6e6e",
    line_lw: float = 1.6,
    text_color: str = "#111111",
):
    total = float(sum(values)) if sum(values) else 1.0

    colors = [palette[i % len(palette)] for i in range(len(values))]
    wedges, _ = ax.pie(
        values,
        startangle=startangle,
        colors=colors,
        wedgeprops=dict(width=donut_width, edgecolor="white", linewidth=2.0),
    )

    ax.text(
        0,
        0,
        "Count\n(%)",
        ha="center",
        va="center",
        fontsize=label_font + 4,
        fontweight="bold",
        color="#333333",
    )

    ax.set_title(title, fontsize=label_font + 10, fontweight="bold", pad=24)

    items = []
    for w, lab, val in zip(wedges, labels, values):
        ang = (w.theta1 + w.theta2) / 2.0
        ang_rad = np.deg2rad(ang)
        x = math.cos(ang_rad)
        y = math.sin(ang_rad)
        side = 1 if x >= 0 else -1

        pct = (val / total) * 100.0
        txt = f"{lab}\n{val:,} ({pct:.{pct_decimals}f}%)"
        items.append(dict(txt=txt, ang=ang_rad, x=x, y=y, side=side))

    left = [it for it in items if it["side"] < 0]
    right = [it for it in items if it["side"] > 0]

    def layout(side_items: List[dict], sign: int) -> List[dict]:
        if not side_items:
            return []
        # Sort by natural y (top to bottom) to keep tidy stacks
        side_items = sorted(side_items, key=lambda d: d["y"], reverse=True)
        pref_y = [it["y"] for it in side_items]
        y_adj = spread_positions(pref_y, min_sep=min_sep, y_min=-1.05, y_max=1.05)
        for it, y_text in zip(side_items, y_adj):
            it["y_text"] = y_text
            it["x_elbow"] = sign * x_elbow
            it["x_text"] = sign * x_text
        return side_items

    left = layout(left, -1)
    right = layout(right, +1)

    for it in left + right:
        ang = it["ang"]
        side = it["side"]

        # points: anchor -> radial-out -> elbow -> horizontal to text
        xa, ya = (r_anchor * math.cos(ang), r_anchor * math.sin(ang))
        xr, yr = (r_radial * math.cos(ang), r_radial * math.sin(ang))
        xe, ye = (it["x_elbow"], it["y_text"])
        xt, yt = (it["x_text"], it["y_text"])

        ax.plot([xa, xr], [ya, yr], color=line_color, lw=line_lw, solid_capstyle="round", zorder=3)
        ax.plot([xr, xe], [yr, ye], color=line_color, lw=line_lw, solid_capstyle="round", zorder=3)
        ax.plot([xe, xt - 0.05 * side], [ye, yt], color=line_color, lw=line_lw, solid_capstyle="round", zorder=3)

        ha = "left" if side > 0 else "right"
        t = ax.text(
            xt,
            yt,
            it["txt"],
            ha=ha,
            va="center",
            fontsize=label_font,
            color=text_color,
            linespacing=1.15,
        )
        # white stroke behind text = “premium” readability
        t.set_path_effects([pe.withStroke(linewidth=3.8, foreground="white")])

    ax.set_aspect("equal")
    ax.set_xlim(-1.9, 1.9)
    ax.set_ylim(-1.25, 1.25)
    ax.axis("off")


def make_one(outdir: Path, title: str, data: Dict[str, int], palette: List[str], basename: str):
    # Sort by count (desc) for nicer reading
    items = sorted(data.items(), key=lambda kv: kv[1], reverse=True)
    labels = [k for k, _ in items]
    values = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(12.5, 9), dpi=150)
    donut_with_callouts(
        ax,
        labels=labels,
        values=values,
        title=title,
        palette=palette,
        min_sep=0.28,      # key for preventing label overlap (2-line labels)
        label_font=12,
    )

    out_png = outdir / f"{basename}.png"
    out_pdf = outdir / f"{basename}.pdf"
    out_svg = outdir / f"{basename}.svg"

    fig.savefig(out_png, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(out_svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="donut_outputs", help="Output folder")
    args = parser.parse_args()

    # Make text editable in Illustrator/Inkscape
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    drug_overdose = {
        "Sedative": 2196,
        "Other Benzodiazepine": 884,
        "Clonazepam": 33,
        "Unknown Drug": 1631,
        "Multidrug": 581,
        "Paracetamol": 177,
        "TCA": 22,
    }

    insecticide = {
        "Unknown Insecticide": 760,
        "Cypermethrin": 117,
        "Lambda Cyhalothrin": 24,
        "Cockroach Killer": 48,
        "Ant Killer": 32,
        "Mosquito Killer": 20,
        "Lice Killer": 19,
    }

    household = {
        "Harpic": 1324,
        "Unknown Household Product": 1199,
        "Savlon": 115,
        "Dettol": 24,
        "Vixol": 52,
        "Hexisol": 24,
        "Wheel Powder": 36,
        "Corrosive": 549,
        "Acid Ingestion": 23,
    }

    make_one(outdir, "Drug overdose (N=5,524)", drug_overdose, PALETTE_BLUE, "donut_drug_overdose")
    make_one(outdir, "Insecticide (N=1,020)", insecticide, PALETTE_GREEN, "donut_insecticide")
    make_one(outdir, "Household products (N=3,346)", household, PALETTE_PURPLE, "donut_household_products")

    print("Done. Files saved to:", outdir.resolve())


if __name__ == "__main__":
    main()
