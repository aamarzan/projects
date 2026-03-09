#!/usr/bin/env python3
"""
Figure 2 (Premium v2.3): Targeted mutation panel across Spike domains and non-spike regions.

Requested changes:
1) NO overlaps: tiered label stacking with stronger spacing.
2) NO leader lines/arrows: labels are plain text only (no annotate arrowprops).
3) Track labels (Spike/ORF1ab/N) are INSIDE the bars/border.
4) Title is TOP-CENTER and slightly smaller.
5) Output stem ALWAYS: Figure_2_Panel_Map (PNG/PDF/EPS). EPS-safe (no alpha).

Exports: Figure_2_Panel_Map.png / .pdf / .eps (600 dpi PNG)
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import to_rgb


def apply_style():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.0,
        "axes.linewidth": 0.95,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06,
    })


DEFAULT_ROWS = [
    ("HV69-70del", "S", 69, "NTD"),
    ("T19I", "S", 19, "NTD"),
    ("T95I", "S", 95, "NTD"),
    ("D138Y", "S", 138, "NTD"),
    ("G142D", "S", 142, "NTD"),
    ("W152C", "S", 152, "NTD"),
    ("D215G", "S", 215, "NTD"),
    ("K417N", "S", 417, "RBD"),
    ("N440K", "S", 440, "RBD"),
    ("VG445PS", "S", 445, "RBD"),
    ("L452R", "S", 452, "RBD"),
    ("E484K/Q", "S", 484, "RBD"),
    ("F486P", "S", 486, "RBD"),
    ("Q677P", "S", 677, "S1/S2"),
    ("I692V", "S", 692, "S2"),
    ("A701V", "S", 701, "S2"),
    ("N764K", "S", 764, "S2"),
    ("N856K", "S", 856, "S2"),
    ("L981F", "S", 981, "S2"),
    ("D1118H", "S", 1118, "S2"),
    ("K1191N", "S", 1191, "S2"),
    ("SGF3675-3677del", "ORF1ab", 3675, "ORF1ab"),
    ("R5716C", "ORF1ab", 5716, "ORF1ab"),
    ("ERS31-33del", "N", 31, "N"),
    ("S413R", "N", 413, "N"),
]


DOMAIN_COLORS = {
    "NTD": "#2563EB",
    "RBD": "#7C3AED",
    "S1/S2": "#0EA5E9",
    "S2": "#059669",
    "ORF1ab": "#F59E0B",
    "N": "#EF4444",
}

SPIKE_DOMAINS = [
    ("NTD", 14, 305),
    ("RBD", 319, 541),
    ("S2", 686, 1213),
]
S1S2_SITE = ("S1/S2", 675, 691)


def tint(hex_color: str, amount: float = 0.88):
    """Mix color towards white (EPS-safe; no alpha)."""
    r, g, b = to_rgb(hex_color)
    return (1 - (1 - r) * (1 - amount),
            1 - (1 - g) * (1 - amount),
            1 - (1 - b) * (1 - amount))


def load_panel(csv_path: Path) -> pd.DataFrame:
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(DEFAULT_ROWS, columns=["mutation", "gene", "protein_pos", "domain"])

    for c in ["mutation", "gene", "domain"]:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in {csv_path}")
    if "protein_pos" not in df.columns:
        df["protein_pos"] = np.nan

    df["mutation"] = df["mutation"].astype(str)
    df["gene"] = df["gene"].astype(str)
    df["domain"] = df["domain"].astype(str)
    df["protein_pos"] = pd.to_numeric(df["protein_pos"], errors="coerce")
    return df


def assign_tiers(xs, min_dx):
    """Greedy tiering so labels in same tier are separated by >= min_dx."""
    tiers_last = []
    tiers = []
    for x in xs:
        placed = False
        for t, lastx in enumerate(tiers_last):
            if x - lastx >= min_dx:
                tiers_last[t] = x
                tiers.append(t)
                placed = True
                break
        if not placed:
            tiers_last.append(x)
            tiers.append(len(tiers_last) - 1)
    return tiers


def main(panel_csv: Path, out_dir: Path):
    apply_style()
    df = load_panel(panel_csv).copy()

    fig = plt.figure(figsize=(10.6, 6.1))
    ax = fig.add_axes([0.07, 0.16, 0.91, 0.76])
    ax.set_axisbelow(True)

    XMAX = 1273
    ax.set_xlim(-20, XMAX + 25)
    ax.set_ylim(0.25, 4.55)

    # More spacing between tracks for tiered labels
    y_spike = 3.60
    y_orf = 2.12
    y_n = 0.78

    BAR_H = 0.30
    BAR_YOFF = BAR_H / 2

    def bar(y, face, label):
        ax.add_patch(Rectangle((0, y - BAR_YOFF), XMAX, BAR_H,
                               facecolor=face, edgecolor="#111827", lw=1.0, zorder=1))
        # Label INSIDE the bar; white tag prevents marker â€œcollisionâ€
        ax.text(
            18, y, label,
            ha="left", va="center",
            fontsize=10, weight="bold", color="#0B1220",
            zorder=50,
            bbox=dict(boxstyle="round,pad=0.20,rounding_size=0.10",
                      facecolor="white", edgecolor="none")
        )

    bar(y_spike, "#F3F4F6", "Spike")
    bar(y_orf, "#FFF7ED", "ORF1ab")
    bar(y_n, "#FEF2F2", "N")

    # Spike domains (tinted fills; no alpha)
    for name, a, b in SPIKE_DOMAINS:
        base = DOMAIN_COLORS[name]
        ax.add_patch(Rectangle((a, y_spike - BAR_YOFF), b - a, BAR_H,
                               facecolor=tint(base, 0.90), edgecolor="none", zorder=2))
        ax.text((a + b) / 2, y_spike + 0.26, name,
                ha="center", va="bottom",
                fontsize=11, weight="bold",
                color=base, zorder=6)

    nm, a, b = S1S2_SITE
    base = DOMAIN_COLORS[nm]
    ax.add_patch(Rectangle((a, y_spike - BAR_YOFF), b - a, BAR_H,
                           facecolor=tint(base, 0.86), edgecolor="none", zorder=3))
    ax.text((a + b) / 2, y_spike + 0.26, nm,
            ha="center", va="bottom",
            fontsize=11, weight="bold",
            color=base, zorder=7)

    # Subtle domain boundary ticks
    for _, a, b in SPIKE_DOMAINS:
        ax.plot([a, a], [y_spike - BAR_YOFF - 0.02, y_spike + BAR_YOFF + 0.02], color="#111827", lw=0.6, zorder=4)
        ax.plot([b, b], [y_spike - BAR_YOFF - 0.02, y_spike + BAR_YOFF + 0.02], color="#111827", lw=0.6, zorder=4)

    ORF_LEN = 7096.0
    N_LEN = 419.0

    def map_x(gene, pos):
        if gene == "S":
            return float(pos)
        if gene == "ORF1ab":
            return float(pos) / ORF_LEN * XMAX
        return float(pos) / N_LEN * XMAX

    def track_y(gene):
        return y_spike if gene == "S" else (y_orf if gene == "ORF1ab" else y_n)

    df["x_plot"] = [map_x(g, p) if pd.notna(p) else np.nan for g, p in zip(df["gene"], df["protein_pos"])]
    df["y_plot"] = [track_y(g) for g in df["gene"]]

    # Strong separation => prevents overlaps (and NO leader lines)
    label_cfg = {
        "S":      dict(min_dx=10,  base=0.30, step=0.18, fs=6.3, rot=42, place="below"),
        "ORF1ab": dict(min_dx=160, base=0.30, step=0.32, fs=6.3, rot=42, place="below"),
        "N":      dict(min_dx=60, base=0.25, step=0.32, fs=6.3, rot=42, place="above"),
    }

    def clamp(v, lo, hi):
        return max(lo, min(hi, v))

    def place_label(x, y, text, tier, cfg):
        # gentle x spread + alternating alignment
        x_nudge = (1 if (tier % 2 == 0) else -1) * (7 + 2.4 * tier)
        ha = "right" if (tier % 2 == 0) else "left"

        if cfg["place"] == "below":
            ylab = y - cfg["base"] - tier * cfg["step"]
            va = "top"
        else:
            ylab = y + cfg["base"] + tier * cfg["step"]
            va = "bottom"

        xlab = clamp(x + x_nudge, 0, XMAX)
        ax.text(xlab, ylab, text,
                ha=ha, va=va,
                rotation=cfg["rot"],
                fontsize=cfg["fs"],
                color="#111827",
                zorder=20)

    # Lollipops + labels
    for gene in ["S", "ORF1ab", "N"]:
        sub = df[df["gene"] == gene].sort_values("x_plot").copy()
        xs = sub["x_plot"].to_numpy()
        tiers = assign_tiers(xs, label_cfg[gene]["min_dx"])

        for (_, row), tier in zip(sub.iterrows(), tiers):
            mut = row["mutation"]
            dom = row["domain"]
            x = float(row["x_plot"])
            y = float(row["y_plot"])
            color = DOMAIN_COLORS.get(dom, "#111827")

            ax.plot([x, x], [y - 0.23, y + 0.23], color=color, lw=1.55, zorder=8)
            ax.plot([x], [y], marker="o", ms=8.0,
                    markerfacecolor=color, markeredgecolor="white",
                    markeredgewidth=1.15, zorder=9)

            place_label(x, y, mut, tier, label_cfg[gene])

    ax.set_yticks([])
    ax.set_xlabel("Protein coordinate (Spike to scale; ORF1ab/N mapped to compressed scales for visualization)")
    ax.grid(axis="x", color="#E5E7EB", lw=0.9)

    for sp in ax.spines.values():
        sp.set_color("#111827")
        sp.set_linewidth(1.0)

    # Title: top-center + smaller
    ax.set_title("Targeted mutation panel across Spike domains and non-spike regions",
                 loc="center", weight="bold", fontsize=10, pad=12)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / "Figure_2_Panel_Map"
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".eps"))
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--panel_csv", default="panel_loci.csv", help="CSV for loci (or use defaults if missing)")
    p.add_argument("--out", default="out_figures")
    args = p.parse_args()
    main(Path(args.panel_csv), Path(args.out))

