#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def apply_style():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06,
    })

DEFAULT_ROWS = [
    # mutation, gene, protein_pos, domain
    ("HV69-70del","S",69,"NTD"),
    ("T19I","S",19,"NTD"),
    ("T95I","S",95,"NTD"),
    ("D138Y","S",138,"NTD"),
    ("G142D","S",142,"NTD"),
    ("W152C","S",152,"NTD"),
    ("D215G","S",215,"NTD"),
    ("K417N","S",417,"RBD"),
    ("N440K","S",440,"RBD"),
    ("L452R","S",452,"RBD"),
    ("E484K/Q","S",484,"RBD"),
    ("VG445PS","S",445,"RBD"),
    ("F486P","S",486,"RBD"),
    ("Q677P","S",677,"S1/S2"),
    ("I692V","S",692,"S2"),
    ("A701V","S",701,"S2"),
    ("N764K","S",764,"S2"),
    ("N856K","S",856,"S2"),
    ("L981F","S",981,"S2"),
    ("D1118H","S",1118,"S2"),
    ("K1191N","S",1191,"S2"),
    ("SGF3675-3677del","ORF1ab",3675,"ORF1ab"),
    ("R5716C","ORF1ab",5716,"ORF1ab"),
    ("ERS31_33del","N",31,"N"),
    ("S413R","N",413,"N"),
]

DOMAIN_ORDER = ["NTD","RBD","S1/S2","S2","ORF1ab","N"]
DOMAIN_COLORS = {
    "NTD": "#2563EB",
    "RBD": "#7C3AED",
    "S1/S2": "#0EA5E9",
    "S2": "#059669",
    "ORF1ab": "#F59E0B",
    "N": "#EF4444",
}

def load_panel(csv_path: Path) -> pd.DataFrame:
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(DEFAULT_ROWS, columns=["mutation","gene","protein_pos","domain"])
    for c in ["mutation","gene","domain"]:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in {csv_path}")
    if "protein_pos" not in df.columns:
        df["protein_pos"] = np.nan
    return df

def main(panel_csv: Path, out_dir: Path):
    apply_style()
    df = load_panel(panel_csv).copy()

    fig = plt.figure(figsize=(7.2, 4.2))
    ax = fig.add_axes([0.10, 0.15, 0.86, 0.75])
    ax.set_axisbelow(True)

    # Draw a simplified "tracks" layout (Spike at top + ORF1ab + N)
    tracks = [("S (Spike)", 3.0), ("ORF1ab", 2.0), ("N", 1.0)]
    x0, x1 = 0, 1300  # protein coordinate scale for Spike (approx); ORF1ab/N will just be separate scales
    ax.set_xlim(-40, 1350)
    ax.set_ylim(0.4, 3.6)

    # Spike domains (approx boundaries; edit if you want exact)
    # NTD ~ 14-305, RBD ~ 319-541, S1/S2 region ~ 650-700, S2 ~ 686-1213
    domains = [
        ("NTD", 14, 305),
        ("RBD", 319, 541),
        ("S1/S2", 650, 700),
        ("S2", 686, 1213),
    ]
    y_spike = 3.0
    ax.add_patch(Rectangle((0, y_spike-0.12), 1273, 0.24, facecolor="#F3F4F6", edgecolor="#111827", lw=0.8))
    for name, a, b in domains:
        ax.add_patch(Rectangle((a, y_spike-0.12), b-a, 0.24, facecolor=DOMAIN_COLORS[name], alpha=0.18, edgecolor="none"))
        ax.text((a+b)/2, y_spike+0.18, name, ha="center", va="bottom", fontsize=8, color=DOMAIN_COLORS[name])

    # ORF1ab and N simple bars (not to scale; “map” purpose)
    ax.add_patch(Rectangle((0, 2.0-0.12), 1300, 0.24, facecolor="#FFF7ED", edgecolor="#111827", lw=0.8))
    ax.text(-25, 2.0, "ORF1ab", ha="right", va="center", fontsize=9)
    ax.add_patch(Rectangle((0, 1.0-0.12), 1300, 0.24, facecolor="#FEF2F2", edgecolor="#111827", lw=0.8))
    ax.text(-25, 1.0, "N", ha="right", va="center", fontsize=9)

    # Plot loci (Spike uses protein_pos; ORF1ab/N also use protein_pos field as index-like)
    def plot_locus(row):
        mut = str(row["mutation"])
        dom = str(row["domain"])
        gene = str(row["gene"])
        pos = float(row["protein_pos"]) if pd.notna(row["protein_pos"]) else np.nan

        if gene == "S":
            y = 3.0
            x = pos
        elif gene == "ORF1ab":
            y = 2.0
            x = min(max(pos/7.0, 10), 1290)  # compress large ORF1ab aa positions into same canvas
        else:  # N
            y = 1.0
            x = min(max(pos*3.0, 10), 1290)

        color = DOMAIN_COLORS.get(dom, "#111827")
        ax.plot([x], [y], marker="o", ms=5, color=color, zorder=5)
        ax.vlines(x, y-0.18, y+0.18, color=color, lw=0.9, alpha=0.9)
        ax.text(x, y-0.28, mut, ha="center", va="top", rotation=45, fontsize=7, color="#111827")

    for _, r in df.iterrows():
        plot_locus(r)

    ax.set_yticks([])
    ax.set_xlabel("Protein coordinate (Spike shown approximately; ORF1ab/N shown on compressed scales for visualization)")
    ax.set_title("Figure 2. Targeted mutation panel across Spike domains and non-spike regions", loc="left", weight="bold")

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
