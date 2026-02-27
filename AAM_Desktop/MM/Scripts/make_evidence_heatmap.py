#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

mpl.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

def save(fig, outbase: Path, dpi: int):
    fig.savefig(outbase.with_suffix(".png"), dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(outbase.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

def gradient_bar(ax, x_center, height, width, cmap, edge="#222", lw=0.6, n=256):
    if height <= 0:
        ax.add_patch(Rectangle((x_center - width/2, 0), width, 0.0001,
                               facecolor="none", edgecolor=edge, linewidth=lw))
        return
    grad = np.linspace(0, 1, n).reshape(n, 1)
    ax.imshow(
        grad,
        extent=(x_center - width/2, x_center + width/2, 0, height),
        origin="lower",
        aspect="auto",
        cmap=cmap,
        zorder=2,
        interpolation="bicubic",
    )
    ax.add_patch(Rectangle((x_center - width/2, 0), width, height,
                           fill=False, edgecolor=edge, linewidth=lw, zorder=3))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", default="figures/heatmap")
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--title_suffix", default="(extractable studies, n=8)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    if "Study" not in df.columns:
        raise SystemExit("CSV must have a 'Study' column.")

    feats = [c for c in df.columns if c != "Study"]
    mat = df[feats].fillna(0).astype(int)
    studies = df["Study"].astype(str)

    # Order rows/cols by coverage
    row_sum = mat.sum(axis=1)
    col_sum = mat.sum(axis=0)
    row_order = row_sum.sort_values(ascending=False).index
    col_order = col_sum.sort_values(ascending=False).index
    mat = mat.loc[row_order, col_order]
    studies = studies.loc[row_order]
    feats = list(col_order)
    col_sum = mat.sum(axis=0).to_numpy()

    # Premium colormaps
    heat_cmap = LinearSegmentedColormap.from_list("heatbin", ["#f5f7fb", "#0b5cad"])
    bar_cmap  = LinearSegmentedColormap.from_list("bargrad",  ["#dbeafe", "#1d4ed8"])

    # === Master figure: equal split 1:1 ===
    fig = plt.figure(figsize=(18, 7.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.18)

    # Panel A: heatmap
    axA = fig.add_subplot(gs[0, 0])
    im = axA.imshow(mat.to_numpy(), aspect="auto", cmap=heat_cmap, vmin=0, vmax=1, interpolation="nearest")

    axA.set_title(f"Bangladesh evidence coverage heatmap {args.title_suffix}", fontsize=16, pad=12)
    axA.set_xticks(np.arange(len(feats)))
    axA.set_xticklabels(feats, rotation=30, ha="right", fontsize=10)
    axA.set_yticks(np.arange(len(studies)))
    axA.set_yticklabels(studies, fontsize=10)

    # subtle grid
    axA.set_xticks(np.arange(-.5, len(feats), 1), minor=True)
    axA.set_yticks(np.arange(-.5, len(studies), 1), minor=True)
    axA.grid(which="minor", color="#000", alpha=0.10, linewidth=0.6)
    axA.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, ax=axA, fraction=0.030, pad=0.02)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Not reported", "Reported"])

    # Panel B: gradient coverage counts
    axB = fig.add_subplot(gs[0, 1])
    x = np.arange(len(feats))
    ymax = int(col_sum.max()) + 1
    axB.set_ylim(0, ymax)
    axB.set_xlim(-0.6, len(feats)-0.4)

    for i, v in enumerate(col_sum):
        gradient_bar(axB, i, float(v), width=0.78, cmap=bar_cmap)
        axB.text(i, v + 0.05, str(int(v)), ha="center", va="bottom", fontsize=10)

    axB.set_xticks(x)
    axB.set_xticklabels(feats, rotation=30, ha="right", fontsize=10)
    axB.set_ylabel("Number of studies reporting", fontsize=12)
    axB.set_title("Coverage counts", fontsize=16, pad=12)
    axB.grid(axis="y", alpha=0.15)

    # Panel labels

    fig.tight_layout()
    # Panel labels (placed using figure coordinates to avoid overlapping the panel titles)
    posA = axA.get_position()
    posB = axB.get_position()
    fig.text(posA.x0 - 0.06, posA.y1 + 0.01, "A", fontsize=16, fontweight="bold")
    fig.text(posB.x0 - 0.02, posB.y1 + 0.01, "B", fontsize=16, fontweight="bold")
    save(fig, outdir / "Evidence_Map_premium", args.dpi)

    # Standalone outputs (optional, still premium)
    fig2, ax2 = plt.subplots(figsize=(12, 5.8))
    ax2.set_ylim(0, ymax)
    ax2.set_xlim(-0.6, len(feats)-0.4)
    for i, v in enumerate(col_sum):
        gradient_bar(ax2, i, float(v), width=0.78, cmap=bar_cmap)
        ax2.text(i, v + 0.05, str(int(v)), ha="center", va="bottom", fontsize=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(feats, rotation=30, ha="right", fontsize=10)
    ax2.set_ylabel("Number of studies reporting", fontsize=12)
    ax2.set_title("Coverage counts", fontsize=16, pad=12)
    ax2.grid(axis="y", alpha=0.15)
    fig2.tight_layout()
    save(fig2, outdir / "Evidence_coverage_counts_premium", args.dpi)

    print("Saved:", outdir)

if __name__ == "__main__":
    main()