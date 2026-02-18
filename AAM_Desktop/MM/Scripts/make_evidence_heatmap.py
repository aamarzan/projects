#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="evidence_matrix_auto.csv or your edited matrix")
    ap.add_argument("--outdir", default="FIGS_HEATMAP")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)

    if "Study" not in df.columns:
        raise SystemExit("CSV must have a 'Study' column.")

    feats = [c for c in df.columns if c != "Study"]
    mat = df[feats].fillna(0).astype(int).to_numpy()
    studies = df["Study"].astype(str).tolist()

    fig_w = max(9, 0.45*len(feats)+4)
    fig_h = max(5, 0.35*len(studies)+2.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(mat, aspect="auto")

    ax.set_xticks(np.arange(len(feats)))
    ax.set_xticklabels(feats, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(studies)))
    ax.set_yticklabels(studies, fontsize=9)

    ax.set_title("Bangladesh evidence coverage heatmap (reported domains/outcomes)", fontsize=14, pad=12)

    # add cell text 0/1
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, str(mat[i, j]), ha="center", va="center", fontsize=8)

    plt.tight_layout()
    fig.savefig(outdir/"Evidence_heatmap.png", dpi=args.dpi)
    fig.savefig(outdir/"Evidence_heatmap.pdf")
    plt.close(fig)

    # also output coverage counts
    counts = mat.sum(axis=0)
    fig2, ax2 = plt.subplots(figsize=(max(9, 0.45*len(feats)+4), 4.8))
    ax2.bar(np.arange(len(feats)), counts)
    ax2.set_xticks(np.arange(len(feats)))
    ax2.set_xticklabels(feats, rotation=30, ha="right", fontsize=9)
    ax2.set_ylabel("Number of studies reporting")
    ax2.set_title("Coverage counts", fontsize=14, pad=12)
    plt.tight_layout()
    fig2.savefig(outdir/"Evidence_coverage_counts.png", dpi=args.dpi)
    fig2.savefig(outdir/"Evidence_coverage_counts.pdf")
    plt.close(fig2)

    print(f"Saved heatmap + counts in: {outdir}")

if __name__ == "__main__":
    main()
