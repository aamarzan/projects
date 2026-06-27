#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def apply_style():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06,
    })

def main(csv_path: Path, out_dir: Path):
    apply_style()
    df = pd.read_csv(csv_path)
    if "Variant" not in df.columns:
        raise ValueError("signature_matrix.csv must contain a 'Variant' column")

    variants = df["Variant"].astype(str).tolist()
    mat = df.drop(columns=["Variant"]).to_numpy(dtype=float)
    cols = df.drop(columns=["Variant"]).columns.astype(str).tolist()

    # premium, legible colormap (0=light, 1=mid, 2=deep)
    cmap = LinearSegmentedColormap.from_list("sig", ["#F8FAFC", "#60A5FA", "#1D4ED8"])

    fig = plt.figure(figsize=(7.2, 4.8))
    ax = fig.add_axes([0.12, 0.20, 0.80, 0.70])

    im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap=cmap, vmin=np.nanmin(mat), vmax=np.nanmax(mat))

    ax.set_yticks(range(len(variants)))
    ax.set_yticklabels(variants)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=60, ha="right")

    # gridlines for crisp matrix
    ax.set_xticks(np.arange(-.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(variants), 1), minor=True)
    ax.grid(which="minor", color="#E5E7EB", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    cax = fig.add_axes([0.93, 0.20, 0.02, 0.70])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Signature value")

    ax.set_title("Figure 4. Variant signature matrix across the targeted mutation panel", loc="left", weight="bold")
    ax.set_xlabel("Targeted loci (assay panel)")
    ax.set_ylabel("Variant / lineage group")

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / "Figure_4_Signature_Heatmap"
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".eps"))
    plt.close(fig)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="signature_matrix.csv")
    p.add_argument("--out", default="out_figures")
    args = p.parse_args()
    main(Path(args.csv), Path(args.out))
