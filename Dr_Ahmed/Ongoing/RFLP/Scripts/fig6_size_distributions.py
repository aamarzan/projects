#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

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

def parse_fragments(s):
    if pd.isna(s): return []
    parts = [p.strip() for p in str(s).replace(",", ";").split(";") if p.strip()]
    out = []
    for p in parts:
        try: out.append(float(p))
        except: pass
    return out

def main(csv_path: Path, out_dir: Path):
    apply_style()
    df = pd.read_csv(csv_path)
    for c in ["amplicon_bp","fragments_bp"]:
        if c not in df.columns:
            raise ValueError(f"assay_table.csv missing '{c}'")

    amplicons = pd.to_numeric(df["amplicon_bp"], errors="coerce").dropna().to_numpy()
    frag_lists = df["fragments_bp"].apply(parse_fragments).tolist()
    fragments = np.array([v for xs in frag_lists for v in xs if np.isfinite(v)], dtype=float)

    fig = plt.figure(figsize=(7.2, 4.8))
    ax1 = fig.add_axes([0.10, 0.58, 0.86, 0.34])
    ax2 = fig.add_axes([0.10, 0.14, 0.86, 0.34])

    ax1.hist(amplicons, bins=15, edgecolor="#111827", linewidth=0.6)
    ax1.set_ylabel("Count")
    ax1.set_title("Figure 6. Distribution of PCR amplicon sizes and expected restriction fragments", loc="left", weight="bold")
    ax1.grid(axis="y", color="#E5E7EB", linewidth=0.6)

    ax2.hist(fragments, bins=20, edgecolor="#111827", linewidth=0.6)
    ax2.set_xlabel("Size (bp)")
    ax2.set_ylabel("Count")
    ax2.grid(axis="y", color="#E5E7EB", linewidth=0.6)

    # highlight small fragments threshold (typical resolution issue)
    ax2.axvline(100, linewidth=1.0, linestyle="--", color="#EF4444")
    ax2.text(102, ax2.get_ylim()[1]*0.90, "100 bp", color="#EF4444", fontsize=8)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / "Figure_6_Size_Distributions"
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".eps"))
    plt.close(fig)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="assay_table.csv")
    p.add_argument("--out", default="out_figures")
    args = p.parse_args()
    main(Path(args.csv), Path(args.out))
