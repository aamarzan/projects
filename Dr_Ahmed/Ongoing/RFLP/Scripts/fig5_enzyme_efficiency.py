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
    for c in ["mutation","enzyme","amplicon_bp","fragments_bp"]:
        if c not in df.columns:
            raise ValueError(f"assay_table.csv missing '{c}'")

    df["frag_list"] = df["fragments_bp"].apply(parse_fragments)
    df["min_sep_bp"] = df["frag_list"].apply(lambda xs: np.nan if len(xs)<2 else np.min(np.diff(sorted(xs))))
    # Enzyme reuse counts
    counts = df["enzyme"].value_counts().sort_values(ascending=True)

    fig = plt.figure(figsize=(7.2, 4.6))
    ax = fig.add_axes([0.18, 0.16, 0.78, 0.76])

    y = np.arange(len(counts))
    ax.barh(y, counts.values, edgecolor="#111827", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(counts.index)
    ax.set_xlabel("Number of loci per enzyme (reuse frequency)")
    ax.set_title("Figure 5. Restriction enzyme reuse across the PCR–RFLP panel", loc="left", weight="bold")
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.6)

    # annotate median fragment separation as “quality hint”
    q = df.dropna(subset=["min_sep_bp"])["min_sep_bp"]
    if len(q) > 0:
        ax.text(0.99, 0.02,
                f"Design QC: median minimum fragment separation = {np.median(q):.0f} bp",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="#374151")

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / "Figure_5_Enzyme_Reuse"
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
