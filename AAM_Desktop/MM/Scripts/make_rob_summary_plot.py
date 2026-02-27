#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

mpl.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

COLOR = {
    "Low": "#2ca25f",
    "Some concerns": "#ffdd57",
    "Unclear": "#bdbdbd",
    "High": "#de2d26",
    "NA": "#ffffff",
}

ORDER = ["Low", "Some concerns", "Unclear", "High"]

def norm(x):
    if pd.isna(x): return "Unclear"
    x = str(x).strip()
    xl = x.lower()
    if xl in ["na", "n/a"]: return "NA"
    if xl in ["low"]: return "Low"
    if xl in ["high"]: return "High"
    if xl in ["some concerns", "concerns", "moderate"]: return "Some concerns"
    if xl in ["unclear", "unknown", "nr", "not reported"]: return "Unclear"
    return x

def save(fig, outbase: Path, dpi: int):
    fig.savefig(outbase.with_suffix(".png"), dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(outbase.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", default="figures/rob")
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)

    domains = [c for c in df.columns if c.startswith("D")]
    studies = df["Study"].astype(str).tolist()

    # Traffic light matrix
    cat_to_int = {"Low":0, "Some concerns":1, "Unclear":2, "High":3, "NA":4}
    mat = np.array([[cat_to_int.get(norm(df.loc[i, d]), 2) for d in domains] for i in range(len(df))])

    cmap = ListedColormap([COLOR["Low"], COLOR["Some concerns"], COLOR["Unclear"], COLOR["High"], COLOR["NA"]])

    fig = plt.figure(figsize=(16, max(5.5, 0.55*len(studies)+2.5)))
    ax = fig.add_subplot(111)
    ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=4, interpolation="nearest")

    ax.set_title("Risk of bias (traffic-light)", fontsize=16, pad=12)
    ax.set_xticks(np.arange(len(domains)))
    ax.set_xticklabels([d.split(" ", 1)[0] for d in domains], rotation=30, ha="right", fontsize=10)
    ax.set_yticks(np.arange(len(studies)))
    ax.set_yticklabels(studies, fontsize=10)

    # subtle grid
    ax.set_xticks(np.arange(-.5, len(domains), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(studies), 1), minor=True)
    ax.grid(which="minor", color="#000", alpha=0.15, linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    # legend
    labels = ["Low","Some concerns","Unclear","High","NA"]
    handles = [plt.Line2D([0],[0], marker="s", color="w", markerfacecolor=COLOR[l], markeredgecolor="#222", markersize=12) for l in labels]
    ax.legend(handles, labels, frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.10))

    # honest note if mostly unclear
    non_na = mat[mat != 4]
    if non_na.size and np.mean(non_na == 2) > 0.80:
        ax.text(0.0, -0.14,
                "Note: Most domains are 'Unclear' because full-text reporting was insufficient for domain-level judgment.",
                transform=ax.transAxes, fontsize=10, color="#444")

    fig.tight_layout()
    save(fig, outdir / "RoB_traffic_light_premium", args.dpi)

    # Domain summary
    props = {c: [] for c in ORDER}
    for d in domains:
        col = df[d].apply(norm)
        total = (col != "NA").sum()
        for c in ORDER:
            props[c].append((col == c).sum()/total if total else 0)

    x = np.arange(len(domains))
    fig2, ax2 = plt.subplots(figsize=(14, 5.6))
    bottom = np.zeros(len(domains))
    for c in ORDER:
        ax2.bar(x, props[c], bottom=bottom, label=c, color=COLOR[c], edgecolor="#222", linewidth=0.4)
        bottom += np.array(props[c])

    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Proportion", fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels([d.split(" ", 1)[0] for d in domains], rotation=30, ha="right", fontsize=10)
    ax2.set_title("Risk of bias summary by domain", fontsize=16, pad=12)
    ax2.legend(frameon=False, ncol=4, loc="upper right")
    ax2.grid(axis="y", alpha=0.15)

    fig2.tight_layout()
    save(fig2, outdir / "RoB_domain_summary_premium", args.dpi)

    print("Saved:", outdir)

if __name__ == "__main__":
    main()