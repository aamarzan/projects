#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

COLOR = {
    "Low": "#2ca25f",
    "Some concerns": "#ffdd57",
    "Unclear": "#bdbdbd",
    "High": "#de2d26",
    "NA": "#ffffff",
    "N/A": "#ffffff",
    "": "#ffffff",
}

def normalize(x):
    if pd.isna(x): return "Unclear"
    x = str(x).strip()
    if x.lower() in ["n/a", "na"]: return "NA"
    if x.lower() in ["low"]: return "Low"
    if x.lower() in ["high"]: return "High"
    if x.lower() in ["some concerns", "concerns", "moderate"]: return "Some concerns"
    if x.lower() in ["unclear", "unknown", "nr", "not reported"]: return "Unclear"
    return x

def traffic_light(df, domains, outpath, dpi=300):
    studies = df["Study"].tolist()
    nrow, ncol = len(studies), len(domains)

    fig_w = max(8, 0.45*ncol + 4)
    fig_h = max(4, 0.35*nrow + 2.5)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, ncol)
    ax.set_ylim(0, nrow)
    ax.invert_yaxis()

    # cells
    for i, s in enumerate(studies):
        for j, d in enumerate(domains):
            v = normalize(df.loc[i, d])
            ax.add_patch(Rectangle((j, i), 1, 1, facecolor=COLOR.get(v, COLOR["Unclear"]),
                                   edgecolor="#333333", linewidth=0.6))
            ax.text(j+0.5, i+0.52, v if v not in ["NA","N/A"] else "",
                    ha="center", va="center", fontsize=8)

    ax.set_xticks([k+0.5 for k in range(ncol)])
    ax.set_xticklabels(domains, rotation=30, ha="right", fontsize=9)
    ax.set_yticks([k+0.5 for k in range(nrow)])
    ax.set_yticklabels(studies, fontsize=9)
    ax.set_title("Risk of bias (traffic-light)", fontsize=14, pad=12)

    # legend
    legend_items = ["Low","Some concerns","Unclear","High","NA"]
    x0 = 0.02
    y0 = 1.02
    for idx, lab in enumerate(legend_items):
        ax.add_patch(Rectangle((x0 + idx*0.13, y0), 0.03, 0.03,
                               transform=ax.transAxes, facecolor=COLOR[lab],
                               edgecolor="#333333", linewidth=0.6, clip_on=False))
        ax.text(x0 + idx*0.13 + 0.035, y0+0.015, lab,
                transform=ax.transAxes, va="center", fontsize=9, clip_on=False)

    plt.tight_layout()
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)

def domain_summary(df, domains, outpath, dpi=300):
    # proportions per domain
    cats = ["Low","Some concerns","Unclear","High"]
    props = {c: [] for c in cats}
    for d in domains:
        col = df[d].apply(normalize)
        total = (col != "NA").sum()
        for c in cats:
            props[c].append((col == c).sum() / total if total else 0.0)

    x = range(len(domains))
    fig, ax = plt.subplots(figsize=(max(8, 0.5*len(domains)+4), 5))
    bottom = [0]*len(domains)
    for c in cats:
        ax.bar(x, props[c], bottom=bottom, label=c)
        bottom = [bottom[i] + props[c][i] for i in range(len(domains))]

    ax.set_xticks(list(x))
    ax.set_xticklabels(domains, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0,1)
    ax.set_ylabel("Proportion")
    ax.set_title("Risk of bias summary by domain", fontsize=14, pad=12)
    ax.legend(frameon=False)
    plt.tight_layout()
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="rob_template.csv (filled)")
    ap.add_argument("--outdir", default="FIGURES_RoB")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)

    # domains = all columns except these
    ignore = {"Study","Design","Overall"}
    domains = [c for c in df.columns if c not in ignore]

    traffic_light(df, domains, outdir/"RoB_traffic_light.png", dpi=args.dpi)
    domain_summary(df, domains, outdir/"RoB_domain_summary.png", dpi=args.dpi)

    print(f"Saved: {outdir/'RoB_traffic_light.png'}")
    print(f"Saved: {outdir/'RoB_domain_summary.png'}")

if __name__ == "__main__":
    main()
