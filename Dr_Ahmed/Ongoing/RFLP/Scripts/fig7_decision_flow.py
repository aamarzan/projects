#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def apply_style():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06,
    })

def add_box(ax, x, y, w, h, text, fc="#F8FAFC", ec="#111827", lw=0.9):
    p = FancyBboxPatch((x,y), w,h, boxstyle="round,pad=0.02,rounding_size=0.06",
                       facecolor=fc, edgecolor=ec, linewidth=lw)
    ax.add_patch(p)
    ax.text(x+w/2, y+h/2, text, ha="center", va="center", color="#111827")
    return p

def arrow(ax, x1,y1,x2,y2, color="#111827"):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),
                                 arrowstyle="-|>", mutation_scale=12,
                                 linewidth=1.0, color=color, shrinkA=4, shrinkB=4))

def main(rules_csv: Path, out_dir: Path):
    apply_style()
    if rules_csv.exists():
        rules = pd.read_csv(rules_csv)
        pairs = list(zip(rules["if"].astype(str), rules["then"].astype(str)))
    else:
        pairs = [
            ("If HV69–70del + NTD pattern\n(T19I/T95I/D138Y etc.)", "Suggest Omicron-lineage screen"),
            ("If K417N + E484K/Q (± N440K/L452R)", "Differentiate Beta/Omicron sub-patterns"),
            ("If ORF1ab SGF3675–3677del\nand/or N-gene deletions", "Support non-spike corroboration"),
            ("If discordant gel patterns", "Recommend Sanger confirmation (representative loci)"),
        ]

    fig = plt.figure(figsize=(7.2, 4.6))
    ax = fig.add_axes([0,0,1,1])
    ax.set_axis_off()

    ax.text(0.03, 0.95, "Figure 7. Interpretation framework for PCR–RFLP patterns (template)", fontsize=11, weight="bold")

    # Build a vertical flow
    x = 0.08
    w = 0.84
    h = 0.12
    y0 = 0.78
    gap = 0.06

    boxes = []
    for i, (cond, action) in enumerate(pairs):
        b = add_box(ax, x, y0 - i*(h+gap), w, h, f"{cond}\n\u2192 {action}",
                    fc="#F1F5F9" if i%2==0 else "#ECFDF5",
                    ec="#111827" if i%2==0 else "#0F766E")
        boxes.append(b)

    for i in range(len(boxes)-1):
        b1, b2 = boxes[i], boxes[i+1]
        arrow(ax,
              b1.get_x()+b1.get_width()/2, b1.get_y(),
              b2.get_x()+b2.get_width()/2, b2.get_y()+b2.get_height(),
              color="#0F766E")

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / "Figure_7_Decision_Flow"
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".eps"))
    plt.close(fig)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rules_csv", default="decision_rules.csv")
    p.add_argument("--out", default="out_figures")
    args = p.parse_args()
    main(Path(args.rules_csv), Path(args.out))
