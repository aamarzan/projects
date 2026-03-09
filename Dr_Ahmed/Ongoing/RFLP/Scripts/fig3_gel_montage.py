#!/usr/bin/env python3
import argparse
from pathlib import Path
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt

def apply_style():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06,
    })

def open_rgb(p: Path) -> Image.Image:
    im = Image.open(p)
    if im.mode != "RGB":
        im = im.convert("RGB")
    return im

def main(img_a: Path, img_b: Path, img_c: Path, out_dir: Path,
         label_a: str, label_b: str, label_c: str):
    apply_style()
    A, B, C = open_rgb(img_a), open_rgb(img_b), open_rgb(img_c)

    fig = plt.figure(figsize=(7.2, 6.2))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1], hspace=0.08)

    for i, (im, lab, panel) in enumerate([(A, label_a, "A"), (B, label_b, "B"), (C, label_c, "C")]):
        ax = fig.add_subplot(gs[i, 0])
        ax.imshow(im)
        ax.set_axis_off()
        ax.text(0.01, 0.98, f"{panel}", transform=ax.transAxes, ha="left", va="top",
                fontsize=12, weight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.25", fc="black", ec="none", alpha=0.55))
        ax.text(0.12, 0.98, lab, transform=ax.transAxes, ha="left", va="top",
                fontsize=10, weight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.25", fc="black", ec="none", alpha=0.40))

    fig.suptitle("Figure 3. Representative PCR–RFLP gels for selected loci", x=0.01, ha="left", fontsize=12, weight="bold")

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / "Figure_3_Representative_Gels"
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".eps"))
    plt.close(fig)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True, help="Panel A image path")
    p.add_argument("--b", required=True, help="Panel B image path")
    p.add_argument("--c", required=True, help="Panel C image path")
    p.add_argument("--la", default="Deletion locus (e.g., HV69–70del)")
    p.add_argument("--lb", default="SNP locus (e.g., K417N or L452R)")
    p.add_argument("--lc", default="Non-spike deletion (e.g., SGF3675–3677del)")
    p.add_argument("--out", default="out_figures")
    args = p.parse_args()
    main(Path(args.a), Path(args.b), Path(args.c), Path(args.out), args.la, args.lb, args.lc)
