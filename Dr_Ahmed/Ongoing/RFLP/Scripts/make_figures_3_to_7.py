#!/usr/bin/env python3
"""
make_figures_3_to_7.py
Build multi-panel manuscript figures (Figures 3–7) from already-burnished gel PNGs.

- DOES NOT alter the gel content (no warping, no band moving, no “AI redraw”).
- Only places your images into a clean montage + adds uniform panel letters + locus labels.
- Exports PNG (high DPI) + PDF + optional EPS.

Run:
  python make_figures_3_to_7.py --root "E:\\...\\Thesis Figures" --config "fig3_7_config.json" --out "out_figures" --dpi 600 --eps
"""

from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt

Image.MAX_IMAGE_PIXELS = None  # allow very large images safely


def apply_style(font_family: str = "DejaVu Sans", base_font: int = 10) -> None:
    mpl.rcParams.update({
        "font.family": font_family,
        "font.size": base_font,
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


def auto_grid(n: int, preferred_cols: int = 2) -> tuple[int, int]:
    if n <= 1:
        return 1, 1
    cols = min(preferred_cols, n)
    rows = math.ceil(n / cols)
    return rows, cols


def build_figure(
    panels: list[dict],
    title: str,
    out_stem: Path,
    dpi: int = 600,
    preferred_cols: int = 2,
    save_eps: bool = False,
) -> None:
    n = len(panels)
    if n == 0:
        raise ValueError(f"No panels provided for {out_stem.name}")

    rows, cols = auto_grid(n, preferred_cols=preferred_cols)

    # Figure sizing: keep ~7.2" width for 2-col layouts (journal-friendly)
    panel_w_in = 3.6
    panel_h_in = 2.8
    fig_w = panel_w_in * cols
    fig_h = panel_h_in * rows + 0.7  # extra for suptitle

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(
        rows, cols,
        left=0.02, right=0.98, bottom=0.02, top=0.90,
        wspace=0.04, hspace=0.06
    )

    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    for idx, p in enumerate(panels):
        r = idx // cols
        c = idx % cols
        ax = fig.add_subplot(gs[r, c])

        img_path = Path(p["file"])
        if not img_path.is_absolute():
            # allow relative paths from root
            img_path = out_stem.parent.parent / img_path  # will be corrected in main()
        # NOTE: main() passes absolute paths; this is just a safeguard.

        im = p["_im"]  # PIL image already loaded
        ax.imshow(im, interpolation="bilinear")
        ax.set_axis_off()

        panel_letter = letters[idx]
        locus = p.get("label", "").strip()

        # Panel letter (uniform)
        ax.text(
            0.01, 0.99, panel_letter,
            transform=ax.transAxes, ha="left", va="top",
            fontsize=12, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.25", fc="black", ec="none", alpha=0.55),
        )

        # Locus label (uniform + professional)
        if locus:
            ax.text(
                0.12, 0.99, locus,
                transform=ax.transAxes, ha="left", va="top",
                fontsize=10, fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.25", fc="black", ec="none", alpha=0.40),
            )

    fig.suptitle(title, x=0.01, ha="left", y=0.985, fontsize=12, fontweight="bold")

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".png"), dpi=dpi)
    fig.savefig(out_stem.with_suffix(".pdf"))
    if save_eps:
        fig.savefig(out_stem.with_suffix(".eps"))
    plt.close(fig)


def main(root: Path, config_path: Path, out_dir: Path, dpi: int, eps: bool) -> None:
    apply_style()

    cfg = json.loads(config_path.read_text(encoding="utf-8"))

    out_dir.mkdir(parents=True, exist_ok=True)

    # Sort figures by numeric order if keys like "Figure_3", "Figure_4"...
    def fig_key(k: str) -> int:
        m = [s for s in k.replace("-", "_").split("_") if s.isdigit()]
        return int(m[0]) if m else 9999

    for fig_name in sorted(cfg.keys(), key=fig_key):
        item = cfg[fig_name]
        title = item["title"]
        preferred_cols = int(item.get("cols", 2))
        panels = item["panels"]

        # load images (absolute paths)
        loaded = []
        for p in panels:
            rel = Path(p["file"])
            img_path = rel if rel.is_absolute() else (root / rel)
            if not img_path.exists():
                raise FileNotFoundError(f"Missing: {img_path}")
            pp = dict(p)
            pp["_im"] = open_rgb(img_path)
            pp["file"] = str(img_path)
            loaded.append(pp)

        out_stem = out_dir / item.get("out_name", fig_name)
        build_figure(
            panels=loaded,
            title=title,
            out_stem=out_stem,
            dpi=dpi,
            preferred_cols=preferred_cols,
            save_eps=eps,
        )

    print(f"Done. Saved to: {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Folder containing your burnished gel PNGs")
    ap.add_argument("--config", required=True, help="JSON config defining Figures 3–7 panels")
    ap.add_argument("--out", default="out_figures", help="Output directory")
    ap.add_argument("--dpi", type=int, default=600, help="PNG export DPI (e.g., 600 or 900)")
    ap.add_argument("--eps", action="store_true", help="Also export EPS (can be large)")
    args = ap.parse_args()

    main(Path(args.root), Path(args.config), Path(args.out), args.dpi, args.eps)
