#!/usr/bin/env python
"""
Figure 6 – Pairwise F_ST among Jazan sub-regions
(re-aligned 202-sequence CelTOS dataset)

Requires in the SAME FOLDER as this script:

    - Supplementary_File_SF_5b AMOVA_Full_Results.xml
        (Arlequin full AMOVA output; contains the text block
         "Population pairwise FSTs" and the PairFstMat matrix)

Outputs:

    Figure6_pairwise_Fst_regions.png  (600 dpi)
"""

from pathlib import Path
import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ----------------------------------------------------------------------
# PERMANOVA statistics from Figure 5
# ----------------------------------------------------------------------
PERMANOVA_PSEUDO_F = 1.75
PERMANOVA_R2 = 0.017
PERMANOVA_P = 0.078

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def parse_pairwise_fst(xml_path: Path) -> np.ndarray:
    """
    Extract the Population pairwise FST matrix from an Arlequin XML file.
    """
    with xml_path.open(encoding="utf-8", errors="ignore") as f:
        txt = f.read()

    marker = "Population pairwise FSTs"
    start = txt.find(marker)
    if start == -1:
        raise ValueError(
            "Could not find 'Population pairwise FSTs' section in "
            f"{xml_path.name}. Check that you exported the full AMOVA output."
        )

    sub = txt[start:]
    end_tag = "</PairFstMat>"
    end_idx = sub.find(end_tag)
    if end_idx != -1:
        sub = sub[:end_idx]

    lines = sub.splitlines()
    rows = []

    for line in lines:
        if "." not in line and "-" not in line:
            continue
        m = re.match(r"\s*(\d+)\s+([-\d.Ee\s]+)$", line)
        if m:
            nums = [float(x) for x in m.group(2).split()]
            rows.append(nums)

    if not rows:
        raise ValueError(
            "No numeric F_ST rows were detected in the 'Population pairwise FSTs' block."
        )

    n = len(rows)
    fst = np.zeros((n, n), dtype=float)

    # rows[i] is the lower-triangular row including the diagonal
    for i, row in enumerate(rows):
        fst[i, : len(row)] = row
        fst[: len(row), i] = row  # mirror to upper triangle

    return fst


def plot_fst_heatmap(fst: np.ndarray, out_path: Path) -> None:
    """
    Make the premium F_ST heatmap.
    """
    n = fst.shape[0]
    labels = [f"Region {i + 1}" for i in range(n)]

    # ------------------------------------------------------------------
    # Colour scale settings
    # ------------------------------------------------------------------
    # Fixed limits from -0.010 to 0.010
    vmin, vmax = -0.010, 0.010

    # Custom dark-blue → sky-blue gradient
    dark_blue = "#c6c7ef"
    light_blue = "#305894"
    cmap = LinearSegmentedColormap.from_list(
        "fst_blue_gradient", [dark_blue, light_blue]
    )
    # ------------------------------------------------------------------

    plt.rcParams.update({
        "font.size": 10,
        "axes.titleweight": "bold",
    })

    fig, ax = plt.subplots(figsize=(4.6, 4.3))

    im = ax.imshow(
        fst,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin="upper",
        zorder=1,
    )

    # Numeric labels on ALL cells (including diagonal) with 5 decimals
    for i in range(n):
        for j in range(n):
            val = fst[i, j]
            # for very dark cells, need to use white text; otherwise black
            text_color = "white" if (val - vmin) / (vmax - vmin) > 0.65 else "black"
            ax.text(
                j,
                i,
                f"{val:.5f}",
                ha="center",
                va="center",
                fontsize=8,
                color=text_color,
                zorder=4,
            )

    # Ticks and labels
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels)

    ax.tick_params(axis="x", bottom=True, top=False,
                   labelbottom=True, labeltop=False)
    ax.tick_params(axis="y", left=True, right=False)

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)  #origin at top-left

    # Colourbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("F$_{ST}$", rotation=90)

    # Styling
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # Main title
    fig.suptitle(
        "Pairwise F$_{ST}$ among Jazan sub-regions",
        y=0.98,
        fontsize=13,
        fontweight="bold",
    )

    # PERMANOVA annotation from Figure 5, just under the title
    fig.text(
        0.5,
        0.83,
        f"PERMANOVA (Region): pseudo-F = {PERMANOVA_PSEUDO_F:.2f}, "
        f"R\u00b2 = {PERMANOVA_R2:.3f}, p = {PERMANOVA_P:.3f}",
        ha="center",
        va="center",
        fontsize=9,
        color="dimgray",
    )

    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out_path, dpi=600)
    plt.close(fig)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    here = Path(__file__).resolve().parent
    xml_name = "Supplementary_File_SF_5b AMOVA_Full_Results.xml"
    xml_path = here / xml_name

    if not xml_path.exists():
        raise FileNotFoundError(
            f"Could not find '{xml_name}' in {here}.\n"
            "Place this script in the same folder as the AMOVA XML file."
        )

    fst = parse_pairwise_fst(xml_path)
    print("Pairwise F_ST matrix:")
    print(fst)

    out_path = here / "Figure6_pairwise_Fst_regions.png"
    plot_fst_heatmap(fst, out_path)
    print(f"Saved figure to: {out_path}")


if __name__ == "__main__":
    main()
