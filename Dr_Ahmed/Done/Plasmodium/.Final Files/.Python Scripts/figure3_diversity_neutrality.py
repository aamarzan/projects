#!/usr/bin/env python
"""
Make Figure 3: genetic diversity + neutrality indices by Jazan sub-region.

Reads:
    Supplementary_File_SF_3a DnaSP_Diversity_Neutrality_by_Group.xlsx
    (sheet 'Diversity_Neutrality')

Uses rows with:
    GroupFactor == 'Region'
and columns:
    Hd, pi, Tajima_D, FuLi_D_star, FuLi_F_star

Outputs:
    Figure3_diversity_neutrality_regions.png  (600 dpi)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm


def main():
    # ------------------------------------------------------------------
    # 1. Load region-wise diversity + neutrality statistics
    # ------------------------------------------------------------------
    here = Path(__file__).resolve().parent

    excel_name = "Supplementary_File_SF_3a DnaSP_Diversity_Neutrality_by_Group.xlsx"
    xlsx_path = here / excel_name
    if not xlsx_path.exists():
        raise FileNotFoundError(
            f"Could not find '{excel_name}' in {here}.\n"
            "Copy your DnaSP summary file into this folder or "
            "edit 'excel_name' at the top of this script."
        )

    df_all = pd.read_excel(xlsx_path, sheet_name="Diversity_Neutrality")

    # Keep only the three Jazan regions
    region_df = df_all[df_all["GroupFactor"] == "Region"].copy()

    # Sort as region1, region2, region3 (in case the order changed)
    region_df["RegionIdx"] = region_df["Group"].str.extract(r"(\d+)").astype(int)
    region_df = region_df.sort_values("RegionIdx")

    regions = region_df["Group"].tolist()
    Hd = region_df["Hd"].values
    pi = region_df["pi"].values
    TajD = region_df["Tajima_D"].values
    FuLiD = region_df["FuLi_D_star"].values
    FuLiF = region_df["FuLi_F_star"].values

    # Nicer labels
    label_map = {r: r.replace("region", "Region ") for r in regions}
    labels = [label_map[r] for r in regions]

    print("Using regions:", labels)
    print(region_df[["Group", "Hd", "pi", "Tajima_D", "FuLi_D_star", "FuLi_F_star"]])

    # ------------------------------------------------------------------
    # 2. Figure layout
    # ------------------------------------------------------------------
    plt.rcParams.update({
        "font.size": 10,
        "axes.titleweight": "bold",
    })

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(7.2, 5.0),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.2]},
    )

    # Slightly stretch x positions to create more gap between regions
    n_regions = len(labels)
    group_spacing = 1.3  # >1 = more gap between groups
    x = np.arange(n_regions) * group_spacing

    # ------------------------------------------------------------------
    # Panel A – Hd (bars) + π (line with separate axis)
    # ------------------------------------------------------------------
    hd_color = "#357F9C"  # single premium blue for all Hd bars

    bar_width = 0.55  # a bit narrower to keep gaps left/right
    bars = ax1.bar(
        x, Hd,
        width=bar_width,
        color=hd_color,
        edgecolor="#164365",
        linewidth=0.8,
        label="Hd",
        zorder=3,
    )

    # Hd labels INSIDE bars near the top
    for xi, yi in zip(x, Hd):
        ax1.text(
            xi, yi - 0.015,
            f"{yi:.2f}",
            ha="center", va="top",
            fontsize=8,
            color="white",
            fontweight="bold",
        )

    ax1.set_ylabel("Haplotype diversity (Hd)")
    ax1.set_ylim(min(Hd) - 0.03, 1.0)

    # Second y-axis for π
    ax1b = ax1.twinx()
    ax1b.plot(
        x, pi,
        marker="o",
        markersize=5,
        linewidth=2,
        color="#2F8F5B",
        label="π",
        zorder=4,
    )

    # Decide y-limits for π then annotate to avoid overlap
    if pi.max() > pi.min():
        ypad = (pi.max() - pi.min()) * 0.6
    else:
        ypad = 0.001
    ax1b.set_ylim(pi.min() - ypad, pi.max() + ypad)
    ax1b.set_ylabel("Nucleotide diversity (π)")

    # π labels slightly to the right of the marker (no overlap with Hd labels)
    for xi, yi in zip(x, pi):
        ax1b.text(
            xi + 0.09, yi,
            f"{yi:.3f}",
            ha="left", va="center",
            fontsize=8,
            color="#14523A",
        )

    # Combined legend for Panel A
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1b.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, frameon=False, loc="upper right")

    # Panel label "A" – lifted higher
    ax1.text(
        -0.06, 1.18, "A",
        transform=ax1.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
    )

    # ------------------------------------------------------------------
    # Panel B – neutrality indices (premium palette + good spacing)
    # ------------------------------------------------------------------
    # Bar width and offsets chosen so bars within a group have small gaps
    width = 0.20          # width of each bar
    delta = 0.25          # distance of side bars from center
    offsets = [-delta, 0.0, +delta]

    # Slightly thicker neutral baseline at 0
    ax2.axhline(0, color="0.6", linewidth=1.2, zorder=1)

    # Premium colours requested
    taj_color = "#015798"   # light magenta
    fulid_color = "#5C6BC0" # light blue
    fulif_color = "#569DAA" # light saffron

    ax2.bar(
        x + offsets[0], TajD,
        width=width,
        label="Tajima's D",
        color=taj_color,
        zorder=2,
    )
    ax2.bar(
        x + offsets[1], FuLiD,
        width=width,
        label="Fu & Li's D*",
        color=fulid_color,
        zorder=2,
    )
    ax2.bar(
        x + offsets[2], FuLiF,
        width=width,
        label="Fu & Li's F*",
        color=fulif_color,
        zorder=2,
    )

    ax2.set_ylabel("Neutrality index value")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)

    # Y-limits:
    # - If all values are between 0 and 3 → lock to [0, 3] as requested.
    # - If all ≥0 but some >3 → [0, max(3, 1.1 * max)].
    # - If some <0 → symmetric range including negatives.
    y_min_data = min(TajD.min(), FuLiD.min(), FuLiF.min())
    y_max_data = max(TajD.max(), FuLiD.max(), FuLiF.max())

    if y_min_data >= 0 and y_max_data <= 3:
        y_min, y_max = 0.0, 3.0
    elif y_min_data >= 0:
        y_min, y_max = 0.0, max(3.0, y_max_data * 1.1)
    else:
        abs_max = max(abs(y_min_data), y_max_data)
        y_min, y_max = -abs_max * 1.1, abs_max * 1.1

    ax2.set_ylim(y_min, y_max)

    # Value labels on each neutrality bar (premium style)
    y_range = y_max - y_min
    label_offset = 0.04 * y_range
    top_margin = 0.03 * y_range

    for i, xi in enumerate(x):
        for off, val, col in [
            (offsets[0], TajD[i], taj_color),
            (offsets[1], FuLiD[i], fulid_color),
            (offsets[2], FuLiF[i], fulif_color),
        ]:
            xpos = xi + off
            if val >= 0:
                va = "bottom"
                ypos = min(val + label_offset, y_max - top_margin)
            else:
                va = "top"
                ypos = max(val - label_offset, y_min + top_margin)
            ax2.text(
                xpos, ypos,
                f"{val:.2f}",
                ha="center", va=va,
                fontsize=8,
                color=col,
                fontweight="semibold",
            )

    # Legend above panel B with good spacing
    ax2.legend(
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.20),
    )

    # Panel label "B"
    ax2.text(
        -0.06, 1.12, "B",
        transform=ax2.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
    )

    ax2.set_xlabel("Jazan sub-region")

    # ------------------------------------------------------------------
    # Styling
    # ------------------------------------------------------------------
    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.45)

    fig.suptitle(
        "Genetic diversity and neutrality indices of P. falciparum CelTOS by Jazan sub-region",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_path = here / "Figure3_diversity_neutrality_regions.png"
    fig.savefig(out_path, dpi=600)
    plt.close(fig)

    print(f"Saved figure to: {out_path}")


if __name__ == "__main__":
    main()
