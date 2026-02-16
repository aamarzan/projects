#!/usr/bin/env python
"""
Figure 4 – Parasitemia across P. falciparum CelTOS haplotypes
(re-aligned 202-sequence dataset)

Requires:
    - Supplementary_File_SF_6a Haplotype Statistics.xlsx
      (sheet: 'counts by group', with columns:
       Haplotype, PD1, PD2, PD3, PD4, PD_Unknown)

Outputs:
    Figure4_parasitemia_by_haplotype.png  (600 dpi)

Note:
    Uses SciPy for the chi-square test:
        pip install scipy
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def main():
    here = Path(__file__).resolve().parent
    xlsx_name = "Supplementary_File_SF_6a Haplotype Statistics.xlsx"
    xlsx_path = here / xlsx_name

    if not xlsx_path.exists():
        raise FileNotFoundError(
            f"Could not find '{xlsx_name}' in {here}.\n"
            "Make sure this script and the Excel file are in the same folder."
        )

    # ------------------------------------------------------------------
    # 1. Load haplotype-level parasitemia counts
    # ------------------------------------------------------------------
    df = pd.read_excel(xlsx_path, sheet_name="counts by group")

    required_cols = {"Haplotype", "PD1", "PD2", "PD3", "PD4", "PD_Unknown"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            "The sheet 'counts by group' must contain columns:\n"
            f"{sorted(required_cols)}"
        )

    # Total isolates per haplotype and number with known PD
    df["N"] = df[["PD1", "PD2", "PD3", "PD4", "PD_Unknown"]].sum(axis=1)
    df["n_known"] = df[["PD1", "PD2", "PD3", "PD4"]].sum(axis=1)

    # Mean parasitemia level (1–4) per haplotype, ignoring PD_Unknown
    levels = np.array([1.0, 2.0, 3.0, 4.0])
    known_counts = df[["PD1", "PD2", "PD3", "PD4"]].to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        df["mean_PD"] = (known_counts @ levels) / df["n_known"].replace(0, np.nan)

    # New 3-class definition: low = PD1, moderate = PD2, high = PD3 + PD4
    df["PD_low"] = df["PD1"]
    df["PD_mod"] = df["PD2"]
    df["PD_high"] = df["PD3"] + df["PD4"]

    # Sort haplotypes by mean parasitemia (highest → lowest)
    df = df.sort_values("mean_PD", ascending=False).reset_index(drop=True)

    # Clean labels for plotting (Hap_1 → H1, etc.)
    df["H_label"] = df["Haplotype"].str.replace("Hap_", "H", regex=False)

    # ------------------------------------------------------------------
    # 2. Chi-square test of independence (3 classes: low/mod/high)
    # ------------------------------------------------------------------
    contingency = df[["PD_low", "PD_mod", "PD_high"]].to_numpy(dtype=float)
    chi2, p_val, dof, expected = stats.chi2_contingency(contingency)

    print("Chi-square test (haplotype × parasitemia class, 3 levels):")
    print(f"  chi² = {chi2:.2f}, df = {dof}, p = {p_val:.4g}")

    # ------------------------------------------------------------------
    # 3. Build the premium plot
    # ------------------------------------------------------------------
    plt.rcParams.update({
        "font.size": 10,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
    })

    fig, ax = plt.subplots(figsize=(9.5, 3.8))

    x = np.arange(len(df))
    mean_PD = df["mean_PD"].to_numpy()
    n_known = df["n_known"].to_numpy()

    # Background bands for low / moderate / high parasitemia
    low_band = "#e9f5fb"      # pale blue
    mid_band = "#fff4d6"      # pale cream
    high_band = "#ffe3e3"     # pale pink

    # low: 1–<2, moderate: 2–<3, high: 3–4
    ax.axhspan(1.0, 2.0, facecolor=low_band, zorder=0)
    ax.axhspan(2.0, 3.0, facecolor=mid_band, zorder=0)
    ax.axhspan(3.0, 4.05, facecolor=high_band, zorder=0)

    # Guideline lines
    for y in (2.0, 3.0, 4.0):
        ax.axhline(y, color="0.75", linestyle="--", linewidth=0.7, zorder=1)

    # Premium lollipop: line + bubble markers (size ∝ n_known)
    line_color = "#b22222"  # rich crimson
    base_marker_size = 24.0
    extra_marker_size = 96.0  # additional size for largest n_known

    marker_sizes = base_marker_size + extra_marker_size * (
        n_known / n_known.max()
    )

    ax.plot(
        x,
        mean_PD,
        color=line_color,
        linewidth=2.0,
        zorder=3,
    )
    ax.scatter(
        x,
        mean_PD,
        s=marker_sizes,
        color=line_color,
        edgecolor="white",
        linewidth=0.7,
        zorder=4,
    )

    # Axis labels / ticks
    ax.set_ylabel("Mean parasitemia level (1–4)")
    ax.set_xlabel("Haplotype")
    ax.set_xticks(x)
    ax.set_xticklabels(df["H_label"], rotation=60, ha="right")

    ax.set_ylim(0.9, 4.1)
    ax.set_yticks([1, 2, 3, 4])

    # Region labels on the right
    x_text = len(df) + 0.3
    ax.text(x_text, 3.8, "High", color="#b22222",
            ha="left", va="center", fontsize=9)
    ax.text(x_text, 3.0, "Moderate", color="#a87400",
            ha="left", va="center", fontsize=9)
    ax.text(x_text, 2.0, "Low", color="#1f4b8c",
            ha="left", va="center", fontsize=9)

    # Chi-square annotation in plot coordinates (top-left)
    ax.text(
        0.01, 0.96,
        rf"$\chi^2$ = {chi2:.1f}, df = {dof}, p = {p_val:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="dimgray",
    )

    # Note about marker sizes – now TOP RIGHT
    ax.text(
        0.99, 0.98,
        "Marker size ∝ isolates per haplotype",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="dimgray",
    )

    # Clean styling
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)

    # Main title – lowered a bit so it doesn't get cropped
    fig.suptitle(
        "Parasitemia across P. falciparum CelTOS haplotypes",
        y=0.96,
        fontsize=13,
        fontweight="bold",
    )
    # Leave extra room at top for the title
    fig.tight_layout(rect=(0, 0, 1, 0.90))

    out_path = here / "Figure4_parasitemia_by_haplotype.png"
    fig.savefig(out_path, dpi=600)
    plt.close(fig)

    print(f"Saved figure to: {out_path}")


if __name__ == "__main__":
    main()
