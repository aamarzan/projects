#!/usr/bin/env python
"""
Figure 5 – Premium PCoA of P. falciparum CelTOS genetic distances
by Jazan sub-region (re-aligned 202-sequence dataset).

Requires in the SAME FOLDER as this script:

  1) CelTOS_pairwise_TamuraNei.xlsx
       - sheet: "MatrixOutput"
       - first column: sequence names (row labels)
       - lower-triangular Tamura–Nei distances (MEGA export)

  2) Supplementary_File_SF_2b Traits.xlsx
       - sheet: "Traits"
       - columns: "Sequences ID", "Region" (region1/region2/region3)

Outputs:

  Figure5_pcoa_regions.png  (600 dpi)
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def clean_id(s: str) -> str:
    """Standardise sequence IDs."""
    return str(s).strip().rstrip(".")


def load_distance_matrix(path: Path) -> Tuple[np.ndarray, list]:
    """Load the pairwise Tamura–Nei distance matrix exported from MEGA."""
    xls = pd.ExcelFile(path)
    if "MatrixOutput" not in xls.sheet_names:
        raise ValueError(
            f"'MatrixOutput' sheet not found in {path.name}. "
            f"Available sheets: {xls.sheet_names}"
        )

    df = pd.read_excel(xls, "MatrixOutput")
    label_col = df.columns[0]
    mat_df = df.set_index(label_col)

    # Clean labels
    mat_df.index = [clean_id(i) for i in mat_df.index]
    mat_df.columns = [clean_id(c) for c in mat_df.columns]

    arr = mat_df.to_numpy(dtype=float)
    n = arr.shape[0]
    D = arr.copy()

    # Fill upper triangle from lower triangle
    iu = np.triu_indices(n, k=1)
    D[iu] = D.T[iu]

    np.fill_diagonal(D, 0.0)
    return D, mat_df.index.tolist()


def load_traits(path: Path) -> pd.DataFrame:
    """Load the Traits sheet and create a cleaned SampleID column."""
    df = pd.read_excel(path, sheet_name="Traits")
    if "Sequences ID" not in df.columns or "Region" not in df.columns:
        raise ValueError(
            "The 'Traits' sheet must contain 'Sequences ID' and 'Region' columns."
        )
    df["SampleID"] = df["Sequences ID"].apply(clean_id)
    return df


def pcoa(distance_matrix: np.ndarray):
    """Classical multidimensional scaling (PCoA) on a distance matrix."""
    D = np.asarray(distance_matrix, float)
    n = D.shape[0]

    D2 = D ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J.dot(D2).dot(J)

    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    pos = eigvals > 1e-10
    eigvals = eigvals[pos]
    eigvecs = eigvecs[:, pos]

    coords = eigvecs * np.sqrt(eigvals)
    return coords, eigvals


def permanova_from_coords(coords: np.ndarray, groups, n_perm: int = 999, seed: int = 123):
    """Distance-based PERMANOVA using PCoA coordinates."""
    Y = np.asarray(coords, float)
    groups = np.asarray(groups)
    rng = np.random.default_rng(seed)
    n = Y.shape[0]

    def anova_F(labels):
        labels = np.asarray(labels)
        levels = np.unique(labels)
        overall = Y.mean(axis=0)
        sst = ((Y - overall) ** 2).sum()
        ssb = 0.0
        for lvl in levels:
            idx = np.where(labels == lvl)[0]
            if len(idx) == 0:
                continue
            centroid = Y[idx].mean(axis=0)
            ssb += len(idx) * ((centroid - overall) ** 2).sum()
        ssw = sst - ssb
        dfb = len(levels) - 1
        dfw = n - len(levels)
        F = (ssb / dfb) / (ssw / dfw)
        R2 = ssb / sst if sst > 0 else np.nan
        return F, R2

    F_obs, R2_obs = anova_F(groups)

    perm_F = np.empty(n_perm)
    for i in range(n_perm):
        perm_labels = rng.permutation(groups)
        perm_F[i], _ = anova_F(perm_labels)
    p_val = (np.sum(perm_F >= F_obs) + 1) / (n_perm + 1)

    return F_obs, R2_obs, p_val


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    here = Path(__file__).resolve().parent

    dist_xlsx = here / "CelTOS_pairwise_TamuraNei.xlsx"
    traits_xlsx = here / "Supplementary_File_SF_2b Traits.xlsx"

    if not dist_xlsx.exists():
        raise FileNotFoundError(f"Could not find '{dist_xlsx.name}' in {here}.")
    if not traits_xlsx.exists():
        raise FileNotFoundError(f"Could not find '{traits_xlsx.name}' in {here}.")

    # 1. Load distance matrix + traits and align samples
    D, dist_ids = load_distance_matrix(dist_xlsx)
    traits = load_traits(traits_xlsx)
    traits_aligned = traits.set_index("SampleID").loc[dist_ids]

    # 2. Run PCoA
    coords, eigvals = pcoa(D)
    pc1, pc2 = coords[:, 0], coords[:, 1]
    pc1_var = eigvals[0] / eigvals.sum() * 100.0
    pc2_var = eigvals[1] / eigvals.sum() * 100.0

    # 3. PERMANOVA using all PCoA axes
    regions_raw = traits_aligned["Region"].values
    region_label_map = {
        "region1": "Region 1",
        "region2": "Region 2",
        "region3": "Region 3",
    }
    regions = np.array([region_label_map.get(r, str(r)) for r in regions_raw])

    F_obs, R2_obs, p_val = permanova_from_coords(coords, regions, n_perm=999)

    print("PCoA summary:")
    print(f"  PC1 variance: {pc1_var:.2f}%")
    print(f"  PC2 variance: {pc2_var:.2f}%")
    print("PERMANOVA (Region):")
    print(f"  pseudo-F = {F_obs:.3f}, R² = {R2_obs:.4f}, p = {p_val:.4f}")

    # ------------------------------------------------------------------
    # 4. Build the premium PCoA figure
    # ------------------------------------------------------------------
    plt.rcParams.update({
        "font.size": 10,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
    })

    fig, ax = plt.subplots(figsize=(5.2, 5.0))

    # Precompute axis limits for quadrant shading
    x_pad = (pc1.max() - pc1.min()) * 0.15 or 0.001
    y_pad = (pc2.max() - pc2.min()) * 0.15 or 0.001
    x_min, x_max = pc1.min() - x_pad, pc1.max() + x_pad
    y_min, y_max = pc2.min() - y_pad, pc2.max() + y_pad

    # Very light gradient-style shading for the four quadrants
    q1_color = "#f5f7ff"  # top-right – cool blue
    q2_color = "#f5fff7"  # top-left – soft green
    q3_color = "#fff8f5"  # bottom-left – warm peach
    q4_color = "#fdf5ff"  # bottom-right – lilac

    # top-left
    ax.add_patch(
        Rectangle(
            (x_min, 0),
            0 - x_min,
            y_max - 0,
            facecolor=q2_color,
            edgecolor="none",
            alpha=0.75,
            zorder=0,
        )
    )
    # top-right
    ax.add_patch(
        Rectangle(
            (0, 0),
            x_max - 0,
            y_max - 0,
            facecolor=q1_color,
            edgecolor="none",
            alpha=0.75,
            zorder=0,
        )
    )
    # bottom-left
    ax.add_patch(
        Rectangle(
            (x_min, y_min),
            0 - x_min,
            0 - y_min,
            facecolor=q3_color,
            edgecolor="none",
            alpha=0.75,
            zorder=0,
        )
    )
    # bottom-right
    ax.add_patch(
        Rectangle(
            (0, y_min),
            x_max - 0,
            0 - y_min,
            facecolor=q4_color,
            edgecolor="none",
            alpha=0.75,
            zorder=0,
        )
    )

    # Colour palette – soft, colour-blind friendly
    color_cycle = ["#3b528b", "#21918c", "#5ec962"]
    unique_regions = list(dict.fromkeys(regions))  # preserve order
    region_colors = {
        reg: color_cycle[i % len(color_cycle)]
        for i, reg in enumerate(unique_regions)
    }

    # Semi-transparent points with thin black border
    for reg in unique_regions:
        mask = regions == reg
        ax.scatter(
            pc1[mask],
            pc2[mask],
            s=35,
            color=region_colors[reg],
            edgecolors="black",
            linewidth=0.15,
            alpha=0.70,
            label=reg,
            zorder=3,
        )

    # Zero axes for reference
    ax.axhline(0, color="0.6", linewidth=0.7, zorder=2)
    ax.axvline(0, color="0.6", linewidth=0.7, zorder=2)

    # Axes labels with variance explained
    ax.set_xlabel(f"PC1 ({pc1_var:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pc2_var:.1f}% variance)")

    # Final limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Fewer, non-overlapping numeric tick labels
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))

    # Legend
    ax.legend(
        title="Jazan sub-region",
        frameon=False,
        loc="upper right",
    )

    # PERMANOVA annotation
    ax.text(
        0.02,
        0.02,
        f"PERMANOVA: pseudo-F = {F_obs:.2f}, "
        f"R\u00b2 = {R2_obs:.3f}, p = {p_val:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="dimgray",
    )

    # Styling
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="both", linestyle=":", linewidth=0.6, alpha=0.25)

    fig.suptitle(
        "PCoA of P. falciparum CelTOS sequences\nby Jazan sub-region",
        y=0.99,
        fontsize=13,
        fontweight="bold",
    )

    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out_path = here / "Figure5_pcoa_regions.png"
    fig.savefig(out_path, dpi=600)
    plt.close(fig)

    print(f"Saved figure to: {out_path}")


if __name__ == "__main__":
    main()
