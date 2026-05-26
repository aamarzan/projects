#!/usr/bin/env python
"""
Compute global dN/dS (omega) from MEGA-exported pairwise distance matrices.

Expected input:
  - CelTOS_pairwise_dN.xlsx
  - CelTOS_pairwise_dS.xlsx

Both files should be the lower-triangular pairwise matrices that MEGA exports,
with:
    row 0, col 1..N : sequence names (header)
    col 0, row 1..N : sequence names (index)
    rows 1..N, cols 1..row : distances (lower triangle)

Usage (from the folder containing the Excel files):

    python compute_global_dnds.py

You can optionally specify paths and the dS threshold:

    python compute_global_dnds.py \
        --dn CelTOS_pairwise_dN.xlsx \
        --ds CelTOS_pairwise_dS.xlsx \
        --ds-min 0.001
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def read_mega_triangular_excel(path: Path) -> pd.DataFrame:
    """
    Read a MEGA lower-triangular distance matrix from an Excel file and
    reconstruct a full symmetric NxN DataFrame with identical row/column names.

    We *do not* drop any all-NaN rows, because the first row (sequence 1) is
    often NaN in the lower triangle export. Instead, we read the raw sheet and
    use the header row + first column to decide the labels.
    """
    print(f"Loading distance matrix from: {path.name}")
    raw = pd.read_excel(path, header=None)

    # Header row: sequence names in columns 1..N
    names = raw.iloc[0, 1:].astype(str).tolist()
    n = len(names)
    if n == 0:
        raise ValueError(f"No sequence names found in header of {path}")

    # Prepare an empty symmetric matrix
    mat = pd.DataFrame(
        np.nan,
        index=names,
        columns=names,
        dtype=float,
    )

    # Fill from the lower triangle in the raw sheet
    n_rows, n_cols = raw.shape
    for i in range(1, n + 1):  # row index in raw (1..N)
        row_name = str(raw.iloc[i, 0])
        if row_name not in mat.index:
            # Sometimes MEGA leaves an extra empty row at bottom
            continue

        # Only columns 1..i contain lower-triangle entries on row i
        max_j = min(i, n_cols - 1)
        for j in range(1, max_j + 1):
            val = raw.iloc[i, j]
            if pd.isna(val):
                continue
            col_name = str(raw.iloc[0, j])
            if col_name not in mat.columns:
                continue

            val = float(val)
            mat.loc[row_name, col_name] = val
            mat.loc[col_name, row_name] = val  # enforce symmetry

    # Diagonal distances are zero by definition
    np.fill_diagonal(mat.values, 0.0)

    return mat


def compute_global_omega(
    dn_mat: pd.DataFrame,
    ds_mat: pd.DataFrame,
    ds_min: float = 0.001,
) -> Tuple[dict, pd.DataFrame]:
    """
    Compute global dN/dS (omega) using only pairs with dS >= ds_min.

    Returns:
        summary: dict with counts and global means
        pairs_df: DataFrame with per-pair dN, dS (after filtering)
    """
    if dn_mat.shape != ds_mat.shape:
        raise ValueError("dN and dS matrices have different shapes")

    if list(dn_mat.index) != list(ds_mat.index) or list(dn_mat.columns) != list(
        ds_mat.columns
    ):
        raise ValueError("Row/column labels differ between dN and dS matrices")

    names = list(dn_mat.index)
    n = len(names)
    n_pairs_total = n * (n - 1) // 2

    records = []
    for i in range(n):
        for j in range(i):  # i > j → each unordered pair once
            s1 = names[i]
            s2 = names[j]
            dN = dn_mat.iat[i, j]
            dS = ds_mat.iat[i, j]

            if pd.isna(dN) or pd.isna(dS):
                continue

            records.append((s1, s2, dN, dS))

    pairs_df = pd.DataFrame(
        records, columns=["seq_i", "seq_j", "dN", "dS"]
    )

    # Filter by dS threshold
    before = len(pairs_df)
    pairs_df = pairs_df.loc[pairs_df["dS"] >= ds_min].copy()
    after = len(pairs_df)

    if after == 0:
        raise ValueError(
            f"No pairs retained after applying dS >= {ds_min}. "
            "Try using a smaller threshold."
        )

    mean_dN = pairs_df["dN"].mean()
    mean_dS = pairs_df["dS"].mean()
    omega = mean_dN / mean_dS

    summary = {
        "n_sequences": n,
        "n_pairs_total": n_pairs_total,
        "n_pairs_in_matrix": before,
        "n_pairs_excluded_ds_lt_min": before - after,
        "n_pairs_used": after,
        "ds_min": ds_min,
        "mean_dN": float(mean_dN),
        "mean_dS": float(mean_dS),
        "omega": float(omega),
    }

    return summary, pairs_df


def main():
    parser = argparse.ArgumentParser(
        description="Compute global dN/dS (omega) from MEGA pairwise matrices."
    )
    parser.add_argument(
        "--dn",
        type=str,
        default="CelTOS_pairwise_dN.xlsx",
        help="Excel file with pairwise dN matrix (MEGA export).",
    )
    parser.add_argument(
        "--ds",
        type=str,
        default="CelTOS_pairwise_dS.xlsx",
        help="Excel file with pairwise dS matrix (MEGA export).",
    )
    parser.add_argument(
        "--ds-min",
        type=float,
        default=0.001,
        help="Minimum dS for including a pair (default: 0.001).",
    )
    parser.add_argument(
        "--save-pairs",
        type=str,
        default="",
        help="Optional: path to save per-pair dN/dS table as CSV.",
    )

    args = parser.parse_args()

    dn_file = Path(args.dn)
    ds_file = Path(args.ds)

    dn_mat = read_mega_triangular_excel(dn_file)
    ds_mat = read_mega_triangular_excel(ds_file)

    summary, pairs_df = compute_global_omega(
        dn_mat, ds_mat, ds_min=args.ds_min
    )

    # Nicely formatted report
    print("\n=== Global dN/dS summary ===")
    print(f"Sequences (N):                  {summary['n_sequences']}")
    print(f"Total pairs (N*(N-1)/2):        {summary['n_pairs_total']}")
    print(f"Pairs present in matrices:      {summary['n_pairs_in_matrix']}")
    print(
        f"Pairs excluded (dS < {summary['ds_min']}): "
        f"{summary['n_pairs_excluded_ds_lt_min']}"
    )
    print(f"Pairs used in final estimate:   {summary['n_pairs_used']}")
    print()
    print(f"Mean dN:                        {summary['mean_dN']:.6f}")
    print(f"Mean dS:                        {summary['mean_dS']:.6f}")
    print(f"Global omega (dN/dS):           {summary['omega']:.4f}")
    print("================================\n")

    if args.save_pairs:
        out_path = Path(args.save_pairs)
        pairs_df.to_csv(out_path, index=False)
        print(f"Per-pair dN/dS table saved to: {out_path}")


if __name__ == "__main__":
    main()
