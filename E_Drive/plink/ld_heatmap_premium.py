#!/usr/bin/env python3
"""
Premium LD (r^2) heatmap from PLINK output.
Supports:
  A) PLINK --r2 square/square0 matrix (optionally gzipped)
  B) PLINK --r2 table output (pairwise), columns containing SNP_A, SNP_B, and R2 (or similar)

Exports: PNG (600 dpi), PDF, SVG

Usage examples:
  # Matrix (recommended)
  python ld_heatmap_premium.py --input chr2_ld.ld.gz --out chr2_ld_heatmap --title "LD heatmap (r²)"

  # Pairwise table
  python ld_heatmap_premium.py --input chr2_ld.ld --out chr2_ld_heatmap --mode pairwise
"""

import argparse
import gzip
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def smart_open(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "rt", encoding="utf-8", errors="replace")


def guess_mode(path: Path) -> str:
    """Heuristic: decide if file is a square matrix or a pairwise table."""
    with smart_open(path) as f:
        head = [next(f, "").strip() for _ in range(3)]
    head = [h for h in head if h]

    if not head:
        raise ValueError("Empty input file.")

    # Pairwise table usually has a header with CHR_A/BP_A/SNP_A ... R2
    h0 = re.split(r"\s+", head[0])
    h0_upper = [x.upper() for x in h0]

    pairwise_markers = {"SNP_A", "SNP_B", "R2", "CHR_A", "BP_A", "CHR_B", "BP_B"}
    if any(x in pairwise_markers for x in h0_upper):
        return "pairwise"

    # If first row has many non-numeric tokens and no pairwise markers, it's probably matrix header
    def is_number(x: str) -> bool:
        try:
            float(x)
            return True
        except Exception:
            return False

    nonnum_ct = sum(not is_number(x) for x in h0)
    if nonnum_ct >= 3 and not any(x in pairwise_markers for x in h0_upper):
        return "matrix"


    # Fallback: look at 2nd line: matrix rows often begin with an rsID then many numbers
    if len(head) >= 2:
        h1 = re.split(r"\s+", head[1])
        if len(h1) > 10:
            return "matrix"

    return "pairwise"


def read_matrix_file(path: Path):
    """
    Read PLINK square matrix text (optionally gzipped).
    Handles both formats:
      - leading blank cell in header row
      - no leading blank cell (best-effort)
    Returns: (ids, mat)
    """
    # Read raw as strings so we can detect structure
    raw = pd.read_csv(path, sep=r"\s+", header=None, compression="infer", dtype=str, engine="python")
    if raw.shape[0] < 2 or raw.shape[1] < 2:
        raise ValueError("Matrix file too small to be valid.")

    # If top-left is NaN-like, header starts at col1
    tl = raw.iat[0, 0]
    tl_is_blank = (tl is None) or (str(tl).strip().lower() in {"nan", ""})

    if tl_is_blank:
        ids = raw.iloc[0, 1:].astype(str).tolist()
        row_ids = raw.iloc[1:, 0].astype(str).tolist()
        mat = raw.iloc[1:, 1:].astype(float).to_numpy()
    else:
        # Try to detect whether first row is header or data
        # If many non-numeric tokens in first row -> treat as header
        first_row = raw.iloc[0, :].astype(str).tolist()

        def is_number(x: str) -> bool:
            try:
                float(x)
                return True
            except Exception:
                return False

        nonnum_ct = sum(not is_number(x) for x in first_row)

        if nonnum_ct >= max(2, raw.shape[1] // 4):
            # Treat first row as header, first col as row IDs
            ids = raw.iloc[0, 1:].astype(str).tolist()
            row_ids = raw.iloc[1:, 0].astype(str).tolist()
            mat = raw.iloc[1:, 1:].astype(float).to_numpy()
        else:
            # No header; pure numeric matrix (rare). Create dummy IDs.
            mat = raw.astype(float).to_numpy()
            n = mat.shape[0]
            ids = [f"v{i+1}" for i in range(n)]
            row_ids = ids

    # Align row_ids with ids if possible
    if len(row_ids) == len(ids) and row_ids != ids:
        # Reorder rows to match header ids if row labels are a permutation
        pos = {v: i for i, v in enumerate(row_ids)}
        if all(v in pos for v in ids):
            mat = mat[[pos[v] for v in ids], :]

    # Sanity
    if mat.shape[0] != mat.shape[1]:
        raise ValueError(f"Loaded matrix is not square: {mat.shape}")
    if mat.shape[0] != len(ids):
        raise ValueError("Matrix size and variant ID count mismatch.")

    return ids, mat


def read_pairwise_file(path: Path):
    """
    Read PLINK pairwise LD table and convert to full symmetric matrix.
    Uses SNP_A/SNP_B + BP_A/BP_B to:
      - preserve real rsIDs
      - sort variants by genomic position
      - label as "rsid (bp)" for journal-ready axes
    Returns: (labels, mat)
    """
    df = pd.read_csv(path, sep=r"\s+", compression="infer", engine="python")
    cols_upper = {c.upper(): c for c in df.columns}

    # Required columns (PLINK --r2 table output normally has these)
    snp_a = cols_upper.get("SNP_A") or cols_upper.get("SNP1")
    snp_b = cols_upper.get("SNP_B") or cols_upper.get("SNP2")
    r2col = cols_upper.get("R2") or cols_upper.get("RSQ")

    bp_a = cols_upper.get("BP_A")
    bp_b = cols_upper.get("BP_B")

    if snp_a is None or snp_b is None or r2col is None:
        raise ValueError("Need SNP_A/SNP_B and R2 columns in the input file (PLINK --r2 output).")

    # Build a variant -> bp map (best-effort, still works if BP columns absent)
    pos_map = {}
    if bp_a and bp_b:
        for s, p in zip(df[snp_a].astype(str), df[bp_a].astype(int)):
            pos_map.setdefault(s, p)
        for s, p in zip(df[snp_b].astype(str), df[bp_b].astype(int)):
            pos_map.setdefault(s, p)

    # Unique variants, sorted by BP if available, else keep discovery order
    variants = pd.unique(pd.concat([df[snp_a], df[snp_b]], ignore_index=True)).astype(str).tolist()
    if pos_map:
        variants.sort(key=lambda v: (pos_map.get(v, 10**18), v))

    idx = {v: i for i, v in enumerate(variants)}
    n = len(variants)

    mat = np.full((n, n), np.nan, dtype=float)
    np.fill_diagonal(mat, 1.0)

    a = df[snp_a].astype(str).map(idx).to_numpy()
    b = df[snp_b].astype(str).map(idx).to_numpy()
    r2 = df[r2col].astype(float).to_numpy()

    mat[a, b] = r2
    mat[b, a] = r2

    # Build final axis labels: "rsid (bp)"
    if pos_map:
        labels = [f"{v} ({pos_map.get(v, 'NA')})" for v in variants]
    else:
        labels = variants

    return labels, mat


def plot_ld(ids, mat, out_prefix: Path, title: str, cmap_name: str = "viridis",
            show: str = "lower", max_labels: int = 60):
    """
    Plot triangular LD heatmap in a square axis (journal standard).
    show: 'lower' or 'upper'
    """
    n = mat.shape[0]
    mat = mat.copy()

    # Mask half
    mask = np.triu(np.ones_like(mat, dtype=bool), k=1) if show == "lower" else np.tril(np.ones_like(mat, dtype=bool), k=-1)
    mat = np.ma.array(mat, mask=mask)

    # Colormap with NaNs shown as white
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color="white")

    # Dynamic figure size (keeps readability)
    base = 7.5
    scale = min(1.8, max(1.0, n / 60.0))
    fig_size = base * scale

    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 600,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 1.0,
    })

    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    ax.set_facecolor("white")

    im = ax.imshow(mat, origin="upper", interpolation="nearest", vmin=0, vmax=1, cmap=cmap)
    ax.set_title(title, pad=14)
    ax.set_xlabel("Variants")
    ax.set_ylabel("Variants")
    ax.set_aspect("equal")

    # Ticks: don’t destroy the figure with 500 labels
    if n <= max_labels:
        ticks = np.arange(n)
    else:
        step = math.ceil(n / max_labels)
        ticks = np.arange(0, n, step)

    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([ids[i] for i in ticks], rotation=90, ha="center", va="top")
    ax.set_yticklabels([ids[i] for i in ticks])

    # Subtle border
    for spine in ax.spines.values():
        spine.set_visible(True)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
    cbar.set_label("LD (r²)", rotation=90)

    # Export
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()

    fig.savefig(out_prefix.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_prefix.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="PLINK LD output file (.ld/.ld.gz etc).")
    ap.add_argument("--out", required=True, help="Output prefix (no extension).")
    ap.add_argument("--mode", choices=["auto", "matrix", "pairwise"], default="auto",
                    help="auto = guess format; matrix = PLINK square; pairwise = PLINK table.")
    ap.add_argument("--title", default="Linkage Disequilibrium heatmap (r²)", help="Figure title.")
    ap.add_argument("--cmap", default="viridis", help="Matplotlib colormap (e.g., viridis, magma, cividis).")
    ap.add_argument("--show", choices=["lower", "upper"], default="lower", help="Show which triangle.")
    ap.add_argument("--max-labels", type=int, default=60, help="Max variant labels per axis.")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_prefix = Path(args.out)

    mode = args.mode
    if mode == "auto":
        mode = guess_mode(in_path)

    if mode == "matrix":
        ids, mat = read_matrix_file(in_path)
    else:
        ids, mat = read_pairwise_file(in_path)

    # Quick diagnostics: warn if matrix is mostly missing
    mat_arr = np.array(mat, dtype=float, copy=False)
    missing_frac = np.isnan(mat_arr).mean()

    if missing_frac > 0.50:
        print(f"[WARN] {missing_frac:.1%} of LD values are missing (NaN). "
              f"This usually means you used default windowed --r2 output. "
              f"Use '--r2 square' to get a full matrix. ")

    plot_ld(ids, mat, out_prefix=out_prefix, title=args.title, cmap_name=args.cmap,
            show=args.show, max_labels=args.max_labels)

    print(f"[OK] Saved: {out_prefix}.png / .pdf / .svg")


if __name__ == "__main__":
    main()
