# premium_seed_sites_figure.py
# Seed-site Figures v13 (3 separate premium figures) — robust labeling + anti-overlap
# Reads:  miRNA.xlsx (Sheet1) from the same folder as this script
# Writes: PNG (600 dpi) + PDF (600 dpi) into .\figure\

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm

# -----------------------
# USER-FACING SETTINGS
# -----------------------
VERSION = "v13"
EXCEL_NAME = "miRNA.xlsx"
SHEET_NAME = "Sheet1"

OUT_DIR_NAME = "figure"

PNG_DPI = 600
PDF_DPI = 600

# Keep PDFs small: rasterize heavy artists but keep text vector
RASTERIZE_DPI = 300  # internal rasterization target used by Matplotlib's mixed-mode renderer

# Deterministic jitter for coincident points
JITTER_SEED = 7
JITTER_POINTS = 6  # jitter radius in "points" (display units) for identical-coordinate stacks

# Label styling
LABEL_FONTSIZE = 9.5
LABEL_BOX_ALPHA = 0.88
LABEL_ARROW_ALPHA = 0.55

# Global look
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titleweight": "bold",
    "axes.titlesize": 18,
    "axes.labelsize": 14,
    "xtick.labelsize": 11.5,
    "ytick.labelsize": 11.5,
    "figure.dpi": 160,
})

# -----------------------
# HELPERS
# -----------------------
def ensure_outdir(base_dir: Path) -> Path:
    outdir = base_dir / OUT_DIR_NAME
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir

def clear_collections(ax):
    for coll in list(ax.collections):
        coll.remove()

def read_data(base_dir: Path) -> pd.DataFrame:
    xlsx = base_dir / EXCEL_NAME
    if not xlsx.exists():
        raise FileNotFoundError(f"Excel not found: {xlsx}")

    df = pd.read_excel(xlsx, sheet_name=SHEET_NAME)

    required = ["Target_id", "seed_best_start", "seed_best_end", "seed_best_wobble"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in Excel sheet: {missing}")

    df = df[required].copy()

    # numeric safety
    df["seed_best_start"] = pd.to_numeric(df["seed_best_start"], errors="coerce")
    df["seed_best_end"] = pd.to_numeric(df["seed_best_end"], errors="coerce")
    df["seed_best_wobble"] = pd.to_numeric(df["seed_best_wobble"], errors="coerce")
    df = df.dropna(subset=["Target_id", "seed_best_start", "seed_best_end", "seed_best_wobble"])

    df["Target_id"] = df["Target_id"].astype(str).str.strip()
    df["seed_best_wobble"] = df["seed_best_wobble"].astype(int)
    df["seed_best_start"] = df["seed_best_start"].astype(int)
    df["seed_best_end"] = df["seed_best_end"].astype(int)
    df["midpoint"] = (df["seed_best_start"] + df["seed_best_end"]) / 2.0

    return df


def save_figure_bundle(fig: plt.Figure, outprefix: Path) -> tuple[Path, Path]:
    png_path = outprefix.with_suffix(".png")
    pdf_path = outprefix.with_suffix(".pdf")

    # Tight layout first
    try:
        fig.tight_layout()
    except Exception:
        pass

    # PNG
    fig.savefig(
        png_path,
        dpi=PNG_DPI,
        bbox_inches="tight",
        facecolor="white",
        transparent=False
    )

    # PDF (mixed-mode: heavy artists rasterized, text stays vector)
    # NOTE: PDF "dpi" affects rasterized artists only. Text remains vector.
    fig.savefig(
        pdf_path,
        dpi=PDF_DPI,
        bbox_inches="tight",
        facecolor="white",
        transparent=False
    )

    return png_path, pdf_path


def compute_entropy_from_counts(counts: np.ndarray) -> float:
    # Shannon entropy (natural log)
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _points_to_data_dx(ax, points: float) -> float:
    # Convert a displacement in points to data units (x direction)
    fig = ax.figure
    fig.canvas.draw()
    bbox = ax.get_window_extent()
    x0, x1 = ax.get_xlim()
    px_per_data = bbox.width / (x1 - x0) if (x1 - x0) != 0 else 1.0
    points_to_px = fig.dpi / 72.0
    return (points * points_to_px) / px_per_data


def _points_to_data_dy(ax, points: float) -> float:
    # Convert a displacement in points to data units (y direction)
    fig = ax.figure
    fig.canvas.draw()
    bbox = ax.get_window_extent()
    y0, y1 = ax.get_ylim()
    px_per_data = bbox.height / (y1 - y0) if (y1 - y0) != 0 else 1.0
    points_to_px = fig.dpi / 72.0
    return (points * points_to_px) / px_per_data


def jitter_coincident_points(ax, xs, ys, labels, radius_points=JITTER_POINTS, seed=JITTER_SEED):
    """
    If multiple points share identical (x,y), spread them in a small circle (deterministic).
    Returns jittered xs, ys.
    """
    rng = np.random.default_rng(seed)
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)

    # map coordinate -> indices
    key = np.round(np.c_[xs, ys], 10)
    keys, inv, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)

    dx_data = _points_to_data_dx(ax, radius_points)
    dy_data = _points_to_data_dy(ax, radius_points)

    xj = xs.copy()
    yj = ys.copy()

    for k_i, c in enumerate(counts):
        if c <= 1:
            continue
        idx = np.where(inv == k_i)[0]
        # spread in a circle with deterministic phase
        base_phase = float(rng.uniform(0, 2 * np.pi))
        for j, ii in enumerate(idx):
            ang = base_phase + (2 * np.pi * j / c)
            xj[ii] += dx_data * math.cos(ang)
            yj[ii] += dy_data * math.sin(ang)

    return xj, yj


def place_all_labels_no_overlap(ax, xs, ys, labels):
    """
    Place ALL labels with overlap-avoidance:
    - tries spiral offsets around each point
    - checks bbox overlaps against already placed labels
    - uses arrows to connect label to point
    """
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    placed_bboxes = []

    # A tasteful label box
    bbox_kw = dict(boxstyle="round,pad=0.25,rounding_size=0.15",
                   fc="white", ec="none", alpha=LABEL_BOX_ALPHA)

    arrow_kw = dict(arrowstyle="-", lw=0.9, color="0.25", alpha=LABEL_ARROW_ALPHA)

    # Candidate offsets in points (spiral)
    # (small -> larger), many options so we almost always find a free spot
    radii = [8, 12, 16, 22, 30, 40, 52, 66]
    angles = np.linspace(0, 2*np.pi, 12, endpoint=False)

    # Place labels in a stable order (largest y then x) so top labels don't get pushed out
    order = np.lexsort((xs, -ys))

    texts = []
    for i in order:
        x, y, lab = float(xs[i]), float(ys[i]), str(labels[i])

        chosen = None
        text_artist = None

        for r in radii:
            for ang in angles:
                dx = r * math.cos(ang)
                dy = r * math.sin(ang)

                t = ax.annotate(
                    lab,
                    xy=(x, y),
                    xytext=(dx, dy),
                    textcoords="offset points",
                    ha="left" if dx >= 0 else "right",
                    va="bottom" if dy >= 0 else "top",
                    fontsize=LABEL_FONTSIZE,
                    bbox=bbox_kw,
                    arrowprops=arrow_kw,
                    annotation_clip=False,
                    zorder=6
                )

                fig.canvas.draw()
                bb = t.get_window_extent(renderer=renderer).expanded(1.05, 1.15)

                # Check overlap with existing label boxes
                overlap = any(bb.overlaps(prev) for prev in placed_bboxes)

                if not overlap:
                    chosen = bb
                    text_artist = t
                    break

                # Not chosen → remove and try next spot
                t.remove()

            if chosen is not None:
                break

        # Fallback (should be rare): place with bigger offset, accept overlap if unavoidable
        if chosen is None:
            t = ax.annotate(
                lab,
                xy=(x, y),
                xytext=(80, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=LABEL_FONTSIZE,
                bbox=bbox_kw,
                arrowprops=arrow_kw,
                annotation_clip=False,
                zorder=6
            )
            fig.canvas.draw()
            chosen = t.get_window_extent(renderer=renderer).expanded(1.05, 1.15)
            text_artist = t

        placed_bboxes.append(chosen)
        texts.append(text_artist)

    # Expand axes view slightly so nothing gets clipped
    ax.margins(x=0.12, y=0.18)
    return texts


def nice_diverging_norm(values, center=0.0):
    """
    Safe diverging normalization:
    - if all values are same (or vmin==vmax), fall back to simple Normalize
    - else use TwoSlopeNorm with correct ordering
    """
    v = np.asarray(values, dtype=float)
    vmin = float(np.nanmin(v))
    vmax = float(np.nanmax(v))

    if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1e-12:
        return Normalize(vmin=vmin - 1e-6, vmax=vmax + 1e-6)

    # ensure center lies within [vmin, vmax]
    vcenter = float(np.clip(center, vmin + 1e-12, vmax - 1e-12))
    # ensure ascending
    if not (vmin < vcenter < vmax):
        # fallback to normalize
        return Normalize(vmin=vmin, vmax=vmax)

    return TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)


# -----------------------
# FIGURE 1 (kept as-is; your v11 already produced a good PDF size)
# -----------------------
def figure_delta_startpos_vs_wt(df: pd.DataFrame, targets: list[str]) -> plt.Figure:
    # Start-position distribution per target
    g = df.groupby(["Target_id", "seed_best_start"], observed=True).size().reset_index(name="n")
    totals = g.groupby("Target_id", observed=True)["n"].transform("sum")
    g["pct"] = 100.0 * g["n"] / totals

    # Pivot to compare to WT
    pivot = g.pivot(index="Target_id", columns="seed_best_start", values="pct").fillna(0.0)
    if "Wild Type" not in pivot.index:
        raise ValueError("Wild Type not found in Target_id. Please ensure it exists exactly as 'Wild Type'.")

    wt = pivot.loc["Wild Type"].values
    starts = pivot.columns.values.astype(int)

    # Δ distribution summary per target: signed "center of mass" shift
    def center_of_mass(pct_row):
        return float(np.sum(starts * pct_row) / (np.sum(pct_row) + 1e-12))

    com = pivot.apply(center_of_mass, axis=1)
    delta = com - com.loc["Wild Type"]

    # Also compute "distribution distance" vs WT (L1 / 2)
    # (0 = identical, 1 = completely different)
    dist = pivot.apply(lambda r: 0.5 * np.sum(np.abs(r.values - wt)) / 100.0, axis=1)

    # Order by delta then distance
    order = delta.sort_values().index.tolist()

    fig = plt.figure(figsize=(10.6, 6.8))
    ax = fig.add_subplot(111)

    y = np.arange(len(order))
    ax.axvline(0, lw=1.2, color="0.75", zorder=1)

    # Color by distance (premium, readable)
    sc = ax.scatter(
        delta.loc[order].values,
        y,
        s=(140 + 1400 * dist.loc[order].values),
        c=dist.loc[order].values,
        cmap="viridis",
        edgecolor="white",
        linewidth=0.9,
        alpha=0.95,
        zorder=3,
        rasterized=True  # keeps PDF small
    )

    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.invert_yaxis()

    ax.set_title(f"Fig 1. Start-position shift vs Wild Type\n"
                 f"(x = Δ center-of-mass of start position; color = distribution distance; size = distance-weighted)")
    ax.set_xlabel("Δ start-position center-of-mass (nt) vs Wild Type")
    ax.set_ylabel("Target / variant")

    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Distribution distance vs WT (L1/2 on %)", rotation=90)

    ax.grid(True, axis="x", alpha=0.22)
    ax.grid(False, axis="y")

    return fig


# -----------------------
# FIGURE 2 (FIXED: all labels, no overlaps, no outside clipping)
# -----------------------
def figure_wobble_signature(df: pd.DataFrame, targets: list[str]) -> plt.Figure:
    # For each target: mean wobble + entropy of wobble composition
    wobble_counts = (
        df.groupby(["Target_id", "seed_best_wobble"], observed=True)
          .size()
          .unstack(fill_value=0)
    )

    # Ensure columns 0..4 exist
    for k in range(0, 5):
        if k not in wobble_counts.columns:
            wobble_counts[k] = 0
    wobble_counts = wobble_counts[[0, 1, 2, 3, 4]]

    n = wobble_counts.sum(axis=1).astype(int)
    mean_w = (wobble_counts.values * np.arange(5)).sum(axis=1) / (n.values + 1e-12)
    ent = np.array([compute_entropy_from_counts(row.values.astype(float)) for _, row in wobble_counts.iterrows()])

    stats = pd.DataFrame({
        "Target_id": wobble_counts.index,
        "n": n.values,
        "mean_wobble": mean_w,
        "entropy": ent,
    }).set_index("Target_id")

    if "Wild Type" not in stats.index:
        raise ValueError("Wild Type not found in Target_id. Please ensure it exists exactly as 'Wild Type'.")

    stats["delta_mean_wobble"] = stats["mean_wobble"] - float(stats.loc["Wild Type", "mean_wobble"])

    # Build figure
    fig = plt.figure(figsize=(12.8, 7.6))
    ax = fig.add_subplot(111)

    # Scatter data
    xs = stats.loc[targets, "mean_wobble"].values
    ys = stats.loc[targets, "entropy"].values
    cs = stats.loc[targets, "delta_mean_wobble"].values
    ns = stats.loc[targets, "n"].values

    # colormap + safe diverging norm
    norm = nice_diverging_norm(cs, center=0.0)

    # Base scatter (rasterized for small PDF)
    sc = ax.scatter(
        xs, ys,
        s=(180 + 1800 * (ns / (ns.max() + 1e-12))),
        c=cs,
        cmap="PuOr",
        norm=norm,
        edgecolor="white",
        linewidth=1.0,
        alpha=0.93,
        zorder=3,
        rasterized=True
    )

    # Reference lines at WT
    ax.axvline(float(stats.loc["Wild Type", "mean_wobble"]), color="0.75", lw=1.2, zorder=1)
    ax.axhline(float(stats.loc["Wild Type", "entropy"]), color="0.75", lw=1.2, zorder=1)

    ax.set_title("Fig 2. Wobble signature map\n"
                 "(x = mean wobble, y = wobble diversity/entropy, color = Δ mean wobble vs WT, size = n)")
    ax.set_xlabel("Mean wobble")
    ax.set_ylabel("Wobble entropy (diversity)")
    ax.grid(True, alpha=0.22)

    # Expand limits a bit before label placement
    ax.margins(x=0.08, y=0.12)

    # IMPORTANT: jitter only if coincident points exist (prevents label piles)
    labels = np.array(targets, dtype=object)
    xj, yj = jitter_coincident_points(ax, xs, ys, labels, radius_points=JITTER_POINTS, seed=JITTER_SEED)

    # Re-plot at jittered positions (hide original points by clearing then re-adding)
    # Keep the same colors/sizes but at shifted coordinates
    clear_collections(ax)
    sc = ax.scatter(
        xj, yj,
        s=(180 + 1800 * (ns / (ns.max() + 1e-12))),
        c=cs,
        cmap="PuOr",
        norm=norm,
        edgecolor="white",
        linewidth=1.0,
        alpha=0.93,
        zorder=3,
        rasterized=True
    )
    ax.axvline(float(stats.loc["Wild Type", "mean_wobble"]), color="0.75", lw=1.2, zorder=1)
    ax.axhline(float(stats.loc["Wild Type", "entropy"]), color="0.75", lw=1.2, zorder=1)

    # Place ALL labels with no overlaps
    place_all_labels_no_overlap(ax, xj, yj, labels)

    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Δ mean wobble vs WT", rotation=90)

    return fig


# -----------------------
# FIGURE 3 (FIXED: if all deltas ~0, still shows all points separated + labeled)
# -----------------------
def figure_peak_shift_vs_wt(df: pd.DataFrame, targets: list[str]) -> plt.Figure:
    # For each target, find the most frequent seed window (start,end)
    g = df.groupby(["Target_id", "seed_best_start", "seed_best_end"], observed=True).size().reset_index(name="n")
    g["rank"] = g.groupby("Target_id", observed=True)["n"].rank(method="first", ascending=False)

    top = g[g["rank"] == 1].copy()
    top["midpoint"] = (top["seed_best_start"] + top["seed_best_end"]) / 2.0

    # Merge median wobble of that top window for each target
    df2 = df.copy()
    df2["window"] = df2["seed_best_start"].astype(str) + "-" + df2["seed_best_end"].astype(str)
    top["window"] = top["seed_best_start"].astype(str) + "-" + top["seed_best_end"].astype(str)

    med_w = (
        df2.groupby(["Target_id", "window"], observed=True)["seed_best_wobble"]
           .median()
           .reset_index(name="median_wobble")
    )
    top = top.merge(med_w, on=["Target_id", "window"], how="left")

    if "Wild Type" not in top["Target_id"].values:
        raise ValueError("Wild Type not found in Target_id. Please ensure it exists exactly as 'Wild Type'.")

    wt_mid = float(top.loc[top["Target_id"] == "Wild Type", "midpoint"].iloc[0])
    wt_medw = float(top.loc[top["Target_id"] == "Wild Type", "median_wobble"].iloc[0])

    # Δ shift vs WT
    top["dx"] = top["midpoint"] - wt_mid
    top["dy"] = top["median_wobble"] - wt_medw

    # Keep targets order
    top = top.set_index("Target_id").loc[targets].reset_index()

    xs = top["dx"].values.astype(float)
    ys = top["dy"].values.astype(float)

    # If everything collapses to 0, we still want to see separation
    fig = plt.figure(figsize=(12.8, 7.6))
    ax = fig.add_subplot(111)

    sc = ax.scatter(
        xs, ys,
        s=(220 + 1600 * (top["n"].values / (top["n"].values.max() + 1e-12))),
        c=top["dy"].values,  # color by dy (peak wobble shift)
        cmap="YlGnBu",
        edgecolor="white",
        linewidth=1.0,
        alpha=0.93,
        zorder=3,
        rasterized=True
    )

    ax.axvline(0, color="0.75", lw=1.2, zorder=1)
    ax.axhline(0, color="0.75", lw=1.2, zorder=1)

    ax.set_title("Fig 3. Peak shift vs Wild Type (top window)\n"
                 "(x = Δ midpoint (nt), y = Δ median wobble at peak)")
    ax.set_xlabel("Δ seed-window midpoint (nt) vs WT")
    ax.set_ylabel("Δ median wobble at peak window vs WT")
    ax.grid(True, alpha=0.22)

    # Add a little margin BEFORE jitter/labels
    ax.margins(x=0.25, y=0.25)

    labels = top["Target_id"].values.astype(object)

    # Jitter coincident points so every variant is visible + labelable
    xj, yj = jitter_coincident_points(ax, xs, ys, labels, radius_points=10, seed=JITTER_SEED + 1)

    # Replace scatter at jittered coords (keep color/size)
    clear_collections(ax)
    sc = ax.scatter(
        xj, yj,
        s=(220 + 1600 * (top["n"].values / (top["n"].values.max() + 1e-12))),
        c=top["dy"].values,
        cmap="YlGnBu",
        edgecolor="white",
        linewidth=1.0,
        alpha=0.93,
        zorder=3,
        rasterized=True
    )
    ax.axvline(0, color="0.75", lw=1.2, zorder=1)
    ax.axhline(0, color="0.75", lw=1.2, zorder=1)

    # Label ALL points with overlap-avoidance
    place_all_labels_no_overlap(ax, xj, yj, labels)

    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Δ median wobble (peak) vs WT", rotation=90)

    return fig


# -----------------------
# MAIN
# -----------------------
def main():
    base_dir = Path(__file__).resolve().parent
    outdir = ensure_outdir(base_dir)

    print(f"=== Seed-site Figures {VERSION} (premium anti-overlap labels) ===")
    print(f"Input : {base_dir / EXCEL_NAME} [{SHEET_NAME}]")
    print(f"Output: {outdir}")

    df = read_data(base_dir)

    targets = sorted(df["Target_id"].unique().tolist())
    # Put WT last (or first) depending on taste — I'll keep WT last as you had earlier runs
    if "Wild Type" in targets:
        targets = [t for t in targets if t != "Wild Type"] + ["Wild Type"]

    print(f"Rows: {len(df):,} | Targets: {len(targets)} | Unique starts: {df['seed_best_start'].nunique()}")

    # ---- Fig 1
    fig1 = figure_delta_startpos_vs_wt(df, targets)
    p1 = outdir / f"Fig1_delta_startpos_vs_WT_{VERSION}"
    png1, pdf1 = save_figure_bundle(fig1, p1)
    plt.close(fig1)
    print(f"[OK] PNG: {png1.name}")
    print(f"[OK] PDF: {pdf1.name} (check size in folder)")

    # ---- Fig 2
    fig2 = figure_wobble_signature(df, targets)
    p2 = outdir / f"Fig2_wobble_signature_map_{VERSION}"
    png2, pdf2 = save_figure_bundle(fig2, p2)
    plt.close(fig2)
    print(f"[OK] PNG: {png2.name}")
    print(f"[OK] PDF: {pdf2.name} (check size in folder)")

    # ---- Fig 3
    fig3 = figure_peak_shift_vs_wt(df, targets)
    p3 = outdir / f"Fig3_peak_shift_vs_WT_{VERSION}"
    png3, pdf3 = save_figure_bundle(fig3, p3)
    plt.close(fig3)
    print(f"[OK] PNG: {png3.name}")
    print(f"[OK] PDF: {pdf3.name} (check size in folder)")

    print("Done.")


if __name__ == "__main__":
    main()
