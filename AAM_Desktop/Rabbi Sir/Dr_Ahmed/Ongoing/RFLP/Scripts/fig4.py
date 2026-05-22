#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 4 — Signature Heatmap + Clustering + Discriminability (A–E)
Boss-level v4 (FAST): keeps the figure the same, but makes export fast.

Why Fig 4 can be slow:
  - At 18×11 inches, exporting PNG/JPG at 600 dpi = 10800×6600 pixels (~71M px).
  - Encoding/compressing that is what takes time (not the plotting).

This script:
  - Always exports PDF (fast, vector text; heatmaps embedded as images).
  - Exports PNG/JPG at separate (usually lower) DPI by default for speed.
  - Lets you request 600-dpi rasters when needed.

Outputs:
  - PDF (always)
  - PNG (optional)
  - JPG (optional)

Truth source:
  - assay_table.csv (thesis-derived)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"]  = 42

try:
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import pdist, squareform
except Exception as e:
    raise SystemExit(
        "SciPy is required for Figure 4 clustering/dendrogram.\n"
        "Install:  pip install scipy\n"
        f"Import error: {e}"
    )

PAL = {
    "ink":      (0.07, 0.09, 0.12),
    "slate":    (0.38, 0.42, 0.50),
    "grid":     (0.86, 0.88, 0.92),
    "paper":    (1.00, 1.00, 1.00),
    "panel_bg": (0.98, 0.985, 0.995),
    "spine":    (0.75, 0.78, 0.85),

    "violet":   (0.45, 0.23, 0.72),
    "indigo":   (0.20, 0.26, 0.55),
    "teal":     (0.05, 0.55, 0.60),
    "mint":     (0.78, 0.93, 0.90),
    "coral":    (0.88, 0.36, 0.30),
    "gold":     (0.88, 0.67, 0.12),
    "midnight": (0.05, 0.08, 0.14),

    "header_l1": (0.90, 0.92, 0.99),
    "header_l2": (0.97, 0.98, 1.00),
}

def draw_gradient_bar(ax, c1, c2, steps=180):
    grad = np.linspace(0, 1, steps)
    rgb  = (1-grad)[:, None]*np.array(c1)[None, :] + grad[:, None]*np.array(c2)[None, :]
    img  = np.tile(rgb[None, :, :], (2, 1, 1))
    ax.imshow(img, extent=[0, 1, 0, 1], aspect="auto", interpolation="bicubic")
    ax.set_axis_off()

def tidy_axes(ax):
    ax.set_facecolor(PAL["panel_bg"])
    for sp in ax.spines.values():
        sp.set_color(PAL["spine"])
        sp.set_linewidth(0.8)

def panel_label(ax, letter):
    ax.text(0.005, 1.01, letter, transform=ax.transAxes,
            ha="left", va="bottom", fontsize=12, fontweight="bold", color=PAL["ink"])

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def parse_fragments(cell):
    if cell is None:
        return []
    s = str(cell).strip()
    if s == "" or s.lower() in {"na", "nan", "none"}:
        return []
    out = []
    for alt in s.split("|"):
        alt = alt.strip()
        if not alt or alt.lower() in {"na", "nan"}:
            continue
        parts = [p.strip() for p in alt.split("+") if p.strip()]
        vals  = []
        ok = True
        for p in parts:
            try:
                vals.append(int(round(float(p))))
            except Exception:
                ok = False
                break
        if ok and vals:
            out.append(sorted(vals, reverse=True))
    return out

def first_or_empty(alts):
    return alts[0] if len(alts) else []

def signature_vector(row, k=3):
    wt  = first_or_empty(parse_fragments(row.get("wt_fragments_bp", "")))
    mut = first_or_empty(parse_fragments(row.get("mut_fragments_bp", "")))

    vec = []
    for i in range(k):
        vec.append(float(wt[i]) if i < len(wt) else np.nan)
    for i in range(k):
        vec.append(float(mut[i]) if i < len(mut) else np.nan)

    amp = row.get("amplicon_bp", np.nan)
    try:
        amp = float(amp)
    except Exception:
        amp = np.nan
    vec.append(amp)

    vec.append(float(len(wt))  if len(wt)  else np.nan)
    vec.append(float(len(mut)) if len(mut) else np.nan)
    return np.array(vec, dtype=float)

def robust_scale_0_1(X):
    Xs = X.copy().astype(float)
    for j in range(Xs.shape[1]):
        col = Xs[:, j]
        m = ~np.isnan(col)
        if m.sum() < 3:
            vals = col[m]
            if vals.size == 0:
                continue
            mn, mx = float(np.min(vals)), float(np.max(vals))
            den = (mx - mn) if (mx - mn) > 1e-9 else 1.0
            col[m] = (vals - mn) / den
            Xs[:, j] = col
            continue
        vals = col[m]
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med)))
        scale = 1.4826*mad if mad > 1e-9 else (np.std(vals) if np.std(vals) > 1e-9 else 1.0)
        z = (vals - med) / scale
        z = np.clip(z, -2.5, 2.5)
        col[m] = (z + 2.5) / 5.0
        Xs[:, j] = col
    return Xs

def locus_discriminator_score(row):
    wt  = first_or_empty(parse_fragments(row.get("wt_fragments_bp", "")))
    mut = first_or_empty(parse_fragments(row.get("mut_fragments_bp", "")))
    if len(wt) == 0 or len(mut) == 0:
        return np.nan
    md = min(abs(a - b) for a in wt for b in mut)
    return float(md) + 20.0*float(abs(len(mut) - len(wt)))

def confusability_matrix(rows, tau=10.0):
    muts = [first_or_empty(parse_fragments(r.get("mut_fragments_bp", ""))) for r in rows]
    n = len(rows)
    M = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i+1, n):
            a, b = muts[i], muts[j]
            if len(a) == 0 or len(b) == 0:
                continue
            if len(a) != len(b):
                continue
            a2, b2 = sorted(a, reverse=True), sorted(b, reverse=True)
            dif = max(abs(a2[t] - b2[t]) for t in range(len(a2)))
            if dif <= tau:
                M[i, j] = 1
                M[j, i] = 1
    return M

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assay_csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--title", default="Signature Heatmap + Clustering + Discriminability")
    ap.add_argument("--tau", type=float, default=10.0)

    # FAST export controls
    ap.add_argument("--pdf", action="store_true", help="Export PDF (default true).")
    ap.add_argument("--png", action="store_true", help="Export PNG.")
    ap.add_argument("--jpg", action="store_true", help="Export JPG.")

    ap.add_argument("--png_dpi", type=int, default=300, help="PNG DPI (fast default 300).")
    ap.add_argument("--jpg_dpi", type=int, default=300, help="JPG DPI (fast default 300).")
    ap.add_argument("--jpg_quality", type=int, default=92, help="JPG quality (1–100). Higher = slower/larger.")

    args = ap.parse_args()

    # Default: PDF + PNG + JPG if none selected (keeps prior behavior but fast DPI)
    if not (args.pdf or args.png or args.jpg):
        args.pdf = args.png = args.jpg = True

    df = pd.read_csv(args.assay_csv)
    df = df[df["locus"].notna()].copy()
    df["locus"] = df["locus"].astype(str)

    k = 3
    X = np.vstack([signature_vector(df.iloc[i].to_dict(), k=k) for i in range(len(df))])

    # impute missing
    X_imp = X.copy()
    for j in range(X_imp.shape[1]):
        col = X_imp[:, j]
        m = ~np.isnan(col)
        med = float(np.median(col[m])) if m.sum() else 0.0
        col[~m] = med
        X_imp[:, j] = col

    X_scaled = robust_scale_0_1(X_imp)

    # clustering
    D = pdist(X_scaled, metric="euclidean")
    Z = linkage(D, method="ward")
    order = dendrogram(Z, no_plot=True)["leaves"]

    df_ord = df.iloc[order].reset_index(drop=True)
    Xo = X_scaled[order, :]

    heat_cols = list(range(0, 2*k)) + [2*k]
    H = Xo[:, heat_cols]

    cmap_heat = LinearSegmentedColormap.from_list(
        "heat_violet", [PAL["header_l2"], (0.77, 0.88, 0.98), (0.52, 0.73, 0.96), PAL["violet"]]
    )
    cmap_dist = LinearSegmentedColormap.from_list(
        "dist_teal", [PAL["header_l2"], PAL["mint"], (0.30, 0.78, 0.75), PAL["teal"]]
    )
    cmap_bin  = ListedColormap([PAL["header_l2"], PAL["coral"]], name="bin_coral")

    dist_mat = squareform(D)
    dist_ord = dist_mat[np.ix_(order, order)]

    rows = [df_ord.iloc[i].to_dict() for i in range(len(df_ord))]
    conf = confusability_matrix(rows, tau=args.tau)

    df_ord["disc_score"] = [locus_discriminator_score(rows[i]) for i in range(len(rows))]
    df_rank = df_ord.dropna(subset=["disc_score"]).sort_values("disc_score", ascending=False).reset_index(drop=True)

    # Layout: large gaps (avoid overlaps)
    fig = plt.figure(figsize=(18, 11), dpi=170)
    fig.patch.set_facecolor(PAL["paper"])
    gs = GridSpec(24, 26, figure=fig, wspace=1.25, hspace=1.25)

    ax_head = fig.add_subplot(gs[0:2, :])
    draw_gradient_bar(ax_head, PAL["header_l1"], PAL["header_l2"])
    ax_head.text(0.5, 0.5, args.title, ha="center", va="center",
                 fontsize=17, fontweight="bold", color=PAL["ink"])

    axB  = fig.add_subplot(gs[3:6, 0:13])
    axA  = fig.add_subplot(gs[7:22, 0:12])
    caxA = fig.add_subplot(gs[7:22, 12:13])

    axC  = fig.add_subplot(gs[3:12, 16:25])
    caxC = fig.add_subplot(gs[3:12, 25:26])

    axD  = fig.add_subplot(gs[13:17, 16:25])
    axE  = fig.add_subplot(gs[18:22, 16:25])

    # B dendrogram (single premium color)
    tidy_axes(axB)
    indigo_hex = matplotlib.colors.to_hex(PAL["indigo"])
    dendrogram(
        Z, ax=axB, orientation="top", no_labels=True,
        color_threshold=0,
        above_threshold_color=indigo_hex,
        link_color_func=lambda k: indigo_hex
    )
    axB.set_xticks([]); axB.set_yticks([])
    panel_label(axB, "B")
    axB.text(0.5, 1.02, "Clustering tree (Ward linkage)", transform=axB.transAxes,
             ha="center", va="bottom", fontsize=11, color=PAL["ink"])

    # A heatmap
    tidy_axes(axA)
    imA = axA.imshow(H, aspect="auto", interpolation="nearest", cmap=cmap_heat, vmin=0.0, vmax=1.0)
    panel_label(axA, "A")
    axA.text(0.5, 1.02, "Standardized fragment-size signature heatmap", transform=axA.transAxes,
             ha="center", va="bottom", fontsize=11, color=PAL["ink"])

    loci = df_ord["locus"].tolist()
    axA.set_yticks(np.arange(len(loci)))
    axA.set_yticklabels(loci, fontsize=8, color=PAL["ink"])

    xnames = [f"WT{j+1}" for j in range(k)] + [f"Mut{j+1}" for j in range(k)] + ["Amplicon"]
    axA.set_xticks(np.arange(len(xnames)))
    axA.set_xticklabels(xnames, fontsize=9, color=PAL["ink"])
    axA.tick_params(axis="x", pad=6)

    for i in range(len(loci) + 1):
        axA.axhline(i - 0.5, color=PAL["grid"], linewidth=0.6)

    tidy_axes(caxA)
    cbA = plt.colorbar(imA, cax=caxA)
    cbA.set_ticks([0.0, 0.5, 1.0])
    cbA.set_ticklabels(["low", "mid", "high"])
    cbA.ax.tick_params(labelsize=8, colors=PAL["slate"])
    cbA.outline.set_edgecolor(PAL["spine"])
    cbA.outline.set_linewidth(0.8)

    # C distance
    tidy_axes(axC)
    panel_label(axC, "C")
    imC = axC.imshow(dist_ord, aspect="auto", interpolation="nearest", cmap=cmap_dist)
    axC.set_xticks([]); axC.set_yticks([])
    axC.text(0.5, 1.02, "Pairwise signature distance", transform=axC.transAxes,
             ha="center", va="bottom", fontsize=11, color=PAL["ink"])

    tidy_axes(caxC)
    cbC = plt.colorbar(imC, cax=caxC)
    cbC.ax.tick_params(labelsize=8, colors=PAL["slate"])
    cbC.outline.set_edgecolor(PAL["spine"])
    cbC.outline.set_linewidth(0.8)

    # D confusability
    tidy_axes(axD)
    panel_label(axD, "D")
    axD.imshow(conf, aspect="auto", interpolation="nearest", cmap=cmap_bin, vmin=0, vmax=1)
    axD.set_xticks([]); axD.set_yticks([])
    pairs = int(np.sum(np.triu(conf, 1)))
    axD.text(0.5, 1.02, f"Confusability (Mut signatures within ±{args.tau:g} bp) • pairs={pairs}",
             transform=axD.transAxes, ha="center", va="bottom", fontsize=10.2, color=PAL["ink"])

    # degree inset (kept)
    deg = conf.sum(axis=1)
    inset = axD.inset_axes([0.84, 0.08, 0.15, 0.84])
    inset.set_facecolor(PAL["panel_bg"])
    inset.barh(np.arange(len(deg)), deg,
               color=matplotlib.colors.to_hex(PAL["teal"]),
               edgecolor=matplotlib.colors.to_hex(PAL["midnight"]), linewidth=0.4)
    inset.invert_yaxis()
    inset.set_xticks([]); inset.set_yticks([])
    for sp in inset.spines.values():
        sp.set_visible(False)
    inset.text(0.0, 1.02, "degree", transform=inset.transAxes, fontsize=8, color=PAL["slate"], ha="left")

    # E discriminators
    tidy_axes(axE)
    panel_label(axE, "E")
    topn = min(10, len(df_rank))
    show = df_rank.iloc[:topn].copy()
    axE.text(0.5, 1.02, "Top discriminators (WT vs Mut separation score)", transform=axE.transAxes,
             ha="center", va="bottom", fontsize=10.6, color=PAL["ink"])

    y = np.arange(topn)[::-1]
    axE.barh(y, show["disc_score"].values[::-1],
             color=matplotlib.colors.to_hex(PAL["gold"]),
             edgecolor=matplotlib.colors.to_hex(PAL["midnight"]), linewidth=0.5)
    axE.set_yticks(y)
    axE.set_yticklabels(show["locus"].values[::-1], fontsize=9, color=PAL["ink"])
    axE.tick_params(axis="x", labelsize=9, colors=PAL["slate"])
    axE.grid(True, axis="x", color=PAL["grid"], linewidth=0.8)
    axE.set_xlabel("Score (higher = more distinct)", fontsize=9, color=PAL["slate"], labelpad=2)

    fig.text(0.02, 0.02,
             "Computed solely from thesis-derived assay_table (expected fragment sizes). "
             "Heatmap uses robust column-wise scaling; confusability is based on mutant bands with tolerance τ.",
             fontsize=8, color=PAL["slate"])

    outdir = Path(args.outdir)
    ensure_dir(outdir)
    base = outdir / "Figure_4_Signature_Discriminability"

    fig.subplots_adjust(left=0.045, right=0.985, top=0.965, bottom=0.065)

    # --- Exports (FAST) ---
    # PDF: fast, crisp
    if args.pdf:
        fig.savefig(str(base) + ".pdf", facecolor=PAL["paper"])

    # PNG/JPG: fast default 300 dpi; raise when needed
    if args.png:
        fig.savefig(str(base) + ".png", dpi=int(args.png_dpi), facecolor=PAL["paper"])

    if args.jpg:
        # Faster JPEG: avoid heavy optimize; keep good quality
        fig.savefig(
            str(base) + ".jpg",
            dpi=int(args.jpg_dpi),
            facecolor=PAL["paper"],
            pil_kwargs={"quality": int(args.jpg_quality), "subsampling": 2}
        )

    plt.close(fig)

if __name__ == "__main__":
    main()
