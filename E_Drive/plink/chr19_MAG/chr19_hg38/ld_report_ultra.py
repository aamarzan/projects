#!/usr/bin/env python3
import argparse, gzip, math, csv
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.ticker import ScalarFormatter

# ---------- Premium colormaps (different per figure) ----------
def cmap_from(name, colors):
    return LinearSegmentedColormap.from_list(name, colors, N=256)

CMAP_LD_MATRIX = cmap_from("RoyalAurora", ["#0b1026", "#123d6a", "#1aa6a6", "#f0c54a", "#fff2b2"])
CMAP_MAF       = cmap_from("Emerald",     ["#071a12", "#0f5132", "#2ecc71", "#b7f7cf"])
CMAP_ARCS      = cmap_from("VioletGold",  ["#190028", "#4d1b7b", "#a64dff", "#f7c948"])
CMAP_DECAY     = cmap_from("SunsetInk",   ["#0b1026", "#4a2b7c", "#c24173", "#f59e0b"])
CMAP_DIST      = cmap_from("TurboSoft",   ["#0b1026", "#1f77b4", "#2ecc71", "#f59e0b", "#e11d48"])

def read_bim(bim_path: Path):
    snps = []
    pos = {}
    with open(bim_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            chrom, snp, cm, bp, a1, a2 = line.split()
            bp = int(bp)
            snps.append(snp)
            pos[snp] = bp
    snps_sorted = sorted(snps, key=lambda s: (pos[s], s))
    return snps_sorted, pos

def read_plink_ld(ld_gz: Path):
    rows = []
    with gzip.open(ld_gz, "rt", encoding="utf-8", errors="replace") as f:
        header = f.readline().strip().split()
        # PLINK --r2 header usually: CHR_A BP_A SNP_A CHR_B BP_B SNP_B R2
        idx = {h:i for i,h in enumerate(header)}
        need = ["SNP_A", "SNP_B", "BP_A", "BP_B", "R2"]
        for n in need:
            if n not in idx:
                raise RuntimeError(f"LD file missing column: {n}. Found: {header}")

        for line in f:
            if not line.strip():
                continue
            p = line.split()
            a = p[idx["SNP_A"]]
            b = p[idx["SNP_B"]]
            bpa = int(float(p[idx["BP_A"]]))
            bpb = int(float(p[idx["BP_B"]]))
            r2 = float(p[idx["R2"]])
            rows.append((a, b, bpa, bpb, r2))
    return rows

def read_frq(frq_path: Path):
    maf = {}
    if frq_path is None:
        return maf
    with open(frq_path, "r", encoding="utf-8", errors="replace") as f:
        header = f.readline().strip().split()
        idx = {h:i for i,h in enumerate(header)}
        if "SNP" not in idx or "MAF" not in idx:
            return maf
        for line in f:
            if not line.strip():
                continue
            p = line.split()
            maf[p[idx["SNP"]]] = float(p[idx["MAF"]])
    return maf

def build_matrix(snps, ld_rows):
    n = len(snps)
    ix = {s:i for i,s in enumerate(snps)}
    M = np.full((n,n), np.nan, dtype=float)
    np.fill_diagonal(M, 1.0)
    for a,b,_,_,r2 in ld_rows:
        if a not in ix or b not in ix:
            continue
        i, j = ix[a], ix[b]
        if i == j:
            continue
        # If duplicates appear, keep the strongest
        cur = M[i,j]
        if np.isnan(cur) or r2 > cur:
            M[i,j] = r2
            M[j,i] = r2
    # any remaining NaNs (missing pairs) -> 0
    M = np.nan_to_num(M, nan=0.0)
    return M

def save_ld_table(out_tsv: Path, ld_rows, pos_map):
    # Expand rows and add distance + category
    out = []
    for a,b,bpa,bpb,r2 in ld_rows:
        if a not in pos_map or b not in pos_map:
            continue
        dist = abs(pos_map[a] - pos_map[b])
        if r2 >= 0.8:
            cat = "High (≥0.8)"
        elif r2 >= 0.2:
            cat = "Moderate (0.2–0.79)"
        else:
            cat = "Low (<0.2)"
        out.append((a, pos_map[a], b, pos_map[b], dist, r2, cat))

    out.sort(key=lambda x: (-x[5], x[4], x[0], x[2]))
    with open(out_tsv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["SNP_A","POS_A","SNP_B","POS_B","DIST_BP","R2","LD_CATEGORY"])
        w.writerows(out)

def figsave(fig, out_prefix: Path, stem: str):
    for ext in ["png","pdf","svg"]:
        p = out_prefix.parent / f"{out_prefix.name}_{stem}.{ext}"
        if ext == "png":
            fig.savefig(p, dpi=600, bbox_inches="tight", facecolor="white")
        else:
            fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)

def plot_ld_heatmap(snps, M, title, out_prefix: Path):
    n = len(snps)
    # Mask upper triangle for clean journal layout
    mask = np.triu(np.ones_like(M, dtype=bool), k=1)
    Mm = np.ma.array(M, mask=mask)
    cmap = CMAP_LD_MATRIX.copy()
    cmap.set_bad(color="white")

    fs = max(7.0, 0.65*n)
    fig, ax = plt.subplots(figsize=(fs, fs), constrained_layout=True)
    im = ax.imshow(Mm, vmin=0, vmax=1, cmap=cmap, interpolation="nearest")

    ax.set_title(title, fontsize=18, pad=14)
    ax.set_xlabel("Variants", fontsize=14)
    ax.set_ylabel("Variants", fontsize=14)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(snps, rotation=45, ha="right", fontsize=11)
    ax.set_yticklabels(snps, fontsize=11)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("LD ($r^2$)", fontsize=12)

    figsave(fig, out_prefix, "LD_matrix_ultra")

def plot_maf_lollipop(snps, maf, title, out_prefix: Path):
    # horizontal lollipop = zero overlap, very journal-friendly
    vals = [(s, maf.get(s, np.nan)) for s in snps]
    vals = [(s,v) for s,v in vals if not np.isnan(v)]
    if not vals:
        return

    # sort by genomic order already
    labels = [s for s,_ in vals]
    y = np.arange(len(vals))[::-1]
    x = np.array([v for _,v in vals])

    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.55*len(vals))), constrained_layout=True)

    norm = mpl.colors.Normalize(vmin=0, vmax=max(0.5, float(np.nanmax(x))))
    colors = [CMAP_MAF(norm(v)) for v in x]

    ax.hlines(y=y, xmin=0, xmax=x, linewidth=4, alpha=0.22)
    ax.scatter(x, y, s=220, c=colors, edgecolor="white", linewidth=1.2, zorder=3)

    for xi, yi in zip(x, y):
        ax.text(min(0.52, xi + 0.02), yi, f"{xi:.3f}", va="center", fontsize=11)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlim(0, 0.52)
    ax.set_xlabel("Minor allele frequency (MAF)", fontsize=13)
    ax.set_title(title, fontsize=18, pad=12)
    ax.grid(axis="x", alpha=0.15)

    sm = mpl.cm.ScalarMappable(cmap=CMAP_MAF, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("MAF", fontsize=12)

    figsave(fig, out_prefix, "MAF_lollipop_ultra")

def plot_ld_arcs(snps, ld_rows, pos_map, title, out_prefix: Path, r2_thr=0.8):
    # Arc diagram (looks premium + avoids label overlap)
    pairs = []
    for a,b,_,_,r2 in ld_rows:
        if a in pos_map and b in pos_map and a != b and r2 >= r2_thr:
            pairs.append((a,b,r2))

    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)

    xs = np.array([pos_map[s] for s in snps], dtype=float)
    span = float(xs.max() - xs.min()) if xs.max() != xs.min() else 1.0

    # baseline
    ax.hlines(0, xs.min(), xs.max(), linewidth=2, alpha=0.25)
    ax.scatter(xs, np.zeros_like(xs), s=260, edgecolor="white", linewidth=1.2, zorder=5)

    # labels: stagger to prevent collisions
    for i, s in enumerate(snps):
        dy = (0.07 if i % 2 == 0 else 0.12) * span
        ax.text(pos_map[s], dy, s, rotation=35, ha="right", va="bottom", fontsize=11)

    norm = mpl.colors.Normalize(vmin=r2_thr, vmax=1.0)

    for a,b,r2 in pairs:
        x0, x1 = pos_map[a], pos_map[b]
        if x0 > x1:
            x0, x1 = x1, x0
        dx = x1 - x0
        mid = (x0 + x1) / 2.0
        height = min(0.45*span, 0.10*span + 0.30*dx)  # scales nicely with distance
        color = CMAP_ARCS(norm(r2))
        lw = 1.2 + 3.0*r2

        verts = [(x0, 0), (mid, height), (x1, 0)]
        codes = [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3]
        patch = PathPatch(MplPath(verts, codes), facecolor="none", edgecolor=color, lw=lw, alpha=0.9)
        ax.add_patch(patch)

    ax.set_title(title, fontsize=18, pad=12)
    ax.set_xlabel("Genomic position (bp, GRCh38)", fontsize=13)
    ax.yaxis.set_visible(False)
    ax.grid(axis="x", alpha=0.12)

    # Force plain bp formatting (avoid 1e7 scientific notation)
    ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax.ticklabel_format(style="plain", axis="x")

    # Colorbar for r2
    sm = mpl.cm.ScalarMappable(cmap=CMAP_ARCS, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label(f"LD ($r^2$) for edges ≥ {r2_thr}", fontsize=12)

    ax.set_ylim(-0.08*span, 0.60*span)

    figsave(fig, out_prefix, f"LD_arcs_r2ge{r2_thr}_ultra")

def plot_ld_decay(ld_rows, title, out_prefix: Path):
    dist = []
    r2s = []
    for a,b,bpa,bpb,r2 in ld_rows:
        if a == b:
            continue
        dist.append(abs(int(bpa) - int(bpb)))
        r2s.append(r2)
    if not dist:
        return
    dist = np.array(dist, dtype=float)
    r2s = np.array(r2s, dtype=float)

    fig, ax = plt.subplots(figsize=(9.5, 5.5), constrained_layout=True)

    # Scatter colored by distance (different palette from matrix)
    norm = mpl.colors.Normalize(vmin=float(dist.min()), vmax=float(dist.max() if dist.max()>dist.min() else dist.min()+1))
    colors = CMAP_DIST(norm(dist))
    ax.scatter(dist, r2s, s=90, c=colors, edgecolor="white", linewidth=0.8, alpha=0.92, zorder=3)

    # Add binned median trend (clean + journal-like)
    nb = min(10, max(4, int(len(dist) / 4)))
    bins = np.linspace(dist.min(), dist.max(), nb+1)
    mids = 0.5*(bins[:-1] + bins[1:])
    med = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = r2s[(dist >= lo) & (dist < hi)]
        med.append(np.nanmedian(m) if len(m) else np.nan)
    med = np.array(med, dtype=float)
    ok = ~np.isnan(med)
    ax.plot(mids[ok], med[ok], linewidth=3, alpha=0.85)

    ax.set_title(title, fontsize=18, pad=12)
    ax.set_xlabel("Distance (bp)", fontsize=13)
    ax.set_ylabel("LD ($r^2$)", fontsize=13)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.12)

    sm = mpl.cm.ScalarMappable(cmap=CMAP_DIST, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Distance (bp)", fontsize=12)

    figsave(fig, out_prefix, "LD_decay_ultra")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ld", required=True, help="PLINK --r2 gz output (e.g., MAG_ld.ld.gz)")
    ap.add_argument("--bim", required=True, help="BIM file (e.g., MAG_rs.bim)")
    ap.add_argument("--frq", default=None, help="PLINK --freq output (e.g., MAG_maf.frq)")
    ap.add_argument("--out-prefix", required=True, help="Output prefix (folder/name)")
    ap.add_argument("--locus", default="MAG locus", help="Locus name for titles")
    ap.add_argument("--panel", default="1000G High Coverage (GRCh38)", help="Panel label for titles")
    ap.add_argument("--r2-threshold", type=float, default=0.8, help="High-LD threshold for arcs")
    args = ap.parse_args()

    ld_path = Path(args.ld)
    bim_path = Path(args.bim)
    frq_path = Path(args.frq) if args.frq else None
    out_prefix = Path(args.out_prefix)

    # Fonts (Windows-friendly)
    mpl.rcParams["font.family"] = "DejaVu Sans"
    mpl.rcParams["axes.titleweight"] = "semibold"

    snps, pos_map = read_bim(bim_path)
    ld_rows = read_plink_ld(ld_path)
    maf = read_frq(frq_path)

    M = build_matrix(snps, ld_rows)

    # Tables
    save_ld_table(out_prefix.parent / f"{out_prefix.name}_LD_pairs.tsv", ld_rows, pos_map)

    # Plots
    plot_ld_heatmap(
        snps, M,
        title=f"{args.locus} — Linkage disequilibrium (r²)\n{args.panel}",
        out_prefix=out_prefix
    )
    plot_maf_lollipop(
        snps, maf,
        title=f"{args.locus} — Minor allele frequency\n{args.panel}",
        out_prefix=out_prefix
    )
    plot_ld_arcs(
        snps, ld_rows, pos_map,
        title=f"{args.locus} — High-LD arc map (r² ≥ {args.r2_threshold})\n{args.panel}",
        out_prefix=out_prefix,
        r2_thr=args.r2_threshold
    )
    plot_ld_decay(
        ld_rows,
        title=f"{args.locus} — LD decay (pairwise)\n{args.panel}",
        out_prefix=out_prefix
    )

    print("[OK] Wrote plots: PNG/PDF/SVG + LD table TSV")

if __name__ == "__main__":
    main()
