#!/usr/bin/env python3
from __future__ import annotations

import os
import math
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap

# -----------------------------
# Premium (light) palettes
# -----------------------------
def _cm(colors, name):
    return LinearSegmentedColormap.from_list(name, colors, N=256)

CM_TREEMAP = _cm(["#f8fafc", "#e0f2fe", "#dbeafe", "#a7f3d0", "#93c5fd"], "treemap_light")
CM_LINE_A  = "#2563eb"  # reads
CM_LINE_B  = "#06b6d4"  # contigs
CM_EQ      = "#94a3b8"  # equality line in Lorenz

# Dot-table row colors
ROW_COL = {
    "Total length (Mb)": "#2563eb",
    "# contigs": "#8b5cf6",
    "N50 (kb)": "#10b981",
    "GC (%)": "#f59e0b",
}

# -----------------------------
# Layout helpers
# -----------------------------
def panel_tag(ax, letter: str):
    ax.text(
        0.01, 0.98, letter,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=11, fontweight="bold", color="white",
        bbox=dict(boxstyle="round,pad=0.25", fc="#0f172a", ec="none", alpha=0.95),
        zorder=50,
    )

def clean_name(s: str) -> str:
    return (s or "").strip().replace("_", " ")

def shorten(s: str, n: int = 18) -> str:
    s = clean_name(s)
    return s if len(s) <= n else (s[: n - 1] + "…")

# -----------------------------
# Kraken2 report parsing
# -----------------------------
@dataclass
class KrakenRow:
    pct: float
    clade: int
    direct: int
    rank: str
    taxid: str
    name: str

def read_kraken_report(path: Path) -> List[KrakenRow]:
    rows: List[KrakenRow] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            try:
                pct = float(parts[0].strip())
                clade = int(parts[1].strip())
                direct = int(parts[2].strip())
            except Exception:
                continue
            rank = parts[3].strip()
            taxid = parts[4].strip()
            name = parts[5].strip()
            rows.append(KrakenRow(pct, clade, direct, rank, taxid, name))
    return rows

def kraken_genus_blocks(rows: List[KrakenRow], top_n: int = 10) -> List[Tuple[str, float, str]]:
    """
    Return treemap blocks: (label, percent, kind)
    kind in {"GENUS","OTHER_GENUS","HIGHER","UNCLASS"}
    """
    if not rows:
        return [("No Kraken2 report", 100.0, "HIGHER")]

    unclass = 0.0
    genus: Dict[str, float] = {}

    for r in rows:
        nm = clean_name(r.name)
        if r.rank == "U":
            unclass = max(unclass, r.pct)
        elif r.rank == "G":
            # Kraken percent is clade-percent; genus clades are disjoint for reads/contigs assignments
            genus[nm] = genus.get(nm, 0.0) + r.pct

    genus_items = sorted(genus.items(), key=lambda x: x[1], reverse=True)
    genus_total = sum(v for _, v in genus_items)
    top = genus_items[:top_n]
    top_sum = sum(v for _, v in top)

    other_genus = max(0.0, genus_total - top_sum)
    higher = max(0.0, 100.0 - unclass - genus_total)

    blocks: List[Tuple[str, float, str]] = []
    for k, v in top:
        blocks.append((k, v, "GENUS"))
    if other_genus > 0.05:
        blocks.append(("Other genera", other_genus, "OTHER_GENUS"))
    if higher > 0.05:
        blocks.append(("Higher-rank classified", higher, "HIGHER"))
    if unclass > 0.05:
        blocks.append(("Unclassified", unclass, "UNCLASS"))

    # Numerical safety: renormalize to 100 if tiny rounding drift
    s = sum(v for _, v, _ in blocks)
    if s > 0:
        blocks = [(lab, v * 100.0 / s, kind) for lab, v, kind in blocks]
    return blocks

def kraken_rank_depth(rows: List[KrakenRow]) -> Dict[str, float]:
    """
    Depth curve approximation:
    fraction reaching at least rank R = 1 - (unclassified + sum direct-at-ranks above R)/total
    Uses 'direct' counts per rank code from Kraken report.
    """
    if not rows:
        return {k: 0.0 for k in ["D","P","C","O","F","G","S","U"]}

    direct_by_rank: Dict[str, int] = {}
    total_direct = 0
    unclass_direct = 0
    for r in rows:
        direct_by_rank[r.rank] = direct_by_rank.get(r.rank, 0) + int(r.direct)
        total_direct += int(r.direct)
        if r.rank == "U":
            unclass_direct += int(r.direct)

    if total_direct <= 0:
        return {k: 0.0 for k in ["D","P","C","O","F","G","S","U"]}

    rank_order = ["D","P","C","O","F","G","S"]
    above_sum = 0
    depth: Dict[str, float] = {}
    # D
    depth["D"] = 100.0 * (1.0 - (unclass_direct / total_direct))
    # deeper ranks
    for i, rk in enumerate(rank_order[1:], start=1):
        above_sum = sum(direct_by_rank.get(r, 0) for r in rank_order[:i])  # ranks above desired
        depth[rk] = 100.0 * (1.0 - ((unclass_direct + above_sum) / total_direct))

    # U as a point (unclassified %)
    depth["U"] = 100.0 * (unclass_direct / total_direct)
    return depth

def diversity_top_genera(rows: List[KrakenRow], top_n: int = 20) -> Tuple[float, float]:
    """Shannon H and Simpson (1-D) among genus-assigned reads/contigs (conditional on genus)."""
    genus: Dict[str, float] = {}
    for r in rows:
        if r.rank == "G":
            genus[clean_name(r.name)] = genus.get(clean_name(r.name), 0.0) + r.pct
    items = sorted(genus.items(), key=lambda x: x[1], reverse=True)[:top_n]
    vals = np.array([v for _, v in items], dtype=float)
    if vals.size == 0 or vals.sum() <= 0:
        return 0.0, 0.0
    p = vals / vals.sum()
    H = float(-(p * np.log(p + 1e-12)).sum())
    simpson_1mD = float(1.0 - (p * p).sum())
    return H, simpson_1mD

# -----------------------------
# Tiny treemap (squarify-style)
# -----------------------------
def _normalize_sizes(sizes, dx, dy):
    total = sum(sizes)
    if total <= 0:
        return [0 for _ in sizes]
    factor = dx * dy / total
    return [s * factor for s in sizes]

def _worst_ratio(row, w):
    if not row:
        return float("inf")
    s = sum(row)
    m = max(row)
    n = min(row)
    if n <= 0:
        return float("inf")
    return max((w * w * m) / (s * s), (s * s) / (w * w * n))

def _layout_row(row, x, y, dx, dy):
    rects = []
    s = sum(row)
    if dx >= dy:
        # horizontal split
        h = s / dx if dx > 0 else 0
        cx = x
        for r in row:
            w = r / h if h > 0 else 0
            rects.append((cx, y, w, h))
            cx += w
        return rects, x, y + h, dx, dy - h
    else:
        # vertical split
        w = s / dy if dy > 0 else 0
        cy = y
        for r in row:
            h = r / w if w > 0 else 0
            rects.append((x, cy, w, h))
            cy += h
        return rects, x + w, y, dx - w, dy

def treemap_rects(values, x, y, dx, dy):
    sizes = _normalize_sizes(values, dx, dy)
    rects = []
    row = []
    w = min(dx, dy)
    sizes = [s for s in sizes if s > 0]

    while sizes:
        c = sizes[0]
        if not row or _worst_ratio(row + [c], w) <= _worst_ratio(row, w):
            row.append(c)
            sizes.pop(0)
        else:
            r, x, y, dx, dy = _layout_row(row, x, y, dx, dy)
            rects.extend(r)
            w = min(dx, dy)
            row = []
    if row:
        r, x, y, dx, dy = _layout_row(row, x, y, dx, dy)
        rects.extend(r)
    return rects

def draw_treemap(ax, blocks: List[Tuple[str, float, str]], title: str):
    ax.set_axis_off()
    ax.set_title(title, fontsize=11, pad=8)

    labels = [b[0] for b in blocks]
    vals = [max(0.0, float(b[1])) for b in blocks]
    kinds = [b[2] for b in blocks]

    rects = treemap_rects(vals, 0, 0, 1, 1)

    # map genus blocks to colors by value (light gradient)
    vmax = max(vals) if vals else 1.0
    def color_for(v, kind):
        if kind == "UNCLASS":
            return "#e2e8f0"
        if kind == "HIGHER":
            return "#cbd5e1"
        if kind == "OTHER_GENUS":
            return "#bfdbfe"
        # GENUS:
        t = (v / vmax) ** 0.6 if vmax > 0 else 0.0
        return CM_TREEMAP(t)

    for (x, y, w, h), lab, v, kind in zip(rects, labels, vals, kinds):
        fc = color_for(v, kind)
        ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor="white", linewidth=1.2))
        area = w * h
        # label only if it fits
        if area > 0.08:
            txt = f"{shorten(lab, 20)}\n{v:.1f}%"
            ax.text(
                x + w - 0.015, y + h - 0.02, txt,
                ha="right", va="top",
                fontsize=9, color="#0f172a",
                bbox=dict(boxstyle="round,pad=0.25", fc=(1,1,1,0.75), ec="none"),
            )

    # little legend note (subtle)
    ax.text(
        0.99, 0.01,
        "Light gradient: higher share → deeper tint",
        ha="right", va="bottom", fontsize=8, color="#64748b"
    )

# -----------------------------
# FASTA metrics
# -----------------------------
def fasta_lengths_gc(path: Path, min_len: int = 0) -> Tuple[np.ndarray, float]:
    """Return lengths array and GC%."""
    if not path.exists():
        return np.array([], dtype=int), float("nan")

    lengths = []
    gc = 0
    total = 0

    seqlen = 0
    seqgc = 0
    for line in path.open("r", encoding="utf-8", errors="ignore"):
        if not line:
            continue
        if line.startswith(">"):
            if seqlen > 0:
                if seqlen >= min_len:
                    lengths.append(seqlen)
                    gc += seqgc
                    total += seqlen
            seqlen = 0
            seqgc = 0
            continue
        s = line.strip().upper()
        if not s:
            continue
        seqlen += len(s)
        seqgc += s.count("G") + s.count("C")

    if seqlen > 0 and seqlen >= min_len:
        lengths.append(seqlen)
        gc += seqgc
        total += seqlen

    if total <= 0:
        return np.array([], dtype=int), float("nan")
    return np.array(lengths, dtype=int), 100.0 * (gc / total)

def n50(lengths: np.ndarray) -> int:
    if lengths.size == 0:
        return 0
    L = np.sort(lengths)[::-1]
    s = L.sum()
    c = np.cumsum(L)
    idx = np.searchsorted(c, 0.5 * s)
    return int(L[min(idx, L.size - 1)])

def compute_metrics(lengths: np.ndarray, gc_pct: float) -> Dict[str, float]:
    return {
        "Total length (Mb)": float(lengths.sum() / 1e6) if lengths.size else 0.0,
        "# contigs": float(lengths.size),
        "N50 (kb)": float(n50(lengths) / 1e3) if lengths.size else 0.0,
        "GC (%)": float(gc_pct) if not math.isnan(gc_pct) else 0.0,
    }

# -----------------------------
# Lorenz + Gini
# -----------------------------
def lorenz_gini(lengths: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    if lengths.size == 0:
        return np.array([0,1]), np.array([0,1]), float("nan")
    L = np.sort(lengths.astype(float))
    cum_len = np.cumsum(L)
    frac_len = np.insert(cum_len / cum_len[-1], 0, 0.0)
    frac_n = np.insert(np.arange(1, L.size + 1) / L.size, 0, 0.0)
    area = float(np.trapezoid(frac_len, frac_n))
    gini = 1.0 - 2.0 * area
    return frac_n, frac_len, gini

# -----------------------------
# Dot-table panel D (no overlaps)
# -----------------------------
def draw_metric_table(ax, metrics_by_filter: Dict[str, Dict[str, float]]):
    ax.set_axis_off()
    ax.set_title("D. Assembly metrics across filters (computed)", fontsize=11, pad=8)

    filters = list(metrics_by_filter.keys())
    rows = ["Total length (Mb)", "# contigs", "N50 (kb)", "GC (%)"]

    # geometry in axes coords
    left, bottom, width, height = 0.02, 0.08, 0.96, 0.80
    ncols = 1 + len(filters)
    nrows = 1 + len(rows)
    cw = width / ncols
    rh = height / nrows

    # header row background
    ax.add_patch(Rectangle((left, bottom + height - rh), width, rh, facecolor="#f1f5f9", edgecolor="none"))

    # grid + headers
    for c in range(ncols):
        for r in range(nrows):
            x = left + c * cw
            y = bottom + (nrows - 1 - r) * rh
            ax.add_patch(Rectangle((x, y), cw, rh, facecolor="none", edgecolor="#e2e8f0", linewidth=1.0))

    ax.text(left + 0.5 * cw, bottom + height - 0.5 * rh, "Metric", ha="center", va="center",
            fontsize=10, fontweight="bold", color="#0f172a")

    for j, f in enumerate(filters):
        ax.text(left + (j + 1.5) * cw, bottom + height - 0.5 * rh, f, ha="center", va="center",
                fontsize=10, fontweight="bold", color="#0f172a")

    # normalization per row (for dot size)
    for i, m in enumerate(rows):
        vals = np.array([metrics_by_filter[f][m] for f in filters], dtype=float)
        vmin, vmax = float(vals.min()), float(vals.max())
        if vmax <= vmin:
            vmax = vmin + 1.0

        # row label
        ax.text(left + 0.5 * cw, bottom + height - (i + 1.5) * rh, m, ha="center", va="center",
                fontsize=9, color="#0f172a")

        for j, f in enumerate(filters):
            v = metrics_by_filter[f][m]
            # cell origin
            x0 = left + (j + 1) * cw
            y0 = bottom + height - (i + 2) * rh

            # dot size (premium, but bounded)
            t = (v - vmin) / (vmax - vmin)
            s = 90 + 260 * (t ** 0.85)
            ax.scatter([x0 + 0.26 * cw], [y0 + 0.5 * rh], s=s,
                       color=ROW_COL[m], edgecolors="#0f172a", linewidths=0.6, zorder=5)

            # value text (never overlaps because it's per-cell)
            if m == "# contigs":
                txt = f"{int(round(v)):,d}"
            elif m == "Total length (Mb)":
                txt = f"{v:.2f}"
            elif m == "N50 (kb)":
                txt = f"{v:.2f}"
            else:
                txt = f"{v:.2f}"
            ax.text(x0 + 0.55 * cw, y0 + 0.5 * rh, txt, ha="left", va="center",
                    fontsize=9, fontweight="bold", color="#0f172a")

    ax.text(0.02, -0.06, "Dot size scaled within each row.", transform=ax.transAxes,
            fontsize=8, color="#64748b", ha="left", va="top", clip_on=False)

# -----------------------------
# Main figure
# -----------------------------
def main():
    I9 = Path(os.environ.get("I9", "/mnt/e/Bacillus WGS/bacillus_paranthracis/06_reports_wsl/I-9"))
    OUTDIR = Path(os.environ.get("OUTDIR", str(I9 / "figures" / "_paper_mainfigs" / "fig_5")))
    OUTDIR.mkdir(parents=True, exist_ok=True)

    kr_reads = I9 / "I-9.reads.kraken2.report.txt"
    kr_cont  = I9 / "I-9.contigs.kraken2.report.txt"

    # FASTA paths
    all_fa = I9 / "I-9.final.contigs.fa"
    bac_fa_candidates = [
        I9 / "bacillus_only" / "I-9.bacillus_only.fa",
        I9 / "qc_clean" / "I-9.bacillus_only.fa",
        I9 / "qc_clean" / "I-9.bacillus_only.fa",
    ]
    bac_fa = next((p for p in bac_fa_candidates if p.exists()), None)

    # read Kraken
    rows_reads = read_kraken_report(kr_reads)
    rows_cont  = read_kraken_report(kr_cont)

    blocks_reads = kraken_genus_blocks(rows_reads, top_n=10)
    blocks_cont  = kraken_genus_blocks(rows_cont,  top_n=10)

    depth_reads = kraken_rank_depth(rows_reads)
    depth_cont  = kraken_rank_depth(rows_cont)

    H_r, S_r = diversity_top_genera(rows_reads, top_n=20)
    H_c, S_c = diversity_top_genera(rows_cont,  top_n=20)

    # assembly metrics (All / >=5kb / Bac-only)
    lens_all, gc_all = fasta_lengths_gc(all_fa, min_len=0)
    lens_5k,  gc_5k  = fasta_lengths_gc(all_fa, min_len=5000)
    if bac_fa is not None:
        lens_bac, gc_bac = fasta_lengths_gc(bac_fa, min_len=0)
    else:
        lens_bac, gc_bac = np.array([], dtype=int), float("nan")

    met_all = compute_metrics(lens_all, gc_all)
    met_5k  = compute_metrics(lens_5k,  gc_5k)
    met_bac = compute_metrics(lens_bac, gc_bac)

    # Lorenz/Gini
    xA, yA, gA = lorenz_gini(lens_all)
    x5, y5, g5 = lorenz_gini(lens_5k)
    xB, yB, gB = lorenz_gini(lens_bac)

    # -----------------------------
    # Figure layout
    # -----------------------------
    fig = plt.figure(figsize=(16, 9), dpi=220)
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1.05, 1.05, 1.15], hspace=0.55, wspace=0.22)

    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])
    axE = fig.add_subplot(gs[2, :])

    fig.suptitle(
        "Main Figure 5 — Taxonomy depth + assembly robustness (A–E)\n"
        "Kraken2 treemaps + rank-depth/diversity + cross-filter assembly metrics + Lorenz/Gini fragmentation",
        fontsize=16, fontweight="bold", y=0.98
    )

    # A/B Treemaps (light gradient)
    draw_treemap(axA, blocks_reads, "A. Reads taxonomy treemap (genus; Kraken2)")
    panel_tag(axA, "A")
    draw_treemap(axB, blocks_cont, "B. Contigs taxonomy treemap (genus; Kraken2)")
    panel_tag(axB, "B")

    # C Rank-depth + diversity
    panel_tag(axC, "C")
    axC.set_title("C. Rank-depth signal + diversity (reads vs contigs)", fontsize=11, pad=8)
    ranks = ["D","P","C","O","F","G","S","U"]
    xs = np.arange(len(ranks))

    y_reads = [depth_reads.get(r, 0.0) for r in ranks]
    y_cont  = [depth_cont.get(r, 0.0) for r in ranks]

    axC.plot(xs, y_reads, marker="o", linewidth=2.6, markersize=5.5, color=CM_LINE_A, label="Reads")
    axC.plot(xs, y_cont,  marker="o", linewidth=2.6, markersize=5.5, color=CM_LINE_B, label="Contigs")

    axC.set_xticks(xs)
    axC.set_xticklabels(ranks)
    axC.set_ylabel("Percent (%)")
    axC.set_ylim(0, 102)
    axC.grid(True, axis="y", alpha=0.25)
    axC.legend(frameon=True, fontsize=9, loc="upper right")

    # Diversity footer
    un_r = depth_reads.get("U", 0.0)
    un_c = depth_cont.get("U", 0.0)
    axC.text(
        0.01, 0.015,
        f"Genus diversity (top20, conditional on genus): Reads H={H_r:.2f}, 1−D={S_r:.2f}\n"
        f"Contigs H={H_c:.2f}, 1−D={S_c:.2f}\n"
        f"Unclassified: Reads {un_r:.1f}% | Contigs {un_c:.1f}%",
        transform=axC.transAxes, ha="left", va="bottom", fontsize=8.3, color="#0f172a",
        bbox=dict(boxstyle="round,pad=0.22", fc=(1,1,1,0.72), ec="none")
    )

    # D Metric dot-table (no overlaps by design)
    panel_tag(axD, "D")
    draw_metric_table(axD, {
        "All": met_all,
        "≥5 kb": met_5k,
        "Bac-only": met_bac,
    })

    # E Lorenz + Gini (legend top-left slightly right of tag)
    panel_tag(axE, "E")
    axE.set_title("E. Assembly fragmentation (Lorenz curves) + Gini index", fontsize=11, pad=8)
    axE.plot([0,1], [0,1], linestyle="--", linewidth=1.5, color=CM_EQ)

    axE.plot(xA, yA, linewidth=2.8, color="#2563eb", label=f"All contigs (Gini={gA:.2f})")
    axE.plot(x5, y5, linewidth=2.8, color="#06b6d4", label=f"≥5 kb contigs (Gini={g5:.2f})")
    axE.plot(xB, yB, linewidth=2.8, color="#22c55e", label=f"Bacillus-only (Gini={gB:.2f})")

    axE.set_xlabel("Cumulative fraction of contigs (sorted by length)")
    axE.set_ylabel("Cumulative fraction of assembly length")
    axE.set_xlim(-0.02, 1.02)
    axE.set_ylim(-0.02, 1.02)
    axE.grid(True, alpha=0.22)

    axE.legend(
        loc="upper left",
        bbox_to_anchor=(0.08, 0.98),  # slightly right of panel tag
        frameon=True,
        fontsize=9
    )

    # save
    base = OUTDIR / "MainFig05_Taxonomy_Assembly_QC"
    fig.savefig(str(base) + ".pdf")
    fig.savefig(str(base) + ".svg")
    fig.savefig(str(base) + ".png", dpi=600)
    fig.savefig(str(base) + ".eps")
    fig.savefig(str(base) + ".jpg", dpi=450)
    plt.close(fig)

    # summary TSV (light, but useful)
    summary = OUTDIR / "MainFig05_Taxonomy_Assembly_QC.summary.tsv"
    with summary.open("w", encoding="utf-8") as w:
        w.write("panel\tkey\tvalue\n")
        w.write(f"A\treads_top_genus\t{blocks_reads[0][0] if blocks_reads else 'NA'}\n")
        w.write(f"B\tcontigs_top_genus\t{blocks_cont[0][0] if blocks_cont else 'NA'}\n")
        w.write(f"C\treads_unclassified_pct\t{un_r:.3f}\n")
        w.write(f"C\tcontigs_unclassified_pct\t{un_c:.3f}\n")
        w.write(f"D\tall_total_mb\t{met_all['Total length (Mb)']:.6f}\n")
        w.write(f"D\tall_contigs\t{met_all['# contigs']:.0f}\n")
        w.write(f"E\tgini_all\t{gA:.6f}\n")

    print(f"Wrote: {base}.pdf/.svg/.png/.jpg/.eps")
    print(f"Wrote: {summary}")

if __name__ == "__main__":
    main()
