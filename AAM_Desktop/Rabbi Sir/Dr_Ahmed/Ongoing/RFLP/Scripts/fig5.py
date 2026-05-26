#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 5 — Enzyme efficiency + reuse network + coverage (A–F)

Keeping everything the same, ONLY updating the header so that:
  - background is LIGHT gradient
  - header wording is premium semi-dark blue
  - NO header ticks/scales/spines (no 0..1 axis marks)

Also restores the missing axis placements (axA/axB/axC/axD/axE/axF) so the script runs.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

PAL = {
    "ink": (0.07, 0.09, 0.12),
    "slate": (0.38, 0.42, 0.50),
    "grid": (0.86, 0.88, 0.92),
    "paper": (1.00, 1.00, 1.00),
    "panel_bg": (0.98, 0.985, 0.995),
    "spine": (0.75, 0.78, 0.85),

    # HEADER (light gradient + premium blue text)
    "header_l1": (0.90, 0.96, 1.00),
    "header_l2": (0.985, 0.995, 1.00),
    "header_text": (0.07, 0.22, 0.44),  # premium semi-dark blue

    # Panel A palette (ocean)
    "a_fill": (0.18, 0.78, 0.86),
    "a_edge": (0.07, 0.21, 0.33),

    # Panel B palette (emerald + lilac)
    "b_enzyme": (0.12, 0.66, 0.47),
    "b_link": (0.55, 0.60, 0.70),

    # Gene colors (soft premium)
    "S": (0.56, 0.46, 0.90),
    "ORF1ab": (0.15, 0.73, 0.77),
    "N": (0.95, 0.53, 0.35),

    # Effects
    "gain": (0.09, 0.66, 0.55),
    "loss": (0.91, 0.38, 0.33),
    "ambig": (0.91, 0.73, 0.18),
    "unknown": (0.74, 0.78, 0.86),

    # Optimization + QC
    "gold": (0.88, 0.67, 0.12),
    "qc_close": (0.88, 0.34, 0.29),
    "qc_tiny": (0.92, 0.72, 0.18),
    "qc_sum": (0.12, 0.18, 0.30),
    "midnight": (0.05, 0.08, 0.14),
}

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def draw_gradient_bar(ax, c1, c2, steps=240):
    """
    HEADER FIX:
    - draw gradient in AXES coordinates (transform=ax.transAxes)
    - hard-disable all axis visuals to prevent 0..1 scale marks
    """
    grad = np.linspace(0, 1, steps)
    rgb = (1 - grad)[:, None] * np.array(c1)[None, :] + grad[:, None] * np.array(c2)[None, :]
    img = np.tile(rgb[None, :, :], (2, 1, 1))

    ax.imshow(
        img,
        extent=[0, 1, 0, 1],
        transform=ax.transAxes,
        aspect="auto",
        interpolation="bicubic",
        zorder=0,
    )

    ax.set_axis_off()
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_frame_on(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

def tidy_axes(ax):
    ax.set_facecolor(PAL["panel_bg"])
    for sp in ax.spines.values():
        sp.set_color(PAL["spine"])
        sp.set_linewidth(0.8)

def panel_label(ax, letter):
    ax.text(0.008, 1.01, letter, transform=ax.transAxes,
            ha="left", va="bottom", fontsize=12, fontweight="bold", color=PAL["ink"])

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
        vals = []
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

def effect_bucket(site_effect):
    if site_effect is None:
        return "unknown"
    s = str(site_effect).strip().lower()
    if s in {"", "na", "nan", "none"}:
        return "unknown"
    has_gain = "gain" in s
    has_loss = "loss" in s
    has_none = "none" in s
    if has_gain and not has_loss and not has_none:
        return "gain"
    if has_loss and not has_gain and not has_none:
        return "loss"
    if has_gain or has_loss:
        return "ambig"
    return "unknown"

def min_band_spacing(bands):
    if bands is None or len(bands) < 2:
        return float("inf")
    b = sorted([int(x) for x in bands], reverse=True)
    diffs = [abs(b[i] - b[i+1]) for i in range(len(b)-1)]
    return float(min(diffs)) if diffs else float("inf")

def safe_int(x):
    try:
        return int(round(float(x)))
    except Exception:
        return None

def greedy_set_cover(universe, sets_by_enzyme):
    uncovered = set(universe)
    chosen, covered = [], set()
    while uncovered:
        best_e, best_gain = None, 0
        for e, s in sets_by_enzyme.items():
            g = len(uncovered & s)
            if g > best_gain:
                best_gain = g
                best_e = e
        if best_e is None or best_gain == 0:
            break
        newly = uncovered & sets_by_enzyme[best_e]
        covered |= newly
        uncovered -= newly
        chosen.append((best_e, len(newly), len(covered)))
    return chosen, uncovered

def spaced(n, top=0.95, bottom=0.06):
    if n <= 1:
        return [0.5]
    return np.linspace(top, bottom, n).tolist()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assay_csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--title", default="Enzyme Efficiency + Reuse Network + Coverage")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--jpg_quality", type=int, default=92)
    ap.add_argument("--qc_tau", type=float, default=10.0)
    ap.add_argument("--top_enzymes", type=int, default=12)
    args = ap.parse_args()

    df = pd.read_csv(args.assay_csv)
    df = df[df["locus"].notna()].copy()
    for c in ["enzyme", "gene", "site_effect"]:
        if c in df.columns:
            df[c] = df[c].astype(str)

    df["enzyme"] = df["enzyme"].replace({"nan": "NA", "NaN": "NA"}).fillna("NA")
    df["gene"]   = df["gene"].replace({"nan": "NA", "NaN": "NA"}).fillna("NA")
    df["locus"]  = df["locus"].astype(str)

    df_assigned = df[df["enzyme"].str.upper().ne("NA")].copy()
    na_count = int((df["enzyme"].str.upper() == "NA").sum())

    enzyme_counts = df_assigned.groupby("enzyme")["locus"].nunique().sort_values(ascending=False)
    enzymes_all = enzyme_counts.index.tolist()
    topE = enzymes_all[:max(1, min(args.top_enzymes, len(enzymes_all)))]

    reuse_counts = enzyme_counts.loc[topE]

    gene_order = ["S", "ORF1ab", "N"]
    cov = (df_assigned
           .assign(gene_norm=df_assigned["gene"].replace({"orf1ab": "ORF1ab", "ORF1AB": "ORF1ab"}))
           .groupby(["enzyme", "gene_norm"])["locus"].nunique()
           .unstack(fill_value=0))
    for g in gene_order:
        if g not in cov.columns:
            cov[g] = 0
    cov = cov[gene_order].reindex(topE)

    df_assigned["effect_bin"] = df_assigned["site_effect"].apply(effect_bucket)
    eff = df_assigned.groupby(["enzyme", "effect_bin"])["locus"].nunique().unstack(fill_value=0).reindex(topE)
    for k in ["gain", "loss", "ambig", "unknown"]:
        if k not in eff.columns:
            eff[k] = 0
    eff = eff[["gain", "loss", "ambig", "unknown"]]

    qc_rows = []
    for _, r in df_assigned.iterrows():
        amp = safe_int(r.get("amplicon_bp", None))
        wt  = first_or_empty(parse_fragments(r.get("wt_fragments_bp", "")))
        mut = first_or_empty(parse_fragments(r.get("mut_fragments_bp", "")))

        too_close = (min(min_band_spacing(wt), min_band_spacing(mut)) <= float(args.qc_tau))
        tiny_frag = (len(wt) > 0 and min(wt) < 50) or (len(mut) > 0 and min(mut) < 50)

        sum_mismatch = False
        if amp is not None:
            if wt and len(wt) > 1 and sum(wt) != amp:
                sum_mismatch = True
            if mut and len(mut) > 1 and sum(mut) != amp:
                sum_mismatch = True

        qc_rows.append({"enzyme": r["enzyme"],
                        "too_close": int(bool(too_close)),
                        "tiny_frag": int(bool(tiny_frag)),
                        "sum_mismatch": int(bool(sum_mismatch))})
    qc = pd.DataFrame(qc_rows)
    qc_agg = qc.groupby("enzyme")[["too_close", "tiny_frag", "sum_mismatch"]].sum().reindex(topE).fillna(0).astype(int)

    sets_by_enzyme = {e: set(df_assigned[df_assigned["enzyme"] == e]["locus"].astype(str).tolist()) for e in enzymes_all}
    universe = set(df_assigned["locus"].astype(str).tolist())
    chosen, uncovered = greedy_set_cover(universe, sets_by_enzyme)

    loci_gene = df_assigned.set_index("locus")["gene"].to_dict()
    loci = sorted(universe, key=lambda L: (loci_gene.get(L, "Z"), L))

    xE, xL = 0.07, 0.93
    y_enz  = spaced(len(enzymes_all))
    y_loci = spaced(len(loci))
    pos_enzyme = {e: (xE, y_enz[i]) for i, e in enumerate(enzymes_all)}
    pos_locus  = {L: (xL, y_loci[i]) for i, L in enumerate(loci)}
    edges = [(r["enzyme"], r["locus"]) for _, r in df_assigned.iterrows()]

    fig = plt.figure(figsize=(18, 12), dpi=170)
    fig.patch.set_facecolor(PAL["paper"])
    gs = GridSpec(44, 32, figure=fig, wspace=1.25, hspace=1.45)

    # ===== HEADER (fixed) =====
    ax_head = fig.add_subplot(gs[0:4, :])
    draw_gradient_bar(ax_head, PAL["header_l1"], PAL["header_l2"])
    ax_head.text(0.5, 0.62, args.title,
                 ha="center", va="center", fontsize=18, fontweight="bold",
                 color=PAL["header_text"], transform=ax_head.transAxes)
    ax_head.text(0.5, 0.30, "Enzyme-level strategy (reuse • coverage • optimization • QC)",
                 ha="center", va="center", fontsize=10.5,
                 color=PAL["slate"], transform=ax_head.transAxes)

    # Axis placements (were missing in your paste)
    axA = fig.add_subplot(gs[5:15, 0:12])
    axC = fig.add_subplot(gs[5:15, 18:30])
    axB = fig.add_subplot(gs[18:36, 0:16])
    axD = fig.add_subplot(gs[18:27, 18:30])
    axE = fig.add_subplot(gs[29:37, 18:30])
    axF = fig.add_subplot(gs[39:44, 18:28])

    # ===== The rest of your panels are unchanged =====
    # A
    tidy_axes(axA); panel_label(axA, "A")
    axA.text(0.5, 1.02, "Enzyme reuse (loci per enzyme)", transform=axA.transAxes,
             ha="center", va="bottom", fontsize=11, color=PAL["ink"])
    show = reuse_counts[::-1]
    y = np.arange(len(show))
    axA.barh(y, show.values, color=PAL["a_fill"], edgecolor=PAL["a_edge"], linewidth=0.8)
    axA.set_yticks(y)
    axA.set_yticklabels(show.index.tolist(), fontsize=9, color=PAL["ink"])
    axA.set_xlabel("Number of loci", fontsize=9, color=PAL["slate"], labelpad=6)
    axA.grid(True, axis="x", color=PAL["grid"], linewidth=0.8)
    axA.tick_params(axis="x", pad=6, colors=PAL["slate"])

    # C
    tidy_axes(axC); panel_label(axC, "C")
    axC.text(0.5, 1.02, "Gene coverage by enzyme (top enzymes)", transform=axC.transAxes,
             ha="center", va="bottom", fontsize=11, color=PAL["ink"])
    idx = np.arange(cov.shape[0])
    left = np.zeros_like(idx, dtype=float)
    gcols = {"S": PAL["S"], "ORF1ab": PAL["ORF1ab"], "N": PAL["N"]}
    for g in ["S", "ORF1ab", "N"]:
        vals = cov[g].values.astype(float)
        axC.barh(idx, vals, left=left, color=gcols[g], edgecolor=PAL["midnight"], linewidth=0.4, label=g)
        left += vals
    axC.set_yticks(idx)
    axC.set_yticklabels(cov.index.tolist(), fontsize=8.8, color=PAL["ink"])
    axC.set_xlabel("Covered loci (count)", fontsize=9, color=PAL["slate"], labelpad=6)
    axC.grid(True, axis="x", color=PAL["grid"], linewidth=0.8)
    axC.tick_params(axis="x", pad=6, colors=PAL["slate"])
    axC.legend(loc="upper right", fontsize=8, frameon=False, ncol=3, bbox_to_anchor=(1.0, 1.01))

    # B
    tidy_axes(axB); panel_label(axB, "B")
    axB.text(0.5, 1.01, "Enzyme–locus bipartite network", transform=axB.transAxes,
             ha="center", va="bottom", fontsize=11, color=PAL["ink"])
    axB.set_xlim(0, 1); axB.set_ylim(0, 1)
    axB.set_xticks([]); axB.set_yticks([])
    for e, L in edges:
        x1, y1 = pos_enzyme[e]
        x2, y2 = pos_locus[L]
        axB.plot([x1, x2], [y1, y2], color=PAL["b_link"], linewidth=0.55)
    for e in enzymes_all:
        x, y0 = pos_enzyme[e]
        axB.scatter([x], [y0], s=85, color=PAL["b_enzyme"], edgecolors=PAL["midnight"], linewidths=0.55, zorder=3)
        axB.text(x - 0.018, y0, e, ha="right", va="center", fontsize=7.8, color=PAL["ink"])
    gene_color = {"S": PAL["S"], "ORF1ab": PAL["ORF1ab"], "N": PAL["N"]}
    for L in loci:
        x, y0 = pos_locus[L]
        g = str(loci_gene.get(L, "S")).strip()
        g = "ORF1ab" if g.lower() == "orf1ab" else g
        col = gene_color.get(g, PAL["S"])
        axB.scatter([x], [y0], s=60, color=col, edgecolors=PAL["midnight"], linewidths=0.5, zorder=3)
        axB.text(x + 0.018, y0, L, ha="left", va="center", fontsize=7.4, color=PAL["ink"])
    axB.text(0.07, 0.015, "Enzymes", fontsize=8.2, color=PAL["slate"], ha="left")
    axB.text(0.93, 0.015, "Loci (colored by gene)", fontsize=8.2, color=PAL["slate"], ha="right")

    # D
    tidy_axes(axD); panel_label(axD, "D")
    axD.text(0.5, 1.02, "Digest-site effect profile per enzyme (top enzymes)", transform=axD.transAxes,
             ha="center", va="bottom", fontsize=11, color=PAL["ink"])
    idx = np.arange(eff.shape[0])
    left = np.zeros_like(idx, dtype=float)
    ecol = {"gain": PAL["gain"], "loss": PAL["loss"], "ambig": PAL["ambig"], "unknown": PAL["unknown"]}
    for c in ["gain", "loss", "ambig", "unknown"]:
        vals = eff[c].values.astype(float)
        axD.barh(idx, vals, left=left, color=ecol[c], edgecolor=PAL["midnight"], linewidth=0.35, label=c)
        left += vals
    axD.set_yticks(idx)
    axD.set_yticklabels(eff.index.tolist(), fontsize=8.8, color=PAL["ink"])
    axD.grid(True, axis="x", color=PAL["grid"], linewidth=0.8)
    axD.tick_params(axis="x", pad=6, colors=PAL["slate"])
    axD.legend(loc="upper right", fontsize=8, frameon=False, ncol=2)

    # E
    tidy_axes(axE); panel_label(axE, "E")
    axE.text(0.5, 1.02, "Minimal enzyme set (greedy set cover)", transform=axE.transAxes,
             ha="center", va="bottom", fontsize=11, color=PAL["ink"])
    axE.set_axis_off()
    total = len(universe)
    lines = [
        f"Universe (assigned loci): {total}",
        f"Uncovered after greedy: {len(uncovered)}",
        "",
        "Selected enzymes (order):"
    ]
    for i, (e, newly, covtot) in enumerate(chosen[:7], 1):
        lines.append(f"{i:>2}. {e:<10} +{newly:<2}  covered={covtot}/{total}")
    if len(chosen) > 7:
        lines.append(f"... +{len(chosen)-7} more")
    axE.text(0.02, 0.90, "\n".join(lines), transform=axE.transAxes,
             ha="left", va="top", fontsize=8.8, color=PAL["ink"], family="DejaVu Sans Mono")
    axE2 = axE.inset_axes([0.62, 0.18, 0.36, 0.70])
    tidy_axes(axE2)
    steps = np.arange(1, len(chosen) + 1)
    covtot = [c for (_, _, c) in chosen]
    if covtot:
        axE2.plot(steps, covtot, color=PAL["gold"], linewidth=2.0)
        axE2.scatter(steps, covtot, color=PAL["gold"], edgecolors=PAL["midnight"], linewidths=0.5, s=26)
        axE2.set_ylim(0, max(covtot) + 1)
    axE2.set_xlabel("Step", fontsize=8, color=PAL["slate"])
    axE2.set_ylabel("Covered loci", fontsize=8, color=PAL["slate"])
    axE2.grid(True, color=PAL["grid"], linewidth=0.8)
    axE2.tick_params(labelsize=8, colors=PAL["slate"])

    # F
    tidy_axes(axF); panel_label(axF, "F")
    axF.text(0.5, 1.02, f"Enzyme QC summary (τ={args.qc_tau:g} bp, top enzymes)", transform=axF.transAxes,
             ha="center", va="bottom", fontsize=11, color=PAL["ink"])
    idx = np.arange(qc_agg.shape[0])
    left = np.zeros_like(idx, dtype=float)
    qcols = [("too_close", PAL["qc_close"]), ("tiny_frag", PAL["qc_tiny"]), ("sum_mismatch", PAL["qc_sum"])]
    for name, col in qcols:
        vals = qc_agg[name].values.astype(float)
        axF.barh(idx, vals, left=left, color=col, edgecolor=PAL["midnight"], linewidth=0.35, label=name)
        left += vals
    axF.set_yticks(idx)
    axF.set_yticklabels(qc_agg.index.tolist(), fontsize=8.8, color=PAL["ink"])
    axF.set_xlabel("Flagged loci (count)", fontsize=9, color=PAL["slate"], labelpad=6)
    axF.grid(True, axis="x", color=PAL["grid"], linewidth=0.8)
    axF.tick_params(axis="x", pad=6, colors=PAL["slate"])
    axF.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)

    footer = (
        "All panels computed from thesis-derived assay_table.csv. "
        f"NA enzymes excluded from enzyme-level plots: {na_count} loci. "
        "Set cover uses assigned enzymes only; QC flags: too_close (min spacing ≤ τ), "
        "tiny_frag (<50 bp), sum_mismatch (fragment sum ≠ amplicon when applicable)."
    )
    fig.text(0.02, 0.02, footer, fontsize=8, color=PAL["slate"])

    outdir = Path(args.outdir)
    ensure_dir(outdir)
    base = outdir / "Figure_5_Enzyme_Strategy"

    fig.subplots_adjust(left=0.04, right=0.975, top=0.965, bottom=0.065)
    fig.savefig(str(base) + ".pdf", facecolor=PAL["paper"])
    fig.savefig(str(base) + ".png", dpi=int(args.dpi), facecolor=PAL["paper"])
    fig.savefig(str(base) + ".jpg", dpi=int(args.dpi), facecolor=PAL["paper"],
                pil_kwargs={"quality": int(args.jpg_quality), "subsampling": 2})
    plt.close(fig)

if __name__ == "__main__":
    main()