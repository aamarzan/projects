#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 6 — Fragment distributions + gel resolution risk (A–F)
Boss-level v3 (NO EPS): PDF + PNG + JPG

Fixes (per your red circles):
1) Panel B: τ label no longer overlaps histogram (boxed callout + leader line).
2) Panel E: legend moved to TOP-LEFT with clean gap from "E" label.
   Panel E palette upgraded to premium (teal + orchid) while keeping WT/Mut meaning.
3) Removed below-axis note in Panel E to avoid bottom clutter; totals moved inside panel.

Truth source: thesis-derived assay_table.csv (expected fragment sizes). No invented results.
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
    "paper": (1.00, 1.00, 1.00),
    "panel_bg": (0.98, 0.985, 0.995),
    "ink": (0.07, 0.09, 0.12),
    "slate": (0.38, 0.42, 0.50),
    "grid": (0.86, 0.88, 0.92),
    "spine": (0.75, 0.78, 0.85),

    # Header
    "header_l1": (0.90, 0.96, 1.00),
    "header_l2": (0.985, 0.995, 1.00),
    "header_text": (0.07, 0.22, 0.44),

    # Base Fig6 palette
    "wt": (0.24, 0.55, 0.85),      # sapphire
    "mut": (0.88, 0.44, 0.29),     # terracotta
    "accent": (0.62, 0.43, 0.92),  # amethyst

    # Panel E premium palette
    "e_wt": (0.10, 0.68, 0.76),    # premium teal
    "e_mut": (0.76, 0.41, 0.83),   # premium orchid

    "risk_low": (0.14, 0.66, 0.55),
    "risk_mid": (0.91, 0.73, 0.18),
    "risk_high": (0.91, 0.38, 0.33),
    "midnight": (0.05, 0.08, 0.14),
}

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def tidy_axes(ax):
    ax.set_facecolor(PAL["panel_bg"])
    for sp in ax.spines.values():
        sp.set_color(PAL["spine"])
        sp.set_linewidth(0.8)

def panel_label(ax, letter):
    ax.text(0.008, 1.01, letter, transform=ax.transAxes,
            ha="left", va="bottom", fontsize=12, fontweight="bold", color=PAL["ink"])

def draw_header(ax, title, subtitle):
    grad = np.linspace(0, 1, 240)
    c1 = np.array(PAL["header_l1"]); c2 = np.array(PAL["header_l2"])
    rgb = (1 - grad)[:, None] * c1[None, :] + grad[:, None] * c2[None, :]
    img = np.tile(rgb[None, :, :], (2, 1, 1))
    ax.imshow(img, extent=[0, 1, 0, 1], transform=ax.transAxes,
              aspect="auto", interpolation="bicubic", zorder=0)
    ax.set_axis_off()
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_frame_on(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.text(0.5, 0.62, title, ha="center", va="center",
            fontsize=18, fontweight="bold", color=PAL["header_text"], transform=ax.transAxes)
    ax.text(0.5, 0.30, subtitle, ha="center", va="center",
            fontsize=10.5, color=PAL["slate"], transform=ax.transAxes)

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
        vals, ok = [], True
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

def kde_line(x, xmin, xmax, points=320, bw=None):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    grid = np.linspace(xmin, xmax, points)
    if x.size == 0:
        return grid, np.zeros_like(grid)
    if bw is None:
        sd = np.std(x) if x.size > 1 else 1.0
        bw = 1.06 * sd * (x.size ** (-1/5)) if sd > 0 else 1.0
        bw = max(bw, 1.0)
    diffs = (grid[:, None] - x[None, :]) / bw
    dens = np.exp(-0.5 * diffs**2).sum(axis=1) / (x.size * bw * np.sqrt(2*np.pi))
    return grid, dens

def risk_score(min_spacing, min_frag, tau=10.0):
    if not np.isfinite(min_spacing):
        s_comp = 0.0
    else:
        s_comp = np.clip((tau*2 - min_spacing) / (tau*2), 0.0, 1.0)
    if not np.isfinite(min_frag):
        f_comp = 0.0
    else:
        f_comp = np.clip((50.0 - min_frag) / 50.0, 0.0, 1.0)
    return 0.65 * s_comp + 0.35 * f_comp

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assay_csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--title", default="Fragment Distributions + Gel Resolution Risk")
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--jpg_quality", type=int, default=92)
    ap.add_argument("--tau", type=float, default=10.0)
    ap.add_argument("--gel_percent", type=float, default=1.5)
    ap.add_argument("--ladder_step_bp", type=int, default=100)
    args = ap.parse_args()

    df = pd.read_csv(args.assay_csv)
    df = df[df["locus"].notna()].copy()
    df["locus"] = df["locus"].astype(str)

    rows = []
    all_wt_frags, all_mut_frags = [], []
    minsp_wt, minsp_mut = [], []

    for _, r in df.iterrows():
        locus = str(r.get("locus", "")).strip()
        amp = safe_int(r.get("amplicon_bp", None))
        wt = first_or_empty(parse_fragments(r.get("wt_fragments_bp", "")))
        mut = first_or_empty(parse_fragments(r.get("mut_fragments_bp", "")))

        if wt:
            all_wt_frags.extend(wt)
            ms_wt = min_band_spacing(wt)
            mf_wt = float(min(wt))
        else:
            ms_wt = float("inf")
            mf_wt = float("nan")

        if mut:
            all_mut_frags.extend(mut)
            ms_mut = min_band_spacing(mut)
            mf_mut = float(min(mut))
        else:
            ms_mut = float("inf")
            mf_mut = float("nan")

        minsp_wt.append(ms_wt)
        minsp_mut.append(ms_mut)

        cand = []
        if np.isfinite(mf_wt): cand.append(mf_wt)
        if np.isfinite(mf_mut): cand.append(mf_mut)
        mf = float(min(cand)) if cand else float("nan")

        ms = min(ms_wt, ms_mut)
        rs = risk_score(ms, mf, tau=float(args.tau))

        rows.append({
            "locus": locus,
            "amplicon_bp": amp if amp is not None else np.nan,
            "min_spacing_bp": ms if np.isfinite(ms) else np.nan,
            "min_frag_bp": mf if np.isfinite(mf) else np.nan,
            "risk": rs,
        })

    met = pd.DataFrame(rows).sort_values("risk", ascending=False).reset_index(drop=True)

    def partition_counts(frags):
        fr = np.asarray(frags, dtype=float)
        fr = fr[np.isfinite(fr)]
        return {
            "<50": int((fr < 50).sum()),
            "50-100": int(((fr >= 50) & (fr < 100)).sum()),
            ">=100": int((fr >= 100).sum()),
            "total": int(fr.size),
        }

    part_wt = partition_counts(all_wt_frags)
    part_mut = partition_counts(all_mut_frags)

    tau = float(args.tau)

    def categorize(row):
        ms = row["min_spacing_bp"]
        mf = row["min_frag_bp"]
        if (np.isfinite(ms) and ms <= tau) or (np.isfinite(mf) and mf < 50):
            return "High caution"
        if (np.isfinite(ms) and ms <= tau*2) or (np.isfinite(mf) and mf < 100):
            return "Caution"
        return "Standard"

    met["action"] = met.apply(categorize, axis=1)
    action_counts = met["action"].value_counts().to_dict()

    fig = plt.figure(figsize=(18, 12), dpi=170)
    fig.patch.set_facecolor(PAL["paper"])
    gs = GridSpec(44, 32, figure=fig, wspace=1.10, hspace=1.45)

    ax_head = fig.add_subplot(gs[0:4, :])
    draw_header(ax_head, args.title,
                "Gel-physics metrics: fragment size • band spacing • locus risk • practical interpretability")

    axA = fig.add_subplot(gs[6:18, 0:20])
    axB = fig.add_subplot(gs[6:18, 22:32])
    axC = fig.add_subplot(gs[20:34, 0:20])
    axD = fig.add_subplot(gs[20:34, 22:32])
    axE = fig.add_subplot(gs[36:44, 0:20])
    axF = fig.add_subplot(gs[36:44, 22:32])

    # A
    tidy_axes(axA); panel_label(axA, "A")
    axA.set_title("Global fragment-size distributions (expected)", fontsize=11, color=PAL["ink"], pad=8)
    xmax = max([max(all_wt_frags) if all_wt_frags else 0,
                max(all_mut_frags) if all_mut_frags else 0, 650])
    gx_wt, gd_wt = kde_line(all_wt_frags, 0, xmax)
    gx_mut, gd_mut = kde_line(all_mut_frags, 0, xmax)
    axA.plot(gx_wt, gd_wt, linewidth=2.4, color=PAL["wt"], label="WT fragments")
    axA.plot(gx_mut, gd_mut, linewidth=2.4, color=PAL["mut"], label="Mut fragments")
    axA.set_xlabel("Fragment size (bp)", fontsize=9, color=PAL["slate"])
    axA.set_ylabel("Density", fontsize=9, color=PAL["slate"])
    axA.grid(True, color=PAL["grid"], linewidth=0.8)
    axA.legend(frameon=False, fontsize=9, loc="upper right")
    for thr in [50, 100]:
        axA.axvline(thr, color=PAL["spine"], linewidth=1.0, linestyle="--")
        axA.text(thr, axA.get_ylim()[1]*0.92, f"{thr} bp", fontsize=8, color=PAL["slate"], ha="center")

    # B (τ callout)
    tidy_axes(axB); panel_label(axB, "B")
    axB.set_title("Nearest-neighbor band spacing (min Δbp)", fontsize=11, color=PAL["ink"], pad=8)
    ms_wt = np.array([x for x in minsp_wt if np.isfinite(x) and x < 1e6], dtype=float)
    ms_mut = np.array([x for x in minsp_mut if np.isfinite(x) and x < 1e6], dtype=float)
    bins = np.linspace(0, max(50, np.nanmax([ms_wt.max() if ms_wt.size else 0,
                                             ms_mut.max() if ms_mut.size else 0, 40])), 18)
    axB.hist(ms_wt, bins=bins, histtype="step", linewidth=2.2, color=PAL["wt"], label="WT")
    axB.hist(ms_mut, bins=bins, histtype="step", linewidth=2.2, color=PAL["mut"], label="Mut")
    axB.axvline(tau, color=PAL["risk_high"], linewidth=1.6, linestyle="--")

    ymax = axB.get_ylim()[1]
    x_span = axB.get_xlim()[1] - axB.get_xlim()[0]
    axB.annotate(
        f"τ = {tau:g} bp",
        xy=(tau, ymax*0.82),
        xytext=(tau + 0.06*x_span, ymax*0.95),
        ha="left", va="center",
        fontsize=8, color=PAL["risk_high"],
        bbox=dict(boxstyle="round,pad=0.20", facecolor=PAL["paper"], edgecolor=PAL["spine"], linewidth=0.8),
        arrowprops=dict(arrowstyle="-", color=PAL["risk_high"], linewidth=0.9),
    )

    axB.set_xlabel("Minimum spacing between adjacent bands (bp)", fontsize=9, color=PAL["slate"])
    axB.set_ylabel("Loci (count)", fontsize=9, color=PAL["slate"])
    axB.grid(True, color=PAL["grid"], linewidth=0.8)
    axB.legend(frameon=False, fontsize=9, loc="upper right")

    # C
    tidy_axes(axC); panel_label(axC, "C")
    axC.set_title("Locus-level gel resolution risk (spacing + visibility)", fontsize=11, color=PAL["ink"], pad=10)
    y = np.arange(len(met))[::-1]
    risks = met["risk"].values[::-1]
    labels = met["locus"].astype(str).values[::-1]
    cols = [(PAL["risk_high"] if r >= 0.66 else PAL["risk_mid"] if r >= 0.33 else PAL["risk_low"]) for r in risks]
    axC.barh(y, risks, color=cols, edgecolor=PAL["midnight"], linewidth=0.35)
    axC.set_yticks(y)
    axC.set_yticklabels(labels, fontsize=7.4, color=PAL["ink"])
    axC.tick_params(axis="y", pad=2)
    axC.margins(x=0.02)
    axC.set_xlabel("Risk score (0–1; higher = harder to resolve)", fontsize=9, color=PAL["slate"])
    axC.set_xlim(0, 1.0)
    axC.grid(True, axis="x", color=PAL["grid"], linewidth=0.8)
    for v in [0.33, 0.66]:
        axC.axvline(v, color=PAL["spine"], linestyle="--", linewidth=1.0)

    # D
    tidy_axes(axD); panel_label(axD, "D")
    axD.set_title("Amplicon size vs smallest fragment", fontsize=11, color=PAL["ink"], pad=8)
    axD.scatter(met["amplicon_bp"].values, met["min_frag_bp"].values,
                s=34, color=PAL["accent"], edgecolors=PAL["midnight"], linewidths=0.4)
    axD.axhline(50, color=PAL["spine"], linestyle="--", linewidth=1.0)
    axD.axhline(100, color=PAL["spine"], linestyle="--", linewidth=1.0)
    axD.set_xlabel("Amplicon size (bp)", fontsize=9, color=PAL["slate"])
    axD.set_ylabel("Smallest expected fragment (bp)", fontsize=9, color=PAL["slate"])
    axD.grid(True, color=PAL["grid"], linewidth=0.8)
    axD.set_ylim(bottom=0)

    # E (legend top-left + premium colors)
    tidy_axes(axE); panel_label(axE, "E")
    axE.set_title("Fragments under practical thresholds", fontsize=11, color=PAL["ink"], pad=8)
    cats = ["<50", "50-100", ">=100"]
    wt_vals = [part_wt[c] for c in cats]
    mut_vals = [part_mut[c] for c in cats]
    x = np.arange(len(cats))
    w = 0.34
    axE.bar(x - w/2, wt_vals, width=w, color=PAL["e_wt"], edgecolor=PAL["midnight"], linewidth=0.45, label="WT")
    axE.bar(x + w/2, mut_vals, width=w, color=PAL["e_mut"], edgecolor=PAL["midnight"], linewidth=0.45, label="Mut")
    axE.set_xticks(x)
    axE.set_xticklabels(cats, fontsize=9, color=PAL["ink"])
    axE.set_ylabel("Fragments (count)", fontsize=9, color=PAL["slate"])
    axE.grid(True, axis="y", color=PAL["grid"], linewidth=0.8)
    axE.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0.10, 0.98),
               handlelength=1.2, borderaxespad=0.0)
    axE.text(0.01, 0.05, f"WT total={part_wt['total']}  •  Mut total={part_mut['total']}",
             transform=axE.transAxes, fontsize=8, color=PAL["slate"], ha="left", va="bottom")

    # F
    tidy_axes(axF); panel_label(axF, "F")
    axF.set_title("Interpretability guide (study gel conditions)", fontsize=11, color=PAL["ink"], pad=8)
    axF.set_axis_off()
    std = int(action_counts.get("Standard", 0))
    cau = int(action_counts.get("Caution", 0))
    hi  = int(action_counts.get("High caution", 0))
    txt = [
        f"Study gel: {args.gel_percent:g}% agarose + {args.ladder_step_bp} bp ladder step.",
        "",
        "Computed interpretability (from expected fragments):",
        f"• Standard: {std} loci  (min spacing > {2*tau:g} bp and smallest fragment ≥ 100 bp)",
        f"• Caution: {cau} loci   (spacing ≤ {2*tau:g} bp or smallest fragment < 100 bp)",
        f"• High caution: {hi} loci (spacing ≤ {tau:g} bp or smallest fragment < 50 bp)",
        "",
        "Operational note: High-caution loci are most likely to show merged/weak bands.",
    ]
    axF.text(0.02, 0.90, "\n".join(txt), transform=axF.transAxes,
             ha="left", va="top", fontsize=8.6, color=PAL["ink"])

    fig.text(
        0.02, 0.02,
        "All panels computed from thesis-derived assay_table.csv (expected fragment sizes). "
        "Risk uses min band spacing and smallest fragment; gel percent/ladder step are study protocol inputs.",
        fontsize=8, color=PAL["slate"]
    )

    outdir = Path(args.outdir)
    ensure_dir(outdir)
    base = outdir / "Figure_6_Gel_Resolution_Risk"

    fig.subplots_adjust(left=0.070, right=0.975, top=0.965, bottom=0.070)

    fig.savefig(str(base) + ".pdf", facecolor=PAL["paper"])
    fig.savefig(str(base) + ".png", dpi=int(args.dpi), facecolor=PAL["paper"])
    fig.savefig(str(base) + ".jpg", dpi=int(args.dpi), facecolor=PAL["paper"],
                pil_kwargs={"quality": int(args.jpg_quality), "subsampling": 2})
    plt.close(fig)

if __name__ == "__main__":
    main()