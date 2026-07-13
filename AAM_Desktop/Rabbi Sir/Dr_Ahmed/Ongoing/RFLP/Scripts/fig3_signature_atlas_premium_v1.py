#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figure 3 (Premium) — PCR–RFLP Signature Atlas (A–F)
- EPS-safe (no transparency), publication layout, multi-panel composite.

Inputs:
  --assay_csv : master assay table (assay_table_complete.csv)
  --gel       : optional path to ONE representative burnished gel PNG/JPG
  --outdir    : output folder (will be created)
Outputs:
  Figure_3_RFLP_Signature_Atlas.(png|pdf|eps)

Notes:
- Band positions are synthetic (computed from fragment sizes) for visual signature comparison.
- Real gel is included only as a small inset (panel F).
"""

import argparse
import math
import os
import re
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

try:
    from PIL import Image
except Exception:
    Image = None


# -----------------------------
# Global style (EPS-safe)
# -----------------------------
mpl.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "white",
    "font.family": "DejaVu Sans",      # cross-platform safe; change to Arial if installed
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# -----------------------------
# Helpers
# -----------------------------
def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def slugify(s: str) -> str:
    s = re.sub(r"\s+", "_", s.strip())
    s = re.sub(r"[^A-Za-z0-9_\-]+", "", s)
    return s

def parse_fragments(cell: str) -> List[List[int]]:
    """
    Parse fragment strings into alternative signatures.
    Examples:
      "77+531" -> [[77,531]]
      "74+95|169" -> [[74,95],[169]]
      "150" -> [[150]]
      ""/NA -> []
    """
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
    s = str(cell).strip()
    if not s or s.lower() in {"na", "nan", "none"}:
        return []
    alts = []
    for alt in s.split("|"):
        alt = alt.strip()
        if not alt:
            continue
        parts = [p.strip() for p in alt.split("+") if p.strip()]
        nums = []
        ok = True
        for p in parts:
            m = re.match(r"^(\d+)\s*(?:bp)?$", p, flags=re.IGNORECASE)
            if not m:
                ok = False
                break
            nums.append(int(m.group(1)))
        if ok and nums:
            alts.append(nums)
    return alts

def fragments_sum_mismatch(amplicon: Optional[int], alts: List[List[int]]) -> bool:
    if amplicon is None or (isinstance(amplicon, float) and np.isnan(amplicon)):
        return False
    if not alts:
        return False
    for sig in alts:
        if sum(sig) != int(amplicon):
            return True
    return False

def draw_gradient_bar(ax, x0, y0, w, h, c1=(0.15,0.35,0.75), c2=(0.05,0.15,0.35), steps=80):
    """
    EPS-safe 'gradient' using many solid rectangles (no alpha).
    """
    for i in range(steps):
        t = i/(steps-1)
        c = (c1[0]*(1-t)+c2[0]*t,
             c1[1]*(1-t)+c2[1]*t,
             c1[2]*(1-t)+c2[2]*t)
        ax.add_patch(Rectangle((x0 + w*i/steps, y0), w/steps, h, facecolor=c, edgecolor=c, lw=0))

def add_panel_label(ax, letter: str):
    ax.text(-0.02, 1.02, letter, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=12, fontweight="bold", color="black")

def read_gel_image(path: str):
    if not path:
        return None
    if Image is None:
        return None
    img = Image.open(path)
    # EPS-safe: avoid alpha
    if img.mode in ("RGBA", "LA"):
        img = img.convert("RGB")
    return np.asarray(img)

def bp_to_gel_y(bp: int, bp_min: int, bp_max: int) -> float:
    """
    Map bp to gel migration position (0 bottom -> 1 top).
    Larger bp stays nearer the top.
    """
    bp = max(bp_min, min(bp_max, bp))
    lo = math.log10(bp_min)
    hi = math.log10(bp_max)
    v = (math.log10(bp) - lo) / (hi - lo + 1e-9)
    return 1.0 - v


# -----------------------------
# Main plotting
# -----------------------------
def make_figure(df: pd.DataFrame, gel_img, out_base: str, title: str):
    # Clean / order
    df = df.copy()

    # Ensure required columns exist
    required = ["locus","gene","aa_pos_start","aa_pos_end","amplicon_bp","enzyme","wt_fragments_bp","mut_fragments_bp","site_effect","notes"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns in assay CSV: {missing}")

    # Sort by gene then aa position
    gene_order = {"S":0, "ORF1ab":1, "N":2}
    df["gene_rank"] = df["gene"].map(lambda g: gene_order.get(str(g), 99))
    df["aa_pos"] = pd.to_numeric(df["aa_pos_start"], errors="coerce")
    df = df.sort_values(["gene_rank","aa_pos","locus"]).reset_index(drop=True)

    # Fragment parsing
    df["wt_alts"] = df["wt_fragments_bp"].apply(parse_fragments)
    df["mut_alts"] = df["mut_fragments_bp"].apply(parse_fragments)

    # Flags for panel E
    df["flag_ambiguous"] = df.apply(lambda r: (len(r["wt_alts"])>1) or (len(r["mut_alts"])>1) or ("|" in str(r.get("wt_fragments_bp",""))) or ("|" in str(r.get("mut_fragments_bp",""))), axis=1)
    df["flag_pending"] = df.apply(lambda r: (pd.isna(r["amplicon_bp"]) or pd.isna(r["enzyme"]) or (not r["wt_alts"]) or (not r["mut_alts"])), axis=1)
    df["flag_sum_mismatch"] = df.apply(lambda r: fragments_sum_mismatch(r["amplicon_bp"], r["wt_alts"]) or fragments_sum_mismatch(r["amplicon_bp"], r["mut_alts"]), axis=1)
    # practical visibility flag: any fragment <50 bp in any signature
    def has_small(alts):
        for sig in alts:
            if any(x < 50 for x in sig):
                return True
        return False
    df["flag_small_frag"] = df["wt_alts"].apply(has_small) | df["mut_alts"].apply(has_small)

    # Determine global bp scale for synthetic gel drawing
    all_frags = []
    for alts in list(df["wt_alts"]) + list(df["mut_alts"]):
        for sig in alts:
            all_frags.extend(sig)
    if not all_frags:
        raise SystemExit("No fragment sizes found in CSV.")
    bp_min = max(30, int(np.percentile(all_frags, 2)))   # robust min
    bp_max = int(np.percentile(all_frags, 98))           # robust max
    bp_min = min(bp_min, 80)
    bp_max = max(bp_max, 600)

    # Layout
    fig = plt.figure(figsize=(15.5, 10.2), dpi=150)
    outer = GridSpec(4, 3, figure=fig,
                     height_ratios=[0.9, 3.0, 2.3, 2.0],
                     width_ratios=[2.6, 2.6, 2.2],
                     hspace=0.30, wspace=0.25)

    # Header (figure-level) - EPS-safe gradient
    ax_head = fig.add_axes([0.02, 0.94, 0.96, 0.05])
    ax_head.set_axis_off()
    draw_gradient_bar(ax_head, 0, 0, 1, 1, c1=(0.25,0.55,0.95), c2=(0.05,0.15,0.35), steps=120)
    ax_head.text(0.5, 0.52, title, ha="center", va="center",
                 fontsize=14, fontweight="bold", color="black")

    # Panel A: gene/locus map
    axA = fig.add_subplot(outer[0, :])
    axA.set_axis_off()
    add_panel_label(axA, "A")

    genes = ["S", "ORF1ab", "N"]
    y_positions = {"S":0.75, "ORF1ab":0.45, "N":0.15}
    # gene lengths are data-driven (max aa in table + margin)
    gene_len = {}
    for g in genes:
        mx = pd.to_numeric(df.loc[df["gene"]==g, "aa_pos_end"], errors="coerce").max()
        if pd.isna(mx):
            continue
        gene_len[g] = int(mx) + (200 if g=="ORF1ab" else 80)

    # Draw tracks
    for g in genes:
        if g not in gene_len:
            continue
        y = y_positions[g]
        axA.add_patch(Rectangle((0.08, y-0.03), 0.84, 0.06,
                                facecolor=(0.96,0.97,0.99), edgecolor=(0.2,0.2,0.2), lw=1.0))
        axA.text(0.02, y, g, ha="left", va="center", fontsize=11, fontweight="bold")
        axA.text(0.92, y, f"{gene_len[g]} aa", ha="right", va="center", fontsize=9)

        # Markers
        sub = df[df["gene"]==g]
        for _, r in sub.iterrows():
            pos = int(r["aa_pos_start"])
            x = 0.08 + 0.84*(pos/gene_len[g])
            axA.add_patch(Rectangle((x-0.002, y-0.035), 0.004, 0.07, facecolor=(0.1,0.1,0.1), edgecolor=(0.1,0.1,0.1), lw=0))
        # Add light tick labels at quartiles
        for frac in [0, 0.25, 0.5, 0.75, 1.0]:
            x = 0.08 + 0.84*frac
            axA.add_patch(Rectangle((x-0.0007, y-0.04), 0.0014, 0.08, facecolor=(0.3,0.3,0.3), edgecolor=(0.3,0.3,0.3), lw=0))
            axA.text(x, y-0.07, f"{int(gene_len[g]*frac)}", ha="center", va="top", fontsize=8, color=(0.2,0.2,0.2))

    # Panel B: synthetic lanes grid (WT vs Mut)
    axB = fig.add_subplot(outer[1:, 0:2])
    axB.set_axis_off()
    add_panel_label(axB, "B")

    # Geometry inside axB (axes coords)
    n = len(df)
    top = 0.96
    bottom = 0.06
    row_h = (top-bottom) / max(n, 1)
    # lanes
    lane_w = 0.06
    gap = 0.018
    x_locus = 0.02
    x_wt = 0.40
    x_mut = x_wt + lane_w + gap
    x_info = x_mut + lane_w + 0.03

    # headers for lanes
    axB.text(x_wt + lane_w/2, 0.99, "WT", ha="center", va="top", fontsize=10, fontweight="bold")
    axB.text(x_mut + lane_w/2, 0.99, "Mut", ha="center", va="top", fontsize=10, fontweight="bold")
    axB.text(x_info, 0.99, "Enzyme / Amplicon", ha="left", va="top", fontsize=10, fontweight="bold")

    # size ruler
    ruler_x = x_wt - 0.06
    axB.text(ruler_x, 0.99, "bp", ha="center", va="top", fontsize=9, fontweight="bold")
    for bp_tick in [bp_max, 400, 200, 100, 50]:
        if bp_tick < bp_min or bp_tick > bp_max:
            continue
        y = bottom + (top-bottom)*bp_to_gel_y(bp_tick, bp_min, bp_max)
        axB.add_patch(Rectangle((ruler_x-0.004, y-0.0008), 0.008, 0.0016, facecolor=(0.15,0.15,0.15), edgecolor=(0.15,0.15,0.15), lw=0))
        axB.text(ruler_x-0.01, y, str(bp_tick), ha="right", va="center", fontsize=8, color=(0.1,0.1,0.1))

    # draw rows
    last_gene = None
    for i, r in df.iterrows():
        y0 = top - (i+1)*row_h
        yc = y0 + row_h/2

        gene = str(r["gene"])
        if gene != last_gene:
            # separator band
            axB.add_patch(Rectangle((0.0, y0 + row_h*0.88), 1.0, row_h*0.12,
                                    facecolor=(0.93,0.95,0.98), edgecolor=(0.85,0.88,0.93), lw=0))
            axB.text(0.0, y0 + row_h*0.94, f"{gene}", ha="left", va="center", fontsize=9, fontweight="bold")
            last_gene = gene

        # locus label
        axB.text(x_locus, yc, str(r["locus"]), ha="left", va="center", fontsize=9)

        # lane backgrounds
        for x in [x_wt, x_mut]:
            axB.add_patch(Rectangle((x, y0 + row_h*0.12), lane_w, row_h*0.76,
                                    facecolor=(0.98,0.98,0.98), edgecolor=(0.25,0.25,0.25), lw=0.8))

        # bands
        def draw_lane(alts, x_left):
            if not alts:
                return
            # primary signature
            primary = alts[0]
            for bp in primary:
                yy = bottom + (top-bottom)*bp_to_gel_y(bp, bp_min, bp_max)
                # clamp into current row
                # Map global gel y into row bounds:
                row_bottom = y0 + row_h*0.14
                row_top = y0 + row_h*0.86
                # convert yy (0..1) to row:
                # using overall bottom/top as gel extent
                yy_row = row_bottom + (yy-bottom)/(top-bottom) * (row_top-row_bottom)
                axB.add_patch(Rectangle((x_left+0.006, yy_row-0.0045), lane_w-0.012, 0.009,
                                        facecolor=(0.08,0.08,0.08), edgecolor=(0.08,0.08,0.08), lw=0))
            # if ambiguous, draw secondary signature with slight x-offset (visual cue)
            if len(alts) > 1:
                secondary = alts[1]
                for bp in secondary:
                    yy = bottom + (top-bottom)*bp_to_gel_y(bp, bp_min, bp_max)
                    row_bottom = y0 + row_h*0.14
                    row_top = y0 + row_h*0.86
                    yy_row = row_bottom + (yy-bottom)/(top-bottom) * (row_top-row_bottom)
                    axB.add_patch(Rectangle((x_left+0.010, yy_row-0.0035), lane_w-0.020, 0.007,
                                            facecolor=(0.35,0.35,0.35), edgecolor=(0.35,0.35,0.35), lw=0))
                # ambiguity badge
                axB.text(x_left+lane_w-0.003, y0+row_h*0.86, "*", ha="right", va="bottom", fontsize=11, fontweight="bold")

        draw_lane(r["wt_alts"], x_wt)
        draw_lane(r["mut_alts"], x_mut)

        # enzyme / amplicon info
        enz = str(r["enzyme"]) if not pd.isna(r["enzyme"]) else "NA"
        amp = r["amplicon_bp"]
        amp_txt = f"{int(amp)} bp" if not pd.isna(amp) else "NA"
        eff = str(r["site_effect"]) if not pd.isna(r["site_effect"]) else ""
        axB.text(x_info, yc, f"{enz}  ·  {amp_txt}  ·  {eff}", ha="left", va="center", fontsize=8.6, color=(0.1,0.1,0.1))

        # QC micro-badges
        bx = 0.94
        if bool(r["flag_pending"]):
            axB.add_patch(Rectangle((bx, yc-0.010), 0.015, 0.020, facecolor=(0.85,0.25,0.25), edgecolor=(0.45,0.10,0.10), lw=0.6))
        if bool(r["flag_ambiguous"]):
            axB.add_patch(Rectangle((bx+0.018, yc-0.010), 0.015, 0.020, facecolor=(0.95,0.70,0.20), edgecolor=(0.50,0.35,0.10), lw=0.6))
        if bool(r["flag_sum_mismatch"]):
            axB.add_patch(Rectangle((bx+0.036, yc-0.010), 0.015, 0.020, facecolor=(0.40,0.40,0.40), edgecolor=(0.15,0.15,0.15), lw=0.6))

    # Legend for QC badges (bottom)
    axB.text(0.74, 0.02, "QC flags:", ha="left", va="center", fontsize=9, fontweight="bold")
    def badge(x, label, fc, ec):
        axB.add_patch(Rectangle((x, 0.012), 0.018, 0.022, facecolor=fc, edgecolor=ec, lw=0.8))
        axB.text(x+0.024, 0.023, label, ha="left", va="center", fontsize=8.5)
    badge(0.82, "Missing/Pending", (0.85,0.25,0.25), (0.45,0.10,0.10))
    badge(0.90, "Ambiguity", (0.95,0.70,0.20), (0.50,0.35,0.10))
    badge(0.97, "Sum mismatch", (0.40,0.40,0.40), (0.15,0.15,0.15))

    # Right column panels (C, D, E/F)
    # Panel C: gain/loss summary
    axC = fig.add_subplot(outer[1, 2])
    add_panel_label(axC, "C")
    sub = df[~df["site_effect"].isna()].copy()
    if len(sub) == 0:
        axC.text(0.5, 0.5, "No site-effect data", ha="center", va="center")
        axC.set_axis_off()
    else:
        counts = (sub.groupby(["gene","site_effect"]).size()
                    .unstack(fill_value=0)
                    .reindex(index=["S","ORF1ab","N"], fill_value=0))
        genes_present = [g for g in counts.index if g in gene_order]
        x = np.arange(len(genes_present))
        gain = counts.get("gain", pd.Series(0, index=counts.index)).loc[genes_present].values
        loss = counts.get("loss", pd.Series(0, index=counts.index)).loc[genes_present].values
        axC.bar(x, gain, label="gain")
        axC.bar(x, loss, bottom=gain, label="loss")
        axC.set_xticks(x, genes_present)
        axC.set_ylabel("Loci (n)")
        axC.set_title("Digest site effect")
        axC.legend(frameon=False, loc="upper right")
        axC.grid(axis="y", color=(0.88,0.88,0.88), lw=0.8)

    # Panel D: amplicon length vs fragment count (+ density-like cue)
    axD = fig.add_subplot(outer[2, 2])
    add_panel_label(axD, "D")
    df_num = df.copy()
    df_num["amp"] = pd.to_numeric(df_num["amplicon_bp"], errors="coerce")
    # fragment count: use primary signature
    def frag_count(alts):
        if not alts:
            return np.nan
        return len(alts[0])
    df_num["wt_nfrag"] = df_num["wt_alts"].apply(frag_count)
    df_num["mut_nfrag"] = df_num["mut_alts"].apply(frag_count)

    # scatter (WT and Mut), jitter to avoid overlap
    x1 = df_num["amp"].values.astype(float)
    y1 = df_num["wt_nfrag"].values.astype(float)
    x2 = df_num["amp"].values.astype(float)
    y2 = df_num["mut_nfrag"].values.astype(float)
    rng = np.random.default_rng(7)
    xj1 = x1 + rng.normal(0, 2.0, size=len(x1))
    xj2 = x2 + rng.normal(0, 2.0, size=len(x2))
    yj1 = y1 + rng.normal(0, 0.03, size=len(y1))
    yj2 = y2 + rng.normal(0, 0.03, size=len(y2))

    axD.scatter(xj1, yj1, s=25, marker="o", label="WT")
    axD.scatter(xj2, yj2, s=25, marker="^", label="Mut")
    axD.set_xlabel("Amplicon (bp)")
    axD.set_ylabel("# fragments (primary)")
    axD.set_title("Amplicon vs fragment count")
    axD.grid(True, color=(0.90,0.90,0.90), lw=0.8)
    axD.legend(frameon=False, loc="upper left")

    # Panel E+F: QC risk badges + gel inset (split)
    subE = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[3, 2], height_ratios=[1.0, 1.1], hspace=0.25)

    axE = fig.add_subplot(subE[0, 0])
    add_panel_label(axE, "E")
    # Build a small risk summary table
    risk_cols = ["flag_pending","flag_ambiguous","flag_sum_mismatch","flag_small_frag"]
    risk_names = ["Pending","Ambig","Mismatch","<50bp"]
    mat = df[risk_cols].astype(int).values
    axE.imshow(mat, aspect="auto", interpolation="nearest")
    axE.set_yticks(range(len(df)))
    axE.set_yticklabels(df["locus"].tolist(), fontsize=7)
    axE.set_xticks(range(len(risk_names)))
    axE.set_xticklabels(risk_names, rotation=0, fontsize=8)
    axE.set_title("QC / ambiguity flags")
    axE.set_xlabel("")
    axE.set_ylabel("")
    # annotate counts
    col_sums = mat.sum(axis=0)
    for j, s in enumerate(col_sums):
        axE.text(j, -0.8, f"n={int(s)}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    # grid lines
    axE.set_xticks(np.arange(-.5, len(risk_names), 1), minor=True)
    axE.set_yticks(np.arange(-.5, len(df), 1), minor=True)
    axE.grid(which="minor", color=(1,1,1), linestyle="-", linewidth=0.8)
    axE.tick_params(which="minor", bottom=False, left=False)

    axF = fig.add_subplot(subE[1, 0])
    add_panel_label(axF, "F")
    axF.set_title("Representative gel (inset)")
    axF.set_axis_off()
    if gel_img is None:
        axF.text(0.5, 0.5, "Provide --gel path to include\none burnished gel panel",
                 ha="center", va="center", fontsize=9, color=(0.2,0.2,0.2))
        axF.add_patch(Rectangle((0.08,0.18), 0.84, 0.64, transform=axF.transAxes,
                                facecolor=(0.98,0.98,0.98), edgecolor=(0.35,0.35,0.35), lw=1.0))
    else:
        axF.imshow(gel_img)
        # border
        axF.add_patch(Rectangle((0,0), 1, 1, transform=axF.transAxes,
                                facecolor="none", edgecolor=(0.15,0.15,0.15), lw=1.2))

    # Footer note
    fig.text(0.02, 0.012,
             "Synthetic lanes in Panel B are computed from expected fragment sizes (assay_table). "
             "Ambiguity (*) indicates alternative signatures reported in the thesis.",
             ha="left", va="bottom", fontsize=8, color=(0.25,0.25,0.25))

    # Save
    safe_mkdir(os.path.dirname(out_base))
    png = out_base + ".png"
    pdf = out_base + ".pdf"
    eps = out_base + ".eps"

    fig.savefig(png, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(eps, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return png, pdf, eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assay_csv", required=True, help="Path to assay_table_complete.csv")
    ap.add_argument("--gel", default="", help="Path to ONE representative burnished gel image (png/jpg). Optional.")
    ap.add_argument("--outdir", required=True, help="Output directory (e.g., ...\\Final Figure\\Fig_3)")
    ap.add_argument("--title", default="Figure 3. PCR–RFLP Signature Atlas", help="Figure title (shown in header bar)")
    args = ap.parse_args()

    df = pd.read_csv(args.assay_csv)
    gel_img = read_gel_image(args.gel) if args.gel else None

    outdir = args.outdir
    safe_mkdir(outdir)

    out_base = os.path.join(outdir, "Figure_3_RFLP_Signature_Atlas")
    make_figure(df, gel_img, out_base, args.title)

    print("Saved:")
    print(" -", out_base + ".png")
    print(" -", out_base + ".pdf")
    print(" -", out_base + ".eps")


if __name__ == "__main__":
    main()
