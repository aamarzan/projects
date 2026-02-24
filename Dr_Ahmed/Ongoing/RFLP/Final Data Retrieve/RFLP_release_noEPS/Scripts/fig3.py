#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figure 3 (Premium v2) — PCR–RFLP Signature Atlas (A–F)
Fixes vs v1:
- No text/axis overlaps (Panel A end labels, Panel B gene headers, Panel E y-label crowding).
- Corrected site-effect counting (gain/loss/ambiguous parsed from thesis strings like "gain|none").
- Premium color system (deep violet ↔ midnight header, teal/coral/gold accents).
- Panel E redesigned as "flag-only matrix" (only loci with ≥1 QC flag), with per-flag colors and counts.
- QC legend consolidated and kept inside bounds.
EPS-safe: no transparency/alpha is used anywhere.

Inputs:
  --assay_csv : assay_table.csv (truth-derived)
  --gel       : optional ONE burnished gel PNG/JPG (kept small, no band changes)
  --outdir    : output folder
Outputs:
  Figure_3_RFLP_Signature_Atlas.(png|pdf|eps)
"""

import argparse
import os
import re
from typing import List, Optional, Tuple, Dict

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
    "font.family": "DejaVu Sans",
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})


# -----------------------------
# Premium palette (solid colors)
# -----------------------------
PAL = {
    "midnight": (0.05, 0.08, 0.18),
    "violet":   (0.45, 0.23, 0.72),
    # Light header (requested) — keeps title readable & print-friendly
    "header_l1": (0.88, 0.90, 0.98),
    "header_l2": (0.95, 0.97, 1.00),
    "teal":     (0.08, 0.55, 0.54),
    "coral":    (0.89, 0.36, 0.26),
    "gold":     (0.90, 0.70, 0.18),
    "slate":    (0.35, 0.39, 0.48),
    "ink":      (0.10, 0.10, 0.10),
    "grid":     (0.90, 0.90, 0.90),
    "track":    (0.96, 0.97, 0.99),
    "band":     (0.08, 0.08, 0.08),
    "band2":    (0.35, 0.35, 0.35),
    "soft":     (0.94, 0.96, 0.99),
    "border":   (0.18, 0.18, 0.18),
}


def safe_mkdir(p: str) -> None:
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


def draw_gradient_bar(ax, x0, y0, w, h, c1, c2, steps=140):
    """EPS-safe gradient: many thin rectangles (no alpha)."""
    for i in range(steps):
        t = i / max(steps - 1, 1)
        c = tuple((1 - t) * c1[j] + t * c2[j] for j in range(3))
        ax.add_patch(Rectangle((x0 + w * (i / steps), y0), w / steps, h,
                               facecolor=c, edgecolor=c, lw=0))


def add_panel_label(ax, letter: str):
    ax.text(-0.02, 1.02, letter, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=12, fontweight="bold", color="black")


def parse_fragments(s) -> List[List[int]]:
    """
    Parses fragment strings:
      "77+531" -> [[77,531]]
      "74+95|169" -> [[74,95], [169]]
      "NA" / empty -> []
    """
    if s is None:
        return []
    s = str(s).strip()
    if s == "" or s.lower() == "na" or s.lower() == "nan":
        return []
    alts = []
    for alt in s.split("|"):
        alt = alt.strip()
        if not alt:
            continue
        parts = []
        for p in alt.split("+"):
            p = p.strip()
            if p.isdigit():
                parts.append(int(p))
        if parts:
            alts.append(sorted(parts))
    return alts


def fragments_sum_mismatch(amp, alts: List[List[int]], tol=1) -> bool:
    if amp is None or (isinstance(amp, float) and np.isnan(amp)):
        return False
    try:
        amp = int(float(amp))
    except Exception:
        return False
    for sig in alts:
        if not sig:
            continue
        if abs(sum(sig) - amp) > tol:
            return True
    return False


def bp_to_gel_y(bp: int, bp_min: int, bp_max: int) -> float:
    """
    Map fragment size to synthetic gel y-position.
    Larger fragments run higher: use log scaling.
    Returns y in [0,1].
    """
    bp = max(bp_min, min(bp_max, int(bp)))
    # log scale: 0 (top) for bp_max, 1 (bottom) for bp_min
    a = np.log(bp_max)
    b = np.log(bp_min)
    y = (a - np.log(bp)) / max(a - b, 1e-9)
    return float(y)


def parse_site_effect(effect: str) -> str:
    """
    Normalize site_effect field:
      "gain" -> gain
      "loss" -> loss
      "gain|none" / "loss|none" -> ambiguous
      anything else -> ""
    """
    if effect is None:
        return ""
    s = str(effect).strip().lower()
    if s == "" or s == "na" or s == "nan":
        return ""
    if "|" in s:
        return "ambiguous"
    if s == "gain":
        return "gain"
    if s == "loss":
        return "loss"
    return ""


def load_gel(path: str):
    if not path:
        return None
    if Image is None:
        return None
    try:
        im = Image.open(path).convert("RGB")  # EPS-safe (no alpha)
        # mild resize for inset clarity
        return im
    except Exception:
        return None


def build_rows_with_gene_headers(df: pd.DataFrame) -> List[Dict]:
    """
    Insert a header spacer row before each gene block to avoid overlaps.
    """
    rows = []
    last_gene = None
    for _, r in df.iterrows():
        gene = str(r["gene"])
        if gene != last_gene:
            rows.append({"_type": "header", "gene": gene})
            last_gene = gene
        d = dict(r)
        d["_type"] = "data"
        rows.append(d)
    return rows


def make_figure(assay_csv: str, gel_path: str, outdir: str, title: str):
    df = pd.read_csv(assay_csv)

    # enforce gene order + sorting by aa_pos
    gene_order = {"S": 0, "ORF1ab": 1, "N": 2}
    df["gene_rank"] = df["gene"].map(lambda g: gene_order.get(str(g), 99))
    df["aa_pos"] = pd.to_numeric(df["aa_pos_start"], errors="coerce")
    df = df.sort_values(["gene_rank", "aa_pos", "locus"]).reset_index(drop=True)

    # fragments
    df["wt_alts"] = df["wt_fragments_bp"].apply(parse_fragments)
    df["mut_alts"] = df["mut_fragments_bp"].apply(parse_fragments)

    # QC flags
    df["flag_ambiguous"] = df.apply(lambda r: (len(r["wt_alts"]) > 1) or (len(r["mut_alts"]) > 1)
                                               or ("|" in str(r.get("wt_fragments_bp", "")))
                                               or ("|" in str(r.get("mut_fragments_bp", ""))), axis=1)
    df["flag_pending"] = df.apply(lambda r: (pd.isna(r["amplicon_bp"]) or pd.isna(r["enzyme"])
                                            or (not r["wt_alts"]) or (not r["mut_alts"])), axis=1)
    df["flag_sum_mismatch"] = df.apply(lambda r: fragments_sum_mismatch(r["amplicon_bp"], r["wt_alts"])
                                               or fragments_sum_mismatch(r["amplicon_bp"], r["mut_alts"]), axis=1)

    def has_small(alts):
        for sig in alts:
            if any(x < 50 for x in sig):
                return True
        return False

    df["flag_small_frag"] = df["wt_alts"].apply(has_small) | df["mut_alts"].apply(has_small)

    # normalized site_effect for correct counting
    df["site_effect_norm"] = df["site_effect"].apply(parse_site_effect)

    # bp scale
    all_frags = []
    for alts in list(df["wt_alts"]) + list(df["mut_alts"]):
        for sig in alts:
            all_frags += sig
    bp_min = 40
    bp_max = int(max(all_frags)) if all_frags else 600
    bp_max = max(bp_max, 600)

    gel_img = load_gel(gel_path)

    # ---------- Layout ----------
    # Wider right column + dedicated bottom space for Panels E–F to prevent overlaps
    fig = plt.figure(figsize=(18.2, 11.8), dpi=150)
    outer = GridSpec(
        2, 2, figure=fig,
        height_ratios=[1.15, 8.85],
        width_ratios=[6.35, 2.85],
        hspace=0.18, wspace=0.18
    )
    # Right column sub-layout (C, D, then E+F)
    right = GridSpecFromSubplotSpec(
        3, 1, subplot_spec=outer[1, 1],
        height_ratios=[1.05, 1.15, 1.70],
        hspace=0.46
    )

    # Header gradient (premium)
    ax_head = fig.add_axes([0.02, 0.94, 0.96, 0.05])
    ax_head.set_axis_off()
    draw_gradient_bar(ax_head, 0, 0, 1, 1, c1=PAL["header_l1"], c2=PAL["header_l2"], steps=160)
    ax_head.text(0.5, 0.52, title, ha="center", va="center",
                 fontsize=14, fontweight="bold", color="black")

    # ---------- Panel A ----------
    axA = fig.add_subplot(outer[0, :])
    axA.set_axis_off()
    # Panel label (placed inside to avoid header overlap)
    axA.text(0.01, 0.98, "A", transform=axA.transAxes,
             ha="left", va="top", fontsize=12, fontweight="bold", color="black")

    genes = ["S", "ORF1ab", "N"]
    y_positions = {"S": 0.75, "ORF1ab": 0.45, "N": 0.15}

    # gene lengths from data (+ margin)
    gene_scale = {}
    for g in genes:
        mx = pd.to_numeric(df.loc[df["gene"] == g, "aa_pos_end"], errors="coerce").max()
        if pd.isna(mx):
            continue
        gene_scale[g] = int(mx) + (200 if g == "ORF1ab" else 80)  # plotting scale (targets + margin)

    for g in genes:
        if g not in gene_scale:
            continue
        y = y_positions[g]
        axA.add_patch(Rectangle((0.08, y - 0.03), 0.82, 0.06,
                                facecolor=PAL["track"], edgecolor=PAL["border"], lw=1.0))
        axA.text(0.02, y, g, ha="left", va="center", fontsize=11, fontweight="bold")
        # NOTE: Track scale is based on assayed target positions (not full-length proteins)

        sub = df[df["gene"] == g]
        for _, r in sub.iterrows():
            pos = int(r["aa_pos_start"])
            x = 0.08 + 0.82 * (pos / gene_scale[g])
            axA.add_patch(Rectangle((x - 0.002, y - 0.035), 0.004, 0.07,
                                    facecolor=PAL["ink"], edgecolor=PAL["ink"], lw=0))

        # minimal ticks (0, 50%, end) — avoid dense labels
        for frac, lab in [(0.0, "0"), (0.5, str(int(gene_scale[g] * 0.5)))]:
            x = 0.08 + 0.82 * frac
            axA.add_patch(Rectangle((x - 0.0007, y - 0.04), 0.0014, 0.08,
                                    facecolor=PAL["slate"], edgecolor=PAL["slate"], lw=0))
            axA.text(x, y - 0.07, lab, ha="center", va="top", fontsize=8, color=PAL["slate"])
        # end tick without numeric label (avoid implying full-length)
        x_end = 0.90
        axA.add_patch(Rectangle((x_end - 0.0007, y - 0.04), 0.0014, 0.08,
                                facecolor=PAL["slate"], edgecolor=PAL["slate"], lw=0))

    # Panel A note (authenticity): scale reflects assayed target AA positions only
    axA.text(0.08, -0.10, "Assayed target amino-acid positions (relative scale; not full-length proteins).",
             transform=axA.transAxes, ha="left", va="top", fontsize=8, color=PAL["slate"])

    # ---------- Panel B ----------
    axB = fig.add_subplot(outer[1, 0])
    axB.set_axis_off()
    add_panel_label(axB, "B")

    rows = build_rows_with_gene_headers(df)
    nrows = len(rows)

    top = 0.965
    bottom = 0.075
    row_h = (top - bottom) / max(nrows, 1)

    lane_w = 0.060
    gap = 0.020
    x_locus = 0.02
    x_wt = 0.40
    x_mut = x_wt + lane_w + gap
    x_info = x_mut + lane_w + 0.035
    x_badges = 0.93

    # headers
    axB.text(x_wt + lane_w / 2, 0.995, "WT", ha="center", va="top", fontsize=10, fontweight="bold")
    axB.text(x_mut + lane_w / 2, 0.995, "Mut", ha="center", va="top", fontsize=10, fontweight="bold")
    axB.text(x_info, 0.995, "Enzyme · Amplicon · Effect", ha="left", va="top", fontsize=10, fontweight="bold")

    # size ruler (left of lanes)
    ruler_x = x_wt - 0.065
    axB.text(ruler_x, 0.995, "bp", ha="center", va="top", fontsize=9, fontweight="bold")
    for bp_tick in [bp_max, 400, 200, 100, 50]:
        if bp_tick < bp_min or bp_tick > bp_max:
            continue
        y = bottom + (top - bottom) * bp_to_gel_y(bp_tick, bp_min, bp_max)
        axB.add_patch(Rectangle((ruler_x - 0.004, y - 0.0008), 0.008, 0.0016,
                                facecolor=PAL["ink"], edgecolor=PAL["ink"], lw=0))
        axB.text(ruler_x - 0.010, y, str(bp_tick), ha="right", va="center", fontsize=8, color=PAL["ink"])

    # QC badges definition (4 flags)
    QC = [
        ("Pending/NA", "flag_pending", PAL["coral"], (0.50, 0.18, 0.12)),
        ("Ambiguous", "flag_ambiguous", PAL["gold"], (0.52, 0.40, 0.10)),
        ("Sum mismatch", "flag_sum_mismatch", PAL["slate"], (0.20, 0.22, 0.28)),
        ("<50 bp", "flag_small_frag", PAL["teal"], (0.05, 0.32, 0.30)),
    ]

    def draw_lane(alts, x_left, y0, row_h):
        if not alts:
            return
        row_bottom = y0 + row_h * 0.18
        row_top = y0 + row_h * 0.86

        primary = alts[0]
        for bp in primary:
            yy = bottom + (top - bottom) * bp_to_gel_y(bp, bp_min, bp_max)
            yy_row = row_bottom + (yy - bottom) / (top - bottom) * (row_top - row_bottom)
            axB.add_patch(Rectangle((x_left + 0.006, yy_row - 0.0044), lane_w - 0.012, 0.0088,
                                    facecolor=PAL["band"], edgecolor=PAL["band"], lw=0))

        if len(alts) > 1:
            secondary = alts[1]
            for bp in secondary:
                yy = bottom + (top - bottom) * bp_to_gel_y(bp, bp_min, bp_max)
                yy_row = row_bottom + (yy - bottom) / (top - bottom) * (row_top - row_bottom)
                axB.add_patch(Rectangle((x_left + 0.010, yy_row - 0.0035), lane_w - 0.020, 0.0070,
                                        facecolor=PAL["band2"], edgecolor=PAL["band2"], lw=0))
            axB.text(x_left + lane_w - 0.003, y0 + row_h * 0.86, "*",
                     ha="right", va="bottom", fontsize=11, fontweight="bold", color=PAL["ink"])

    # draw all rows
    for i, r in enumerate(rows):
        y0 = top - (i + 1) * row_h
        yc = y0 + row_h / 2

        if r["_type"] == "header":
            # clean header row: extra spacing to prevent overlap
            axB.add_patch(Rectangle((0.0, y0 + row_h * 0.05), 1.0, row_h * 0.90,
                                    facecolor=PAL["soft"], edgecolor=(0.84, 0.88, 0.94), lw=0.8))
            axB.text(0.01, yc, str(r["gene"]), ha="left", va="center",
                     fontsize=9.6, fontweight="bold", color=PAL["ink"])
            continue

        # locus label
        axB.text(x_locus, yc, str(r["locus"]), ha="left", va="center", fontsize=9, color=PAL["ink"])

        # lane backgrounds
        for x in [x_wt, x_mut]:
            axB.add_patch(Rectangle((x, y0 + row_h * 0.12), lane_w, row_h * 0.76,
                                    facecolor=(0.985, 0.985, 0.985),
                                    edgecolor=(0.25, 0.25, 0.25), lw=0.8))

        # bands
        draw_lane(r["wt_alts"], x_wt, y0, row_h)
        draw_lane(r["mut_alts"], x_mut, y0, row_h)

        # info
        enz = str(r["enzyme"]) if not pd.isna(r["enzyme"]) else "NA"
        amp = r["amplicon_bp"]
        amp_txt = f"{int(float(amp))} bp" if not pd.isna(amp) else "NA"
        eff = str(r.get("site_effect", "")) if not pd.isna(r.get("site_effect", np.nan)) else ""
        axB.text(x_info, yc, f"{enz}  ·  {amp_txt}  ·  {eff}",
                 ha="left", va="center", fontsize=8.7, color=PAL["slate"])

        # compact QC badges (stay inside bounds)
        bw = 0.014
        bh = row_h * 0.55
        x0 = x_badges
        for k, (_, col, fc, ec) in enumerate(QC):
            if bool(r.get(col, False)):
                axB.add_patch(Rectangle((x0 + k * (bw + 0.006), yc - bh / 2),
                                        bw, bh, facecolor=fc, edgecolor=ec, lw=0.7))

    # QC legend INSIDE panel B at bottom-right (no overflow)
    leg_y = 0.025
    axB.text(0.60, leg_y, "QC flags:", ha="left", va="center", fontsize=9, fontweight="bold", color=PAL["ink"])
    lx = 0.70
    for name, _, fc, ec in QC:
        axB.add_patch(Rectangle((lx, leg_y - 0.012), 0.016, 0.022, facecolor=fc, edgecolor=ec, lw=0.8))
        axB.text(lx + 0.022, leg_y, name, ha="left", va="center", fontsize=8.2, color=PAL["ink"])
        lx += 0.11  # spacing tuned to fit

    # ---------- Panel C ----------
    axC = fig.add_subplot(right[0, 0])
    add_panel_label(axC, "C")

    sub = df[df["site_effect_norm"] != ""].copy()
    if len(sub) == 0:
        axC.text(0.5, 0.5, "No site-effect data", ha="center", va="center")
        axC.set_axis_off()
    else:
        counts = (sub.groupby(["gene", "site_effect_norm"]).size()
                    .unstack(fill_value=0)
                    .reindex(index=["S", "ORF1ab", "N"], fill_value=0))
        genes_present = [g for g in ["S", "ORF1ab", "N"] if g in counts.index]
        x = np.arange(len(genes_present))
        gain = counts.get("gain", pd.Series(0, index=counts.index)).loc[genes_present].values
        loss = counts.get("loss", pd.Series(0, index=counts.index)).loc[genes_present].values
        amb  = counts.get("ambiguous", pd.Series(0, index=counts.index)).loc[genes_present].values

        axC.bar(x, gain, color=PAL["teal"], label="gain")
        axC.bar(x, loss, bottom=gain, color=PAL["coral"], label="loss")
        axC.bar(x, amb, bottom=gain + loss, color=PAL["gold"], label="ambiguous")

        axC.set_xticks(x, genes_present)
        axC.set_ylabel("Loci (n)", fontsize=9, labelpad=2)
        axC.set_title("Digest-site effect", pad=6, fontsize=10)
        axC.legend(frameon=False, fontsize=8, loc="upper right")
        axC.grid(axis="y", color=PAL["grid"], lw=0.9)

    # ---------- Panel D ----------
    axD = fig.add_subplot(right[1, 0])
    add_panel_label(axD, "D")

    df_num = df.copy()
    df_num["amp"] = pd.to_numeric(df_num["amplicon_bp"], errors="coerce")

    def frag_count(alts):
        if not alts:
            return np.nan
        return len(alts[0])

    df_num["wt_nfrag"] = df_num["wt_alts"].apply(frag_count)
    df_num["mut_nfrag"] = df_num["mut_alts"].apply(frag_count)

    x1 = df_num["amp"].values.astype(float)
    y1 = df_num["wt_nfrag"].values.astype(float)
    x2 = df_num["amp"].values.astype(float)
    y2 = df_num["mut_nfrag"].values.astype(float)

    rng = np.random.default_rng(7)
    xj1 = x1 + rng.normal(0, 2.0, size=len(x1))
    xj2 = x2 + rng.normal(0, 2.0, size=len(x2))
    yj1 = y1 + rng.normal(0, 0.03, size=len(y1))
    yj2 = y2 + rng.normal(0, 0.03, size=len(y2))

    # subtle density cue via binned contour (no alpha)
    valid = np.isfinite(x1) & np.isfinite(y1)
    if valid.sum() >= 5:
        H, xe, ye = np.histogram2d(x1[valid], y1[valid], bins=[14, 6])
        xc = (xe[:-1] + xe[1:]) / 2
        yc = (ye[:-1] + ye[1:]) / 2
        X, Y = np.meshgrid(xc, yc, indexing="xy")
        # draw low-level contours only if non-zero
        if np.max(H) > 0:
            levels = sorted(set([1, 2, 3, int(np.max(H))]))
            axD.contour(X, Y, H.T, levels=levels, colors=[PAL["slate"]], linewidths=0.9)

    axD.scatter(xj1, yj1, s=30, marker="o", color=PAL["midnight"], label="WT")
    axD.scatter(xj2, yj2, s=32, marker="^", color=PAL["violet"], label="Mut")

    axD.set_xlabel("Amplicon (bp)", fontsize=9, labelpad=2)
    axD.set_ylabel("# fragments (primary)", fontsize=9, labelpad=2)
    axD.set_title("Amplicon vs fragment count", pad=6, fontsize=10)
    axD.grid(True, color=PAL["grid"], lw=0.9)
    axD.legend(frameon=False, loc="upper left", fontsize=8, handletextpad=0.4, borderaxespad=0.2)

    # ---------- Panel E + F (side-by-side) ----------
    subEF = GridSpecFromSubplotSpec(1, 2, subplot_spec=right[2, 0], width_ratios=[1.95, 1.05], wspace=0.38)

    axE = fig.add_subplot(subEF[0, 0])
    add_panel_label(axE, "E")

    # Flag-only locus list (prevents label crowding)
    risk_cols = ["flag_pending", "flag_ambiguous", "flag_sum_mismatch", "flag_small_frag"]
    risk_names = ["Pending/NA", "Ambiguous", "Sum mismatch", "<50 bp"]
    flag_df = df.copy()
    flag_df["any_flag"] = flag_df[risk_cols].any(axis=1)
    flag_df = flag_df[flag_df["any_flag"]].reset_index(drop=True)

    axE.set_title("QC flags", pad=4, fontsize=10)
    axE.set_xlim(0, len(risk_cols))
    axE.set_ylim(0, max(len(flag_df), 1))
    axE.invert_yaxis()
    axE.set_xticks(np.arange(len(risk_cols)) + 0.5)
    # add counts directly in xtick labels
    counts_all = df[risk_cols].sum(axis=0).astype(int).tolist()
    axE.set_xticklabels([f"{nm}\n(n={c})" for nm, c in zip(["Pending","Ambig","Sum mis","<50bp"], counts_all)], fontsize=7)
    axE.tick_params(axis="x", pad=6)
    for lab in axE.get_xticklabels(): lab.set_rotation(0); lab.set_ha("center")
    axE.set_yticks(np.arange(len(flag_df)) + 0.5)

    if len(flag_df) == 0:
        axE.set_yticklabels([])
        axE.text(0.5, 0.5, "No QC flags triggered", transform=axE.transAxes,
                 ha="center", va="center", fontsize=9, color=PAL["slate"])
    else:
        fs_y = 8 if len(flag_df) <= 10 else (7 if len(flag_df) <= 14 else 6)
        axE.set_yticklabels(flag_df["locus"].tolist(), fontsize=fs_y)

        # draw matrix cells with per-flag colors
        for i in range(len(flag_df)):
            for j, (nm, col, fc, ec) in enumerate(QC):
                v = bool(flag_df.loc[i, col])
                if v:
                    axE.add_patch(Rectangle((j, i), 1, 1, facecolor=fc, edgecolor=ec, lw=0.6))
                else:
                    axE.add_patch(Rectangle((j, i), 1, 1, facecolor="white", edgecolor=(0.85, 0.88, 0.92), lw=0.6))

        # outer border
        axE.add_patch(Rectangle((0, 0), len(risk_cols), len(flag_df),
                                facecolor="none", edgecolor=PAL["border"], lw=1.0))

    axE.tick_params(axis="x", bottom=False, top=False)
    axE.tick_params(axis="y", left=False)

    axF = fig.add_subplot(subEF[0, 1])
    add_panel_label(axF, "F")
    axF.set_axis_off()
    axF.set_title("Gel inset", pad=4, fontsize=10)

    if gel_img is None:
        axF.add_patch(Rectangle((0.06, 0.12), 0.88, 0.76, transform=axF.transAxes,
                                facecolor=(0.98, 0.98, 0.98), edgecolor=PAL["border"], lw=1.0))
        axF.text(0.5, 0.5, "Provide --gel for\none burnished gel panel",
                 transform=axF.transAxes, ha="center", va="center", fontsize=9, color=PAL["slate"])
    else:
        axF.imshow(gel_img)
        axF.add_patch(Rectangle((0, 0), 1, 1, transform=axF.transAxes,
                                facecolor="none", edgecolor=PAL["border"], lw=1.2))

    # Final spacing polish (prevents panel-title collisions when saving with tight bbox)
    fig.subplots_adjust(top=0.93, bottom=0.07, left=0.05, right=0.98)
    # Footer note
    fig.text(0.02, 0.012,
             "Panel B synthetic lanes are computed from expected fragment sizes (assay_table). "
             "Ambiguity (*) denotes alternative signatures reported in the thesis.",
             ha="left", va="bottom", fontsize=8, color=PAL["slate"])

    # Save
    safe_mkdir(outdir)
    out_base = os.path.join(outdir, "Figure_3_RFLP_Signature_Atlas")
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
    ap.add_argument("--assay_csv", required=True, help="Path to assay_table.csv")
    ap.add_argument("--gel", default="", help="Path to ONE representative burnished gel image (png/jpg). Optional.")
    ap.add_argument("--outdir", required=True, help="Output directory (e.g., E:/.../Final Figure/Fig_3)")
    ap.add_argument("--title", default="PCR–RFLP Signature Atlas", help="Figure header title")
    args = ap.parse_args()

    make_figure(args.assay_csv, args.gel, args.outdir, args.title)


if __name__ == "__main__":
    main()