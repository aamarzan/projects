#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 7 — Decision flow + QC thresholds (A–D)
Boss-level v5 (NO EPS): PDF + PNG + JPG

Fixes vs v4 (your zoomed D panel):
1) Panel D:
   - All text now fits inside the panel (no right-edge cropping):
     * Title wrapped to two lines and left-aligned.
     * Shorter descriptions + tighter wrapping width.
   - Reporting safety box moved DOWN and LEFT, and resized so it never overlaps "Invalid".
   - Increased vertical spacing between legend items.
2) Panel C:
   - Kept: visible "C" label and title safe zone; gel image drawn in inset axes.

Other panels unchanged. Truth source: thesis-derived assay_table.csv (expected fragment sizes).
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, Polygon
from PIL import Image
import textwrap as _tw

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

PAL = {
    "paper": (1.00, 1.00, 1.00),
    "panel_bg": (0.98, 0.985, 0.995),
    "ink": (0.07, 0.09, 0.12),
    "slate": (0.38, 0.42, 0.50),
    "grid": (0.86, 0.88, 0.92),
    "spine": (0.75, 0.78, 0.85),

    # Header (light)
    "header_l1": (0.90, 0.96, 1.00),
    "header_l2": (0.985, 0.995, 1.00),
    "header_text": (0.07, 0.22, 0.44),

    # Fig7 palette
    "warn": (0.91, 0.73, 0.18),       # amber
    "bad": (0.91, 0.38, 0.33),        # coral
    "mut": (0.76, 0.41, 0.83),        # orchid
    "wt": (0.10, 0.68, 0.76),         # teal
}

ARROW_LW = 1.85  # premium thickness

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def tidy_axes(ax):
    ax.set_facecolor(PAL["panel_bg"])
    for sp in ax.spines.values():
        sp.set_color(PAL["spine"])
        sp.set_linewidth(0.8)

def panel_label(ax, letter, inside=False, z=10):
    if inside:
        ax.text(0.000, 0.985, letter, transform=ax.transAxes,
                ha="left", va="top", fontsize=12, fontweight="bold",
                color=PAL["ink"], zorder=z)
    else:
        ax.text(0.008, 1.01, letter, transform=ax.transAxes,
                ha="left", va="bottom", fontsize=12, fontweight="bold",
                color=PAL["ink"], zorder=z)

def draw_header(ax, title, subtitle):
    grad = np.linspace(0, 1, 240)
    c1 = np.array(PAL["header_l1"]); c2 = np.array(PAL["header_l2"])
    rgb = (1 - grad)[:, None] * c1[None, :] + grad[:, None] * c2[None, :]
    img = np.tile(rgb[None, :, :], (2, 1, 1))
    ax.imshow(img, extent=[0, 1, 0, 1], transform=ax.transAxes,
              aspect="auto", interpolation="bicubic", zorder=0)
    ax.set_axis_off()
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

def compute_qc_from_assay(df: pd.DataFrame, tau: float):
    rows = []
    for _, r in df.iterrows():
        amp = safe_int(r.get("amplicon_bp", None))
        wt = first_or_empty(parse_fragments(r.get("wt_fragments_bp", "")))
        mut = first_or_empty(parse_fragments(r.get("mut_fragments_bp", "")))

        ms = min(min_band_spacing(wt), min_band_spacing(mut))
        cand = []
        if wt: cand.append(min(wt))
        if mut: cand.append(min(mut))
        mf = float(min(cand)) if cand else float("nan")

        sum_mismatch = False
        if amp is not None:
            if wt and len(wt) > 1 and sum(wt) != amp:
                sum_mismatch = True
            if mut and len(mut) > 1 and sum(mut) != amp:
                sum_mismatch = True

        rows.append({
            "locus": str(r.get("locus", "")),
            "min_spacing": ms if np.isfinite(ms) else np.nan,
            "min_frag": mf if np.isfinite(mf) else np.nan,
            "sum_mismatch": bool(sum_mismatch),
        })

    met = pd.DataFrame(rows)
    n_total = int(met["locus"].nunique())
    high = met[(met["min_spacing"].notna() & (met["min_spacing"] <= tau)) | (met["min_frag"].notna() & (met["min_frag"] < 50))]
    caution = met[(met["min_spacing"].notna() & (met["min_spacing"] <= 2*tau)) | (met["min_frag"].notna() & (met["min_frag"] < 100))]
    std = met[~met.index.isin(caution.index)]

    return {
        "total_loci": n_total,
        "high_caution_loci": int(high["locus"].nunique()),
        "caution_loci": int(caution["locus"].nunique()),
        "standard_loci": int(std["locus"].nunique()),
        "sum_mismatch_loci": int(met[met["sum_mismatch"]]["locus"].nunique()),
    }

def box(ax, x, y, w, h, text, fc, ec, fontsize=9.6, bold=False):
    patch = FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0.012,rounding_size=0.02",
                           facecolor=fc, edgecolor=ec, linewidth=1.0)
    ax.add_patch(patch)
    ax.text(x + w/2, y + h/2, text,
            ha="center", va="center",
            fontsize=fontsize, color=PAL["ink"],
            fontweight=("bold" if bold else "normal"))

def diamond(ax, cx, cy, w, h, text, fc, ec, fontsize=9.3):
    pts = np.array([
        [cx, cy + h/2],
        [cx + w/2, cy],
        [cx, cy - h/2],
        [cx - w/2, cy],
    ])
    poly = Polygon(pts, closed=True, facecolor=fc, edgecolor=ec, linewidth=1.0)
    ax.add_patch(poly)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize, color=PAL["ink"])

def arrow(ax, x1, y1, x2, y2, color=None):
    if color is None:
        color = PAL["spine"]
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", lw=ARROW_LW, color=color,
                                shrinkA=0, shrinkB=0, mutation_scale=12))

def poly_arrow(ax, pts, color=None):
    if color is None:
        color = PAL["spine"]
    pts = np.asarray(pts, dtype=float)
    ax.plot(pts[:-1, 0], pts[:-1, 1], color=color, lw=ARROW_LW, solid_capstyle="round")
    x1, y1 = pts[-2]
    x2, y2 = pts[-1]
    arrow(ax, x1, y1, x2, y2, color=color)

def wrap(s, width):
    return "\n".join(_tw.wrap(s, width=width))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assay_csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--jpg_quality", type=int, default=92)
    ap.add_argument("--tau", type=float, default=10.0)
    ap.add_argument("--gel_percent", type=float, default=1.5)
    ap.add_argument("--ladder", default="100 bp ladder (TriDye 100 bp)")
    ap.add_argument("--gel_inset", default="", help="Optional path to a cleaned gel image (PNG/JPG).")
    args = ap.parse_args()

    df = pd.read_csv(args.assay_csv)
    df = df[df["locus"].notna()].copy()
    qc = compute_qc_from_assay(df, float(args.tau))

    fig = plt.figure(figsize=(18, 12), dpi=170)
    fig.patch.set_facecolor(PAL["paper"])
    gs = GridSpec(44, 32, figure=fig, wspace=1.20, hspace=1.35)

    ax_head = fig.add_subplot(gs[0:4, :])
    draw_header(ax_head,
                "Decision Flow + QC Thresholds",
                "Implementation and reporting safety: standardized calling • QC gates • ambiguity handling")

    # Layout (same as v4, C slightly wider; D wide)
    axA = fig.add_subplot(gs[6:42, 0:18])

    # shift B/C/D left (keeps same widths + keeps a 1-col spacer between C and D)
    axB = fig.add_subplot(gs[6:24, 18:30])
    axC = fig.add_subplot(gs[26:34, 18:23])   # C width unchanged
    axD = fig.add_subplot(gs[26:42, 24:30])   # D width unchanged (col 23 is spacer)

    # ---- A ----
    tidy_axes(axA); panel_label(axA, "A")
    axA.text(0.5, 1.01, "Interpretation workflow (PCR–RFLP call + reporting)",
             transform=axA.transAxes, ha="center", va="bottom", fontsize=11, color=PAL["ink"])
    axA.set_xlim(0, 1); axA.set_ylim(0, 1)
    axA.set_xticks([]); axA.set_yticks([])

    box(axA, 0.08, 0.86, 0.44, 0.09, "Input: digested PCR product\n(+ ladder + controls)",
        PAL["panel_bg"], PAL["spine"], bold=True)
    diamond(axA, 0.30, 0.73, 0.38, 0.12, "QC PASS?\n(ladder • controls • band quality)",
            (0.96, 0.98, 1.00), PAL["spine"])
    box(axA, 0.08, 0.57, 0.44, 0.09, "Match expected pattern?\n(WT vs Mut signature)",
        (0.96, 0.98, 1.00), PAL["spine"], bold=True)
    diamond(axA, 0.30, 0.45, 0.38, 0.12, "Unambiguous match?",
            (0.99, 0.98, 0.94), PAL["spine"])
    box(axA, 0.08, 0.28, 0.44, 0.09, "Call: WT or Mut\n(report locus + enzyme)",
        (0.93, 0.99, 0.97), PAL["spine"], bold=True)
    box(axA, 0.08, 0.12, 0.44, 0.11,
        "Call: Ambiguous\n(repeat digest / alt gel)\nIf merged/weak bands: confirm by sequencing",
        (1.00, 0.97, 0.93), PAL["spine"], bold=True)
    box(axA, 0.58, 0.66, 0.34, 0.10, "FAIL QC\n(re-run PCR/digest;\ncheck reagents)",
        (1.00, 0.94, 0.94), PAL["spine"], bold=True)
    box(axA, 0.58, 0.48, 0.34, 0.10, "If repeated FAIL:\nreport as 'invalid'\n(do not call)",
        (1.00, 0.94, 0.94), PAL["spine"], bold=True)

    arrow(axA, 0.30, 0.86, 0.30, 0.79)
    arrow(axA, 0.30, 0.67, 0.30, 0.66)
    axA.text(0.245, 0.705, "YES", fontsize=9, color=PAL["slate"], ha="right", va="center")
    arrow(axA, 0.49, 0.73, 0.58, 0.71)
    axA.text(0.535, 0.745, "NO", fontsize=9, color=PAL["slate"], ha="center", va="bottom")
    arrow(axA, 0.75, 0.66, 0.75, 0.58)

    arrow(axA, 0.30, 0.57, 0.30, 0.51)
    arrow(axA, 0.30, 0.39, 0.30, 0.37)
    axA.text(0.245, 0.415, "YES", fontsize=9, color=PAL["slate"], ha="right", va="center")
    poly_arrow(axA, [(0.49, 0.45), (0.55, 0.45), (0.55, 0.24), (0.30, 0.23)], color=PAL["spine"])
    axA.text(0.52, 0.465, "NO", fontsize=9, color=PAL["slate"], ha="center", va="bottom")

    # ---- B ----
    tidy_axes(axB); panel_label(axB, "B")
    axB.text(0.5, 1.01, "QC checkpoints (pass/fail) + assay-derived counts",
             transform=axB.transAxes, ha="center", va="bottom", fontsize=11, color=PAL["ink"])
    axB.axis("off")

    table_rows = [
        ("Ladder present", "Required", args.ladder),
        ("Gel condition", "Study protocol", f"{args.gel_percent:g}% agarose"),
        ("Controls", "Pass", "Positive + negative controls valid"),
        ("Band count", "Consistent", "No missing major bands"),
        ("Band spacing", f"High risk if ≤ {args.tau:g} bp", f"{qc['high_caution_loci']} loci flagged (physics)"),
        ("Visibility", "High risk if <50 bp", "Handled in physics risk score"),
        ("Caution zone", f"≤ {2*args.tau:g} bp or <100 bp", f"{qc['caution_loci']} loci need caution"),
        ("Sum check", "If applicable", f"{qc['sum_mismatch_loci']} loci sum≠amplicon"),
        ("Reporting", "If QC fail", "Do not call; re-run / invalid"),
    ]
    cell_text = [[a, b, c] for (a, b, c) in table_rows]
    col_labels = ["Checkpoint", "Threshold", "Action / Note"]

    tbl = axB.table(cellText=cell_text, colLabels=col_labels, loc="center",
                    cellLoc="left", colLoc="left", edges="closed")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.6)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(PAL["spine"])
        cell.set_linewidth(0.7)
        if r == 0:
            cell.set_facecolor((0.93, 0.97, 1.00))
            cell.set_text_props(color=PAL["ink"], fontweight="bold")
        else:
            cell.set_facecolor(PAL["panel_bg"])
            cell.set_text_props(color=PAL["ink"])
        if c == 0:
            cell.set_width(0.33)
        elif c == 1:
            cell.set_width(0.30)
        else:
            cell.set_width(0.37)
    tbl.scale(1.0, 1.35)

    # ---- C ----
    tidy_axes(axC)
    axC.set_facecolor((1, 1, 1))
    panel_label(axC, "C", inside=True, z=30)
    axC.text(0.58, 0.985, "Representative gel (inset)", transform=axC.transAxes,
             ha="center", va="top", fontsize=10.5, color=PAL["ink"], zorder=30)
    axC.set_xticks([]); axC.set_yticks([])

    img_ax = axC.inset_axes([0.08, 0.10, 0.84, 0.78])
    img_ax.set_xticks([]); img_ax.set_yticks([])
    img_ax.set_facecolor((1, 1, 1))
    for sp in img_ax.spines.values():
        sp.set_visible(False)

    gel_path = Path(args.gel_inset) if args.gel_inset else None
    if gel_path and gel_path.exists():
        im = Image.open(gel_path).convert("RGB")
        img_ax.imshow(im)
    else:
        img_ax.text(0.5, 0.5, "Gel inset (optional)\nProvide --gel_inset path",
                    ha="center", va="center", fontsize=8.6, color=PAL["slate"], transform=img_ax.transAxes)

    for sp in axC.spines.values():
        sp.set_color(PAL["spine"])
        sp.set_linewidth(0.9)

    # ---- D (FIXED) ----
    tidy_axes(axD)
    panel_label(axD, "D", inside=True, z=30)
    # title wrapped + left aligned (prevents right cropping)
    axD.text(0.06, 0.985, "Interpretation legend\n(calls + reporting)",
             transform=axD.transAxes, ha="left", va="top",
             fontsize=10.4, color=PAL["ink"], zorder=30)
    axD.axis("off")

    items = [
        ("WT call", "Bands match WT signature (within tolerance).", PAL["wt"]),
        ("Mut call", "Bands match Mut signature (within tolerance).", PAL["mut"]),
        ("Ambiguous", "Merged/weak bands OR high physics risk → repeat; sequence if needed.", PAL["warn"]),
        ("Invalid", "QC fail (ladder / control / digest) → no call.", PAL["bad"]),
    ]

    y = 0.82
    step = 0.20
    for name, desc, col in items:
        axD.add_patch(FancyBboxPatch((0.06, y-0.06), 0.18, 0.10,
                                     boxstyle="round,pad=0.01,rounding_size=0.02",
                                     facecolor=col, edgecolor=PAL["spine"], linewidth=1.0))
        axD.text(0.27, y-0.012, name, ha="left", va="center",
                 fontsize=9.4, fontweight="bold", color=PAL["ink"])
        axD.text(0.27, y-0.066, wrap(desc, 40), ha="left", va="top",
                 fontsize=8.1, color=PAL["slate"])
        y -= step

    # --- Reporting safety box (manual controls) ---
    rep_txt = ("Reporting safety: record locus, enzyme, expected sizes, and ladder. "
            "For ambiguous/invalid, document the reason (merged bands, weak bands, QC fail). "
            "If high-caution loci are important, prioritize confirmatory sequencing.")

    REP_X, REP_Y, REP_W, REP_H = 0.03, 0.000, 1.35, 0.075  # <-- TUNE THESE 4 NUMBERS
    TXT_X = REP_X + 0.02
    TXT_Y = REP_Y + REP_H - 0.01

    axD.add_patch(FancyBboxPatch((REP_X, REP_Y), REP_W, REP_H,
                                boxstyle="round,pad=0.012,rounding_size=0.02",
                                facecolor=(0.985, 0.99, 1.00),
                                edgecolor=PAL["spine"], linewidth=0.9))

    axD.text(TXT_X, TXT_Y, wrap(rep_txt, 66),  # wrap width can be tuned too
            ha="left", va="top", fontsize=7.6, color=PAL["ink"])

    # Footer
    fig.text(
        0.02, 0.02,
        "Figure 7 is implementation-focused: operational calling workflow + QC gates derived from expected fragments (assay_table). "
        "Protocol inputs (gel % / ladder) should match the thesis methods used in the study.",
        fontsize=8, color=PAL["slate"]
    )

    outdir = Path(args.outdir)
    ensure_dir(outdir)
    base = outdir / "Figure_7_Decision_Flow_QC"

    fig.subplots_adjust(left=0.05, right=0.995, top=0.965, bottom=0.070)

    fig.savefig(str(base) + ".pdf", facecolor=PAL["paper"])
    fig.savefig(str(base) + ".png", dpi=int(args.dpi), facecolor=PAL["paper"])
    fig.savefig(str(base) + ".jpg", dpi=int(args.dpi), facecolor=PAL["paper"],
                pil_kwargs={"quality": int(args.jpg_quality), "subsampling": 2})
    plt.close(fig)

if __name__ == "__main__":
    main()
