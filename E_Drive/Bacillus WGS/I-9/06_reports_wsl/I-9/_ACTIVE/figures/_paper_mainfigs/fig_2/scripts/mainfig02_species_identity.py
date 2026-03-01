#!/usr/bin/env python3
from __future__ import annotations

import os
import csv
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch

# ----------------------------
# Config
# ----------------------------
I9 = Path(os.environ.get("I9", "/mnt/e/Bacillus WGS/bacillus_paranthracis/06_reports_wsl/I-9"))
OUTDIR = Path(os.environ.get("OUTDIR", str(I9 / "figures" / "_paper_mainfigs" / "fig_2")))
OUTDIR.mkdir(parents=True, exist_ok=True)

REFCHECK = I9 / "qc_clean" / "refcheck"
IN_BAC = REFCHECK / "ANI_manualrefs_baconly_clean.tsv"
IN_ALL = REFCHECK / "ANI_manualrefs_allcontigs_clean.tsv"

KRAKEN_READ = I9 / "I-9.reads.kraken2.report.txt"
KRAKEN_CONTIG = I9 / "I-9.contigs.kraken2.report.txt"
KRAKEN_CONTIG_MIN5 = I9 / "I-9.contigs.min5kb.kraken2.report.txt"

OUT_PDF = OUTDIR / "MainFig02_Species_Identity.pdf"
OUT_SVG = OUTDIR / "MainFig02_Species_Identity.svg"
OUT_PNG = OUTDIR / "MainFig02_Species_Identity.png"
OUT_JPG = OUTDIR / "MainFig02_Species_Identity.jpg"
OUT_EPS = OUTDIR / "MainFig02_Species_Identity.eps"
OUT_TSV = OUTDIR / "MainFig02_Species_Identity.tsv"

# ----------------------------
# Style (premium / clean)
# ----------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

COL_BAC = "#2563eb"
COL_ALL = "#60a5fa"
COL_EDGE = "#0f172a"
COL_GRID = "#e5e7eb"
COL_NOTE = "#6b7280"
COL_LINE = "#c7d2fe"

COL_OK_BG = "#dcfce7"
COL_NA_BG = "#f3f4f6"
COL_OK = "#16a34a"
COL_NA = "#9ca3af"

MARK = {"GCA": "o", "GCF": "s", "OTHER": "D"}

def to_float(x: str) -> Optional[float]:
    x = (x or "").strip()
    if not x:
        return None
    try:
        return float(x)
    except Exception:
        return None

def acc_type(acc: str) -> str:
    acc = (acc or "").strip()
    if acc.startswith("GCA_"): return "GCA"
    if acc.startswith("GCF_"): return "GCF"
    return "OTHER"

def nice_species(s: str) -> str:
    s = (s or "").strip().replace("_", " ")
    if not s:
        return "NA"
    sl = s.lower()
    if sl.startswith("bacillus "):
        return "B. " + s.split(" ", 1)[1]
    if sl.startswith("enterobacter "):
        return "E. " + s.split(" ", 1)[1]
    return s[:1].upper() + s[1:]

def add_panel_label(ax, letter: str):
    ax.text(
        0.01, 0.98, letter, transform=ax.transAxes,
        va="top", ha="left", fontsize=10, fontweight="bold", color="white",
        bbox=dict(boxstyle="round,pad=0.25", fc="#111827", ec="#111827")
    )

@dataclass
class RefRow:
    ref_folder: str
    accession: str
    ani: float
    af_ref: float
    af_query: float

def read_refcheck(path: Path) -> List[RefRow]:
    rows: List[RefRow] = []
    if not path.exists():
        return rows
    with path.open("r", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        for d in r:
            if (d.get("status") or "").strip() != "OK":
                continue
            ani = to_float(d.get("ani"))
            afr = to_float(d.get("af_ref"))
            afq = to_float(d.get("af_query"))
            if ani is None or afr is None or afq is None:
                continue
            rows.append(RefRow(
                ref_folder=(d.get("ref_folder") or "NA").strip(),
                accession=(d.get("accession") or "NA").strip(),
                ani=float(ani),
                af_ref=float(afr),
                af_query=float(afq),
            ))
    rows.sort(key=lambda x: (x.ani, x.af_query), reverse=True)
    return rows

def parse_kraken_top_species(report_path: Path) -> Tuple[str, Optional[float], Optional[float]]:
    if not report_path.exists():
        return ("NA", None, None)

    unclassified = None
    best_name = None
    best_pct = None

    with report_path.open("r", errors="ignore") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            try:
                pct = float(parts[0].strip())
            except Exception:
                continue
            rank = parts[3].strip()
            name = parts[5].strip()

            if name.lower() == "unclassified":
                unclassified = pct
                continue

            if rank.startswith("S"):
                if best_pct is None or pct > best_pct:
                    best_pct = pct
                    best_name = name

    if best_name is None:
        return ("NA", None, unclassified)
    return (nice_species(best_name), best_pct, unclassified)

def write_summary_tsv(path: Path, d: Dict[str, str]):
    with path.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["field", "value"])
        for k, v in d.items():
            w.writerow([k, v])

def save_all(fig):
    fig.savefig(OUT_PDF, facecolor="white", bbox_inches="tight", pad_inches=0.18)
    fig.savefig(OUT_SVG, facecolor="white", bbox_inches="tight", pad_inches=0.18)
    fig.savefig(OUT_PNG, dpi=600, facecolor="white", bbox_inches="tight", pad_inches=0.18)
    fig.savefig(OUT_EPS, facecolor="white", bbox_inches="tight", pad_inches=0.18)

    try:
        from PIL import Image
        im = Image.open(OUT_PNG).convert("RGB")
        im.save(OUT_JPG, quality=95, optimize=True)
    except Exception:
        fig.savefig(OUT_JPG, dpi=600, facecolor="white", bbox_inches="tight", pad_inches=0.18)

def paired_by_species(rows: List[RefRow]) -> Dict[str, Dict[str, RefRow]]:
    m: Dict[str, Dict[str, RefRow]] = {}
    for r in rows:
        m.setdefault(r.ref_folder, {})
        m[r.ref_folder][acc_type(r.accession)] = r
    return m

def _auto_callout_xytext(best: RefRow) -> Tuple[float, float]:
    x = best.af_query / 100.0
    y = (best.ani - 80.0) / 20.0
    xt = 0.05 if x > 0.55 else 0.62
    yt = 0.92 if y < 0.75 else 0.78
    return (xt, yt)

def draw_refcheck_scatter(ax, rows: List[RefRow], title: str):
    ax.set_title(title)
    ax.grid(True, color=COL_GRID, linewidth=1)
    ax.set_xlim(0, 100)
    ax.set_ylim(80, 100)
    ax.set_xlabel("AF_query (%)")
    ax.set_ylabel("ANI (%)")

    for spine in ax.spines.values():
        spine.set_color("#334155")

    if not rows:
        ax.text(0.5, 0.5, "No OK rows", transform=ax.transAxes, ha="center", va="center")
        return

    folders = sorted({r.ref_folder for r in rows})
    pal = [COL_ALL, COL_BAC, "#1d4ed8", "#0ea5e9", "#64748b", "#94a3b8"]
    c_map = {f: pal[i % len(pal)] for i, f in enumerate(folders)}

    pairs = paired_by_species(rows)
    for sp, d in pairs.items():
        if "GCA" in d and "GCF" in d:
            ax.plot([d["GCA"].af_query, d["GCF"].af_query],
                    [d["GCA"].ani, d["GCF"].ani],
                    color=COL_LINE, linewidth=2.0, zorder=1)

    for r in rows:
        m = MARK.get(acc_type(r.accession), "D")
        size = 80 + r.af_ref * 0.6
        ax.scatter(r.af_query, r.ani, s=size, marker=m,
                   c=c_map.get(r.ref_folder, COL_ALL),
                   edgecolors=COL_EDGE, linewidths=1.0, zorder=6)

    h, lab = [], []
    for k in ["GCA", "GCF"]:
        h.append(ax.scatter([], [], s=95, marker=MARK[k], c="#cbd5e1",
                            edgecolors=COL_EDGE, linewidths=1.0))
        lab.append(k)
    ax.legend(h, lab, title="Accession", loc="lower right", frameon=True, fontsize=9, title_fontsize=9)

    best = rows[0]
    txt = f"{nice_species(best.ref_folder)}\n{best.accession}\nANI {best.ani:.2f} | AFq {best.af_query:.2f}"
    xt, yt = _auto_callout_xytext(best)

    ax.annotate(
        txt,
        xy=(best.af_query, best.ani), xycoords="data",
        xytext=(xt, yt), textcoords="axes fraction",
        ha="left", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#94a3b8"),
        arrowprops=dict(arrowstyle="-", color="#94a3b8", lw=1),
        annotation_clip=False,
        zorder=2
    )

def _badge(ax, y_ax: float, label: str, val: float, color: str):
    # Overlap-proof stacked badge inside axes
    x_dot = 0.02
    x_txt = 0.06
    ax.scatter([x_dot], [y_ax], transform=ax.transAxes, s=140,
               c=color, edgecolors=COL_EDGE, linewidths=1.1, zorder=10, clip_on=False)
    ax.text(x_txt, y_ax, f"{label}: {val:.2f}", transform=ax.transAxes,
            ha="left", va="center", fontsize=10, fontweight="bold",
            color="#111827", zorder=10)

def draw_modern_dumbbell(ax_ani, ax_af, bac: List[RefRow], allc: List[RefRow]):
    ax_ani.set_title("C1. Robustness: Top-hit ANI (bac-only → all contigs)")
    ax_af.set_title("C2. Robustness: Top-hit AF_query (bac-only → all contigs)")

    for ax in (ax_ani, ax_af):
        ax.grid(True, axis="x", color=COL_GRID, linewidth=1)
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#334155")

    if not bac or not allc:
        ax_ani.text(0.5, 0.5, "Missing refcheck rows", transform=ax_ani.transAxes, ha="center", va="center")
        ax_af.text(0.5, 0.5, "Missing refcheck rows", transform=ax_af.transAxes, ha="center", va="center")
        return

    ani_b, ani_a = bac[0].ani, allc[0].ani
    af_b, af_a = bac[0].af_query, allc[0].af_query

    # --- ANI dumbbell (NO numeric labels on dots; badges instead => overlap-proof) ---
    ax_ani.set_xlim(80, 101)
    y = 0
    ax_ani.plot([ani_b, ani_a], [y, y], color="#94a3b8", lw=3, zorder=1)
    ax_ani.scatter([ani_b], [y], s=260, c=COL_BAC, edgecolors=COL_EDGE, linewidths=1.1, zorder=6)
    ax_ani.scatter([ani_a], [y], s=260, c=COL_ALL, edgecolors=COL_EDGE, linewidths=1.1, zorder=6)
    ax_ani.set_xlabel("ANI (%)")
    ax_ani.set_ylim(-0.8, 0.8)

    # stacked badges (guaranteed no overlap)
    _badge(ax_ani, 0.82, "Bac-only", ani_b, COL_BAC)
    _badge(ax_ani, 0.64, "All contigs", ani_a, COL_ALL)

    # --- AF dumbbell (also badges; remove legend for cleaner premium look) ---
    ax_af.set_xlim(0, 103)
    y = 0
    ax_af.plot([af_b, af_a], [y, y], color="#94a3b8", lw=3, zorder=1)
    ax_af.scatter([af_b], [y], s=260, c=COL_BAC, edgecolors=COL_EDGE, linewidths=1.1, zorder=6)
    ax_af.scatter([af_a], [y], s=260, c=COL_ALL, edgecolors=COL_EDGE, linewidths=1.1, zorder=6)
    ax_af.set_xlabel("AF_query (%)")
    ax_af.set_ylim(-0.8, 0.8)

    _badge(ax_af, 0.82, "Bac-only", af_b, COL_BAC)
    _badge(ax_af, 0.64, "All contigs", af_a, COL_ALL)

def draw_scoreboard(ax, primary: str, reads: str, contigs: str):
    ax.axis("off")
    ax.set_title("D. Consistency scoreboard (primary ANI vs Kraken top labels)", pad=10)

    x0, y0 = 0.02, 0.18
    w_method, w_label, w_agree = 0.30, 0.52, 0.14
    row_h = 0.22

    ax.text(x0, y0 + 3*row_h, "Method", weight="bold", color="#111827")
    ax.text(x0 + w_method, y0 + 3*row_h, "Top label", weight="bold", color="#111827")
    ax.text(x0 + w_method + w_label, y0 + 3*row_h, "Agree", weight="bold", color="#111827")

    def pill(x, y, w, h, text, bg, fg):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                           fc=bg, ec=COL_GRID, linewidth=1)
        ax.add_patch(p)
        ax.text(x+w/2, y+h/2, text, ha="center", va="center",
                fontsize=12, color=fg, fontweight="bold")

    def row(i, m, lab, agree: Optional[bool]):
        y = y0 + (2 - i) * row_h
        box = FancyBboxPatch((x0, y), w_method+w_label+w_agree, row_h*0.85,
                             boxstyle="round,pad=0.02", fc="white", ec=COL_GRID, linewidth=1)
        ax.add_patch(box)
        ax.text(x0+0.01, y+row_h*0.42, m, va="center", fontsize=9)
        ax.text(x0+w_method+0.01, y+row_h*0.42, lab, va="center", fontsize=9)

        px = x0 + w_method + w_label + 0.02
        pw = w_agree - 0.04
        py = y + row_h*0.18
        ph = row_h*0.50

        if agree is True:
            pill(px, py, pw, ph, "✓", COL_OK_BG, COL_OK)
        elif agree is False:
            pill(px, py, pw, ph, "✗", "#fee2e2", "#dc2626")
        else:
            pill(px, py, pw, ph, "—", COL_NA_BG, COL_NA)

    reads_agree = None if reads == "NA" else (primary != "NA" and reads == primary)
    contigs_agree = None if contigs == "NA" else (primary != "NA" and contigs == primary)

    row(0, "Primary (ANI refcheck)", primary, None)
    row(1, "Reads (Kraken2 top species)", reads, reads_agree)
    row(2, "Contigs (Kraken2 top species)", contigs, contigs_agree)

def main():
    bac = read_refcheck(IN_BAC)
    allc = read_refcheck(IN_ALL)

    reads_sp, reads_pct, reads_uncl = parse_kraken_top_species(KRAKEN_READ)
    cont_src = KRAKEN_CONTIG_MIN5 if KRAKEN_CONTIG_MIN5.exists() else KRAKEN_CONTIG
    cont_sp, cont_pct, cont_uncl = parse_kraken_top_species(cont_src)

    primary = nice_species(bac[0].ref_folder) if bac else "NA"

    summary = {
        "primary_species_ANI": primary,
        "reads_top_species": reads_sp,
        "reads_top_pct": "NA" if reads_pct is None else f"{reads_pct:.4f}",
        "reads_unclassified_pct": "NA" if reads_uncl is None else f"{reads_uncl:.4f}",
        "contigs_top_species_min5kb": cont_sp,
        "contigs_top_pct_min5kb": "NA" if cont_pct is None else f"{cont_pct:.4f}",
        "contigs_unclassified_pct_min5kb": "NA" if cont_uncl is None else f"{cont_uncl:.4f}",
        "ani_bac_only": "NA" if not bac else f"{bac[0].ani:.4f}",
        "afq_bac_only": "NA" if not bac else f"{bac[0].af_query:.4f}",
        "ani_all_contigs": "NA" if not allc else f"{allc[0].ani:.4f}",
        "afq_all_contigs": "NA" if not allc else f"{allc[0].af_query:.4f}",
        "best_accession_bac_only": "NA" if not bac else bac[0].accession,
        "best_accession_all_contigs": "NA" if not allc else allc[0].accession,
    }
    write_summary_tsv(OUT_TSV, summary)

    fig = plt.figure(figsize=(18, 10), dpi=250)
    fig.patch.set_facecolor("white")

    gs = GridSpec(3, 10, figure=fig,
                  height_ratios=[1.05, 0.80, 0.75],
                  wspace=0.90, hspace=1.05)

    axA = fig.add_subplot(gs[0, 0:5])
    axB = fig.add_subplot(gs[0, 5:10])
    axC1 = fig.add_subplot(gs[1, 0:5])
    axC2 = fig.add_subplot(gs[1, 5:10])
    axD = fig.add_subplot(gs[2, 0:10])

    draw_refcheck_scatter(axA, bac, "A. Refcheck scatter: Bacillus-only contigs (GCA+GCF pairs)")
    draw_refcheck_scatter(axB, allc, "B. Refcheck scatter: All contigs (GCA+GCF pairs)")
    add_panel_label(axA, "A")
    add_panel_label(axB, "B")

    draw_modern_dumbbell(axC1, axC2, bac, allc)
    add_panel_label(axC1, "C")

    draw_scoreboard(axD, primary=primary, reads=reads_sp, contigs=cont_sp)
    add_panel_label(axD, "D")

    fig.suptitle(
        "Main Figure 2 — Species confirmation + reproducibility (A–D)\n"
        "ANI refcheck (GCA+GCF pairs) + robustness (bac-only vs all contigs) + compact consistency scoreboard",
        fontsize=16, fontweight="bold", y=0.98
    )

    # Save
    from matplotlib import pyplot as _plt
    fig.savefig(OUT_PDF, facecolor="white", bbox_inches="tight", pad_inches=0.18)
    fig.savefig(OUT_SVG, facecolor="white", bbox_inches="tight", pad_inches=0.18)
    fig.savefig(OUT_PNG, dpi=600, facecolor="white", bbox_inches="tight", pad_inches=0.18)
    fig.savefig(OUT_EPS, facecolor="white", bbox_inches="tight", pad_inches=0.18)
    try:
        from PIL import Image
        Image.open(OUT_PNG).convert("RGB").save(OUT_JPG, quality=95, optimize=True)
    except Exception:
        fig.savefig(OUT_JPG, dpi=600, facecolor="white", bbox_inches="tight", pad_inches=0.18)
    _plt.close(fig)

    print(f"Wrote: {OUT_PDF}")
    print(f"Wrote: {OUT_SVG}")
    print(f"Wrote: {OUT_PNG}")
    print(f"Wrote: {OUT_JPG}")
    print(f"Wrote: {OUT_EPS}")
    print(f"Wrote: {OUT_TSV}")

if __name__ == "__main__":
    main()
