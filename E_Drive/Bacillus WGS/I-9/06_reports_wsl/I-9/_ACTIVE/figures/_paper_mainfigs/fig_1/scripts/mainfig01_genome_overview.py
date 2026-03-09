#!/usr/bin/env python3
from __future__ import annotations

import os
import csv
import math
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import NullFormatter


# ----------------------------
# Config (override by env vars)
# ----------------------------
I9 = Path(os.environ.get("I9", "/mnt/e/Bacillus WGS/bacillus_paranthracis/06_reports_wsl/I-9"))
OUTDIR = Path(os.environ.get("OUTDIR", "/mnt/e/Bacillus WGS/_paper_mainfigs/fig_1"))
PIPE_FIGDIR = I9 / "figures"  # optional copy-back

OUTDIR.mkdir(parents=True, exist_ok=True)

# Inputs
READS_REPORT = I9 / "I-9.reads.kraken2.report.txt"
# bp-weighted contig composition produced earlier by your pipeline
CONTIG_BP_TSV = I9 / "figures" / "Fig01_taxonomy_bpweighted_allcontigs.tsv"
QUAST_TSV = I9 / "quast_I-9" / "report.tsv"
FASTA = I9 / "I-9.final.contigs.fa"

# Outputs (paper folder)
OUT_PDF = OUTDIR / "MainFig01_Genome_Overview.pdf"
OUT_SVG = OUTDIR / "MainFig01_Genome_Overview.svg"
OUT_PNG = OUTDIR / "MainFig01_Genome_Overview.png"
OUT_JPG = OUTDIR / "MainFig01_Genome_Overview.jpg"
OUT_EPS = OUTDIR / "MainFig01_Genome_Overview.eps"

# -----------------------------------
# Global styling (journal-ish + clean)
# -----------------------------------
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#FBFCFF",
    "axes.edgecolor": "#20242A",
    "axes.linewidth": 0.8,
    "grid.color": "#3A4250",
    "grid.alpha": 0.12,
    "grid.linewidth": 0.8,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

PALETTE = [
    "#5B8FF9", "#5AD8A6", "#F6BD16", "#E8684A",
    "#6DC8EC", "#9270CA", "#FF99C3", "#9FE6B8",
]
COL_UNCLASS = "#C7CCD9"
COL_HIRANK = "#8E99AB"
COL_OTHER = "#AAB2C2"

def safe_float(x: str) -> float | None:
    try:
        x = (x or "").strip()
        if not x:
            return None
        return float(x)
    except Exception:
        return None

def panel_tag(ax, letter: str):
    ax.text(
        0.01, 0.98, letter,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=11, fontweight="bold",
        color="white",
        bbox=dict(boxstyle="round,pad=0.25", fc="#111827", ec="none", alpha=0.95),
        zorder=50,
    )

# ----------------------------
# FASTA: contig length + GC%
# ----------------------------
def iter_fasta_lengths_gc(fa: Path):
    name = None
    seq_len = 0
    gc = 0
    with fa.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None and seq_len > 0:
                    yield name, seq_len, (100.0 * gc / seq_len)
                name = line[1:].split()[0]
                seq_len = 0
                gc = 0
            else:
                s = line.upper()
                seq_len += len(s)
                gc += s.count("G") + s.count("C")
        if name is not None and seq_len > 0:
            yield name, seq_len, (100.0 * gc / seq_len)

def compute_nx(lengths_desc, x_list=(10,20,30,40,50,60,70,80,90)):
    total = sum(lengths_desc)
    nx = {}
    csum = 0
    idx = 0
    for x in x_list:
        target = total * (x/100.0)
        while idx < len(lengths_desc) and csum < target:
            csum += lengths_desc[idx]
            idx += 1
        nx[x] = lengths_desc[max(0, idx-1)] if lengths_desc else 0
    return nx

# ----------------------------
# Kraken2 report: read-level
# ----------------------------
def parse_kraken_report(report: Path, top_n=5):
    """
    Return categories (name -> percent) with:
      - top species
      - Other species
      - Higher-rank classified (classified non-species)
      - Unclassified
    Uses clade-reads for species lines to avoid double count.
    """
    if not report.exists():
        raise FileNotFoundError(f"Missing Kraken2 report: {report}")

    rows = []
    with report.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            pct = safe_float(parts[0])
            clade = int(parts[1])
            rank = parts[3].strip()
            taxid = parts[4].strip()
            name = parts[5].strip()
            rows.append((pct, clade, rank, taxid, name))

    # total reads (root)
    total = None
    for pct, clade, rank, taxid, name in rows:
        if rank == "R" or taxid == "1" or name.lower() == "root":
            total = clade
            break
    if total is None:
        total = max(r[1] for r in rows) if rows else 0

    # unclassified
    unclassified = 0
    for pct, clade, rank, taxid, name in rows:
        if rank == "U" or taxid == "0" or name.lower() == "unclassified":
            unclassified = clade
            break

    classified = max(0, total - unclassified)

    # species clade reads
    species = []
    for pct, clade, rank, taxid, name in rows:
        if rank == "S" and clade > 0:
            nm = name.strip()
            species.append((clade, nm))

    species.sort(reverse=True, key=lambda x: x[0])
    top = species[:top_n]
    top_sum = sum(v for v,_ in top)
    species_sum = sum(v for v,_ in species)

    other_species = max(0, species_sum - top_sum)
    higher_rank = max(0, classified - species_sum)

    # convert to percent
    out = []
    for v, nm in top:
        out.append((nm, 100.0 * v / total if total else 0))
    if other_species > 0:
        out.append(("Other species", 100.0 * other_species / total if total else 0))
    if higher_rank > 0:
        out.append(("Higher-rank classified", 100.0 * higher_rank / total if total else 0))
    out.append(("Unclassified", 100.0 * unclassified / total if total else 0))

    # keep stable ordering: top taxa..., Other..., Higher-rank..., Unclassified
    return out

# ---------------------------------------
# bp-weighted contig composition from TSV
# ---------------------------------------
def parse_bp_weighted_tsv(tsv: Path, top_n=5):
    """
    Expect a TSV with at least:
      - a taxon/name column
      - a percent column
    We'll auto-detect.
    """
    if not tsv.exists():
        # fallback: if missing, return 100% unclassified
        return [("Unclassified", 100.0)]

    with tsv.open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f, delimiter="\t")
        cols = reader.fieldnames or []
        # guess name col
        name_col = None
        for c in cols:
            lc = c.lower()
            if "tax" in lc or "name" in lc or "label" in lc:
                name_col = c
                break
        if name_col is None:
            name_col = cols[0] if cols else None

        # guess percent col
        pct_col = None
        for c in cols:
            lc = c.lower()
            if "percent" in lc or lc in {"pct", "p"} or "bp" in lc and "pct" in lc:
                pct_col = c
                break
        # if not found, take first numeric column
        rows = []
        for r in reader:
            if not r:
                continue
            nm = (r.get(name_col, "") if name_col else "").strip()
            if not nm:
                continue
            if pct_col:
                pct = safe_float(r.get(pct_col, ""))
            else:
                pct = None
                for c in cols[1:]:
                    v = safe_float(r.get(c, ""))
                    if v is not None:
                        pct = v
                        break
            if pct is None:
                continue
            rows.append((pct, nm))

    rows.sort(reverse=True, key=lambda x: x[0])

    top = rows[:top_n]
    top_sum = sum(p for p,_ in top)
    rest = max(0.0, 100.0 - top_sum)

    out = [(nm, pct) for pct, nm in top if pct > 0]
    if rest > 0.0:
        out.append(("Other / remaining", rest))

    # if a row called Unclassified exists in top, keep it last
    un = [x for x in out if x[0].lower().startswith("unclassified")]
    out = [x for x in out if x not in un]
    if un:
        out.extend(un)

    return out

# ----------------------------
# QUAST metrics (robust parse)
# ----------------------------
def parse_quast_report(report_tsv: Path):
    metrics = {
        "# contigs": None,
        "Total length (bp)": None,
        "Largest contig (bp)": None,
        "N50 (bp)": None,
        "L50": None,
        "GC (%)": None,
    }
    if not report_tsv.exists():
        return metrics

    # QUAST report.tsv: tab-separated with "Assembly" column and metrics in rows OR wide format.
    # We'll do a simple scan for known keys in first column.
    with report_tsv.open("r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]

    # If it's "metric\tvalue\t..." style:
    for ln in lines:
        parts = ln.split("\t")
        if len(parts) < 2:
            continue
        k = parts[0].strip()
        v = parts[1].strip()
        lk = k.lower()
        if lk == "# contigs":
            metrics["# contigs"] = v
        elif lk.startswith("total length"):
            metrics["Total length (bp)"] = v
        elif lk.startswith("largest contig"):
            metrics["Largest contig (bp)"] = v
        elif lk.startswith("n50"):
            metrics["N50 (bp)"] = v
        elif lk.startswith("l50"):
            metrics["L50"] = v
        elif lk.startswith("gc"):
            metrics["GC (%)"] = v

    return metrics

# ----------------------------
# Plot helpers
# ----------------------------
def stacked_barh(ax, items, xlabel, title):
    """
    items: list of (name, percent)
    """
    names = [n for n,_ in items]
    vals = [v for _,v in items]

    # color map
    color_map = {}
    pal_i = 0
    for n in names:
        ln = n.lower()
        if ln.startswith("unclassified"):
            color_map[n] = COL_UNCLASS
        elif "higher-rank" in ln:
            color_map[n] = COL_HIRANK
        elif "other" in ln:
            color_map[n] = COL_OTHER
        else:
            color_map[n] = PALETTE[pal_i % len(PALETTE)]
            pal_i += 1

    left = 0.0
    handles = []
    labels = []
    for n, v in zip(names, vals):
        if v <= 0:
            continue
        h = ax.barh([0], [v], left=left, height=0.45, color=color_map[n], edgecolor="white", linewidth=1.2)
        left += v
        handles.append(h[0])
        labels.append(n)

        # label only if big enough
        if v >= 8:
            ax.text(left - v/2, 0, f"{v:.1f}%", ha="center", va="center", fontsize=9, color="#111827")

    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xlabel(xlabel)
    ax.set_title(title, pad=8)
    ax.grid(True, axis="x")
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)

    return handles, labels

def annotate_points(ax, points, offsets):
    """
    points: list of (x, y, label)
    offsets: list of (dx, dy) in points
    """
    for i, (x, y, lab) in enumerate(points):
        dx, dy = offsets[i % len(offsets)]
        ax.scatter([x], [y], s=85, marker="*", edgecolor="#111827", linewidth=0.8, zorder=10)
        ax.annotate(
            lab,
            xy=(x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8,
            color="#111827",
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#D0D5DD", alpha=0.95),
            arrowprops=dict(arrowstyle="-", color="#667085", lw=0.8, alpha=0.8),
            zorder=11,
        )

# ----------------------------
# Main
# ----------------------------
def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # Data
    reads_items = parse_kraken_report(READS_REPORT, top_n=5)
    contig_items = parse_bp_weighted_tsv(CONTIG_BP_TSV, top_n=5)
    quast = parse_quast_report(QUAST_TSV)

    contig_ids = []
    lengths = []
    gcs = []
    for cid, L, gc in iter_fasta_lengths_gc(FASTA):
        contig_ids.append(cid)
        lengths.append(L)
        gcs.append(gc)

    if not lengths:
        raise RuntimeError(f"No contigs parsed from {FASTA}")

    total_len = sum(lengths)
    lengths_desc = sorted(lengths, reverse=True)
    nx = compute_nx(lengths_desc)

    # cumulative fraction curve (x ascending)
    lengths_asc = sorted(lengths)
    csum = 0
    x_curve = []
    y_curve = []
    for L in lengths_asc:
        csum += L
        x_curve.append(L)
        y_curve.append(csum / total_len if total_len else 0)

    # Choose outliers to label (limit to avoid overlap)
    # score = length_rank + GC deviation
    gc_med = sorted(gcs)[len(gcs)//2]
    scores = []
    for cid, L, gc in zip(contig_ids, lengths, gcs):
        dev = abs(gc - gc_med)
        score = (L / max(lengths)) * 1.2 + (dev / 10.0)
        scores.append((score, cid, L, gc, dev))
    scores.sort(reverse=True, key=lambda x: x[0])
    outliers = []
    for _, cid, L, gc, dev in scores:
        # pick only meaningful outliers
        if L >= (sorted(lengths)[int(0.98*len(lengths))]) or dev >= 7.0:
            outliers.append((L, gc, cid))
        if len(outliers) >= 7:
            break

    # --------------------------------
    # Layout: 4 rows (legend row added)
    # --------------------------------
    fig = plt.figure(figsize=(14.5, 9.5), dpi=250)
    gs = GridSpec(
        4, 2, figure=fig,
        height_ratios=[1.05, 0.34, 1.40, 1.85],
        wspace=0.25, hspace=0.55
    )

    # A/B
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])

    hA, lA = stacked_barh(axA, reads_items, "Percent of reads (%)", "Read-level taxonomy (Kraken2)")
    hB, lB = stacked_barh(axB, contig_items, "Percent of assembly length (%)", "Contig taxonomy (bp-weighted; Kraken2)")

    panel_tag(axA, "A")
    panel_tag(axB, "B")

    # Legend row (prevents overlap with axes)
    axLeg = fig.add_subplot(gs[1, :])
    axLeg.axis("off")
    # Merge handles/labels uniquely in order
    seen = set()
    handles = []
    labels = []
    for h, lab in list(zip(hA, lA)) + list(zip(hB, lB)):
        if lab in seen:
            continue
        seen.add(lab)
        handles.append(h)
        labels.append(lab)
    axLeg.legend(
        handles, labels, ncol=4, loc="center",
        frameon=False, fontsize=9, handlelength=1.8, columnspacing=1.6
    )

    # C (table)
    axC = fig.add_subplot(gs[2, 0])
    axC.axis("off")
    panel_tag(axC, "C")

    rows = [
        ("Assembly", FASTA.name),
        ("# contigs", quast.get("# contigs") or f"{len(lengths):,}"),
        ("Total length (bp)", quast.get("Total length (bp)") or f"{total_len:,}"),
        ("Largest contig (bp)", quast.get("Largest contig (bp)") or f"{max(lengths):,}"),
        ("N50 (bp)", quast.get("N50 (bp)") or "NA"),
        ("L50", quast.get("L50") or "NA"),
        ("GC (%)", quast.get("GC (%)") or f"{(sum(gcs)/len(gcs)):.2f}"),
    ]

    tbl = axC.table(
        cellText=[[k, v] for k, v in rows],
        colLabels=["Metric", "Value"],
        loc="center",
        cellLoc="left",
        colLoc="left",
        bbox=[0.00, 0.00, 1.00, 1.00],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.6)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#E5E7EB")
        if r == 0:
            cell.set_facecolor("#EEF2FF")
            cell.set_text_props(weight="bold", color="#111827")
        else:
            cell.set_facecolor("white" if r % 2 else "#FBFCFF")

    axC.set_title("Assembly key metrics (computed)", pad=10)

    # D (nested grid: continuity + Nx + histogram)
    sub = gs[2, 1].subgridspec(
        2, 2,
        width_ratios=[2.8, 1.25],
        height_ratios=[1.0, 1.0],
        wspace=0.42,
        hspace=0.72
    )
    axD = fig.add_subplot(sub[:, 0])     # span both rows
    axNx = fig.add_subplot(sub[0, 1])
    axH = fig.add_subplot(sub[1, 1])

    panel_tag(axD, "D")

    # continuity curve
    axD.plot(x_curve, y_curve, lw=2.8, color="#3B82F6")
    axD.set_xscale("log")
    axD.xaxis.set_minor_formatter(NullFormatter())
    axD.set_ylim(0, 1.02)
    axD.set_xlabel("Contig length (bp, log scale)", labelpad=6)
    axD.set_ylabel("Cumulative assembly fraction", labelpad=6)
    axD.grid(True, which="major")
    axD.set_title("Assembly continuity (cumulative fraction)", pad=8)

    # Nx curve
    xs = sorted(nx.keys())
    ys = [nx[x] for x in xs]
    axNx.plot(xs, ys, marker="o", lw=2.0, color="#8B5CF6")
    axNx.set_title("Nx curve", pad=6)
    axNx.set_xlabel("")  # remove to avoid crowding
    axNx.set_ylabel("")  # remove to avoid crowding
    axNx.tick_params(labelsize=8)
    axNx.grid(True)

    # histogram
    axH.hist(lengths, bins=18, color="#38BDF8", edgecolor="white", linewidth=0.8)
    axH.set_xscale("log")
    axH.xaxis.set_minor_formatter(NullFormatter())
    axH.set_title("Length histogram", pad=6)
    axH.set_xlabel("bp (log)", fontsize=9, labelpad=2)
    axH.set_ylabel("")  # keep clean
    axH.tick_params(labelsize=8)
    axH.grid(True)

    # E (hexbin GC vs length)
    axE = fig.add_subplot(gs[3, :])
    panel_tag(axE, "E")

    hb = axE.hexbin(
        lengths, gcs,
        xscale="log",
        gridsize=65,
        mincnt=1,
        linewidths=0,
    )
    hb.set_alpha(1.0)  # EPS-safe

    axE.set_xlabel("Contig length (bp, log10 scale)")
    axE.set_ylabel("GC (%)")
    axE.set_title("GC% vs contig length (density + flagged outliers)", pad=10)
    axE.grid(True, which="major")
    axE.xaxis.set_minor_formatter(NullFormatter())

    cbar = fig.colorbar(hb, ax=axE, pad=0.010, fraction=0.03)
    cbar.set_label("Contigs per bin")

    # annotate outliers with offsets to avoid overlap
    offsets = [(10, 10), (12, -14), (16, 6), (-28, 10), (-30, -14), (18, 18), (-34, 18)]
    annotate_points(axE, outliers, offsets)

    # Title (extra top space so it never collides)
    fig.suptitle(
        "Main Figure 1 — Data + Assembly + Taxonomic signal (A–E)\n"
        "Reads vs contigs taxonomy + computed assembly metrics + continuity + GC structure",
        fontsize=15, fontweight="bold", y=0.992
    )

    # Tight but safe spacing (prevents D/E collisions)
    fig.subplots_adjust(top=0.89, bottom=0.06)

    # Save
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_SVG)
    fig.savefig(OUT_PNG, dpi=600)
    fig.savefig(OUT_JPG, dpi=600)
    fig.savefig(OUT_EPS)  # EPS-safe (no alpha)

    plt.close(fig)

    # Optional: copy into pipeline figures folder too
    try:
        PIPE_FIGDIR.mkdir(parents=True, exist_ok=True)
        for p in [OUT_PDF, OUT_SVG, OUT_PNG, OUT_JPG, OUT_EPS]:
            (PIPE_FIGDIR / p.name).write_bytes(p.read_bytes())
    except Exception:
        pass

    print(f"Wrote: {OUT_PDF}")
    print(f"Wrote: {OUT_SVG}")
    print(f"Wrote: {OUT_PNG}")
    print(f"Wrote: {OUT_JPG}")
    print(f"Wrote: {OUT_EPS}")

if __name__ == "__main__":
    main()
