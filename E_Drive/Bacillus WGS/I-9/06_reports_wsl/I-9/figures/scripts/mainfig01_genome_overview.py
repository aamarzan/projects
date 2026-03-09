#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# ------------------------- helpers -------------------------
def iter_fasta(path: Path):
    name = None
    seq_parts = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(seq_parts)
                name = line[1:].strip().split()[0]
                seq_parts = []
            else:
                seq_parts.append(line.strip())
        if name is not None:
            yield name, "".join(seq_parts)


def gc_percent(seq: str) -> float:
    s = seq.upper()
    if not s:
        return 0.0
    gc = s.count("G") + s.count("C")
    return 100.0 * gc / max(1, len(s))


def fmt_int(x) -> str:
    try:
        return f"{int(x):,}"
    except Exception:
        return str(x)


def pick_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists() and p.stat().st_size > 0:
            return p
    return None


def panel_label(ax, label: str):
    ax.text(
        0.01, 0.99, label,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=12, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="black", linewidth=0.8),
        zorder=10,
    )


# ------------------------- Kraken report parsing -------------------------
def parse_kraken_report(report_path: Path):
    """
    Kraken2 report format (tab):
    pct  clade  direct  rank  taxid  name(with indentation)
    """
    rows = []
    with report_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 6:
                continue
            pct = float(parts[0])
            clade = int(parts[1])
            direct = int(parts[2])
            rank = parts[3].strip()
            taxid = parts[4].strip()
            raw_name = parts[5]
            depth = len(raw_name) - len(raw_name.lstrip(" "))
            name = raw_name.lstrip()
            rows.append(
                dict(
                    pct=pct,
                    clade=clade,
                    direct=direct,
                    rank=rank,
                    taxid=taxid,
                    name=name,
                    depth=depth,
                )
            )
    return rows


def build_parent_maps(rows):
    """
    Uses indentation depth to build parent relationships.
    """
    parent = {}
    rank = {}
    name = {}
    stack = []  # taxids by depth
    depth_stack = []

    for r in rows:
        d = r["depth"]
        tax = r["taxid"]
        # normalize stack
        while depth_stack and depth_stack[-1] >= d:
            stack.pop()
            depth_stack.pop()

        parent[tax] = stack[-1] if stack else None
        rank[tax] = r["rank"]
        name[tax] = r["name"]

        stack.append(tax)
        depth_stack.append(d)

    return parent, rank, name


def to_species_taxid(taxid: str, parent, rank) -> str | None:
    if taxid == "0":
        return "0"
    cur = taxid
    seen = set()
    while cur is not None and cur in parent and cur not in seen:
        seen.add(cur)
        r = rank.get(cur, "")
        if r.startswith("S"):
            return cur
        cur = parent.get(cur)
    return None


def composition_from_report(report_rows, top_n=6):
    """
    Creates categories that sum to 100%:
    top species + other species + higher-rank classified + unclassified
    """
    unclassified = 0.0
    species = []
    for r in report_rows:
        if r["taxid"] == "0" or r["name"].lower() == "unclassified":
            unclassified = r["pct"]
        if r["rank"].startswith("S"):
            species.append((r["name"], r["pct"]))

    species.sort(key=lambda x: x[1], reverse=True)
    total_species = sum(p for _, p in species)
    top = species[:top_n]
    top_sum = sum(p for _, p in top)

    other_species = max(0.0, total_species - top_sum)
    higher_rank = max(0.0, 100.0 - unclassified - total_species)

    labels = [n for n, _ in top]
    values = [p for _, p in top]
    if other_species > 0.001:
        labels.append("Other species")
        values.append(other_species)
    if higher_rank > 0.001:
        labels.append("Higher-rank classified")
        values.append(higher_rank)
    labels.append("Unclassified")
    values.append(max(0.0, unclassified))

    # Normalize tiny rounding drift
    s = sum(values)
    if abs(s - 100.0) > 0.05 and s > 0:
        # adjust last element
        values[-1] += (100.0 - s)

    return labels, values


# ------------------------- Kraken output parsing (bp-weighted) -------------------------
def parse_kraken_out(out_path: Path):
    """
    Kraken2 output format typical:
    C/U  seqid  taxid  length  ...
    """
    recs = []
    with out_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            status = parts[0].strip()
            seqid = parts[1].strip()
            taxid = parts[2].strip()
            try:
                seqlen = int(parts[3])
            except Exception:
                seqlen = None
            recs.append((status, seqid, taxid, seqlen))
    return recs


def bp_weighted_species(out_recs, contig_len_map, parent, rank, taxname, top_n=6):
    """
    bp-weighted composition collapsed to species using report hierarchy.
    Returns categories summing to 100%.
    """
    bp_by = defaultdict(int)
    total_bp = 0

    for status, seqid, taxid, seqlen in out_recs:
        L = seqlen if seqlen is not None else contig_len_map.get(seqid, 0)
        if L <= 0:
            continue
        total_bp += L
        if status.upper() == "U":
            bp_by["0"] += L
            continue
        sp = to_species_taxid(taxid, parent, rank)
        if sp is None:
            bp_by["__higher__"] += L
        else:
            bp_by[sp] += L

    if total_bp == 0:
        return ["Unclassified"], [100.0]

    unclassified_pct = 100.0 * bp_by.get("0", 0) / total_bp
    higher_pct = 100.0 * bp_by.get("__higher__", 0) / total_bp

    species_items = []
    for k, v in bp_by.items():
        if k in ("0", "__higher__"):
            continue
        nm = taxname.get(k, k)
        species_items.append((nm, 100.0 * v / total_bp))

    species_items.sort(key=lambda x: x[1], reverse=True)
    top = species_items[:top_n]
    top_sum = sum(p for _, p in top)
    all_species_sum = sum(p for _, p in species_items)
    other_species = max(0.0, all_species_sum - top_sum)

    labels = [n for n, _ in top]
    values = [p for _, p in top]
    if other_species > 0.001:
        labels.append("Other species")
        values.append(other_species)
    if higher_pct > 0.001:
        labels.append("Higher-rank classified")
        values.append(higher_pct)
    labels.append("Unclassified")
    values.append(max(0.0, unclassified_pct))

    # Normalize rounding
    s = sum(values)
    if abs(s - 100.0) > 0.05 and s > 0:
        values[-1] += (100.0 - s)

    return labels, values


# ------------------------- QUAST parsing -------------------------
def parse_quast_tsv(path: Path) -> dict[str, str]:
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f, delimiter="\t")
        for r in reader:
            if r and any(c.strip() for c in r):
                rows.append([c.strip() for c in r])

    if not rows:
        return {}

    # Case 1: transposed (header row begins with "Assembly")
    if rows[0][0].lower() == "assembly" and len(rows) >= 2:
        header = rows[0]
        data = rows[1]
        m = {"Assembly": data[0] if data else ""}
        for h, v in zip(header[1:], data[1:]):
            m[h] = v
        return m

    # Case 2: normal with "metric/value" or "Statistic"
    m = {}
    # Skip header row if looks like header
    start = 0
    if rows[0][0].lower() in ("statistic", "metric"):
        start = 1
    for r in rows[start:]:
        if len(r) >= 2:
            m[r[0]] = r[1]
    return m


def find_metric(m: dict[str, str], keywords: list[str]) -> str | None:
    for k in m.keys():
        kk = k.lower()
        if all(w.lower() in kk for w in keywords):
            return m[k]
    return None


# ------------------------- Main Figure 1 -------------------------
def main():
    ap = argparse.ArgumentParser(description="Main Figure 1: Data + Assembly + Taxonomic signal (A–E)")
    ap.add_argument("--i9", default="/mnt/e/Bacillus WGS/bacillus_paranthracis/06_reports_wsl/I-9", help="I-9 WSL folder")
    ap.add_argument("--top", type=int, default=6, help="Top N species shown in composition panels")
    args = ap.parse_args()

    I9 = Path(args.i9)

    # Inputs (auto-detect with fallbacks)
    reads_report = pick_existing([
        I9 / "I-9.reads.kraken2.report.txt",
        I9 / "I-9.reads.kraken2.report",
    ])
    contigs_report = pick_existing([
        I9 / "I-9.contigs.kraken2.report.txt",
        I9 / "I-9.contigs.min5kb.kraken2.report.txt",
        I9 / "I-9.contigs.min5kb.kraken2.report.noconf.txt",
    ])
    contigs_out = pick_existing([
        I9 / "I-9.contigs.kraken2.out.txt",
        I9 / "I-9.contigs.min5kb.kraken2.out.txt",
        I9 / "I-9.contigs.min5kb.kraken2.out.noconf.txt",
    ])
    contigs_fa = pick_existing([
        I9 / "I-9.final.contigs.fa",
        I9 / "I-9.contigs.min1kb.fa",
    ])

    quast_tsv = pick_existing([
        I9 / "quast_I-9" / "report.tsv",
        I9 / "quast_I-9" / "transposed_report.tsv",
    ])

    missing = []
    if reads_report is None: missing.append("reads kraken2 report")
    if contigs_report is None: missing.append("contigs kraken2 report")
    if contigs_fa is None: missing.append("contigs fasta")
    if quast_tsv is None: missing.append("quast report.tsv")
    if missing:
        raise SystemExit("Missing required inputs: " + ", ".join(missing))

    # Output folder
    outdir = I9 / "figures" / "_paper_mainfigs"
    outdir.mkdir(parents=True, exist_ok=True)
    out_pdf = outdir / "MainFig01_Genome_Overview.pdf"
    out_svg = outdir / "MainFig01_Genome_Overview.svg"
    out_tsv = outdir / "MainFig01_Genome_Overview.tsv"

    # Parse contigs fasta -> lengths & GC
    contig_len = {}
    contig_gc = {}
    for cid, seq in iter_fasta(contigs_fa):
        contig_len[cid] = len(seq)
        contig_gc[cid] = gc_percent(seq)

    lengths = sorted(contig_len.values(), reverse=True)
    total_len = sum(lengths) if lengths else 0

    # Nx values
    nx_points = []
    if total_len > 0:
        for x in [10, 25, 50, 75, 90]:
            target = total_len * (x / 100.0)
            s = 0
            nx = lengths[-1]
            for L in lengths:
                s += L
                if s >= target:
                    nx = L
                    break
            nx_points.append((x, nx))

    # QUAST metrics
    q = parse_quast_tsv(quast_tsv)
    q_n50 = find_metric(q, ["n50"]) or "NA"
    q_l50 = find_metric(q, ["l50"]) or "NA"
    q_gc  = find_metric(q, ["gc"]) or "NA"
    q_total = find_metric(q, ["total", "length"]) or str(total_len)
    q_contigs = find_metric(q, ["# contigs"]) or find_metric(q, ["contigs"]) or str(len(lengths))
    q_largest = find_metric(q, ["largest", "contig"]) or (fmt_int(max(lengths)) if lengths else "NA")

    # Kraken reports
    rr = parse_kraken_report(reads_report)
    cr = parse_kraken_report(contigs_report)

    read_labels, read_vals = composition_from_report(rr, top_n=args.top)

    # contig bp-weighted (preferred); fallback to report % if no out file
    parent, rank, taxname = build_parent_maps(cr)
    if contigs_out and contigs_out.exists():
        out_recs = parse_kraken_out(contigs_out)
        cont_labels, cont_vals = bp_weighted_species(out_recs, contig_len, parent, rank, taxname, top_n=args.top)
        cont_title = "Contigs composition (bp-weighted)"
    else:
        cont_labels, cont_vals = composition_from_report(cr, top_n=args.top)
        cont_title = "Contigs composition (report %)"

    # Color mapping across A/B
    # Use a stable palette for taxa labels; special categories are fixed
    base_palette = list(plt.get_cmap("tab20").colors)
    color_map = {}
    special = {
        "Unclassified": (0.65, 0.65, 0.65),
        "Higher-rank classified": (0.35, 0.35, 0.35),
        "Other species": (0.80, 0.80, 0.80),
    }
    for k, v in special.items():
        color_map[k] = v

    taxa_all = []
    for lab in read_labels + cont_labels:
        if lab not in special and lab not in taxa_all:
            taxa_all.append(lab)

    for i, t in enumerate(taxa_all):
        color_map[t] = base_palette[i % len(base_palette)]

    # Build contig->species assignment for panel E coloring (using contigs_out if present)
    contig_species = {}
    top_species_set = set([x for x in cont_labels if x not in special][: min(3, len(cont_labels))])

    if contigs_out and contigs_out.exists():
        out_recs = parse_kraken_out(contigs_out)
        for status, seqid, taxid, _ in out_recs:
            if status.upper() == "U":
                contig_species[seqid] = "Unclassified"
            else:
                sp = to_species_taxid(taxid, parent, rank)
                if sp is None:
                    contig_species[seqid] = "Higher-rank classified"
                else:
                    contig_species[seqid] = taxname.get(sp, sp)
    else:
        # fallback: no per-contig colors
        for cid in contig_len.keys():
            contig_species[cid] = "Other species"

    # ------------------------- plotting style -------------------------
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
    })

    fig = plt.figure(figsize=(13, 9), dpi=220)
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1.05, 1.05, 1.6], hspace=0.35, wspace=0.25)

    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])
    axE = fig.add_subplot(gs[2, :])

    # ---- Panel A: Reads composition ----
    y = list(range(len(read_labels)))[::-1]
    axA.barh(
        y,
        read_vals[::-1],
        color=[color_map.get(l, (0.5, 0.5, 0.5)) for l in read_labels[::-1]],
        edgecolor="black",
        linewidth=0.4,
    )
    axA.set_yticks(y)
    axA.set_yticklabels(read_labels[::-1])
    axA.set_xlim(0, 100)
    axA.set_xlabel("Percent of reads (%)")
    axA.set_title("Read-level taxonomy (Kraken2 report)")
    axA.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.8)
    axA.spines["top"].set_visible(False)
    axA.spines["right"].set_visible(False)
    panel_label(axA, "a")

    # ---- Panel B: Contigs composition (bp-weighted) ----
    y2 = list(range(len(cont_labels)))[::-1]
    axB.barh(
        y2,
        cont_vals[::-1],
        color=[color_map.get(l, (0.5, 0.5, 0.5)) for l in cont_labels[::-1]],
        edgecolor="black",
        linewidth=0.4,
    )
    axB.set_yticks(y2)
    axB.set_yticklabels(cont_labels[::-1])
    axB.set_xlim(0, 100)
    axB.set_xlabel("Percent of assembly length (%)")
    axB.set_title(cont_title)
    axB.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.8)
    axB.spines["top"].set_visible(False)
    axB.spines["right"].set_visible(False)
    panel_label(axB, "b")

    # ---- Panel C: QUAST metrics table ----
    axC.axis("off")
    lines = [
        ("Assembly", contigs_fa.name),
        ("# contigs", q_contigs),
        ("Total length (bp)", q_total),
        ("Largest contig (bp)", q_largest),
        ("N50 (bp)", q_n50),
        ("L50", q_l50),
        ("GC (%)", q_gc),
        ("Contigs FASTA used", contigs_fa.name),
        ("QUAST source", quast_tsv.name),
    ]
    # pretty box
    axC.text(
        0.02, 0.96,
        "QUAST key metrics",
        ha="left", va="top",
        fontsize=11, fontweight="bold",
        transform=axC.transAxes,
    )
    y0 = 0.86
    dy = 0.085
    for k, v in lines:
        axC.text(0.02, y0, k, ha="left", va="top", transform=axC.transAxes, fontsize=9, fontweight="bold")
        axC.text(0.52, y0, str(v), ha="left", va="top", transform=axC.transAxes, fontsize=9)
        y0 -= dy
    # border
    axC.add_patch(plt.Rectangle((0.01, 0.05), 0.98, 0.90, fill=False, lw=0.9, ec="black", transform=axC.transAxes))
    panel_label(axC, "c")

    # ---- Panel D: Assembly continuity (CDF) + histogram inset + Nx inset ----
    if lengths:
        lengths_sorted = sorted(lengths)
        cumsum = []
        s = 0
        for L in lengths_sorted:
            s += L
            cumsum.append(s / total_len)

        axD.step(lengths_sorted, cumsum, where="post", linewidth=1.4)
        axD.set_xscale("log")
        axD.set_ylim(0, 1.02)
        axD.set_xlabel("Contig length (bp, log scale)")
        axD.set_ylabel("Cumulative assembly fraction")
        axD.set_title("Assembly continuity + Nx summary")
        axD.grid(axis="both", linestyle=":", linewidth=0.6, alpha=0.8)
        axD.spines["top"].set_visible(False)
        axD.spines["right"].set_visible(False)

        # Histogram inset
        ax_hist = inset_axes(axD, width="45%", height="45%", loc="lower right", borderpad=1.0)
        ax_hist.hist(lengths_sorted, bins=25)
        ax_hist.set_xscale("log")
        ax_hist.set_title("Length histogram", fontsize=8)
        ax_hist.tick_params(labelsize=7)
        ax_hist.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.8)
        ax_hist.spines["top"].set_visible(False)
        ax_hist.spines["right"].set_visible(False)

        # Nx inset
        ax_nx = inset_axes(axD, width="45%", height="45%", loc="upper right", borderpad=1.0)
        if nx_points:
            xs = [p for p, _ in nx_points]
            ys = [n for _, n in nx_points]
            ax_nx.plot(xs, ys, marker="o", linewidth=1.0)
            ax_nx.set_title("Nx curve", fontsize=8)
            ax_nx.set_xlabel("x", fontsize=7)
            ax_nx.set_ylabel("Nx (bp)", fontsize=7)
            ax_nx.tick_params(labelsize=7)
            ax_nx.grid(axis="both", linestyle=":", linewidth=0.5, alpha=0.8)
            ax_nx.spines["top"].set_visible(False)
            ax_nx.spines["right"].set_visible(False)

    panel_label(axD, "d")

    # ---- Panel E: GC% vs contig length (colored by top taxa) ----
    xs = []
    ys = []
    cs = []
    for cid, L in contig_len.items():
        xs.append(L)
        ys.append(contig_gc.get(cid, 0.0))
        sp = contig_species.get(cid, "Other species")
        if sp in top_species_set:
            cs.append(color_map.get(sp, (0.2, 0.2, 0.8)))
        else:
            # non-top = gray
            cs.append((0.55, 0.55, 0.55))

    axE.scatter(xs, ys, s=14, c=cs, edgecolors="black", linewidths=0.25, alpha=0.95)
    axE.set_xscale("log")
    axE.set_xlabel("Contig length (bp, log scale)")
    axE.set_ylabel("GC (%)")
    axE.set_title("GC vs contig size (outliers/contamination signal)")
    axE.grid(axis="both", linestyle=":", linewidth=0.6, alpha=0.8)
    axE.spines["top"].set_visible(False)
    axE.spines["right"].set_visible(False)
    panel_label(axE, "e")

    # Legend for top taxa colors
    handles = []
    labels = []
    for sp in sorted(top_species_set):
        handles.append(axE.scatter([], [], s=40, c=[color_map.get(sp)], edgecolors="black", linewidths=0.4))
        labels.append(sp.replace("_", " "))
    if handles:
        axE.legend(handles, labels, title="Top taxa (from contigs bp-weighted)", loc="lower right", frameon=True)

    fig.suptitle(
        "Main Figure 1 — Data + Assembly + Taxonomic signal (a–e)\n"
        "Read composition + bp-weighted contig composition + QUAST + continuity + GC outlier structure",
        fontsize=13, fontweight="bold"
    )

    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)

    # TSV export (for manuscript table / supplement / reproducibility)
    with out_tsv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["SECTION", "KEY", "VALUE"])
        w.writerow(["INPUT", "reads_report", str(reads_report)])
        w.writerow(["INPUT", "contigs_report", str(contigs_report)])
        w.writerow(["INPUT", "contigs_out", str(contigs_out) if contigs_out else "NA"])
        w.writerow(["INPUT", "contigs_fasta", str(contigs_fa)])
        w.writerow(["INPUT", "quast_tsv", str(quast_tsv)])

        w.writerow(["QUAST", "#contigs", q_contigs])
        w.writerow(["QUAST", "total_length_bp", q_total])
        w.writerow(["QUAST", "largest_contig_bp", q_largest])
        w.writerow(["QUAST", "N50_bp", q_n50])
        w.writerow(["QUAST", "L50", q_l50])
        w.writerow(["QUAST", "GC_percent", q_gc])

        w.writerow(["KRAKEN_READS", "category", "percent"])
        for lab, val in zip(read_labels, read_vals):
            w.writerow(["KRAKEN_READS", lab, f"{val:.4f}"])

        w.writerow(["KRAKEN_CONTIGS_BP", "category", "percent"])
        for lab, val in zip(cont_labels, cont_vals):
            w.writerow(["KRAKEN_CONTIGS_BP", lab, f"{val:.4f}"])

        if nx_points:
            for x, n in nx_points:
                w.writerow(["NX", f"N{x}", n])

        # a few contig length quantiles
        if lengths_sorted:
            def qtile(p):
                idx = int(round((len(lengths_sorted)-1) * p))
                return lengths_sorted[max(0, min(len(lengths_sorted)-1, idx))]
            w.writerow(["CONTIGS", "count", len(lengths_sorted)])
            w.writerow(["CONTIGS", "total_bp", total_len])
            w.writerow(["CONTIGS", "p50_length_bp", qtile(0.50)])
            w.writerow(["CONTIGS", "p90_length_bp", qtile(0.90)])

    print(f"Wrote: {out_pdf}")
    print(f"Wrote: {out_svg}")
    print(f"Wrote: {out_tsv}")


if __name__ == "__main__":
    main()
