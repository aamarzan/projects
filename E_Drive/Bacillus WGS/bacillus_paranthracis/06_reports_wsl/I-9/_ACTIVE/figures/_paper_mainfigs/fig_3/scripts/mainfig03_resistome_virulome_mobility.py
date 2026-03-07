#!/usr/bin/env python3
from __future__ import annotations

import os, csv
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.colors import LinearSegmentedColormap

# ----------------------------
# Config
# ----------------------------
I9 = Path(os.environ.get("I9", "/mnt/e/Bacillus WGS/bacillus_paranthracis/06_reports_wsl/I-9"))
OUTDIR = Path(os.environ.get("OUTDIR", str(I9 / "figures" / "_paper_mainfigs" / "fig_3")))
OUTDIR.mkdir(parents=True, exist_ok=True)

AMR_DIR = I9 / "amr"
AMRFINDER = AMR_DIR / "I-9.amrfinder.tsv"
ABR_CARD = AMR_DIR / "I-9.abricate.card.tab"
ABR_NCBI = AMR_DIR / "I-9.abricate.ncbi.tab"
ABR_VFDB = AMR_DIR / "I-9.abricate.vfdb.tab"
ABR_PLASMID = AMR_DIR / "I-9.abricate.plasmidfinder.tab"

PROKKA_GFF = I9 / "annotation_prokka" / "I-9.gff"
ASM_FASTA  = I9 / "I-9.final.contigs.fa"

MIN_ID = float(os.environ.get("MIN_ID", "80"))
MIN_COV = float(os.environ.get("MIN_COV", "60"))

OUT_PDF = OUTDIR / "MainFig03_Resistome_Virulome_Mobility.pdf"
OUT_SVG = OUTDIR / "MainFig03_Resistome_Virulome_Mobility.svg"
OUT_PNG = OUTDIR / "MainFig03_Resistome_Virulome_Mobility.png"
OUT_JPG = OUTDIR / "MainFig03_Resistome_Virulome_Mobility.jpg"
OUT_EPS = OUTDIR / "MainFig03_Resistome_Virulome_Mobility.eps"
OUT_TSV = OUTDIR / "MainFig03_Resistome_Virulome_Mobility.summary.tsv"

# ----------------------------
# Helpers
# ----------------------------
def make_cmap(colors: list[str], name="grad"):
    return LinearSegmentedColormap.from_list(name, colors)

C_BLUE = make_cmap(["#D7EEFF", "#5FB3FF", "#1E5BE0"], "bluegrad")
C_GREEN = make_cmap(["#E7FFF2", "#6FE3B2", "#0E9F6E"], "greengrad")

def panel_tag(ax, tag: str):
    ax.text(
        0.01, 0.98, tag,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=10, fontweight="bold",
        color="white",
        bbox=dict(boxstyle="round,pad=0.25", fc="#111827", ec="none"),
        zorder=20
    )

def safe_float(x: str):
    try:
        return float((x or "").strip())
    except Exception:
        return None

def read_tsv(path: Path):
    if (not path.exists()) or path.stat().st_size == 0:
        return [], []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        rows = list(csv.reader(f, delimiter="\t"))
    if not rows:
        return [], []
    header = [h.strip().lstrip("#") for h in rows[0]]
    out = []
    for r in rows[1:]:
        if not r:
            continue
        if len(r) < len(header):
            r += [""] * (len(header) - len(r))
        out.append({header[i]: r[i].strip() for i in range(len(header))})
    return header, out

def pick_col(header, candidates):
    hl = {h.lower(): h for h in header}
    for c in candidates:
        if c.lower() in hl:
            return hl[c.lower()]
    for h in header:
        for c in candidates:
            if c.lower() in h.lower():
                return h
    return None

def parse_amrfinder(path: Path):
    h, rows = read_tsv(path)
    if not rows:
        return []
    col_gene  = pick_col(h, ["Gene symbol", "Gene", "gene"])
    col_class = pick_col(h, ["Class", "Resistance class"])
    col_sub   = pick_col(h, ["Subclass"])
    out = []
    for r in rows:
        gene = (r.get(col_gene, "") if col_gene else "").strip() or "NA"
        cls  = (r.get(col_class, "") if col_class else "").strip()
        sub  = (r.get(col_sub, "") if col_sub else "").strip()
        out.append({"gene": gene, "class": (cls or sub or "Other")})
    return out

def parse_abricate(path: Path):
    h, rows = read_tsv(path)
    if not rows:
        return []
    col_gene = pick_col(h, ["GENE", "gene"])
    col_seq  = pick_col(h, ["SEQUENCE", "Sequence", "CONTIG", "contig"])
    col_id   = pick_col(h, ["%IDENTITY", "IDENTITY"])
    col_cov  = pick_col(h, ["%COVERAGE", "COVERAGE"])
    out = []
    for r in rows:
        out.append({
            "gene": (r.get(col_gene, "") if col_gene else "").strip() or "NA",
            "seq":  (r.get(col_seq, "") if col_seq else "").strip() or "",
            "pident": safe_float(r.get(col_id, "")) or 0.0 if col_id else 0.0,
            "pcov":   safe_float(r.get(col_cov, "")) or 0.0 if col_cov else 0.0,
        })
    return out

def filter_conf(rows):
    return [r for r in rows if (r.get("pident", 0) >= MIN_ID and r.get("pcov", 0) >= MIN_COV)]

def fasta_lengths(fa: Path):
    lens = {}
    if not fa.exists():
        return lens
    name = None
    buf = []
    with fa.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    lens[name] = sum(len(x) for x in buf)
                name = line[1:].split()[0]
                buf = []
            else:
                buf.append(line)
        if name is not None:
            lens[name] = sum(len(x) for x in buf)
    return lens

def gene_density_curve_gff(gff: Path, contig: str, contig_len: int, bins: int = 90):
    """Returns (xs, dens_norm) for CDS density on the chosen contig."""
    if (not gff.exists()) or not contig or not contig_len:
        return [], []
    counts = [0.0] * bins
    with gff.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            seqid, ftype = parts[0], parts[2]
            if seqid != contig:
                continue
            if ftype not in ("CDS", "gene"):
                continue
            try:
                start = int(parts[3]); end = int(parts[4])
            except Exception:
                continue
            mid = (start + end) / 2.0
            idx = int((max(1.0, min(mid, contig_len)) - 1.0) / contig_len * bins)
            if idx < 0: idx = 0
            if idx >= bins: idx = bins - 1
            counts[idx] += 1.0

    # smooth (simple moving average)
    sm = counts[:]
    for _ in range(2):
        sm2 = sm[:]
        for i in range(bins):
            a = sm[i-1] if i-1 >= 0 else sm[i]
            b = sm[i]
            c = sm[i+1] if i+1 < bins else sm[i]
            sm2[i] = (a + b + c) / 3.0
        sm = sm2

    mx = max(sm) if sm else 0.0
    dens = [(v / mx) if mx > 0 else 0.0 for v in sm]
    xs = [((i + 0.5) / bins) * contig_len for i in range(bins)]
    return xs, dens

def save_all(fig):
    fig.savefig(OUT_PNG, dpi=600)
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_SVG)
    fig.savefig(OUT_EPS, format="eps")
    try:
        from PIL import Image
        im = Image.open(OUT_PNG).convert("RGB")
        im.save(OUT_JPG, quality=95, optimize=True)
    except Exception:
        fig.savefig(OUT_JPG, dpi=600)

# ----------------------------
# Main
# ----------------------------
def main():
    amr = parse_amrfinder(AMRFINDER)
    card_hits = parse_abricate(ABR_CARD)
    ncbi_hits = parse_abricate(ABR_NCBI)
    vfdb_conf = filter_conf(parse_abricate(ABR_VFDB))
    plasm_conf = filter_conf(parse_abricate(ABR_PLASMID))

    # Panel A data
    cls = Counter([r["class"] for r in amr]) if amr else Counter()
    topA = cls.most_common(8)

    # Panel B data
    amr_genes = [r["gene"] for r in amr if r["gene"] != "NA"]
    card_genes = [r["gene"] for r in card_hits if r["gene"] != "NA"]
    ncbi_genes = [r["gene"] for r in ncbi_hits if r["gene"] != "NA"]
    freq = Counter(amr_genes) + Counter(card_genes) + Counter(ncbi_genes)
    genes = [g for g, _ in freq.most_common(6)]
    tools = ["AMRFinder", "CARD", "NCBI"]
    gset_amr, gset_card, gset_ncbi = set(amr_genes), set(card_genes), set(ncbi_genes)

    # Panel E contig choice
    lens = fasta_lengths(ASM_FASTA)
    amr_by_contig = Counter([r["seq"] for r in (card_hits + ncbi_hits) if r.get("seq")])
    contig = amr_by_contig.most_common(1)[0][0] if amr_by_contig else (max(lens, key=lambda k: lens[k]) if lens else "")
    contig_len = lens.get(contig, None)

    # ---------------- Layout ----------------
    fig = plt.figure(figsize=(16, 10), dpi=200)
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1.0, 1.05, 1.25], width_ratios=[1.0, 1.35],
                  hspace=0.52, wspace=0.25)

    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])
    axE = fig.add_subplot(gs[2, :])

    fig.suptitle(
        "Main Figure 3 — Resistome + Virulome + Mobility + Genomic context (A–E)\n"
        "AMRFinder × Abricate (CARD/NCBI/VFDB/PlasmidFinder) concordance + integrated contig mapping",
        fontsize=16, fontweight="bold", y=0.985
    )
    fig.subplots_adjust(top=0.885)

    # ---------------- Panel A ----------------
    panel_tag(axA, "A")
    axA.set_title("Resistome composition (AMRFinder)", fontsize=12, pad=10)
    if not topA:
        axA.axis("off")
    else:
        labels = [k.replace("_", "-") for k, _ in topA][::-1]
        vals = [v for _, v in topA][::-1]
        cols = [C_BLUE((i + 1) / (len(vals) + 1)) for i in range(len(vals))]
        axA.barh(labels, vals, color=cols, edgecolor="#0B1B3A", linewidth=0.6)
        for y, v in enumerate(vals):
            axA.text(v + 0.05, y, str(v), va="center", fontsize=9, color="#0B1B3A")
        axA.set_xlabel("Hit count", fontsize=10)
        axA.grid(axis="x", color="#E7ECF3")
        axA.set_axisbelow(True)

    # ---------------- Panel B ----------------
    panel_tag(axB, "B")
    axB.set_title("Multi-tool concordance (AMRFinder × CARD × NCBI)", fontsize=12, pad=8)

    if not genes:
        axB.axis("off")
    else:
        mat = []
        meta = []
        gene2class = {}
        for r in amr:
            if r["gene"] != "NA":
                gene2class[r["gene"]] = r["class"]

        for g in genes:
            mat.append([1 if g in gset_amr else 0,
                        1 if g in gset_card else 0,
                        1 if g in gset_ncbi else 0])
            meta.append(gene2class.get(g, "NA"))

        axB.imshow(mat, aspect="auto",
                   cmap=make_cmap(["#F5F7FA", "#A7C9FF", "#1E5BE0"], "present"),
                   vmin=0, vmax=1)

        axB.set_yticks(range(len(genes)))
        axB.set_yticklabels(genes, fontsize=9)

        axB.set_xticks(range(len(tools)))
        axB.set_xticklabels(tools, fontsize=10, fontweight="bold")
        axB.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True, pad=6)

        axB.set_xticks([x - 0.5 for x in range(1, len(tools))], minor=True)
        axB.set_yticks([y - 0.5 for y in range(1, len(genes))], minor=True)
        axB.grid(which="minor", color="#E7ECF3", linewidth=1)
        axB.tick_params(which="minor", left=False, bottom=False)

        for i in range(len(genes)):
            for j in range(len(tools)):
                if mat[i][j] == 1:
                    axB.text(j, i, "✓", ha="center", va="center",
                             fontsize=12, fontweight="bold", color="#0B1B3A")

        axB2 = axB.inset_axes([1.02, 0.02, 0.20, 0.96])
        axB2.axis("off")
        axB2.text(0.0, 1.0, "Meta", ha="left", va="top", fontsize=10, fontweight="bold", color="#111827")
        y0 = 0.92
        for m in meta:
            axB2.text(0.0, y0, (m or "NA"), ha="left", va="top", fontsize=9, color="#334155")
            y0 -= 0.12

        axB.text(0.99, -0.18, "Presence = gene-name match (conservative).",
                 transform=axB.transAxes, ha="right", va="top",
                 fontsize=8, color="#6B7280")

    # ---------------- Panel C ----------------
    panel_tag(axC, "C")
    axC.set_title("Virulome snapshot (VFDB)", fontsize=12, pad=10)
    if vfdb_conf:
        vfdb_conf_sorted = sorted(vfdb_conf, key=lambda r: (r["pident"] * r["pcov"]), reverse=True)[:10]
        ylabels = [r["gene"] for r in vfdb_conf_sorted][::-1]
        xs = [r["pident"] for r in vfdb_conf_sorted][::-1]
        ss = [max(40, r["pcov"] * 3.0) for r in vfdb_conf_sorted][::-1]
        axC.scatter(xs, range(len(ylabels)), s=ss, c=xs, cmap=C_GREEN, edgecolors="#0B1B3A", linewidths=0.6)
        axC.set_yticks(range(len(ylabels)))
        axC.set_yticklabels(ylabels, fontsize=9)
        axC.set_xlim(0, 100)
        axC.set_xlabel("Percent identity (%)", fontsize=10)
        axC.grid(axis="x", color="#E7ECF3")
        axC.set_axisbelow(True)
    else:
        axC.axis("off")
        bb = FancyBboxPatch((0.02, 0.05), 0.96, 0.90,
                            boxstyle="round,pad=0.02,rounding_size=0.02",
                            transform=axC.transAxes, fc="#FBFCFE", ec="#D9DEE7", lw=1.0)
        axC.add_patch(bb)
        axC.text(0.05, 0.90, "No confident VFDB hits", transform=axC.transAxes,
                 fontsize=12, fontweight="bold", color="#111827", va="top")
        axC.text(0.05, 0.80,
                 f"Thresholds: identity ≥ {MIN_ID:.0f}%, coverage ≥ {MIN_COV:.0f}%",
                 transform=axC.transAxes, fontsize=9, color="#334155", va="top")

    # ---------------- Panel D ----------------
    panel_tag(axD, "D")
    axD.set_title("Mobility signals (PlasmidFinder)", fontsize=12, pad=10)
    if plasm_conf:
        top = sorted(plasm_conf, key=lambda r: (r["pident"] * r["pcov"]), reverse=True)[:10]
        labels = [r["gene"] for r in top][::-1]
        ids = [r["pident"] for r in top][::-1]
        axD.hlines(range(len(labels)), [0]*len(labels), ids, lw=3, color="#B6D6FF")
        axD.plot(ids, range(len(labels)), "o", ms=9, color="#1E5BE0", mec="#0B1B3A", mew=0.6)
        axD.set_yticks(range(len(labels)))
        axD.set_yticklabels(labels, fontsize=9)
        axD.set_xlim(0, 100)
        axD.set_xlabel("Percent identity (%)", fontsize=10)
        axD.grid(axis="x", color="#E7ECF3")
        axD.set_axisbelow(True)
    else:
        axD.axis("off")
        bb = FancyBboxPatch((0.02, 0.05), 0.96, 0.90,
                            boxstyle="round,pad=0.02,rounding_size=0.02",
                            transform=axD.transAxes, fc="#FBFCFE", ec="#D9DEE7", lw=1.0)
        axD.add_patch(bb)
        axD.text(0.05, 0.90, "No plasmid replicon detected", transform=axD.transAxes,
                 fontsize=12, fontweight="bold", color="#111827", va="top")

    # ---------------- Panel E ----------------
    panel_tag(axE, "E")
    axE.set_title("Integrated genomic context (chosen contig): AMR + VF + plasmid + gene density",
                  fontsize=12, pad=10)
    axE.grid(axis="x", color="#EEF2F7")
    axE.set_axisbelow(True)

    if not contig or contig_len is None:
        axE.axis("off")
    else:
        axE.set_xlim(0, contig_len)
        axE.set_ylim(0, 4.2)
        axE.set_yticks([3.0, 2.0, 1.0])
        axE.set_yticklabels(["AMR", "Virulence", "Plasmid"], fontsize=10)
        axE.set_xlabel("Genomic position (bp)", fontsize=10)

        # Legend (kept as-is)
        leg = axE.inset_axes([0.06, 0.80, 0.18, 0.18])
        leg.axis("off")
        def leg_item(y, label, color):
            leg.add_patch(Rectangle((0.02, y-0.08), 0.10, 0.12, fc=color, ec="#0B1B3A", lw=0.6))
            leg.text(0.16, y, label, va="center", fontsize=9, color="#111827")
        leg_item(0.80, "AMR", "#1E5BE0")
        leg_item(0.55, "VF", "#0E9F6E")
        leg_item(0.30, "PLASMID", "#5A2DCC")

        axE.text(0.99, 0.93,
                 f"Contig: {contig}\nLength: {contig_len:,} bp",
                 transform=axE.transAxes, ha="right", va="top",
                 fontsize=9, color="#111827",
                 bbox=dict(boxstyle="round,pad=0.25", fc="#F3F6FB", ec="#D9DEE7", lw=0.8))

        # ✅ RESTORED: Gene density track (the missing part)
        xs, dens = gene_density_curve_gff(PROKKA_GFF, contig, contig_len, bins=90)
        if xs:
            y0 = 0.25
            y = [y0 + 0.55*d for d in dens]
            axE.fill_between(xs, y0, y, color=C_BLUE(0.45), alpha=0.35, zorder=1)
            axE.plot(xs, y, color=C_BLUE(0.85), lw=1.6, zorder=2)
            axE.text(0.012*contig_len, y0+0.02, "Gene density",
                     fontsize=9, color="#1F2937", va="bottom")

    # Summary TSV
    with OUT_TSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["panel", "metric", "value"])
        w.writerow(["A", "amrfinder_total_hits", str(sum(cls.values()))])
        w.writerow(["B", "genes_in_concordance_panel", str(len(genes))])
        w.writerow(["C", "vfdb_confident_hits", str(len(vfdb_conf))])
        w.writerow(["D", "plasmid_confident_hits", str(len(plasm_conf))])
        w.writerow(["E", "chosen_contig", contig or "NA"])
        w.writerow(["E", "chosen_contig_length_bp", str(contig_len) if contig_len else "NA"])

    save_all(fig)
    plt.close(fig)

    print(f"Wrote: {OUT_PDF}")
    print(f"Wrote: {OUT_SVG}")
    print(f"Wrote: {OUT_PNG}")
    print(f"Wrote: {OUT_JPG}")
    print(f"Wrote: {OUT_EPS}")
    print(f"Wrote: {OUT_TSV}")

if __name__ == "__main__":
    main()
