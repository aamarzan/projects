import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from figure_helper_common import (
    setup_rcparams, read_csv, as_int, heatmap_from_long, guess_col,
    save_png_pdf
)

setup_rcparams()

WORK = "/mnt/e/DrAhmed/Ongoing/WGS/Result"
INDIR = f"{WORK}/_FIGURE_RAW_MATERIALS_V2"
OUTDIR = f"{WORK}/_G3/output/main"
os.makedirs(OUTDIR, exist_ok=True)

species_counts = read_csv(f"{INDIR}/species_counts_166.csv")
amr_gene = read_csv(f"{INDIR}/amr_gene_by_species_top200_166.csv")
vf_gene = read_csv(f"{INDIR}/virulence_gene_by_species_top200_166.csv")
sp_order_major = [r["TopSpecies"] for r in species_counts if as_int(r.get("Count", 0)) >= 4]

fig = plt.figure(figsize=(15.8, 10.0))
gs = GridSpec(2, 1, figure=fig, height_ratios=[1.0, 1.0], hspace=0.34)

# Panel A
ax = fig.add_subplot(gs[0, 0])
row_col = guess_col(amr_gene, ["TopSpecies", "Species"])
key_col = guess_col(amr_gene, ["AMR_Gene", "Gene", "AMR_TopGene"])
val_col = guess_col(amr_gene, ["Count", "Hits"])

im, data, keys = heatmap_from_long(
    ax,
    amr_gene,
    sp_order_major,
    row_col,
    key_col,
    val_col,
    top_n=16,
    cmap_name="blue",
    title="A. Dominant AMR gene signatures",
    annotate=False,
    log1p=True,
    xtick_width=14,
)
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cb.set_label("log(1 + count)")

# Panel B
ax = fig.add_subplot(gs[1, 0])
row_col = guess_col(vf_gene, ["TopSpecies", "Species"])
key_col = guess_col(vf_gene, ["VFDB_Gene", "Virulence_Gene", "Gene"])
val_col = guess_col(vf_gene, ["Count", "Hits"])

im, data, keys = heatmap_from_long(
    ax,
    vf_gene,
    sp_order_major,
    row_col,
    key_col,
    val_col,
    top_n=16,
    cmap_name="rose",
    title="B. Dominant virulence gene signatures",
    annotate=False,
    log1p=True,
    xtick_width=14,
)
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cb.set_label("log(1 + count)")

fig.suptitle("Figure 5. Gene-level resistance and virulence landscape", y=1.01, fontsize=20, fontweight="bold")
save_png_pdf(fig, "Figure05_Gene_Signatures_G3", OUTDIR)
print("Saved Figure 5 to:", OUTDIR)
