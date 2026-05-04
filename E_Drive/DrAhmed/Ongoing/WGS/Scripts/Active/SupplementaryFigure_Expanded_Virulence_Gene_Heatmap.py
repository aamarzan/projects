import os
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

from figure_helper_wgs_remaining import (
    setup_rcparams, read_csv, guess_col, major_species_order,
    norm, as_float, species_short, wrap, CMAPS, save_png_pdf
)

setup_rcparams()

WORK = "/mnt/e/DrAhmed/Ongoing/WGS/Result"
RAW = f"{WORK}/_FIGURE_RAW_MATERIALS_V2"
MASTER = f"{WORK}/_BIOLOGY_LAYER_V2/PrimaryResults_v4_withBiology_166.csv"
OUTDIR = f"{WORK}/_G4_REMAINING/output/supplementary"
os.makedirs(OUTDIR, exist_ok=True)

master_rows = read_csv(MASTER)
major_species = major_species_order(master_rows, min_n=4)
rows = read_csv(f"{RAW}/virulence_gene_by_species_top200_166.csv")

row_col = guess_col(rows, ["TopSpecies", "Species"])
key_col = guess_col(rows, ["VFDB_Gene", "Virulence_Gene", "Gene"])
val_col = guess_col(rows, ["Count", "Hits"])

counter_by_key = Counter()
for r in rows:
    k = norm(r.get(key_col, ""))
    if k:
        counter_by_key[k] += as_float(r.get(val_col, 0))
keys = [k for k, _ in counter_by_key.most_common(35)]

data = np.zeros((len(major_species), len(keys)), dtype=float)
rix = {r: i for i, r in enumerate(major_species)}
kix = {k: i for i, k in enumerate(keys)}
for r in rows:
    rr = norm(r.get(row_col, ""))
    kk = norm(r.get(key_col, ""))
    vv = as_float(r.get(val_col, 0))
    if rr in rix and kk in kix:
        data[rix[rr], kix[kk]] += vv

fig, ax = plt.subplots(figsize=(18, 6.8))
im = ax.imshow(np.log1p(data), aspect="auto", cmap=CMAPS["rose"])
ax.set_yticks(range(len(major_species)))
ax.set_yticklabels([species_short(x) for x in major_species])
ax.set_xticks(range(len(keys)))
ax.set_xticklabels([wrap(k, 15) for k in keys], rotation=35, ha="right")
ax.set_title("Expanded virulence gene landscape", pad=10)
cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label("log(1 + count)")
fig.suptitle("Expanded virulence gene heatmap", y=0.995, fontsize=20, fontweight="bold")
save_png_pdf(fig, "SupplementaryFigure_Expanded_Virulence_Gene_Heatmap", OUTDIR)
print("Saved supplementary virulence gene figure to:", OUTDIR)