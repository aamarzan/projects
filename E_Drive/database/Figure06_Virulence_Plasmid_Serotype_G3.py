import os
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from figure_helper_common import (
    setup_rcparams, read_csv, as_int, norm, species_short,
    heatmap_from_long, guess_col, save_png_pdf, style_ax,
    palette_list, annot_barh
)

setup_rcparams()

WORK = "/mnt/e/DrAhmed/Ongoing/WGS/Result"
RAW = f"{WORK}/_FIGURE_RAW_MATERIALS_V2"
CLEAN = f"{WORK}/_G3/clean"
OUTDIR = f"{WORK}/_G3/output/main"
os.makedirs(OUTDIR, exist_ok=True)

species_counts = read_csv(f"{RAW}/species_counts_166.csv")
feature_prev = read_csv(f"{CLEAN}/species_feature_prevalence_from_master_166.csv")
plasmid_rep = read_csv(f"{CLEAN}/plasmid_replicon_by_species_166.csv")
serotype = read_csv(f"{CLEAN}/serotype_distribution_166.csv")
sp_order_all = [r["TopSpecies"] for r in species_counts]

fig = plt.figure(figsize=(15.6, 10.6))
gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.0], width_ratios=[1.05, 1.25], hspace=0.32, wspace=0.26)

# Panel A
ax = fig.add_subplot(gs[:, 0])
prev_map = {r["TopSpecies"]: r for r in feature_prev}
species = sp_order_all
vf_pct = []
pl_pct = []

for s in species:
    rr = prev_map[s]
    n = max(as_int(rr["Samples_n"]), 1)
    vf_pct.append(100.0 * as_int(rr["Samples_with_VFDB_hits"]) / n)
    pl_pct.append(100.0 * as_int(rr["Samples_with_Plasmid_hits"]) / n)

y = np.arange(len(species))
h = 0.36
ax.barh(y - h / 2, vf_pct, height=h, color="#c23b6f", label="VFDB-positive samples", zorder=3)
ax.barh(y + h / 2, pl_pct, height=h, color="#6a4bd9", label="Plasmid-positive samples", zorder=3)

ax.set_yticks(y)
ax.set_yticklabels([species_short(s) for s in species])
ax.invert_yaxis()
style_ax(ax, "A. Prevalence of biological feature carriage", "Samples with feature (%)", None, "x")
ax.legend(loc="lower right", frameon=False)
ax.set_xlim(0, 105)

# Panel B
ax = fig.add_subplot(gs[0, 1])
if plasmid_rep:
    row_col = guess_col(plasmid_rep, ["TopSpecies", "Species"])
    key_col = guess_col(plasmid_rep, ["Plasmid_Replicon", "Replicon"])
    val_col = guess_col(plasmid_rep, ["Count", "Hits"])

    im, data, keys = heatmap_from_long(
        ax,
        plasmid_rep,
        sp_order_all,
        row_col,
        key_col,
        val_col,
        top_n=10,
        cmap_name="violet",
        title="B. Dominant plasmid replicon architecture",
        annotate=True,
        log1p=True,
        xtick_width=14,
    )
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("log(1 + count)")
else:
    ax.axis("off")
    ax.text(0.5, 0.5, "No plasmid replicon table available", ha="center", va="center", fontsize=12)

# Panel C
ax = fig.add_subplot(gs[1, 1])
if serotype:
    sero_counter = Counter([norm(r["Serotype_Call"]) for r in serotype if norm(r["Serotype_Call"])])
    labs = list(sero_counter.keys())
    vals = [sero_counter[k] for k in labs]

    lab2 = []
    for x in labs:
        x = x.replace("   ", " ")
        parts = x.split()
        if len(parts) >= 3:
            lab2.append(f"{parts[0]} {parts[1]}\n{' '.join(parts[2:])}")
        else:
            lab2.append(x)

    bars = ax.barh(
        lab2,
        vals,
        color=palette_list("green", len(vals)),
        edgecolor="#0f5b26",
        linewidth=0.7,
        zorder=3,
    )
    ax.invert_yaxis()
    style_ax(ax, "C. Informative serotype calls", "Samples", None, "x")
    annot_barh(ax, bars, "{:.0f}", xpad=0.02)
else:
    ax.axis("off")
    ax.text(0.5, 0.5, "No informative serotype calls", ha="center", va="center", fontsize=12)

fig.suptitle("Figure 6. Virulence, plasmid, and serotype architecture", y=1.01, fontsize=20, fontweight="bold")
save_png_pdf(fig, "Figure06_Virulence_Plasmid_Serotype_G3", OUTDIR)
print("Saved Figure 6 to:", OUTDIR)
