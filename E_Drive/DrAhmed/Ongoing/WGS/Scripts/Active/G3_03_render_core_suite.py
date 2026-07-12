import os
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from G3_02_plotting_helper import (
    setup_rcparams, read_csv, first_existing, as_int, as_float, norm,
    guess_col, species_short, wrap, major_species_from_counts,
    palette_list, style_ax, annot_barh, annot_barv, heatmap_from_long,
    save_png_pdf, ensure_dir
)

setup_rcparams()

WORK = "/mnt/e/DrAhmed/Ongoing/WGS/Result"
RAW = first_existing([
    f"{WORK}/_FIGURE_RAW_MATERIALS_V2/species_counts_166.csv",
    f"{WORK}/_FIGURE_RAW_MATERIALS/species_counts_166.csv",
]).rsplit("/", 1)[0]
CLEAN = f"{WORK}/_G3/clean"
OUT = f"{WORK}/_G3/output/main"
SUPP = f"{WORK}/_G3/output/supplementary"
ensure_dir(OUT)
ensure_dir(SUPP)

species_counts = read_csv(f"{RAW}/species_counts_166.csv")
species_bio = read_csv(f"{RAW}/species_biology_summary_166.csv")
amr_class = read_csv(f"{RAW}/amr_class_by_species_166.csv")
amr_gene = read_csv(f"{RAW}/amr_gene_by_species_top200_166.csv")
vf_gene = read_csv(f"{RAW}/virulence_gene_by_species_top200_166.csv")

feature_prev = read_csv(f"{CLEAN}/species_feature_prevalence_from_master_166.csv")
plasmid_rep = read_csv(f"{CLEAN}/plasmid_replicon_by_species_166.csv")
serotype = read_csv(f"{CLEAN}/serotype_distribution_166.csv")

# ---------- common ----------
sp_order_all = [r["TopSpecies"] for r in species_counts]
sp_order_major = major_species_from_counts(species_counts, min_n=4)

# =====================================
# Figure 1: cohort + species overview
# =====================================
sp_to_n = {r["TopSpecies"]: as_int(r["Count"]) for r in species_counts}
bio_map = {r["TopSpecies"]: r for r in species_bio}

highc = sum(as_int(r.get("HighConfidence_n", 0)) for r in species_bio)
prio = sum(as_int(r.get("PriorityReview_n", 0)) for r in species_bio)
total = sum(as_int(r.get("Samples_n", 0)) for r in species_bio)
other = max(total - highc - prio, 0)

fig = plt.figure(figsize=(14.6, 10.4))
gs = GridSpec(2, 2, figure=fig, width_ratios=[1.05, 1.45], height_ratios=[1.0, 1.0], hspace=0.35, wspace=0.25)

# A: donut
ax = fig.add_subplot(gs[0, 0])
sizes = [highc, prio, other]
labels = ["High-confidence", "Priority-review", "Other/remaining"]
colors = ["#1f5fbf", "#7b4fd6", "#c7d2e5"]
wedges, _ = ax.pie(
    sizes, startangle=90, counterclock=False,
    colors=colors, wedgeprops=dict(width=0.38, edgecolor="white", linewidth=2)
)
ax.text(0, 0.06, f"{total}", ha="center", va="center", fontsize=24, fontweight="bold")
ax.text(0, -0.14, "samples", ha="center", va="center", fontsize=11, color="#4b5563")
ax.set_title("A. Cohort confidence architecture", loc="left", pad=12)
ax.legend(wedges, [f"{l} (n={s})" for l, s in zip(labels, sizes)], loc="lower center",
          bbox_to_anchor=(0.5, -0.18), ncol=1, frameon=False)

# B: species counts
ax = fig.add_subplot(gs[0, 1])
labels = [species_short(s) for s in sp_order_all]
vals = [sp_to_n[s] for s in sp_order_all]
colors = palette_list("teal", len(vals))
bars = ax.barh(labels, vals, color=colors, edgecolor="#18324a", linewidth=0.7, zorder=3)
ax.invert_yaxis()
style_ax(ax, "B. Species distribution across all 166 samples", "Samples", None, "x")
annot_barh(ax, bars, "{:.0f}")
ax.set_xlim(0, max(vals) * 1.18)

# C: confidence composition by species
ax = fig.add_subplot(gs[1, :])
species = sp_order_all
h = [as_int(bio_map[s].get("HighConfidence_n", 0)) for s in species]
p = [as_int(bio_map[s].get("PriorityReview_n", 0)) for s in species]
o = [max(as_int(bio_map[s].get("Samples_n", 0)) - hh - pp, 0) for s, hh, pp in zip(species, h, p)]
y = np.arange(len(species))
ax.barh(y, h, color="#1f5fbf", label="High-confidence", zorder=3)
ax.barh(y, p, left=h, color="#7b4fd6", label="Priority-review", zorder=3)
ax.barh(y, o, left=np.array(h) + np.array(p), color="#d8dfeb", label="Other/remaining", zorder=3)
ax.set_yticks(y)
ax.set_yticklabels([species_short(s) for s in species])
ax.invert_yaxis()
style_ax(ax, "C. Confidence composition within species groups", "Samples", None, "x")
ax.legend(loc="lower right", frameon=False, ncol=3)
for yi, tot in zip(y, [as_int(bio_map[s]["Samples_n"]) for s in species]):
    ax.text(tot + max(vals) * 0.02, yi, str(tot), va="center", ha="left", fontsize=9)
ax.set_xlim(0, max(vals) * 1.18)

fig.suptitle("Figure 1. Cohort and taxonomic overview", y=1.02, fontsize=20, fontweight="bold")
save_png_pdf(fig, "Figure01_Cohort_Taxon_Overview_G3", OUT)

# =====================================
# Figure 2: biology profile
# =====================================
fig = plt.figure(figsize=(15.2, 9.6))
gs = GridSpec(2, 2, figure=fig, hspace=0.34, wspace=0.22)

metric_specs = [
    ("Median_AMR_Genes_n", "Median AMR genes", "blue"),
    ("Median_AMR_Classes_n", "Median AMR classes", "amber"),
    ("Median_VFDB_Hits_n", "Median VFDB hits", "rose"),
    ("Median_Plasmid_Hits_n", "Median plasmid hits", "violet"),
]

for i, (col, ttl, cmap_name) in enumerate(metric_specs):
    ax = fig.add_subplot(gs[i // 2, i % 2])
    rows = [bio_map[s] for s in sp_order_major]
    vals = [as_float(r.get(col, 0)) for r in rows]
    labels = [f"{species_short(s)}  (n={as_int(bio_map[s]['Samples_n'])})" for s in sp_order_major]
    colors = palette_list(cmap_name, len(vals))
    bars = ax.barh(labels, vals, color=colors, edgecolor="#243447", linewidth=0.7, zorder=3)
    ax.invert_yaxis()
    style_ax(ax, ttl, ttl, None, "x")
    ax.set_title(chr(65 + i) + f". {ttl}", loc="left", pad=10)
    annot_barh(ax, bars, "{:.1f}" if any(v % 1 for v in vals) else "{:.0f}", xpad=0.02)
    ax.set_xlim(0, max(vals) * 1.22 if max(vals) > 0 else 1)

fig.suptitle("Figure 2. Species-level biological burden profile", y=1.01, fontsize=20, fontweight="bold")
save_png_pdf(fig, "Figure02_Biology_Profile_G3", OUT)

# =====================================
# Figure 4: AMR class architecture
# =====================================
fig = plt.figure(figsize=(15.2, 9.9))
gs = GridSpec(2, 1, figure=fig, height_ratios=[1.45, 1.0], hspace=0.35)

ax = fig.add_subplot(gs[0, 0])
row_col = guess_col(amr_class, ["TopSpecies", "Species"])
key_col = guess_col(amr_class, ["AMR_Class", "Class"])
val_col = guess_col(amr_class, ["Count", "Hits"])
im, data, keys = heatmap_from_long(
    ax, amr_class, sp_order_major, row_col, key_col, val_col,
    top_n=14, cmap_name="amber",
    title="A. Dominant AMR classes across major species",
    annotate=True, log1p=True
)
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cb.set_label("log(1 + count)")

ax = fig.add_subplot(gs[1, 0])
totals = []
for k in keys:
    total = 0
    for r in amr_class:
        if norm(r.get(key_col, "")) == k:
            total += as_float(r.get(val_col, 0))
    totals.append((k, total))
totals.sort(key=lambda x: (-x[1], x[0]))
k2 = [x[0] for x in totals]
v2 = [x[1] for x in totals]
bars = ax.barh([wrap(x, 20) for x in k2], v2, color=palette_list("amber", len(v2)), edgecolor="#4f2f00", linewidth=0.7, zorder=3)
ax.invert_yaxis()
style_ax(ax, "B. Total burden of dominant AMR classes", "Total count", None, "x")
annot_barh(ax, bars, "{:.0f}", xpad=0.01)

fig.suptitle("Figure 4. Antimicrobial resistance class architecture", y=1.01, fontsize=20, fontweight="bold")
save_png_pdf(fig, "Figure04_AMR_Class_Architecture_G3", OUT)

# =====================================
# Figure 5: gene signatures
# =====================================
fig = plt.figure(figsize=(15.8, 10.0))
gs = GridSpec(2, 1, figure=fig, height_ratios=[1.0, 1.0], hspace=0.34)

# A AMR genes
ax = fig.add_subplot(gs[0, 0])
row_col = guess_col(amr_gene, ["TopSpecies", "Species"])
key_col = guess_col(amr_gene, ["AMR_Gene", "Gene", "AMR_TopGene"])
val_col = guess_col(amr_gene, ["Count", "Hits"])
im, data, keys = heatmap_from_long(
    ax, amr_gene, sp_order_major, row_col, key_col, val_col,
    top_n=16, cmap_name="blue",
    title="A. Dominant AMR gene signatures",
    annotate=False, log1p=True
)
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cb.set_label("log(1 + count)")

# B virulence genes
ax = fig.add_subplot(gs[1, 0])
row_col = guess_col(vf_gene, ["TopSpecies", "Species"])
key_col = guess_col(vf_gene, ["VFDB_Gene", "Virulence_Gene", "Gene"])
val_col = guess_col(vf_gene, ["Count", "Hits"])
im, data, keys = heatmap_from_long(
    ax, vf_gene, sp_order_major, row_col, key_col, val_col,
    top_n=16, cmap_name="rose",
    title="B. Dominant virulence gene signatures",
    annotate=False, log1p=True
)
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cb.set_label("log(1 + count)")

fig.suptitle("Figure 5. Gene-level resistance and virulence landscape", y=1.01, fontsize=20, fontweight="bold")
save_png_pdf(fig, "Figure05_Gene_Signatures_G3", OUT)

# =====================================
# Figure 6: virulence/plasmid prevalence + replicon architecture
# =====================================
fig = plt.figure(figsize=(15.6, 10.6))
gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.0], width_ratios=[1.05, 1.25], hspace=0.32, wspace=0.26)

# A prevalence grouped bars
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
ax.barh(y - h/2, vf_pct, height=h, color="#c23b6f", label="VFDB-positive samples", zorder=3)
ax.barh(y + h/2, pl_pct, height=h, color="#6a4bd9", label="Plasmid-positive samples", zorder=3)
ax.set_yticks(y)
ax.set_yticklabels([species_short(s) for s in species])
ax.invert_yaxis()
style_ax(ax, "A. Prevalence of biological feature carriage", "Samples with feature (%)", None, "x")
ax.legend(loc="lower right", frameon=False)
ax.set_xlim(0, 105)

# B plasmid replicon heatmap
ax = fig.add_subplot(gs[0, 1])
if plasmid_rep:
    row_col = guess_col(plasmid_rep, ["TopSpecies", "Species"])
    key_col = guess_col(plasmid_rep, ["Plasmid_Replicon", "Replicon"])
    val_col = guess_col(plasmid_rep, ["Count", "Hits"])
    im, data, keys = heatmap_from_long(
        ax, plasmid_rep, sp_order_all, row_col, key_col, val_col,
        top_n=10, cmap_name="violet",
        title="B. Dominant plasmid replicon architecture",
        annotate=True, log1p=True
    )
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("log(1 + count)")
else:
    ax.axis("off")
    ax.text(0.5, 0.5, "No plasmid replicon table available", ha="center", va="center", fontsize=12)

# C serotype mini-panel
ax = fig.add_subplot(gs[1, 1])
if serotype:
    sero_counter = Counter([norm(r["Serotype_Call"]) for r in serotype if norm(r["Serotype_Call"])])
    labs = list(sero_counter.keys())
    vals = [sero_counter[k] for k in labs]
    bars = ax.barh([wrap(x, 18) for x in labs], vals, color=palette_list("green", len(vals)), edgecolor="#0f5b26", linewidth=0.7, zorder=3)
    ax.invert_yaxis()
    style_ax(ax, "C. Informative serotype calls", "Samples", None, "x")
    annot_barh(ax, bars, "{:.0f}", xpad=0.02)
else:
    ax.axis("off")
    ax.text(0.5, 0.5, "No informative serotype calls", ha="center", va="center", fontsize=12)

fig.suptitle("Figure 6. Virulence, plasmid, and serotype architecture", y=1.01, fontsize=20, fontweight="bold")
save_png_pdf(fig, "Figure06_Virulence_Plasmid_Serotype_G3", OUT)

print("Rendered core premium suite into:", OUT)
