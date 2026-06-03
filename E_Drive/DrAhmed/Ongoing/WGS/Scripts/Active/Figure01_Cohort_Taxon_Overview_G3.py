import os
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt

from figure_helper_common import (
    setup_rcparams, read_csv, as_int, species_short, style_ax,
    palette_list, annot_barh, save_png_pdf
)

setup_rcparams()

WORK = "/mnt/e/DrAhmed/Ongoing/WGS/Result"
INDIR = f"{WORK}/_FIGURE_RAW_MATERIALS_V2"
OUTDIR = f"{WORK}/_G3/output/main"
os.makedirs(OUTDIR, exist_ok=True)

species_counts = read_csv(f"{INDIR}/species_counts_166.csv")
species_bio = read_csv(f"{INDIR}/species_biology_summary_166.csv")

sp_order = [r["TopSpecies"] for r in species_counts]
sp_to_n = {r["TopSpecies"]: as_int(r["Count"]) for r in species_counts}
bio_map = {r["TopSpecies"]: r for r in species_bio}

highc = sum(as_int(r.get("HighConfidence_n", 0)) for r in species_bio)
prio = sum(as_int(r.get("PriorityReview_n", 0)) for r in species_bio)
total = sum(as_int(r.get("Samples_n", 0)) for r in species_bio)
other = max(total - highc - prio, 0)

fig = plt.figure(figsize=(14.6, 10.4))
gs = GridSpec(2, 2, figure=fig, width_ratios=[1.05, 1.45], height_ratios=[1.0, 1.0], hspace=0.35, wspace=0.25)

# Panel A
ax = fig.add_subplot(gs[0, 0])
sizes = [highc, prio, other]
labels = ["High-confidence", "Priority-review", "Other/remaining"]
colors = ["#1f5fbf", "#7b4fd6", "#c7d2e5"]

wedges, _ = ax.pie(
    sizes,
    startangle=90,
    counterclock=False,
    colors=colors,
    wedgeprops=dict(width=0.38, edgecolor="white", linewidth=2),
)
ax.text(0, 0.06, f"{total}", ha="center", va="center", fontsize=24, fontweight="bold")
ax.text(0, -0.14, "samples", ha="center", va="center", fontsize=11, color="#4b5563")
ax.set_title("A. Cohort confidence architecture", loc="left", pad=12)
ax.legend(
    wedges,
    [f"{l} (n={s})" for l, s in zip(labels, sizes)],
    loc="lower center",
    bbox_to_anchor=(0.5, -0.18),
    ncol=1,
    frameon=False,
)

# Panel B
ax = fig.add_subplot(gs[0, 1])
labels2 = [species_short(s) for s in sp_order]
vals2 = [sp_to_n[s] for s in sp_order]
colors2 = palette_list("teal", len(vals2))
bars = ax.barh(labels2, vals2, color=colors2, edgecolor="#18324a", linewidth=0.7, zorder=3)
ax.invert_yaxis()
style_ax(ax, "B. Species distribution across all 166 samples", "Samples", None, "x")
annot_barh(ax, bars, "{:.0f}")
ax.set_xlim(0, max(vals2) * 1.18)

# Panel C
ax = fig.add_subplot(gs[1, :])
species = sp_order
h = [as_int(bio_map[s].get("HighConfidence_n", 0)) for s in species]
p = [as_int(bio_map[s].get("PriorityReview_n", 0)) for s in species]
o = [max(as_int(bio_map[s].get("Samples_n", 0)) - hh - pp, 0) for s, hh, pp in zip(species, h, p)]
y = list(range(len(species)))

ax.barh(y, h, color="#1f5fbf", label="High-confidence", zorder=3)
ax.barh(y, p, left=h, color="#7b4fd6", label="Priority-review", zorder=3)
ax.barh(y, o, left=[hh + pp for hh, pp in zip(h, p)], color="#d8dfeb", label="Other/remaining", zorder=3)

ax.set_yticks(y)
ax.set_yticklabels([species_short(s) for s in species])
ax.invert_yaxis()
style_ax(ax, "C. Confidence composition within species groups", "Samples", None, "x")
ax.legend(loc="lower right", frameon=False, ncol=3)

for yi, tot in zip(y, [as_int(bio_map[s]["Samples_n"]) for s in species]):
    ax.text(tot + max(vals2) * 0.02, yi, str(tot), va="center", ha="left", fontsize=9)

ax.set_xlim(0, max(vals2) * 1.18)

fig.suptitle("Cohort and taxonomic overview", y=1.02, fontsize=20, fontweight="bold")
save_png_pdf(fig, "Figure01_Cohort_Taxon_Overview_G3", OUTDIR)
print("Saved Figure 1 to:", OUTDIR)
