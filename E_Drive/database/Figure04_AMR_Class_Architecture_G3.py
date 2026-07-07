import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from figure_helper_common import (
    setup_rcparams, read_csv, as_int, heatmap_from_long, guess_col,
    save_png_pdf, style_ax, palette_list, annot_barh
)

setup_rcparams()

WORK = "/mnt/e/DrAhmed/Ongoing/WGS/Result"
INDIR = f"{WORK}/_FIGURE_RAW_MATERIALS_V2"
OUTDIR = f"{WORK}/_G3/output/main"
os.makedirs(OUTDIR, exist_ok=True)

species_counts = read_csv(f"{INDIR}/species_counts_166.csv")
amr_class = read_csv(f"{INDIR}/amr_class_by_species_166.csv")
sp_order_major = [r["TopSpecies"] for r in species_counts if as_int(r.get("Count", 0)) >= 4]

fig = plt.figure(figsize=(15.2, 9.9))
gs = GridSpec(2, 1, figure=fig, height_ratios=[1.45, 1.0], hspace=0.35)

# Panel A
ax = fig.add_subplot(gs[0, 0])
row_col = guess_col(amr_class, ["TopSpecies", "Species"])
key_col = guess_col(amr_class, ["AMR_Class", "Class"])
val_col = guess_col(amr_class, ["Count", "Hits"])

im, data, keys = heatmap_from_long(
    ax,
    amr_class,
    sp_order_major,
    row_col,
    key_col,
    val_col,
    top_n=14,
    cmap_name="amber",
    title="A. Dominant AMR classes across major species",
    annotate=True,
    log1p=True,
    xtick_width=12,
)

cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cb.set_label("log(1 + count)")

# Panel B
ax = fig.add_subplot(gs[1, 0])
totals = []
for k in keys:
    total = 0
    for r in amr_class:
        if str(r.get(key_col, "")).strip() == k:
            try:
                total += float(r.get(val_col, 0))
            except Exception:
                pass
    totals.append((k, total))

totals.sort(key=lambda x: (-x[1], x[0]))
k2 = [x[0] for x in totals]
v2 = [x[1] for x in totals]

bars = ax.barh(
    [x.replace(" ", "\n") if len(x) > 18 else x for x in k2],
    v2,
    color=palette_list("amber", len(v2)),
    edgecolor="#4f2f00",
    linewidth=0.7,
    zorder=3,
)
ax.invert_yaxis()
style_ax(ax, "B. Total burden of dominant AMR classes", "Total count", None, "x")
annot_barh(ax, bars, "{:.0f}", xpad=0.01)

fig.suptitle("Figure 4. Antimicrobial resistance class architecture", y=1.01, fontsize=20, fontweight="bold")
save_png_pdf(fig, "Figure04_AMR_Class_Architecture_G3", OUTDIR)
print("Saved Figure 4 to:", OUTDIR)
