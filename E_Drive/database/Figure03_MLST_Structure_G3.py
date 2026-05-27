import os
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from figure_helper_common import (
    setup_rcparams, read_csv, as_int, norm, species_short, wrap,
    save_png_pdf, style_ax, palette_list
)

setup_rcparams()

WORK = "/mnt/e/DrAhmed/Ongoing/WGS/Result"
CLEAN = f"{WORK}/_G3/clean"
OUTDIR = f"{WORK}/_G3/output/main"
os.makedirs(OUTDIR, exist_ok=True)

rows = read_csv(f"{CLEAN}/mlst_species_st_counts_clean_166.csv")

tmp = defaultdict(int)
for r in rows:
    tmp[norm(r["TopSpecies"])] += as_int(r["Count"])

species_order = [sp for sp, n in sorted(tmp.items(), key=lambda x: (-x[1], x[0])) if n >= 2]

panel_data = {}
for sp in species_order:
    rr = [r for r in rows if norm(r["TopSpecies"]) == sp]
    rr.sort(key=lambda x: (-as_int(x["Count"]), norm(x["Scheme_ST"])))
    top = rr[:12]
    rest = rr[12:]

    data = [(norm(x["Scheme_ST"]).replace(" | ", " "), as_int(x["Count"])) for x in top]
    rest_n = sum(as_int(x["Count"]) for x in rest)
    if rest_n > 0:
        data.append(("Other", rest_n))
    panel_data[sp] = data

n = len(species_order)
ncols = 2 if n <= 4 else 3
nrows = int(np.ceil(n / ncols))

fig = plt.figure(figsize=(15.5, 4.6 * nrows))
gs = GridSpec(nrows, ncols, figure=fig, hspace=0.42, wspace=0.28)

cmaps = ["teal", "blue", "violet", "amber", "rose", "green"]

for i, sp in enumerate(species_order):
    ax = fig.add_subplot(gs[i // ncols, i % ncols])
    data = panel_data[sp]

    labels = [wrap(x[0], 22) for x in data][::-1]
    vals = [x[1] for x in data][::-1]
    colors = palette_list(cmaps[i % len(cmaps)], len(vals))

    bars = ax.barh(labels, vals, color=colors, edgecolor="#243447", linewidth=0.7, zorder=3)
    style_ax(ax, f"{chr(65+i)}. {species_short(sp)}", "Samples", None, "x")

    xmax = max(vals) if vals else 1
    ax.set_xlim(0, xmax * 1.22)

    for b in bars:
        v = b.get_width()
        y = b.get_y() + b.get_height() / 2
        ax.text(v + xmax * 0.03, y, str(int(v)), va="center", ha="left", fontsize=9)

for j in range(i + 1, nrows * ncols):
    ax = fig.add_subplot(gs[j // ncols, j % ncols])
    ax.axis("off")

fig.suptitle("Figure 3. MLST population structure within resolved species groups", y=1.01, fontsize=20, fontweight="bold")
save_png_pdf(fig, "Figure03_MLST_Structure_G3", OUTDIR)
print("Saved Figure 3 to:", OUTDIR)
