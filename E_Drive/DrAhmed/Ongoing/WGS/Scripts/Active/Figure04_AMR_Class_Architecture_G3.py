import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.gridspec import GridSpec

from figure_helper_common import (
    setup_rcparams,
    read_csv,
    as_int,
    guess_col,
    save_png_pdf,
)

setup_rcparams()

WORK = "/mnt/e/DrAhmed/Ongoing/WGS/Result"
INDIR = f"{WORK}/_FIGURE_RAW_MATERIALS_V2"
OUTDIR = f"{WORK}/_G3/output/main"
os.makedirs(OUTDIR, exist_ok=True)

# =========================================================
# Helpers
# =========================================================
AMBER_CMAP = LinearSegmentedColormap.from_list(
    "premium_amber",
    ["#fbf3d5", "#f4cf6b", "#eea61a", "#c97d00", "#8a5300"]
)

SPECIES_SHORT = {
    "Serratia marcescens": "S. marcescens",
    "Acinetobacter baumannii": "A. baumannii",
    "Klebsiella pneumoniae": "K. pneumoniae",
    "Pseudomonas aeruginosa": "P. aeruginosa",
    "Escherichia coli": "E. coli",
    "Homo sapiens": "H. sapiens",
    "Serratia nevei": "S. nevei",
}

CLASS_LABEL_MAP = {
    "QUATERNARY AMMONIUM": "QUATERNARY\nAMMONIUM",
    "COPPER/SILVER": "COPPER/\nSILVER",
    "AMINOGLYCOSIDE": "AMINO-\nGLYCOSIDE",
    "BETA-LACTAM": "BETA-\nLACTAM",
}

def short_species(x):
    return SPECIES_SHORT.get(str(x).strip(), str(x).strip())

def wrap_class_label(x):
    x = str(x).strip()
    if x in CLASS_LABEL_MAP:
        return CLASS_LABEL_MAP[x]
    if "/" in x and len(x) > 12:
        return x.replace("/", "/\n", 1)
    if " " in x and len(x) > 14:
        parts = x.split()
        mid = len(parts) // 2
        return " ".join(parts[:mid]) + "\n" + " ".join(parts[mid:])
    return x

def as_float(x, default=0.0):
    try:
        s = str(x).strip()
        if s == "":
            return default
        return float(s)
    except Exception:
        return default

def style_clean_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#6b7280")
    ax.spines["bottom"].set_color("#6b7280")

def build_matrix(long_rows, row_order, row_col, key_col, val_col, top_n=14):
    # Total by class across selected species
    totals = {}
    for r in long_rows:
        sp = str(r.get(row_col, "")).strip()
        key = str(r.get(key_col, "")).strip()
        if sp not in row_order or key == "":
            continue
        totals[key] = totals.get(key, 0.0) + as_float(r.get(val_col, 0))

    key_order = [k for k, _ in sorted(totals.items(), key=lambda x: (-x[1], x[0]))[:top_n]]

    row_index = {sp: i for i, sp in enumerate(row_order)}
    key_index = {k: i for i, k in enumerate(key_order)}

    data = np.zeros((len(row_order), len(key_order)), dtype=float)
    for r in long_rows:
        sp = str(r.get(row_col, "")).strip()
        key = str(r.get(key_col, "")).strip()
        if sp not in row_index or key not in key_index:
            continue
        data[row_index[sp], key_index[key]] += as_float(r.get(val_col, 0))

    return data, key_order

# =========================================================
# Load data
# =========================================================
species_counts = read_csv(f"{INDIR}/species_counts_166.csv")
amr_class = read_csv(f"{INDIR}/amr_class_by_species_166.csv")

row_col = guess_col(amr_class, ["TopSpecies", "Species"])
key_col = guess_col(amr_class, ["AMR_Class", "Class"])
val_col = guess_col(amr_class, ["Count", "Hits"])

sp_order_major = [
    r["TopSpecies"]
    for r in species_counts
    if as_int(r.get("Count", 0)) >= 4
]

# =========================================================
# Build matrix
# =========================================================
data, keys = build_matrix(
    amr_class,
    sp_order_major,
    row_col,
    key_col,
    val_col,
    top_n=14,
)

data_log = np.log1p(data)

row_totals = data.sum(axis=1)
col_totals = data.sum(axis=0)

# =========================================================
# Figure layout
# =========================================================
fig = plt.figure(figsize=(18.2, 12.4))
gs = GridSpec(
    2, 2,
    figure=fig,
    height_ratios=[1.26, 1.02],
    width_ratios=[1.0, 0.19],
    hspace=0.38,
    wspace=0.12
)

# =========================================================
# Panel A: heatmap
# =========================================================
ax_hm = fig.add_subplot(gs[0, 0])

im = ax_hm.imshow(
    data_log,
    cmap=AMBER_CMAP,
    aspect="auto",
    interpolation="nearest",
    zorder=2
)

# ticks
ax_hm.set_xticks(range(len(keys)))
ax_hm.set_xticklabels(
    [wrap_class_label(k) for k in keys],
    rotation=34,
    ha="right"
)
ax_hm.tick_params(axis="x", pad=8)

ax_hm.set_yticks(range(len(sp_order_major)))
ax_hm.set_yticklabels([short_species(x) for x in sp_order_major])
ax_hm.tick_params(axis="y", pad=8)

# minor grid for cells
ax_hm.set_xticks(np.arange(-.5, len(keys), 1), minor=True)
ax_hm.set_yticks(np.arange(-.5, len(sp_order_major), 1), minor=True)
ax_hm.grid(which="minor", color="#f3efe3", linewidth=1.2)
ax_hm.tick_params(which="minor", bottom=False, left=False)

style_clean_axis(ax_hm)
ax_hm.set_title(
    "A. Dominant AMR classes across major species",
    loc="left",
    pad=12,
    fontweight="bold"
)

# annotations
vmax = data.max() if data.size else 1
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        val = int(round(data[i, j]))
        if val <= 0:
            continue
        txt_color = "white" if val >= vmax * 0.22 else "#3b2a12"
        ax_hm.text(
            j, i, f"{val}",
            ha="center", va="center",
            fontsize=9.4, fontweight="bold",
            color=txt_color
        )

# colorbar
cb = fig.colorbar(im, ax=ax_hm, fraction=0.026, pad=0.02)
cb.set_label("log(1 + count)")
cb.outline.set_edgecolor("#8a5300")

# =========================================================
# Panel A-right: row totals
# =========================================================
ax_rt = fig.add_subplot(gs[0, 1])

rt_colors = [AMBER_CMAP(0.40 + 0.45 * (v / row_totals.max())) if row_totals.max() > 0 else AMBER_CMAP(0.55)
             for v in row_totals]

bars_rt = ax_rt.barh(
    range(len(sp_order_major)),
    row_totals,
    color=rt_colors,
    edgecolor="#8a5300",
    linewidth=0.8,
    height=0.66,
    zorder=3
)

ax_rt.set_yticks(range(len(sp_order_major)))
ax_rt.set_yticklabels([])
ax_rt.invert_yaxis()
ax_rt.set_xlabel("Row total")
ax_rt.set_title(
    "Species total",
    pad=12,
    fontweight="bold",
    fontsize=12
)
ax_rt.grid(axis="x", color="#e5e7eb", linewidth=0.8)
ax_rt.set_axisbelow(True)
style_clean_axis(ax_rt)

rt_xmax = row_totals.max() if len(row_totals) else 1
ax_rt.set_xlim(0, rt_xmax * 1.22)

for b, val in zip(bars_rt, row_totals):
    ax_rt.text(
        b.get_width() + rt_xmax * 0.03,
        b.get_y() + b.get_height()/2,
        f"{int(round(val))}",
        ha="left", va="center",
        fontsize=9.5, fontweight="bold", color="#4b3a17"
    )

# =========================================================
# Panel B: premium ranked burden chart + cumulative line
# =========================================================
ax_bar = fig.add_subplot(gs[1, :])

x = np.arange(len(keys))
vals = col_totals
norm = Normalize(vmin=vals.min() if len(vals) else 0, vmax=vals.max() if len(vals) else 1)
bar_colors = [AMBER_CMAP(0.30 + 0.60 * norm(v)) for v in vals]

bars = ax_bar.bar(
    x,
    vals,
    width=0.72,
    color=bar_colors,
    edgecolor="#6b3f00",
    linewidth=0.9,
    zorder=3
)

ax_bar.set_xticks(x)
ax_bar.set_xticklabels([wrap_class_label(k) for k in keys], rotation=28, ha="right")
ax_bar.tick_params(axis="x", pad=10)
ax_bar.set_ylabel("Total count")
ax_bar.set_title(
    "B. Total burden of dominant AMR classes",
    loc="left",
    pad=12,
    fontweight="bold"
)

ax_bar.grid(axis="y", color="#e5e7eb", linewidth=0.8)
ax_bar.set_axisbelow(True)
style_clean_axis(ax_bar)

bar_ymax = vals.max() if len(vals) else 1
ax_bar.set_ylim(0, bar_ymax * 1.22)

for rect, val in zip(bars, vals):
    ax_bar.text(
        rect.get_x() + rect.get_width()/2,
        rect.get_height() + bar_ymax * 0.02,
        f"{int(round(val))}",
        ha="center", va="bottom",
        fontsize=9.5, fontweight="bold", color="#4b3a17"
    )

# cumulative percentage line
ax_line = ax_bar.twinx()
cum = np.cumsum(vals) / np.sum(vals) * 100 if np.sum(vals) > 0 else np.zeros_like(vals)
ax_line.plot(
    x,
    cum,
    color="#6d1f00",
    linewidth=2.2,
    marker="o",
    markersize=5.5,
    zorder=4
)
ax_line.set_ylim(0, 105)
ax_line.set_ylabel("Cumulative share (%)")
ax_line.spines["top"].set_visible(False)
ax_line.spines["left"].set_visible(False)
ax_line.spines["right"].set_color("#6b7280")
ax_line.tick_params(axis="y", colors="#6d1f00")

# small legend
line_proxy = plt.Line2D([0], [0], color="#6d1f00", marker="o", linewidth=2.2, markersize=5.5)
ax_bar.legend(
    [line_proxy],
    ["Cumulative share"],
    frameon=False,
    loc="upper left"
)

# =========================================================
# Global title / footnote
# =========================================================
fig.suptitle(
    "Antimicrobial resistance class architecture",
    y=0.985,
    fontsize=23,
    fontweight="bold"
)

fig.text(
    0.5, 0.025,
    "Panel A shows species-by-class burden as a log-scaled heatmap with species totals. "
    "Panel B ranks the same dominant classes by total burden and adds cumulative contribution across the cohort.",
    ha="center", va="bottom",
    fontsize=10.5, color="#555555"
)

fig.subplots_adjust(
    left=0.08,
    right=0.96,
    top=0.91,
    bottom=0.11
)

save_png_pdf(fig, "Figure04_AMR_Class_Architecture_G3", OUTDIR)
print("Saved Figure 4 to:", OUTDIR)