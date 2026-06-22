import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from figure_helper_wgs_remaining import (
    setup_rcparams, read_csv, as_float, norm,
    species_short, SPECIES_COLORS,
    major_species_order, save_png_pdf,
    load_confidence_map, confidence_of
)

setup_rcparams()

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 17,
    "axes.labelsize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 11,
})

# =========================================================
# Paths
# =========================================================
WORK = "/mnt/e/DrAhmed/Ongoing/WGS/Result"
MASTER = f"{WORK}/_BIOLOGY_LAYER_V2/PrimaryResults_v4_withBiology_166.csv"
OUTDIR = f"{WORK}/_G4_REMAINING/output/main"
os.makedirs(OUTDIR, exist_ok=True)

# =========================================================
# Helpers
# =========================================================
def style_axis(ax, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#6b7280")
    ax.spines["bottom"].set_color("#6b7280")

    if grid_axis in ("x", "both"):
        ax.grid(axis="x", color="#e5e7eb", linewidth=0.8, zorder=0)
    if grid_axis in ("y", "both"):
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

def fmt_int(x, pos):
    try:
        return f"{int(x):,}"
    except Exception:
        return str(x)

def draw_distribution_panel(
    ax,
    rows,
    species_order,
    col,
    title,
    letter,
    ylabel,
    scale=1.0,
    rng=None
):
    data = []
    labels = []
    colors = []
    counts = []

    for sp in species_order:
        vals = [as_float(r.get(col, 0.0)) / scale for r in rows if norm(r.get("TopSpecies1", "")) == sp]
        vals = [v for v in vals if v > 0]
        if not vals:
            continue
        data.append(vals)
        labels.append(sp)
        counts.append(len(vals))
        colors.append(SPECIES_COLORS.get(sp, "#64748b"))

    pos = np.arange(1, len(data) + 1)

    # Violin layer
    vp = ax.violinplot(
        data,
        positions=pos,
        widths=0.78,
        showmeans=False,
        showmedians=False,
        showextrema=False
    )
    for body, c in zip(vp["bodies"], colors):
        body.set_facecolor(c)
        body.set_edgecolor(c)
        body.set_alpha(0.18)

    # Box layer
    bp = ax.boxplot(
        data,
        positions=pos,
        patch_artist=True,
        widths=0.34,
        medianprops=dict(color="#111827", linewidth=1.5),
        boxprops=dict(linewidth=0.95, color="#4b5563"),
        whiskerprops=dict(linewidth=0.9, color="#6b7280"),
        capprops=dict(linewidth=0.9, color="#6b7280"),
        flierprops=dict(marker="o", markersize=0)
    )
    for box, c in zip(bp["boxes"], colors):
        box.set_facecolor(c)
        box.set_alpha(0.32)

    # Jittered points
    for x, vals, c in zip(pos, data, colors):
        jit = rng.normal(0, 0.040, size=len(vals)) if rng is not None else np.zeros(len(vals))
        ax.scatter(
            np.full(len(vals), x) + jit,
            vals,
            s=16,
            alpha=0.72,
            zorder=3,
            color=c,
            edgecolor="white",
            linewidth=0.35
        )

    ax.set_xticks(pos)
    ax.set_xticklabels(
        [f"{species_short(sp)}\n(n={n})" for sp, n in zip(labels, counts)],
        linespacing=1.06
    )
    ax.tick_params(axis="x", pad=8)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{letter}. {title}", loc="left", pad=12, fontweight="bold")
    style_axis(ax, "y")
    ax.margins(x=0.08)

# =========================================================
# Load data
# =========================================================
rows = read_csv(MASTER)
conf_map, _, _ = load_confidence_map(WORK)
major_species = major_species_order(rows, min_n=4)
rows = [r for r in rows if norm(r.get("TopSpecies1", "")) in major_species]

for r in rows:
    r["ConfidenceClass"] = confidence_of(norm(r.get("Sample", "")), conf_map)

rng = np.random.default_rng(42)

# bubble sizing based on total length in Mb
all_lengths_mb = [as_float(r.get("TotalLength", 0)) / 1e6 for r in rows if as_float(r.get("TotalLength", 0)) > 0]
len_lo = min(all_lengths_mb) if all_lengths_mb else 0.0
len_hi = max(all_lengths_mb) if all_lengths_mb else 1.0

def bubble_size(length_mb):
    if len_hi <= len_lo:
        return 120
    frac = (length_mb - len_lo) / (len_hi - len_lo)
    frac = max(0.0, min(1.0, frac))
    return 60 + frac * 220

# =========================================================
# Figure layout: 2 x 2 only
# A B top
# C D bottom
# legends outside figure area on right
# =========================================================
fig = plt.figure(figsize=(20.6, 14.0))
gs = GridSpec(
    2, 2,
    figure=fig,
    hspace=0.44,
    wspace=0.24
)

# =========================================================
# Panel A: N50
# =========================================================
axA = fig.add_subplot(gs[0, 0])
draw_distribution_panel(
    axA, rows, major_species,
    col="N50",
    title="N50 by major species",
    letter="A",
    ylabel="N50 (kb)",
    scale=1000.0,
    rng=rng
)
axA.yaxis.set_major_formatter(FuncFormatter(fmt_int))

# =========================================================
# Panel B: contigs
# =========================================================
axB = fig.add_subplot(gs[0, 1])
draw_distribution_panel(
    axB, rows, major_species,
    col="Contigs",
    title="Contig count by major species",
    letter="B",
    ylabel="Contig count",
    scale=1.0,
    rng=rng
)
axB.yaxis.set_major_formatter(FuncFormatter(fmt_int))

# =========================================================
# Panel C: total length
# =========================================================
axC = fig.add_subplot(gs[1, 0])
draw_distribution_panel(
    axC, rows, major_species,
    col="TotalLength",
    title="Total assembly length by major species",
    letter="C",
    ylabel="Total assembly length (Mb)",
    scale=1e6,
    rng=rng
)

# =========================================================
# Panel D: assembly landscape
# =========================================================
axD = fig.add_subplot(gs[1, 1])

marker_map = {
    "High-confidence": "o",
    "Priority-review": "^",
    "Other/remaining": "s",
    "Unassigned": "s",
}

for conf in ["High-confidence", "Priority-review", "Other/remaining"]:
    sub = [r for r in rows if r["ConfidenceClass"] == conf]
    x = [as_float(r.get("N50", 0)) / 1000.0 for r in sub]
    y = [as_float(r.get("Contigs", 0)) for r in sub]
    sizes = [bubble_size(as_float(r.get("TotalLength", 0)) / 1e6) for r in sub]
    colors = [SPECIES_COLORS.get(norm(r.get("TopSpecies1", "")), "#64748b") for r in sub]

    axD.scatter(
        x, y,
        s=sizes,
        c=colors,
        marker=marker_map.get(conf, "o"),
        alpha=0.74,
        edgecolor="#1f2937",
        linewidth=0.38,
        zorder=3
    )

# cohort medians
x_med = np.median([as_float(r.get("N50", 0)) / 1000.0 for r in rows if as_float(r.get("N50", 0)) > 0])
y_med = np.median([as_float(r.get("Contigs", 0)) for r in rows if as_float(r.get("Contigs", 0)) > 0])
axD.axvline(x_med, color="#9ca3af", linewidth=1.0, linestyle="--", zorder=1)
axD.axhline(y_med, color="#9ca3af", linewidth=1.0, linestyle="--", zorder=1)

axD.set_title("D. Assembly quality landscape", loc="left", pad=12, fontweight="bold")
axD.set_xlabel("N50 (kb)")
axD.set_ylabel("Contig count")
axD.yaxis.set_major_formatter(FuncFormatter(fmt_int))
style_axis(axD, "both")

# =========================================================
# Legends anchored directly to panel D
# =========================================================
conf_handles = [
    Line2D(
        [0], [0],
        marker=marker_map[k],
        linestyle="",
        markerfacecolor="#94a3b8",
        markeredgecolor="#111827",
        markeredgewidth=0.8,
        markersize=7.5,
        label=k
    )
    for k in ["High-confidence", "Priority-review", "Other/remaining"]
]

species_handles = [
    Line2D(
        [0], [0],
        marker="o",
        linestyle="",
        markerfacecolor=SPECIES_COLORS.get(k, "#64748b"),
        markeredgecolor="none",
        markersize=7.5,
        label=species_short(k)
    )
    for k in major_species
]

size_vals = [4.0, 5.5, 7.0]
size_handles = [
    axD.scatter([], [], s=bubble_size(v), facecolor="#cbd5e1", edgecolor="#1f2937", linewidth=0.6)
    for v in size_vals
]

# =========================================================
# Legends anchored to panel D with equal vertical gaps
# =========================================================
LEGEND_X = 1.01      # same invisible left margin for all 3 legend blocks
LEGEND_TOP = 0.98    # top start position beside panel D
LEGEND_GAP = 0.045   # equal gap between legend blocks (axes fraction)

def legend_height_in_axes(ax, legend, fig):
    """Return rendered legend height in axes-coordinate units."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = legend.get_window_extent(renderer=renderer)
    inv = ax.transAxes.inverted()
    y0 = inv.transform((0, bbox.y0))[1]
    y1 = inv.transform((0, bbox.y1))[1]
    return (y1 - y0)

# Legend 1
leg1 = axD.legend(
    handles=conf_handles,
    title="Confidence",
    frameon=False,
    loc="upper left",
    bbox_to_anchor=(LEGEND_X, LEGEND_TOP),
    borderaxespad=0.0,
    handletextpad=0.45,
    labelspacing=0.35,
    fontsize=9.2,
    title_fontsize=10.2
)
axD.add_artist(leg1)

h1 = legend_height_in_axes(axD, leg1, fig)

# Legend 2 (placed with equal gap below legend 1)
y2 = LEGEND_TOP - h1 - LEGEND_GAP
leg2 = axD.legend(
    handles=species_handles,
    title="Species",
    frameon=False,
    loc="upper left",
    bbox_to_anchor=(LEGEND_X, y2),
    borderaxespad=0.0,
    handletextpad=0.45,
    labelspacing=0.32,
    fontsize=9.2,
    title_fontsize=10.2
)
axD.add_artist(leg2)

h2 = legend_height_in_axes(axD, leg2, fig)

# Legend 3 (placed with equal gap below legend 2)
y3 = y2 - h2 - LEGEND_GAP
leg3 = axD.legend(
    handles=size_handles,
    labels=[f"{v:.1f} Mb" for v in size_vals],
    title="Bubble size\n(total length)",
    frameon=False,
    loc="upper left",
    bbox_to_anchor=(LEGEND_X, y3),
    borderaxespad=0.0,
    handletextpad=0.60,
    labelspacing=0.35,
    fontsize=9.2,
    title_fontsize=10.2
)
axD.add_artist(leg3)

# =========================================================
# Title and footer
# =========================================================
fig.suptitle(
    "Assembly quality and genome architecture",
    y=0.982,
    fontsize=23,
    fontweight="bold"
)

fig.text(
    0.5, 0.020,
    "Panels A–C summarize N50, contig burden, and total assembly length across major species. "
    "Panel D integrates N50, contig count, species identity, confidence class, and assembly span in a single landscape view.",
    ha="center", va="bottom",
    fontsize=10.5, color="#4b5563"
)

fig.subplots_adjust(
    left=0.07,
    right=0.81,
    top=0.91,
    bottom=0.10
)

save_png_pdf(fig, "Figure07_Assembly_QC_Architecture", OUTDIR)
print("Saved Figure 7 to:", OUTDIR)