import os
import math
import matplotlib.pyplot as plt

from figure_helper_common import (
    setup_rcparams, read_csv, as_float, as_int, species_short,
    palette_list, save_png_pdf
)

setup_rcparams()

WORK = "/mnt/e/DrAhmed/Ongoing/WGS/Result"
INDIR = f"{WORK}/_FIGURE_RAW_MATERIALS_V2"
OUTDIR = f"{WORK}/_G3/output/main"
os.makedirs(OUTDIR, exist_ok=True)

species_bio = read_csv(f"{INDIR}/species_biology_summary_166.csv")
bio_map = {r["TopSpecies"]: r for r in species_bio}

# Keep major species only
sp_order_major = [r["TopSpecies"] for r in species_bio if as_int(r.get("Samples_n", 0)) >= 4]

metric_specs = [
    ("Median_AMR_Genes_n",   "Median AMR genes",    "blue"),
    ("Median_AMR_Classes_n", "Median AMR classes",  "amber"),
    ("Median_VFDB_Hits_n",   "Median VFDB hits",    "rose"),
    ("Median_Plasmid_Hits_n","Median plasmid hits", "violet"),
]

# -----------------------------
# helper functions
# -----------------------------
def nice_species_label(species):
    n = as_int(bio_map[species].get("Samples_n", 0))
    return f"{species_short(species)}  (n={n})"

def style_axis(ax):
    ax.set_facecolor("white")
    ax.grid(axis="x", color="#D9E2EC", linewidth=0.8, alpha=0.9, zorder=0)
    ax.grid(axis="y", visible=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#A0AEC0")
    ax.spines["bottom"].set_color("#A0AEC0")

    ax.tick_params(axis="x", labelsize=10, colors="#243447")
    ax.tick_params(axis="y", labelsize=11, colors="#243447", pad=6)

def add_bar_labels(ax, bars, values, xmax, is_int=False):
    for bar, v in zip(bars, values):
        y = bar.get_y() + bar.get_height() / 2
        label = f"{int(round(v))}" if is_int else f"{v:.1f}"

        # adaptive placement:
        # big bars -> label inside right edge
        # small bars -> label outside bar
        if xmax > 0 and v >= 0.18 * xmax:
            x = max(v - 0.02 * xmax, 0.01 * xmax)
            ax.text(
                x, y, label,
                va="center", ha="right",
                fontsize=10, color="white", fontweight="bold",
                zorder=5
            )
        else:
            x = v + 0.015 * xmax
            ax.text(
                x, y, label,
                va="center", ha="left",
                fontsize=10, color="#243447", fontweight="bold",
                zorder=5
            )

def make_panel(ax, col, title, cmap_name, panel_letter):
    rows = [bio_map[s] for s in sp_order_major]
    labels = [nice_species_label(s) for s in sp_order_major]
    vals = [as_float(r.get(col, 0)) for r in rows]

    # sort within each panel for cleaner reading
    order = sorted(range(len(vals)), key=lambda i: vals[i], reverse=True)
    labels = [labels[i] for i in order]
    vals   = [vals[i] for i in order]

    colors = palette_list(cmap_name, len(vals))

    bars = ax.barh(
        labels, vals,
        color=colors,
        edgecolor="#243447",
        linewidth=0.8,
        height=0.68,
        zorder=3
    )
    ax.invert_yaxis()

    style_axis(ax)

    ax.set_title(
        f"{panel_letter}. {title}",
        loc="left",
        fontsize=13,
        fontweight="bold",
        color="#1F2937",
        pad=12
    )

    ax.set_xlabel(title, fontsize=11, color="#243447", labelpad=8)

    vmax = max(vals) if vals else 0
    if vmax <= 0:
        xmax = 1
    else:
        # add slightly larger padding for safety
        xmax = vmax * 1.24 + max(0.5, 0.03 * vmax)
    ax.set_xlim(0, xmax)

    is_int = all(abs(v - round(v)) < 1e-9 for v in vals)
    add_bar_labels(ax, bars, vals, xmax, is_int=is_int)

# -----------------------------
# build figure
# -----------------------------
fig, axes = plt.subplots(2, 2, figsize=(17.0, 10.8))
axes = axes.flatten()

for i, (col, ttl, cmap_name) in enumerate(metric_specs):
    make_panel(axes[i], col, ttl, cmap_name, chr(65 + i))

fig.suptitle(
    "Figure 2. Species-level biological burden profile",
    y=0.985,
    fontsize=20,
    fontweight="bold",
    color="#111827"
)

fig.text(
    0.5, 0.012,
    "Species ordered within each panel by descending burden; labels show abbreviated species name and sample count.",
    ha="center", va="bottom",
    fontsize=10.5, color="#4A5568"
)

# More breathing room to eliminate collisions
plt.subplots_adjust(
    left=0.23,
    right=0.98,
    top=0.91,
    bottom=0.08,
    hspace=0.42,
    wspace=0.30
)

save_png_pdf(fig, "Figure02_Biology_Profile_G3", OUTDIR)
print("Saved Figure 2 to:", OUTDIR)