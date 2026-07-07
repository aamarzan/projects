# =========================================================
# Figure 8 — Priority-review architecture
# Premium G4 replacement script with sensitivity analysis
# =========================================================

import os
import re
import textwrap
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from figure_helper_wgs_remaining import (
    setup_rcparams, read_csv, as_float, norm, zscore_rows,
    save_png_pdf, load_confidence_map, confidence_of,
    SPECIES_COLORS, CONF_COLORS, count_semicolon_items,
    build_mlst_label, species_short
)

setup_rcparams()

# =========================================================
# Premium global styling
# =========================================================
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12.5,
    "axes.titlesize": 16.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 13.2,
    "xtick.labelsize": 10.8,
    "ytick.labelsize": 10.8,
    "figure.titlesize": 24,
    "figure.titleweight": "bold",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 700,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# =========================================================
# Paths
# =========================================================
WORK = "/mnt/e/DrAhmed/Ongoing/WGS/Result"
MASTER = f"{WORK}/_BIOLOGY_LAYER_V2/PrimaryResults_v4_withBiology_166.csv"
OUTDIR = f"{WORK}/_G4_REMAINING/output/main"
os.makedirs(OUTDIR, exist_ok=True)

OUTNAME = "Figure08_Priority_Review_Architecture"


# =========================================================
# Helpers
# =========================================================
def wrap_label(s, width=12):
    s = str(s).strip()
    if not s:
        return ""
    return "\n".join(textwrap.wrap(s, width=width, break_long_words=False))


def compact_mlst_label(x):
    s = str(x).strip()
    if not s:
        return ""
    m = re.search(r"(ST[0-9A-Za-z]+)", s)
    if m:
        return m.group(1)
    return ""


def zscore_cols(arr):
    arr = np.asarray(arr, dtype=float)
    out = np.zeros_like(arr, dtype=float)

    for j in range(arr.shape[1]):
        col = arr[:, j]
        mu = np.nanmean(col)
        sd = np.nanstd(col)
        if sd <= 1e-12 or np.isnan(sd):
            out[:, j] = 0.0
        else:
            out[:, j] = (col - mu) / sd

    return out


def style_axis(ax, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#94a3b8")
    ax.spines["bottom"].set_color("#94a3b8")
    ax.tick_params(colors="#334155")

    if grid_axis in ("x", "both"):
        ax.grid(axis="x", color="#e5e7eb", linewidth=0.8, zorder=0)
    if grid_axis in ("y", "both"):
        ax.grid(axis="y", color="#eef2f7", linewidth=0.8, zorder=0)

    ax.set_axisbelow(True)


def add_panel_card(ax, fc="#ffffff", ec="#e5e7eb"):
    ax.set_facecolor(fc)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(ec)
        spine.set_linewidth(1.05)


def premium_diverging_cmap():
    return LinearSegmentedColormap.from_list(
        "priority_diverging",
        [
            "#244c9a",
            "#6f9fd5",
            "#dce9f7",
            "#f7f7f7",
            "#f6c7ba",
            "#df755d",
            "#a50026",
        ]
    )


def premium_warm_cmap():
    return LinearSegmentedColormap.from_list(
        "priority_warm",
        ["#fff7ed", "#fed7aa", "#fb923c", "#dc2626", "#7f1d1d"]
    )


def premium_teal_cmap():
    return LinearSegmentedColormap.from_list(
        "priority_teal",
        ["#ecfeff", "#a5f3fc", "#22d3ee", "#0891b2", "#164e63"]
    )


def bubble_size(plasmid_hits):
    return 42 + 36 * np.sqrt(max(plasmid_hits, 0) + 1)


def draw_manual_hm_legend(fig, ax, cmap, vmin, vmax, label):
    # CONTROL:
    # More negative y = lower gradient.
    # Less negative y = higher gradient.
    # Recommended range: -0.170 to -0.205
    gradient_y = -0.182

    cax = inset_axes(
        ax,
        width="58%",
        height="2.2%",
        loc="lower center",
        bbox_to_anchor=(0.0, gradient_y, 1.0, 1.0),
        bbox_transform=ax.transAxes,
        borderpad=0
    )

    grad = np.linspace(vmin, vmax, 512).reshape(1, -1)
    cax.imshow(
        grad,
        aspect="auto",
        cmap=cmap,
        extent=[vmin, vmax, 0, 1],
        interpolation="nearest"
    )

    cax.set_yticks([])
    cax.set_xlim(vmin, vmax)
    cax.tick_params(axis="x", labelsize=9.5, pad=2, length=3, colors="#334155")
    cax.set_xlabel(label, fontsize=10.5, labelpad=3, color="#334155")

    for sp in cax.spines.values():
        sp.set_visible(False)

    return cax


def selected_species_order(rows):
    counts = Counter(norm(r.get("TopSpecies1", "")) for r in rows if norm(r.get("TopSpecies1", "")))
    species = [sp for sp, n in counts.most_common() if n >= 4]

    preferred = [
        "Serratia marcescens",
        "Acinetobacter baumannii",
        "Klebsiella pneumoniae",
        "Pseudomonas aeruginosa",
        "Escherichia coli",
    ]

    species = sorted(
        species,
        key=lambda sp: preferred.index(sp) if sp in preferred else 999
    )

    return species


def make_species_legend_handles(species_list):
    return [
        Patch(
            facecolor=SPECIES_COLORS.get(sp, "#cbd5e1"),
            edgecolor="white",
            label=species_short(sp)
        )
        for sp in species_list
    ]


def make_confidence_legend_handles():
    labels = ["Priority-review", "High-confidence"]
    return [
        Patch(
            facecolor=CONF_COLORS.get(label, "#cbd5e1"),
            edgecolor="white",
            label=label
        )
        for label in labels
    ]


# =========================================================
# Load and derive data
# =========================================================
rows = read_csv(MASTER)
conf_map, highc, prio = load_confidence_map(WORK)

features = [
    ("UnclassifiedPct", "Unclassified %"),
    ("Contigs", "Contigs"),
    ("N50", "N50"),
    ("TotalLength", "Total length"),
    ("GC_percent", "GC%"),
    ("AMRFinder_Hits", "AMR genes"),
    ("AMR_Class_Count", "AMR classes"),
    ("VFDB_Hits", "VFDB hits"),
    ("Plasmid_Hits", "Plasmid hits"),
]

feature_cols = [c for c, _ in features]
feature_labels = [l for _, l in features]

for idx, r in enumerate(rows):
    r["_RowIndex"] = idx
    r["ConfidenceClass"] = confidence_of(norm(r.get("Sample", "")), conf_map)
    r["AMR_Class_Count"] = count_semicolon_items(r.get("AMR_Classes", ""))
    r["MLST_Label"] = build_mlst_label(r)
    r["MLST_Label_Short"] = compact_mlst_label(r["MLST_Label"])

major_species = selected_species_order(rows)

raw_all = np.asarray(
    [[as_float(r.get(col, 0)) for col in feature_cols] for r in rows],
    dtype=float
)
z_all = zscore_cols(raw_all)
global_score = np.mean(np.abs(z_all), axis=1)

for i, r in enumerate(rows):
    r["_PriorityScore"] = float(global_score[i])


# =========================================================
# Select priority-review samples + compact high-confidence references
# =========================================================
priority_rows = [r for r in rows if r["ConfidenceClass"] == "Priority-review"]

reference_rows = []
for sp in major_species:
    all_sp_idx = [
        r["_RowIndex"] for r in rows
        if norm(r.get("TopSpecies1", "")) == sp
    ]
    hc_sp = [
        r for r in rows
        if r["ConfidenceClass"] == "High-confidence"
        and norm(r.get("TopSpecies1", "")) == sp
    ]

    if not hc_sp:
        continue

    sp_centroid = np.median(z_all[all_sp_idx, :], axis=0)

    def ref_distance(r):
        idx = r["_RowIndex"]
        return float(np.mean(np.abs(z_all[idx, :] - sp_centroid)))

    hc_sp.sort(key=lambda r: (ref_distance(r), norm(r.get("Sample", ""))))
    reference_rows.append(hc_sp[0])

seen = set()
selected = []

for r in priority_rows + reference_rows:
    sample = norm(r.get("Sample", ""))
    if sample and sample not in seen:
        seen.add(sample)
        selected.append(r)

selected.sort(
    key=lambda r: (
        0 if r["ConfidenceClass"] == "Priority-review" else 1,
        major_species.index(norm(r.get("TopSpecies1", "")))
        if norm(r.get("TopSpecies1", "")) in major_species else 999,
        -r["_PriorityScore"],
        norm(r.get("Sample", ""))
    )
)

n_sel = len(selected)

sample_labels = [norm(r.get("Sample", "")) for r in selected]
species_track = [norm(r.get("TopSpecies1", "")) for r in selected]
conf_track = [r["ConfidenceClass"] for r in selected]
mlst_labels = [r["MLST_Label_Short"] for r in selected]
selected_idx = [r["_RowIndex"] for r in selected]

raw_mat = np.asarray(
    [[as_float(r.get(col, 0)) for col in feature_cols] for r in selected],
    dtype=float
)
zmat_row = np.asarray(zscore_rows(raw_mat.tolist()), dtype=float)
selected_global_z = z_all[selected_idx, :]
deviation_score = np.asarray([r["_PriorityScore"] for r in selected], dtype=float)

priority_mask = np.array([c == "Priority-review" for c in conf_track], dtype=bool)
ref_mask = np.array([c == "High-confidence" for c in conf_track], dtype=bool)

priority_idx_all = [r["_RowIndex"] for r in rows if r["ConfidenceClass"] == "Priority-review"]
highconf_idx_all = [r["_RowIndex"] for r in rows if r["ConfidenceClass"] == "High-confidence"]

if priority_idx_all:
    priority_median = np.median(z_all[priority_idx_all, :], axis=0)

    if ref_mask.any():
        ref_median = np.median(selected_global_z[ref_mask, :], axis=0)
    elif highconf_idx_all:
        ref_median = np.median(z_all[highconf_idx_all, :], axis=0)
    else:
        ref_median = np.median(z_all, axis=0)

    priority_feature_delta = priority_median - ref_median
else:
    priority_feature_delta = np.zeros(len(feature_cols), dtype=float)


# =========================================================
# Figure setup
# =========================================================
fig_h = max(18.0, 0.35 * max(n_sel, 20) + 8.7)
fig = plt.figure(figsize=(29.2, fig_h), facecolor="white")

# Reduced left-right gap and gave more usable space to Panels B-F
outer = GridSpec(
    1, 2,
    figure=fig,
    width_ratios=[1.34, 1.28],
    wspace=0.055
)

# Narrower metadata column removes the large blank gap after Part A
left = outer[0, 0].subgridspec(
    1, 4,
    width_ratios=[0.042, 0.042, 0.705, 0.215],
    wspace=0.018
)

# Larger right-side panels with tighter but safe spacing
right = outer[0, 1].subgridspec(
    3, 2,
    height_ratios=[1.02, 1.12, 0.90],
    hspace=0.46,
    wspace=0.32
)

hm_cmap = premium_diverging_cmap()
warm_cmap = premium_warm_cmap()
teal_cmap = premium_teal_cmap()


# =========================================================
# Panel A — species strip
# =========================================================
ax_species = fig.add_subplot(left[0, 0])
rgba_species = np.array(
    [mcolors.to_rgba(SPECIES_COLORS.get(c, "#e5e7eb")) for c in species_track]
).reshape(len(species_track), 1, 4)

ax_species.imshow(rgba_species, aspect="auto", interpolation="nearest")
ax_species.set_xticks([])
ax_species.set_yticks([])
ax_species.set_title("")
ax_species.text(
    0.50, 1.018,
    "Species",
    transform=ax_species.transAxes,
    rotation=90,
    ha="center",
    va="bottom",
    fontsize=8.8,
    fontweight="bold",
    color="#111827",
    clip_on=False
)
for sp in ax_species.spines.values():
    sp.set_visible(False)


# =========================================================
# Panel A — confidence strip
# =========================================================
ax_conf = fig.add_subplot(left[0, 1])
rgba_conf = np.array(
    [mcolors.to_rgba(CONF_COLORS.get(c, "#e5e7eb")) for c in conf_track]
).reshape(len(conf_track), 1, 4)

ax_conf.imshow(rgba_conf, aspect="auto", interpolation="nearest")
ax_conf.set_xticks([])
ax_conf.set_yticks([])
ax_conf.set_title("")
ax_conf.text(
    0.50, 1.018,
    "Review",
    transform=ax_conf.transAxes,
    rotation=90,
    ha="center",
    va="bottom",
    fontsize=8.8,
    fontweight="bold",
    color="#111827",
    clip_on=False
)
for sp in ax_conf.spines.values():
    sp.set_visible(False)


# =========================================================
# Panel A — heatmap
# =========================================================
ax_hm = fig.add_subplot(left[0, 2])
add_panel_card(ax_hm)

im = ax_hm.imshow(
    zmat_row,
    aspect="auto",
    cmap=hm_cmap,
    vmin=-2.5,
    vmax=2.5,
    interpolation="nearest"
)

ax_hm.set_yticks(np.arange(n_sel))
ax_hm.set_yticklabels(sample_labels, fontsize=9.4)
ax_hm.tick_params(axis="y", pad=3)

for tick, c in zip(ax_hm.get_yticklabels(), conf_track):
    if c == "Priority-review":
        tick.set_fontweight("bold")
        tick.set_color("#000000")
    else:
        tick.set_color("#475569")

ax_hm.set_xticks(np.arange(len(feature_labels)))
ax_hm.set_xticklabels(
    [wrap_label(x, 9) for x in feature_labels],
    rotation=34,
    ha="right",
    fontsize=10.2
)
ax_hm.tick_params(axis="x", pad=9)

ax_hm.set_title(
    "A. Priority-review architecture heatmap",
    fontsize=16.8,
    pad=13,
    loc="left",
    fontweight="bold"
)
ax_hm.set_xlabel("Within-sample scaled assembly and biological features", fontsize=12.8, labelpad=12)
ax_hm.set_ylabel("Samples", fontsize=12.8, labelpad=11)

ax_hm.set_xticks(np.arange(-0.5, len(feature_labels), 1), minor=True)
ax_hm.set_yticks(np.arange(-0.5, n_sel, 1), minor=True)
ax_hm.grid(which="minor", color="#f1f5f9", linewidth=0.72)
ax_hm.tick_params(which="minor", bottom=False, left=False)

# Separator between priority samples and compact references
if priority_mask.any() and ref_mask.any():
    last_priority = np.where(priority_mask)[0].max()
    ax_hm.axhline(last_priority + 0.5, color="#111827", lw=1.15, alpha=0.72)

draw_manual_hm_legend(
    fig=fig,
    ax=ax_hm,
    cmap=hm_cmap,
    vmin=-2.5,
    vmax=2.5,
    label="Within-sample z-score"
)


# =========================================================
# Panel A — metadata column
# =========================================================
ax_meta = fig.add_subplot(left[0, 3])
add_panel_card(ax_meta)
ax_meta.set_xlim(0, 1)
ax_meta.set_ylim(-0.5, n_sel - 0.5)
ax_meta.invert_yaxis()
ax_meta.axis("off")
ax_meta.set_title("Species / ST", fontsize=10.8, pad=8, fontweight="bold")

for i, (sp, st, c) in enumerate(zip(species_track, mlst_labels, conf_track)):
    txt = species_short(sp)
    if st:
        txt = f"{txt} | {st}"

    ax_meta.text(
        0.035, i, txt,
        va="center",
        ha="left",
        fontsize=9.2,
        fontweight="bold" if c == "Priority-review" else "normal",
        color="#111827" if c == "Priority-review" else "#475569"
    )

# =========================================================
# Panel B — selected cohort composition
# =========================================================
axB = fig.add_subplot(right[0, 0])
add_panel_card(axB)
style_axis(axB, "x")

sp_counts_priority = []
sp_counts_ref = []
sp_labels = []

for sp in major_species:
    c1 = sum(
        1 for r in selected
        if norm(r.get("TopSpecies1", "")) == sp
        and r["ConfidenceClass"] == "Priority-review"
    )
    c2 = sum(
        1 for r in selected
        if norm(r.get("TopSpecies1", "")) == sp
        and r["ConfidenceClass"] == "High-confidence"
    )

    if c1 + c2 > 0:
        sp_labels.append(species_short(sp))
        sp_counts_priority.append(c1)
        sp_counts_ref.append(c2)

y = np.arange(len(sp_labels))

axB.barh(
    y,
    sp_counts_priority,
    color="#d9485f",
    edgecolor="#7f1d1d",
    linewidth=0.75,
    height=0.64,
    label="Priority-review",
    zorder=3
)

axB.barh(
    y,
    sp_counts_ref,
    left=sp_counts_priority,
    color="#64748b",
    edgecolor="#374151",
    linewidth=0.75,
    height=0.64,
    label="High-confidence ref",
    zorder=3
)

for yi, a, b in zip(y, sp_counts_priority, sp_counts_ref):
    if a > 0:
        axB.text(
            a / 2,
            yi,
            str(a),
            va="center",
            ha="center",
            fontsize=10.5,
            color="white",
            fontweight="bold"
        )
    if b > 0:
        axB.text(
            a + b / 2,
            yi,
            str(b),
            va="center",
            ha="center",
            fontsize=10.5,
            color="white",
            fontweight="bold"
        )

axB.set_yticks(y)
axB.set_yticklabels(sp_labels, fontsize=10.8)
axB.invert_yaxis()
axB.set_xlabel("Selected samples")
axB.set_title("B. Selected cohort composition", loc="left", pad=10)
axB.legend(frameon=False, fontsize=9.4, loc="lower right")


# =========================================================
# Panel C — priority-score ranked samples
# =========================================================
axC = fig.add_subplot(right[0, 1])
add_panel_card(axC)
style_axis(axC, "x")

priority_selected = [r for r in selected if r["ConfidenceClass"] == "Priority-review"]
priority_selected.sort(key=lambda r: (-r["_PriorityScore"], norm(r.get("Sample", ""))))
ranked = priority_selected[:12]

if ranked:
    labels = [norm(r.get("Sample", "")) for r in ranked]
    vals = [r["_PriorityScore"] for r in ranked]
    colors = [
        SPECIES_COLORS.get(norm(r.get("TopSpecies1", "")), "#94a3b8")
        for r in ranked
    ]

    y = np.arange(len(ranked))
    axC.barh(
        y,
        vals,
        color=colors,
        edgecolor="#334155",
        linewidth=0.65,
        height=0.62,
        zorder=3
    )

    axC.set_yticks(y)
    axC.set_yticklabels([wrap_label(s, 18) for s in labels], fontsize=9.4)
    axC.invert_yaxis()
    axC.set_xlabel("Composite deviation score")
    axC.set_title("C. Highest-priority samples", loc="left", pad=10)

    xmax = max(vals) * 1.16 if vals else 1
    axC.set_xlim(0, xmax)

    for yi, v in zip(y, vals):
        axC.text(
            v + xmax * 0.018,
            yi,
            f"{v:.2f}",
            va="center",
            ha="left",
            fontsize=9.4,
            fontweight="bold",
            color="#111827"
        )
else:
    axC.axis("off")
    axC.text(
        0.5,
        0.5,
        "No priority-review samples detected",
        ha="center",
        va="center",
        fontsize=11.5,
        fontweight="bold",
        color="#64748b",
        transform=axC.transAxes
    )


# =========================================================
# Panel D — biological burden bubble plot
# =========================================================
axD = fig.add_subplot(right[1, 0])
add_panel_card(axD)
style_axis(axD, "both")

x = np.array([as_float(r.get("AMRFinder_Hits", 0)) for r in selected], dtype=float)
y = np.array([as_float(r.get("VFDB_Hits", 0)) for r in selected], dtype=float)
p = np.array([as_float(r.get("Plasmid_Hits", 0)) for r in selected], dtype=float)

edge_colors = [
    "#7f1d1d" if c == "Priority-review" else "#475569"
    for c in conf_track
]
line_widths = [
    1.25 if c == "Priority-review" else 0.85
    for c in conf_track
]

sc = axD.scatter(
    x,
    y,
    s=[bubble_size(v) for v in p],
    c=deviation_score,
    cmap=teal_cmap,
    alpha=0.88,
    edgecolor=edge_colors,
    linewidth=line_widths,
    zorder=3
)

prio_indices = [i for i, c in enumerate(conf_track) if c == "Priority-review"]
prio_rank = sorted(prio_indices, key=lambda i: deviation_score[i], reverse=True)[:5]

offsets = [(14, 10), (14, -12), (-14, 13), (-14, -13), (12, 18)]

for k, i in enumerate(prio_rank):
    dx, dy = offsets[k % len(offsets)]
    axD.annotate(
        sample_labels[i],
        (x[i], y[i]),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=8.8,
        ha="left" if dx > 0 else "right",
        va="bottom" if dy > 0 else "top",
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#cbd5e1", alpha=0.96),
        arrowprops=dict(arrowstyle="-", color="#94a3b8", lw=0.70),
        zorder=5
    )

axD.set_xlabel("AMR genes")
axD.set_ylabel("VFDB hits")
axD.set_title("D. Biological burden context", loc="left", pad=10)

# Deviation-score colorbar moved lower to avoid the plasmid-size legend
cax = inset_axes(
    axD,
    width="3.2%",
    height="46%",
    loc="lower right",
    bbox_to_anchor=(0.0, 0.065, 1.0, 1.0),
    bbox_transform=axD.transAxes,
    borderpad=1.25
)
cb = fig.colorbar(sc, cax=cax)
cb.set_label("Deviation score", fontsize=10.4, labelpad=5)
cb.ax.tick_params(labelsize=9.2)
cb.outline.set_visible(False)

# Bubble-size legend moved to top-right corner
max_p = int(np.nanmax(p)) if len(p) else 1
legend_vals = sorted(set([1, max(1, max_p // 2), max(1, max_p)]))

handles = [
    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="",
        markerfacecolor="#a5f3fc",
        markeredgecolor="#475569",
        markersize=np.sqrt(bubble_size(v)) / 1.95,
        label=f"{v}"
    )
    for v in legend_vals
]

legD = axD.legend(
    handles=handles,
    title="Plasmid hits",
    frameon=True,
    loc="upper right",
    bbox_to_anchor=(0.985, 0.985),
    borderaxespad=0.25,
    fontsize=9.8,
    title_fontsize=10.2,
    handletextpad=0.8,
    labelspacing=0.45
)
legD.get_frame().set_facecolor("white")
legD.get_frame().set_edgecolor("#e2e8f0")
legD.get_frame().set_alpha(0.92)


# =========================================================
# Panel E — priority feature shifts
# =========================================================
axE = fig.add_subplot(right[1, 1])
add_panel_card(axE)
style_axis(axE, "x")

order = np.argsort(np.abs(priority_feature_delta))[::-1]
feat_plot = [feature_labels[i] for i in order]
vals_plot = [float(priority_feature_delta[i]) for i in order]

max_abs = max([abs(v) for v in vals_plot] + [1e-12])
bar_norm = Normalize(vmin=0, vmax=max_abs)

bar_colors = []
for v in vals_plot:
    if v >= 0:
        bar_colors.append(warm_cmap(0.25 + 0.70 * bar_norm(abs(v))))
    else:
        bar_colors.append(hm_cmap(0.12 + 0.28 * bar_norm(abs(v))))

ypos = np.arange(len(feat_plot))

axE.barh(
    ypos,
    vals_plot,
    color=bar_colors,
    edgecolor="#475569",
    linewidth=0.65,
    height=0.62,
    zorder=3
)

axE.axvline(0, color="#111827", lw=1.0)

xlim = max(max_abs * 1.32, 0.5)
axE.set_xlim(-xlim, xlim)

for yi, val in enumerate(vals_plot):
    txt = f"{val:.2f}"

    if abs(val) >= 0.70:
        axE.text(
            val / 2,
            yi,
            txt,
            ha="center",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            color="white" if val > 0 else "#111827",
            zorder=5
        )
    else:
        offset = xlim * 0.035
        axE.text(
            val + offset if val >= 0 else val - offset,
            yi,
            txt,
            ha="left" if val >= 0 else "right",
            va="center",
            fontsize=9.4,
            fontweight="bold",
            color="#111827",
            zorder=5
        )

axE.set_yticks(ypos)
axE.set_yticklabels([wrap_label(f, 10) for f in feat_plot], fontsize=9.5)
axE.invert_yaxis()
axE.set_xlabel("Priority − reference median z-shift")
axE.set_title("E. Priority feature shifts", loc="left", pad=10)


# =========================================================
# Panel F — sensitivity analysis
# =========================================================
axF = fig.add_subplot(right[2, :])
add_panel_card(axF)
style_axis(axF, "y")

percentiles = [75, 80, 85, 90, 95]
thresholds = np.percentile(global_score, percentiles)

current_priority_all = np.array([
    r["ConfidenceClass"] == "Priority-review"
    for r in rows
], dtype=bool)

flagged_counts = []
overlap_counts = []

for thr in thresholds:
    flagged = global_score >= thr
    flagged_counts.append(int(np.sum(flagged)))
    overlap_counts.append(int(np.sum(flagged & current_priority_all)))

xpos = np.arange(len(percentiles))
bar_w = 0.36

axF.bar(
    xpos - bar_w / 2,
    flagged_counts,
    width=bar_w,
    color="#94a3b8",
    edgecolor="#475569",
    linewidth=0.75,
    label="Score-selected samples",
    zorder=3
)

axF.bar(
    xpos + bar_w / 2,
    overlap_counts,
    width=bar_w,
    color="#d9485f",
    edgecolor="#7f1d1d",
    linewidth=0.75,
    label="Overlap with priority-review",
    zorder=3
)

for x0, total, overlap in zip(xpos, flagged_counts, overlap_counts):
    axF.text(
        x0 - bar_w / 2,
        total + max(flagged_counts) * 0.025,
        str(total),
        ha="center",
        va="bottom",
        fontsize=9.4,
        fontweight="bold",
        color="#111827"
    )
    axF.text(
        x0 + bar_w / 2,
        overlap + max(flagged_counts) * 0.025,
        str(overlap),
        ha="center",
        va="bottom",
        fontsize=9.4,
        fontweight="bold",
        color="#111827"
    )

axF.set_xticks(xpos)
axF.set_xticklabels([f"≥{p}th" for p in percentiles], fontsize=10.0)
axF.set_ylabel("Samples")
axF.set_xlabel("Composite deviation-score cut-off")
axF.set_title("F. Sensitivity analysis of priority-score thresholds", loc="left", pad=10)

axF.set_ylim(0, max(flagged_counts + overlap_counts + [1]) * 1.22)
axF.legend(
    frameon=False,
    loc="upper right",
    ncol=2,
    fontsize=9.4,
    handlelength=1.2,
    columnspacing=1.4
)


# =========================================================
# Panel-specific legends and final layout
# =========================================================
species_handles = make_species_legend_handles(major_species)
conf_handles = make_confidence_legend_handles()

# Panel A species legend: under heatmap, just above the gradient block
# CONTROL:
# More negative y = lower species legend.
# Less negative y = higher species legend.
# Recommended range: -0.125 to -0.155
leg_species_A = ax_hm.legend(
    handles=species_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.135),
    ncol=min(len(species_handles), 5),
    frameon=True,
    fontsize=8.9,
    handlelength=1.0,
    handletextpad=0.35,
    columnspacing=0.85,
    borderpad=0.28
)
leg_species_A.get_frame().set_facecolor("white")
leg_species_A.get_frame().set_edgecolor("#e2e8f0")
leg_species_A.get_frame().set_alpha(0.94)

# Panel C confidence legend: bottom-right corner of Panel C block
leg_conf_C = axC.legend(
    handles=conf_handles,
    loc="lower right",
    bbox_to_anchor=(0.985, 0.035),
    frameon=True,
    fontsize=8.9,
    handlelength=1.0,
    handletextpad=0.45,
    borderpad=0.35,
    labelspacing=0.36
)
leg_conf_C.get_frame().set_facecolor("white")
leg_conf_C.get_frame().set_edgecolor("#e2e8f0")
leg_conf_C.get_frame().set_alpha(0.94)

fig.suptitle(
    "Priority-review architecture with compact high-confidence references",
    y=0.988,
    fontsize=24.5,
    fontweight="bold"
)

fig.subplots_adjust(
    left=0.038,
    right=0.990,
    top=0.928,
    bottom=0.102
)

# =========================================================
# Controlled final typography boost
# =========================================================
# Increases most fonts while protecting dense/overlap-prone labels.
protected_axes_soft = {ax_hm, ax_meta, axC, axD}

for txt in fig.findobj(mpl.text.Text):
    if not txt.get_text().strip():
        continue

    old = txt.get_fontsize()
    parent_ax = txt.axes

    # Heatmap sample labels, metadata labels, ranked labels, and Panel D annotations
    # receive only a small bump to prevent overlap.
    if parent_ax in protected_axes_soft:
        if old <= 9.5:
            new = old + 0.55
        elif old <= 11.0:
            new = old + 0.75
        else:
            new = old + 1.05
    else:
        if old <= 9.5:
            new = old + 1.10
        elif old <= 12.0:
            new = old + 1.35
        elif old <= 17.0:
            new = old + 1.65
        else:
            new = old + 1.90

    txt.set_fontsize(new)

save_png_pdf(fig, OUTNAME, OUTDIR)
print("Saved Figure 8 to:", OUTDIR)