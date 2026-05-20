import os
import csv
from collections import defaultdict

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# =========================================================
# Global style
# =========================================================
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12.5,
    "axes.titlesize": 18,
    "axes.titleweight": "bold",
    "axes.labelsize": 14.5,
    "xtick.labelsize": 11.5,
    "ytick.labelsize": 12.0,
    "figure.titlesize": 25,
    "figure.titleweight": "bold",
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# =========================================================
# Paths
# =========================================================
WORK = "/mnt/e/DrAhmed/Ongoing/WGS/Result"
MASTER = f"{WORK}/_BIOLOGY_LAYER_V2/PrimaryResults_v4_withBiology_166.csv"
OUTDIR = f"{WORK}/_G3/output/main"
os.makedirs(OUTDIR, exist_ok=True)

HIGHCONF_CANDIDATES = [
    f"{WORK}/_PRELIMINARY_REPORT_READY/Table5_HighConfidenceSamples.csv",
    f"{WORK}/_MANUSCRIPT_FINAL/Table5_HighConfidenceSamples.csv",
]
PRIORITY_CANDIDATES = [
    f"{WORK}/_PRELIMINARY_REPORT_READY/Table4_PriorityReviewSamples.csv",
    f"{WORK}/_PRELIMINARY_PACK/Table_FlaggedSamples_refined.csv",
]

TREE_IMAGE_CANDIDATES = {
    "Acinetobacter baumannii": [
        f"{WORK}/Result copy/AB_tree_matrix.png",
        f"{WORK}/result_copy/AB_tree_matrix.png",
    ],
    "Serratia marcescens": [
        f"{WORK}/Result copy/SM_tree_matrix.png",
        f"{WORK}/result_copy/SM_tree_matrix.png",
    ],
    "Klebsiella pneumoniae": [
        f"{WORK}/Result copy/KP_tree_matrix.png",
        f"{WORK}/result_copy/KP_tree_matrix.png",
    ],
    "Pseudomonas aeruginosa": [
        f"{WORK}/Result copy/PA_tree_matrix.png",
        f"{WORK}/result_copy/PA_tree_matrix.png",
    ],
}

OUTNAME = "Figure03_Species_Confirmation_Relatedness_Architecture"

# =========================================================
# Helpers
# =========================================================
def first_existing(paths):
    for p in paths:
        if os.path.isfile(p):
            return p
    return None


def read_csv(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def write_png_pdf(fig, outdir, outname):
    png = os.path.join(outdir, outname + ".png")
    pdf = os.path.join(outdir, outname + ".pdf")
    fig.savefig(png, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    print("Saved:", png)
    print("Saved:", pdf)


def as_float(x, default=np.nan):
    try:
        s = str(x).strip()
        if s == "":
            return default
        return float(s)
    except Exception:
        return default


def norm(x):
    return str(x).strip()


def species_short(name):
    parts = norm(name).split()
    if len(parts) >= 2:
        return f"{parts[0][0]}. {' '.join(parts[1:])}"
    return norm(name)


def find_sample_col(rows):
    if not rows:
        return None
    keys = list(rows[0].keys())
    for cand in ["Sample", "sample", "SampleID", "Sample_ID"]:
        if cand in keys:
            return cand
    for k in keys:
        if "sample" in k.lower():
            return k
    return None


def sample_set_from_candidates(paths):
    p = first_existing(paths)
    if not p:
        return set()
    rows = read_csv(p)
    sc = find_sample_col(rows)
    if not sc:
        return set()
    return {norm(r.get(sc, "")) for r in rows if norm(r.get(sc, ""))}


def trim_white(img, white_thresh=0.985):
    arr = np.asarray(img)

    if arr.ndim == 2:
        mask = arr < white_thresh
    else:
        rgb = arr[..., :3]
        mask = np.any(rgb < white_thresh, axis=2)

    if not np.any(mask):
        return arr

    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1

    pad_y = max(2, int((y1 - y0) * 0.02))
    pad_x = max(2, int((x1 - x0) * 0.02))

    y0 = max(0, y0 - pad_y)
    y1 = min(arr.shape[0], y1 + pad_y)
    x0 = max(0, x0 - pad_x)
    x1 = min(arr.shape[1], x1 + pad_x)

    return arr[y0:y1, x0:x1]


def add_soft_panel_bg(ax, face="#f8fafc"):
    ax.set_facecolor(face)
    for s in ax.spines.values():
        s.set_visible(False)


def style_axis(ax, grid_axis="both"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#6b7280")
    ax.spines["bottom"].set_color("#6b7280")
    ax.tick_params(colors="#374151")

    if grid_axis in ("x", "both"):
        ax.grid(axis="x", color="#e5e7eb", linewidth=0.9, zorder=0)
    if grid_axis in ("y", "both"):
        ax.grid(axis="y", color="#eef2f7", linewidth=0.9, zorder=0)

    ax.set_axisbelow(True)


def map_size(vals, smin=420, smax=2200):
    arr = np.asarray(vals, dtype=float)
    if arr.size == 0:
        return arr
    vmin = np.nanmin(arr)
    vmax = np.nanmax(arr)
    if vmax == vmin:
        return np.full_like(arr, (smin + smax) / 2.0)
    return smin + (arr - vmin) * (smax - smin) / (vmax - vmin)


def legend_marker_size(n, nmin, nmax, ms_min=10, ms_max=24):
    if nmax == nmin:
        return (ms_min + ms_max) / 2.0
    return ms_min + (n - nmin) * (ms_max - ms_min) / (nmax - nmin)


def make_nonoverlap_y_positions(yvals, ymin, ymax, min_gap=3.8):
    """
    Spread label anchor y-positions to avoid overlaps in a neat right-side column.
    """
    idx_sorted = np.argsort(yvals)[::-1]
    placed = np.zeros_like(yvals, dtype=float)

    current_top = ymax
    for idx in idx_sorted:
        y = yvals[idx]
        y_new = min(y, current_top)
        placed[idx] = y_new
        current_top = y_new - min_gap

    # if the bottom fell below ymin, shift everything upward/downward together
    min_placed = np.min(placed)
    if min_placed < ymin:
        placed += (ymin - min_placed)

    max_placed = np.max(placed)
    if max_placed > ymax:
        placed -= (max_placed - ymax)

    return placed


# =========================================================
# Read data
# =========================================================
master_rows = read_csv(MASTER)
highconf_set = sample_set_from_candidates(HIGHCONF_CANDIDATES)
priority_set = sample_set_from_candidates(PRIORITY_CANDIDATES)

species_groups = defaultdict(list)
for r in master_rows:
    sp = norm(r.get("TopSpecies1", ""))
    if sp:
        species_groups[sp].append(r)

dominant_species = [
    sp for sp, rr in sorted(species_groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    if len(rr) >= 4
]

species_stats = []
for sp in dominant_species:
    rr = species_groups[sp]
    samples = [norm(x.get("Sample", "")) for x in rr]
    n = len(rr)

    med_top = float(np.nanmedian([as_float(x.get("TopSpecies1Pct", np.nan)) for x in rr]))
    med_unc = float(np.nanmedian([as_float(x.get("UnclassifiedPct", np.nan)) for x in rr]))

    vals_top = [as_float(x.get("TopSpecies1Pct", np.nan)) for x in rr]
    vals_top = [v for v in vals_top if not np.isnan(v)]

    hc_n = sum(1 for s in samples if s in highconf_set)
    pr_n = sum(1 for s in samples if s in priority_set)
    other_n = max(n - hc_n - pr_n, 0)

    species_stats.append({
        "species": sp,
        "n": n,
        "hc_n": hc_n,
        "pr_n": pr_n,
        "other_n": other_n,
        "priority_frac": pr_n / n if n else 0.0,
        "highconf_frac": hc_n / n if n else 0.0,
        "med_top": med_top,
        "med_unc": med_unc,
        "top_vals": vals_top,
    })

# Use a stable order for Panels A-B:
species_stats = sorted(species_stats, key=lambda d: (-d["n"], d["species"]))

# =========================================================
# Color systems
# =========================================================
confirm_cmap = LinearSegmentedColormap.from_list(
    "confirm", ["#eff6ff", "#93c5fd", "#38bdf8", "#0f766e"]
)

species_palette = {
    "Serratia marcescens": "#2563eb",
    "Acinetobacter baumannii": "#0f766e",
    "Klebsiella pneumoniae": "#8b5cf6",
    "Pseudomonas aeruginosa": "#d946ef",
    "Escherichia coli": "#f59e0b",
}

# =========================================================
# Figure layout
# =========================================================
fig = plt.figure(figsize=(21.0, 16.8), facecolor="white")
gs = GridSpec(
    3, 2, figure=fig,
    width_ratios=[1.12, 0.88],
    height_ratios=[1.05, 1.0, 1.0],
    hspace=0.32, wspace=0.22
)

# =========================================================
# Panel A: species confirmation bubble landscape
# =========================================================
axA = fig.add_subplot(gs[0, 0])
style_axis(axA, "both")

# background guidance zones
axA.axvspan(80, 100, ymin=0.0, ymax=0.42, color="#ecfdf5", zorder=0)
axA.axvspan(0, 80, ymin=0.58, ymax=1.0, color="#fff1f2", zorder=0)
axA.axvline(80, color="#d1d5db", lw=1.0, ls="--", zorder=1)
axA.axhline(15, color="#d1d5db", lw=1.0, ls="--", zorder=1)

x = np.array([d["med_top"] for d in species_stats], dtype=float)
y = np.array([d["med_unc"] for d in species_stats], dtype=float)
nvals = np.array([d["n"] for d in species_stats], dtype=float)
size = map_size(nvals, 420, 2200)

# color by high-confidence fraction
color_vals = np.array([d["highconf_frac"] for d in species_stats], dtype=float)
cnorm = Normalize(vmin=0.0, vmax=1.0)

axA.scatter(
    x, y,
    s=size,
    c=color_vals,
    cmap=confirm_cmap,
    norm=cnorm,
    edgecolor="#334155",
    linewidth=1.15,
    alpha=0.96,
    zorder=3
)

# tighter y-limit to reduce wasted vertical space
ymax = max(18, np.nanmax(y) * 2.2 if len(y) else 18)
axA.set_xlim(0, 100)
axA.set_ylim(0, ymax)
axA.set_xticks(np.arange(0, 101, 20))
axA.set_xlabel("Median dominant-species assignment (%)")
axA.set_ylabel("Median unclassified fraction (%)")
axA.set_title("A. Species-confirmation landscape across dominant species", loc="left", pad=12)

# small zone labels, controlled and wrapped within panel boundary
axA.text(
    2.0, ymax - 0.9,
    "Lower support /\nhigher ambiguity",
    color="#9f1239",
    fontsize=10.1,
    fontweight="bold",
    ha="left",
    va="top",
    linespacing=1.12,
    bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#fee2e2", alpha=0.72),
    zorder=4
)

axA.text(
    98.0, 1.05,
    "Higher support /\nlower ambiguity",
    color="#166534",
    fontsize=10.1,
    fontweight="bold",
    ha="right",
    va="bottom",
    linespacing=1.12,
    bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#dcfce7", alpha=0.74),
    zorder=4
)

# short point labels only, with controlled offsets
offsets = [
    (-16, 14),   # S. marcescens
    (-6, 18),    # A. baumannii
    (8, 11),     # K. pneumoniae
    (10, -14),   # P. aeruginosa
    (-10, 12),   # E. coli
]

for d, (dx, dy) in zip(species_stats, offsets):
    axA.annotate(
        species_short(d["species"]),
        xy=(d["med_top"], d["med_unc"]),
        xytext=(dx, dy),
        textcoords="offset points",
        ha="left" if dx >= 0 else "right",
        va="bottom" if dy >= 0 else "top",
        fontsize=10.8,
        fontweight="bold",
        color="#0f172a",
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#e2e8f0", alpha=0.96),
        arrowprops=dict(arrowstyle="-", color="#94a3b8", lw=0.9),
        zorder=5
    )

# -------- species summary box: right-middle, fixed reference segment --------
summary_ax = axA.inset_axes([0.60, 0.40, 0.36, 0.31])
summary_ax.set_xlim(0, 1)
summary_ax.set_ylim(0, 1)
summary_ax.axis("off")

summary_ax.add_patch(
    plt.Rectangle(
        (0.0, 0.0), 1.0, 1.0,
        transform=summary_ax.transAxes,
        facecolor="white",
        edgecolor="#e2e8f0",
        linewidth=1.0,
        alpha=0.95,
        zorder=-1
    )
)

summary_ax.text(
    0.02, 0.98, "Species summary",
    ha="left", va="top",
    fontsize=11.0, fontweight="bold", color="#111827"
)

row_ys = np.linspace(0.78, 0.10, len(species_stats))
for yy, d in zip(row_ys, species_stats):
    summary_ax.scatter(
        [0.05], [yy],
        s=58,
        color=species_palette.get(d["species"], "#64748b"),
        edgecolor="#334155",
        linewidth=0.8,
        zorder=2
    )
    summary_ax.text(
        0.10, yy,
        f"{species_short(d['species'])}  |  n={d['n']}  |  HC {d['hc_n']}  |  PR {d['pr_n']}",
        ha="left", va="center",
        fontsize=10.2, fontweight="bold", color="#0f172a",
        zorder=2
    )

# -------- size legend: centre-left position, correctly scaled circles --------
unique_n = sorted(set(int(v) for v in nvals if not np.isnan(v)))

if len(unique_n) >= 4:
    legend_vals = [
        unique_n[0],
        unique_n[len(unique_n) // 3],
        unique_n[(2 * len(unique_n)) // 3],
        unique_n[-1],
    ]
elif len(unique_n) >= 2:
    legend_vals = unique_n
else:
    legend_vals = unique_n if unique_n else [1]

# remove accidental duplicates while preserving order
legend_vals = list(dict.fromkeys(legend_vals))

size_ax = axA.inset_axes([0.035, 0.405, 0.34, 0.20])
size_ax.set_xlim(0, 1)
size_ax.set_ylim(0, 1)
size_ax.axis("off")

size_ax.add_patch(
    plt.Rectangle(
        (0.0, 0.0), 1.0, 1.0,
        transform=size_ax.transAxes,
        facecolor="white",
        edgecolor="#e2e8f0",
        linewidth=1.0,
        alpha=0.93,
        zorder=-1
    )
)

size_ax.text(
    0.03, 0.94, "Species block size",
    ha="left", va="top",
    fontsize=10.9,
    fontweight="bold",
    color="#111827"
)

# scale legend circles using the same n-range logic as the main bubble field
nmin = float(np.nanmin(nvals)) if len(nvals) else 1.0
nmax = float(np.nanmax(nvals)) if len(nvals) else 1.0

legend_sizes = []
for v in legend_vals:
    if nmax == nmin:
        ss = 520
    else:
        ss = 180 + ((v - nmin) / (nmax - nmin)) * (1080 - 180)
    legend_sizes.append(ss)

xlocs = np.linspace(0.16, 0.86, len(legend_vals))
for xv, v, ss in zip(xlocs, legend_vals, legend_sizes):
    size_ax.scatter(
        [xv], [0.52],
        s=ss,
        color="#9ec5fe",
        edgecolor="#334155",
        linewidth=1.05,
        alpha=0.96,
        zorder=2
    )
    size_ax.text(
        xv, 0.08,
        f"n={v}",
        ha="center",
        va="bottom",
        fontsize=9.8,
        fontweight="bold",
        color="#334155",
        zorder=3
    )

# -------- high-confidence fraction colorbar: above species summary, top-aligned visually --------
# Species summary box = [0.60, 0.40, 0.36, 0.31]
# This keeps the colorbar directly above that box, while lifting it into the same
# upper visual band as the "Lower support / higher ambiguity" guidance label.
caxA = axA.inset_axes([0.60, 0.865, 0.36, 0.050])

cbarA = fig.colorbar(
    ScalarMappable(norm=cnorm, cmap=confirm_cmap),
    cax=caxA,
    orientation="horizontal"
)

cbarA.set_label(
    "High-confidence fraction",
    fontsize=10.2,
    labelpad=1.5
)
cbarA.ax.tick_params(labelsize=9.1, pad=1)
cbarA.outline.set_visible(False)

# Cleaner colorbar frame for premium appearance
for spine in cbarA.ax.spines.values():
    spine.set_visible(False)

# =========================================================
# Panel B: within-species support distributions
# =========================================================
axB = fig.add_subplot(gs[0, 1])
style_axis(axB, "x")

rng = np.random.default_rng(42)

plot_species = [d["species"] for d in species_stats]
positions = np.arange(len(plot_species), 0, -1)
violin_data = [next(d["top_vals"] for d in species_stats if d["species"] == sp) for sp in plot_species]

vp = axB.violinplot(
    violin_data,
    positions=positions,
    vert=False,
    widths=0.78,
    showmeans=False,
    showmedians=False,
    showextrema=False
)

for body, sp in zip(vp["bodies"], plot_species):
    color = species_palette.get(sp, "#64748b")
    body.set_facecolor(color)
    body.set_edgecolor(color)
    body.set_alpha(0.18)
    body.set_linewidth(1.1)

for pos, sp in zip(positions, plot_species):
    rr = species_groups[sp]
    xs = [as_float(r.get("TopSpecies1Pct", np.nan)) for r in rr]
    xs = [v for v in xs if not np.isnan(v)]
    if not xs:
        continue

    ys = rng.normal(pos, 0.042, size=len(xs))

    point_colors = []
    for r in rr:
        sample = norm(r.get("Sample", ""))
        tp = as_float(r.get("TopSpecies1Pct", np.nan))
        if np.isnan(tp):
            continue
        if sample in priority_set:
            point_colors.append("#ef4444")
        elif sample in highconf_set:
            point_colors.append(species_palette.get(sp, "#64748b"))
        else:
            point_colors.append("#cbd5e1")

    if len(point_colors) != len(xs):
        point_colors = [species_palette.get(sp, "#64748b")] * len(xs)

    axB.scatter(
        xs, ys,
        s=24,
        color=point_colors,
        edgecolor="white",
        linewidth=0.45,
        alpha=0.92,
        zorder=3
    )

    q1, med, q3 = np.percentile(xs, [25, 50, 75])
    axB.plot([q1, q3], [pos, pos], color="#334155", lw=2.0, zorder=4)
    axB.scatter([med], [pos], marker="D", s=46, color="#111827", edgecolor="white", linewidth=0.55, zorder=5)

axB.set_xlim(0, 100)
axB.set_yticks(positions)
axB.set_yticklabels([
    f"{species_short(sp)} (n={len(species_groups[sp])})"
    for sp in plot_species
])
axB.set_xlabel("Per-sample dominant-species assignment (%)")
axB.set_title("B. Within-species assignment support distributions", loc="left", pad=12)

# subtle right-side summary text, kept outside crowding zone
for pos, sp in zip(positions, plot_species):
    stat_d = next(d for d in species_stats if d["species"] == sp)
    axB.text(
        100.8, pos,
        f"HC {stat_d['hc_n']} | PR {stat_d['pr_n']}",
        ha="left", va="center",
        fontsize=10.5, fontweight="bold", color="#334155"
    )

axB.legend(
    handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#ef4444",
               markeredgecolor="white", markersize=8, label="Priority-review"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#64748b",
               markeredgecolor="white", markersize=8, label="High-confidence"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#cbd5e1",
               markeredgecolor="white", markersize=8, label="Other"),
    ],
    frameon=False,
    loc="lower left",
    bbox_to_anchor=(0.0, 0.01),
    ncol=3,
    columnspacing=1.3,
    handletextpad=0.45
)

# =========================================================
# Panels C-F: relatedness matrices
# =========================================================
related_species_order = [
    "Acinetobacter baumannii",
    "Serratia marcescens",
    "Klebsiella pneumoniae",
    "Pseudomonas aeruginosa",
]
letters = ["C", "D", "E", "F"]
axes = [
    fig.add_subplot(gs[1, 0]),
    fig.add_subplot(gs[1, 1]),
    fig.add_subplot(gs[2, 0]),
    fig.add_subplot(gs[2, 1]),
]

for ax, sp, letter in zip(axes, related_species_order, letters):
    add_soft_panel_bg(ax, face="#fcfdff")
    img_path = first_existing(TREE_IMAGE_CANDIDATES.get(sp, []))
    n_sp = len(species_groups.get(sp, []))

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"{letter}. {species_short(sp)} higher-resolution relatedness (n={n_sp})",
        loc="left", pad=10
    )

    if img_path and os.path.isfile(img_path):
        try:
            img = plt.imread(img_path)
            img = trim_white(img)
            ax.imshow(img, aspect="auto")
        except Exception as e:
            ax.text(
                0.5, 0.5,
                f"Could not load matrix image\n{os.path.basename(img_path)}\n{e}",
                ha="center", va="center",
                fontsize=12, fontweight="bold",
                transform=ax.transAxes
            )
    else:
        ax.text(
            0.5, 0.5,
            "Relatedness matrix image not found",
            ha="center", va="center",
            fontsize=12.5, fontweight="bold",
            transform=ax.transAxes
        )

    for side in ["left", "right", "top", "bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("#cbd5e1")
        ax.spines[side].set_linewidth(1.25)

# =========================================================
# Title / layout
# =========================================================
fig.suptitle(
    "Species confirmation and higher-resolution relatedness architecture",
    y=0.988,
    fontsize=26.8,
    fontweight="bold"
)

# no footer: keeps the figure cleaner and avoids crowding
fig.subplots_adjust(left=0.055, right=0.965, top=0.935, bottom=0.055)
write_png_pdf(fig, OUTDIR, OUTNAME)
print("Saved Figure 3 to:", OUTDIR)