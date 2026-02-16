import numpy as np
import matplotlib.pyplot as plt
import os

from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import FuncFormatter, MultipleLocator
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patheffects as pe


# -----------------------------
# 1) INPUT DATA (update as required)
# -----------------------------
labels = ["Cells only", "Vector only", "F1R1", "F2R1", "F3R1", "F4R1", "F3R2"]

means = np.array([1_000, 10_000, 2_000, 4_500, 6_000, 16_500, 21_500], dtype=float)
errs  = np.array([1_200, 1_200, 1_200, 1_200, 1_200, 1_200, 1_200], dtype=float)

# Per-comparison text offsets to prevent annotation overlap (dx expressed as a fraction of the x-range).
p_text_offsets = {
    ("F1R1", "F4R1"): {"text_dx_frac": 0.04, "text_dy": 0.0},  # push right a bit
}

# Per-comparison bracket offsets (applies to bracket geometry, not only label position).
p_bracket_offsets = {
    ("F1R1", "F4R1"): {"x_end_mult": 1.12},   # make this bracket more to the right
}

comparisons = [
    ("F1R1", "Vector only", "P = 0.001"),
    ("F1R1", "F2R1", "P = 0.189"),
    ("F2R1", "F3R1", "P = 0.045"),
    ("F4R1", "F3R2", "P = 0.001"),
    ("F1R1", "F4R1", "P = 0.001"),
]

# -----------------------------
# 1B) CONSTRUCT SCHEMATIC PARAMETERS (configured to reproduce the reference layout)
# -----------------------------
# Normalized schematic coordinates (0–1). Adjust only if layout refinement is required.
construct_layout = {
    "Cells only":  {"show_line": True, "show_luc": False, "x0": 0.10, "x1": 0.95, "luc_x": 0.90},

    # Place the LUC cassette toward the right to match the reference schematic.
    "Vector only": {"show_line": True, "show_luc": True,  "x0": 0.10, "x1": 0.95, "luc_x": 0.80},
    "F1R1":        {"show_line": True, "show_luc": True,  "x0": 0.10, "x1": 0.95, "luc_x": 0.80},
    "F2R1":        {"show_line": True, "show_luc": True,  "x0": 0.10, "x1": 0.95, "luc_x": 0.76},

    # Progressively shift the LUC cassette leftward to represent shorter fragments.
    "F3R1":        {"show_line": True, "show_luc": True,  "x0": 0.10, "x1": 0.95, "luc_x": 0.66},
    "F4R1":        {"show_line": True, "show_luc": True,  "x0": 0.10, "x1": 0.95, "luc_x": 0.57},

    # Configure the lower construct to start later, with the LUC cassette positioned mid-right.
    "F3R2":        {"show_line": True, "show_luc": True,  "x0": 0.28, "x1": 0.95, "luc_x": 0.66},
}

# -----------------------------
# 2) STYLE (journal/premium)
# -----------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.linewidth": 1.2,
})

sky_blue  = "#8BA1AC"
dark_blue = "#134058"
cmap = LinearSegmentedColormap.from_list("premium_blue", [sky_blue, dark_blue])

cells_grey = "#D9DDE3"
ink = "#0B1F3B"


def comma_formatter(x, pos):
    return f"{int(x):,}"


def add_detached_bracket(ax, y1, y2, x_start, x_end, text,
                         text_pad_frac=0.02, text_dx_frac=0.0, text_dy=0.0, lw=1.2):
    ax.plot([x_start, x_end], [y1, y1], color=ink, lw=lw, zorder=5)
    ax.plot([x_start, x_end], [y2, y2], color=ink, lw=lw, zorder=5)
    ax.plot([x_end, x_end], [y1, y2], color=ink, lw=lw, zorder=5)

    x_text = x_end + (text_pad_frac + text_dx_frac) * ax.get_xlim()[1]
    y_text = (y1 + y2) / 2 + text_dy
    ax.text(x_text, y_text, text, va="center", ha="left",
            fontsize=12, fontweight="bold", color=ink, zorder=6)


def draw_construct_row(ax_s, y, cfg):
    """Journal-style schematic with flat ends, rectangular caps, and a rounded LUC cassette."""
    if not cfg.get("show_line", True):
        return

    x0, x1 = cfg["x0"], cfg["x1"]

    # Backbone line (formal)
    ax_s.plot([x0, x1], [y, y], color=ink, lw=7, solid_capstyle="butt", zorder=2)

    # Rectangular end caps (formal)
    cap_h = 0.28
    cap_w = 0.018
    ax_s.add_patch(Rectangle((x0 - cap_w/2, y - cap_h/2), cap_w, cap_h,
                             facecolor=ink, edgecolor=ink, linewidth=0, zorder=3))
    ax_s.add_patch(Rectangle((x1 - cap_w/2, y - cap_h/2), cap_w, cap_h,
                             facecolor=ink, edgecolor=ink, linewidth=0, zorder=3))

    if not cfg.get("show_luc", True):
        return

    luc_x = cfg.get("luc_x", (x0 + x1) / 2)

    # LUC cassette (premium)
    luc_w = 0.22
    luc_h = 0.42
    box = FancyBboxPatch(
        (luc_x - luc_w/2, y - luc_h/2),
        luc_w, luc_h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        facecolor=ink, edgecolor=ink, linewidth=1.0, zorder=5
    )
    box.set_path_effects([
        pe.SimplePatchShadow(offset=(1.2, -1.2), alpha=0.18),
        pe.Normal()
    ])
    ax_s.add_patch(box)

    ax_s.text(luc_x, y, "LUC", color="white",
              ha="center", va="center",
              fontsize=11, fontweight="bold",
              zorder=6,
              path_effects=[pe.withStroke(linewidth=1.2, foreground=ink)])


# -----------------------------
# 3) BUILD PLOT (2-panel layout)
# -----------------------------
fig = plt.figure(figsize=(10.2, 5.2), dpi=160)
gs = GridSpec(1, 2, width_ratios=[1.9, 5.0], wspace=0.03)

ax_s = fig.add_subplot(gs[0, 0])
ax   = fig.add_subplot(gs[0, 1], sharey=ax_s)

labels_r = labels[::-1]
means_r  = means[::-1]
errs_r   = errs[::-1]
y = np.arange(len(labels_r))

# -----------------------------
# 3A) Left panel: construct schematic with manually positioned y-axis labels
# -----------------------------
ax_s.set_xlim(0, 1)
ax_s.set_ylim(-0.5, len(labels_r) - 0.5)
ax_s.set_xticks([])
ax_s.set_yticks(y)
ax_s.set_yticklabels([])
ax_s.tick_params(axis="y", left=False, right=False)

for spine in ["top", "right", "bottom", "left"]:
    ax_s.spines[spine].set_visible(False)

# Render y-axis labels manually to ensure consistent visibility and placement.
for yi, lab in zip(y, labels_r):
    ax_s.text(
        -0.08, yi, lab,
        transform=ax_s.get_yaxis_transform(),
        ha="right", va="center",
        fontsize=14, fontweight="bold", color=ink,
        clip_on=False
    )

# Draw construct schematics for each condition.
for yi, lab in zip(y, labels_r):
    cfg = construct_layout.get(lab, {"show_line": True, "show_luc": True, "x0": 0.10, "x1": 0.95, "luc_x": 0.80})
    draw_construct_row(ax_s, yi, cfg)

# -----------------------------
# 3B) Right panel: bars + errors
# -----------------------------
non_cells_mask = np.array([lab != "Cells only" for lab in labels_r])
norm = Normalize(vmin=means_r[non_cells_mask].min(), vmax=means_r[non_cells_mask].max())

colors = []
for lab, m in zip(labels_r, means_r):
    colors.append(cells_grey if lab == "Cells only" else cmap(norm(m)))

ax.barh(y, means_r, height=0.56, color=colors, edgecolor="none", zorder=2)
ax.errorbar(means_r, y, xerr=errs_r, fmt="none",
            ecolor=ink, elinewidth=1.2, capsize=6, capthick=1.2, zorder=3)

ax.set_yticks(y)
ax.set_yticklabels([])
ax.tick_params(axis="y", length=0)

ax.set_xlabel("Relative Luciferase Unit (RLU)", fontsize=16, fontweight="bold",
              color=ink, labelpad=14)

# Use tick locators appropriate to the data scale (small-range axis).
ax.xaxis.set_major_formatter(FuncFormatter(comma_formatter))
ax.xaxis.set_major_locator(MultipleLocator(5_000))   # <--- key fix
ax.xaxis.set_minor_locator(MultipleLocator(2_500))

ax.tick_params(axis="x", labelsize=12, width=1.2, length=8, colors=ink)
ax.tick_params(axis="x", which="minor", length=4, width=1.0, colors=ink)

# Rotate x-axis tick labels to improve readability.
plt.setp(ax.get_xticklabels(), rotation=90, ha="center", va="top")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color(ink)
ax.spines["bottom"].set_color(ink)

# Extend x-axis limits to accommodate statistical brackets and annotation text.
xmax_data = (means_r + errs_r).max()
short_x = xmax_data * 1.18
long_x  = xmax_data * 1.45
# Reserve additional right-side margin for extended brackets (if specified).
max_x_end = long_x
for (a, b), o in p_bracket_offsets.items():
    max_x_end = max(max_x_end, long_x * o.get("x_end_mult", 1))

ax.set_xlim(0, max_x_end)

# -----------------------------
# 4) STATISTICAL ANNOTATIONS (aligned bracket start positions)
# -----------------------------
label_to_y   = {lab: yi for lab, yi in zip(labels_r, y)}
label_to_end = {lab: (m + e) for lab, m, e in zip(labels_r, means_r, errs_r)}

gap_abs  = xmax_data * 0.08
min_span = xmax_data * 0.06

involved_labels = set([a for a, _, _ in comparisons] + [b for _, b, _ in comparisons])
common_start = max(label_to_end[l] for l in involved_labels) + gap_abs

for top, bottom, text in comparisons:
    if top not in label_to_y or bottom not in label_to_y:
        continue

    y1 = label_to_y[top]
    y2 = label_to_y[bottom]
    y_low, y_high = (min(y1, y2), max(y1, y2))

    x_end = long_x if abs(y_high - y_low) >= 2 else short_x
    # Apply optional per-comparison horizontal offset to the selected bracket.
    k2 = (top, bottom) if (top, bottom) in p_bracket_offsets else (bottom, top)
    x_end *= p_bracket_offsets.get(k2, {}).get("x_end_mult", 1.0)

    x_start = common_start
    if x_end < x_start + min_span:
        x_end = x_start + min_span

    key = (top, bottom) if (top, bottom) in p_text_offsets else (bottom, top)
    opts = p_text_offsets.get(key, {})
    add_detached_bracket(
        ax, y_low, y_high,
        x_start=x_start, x_end=x_end, text=text,
        **opts
    )


# Adjust margins to prevent clipping of labels and rotated tick marks.
fig.subplots_adjust(left=0.32, bottom=0.18)
fig.canvas.draw()

# -----------------------------
# 5) EXPORT
# -----------------------------
out_dir = r"C:\Users\User\OneDrive\2. Personal\AAM\Desktop\Luciferase Unit\final figures"
os.makedirs(out_dir, exist_ok=True)

out_base = os.path.join(out_dir, "IL-33_5-UTR")

fig.savefig(out_base + ".png", dpi=600, bbox_inches="tight")
fig.savefig(out_base + ".pdf", bbox_inches="tight")
fig.savefig(out_base + ".svg", bbox_inches="tight")

print("Saved files to:")
print(out_dir)
print("PNG:", out_base + ".png")
print("PDF:", out_base + ".pdf")
print("SVG:", out_base + ".svg")

plt.show()