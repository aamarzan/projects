import numpy as np
import matplotlib.pyplot as plt
import os

from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import FuncFormatter, MultipleLocator
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patheffects as pe


# -----------------------------
# 1) DATA
# -----------------------------
labels = ["Cells only", "Vector only", "F1R1", "F1R2", "F2R2"]

means = np.array([50_000, 2_500_000, 1_350_000, 2_600_000, 1_600_000], dtype=float)
errs  = np.array([120_000, 120_000, 120_000, 120_000, 120_000], dtype=float)

comparisons = [
    ("F1R1", "Vector only", "P < 0.0001"),
    ("F1R2", "F2R2", "P = 0.331"),
    ("F1R1", "F1R2", "P < 0.0001"),
]

# -----------------------------
# 1B) CONSTRUCT SCHEMATIC SETTINGS 
# -----------------------------
# Parameters for schematic rendering only (normalized coordinates: 0–1).
construct_layout = {
    "Cells only":  {"show_line": True, "show_luc": False, "x0": 0.10, "x1": 0.95, "luc_x": 0.70},
    "Vector only": {"show_line": True, "show_luc": True,  "x0": 0.10, "x1": 0.95, "luc_x": 0.60},
    "F1R1":        {"show_line": True, "show_luc": True,  "x0": 0.10, "x1": 0.95, "luc_x": 0.62},
    "F1R2":        {"show_line": True, "show_luc": True,  "x0": 0.25, "x1": 0.95, "luc_x": 0.60},
    "F2R2":        {"show_line": True, "show_luc": True,  "x0": 0.25, "x1": 0.95, "luc_x": 0.42},
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


def add_detached_bracket(ax, y1, y2, x_start, x_end, text, text_pad_frac=0.02, lw=1.2):
    ax.plot([x_start, x_end], [y1, y1], color=ink, lw=lw, zorder=5)
    ax.plot([x_start, x_end], [y2, y2], color=ink, lw=lw, zorder=5)
    ax.plot([x_end, x_end], [y1, y2], color=ink, lw=lw, zorder=5)

    x_text = x_end + text_pad_frac * ax.get_xlim()[1]
    y_text = (y1 + y2) / 2
    ax.text(x_text, y_text, text, va="center", ha="left",
            fontsize=12, fontweight="bold", color=ink, zorder=6)


def draw_construct_row(ax_s, y, cfg):
    """
    Premium schematic: thick rounded backbone, clean end-caps,
    rounded LUC cassette with subtle shadow + crisp text.
    """
    if not cfg.get("show_line", True):
        return

    x0, x1 = cfg["x0"], cfg["x1"]

    # Backbone line (rounded ends)
    ax_s.plot([x0, x1], [y, y],
          color=ink, lw=7,
          solid_capstyle="butt", zorder=2)


    # End caps
    cap_h = 0.28     # height of the cap
    cap_w = 0.018    # thickness of the cap (in 0..1 schematic units)

    left_cap = Rectangle((x0 - cap_w/2, y - cap_h/2), cap_w, cap_h,
                        facecolor=ink, edgecolor=ink, linewidth=0, zorder=3)
    right_cap = Rectangle((x1 - cap_w/2, y - cap_h/2), cap_w, cap_h,
                        facecolor=ink, edgecolor=ink, linewidth=0, zorder=3)

    ax_s.add_patch(left_cap)
    ax_s.add_patch(right_cap)


    if not cfg.get("show_luc", True):
        return

    luc_x = cfg.get("luc_x", (x0 + x1) / 2)

    # Cassette size
    luc_w = 0.22
    luc_h = 0.42

    box = FancyBboxPatch(
        (luc_x - luc_w/2, y - luc_h/2),
        luc_w, luc_h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        facecolor=ink,
        edgecolor=ink,
        linewidth=1.0,
        zorder=5
    )

    # Apply a subtle shadow to improve visual separation while maintaining journal-appropriate styling.
    box.set_path_effects([
        pe.SimplePatchShadow(offset=(1.2, -1.2), alpha=0.20),
        pe.Normal()
    ])
    ax_s.add_patch(box)

    ax_s.text(luc_x, y, "LUC",
              color="white",
              ha="center", va="center",
              fontsize=11, fontweight="bold",
              zorder=6,
              path_effects=[pe.withStroke(linewidth=1.2, foreground=ink)])


# -----------------------------
# 3) BUILD PLOT (2-panel layout)
# -----------------------------
fig = plt.figure(figsize=(10.2, 4.8), dpi=160)
gs = GridSpec(1, 2, width_ratios=[1.8, 5.0], wspace=0.03)

ax_s = fig.add_subplot(gs[0, 0])              # schematic panel (left)
ax   = fig.add_subplot(gs[0, 1], sharey=ax_s) # bar plot panel (right)

# Reverse order so the first condition appears at the top of the panel.
labels_r = labels[::-1]
means_r  = means[::-1]
errs_r   = errs[::-1]
y = np.arange(len(labels_r))

# -----------------------------
# 3A) Left panel: schematic + custom y labels (bulletproof)
# -----------------------------
ax_s.set_xlim(0, 1)
ax_s.set_ylim(-0.5, len(labels_r) - 0.5)
ax_s.set_xticks([])

# Hide default tick labels (custom labels are drawn manually for layout stability).
ax_s.set_yticks(y)
ax_s.set_yticklabels([])
ax_s.tick_params(axis="y", left=False, right=False)

# Remove spines for a clean schematic panel.
for spine in ["top", "right", "bottom", "left"]:
    ax_s.spines[spine].set_visible(False)

# Draw y-axis labels manually to ensure consistent placement and prevent clipping during export.
for yi, lab in zip(y, labels_r):
    ax_s.text(
        -0.08, yi, lab,  # x is slightly left of the axis
        transform=ax_s.get_yaxis_transform(),  # x in axes coords, y in data coords
        ha="right", va="center",
        fontsize=14, fontweight="bold", color=ink,
        clip_on=False  # Allow text rendering outside the axes region to avoid clipping.
    )

# Draw construct schematics
for yi, lab in zip(y, labels_r):
    cfg = construct_layout.get(
        lab,
        {"show_line": True, "show_luc": True, "x0": 0.10, "x1": 0.95, "luc_x": 0.60}
    )
    draw_construct_row(ax_s, yi, cfg)

# -----------------------------
# 3B) Right panel: bars + errors
# -----------------------------
# Color mapping: "Cells only" uses a neutral grey; other conditions are scaled by RLU magnitude.
non_cells_mask = np.array([lab != "Cells only" for lab in labels_r])
if non_cells_mask.any():
    norm = Normalize(vmin=means_r[non_cells_mask].min(), vmax=means_r[non_cells_mask].max())
else:
    norm = Normalize(vmin=means_r.min(), vmax=means_r.max())

colors = []
for lab, m in zip(labels_r, means_r):
    colors.append(cells_grey if lab == "Cells only" else cmap(norm(m)))

# Bars
bar_h = 0.56
ax.barh(y, means_r, height=bar_h, color=colors, edgecolor="none", zorder=2)

# Error bars
ax.errorbar(means_r, y, xerr=errs_r, fmt="none",
            ecolor=ink, elinewidth=1.2, capsize=6, capthick=1.2, zorder=3)

# Remove duplicate y labels on right axis
ax.set_yticks(y)
ax.set_yticklabels([])
ax.tick_params(axis="y", length=0)

# X axis formatting
ax.set_xlabel("Relative Luciferase Unit (RLU)", fontsize=16, fontweight="bold",
              color=ink, labelpad=18)

ax.xaxis.set_major_formatter(FuncFormatter(comma_formatter))
ax.xaxis.set_major_locator(MultipleLocator(500_000))
ax.tick_params(axis="x", labelsize=12, width=1.2, length=8, colors=ink)

# Rotate x-axis tick labels vertically to prevent overlap.
plt.setp(ax.get_xticklabels(), rotation=90, ha="center", va="top")

# Spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color(ink)
ax.spines["bottom"].set_color(ink)

# Set x-axis limits to reserve space for statistical brackets and annotations.
xmax = (means_r + errs_r).max()
ax.set_xlim(0, xmax * 1.25)

# -----------------------------
# 4) P-VALUE BRACKETS (detached + ALIGNED START)
# -----------------------------
label_to_y   = {lab: yi for lab, yi in zip(labels_r, y)}
label_to_end = {lab: (m + e) for lab, m, e in zip(labels_r, means_r, errs_r)}

short_x = xmax * 1.18
long_x  = xmax * 1.45

gap_abs  = xmax * 0.08
min_span = xmax * 0.06

involved_labels = set([a for a, _, _ in comparisons] + [b for _, b, _ in comparisons])
common_start = max(label_to_end[l] for l in involved_labels) + gap_abs

for top, bottom, text in comparisons:
    if top not in label_to_y or bottom not in label_to_y:
        continue

    y1 = label_to_y[top]
    y2 = label_to_y[bottom]
    y_low, y_high = (min(y1, y2), max(y1, y2))

    x_end = long_x if abs(y_high - y_low) >= 2 else short_x
    x_start = common_start

    if x_end < x_start + min_span:
        x_end = x_start + min_span

    add_detached_bracket(ax, y_low, y_high, x_start=x_start, x_end=x_end, text=text)

# Increase margins to prevent label clipping during export.
fig.subplots_adjust(left=0.32, bottom=0.40)  # more space for rotated x tick labels

# Render the canvas before saving so tight bounding boxes include all artists.
fig.canvas.draw()

# -------------------------------------
# 5) EXPORT (publication-ready formats)
# -------------------------------------
out_dir = r"C:\Users\User\OneDrive\2. Personal\AAM\Desktop\Luciferase Unit\final figures"
os.makedirs(out_dir, exist_ok=True)

out_base = os.path.join(out_dir, "IL-33_3-UTR")

fig.savefig(out_base + ".png", dpi=600, bbox_inches="tight")
fig.savefig(out_base + ".pdf", bbox_inches="tight")
fig.savefig(out_base + ".svg", bbox_inches="tight")

print("Saved files to:")
print(out_dir)
print("PNG:", out_base + ".png")
print("PDF:", out_base + ".pdf")
print("SVG:", out_base + ".svg")

plt.show()
print("Current working directory:", os.getcwd())
