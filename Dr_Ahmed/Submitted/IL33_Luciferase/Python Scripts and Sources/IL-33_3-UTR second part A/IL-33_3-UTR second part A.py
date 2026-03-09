import numpy as np
import matplotlib.pyplot as plt
import os

from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import FuncFormatter, MultipleLocator
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patheffects as pe


# -----------------------------
# 1) DATA (edit this section)
# -----------------------------
labels = ["Cells only", "Vector only", "F1R3", "F2R3", "F3R3", "F4R1"]

# Replace with your real mean RLU values (example placeholders)
means = np.array([80_000, 3_500_000, 800_000, 250_000, 200_000, 200_000], dtype=float)

# Replace with your real SEM/SD (example placeholders)
errs  = np.array([40_000, 150_000, 120_000, 60_000, 50_000, 45_000], dtype=float)


# -----------------------------
# 1B) CONSTRUCT SCHEMATIC SETTINGS (match your reference image)
# -----------------------------
# Schematic-only coordinates (0..1). Tune luc_x/x0 visually if needed.
construct_layout = {
    # Cells only: no LUC, but a small terminal box near the right (like your image)
    "Cells only":  {"show_line": True, "show_luc": False, "x0": 0.10, "x1": 0.92, "end_box": False},

    # Vector only + F1R3: LUC close to the right end
    "Vector only": {"show_line": True, "show_luc": True,  "x0": 0.10, "x1": 0.92, "luc_x": 0.82},
    "F1R3":        {"show_line": True, "show_luc": True,  "x0": 0.28, "x1": 0.92, "luc_x": 0.82},

    # F2R3, F3R3: LUC progressively more left (as in the reference)
    "F2R3":        {"show_line": True, "show_luc": True,  "x0": 0.28, "x1": 0.92, "luc_x": 0.70},
    "F3R3":        {"show_line": True, "show_luc": True,  "x0": 0.28, "x1": 0.92, "luc_x": 0.62},

    # F4R1: LUC further left
    "F4R1":        {"show_line": True, "show_luc": True,  "x0": 0.10, "x1": 0.92, "luc_x": 0.54},
}


# -----------------------------
# 2) STYLE (journal/premium)
# -----------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",   # change to Arial if installed
    "font.size": 12,
    "axes.linewidth": 1.2,
})

# Premium blue gradient (you can change hex codes if you want)
sky_blue  = "#8BA1AC"
dark_blue = "#134058"
cmap = LinearSegmentedColormap.from_list("premium_blue", [sky_blue, dark_blue])

cells_grey = "#D9DDE3"
ink = "#0B1F3B"


def comma_formatter(x, pos):
    return f"{int(x):,}"


def draw_construct_row(ax_s, y, cfg):
    """Formal construct: flat backbone + rectangular end caps + premium LUC cassette."""
    if not cfg.get("show_line", True):
        return

    x0, x1 = cfg["x0"], cfg["x1"]

    # Backbone line (formal)
    ax_s.plot([x0, x1], [y, y], color=ink, lw=7, solid_capstyle="butt", zorder=2)

    # Rectangular end caps
    cap_h = 0.28
    cap_w = 0.018
    ax_s.add_patch(Rectangle((x0 - cap_w/2, y - cap_h/2), cap_w, cap_h,
                             facecolor=ink, edgecolor=ink, linewidth=0, zorder=3))
    ax_s.add_patch(Rectangle((x1 - cap_w/2, y - cap_h/2), cap_w, cap_h,
                             facecolor=ink, edgecolor=ink, linewidth=0, zorder=3))

    # Optional terminal box for "Cells only" (matches your reference look)
    if cfg.get("end_box", False):
        box_w = 0.16
        box_h = 0.46
        end_box = Rectangle((x1 - box_w, y - box_h/2), box_w, box_h,
                            facecolor=ink, edgecolor=ink, linewidth=0, zorder=4)
        ax_s.add_patch(end_box)
        return

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
# 3) BUILD FIGURE (2-panel layout)
# -----------------------------
fig = plt.figure(figsize=(10.8, 5.4), dpi=160)
gs = GridSpec(1, 2, width_ratios=[2.1, 5.2], wspace=0.03)

ax_s = fig.add_subplot(gs[0, 0])              # schematic
ax   = fig.add_subplot(gs[0, 1], sharey=ax_s) # bars

# Reverse so first label appears at top
labels_r = labels[::-1]
means_r  = means[::-1]
errs_r   = errs[::-1]
y = np.arange(len(labels_r))

# Panel label "A."
fig.text(0.015, 0.97, "A.", ha="left", va="top", fontsize=18, fontweight="bold", color=ink)

# -----------------------------
# 3A) Left panel: schematic + manual y labels
# -----------------------------
ax_s.set_xlim(0, 1)
ax_s.set_ylim(-0.5, len(labels_r) - 0.5)
ax_s.set_xticks([])
ax_s.set_yticks(y)
ax_s.set_yticklabels([])
ax_s.tick_params(axis="y", left=False, right=False)

for spine in ["top", "right", "bottom", "left"]:
    ax_s.spines[spine].set_visible(False)

# Manual y labels (never disappear)
for yi, lab in zip(y, labels_r):
    ax_s.text(
        -0.08, yi, lab,
        transform=ax_s.get_yaxis_transform(),
        ha="right", va="center",
        fontsize=14, fontweight="bold", color=ink,
        clip_on=False
    )

# Draw constructs
for yi, lab in zip(y, labels_r):
    cfg = construct_layout.get(
        lab,
        {"show_line": True, "show_luc": True, "x0": 0.10, "x1": 0.92, "luc_x": 0.75}
    )
    draw_construct_row(ax_s, yi, cfg)

# -----------------------------
# 3B) Right panel: bars + errors
# -----------------------------
# Colors: Cells only grey; others mapped by RLU
non_cells_mask = np.array([lab != "Cells only" for lab in labels_r])
if non_cells_mask.any():
    norm = Normalize(vmin=means_r[non_cells_mask].min(), vmax=means_r[non_cells_mask].max())
else:
    norm = Normalize(vmin=means_r.min(), vmax=means_r.max())

colors = []
for lab, m in zip(labels_r, means_r):
    colors.append(cells_grey if lab == "Cells only" else cmap(norm(m)))

ax.barh(y, means_r, height=0.56, color=colors, edgecolor="none", zorder=2)
ax.errorbar(means_r, y, xerr=errs_r, fmt="none",
            ecolor=ink, elinewidth=1.2, capsize=6, capthick=1.2, zorder=3)

# Remove duplicate y labels on right axis
ax.set_yticks(y)
ax.set_yticklabels([])
ax.tick_params(axis="y", length=0)

# X axis formatting (match the reference scale)
ax.set_xlabel("Relative Luciferase Unit (RLU)", fontsize=16, fontweight="bold",
              color=ink, labelpad=14)

ax.xaxis.set_major_formatter(FuncFormatter(comma_formatter))
ax.xaxis.set_major_locator(MultipleLocator(500_000))
ax.xaxis.set_minor_locator(MultipleLocator(250_000))

ax.tick_params(axis="x", labelsize=12, width=1.2, length=8, colors=ink)
ax.tick_params(axis="x", which="minor", length=4, width=1.0, colors=ink)

# Vertical tick labels like the original image
plt.setp(ax.get_xticklabels(), rotation=90, ha="center", va="top")

# Spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color(ink)
ax.spines["bottom"].set_color(ink)

# Hard-set axis max like the reference (0..4,000,000)
ax.set_xlim(0, 4_000_000)

# Layout: extra bottom space for vertical tick labels
fig.subplots_adjust(left=0.30, bottom=0.26)
fig.canvas.draw()

# -----------------------------
# 5) EXPORT
# -----------------------------
out_dir = r"C:\Users\User\OneDrive\2. Personal\AAM\Desktop\Luciferase Unit\final figures"
os.makedirs(out_dir, exist_ok=True)

out_base = os.path.join(out_dir, "IL-33_3-UTR second")

fig.savefig(out_base + ".png", dpi=600, bbox_inches="tight")
fig.savefig(out_base + ".pdf", bbox_inches="tight")
fig.savefig(out_base + ".svg", bbox_inches="tight")

print("Saved files to:")
print(out_dir)
print("PNG:", out_base + ".png")
print("PDF:", out_base + ".pdf")
print("SVG:", out_base + ".svg")

plt.show()
