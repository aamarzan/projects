import numpy as np
import matplotlib.pyplot as plt
import os

from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import FuncFormatter, MultipleLocator
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patheffects as pe


# -----------------------------
# 1) DATA (EDIT THIS SECTION)
# -----------------------------
# Top-to-bottom order (like the figure):
labels = ["C-A", "T-A", "T-G"]

# Replace with your real mean RLU values
means = np.array([260_000, 180_000, 2_250_000], dtype=float)

# Replace with your real SEM/SD values
errs  = np.array([220_000, 215_000, 220_000], dtype=float)

# Alleles shown at each SNP per haplotype row
# (SNP1 allele, SNP2 allele)
allele_map = {
    "C-A": ("C", "A"),
    "T-A": ("T", "A"),
    "T-G": ("T", "G"),
}

# P-value comparisons (as in the reference image)
comparisons = [
    ("C-A", "T-A", "P = 0.037"),
    ("T-A", "T-G", "P = 0.001"),
    ("C-A", "T-G", "P = 0.002"),
]

# Optional: exact bracket endpoints to mimic nesting (in data units)
# You can tweak these if you want the brackets to look identical.
bracket_end_abs = {
    ("C-A", "T-A"): 2_150_000,
    ("T-A", "T-G"): 2_350_000,
    ("C-A", "T-G"): 3_500_000,
}

# -----------------------------
# 1B) SNP COLUMN SETTINGS (schematic)
# -----------------------------
snp_center = 0.42
snp_gap = 0.36          # increase this to separate more (0.30–0.45)
snp1 = {"label": "rs55726619 (C/T)", "x": snp_center - snp_gap/2}
snp2 = {"label": "rs1048274 (A/G)", "x": snp_center + snp_gap/2}

# Construct backbone and LUC position (0..1 schematic coords)
construct_layout_default = {"x0": 0.10, "x1": 0.92, "luc_x": 0.78}


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
                         text_pad=0.015, lw=1.2):
    """
    Detached bracket:
      - horizontal lines from x_start -> x_end (top & bottom)
      - vertical line at x_end
      - text to the right (not clipped)
    """
    ax.plot([x_start, x_end], [y1, y1], color=ink, lw=lw, zorder=5)
    ax.plot([x_start, x_end], [y2, y2], color=ink, lw=lw, zorder=5)
    ax.plot([x_end, x_end], [y1, y2], color=ink, lw=lw, zorder=5)

    x_text = x_end + text_pad * ax.get_xlim()[1]
    y_text = (y1 + y2) / 2
    ax.text(x_text, y_text, text,
            va="center", ha="left",
            fontsize=12, fontweight="bold", color=ink,
            zorder=6, clip_on=False)


def draw_construct_row(ax_s, y, x0, x1, luc_x):
    """Formal construct: flat backbone + rectangular end caps + premium LUC cassette."""
    # Backbone
    ax_s.plot([x0, x1], [y, y], color=ink, lw=7, solid_capstyle="butt", zorder=2)

    # End caps
    cap_h = 0.28
    cap_w = 0.018
    ax_s.add_patch(Rectangle((x0 - cap_w/2, y - cap_h/2), cap_w, cap_h,
                             facecolor=ink, edgecolor=ink, linewidth=0, zorder=3))
    ax_s.add_patch(Rectangle((x1 - cap_w/2, y - cap_h/2), cap_w, cap_h,
                             facecolor=ink, edgecolor=ink, linewidth=0, zorder=3))

    # LUC cassette
    luc_w = 0.10
    luc_h = 0.30
    box = FancyBboxPatch(
        (luc_x - luc_w/2, y - luc_h/2),
        luc_w, luc_h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
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
fig = plt.figure(figsize=(11.2, 4.8), dpi=160)
gs = GridSpec(1, 2, width_ratios=[4, 4.0], wspace=0.00)

ax_s = fig.add_subplot(gs[0, 0])              # schematic
ax   = fig.add_subplot(gs[0, 1], sharey=ax_s) # bars

# Reverse so first label appears at top
labels_r = labels[::-1]
means_r  = means[::-1]
errs_r   = errs[::-1]
y = np.arange(len(labels_r))

# Panel label and title
fig.text(0.015, 0.97, "B.", ha="left", va="top",
         fontsize=18, fontweight="bold", color=ink)
fig.text(0.08, 0.97, "Haplotypes of", ha="left", va="top",
         fontsize=18, fontweight="bold", color=ink)

# -----------------------------
# 3A) Left panel: haplotype schematic (SNP columns + LUC)
# -----------------------------
ax_s.set_xlim(0, 1)
ax_s.set_ylim(-0.5, len(labels_r) - 0.5)
ax_s.set_xticks([])
ax_s.set_yticks(y)
ax_s.set_yticklabels([])
ax_s.tick_params(axis="y", left=False, right=False)

for spine in ["top", "right", "bottom", "left"]:
    ax_s.spines[spine].set_visible(False)

# SNP dashed vertical guides (across rows)
dash = (0, (1.0, 2.5))
ax_s.axvline(snp1["x"], ymin=0.10, ymax=0.90, color=ink, lw=1.2, linestyle=dash, zorder=1)
ax_s.axvline(snp2["x"], ymin=0.10, ymax=0.90, color=ink, lw=1.2, linestyle=dash, zorder=1)

# SNP labels at top
ax_s.text(snp1["x"], 1.02, snp1["label"], transform=ax_s.transAxes,
          ha="center", va="bottom", fontsize=11, fontweight="bold", color=ink)
ax_s.text(snp2["x"], 1.02, snp2["label"], transform=ax_s.transAxes,
          ha="center", va="bottom", fontsize=11, fontweight="bold", color=ink)

# Draw each haplotype row: construct + allele letters
x0 = construct_layout_default["x0"]
x1 = construct_layout_default["x1"]
luc_x = construct_layout_default["luc_x"]

for yi, lab in zip(y, labels_r):
    draw_construct_row(ax_s, yi, x0=x0, x1=x1, luc_x=luc_x)

    a1, a2 = allele_map.get(lab, ("", ""))
    ax_s.text(snp1["x"], yi + 0.23, a1, ha="center", va="bottom",
              fontsize=12, fontweight="bold", color=ink)
    ax_s.text(snp2["x"], yi + 0.23, a2, ha="center", va="bottom",
              fontsize=12, fontweight="bold", color=ink)

# -----------------------------
# 3B) Right panel: bars + errors
# -----------------------------
non_mask = np.ones_like(means_r, dtype=bool)
norm = Normalize(vmin=means_r[non_mask].min(), vmax=means_r[non_mask].max())

colors = [cmap(norm(m)) for m in means_r]

ax.barh(y, means_r, height=0.56, color=colors, edgecolor="none", zorder=2)
ax.errorbar(means_r, y, xerr=errs_r, fmt="none",
            ecolor=ink, elinewidth=1.2, capsize=6, capthick=1.2, zorder=3)

ax.set_yticks(y)
ax.set_yticklabels([])
ax.tick_params(axis="y", length=0)

ax.set_xlabel("Relative Luciferase Unit (RLU)", fontsize=16, fontweight="bold",
              color=ink, labelpad=14)

ax.xaxis.set_major_formatter(FuncFormatter(comma_formatter))
ax.xaxis.set_major_locator(MultipleLocator(500_000))
ax.xaxis.set_minor_locator(MultipleLocator(250_000))

ax.tick_params(axis="x", labelsize=12, width=1.2, length=8, colors=ink)
ax.tick_params(axis="x", which="minor", length=4, width=1.0, colors=ink)

# Vertical tick labels
plt.setp(ax.get_xticklabels(), rotation=90, ha="center", va="top")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color(ink)
ax.spines["bottom"].set_color(ink)

# Match the reference scale (~0..2,500,000), but allow room for bracket text
x_axis_max_ticks = 2_500_000
ax.set_xlim(0, 3_499_999)
# Keep locator so last tick shown is 2,500,000
ax.xaxis.set_major_locator(MultipleLocator(500_000))

# -----------------------------
# 4) P-VALUE BRACKETS (nested)
# -----------------------------
label_to_y   = {lab: yi for lab, yi in zip(labels_r, y)}
label_to_end = {lab: (m + e) for lab, m, e in zip(labels_r, means_r, errs_r)}

# Start all brackets after the furthest errorbar end
max_end = max(label_to_end.values())
gap = 0.05 * x_axis_max_ticks
common_start = max_end + gap

for a, b, ptxt in comparisons:
    if a not in label_to_y or b not in label_to_y:
        continue

    y1, y2 = label_to_y[a], label_to_y[b]
    y_low, y_high = min(y1, y2), max(y1, y2)

    # nested end positions (or fallback)
    k = (a, b) if (a, b) in bracket_end_abs else (b, a)
    x_end = bracket_end_abs.get(k, common_start + 300_000)

    # Ensure a visible span
    if x_end < common_start + 120_000:
        x_end = common_start + 120_000

    add_detached_bracket(ax, y_low, y_high, x_start=common_start, x_end=x_end, text=ptxt)

# -----------------------------
# 5) LAYOUT + EXPORT
# -----------------------------
fig.subplots_adjust(left=0.050, bottom=0.26, top=.70)
fig.canvas.draw()

out_dir = r"C:\Users\User\OneDrive\2. Personal\AAM\Desktop\Luciferase Unit\final figures"
os.makedirs(out_dir, exist_ok=True)

out_base = os.path.join(out_dir, "IL-33_3-UTR second part B")

fig.savefig(out_base + ".png", dpi=600, bbox_inches="tight")
fig.savefig(out_base + ".pdf", bbox_inches="tight")
fig.savefig(out_base + ".svg", bbox_inches="tight")

print("Saved files to:")
print(out_dir)
print("PNG:", out_base + ".png")
print("PDF:", out_base + ".pdf")
print("SVG:", out_base + ".svg")

plt.show()
