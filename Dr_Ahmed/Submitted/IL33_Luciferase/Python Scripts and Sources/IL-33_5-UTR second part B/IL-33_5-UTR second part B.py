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
labels = ["C-G", "C-C"]

means = np.array([1050, 2600], dtype=float)
errs  = np.array([350, 350], dtype=float)

allele_map = {
    "C-G": ("C", "G"),
    "C-C": ("C", "C"),
}

comparisons = [
    ("C-G", "C-C", "P = 0.006"),
]


# -----------------------------
# 1B) SNP COLUMN SETTINGS (schematic)
# -----------------------------
snp_center = 0.44
snp_gap = 0.34
snp1 = {"label": "rs12343534 (A/C)", "x": snp_center - snp_gap/2}
snp2 = {"label": "rs73639580 (C/G)", "x": snp_center + snp_gap/2}

construct_layout_default = {"x0": 0.12, "x1": 0.93, "luc_x": 0.78}


# -----------------------------
# 2) STYLE
# -----------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.linewidth": 1.2,
})

sky_blue  = "#8BA1AC"
dark_blue = "#134058"
cmap = LinearSegmentedColormap.from_list("premium_blue", [sky_blue, dark_blue])

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
            fontsize=12, fontweight="bold", color=ink, zorder=6, clip_on=False)


def draw_construct_row(ax_s, y, x0, x1, luc_x):
    ax_s.plot([x0, x1], [y, y], color=ink, lw=7, solid_capstyle="butt", zorder=2)

    cap_h = 0.28
    cap_w = 0.018
    ax_s.add_patch(Rectangle((x0 - cap_w/2, y - cap_h/2), cap_w, cap_h,
                             facecolor=ink, edgecolor=ink, linewidth=0, zorder=3))
    ax_s.add_patch(Rectangle((x1 - cap_w/2, y - cap_h/2), cap_w, cap_h,
                             facecolor=ink, edgecolor=ink, linewidth=0, zorder=3))

    luc_w = 0.14
    luc_h = 0.32
    box = FancyBboxPatch(
        (luc_x - luc_w/2, y - luc_h/2),
        luc_w, luc_h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor=ink, edgecolor=ink, linewidth=1.0, zorder=5
    )
    box.set_path_effects([
        pe.SimplePatchShadow(offset=(1.1, -1.1), alpha=0.16),
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
fig = plt.figure(figsize=(11.0, 4.8), dpi=160)
gs = GridSpec(1, 2, width_ratios=[4.2, 4.0], wspace=0.00)

ax_s = fig.add_subplot(gs[0, 0])
ax   = fig.add_subplot(gs[0, 1], sharey=ax_s)

labels_r = labels[::-1]
means_r  = means[::-1]
errs_r   = errs[::-1]
y = np.arange(len(labels_r))

# ✅ Panel label "B." (top-left)
fig.text(0.015, 0.97, "B.", ha="left", va="top",
         fontsize=18, fontweight="bold", color=ink)
fig.text(0.08, 0.97, "Haplotypes of", ha="left", va="top",
         fontsize=18, fontweight="bold", color=ink)

# -----------------------------
# 3A) Left panel: schematic
# -----------------------------
ax_s.set_xlim(0, 1)
ax_s.set_ylim(-0.5, len(labels_r) - 0.5)
ax_s.set_xticks([])
ax_s.set_yticks(y)
ax_s.set_yticklabels([])
ax_s.tick_params(axis="y", left=False, right=False)

for spine in ["top", "right", "bottom", "left"]:
    ax_s.spines[spine].set_visible(False)

dash = (0, (1.0, 2.5))
ax_s.axvline(snp1["x"], ymin=0.12, ymax=0.88, color=ink, lw=1.2, linestyle=dash, zorder=1)
ax_s.axvline(snp2["x"], ymin=0.12, ymax=0.88, color=ink, lw=1.2, linestyle=dash, zorder=1)

ax_s.text(snp1["x"], 1.02, snp1["label"], transform=ax_s.transAxes,
          ha="center", va="bottom", fontsize=11, fontweight="bold", color=ink)
ax_s.text(snp2["x"], 1.02, snp2["label"], transform=ax_s.transAxes,
          ha="center", va="bottom", fontsize=11, fontweight="bold", color=ink)

# ✅ Removed left-side haplotype labels ("C-G", "C-C")

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
norm = Normalize(vmin=means_r.min(), vmax=means_r.max())
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
ax.xaxis.set_major_locator(MultipleLocator(500))
ax.xaxis.set_minor_locator(MultipleLocator(250))

ax.tick_params(axis="x", labelsize=12, width=1.2, length=8, colors=ink)
ax.tick_params(axis="x", which="minor", length=4, width=1.0, colors=ink)

plt.setp(ax.get_xticklabels(), rotation=90, ha="center", va="top")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color(ink)
ax.spines["bottom"].set_color(ink)

# ✅ extend xlim to include bracket/text; user wants p-value around x=4500
ax.set_xlim(0, 3499)

# -----------------------------
# 4) P-VALUE BRACKET (force end at x=4500)
# -----------------------------
label_to_y = {lab: yi for lab, yi in zip(labels_r, y)}
label_to_end = {lab: (m + e) for lab, m, e in zip(labels_r, means_r, errs_r)}

max_end = max(label_to_end.values())
x_start = max_end + 150  # small gap after the furthest errorbar end
x_end   = 3500           # ✅ requested position for the bracket/text anchor

for a, b, ptxt in comparisons:
    y1, y2 = label_to_y[a], label_to_y[b]
    add_detached_bracket(ax, min(y1, y2), max(y1, y2),
                         x_start=x_start, x_end=x_end, text=ptxt, text_pad_frac=0.01)

# -----------------------------
# 5) LAYOUT + EXPORT
# -----------------------------
fig.subplots_adjust(left=0.06, bottom=0.26, top=0.72)
fig.canvas.draw()

out_dir = r"C:\Users\User\OneDrive\2. Personal\AAM\Desktop\Luciferase Unit\final figures"
os.makedirs(out_dir, exist_ok=True)

out_base = os.path.join(out_dir, "IL-33_5-UTR_second_part_B")

fig.savefig(out_base + ".png", dpi=600, bbox_inches="tight")
fig.savefig(out_base + ".pdf", bbox_inches="tight")
fig.savefig(out_base + ".svg", bbox_inches="tight")

print("Saved files to:")
print(out_dir)
print("PNG:", out_base + ".png")
print("PDF:", out_base + ".pdf")
print("SVG:", out_base + ".svg")

plt.show()
