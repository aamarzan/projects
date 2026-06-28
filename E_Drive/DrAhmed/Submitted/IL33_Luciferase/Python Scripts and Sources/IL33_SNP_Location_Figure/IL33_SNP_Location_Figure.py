import numpy as np
import matplotlib.pyplot as plt
import os

from matplotlib.patches import Rectangle, FancyBboxPatch, Polygon
from matplotlib.ticker import FuncFormatter

# =============================
# CENTRAL PARAMETERS (EDIT IN THIS SECTION ONLY)
# =============================

# --- Font configuration (applies to all text elements) ---
FS_BASE = 19                # Global base font size
FS_AXIS_TICKS = 15          # X-axis tick label font size
FS_BOTTOM_LEFT = 15         # Bottom-left annotation font size
FS_CHR_TITLE = 17           # Chromosome label font size
FS_GRCH = 17                # Reference assembly label font size
FS_GENE_LABEL = 15          # Gene label font size
FS_EXON_MAIN = 13           # Exon1/Exon2 label font size
FS_EXON_CLUSTER = 13        # Exon3–Exon8 label font size
FS_SNP = 13                 # SNP (rs...) label font size

# --- Exon box geometry (controlled centrally) ---
EXON_H = 0.0                # Exon box height (y-axis units)
EXON_ROUND = 0.525          # Exon box corner rounding
EXON_PAD = 0.03             # Padding inside exon box style
EXON_WIDTH_SCALE = 1.00     # Global scaling factor applied to exon widths (bp)
EXON_MIN_GAP_BP = 220       # Minimum separation between packed exon boxes (bp)

# --- SNP box geometry (controlled centrally) ---
SNP_H = 0.02                # SNP box height (y-axis units)
SNP_ROUND = 0.010           # SNP box corner rounding
SNP_PAD = 0.01              # Padding inside SNP box style
SNP_WIDTH_MIN = 2200        # Minimum SNP box width (bp)
SNP_WIDTH_PER_CHAR = 230    # Additional SNP width per character (bp)
SNP_WIDTH_BONUS = 700       # Extra SNP width padding (bp)
LANE_MIN_GAP_BP = 250       # Minimum separation for SNP boxes within the same y-lane (bp)

# --- Connector line styling ---
DASH = (0, (1.0, 2.5))      # Dash pattern for dotted connectors
LINE_W = 1.2                # Connector line width
AXIS_LINE_W = 1.6           # Baseline axis thickness

# --- Output geometry (fixed pixel dimensions) ---
OUT_W_PX = 7500             # Output width in pixels
OUT_H_PX = 3300             # Output height in pixels
OUT_DPI  = 300              # Output DPI (pixels = inches * DPI)

# =============================
# 0) OUTPUT PATHS
# =============================
out_dir = r"C:\Users\User\OneDrive\2. Personal\AAM\Desktop\Luciferase Unit\final figures"
os.makedirs(out_dir, exist_ok=True)
out_base = os.path.join(out_dir, "IL33_SNP_Location_Figure")

# =============================
# 1) GLOBAL STYLE SETTINGS
# =============================
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": FS_BASE,
    "axes.linewidth": 1.2,
})

ink = "#0B1F3B"
exon_color = "#B56A78"
chr_fill = "#F4D6D8"
chr_block = "#6F6F6F"
snp_box_fc = "#C9C6EA"
snp_box_ec = "none"

def comma_fmt(x, pos):
    return f"{int(x):,}"

# =============================
# 2) GENOMIC COORDINATE AXIS (bp)
# =============================
XMIN = -3000
XMAX = 45000
XTICKS = [0, 15000, 20000, 25000, 30000, 35000, 40000, 45000]

# =============================
# 3) PANEL LAYOUT (y-positions)
# =============================
y_chr = 0.92
h_chr = 0.06

y_gene = 0.70
y_exon_center = y_gene + 0.03
h_exon = EXON_H

y_baseline = 0.08

# =============================
# 4) EXON DEFINITIONS (name, center position, width in bp)
# =============================
EXONS = [
    ("Exon1",  2500,  2200),
    ("Exon2", 23000,  2400),
    ("Exon3", 34000,  1800),
    ("Exon4", 35400,  1800),
    ("Exon5", 36700,  1800),
    ("Exon6", 38000,  1800),
    ("Exon7", 39200,  1800),
    ("Exon8", 40400,  1800),
]

# Apply global scaling to exon widths (optional)
# (This is used for consistent visual emphasis without changing exon coordinates.)
EXONS = [(n, cx, w * EXON_WIDTH_SCALE) for (n, cx, w) in EXONS]

# =============================
# 5) SNP DEFINITIONS
#   - All SNPs connect down to the baseline.
#   - Only selected SNPs additionally connect to an intron gap between exon pairs.
# =============================
SNPS = [
    dict(label="rs73639580", x=-1200, y=0.44, connect_to_exon=False),
    dict(label="rs12343534", x= 1200, y=0.55, connect_to_exon=False),

    dict(label="rs10815393", x=24000, y=0.20, connect_to_exon=False),
    dict(label="rs10118795", x=24800, y=0.48, connect_to_exon=False),
    dict(label="rs4742170",  x=27200, y=0.38, connect_to_exon=False),
    dict(label="rs10975514", x=29500, y=0.52, connect_to_exon=False),
    dict(label="rs10975516", x=32200, y=0.33, connect_to_exon=False),
    dict(label="rs1317230",  x=33400, y=0.24, connect_to_exon=False),
    dict(label="rs1929992",  x=34050, y=0.41, connect_to_exon=False),

    dict(label="rs7044343",  x=35000, y=0.54, connect_to_exon=True,
         anchor_between=("Exon3", "Exon4"), style="elbow"),
    dict(label="rs10975520", x=35000, y=0.46, connect_to_exon=True,
         anchor_between=("Exon3", "Exon4"), style="elbow"),

    dict(label="rs1332290",  x=37400, y=0.48, connect_to_exon=True,
         anchor_between=("Exon6", "Exon7"), style="elbow"),
    dict(label="rs12336076", x=37200, y=0.40, connect_to_exon=True,
         anchor_between=("Exon6", "Exon7"), style="elbow"),

    dict(label="rs8172",     x=40500, y=0.46, connect_to_exon=True,
         anchor_between=("Exon7", "Exon8"), style="elbow"),

    dict(label="rs55726619", x=39500, y=0.23, connect_to_exon=False),
    dict(label="rs1048274",  x=43200, y=0.28, connect_to_exon=False),
]

# =============================
# 6) HELPER FUNCTIONS
# =============================
def pack_exons(exons, min_gap_bp):
    packed = []
    prev_end = None
    for name, cx, w in exons:
        start = cx - w/2
        end   = cx + w/2
        if prev_end is not None and start < prev_end + min_gap_bp:
            start = prev_end + min_gap_bp
            end = start + w
            cx = (start + end) / 2
        packed.append(dict(name=name, cx=cx, w=w, start=start, end=end))
        prev_end = end
    return packed

def draw_round_box(ax, x, y, w_bp, h, fc, ec, r=0.012, pad=0.01, z=5):
    box = FancyBboxPatch(
        (x - w_bp/2, y - h/2),
        w_bp, h,
        boxstyle=f"round,pad={pad},rounding_size={r}",
        facecolor=fc, edgecolor=ec, linewidth=0.0,
        zorder=z
    )
    ax.add_patch(box)
    return box

def snp_box_width_bp(label):
    return max(SNP_WIDTH_MIN, SNP_WIDTH_PER_CHAR * len(label) + SNP_WIDTH_BONUS)

def v_dashed(ax, x, y0, y1, z=2):
    ax.plot([x, x], [y0, y1], color=ink, lw=LINE_W, linestyle=DASH, zorder=z)

def h_dashed(ax, x0, x1, y, z=2):
    ax.plot([x0, x1], [y, y], color=ink, lw=LINE_W, linestyle=DASH, zorder=z)

def spread_snp_y(snps, y_min=0.18, y_max=0.60):
    ys = sorted(set(s["y"] for s in snps))
    if len(ys) == 1:
        return {ys[0]: (y_min + y_max)/2}
    return {y: y_min + i*(y_max - y_min)/(len(ys)-1) for i, y in enumerate(ys)}

def pack_boxes_in_lane(items, min_gap_bp=250):
    items = sorted(items, key=lambda d: d["x_box"])
    prev_end = None
    for it in items:
        start = it["x_box"] - it["w_bp"]/2
        end   = it["x_box"] + it["w_bp"]/2
        if prev_end is not None and start < prev_end + min_gap_bp:
            shift = (prev_end + min_gap_bp) - start
            it["x_box"] += shift
            start += shift
            end   += shift
        prev_end = end
    return items

# =============================
# 7) INITIALIZE FIGURE AND AXES
# =============================
fig, ax = plt.subplots(figsize=(OUT_W_PX/OUT_DPI, OUT_H_PX/OUT_DPI), dpi=OUT_DPI)
ax.set_xlim(XMIN, XMAX)
ax.set_ylim(0, 1)

ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

ax.spines["bottom"].set_position(("data", y_baseline))
ax.spines["bottom"].set_color("#7A7A7A")
ax.spines["bottom"].set_linewidth(AXIS_LINE_W)

ax.set_xticks(XTICKS)
ax.xaxis.set_major_formatter(FuncFormatter(comma_fmt))
ax.tick_params(axis="x", length=7, width=1.2, colors="#4F4F4F", labelsize=FS_AXIS_TICKS)

ax.text(
    XMIN, y_baseline - 0.06,
    "Location of the SNPs\ninvestigated in this\nstudy",
    ha="left", va="top", fontsize=FS_BOTTOM_LEFT, color="#3A3A3A"
)

# =============================
# 8) CHROMOSOME CONTEXT TRACK
# =============================
ax.text(XMIN + 200, y_chr + h_chr/2 + 0.03, "Chromosome 9",
        ha="left", va="bottom", fontsize=FS_CHR_TITLE, color="#3A3A3A", fontstyle="italic")

ax.add_patch(Rectangle((XMIN, y_chr - h_chr/2), XMAX - XMIN, h_chr,
                       facecolor=chr_fill, edgecolor="none", zorder=1))

GRCH_X = 30000
ax.text(GRCH_X, y_chr, "(GRCh37.p13)", ha="center", va="center",
        fontsize=FS_GRCH, color="#3A3A3A", fontstyle="italic")

for pos in [17000, 25000, 37000]:
    w = 900
    ax.add_patch(Rectangle((pos - w/2, y_chr - h_chr/2), w, h_chr,
                           facecolor=chr_block, edgecolor="none", zorder=2))

ax.plot([GRCH_X,  1000], [y_chr - h_chr/2, y_gene + 0.06], color="#5A5A5A", lw=1.0, linestyle=":", zorder=1)
ax.plot([GRCH_X, 44000], [y_chr - h_chr/2, y_gene + 0.06], color="#5A5A5A", lw=1.0, linestyle=":", zorder=1)

# =============================
# 9) GENE TRACK WITH ORIENTATION MARKERS
# =============================
gene_start = 0
gene_end = 45000
ax.plot([gene_start, gene_end], [y_gene, y_gene], color="#3B3B3B", lw=1.6, zorder=2)

tri_L = Polygon([[gene_start - 900, y_gene + 0.04],
                 [gene_start - 900, y_gene - 0.04],
                 [gene_start - 200, y_gene]],
                closed=True, facecolor="#1F1F1F", edgecolor="none", zorder=3)
ax.add_patch(tri_L)

tri_R = Polygon([[gene_end + 200, y_gene + 0.04],
                 [gene_end + 200, y_gene - 0.04],
                 [gene_end + 900, y_gene]],
                closed=True, facecolor="#1F1F1F", edgecolor="none", zorder=3)
ax.add_patch(tri_R)

ax.text(gene_start - 1300, y_gene + 0.05, "5′", ha="center", va="center", fontsize=10, color="#1F1F1F")
ax.text(gene_end + 1300,   y_gene + 0.05, "3′", ha="center", va="center", fontsize=10, color="#1F1F1F")
ax.text(gene_start - 2600, y_gene, "IL-33\nGene", ha="right", va="center",
        fontsize=FS_GENE_LABEL, color="#1F1F1F")

# =============================
# 10) EXON TRACK (packed to prevent overlap)
# =============================
exon_head = EXONS[:2]
exon_cluster = EXONS[2:]
packed_cluster = pack_exons(exon_cluster, EXON_MIN_GAP_BP)

packed_exons = [dict(name=n, cx=cx, w=w, start=cx-w/2, end=cx+w/2) for (n, cx, w) in exon_head] + packed_cluster

intron_x = {}
for i in range(len(packed_exons) - 1):
    a = packed_exons[i]
    b = packed_exons[i + 1]
    intron_x[(a["name"], b["name"])] = (a["end"] + b["start"]) / 2

for e in packed_exons:
    draw_round_box(ax, (e["start"] + e["end"]) / 2, y_exon_center,
                   e["end"] - e["start"], h_exon,
                   fc=exon_color, ec="none",
                   r=EXON_ROUND, pad=EXON_PAD, z=4)

    fs = FS_EXON_CLUSTER if e["name"] in ["Exon3","Exon4","Exon5","Exon6","Exon7","Exon8"] else FS_EXON_MAIN
    ax.text(e["cx"], y_exon_center, e["name"],
            ha="center", va="center", fontsize=fs, color="white", fontweight="bold", zorder=5)

# =============================
# 11) SNP TRACK AND CONNECTORS
#   - SNP connectors terminate at the SNP box boundaries (no pass-through).
#   - SNP labels are spread across y-lanes to minimize overlap.
# =============================
y_exon_bottom = y_exon_center - h_exon/2

y_map = spread_snp_y(SNPS, y_min=0.18, y_max=0.60)
for s in SNPS:
    s["y"] = y_map[s["y"]]

snp_geom = []
for s in SNPS:
    w_bp = snp_box_width_bp(s["label"])
    y = s["y"]
    y0 = y - SNP_H/2
    y1 = y + SNP_H/2
    snp_geom.append({**s, "w_bp": w_bp, "y0": y0, "y1": y1, "x_box": s["x"]})

lanes = {}
for s in snp_geom:
    lanes.setdefault(s["y"], []).append(s)
for _, items in lanes.items():
    pack_boxes_in_lane(items, min_gap_bp=LANE_MIN_GAP_BP)

for s in snp_geom:
    x_true = s["x"]
    x_box  = s["x_box"]
    v_dashed(ax, x_true, y_baseline, s["y0"], z=2)
    if abs(x_true - x_box) > 1e-6:
        h_dashed(ax, x_true, x_box, s["y0"], z=2)

for s in snp_geom:
    if not s.get("connect_to_exon", False):
        continue

    anchor_pair = s.get("anchor_between", None)
    x_anchor = intron_x.get(anchor_pair, s["x"])

    elbow_y = min(y_exon_bottom - 0.02, s["y1"] + 0.12)
    elbow_y = max(elbow_y, s["y1"] + 0.06)

    v_dashed(ax, x_anchor, y_exon_bottom, elbow_y, z=2)
    h_dashed(ax, x_anchor, s["x_box"], elbow_y, z=2)
    v_dashed(ax, s["x_box"], elbow_y, s["y1"], z=2)

for s in snp_geom:
    draw_round_box(ax, s["x_box"], s["y"], s["w_bp"], SNP_H,
                   fc=snp_box_fc, ec=snp_box_ec, r=SNP_ROUND, pad=SNP_PAD, z=6)
    ax.text(s["x_box"], s["y"], s["label"], ha="center", va="center",
            fontsize=FS_SNP, fontweight="bold", color="#1F1F1F", zorder=7)

# =============================
# 12) EXPORT (publication-ready formats)
# =============================
plt.subplots_adjust(left=0.05, right=0.99, top=0.95, bottom=0.10)

fig.savefig(out_base + ".png", dpi=OUT_DPI, bbox_inches="tight")
fig.savefig(out_base + ".pdf", bbox_inches="tight")
fig.savefig(out_base + ".svg", bbox_inches="tight")

print("Saved:")
print(out_base + ".png")
print(out_base + ".pdf")
print(out_base + ".svg")

plt.show()
