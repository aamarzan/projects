import os, csv, math, textwrap
from collections import Counter, defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import gridspec

WORK = "/mnt/e/DrAhmed/Ongoing/WGS/Result"
INDIR = f"{WORK}/_FIGURE_RAW_MATERIALS_V2"
OUTDIR = f"{WORK}/_MANUSCRIPT_FIGURES_G2_V2"
os.makedirs(OUTDIR, exist_ok=True)

# -----------------------------
# helpers
# -----------------------------
def first_existing(paths):
    for p in paths:
        if os.path.isfile(p):
            return p
    return None

def read_csv(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))

def as_int(x, default=0):
    try:
        s = str(x).strip()
        if s == "":
            return default
        return int(float(s))
    except:
        return default

def as_float(x, default=0.0):
    try:
        s = str(x).strip()
        if s == "":
            return default
        return float(s)
    except:
        return default

def wrap(s, width=18):
    return textwrap.fill(str(s), width=width)

def species_short(s):
    mp = {
        "Acinetobacter baumannii": "A. baumannii",
        "Klebsiella pneumoniae": "K. pneumoniae",
        "Pseudomonas aeruginosa": "P. aeruginosa",
        "Escherichia coli": "E. coli",
        "Serratia marcescens": "S. marcescens",
        "Serratia nevei": "S. nevei",
        "Homo sapiens": "H. sapiens",
    }
    return mp.get(s, s)

def savefig_all(fig, stem):
    png = os.path.join(OUTDIR, stem + ".png")
    pdf = os.path.join(OUTDIR, stem + ".pdf")
    svg = os.path.join(OUTDIR, stem + ".svg")
    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)

def counter_top_other(counter_obj, topn=10):
    items = counter_obj.most_common(topn)
    other = sum(counter_obj.values()) - sum(v for _, v in items)
    labels = [k for k, _ in items]
    values = [v for _, v in items]
    if other > 0:
        labels.append("Other")
        values.append(other)
    return labels, values

def normalize_columns(mat):
    m = np.array(mat, dtype=float)
    out = np.zeros_like(m)
    for j in range(m.shape[1]):
        mx = np.nanmax(m[:, j]) if m.shape[0] else 0
        if mx > 0:
            out[:, j] = m[:, j] / mx
    return out

def build_matrix(rows, row_order, col_order, row_key, col_key, val_key):
    rmap = {r:i for i, r in enumerate(row_order)}
    cmap = {c:i for i, c in enumerate(col_order)}
    mat = np.zeros((len(row_order), len(col_order)), dtype=float)
    for row in rows:
        r = row.get(row_key, "")
        c = row.get(col_key, "")
        if r in rmap and c in cmap:
            mat[rmap[r], cmap[c]] += as_float(row.get(val_key, 0))
    return mat

def existing_required(path, label):
    if not path:
        raise SystemExit(f"Missing required input for {label}")
    return path

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 15,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 18,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
})

# -----------------------------
# input files (robust lookup)
# -----------------------------
species_counts_fp = existing_required(first_existing([
    f"{INDIR}/species_counts_166.csv",
]), "species counts")

species_conf_fp = existing_required(first_existing([
    f"{INDIR}/species_confidence_counts_166.csv",
]), "species confidence counts")

species_bio_fp = existing_required(first_existing([
    f"{INDIR}/species_biology_summary_166.csv",
]), "species biology summary")

mlst_fp = existing_required(first_existing([
    f"{INDIR}/mlst_species_st_counts_166.csv",
]), "MLST species/ST counts")

amr_class_fp = existing_required(first_existing([
    f"{INDIR}/amr_class_by_species_166.csv",
]), "AMR class by species")

amr_gene_fp = existing_required(first_existing([
    f"{INDIR}/amr_gene_by_species_top200_166.csv",
    f"{INDIR}/amr_genes_by_species_top200_166.csv",
]), "AMR gene by species")

vfdb_fp = existing_required(first_existing([
    f"{INDIR}/virulence_gene_by_species_top200_166.csv",
]), "virulence gene by species")

plasmid_fp = existing_required(first_existing([
    f"{INDIR}/plasmid_replicon_by_species_166.csv",
    f"{INDIR}/plasmid_gene_by_species_top200_166.csv",
    f"{INDIR}/plasmid_replicon_by_species_top200_166.csv",
]), "plasmid replicon by species")

serotype_fp = first_existing([
    f"{INDIR}/serotype_counts_166.csv",
])

species_counts = read_csv(species_counts_fp)
species_conf = read_csv(species_conf_fp)
species_bio = read_csv(species_bio_fp)
mlst_rows = read_csv(mlst_fp)
amr_class_rows = read_csv(amr_class_fp)
amr_gene_rows = read_csv(amr_gene_fp)
vfdb_rows = read_csv(vfdb_fp)
plasmid_rows = read_csv(plasmid_fp)
serotype_rows = read_csv(serotype_fp) if serotype_fp else []

species_order = [r["TopSpecies"] for r in species_counts]
species_short_order = [species_short(x) for x in species_order]
species_count_map = {r["TopSpecies"]: as_int(r["Count"]) for r in species_counts}
conf_map = {r["TopSpecies"]: r for r in species_conf}
bio_map = {r["TopSpecies"]: r for r in species_bio}

# palettes
cmap_blue = LinearSegmentedColormap.from_list("bluegrad", ["#e8f1ff", "#7aaeff", "#1e4e9a"])
cmap_teal = LinearSegmentedColormap.from_list("tealgrad", ["#e7fff8", "#58d3b2", "#0f6c5c"])
cmap_gold = LinearSegmentedColormap.from_list("goldgrad", ["#fff6dc", "#f6c45a", "#a45c00"])
cmap_mag = LinearSegmentedColormap.from_list("maggrad", ["#faebff", "#d377ff", "#6e269e"])
cmap_heat = LinearSegmentedColormap.from_list("heatgrad", ["#fff7ec", "#fdd49e", "#fc8d59", "#d7301f", "#7f0000"])
cmap_green = LinearSegmentedColormap.from_list("greengrad", ["#edf8e9", "#74c476", "#006d2c"])
cmap_purple = LinearSegmentedColormap.from_list("purplegrad", ["#f2f0f7", "#9e9ac8", "#54278f"])

# -----------------------------
# FIGURE 1
# Cohort + species + confidence + biology scatter
# -----------------------------
total = sum(species_count_map.values())
hc_total = sum(as_int(conf_map.get(s, {}).get("HighConfidence_n", 0)) for s in species_order)
pr_total = sum(as_int(conf_map.get(s, {}).get("PriorityReview_n", 0)) for s in species_order)
other_total = max(total - hc_total - pr_total, 0)

fig = plt.figure(figsize=(18, 12))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.28, wspace=0.25)

# A
axA = fig.add_subplot(gs[0, 0])
vals = [hc_total, pr_total, other_total]
labs = ["High-confidence", "Priority-review", "Other"]
cols = [cmap_teal(0.80), cmap_gold(0.75), cmap_blue(0.75)]
left = 0
for v, lab, c in zip(vals, labs, cols):
    axA.barh([0], [v], left=left, height=0.55, color=c, edgecolor="white", linewidth=1.2)
    if v > 0:
        axA.text(left + v/2, 0, f"{lab}\n{v}", ha="center", va="center", fontweight="bold", fontsize=12)
    left += v
axA.set_xlim(0, total)
axA.set_yticks([])
axA.set_xlabel("Samples")
axA.set_title("A. Cohort composition", loc="left", fontweight="bold")
axA.text(0, 0.55, f"Total samples = {total}", fontsize=12, fontweight="bold")
axA.grid(axis="x", alpha=0.15)

# B
axB = fig.add_subplot(gs[0, 1])
counts = [species_count_map[s] for s in species_order]
yy = np.arange(len(species_order))
bar_cols = [cmap_blue(0.35 + 0.5*(i/max(len(counts)-1,1))) for i in range(len(counts))]
axB.barh(yy, counts, color=bar_cols, edgecolor="white", linewidth=0.8)
axB.set_yticks(yy)
axB.set_yticklabels(species_short_order)
axB.invert_yaxis()
axB.set_xlabel("Samples")
axB.set_title("B. Species distribution", loc="left", fontweight="bold")
axB.grid(axis="x", alpha=0.15)
for i, v in enumerate(counts):
    axB.text(v + 0.6, i, str(v), va="center", fontsize=10, fontweight="bold")

# C
axC = fig.add_subplot(gs[1, 0])
hc = np.array([as_int(conf_map.get(s, {}).get("HighConfidence_n", 0)) for s in species_order])
pr = np.array([as_int(conf_map.get(s, {}).get("PriorityReview_n", 0)) for s in species_order])
ot = np.array([as_int(conf_map.get(s, {}).get("Other_n", 0)) for s in species_order])
axC.barh(yy, hc, color=cmap_teal(0.80), edgecolor="white", linewidth=0.8, label="High-confidence")
axC.barh(yy, pr, left=hc, color=cmap_gold(0.75), edgecolor="white", linewidth=0.8, label="Priority-review")
axC.barh(yy, ot, left=hc+pr, color=cmap_blue(0.75), edgecolor="white", linewidth=0.8, label="Other")
axC.set_yticks(yy)
axC.set_yticklabels(species_short_order)
axC.invert_yaxis()
axC.set_xlabel("Samples")
axC.set_title("C. Confidence structure within species", loc="left", fontweight="bold")
axC.legend(frameon=False, ncol=3, loc="lower right")
axC.grid(axis="x", alpha=0.15)

# D
axD = fig.add_subplot(gs[1, 1])
x = [as_float(bio_map[s].get("Median_AMR_Genes_n", 0)) for s in species_order]
y = [as_float(bio_map[s].get("Median_VFDB_Hits_n", 0)) for s in species_order]
sizes = [max(80, 18 * as_float(bio_map[s].get("Samples_n", species_count_map[s]))) for s in species_order]
colors = [as_float(bio_map[s].get("Median_Plasmid_Hits_n", 0)) for s in species_order]
sc = axD.scatter(x, y, s=sizes, c=colors, cmap=cmap_mag, edgecolor="black", linewidth=0.6, alpha=0.85)
for i, s in enumerate(species_order):
    axD.text(x[i] + 0.4, y[i] + 0.4, species_short(s), fontsize=9, fontweight="bold")
axD.set_xlabel("Median AMR genes per sample")
axD.set_ylabel("Median VFDB hits per sample")
axD.set_title("D. Species biological profile", loc="left", fontweight="bold")
cbar = fig.colorbar(sc, ax=axD, fraction=0.045, pad=0.03)
cbar.set_label("Median plasmid hits")
axD.grid(alpha=0.15)

fig.suptitle("Figure 1. Cohort structure and species-level overview", y=0.995, fontweight="bold")
savefig_all(fig, "Figure01_Cohort_Species_Overview_v2")

# -----------------------------
# FIGURE 2
# Species biology heatmap + max burden bars
# -----------------------------
metrics = [
    ("Samples_n", "Samples"),
    ("HighConfidence_n", "High-conf"),
    ("PriorityReview_n", "Priority"),
    ("Median_AMR_Genes_n", "Median\nAMR genes"),
    ("Median_AMR_Classes_n", "Median\nAMR classes"),
    ("Median_VFDB_Hits_n", "Median\nVFDB hits"),
    ("Median_Plasmid_Hits_n", "Median\nplasmid hits"),
    ("Max_AMR_Genes_n", "Max\nAMR genes"),
    ("Max_VFDB_Hits_n", "Max\nVFDB hits"),
    ("Max_Plasmid_Hits_n", "Max\nplasmid hits"),
]
data = []
for s in species_order:
    row = []
    src = bio_map.get(s, {})
    for key, _ in metrics:
        if key in src:
            row.append(as_float(src[key], 0))
        elif key in conf_map.get(s, {}):
            row.append(as_float(conf_map[s][key], 0))
        else:
            row.append(0.0)
    data.append(row)
data = np.array(data, dtype=float)
norm = normalize_columns(data)

fig = plt.figure(figsize=(17, 9))
gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[2.8, 1.0], wspace=0.22)

ax1 = fig.add_subplot(gs[0, 0])
im = ax1.imshow(norm, aspect="auto", cmap=cmap_mag)
ax1.set_yticks(np.arange(len(species_order)))
ax1.set_yticklabels(species_short_order)
ax1.set_xticks(np.arange(len(metrics)))
ax1.set_xticklabels([lab for _, lab in metrics])
ax1.set_title("A. Species-level biology burden heatmap", loc="left", fontweight="bold")
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        v = data[i, j]
        txt = f"{int(v)}" if abs(v - int(v)) < 1e-9 else f"{v:.1f}"
        ax1.text(j, i, txt, ha="center", va="center",
                 fontsize=9, fontweight="bold",
                 color=("white" if norm[i, j] > 0.55 else "black"))
cbar = fig.colorbar(im, ax=ax1, fraction=0.03, pad=0.02)
cbar.set_label("Column-normalized intensity")

ax2 = fig.add_subplot(gs[0, 1])
max_amr = [as_float(bio_map[s].get("Max_AMR_Genes_n", 0)) for s in species_order]
max_vf  = [as_float(bio_map[s].get("Max_VFDB_Hits_n", 0)) for s in species_order]
max_pl  = [as_float(bio_map[s].get("Max_Plasmid_Hits_n", 0)) for s in species_order]
pos = np.arange(len(species_order))
h = 0.22
ax2.barh(pos - h, max_amr, height=h, color=cmap_heat(0.75), label="Max AMR genes")
ax2.barh(pos,     max_vf,  height=h, color=cmap_green(0.75), label="Max VFDB hits")
ax2.barh(pos + h, max_pl,  height=h, color=cmap_purple(0.75), label="Max plasmid hits")
ax2.set_yticks(pos)
ax2.set_yticklabels(species_short_order)
ax2.invert_yaxis()
ax2.set_xlabel("Maximum count")
ax2.set_title("B. Maximum burden comparison", loc="left", fontweight="bold")
ax2.legend(frameon=False, loc="lower right")
ax2.grid(axis="x", alpha=0.15)

fig.suptitle("Figure 2. Species biology burden summary", y=0.99, fontweight="bold")
savefig_all(fig, "Figure02_Species_Biology_Burden_v2")

# -----------------------------
# FIGURE 3
# MLST - top STs for major species
# -----------------------------
mlst_by_species = defaultdict(Counter)
for r in mlst_rows:
    sp = r.get("TopSpecies", "")
    st = r.get("Scheme_ST", "").strip()
    ct = as_int(r.get("Count", 0))
    if sp and st:
        mlst_by_species[sp][st] += ct

mlst_species = [s for s in species_order if len(mlst_by_species[s]) > 0][:6]

n = max(1, len(mlst_species))
ncols = 2
nrows = math.ceil(n / ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(17, 5.2 * nrows))
axes = np.array(axes).reshape(-1)

for ax, sp in zip(axes, mlst_species):
    labels, values = counter_top_other(mlst_by_species[sp], topn=10)
    labels = labels[::-1]
    values = values[::-1]
    cols = [cmap_blue(0.30 + 0.60*(i/max(len(values)-1,1))) for i in range(len(values))]
    ax.barh(range(len(values)), values, color=cols, edgecolor="white", linewidth=0.8)
    ax.set_yticks(range(len(values)))
    ax.set_yticklabels([wrap(x.replace(" | ", "\n"), 20) for x in labels])
    ax.set_title(species_short(sp), loc="left", fontweight="bold")
    ax.grid(axis="x", alpha=0.15)
    for i, v in enumerate(values):
        ax.text(v + 0.08, i, str(v), va="center", fontsize=9, fontweight="bold")

for k in range(len(mlst_species), len(axes)):
    axes[k].axis("off")

fig.suptitle("Figure 3. MLST population structure in major species", y=0.995, fontweight="bold")
fig.tight_layout()
savefig_all(fig, "Figure03_MLST_Structure_v2")

# -----------------------------
# FIGURE 4
# AMR class architecture
# -----------------------------
amr_class_by_species = defaultdict(Counter)
for r in amr_class_rows:
    sp = r.get("TopSpecies", "")
    cl = r.get("AMR_Class", "").strip()
    ct = as_int(r.get("Count", 0))
    if sp and cl:
        amr_class_by_species[sp][cl] += ct

top_species_amr = species_order[:5]
global_class_counter = Counter()
for sp in top_species_amr:
    global_class_counter.update(amr_class_by_species[sp])
top_classes = [k for k, _ in global_class_counter.most_common(12)]

amr_mat = np.zeros((len(top_species_amr), len(top_classes)), dtype=float)
for i, sp in enumerate(top_species_amr):
    for j, cl in enumerate(top_classes):
        amr_mat[i, j] = amr_class_by_species[sp][cl]

fig = plt.figure(figsize=(18, 10))
gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1.4, 1.0], hspace=0.35)

ax1 = fig.add_subplot(gs[0, 0])
im = ax1.imshow(np.log1p(amr_mat), aspect="auto", cmap=cmap_heat)
ax1.set_yticks(np.arange(len(top_species_amr)))
ax1.set_yticklabels([species_short(x) for x in top_species_amr])
ax1.set_xticks(np.arange(len(top_classes)))
ax1.set_xticklabels([wrap(x, 14) for x in top_classes], rotation=35, ha="right")
ax1.set_title("A. AMR class heatmap across major species", loc="left", fontweight="bold")
for i in range(amr_mat.shape[0]):
    for j in range(amr_mat.shape[1]):
        if amr_mat[i, j] > 0:
            ax1.text(j, i, str(int(amr_mat[i, j])),
                     ha="center", va="center", fontsize=9, fontweight="bold",
                     color=("white" if np.log1p(amr_mat[i, j]) > np.nanmax(np.log1p(amr_mat))*0.45 else "black"))
cbar = fig.colorbar(im, ax=ax1, fraction=0.025, pad=0.015)
cbar.set_label("log(1 + count)")

ax2 = fig.add_subplot(gs[1, 0])
x = np.arange(len(top_species_amr))
bottom = np.zeros(len(top_species_amr))
stack_classes = top_classes[:8]
stack_cols = [cmap_heat(0.25 + 0.65*(i/max(len(stack_classes)-1,1))) for i in range(len(stack_classes))]
for c, col in zip(stack_classes, stack_cols):
    vals = np.array([amr_class_by_species[sp][c] for sp in top_species_amr], dtype=float)
    ax2.bar(x, vals, bottom=bottom, color=col, edgecolor="white", linewidth=0.6, label=c)
    bottom += vals
ax2.set_xticks(x)
ax2.set_xticklabels([species_short(x) for x in top_species_amr], rotation=20, ha="right")
ax2.set_ylabel("Count")
ax2.set_title("B. Stacked burden of dominant AMR classes", loc="left", fontweight="bold")
ax2.legend(frameon=False, ncol=4, bbox_to_anchor=(0.5, -0.28), loc="upper center")
ax2.grid(axis="y", alpha=0.15)

fig.suptitle("Figure 4. Antimicrobial resistance class landscape", y=0.99, fontweight="bold")
savefig_all(fig, "Figure04_AMR_Class_Architecture_v2")

# -----------------------------
# FIGURE 5
# AMR gene signature
# -----------------------------
amr_gene_by_species = defaultdict(Counter)
gene_key = None
for c in ["AMR_Gene", "Gene", "AMR_TopGene"]:
    if amr_gene_rows and c in amr_gene_rows[0]:
        gene_key = c
        break
if gene_key is None:
    gene_key = list(amr_gene_rows[0].keys())[1]

for r in amr_gene_rows:
    sp = r.get("TopSpecies", "")
    gn = r.get(gene_key, "").strip()
    ct = as_int(r.get("Count", 0))
    if sp and gn:
        amr_gene_by_species[sp][gn] += ct

top_species_gene = species_order[:5]
global_gene_counter = Counter()
for sp in top_species_gene:
    global_gene_counter.update(amr_gene_by_species[sp])
top_genes = [k for k, _ in global_gene_counter.most_common(15)]

gene_mat = np.zeros((len(top_species_gene), len(top_genes)), dtype=float)
for i, sp in enumerate(top_species_gene):
    for j, g in enumerate(top_genes):
        gene_mat[i, j] = amr_gene_by_species[sp][g]

fig = plt.figure(figsize=(18, 10))
gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1.35, 1.0], hspace=0.36)

ax1 = fig.add_subplot(gs[0, 0])
im = ax1.imshow(np.log1p(gene_mat), aspect="auto", cmap=cmap_purple)
ax1.set_yticks(np.arange(len(top_species_gene)))
ax1.set_yticklabels([species_short(x) for x in top_species_gene])
ax1.set_xticks(np.arange(len(top_genes)))
ax1.set_xticklabels([wrap(x, 14) for x in top_genes], rotation=35, ha="right")
ax1.set_title("A. Top AMR gene heatmap", loc="left", fontweight="bold")
for i in range(gene_mat.shape[0]):
    for j in range(gene_mat.shape[1]):
        if gene_mat[i, j] > 0:
            ax1.text(j, i, str(int(gene_mat[i, j])),
                     ha="center", va="center", fontsize=8.5, fontweight="bold",
                     color=("white" if np.log1p(gene_mat[i, j]) > np.nanmax(np.log1p(gene_mat))*0.45 else "black"))
cbar = fig.colorbar(im, ax=ax1, fraction=0.025, pad=0.015)
cbar.set_label("log(1 + count)")

ax2 = fig.add_subplot(gs[1, 0])
gene_totals = [(g, int(sum(gene_mat[:, j]))) for j, g in enumerate(top_genes)]
gene_totals = sorted(gene_totals, key=lambda x: x[1], reverse=True)[:12]
labs = [wrap(g, 18) for g, _ in gene_totals][::-1]
vals = [v for _, v in gene_totals][::-1]
cols = [cmap_purple(0.28 + 0.62*(i/max(len(vals)-1,1))) for i in range(len(vals))]
ax2.barh(range(len(vals)), vals, color=cols, edgecolor="white", linewidth=0.8)
ax2.set_yticks(range(len(vals)))
ax2.set_yticklabels(labs)
ax2.set_xlabel("Total count")
ax2.set_title("B. Dominant AMR genes across major species", loc="left", fontweight="bold")
ax2.grid(axis="x", alpha=0.15)
for i, v in enumerate(vals):
    ax2.text(v + 0.3, i, str(v), va="center", fontsize=9, fontweight="bold")

fig.suptitle("Figure 5. AMR gene signature landscape", y=0.99, fontweight="bold")
savefig_all(fig, "Figure05_AMR_Gene_Signatures_v2")

# -----------------------------
# FIGURE 6
# Virulence + plasmid
# -----------------------------
vfdb_by_species = defaultdict(Counter)
vf_key = None
for c in ["VFDB_Gene", "Gene"]:
    if vfdb_rows and c in vfdb_rows[0]:
        vf_key = c
        break
if vf_key is None:
    vf_key = list(vfdb_rows[0].keys())[1]

for r in vfdb_rows:
    sp = r.get("TopSpecies", "")
    gn = r.get(vf_key, "").strip()
    ct = as_int(r.get("Count", 0))
    if sp and gn:
        vfdb_by_species[sp][gn] += ct

plas_by_species = defaultdict(Counter)
pl_key = None
for c in ["Plasmid_Replicon", "Replicon", "Plasmid_Gene"]:
    if plasmid_rows and c in plasmid_rows[0]:
        pl_key = c
        break
if pl_key is None:
    pl_key = list(plasmid_rows[0].keys())[1]

for r in plasmid_rows:
    sp = r.get("TopSpecies", "")
    gn = r.get(pl_key, "").strip()
    ct = as_int(r.get("Count", 0))
    if sp and gn:
        plas_by_species[sp][gn] += ct

top_species_vp = species_order[:5]

vf_global = Counter()
for sp in top_species_vp:
    vf_global.update(vfdb_by_species[sp])
top_vf = [k for k, _ in vf_global.most_common(12)]

pl_global = Counter()
for sp in top_species_vp:
    pl_global.update(plas_by_species[sp])
top_pl = [k for k, _ in pl_global.most_common(10)]

vf_mat = np.zeros((len(top_species_vp), len(top_vf)), dtype=float)
for i, sp in enumerate(top_species_vp):
    for j, g in enumerate(top_vf):
        vf_mat[i, j] = vfdb_by_species[sp][g]

pl_mat = np.zeros((len(top_species_vp), len(top_pl)), dtype=float)
for i, sp in enumerate(top_species_vp):
    for j, g in enumerate(top_pl):
        pl_mat[i, j] = plas_by_species[sp][g]

fig = plt.figure(figsize=(19, 12))
gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.28, hspace=0.34)

# A virulence heatmap
axA = fig.add_subplot(gs[0, 0])
im1 = axA.imshow(np.log1p(vf_mat), aspect="auto", cmap=cmap_green)
axA.set_yticks(np.arange(len(top_species_vp)))
axA.set_yticklabels([species_short(x) for x in top_species_vp])
axA.set_xticks(np.arange(len(top_vf)))
axA.set_xticklabels([wrap(x, 14) for x in top_vf], rotation=35, ha="right")
axA.set_title("A. Virulence gene heatmap", loc="left", fontweight="bold")
for i in range(vf_mat.shape[0]):
    for j in range(vf_mat.shape[1]):
        if vf_mat[i, j] > 0:
            axA.text(j, i, str(int(vf_mat[i, j])),
                     ha="center", va="center", fontsize=8.3, fontweight="bold",
                     color=("white" if np.log1p(vf_mat[i, j]) > np.nanmax(np.log1p(vf_mat))*0.45 else "black"))
cbar1 = fig.colorbar(im1, ax=axA, fraction=0.035, pad=0.02)
cbar1.set_label("log(1 + count)")

# B plasmid heatmap
axB = fig.add_subplot(gs[0, 1])
if len(top_pl) > 0:
    im2 = axB.imshow(np.log1p(pl_mat), aspect="auto", cmap=cmap_blue)
    axB.set_yticks(np.arange(len(top_species_vp)))
    axB.set_yticklabels([species_short(x) for x in top_species_vp])
    axB.set_xticks(np.arange(len(top_pl)))
    axB.set_xticklabels([wrap(x, 14) for x in top_pl], rotation=35, ha="right")
    axB.set_title("B. Plasmid replicon heatmap", loc="left", fontweight="bold")
    for i in range(pl_mat.shape[0]):
        for j in range(pl_mat.shape[1]):
            if pl_mat[i, j] > 0:
                axB.text(j, i, str(int(pl_mat[i, j])),
                         ha="center", va="center", fontsize=8.3, fontweight="bold",
                         color=("white" if np.log1p(pl_mat[i, j]) > np.nanmax(np.log1p(pl_mat))*0.45 else "black"))
    cbar2 = fig.colorbar(im2, ax=axB, fraction=0.035, pad=0.02)
    cbar2.set_label("log(1 + count)")
else:
    axB.axis("off")
    axB.text(0.5, 0.5, "No informative plasmid replicon panel available", ha="center", va="center", fontsize=13, fontweight="bold")

# C samples with virulence hits
axC = fig.add_subplot(gs[1, 0])
vf_samples = [as_int(bio_map[s].get("Samples_with_VFDB_hits", 0)) for s in species_order if s in bio_map]
vf_species = [s for s in species_order if s in bio_map]
colsC = [cmap_green(0.30 + 0.60*(i/max(len(vf_species)-1,1))) for i in range(len(vf_species))]
axC.barh(np.arange(len(vf_species)), vf_samples, color=colsC, edgecolor="white", linewidth=0.8)
axC.set_yticks(np.arange(len(vf_species)))
axC.set_yticklabels([species_short(s) for s in vf_species])
axC.invert_yaxis()
axC.set_xlabel("Samples")
axC.set_title("C. Samples carrying virulence features", loc="left", fontweight="bold")
axC.grid(axis="x", alpha=0.15)
for i, v in enumerate(vf_samples):
    axC.text(v + 0.4, i, str(v), va="center", fontsize=9, fontweight="bold")

# D samples with plasmid hits
axD = fig.add_subplot(gs[1, 1])
pl_samples = [as_int(bio_map[s].get("Samples_with_plasmid_hits", 0)) for s in species_order if s in bio_map]
pl_species = [s for s in species_order if s in bio_map]
colsD = [cmap_blue(0.30 + 0.60*(i/max(len(pl_species)-1,1))) for i in range(len(pl_species))]
axD.barh(np.arange(len(pl_species)), pl_samples, color=colsD, edgecolor="white", linewidth=0.8)
axD.set_yticks(np.arange(len(pl_species)))
axD.set_yticklabels([species_short(s) for s in pl_species])
axD.invert_yaxis()
axD.set_xlabel("Samples")
axD.set_title("D. Samples carrying plasmid features", loc="left", fontweight="bold")
axD.grid(axis="x", alpha=0.15)
for i, v in enumerate(pl_samples):
    axD.text(v + 0.4, i, str(v), va="center", fontsize=9, fontweight="bold")

fig.suptitle("Figure 6. Virulence and plasmid feature architecture", y=0.99, fontweight="bold")
savefig_all(fig, "Figure06_Virulence_Plasmid_Features_v2")

# -----------------------------
# SUPP FIGURE 1
# Serotype distribution (supplementary)
# -----------------------------
if serotype_rows:
    fig, ax = plt.subplots(figsize=(10, 6))
    labs = [wrap(r.get("Serotype", "NA"), 16) for r in serotype_rows]
    vals = [as_int(r.get("Count", 0)) for r in serotype_rows]
    cols = [cmap_gold(0.30 + 0.60*(i/max(len(vals)-1,1))) for i in range(len(vals))]
    ax.bar(range(len(vals)), vals, color=cols, edgecolor="white", linewidth=0.8)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labs, rotation=30, ha="right")
    ax.set_ylabel("Samples")
    ax.set_title("Supplementary Figure 1. Informative serotype distribution", fontweight="bold")
    ax.grid(axis="y", alpha=0.15)
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals)*0.03 if max(vals) > 0 else 0.1, str(v), ha="center", va="bottom", fontweight="bold")
    savefig_all(fig, "SupplementaryFigure01_Serotype_Distribution_v2")

# -----------------------------
# README
# -----------------------------
with open(os.path.join(OUTDIR, "README.txt"), "w", encoding="utf-8") as f:
    f.write("Premium G2v2 manuscript figure pack\n")
    f.write(f"Input: {INDIR}\n")
    f.write(f"Output: {OUTDIR}\n")
    f.write("Generated files:\n")
    for fn in sorted(os.listdir(OUTDIR)):
        f.write(f"- {fn}\n")

print("G2v2 figure pack written to:", OUTDIR)
for fn in sorted(os.listdir(OUTDIR)):
    print(fn)
