import os
import math
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np

from figure_helper_common import (
    setup_rcparams, read_csv, as_float, as_int, species_short,
    style_ax, palette_list, save_png_pdf
)

setup_rcparams()

WORK = "/mnt/e/DrAhmed/Ongoing/WGS/Result"
INDIR = f"{WORK}/_G3/clean"
OUTDIR = f"{WORK}/_G3/output/main"
os.makedirs(OUTDIR, exist_ok=True)

# -----------------------------
# Load data
# -----------------------------
feature_prev = read_csv(f"{INDIR}/species_feature_prevalence_from_master_166.csv")
plasmid_rows = read_csv(f"{INDIR}/plasmid_replicon_by_species_166.csv")
sero_rows = read_csv(f"{INDIR}/serotype_distribution_166.csv")

# -----------------------------
# Helpers
# -----------------------------
def short_replicon_label(x):
    x = str(x).strip()
    replacements = {
        "IncFIB(K)_1_Kpn3": "IncFIB(K)",
        "IncFII_1_pKP91": "IncFII",
        "IncA/C2_1": "IncA/C2",
        "Col440I_1": "Col440I",
        "IncL/M(pMU407)_1_pMU407": "IncL/M",
        "ColRNAI_1": "ColRNAI",
        "Col440II_1": "Col440II",
        "FII(pBK30683)_1": "FII(pBK30683)",
        "pSM22_1": "pSM22",
        "pADAP_1": "pADAP",
        "IncFIB(pQil)_1_pQil": "IncFIB(pQil)",
        "IncFII(pCRY)_1_pCRY": "IncFII(pCRY)",
        "IncP6_1": "IncP6",
        "IncQ2_1": "IncQ2",
    }
    return replacements.get(x, x.replace("_1", "").replace("_", " "))

def parse_serotype_label(sample, call):
    txt = str(call).strip()
    parts = txt.split()
    if len(parts) >= 3:
        return f"{sample}   {parts[-2]}\n{parts[-1]}"
    elif len(parts) >= 2:
        return f"{sample}\n{parts[-1]}"
    else:
        return f"{sample}\n{txt}"

# -----------------------------
# Species order
# -----------------------------
species_order = [
    "Serratia marcescens",
    "Acinetobacter baumannii",
    "Klebsiella pneumoniae",
    "Pseudomonas aeruginosa",
    "Escherichia coli",
    "Serratia nevei",
    "Homo sapiens",
]

species_display = {
    "Serratia marcescens": "S. marcescens",
    "Acinetobacter baumannii": "A. baumannii",
    "Klebsiella pneumoniae": "K. pneumoniae",
    "Pseudomonas aeruginosa": "P. aeruginosa",
    "Escherichia coli": "E. coli",
    "Serratia nevei": "S. nevei",
    "Homo sapiens": "H. sapiens",
}

# -----------------------------
# Panel A data
# -----------------------------
prev_map = {r["TopSpecies"]: r for r in feature_prev}
panelA_species = [s for s in species_order if s in prev_map]

vfdb_pct = []
plasmid_pct = []
panelA_labels = []

for s in panelA_species:
    r = prev_map[s]
    n = max(as_int(r.get("Samples_n", 0)), 1)
    vf = 100.0 * as_int(r.get("Samples_with_VFDB_hits", 0)) / n
    pl = 100.0 * as_int(r.get("Samples_with_Plasmid_hits", 0)) / n
    vfdb_pct.append(vf)
    plasmid_pct.append(pl)
    panelA_labels.append(species_display.get(s, s))

# -----------------------------
# Panel B data
# Keep top plasmid replicons by total count
# -----------------------------
from collections import defaultdict
mat = defaultdict(dict)
repl_totals = defaultdict(int)

for r in plasmid_rows:
    sp = r["TopSpecies"]
    repl = r["Plasmid_Replicon"]
    cnt = as_int(r["Count"], 0)
    mat[sp][repl] = cnt
    repl_totals[repl] += cnt

top_repl = [k for k, v in sorted(repl_totals.items(), key=lambda x: (-x[1], x[0]))[:10]]
panelB_species = [s for s in species_order if s in mat]

heat = np.zeros((len(panelB_species), len(top_repl)), dtype=float)
annot = np.empty((len(panelB_species), len(top_repl)), dtype=object)

for i, sp in enumerate(panelB_species):
    for j, repl in enumerate(top_repl):
        val = mat.get(sp, {}).get(repl, 0)
        heat[i, j] = math.log10(1 + val) if val > 0 else 0
        annot[i, j] = str(val) if val > 0 else ""

top_repl_labels = [short_replicon_label(x) for x in top_repl]

# -----------------------------
# Panel C data
# -----------------------------
panelC_labels = []
panelC_vals = []
for r in sero_rows:
    sample = r["Sample"]
    call = r["Serotype_Call"]
    panelC_labels.append(parse_serotype_label(sample, call))
    panelC_vals.append(1)

# -----------------------------
# Figure layout
# -----------------------------
fig = plt.figure(figsize=(18, 12))

# Outer layout
outer = GridSpec(
    1, 2, figure=fig,
    width_ratios=[1.0, 1.22],
    wspace=0.28
)

# Left side = Panel A
axA = fig.add_subplot(outer[0, 0])

# Right side = nested grid for B and C
right = outer[0, 1].subgridspec(
    2, 1,
    height_ratios=[1.08, 0.92],
    hspace=0.42
)
axB = fig.add_subplot(right[0, 0])
axC = fig.add_subplot(right[1, 0])

# -----------------------------
# Panel A
# -----------------------------
y = np.arange(len(panelA_labels))
bar_h = 0.36

axA.barh(
    y - bar_h/2, vfdb_pct,
    height=bar_h,
    color="#c43d75",
    edgecolor="#8f2753",
    linewidth=0.8,
    label="VFDB-positive samples",
    zorder=3
)
axA.barh(
    y + bar_h/2, plasmid_pct,
    height=bar_h,
    color="#6b4fd3",
    edgecolor="#4930a6",
    linewidth=0.8,
    label="Plasmid-positive samples",
    zorder=3
)

axA.set_yticks(y)
axA.set_yticklabels(panelA_labels, fontsize=12)
axA.invert_yaxis()
axA.set_xlim(0, 105)
axA.set_xlabel("Samples with feature (%)", fontsize=13)
axA.set_title("A. Prevalence of biological feature carriage", loc="left", pad=12, fontsize=17, fontweight="bold")
axA.grid(axis="x", alpha=0.25, zorder=0)
axA.legend(frameon=False, fontsize=11, loc="lower right")
for spine in ["top", "right"]:
    axA.spines[spine].set_visible(False)

# -----------------------------
# Panel B
# -----------------------------
im = axB.imshow(heat, aspect="auto", cmap="Purples")

axB.set_yticks(np.arange(len(panelB_species)))
axB.set_yticklabels([species_display.get(s, s) for s in panelB_species], fontsize=12)

axB.set_xticks(np.arange(len(top_repl)))
axB.set_xticklabels(top_repl_labels, rotation=32, ha="right", fontsize=11)
axB.tick_params(axis="x", pad=6)

for i in range(len(panelB_species)):
    for j in range(len(top_repl)):
        if annot[i, j]:
            axB.text(j, i, annot[i, j], ha="center", va="center", fontsize=10, color="white")

axB.set_title("B. Dominant plasmid replicon architecture", loc="left", pad=12, fontsize=17, fontweight="bold")
for spine in ["top", "right"]:
    axB.spines[spine].set_visible(False)

cbar = fig.colorbar(im, ax=axB, fraction=0.035, pad=0.02)
cbar.set_label("log(1 + count)", fontsize=12)

# -----------------------------
# Panel C
# -----------------------------
yy = np.arange(len(panelC_labels))
colors = plt.cm.Greens(np.linspace(0.35, 0.82, len(panelC_labels)))

bars = axC.barh(
    yy, panelC_vals,
    color=colors,
    edgecolor="#2f6b2f",
    linewidth=0.8,
    zorder=3
)

axC.set_yticks(yy)
axC.set_yticklabels(panelC_labels, fontsize=12)
axC.invert_yaxis()
axC.set_xlim(0, 1.06)
axC.set_xlabel("Samples", fontsize=13)
axC.set_title("C. Informative serotype calls", loc="left", pad=14, fontsize=17, fontweight="bold")
axC.grid(axis="x", alpha=0.25, zorder=0)

for b in bars:
    axC.text(
        b.get_width() + 0.02,
        b.get_y() + b.get_height()/2,
        f"{int(b.get_width())}",
        va="center",
        ha="left",
        fontsize=11
    )

for spine in ["top", "right"]:
    axC.spines[spine].set_visible(False)

# -----------------------------
# Main title
# -----------------------------
fig.suptitle(
    "Virulence, plasmid, and serotype architecture",
    y=0.97,
    fontsize=24,
    fontweight="bold"
)

fig.subplots_adjust(top=0.85, bottom=0.10, left=0.09, right=0.97)

save_png_pdf(fig, "Figure06_Virulence_Plasmid_Serotype_G3", OUTDIR)
print("Saved Figure 6 to:", OUTDIR)