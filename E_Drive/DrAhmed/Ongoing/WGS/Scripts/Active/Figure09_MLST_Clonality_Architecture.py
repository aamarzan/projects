import os
import re
import math
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

# =========================
# Global styling
# =========================
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "axes.titlesize": 18,
    "axes.titleweight": "bold",
    "axes.labelsize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.titlesize": 24,
    "figure.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#555555",
    "axes.linewidth": 0.9,
    "grid.color": "#d9dde3",
    "grid.linewidth": 0.85,
    "grid.alpha": 1.0,
    "savefig.dpi": 500,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# =========================
# Paths
# =========================
WORK = "/mnt/e/DrAhmed/Ongoing/WGS/Result"
MLST_CSV = f"{WORK}/_G3/clean/mlst_species_st_counts_clean_166.csv"
MASTER_CSV = f"{WORK}/_BIOLOGY_LAYER_V2/PrimaryResults_v4_withBiology_166.csv"
OUTDIR = f"{WORK}/_G4_REMAINING/output/main"
os.makedirs(OUTDIR, exist_ok=True)

OUTNAME = "Figure09_MLST_Clonality_Architecture"

# =========================
# Helper functions
# =========================
def save_png_pdf(fig, outdir, name):
    png = os.path.join(outdir, f"{name}.png")
    pdf = os.path.join(outdir, f"{name}.pdf")
    fig.savefig(png, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    print("Saved PNG:", png)
    print("Saved PDF:", pdf)

def first_existing(df, candidates, required=True):
    cols = list(df.columns)

    # exact match first
    for cand in candidates:
        for col in cols:
            if col.lower() == cand.lower():
                return col

    # contains match
    for cand in candidates:
        for col in cols:
            if cand.lower() in col.lower():
                return col

    if required:
        raise KeyError(
            f"None of the candidate columns were found.\n"
            f"Candidates: {candidates}\nAvailable: {list(df.columns)}"
        )
    return None

def to_num(s, default=np.nan):
    try:
        if pd.isna(s):
            return default
        return float(str(s).strip())
    except Exception:
        return default

def species_short(name):
    name = str(name).strip()
    parts = name.split()
    if len(parts) >= 2:
        return f"{parts[0][0]}. {' '.join(parts[1:])}"
    return name

def clean_text(s):
    if pd.isna(s):
        return ""
    return str(s).strip()

def normalize_st(x):
    """
    Convert variants like:
      'serratia | ST362'
      'ST362'
      '362'
      'escherichia_2_ST2'
    into a readable ST label like 'ST362', 'ST2', etc.
    """
    s = clean_text(x)
    if not s:
        return np.nan

    if "|" in s:
        s = s.split("|")[-1].strip()

    # standard ST pattern
    m = re.search(r'(?i)\bST[\s\-_]*([A-Za-z0-9]+)\b', s)
    if m:
        return f"ST{m.group(1)}"

    # trailing _STxx style
    m = re.search(r'(?i)_ST[\s\-_]*([A-Za-z0-9]+)', s)
    if m:
        return f"ST{m.group(1)}"

    # purely numeric
    if re.fullmatch(r"\d+", s):
        return f"ST{s}"

    return s

def nice_count_label(v):
    try:
        return f"{int(round(v))}"
    except Exception:
        return str(v)

def scale_bubble_size(counts, min_size=180, max_size=2300):
    counts = np.asarray(counts, dtype=float)
    if len(counts) == 0:
        return counts
    cmin = np.nanmin(counts)
    cmax = np.nanmax(counts)
    if cmax == cmin:
        return np.full_like(counts, (min_size + max_size) / 2.0)
    return min_size + (counts - cmin) * (max_size - min_size) / (cmax - cmin)

def draw_violin_panel(ax, df, species_order, value_col, cmap, title):
    """
    Horizontal violin plot with overlaid points, median marker, and IQR band.
    """
    rng = np.random.default_rng(42)

    plot_data = []
    med_map = {}
    n_map = {}

    for sp in species_order:
        vals = df.loc[df["Species"] == sp, value_col].dropna().astype(float).values
        if len(vals) > 0:
            plot_data.append(vals)
            med_map[sp] = float(np.median(vals))
            n_map[sp] = len(vals)
        else:
            plot_data.append(np.array([np.nan]))
            med_map[sp] = np.nan
            n_map[sp] = 0

    medians = [med_map[s] for s in species_order]
    valid_meds = [m for m in medians if not np.isnan(m)]
    if len(valid_meds) == 0:
        valid_meds = [0, 1]

    norm = Normalize(vmin=min(valid_meds), vmax=max(valid_meds))

    positions = np.arange(len(species_order), 0, -1)

    vp = ax.violinplot(
        dataset=plot_data,
        positions=positions,
        vert=False,
        widths=0.82,
        showmeans=False,
        showmedians=False,
        showextrema=False
    )

    for body, sp in zip(vp["bodies"], species_order):
        color = cmap(norm(med_map[sp])) if not np.isnan(med_map[sp]) else "#d1d5db"
        body.set_facecolor(color)
        body.set_edgecolor("#6b7280")
        body.set_alpha(0.78)
        body.set_linewidth(0.9)

    # overlay raw points and median/IQR
    xmax = 0
    for pos, sp in zip(positions, species_order):
        vals = df.loc[df["Species"] == sp, value_col].dropna().astype(float).values
        if len(vals) == 0:
            continue

        xmax = max(xmax, float(np.nanmax(vals)))

        jitter = rng.normal(0, 0.045, size=len(vals))
        y = pos + jitter

        ax.scatter(
            vals, y,
            s=30,
            color=cmap(norm(np.median(vals))),
            edgecolor="white",
            linewidth=0.55,
            alpha=0.90,
            zorder=3
        )

        q1, med, q3 = np.percentile(vals, [25, 50, 75])

        ax.plot([q1, q3], [pos, pos], color="#374151", lw=2.0, zorder=4)
        ax.scatter([med], [pos], marker="D", s=55, color="#111827",
                   edgecolor="white", linewidth=0.55, zorder=5)

        ax.text(
            xmax * 1.03 if xmax > 0 else med + 0.2, pos,
            f"median={med:.1f}",
            va="center", ha="left",
            fontsize=11.5, color="#374151"
        )

    ax.set_yticks(positions)
    ax.set_yticklabels([f"{species_short(s)} (n={n_map[s]})" for s in species_order])
    ax.set_xlabel("VFDB hits per sample")
    ax.set_ylabel("Dominant species")
    ax.set_title(title, loc="left", pad=12)

    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(left=0, right=(xmax * 1.24 if xmax > 0 else 1))

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.024, pad=0.02)
    cbar.set_label("Species median VFDB hit count")
    cbar.ax.tick_params(labelsize=11)

# =========================
# Read input tables
# =========================
mlst = pd.read_csv(MLST_CSV)
master = pd.read_csv(MASTER_CSV)

# =========================
# Detect columns in master
# =========================
species_col = first_existing(master, [
    "TopSpecies", "Species", "top_species"
])

sample_col = first_existing(master, [
    "Sample", "Sample_ID", "SampleID", "sample", "sample_id"
], required=False)

st_col = first_existing(master, [
    "Scheme_ST", "MLST_ST", "ST", "MLST", "mlst", "mlst_st"
], required=False)

amr_col = first_existing(master, [
    "AMR_Genes_n", "AMR_Gene_Count", "AMR_Genes",
    "AMR gene count", "AMR", "ResFinder_Genes_n"
])

vfdb_col = first_existing(master, [
    "VFDB_Hits_n", "VFDB_Hits", "VFDB hits",
    "Virulence_Genes_n", "VFDB_gene_count", "vfdb"
])

# =========================
# Clean master
# =========================
master["Species"] = master[species_col].astype(str).str.strip()
master["AMR"] = master[amr_col].apply(to_num)
master["VFDB"] = master[vfdb_col].apply(to_num)

if st_col is not None:
    master["ST_norm"] = master[st_col].apply(normalize_st)
else:
    master["ST_norm"] = np.nan

# dominant species: at least 4 samples
species_n = master["Species"].value_counts()
dominant_species = list(species_n[species_n >= 4].index)

# =========================
# Clean MLST table
# =========================
mlst["Species"] = mlst["TopSpecies"].astype(str).str.strip()
mlst["Count"] = pd.to_numeric(mlst["Count"], errors="coerce").fillna(0)
mlst["ST_norm"] = mlst["Scheme_ST"].apply(normalize_st)
mlst = mlst[mlst["Species"].isin(dominant_species)].copy()

# keep only meaningful rows
mlst = mlst[(mlst["Count"] > 0) & (~mlst["ST_norm"].isna())].copy()

# =========================
# Clone-level AMR/VFDB medians from master
# =========================
clone_amr = (
    master.dropna(subset=["Species"])
    .groupby(["Species", "ST_norm"], dropna=False)["AMR"]
    .median()
    .reset_index()
    .rename(columns={"AMR": "Median_AMR"})
)

clone_vfdb = (
    master.dropna(subset=["Species"])
    .groupby(["Species", "ST_norm"], dropna=False)["VFDB"]
    .median()
    .reset_index()
    .rename(columns={"VFDB": "Median_VFDB"})
)

species_amr_fallback = master.groupby("Species")["AMR"].median().to_dict()
species_vfdb_fallback = master.groupby("Species")["VFDB"].median().to_dict()

clone = (
    mlst.merge(clone_amr, on=["Species", "ST_norm"], how="left")
        .merge(clone_vfdb, on=["Species", "ST_norm"], how="left")
)

clone["Median_AMR"] = clone.apply(
    lambda r: species_amr_fallback.get(r["Species"], np.nan)
    if pd.isna(r["Median_AMR"]) else r["Median_AMR"], axis=1
)

clone["Median_VFDB"] = clone.apply(
    lambda r: species_vfdb_fallback.get(r["Species"], np.nan)
    if pd.isna(r["Median_VFDB"]) else r["Median_VFDB"], axis=1
)

# rank top clonotypes
TOP_N = 18
clone = clone.sort_values(
    by=["Count", "Median_AMR", "Median_VFDB"],
    ascending=[False, False, False]
).head(TOP_N).copy()

clone["DisplayLabel"] = clone.apply(
    lambda r: f"{species_short(r['Species'])}\n{r['ST_norm']}",
    axis=1
)

# =========================
# Species order for violin
# =========================
vf_species_df = master[master["Species"].isin(dominant_species)].copy()
vf_summary = (
    vf_species_df.groupby("Species")
    .agg(
        n=("Species", "size"),
        Median_VFDB=("VFDB", "median"),
        Median_AMR=("AMR", "median")
    )
    .reset_index()
)

vf_summary = vf_summary[vf_summary["n"] >= 4].copy()
vf_summary = vf_summary.sort_values(
    by=["Median_VFDB", "n", "Median_AMR"],
    ascending=[False, False, False]
)

species_order_violin = list(vf_summary["Species"])

# =========================
# Colormaps
# =========================
amr_cmap = LinearSegmentedColormap.from_list(
    "amrblue_premium",
    ["#eff6ff", "#bfdbfe", "#60a5fa", "#2563eb", "#123b82"]
)

vfdb_cmap = LinearSegmentedColormap.from_list(
    "vfdb_premium",
    ["#fff7ed", "#fbcfe8", "#f472b6", "#c026d3", "#6b21a8"]
)

# =========================
# Build figure
# =========================
fig = plt.figure(figsize=(18.5, 13.5), facecolor="white")
gs = GridSpec(
    2, 1,
    height_ratios=[1.15, 1.0],
    hspace=0.42,
    figure=fig
)

axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[1, 0])

# -------------------------
# Panel A: bubble plot
# -------------------------
x = np.arange(len(clone))
y = np.zeros(len(clone))

sizes = scale_bubble_size(clone["Count"].values, min_size=180, max_size=2300)

amr_vals = clone["Median_AMR"].astype(float).values
valid_amr = amr_vals[~np.isnan(amr_vals)]
if len(valid_amr) == 0:
    valid_amr = np.array([0, 1])

normA = Normalize(vmin=float(np.nanmin(valid_amr)), vmax=float(np.nanmax(valid_amr)))

axA.scatter(
    x, y,
    s=sizes,
    c=clone["Median_AMR"],
    cmap=amr_cmap,
    norm=normA,
    alpha=0.94,
    edgecolor="#334155",
    linewidth=0.7,
    zorder=3
)

# vertical guide lines
for xi in x:
    axA.axvline(x=xi, ymin=0.10, ymax=0.90, color="#d1d5db", lw=0.9, zorder=0)

axA.axhline(0, color="#9ca3af", lw=1.05, zorder=1)

# count annotations
for xi, cnt in zip(x, clone["Count"].values):
    axA.text(
        xi, 0.17,
        nice_count_label(cnt),
        ha="center", va="center",
        fontsize=13,
        fontweight="bold",
        color="#1f2937",
        clip_on=False
    )

axA.set_xticks(x)
axA.set_xticklabels(clone["DisplayLabel"], rotation=32, ha="right")
axA.set_yticks([])
axA.set_xlim(-0.85, len(clone) - 0.15)
axA.set_ylim(-0.78, 0.95)
axA.set_xlabel("Dominant clonotypes (species + ST)")
axA.set_title(
    "A. Dominant clonotypes sized by frequency and colored by median AMR burden",
    loc="left", pad=10
)

# cleaner spines
axA.spines["left"].set_visible(False)
axA.grid(False)

# colorbar for Panel A
smA = ScalarMappable(norm=normA, cmap=amr_cmap)
smA.set_array([])
cbarA = plt.colorbar(smA, ax=axA, fraction=0.022, pad=0.02)
cbarA.set_label("Median AMR gene count")
cbarA.ax.tick_params(labelsize=11)

# ---------- Panel A: clearer clone-frequency size legend ----------
size_legend_vals = [1, 3, 11, 15]

def _legend_marker_size(n):
    # Scatter 's' is area in pt^2, but legend markersize is diameter in pt.
    # So we create a wider visual spread here for readability.
    area = np.interp(n, [min(size_legend_vals), max(size_legend_vals)], [120, 1050])
    return np.sqrt(area)

size_handles = [
    Line2D(
        [0], [0],
        marker='o',
        linestyle='',
        markersize=_legend_marker_size(v),
        markerfacecolor='#8db9e8',
        markeredgecolor='#4b5563',
        markeredgewidth=0.8,
        label=f"n={v}"
    )
    for v in size_legend_vals
]

leg = axA.legend(
    handles=size_handles,
    title="Clone frequency",
    frameon=False,
    loc="upper right",
    bbox_to_anchor=(0.985, 1.17),
    ncol=2,
    columnspacing=1.8,
    handletextpad=0.8,
    labelspacing=1.0,
    borderaxespad=0.0
)

plt.setp(leg.get_title(), fontsize=14, fontweight="bold")
for txt in leg.get_texts():
    txt.set_fontsize(12.5)
    
# -------------------------
# Panel B: violin plot
# -------------------------
draw_violin_panel(
    ax=axB,
    df=vf_species_df,
    species_order=species_order_violin,
    value_col="VFDB",
    cmap=vfdb_cmap,
    title="B. Virulence-burden distributions across dominant MLST-bearing species"
)

# -------------------------
# Global title and footnote
# -------------------------
fig.suptitle(
    "MLST-defined clonality architecture across dominant species",
    y=0.985
)

# summary text
n_clones = len(clone)
n_species = len(set(clone["Species"]))
n_violin_species = len(species_order_violin)


plt.subplots_adjust(
    top=0.90,
    bottom=0.08,
    left=0.07,
    right=0.92
)

save_png_pdf(fig, OUTDIR, OUTNAME)
plt.close(fig)

print("Saved Figure 9 to:", OUTDIR)