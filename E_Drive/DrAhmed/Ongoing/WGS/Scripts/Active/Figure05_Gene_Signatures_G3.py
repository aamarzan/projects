# =========================================================
# Figure 5 — AMR–virulence–plasmid co-occurrence/context
# Premium G3 replacement script
# =========================================================

import os
import re
import glob
import math
import textwrap
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D
from matplotlib import patheffects as pe


# =========================================================
# Global style
# =========================================================
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11.5,
    "axes.titlesize": 15.8,
    "axes.titleweight": "bold",
    "axes.labelsize": 12.2,
    "xtick.labelsize": 9.6,
    "ytick.labelsize": 10.4,
    "figure.titlesize": 24,
    "figure.titleweight": "bold",
    "savefig.dpi": 700,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# =========================================================
# Paths
# =========================================================
WORK = "/mnt/e/DrAhmed/Ongoing/WGS/Result"
INDIR = f"{WORK}/_FIGURE_RAW_MATERIALS_V2"
OUTDIR = f"{WORK}/_G3/output/main"
os.makedirs(OUTDIR, exist_ok=True)

OUTNAME = "Figure05_AMR_Virulence_Plasmid_Context_G3"

MASTER_CANDIDATES = [
    f"{WORK}/_BIOLOGY_LAYER_V2/PrimaryResults_v4_withBiology_166.csv",
    f"{WORK}/_MANUSCRIPT_FINAL/PrimaryResults_v4_withBiology_166.csv",
    f"{INDIR}/PrimaryResults_v4_withBiology_166.csv",
    f"{INDIR}/sample_context_166.csv",
    f"{INDIR}/amr_virulence_plasmid_context_166.csv",
]

SPECIES_COUNTS_CANDIDATES = [
    f"{INDIR}/species_counts_166.csv",
]

AMR_LONG_CANDIDATES = [
    f"{INDIR}/amr_gene_by_species_top200_166.csv",
    f"{INDIR}/AMR_gene_by_species_top200_166.csv",
    f"{INDIR}/amrfinder_gene_by_species_top200_166.csv",
]

VF_LONG_CANDIDATES = [
    f"{INDIR}/virulence_gene_by_species_top200_166.csv",
    f"{INDIR}/vf_gene_by_species_top200_166.csv",
    f"{INDIR}/VFDB_gene_by_species_top200_166.csv",
]

PLASMID_LONG_CANDIDATES = [
    f"{INDIR}/plasmid_by_species_top200_166.csv",
    f"{INDIR}/plasmid_replicon_by_species_top200_166.csv",
    f"{INDIR}/incgroup_by_species_top200_166.csv",
    f"{INDIR}/plasmid_incgroup_by_species_top200_166.csv",
    f"{INDIR}/plasmid_replicon_counts_by_species_166.csv",
]

TORMES_PLASMID_DIRS = [
    f"{WORK}/Result copy/tormes_all+plasmid+serotype(Enterobacteriaceae)/plasmids",
    f"{WORK}/Result copy/tormes_all_samples/plasmids",
    f"{WORK}/result_copy/tormes_all+plasmid+serotype(Enterobacteriaceae)/plasmids",
    f"{WORK}/result_copy/tormes_all_samples/plasmids",
]


# =========================================================
# Color systems
# =========================================================
species_palette = {
    "Serratia marcescens": "#2563eb",
    "Acinetobacter baumannii": "#0f766e",
    "Klebsiella pneumoniae": "#8b5cf6",
    "Pseudomonas aeruginosa": "#d946ef",
    "Escherichia coli": "#f59e0b",
}

amr_cmap = LinearSegmentedColormap.from_list(
    "amr_blue", ["#f8fbff", "#dbeafe", "#93c5fd", "#2563eb", "#0f172a"]
)
vf_cmap = LinearSegmentedColormap.from_list(
    "vf_rose", ["#fff7fb", "#fce7f3", "#f9a8d4", "#db2777", "#831843"]
)
plasmid_cmap = LinearSegmentedColormap.from_list(
    "plasmid_gold", ["#fffdf5", "#fef3c7", "#fbbf24", "#b45309", "#451a03"]
)

context_colors = {
    "AMR + virulence + plasmid": "#111827",
    "AMR + plasmid": "#2563eb",
    "AMR + virulence": "#7c3aed",
    "AMR only": "#60a5fa",
    "virulence + plasmid": "#db2777",
    "virulence only": "#f9a8d4",
    "plasmid only": "#f59e0b",
    "no detected context": "#e5e7eb",
}


# =========================================================
# General helpers
# =========================================================
def first_existing(paths):
    for p in paths:
        if p and os.path.isfile(p):
            return p
    return None


def safe_read_table(path):
    if not path or not os.path.isfile(path):
        return pd.DataFrame()

    try:
        df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8", dtype=str)
    except Exception:
        try:
            df = pd.read_csv(path, sep="\t", encoding="utf-8", dtype=str)
        except Exception:
            df = pd.read_csv(path, encoding="utf-8", dtype=str)

    df.columns = [str(c).strip() for c in df.columns]
    return df.fillna("")


def as_num(x, default=0.0):
    try:
        s = str(x).strip()
        if s == "":
            return default
        return float(s)
    except Exception:
        return default


def guess_col(df, candidates, contains=None):
    if df is None or df.empty:
        return None

    cols = list(df.columns)
    lower = {c.lower(): c for c in cols}

    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]

    if contains:
        for c in cols:
            cl = c.lower()
            if all(tok.lower() in cl for tok in contains):
                return c

    return None


def clean_text(x):
    return str(x).strip()


def clean_feature_name(x):
    s = clean_text(x)
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" ;,|")
    if s.lower() in {"", "na", "n/a", "none", "nan", "-", ".", "not detected"}:
        return ""
    return s


def split_features(x):
    """
    Split common feature-list strings while preserving biologically meaningful slashes,
    e.g. yagV/ecpE should remain one feature.
    """
    s = clean_text(x)
    if s.lower() in {"", "na", "n/a", "none", "nan", "-", ".", "not detected"}:
        return []

    parts = re.split(r"[;,|]+", s)
    out = []
    for p in parts:
        p = clean_feature_name(p)
        if p:
            out.append(p)
    return list(dict.fromkeys(out))


def short_species(name):
    s = clean_text(name)
    parts = s.split()
    if len(parts) >= 2:
        return f"{parts[0][0]}. {' '.join(parts[1:])}"
    return s


def species_from_sample(sample):
    s = clean_text(sample)
    prefix = re.split(r"[-_]", s)[0].upper()

    if prefix == "SM":
        return "Serratia marcescens"
    if prefix == "AB":
        return "Acinetobacter baumannii"
    if prefix == "KP":
        return "Klebsiella pneumoniae"
    if prefix == "PA":
        return "Pseudomonas aeruginosa"
    if prefix == "EC":
        return "Escherichia coli"

    return ""


def wrap_label(s, width=16):
    return "\n".join(textwrap.wrap(str(s), width=width, break_long_words=False))


def add_panel_card(ax, fc="#ffffff", ec="#e5e7eb"):
    ax.set_facecolor(fc)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(ec)
        spine.set_linewidth(1.05)


def style_axis(ax, grid_axis="x"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#94a3b8")
    ax.spines["bottom"].set_color("#94a3b8")
    ax.tick_params(colors="#334155")

    if grid_axis in ("x", "both"):
        ax.grid(axis="x", color="#e5e7eb", lw=0.8, zorder=0)
    if grid_axis in ("y", "both"):
        ax.grid(axis="y", color="#eef2f7", lw=0.8, zorder=0)

    ax.set_axisbelow(True)


def save_png_pdf(fig, outdir, outname):
    png = os.path.join(outdir, outname + ".png")
    pdf = os.path.join(outdir, outname + ".pdf")
    fig.savefig(png, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    print("Saved:", png)
    print("Saved:", pdf)


# =========================================================
# Data preparation helpers
# =========================================================
def read_long_feature_table(path, species_candidates, feature_candidates, count_candidates):
    df = safe_read_table(path)
    if df.empty:
        return pd.DataFrame(columns=["species", "feature", "count"])

    sp_col = guess_col(df, species_candidates, contains=["species"])
    feat_col = guess_col(df, feature_candidates)
    count_col = guess_col(df, count_candidates, contains=["count"])

    if sp_col is None or feat_col is None:
        return pd.DataFrame(columns=["species", "feature", "count"])

    rows = []
    for _, r in df.iterrows():
        sp = clean_text(r.get(sp_col, ""))
        feats = split_features(r.get(feat_col, ""))
        if not feats:
            continue

        count = as_num(r.get(count_col, 1), 1.0) if count_col else 1.0

        for feat in feats:
            rows.append({
                "species": sp,
                "feature": feat,
                "count": count,
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["species", "feature", "count"])

    out = (
        out.groupby(["species", "feature"], as_index=False)["count"]
        .sum()
        .sort_values(["species", "count"], ascending=[True, False])
    )
    return out


def get_species_order(species_counts_df, master_df, min_count=4):
    order = []

    if not species_counts_df.empty:
        sp_col = guess_col(species_counts_df, ["TopSpecies", "TopSpecies1", "Species"])
        ct_col = guess_col(species_counts_df, ["Count", "N", "Samples", "n"])
        if sp_col:
            tmp = []
            for _, r in species_counts_df.iterrows():
                sp = clean_text(r.get(sp_col, ""))
                n = as_num(r.get(ct_col, 0), 0) if ct_col else 0
                if sp and n >= min_count:
                    tmp.append((sp, n))
            order = [sp for sp, _ in sorted(tmp, key=lambda x: (-x[1], x[0]))]

    if not order and not master_df.empty:
        sp_col = guess_col(master_df, ["TopSpecies", "TopSpecies1", "Species"], contains=["species"])
        if sp_col:
            counts = Counter(clean_text(x) for x in master_df[sp_col] if clean_text(x))
            order = [sp for sp, n in counts.most_common() if n >= min_count]

    return order


def top_features(long_df, top_n=14):
    if long_df.empty:
        return []

    totals = (
        long_df.groupby("feature")["count"]
        .sum()
        .sort_values(ascending=False)
    )

    return list(totals.head(top_n).index)


def matrix_from_long(long_df, species_order, feature_order, log1p=True):
    mat = np.zeros((len(species_order), len(feature_order)), dtype=float)

    sp_i = {sp: i for i, sp in enumerate(species_order)}
    ft_i = {f: i for i, f in enumerate(feature_order)}

    for _, r in long_df.iterrows():
        sp = r["species"]
        ft = r["feature"]
        if sp in sp_i and ft in ft_i:
            mat[sp_i[sp], ft_i[ft]] += as_num(r["count"], 0)

    if log1p:
        return np.log1p(mat)
    return mat


def draw_heatmap(ax, mat, species_order, features, cmap, title, cbar_label):
    add_panel_card(ax)

    if mat.size == 0 or len(species_order) == 0 or len(features) == 0:
        ax.axis("off")
        ax.text(
            0.5, 0.5,
            "Required source table not found\nor no eligible features detected",
            ha="center", va="center",
            fontsize=12.5, fontweight="bold", color="#64748b",
            transform=ax.transAxes
        )
        ax.set_title(title, loc="left", pad=12)
        return None

    im = ax.imshow(mat, aspect="auto", cmap=cmap, interpolation="nearest")

    ax.set_yticks(np.arange(len(species_order)))
    ax.set_yticklabels([short_species(s) for s in species_order], fontsize=9.8)

    ax.set_xticks(np.arange(len(features)))
    ax.set_xticklabels(
        [wrap_label(f, 13) for f in features],
        rotation=45,
        ha="right",
        va="top",
        fontsize=8.8
    )

    ax.set_title(title, loc="left", pad=12)
    ax.tick_params(axis="both", length=0)

    # Thin white cell boundaries
    ax.set_xticks(np.arange(-0.5, len(features), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(species_order), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.15)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Highlight row labels with species colors
    for tick, sp in zip(ax.get_yticklabels(), species_order):
        tick.set_color(species_palette.get(sp, "#334155"))
        tick.set_fontweight("bold")

    cb = plt.colorbar(im, ax=ax, fraction=0.028, pad=0.018)
    cb.set_label(cbar_label, fontsize=9.8)
    cb.ax.tick_params(labelsize=8.8)
    cb.outline.set_visible(False)

    return im


def extract_inc_tokens_from_text(text):
    s = str(text)

    # Common plasmid replicon / incompatibility token patterns.
    tokens = re.findall(
        r"\b(?:Inc[A-Za-z0-9_.:/()+-]+|Col[A-Za-z0-9_.:/()+-]+|rep[A-Za-z0-9_.:/()+-]+)\b",
        s
    )

    cleaned = []
    for t in tokens:
        t = t.strip(" ;,|")
        if len(t) >= 3:
            cleaned.append(t)

    return list(dict.fromkeys(cleaned))


def read_tormes_plasmids(sample_to_species):
    rows = []

    for d in TORMES_PLASMID_DIRS:
        if not os.path.isdir(d):
            continue

        for path in glob.glob(os.path.join(d, "*_plasmids.tab")):
            sample = os.path.basename(path).replace("_plasmids.tab", "")
            sp = sample_to_species.get(sample, "") or species_from_sample(sample)

            df = safe_read_table(path)
            if df.empty:
                continue

            incs = []
            for _, r in df.iterrows():
                joined = " ".join(str(v) for v in r.values)
                incs.extend(extract_inc_tokens_from_text(joined))

            incs = list(dict.fromkeys([x for x in incs if x]))

            for inc in incs:
                rows.append({
                    "species": sp,
                    "feature": inc,
                    "sample": sample,
                    "count": 1,
                })

    if not rows:
        return pd.DataFrame(columns=["species", "feature", "sample", "count"])

    out = pd.DataFrame(rows)
    out = out[out["species"].astype(str).str.len() > 0].copy()

    # Count each Inc group once per sample/species.
    out = out.drop_duplicates(["species", "feature", "sample"])
    return out


def build_sample_maps(master_df, plasmid_sample_df):
    sample_col = guess_col(master_df, ["Sample", "SampleID", "Sample_ID", "sample"])
    sp_col = guess_col(master_df, ["TopSpecies", "TopSpecies1", "Species"], contains=["species"])

    sample_to_species = {}
    if sample_col and sp_col:
        for _, r in master_df.iterrows():
            sample = clean_text(r.get(sample_col, ""))
            sp = clean_text(r.get(sp_col, ""))
            if sample and sp:
                sample_to_species[sample] = sp

    amr_col = guess_col(master_df, [
        "AMR_Genes", "AMRGenes", "AMRFinder_Genes", "AMRFinderPlus_Genes",
        "ResistanceGenes", "Resistance_Genes", "ARGs", "AMR_gene_list",
        "AMR_TopGene", "AMR"
    ])

    vf_col = guess_col(master_df, [
        "Virulence_Genes", "VFDB_Genes", "VFDB_Gene", "VirulenceGenes",
        "Virulence_gene_list", "Virulence", "VF"
    ])

    plasmid_col = guess_col(master_df, [
        "Plasmid", "Plasmids", "Plasmid_IncGroup", "IncGroup", "Inc_Group",
        "Replicon", "Replicons", "Plasmid_Replicon"
    ])

    sample_to_amr = defaultdict(list)
    sample_to_vf = defaultdict(list)
    sample_to_plasmid = defaultdict(list)

    if sample_col:
        for _, r in master_df.iterrows():
            sample = clean_text(r.get(sample_col, ""))
            if not sample:
                continue

            if amr_col:
                sample_to_amr[sample].extend(split_features(r.get(amr_col, "")))

            if vf_col:
                sample_to_vf[sample].extend(split_features(r.get(vf_col, "")))

            if plasmid_col:
                sample_to_plasmid[sample].extend(split_features(r.get(plasmid_col, "")))

    if not plasmid_sample_df.empty:
        for _, r in plasmid_sample_df.iterrows():
            sample = clean_text(r.get("sample", ""))
            feature = clean_text(r.get("feature", ""))
            if sample and feature:
                sample_to_plasmid[sample].append(feature)

    # Deduplicate each sample feature list
    sample_to_amr = {k: list(dict.fromkeys(v)) for k, v in sample_to_amr.items()}
    sample_to_vf = {k: list(dict.fromkeys(v)) for k, v in sample_to_vf.items()}
    sample_to_plasmid = {k: list(dict.fromkeys(v)) for k, v in sample_to_plasmid.items()}

    return sample_to_species, sample_to_amr, sample_to_vf, sample_to_plasmid


def build_context_counts(species_order, sample_to_species, sample_to_amr, sample_to_vf, sample_to_plasmid):
    rows = []

    all_samples = set(sample_to_species)
    all_samples |= set(sample_to_amr)
    all_samples |= set(sample_to_vf)
    all_samples |= set(sample_to_plasmid)

    for sample in all_samples:
        sp = sample_to_species.get(sample, "") or species_from_sample(sample)
        if sp not in species_order:
            continue

        has_amr = len(sample_to_amr.get(sample, [])) > 0
        has_vf = len(sample_to_vf.get(sample, [])) > 0
        has_plasmid = len(sample_to_plasmid.get(sample, [])) > 0

        if has_amr and has_vf and has_plasmid:
            category = "AMR + virulence + plasmid"
        elif has_amr and has_plasmid:
            category = "AMR + plasmid"
        elif has_amr and has_vf:
            category = "AMR + virulence"
        elif has_vf and has_plasmid:
            category = "virulence + plasmid"
        elif has_amr:
            category = "AMR only"
        elif has_vf:
            category = "virulence only"
        elif has_plasmid:
            category = "plasmid only"
        else:
            category = "no detected context"

        rows.append({
            "sample": sample,
            "species": sp,
            "category": category
        })

    if not rows:
        return pd.DataFrame(columns=["species", "category", "count"])

    df = pd.DataFrame(rows)
    return df.groupby(["species", "category"], as_index=False).size().rename(columns={"size": "count"})


# =========================================================
# Read source data
# =========================================================
species_counts_path = first_existing(SPECIES_COUNTS_CANDIDATES)
master_path = first_existing(MASTER_CANDIDATES)
amr_path = first_existing(AMR_LONG_CANDIDATES)
vf_path = first_existing(VF_LONG_CANDIDATES)
plasmid_long_path = first_existing(PLASMID_LONG_CANDIDATES)

species_counts = safe_read_table(species_counts_path)
master = safe_read_table(master_path)

amr_long = read_long_feature_table(
    amr_path,
    species_candidates=["TopSpecies", "TopSpecies1", "Species"],
    feature_candidates=["AMR_Gene", "Gene", "AMR_TopGene", "Element", "Feature"],
    count_candidates=["Count", "Hits", "N", "n"]
)

vf_long = read_long_feature_table(
    vf_path,
    species_candidates=["TopSpecies", "TopSpecies1", "Species"],
    feature_candidates=["VFDB_Gene", "Virulence_Gene", "Gene", "Element", "Feature"],
    count_candidates=["Count", "Hits", "N", "n"]
)

plasmid_long = read_long_feature_table(
    plasmid_long_path,
    species_candidates=["TopSpecies", "TopSpecies1", "Species"],
    feature_candidates=["IncGroup", "Inc_Group", "Plasmid", "Replicon", "Gene", "Feature"],
    count_candidates=["Count", "Hits", "N", "n"]
)

species_order = get_species_order(species_counts, master, min_count=4)

# Preliminary sample/species mapping before TORMES parsing
sample_to_species_initial = {}
if not master.empty:
    sample_col_tmp = guess_col(master, ["Sample", "SampleID", "Sample_ID", "sample"])
    sp_col_tmp = guess_col(master, ["TopSpecies", "TopSpecies1", "Species"], contains=["species"])
    if sample_col_tmp and sp_col_tmp:
        sample_to_species_initial = {
            clean_text(r.get(sample_col_tmp, "")): clean_text(r.get(sp_col_tmp, ""))
            for _, r in master.iterrows()
            if clean_text(r.get(sample_col_tmp, "")) and clean_text(r.get(sp_col_tmp, ""))
        }

plasmid_sample_long = read_tormes_plasmids(sample_to_species_initial)

# If no aggregated plasmid table exists, build one from TORMES sample-level tabs.
if plasmid_long.empty and not plasmid_sample_long.empty:
    plasmid_long = (
        plasmid_sample_long
        .drop_duplicates(["species", "feature", "sample"])
        .groupby(["species", "feature"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )

sample_to_species, sample_to_amr, sample_to_vf, sample_to_plasmid = build_sample_maps(
    master,
    plasmid_sample_long
)

if not species_order:
    species_order = sorted(set(amr_long["species"]) | set(vf_long["species"]) | set(plasmid_long["species"]))

# Keep only species with actual data in this figure.
species_order = [
    sp for sp in species_order
    if (
        sp in set(amr_long["species"])
        or sp in set(vf_long["species"])
        or sp in set(plasmid_long["species"])
        or sp in set(sample_to_species.values())
    )
]

# Stable fallback order if needed
species_order = sorted(
    species_order,
    key=lambda sp: [
        "Serratia marcescens",
        "Acinetobacter baumannii",
        "Klebsiella pneumoniae",
        "Pseudomonas aeruginosa",
        "Escherichia coli",
    ].index(sp) if sp in [
        "Serratia marcescens",
        "Acinetobacter baumannii",
        "Klebsiella pneumoniae",
        "Pseudomonas aeruginosa",
        "Escherichia coli",
    ] else 999
)

context_counts = build_context_counts(
    species_order,
    sample_to_species,
    sample_to_amr,
    sample_to_vf,
    sample_to_plasmid
)

print("[Figure 5] Source files")
print("  species_counts:", species_counts_path)
print("  master:", master_path)
print("  AMR:", amr_path)
print("  virulence:", vf_path)
print("  plasmid aggregated:", plasmid_long_path)
print("  TORMES plasmid sample rows:", len(plasmid_sample_long))
print("  species_order:", species_order)


# =========================================================
# Feature selections
# =========================================================
amr_features = top_features(amr_long[amr_long["species"].isin(species_order)], top_n=14)
vf_features = top_features(vf_long[vf_long["species"].isin(species_order)], top_n=14)
plasmid_features = top_features(plasmid_long[plasmid_long["species"].isin(species_order)], top_n=14)

amr_mat = matrix_from_long(amr_long, species_order, amr_features, log1p=True)
vf_mat = matrix_from_long(vf_long, species_order, vf_features, log1p=True)
plasmid_mat_raw = matrix_from_long(plasmid_long, species_order, plasmid_features, log1p=False)


# =========================================================
# Panel A — context burden
# =========================================================
def draw_context_panel(ax):
    add_panel_card(ax)
    style_axis(ax, "x")

    ax.set_title(
        "A. AMR–virulence–plasmid sample context by species",
        loc="left",
        pad=12
    )

    if context_counts.empty:
        ax.axis("off")
        ax.text(
            0.5, 0.55,
            "Sample-level context columns were not detected.\n"
            "The remaining panels use species-level AMR, virulence and plasmid summaries.",
            ha="center",
            va="center",
            fontsize=12.5,
            fontweight="bold",
            color="#64748b",
            transform=ax.transAxes
        )
        return

    categories = [
        "AMR + virulence + plasmid",
        "AMR + plasmid",
        "AMR + virulence",
        "virulence + plasmid",
        "AMR only",
        "virulence only",
        "plasmid only",
        "no detected context",
    ]

    y = np.arange(len(species_order))
    left = np.zeros(len(species_order), dtype=float)

    pivot = (
        context_counts
        .pivot_table(index="species", columns="category", values="count", aggfunc="sum", fill_value=0)
        .reindex(species_order)
        .fillna(0)
    )

    for cat in categories:
        if cat not in pivot.columns:
            continue

        vals = pivot[cat].values.astype(float)
        if np.all(vals == 0):
            continue

        ax.barh(
            y,
            vals,
            left=left,
            height=0.66,
            color=context_colors.get(cat, "#cbd5e1"),
            edgecolor="white",
            linewidth=1.1,
            label=cat,
            zorder=3
        )

        for i, v in enumerate(vals):
            if v >= 2:
                ax.text(
                    left[i] + v / 2,
                    y[i],
                    f"{int(v)}",
                    ha="center",
                    va="center",
                    fontsize=8.8,
                    fontweight="bold",
                    color="white" if cat in {"AMR + virulence + plasmid", "AMR + plasmid", "AMR + virulence"} else "#111827",
                    zorder=4
                )

        left += vals

    ax.set_yticks(y)
    ax.set_yticklabels([short_species(s) for s in species_order])
    ax.invert_yaxis()
    ax.set_xlabel("Number of samples")
    ax.set_xlim(0, max(left.max() * 1.08, 1))

    for tick, sp in zip(ax.get_yticklabels(), species_order):
        tick.set_color(species_palette.get(sp, "#334155"))
        tick.set_fontweight("bold")

    ax.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.34),
        ncol=4,
        fontsize=8.7,
        handlelength=1.2,
        columnspacing=1.1
    )


# =========================================================
# Panel D — plasmid bubble matrix
# =========================================================
def draw_plasmid_bubble_panel(ax):
    add_panel_card(ax)

    ax.set_title(
        "D. Plasmid incompatibility / replicon context",
        loc="left",
        pad=12
    )

    if plasmid_mat_raw.size == 0 or len(plasmid_features) == 0:
        ax.axis("off")
        ax.text(
            0.5, 0.5,
            "No plasmid Inc/replicon table detected.\n"
            "Expected aggregated plasmid table or TORMES *_plasmids.tab files.",
            ha="center",
            va="center",
            fontsize=12.5,
            fontweight="bold",
            color="#64748b",
            transform=ax.transAxes
        )
        return

    xs, ys, sizes, colors = [], [], [], []
    max_count = np.nanmax(plasmid_mat_raw) if plasmid_mat_raw.size else 1
    max_count = max(max_count, 1)

    for i in range(len(species_order)):
        for j in range(len(plasmid_features)):
            val = plasmid_mat_raw[i, j]
            if val <= 0:
                continue
            xs.append(j)
            ys.append(i)
            sizes.append(80 + (val / max_count) * 1050)
            colors.append(val)

    ax.scatter(
        xs,
        ys,
        s=sizes,
        c=colors,
        cmap=plasmid_cmap,
        edgecolor="#78350f",
        linewidth=0.8,
        alpha=0.92,
        zorder=3
    )

    ax.set_yticks(np.arange(len(species_order)))
    ax.set_yticklabels([short_species(s) for s in species_order], fontsize=9.8)

    ax.set_xticks(np.arange(len(plasmid_features)))
    ax.set_xticklabels(
        [wrap_label(f, 12) for f in plasmid_features],
        rotation=45,
        ha="right",
        va="top",
        fontsize=8.8
    )

    ax.set_xlim(-0.6, len(plasmid_features) - 0.4)
    ax.set_ylim(len(species_order) - 0.5, -0.5)
    ax.tick_params(axis="both", length=0)

    ax.set_xticks(np.arange(-0.5, len(plasmid_features), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(species_order), 1), minor=True)
    ax.grid(which="minor", color="#f1f5f9", linestyle="-", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    for tick, sp in zip(ax.get_yticklabels(), species_order):
        tick.set_color(species_palette.get(sp, "#334155"))
        tick.set_fontweight("bold")

    sm = ScalarMappable(norm=Normalize(vmin=0, vmax=max_count), cmap=plasmid_cmap)
    cb = plt.colorbar(sm, ax=ax, fraction=0.030, pad=0.018)
    cb.set_label("Samples / hits", fontsize=9.8)
    cb.ax.tick_params(labelsize=8.8)
    cb.outline.set_visible(False)

    # Size legend
    legend_vals = sorted(set([1, int(round(max_count / 2)), int(max_count)]))
    legend_vals = [v for v in legend_vals if v > 0]

    handles = [
        Line2D(
            [0], [0],
            marker="o",
            color="w",
            markerfacecolor="#fbbf24",
            markeredgecolor="#78350f",
            markersize=math.sqrt(80 + (v / max_count) * 1050) / 1.9,
            label=str(v)
        )
        for v in legend_vals
    ]

    ax.legend(
        handles=handles,
        title="Count",
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(1.00, 1.02),
        fontsize=8.8,
        title_fontsize=9.2
    )


# =========================================================
# Panel E — contextual network
# =========================================================
def draw_node(ax, x, y, label, fc, ec="#cbd5e1", text_color="#111827", w=0.155, h=0.052, fs=8.8):
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.010,rounding_size=0.018",
        facecolor=fc,
        edgecolor=ec,
        linewidth=1.0,
        zorder=5
    )
    patch.set_path_effects([
        pe.withSimplePatchShadow(offset=(1.2, -1.2), alpha=0.12),
        pe.Normal()
    ])
    ax.add_patch(patch)
    ax.text(
        x,
        y,
        wrap_label(label, 18),
        ha="center",
        va="center",
        fontsize=fs,
        fontweight="bold",
        color=text_color,
        zorder=6
    )


def draw_edge(ax, p1, p2, weight, max_weight, color, alpha=0.30):
    lw = 0.45 + 4.2 * (weight / max(max_weight, 1))
    ax.plot(
        [p1[0], p2[0]],
        [p1[1], p2[1]],
        color=color,
        lw=lw,
        alpha=alpha,
        solid_capstyle="round",
        zorder=1
    )


def draw_context_network(ax):
    add_panel_card(ax)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.set_title(
        "E. Integrated AMR–plasmid–virulence context network",
        loc="left",
        pad=12
    )

    top_amr = top_features(amr_long[amr_long["species"].isin(species_order)], top_n=7)
    top_vf = top_features(vf_long[vf_long["species"].isin(species_order)], top_n=7)
    top_plasmid = top_features(plasmid_long[plasmid_long["species"].isin(species_order)], top_n=6)

    if not top_amr and not top_vf and not top_plasmid:
        ax.text(
            0.5,
            0.5,
            "No network-eligible AMR, virulence or plasmid features detected.",
            ha="center",
            va="center",
            fontsize=12.5,
            fontweight="bold",
            color="#64748b"
        )
        return

    col_x = {
        "Species": 0.08,
        "AMR": 0.34,
        "Plasmid": 0.62,
        "Virulence": 0.90,
    }

    for label, x in col_x.items():
        ax.text(
            x,
            0.95,
            label,
            ha="center",
            va="center",
            fontsize=11.0,
            fontweight="bold",
            color="#0f172a",
            bbox=dict(boxstyle="round,pad=0.25", fc="#f8fafc", ec="#e2e8f0")
        )

    node_pos = {}

    def spread(items, top=0.84, bottom=0.18):
        if not items:
            return []
        if len(items) == 1:
            return [0.5]
        return np.linspace(top, bottom, len(items))

    for y, sp in zip(spread(species_order), species_order):
        node_pos[("species", sp)] = (col_x["Species"], y)
        draw_node(
            ax,
            col_x["Species"],
            y,
            short_species(sp),
            fc=species_palette.get(sp, "#64748b"),
            ec="white",
            text_color="white",
            w=0.155,
            h=0.058,
            fs=8.7
        )

    for y, feat in zip(spread(top_amr), top_amr):
        node_pos[("amr", feat)] = (col_x["AMR"], y)
        draw_node(ax, col_x["AMR"], y, feat, fc="#dbeafe", ec="#60a5fa", w=0.165, h=0.054)

    for y, feat in zip(spread(top_plasmid), top_plasmid):
        node_pos[("plasmid", feat)] = (col_x["Plasmid"], y)
        draw_node(ax, col_x["Plasmid"], y, feat, fc="#fef3c7", ec="#f59e0b", w=0.175, h=0.054)

    for y, feat in zip(spread(top_vf), top_vf):
        node_pos[("vf", feat)] = (col_x["Virulence"], y)
        draw_node(ax, col_x["Virulence"], y, feat, fc="#fce7f3", ec="#f472b6", w=0.170, h=0.054)

    edges = []

    # Species-linked AMR edges
    for _, r in amr_long.iterrows():
        sp, feat, c = r["species"], r["feature"], as_num(r["count"], 0)
        if sp in species_order and feat in top_amr:
            edges.append((("species", sp), ("amr", feat), c, "#2563eb"))

    # Species-linked plasmid edges
    for _, r in plasmid_long.iterrows():
        sp, feat, c = r["species"], r["feature"], as_num(r["count"], 0)
        if sp in species_order and feat in top_plasmid:
            edges.append((("species", sp), ("plasmid", feat), c, "#b45309"))

    # Species-linked virulence edges
    for _, r in vf_long.iterrows():
        sp, feat, c = r["species"], r["feature"], as_num(r["count"], 0)
        if sp in species_order and feat in top_vf:
            edges.append((("species", sp), ("vf", feat), c, "#db2777"))

    # Sample-level AMR -> plasmid and plasmid -> virulence co-detection edges, if available.
    pair_counter_ap = Counter()
    pair_counter_pv = Counter()

    all_samples = set(sample_to_amr) | set(sample_to_vf) | set(sample_to_plasmid)
    for sample in all_samples:
        amrs = [x for x in sample_to_amr.get(sample, []) if x in top_amr]
        vfs = [x for x in sample_to_vf.get(sample, []) if x in top_vf]
        pls = [x for x in sample_to_plasmid.get(sample, []) if x in top_plasmid]

        for a in amrs:
            for p in pls:
                pair_counter_ap[(a, p)] += 1

        for p in pls:
            for v in vfs:
                pair_counter_pv[(p, v)] += 1

    for (a, p), c in pair_counter_ap.items():
        edges.append((("amr", a), ("plasmid", p), c, "#475569"))

    for (p, v), c in pair_counter_pv.items():
        edges.append((("plasmid", p), ("vf", v), c, "#475569"))

    if edges:
        max_w = max(w for _, _, w, _ in edges)
        for n1, n2, w, color in sorted(edges, key=lambda x: x[2]):
            if n1 in node_pos and n2 in node_pos:
                draw_edge(ax, node_pos[n1], node_pos[n2], w, max_w, color)

    ax.text(
        0.01,
        0.035,
        "Edges are weighted by source-table counts; AMR–plasmid and plasmid–virulence links use sample-level co-detection when available.",
        ha="left",
        va="bottom",
        fontsize=8.7,
        color="#64748b",
        transform=ax.transAxes
    )


# =========================================================
# Figure layout
# =========================================================
fig = plt.figure(figsize=(22.0, 16.2), facecolor="white")

gs = GridSpec(
    3,
    2,
    figure=fig,
    width_ratios=[1.06, 1.0],
    height_ratios=[0.88, 1.0, 1.16],
    hspace=0.42,
    wspace=0.22
)

axA = fig.add_subplot(gs[0, :])
draw_context_panel(axA)

axB = fig.add_subplot(gs[1, 0])
draw_heatmap(
    axB,
    amr_mat,
    species_order,
    amr_features,
    amr_cmap,
    "B. Dominant AMR gene signatures by species",
    "log(1 + count)"
)

axC = fig.add_subplot(gs[1, 1])
draw_heatmap(
    axC,
    vf_mat,
    species_order,
    vf_features,
    vf_cmap,
    "C. Dominant virulence gene signatures by species",
    "log(1 + count)"
)

axD = fig.add_subplot(gs[2, 0])
draw_plasmid_bubble_panel(axD)

axE = fig.add_subplot(gs[2, 1])
draw_context_network(axE)


# =========================================================
# Title and final polish
# =========================================================
fig.suptitle(
    "AMR–virulence–plasmid co-occurrence and genomic context architecture",
    y=0.992,
    fontsize=25.5,
    fontweight="bold"
)

fig.subplots_adjust(left=0.055, right=0.965, top=0.925, bottom=0.065)

save_png_pdf(fig, OUTDIR, OUTNAME)
print("Saved Figure 5 to:", OUTDIR)