# =========================================================
# Figure 6 — Pangenome / Core-Accessory Genome Architecture
# Premium WGS manuscript figure
# =========================================================

# =========================================================
# Paths
# =========================================================
import os

# Use this path when running from WSL/Linux Python
WORK = "/mnt/e/DrAhmed/Ongoing/WGS/Result"

# If you run this script with Windows Python instead of WSL Python,
# comment the WSL line above and use this instead:
# WORK = r"E:/DrAhmed/Ongoing/WGS/Result"

MASTER = f"{WORK}/_BIOLOGY_LAYER_V2/PrimaryResults_v4_withBiology_166.csv"

OUTDIR = f"{WORK}/_G5_PANGENOME/output/main"
CACHE_DIR = f"{WORK}/_G5_PANGENOME/cache"

os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

OUTNAME = "Figure06_Pangenome_Core_Accessory_Architecture"

PANGENOME_CANDIDATES = [
    f"{WORK}/_PANGENOME/mmseqs/gene_presence_absence.csv",
    f"{WORK}/_PANGENOME/gene_presence_absence.csv",
    f"{WORK}/_PANGENOME/panaroo/gene_presence_absence.csv",
    f"{WORK}/_PANGENOME/roary/gene_presence_absence.csv",
    f"{WORK}/pangenome/gene_presence_absence.csv",
    f"{WORK}/panaroo/gene_presence_absence.csv",
    f"{WORK}/roary/gene_presence_absence.csv",
]

HIGHCONF_CANDIDATES = [
    f"{WORK}/_PRELIMINARY_REPORT_READY/Table5_HighConfidenceSamples.csv",
    f"{WORK}/_MANUSCRIPT_FINAL/Table5_HighConfidenceSamples.csv",
]

PRIORITY_CANDIDATES = [
    f"{WORK}/_PRELIMINARY_REPORT_READY/Table4_PriorityReviewSamples.csv",
    f"{WORK}/_PRELIMINARY_PACK/Table_FlaggedSamples_refined.csv",
]

import re
import csv
import math
import random
import urllib.parse
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm, Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


# =========================================================
# Global style
# =========================================================
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11.5,
    "axes.titlesize": 15.8,
    "axes.titleweight": "bold",
    "axes.labelsize": 12.6,
    "xtick.labelsize": 10.0,
    "ytick.labelsize": 10.2,
    "figure.titlesize": 24.5,
    "figure.titleweight": "bold",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 700,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# =========================================================
# Paths
# =========================================================
WORK = "/mnt/e/DrAhmed/Ongoing/WGS/Result"
MASTER = f"{WORK}/_BIOLOGY_LAYER_V2/PrimaryResults_v4_withBiology_166.csv"

OUTDIR = f"{WORK}/_G5_PANGENOME/output/main"
CACHE_DIR = f"{WORK}/_G5_PANGENOME/cache"
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

OUTNAME = "Figure06_Pangenome_Core_Accessory_Architecture"

# Preferred true pangenome outputs if available.
PANGENOME_CANDIDATES = [
    f"{WORK}/_PANGENOME/gene_presence_absence.csv",
    f"{WORK}/_PANGENOME/panaroo/gene_presence_absence.csv",
    f"{WORK}/_PANGENOME/roary/gene_presence_absence.csv",
    f"{WORK}/pangenome/gene_presence_absence.csv",
    f"{WORK}/panaroo/gene_presence_absence.csv",
    f"{WORK}/roary/gene_presence_absence.csv",
]

HIGHCONF_CANDIDATES = [
    f"{WORK}/_PRELIMINARY_REPORT_READY/Table5_HighConfidenceSamples.csv",
    f"{WORK}/_MANUSCRIPT_FINAL/Table5_HighConfidenceSamples.csv",
]

PRIORITY_CANDIDATES = [
    f"{WORK}/_PRELIMINARY_REPORT_READY/Table4_PriorityReviewSamples.csv",
    f"{WORK}/_PRELIMINARY_PACK/Table_FlaggedSamples_refined.csv",
]


# =========================================================
# Colors
# =========================================================
SPECIES_COLORS = {
    "Serratia marcescens": "#2563eb",
    "Acinetobacter baumannii": "#0f766e",
    "Klebsiella pneumoniae": "#8b5cf6",
    "Pseudomonas aeruginosa": "#d946ef",
    "Escherichia coli": "#f59e0b",
}

CONF_COLORS = {
    "High-confidence": "#2563eb",
    "Priority-review": "#dc2626",
    "Other/remaining": "#94a3b8",
    "Other": "#94a3b8",
}

PART_COLORS = {
    "Core": "#0f172a",
    "Soft-core": "#2563eb",
    "Shell": "#f59e0b",
    "Cloud": "#e11d48",
}

CATEGORY_COLORS = {
    "AMR / stress": "#dc2626",
    "Virulence / motility": "#db2777",
    "Mobile / plasmid": "#7c3aed",
    "Transport / membrane": "#0891b2",
    "Metabolism": "#16a34a",
    "Regulation / signaling": "#f59e0b",
    "Hypothetical / unknown": "#94a3b8",
    "Other annotated": "#64748b",
}

PAN_CMAP = LinearSegmentedColormap.from_list(
    "pan_premium",
    ["#eff6ff", "#bfdbfe", "#60a5fa", "#2563eb", "#0f172a"]
)

PRESENCE_CMAP = ListedColormap(["#f8fafc", "#111827"])
PRESENCE_NORM = BoundaryNorm([-0.5, 0.5, 1.5], PRESENCE_CMAP.N)

DENSITY_CMAP = LinearSegmentedColormap.from_list(
    "density_cmap",
    ["#f8fafc", "#dbeafe", "#93c5fd", "#1d4ed8", "#0f172a"]
)


# =========================================================
# Basic helpers
# =========================================================
def first_existing(paths):
    for p in paths:
        if p and os.path.isfile(p):
            return p
    return None


def read_csv_auto(path):
    if not path or not os.path.isfile(path):
        return pd.DataFrame()

    try:
        df = pd.read_csv(path, sep=None, engine="python", dtype=str, encoding="utf-8")
    except Exception:
        try:
            df = pd.read_csv(path, dtype=str, encoding="utf-8")
        except Exception:
            df = pd.read_csv(path, dtype=str, encoding="latin1")

    df.columns = [str(c).strip() for c in df.columns]
    return df.fillna("")


def norm(x):
    return str(x).strip()


def sample_key(x):
    s = norm(x)
    s = os.path.basename(s)
    s = re.sub(r"\.(gff|gff3|gbk|fna|fa|fasta|ffn|faa)$", "", s, flags=re.I)
    s = re.sub(r"[^A-Za-z0-9]+", "", s)
    return s.upper()


def clean_feature(x):
    s = str(x).strip()
    s = urllib.parse.unquote(s)
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" ;,|")
    return s


def species_short(sp):
    parts = norm(sp).split()
    if len(parts) >= 2:
        return f"{parts[0][0]}. {' '.join(parts[1:])}"
    return norm(sp)


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


def sample_set_from_table(paths):
    p = first_existing(paths)
    if not p:
        return set()

    df = read_csv_auto(p)
    sc = guess_col(df, ["Sample", "sample", "SampleID", "Sample_ID"], contains=["sample"])
    if not sc:
        return set()

    return {norm(x) for x in df[sc] if norm(x)}


def species_from_sample(sample):
    s = norm(sample)
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
    if s.upper().startswith("VIM_PSEUDOMONAS"):
        return "Pseudomonas aeruginosa"

    return ""


def add_panel_card(ax):
    ax.set_facecolor("white")
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color("#e2e8f0")
        s.set_linewidth(1.0)


def style_axis(ax, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#94a3b8")
    ax.spines["bottom"].set_color("#94a3b8")
    ax.tick_params(colors="#334155")

    if grid_axis in ("x", "both"):
        ax.grid(axis="x", color="#e5e7eb", linewidth=0.8, zorder=0)
    if grid_axis in ("y", "both"):
        ax.grid(axis="y", color="#eef2f7", linewidth=0.8, zorder=0)

    ax.set_axisbelow(True)


def save_png_pdf(fig, outdir, outname):
    png = os.path.join(outdir, outname + ".png")
    pdf = os.path.join(outdir, outname + ".pdf")
    fig.savefig(png, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    print("Saved:", png)
    print("Saved:", pdf)


# =========================================================
# Load metadata
# =========================================================
master = read_csv_auto(MASTER)

sample_to_species = {}
sample_to_confidence = {}

if not master.empty:
    sample_col = guess_col(master, ["Sample", "sample", "SampleID", "Sample_ID"], contains=["sample"])
    species_col = guess_col(master, ["TopSpecies1", "TopSpecies", "Species"], contains=["species"])

    if sample_col and species_col:
        for _, r in master.iterrows():
            sample = norm(r.get(sample_col, ""))
            sp = norm(r.get(species_col, ""))
            if sample and sp:
                sample_to_species[sample] = sp

    conf_col = guess_col(master, ["ConfidenceClass", "Confidence", "ReviewClass"])
    if sample_col and conf_col:
        for _, r in master.iterrows():
            sample = norm(r.get(sample_col, ""))
            cc = norm(r.get(conf_col, ""))
            if sample and cc:
                sample_to_confidence[sample] = cc

# Confidence override from project tables
highconf_set = sample_set_from_table(HIGHCONF_CANDIDATES)
priority_set = sample_set_from_table(PRIORITY_CANDIDATES)

for s in highconf_set:
    sample_to_confidence[s] = "High-confidence"
for s in priority_set:
    sample_to_confidence[s] = "Priority-review"

sample_key_to_master = {sample_key(s): s for s in sample_to_species}


# =========================================================
# Presence/absence matrix loaders
# =========================================================
ROARY_METADATA_COLS = {
    "gene",
    "non-unique gene name",
    "annotation",
    "no. isolates",
    "no. sequences",
    "avg sequences per isolate",
    "genome fragment",
    "order within fragment",
    "accessory fragment",
    "accessory order with fragment",
    "qc",
    "min group size nuc",
    "max group size nuc",
    "avg group size nuc",
    "min group size aa",
    "max group size aa",
    "avg group size aa",
}


def discover_gene_presence_absence_files():
    found = []

    for p in PANGENOME_CANDIDATES:
        if os.path.isfile(p):
            found.append(p)

    # Controlled recursive discovery
    for root, dirs, files in os.walk(WORK):
        low = root.lower()

        if any(skip in low for skip in [
            "_g3", "_g4", "_g5", "output", "cache", "_tmp_extract", "upload_parts"
        ]):
            continue

        for f in files:
            if f.lower() == "gene_presence_absence.csv":
                found.append(os.path.join(root, f))

    # Deduplicate
    out = []
    seen = set()
    for p in found:
        ap = os.path.abspath(p)
        if ap not in seen:
            seen.add(ap)
            out.append(p)

    return out


def infer_gpa_sample_cols(df):
    cols = list(df.columns)

    known_master_keys = set(sample_key_to_master.keys())

    direct_cols = []
    for c in cols:
        if c.lower() in ROARY_METADATA_COLS:
            continue
        ck = sample_key(c)
        if ck in known_master_keys:
            direct_cols.append(c)

    if len(direct_cols) >= 3:
        return direct_cols

    # Roary/Panaroo conventional fallback: samples after QC column
    qc_candidates = [i for i, c in enumerate(cols) if c.lower() == "qc"]
    if qc_candidates:
        idx = qc_candidates[-1]
        candidate_cols = cols[idx + 1:]
        if len(candidate_cols) >= 3:
            return candidate_cols

    # Panaroo may include metadata before sample columns, but no QC in some versions.
    non_meta = [c for c in cols if c.lower() not in ROARY_METADATA_COLS]
    if len(non_meta) > 10:
        return non_meta[3:]

    return []


def load_gpa_matrix(path):
    df = read_csv_auto(path)
    if df.empty:
        return None, None, None

    sample_cols = infer_gpa_sample_cols(df)
    if not sample_cols:
        return None, None, None

    gene_col = guess_col(df, ["Gene", "gene"])
    annot_col = guess_col(df, ["Annotation", "annotation", "Product", "product"])
    nonunique_col = guess_col(df, ["Non-unique Gene name", "Non_unique_Gene_name"])

    family_ids = []
    display_names = []
    annotations = []

    for i, r in df.iterrows():
        g = clean_feature(r.get(gene_col, "")) if gene_col else ""
        n = clean_feature(r.get(nonunique_col, "")) if nonunique_col else ""
        a = clean_feature(r.get(annot_col, "")) if annot_col else ""

        label = g or n or a or f"gene_family_{i + 1}"
        family_ids.append(f"FAM_{i + 1:07d}")
        display_names.append(label)
        annotations.append(a or label)

    matrix = pd.DataFrame(False, index=family_ids, columns=[])

    col_rename = {}
    for c in sample_cols:
        ck = sample_key(c)
        if ck in sample_key_to_master:
            sample = sample_key_to_master[ck]
        else:
            sample = clean_feature(c)

        col_rename[c] = sample

    for c in sample_cols:
        sample = col_rename[c]
        vals = df[c].astype(str).fillna("").str.strip()
        present = ~vals.isin(["", "0", "0.0", "nan", "NaN", "-", "None"])
        matrix[sample] = present.values

    # Merge duplicate sample columns if needed
    matrix = matrix.groupby(level=0, axis=1).max()

    info = pd.DataFrame({
        "family_id": family_ids,
        "display_name": display_names,
        "annotation": annotations,
        "source": os.path.basename(path),
    }).set_index("family_id")

    return matrix.astype(bool), info, path


def parse_gff_attributes(attr_text):
    out = {}
    for part in attr_text.split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = urllib.parse.unquote(v.strip())
    return out


def normalize_annotation_family(gene, product, locus, sample):
    gene = clean_feature(gene)
    product = clean_feature(product)

    bad = {"", "na", "n/a", "none", "hypothetical protein", "uncharacterized protein"}

    if gene and gene.lower() not in bad:
        return "gene::" + gene.lower(), gene, product or gene

    if product and product.lower() not in bad and "hypothetical" not in product.lower():
        p = re.sub(r"\s+", " ", product.lower())
        p = re.sub(r"[^a-z0-9_ /()+.-]+", "", p)
        return "product::" + p, product, product

    # Conservative fallback: do not collapse all hypothetical proteins together.
    fam = f"hypothetical::{sample}::{locus}"
    return fam, "hypothetical protein", "hypothetical protein"


def choose_sample_from_gff_path(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    parent = os.path.basename(os.path.dirname(path))
    grand = os.path.basename(os.path.dirname(os.path.dirname(path)))

    candidates = [stem, parent, grand]

    for c in candidates:
        ck = sample_key(c)
        if ck in sample_key_to_master:
            return sample_key_to_master[ck]

    for c in candidates:
        if species_from_sample(c):
            return c

    return stem


def discover_gff_files():
    gffs = []

    for root, dirs, files in os.walk(WORK):
        low = root.lower()

        if any(skip in low for skip in [
            "_g3", "_g4", "_g5", "output", "cache", "_tmp_extract", "upload_parts"
        ]):
            continue

        for f in files:
            if f.lower().endswith((".gff", ".gff3")):
                gffs.append(os.path.join(root, f))

    return gffs


def build_matrix_from_gff_cache():
    matrix_cache = os.path.join(CACHE_DIR, "annotation_family_presence_matrix.csv")
    info_cache = os.path.join(CACHE_DIR, "annotation_family_info.csv")

    if os.path.isfile(matrix_cache) and os.path.isfile(info_cache):
        print("[Pangenome] Loading cached annotation-family matrix.")
        matrix = pd.read_csv(matrix_cache, index_col=0)
        matrix = matrix.astype(bool)
        info = pd.read_csv(info_cache, index_col=0)
        return matrix, info, "Annotation-family proxy from cached GFF parsing"

    gff_files = discover_gff_files()
    print(f"[Pangenome] GFF files discovered: {len(gff_files)}")

    records = []
    family_annotation = {}

    for path in gff_files:
        sample = choose_sample_from_gff_path(path)
        sp = sample_to_species.get(sample, "") or species_from_sample(sample)

        # Keep only samples that can be placed biologically
        if not sp:
            continue

        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if line.startswith("##FASTA"):
                        break
                    if line.startswith("#"):
                        continue

                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 9:
                        continue

                    ftype = parts[2]
                    if ftype not in {"CDS", "gene"}:
                        continue

                    attrs = parse_gff_attributes(parts[8])
                    gene = attrs.get("gene", "") or attrs.get("Name", "")
                    product = attrs.get("product", "")
                    locus = attrs.get("locus_tag", "") or attrs.get("ID", "") or f"{parts[0]}_{parts[3]}_{parts[4]}"

                    fam_key, display, annot = normalize_annotation_family(gene, product, locus, sample)

                    records.append((fam_key, sample))
                    if fam_key not in family_annotation:
                        family_annotation[fam_key] = {
                            "display_name": display,
                            "annotation": annot,
                            "source": "GFF annotation-family fallback",
                        }

        except Exception as e:
            print(f"[Pangenome] Could not parse {path}: {e}")

    if not records:
        raise RuntimeError(
            "No usable pangenome matrix or GFF-derived annotations were found. "
            "Please provide Panaroo/Roary gene_presence_absence.csv or Prokka/Bakta GFF files."
        )

    df = pd.DataFrame(records, columns=["family_id", "sample"]).drop_duplicates()
    matrix = pd.crosstab(df["family_id"], df["sample"]).astype(bool)

    info = pd.DataFrame.from_dict(family_annotation, orient="index")
    info.index.name = "family_id"

    matrix.to_csv(matrix_cache)
    info.to_csv(info_cache)

    return matrix, info, "Annotation-family proxy from Prokka/Bakta GFF files"


def load_presence_matrix():
    gpa_files = discover_gene_presence_absence_files()

    candidates = []
    for p in gpa_files:
        try:
            matrix, info, source = load_gpa_matrix(p)
            if matrix is not None and matrix.shape[1] >= 3:
                matched = sum(1 for c in matrix.columns if c in sample_to_species)
                candidates.append((matched, matrix.shape[1], matrix.shape[0], matrix, info, source))
                print(f"[Pangenome] Candidate GPA: {p} | families={matrix.shape[0]} | samples={matrix.shape[1]} | matched={matched}")
        except Exception as e:
            print(f"[Pangenome] Failed GPA load: {p} | {e}")

    if candidates:
        candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        matched, ns, nf, matrix, info, source = candidates[0]
        print(f"[Pangenome] Using true pangenome matrix: {source}")
        return matrix, info, f"Orthogroup matrix from {os.path.basename(source)}"

    print("[Pangenome] No Roary/Panaroo gene_presence_absence.csv found. Falling back to GFF-derived annotation-family matrix.")
    return build_matrix_from_gff_cache()


# =========================================================
# Build pangenome data
# =========================================================
presence, family_info, source_note = load_presence_matrix()

# Keep only samples with known or inferable species
matrix_samples = list(presence.columns)

for s in matrix_samples:
    if s not in sample_to_species:
        inferred = species_from_sample(s)
        if inferred:
            sample_to_species[s] = inferred

    if s not in sample_to_confidence:
        sample_to_confidence[s] = "Other/remaining"

dominant_species_preferred = [
    "Serratia marcescens",
    "Acinetobacter baumannii",
    "Klebsiella pneumoniae",
    "Pseudomonas aeruginosa",
    "Escherichia coli",
]

samples_by_species = {}
for sp in dominant_species_preferred:
    ss = [s for s in matrix_samples if sample_to_species.get(s, "") == sp]
    if len(ss) >= 3:
        samples_by_species[sp] = ss

if not samples_by_species:
    raise RuntimeError("No dominant species had at least 3 samples in the pangenome matrix.")

dominant_species = list(samples_by_species.keys())

print("[Pangenome] Dominant species included:")
for sp, ss in samples_by_species.items():
    print(f"  {sp}: n={len(ss)}")


# =========================================================
# Core/accessory partition
# =========================================================
def partition_species(matrix, samples):
    sub = matrix[samples].astype(bool)
    n = len(samples)
    counts = sub.sum(axis=1).values.astype(float)
    freq = counts / max(n, 1)

    present = counts > 0
    core = present & (freq >= 0.99)
    soft = present & (freq >= 0.95) & (freq < 0.99)
    shell = present & (freq >= 0.15) & (freq < 0.95)
    cloud = present & (freq < 0.15)

    return {
        "n": n,
        "pan": int(present.sum()),
        "core": int(core.sum()),
        "soft": int(soft.sum()),
        "shell": int(shell.sum()),
        "cloud": int(cloud.sum()),
        "freq": freq,
        "counts": counts,
        "present_mask": present,
        "core_mask": core,
        "soft_mask": soft,
        "shell_mask": shell,
        "cloud_mask": cloud,
    }


partitions = {
    sp: partition_species(presence, samples_by_species[sp])
    for sp in dominant_species
}


# =========================================================
# Accumulation curves
# =========================================================
def accumulation_curves(sub_matrix, iterations=160, seed=42):
    rng = np.random.default_rng(seed)
    arr = sub_matrix.astype(bool).values
    n_fam, n_samp = arr.shape

    pan_vals = np.zeros((iterations, n_samp), dtype=float)
    core_vals = np.zeros((iterations, n_samp), dtype=float)

    for it in range(iterations):
        order = rng.permutation(n_samp)

        union = np.zeros(n_fam, dtype=bool)
        inter = np.ones(n_fam, dtype=bool)

        for k, idx in enumerate(order):
            col = arr[:, idx]
            union |= col
            inter &= col
            pan_vals[it, k] = union.sum()
            core_vals[it, k] = inter.sum()

    return {
        "k": np.arange(1, n_samp + 1),
        "pan_mean": pan_vals.mean(axis=0),
        "pan_lo": np.percentile(pan_vals, 5, axis=0),
        "pan_hi": np.percentile(pan_vals, 95, axis=0),
        "core_mean": core_vals.mean(axis=0),
        "core_lo": np.percentile(core_vals, 5, axis=0),
        "core_hi": np.percentile(core_vals, 95, axis=0),
    }


curves = {}
for sp in dominant_species:
    sub = presence[samples_by_species[sp]]
    iters = 70 if len(samples_by_species[sp]) <= 30 else 45
    curves[sp] = accumulation_curves(sub, iterations=iters, seed=42)


# =========================================================
# Per-sample accessory burden
# =========================================================
sample_accessory_records = []

for sp in dominant_species:
    ss = samples_by_species[sp]
    sub = presence[ss].astype(bool)
    part = partitions[sp]

    core_mask = pd.Series(part["core_mask"], index=presence.index)
    accessory_families = presence.index[~core_mask.values]

    for s in ss:
        acc_count = int(sub.loc[accessory_families, s].sum())
        total_count = int(sub[s].sum())
        sample_accessory_records.append({
            "sample": s,
            "species": sp,
            "confidence": sample_to_confidence.get(s, "Other/remaining"),
            "accessory_count": acc_count,
            "total_family_count": total_count,
            "accessory_fraction": acc_count / total_count if total_count else 0.0,
        })

accessory_df = pd.DataFrame(sample_accessory_records)


# =========================================================
# Functional category assignment
# =========================================================
def functional_category(text):
    s = str(text).lower()

    if any(x in s for x in [
        "bla", "beta-lactam", "betalactam", "aminoglycoside", "tetracycline",
        "sulfonamide", "sul1", "qac", "mercury", "arsenic", "copper", "silver",
        "efflux", "antibiotic", "resistance", "fosfomycin", "chloramphenicol",
        "macrolide", "metal resistance"
    ]):
        return "AMR / stress"

    if any(x in s for x in [
        "virulence", "flagell", "fimbr", "adhesin", "pilus", "pili",
        "secretion", "siderophore", "capsule", "biofilm", "toxin", "hemolysin",
        "yersiniabactin", "aerobactin", "fimbria", "motility", "chemotaxis"
    ]):
        return "Virulence / motility"

    if any(x in s for x in [
        "transposase", "integrase", "plasmid", "conjug", "phage",
        "recombinase", "insertion sequence", "transposon", "resolvase",
        "mobile element", "relaxase"
    ]):
        return "Mobile / plasmid"

    if any(x in s for x in [
        "transporter", "permease", "pump", "porin", "membrane", "channel",
        "export", "import", "abc transporter"
    ]):
        return "Transport / membrane"

    if any(x in s for x in [
        "dehydrogenase", "synthase", "kinase", "reductase", "transferase",
        "ligase", "oxidase", "isomerase", "hydrolase", "metabolism",
        "biosynthesis", "catabolic", "anabolic"
    ]):
        return "Metabolism"

    if any(x in s for x in [
        "regulator", "transcription", "sensor", "response regulator",
        "two-component", "repressor", "activator", "signaling"
    ]):
        return "Regulation / signaling"

    if any(x in s for x in [
        "hypothetical", "uncharacterized", "duf", "unknown"
    ]):
        return "Hypothetical / unknown"

    return "Other annotated"


family_info["category"] = [
    functional_category(
        f"{family_info.loc[idx, 'display_name']} {family_info.loc[idx, 'annotation']}"
    )
    for idx in family_info.index
]


category_by_species = defaultdict(Counter)

for sp in dominant_species:
    ss = samples_by_species[sp]
    sub = presence[ss].astype(bool)
    part = partitions[sp]

    core_mask = pd.Series(part["core_mask"], index=presence.index)
    accessory_fams = presence.index[(sub.sum(axis=1) > 0).values & (~core_mask.values)]

    for fam in accessory_fams:
        cat = family_info.loc[fam, "category"] if fam in family_info.index else "Other annotated"
        category_by_species[sp][cat] += 1


# =========================================================
# Openness / stability diagnostics
# =========================================================
diagnostic_records = []

for sp in dominant_species:
    c = curves[sp]
    k = c["k"]
    pan = c["pan_mean"]

    if len(k) >= 3:
        x = np.log(k[1:])
        y = np.log(np.maximum(pan[1:], 1))
        gamma = float(np.polyfit(x, y, 1)[0])
    else:
        gamma = np.nan

    part = partitions[sp]
    pan_final = max(part["pan"], 1)
    core_frac = part["core"] / pan_final
    acc_frac = 1.0 - core_frac

    diagnostic_records.append({
        "species": sp,
        "n": part["n"],
        "pan": part["pan"],
        "core": part["core"],
        "gamma": gamma,
        "core_fraction": core_frac,
        "accessory_fraction": acc_frac,
    })

diagnostic_df = pd.DataFrame(diagnostic_records)


# =========================================================
# Panel functions
# =========================================================
def draw_panel_A(ax):
    add_panel_card(ax)
    style_axis(ax, "x")

    labels = [species_short(sp) + f"\n(n={partitions[sp]['n']})" for sp in dominant_species]
    y = np.arange(len(dominant_species))

    left = np.zeros(len(dominant_species), dtype=float)

    parts = [
        ("Core", "core"),
        ("Soft-core", "soft"),
        ("Shell", "shell"),
        ("Cloud", "cloud"),
    ]

    for label, key in parts:
        vals = np.array([partitions[sp][key] for sp in dominant_species], dtype=float)
        ax.barh(
            y,
            vals,
            left=left,
            height=0.66,
            color=PART_COLORS[label],
            edgecolor="white",
            linewidth=1.0,
            label=label,
            zorder=3
        )

        for yi, v, lft in zip(y, vals, left):
            if v > max(15, vals.max() * 0.04):
                ax.text(
                    lft + v / 2,
                    yi,
                    f"{int(v):,}",
                    ha="center",
                    va="center",
                    fontsize=8.8,
                    fontweight="bold",
                    color="white" if label in {"Core", "Soft-core", "Cloud"} else "#111827",
                    zorder=4
                )

        left += vals

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.8)
    ax.invert_yaxis()
    ax.set_xlabel("Gene families / orthogroups")
    ax.set_title("A. Core/accessory genome partition", loc="left", pad=10)

    ax.legend(
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(0.995, 0.995),
        ncol=2,
        fontsize=9.0,
        handlelength=1.0,
        columnspacing=0.9,
        borderaxespad=0.2
    )


def draw_panel_B(ax):
    add_panel_card(ax)
    style_axis(ax, "both")

    for sp in dominant_species:
        c = curves[sp]
        color = SPECIES_COLORS.get(sp, "#64748b")

        ax.plot(
            c["k"],
            c["pan_mean"],
            color=color,
            lw=2.3,
            label=species_short(sp),
            zorder=4
        )
        ax.fill_between(
            c["k"],
            c["pan_lo"],
            c["pan_hi"],
            color=color,
            alpha=0.12,
            linewidth=0,
            zorder=2
        )

        # Core decay as faint dashed line
        ax.plot(
            c["k"],
            c["core_mean"],
            color=color,
            lw=1.35,
            ls="--",
            alpha=0.52,
            zorder=3
        )

    ax.set_xlabel("Genomes added")
    ax.set_ylabel("Gene families")
    ax.set_title("B. Pangenome accumulation and core-genome decay", loc="left", pad=10)

    ax.text(
        0.012,
        0.035,
        "Solid lines: pangenome accumulation; dashed lines: retained core families; shaded bands: 5th–95th percentile across random genome orders.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.7,
        color="#64748b"
    )

    ax.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.98),
        ncol=2,
        fontsize=9.0,
        handlelength=1.5,
        columnspacing=1.0
    )


def draw_panel_C(ax):
    add_panel_card(ax)

    all_samples = []
    for sp in dominant_species:
        ss = samples_by_species[sp]
        ss = sorted(
            ss,
            key=lambda s: (
                {"Priority-review": 0, "High-confidence": 1}.get(sample_to_confidence.get(s, "Other/remaining"), 2),
                s
            )
        )
        all_samples.extend(ss)

    sub = presence[all_samples].astype(bool)

    prev = sub.mean(axis=1).values
    variance = prev * (1.0 - prev)
    non_static = (prev > 0.02) & (prev < 0.98)

    candidate_idx = np.where(non_static)[0]
    if len(candidate_idx) == 0:
        candidate_idx = np.arange(sub.shape[0])

    top_n = min(360, len(candidate_idx))
    top_idx = candidate_idx[np.argsort(variance[candidate_idx])[::-1][:top_n]]

    mat = sub.iloc[top_idx, :].T.astype(int).values
    fams = list(sub.index[top_idx])

    # Cluster columns if scipy is available.
    col_order = np.arange(mat.shape[1])
    try:
        from scipy.cluster.hierarchy import linkage, leaves_list
        if mat.shape[1] > 2:
            Z = linkage(mat.T, method="average", metric="jaccard")
            col_order = leaves_list(Z)
    except Exception:
        col_order = np.argsort(mat.mean(axis=0))

    mat = mat[:, col_order]
    fams = [fams[i] for i in col_order]

    ax.imshow(
        mat,
        aspect="auto",
        cmap=PRESENCE_CMAP,
        norm=PRESENCE_NORM,
        interpolation="nearest"
    )

    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_title(
        f"C. Compressed accessory gene presence/absence architecture ({mat.shape[1]} high-variance families)",
        loc="left",
        pad=10
    )
    ax.set_xlabel("Accessory gene families / orthogroups clustered by co-presence")
    ax.set_ylabel("Samples ordered by dominant species and review class")

    # Species boundaries
    boundaries = []
    cursor = 0
    for sp in dominant_species:
        n = len(samples_by_species[sp])
        boundaries.append((cursor, cursor + n, sp))
        cursor += n

    for start, end, sp in boundaries:
        if start > 0:
            ax.axhline(start - 0.5, color="white", lw=2.0)
        ax.text(
            -0.040,
            (start + end - 1) / 2,
            species_short(sp),
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=9.2,
            fontweight="bold",
            color="#111827",
            clip_on=False,
            zorder=20
        )

    # Left species strip
    strip_colors = [
        mcolors.to_rgba(SPECIES_COLORS.get(sample_to_species.get(s, ""), "#e5e7eb"))
        for s in all_samples
    ]
    strip = np.array(strip_colors).reshape(len(all_samples), 1, 4)

    strip_ax = ax.inset_axes([-0.032, 0.0, 0.014, 1.0], transform=ax.transAxes)
    strip_ax.imshow(strip, aspect="auto", interpolation="nearest")
    strip_ax.set_xticks([])
    strip_ax.set_yticks([])
    for spn in strip_ax.spines.values():
        spn.set_visible(False)

    # Confidence strip
    conf_colors = [
        mcolors.to_rgba(CONF_COLORS.get(sample_to_confidence.get(s, "Other/remaining"), "#94a3b8"))
        for s in all_samples
    ]
    conf_strip = np.array(conf_colors).reshape(len(all_samples), 1, 4)

    conf_ax = ax.inset_axes([-0.015, 0.0, 0.014, 1.0], transform=ax.transAxes)
    conf_ax.imshow(conf_strip, aspect="auto", interpolation="nearest")
    conf_ax.set_xticks([])
    conf_ax.set_yticks([])
    for spn in conf_ax.spines.values():
        spn.set_visible(False)

    legend_handles = [
        Patch(facecolor="#111827", edgecolor="none", label="Present"),
        Patch(facecolor="#f8fafc", edgecolor="#cbd5e1", label="Absent"),
        Patch(facecolor=CONF_COLORS["Priority-review"], edgecolor="none", label="Priority-review strip"),
        Patch(facecolor=CONF_COLORS["High-confidence"], edgecolor="none", label="High-confidence strip"),
    ]
    ax.legend(
        handles=legend_handles,
        frameon=False,
        loc="upper right",
        ncol=4,
        fontsize=8.6,
        handlelength=1.0,
        columnspacing=0.8
    )


def draw_panel_D(ax):
    add_panel_card(ax)
    style_axis(ax, "y")

    rng = np.random.default_rng(42)

    data_by_species = [
        accessory_df.loc[accessory_df["species"] == sp, "accessory_count"].values
        for sp in dominant_species
    ]

    positions = np.arange(len(dominant_species))

    vp = ax.violinplot(
        data_by_species,
        positions=positions,
        widths=0.78,
        showmeans=False,
        showmedians=False,
        showextrema=False
    )

    for body, sp in zip(vp["bodies"], dominant_species):
        color = SPECIES_COLORS.get(sp, "#64748b")
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.18)
        body.set_linewidth(1.0)

    bp = ax.boxplot(
        data_by_species,
        positions=positions,
        widths=0.28,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="#111827", linewidth=1.3),
        boxprops=dict(linewidth=0.85, color="#475569"),
        whiskerprops=dict(linewidth=0.75, color="#64748b"),
        capprops=dict(linewidth=0.75, color="#64748b"),
    )

    for patch, sp in zip(bp["boxes"], dominant_species):
        patch.set_facecolor(SPECIES_COLORS.get(sp, "#64748b"))
        patch.set_alpha(0.35)

    for x0, sp in zip(positions, dominant_species):
        tmp = accessory_df[accessory_df["species"] == sp]
        jitter = rng.normal(0, 0.055, size=len(tmp))

        for j, (_, r) in enumerate(tmp.iterrows()):
            cc = r["confidence"]
            ax.scatter(
                x0 + jitter[j],
                r["accessory_count"],
                s=30 if cc == "Priority-review" else 22,
                color=CONF_COLORS.get(cc, "#94a3b8"),
                edgecolor="white",
                linewidth=0.45,
                alpha=0.88,
                zorder=4
            )

        if len(tmp):
            med = float(np.median(tmp["accessory_count"]))
            ax.text(
                x0,
                med,
                f"{med:,.0f}",
                ha="center",
                va="center",
                fontsize=8.5,
                fontweight="bold",
                color="white",
                bbox=dict(boxstyle="round,pad=0.18", fc="#111827", ec="white", alpha=0.86),
                zorder=5
            )

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [species_short(sp) + f"\n(n={len(samples_by_species[sp])})" for sp in dominant_species],
        fontsize=9.2
    )
    ax.set_ylabel("Accessory gene families per genome")
    ax.set_title("D. Per-sample accessory genome burden", loc="left", pad=10)

    handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=CONF_COLORS["Priority-review"],
               markeredgecolor="white", markersize=7, label="Priority-review"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=CONF_COLORS["High-confidence"],
               markeredgecolor="white", markersize=7, label="High-confidence"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=CONF_COLORS["Other/remaining"],
               markeredgecolor="white", markersize=7, label="Other/remaining"),
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        loc="upper right",
        fontsize=8.8,
        handletextpad=0.35
    )


def draw_panel_E(ax):
    add_panel_card(ax)
    style_axis(ax, "x")

    cats = list(CATEGORY_COLORS.keys())
    y = np.arange(len(dominant_species))
    left = np.zeros(len(dominant_species), dtype=float)

    totals = []
    for sp in dominant_species:
        totals.append(sum(category_by_species[sp].values()))
    totals = np.array(totals, dtype=float)
    totals[totals == 0] = 1.0

    for cat in cats:
        vals = np.array([category_by_species[sp].get(cat, 0) for sp in dominant_species], dtype=float)
        pct = vals / totals * 100.0

        ax.barh(
            y,
            pct,
            left=left,
            height=0.66,
            color=CATEGORY_COLORS[cat],
            edgecolor="white",
            linewidth=0.75,
            label=cat,
            zorder=3
        )

        left += pct

    ax.set_yticks(y)
    ax.set_yticklabels(
        [species_short(sp) for sp in dominant_species],
        fontsize=9.8
    )
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Accessory families by annotation category (%)")
    ax.set_title("E. Accessory-gene functional category summary", loc="left", pad=10)

    ax.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.36),
        ncol=2,
        fontsize=8.1,
        handlelength=1.0,
        columnspacing=0.9
    )


def draw_panel_F(ax):
    add_panel_card(ax)
    style_axis(ax, "both")

    x = diagnostic_df["gamma"].values.astype(float)
    y = diagnostic_df["core_fraction"].values.astype(float)
    n = diagnostic_df["n"].values.astype(float)

    sizes = 120 + 13 * n

    for _, r in diagnostic_df.iterrows():
        sp = r["species"]
        ax.scatter(
            r["gamma"],
            r["core_fraction"],
            s=120 + 13 * r["n"],
            color=SPECIES_COLORS.get(sp, "#64748b"),
            edgecolor="#111827",
            linewidth=0.9,
            alpha=0.86,
            zorder=4
        )
        ax.annotate(
            species_short(sp),
            (r["gamma"], r["core_fraction"]),
            xytext=(9, 6),
            textcoords="offset points",
            fontsize=8.7,
            fontweight="bold",
            color="#111827",
            bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="#e2e8f0", alpha=0.92),
            zorder=5
        )

    ax.set_xlabel("Pangenome expansion coefficient")
    ax.set_ylabel("Final core fraction")
    ax.set_title("F. Pangenome openness and core stability", loc="left", pad=10)

    if np.isfinite(x).any():
        ax.set_xlim(max(0, np.nanmin(x) - 0.05), np.nanmax(x) + 0.08)
    ax.set_ylim(0, min(1.0, max(0.1, np.nanmax(y) + 0.12)))

    ax.text(
        0.025,
        0.035,
        "Higher expansion coefficient suggests a more open accessory repertoire;",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.2,
        color="#64748b"
    )

    legend_vals = sorted(set([int(np.nanmin(n)), int(np.nanmedian(n)), int(np.nanmax(n))]))
    legend_vals = [v for v in legend_vals if v > 0]
    handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="",
            markerfacecolor="#cbd5e1",
            markeredgecolor="#111827",
            markersize=math.sqrt(120 + 13 * v) / 1.65,
            label=f"n={v}"
        )
        for v in legend_vals
    ]
    ax.legend(
        handles=handles,
        title="Species genomes",
        frameon=False,
        loc="upper right",
        fontsize=8.4,
        title_fontsize=8.7
    )


# =========================================================
# Figure layout
# =========================================================
fig = plt.figure(figsize=(26.0, 18.0), facecolor="white")

gs = GridSpec(
    3,
    3,
    figure=fig,
    width_ratios=[1.05, 1.12, 0.92],
    height_ratios=[0.92, 1.28, 1.02],
    hspace=0.38,
    wspace=0.26
)

axA = fig.add_subplot(gs[0, 0])
draw_panel_A(axA)

axB = fig.add_subplot(gs[0, 1:])
draw_panel_B(axB)

axC = fig.add_subplot(gs[1, :])
draw_panel_C(axC)

axD = fig.add_subplot(gs[2, 0])
draw_panel_D(axD)

axE = fig.add_subplot(gs[2, 1])
draw_panel_E(axE)

axF = fig.add_subplot(gs[2, 2])
draw_panel_F(axF)


# =========================================================
# Figure title and footer
# =========================================================
fig.suptitle(
    "Pangenome and core-accessory genome architecture across dominant species",
    y=0.992,
    fontsize=25.5,
    fontweight="bold"
)

fig.subplots_adjust(
    left=0.070,
    right=0.975,
    top=0.925,
    bottom=0.055
)

save_png_pdf(fig, OUTDIR, OUTNAME)

print("Saved pangenome figure to:", OUTDIR)
print("Included dominant species:", ", ".join(dominant_species))
print("Source note:", source_note)