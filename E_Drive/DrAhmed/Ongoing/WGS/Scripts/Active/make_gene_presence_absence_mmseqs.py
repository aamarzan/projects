# =========================================================
# Make gene_presence_absence.csv from Prokka .faa.gz files
# MMseqs2-based orthogroup clustering
# =========================================================

from importlib.resources import path
import os
import re
import csv
import gzip
import shutil
import subprocess
from pathlib import Path
from collections import defaultdict, Counter

# =========================================================
# Paths
# =========================================================
WORK = Path("/mnt/e/DrAhmed/Ongoing/WGS/Result")

BACTOPIA_RESULTS = WORK / "Result copy" / "bactopia_Results" / "results"

OUTROOT = WORK / "_PANGENOME" / "mmseqs"
FAA_DIR = OUTROOT / "faa_all"
TMP_DIR = OUTROOT / "tmp"
MMSEQS_DIR = OUTROOT / "mmseqs_work"

OUTROOT.mkdir(parents=True, exist_ok=True)
FAA_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)
MMSEQS_DIR.mkdir(parents=True, exist_ok=True)

COMBINED_FASTA = OUTROOT / "all_proteins.faa"
CLUSTER_TSV = OUTROOT / "clusters.tsv"
OUT_CSV = OUTROOT / "gene_presence_absence.csv"
SUMMARY_CSV = OUTROOT / "gene_presence_absence_summary.csv"
BAD_FASTA_REPORT = OUTROOT / "bad_or_truncated_faa_files.csv"
bad_faa_files = []

# =========================================================
# Parameters
# =========================================================
# For bacterial pangenome:
# 0.90 = strict orthogroup-like clustering.
# If too fragmented, try 0.80.
MIN_SEQ_ID = "0.90"

# Coverage mode:
# 0 = coverage of query and target.
# 1 = target coverage.
# 2 = query coverage.
COV_MODE = "0"
COVERAGE = "0.80"

THREADS = "8"


# =========================================================
# Helpers
# =========================================================
def sample_from_path(path: Path) -> str:
    """
    Expected Bactopia Prokka path:
    .../results/SAMPLE/SAMPLE/main/annotator/prokka/SAMPLE.faa.gz
    """
    name = path.name
    name = re.sub(r"\.faa\.gz$", "", name)
    name = re.sub(r"\.faa$", "", name)
    return name


def species_from_sample(sample: str) -> str:
    s = sample.strip()

    if s.startswith("SM") or s.startswith("SM_") or s.startswith("SM-"):
        return "Serratia marcescens"
    if s.startswith("AB") or s.startswith("AB_") or s.startswith("AB-"):
        return "Acinetobacter baumannii"
    if s.startswith("KP") or s.startswith("KP_") or s.startswith("KP-"):
        return "Klebsiella pneumoniae"
    if s.startswith("PA") or s.startswith("PA_") or s.startswith("PA-"):
        return "Pseudomonas aeruginosa"
    if s.startswith("EC") or s.startswith("EC_") or s.startswith("EC-"):
        return "Escherichia coli"
    if s.startswith("VIM_pseudomonas"):
        return "Pseudomonas aeruginosa"

    return "Unknown"


def clean_header_id(header: str) -> str:
    header = header.strip()
    if header.startswith(">"):
        header = header[1:]
    return header.split()[0]


def clean_annotation(header: str) -> str:
    """
    Prokka FAA headers often look like:
    >locus_tag product description
    """
    header = header.strip()
    if header.startswith(">"):
        header = header[1:]

    parts = header.split(maxsplit=1)
    if len(parts) == 2:
        return parts[1].strip()
    return ""


def find_faa_files():
    files = []

    for p in BACTOPIA_RESULTS.rglob("*.faa.gz"):
        if "annotator/prokka" in str(p).replace("\\", "/"):
            files.append(p)

    for p in BACTOPIA_RESULTS.rglob("*.faa"):
        if "annotator/prokka" in str(p).replace("\\", "/"):
            files.append(p)

    files = sorted(files)
    return files


def read_fasta_any(path: Path):
    """
    Robust FASTA reader for .faa and .faa.gz files.
    If a gzip file is truncated/corrupted, it returns all records read before
    the crash and reports the file instead of stopping the whole run.
    """
    opener = gzip.open if str(path).endswith(".gz") else open

    header = None
    seq_chunks = []

    try:
        with opener(path, "rt", encoding="utf-8", errors="replace") as f:
            while True:
                try:
                    line = f.readline()
                except EOFError:
                    print(f"[WARNING] Truncated gzip file detected, using partial records only: {path}")
                    bad_faa_files.append(str(path))
                    break

                if not line:
                    break

                line = line.rstrip("\n")
                if not line:
                    continue

                if line.startswith(">"):
                    if header is not None:
                        yield header, "".join(seq_chunks)
                    header = line
                    seq_chunks = []
                else:
                    seq_chunks.append(line.strip())

    except EOFError:
        print(f"[WARNING] Could not fully read truncated gzip file: {path}")
    except OSError as e:
        print(f"[WARNING] Skipping unreadable FASTA file: {path} | {e}")
        bad_faa_files.append(str(path))
        return

    if header is not None and seq_chunks:
        yield header, "".join(seq_chunks)


def run(cmd, cwd=None):
    print("\n[CMD]", " ".join(str(x) for x in cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


# =========================================================
# Check MMseqs2
# =========================================================
if shutil.which("mmseqs") is None:
    raise RuntimeError(
        "MMseqs2 was not found. Install it first in WSL:\n"
        "  conda install -c bioconda mmseqs2 -y\n"
        "or:\n"
        "  mamba install -c bioconda mmseqs2 -y"
    )


# =========================================================
# Step 1: discover Prokka protein files
# =========================================================
faa_files = find_faa_files()

if not faa_files:
    raise RuntimeError(f"No Prokka .faa/.faa.gz files found under: {BACTOPIA_RESULTS}")

print(f"[INFO] Found FAA files: {len(faa_files)}")


# =========================================================
# Step 2: build combined protein FASTA with sample-aware IDs
# =========================================================
protein_to_sample = {}
protein_to_original_id = {}
protein_to_annotation = {}
sample_to_species = {}

n_proteins = 0
samples_seen = set()

with open(COMBINED_FASTA, "w", encoding="utf-8", newline="\n") as out:
    for faa in faa_files:
        sample = sample_from_path(faa)
        species = species_from_sample(sample)

        if species == "Unknown":
            continue

        samples_seen.add(sample)
        sample_to_species[sample] = species

        local_i = 0
        for header, seq in read_fasta_any(faa):
            seq = seq.replace("*", "").strip()

            if len(seq) < 20:
                continue

            local_i += 1
            n_proteins += 1

            original_id = clean_header_id(header)
            annotation = clean_annotation(header)

            new_id = f"{sample}|prot_{local_i:06d}|{original_id}"

            protein_to_sample[new_id] = sample
            protein_to_original_id[new_id] = original_id
            protein_to_annotation[new_id] = annotation

            out.write(f">{new_id}\n")
            for i in range(0, len(seq), 80):
                out.write(seq[i:i+80] + "\n")

print(f"[INFO] Samples included: {len(samples_seen)}")
print(f"[INFO] Proteins included: {n_proteins}")
print(f"[INFO] Combined FASTA: {COMBINED_FASTA}")


# =========================================================
# Step 3: run MMseqs easy-cluster
# =========================================================
cluster_prefix = MMSEQS_DIR / "pan_clusters"

if CLUSTER_TSV.exists():
    print(f"[INFO] Existing cluster TSV found, using: {CLUSTER_TSV}")
else:
    run([
        "mmseqs",
        "easy-cluster",
        str(COMBINED_FASTA),
        str(cluster_prefix),
        str(TMP_DIR),
        "--min-seq-id", MIN_SEQ_ID,
        "-c", COVERAGE,
        "--cov-mode", COV_MODE,
        "--threads", THREADS,
    ])

    mmseqs_tsv = Path(str(cluster_prefix) + "_cluster.tsv")

    if not mmseqs_tsv.exists():
        raise RuntimeError(f"MMseqs cluster TSV not found: {mmseqs_tsv}")

    shutil.copy2(mmseqs_tsv, CLUSTER_TSV)

print(f"[INFO] Cluster TSV: {CLUSTER_TSV}")


# =========================================================
# Step 4: parse MMseqs clusters
# =========================================================
cluster_to_members = defaultdict(list)

with open(CLUSTER_TSV, "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 2:
            continue

        rep, member = parts[0], parts[1]
        cluster_to_members[rep].append(member)

print(f"[INFO] Orthogroup clusters: {len(cluster_to_members)}")


# =========================================================
# Step 5: create Roary-like gene_presence_absence.csv
# =========================================================
samples = sorted(samples_seen, key=lambda x: (species_from_sample(x), x))

header_cols = [
    "Gene",
    "Non-unique Gene name",
    "Annotation",
    "No. isolates",
    "No. sequences",
    "Avg sequences per isolate",
    "Genome Fragment",
    "Order within Fragment",
    "Accessory Fragment",
    "Accessory Order with Fragment",
    "QC",
    "Min group size nuc",
    "Max group size nuc",
    "Avg group size nuc",
    "Min group size aa",
    "Max group size aa",
    "Avg group size aa",
]

rows = []

for idx, (rep, members) in enumerate(cluster_to_members.items(), start=1):
    sample_to_loci = defaultdict(list)
    annotations = []

    for m in members:
        sample = protein_to_sample.get(m, "")
        original = protein_to_original_id.get(m, m)
        annot = protein_to_annotation.get(m, "")

        if sample:
            sample_to_loci[sample].append(original)
        if annot:
            annotations.append(annot)

    if not sample_to_loci:
        continue

    annotation = ""
    if annotations:
        annotation = Counter(annotations).most_common(1)[0][0]

    no_isolates = len(sample_to_loci)
    no_sequences = sum(len(v) for v in sample_to_loci.values())
    avg_seq = no_sequences / max(no_isolates, 1)

    gene_name = f"OG{idx:07d}"

    row = {
        "Gene": gene_name,
        "Non-unique Gene name": gene_name,
        "Annotation": annotation,
        "No. isolates": no_isolates,
        "No. sequences": no_sequences,
        "Avg sequences per isolate": f"{avg_seq:.3f}",
        "Genome Fragment": "",
        "Order within Fragment": "",
        "Accessory Fragment": "",
        "Accessory Order with Fragment": "",
        "QC": "",
        "Min group size nuc": "",
        "Max group size nuc": "",
        "Avg group size nuc": "",
        "Min group size aa": "",
        "Max group size aa": "",
        "Avg group size aa": "",
    }

    for sample in samples:
        loci = sample_to_loci.get(sample, [])
        row[sample] = "\t".join(loci) if loci else ""

    rows.append(row)

# Sort: most prevalent families first
rows.sort(key=lambda r: (-int(r["No. isolates"]), r["Gene"]))

with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=header_cols + samples)
    writer.writeheader()
    writer.writerows(rows)

print(f"[DONE] Written: {OUT_CSV}")
print(f"[DONE] Rows/orthogroups: {len(rows)}")
print(f"[DONE] Samples: {len(samples)}")


# =========================================================
# Step 6: summary table
# =========================================================
species_counts = Counter(sample_to_species.get(s, "Unknown") for s in samples)

with open(SUMMARY_CSV, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["metric", "value"])
    w.writerow(["samples", len(samples)])
    w.writerow(["proteins", n_proteins])
    w.writerow(["orthogroups", len(rows)])
    w.writerow(["min_seq_id", MIN_SEQ_ID])
    w.writerow(["coverage", COVERAGE])
    w.writerow(["cov_mode", COV_MODE])
    for sp, n in species_counts.items():
        w.writerow([f"species_samples::{sp}", n])
    if bad_faa_files:
        with open(BAD_FASTA_REPORT, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["bad_or_truncated_faa_file"])
            for p in sorted(set(bad_faa_files)):
                w.writerow([p])

    print(f"[WARNING] Bad/truncated FAA file report written: {BAD_FASTA_REPORT}")
print(f"[DONE] Summary: {SUMMARY_CSV}")