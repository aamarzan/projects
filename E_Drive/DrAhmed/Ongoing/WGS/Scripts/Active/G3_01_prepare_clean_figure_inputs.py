import os, csv, re
from collections import Counter, defaultdict

WORK   = "/mnt/e/DrAhmed/Ongoing/WGS/Result"
RAW    = f"{WORK}/_FIGURE_RAW_MATERIALS_V2"
MASTER = f"{WORK}/_BIOLOGY_LAYER_V2/PrimaryResults_v4_withBiology_166.csv"
OUTDIR = f"{WORK}/_G3/clean"
os.makedirs(OUTDIR, exist_ok=True)

MLST_IN = f"{RAW}/mlst_species_st_counts_166.csv"

def read_csv(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))

def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def as_int(x, default=0):
    try:
        s = str(x).strip()
        if s == "":
            return default
        return int(float(s))
    except:
        return default

def norm(s):
    return str(s).strip()

def root_from_scheme_st(scheme_st):
    s = norm(scheme_st)
    if "|" in s:
        return s.split("|", 1)[0].strip().lower()
    return s.lower()

def valid_root_for_species(species, root):
    species = norm(species)
    root = root.lower()

    allowed = {
        "Acinetobacter baumannii": ["abaumannii"],
        "Klebsiella pneumoniae": ["klebsiella", "kpneumoniae", "kleb"],
        "Pseudomonas aeruginosa": ["paeruginosa", "pseudomonas"],
        "Escherichia coli": ["escherichia", "ecoli"],
        "Serratia marcescens": ["serratia"],
        "Serratia nevei": ["serratia"],
        "Homo sapiens": [],
    }

    vals = allowed.get(species, [])
    return any(root == a or root.startswith(a) for a in vals)

def split_semicolon_field(x):
    if x is None:
        return []
    s = str(x).strip()
    if s == "" or s.lower() in {"na", "none", "no", "nan"}:
        return []
    return [p.strip() for p in s.split(";") if p.strip()]

def parse_token_and_count(token):
    """
    Accept both:
      IncFIB(K)_1_Kpn3
      IncFIB(K)_1_Kpn3(2)
    """
    token = token.strip()
    m = re.match(r"^(.*?)(?:\((\d+)\))?$", token)
    if not m:
        return token, 1
    name = m.group(1).strip()
    n = int(m.group(2)) if m.group(2) else 1
    return name, n

if not os.path.isfile(MASTER):
    raise SystemExit(f"Missing master file: {MASTER}")
if not os.path.isfile(MLST_IN):
    raise SystemExit(f"Missing MLST file: {MLST_IN}")

master_rows = read_csv(MASTER)
mlst_rows = read_csv(MLST_IN)

# --------------------------------------------------
# 1) Clean MLST table for Figure 3
# --------------------------------------------------
mlst_keep = []
mlst_drop = []

for r in mlst_rows:
    sp = norm(r.get("TopSpecies", ""))
    scheme_st = norm(r.get("Scheme_ST", ""))
    count = as_int(r.get("Count", 0))
    root = root_from_scheme_st(scheme_st)

    if valid_root_for_species(sp, root):
        mlst_keep.append({
            "TopSpecies": sp,
            "Scheme_ST": scheme_st,
            "Count": count,
            "SchemeRoot": root
        })
    else:
        mlst_drop.append({
            "TopSpecies": sp,
            "Scheme_ST": scheme_st,
            "Count": count,
            "SchemeRoot": root,
            "DropReason": "scheme_root_not_consistent_with_species"
        })

write_csv(
    f"{OUTDIR}/mlst_species_st_counts_clean_166.csv",
    mlst_keep,
    ["TopSpecies", "Scheme_ST", "Count", "SchemeRoot"]
)

write_csv(
    f"{OUTDIR}/mlst_species_st_counts_dropped_166.csv",
    mlst_drop,
    ["TopSpecies", "Scheme_ST", "Count", "SchemeRoot", "DropReason"]
)

# --------------------------------------------------
# 2) Rebuild plasmid replicon by species from MASTER
# --------------------------------------------------
plasmid_counter = Counter()

for r in master_rows:
    sp = norm(r.get("TopSpecies1", ""))
    reps = split_semicolon_field(r.get("Plasmid_Replicons", ""))
    if not sp:
        continue
    for token in reps:
        name, n = parse_token_and_count(token)
        if name:
            plasmid_counter[(sp, name)] += n

plasmid_rows = []
for (sp, rep), n in sorted(plasmid_counter.items(), key=lambda x: (x[0][0], -x[1], x[0][1])):
    plasmid_rows.append({
        "TopSpecies": sp,
        "Plasmid_Replicon": rep,
        "Count": n
    })

write_csv(
    f"{OUTDIR}/plasmid_replicon_by_species_166.csv",
    plasmid_rows,
    ["TopSpecies", "Plasmid_Replicon", "Count"]
)

# --------------------------------------------------
# 3) Rebuild Figure 6 prevalence directly from MASTER
# --------------------------------------------------
species_n = Counter()
vf_n = Counter()
pl_n = Counter()

for r in master_rows:
    sp = norm(r.get("TopSpecies1", ""))
    if not sp:
        continue
    species_n[sp] += 1
    if as_int(r.get("VFDB_Hits", 0)) > 0:
        vf_n[sp] += 1
    if as_int(r.get("Plasmid_Hits", 0)) > 0:
        pl_n[sp] += 1

prev_rows = []
for sp in sorted(species_n):
    prev_rows.append({
        "TopSpecies": sp,
        "Samples_n": species_n[sp],
        "Samples_with_VFDB_hits": vf_n[sp],
        "Samples_with_Plasmid_hits": pl_n[sp]
    })

write_csv(
    f"{OUTDIR}/species_feature_prevalence_from_master_166.csv",
    prev_rows,
    ["TopSpecies", "Samples_n", "Samples_with_VFDB_hits", "Samples_with_Plasmid_hits"]
)

# --------------------------------------------------
# 4) Rebuild informative serotype file from MASTER
# --------------------------------------------------
sero_rows = []
for r in master_rows:
    sample = norm(r.get("Sample", ""))
    inf = norm(r.get("Serotype_Informative", ""))
    call = norm(r.get("Serotype_Call", ""))
    if not sample:
        continue
    if inf.lower() in {"yes", "true", "1"} and call and call.lower() not in {"na", "none", "nan"}:
        sero_rows.append({
            "Sample": sample,
            "Serotype_Call": call
        })

write_csv(
    f"{OUTDIR}/serotype_distribution_166.csv",
    sero_rows,
    ["Sample", "Serotype_Call"]
)

# --------------------------------------------------
# 5) Summary
# --------------------------------------------------
with open(f"{OUTDIR}/README.txt", "w", encoding="utf-8") as f:
    f.write("Clean G3 inputs prepared.\n")
    f.write(f"Master rows: {len(master_rows)}\n")
    f.write(f"MLST input rows: {len(mlst_rows)}\n")
    f.write(f"MLST kept rows: {len(mlst_keep)}\n")
    f.write(f"MLST dropped rows: {len(mlst_drop)}\n")
    f.write(f"Plasmid replicon rows rebuilt: {len(plasmid_rows)}\n")
    f.write(f"Serotype rows rebuilt: {len(sero_rows)}\n")
    f.write("\nFiles:\n")
    f.write("- mlst_species_st_counts_clean_166.csv\n")
    f.write("- mlst_species_st_counts_dropped_166.csv\n")
    f.write("- plasmid_replicon_by_species_166.csv\n")
    f.write("- species_feature_prevalence_from_master_166.csv\n")
    f.write("- serotype_distribution_166.csv\n")

print("Wrote:", f"{OUTDIR}/mlst_species_st_counts_clean_166.csv")
print("Wrote:", f"{OUTDIR}/mlst_species_st_counts_dropped_166.csv")
print("Wrote:", f"{OUTDIR}/plasmid_replicon_by_species_166.csv")
print("Wrote:", f"{OUTDIR}/species_feature_prevalence_from_master_166.csv")
print("Wrote:", f"{OUTDIR}/serotype_distribution_166.csv")
print("Wrote:", f"{OUTDIR}/README.txt")
print("MLST kept:", len(mlst_keep))
print("MLST dropped:", len(mlst_drop))
print("Plasmid replicon rows rebuilt:", len(plasmid_rows))
print("Serotype rows rebuilt:", len(sero_rows))
