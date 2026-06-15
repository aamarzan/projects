import os, csv, re, math
from collections import Counter, defaultdict

WORK = "/mnt/e/DrAhmed/Ongoing/WGS/Result"
MASTER = f"{WORK}/_BIOLOGY_LAYER_V2/PrimaryResults_v4_withBiology_166.csv"
OUTDIR = f"{WORK}/_FIGURE_RAW_MATERIALS_V2"

HIGHCONF_CANDIDATES = [
    f"{WORK}/_PRELIMINARY_REPORT_READY/Table5_HighConfidenceSamples.csv",
    f"{WORK}/_MANUSCRIPT_FINAL/Table5_HighConfidenceSamples.csv",
]
PRIORITY_CANDIDATES = [
    f"{WORK}/_PRELIMINARY_REPORT_READY/Table4_PriorityReviewSamples.csv",
    f"{WORK}/_PRELIMINARY_PACK/Table_FlaggedSamples_refined.csv",
]

os.makedirs(OUTDIR, exist_ok=True)

def read_csv(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))

def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def first_existing(paths):
    for p in paths:
        if os.path.isfile(p):
            return p
    return None

def sample_set_from_csv(path):
    if not path or not os.path.isfile(path):
        return set()
    rows = read_csv(path)
    if not rows:
        return set()
    possible = ["Sample", "sample", "SAMPLE", "SampleID", "Sample_ID"]
    col = None
    for c in possible:
        if c in rows[0]:
            col = c
            break
    if col is None:
        col = list(rows[0].keys())[0]
    out = set()
    for r in rows:
        s = (r.get(col) or "").strip()
        if s:
            out.add(s)
    return out

def as_float(x, default=0.0):
    try:
        s = str(x).strip()
        if s == "":
            return default
        return float(s)
    except:
        return default

def as_int(x, default=0):
    try:
        s = str(x).strip()
        if s == "":
            return default
        return int(float(s))
    except:
        return default

def median(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return 0
    vals = sorted(vals)
    n = len(vals)
    m = n // 2
    if n % 2 == 1:
        return vals[m]
    return (vals[m - 1] + vals[m]) / 2

def split_semicolon_field(x):
    x = (x or "").strip()
    if not x:
        return []
    return [p.strip() for p in x.split(";") if p.strip()]

def parse_counted_items(x):
    """
    'blaOXA-66(1); blaADC-30(1)' -> list of (item, count)
    If count missing, assume 1.
    """
    items = []
    for part in split_semicolon_field(x):
        m = re.match(r"^(.*?)(?:\((\d+)\))?$", part)
        if not m:
            continue
        name = (m.group(1) or "").strip()
        cnt = int(m.group(2)) if m.group(2) else 1
        if name:
            items.append((name, cnt))
    return items

master_rows = read_csv(MASTER)
highconf_file = first_existing(HIGHCONF_CANDIDATES)
priority_file = first_existing(PRIORITY_CANDIDATES)

highconf = sample_set_from_csv(highconf_file)
priority = sample_set_from_csv(priority_file)

# ----------------------------
# frozen sample-level master for figures
# ----------------------------
feature_rows = []
species_counter = Counter()
species_conf_counter = defaultdict(lambda: {"HighConfidence": 0, "PriorityReview": 0, "Other": 0})

amr_class_by_species = defaultdict(Counter)
amr_gene_by_species = defaultdict(Counter)
mlst_st_by_species = defaultdict(Counter)
vfdb_gene_by_species = defaultdict(Counter)
plasmid_gene_by_species = defaultdict(Counter)
serotype_counter = Counter()

species_summary_tmp = defaultdict(list)

for r in master_rows:
    sample = (r.get("Sample") or "").strip()
    sp = (r.get("TopSpecies1") or "").strip()
    st_scheme = (r.get("MLST_Scheme") or "").strip()
    st = (r.get("MLST_ST") or "").strip()

    amr_hits = as_int(r.get("AMRFinder_Hits"), 0)
    vfdb_hits = as_int(r.get("VFDB_Hits"), 0)
    plasmid_hits = as_int(r.get("Plasmid_Hits"), 0)

    amr_classes = r.get("AMR_Classes", "")
    amr_topgenes = r.get("AMR_TopGenes", "")
    vfdb_genes = r.get("VFDB_Genes", "")
    plasmid_reps = r.get("Plasmid_Replicons", "")
    serotype_call = (r.get("Serotype_Call") or "").strip()
    informative = (r.get("Serotype_Informative") or "").strip().lower() == "yes"

    hc = "Yes" if sample in highconf else "No"
    pr = "Yes" if sample in priority else "No"

    if sp:
        species_counter[sp] += 1
        if hc == "Yes":
            species_conf_counter[sp]["HighConfidence"] += 1
        elif pr == "Yes":
            species_conf_counter[sp]["PriorityReview"] += 1
        else:
            species_conf_counter[sp]["Other"] += 1

    feature_rows.append({
        "Sample": sample,
        "TopSpecies": sp,
        "HighConfidence": hc,
        "PriorityReview": pr,
        "ST": st,
        "MLST_Scheme": st_scheme,
        "AMR_Genes_n": amr_hits,
        "AMR_Classes_n": len(parse_counted_items(amr_classes)),
        "VFDB_Hits_n": vfdb_hits,
        "Plasmid_Hits_n": plasmid_hits,
        "Serotype": serotype_call if informative else ""
    })

    if sp and st_scheme and st and st != "-":
        mlst_st_by_species[sp][f"{st_scheme} | ST{st}"] += 1

    for cls, cnt in parse_counted_items(amr_classes):
        if sp:
            amr_class_by_species[sp][cls] += cnt

    for gene, cnt in parse_counted_items(amr_topgenes):
        if sp:
            amr_gene_by_species[sp][gene] += cnt

    for gene, cnt in parse_counted_items(vfdb_genes):
        if sp:
            vfdb_gene_by_species[sp][gene] += cnt

    for rep, cnt in parse_counted_items(plasmid_reps):
        if sp:
            plasmid_gene_by_species[sp][rep] += cnt

    if informative and serotype_call:
        serotype_counter[serotype_call] += 1

    species_summary_tmp[sp].append({
        "HC": 1 if hc == "Yes" else 0,
        "PR": 1 if pr == "Yes" else 0,
        "AMR_Genes_n": amr_hits,
        "AMR_Classes_n": len(parse_counted_items(amr_classes)),
        "VFDB_Hits_n": vfdb_hits,
        "Plasmid_Hits_n": plasmid_hits,
    })

# species counts
species_rows = [{"TopSpecies": k, "Count": v} for k, v in species_counter.most_common()]
write_csv(
    f"{OUTDIR}/species_counts_166.csv",
    species_rows,
    ["TopSpecies", "Count"]
)

# confidence counts by species
conf_rows = []
for sp, c in sorted(species_conf_counter.items(), key=lambda x: (-species_counter[x[0]], x[0])):
    conf_rows.append({
        "TopSpecies": sp,
        "HighConfidence_n": c["HighConfidence"],
        "PriorityReview_n": c["PriorityReview"],
        "Other_n": c["Other"],
    })
write_csv(
    f"{OUTDIR}/species_confidence_counts_166.csv",
    conf_rows,
    ["TopSpecies", "HighConfidence_n", "PriorityReview_n", "Other_n"]
)

# species biology summary
summary_rows = []
for sp, vals in sorted(species_summary_tmp.items(), key=lambda x: (-species_counter[x[0]], x[0])):
    summary_rows.append({
        "TopSpecies": sp,
        "Samples_n": len(vals),
        "HighConfidence_n": sum(v["HC"] for v in vals),
        "PriorityReview_n": sum(v["PR"] for v in vals),
        "Median_AMR_Genes_n": median([v["AMR_Genes_n"] for v in vals]),
        "Median_AMR_Classes_n": median([v["AMR_Classes_n"] for v in vals]),
        "Median_VFDB_Hits_n": median([v["VFDB_Hits_n"] for v in vals]),
        "Median_Plasmid_Hits_n": median([v["Plasmid_Hits_n"] for v in vals]),
        "Max_AMR_Genes_n": max(v["AMR_Genes_n"] for v in vals),
        "Max_VFDB_Hits_n": max(v["VFDB_Hits_n"] for v in vals),
        "Max_Plasmid_Hits_n": max(v["Plasmid_Hits_n"] for v in vals),
    })
write_csv(
    f"{OUTDIR}/species_biology_summary_166.csv",
    summary_rows,
    [
        "TopSpecies","Samples_n","HighConfidence_n","PriorityReview_n",
        "Median_AMR_Genes_n","Median_AMR_Classes_n","Median_VFDB_Hits_n","Median_Plasmid_Hits_n",
        "Max_AMR_Genes_n","Max_VFDB_Hits_n","Max_Plasmid_Hits_n"
    ]
)

# long sample matrix
write_csv(
    f"{OUTDIR}/sample_feature_matrix_166.csv",
    feature_rows,
    ["Sample","TopSpecies","HighConfidence","PriorityReview","ST","MLST_Scheme",
     "AMR_Genes_n","AMR_Classes_n","VFDB_Hits_n","Plasmid_Hits_n","Serotype"]
)

# MLST species/ST counts
mlst_rows = []
for sp, cnts in sorted(mlst_st_by_species.items(), key=lambda x: (-species_counter[x[0]], x[0])):
    for scheme_st, cnt in cnts.most_common():
        mlst_rows.append({"TopSpecies": sp, "Scheme_ST": scheme_st, "Count": cnt})
write_csv(
    f"{OUTDIR}/mlst_species_st_counts_166.csv",
    mlst_rows,
    ["TopSpecies","Scheme_ST","Count"]
)

# AMR class by species
amr_class_rows = []
for sp, cnts in sorted(amr_class_by_species.items(), key=lambda x: (-species_counter[x[0]], x[0])):
    for cls, cnt in cnts.most_common():
        amr_class_rows.append({"TopSpecies": sp, "AMR_Class": cls, "Count": cnt})
write_csv(
    f"{OUTDIR}/amr_class_by_species_166.csv",
    amr_class_rows,
    ["TopSpecies","AMR_Class","Count"]
)

# AMR gene top200
amr_gene_rows = []
for sp, cnts in sorted(amr_gene_by_species.items(), key=lambda x: (-species_counter[x[0]], x[0])):
    for gene, cnt in cnts.most_common(200):
        amr_gene_rows.append({"TopSpecies": sp, "AMR_Gene": gene, "Count": cnt})
write_csv(
    f"{OUTDIR}/amr_gene_by_species_top200_166.csv",
    amr_gene_rows,
    ["TopSpecies","AMR_Gene","Count"]
)

# VFDB gene top200
vfdb_rows = []
for sp, cnts in sorted(vfdb_gene_by_species.items(), key=lambda x: (-species_counter[x[0]], x[0])):
    for gene, cnt in cnts.most_common(200):
        vfdb_rows.append({"TopSpecies": sp, "VFDB_Gene": gene, "Count": cnt})
write_csv(
    f"{OUTDIR}/virulence_gene_by_species_top200_166.csv",
    vfdb_rows,
    ["TopSpecies","VFDB_Gene","Count"]
)

# plasmid replicon top200
plasmid_rows = []
for sp, cnts in sorted(plasmid_gene_by_species.items(), key=lambda x: (-species_counter[x[0]], x[0])):
    for rep, cnt in cnts.most_common(200):
        plasmid_rows.append({"TopSpecies": sp, "Plasmid_Replicon": rep, "Count": cnt})
write_csv(
    f"{OUTDIR}/plasmid_gene_by_species_top200_166.csv",
    plasmid_rows,
    ["TopSpecies","Plasmid_Replicon","Count"]
)

# serotype counts
sero_rows = [{"Serotype": k, "Count": v} for k, v in serotype_counter.most_common()]
write_csv(
    f"{OUTDIR}/serotype_counts_166.csv",
    sero_rows,
    ["Serotype","Count"]
)

# freeze master
write_csv(
    f"{OUTDIR}/figure_master_frozen_166.csv",
    master_rows,
    list(master_rows[0].keys()) if master_rows else []
)

# sources
sources = [
    {"Resource":"MASTER", "Path":MASTER},
    {"Resource":"HighConfidence", "Path":highconf_file or ""},
    {"Resource":"PriorityReview", "Path":priority_file or ""},
]
write_csv(f"{OUTDIR}/sources_used.tsv", sources, ["Resource","Path"])

with open(f"{OUTDIR}/README.txt", "w", encoding="utf-8") as f:
    f.write(f"Figure raw materials rebuilt in: {OUTDIR}\n")
    f.write(f"Master samples: {len(master_rows)}\n")
    f.write(f"Species categories: {len(species_counter)}\n")
    f.write(f"MLST species/ST rows: {len(mlst_rows)}\n")
    f.write(f"AMR class rows: {len(amr_class_rows)}\n")
    f.write(f"AMR gene rows: {len(amr_gene_rows)}\n")
    f.write(f"Virulence gene rows: {len(vfdb_rows)}\n")
    f.write(f"Plasmid replicon rows: {len(plasmid_rows)}\n")
    f.write(f"Serotype categories: {len(sero_rows)}\n")
    f.write(f"HighConfidence samples loaded: {len(highconf)}\n")
    f.write(f"PriorityReview samples loaded: {len(priority)}\n")

print("Wrote rebuilt G1 pack to:", OUTDIR)
print("Master samples:", len(master_rows))
print("Species categories:", len(species_counter))
print("MLST species/ST rows:", len(mlst_rows))
print("AMR class rows:", len(amr_class_rows))
print("AMR gene rows:", len(amr_gene_rows))
print("Virulence gene rows:", len(vfdb_rows))
print("Plasmid replicon rows:", len(plasmid_rows))
print("Serotype categories:", len(sero_rows))
print("HighConfidence loaded:", len(highconf))
print("PriorityReview loaded:", len(priority))
