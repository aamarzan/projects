import os, csv, re
from collections import Counter, defaultdict

WORK = "/mnt/e/DrAhmed/Ongoing/WGS/Result"
RAW = f"{WORK}/_FIGURE_RAW_MATERIALS_V2"
MASTER = f"{WORK}/_BIOLOGY_LAYER_V2/PrimaryResults_v4_withBiology_166.csv"
REVIEW = f"{WORK}/_G3/review"
os.makedirs(REVIEW, exist_ok=True)

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

# ---------- inputs ----------
species_counts_fp   = f"{RAW}/species_counts_166.csv"
species_conf_fp     = f"{RAW}/species_confidence_counts_166.csv"
species_bio_fp      = f"{RAW}/species_biology_summary_166.csv"
mlst_fp             = f"{RAW}/mlst_species_st_counts_166.csv"
amr_class_fp        = f"{RAW}/amr_class_by_species_166.csv"
amr_gene_fp         = f"{RAW}/amr_gene_by_species_top200_166.csv"
vfdb_fp             = f"{RAW}/virulence_gene_by_species_top200_166.csv"
plasmid_fp          = f"{RAW}/plasmid_replicon_by_species_166.csv"
serotype_fp         = f"{RAW}/serotype_distribution_166.csv"

needed = [
    species_counts_fp, species_conf_fp, species_bio_fp, mlst_fp,
    amr_class_fp, amr_gene_fp, vfdb_fp, plasmid_fp, MASTER
]

missing = [p for p in needed if not os.path.isfile(p)]

species_counts = read_csv(species_counts_fp) if os.path.isfile(species_counts_fp) else []
species_conf   = read_csv(species_conf_fp) if os.path.isfile(species_conf_fp) else []
species_bio    = read_csv(species_bio_fp) if os.path.isfile(species_bio_fp) else []
mlst_rows      = read_csv(mlst_fp) if os.path.isfile(mlst_fp) else []
amr_class_rows = read_csv(amr_class_fp) if os.path.isfile(amr_class_fp) else []
amr_gene_rows  = read_csv(amr_gene_fp) if os.path.isfile(amr_gene_fp) else []
vfdb_rows      = read_csv(vfdb_fp) if os.path.isfile(vfdb_fp) else []
plasmid_rows   = read_csv(plasmid_fp) if os.path.isfile(plasmid_fp) else []
master_rows    = read_csv(MASTER) if os.path.isfile(MASTER) else []

# ---------- master header audit ----------
master_header = list(master_rows[0].keys()) if master_rows else []
species_bio_header = list(species_bio[0].keys()) if species_bio else []

required_for_fig6 = ["VFDB_Hits", "Plasmid_Hits", "TopSpecies1"]
required_for_fig6_summary = ["Samples_with_VFDB_hits", "Samples_with_plasmid_hits"]

fig6_master_ok = all(c in master_header for c in required_for_fig6)
fig6_summary_ok = all(c in species_bio_header for c in required_for_fig6_summary)

# ---------- MLST cross-species audit ----------
species_keywords = {
    "Acinetobacter baumannii": ["abaumannii"],
    "Klebsiella pneumoniae": ["kleb", "kpneumoniae"],
    "Pseudomonas aeruginosa": ["paeruginosa", "pseudomonas"],
    "Escherichia coli": ["escherichia", "ecoli"],
    "Serratia marcescens": ["serratia"],
    "Serratia nevei": ["serratia"],
    "Homo sapiens": ["homo", "human"],
}

mlst_suspects = []
scheme_root_counter = Counter()

for r in mlst_rows:
    sp = r.get("TopSpecies", "").strip()
    st = r.get("Scheme_ST", "").strip()
    ct = as_int(r.get("Count", 0))

    if not sp or not st:
        continue

    scheme_root = st.split("|")[0].strip()
    scheme_root_counter[(sp, scheme_root)] += ct

    low = st.lower()
    foreign_hits = []
    for other_sp, kws in species_keywords.items():
        if other_sp == sp:
            continue
        if any(k in low for k in kws):
            foreign_hits.append(other_sp)

    # flag obvious oddities
    if foreign_hits or low.startswith("plasmid "):
        mlst_suspects.append({
            "TopSpecies": sp,
            "Scheme_ST": st,
            "Count": ct,
            "Reason": "plasmid_scheme" if low.startswith("plasmid ") else "cross_species_label",
            "ForeignSpeciesDetected": "; ".join(foreign_hits)
        })

# ---------- Figure 6 prevalence directly from master ----------
vf_prev = Counter()
pl_prev = Counter()
species_counts_master = Counter()

for r in master_rows:
    sp = r.get("TopSpecies1", "").strip()
    if not sp:
        continue
    species_counts_master[sp] += 1
    if as_int(r.get("VFDB_Hits", 0)) > 0:
        vf_prev[sp] += 1
    if as_int(r.get("Plasmid_Hits", 0)) > 0:
        pl_prev[sp] += 1

prev_rows = []
for sp in sorted(species_counts_master):
    prev_rows.append({
        "TopSpecies": sp,
        "Samples_n": species_counts_master[sp],
        "Samples_with_VFDB_hits_from_master": vf_prev[sp],
        "Samples_with_Plasmid_hits_from_master": pl_prev[sp]
    })

write_csv(
    f"{REVIEW}/G3_mlst_suspect_labels.csv",
    mlst_suspects,
    ["TopSpecies", "Scheme_ST", "Count", "Reason", "ForeignSpeciesDetected"]
)

scheme_rows = []
for (sp, scheme_root), ct in sorted(scheme_root_counter.items(), key=lambda x: (x[0][0], -x[1], x[0][1])):
    scheme_rows.append({
        "TopSpecies": sp,
        "SchemeRoot": scheme_root,
        "Count": ct
    })

write_csv(
    f"{REVIEW}/G3_mlst_scheme_roots.csv",
    scheme_rows,
    ["TopSpecies", "SchemeRoot", "Count"]
)

write_csv(
    f"{REVIEW}/G3_vf_plasmid_prevalence_from_master.csv",
    prev_rows,
    ["TopSpecies", "Samples_n", "Samples_with_VFDB_hits_from_master", "Samples_with_Plasmid_hits_from_master"]
)

with open(f"{REVIEW}/G3_input_audit.txt", "w", encoding="utf-8") as f:
    f.write("G3 FIGURE INPUT AUDIT\n")
    f.write("=====================\n\n")

    f.write("Missing required files:\n")
    if missing:
        for p in missing:
            f.write(f"- {p}\n")
    else:
        f.write("- none\n")
    f.write("\n")

    f.write("Headers\n")
    f.write("-------\n")
    f.write(f"species_biology_summary columns: {', '.join(species_bio_header) if species_bio_header else 'NONE'}\n")
    f.write(f"master columns: {', '.join(master_header) if master_header else 'NONE'}\n\n")

    f.write("Figure-level readiness\n")
    f.write("----------------------\n")
    f.write(f"Figure 1 source readiness: {'OK' if species_counts and species_conf and species_bio else 'CHECK'}\n")
    f.write(f"Figure 2 source readiness: {'OK' if species_bio else 'CHECK'}\n")
    f.write(f"Figure 3 source readiness: {'CHECK - inspect MLST suspects' if mlst_suspects else 'OK'}\n")
    f.write(f"Figure 4 source readiness: {'OK' if amr_class_rows else 'CHECK'}\n")
    f.write(f"Figure 5 source readiness: {'OK' if amr_gene_rows else 'CHECK'}\n")
    f.write(f"Figure 6 summary-based readiness: {'OK' if fig6_summary_ok else 'NOT OK'}\n")
    f.write(f"Figure 6 master-based readiness: {'OK' if fig6_master_ok else 'CHECK'}\n")
    f.write(f"Supplementary serotype readiness: {'OK' if os.path.isfile(serotype_fp) else 'OPTIONAL / MISSING'}\n\n")

    f.write("Interpretation\n")
    f.write("--------------\n")
    if not fig6_summary_ok and fig6_master_ok:
        f.write("- Figure 6 should be rebuilt from PrimaryResults_v4_withBiology_166.csv, not from species_biology_summary_166.csv.\n")
    if mlst_suspects:
        f.write(f"- MLST suspect rows detected: {len(mlst_suspects)}. Figure 3 needs scheme cleaning or explicit filtering.\n")
    else:
        f.write("- No MLST cross-species suspect labels detected.\n")

    f.write("\nQuick counts\n")
    f.write("------------\n")
    f.write(f"Species rows in species_counts_166.csv: {len(species_counts)}\n")
    f.write(f"MLST rows: {len(mlst_rows)}\n")
    f.write(f"AMR class rows: {len(amr_class_rows)}\n")
    f.write(f"AMR gene rows: {len(amr_gene_rows)}\n")
    f.write(f"Virulence rows: {len(vfdb_rows)}\n")
    f.write(f"Plasmid rows: {len(plasmid_rows)}\n")
    f.write(f"Master rows: {len(master_rows)}\n")

print("Wrote:", f"{REVIEW}/G3_input_audit.txt")
print("Wrote:", f"{REVIEW}/G3_mlst_suspect_labels.csv")
print("Wrote:", f"{REVIEW}/G3_mlst_scheme_roots.csv")
print("Wrote:", f"{REVIEW}/G3_vf_plasmid_prevalence_from_master.csv")
