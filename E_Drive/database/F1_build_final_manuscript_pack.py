import os, csv, re, statistics
from collections import Counter, defaultdict

WORK = "/mnt/e/DrAhmed/Ongoing/WGS/Result"
MASTER = f"{WORK}/_BIOLOGY_LAYER_V2/PrimaryResults_v4_withBiology_166.csv"
OUTDIR = f"{WORK}/_MANUSCRIPT_FINAL"
os.makedirs(OUTDIR, exist_ok=True)

# Prefer the already prepared refined sample tables if present
priority_candidates = [
    f"{WORK}/_PRELIMINARY_REPORT_READY/Table4_PriorityReviewSamples.csv",
    f"{WORK}/_PRELIMINARY_PACK/Table_FlaggedSamples_refined.csv",
]
highconf_candidates = [
    f"{WORK}/_PRELIMINARY_REPORT_READY/Table5_HighConfidenceSamples.csv",
]

def first_existing(paths):
    for p in paths:
        if os.path.isfile(p):
            return p
    return None

PRIORITY_FILE = first_existing(priority_candidates)
HIGHCONF_FILE = first_existing(highconf_candidates)

if not os.path.isfile(MASTER):
    raise SystemExit(f"Missing master file: {MASTER}")

def read_csv(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))

def load_sample_set(path):
    if not path or not os.path.isfile(path):
        return set()
    rows = read_csv(path)
    out = set()
    for r in rows:
        if "Sample" in r and r["Sample"]:
            out.add(r["Sample"])
    return out

rows = read_csv(MASTER)
priority = load_sample_set(PRIORITY_FILE)
highconf = load_sample_set(HIGHCONF_FILE)

# fallback if HighConfidence file missing: everything not in priority becomes high-confidence
if not highconf:
    highconf = {r["Sample"] for r in rows if r["Sample"] not in priority}

# annotate final status
final_rows = []
for r in rows:
    rr = dict(r)
    s = r["Sample"]
    if s in priority:
        rr["ReviewStatus"] = "PriorityReview"
    elif s in highconf:
        rr["ReviewStatus"] = "HighConfidence"
    else:
        rr["ReviewStatus"] = "OtherFlagged"
    final_rows.append(rr)

# write frozen master
frozen_master = f"{OUTDIR}/MasterResults_final_166.csv"
with open(frozen_master, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(final_rows[0].keys()))
    w.writeheader()
    w.writerows(final_rows)

# helper
def parse_amr_classes(txt):
    vals = []
    for part in (txt or "").split(";"):
        part = part.strip()
        m = re.match(r"(.+)\((\d+)\)$", part)
        if m:
            vals.append((m.group(1).strip(), int(m.group(2))))
    return vals

def plausible_mlst(species, scheme):
    species = (species or "").strip().lower()
    scheme = (scheme or "").strip().lower()
    if not species or not scheme or scheme == "-":
        return False
    if species == "acinetobacter baumannii":
        return scheme.startswith("abaumannii")
    if species == "pseudomonas aeruginosa":
        return scheme.startswith("paeruginosa")
    if species in ("serratia marcescens", "serratia nevei"):
        return scheme.startswith("serratia")
    if species == "escherichia coli":
        return scheme.startswith("escherichia")
    if species == "klebsiella pneumoniae":
        return scheme.startswith("kleb")
    return False

# split
main_rows = [r for r in final_rows if r["ReviewStatus"] == "HighConfidence"]
priority_rows = [r for r in final_rows if r["ReviewStatus"] == "PriorityReview"]

# Table 1 species counts
species_counts = Counter(r["TopSpecies1"] for r in final_rows if r.get("TopSpecies1"))
with open(f"{OUTDIR}/Table1_SpeciesCounts.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["TopSpecies","Count"])
    for sp, c in species_counts.most_common():
        w.writerow([sp, c])

# Table 2 reliable MLST by species
mlst_counts = defaultdict(Counter)
for r in main_rows:
    sp = r.get("TopSpecies1","")
    scheme = r.get("MLST_Scheme","")
    st = r.get("MLST_ST","")
    if plausible_mlst(sp, scheme) and st and st != "-":
        mlst_counts[sp][f"{scheme} | ST{st}"] += 1

with open(f"{OUTDIR}/Table2_MLST_bySpecies.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["TopSpecies","Scheme_ST","Count"])
    for sp in sorted(mlst_counts):
        for k, v in mlst_counts[sp].most_common():
            w.writerow([sp, k, v])

# Table 3 AMR class burden
amr_counts = defaultdict(Counter)
for r in main_rows:
    sp = r.get("TopSpecies1","")
    for cls, n in parse_amr_classes(r.get("AMR_Classes","")):
        amr_counts[sp][cls] += n

with open(f"{OUTDIR}/Table3_AMRClassBurden_bySpecies.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["TopSpecies","AMR_Class","TotalHitsAcrossSamples"])
    for sp in sorted(amr_counts):
        for cls, n in amr_counts[sp].most_common():
            w.writerow([sp, cls, n])

# Table 4 priority review
with open(f"{OUTDIR}/Table4_PriorityReviewSamples.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(priority_rows[0].keys()) if priority_rows else list(final_rows[0].keys()))
    w.writeheader()
    if priority_rows:
        w.writerows(priority_rows)

# Table 5 high-confidence samples
with open(f"{OUTDIR}/Table5_HighConfidenceSamples.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(main_rows[0].keys()) if main_rows else list(final_rows[0].keys()))
    w.writeheader()
    if main_rows:
        w.writerows(main_rows)

# Table 6 virulence summary by species
vir_species = defaultdict(list)
for r in main_rows:
    try:
        vf_hits = int(r.get("VFDB_Hits","0") or 0)
    except:
        vf_hits = 0
    vir_species[r.get("TopSpecies1","")].append(vf_hits)

with open(f"{OUTDIR}/Table6_VirulenceBurden_bySpecies.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["TopSpecies","Samples_n","Samples_with_VFDB_hits","Median_VFDB_hits","Max_VFDB_hits"])
    for sp in sorted(vir_species):
        vals = vir_species[sp]
        w.writerow([
            sp,
            len(vals),
            sum(1 for x in vals if x > 0),
            statistics.median(vals) if vals else 0,
            max(vals) if vals else 0
        ])

# Table 7 plasmid summary by species
pl_species = defaultdict(list)
for r in main_rows:
    try:
        pl_hits = int(r.get("Plasmid_Hits","0") or 0)
    except:
        pl_hits = 0
    pl_species[r.get("TopSpecies1","")].append(pl_hits)

with open(f"{OUTDIR}/Table7_PlasmidBurden_bySpecies.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["TopSpecies","Samples_n","Samples_with_plasmid_hits","Median_plasmid_hits","Max_plasmid_hits"])
    for sp in sorted(pl_species):
        vals = pl_species[sp]
        w.writerow([
            sp,
            len(vals),
            sum(1 for x in vals if x > 0),
            statistics.median(vals) if vals else 0,
            max(vals) if vals else 0
        ])

# summary text
with open(f"{OUTDIR}/Summary.txt", "w", encoding="utf-8") as f:
    f.write(f"Total samples: {len(final_rows)}\n")
    f.write(f"High-confidence samples: {len(main_rows)}\n")
    f.write(f"Priority-review samples: {len(priority_rows)}\n")
    f.write(f"Species categories: {len(species_counts)}\n")
    f.write(f"Samples with VFDB hits > 0: {sum(1 for r in final_rows if int(r.get('VFDB_Hits','0') or 0) > 0)}\n")
    f.write(f"Samples with plasmid hits > 0: {sum(1 for r in final_rows if int(r.get('Plasmid_Hits','0') or 0) > 0)}\n")
    f.write(f"Samples with informative serotype call: {sum(1 for r in final_rows if r.get('Serotype_Informative','No') == 'Yes')}\n")
    f.write("\nTop species counts:\n")
    for sp, c in species_counts.most_common():
        f.write(f"- {sp}: {c}\n")

print("Wrote frozen master and final manuscript tables to:", OUTDIR)
print("Priority source:", PRIORITY_FILE)
print("High-confidence source:", HIGHCONF_FILE if HIGHCONF_FILE else "fallback=not in priority")
print("Total:", len(final_rows))
print("HighConfidence:", len(main_rows))
print("PriorityReview:", len(priority_rows))
