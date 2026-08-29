import csv, os, re
from collections import Counter, defaultdict

WORK = "/mnt/e/DrAhmed/Ongoing/WGS/Result"
MASTER = f"{WORK}/_PRIMARY_RESULTS/PrimaryResults_v3_166.csv"
FLAGS  = f"{WORK}/_PRELIMINARY_PACK/Table_FlaggedSamples_refined.csv"
OUTDIR = f"{WORK}/_MANUSCRIPT_CURATED"

os.makedirs(OUTDIR, exist_ok=True)

# -----------------------------
# load data
# -----------------------------
with open(MASTER, "r", encoding="utf-8", errors="replace") as f:
    rows = list(csv.DictReader(f))

flagged = {}
with open(FLAGS, "r", encoding="utf-8", errors="replace") as f:
    for r in csv.DictReader(f):
        flagged[r["Sample"]] = r.get("FlagReason", "")

# -----------------------------
# classify samples
# -----------------------------
main_rows = []
flag_rows = []

for r in rows:
    rr = dict(r)
    rr["FlagReason"] = flagged.get(r["Sample"], "")
    if r["Sample"] in flagged:
        flag_rows.append(rr)
    else:
        main_rows.append(rr)

# -----------------------------
# define refined priority review
# -----------------------------
priority_rows = []
for r in flag_rows:
    reason = r["FlagReason"]
    if any(x in reason for x in [
        "HighUnclassified",
        "HumanTaxon",
        "UnusualTopSpecies",
        "VeryLowTopSpeciesPct",
        "BadAssembly"
    ]):
        priority_rows.append(r)

# -----------------------------
# biologically plausible MLST mapping for MAIN text
# -----------------------------
def plausible_mlst(species, scheme):
    species = (species or "").strip().lower()
    scheme = (scheme or "").strip().lower()

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

# -----------------------------
# Table 1: main species distribution
# -----------------------------
species_counter = Counter(r["TopSpecies1"] for r in rows if r.get("TopSpecies1"))
with open(f"{OUTDIR}/Table1_SpeciesDistribution.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["TopSpecies","Count"])
    for sp, c in species_counter.most_common():
        w.writerow([sp, c])

# -----------------------------
# Table 2: manuscript-core MLST only
# -----------------------------
core_mlst = defaultdict(Counter)
for r in main_rows:
    sp = r.get("TopSpecies1","")
    scheme = r.get("MLST_Scheme","")
    st = r.get("MLST_ST","")
    if sp and scheme and st and scheme != "-" and st != "-" and plausible_mlst(sp, scheme):
        core_mlst[sp][f"{scheme} | ST{st}"] += 1

with open(f"{OUTDIR}/Table2_CoreMLST.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["TopSpecies","Scheme_ST","Count"])
    for sp in sorted(core_mlst):
        for k, v in core_mlst[sp].most_common():
            w.writerow([sp, k, v])

# -----------------------------
# Table 3: AMR burden by species
# -----------------------------
def parse_class_summary(txt):
    out = []
    for part in (txt or "").split(";"):
        part = part.strip()
        m = re.match(r"(.+)\((\d+)\)$", part)
        if m:
            out.append((m.group(1).strip(), int(m.group(2))))
    return out

species_class_counts = defaultdict(Counter)
for r in rows:
    sp = r.get("TopSpecies1","")
    for cls, n in parse_class_summary(r.get("AMR_Classes","")):
        species_class_counts[sp][cls] += n

with open(f"{OUTDIR}/Table3_AMRClassBurden.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["TopSpecies","AMR_Class","TotalHitsAcrossSamples"])
    for sp in sorted(species_class_counts):
        for cls, n in species_class_counts[sp].most_common():
            w.writerow([sp, cls, n])

# -----------------------------
# Table 4: main integrated result table
# -----------------------------
with open(f"{OUTDIR}/Table4_MainIntegratedResults.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(main_rows[0].keys()))
    w.writeheader()
    for r in main_rows:
        w.writerow(r)

# -----------------------------
# Supplementary review panels
# -----------------------------
with open(f"{OUTDIR}/Supplementary_PriorityReview.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(priority_rows[0].keys()))
    w.writeheader()
    for r in priority_rows:
        w.writerow(r)

kp_unresolved = [r for r in rows if r.get("TopSpecies1","") == "Klebsiella pneumoniae" and not plausible_mlst(r.get("TopSpecies1",""), r.get("MLST_Scheme",""))]
with open(f"{OUTDIR}/Supplementary_Klebsiella_UnresolvedMLST.csv", "w", newline="", encoding="utf-8") as f:
    if kp_unresolved:
        w = csv.DictWriter(f, fieldnames=list(kp_unresolved[0].keys()))
        w.writeheader()
        for r in kp_unresolved:
            w.writerow(r)

serratia_review = [r for r in priority_rows if "serratia" in (r.get("TopSpecies1","").lower()) or r["Sample"].startswith("SM")]
with open(f"{OUTDIR}/Supplementary_Serratia_Review.csv", "w", newline="", encoding="utf-8") as f:
    if serratia_review:
        w = csv.DictWriter(f, fieldnames=list(serratia_review[0].keys()))
        w.writeheader()
        for r in serratia_review:
            w.writerow(r)

# -----------------------------
# Summary
# -----------------------------
with open(f"{OUTDIR}/CuratedSummary.txt", "w", encoding="utf-8") as f:
    f.write(f"Total samples: {len(rows)}\n")
    f.write(f"Main-analysis samples: {len(main_rows)}\n")
    f.write(f"Flagged samples: {len(flag_rows)}\n")
    f.write(f"Priority-review samples: {len(priority_rows)}\n")
    f.write(f"Klebsiella unresolved-MLST samples: {len(kp_unresolved)}\n")
    f.write(f"Serratia review samples: {len(serratia_review)}\n")
    f.write("\nSpecies distribution:\n")
    for sp, c in species_counter.most_common():
        f.write(f"- {sp}: {c}\n")

print("Wrote curated manuscript tables to:", OUTDIR)
print("Total:", len(rows))
print("Main:", len(main_rows))
print("Flagged:", len(flag_rows))
print("Priority-review:", len(priority_rows))
print("KP unresolved MLST:", len(kp_unresolved))
print("Serratia review:", len(serratia_review))
