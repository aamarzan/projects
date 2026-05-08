import os, csv
from collections import defaultdict
import matplotlib.pyplot as plt

WORK = "/mnt/e/DrAhmed/Ongoing/WGS/Result"
INDIR = f"{WORK}/_MANUSCRIPT_FINAL"
OUTDIR = f"{WORK}/_MANUSCRIPT_FIGURES"
os.makedirs(OUTDIR, exist_ok=True)

def read_csv(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))

# Figure 1: cohort overview
summary_txt = os.path.join(INDIR, "Summary.txt")
vals = {}
with open(summary_txt, "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        if ":" in line:
            k, v = line.strip().split(":", 1)
            vals[k.strip()] = v.strip()

total = int(vals.get("Total samples", "166"))
highc = int(vals.get("High-confidence samples", "106"))
prio  = int(vals.get("Priority-review samples", "29"))
other = max(total - highc - prio, 0)

plt.figure(figsize=(7,5))
plt.bar(["High-confidence","Priority-review","Other/remaining"], [highc, prio, other])
plt.ylabel("Number of samples")
plt.title("Cohort overview")
plt.tight_layout()
plt.savefig(f"{OUTDIR}/Figure1_CohortOverview.png", dpi=300)
plt.close()

# Figure 2: species distribution
rows = read_csv(f"{INDIR}/Table1_SpeciesCounts.csv")
species = [r["TopSpecies"] for r in rows]
counts  = [int(r["Count"]) for r in rows]

plt.figure(figsize=(10,6))
plt.bar(species, counts)
plt.ylabel("Number of samples")
plt.title("Species distribution across all 166 samples")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(f"{OUTDIR}/Figure2_SpeciesDistribution.png", dpi=300)
plt.close()

# Figure 3: MLST by species
rows = read_csv(f"{INDIR}/Table2_MLST_bySpecies.csv")
labels = [f'{r["TopSpecies"]}\n{r["Scheme_ST"]}' for r in rows]
count = [int(r["Count"]) for r in rows]

plt.figure(figsize=(12,7))
plt.bar(labels, count)
plt.ylabel("Number of samples")
plt.title("Reliable MLST distributions by species")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/Figure3_MLST_bySpecies.png", dpi=300)
plt.close()

# Figure 4: AMR class burden
rows = read_csv(f"{INDIR}/Table3_AMRClassBurden_bySpecies.csv")
species_classes = defaultdict(dict)
all_classes = []
species_set = []

for r in rows:
    sp = r["TopSpecies"]
    cl = r["AMR_Class"]
    n = int(r["TotalHitsAcrossSamples"])
    species_classes[sp][cl] = n
    if cl not in all_classes:
        all_classes.append(cl)
    if sp not in species_set:
        species_set.append(sp)

bottom = [0] * len(species_set)
plt.figure(figsize=(11,7))
for cl in all_classes:
    vals = [species_classes[s].get(cl, 0) for s in species_set]
    plt.bar(species_set, vals, bottom=bottom, label=cl)
    bottom = [bottom[i] + vals[i] for i in range(len(vals))]
plt.ylabel("Total AMR class hits")
plt.title("AMR class burden by species")
plt.xticks(rotation=45, ha="right")
plt.legend(fontsize=7)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/Figure4_AMRClassBurden.png", dpi=300)
plt.close()

# Figure 5A: virulence burden
rows = read_csv(f"{INDIR}/Table6_VirulenceBurden_bySpecies.csv")
species = [r["TopSpecies"] for r in rows]
med_vf  = [float(r["Median_VFDB_hits"]) for r in rows]

plt.figure(figsize=(9,6))
plt.bar(species, med_vf)
plt.ylabel("Median VFDB hits")
plt.title("Median virulence burden by species")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(f"{OUTDIR}/Figure5A_VirulenceBurden.png", dpi=300)
plt.close()

# Figure 5B: plasmid burden
rows = read_csv(f"{INDIR}/Table7_PlasmidBurden_bySpecies.csv")
species = [r["TopSpecies"] for r in rows]
med_pl  = [float(r["Median_plasmid_hits"]) for r in rows]

plt.figure(figsize=(9,6))
plt.bar(species, med_pl)
plt.ylabel("Median plasmid hits")
plt.title("Median plasmid burden by species")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(f"{OUTDIR}/Figure5B_PlasmidBurden.png", dpi=300)
plt.close()

print("Figures written to:", OUTDIR)
for fn in sorted(os.listdir(OUTDIR)):
    print(fn)
