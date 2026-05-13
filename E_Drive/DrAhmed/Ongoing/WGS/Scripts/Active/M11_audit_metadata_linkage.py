import os
import csv
from collections import Counter, defaultdict

WORK = "/mnt/e/DrAhmed/Ongoing/WGS"
RESULT = f"{WORK}/Result"
META = f"{WORK}/Metadata"

MASTER = f"{RESULT}/_BIOLOGY_LAYER_V2/PrimaryResults_v4_withBiology_166.csv"
SM_CLEAN = f"{META}/SM_Metadata_Shahad_clean_for_join.csv"

OUTDIR = f"{RESULT}/_METADATA_AUDIT"
os.makedirs(OUTDIR, exist_ok=True)

def read_csv(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))

def norm(x):
    return str(x).strip()

master = read_csv(MASTER)
sm = read_csv(SM_CLEAN) if os.path.isfile(SM_CLEAN) else []

master_by_sample = {norm(r["Sample"]): r for r in master}
sm_by_sample = {norm(r["Sample"]): r for r in sm if norm(r.get("Sample", ""))}

matched = []
unmatched_master = []
for s, r in master_by_sample.items():
    if s in sm_by_sample:
        x = dict(r)
        x.update({
            "Meta_Gender": sm_by_sample[s].get("Gender", ""),
            "Meta_Age": sm_by_sample[s].get("Age", ""),
            "Meta_DateOfIsolation": sm_by_sample[s].get("DateOfIsolation", ""),
            "Meta_Specimen": sm_by_sample[s].get("Specimen", ""),
            "Meta_Source": sm_by_sample[s].get("MetadataSource", ""),
        })
        matched.append(x)
    else:
        unmatched_master.append(r)

# species-level match counts
species_counts = Counter()
species_match = Counter()
for r in master:
    sp = norm(r.get("TopSpecies1", ""))
    species_counts[sp] += 1
for r in matched:
    sp = norm(r.get("TopSpecies1", ""))
    species_match[sp] += 1

with open(f"{OUTDIR}/metadata_species_linkage_summary.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["TopSpecies", "TotalSamples", "MatchedSamples", "UnmatchedSamples"])
    w.writeheader()
    for sp, n in species_counts.most_common():
        m = species_match.get(sp, 0)
        w.writerow({
            "TopSpecies": sp,
            "TotalSamples": n,
            "MatchedSamples": m,
            "UnmatchedSamples": n - m,
        })

with open(f"{OUTDIR}/metadata_matched_samples.csv", "w", newline="", encoding="utf-8") as f:
    if matched:
        w = csv.DictWriter(f, fieldnames=list(matched[0].keys()))
        w.writeheader()
        for r in matched:
            w.writerow(r)

with open(f"{OUTDIR}/metadata_unmatched_master_samples.csv", "w", newline="", encoding="utf-8") as f:
    if unmatched_master:
        w = csv.DictWriter(f, fieldnames=list(unmatched_master[0].keys()))
        w.writeheader()
        for r in unmatched_master:
            w.writerow(r)

with open(f"{OUTDIR}/README.txt", "w", encoding="utf-8") as f:
    f.write(f"Total master samples: {len(master)}\n")
    f.write(f"Directly matched SM metadata samples: {len(matched)}\n")
    f.write(f"Unmatched master samples: {len(unmatched_master)}\n")
    f.write("\nInterpretation:\n")
    f.write("- This audit only uses direct Sample ID matching against the cleaned SM metadata.\n")
    f.write("- Metadata_Manahil workbook is not auto-joined here because it does not currently provide a trustworthy direct AB/KP/... sample key for the master WGS sample IDs.\n")
    f.write("- Do not force joins by species alone.\n")

print("Wrote metadata audit to:", OUTDIR)
print("Directly matched samples:", len(matched))