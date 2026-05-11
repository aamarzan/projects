import csv, os, re
from collections import Counter, defaultdict

WORK = "/mnt/e/DrAhmed/Ongoing/WGS/Result"
MASTER = f"{WORK}/_PRIMARY_RESULTS/PrimaryResults_v3_166.csv"
FLAGS  = f"{WORK}/_PRELIMINARY_PACK/Table_FlaggedSamples_refined.csv"
OUTDIR = f"{WORK}/_MANUSCRIPT_READY"

os.makedirs(OUTDIR, exist_ok=True)

rows = []
with open(MASTER, "r", encoding="utf-8", errors="replace") as f:
    rows = list(csv.DictReader(f))

flagged = {}
with open(FLAGS, "r", encoding="utf-8", errors="replace") as f:
    for r in csv.DictReader(f):
        flagged[r["Sample"]] = r.get("FlagReason", "")

# -------- classify samples --------
high_conf = []
review = []
for r in rows:
    s = r["Sample"]
    if s in flagged:
        rr = dict(r)
        rr["FlagReason"] = flagged[s]
        review.append(rr)
    else:
        high_conf.append(r)

# -------- table 1: species counts --------
species_counter = Counter(r.get("TopSpecies1","") for r in rows if r.get("TopSpecies1",""))
with open(f"{OUTDIR}/Table1_SpeciesDistribution.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["TopSpecies","Count"])
    for sp, c in species_counter.most_common():
        w.writerow([sp, c])

# -------- table 2: MLST counts among high-confidence --------
species_mlst = defaultdict(Counter)
for r in high_conf:
    sp = r.get("TopSpecies1","")
    scheme = r.get("MLST_Scheme","")
    st = r.get("MLST_ST","")
    if sp and scheme and scheme != "-" and st and st != "-":
        species_mlst[sp][f"{scheme} | ST{st}"] += 1

with open(f"{OUTDIR}/Table2_MLST_HighConfidence.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["TopSpecies","Scheme_ST","Count"])
    for sp in sorted(species_mlst):
        for k, v in species_mlst[sp].most_common():
            w.writerow([sp, k, v])

# -------- table 3: AMR class burden by species --------
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

# -------- table 4: priority review --------
priority = []
for r in review:
    reason = r.get("FlagReason","")
    if any(x in reason for x in ["HighUnclassified", "HumanTaxon", "UnusualTopSpecies", "BadAssembly", "VeryLowTopSpeciesPct"]):
        priority.append(r)

if priority:
    with open(f"{OUTDIR}/Table4_PriorityReview.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(priority[0].keys()))
        w.writeheader()
        for r in priority:
            w.writerow(r)

# -------- table 5: high-confidence final table --------
if high_conf:
    with open(f"{OUTDIR}/Table5_HighConfidenceIntegrated.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(high_conf[0].keys()))
        w.writeheader()
        for r in high_conf:
            w.writerow(r)

# -------- focused review panels --------
kp_review = [r for r in review if (r["Sample"].startswith("KP-") or (r.get("TopSpecies1","").lower() == "klebsiella pneumoniae"))]
sm_review = [r for r in review if (r["Sample"].startswith("SM") or "serratia" in r.get("TopSpecies1","").lower())]

if kp_review:
    with open(f"{OUTDIR}/ReviewPanel_Klebsiella.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(kp_review[0].keys()))
        w.writeheader()
        for r in kp_review:
            w.writerow(r)

if sm_review:
    with open(f"{OUTDIR}/ReviewPanel_Serratia.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(sm_review[0].keys()))
        w.writeheader()
        for r in sm_review:
            w.writerow(r)

# -------- summary --------
with open(f"{OUTDIR}/ManuscriptSummary.txt", "w", encoding="utf-8") as f:
    f.write(f"Total samples: {len(rows)}\n")
    f.write(f"High-confidence samples: {len(high_conf)}\n")
    f.write(f"Flagged samples: {len(review)}\n")
    f.write(f"Priority review samples: {len(priority)}\n")
    f.write(f"Klebsiella review panel: {len(kp_review)}\n")
    f.write(f"Serratia review panel: {len(sm_review)}\n")
    f.write("\nSpecies distribution:\n")
    for sp, c in species_counter.most_common():
        f.write(f"- {sp}: {c}\n")

print("Wrote manuscript-ready tables to:", OUTDIR)
print("Total samples:", len(rows))
print("High-confidence:", len(high_conf))
print("Flagged:", len(review))
print("Priority review:", len(priority))
print("Klebsiella review:", len(kp_review))
print("Serratia review:", len(sm_review))
