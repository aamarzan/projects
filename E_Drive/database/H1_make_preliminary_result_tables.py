import csv, os, re
from collections import Counter, defaultdict

WORK="/mnt/e/DrAhmed/Ongoing/WGS/Result"
MASTER=f"{WORK}/_PRIMARY_RESULTS/PrimaryResults_v3_166.csv"
FLAGS=f"{WORK}/_PRELIMINARY_PACK/Table_FlaggedSamples_refined.csv"
OUTDIR=f"{WORK}/_PRELIMINARY_REPORT_READY"

os.makedirs(OUTDIR, exist_ok=True)

# -----------------------------
# load master
# -----------------------------
rows=[]
with open(MASTER,"r",encoding="utf-8",errors="replace") as f:
    rows=list(csv.DictReader(f))

flagged=set()
flag_reason={}
with open(FLAGS,"r",encoding="utf-8",errors="replace") as f:
    for r in csv.DictReader(f):
        flagged.add(r["Sample"])
        flag_reason[r["Sample"]] = r.get("FlagReason","")

# -----------------------------
# Table 1: species counts
# -----------------------------
species_counter=Counter(r.get("TopSpecies1","") for r in rows if r.get("TopSpecies1",""))

with open(f"{OUTDIR}/Table1_SpeciesCounts.csv","w",newline="",encoding="utf-8") as f:
    w=csv.writer(f)
    w.writerow(["TopSpecies","Count"])
    for sp,c in species_counter.most_common():
        w.writerow([sp,c])

# -----------------------------
# Table 2: MLST counts by species
# -----------------------------
species_mlst=defaultdict(Counter)
for r in rows:
    sp=r.get("TopSpecies1","")
    scheme=r.get("MLST_Scheme","")
    st=r.get("MLST_ST","")
    if sp and scheme and scheme != "-" and st and st != "-":
        species_mlst[sp][f"{scheme} | ST{st}"] += 1

with open(f"{OUTDIR}/Table2_MLST_bySpecies.csv","w",newline="",encoding="utf-8") as f:
    w=csv.writer(f)
    w.writerow(["TopSpecies","Scheme_ST","Count"])
    for sp in sorted(species_mlst):
        for k,v in species_mlst[sp].most_common():
            w.writerow([sp,k,v])

# -----------------------------
# Table 3: AMR class burden by species
# -----------------------------
def parse_class_summary(txt):
    out=[]
    for part in (txt or "").split(";"):
        part=part.strip()
        m=re.match(r"(.+)\((\d+)\)$", part)
        if m:
            out.append((m.group(1).strip(), int(m.group(2))))
    return out

species_class_counts=defaultdict(Counter)
for r in rows:
    sp=r.get("TopSpecies1","")
    for cls,n in parse_class_summary(r.get("AMR_Classes","")):
        species_class_counts[sp][cls] += n

with open(f"{OUTDIR}/Table3_AMRClassBurden_bySpecies.csv","w",newline="",encoding="utf-8") as f:
    w=csv.writer(f)
    w.writerow(["TopSpecies","AMR_Class","TotalHitsAcrossSamples"])
    for sp in sorted(species_class_counts):
        for cls,n in species_class_counts[sp].most_common():
            w.writerow([sp,cls,n])

# -----------------------------
# Table 4: priority review samples
# -----------------------------
priority=[]
for r in rows:
    s=r["Sample"]
    if s in flagged:
        reason=flag_reason.get(s,"")
        if any(x in reason for x in ["HighUnclassified", "HumanTaxon", "UnusualTopSpecies", "BadAssembly"]):
            rr=dict(r)
            rr["FlagReason"]=reason
            priority.append(rr)

fields=list(priority[0].keys()) if priority else []
with open(f"{OUTDIR}/Table4_PriorityReviewSamples.csv","w",newline="",encoding="utf-8") as f:
    if fields:
        w=csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in priority:
            w.writerow(r)

# -----------------------------
# Table 5: high-confidence samples
# -----------------------------
high_conf=[]
for r in rows:
    s=r["Sample"]
    reason=flag_reason.get(s,"")
    unclass=float(r.get("UnclassifiedPct") or 0)
    top1pct=float(r.get("TopSpecies1Pct") or 0)
    sp=(r.get("TopSpecies1") or "").strip().lower()

    if (
        s not in flagged and
        unclass < 10 and
        top1pct >= 40 and
        sp not in ["homo sapiens","serratia nevei"]
    ):
        high_conf.append(r)

with open(f"{OUTDIR}/Table5_HighConfidenceSamples.csv","w",newline="",encoding="utf-8") as f:
    if high_conf:
        w=csv.DictWriter(f, fieldnames=high_conf[0].keys())
        w.writeheader()
        for r in high_conf:
            w.writerow(r)

# -----------------------------
# Summary text
# -----------------------------
with open(f"{OUTDIR}/Summary_for_Ahmed.txt","w",encoding="utf-8") as f:
    f.write(f"Total samples analysed: {len(rows)}\n")
    f.write(f"Unique top species: {len(species_counter)}\n")
    f.write(f"Refined flagged samples: {len(flagged)}\n")
    f.write(f"Priority review samples: {len(priority)}\n")
    f.write(f"High-confidence samples: {len(high_conf)}\n")
    f.write("\nTop species counts:\n")
    for sp,c in species_counter.most_common():
        f.write(f"- {sp}: {c}\n")

print("Wrote:", f"{OUTDIR}/Table1_SpeciesCounts.csv")
print("Wrote:", f"{OUTDIR}/Table2_MLST_bySpecies.csv")
print("Wrote:", f"{OUTDIR}/Table3_AMRClassBurden_bySpecies.csv")
print("Wrote:", f"{OUTDIR}/Table4_PriorityReviewSamples.csv")
print("Wrote:", f"{OUTDIR}/Table5_HighConfidenceSamples.csv")
print("Wrote:", f"{OUTDIR}/Summary_for_Ahmed.txt")
print("Total rows:", len(rows))
print("Flagged:", len(flagged))
print("Priority review:", len(priority))
print("High confidence:", len(high_conf))
