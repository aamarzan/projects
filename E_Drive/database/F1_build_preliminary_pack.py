import csv, os
from collections import Counter

WORK="/mnt/e/DrAhmed/Ongoing/WGS/Result"
MASTER=f"{WORK}/_PRIMARY_RESULTS/PrimaryResults_v2_166.csv"
BADASM=f"{WORK}/_ASSEMBLY_QC2/assemblies_manifest_STILL_BAD.tsv"
OUTDIR=f"{WORK}/_PRELIMINARY_PACK"

os.makedirs(OUTDIR, exist_ok=True)

species_counter=Counter()
flag_rows=[]
all_rows=[]

with open(MASTER,"r",encoding="utf-8",errors="replace") as f:
    rows=list(csv.DictReader(f))

for r in rows:
    all_rows.append(r)
    sp=r.get("TopSpecies1","")
    if sp:
        species_counter[sp]+=1

    unclass=float(r.get("UnclassifiedPct") or 0)
    top1pct=float(r.get("TopSpecies1Pct") or 0)

    reasons=[]
    if unclass >= 5:
        reasons.append("HighUnclassified")
    if top1pct < 50:
        reasons.append("LowTopSpeciesPct")
    if sp.lower() == "homo sapiens":
        reasons.append("HumanTaxon")
    if sp.lower() == "serratia nevei":
        reasons.append("UnusualTopSpecies")

    if reasons:
        rr=dict(r)
        rr["FlagReason"]=";".join(reasons)
        flag_rows.append(rr)

bad_asm=set()
with open(BADASM,"r",encoding="utf-8",errors="replace") as f:
    next(f, None)
    for line in f:
        if line.strip():
            bad_asm.add(line.split("\t")[0].strip())

species_out=f"{OUTDIR}/Table_SpeciesCounts.csv"
flags_out=f"{OUTDIR}/Table_FlaggedSamples.csv"
summary_txt=f"{OUTDIR}/Summary_Overview.txt"
badasm_out=f"{OUTDIR}/Table_BadAssemblies.csv"

with open(species_out,"w",newline="",encoding="utf-8") as f:
    w=csv.writer(f)
    w.writerow(["TopSpecies","CountSamples"])
    for sp,c in species_counter.most_common():
        w.writerow([sp,c])

with open(flags_out,"w",newline="",encoding="utf-8") as f:
    fields=list(rows[0].keys())+["FlagReason"]
    w=csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in sorted(flag_rows, key=lambda x: x["Sample"]):
        w.writerow(r)

with open(badasm_out,"w",newline="",encoding="utf-8") as f:
    w=csv.writer(f)
    w.writerow(["Sample"])
    for s in sorted(bad_asm):
        w.writerow([s])

with open(summary_txt,"w",encoding="utf-8") as f:
    f.write(f"Total samples in master: {len(rows)}\n")
    f.write(f"Unique top species: {len(species_counter)}\n")
    f.write(f"Flagged samples: {len(flag_rows)}\n")
    f.write(f"Assemblies still bad: {len(bad_asm)}\n")
    f.write("\nTop species counts:\n")
    for sp,c in species_counter.most_common(15):
        f.write(f"- {sp}: {c}\n")

print("Wrote:", species_out)
print("Wrote:", flags_out)
print("Wrote:", badasm_out)
print("Wrote:", summary_txt)
print("Total samples:", len(rows))
print("Flagged samples:", len(flag_rows))
print("Bad assemblies:", len(bad_asm))
