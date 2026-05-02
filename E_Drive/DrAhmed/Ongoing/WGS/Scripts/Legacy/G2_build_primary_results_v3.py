import csv, os

WORK="/mnt/e/DrAhmed/Ongoing/WGS/Result"
MASTER=f"{WORK}/_PRIMARY_RESULTS/PrimaryResults_v1_166.csv"
MLST=f"{WORK}/_PRIMARY_RESULTS/MLST_clean_166.csv"
AMR=f"{WORK}/_PRIMARY_RESULTS/AMRFinder_clean_166.csv"
OUT=f"{WORK}/_PRIMARY_RESULTS/PrimaryResults_v3_166.csv"

master={}
with open(MASTER,"r",encoding="utf-8",errors="replace") as f:
    for row in csv.DictReader(f):
        master[row["Sample"]] = row

mlst={}
with open(MLST,"r",encoding="utf-8",errors="replace") as f:
    for row in csv.DictReader(f):
        mlst[row["Sample"]] = row

amr={}
with open(AMR,"r",encoding="utf-8",errors="replace") as f:
    for row in csv.DictReader(f):
        amr[row["Sample"]] = row

samples=sorted(master.keys())

fields=[
    "Sample",
    "UnclassifiedPct","TopSpecies1","TopSpecies1Pct","TopSpecies2","TopSpecies2Pct",
    "Contigs","LargestContig","TotalLength","GC_percent","N50","N75","L50","L75",
    "MLST_Scheme","MLST_ST","MLST_Alleles",
    "AMRFinder_Hits","AMR_Classes","AMR_TopGenes"
]

with open(OUT,"w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for s in samples:
        row={"Sample": s}
        row.update({k: master[s].get(k,"") for k in [
            "UnclassifiedPct","TopSpecies1","TopSpecies1Pct","TopSpecies2","TopSpecies2Pct",
            "Contigs","LargestContig","TotalLength","GC_percent","N50","N75","L50","L75"
        ]})
        if s in mlst:
            row["MLST_Scheme"] = mlst[s].get("MLST_Scheme","")
            row["MLST_ST"] = mlst[s].get("MLST_ST","")
            row["MLST_Alleles"] = mlst[s].get("MLST_Alleles","")
        if s in amr:
            row["AMRFinder_Hits"] = amr[s].get("AMRFinder_Hits","")
            row["AMR_Classes"] = amr[s].get("AMR_Classes","")
            row["AMR_TopGenes"] = amr[s].get("AMR_TopGenes","")
        w.writerow(row)

print("Merged samples:", len(samples))
print("Wrote:", OUT)
