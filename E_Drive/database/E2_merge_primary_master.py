import csv, os

MASTER="/mnt/e/DrAhmed/Ongoing/WGS/Result/_PRIMARY_RESULTS/PrimaryResults_v1_166.csv"
MLST="/mnt/e/DrAhmed/Ongoing/WGS/Result/_PRIMARY_RESULTS/MLST_summary_166.csv"
AMR="/mnt/e/DrAhmed/Ongoing/WGS/Result/_PRIMARY_RESULTS/AMRFinder_summary_166.csv"
OUT="/mnt/e/DrAhmed/Ongoing/WGS/Result/_PRIMARY_RESULTS/PrimaryResults_v2_166.csv"

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
fields=list(next(iter(master.values())).keys()) + ["MLST_Preview","AMRFinder_Hits"]

with open(OUT,"w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for s in samples:
        row=dict(master[s])
        row["MLST_Preview"] = mlst.get(s, {}).get("MLST_Preview","")
        row["AMRFinder_Hits"] = amr.get(s, {}).get("AMRFinder_Hits","")
        w.writerow(row)

print("Merged samples:", len(samples))
print("Wrote:", OUT)
