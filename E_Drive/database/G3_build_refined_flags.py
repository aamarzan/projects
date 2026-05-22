import csv, os

WORK="/mnt/e/DrAhmed/Ongoing/WGS/Result"
MASTER=f"{WORK}/_PRIMARY_RESULTS/PrimaryResults_v3_166.csv"
BADASM=f"{WORK}/_ASSEMBLY_QC2/assemblies_manifest_STILL_BAD.tsv"
OUT=f"{WORK}/_PRELIMINARY_PACK/Table_FlaggedSamples_refined.csv"

bad_asm=set()
with open(BADASM,"r",encoding="utf-8",errors="replace") as f:
    next(f, None)
    for line in f:
        if line.strip():
            bad_asm.add(line.split("\t")[0].strip())

rows=[]
with open(MASTER,"r",encoding="utf-8",errors="replace") as f:
    for r in csv.DictReader(f):
        sample=r["Sample"]
        reasons=[]

        unclass=float(r.get("UnclassifiedPct") or 0)
        top1=(r.get("TopSpecies1") or "").strip().lower()
        top1pct=float(r.get("TopSpecies1Pct") or 0)

        if sample in bad_asm:
            reasons.append("BadAssembly")
        if unclass >= 10:
            reasons.append("HighUnclassified>=10")
        if top1 == "homo sapiens":
            reasons.append("HumanTaxon")
        if top1 == "serratia nevei":
            reasons.append("UnusualTopSpecies")
        if top1pct < 40:
            reasons.append("VeryLowTopSpeciesPct<40")

        if reasons:
            rr=dict(r)
            rr["FlagReason"]=";".join(reasons)
            rows.append(rr)

fields=list(rows[0].keys()) if rows else []
with open(OUT,"w",newline="",encoding="utf-8") as f:
    if fields:
        w=csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

print("Refined flagged samples:", len(rows))
print("Wrote:", OUT)
