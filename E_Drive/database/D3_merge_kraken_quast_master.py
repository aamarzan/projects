import csv, os

KR="/mnt/e/DrAhmed/Ongoing/WGS/Result/_KRAKEN_BEST_166/KRAKEN_top_species_per_sample.csv"
QU="/mnt/e/DrAhmed/Ongoing/WGS/Result/_QUAST_BEST_166/QUAST_summary_per_sample.csv"
OUT="/mnt/e/DrAhmed/Ongoing/WGS/Result/_MASTER/Kraken_QUAST_master_166.csv"

os.makedirs("/mnt/e/DrAhmed/Ongoing/WGS/Result/_MASTER", exist_ok=True)

kr={}
with open(KR,"r",encoding="utf-8",errors="replace") as f:
    for row in csv.DictReader(f):
        kr[row["Sample"]] = row

qu={}
with open(QU,"r",encoding="utf-8",errors="replace") as f:
    for row in csv.DictReader(f):
        qu[row["Sample"]] = row

samples=sorted(set(kr.keys()) | set(qu.keys()))

fields=[
    "Sample",
    "UnclassifiedPct","TopSpecies1","TopSpecies1Pct","TopSpecies2","TopSpecies2Pct",
    "Contigs","LargestContig","TotalLength","GC_percent","N50","N75","L50","L75"
]

with open(OUT,"w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for s in samples:
        row={"Sample": s}
        if s in kr:
            row.update({
                "UnclassifiedPct": kr[s].get("UnclassifiedPct",""),
                "TopSpecies1": kr[s].get("TopSpecies1",""),
                "TopSpecies1Pct": kr[s].get("TopSpecies1Pct",""),
                "TopSpecies2": kr[s].get("TopSpecies2",""),
                "TopSpecies2Pct": kr[s].get("TopSpecies2Pct",""),
            })
        if s in qu:
            row.update({
                "Contigs": qu[s].get("Contigs",""),
                "LargestContig": qu[s].get("LargestContig",""),
                "TotalLength": qu[s].get("TotalLength",""),
                "GC_percent": qu[s].get("GC_percent",""),
                "N50": qu[s].get("N50",""),
                "N75": qu[s].get("N75",""),
                "L50": qu[s].get("L50",""),
                "L75": qu[s].get("L75",""),
            })
        w.writerow(row)

print("Merged samples:", len(samples))
print("Wrote:", OUT)
