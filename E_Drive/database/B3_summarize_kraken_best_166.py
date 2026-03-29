import glob, os, re, csv
from collections import Counter

REPORT_DIR="/mnt/e/DrAhmed/Ongoing/WGS/Result/_KRAKEN_BEST_166/reports"
out1="/mnt/e/DrAhmed/Ongoing/WGS/Result/_KRAKEN_BEST_166/KRAKEN_top_species_per_sample.csv"
out2="/mnt/e/DrAhmed/Ongoing/WGS/Result/_KRAKEN_BEST_166/KRAKEN_unique_top_species.csv"

def parse_report(fp):
    rows=[]
    with open(fp,"r",encoding="utf-8",errors="replace") as f:
        for line in f:
            parts=line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            try:
                pct=float(parts[0])
            except:
                continue
            rank=parts[3].strip()
            name=parts[5].strip()
            rows.append((pct, rank, name))
    return rows

def unclassified_pct(rows):
    for pct,rank,name in rows:
        if rank=="U" or name.lower()=="unclassified":
            return pct
    return 0.0

def top_species(rows):
    sp=[(pct,name) for pct,rank,name in rows if re.match(r"^S", rank)]
    sp.sort(reverse=True, key=lambda x: x[0])
    if not sp:
        return ("",0.0,"",0.0)
    t1=sp[0]
    t2=sp[1] if len(sp)>1 else ("",0.0)
    return (t1[1],t1[0],t2[1],t2[0])

reports=sorted(glob.glob(os.path.join(REPORT_DIR,"*.txt")))
counter=Counter()
rows_out=[]

for rp in reports:
    sample=os.path.basename(rp).replace(".kraken.report.txt","")
    rows=parse_report(rp)
    u=unclassified_pct(rows)
    s1,p1,s2,p2=top_species(rows)
    if s1:
        counter[s1]+=1
    rows_out.append([sample,u,s1,p1,s2,p2,rp])

with open(out1,"w",newline="",encoding="utf-8") as f:
    w=csv.writer(f)
    w.writerow(["Sample","UnclassifiedPct","TopSpecies1","TopSpecies1Pct","TopSpecies2","TopSpecies2Pct","ReportPath"])
    w.writerows(sorted(rows_out,key=lambda x:x[0]))

with open(out2,"w",newline="",encoding="utf-8") as f:
    w=csv.writer(f)
    w.writerow(["TopSpecies","CountSamples"])
    for sp,c in counter.most_common():
        w.writerow([sp,c])

print("Reports processed:", len(reports))
print("Wrote:", out1)
print("Wrote:", out2)
