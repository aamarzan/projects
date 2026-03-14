import glob, os, csv, re
from collections import Counter

def parse_report(fp):
    rows=[]
    with open(fp, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts=line.rstrip("\n").split("\t")
            if len(parts) < 6: 
                continue
            pct=float(parts[0])
            rank=parts[3].strip()
            name=parts[5].strip()
            rows.append((pct, rank, name))
    return rows

def get_unclassified(rows):
    for pct, rank, name in rows:
        if rank == "U" or name.lower() == "unclassified":
            return pct
    return 0.0

def top_species(rows):
    # Accept S, S1, S2...
    sp=[(pct,name) for pct,rank,name in rows if re.match(r"^S", rank)]
    sp.sort(reverse=True, key=lambda x: x[0])
    if not sp:
        return ("",0.0,"",0.0)
    top1=sp[0]
    top2=sp[1] if len(sp)>1 else ("",0.0)
    return (top1[1], top1[0], top2[1], top2[0])

reports = sorted(glob.glob("reports/*.txt"))
out_rows=[]
counter=Counter()

for rp in reports:
    base=os.path.basename(rp)
    sample=re.sub(r"\.kraken2\.report\.txt$", "", base)
    sample=re.sub(r"\.report\.txt$", "", sample)
    sample=re.sub(r"\.txt$", "", sample)

    rows=parse_report(rp)
    uncls=get_unclassified(rows)
    t1, t1p, t2, t2p = top_species(rows)

    if t1:
        counter[t1]+=1

    out_rows.append({
        "Sample": sample,
        "TopSpecies": t1,
        "TopSpeciesPct": round(t1p,3),
        "Top2Species": t2,
        "Top2SpeciesPct": round(t2p,3),
        "UnclassifiedPct": round(uncls,3),
        "ReportFile": base
    })

with open("KRAKEN_existing_top_species_per_sample.csv","w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f, fieldnames=list(out_rows[0].keys()) if out_rows else ["Sample"])
    if out_rows: w.writeheader(); w.writerows(out_rows)

with open("KRAKEN_existing_unique_species.csv","w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f, fieldnames=["Species","SamplesWithThisTopSpecies"])
    w.writeheader()
    for sp,c in counter.most_common():
        w.writerow({"Species": sp, "SamplesWithThisTopSpecies": c})

print("Reports processed:", len(reports))
print("Unique TopSpecies:", len(counter))
