import os, csv, re

LIST="/mnt/e/DrAhmed/Ongoing/WGS/Result/_INVENTORY/mlst_amrfinder_real.list.txt"
OUTDIR="/mnt/e/DrAhmed/Ongoing/WGS/Result/_PRIMARY_RESULTS"

os.makedirs(OUTDIR, exist_ok=True)

mlst_out=os.path.join(OUTDIR, "MLST_summary_166.csv")
amr_out=os.path.join(OUTDIR, "AMRFinder_summary_166.csv")

mlst_rows=[]
amr_rows=[]

def sample_from_path(p):
    parts=p.replace("\\","/").split("/")
    for i,x in enumerate(parts):
        if x == "results" and i+2 < len(parts):
            return parts[i+1]
    bn=os.path.basename(p)
    return re.sub(r"\.tsv$","",bn)

with open(LIST,"r",encoding="utf-8",errors="replace") as f:
    files=[x.strip() for x in f if x.strip()]

for fp in files:
    sample=sample_from_path(fp)
    low=fp.lower()

    if "/tools/mlst/" in low:
        try:
            with open(fp,"r",encoding="utf-8",errors="replace") as fh:
                lines=[x.rstrip("\n") for x in fh if x.strip()]
            # keep raw content compactly; exact parser can be refined later
            preview=" | ".join(lines[:3])
            mlst_rows.append({"Sample": sample, "MLST_File": fp, "MLST_Preview": preview})
        except Exception as e:
            mlst_rows.append({"Sample": sample, "MLST_File": fp, "MLST_Preview": f"ERROR: {e}"})

    elif "/tools/amrfinderplus/" in low:
        try:
            with open(fp,"r",encoding="utf-8",errors="replace") as fh:
                rows=list(csv.reader(fh, delimiter="\t"))
            n_rows=max(0, len(rows)-1)
            amr_rows.append({"Sample": sample, "AMRFinder_File": fp, "AMRFinder_Hits": n_rows})
        except Exception as e:
            amr_rows.append({"Sample": sample, "AMRFinder_File": fp, "AMRFinder_Hits": f"ERROR: {e}"})

with open(mlst_out,"w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f, fieldnames=["Sample","MLST_File","MLST_Preview"])
    w.writeheader()
    for r in sorted(mlst_rows, key=lambda x: x["Sample"]):
        w.writerow(r)

with open(amr_out,"w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f, fieldnames=["Sample","AMRFinder_File","AMRFinder_Hits"])
    w.writeheader()
    for r in sorted(amr_rows, key=lambda x: x["Sample"]):
        w.writerow(r)

print("MLST rows:", len(mlst_rows))
print("AMRFinder rows:", len(amr_rows))
print("Wrote:", mlst_out)
print("Wrote:", amr_out)
