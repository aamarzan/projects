import csv, os
from collections import Counter, defaultdict

WORK="/mnt/e/DrAhmed/Ongoing/WGS/Result"
MLST_LIST=f"{WORK}/_PRIMARY_RESULTS/MLST_summary_166.csv"
AMR_LIST=f"{WORK}/_PRIMARY_RESULTS/AMRFinder_summary_166.csv"

OUT_MLST=f"{WORK}/_PRIMARY_RESULTS/MLST_clean_166.csv"
OUT_AMR=f"{WORK}/_PRIMARY_RESULTS/AMRFinder_clean_166.csv"

# -------------------------
# MLST
# -------------------------
mlst_rows=[]
with open(MLST_LIST,"r",encoding="utf-8",errors="replace") as f:
    for row in csv.DictReader(f):
        sample=row["Sample"]
        preview=row.get("MLST_Preview","").strip()

        parts=preview.split()
        scheme=""
        st=""
        alleles=""

        if len(parts) >= 3:
            # first token is filename
            scheme=parts[1]
            st=parts[2]
            alleles=" ".join(parts[3:])

        mlst_rows.append({
            "Sample": sample,
            "MLST_Scheme": scheme,
            "MLST_ST": st,
            "MLST_Alleles": alleles
        })

with open(OUT_MLST,"w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f, fieldnames=["Sample","MLST_Scheme","MLST_ST","MLST_Alleles"])
    w.writeheader()
    for r in sorted(mlst_rows, key=lambda x: x["Sample"]):
        w.writerow(r)

# -------------------------
# AMRFinder
# -------------------------
amr_rows=[]
with open(AMR_LIST,"r",encoding="utf-8",errors="replace") as f:
    amr_files={row["Sample"]: row["AMRFinder_File"] for row in csv.DictReader(f)}

for sample, fp in sorted(amr_files.items()):
    total_hits=0
    genes=Counter()
    classes=Counter()

    try:
        with open(fp,"r",encoding="utf-8",errors="replace") as fh:
            reader=csv.DictReader(fh, delimiter="\t")
            for r in reader:
                total_hits += 1
                gene=(r.get("Element symbol") or "").strip()
                cl=(r.get("Class") or "").strip()
                if gene:
                    genes[gene] += 1
                if cl:
                    classes[cl] += 1

        top_genes="; ".join([f"{k}({v})" for k,v in genes.most_common(10)])
        class_summary="; ".join([f"{k}({v})" for k,v in classes.most_common(10)])

    except Exception as e:
        total_hits=""
        top_genes=f"ERROR: {e}"
        class_summary=""

    amr_rows.append({
        "Sample": sample,
        "AMRFinder_Hits": total_hits,
        "AMR_Classes": class_summary,
        "AMR_TopGenes": top_genes
    })

with open(OUT_AMR,"w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f, fieldnames=["Sample","AMRFinder_Hits","AMR_Classes","AMR_TopGenes"])
    w.writeheader()
    for r in amr_rows:
        w.writerow(r)

print("Wrote:", OUT_MLST)
print("Wrote:", OUT_AMR)
print("MLST rows:", len(mlst_rows))
print("AMR rows:", len(amr_rows))
