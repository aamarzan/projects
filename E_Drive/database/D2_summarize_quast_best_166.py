import glob, os, csv

REPORT_DIR="/mnt/e/DrAhmed/Ongoing/WGS/Result/_QUAST_BEST_166/reports"
OUT="/mnt/e/DrAhmed/Ongoing/WGS/Result/_QUAST_BEST_166/QUAST_summary_per_sample.csv"

wanted = {
    "# contigs": "Contigs",
    "# contigs (>= 0 bp)": "Contigs",
    "# contigs (>= 500 bp)": "Contigs",
    "Largest contig": "LargestContig",
    "Total length": "TotalLength",
    "Total length (>= 0 bp)": "TotalLength",
    "Total length (>= 500 bp)": "TotalLength",
    "GC (%)": "GC_percent",
    "N50": "N50",
    "N75": "N75",
    "L50": "L50",
    "L75": "L75",
}

def norm_sample(fp):
    bn=os.path.basename(fp)
    return bn.replace(".transposed_report.tsv","").replace(".report.tsv","")

def parse_any_tsv(fp):
    with open(fp,"r",encoding="utf-8",errors="replace") as f:
        rows=list(csv.reader(f, delimiter="\t"))

    d={}
    if not rows:
        return d

    # orientation 1: headers on first row, values on second row
    if len(rows) >= 2 and any(h in wanted for h in rows[0]):
        headers=rows[0]
        vals=rows[1]
        for h,v in zip(headers, vals):
            if h in wanted and wanted[h] not in d:
                d[wanted[h]] = v
        return d

    # orientation 2: metric in col1, value in col2
    for r in rows:
        if len(r) >= 2:
            k=r[0].strip()
            v=r[1].strip()
            if k in wanted and wanted[k] not in d:
                d[wanted[k]] = v
    return d

files=sorted(glob.glob(os.path.join(REPORT_DIR,"*.tsv")))
rows_out=[]

for fp in files:
    sample=norm_sample(fp)
    d=parse_any_tsv(fp)
    row={
        "Sample": sample,
        "Contigs": d.get("Contigs",""),
        "LargestContig": d.get("LargestContig",""),
        "TotalLength": d.get("TotalLength",""),
        "GC_percent": d.get("GC_percent",""),
        "N50": d.get("N50",""),
        "N75": d.get("N75",""),
        "L50": d.get("L50",""),
        "L75": d.get("L75",""),
        "SourceFile": fp,
    }
    rows_out.append(row)

with open(OUT,"w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f, fieldnames=[
        "Sample","Contigs","LargestContig","TotalLength","GC_percent","N50","N75","L50","L75","SourceFile"
    ])
    w.writeheader()
    for r in sorted(rows_out, key=lambda x: x["Sample"]):
        w.writerow(r)

print("Reports processed:", len(files))
print("Wrote:", OUT)
