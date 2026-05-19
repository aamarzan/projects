from pathlib import Path
import csv
import matplotlib.pyplot as plt

I9 = Path("/mnt/e/Bacillus_WGS/bacillus_subtilis/06_reports_wsl/I-9")
outdir = I9 / "figures"
outdir.mkdir(parents=True, exist_ok=True)

counts = I9 / "tables_bacillus_only" / "I-9.bacillus_only.AMR_counts.tsv"
tools = []
hits = []

with counts.open() as f:
    rdr = csv.DictReader(f, delimiter="\t")
    for r in rdr:
        tools.append(r["tool"])
        hits.append(int(r["hits"]))

plt.figure(figsize=(9,5))
plt.bar(tools, hits)
plt.xticks(rotation=30, ha="right")
plt.ylabel("Number of hits")
plt.title("AMR/VFDB hits (Bacillus-only assembly)")
plt.tight_layout()
plt.savefig(outdir / "Fig04_AMR_counts.pdf")
plt.savefig(outdir / "Fig04_AMR_counts.svg")
print("Saved Fig04")
