from pathlib import Path
import matplotlib.pyplot as plt
import csv

I9 = Path("/mnt/e/Bacillus_WGS/bacillus_subtilis/06_reports_wsl/I-9")
outdir = I9 / "figures"
outdir.mkdir(parents=True, exist_ok=True)

files = [
    ("B. paranthracis", I9 / "qc_clean" / "ANI_bpar.tsv"),
    ("B. cereus", I9 / "qc_clean" / "ANI_bcereus.tsv"),
    ("B. thuringiensis", I9 / "qc_clean" / "ANI_bthuringiensis.tsv"),
]

rows = []
for label, fp in files:
    with fp.open() as f:
        rdr = csv.DictReader(f, delimiter="\t")
        r = next(rdr)
    rows.append((label, float(r["ANI"]), float(r["Align_fraction_ref"]), float(r["Align_fraction_query"])))

rows.sort(key=lambda x: x[1])  # sort by ANI
labels = [r[0] for r in rows]
ani    = [r[1] for r in rows]

with (outdir / "Fig03_ANI_comparison_table.tsv").open("w") as w:
    w.write("reference\tANI\tAF_ref\tAF_query\n")
    for r in rows:
        w.write(f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]}\n")

plt.figure(figsize=(9,5))
plt.barh(labels, ani)
plt.xlabel("ANI (%)")
plt.title("ANI of I-9 (Bacillus-only) vs RefSeq references")
plt.tight_layout()
plt.savefig(outdir / "Fig03_ANI_comparison.pdf")
plt.savefig(outdir / "Fig03_ANI_comparison.svg")
print("Saved Fig03")
