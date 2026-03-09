from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

I9 = Path("/mnt/e/Bacillus_WGS/bacillus_subtilis/06_reports_wsl/I-9")
outdir = I9 / "figures"
outdir.mkdir(parents=True, exist_ok=True)

quast_tsv = I9 / "quast_I-9" / "report.tsv"
df = pd.read_csv(quast_tsv, sep="\t")

# report.tsv is 2-column-ish with first col metric name
# Make key-value dict
kv = dict(zip(df.iloc[:,0], df.iloc[:,1]))

metrics = {
    "Total length": float(str(kv.get("Total length (>= 0 bp)", "0")).replace(",","")),
    "N50": float(str(kv.get("N50", "0")).replace(",","")),
    "Contigs": float(str(kv.get("# contigs (>= 0 bp)", "0")).replace(",","")),
    "GC%": float(str(kv.get("GC (%)", "0")).replace(",","")),
}

plotdf = pd.DataFrame({"metric": list(metrics.keys()), "value": list(metrics.values())})

plt.figure(figsize=(8,5))
plt.bar(plotdf["metric"], plotdf["value"])
plt.ylabel("Value")
plt.title("I-9 assembly summary (QUAST)")
plt.tight_layout()
plt.savefig(outdir / "Fig02_quast_summary.pdf")
plt.savefig(outdir / "Fig02_quast_summary.svg")
print("Saved Fig02")
