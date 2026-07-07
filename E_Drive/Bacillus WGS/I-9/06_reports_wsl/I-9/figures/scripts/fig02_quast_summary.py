from pathlib import Path
import matplotlib.pyplot as plt

I9 = Path("/mnt/e/Bacillus_WGS/bacillus_subtilis/06_reports_wsl/I-9")
outdir = I9 / "figures"
outdir.mkdir(parents=True, exist_ok=True)

tsv = I9 / "quast_I-9" / "report.tsv"
rows = []
with tsv.open() as f:
    for line in f:
        line = line.rstrip("\n")
        if not line:
            continue
        parts = line.split("\t")
        rows.append(parts)

# QUAST report.tsv usually: metric_name \t value   OR metric_name \t assembly_value
# We take first column as key and last column as value.
kv = {}
for parts in rows[1:]:
    if len(parts) < 2:
        continue
    k = parts[0].strip()
    v = parts[-1].strip()
    kv[k] = v

def num(x):
    return float(str(x).replace(",","").strip())

def get_any(keys):
    for k in keys:
        if k in kv:
            return kv[k]
    return "0"

total_len = num(get_any(["Total length (>= 0 bp)", "Total length (>= 500 bp)"]))
n50       = num(get_any(["N50"]))
contigs   = num(get_any(["# contigs (>= 0 bp)", "# contigs (>= 500 bp)"]))
gc        = num(get_any(["GC (%)"]))

# save table
with (outdir / "Fig02_quast_summary.tsv").open("w") as w:
    w.write("metric\tvalue\n")
    w.write(f"Total_length_bp\t{total_len}\n")
    w.write(f"N50_bp\t{n50}\n")
    w.write(f"Contigs\t{contigs}\n")
    w.write(f"GC_percent\t{gc}\n")

# plot (scaled bars split into 2 plots so it looks good)
plt.figure(figsize=(8,4))
plt.bar(["Total length (bp)", "N50 (bp)"], [total_len, n50])
plt.title("I-9 assembly length metrics (QUAST)")
plt.tight_layout()
plt.savefig(outdir / "Fig02A_quast_length_metrics.pdf")
plt.savefig(outdir / "Fig02A_quast_length_metrics.svg")

plt.figure(figsize=(8,4))
plt.bar(["Contigs", "GC (%)"], [contigs, gc])
plt.title("I-9 assembly count/GC metrics (QUAST)")
plt.tight_layout()
plt.savefig(outdir / "Fig02B_quast_count_gc.pdf")
plt.savefig(outdir / "Fig02B_quast_count_gc.svg")

print("Saved Fig02A/Fig02B")
