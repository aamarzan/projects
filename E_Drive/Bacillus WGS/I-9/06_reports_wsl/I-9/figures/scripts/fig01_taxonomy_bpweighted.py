from pathlib import Path
import matplotlib.pyplot as plt

I9 = Path("/mnt/e/Bacillus_WGS/bacillus_subtilis/06_reports_wsl/I-9")
outdir = I9 / "figures"
outdir.mkdir(parents=True, exist_ok=True)

kraken_out = I9 / "qc_clean" / "I-9.allcontigs.kraken2.out.txt"

bp = {}
tot = 0
with kraken_out.open() as f:
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 4:
            continue
        status, _, taxname, seqlen = parts[0], parts[1], parts[2], int(parts[3])
        if status != "C":
            continue
        tot += seqlen
        bp[taxname] = bp.get(taxname, 0) + seqlen

items = sorted(bp.items(), key=lambda x: x[1], reverse=True)
topn = 12
top = items[:topn]
other_bp = sum(v for _, v in items[topn:])
plot_items = top + ([("Other", other_bp)] if other_bp > 0 else [])

labels = [k for k, _ in plot_items][::-1]
pcts   = [(100*v/tot) for _, v in plot_items][::-1]

# save table
with (outdir / "Fig01_taxonomy_bpweighted_allcontigs.tsv").open("w") as w:
    w.write("taxon\tbp\tpct\n")
    for k, v in plot_items:
        w.write(f"{k}\t{v}\t{100*v/tot:.4f}\n")

plt.figure(figsize=(10,6))
plt.barh(labels, pcts)
plt.xlabel("Percent of assembled bp (Kraken2 assignment)")
plt.ylabel("Taxon")
plt.title("I-9 assembly composition (bp-weighted, all contigs)")
plt.tight_layout()
plt.savefig(outdir / "Fig01_taxonomy_bpweighted_allcontigs.pdf")
plt.savefig(outdir / "Fig01_taxonomy_bpweighted_allcontigs.svg")
print("Saved Fig01")
