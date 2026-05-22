from pathlib import Path
from Bio import SeqIO
from dna_features_viewer import GraphicFeature, GraphicRecord
import matplotlib.pyplot as plt

I9 = Path("/mnt/e/Bacillus_WGS/bacillus_subtilis/06_reports_wsl/I-9")
outdir = I9 / "figures"
outdir.mkdir(parents=True, exist_ok=True)

gbk = Path("/mnt/e/Bacillus_WGS/_db/prokka_I9_bacillus_only/I-9.bacillus_only.gbk")
records = list(SeqIO.parse(str(gbk), "genbank"))
records.sort(key=lambda r: len(r.seq), reverse=True)

rec = records[0]  # largest contig
seq_len = len(rec.seq)

features = []
for feat in rec.features:
    if feat.type != "CDS":
        continue
    start = int(feat.location.start)
    end = int(feat.location.end)
    strand = 1 if feat.location.strand != -1 else -1
    features.append(
        GraphicFeature(start=start, end=end, strand=strand, label=None)
    )

gr = GraphicRecord(sequence_length=seq_len, features=features)

fig, ax = plt.subplots(1, 1, figsize=(16, 3))
gr.plot(ax=ax, with_ruler=True)
ax.set_title(f"I-9 (Bacillus-only) gene map — largest contig ({rec.id}, {seq_len:,} bp)")
plt.tight_layout()
plt.savefig(outdir / "Fig05_genemap_largest_contig.pdf")
plt.savefig(outdir / "Fig05_genemap_largest_contig.svg")
print("Saved Fig05")
