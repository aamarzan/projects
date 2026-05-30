from pathlib import Path
import csv
from Bio import SeqIO
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow

I9 = Path("/mnt/e/Bacillus_WGS/bacillus_subtilis/06_reports_wsl/I-9")
outdir = I9 / "figures"
outdir.mkdir(parents=True, exist_ok=True)

gbk = Path("/mnt/e/Bacillus_WGS/_db/prokka_I9_bacillus_only/I-9.bacillus_only.gbk")
amr = I9 / "amr_bacillus_only" / "I-9.bacillus_only.amrfinder.tsv"

# 1) pick contig with most AMR hits + region bounds
hits = []
with amr.open() as f:
    rdr = csv.DictReader(f, delimiter="\t")
    for r in rdr:
        contig = r.get("Contig id") or r.get("Contig") or r.get("contig")
        if not contig:
            continue
        start = int(r["Start"])
        stop  = int(r["Stop"])
        gene  = r.get("Gene symbol","")
        hits.append((contig, start, stop, gene))

if hits:
    from collections import Counter
    top_contig = Counter([h[0] for h in hits]).most_common(1)[0][0]
    h2 = [h for h in hits if h[0] == top_contig]
    region_start = max(0, min(h[1] for h in h2) - 5000)
    region_end   = max(h[2] for h in h2) + 5000
else:
    top_contig = None

# 2) load genbank and pick record
records = list(SeqIO.parse(str(gbk), "genbank"))
rec = None
if top_contig:
    for r in records:
        if r.id == top_contig or r.name == top_contig or r.description.split()[0] == top_contig:
            rec = r
            break

if rec is None:
    # fallback: largest contig
    records.sort(key=lambda x: len(x.seq), reverse=True)
    rec = records[0]
    region_start = 0
    region_end = min(len(rec.seq), 60000)

seq_len = len(rec.seq)
region_end = min(region_end, seq_len)

# 3) collect CDS features in region
cds = []
for feat in rec.features:
    if feat.type != "CDS":
        continue
    s = int(feat.location.start)
    e = int(feat.location.end)
    if e < region_start or s > region_end:
        continue
    strand = feat.location.strand or 1
    cds.append((s, e, strand))

# 4) AMR labels in this contig/region
amr_labels = []
for c, s, e, g in hits:
    if (c == rec.id or c == rec.name or c in rec.description) and not (e < region_start or s > region_end):
        amr_labels.append((s, e, g))

# 5) plot
plt.figure(figsize=(16,4))
ax = plt.gca()
ax.set_xlim(region_start, region_end)
ax.set_ylim(-2.5, 2.5)
ax.set_yticks([])
ax.set_xlabel("Genomic position (bp)")
ax.set_title(f"I-9 Bacillus-only AMR region gene map: {rec.id} ({region_start:,}-{region_end:,} bp)")

def draw_arrow(x1, x2, y, strand, height=0.8):
    width = max(1, x2 - x1)
    if strand >= 0:
        ax.add_patch(FancyArrow(x1, y-height/2, width, 0, width=height, length_includes_head=True))
    else:
        ax.add_patch(FancyArrow(x2, y-height/2, -width, 0, width=height, length_includes_head=True))

# CDS arrows (two tracks by strand)
for s, e, strand in cds:
    y = 1.0 if strand >= 0 else -1.0
    draw_arrow(s, e, y, strand)

# AMR labels (top)
for s, e, g in amr_labels:
    mid = (s+e)//2
    ax.text(mid, 2.0, g, ha="center", va="bottom", fontsize=8, rotation=0)

plt.tight_layout()
plt.savefig(outdir / "Fig05_AMR_region_genemap.pdf")
plt.savefig(outdir / "Fig05_AMR_region_genemap.svg")
print("Saved Fig05")
