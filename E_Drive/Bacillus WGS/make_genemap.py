from pygenomeviz import Genbank, GenomeViz
from pathlib import Path

gbk = "/mnt/e/Bacillus_WGS/_db/prokka_results_new/I-9.gbk"
out = Path("/mnt/e/Bacillus_WGS/bacillus_subtilis/06_reports_wsl/I-9/figures")
out.mkdir(parents=True, exist_ok=True)

gb = Genbank(gbk)

# Pick top 5 contigs by length
records = sorted(gb.records, key=lambda r: len(r.seq), reverse=True)[:5]

gv = GenomeViz(fig_width=14, track_height=1.2, feature_track_ratio=0.75)

for rec in records:
    track = gv.add_feature_track(rec.id, size=len(rec.seq))
    track.add_genbank_features(rec, feature_type="CDS", plotstyle="arrow", label_type=None)

gv.savefig(out / "I-9_genemap_top5contigs.svg")
gv.savefig(out / "I-9_genemap_top5contigs.pdf")
print("Saved:", out / "I-9_genemap_top5contigs.svg")
