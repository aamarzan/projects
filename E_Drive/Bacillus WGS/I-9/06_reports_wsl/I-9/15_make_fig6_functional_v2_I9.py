import os
import pandas as pd
import matplotlib.pyplot as plt

I9 = "/mnt/e/Bacillus WGS/bacillus_paranthracis/06_reports_wsl/I-9"
SUM = f"{I9}/07_functional_genomics/summary"
OUTDIR = f"{I9}/figures/_paper_mainfigs/fig_6_functional_genomics_v2"
os.makedirs(OUTDIR, exist_ok=True)

ko  = pd.read_csv(f"{SUM}/KOfam_KO_top50.tsv", sep="\t")
cls = pd.read_csv(f"{SUM}/dbCAN_class_counts.tsv", sep="\t")
fam = pd.read_csv(f"{SUM}/dbCAN_family_counts.tsv", sep="\t")
tox = pd.read_csv(f"{SUM}/Toxin_GI_keyword_screen.tsv", sep="\t")

# Plot choices
ko_plot  = ko.head(20).copy()
fam_plot = fam.sort_values("protein_count", ascending=False).head(20).copy()
cls_plot = cls.sort_values("protein_count", ascending=False).copy()

fig = plt.figure(figsize=(16, 10))

# A) Top KOs (counts)
ax1 = plt.subplot(2,2,1)
ax1.barh(list(reversed(ko_plot["KO"].astype(str))), list(reversed(ko_plot["count"])))
ax1.set_title("A) Top KOfam KO assignments (Top 20 by frequency)")
ax1.set_xlabel("Count")
ax1.set_ylabel("KO")

# B) CAZy class profile (protein_count)
ax2 = plt.subplot(2,2,2)
if len(cls_plot):
    ax2.bar(cls_plot["class"].astype(str), cls_plot["protein_count"])
ax2.set_title("B) CAZyme class profile (dbCAN; proteins with ≥1 hit)")
ax2.set_xlabel("CAZy class")
ax2.set_ylabel("Protein count")

# C) Top CAZy families (protein_count)
ax3 = plt.subplot(2,2,3)
ax3.barh(list(reversed(fam_plot["family"].astype(str))), list(reversed(fam_plot["protein_count"])))
ax3.set_title("C) Top CAZy families (Top 20 by proteins)")
ax3.set_xlabel("Protein count")
ax3.set_ylabel("Family")

# D) Toxin/GI keyword screen summary
ax4 = plt.subplot(2,2,4)
tox_hits = max(len(tox)-1, 0)  # header excluded
ax4.axis("off")
ax4.text(0.02, 0.75, "D) Targeted toxin/GI keyword screen", fontsize=14, fontweight="bold")
ax4.text(0.02, 0.55, f"Matched features: {tox_hits}", fontsize=12)
ax4.text(0.02, 0.35, "Note: this is a keyword-based screen\n(not a definitive absence/presence call).", fontsize=10)

plt.tight_layout()

for ext in ["png","pdf","svg"]:
    fig.savefig(f"{OUTDIR}/fig_6_functional_genomics_v2.{ext}", dpi=400)

print("Saved Fig_6 v2 to:", OUTDIR)
