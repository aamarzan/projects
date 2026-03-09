import os
import pandas as pd
import matplotlib.pyplot as plt

I9 = "/mnt/e/Bacillus WGS/bacillus_paranthracis/06_reports_wsl/I-9"
SUM = f"{I9}/07_functional_genomics/summary"
OUTDIR = f"{I9}/figures/_paper_mainfigs/fig_6_functional_genomics_v3"
os.makedirs(OUTDIR, exist_ok=True)

ko  = pd.read_csv(f"{SUM}/KOfam_KO_top50.tsv", sep="\t")
cls = pd.read_csv(f"{SUM}/dbCAN_class_counts.tsv", sep="\t")
fam = pd.read_csv(f"{SUM}/dbCAN_family_simple_counts.tsv", sep="\t")
tox = pd.read_csv(f"{SUM}/Toxin_GI_keyword_screen.tsv", sep="\t")

ko_plot  = ko.head(20).copy()
cls_plot = cls.sort_values("protein_count", ascending=False).copy()
fam_plot = fam.head(20).copy()

fig = plt.figure(figsize=(16, 10))

# A) Top KOs (protein_count)
ax1 = plt.subplot(2,2,1)
ax1.barh(ko_plot["KO"].astype(str).to_list()[::-1], ko_plot["protein_count"].to_list()[::-1])
ax1.set_title("A) Top KEGG Orthologs (KOfam best-hit per protein; Top 20)")
ax1.set_xlabel("Protein count")
ax1.set_ylabel("KO")

# B) CAZy class profile
ax2 = plt.subplot(2,2,2)
ax2.bar(cls_plot["class"].astype(str), cls_plot["protein_count"])
ax2.set_title("B) CAZyme class profile (filtered; best-hit per protein)")
ax2.set_xlabel("CAZy class")
ax2.set_ylabel("Protein count")

# C) Top CAZy families
ax3 = plt.subplot(2,2,3)
ax3.barh(fam_plot["family"].astype(str).to_list()[::-1], fam_plot["protein_count"].to_list()[::-1])
ax3.set_title("C) Top CAZyme families (filtered; Top 20 by proteins)")
ax3.set_xlabel("Protein count")
ax3.set_ylabel("CAZy family")

# D) Keyword screen count (informational)
ax4 = plt.subplot(2,2,4)
n_hits = 0 if tox.empty else (len(tox) - 0)  # header handled by pandas
ax4.axis("off")
ax4.text(0.02, 0.85, "D) GI/toxin keyword screen (Bakta products/genes)", fontsize=14, weight="bold")
ax4.text(0.02, 0.65, f"Keyword matches found: {n_hits}", fontsize=12)
ax4.text(0.02, 0.48, "Note: keyword screen ≠ confirmation of toxin activity.\nUse as a prioritization layer.", fontsize=10)

plt.tight_layout()

for ext in ["png","pdf","svg"]:
    fig.savefig(f"{OUTDIR}/fig_6_functional_genomics_v3.{ext}", dpi=400)

print("Saved Fig_6 v3 to:", OUTDIR)
