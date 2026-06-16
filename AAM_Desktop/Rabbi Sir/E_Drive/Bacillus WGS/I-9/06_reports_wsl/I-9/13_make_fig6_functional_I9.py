import os
import pandas as pd
import matplotlib.pyplot as plt

I9 = "/mnt/e/Bacillus WGS/bacillus_paranthracis/06_reports_wsl/I-9"
SUM = f"{I9}/07_functional_genomics/summary"

ko = pd.read_csv(f"{SUM}/KOfam_KO_top50.tsv", sep="\t")
cls = pd.read_csv(f"{SUM}/dbCAN_class_counts.tsv", sep="\t")
fam = pd.read_csv(f"{SUM}/dbCAN_family_counts.tsv", sep="\t")
tox = pd.read_csv(f"{SUM}/Toxin_GI_keyword_screen.tsv", sep="\t")

# top slices for plotting
ko_plot = ko.head(20).copy()
fam_plot = fam.head(20).copy()
cls_plot = cls.copy()

# Create output dir inside your paper figs folder
OUTDIR = f"{I9}/figures/_paper_mainfigs/fig_6_functional_genomics"
os.makedirs(OUTDIR, exist_ok=True)

plt.figure(figsize=(16,10))

# Panel A: Top KO counts
ax1 = plt.subplot(2,2,1)
ax1.barh(list(reversed(ko_plot["KO"].astype(str))), list(reversed(ko_plot["count"])))
ax1.set_title("A) Top KOfam KO assignments (Top 20 by frequency)")
ax1.set_xlabel("Count")
ax1.set_ylabel("KO")

# Panel B: dbCAN class counts
ax2 = plt.subplot(2,2,2)
if len(cls_plot) > 0:
    ax2.bar(cls_plot["class"].astype(str), cls_plot["count"])
ax2.set_title("B) CAZyme class profile (dbCAN)")
ax2.set_xlabel("CAZy class")
ax2.set_ylabel("Hit count")

# Panel C: Top CAZy families
ax3 = plt.subplot(2,2,3)
ax3.barh(list(reversed(fam_plot["family"].astype(str))), list(reversed(fam_plot["count"])))
ax3.set_title("C) Top CAZy families (Top 20 by frequency)")
ax3.set_xlabel("Hit count")
ax3.set_ylabel("Family")

# Panel D: Toxin/GI keyword screen summary
ax4 = plt.subplot(2,2,4)
ax4.axis("off")
n_tox = 0 if tox.shape[0] == 0 else tox.shape[0]
txt = [
    "D) Targeted keyword screen (Bakta annotations)",
    "",
    f"Keyword-matched features: {n_tox}",
    "",
    "Note: keyword screen is not proof of virulence;",
    "it flags candidates for manual confirmation (gene context, homology)."
]
ax4.text(0.02, 0.95, "\n".join(txt), va="top", fontsize=12)

plt.tight_layout()

# Save multiple formats
for ext in ["png","pdf","svg"]:
    plt.savefig(f"{OUTDIR}/fig_6_functional_genomics.{ext}", dpi=400)

print("Saved Fig_6 to:", OUTDIR)
