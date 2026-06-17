import os
import matplotlib.pyplot as plt

from figure_helper_wgs_remaining import (
    setup_rcparams, read_csv, norm, palette_list, style_ax, save_png_pdf
)

setup_rcparams()

WORK = "/mnt/e/DrAhmed/Ongoing/WGS/Result"
CLEAN = f"{WORK}/_G3/clean"
OUTDIR = f"{WORK}/_G4_REMAINING/output/supplementary"
os.makedirs(OUTDIR, exist_ok=True)

rows = read_csv(f"{CLEAN}/serotype_distribution_166.csv")

labels = []
vals = []
for r in rows:
    sample = norm(r.get("Sample", ""))
    call = norm(r.get("Serotype_Call", "")).replace("   ", " ")
    parts = call.split()
    if len(parts) >= 2:
        label = f"{sample}\n{' '.join(parts[-2:])}"
    else:
        label = f"{sample}\n{call}"
    labels.append(label)
    vals.append(1)

fig, ax = plt.subplots(figsize=(9.5, 5.8))
bars = ax.barh(labels, vals, color=palette_list("green", len(vals)), edgecolor="#0f5b26", linewidth=0.8, zorder=3)
ax.invert_yaxis()
style_ax(ax, "Informative serotype mini-panel", "Samples", None, "x")
ax.set_xlim(0, 1.08)
for b in bars:
    ax.text(1.02, b.get_y() + b.get_height()/2, "1", va="center", ha="left", fontsize=10)

fig.suptitle("Informative serotype calls", y=0.98, fontsize=18, fontweight="bold")
save_png_pdf(fig, "SupplementaryFigure_Serotype_Mini", OUTDIR)
print("Saved supplementary serotype figure to:", OUTDIR)