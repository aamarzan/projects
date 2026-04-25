import os
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from figure_helper_wgs_remaining import (
    setup_rcparams, read_csv, ensure_dir, norm, coerce_age,
    style_ax, palette_list, save_png_pdf, build_metadata_match_table, species_short
)

setup_rcparams()

WORK = "/mnt/e/DrAhmed/Ongoing/WGS/Result"
MASTER = f"{WORK}/_BIOLOGY_LAYER_V2/PrimaryResults_v4_withBiology_166.csv"
METADATA_DIR = "/mnt/e/DrAhmed/Ongoing/WGS/Metadata"
OUTDIR = f"{WORK}/_G4_REMAINING/output/main"
ensure_dir(OUTDIR)

master_rows = read_csv(MASTER)
matched = build_metadata_match_table(master_rows, METADATA_DIR)

if not matched:
    raise SystemExit(f"No matched metadata rows found under: {METADATA_DIR}")

source_counts = Counter(r["MetadataSource"] for r in matched if norm(r["MetadataSource"]))
specimen_counts = Counter(norm(r["Specimen"]) or "Unknown" for r in matched)
species_age = defaultdict(list)
for r in matched:
    age = coerce_age(r.get("Age", ""))
    if age is not None:
        species_age[norm(r["TopSpecies1"])].append(age)

outcome_counts = Counter(norm(r["Outcome"]) or "Unknown" for r in matched)
ward_counts = Counter(norm(r["Ward_or_Unit"]) or "Unknown" for r in matched)

fig = plt.figure(figsize=(16.2, 10.4))
gs = GridSpec(2, 2, figure=fig, hspace=0.34, wspace=0.24)

ax = fig.add_subplot(gs[0, 0])
src_labels = list(source_counts.keys())
src_vals = [source_counts[x] for x in src_labels]
bars = ax.barh(src_labels, src_vals, color=palette_list("slate", len(src_vals)), edgecolor="#243447", linewidth=0.7, zorder=3)
ax.invert_yaxis()
style_ax(ax, "A. Metadata source coverage among matched samples", "Matched samples", None, "x")
for b in bars:
    ax.text(b.get_width() + max(src_vals)*0.02, b.get_y()+b.get_height()/2, f"{int(b.get_width())}", va="center", ha="left", fontsize=9)

ax = fig.add_subplot(gs[0, 1])
spec_items = sorted(specimen_counts.items(), key=lambda x: (-x[1], x[0]))[:12]
labs = [x[0] for x in spec_items]
vals = [x[1] for x in spec_items]
bars = ax.barh(labs, vals, color=palette_list("teal", len(vals)), edgecolor="#0d5c63", linewidth=0.7, zorder=3)
ax.invert_yaxis()
style_ax(ax, "B. Specimen distribution among matched samples", "Samples", None, "x")
for b in bars:
    ax.text(b.get_width() + max(vals)*0.02, b.get_y()+b.get_height()/2, f"{int(b.get_width())}", va="center", ha="left", fontsize=9)

ax = fig.add_subplot(gs[1, 0])
items = [(sp, ages) for sp, ages in species_age.items() if len(ages) > 0]
items.sort(key=lambda x: (-len(x[1]), x[0]))
data = [x[1] for x in items]
pos = list(range(1, len(items) + 1))
labels = [f"{species_short(sp)}\n(n={len(ages)})" for sp, ages in items]
bp = ax.boxplot(
    data, positions=pos, patch_artist=True, widths=0.6,
    medianprops=dict(color="#111827", linewidth=1.5),
    boxprops=dict(linewidth=0.9, color="#4b5563"),
    whiskerprops=dict(linewidth=0.9, color="#6b7280"),
    capprops=dict(linewidth=0.9, color="#6b7280"),
    flierprops=dict(marker="o", markersize=0)
)
cols = palette_list("rose", len(pos))
for box, c in zip(bp["boxes"], cols):
    box.set_facecolor(c)
    box.set_alpha(0.42)
ax.set_xticks(pos)
ax.set_xticklabels(labels)
style_ax(ax, "C. Age distribution across matched species groups", None, "Age (years)", "y")

ax = fig.add_subplot(gs[1, 1])
comb_items = sorted(outcome_counts.items(), key=lambda x: (-x[1], x[0]))[:6]
ward_items = sorted(ward_counts.items(), key=lambda x: (-x[1], x[0]))[:6]
labels = [f"Outcome: {k}" for k, _ in comb_items] + [f"Unit: {k}" for k, _ in ward_items]
vals = [v for _, v in comb_items] + [v for _, v in ward_items]
bars = ax.barh(labels, vals, color=palette_list("green", len(vals)), edgecolor="#0f5b26", linewidth=0.7, zorder=3)
ax.invert_yaxis()
style_ax(ax, "D. Outcome and care-unit overview", "Samples", None, "x")
for b in bars:
    ax.text(b.get_width() + max(vals)*0.02, b.get_y()+b.get_height()/2, f"{int(b.get_width())}", va="center", ha="left", fontsize=9)

fig.suptitle("Clinical metadata integration architecture", y=0.99, fontsize=22, fontweight="bold")
fig.text(
    0.5, 0.01,
    f"Matched metadata rows: {len(matched)}. Figure quality depends on match coverage and the richness of available metadata exports under the Metadata folder.",
    ha="center", va="bottom", fontsize=10.2, color="#4b5563"
)

save_png_pdf(fig, "Figure10_Metadata_Integration_Architecture", OUTDIR)
print("Saved Figure 10 to:", OUTDIR)