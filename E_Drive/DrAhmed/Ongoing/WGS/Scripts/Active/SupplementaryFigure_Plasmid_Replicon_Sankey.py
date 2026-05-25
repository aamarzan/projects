import os
from collections import Counter

from figure_helper_wgs_remaining import read_csv, norm, as_int, ensure_dir

WORK = "/mnt/e/DrAhmed/Ongoing/WGS/Result"
CLEAN = f"{WORK}/_G3/clean"
OUTDIR = f"{WORK}/_G4_REMAINING/output/supplementary"
ensure_dir(OUTDIR)

rows = read_csv(f"{CLEAN}/plasmid_replicon_by_species_166.csv")

species = []
replicons = []
links = []

for r in rows:
    sp = norm(r.get("TopSpecies", ""))
    rp = norm(r.get("Plasmid_Replicon", ""))
    ct = as_int(r.get("Count", 0))
    if ct <= 0:
        continue
    if sp not in species:
        species.append(sp)
    if rp not in replicons:
        replicons.append(rp)
    links.append((sp, rp, ct))

rep_total = Counter()
for sp, rp, ct in links:
    rep_total[rp] += ct

top_rep = {rp for rp, _ in rep_total.most_common(12)}
links = [(sp, rp, ct) for sp, rp, ct in links if rp in top_rep]
replicons = [rp for rp in replicons if rp in top_rep]

try:
    import plotly.graph_objects as go
except Exception as e:
    raise SystemExit(f"plotly is required for this script: {e}")

labels = species + replicons
idx = {x: i for i, x in enumerate(labels)}

source = [idx[sp] for sp, rp, ct in links]
target = [idx[rp] for sp, rp, ct in links]
value = [ct for sp, rp, ct in links]

node_colors = ["#1f77b4"] * len(species) + ["#8b5cf6"] * len(replicons)

fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=18,
        thickness=20,
        line=dict(color="rgba(40,40,40,0.3)", width=0.5),
        label=labels,
        color=node_colors,
    ),
    link=dict(
        source=source,
        target=target,
        value=value,
        color="rgba(139,92,246,0.35)"
    )
)])

fig.update_layout(
    title="Plasmid replicon flow from species to dominant replicons",
    font=dict(size=12),
    width=1400,
    height=800,
    paper_bgcolor="white",
    plot_bgcolor="white",
    margin=dict(l=20, r=20, t=60, b=20),
)

html_path = os.path.join(OUTDIR, "SupplementaryFigure_Plasmid_Replicon_Sankey.html")
fig.write_html(html_path)

try:
    png_path = os.path.join(OUTDIR, "SupplementaryFigure_Plasmid_Replicon_Sankey.png")
    pdf_path = os.path.join(OUTDIR, "SupplementaryFigure_Plasmid_Replicon_Sankey.pdf")
    fig.write_image(png_path, scale=2)
    fig.write_image(pdf_path, scale=2)
    print("Saved HTML, PNG, and PDF to:", OUTDIR)
except Exception:
    print("Saved HTML to:", html_path)
    print("Static export skipped because plotly image export backend is unavailable.")