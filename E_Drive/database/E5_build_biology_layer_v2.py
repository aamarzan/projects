import os, csv

WORK = "/mnt/e/DrAhmed/Ongoing/WGS/Result"
ROOT = f"{WORK}/Result copy/tormes_all+plasmid+serotype(Enterobacteriaceae)"
MASTER = f"{WORK}/_PRIMARY_RESULTS/PrimaryResults_v3_166.csv"
OUTDIR = f"{WORK}/_BIOLOGY_LAYER_V2"
os.makedirs(OUTDIR, exist_ok=True)

VFDB_DIR = f"{ROOT}/virulence_genes"
PLASMID_DIR = f"{ROOT}/plasmids"
SERO_DIR = f"{ROOT}/serotyping"

def read_lines(fp):
    try:
        with open(fp, "r", encoding="utf-8", errors="replace") as f:
            return [x.rstrip("\n") for x in f]
    except:
        return []

def parse_abricate_tab(fp):
    """
    Parse TORMES/Abricate-like .tab file:
    header starts with #FILE ...
    data lines are tab-separated rows after header
    """
    lines = [x for x in read_lines(fp) if x.strip()]
    if not lines:
        return 0, "", ""

    header_idx = None
    for i, ln in enumerate(lines):
        if ln.startswith("#FILE") or ln.lstrip("#").startswith("FILE\t"):
            header_idx = i
            break

    if header_idx is None:
        # fallback: no recognizable header, treat all non-comment lines as data
        data = [ln for ln in lines if not ln.startswith("#")]
        header = []
    else:
        header = lines[header_idx].lstrip("#").split("\t")
        data = lines[header_idx + 1:]

    if not data:
        return 0, "", ""

    rows = [d.split("\t") for d in data if d.strip()]
    if not rows:
        return 0, "", ""

    gene_idx = None
    prod_idx = None
    if header:
        h = [x.strip().lower() for x in header]
        if "gene" in h:
            gene_idx = h.index("gene")
        if "product" in h:
            prod_idx = h.index("product")

    genes = []
    products = []
    for r in rows:
        if gene_idx is not None and gene_idx < len(r):
            g = r[gene_idx].strip()
            if g:
                genes.append(g)
        if prod_idx is not None and prod_idx < len(r):
            p = r[prod_idx].strip()
            if p:
                products.append(p)

    # de-duplicate while preserving order
    def uniq_keep_order(vals):
        seen = set()
        out = []
        for v in vals:
            if v not in seen:
                seen.add(v)
                out.append(v)
        return out

    genes = uniq_keep_order(genes)
    products = uniq_keep_order(products)

    return len(rows), "; ".join(genes[:15]), "; ".join(products[:8])

def parse_serotype(fp, sample):
    lines = [x.strip() for x in read_lines(fp) if x.strip()]
    if not lines:
        return "No", ""

    # if file only contains sample ID, it is non-informative
    if len(lines) == 1 and lines[0] == sample:
        return "No", ""

    # otherwise keep first few informative lines
    keep = lines[:5]
    return "Yes", " | ".join(keep)

with open(MASTER, "r", encoding="utf-8", errors="replace") as f:
    master_rows = list(csv.DictReader(f))

bio_rows = []
for r in master_rows:
    s = r["Sample"]
    vf = f"{VFDB_DIR}/{s}_vfdb.tab"
    pl = f"{PLASMID_DIR}/{s}_plasmids.tab"
    se = f"{SERO_DIR}/{s}/{s}_serotype"

    vf_hits, vf_genes, vf_products = parse_abricate_tab(vf) if os.path.isfile(vf) else (0, "", "")
    pl_hits, pl_genes, pl_products = parse_abricate_tab(pl) if os.path.isfile(pl) else (0, "", "")
    sero_ok, sero_text = parse_serotype(se, s) if os.path.isfile(se) else ("No", "")

    bio_rows.append({
        "Sample": s,
        "VFDB_File": "Yes" if os.path.isfile(vf) else "No",
        "VFDB_Hits": vf_hits,
        "VFDB_Genes": vf_genes,
        "VFDB_Products": vf_products,
        "Plasmid_File": "Yes" if os.path.isfile(pl) else "No",
        "Plasmid_Hits": pl_hits,
        "Plasmid_Replicons": pl_genes,
        "Plasmid_Products": pl_products,
        "Serotype_Informative": sero_ok,
        "Serotype_Call": sero_text,
    })

bio_csv = f"{OUTDIR}/BiologyLayer_summary_166.csv"
with open(bio_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(bio_rows[0].keys()))
    w.writeheader()
    w.writerows(bio_rows)

bio_map = {r["Sample"]: r for r in bio_rows}
merged = []
for r in master_rows:
    rr = dict(r)
    rr.update(bio_map[r["Sample"]])
    merged.append(rr)

merged_csv = f"{OUTDIR}/PrimaryResults_v4_withBiology_166.csv"
with open(merged_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(merged[0].keys()))
    w.writeheader()
    w.writerows(merged)

n_vf_files = sum(1 for r in bio_rows if r["VFDB_File"] == "Yes")
n_vf_hits  = sum(1 for r in bio_rows if int(r["VFDB_Hits"]) > 0)
n_pl_files = sum(1 for r in bio_rows if r["Plasmid_File"] == "Yes")
n_pl_hits  = sum(1 for r in bio_rows if int(r["Plasmid_Hits"]) > 0)
n_sero     = sum(1 for r in bio_rows if r["Serotype_Informative"] == "Yes")

with open(f"{OUTDIR}/BiologyLayer_summary.txt", "w", encoding="utf-8") as f:
    f.write(f"Samples in master: {len(master_rows)}\n")
    f.write(f"Samples with VFDB file: {n_vf_files}\n")
    f.write(f"Samples with VFDB hits > 0: {n_vf_hits}\n")
    f.write(f"Samples with plasmid file: {n_pl_files}\n")
    f.write(f"Samples with plasmid hits > 0: {n_pl_hits}\n")
    f.write(f"Samples with informative serotype call: {n_sero}\n")

print("Wrote:", bio_csv)
print("Wrote:", merged_csv)
print("Samples in master:", len(master_rows))
print("Samples with VFDB file:", n_vf_files)
print("Samples with VFDB hits > 0:", n_vf_hits)
print("Samples with plasmid file:", n_pl_files)
print("Samples with plasmid hits > 0:", n_pl_hits)
print("Samples with informative serotype call:", n_sero)
