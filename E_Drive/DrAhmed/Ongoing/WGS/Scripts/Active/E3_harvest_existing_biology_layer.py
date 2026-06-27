import os, csv, re
from collections import defaultdict

WORK = "/mnt/e/DrAhmed/Ongoing/WGS/Result"
ROOT = f"{WORK}/Result copy"
MASTER = f"{WORK}/_PRIMARY_RESULTS/PrimaryResults_v3_166.csv"
OUTDIR = f"{WORK}/_BIOLOGY_LAYER"
os.makedirs(OUTDIR, exist_ok=True)

TORMES = f"{ROOT}/tormes_all+plasmid+serotype(Enterobacteriaceae)"
VFDB_DIR = f"{TORMES}/virulence_genes"
PLASMID_DIR = f"{TORMES}/plasmids"
SERO_DIR = f"{TORMES}/serotyping"

def read_nonempty_lines(fp):
    try:
        with open(fp, "r", encoding="utf-8", errors="replace") as f:
            return [x.rstrip("\n") for x in f if x.strip()]
    except:
        return []

def parse_tab_summary(fp, kind="generic"):
    lines = read_nonempty_lines(fp)
    if not lines:
        return 0, ""

    rows = [ln.split("\t") for ln in lines if not ln.startswith("#")]
    if not rows:
        return 0, ""

    header = rows[0]
    header_l = [x.strip().lower() for x in header]

    # heuristic header detection
    likely_header = any(x in header_l for x in [
        "sequence","gene","genes","sample","file","qseqid","sseqid",
        "accession","database","contig","start","end","identity"
    ])

    data = rows[1:] if likely_header else rows
    if not data:
        return 0, ""

    gene_idx = 0
    if likely_header:
        preferred = []
        if kind == "vfdb":
            preferred = ["gene","genes","sequence","qseqid","sseqid","accession"]
        elif kind == "plasmid":
            preferred = ["plasmid","replicon","sequence","qseqid","sseqid","accession"]
        else:
            preferred = ["gene","sequence","qseqid","sseqid","accession"]
        for p in preferred:
            if p in header_l:
                gene_idx = header_l.index(p)
                break

    vals = []
    for r in data:
        if not r:
            continue
        v = ""
        if gene_idx < len(r):
            v = r[gene_idx].strip()
        if not v:
            for cell in r:
                if cell.strip():
                    v = cell.strip()
                    break
        if v:
            vals.append(v)

    uniq = []
    seen = set()
    for v in vals:
        if v not in seen:
            uniq.append(v)
            seen.add(v)

    return len(data), "; ".join(uniq[:10])

def parse_serotype_file(fp):
    lines = read_nonempty_lines(fp)
    if not lines:
        return ""
    # keep first few informative nonempty lines
    keep = []
    for ln in lines:
        ln = ln.strip()
        if ln:
            keep.append(ln)
        if len(keep) >= 5:
            break
    return " | ".join(keep)

# load master
with open(MASTER, "r", encoding="utf-8", errors="replace") as f:
    master_rows = list(csv.DictReader(f))

samples = [r["Sample"] for r in master_rows]

bio_rows = []
for s in samples:
    vf = os.path.join(VFDB_DIR, f"{s}_vfdb.tab")
    pl = os.path.join(PLASMID_DIR, f"{s}_plasmids.tab")
    sero = os.path.join(SERO_DIR, s, f"{s}_serotype")

    vf_n, vf_preview = parse_tab_summary(vf, "vfdb") if os.path.isfile(vf) else (0, "")
    pl_n, pl_preview = parse_tab_summary(pl, "plasmid") if os.path.isfile(pl) else (0, "")
    sero_preview = parse_serotype_file(sero) if os.path.isfile(sero) else ""

    bio_rows.append({
        "Sample": s,
        "VFDB_File": "Yes" if os.path.isfile(vf) else "No",
        "VFDB_Hits": vf_n,
        "VFDB_Preview": vf_preview,
        "Plasmid_File": "Yes" if os.path.isfile(pl) else "No",
        "Plasmid_Hits": pl_n,
        "Plasmid_Preview": pl_preview,
        "Serotype_File": "Yes" if os.path.isfile(sero) else "No",
        "Serotype_Preview": sero_preview,
    })

# write biology summary
bio_csv = f"{OUTDIR}/BiologyLayer_summary_166.csv"
with open(bio_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(bio_rows[0].keys()))
    w.writeheader()
    for r in bio_rows:
        w.writerow(r)

# merge with master
bio_map = {r["Sample"]: r for r in bio_rows}
merged_rows = []
for r in master_rows:
    rr = dict(r)
    br = bio_map.get(r["Sample"], {})
    rr.update(br)
    merged_rows.append(rr)

merged_csv = f"{OUTDIR}/PrimaryResults_v4_withBiology_166.csv"
with open(merged_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(merged_rows[0].keys()))
    w.writeheader()
    for r in merged_rows:
        w.writerow(r)

# small summary
n_vf = sum(1 for r in bio_rows if r["VFDB_File"] == "Yes")
n_pl = sum(1 for r in bio_rows if r["Plasmid_File"] == "Yes")
n_se = sum(1 for r in bio_rows if r["Serotype_File"] == "Yes")

with open(f"{OUTDIR}/BiologyLayer_summary.txt", "w", encoding="utf-8") as f:
    f.write(f"Samples in master: {len(samples)}\n")
    f.write(f"Samples with VFDB file: {n_vf}\n")
    f.write(f"Samples with plasmid file: {n_pl}\n")
    f.write(f"Samples with serotype file: {n_se}\n")

print("Wrote:", bio_csv)
print("Wrote:", merged_csv)
print("Samples in master:", len(samples))
print("Samples with VFDB file:", n_vf)
print("Samples with plasmid file:", n_pl)
print("Samples with serotype file:", n_se)
