import os, re, shutil

WORK="/mnt/e/DrAhmed/Ongoing/WGS/Result"
SAMPLE_MAN=f"{WORK}/_MANIFESTS/reads_manifest.tsv"
QLIST=f"{WORK}/_INVENTORY/quast_reports.list.txt"
DEST=f"{WORK}/_QUAST_BEST_166/reports"
BEST_TSV=f"{WORK}/_QUAST_BEST_166/quast_best_166.tsv"
MISS_TSV=f"{WORK}/_QUAST_BEST_166/quast_missing_166.tsv"

os.makedirs(DEST, exist_ok=True)
os.makedirs(os.path.dirname(BEST_TSV), exist_ok=True)

samples=[]
with open(SAMPLE_MAN,"r",encoding="utf-8",errors="replace") as f:
    for line in f:
        if not line.strip():
            continue
        s=line.split("\t",1)[0].strip()
        if s and s not in samples:
            samples.append(s)

paths=[]
with open(QLIST,"r",encoding="utf-8",errors="replace") as f:
    for line in f:
        p=line.strip().rstrip("\r")
        if p and os.path.isfile(p):
            paths.append(p)

def match_sample(p, samples):
    for s in samples:
        if f"/{s}_" in p or f"/{s}/" in p or f"{s}_" in os.path.basename(p):
            return s
        if re.search(rf"(^|[^A-Za-z0-9]){re.escape(s)}([^A-Za-z0-9]|$)", p):
            return s
    return None

def quast_score(p):
    base=os.path.basename(p).lower()
    score=0
    if base == "transposed_report.tsv":
        score += 100
    elif base == "report.tsv":
        score += 80
    if "genome_stats" in p.lower():
        score += 50
    if "quast" in p.lower():
        score += 20
    try:
        score += min(os.path.getsize(p) // 100, 1000)
    except:
        pass
    return score

best = {s: (-1, None) for s in samples}

for p in paths:
    s = match_sample(p, samples)
    if not s:
        continue
    sc = quast_score(p)
    if sc > best[s][0]:
        best[s] = (sc, p)

with open(BEST_TSV,"w",encoding="utf-8") as fo:
    fo.write("Sample\tScore\tSourcePath\tDestPath\n")
    for s in samples:
        sc, p = best[s]
        if not p:
            continue
        ext = ".transposed_report.tsv" if os.path.basename(p) == "transposed_report.tsv" else ".report.tsv"
        dest_path = os.path.join(DEST, f"{s}{ext}")
        shutil.copy2(p, dest_path)
        fo.write(f"{s}\t{sc}\t{p}\t{dest_path}\n")

missing=[]
with open(MISS_TSV,"w",encoding="utf-8") as fo:
    fo.write("Sample\n")
    for s in samples:
        if not best[s][1]:
            fo.write(f"{s}\n")
            missing.append(s)

copied = len([fn for fn in os.listdir(DEST) if fn.endswith(".tsv")])
print("Samples expected:", len(samples))
print("QUAST candidate files:", len(paths))
print("Copied reports:", copied)
print("Missing samples:", len(missing))
print("Best table:", BEST_TSV)
print("Missing list:", MISS_TSV)
