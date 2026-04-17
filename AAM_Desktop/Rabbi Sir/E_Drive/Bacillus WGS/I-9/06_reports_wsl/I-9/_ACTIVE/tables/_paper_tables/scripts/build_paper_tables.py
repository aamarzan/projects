#!/usr/bin/env python3
from __future__ import annotations

import os
import json
import csv
from pathlib import Path
from collections import defaultdict

# -----------------------------
# Paths
# -----------------------------
I9 = Path(os.environ.get("I9", "/mnt/e/Bacillus WGS/bacillus_paranthracis/06_reports_wsl/I-9"))
OUTDIR = Path(os.environ.get("OUTDIR", str(I9 / "figures" / "_paper_mainfigs" / "_paper_tables")))
OUTDIR.mkdir(parents=True, exist_ok=True)

PROJECT_BASE = I9.parent.parent  # .../bacillus_paranthracis

# Common inputs
KRAKEN_READ = I9 / "I-9.reads.kraken2.report.txt"
KRAKEN_CONTIG = I9 / "I-9.contigs.kraken2.report.txt"
KRAKEN_CONTIG_MIN5 = I9 / "I-9.contigs.min5kb.kraken2.report.txt"

REFCHECK = I9 / "qc_clean" / "refcheck"
ANI_BAC = REFCHECK / "ANI_manualrefs_baconly_clean.tsv"
ANI_ALL = REFCHECK / "ANI_manualrefs_allcontigs_clean.tsv"

MLST = I9 / "I-9.mlst.txt"

# Assembly FASTAs (for computed stats)
FA_ALL = I9 / "I-9.final.contigs.fa"
FA_MIN5 = I9 / "I-9.contigs.min5kb.fa"  # may or may not exist
FA_BAC = I9 / "bacillus_only" / "I-9.bacillus_only.fa"

# Fastp json (likely outside wsl folder)
FASTP_JSON_CANDIDATES = [
    PROJECT_BASE / "06_reports" / "I-9.fastp.json",
    PROJECT_BASE / "06_reports" / "I9.fastp.json",
]

# Abricate / AMR clean tables
TABLES = I9 / "tables"
AMR_CLEAN = TABLES / "I-9.amrfinder.clean.tsv"
ABR_CARD = TABLES / "I-9.abricate.card.clean.tsv"
ABR_NCBI = TABLES / "I-9.abricate.ncbi.clean.tsv"
ABR_VFDB = TABLES / "I-9.abricate.vfdb.clean.tsv"
ABR_PLASMID = TABLES / "I-9.abricate.plasmidfinder.clean.tsv"


# -----------------------------
# Helpers
# -----------------------------
def safe_read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None

def find_first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None

def tsv_read_dicts(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", newline="", encoding="utf-8", errors="replace") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            rows.append({k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in row.items()})
    return rows

def to_float(x):
    try:
        if x is None:
            return None
        s = str(x).strip().replace("%", "")
        if s == "":
            return None
        return float(s)
    except Exception:
        return None

def fasta_stats(fa: Path):
    """
    Returns: dict with contigs, total_len, gc_pct, largest, n50, l50
    """
    if not fa.exists():
        return None

    lens = []
    gc = 0
    total = 0

    cur_len = 0
    cur_gc = 0
    in_seq = False

    with fa.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if in_seq:
                    lens.append(cur_len)
                    gc += cur_gc
                    total += cur_len
                in_seq = True
                cur_len = 0
                cur_gc = 0
            else:
                seq = line.upper()
                cur_len += len(seq)
                cur_gc += seq.count("G") + seq.count("C")

        if in_seq:
            lens.append(cur_len)
            gc += cur_gc
            total += cur_len

    if not lens or total == 0:
        return None

    lens_sorted = sorted(lens, reverse=True)
    largest = lens_sorted[0]

    # N50/L50
    half = total * 0.5
    run = 0
    n50 = None
    l50 = None
    for i, L in enumerate(lens_sorted, start=1):
        run += L
        if run >= half:
            n50 = L
            l50 = i
            break

    gc_pct = (gc / total) * 100.0
    return {
        "contigs": len(lens),
        "total_len": total,
        "gc_pct": gc_pct,
        "largest": largest,
        "n50": n50,
        "l50": l50,
    }

def parse_fastp_json(fp: Path):
    """
    Returns combined before/after totals across read1+read2 if present.
    """
    if not fp or not fp.exists():
        return None
    try:
        data = json.loads(fp.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None

    summ = data.get("summary", {})
    bef = summ.get("before_filtering", {})
    aft = summ.get("after_filtering", {})

    def extract(block):
        # fastp sometimes stores totals per read; often already combined
        total_reads = block.get("total_reads", None)
        total_bases = block.get("total_bases", None)
        q30 = block.get("q30_bases", None)
        q20 = block.get("q20_bases", None)

        # If not present, try read1/read2 fields
        if total_reads is None and "read1" in block and "read2" in block:
            r1 = block["read1"]; r2 = block["read2"]
            total_reads = (r1.get("total_reads", 0) or 0) + (r2.get("total_reads", 0) or 0)
            total_bases = (r1.get("total_bases", 0) or 0) + (r2.get("total_bases", 0) or 0)
            q30 = (r1.get("q30_bases", 0) or 0) + (r2.get("q30_bases", 0) or 0)
            q20 = (r1.get("q20_bases", 0) or 0) + (r2.get("q20_bases", 0) or 0)

        if total_bases and q30 is not None:
            q30_pct = (q30 / total_bases) * 100.0
        else:
            q30_pct = None

        return {
            "reads": total_reads,
            "bases": total_bases,
            "q30_pct": q30_pct,
            "q20_bases": q20,
            "q30_bases": q30,
        }

    return {
        "before": extract(bef),
        "after": extract(aft),
    }

def parse_kraken_top_species(report_path: Path, topn=5):
    """
    Kraken2 report format:
    percent clade_reads taxon_reads rank taxid name(with indent)
    We'll return top species rank (S) and also the "unclassified" percent if present.
    """
    if not report_path.exists():
        return {"unclassified_pct": None, "top_species": []}

    top = []
    unclassified = None

    with report_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 6:
                continue
            pct = to_float(parts[0])
            rank = parts[3].strip()
            name = parts[5].strip()

            # unclassified line often has name "unclassified"
            if name.lower() == "unclassified":
                unclassified = pct
                continue

            if rank == "S" and pct is not None:
                top.append((pct, name.strip()))

    top.sort(reverse=True, key=lambda x: x[0])
    return {"unclassified_pct": unclassified, "top_species": top[:topn]}

def best_ref_hit(ani_tsv: Path):
    rows = tsv_read_dicts(ani_tsv)
    if not rows:
        return None, []

    # detect columns
    # common: ANI, AF_query, AF_ref, acc/accession, ref/species
    def get(row, keys):
        for k in keys:
            if k in row and row[k] != "":
                return row[k]
        # also case-insensitive match
        low = {kk.lower(): kk for kk in row.keys()}
        for k in keys:
            kk = low.get(k.lower())
            if kk and row.get(kk, "") != "":
                return row[kk]
        return ""

    enriched = []
    for r in rows:
        ani = to_float(get(r, ["ANI", "ani"]))
        afq = to_float(get(r, ["AF_query", "af_query", "AFq", "afq"]))
        afr = to_float(get(r, ["AF_ref", "af_ref", "AFr", "afr"]))
        acc = get(r, ["accession", "acc", "Accession"])
        ref = get(r, ["reference", "ref", "species", "taxon", "name"])
        enriched.append({
            "ref": ref,
            "acc": acc,
            "ani": ani,
            "af_query": afq,
            "af_ref": afr,
        })

    # choose max ANI, tie by AF_query
    def key(x):
        return (
            x["ani"] if x["ani"] is not None else -1.0,
            x["af_query"] if x["af_query"] is not None else -1.0,
        )

    best = max(enriched, key=key)
    # also sort top 10 for potential use
    top10 = sorted(enriched, key=key, reverse=True)[:10]
    return best, top10

def parse_mlst(path: Path):
    if not path.exists():
        return None
    t = safe_read_text(path)
    if not t:
        return None
    # keep first non-empty line
    for line in t.splitlines():
        if line.strip():
            return line.strip()
    return None

def write_tsv(path: Path, header: list[str], rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})

def read_abricate_clean(path: Path, source_name: str, feature_type: str):
    rows = tsv_read_dicts(path)
    out = []
    for r in rows:
        gene = r.get("GENE") or r.get("Gene") or r.get("gene") or ""
        seq = r.get("SEQUENCE") or r.get("Sequence") or r.get("sequence") or ""
        start = r.get("START") or r.get("Start") or r.get("start") or ""
        end = r.get("END") or r.get("End") or r.get("end") or ""
        ident = r.get("%IDENTITY") or r.get("IDENTITY") or r.get("identity") or ""
        cov = r.get("%COVERAGE") or r.get("COVERAGE") or r.get("coverage") or ""
        product = r.get("PRODUCT") or r.get("Product") or r.get("product") or r.get("RESISTANCE") or ""
        out.append({
            "feature_type": feature_type,
            "gene": gene,
            "class_or_category": "",
            "product": product,
            "support_sources": source_name,
            "best_identity_pct": ident,
            "best_coverage_pct": cov,
            "contig": seq,
            "n_hits": 1,
        })
    return out

def read_amrfinder_clean(path: Path):
    rows = tsv_read_dicts(path)
    out = []
    for r in rows:
        # common AMRFinder headers can vary; handle typical ones
        gene = r.get("Gene symbol") or r.get("Gene") or r.get("gene") or r.get("Element symbol") or ""
        cls = r.get("Class") or r.get("class") or r.get("AMR_class") or ""
        product = r.get("Protein name") or r.get("Product") or r.get("product") or r.get("Element name") or ""
        ident = r.get("% Identity") or r.get("%Identity") or r.get("Identity") or r.get("identity") or ""
        cov = r.get("% Coverage") or r.get("%Coverage") or r.get("Coverage") or r.get("coverage") or ""
        contig = r.get("Contig") or r.get("Sequence name") or r.get("sequence") or r.get("Reference sequence") or ""
        out.append({
            "feature_type": "AMR",
            "gene": gene,
            "class_or_category": cls,
            "product": product,
            "support_sources": "AMRFinder",
            "best_identity_pct": ident,
            "best_coverage_pct": cov,
            "contig": contig,
            "n_hits": 1,
        })
    return out


# -----------------------------
# Build Table 1
# -----------------------------
def build_table_1():
    fastp_json = find_first_existing(FASTP_JSON_CANDIDATES)
    if fastp_json is None:
        # fallback search
        cand = list(PROJECT_BASE.rglob("I-9*.fastp.json"))
        if cand:
            fastp_json = cand[0]

    fp = parse_fastp_json(fastp_json) if fastp_json else None

    stats_all = fasta_stats(FA_ALL)
    stats_min5 = fasta_stats(FA_MIN5) if FA_MIN5.exists() else None
    stats_bac = fasta_stats(FA_BAC)

    rows = []

    def add_row(label, st):
        if st is None:
            return
        rows.append({
            "sample": "I-9",
            "assembly_filter": label,
            "contigs": st["contigs"],
            "total_length_bp": st["total_len"],
            "largest_contig_bp": st["largest"],
            "N50_bp": st["n50"],
            "L50": st["l50"],
            "GC_percent": f"{st['gc_pct']:.2f}",
            "fastp_raw_reads": fp["before"]["reads"] if fp else "NA",
            "fastp_raw_Q30_percent": f"{fp['before']['q30_pct']:.2f}" if (fp and fp["before"]["q30_pct"] is not None) else "NA",
            "fastp_trim_reads": fp["after"]["reads"] if fp else "NA",
            "fastp_trim_Q30_percent": f"{fp['after']['q30_pct']:.2f}" if (fp and fp["after"]["q30_pct"] is not None) else "NA",
            "fastp_json_path": str(fastp_json) if fastp_json else "NA",
            "fasta_path": str(st.get("_path","")) or "",
        })

    if stats_all: stats_all["_path"] = str(FA_ALL)
    if stats_min5: stats_min5["_path"] = str(FA_MIN5)
    if stats_bac: stats_bac["_path"] = str(FA_BAC)

    add_row("all_contigs", stats_all)
    add_row("min5kb_contigs", stats_min5)
    add_row("bacillus_only", stats_bac)

    header = [
        "sample","assembly_filter",
        "contigs","total_length_bp","largest_contig_bp","N50_bp","L50","GC_percent",
        "fastp_raw_reads","fastp_raw_Q30_percent","fastp_trim_reads","fastp_trim_Q30_percent",
        "fastp_json_path","fasta_path"
    ]
    out = OUTDIR / "Table01_QC_Assembly_Summary.tsv"
    write_tsv(out, header, rows)
    return out


# -----------------------------
# Build Table 2
# -----------------------------
def build_table_2():
    kr_r = parse_kraken_top_species(KRAKEN_READ, topn=3)
    kr_c = parse_kraken_top_species(KRAKEN_CONTIG, topn=3)
    kr_m = parse_kraken_top_species(KRAKEN_CONTIG_MIN5, topn=3)

    best_bac, top10_bac = best_ref_hit(ANI_BAC)
    best_all, top10_all = best_ref_hit(ANI_ALL)

    mlst_line = parse_mlst(MLST)

    def fmt_top(top):
        if not top:
            return "NA"
        return "; ".join([f"{name} ({pct:.2f}%)" for pct, name in top])

    row = {
        "sample": "I-9",
        "kraken_reads_top_species": fmt_top(kr_r["top_species"]),
        "kraken_reads_unclassified_pct": kr_r["unclassified_pct"] if kr_r["unclassified_pct"] is not None else "NA",
        "kraken_contigs_top_species": fmt_top(kr_c["top_species"]),
        "kraken_contigs_unclassified_pct": kr_c["unclassified_pct"] if kr_c["unclassified_pct"] is not None else "NA",
        "kraken_contigs_min5kb_top_species": fmt_top(kr_m["top_species"]),
        "kraken_contigs_min5kb_unclassified_pct": kr_m["unclassified_pct"] if kr_m["unclassified_pct"] is not None else "NA",
        "ani_best_baconly_ref": best_bac["ref"] if best_bac else "NA",
        "ani_best_baconly_acc": best_bac["acc"] if best_bac else "NA",
        "ani_best_baconly_ANI": best_bac["ani"] if best_bac else "NA",
        "ani_best_baconly_AF_query": best_bac["af_query"] if best_bac else "NA",
        "ani_best_baconly_AF_ref": best_bac["af_ref"] if best_bac else "NA",
        "ani_best_all_ref": best_all["ref"] if best_all else "NA",
        "ani_best_all_acc": best_all["acc"] if best_all else "NA",
        "ani_best_all_ANI": best_all["ani"] if best_all else "NA",
        "ani_best_all_AF_query": best_all["af_query"] if best_all else "NA",
        "ani_best_all_AF_ref": best_all["af_ref"] if best_all else "NA",
        "mlst_line": mlst_line or "NA",
        "refcheck_baconly_path": str(ANI_BAC) if ANI_BAC.exists() else "NA",
        "refcheck_allcontigs_path": str(ANI_ALL) if ANI_ALL.exists() else "NA",
    }

    header = list(row.keys())
    out = OUTDIR / "Table02_Species_Confirmation.tsv"
    write_tsv(out, header, [row])

    # optional: top10 refcheck detail (deep / supplementary)
    if top10_bac:
        out2 = OUTDIR / "Table02A_Refcheck_Top10_BacOnly.tsv"
        write_tsv(out2, ["ref","acc","ani","af_query","af_ref"], top10_bac)
    if top10_all:
        out3 = OUTDIR / "Table02B_Refcheck_Top10_AllContigs.tsv"
        write_tsv(out3, ["ref","acc","ani","af_query","af_ref"], top10_all)

    return out


# -----------------------------
# Build Table 3
# -----------------------------
def build_table_3():
    records = []

    # AMR
    if AMR_CLEAN.exists():
        records.extend(read_amrfinder_clean(AMR_CLEAN))

    # Abricate-derived
    if ABR_CARD.exists():
        records.extend(read_abricate_clean(ABR_CARD, "Abricate:CARD", "AMR"))
    if ABR_NCBI.exists():
        records.extend(read_abricate_clean(ABR_NCBI, "Abricate:NCBI", "AMR"))
    if ABR_VFDB.exists():
        records.extend(read_abricate_clean(ABR_VFDB, "Abricate:VFDB", "VF"))
    if ABR_PLASMID.exists():
        records.extend(read_abricate_clean(ABR_PLASMID, "Abricate:PlasmidFinder", "PLASMID"))

    if not records:
        # still write an empty table with headers
        out = OUTDIR / "Table03_Gene_Content_Catalog.tsv"
        write_tsv(out, ["feature_type","gene","class_or_category","product","support_sources","best_identity_pct","best_coverage_pct","contig","n_hits"], [])
        return out

    # merge by (feature_type, gene)
    grouped = {}
    for r in records:
        ft = (r.get("feature_type") or "OTHER").strip()
        gene = (r.get("gene") or "").strip()
        key = (ft, gene)
        if key not in grouped:
            grouped[key] = dict(r)
        else:
            # merge support sources
            s = set((grouped[key].get("support_sources","") or "").split("; "))
            s.add(r.get("support_sources",""))
            grouped[key]["support_sources"] = "; ".join([x for x in sorted(s) if x])

            # best identity/coverage (numeric if possible)
            bi_old = to_float(grouped[key].get("best_identity_pct"))
            bi_new = to_float(r.get("best_identity_pct"))
            if bi_new is not None and (bi_old is None or bi_new > bi_old):
                grouped[key]["best_identity_pct"] = r.get("best_identity_pct")

            bc_old = to_float(grouped[key].get("best_coverage_pct"))
            bc_new = to_float(r.get("best_coverage_pct"))
            if bc_new is not None and (bc_old is None or bc_new > bc_old):
                grouped[key]["best_coverage_pct"] = r.get("best_coverage_pct")

            grouped[key]["n_hits"] = int(grouped[key].get("n_hits", 1)) + 1

            # keep contig if consistent; else mark multiple
            c_old = grouped[key].get("contig","")
            c_new = r.get("contig","")
            if c_old and c_new and c_old != c_new:
                grouped[key]["contig"] = "multiple"

    rows = list(grouped.values())
    rows.sort(key=lambda x: (x.get("feature_type",""), x.get("gene","")))

    header = [
        "feature_type","gene","class_or_category","product",
        "support_sources","best_identity_pct","best_coverage_pct",
        "contig","n_hits"
    ]
    out = OUTDIR / "Table03_Gene_Content_Catalog.tsv"
    write_tsv(out, header, rows)
    return out


def main():
    out1 = build_table_1()
    out2 = build_table_2()
    out3 = build_table_3()

    manifest = OUTDIR / "Tables_manifest.tsv"
    rows = [
        {"table":"Table01_QC_Assembly_Summary","path":str(out1)},
        {"table":"Table02_Species_Confirmation","path":str(out2)},
        {"table":"Table03_Gene_Content_Catalog","path":str(out3)},
    ]
    write_tsv(manifest, ["table","path"], rows)

    print("Wrote tables to:", OUTDIR)
    print(" -", out1.name)
    print(" -", out2.name)
    print(" -", out3.name)
    print(" -", manifest.name)

if __name__ == "__main__":
    main()
