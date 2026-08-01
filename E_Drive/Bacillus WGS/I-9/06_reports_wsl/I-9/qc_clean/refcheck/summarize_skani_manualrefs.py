#!/usr/bin/env python3
from __future__ import annotations
import csv, re
from pathlib import Path

I9 = Path("/mnt/e/Bacillus WGS/I-9/06_reports_wsl/I-9")
OUTDIR = I9 / "qc_clean" / "refcheck"
RAW = OUTDIR / "raw"

OUT_BAC = OUTDIR / "ANI_manualrefs_baconly_clean.tsv"
OUT_ALL = OUTDIR / "ANI_manualrefs_allcontigs_clean.tsv"
HDR_BAC = OUTDIR / "detected_headers_baconly.txt"
HDR_ALL = OUTDIR / "detected_headers_allcontigs.txt"

PAT = re.compile(r"(?P<ref_folder>.+)__(?P<acc>GC[AF]_\d+\.\d+)__(?P<tag>baconly|allcontigs)\.tsv$")

def try_float(x: str) -> str:
    x = (x or "").strip()
    if not x:
        return ""
    try:
        float(x)
        return x
    except Exception:
        return ""

def parse_one(p: Path):
    m = PAT.match(p.name)
    if not m:
        return None

    ref_folder = m.group("ref_folder")
    acc = m.group("acc")
    tag = m.group("tag")

    if p.stat().st_size == 0:
        return (ref_folder, acc, tag, "", "", "", str(p), "EMPTY", "")

    lines = p.read_text(errors="replace").splitlines()
    if len(lines) < 2:
        # skani rule: if AF < 15%, no output is given (usually header only)
        # also happens for low ANI with default settings
        header = lines[0] if lines else ""
        return (ref_folder, acc, tag, "", "", "", str(p), "NO_OUTPUT", header)

    header = [h.strip().lower() for h in lines[0].split("\t")]
    row = lines[1].split("\t")

    # Helper to find a column by substring
    def find_col(wants):
        for i, h in enumerate(header):
            for w in wants:
                if w in h:
                    return i
        return None

    i_ani = find_col(["ani"])
    i_af_ref = find_col(["aligned_fraction_reference", "aligned_fraction_ref", "af_ref", "ref_af"])
    i_af_q = find_col(["aligned_fraction_query", "af_query", "query_af"])

    # Fallback: common skani order: ref, query, ANI, AF_ref, AF_query, ...
    if i_ani is None and len(row) >= 3:
        i_ani = 2
    if i_af_ref is None and len(row) >= 4:
        i_af_ref = 3
    if i_af_q is None and len(row) >= 5:
        i_af_q = 4

    ani = try_float(row[i_ani]) if i_ani is not None and i_ani < len(row) else ""
    af_ref = try_float(row[i_af_ref]) if i_af_ref is not None and i_af_ref < len(row) else ""
    af_q = try_float(row[i_af_q]) if i_af_q is not None and i_af_q < len(row) else ""

    status = "OK" if ani else "MISSING_ANI"
    return (ref_folder, acc, tag, ani, af_ref, af_q, str(p), status, "\t".join(lines[0:1]))

def write(tag: str, out_tsv: Path, out_hdr: Path):
    recs = []
    headers_seen = set()
    for p in sorted(RAW.glob(f"*__{tag}.tsv")):
        rec = parse_one(p)
        if not rec:
            continue
        ref_folder, acc, rtag, ani, af_ref, af_q, raw_file, status, hdr = rec
        if hdr:
            headers_seen.add(hdr)
        recs.append((ref_folder, acc, ani, af_ref, af_q, raw_file, status))

    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with out_tsv.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["ref_folder","accession","ani","af_ref","af_query","raw_file","status"])
        for r in recs:
            w.writerow(list(r))

    with out_hdr.open("w") as f:
        for h in sorted(headers_seen):
            f.write(h + "\n")

    print(f"Wrote: {out_tsv}")
    print(f"Headers: {out_hdr}")

def main():
    if not RAW.exists():
        raise SystemExit(f"RAW folder missing: {RAW}")
    write("baconly", OUT_BAC, HDR_BAC)
    write("allcontigs", OUT_ALL, HDR_ALL)

if __name__ == "__main__":
    main()
