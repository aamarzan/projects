import argparse
import gzip
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

def read_ld(ld_gz: str) -> pd.DataFrame:
    # PLINK .ld.gz is tab/space-delimited, with header on first line
    with gzip.open(ld_gz, "rt") as f:
        header = f.readline().strip().split()
    df = pd.read_csv(ld_gz, sep=r"\s+", compression="gzip")
    # Normalize expected columns
    # Common PLINK columns: CHR_A BP_A SNP_A CHR_B BP_B SNP_B R2 [DP] [DPRIME]
    # Some builds use "DPRIME" or "Dprime"
    df.columns = [c.upper() for c in df.columns]
    return df

def read_freq(freq_frq: str) -> pd.DataFrame:
    df = pd.read_csv(freq_frq, sep=r"\s+")
    df.columns = [c.upper() for c in df.columns]
    # Columns typically: CHR SNP A1 A2 MAF NCHROBS
    return df[["SNP", "MAF"]].rename(columns={"SNP": "SNP_ID"})

def save_plots(df_pairs: pd.DataFrame, out_prefix: str):
    out_prefix = Path(out_prefix)

    # 1) Histogram of R2
    plt.figure(figsize=(7.0, 4.5))
    plt.hist(df_pairs["R2"], bins=15)
    plt.xlabel("LD (R²)")
    plt.ylabel("Number of SNP pairs")
    plt.title("Distribution of pairwise LD (R²)")
    plt.tight_layout()
    plt.savefig(out_prefix.with_name(out_prefix.name + "_r2_hist.png"), dpi=600)
    plt.savefig(out_prefix.with_name(out_prefix.name + "_r2_hist.pdf"))
    plt.close()

    # 2) R2 vs distance (bp) with binned median line
    d = df_pairs.copy()
    d["DIST_BP"] = (d["BP_B"] - d["BP_A"]).abs()
    plt.figure(figsize=(7.0, 4.5))
    plt.scatter(d["DIST_BP"], d["R2"], s=18)
    plt.xlabel("Physical distance (bp)")
    plt.ylabel("LD (R²)")
    plt.title("LD decay (R² vs distance)")

    # Bin distances to draw a clean trend line (journal friendly)
    # 10 SNPs only -> keep bins small but stable
    bins = pd.qcut(d["DIST_BP"].rank(method="first"), q=min(6, len(d)), duplicates="drop")
    trend = d.groupby(bins, observed=True).agg(
        dist_median=("DIST_BP", "median"),
        r2_median=("R2", "median"),
    ).sort_values("dist_median")
    plt.plot(trend["dist_median"], trend["r2_median"], linewidth=2.0)

    plt.tight_layout()
    plt.savefig(out_prefix.with_name(out_prefix.name + "_r2_vs_distance.png"), dpi=600)
    plt.savefig(out_prefix.with_name(out_prefix.name + "_r2_vs_distance.pdf"))
    plt.close()

    # 3) Per-SNP "hubness": mean and max R2 with other SNPs
    # Build symmetric per-SNP summary from pairwise list
    a = d[["SNP_A", "R2"]].rename(columns={"SNP_A": "SNP"})
    b = d[["SNP_B", "R2"]].rename(columns={"SNP_B": "SNP"})
    per = pd.concat([a, b], ignore_index=True)
    per_sum = per.groupby("SNP").agg(mean_r2=("R2", "mean"), max_r2=("R2", "max")).sort_values("mean_r2", ascending=False)

    plt.figure(figsize=(8.2, 4.8))
    plt.bar(per_sum.index, per_sum["mean_r2"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Mean R² with other SNPs")
    plt.title("Per-SNP LD connectivity (mean R²)")
    plt.tight_layout()
    plt.savefig(out_prefix.with_name(out_prefix.name + "_mean_r2_per_snp.png"), dpi=600)
    plt.savefig(out_prefix.with_name(out_prefix.name + "_mean_r2_per_snp.pdf"))
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ld", required=True, help="PLINK .ld.gz file (e.g., ahmed10_ld.ld.gz)")
    ap.add_argument("--freq", required=True, help="PLINK .frq file (e.g., ahmed10_freq.frq)")
    ap.add_argument("--out", required=True, help="Output prefix (e.g., ahmed10_report)")
    args = ap.parse_args()

    df = read_ld(args.ld)

    # Require key columns
    need = {"BP_A", "BP_B", "SNP_A", "SNP_B", "R2"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"LD file missing columns: {sorted(missing)}. Columns found: {list(df.columns)}")

    df["DIST_BP"] = (df["BP_B"] - df["BP_A"]).abs()
    df["DIST_KB"] = df["DIST_BP"] / 1000.0

    # Attach MAFs
    frq = read_freq(args.freq)
    frq_map = dict(zip(frq["SNP_ID"], frq["MAF"]))

    df["MAF_A"] = df["SNP_A"].map(frq_map)
    df["MAF_B"] = df["SNP_B"].map(frq_map)

    # Find a D' column if present
    dprime_col = None
    for cand in ["DPRIME", "DPRIME", "DPRIME", "DPRIME", "DPRIME", "DPRIME", "DPRIME", "DPRIME"]:
        pass
    # Actual common names:
    for cand in ["DPRIME", "DPRIME", "DPRIME", "DPRIME", "DPRIME"]:
        pass
    for cand in ["DPRIME", "DPRIME", "DPRIME"]:
        pass
    # Real check:
    for cand in ["DPRIME", "DPRIME", "DPRIME", "DPRIME", "DP", "DPRIME", "DPRIME", "DPRIME"]:
        if cand in df.columns:
            dprime_col = cand
            break
    # Some builds use "DPRIME" or "DPRIME" already uppercased; "DP" can exist too
    if dprime_col is None:
        for cand in ["DPRIME", "DPRIME", "DPRIME", "DPRIME", "DPRIME", "DPRIME", "DPRIME", "DPRIME"]:
            if cand in df.columns:
                dprime_col = cand
                break
    # If none, that's fine; table will still be valid with R2.

    # LD strength category for easy manuscript description
    def cat(r2):
        if r2 >= 0.8: return "Very high (≥0.80)"
        if r2 >= 0.6: return "High (0.60–0.79)"
        if r2 >= 0.4: return "Moderate (0.40–0.59)"
        if r2 >= 0.2: return "Low (0.20–0.39)"
        return "Very low (<0.20)"
    df["LD_CLASS"] = df["R2"].apply(cat)

    # Pairwise table
    cols = ["SNP_A", "BP_A", "MAF_A", "SNP_B", "BP_B", "MAF_B", "DIST_BP", "DIST_KB", "R2"]
    if dprime_col:
        cols.append(dprime_col)
    cols.append("LD_CLASS")

    pairs = df[cols].sort_values(["R2", "DIST_BP"], ascending=[False, True]).reset_index(drop=True)

    # Per-SNP summary
    per = pd.concat([
        df[["SNP_A", "R2"]].rename(columns={"SNP_A": "SNP"}),
        df[["SNP_B", "R2"]].rename(columns={"SNP_B": "SNP"}),
    ], ignore_index=True)

    snp_summary = per.groupby("SNP").agg(
        mean_r2=("R2", "mean"),
        max_r2=("R2", "max"),
        n_pairs=("R2", "size"),
        n_r2_ge_0p8=("R2", lambda x: (x >= 0.8).sum()),
        n_r2_ge_0p5=("R2", lambda x: (x >= 0.5).sum()),
    ).sort_values("mean_r2", ascending=False).reset_index()

    out = Path(args.out)

    # Write TSVs
    pairs.to_csv(out.with_suffix("") .as_posix() + "_LD_pairs.tsv", sep="\t", index=False)
    snp_summary.to_csv(out.with_suffix("") .as_posix() + "_LD_snp_summary.tsv", sep="\t", index=False)

    # Write Excel (nice for sharing)
    xlsx_path = out.with_suffix("") .as_posix() + "_LD_report.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as w:
        pairs.to_excel(w, sheet_name="Pairwise_LD", index=False)
        snp_summary.to_excel(w, sheet_name="Per_SNP_Summary", index=False)

    # Plots
    save_plots(df_pairs=df, out_prefix=out.with_suffix(""))

    print("Wrote:")
    print(" -", out.with_suffix("").as_posix() + "_LD_pairs.tsv")
    print(" -", out.with_suffix("").as_posix() + "_LD_snp_summary.tsv")
    print(" -", xlsx_path)
    print(" -", out.with_suffix("").as_posix() + "_r2_hist.(png/pdf)")
    print(" -", out.with_suffix("").as_posix() + "_r2_vs_distance.(png/pdf)")
    print(" -", out.with_suffix("").as_posix() + "_mean_r2_per_snp.(png/pdf)")

if __name__ == "__main__":
    main()
