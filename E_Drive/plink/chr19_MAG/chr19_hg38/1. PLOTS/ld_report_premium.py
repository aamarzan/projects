import gzip
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

LD_GZ   = "MAG_ld.ld.gz"
FRQ_TXT = "MAG_freq.frq"
PREFIX  = "MAG_report"
EDGE_R2 = 0.8   # network threshold

def read_ld(path):
    # PLINK .ld.gz is whitespace-delimited
    df = pd.read_csv(path, sep=r"\s+", compression="gzip")
    # standardize expected columns
    # common: CHR_A BP_A SNP_A CHR_B BP_B SNP_B R2 (and maybe DP, MAF_A/B, ...)
    for col in ["BP_A","BP_B","SNP_A","SNP_B","R2"]:
        if col not in df.columns:
            raise SystemExit(f"Missing expected column {col} in {path}. Columns={list(df.columns)}")
    df["DIST_BP"] = (df["BP_B"] - df["BP_A"]).abs()
    return df

def read_frq(path):
    # PLINK --freq output columns: CHR SNP A1 A2 MAF NCHROBS
    if not (path and pd.io.common.file_exists(path)):
        return None
    frq = pd.read_csv(path, sep=r"\s+")
    if "SNP" in frq.columns:
        frq = frq.rename(columns={"SNP":"ID"})
    return frq

def save_tables(ld, frq):
    ld_out = ld.copy()
    # keep a clean column order
    cols = [c for c in ["SNP_A","BP_A","SNP_B","BP_B","DIST_BP","R2"] if c in ld_out.columns] + \
           [c for c in ld_out.columns if c not in ["SNP_A","BP_A","SNP_B","BP_B","DIST_BP","R2"]]
    ld_out = ld_out[cols]
    ld_out.to_csv(f"{PREFIX}_LD_pairs.tsv", sep="\t", index=False)

    # Excel with simple formatting
    try:
        with pd.ExcelWriter(f"{PREFIX}_LD_pairs.xlsx", engine="openpyxl") as xw:
            ld_out.to_excel(xw, sheet_name="LD_pairs", index=False)
            if frq is not None:
                frq.to_excel(xw, sheet_name="Allele_freq", index=False)
    except Exception as e:
        print("[WARN] Excel export failed:", e)

    # summary text
    n_pairs = len(ld)
    r2_max = ld["R2"].max()
    r2_mean = ld["R2"].mean()
    with open(f"{PREFIX}_summary.txt","w") as f:
        f.write(f"Pairs: {n_pairs}\n")
        f.write(f"R2 mean: {r2_mean:.4f}\n")
        f.write(f"R2 max : {r2_max:.4f}\n")
        if "DIST_BP" in ld.columns:
            f.write(f"Distance bp min/median/max: {ld['DIST_BP'].min()} / {int(ld['DIST_BP'].median())} / {ld['DIST_BP'].max()}\n")

def plot_r2_vs_distance(ld):
    plt.figure(figsize=(6.5,4.2))
    plt.scatter(ld["DIST_BP"], ld["R2"], s=18)
    plt.xlabel("Distance (bp)")
    plt.ylabel(r"LD ($r^2$)")
    plt.title("LD decay (pairwise)")
    plt.tight_layout()
    for ext in ["png","pdf","svg"]:
        plt.savefig(f"{PREFIX}_r2_vs_distance.{ext}", dpi=300)
    plt.close()

def plot_network(ld):
    # nodes on x-axis by position; edges for high LD
    nodes = pd.unique(ld[["SNP_A","SNP_B"]].values.ravel("K"))
    # best guess positions from BP_A/B
    pos_map = {}
    for _,r in ld.iterrows():
        pos_map.setdefault(r["SNP_A"], r["BP_A"])
        pos_map.setdefault(r["SNP_B"], r["BP_B"])
    nodes = sorted(nodes, key=lambda x: pos_map.get(x, 0))

    x = np.array([pos_map.get(n, i) for i,n in enumerate(nodes)], dtype=float)
    y = np.zeros_like(x)

    plt.figure(figsize=(7.2,2.6))
    plt.scatter(x, y, s=40)
    for xi, n in zip(x, nodes):
        plt.text(xi, 0.02, n, rotation=45, ha="left", va="bottom", fontsize=8)

    # arcs
    for _,r in ld.iterrows():
        if r["R2"] >= EDGE_R2:
            a = nodes.index(r["SNP_A"])
            b = nodes.index(r["SNP_B"])
            xa, xb = x[a], x[b]
            xm = (xa+xb)/2
            h  = abs(xb-xa)/2
            t = np.linspace(0, np.pi, 80)
            plt.plot(xm + (abs(xb-xa)/2)*np.cos(t), 0.10 + (h/ (x.max()-x.min()+1e-9))*np.sin(t), linewidth=1)

    plt.yticks([])
    plt.xlabel("Genomic position (bp)")
    plt.title(f"High-LD network (edges r² ≥ {EDGE_R2})")
    plt.tight_layout()
    for ext in ["png","pdf","svg"]:
        plt.savefig(f"{PREFIX}_LD_network.{ext}", dpi=300)
    plt.close()

def plot_maf(frq):
    if frq is None or "MAF" not in frq.columns:
        return
    frq = frq.sort_values("MAF", ascending=False)
    plt.figure(figsize=(6.5,4.0))
    plt.bar(frq["ID"], frq["MAF"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("MAF")
    plt.title("Minor allele frequency")
    plt.tight_layout()
    for ext in ["png","pdf","svg"]:
        plt.savefig(f"{PREFIX}_MAF.{ext}", dpi=300)
    plt.close()

def plot_matrix(ld):
    # build r2 matrix for heatmap-like plot
    snps = sorted(pd.unique(ld[["SNP_A","SNP_B"]].values.ravel("K")))
    idx = {s:i for i,s in enumerate(snps)}
    mat = np.ones((len(snps),len(snps))) * np.nan
    np.fill_diagonal(mat, 1.0)
    for _,r in ld.iterrows():
        i,j = idx[r["SNP_A"]], idx[r["SNP_B"]]
        mat[i,j] = r["R2"]
        mat[j,i] = r["R2"]

    plt.figure(figsize=(5.8,5.2))
    im = plt.imshow(mat, vmin=0, vmax=1)
    plt.xticks(range(len(snps)), snps, rotation=90, fontsize=8)
    plt.yticks(range(len(snps)), snps, fontsize=8)
    plt.title(r"LD matrix ($r^2$)")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    for ext in ["png","pdf","svg"]:
        plt.savefig(f"{PREFIX}_LD_matrix.{ext}", dpi=300)
    plt.close()

if __name__ == "__main__":
    ld = read_ld(LD_GZ)
    frq = read_frq(FRQ_TXT)
    save_tables(ld, frq)
    plot_r2_vs_distance(ld)
    plot_network(ld)
    plot_maf(frq)
    plot_matrix(ld)
    print("[OK] Wrote tables + plots with prefix:", PREFIX)