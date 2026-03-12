#!/usr/bin/env python3
from __future__ import annotations

import os, csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.colors import LinearSegmentedColormap

HAVE_SCIPY = True
try:
    from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
    from scipy.spatial.distance import squareform
except Exception:
    HAVE_SCIPY = False

def _cm(colors, name):
    return LinearSegmentedColormap.from_list(name, colors, N=256)

CM_HEAT = _cm(["#071024", "#0d2a4d", "#1f8aa5", "#f6f7fb"], "heat_premium")
CM_DIST = _cm(["#071024", "#2b1b59", "#ff9a6a", "#f6f7fb"], "dist_premium")

TYPE_COLORS = {
    "AMR_gene":   "#2b6cff",
    "AMR_class":  "#1aa6b7",
    "CARD_gene":  "#6a4cff",
    "NCBI_gene":  "#3c7dd9",
    "VF_gene":    "#1db954",
    "PLASMID_rep":"#ff7a59",
    "OTHER":      "#94a3b8",
}

def panel_tag(ax, letter: str):
    ax.text(
        0.01, 0.98, letter,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=11, fontweight="bold", color="white",
        bbox=dict(boxstyle="round,pad=0.25", fc="#0f172a", ec="none", alpha=0.95),
        zorder=50,
    )

def to_float(x):
    try:
        return float(str(x).strip().replace("%",""))
    except Exception:
        return None

def shorten(s: str, maxlen=18):
    s = s.replace("_"," ").strip()
    if len(s) <= maxlen:
        return s
    return s[:maxlen-1] + "…"

def sniff_delim(path: Path) -> str:
    # robust: try csv.Sniffer, else fall back
    txt = path.read_text(errors="ignore").splitlines()
    txt = [ln for ln in txt if ln.strip() and not ln.startswith("#")]
    if not txt:
        return "\t"
    sample = "\n".join(txt[:20])
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters="\t,;")
        return dialect.delimiter
    except Exception:
        if "\t" in txt[0]:
            return "\t"
        if "," in txt[0]:
            return ","
        return "\t"

def read_table(path: Path):
    if not path.exists():
        return [], []
    delim = sniff_delim(path)
    lines = path.read_text(errors="ignore").splitlines()
    lines = [ln for ln in lines if ln.strip() and not ln.startswith("#")]
    if not lines:
        return [], []
    header = [h.strip() for h in lines[0].split(delim)]
    rows = []
    for ln in lines[1:]:
        parts = [p.strip() for p in ln.split(delim)]
        if len(parts) != len(header):
            continue
        rows.append(dict(zip(header, parts)))
    return header, rows

def find_col(header: list[str], candidates: list[str]) -> str|None:
    hlow = {h.lower(): h for h in header}
    for c in candidates:
        c = c.lower()
        if c in hlow:
            return hlow[c]
    # fuzzy contains
    for c in candidates:
        c = c.lower()
        for h in header:
            if c in h.lower():
                return h
    return None

def feat_type(feat: str) -> str:
    if feat.startswith("AMR_gene:"): return "AMR_gene"
    if feat.startswith("AMR_class:"): return "AMR_class"
    if feat.startswith("CARD_gene:"): return "CARD_gene"
    if feat.startswith("NCBI_gene:"): return "NCBI_gene"
    if feat.startswith("VF_gene:"): return "VF_gene"
    if feat.startswith("PLASMID_rep:"): return "PLASMID_rep"
    return "OTHER"

def collect_inputs(i9: Path):
    tables = i9 / "tables"
    amrdir = i9 / "amr"
    return {
        "amrfinder": (tables / "I-9.amrfinder.clean.tsv") if (tables / "I-9.amrfinder.clean.tsv").exists() else (amrdir / "I-9.amrfinder.tsv"),
        "card":     (tables / "I-9.abricate.card.clean.tsv") if (tables / "I-9.abricate.card.clean.tsv").exists() else (amrdir / "I-9.abricate.card.tab"),
        "ncbi":     (tables / "I-9.abricate.ncbi.clean.tsv") if (tables / "I-9.abricate.ncbi.clean.tsv").exists() else (amrdir / "I-9.abricate.ncbi.tab"),
        "vfdb":     (tables / "I-9.abricate.vfdb.clean.tsv") if (tables / "I-9.abricate.vfdb.clean.tsv").exists() else (amrdir / "I-9.abricate.vfdb.tab"),
        "plasmid":  (tables / "I-9.abricate.plasmidfinder.clean.tsv") if (tables / "I-9.abricate.plasmidfinder.clean.tsv").exists() else (amrdir / "I-9.abricate.plasmidfinder.tab"),
    }

def build_matrix(i9: Path):
    paths = collect_inputs(i9)
    contig_feats = defaultdict(lambda: defaultdict(float))

    # AMRFinder
    if paths["amrfinder"].exists():
        hdr, rows = read_table(paths["amrfinder"])
        c_contig = find_col(hdr, ["contig","sequence","seqid","SEQUENCE","Contig"])
        c_gene   = find_col(hdr, ["gene","Gene","GENE","symbol","Gene symbol","GENE SYMBOL"])
        c_class  = find_col(hdr, ["class","drug_class","drug class","subclass","CLASS"])
        for r in rows:
            contig = r.get(c_contig, "") if c_contig else ""
            gene   = r.get(c_gene, "")   if c_gene else ""
            cls    = r.get(c_class, "")  if c_class else ""
            if contig:
                if gene:
                    contig_feats[contig][f"AMR_gene:{gene}"] = 1.0
                if cls:
                    contig_feats[contig][f"AMR_class:{cls}"] = 1.0

    # Abricate-like tables
    def load_abricate(path: Path, prefix: str):
        if not path.exists():
            return
        hdr, rows = read_table(path)
        c_contig = find_col(hdr, ["sequence","SEQUENCE","contig","seqid"])
        c_gene   = find_col(hdr, ["gene","GENE","Gene","best_hit","hit"])
        c_ident  = find_col(hdr, ["%identity","identity","pident","pct_identity","IDENTITY"])
        c_cov    = find_col(hdr, ["%coverage","coverage","pcov","pct_coverage","COVERAGE"])
        for r in rows:
            contig = r.get(c_contig, "") if c_contig else ""
            gene   = r.get(c_gene, "")   if c_gene else ""
            if not contig or not gene:
                continue
            ident = to_float(r.get(c_ident, "")) if c_ident else None
            cov   = to_float(r.get(c_cov, ""))   if c_cov else None
            w = 1.0
            if ident is not None: w *= max(0.0, min(1.0, ident/100.0))
            if cov is not None:   w *= max(0.0, min(1.0, cov/100.0))
            w = max(w, 0.25)  # keep low hits visible
            key = f"{prefix}:{gene}"
            contig_feats[contig][key] = max(contig_feats[contig][key], w)

    load_abricate(paths["card"], "CARD_gene")
    load_abricate(paths["ncbi"], "NCBI_gene")
    load_abricate(paths["vfdb"], "VF_gene")
    load_abricate(paths["plasmid"], "PLASMID_rep")

    contigs = sorted(contig_feats.keys())
    feats = sorted({f for c in contigs for f in contig_feats[c].keys()})

    if not contigs or not feats:
        return [], [], np.zeros((0,0)), paths

    X = np.zeros((len(contigs), len(feats)), float)
    for i,c in enumerate(contigs):
        for j,f in enumerate(feats):
            if f in contig_feats[c]:
                X[i,j] = contig_feats[c][f]

    # keep contigs that actually have signal
    keep = np.where(X.sum(axis=1) > 0)[0]
    contigs = [contigs[i] for i in keep]
    X = X[keep,:]

    return contigs, feats, X, paths

def zscore_cols(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)
    sd[sd==0] = 1.0
    return (X - mu) / sd

def corr_dist(Xz):
    n = Xz.shape[0]
    D = np.zeros((n,n), float)
    for i in range(n):
        for j in range(i+1, n):
            a = Xz[i]; b = Xz[j]
            if np.allclose(a, a[0]) or np.allclose(b, b[0]):
                d = 1.0
            else:
                c = np.corrcoef(a,b)[0,1]
                d = 1.0 if np.isnan(c) else float(1.0 - c)
            D[i,j]=D[j,i]=np.clip(d,0.0,2.0)
    return D

def choose_features(X, feats, k=35):
    prev = (X>0).mean(axis=0)
    var  = X.var(axis=0)
    score = prev*(var+1e-9)
    idx = np.argsort(score)[::-1]
    idx = [int(i) for i in idx if prev[i] > 0][:k]
    return idx

def save_all(fig, outbase: Path, dpi=600):
    out_pdf = outbase.with_suffix(".pdf")
    out_svg = outbase.with_suffix(".svg")
    out_eps = outbase.with_suffix(".eps")
    out_png = outbase.with_suffix(".png")
    out_jpg = outbase.with_suffix(".jpg")

    fig.savefig(out_pdf)
    fig.savefig(out_svg)
    fig.savefig(out_eps)
    fig.savefig(out_png, dpi=dpi)

    try:
        from PIL import Image
        im = Image.open(out_png).convert("RGB")
        im.save(out_jpg, quality=95, subsampling=0, optimize=True)
    except Exception:
        fig.savefig(out_jpg, dpi=300)

    return out_pdf, out_svg, out_png, out_jpg, out_eps

def write_tsvs(outdir: Path, prefix: str, contigs, feats, order, clusters, D, disc):
    outdir.mkdir(parents=True, exist_ok=True)

    p1 = outdir / f"{prefix}.clusters.tsv"
    with p1.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["contig","cluster"])
        for i in order:
            w.writerow([contigs[i], int(clusters[i])])

    p2 = outdir / f"{prefix}.discriminators.tsv"
    with p2.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["feature","type","abs_mean_z_gap"])
        for feat,gap in disc:
            w.writerow([feat, feat_type(feat), f"{gap:.4f}"])

    p3 = outdir / f"{prefix}.distance_matrix.tsv"
    with p3.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow([""] + [contigs[i] for i in order])
        for ii in order:
            w.writerow([contigs[ii]] + [f"{D[ii,jj]:.6f}" for jj in order])

def main():
    I9 = Path(os.environ.get("I9", "/mnt/e/Bacillus WGS/bacillus_paranthracis/06_reports_wsl/I-9"))
    OUTDIR = Path(os.environ.get("OUTDIR", str(I9 / "figures" / "_paper_mainfigs" / "fig_4")))
    OUTDIR.mkdir(parents=True, exist_ok=True)

    CONF_TOL = float(os.environ.get("CONF_TOL", "0.12"))
    TOP_FEATS = int(os.environ.get("TOP_FEATS", "35"))
    TOP_DISC  = int(os.environ.get("TOP_DISC", "12"))

    contigs, feats, X, paths = build_matrix(I9)
    if len(contigs) == 0 or len(feats) == 0 or X.size == 0:
        print("No features found in AMR/Abricate tables; cannot build Fig4.")
        # show what files it tried (very useful)
        for k,v in paths.items():
            print(f"  {k}: {v}  exists={v.exists()}")
        return

    # select features
    idx = choose_features(X, feats, k=TOP_FEATS)
    feats_sel = [feats[i] for i in idx]
    X_sel = X[:, idx]
    Xz = zscore_cols(X_sel)

    # cluster / order
    D = corr_dist(Xz)
    if HAVE_SCIPY and Xz.shape[0] >= 2:
        Z = linkage(squareform(D, checks=False), method="average")
        order = dendrogram(Z, no_plot=True)["leaves"]
        clusters = fcluster(Z, t=2, criterion="maxclust")
    else:
        order = list(range(Xz.shape[0]))
        clusters = np.ones((Xz.shape[0],), dtype=int)

    # discriminators between top 2 clusters
    cl_ids, cl_counts = np.unique(clusters, return_counts=True)
    cl_sorted = [int(x) for x in cl_ids[np.argsort(cl_counts)[::-1]]]
    c1 = cl_sorted[0]
    c2 = cl_sorted[1] if len(cl_sorted) > 1 else cl_sorted[0]

    disc = []
    for j, feat in enumerate(feats_sel):
        a = Xz[clusters == c1, j]
        b = Xz[clusters == c2, j]
        if len(a)==0 or len(b)==0:
            continue
        gap = abs(float(np.mean(a) - np.mean(b)))
        disc.append((feat, gap))
    disc.sort(key=lambda x: x[1], reverse=True)
    disc_top = disc[:TOP_DISC]

    # confusability
    tri = []
    n = D.shape[0]
    for i in range(n):
        for j in range(i+1, n):
            tri.append((D[i,j], i, j))
    tri.sort(key=lambda x: x[0])
    near = [(d,i,j) for (d,i,j) in tri if d <= CONF_TOL][:8]

    # ---------------- layout (no blue box) ----------------
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 200
    })

    fig = plt.figure(figsize=(16.5, 11.5))
    gs = GridSpec(3, 3, figure=fig,
                  height_ratios=[1.25, 0.85, 0.85],
                  width_ratios=[0.55, 1.75, 1.20],
                  hspace=0.55, wspace=0.35)

    ax_den = fig.add_subplot(gs[0,0])
    subA = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0,1], height_ratios=[0.10,0.90], hspace=0.02)
    ax_strip = fig.add_subplot(subA[0,0])
    ax_heat  = fig.add_subplot(subA[1,0])
    ax_dist  = fig.add_subplot(gs[0,2])

    # B dendrogram: NO leaf labels (prevents overlap)
    if HAVE_SCIPY and n >= 2:
        dendrogram(linkage(squareform(D, checks=False), method="average"),
                   orientation="left", no_labels=True, ax=ax_den)
    ax_den.invert_yaxis()
    ax_den.set_title("B. Hierarchical clustering (contigs)", pad=10)
    ax_den.set_xticks([]); ax_den.set_yticks([])
    for sp in ax_den.spines.values(): sp.set_visible(False)
    panel_tag(ax_den, "B")

    # A heatmap (ordered)
    Xo = Xz[order,:]
    contigs_o = [contigs[i] for i in order]
    xlabs = [shorten(f.split(":",1)[-1], 16) for f in feats_sel]

    im = ax_heat.imshow(Xo, aspect="auto", cmap=CM_HEAT, vmin=-2.5, vmax=2.5, interpolation="nearest")
    ax_heat.set_title("A. Contig feature heatmap (z-scored; clustered order)", pad=10)
    panel_tag(ax_heat, "A")

    ax_heat.set_xticks(np.arange(len(xlabs)))
    ax_heat.set_xticklabels(xlabs, rotation=45, ha="right", rotation_mode="anchor", fontsize=8)
    ax_heat.tick_params(axis="x", pad=2)

    ax_heat.set_yticks(np.arange(len(contigs_o)))
    if len(contigs_o) <= 30:
        ax_heat.set_yticklabels(contigs_o, fontsize=8)
    else:
        ax_heat.set_yticklabels([c if (i%2==0) else "" for i,c in enumerate(contigs_o)], fontsize=8)

    # feature type strip
    ax_strip.set_xlim(-0.5, len(feats_sel)-0.5)
    ax_strip.set_ylim(0,1)
    for j, feat in enumerate(feats_sel):
        t = feat_type(feat)
        ax_strip.add_patch(plt.Rectangle((j-0.5, 0), 1.0, 1.0, color=TYPE_COLORS.get(t,"#94a3b8"), lw=0))
    ax_strip.set_xticks([]); ax_strip.set_yticks([])
    for sp in ax_strip.spines.values(): sp.set_visible(False)

    cax = fig.add_axes([ax_heat.get_position().x1 + 0.004, ax_heat.get_position().y0,
                        0.010, ax_heat.get_position().height])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Standardized signal (z)", fontsize=9)

    # C distance matrix (ordered)
    Do = D[np.ix_(order, order)]
    imd = ax_dist.imshow(Do, cmap=CM_DIST, vmin=0.0, vmax=min(1.0, float(np.max(Do))))
    ax_dist.set_title("C. Pairwise distance matrix (correlation distance)", pad=10)
    panel_tag(ax_dist, "C")
    ax_dist.set_xticks([]); ax_dist.set_yticks([])
    cbar = fig.colorbar(imd, ax=ax_dist, fraction=0.046, pad=0.04)
    cbar.set_label("Distance", fontsize=9)

    # D confusability
    ax_conf = fig.add_subplot(gs[1,:])
    panel_tag(ax_conf, "D")
    ax_conf.set_title("D. Confusability (near-matches under tolerance)", pad=10)
    dvals = [d for d,_,_ in tri]
    ax_conf.hist(dvals, bins=24, alpha=0.85)
    ax_conf.axvline(CONF_TOL, color="#ef4444", lw=2.0, label=f"conf_tol={CONF_TOL:.2f}")
    ax_conf.set_xlabel("Pairwise distance"); ax_conf.set_ylabel("Pair count")
    ax_conf.grid(axis="y", alpha=0.2)
    ax_conf.legend(loc="upper right", frameon=True, fontsize=9)

    if near:
        lines = [f"Near pairs (d ≤ {CONF_TOL:.2f}) [top {len(near)}]:"]
        for d,i,j in near:
            lines.append(f"  {contigs[i]} ↔ {contigs[j]}   d={d:.3f}")
        txt = "\n".join(lines)
    else:
        txt = f"No near pairs found at d ≤ {CONF_TOL:.2f}."
    ax_conf.text(0.01,0.98,txt, transform=ax_conf.transAxes, ha="left", va="top",
                 fontsize=9, bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#e5e7eb", alpha=0.95))

    # E discriminators (lollipop)
    ax_disc = fig.add_subplot(gs[2,:])
    panel_tag(ax_disc, "E")
    ax_disc.set_title("E. Top discriminators (features separating the top 2 clusters)", pad=10)

    if disc_top:
        feats_d = [f for f,_ in disc_top][::-1]
        gaps_d  = [g for _,g in disc_top][::-1]
        y = np.arange(len(feats_d))
        for yi, feat, gap in zip(y, feats_d, gaps_d):
            t = feat_type(feat)
            col = TYPE_COLORS.get(t,"#2b6cff")
            ax_disc.hlines(yi, 0, gap, color=col, alpha=0.35, lw=4)
            ax_disc.scatter([gap],[yi], s=80, color=col, edgecolors="#0f172a", linewidths=0.6, zorder=5)
        ax_disc.set_yticks(y)
        ax_disc.set_yticklabels([shorten(f.split(":",1)[-1], 28) for f in feats_d], fontsize=9)
        ax_disc.set_xlabel("Separation margin (|Δ mean z| between top 2 clusters)")
        ax_disc.grid(axis="x", alpha=0.2)
    else:
        ax_disc.text(0.5,0.5,"No discriminators (insufficient separation).",
                     transform=ax_disc.transAxes, ha="center", va="center", fontsize=11)

    fig.suptitle(
        "Main Figure 4 — WGS structure & discriminability (A–E)\n"
        "Contig-level gene-content vectors (AMRFinder + Abricate) → clustering, distances, confusability, top discriminators",
        fontsize=16, fontweight="bold", y=0.98
    )

    outbase = OUTDIR / "MainFig04_WGS_Discriminability"
    save_all(fig, outbase, dpi=600)
    plt.close(fig)

    write_tsvs(OUTDIR, "MainFig04_WGS_Discriminability", contigs, feats_sel, order, clusters, D, disc_top)

    print(f"Wrote: {outbase.with_suffix('.pdf')}")
    print(f"Wrote: {outbase.with_suffix('.png')}")
    print(f"Wrote: {OUTDIR / 'MainFig04_WGS_Discriminability.clusters.tsv'}")
    print(f"Wrote: {OUTDIR / 'MainFig04_WGS_Discriminability.discriminators.tsv'}")
    print(f"Wrote: {OUTDIR / 'MainFig04_WGS_Discriminability.distance_matrix.tsv'}")

if __name__ == "__main__":
    main()
