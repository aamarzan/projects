import os
import csv
import textwrap
from collections import Counter

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def read_csv(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def as_int(x, default=0):
    try:
        s = str(x).strip()
        if s == "":
            return default
        return int(float(s))
    except Exception:
        return default


def as_float(x, default=0.0):
    try:
        s = str(x).strip()
        if s == "":
            return default
        return float(s)
    except Exception:
        return default


def norm(x):
    return str(x).strip()


def setup_rcparams():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 17,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.linewidth": 0.9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def style_ax(ax, title=None, xlabel=None, ylabel=None, grid_axis="x"):
    if title:
        ax.set_title(title, pad=10)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#444444")
    ax.spines["bottom"].set_color("#444444")

    if grid_axis in {"x", "both"}:
        ax.grid(axis="x", color="#d9dde5", lw=0.8, alpha=0.9, zorder=0)
    if grid_axis in {"y", "both"}:
        ax.grid(axis="y", color="#e8ebf1", lw=0.8, alpha=0.85, zorder=0)


def species_short(name):
    m = {
        "Serratia marcescens": "S. marcescens",
        "Acinetobacter baumannii": "A. baumannii",
        "Klebsiella pneumoniae": "K. pneumoniae",
        "Pseudomonas aeruginosa": "P. aeruginosa",
        "Escherichia coli": "E. coli",
        "Serratia nevei": "S. nevei",
        "Homo sapiens": "H. sapiens",
    }
    return m.get(name, name)


def wrap(s, width=18):
    return "\n".join(textwrap.wrap(str(s), width=width, break_long_words=False))


def cmap_from_hex(hex_list, name):
    return LinearSegmentedColormap.from_list(name, hex_list)


CMAPS = {
    "blue": cmap_from_hex(["#dcecff", "#7ab6ff", "#1f5fbf", "#0b2f6b"], "g3blue"),
    "teal": cmap_from_hex(["#d7fbf5", "#7be3d5", "#14a7a5", "#0d5c63"], "g3teal"),
    "violet": cmap_from_hex(["#f0e5ff", "#c29cff", "#7b4fd6", "#43217f"], "g3violet"),
    "amber": cmap_from_hex(["#fff1cc", "#ffcf6d", "#f39c12", "#8a4f00"], "g3amber"),
    "rose": cmap_from_hex(["#ffe2eb", "#ff99b6", "#dc4a7a", "#7a1638"], "g3rose"),
    "green": cmap_from_hex(["#e4f7df", "#9ad88f", "#38a34f", "#0f5b26"], "g3green"),
}


def palette_list(cmap_name, n, vmin=0.25, vmax=0.95):
    cmap = CMAPS[cmap_name]
    if n <= 1:
        return [cmap(0.75)]
    return [cmap(vmin + (vmax - vmin) * i / (n - 1)) for i in range(n)]


def annot_barh(ax, bars, fmt="{:.0f}", xpad=0.015, fontsize=9):
    xmax = max(ax.get_xlim()[1], 1)
    for b in bars:
        v = b.get_width()
        y = b.get_y() + b.get_height() / 2
        ax.text(v + xmax * xpad, y, fmt.format(v), va="center", ha="left", fontsize=fontsize)


def annot_barv(ax, bars, fmt="{:.0f}", ypad=0.015, fontsize=9):
    ymax = max(ax.get_ylim()[1], 1)
    for b in bars:
        v = b.get_height()
        x = b.get_x() + b.get_width() / 2
        ax.text(x, v + ymax * ypad, fmt.format(v), va="bottom", ha="center", fontsize=fontsize)


def save_png_pdf(fig, stem, outdir):
    ensure_dir(outdir)
    png = os.path.join(outdir, f"{stem}.png")
    pdf = os.path.join(outdir, f"{stem}.pdf")
    fig.savefig(png, dpi=400)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def guess_col(rows, candidates):
    if not rows:
        return None
    cols = list(rows[0].keys())
    lowmap = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lowmap:
            return lowmap[cand.lower()]
    for cand in candidates:
        for c in cols:
            if cand.lower() in c.lower():
                return c
    return None


def heatmap_from_long(ax, rows, row_order, row_col, key_col, val_col,
                      top_n=12, cmap_name="blue", title=None,
                      annotate=False, log1p=True, xtick_width=16):
    counter_by_key = Counter()
    for r in rows:
        key = norm(r.get(key_col, ""))
        if key:
            counter_by_key[key] += as_float(r.get(val_col, 0.0), 0.0)

    keys = [k for k, _ in counter_by_key.most_common(top_n)]
    data = np.zeros((len(row_order), len(keys)), dtype=float)
    rix = {r: i for i, r in enumerate(row_order)}
    kix = {k: i for i, k in enumerate(keys)}

    for r in rows:
        rr = norm(r.get(row_col, ""))
        kk = norm(r.get(key_col, ""))
        vv = as_float(r.get(val_col, 0.0), 0.0)
        if rr in rix and kk in kix:
            data[rix[rr], kix[kk]] += vv

    plot_data = np.log1p(data) if log1p else data
    im = ax.imshow(plot_data, aspect="auto", cmap=CMAPS[cmap_name])

    ax.set_yticks(range(len(row_order)))
    ax.set_yticklabels([species_short(x) for x in row_order])
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([wrap(k, xtick_width) for k in keys], rotation=35, ha="right")

    if title:
        ax.set_title(title, pad=10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if annotate:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if data[i, j] > 0:
                    ax.text(j, i, f"{int(data[i, j])}", ha="center", va="center", fontsize=8, color="white")

    return im, data, keys
