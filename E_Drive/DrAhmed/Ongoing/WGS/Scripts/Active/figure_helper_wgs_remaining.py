import os
import csv
import textwrap
from collections import Counter, defaultdict

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def read_csv(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def find_first_existing(paths):
    for p in paths:
        if os.path.isfile(p):
            return p
    return None


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


def wrap(s, width=18):
    return "\n".join(textwrap.wrap(str(s), width=width, break_long_words=False))


def cmap_from_hex(hex_list, name):
    return LinearSegmentedColormap.from_list(name, hex_list)


CMAPS = {
    "blue": cmap_from_hex(["#dcecff", "#7ab6ff", "#1f5fbf", "#0b2f6b"], "g4blue"),
    "teal": cmap_from_hex(["#d7fbf5", "#7be3d5", "#14a7a5", "#0d5c63"], "g4teal"),
    "violet": cmap_from_hex(["#f0e5ff", "#c29cff", "#7b4fd6", "#43217f"], "g4violet"),
    "amber": cmap_from_hex(["#fff1cc", "#ffcf6d", "#f39c12", "#8a4f00"], "g4amber"),
    "rose": cmap_from_hex(["#ffe2eb", "#ff99b6", "#dc4a7a", "#7a1638"], "g4rose"),
    "green": cmap_from_hex(["#e4f7df", "#9ad88f", "#38a34f", "#0f5b26"], "g4green"),
    "slate": cmap_from_hex(["#eff3f8", "#b9c6d8", "#6b87b3", "#2b4b7c"], "g4slate"),
}


def palette_list(cmap_name, n, vmin=0.25, vmax=0.95):
    cmap = CMAPS[cmap_name]
    if n <= 1:
        return [cmap(0.75)]
    return [cmap(vmin + (vmax - vmin) * i / (n - 1)) for i in range(n)]


SPECIES_COLORS = {
    "Serratia marcescens": "#1f77b4",
    "Acinetobacter baumannii": "#2ca02c",
    "Klebsiella pneumoniae": "#ff7f0e",
    "Pseudomonas aeruginosa": "#9467bd",
    "Escherichia coli": "#d62728",
    "Serratia nevei": "#17becf",
    "Homo sapiens": "#7f7f7f",
}


CONF_COLORS = {
    "High-confidence": "#1f5fbf",
    "Priority-review": "#7b4fd6",
    "Other/remaining": "#c7d2e5",
    "Unassigned": "#e5e7eb",
}


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


def save_png_pdf(fig, stem, outdir):
    ensure_dir(outdir)
    png = os.path.join(outdir, f"{stem}.png")
    pdf = os.path.join(outdir, f"{stem}.pdf")
    fig.savefig(png, dpi=400)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def split_counted_items(field):
    out = []
    s = norm(field)
    if s == "" or s.lower() in {"na", "none", "nan", "no"}:
        return out
    parts = [p.strip() for p in s.split(";") if p.strip()]
    for token in parts:
        if token.endswith(")") and "(" in token:
            base = token[:token.rfind("(")].strip()
            try:
                count = int(token[token.rfind("(")+1:-1])
            except Exception:
                base, count = token, 1
        else:
            base, count = token, 1
        out.append((base, count))
    return out


def count_semicolon_items(field):
    s = norm(field)
    if s == "" or s.lower() in {"na", "none", "nan", "no"}:
        return 0
    return len([p for p in s.split(";") if p.strip()])


def build_mlst_label(row):
    scheme = norm(row.get("MLST_Scheme", ""))
    st = norm(row.get("MLST_ST", ""))
    if scheme == "" and st == "":
        return ""
    if scheme == "":
        return f"ST{st}"
    return f"{scheme} ST{st}"


def major_species_order(master_rows, min_n=4):
    cnt = Counter(norm(r.get("TopSpecies1", "")) for r in master_rows if norm(r.get("TopSpecies1", "")))
    return [sp for sp, n in cnt.most_common() if n >= min_n]


def load_confidence_map(work):
    highc_paths = [
        f"{work}/_MANUSCRIPT_FINAL/Table5_HighConfidenceSamples.csv",
        f"{work}/_PRELIMINARY_REPORT_READY/Table5_HighConfidenceSamples.csv",
    ]
    prio_paths = [
        f"{work}/_MANUSCRIPT_FINAL/Table4_PriorityReviewSamples.csv",
        f"{work}/_PRELIMINARY_REPORT_READY/Table4_PriorityReviewSamples.csv",
        f"{work}/_PRELIMINARY_PACK/Table_FlaggedSamples_refined.csv",
    ]

    highc = set()
    prio = set()

    for p in highc_paths:
        if os.path.isfile(p):
            rows = read_csv(p)
            col = guess_col(rows, ["Sample"])
            if col:
                highc = {norm(r[col]) for r in rows if norm(r[col])}
                break

    for p in prio_paths:
        if os.path.isfile(p):
            rows = read_csv(p)
            col = guess_col(rows, ["Sample"])
            if col:
                prio = {norm(r[col]) for r in rows if norm(r[col])}
                break

    cmap = {}
    for s in highc:
        cmap[s] = "High-confidence"
    for s in prio:
        cmap[s] = "Priority-review"
    return cmap, highc, prio


def confidence_of(sample, conf_map):
    return conf_map.get(sample, "Other/remaining")


def zscore_rows(data):
    arr = np.array(data, dtype=float)
    if arr.size == 0:
        return arr
    means = arr.mean(axis=0)
    stds = arr.std(axis=0)
    stds[stds == 0] = 1.0
    return (arr - means) / stds


def coerce_age(val):
    s = norm(val)
    if s == "":
        return None
    import re
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def soft_find_value(row, candidates):
    row_norm = {str(k).strip().lower(): v for k, v in row.items()}
    for cand in candidates:
        for k, v in row_norm.items():
            if cand.lower() == k or cand.lower() in k:
                return v
    return ""


def load_metadata_tables(metadata_dir):
    tables = []
    if not os.path.isdir(metadata_dir):
        return tables

    for fn in os.listdir(metadata_dir):
        fp = os.path.join(metadata_dir, fn)
        if os.path.isfile(fp) and fn.lower().endswith(".csv"):
            rows = read_csv(fp)
            sample_col = guess_col(rows, ["Sample", "Sample No.", "SAMPLE NO.", "sample no", "#", "Name"])
            tables.append({"source": fn, "rows": rows, "sample_col": sample_col})

    for fn in os.listdir(metadata_dir):
        fp = os.path.join(metadata_dir, fn)
        if not (os.path.isfile(fp) and fn.lower().endswith((".xlsx", ".xlsm"))):
            continue

        loaded = False
        try:
            import pandas as pd
            xls = pd.ExcelFile(fp)
            for sh in xls.sheet_names:
                df = pd.read_excel(fp, sheet_name=sh)
                rows = df.fillna("").to_dict(orient="records")
                sample_col = guess_col(rows, ["Sample", "Sample No.", "SAMPLE NO.", "sample no", "#", "Name"])
                tables.append({"source": f"{fn}::{sh}", "rows": rows, "sample_col": sample_col})
            loaded = True
        except Exception:
            pass
        if loaded:
            continue

        try:
            from openpyxl import load_workbook
            wb = load_workbook(fp, read_only=True, data_only=True)
            for ws in wb.worksheets:
                values = list(ws.values)
                if not values:
                    continue
                header = [norm(x) for x in values[0]]
                rows = []
                for row in values[1:]:
                    rows.append({header[i]: ("" if i >= len(row) or row[i] is None else row[i]) for i in range(len(header))})
                sample_col = guess_col(rows, ["Sample", "Sample No.", "SAMPLE NO.", "sample no", "#", "Name"])
                tables.append({"source": f"{fn}::{ws.title}", "rows": rows, "sample_col": sample_col})
        except Exception:
            pass

    return tables


def build_metadata_match_table(master_rows, metadata_dir):
    tables = load_metadata_tables(metadata_dir)
    master_by_sample = {norm(r.get("Sample", "")): r for r in master_rows}
    matched = []

    for tb in tables:
        sample_col = tb["sample_col"]
        if not sample_col:
            continue
        for r in tb["rows"]:
            sample = norm(r.get(sample_col, ""))
            if sample == "" or sample not in master_by_sample:
                continue
            base = master_by_sample[sample]
            out = {
                "Sample": sample,
                "TopSpecies1": norm(base.get("TopSpecies1", "")),
                "MLST_Label": build_mlst_label(base),
                "AMRFinder_Hits": as_int(base.get("AMRFinder_Hits", 0)),
                "VFDB_Hits": as_int(base.get("VFDB_Hits", 0)),
                "Plasmid_Hits": as_int(base.get("Plasmid_Hits", 0)),
                "MetadataSource": tb["source"],
                "Specimen": norm(soft_find_value(r, ["specimen", "site", "culture", "sample type"])),
                "Age": soft_find_value(r, ["age"]),
                "Gender": norm(soft_find_value(r, ["gender", "sex"])),
                "Ward_or_Unit": norm(soft_find_value(r, ["icu", "ward", "unit", "word"])),
                "Outcome": norm(soft_find_value(r, ["death", "survive", "outcome"])),
                "AdmissionReason": norm(soft_find_value(r, ["admission reason", "complain", "summary case", "history"])),
            }
            matched.append(out)

    best = {}
    for r in matched:
        score = sum(1 for k in ["Specimen", "Age", "Gender", "Ward_or_Unit", "Outcome", "AdmissionReason"] if norm(r.get(k, "")) != "")
        s = r["Sample"]
        if s not in best or score > best[s][0]:
            best[s] = (score, r)
    return [v[1] for v in best.values()]