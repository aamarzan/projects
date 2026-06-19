# auto_snakebite_plots_ultramax_v5.py
# 10/10 manuscript-grade figures: smart plot selection, ordering, figure stamping, captions, index, premium gradients.

import argparse
import re
from pathlib import Path
from textwrap import fill
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe
import matplotlib.dates as mdates


# -----------------------------
# Premium gradient palettes (curated, journal-safe)
# -----------------------------
def premium_cmaps():
    return {
        "qc": LinearSegmentedColormap.from_list("qc", ["#F7F7F7", "#BDBDBD", "#424242"]),
        "demographics": LinearSegmentedColormap.from_list("demo", ["#E7F6F5", "#4ED6C6", "#00796B"]),
        "time": LinearSegmentedColormap.from_list("time", ["#EAF2FF", "#7FB3FF", "#1C4E80"]),
        "exposure": LinearSegmentedColormap.from_list("expo", ["#F2ECFF", "#7C3AED", "#FDBA74"]),
        "treatment": LinearSegmentedColormap.from_list("treat", ["#FFF3E8", "#FB7185", "#B45309"]),
        "outcome": LinearSegmentedColormap.from_list("out", ["#EAF2FF", "#1E3A8A", "#DC2626"]),
    }


# -----------------------------
# Matplotlib journal style
# -----------------------------
def setup_mpl(style="ultra"):
    plt.rcParams.update({
        "figure.dpi": 180,
        "savefig.dpi": 900,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06,
        "figure.constrained_layout.use": True,

        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,

        "axes.linewidth": 0.85,
        "lines.linewidth": 1.6,

        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    if style.lower() == "nature":
        plt.rcParams.update({
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "lines.linewidth": 1.4,
        })


# -----------------------------
# Utilities
# -----------------------------
def make_unique_columns(cols):
    seen, out = {}, []
    for c in cols:
        c = str(c).strip()
        if c in seen:
            seen[c] += 1
            out.append(f"{c} ({seen[c]})")
        else:
            seen[c] = 0
            out.append(c)
    return out


def safe_filename(name: str) -> str:
    name = re.sub(r"[^\w\-\.\(\) ]+", "", name)
    name = name.strip().replace(" ", "_")
    return name[:180] if len(name) > 180 else name


def wrap_labels(labels, width=22):
    return [fill(str(l), width=width) for l in labels]


def format_axes(ax):
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", alpha=0.14, linewidth=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", length=5, width=0.9)


def clean_text(x):
    if pd.isna(x):
        return np.nan
    s = str(x).replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    if s == "":
        return np.nan

    low = s.lower()
    # normalize "not recorded" variants
    if low in {"na", "n/a", "not applicable", "not app", "not mention", "not mentioned", "not mention.", "not mentioned.", "not recorded"}:
        return "Not recorded"
    if low in {"-", "--"}:
        return np.nan
    return s


def normalize_yes_no(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if s in {"yes", "y", "true", "1"}:
        return "Yes"
    if s in {"no", "n", "false", "0"}:
        return "No"
    return clean_text(x)

def drop_labels_case_insensitive(series: pd.Series, drop_labels) -> pd.Series:
    """Drop categories like 'Not recorded' regardless of case."""
    s = series.map(clean_text)
    drop_set = {str(x).strip().casefold() for x in drop_labels}
    m = s.notna() & s.astype(str).str.strip().str.casefold().isin(drop_set)
    return s.mask(m, np.nan)

def merge_labels_case_insensitive(series: pd.Series, from_labels, to_label: str) -> pd.Series:
    """Merge categories like 'Not recorded' into 'No' regardless of case."""
    s = series.map(clean_text)
    from_set = {str(x).strip().casefold() for x in from_labels}
    m = s.notna() & s.astype(str).str.strip().str.casefold().isin(from_set)
    s.loc[m] = to_label
    return s

def _std_label(v: str) -> str:
    """Pretty label while still allowing case-insensitive grouping."""
    v = re.sub(r"\s+", " ", str(v).strip())
    if v == "":
        return v
    low = v.casefold()

    if low == "not recorded":
        return "Not recorded"
    if low in {"yes", "no"}:
        return low.title()

    # keep short acronyms like WBCT, UHC, ICU
    if v.isupper() and v.isalpha() and len(v) <= 5:
        return v

    # sentence case (arm -> Arm, ARM -> Arm)
    return v[:1].upper() + v[1:].lower()


def standardize_series_casefold(series: pd.Series) -> pd.Series:
    """
    Case-insensitive standardization:
    - Groups values by casefold key
    - Uses the most frequent raw spelling as representative
    - Then prettifies it via _std_label
    """
    s = series.map(clean_text)
    out = pd.Series(np.nan, index=series.index, dtype=object)

    mask = s.notna()
    if not mask.any():
        return out

    vals = s.loc[mask].astype(str)
    keys = vals.str.casefold()

    rep = vals.groupby(keys).agg(lambda x: x.value_counts().idxmax())
    rep = rep.map(_std_label)

    out.loc[mask] = keys.map(rep).values
    return out


# ---- F15: Site-of-bite bucketing ----
BITE_SITE_ORDER = [
    "Foot",
    "Leg",
    "Arm",
    "Others",
    "Thigh",
    "Finger",
    "Hand",
    "Fore arm",
    "Toes",
    "Trunk",
    "Head and neck",
]

def map_bite_site(x):
    s = clean_text(x)
    if pd.isna(s):
        return np.nan
    t = str(s).strip().casefold()
    t = re.sub(r"[^a-z0-9\s\-/]", " ", t)
    t = re.sub(r"\s+", " ", t)

    # order matters: specific first
    if "forearm" in t or "fore arm" in t or "fore-arm" in t:
        return "Fore arm"
    if "hand" in t:
        return "Hand"
    if "finger" in t or "thumb" in t:
        return "Finger"
    if "toe" in t or "toes" in t:
        return "Toes"
    if "foot" in t or "feet" in t:
        return "Foot"
    if "thigh" in t:
        return "Thigh"
    if "leg" in t or "calf" in t or "knee" in t:
        return "Leg"
    if "arm" in t or "elbow" in t:
        return "Arm"
    if "head" in t or "neck" in t or "face" in t or "scalp" in t:
        return "Head and neck"
    if any(k in t for k in ["trunk", "chest", "abdomen", "back", "waist", "torso"]):
        return "Trunk"

    return "Others"


# ---- Outcome normalization for F30/F31/F33 ----
OUTCOME_ORDER = ["Improved", "Died", "Improved with disability"]

def normalize_outcome(x):
    s = clean_text(x)
    if pd.isna(s):
        return np.nan
    t = str(s).strip().casefold()

    # any disability variant
    if "disab" in t:
        return "Improved with disability"

    # any death variant -> Died
    if t in {"death", "dead", "died", "expired"} or "die" in t:
        return "Died"

    # everything else -> Improved
    return "Improved"

def gradient_colors(values, cmap):
    values = np.asarray(values, dtype=float)
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if vmax == vmin:
        vmax = vmin + 1.0
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    # Use rank scaling for nicer contrast in small datasets (10/10 polish)
    ranks = pd.Series(values).rank(method="average").values
    ranks = (ranks - ranks.min()) / (ranks.max() - ranks.min() + 1e-9)
    ranks = 0.12 + ranks * 0.84  # avoid washed-out extremes
    colors = cmap(ranks)
    return colors, norm


def add_colorbar(ax, norm, cmap, label):
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, pad=0.01, fraction=0.05)
    cb.set_label(label, fontsize=9)


def choose_figsize(n_items, single=(7.2, 4.4), double=(8.2, 5.0)):
    # if many categories/groups, use a slightly larger canvas
    return double if n_items >= 10 else single


# -----------------------------
# Export manager (vector + high-DPI PNG) + figure stamping + index/captions
# -----------------------------
class Exporter:
    def __init__(self, out_dir: Path, pdf: PdfPages, dataset_label: str = "", only_set=None, skip_set=None):
        self.out_dir = out_dir
        self.pdf = pdf
        self.i = 0
        self.records = []
        self.dataset_label = dataset_label.strip()
        self.only_set = only_set  # None or set like {"F25"}
        self.skip_set = skip_set or set()

    def stamp(self, fig, fig_id: str):
        # ✅ REMOVE figure serial stamp (F01, F02, ...)
        # fig.text(0.01, 0.99, fig_id, ha="left", va="top",
        #          fontsize=12, fontweight="bold")

        # Optional: keep dataset label (right side) if you want
        if self.dataset_label:
            fig.text(
                0.99, 0.99, self.dataset_label,
                ha="right", va="top",
                fontsize=9, alpha=0.7
            )

    def save(self, fig, title: str, caption_hint: str = ""):
        self.i += 1
        fig_id = f"F{self.i:02d}"

        # --- Only/Skip figure export (keeps figure numbering stable) ---
        if self.only_set is not None and fig_id not in self.only_set:
            plt.close(fig)
            return
        if fig_id in self.skip_set:
            plt.close(fig)
            return

        name = safe_filename(f"{fig_id}_{title}")


        self.stamp(fig, fig_id)

        png = self.out_dir / f"{name}.png"
        pdf_path = self.out_dir / f"{name}.pdf"
        svg = self.out_dir / f"{name}.svg"

        fig.savefig(png)
        fig.savefig(pdf_path)
        fig.savefig(svg)
        self.pdf.savefig(fig)
        plt.close(fig)

        self.records.append({
            "Figure": fig_id,
            "Title": title,
            "PNG": str(png.name),
            "PDF": str(pdf_path.name),
            "SVG": str(svg.name),
            "Caption_hint": caption_hint
        })

    def finalize(self):
        # figure index
        idx = pd.DataFrame(self.records)
        idx.to_csv(self.out_dir / "figure_index.csv", index=False)

        # captions
        lines = ["# Suggested captions", ""]
        for r in self.records:
            hint = r["Caption_hint"] if r["Caption_hint"] else "Describe what is shown, include n, and key patterns."
            lines.append(f"**{r['Figure']} — {r['Title']}**  \n{hint}\n")
        (self.out_dir / "captions.md").write_text("\n".join(lines), encoding="utf-8")


# -----------------------------
# 10/10 plot primitives
# -----------------------------
def counts_top_other(series, top_n=15, other_label="Other", preferred_order=None):
    s = series.map(clean_text).dropna().astype(str)
    if s.empty:
        return None

    # Case-insensitive grouping
    keys = s.str.casefold()
    vc = keys.value_counts()

    # Representative label per casefold key
    rep = s.groupby(keys).agg(lambda x: x.value_counts().idxmax()).map(_std_label)
    vc.index = [rep[k] for k in vc.index]

    # Preferred order (also prettified)
    if preferred_order:
        pref = [_std_label(x) for x in preferred_order]
        for k in pref:
            if k not in vc.index:
                vc.loc[k] = 0
        vc = vc.loc[pref + [x for x in vc.index if x not in pref]]
        return vc

    if top_n is not None and len(vc) > top_n:
        top = vc.iloc[:top_n]
        other = vc.iloc[top_n:].sum()
        vc = top.copy()
        vc[other_label] = other

    return vc


def rounded_barh(ax, y, values, colors, height=0.72):
    for yi, val, col in zip(y, values, colors):
        patch = FancyBboxPatch(
            (0, yi - height / 2),
            float(val),
            height,
            boxstyle="round,pad=0.02,rounding_size=0.18",
            linewidth=0.0,
            facecolor=col,
            alpha=0.98
        )
        patch.set_path_effects([
            pe.SimplePatchShadow(offset=(0.6, -0.6), alpha=0.10),
            pe.Normal()
        ])
        ax.add_patch(patch)
    ax.set_ylim(-0.8, len(values) - 0.2)


def lollipop(ax, labels, values, colors):
    y = np.arange(len(values))
    ax.hlines(y, 0, values, color=colors, alpha=0.55, linewidth=3.2)
    ax.scatter(values, y, s=85, c=colors, edgecolors="none")
    ax.set_yticks(y)
    ax.set_yticklabels(wrap_labels(labels, width=24))
    ax.invert_yaxis()


def categorical_ultramax(series, title, exporter: Exporter, cmap, top_n=15, show_colorbar=False, preferred_order=None):
    # fixed order for yes/no fields if present
    preferred = None
    s_clean = series.map(clean_text)
    uniq = set(s_clean.dropna().astype(str).str.strip().str.casefold().unique())

    # Only include "Not recorded" in the preferred order if it actually exists
    if uniq.issubset({"yes", "no", "not recorded"}):
        if "not recorded" in uniq:
            preferred = ["No", "Yes", "Not recorded"]
        else:
            preferred = ["No", "Yes"]

    vc = counts_top_other(series, top_n=top_n, preferred_order=(preferred_order or preferred))
    if vc is None or vc.sum() == 0:
        return

    labels = vc.index.tolist()
    values = vc.values.astype(float)
    colors, norm = gradient_colors(values, cmap)

    fig = plt.figure(figsize=choose_figsize(len(values)))
    ax = plt.gca()
    ax.set_title(title, loc="left", pad=10)

    # Smart chart selection:
    # - Few categories: rounded bars (premium)
    # - Many categories: lollipop/dot (cleaner for high-IF)
    if len(values) <= 9:
        y = np.arange(len(values))
        rounded_barh(ax, y, values, colors)
        ax.set_yticks(y)
        ax.set_yticklabels(wrap_labels(labels, width=24))
        ax.invert_yaxis()
    else:
        lollipop(ax, labels, values, colors)

    ax.set_xlabel("Count")
    format_axes(ax)
    ax.set_xlim(0, max(values) * 1.20)

    total = values.sum()
    for yi, v in enumerate(values):
        if v > 0:
            ax.text(
                v, yi, f"  {int(v)} ({(v/total)*100:.1f}%)",
                va="center", ha="left", fontsize=9,
                path_effects=[pe.withStroke(linewidth=3, foreground="white", alpha=0.85)]
            )

    if show_colorbar:
        add_colorbar(ax, norm, cmap, "Count")

    exporter.save(fig, title, caption_hint=f"Distribution of {title.lower()} (n={int(total)}). Values shown as count and percent.")


def stacked_100_ultramax(df, row_var, col_var, title, exporter: Exporter, cmap, top_rows=10, top_cols=8, drop_row_values=None):
    tmp = df[[row_var, col_var]].copy()

    tmp[row_var] = standardize_series_casefold(tmp[row_var])
    tmp[col_var] = standardize_series_casefold(tmp[col_var])

    tmp = tmp.dropna()
    if tmp.empty:
        return

    if drop_row_values:
        drop_set = {_std_label(x) for x in drop_row_values}
        tmp = tmp[~tmp[row_var].isin(drop_set)]
        if tmp.empty:
            return

    ct = pd.crosstab(tmp[row_var], tmp[col_var])

    if ct.shape[0] > top_rows:
        ct = ct.loc[ct.sum(axis=1).sort_values(ascending=False).head(top_rows).index]
    if ct.shape[1] > top_cols:
        ct = ct[ct.sum(axis=0).sort_values(ascending=False).head(top_cols).index]

    ct = ct.div(ct.sum(axis=1), axis=0).fillna(0) * 100
    seg_colors = cmap(np.linspace(0.12, 0.92, ct.shape[1]))

    fig = plt.figure(figsize=(8.2, 5.0))
    ax = plt.gca()
    ax.set_title(title + " (100% stacked)", loc="left", pad=10)

    ct.plot(kind="bar", stacked=True, ax=ax, color=seg_colors, width=0.72)

    ax.set_ylabel("Percent")
    ax.set_xlabel("")
    plt.xticks(rotation=35, ha="right")
    ax.legend(title=col_var, bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    format_axes(ax)

    exporter.save(fig, title, caption_hint=f"Composition of {col_var.lower()} within each {row_var.lower()} (100% stacked).")


def hist_ultramax(series, title, exporter: Exporter, cmap):
    x = pd.to_numeric(series, errors="coerce").dropna()
    if x.empty:
        return

    # Freedman–Diaconis bins
    q75, q25 = np.percentile(x, [75, 25])
    iqr = q75 - q25
    if iqr > 0:
        bw = 2 * iqr * (len(x) ** (-1/3))
        bins = int(np.ceil((x.max() - x.min()) / bw)) if bw > 0 else 12
    else:
        bins = 12
    bins = int(np.clip(bins, 8, 28))

    counts, edges = np.histogram(x, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    widths = np.diff(edges)

    colors, norm = gradient_colors(counts.astype(float), cmap)

    fig = plt.figure(figsize=(7.6, 4.6))
    ax = plt.gca()
    ax.set_title(f"{title} (n={len(x)})", loc="left", pad=10)

    ax.bar(centers, counts, width=widths, color=colors, edgecolor="none")

    # subtle rug (adds polish without clutter)
    ax.plot(x.values, np.full_like(x.values, -0.25), "|", markersize=8, alpha=0.22)

    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
    format_axes(ax)
    add_colorbar(ax, norm, cmap, "Bin count")

    exporter.save(fig, title, caption_hint=f"Histogram of {title.lower()} (n={len(x)}). Bin color indicates bin frequency.")

def antivenom_dose_discrete_with_undetermined(series, title, exporter: Exporter, cmap):
    s = series.map(clean_text)
    if s.dropna().empty:
        return

    # count "Undetermined" (case-insensitive)
    und_mask = s.dropna().astype(str).str.strip().str.casefold().eq("undetermined")
    und_n = int(und_mask.sum())

    # numeric doses (keep integers like 2, 3, 10, 20, 26, 30, 40)
    s_num = s.dropna().astype(str)
    s_num = s_num[~s_num.str.strip().str.casefold().eq("undetermined")]
    x = pd.to_numeric(s_num, errors="coerce").dropna()
    if x.empty and und_n == 0:
        return

    # Convert to int safely (vials should be discrete)
    x_int = x.round().astype(int)

    counts_num = x_int.value_counts().sort_index()
    labels = [str(v) for v in counts_num.index.tolist()]
    values = counts_num.values.astype(float).tolist()

    # Append Undetermined at the END
    if und_n > 0:
        labels.append("Undetermined")
        values.append(float(und_n))

    values_arr = np.asarray(values, dtype=float)
    colors, norm = gradient_colors(values_arr, cmap)

    fig = plt.figure(figsize=choose_figsize(len(values), single=(7.6, 4.6), double=(8.6, 5.2)))
    ax = plt.gca()
    ax.set_title(f"{title} (n={int(values_arr.sum())})", loc="left", pad=10)

    x_pos = np.arange(len(values))
    ax.bar(x_pos, values_arr, color=colors, edgecolor="none", width=0.78)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_xlabel("Dose (vials)")
    ax.set_ylabel("Count")
    format_axes(ax)

    # value labels
    total = float(values_arr.sum()) if values_arr.sum() > 0 else 1.0
    for i, v in enumerate(values_arr):
        if v > 0:
            ax.text(
                i, v, f"{int(v)} ({(v/total)*100:.1f}%)",
                ha="center", va="bottom", fontsize=9,
                path_effects=[pe.withStroke(linewidth=3, foreground="white", alpha=0.85)]
            )

    exporter.save(fig, title, caption_hint="Discrete distribution of antivenom vial doses; 'Undetermined' shown as a separate category at the end.")

def violin_box_jitter_ultramax(df, numeric_col, group_col, title, exporter: Exporter, cmap, top_groups=12):
    x = pd.to_numeric(df[numeric_col], errors="coerce")
    g = df[group_col].map(clean_text)
    tmp = pd.DataFrame({numeric_col: x, group_col: g}).dropna()
    if tmp.empty:
        return

    # order groups by median (10/10 readability)
    med = tmp.groupby(group_col)[numeric_col].median().sort_values(ascending=True)
    groups = med.index.tolist()[:top_groups]

    tmp = tmp[tmp[group_col].isin(groups)]
    if tmp.empty:
        return

    # re-order exactly by median
    groups = med.loc[[g for g in med.index if g in groups]].index.tolist()
    data = [tmp.loc[tmp[group_col] == gg, numeric_col].values for gg in groups]
    ns = [len(v) for v in data]

    fig = plt.figure(figsize=(8.6, 5.2))
    ax = plt.gca()
    ax.set_title(title, loc="left", pad=10)

    viol = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False)
    for body in viol["bodies"]:
        body.set_alpha(0.16)
        body.set_edgecolor("none")

    ax.boxplot(
        data,
        labels=[f"{fill(gg, 12)}\n(n={n})" for gg, n in zip(groups, ns)],
        notch=True,
        showfliers=False,
        widths=0.52
    )

    vals_all = tmp[numeric_col].values.astype(float)
    _, norm = gradient_colors(vals_all, cmap)

    rng = np.random.default_rng(7)
    for i, gg in enumerate(groups, start=1):
        vals = tmp.loc[tmp[group_col] == gg, numeric_col].values.astype(float)
        jitter_x = rng.normal(loc=i, scale=0.06, size=len(vals))
        ax.scatter(jitter_x, vals, s=18, alpha=0.55, c=vals, cmap=cmap, norm=norm, edgecolors="none")

    ax.set_xlabel(group_col)
    ax.set_ylabel(numeric_col)
    plt.xticks(rotation=30, ha="right")
    format_axes(ax)
    add_colorbar(ax, norm, cmap, numeric_col)

    exporter.save(fig, title, caption_hint=f"Distribution of {numeric_col.lower()} across {group_col.lower()} (violin + box + value-colored jitter). Groups are ordered by median.")


def timeseries_daily_ultramax(df, date_col, title, exporter: Exporter, cmap, by_col=None):
    d = df.dropna(subset=[date_col]).copy()
    if d.empty:
        return

    d["date"] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=["date"])
    if d.empty:
        return

    fig = plt.figure(figsize=(8.6, 5.0))
    ax = plt.gca()
    ax.set_title(title, loc="left", pad=10)

    if by_col is None:
        daily = d.groupby(d["date"].dt.date).size()
        dates = pd.to_datetime(daily.index)
        y = daily.values.astype(float)

        colors, norm = gradient_colors(y, cmap)
        ax.plot(dates, y, alpha=0.22, linewidth=1.2)
        ax.scatter(dates, y, c=y, cmap=cmap, norm=norm, s=46, edgecolors="none")
        add_colorbar(ax, norm, cmap, "Daily count")
        ax.set_ylabel("Count")
    else:
        tmp = d.copy()
        tmp[by_col] = tmp[by_col].map(clean_text)
        tmp = tmp.dropna(subset=[by_col])
        if tmp.empty:
            return
        ct = pd.crosstab(tmp["date"].dt.date, tmp[by_col])
        ct.index = pd.to_datetime(ct.index)
        ct.sort_index(inplace=True)

        line_colors = cmap(np.linspace(0.12, 0.92, ct.shape[1]))
        for (col, c) in zip(ct.columns, line_colors):
            ax.plot(ct.index, ct[col].values, label=col, color=c, alpha=0.95)

        ax.legend(title=by_col, bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
        ax.set_ylabel("Count")

    ax.set_xlabel("Date")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    format_axes(ax)

    exporter.save(fig, title, caption_hint=f"Admissions over time using {date_col.lower()}" + (f", stratified by {by_col.lower()}." if by_col else "."))


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV or XLSX")

    # accept BOTH --out and --outdir (alias)
    ap.add_argument("--out", default=None, help="Output folder")
    ap.add_argument("--outdir", default=None, help="Alias for --out (compatibility)")

    # Excel sheet support
    ap.add_argument("--sheet", default=None, help="Excel sheet name (optional)")

    ap.add_argument("--style", default="ultra", choices=["ultra", "nature"], help="Typography preset")
    ap.add_argument("--topn", type=int, default=15, help="Top N categories (rest -> Other)")
    ap.add_argument("--only", default=None, help="Comma-separated figure IDs to export, e.g. F25 or F25,F26")
    ap.add_argument("--skip", default=None, help="Comma-separated figure IDs to skip, e.g. F01,F02")

    args = ap.parse_args()

    setup_mpl(args.style)
    cmaps = premium_cmaps()

    # ✅ define in_path BEFORE using it
    in_path = Path(args.input)

    # ✅ resolve output folder
    out_value = args.outdir or args.out or "snakebite_plots_ultramax"
    out_dir = Path(out_value)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ✅ load data (sheet supported)
    if in_path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(in_path, sheet_name=args.sheet if args.sheet else 0)
    else:
        df = pd.read_csv(in_path)


    df.columns = make_unique_columns(df.columns)

    # Clean string columns
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(clean_text)

    # Parse date
    if "Date of admission" in df.columns:
        df["Date of admission"] = pd.to_datetime(df["Date of admission"], errors="coerce", dayfirst=True)

    # Normalize yes/no-like columns
    yesno_keywords = [
        "venomous", "given", "needed", "provided", "developed",
        "identified", "reaction", "dialysis", "transfused", "tourniquet",
        "immobilization", "incision", "herbal", "ohza", "previous", "history"
    ]
    for c in df.columns:
        if any(k in c.lower() for k in yesno_keywords):
            df[c] = df[c].map(normalize_yes_no)

    
    with PdfPages(out_dir / "combined_plots.pdf") as pdf:
        only_set = {x.strip().upper() for x in args.only.split(",") if x.strip()} if args.only else None
        skip_set = {x.strip().upper() for x in args.skip.split(",") if x.strip()} if args.skip else set()
        exporter = Exporter(out_dir, pdf, only_set=only_set, skip_set=skip_set)


        # QC: missingness heatmap
        miss = df.isna().astype(int).values
        fig = plt.figure(figsize=(8.8, 4.4))
        ax = plt.gca()
        im = ax.imshow(miss, aspect="auto", cmap=cmaps["qc"])
        ax.set_title("Missingness heatmap (1 = missing)", loc="left", pad=10)
        ax.set_xlabel("Columns")
        ax.set_ylabel("Rows")
        plt.colorbar(im, ax=ax, pad=0.01, fraction=0.05)
        exporter.save(fig, "Missingness heatmap", caption_hint="Row-wise and column-wise missingness pattern (1=missing).")

        # QC: % missing per column (cleaner dot-plot)
        miss_pct = (df.isna().mean() * 100).sort_values(ascending=False)
        fig = plt.figure(figsize=(8.8, 4.8))
        ax = plt.gca()
        vals = miss_pct.values.astype(float)
        colors, norm = gradient_colors(vals, cmaps["qc"])
        y = np.arange(len(vals))
        ax.hlines(y, 0, vals, color=colors, alpha=0.55, linewidth=3.0)
        ax.scatter(vals, y, s=80, c=colors, edgecolors="none")
        ax.set_yticks(y)
        ax.set_yticklabels(wrap_labels(miss_pct.index.tolist(), width=22))
        ax.invert_yaxis()
        ax.set_xlabel("% missing")
        ax.set_title("Percent missing by column", loc="left", pad=10)
        format_axes(ax)
        add_colorbar(ax, norm, cmaps["qc"], "% missing")
        exporter.save(fig, "Percent missing by column", caption_hint="Percent missingness per variable (dot-plot).")

        # Demographics
        if "Study site" in df.columns:
            categorical_ultramax(df["Study site"], "Records by study site", exporter, cmaps["demographics"], top_n=None, show_colorbar=True)

        if "Sex" in df.columns:
            categorical_ultramax(df["Sex"], "Sex distribution", exporter, cmaps["demographics"], top_n=None)
            if "Study site" in df.columns:
                stacked_100_ultramax(df, "Study site", "Sex", "Sex by study site", exporter, cmaps["demographics"])

        if "Age" in df.columns:
            hist_ultramax(df["Age"], "Age distribution", exporter, cmaps["demographics"])
            if "Study site" in df.columns:
                violin_box_jitter_ultramax(df, "Age", "Study site", "Age by study site", exporter, cmaps["demographics"])
            if "Sex" in df.columns:
                violin_box_jitter_ultramax(df, "Age", "Sex", "Age by sex", exporter, cmaps["demographics"], top_groups=5)

        if "Occupation" in df.columns:
            categorical_ultramax(df["Occupation"], "Occupation (top categories)", exporter, cmaps["demographics"], top_n=args.topn)

        if "if Female, Pregnancy status" in df.columns and "Sex" in df.columns:
            females = df[df["Sex"].astype(str).str.strip().str.lower().eq("female")]
            if not females.empty:
                categorical_ultramax(females["if Female, Pregnancy status"], "Pregnancy status (females only)", exporter, cmaps["demographics"], top_n=None, show_colorbar=True)

        # Time
        if "Date of admission" in df.columns:
            timeseries_daily_ultramax(df, "Date of admission", "Admissions over time (daily)", exporter, cmaps["time"], by_col=None)
            if "Study site" in df.columns:
                timeseries_daily_ultramax(df, "Date of admission", "Admissions over time by study site", exporter, cmaps["time"], by_col="Study site")

            d = df.dropna(subset=["Date of admission"]).copy()
            if not d.empty:
                categorical_ultramax(d["Date of admission"].dt.day_name(), "Admissions by day of week", exporter, cmaps["time"], top_n=None)

        # Exposure / bite
        for col, ttl, topn in [
            ("Venomous", "Venomous status", None),
            ("Site of bite", "Site of bite", args.topn),
            ("Location where bite happened", "Location where bite happened", args.topn),
            ("Activity of victim during bite", "Activity during bite", args.topn),
            ("If awake, any interaction with the biting snake", "Interaction with snake", args.topn),
        ]:
            if col in df.columns:

                if col == "Venomous":
                    ser = drop_labels_case_insensitive(df[col], {"Not recorded"})
                    categorical_ultramax(ser, ttl, exporter, cmaps["exposure"], top_n=topn)
                    continue

                if col == "Site of bite":
                    ser = df[col].map(map_bite_site)
                    categorical_ultramax(ser, ttl, exporter, cmaps["exposure"], top_n=None)  # sorted high->low automatically
                    continue

                categorical_ultramax(df[col], ttl, exporter, cmaps["exposure"], top_n=topn)


        # Treatment
        DROP_NOT_RECORDED = {
            "Antivenom given",
            "Assisted ventilation provided",
            "Dialysis needed",
            "Blood/Plasma transfused",
            "Bedside 20 min WBCT",
            "Necrosis developed during hospital stay",
        }

        MERGE_NOTREC_TO_NO = {
            "Early antivenom reaction",
        }

        for col, ttl in [
            ("Antivenom given", "Antivenom given"),
            ("Early antivenom reaction", "Early antivenom reaction"),
            ("Assisted ventilation provided", "Assisted ventilation provided"),
            ("Dialysis needed", "Dialysis needed"),
            ("Blood/Plasma transfused", "Blood/Plasma transfused"),
            ("Bedside 20 min WBCT", "Bedside 20 min WBCT"),
            ("Necrosis developed during hospital stay", "Necrosis developed during stay"),
        ]:
            if col in df.columns:
                ser = df[col]

                # F19/F21/F22/F23/F24: drop Not recorded
                if col in DROP_NOT_RECORDED:
                    ser = drop_labels_case_insensitive(ser, {"Not recorded"})

                # F20: merge Not recorded into No
                if col in MERGE_NOTREC_TO_NO:
                    ser = merge_labels_case_insensitive(ser, {"Not recorded"}, "No")

                categorical_ultramax(ser, ttl, exporter, cmaps["treatment"], top_n=None)
                

        if "Total dose of antivenom (vials)" in df.columns:
            antivenom_dose_discrete_with_undetermined(
                df["Total dose of antivenom (vials)"],
                "Antivenom dose (vials)",
                exporter,
                cmaps["treatment"]
            )

            if "Venomous" in df.columns:
                violin_box_jitter_ultramax(df, "Total dose of antivenom (vials)", "Venomous",
                                           "Antivenom dose by venomous status", exporter, cmaps["treatment"], top_groups=5)
            if "Early antivenom reaction" in df.columns:
                violin_box_jitter_ultramax(df, "Total dose of antivenom (vials)", "Early antivenom reaction",
                                           "Antivenom dose by early reaction", exporter, cmaps["treatment"], top_groups=5)

        # Outcome
        if "Outcome" in df.columns:
            # normalize outcomes (everything else -> Improved)
            df["Outcome"] = df["Outcome"].map(normalize_outcome)

            categorical_ultramax(df["Outcome"], "Outcome", exporter, cmaps["outcome"], top_n=None, preferred_order=OUTCOME_ORDER)

            if "Study site" in df.columns:
                stacked_100_ultramax(df, "Study site", "Outcome", "Outcome by study site", exporter, cmaps["outcome"], top_cols=3)

            if "Venomous" in df.columns:
                # F32: remove the "Yes" row
                stacked_100_ultramax(df, "Venomous", "Outcome", "Outcome by venomous status", exporter, cmaps["outcome"], top_cols=3, drop_row_values={"Yes"})

            if "Age" in df.columns:
                violin_box_jitter_ultramax(df, "Age", "Outcome", "Age by outcome", exporter, cmaps["outcome"], top_groups=10)

        # Save clean snapshot
        df.to_csv(out_dir / "cleaned_snapshot.csv", index=False)

        exporter.finalize()

    print(f"✅ UltraMax complete: {out_dir.resolve()}")
    print("Outputs: combined_plots.pdf + per-figure PNG/PDF/SVG + figure_index.csv + captions.md + cleaned_snapshot.csv")


if __name__ == "__main__":
    main()
