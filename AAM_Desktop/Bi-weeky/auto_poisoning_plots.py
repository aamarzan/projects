# auto_poisoning_plots_ultramax_v7_journal.py
# UltraMax premium + journal-styled figure generator for poisoning/IPD datasets
# Exports: per-figure PDF+SVG+900DPI PNG + combined multi-page PDF + figure_index.csv + captions.md
# NOTE: Drops PII columns from cleaned exports.

from __future__ import annotations

import os
import re
import math
import argparse
import textwrap
import matplotlib.dates as mdates
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib.patheffects as pe
from matplotlib.ticker import MultipleLocator, FuncFormatter

# ============================================================
# GLOBAL SWITCH: do not show figure id on the figure itself
# ============================================================
SHOW_FIG_ID_ON_FIGURE = False

# ============================================================
# FIGURE-SPECIFIC TWEAKS
# ============================================================
LEGEND_BELOW_FIGS = {"F07", "F55"}  # move legend under x-axis (premium)

# Histogram x-limits (and values outside range are excluded)
HIST_XLIM: Dict[str, Tuple[float, float]] = {
    "F14": (0, 10),
    "F16": (0, 30),
    "F21": (0, 170),
    "F23": (0, 180),
    "F25": (0, 170),
    "F27": (0, 50),
    "F29": (0, 15),
    "F31": (0, 175),
    "F33": (0, 150),
}

# Violin / value filtering rules by Figure ID
# "max": drop values > max
# "between": keep only values between [lo, hi]
VIOLIN_FILTERS: Dict[str, Dict[str, Any]] = {
    "F13": {"max": 100},                # tablet count
    "F22": {"max": 150},                # temperature by type
    "F24": {"max": 200},                # pulse by type
    "F26": {"max": 200},                # SpO2 by type
    "F28": {"between": (10, 50)},       # RR by type: remove <10, also cap at 50
    "F30": {"max": 15},                 # GCS by type
    "F34": {"max": 150},                # DBP by type

    # LABs (user updated corrections)
    "F35": {"max": 45000},
    "F36": {"max": 250},
    "F37": {"max": 100},
    "F38": {"between": (0, 7.5)},
    "F39": {"max": 25},
    "F40": {"between": (75, 175)},
    "F41": {"max": 20},
    "F42": {"between": (80, 120)},
    "F44": {"max": 200},
}

# For F45–F52: robust clinically-impossible removal (generic + mild)
ROBUST_OUTLIER_FIGS = {f"F{n:02d}" for n in range(45, 53)}

POISON_UNKNOWN_LABEL = "Unknown"

# F11: remove Unknown/Other (applied to whichever plot becomes F11)
F11_DROP_LABELS = {
    "Other", "Unknown", "Unknown/Other", "Other/Unknown",
    "Other / unspecified"
}

# Outcomes: rename Death -> Died (everywhere)
OUTCOME_DEATH_LABEL = "Died"
OUTCOME_ABSCONDED_LABEL = "Absconded"
OUTCOME_UNKNOWN_LABEL = "Unknown outcome"
OUTCOME_SURV_NO_COMP = "Survived (no complications)"
OUTCOME_SURV_COMP = "Survived (with complications)"

# F54–F56: DO NOT merge unknown into absconded (revert previous merge)
MERGE_UNKNOWN_OUTCOME_INTO_ABSCONDED = False

# Label wrapping like Excel
WRAP_WIDTH_DEFAULT = 14
BANDDOWN_AND_TO_AMP = True  # "X and Y" -> "X\n& Y" when long

# Always keep these at the right-most end of bar charts
SPECIAL_LAST = ["Other", "Unknown"]


# ============================================================
# SAFE "SPELLING / LABEL CORRECTIONS"
# (controlled mapping only; no random guessing)
# ============================================================
def _cf(s: str) -> str:
    return str(s).strip().casefold()

LABEL_CORRECTIONS_CASEFOLD = {
    # poison type corrections
    "herpic": "Harpic",
    "street poisoning": "Street Poisoning",
    "commuter poisoning": "Commuter",

    # common user example: construction typos
    "manual labour and contruction": "manual labour and construction",
    "manual labour and contraction": "manual labour and construction",
    "manual labour and construction": "manual labour and construction",

    # outcome correction
    "death": OUTCOME_DEATH_LABEL,
    "died": OUTCOME_DEATH_LABEL,
}

def apply_label_corrections(x):
    if pd.isna(x):
        return x
    s = str(x)
    key = _cf(s)
    if key in LABEL_CORRECTIONS_CASEFOLD:
        return LABEL_CORRECTIONS_CASEFOLD[key]
    return s

# ---------------------------
# Helper normalizers
# ---------------------------
def _is_yes(x) -> bool:
    if pd.isna(x):
        return False
    s = str(x).strip().casefold()
    return s in {"yes", "y", "1", "true", "t"}

def norm_text(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    return s

def to_nan_if_bad(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s == "" or s.upper() in {"NA", "N/A", "NULL", "NONE"}:
        return np.nan
    if "#VALUE!" in s.upper():
        return np.nan
    return x

def norm_category_exact(x, unknown="Unknown"):
    """Keep Excel label but normalize invisible junk/spaces; apply safe corrections."""
    x = to_nan_if_bad(x)
    if pd.isna(x):
        return unknown
    s = str(x).replace("\u00A0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    s = s.replace("–", "-").replace("—", "-").replace("−", "-")
    if s == "" or s.casefold() in {"na", "n/a", "none", "null"}:
        return unknown
    s = apply_label_corrections(s)
    return s

def parse_date_series(s: pd.Series) -> pd.Series:
    s2 = s.apply(to_nan_if_bad)
    return pd.to_datetime(s2, errors="coerce", dayfirst=True)

def parse_bp(bp_series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    sys_list, dia_list = [], []
    for v in bp_series.fillna(""):
        t = norm_text(v)
        m = re.match(r"^\s*(\d{2,3})\s*/\s*(\d{2,3})\s*$", t)
        if m:
            sys_list.append(float(m.group(1)))
            dia_list.append(float(m.group(2)))
        else:
            sys_list.append(np.nan)
            dia_list.append(np.nan)
    return pd.Series(sys_list, index=bp_series.index), pd.Series(dia_list, index=bp_series.index)

def parse_duration_to_hours(val) -> float:
    if pd.isna(val):
        return np.nan
    s = norm_text(val).lower()
    if s == "" or "value!" in s:
        return np.nan
    if "instant" in s:
        return 0.0
    m = re.search(r"([-+]?\d+(\.\d+)?)", s)
    if not m:
        return np.nan
    num = float(m.group(1))
    if "min" in s:
        return num / 60.0
    if "hour" in s or "hr" in s or "hrs" in s or s.endswith("h"):
        return num
    return num

def parse_amount_ml(val) -> float:
    if pd.isna(val):
        return np.nan
    s = norm_text(val).lower()
    if s == "" or "value!" in s:
        return np.nan
    if "tab" in s or "tablet" in s or "bottle" in s or "1/2" in s:
        return np.nan
    m = re.search(r"([-+]?\d+(\.\d+)?)", s)
    if not m:
        return np.nan
    return float(m.group(1))

def parse_tab_count(val) -> float:
    if pd.isna(val):
        return np.nan
    s = norm_text(val).lower()
    if "tab" not in s and "tablet" not in s:
        return np.nan
    m = re.search(r"([-+]?\d+(\.\d+)?)", s)
    if not m:
        return np.nan
    return float(m.group(1))

def as_numeric(series: pd.Series) -> pd.Series:
    s = series.apply(to_nan_if_bad)
    return pd.to_numeric(s, errors="coerce")

def _percent_formatter(x, pos):
    return f"{int(x)}"

# ---------------------------
# Label wrapping (Excel-like)
# ---------------------------
def _banddown_and(s: str) -> str:
    if not BANDDOWN_AND_TO_AMP:
        return s
    if len(s) < 18:
        return s
    m = re.search(r"\s+and\s+", s, flags=re.IGNORECASE)
    if not m:
        return s
    return s[:m.start()].rstrip() + "\n& " + s[m.end():].lstrip()

def wrap_label_excel_like(s: str, width: int = WRAP_WIDTH_DEFAULT) -> str:
    s = str(s)
    s = _banddown_and(s)
    parts = s.split("\n")
    wrapped = [textwrap.fill(p, width=width, break_long_words=False) for p in parts]
    return "\n".join(wrapped)

def _wrap_labels(labels, width=WRAP_WIDTH_DEFAULT):
    return [wrap_label_excel_like(x, width=width) for x in labels]

# ---------------------------
# Global aesthetic defaults
# ---------------------------
plt.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 900,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 15,
    "axes.labelsize": 12,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10,
    "axes.linewidth": 1.0,
})

# ---------------------------
# Premium palettes
# ---------------------------
PALETTES = {
    "overview": ["#0B1320", "#1D4ED8", "#22C55E", "#A855F7"],
    "demographics": ["#0B1320", "#0EA5E9", "#22C55E", "#F59E0B"],
    "exposure": ["#0B1320", "#F97316", "#EC4899", "#8B5CF6"],
    "symptoms": ["#0B1320", "#14B8A6", "#60A5FA", "#A855F7"],
    "vitals": ["#0B1320", "#10B981", "#06B6D4", "#3B82F6"],
    "labs": ["#0B1320", "#F59E0B", "#FB7185", "#8B5CF6"],
    "outcomes": ["#0B1320", "#22C55E", "#F59E0B", "#EF4444"],
    "treatment": ["#0B1320", "#60A5FA", "#34D399", "#F472B6"],
    "ultra_heat": ["#060A15", "#0B4F6C", "#01BAEF", "#20BF55", "#F6AE2D", "#F26419", "#A4508B"],
}

def make_cmap(stops: List[str], name: str) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(name, stops)

CMAPS = {k: make_cmap(v, f"cmap_{k}") for k, v in PALETTES.items()}

# ---------------------------
# Utilities
# ---------------------------
def safe_mkdir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def pretty_ax(ax):
    ax.grid(True, axis="y", alpha=0.18, linewidth=0.9)
    ax.set_axisbelow(True)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

def stamp(ax, fig_id: str):
    if not SHOW_FIG_ID_ON_FIGURE:
        return
    ax.text(
        0.01, 0.99, fig_id,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=12.5, fontweight="bold",
        color="#0B1320",
        path_effects=[pe.withStroke(linewidth=4, foreground="white", alpha=0.9)]
    )

def gradient_colors(values: np.ndarray, cmap: LinearSegmentedColormap) -> List:
    v = np.array(values, dtype=float)
    if np.isfinite(v).any():
        v = np.nan_to_num(v, nan=float(np.nanmin(v)))
    else:
        v = np.nan_to_num(v, nan=0.0)
    lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
    if math.isclose(lo, hi):
        return [cmap(0.75) for _ in v]
    norm = Normalize(vmin=lo, vmax=hi)
    return [cmap(norm(x)) for x in v]

def save_figure(fig, outdir: str, fname_base: str, dpi: int = 900):
    png = os.path.join(outdir, f"{fname_base}.png")
    pdf = os.path.join(outdir, f"{fname_base}.pdf")
    svg = os.path.join(outdir, f"{fname_base}.svg")
    fig.savefig(png, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")

def _add_bar_value_labels(ax, bars, fmt="{:,.0f}"):
    heights = [b.get_height() for b in bars]
    if not heights:
        return
    y_max = max(heights) if max(heights) > 0 else 1.0
    offset = 0.018 * y_max
    for b in bars:
        h = b.get_height()
        if np.isnan(h):
            continue
        ax.text(
            b.get_x() + b.get_width() / 2.0,
            h + offset,
            fmt.format(h),
            ha="center", va="bottom",
            fontsize=10,
            color="#0B1320",
            path_effects=[pe.withStroke(linewidth=3, foreground="white", alpha=0.85)]
        )
    ax.set_ylim(0, y_max * 1.10)

# ---------------------------
# Ordering helpers: keep Other/Unknown at rightmost
# ---------------------------
def reorder_special_last(labels: List[str], specials: List[str] = SPECIAL_LAST) -> List[str]:
    labels = list(labels)
    present_specials = [s for s in specials if s in labels]
    main = [x for x in labels if x not in present_specials]
    return main + present_specials

def reorder_series_special_last(counts: pd.Series, specials: List[str] = SPECIAL_LAST) -> pd.Series:
    counts = counts.copy()
    idx = counts.index.astype(str).tolist()
    present_specials = [s for s in specials if s in idx]
    main = counts.drop(index=[s for s in present_specials if s in counts.index], errors="ignore")
    main = main.sort_values(ascending=False)
    tail = counts.loc[present_specials] if present_specials else pd.Series(dtype=counts.dtype)
    out = pd.concat([main, tail])
    return out

def reorder_columns_special_last(df: pd.DataFrame, specials: List[str] = SPECIAL_LAST) -> pd.DataFrame:
    cols = df.columns.astype(str).tolist()
    cols2 = reorder_special_last(cols, specials=specials)
    return df.reindex(columns=cols2)

# ============================================================
# EXACT COUNT REPORT (matches poison_type_counts.py behavior)
# - raw exact counts (as in Excel)
# - normalized counts (casefold + whitespace normalized)
# Saves CSVs in outdir and prints top 50 to console
# ============================================================
def _series_nonempty_exact(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype="string")
    s = df[col].astype("string")
    # preserve exact text; just remove invisible NBSP so strip works reliably
    s = s.str.replace("\u00A0", " ", regex=False)
    mask = s.notna() & (s.str.strip() != "")
    return s[mask]

def _normalize_for_counting(s: pd.Series) -> pd.Series:
    s = s.astype("string")
    s = s.str.replace("\u00A0", " ", regex=False)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s.str.casefold()

def write_exact_count_report(df: pd.DataFrame, col: str, outdir: str, prefix: str, top_n: int = 50) -> None:
    s_raw = _series_nonempty_exact(df, col)
    if s_raw.empty:
        print(f"\n[{prefix}] Column '{col}' not found or has no non-empty values.")
        return

    raw_counts = s_raw.value_counts(dropna=False)
    s_norm = _normalize_for_counting(s_raw)
    norm_counts = s_norm.value_counts(dropna=False)

    print("\n=== TOTAL (non-empty) ===")
    print(f"Raw exact total:       {len(s_raw)}")
    print(f"Normalized total:      {len(s_norm)}")

    print(f"\n=== RAW EXACT COUNTS (top {top_n}) ===")
    raw_df = raw_counts.head(top_n).rename_axis("poison_type_raw").reset_index(name="count_raw_exact")
    print(raw_df.to_string(index=False))

    print(f"\n=== NORMALIZED COUNTS (top {top_n}) ===")
    norm_df = norm_counts.head(top_n).rename_axis("poison_type_normalized").reset_index(name="count_normalized")
    print(norm_df.to_string(index=False))

    # Save full CSVs
    raw_full = raw_counts.rename_axis("poison_type_raw").reset_index(name="count_raw_exact")
    norm_full = norm_counts.rename_axis("poison_type_normalized").reset_index(name="count_normalized")

    raw_full.to_csv(os.path.join(outdir, f"{prefix}__raw_exact_counts.csv"), index=False, encoding="utf-8-sig")
    norm_full.to_csv(os.path.join(outdir, f"{prefix}__normalized_counts.csv"), index=False, encoding="utf-8-sig")

# ---------------------------
# Data transforms
# ---------------------------
def collapse_poison_low_counts(df: pd.DataFrame, col: str, min_n: int = 100,
                               other_label: str = "Other", keep_unknown: str = "Unknown") -> pd.DataFrame:
    if col not in df.columns:
        return df
    s = df[col].apply(lambda v: norm_category_exact(v, unknown=keep_unknown))
    vc = s.value_counts(dropna=False)
    low = set(vc[vc < min_n].index.tolist())
    low.discard(keep_unknown)
    low.discard(other_label)
    df = df.copy()
    df[col] = s.where(~s.isin(low), other_label)
    return df

def apply_violin_filter(series: pd.Series, fig_id: str) -> pd.Series:
    s = series.copy()
    rule = VIOLIN_FILTERS.get(fig_id)
    if rule:
        if "max" in rule:
            s = s.where(s <= float(rule["max"]))
        if "between" in rule:
            lo, hi = rule["between"]
            s = s.where(s.between(float(lo), float(hi)))
        return s

    if fig_id in ROBUST_OUTLIER_FIGS:
        x = s.dropna()
        if x.empty:
            return s
        s = s.where(s >= 0)
        x = s.dropna()
        if x.empty:
            return s
        q1 = x.quantile(0.25)
        q3 = x.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            lo = x.quantile(0.005)
            hi = x.quantile(0.995)
        else:
            lo = q1 - 3.0 * iqr
            hi = q3 + 3.0 * iqr
        s = s.where(s.between(lo, hi))
    return s

def filter_hist_values(vals: np.ndarray, fig_id: str) -> np.ndarray:
    vals = vals[np.isfinite(vals)]
    if fig_id not in HIST_XLIM:
        return vals
    lo, hi = HIST_XLIM[fig_id]
    return vals[(vals >= lo) & (vals <= hi)]

# ---------------------------
# Plot builders
# ---------------------------
def plot_bar_gradient(counts: pd.Series, title: str, xlabel: str, ylabel: str,
                      cmap_key: str, fig_id: str, rotate: int = 35, wrap_width: int = 14,
                      drop_labels: Optional[set] = None):
    counts = counts.copy()
    counts.index = counts.index.astype(str)

    if drop_labels:
        counts = counts[~counts.index.isin(drop_labels)]

    counts = reorder_series_special_last(counts, specials=SPECIAL_LAST)

    labels_raw = counts.index.tolist()
    labels_disp = _wrap_labels(labels_raw, width=wrap_width)
    vals = counts.values.astype(float)

    cmap = CMAPS[cmap_key]
    cols = gradient_colors(vals, cmap)

    fig, ax = plt.subplots(figsize=(11.6, 5.8))
    x = np.arange(len(labels_disp))
    bars = ax.bar(x, vals, color=cols, edgecolor=(0, 0, 0, 0.12), linewidth=1.0)

    for b in bars:
        b.set_path_effects([pe.SimplePatchShadow(offset=(1.2, -1.2), alpha=0.18), pe.Normal()])

    _add_bar_value_labels(ax, bars, fmt="{:,.0f}")

    ax.set_title(title, pad=12, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_disp, rotation=rotate, ha="right")

    pretty_ax(ax)
    stamp(ax, fig_id)
    fig.tight_layout()
    return fig

def plot_stacked_100(df: pd.DataFrame, index_col: str, col_col: str, title: str,
                     cmap_key: str, fig_id: str):
    tab = pd.crosstab(df[index_col], df[col_col], normalize="index") * 100
    tab = tab.loc[tab.sum(axis=1).sort_values(ascending=False).index]
    tab = reorder_columns_special_last(tab, specials=SPECIAL_LAST)

    fig, ax = plt.subplots(figsize=(11.6, 5.9))
    cmap = CMAPS[cmap_key]
    colors = [cmap(x) for x in np.linspace(0.25, 0.9, tab.shape[1])]

    bottom = np.zeros(tab.shape[0])
    x = np.arange(tab.shape[0])

    for i, col in enumerate(tab.columns):
        vals = tab[col].values
        ax.bar(x, vals, bottom=bottom, color=colors[i],
               edgecolor=(0, 0, 0, 0.08), linewidth=0.8, label=str(col))
        bottom += vals

    ax.set_title(title, pad=12, fontweight="bold")
    ax.set_ylabel("Percent (%)")
    ax.set_ylim(0, 100)
    pretty_ax(ax)
    stamp(ax, fig_id)

    labels_disp = _wrap_labels(tab.index.astype(str).tolist(), width=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_disp, rotation=35, ha="right")

    if fig_id in LEGEND_BELOW_FIGS or fig_id in {"F47", "F48"}:
        ax.legend(frameon=False, ncols=min(4, tab.shape[1]),
                  loc="upper center", bbox_to_anchor=(0.5, -0.18))
        fig.subplots_adjust(bottom=0.28)
    else:
        ax.legend(frameon=False, ncols=min(4, tab.shape[1]), loc="upper right")

    return fig

def plot_violin_by_group(df: pd.DataFrame, y: str, group: str, title: str,
                         cmap_key: str, fig_id: str):
    """
    FIXED:
    - observed=True to avoid unused categorical groups
    - drop groups with <2 points (prevents matplotlib zero-size reduction / KDE failures)
    """
    sub = df[[y, group]].dropna()
    if sub.empty:
        return None

    med = sub.groupby(group, observed=True)[y].median().dropna().sort_values()
    groups = med.index.tolist()
    if not groups:
        return None

    data: List[np.ndarray] = []
    kept_groups: List[Any] = []
    for g in groups:
        arr = sub.loc[sub[group] == g, y].dropna().values
        if len(arr) >= 2:
            data.append(arr)
            kept_groups.append(g)

    if not data:
        return None

    meds = np.array([np.median(a) for a in data], dtype=float)

    cmap = CMAPS[cmap_key]
    cols = gradient_colors(meds, cmap)

    fig, ax = plt.subplots(figsize=(12.4, 6.2))
    parts = ax.violinplot(data, showmeans=False, showmedians=True, showextrema=False)

    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(cols[i])
        body.set_edgecolor((0, 0, 0, 0.18))
        body.set_linewidth(1.2)
        body.set_alpha(0.95)

    parts["cmedians"].set_color("#0B1320")
    parts["cmedians"].set_linewidth(2.2)

    rng = np.random.default_rng(7)
    for i, arr in enumerate(data, start=1):
        xj = rng.normal(i, 0.05, size=len(arr))
        ax.scatter(xj, arr, s=28, alpha=0.30, edgecolors="none", color="#0B1320")

    ax.set_xticks(range(1, len(kept_groups) + 1))
    ax.set_xticklabels(_wrap_labels([str(g) for g in kept_groups], width=14),
                       rotation=35, ha="right")
    ax.set_ylabel(y)
    ax.set_title(title, pad=12, fontweight="bold")
    pretty_ax(ax)
    stamp(ax, fig_id)
    fig.tight_layout()
    return fig

def plot_heatmap(table: pd.DataFrame, title: str, cmap_key: str, fig_id: str,
                 xlabel: str, ylabel: str, is_corr: bool = False):
    fig, ax = plt.subplots(figsize=(12.6, 6.8))

    if is_corr or fig_id == "F53":
        cmap = plt.get_cmap("RdBu_r")
        im = ax.imshow(table.values, aspect="auto", interpolation="nearest", cmap=cmap,
                       vmin=-1, vmax=1)
    else:
        if fig_id == "F45":
            cmap = CMAPS["ultra_heat"]
            im = ax.imshow(table.values, aspect="auto", interpolation="bilinear", cmap=cmap)
        else:
            cmap = CMAPS[cmap_key]
            im = ax.imshow(table.values, aspect="auto", interpolation="nearest", cmap=cmap)

    ax.set_xticks(np.arange(table.shape[1]))
    ax.set_xticklabels(_wrap_labels(table.columns.astype(str).tolist(), width=12),
                       rotation=35, ha="right")
    ax.set_yticks(np.arange(table.shape[0]))
    ax.set_yticklabels(_wrap_labels(table.index.astype(str).tolist(), width=18))
    ax.set_title(title, pad=14, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.outline.set_visible(False)

    stamp(ax, fig_id)
    fig.tight_layout()
    return fig

# ---------------------------
# Poison type by site 100% stacked (keep Other/Unknown last)
# ---------------------------
def plot_poison_type_by_site_100pct_premium(
    df: pd.DataFrame,
    site_col: str,
    poison_col: str,
    title: str,
    fig_id: str,
    cmap_name: str = "magma",
):
    sub = df[[site_col, poison_col]].copy()
    sub[site_col] = sub[site_col].apply(lambda v: norm_category_exact(v, unknown="Unknown site"))
    sub[poison_col] = sub[poison_col].apply(lambda v: norm_category_exact(v, unknown=POISON_UNKNOWN_LABEL))
    sub = sub.dropna(subset=[site_col, poison_col])
    if sub.empty:
        return None

    tab = sub.pivot_table(index=site_col, columns=poison_col, aggfunc="size", fill_value=0, observed=True)
    poison_order = tab.sum(axis=0).sort_values(ascending=False).index.astype(str).tolist()
    poison_order = reorder_special_last(poison_order, specials=SPECIAL_LAST)
    tab = tab.reindex(columns=poison_order, fill_value=0)

    pct = tab.div(tab.sum(axis=1), axis=0) * 100
    sites = pct.index.tolist()
    x = np.arange(len(sites))

    cmap = plt.get_cmap(cmap_name)
    cols = [cmap(v) for v in np.linspace(0.15, 0.92, max(len(poison_order), 1))]

    fig, ax = plt.subplots(figsize=(13.8, 7.8))
    bottom = np.zeros(len(sites))

    for i, p in enumerate(poison_order):
        vals = pct[p].values
        ax.bar(
            x, vals,
            bottom=bottom,
            width=0.78,
            color=cols[i],
            edgecolor=(0, 0, 0, 0.08),
            linewidth=0.5,
            label=p,
            zorder=3,
        )
        bottom += vals

    title_pad = 42 if fig_id == "F10" else 14
    ax.set_title(title, fontsize=20, fontweight="bold", pad=title_pad)
    stamp(ax, fig_id)

    ax.set_xticks(x)
    ax.set_xticklabels(_wrap_labels(sites, width=8), rotation=35, ha="right", fontsize=12)

    ax.set_ylim(0, 100)
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_major_formatter(FuncFormatter(_percent_formatter))
    ax.set_ylabel("Percent (%)", fontsize=13)
    ax.grid(axis="y", alpha=0.18, linewidth=1.0, zorder=0)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    n_site = tab.sum(axis=1).reindex(sites).values
    for i, n in enumerate(n_site):
        ax.text(i, 101.2, f"n={int(n)}", ha="center", va="bottom", fontsize=10, color=(0, 0, 0, 0.75))

    ncol = min(6, max(3, int(math.ceil(len(poison_order) / 3))))
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=ncol,
        frameon=False,
        fontsize=10,
        title="Poisoning type",
        title_fontsize=11,
        handlelength=1.4,
        columnspacing=1.2,
    )

    if fig_id == "F10":
        fig.subplots_adjust(bottom=0.28, top=0.82)
    else:
        fig.subplots_adjust(bottom=0.28)

    return fig

def plot_outcome_by_poison_type_100pct_premium(
    df: pd.DataFrame,
    poison_col: str,
    outcome_col: str,
    title: str,
    fig_id: str,
):
    sub = df[[poison_col, outcome_col]].copy()
    sub[poison_col] = sub[poison_col].apply(lambda v: norm_category_exact(v, unknown=POISON_UNKNOWN_LABEL))
    sub[outcome_col] = sub[outcome_col].apply(lambda v: norm_category_exact(v, unknown=OUTCOME_UNKNOWN_LABEL))
    sub = sub.dropna(subset=[poison_col, outcome_col])
    if sub.empty:
        return None

    n_by_poison = sub.groupby(poison_col, observed=True).size().sort_values(ascending=False)
    poison_order = n_by_poison.index.astype(str).tolist()
    poison_order = reorder_special_last(poison_order, specials=SPECIAL_LAST)

    tab = (
        sub.pivot_table(index=poison_col, columns=outcome_col, aggfunc="size", fill_value=0, observed=True)
        .reindex(poison_order)
    )
    pct = tab.div(tab.sum(axis=1), axis=0) * 100

    outcome_order = [
        OUTCOME_ABSCONDED_LABEL,
        OUTCOME_DEATH_LABEL,
        OUTCOME_SURV_NO_COMP,
        OUTCOME_SURV_COMP,
        OUTCOME_UNKNOWN_LABEL,
    ]
    for c in outcome_order:
        if c not in pct.columns:
            pct[c] = 0.0
    pct = pct[outcome_order]

    outcome_colors = {
        OUTCOME_ABSCONDED_LABEL: "#1f9d55",
        OUTCOME_DEATH_LABEL: "#7cb342",
        OUTCOME_SURV_NO_COMP: "#f39c12",
        OUTCOME_SURV_COMP: "#e74c3c",
        OUTCOME_UNKNOWN_LABEL: "#95a5a6",
    }

    labels_y = [f"{p}  (n={int(n_by_poison[p])})" for p in pct.index]
    labels_y = _wrap_labels(labels_y, width=26)

    fig_h = max(6.0, 0.33 * len(labels_y) + 2.2)
    fig, ax = plt.subplots(figsize=(12.8, fig_h))

    left = np.zeros(len(pct))
    y = np.arange(len(pct))

    for col in pct.columns:
        vals = pct[col].values
        ax.barh(
            y, vals, left=left,
            color=outcome_colors.get(col, "#bdc3c7"),
            edgecolor=(0, 0, 0, 0.10),
            linewidth=0.6,
            height=0.78,
            label=col,
            zorder=3,
        )
        left += vals

    ax.set_title(title, fontsize=20, fontweight="bold", pad=16)
    stamp(ax, fig_id)

    ax.set_yticks(y)
    ax.set_yticklabels(labels_y, fontsize=10)
    ax.invert_yaxis()

    ax.set_xlim(0, 100)
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.xaxis.set_major_formatter(FuncFormatter(_percent_formatter))
    ax.set_xlabel("Percent (%)", fontsize=13)
    ax.grid(axis="x", alpha=0.18, linewidth=1.0, zorder=0)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10),
              ncol=3, frameon=False, fontsize=11)
    fig.subplots_adjust(bottom=0.18)
    fig.tight_layout()
    return fig

# ---------------------------
# Main pipeline
# ---------------------------
PII_COLUMNS = {
    "Patient's Name", "Patient’s Name", "Contact number", "Contact no", "Registration Number"
}

SYMPTOM_CANDIDATES = [
    "Vomited after ingestion", "Fever", "Vomiting", "Diarrhoea", "Abdominal pain",
    "Abdominal distension", "Cough", "Shortness of breath", "Heart burn", "Oral ulcers",
    "Leg swelling", "Reduced urine output", "Unconsciousness", "Convulsion", "Chest pain",
    "Bleeding tendency", "Shock"
]

VITAL_CANDIDATES = {
    "Temperature": "Temperature",
    "Pulse (beats/min)": "Pulse",
    "SpO2": "SpO₂",
    "GCS": "GCS",
    "Respiratory rate": "Respiratory rate"
}

LAB_CANDIDATES = [
    "Total WBC count(/mm3)", "Neutrophil (%)", "Lymphocytes(%)", "Platelates(/mm3)",
    "S. creatinine(mg/dL)", "Na+", "k+", "Cl-", "Ca2+", "Mg2+",
    "S.uric acid(mg/dL)", "Random blood sugar (mmol/L)",
    "pH", "HCO3-", "PCO2", "PO2", "Anion gap",
    "S.bilirubin(mg/dL)", "S. amylase(u/L)", "SGPT", "SGOT",
]

def load_data(path: str, sheet: Optional[str] = None) -> pd.DataFrame:
    if path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path, sheet_name=sheet if sheet else 0)
    else:
        df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df

def clean_core(df: pd.DataFrame) -> pd.DataFrame:
    """
    Core cleaning.
    Key rule for exposure fields:
    - Keep blanks as blanks (unknown="") so placeholders don't pollute mapping.
    - Actual strings like "Unknown" remain as "Unknown" (not forced).
    """
    df = df.copy()

    for c in list(df.columns):
        if c in PII_COLUMNS:
            df.drop(columns=[c], inplace=True)

    # Normal categorical cleaning for non-exposure fields
    for c in ["Study site", "Sex", "Living area", "Presentation area", "Occupation"]:
        if c in df.columns:
            df[c] = df[c].apply(lambda v: norm_category_exact(v, unknown="Unknown"))

    # Exposure fields: keep blanks (unknown="") so we don't pollute mapping
    for c in ["Types of poisoning", "Name of the specific component"]:
        if c in df.columns:
            df[c] = df[c].apply(lambda v: norm_category_exact(v, unknown=""))

    for c in ["Date of admission", "Date of ingestion"]:
        if c in df.columns:
            df[c] = parse_date_series(df[c])

    if "Age (in years)" in df.columns:
        df["Age_years"] = as_numeric(df["Age (in years)"])

    if "Amount of ingestion (ml)" in df.columns:
        df["Ingest_ml"] = df["Amount of ingestion (ml)"].apply(parse_amount_ml)
        df["Ingest_tabs"] = df["Amount of ingestion (ml)"].apply(parse_tab_count)

    for c in ["Time to symptoms onset (in hrs)", "Time to presentation (in hrs)"]:
        if c in df.columns:
            df[c + "_hours"] = df[c].apply(parse_duration_to_hours)

    if "Blood pressure" in df.columns:
        sys, dia = parse_bp(df["Blood pressure"])
        df["SBP"] = sys
        df["DBP"] = dia

    for raw, nice in VITAL_CANDIDATES.items():
        if raw in df.columns:
            df[nice] = as_numeric(df[raw])

    for raw in LAB_CANDIDATES:
        if raw in df.columns:
            df[raw] = as_numeric(df[raw])

    for s in SYMPTOM_CANDIDATES:
        if s in df.columns:
            df[s] = df[s].apply(norm_text).str.title()
            df.loc[~df[s].isin(["Yes", "No"]), s] = np.nan

    for c in ["Survived without complications", "Survived with complications", "Death", "Absconded"]:
        if c in df.columns:
            df[c] = df[c].apply(norm_text).str.title()
            df.loc[~df[c].isin(["Yes", "No"]), c] = np.nan

    return df

# ============================================================
# 16-category poisoning scheme (STRICT, Excel list-aligned)
# IMPORTANT FIXES:
# 1) Category names match your required 16 labels exactly
# 2) Mapping includes exact Excel spellings (kerosine, pyerithroid, etc.)
# 3) Priority is: TYPE first, COMPONENT only as fallback
# ============================================================

POISON_16_CATEGORIES = [
    "OPC",
    "Non-OPC",
    "Insecticide",
    "Herbicide",
    "Fungicide",
    "Rodenticide",
    "Aluminium phosphide",
    "Drug overdose",
    "Alcohol overdose",
    "Household products",
    "Corrosive ingestion",
    "Chemical Inhalation",
    "Bites & stings",
    "Commuter Poisoning",
    "Other",
    "Unknown",
]

UNKNOWN_TOKENS = {
    "unknown", "unknown poisoning", "unknown poison", "unknown bite", "unknown exposure",
    "unknown/other", "other/unknown",
    "na", "n/a", "null", "none", ""
}

def norm_key_for_lookup(x) -> str:
    """Normalize text so Excel-style labels match reliably."""
    if pd.isna(x):
        return ""
    s = str(x).strip().casefold()
    s = s.replace("&", " and ")
    s = s.replace("\u00A0", " ")
    s = s.replace("–", "-").replace("—", "-").replace("−", "-")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def is_unknown_like(x) -> bool:
    return norm_key_for_lookup(x) in UNKNOWN_TOKENS

def build_poison16_map() -> Dict[str, Tuple[str, str]]:
    """
    Returns: dict[norm_key] = (category, subcategory_display)
    """
    m: Dict[str, Tuple[str, str]] = {}

    def add(category: str, subcat: str, *labels: str):
        for lab in labels:
            m[norm_key_for_lookup(lab)] = (category, subcat)

    # 1) OPC
    add("OPC", "OPC", "OPC")
    add("OPC", "Mixed OPC", "Mixed OPC")

    # 2) Non-OPC
    add("Non-OPC", "Non OPC", "Non OPC")
    add("Non-OPC", "Carbamate", "Carbamate")
    add("Non-OPC", "Pesticide", "Pesticide")

    # 3) Insecticide
    add("Insecticide", "Insecticide", "Insecticide")
    add("Insecticide", "Cypermethrin", "Cypermethrin")
    add("Insecticide", "Lambda Cyhalothrin", "Lambda Cyhalothrin")
    # Excel spelling included:
    add("Insecticide", "Pyrethroid", "Pyrethroid", "Pyerithroid")
    add("Insecticide", "Cockroach Killer", "Cockroach Killer")
    add("Insecticide", "Ant Killer", "Ant Killer")
    add("Insecticide", "Mosquito Killer", "Mosquito Killer")
    add("Insecticide", "Lice killer", "Lice killer", "Lice Killer")

    # 4) Herbicide
    add("Herbicide", "Paraquat", "Paraquat")
    add("Herbicide", "Herbicide", "Herbicide")

    # 5) Fungicide
    # Fungicide should be counted as Other from now on
    add("Other", "Fungicide", "Fungicide")


    # 6) Rodenticide
    add("Rodenticide", "Rat killer", "Rat killer", "Rat Killer")

    # 7) Aluminium phosphide
    add("Aluminium phosphide", "Aluminium Phosphide", "Aluminium Phosphide")
    add("Aluminium phosphide", "Gas Tablet", "Gas Tablet")
    add("Aluminium phosphide", "Karri", "Karri")

    # 8) Drug overdose
    add("Drug overdose", "Sedative", "Sedative")
    add("Drug overdose", "Benzodiazepine", "Benzodiazepine")
    add("Drug overdose", "Clonazepam", "Clonazepam")
    add("Drug overdose", "Drug Overdose", "Drug Overdose")
    add("Drug overdose", "Multidrug", "Multidrug")
    add("Drug overdose", "Paracetamol", "Paracetamol")
    add("Drug overdose", "TCA", "TCA")
    add("Drug overdose", "Tablet", "Tablet")

    # 9) Alcohol overdose
    add("Alcohol overdose", "Alcohol Overdose", "Alcohol Overdose")

    # Methanol should be counted as Other (not alcohol overdose)
    add("Other", "Methanol", "Methanol")


    # 10) Household products
    add("Household products", "Harpic", "Harpic")
    add("Household products", "Household product", "Household product", "Household Product")
    add("Household products", "Savlon", "Savlon")
    add("Household products", "Dettol", "Dettol")
    add("Household products", "Vixol", "Vixol")
    add("Household products", "Hexisol", "Hexisol")
    add("Household products", "Wheel Powder", "Wheel Powder")

    # 11) Corrosive ingestion
    # Corrosive should be counted as Household products from now on
    add("Household products", "Corrosive", "Corrosive")
    add("Household products", "Acid Ingestion", "Acid Ingestion")


    # 12) Chemical Inhalation
    add("Chemical Inhalation", "Chemical", "Chemical")
    # Excel spelling included:
    add("Chemical Inhalation", "kerosine", "kerosine", "Kerosine", "Kerosene")
    add("Chemical Inhalation", "Diesel", "Diesel")
    add("Chemical Inhalation", "Tarpine", "Tarpine")
    add("Chemical Inhalation", "Inhalation", "Inhalation")

    # 13) Bites & stings
    add("Bites & stings", "Bee Sting", "Bee Sting")
    add("Bites & stings", "Insect Bite", "Insect Bite")

    # 14) Commuter Poisoning
    add("Commuter Poisoning", "Street Poisoning", "Street Poisoning")
    add("Commuter Poisoning", "Commuter", "Commuter")

    # 15) Other
    add("Other", "Other", "Other")
    add("Other", "Homicidal", "Homicidal")

    # 16) Unknown
    # Unknown should be counted as Other from now on
    add("Other", "Unknown", "Unknown")


    return m

POISON16_MAP = build_poison16_map()

def map_value_to_poison16(val: Any) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (category, subcategory) or (None, None) if unmapped or unknown-like.
    """
    if pd.isna(val):
        return None, None
    k = norm_key_for_lookup(val)
    if k in UNKNOWN_TOKENS:
        return None, None
    return POISON16_MAP.get(k, (None, None))

def apply_poison_16_scheme(
    df: pd.DataFrame,
    type_col: str = "Types of poisoning",
    component_col: str = "Name of the specific component"
) -> pd.DataFrame:
    """
    Creates the fixed 16-category Types of poisoning used across all plots.

    CRITICAL FIX:
    - TYPE is used first (to match Excel's 'Types of poisoning' distribution)
    - COMPONENT is only a fallback when TYPE is blank/unknown/unmapped
    """
    df = df.copy()

    df[f"{type_col}__raw"] = df[type_col] if type_col in df.columns else ""
    df[f"{component_col}__raw"] = df[component_col] if component_col in df.columns else ""

    cats: List[str] = []
    subs: List[str] = []
    srcs: List[str] = []

    for _, row in df.iterrows():
        raw_type = row.get(f"{type_col}__raw", "")
        raw_comp = row.get(f"{component_col}__raw", "")

        t_cat, t_sub = map_value_to_poison16(raw_type)
        c_cat, c_sub = map_value_to_poison16(raw_comp)

        # ✅ 1) Use TYPE first
        if t_cat is not None:
            cats.append(t_cat); subs.append(t_sub or ""); srcs.append("type")
            continue

        # ✅ 2) If TYPE is blank/unknown/unmapped, fallback to COMPONENT
        if c_cat is not None:
            cats.append(c_cat); subs.append(c_sub or ""); srcs.append("component_fallback")
            continue

        # 3) Nothing mapped -> Other (Unknown is counted as Other from now on)
        if is_unknown_like(raw_type) and is_unknown_like(raw_comp):
            cats.append("Other"); subs.append("Unknown"); srcs.append("none")
        elif norm_key_for_lookup(raw_type) == "" and norm_key_for_lookup(raw_comp) == "":
            cats.append("Other"); subs.append("Unknown"); srcs.append("none")
        else:
            cats.append("Other"); subs.append("Other"); srcs.append("unmapped")


    df[type_col] = pd.Categorical(cats, categories=POISON_16_CATEGORIES, ordered=True)
    df[type_col] = df[type_col].cat.remove_unused_categories()
    df["Poison16_subcategory"] = subs
    df["Poison16_source"] = srcs
    return df

@dataclass
class FigRec:
    fig_id: str
    fname: str
    title: str
    caption: str

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Excel/CSV dataset path")
    ap.add_argument("--sheet", default=None, help="Excel sheet name (optional)")
    ap.add_argument("--outdir", default="poisoning_plots_ultramax_v7", help="Output directory")
    ap.add_argument("--only", default=None, help="Comma-separated figure IDs to generate, e.g. F38 or F14,F38")
    ap.add_argument("--skip", default=None, help="Comma-separated figure IDs to skip, e.g. F01,F02")
    args = ap.parse_args()

    outdir = args.outdir
    safe_mkdir(outdir)
    perfig = os.path.join(outdir, "figures")
    safe_mkdir(perfig)

    df_raw = load_data(args.input, args.sheet)
    # --- EXACT RAW COUNTS (matches your poison_type_counts.py) ---
    write_exact_count_report(
        df_raw,
        col="Types of poisoning",
        outdir=outdir,
        prefix="poison_type_counts__BEFORE_any_cleaning",
        top_n=50
    )

    write_exact_count_report(
        df_raw,
        col="Name of the specific component",
        outdir=outdir,
        prefix="component_counts__BEFORE_any_cleaning",
        top_n=50
    )

    df = clean_core(df_raw)

    # ✅ Apply Excel-aligned fixed 16-category poisoning scheme
    df = apply_poison_16_scheme(
        df,
        type_col="Types of poisoning",
        component_col="Name of the specific component"
    )
    # --- VERIFY: export 16-category counts + row-level audit ---
    counts16 = (
        df["Types of poisoning"]
        .value_counts(dropna=False)
        .reindex(POISON_16_CATEGORIES, fill_value=0)
    )
    counts16.to_csv(os.path.join(outdir, "poison16_category_counts.csv"), header=["count"])

    df[[
        "Types of poisoning__raw",
        "Name of the specific component__raw",
        "Types of poisoning",
        "Poison16_subcategory",
        "Poison16_source"
    ]].to_csv(os.path.join(outdir, "poison16_row_audit.csv"), index=False, encoding="utf-8-sig")

    df.to_csv(os.path.join(outdir, "cleaned_snapshot_no_pii.csv"), index=False)

    figs: List[FigRec] = []
    pdf_path = os.path.join(outdir, "combined_plots.pdf")
    pdf = PdfPages(pdf_path)

    def add(fig, rec: FigRec):
        if fig is None:
            return
        if only_set is not None and rec.fig_id not in only_set:
            plt.close(fig)
            return
        if rec.fig_id in skip_set:
            plt.close(fig)
            return

        fname_base = f"{rec.fig_id}_{rec.fname}"
        save_figure(fig, perfig, fname_base, dpi=900)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        figs.append(rec)

    fig_counter = 1
    def next_id():
        nonlocal fig_counter
        fid = f"F{fig_counter:02d}"
        fig_counter += 1
        return fid

    only_set = None
    skip_set = set()

    if args.only:
        only_set = {x.strip().upper() for x in args.only.split(",") if x.strip()}
    if args.skip:
        skip_set = {x.strip().upper() for x in args.skip.split(",") if x.strip()}

    # ---------------------------
    # OVERVIEW
    # ---------------------------
    if "Study site" in df.columns:
        vc = df["Study site"].value_counts(dropna=True)
        fid = next_id()
        fig = plot_bar_gradient(vc, "Case count by study site", "Study site", "Cases",
                                "overview", fid, rotate=25, wrap_width=10)
        add(fig, FigRec(fid, "cases_by_site", "Case count by study site",
                        "Number of admissions recorded per study site."))

    if "Date of admission" in df.columns and df["Date of admission"].notna().any():
        daily = df.dropna(subset=["Date of admission"]).groupby("Date of admission").size().sort_index()
        fid = next_id()
        fig, ax = plt.subplots(figsize=(12.2, 5.6))
        x = daily.index
        y = daily.values
        cols = gradient_colors(y, CMAPS["overview"])
        ax.plot(x, y, linewidth=2.2, color="#0B1320", alpha=0.85)
        ax.scatter(x, y, s=55, c=cols, edgecolors=(0, 0, 0, 0.18), linewidths=0.8)
        ax.set_title("Admissions over time", pad=12, fontweight="bold")
        ax.set_xlabel("Date of admission")
        ax.set_ylabel("Cases")
        pretty_ax(ax)
        stamp(ax, fid)
        # 1-month gap ticks on x-axis
        loc = mdates.MonthLocator(interval=1)
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # Mar, Apr, ... , Jan
        fig.autofmt_xdate(rotation=25)
        add(fig, FigRec(fid, "admissions_over_time", "Admissions over time",
                        "Daily admissions count across the study period."))

    for c, key in [("Living area", "overview"), ("Presentation area", "overview")]:
        if c in df.columns:
            vc = df[c].dropna().value_counts().head(8)
            if len(vc) > 0:
                fid = next_id()
                fig = plot_bar_gradient(vc, f"{c} distribution", c, "Cases",
                                        key, fid, rotate=25, wrap_width=14)
                add(fig, FigRec(fid, f"{c.lower().replace(' ', '_')}_distribution",
                                f"{c} distribution", f"Distribution of {c.lower()} among admissions."))

    if "Age_years" in df.columns and df["Age_years"].notna().any():
        fid = next_id()
        fig, ax = plt.subplots(figsize=(11.8, 5.8))
        ages = df["Age_years"].dropna().values
        ax.hist(ages, bins=18, edgecolor=(0, 0, 0, 0.12), color=CMAPS["demographics"](0.72))
        ax.set_title("Age distribution", pad=12, fontweight="bold")
        ax.set_xlabel("Age (years)")
        ax.set_ylabel("Count")
        pretty_ax(ax)
        stamp(ax, fid)
        add(fig, FigRec(fid, "age_distribution", "Age distribution",
                        "Distribution of patient age (years)."))

        if "Study site" in df.columns:
            fid = next_id()
            fig = plot_violin_by_group(df, "Age_years", "Study site",
                                       "Age by study site", "demographics", fid)
            add(fig, FigRec(fid, "age_by_site", "Age by study site",
                            "Age distribution stratified by study site; sites are ordered by median age."))

    # ---------------------------
    # F07: Sex by site (legend below)
    # ---------------------------
    if "Sex" in df.columns and "Study site" in df.columns:
        sub = df.dropna(subset=["Sex", "Study site"])
        if not sub.empty:
            fid = next_id()
            fig = plot_stacked_100(sub, "Study site", "Sex",
                                   "Sex distribution by study site (100% stacked)",
                                   "demographics", fid)
            add(fig, FigRec(fid, "sex_by_site_100pct", "Sex distribution by study site",
                            "Proportion of sex categories within each site."))

    # ---------------------------
    # F08: Occupation (wrapped)
    # ---------------------------
    if "Occupation" in df.columns:
        vc = df["Occupation"].dropna().value_counts().head(10)
        if len(vc) > 0:
            fid = next_id()
            fig = plot_bar_gradient(vc, "Occupation (Top categories)", "Occupation", "Cases",
                                    "demographics", fid, rotate=35, wrap_width=18)
            add(fig, FigRec(fid, "occupation_top", "Occupation distribution",
                            "Top occupation categories among admissions."))

    # ---------------------------
    # Exposure: poison type overall + by site
    # ---------------------------
    if "Types of poisoning" in df.columns:
        vc = df["Types of poisoning"].value_counts(dropna=True)
        vc = vc[vc > 0]

        fid = next_id()
        fig = plot_bar_gradient(vc, "Types of poisoning (overall)", "Type", "Cases",
                                "exposure", fid, rotate=35, wrap_width=18)
        add(fig, FigRec(fid, "poison_type_overall", "Types of poisoning",
                        "Overall distribution of poisoning categories (fixed 16-category scheme)."))

        if "Study site" in df.columns:
            sub = df.dropna(subset=["Study site", "Types of poisoning"])
            if not sub.empty:
                fid = next_id()
                fig = plot_poison_type_by_site_100pct_premium(
                    sub,
                    site_col="Study site",
                    poison_col="Types of poisoning",
                    title="Poisoning type distribution by study site (100% stacked)",
                    fig_id=fid,
                    cmap_name="magma",
                )
                add(fig, FigRec(fid, "poison_type_by_site_100pct", "Poison type by site",
                                "Within-site composition of poisoning categories (fixed scheme)."))

    # ---------------------------
    # F11: Specific component (Top) — remove Unknown/Other
    # ---------------------------
    if "Name of the specific component" in df.columns:
        vc = df["Name of the specific component"].dropna().value_counts().head(15)
        if len(vc) > 0:
            fid = next_id()
            drop = F11_DROP_LABELS if fid == "F11" else None
            fig = plot_bar_gradient(vc, "Specific component (Top)", "Component", "Cases",
                                    "exposure", fid, rotate=35, wrap_width=18,
                                    drop_labels=drop)
            add(fig, FigRec(fid, "specific_component_top", "Specific component",
                            "Most frequently recorded specific components (Unknown/Other removed for F11)."))

    # Ingestion ml / tabs
    if "Ingest_ml" in df.columns and df["Ingest_ml"].notna().sum() >= 8 and "Types of poisoning" in df.columns:
        fid = next_id()
        df2 = df.copy()
        df2["Ingest_ml"] = apply_violin_filter(df2["Ingest_ml"], fid)
        fig = plot_violin_by_group(df2, "Ingest_ml", "Types of poisoning",
                                   "Amount ingested (mL) by poisoning type",
                                   "exposure", fid)
        add(fig, FigRec(fid, "ingest_ml_by_type", "Ingested amount (mL)",
                        "Distribution of ingested volume in mL (numeric entries only), stratified by poisoning type."))

    if "Ingest_tabs" in df.columns and df["Ingest_tabs"].notna().sum() >= 8 and "Types of poisoning" in df.columns:
        fid = next_id()
        df2 = df.copy()
        df2["Ingest_tabs"] = apply_violin_filter(df2["Ingest_tabs"], fid)
        fig = plot_violin_by_group(df2, "Ingest_tabs", "Types of poisoning",
                                   "Tablet count by poisoning type",
                                   "exposure", fid)
        add(fig, FigRec(fid, "ingest_tabs_by_type", "Tablet count",
                        "Distribution of tablet counts among types (figure-specific filtering applied)."))

    # Time distributions (hist + violin)
    for raw, nice in [
        ("Time to symptoms onset (in hrs)_hours", "Time to symptom onset (hours)"),
        ("Time to presentation (in hrs)_hours", "Time to presentation (hours)")
    ]:
        if raw in df.columns and df[raw].notna().sum() >= 8:
            fid = next_id()
            fig, ax = plt.subplots(figsize=(11.8, 5.8))
            vals = filter_hist_values(df[raw].dropna().values, fid)
            ax.hist(vals, bins=16, edgecolor=(0, 0, 0, 0.12), color=CMAPS["exposure"](0.72))
            ax.set_title(nice, pad=12, fontweight="bold")
            ax.set_xlabel("Hours")
            ax.set_ylabel("Count")
            if fid in HIST_XLIM:
                ax.set_xlim(*HIST_XLIM[fid])
            pretty_ax(ax)
            stamp(ax, fid)
            add(fig, FigRec(fid, raw.replace(" ", "_").lower(), nice,
                            f"Distribution of {nice.lower()} (restricted to requested x-axis range)."))

            if "Types of poisoning" in df.columns:
                fid = next_id()
                df2 = df.copy()
                df2[raw] = apply_violin_filter(df2[raw], fid)
                fig = plot_violin_by_group(df2, raw, "Types of poisoning",
                                           f"{nice} by poisoning type", "exposure", fid)
                add(fig, FigRec(fid, f"{raw}_by_type".replace(" ", "_").lower(),
                                f"{nice} by poisoning type",
                                f"{nice} stratified by poisoning type."))

    # Symptoms
    available_symptoms = [s for s in SYMPTOM_CANDIDATES if s in df.columns]
    if available_symptoms:
        sym_counts = {s: (df[s].astype(str).str.title() == "Yes").sum() for s in available_symptoms}
        sym_ser = pd.Series(sym_counts).sort_values(ascending=False)
        fid = next_id()
        fig = plot_bar_gradient(sym_ser, "Symptom prevalence (count of 'Yes')", "Symptom", "Yes count",
                                "symptoms", fid, rotate=35, wrap_width=18)
        add(fig, FigRec(fid, "symptoms_yes_counts", "Symptom prevalence",
                        "Count of 'Yes' for each symptom field."))

        if "Types of poisoning" in df.columns:
            sub = df.dropna(subset=["Types of poisoning"])
            if not sub.empty:
                mat, idx = [], []
                for s in available_symptoms:
                    tmp = sub.groupby("Types of poisoning", observed=True)[s].apply(lambda x: (x == "Yes").mean() * 100)
                    mat.append(tmp)
                    idx.append(s)
                table = pd.DataFrame(mat, index=idx).fillna(0.0)
                table = table.loc[:, table.mean(axis=0).sort_values(ascending=False).index]
                fid = next_id()
                fig = plot_heatmap(table, "Symptom prevalence by poisoning type (%)",
                                   "symptoms", fid, xlabel="Poisoning type", ylabel="Symptom")
                add(fig, FigRec(fid, "symptom_heatmap_by_poison_type", "Symptoms by poisoning type",
                                "Heatmap showing symptom prevalence (%) across poisoning types."))

        if "Study site" in df.columns:
            sub = df.dropna(subset=["Study site"])
            if not sub.empty:
                mat, idx = [], []
                for s in available_symptoms:
                    tmp = sub.groupby("Study site")[s].apply(lambda x: (x == "Yes").mean() * 100)
                    mat.append(tmp)
                    idx.append(s)
                table = pd.DataFrame(mat, index=idx).fillna(0.0)
                table = table.loc[:, table.mean(axis=0).sort_values(ascending=False).index]
                fid = next_id()
                fig = plot_heatmap(table, "Symptom prevalence by study site (%)",
                                   "symptoms", fid, xlabel="Study site", ylabel="Symptom")
                add(fig, FigRec(fid, "symptom_heatmap_by_site", "Symptoms by study site",
                                "Heatmap showing symptom prevalence (%) across study sites."))

    # Vitals & Exam
    for nice in ["Temperature", "Pulse", "SpO₂", "Respiratory rate", "GCS", "SBP", "DBP"]:
        if nice in df.columns and df[nice].notna().sum() >= 8:
            fid = next_id()
            fig, ax = plt.subplots(figsize=(11.8, 5.8))
            vals = filter_hist_values(df[nice].dropna().values, fid)
            ax.hist(vals, bins=16, edgecolor=(0, 0, 0, 0.12), color=CMAPS["vitals"](0.72))
            ax.set_title(f"{nice} distribution", pad=12, fontweight="bold")
            ax.set_xlabel(nice)
            ax.set_ylabel("Count")
            if fid in HIST_XLIM:
                ax.set_xlim(*HIST_XLIM[fid])
            pretty_ax(ax)
            stamp(ax, fid)
            add(fig, FigRec(fid, f"{nice.lower().replace(' ', '_')}_distribution",
                            f"{nice} distribution", "Distribution restricted to requested x-axis range."))

            if "Types of poisoning" in df.columns:
                fid = next_id()
                df2 = df.copy()
                df2[nice] = apply_violin_filter(df2[nice], fid)
                fig = plot_violin_by_group(df2, nice, "Types of poisoning",
                                           f"{nice} by poisoning type", "vitals", fid)
                add(fig, FigRec(fid, f"{nice.lower().replace(' ', '_')}_by_type",
                                f"{nice} by poisoning type", "Figure-specific outlier rules applied."))

    # Labs
    lab_cols = [c for c in LAB_CANDIDATES if c in df.columns and df[c].notna().sum() >= 8]
    if lab_cols and "Types of poisoning" in df.columns:
        for c in lab_cols:
            fid = next_id()
            df2 = df.copy()
            df2[c] = apply_violin_filter(df2[c], fid)
            fig = plot_violin_by_group(df2, c, "Types of poisoning", f"{c} by poisoning type",
                                       "labs", fid)
            add(fig, FigRec(fid, f"lab_{re.sub(r'[^a-z0-9]+','_',c.lower()).strip('_')}_by_type",
                            f"{c} by poisoning type", "Figure-specific outlier rules applied."))

        # Correlation matrix: F53 journal-standard palette
        num_cols = []
        for c in ["Age_years", "Temperature", "Pulse", "SpO₂", "Respiratory rate", "GCS", "SBP", "DBP"] + lab_cols:
            if c in df.columns and df[c].notna().sum() >= 8:
                num_cols.append(c)
        if len(num_cols) >= 6:
            corr = df[num_cols].corr(numeric_only=True).fillna(0.0)
            fid = next_id()
            fig = plot_heatmap(corr, "Correlation matrix (vitals + labs)", "labs", fid,
                               xlabel="Variable", ylabel="Variable", is_corr=True)
            add(fig, FigRec(fid, "corr_vitals_labs", "Correlation matrix",
                            "Pearson correlation matrix across available vitals and laboratory variables."))

    # Outcomes (keep Unknown outcome separate; rename Death -> Died)
    outcome_fields = [c for c in ["Survived without complications", "Survived with complications", "Death", "Absconded"]
                      if c in df.columns]

    if outcome_fields:
        def infer_outcome(row):
            if "Death" in row and _is_yes(row["Death"]):
                return OUTCOME_DEATH_LABEL
            if "Absconded" in row and _is_yes(row["Absconded"]):
                return OUTCOME_ABSCONDED_LABEL
            if "Survived with complications" in row and _is_yes(row["Survived with complications"]):
                return OUTCOME_SURV_COMP
            if "Survived without complications" in row and _is_yes(row["Survived without complications"]):
                return OUTCOME_SURV_NO_COMP
            return OUTCOME_UNKNOWN_LABEL

        df["Outcome_label"] = df.apply(infer_outcome, axis=1)

        if MERGE_UNKNOWN_OUTCOME_INTO_ABSCONDED:
            df["Outcome_label"] = df["Outcome_label"].replace({OUTCOME_UNKNOWN_LABEL: OUTCOME_ABSCONDED_LABEL})

        vc = df["Outcome_label"].value_counts(dropna=True)
        if len(vc) > 0:
            fid = next_id()
            fig = plot_bar_gradient(vc, "Outcome distribution", "Outcome", "Cases",
                                    "outcomes", fid, rotate=25, wrap_width=18)
            add(fig, FigRec(fid, "outcome_distribution", "Outcome distribution",
                            "Distribution of outcomes based on recorded outcome fields."))

        if "Study site" in df.columns and df["Outcome_label"].notna().any():
            sub = df.dropna(subset=["Study site", "Outcome_label"])
            if not sub.empty:
                fid = next_id()
                fig = plot_stacked_100(sub, "Study site", "Outcome_label",
                                       "Outcome by study site (100% stacked)",
                                       "outcomes", fid)
                add(fig, FigRec(fid, "outcome_by_site_100pct", "Outcome by study site",
                                "Within-site composition of outcomes."))

        if "Types of poisoning" in df.columns and df["Outcome_label"].notna().any():
            sub = df.dropna(subset=["Types of poisoning", "Outcome_label"])
            if not sub.empty:
                fid = next_id()
                fig = plot_outcome_by_poison_type_100pct_premium(
                    sub,
                    poison_col="Types of poisoning",
                    outcome_col="Outcome_label",
                    title="Outcome by poisoning type (100% stacked)",
                    fig_id=fid,
                )
                add(fig, FigRec(fid, "outcome_by_poison_type_100pct", "Outcome by poisoning type",
                                "Within-type composition of outcomes."))

    # Treatment (top 15 meds)
    name_cols = [c for c in df_raw.columns if str(c).strip().lower().startswith("name")]
    if name_cols:
        meds = []
        for c in name_cols:
            if c in df_raw.columns:
                meds.extend([norm_text(x) for x in df_raw[c].dropna().tolist()])
        meds = [m for m in meds if m and m.lower() not in {"na", "n/a", "none"}]
        if meds:
            vc = pd.Series(meds).value_counts().head(15)
            fid = next_id()
            fig = plot_bar_gradient(vc, "Medications administered (Top 15)", "Medication", "Count",
                                    "treatment", fid, rotate=35, wrap_width=20)
            add(fig, FigRec(fid, "medications_top15", "Medications (Top 15)",
                            "Most frequently recorded medication names across treatment fields (top 15)."))

    pdf.close()

    fig_index = pd.DataFrame([{
        "Figure_ID": r.fig_id,
        "Filename_Base": f"{r.fig_id}_{r.fname}",
        "Title": r.title,
        "Caption": r.caption
    } for r in figs])
    fig_index.to_csv(os.path.join(outdir, "figure_index.csv"), index=False)

    with open(os.path.join(outdir, "captions.md"), "w", encoding="utf-8") as f:
        for r in figs:
            f.write(f"**{r.fig_id}. {r.title}.** {r.caption}\n\n")

    print(f"Done. Exported {len(figs)} figures.")
    print(f"- Combined PDF: {pdf_path}")
    print(f"- Per-figure files in: {perfig}")
    print(f"- figure_index.csv and captions.md in: {outdir}")

if __name__ == "__main__":
    main()
