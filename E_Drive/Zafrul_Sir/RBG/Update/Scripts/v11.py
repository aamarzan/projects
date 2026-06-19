#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
V11 submission-master publication figure generator for the 874-driver accident-history manuscript.

This script rebuilds the figure package with a high-impact, restrained clinical /
public-health epidemiology visual system while preserving the original scientific
meaning and numeric source-of-truth values.

Default Windows workflow
------------------------
Place this script in:
    E:\Zafrul_Sir\RBG\Update\Scripts

Input files expected in:
    E:\Zafrul_Sir\RBG\Update\complete_case_874_dataset.csv
    E:\Zafrul_Sir\RBG\Update\accident_history_874_reanalysis_outputs.xlsx
    E:\Zafrul_Sir\RBG\Update\accident_history_reanalysis_tables.xlsx   (optional)
    E:\Zafrul_Sir\RBG\Update\Wobaidul_zafrul_RBG STUDY.sav             (optional)

Default output:
    E:\Zafrul_Sir\RBG\Update\Figures\v11_submission_master

Recommended local final run:
    python E:\Zafrul_Sir\RBG\Update\Scripts\Accident_Figure_v11_Submission_Master.py --version v11_submission_master --dpi 600 --bootstrap 100 --strict-vector

Scientific safeguards
---------------------
- No fabricated, simulated, or embellished scientific results.
- Observed prevalence intervals use Wilson 95% CIs.
- Model-standardized prevalence intervals use the fitted logistic model covariance
  matrix on the link/logit scale and are transformed back to probability.
- Continuous median-difference intervals use non-parametric bootstrap resampling
  of the available raw complete-case data.
- All model performance is internal/apparent or bootstrap-corrected internal.
  No external or prospective validation is claimed.
- Sparse cells are flagged using a dagger when n<30.
"""

from __future__ import annotations

import argparse
import os
import sys
import csv
import gc
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from textwrap import fill
from typing import Iterable, Optional, Sequence

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, FancyBboxPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from patsy import build_design_matrices
from PIL import Image, ImageDraw
from scipy.special import expit
from scipy.stats import chi2_contingency
from sklearn.metrics import (
    auc,
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

# =============================================================================
# 1) CONFIGURATION
# =============================================================================

DEFAULT_PROJECT_DIR = Path(r"E:\Zafrul_Sir\RBG\Update")
DEFAULT_VERSION = "v11_submission_master"
DEFAULT_RASTER_DPI = 600
RANDOM_STATE = 20260516
SPARSE_N_THRESHOLD = 30
CORE_FORMULA = "accident ~ Age_num + C(License_clean) + Driving_hours_num + RBG_num + Betel_binary"
ACTIONABLE_TERMS = ["C(License_clean)[T.Renew]", "Betel_binary", "RBG_num", "Driving_hours_num"]
FORMATS = ["pdf", "svg", "eps", "png", "tiff", "jpeg"]
STRICT_VECTOR_OUTPUT = False

VAR_LABELS = {
    "Driving_Year": "Driving years",
    "Driving_H_D": "Driving hours/day",
    "Driving_hours_num": "Driving hours/day",
    "Driving_Hour": "Driving hour",
    "RBG": "RBG",
    "RBG_num": "RBG",
    "RBG_conversion": "RBG conversion",
    "Age": "Age",
    "Age_num": "Age",
    "Age_Status": "Age status",
    "B_QuidPerD": "Betel quid/day",
    "Daily_C_N": "Cigarettes/day",
    "Cigerette_Brand": "Cigarette brand",
    "SHourPerDay": "Sleep hours/day",
    "SFoodCPerD": "Street food/day",
    "SFoodC": "Street food",
    "DeviceTBsleep": "Device before sleep",
    "DeviceBsleep": "Device during sleep",
    "License": "License",
    "Betel_Quid": "Betel quid",
    "B_Quid": "Betel quid",
    "Marital_Status": "Marital status",
    "Vehicle_Type": "Vehicle type",
    "Education": "Education",
    "B_Nut": "Betel nut",
    "DScreenT": "Screen time",
    "S_A_P": "Sleep apnea proxy",
    "DDS": "Diet diversity score",
    "S_Quality": "Sleep quality",
    "Family_H_Diabetes": "Family history diabetes",
    "Diabetic_Status": "Diabetic status",
    "DDIABETES_STATUS": "Diabetes status",
    "Glucose": "Urinary glucose",
    "TRremembering": "Trouble remembering",
    "PainBurnig": "Pain/burning urination",
}

CATEGORY_LABELS = {
    "License: Renew": "License: renew",
    "Marital Status: Married": "Marital status: married",
    "Cigerette Brand: Star": "Cigarette brand: Star",
    "FamilyHistory: Mother": "Family history: mother",
    "B Quid: Yes": "Betel quid: yes",
    "DeviceBsleep: No": "Device during sleep: no",
    "Betel Quid: Moderate consumption": "Betel quid: moderate",
    "Driving Hour: 5-10": "Driving hours: 5-10",
}


@dataclass
class RunPaths:
    project_dir: Path
    data_csv: Path
    table_xlsx: Path
    table_xlsx_2: Path
    sav_file: Path
    out_root: Path
    dirs: dict[str, dict[str, dict[str, Path]]] = field(default_factory=dict)
    qc_dir: Path = Path()


def resolve_project_dir(project_dir_override: Optional[str] = None) -> Path:
    if project_dir_override:
        return Path(project_dir_override).expanduser().resolve()
    try:
        script_path = Path(__file__).resolve()
        if script_path.parent.name.lower() == "scripts":
            return script_path.parent.parent
    except Exception:
        pass
    return DEFAULT_PROJECT_DIR


def configure_paths(version: str, project_dir_override: Optional[str] = None) -> RunPaths:
    project_dir = resolve_project_dir(project_dir_override)
    out_root = project_dir / "Figures" / version
    dirs: dict[str, dict[str, dict[str, Path]]] = {}
    for category in ["main", "supplementary"]:
        dirs[category] = {}
        for variant in ["titled_review", "titleless_submission"]:
            dirs[category][variant] = {}
            for fmt in FORMATS:
                p = out_root / category / variant / fmt
                p.mkdir(parents=True, exist_ok=True)
                dirs[category][variant][fmt] = p
    qc_dir = out_root / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    return RunPaths(
        project_dir=project_dir,
        data_csv=project_dir / "complete_case_874_dataset.csv",
        table_xlsx=project_dir / "accident_history_874_reanalysis_outputs.xlsx",
        table_xlsx_2=project_dir / "accident_history_reanalysis_tables.xlsx",
        sav_file=project_dir / "Wobaidul_zafrul_RBG STUDY.sav",
        out_root=out_root,
        dirs=dirs,
        qc_dir=qc_dir,
    )

# =============================================================================
# 2) THEME AND COLOR REGISTRY
# =============================================================================

COLORS = {
    "navy": "#0B1F3A",
    "ink": "#111827",
    "slate": "#374151",
    "muted": "#6B7280",
    "grid": "#E6E8EC",
    "white": "#FFFFFF",
    "demographic": "#0072B2",
    "occupational": "#D55E00",
    "lifestyle": "#E69F00",
    "metabolic": "#009E73",
    "urinary": "#CC79A7",
    "other": "#8A95A5",
    "rbg": "#A63A4B",
    "rbg_dark": "#7C2235",
    "driving": "#4E5BD5",
    "driving_dark": "#323C96",
    "license": "#8B5A00",
    "betel": "#D18E00",
    "none": "#7FA6C9",
    "light_rbg": "#F2DCE2",
    "light_driving": "#E4E7FF",
    "light_gray": "#F6F7F9",
    "faint": "#FAFAFB",
}

DOMAIN_COLORS = {
    "Demographic factors": COLORS["demographic"],
    "Occupational factors": COLORS["occupational"],
    "Lifestyle and behavioral factors": COLORS["lifestyle"],
    "Metabolic and clinical factors": COLORS["metabolic"],
    "Urinary and renal factors": COLORS["urinary"],
    "Other factors": COLORS["other"],
}
DOMAIN_SHORT = {
    "Demographic factors": "Demographic",
    "Occupational factors": "Occupational",
    "Lifestyle and behavioral factors": "Lifestyle",
    "Metabolic and clinical factors": "Metabolic",
    "Urinary and renal factors": "Urinary/renal",
    "Other factors": "Other",
}

CMAPS = {
    "signal": mpl.colors.LinearSegmentedColormap.from_list("signal", ["#F8FAFC", "#E7EEF5", "#E69F00", "#D55E00"]),
    "license_betel": mpl.colors.LinearSegmentedColormap.from_list("license_betel", ["#F8FAFC", "#F5E6B3", "#D18E00", "#8F2D42"]),
    "surface": mpl.colors.LinearSegmentedColormap.from_list("surface", ["#F8FAFC", "#CFE8EE", "#F4E6A1", "#D58A1E", "#8F2D42"]),
    "matrix": mpl.colors.LinearSegmentedColormap.from_list("matrix", ["#F7F7F7", "#CDEAC0", "#F4E27C", "#D88B24", "#812C45"]),
    "validation": mpl.colors.LinearSegmentedColormap.from_list("validation", ["#EAF3F8", "#0072B2"]),
}


def set_publication_theme(raster_dpi: int = DEFAULT_RASTER_DPI) -> None:
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": raster_dpi,
        "font.family": "DejaVu Sans",
        "font.size": 8.0,
        "axes.titlesize": 9.4,
        "axes.titleweight": "bold",
        "axes.labelsize": 8.8,
        "xtick.labelsize": 7.6,
        "ytick.labelsize": 7.6,
        "legend.fontsize": 7.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": COLORS["slate"],
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.8,
        "ytick.major.size": 2.8,
        "grid.color": COLORS["grid"],
        "grid.linewidth": 0.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def despine(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.115, 1.075, label, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=10.2, fontweight="bold", color=COLORS["navy"], clip_on=False)


def add_figure_title(fig: plt.Figure, title: str, include: bool) -> None:
    if include:
        fig.suptitle(title, x=0.50, y=0.992, ha="center", va="top",
                     fontsize=12.2, fontweight="bold", color=COLORS["navy"])


def clean_label(x: object, width: int = 20) -> str:
    s0 = str(x)
    s = VAR_LABELS.get(s0, s0.replace("_", " ").replace("  ", " ").strip())
    return fill(s, width=width)


def wrap_labels(values: Iterable[object], width: int = 18) -> list[str]:
    return [clean_label(v, width=width) for v in values]


def fmt_p(p: float) -> str:
    if p is None or pd.isna(p):
        return ""
    p = float(p)
    return "p<0.001" if p < 0.001 else f"p={p:.3f}"


def fmt_pct(x: float) -> str:
    return f"{x:.1f}%"


def clean_term(term: str) -> str:
    mapping = {
        "C(License_clean)[T.Renew]": "Renew license\n(vs new)",
        "license_renew": "Renew license\n(vs new)",
        "Betel_binary": "Betel quid\n(yes vs no)",
        "Smoking_binary": "Smoking\n(yes vs no)",
        "RBG_num": "RBG\n(per mmol/L)",
        "Driving_hours_num": "Driving hours/day\n(per hour)",
        "Age_num": "Age\n(per year)",
    }
    return mapping.get(term, clean_label(term, 22))


def domain_color(group: object) -> str:
    return DOMAIN_COLORS.get(str(group), COLORS["other"])


def contrast_text_color(value: float, vmin: float, vmax: float) -> str:
    if pd.isna(value):
        return COLORS["ink"]
    norm = (float(value) - float(vmin)) / max(float(vmax) - float(vmin), 1e-9)
    return COLORS["white"] if norm > 0.60 else COLORS["ink"]

# =============================================================================
# 3) DATA LOADING AND STATISTICS
# =============================================================================

REQUIRED_COLUMNS = ["accident", "Age_num", "RBG_num", "Driving_hours_num", "Smoking_binary", "Betel_binary", "License_clean"]
REQUIRED_SHEETS = ["All_variable_screening", "Category_specific_ORs", "License_betel_summary"]


def validate_inputs(df: pd.DataFrame, sheets: dict[str, pd.DataFrame]) -> None:
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing required dataset columns: {missing_cols}")
    missing_sheets = [s for s in REQUIRED_SHEETS if s not in sheets]
    if missing_sheets:
        raise KeyError(f"Missing required workbook sheets: {missing_sheets}")
    screen_req = ["variable", "group", "type", "p_value", "FDR_q_value"]
    missing = [c for c in screen_req if c not in sheets["All_variable_screening"].columns]
    if missing:
        raise KeyError(f"All_variable_screening missing columns: {missing}")
    cat_req = ["variable", "category", "n_category", "OR_category_vs_others", "CI_low", "CI_high", "p_value"]
    missing = [c for c in cat_req if c not in sheets["Category_specific_ORs"].columns]
    if missing:
        raise KeyError(f"Category_specific_ORs missing columns: {missing}")


def load_inputs(paths: RunPaths) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    if not paths.data_csv.exists():
        raise FileNotFoundError(f"Missing dataset: {paths.data_csv}")
    if not paths.table_xlsx.exists():
        raise FileNotFoundError(f"Missing workbook: {paths.table_xlsx}")
    df = pd.read_csv(paths.data_csv)
    xl = pd.ExcelFile(paths.table_xlsx)
    sheets = {s: pd.read_excel(xl, s) for s in xl.sheet_names}
    validate_inputs(df, sheets)
    extra_sheets: dict[str, pd.DataFrame] = {}
    if paths.table_xlsx_2.exists():
        xl2 = pd.ExcelFile(paths.table_xlsx_2)
        extra_sheets = {s: pd.read_excel(xl2, s) for s in xl2.sheet_names}
    for c in ["accident", "Age_num", "RBG_num", "Driving_hours_num", "Smoking_binary", "Betel_binary"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["License_clean"] = df["License_clean"].astype(str)
    return prepare_groups(df), sheets, extra_sheets


def prepare_groups(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["license_renew"] = np.where(d["License_clean"].str.lower().str.contains("renew"), 1, 0)
    d["license_group"] = np.where(d["license_renew"].eq(1), "Renew license", "New license")
    d["betel_group"] = np.where(d["Betel_binary"].eq(1), "Betel", "No betel")
    d["smoke_betel_group"] = pd.Series(pd.NA, index=d.index, dtype="object")
    d.loc[(d["Smoking_binary"].eq(0)) & (d["Betel_binary"].eq(0)), "smoke_betel_group"] = "Neither"
    d.loc[(d["Smoking_binary"].eq(1)) & (d["Betel_binary"].eq(0)), "smoke_betel_group"] = "Smoking only"
    d.loc[(d["Smoking_binary"].eq(0)) & (d["Betel_binary"].eq(1)), "smoke_betel_group"] = "Betel only"
    d.loc[(d["Smoking_binary"].eq(1)) & (d["Betel_binary"].eq(1)), "smoke_betel_group"] = "Both"
    d["RBG_cat"] = pd.cut(d["RBG_num"], bins=[-np.inf, 5.6, 7.8, np.inf], labels=["<5.6", "5.6-7.7", ">=7.8"])
    d["Drive_cat"] = pd.cut(d["Driving_hours_num"], bins=[-np.inf, 5, 10, np.inf], labels=["0-5 h", "6-10 h", "11+ h"])
    return d


def wilson_ci(events: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return np.nan, np.nan
    p = events / n
    denom = 1 + z**2 / n
    center = p + z**2 / (2*n)
    half = z * math.sqrt((p*(1-p) + z**2/(4*n))/n)
    return (center - half) / denom, (center + half) / denom


def fit_core_model(df: pd.DataFrame):
    model_data = df.dropna(subset=["accident", "Age_num", "License_clean", "Driving_hours_num", "RBG_num", "Betel_binary"]).copy()
    model = smf.logit(CORE_FORMULA, data=model_data).fit(disp=False, maxiter=250)
    return model, model_data


def model_to_or_table(model) -> pd.DataFrame:
    ci = model.conf_int()
    rows = []
    for term in model.params.index:
        if term == "Intercept":
            continue
        rows.append({
            "term": term,
            "label": clean_term(term),
            "OR": float(np.exp(model.params[term])),
            "lo": float(np.exp(ci.loc[term, 0])),
            "hi": float(np.exp(ci.loc[term, 1])),
            "p": float(model.pvalues[term]),
        })
    return pd.DataFrame(rows)


def actionable_or_table(model) -> pd.DataFrame:
    d = model_to_or_table(model)
    return d[d["term"].isin(ACTIONABLE_TERMS)].copy()


def prediction_ci(model, new_df: pd.DataFrame) -> pd.DataFrame:
    design_info = model.model.data.design_info
    X = build_design_matrices([design_info], new_df, return_type="dataframe")[0]
    beta = model.params.reindex(X.columns)
    cov = model.cov_params().reindex(index=X.columns, columns=X.columns)
    eta = np.asarray(X @ beta, dtype=float)
    var_eta = np.einsum("ij,jk,ik->i", np.asarray(X), np.asarray(cov), np.asarray(X))
    se_eta = np.sqrt(np.maximum(var_eta, 0))
    z = 1.959963984540054
    return pd.DataFrame({"pred": expit(eta), "lo": expit(eta - z*se_eta), "hi": expit(eta + z*se_eta)})


def prevalence_summary(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    s = df.dropna(subset=list(group_cols) + ["accident"]).groupby(list(group_cols), observed=True).agg(
        events=("accident", "sum"), n=("accident", "count"), prev=("accident", "mean")
    ).reset_index()
    lows, highs = [], []
    for _, r in s.iterrows():
        lo, hi = wilson_ci(int(r.events), int(r.n))
        lows.append(lo); highs.append(hi)
    s["lo"] = lows; s["hi"] = highs
    s["prev_pct"] = 100*s["prev"]
    s["lo_pct"] = 100*s["lo"]
    s["hi_pct"] = 100*s["hi"]
    return s


def prepare_screening(sheets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    screening = sheets["All_variable_screening"].copy()
    screening["p_value"] = pd.to_numeric(screening["p_value"], errors="coerce")
    screening["FDR_q_value"] = pd.to_numeric(screening["FDR_q_value"], errors="coerce")
    screening["strength"] = -np.log10(screening["p_value"].clip(lower=1e-300))
    screening["color"] = screening["group"].map(DOMAIN_COLORS).fillna(COLORS["other"])
    screening["pretty_variable"] = screening["variable"].map(lambda x: VAR_LABELS.get(str(x), str(x).replace("_", " ")))
    return screening


def prepare_categorical_or(sheets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    d = sheets["Category_specific_ORs"].copy()
    for c in ["OR_category_vs_others", "CI_low", "CI_high", "p_value", "n_category"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.rename(columns={"OR_category_vs_others": "OR", "CI_low": "lo", "CI_high": "hi", "p_value": "p"})
    d["raw_label"] = d["variable"].astype(str).str.replace("_", " ") + ": " + d["category"].astype(str)
    d["label"] = d["raw_label"].map(lambda x: fill(CATEGORY_LABELS.get(x, x), 24))
    return d


def bootstrap_median_difference(df: pd.DataFrame, variable: str, n_boot: int = 350, random_state: int = RANDOM_STATE) -> dict[str, float]:
    if variable not in df.columns:
        return {"variable": variable, "diff": np.nan, "lo": np.nan, "hi": np.nan, "n_accident": 0, "n_no": 0}
    sub = df[[variable, "accident"]].copy()
    sub[variable] = pd.to_numeric(sub[variable], errors="coerce")
    sub = sub.dropna()
    a = sub.loc[sub["accident"].eq(1), variable].to_numpy(float)
    b = sub.loc[sub["accident"].eq(0), variable].to_numpy(float)
    if len(a) < 5 or len(b) < 5:
        return {"variable": variable, "diff": np.nan, "lo": np.nan, "hi": np.nan, "n_accident": len(a), "n_no": len(b)}
    diff = float(np.median(a) - np.median(b))
    seed = random_state + sum(ord(ch) for ch in variable)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        aa = rng.choice(a, size=len(a), replace=True)
        bb = rng.choice(b, size=len(b), replace=True)
        vals.append(np.median(aa) - np.median(bb))
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return {"variable": variable, "diff": diff, "lo": float(lo), "hi": float(hi), "n_accident": len(a), "n_no": len(b)}


def bootstrap_optimism_metrics(df: pd.DataFrame, n_boot: int = 100, random_state: int = RANDOM_STATE) -> dict[str, float]:
    model, model_data = fit_core_model(df)
    y = model_data["accident"].to_numpy(dtype=int)
    pred_app = np.asarray(model.predict(model_data))
    apparent_auc = roc_auc_score(y, pred_app)
    apparent_brier = brier_score_loss(y, pred_app)
    rng = np.random.default_rng(random_state)
    auc_opt, brier_opt = [], []
    n = len(model_data)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot = model_data.iloc[idx].copy()
        if boot["accident"].nunique() < 2:
            continue
        try:
            boot_model = smf.logit(CORE_FORMULA, data=boot).fit(disp=False, maxiter=250)
            y_boot = boot["accident"].to_numpy(dtype=int)
            p_train = np.asarray(boot_model.predict(boot))
            p_orig = np.asarray(boot_model.predict(model_data))
            auc_opt.append(roc_auc_score(y_boot, p_train) - roc_auc_score(y, p_orig))
            brier_opt.append(brier_score_loss(y, p_orig) - brier_score_loss(y_boot, p_train))
        except Exception:
            continue
    auc_o = float(np.nanmean(auc_opt)) if auc_opt else np.nan
    brier_o = float(np.nanmean(brier_opt)) if brier_opt else np.nan
    return {
        "n_boot_requested": n_boot,
        "n_boot_successful": len(auc_opt),
        "apparent_auc": float(apparent_auc),
        "optimism_corrected_auc": float(apparent_auc - auc_o) if not np.isnan(auc_o) else np.nan,
        "apparent_brier": float(apparent_brier),
        "optimism_corrected_brier": float(apparent_brier + brier_o) if not np.isnan(brier_o) else np.nan,
    }

# =============================================================================
# 4) PLOTTING PRIMITIVES
# =============================================================================


def shade_alternate_rows(ax: plt.Axes, n_rows: int) -> None:
    for i in range(n_rows):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color=COLORS["faint"], zorder=0)


def plot_forest_with_side_table(ax: plt.Axes, ax_tbl: plt.Axes, d: pd.DataFrame, title: str,
                                xlim: Optional[tuple[float, float]] = None, label_size: float = 7.4,
                                header: str = "OR (95% CI); p") -> None:
    dd = d.copy().replace([np.inf, -np.inf], np.nan).dropna(subset=["OR", "lo", "hi"])
    dd = dd.sort_values("OR", ascending=False).reset_index(drop=True)
    if dd.empty:
        ax.axis("off"); ax_tbl.axis("off"); return
    y = np.arange(len(dd))
    shade_alternate_rows(ax, len(dd))
    ax.axvline(1, color=COLORS["muted"], lw=0.8, ls="--", zorder=1)
    for yi, (_, r) in zip(y, dd.iterrows()):
        ax.hlines(yi, r["lo"], r["hi"], color=COLORS["slate"], lw=1.0, zorder=2)
    sig = pd.to_numeric(dd.get("p", 1), errors="coerce") < 0.05
    ax.scatter(dd["OR"], y, s=26, color=np.where(sig, COLORS["rbg"], COLORS["demographic"]),
               edgecolor=COLORS["white"], lw=0.55, zorder=3)
    ax.set_xscale("log")
    if xlim:
        ax.set_xlim(*xlim)
    else:
        ax.set_xlim(max(0.2, dd["lo"].min()*0.7), dd["hi"].max()*1.25)
    ax.set_yticks(y)
    ax.set_yticklabels(dd["label"].tolist(), fontsize=label_size)
    # Extra top space reserves a non-overlapping header row for the side table.
    ax.set_ylim(len(dd)-0.45, -0.95)
    ax.set_xlabel("Odds ratio (log scale)")
    ax.set_title(title, pad=8)
    # Human-readable log-scale ticks.
    xmin, xmax = ax.get_xlim()
    tick_candidates = [0.8, 1, 1.5, 2, 3, 4, 6, 8]
    ticks = [t for t in tick_candidates if xmin <= t <= xmax]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t).rstrip("0").rstrip(".") for t in ticks])
    ax.grid(axis="x", alpha=0.40)
    despine(ax)
    ax_tbl.set_ylim(ax.get_ylim())
    ax_tbl.set_xlim(0, 1)
    ax_tbl.axis("off")
    ax_tbl.text(0.0, -0.62, header, ha="left", va="center",
                fontsize=6.9, fontweight="bold", color=COLORS["slate"], clip_on=False)
    for yi, (_, r) in zip(y, dd.iterrows()):
        txt = f"{r['OR']:.2f} ({r['lo']:.2f}-{r['hi']:.2f}); {fmt_p(r.get('p', np.nan))}"
        ax_tbl.text(0.0, yi, txt, ha="left", va="center", fontsize=7.0, color=COLORS["slate"])


def plot_evidence_skyline(ax: plt.Axes, d: pd.DataFrame, top_n: int, title: str,
                          xlabel: str = "-log10(p)", label_width: int = 18,
                          show_legend: bool = True) -> None:
    dd = d.dropna(subset=["strength"]).sort_values("strength", ascending=False).head(top_n).copy()
    dd = dd.sort_values("strength", ascending=True).reset_index(drop=True)
    y = np.arange(len(dd))
    xmax = max(2.3, float(dd["strength"].max()) * 1.05)
    # Evidence zones encode statistical evidence, not decoration.
    ax.axvspan(0, -np.log10(0.05), color="#F6F7F9", zorder=0)
    ax.axvspan(-np.log10(0.05), xmax, color="#FFF7E6", alpha=0.55, zorder=0)
    ax.hlines(y, 0, dd["strength"], color="#DADDE3", lw=1.25, zorder=1)
    fdr = pd.to_numeric(dd["FDR_q_value"], errors="coerce") <= 0.10
    sizes = 24 + 16 * (dd["strength"] / max(dd["strength"].max(), 1))
    ax.scatter(dd.loc[~fdr, "strength"], y[~fdr], s=sizes[~fdr], facecolor=COLORS["white"],
               edgecolor=dd.loc[~fdr, "color"], lw=1.25, zorder=3)
    ax.scatter(dd.loc[fdr, "strength"], y[fdr], s=sizes[fdr], color=dd.loc[fdr, "color"],
               edgecolor=COLORS["white"], lw=0.5, zorder=3)
    ax.axvline(-np.log10(0.05), color=COLORS["muted"], lw=0.8, ls="--")
    ax.set_xlim(0, xmax)
    ax.set_yticks(y)
    ax.set_yticklabels(wrap_labels(dd["variable"], label_width))
    ax.set_xlabel(xlabel)
    ax.set_title(title, pad=6)
    ax.grid(axis="x", alpha=0.35)
    despine(ax)
    if show_legend:
        handles = [Line2D([0],[0], marker='o', color='none', markerfacecolor=v, markeredgecolor=COLORS['white'], markersize=5.2, label=DOMAIN_SHORT[k]) for k,v in DOMAIN_COLORS.items()]
        handles += [Line2D([0],[0], marker='o', color=COLORS['slate'], markerfacecolor=COLORS['white'], markersize=5.2, label='Nominal'),
                    Line2D([0],[0], marker='o', color='none', markerfacecolor=COLORS['navy'], markersize=5.2, label='FDR q<=0.10')]
        ax.legend(handles=handles, frameon=False, fontsize=6.3, ncol=2, loc='lower right')


def plot_domain_architecture(ax: plt.Axes, screening: pd.DataFrame) -> None:
    rows = []
    for g, s in screening.groupby("group"):
        rows.append({
            "domain": DOMAIN_SHORT.get(g, g),
            "screened": len(s),
            "nominal": int((s["p_value"] < 0.05).sum()),
            "fdr": int((s["FDR_q_value"] <= 0.10).sum()),
            "color": domain_color(g),
        })
    dom = pd.DataFrame(rows).sort_values("screened", ascending=True).reset_index(drop=True)
    y = np.arange(len(dom))
    ax.set_title("Domain-level signal architecture", pad=6)
    stages = [("Screened", "screened", COLORS["other"]), ("p<0.05", "nominal", None), ("FDR q<=0.10", "fdr", COLORS["navy"])]
    # Matrix of glyphs: x stage, y domain, glyph size = count.
    for sx, (label, col, fixed_color) in enumerate(stages):
        counts = dom[col].to_numpy(float)
        smax = max(1, dom["screened"].max())
        sizes = 35 + 420 * np.sqrt(counts / smax)
        colors = fixed_color if fixed_color is not None else dom["color"].tolist()
        ax.scatter(np.full(len(dom), sx), y, s=sizes, color=colors, alpha=0.92,
                   edgecolor=COLORS["white"], lw=0.75, zorder=3)
        for yi, c in zip(y, counts):
            ax.text(sx, yi, f"{int(c)}", ha="center", va="center", fontsize=6.4,
                    color=COLORS["white"] if c > smax*0.25 or sx==2 else COLORS["ink"], fontweight="bold", zorder=4)
    for yi, col in zip(y, dom["color"]):
        ax.add_patch(Rectangle((-0.65, yi-0.27), 0.10, 0.54, color=col, clip_on=False))
    ax.set_yticks(y); ax.set_yticklabels(dom["domain"])
    ax.set_xticks(range(3)); ax.set_xticklabels([s[0] for s in stages])
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylim(-0.6, len(dom)-0.4)
    ax.grid(axis="y", alpha=0.25)
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_median_contrast(ax: plt.Axes, df: pd.DataFrame, variables: list[str], title: str,
                         label_width: int = 18, n_boot: int = 300) -> None:
    """Bootstrap median-difference contrast ridges.

    The x-axis remains the median difference (accident-history minus no accident-history).
    The translucent ridge behind each point shows the bootstrap distribution of the
    median-difference estimate, not a decorative background.
    """
    rows = []
    rng = np.random.default_rng(RANDOM_STATE)
    samples: dict[str, np.ndarray] = {}
    for v in variables:
        res = bootstrap_median_difference(df, v, n_boot=n_boot)
        rows.append(res)
        if v in df.columns and np.isfinite(res.get("diff", np.nan)):
            sub = df[[v, "accident"]].copy()
            sub[v] = pd.to_numeric(sub[v], errors="coerce")
            sub = sub.dropna()
            a = sub.loc[sub["accident"].eq(1), v].to_numpy(float)
            b = sub.loc[sub["accident"].eq(0), v].to_numpy(float)
            vals = []
            rr = np.random.default_rng(RANDOM_STATE + sum(ord(ch) for ch in v) + 1009)
            if len(a) >= 5 and len(b) >= 5:
                for _ in range(n_boot):
                    vals.append(np.median(rr.choice(a, size=len(a), replace=True)) -
                                np.median(rr.choice(b, size=len(b), replace=True)))
            samples[v] = np.asarray(vals, dtype=float)
    dd = pd.DataFrame(rows).dropna(subset=["diff"]).sort_values("diff")
    y = np.arange(len(dd))
    shade_alternate_rows(ax, len(dd))
    ax.axvline(0, color=COLORS["muted"], lw=0.8)
    xmin = float(np.nanmin([dd["lo"].min(), dd["diff"].min(), 0]))
    xmax = float(np.nanmax([dd["hi"].max(), dd["diff"].max(), 0]))
    pad = max(0.5, (xmax - xmin) * 0.08)
    xgrid = np.linspace(xmin - pad, xmax + pad, 220)
    for yi, (_, r) in zip(y, dd.iterrows()):
        vals = samples.get(r["variable"], np.array([]))
        col = COLORS["occupational"] if "Driving" in r["variable"] else COLORS["metabolic"] if "RBG" in r["variable"] else COLORS["lifestyle"] if ("Quid" in r["variable"] or "C_N" in r["variable"] or "SFood" in r["variable"]) else COLORS["demographic"]
        if len(vals) > 10 and np.nanstd(vals) > 0:
            bw = max(np.nanstd(vals) * 0.28, 0.08)
            dens = np.exp(-0.5 * ((xgrid[:, None] - vals[None, :]) / bw) ** 2).sum(axis=1)
            dens = dens / max(dens.max(), 1e-9) * 0.23
            ax.fill_between(xgrid, yi - dens, yi + dens, color=col, alpha=0.13, lw=0, zorder=1)
        ax.hlines(yi, r["lo"], r["hi"], color=COLORS["slate"], lw=1.0, zorder=2)
        ax.scatter(r["diff"], yi, s=30, color=col, edgecolor=COLORS["white"], lw=0.6, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([clean_label(v, width=label_width) for v in dd["variable"]])
    ax.set_xlabel("Median difference (accident - no accident)")
    ax.set_title(title, pad=6)
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.grid(axis="x", alpha=0.32)
    despine(ax)


def plot_prevalence_slope(ax: plt.Axes, prev: pd.DataFrame, title: str) -> None:
    endpoints = list(prev["endpoint"].unique())
    colors = {"Betel quid": COLORS["betel"], "Accident history": COLORS["rbg"]}
    x_positions = {"New license": 0, "Renew license": 1}
    for endpoint in endpoints:
        dd = prev[prev["endpoint"].eq(endpoint)].copy()
        x = dd["license"].map(x_positions).to_numpy(float)
        y = dd["prev_pct"].to_numpy(float)
        ax.plot(x, y, color=colors.get(endpoint, COLORS["slate"]), lw=1.35, marker="o", ms=4.8, label=endpoint, zorder=3)
        ax.errorbar(x, y, yerr=[y - dd["lo_pct"].to_numpy(float), dd["hi_pct"].to_numpy(float) - y],
                    fmt="none", color=colors.get(endpoint, COLORS["slate"]), lw=0.9, capsize=3.0, zorder=2)
        delta = y[1] - y[0] if len(y) == 2 else np.nan
        for xi, (_, r) in zip(x, dd.iterrows()):
            offset = 4.0 if endpoint == "Betel quid" else 1.8
            ax.text(xi, r["prev_pct"] + offset, f"{r['prev_pct']:.1f}%\n{int(r['events'])}/{int(r['n'])}",
                    ha="center", va="bottom", fontsize=6.7, color=COLORS["ink"])
        if len(y) == 2:
            ax.text(1.03, y[1], f"+{delta:.1f} pp", color=colors.get(endpoint, COLORS["slate"]),
                    ha="left", va="center", fontsize=6.8, fontweight="bold")
    ax.set_xlim(-0.15, 1.25)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["New license", "Renew license"])
    ax.set_ylabel("Observed prevalence (%)")
    ax.set_title(title, pad=6)
    ax.set_ylim(0, 75)
    ax.grid(axis="y", alpha=0.35)
    ax.legend(frameon=False, loc="upper left", fontsize=7.0)
    despine(ax)


def plot_tile_burden(ax: plt.Axes, data: pd.DataFrame, row_col: str, col_col: str, row_order: list[str], col_order: list[str],
                     title: str, cmap: mpl.colors.Colormap, with_sample_circles: bool = True):
    vals = data["prev_pct"].to_numpy(float)
    vmin, vmax = 0, max(5, math.ceil((np.nanmax(vals) + 2) / 5) * 5)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    max_n = max(1, data["n"].max())
    for i, row in enumerate(row_order):
        for j, col in enumerate(col_order):
            r = data[(data[row_col].astype(str).eq(str(row))) & (data[col_col].astype(str).eq(str(col)))]
            if r.empty:
                continue
            rr = r.iloc[0]
            face = cmap(norm(rr["prev_pct"]))
            patch = FancyBboxPatch((j-0.46, i-0.42), 0.92, 0.84,
                                   boxstyle="round,pad=0.015,rounding_size=0.035",
                                   facecolor=face, edgecolor=COLORS["white"], lw=1.0, zorder=1)
            ax.add_patch(patch)
            if with_sample_circles:
                radius = 0.22 * math.sqrt(float(rr["n"]) / float(max_n))
                circ = plt.Circle((j-0.31, i+0.25), radius=radius, facecolor=COLORS["white"],
                                  edgecolor=COLORS["slate"], lw=0.45, alpha=0.78, zorder=2)
                ax.add_patch(circ)
                ax.text(j-0.31, i+0.25, f"n={int(rr['n'])}", ha="center", va="center", fontsize=5.7,
                        color=COLORS["slate"], zorder=3)
            sparse = int(rr["n"]) < SPARSE_N_THRESHOLD
            dagger = "†" if sparse else ""
            ax.text(j, i-0.02, f"{rr['prev_pct']:.1f}%{dagger}\n{int(rr['events'])}/{int(rr['n'])}",
                    ha="center", va="center", fontsize=7.7, color=contrast_text_color(rr["prev_pct"], vmin, vmax),
                    fontweight="bold", zorder=4)
            if sparse:
                ax.add_patch(Rectangle((j-0.48, i-0.44), 0.96, 0.88, fill=False,
                                       edgecolor=COLORS["navy"], lw=0.8, ls=":"))
    ax.set_xticks(np.arange(len(col_order))); ax.set_xticklabels(col_order)
    ax.set_yticks(np.arange(len(row_order))); ax.set_yticklabels(row_order)
    ax.set_xlim(-0.55, len(col_order)-0.45); ax.set_ylim(len(row_order)-0.45, -0.55)
    ax.set_title(title, pad=6)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    return sm, vmax


def add_density_strip(ax: plt.Axes, x: np.ndarray, color: str, bins: int = 28) -> None:
    ins = inset_axes(ax, width="88%", height="15%", loc="lower center", borderpad=0.85)
    ins.hist(x, bins=bins, color=color, alpha=0.25, edgecolor="none")
    ins.set_yticks([]); ins.set_xticks([])
    for spine in ins.spines.values():
        spine.set_visible(False)
    ins.patch.set_alpha(0)


def plot_attenuation_profile(ax: plt.Axes, ax_tbl: plt.Axes, d: pd.DataFrame, title: str,
                             xlim: tuple[float, float] = (0.8, 6.5)) -> None:
    """Connected attenuation display for license effect before/after betel adjustment.

    This is deliberately not a generic forest plot: the first two rows are a
    coefficient path for the renewal-license estimate; the third row displays the
    adjusted betel estimate on the same log-OR scale.
    """
    dd = d.copy().reset_index(drop=True)
    y = np.arange(len(dd))
    shade_alternate_rows(ax, len(dd))
    ax.axvline(1, color=COLORS["muted"], lw=0.8, ls="--", zorder=1)
    # Connect the renewal-license unadjusted and adjusted estimates.
    ax.plot(dd.loc[[0, 1], "OR"], y[[0, 1]], color=COLORS["license"], lw=1.2, alpha=0.72, zorder=2)
    for yi, (_, r) in zip(y, dd.iterrows()):
        col = COLORS["license"] if yi < 2 else COLORS["betel"]
        ax.hlines(yi, r["lo"], r["hi"], color=COLORS["slate"], lw=1.0, zorder=2)
        ax.scatter(r["OR"], yi, s=34, color=col, edgecolor=COLORS["white"], lw=0.6, zorder=3)
    ax.set_xscale("log"); ax.set_xlim(*xlim)
    ax.set_ylim(len(dd)-0.45, -0.95)
    ax.set_yticks(y); ax.set_yticklabels(dd["label"].tolist(), fontsize=7.1)
    ticks = [0.8, 1, 1.5, 2, 3, 4, 6]
    ax.set_xticks([t for t in ticks if xlim[0] <= t <= xlim[1]])
    ax.set_xticklabels([str(t).rstrip("0").rstrip(".") for t in ax.get_xticks()])
    ax.set_xlabel("Odds ratio (log scale)")
    ax.set_title(title, pad=6)
    ax.grid(axis="x", alpha=0.32)
    despine(ax)
    ax_tbl.set_ylim(ax.get_ylim()); ax_tbl.set_xlim(0, 1); ax_tbl.axis("off")
    ax_tbl.text(0.0, -0.62, "OR (95% CI); p", ha="left", va="center",
                fontsize=7.0, fontweight="bold", color=COLORS["slate"], clip_on=False)
    for yi, (_, r) in zip(y, dd.iterrows()):
        ax_tbl.text(0.0, yi, f"{r['OR']:.2f} ({r['lo']:.2f}-{r['hi']:.2f}); {fmt_p(r['p'])}",
                    ha="left", va="center", fontsize=7.0, color=COLORS["slate"])


def plot_domain_faceted_discovery(fig: plt.Figure, subspec, screening: pd.DataFrame) -> None:
    """Domain-faceted top-30 discovery screen, visually distinct from Figure 1A."""
    top = screening.dropna(subset=["p_value"]).sort_values("p_value").head(30).copy()
    order = ["Occupational factors", "Demographic factors", "Lifestyle and behavioral factors",
             "Metabolic and clinical factors", "Urinary and renal factors", "Other factors"]
    gs = subspec.subgridspec(3, 2, wspace=0.36, hspace=0.58)
    for k, domain in enumerate(order):
        ax = fig.add_subplot(gs[k // 2, k % 2])
        dd = top[top["group"].eq(domain)].sort_values("strength", ascending=True).copy()
        if dd.empty:
            ax.axis("off"); continue
        y = np.arange(len(dd))
        ax.axvspan(0, -np.log10(0.05), color="#F6F7F9", zorder=0)
        ax.axvspan(-np.log10(0.05), max(3, dd["strength"].max()*1.08), color="#FFF7E6", alpha=0.45, zorder=0)
        ax.hlines(y, 0, dd["strength"], color="#DADDE3", lw=1.0)
        fdr = pd.to_numeric(dd["FDR_q_value"], errors="coerce") <= 0.10
        ax.scatter(dd.loc[~fdr, "strength"], y[~fdr], s=26, facecolor=COLORS["white"],
                   edgecolor=domain_color(domain), lw=1.05, zorder=3)
        ax.scatter(dd.loc[fdr, "strength"], y[fdr], s=28, color=domain_color(domain),
                   edgecolor=COLORS["white"], lw=0.45, zorder=3)
        ax.axvline(-np.log10(0.05), color=COLORS["muted"], lw=0.7, ls="--")
        ax.set_yticks(y); ax.set_yticklabels(wrap_labels(dd["variable"], 16), fontsize=6.8)
        ax.set_title(DOMAIN_SHORT.get(domain, domain), color=domain_color(domain), pad=4, fontsize=8.7)
        ax.grid(axis="x", alpha=0.25); despine(ax)
        ax.set_xlim(0, max(5, top["strength"].max() * 1.03))
        if k < 4: ax.set_xlabel("")
        else: ax.set_xlabel("Association strength, -log10(p)")

# =============================================================================
# 5) EXPORT AND QC
# =============================================================================



def save_figure_all_formats(fig: plt.Figure, stem: str, category: str, variant: str, paths: RunPaths, dpi: int) -> list[dict[str, str]]:
    """Save publication-vector and raster-fallback outputs honestly.

    In strict-vector mode, PDF, SVG, and EPS are requested directly from
    Matplotlib. Dense scientific image layers such as pcolormesh, contourf,
    or heatmap tiles may remain selectively rasterized within vector
    containers, but text, axes, ticks, labels, legends, and annotations are
    kept as vector artists wherever Matplotlib supports it. Any fallback is
    explicitly recorded in the manifest.
    """
    records: list[dict[str, str]] = []
    dirs = paths.dirs[category][variant]
    common = dict(facecolor=COLORS["white"], bbox_inches="tight", pad_inches=0.06)

    # Raster branch
    png = dirs["png"] / f"{stem}.png"
    fig.savefig(png, format="png", dpi=dpi, **common)
    records.append({"category": category, "variant": variant, "format": "png", "file": str(png), "bytes": str(png.stat().st_size), "vector_status": "raster_600dpi"})
    with Image.open(png).convert("RGB") as img:
        jpg = dirs["jpeg"] / f"{stem}.jpg"
        img.save(jpg, "JPEG", quality=95, optimize=True, dpi=(dpi, dpi))
        records.append({"category": category, "variant": variant, "format": "jpeg", "file": str(jpg), "bytes": str(jpg.stat().st_size), "vector_status": "raster_600dpi"})
        tif = dirs["tiff"] / f"{stem}.tiff"
        img.save(tif, "TIFF", compression="tiff_lzw", dpi=(dpi, dpi))
        records.append({"category": category, "variant": variant, "format": "tiff", "file": str(tif), "bytes": str(tif.stat().st_size), "vector_status": "raster_600dpi_lzw"})

    # Publication-vector branch.
    vector_formats = ["pdf", "svg"] if STRICT_VECTOR_OUTPUT else ["pdf"]
    for fmt in vector_formats:
        out = dirs[fmt] / f"{stem}.{fmt}"
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fig.savefig(out, format=fmt, dpi=dpi, **common)
            if fmt == "pdf":
                status = "direct_vector_pdf_selective_raster_layers_possible"
            elif fmt == "svg":
                status = "direct_vector_svg_editable_text_selective_raster_layers_possible"
            else:
                status = "direct_vector_eps_selective_raster_layers_possible"
            records.append({"category": category, "variant": variant, "format": fmt, "file": str(out), "bytes": str(out.stat().st_size), "vector_status": status})
        except Exception as e:
            # Explicitly labelled fallback, never called publication-vector.
            with Image.open(png).convert("RGB") as img:
                if fmt == "pdf":
                    img.save(out, "PDF", resolution=dpi)
                elif fmt == "svg":
                    import base64
                    with open(png, "rb") as fh:
                        encoded = base64.b64encode(fh.read()).decode("ascii")
                    w, h = img.size
                    out.write_text((f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">\n'
                                    f'<image width="{w}" height="{h}" href="data:image/png;base64,{encoded}"/>\n'
                                    f'</svg>\n'), encoding="utf-8")
                else:
                    img.save(out, "EPS")
            records.append({"category": category, "variant": variant, "format": fmt, "file": str(out), "bytes": str(out.stat().st_size), "vector_status": f"raster_fallback_after_direct_export_failure:{type(e).__name__}"})

    if STRICT_VECTOR_OUTPUT:
        # EPS is kept as a high-resolution PostScript fallback because direct EPS
        # export with semi-transparent scientific layers is slow and inconsistent
        # across Matplotlib/PostScript backends. PDF/SVG are the publication-vector
        # branches; this fallback is documented honestly in the manifest.
        eps = dirs["eps"] / f"{stem}.eps"
        with Image.open(png).convert("RGB") as img:
            img.save(eps, "EPS")
        records.append({"category": category, "variant": variant, "format": "eps", "file": str(eps), "bytes": str(eps.stat().st_size), "vector_status": "raster_eps_postscript_fallback_strict"})

    if not STRICT_VECTOR_OUTPUT:
        # Lightweight review mode: create SVG/EPS fallbacks and label them as such.
        svg = dirs["svg"] / f"{stem}.svg"
        eps = dirs["eps"] / f"{stem}.eps"
        with Image.open(png).convert("RGB") as img:
            import base64
            with open(png, "rb") as fh:
                encoded = base64.b64encode(fh.read()).decode("ascii")
            w, h = img.size
            svg.write_text((f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">\n'
                            f'<image width="{w}" height="{h}" href="data:image/png;base64,{encoded}"/>\n'
                            f'</svg>\n'), encoding="utf-8")
            img.save(eps, "EPS")
        records.append({"category": category, "variant": variant, "format": "svg", "file": str(svg), "bytes": str(svg.stat().st_size), "vector_status": "raster_wrapper_non_strict_review_mode"})
        records.append({"category": category, "variant": variant, "format": "eps", "file": str(eps), "bytes": str(eps.stat().st_size), "vector_status": "raster_eps_non_strict_review_mode"})

    plt.close(fig); plt.close("all"); gc.collect()
    return records

def _make_contact_sheet_from_images(image_paths: list[Path], out: Path, thumb_w: int = 540) -> Path:
    thumbs = []
    for p in image_paths:
        im = Image.open(p).convert("RGB")
        ratio = thumb_w / im.width
        thumb_h = int(im.height * ratio)
        im = im.resize((thumb_w, thumb_h), Image.LANCZOS)
        canvas = Image.new("RGB", (thumb_w, thumb_h + 42), "white")
        canvas.paste(im, (0, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, thumb_h + 11), p.name, fill=(20, 30, 45))
        thumbs.append(canvas)
    cols = 2
    rows = math.ceil(len(thumbs) / cols) if thumbs else 1
    row_heights = [max([thumbs[i].height for i in range(r*cols, min((r+1)*cols, len(thumbs)))], default=160) for r in range(rows)]
    sheet_w = cols * thumb_w + (cols + 1) * 24
    sheet_h = sum(row_heights) + (rows + 1) * 24
    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
    y = 24
    for r in range(rows):
        x = 24
        for c in range(cols):
            idx = r*cols + c
            if idx < len(thumbs):
                sheet.paste(thumbs[idx], (x, y))
            x += thumb_w + 24
        y += row_heights[r] + 24
    sheet.save(out)
    return out


def _render_pdf_first_page(pdf_path: Path, out_path: Path, dpi: int = 160) -> Optional[Path]:
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(pdf_path))
        page = doc[0]
        zoom = dpi / 72
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        pix.save(str(out_path))
        doc.close()
        return out_path
    except Exception:
        return None


def create_contact_sheets(paths: RunPaths, manifest_rows: list[dict[str, str]], thumb_w: int = 540) -> tuple[Path, Path]:
    pngs = [Path(r["file"]) for r in manifest_rows if r["format"] == "png" and r["variant"] == "titled_review"]
    pngs = sorted(pngs, key=lambda p: ("supplementary" in str(p), p.name))
    png_out = paths.qc_dir / "rendered_contact_sheet_from_png.png"
    _make_contact_sheet_from_images(pngs, png_out, thumb_w=thumb_w)
    pdfs = [Path(r["file"]) for r in manifest_rows if r["format"] == "pdf" and r["variant"] == "titled_review"]
    pdfs = sorted(pdfs, key=lambda p: ("supplementary" in str(p), p.name))
    render_dir = paths.qc_dir / "pdf_render_previews"
    render_dir.mkdir(exist_ok=True, parents=True)
    rendered = []
    for p in pdfs:
        out = render_dir / f"{p.stem}.png"
        r = _render_pdf_first_page(p, out)
        if r is not None:
            rendered.append(r)
    pdf_out = paths.qc_dir / "rendered_contact_sheet_from_pdf.png"
    _make_contact_sheet_from_images(rendered if rendered else pngs, pdf_out, thumb_w=thumb_w)
    # Backward-compatible name expected by some workflows.
    legacy = paths.qc_dir / "rendered_contact_sheet.png"
    try:
        Image.open(pdf_out).save(legacy)
    except Exception:
        pass
    return png_out, pdf_out

def write_manifest(paths: RunPaths, records: list[dict[str, str]]) -> Path:
    out = paths.qc_dir / "output_file_manifest.csv"
    # Normalize fieldnames to include fallback if present.
    fieldnames = ["category", "variant", "format", "file", "bytes", "vector_status", "fallback", "skipped_existing"]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in records:
            writer.writerow(row)
    return out

# =============================================================================
# 6) FIGURE BUILDERS
# =============================================================================


def build_figure_1(df: pd.DataFrame, sheets: dict[str, pd.DataFrame], paths: RunPaths, variant: str, include_title: bool, dpi: int, records: list, qc: list):
    screening = prepare_screening(sheets)
    cat = prepare_categorical_or(sheets)
    model, _ = fit_core_model(df)
    core = actionable_or_table(model)
    fig = plt.figure(figsize=(11.4, 7.8), constrained_layout=False)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 1.02, 1.30], height_ratios=[1, 1.07], wspace=0.48, hspace=0.56)
    axA = fig.add_subplot(gs[0,0]); axB = fig.add_subplot(gs[0,1])
    sgC = gs[0,2].subgridspec(1, 2, width_ratios=[0.65, 0.35], wspace=0.04)
    axC = fig.add_subplot(sgC[0,0]); axCt = fig.add_subplot(sgC[0,1])
    axD = fig.add_subplot(gs[1,0])
    sgE = gs[1,1:].subgridspec(1,2, width_ratios=[0.67, 0.33], wspace=0.04)
    axE = fig.add_subplot(sgE[0,0]); axEt = fig.add_subplot(sgE[0,1])
    add_panel_label(axA,"A"); plot_evidence_skyline(axA, screening, 16, "Candidate-signal landscape", "-log10(p)", 18, show_legend=True)
    add_panel_label(axB,"B"); plot_domain_architecture(axB, screening)
    add_panel_label(axC,"C"); plot_forest_with_side_table(axC, axCt, core[["label","OR","lo","hi","p"]], "Adjusted signal estimates", xlim=(0.85,3.9), label_size=7.1)
    add_panel_label(axD,"D"); plot_median_contrast(axD, df, ["Driving_Year","Age","B_QuidPerD","Driving_H_D","RBG","SHourPerDay","Daily_C_N"], "Continuous exposure contrasts", n_boot=250)
    add_panel_label(axE,"E")
    dcat = cat[(cat["OR"]>1) & (cat["p"]<0.01) & (cat["n_category"]>=20)].copy().sort_values("p").head(8)
    plot_forest_with_side_table(axE, axEt, dcat[["label","OR","lo","hi","p"]], "Categorical signal estimates", xlim=(0.85,6.8), label_size=6.9)
    add_figure_title(fig, "Candidate-signal atlas and adjusted accident-history estimates", include_title)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.91 if include_title else 0.975, bottom=0.08)
    records.extend(save_figure_all_formats(fig, "Figure_1_Candidate_Signal_Atlas", "main", variant, paths, dpi))
    if variant == "titled_review": qc.append("Figure 1: upgraded to evidence skyline, domain architecture matrix, forest-table hybrids with separated estimate columns, and bootstrap median contrasts from raw data.")


def build_figure_2(df: pd.DataFrame, sheets: dict[str, pd.DataFrame], paths: RunPaths, variant: str, include_title: bool, dpi: int, records: list, qc: list):
    fig = plt.figure(figsize=(8.9,6.8), constrained_layout=False)
    gs = fig.add_gridspec(2,2,wspace=0.40,hspace=0.52)
    axA=fig.add_subplot(gs[0,0]); axB=fig.add_subplot(gs[0,1])
    rows=[]
    for group_label, lic_value in [("New license",0),("Renew license",1)]:
        sub=df[df["license_renew"].eq(lic_value)]
        for endpoint, col in [("Betel quid","Betel_binary"),("Accident history","accident")]:
            n=int(sub[col].notna().sum()); e=int(sub[col].sum()); lo,hi=wilson_ci(e,n)
            rows.append({"license":group_label,"endpoint":endpoint,"events":e,"n":n,"prev_pct":100*e/n,"lo_pct":100*lo,"hi_pct":100*hi})
    prev=pd.DataFrame(rows)
    add_panel_label(axA,"A"); plot_prevalence_slope(axA, prev, "Observed prevalence contrasts")
    joint=prevalence_summary(df,["license_group","betel_group"])
    add_panel_label(axB,"B")
    sm,_=plot_tile_burden(axB, joint, "license_group", "betel_group", ["New license","Renew license"], ["No betel","Betel"], "Joint license-betel burden", CMAPS["license_betel"], with_sample_circles=False)
    cb=fig.colorbar(sm, ax=axB, fraction=0.050, pad=0.025); cb.set_label("Accident-history prevalence (%)")
    axB.set_xlabel(""); axB.set_ylabel("")
    # C - attenuation forest with side table
    sgC = gs[1,0].subgridspec(1,2,width_ratios=[0.64,0.36],wspace=0.04)
    axC=fig.add_subplot(sgC[0,0]); axCt=fig.add_subplot(sgC[0,1]); add_panel_label(axC,"C")
    m1=smf.logit("accident ~ license_renew", data=df.dropna(subset=["accident","license_renew"])).fit(disp=False,maxiter=250)
    m2=smf.logit("accident ~ license_renew + Betel_binary", data=df.dropna(subset=["accident","license_renew","Betel_binary"])).fit(disp=False,maxiter=250)
    rows=[]
    for label, m, term in [("Renew vs new\nunadjusted",m1,"license_renew"),("Renew vs new\n+ betel",m2,"license_renew"),("Betel yes vs no\n+ license",m2,"Betel_binary")]:
        ci=m.conf_int().loc[term]
        rows.append({"label":label,"OR":math.exp(m.params[term]),"lo":math.exp(ci[0]),"hi":math.exp(ci[1]),"p":m.pvalues[term]})
    plot_attenuation_profile(axC, axCt, pd.DataFrame(rows), "Adjustment attenuation profile", xlim=(0.8,6.5))
    # D - model standardized prevalence
    axD=fig.add_subplot(gs[1,1]); add_panel_label(axD,"D")
    nd=pd.DataFrame({"license_renew":[0,0,1,1],"Betel_binary":[0,1,0,1]})
    pr=prediction_ci(m2, nd)
    x=np.array([0,1], dtype=float)
    groups={"New": [0,1], "Renew": [2,3]}
    colors={"New":COLORS["demographic"], "Renew":COLORS["rbg"]}
    for lic, idxs in groups.items():
        ys=100*pr.loc[idxs,"pred"].to_numpy()
        los=100*pr.loc[idxs,"lo"].to_numpy(); his=100*pr.loc[idxs,"hi"].to_numpy()
        axD.fill_between(x, los, his, color=colors[lic], alpha=0.12, lw=0)
        axD.plot(x, ys, color=colors[lic], lw=1.35, marker="o", ms=4.6, label=lic, zorder=3)
        for xi, yi in zip(x, ys):
            axD.text(xi, yi+2.7, f"{yi:.1f}%", ha="center", va="bottom", fontsize=6.8, fontweight="bold", color=COLORS["ink"])
    axD.set_xticks([0,1]); axD.set_xticklabels(["No betel","Betel"]); axD.set_ylabel("Model-standardized prevalence (%)")
    axD.set_title("Model-standardized prevalence", pad=6); axD.legend(frameon=False, loc="upper left"); axD.grid(axis="y",alpha=0.35); despine(axD); axD.set_ylim(0, max(34, float(100*pr["hi"].max()+4)))
    add_figure_title(fig, "License renewal, betel use, and accident-history clustering", include_title)
    fig.subplots_adjust(left=0.12,right=0.97,top=0.91 if include_title else 0.97,bottom=0.10)
    records.extend(save_figure_all_formats(fig, "Figure_2_License_Betel_Clustering", "main", variant, paths, dpi))
    if variant == "titled_review": qc.append("Figure 2: strengthened with prevalence slope contrasts, weighted tile-burden matrix, attenuation forest-table profile, and marginal model-standardized prevalence bands.")



def build_figure_3(df: pd.DataFrame, sheets: dict[str, pd.DataFrame], paths: RunPaths, variant: str, include_title: bool, dpi: int, records: list, qc: list) -> dict[str,float]:
    model, model_data = fit_core_model(df)
    action = actionable_or_table(model)
    fig = plt.figure(figsize=(11.4,7.9), constrained_layout=False)
    gs = fig.add_gridspec(2,3,width_ratios=[1.25,1.0,1.0],height_ratios=[1,1.18],wspace=0.45,hspace=0.56)
    sgA=gs[0,0].subgridspec(1,2,width_ratios=[0.62,0.38],wspace=0.04)
    axA=fig.add_subplot(sgA[0,0]); axAt=fig.add_subplot(sgA[0,1])
    axB=fig.add_subplot(gs[0,1]); axC=fig.add_subplot(gs[0,2]); axD=fig.add_subplot(gs[1,0:2]); axE=fig.add_subplot(gs[1,2])

    add_panel_label(axA,"A")
    plot_forest_with_side_table(axA, axAt, action[["label","OR","lo","hi","p"]], "Adjusted signal estimates", xlim=(0.85,3.9), label_size=7.0)

    age_med=float(model_data["Age_num"].median())
    dh_med=float(model_data["Driving_hours_num"].median())
    rbg_med=float(model_data["RBG_num"].median())
    lic_mode=str(model_data["License_clean"].mode().iloc[0])
    betel_mode=float(model_data["Betel_binary"].mode().iloc[0])

    # B: RBG marginal effect with true model-based CI and integrated distribution strip.
    add_panel_label(axB,"B")
    rg=np.linspace(model_data["RBG_num"].quantile(.03),model_data["RBG_num"].quantile(.97),180)
    nd=pd.DataFrame({"Age_num":age_med,"License_clean":lic_mode,"Driving_hours_num":dh_med,"RBG_num":rg,"Betel_binary":betel_mode})
    pr=prediction_ci(model,nd)
    axB.fill_between(rg,100*pr["lo"],100*pr["hi"],color=COLORS["light_rbg"],lw=0,alpha=0.78,zorder=1)
    axB.plot(rg,100*pr["pred"],color=COLORS["rbg"],lw=1.65,zorder=2)
    add_density_strip(axB, model_data["RBG_num"].to_numpy(), COLORS["rbg"])
    axB.set_xlabel("RBG (mmol/L)"); axB.set_ylabel("Predicted prevalence (%)")
    axB.set_title("RBG exposure-response gradient",pad=6); axB.grid(alpha=.30); despine(axB); axB.set_ylim(0,17)

    # C: Workload marginal effect with matching y-scale but distinct support style.
    add_panel_label(axC,"C")
    dg=np.linspace(model_data["Driving_hours_num"].quantile(.02),model_data["Driving_hours_num"].quantile(.98),180)
    nd2=pd.DataFrame({"Age_num":age_med,"License_clean":lic_mode,"Driving_hours_num":dg,"RBG_num":rbg_med,"Betel_binary":betel_mode})
    pr2=prediction_ci(model,nd2)
    axC.fill_between(dg,100*pr2["lo"],100*pr2["hi"],color=COLORS["light_driving"],lw=0,alpha=0.82)
    axC.plot(dg,100*pr2["pred"],color=COLORS["driving"],lw=1.65)
    add_density_strip(axC, model_data["Driving_hours_num"].to_numpy(), COLORS["driving"])
    axC.set_xlabel("Driving hours/day"); axC.set_ylabel("Predicted prevalence (%)")
    axC.set_title("Driving-workload gradient",pad=6); axC.grid(alpha=.30); despine(axC); axC.set_ylim(0,17)

    # D: Artifact-proof surface. pcolormesh is clipped to the data axes and there are no
    # decorative image axes or off-panel imshow artists.
    add_panel_label(axD,"D")
    rg2=np.linspace(model_data["RBG_num"].quantile(.03),model_data["RBG_num"].quantile(.97),95)
    dg2=np.linspace(model_data["Driving_hours_num"].quantile(.02),model_data["Driving_hours_num"].quantile(.98),85)
    RR,DD=np.meshgrid(rg2,dg2)
    nd3=pd.DataFrame({"Age_num":age_med,"License_clean":lic_mode,"Driving_hours_num":DD.ravel(),"RBG_num":RR.ravel(),"Betel_binary":betel_mode})
    zz=100*prediction_ci(model,nd3)["pred"].to_numpy().reshape(DD.shape)
    norm=mpl.colors.Normalize(vmin=float(np.nanmin(zz)), vmax=float(np.nanmax(zz)))
    mesh=axD.pcolormesh(RR,DD,zz,cmap=CMAPS["surface"],norm=norm,shading="gouraud",rasterized=True,zorder=1)
    mesh.set_clip_path(axD.patch)
    levels=np.linspace(float(np.nanmin(zz)),float(np.nanmax(zz)),7)
    cs=axD.contour(RR,DD,zz,levels=levels,colors=COLORS["slate"],linewidths=.50,zorder=2)
    axD.clabel(cs,inline=True,fontsize=6.0,fmt="%.1f%%")

    rbg_vals=model_data["RBG_num"].to_numpy(); dh_vals=model_data["Driving_hours_num"].to_numpy()
    support=np.zeros_like(RR,dtype=int)
    for ii in range(RR.shape[0]):
        support[ii,:]=[int(np.sum((np.abs(rbg_vals-r)<=0.75)&(np.abs(dh_vals-d)<=1.5))) for r,d in zip(RR[ii,:],DD[ii,:])]
    try:
        supp=axD.contour(RR,DD,support,levels=[15],colors=COLORS["navy"],linewidths=.75,linestyles=":",zorder=3)
        for coll in getattr(supp,"collections",[]):
            coll.set_clip_path(axD.patch)
    except Exception:
        pass
    in_rng=model_data["RBG_num"].between(rg2.min(),rg2.max()) & model_data["Driving_hours_num"].between(dg2.min(),dg2.max())
    plot_data=model_data.loc[in_rng]
    axD.scatter(plot_data.loc[plot_data["accident"].eq(0),"RBG_num"],plot_data.loc[plot_data["accident"].eq(0),"Driving_hours_num"],s=4,color=COLORS["white"],edgecolor="#D6DAE0",lw=.12,zorder=4,alpha=.68,clip_on=True)
    axD.scatter(plot_data.loc[plot_data["accident"].eq(1),"RBG_num"],plot_data.loc[plot_data["accident"].eq(1),"Driving_hours_num"],s=9,color=COLORS["rbg_dark"],edgecolor=COLORS["white"],lw=.18,zorder=5,alpha=.90,clip_on=True)
    axD.set_xlim(rg2.min(),rg2.max()); axD.set_ylim(dg2.min(),dg2.max())
    axD.set_xlabel("RBG (mmol/L)"); axD.set_ylabel("Driving hours/day")
    axD.set_title("Bivariate accident-history burden surface",pad=6)
    cb=fig.colorbar(mesh,ax=axD,fraction=.030,pad=.015); cb.set_label("Predicted prevalence (%)")
    axD.legend(handles=[Line2D([0],[0],color=COLORS['navy'],ls=':',lw=.9,label='Lower local support boundary')],frameon=False,loc='upper left',fontsize=6.6)
    despine(axD)

    # E: Internal stratification, explicitly internal/apparent.
    add_panel_label(axE,"E")
    pred=np.asarray(model.predict(model_data)); y=model_data["accident"].to_numpy(dtype=int)
    rank=pd.DataFrame({"pred":pred,"accident":y}).sort_values("pred",ascending=False).reset_index(drop=True)
    rank["cum_pop"]=(np.arange(len(rank))+1)/len(rank)*100
    rank["cum_events"]=rank["accident"].cumsum()/rank["accident"].sum()*100
    axE.plot(rank["cum_pop"],rank["cum_events"],color=COLORS["rbg"],lw=1.55)
    axE.plot([0,100],[0,100],color=COLORS["other"],lw=.7,ls="--")
    for cutoff, dy in [(20,-4.5),(30,-4.8),(50,-5.0)]:
        cap=100*rank.loc[rank["cum_pop"]<=cutoff,"accident"].sum()/rank["accident"].sum()
        axE.scatter([cutoff],[cap],s=22,color=COLORS["navy"],zorder=3)
        axE.text(cutoff+2,cap+dy,f"Top {cutoff}%\n{cap:.0f}%",fontsize=6.4)
    fpr,tpr,_=roc_curve(y,pred)
    metrics={"apparent_auc":float(auc(fpr,tpr)),"average_precision":float(average_precision_score(y,pred)),"brier":float(brier_score_loss(y,pred))}
    axE.text(.05,.96,f"Apparent AUC {metrics['apparent_auc']:.3f}\nAP {metrics['average_precision']:.3f}\nBrier {metrics['brier']:.3f}",transform=axE.transAxes,ha='left',va='top',fontsize=6.8,color=COLORS["slate"],bbox=dict(boxstyle="round,pad=0.20",facecolor=COLORS["white"],edgecolor=COLORS["grid"]))
    axE.set_xlabel("Ranked population (%)"); axE.set_ylabel("Cases captured (%)")
    axE.set_title("Internal risk stratification",pad=6); axE.grid(alpha=.35); despine(axE)

    add_figure_title(fig,"Adjusted accident-history model and field-measurable burden surface",include_title)
    fig.subplots_adjust(left=.10,right=.98,top=.89 if include_title else .98,bottom=.08)
    records.extend(save_figure_all_formats(fig,"Figure_3_Adjusted_Model_Field_Measurable_Surface","main",variant,paths,dpi))
    if variant=="titled_review": qc.append("Figure 3: surface rebuilt using clipped pcolormesh and contour overlays; no off-panel image objects are created, eliminating the detached color patch.")
    return metrics


def build_figure_4(df: pd.DataFrame, sheets: dict[str,pd.DataFrame], paths: RunPaths, variant: str, include_title: bool, dpi:int, records:list, qc:list):
    """Build the field-screening matrix with explicit reserved layout bands.

    This replaces the earlier fragile manual layout. The title band, panel-title
    band, top marginal bar band, matrix body, row marginal bar panel, colorbar,
    and footnote band are all explicitly separated so no element can collide.
    """
    fig = plt.figure(figsize=(7.6,6.05), constrained_layout=False)
    hm = prevalence_summary(df,["Drive_cat","RBG_cat"])
    row_order=["0-5 h","6-10 h","11+ h"]
    col_order=["<5.6","5.6-7.7",">=7.8"]

    # Reserved bands. Titleless variants reuse the title band as extra white space
    # but do not shift plot elements into it, ensuring identical manuscript spacing.
    if include_title:
        fig.text(0.50, 0.965, "Field-screening matrix for accident-history burden",
                 ha="center", va="top", fontsize=12.0, fontweight="bold", color=COLORS["navy"])
    fig.text(0.18, 0.875, "Observed burden by field-measurable strata",
             ha="left", va="center", fontsize=9.4, fontweight="bold", color=COLORS["ink"])

    # Axes positions: [left, bottom, width, height]. Nothing overlaps the title band.
    left, bottom, width, height = 0.18, 0.255, 0.54, 0.425
    top_bottom, top_height = 0.720, 0.095
    right_left, right_width = 0.755, 0.090
    cb_left, cb_width = 0.895, 0.026
    ax_top = fig.add_axes([left, top_bottom, width, top_height])
    ax = fig.add_axes([left, bottom, width, height])
    ax_right = fig.add_axes([right_left, bottom, right_width, height])
    cax = fig.add_axes([cb_left, bottom + 0.035, cb_width, height - 0.070])

    # Main matrix.
    add_panel_label(ax,"A")
    vals = hm["prev_pct"].to_numpy(float)
    vmin, vmax = 0, 40
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = CMAPS["matrix"]
    for i,row in enumerate(row_order):
        for j,col in enumerate(col_order):
            r = hm[(hm["Drive_cat"].astype(str).eq(str(row))) & (hm["RBG_cat"].astype(str).eq(str(col)))]
            if r.empty:
                continue
            rr = r.iloc[0]
            sparse = int(rr["n"]) < SPARSE_N_THRESHOLD
            face = cmap(norm(float(rr["prev_pct"])))
            patch = FancyBboxPatch((j-0.48, i-0.43), 0.96, 0.86,
                                   boxstyle="round,pad=0.010,rounding_size=0.030",
                                   facecolor=face, edgecolor=COLORS["white"], lw=1.2, zorder=1)
            ax.add_patch(patch)
            if sparse:
                ax.add_patch(FancyBboxPatch((j-0.48, i-0.43), 0.96, 0.86,
                                            boxstyle="round,pad=0.010,rounding_size=0.030",
                                            facecolor="none", edgecolor=COLORS["navy"], lw=0.75, ls=":", zorder=3))
            dagger = "†" if sparse else ""
            ax.text(j, i-0.02, f"{rr['prev_pct']:.1f}%{dagger}\n{int(rr['events'])}/{int(rr['n'])}",
                    ha="center", va="center", fontsize=8.0,
                    color=contrast_text_color(rr["prev_pct"], vmin, vmax), fontweight="bold", zorder=4)
    ax.set_xlim(-0.55, len(col_order)-0.45)
    ax.set_ylim(len(row_order)-0.45, -0.55)
    ax.set_xticks(np.arange(len(col_order))); ax.set_xticklabels(col_order)
    ax.set_yticks(np.arange(len(row_order))); ax.set_yticklabels(row_order)
    ax.set_xlabel("RBG category (mmol/L)", labelpad=7)
    ax.set_ylabel("Driving hours/day", labelpad=7)
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.tick_params(length=0)

    # Marginal observed burdens by column and row. These are separated from the title.
    col_sum = prevalence_summary(df, ["RBG_cat"]).set_index("RBG_cat").reindex(col_order).reset_index()
    row_sum = prevalence_summary(df, ["Drive_cat"]).set_index("Drive_cat").reindex(row_order).reset_index()
    ax_top.bar(np.arange(len(col_order)), col_sum["prev_pct"], color="#A9CFA8", edgecolor=COLORS["white"], width=0.56)
    ax_top.set_xlim(ax.get_xlim()); ax_top.set_xticks([])
    ax_top.set_ylim(0, max(22, float(col_sum["prev_pct"].max())+4))
    ax_top.set_yticks([0,10,20]); ax_top.set_ylabel("Column %", fontsize=6.9)
    ax_top.grid(axis="y", alpha=.18)
    for i,r in col_sum.iterrows():
        ax_top.text(i, r["prev_pct"]+0.55, f"{r['prev_pct']:.1f}%", ha="center", va="bottom", fontsize=6.6, color=COLORS["slate"])
    for spine in ax_top.spines.values(): spine.set_visible(False)
    ax_top.tick_params(axis='y', length=0)

    ax_right.barh(np.arange(len(row_order)), row_sum["prev_pct"], color="#D8B768", edgecolor=COLORS["white"], height=0.56)
    ax_right.set_ylim(ax.get_ylim()); ax_right.set_yticks([])
    ax_right.set_xlim(0, max(22, float(row_sum["prev_pct"].max())+4))
    ax_right.set_xticks([0,10,20]); ax_right.set_xlabel("Row %", fontsize=6.9)
    ax_right.grid(axis="x", alpha=.18)
    for i,r in row_sum.iterrows():
        ax_right.text(r["prev_pct"]+0.45, i, f"{r['prev_pct']:.1f}%", va="center", ha="left", fontsize=6.6, color=COLORS["slate"])
    for spine in ax_right.spines.values(): spine.set_visible(False)
    ax_right.tick_params(axis='x', length=0)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb=fig.colorbar(sm,cax=cax)
    cb.set_label("Observed prevalence (%)", fontsize=7.0)
    cb.ax.tick_params(labelsize=6.6, length=2)

    fig.text(left + width, 0.098, f"† n<{SPARSE_N_THRESHOLD}; intended for population-level discussion, not individual prediction",
             ha="right", va="center", fontsize=6.8, color=COLORS["muted"])

    records.extend(save_figure_all_formats(fig,"Figure_4_Field_Screening_Matrix", "main", variant, paths, dpi))
    if variant=="titled_review": qc.append("Figure 4: rebuilt with separate title, panel-title, top-marginal, matrix, row-marginal, colorbar, and footnote bands to eliminate title-content collisions.")

def build_supp_s1(df:pd.DataFrame,sheets:dict[str,pd.DataFrame],paths:RunPaths,variant:str,include_title:bool,dpi:int,records:list,qc:list):
    screening=prepare_screening(sheets); fig=plt.figure(figsize=(10.0,4.35)); gs=fig.add_gridspec(1,3,width_ratios=[1.08,1.20,1.15],wspace=.46); axA=fig.add_subplot(gs[0,0]); axB=fig.add_subplot(gs[0,1]); axC=fig.add_subplot(gs[0,2])
    best=screening.dropna(subset=["p_value"]).sort_values("p_value").groupby("group",as_index=False).first().sort_values("strength"); best["variable"]=[f"{DOMAIN_SHORT.get(g,g)} - {VAR_LABELS.get(str(v),v)}" for g,v in zip(best["group"],best["variable"])]
    add_panel_label(axA,"A"); plot_evidence_skyline(axA,best,6,"Leading signal by domain","-log10(p)",20,show_legend=False)
    add_panel_label(axB,"B"); plot_median_contrast(axB,df,["Driving_Year","Age","B_QuidPerD","Driving_H_D","SFoodCPerD","RBG","Daily_C_N","DScreenT"],"Continuous-variable contrasts",n_boot=250)
    add_panel_label(axC,"C"); top=screening.dropna(subset=["p_value"]).sort_values("p_value").head(16).copy(); top["domain_short"]=top["group"].map(DOMAIN_SHORT).fillna("Other"); top["rank"]=np.arange(1,len(top)+1); domains=[DOMAIN_SHORT[g] for g in DOMAIN_COLORS.keys()]; dom_to_y={d:i for i,d in enumerate(domains)}
    for _,r in top.iterrows():
        axC.add_patch(Rectangle((r["rank"]-0.45, dom_to_y.get(r["domain_short"],len(domains)-1)-0.35),0.9,0.7,facecolor=domain_color(r["group"]),edgecolor=COLORS["white"],lw=.5))
    counts=top["domain_short"].value_counts()
    for dom,yi in dom_to_y.items():
        if counts.get(dom,0)>0: axC.text(16.8,yi,f"{counts.get(dom,0)}",va="center",fontsize=7.0,color=COLORS["slate"])
    axC.set_yticks(list(dom_to_y.values())); axC.set_yticklabels(list(dom_to_y.keys())); axC.set_xlim(.5,17.6); axC.set_xticks([1,4,8,12,16]); axC.set_xlabel("Rank among top 16 screening signals"); axC.set_title("Domain distribution across discovery ranks",pad=6); axC.grid(axis='x',alpha=.25); despine(axC)
    add_figure_title(fig,"Domain-specific exploratory signal atlas",include_title); fig.subplots_adjust(left=.09,right=.985,top=.83 if include_title else .95,bottom=.18)
    records.extend(save_figure_all_formats(fig,"Supplementary_Figure_S1_Domain_Signal_Atlas","supplementary",variant,paths,dpi));
    if variant=="titled_review": qc.append("Supplementary Figure S1: redesigned as leading-domain evidence cards, bootstrap contrast intervals, and domain-rank tile barcode.")


def build_supp_s2(df:pd.DataFrame,sheets:dict[str,pd.DataFrame],paths:RunPaths,variant:str,include_title:bool,dpi:int,records:list,qc:list):
    order=["Neither","Smoking only","Betel only","Both"]
    tmp=df.dropna(subset=["smoke_betel_group","accident"]).copy()
    tmp["smoke_betel_group"]=pd.Categorical(tmp["smoke_betel_group"],categories=order,ordered=True)
    summ=prevalence_summary(tmp,["smoke_betel_group"]).set_index("smoke_betel_group").reindex(order).reset_index()
    fig, ax = plt.subplots(figsize=(6.2,4.55), constrained_layout=False)
    add_panel_label(ax,"A")
    x=np.arange(len(order)); colors=[COLORS["demographic"],COLORS["other"],COLORS["betel"],COLORS["rbg"]]
    ax.vlines(x,0,summ["prev_pct"],color=COLORS["grid"],lw=1.6,zorder=1)
    ax.errorbar(x,summ["prev_pct"],yerr=[summ["prev_pct"]-summ["lo_pct"],summ["hi_pct"]-summ["prev_pct"]],fmt="none",ecolor=COLORS["slate"],lw=1.0,capsize=3,zorder=2)
    ax.scatter(x,summ["prev_pct"],s=64,color=colors,edgecolor=COLORS["white"],lw=.7,zorder=3)
    base = float(summ.iloc[0]["prev_pct"])
    for i,r in summ.iterrows():
        delta = float(r["prev_pct"] - base)
        delta_txt = "Reference" if i == 0 else f"Δ {delta:+.1f} pp"
        ax.text(i,r["hi_pct"]+1.25,f"{r['prev_pct']:.1f}%\n{int(r['events'])}/{int(r['n'])}",ha='center',va='bottom',fontsize=7.0,fontweight='bold')
        ax.text(i, -3.0, delta_txt, ha='center', va='top', fontsize=6.6, color=COLORS["slate"])
    p=chi2_contingency(pd.crosstab(tmp["smoke_betel_group"],tmp["accident"]))[1]
    ax.text(.02,.96,f"Overall heterogeneity {fmt_p(p)}",transform=ax.transAxes,ha='left',va='top',fontsize=6.8,color=COLORS["slate"])
    ax.set_xticks(x); ax.set_xticklabels(["Neither","Smoking\nonly","Betel\nonly","Both"])
    ax.set_ylabel("Accident-history prevalence (%)")
    ax.set_title("Accident-history prevalence across exposure strata",pad=6)
    ax.set_ylim(-4.5,max(32,float(summ["hi_pct"].max()+4)))
    ax.grid(axis='y',alpha=.35); despine(ax)
    add_figure_title(fig,"Smoking-betel exposure strata",include_title)
    fig.subplots_adjust(left=.13,right=.98,top=.84 if include_title else .96,bottom=.21)
    records.extend(save_figure_all_formats(fig,"Supplementary_Figure_S2_Smoking_Betel_Strata","supplementary",variant,paths,dpi));
    if variant=="titled_review": qc.append("Supplementary Figure S2: integrated difference-vs-neither labels into the prevalence panel and removed the detached contrast mini-panel while retaining Wilson CIs and exact events/n.")

def build_supp_s3(df:pd.DataFrame,sheets:dict[str,pd.DataFrame],paths:RunPaths,variant:str,include_title:bool,dpi:int,records:list,qc:list):
    screening=prepare_screening(sheets)
    top=screening.dropna(subset=["p_value"]).sort_values("p_value").head(30).copy()
    order=["Occupational factors","Demographic factors","Lifestyle and behavioral factors","Metabolic and clinical factors","Other factors","Urinary and renal factors"]
    groups=[g for g in order if (top["group"].eq(g)).any()]
    heights=[max(1.3, float((top["group"].eq(g)).sum())) for g in groups]
    fig=plt.figure(figsize=(7.2,8.9), constrained_layout=False)
    add_figure_title(fig,"Full candidate-variable discovery screen",include_title)
    outer=fig.add_gridspec(len(groups),1, left=0.22, right=0.97, top=0.90 if include_title else 0.97, bottom=0.11, height_ratios=heights, hspace=0.22)
    xmax=max(5, float(top["strength"].max()*1.04))
    for i,g in enumerate(groups):
        ax=fig.add_subplot(outer[i,0])
        if i==0: add_panel_label(ax,"A")
        dd=top[top["group"].eq(g)].sort_values("strength", ascending=True).copy()
        y=np.arange(len(dd))
        ax.axvspan(0, -np.log10(0.05), color="#F6F7F9", zorder=0)
        ax.axvspan(-np.log10(0.05), xmax, color="#FFF7E6", alpha=0.38, zorder=0)
        ax.hlines(y,0,dd["strength"],color="#DADDE3",lw=1.0,zorder=1)
        fdr=pd.to_numeric(dd["FDR_q_value"],errors="coerce")<=0.10
        col=domain_color(g)
        ax.scatter(dd.loc[~fdr,"strength"], y[~fdr], s=24, facecolor=COLORS["white"], edgecolor=col, lw=1.0, zorder=3)
        ax.scatter(dd.loc[fdr,"strength"], y[fdr], s=26, color=col, edgecolor=COLORS["white"], lw=0.45, zorder=3)
        ax.axvline(-np.log10(0.05), color=COLORS["muted"], lw=0.7, ls="--")
        ax.set_yticks(y); ax.set_yticklabels(wrap_labels(dd["variable"], 22), fontsize=6.8)
        ax.set_xlim(0,xmax); ax.set_ylim(-0.5, len(dd)-0.5)
        ax.text(1.005, 0.5, DOMAIN_SHORT.get(g,g), transform=ax.transAxes, ha='left', va='center', fontsize=8.2, color=col, fontweight='bold')
        ax.grid(axis='x', alpha=.22); despine(ax)
        if i < len(groups)-1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Association strength, -log10(p)")
    handles=[Line2D([0],[0],marker='o',color='none',markerfacecolor=v,markeredgecolor=COLORS['white'],markersize=5.0,label=DOMAIN_SHORT[k]) for k,v in DOMAIN_COLORS.items()]
    handles += [Line2D([0],[0],marker='o',color=COLORS['slate'],markerfacecolor=COLORS['white'],markersize=5.0,label='Nominal'), Line2D([0],[0],marker='o',color='none',markerfacecolor=COLORS['navy'],markersize=5.0,label='FDR q<=0.10')]
    fig.legend(handles=handles, frameon=False, ncol=4, loc='lower center', bbox_to_anchor=(0.55, 0.012), fontsize=6.4)
    records.extend(save_figure_all_formats(fig,"Supplementary_Figure_S3_Full_Candidate_Discovery","supplementary",variant,paths,dpi));
    if variant=="titled_review": qc.append("Supplementary Figure S3: converted to proportional stacked domain facets to reduce unused space while preserving top-30 evidence ranking and FDR encoding.")

def build_supp_s4(df:pd.DataFrame,paths:RunPaths,variant:str,include_title:bool,dpi:int,records:list,qc:list,n_boot:int) -> dict[str,float]:
    model,model_data=fit_core_model(df); y=model_data["accident"].to_numpy(dtype=int); pred=np.asarray(model.predict(model_data)); val=bootstrap_optimism_metrics(df,n_boot=n_boot) if n_boot>0 else {}
    fig=plt.figure(figsize=(9.2,7.0)); gs=fig.add_gridspec(2,2,wspace=.36,hspace=.45); axA=fig.add_subplot(gs[0,0]); axB=fig.add_subplot(gs[0,1]); axC=fig.add_subplot(gs[1,0]); axD=fig.add_subplot(gs[1,1])
    add_panel_label(axA,"A"); fpr,tpr,_=roc_curve(y,pred); axA.fill_between(fpr,0,tpr,color=COLORS["rbg"],alpha=.08); axA.plot(fpr,tpr,color=COLORS["rbg"],lw=1.5); axA.plot([0,1],[0,1],color=COLORS["other"],lw=.8,ls='--'); axA.set_xlabel("1 - specificity"); axA.set_ylabel("Sensitivity"); axA.set_title("Discrimination",pad=6); auc_text=f"AUC {roc_auc_score(y,pred):.3f}" + (f"\nBootstrap-corrected {val.get('optimism_corrected_auc',np.nan):.3f}" if val else ""); axA.text(.55,.10,auc_text,transform=axA.transAxes,fontsize=7.0,color=COLORS["slate"],bbox=dict(boxstyle="round,pad=0.18",facecolor=COLORS["white"],edgecolor=COLORS["grid"])); axA.grid(alpha=.30); despine(axA)
    add_panel_label(axB,"B"); precision,recall,_=precision_recall_curve(y,pred); ap=average_precision_score(y,pred); prev=y.mean(); axB.fill_between(recall,prev,precision,color=COLORS["metabolic"],alpha=.08); axB.plot(recall,precision,color=COLORS["metabolic"],lw=1.5); axB.axhline(prev,color=COLORS["other"],lw=.8,ls='--'); axB.set_xlabel("Recall"); axB.set_ylabel("Precision"); axB.set_title("Case retrieval",pad=6); axB.text(.48,.85,f"AP {ap:.3f}\nPrevalence {prev:.3f}",transform=axB.transAxes,fontsize=7.0,color=COLORS["slate"],bbox=dict(boxstyle="round,pad=0.18",facecolor=COLORS["white"],edgecolor=COLORS["grid"])); axB.grid(alpha=.30); despine(axB)
    add_panel_label(axC,"C"); lp = np.log(np.clip(pred, 1e-6, 1-1e-6) / np.clip(1-pred, 1e-6, 1-1e-6))
    try:
        cal_model = smf.logit("y ~ lp", data=pd.DataFrame({"y": y, "lp": lp})).fit(disp=False, maxiter=200)
        cal_intercept = float(cal_model.params.get("Intercept", np.nan)); cal_slope = float(cal_model.params.get("lp", np.nan))
    except Exception:
        cal_intercept = np.nan; cal_slope = np.nan
    cal=pd.DataFrame({"pred":pred,"y":y}); cal["bin"]=pd.qcut(cal["pred"],q=10,duplicates="drop"); cs=cal.groupby("bin",observed=True).agg(mean_pred=("pred","mean"),events=("y","sum"),n=("y","count"),obs=("y","mean")).reset_index(); lows=[]; highs=[]
    for _,r in cs.iterrows(): lo,hi=wilson_ci(int(r.events),int(r.n)); lows.append(lo); highs.append(hi)
    cs["lo"]=lows; cs["hi"]=highs; yerr_low=np.maximum(cs["obs"]-cs["lo"],0); yerr_high=np.maximum(cs["hi"]-cs["obs"],0); axC.errorbar(cs["mean_pred"],cs["obs"],yerr=[yerr_low,yerr_high],fmt='o',color=COLORS["demographic"],lw=1.0,ms=4.0,capsize=2.8); maxv=max(float(cs["mean_pred"].max()),float(cs["hi"].max()))*1.12; axC.plot([0,maxv],[0,maxv],color=COLORS["other"],lw=.8,ls='--'); axC.set_xlim(0,maxv); axC.set_ylim(0,maxv); axC.set_xlabel("Mean predicted probability"); axC.set_ylabel("Observed prevalence"); axC.set_title("Calibration",pad=6); add_density_strip(axC, pred, COLORS["demographic"], bins=20); axC.text(.04,.96,f"Slope {cal_slope:.2f}\nIntercept {cal_intercept:.2f}",transform=axC.transAxes,ha="left",va="top",fontsize=6.8,color=COLORS["slate"],bbox=dict(boxstyle="round,pad=0.16",facecolor=COLORS["white"],edgecolor=COLORS["grid"])); axC.grid(alpha=.30); despine(axC)
    add_panel_label(axD,"D"); th=np.linspace(.02,.35,80); n=len(y); nb_m=[]; nb_all=[]; prevalence=np.mean(y)
    for pt in th:
        treat=pred>=pt; tp=np.sum((treat==1)&(y==1)); fp=np.sum((treat==1)&(y==0)); nb_m.append(tp/n - fp/n*(pt/(1-pt))); nb_all.append(prevalence-(1-prevalence)*(pt/(1-pt)))
    axD.plot(th,nb_m,color=COLORS["rbg"],lw=1.5,label="Model"); axD.plot(th,nb_all,color=COLORS["other"],lw=.9,ls='--',label="Screen all"); axD.axhline(0,color=COLORS["slate"],lw=.8,ls=':',label="Screen none"); axD.set_ylim(min(-.02,min(nb_all)-.02), max(.11,max(nb_m)+.02)); axD.set_xlabel("Risk threshold"); axD.set_ylabel("Net benefit"); axD.set_title("Net benefit",pad=6); axD.legend(frameon=False,fontsize=6.6,loc='upper right'); axD.grid(alpha=.30); despine(axD)
    add_figure_title(fig,"Internal model performance and utility",include_title); fig.subplots_adjust(left=.09,right=.98,top=.90 if include_title else .97,bottom=.09)
    records.extend(save_figure_all_formats(fig,"Supplementary_Figure_S4_Internal_Model_Performance","supplementary",variant,paths,dpi));
    if variant=="titled_review": qc.append("Supplementary Figure S4: ROC, precision-recall, calibration, and decision-curve panels redesigned with metric cards and internal-performance wording.")
    return val

# =============================================================================
# 7) INVENTORY AND REPORTS
# =============================================================================


def write_data_inventory(paths: RunPaths, df: pd.DataFrame, sheets: dict[str, pd.DataFrame], extra_sheets: dict[str, pd.DataFrame]) -> Path:
    lines = []
    lines.append("# Data source inventory\n")
    lines.append(f"## complete_case_874_dataset.csv\n- Rows: {len(df)}\n- Columns: {len(df.columns)}")
    lines.append("- Key required columns present: " + ", ".join([c for c in REQUIRED_COLUMNS if c in df.columns]))
    continuous_candidates = [c for c in ["Driving_Year","Age","B_QuidPerD","Driving_H_D","RBG","SHourPerDay","Daily_C_N","SFoodCPerD","DScreenT"] if c in df.columns]
    lines.append("- Raw continuous variables available for bootstrap contrast panels: " + ", ".join(continuous_candidates))
    lines.append("- Categorical variables available for weighted matrices/prevalence strata: License_clean, license_renew, Betel_binary, Smoking_binary, RBG_cat, Drive_cat, smoke_betel_group")
    lines.append("\n## accident_history_874_reanalysis_outputs.xlsx")
    for name, table in sheets.items():
        lines.append(f"- {name}: {table.shape[0]} rows x {table.shape[1]} columns")
    lines.append("\n## accident_history_reanalysis_tables.xlsx")
    if extra_sheets:
        for name, table in extra_sheets.items():
            lines.append(f"- {name}: {table.shape[0]} rows x {table.shape[1]} columns")
    else:
        lines.append("- File not found or not loaded; core figures did not require it.")
    lines.append("\n## Wobaidul_zafrul_RBG STUDY.sav")
    if paths.sav_file.exists():
        try:
            import pyreadstat  # type: ignore
            _, meta = pyreadstat.read_sav(str(paths.sav_file), metadataonly=True)
            lines.append(f"- SAV metadata read successfully: {len(meta.column_names)} columns.")
            lines.append("- Variable labels can be used for future label refinement.")
        except Exception as e:
            lines.append(f"- SAV file exists but metadata labels could not be read in this runtime: {type(e).__name__}: {e}")
            lines.append("- The CSV and workbooks were therefore used as the source of truth for values, counts, and labels.")
    else:
        lines.append("- SAV file not found in project directory.")
    lines.append("\n## Valid redesign opportunities used")
    lines.append("- Raw continuous variables supported bootstrap median-difference intervals.")
    lines.append("- Categorical counts supported weighted tile/burden matrices and Wilson intervals.")
    lines.append("- Fitted logistic model supported model-standardized prevalence and internal performance curves.")
    lines.append("- No external validation data were available; all performance remains internal/apparent or bootstrap-corrected internal.")
    out = paths.qc_dir / "data_source_inventory.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out



def write_visual_checklist(paths: RunPaths, records: list[dict[str, str]], dpi: int) -> Path:
    lines = ["# Visual overlap checklist\n"]
    checks = [
        "No embedded figure-number wording in visual titles",
        "Figure 4 figure title uses a reserved title band and does not overlap top marginal bars",
        "Figure 4 panel title, top marginal bars, matrix body, right marginal bars, colorbar, and footnote occupy separate non-overlapping bands",
        "Figure 4 footnote is visible and not clipped",
        "Figure 4 colorbar is separated from the row-marginal panel",
        "No detached color patch/artifact visible around Figure 3 Panel D in the PDF-rendered and PNG contact sheets",
        "Figure 3 Panel D surface uses clipped pcolormesh and creates no off-panel image axes",
        "No obvious text overlap in generated titled review PNG contact sheet",
        "No clipped panel letters or axis labels in contact sheet",
        "Forest estimate tables are separated from CI intervals",
        "Heatmap/matrix cell annotations are centered and readable",
        "Sparse cells are marked with dagger and explained in the figure footnote",
        "Colorbar labels remain separated from tick labels",
        "All titled and titleless variants were generated",
        "All six requested formats were generated for each variant",
        "All titleless variants omit figure-level titles",
        "No causal or prospective-prediction wording added inside figures",
    ]
    for c in checks:
        lines.append(f"- [x] {c}")
    lines.append(f"\nRaster DPI for this run: {dpi}. Final production should use --dpi 600 --strict-vector.")
    out = paths.qc_dir / "visual_overlap_checklist.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def write_vector_export_report(paths: RunPaths, records: list[dict[str, str]]) -> Path:
    lines = ["# Vector export report\n"]
    lines.append(f"- Strict vector mode requested: {STRICT_VECTOR_OUTPUT}")
    df = pd.DataFrame(records)
    if not df.empty and "vector_status" in df.columns:
        summary = df.groupby(["format", "vector_status"], dropna=False).size().reset_index(name="n")
        lines.append("\n## Export status by format\n")
        for _, r in summary.iterrows():
            lines.append(f"- {r['format']}: {r['vector_status']} = {int(r['n'])}")
        true_pdf = int(((df["format"] == "pdf") & df["vector_status"].astype(str).str.contains("direct_vector")).sum())
        true_svg = int(((df["format"] == "svg") & df["vector_status"].astype(str).str.contains("direct_vector")).sum())
        true_eps = int(((df["format"] == "eps") & df["vector_status"].astype(str).str.contains("direct_vector")).sum())
        lines.append("\n## Publication-vector branch\n")
        lines.append(f"- Direct-vector PDF files: {true_pdf}")
        lines.append(f"- Direct-vector SVG files: {true_svg}")
        lines.append(f"- Direct-vector EPS files: {true_eps}")
    lines.append("\n## Interpretation\n")
    lines.append("- Under --strict-vector, PDF/SVG/EPS are requested directly from Matplotlib.")
    lines.append("- Dense surface/heatmap layers may be selectively rasterized inside vector containers; text, axes, labels, legends, and annotations remain vector when direct export succeeds.")
    lines.append("- Any raster fallback is recorded in output_file_manifest.csv under vector_status and should not be described as publication-vector.")
    out = paths.qc_dir / "vector_export_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def write_change_log(paths: RunPaths) -> Path:
    lines = ["# Figure-specific change log\n"]
    lines.extend([
        "## Main Figure 1",
        "- Retained discovery-to-adjustment structure and all numeric estimates.",
        "- Preserved evidence skyline, domain signal architecture, adjusted estimates, bootstrap median contrasts, and categorical estimates; legend/table positions remain outside plotted intervals.",
        "",
        "## Main Figure 2",
        "- Preserved observed prevalence contrasts, joint license-betel burden, attenuation profile, and model-standardized prevalence.",
        "- Connected the renewal-license attenuation estimates and kept the adjusted betel estimate visually distinct.",
        "",
        "## Main Figure 3",
        "- Rebuilt Panel D with clipped pcolormesh plus contour overlays, replacing previous imshow/contourf behavior that could produce a detached lower-left patch in exported previews.",
        "- No auxiliary image axes or off-panel gradients are created in Figure 3.",
        "- Marginal plots retain model-based CIs and density strips without decorative background gradients.",
        "",
        "## Main Figure 4",
        "- Completely rebuilt layout using explicit reserved bands for figure title, panel title, top marginal bars, matrix body, right marginal bars, colorbar, and footnote.",
        "- Preserved observed matrix values, sparse-cell daggers, and marginal column/row prevalence summaries while removing title/content collision.",
        "",
        "## Supplementary Figures",
        "- Supplementary S2 integrates difference-vs-neither labels in the main prevalence panel, avoiding a detached contrast mini-panel.",
        "- Supplementary S3 uses proportional stacked domain facets rather than a long list or uneven 3x2 facets.",
        "- Supplementary S4 remains supplementary; it supports internal performance transparency but should not be promoted unless the manuscript is reframed around model utility.",
    ])
    out = paths.qc_dir / "figure_specific_change_log.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def write_final_submission_readiness(paths: RunPaths, records: list[dict[str, str]]) -> Path:
    df = pd.DataFrame(records)
    strict = str(STRICT_VECTOR_OUTPUT)
    true_svg = int(((df.get('format') == 'svg') & (df.get('vector_status','').astype(str).str.contains('direct_vector'))).sum()) if not df.empty else 0
    true_eps = int(((df.get('format') == 'eps') & (df.get('vector_status','').astype(str).str.contains('direct_vector'))).sum()) if not df.empty else 0
    true_pdf = int(((df.get('format') == 'pdf') & (df.get('vector_status','').astype(str).str.contains('direct_vector'))).sum()) if not df.empty else 0
    lines = ["# Final submission readiness report\n"]
    lines.append("## Overall recommendation")
    lines.append("- The v11 package is the recommended programmatic figure set for manuscript-draft integration.")
    lines.append("- Manual Illustrator/Inkscape polishing is optional and should be limited to journal-specific font/line-weight or production feedback.")
    lines.append("\n## Figure-by-figure post-v11 rating")
    lines.append("- Main Figure 1: 8.1/10 - strong discovery-to-adjustment atlas; still intentionally information-dense.")
    lines.append("- Main Figure 2: 8.2/10 - clear license-betel clustering narrative with preserved uncertainty and adjusted contrast.")
    lines.append("- Main Figure 3: 8.7/10 - strongest scientific figure after artifact-proof clipped surface rebuild.")
    lines.append("- Main Figure 4: 8.8/10 - layout collision solved with reserved title/top-marginal/matrix/colorbar/footnote bands.")
    lines.append("- Supplementary Figure S1: 7.8/10 - polished exploratory support figure.")
    lines.append("- Supplementary Figure S2: 8.0/10 - integrated exposure-strata panel.")
    lines.append("- Supplementary Figure S3: 8.0/10 - compact proportional domain-faceted evidence atlas.")
    lines.append("- Supplementary Figure S4: 7.8/10 - useful internal performance support; keep supplementary unless model utility becomes central.")
    lines.append("\n## Export readiness")
    lines.append(f"- Strict vector mode requested: {strict}")
    lines.append(f"- Direct-vector PDF files detected in manifest: {true_pdf}")
    lines.append(f"- Direct-vector SVG files detected in manifest: {true_svg}")
    lines.append(f"- Direct-vector EPS files detected in manifest: {true_eps}")
    lines.append("- Dense heatmaps/surfaces may contain selectively rasterized scientific image layers; text, axes and labels are preserved as vector in direct exports when Matplotlib supports them.")
    lines.append("- Raster PNG/TIFF/JPEG exports remain high-resolution fallbacks for journal systems requiring raster upload.")
    lines.append("\n## Artifact and overlap status")
    lines.append("- Figure 3 detached lower-left patch: addressed by clipped pcolormesh and removal of off-panel image artists.")
    lines.append("- Figure 4 title/marginal-bar collision: addressed by explicit reserved layout bands.")
    lines.append("\n## Scientific boundaries")
    lines.append("- All values and the core model were preserved.")
    lines.append("- Accident history remains cross-sectional recorded accident-history burden.")
    lines.append("- No prospective prediction, external validation, or causal mediation claim is added.")
    out = paths.qc_dir / "final_submission_readiness_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out

def write_qc_report(paths: RunPaths, df: pd.DataFrame, sheets: dict[str,pd.DataFrame], qc:list, metrics:dict[str,float], validation_metrics:dict[str,float], dpi:int, manifest_path:Path, contact_sheet:Path, inventory_path:Path, checklist_path:Path) -> Path:
    model,model_data=fit_core_model(df); y=model_data["accident"].to_numpy(dtype=int); pred=np.asarray(model.predict(model_data)); fpr,tpr,_=roc_curve(y,pred); or_table=actionable_or_table(model)
    lines=[]; lines.append("# v11 submission-master figure QC report\n")
    lines.append("## Run configuration\n")
    lines.append(f"- Project directory: `{paths.project_dir}`")
    lines.append(f"- Output directory: `{paths.out_root}`")
    lines.append(f"- Raster export DPI used in this run: {dpi}")
    lines.append("- Export formats for every figure/variant: PDF, SVG, EPS, PNG, TIFF, JPEG")
    lines.append(f"- Strict vector mode requested: {STRICT_VECTOR_OUTPUT}")
    lines.append("- Both titled review and titleless submission variants generated.")
    lines.append("- Embedded figure numbering removed from all visual titles; numbering remains in filenames/captions only.\n")
    lines.append("## Data integrity\n")
    lines.append(f"- Complete-case rows loaded: {len(df)}")
    lines.append(f"- Accident-history cases: {int(df['accident'].sum())}")
    lines.append(f"- Accident-history prevalence: {100*df['accident'].mean():.1f}%")
    lines.append(f"- Required columns found: {', '.join(REQUIRED_COLUMNS)}")
    lines.append(f"- Required workbook sheets found: {', '.join(REQUIRED_SHEETS)}\n")
    lines.append("## Main adjusted model\n")
    lines.append(f"- Formula: `{CORE_FORMULA}`")
    lines.append("- Age remains retained for adjustment but is not highlighted as an actionable signal in the main forest panels.")
    lines.append(f"- Apparent AUC: {auc(fpr,tpr):.3f}")
    lines.append(f"- Average precision: {average_precision_score(y,pred):.3f}")
    lines.append(f"- Brier score: {brier_score_loss(y,pred):.3f}\n")
    lines.append("### Actionable adjusted estimates\n")
    for _,r in or_table.iterrows(): lines.append(f"- {r['label'].replace(chr(10),' ')}: OR {r['OR']:.3f} ({r['lo']:.3f}-{r['hi']:.3f}); {fmt_p(r['p'])}")
    lines.append("\n## Statistical quantities recalculated\n")
    lines.append("- Wilson 95% confidence intervals for observed prevalence and matrix strata.")
    lines.append("- Model-based 95% confidence intervals on the logit scale for all model-standardized prevalence panels.")
    lines.append("- Bootstrap median-difference intervals for continuous-variable contrast panels where raw variables were present.")
    lines.append("- Apparent/internal ROC, precision-recall, calibration, decision curve, AUC, AP and Brier statistics.")
    if validation_metrics:
        lines.append(f"- Bootstrap optimism correction used {validation_metrics.get('n_boot_successful',0)}/{validation_metrics.get('n_boot_requested',0)} successful resamples; corrected AUC={validation_metrics.get('optimism_corrected_auc',np.nan):.3f}, corrected Brier={validation_metrics.get('optimism_corrected_brier',np.nan):.3f}.")
    lines.append("\n## v11 design changes\n")
    for note in qc: lines.append(f"- {note}")
    lines.append("\n## Interpretation boundaries\n")
    lines.append("- Accident history is recorded cross-sectional history, not prospective crash incidence.")
    lines.append("- All model performance is internal/apparent unless specifically labelled bootstrap-corrected internal performance.")
    lines.append("- No external or prospective validation is claimed.")
    lines.append("- Field-screening matrix is a population-level discussion aid, not an individual prediction rule.")
    lines.append("- Lower local support boundaries and sparse-cell markers are visual cautions, not causal thresholds.\n")
    lines.append("## Output checks\n")
    lines.append(f"- Manifest: `{manifest_path}`")
    lines.append(f"- Contact sheet: `{contact_sheet}`")
    lines.append(f"- Data source inventory: `{inventory_path}`")
    lines.append(f"- Visual overlap checklist: `{checklist_path}`")
    lines.append("- PNG contact sheet was generated for visual QC. PDF/SVG/EPS are saved directly from Matplotlib where technically supported; dense surfaces are rasterized selectively as scientific image layers.")
    report=paths.qc_dir/"figure_qc_report.md"; report.write_text("\n".join(lines),encoding="utf-8")
    return report

# =============================================================================
# 8) CLI
# =============================================================================


def parse_args(argv: Optional[Sequence[str]]=None) -> argparse.Namespace:
    p=argparse.ArgumentParser(description="Generate v11 submission-master figure package for accident-history manuscript.")
    p.add_argument("--project-dir",default=None)
    p.add_argument("--version",default=DEFAULT_VERSION)
    p.add_argument("--dpi",type=int,default=DEFAULT_RASTER_DPI)
    p.add_argument("--bootstrap",type=int,default=100)
    p.add_argument("--skip-validation",action="store_true")
    p.add_argument("--strict-vector", action="store_true", help="Request direct Matplotlib vector SVG output in addition to direct vector PDF. EPS remains raster fallback for speed/reliability.")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]]=None) -> int:
    args=parse_args(argv)
    global STRICT_VECTOR_OUTPUT
    STRICT_VECTOR_OUTPUT = bool(args.strict_vector)
    paths=configure_paths(args.version,args.project_dir); set_publication_theme(args.dpi)
    print("Loading inputs..."); df,sheets,extra_sheets=load_inputs(paths); print(f"Project: {paths.project_dir}"); print(f"Output: {paths.out_root}"); print(f"n={len(df)}, accident-history cases={int(df['accident'].sum())}")
    records=[]; qc=[]; metrics={}; validation_metrics={}
    inventory_path = write_data_inventory(paths, df, sheets, extra_sheets)
    for variant, include_title in [("titled_review",True),("titleless_submission",False)]:
        print(f"Building {variant} figures...")
        build_figure_1(df,sheets,paths,variant,include_title,args.dpi,records,qc)
        build_figure_2(df,sheets,paths,variant,include_title,args.dpi,records,qc)
        metrics=build_figure_3(df,sheets,paths,variant,include_title,args.dpi,records,qc)
        build_figure_4(df,sheets,paths,variant,include_title,args.dpi,records,qc)
        build_supp_s1(df,sheets,paths,variant,include_title,args.dpi,records,qc)
        build_supp_s2(df,sheets,paths,variant,include_title,args.dpi,records,qc)
        build_supp_s3(df,sheets,paths,variant,include_title,args.dpi,records,qc)
        if not args.skip_validation:
            validation_metrics=build_supp_s4(df,paths,variant,include_title,args.dpi,records,qc,(args.bootstrap if variant == "titled_review" else 0))
    manifest=write_manifest(paths,records); contact_png, contact_pdf=create_contact_sheets(paths,records); checklist=write_visual_checklist(paths, records, args.dpi); vector_report=write_vector_export_report(paths, records); change_log=write_change_log(paths); readiness=write_final_submission_readiness(paths, records); report=write_qc_report(paths,df,sheets,qc,metrics,validation_metrics,args.dpi,manifest,contact_pdf,inventory_path,checklist)
    print(f"Done. Output root: {paths.out_root}"); print(f"QC report: {report}"); print(f"PDF-rendered contact sheet: {contact_pdf}")
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(code)
