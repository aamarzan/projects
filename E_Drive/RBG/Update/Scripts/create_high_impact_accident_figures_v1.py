#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
10/10 journal-grade multi-panel figures for accident-history re-analysis
=======================================================================

This script creates advanced, multi-panel, manuscript-quality figures for the
professional-driver accident-history re-analysis. It is intentionally NOT a
simple graphical copy of the tables. Each figure answers a distinct scientific
question:

Figure 1. Complete-case validity and outcome architecture
Figure 2. Multidimensional accident phenotype and separability
Figure 3. Adjusted inference, uncertainty, and nonlinear risk surface
Figure 4. Predictive performance, calibration, and clinical utility
Figure 5. Actionable risk stratification and occupational-health translation

Default Windows project structure
---------------------------------
E:\RBG\Update
│   Wobaidul_zafrul_RBG STUDY.sav
│   accident_history_reanalysis_tables.xlsx
├───Figures
└───Scripts
        create_10of10_accident_figures.py

Run from PowerShell:
    cd E:\RBG\Update
    python .\Scripts\create_10of10_accident_figures.py

Install dependencies if needed:
    python -m pip install pandas numpy matplotlib scipy statsmodels scikit-learn openpyxl pyreadstat

Outputs
-------
High-resolution PNG, PDF, and SVG files saved to:
    E:\RBG\Update\Figures\HighImpact_10of10

Statistical note
----------------
All accident-history figures are complete-case analyses because accident history
is missing for a large proportion of the raw SPSS dataset. This should be stated
in the manuscript.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import math
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import gridspec
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
from scipy import stats
from scipy.special import expit
import statsmodels.formula.api as smf
from patsy import build_design_matrices
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# 1) USER SETTINGS
# =============================================================================

PROJECT_DIR = Path(r"E:\RBG\Update")
DATA_SAV = PROJECT_DIR / "Wobaidul_zafrul_RBG STUDY.sav"
TABLE_XLSX = PROJECT_DIR / "accident_history_reanalysis_tables.xlsx"
OUT_DIR = PROJECT_DIR / "Figures" / "HighImpact_10of10"

DPI = 600
BOOT_N = 300          # used for coefficient-stability panel; increase to 1000 for final manuscript if time allows
RANDOM_STATE = 20260505

# Premium scientific palette: restrained, color-blind conscious
NAVY = "#071E3D"
INK = "#111827"
BLUE = "#2F6F9F"
TEAL = "#2A9D8F"
GOLD = "#D99A2B"
CRIMSON = "#B23A48"
PURPLE = "#6D5BD0"
GREY = "#6B7280"
LIGHT = "#F7F8FB"
GRID = "#E5E7EB"
WHITE = "#FFFFFF"

PANEL_BG = "#FBFCFF"

# =============================================================================
# 2) GLOBAL STYLE AND BASIC HELPERS
# =============================================================================

def set_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.titlesize": 12,
        "axes.labelsize": 10.5,
        "axes.titleweight": "bold",
        "axes.edgecolor": "#1F2937",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.5,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def savefig(fig: plt.Figure, stem: str) -> None:
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT_DIR / f"{stem}.{ext}", bbox_inches="tight", dpi=DPI, facecolor=WHITE)
    plt.close(fig)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.10, 1.08, label, transform=ax.transAxes, ha="left", va="top",
            fontsize=14, fontweight="bold", color=NAVY)


def clean_label(x) -> str:
    if pd.isna(x):
        return "Missing"
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def norm_name(x: str) -> str:
    return re.sub(r"[^a-z0-9]", "", x.lower())


def find_col(df: pd.DataFrame, candidates: Sequence[str], required: bool = True) -> Optional[str]:
    norm_map = {norm_name(c): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        key = norm_name(cand)
        if key in norm_map:
            return norm_map[key]
    # also allow partial matching for common names
    for cand in candidates:
        key = norm_name(cand)
        hits = [orig for nk, orig in norm_map.items() if key in nk or nk in key]
        if hits:
            return hits[0]
    if required:
        raise KeyError(f"Could not find column. Tried {candidates}. Available: {list(df.columns)[:80]}")
    return None


def p_text(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "p<0.001"
    return f"p={p:.3f}"


def ci_binomial_wilson(x: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    phat = x / n
    denom = 1 + z**2 / n
    centre = phat + z**2 / (2*n)
    half = z * math.sqrt((phat*(1-phat) + z**2/(4*n)) / n)
    return (centre - half) / denom, (centre + half) / denom


def cliffs_delta(x: Sequence[float], y: Sequence[float]) -> float:
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    y = np.asarray(pd.Series(y).dropna(), dtype=float)
    if len(x) == 0 or len(y) == 0:
        return np.nan
    # efficient rank-based Cliff's delta
    xy = np.concatenate([x, y])
    ranks = stats.rankdata(xy)
    rx = ranks[:len(x)].sum()
    u = rx - len(x)*(len(x)+1)/2
    return (2*u)/(len(x)*len(y)) - 1


def smd_numeric(a: Sequence[float], b: Sequence[float]) -> float:
    a = pd.to_numeric(pd.Series(a), errors="coerce").dropna().astype(float)
    b = pd.to_numeric(pd.Series(b), errors="coerce").dropna().astype(float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = math.sqrt((a.var(ddof=1) + b.var(ddof=1))/2)
    if pooled == 0:
        return 0.0
    return (a.mean() - b.mean()) / pooled


def smd_categorical(a: Sequence, b: Sequence) -> float:
    aa = pd.Series(a).map(clean_label).dropna()
    bb = pd.Series(b).map(clean_label).dropna()
    levels = sorted(set(aa.unique()).union(set(bb.unique())))
    vals = []
    for lv in levels:
        p1 = (aa == lv).mean() if len(aa) else np.nan
        p0 = (bb == lv).mean() if len(bb) else np.nan
        p = (p1 + p0) / 2
        denom = math.sqrt(max(p*(1-p), 1e-12))
        vals.append((p1 - p0) / denom)
    if not vals:
        return np.nan
    return float(vals[np.argmax(np.abs(vals))])


def odds_ratio_2x2(a: int, b: int, c: int, d: int) -> Tuple[float, float, float]:
    # Haldane-Anscombe correction
    a, b, c, d = a+0.5, b+0.5, c+0.5, d+0.5
    orv = (a*d)/(b*c)
    se = math.sqrt(1/a + 1/b + 1/c + 1/d)
    return orv, math.exp(math.log(orv)-1.96*se), math.exp(math.log(orv)+1.96*se)


def parse_p_value(x) -> float:
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.floating)):
        return float(x)
    s = str(x).strip()
    if s.startswith("<"):
        try:
            return float(s.replace("<", "")) / 2
        except Exception:
            return 0.0005
    try:
        return float(s)
    except Exception:
        return np.nan

# =============================================================================
# 3) IMPORT, PREPARE, MODEL
# =============================================================================

def load_spss(path: Path) -> pd.DataFrame:
    try:
        import pyreadstat  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "pyreadstat is required to read the SPSS file. Install with:\n"
            "    python -m pip install pyreadstat"
        ) from exc
    df, _ = pyreadstat.read_sav(str(path), apply_value_formats=True)
    return df


def prepare_data(raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    col: Dict[str, str] = {}
    col["accident_history"] = find_col(raw, ["Accident_History", "Accident History", "AccidentHistory"])
    col["age"] = find_col(raw, ["Age", "age"])
    col["license"] = find_col(raw, ["License", "License_Type", "Licence"])
    col["driving_hours"] = find_col(raw, ["Driving_H_D", "Driving hours/day", "Driving_Hours_Day", "Driving_Hours"])
    col["rbg"] = find_col(raw, ["RBG", "Random blood glucose", "Random_Blood_Glucose"])
    col["betel"] = find_col(raw, ["B_Quid", "Betel_Quid", "Betel quid intake", "Betel_Quid_Intake"])

    optional = {
        "driving_years": ["Driving_Since", "Driving years", "Driving_Years", "Driving since"],
        "bmi": ["BMI", "Body_Mass_Index"],
        "pulse": ["Pulse", "Pulse_Rate", "Pulse rate"],
        "smoking": ["Smoking", "Smoke", "Smoking_Status"],
        "sleep": ["Sleep_Status", "Sleep status", "Sleep_duration_status", "Sleep_Duration_Status"],
        "glycemic": ["Diabetic_Status", "Glycemic status", "Glycemic_Status"],
        "urine_glucose": ["Urine_Glucose", "Glucose", "Urinary glucose", "Urinary_Glucose"],
        "urine_protein": ["Urine_Protein", "Protein", "Urinary protein", "Urinary_Protein"],
        "education": ["Education", "Educational_Status"],
        "marital": ["Marital_Status", "Marital status", "Marital"],
        "bp": ["BP_Status", "Blood pressure", "Blood_Pressure_Status"],
        "physical": ["Physical_Activity", "Physical activity", "Physical_activity_status"],
        "diet": ["Diet_Status", "Diet status"],
    }
    for k, vals in optional.items():
        c = find_col(raw, vals, required=False)
        if c is not None:
            col[k] = c

    # Keep all records for missingness diagnostics; create complete-case subset for accident figures.
    raw["accident_observed"] = raw[col["accident_history"]].notna()

    sub = raw[raw["accident_observed"]].copy()
    sub["accident_label"] = sub[col["accident_history"]].map(clean_label)
    sub["accident"] = sub["accident_label"].str.lower().eq("yes").astype(int)
    sub["accident_group"] = np.where(sub["accident"].eq(1), "Accident history", "No accident")

    # Numeric standard names
    rename_map = {col["age"]: "Age", col["driving_hours"]: "Driving_H_D", col["rbg"]: "RBG"}
    if "driving_years" in col: rename_map[col["driving_years"]] = "Driving_years"
    if "bmi" in col: rename_map[col["bmi"]] = "BMI"
    if "pulse" in col: rename_map[col["pulse"]] = "Pulse_rate"
    sub = sub.rename(columns=rename_map)

    for c in ["Age", "Driving_H_D", "RBG", "Driving_years", "BMI", "Pulse_rate"]:
        if c in sub.columns:
            sub[c] = pd.to_numeric(sub[c], errors="coerce")

    # Categorical standard names as plain object dtype, avoiding pandas StringDtype/patsy issue.
    sub["license_model"] = sub[col["license"]].map(clean_label).astype(object)
    sub["betel_model"] = sub[col["betel"]].map(clean_label).astype(object)

    sub["drive_hours_cat"] = pd.cut(
        sub["Driving_H_D"], bins=[-0.001, 5, 10, np.inf], labels=["0-5 h", "6-10 h", "11+ h"]
    )
    if "glycemic" in col:
        g = sub[col["glycemic"]].map(clean_label).str.lower()
        sub["glycemic_binary"] = np.where(g.eq("normal"), "Normal", "Prediabetes/Diabetes").astype(object)
    if "urine_glucose" in col:
        ug = sub[col["urine_glucose"]].map(clean_label).str.lower()
        sub["glycosuria_binary"] = np.where(ug.str.contains("negative|no", regex=True), "No glycosuria", "Glycosuria").astype(object)
    if "urine_protein" in col:
        up = sub[col["urine_protein"]].map(clean_label).str.lower()
        sub["proteinuria_binary"] = np.where(up.str.contains("negative|no", regex=True), "No proteinuria", "Proteinuria").astype(object)
    if "smoking" in col:
        sub["smoking_plot"] = sub[col["smoking"]].map(clean_label).astype(object)
    if "sleep" in col:
        sub["sleep_plot"] = sub[col["sleep"]].map(clean_label).astype(object)
    if "education" in col:
        sub["education_plot"] = sub[col["education"]].map(clean_label).astype(object)
    if "bp" in col:
        sub["bp_plot"] = sub[col["bp"]].map(clean_label).astype(object)

    return sub, col


def fit_model(sub: pd.DataFrame):
    model_df = sub[["accident", "Age", "license_model", "Driving_H_D", "RBG", "betel_model"]].dropna().copy()
    model_df["accident"] = pd.to_numeric(model_df["accident"], errors="coerce").astype(int)
    for c in ["Age", "Driving_H_D", "RBG"]:
        model_df[c] = pd.to_numeric(model_df[c], errors="coerce").astype(float)
    for c in ["license_model", "betel_model"]:
        model_df[c] = model_df[c].astype(str).astype(object)
    model_df = model_df.dropna().copy()
    formula = "accident ~ Age + C(license_model) + Driving_H_D + RBG + C(betel_model)"
    model = smf.logit(formula, data=model_df).fit(disp=False, maxiter=200)
    pred = model.predict(model_df)
    return model, model_df, np.asarray(pred)


def readable_term(term: str) -> str:
    if term == "Age": return "Age\n(per year)"
    if term == "Driving_H_D": return "Driving hours/day\n(per hour)"
    if term == "RBG": return "RBG\n(per mmol/L)"
    m = re.match(r"C\((.*?)\)\[T\.(.*?)\]", term)
    if m:
        var, val = m.group(1), m.group(2)
        if var == "license_model": return f"License: {val}\nvs reference"
        if var == "betel_model": return f"Betel quid: {val}\nvs reference"
    return term


def model_or_table(model) -> pd.DataFrame:
    conf = model.conf_int()
    rows = []
    for term in model.params.index:
        if term == "Intercept":
            continue
        rows.append({
            "term": term,
            "label": readable_term(term),
            "coef": model.params[term],
            "OR": math.exp(model.params[term]),
            "lo": math.exp(conf.loc[term, 0]),
            "hi": math.exp(conf.loc[term, 1]),
            "p": model.pvalues[term],
        })
    return pd.DataFrame(rows)


def design_predict_ci(model, newdf: pd.DataFrame, n_draws: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Build design matrix using original patsy design; then simulate beta from MVN.
    for c in ["license_model", "betel_model"]:
        if c in newdf.columns:
            newdf[c] = newdf[c].astype(str).astype(object)
    X = build_design_matrices([model.model.data.design_info], newdf, return_type="dataframe")[0]
    beta = model.params.values
    cov = model.cov_params().values
    rng = np.random.default_rng(RANDOM_STATE)
    draws = rng.multivariate_normal(beta, cov, size=n_draws)
    pp = expit(np.asarray(X) @ draws.T)
    pred = expit(np.asarray(X) @ beta)
    lo = np.percentile(pp, 2.5, axis=1)
    hi = np.percentile(pp, 97.5, axis=1)
    return pred, lo, hi


def bootstrap_coefficients(model_df: pd.DataFrame, n_boot: int = BOOT_N) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    terms = []
    rows = []
    formula = "accident ~ Age + C(license_model) + Driving_H_D + RBG + C(betel_model)"
    for b in range(n_boot):
        idx = rng.integers(0, len(model_df), len(model_df))
        samp = model_df.iloc[idx].copy()
        # skip samples lacking both outcomes or a categorical level causing singularity
        try:
            m = smf.logit(formula, data=samp).fit(disp=False, maxiter=120)
            if not terms:
                terms = [t for t in m.params.index if t != "Intercept"]
            for t in terms:
                if t in m.params.index:
                    rows.append({"boot": b, "term": t, "OR": math.exp(m.params[t])})
        except Exception:
            continue
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out

# =============================================================================
# 4) FIGURE 1 — COMPLETE-CASE VALIDITY
# =============================================================================

def fig1_complete_case_validity(raw: pd.DataFrame, sub: pd.DataFrame, col: Dict[str, str]) -> None:
    fig = plt.figure(figsize=(16, 10), facecolor=WHITE)
    gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1.0, 1.1], wspace=0.22, hspace=0.28)
    ax_flow = fig.add_subplot(gs[0, 0])
    ax_donut = fig.add_subplot(gs[0, 1])
    ax_ci = fig.add_subplot(gs[0, 2])
    ax_smd = fig.add_subplot(gs[1, 0:2])
    ax_missing = fig.add_subplot(gs[1, 2])

    total = len(raw)
    observed = int(raw[col["accident_history"]].notna().sum())
    missing = total - observed
    yes = int(sub["accident"].sum())
    no = int((sub["accident"] == 0).sum())

    # A. Flow diagram
    ax_flow.axis("off")
    add_panel_label(ax_flow, "A")
    boxes = [
        (0.08, 0.70, 0.84, 0.18, f"Raw SPSS cohort\nN={total:,}"),
        (0.08, 0.40, 0.38, 0.17, f"Observed accident history\nn={observed:,} ({observed/total*100:.1f}%)"),
        (0.54, 0.40, 0.38, 0.17, f"Missing accident history\nn={missing:,} ({missing/total*100:.1f}%)"),
        (0.08, 0.12, 0.38, 0.17, f"No accident\nn={no:,}"),
        (0.54, 0.12, 0.38, 0.17, f"Accident history\nn={yes:,}"),
    ]
    for i, (x, y, w, h, text) in enumerate(boxes):
        fc = ["#EEF4FF", "#EAF7F4", "#F3F4F6", "#EAF2F8", "#FCECEF"][i]
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.025",
                              fc=fc, ec=NAVY, lw=1.1)
        ax_flow.add_patch(rect)
        ax_flow.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=11, color=INK, fontweight="bold")
    for xy1, xy2 in [((0.35,0.70),(0.27,0.57)), ((0.65,0.70),(0.73,0.57)), ((0.27,0.40),(0.27,0.29)), ((0.27,0.40),(0.73,0.29))]:
        ax_flow.add_patch(FancyArrowPatch(xy1, xy2, arrowstyle="-|>", mutation_scale=12, color=GREY, lw=1.2))
    ax_flow.set_title("Analytic cohort flow", pad=8)

    # B. Donut chart
    add_panel_label(ax_donut, "B")
    wedges, _ = ax_donut.pie([no, yes], startangle=90, colors=[BLUE, CRIMSON],
                             wedgeprops=dict(width=0.42, edgecolor=WHITE, linewidth=2))
    ax_donut.text(0, 0.05, f"{yes/(yes+no)*100:.1f}%", ha="center", va="center",
                  fontsize=24, color=NAVY, fontweight="bold")
    ax_donut.text(0, -0.17, "observed accident\nhistory", ha="center", va="center", fontsize=10)
    ax_donut.legend(wedges, [f"No accident: {no:,}", f"Accident: {yes:,}"], loc="lower center",
                    bbox_to_anchor=(0.5, -0.15), frameon=False)
    ax_donut.set_title("Outcome burden among complete cases")

    # C. Wilson confidence interval for outcome prevalence
    add_panel_label(ax_ci, "C")
    lo, hi = ci_binomial_wilson(yes, yes+no)
    prev = yes/(yes+no)
    ax_ci.errorbar([prev*100], [0], xerr=[[ (prev-lo)*100 ], [ (hi-prev)*100 ]], fmt="o",
                   color=CRIMSON, ecolor=NAVY, elinewidth=2.2, capsize=6, markersize=10)
    ax_ci.set_yticks([0]); ax_ci.set_yticklabels(["Accident history"])
    ax_ci.set_xlabel("Prevalence among observed participants (%)")
    ax_ci.set_xlim(0, max(20, hi*100+4))
    ax_ci.grid(axis="x", alpha=0.5)
    ax_ci.text(prev*100, 0.18, f"{prev*100:.1f}%\n95% CI {lo*100:.1f}-{hi*100:.1f}%", ha="center", va="bottom",
               fontsize=10, fontweight="bold", color=INK)
    ax_ci.set_title("Uncertainty around observed accident prevalence")

    # D. Missingness-bias SMD diagnostics: observed vs missing outcome
    add_panel_label(ax_smd, "D")
    smd_rows = []
    numeric_candidates = [("Age", col.get("age")), ("Driving hours/day", col.get("driving_hours")),
                          ("RBG", col.get("rbg")), ("Driving years", col.get("driving_years")),
                          ("BMI", col.get("bmi")), ("Pulse", col.get("pulse"))]
    for label, c in numeric_candidates:
        if c and c in raw.columns:
            val = smd_numeric(raw.loc[raw["accident_observed"], c], raw.loc[~raw["accident_observed"], c])
            if not pd.isna(val): smd_rows.append((label, val))
    categorical_candidates = [("License", col.get("license")), ("Betel quid", col.get("betel")), ("Smoking", col.get("smoking")),
                              ("Education", col.get("education")), ("Glycemic", col.get("glycemic")), ("Urinary glucose", col.get("urine_glucose"))]
    for label, c in categorical_candidates:
        if c and c in raw.columns:
            val = smd_categorical(raw.loc[raw["accident_observed"], c], raw.loc[~raw["accident_observed"], c])
            if not pd.isna(val): smd_rows.append((label, val))
    smd_df = pd.DataFrame(smd_rows, columns=["Variable", "SMD"]).dropna()
    smd_df["abs"] = smd_df["SMD"].abs()
    smd_df = smd_df.sort_values("abs", ascending=True).tail(12)
    y = np.arange(len(smd_df))
    ax_smd.axvline(0, color=INK, lw=1)
    ax_smd.axvline(0.10, color=GOLD, ls="--", lw=1)
    ax_smd.axvline(-0.10, color=GOLD, ls="--", lw=1)
    ax_smd.hlines(y, 0, smd_df["SMD"], color=GREY, lw=1.4)
    ax_smd.scatter(smd_df["SMD"], y, s=70, color=np.where(smd_df["abs"]>=0.10, CRIMSON, BLUE), edgecolor=WHITE, zorder=3)
    ax_smd.set_yticks(y); ax_smd.set_yticklabels(smd_df["Variable"])
    ax_smd.set_xlabel("Standardized mean/proportion difference: observed vs missing accident-history field")
    ax_smd.set_title("Missingness diagnostic: is the complete-case subset exchangeable?")
    ax_smd.grid(axis="x", alpha=0.5)

    # E. Percent missing by a few key observed variables, using raw categories/tertiles
    add_panel_label(ax_missing, "E")
    miss_rows = []
    if col.get("license"):
        for k, g in raw.groupby(raw[col["license"]].map(clean_label), dropna=True):
            if len(g) >= 20:
                miss_rows.append((f"License: {k}", (~g["accident_observed"]).mean()*100, len(g)))
    if col.get("betel"):
        for k, g in raw.groupby(raw[col["betel"]].map(clean_label), dropna=True):
            if len(g) >= 20:
                miss_rows.append((f"Betel: {k}", (~g["accident_observed"]).mean()*100, len(g)))
    for label, c in [("Age tertile", col.get("age")), ("RBG tertile", col.get("rbg"))]:
        if c:
            x = pd.to_numeric(raw[c], errors="coerce")
            try:
                bins = pd.qcut(x, q=3, duplicates="drop")
                for k, idx in bins.dropna().groupby(bins.dropna()).groups.items():
                    g = raw.loc[idx]
                    miss_rows.append((f"{label}: {str(k)}", (~g["accident_observed"]).mean()*100, len(g)))
            except Exception:
                pass
    md = pd.DataFrame(miss_rows, columns=["Stratum", "Missing %", "n"]).sort_values("Missing %").tail(10)
    ax_missing.barh(np.arange(len(md)), md["Missing %"], color=PURPLE, alpha=0.85)
    ax_missing.set_yticks(np.arange(len(md))); ax_missing.set_yticklabels(md["Stratum"], fontsize=8.2)
    ax_missing.set_xlabel("Accident-history missingness (%)")
    ax_missing.set_title("Where is accident-history missingness concentrated?")
    ax_missing.grid(axis="x", alpha=0.45)

    fig.suptitle("Figure 1. Complete-case validity and accident-outcome architecture", fontsize=18, fontweight="bold", color=NAVY, y=0.99)
    fig.text(0.01, 0.01, "Purpose: documents outcome incompleteness, observed accident burden, and whether missing accident history may bias inference.", color=GREY, fontsize=10)
    savefig(fig, "Figure_1_complete_case_validity_and_outcome_architecture")

# =============================================================================
# 5) FIGURE 2 — ACCIDENT PHENOTYPE SEPARABILITY
# =============================================================================

def fig2_accident_phenotype(sub: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(16, 11), facecolor=WHITE)
    gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.22, hspace=0.28)
    ax_eff = fig.add_subplot(gs[0, 0])
    ax_corr = fig.add_subplot(gs[0, 1])
    ax_pair = fig.add_subplot(gs[0, 2])
    ax_ecdf = fig.add_subplot(gs[1, 0])
    ax_density = fig.add_subplot(gs[1, 1])
    ax_rank = fig.add_subplot(gs[1, 2])

    # A. Robust effect-size fingerprint for continuous variables
    add_panel_label(ax_eff, "A")
    conts = [("Age", "Age"), ("Driving_H_D", "Driving hours/day"), ("Driving_years", "Driving years"),
             ("RBG", "RBG"), ("BMI", "BMI"), ("Pulse_rate", "Pulse rate")]
    rows = []
    for c, label in conts:
        if c in sub.columns and sub[c].notna().sum() > 30:
            yes = sub.loc[sub["accident"]==1, c]
            no = sub.loc[sub["accident"]==0, c]
            d = cliffs_delta(yes, no)
            p = stats.mannwhitneyu(no.dropna(), yes.dropna(), alternative="two-sided").pvalue
            rows.append((label, d, p))
    edf = pd.DataFrame(rows, columns=["Variable", "delta", "p"]).sort_values("delta")
    y = np.arange(len(edf))
    ax_eff.axvline(0, color=INK, lw=1)
    ax_eff.hlines(y, 0, edf["delta"], color=GREY, lw=1.5)
    ax_eff.scatter(edf["delta"], y, s=75, color=np.where(edf["p"]<0.05, CRIMSON, BLUE), edgecolor=WHITE)
    ax_eff.set_yticks(y); ax_eff.set_yticklabels(edf["Variable"])
    ax_eff.set_xlabel("Cliff's delta: accident vs no accident")
    ax_eff.set_title("Distribution-free phenotype shift")
    ax_eff.grid(axis="x", alpha=0.45)

    # B. Correlation structure among continuous variables
    add_panel_label(ax_corr, "B")
    cont_cols = [c for c, _ in conts if c in sub.columns and sub[c].notna().sum() > 30]
    labels = [dict(conts)[c] for c in cont_cols]
    corr = sub[cont_cols].corr(method="spearman")
    im = ax_corr.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax_corr.set_xticks(range(len(labels))); ax_corr.set_xticklabels(labels, rotation=45, ha="right")
    ax_corr.set_yticks(range(len(labels))); ax_corr.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax_corr.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", fontsize=8,
                         color=WHITE if abs(corr.iloc[i,j]) > 0.55 else INK)
    ax_corr.set_title("Spearman correlation architecture")
    cbar = fig.colorbar(im, ax=ax_corr, fraction=0.046, pad=0.04)
    cbar.set_label("ρ")

    # C. Metabolic–workload plane
    add_panel_label(ax_pair, "C")
    ax_pair.scatter(sub.loc[sub["accident"]==0, "Driving_H_D"], sub.loc[sub["accident"]==0, "RBG"],
                    s=18, color=BLUE, alpha=0.35, label="No accident")
    ax_pair.scatter(sub.loc[sub["accident"]==1, "Driving_H_D"], sub.loc[sub["accident"]==1, "RBG"],
                    s=32, color=CRIMSON, alpha=0.75, label="Accident")
    ax_pair.set_xlabel("Driving hours/day")
    ax_pair.set_ylabel("RBG (mmol/L)")
    ax_pair.set_title("Joint metabolic–occupational risk plane")
    ax_pair.legend(frameon=False)
    ax_pair.grid(alpha=0.35)

    # D. ECDF for driving years (cumulative exposure)
    add_panel_label(ax_ecdf, "D")
    xcol = "Driving_years" if "Driving_years" in sub.columns else "Driving_H_D"
    for grp, color in [("No accident", BLUE), ("Accident history", CRIMSON)]:
        x = np.sort(sub.loc[sub["accident_group"]==grp, xcol].dropna())
        yv = np.arange(1, len(x)+1)/len(x)
        ax_ecdf.step(x, yv, where="post", color=color, lw=2.2, label=grp)
    ax_ecdf.set_xlabel("Driving years" if xcol=="Driving_years" else "Driving hours/day")
    ax_ecdf.set_ylabel("Empirical cumulative probability")
    ax_ecdf.set_title("Cumulative exposure separation")
    ax_ecdf.legend(frameon=False)
    ax_ecdf.grid(alpha=0.35)

    # E. RBG density with clinically interpretable shaded upper tail
    add_panel_label(ax_density, "E")
    for grp, color in [("No accident", BLUE), ("Accident history", CRIMSON)]:
        x = sub.loc[sub["accident_group"]==grp, "RBG"].dropna().astype(float)
        if len(x) > 3:
            kde = stats.gaussian_kde(x)
            grid = np.linspace(max(2, x.quantile(0.01)), min(max(sub["RBG"].max(), 9), sub["RBG"].quantile(0.995)), 300)
            ax_density.plot(grid, kde(grid), color=color, lw=2.3, label=grp)
            ax_density.fill_between(grid, 0, kde(grid), color=color, alpha=0.13)
    ax_density.axvline(7.8, color=GOLD, ls="--", lw=1.6, label="RBG dysglycemia threshold")
    ax_density.set_xlabel("RBG (mmol/L)")
    ax_density.set_ylabel("Density")
    ax_density.set_title("Metabolic upper-tail enrichment")
    ax_density.legend(frameon=False, fontsize=8.5)

    # F. Age-adjusted rank association summary for selected continuous variables
    add_panel_label(ax_rank, "F")
    rank_rows = []
    for c, label in conts:
        if c in sub.columns and c != "Age" and sub[c].notna().sum() > 30:
            temp = sub[["accident", "Age", c]].dropna()
            # residualize variable by age using simple linear regression; correlate residual with accident
            slope, intercept, *_ = stats.linregress(temp["Age"], temp[c])
            resid = temp[c] - (intercept + slope*temp["Age"])
            rho, p = stats.spearmanr(resid, temp["accident"])
            rank_rows.append((label, rho, p))
    rdf = pd.DataFrame(rank_rows, columns=["Variable", "rho", "p"]).sort_values("rho")
    yy = np.arange(len(rdf))
    ax_rank.axvline(0, color=INK, lw=1)
    ax_rank.hlines(yy, 0, rdf["rho"], color=GREY, lw=1.4)
    ax_rank.scatter(rdf["rho"], yy, s=70, color=np.where(rdf["p"]<0.05, CRIMSON, BLUE), edgecolor=WHITE)
    ax_rank.set_yticks(yy); ax_rank.set_yticklabels(rdf["Variable"])
    ax_rank.set_xlabel("Spearman ρ with accident after age residualization")
    ax_rank.set_title("Age-residualized association screen")
    ax_rank.grid(axis="x", alpha=0.45)

    fig.suptitle("Figure 2. Multidimensional accident phenotype and separability", fontsize=18, fontweight="bold", color=NAVY, y=0.99)
    fig.text(0.01, 0.01, "Purpose: identifies whether accident-history cases form a distinct phenotype across workload, cumulative exposure, metabolic profile, and correlation structure.", color=GREY, fontsize=10)
    savefig(fig, "Figure_2_multidimensional_accident_phenotype")

# =============================================================================
# 6) FIGURE 3 — ADJUSTED INFERENCE AND RISK SURFACE
# =============================================================================

def fig3_adjusted_inference(model, model_df: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(16, 11), facecolor=WHITE)
    gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.24, hspace=0.28)
    ax_for = fig.add_subplot(gs[0, 0])
    ax_rbg = fig.add_subplot(gs[0, 1])
    ax_dh = fig.add_subplot(gs[0, 2])
    ax_heat = fig.add_subplot(gs[1, 0:2])
    ax_boot = fig.add_subplot(gs[1, 2])

    tab = model_or_table(model).sort_values("OR")
    tab.to_csv(OUT_DIR / "Figure_3_adjusted_OR_table.csv", index=False)

    # A. Forest plot
    add_panel_label(ax_for, "A")
    y = np.arange(len(tab))
    ax_for.axvline(1, color=INK, ls="--", lw=1.1)
    ax_for.errorbar(tab["OR"], y, xerr=[tab["OR"]-tab["lo"], tab["hi"]-tab["OR"]], fmt="none", ecolor=NAVY, elinewidth=1.6, capsize=3)
    ax_for.scatter(tab["OR"], y, s=85, color=np.where(tab["p"]<0.05, CRIMSON, BLUE), edgecolor=WHITE, zorder=3)
    ax_for.set_xscale("log")
    ax_for.set_yticks(y); ax_for.set_yticklabels(tab["label"])
    ax_for.set_xlabel("Adjusted odds ratio (log scale)")
    ax_for.set_title("Adjusted effect sizes")
    ax_for.grid(axis="x", alpha=0.45)

    # Median prediction template
    age_med = float(model_df["Age"].median())
    lic_ref = str(model_df["license_model"].mode().iloc[0])
    betel_ref = str(model_df["betel_model"].mode().iloc[0])
    dh_med = float(model_df["Driving_H_D"].median())
    rbg_med = float(model_df["RBG"].median())

    # B. Adjusted RBG marginal curve
    add_panel_label(ax_rbg, "B")
    grid = np.linspace(model_df["RBG"].quantile(0.02), model_df["RBG"].quantile(0.98), 160)
    nd = pd.DataFrame({"Age": age_med, "license_model": lic_ref, "Driving_H_D": dh_med, "RBG": grid, "betel_model": betel_ref})
    nd["license_model"] = nd["license_model"].astype(object); nd["betel_model"] = nd["betel_model"].astype(object)
    pr, lo, hi = design_predict_ci(model, nd, n_draws=1000)
    ax_rbg.plot(grid, pr*100, color=CRIMSON, lw=2.4)
    ax_rbg.fill_between(grid, lo*100, hi*100, color=CRIMSON, alpha=0.18)
    ax_rbg.set_xlabel("RBG (mmol/L)")
    ax_rbg.set_ylabel("Adjusted accident probability (%)")
    ax_rbg.set_title("Metabolic gradient")
    ax_rbg.grid(alpha=0.35)

    # C. Adjusted driving-hour marginal curve
    add_panel_label(ax_dh, "C")
    grid2 = np.linspace(model_df["Driving_H_D"].quantile(0.02), model_df["Driving_H_D"].quantile(0.98), 160)
    nd2 = pd.DataFrame({"Age": age_med, "license_model": lic_ref, "Driving_H_D": grid2, "RBG": rbg_med, "betel_model": betel_ref})
    nd2["license_model"] = nd2["license_model"].astype(object); nd2["betel_model"] = nd2["betel_model"].astype(object)
    pr2, lo2, hi2 = design_predict_ci(model, nd2, n_draws=1000)
    ax_dh.plot(grid2, pr2*100, color=PURPLE, lw=2.4)
    ax_dh.fill_between(grid2, lo2*100, hi2*100, color=PURPLE, alpha=0.18)
    ax_dh.set_xlabel("Driving hours/day")
    ax_dh.set_ylabel("Adjusted accident probability (%)")
    ax_dh.set_title("Workload gradient")
    ax_dh.grid(alpha=0.35)

    # D. 2D risk surface
    add_panel_label(ax_heat, "D")
    rbg_grid = np.linspace(model_df["RBG"].quantile(0.03), model_df["RBG"].quantile(0.97), 80)
    dh_grid = np.linspace(model_df["Driving_H_D"].quantile(0.03), model_df["Driving_H_D"].quantile(0.97), 70)
    RR, DD = np.meshgrid(rbg_grid, dh_grid)
    nd3 = pd.DataFrame({"Age": age_med, "license_model": lic_ref, "Driving_H_D": DD.ravel(), "RBG": RR.ravel(), "betel_model": betel_ref})
    nd3["license_model"] = nd3["license_model"].astype(object); nd3["betel_model"] = nd3["betel_model"].astype(object)
    zz = np.asarray(model.predict(nd3)).reshape(DD.shape) * 100
    im = ax_heat.contourf(RR, DD, zz, levels=16, cmap="magma")
    cs = ax_heat.contour(RR, DD, zz, colors=WHITE, levels=6, linewidths=0.7, alpha=0.8)
    ax_heat.clabel(cs, inline=True, fontsize=8, fmt="%.1f%%")
    ax_heat.scatter(model_df["RBG"], model_df["Driving_H_D"], c=model_df["accident"], cmap=mpl.colors.ListedColormap(["#FFFFFF44", "#00FFFFAA"]), s=10, alpha=0.45, edgecolor="none")
    ax_heat.set_xlabel("RBG (mmol/L)")
    ax_heat.set_ylabel("Driving hours/day")
    ax_heat.set_title("Adjusted risk surface: metabolic stress × driving workload")
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.032, pad=0.02)
    cbar.set_label("Predicted probability (%)")

    # E. Bootstrap coefficient stability
    add_panel_label(ax_boot, "E")
    boot = bootstrap_coefficients(model_df, n_boot=BOOT_N)
    if not boot.empty:
        order = list(tab.sort_values("OR")["term"])
        positions = np.arange(len(order))
        for i, term in enumerate(order):
            vals = boot.loc[boot["term"]==term, "OR"].replace([np.inf, -np.inf], np.nan).dropna()
            if len(vals) > 10:
                parts = ax_boot.violinplot(vals, positions=[i], vert=False, widths=0.70, showmeans=False, showextrema=False)
                for pc in parts["bodies"]:
                    pc.set_facecolor(CRIMSON if np.median(vals)>1 else BLUE)
                    pc.set_edgecolor("none")
                    pc.set_alpha(0.45)
                ax_boot.scatter(np.median(vals), i, color=NAVY, s=35, zorder=3)
        ax_boot.axvline(1, color=INK, ls="--", lw=1)
        ax_boot.set_xscale("log")
        ax_boot.set_yticks(positions); ax_boot.set_yticklabels([readable_term(t) for t in order], fontsize=8.5)
        ax_boot.set_xlabel("Bootstrap OR distribution")
        ax_boot.set_title(f"Coefficient stability ({BOOT_N} resamples)")
        ax_boot.grid(axis="x", alpha=0.45)
    else:
        ax_boot.text(0.5, 0.5, "Bootstrap failed\ninsufficient stable samples", ha="center", va="center")
        ax_boot.axis("off")

    fig.suptitle("Figure 3. Adjusted inference, uncertainty, and nonlinear risk surface", fontsize=18, fontweight="bold", color=NAVY, y=0.99)
    fig.text(0.01, 0.01, "Purpose: moves beyond OR tables by showing adjusted magnitude, uncertainty, marginal absolute risk, joint risk surface, and bootstrap coefficient stability.", color=GREY, fontsize=10)
    savefig(fig, "Figure_3_adjusted_inference_uncertainty_risk_surface")

# =============================================================================
# 7) FIGURE 4 — PERFORMANCE, CALIBRATION, DECISION CURVE
# =============================================================================

def decision_curve(y: np.ndarray, p: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
    n = len(y)
    rows = []
    prevalence = y.mean()
    for pt in thresholds:
        pred_pos = p >= pt
        tp = ((pred_pos) & (y == 1)).sum()
        fp = ((pred_pos) & (y == 0)).sum()
        nb = tp/n - fp/n * (pt/(1-pt))
        treat_all = prevalence - (1-prevalence)*(pt/(1-pt))
        rows.append({"threshold": pt, "model": nb, "treat_all": treat_all, "treat_none": 0.0})
    return pd.DataFrame(rows)


def fig4_performance(model_df: pd.DataFrame, pred: np.ndarray) -> None:
    y = model_df["accident"].to_numpy()
    fig = plt.figure(figsize=(16, 11), facecolor=WHITE)
    gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.22, hspace=0.28)
    ax_roc = fig.add_subplot(gs[0, 0])
    ax_pr = fig.add_subplot(gs[0, 1])
    ax_cal = fig.add_subplot(gs[0, 2])
    ax_dca = fig.add_subplot(gs[1, 0:2])
    ax_risk = fig.add_subplot(gs[1, 2])

    # A. ROC
    add_panel_label(ax_roc, "A")
    fpr, tpr, _ = roc_curve(y, pred)
    roc_auc = auc(fpr, tpr)
    ax_roc.plot(fpr, tpr, color=CRIMSON, lw=2.4, label=f"AUC={roc_auc:.3f}")
    ax_roc.plot([0,1], [0,1], color=GREY, ls="--")
    ax_roc.set_xlabel("1 - Specificity"); ax_roc.set_ylabel("Sensitivity")
    ax_roc.set_title("Discrimination: ROC")
    ax_roc.legend(frameon=False, loc="lower right")
    ax_roc.set_aspect("equal", adjustable="box")
    ax_roc.grid(alpha=0.35)

    # B. Precision-recall
    add_panel_label(ax_pr, "B")
    precision, recall, _ = precision_recall_curve(y, pred)
    ap = average_precision_score(y, pred)
    ax_pr.plot(recall, precision, color=PURPLE, lw=2.4, label=f"Average precision={ap:.3f}")
    ax_pr.axhline(y.mean(), color=GREY, ls="--", lw=1.2, label=f"Event rate={y.mean():.3f}")
    ax_pr.set_xlabel("Recall / sensitivity"); ax_pr.set_ylabel("Precision / PPV")
    ax_pr.set_title("Rare-event performance: precision-recall")
    ax_pr.legend(frameon=False)
    ax_pr.grid(alpha=0.35)

    # C. Calibration
    add_panel_label(ax_cal, "C")
    frac, meanp = calibration_curve(y, pred, n_bins=8, strategy="quantile")
    ax_cal.plot([0, max(meanp.max(), frac.max())*1.2], [0, max(meanp.max(), frac.max())*1.2], color=GREY, ls="--", label="Ideal")
    ax_cal.plot(meanp, frac, marker="o", color=BLUE, lw=2.3, label="Observed")
    ax_cal.set_xlabel("Predicted risk")
    ax_cal.set_ylabel("Observed event proportion")
    ax_cal.set_title("Calibration by risk quantile")
    ax_cal.text(0.98, 0.05, f"Brier={brier_score_loss(y,pred):.3f}", transform=ax_cal.transAxes, ha="right",
                bbox=dict(boxstyle="round,pad=0.25", fc=WHITE, ec=GRID))
    ax_cal.legend(frameon=False)
    ax_cal.grid(alpha=0.35)

    # D. Decision curve analysis
    add_panel_label(ax_dca, "D")
    thresholds = np.linspace(0.02, 0.40, 120)
    dca = decision_curve(y, pred, thresholds)
    ax_dca.plot(dca["threshold"], dca["model"], color=CRIMSON, lw=2.5, label="Prediction model")
    ax_dca.plot(dca["threshold"], dca["treat_all"], color=GREY, lw=1.8, ls="--", label="Assume all high risk")
    ax_dca.plot(dca["threshold"], dca["treat_none"], color=INK, lw=1.4, ls=":", label="Assume none high risk")
    ax_dca.set_xlabel("Risk threshold probability")
    ax_dca.set_ylabel("Net benefit")
    ax_dca.set_title("Decision-curve analysis: clinical/occupational utility")
    ax_dca.legend(frameon=False, ncol=3)
    ax_dca.grid(alpha=0.35)

    # E. Predicted risk distributions
    add_panel_label(ax_risk, "E")
    bins = np.linspace(0, max(pred.max()*1.1, 0.30), 30)
    ax_risk.hist(pred[y==0], bins=bins, density=True, alpha=0.55, color=BLUE, label="No accident")
    ax_risk.hist(pred[y==1], bins=bins, density=True, alpha=0.70, color=CRIMSON, label="Accident")
    ax_risk.set_xlabel("Predicted accident risk")
    ax_risk.set_ylabel("Density")
    ax_risk.set_title("Case/non-case risk separation")
    ax_risk.legend(frameon=False)
    ax_risk.grid(axis="y", alpha=0.35)

    fig.suptitle("Figure 4. Model discrimination, calibration, and decision value", fontsize=18, fontweight="bold", color=NAVY, y=0.99)
    fig.text(0.01, 0.01, "Purpose: evaluates whether the adjusted model is not only statistically significant but also predictive, calibrated, and useful across practical risk thresholds.", color=GREY, fontsize=10)
    savefig(fig, "Figure_4_model_performance_calibration_decision_curve")

# =============================================================================
# 8) FIGURE 5 — ACTIONABLE RISK STRATIFICATION
# =============================================================================

def fig5_actionable_translation(model, model_df: pd.DataFrame, sub: pd.DataFrame) -> None:
    pred = np.asarray(model.predict(model_df))
    fig = plt.figure(figsize=(16, 11), facecolor=WHITE)
    gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.24, hspace=0.30)
    ax_nom = fig.add_subplot(gs[0, 0])
    ax_strata = fig.add_subplot(gs[0, 1])
    ax_obs = fig.add_subplot(gs[0, 2])
    ax_joint = fig.add_subplot(gs[1, 0:2])
    ax_rank = fig.add_subplot(gs[1, 2])

    # A. Coefficient-weighted point scale / compact nomogram
    add_panel_label(ax_nom, "A")
    tab = model_or_table(model).copy()
    tab["points"] = tab["coef"] / tab["coef"].abs().max() * 100
    tab = tab.sort_values("points")
    y = np.arange(len(tab))
    ax_nom.axvline(0, color=INK, lw=1)
    ax_nom.barh(y, tab["points"], color=np.where(tab["points"]>0, CRIMSON, BLUE), alpha=0.85)
    ax_nom.set_yticks(y); ax_nom.set_yticklabels(tab["label"], fontsize=8.5)
    ax_nom.set_xlabel("Relative model points\n(scaled to strongest coefficient = 100)")
    ax_nom.set_title("Compact risk-score anatomy")
    ax_nom.grid(axis="x", alpha=0.35)

    # Risk strata based on predicted risk tertile/quartiles with event meaning
    model_df2 = model_df.copy()
    model_df2["pred"] = pred
    try:
        model_df2["risk_stratum"] = pd.qcut(model_df2["pred"], q=[0, .5, .8, 1.0], labels=["Lower 50%", "Next 30%", "Highest 20%"], duplicates="drop")
    except Exception:
        model_df2["risk_stratum"] = pd.cut(model_df2["pred"], bins=3, labels=["Low", "Moderate", "High"])

    # B. Predicted-risk strata composition
    add_panel_label(ax_strata, "B")
    comp = model_df2.groupby("risk_stratum", observed=True)["accident"].agg(["sum", "count"]).reset_index()
    comp["prev"] = comp["sum"] / comp["count"] * 100
    lohi = comp.apply(lambda r: ci_binomial_wilson(int(r["sum"]), int(r["count"])), axis=1)
    comp["lo"] = [x[0]*100 for x in lohi]; comp["hi"] = [x[1]*100 for x in lohi]
    x = np.arange(len(comp))
    ax_strata.bar(x, comp["prev"], color=[BLUE, GOLD, CRIMSON][:len(comp)], alpha=0.88)
    ax_strata.errorbar(x, comp["prev"], yerr=[comp["prev"]-comp["lo"], comp["hi"]-comp["prev"]], fmt="none", ecolor=INK, capsize=4)
    for i, r in comp.iterrows():
        ax_strata.text(i, r["prev"]+1, f"{int(r['sum'])}/{int(r['count'])}", ha="center", fontsize=9, fontweight="bold")
    ax_strata.set_xticks(x); ax_strata.set_xticklabels(comp["risk_stratum"], rotation=15)
    ax_strata.set_ylabel("Observed accident prevalence (%)")
    ax_strata.set_title("Risk-strata validation")
    ax_strata.grid(axis="y", alpha=0.35)

    # C. Lorenz/capture curve: how many accidents captured by top risk fraction
    add_panel_label(ax_obs, "C")
    dd = model_df2.sort_values("pred", ascending=False).reset_index(drop=True)
    dd["cum_pop"] = (np.arange(len(dd)) + 1) / len(dd)
    dd["cum_events"] = dd["accident"].cumsum() / dd["accident"].sum()
    ax_obs.plot(dd["cum_pop"]*100, dd["cum_events"]*100, color=CRIMSON, lw=2.5)
    ax_obs.plot([0,100], [0,100], color=GREY, ls="--")
    for cutoff in [20, 30, 50]:
        ev = float(dd.loc[dd["cum_pop"] <= cutoff/100, "accident"].sum() / dd["accident"].sum() * 100)
        ax_obs.scatter([cutoff], [ev], color=NAVY, s=45)
        ax_obs.text(cutoff, ev+3, f"Top {cutoff}%\ncaptures {ev:.0f}%", ha="center", fontsize=8.5)
    ax_obs.set_xlabel("Population ranked high-to-low risk (%)")
    ax_obs.set_ylabel("Accidents captured (%)")
    ax_obs.set_title("Risk concentration / capture curve")
    ax_obs.grid(alpha=0.35)

    # D. Joint occupational-metabolic strata heatmap based on clinically meaningful bins
    add_panel_label(ax_joint, "D")
    temp = model_df.copy()
    temp["pred"] = pred
    temp["RBG_bin"] = pd.cut(temp["RBG"], bins=[-np.inf, 5.6, 7.8, np.inf], labels=["<5.6", "5.6-7.7", "≥7.8"])
    temp["Drive_bin"] = pd.cut(temp["Driving_H_D"], bins=[-np.inf, 5, 10, np.inf], labels=["0-5 h", "6-10 h", "11+ h"])
    pv = temp.groupby(["Drive_bin", "RBG_bin"], observed=True).agg(events=("accident","sum"), n=("accident","count"), mean_pred=("pred","mean")).reset_index()
    matrix = pv.pivot(index="Drive_bin", columns="RBG_bin", values="mean_pred") * 100
    im = ax_joint.imshow(matrix.values, cmap="magma", aspect="auto")
    ax_joint.set_xticks(range(matrix.shape[1])); ax_joint.set_xticklabels(matrix.columns)
    ax_joint.set_yticks(range(matrix.shape[0])); ax_joint.set_yticklabels(matrix.index)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            row = pv[(pv["Drive_bin"]==matrix.index[i]) & (pv["RBG_bin"]==matrix.columns[j])]
            if len(row):
                r = row.iloc[0]
                ax_joint.text(j, i, f"{matrix.iloc[i,j]:.1f}%\n{int(r.events)}/{int(r.n)}", ha="center", va="center",
                              fontsize=10, fontweight="bold", color=WHITE if matrix.iloc[i,j] > np.nanmedian(matrix.values) else INK)
    ax_joint.set_xlabel("RBG category (mmol/L)")
    ax_joint.set_ylabel("Driving hours/day")
    ax_joint.set_title("Actionable joint risk grid: workload × glycemia")
    cbar = fig.colorbar(im, ax=ax_joint, fraction=0.028, pad=0.02)
    cbar.set_label("Mean adjusted predicted risk (%)")

    # E. Individual priority list profile: top 10 highest-risk complete cases, anonymized ranks
    add_panel_label(ax_rank, "E")
    top = model_df2.sort_values("pred", ascending=False).head(10).copy().reset_index(drop=True)
    ax_rank.barh(np.arange(len(top)), top["pred"]*100, color=CRIMSON, alpha=0.85)
    ax_rank.set_yticks(np.arange(len(top))); ax_rank.set_yticklabels([f"Rank {i+1}" for i in range(len(top))])
    ax_rank.invert_yaxis()
    ax_rank.set_xlabel("Predicted accident risk (%)")
    ax_rank.set_title("Anonymized highest-risk profiles")
    for i, r in top.iterrows():
        txt = f"Age {r['Age']:.0f}, {r['Driving_H_D']:.0f}h, RBG {r['RBG']:.1f}"
        ax_rank.text(r["pred"]*100 + 0.5, i, txt, va="center", fontsize=8.3)
    ax_rank.grid(axis="x", alpha=0.35)

    fig.suptitle("Figure 5. Actionable risk stratification and occupational-health translation", fontsize=18, fontweight="bold", color=NAVY, y=0.99)
    fig.text(0.01, 0.01, "Purpose: translates the adjusted model into interpretable risk ranking, workload–glycemia strata, and prevention-oriented prioritization.", color=GREY, fontsize=10)
    savefig(fig, "Figure_5_actionable_risk_stratification_translation")

# =============================================================================
# 9) SUPPLEMENTARY: OPTIONAL UNADJUSTED OR ATLAS FROM EXCEL
# =============================================================================

def supplementary_unadjusted_or_atlas() -> None:
    if not TABLE_XLSX.exists():
        return
    try:
        df = pd.read_excel(TABLE_XLSX, sheet_name="Unadjusted_ORs", header=2)
    except Exception:
        return
    cols = {c.lower().strip(): c for c in df.columns}
    if "or" not in cols:
        return
    df = df.rename(columns={
        cols.get("variable", "Variable"): "Variable",
        cols.get("category vs reference", "Category vs reference"): "Category",
        cols.get("or", "OR"): "OR",
        cols.get("95% ci low", "95% CI low"): "lo",
        cols.get("95% ci high", "95% CI high"): "hi",
        cols.get("p value", "P value"): "p",
    })
    df["Variable"] = df["Variable"].ffill()
    df = df[df["OR"].notna()].copy()
    if df.empty:
        return
    df["p_num"] = df["p"].map(parse_p_value)
    df["label"] = df["Variable"].astype(str) + ": " + df["Category"].astype(str)
    df = df.sort_values("OR")

    fig, ax = plt.subplots(figsize=(10, max(7, 0.42*len(df))), facecolor=WHITE)
    y = np.arange(len(df))
    ax.axvline(1, color=INK, ls="--", lw=1)
    ax.errorbar(df["OR"], y, xerr=[df["OR"]-df["lo"], df["hi"]-df["OR"]], fmt="none", ecolor=NAVY, elinewidth=1.3, capsize=3)
    ax.scatter(df["OR"], y, s=58, color=np.where(df["p_num"]<0.05, CRIMSON, BLUE), edgecolor=WHITE)
    ax.set_xscale("log")
    ax.set_yticks(y); ax.set_yticklabels(df["label"], fontsize=8.8)
    ax.set_xlabel("Unadjusted odds ratio (log scale)")
    ax.set_title("Supplementary Figure. Unadjusted association atlas", fontsize=15, fontweight="bold", color=NAVY)
    ax.grid(axis="x", alpha=0.45)
    savefig(fig, "Supplementary_unadjusted_association_atlas")

# =============================================================================
# 10) MAIN
# =============================================================================

def main() -> None:
    set_style()
    ensure_dir(OUT_DIR)

    if not DATA_SAV.exists():
        raise FileNotFoundError(f"SPSS file not found: {DATA_SAV}")

    raw = load_spss(DATA_SAV)
    sub, col = prepare_data(raw)
    model, model_df, pred = fit_model(sub)

    # Save reproducibility outputs
    model_or_table(model).to_csv(OUT_DIR / "final_adjusted_OR_table_used_in_figures.csv", index=False)
    with open(OUT_DIR / "run_summary.txt", "w", encoding="utf-8") as f:
        f.write("10/10 accident-history figure generation summary\n")
        f.write("================================================\n")
        f.write(f"Raw N: {len(raw)}\n")
        f.write(f"Accident history observed: {len(sub)}\n")
        f.write(f"Accident events in observed subset: {int(sub['accident'].sum())}\n")
        f.write(f"Model N: {len(model_df)}\n")
        f.write(f"AUC: {auc(*roc_curve(model_df['accident'], pred)[:2]):.3f}\n")
        f.write(f"Brier: {brier_score_loss(model_df['accident'], pred):.3f}\n\n")
        f.write(str(model.summary()))

    fig1_complete_case_validity(raw, sub, col)
    fig2_accident_phenotype(sub)
    fig3_adjusted_inference(model, model_df)
    fig4_performance(model_df, pred)
    fig5_actionable_translation(model, model_df, sub)
    supplementary_unadjusted_or_atlas()

    print("Done. 10/10 high-impact figures saved to:", OUT_DIR)
    for p in sorted(OUT_DIR.glob("*")):
        print(" -", p.name)


if __name__ == "__main__":
    main()
