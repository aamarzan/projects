#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
High-impact manuscript-grade figures for the accident-history re-analysis.

Purpose
-------
This script reads the raw SPSS dataset and/or the re-analysis Excel workbook,
reconstructs the accident-history complete-case dataset, fits the final logistic
regression model, and creates advanced journal-grade figures.

Core figures produced
---------------------
Figure 1. Outcome availability and observed accident burden
Figure 2. Continuous-variable accident signature: violin/box/strip panels with
          Mann-Whitney p values and Cliff's delta effect sizes
Figure 3. Adjusted multivariable odds-ratio forest plot
Figure 4. Model-predicted accident probability curves across RBG and daily
          driving hours, stratified by license and betel quid intake
Figure 5. Categorical risk landscape heatmap: accident prevalence and n per
          subgroup
Figure 6. Model performance: ROC curve and calibration curve

Recommended citation note in manuscript
---------------------------------------
All accident-history figures should be described as complete-case analyses,
because accident history is observed for only 874/2441 participants in the raw
file used for this re-analysis.

Dependencies
------------
Required: pandas, numpy, matplotlib, seaborn, scipy, statsmodels, scikit-learn,
          openpyxl
Required for SPSS .sav import: pyreadstat

Install, if needed:
    pip install pandas numpy matplotlib seaborn scipy statsmodels scikit-learn openpyxl pyreadstat

Usage
-----
Edit DATA_SAV and TABLE_XLSX below, then run:
    python create_high_impact_accident_figures.py

Outputs
-------
A folder named 'accident_history_high_impact_figures' containing PNG, PDF, and SVG
versions of each figure, plus a CSV summary of the final adjusted model.
"""

from __future__ import annotations
from pathlib import Path

PROJECT_DIR = Path(r"E:\RBG\Update")
DATA_PATH = PROJECT_DIR / "Wobaidul_zafrul_RBG STUDY.sav"
TABLES_PATH = PROJECT_DIR / "accident_history_reanalysis_tables.xlsx"
OUTPUT_DIR = PROJECT_DIR / "Figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

import os
import re
import sys
import math
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
import statsmodels.formula.api as smf
from sklearn.metrics import (
    auc,
    roc_auc_score,
    roc_curve,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------------------------------------------------------
# 1) USER SETTINGS
# -----------------------------------------------------------------------------

# Windows project paths. Run this script from: E:\RBG\Update
PROJECT_DIR = Path(r"E:\RBG\Update")
DATA_SAV = PROJECT_DIR / "Wobaidul_zafrul_RBG STUDY.sav"
TABLE_XLSX = PROJECT_DIR / "accident_history_reanalysis_tables.xlsx"
OUT_DIR = PROJECT_DIR / "Figures"

# Manuscript image settings. Most journals accept 300 dpi; 600 dpi is safer.
DPI = 600
BASE_FONT_SIZE = 12
TITLE_FONT_SIZE = 15
LABEL_FONT_SIZE = 12
ANNOT_FONT_SIZE = 10

# Figure palette: color-blind friendly and restrained.
COLORS = {
    "no": "#2B6C8A",          # muted blue
    "yes": "#C44E52",         # muted red
    "accent": "#D99800",      # gold
    "dark": "#0B1F3A",        # premium dark navy
    "grey": "#6B7280",
    "lightgrey": "#E5E7EB",
}

# -----------------------------------------------------------------------------
# 2) GENERAL HELPERS
# -----------------------------------------------------------------------------

def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_publication_theme() -> None:
    """Set global plotting style for journal-ready figures."""
    sns.set_theme(
        context="paper",
        style="whitegrid",
        font="DejaVu Sans",
        rc={
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "font.size": BASE_FONT_SIZE,
            "axes.titlesize": TITLE_FONT_SIZE,
            "axes.labelsize": LABEL_FONT_SIZE,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.titleweight": "bold",
            "axes.edgecolor": "#1F2937",
            "grid.color": "#E5E7EB",
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        },
    )


def save_figure(fig: plt.Figure, name: str, out_dir: Path = OUT_DIR) -> None:
    """Save figure in journal-friendly raster and vector formats."""
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(out_dir / f"{name}.{ext}", bbox_inches="tight", dpi=DPI)
    plt.close(fig)


def clean_label(x) -> str:
    """Human-friendly labels for values imported from SPSS or Excel."""
    if pd.isna(x):
        return "Missing"
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def find_col(df: pd.DataFrame, candidates: Sequence[str], required: bool = True) -> Optional[str]:
    """Find a column by trying exact and normalized names."""
    columns = list(df.columns)
    norm_map = {re.sub(r"[^a-z0-9]", "", c.lower()): c for c in columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        key = re.sub(r"[^a-z0-9]", "", cand.lower())
        if key in norm_map:
            return norm_map[key]
    if required:
        raise KeyError(
            f"Could not find required column. Tried: {candidates}.\n"
            f"Available columns include: {columns[:60]}"
        )
    return None


def p_format(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "p<0.001"
    return f"p={p:.3f}"


def cliffs_delta(x: Sequence[float], y: Sequence[float]) -> float:
    """Compute Cliff's delta for two independent samples."""
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    y = np.asarray(pd.Series(y).dropna(), dtype=float)
    if len(x) == 0 or len(y) == 0:
        return np.nan
    # Efficient enough for n=874.
    gt = 0
    lt = 0
    for xi in x:
        gt += np.sum(xi > y)
        lt += np.sum(xi < y)
    return (gt - lt) / (len(x) * len(y))


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


def parse_count_percent(cell) -> Tuple[Optional[int], Optional[float]]:
    """Parse strings like '250 (32.1%)'."""
    if pd.isna(cell):
        return None, None
    m = re.match(r"\s*(\d+)\s*\(([-0-9.]+)%\)\s*", str(cell))
    if not m:
        return None, None
    return int(m.group(1)), float(m.group(2))

# -----------------------------------------------------------------------------
# 3) DATA IMPORT AND CLEANING
# -----------------------------------------------------------------------------

def load_raw_spss(path: Path) -> pd.DataFrame:
    """Load the raw SPSS file using pyreadstat."""
    try:
        import pyreadstat  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "pyreadstat is required to read the SPSS .sav file. Install it with:\n"
            "    pip install pyreadstat\n"
            "Alternatively, export the SPSS file as CSV and adapt load_raw_spss()."
        ) from exc
    df, meta = pyreadstat.read_sav(str(path), apply_value_formats=True)
    return df


def prepare_accident_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Create complete-case accident-history analysis dataset."""
    col = {}
    col["accident_history"] = find_col(df, ["Accident_History", "Accident History", "AccidentHistory"])
    col["age"] = find_col(df, ["Age", "age"])
    col["license"] = find_col(df, ["License", "License_Type", "Licence"])
    col["driving_hours"] = find_col(df, ["Driving_H_D", "Driving hours/day", "Driving_Hours_Day", "Driving_Hours"])
    col["rbg"] = find_col(df, ["RBG", "Random blood glucose", "Random_Blood_Glucose"])
    col["betel"] = find_col(df, ["B_Quid", "Betel_Quid", "Betel quid intake", "Betel_Quid_Intake"])

    # Optional columns used by some figures.
    optional_candidates = {
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
    }
    for key, candidates in optional_candidates.items():
        c = find_col(df, candidates, required=False)
        if c is not None:
            col[key] = c

    sub = df[df[col["accident_history"]].notna()].copy()
    sub["accident_label"] = sub[col["accident_history"]].map(lambda v: clean_label(v))
    sub["accident"] = sub["accident_label"].str.lower().eq("yes").astype(int)
    sub["accident_group"] = np.where(sub["accident"].eq(1), "Accident history", "No accident")

    # Numeric columns.
    for key in ["age", "driving_hours", "rbg", "driving_years", "bmi", "pulse"]:
        if key in col:
            sub[col[key]] = pd.to_numeric(sub[col[key]], errors="coerce")

    # Harmonize categorical columns used in the model.
    sub["license_model"] = sub[col["license"]].map(clean_label)
    sub["betel_model"] = sub[col["betel"]].map(clean_label)

    # Common binary/categorical helper variables.
    sub["drive_hours_cat"] = pd.cut(
        sub[col["driving_hours"]],
        bins=[-0.001, 5, 10, np.inf],
        labels=["0-5 h", "6-10 h", "11+ h"],
    )
    if "glycemic" in col:
        sub["glycemic_binary"] = np.where(
            sub[col["glycemic"]].map(clean_label).str.lower().eq("normal"),
            "Normal", "Prediabetes/Diabetes"
        )
    if "urine_glucose" in col:
        sub["glycosuria_binary"] = np.where(
            sub[col["urine_glucose"]].map(clean_label).str.lower().str.contains("negative|no", regex=True),
            "No glycosuria", "Glycosuria"
        )
    if "urine_protein" in col:
        sub["proteinuria_binary"] = np.where(
            sub[col["urine_protein"]].map(clean_label).str.lower().str.contains("negative|no", regex=True),
            "No proteinuria", "Proteinuria"
        )

    # Store standardized column aliases.
    sub = sub.rename(columns={
        col["age"]: "Age",
        col["driving_hours"]: "Driving_H_D",
        col["rbg"]: "RBG",
    })
    if "driving_years" in col:
        sub = sub.rename(columns={col["driving_years"]: "Driving_years"})
    if "bmi" in col:
        sub = sub.rename(columns={col["bmi"]: "BMI"})
    if "pulse" in col:
        sub = sub.rename(columns={col["pulse"]: "Pulse_rate"})

    return sub, col


def fit_final_model(sub: pd.DataFrame):
    """Fit the primary adjusted model used in the report.

    Important Windows/Python 3.12 compatibility fix:
    statsmodels/patsy can fail with pandas StringDtype during prediction.
    Therefore, categorical model columns are converted to ordinary Python
    object strings before fitting and prediction.
    """
    model_df = sub[["accident", "Age", "license_model", "Driving_H_D", "RBG", "betel_model"]].dropna().copy()

    # Force clean numeric columns.
    for c in ["accident", "Age", "Driving_H_D", "RBG"]:
        model_df[c] = pd.to_numeric(model_df[c], errors="coerce")

    # Force categorical columns to standard object dtype, not pandas StringDtype.
    model_df["license_model"] = model_df["license_model"].astype(str).astype(object)
    model_df["betel_model"] = model_df["betel_model"].astype(str).astype(object)
    model_df = model_df.dropna().copy()

    formula = "accident ~ Age + C(license_model) + Driving_H_D + RBG + C(betel_model)"
    model = smf.logit(formula, data=model_df).fit(disp=False, maxiter=200)
    pred = model.predict(model_df)
    return model, model_df, pred


def adjusted_model_table(model) -> pd.DataFrame:
    params = model.params
    conf = model.conf_int()
    pvals = model.pvalues
    rows = []
    for term in params.index:
        if term == "Intercept":
            continue
        rows.append({
            "term": term,
            "label": readable_model_term(term),
            "aOR": np.exp(params[term]),
            "ci_low": np.exp(conf.loc[term, 0]),
            "ci_high": np.exp(conf.loc[term, 1]),
            "p": pvals[term],
        })
    return pd.DataFrame(rows)


def readable_model_term(term: str) -> str:
    replacements = {
        "Age": "Age, per 1 year",
        "Driving_H_D": "Driving hours/day, per 1 hour",
        "RBG": "RBG, per 1 mmol/L",
    }
    if term in replacements:
        return replacements[term]
    m = re.match(r"C\((.*?)\)\[T\.(.*?)\]", term)
    if m:
        var, val = m.group(1), m.group(2)
        if var == "license_model":
            return f"License: {val} vs reference"
        if var == "betel_model":
            return f"Betel quid: {val} vs reference"
        return f"{var}: {val} vs reference"
    return term

# -----------------------------------------------------------------------------
# 4) FIGURES
# -----------------------------------------------------------------------------

def figure_1_outcome_availability(raw_df: pd.DataFrame, sub: pd.DataFrame, colmap: Dict[str, str]) -> None:
    """Outcome availability and accident burden."""
    total = len(raw_df)
    observed = raw_df[colmap["accident_history"]].notna().sum()
    missing = total - observed
    no_acc = int((sub["accident"] == 0).sum())
    yes_acc = int((sub["accident"] == 1).sum())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [1.0, 1.15]})

    ax = axes[0]
    bars = ax.barh(["Observed", "Missing"], [observed, missing], color=[COLORS["no"], COLORS["lightgrey"]])
    ax.set_xlim(0, total * 1.08)
    ax.set_xlabel("Participants")
    ax.set_title("Accident-history outcome availability")
    for b, value in zip(bars, [observed, missing]):
        ax.text(value + total * 0.015, b.get_y() + b.get_height() / 2,
                f"{value:,}\n({value/total*100:.1f}%)", va="center", fontsize=10)
    ax.grid(axis="x", alpha=0.4)

    ax = axes[1]
    sizes = [no_acc, yes_acc]
    wedges, _ = ax.pie(
        sizes,
        startangle=90,
        colors=[COLORS["no"], COLORS["yes"]],
        wedgeprops=dict(width=0.38, edgecolor="white", linewidth=2),
    )
    ax.text(0, 0.05, f"{yes_acc/(yes_acc+no_acc)*100:.1f}%", ha="center", va="center",
            fontsize=24, fontweight="bold", color=COLORS["dark"])
    ax.text(0, -0.18, "reported accident\nhistory", ha="center", va="center", fontsize=11)
    ax.legend(wedges, [f"No accident ({no_acc:,})", f"Accident history ({yes_acc:,})"],
              loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=1, frameon=False)
    ax.set_title("Observed accident-history distribution")

    fig.suptitle("Figure 1. Outcome completeness and analytic cohort", y=1.02,
                 fontsize=16, fontweight="bold", color=COLORS["dark"])
    fig.text(0.01, -0.02,
             "Complete-case accident-history analysis: participants with missing accident-history values are excluded from accident models.",
             fontsize=9, color=COLORS["grey"])
    fig.tight_layout()
    save_figure(fig, "Figure_1_outcome_availability_and_accident_burden")


def figure_2_continuous_signature(sub: pd.DataFrame) -> None:
    """Continuous distributions by accident status with non-parametric effect sizes."""
    candidates = [
        ("Age", "Age (years)"),
        ("Driving_H_D", "Driving hours/day"),
        ("Driving_years", "Driving years"),
        ("RBG", "Random blood glucose (mmol/L)"),
        ("BMI", "BMI (kg/m²)"),
        ("Pulse_rate", "Pulse rate (bpm)"),
    ]
    vars_available = [(c, lab) for c, lab in candidates if c in sub.columns and sub[c].notna().sum() > 20]
    vars_available = vars_available[:6]

    n = len(vars_available)
    ncols = 3
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.6 * nrows))
    axes = np.asarray(axes).reshape(-1)
    order = ["No accident", "Accident history"]

    for ax, (col, label) in zip(axes, vars_available):
        plot_df = sub[[col, "accident_group"]].dropna()
        sns.violinplot(
            data=plot_df, x="accident_group", y=col, order=order,
            palette=[COLORS["no"], COLORS["yes"]], inner=None, cut=0,
            linewidth=1, alpha=0.85, ax=ax,
        )
        sns.boxplot(
            data=plot_df, x="accident_group", y=col, order=order,
            width=0.22, showcaps=True, showfliers=False,
            boxprops={"facecolor": "white", "edgecolor": COLORS["dark"], "linewidth": 1.2},
            medianprops={"color": COLORS["dark"], "linewidth": 1.4},
            whiskerprops={"color": COLORS["dark"], "linewidth": 1.0},
            capprops={"color": COLORS["dark"], "linewidth": 1.0},
            ax=ax,
        )
        sns.stripplot(
            data=plot_df.sample(min(len(plot_df), 450), random_state=7),
            x="accident_group", y=col, order=order,
            color="#111827", alpha=0.20, jitter=0.18, size=2.0, ax=ax,
        )
        x_no = plot_df.loc[plot_df["accident_group"].eq("No accident"), col]
        x_yes = plot_df.loc[plot_df["accident_group"].eq("Accident history"), col]
        p = stats.mannwhitneyu(x_no, x_yes, alternative="two-sided", nan_policy="omit").pvalue
        delta = cliffs_delta(x_yes, x_no)  # positive means higher in accident group
        ax.set_title(label)
        ax.set_xlabel("")
        ax.set_ylabel(label)
        ax.text(0.02, 0.96, f"{p_format(p)}\nCliff's δ={delta:.2f}", transform=ax.transAxes,
                ha="left", va="top", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COLORS["lightgrey"], alpha=0.95))
        ax.tick_params(axis="x", rotation=10)

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle("Figure 2. Continuous-variable accident signature", y=1.01,
                 fontsize=16, fontweight="bold", color=COLORS["dark"])
    fig.text(0.01, -0.01,
             "Violin width shows distribution density; box shows median/IQR; dots are a reproducible random sample for visual clarity. P values are Mann-Whitney U tests.",
             fontsize=9, color=COLORS["grey"])
    fig.tight_layout()
    save_figure(fig, "Figure_2_continuous_variable_accident_signature")


def figure_3_adjusted_forest(model) -> pd.DataFrame:
    """Adjusted multivariable forest plot."""
    tab = adjusted_model_table(model)
    tab = tab.sort_values("aOR", ascending=True).reset_index(drop=True)
    tab.to_csv(OUT_DIR / "adjusted_model_for_forest_plot.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    y = np.arange(len(tab))
    sig = tab["p"] < 0.05
    colors = np.where(sig, COLORS["yes"], COLORS["grey"])

    ax.errorbar(
        tab["aOR"], y,
        xerr=[tab["aOR"] - tab["ci_low"], tab["ci_high"] - tab["aOR"]],
        fmt="none", ecolor="#374151", elinewidth=1.6, capsize=3, zorder=1,
    )
    ax.scatter(tab["aOR"], y, s=75, color=colors, edgecolor="white", linewidth=0.8, zorder=2)
    ax.axvline(1, color=COLORS["dark"], linestyle="--", linewidth=1.2)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(mticker.FixedLocator([0.5, 0.75, 1, 1.5, 2, 3, 4]))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.set_yticks(y)
    ax.set_yticklabels(tab["label"])
    ax.set_xlabel("Adjusted odds ratio, log scale")
    ax.set_title("Adjusted predictors of accident history")
    ax.grid(axis="x", which="major", alpha=0.45)
    ax.grid(axis="y", alpha=0)

    for i, row in tab.iterrows():
        ax.text(row["ci_high"] * 1.08, i,
                f"aOR {row['aOR']:.2f} ({row['ci_low']:.2f}-{row['ci_high']:.2f}); {p_format(row['p'])}",
                va="center", fontsize=9)

    x_max = max(tab["ci_high"]) * 2.2
    ax.set_xlim(0.75 if min(tab["ci_low"]) > 0.75 else max(0.25, min(tab["ci_low"]) * 0.75), x_max)
    fig.suptitle("Figure 3. Multivariable adjusted effect-size profile", y=1.02,
                 fontsize=16, fontweight="bold", color=COLORS["dark"])
    fig.text(0.01, -0.03,
             "Model: accident history ~ age + license + daily driving hours + random blood glucose + betel quid intake. Red markers indicate p<0.05.",
             fontsize=9, color=COLORS["grey"])
    fig.tight_layout()
    save_figure(fig, "Figure_3_adjusted_odds_ratio_forest_plot")
    return tab


def figure_4_predicted_probability_curves(model, model_df: pd.DataFrame) -> None:
    """Adjusted predicted probabilities across RBG and driving hours."""
    age_med = float(model_df["Age"].median())
    rbg_grid = np.linspace(
        max(3.2, model_df["RBG"].quantile(0.02)),
        min(model_df["RBG"].quantile(0.98), max(model_df["RBG"].max(), 8.5)),
        120,
    )
    drive_levels = [5, 8, 11]
    licenses = list(pd.Series(model_df["license_model"].dropna().unique()).sort_values())
    betel_levels = list(pd.Series(model_df["betel_model"].dropna().unique()).sort_values())

    # Limit to sensible 2 x 2 display; if labels differ, use first two sorted levels.
    licenses = licenses[:2]
    betel_levels = betel_levels[:2]

    fig, axes = plt.subplots(1, len(licenses), figsize=(13.5, 5.4), sharey=True)
    if len(licenses) == 1:
        axes = [axes]

    line_styles = {5: "-", 8: "--", 11: ":"}
    line_colors = [COLORS["no"], COLORS["yes"]]

    for ax, lic in zip(axes, licenses):
        for b_i, betel in enumerate(betel_levels):
            for dh in drive_levels:
                pred_df = pd.DataFrame({
                    "Age": np.repeat(age_med, len(rbg_grid)).astype(float),
                    "license_model": np.repeat(str(lic), len(rbg_grid)).astype(object),
                    "Driving_H_D": np.repeat(float(dh), len(rbg_grid)).astype(float),
                    "RBG": rbg_grid.astype(float),
                    "betel_model": np.repeat(str(betel), len(rbg_grid)).astype(object),
                })

                # Keep categorical predictors as plain object dtype for patsy/statsmodels.
                pred_df["license_model"] = pred_df["license_model"].astype(object)
                pred_df["betel_model"] = pred_df["betel_model"].astype(object)
                prob = model.predict(pred_df)
                ax.plot(
                    rbg_grid, prob * 100,
                    linestyle=line_styles[dh],
                    color=line_colors[b_i % len(line_colors)],
                    linewidth=2.0,
                    label=f"{betel}, {dh} h/day",
                )
        ax.set_title(f"License: {lic}")
        ax.set_xlabel("Random blood glucose (mmol/L)")
        ax.set_ylim(0, None)
        ax.grid(True, alpha=0.35)
    axes[0].set_ylabel("Adjusted predicted probability of accident history (%)")
    axes[-1].legend(title="Betel quid, driving hours", bbox_to_anchor=(1.03, 1), loc="upper left", frameon=False)

    fig.suptitle("Figure 4. Adjusted response surface: metabolic and occupational risk gradient",
                 y=1.02, fontsize=16, fontweight="bold", color=COLORS["dark"])
    fig.text(0.01, -0.02,
             f"Predictions are generated from the final logistic model at median age ({age_med:.1f} years), varying RBG, license status, betel quid intake, and daily driving hours.",
             fontsize=9, color=COLORS["grey"])
    fig.tight_layout()
    save_figure(fig, "Figure_4_adjusted_predicted_accident_probability_curves")


def figure_5_categorical_risk_heatmap(sub: pd.DataFrame, colmap: Dict[str, str]) -> None:
    """Heatmap of accident prevalence across clinically/intervention-relevant subgroups."""
    var_specs: List[Tuple[str, str]] = []

    # Model and table variables.
    var_specs.append(("license_model", "License"))
    var_specs.append(("betel_model", "Betel quid"))
    if "smoking" in colmap:
        sub["smoking_plot"] = sub[colmap["smoking"]].map(clean_label)
        var_specs.append(("smoking_plot", "Smoking"))
    if "sleep" in colmap:
        sub["sleep_plot"] = sub[colmap["sleep"]].map(clean_label)
        var_specs.append(("sleep_plot", "Sleep status"))
    if "glycemic_binary" in sub.columns:
        var_specs.append(("glycemic_binary", "Glycemic status"))
    var_specs.append(("drive_hours_cat", "Driving hours/day"))
    if "glycosuria_binary" in sub.columns:
        var_specs.append(("glycosuria_binary", "Urinary glucose"))
    if "proteinuria_binary" in sub.columns:
        var_specs.append(("proteinuria_binary", "Urinary protein"))

    rows = []
    for var, label in var_specs:
        tmp = sub[[var, "accident"]].dropna().copy()
        # Preserve a reasonable category order.
        for cat, g in tmp.groupby(var, observed=True):
            n = len(g)
            if n < 10:
                continue
            events = int(g["accident"].sum())
            prev = events / n * 100
            rows.append({"Variable": label, "Category": str(cat), "n": n, "events": events, "prevalence": prev})

    heat = pd.DataFrame(rows)
    heat["row_label"] = heat["Variable"] + " | " + heat["Category"]
    heat = heat.sort_values(["Variable", "prevalence"], ascending=[True, False]).reset_index(drop=True)

    fig_h = max(6.0, 0.38 * len(heat) + 1.8)
    fig, ax = plt.subplots(figsize=(8.5, fig_h))
    values = heat[["prevalence"]].to_numpy()
    annot = heat.apply(lambda r: f"{r['prevalence']:.1f}%\n{int(r['events'])}/{int(r['n'])}", axis=1).to_numpy().reshape(-1, 1)

    sns.heatmap(
        values,
        annot=annot,
        fmt="",
        cmap=sns.light_palette(COLORS["yes"], as_cmap=True),
        cbar_kws={"label": "Accident history prevalence (%)"},
        linewidths=0.6,
        linecolor="white",
        yticklabels=heat["row_label"],
        xticklabels=["Accident history"],
        ax=ax,
    )
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.set_title("Subgroup accident-history prevalence landscape")

    fig.suptitle("Figure 5. Categorical risk landscape for accident history", y=1.005,
                 fontsize=16, fontweight="bold", color=COLORS["dark"])
    fig.text(0.01, -0.01,
             "Cells show accident-history prevalence and event/total counts. This is descriptive and complements, but does not replace, adjusted regression.",
             fontsize=9, color=COLORS["grey"])
    fig.tight_layout()
    save_figure(fig, "Figure_5_categorical_accident_risk_heatmap")


def figure_6_model_performance(model_df: pd.DataFrame, pred: np.ndarray) -> None:
    """ROC and calibration plot for the final logistic model."""
    y = model_df["accident"].to_numpy()
    fpr, tpr, _ = roc_curve(y, pred)
    roc_auc = auc(fpr, tpr)
    brier = brier_score_loss(y, pred)
    frac_pos, mean_pred = calibration_curve(y, pred, n_bins=8, strategy="quantile")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))

    ax = axes[0]
    ax.plot(fpr, tpr, color=COLORS["yes"], linewidth=2.4, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color=COLORS["grey"], linestyle="--", linewidth=1.2)
    ax.set_xlabel("1 - Specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title("Discrimination: ROC curve")
    ax.legend(frameon=False, loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")

    ax = axes[1]
    ax.plot([0, 1], [0, 1], color=COLORS["grey"], linestyle="--", linewidth=1.2, label="Perfect calibration")
    ax.plot(mean_pred, frac_pos, marker="o", color=COLORS["no"], linewidth=2.2, label="Observed calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed accident proportion")
    ax.set_title("Calibration by prediction quantiles")
    ax.set_xlim(0, max(0.30, float(np.max(mean_pred)) * 1.15))
    ax.set_ylim(0, max(0.30, float(np.max(frac_pos)) * 1.15))
    ax.legend(frameon=False, loc="upper left")
    ax.text(0.98, 0.04, f"Brier score = {brier:.3f}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COLORS["lightgrey"], alpha=0.95))

    fig.suptitle("Figure 6. Final model performance and reliability", y=1.02,
                 fontsize=16, fontweight="bold", color=COLORS["dark"])
    fig.text(0.01, -0.02,
             "ROC quantifies discrimination; calibration compares predicted versus observed accident-history risk across risk strata.",
             fontsize=9, color=COLORS["grey"])
    fig.tight_layout()
    save_figure(fig, "Figure_6_model_roc_and_calibration")

# -----------------------------------------------------------------------------
# 5) OPTIONAL TABLE-ONLY FIGURE FROM EXCEL: ENHANCED UNADJUSTED FOREST PLOT
# -----------------------------------------------------------------------------

def figure_7_unadjusted_forest_from_excel(table_xlsx: Path) -> None:
    """Create an optional unadjusted forest plot directly from the Excel workbook."""
    if not table_xlsx.exists():
        return
    df = pd.read_excel(table_xlsx, sheet_name="Unadjusted_ORs", header=2)
    df = df.rename(columns={
        "Variable": "Variable",
        "Category vs reference": "Category",
        "OR": "OR",
        "95% CI low": "ci_low",
        "95% CI high": "ci_high",
        "P value": "p",
    })
    df["Variable"] = df["Variable"].ffill()
    df = df[df["OR"].notna()].copy()
    df["p_num"] = df["p"].map(parse_p_value)
    df["label"] = df["Variable"].astype(str) + ": " + df["Category"].astype(str)
    df = df.sort_values("OR", ascending=True).reset_index(drop=True)

    fig_h = max(7, 0.38 * len(df) + 1.8)
    fig, ax = plt.subplots(figsize=(9.5, fig_h))
    y = np.arange(len(df))
    colors = np.where(df["p_num"] < 0.05, COLORS["yes"], COLORS["grey"])
    ax.errorbar(
        df["OR"], y,
        xerr=[df["OR"] - df["ci_low"], df["ci_high"] - df["OR"]],
        fmt="none", ecolor="#374151", elinewidth=1.4, capsize=3,
    )
    ax.scatter(df["OR"], y, s=58, color=colors, edgecolor="white", linewidth=0.8)
    ax.axvline(1, color=COLORS["dark"], linestyle="--", linewidth=1.1)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(df["label"])
    ax.set_xlabel("Unadjusted odds ratio, log scale")
    ax.set_title("Unadjusted association screening")
    ax.grid(axis="x", alpha=0.45)
    ax.grid(axis="y", alpha=0)
    ax.set_xlim(max(0.4, df["ci_low"].min() * 0.75), df["ci_high"].max() * 1.25)
    fig.suptitle("Supplementary Figure. Unadjusted accident-history associations", y=1.005,
                 fontsize=16, fontweight="bold", color=COLORS["dark"])
    fig.text(0.01, -0.01,
             "This figure is best used as a screening/secondary figure; adjusted Figure 3 should be emphasized for inference.",
             fontsize=9, color=COLORS["grey"])
    fig.tight_layout()
    save_figure(fig, "Supplementary_Figure_unadjusted_odds_ratio_forest_plot")

# -----------------------------------------------------------------------------
# 6) MAIN EXECUTION
# -----------------------------------------------------------------------------

def main() -> None:
    ensure_output_dir(OUT_DIR)
    set_publication_theme()

    if not DATA_SAV.exists():
        raise FileNotFoundError(f"SPSS file not found: {DATA_SAV}")

    raw_df = load_raw_spss(DATA_SAV)
    sub, colmap = prepare_accident_dataset(raw_df)
    model, model_df, pred = fit_final_model(sub)

    # Save key reproducibility outputs.
    model_summary = adjusted_model_table(model)
    model_summary.to_csv(OUT_DIR / "final_adjusted_model_summary.csv", index=False)
    with open(OUT_DIR / "model_summary.txt", "w", encoding="utf-8") as f:
        f.write(str(model.summary()))
        f.write("\n\nAUC: %.3f\n" % roc_auc_score(model_df["accident"], pred))
        f.write("Brier score: %.3f\n" % brier_score_loss(model_df["accident"], pred))
        f.write("N used: %d\n" % len(model_df))
        f.write("Accident events: %d\n" % int(model_df["accident"].sum()))

    # Create figures.
    figure_1_outcome_availability(raw_df, sub, colmap)
    figure_2_continuous_signature(sub)
    figure_3_adjusted_forest(model)
    figure_4_predicted_probability_curves(model, model_df)
    figure_5_categorical_risk_heatmap(sub, colmap)
    figure_6_model_performance(model_df, pred)
    figure_7_unadjusted_forest_from_excel(TABLE_XLSX)

    print("Done. Figures saved to:", OUT_DIR)
    print("Generated files:")
    for path in sorted(OUT_DIR.glob("*")):
        print(" -", path.name)


if __name__ == "__main__":
    main()
