#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Lancet-style multi-panel figure generator for the 874-driver accident-history re-analysis
=======================================================================================

Purpose
-------
Creates manuscript-ready, reviewer-friendly, multi-panel figures from the complete-case
accident-history re-analysis. The figures are designed to be beautiful, readable, and
plain-language while preserving statistical rigor.

Inputs expected in the project folder:
    complete_case_874_dataset.csv
    accident_history_874_reanalysis_outputs.xlsx

Default Windows run:
    cd E:\Zafrul_Sir\RBG\Update
    python .\create_lancet_masterpiece_reanalysis_874_figures_v3.py

Outputs:
    E:\Zafrul_Sir\RBG\Update\Figures\Reanalysis_874_Lancet_Masterpiece_v3

Files produced:
    Figure_1_Discovery_Atlas.*
    Figure_2_License_Betel_Pathway.*
    Figure_3_Smoking_Betel_RBG_Pathway.*
    Figure_4_Core_Model_Risk_Surface.*
    Figure_5_Domain_Signal_Atlas.*
    Figure_6_Reviewer_Friendly_Dashboard.*
    plus CSVs used for reproducibility.

Design rules
------------
- 4–6 panels per figure.
- No cramped captions inside panels.
- Large figure size and controlled spacing to avoid overlap.
- Plain-language titles.
- Consistent but premium palette.
- Every panel answers a different question.
- Uses only 874 complete cases for accident-history analysis.

Scientific caution
------------------
These figures show associations, not causation. Pathway panels are exploratory.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import fill
import math
import re
import warnings
import gc
import sys
import subprocess
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch
from matplotlib.lines import Line2D
from scipy import stats
from scipy.special import expit
import statsmodels.formula.api as smf
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, brier_score_loss

warnings.filterwarnings("ignore")

# =============================================================================
# 1) PATHS AND SETTINGS
# =============================================================================

# The script is robust to being moved. It first uses the current working directory.
# If you want to force a fixed path, uncomment and edit PROJECT_DIR below.
PROJECT_DIR = Path.cwd()
# PROJECT_DIR = Path(r"E:\Zafrul_Sir\RBG\Update")

DATA_CSV = PROJECT_DIR / "complete_case_874_dataset.csv"
TABLE_XLSX = PROJECT_DIR / "accident_history_874_reanalysis_outputs.xlsx"
OUT_DIR = PROJECT_DIR / "Figures" / "Reanalysis_874_Lancet_Masterpiece_v3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300
SAVE_SVG = False  # set True only if you specifically need SVG; complex heatmaps can make SVG very slow/large
RANDOM_STATE = 20260516
np.random.seed(RANDOM_STATE)

# =============================================================================
# 2) STYLE
# =============================================================================

# Premium restrained palette
NAVY = "#071E3D"
INK = "#111827"
SLATE = "#334155"
MUTED = "#64748B"
GRID = "#E5E7EB"
BG = "#FBFCFF"
WHITE = "#FFFFFF"
BLUE = "#2F6F9F"
SKY = "#86BBD8"
TEAL = "#2A9D8F"
GOLD = "#D99A2B"
AMBER = "#F2C14E"
CRIMSON = "#B23A48"
ROSE = "#E295A3"
PURPLE = "#6D5BD0"
VIOLET = "#8B5CF6"
GREY = "#94A3B8"
GREEN = "#3B8C66"

DOMAIN_COLORS = {
    "Demographic factors": BLUE,
    "Occupational factors": CRIMSON,
    "Lifestyle and behavioral factors": GOLD,
    "Metabolic and clinical factors": TEAL,
    "Urinary and renal factors": PURPLE,
    "Other factors": GREY,
}


def set_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.titlesize": 13.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 11.5,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 10.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#263241",
        "axes.linewidth": 0.8,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def savefig(fig: plt.Figure, stem: str) -> None:
    exts = ["png", "pdf"] + (["svg"] if SAVE_SVG else [])
    for ext in exts:
        fig.savefig(OUT_DIR / f"{stem}.{ext}", dpi=DPI, bbox_inches="tight", pad_inches=0.20, facecolor=WHITE)
    plt.close(fig)
    plt.close("all")
    gc.collect()


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(0.012, 0.985, label, transform=ax.transAxes, ha="left", va="top",
            fontsize=14.5, fontweight="bold", color=WHITE, zorder=30,
            bbox=dict(boxstyle="round,pad=0.18,rounding_size=0.05", fc=NAVY, ec=NAVY, lw=0))


def add_soft_background(ax: plt.Axes, c1: str = "#F8FAFC", c2: str = "#FFF7ED", alpha: float = 0.70) -> None:
    """Adds a subtle horizontal gradient inside an axes, without depending on seaborn."""
    xlim = ax.get_xlim(); ylim = ax.get_ylim()
    grad = np.linspace(0, 1, 256)[None, :]
    cmap = mpl.colors.LinearSegmentedColormap.from_list("panel_grad", [c1, c2])
    ax.imshow(grad, extent=[xlim[0], xlim[1], ylim[0], ylim[1]], aspect="auto", cmap=cmap,
              alpha=alpha, zorder=-20, interpolation="bicubic")
    ax.set_xlim(xlim); ax.set_ylim(ylim)


def title_block(fig: plt.Figure, title: str, subtitle: str = "") -> None:
    fig.text(0.5, 0.982, title, ha="center", va="top", fontsize=19, fontweight="bold", color=NAVY)
    if subtitle:
        fig.text(0.5, 0.954, subtitle, ha="center", va="top", fontsize=11.5, color=MUTED)


def footer(fig: plt.Figure, text: str) -> None:
    fig.text(0.012, 0.016, text, ha="left", va="bottom", fontsize=10.5, color=MUTED)


def clean_term(t: str) -> str:
    replacements = {
        "C(License_clean)[T.Renew]": "Renew license\n(vs new)",
        "Betel_binary": "Betel quid\n(yes vs no)",
        "Smoking_binary": "Smoking\n(yes vs no)",
        "RBG_num": "RBG\n(per mmol/L)",
        "Driving_hours_num": "Driving hours/day\n(per hour)",
        "Age_num": "Age\n(per year)",
        "license_renew": "Renew license\n(vs new)",
        "RBG pathway model": "RBG\n(per mmol/L)",
    }
    if t in replacements:
        return replacements[t]
    t = re.sub(r"C\((.*?)\)\[T\.(.*?)\]", lambda m: f"{m.group(1)}: {m.group(2)}", str(t))
    t = t.replace("_", " ")
    return fill(t, 24)


def fmt_p(p: float) -> str:
    if pd.isna(p): return ""
    return "p<0.001" if p < 0.001 else f"p={p:.3f}"


def ci_wilson(x: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    phat = x / n
    denom = 1 + z*z/n
    centre = phat + z*z/(2*n)
    half = z * math.sqrt((phat*(1-phat) + z*z/(4*n))/n)
    return ((centre-half)/denom, (centre+half)/denom)


def odds_ci(a, b, c, d) -> Tuple[float, float, float]:
    # exposed accident, exposed no, unexposed accident, unexposed no
    a, b, c, d = a+0.5, b+0.5, c+0.5, d+0.5
    OR = (a*d)/(b*c)
    se = math.sqrt(1/a+1/b+1/c+1/d)
    return OR, math.exp(math.log(OR)-1.96*se), math.exp(math.log(OR)+1.96*se)


def prevalence_table(df: pd.DataFrame, group: str, label_map: Optional[Dict]=None) -> pd.DataFrame:
    g = df.groupby(group, dropna=False)["accident"].agg(["sum", "count"]).reset_index()
    g["prev"] = 100*g["sum"]/g["count"]
    lows, highs = [], []
    for _, r in g.iterrows():
        lo, hi = ci_wilson(int(r["sum"]), int(r["count"]))
        lows.append(lo*100); highs.append(hi*100)
    g["lo"] = lows; g["hi"] = highs
    if label_map:
        g["label"] = g[group].map(label_map)
    else:
        g["label"] = g[group].astype(str)
    return g


def logistic_or_table(formula: str, data: pd.DataFrame) -> pd.DataFrame:
    m = smf.logit(formula, data=data).fit(disp=False, maxiter=200)
    ci = m.conf_int()
    rows = []
    for term in m.params.index:
        if term == "Intercept":
            continue
        rows.append({
            "term": term,
            "label": clean_term(term),
            "OR": float(np.exp(m.params[term])),
            "lo": float(np.exp(ci.loc[term, 0])),
            "hi": float(np.exp(ci.loc[term, 1])),
            "p": float(m.pvalues[term]),
            "coef": float(m.params[term]),
        })
    return pd.DataFrame(rows), m


def forest(ax: plt.Axes, df: pd.DataFrame, title: str, xlim: Optional[Tuple[float, float]]=None,
           color_col: Optional[str] = None, annotate: bool = True) -> None:
    d = df.copy().replace([np.inf, -np.inf], np.nan).dropna(subset=["OR", "lo", "hi"])
    if d.empty:
        ax.axis("off"); return
    d = d.sort_values("OR")
    y = np.arange(len(d))
    colors = [color_col if color_col else (CRIMSON if p < 0.05 else BLUE) for p in d.get("p", pd.Series([1]*len(d)))]
    ax.axvline(1, color=SLATE, ls="--", lw=1.1, zorder=1)
    ax.hlines(y, d["lo"], d["hi"], color="#3B4757", lw=1.5, zorder=2)
    ax.scatter(d["OR"], y, s=70, color=colors, edgecolor=WHITE, lw=0.8, zorder=3)
    ax.set_xscale("log")
    if xlim:
        ax.set_xlim(*xlim)
    ax.set_yticks(y); ax.set_yticklabels(d["label"].tolist())
    ax.set_xlabel("Odds ratio (log scale)")
    ax.set_title(title, pad=10)
    ax.grid(axis="x", alpha=0.45)
    if annotate:
        xmax = ax.get_xlim()[1]
        for yi, (_, r) in enumerate(d.iterrows()):
            xtext = min(max(r["hi"]*1.10, r["OR"]*1.15), xmax/1.18)
            ax.text(xtext, yi, f"{r['OR']:.2f}  {fmt_p(r.get('p', np.nan))}", va="center", fontsize=9.8, color=SLATE)


def bar_label(ax, x, y, text, color=INK, dy=0.5, fs=10.5):
    ax.text(x, y + dy, text, ha="center", va="bottom", fontsize=fs, fontweight="bold", color=color)


def box_card(ax: plt.Axes, xy, width, height, title, body, fc, ec=None, title_color=INK, body_color=SLATE, fs=11):
    ec = ec or fc
    rect = FancyBboxPatch(xy, width, height, boxstyle="round,pad=0.016,rounding_size=0.025",
                          fc=fc, ec=ec, lw=1.2, alpha=0.97)
    ax.add_patch(rect)
    ax.text(xy[0]+width*0.06, xy[1]+height*0.65, title, ha="left", va="center",
            fontsize=fs+1, fontweight="bold", color=title_color)
    ax.text(xy[0]+width*0.06, xy[1]+height*0.34, fill(body, 32), ha="left", va="center",
            fontsize=fs-0.5, color=body_color)

# =============================================================================
# 3) LOAD DATA
# =============================================================================

def load_inputs():
    if not DATA_CSV.exists():
        raise FileNotFoundError(f"Missing CSV: {DATA_CSV}\nRun the 874 re-analysis script first or copy the file into the project folder.")
    if not TABLE_XLSX.exists():
        raise FileNotFoundError(f"Missing Excel workbook: {TABLE_XLSX}")
    df = pd.read_csv(DATA_CSV)
    xl = pd.ExcelFile(TABLE_XLSX)
    sheets = {s: pd.read_excel(xl, s) for s in xl.sheet_names}

    # Ensure key variables are properly typed
    for c in ["accident", "Age_num", "RBG_num", "Driving_hours_num", "Smoking_binary", "Betel_binary"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "License_clean" in df.columns:
        df["License_clean"] = df["License_clean"].astype(str)
    return df, sheets


def prepare_exposure_groups(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["license_renew"] = np.where(d["License_clean"].str.lower().str.contains("renew"), 1, 0)
    d["license_group"] = np.where(d["license_renew"].eq(1), "Renew license", "New license")
    d["betel_group"] = np.where(d["Betel_binary"].eq(1), "Betel", "No betel")
    d["smoking_group"] = np.where(d["Smoking_binary"].eq(1), "Smoking", "No smoking")
    d["smoke_betel_group"] = pd.Series(pd.NA, index=d.index, dtype="object")
    d.loc[(d["Smoking_binary"].eq(0)) & (d["Betel_binary"].eq(0)), "smoke_betel_group"] = "Neither"
    d.loc[(d["Smoking_binary"].eq(1)) & (d["Betel_binary"].eq(0)), "smoke_betel_group"] = "Smoking only"
    d.loc[(d["Smoking_binary"].eq(0)) & (d["Betel_binary"].eq(1)), "smoke_betel_group"] = "Betel only"
    d.loc[(d["Smoking_binary"].eq(1)) & (d["Betel_binary"].eq(1)), "smoke_betel_group"] = "Both"
    d["RBG_cat"] = pd.cut(d["RBG_num"], bins=[-np.inf, 5.6, 7.8, np.inf], labels=["<5.6", "5.6–7.7", "≥7.8"])
    d["Drive_cat"] = pd.cut(d["Driving_hours_num"], bins=[-np.inf, 5, 10, np.inf], labels=["0–5 h", "6–10 h", "11+ h"])
    return d

# =============================================================================
# 4) FIGURES
# =============================================================================

def figure1_discovery(df: pd.DataFrame, sheets: Dict[str, pd.DataFrame]):
    screening = sheets["All_variable_screening"].copy()
    cat = sheets["Category_specific_ORs"].copy()
    core = sheets["Core_adjusted_model"].copy()
    screening["p_value"] = pd.to_numeric(screening["p_value"], errors="coerce")
    screening["FDR_q_value"] = pd.to_numeric(screening["FDR_q_value"], errors="coerce")
    screening["strength"] = -np.log10(screening["p_value"].clip(lower=1e-300))
    screening["domain_color"] = screening["group"].map(DOMAIN_COLORS).fillna(GREY)
    scr = screening.dropna(subset=["p_value"]).sort_values("p_value").reset_index(drop=True)
    scr["rank"] = np.arange(1, len(scr)+1)

    fig = plt.figure(figsize=(21, 13.2), facecolor=WHITE)
    gs = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[1.33, 1.0, 1.0], height_ratios=[1, 1], wspace=0.34, hspace=0.38)
    axA = fig.add_subplot(gs[0, 0:2])
    axB = fig.add_subplot(gs[0, 2])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])
    axE = fig.add_subplot(gs[1, 2])

    # A: Discovery skyline
    add_panel_label(axA, "A")
    axA.scatter(scr["rank"], scr["strength"], s=46, c=scr["domain_color"], edgecolor=WHITE, lw=0.5, alpha=0.95)
    axA.axhline(-np.log10(0.05), color=SLATE, ls="--", lw=1, label="p=0.05")
    fdr = scr[scr["FDR_q_value"].le(0.10, fill_value=False)]
    if len(fdr):
        axA.scatter(fdr["rank"], fdr["strength"], s=92, facecolors="none", edgecolors=NAVY, lw=1.3, label="FDR q≤0.10")
    top = scr.head(8)
    offsets = [(0, 12), (8, 18), (8, -16), (10, 10), (10, -18), (12, 14), (12, -14), (10, 8)]
    for (i, r), off in zip(top.iterrows(), offsets):
        label = str(r["variable"]).replace("_", " ")
        axA.annotate(fill(label, 18), xy=(r["rank"], r["strength"]), xytext=off,
                     textcoords="offset points", fontsize=9.4, color=INK,
                     arrowprops=dict(arrowstyle="-", color=GREY, lw=0.7), clip_on=False)
    handles = [Line2D([0], [0], marker="o", color="w", label=k, markerfacecolor=v, markersize=8)
               for k, v in DOMAIN_COLORS.items()]
    axA.legend(handles=handles, ncol=3, loc="upper right", frameon=False, bbox_to_anchor=(1.0, 1.0), fontsize=9.5)
    axA.set_xlabel("All screened variables ranked by p-value")
    axA.set_ylabel("Association strength, −log10(p)")
    axA.set_title("All candidate factors screened inside the 874-driver cohort")
    axA.grid(alpha=0.35)
    add_soft_background(axA, "#F8FAFC", "#FEF3C7", 0.26)

    # B: Domain contribution bars
    add_panel_label(axB, "B")
    domains = []
    for g, s in screening.groupby("group"):
        domains.append({"group": g, "screened": len(s), "p05": int((s["p_value"] < 0.05).sum()), "fdr": int((s["FDR_q_value"] <= 0.10).sum())})
    dom = pd.DataFrame(domains).sort_values("screened")
    y = np.arange(len(dom))
    axB.barh(y, dom["screened"], color="#E5E7EB", height=0.65, label="Screened")
    axB.barh(y, dom["p05"], color="#A5B4FC", height=0.42, label="p<0.05")
    axB.barh(y, dom["fdr"], color=PURPLE, height=0.22, label="FDR q≤0.10")
    axB.set_yticks(y); axB.set_yticklabels([fill(x, 20) for x in dom["group"]])
    axB.set_xlabel("Number of variables")
    axB.set_title("Where did the signals come from?")
    axB.legend(frameon=False, loc="lower right")
    axB.grid(axis="x", alpha=0.35)
    add_soft_background(axB, "#F8FAFC", "#FFF7ED", 0.24)

    # C: Top continuous differences
    add_panel_label(axC, "C")
    cont = screening[screening["type"].eq("numeric")].copy().sort_values("p_value").head(10)
    cont["diff"] = np.nan
    # parse medians from strings if available
    def med(s):
        try: return float(str(s).split("[")[0].strip())
        except Exception: return np.nan
    cont["no_med"] = cont["no_accident_median_IQR"].map(med)
    cont["acc_med"] = cont["accident_median_IQR"].map(med)
    cont["diff"] = cont["acc_med"] - cont["no_med"]
    cont = cont.sort_values("diff")
    yy = np.arange(len(cont))
    axC.axvline(0, color=SLATE, lw=1)
    axC.hlines(yy, 0, cont["diff"], color=GREY, lw=1.3)
    axC.scatter(cont["diff"], yy, s=65, c=cont["group"].map(DOMAIN_COLORS).fillna(GREY), edgecolor=WHITE, lw=0.8)
    axC.set_yticks(yy); axC.set_yticklabels([fill(v.replace("_", " "), 19) for v in cont["variable"]])
    axC.set_xlabel("Median difference: accident − no accident")
    axC.set_title("Top continuous differences")
    axC.grid(axis="x", alpha=0.35)
    add_soft_background(axC, "#F8FAFC", "#ECFDF5", 0.25)

    # D: Top category-specific ORs (remove inverse duplicates, select high ORs)
    add_panel_label(axD, "D")
    dcat = cat.copy()
    dcat["p_value"] = pd.to_numeric(dcat["p_value"], errors="coerce")
    dcat["OR_category_vs_others"] = pd.to_numeric(dcat["OR_category_vs_others"], errors="coerce")
    dcat = dcat[(dcat["n_category"] >= 20) & (dcat["OR_category_vs_others"] > 1.1)].sort_values("p_value").head(9)
    dcat["label"] = dcat["variable"].astype(str).str.replace("_", " ") + ": " + dcat["category"].astype(str)
    dd = pd.DataFrame({"label": [fill(x, 26) for x in dcat["label"]], "OR": dcat["OR_category_vs_others"], "lo": dcat["CI_low"], "hi": dcat["CI_high"], "p": dcat["p_value"]})
    forest(axD, dd, "Top categorical risk signals", xlim=(0.35, max(8, float(dd["hi"].max())*1.1)))
    add_soft_background(axD, "#F8FAFC", "#FCE7F3", 0.20)

    # E: core model with easy labels
    add_panel_label(axE, "E")
    co = core[core["term"].ne("Intercept")].copy()
    co = co.rename(columns={"CI_low":"lo", "CI_high":"hi", "p_value":"p"})
    co["label"] = co["term"].map(clean_term)
    forest(axE, co, "What remained important after adjustment?", xlim=(0.75, 4.2))
    add_soft_background(axE, "#F8FAFC", "#EEF2FF", 0.22)

    title_block(fig, "Figure 1. Complete-case discovery atlas of accident-history associations",
                "All analyses use only the 874 drivers with recorded accident-history data")
    footer(fig, "Plain message: several domains showed signals, but the simplest adjusted model highlighted license renewal, betel quid, RBG, and driving hours/day.")
    fig.subplots_adjust(left=0.065, right=0.985, top=0.905, bottom=0.08)
    savefig(fig, "Figure_1_Discovery_Atlas")


def figure2_license_betel(df: pd.DataFrame, sheets: Dict[str, pd.DataFrame]):
    fig = plt.figure(figsize=(21, 13), facecolor=WHITE)
    gs = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[1.05, 1.05, 1.25], height_ratios=[1, 1], wspace=0.35, hspace=0.42)
    axA = fig.add_subplot(gs[0,0]); axB = fig.add_subplot(gs[0,1]); axC = fig.add_subplot(gs[0,2])
    axD = fig.add_subplot(gs[1,0]); axE = fig.add_subplot(gs[1,1]); axF = fig.add_subplot(gs[1,2])

    # Inputs
    lic_sum = sheets["License_betel_summary"].copy()
    lic_sum = lic_sum.sort_values("license_renew")
    lic_model = sheets["License_accident_model"]
    lic_betel_model = sheets["License_betel_accident_model"]
    interact = sheets["License_betel_interaction_model"]
    to_betel = sheets["License_to_betel_model"]

    # A path diagram
    add_panel_label(axA, "A")
    axA.axis("off")
    box_card(axA, (0.08,0.65), 0.30, 0.18, "Renew license", "more betel quid", "#E0F2FE", BLUE)
    box_card(axA, (0.42,0.65), 0.30, 0.18, "Betel quid", "co-signal", "#FEF3C7", GOLD)
    box_card(axA, (0.76,0.65), 0.20, 0.18, "Accident history", "higher prevalence", "#FCE7F3", CRIMSON, fs=10)
    for x1,x2 in [(0.38,0.42),(0.72,0.76)]:
        axA.add_patch(FancyArrowPatch((x1,0.74), (x2,0.74), arrowstyle="-|>", mutation_scale=16, lw=1.3, color=SLATE))
    ors = {
        "Renew → betel": to_betel.loc[to_betel["term"].eq("license_renew"), "OR"].iloc[0],
        "Renew → accident": lic_model.loc[lic_model["term"].eq("license_renew"), "OR"].iloc[0],
        "Renew + betel": lic_betel_model.loc[lic_betel_model["term"].eq("license_renew"), "OR"].iloc[0],
    }
    axA.text(0.08,0.42, f"Renew → betel: OR {ors['Renew → betel']:.2f}\nRenew → accident: OR {ors['Renew → accident']:.2f}\nAfter adding betel: OR {ors['Renew + betel']:.2f}",
             ha="left", va="top", fontsize=12, color=INK,
             bbox=dict(boxstyle="round,pad=0.35", fc=WHITE, ec=GRID))
    axA.text(0.08,0.13, "Meaning: betel quid is related to both license status and accident history,\nbut it does not fully explain the license signal.",
             ha="left", va="top", fontsize=11.2, color=SLATE)
    axA.set_title("Hypothesis map: license → betel → accident")

    # B dual prevalence by license
    add_panel_label(axB, "B")
    x = np.arange(len(lic_sum)); width = 0.34
    axB.bar(x-width/2, lic_sum["betel_percent"], width, color=GOLD, label="Betel quid", edgecolor=WHITE)
    axB.bar(x+width/2, lic_sum["accident_percent"], width, color=CRIMSON, label="Accident history", edgecolor=WHITE)
    for i, r in lic_sum.iterrows():
        ix = list(lic_sum.index).index(i)
        bar_label(axB, ix-width/2, r["betel_percent"], f"{r['betel_percent']:.1f}%", dy=1.2)
        bar_label(axB, ix+width/2, r["accident_percent"], f"{r['accident_percent']:.1f}%", dy=1.2)
    axB.set_xticks(x); axB.set_xticklabels(lic_sum["license_group"])
    axB.set_ylabel("Prevalence (%)"); axB.set_ylim(0, max(75, lic_sum["betel_percent"].max()+12))
    axB.set_title("Renew-license drivers showed two visible differences")
    axB.legend(frameon=False, loc="upper left")
    axB.grid(axis="y", alpha=0.35); add_soft_background(axB, "#F8FAFC", "#FFFBEB", 0.22)

    # C heatmap license x betel
    add_panel_label(axC, "C")
    tmp = df.groupby(["license_group", "betel_group"], observed=True)["accident"].agg(["sum","count"]).reset_index()
    tmp["prev"] = 100*tmp["sum"]/tmp["count"]
    matrix = tmp.pivot(index="license_group", columns="betel_group", values="prev").reindex(["New license", "Renew license"])[["No betel", "Betel"]]
    im = axC.imshow(matrix.values, cmap=mpl.colors.LinearSegmentedColormap.from_list("risk", ["#F8FAFC", "#F7D774", CRIMSON]), aspect="auto", vmin=0, vmax=max(30, matrix.max().max()))
    axC.set_xticks(range(matrix.shape[1])); axC.set_xticklabels(matrix.columns)
    axC.set_yticks(range(matrix.shape[0])); axC.set_yticklabels(matrix.index)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            row = tmp[(tmp["license_group"]==matrix.index[i]) & (tmp["betel_group"]==matrix.columns[j])]
            if len(row):
                r=row.iloc[0]
                axC.text(j,i, f"{r['prev']:.1f}%\n{int(r['sum'])}/{int(r['count'])}", ha="center", va="center", fontsize=12, fontweight="bold", color=INK)
    axC.set_title("Accident history by license and betel quid")
    cb=fig.colorbar(im, ax=axC, fraction=0.045, pad=0.02); cb.set_label("Accident history (%)")

    # D model comparison forest
    add_panel_label(axD, "D")
    rows=[]
    rows.append({"label":"License only\nrenew vs new", "OR":ors['Renew → accident'], "lo":np.nan, "hi":np.nan, "p":lic_model.loc[lic_model["term"].eq("license_renew"), "p_value"].iloc[0]})
    rows.append({"label":"License + betel\nrenew vs new", "OR":ors['Renew + betel'], "lo":np.nan, "hi":np.nan, "p":lic_betel_model.loc[lic_betel_model["term"].eq("license_renew"), "p_value"].iloc[0]})
    rows.append({"label":"License + betel\nbetel yes vs no", "OR":lic_betel_model.loc[lic_betel_model["term"].eq("Betel_binary"), "OR"].iloc[0], "lo":np.nan, "hi":np.nan, "p":lic_betel_model.loc[lic_betel_model["term"].eq("Betel_binary"), "p_value"].iloc[0]})
    rows.append({"label":"Interaction\nrenew × betel", "OR":interact.loc[interact["term"].str.contains(":", regex=False), "OR"].iloc[0], "lo":np.nan, "hi":np.nan, "p":interact.loc[interact["term"].str.contains(":", regex=False), "p_value"].iloc[0]})
    md=pd.DataFrame(rows)
    y=np.arange(len(md))
    axD.axvline(1, color=SLATE, ls="--", lw=1)
    axD.scatter(md["OR"], y, s=92, c=[CRIMSON, CRIMSON, GOLD, PURPLE], edgecolor=WHITE)
    for yi, r in md.iterrows():
        axD.text(r["OR"]*1.08, yi, f"OR {r['OR']:.2f}, {fmt_p(r['p'])}", va="center", fontsize=10.5, color=SLATE)
    axD.set_xscale("log"); axD.set_xlim(0.65, 5.0)
    axD.set_yticks(y); axD.set_yticklabels(md["label"])
    axD.set_xlabel("Odds ratio (log scale)"); axD.set_title("Does betel explain the license signal?")
    axD.grid(axis="x", alpha=0.35); add_soft_background(axD, "#F8FAFC", "#EEF2FF", 0.24)

    # E predicted risk bars from simple model with license/betel
    add_panel_label(axE, "E")
    model = smf.logit("accident ~ license_renew + Betel_binary", data=df.dropna(subset=["accident","license_renew","Betel_binary"])).fit(disp=False)
    pred_df = pd.DataFrame({
        "license_renew":[0,0,1,1],
        "Betel_binary":[0,1,0,1],
        "label":["New\nNo betel", "New\nBetel", "Renew\nNo betel", "Renew\nBetel"]
    })
    pred_df["risk"] = model.predict(pred_df)*100
    axE.bar(np.arange(len(pred_df)), pred_df["risk"], color=[SKY, GOLD, ROSE, CRIMSON], edgecolor=WHITE)
    for i, r in pred_df.iterrows():
        bar_label(axE, i, r["risk"], f"{r['risk']:.1f}%", dy=0.6)
    axE.set_xticks(np.arange(len(pred_df))); axE.set_xticklabels(pred_df["label"])
    axE.set_ylabel("Model-predicted accident history (%)"); axE.set_ylim(0, max(26, pred_df["risk"].max()+5))
    axE.set_title("Simple predicted-risk translation")
    axE.grid(axis="y", alpha=0.35); add_soft_background(axE, "#F8FAFC", "#FDF2F8", 0.20)

    # F conclusion cards
    add_panel_label(axF, "F")
    axF.axis("off")
    cards=[
        ("1", "Renew-license drivers had higher betel quid intake.", GOLD),
        ("2", "Renew-license drivers had higher accident-history prevalence.", CRIMSON),
        ("3", "Betel quid was also associated with accident history.", TEAL),
        ("4", "No strong evidence that betel fully explains the license signal.", PURPLE),
    ]
    y0=0.78
    for i,(num,txt,col) in enumerate(cards):
        axF.add_patch(FancyBboxPatch((0.05,y0-i*0.18),0.12,0.11,boxstyle="round,pad=0.018",fc=col,ec=col))
        axF.text(0.11,y0+0.055-i*0.18,num,ha="center",va="center",fontsize=13,fontweight="bold",color=WHITE)
        axF.text(0.21,y0+0.055-i*0.18,fill(txt,38),ha="left",va="center",fontsize=12,color=INK)
    axF.text(0.05,0.07,"Reviewer-safe wording:\nBetel quid partly co-occurs with renew-license status,\nbut the license–accident signal remains after betel adjustment.", fontsize=11.5, color=SLATE,
             bbox=dict(boxstyle="round,pad=0.35", fc=WHITE, ec=GRID))
    axF.set_title("Safe conclusion")

    title_block(fig, "Figure 2. License type, betel quid intake, and accident-history pathway",
                "A plain-language pathway test, not a causal proof")
    footer(fig, "Plain message: renew-license drivers had more betel quid intake and more accident history; betel is an important co-signal but not the whole explanation.")
    fig.subplots_adjust(left=0.065, right=0.98, top=0.90, bottom=0.08)
    savefig(fig, "Figure_2_License_Betel_Pathway")


def figure3_smoking_betel_rbg(df: pd.DataFrame, sheets: Dict[str, pd.DataFrame]):
    fig = plt.figure(figsize=(21, 13), facecolor=WHITE)
    gs = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[1,1,1], height_ratios=[1,1], wspace=0.36, hspace=0.42)
    axA=fig.add_subplot(gs[0,0]); axB=fig.add_subplot(gs[0,1]); axC=fig.add_subplot(gs[0,2])
    axD=fig.add_subplot(gs[1,0]); axE=fig.add_subplot(gs[1,1]); axF=fig.add_subplot(gs[1,2])

    order=["Neither","Smoking only","Betel only","Both"]
    colors=[BLUE, GREY, GOLD, CRIMSON]
    tmp=df.dropna(subset=["smoke_betel_group","RBG_num","accident"]).copy()
    tmp["smoke_betel_group"] = pd.Categorical(tmp["smoke_betel_group"], categories=order, ordered=True)
    summ=tmp.groupby("smoke_betel_group", observed=True).agg(n=("accident","count"), events=("accident","sum"), prev=("accident",lambda x:100*x.mean()), rbg_med=("RBG_num","median"), rbg_mean=("RBG_num","mean")).reset_index()
    lows=[]; highs=[]
    for _,r in summ.iterrows():
        lo,hi=ci_wilson(int(r.events), int(r.n)); lows.append(lo*100); highs.append(hi*100)
    summ["lo"]=lows; summ["hi"]=highs

    # A bubble prevalence and size
    add_panel_label(axA,"A")
    x=np.arange(len(summ))
    axA.scatter(x, summ["prev"], s=summ["n"]*2.2, c=colors, edgecolor=WHITE, lw=1.2, alpha=0.95)
    for i,r in summ.iterrows():
        axA.text(i, r["prev"]+1.3, f"{r['prev']:.1f}%\n{int(r['events'])}/{int(r['n'])}", ha="center", fontsize=10.5, fontweight="bold")
    axA.set_xticks(x); axA.set_xticklabels(["Neither", "Smoking\nonly", "Betel\nonly", "Both"])
    axA.set_ylabel("Accident-history prevalence (%)")
    axA.set_ylim(0, max(30, summ["prev"].max()+7))
    axA.set_title("Four behavioral exposure groups")
    axA.grid(axis="y", alpha=0.35); add_soft_background(axA,"#F8FAFC","#FFF7ED",0.22)

    # B RBG violin/box with summary
    add_panel_label(axB,"B")
    data=[tmp.loc[tmp["smoke_betel_group"].eq(g),"RBG_num"].dropna() for g in order]
    parts=axB.violinplot(data, positions=x, widths=0.72, showmeans=False, showextrema=False, showmedians=False)
    for pc,c in zip(parts['bodies'], colors):
        pc.set_facecolor(c); pc.set_edgecolor("none"); pc.set_alpha(0.30)
    for i, vals in enumerate(data):
        q1,q2,q3=np.percentile(vals,[25,50,75])
        axB.plot([i-0.20,i+0.20],[q2,q2],color=INK,lw=2.2)
        axB.add_patch(Rectangle((i-0.15,q1),0.30,q3-q1,fc=WHITE,ec=INK,lw=1.1,alpha=0.8))
        # jitter points downsample
        rng=np.random.default_rng(RANDOM_STATE+i)
        sample=vals.sample(min(len(vals),80), random_state=RANDOM_STATE+i)
        axB.scatter(i + rng.normal(0,0.055,len(sample)), sample, s=10, color=c, alpha=0.25, edgecolor="none")
    p=stats.kruskal(*data).pvalue
    axB.text(0.98,0.94,f"RBG group difference\n{fmt_p(p)}", transform=axB.transAxes, ha="right", va="top", fontsize=10.5,
             bbox=dict(boxstyle="round,pad=0.25", fc=WHITE, ec=GRID))
    axB.set_xticks(x); axB.set_xticklabels(["Neither", "Smoking\nonly", "Betel\nonly", "Both"])
    axB.set_ylabel("RBG (mmol/L)"); axB.set_title("Does combined exposure show higher RBG?")
    axB.grid(axis="y", alpha=0.35); add_soft_background(axB,"#F8FAFC","#ECFDF5",0.20)

    # C prevalence bars with CIs
    add_panel_label(axC,"C")
    axC.bar(x, summ["prev"], color=colors, edgecolor=WHITE, alpha=0.95)
    axC.errorbar(x, summ["prev"], yerr=[summ["prev"]-summ["lo"], summ["hi"]-summ["prev"]], fmt="none", ecolor=SLATE, capsize=4, lw=1.3)
    for i,r in summ.iterrows(): bar_label(axC, i, r["prev"], f"{r['prev']:.1f}%", dy=1.1)
    axC.set_xticks(x); axC.set_xticklabels(["Neither", "Smoking\nonly", "Betel\nonly", "Both"])
    axC.set_ylabel("Accident-history prevalence (%)"); axC.set_ylim(0, max(32, summ["hi"].max()+4))
    axC.set_title("Accident history rises across exposure groups")
    axC.grid(axis="y", alpha=0.35); add_soft_background(axC,"#F8FAFC","#FCE7F3",0.20)

    # D OR comparison with/without RBG
    add_panel_label(axD,"D")
    mod1, m1 = logistic_or_table("accident ~ C(smoke_betel_group)", tmp)
    mod2, m2 = logistic_or_table("accident ~ C(smoke_betel_group) + RBG_num", tmp)
    # Focus comparisons vs Betel only? Statsmodels reference defaults alphabetical? Neither maybe? use create custom ref Neither
    tmp2=tmp.copy(); tmp2["smoke_betel_group"] = tmp2["smoke_betel_group"].cat.reorder_categories(order)
    m1=smf.logit("accident ~ C(smoke_betel_group, Treatment(reference='Neither'))", data=tmp2).fit(disp=False)
    m2=smf.logit("accident ~ C(smoke_betel_group, Treatment(reference='Neither')) + RBG_num", data=tmp2).fit(disp=False)
    rows=[]
    for term in m1.params.index:
        if term=="Intercept": continue
        rows.append({"term":term,"model":"Group only","OR":math.exp(m1.params[term]),"p":m1.pvalues[term]})
    for term in m2.params.index:
        if term=="Intercept": continue
        rows.append({"term":term,"model":"Group + RBG","OR":math.exp(m2.params[term]),"p":m2.pvalues[term]})
    comp=pd.DataFrame(rows)
    comp=comp[comp["term"].str.contains("smoke_betel_group")].copy()
    comp["label"]=comp["term"].str.extract(r"T\.(.*?)\]")[0].fillna(comp["term"])
    labels=["Smoking only", "Betel only", "Both"]
    ypos=np.arange(len(labels))
    axD.axvline(1,color=SLATE,ls="--",lw=1)
    for j,model_name in enumerate(["Group only","Group + RBG"]):
        dd=comp[comp["model"].eq(model_name)]
        xs=[]; ps=[]
        for lab in labels:
            row=dd[dd["label"].eq(lab)]
            xs.append(row["OR"].iloc[0] if len(row) else np.nan)
            ps.append(row["p"].iloc[0] if len(row) else np.nan)
        axD.scatter(xs, ypos+(j-0.5)*0.18, s=70, color=CRIMSON if j==0 else BLUE, label=model_name, edgecolor=WHITE, zorder=3)
        for xi, yi, pp in zip(xs, ypos+(j-0.5)*0.18, ps):
            if not pd.isna(xi): axD.text(xi*1.06, yi, f"{xi:.2f}, {fmt_p(pp)}", va="center", fontsize=9.5, color=SLATE)
    # add RBG effect text
    rbg_or=math.exp(m2.params.get("RBG_num", np.nan)); rbg_p=m2.pvalues.get("RBG_num", np.nan)
    axD.text(0.02,0.03,f"RBG in pathway model: OR {rbg_or:.2f} per mmol/L, {fmt_p(rbg_p)}", transform=axD.transAxes, fontsize=10.5,
             bbox=dict(boxstyle="round,pad=0.25", fc=WHITE, ec=GRID))
    axD.set_xscale("log"); axD.set_xlim(0.45,3.5)
    axD.set_yticks(ypos); axD.set_yticklabels([f"{l}\nvs neither" for l in labels])
    axD.set_xlabel("Odds ratio for accident history"); axD.set_title("Does RBG explain the smoking/betel signal?")
    axD.legend(frameon=False, loc="upper right")
    axD.grid(axis="x", alpha=0.35); add_soft_background(axD,"#F8FAFC","#EEF2FF",0.24)

    # E predicted risk vs RBG by group
    add_panel_label(axE,"E")
    # fit smooth model with RBG and group
    m=smf.logit("accident ~ RBG_num + C(smoke_betel_group, Treatment(reference='Neither'))", data=tmp2).fit(disp=False)
    grid=np.linspace(tmp2["RBG_num"].quantile(.03), tmp2["RBG_num"].quantile(.97), 120)
    for lab,c in zip(order,colors):
        nd=pd.DataFrame({"RBG_num":grid,"smoke_betel_group":lab})
        nd["smoke_betel_group"]=pd.Categorical(nd["smoke_betel_group"], categories=order, ordered=True)
        axE.plot(grid, m.predict(nd)*100, color=c, lw=2.2, label=lab)
    axE.set_xlabel("RBG (mmol/L)"); axE.set_ylabel("Predicted accident history (%)")
    axE.set_title("Same RBG scale, different exposure groups")
    axE.legend(frameon=False, ncol=2, loc="upper left", fontsize=9.5)
    axE.grid(alpha=0.35); add_soft_background(axE,"#F8FAFC","#FFFBEB",0.20)

    # F pathway summary diagram
    add_panel_label(axF,"F")
    axF.axis("off")
    box_card(axF,(0.19,0.70),0.62,0.14,"Smoking + betel quid","combined exposure group showed the highest accident prevalence", "#FCE7F3", CRIMSON, fs=11)
    box_card(axF,(0.24,0.43),0.52,0.14,"RBG","independent accident-history signal", "#ECFDF5", TEAL, fs=11)
    box_card(axF,(0.24,0.16),0.52,0.14,"Accident history","association observed; causation cannot be claimed", "#EFF6FF", BLUE, fs=11)
    axF.add_patch(FancyArrowPatch((0.50,0.70),(0.50,0.57),arrowstyle="-|>",mutation_scale=18,color=TEAL,lw=1.5))
    axF.add_patch(FancyArrowPatch((0.50,0.43),(0.50,0.30),arrowstyle="-|>",mutation_scale=18,color=SLATE,lw=1.5))
    axF.text(0.07,0.04,"Safe wording: RBG was strongly associated with accident history,\nbut the smoking/betel → RBG pathway remains exploratory.", fontsize=11.3, color=SLATE,
             bbox=dict(boxstyle="round,pad=0.30",fc=WHITE,ec=GRID))
    axF.set_title("Simple pathway interpretation")

    title_block(fig,"Figure 3. Smoking, betel quid, RBG, and accident-history pathway",
                "Testing Dr. Zafrul’s behavioral–metabolic hypothesis without overclaiming causality")
    footer(fig,"Plain message: combined smoking and betel identified a higher accident-burden subgroup; RBG was independently important, but causation through RBG cannot be claimed.")
    fig.subplots_adjust(left=0.065, right=0.985, top=0.90, bottom=0.08)
    savefig(fig,"Figure_3_Smoking_Betel_RBG_Pathway")


def figure4_core_model(df: pd.DataFrame, sheets: Dict[str, pd.DataFrame]):
    # Core model fits and predictions
    data=df.dropna(subset=["accident","Age_num","Driving_hours_num","RBG_num","Smoking_binary","Betel_binary","License_clean"]).copy()
    data["License_clean"] = data["License_clean"].astype(str)
    formula="accident ~ Age_num + C(License_clean) + Driving_hours_num + RBG_num + Smoking_binary + Betel_binary"
    model=smf.logit(formula, data=data).fit(disp=False,maxiter=200)
    pred=model.predict(data)
    ortab,_=logistic_or_table(formula,data)

    fig=plt.figure(figsize=(21,13),facecolor=WHITE)
    gs=gridspec.GridSpec(2,3,figure=fig,width_ratios=[1.05,1,1],height_ratios=[1,1.12],wspace=0.36,hspace=0.42)
    axA=fig.add_subplot(gs[0,0]); axB=fig.add_subplot(gs[0,1]); axC=fig.add_subplot(gs[0,2])
    axD=fig.add_subplot(gs[1,0:2]); axE=fig.add_subplot(gs[1,2])

    add_panel_label(axA,"A")
    forest(axA, ortab, "Final simple adjusted model", xlim=(0.75,4.2))
    add_soft_background(axA,"#F8FAFC","#F5F3FF",0.22)

    # B RBG gradient
    add_panel_label(axB,"B")
    med={"Age_num":data["Age_num"].median(),"Driving_hours_num":data["Driving_hours_num"].median(),"Smoking_binary":0,"Betel_binary":0,"License_clean":data["License_clean"].mode().iloc[0]}
    grid=np.linspace(data["RBG_num"].quantile(.03), data["RBG_num"].quantile(.97), 140)
    nd=pd.DataFrame({**med,"RBG_num":grid})
    pr=model.predict(nd)*100
    axB.plot(grid,pr,color=CRIMSON,lw=2.6)
    axB.fill_between(grid, pr*0.75, pr*1.25, color=CRIMSON, alpha=0.14)
    axB.set_xlabel("RBG (mmol/L)"); axB.set_ylabel("Predicted accident-history risk (%)")
    axB.set_title("RBG gradient: higher glucose, higher predicted risk")
    axB.grid(alpha=0.35); add_soft_background(axB,"#F8FAFC","#FCE7F3",0.20)

    # C driving workload gradient
    add_panel_label(axC,"C")
    grid2=np.linspace(0, data["Driving_hours_num"].quantile(.98), 140)
    nd2=pd.DataFrame({**med,"Driving_hours_num":grid2,"RBG_num":data["RBG_num"].median()})
    pr2=model.predict(nd2)*100
    axC.plot(grid2,pr2,color=PURPLE,lw=2.6)
    axC.fill_between(grid2, pr2*0.75, pr2*1.25, color=PURPLE, alpha=0.14)
    axC.set_xlabel("Driving hours/day"); axC.set_ylabel("Predicted accident-history risk (%)")
    axC.set_title("Workload gradient: longer driving, higher predicted risk")
    axC.grid(alpha=0.35); add_soft_background(axC,"#F8FAFC","#EEF2FF",0.20)

    # D 2D risk surface
    add_panel_label(axD,"D")
    rbg_grid=np.linspace(data["RBG_num"].quantile(.03), data["RBG_num"].quantile(.97), 70)
    dh_grid=np.linspace(0, data["Driving_hours_num"].quantile(.97), 60)
    RR,DD=np.meshgrid(rbg_grid,dh_grid)
    nd3=pd.DataFrame({"Age_num":data["Age_num"].median(),"License_clean":data["License_clean"].mode().iloc[0],"Driving_hours_num":DD.ravel(),"RBG_num":RR.ravel(),"Smoking_binary":0,"Betel_binary":0})
    ZZ=model.predict(nd3).values.reshape(RR.shape)*100
    im=axD.contourf(RR,DD,ZZ,levels=14,cmap=mpl.colors.LinearSegmentedColormap.from_list("surface", ["#111827","#4C1D95","#B23A48","#FDE68A"]))
    cs=axD.contour(RR,DD,ZZ,levels=7,colors=WHITE,linewidths=.75,alpha=.80)
    axD.clabel(cs,fmt="%.1f%%",fontsize=9.5)
    inside=data["RBG_num"].between(rbg_grid.min(),rbg_grid.max()) & data["Driving_hours_num"].between(dh_grid.min(),dh_grid.max())
    axD.scatter(data.loc[inside,"RBG_num"],data.loc[inside,"Driving_hours_num"],c=data.loc[inside,"accident"],s=13,cmap=mpl.colors.ListedColormap(["#FFFFFF66","#00FFFFCC"]),edgecolor="none",alpha=.72)
    axD.set_xlabel("RBG (mmol/L)"); axD.set_ylabel("Driving hours/day")
    axD.set_title("Combined risk surface: RBG × daily driving workload")
    cb=fig.colorbar(im,ax=axD,fraction=.030,pad=.02); cb.set_label("Predicted accident-history risk (%)")

    # E model performance + capture curve as compact
    add_panel_label(axE,"E")
    y=data["accident"].values
    fpr,tpr,_=roc_curve(y,pred); roc_auc=auc(fpr,tpr)
    ap=average_precision_score(y,pred); brier=brier_score_loss(y,pred)
    ordered=data.assign(pred=pred).sort_values("pred",ascending=False).reset_index(drop=True)
    ordered["cum_pop"]=(np.arange(len(ordered))+1)/len(ordered)*100
    ordered["cum_events"]=ordered["accident"].cumsum()/ordered["accident"].sum()*100
    axE.plot(ordered["cum_pop"],ordered["cum_events"],color=CRIMSON,lw=2.3,label="Risk-ranked capture")
    axE.plot([0,100],[0,100],color=GREY,ls="--",lw=1)
    for cutoff in [20,30,50]:
        ev=float(ordered.loc[ordered["cum_pop"]<=cutoff,"accident"].sum()/ordered["accident"].sum()*100)
        axE.scatter(cutoff,ev,color=NAVY,s=45,zorder=3)
        axE.text(cutoff+2,ev-5 if ev>50 else ev+3,f"Top {cutoff}%\n{ev:.0f}% cases",fontsize=9.8,color=INK)
    axE.text(.04,.96,f"AUC {roc_auc:.3f}\nAverage precision {ap:.3f}\nBrier {brier:.3f}",transform=axE.transAxes,ha="left",va="top",fontsize=11,
             bbox=dict(boxstyle="round,pad=0.30",fc=WHITE,ec=GRID))
    axE.set_xlabel("Population ranked high-to-low risk (%)"); axE.set_ylabel("Accident-history cases captured (%)")
    axE.set_title("Risk ranking: small group captures many cases")
    axE.grid(alpha=0.35); add_soft_background(axE,"#F8FAFC","#ECFDF5",0.18)

    title_block(fig,"Figure 4. Final adjusted model and practical accident-risk surface",
                "Simple model + absolute risk curves + risk-ranked public-health translation")
    footer(fig,"Plain message: RBG and daily driving hours show interpretable risk gradients; the highest-risk portion of drivers captures a disproportionate share of accident-history cases.")
    fig.subplots_adjust(left=0.065,right=0.985,top=0.90,bottom=0.08)
    savefig(fig,"Figure_4_Core_Model_Risk_Surface")


def figure5_domain_atlas(df: pd.DataFrame, sheets: Dict[str,pd.DataFrame]):
    screening=sheets["All_variable_screening"].copy()
    cat=sheets["Category_specific_ORs"].copy()
    screening["p_value"]=pd.to_numeric(screening["p_value"], errors="coerce")
    screening["strength"]=-np.log10(screening["p_value"].clip(lower=1e-300))

    fig=plt.figure(figsize=(21,13),facecolor=WHITE)
    gs=gridspec.GridSpec(2,3,figure=fig,width_ratios=[1.05,1.05,1.05],height_ratios=[1,1.05],wspace=.39,hspace=.42)
    axA=fig.add_subplot(gs[0,0]); axB=fig.add_subplot(gs[0,1]); axC=fig.add_subplot(gs[0,2])
    axD=fig.add_subplot(gs[1,0]); axE=fig.add_subplot(gs[1,1]); axF=fig.add_subplot(gs[1,2])

    # A strongest by domain
    add_panel_label(axA,"A")
    best=screening.dropna(subset=["p_value"]).sort_values("p_value").groupby("group", as_index=False).first()
    best=best.sort_values("strength")
    y=np.arange(len(best))
    axA.barh(y,best["strength"],color=best["group"].map(DOMAIN_COLORS),alpha=.88)
    axA.set_yticks(y); axA.set_yticklabels([fill(f"{r['group'].split()[0]} — {str(r['variable']).replace('_',' ')}",26) for _,r in best.iterrows()])
    axA.set_xlabel("Association strength, −log10(p)"); axA.set_title("Strongest variable in each domain")
    axA.grid(axis="x",alpha=.35); add_soft_background(axA,"#F8FAFC","#EEF2FF",.20)

    # B metabolic/urinary forest
    add_panel_label(axB,"B")
    d=cat.copy(); d["p_value"]=pd.to_numeric(d["p_value"],errors="coerce"); d["OR_category_vs_others"]=pd.to_numeric(d["OR_category_vs_others"],errors="coerce")
    mask=d["variable"].astype(str).str.contains("RBG|Glucose|Urinary|Diabetic|Glycemic|Protein",case=False,regex=True)
    mb=d[mask & (d["n_category"]>=15)].sort_values("p_value").head(8).copy()
    mb["label"]=(mb["variable"].astype(str).str.replace("_"," ")+": "+mb["category"].astype(str)).map(lambda x: fill(x,24))
    forest(axB,pd.DataFrame({"label":mb["label"],"OR":mb["OR_category_vs_others"],"lo":mb["CI_low"],"hi":mb["CI_high"],"p":mb["p_value"]}),"Metabolic/urinary accident-history signals",xlim=(0.25, max(8,mb["CI_high"].max()*1.1)))
    add_soft_background(axB,"#F8FAFC","#ECFDF5",.20)

    # C lifestyle forest
    add_panel_label(axC,"C")
    mask=d["variable"].astype(str).str.contains("Betel|Smoking|Sleep|Screen|Food|B_Quid|SFood|JFood",case=False,regex=True)
    lf=d[mask & (d["n_category"]>=20)].sort_values("p_value").head(8).copy()
    lf["label"]=(lf["variable"].astype(str).str.replace("_"," ")+": "+lf["category"].astype(str)).map(lambda x: fill(x,24))
    forest(axC,pd.DataFrame({"label":lf["label"],"OR":lf["OR_category_vs_others"],"lo":lf["CI_low"],"hi":lf["CI_high"],"p":lf["p_value"]}),"Lifestyle/behavioral signals",xlim=(0.25, max(8,lf["CI_high"].max()*1.1)))
    add_soft_background(axC,"#F8FAFC","#FFF7ED",.20)

    # D continuous signature with med diff
    add_panel_label(axD,"D")
    cont=screening[screening["type"].eq("numeric")].sort_values("p_value").head(12).copy()
    def med(s):
        try: return float(str(s).split("[")[0].strip())
        except: return np.nan
    cont["diff"]=cont["accident_median_IQR"].map(med)-cont["no_accident_median_IQR"].map(med)
    cont=cont.sort_values("diff")
    y=np.arange(len(cont))
    axD.axvline(0,color=SLATE,lw=1)
    axD.hlines(y,0,cont["diff"],color=GREY,lw=1.25)
    axD.scatter(cont["diff"],y,s=65,c=cont["group"].map(DOMAIN_COLORS),edgecolor=WHITE)
    for yi,(_,r) in enumerate(cont.iterrows()):
        axD.text(r["diff"]+0.15 if r["diff"]>=0 else r["diff"]-0.15, yi, fmt_p(r["p_value"]), va="center", fontsize=9.1, color=SLATE,
                 ha="left" if r["diff"]>=0 else "right")
    axD.set_yticks(y); axD.set_yticklabels([fill(v.replace("_"," "),22) for v in cont["variable"]])
    axD.set_xlabel("Median difference: accident − no accident")
    axD.set_title("Continuous-variable accident signature")
    axD.grid(axis="x",alpha=.35); add_soft_background(axD,"#F8FAFC","#F0FDFA",.20)

    # E clinical/demographic/occupational summary as bars of top signals grouped
    add_panel_label(axE,"E")
    top=screening.dropna(subset=["p_value"]).sort_values("p_value").head(16).copy()
    counts=top.groupby("group").size().reindex(DOMAIN_COLORS.keys()).fillna(0)
    x=np.arange(len(counts))
    axE.bar(x,counts.values,color=[DOMAIN_COLORS[k] for k in counts.index],edgecolor=WHITE)
    for i,v in enumerate(counts.values):
        axE.text(i,v+0.12,str(int(v)),ha="center",fontsize=12,fontweight="bold")
    axE.set_xticks(x); axE.set_xticklabels([fill(k.replace(" factors",""),10) for k in counts.index],rotation=0)
    axE.set_ylabel("Number among top 16 signals")
    axE.set_title("Signal mix among strongest findings")
    axE.grid(axis="y",alpha=.35); add_soft_background(axE,"#F8FAFC","#FDF2F8",.20)

    # F simple explanation cards
    add_panel_label(axF,"F")
    axF.axis("off")
    messages=[("Occupational", "Driving years, daily hours, and license group showed strong signals.", CRIMSON),
              ("Behavioral", "Betel quid and related habits clustered with accident history.", GOLD),
              ("Metabolic", "RBG remained important after adjustment.", TEAL),
              ("Demographic", "Age and marital status differed, but age weakened after adjustment.", BLUE),
              ("Urinary/clinical", "Some urine and symptom variables appeared in screening, mostly exploratory.", PURPLE)]
    y0=.80
    for i,(head,body,col) in enumerate(messages):
        box_card(axF,(.07,y0-i*.17),.85,.11,head,body,mpl.colors.to_hex(mpl.colors.to_rgba(col,0.13)),col,fs=10.5)
    axF.set_title("How to explain the atlas simply")

    title_block(fig,"Figure 5. Domain-specific atlas of associated factors",
                "The 874-driver analysis was not limited to pre-selected predictors")
    footer(fig,"Plain message: multiple domains were screened; the most interpretable signals converged on exposure, behavior, and metabolic health.")
    fig.subplots_adjust(left=.065,right=.985,top=.90,bottom=.08)
    savefig(fig,"Figure_5_Domain_Signal_Atlas")


def figure6_dashboard(df: pd.DataFrame, sheets: Dict[str,pd.DataFrame]):
    fig=plt.figure(figsize=(21,13),facecolor=WHITE)
    gs=gridspec.GridSpec(2,3,figure=fig,width_ratios=[1,1,1],height_ratios=[1,1.05],wspace=.36,hspace=.42)
    axA=fig.add_subplot(gs[0,0]); axB=fig.add_subplot(gs[0,1]); axC=fig.add_subplot(gs[0,2])
    axD=fig.add_subplot(gs[1,0:2]); axE=fig.add_subplot(gs[1,2])

    # A population cards
    add_panel_label(axA,"A"); axA.axis("off")
    n=len(df); events=int(df["accident"].sum()); prev=100*df["accident"].mean()
    cards=[("Complete-case drivers",f"{n:,}",BLUE),("Accident history",f"{events:,}",CRIMSON),("Prevalence",f"{prev:.1f}%",GOLD)]
    y0=.70
    for i,(lab,val,col) in enumerate(cards):
        axA.add_patch(FancyBboxPatch((.12,y0-i*.22),.72,.15,boxstyle="round,pad=.02",fc=mpl.colors.to_hex(mpl.colors.to_rgba(col,.13)),ec=col,lw=1.4))
        axA.text(.19,y0+.075-i*.22,lab,ha="left",va="center",fontsize=12,fontweight="bold",color=INK)
        axA.text(.78,y0+.075-i*.22,val,ha="right",va="center",fontsize=20,fontweight="bold",color=col)
    axA.set_title("Study population used for this re-analysis")

    # B top adjusted signals
    add_panel_label(axB,"B"); axB.axis("off")
    core=sheets["Core_adjusted_model"].copy()
    core=core[core["term"].ne("Intercept")].copy()
    core["label"]=core["term"].map(clean_term).str.replace("\n"," ")
    core["rank_abs"]=np.abs(np.log(core["OR"]))
    core=core.sort_values("rank_abs",ascending=False)
    y0=.78
    palette=[CRIMSON,GOLD,TEAL,PURPLE,GREY]
    for i,(_,r) in enumerate(core.head(5).iterrows()):
        axB.add_patch(Circle((.12,y0-i*.15),.035,fc=palette[i],ec=palette[i]))
        axB.text(.12,y0-i*.15,str(i+1),ha="center",va="center",fontsize=11,fontweight="bold",color=WHITE)
        axB.text(.20,y0+.025-i*.15,clean_term(r["term"]).replace("\n"," "),ha="left",va="center",fontsize=12.5,fontweight="bold",color=INK)
        axB.text(.20,y0-.030-i*.15,f"OR {r['OR']:.2f}, {fmt_p(r['p_value'])}",ha="left",va="center",fontsize=10.8,color=MUTED)
    axB.set_title("Most important adjusted signals")

    # C risk grid practical screen from core model
    add_panel_label(axC,"C")
    data=df.dropna(subset=["accident","Age_num","Driving_hours_num","RBG_num","Smoking_binary","Betel_binary","License_clean"]).copy()
    model=smf.logit("accident ~ Age_num + C(License_clean) + Driving_hours_num + RBG_num + Smoking_binary + Betel_binary",data=data).fit(disp=False,maxiter=200)
    data["pred"]=model.predict(data)
    grid=data.groupby(["Drive_cat","RBG_cat"],observed=True).agg(events=("accident","sum"),n=("accident","count"),risk=("pred","mean")).reset_index()
    matrix=grid.pivot(index="Drive_cat",columns="RBG_cat",values="risk")*100
    im=axC.imshow(matrix.values,cmap=mpl.colors.LinearSegmentedColormap.from_list("grid",["#F8FAFC","#FDE68A",CRIMSON]),aspect="auto")
    axC.set_xticks(range(matrix.shape[1])); axC.set_xticklabels(matrix.columns)
    axC.set_yticks(range(matrix.shape[0])); axC.set_yticklabels(matrix.index)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            row=grid[(grid["Drive_cat"].eq(matrix.index[i])) & (grid["RBG_cat"].eq(matrix.columns[j]))]
            if len(row):
                r=row.iloc[0]
                axC.text(j,i,f"{matrix.iloc[i,j]:.1f}%\n{int(r.events)}/{int(r.n)}",ha="center",va="center",fontsize=11,fontweight="bold",color=INK)
    cb=fig.colorbar(im,ax=axC,fraction=.045,pad=.02); cb.set_label("Predicted risk (%)")
    axC.set_xlabel("RBG category"); axC.set_ylabel("Driving hours/day")
    axC.set_title("Simple practical screening idea")

    # D one-slide explanation
    add_panel_label(axD,"D"); axD.axis("off")
    steps=[("1","We used only 874 drivers","because only these drivers had accident-history data."),
           ("2","We screened all available factors","not just the factors shown in the first analysis."),
           ("3","Strong signals converged","license renewal, betel quid, RBG, and driving hours."),
           ("4","Pathway hypotheses were tested carefully","smoking/betel/RBG and license/betel/accident pathways."),
           ("5","We avoid causal overclaiming","the results show association, not proof of causation.")]
    y=.79
    cols=[BLUE,TEAL,GOLD,PURPLE,CRIMSON]
    for i,(num,head,body) in enumerate(steps):
        axD.add_patch(FancyBboxPatch((.05,y-i*.16),.09,.09,boxstyle="round,pad=.02",fc=cols[i],ec=cols[i]))
        axD.text(.095,y+.045-i*.16,num,ha="center",va="center",color=WHITE,fontweight="bold",fontsize=12)
        axD.text(.18,y+.055-i*.16,head,ha="left",va="center",fontsize=13,fontweight="bold",color=INK)
        axD.text(.18,y+.015-i*.16,body,ha="left",va="center",fontsize=11.5,color=SLATE)
    axD.set_title("One-slide explanation for reviewers")

    # E title options and best framing
    add_panel_label(axE,"E"); axE.axis("off")
    titles=[("Best scientific title","Factors Associated With Accident History Among Professional Drivers: A Complete-Case Analysis of 874 Drivers"),
            ("Mechanism-focused option","Occupational, Behavioral, and Metabolic Correlates of Accident History Among Professional Drivers"),
            ("Simplest reviewer-friendly option","Accident History and Driver Health Factors in Professional Drivers")]
    y=.72
    for i,(head,txt) in enumerate(titles):
        axE.text(.05,y-i*.24,head,fontsize=12.2,fontweight="bold",color=NAVY,ha="left")
        axE.text(.05,y-.08-i*.24,fill(txt,42),fontsize=11.1,color=INK,ha="left",
                 bbox=dict(boxstyle="round,pad=.30",fc=WHITE,ec=GRID))
    axE.set_title("Best manuscript title options")

    title_block(fig,"Figure 6. Plain-language take-home dashboard",
                "Designed for reviewers, supervisors, and non-statistical audiences")
    footer(fig,"Plain message: a reviewer-friendly summary that explains what was analyzed, what was found, and what cannot be overclaimed.")
    fig.subplots_adjust(left=.065,right=.985,top=.90,bottom=.08)
    savefig(fig,"Figure_6_Reviewer_Friendly_Dashboard")

# =============================================================================
# 5) MAIN
# =============================================================================


def main():
    """Main entry point.

    By default, the script launches each figure in a fresh Python process.
    This avoids rare Matplotlib memory/font slowdowns when saving many large
    multi-panel PDF/PNG files sequentially in one process.
    """
    set_style()

    # Child mode: create only one figure.
    if "--figure" in sys.argv:
        fig_number = int(sys.argv[sys.argv.index("--figure") + 1])
        df, sheets = load_inputs()
        df = prepare_exposure_groups(df)
        jobs = {
            1: figure1_discovery,
            2: figure2_license_betel,
            3: figure3_smoking_betel_rbg,
            4: figure4_core_model,
            5: figure5_domain_atlas,
            6: figure6_dashboard,
        }
        print(f"Creating Figure {fig_number}...", flush=True)
        jobs[fig_number](df, sheets)
        print(f"Finished Figure {fig_number}.", flush=True)
        return

    # Parent mode: check inputs once and export shared dataset.
    df, sheets = load_inputs()
    df = prepare_exposure_groups(df)
    df[["accident","Age_num","RBG_num","Driving_hours_num","License_clean","Smoking_binary","Betel_binary","smoke_betel_group","RBG_cat","Drive_cat"]].to_csv(OUT_DIR/"analysis_dataset_used_for_figures.csv", index=False)

    for i in range(1, 7):
        print(f"Launching Figure {i} in a fresh process...", flush=True)
        subprocess.check_call([sys.executable, str(Path(__file__).resolve()), "--figure", str(i)], cwd=str(PROJECT_DIR))

    with open(OUT_DIR/"READ_ME_figure_set.txt", "w", encoding="utf-8") as f:
        f.write("Lancet-style 874-driver accident-history figure set\n")
        f.write("=================================================\n")
        f.write(f"Complete-case N: {len(df)}\n")
        f.write(f"Accident-history events: {int(df['accident'].sum())}\n")
        f.write(f"Accident-history prevalence: {df['accident'].mean()*100:.1f}%\n")
        f.write("\nThese figures show associations, not causation. Pathway analyses are exploratory.\n")

    print("Done. Figures saved to:", OUT_DIR)
    for p in sorted(OUT_DIR.glob("Figure_*.png")):
        print(" -", p.name)

if __name__ == "__main__":
    main()
