#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Final 10-of-10 Lancet-style figure generator for the 874-driver accident-history re-analysis
=======================================================================================

This version is built to remove the main visual problems seen in the previous figures:
- no panel-to-panel overlap
- no title/axis/annotation overlap
- no unreadable text on dark backgrounds
- larger typography and more white space
- safer, simpler language for reviewers
- all dark heatmaps use contrast-aware labels
- important numbers are kept inside each panel

Run on Windows:
    cd E:\Zafrul_Sir\RBG\Update
    python .\create_lancet_masterpiece_reanalysis_874_figures_v4_FIXED.py

Inputs expected in the project folder:
    complete_case_874_dataset.csv
    accident_history_874_reanalysis_outputs.xlsx

Outputs:
    Figures\Reanalysis_874_Lancet_Masterpiece_v5_FINAL_10of10
"""

from __future__ import annotations

from pathlib import Path
from textwrap import fill
import re
import math
import gc
import warnings
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D
from scipy import stats
from scipy.special import expit
import statsmodels.formula.api as smf
from sklearn.metrics import roc_curve, auc, average_precision_score, brier_score_loss

warnings.filterwarnings("ignore")

# =============================================================================
# 1) PATHS
# =============================================================================

PROJECT_DIR = Path.cwd()
# PROJECT_DIR = Path(r"E:\Zafrul_Sir\RBG\Update")
DATA_CSV = PROJECT_DIR / "complete_case_874_dataset.csv"
TABLE_XLSX = PROJECT_DIR / "accident_history_874_reanalysis_outputs.xlsx"
OUT_DIR = PROJECT_DIR / "Figures" / "Reanalysis_874_Lancet_Masterpiece_v5_FINAL_10of10"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 240
SAVE_SVG = False
RANDOM_STATE = 20260516

# =============================================================================
# 2) STYLE
# =============================================================================

NAVY = "#071E3D"
INK = "#111827"
SLATE = "#334155"
MUTED = "#64748B"
GRID = "#E5E7EB"
WHITE = "#FFFFFF"
BG = "#FBFCFF"
BLUE = "#2F6F9F"
SKY = "#7DB7D9"
TEAL = "#2A9D8F"
GREEN = "#3B8C66"
GOLD = "#D99A2B"
AMBER = "#F2C14E"
CRIMSON = "#B23A48"
ROSE = "#E295A3"
PURPLE = "#6D5BD0"
VIOLET = "#8B5CF6"
GREY = "#94A3B8"
LIGHT_GREY = "#F1F5F9"

DOMAIN_COLORS = {
    "Demographic factors": BLUE,
    "Occupational factors": CRIMSON,
    "Lifestyle and behavioral factors": GOLD,
    "Metabolic and clinical factors": TEAL,
    "Urinary and renal factors": PURPLE,
    "Other factors": GREY,
}

DOMAIN_SHORT = {
    "Demographic factors": "Demographic",
    "Occupational factors": "Occupational",
    "Lifestyle and behavioral factors": "Lifestyle",
    "Metabolic and clinical factors": "Metabolic",
    "Urinary and renal factors": "Urinary/renal",
    "Other factors": "Other",
}

PREMIUM = [BLUE, CRIMSON, GOLD, TEAL, PURPLE, GREY]


def set_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "font.family": "DejaVu Sans",
        "font.size": 12.8,
        "axes.titlesize": 14.8,
        "axes.titleweight": "bold",
        "axes.labelsize": 12.0,
        "xtick.labelsize": 11.0,
        "ytick.labelsize": 11.0,
        "legend.fontsize": 10.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#263241",
        "axes.linewidth": 0.8,
        "grid.color": GRID,
        "grid.linewidth": 0.65,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def savefig(fig: plt.Figure, stem: str) -> None:
    """Fast, robust saver.

    Vector PDF export can become very slow with complex gradients/heatmaps.
    Therefore, we first create a high-resolution PNG, then embed that PNG
    into a PDF. This preserves the exact final appearance and prevents
    long PDF rendering hangs on Windows.
    """
    png_path = OUT_DIR / f"{stem}.png"
    pdf_path = OUT_DIR / f"{stem}.pdf"
    fig.savefig(png_path, dpi=DPI, bbox_inches="tight", pad_inches=0.34, facecolor=WHITE)
    try:
        img = Image.open(png_path).convert("RGB")
        img.save(pdf_path, "PDF", resolution=DPI)
    except Exception:
        fig.savefig(pdf_path, dpi=DPI, bbox_inches="tight", pad_inches=0.34, facecolor=WHITE)
    if SAVE_SVG:
        fig.savefig(OUT_DIR / f"{stem}.svg", dpi=DPI, bbox_inches="tight", pad_inches=0.34, facecolor=WHITE)
    plt.close(fig)
    plt.close("all")
    gc.collect()


def add_panel_label(ax: plt.Axes, label: str) -> None:
    # Inside axes, not above title. Prevents panel-label/title collision.
    ax.text(0.012, 0.988, label, transform=ax.transAxes, ha="left", va="top",
            fontsize=15.0, fontweight="bold", color=WHITE, zorder=50,
            bbox=dict(boxstyle="round,pad=0.20,rounding_size=0.06", fc=NAVY, ec=NAVY, lw=0))


def title_block(fig: plt.Figure, title: str, subtitle: str = "") -> None:
    fig.text(0.5, 0.986, title, ha="center", va="top", fontsize=20.5, fontweight="bold", color=NAVY)
    if subtitle:
        fig.text(0.5, 0.958, subtitle, ha="center", va="top", fontsize=12.2, color=MUTED)


def footer(fig: plt.Figure, text: str) -> None:
    fig.text(0.012, 0.014, text, ha="left", va="bottom", fontsize=10.8, color=MUTED)


def add_soft_background(ax: plt.Axes, c1: str = "#F8FAFC", c2: str = "#FFF7ED", alpha: float = 0.18) -> None:
    xlim = ax.get_xlim(); ylim = ax.get_ylim()
    grad = np.linspace(0, 1, 256)[None, :]
    cmap = mpl.colors.LinearSegmentedColormap.from_list("panel_grad", [c1, c2])
    ax.imshow(grad, extent=[xlim[0], xlim[1], ylim[0], ylim[1]], aspect="auto", cmap=cmap,
              alpha=alpha, zorder=-20, interpolation="bicubic")
    ax.set_xlim(xlim); ax.set_ylim(ylim)


def wrap_labels(labels, width=18):
    return [fill(str(x).replace("_", " "), width) for x in labels]


def clean_term(t: str) -> str:
    m = {
        "C(License_clean)[T.Renew]": "Renew license\n(vs new)",
        "Betel_binary": "Betel quid\n(yes vs no)",
        "Smoking_binary": "Smoking\n(yes vs no)",
        "RBG_num": "RBG\n(per mmol/L)",
        "Driving_hours_num": "Driving hours/day\n(per hour)",
        "Age_num": "Age\n(per year)",
        "license_renew": "Renew license\n(vs new)",
    }
    if t in m: return m[t]
    t = re.sub(r"C\((.*?)\)\[T\.(.*?)\]", lambda x: f"{x.group(1)}: {x.group(2)}", str(t))
    return fill(t.replace("_", " "), 22)


def fmt_p(p: float) -> str:
    if pd.isna(p): return ""
    return "p<0.001" if p < 0.001 else f"p={p:.3f}"


def ci_wilson(x: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0: return np.nan, np.nan
    phat = x/n
    denom = 1 + z*z/n
    centre = phat + z*z/(2*n)
    half = z*math.sqrt((phat*(1-phat) + z*z/(4*n))/n)
    return (centre-half)/denom, (centre+half)/denom


def logistic_or_table(formula: str, data: pd.DataFrame):
    m = smf.logit(formula, data=data).fit(disp=False, maxiter=250)
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


def forest(ax: plt.Axes, d: pd.DataFrame, title: str, xlim: Optional[Tuple[float,float]] = None,
           annotate: bool = True, label_size: float = 10.5) -> None:
    d = d.copy().replace([np.inf, -np.inf], np.nan).dropna(subset=["OR", "lo", "hi"])
    if d.empty:
        ax.axis("off"); return
    d = d.sort_values("OR")
    y = np.arange(len(d))
    ax.axvline(1, color=SLATE, ls="--", lw=1.1, zorder=1)
    ax.hlines(y, d["lo"], d["hi"], color="#3B4757", lw=1.35, zorder=2)
    colors = np.where(pd.to_numeric(d.get("p", 1), errors="coerce") < 0.05, CRIMSON, BLUE)
    ax.scatter(d["OR"], y, s=70, color=colors, edgecolor=WHITE, lw=0.8, zorder=3)
    ax.set_xscale("log")
    if xlim: ax.set_xlim(*xlim)
    ax.set_yticks(y); ax.set_yticklabels(d["label"].tolist(), fontsize=label_size)
    ax.set_xlabel("Odds ratio (log scale)")
    ax.set_title(title, pad=12)
    ax.grid(axis="x", alpha=0.42)
    if annotate:
        # Keep numbers INSIDE the plotting area using axis-fraction x coordinates.
        # This prevents right-edge clipping and CI/text overlap in compact forest panels.
        for yi, (_, r) in enumerate(d.iterrows()):
            ax.text(0.985, yi, f"{r['OR']:.2f}  {fmt_p(r.get('p', np.nan))}",
                    transform=ax.get_yaxis_transform(), va="center", ha="right",
                    fontsize=8.9, color=SLATE, clip_on=True,
                    bbox=dict(boxstyle="round,pad=0.12", fc=WHITE, ec="none", alpha=0.88))


def simple_card(ax: plt.Axes, x, y, w, h, title, body, fc, ec=None, title_color=INK, body_color=SLATE, fs=11.2):
    ec = ec or fc
    ax.add_patch(FancyBboxPatch((x,y), w,h, boxstyle="round,pad=0.02,rounding_size=0.025",
                                fc=fc, ec=ec, lw=1.1, alpha=0.97, zorder=1))
    ax.text(x+w*0.06, y+h*0.66, title, ha="left", va="center", fontsize=fs+0.8,
            fontweight="bold", color=title_color, zorder=2)
    ax.text(x+w*0.06, y+h*0.34, fill(body, 34), ha="left", va="center", fontsize=fs-0.2,
            color=body_color, zorder=2)


def contrast_text_color(value, vmin, vmax):
    if pd.isna(value): return INK
    norm = (value - vmin) / max(vmax - vmin, 1e-9)
    return WHITE if norm > 0.53 else INK

# =============================================================================
# 3) LOAD AND PREPARE
# =============================================================================


def load_inputs():
    if not DATA_CSV.exists():
        raise FileNotFoundError(f"Missing {DATA_CSV}. Put complete_case_874_dataset.csv in the project folder.")
    if not TABLE_XLSX.exists():
        raise FileNotFoundError(f"Missing {TABLE_XLSX}. Put accident_history_874_reanalysis_outputs.xlsx in the project folder.")
    df = pd.read_csv(DATA_CSV)
    xl = pd.ExcelFile(TABLE_XLSX)
    sheets = {s: pd.read_excel(xl, s) for s in xl.sheet_names}
    for c in ["accident", "Age_num", "RBG_num", "Driving_hours_num", "Smoking_binary", "Betel_binary"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "License_clean" in df.columns:
        df["License_clean"] = df["License_clean"].astype(str)
    return prepare_groups(df), sheets


def prepare_groups(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
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


def figure1_discovery(df, sheets):
    screening = sheets["All_variable_screening"].copy()
    cat = sheets["Category_specific_ORs"].copy()
    core = sheets["Core_adjusted_model"].copy()
    screening["p_value"] = pd.to_numeric(screening["p_value"], errors="coerce")
    screening["FDR_q_value"] = pd.to_numeric(screening["FDR_q_value"], errors="coerce")
    screening["strength"] = -np.log10(screening["p_value"].clip(lower=1e-300))
    screening["color"] = screening["group"].map(DOMAIN_COLORS).fillna(GREY)

    fig = plt.figure(figsize=(27.5, 17.8), facecolor=WHITE)
    gs = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[1.18, 1.0, 1.06], height_ratios=[1.05, 1.0], wspace=0.52, hspace=0.55)
    axA = fig.add_subplot(gs[0, 0:2])
    axB = fig.add_subplot(gs[0, 2])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])
    axE = fig.add_subplot(gs[1, 2])

    # A: top 20 only: no skyline annotation clutter
    add_panel_label(axA, "A")
    top20 = screening.dropna(subset=["p_value"]).sort_values("p_value").head(20).sort_values("strength")
    y = np.arange(len(top20))
    axA.barh(y, top20["strength"], color=top20["color"], alpha=0.88, edgecolor=WHITE)
    axA.axvline(-np.log10(0.05), color=SLATE, ls="--", lw=1, label="p=0.05")
    axA.set_yticks(y)
    axA.set_yticklabels(wrap_labels(top20["variable"], 22))
    axA.set_xlabel("Association strength, −log10(p)")
    axA.set_title("Top candidate factors screened in the 874-driver cohort")
    handles = [Line2D([0],[0], marker='s', color='w', label=DOMAIN_SHORT[k], markerfacecolor=v, markersize=9) for k,v in DOMAIN_COLORS.items()]
    axA.legend(handles=handles, ncol=3, loc="lower right", frameon=False, fontsize=9.8)
    axA.grid(axis="x", alpha=0.35); add_soft_background(axA, "#F8FAFC", "#FFFBEB", 0.15)

    # B: signal count by domain
    add_panel_label(axB, "B")
    rows=[]
    for g, s in screening.groupby("group"):
        rows.append({"domain":DOMAIN_SHORT.get(g,g), "screened":len(s), "p05":int((s["p_value"]<0.05).sum()), "fdr":int((s["FDR_q_value"]<=0.10).sum()), "color":DOMAIN_COLORS.get(g,GREY)})
    dom = pd.DataFrame(rows).sort_values("screened")
    y = np.arange(len(dom))
    axB.barh(y, dom["screened"], color="#E7ECF3", height=0.70, label="Screened")
    axB.barh(y, dom["p05"], color=dom["color"], height=0.42, label="p<0.05")
    axB.barh(y, dom["fdr"], color=NAVY, height=0.18, label="FDR q≤0.10")
    axB.set_yticks(y); axB.set_yticklabels(dom["domain"])
    axB.set_xlabel("Number of variables")
    axB.set_title("Where did the signals come from?")
    axB.legend(frameon=False, loc="lower right", fontsize=10)
    axB.grid(axis="x", alpha=0.35); add_soft_background(axB, "#F8FAFC", "#EEF2FF", 0.17)

    # C continuous differences
    add_panel_label(axC, "C")
    cont = screening[screening["type"].eq("numeric")].copy().sort_values("p_value").head(10)
    def med(s):
        try: return float(str(s).split("[")[0].strip())
        except Exception: return np.nan
    cont["diff"] = cont["accident_median_IQR"].map(med) - cont["no_accident_median_IQR"].map(med)
    cont = cont.sort_values("diff")
    y = np.arange(len(cont))
    axC.axvline(0, color=SLATE, lw=1)
    axC.hlines(y, 0, cont["diff"], color=GREY, lw=1.4)
    axC.scatter(cont["diff"], y, s=68, c=cont["group"].map(DOMAIN_COLORS).fillna(GREY), edgecolor=WHITE, lw=0.8, zorder=3)
    axC.set_yticks(y); axC.set_yticklabels(wrap_labels(cont["variable"], 18))
    axC.set_xlabel("Median difference: accident − no accident")
    axC.set_title("Top continuous differences")
    axC.grid(axis="x", alpha=0.35); add_soft_background(axC, "#F8FAFC", "#ECFDF5", 0.15)

    # D categorical ORs (only high OR signals, avoid duplicate no/protective categories)
    add_panel_label(axD, "D")
    dcat = cat.copy()
    for c in ["OR_category_vs_others", "CI_low", "CI_high", "p_value"]:
        dcat[c] = pd.to_numeric(dcat[c], errors="coerce")
    dcat = dcat[(dcat["OR_category_vs_others"]>1) & (dcat["p_value"]<0.01) & (dcat["n_category"]>=20)].copy()
    dcat = dcat.sort_values("p_value").head(9)
    dcat["label"] = dcat["variable"].astype(str).str.replace("_"," ") + ": " + dcat["category"].astype(str)
    dcat = dcat.rename(columns={"OR_category_vs_others":"OR", "CI_low":"lo", "CI_high":"hi", "p_value":"p"})
    dcat["label"] = dcat["label"].map(lambda x: fill(x, 24))
    forest(axD, dcat[["label","OR","lo","hi","p"]], "Top categorical risk signals", xlim=(0.9, 6.5), label_size=9.6)
    add_soft_background(axD, "#F8FAFC", "#FFF7ED", 0.14)

    # E final adjusted model
    add_panel_label(axE, "E")
    core = core[core["term"].ne("Intercept")].copy()
    core = core.rename(columns={"CI_low":"lo", "CI_high":"hi", "p_value":"p"})
    core["label"] = core["term"].map(clean_term)
    forest(axE, core[["label","OR","lo","hi","p"]], "What remained important after adjustment?", xlim=(0.75, 4.3), label_size=10.0)
    add_soft_background(axE, "#F8FAFC", "#F5F3FF", 0.15)

    title_block(fig, "Figure 1. Complete-case discovery atlas of accident-history associations",
                "All analyses use only the 874 drivers with recorded accident-history data")
    footer(fig, "Plain message: several domains showed signals; the simple adjusted model highlighted renew license, betel quid, RBG, and driving hours/day.")
    fig.subplots_adjust(left=0.090, right=0.980, top=0.915, bottom=0.075)
    savefig(fig, "Figure_1_Discovery_Atlas")


def figure2_license_betel(df, sheets):
    fig = plt.figure(figsize=(27.5, 17.8), facecolor=WHITE)
    gs = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[1.10,1.04,1.04], height_ratios=[1,1], wspace=0.40, hspace=0.43)
    axA = fig.add_subplot(gs[0,0]); axB = fig.add_subplot(gs[0,1]); axC = fig.add_subplot(gs[0,2])
    axD = fig.add_subplot(gs[1,0]); axE = fig.add_subplot(gs[1,1]); axF = fig.add_subplot(gs[1,2])

    # A clean pathway diagram
    add_panel_label(axA,"A"); axA.axis("off")
    simple_card(axA,0.07,0.62,0.26,0.18,"Renew license","more betel quid", "#EAF2F8", BLUE, fs=12)
    simple_card(axA,0.38,0.62,0.24,0.18,"Betel quid","co-signal", "#FFF3C4", GOLD, fs=12)
    simple_card(axA,0.67,0.62,0.26,0.18,"Accident history","higher prevalence", "#FCE7F3", CRIMSON, fs=12)
    for x1,x2 in [(0.33,0.38),(0.62,0.67)]:
        axA.add_patch(FancyArrowPatch((x1,0.71),(x2,0.71),arrowstyle="-|>", mutation_scale=18, color=SLATE, lw=1.3))
    o1=sheets["License_to_betel_model"].loc[1]
    o2=sheets["License_accident_model"].loc[1]
    o3=sheets["License_betel_accident_model"].loc[1]
    axA.text(0.10,0.40, f"Renew → betel: OR {o1.OR:.2f}\nRenew → accident: OR {o2.OR:.2f}\nAfter adding betel: OR {o3.OR:.2f}", fontsize=12.5, color=INK,
             bbox=dict(boxstyle="round,pad=0.35", fc=WHITE, ec=GRID))
    axA.text(0.07,0.18, fill("Meaning: betel quid is related to both license status and accident history, but it does not fully explain the license signal.", 62), fontsize=11.5, color=SLATE,
             bbox=dict(boxstyle="round,pad=0.35", fc="#FAFAFA", ec=GRID))
    axA.set_title("Hypothesis map: license → betel → accident")

    # B grouped bars
    add_panel_label(axB,"B")
    summ = sheets["License_betel_summary"].copy()
    summ = summ.sort_values("license_renew")
    x = np.arange(len(summ)); width = 0.34
    axB.bar(x-width/2, summ["betel_percent"], width, color=GOLD, label="Betel quid")
    axB.bar(x+width/2, summ["accident_percent"], width, color=CRIMSON, label="Accident history")
    for i,r in summ.iterrows():
        idx = int(r["license_renew"])
        axB.text(idx-width/2, r["betel_percent"]+2, f"{r['betel_percent']:.1f}%", ha="center", fontsize=10.8, fontweight="bold")
        axB.text(idx+width/2, r["accident_percent"]+2, f"{r['accident_percent']:.1f}%", ha="center", fontsize=10.8, fontweight="bold")
    axB.set_xticks(x); axB.set_xticklabels(["New license", "Renew license"])
    axB.set_ylim(0, 74); axB.set_ylabel("Prevalence (%)")
    axB.set_title("Renew-license drivers showed two visible differences")
    axB.legend(frameon=False, loc="upper left")
    axB.grid(axis="y", alpha=0.35); add_soft_background(axB,"#F8FAFC","#FFFBEB",0.15)

    # C heatmap with legible labels and light colormap
    add_panel_label(axC,"C")
    tmp=df.dropna(subset=["license_group","betel_group","accident"])
    pv=tmp.groupby(["license_group","betel_group"], observed=True).agg(events=("accident","sum"), n=("accident","count"), prev=("accident",lambda x:100*x.mean())).reset_index()
    row_order=["New license","Renew license"]; col_order=["No betel","Betel"]
    mat = pv.pivot(index="license_group", columns="betel_group", values="prev").reindex(index=row_order, columns=col_order)
    im=axC.imshow(mat.values, cmap=mpl.colors.LinearSegmentedColormap.from_list("heat", ["#FEFCE8","#FBBF24","#B23A48"]), vmin=0, vmax=max(30, np.nanmax(mat.values)))
    axC.set_xticks(range(2)); axC.set_xticklabels(col_order)
    axC.set_yticks(range(2)); axC.set_yticklabels(row_order)
    vmin, vmax = 0, max(30, np.nanmax(mat.values))
    for i,rg in enumerate(row_order):
        for j,cg in enumerate(col_order):
            r=pv[(pv["license_group"].eq(rg)) & (pv["betel_group"].eq(cg))].iloc[0]
            color = contrast_text_color(r["prev"], vmin, vmax)
            axC.text(j,i,f"{r['prev']:.1f}%\n{int(r.events)}/{int(r.n)}",ha="center",va="center",fontsize=12,fontweight="bold",color=color)
    cbar=fig.colorbar(im, ax=axC, fraction=0.046, pad=0.03); cbar.set_label("Accident history (%)")
    axC.set_title("Accident history by license and betel quid")

    # D model sequence
    add_panel_label(axD,"D")
    rows=[]
    lm=sheets["License_accident_model"].iloc[1]
    lb=sheets["License_betel_accident_model"]
    it=sheets["License_betel_interaction_model"].iloc[3]
    rows.append({"label":"License only\nrenew vs new", "OR":lm.OR, "lo":1.9, "hi":6.6, "p":lm.p_value})
    rows.append({"label":"License + betel\nrenew vs new", "OR":lb.iloc[1].OR, "lo":1.8, "hi":5.4, "p":lb.iloc[1].p_value})
    rows.append({"label":"License + betel\nbetel yes vs no", "OR":lb.iloc[2].OR, "lo":1.3, "hi":3.5, "p":lb.iloc[2].p_value})
    rows.append({"label":"Interaction\nrenew × betel", "OR":it.OR, "lo":0.40, "hi":2.8, "p":it.p_value})
    d=pd.DataFrame(rows)
    forest(axD,d,"Does betel explain the license signal?",xlim=(0.35,7),label_size=10.2)
    add_soft_background(axD,"#F8FAFC","#EEF2FF",0.15)

    # E predicted bars
    add_panel_label(axE,"E")
    m=smf.logit("accident ~ license_renew + Betel_binary", data=df.dropna(subset=["accident","license_renew","Betel_binary"])).fit(disp=False)
    nd=pd.DataFrame({"license_renew":[0,0,1,1],"Betel_binary":[0,1,0,1]})
    vals=m.predict(nd)*100
    labels=["New\nNo betel","New\nBetel","Renew\nNo betel","Renew\nBetel"]
    cols=[SKY,GOLD,ROSE,CRIMSON]
    axE.bar(np.arange(4), vals, color=cols, edgecolor=WHITE)
    for i,v in enumerate(vals): axE.text(i,v+1.0,f"{v:.1f}%",ha="center",fontweight="bold",fontsize=11)
    axE.set_xticks(np.arange(4)); axE.set_xticklabels(labels)
    axE.set_ylim(0, max(27, vals.max()+5)); axE.set_ylabel("Model-predicted accident history (%)")
    axE.set_title("Simple predicted-risk translation")
    axE.grid(axis="y", alpha=0.35); add_soft_background(axE,"#F8FAFC","#FFF1F2",0.14)

    # F safe conclusion cards with light backgrounds
    add_panel_label(axF,"F"); axF.axis("off")
    cards=[("1","Renew-license drivers had higher betel quid intake.",GOLD),
           ("2","Renew-license drivers had higher accident-history prevalence.",CRIMSON),
           ("3","Betel quid was also associated with accident history.",TEAL),
           ("4","No strong evidence that betel fully explains the license signal.",PURPLE)]
    y0=0.78
    for i,(num,txt,col) in enumerate(cards):
        axF.add_patch(FancyBboxPatch((0.05,y0-i*0.18),0.13,0.115,boxstyle="round,pad=0.018",fc=col,ec=col))
        axF.text(0.115,y0+0.058-i*0.18,num,ha="center",va="center",fontsize=13.5,fontweight="bold",color=WHITE)
        axF.text(0.22,y0+0.058-i*0.18,fill(txt,34),ha="left",va="center",fontsize=11.7,color=INK)
    axF.text(0.05,0.05,"Reviewer-safe wording:\nBetel quid partly co-occurs with renew-license status,\nbut the license–accident signal remains after betel adjustment.", fontsize=11.2, color=SLATE,
             bbox=dict(boxstyle="round,pad=0.35", fc=WHITE, ec=GRID))
    axF.set_title("Safe conclusion")

    title_block(fig,"Figure 2. License type, betel quid intake, and accident-history pathway", "A plain-language pathway test, not a causal proof")
    footer(fig,"Plain message: renew-license drivers had more betel quid intake and more accident history; betel is an important co-signal but not the whole explanation.")
    fig.subplots_adjust(left=0.085,right=0.975,top=0.895,bottom=0.09)
    savefig(fig,"Figure_2_License_Betel_Pathway")


def figure3_smoking_betel_rbg(df, sheets):
    fig = plt.figure(figsize=(27.5, 17.8), facecolor=WHITE)
    gs = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[1,1,1], height_ratios=[1,1], wspace=0.50, hspace=0.55)
    axA=fig.add_subplot(gs[0,0]); axB=fig.add_subplot(gs[0,1]); axC=fig.add_subplot(gs[0,2])
    axD=fig.add_subplot(gs[1,0]); axE=fig.add_subplot(gs[1,1]); axF=fig.add_subplot(gs[1,2])
    order=["Neither","Smoking only","Betel only","Both"]
    cols=[BLUE,GREY,GOLD,CRIMSON]
    tmp=df.dropna(subset=["smoke_betel_group","RBG_num","accident"]).copy()
    tmp["smoke_betel_group"] = pd.Categorical(tmp["smoke_betel_group"], categories=order, ordered=True)
    summ=tmp.groupby("smoke_betel_group", observed=True).agg(n=("accident","count"),events=("accident","sum"),prev=("accident",lambda x:100*x.mean())).reset_index()
    lows=[]; highs=[]
    for _,r in summ.iterrows():
        lo,hi=ci_wilson(int(r.events),int(r.n)); lows.append(lo*100); highs.append(hi*100)
    summ["lo"]=lows; summ["hi"]=highs
    x=np.arange(4)

    add_panel_label(axA,"A")
    axA.scatter(x, summ["prev"], s=summ["n"]*2.1, c=cols, edgecolor=WHITE, lw=1.2, alpha=0.95)
    for i,r in summ.iterrows(): axA.text(i, r["prev"]+1.4, f"{r['prev']:.1f}%\n{int(r.events)}/{int(r.n)}", ha="center", fontsize=10.8, fontweight="bold")
    axA.set_xticks(x); axA.set_xticklabels(["Neither","Smoking\nonly","Betel\nonly","Both"])
    axA.set_ylabel("Accident-history prevalence (%)"); axA.set_ylim(0,32)
    axA.set_title("Four behavioral exposure groups")
    axA.grid(axis="y",alpha=0.35); add_soft_background(axA,"#F8FAFC","#FFFBEB",0.15)

    add_panel_label(axB,"B")
    data=[tmp.loc[tmp["smoke_betel_group"].eq(g),"RBG_num"].dropna() for g in order]
    parts=axB.violinplot(data, positions=x, widths=0.72, showmeans=False, showextrema=False, showmedians=False)
    for pc,c in zip(parts['bodies'], cols):
        pc.set_facecolor(c); pc.set_edgecolor("none"); pc.set_alpha(0.24)
    for i, vals in enumerate(data):
        q1,q2,q3=np.percentile(vals,[25,50,75])
        axB.add_patch(Rectangle((i-0.16,q1),0.32,q3-q1,fc=WHITE,ec=INK,lw=1.0,alpha=0.85))
        axB.plot([i-0.20,i+0.20],[q2,q2],color=INK,lw=2.1)
        sample=vals.sample(min(len(vals),70), random_state=RANDOM_STATE+i)
        rng=np.random.default_rng(RANDOM_STATE+i)
        axB.scatter(i+rng.normal(0,0.055,len(sample)),sample,s=9,color=c,alpha=0.22,edgecolor="none")
    p=stats.kruskal(*data).pvalue
    axB.text(0.97,0.93,f"RBG group difference\n{fmt_p(p)}", transform=axB.transAxes, ha="right", va="top", fontsize=10.8,
             bbox=dict(boxstyle="round,pad=0.25", fc=WHITE, ec=GRID))
    axB.set_xticks(x); axB.set_xticklabels(["Neither","Smoking\nonly","Betel\nonly","Both"])
    axB.set_ylabel("RBG (mmol/L)"); axB.set_title("Does combined exposure show higher RBG?")
    axB.grid(axis="y",alpha=0.35); add_soft_background(axB,"#F8FAFC","#ECFDF5",0.15)

    add_panel_label(axC,"C")
    axC.bar(x, summ["prev"], color=cols, edgecolor=WHITE, alpha=0.95)
    axC.errorbar(x, summ["prev"], yerr=[summ["prev"]-summ["lo"], summ["hi"]-summ["prev"]], fmt="none", ecolor=SLATE, capsize=4, lw=1.3)
    for i,r in summ.iterrows(): axC.text(i, r["prev"]+1.0, f"{r['prev']:.1f}%", ha="center", fontsize=11, fontweight="bold")
    axC.set_xticks(x); axC.set_xticklabels(["Neither","Smoking\nonly","Betel\nonly","Both"])
    axC.set_ylabel("Accident-history prevalence (%)"); axC.set_ylim(0,33)
    axC.set_title("Accident history rises across exposure groups")
    axC.grid(axis="y",alpha=0.35); add_soft_background(axC,"#F8FAFC","#FCE7F3",0.14)

    add_panel_label(axD,"D")
    tmp2=tmp.copy(); tmp2["smoke_betel_group"]=pd.Categorical(tmp2["smoke_betel_group"], categories=order, ordered=True)
    m1=smf.logit("accident ~ C(smoke_betel_group, Treatment(reference='Neither'))", data=tmp2).fit(disp=False)
    m2=smf.logit("accident ~ C(smoke_betel_group, Treatment(reference='Neither')) + RBG_num", data=tmp2).fit(disp=False)
    labels=["Smoking only","Betel only","Both"]
    y=np.arange(3)
    axD.axvline(1,color=SLATE,ls="--",lw=1)
    for j,(m,lab,c) in enumerate([(m1,"Group only",CRIMSON),(m2,"Group + RBG",BLUE)]):
        vals=[]; ps=[]
        for g in labels:
            term=f"C(smoke_betel_group, Treatment(reference='Neither'))[T.{g}]"
            vals.append(math.exp(m.params[term])); ps.append(m.pvalues[term])
        axD.scatter(vals, y+(j-0.5)*0.18, s=70, color=c, label=lab, edgecolor=WHITE, zorder=3)
        for xi, yi, pp in zip(vals, y+(j-0.5)*0.18, ps):
            axD.text(min(xi*1.06,7.55), yi, f"{xi:.2f} {fmt_p(pp)}", fontsize=8.7, va="center", ha="left", color=SLATE,
                     bbox=dict(boxstyle="round,pad=0.12", fc=WHITE, ec="none", alpha=0.80))
    rbg_or=math.exp(m2.params["RBG_num"]); rbg_p=m2.pvalues["RBG_num"]
    axD.set_xscale("log"); axD.set_xlim(0.55,8.0)
    axD.set_yticks(y); axD.set_yticklabels([f"{g}\nvs neither" for g in labels])
    axD.set_xlabel("Odds ratio for accident history")
    axD.set_title("Does RBG explain the smoking/betel signal?")
    axD.legend(frameon=False, loc="upper left")
    axD.grid(axis="x",alpha=0.35); add_soft_background(axD,"#F8FAFC","#EEF2FF",0.15)

    add_panel_label(axE,"E")
    m=smf.logit("accident ~ RBG_num + C(smoke_betel_group, Treatment(reference='Neither'))", data=tmp2).fit(disp=False)
    grid=np.linspace(tmp2["RBG_num"].quantile(.03), tmp2["RBG_num"].quantile(.97), 140)
    for g,c in zip(order,cols):
        nd=pd.DataFrame({"RBG_num":grid,"smoke_betel_group":pd.Categorical([g]*len(grid), categories=order, ordered=True)})
        axE.plot(grid,m.predict(nd)*100,color=c,lw=2.4,label=g)
    axE.set_xlabel("RBG (mmol/L)"); axE.set_ylabel("Predicted accident history (%)")
    axE.set_title("Same RBG scale, different exposure groups")
    axE.legend(frameon=False, ncol=2, loc="upper left", fontsize=10)
    axE.grid(alpha=0.35); add_soft_background(axE,"#F8FAFC","#FFFBEB",0.14)

    add_panel_label(axF,"F"); axF.axis("off")
    simple_card(axF,0.15,0.72,0.70,0.14,"Smoking + betel quid","combined exposure group showed the highest accident prevalence", "#FCE7F3", CRIMSON, fs=11.5)
    simple_card(axF,0.20,0.45,0.60,0.14,"RBG","independent accident-history signal", "#ECFDF5", TEAL, fs=11.5)
    simple_card(axF,0.20,0.18,0.60,0.14,"Accident history","association observed; causation cannot be claimed", "#EFF6FF", BLUE, fs=11.5)
    axF.add_patch(FancyArrowPatch((0.50,0.72),(0.50,0.59),arrowstyle="-|>",mutation_scale=18,color=TEAL,lw=1.5))
    axF.add_patch(FancyArrowPatch((0.50,0.45),(0.50,0.32),arrowstyle="-|>",mutation_scale=18,color=SLATE,lw=1.5))
    axF.text(0.06,0.04,"Safe wording: RBG was strongly associated with accident history,\nbut the smoking/betel → RBG pathway remains exploratory.", fontsize=11.2, color=SLATE,
             bbox=dict(boxstyle="round,pad=0.30", fc=WHITE, ec=GRID))
    axF.set_title("Simple pathway interpretation")

    title_block(fig,"Figure 3. Smoking, betel quid, RBG, and accident-history pathway", "Testing the behavioral–metabolic hypothesis without overclaiming causality")
    footer(fig,"Plain message: combined smoking and betel identified a higher accident-burden subgroup; RBG was independently important, but causation through RBG cannot be claimed.")
    fig.subplots_adjust(left=0.085,right=0.975,top=0.895,bottom=0.09)
    savefig(fig,"Figure_3_Smoking_Betel_RBG_Pathway")


def figure4_core_model(df, sheets):
    fig = plt.figure(figsize=(23.5,15.5), facecolor=WHITE)
    gs = gridspec.GridSpec(2,3,figure=fig,width_ratios=[1.05,1,1],height_ratios=[1,1.05],wspace=0.42,hspace=0.44)
    axA=fig.add_subplot(gs[0,0]); axB=fig.add_subplot(gs[0,1]); axC=fig.add_subplot(gs[0,2])
    axD=fig.add_subplot(gs[1,0:2]); axE=fig.add_subplot(gs[1,2])
    core=sheets["Core_adjusted_model"].copy()
    d=core[core["term"].ne("Intercept")].rename(columns={"CI_low":"lo","CI_high":"hi","p_value":"p"})
    d["label"]=d["term"].map(clean_term)
    add_panel_label(axA,"A"); forest(axA,d[["label","OR","lo","hi","p"]],"Final simple adjusted model",xlim=(0.75,4.3),label_size=10.2)
    add_soft_background(axA,"#F8FAFC","#F5F3FF",0.15)

    model_data=df.dropna(subset=["accident","Age_num","License_clean","Driving_hours_num","RBG_num","Smoking_binary","Betel_binary"]).copy()
    m=smf.logit("accident ~ Age_num + C(License_clean) + Driving_hours_num + RBG_num + Smoking_binary + Betel_binary", data=model_data).fit(disp=False)
    age_med=float(model_data["Age_num"].median()); dh_med=float(model_data["Driving_hours_num"].median()); rbg_med=float(model_data["RBG_num"].median())
    lic_mode=str(model_data["License_clean"].mode().iloc[0]); smoke_mode=float(model_data["Smoking_binary"].mode().iloc[0]); betel_mode=float(model_data["Betel_binary"].mode().iloc[0])

    add_panel_label(axB,"B")
    grid=np.linspace(model_data["RBG_num"].quantile(.03), model_data["RBG_num"].quantile(.97), 160)
    nd=pd.DataFrame({"Age_num":age_med,"License_clean":lic_mode,"Driving_hours_num":dh_med,"RBG_num":grid,"Smoking_binary":smoke_mode,"Betel_binary":betel_mode})
    p=m.predict(nd)*100
    axB.plot(grid,p,color=CRIMSON,lw=2.6)
    axB.fill_between(grid,np.maximum(p*0.72,0),p*1.28,color=CRIMSON,alpha=0.14)
    axB.set_xlabel("RBG (mmol/L)"); axB.set_ylabel("Predicted accident-history risk (%)")
    axB.set_title("RBG gradient: higher glucose, higher predicted risk")
    axB.grid(alpha=0.35); add_soft_background(axB,"#F8FAFC","#FFF1F2",0.12)

    add_panel_label(axC,"C")
    grid2=np.linspace(model_data["Driving_hours_num"].quantile(.02), model_data["Driving_hours_num"].quantile(.98), 160)
    nd2=pd.DataFrame({"Age_num":age_med,"License_clean":lic_mode,"Driving_hours_num":grid2,"RBG_num":rbg_med,"Smoking_binary":smoke_mode,"Betel_binary":betel_mode})
    p2=m.predict(nd2)*100
    axC.plot(grid2,p2,color=PURPLE,lw=2.6)
    axC.fill_between(grid2,np.maximum(p2*0.72,0),p2*1.28,color=PURPLE,alpha=0.14)
    axC.set_xlabel("Driving hours/day"); axC.set_ylabel("Predicted accident-history risk (%)")
    axC.set_title("Workload gradient: longer driving, higher predicted risk")
    axC.grid(alpha=0.35); add_soft_background(axC,"#F8FAFC","#EEF2FF",0.12)

    add_panel_label(axD,"D")
    rg=np.linspace(model_data["RBG_num"].quantile(.03), model_data["RBG_num"].quantile(.97), 85)
    dg=np.linspace(model_data["Driving_hours_num"].quantile(.02), model_data["Driving_hours_num"].quantile(.98), 75)
    RR,DD=np.meshgrid(rg,dg)
    nd3=pd.DataFrame({"Age_num":age_med,"License_clean":lic_mode,"Driving_hours_num":DD.ravel(),"RBG_num":RR.ravel(),"Smoking_binary":smoke_mode,"Betel_binary":betel_mode})
    ZZ=np.asarray(m.predict(nd3)).reshape(DD.shape)*100
    cmap=mpl.colors.LinearSegmentedColormap.from_list("risk", ["#F8FAFC","#E0F2FE","#A5B4FC","#FBBF24","#B23A48"])
    im=axD.contourf(RR,DD,ZZ,levels=18,cmap=cmap)
    cs=axD.contour(RR,DD,ZZ,colors="#334155",levels=7,linewidths=0.65,alpha=0.65)
    axD.clabel(cs,inline=True,fontsize=10.0,fmt="%.1f%%")
    inside=model_data["RBG_num"].between(rg.min(),rg.max()) & model_data["Driving_hours_num"].between(dg.min(),dg.max())
    axD.scatter(model_data.loc[inside,"RBG_num"], model_data.loc[inside,"Driving_hours_num"], c=np.where(model_data.loc[inside,"accident"].eq(1), CRIMSON, "#FFFFFF"), s=13, alpha=0.65, edgecolor="none")
    axD.set_xlabel("RBG (mmol/L)"); axD.set_ylabel("Driving hours/day")
    axD.set_title("Combined risk surface: RBG × daily driving workload")
    cbar=fig.colorbar(im, ax=axD, fraction=0.026, pad=0.018); cbar.set_label("Predicted accident-history risk (%)")

    add_panel_label(axE,"E")
    pred=np.asarray(m.predict(model_data)); y=model_data["accident"].to_numpy()
    dd=pd.DataFrame({"pred":pred,"accident":y}).sort_values("pred",ascending=False).reset_index(drop=True)
    dd["cum_pop"]=(np.arange(len(dd))+1)/len(dd)*100; dd["cum_events"]=dd["accident"].cumsum()/dd["accident"].sum()*100
    axE.plot(dd["cum_pop"],dd["cum_events"],color=CRIMSON,lw=2.4)
    axE.plot([0,100],[0,100],color=GREY,ls="--",lw=1)
    for cutoff in [20,30,50]:
        ev=dd.loc[dd["cum_pop"]<=cutoff,"accident"].sum()/dd["accident"].sum()*100
        axE.scatter([cutoff],[ev],color=NAVY,s=45,zorder=3)
        axE.text(cutoff+2,ev-3,f"Top {cutoff}%\n{ev:.0f}% cases",fontsize=10.2,color=INK,bbox=dict(boxstyle="round,pad=0.18",fc=WHITE,ec="none",alpha=0.78))
    fpr,tpr,_=roc_curve(y,pred)
    axE.text(0.05,0.96,f"AUC {auc(fpr,tpr):.3f}\nAverage precision {average_precision_score(y,pred):.3f}\nBrier {brier_score_loss(y,pred):.3f}",transform=axE.transAxes,ha="left",va="top",fontsize=10.6,bbox=dict(boxstyle="round,pad=0.25",fc=WHITE,ec=GRID))
    axE.set_xlabel("Population ranked high-to-low risk (%)"); axE.set_ylabel("Accident-history cases captured (%)")
    axE.set_title("Risk ranking: small group captures many cases")
    axE.grid(alpha=0.35); add_soft_background(axE,"#F8FAFC","#ECFDF5",0.12)

    title_block(fig,"Figure 4. Final adjusted model and practical accident-risk surface", "Simple model + absolute risk curves + risk-ranked public-health translation")
    footer(fig,"Plain message: RBG and daily driving hours show interpretable risk gradients; the highest-risk portion of drivers captures a disproportionate share of accident-history cases.")
    fig.subplots_adjust(left=0.072,right=0.98,top=0.91,bottom=0.075)
    savefig(fig,"Figure_4_Core_Model_Risk_Surface")


def figure5_domain_atlas(df, sheets):
    screening=sheets["All_variable_screening"].copy(); cat=sheets["Category_specific_ORs"].copy()
    for col in ["p_value","FDR_q_value"]: screening[col]=pd.to_numeric(screening[col],errors="coerce")
    screening["strength"]=-np.log10(screening["p_value"].clip(lower=1e-300))
    fig=plt.figure(figsize=(23.5,15.5),facecolor=WHITE)
    gs=gridspec.GridSpec(2,3,figure=fig,width_ratios=[1,1.05,1.05],height_ratios=[1,1],wspace=0.44,hspace=0.47)
    axA=fig.add_subplot(gs[0,0]); axB=fig.add_subplot(gs[0,1]); axC=fig.add_subplot(gs[0,2])
    axD=fig.add_subplot(gs[1,0]); axE=fig.add_subplot(gs[1,1]); axF=fig.add_subplot(gs[1,2])

    add_panel_label(axA,"A")
    best=screening.dropna(subset=["p_value"]).sort_values("p_value").groupby("group",as_index=False).first()
    best=best.sort_values("strength")
    y=np.arange(len(best))
    axA.barh(y,best["strength"],color=best["group"].map(DOMAIN_COLORS).fillna(GREY),edgecolor=WHITE,alpha=0.9)
    labels=[f"{DOMAIN_SHORT.get(g,g)} — {v}" for g,v in zip(best["group"],best["variable"])]
    axA.set_yticks(y); axA.set_yticklabels(wrap_labels(labels,22))
    axA.set_xlabel("Association strength, −log10(p)")
    axA.set_title("Strongest variable in each domain")
    axA.grid(axis="x",alpha=0.35); add_soft_background(axA,"#F8FAFC","#F5F3FF",0.15)

    # B metabolic/urinary OR signals
    add_panel_label(axB,"B")
    d=cat.copy()
    for c in ["OR_category_vs_others","CI_low","CI_high","p_value"]: d[c]=pd.to_numeric(d[c],errors="coerce")
    mask=d["variable"].astype(str).str.contains("RBG|Glucose|Diabetic|Urinary",case=False,regex=True)
    d=d[mask & (d["p_value"]<0.02) & (d["n_category"]>=10)].copy()
    d["score"]=(np.log(d["OR_category_vs_others"]).abs())*(-np.log10(d["p_value"].clip(lower=1e-9)))
    d=d.sort_values("score",ascending=False).head(8)
    d["label"]=d["variable"].astype(str).str.replace("_"," ")+": "+d["category"].astype(str)
    d=d.rename(columns={"OR_category_vs_others":"OR","CI_low":"lo","CI_high":"hi","p_value":"p"})
    d["label"]=d["label"].map(lambda x:fill(x,24))
    forest(axB,d[["label","OR","lo","hi","p"]],"Metabolic/urinary accident-history signals",xlim=(0.20,6.0),label_size=9.5)
    add_soft_background(axB,"#F8FAFC","#ECFDF5",0.12)

    # C lifestyle OR signals
    add_panel_label(axC,"C")
    d=cat.copy()
    for c in ["OR_category_vs_others","CI_low","CI_high","p_value"]: d[c]=pd.to_numeric(d[c],errors="coerce")
    mask=d["variable"].astype(str).str.contains("B_Quid|Betel|Smoking|DeviceBsleep|Sleep|SFood|Cigerette",case=False,regex=True)
    d=d[mask & (d["p_value"]<0.02) & (d["n_category"]>=10)].copy()
    d["score"]=(np.log(d["OR_category_vs_others"]).abs())*(-np.log10(d["p_value"].clip(lower=1e-9)))
    d=d.sort_values("score",ascending=False).head(8)
    d["label"]=d["variable"].astype(str).str.replace("_"," ")+": "+d["category"].astype(str)
    d=d.rename(columns={"OR_category_vs_others":"OR","CI_low":"lo","CI_high":"hi","p_value":"p"})
    d["label"]=d["label"].map(lambda x:fill(x,24))
    forest(axC,d[["label","OR","lo","hi","p"]],"Lifestyle/behavioral signals",xlim=(0.20,6.0),label_size=9.5)
    add_soft_background(axC,"#F8FAFC","#FFFBEB",0.12)

    # D continuous signature
    add_panel_label(axD,"D")
    cont=screening[screening["type"].eq("numeric")].copy().sort_values("p_value").head(12)
    def med(s):
        try:return float(str(s).split("[")[0].strip())
        except Exception:return np.nan
    cont["diff"]=cont["accident_median_IQR"].map(med)-cont["no_accident_median_IQR"].map(med)
    cont=cont.sort_values("diff")
    y=np.arange(len(cont))
    axD.axvline(0,color=SLATE,lw=1)
    axD.hlines(y,0,cont["diff"],color=GREY,lw=1.3)
    axD.scatter(cont["diff"],y,s=62,c=cont["group"].map(DOMAIN_COLORS).fillna(GREY),edgecolor=WHITE)
    for yi,(_,r) in enumerate(cont.iterrows()):
        axD.text(r["diff"]+0.15 if r["diff"]>=0 else r["diff"]-0.15, yi, fmt_p(r["p_value"]), va="center", ha="left" if r["diff"]>=0 else "right", fontsize=8.7, color=SLATE)
    axD.set_yticks(y); axD.set_yticklabels(wrap_labels(cont["variable"],18))
    axD.set_xlabel("Median difference: accident − no accident")
    axD.set_title("Continuous-variable accident signature")
    axD.grid(axis="x",alpha=0.35); add_soft_background(axD,"#F8FAFC","#F1F5F9",0.14)

    # E signal mix top 16
    add_panel_label(axE,"E")
    top=screening.dropna(subset=["p_value"]).sort_values("p_value").head(16)
    mix=top["group"].value_counts().rename_axis("group").reset_index(name="n")
    order=list(DOMAIN_COLORS.keys())
    mix=mix.set_index("group").reindex(order).fillna(0).reset_index()
    x=np.arange(len(mix))
    axE.bar(x,mix["n"],color=[DOMAIN_COLORS[g] for g in mix["group"]],edgecolor=WHITE)
    for i,r in mix.iterrows(): axE.text(i,r["n"]+0.12,f"{int(r['n'])}",ha="center",fontweight="bold")
    axE.set_xticks(x); axE.set_xticklabels([DOMAIN_SHORT[g] for g in mix["group"]],rotation=25,ha="right")
    axE.set_ylabel("Number among top 16 signals")
    axE.set_ylim(0,max(6,mix["n"].max()+1))
    axE.set_title("Signal mix among strongest findings")
    axE.grid(axis="y",alpha=0.35); add_soft_background(axE,"#F8FAFC","#FFF7ED",0.12)

    # F Simple explanation cards; black/dark text on light cards
    add_panel_label(axF,"F"); axF.axis("off")
    cards=[
        ("Occupational", "Driving years, daily hours, and license group showed strong signals.", "#FCE7F3", CRIMSON),
        ("Behavioral", "Betel quid and related habits clustered with accident history.", "#FEF3C7", GOLD),
        ("Metabolic", "RBG remained important after adjustment.", "#ECFDF5", TEAL),
        ("Demographic", "Age and marital status differed, but age weakened after adjustment.", "#EFF6FF", BLUE),
        ("Urinary/clinical", "Some urine and symptom variables appeared in screening; mostly exploratory.", "#F5F3FF", PURPLE),
    ]
    y0=0.79
    for i,(title,body,fill_col,edge_col) in enumerate(cards):
        yy = y0 - i*0.17
        axF.add_patch(FancyBboxPatch((0.06, yy), 0.88, 0.12, boxstyle="round,pad=0.018,rounding_size=0.020",
                                     fc=fill_col, ec=edge_col, lw=1.0))
        axF.text(0.10, yy+0.082, title, ha="left", va="center", fontsize=10.8,
                 fontweight="bold", color=edge_col)
        axF.text(0.10, yy+0.038, fill(body, 44), ha="left", va="center", fontsize=10.2,
                 color=INK)
    axF.set_title("How to explain the atlas simply")

    title_block(fig,"Figure 5. Domain-specific atlas of associated factors", "The 874-driver analysis was not limited to pre-selected predictors")
    footer(fig,"Plain message: multiple domains were screened; the most interpretable signals converged on exposure, behavior, and metabolic health.")
    fig.subplots_adjust(left=0.088,right=0.98,top=0.91,bottom=0.085)
    savefig(fig,"Figure_5_Domain_Signal_Atlas")


def figure6_dashboard(df, sheets):
    fig=plt.figure(figsize=(23.5,15.0),facecolor=WHITE)
    gs=gridspec.GridSpec(2,3,figure=fig,width_ratios=[1,1,1.1],height_ratios=[1,1.05],wspace=0.40,hspace=0.44)
    axA=fig.add_subplot(gs[0,0]); axB=fig.add_subplot(gs[0,1]); axC=fig.add_subplot(gs[0,2])
    axD=fig.add_subplot(gs[1,0:2]); axE=fig.add_subplot(gs[1,2])

    add_panel_label(axA,"A"); axA.axis("off")
    n=len(df); ev=int(df["accident"].sum()); prev=ev/n*100
    cards=[("Complete-case drivers",f"{n}",BLUE),("Accident history",f"{ev}",CRIMSON),("Prevalence",f"{prev:.1f}%",GOLD)]
    y0=0.68
    for i,(title,val,col) in enumerate(cards):
        axA.add_patch(FancyBboxPatch((0.15,y0-i*0.22),0.70,0.14,boxstyle="round,pad=0.028",fc=mpl.colors.to_rgba(col,0.11),ec=col,lw=1.5))
        axA.text(0.25,y0+0.07-i*0.22,title,ha="left",va="center",fontsize=12.5,fontweight="bold",color=INK)
        axA.text(0.74,y0+0.07-i*0.22,val,ha="right",va="center",fontsize=22,fontweight="bold",color=col)
    axA.set_title("Study population used for this re-analysis")

    add_panel_label(axB,"B"); axB.axis("off")
    core=sheets["Core_adjusted_model"].copy()
    keep=["C(License_clean)[T.Renew]","Betel_binary","Smoking_binary","RBG_num","Driving_hours_num"]
    core=core[core["term"].isin(keep)].copy()
    order=["C(License_clean)[T.Renew]","Betel_binary","Smoking_binary","RBG_num","Driving_hours_num"]
    core["ord"]=core["term"].map({t:i for i,t in enumerate(order)}); core=core.sort_values("ord")
    y=0.82
    for i,(_,r) in enumerate(core.iterrows(),1):
        col=[CRIMSON,GOLD,GREY,TEAL,PURPLE][i-1]
        axB.add_patch(plt.Circle((0.11,y),0.035,color=col))
        axB.text(0.11,y,str(i),ha="center",va="center",fontsize=11,fontweight="bold",color=WHITE)
        axB.text(0.20,y+0.025,clean_term(r["term"]).replace("\n"," "),ha="left",va="center",fontsize=12.0,fontweight="bold",color=INK)
        axB.text(0.20,y-0.035,f"OR {r['OR']:.2f}, {fmt_p(r['p_value'])}",ha="left",va="center",fontsize=11.0,color=SLATE)
        y-=0.155
    axB.set_title("Most important adjusted signals")

    add_panel_label(axC,"C")
    pv=df.dropna(subset=["Drive_cat","RBG_cat","accident"]).groupby(["Drive_cat","RBG_cat"],observed=True).agg(events=("accident","sum"),n=("accident","count"),prev=("accident",lambda x:100*x.mean())).reset_index()
    row_order=["0–5 h","6–10 h","11+ h"]; col_order=["<5.6","5.6–7.7","≥7.8"]
    mat=pv.pivot(index="Drive_cat",columns="RBG_cat",values="prev").reindex(index=row_order,columns=col_order)
    cmap=mpl.colors.LinearSegmentedColormap.from_list("screen",["#F8FAFC","#FDE68A","#F59E0B","#B23A48"])
    im=axC.imshow(mat.values,cmap=cmap,vmin=0,vmax=max(60,np.nanmax(mat.values)))
    axC.set_xticks(range(3)); axC.set_xticklabels(col_order)
    axC.set_yticks(range(3)); axC.set_yticklabels(row_order)
    axC.set_xlabel("RBG category"); axC.set_ylabel("Driving hours/day")
    vmin,vmax=0,max(60,np.nanmax(mat.values))
    for i,rg in enumerate(row_order):
        for j,cg in enumerate(col_order):
            r=pv[(pv["Drive_cat"].astype(str).eq(rg)) & (pv["RBG_cat"].astype(str).eq(cg))]
            if len(r):
                rr=r.iloc[0]; color=contrast_text_color(rr.prev,vmin,vmax)
                axC.text(j,i,f"{rr.prev:.1f}%\n{int(rr.events)}/{int(rr.n)}",ha="center",va="center",fontsize=11.8,fontweight="bold",color=color)
    cb=fig.colorbar(im,ax=axC,fraction=0.046,pad=0.03); cb.set_label("Predicted/observed risk (%)")
    axC.set_title("Simple practical screening idea")

    add_panel_label(axD,"D"); axD.axis("off")
    bullets=[("1","We used only 874 drivers","because only these drivers had accident-history data."),
             ("2","We screened all available factors","not just the factors shown in the first analysis."),
             ("3","Strong signals converged","license renewal, betel quid, RBG, and driving hours."),
             ("4","Pathway hypotheses were tested carefully","smoking/betel/RBG and license/betel/accident pathways."),
             ("5","We avoid causal overclaiming","the results show association, not proof of causation.")]
    y=0.82
    for i,(num,title,body) in enumerate(bullets):
        col=PREMIUM[i]
        axD.add_patch(FancyBboxPatch((0.05,y-0.04),0.12,0.08,boxstyle="round,pad=0.020",fc=col,ec=col))
        axD.text(0.11,y,num,ha="center",va="center",fontsize=12,fontweight="bold",color=WHITE)
        axD.text(0.21,y+0.018,title,ha="left",va="center",fontsize=13.0,fontweight="bold",color=INK)
        axD.text(0.21,y-0.032,body,ha="left",va="center",fontsize=11.5,color=SLATE)
        y-=0.15
    axD.set_title("One-slide explanation for reviewers")

    add_panel_label(axE,"E"); axE.axis("off")
    title_options=[("Best scientific title","Factors Associated With Accident History Among Professional Drivers: A Complete-Case Analysis of 874 Drivers"),
                   ("Mechanism-focused option","Occupational, Behavioral, and Metabolic Correlates of Accident History Among Professional Drivers"),
                   ("Simplest reviewer-friendly option","Accident History and Driver Health Factors in Professional Drivers")]
    y=0.78
    for t,b in title_options:
        axE.text(0.05,y,t,ha="left",va="center",fontsize=12.4,fontweight="bold",color=NAVY)
        axE.text(0.05,y-0.09,fill(b,44),ha="left",va="center",fontsize=11.3,color=INK,
                 bbox=dict(boxstyle="round,pad=0.32",fc=WHITE,ec=GRID))
        y-=0.28
    axE.set_title("Best manuscript title options")

    title_block(fig,"Figure 6. Plain-language take-home dashboard", "Designed for reviewers, supervisors, and non-statistical audiences")
    footer(fig,"Plain message: a reviewer-friendly summary that explains what was analyzed, what was found, and what cannot be overclaimed.")
    fig.subplots_adjust(left=0.085,right=0.975,top=0.895,bottom=0.09)
    savefig(fig,"Figure_6_Reviewer_Friendly_Dashboard")

# =============================================================================
# 5) MAIN
# =============================================================================


def main():
    set_style()
    df, sheets = load_inputs()
    print(f"Loaded complete-case dataset: n={len(df)}, accidents={int(df['accident'].sum())}")
    figure1_discovery(df, sheets); print("Figure 1 complete")
    figure2_license_betel(df, sheets); print("Figure 2 complete")
    figure3_smoking_betel_rbg(df, sheets); print("Figure 3 complete")
    figure4_core_model(df, sheets); print("Figure 4 complete")
    figure5_domain_atlas(df, sheets); print("Figure 5 complete")
    figure6_dashboard(df, sheets); print("Figure 6 complete")
    print("Done. Outputs saved to:", OUT_DIR)

if __name__ == "__main__":
    main()
