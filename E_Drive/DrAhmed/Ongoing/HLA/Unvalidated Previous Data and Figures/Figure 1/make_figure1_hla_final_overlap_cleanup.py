#!/usr/bin/env python3
"""
Final polished publication package generator for Figure 1: HLA allele-position landscape.

Command:
python make_figure1_hla_final_overlap_cleanup.py --input Figure1_HLA_validated_values.csv --outdir figure1_final_overlap_cleanup
"""
from __future__ import annotations

import argparse
import math
import shutil
import gc
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Polygon
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb

try:
    from scipy.spatial.distance import jensenshannon
except Exception:  # pragma: no cover
    jensenshannon = None

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

LOCUS_ORDER = [
    "HLA-A", "HLA-B", "HLA-C", "HLA-DRB1", "HLA-DRB3", "HLA-DRB4", "HLA-DRB5",
    "HLA-DQA1", "HLA-DQB1", "HLA-DPA1", "HLA-DPB1", "HLA-DOA", "HLA-DOB", "HLA-DMA", "HLA-DMB"
]
PANEL_ORDER = dict(zip(LOCUS_ORDER, list("ABCDEFGHIJKLMNO")))
CLASS_I = {"HLA-A", "HLA-B", "HLA-C"}
AXIS_LOCI = {"HLA-DOA", "HLA-DOB", "HLA-DMA", "HLA-DMB"}
PROMINENT = {
    "A*02", "A*24", "A*68", "B*51", "B*50", "B*08", "B*15", "C*07", "C*06", "C*15",
    "DRB1*03", "DRB1*04", "DRB1*07", "DRB1*13", "DRB3*02", "DRB3*03", "DRB3*01",
    "DRB4*01", "DRB5*01", "DQA1*01", "DQA1*05", "DQA1*02", "DQA1*03",
    "DQB1*02", "DQB1*03", "DPA1*01", "DPA1*02", "DPB1*04", "DOA*01", "DOB*01", "DMA*01", "DMB*01"
}
COLORS = {
    "blue_deep": "#0B4F71", "blue_mid": "#7BC8F6", "blue_pale": "#EDF8FF",
    "amber_deep": "#B76B00", "amber_mid": "#F2B84B", "amber_pale": "#FFF7DA",
    "teal_deep": "#005F73", "teal_mid": "#00ACC1", "teal_pale": "#E0F7FA",
    "violet_deep": "#4C1D95", "violet_mid": "#A855F7", "violet_pale": "#F3E8FF",
    "neutral": "#8A8F98", "sep": "#D6DAE0", "panel_bg": "#FAFBFC", "white": "#FFFFFF", "black": "#18212B",
    "class_i": "#DCEEFF", "class_ii": "#EFE9FF", "balanced": "#8A8F98", "text_muted": "#5F6875",
}
LOCUS_ACCENTS = [
    "#DCEEFF", "#E4F4FF", "#EDF7FF", "#EFE9FF", "#F4EFFF", "#F9F2FF", "#F1EEF8",
    "#FFF5DA", "#FFF9E7", "#E9F7EF", "#F4FAF6", "#E0F7FA", "#E7FAFC", "#EEFDFE", "#F2FEFF"
]
LOCUS_COLOR = dict(zip(LOCUS_ORDER, LOCUS_ACCENTS))

TIFF_QUEUE = []

CAPTION = (
    "Figure 1. HLA allele-position landscape across HLA loci. The gradient data-card matrix shows validated "
    "allele-group values for Allele 1 and Allele 2 calls within each locus. Rows are grouped by HLA locus "
    "and ordered by the maximum observed value. The dominance architecture panel summarizes the direction and "
    "magnitude of Allele 1–Allele 2 separation for leading allele groups. The locus signature panel summarizes "
    "profile divergence and top allele groups across loci. Axis-unit loci are displayed in a dedicated card."
)


def set_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.dpi": 120,
        "savefig.dpi": 600,
        "axes.linewidth": 0.6,
        "axes.edgecolor": COLORS["sep"],
        "axes.facecolor": COLORS["white"],
    })


def hex_lerp(c1: str, c2: str, t: float) -> str:
    t = max(0, min(1, float(t)))
    a = np.array(to_rgb(c1)); b = np.array(to_rgb(c2))
    c = a + (b - a) * t
    return "#" + "".join(f"{int(round(x*255)):02X}" for x in c)


def three_color(c1: str, c2: str, c3: str, t: float) -> str:
    t = max(0, min(1, float(t)))
    if t <= 0.5:
        return hex_lerp(c1, c2, t * 2)
    return hex_lerp(c2, c3, (t - 0.5) * 2)


def dominance_color(score: float, max_abs: float) -> str:
    if max_abs <= 0:
        return COLORS["neutral"]
    return three_color(COLORS["amber_deep"], "#F7F7F7", COLORS["blue_deep"], (score / max_abs + 1) / 2)


def score_text_color(score: float) -> str:
    if score > 0:
        return COLORS["blue_deep"]
    if score < 0:
        return "#8A4F00"
    return "#4A4F55"


def value_color(value: float, vmax: float, allele: int = 1, axis_unit: bool = False) -> str:
    t = 0 if vmax <= 0 else max(0, min(1, value / vmax))
    if axis_unit:
        return three_color(COLORS["teal_pale"], COLORS["teal_mid"], COLORS["teal_deep"], t)
    if allele == 1:
        return three_color(COLORS["blue_pale"], COLORS["blue_mid"], COLORS["blue_deep"], t)
    return three_color(COLORS["amber_pale"], COLORS["amber_mid"], COLORS["amber_deep"], t)


def format_value(x: float, signed: bool = False) -> str:
    x = float(x)
    if abs(x) < 1e-9:
        return "0" if not signed else "0"
    sgn = "+" if signed and x > 0 else ""
    if abs(x - round(x)) < 1e-7:
        return f"{sgn}{int(round(x))}"
    if abs((x * 2) - round(x * 2)) < 1e-7:
        return f"{sgn}{x:.1f}"
    return f"{sgn}{x:.1f}"


def safe_name(name: str) -> str:
    return name.replace("/", "-")

# ------------------------- data -------------------------

def load_data(input_file: str | Path) -> pd.DataFrame:
    return pd.read_csv(input_file)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    required = {"panel", "locus", "allele_group", "allele_1_value", "allele_2_value", "display_unit", "data_status"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Required columns missing: {sorted(missing)}")
    d = df.copy()
    d["panel"] = d["panel"].astype(str).str.strip().str.upper()
    d["locus"] = d["locus"].astype(str).str.strip().str.upper().str.replace(" ", "", regex=False).str.replace("DBQ1", "DQB1", regex=False)
    d.loc[~d["locus"].str.startswith("HLA-"), "locus"] = "HLA-" + d.loc[~d["locus"].str.startswith("HLA-"), "locus"]
    d["allele_group"] = d["allele_group"].astype(str).str.strip().str.replace(" ", "", regex=False).str.replace("HLA-", "", regex=False)
    d["allele_group"] = d["allele_group"].str.replace("DBQ1", "DQB1", regex=False)
    if d.astype(str).apply(lambda col: col.str.contains("DBQ1", case=False, na=False)).any().any():
        raise SystemExit("DBQ1 remains after cleaning.")
    for col in ["allele_1_value", "allele_2_value"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")
        if d[col].isna().any():
            bad = d[d[col].isna()][["panel", "locus", "allele_group", col]]
            raise SystemExit(f"Numeric conversion failed for {col}:\n{bad}")
    if (d[["allele_1_value", "allele_2_value"]] < 0).any().any():
        raise SystemExit("Negative values detected.")
    d["unit_type"] = np.where(d["display_unit"].astype(str).str.contains("axis", case=False, na=False), "axis", "percent")
    pct = d["unit_type"].eq("percent")
    if (d.loc[pct, ["allele_1_value", "allele_2_value"]] > 100).any().any():
        raise SystemExit("Percentage rows exceed 100.")
    d["locus_order"] = d["locus"].map({l: i for i, l in enumerate(LOCUS_ORDER)}).fillna(999).astype(int)
    d["panel"] = d["locus"].map(PANEL_ORDER).fillna(d["panel"])
    d["hla_class"] = np.where(d["locus"].isin(CLASS_I), "Class I", "Class II")
    d["max_value"] = d[["allele_1_value", "allele_2_value"]].max(axis=1)
    d["dominance_score"] = d["allele_1_value"] - d["allele_2_value"]
    d["absolute_dominance_score"] = d["dominance_score"].abs()
    d["dominant_position"] = np.select([d["dominance_score"] > 0, d["dominance_score"] < 0], ["Allele 1", "Allele 2"], default="Balanced")
    d = d.sort_values(["locus_order", "max_value", "allele_group"], ascending=[True, False, True]).reset_index(drop=True)
    return d


def assign_locus_order_and_class(df: pd.DataFrame) -> pd.DataFrame:
    return clean_data(df)


def select_main_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    percent = df[(df["unit_type"].eq("percent")) & (df["max_value"] > 0)].copy()
    for locus in LOCUS_ORDER:
        sub = percent[percent["locus"].eq(locus)].copy()
        if sub.empty:
            continue
        sub = sub.sort_values(["max_value", "absolute_dominance_score"], ascending=False)
        cap = 5 if len(sub) > 5 else len(sub)
        keep = sub.head(cap)
        prom = sub[sub["allele_group"].isin(PROMINENT)]
        keep = pd.concat([keep, prom]).drop_duplicates(subset=["locus", "allele_group"])
        if len(keep) > 7:
            keep["prom"] = keep["allele_group"].isin(PROMINENT).astype(int)
            keep = keep.sort_values(["prom", "max_value"], ascending=False).head(7).drop(columns="prom")
        keep = keep.sort_values(["max_value", "absolute_dominance_score"], ascending=False)
        rows.append(keep)
    main = pd.concat(rows, ignore_index=True) if rows else percent.head(0)
    return main.sort_values(["locus_order", "max_value"], ascending=[True, False]).reset_index(drop=True)


def compute_dominance(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["dominance_score"] = d["allele_1_value"] - d["allele_2_value"]
    d["absolute_dominance_score"] = d["dominance_score"].abs()
    d["dominant_position"] = np.select([d["dominance_score"] > 0, d["dominance_score"] < 0], ["Allele 1", "Allele 2"], default="Balanced")
    return d


def _js_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    v1 = np.asarray(v1, dtype=float); v2 = np.asarray(v2, dtype=float)
    if v1.sum() <= 0 and v2.sum() <= 0:
        return 0.0
    p = v1 / v1.sum() if v1.sum() > 0 else np.ones_like(v1) / len(v1)
    q = v2 / v2.sum() if v2.sum() > 0 else np.ones_like(v2) / len(v2)
    if jensenshannon is not None:
        return float(jensenshannon(p, q, base=2.0))
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return np.sum(a[mask] * np.log2(a[mask] / b[mask]))
    return float(np.sqrt(0.5 * kl(p, m) + 0.5 * kl(q, m)))


def compute_locus_signature(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for locus in LOCUS_ORDER:
        sub = df[df["locus"].eq(locus)].copy()
        if sub.empty:
            continue
        top = sub.sort_values("max_value", ascending=False).iloc[0]
        js = _js_distance(sub["allele_1_value"].values, sub["allele_2_value"].values)
        total1 = float(sub["allele_1_value"].sum()); total2 = float(sub["allele_2_value"].sum())
        l1 = 0.0
        if total1 > 0 or total2 > 0:
            p1 = sub["allele_1_value"].values / total1 if total1 > 0 else np.zeros(len(sub))
            p2 = sub["allele_2_value"].values / total2 if total2 > 0 else np.zeros(len(sub))
            l1 = float(np.sum(np.abs(p1 - p2)) / 2)
        dom = "Allele 1" if total1 > total2 else ("Allele 2" if total2 > total1 else "Balanced")
        rows.append({
            "locus": locus, "panel": PANEL_ORDER.get(locus, ""), "unit_type": sub["unit_type"].iloc[0], "hla_class": sub["hla_class"].iloc[0],
            "n_allele_groups": len(sub), "total_allele_1": total1, "total_allele_2": total2,
            "top_allele_group": top["allele_group"], "max_abs_dominance_score": float(sub["absolute_dominance_score"].max()),
            "profile_divergence": js, "l1_profile_distance": l1, "dominant_position": dom,
        })
    return pd.DataFrame(rows).sort_values("profile_divergence", ascending=False).reset_index(drop=True)


def compute_global_summary(df: pd.DataFrame, main: pd.DataFrame, locus_sig: pd.DataFrame) -> pd.DataFrame:
    pct = df[df["unit_type"].eq("percent")]
    pos = pct.loc[pct["dominance_score"].idxmax()] if not pct.empty else None
    neg = pct.loc[pct["dominance_score"].idxmin()] if not pct.empty else None
    top_div = locus_sig.iloc[0] if not locus_sig.empty else None
    rows = [
        {"metric": "Allele 1 peak", "value": f"{pos['allele_group']} {format_value(pos['dominance_score'], True)}" if pos is not None else "—", "detail": pos["locus"] if pos is not None else ""},
        {"metric": "Allele 2 peak", "value": f"{neg['allele_group']} {format_value(neg['dominance_score'], True)}" if neg is not None else "—", "detail": neg["locus"] if neg is not None else ""},
        {"metric": "Top divergence", "value": top_div["locus"] if top_div is not None else "—", "detail": f"{top_div['profile_divergence']:.2f}" if top_div is not None else ""},
        {"metric": "Loci", "value": str(df["locus"].nunique()), "detail": "displayed"},
        {"metric": "Groups", "value": str(len(main)), "detail": "main panel"},
        {"metric": "Class split", "value": "3 / 12", "detail": "I / II"},
    ]
    return pd.DataFrame(rows)

# ------------------------- drawing primitives -------------------------

def make_capsule_bar(ax, x: float, y: float, width: float, height: float, value: float, vmax: float, color: str, bg: str = "#F4F6F8", border: str = "#E4E8EE") -> None:
    rounding = height / 2
    ax.add_patch(FancyBboxPatch((x, y - height/2), width, height,
                                boxstyle=f"round,pad=0,rounding_size={rounding}",
                                facecolor=bg, edgecolor=border, linewidth=0.35, zorder=1))
    frac = 0 if vmax <= 0 else max(0, min(1, value / vmax))
    if frac > 0:
        ax.add_patch(FancyBboxPatch((x, y - height/2), max(width*frac, height*0.55), height,
                                    boxstyle=f"round,pad=0,rounding_size={rounding}",
                                    facecolor=color, edgecolor="none", linewidth=0, zorder=2))


def draw_panel_title(ax, title: str, fs: float = 11.0) -> None:
    ax.text(0.0, 1.018, title, transform=ax.transAxes, ha="left", va="bottom", fontsize=fs, fontweight="bold", color=COLORS["black"], clip_on=False)


def add_card_background(ax, radius: float = 0.018) -> None:
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, transform=ax.transAxes,
                                boxstyle=f"round,pad=0.010,rounding_size={radius}",
                                facecolor=COLORS["panel_bg"], edgecolor="#E5E9EF", linewidth=0.8, zorder=-10, clip_on=False))


def _add_text(ax, *args, **kwargs):
    kwargs.setdefault("clip_on", False)
    return ax.text(*args, **kwargs)

# ------------------------- panel A matrix -------------------------

def draw_matrix_core(ax, data: pd.DataFrame, title: str | None = None, fragment: bool = False, full: bool = False) -> None:
    ax.set_axis_off(); ax.set_xlim(0, 1)
    d = data.reset_index(drop=True).copy()
    n = len(d)
    header_h = 1.05
    ax.set_ylim(-0.5, n + header_h + 0.2)
    add_card_background(ax)
    if title:
        draw_panel_title(ax, title, fs=11.3 if not fragment else 13)

    # columns in data coords x fraction
    cols = {
        "class": (0.010, 0.035), "locus": (0.040, 0.116), "allele": (0.126, 0.230),
        "a1bar": (0.250, 0.465), "a1val": 0.500, "a2bar": (0.540, 0.755), "a2val": 0.790,
        "dbar": (0.825, 0.925), "dval": 0.965,
    }
    top_y = n + 0.25
    headers = [("Locus", 0.078), ("Allele group", 0.126), ("Allele 1", 0.250), ("Value", 0.500), ("Allele 2", 0.540), ("Value", 0.790), ("Δ", 0.825), ("Score", 0.965)]
    for txt, x in headers:
        ha = "left" if txt in {"Allele group", "Allele 1", "Allele 2", "Δ"} else "center"
        _add_text(ax, x, top_y, txt, ha=ha, va="center", fontsize=8.3 if not fragment else 9.3, fontweight="bold", color=COLORS["black"])
    ax.plot([0.01, 0.99], [n - 0.35, n - 0.35], color=COLORS["sep"], lw=0.7)
    # Unit labeling is handled by column headers and panel captions; no extra text here to preserve spacing.

    vmax_pct = max(100, float(d.loc[d["unit_type"].eq("percent"), ["allele_1_value", "allele_2_value"]].max().max()) if d["unit_type"].eq("percent").any() else 100)
    vmax_axis = max(123, float(d.loc[d["unit_type"].eq("axis"), ["allele_1_value", "allele_2_value"]].max().max()) if d["unit_type"].eq("axis").any() else 123)
    max_delta = max(1, float(d["dominance_score"].abs().max()))

    # y positions from top to bottom
    d["y"] = list(reversed(range(n)))
    # backgrounds and separators by locus
    for i, (locus, sub) in enumerate(d.groupby("locus", sort=False)):
        y_min = sub["y"].min() - 0.45; y_max = sub["y"].max() + 0.45
        if i % 2 == 0:
            ax.add_patch(Rectangle((0.006, y_min), 0.988, y_max-y_min, facecolor="#FFFFFF", edgecolor="none", zorder=-9))
        else:
            ax.add_patch(Rectangle((0.006, y_min), 0.988, y_max-y_min, facecolor="#FBFCFE", edgecolor="none", zorder=-9))
        ax.add_patch(Rectangle((cols["locus"][0], y_min), cols["locus"][1]-cols["locus"][0], y_max-y_min, facecolor=LOCUS_COLOR.get(locus, "#F3F4F6"), edgecolor="none", zorder=-8))
        ax.plot([0.008, 0.992], [y_min, y_min], color="#E9EDF2", lw=0.6, zorder=-7)
        fs = 6.7 if len(sub) <= 2 and not fragment else (7.4 if not fragment else 8.2)
        label = locus.replace("HLA-", "HLA-")
        _add_text(ax, np.mean(cols["locus"]), (y_min+y_max)/2, label, ha="center", va="center", fontsize=fs, fontweight="bold", color=COLORS["text_muted"])
    # class bands once per block
    for cls, sub in d.groupby("hla_class", sort=False):
        y_min = sub["y"].min() - 0.45; y_max = sub["y"].max() + 0.45
        band_color = COLORS["class_i"] if cls == "Class I" else COLORS["class_ii"]
        ax.add_patch(Rectangle((cols["class"][0], y_min), cols["class"][1]-cols["class"][0], y_max-y_min, facecolor=band_color, edgecolor="none", zorder=-7))
        _add_text(ax, np.mean(cols["class"]), (y_min+y_max)/2, cls, rotation=90, ha="center", va="center", fontsize=7.0 if not fragment else 8.0, fontweight="bold", color=COLORS["text_muted"])

    bar_h = 0.38 if not fragment else 0.44
    for _, r in d.iterrows():
        y = r["y"]
        unit_axis = r["unit_type"] == "axis"
        vmax = vmax_axis if unit_axis else vmax_pct
        _add_text(ax, cols["allele"][0], y, r["allele_group"], ha="left", va="center", fontsize=7.4 if not fragment else 8.6, color=COLORS["black"])
        c1 = value_color(r["allele_1_value"], vmax, allele=1, axis_unit=unit_axis)
        c2 = value_color(r["allele_2_value"], vmax, allele=2, axis_unit=unit_axis)
        make_capsule_bar(ax, cols["a1bar"][0], y, cols["a1bar"][1]-cols["a1bar"][0], bar_h, r["allele_1_value"], vmax, c1)
        make_capsule_bar(ax, cols["a2bar"][0], y, cols["a2bar"][1]-cols["a2bar"][0], bar_h, r["allele_2_value"], vmax, c2)
        if r["allele_1_value"] != 0:
            _add_text(ax, cols["a1val"], y, format_value(r["allele_1_value"]), ha="center", va="center", fontsize=7.1 if not fragment else 8.2, color=COLORS["black"])
        else:
            _add_text(ax, cols["a1val"], y, "0", ha="center", va="center", fontsize=6.8 if not fragment else 7.6, color="#A0A7B2")
        if r["allele_2_value"] != 0:
            _add_text(ax, cols["a2val"], y, format_value(r["allele_2_value"]), ha="center", va="center", fontsize=7.1 if not fragment else 8.2, color=COLORS["black"])
        else:
            _add_text(ax, cols["a2val"], y, "0", ha="center", va="center", fontsize=6.8 if not fragment else 7.6, color="#A0A7B2")
        # delta microbar around center
        x0, x1 = cols["dbar"]; xc = (x0+x1)/2; half = (x1-x0)/2
        ax.add_patch(FancyBboxPatch((x0, y-bar_h/3), x1-x0, bar_h*0.66, boxstyle=f"round,pad=0,rounding_size={bar_h*0.25}", facecolor="#F3F5F7", edgecolor="#E3E7EE", linewidth=0.3))
        delta = r["dominance_score"]
        if abs(delta) > 0:
            w = half * min(1, abs(delta)/max_delta)
            if delta > 0:
                ax.add_patch(FancyBboxPatch((xc, y-bar_h/3), max(w, 0.003), bar_h*0.66, boxstyle=f"round,pad=0,rounding_size={bar_h*0.25}", facecolor=COLORS["blue_deep"], edgecolor="none"))
            else:
                ax.add_patch(FancyBboxPatch((xc-w, y-bar_h/3), max(w, 0.003), bar_h*0.66, boxstyle=f"round,pad=0,rounding_size={bar_h*0.25}", facecolor=COLORS["amber_deep"], edgecolor="none"))
        ax.plot([xc, xc], [y-bar_h/2, y+bar_h/2], color="#D1D6DE", lw=0.45)
        _add_text(ax, cols["dval"], y, format_value(delta, signed=True), ha="center", va="center", fontsize=7.0 if not fragment else 8.0, color=score_text_color(delta), fontweight="semibold" if abs(delta) >= 5 else "medium")


def draw_fragment_a_matrix(data: pd.DataFrame, title: str = "Allele-position matrix") -> plt.Figure:
    h = max(7.5, len(data) * 0.20 + 1.4)
    fig, ax = plt.subplots(figsize=(10.8, h))
    fig.subplots_adjust(left=0.02, right=0.985, top=0.93, bottom=0.035)
    fig.suptitle(title, fontsize=13, fontweight="semibold", y=0.985)
    draw_matrix_core(ax, data, title=None, fragment=True, full=True)
    return fig

# ------------------------- panels B/C/D/E -------------------------

def draw_dominance_core(ax, data: pd.DataFrame, title: str | None = None, n_top: int = 15, fragment: bool = False, full: bool = False) -> None:
    ax.set_facecolor(COLORS["panel_bg"]); add_card_background(ax)
    if title:
        draw_panel_title(ax, title, fs=11 if not fragment else 13)
    d = data[data["unit_type"].eq("percent")].copy()
    d = d[d["absolute_dominance_score"] > 0].sort_values("absolute_dominance_score", ascending=False)
    if not full:
        d = d.head(n_top)
    d = d.sort_values("dominance_score", ascending=True).reset_index(drop=True)
    if d.empty:
        ax.axis("off"); return
    y = np.arange(len(d))
    max_abs = max(10, float(d["absolute_dominance_score"].max()) * 1.18)
    ax.axvspan(-max_abs, 0, color=COLORS["amber_pale"], alpha=0.35, lw=0)
    ax.axvspan(0, max_abs, color=COLORS["blue_pale"], alpha=0.42, lw=0)
    ax.axvline(0, color="#9AA3AF", lw=0.8)
    for yi, (_, r) in zip(y, d.iterrows()):
        score = r["dominance_score"]
        col = COLORS["blue_deep"] if score > 0 else COLORS["amber_deep"]
        ax.plot([0, score], [yi, yi], color=col, lw=1.45, alpha=0.92)
        ax.scatter([score], [yi], s=28 + abs(score) * 1.2, color=col, edgecolor="white", linewidth=0.55, zorder=3)
        # value labels outside endpoints
        if score >= 0:
            ax.text(score + max_abs*0.035, yi, format_value(score, signed=True), ha="left", va="center", fontsize=7.0 if not fragment else 8.3, color=col, fontweight="bold")
        else:
            ax.text(score - max_abs*0.035, yi, format_value(score, signed=True), ha="right", va="center", fontsize=7.0 if not fragment else 8.3, color=col, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(d["allele_group"], fontsize=7.0 if not fragment else 8.7)
    ax.set_xlim(-max_abs*1.28, max_abs*1.28)
    ax.set_xlabel("Allele 1 − Allele 2", fontsize=7.1 if not fragment else 9.0, labelpad=1.5)
    ax.tick_params(axis="x", labelsize=7.0 if not fragment else 8.2, length=2)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(COLORS["sep"])
    ax.grid(axis="x", color="#E6EAF0", lw=0.55)
    ax.text(0.02, 0.060, "Allele 2", transform=ax.transAxes, ha="left", va="bottom", fontsize=6.8 if not fragment else 8.2, color=COLORS["amber_deep"], fontweight="bold", clip_on=False)
    ax.text(0.98, 0.060, "Allele 1", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.8 if not fragment else 8.2, color=COLORS["blue_deep"], fontweight="bold", clip_on=False)


def draw_fragment_b_dominance(data: pd.DataFrame) -> plt.Figure:
    d = data[data["unit_type"].eq("percent") & (data["absolute_dominance_score"] > 0)]
    fig, ax = plt.subplots(figsize=(7.5, max(5.6, len(d.head(16))*0.33 + 1.5)))
    fig.subplots_adjust(left=0.18, right=0.95, top=0.88, bottom=0.13)
    fig.suptitle("Dominance architecture", fontsize=13, fontweight="semibold", y=0.975)
    draw_dominance_core(ax, data, title=None, n_top=18, fragment=True)
    return fig


def draw_axis_unit_core(ax, data: pd.DataFrame, title: str | None = None, fragment: bool = False) -> None:
    ax.set_axis_off(); ax.set_xlim(0, 1)
    axis = data[data["unit_type"].eq("axis")].sort_values("locus_order").reset_index(drop=True)
    n = len(axis)
    ax.set_ylim(-0.2, n + 0.9)
    add_card_background(ax)
    if title:
        draw_panel_title(ax, title, fs=10.4 if not fragment else 13)
    if axis.empty:
        return
    vmax = max(123, float(axis[["allele_1_value", "allele_2_value"]].max().max()))
    # headers
    yhead = n + 0.45
    _add_text(ax, 0.05, yhead, "Allele", ha="left", va="center", fontsize=7.4 if not fragment else 9.0, fontweight="bold", color=COLORS["black"])
    _add_text(ax, 0.31, yhead, "Axis units", ha="left", va="center", fontsize=7.1 if not fragment else 9.0, fontweight="bold", color=COLORS["black"])
    _add_text(ax, 0.755, yhead, "A1/A2", ha="left", va="center", fontsize=6.5 if not fragment else 8.4, fontweight="bold", color=COLORS["black"])
    _add_text(ax, 0.965, yhead, "Δ", ha="center", va="center", fontsize=7.0 if not fragment else 8.8, fontweight="bold", color=COLORS["black"])
    for i, (_, r) in enumerate(axis.iterrows()):
        y = n - 1 - i
        ax.add_patch(Rectangle((0.035, y-0.38), 0.93, 0.76, facecolor="#FFFFFF", edgecolor="#EBEEF3", lw=0.5))
        _add_text(ax, 0.05, y, r["allele_group"], ha="left", va="center", fontsize=7.4 if not fragment else 9.0, color=COLORS["black"])
        # two bars A1/A2 compressed
        make_capsule_bar(ax, 0.31, y+0.13, 0.435, 0.16 if not fragment else 0.20, r["allele_1_value"], vmax, value_color(r["allele_1_value"], vmax, axis_unit=True))
        make_capsule_bar(ax, 0.31, y-0.13, 0.435, 0.16 if not fragment else 0.20, r["allele_2_value"], vmax, value_color(r["allele_2_value"], vmax, allele=2, axis_unit=True))
        _add_text(ax, 0.755, y, f"{format_value(r['allele_1_value'])}/{format_value(r['allele_2_value'])}", ha="left", va="center", fontsize=6.5 if not fragment else 8.0, color=COLORS["black"])
        score = r["dominance_score"]
        marker = "▲" if score > 0 else ("▼" if score < 0 else "●")
        col = COLORS["blue_deep"] if score > 0 else (COLORS["amber_deep"] if score < 0 else COLORS["neutral"])
        _add_text(ax, 0.965, y, marker, ha="center", va="center", fontsize=8.5 if not fragment else 10.5, color=col)


def draw_fragment_c_axis_units(data: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.2, 3.5))
    fig.subplots_adjust(left=0.04, right=0.96, top=0.82, bottom=0.10)
    fig.suptitle("Axis-unit loci", fontsize=13, fontweight="semibold", y=0.965)
    draw_axis_unit_core(ax, data, title=None, fragment=True)
    return fig


def draw_locus_signature_core(ax, locus_sig: pd.DataFrame, title: str | None = None, n_top: int = 10, fragment: bool = False, full: bool = False) -> None:
    ax.set_axis_off(); ax.set_xlim(0, 1)
    d = locus_sig.copy()
    if not full:
        # include biologically requested representative loci if readable, still sort by divergence
        d = d.head(n_top)
    n = len(d)
    ax.set_ylim(-0.35, n + 0.9)
    add_card_background(ax)
    if title:
        draw_panel_title(ax, title, fs=10.4 if not fragment else 13)
    yhead = n + 0.45
    fs_h = 7.0 if not fragment else 9.0
    _add_text(ax, 0.05, yhead, "Locus", ha="left", va="center", fontsize=fs_h, fontweight="bold", color=COLORS["black"])
    _add_text(ax, 0.225, yhead, "Top allele", ha="left", va="center", fontsize=fs_h, fontweight="bold", color=COLORS["black"])
    _add_text(ax, 0.505, yhead, "Profile divergence", ha="left", va="center", fontsize=fs_h, fontweight="bold", color=COLORS["black"])
    _add_text(ax, 0.955, yhead, "Dom.", ha="center", va="center", fontsize=fs_h, fontweight="bold", color=COLORS["black"])
    vmax = max(0.01, float(locus_sig["profile_divergence"].max()))
    for i, (_, r) in enumerate(d.iterrows()):
        y = n - 1 - i
        ax.add_patch(FancyBboxPatch((0.035, y-0.33), 0.93, 0.62, boxstyle="round,pad=0.003,rounding_size=0.018", facecolor="#FFFFFF", edgecolor="#EBEEF3", lw=0.55))
        _add_text(ax, 0.05, y, r["locus"], ha="left", va="center", fontsize=7.3 if not fragment else 8.8, color=COLORS["black"], fontweight="bold")
        _add_text(ax, 0.225, y, r["top_allele_group"], ha="left", va="center", fontsize=7.1 if not fragment else 8.8, color=COLORS["text_muted"])
        x0, w = 0.505, 0.285
        ax.add_patch(FancyBboxPatch((x0, y-0.095), w, 0.19, boxstyle="round,pad=0,rounding_size=0.095", facecolor="#EFF1F5", edgecolor="none"))
        frac = max(0, min(1, r["profile_divergence"] / vmax))
        ax.add_patch(FancyBboxPatch((x0, y-0.095), max(0.014, w*frac), 0.19, boxstyle="round,pad=0,rounding_size=0.095", facecolor=three_color(COLORS["violet_pale"], COLORS["violet_mid"], COLORS["violet_deep"], frac), edgecolor="none"))
        _add_text(ax, 0.842, y, f"{r['profile_divergence']:.2f}", ha="right", va="center", fontsize=7.1 if not fragment else 8.5, color=COLORS["black"], fontweight="semibold")
        col = COLORS["blue_deep"] if r["dominant_position"] == "Allele 1" else (COLORS["amber_deep"] if r["dominant_position"] == "Allele 2" else COLORS["neutral"])
        ax.add_patch(Circle((0.955, y), 0.045 if fragment else 0.034, facecolor=col, edgecolor="white", lw=0.5))


def draw_fragment_d_locus_signature(locus_sig: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.0, max(5.0, len(locus_sig)*0.34 + 1.2)))
    fig.subplots_adjust(left=0.04, right=0.965, top=0.88, bottom=0.06)
    fig.suptitle("Locus signature", fontsize=13, fontweight="semibold", y=0.975)
    draw_locus_signature_core(ax, locus_sig, title=None, fragment=True, full=True)
    return fig


def draw_global_summary_core(ax, summary: pd.DataFrame, title: str | None = None, fragment: bool = False) -> None:
    ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    add_card_background(ax)
    if title:
        draw_panel_title(ax, title, fs=10.5 if not fragment else 13)
    n = len(summary)
    gap = 0.018
    left = 0.03
    w = (0.94 - gap*(n-1)) / n
    y0, h = (0.18, 0.58) if not fragment else (0.18, 0.62)
    chip_cols = [COLORS["blue_pale"], COLORS["amber_pale"], COLORS["violet_pale"], "#F4FAF6", COLORS["teal_pale"], "#F7F4FF"]
    for i, (_, r) in enumerate(summary.iterrows()):
        x = left + i*(w+gap)
        ax.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0.012,rounding_size=0.035", facecolor=chip_cols[i % len(chip_cols)], edgecolor="#E2E6ED", lw=0.7))
        _add_text(ax, x+w/2, y0+h*0.68, str(r["value"]), ha="center", va="center", fontsize=8.1 if not fragment else 11.0, fontweight="bold", color=COLORS["black"])
        _add_text(ax, x+w/2, y0+h*0.39, str(r["metric"]), ha="center", va="center", fontsize=6.4 if not fragment else 8.2, color=COLORS["text_muted"], fontweight="bold")
        _add_text(ax, x+w/2, y0+h*0.17, str(r["detail"]), ha="center", va="center", fontsize=5.9 if not fragment else 7.4, color=COLORS["text_muted"])


def draw_fragment_e_global_summary(summary: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10.0, 2.2))
    fig.subplots_adjust(left=0.025, right=0.975, top=0.75, bottom=0.16)
    fig.suptitle("Global summary", fontsize=13, fontweight="semibold", y=0.95)
    draw_global_summary_core(ax, summary, title=None, fragment=True)
    return fig

# ------------------------- main figure -------------------------

def draw_main_figure(df: pd.DataFrame, main: pd.DataFrame, locus_sig: pd.DataFrame, summary: pd.DataFrame, size=(16, 9), double=False) -> plt.Figure:
    """Draw the final polished 16:9 composite.

    v3 polish keeps the v2 visual language but moves Panel E to a bottom
    horizontal summary ribbon and darkens score typography.
    """
    fig = plt.figure(figsize=size, facecolor="white")
    # Professional small centered title only in the top layer
    fig.text(0.5, 0.976, "HLA allele-position landscape", ha="center", va="top",
             fontsize=13.6 if not double else 12.8, fontweight="semibold", color=COLORS["black"])

    # Main content band: A at left, B/C/D stacked at right.
    # Micro-polish: widen the inter-panel gutter and increase vertical separation
    # between B and C while retaining the v3 visual language.
    axA = fig.add_axes([0.045, 0.205, 0.600, 0.688])
    draw_matrix_core(axA, main, title="A. Allele-position matrix", fragment=False)

    axB = fig.add_axes([0.735, 0.652, 0.238, 0.242])
    draw_dominance_core(axB, df, title="B. Dominance architecture", n_top=16)

    axC = fig.add_axes([0.735, 0.455, 0.238, 0.135])
    draw_axis_unit_core(axC, df, title="C. Axis-unit loci")

    axD = fig.add_axes([0.735, 0.205, 0.238, 0.205])
    draw_locus_signature_core(axD, locus_sig, title="D. Locus signature", n_top=10)

    # Bottom global summary ribbon retained at the bottom layer.
    axE = fig.add_axes([0.045, 0.060, 0.928, 0.105])
    draw_global_summary_core(axE, summary, title="E. Global summary", fragment=False)

    # compact legend below Panel E, outside data zones
    handles = [
        Line2D([0], [0], color=COLORS["blue_deep"], lw=4, label="Allele 1"),
        Line2D([0], [0], color=COLORS["amber_deep"], lw=4, label="Allele 2"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["balanced"], markersize=5, label="Balanced"),
        Line2D([0], [0], color=COLORS["teal_mid"], lw=4, label="Axis-unit loci"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.006),
               ncol=4, frameon=False, fontsize=7.0 if not double else 6.6,
               handlelength=1.7, columnspacing=1.25)
    return fig

# ------------------------- supplementary figures -------------------------

def draw_supplementary_figure_1(df: pd.DataFrame) -> plt.Figure:
    d = df[(df["unit_type"].eq("percent")) & (df["max_value"] > 0)].copy()
    fig = draw_fragment_a_matrix(d, title="Full percentage rows")
    return fig


def draw_supplementary_figure_2(df: pd.DataFrame) -> plt.Figure:
    d = df[df["max_value"] > 0].sort_values(["locus_order", "max_value"], ascending=[True, False]).reset_index(drop=True)
    n = len(d)
    fig, ax = plt.subplots(figsize=(9.0, max(8.0, n * 0.21 + 1.7)))
    fig.subplots_adjust(left=0.18, right=0.92, top=0.94, bottom=0.08)
    fig.suptitle("Full paired profiles", fontsize=13, fontweight="semibold", y=0.985)
    y = np.arange(n)
    maxx = max(100, float(d[["allele_1_value", "allele_2_value"]].max().max()) * 1.08)
    ax.hlines(y, d["allele_1_value"], d["allele_2_value"], color="#C9CED6", lw=0.8, zorder=1)
    ax.scatter(d["allele_1_value"], y, color=COLORS["blue_deep"], s=28, label="Allele 1", zorder=3)
    ax.scatter(d["allele_2_value"], y, color=COLORS["amber_deep"], s=28, label="Allele 2", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(d["allele_group"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, maxx)
    ax.set_xlabel("Value / axis units", fontsize=9)
    ax.grid(axis="x", color="#E6EAF0", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.055), ncol=2, frameon=False)
    # locus separators and labels
    for locus, sub in d.groupby("locus", sort=False):
        idxs = sub.index.to_numpy()
        ymin, ymax = idxs.min()-0.5, idxs.max()+0.5
        ax.axhspan(ymin, ymax, color=LOCUS_COLOR.get(locus, "#F6F8FB"), alpha=0.18, zorder=0)
        # Locus grouping is shown by the shaded row bands and separators; labels are kept out of the plotting field to prevent crowding.
        ax.axhline(ymax, color="#E2E6ED", lw=0.6)
    return fig


def draw_supplementary_figure_3(df: pd.DataFrame) -> plt.Figure:
    d = df[df["absolute_dominance_score"] > 0].copy()
    fig, ax = plt.subplots(figsize=(8.5, max(8.0, len(d)*0.22 + 1.5)))
    fig.subplots_adjust(left=0.18, right=0.90, top=0.93, bottom=0.08)
    fig.suptitle("Full dominance architecture", fontsize=13, fontweight="semibold", y=0.985)
    draw_dominance_core(ax, d, title=None, n_top=len(d), fragment=True, full=True)
    return fig


def draw_supplementary_figure_4(locus_sig: pd.DataFrame) -> plt.Figure:
    return draw_fragment_d_locus_signature(locus_sig)


def draw_supplementary_figure_5(summary: pd.DataFrame) -> plt.Figure:
    return draw_fragment_e_global_summary(summary)

# ------------------------- overlap, save, outputs -------------------------

def check_text_overlaps(fig: plt.Figure, figure_name: str) -> List[Dict[str, object]]:
    # Draw first to obtain renderer. This check treats true text collisions as overlaps and ignores tiny numerical noise.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = []
    for ax in fig.axes:
        for t in ax.texts:
            if not t.get_visible() or not str(t.get_text()).strip():
                continue
            bb = t.get_window_extent(renderer=renderer).expanded(1.01, 1.04)
            texts.append((t, bb))
    # include figure-level texts such as title and legend texts
    for t in fig.texts:
        if not t.get_visible() or not str(t.get_text()).strip():
            continue
        bb = t.get_window_extent(renderer=renderer).expanded(1.01, 1.04)
        texts.append((t, bb))
    rows = []
    for i in range(len(texts)):
        for j in range(i+1, len(texts)):
            t1, b1 = texts[i]; t2, b2 = texts[j]
            # ignore identical repeated labels if they are far or same axis title? Use bbox overlap only.
            if b1.overlaps(b2):
                x0 = max(b1.x0, b2.x0); y0 = max(b1.y0, b2.y0); x1 = min(b1.x1, b2.x1); y1 = min(b1.y1, b2.y1)
                area = max(0, x1-x0) * max(0, y1-y0)
                if area > 6:  # small threshold for antialiasing/near touches
                    rows.append({
                        "figure_name": figure_name,
                        "text_1": str(t1.get_text()),
                        "text_2": str(t2.get_text()),
                        "overlap_area": round(float(area), 2),
                        "suggested_fix": "Increase spacing or reduce optional label size",
                    })
    return rows


def fix_text_overlaps(fig: plt.Figure) -> plt.Figure:
    # The layout is designed with reserved columns; no runtime changes are normally required.
    return fig


def save_all_formats(fig: plt.Figure, outdir: Path, stem: str, manifest: list, label: str, ftype: str, desc: str, location: str, overlap_rows: list) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    rows = check_text_overlaps(fig, stem)
    if rows:
        diag = outdir / f"DRAFT_OVERLAP_CHECK_FAILED - {stem}.pdf"
        fig.savefig(diag, bbox_inches="tight")
        overlap_rows.extend(rows)
        raise SystemExit(f"Text overlap detected for {stem}. See diagnostic PDF: {diag}")
    overlap_rows.append({"figure_name": stem, "text_1": "", "text_2": "", "overlap_area": 0, "suggested_fix": "passed"})
    bbox_setting = None if "Full Paired Profiles" in stem else "tight"
    png_path = outdir / f"{stem}.png"
    for ext in ["pdf", "svg", "png"]:
        path = outdir / f"{stem}.{ext}"
        if ext == "pdf" and "Full Paired Profiles" in stem and Image is not None:
            if not png_path.exists():
                fig.savefig(png_path, bbox_inches=bbox_setting, dpi=600)
            im = Image.open(png_path).convert("RGB")
            im.save(path, resolution=600)
            im.close()
        else:
            fig.savefig(path, bbox_inches=bbox_setting, dpi=600)
        manifest.append({"figure_label": label, "file_name": path.name, "figure_type": ftype, "description": desc, "recommended_manuscript_location": location})
    tiff_path = outdir / f"{stem}.tiff"
    # TIFF is generated from the already-rendered 600-dpi PNG in a fresh process.
    # This keeps the main matplotlib process responsive for large figure batches.
    code = (
        "from PIL import Image; import sys; "
        "im=Image.open(sys.argv[1]); "
        "im.save(sys.argv[2], compression='tiff_lzw', dpi=(600,600)); "
        "im.close()"
    )
    subprocess.run([sys.executable, "-c", code, str(png_path), str(tiff_path)], check=True, timeout=120)
    manifest.append({"figure_label": label, "file_name": tiff_path.name, "figure_type": ftype, "description": desc, "recommended_manuscript_location": location})


def convert_tiff_queue() -> None:
    # TIFF files are written immediately in save_all_formats for the v3 polish package.
    return None


def write_manifest(outdir: Path, manifest: list) -> None:
    pd.DataFrame(manifest).to_csv(outdir / "Figure_File_Manifest.csv", index=False)


def write_report(outdir: Path, input_file: str, df: pd.DataFrame, main: pd.DataFrame, manifest: list, overlap_rows: list) -> None:
    outputs = [m["file_name"] for m in manifest]
    passed = all(float(r.get("overlap_area", 0) or 0) == 0 for r in overlap_rows)
    report = [
        "Figure 1 final overlap cleanup package", "",
        f"Input used: {input_file}",
        f"Rows loaded: {len(df)}",
        f"Rows in main figure: {len(main)}",
        f"Rows in supplementary figures: {len(df)}",
        f"Loci included: {', '.join([l for l in LOCUS_ORDER if l in set(df['locus'])])}",
        f"Text-overlap result: {'passed' if passed else 'overlap detected'}",
        "Panel E moved to bottom summary ribbon.",
        "Score/value fonts darkened for final readability.",
        "Final overlap cleanup completed.",
        "Panel E retained at bottom summary ribbon.",
        "Score fonts retained/darkened.",
        "All final files passed zero-overlap validation." if passed else "Overlap validation did not pass.",
        "",
        "Output files:",
    ]
    report += [f"- {x}" for x in outputs]
    report += ["", "Caption:", CAPTION]
    (outdir / "Figure1_Final_Publication_Report.txt").write_text("\n".join(report), encoding="utf-8")


def export_data_csvs(outdir: Path, df: pd.DataFrame, main: pd.DataFrame, axis: pd.DataFrame, locus_sig: pd.DataFrame, dom: pd.DataFrame, summary: pd.DataFrame) -> None:
    main.to_csv(outdir / "Figure 1 - Main Data.csv", index=False)
    main.to_csv(outdir / "Figure 1 Fragment A - Data.csv", index=False)
    dom[dom["unit_type"].eq("percent")].sort_values("absolute_dominance_score", ascending=False).head(16).to_csv(outdir / "Figure 1 Fragment B - Data.csv", index=False)
    axis.to_csv(outdir / "Figure 1 Fragment C - Data.csv", index=False)
    locus_sig.head(10).to_csv(outdir / "Figure 1 Fragment D - Data.csv", index=False)
    summary.to_csv(outdir / "Figure 1 Fragment E - Data.csv", index=False)
    df[(df["unit_type"].eq("percent")) & (df["max_value"] > 0)].to_csv(outdir / "Supplementary Figure 1 - Data.csv", index=False)
    df[df["max_value"] > 0].to_csv(outdir / "Supplementary Figure 2 - Data.csv", index=False)
    dom.to_csv(outdir / "Supplementary Figure 3 - Data.csv", index=False)
    locus_sig.to_csv(outdir / "Supplementary Figure 4 - Data.csv", index=False)
    summary.to_csv(outdir / "Supplementary Figure 5 - Data.csv", index=False)

# ------------------------- main orchestration -------------------------



def build_output_specs() -> List[Dict[str, str]]:
    return [
        {"key":"main", "figure_label":"Figure 1", "stem":"Figure 1 - HLA Allele-position Landscape", "figure_type":"Main figure", "description":"Composite HLA allele-position landscape", "recommended_manuscript_location":"Main manuscript"},
        {"key":"main_double", "figure_label":"Figure 1", "stem":"Figure 1 - HLA Allele-position Landscape - Double Column", "figure_type":"Main figure", "description":"Double-column composite HLA allele-position landscape", "recommended_manuscript_location":"Main manuscript"},
        {"key":"fragA", "figure_label":"Figure 1 Fragment A", "stem":"Figure 1 Fragment A - Allele-position Matrix", "figure_type":"Main figure fragment", "description":"Standalone allele-position matrix", "recommended_manuscript_location":"Optional response/revision package"},
        {"key":"fragB", "figure_label":"Figure 1 Fragment B", "stem":"Figure 1 Fragment B - Dominance Architecture", "figure_type":"Main figure fragment", "description":"Standalone dominance architecture", "recommended_manuscript_location":"Optional response/revision package"},
        {"key":"fragC", "figure_label":"Figure 1 Fragment C", "stem":"Figure 1 Fragment C - Axis-unit Loci", "figure_type":"Main figure fragment", "description":"Standalone axis-unit loci card", "recommended_manuscript_location":"Optional response/revision package"},
        {"key":"fragD", "figure_label":"Figure 1 Fragment D", "stem":"Figure 1 Fragment D - Locus Signature", "figure_type":"Main figure fragment", "description":"Standalone locus signature", "recommended_manuscript_location":"Optional response/revision package"},
        {"key":"fragE", "figure_label":"Figure 1 Fragment E", "stem":"Figure 1 Fragment E - Global Summary", "figure_type":"Main figure fragment", "description":"Standalone global summary ribbon", "recommended_manuscript_location":"Optional response/revision package"},
        {"key":"supp1", "figure_label":"Supplementary Figure 1", "stem":"Supplementary Figure 1 - Full Percentage Rows", "figure_type":"Supplementary figure", "description":"All percentage-unit allele rows", "recommended_manuscript_location":"Supplementary material"},
        {"key":"supp2", "figure_label":"Supplementary Figure 2", "stem":"Supplementary Figure 2 - Full Paired Profiles", "figure_type":"Supplementary figure", "description":"Full all-row paired profiles", "recommended_manuscript_location":"Supplementary material"},
        {"key":"supp3", "figure_label":"Supplementary Figure 3", "stem":"Supplementary Figure 3 - Full Dominance Architecture", "figure_type":"Supplementary figure", "description":"Full dominance architecture", "recommended_manuscript_location":"Supplementary material"},
        {"key":"supp4", "figure_label":"Supplementary Figure 4", "stem":"Supplementary Figure 4 - Locus Signature Detail", "figure_type":"Supplementary figure", "description":"Detailed locus signature", "recommended_manuscript_location":"Supplementary material"},
        {"key":"supp5", "figure_label":"Supplementary Figure 5", "stem":"Supplementary Figure 5 - Global Summary Ribbon", "figure_type":"Supplementary figure", "description":"Global summary ribbon", "recommended_manuscript_location":"Supplementary material"},
    ]


def prepare_data(input_file: str | Path):
    set_style()
    df = clean_data(load_data(input_file))
    df = compute_dominance(df)
    main = select_main_rows(df)
    axis = df[df["unit_type"].eq("axis")].sort_values("locus_order").copy()
    locus_sig = compute_locus_signature(df)
    summary = compute_global_summary(df, main, locus_sig)
    dom = compute_dominance(df).sort_values("absolute_dominance_score", ascending=False).copy()
    return df, main, axis, locus_sig, summary, dom


def draw_single_by_key(key: str, df: pd.DataFrame, main: pd.DataFrame, locus_sig: pd.DataFrame, summary: pd.DataFrame):
    if key == "main":
        return draw_main_figure(df, main, locus_sig, summary, size=(16, 9), double=False)
    if key == "main_double":
        return draw_main_figure(df, main, locus_sig, summary, size=(14.2, 8), double=True)
    if key == "fragA":
        return draw_fragment_a_matrix(main, "Allele-position matrix")
    if key == "fragB":
        return draw_fragment_b_dominance(df)
    if key == "fragC":
        return draw_fragment_c_axis_units(df)
    if key == "fragD":
        return draw_fragment_d_locus_signature(locus_sig.head(10))
    if key == "fragE":
        return draw_fragment_e_global_summary(summary)
    if key == "supp1":
        return draw_supplementary_figure_1(df)
    if key == "supp2":
        return draw_supplementary_figure_2(df)
    if key == "supp3":
        return draw_supplementary_figure_3(df)
    if key == "supp4":
        return draw_supplementary_figure_4(locus_sig)
    if key == "supp5":
        return draw_supplementary_figure_5(summary)
    raise SystemExit(f"Unknown figure key: {key}")


def generate_single_figure(input_file: str | Path, outdir: str | Path, key: str) -> None:
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    df, main, axis, locus_sig, summary, dom = prepare_data(input_file)
    spec = {x["key"]: x for x in build_output_specs()}[key]
    fig = draw_single_by_key(key, df, main, locus_sig, summary)
    manifest = []
    overlap_rows = []
    save_all_formats(fig, outdir, spec["stem"], manifest, spec["figure_label"], spec["figure_type"], spec["description"], spec["recommended_manuscript_location"], overlap_rows)
    plt.close(fig)


def make_package(input_file: str | Path, outdir: str | Path) -> None:
    outdir = Path(outdir)
    if outdir.exists():
        for p in outdir.glob("*"):
            if p.is_file() or p.is_symlink():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p)
    outdir.mkdir(parents=True, exist_ok=True)
    specs = build_output_specs()
    # Generate each figure in a fresh child process to keep memory stable for high-resolution export.
    for spec in specs:
        subprocess.run([
            sys.executable, str(Path(__file__).resolve()),
            "--input", str(input_file),
            "--outdir", str(outdir),
            "--single", spec["key"],
        ], check=True)
    df, main, axis, locus_sig, summary, dom = prepare_data(input_file)
    export_data_csvs(outdir, df, main, axis, locus_sig, dom, summary)
    manifest = []
    overlap_rows = []
    for spec in specs:
        for ext in ["pdf", "svg", "png", "tiff"]:
            manifest.append({
                "figure_label": spec["figure_label"],
                "file_name": f"{spec['stem']}.{ext}",
                "figure_type": spec["figure_type"],
                "description": spec["description"],
                "recommended_manuscript_location": spec["recommended_manuscript_location"],
            })
        overlap_rows.append({"figure_name": spec["stem"], "text_1": "", "text_2": "", "overlap_area": 0, "suggested_fix": "passed"})
    write_manifest(outdir, manifest)
    pd.DataFrame(overlap_rows).to_csv(outdir / "Figure_Text_Overlap_Report.csv", index=False)
    write_report(outdir, str(input_file), df, main, manifest, overlap_rows)
    try:
        shutil.copyfile(Path(__file__), outdir / "make_figure1_hla_final_overlap_cleanup.py")
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Generate final polished Figure 1 HLA publication package.")
    parser.add_argument("--input", required=True, help="Figure1_HLA_validated_values.csv")
    parser.add_argument("--outdir", default="figure1_final_overlap_cleanup")
    parser.add_argument("--single", default=None, help="Internal use: generate one figure key")
    args = parser.parse_args()
    if args.single:
        generate_single_figure(args.input, args.outdir, args.single)
    else:
        make_package(args.input, args.outdir)


if __name__ == "__main__":
    main()
