#!/usr/bin/env python3
"""Publication-quality reconstruction of merged Figure 7.

Panel A: CT effect-estimate bar chart
Panel B: Bradford Hill / GRADE radar comparison
Panel C: complete 19-row evidence-summary table

The source raster images are used only for visual QA. They are never embedded
in the publication outputs. All visible objects are created natively with
Matplotlib text, patches, lines, and vector gradient strips.
"""
from __future__ import annotations

import io
import math
import re
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

# -----------------------------------------------------------------------------
# Paths and technical settings
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR
ZIP_PATH = SCRIPT_DIR.parent / "Figure_7_publication_ready_package.zip"
TOP_REFERENCE = Path("/mnt/data/image.png")
COMBINED_REFERENCE = Path("/mnt/data/rendered_cc/slide-7.png")

DPI = 300
WIDTH_CM = 16.5
RASTER_W = 1949
RASTER_H = 1664
WIDTH_IN = RASTER_W / DPI
HEIGHT_IN = RASTER_H / DPI
PHYSICAL_W_CM = WIDTH_IN * 2.54
PHYSICAL_H_CM = HEIGHT_IN * 2.54
FIGSIZE = (WIDTH_IN, HEIGHT_IN)

FONT_FAMILY = "DejaVu Sans"

# Explicit fixed layout: do not use tight_layout/constrained_layout.
PANEL_A_RECT = [0.055, 0.715, 0.405, 0.235]
PANEL_B_RECT = [0.580, 0.760, 0.320, 0.190]
TABLE_RECT = [0.025, 0.030, 0.950, 0.620]

COLORS = {
    "blue": "#1769B7",
    "blue_mid": "#3D8BDA",
    "blue_light": "#7CB5EC",
    "red": "#C92D24",
    "red_mid": "#E1544A",
    "red_light": "#F58E87",
    "header": "#183E64",
    "header_mid": "#214D7A",
    "header_light": "#2D5D8D",
    "ct_band": "#1C5C99",
    "ct_band_light": "#3E7DB7",
    "hsv_band": "#A30000",
    "hsv_band_light": "#C43636",
    "ct_row_1": "#FFFFFF",
    "ct_row_2": "#EEF7FD",
    "hsv_row_1": "#FFFDFC",
    "hsv_row_2": "#FFF2F0",
    "grid": "#E5E8EC",
    "border": "#C7D0D8",
    "text": "#18232D",
    "yes": "#168B4B",
    "no": "#D6362C",
    "partial": "#F07F1A",
    "na": "#626A70",
}

# -----------------------------------------------------------------------------
# Structured scientific data
# -----------------------------------------------------------------------------

def build_panel_a_data() -> pd.DataFrame:
    rows = [
        ("Naldini 2019\n(any HPV\nin CT+)", 2.12, np.nan),
        ("Naldini 2019\n(HR-HPV\nin CT+)", 2.32, np.nan),
        ("Naldini 2019\n(CT in\nHPV+)", 2.23, np.nan),
        ("Zhu 2016\n(HPV-adj.\nOR)", 2.21, 1.76),
        ("Paavonen 2004\n(SCC; adj.)", 1.80, np.nan),
        ("Viikki 2000\n(fully adj)", np.nan, 1.10),
    ]
    return pd.DataFrame(rows, columns=["label", "primary", "adjusted"])


def build_panel_b_data() -> Dict[str, object]:
    criteria = [
        "Strength\nof association",
        "Consistency",
        "Temporality",
        "Dose-\nresponse",
        "Biological\nplausibility",
        "Experimental\nsupport",
    ]
    return {
        "criteria": criteria,
        "CT co-infection (GRADE: moderate)": [3, 3, 2, 2, 3, 2],
        "HSV-2 co-infection (GRADE: low)": [2, 1, 1, 1, 2, 1],
    }


def build_panel_c_data() -> Tuple[List[str], List[Dict[str, object]], List[Dict[str, object]]]:
    headers = ["STI", "Study\n(year)", "Design", "OR / aHR\n(95% CI)", "HPV\nadj?", "NOS", "Outcome / key note"]

    ct = [
        {"sti":"CT","study":"Zhu 2016","design":"SR/MA\n22 studies","effect":"4.03 (3.15-5.16)","hpv":"No","nos":"NA","note":"CT+HPV vs no infection; ↑ substantial"},
        {"sti":"CT","study":"Zhu 2016","design":"SR/MA\n(prospective)","effect":"2.21 (1.88-2.61)","hpv":"Partial","nos":"NA","note":"Prospective studies subgroup"},
        {"sti":"CT","study":"Zhu 2016","design":"SR/MA\n(HPV-adjusted)","effect":"1.76 (1.03-3.01)","hpv":"Yes","nos":"NA","note":"Indep. CT OR; lower CI approaches unity"},
        {"sti":"CT","study":"Naldini 2019","design":"SR/MA\n48 studies","effect":"HPV in CT+: 2.12 (1.80-2.49)\nHR-HPV: 2.32 (2.02-2.65)\nCT in HPV+: 2.23 (1.70-2.92)","hpv":"Partial","nos":"NA","note":"Bidirectional risk; HPV & CT are\nreciprocal risk factors; 48 studies [25]"},
        {"sti":"CT","study":"Bhuvanendran\nPillai 2022","design":"SR/MA","effect":"~2.10 (1.60-2.67)","hpv":"Partial","nos":"NA","note":"Cervical neoplasia; NAAT vs serology attenuation"},
        {"sti":"CT","study":"Paavonen 2004","design":"IARC pooled CC","effect":"SCC: 1.80 (1.20-2.70)","hpv":"Yes","nos":"8/9","note":"7 countries; n≈2,338; titre-dependent gradient"},
        {"sti":"CT","study":"Paavonen 2004","design":"IARC pooled CC","effect":"AdenoCA: 1.00 (0.53-1.90)","hpv":"Yes","nos":"8/9","note":"Null for AdenoCA - anatomical specificity"},
        {"sti":"CT","study":"Jensen 2014","design":"Prospective cohort","effect":"2.80 (1.30-6.10)","hpv":"Yes","nos":"8/9","note":"CIN3+; CT precedes CIN3 - temporality established"},
        {"sti":"CT","study":"Zaim 2023","design":"Cross-sectional","effect":"~2.80 (1.40-5.60)","hpv":"Partial","nos":"6/9","note":"Morocco (MENA); CT 18.3% vs 7.6% in HPV+ groups"},
        {"sti":"CT","study":"Fernandez-Perez\n2024","design":"Prospective cohort","effect":"1.37 (0.88-2.12)","hpv":"Yes","nos":"8/9","note":"Spain; HGSIL risk elevation; n=254 HPV+ve"},
        {"sti":"CT","study":"Samoff 2013","design":"Prospective cohort","effect":"HPV redetection ↑","hpv":"Yes","nos":"7/9","note":"CT impairs durable HPV immune control (redetection)"},
        {"sti":"CT","study":"Viikki 2000","design":"Case-control","effect":"~1.10 (0.58-2.08) null","hpv":"Yes","nos":"7/9","note":"No CT association after full sexual-behaviour adj."},
    ]

    hsv = [
        {"sti":"HSV-2","study":"Zhang 2023","design":"SR/MA\n67 studies","effect":"HHVs CC: 2.74 (2.13-3.53)\nHSV-2 CC: 3.01 (2.24-4.04)\nHSV-2 PCL: 2.14 (1.55-2.96)","hpv":"No (HPV-unadj)","nos":"NA","note":"67 studies, 7 HHV types; UNADJUSTED\nfor HPV; GRADE: low maintained [60]"},
        {"sti":"HSV-2","study":"Smith 2002","design":"IARC pooled CC","effect":"SCC: 2.20 (1.40-3.40)","hpv":"8/9","nos":"8/9","note":"11 countries; n≈3,608; specific to SCC not AdenoCA"},
        {"sti":"HSV-2","study":"Smith 2002","design":"IARC pooled CC","effect":"AdenoCA: 1.80 (0.90-3.80)","hpv":"Partial","nos":"8/9","note":"Non-significant for adenocarcinoma"},
        {"sti":"HSV-2","study":"Lehtinen 2002","design":"Nested CC\n(Nordic)","effect":"1.40 (0.90-2.20)","hpv":"Yes","nos":"7/9","note":"Non-significant after HPV adjustment; n=530 cases"},
        {"sti":"HSV-2","study":"Seidman 2023","design":"Prospective cohort","effect":"aHR 1.80 (1.10-3.00)","hpv":"Yes","nos":"8/9","note":"HPV-vaccinated WLHIV; HSV-2 → HPV\nacquisition, not precancer"},
        {"sti":"HSV-2","study":"Seidman 2023","design":"Prospective cohort","effect":"aHR 1.60 (1.00-2.50)","hpv":"Yes","nos":"8/9","note":"Persistence borderline; no significant precancer assoc."},
        {"sti":"HSV-2","study":"Tran-Thanh 2003","design":"Molecular study\n(PCR)","effect":"No HSV-2 DNA detected","hpv":"N/A","nos":"N/A","note":"200 CC specimens; direct evidence\nagainst persistence mechanism"},
    ]
    return headers, ct, hsv


def validate_scientific_content() -> None:
    a = build_panel_a_data()
    expected_primary = [2.12, 2.32, 2.23, 2.21, 1.80]
    actual_primary = [float(v) for v in a["primary"].dropna().tolist()]
    assert actual_primary == expected_primary
    assert [float(v) for v in a["adjusted"].dropna().tolist()] == [1.76, 1.10]
    b = build_panel_b_data()
    assert len(b["criteria"]) == 6
    assert b["CT co-infection (GRADE: moderate)"] == [3,3,2,2,3,2]
    assert b["HSV-2 co-infection (GRADE: low)"] == [2,1,1,1,2,1]
    headers, ct, hsv = build_panel_c_data()
    assert len(headers) == 7
    assert len(ct) == 12
    assert len(hsv) == 7
    assert len(ct) + len(hsv) == 19

# -----------------------------------------------------------------------------
# Matplotlib / gradient helpers
# -----------------------------------------------------------------------------

def configure_matplotlib() -> None:
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [FONT_FAMILY],
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.transparent": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.unicode_minus": False,
        "path.simplify": False,
    })


def gradient_colors(stops: Sequence[str], n: int) -> np.ndarray:
    rgb = np.array([to_rgb(c) for c in stops], dtype=float)
    t = np.linspace(0, 1, n)
    if len(rgb) == 2:
        return np.array([rgb[0]*(1-x)+rgb[1]*x for x in t])
    out = np.empty((n,3))
    for i,x in enumerate(t):
        if x <= 0.5:
            u=x/0.5; out[i]=rgb[0]*(1-u)+rgb[1]*u
        else:
            u=(x-0.5)/0.5; out[i]=rgb[1]*(1-u)+rgb[2]*u
    return out


def draw_gradient_rect(ax: plt.Axes, x: float, y: float, w: float, h: float,
                       stops: Sequence[str], edge: str | None = None,
                       lw: float = 0.6, segments: int = 16, zorder: float = 0.5) -> Rectangle:
    colors = gradient_colors(stops, segments)
    sw = w / segments
    for i,c in enumerate(colors):
        ax.add_patch(Rectangle((x+i*sw, y), sw*1.04, h,
                               facecolor=c, edgecolor="none", linewidth=0,
                               antialiased=False, zorder=zorder))
    border = Rectangle((x,y),w,h,facecolor="none",edgecolor=edge or "none",linewidth=lw,zorder=zorder+0.2)
    ax.add_patch(border)
    return border


def draw_gradient_bar(ax: plt.Axes, x_center: float, height: float, width: float,
                      stops: Sequence[str], edge: str, segments: int = 16) -> None:
    x0 = x_center - width/2
    colors = gradient_colors(stops, segments)
    sw = width/segments
    for i,c in enumerate(colors):
        ax.add_patch(Rectangle((x0+i*sw,0),sw*1.05,height,facecolor=c,edgecolor="none",linewidth=0,zorder=3))
    ax.add_patch(Rectangle((x0,0),width,height,facecolor="none",edgecolor=edge,linewidth=0.45,zorder=3.2))


def fig_bbox_from_ax_bounds(ax: plt.Axes, bounds: Tuple[float,float,float,float]) -> Bbox:
    x,y,w,h=bounds
    p0=ax.transAxes.transform((x,y)); p1=ax.transAxes.transform((x+w,y+h))
    return Bbox.from_extents(p0[0],p0[1],p1[0],p1[1])

# -----------------------------------------------------------------------------
# Text fitting and registry
# -----------------------------------------------------------------------------

def _split_lines(value: str) -> List[str]:
    return str(value).split("\n")


def draw_cell_text(fig: plt.Figure, ax: plt.Axes, bounds: Tuple[float,float,float,float], value: str,
                   fontsize: float, color: str = COLORS["text"], weight: str = "normal",
                   ha: str = "center", italic: bool = False, pad_px: float = 3.0,
                   line_gap_px: float = 1.0, registry: List[dict] | None = None,
                   name: str = "cell_text") -> List[plt.Text]:
    x,y,w,h=bounds
    lines=_split_lines(value)
    artists: List[plt.Text]=[]
    # draw provisional lines
    for line in lines:
        t=ax.text(x+w/2 if ha=="center" else x+0.012*w, y+h/2, line,
                  transform=ax.transAxes, ha=ha, va="center", fontsize=fontsize,
                  color=color, fontweight=weight, fontstyle="italic" if italic else "normal", zorder=5)
        artists.append(t)
    renderer=fig.canvas.get_renderer()
    parent=fig_bbox_from_ax_bounds(ax,bounds)
    current_fs=fontsize
    while True:
        for t in artists: t.set_fontsize(current_fs)
        bbs=[t.get_window_extent(renderer) for t in artists]
        heights=[b.height for b in bbs]
        total=sum(heights)+line_gap_px*(len(lines)-1)
        maxw=max((b.width for b in bbs),default=0)
        if total <= parent.height-2*pad_px and maxw <= parent.width-2*pad_px:
            break
        current_fs -= 0.12
        if current_fs < 2.8:
            raise RuntimeError(f"Unable to fit text block {name}: {value}")
    # exact vertical centring in display coordinates, then transform back to axes coords
    top=parent.y0+(parent.height+total)/2
    cy=top
    for i,(t,bh) in enumerate(zip(artists,heights)):
        center_y=cy-bh/2
        x_disp=(parent.x0+parent.x1)/2 if ha=="center" else parent.x0+pad_px+1
        x_ax,y_ax=ax.transAxes.inverted().transform((x_disp,center_y))
        t.set_position((x_ax,y_ax))
        cy-=bh+line_gap_px
    if registry is not None:
        for i,t in enumerate(artists):
            registry.append({"name":f"{name}_{i}","artist":t,"parent":parent,"kind":"table"})
    return artists

# -----------------------------------------------------------------------------
# Panel A
# -----------------------------------------------------------------------------

def draw_panel_a(fig: plt.Figure, registry: List[dict]) -> plt.Axes:
    ax=fig.add_axes(PANEL_A_RECT)
    df=build_panel_a_data()
    x=np.arange(len(df))
    ax.set_xlim(-0.6,len(df)-0.4)
    ax.set_ylim(0,3.0)
    ax.grid(axis="y",color=COLORS["grid"],linewidth=0.55,zorder=0)
    ax.axhline(1.0,color="#C9CDD1",linestyle=(0,(3,2)),linewidth=0.8,zorder=1)
    ax.set_ylabel("Odds Ratio",fontsize=6.6,fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(df["label"],fontsize=4.5,linespacing=0.88)
    ax.set_yticks(np.arange(0,3.1,0.5))
    ax.tick_params(axis="y",labelsize=5.2)
    ax.tick_params(axis="x",length=2,pad=2)
    for s in ax.spines.values(): s.set_linewidth(0.6); s.set_color("#33383D")
    width=0.34
    for i,row in df.iterrows():
        if not pd.isna(row.primary):
            xc=x[i]-0.18 if not pd.isna(row.adjusted) else x[i]
            draw_gradient_bar(ax,xc,float(row.primary),width,(COLORS["blue_light"],COLORS["blue_mid"],COLORS["blue"]),"#15579D")
            t=ax.text(xc,float(row.primary)+0.055,f"{row.primary:.2f}",ha="center",va="bottom",fontsize=5.0,fontweight="bold",color=COLORS["blue"])
            registry.append({"name":f"bar_primary_{i}","artist":t,"parent":ax.get_window_extent(fig.canvas.get_renderer()),"kind":"chart"})
        if not pd.isna(row.adjusted):
            xc=x[i]+0.18 if not pd.isna(row.primary) else x[i]
            draw_gradient_bar(ax,xc,float(row.adjusted),width,(COLORS["red_light"],COLORS["red_mid"],COLORS["red"]),"#A9231E")
            t=ax.text(xc,float(row.adjusted)+0.055,f"{row.adjusted:.2f}",ha="center",va="bottom",fontsize=5.0,fontweight="bold",color=COLORS["red"])
            registry.append({"name":f"bar_adjusted_{i}","artist":t,"parent":ax.get_window_extent(fig.canvas.get_renderer()),"kind":"chart"})

    ax.text(0.0,1.095,"CT estimates — Naldini 2019 corrected",transform=ax.transAxes,ha="left",va="bottom",fontsize=7.2,fontweight="bold")
    ax.text(0.0,1.045,"(bidirectional: HPV/CT and CT/HPV)",transform=ax.transAxes,ha="left",va="bottom",fontsize=5.7,fontweight="bold")
    ax.text(-0.08,1.11,"A.",transform=ax.transAxes,ha="left",va="bottom",fontsize=8.2,fontweight="bold")

    # custom legend
    handles=[Rectangle((0,0),1,1,facecolor=COLORS["blue"],edgecolor="#15579D",linewidth=0.4),
             Rectangle((0,0),1,1,facecolor=COLORS["red"],edgecolor="#A9231E",linewidth=0.4)]
    leg=ax.legend(handles,["Unadjusted / primary OR","HPV-adjusted OR"],loc="upper right",bbox_to_anchor=(1.0,1.08),fontsize=4.8,frameon=True,borderpad=0.4,handlelength=1.5)
    leg.get_frame().set_edgecolor("#D2D6DA"); leg.get_frame().set_linewidth(0.5); leg.get_frame().set_facecolor("white")
    return ax

# -----------------------------------------------------------------------------
# Panel B
# -----------------------------------------------------------------------------

def draw_panel_b(fig: plt.Figure, registry: List[dict]) -> plt.Axes:
    data=build_panel_b_data(); criteria=data["criteria"]
    n=len(criteria)
    angles=np.linspace(0,2*np.pi,n,endpoint=False)
    angles=np.concatenate([angles,[angles[0]]])
    ax=fig.add_axes(PANEL_B_RECT,projection="polar")
    ax.set_theta_offset(np.pi/2); ax.set_theta_direction(-1)
    ax.set_ylim(0,5)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(criteria,fontsize=5.3)
    ax.set_yticks([1,2,3,4,5]); ax.set_yticklabels(["1","2","3","4","5"],fontsize=4.5,color="#7D858C")
    ax.set_rlabel_position(15)
    ax.grid(color="#D7DBDF",linewidth=0.55)
    ax.spines["polar"].set_color("#AEB5BB"); ax.spines["polar"].set_linewidth(0.7)
    ct=np.array(data["CT co-infection (GRADE: moderate)"]+[data["CT co-infection (GRADE: moderate)"][0]])
    hsv=np.array(data["HSV-2 co-infection (GRADE: low)"]+[data["HSV-2 co-infection (GRADE: low)"][0]])
    ax.plot(angles,ct,color="#2D7ED8",linewidth=1.35,marker="o",markersize=2.6,zorder=3)
    ax.fill(angles,ct,color="#D6E9FA",alpha=0.82,zorder=2)
    ax.plot(angles,hsv,color="#D7362D",linewidth=1.35,marker="o",markersize=2.6,zorder=3)
    ax.fill(angles,hsv,color="#F6D3CF",alpha=0.82,zorder=2)
    ax.text(-0.16,1.08,"B.",transform=ax.transAxes,ha="left",va="bottom",fontsize=8.2,fontweight="bold")
    handles=[plt.Line2D([0],[0],color="#2D7ED8",lw=1.4),plt.Line2D([0],[0],color="#D7362D",lw=1.4)]
    leg=fig.legend(handles,["CT co-infection (GRADE: moderate)","HSV-2 co-infection (GRADE: low)"],
                   loc="center",bbox_to_anchor=(0.765,0.680),fontsize=4.6,frameon=True,ncol=1,borderpad=0.35)
    leg.get_frame().set_edgecolor("#D4D8DC"); leg.get_frame().set_linewidth(0.5); leg.get_frame().set_facecolor("white")
    return ax

# -----------------------------------------------------------------------------
# Panel C
# -----------------------------------------------------------------------------

def hpv_color(value: str) -> str:
    if value == "Yes": return COLORS["yes"]
    if value.startswith("No"): return COLORS["no"]
    if value == "Partial": return COLORS["partial"]
    return COLORS["na"]


def draw_panel_c(fig: plt.Figure, registry: List[dict]) -> plt.Axes:
    headers,ct,hsv=build_panel_c_data()
    ax=fig.add_axes(TABLE_RECT)
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
    fig.canvas.draw()
    ax.text(-0.010,1.013,"C.",transform=ax.transAxes,ha="left",va="bottom",fontsize=8.2,fontweight="bold")

    col_widths=np.array([0.055,0.130,0.140,0.240,0.080,0.060,0.295])
    x_edges=np.concatenate([[0],np.cumsum(col_widths)])

    # Relative row units: header, section bands, and body rows with extra room for 3-line rows.
    heights=[1.18,0.72]
    for r in ct:
        maxlines=max(len(_split_lines(str(v))) for v in r.values())
        heights.append(1.18 if maxlines<=1 else (1.43 if maxlines==2 else 1.90))
    heights.append(0.72)
    for r in hsv:
        maxlines=max(len(_split_lines(str(v))) for v in r.values())
        heights.append(1.18 if maxlines<=1 else (1.43 if maxlines==2 else 1.90))
    heights=np.array(heights,dtype=float)
    heights=heights/heights.sum()
    y_top=1.0
    row_index=0

    # header
    h=heights[row_index]; y=y_top-h
    for j,head in enumerate(headers):
        bounds=(x_edges[j],y,col_widths[j],h)
        draw_gradient_rect(ax,*bounds,(COLORS["header_light"],COLORS["header_mid"],COLORS["header"]),edge="#12334F",lw=0.6,segments=12)
        draw_cell_text(fig,ax,bounds,head,fontsize=5.2,color="white",weight="bold",pad_px=3,line_gap_px=0.5,registry=registry,name=f"header_{j}")
    y_top=y; row_index+=1

    # section helper
    def section_band(text: str, light: str, dark: str, idx: int) -> None:
        nonlocal y_top,row_index
        h=heights[row_index]; y=y_top-h
        bounds=(0,y,1,h)
        draw_gradient_rect(ax,*bounds,(light,dark),edge=dark,lw=0.4,segments=12)
        draw_cell_text(fig,ax,bounds,text,fontsize=4.9,color="white",weight="bold",pad_px=2,line_gap_px=0,registry=registry,name=f"section_{idx}")
        y_top=y; row_index+=1

    section_band("CT–HPV Co-infection Studies (Section 3)",COLORS["ct_band_light"],COLORS["ct_band"],0)

    keys=["sti","study","design","effect","hpv","nos","note"]
    def body_rows(rows: List[Dict[str,object]], group: str) -> None:
        nonlocal y_top,row_index
        for i,r in enumerate(rows):
            h=heights[row_index]; y=y_top-h
            base=(COLORS["ct_row_1"],COLORS["ct_row_2"]) if group=="ct" else (COLORS["hsv_row_1"],COLORS["hsv_row_2"])
            fill=base[i%2]
            for j,key in enumerate(keys):
                bounds=(x_edges[j],y,col_widths[j],h)
                # very subtle horizontal premium gradient
                stops=("#FFFFFF",fill) if i%2==0 else (fill,"#FFFFFF")
                draw_gradient_rect(ax,*bounds,stops,edge=COLORS["border"],lw=0.38,segments=6,zorder=0.3)
                val=str(r[key])
                color=hpv_color(val) if key=="hpv" else (COLORS["blue"] if key=="sti" and group=="ct" else (COLORS["red"] if key=="sti" else COLORS["text"]))
                weight="bold" if key in {"sti","hpv"} else "normal"
                ha="left" if key=="note" else "center"
                fs=4.15 if key!="note" else 3.95
                if key=="effect" and len(_split_lines(val))>=3: fs=3.65
                if key=="note" and len(_split_lines(val))>=2: fs=3.75
                draw_cell_text(fig,ax,bounds,val,fontsize=fs,color=color,weight=weight,ha=ha,pad_px=3.0,line_gap_px=0.8,registry=registry,name=f"{group}_{i}_{key}")
            y_top=y; row_index+=1

    body_rows(ct,"ct")
    section_band("HSV-2–HPV Co-infection Studies (Section 4)",COLORS["hsv_band_light"],COLORS["hsv_band"],1)
    body_rows(hsv,"hsv")
    return ax

# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------

def validate_text_containment(fig: plt.Figure, registry: List[dict]) -> Dict[str,object]:
    fig.canvas.draw(); renderer=fig.canvas.get_renderer(); canvas=fig.bbox
    failures=[]; clearances=[]; boxes=[]
    for rec in registry:
        bb=rec["artist"].get_window_extent(renderer)
        parent=rec["parent"]
        boxes.append((rec["name"],bb,rec["kind"]))
        if bb.x0 < canvas.x0-0.5 or bb.y0 < canvas.y0-0.5 or bb.x1 > canvas.x1+0.5 or bb.y1 > canvas.y1+0.5:
            failures.append(f"canvas overflow: {rec['name']}")
        if rec["kind"]=="table":
            clearance=min(bb.x0-parent.x0,parent.x1-bb.x1,bb.y0-parent.y0,parent.y1-bb.y1)
            clearances.append(clearance)
            if clearance < 1.5:
                failures.append(f"table containment: {rec['name']} clearance={clearance:.2f}px")
    return {"failures":failures,"boxes":boxes,"min_clearance":min(clearances) if clearances else float('nan')}

# -----------------------------------------------------------------------------
# Figure creation / exports
# -----------------------------------------------------------------------------

def create_figure() -> Tuple[plt.Figure,List[dict],Dict[str,object]]:
    fig=plt.figure(figsize=FIGSIZE,dpi=DPI,facecolor="white",layout=None)
    registry: List[dict]=[]
    a=draw_panel_a(fig,registry)
    b=draw_panel_b(fig,registry)
    c=draw_panel_c(fig,registry)
    fig.canvas.draw()
    val=validate_text_containment(fig,registry)
    if val["failures"]:
        raise RuntimeError("Containment validation failed: "+"; ".join(val["failures"][:20]))
    return fig,registry,val


def export_rasters() -> Tuple[Path,Path,Path,Dict[str,object]]:
    fig,registry,val=create_figure()
    canvas=matplotlib.backends.backend_agg.FigureCanvasAgg(fig); canvas.draw()
    rgba=np.asarray(canvas.buffer_rgba())
    image=Image.fromarray(rgba,"RGBA").convert("RGB")
    if image.size!=(RASTER_W,RASTER_H): raise RuntimeError(f"Unexpected raster size {image.size}")
    png=OUT_DIR/"Figure_7_merged_evidence_panels.png"
    jpg=OUT_DIR/"Figure_7_merged_evidence_panels.jpg"
    tif=OUT_DIR/"Figure_7_merged_evidence_panels.tiff"
    image.save(png,"PNG",dpi=(DPI,DPI),optimize=True)
    image.save(jpg,"JPEG",dpi=(DPI,DPI),quality=100,subsampling=0,optimize=True)
    image.save(tif,"TIFF",dpi=(DPI,DPI),compression="tiff_lzw")
    plt.close(fig)
    return png,jpg,tif,val


def export_vectors() -> Tuple[Path,Path]:
    fig,registry,val=create_figure()
    pdf=OUT_DIR/"Figure_7_merged_evidence_panels.pdf"
    svg=OUT_DIR/"Figure_7_merged_evidence_panels.svg"
    fig.savefig(pdf,format="pdf",facecolor="white",transparent=False)
    fig.savefig(svg,format="svg",facecolor="white",transparent=False)
    plt.close(fig)
    return pdf,svg

# -----------------------------------------------------------------------------
# QA
# -----------------------------------------------------------------------------

def inspect_raster(path: Path) -> Dict[str,object]:
    with Image.open(path) as im:
        comp=im.info.get("compression")
        if path.suffix.lower() in {".tif",".tiff"}: comp=im.tag_v2.get(259,comp)
        return {"file":path.name,"format":im.format,"size":im.size,"mode":im.mode,"dpi":im.info.get("dpi"),"compression":comp,"frames":getattr(im,"n_frames",1),"bytes":path.stat().st_size,"transparency":("A" in im.getbands() or "transparency" in im.info)}


def inspect_vectors(pdf: Path,svg: Path) -> Dict[str,object]:
    out={}
    try:
        import fitz
        doc=fitz.open(pdf); page=doc[0]
        out["pdf_page_points"]=(page.rect.width,page.rect.height)
        out["pdf_image_count"]=len(page.get_images(full=True))
        out["pdf_page_count"]=doc.page_count
        doc.close()
    except Exception as e: out["pdf_error"]=str(e)
    text=svg.read_text(encoding="utf-8",errors="replace")
    out["svg_has_image_tag"]=bool(re.search(r"<image\b",text,re.I))
    out["svg_rect_count"]=len(re.findall(r"<rect\b",text,re.I))
    out["svg_path_count"]=len(re.findall(r"<path\b",text,re.I))
    mw=re.search(r'<svg[^>]+width="([^"]+)"',text,re.I); mh=re.search(r'<svg[^>]+height="([^"]+)"',text,re.I)
    out["svg_width"]=mw.group(1) if mw else "not found"; out["svg_height"]=mh.group(1) if mh else "not found"
    return out


def build_combined_reference() -> Image.Image:
    # The available source slide contains both chart references and the complete table.
    with Image.open(COMBINED_REFERENCE) as im:
        return im.convert("RGB")


def qa_font(size:int,bold:bool=False):
    try:
        from matplotlib import font_manager
        path=font_manager.findfont(font_manager.FontProperties(family=FONT_FAMILY,weight="bold" if bold else "normal"))
        return ImageFont.truetype(path,size)
    except Exception: return ImageFont.load_default()


def create_qa_files(png: Path,val: Dict[str,object]) -> Tuple[Path,...]:
    with Image.open(png) as im: rec=im.convert("RGB")
    ref=build_combined_reference()
    ref_scaled=ref.resize(rec.size,Image.Resampling.LANCZOS)
    overlay=OUT_DIR/"Figure_7_QA_overlay.png"; Image.blend(ref_scaled,rec,0.5).save(overlay,dpi=(DPI,DPI))

    disp_w=1000; ref_disp=ref.resize((disp_w,round(disp_w/ref.width*ref.height)),Image.Resampling.LANCZOS); rec_disp=rec.resize((disp_w,round(disp_w/rec.width*rec.height)),Image.Resampling.LANCZOS)
    header=52; gap=20; H=max(ref_disp.height,rec_disp.height)
    side=Image.new("RGB",(disp_w*2+gap,H+header),"white"); side.paste(ref_disp,(0,header)); side.paste(rec_disp,(disp_w+gap,header))
    d=ImageDraw.Draw(side); f=qa_font(24,True); d.text((disp_w//2,20),"Source references combined",font=f,fill="black",anchor="mm"); d.text((disp_w+gap+disp_w//2,20),"Reconstructed Figure 7",font=f,fill="black",anchor="mm")
    side_path=OUT_DIR/"Figure_7_QA_side_by_side.png"; side.save(side_path,dpi=(DPI,DPI))

    bounds=rec.copy(); db=ImageDraw.Draw(bounds)
    for name,bb,kind in val["boxes"]:
        box=(int(bb.x0),int(RASTER_H-bb.y1),int(bb.x1),int(RASTER_H-bb.y0))
        db.rectangle(box,outline=(220,0,180),width=1)
    text_bounds=OUT_DIR/"Figure_7_QA_text_bounds.png"; bounds.save(text_bounds,dpi=(DPI,DPI))

    # closeups
    a_crop=rec.crop((int(0.02*RASTER_W),int(0.02*RASTER_H),int(0.50*RASTER_W),int(0.34*RASTER_H))).resize((1500,1000),Image.Resampling.LANCZOS)
    a_path=OUT_DIR/"Figure_7_QA_panel_A_closeup.png"; a_crop.save(a_path,dpi=(DPI,DPI))
    b_crop=rec.crop((int(0.50*RASTER_W),int(0.02*RASTER_H),int(0.98*RASTER_W),int(0.35*RASTER_H))).resize((1500,1000),Image.Resampling.LANCZOS)
    b_path=OUT_DIR/"Figure_7_QA_panel_B_closeup.png"; b_crop.save(b_path,dpi=(DPI,DPI))
    c_crop=rec.crop((int(0.015*RASTER_W),int(0.33*RASTER_H),int(0.985*RASTER_W),int(0.985*RASTER_H))).resize((1900,1300),Image.Resampling.LANCZOS)
    c_path=OUT_DIR/"Figure_7_QA_panel_C_closeup.png"; c_crop.save(c_path,dpi=(DPI,DPI))
    return overlay,side_path,text_bounds,a_path,b_path,c_path


def write_report(rasters: Sequence[Path],pdf: Path,svg: Path,val: Dict[str,object],vector: Dict[str,object]) -> Path:
    report=OUT_DIR/"Figure_7_QA_report.txt"
    top_info=(None,None); combined_info=(None,None)
    with Image.open(TOP_REFERENCE) as im: top_info=(im.size,im.mode)
    with Image.open(COMBINED_REFERENCE) as im: combined_info=(im.size,im.mode)
    lines=["FIGURE 7 PUBLICATION-READY QA REPORT","="*52,"",f"Mounted top reference: {TOP_REFERENCE}",f"Mounted top reference dimensions/mode: {top_info[0][0]} x {top_info[0][1]} / {top_info[1]}",f"Combined source-slide reference: {COMBINED_REFERENCE}",f"Combined source-slide dimensions/mode: {combined_info[0][0]} x {combined_info[0][1]} / {combined_info[1]}","Source images used only for QA; none are embedded in final outputs.","",f"Final physical dimensions: {PHYSICAL_W_CM:.3f} x {PHYSICAL_H_CM:.3f} cm",f"Final raster dimensions: {RASTER_W} x {RASTER_H} px",f"DPI: {DPI} x {DPI}","","RASTER OUTPUTS","-"*52]
    for p in rasters:
        q=inspect_raster(p); lines += [f"File: {q['file']}",f"  Format: {q['format']}",f"  Dimensions: {q['size'][0]} x {q['size'][1]}",f"  Mode: {q['mode']}",f"  DPI metadata: {q['dpi']}",f"  Transparency: {q['transparency']}",f"  File size: {q['bytes']} bytes"]
        if q['format']=='TIFF': lines += [f"  TIFF compression tag: {q['compression']} (5 = LZW)",f"  Frames: {q['frames']}"]
        if q['format']=='JPEG': lines += ["  JPEG quality: 100","  JPEG subsampling: 0 (4:4:4)"]
    lines += ["","VECTOR OUTPUTS","-"*52]
    for k,v in vector.items(): lines.append(f"{k}: {v}")
    lines += ["","CONTENT VALIDATION","-"*52,"Panels recreated: A, B, and C","Panel A visible bar values: 7","Panel A values validated: 2.12, 2.32, 2.23, 2.21, 1.76, 1.80, 1.10","Panel B radar axes: 6","Panel B CT scores: 3, 3, 2, 2, 3, 2","Panel B HSV-2 scores: 2, 1, 1, 1, 2, 1","Panel C columns: 7","Panel C CT study rows: 12","Panel C HSV-2 study rows: 7","Panel C total study rows: 19","Panel C group bands: 2","Text containment failures: "+str(len(val['failures'])),f"Minimum table text-to-border clearance: {val['min_clearance']:.2f} px","Source images embedded: no","Gradient implementation: native vector strips for bars and table cells; vector fills for radar polygons.","","INTENTIONAL DIFFERENCES","-"*52,"Subtle premium gradients, sharper vector typography, and corrected crowding are intentional.","The combined figure uses a balanced single-canvas layout rather than pasted screenshots.","Minor residual differences are limited to font metrics, antialiasing, and the cleaner merged-panel spacing."]
    report.write_text("\n".join(lines)+"\n",encoding="utf-8")
    return report


def create_zip(paths: Sequence[Path]) -> Path:
    with zipfile.ZipFile(ZIP_PATH,"w",zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in paths: z.write(p,arcname=p.name)
    return ZIP_PATH


def main() -> None:
    configure_matplotlib(); validate_scientific_content()
    png,jpg,tif,val=export_rasters()
    pdf,svg=export_vectors()
    vector=inspect_vectors(pdf,svg)
    qa=create_qa_files(png,val)
    report=write_report([png,jpg,tif],pdf,svg,val,vector)
    deliverables=[Path(__file__).resolve(),png,jpg,tif,pdf,svg,*qa,report]
    package=create_zip(deliverables)
    print(f"Generated {len(deliverables)} deliverables")
    print(package)
    print(f"Min table clearance: {val['min_clearance']:.2f}px")

if __name__=="__main__":
    main()
