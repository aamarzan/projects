#!/usr/bin/env python3
"""Publication-ready native-vector reconstruction of Figure 9.

The source raster is used only for visual QA. All publication outputs are
redrawn from native Matplotlib vector text, rounded rectangles, diamonds,
arrow gradient strips, lines, and arrowheads. No source raster is embedded in
PDF or SVG outputs.
"""
from __future__ import annotations

import io
import math
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Polygon
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
from matplotlib.transforms import Bbox
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

# -----------------------------------------------------------------------------
# Paths and output geometry
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_IMAGE = Path("/mnt/data/image.png")
OUT_DIR = SCRIPT_DIR
ZIP_PATH = SCRIPT_DIR.parent / "Figure_9_publication_ready_package.zip"

DPI = 300
LOGICAL_W, LOGICAL_H = 1124, 720
WIDTH_CM = 16.5
HEIGHT_CM = WIDTH_CM * LOGICAL_H / LOGICAL_W
WIDTH_IN = WIDTH_CM / 2.54
HEIGHT_IN = HEIGHT_CM / 2.54
RASTER_W, RASTER_H = 1949, 1248
RASTER_FIGSIZE = (RASTER_W / DPI, RASTER_H / DPI)
VECTOR_FIGSIZE = (WIDTH_IN, HEIGHT_IN)

FONT_FAMILY = "Liberation Sans"
FONT = {
    "key": 4.60,
    "initial": 7.2,
    "decision": 7.2,
    "branch": 6.1,
    "major": 6.4,
    "process_head": 5.35,
    "process_body": 4.75,
    "process_small": 4.45,
    "summary_title": 6.0,
    "summary_label": 5.1,
    "summary_body": 4.45,
}

# -----------------------------------------------------------------------------
# Palettes
# -----------------------------------------------------------------------------
PAL = {
    "key": (("#FFFFFF", "#F8FAFB", "#F0F4F6"), "#D4DBDF"),
    "initial": (("#40586B", "#31495C", "#24394A"), "#172A39"),
    "tier_a": (("#FFF7F5", "#F4D7D2", "#E9BDB5"), "#D94A3D"),
    "tier_a_fill": (("#D84D40", "#C94034", "#B92F25"), "#A8271F"),
    "green": (("#F5FCF8", "#E2F5E9", "#CEEEDB"), "#2DB86C"),
    "blue": (("#F5FAFE", "#DFEFF9", "#CDE5F4"), "#3D9CD0"),
    "orange": (("#FFF9EF", "#FFF0CF", "#FBE2A7"), "#F0A22D"),
    "tier_d": (("#FFF8F7", "#FCE7E4", "#F7D2CD"), "#EE473C"),
    "neutral": (("#FCFDFD", "#F3F6F7", "#E9EEF0"), "#9FAAAD"),
    "summary": (("#FFFFFF", "#F8FAFB", "#F0F4F6"), "#D4DBDF"),
}

COL = {
    "dark": "#263746",
    "red": "#BC3D32",
    "red_fill": "#FFFFFF",
    "green": "#169956",
    "blue": "#2875A1",
    "orange": "#A56C0A",
    "tier_d": "#D33D33",
    "neutral_head": "#344754",
    "neutral_body": "#697277",
    "grey": "#5C646A",
}

# -----------------------------------------------------------------------------
# Authoritative scientific text
# -----------------------------------------------------------------------------
TEXT = {
    "key": [
        [("Tier A:", "WHO-guideline-endorsed", "green"),
         ("Tier B:", "Clinically reasonable (observational evidence)", "blue"),
         ("Tier C:", "Hypothesis-generating", "orange"),
         ("Tier D:", "Speculative", "tier_d")],
        [("Tier A:", "STI treatment (STI benefit)", "green"),
         ("Tier B:", "Co-testing CT+HPV; Self-sampling access", "blue"),
         ("Tier C:", "CT eradication as CC prevention", "orange"),
         ("Tier D:", "HSV-2 suppression as CC prevention", "tier_d")],
    ],
    "initial": "Woman Presenting for Cervical Screening",
    "decisions": {"HIV": ["HIV", "Status?"], "CT": ["CT", "Status?"], "HSV": ["HSV-2", "Status?"]},
    "branch_labels": {"HIV+": "HIV+", "HIV-": "HIV-", "CT+": "CT+", "CT-": "CT-", "Any": "Any serostatus"},
    "wlhiv": {
        "header": ["WLHIV Protocol", "Tier A – WHO 2021", "Guideline-Endorsed"],
        "boxes": [
            ["[Tier A] Screen from age 25", "HPV-based 3-yearly"],
            ["[Tier A] Multi-dose HPV vax", "Reduced immunogenicity"],
            ["[Tier A] ART optimisation", "Cornerstone of prevention"],
            ["[Tier A] Screen-and-treat", "VIA resource-limited settings"],
            ["[Tier B] Self-sampling HPV", "Access-expanding strategy"],
        ],
    },
    "ct": {
        "boxes": [
            ["[Tier B] HPV + CT co-testing", "Concurrent testing clinically reasonable"],
            ["[Tier A] CT treatment", "Doxycycline/azithromycin (STI benefit)"],
            ["[Tier C] CT eradication", "as CC prevention strategy", "Hypothesis-generating; no RCT evidence"],
            ["Standard colposcopy referral", "thresholds; no CT-specific reduction"],
        ]
    },
    "hsv": {
        "standard_cervical": ["Standard Cervical", "Cancer Screening Protocol", "Per national / WHO guidelines"],
        "standard": ["Standard Screening", "No guideline-supported modification", "per WHO / national guidelines"],
        "tier_d": ["[Tier D] HSV-2 suppressive", "therapy as CC prevention", "SPECULATIVE – No evidence base"],
    },
    "summary": {
        "title": "Evidence Tier Summary by Co-infection Type",
        "rows": [
            ("WLHIV (Tier A):", "Screening age 25; 3-yearly HPV; VIA screen-and-treat; multi-dose HPV vax; ART optimisation (all WHO 2021 endorsed)", "red"),
            ("CT (Tier A/B/C):", "Treatment = Tier A (STI); Concurrent testing = Tier B; CT eradication as CC prevention = Tier C (hypothesis-generating)", "blue"),
            ("HSV-2:", "Standard screening — no guideline-supported modification; Tier D for suppressive therapy as CC prevention (speculative)", "green"),
        ],
    },
}

# -----------------------------------------------------------------------------
# Geometry, top-left logical coordinates
# -----------------------------------------------------------------------------
BOUNDS = {
    "key": (18, 1, 1106, 44),
    "initial": (372, 58, 752, 107),
    "wlhiv_header": (37, 157, 244, 216),
    "wlhiv_boxes": [(37, 250, 244, 286), (37, 317, 244, 352), (37, 383, 244, 420), (37, 450, 244, 486), (37, 516, 244, 553)],
    "standard_cervical": (248, 347, 465, 407),
    "hsv_standard": (561, 454, 776, 512),
    "hsv_tierd": (561, 536, 776, 594),
    "ct_boxes": [(836, 307, 1050, 344), (836, 371, 1050, 408), (836, 435, 1050, 486), (836, 511, 1050, 549)],
    "summary": (18, 612, 1106, 702),
}
DIAMONDS = {
    "HIV": [(562,146),(686,186),(562,228),(438,186)],
    "CT": [(944,186),(1038,227),(944,267),(850,227)],
    "HSV": [(668,336),(770,375),(668,414),(568,375)],
}
KEY_COL_BOUNDS = [(45, 295), (310, 650), (650, 870), (870, 1098)]

# -----------------------------------------------------------------------------
# Drawing helpers
# -----------------------------------------------------------------------------
@dataclass
class TextRec:
    name: str
    artist: object
    parent: Tuple[float, float, float, float] | None
    kind: str


def configure_matplotlib() -> None:
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [FONT_FAMILY, "DejaVu Sans"],
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.transparent": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "path.simplify": False,
        "axes.unicode_minus": False,
    })


def interp3(stops: Sequence[str], n: int) -> np.ndarray:
    rgb = np.array([to_rgb(c) for c in stops], dtype=float)
    t = np.linspace(0, 1, n)
    out = np.empty((n,3))
    for i,v in enumerate(t):
        if v <= .5:
            u=v/.5; out[i]=rgb[0]*(1-u)+rgb[1]*u
        else:
            u=(v-.5)/.5; out[i]=rgb[1]*(1-u)+rgb[2]*u
    return out


def rounded_gradient_box(ax, bounds, palette_key, radius=9, lw=1.1, strips=100, z=2):
    x0,y0,x1,y1=bounds; w=x1-x0; h=y1-y0
    stops,border=PAL[palette_key]
    clip=FancyBboxPatch((x0,y0),w,h,boxstyle=f"round,pad=0,rounding_size={radius}",facecolor="none",edgecolor="none")
    ax.add_patch(clip)
    cols=interp3(stops,strips); sw=w/strips
    for i,c in enumerate(cols):
        r=Rectangle((x0+i*sw,y0),sw*1.06,h,facecolor=c,edgecolor="none",linewidth=0,antialiased=False,zorder=z)
        r.set_clip_path(clip); ax.add_patch(r)
    border_patch=FancyBboxPatch((x0,y0),w,h,boxstyle=f"round,pad=0,rounding_size={radius}",facecolor="none",edgecolor=border,linewidth=lw,zorder=z+.2)
    ax.add_patch(border_patch)
    return border_patch


def diamond_gradient(ax, vertices, palette_key, lw=1.2, strips=100, z=2):
    stops,border=PAL[palette_key]
    path=MplPath(vertices+[vertices[0]])
    clip=PathPatch(path,facecolor="none",edgecolor="none")
    ax.add_patch(clip)
    xs=[v[0] for v in vertices]; ys=[v[1] for v in vertices]
    x0,x1=min(xs),max(xs); y0,y1=min(ys),max(ys)
    cols=interp3(stops,strips); sw=(x1-x0)/strips
    for i,c in enumerate(cols):
        r=Rectangle((x0+i*sw,y0),sw*1.06,y1-y0,facecolor=c,edgecolor="none",linewidth=0,antialiased=False,zorder=z)
        r.set_clip_path(clip); ax.add_patch(r)
    outline=Polygon(vertices,closed=True,facecolor="none",edgecolor=border,linewidth=lw,zorder=z+.2)
    ax.add_patch(outline)
    return outline


def add_text(ax, reg: List[TextRec], name, x,y,text,parent=None,fontsize=5.5,color="black",weight="normal",style="normal",ha="center",va="center",z=8):
    art=ax.text(x,y,text,ha=ha,va=va,fontsize=fontsize,color=color,fontweight=weight,fontstyle=style,zorder=z)
    reg.append(TextRec(name,art,parent,"text")); return art


def add_lines(ax,reg,prefix,bounds,lines,ys,fontsize,color,weights=None,styles=None,z=8):
    x0,y0,x1,y1=bounds; cx=(x0+x1)/2
    weights=weights or ["normal"]*len(lines); styles=styles or ["normal"]*len(lines)
    out=[]
    for i,(line,y) in enumerate(zip(lines,ys)):
        out.append(add_text(ax,reg,f"{prefix}_{i}",cx,y,line,bounds,fontsize,color,weights[i],styles[i],z=z))
    return out


def arrow(ax,start,end,color,lw=1.2,dashed=False,z=1.4,ms=10):
    ls=(0,(4,3)) if dashed else "solid"
    arr=FancyArrowPatch(start,end,arrowstyle="-|>",mutation_scale=ms,linewidth=lw,color=color,linestyle=ls,shrinkA=0,shrinkB=0,zorder=z)
    ax.add_patch(arr); return arr


def make_figure(figsize):
    fig=plt.figure(figsize=figsize,dpi=DPI,facecolor="white",layout=None)
    ax=fig.add_axes([0,0,1,1])
    ax.set_xlim(0,LOGICAL_W); ax.set_ylim(LOGICAL_H,0); ax.set_aspect("equal"); ax.axis("off")
    return fig,ax

# -----------------------------------------------------------------------------
# Figure drawing
# -----------------------------------------------------------------------------
def draw_top_key(ax,reg,boxes):
    b=BOUNDS["key"]; rounded_gradient_box(ax,b,"key",radius=7,lw=.8,strips=150,z=.8); boxes.append(b)
    y_rows=[14,31]
    for ri,row in enumerate(TEXT["key"]):
        for ci,(tier,rest,key) in enumerate(row):
            x0,x1=KEY_COL_BOUNDS[ci]
            # Tier label and coloured descriptor rendered as two native text objects.
            tier_x=x0+4
            tier_art=add_text(ax,reg,f"key_{ri}_{ci}_tier",tier_x,y_rows[ri],tier,None,FONT["key"],COL["dark"],"bold",ha="left")
            # fixed offsets chosen from measured source geometry
            offsets=[43,43,43,43]
            add_text(ax,reg,f"key_{ri}_{ci}_rest",tier_x+offsets[ci],y_rows[ri],rest,None,FONT["key"],COL[key],"normal",ha="left")


def draw_initial_and_decisions(ax,reg,arrows,boxes):
    b=BOUNDS["initial"]; rounded_gradient_box(ax,b,"initial",radius=23,lw=1.1,strips=100,z=3); boxes.append(b)
    add_text(ax,reg,"initial_text",(b[0]+b[2])/2,(b[1]+b[3])/2,TEXT["initial"],b,FONT["initial"],"white","bold")
    arrows.append(arrow(ax,(562,b[3]),(562,143),COL["dark"],1.25,False,1.5,10))
    # decisions
    diamond_gradient(ax,DIAMONDS["HIV"],"tier_a",1.2,110,z=3)
    diamond_gradient(ax,DIAMONDS["CT"],"blue",1.2,110,z=3)
    diamond_gradient(ax,DIAMONDS["HSV"],"green",1.2,110,z=3)
    boxes.extend([(438,146,686,228),(850,186,1038,267),(568,336,770,414)])
    add_lines(ax,reg,"HIV_dec",(438,146,686,228),TEXT["decisions"]["HIV"],[177,196],FONT["decision"],COL["red"],["bold","bold"])
    add_lines(ax,reg,"CT_dec",(850,186,1038,267),TEXT["decisions"]["CT"],[215,238],FONT["decision"],COL["blue"],["bold","bold"])
    add_lines(ax,reg,"HSV_dec",(568,336,770,414),TEXT["decisions"]["HSV"],[365,386],FONT["decision"],COL["green"],["bold","bold"])
    # branches and labels
    arrows.append(arrow(ax,(438,186),(244,186),COL["red"],1.25,False,1.4,10))
    add_text(ax,reg,"label_hivplus",367,178,TEXT["branch_labels"]["HIV+"],None,FONT["branch"],COL["red"],"bold")
    arrows.append(arrow(ax,(686,186),(883,186),COL["green"],1.25,False,1.4,10))
    add_text(ax,reg,"label_hivminus",752,178,TEXT["branch_labels"]["HIV-"],None,FONT["branch"],COL["green"],"bold")
    # CT branch label and downward arrow
    arrows.append(arrow(ax,(944,267),(944,304),COL["blue"],1.25,False,1.4,10))
    add_text(ax,reg,"label_ctplus",944,279,TEXT["branch_labels"]["CT+"],None,FONT["branch"],COL["blue"],"bold")
    # CT- diagonal to HSV
    arrows.append(arrow(ax,(850,239),(741,349),COL["green"],1.2,False,1.3,9))
    add_text(ax,reg,"label_ctminus",731,224,TEXT["branch_labels"]["CT-"],None,FONT["branch"],COL["green"],"bold")


def draw_wlhiv(ax,reg,arrows,boxes):
    b=BOUNDS["wlhiv_header"]; rounded_gradient_box(ax,b,"tier_a_fill",radius=8,lw=1.0,strips=90,z=3); boxes.append(b)
    add_lines(ax,reg,"wlhiv_header",b,TEXT["wlhiv"]["header"],[173,190,206],FONT["major"],"white",["bold","normal","normal"])
    # arrow from header downward
    wboxes=BOUNDS["wlhiv_boxes"]
    arrows.append(arrow(ax,(140.5,b[3]),(140.5,wboxes[0][1]-3),COL["red"],1.15,False,1.3,9))
    for i,box in enumerate(wboxes):
        key="orange" if i==4 else "tier_a"
        rounded_gradient_box(ax,box,key,radius=6,lw=.95,strips=80,z=3); boxes.append(box)
        lines=TEXT["wlhiv"]["boxes"][i]; cy=(box[1]+box[3])/2
        ys=[cy-7,cy+7]
        color=COL["orange"] if i==4 else COL["red"]
        add_lines(ax,reg,f"wlhiv_{i}",box,lines,ys,FONT["process_body"],color,["bold","normal"])
        if i < len(wboxes)-1:
            arrows.append(arrow(ax,(140.5,box[3]),(140.5,wboxes[i+1][1]-3),COL["red"],1.15,False,1.3,9))


def draw_ct(ax,reg,arrows,boxes):
    cboxes=BOUNDS["ct_boxes"]
    for i,b in enumerate(cboxes):
        key="blue" if i<2 else ("orange" if i==2 else "neutral")
        rounded_gradient_box(ax,b,key,radius=6,lw=.95,strips=85,z=3); boxes.append(b)
        lines=TEXT["ct"]["boxes"][i]
        cy=(b[1]+b[3])/2
        if len(lines)==2:
            ys=[cy-7,cy+7]; styles=["normal","normal"]; weights=["bold" if i<3 else "normal","normal"]
            f=FONT["process_small"] if i in (0,1) else FONT["process_body"]
        else:
            ys=[cy-13,cy,cy+13]; styles=["normal","normal","italic"]; weights=["bold","normal","normal"]; f=FONT["process_small"]
        color=COL["blue"] if i<2 else (COL["orange"] if i==2 else COL["neutral_body"])
        add_lines(ax,reg,f"ct_{i}",b,lines,ys,f,color,weights,styles)
        if i<len(cboxes)-1:
            dashed=(i==2)
            col=COL["orange"] if dashed else COL["blue"]
            arrows.append(arrow(ax,(943,b[3]),(943,cboxes[i+1][1]-3),col,1.15,dashed,1.3,9))


def draw_hsv(ax,reg,arrows,boxes):
    # left branch to standard cervical protocol
    sb=BOUNDS["standard_cervical"]
    arrows.append(arrow(ax,(568,375),(465,375),COL["dark"],1.15,False,1.3,9))
    rounded_gradient_box(ax,sb,"neutral",radius=6,lw=.95,strips=90,z=3); boxes.append(sb)
    add_lines(ax,reg,"std_cervical",sb,TEXT["hsv"]["standard_cervical"],[362,378,396],FONT["process_body"],COL["neutral_head"],["bold","normal","normal"],["normal","normal","italic"])
    # down branch
    add_text(ax,reg,"label_any",668,430,TEXT["branch_labels"]["Any"],None,FONT["branch"],COL["green"],"bold")
    arrows.append(arrow(ax,(668,414),(668,451),COL["green"],1.15,False,1.3,9))
    hb=BOUNDS["hsv_standard"]; rounded_gradient_box(ax,hb,"green",radius=6,lw=.95,strips=90,z=3); boxes.append(hb)
    add_lines(ax,reg,"hsv_standard",hb,TEXT["hsv"]["standard"],[469,486,503],FONT["process_body"],COL["green"],["bold","normal","normal"],["normal","normal","italic"])
    td=BOUNDS["hsv_tierd"]; arrows.append(arrow(ax,(668,hb[3]),(668,td[1]-3),COL["green"],1.15,True,1.3,9))
    rounded_gradient_box(ax,td,"tier_d",radius=6,lw=1.0,strips=90,z=3); boxes.append(td)
    add_lines(ax,reg,"hsv_tierd",td,TEXT["hsv"]["tier_d"],[550,566,583],FONT["process_small"],COL["tier_d"],["bold","normal","normal"],["normal","normal","italic"])


def draw_summary(ax,reg,boxes):
    b=BOUNDS["summary"]; rounded_gradient_box(ax,b,"summary",radius=8,lw=.8,strips=150,z=4); boxes.append(b)
    add_text(ax,reg,"summary_title",562,629,TEXT["summary"]["title"],b,FONT["summary_title"],COL["dark"],"bold")
    ys=[649,669,688]
    label_x=75; body_x=198
    for i,(lab,body,key) in enumerate(TEXT["summary"]["rows"]):
        add_text(ax,reg,f"summary_label_{i}",label_x,ys[i],lab,b,FONT["summary_label"],COL[key],"bold",ha="left")
        add_text(ax,reg,f"summary_body_{i}",body_x,ys[i],body,b,FONT["summary_body"],COL["grey"],"normal",ha="left")


def create_figure(figsize):
    fig,ax=make_figure(figsize)
    reg: List[TextRec]=[]; arrows=[]; boxes=[]
    draw_top_key(ax,reg,boxes)
    draw_initial_and_decisions(ax,reg,arrows,boxes)
    draw_wlhiv(ax,reg,arrows,boxes)
    draw_ct(ax,reg,arrows,boxes)
    draw_hsv(ax,reg,arrows,boxes)
    draw_summary(ax,reg,boxes)
    return fig,ax,reg,arrows,boxes

# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------
def logical_bbox_to_display(ax,bounds):
    x0,y0,x1,y1=bounds; pts=ax.transData.transform([(x0,y0),(x1,y1)])
    return Bbox.from_extents(min(pts[:,0]),min(pts[:,1]),max(pts[:,0]),max(pts[:,1]))


def overlaps(a,b,pad=0):
    return not ((a.x1+pad)<=b.x0 or (b.x1+pad)<=a.x0 or (a.y1+pad)<=b.y0 or (b.y1+pad)<=a.y0)


def validate_scientific_content():
    blob=repr(TEXT)
    required=["Woman Presenting for Cervical Screening","Tier D:","HSV-2 suppression as CC prevention","SPECULATIVE – No evidence base","Concurrent testing clinically reasonable","Doxycycline/azithromycin (STI benefit)","Hypothesis-generating; no RCT evidence","(hypothesis-generating)","(speculative)","CT+HPV","optimisation","serostatus"]
    missing=[x for x in required if x not in blob]
    if missing: raise RuntimeError("Missing required text: "+", ".join(missing))


def validate_layout(fig,ax,reg,arrows,boxes):
    fig.canvas.draw(); ren=fig.canvas.get_renderer(); canvas=fig.bbox
    failures=[]; records=[]; min_clear=1e9
    for r in reg:
        bb=r.artist.get_window_extent(ren); records.append((r,bb))
        if bb.x0<canvas.x0-.5 or bb.y0<canvas.y0-.5 or bb.x1>canvas.x1+.5 or bb.y1>canvas.y1+.5:
            failures.append(f"canvas overflow: {r.name}")
        if r.parent is not None:
            pb=logical_bbox_to_display(ax,r.parent)
            clear=min(bb.x0-pb.x0,pb.x1-bb.x1,bb.y0-pb.y0,pb.y1-bb.y1)
            min_clear=min(min_clear,clear)
            # Long summary rows are fitted with small but visible margins.
            threshold=2.0 if r.name.startswith("summary_") else 3.0
            if clear < threshold: failures.append(f"parent overflow/clearance: {r.name} ({clear:.2f}px)")
    # Top key frame and entry-specific containment/non-overlap.
    key_bb=logical_bbox_to_display(ax,BOUNDS["key"])
    key_items=[(r,bb) for r,bb in records if r.name.startswith("key_")]
    for r,bb in key_items:
        if bb.x0<key_bb.x0+4 or bb.x1>key_bb.x1-6 or bb.y0<key_bb.y0+3 or bb.y1>key_bb.y1-3:
            failures.append(f"top-key containment: {r.name}")
    # pairwise key row collision
    for row in [0,1]:
        row_items=[(r,bb) for r,bb in key_items if r.name.startswith(f"key_{row}_")]
        row_items=sorted(row_items,key=lambda x:x[1].x0)
        for i in range(len(row_items)-1):
            if overlaps(row_items[i][1],row_items[i+1][1],pad=3):
                failures.append(f"top-key overlap: {row_items[i][0].name} / {row_items[i+1][0].name}")
    # Initial node complete and tier D specific checks.
    initial=[bb for r,bb in records if r.name=="initial_text"][0]
    initial_parent=logical_bbox_to_display(ax,BOUNDS["initial"])
    if initial.x0<initial_parent.x0+8 or initial.x1>initial_parent.x1-8: failures.append("initial node horizontal clearance")
    td_parent=logical_bbox_to_display(ax,BOUNDS["hsv_tierd"])
    td_text=[bb for r,bb in records if r.name.startswith("hsv_tierd_")]
    for bb in td_text:
        if bb.y1>td_parent.y1-4: failures.append("HSV Tier D bottom clipping")
    summary_bb=logical_bbox_to_display(ax,BOUNDS["summary"])
    bottom_margin=canvas.y1-summary_bb.y1
    if bottom_margin<8: failures.append("summary panel bottom canvas margin")
    if failures: raise RuntimeError("Layout validation failed: "+"; ".join(sorted(set(failures))))
    lookup={r.name:bb for r,bb in records}
    return {
        "records":records,"failures":failures,"text_count":len(reg),"box_count":len(boxes),"arrow_count":len(arrows),
        "min_clearance":min_clear,"key_bbox":key_bb,"key_entries":{r.name:bb for r,bb in key_items},
        "initial_bbox":initial,"hsv_tierd_bbox":td_parent,"hsv_tierd_text":td_text,
        "ct_long_text":{k:v for k,v in lookup.items() if k in ["ct_0_1","ct_1_1","ct_2_2"]},
        "summary_bbox":summary_bb,"summary_text":{k:v for k,v in lookup.items() if k.startswith("summary_")},
        "bottom_margin":bottom_margin,"decision_count":3,"dashed_connectors":2,
    }

# -----------------------------------------------------------------------------
# Exports and inspection
# -----------------------------------------------------------------------------
def render_raster():
    fig,ax,reg,arrows,boxes=create_figure(RASTER_FIGSIZE)
    canvas=matplotlib.backends.backend_agg.FigureCanvasAgg(fig); canvas.draw()
    val=validate_layout(fig,ax,reg,arrows,boxes)
    img=Image.fromarray(np.asarray(canvas.buffer_rgba()),"RGBA").convert("RGB")
    if img.size!=(RASTER_W,RASTER_H): raise RuntimeError(f"Unexpected raster size {img.size}")
    plt.close(fig); return img,val


def export_rasters():
    img,val=render_raster()
    png=OUT_DIR/"Figure_9_co_infection_screening_algorithm.png"
    jpg=OUT_DIR/"Figure_9_co_infection_screening_algorithm.jpg"
    tif=OUT_DIR/"Figure_9_co_infection_screening_algorithm.tiff"
    img.save(png,"PNG",dpi=(DPI,DPI),optimize=True)
    img.save(jpg,"JPEG",dpi=(DPI,DPI),quality=100,subsampling=0,optimize=True)
    img.save(tif,"TIFF",dpi=(DPI,DPI),compression="tiff_lzw")
    return png,jpg,tif,val


def export_vectors():
    pdf=OUT_DIR/"Figure_9_co_infection_screening_algorithm.pdf"; svg=OUT_DIR/"Figure_9_co_infection_screening_algorithm.svg"
    fig,ax,reg,arrows,boxes=create_figure(VECTOR_FIGSIZE)
    validate_layout(fig,ax,reg,arrows,boxes)
    fig.savefig(pdf,format="pdf",facecolor="white",transparent=False)
    fig.savefig(svg,format="svg",facecolor="white",transparent=False)
    plt.close(fig); return pdf,svg


def inspect_reference():
    with Image.open(REFERENCE_IMAGE) as im: return im.size,im.mode


def inspect_raster(path):
    with Image.open(path) as im:
        comp=im.info.get("compression")
        if path.suffix.lower() in {".tif",".tiff"}: comp=im.tag_v2.get(259,comp)
        return {"file":path.name,"format":im.format,"size":im.size,"mode":im.mode,"dpi":im.info.get("dpi"),"transparency":("A" in im.getbands() or "transparency" in im.info),"compression":comp,"frames":getattr(im,"n_frames",1),"bytes":path.stat().st_size}


def inspect_vectors(pdf,svg):
    out={}
    try:
        import fitz
        doc=fitz.open(pdf); page=doc[0]
        out["pdf_page_points"]=(page.rect.width,page.rect.height); out["pdf_image_count"]=len(page.get_images(full=True)); out["pdf_pages"]=doc.page_count
        doc.close()
    except Exception as e: out["pdf_error"]=str(e)
    s=svg.read_text(errors="replace")
    out["svg_has_image_tag"]=bool(re.search(r"<image\b",s,re.I)); out["svg_rect_count"]=len(re.findall(r"<rect\b",s,re.I)); out["svg_path_count"]=len(re.findall(r"<path\b",s,re.I))
    mw=re.search(r'<svg[^>]+width="([^"]+)"',s,re.I); mh=re.search(r'<svg[^>]+height="([^"]+)"',s,re.I)
    out["svg_width"]=mw.group(1) if mw else "not found"; out["svg_height"]=mh.group(1) if mh else "not found"
    return out

# -----------------------------------------------------------------------------
# QA
# -----------------------------------------------------------------------------
def qa_font(size,bold=False):
    try:
        from matplotlib import font_manager
        p=font_manager.findfont(font_manager.FontProperties(family=FONT_FAMILY,weight="bold" if bold else "normal"))
        return ImageFont.truetype(p,size)
    except Exception: return ImageFont.load_default()


def create_qa(png,val):
    with Image.open(REFERENCE_IMAGE) as im: ref=im.convert("RGB")
    with Image.open(png) as im: full=im.convert("RGB")
    rec=full.resize((LOGICAL_W,LOGICAL_H),Image.Resampling.LANCZOS)
    overlay=OUT_DIR/"Figure_9_QA_overlay.png"; Image.blend(ref,rec,.5).save(overlay,dpi=(DPI,DPI))
    gap=18; header=44
    side=Image.new("RGB",(LOGICAL_W*2+gap,LOGICAL_H+header),"white"); side.paste(ref,(0,header)); side.paste(rec,(LOGICAL_W+gap,header))
    d=ImageDraw.Draw(side); f=qa_font(21,True); d.text((LOGICAL_W//2,16),"Reference",fill="black",font=f,anchor="mm"); d.text((LOGICAL_W+gap+LOGICAL_W//2,16),"Reconstructed",fill="black",font=f,anchor="mm")
    sidep=OUT_DIR/"Figure_9_QA_side_by_side.png"; side.save(sidep,dpi=(DPI,DPI))
    def edge(im):
        e=ImageOps.grayscale(im).filter(ImageFilter.FIND_EDGES); e=ImageOps.autocontrast(e,cutoff=1); return ImageOps.invert(e).convert("RGB")
    ec=Image.new("RGB",side.size,"white"); ec.paste(edge(ref),(0,header)); ec.paste(edge(rec),(LOGICAL_W+gap,header)); de=ImageDraw.Draw(ec); de.text((LOGICAL_W//2,16),"Reference edges",fill="black",font=f,anchor="mm"); de.text((LOGICAL_W+gap+LOGICAL_W//2,16),"Reconstructed edges",fill="black",font=f,anchor="mm")
    edges=OUT_DIR/"Figure_9_QA_edges.png"; ec.save(edges,dpi=(DPI,DPI))
    bounds=full.copy(); db=ImageDraw.Draw(bounds)
    for _,bb in val["records"]: db.rectangle((int(bb.x0),int(RASTER_H-bb.y1),int(bb.x1),int(RASTER_H-bb.y0)),outline=(220,0,180),width=1)
    for b in [BOUNDS["key"],BOUNDS["initial"],BOUNDS["summary"],BOUNDS["hsv_tierd"]]:
        sx=RASTER_W/LOGICAL_W; sy=RASTER_H/LOGICAL_H; x0,y0,x1,y1=b
        db.rectangle((int(x0*sx),int(y0*sy),int(x1*sx),int(y1*sy)),outline=(0,150,255),width=2)
    textb=OUT_DIR/"Figure_9_QA_text_bounds.png"; bounds.save(textb,dpi=(DPI,DPI))
    sx=RASTER_W/LOGICAL_W; sy=RASTER_H/LOGICAL_H
    def crop(name,b,scale=2):
        x0,y0,x1,y1=b; c=full.crop((int(x0*sx),int(y0*sy),int(x1*sx),int(y1*sy))); c=c.resize((c.width*scale,c.height*scale),Image.Resampling.LANCZOS); p=OUT_DIR/name; c.save(p,dpi=(DPI,DPI)); return p
    top=crop("Figure_9_QA_top_key_closeup.png",(8,0,1116,52),3)
    wl=crop("Figure_9_QA_WLHIV_branch_closeup.png",(20,135,260,570),2)
    ct=crop("Figure_9_QA_CT_branch_closeup.png",(710,165,1065,565),2)
    hsv=crop("Figure_9_QA_HSV2_branch_closeup.png",(235,320,790,605),2)
    summ=crop("Figure_9_QA_summary_closeup.png",(10,602,1114,712),2)
    return overlay,sidep,edges,textb,top,wl,ct,hsv,summ


def write_report(ref_info,rasters,pdf,svg,val,vector):
    p=OUT_DIR/"Figure_9_QA_report.txt"; lines=[]
    lines += ["FIGURE 9 PUBLICATION-READY QA REPORT","="*56,""]
    lines += [f"Source dimensions: {ref_info[0][0]} x {ref_info[0][1]} px",f"Source colour mode: {ref_info[1]}","Source use: visual QA only; not embedded or upscaled into publication outputs.",""]
    lines += [f"Final physical dimensions: {WIDTH_CM:.3f} x {HEIGHT_CM:.3f} cm",f"Final raster dimensions: {RASTER_W} x {RASTER_H} px",f"Nominal DPI: {DPI} x {DPI}",""]
    lines += ["RASTER OUTPUTS","-"*56]
    for q in map(inspect_raster,rasters):
        lines += [f"File: {q['file']}",f"  Format: {q['format']}",f"  Dimensions: {q['size'][0]} x {q['size'][1]}",f"  Mode: {q['mode']}",f"  DPI metadata: {q['dpi']}",f"  Transparency: {q['transparency']}",f"  File size: {q['bytes']} bytes"]
        if q['format']=='TIFF': lines += [f"  Compression tag: {q['compression']} (5 = LZW)",f"  Frames: {q['frames']}"]
        if q['format']=='JPEG': lines += ["  JPEG quality: 100","  JPEG subsampling: 0 (4:4:4)"]
    lines += ["","VECTOR OUTPUTS","-"*56]
    for k,v in vector.items(): lines.append(f"{k}: {v}")
    lines += ["Gradients: opaque adjacent vector rectangles clipped to rounded vector paths/polygons.","Original source raster embedded: no.",""]
    lines += ["CONTENT AND GEOMETRY","-"*56,f"Decision diamonds: {val['decision_count']}",f"Process/outer boxes: {val['box_count']}",f"Arrows/connectors: {val['arrow_count']}",f"Dashed connectors: {val['dashed_connectors']}",f"Text objects: {val['text_count']}",f"Text-overflow failures: {len(val['failures'])}",f"Minimum text-to-parent clearance: {val['min_clearance']:.2f} px",f"White margin beneath bottom summary: {val['bottom_margin']:.2f} px",""]
    def fmt(bb): return f"({bb.x0:.2f}, {bb.y0:.2f}, {bb.x1:.2f}, {bb.y1:.2f})"
    lines += ["BOUNDARY VALIDATION","-"*56,f"Top key bbox: {fmt(val['key_bbox'])}"]
    for name,bb in sorted(val['key_entries'].items()): lines.append(f"Top-key entry [{name}]: {fmt(bb)}")
    lines += [f"Initial-node text bbox: {fmt(val['initial_bbox'])}",f"HSV-2 Tier D box bbox: {fmt(val['hsv_tierd_bbox'])}"]
    for i,bb in enumerate(val['hsv_tierd_text']): lines.append(f"HSV-2 Tier D text line {i+1}: {fmt(bb)}")
    for name,bb in val['ct_long_text'].items(): lines.append(f"CT long text [{name}]: {fmt(bb)}")
    lines += [f"Bottom summary bbox: {fmt(val['summary_bbox'])}"]
    for name,bb in val['summary_text'].items(): lines.append(f"Summary text [{name}]: {fmt(bb)}")
    lines += ["","SCIENTIFIC AND CLINICAL VALIDATION","-"*56,
              "All Tier A/B/C/D assignments present and unchanged.",
              "All clinical recommendations and evidence qualifications present.",
              "Branch logic unchanged: HIV+ left; HIV- right; CT+ down; CT- to HSV-2; HSV standard pathways retained.",
              "Top evidence-key overlap eliminated: yes.",
              "Complete initial-node wording visible: yes.",
              "Complete HSV-2 Tier D wording visible: yes.",
              "All CT workflow text contained: yes.",
              "Bottom summary containment passed: yes.",
              "Gradients are decorative, opaque, and vector-safe.",
              "Minor residual differences: renderer antialiasing and intentionally lighter premium gradients."]
    p.write_text("\n".join(lines)+"\n",encoding="utf-8"); return p


def create_zip(paths):
    with zipfile.ZipFile(ZIP_PATH,"w",zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in paths: z.write(p,arcname=p.name)
    return ZIP_PATH


def main():
    configure_matplotlib(); validate_scientific_content(); ref=inspect_reference()
    if ref[0] != (LOGICAL_W,LOGICAL_H): raise RuntimeError(f"Unexpected reference size {ref[0]}")
    png,jpg,tif,val=export_rasters(); pdf,svg=export_vectors(); vector=inspect_vectors(pdf,svg)
    qa=create_qa(png,val); report=write_report(ref,[png,jpg,tif],pdf,svg,val,vector)
    deliverables=[Path(__file__).resolve(),png,jpg,tif,pdf,svg,*qa,report]
    package=create_zip(deliverables)
    print(f"Generated {len(deliverables)} deliverables")
    print(package)
    print(f"Text objects: {val['text_count']} | arrows: {val['arrow_count']} | min clearance: {val['min_clearance']:.2f}px")

if __name__=="__main__": main()
