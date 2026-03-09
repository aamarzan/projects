from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch
import numpy as np
import requests


@dataclass
class Config:
    excel_path: str = r"C:\Users\User\OneDrive\2. Personal\AAM\Desktop\Mixed OP\Final\Map.xlsx"
    sheet_name: str = "Map"
    out_dir: str = r"C:\Users\User\OneDrive\2. Personal\AAM\Desktop\Mixed OP\Final"
    out_stem: str = "figure_mixed_op_study_sites_bangladesh_premium"

    district_json: str = "BD_Districts.json"
    division_json: str = "BD_Divisions.json"
    district_url: str = (
        "https://github.com/fahim-muntasir-niloy/Bangladesh-geojson_project/raw/refs/heads/main/"
        "Geo%20Json%20Maps/BD_Districts.json"
    )
    division_url: str = (
        "https://github.com/fahim-muntasir-niloy/Bangladesh-geojson_project/raw/refs/heads/main/"
        "Geo%20Json%20Maps/BD_Divisions.json"
    )
    dpi: int = 700
    figsize: Tuple[float, float] = (8.9, 10.8)

    ocean: str = "#EAF3FB"
    land: str = "#FBFBF8"
    site_district_fill: str = "#EEF5FB"
    district_edge: str = "#AAB8C7"
    division_edge: str = "#475569"
    country_edge: str = "#111827"
    grid_color: str = "#D5E2EF"
    marker_face: str = "#0B4F8A"
    marker_edge: str = "white"
    count_box_face: str = "#FFFFFF"
    count_box_edge: str = "#1F2937"
    count_text: str = "#8A1538"
    site_text: str = "#0B3D69"

    title: str = "Study sites and enrolled mixed OP cases in Bangladesh"
    subtitle: str = "Nine sentinel hospitals; box area is proportional to site-level participant count"


CFG = Config()

# Hospital site metadata
SITE_META: Dict[str, Dict[str, object]] = {
    "CMCH": {"district": "Chittagong", "lon": 91.7832, "lat": 22.3569},
    "DMCH": {"district": "Dhaka", "lon": 90.4074, "lat": 23.7250},
    "JMCH": {"district": "Jessore", "lon": 89.2081, "lat": 23.1664},
    "KMCH": {"district": "Khulna", "lon": 89.5403, "lat": 22.8456},
    "MMCH": {"district": "Mymensingh", "lon": 90.4203, "lat": 24.7471},
    "RMCH": {"district": "Rajshahi", "lon": 88.6042, "lat": 24.3745},
    "RpMCH": {"district": "Rangpur", "lon": 89.2752, "lat": 25.7439},
    "SOMCH": {"district": "Sylhet", "lon": 91.8687, "lat": 24.8949},
    "SZMCH": {"district": "Bogra", "lon": 89.3776, "lat": 24.8465},
}

# Tuned label box anchors in lon/lat so the boxes do not collide with markers.
LABEL_BOX_POS: Dict[str, Tuple[float, float]] = {
    "CMCH": (92.10, 22.31),
    "DMCH": (90.79, 23.84),
    "JMCH": (88.77, 23.06),
    "KMCH": (89.11, 22.91),
    "MMCH": (90.78, 25.03),
    "RMCH": (88.15, 24.63),
    "RpMCH": (88.82, 26.03),
    "SOMCH": (92.13, 25.00),
    "SZMCH": (89.63, 24.82),
}

ALIASES = {
    "Chattogram": "Chittagong",
    "Jashore": "Jessore",
    "Barishal": "Barisal",
    "Bogura": "Bogra",
}


def ensure_geojson(local_path: Path, url: str) -> Path:
    if local_path.exists():
        return local_path
    local_path.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    local_path.write_bytes(r.content)
    return local_path


def read_counts(excel_path: Path, sheet_name: str) -> pd.DataFrame:
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    cols = {str(c).strip().lower(): c for c in df.columns}
    if "site" not in cols:
        raise ValueError(f"'Site' column not found in {excel_path}. Found: {list(df.columns)}")
    site_col = cols["site"]
    pid_col = cols.get("patient id")

    keep_cols = [site_col] + ([pid_col] if pid_col else [])
    x = df[keep_cols].copy()
    x.columns = ["Site"] + (["Patient ID"] if pid_col else [])
    x["Site"] = x["Site"].astype(str).str.strip()
    x = x[x["Site"].isin(SITE_META)].copy()
    if pid_col:
        counts = x.groupby("Site")["Patient ID"].nunique().rename("Count").reset_index()
    else:
        counts = x.groupby("Site").size().rename("Count").reset_index()

    for site in SITE_META:
        if site not in counts["Site"].tolist():
            counts = pd.concat([counts, pd.DataFrame({"Site": [site], "Count": [0]})], ignore_index=True)
    counts["Count"] = counts["Count"].astype(int)
    return counts.sort_values("Count", ascending=False).reset_index(drop=True)


def load_geo_layers(district_path: Path, division_path: Path):
    districts = gpd.read_file(district_path)
    divisions = gpd.read_file(division_path)
    districts["ADM2_EN"] = districts["ADM2_EN"].astype(str).str.strip()
    divisions["ADM1_EN"] = divisions["ADM1_EN"].astype(str).str.strip()
    if districts.crs is None:
        districts = districts.set_crs("EPSG:4326")
    if divisions.crs is None:
        divisions = divisions.set_crs("EPSG:4326")
    return districts, divisions


def add_header(ax, total_n: int):
    ax.text(
        0.5, 1.035, CFG.title,
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=18, fontweight="bold", color="#0F172A",
    )
    ax.text(
        0.5, 1.007, f"{CFG.subtitle} (overall n={total_n:,})",
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=10.5, color="#334155",
    )


def add_north_arrow(ax):
    ax.annotate(
        "N", xy=(0.94, 0.90), xytext=(0.94, 0.82),
        xycoords="axes fraction", textcoords="axes fraction",
        ha="center", va="center", fontsize=12, fontweight="bold", color="#0F172A",
        arrowprops=dict(arrowstyle="-|>", lw=1.5, color="#0F172A"), zorder=30,
    )


def format_axes(ax, bounds):
    minx, miny, maxx, maxy = bounds
    xpad, ypad = 0.45, 0.38
    xlim = (minx - xpad, maxx + xpad)
    ylim = (miny - ypad, maxy + ypad)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")

    xticks = np.arange(np.floor(xlim[0]) + 1, np.ceil(xlim[1]), 1.0)
    yticks = np.arange(np.floor(ylim[0]) + 1, np.ceil(ylim[1]), 1.0)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([f"{x:.0f}°E" for x in xticks], fontsize=9.5)
    ax.set_yticklabels([f"{y:.0f}°N" for y in yticks], fontsize=9.5)
    ax.tick_params(top=False, right=False, labeltop=False, labelright=False, direction="out", length=3.6)

    for x in xticks:
        ax.axvline(x, color=CFG.grid_color, lw=0.55, zorder=0)
    for y in yticks:
        ax.axhline(y, color=CFG.grid_color, lw=0.55, zorder=0)

    for s in ax.spines.values():
        s.set_linewidth(0.95)
        s.set_color("#475569")


def size_box(count: int, min_count: int, max_count: int) -> Tuple[float, float]:
    if max_count == min_count:
        norm = 1.0
    else:
        norm = (count - min_count) / (max_count - min_count)
    # use sqrt scaling so differences are visible but not extreme
    s = 0.35 + 0.65 * np.sqrt(max(norm, 0))
    width = 0.22 + 0.22 * s
    height = 0.11 + 0.08 * s
    return width, height


def draw_scaled_box(ax, x: float, y: float, count: int, width: float, height: float):
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        facecolor=CFG.count_box_face, edgecolor=CFG.count_box_edge,
        linewidth=1.0, zorder=24,
        path_effects=[pe.withSimplePatchShadow(offset=(0.03, -0.03), alpha=0.18)],
    )
    ax.add_patch(patch)
    txt = ax.text(
        x, y, f"{count}", ha="center", va="center",
        fontsize=11.5, fontweight="bold", color=CFG.count_text, zorder=25,
    )
    txt.set_path_effects([pe.withStroke(linewidth=2.0, foreground="white")])


def annotate_sites(ax, counts_df: pd.DataFrame):
    counts_map = dict(zip(counts_df["Site"], counts_df["Count"]))
    vals = list(counts_map.values())
    min_count, max_count = min(vals), max(vals)

    for site, meta in SITE_META.items():
        lon, lat = float(meta["lon"]), float(meta["lat"])
        bx, by = LABEL_BOX_POS[site]
        count = int(counts_map.get(site, 0))
        width, height = size_box(count, min_count, max_count)

        ax.scatter(
            lon, lat, s=92, marker="o", facecolor=CFG.marker_face,
            edgecolor=CFG.marker_edge, linewidth=1.5, zorder=22,
        )

        # leader line to nearest edge of box
        line_end_x = bx - width / 2 if bx > lon else bx + width / 2
        ax.plot([lon, line_end_x], [lat, by], color="#64748B", lw=1.0, zorder=21)

        draw_scaled_box(ax, bx, by, count, width, height)

        label = ax.text(
            bx, by - height / 2 - 0.045, site,
            ha="center", va="top", fontsize=8.7, fontweight="bold",
            color=CFG.site_text, zorder=25,
        )
        label.set_path_effects([pe.withStroke(linewidth=2.2, foreground="white")])


def build_map(excel_path: Path, outdir: Path, out_stem: str, district_json: Path, division_json: Path):
    counts_df = read_counts(excel_path, CFG.sheet_name)
    districts, divisions = load_geo_layers(district_json, division_json)

    site_districts = {ALIASES.get(v["district"], v["district"]) for v in SITE_META.values()}
    districts = districts.copy()
    districts["site_flag"] = districts["ADM2_EN"].isin(site_districts)

    fig, ax = plt.subplots(figsize=CFG.figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor(CFG.ocean)

    districts.plot(ax=ax, color=CFG.land, edgecolor=CFG.district_edge, linewidth=0.42, zorder=1)
    districts[districts["site_flag"]].plot(
        ax=ax, color=CFG.site_district_fill, edgecolor=CFG.district_edge, linewidth=0.55, zorder=2
    )
    divisions.boundary.plot(ax=ax, color=CFG.division_edge, linewidth=1.10, zorder=3)

    country_outline = districts.dissolve().boundary
    country_outline.plot(ax=ax, color=CFG.country_edge, linewidth=1.28, zorder=4)

    annotate_sites(ax, counts_df)
    format_axes(ax, districts.total_bounds)
    add_header(ax, int(counts_df["Count"].sum()))
    add_north_arrow(ax)

    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.07, top=0.92)
    outdir.mkdir(parents=True, exist_ok=True)
    out_base = outdir / out_stem
    fig.savefig(out_base.with_suffix(".png"), dpi=CFG.dpi, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    counts_df.to_csv(outdir / f"{out_stem}_site_counts.csv", index=False)
    plt.close(fig)



def parse_args():
    ap = argparse.ArgumentParser(description="Create a publication-grade Bangladesh mixed OP study-site map.")
    ap.add_argument("--excel", default=CFG.excel_path)
    ap.add_argument("--sheet", default=CFG.sheet_name)
    ap.add_argument("--outdir", default=CFG.out_dir)
    ap.add_argument("--stem", default=CFG.out_stem)
    ap.add_argument("--district-json", default=CFG.district_json)
    ap.add_argument("--division-json", default=CFG.division_json)
    return ap.parse_args()


def main():
    args = parse_args()
    excel_path = Path(args.excel)
    outdir = Path(args.outdir)
    district_json = ensure_geojson(Path(args.district_json), CFG.district_url)
    division_json = ensure_geojson(Path(args.division_json), CFG.division_url)
    CFG.sheet_name = args.sheet
    build_map(excel_path, outdir, args.stem, district_json, division_json)
    print(f"[OK] Saved map to {outdir}")


if __name__ == "__main__":
    main()
