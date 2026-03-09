from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

# Optional geospatial backends
try:
    import geopandas as gpd
except Exception:
    gpd = None

try:
    from mpl_toolkits.basemap import Basemap
except Exception:
    Basemap = None


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
@dataclass
class Config:
    excel_path: str = r"C:\Users\User\OneDrive\2. Personal\AAM\Desktop\Mixed OP\Final\Map.xlsx"
    sheet_name: str = "Map"
    out_dir: str = r"C:\Users\User\OneDrive\2. Personal\AAM\Desktop\Mixed OP\Final"
    out_stem: str = "figure_mixed_op_study_sites_bangladesh"

    gadm_local_json: str = "gadm41_BGD_2.json"
    gadm_url: str = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_BGD_2.json"
    dpi: int = 600
    figsize: Tuple[float, float] = (8.6, 10.0)

    title: str = "Study sites and enrolled mixed OP cases in Bangladesh"
    subtitle: str = "Nine sentinel hospitals; labels show site code and participant count"

    # palette
    ocean: str = "#EAF3FB"
    land: str = "#F7F7F5"
    site_district_fill: str = "#D9E8F5"
    district_edge: str = "#8A8A8A"
    division_edge: str = "#2F4858"
    country_edge: str = "#16212B"
    marker_face: str = "#0D3B66"
    marker_edge: str = "white"
    count_box_face: str = "white"
    count_box_edge: str = "#1F2937"
    count_text: str = "#8A1538"
    label_text: str = "#0D3B66"
    legend_face: str = "white"


CFG = Config()

# Approximate hospital coordinates (city-level)
SITE_META: Dict[str, Dict[str, object]] = {
    "CMCH": {"district": "Chittagong", "lon": 91.7832, "lat": 22.3569, "name": "Chattogram Medical College Hospital"},
    "DMCH": {"district": "Dhaka", "lon": 90.4074, "lat": 23.7250, "name": "Dhaka Medical College Hospital"},
    "JMCH": {"district": "Jessore", "lon": 89.2081, "lat": 23.1664, "name": "Jashore Medical College Hospital"},
    "KMCH": {"district": "Khulna", "lon": 89.5403, "lat": 22.8456, "name": "Khulna Medical College Hospital"},
    "MMCH": {"district": "Mymensingh", "lon": 90.4203, "lat": 24.7471, "name": "Mymensingh Medical College Hospital"},
    "RMCH": {"district": "Rajshahi", "lon": 88.6042, "lat": 24.3745, "name": "Rajshahi Medical College Hospital"},
    "RpMCH": {"district": "Rangpur", "lon": 89.2752, "lat": 25.7439, "name": "Rangpur Medical College Hospital"},
    "SOMCH": {"district": "Sylhet", "lon": 91.8687, "lat": 24.8949, "name": "Sylhet MAG Osmani Medical College Hospital"},
    "SZMCH": {"district": "Bogra", "lon": 89.3776, "lat": 24.8465, "name": "Shaheed Ziaur Rahman Medical College Hospital"},
}

# Manually tuned label offsets (lon, lat) to prevent overlap and improve balance.
LABEL_OFFSETS: Dict[str, Tuple[float, float]] = {
    "CMCH": (0.34, -0.16),
    "DMCH": (0.40, -0.03),
    "JMCH": (-0.46, -0.20),
    "KMCH": (-0.42, -0.05),
    "MMCH": (0.36, 0.18),
    "RMCH": (-0.48, 0.14),
    "RpMCH": (-0.45, 0.18),
    "SOMCH": (0.30, 0.08),
    "SZMCH": (0.25, -0.12),
}

ALIASES = {
    "chattogram": "Chittagong",
    "bogura": "Bogra",
    "barisal": "Barishal",
    "jashore": "Jessore",
}


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------
def read_counts(excel_path: str, sheet_name: str) -> pd.DataFrame:
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    cols = {str(c).strip().lower(): c for c in df.columns}
    if "site" not in cols:
        raise ValueError(f"Column 'Site' not found in {excel_path} [{sheet_name}]. Columns: {list(df.columns)}")

    site_col = cols["site"]
    patient_col = cols.get("patient id")

    out = df[[site_col] + ([patient_col] if patient_col else [])].copy()
    out.columns = ["Site"] + (["Patient ID"] if patient_col else [])
    out["Site"] = out["Site"].astype(str).str.strip()
    out = out[out["Site"].isin(SITE_META)].copy()

    if patient_col:
        counts = out.groupby("Site")["Patient ID"].nunique().rename("Count").reset_index()
    else:
        counts = out.groupby("Site").size().rename("Count").reset_index()

    for site in SITE_META:
        if site not in counts["Site"].tolist():
            counts = pd.concat([counts, pd.DataFrame({"Site": [site], "Count": [0]})], ignore_index=True)

    counts["Count"] = counts["Count"].astype(int)
    counts = counts.sort_values("Site").reset_index(drop=True)
    return counts


# ---------------------------------------------------------------------
# Geo loading
# ---------------------------------------------------------------------
def _resolve_district_name(name: str) -> str:
    key = str(name).strip().lower()
    return ALIASES.get(key, name)


def try_load_gadm(local_json: str, remote_url: str):
    if gpd is None:
        return None

    candidate = Path(local_json)
    if candidate.exists():
        gdf = gpd.read_file(candidate)
        return _prep_gadm(gdf)

    # Try remote only when geopandas/fiona can reach the URL on the user's machine.
    try:
        gdf = gpd.read_file(remote_url)
        return _prep_gadm(gdf)
    except Exception:
        return None


def _prep_gadm(gdf):
    gdf = gdf.copy()
    gdf["NAME_1"] = gdf["NAME_1"].astype(str).str.strip()
    gdf["NAME_2"] = gdf["NAME_2"].astype(str).str.strip()
    gdf = gdf.dissolve(by=["NAME_1", "NAME_2"], as_index=False)
    div = gdf.dissolve(by="NAME_1", as_index=False)
    return gdf, div


# ---------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------
def add_header(ax, total_n: int):
    ax.text(
        0.5, 1.035, CFG.title,
        transform=ax.transAxes,
        ha="center", va="bottom",
        fontsize=17, fontweight="bold", color="#0F172A",
    )
    ax.text(
        0.5, 1.005, f"{CFG.subtitle} (overall n={total_n:,})",
        transform=ax.transAxes,
        ha="center", va="bottom",
        fontsize=10.5, color="#334155",
    )


def add_footer(ax):
    footer = (
        "Site labels are manually displaced with leader lines to improve readability; "
        "counts were derived from the workbook by site."
    )
    ax.text(
        0.01, -0.055, footer,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=8.5, color="#475569",
    )


def add_north_arrow(ax):
    ax.annotate(
        "N",
        xy=(0.94, 0.90), xytext=(0.94, 0.82),
        xycoords="axes fraction", textcoords="axes fraction",
        ha="center", va="center",
        fontsize=12, fontweight="bold", color="#0F172A",
        arrowprops=dict(arrowstyle="-|>", lw=1.4, color="#0F172A"),
        zorder=30,
    )


def add_site_legend(ax):
    handle = Line2D([0], [0], marker="o", linestyle="", markersize=10,
                    markerfacecolor=CFG.marker_face, markeredgecolor=CFG.marker_edge,
                    markeredgewidth=1.3)
    leg = ax.legend(
        [handle], ["Study site"],
        loc="lower left", bbox_to_anchor=(0.02, 0.02),
        frameon=True, fancybox=True, framealpha=0.96,
        facecolor=CFG.legend_face, edgecolor="#CBD5E1", fontsize=9,
    )
    leg.get_frame().set_linewidth(0.9)


def add_summary_box(ax, counts_df: pd.DataFrame):
    total = int(counts_df["Count"].sum())
    lines = ["Site counts", ""]
    order = counts_df.sort_values("Count", ascending=False)
    for _, r in order.iterrows():
        lines.append(f"{r['Site']}: {int(r['Count'])}")
    lines.append("")
    lines.append(f"Total: {total:,}")
    txt = "\n".join(lines)
    ax.text(
        0.02, 0.98, txt,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=9.0, color="#0F172A",
        bbox=dict(boxstyle="round,pad=0.40", fc="white", ec="#CBD5E1", lw=0.9, alpha=0.97),
        zorder=25,
    )


def annotate_sites(ax, counts_df: pd.DataFrame):
    counts_map = dict(zip(counts_df["Site"], counts_df["Count"]))

    for site, meta in SITE_META.items():
        lon = float(meta["lon"])
        lat = float(meta["lat"])
        dx, dy = LABEL_OFFSETS.get(site, (0.22, 0.10))
        lx, ly = lon + dx, lat + dy
        count = int(counts_map.get(site, 0))

        # site marker
        ax.scatter(
            lon, lat, s=92, marker="o",
            facecolor=CFG.marker_face, edgecolor=CFG.marker_edge,
            linewidth=1.6, zorder=20,
        )

        # leader line
        ax.plot([lon, lx], [lat, ly], color="#64748B", lw=1.0, zorder=18)

        # count badge
        count_text = ax.text(
            lx, ly + 0.035, f"{count}",
            ha="center", va="center",
            fontsize=11.0, fontweight="bold", color=CFG.count_text,
            zorder=23,
            bbox=dict(boxstyle="round,pad=0.28", fc=CFG.count_box_face, ec=CFG.count_box_edge, lw=0.85),
        )
        count_text.set_path_effects([pe.withStroke(linewidth=1.8, foreground="white")])

        # site code under the badge
        label = ax.text(
            lx, ly - 0.07, site,
            ha="center", va="top",
            fontsize=8.6, fontweight="bold", color=CFG.label_text,
            zorder=23,
        )
        label.set_path_effects([pe.withStroke(linewidth=2.2, foreground="white")])


def _format_axes(ax, xlim, ylim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")

    xticks = np.arange(np.floor(xlim[0]), np.ceil(xlim[1]) + 0.01, 1.0)
    yticks = np.arange(np.floor(ylim[0]), np.ceil(ylim[1]) + 0.01, 1.0)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([f"{x:.0f}°E" for x in xticks], fontsize=9)
    ax.set_yticklabels([f"{y:.0f}°N" for y in yticks], fontsize=9)
    ax.tick_params(top=False, right=False, labeltop=False, labelright=False, direction="out", length=3.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_color("#475569")


# ---------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------
def build_with_gadm(counts_df: pd.DataFrame, out_base: Path, gadm):
    gdf, div = gadm
    site_districts = {_resolve_district_name(meta["district"]) for meta in SITE_META.values()}
    gdf = gdf.copy()
    gdf["site_flag"] = gdf["NAME_2"].isin(site_districts)

    fig, ax = plt.subplots(figsize=CFG.figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor(CFG.ocean)

    gdf.plot(
        ax=ax,
        color=CFG.land,
        edgecolor=CFG.district_edge,
        linewidth=0.40,
        zorder=1,
    )
    gdf[gdf["site_flag"]].plot(
        ax=ax,
        color=CFG.site_district_fill,
        edgecolor=CFG.district_edge,
        linewidth=0.55,
        zorder=2,
    )
    div.boundary.plot(ax=ax, color=CFG.division_edge, linewidth=0.9, zorder=3)
    try:
        outline = gdf.unary_union
        gpd.GeoSeries([outline]).boundary.plot(ax=ax, color=CFG.country_edge, linewidth=1.2, zorder=4)
    except Exception:
        pass

    annotate_sites(ax, counts_df)

    minx, miny, maxx, maxy = gdf.total_bounds
    xpad = 0.45
    ypad = 0.35
    _format_axes(ax, (minx - xpad, maxx + xpad), (miny - ypad, maxy + ypad))

    add_header(ax, int(counts_df["Count"].sum()))
    add_summary_box(ax, counts_df)
    add_north_arrow(ax)
    add_site_legend(ax)
    add_footer(ax)

    fig.subplots_adjust(left=0.06, right=0.96, bottom=0.08, top=0.90)
    fig.savefig(out_base.with_suffix(".png"), dpi=CFG.dpi, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def build_with_basemap(counts_df: pd.DataFrame, out_base: Path):
    if Basemap is None:
        raise RuntimeError("Neither GeoPandas nor Basemap mapping backend is available.")

    fig, ax = plt.subplots(figsize=CFG.figsize)
    fig.patch.set_facecolor("white")

    m = Basemap(
        projection="merc",
        llcrnrlon=87.2, llcrnrlat=20.4,
        urcrnrlon=92.9, urcrnrlat=26.8,
        resolution="i", ax=ax,
    )
    m.drawmapboundary(fill_color=CFG.ocean, linewidth=0.0)
    m.fillcontinents(color=CFG.land, lake_color=CFG.ocean, zorder=1)
    m.drawcountries(color=CFG.country_edge, linewidth=1.0, zorder=2)
    m.drawcoastlines(color=CFG.country_edge, linewidth=0.7, zorder=2)
    m.drawrivers(color="#9CC8E7", linewidth=0.45, zorder=1)

    # faint graticule without labels; axes labels handled manually below.
    m.drawparallels(np.arange(21, 27, 1), labels=[0, 0, 0, 0], color="#D6E2EC", dashes=[1, 0], linewidth=0.5)
    m.drawmeridians(np.arange(88, 93, 1), labels=[0, 0, 0, 0], color="#D6E2EC", dashes=[1, 0], linewidth=0.5)

    # axis ticks with projected coordinates but lon/lat labels
    xt_lon = np.arange(88, 93, 1)
    yt_lat = np.arange(21, 27, 1)
    xticks = [m(lon, 20.4)[0] for lon in xt_lon]
    yticks = [m(87.2, lat)[1] for lat in yt_lat]
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([f"{lon}°E" for lon in xt_lon], fontsize=9)
    ax.set_yticklabels([f"{lat}°N" for lat in yt_lat], fontsize=9)
    ax.tick_params(top=False, right=False, labeltop=False, labelright=False, direction="out", length=3.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_color("#475569")

    # project marker and annotation points
    counts_map = dict(zip(counts_df["Site"], counts_df["Count"]))
    for site, meta in SITE_META.items():
        lon = float(meta["lon"])
        lat = float(meta["lat"])
        dx, dy = LABEL_OFFSETS.get(site, (0.22, 0.10))
        lx, ly = lon + dx, lat + dy
        count = int(counts_map.get(site, 0))

        x, y = m(lon, lat)
        lx_p, ly_p = m(lx, ly)
        ax.scatter(x, y, s=92, marker="o", facecolor=CFG.marker_face, edgecolor=CFG.marker_edge,
                   linewidth=1.6, zorder=20)
        ax.plot([x, lx_p], [y, ly_p], color="#64748B", lw=1.0, zorder=18)
        ct = ax.text(lx_p, ly_p + 15000, f"{count}", ha="center", va="center", fontsize=11.0,
                     fontweight="bold", color=CFG.count_text, zorder=23,
                     bbox=dict(boxstyle="round,pad=0.28", fc=CFG.count_box_face, ec=CFG.count_box_edge, lw=0.85))
        ct.set_path_effects([pe.withStroke(linewidth=1.8, foreground="white")])
        lab = ax.text(lx_p, ly_p - 18000, site, ha="center", va="top", fontsize=8.6,
                      fontweight="bold", color=CFG.label_text, zorder=23)
        lab.set_path_effects([pe.withStroke(linewidth=2.2, foreground="white")])

    add_header(ax, int(counts_df["Count"].sum()))
    add_summary_box(ax, counts_df)
    add_north_arrow(ax)
    add_site_legend(ax)
    add_footer(ax)

    fig.subplots_adjust(left=0.06, right=0.96, bottom=0.08, top=0.90)
    fig.savefig(out_base.with_suffix(".png"), dpi=CFG.dpi, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create a publication-grade Bangladesh site map with counts.")
    p.add_argument("--excel", default=CFG.excel_path, help="Path to the Excel workbook.")
    p.add_argument("--sheet", default=CFG.sheet_name, help="Sheet containing at least a 'Site' column.")
    p.add_argument("--outdir", default=CFG.out_dir, help="Output directory for figure files.")
    p.add_argument("--stem", default=CFG.out_stem, help="Output filename stem.")
    p.add_argument("--gadm-json", default=CFG.gadm_local_json, help="Optional local GADM Bangladesh ADM2 GeoJSON path.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    excel_path = Path(args.excel)
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_base = out_dir / args.stem

    counts_df = read_counts(str(excel_path), args.sheet)

    gadm = try_load_gadm(args.gadm_json, CFG.gadm_url)
    if gadm is not None:
        build_with_gadm(counts_df, out_base, gadm)
        backend_used = "GeoPandas/GADM"
    else:
        build_with_basemap(counts_df, out_base)
        backend_used = "Basemap fallback"

    summary_path = out_dir / f"{args.stem}_site_counts.csv"
    counts_df.sort_values("Count", ascending=False).to_csv(summary_path, index=False)

    print(f"[OK] Figure exported with {backend_used}:\n  {out_base.with_suffix('.png')}\n  {out_base.with_suffix('.pdf')}\n  {out_base.with_suffix('.svg')}\n  {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
