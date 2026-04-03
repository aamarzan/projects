# make_poison_site_maps_bd.py
# ------------------------------------------------------------
# Generates 3 Bangladesh figures (district-level):
# 1) Map of 10 study sites (dark blue squares + labels)
# 2) Choropleth map from Excel sheet "alphos" (Counts by site-district)
# 3) Choropleth map from Excel sheet "paraquat" (Counts by site-district)
#
# Excel expected format (per sheet): columns ["Site", "Count"]
# Example sheets: alphos, paraquat
#
# Outputs: PNG (600 dpi) + PDF + SVG for each figure
# ------------------------------------------------------------

from __future__ import annotations

import re
import sys
import difflib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
import matplotlib.patheffects as pe


# =============================
# Configuration
# =============================
@dataclass
class Config:
    # GADM (Admin level 2 = districts) GeoJSON
    gadm_adm2_url: str = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_BGD_2.json"

    # Set this to your Windows path when running locally:
    # excel_path: str = r"C:\Users\User\OneDrive\2. Personal\AAM\Desktop\Bi-weeky\poison_map.xlsx"
    excel_path: str = "poison_map.xlsx"

    out_dir: str = "poison_site_maps_out"
    dpi: int = 600
    figsize: Tuple[float, float] = (9.5, 11.5)

    # Titles
    title_sites: str = "Bangladesh Study Sites (n=10)"
    title_alphos: str = "Aluminium Phosphide (AlP) Cases by Study Site (Bangladesh)"
    title_paraquat: str = "Paraquat Cases by Study Site (Bangladesh)"

    # Count box (top-right inside axes)
    countbox_fontsize: float = 15.0
    countbox_pad: float = 0.5
    countbox_facecolor: str = "white"
    countbox_alpha: float = 0.92
    countbox_edgecolor: str = "#111111"
    countbox_linewidth: float = 0.8

    # Base fills/borders
    base_fill: str = "#F2F2F2"
    district_border_color: str = "#2b2b2b"
    district_border_lw: float = 0.40

    division_border_color: str = "#0f0f0f"
    division_border_lw: float = 0.95

    country_border_color: str = "#0f0f0f"
    country_border_lw: float = 1.25

    # Axis ticks (lon/lat)
    tick_fontsize: int = 9
    n_ticks: int = 6
    tick_decimals: int = 1

    # Site marker style
    site_marker: str = "s"
    site_color: str = "#08306B"  # dark blue
    site_edge: str = "#111111"
    site_edge_lw: float = 0.6
    site_size: float = 90  # marker area

    # Site label style
    site_label_fs: float = 9.0
    site_label_color: str = "#08306B"
    site_label_weight: str = "bold"

    # Choropleth colormap
    cmap: str = "Blues"

    # --- IMPORTANT EDIT POINT ---
    # Map each site code to its district name (edit here if needed).
    site_to_district_user: Dict[str, str] = None  # filled in __post_init__


CFG = Config()
if CFG.site_to_district_user is None:
    CFG.site_to_district_user = {
        "CMCH": "Chittagong",
        "DMCH": "Dhaka",
        "JMCH": "Jessore",
        "KMCH": "Khulna",
        "MMCH": "Mymensingh",
        "RMCH": "Rajshahi",
        "RpMCH": "Rangpur",
        "SBMCH": "Barishal",
        "SOMCH": "Sylhet",
        "SZMCH": "Bogra",
    }


# =============================
# Name normalization / matching
# =============================
def _norm_key(s: str) -> str:
    s = str(s).strip().lower().replace("&", "and")
    return re.sub(r"[^a-z0-9]", "", s)


# District name aliases to handle GADM differences (Bogura vs Bogra etc.)
ALIASES = {
    _norm_key("Chattogram"): _norm_key("Chittagong"),
    _norm_key("Bogura"): _norm_key("Bogra"),
    _norm_key("Barisal"): _norm_key("Barishal"),
    _norm_key("Jeshore"): _norm_key("Jessore"),
    _norm_key("Jessore"): _norm_key("Jessore"),
    _norm_key("Cox Bazar"): _norm_key("Cox's Bazar"),
    _norm_key("Coxs Bazar"): _norm_key("Cox's Bazar"),
}


def resolve_name(user_name: str, norm_to_real: Dict[str, str]) -> str:
    """
    Resolve a user-provided district name to the exact NAME_2 in GADM.
    Uses aliases + fuzzy match.
    """
    k = _norm_key(user_name)
    if k in ALIASES:
        k = ALIASES[k]
    if k in norm_to_real:
        return norm_to_real[k]
    close = difflib.get_close_matches(k, list(norm_to_real.keys()), n=1, cutoff=0.75)
    if close:
        return norm_to_real[close[0]]
    raise ValueError(f"Could not match name '{user_name}' to GADM districts.")


# =============================
# Geo loading
# =============================
def load_bd_districts(gadm_url: str) -> gpd.GeoDataFrame:
    """
    Load Bangladesh districts (GADM ADM2), dissolve to one row per district,
    keep division field NAME_1 and district field NAME_2.
    """
    gdf = gpd.read_file(gadm_url)

    required = {"NAME_1", "NAME_2", "geometry"}
    missing = required - set(gdf.columns)
    if missing:
        raise RuntimeError(f"GADM file missing columns: {missing}")

    gdf["NAME_1"] = gdf["NAME_1"].astype(str).str.strip()
    gdf["NAME_2"] = gdf["NAME_2"].astype(str).str.strip()

    # dissolve so each district is one polygon
    gdf = gdf.dissolve(by=["NAME_1", "NAME_2"], as_index=False)

    return gdf


def division_boundaries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Create division boundaries from district layer."""
    return gdf.dissolve(by="NAME_1", as_index=False)


# =============================
# Excel reading
# =============================
def read_site_counts(excel_path: str, sheet_name: str) -> pd.DataFrame:
    """
    Read a sheet with columns: Site, Count
    Returns a cleaned DataFrame with those columns.
    """
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    cols = {c.lower().strip(): c for c in df.columns}

    if "site" not in cols or "count" not in cols:
        raise ValueError(f"Sheet '{sheet_name}' must have columns 'Site' and 'Count'. Found: {list(df.columns)}")

    out = df[[cols["site"], cols["count"]]].copy()
    out.columns = ["Site", "Count"]
    out["Site"] = out["Site"].astype(str).str.strip()
    out["Count"] = pd.to_numeric(out["Count"], errors="coerce").fillna(0).astype(int)
    return out


def attach_counts_to_districts(
    gdf: gpd.GeoDataFrame,
    site_counts: pd.DataFrame,
    site_to_district_resolved: Dict[str, str],
) -> gpd.GeoDataFrame:
    """
    Adds a 'Count' column to the district GeoDataFrame.
    Only the 10 site districts get numeric values; others are NaN.
    """
    # map each Site -> District (resolved)
    site_counts = site_counts.copy()
    site_counts["District"] = site_counts["Site"].map(site_to_district_resolved)

    if site_counts["District"].isna().any():
        missing = site_counts.loc[site_counts["District"].isna(), "Site"].tolist()
        raise ValueError(f"Some site codes are missing from site_to_district mapping: {missing}")

    # Aggregate by District (in case duplicates happen)
    agg = site_counts.groupby("District", as_index=False)["Count"].sum()

    gdf2 = gdf.copy()
    gdf2["Count"] = np.nan
    for _, row in agg.iterrows():
        gdf2.loc[gdf2["NAME_2"] == row["District"], "Count"] = int(row["Count"])
    return gdf2


# =============================
# Plot helpers
# =============================
def set_lonlat_ticks(ax, gdf: gpd.GeoDataFrame) -> None:
    """Add lon/lat ticks around frame similar to your previous maps."""
    minx, miny, maxx, maxy = gdf.total_bounds
    xt = np.linspace(minx, maxx, CFG.n_ticks)
    yt = np.linspace(miny, maxy, CFG.n_ticks)

    fmt = f"{{:.{CFG.tick_decimals}f}}"
    ax.set_xticks(xt)
    ax.set_yticks(yt)
    ax.set_xticklabels([fmt.format(v) for v in xt], fontsize=CFG.tick_fontsize)
    ax.set_yticklabels([fmt.format(v) for v in yt], fontsize=CFG.tick_fontsize)

    ax.set_xlabel("Longitude", fontsize=CFG.tick_fontsize + 1)
    ax.set_ylabel("Latitude", fontsize=CFG.tick_fontsize + 1)

    ax.tick_params(
        axis="both",
        which="both",
        direction="out",
        top=True,
        right=True,
        labeltop=True,
        labelright=True,
        length=3.0,
        width=0.8,
        pad=2,
    )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color("#111111")


def plot_base_layers(ax, gdf: gpd.GeoDataFrame, gdf_div: gpd.GeoDataFrame) -> None:
    """Draw base district polygons + division borders + outer border."""
    # Base districts
    gdf.plot(
        ax=ax,
        color=CFG.base_fill,
        edgecolor=CFG.district_border_color,
        linewidth=CFG.district_border_lw,
        zorder=1,
    )

    # Division borders
    gdf_div.boundary.plot(
        ax=ax,
        color=CFG.division_border_color,
        linewidth=CFG.division_border_lw,
        zorder=2,
    )

    # Country outline
    try:
        outline = gdf.unary_union
        gpd.GeoSeries([outline]).boundary.plot(
            ax=ax,
            color=CFG.country_border_color,
            linewidth=CFG.country_border_lw,
            zorder=3,
        )
    except Exception:
        pass


def site_marker_positions(gdf: gpd.GeoDataFrame, site_to_district_resolved: Dict[str, str]) -> Dict[str, Tuple[float, float]]:
    """
    Compute a representative point inside each site district polygon.
    Returns: site_code -> (x, y)
    """
    reps = gdf.set_index("NAME_2").geometry.representative_point()
    out: Dict[str, Tuple[float, float]] = {}
    for site, dist_name in site_to_district_resolved.items():
        pt = reps.loc[dist_name]
        out[site] = (float(pt.x), float(pt.y))
    return out


def draw_sites(ax, site_xy: Dict[str, Tuple[float, float]]) -> None:
    """Draw squares + site labels (codes) near each marker."""
    xs = [xy[0] for xy in site_xy.values()]
    ys = [xy[1] for xy in site_xy.values()]

    ax.scatter(
        xs,
        ys,
        s=CFG.site_size,
        marker=CFG.site_marker,
        facecolor=CFG.site_color,
        edgecolor=CFG.site_edge,
        linewidth=CFG.site_edge_lw,
        zorder=6,
    )

    # labels (slight offset for readability)
    for site, (x, y) in site_xy.items():
        t = ax.text(
            x + 0.06, y + 0.04,
            site,
            fontsize=CFG.site_label_fs,
            color=CFG.site_label_color,
            fontweight=CFG.site_label_weight,
            ha="left", va="bottom",
            zorder=7,
            clip_on=True,
        )
        # subtle white halo so label is readable over borders
        t.set_path_effects([pe.Stroke(linewidth=2.0, foreground="white"), pe.Normal()])


def save_figure(fig, out_base: Path) -> None:
    """Save PNG/PDF/SVG with consistent settings."""
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), dpi=CFG.dpi, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


# =============================
# Figure builders
# =============================
def make_sites_map(gdf: gpd.GeoDataFrame, gdf_div: gpd.GeoDataFrame, site_xy: Dict[str, Tuple[float, float]], out_base: Path) -> None:
    fig, ax = plt.subplots(figsize=CFG.figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    plot_base_layers(ax, gdf, gdf_div)
    draw_sites(ax, site_xy)

    ax.set_title(CFG.title_sites, fontsize=14, pad=10)
    set_lonlat_ticks(ax, gdf)

    # Legend (top-right inside)
    handles = [
        Line2D([0], [0], marker=CFG.site_marker, color="none",
               markerfacecolor=CFG.site_color, markeredgecolor=CFG.site_edge,
               markersize=8, label="Study site"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=True, framealpha=0.96, fontsize=10)

    save_figure(fig, out_base)

def add_count_box(
    ax,
    site_counts: pd.DataFrame,
    title: Optional[str] = None,
) -> None:
    """
    Draw a small 'Site: Count' box at the top-right inside axes (not touching border).
    Expects df columns: Site, Count
    """
    df = site_counts.copy()
    df["Site"] = df["Site"].astype(str).str.strip()
    df["Count"] = pd.to_numeric(df["Count"], errors="coerce").fillna(0).astype(int)

    # Sort: highest count first (more readable)
    df = df.sort_values("Count", ascending=False)

    lines = []
    if title:
        lines.append(str(title))
    for _, r in df.iterrows():
        lines.append(f"{r['Site']}: {r['Count']}")
    text = "\n".join(lines)

    ax.text(
        0.985, 0.985, text,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=CFG.countbox_fontsize,
        bbox=dict(
            boxstyle=f"round,pad={CFG.countbox_pad}",
            facecolor=CFG.countbox_facecolor,
            alpha=CFG.countbox_alpha,
            edgecolor=CFG.countbox_edgecolor,
            linewidth=CFG.countbox_linewidth,
        ),
        zorder=20,
    )

def make_counts_map(
    gdf_counts: gpd.GeoDataFrame,
    gdf_div: gpd.GeoDataFrame,
    site_xy: Dict[str, Tuple[float, float]],
    title: str,
    out_base: Path,
    site_counts_df: pd.DataFrame,
) -> None:
    fig, ax = plt.subplots(figsize=CFG.figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Choropleth: only site districts have Count values; others are NaN
    vmax = float(np.nanmax(gdf_counts["Count"].to_numpy())) if np.isfinite(np.nanmax(gdf_counts["Count"].to_numpy())) else 1.0
    norm = Normalize(vmin=0, vmax=max(vmax, 1.0))

    gdf_counts.plot(
        ax=ax,
        column="Count",
        cmap=CFG.cmap,
        norm=norm,
        missing_kwds={"color": CFG.base_fill},
        edgecolor=CFG.district_border_color,
        linewidth=CFG.district_border_lw,
        zorder=1,
    )

    # Division borders + outline
    gdf_div.boundary.plot(ax=ax, color=CFG.division_border_color, linewidth=CFG.division_border_lw, zorder=2)
    try:
        outline = gdf_counts.unary_union
        gpd.GeoSeries([outline]).boundary.plot(ax=ax, color=CFG.country_border_color, linewidth=CFG.country_border_lw, zorder=3)
    except Exception:
        pass

    # Sites overlay
    draw_sites(ax, site_xy)

    ax.set_title(title, fontsize=14, pad=10)
    set_lonlat_ticks(ax, gdf_counts)
    add_count_box(ax, site_counts_df)
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(CFG.cmap), norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Count", fontsize=10)

    # Legend for sites
    handles = [
        Line2D([0], [0], marker=CFG.site_marker, color="none",
               markerfacecolor=CFG.site_color, markeredgecolor=CFG.site_edge,
               markersize=8, label="Study site"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=True, framealpha=0.96, fontsize=10)

    save_figure(fig, out_base)


# =============================
# Main
# =============================
def main() -> int:
    out_dir = Path(CFG.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load GADM districts
    gdf = load_bd_districts(CFG.gadm_adm2_url)
    gdf_div = division_boundaries(gdf)

    # Build resolver dict for districts
    districts = sorted(gdf["NAME_2"].unique().tolist())
    norm_to_real_dist = {_norm_key(d): d for d in districts}

    # Resolve site->district mapping to exact GADM spellings
    site_to_district_resolved: Dict[str, str] = {}
    for site, dist_user in CFG.site_to_district_user.items():
        site_to_district_resolved[site] = resolve_name(dist_user, norm_to_real_dist)

    # Compute marker positions once
    site_xy = site_marker_positions(gdf, site_to_district_resolved)

    # ---- Figure 1: Sites map
    make_sites_map(
        gdf=gdf,
        gdf_div=gdf_div,
        site_xy=site_xy,
        out_base=out_dir / "fig1_bd_study_sites",
    )

    # ---- Figure 2: alphos choropleth
    alphos = read_site_counts(CFG.excel_path, "alphos")
    gdf_alphos = attach_counts_to_districts(gdf, alphos, site_to_district_resolved)
    make_counts_map(
        gdf_counts=gdf_alphos,
        gdf_div=gdf_div,
        site_xy=site_xy,
        title=CFG.title_alphos,
        out_base=out_dir / "fig2_bd_alphos_counts",
        site_counts_df=alphos,
    )

    # ---- Figure 3: paraquat choropleth
    paraquat = read_site_counts(CFG.excel_path, "paraquat")
    gdf_paraquat = attach_counts_to_districts(gdf, paraquat, site_to_district_resolved)
    make_counts_map(
        gdf_counts=gdf_paraquat,
        gdf_div=gdf_div,
        site_xy=site_xy,
        title=CFG.title_paraquat,
        out_base=out_dir / "fig3_bd_paraquat_counts",
        site_counts_df=paraquat,
    )

    print(f"[OK] Saved outputs in: {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        raise
