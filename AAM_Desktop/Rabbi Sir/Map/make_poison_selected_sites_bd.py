# make_poison_selected_sites_bd.py
# ------------------------------------------------------------
# Generates a Bangladesh district-level site map showing ONLY:
# CMCH, RMCH, SZMCH, DMCH, SOMCH, CoxMCH
#
# No counts are used and no count box is drawn.
# Outputs: PNG (600 dpi) + PDF + SVG
# ------------------------------------------------------------

from __future__ import annotations

import re
import sys
import difflib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.lines import Line2D


# =============================
# Configuration
# =============================
@dataclass
class Config:
    # GADM (Admin level 2 = districts) GeoJSON
    gadm_adm2_url: str = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_BGD_2.json"

    out_dir: str = "poison_site_maps_out"
    dpi: int = 600
    figsize: Tuple[float, float] = (9.5, 11.5)

    # Title
    title_sites: str = "Bangladesh Study Sites (n=6)"

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
    site_color: str = "#08306B"
    site_edge: str = "#111111"
    site_edge_lw: float = 0.6
    site_size: float = 90

    # Site label style
    site_label_fs: float = 9.0
    site_label_color: str = "#08306B"
    site_label_weight: str = "bold"

    # Only requested sites
    site_to_district_user: Dict[str, str] = None


CFG = Config()
if CFG.site_to_district_user is None:
    CFG.site_to_district_user = {
        "CMCH": "Chittagong",
        "DMCH": "Dhaka",
        "RMCH": "Rajshahi",
        "SOMCH": "Sylhet",
        "SZMCH": "Bogra",
        "CoxMCH": "Cox's Bazar",
    }


# =============================
# Name normalization / matching
# =============================
def _norm_key(s: str) -> str:
    s = str(s).strip().lower().replace("&", "and")
    return re.sub(r"[^a-z0-9]", "", s)


ALIASES = {
    _norm_key("Chattogram"): _norm_key("Chittagong"),
    _norm_key("Bogura"): _norm_key("Bogra"),
    _norm_key("Barisal"): _norm_key("Barishal"),
    _norm_key("Jeshore"): _norm_key("Jessore"),
    _norm_key("Cox Bazar"): _norm_key("Cox's Bazar"),
    _norm_key("Coxs Bazar"): _norm_key("Cox's Bazar"),
}


def resolve_name(user_name: str, norm_to_real: Dict[str, str]) -> str:
    """Resolve a user-provided district name to the exact GADM district name."""
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

    return gdf.dissolve(by=["NAME_1", "NAME_2"], as_index=False)


def division_boundaries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return gdf.dissolve(by="NAME_1", as_index=False)


# =============================
# Plot helpers
# =============================
def set_lonlat_ticks(ax, gdf: gpd.GeoDataFrame) -> None:
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
    gdf.plot(
        ax=ax,
        color=CFG.base_fill,
        edgecolor=CFG.district_border_color,
        linewidth=CFG.district_border_lw,
        zorder=1,
    )

    gdf_div.boundary.plot(
        ax=ax,
        color=CFG.division_border_color,
        linewidth=CFG.division_border_lw,
        zorder=2,
    )

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


def site_marker_positions(
    gdf: gpd.GeoDataFrame,
    site_to_district_resolved: Dict[str, str],
) -> Dict[str, Tuple[float, float]]:
    reps = gdf.set_index("NAME_2").geometry.representative_point()
    out: Dict[str, Tuple[float, float]] = {}
    for site, dist_name in site_to_district_resolved.items():
        pt = reps.loc[dist_name]
        out[site] = (float(pt.x), float(pt.y))
    return out


def draw_sites(ax, site_xy: Dict[str, Tuple[float, float]]) -> None:
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

    for site, (x, y) in site_xy.items():
        t = ax.text(
            x + 0.06,
            y + 0.04,
            site,
            fontsize=CFG.site_label_fs,
            color=CFG.site_label_color,
            fontweight=CFG.site_label_weight,
            ha="left",
            va="bottom",
            zorder=7,
            clip_on=True,
        )
        t.set_path_effects([pe.Stroke(linewidth=2.0, foreground="white"), pe.Normal()])


def save_figure(fig, out_base: Path) -> None:
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), dpi=CFG.dpi, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


# =============================
# Figure builder
# =============================
def make_sites_map(
    gdf: gpd.GeoDataFrame,
    gdf_div: gpd.GeoDataFrame,
    site_xy: Dict[str, Tuple[float, float]],
    out_base: Path,
) -> None:
    fig, ax = plt.subplots(figsize=CFG.figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    plot_base_layers(ax, gdf, gdf_div)
    draw_sites(ax, site_xy)

    ax.set_title(CFG.title_sites, fontsize=14, pad=10)
    set_lonlat_ticks(ax, gdf)

    handles = [
        Line2D(
            [0], [0],
            marker=CFG.site_marker,
            color="none",
            markerfacecolor=CFG.site_color,
            markeredgecolor=CFG.site_edge,
            markersize=8,
            label="Study site",
        ),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=True, framealpha=0.96, fontsize=10)

    save_figure(fig, out_base)


# =============================
# Main
# =============================
def main() -> int:
    out_dir = Path(CFG.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gdf = load_bd_districts(CFG.gadm_adm2_url)
    gdf_div = division_boundaries(gdf)

    districts = sorted(gdf["NAME_2"].unique().tolist())
    norm_to_real_dist = {_norm_key(d): d for d in districts}

    site_to_district_resolved: Dict[str, str] = {}
    for site, dist_user in CFG.site_to_district_user.items():
        site_to_district_resolved[site] = resolve_name(dist_user, norm_to_real_dist)

    site_xy = site_marker_positions(gdf, site_to_district_resolved)

    make_sites_map(
        gdf=gdf,
        gdf_div=gdf_div,
        site_xy=site_xy,
        out_base=out_dir / "fig_bd_selected_study_sites",
    )

    print(f"[OK] Saved outputs in: {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        raise
