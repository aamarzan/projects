from __future__ import annotations

import difflib
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Tuple

import geopandas as gpd
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


@dataclass
class Config:
    # Prefer local district files first so the script works offline.
    local_district_candidates: Tuple[str, ...] = (
        "bangladesh_districts.geojson",
        "bangladesh_districts.json",
        "gadm41_BGD_2.geojson",
        "gadm41_BGD_2.json",
    )
    gadm_adm2_url: str = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_BGD_2.json"

    out_dir: str = "poison_site_maps_out"
    dpi: int = 600
    figsize: Tuple[float, float] = (10.2, 12.0)

    title_sites: str = "Bangladesh Study Sites (n=6)"
    title_fontsize: float = 18.0

    base_fill: str = "#F7F9FC"
    district_border_color: str = "#7F8C9A"
    district_border_lw: float = 0.42
    division_border_color: str = "#2C3E50"
    division_border_lw: float = 1.05
    country_border_color: str = "#111111"
    country_border_lw: float = 1.30

    tick_fontsize: int = 11
    axis_label_fontsize: int = 12
    legend_fontsize: int = 11
    n_ticks: int = 6
    tick_decimals: int = 1

    site_marker: str = "s"
    site_color: str = "#0057D9"     # brighter blue tone
    site_edge: str = "#0B2E59"
    site_edge_lw: float = 0.8
    site_size: float = 150

    site_label_fs: float = 12.5
    site_label_color: str = "#0057D9"
    site_label_weight: str = "bold"
    label_halo_lw: float = 3.0

    site_to_district_user: Dict[str, str] = field(default_factory=lambda: {
        "CMCH": "Chittagong",
        "DMCH": "Dhaka",
        "RMCH": "Rajshahi",
        "SOMCH": "Sylhet",
        "SZMCH": "Bogra",
        "CoxMCH": "Cox's Bazar",
    })

    label_offsets: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "CMCH": (0.10, 0.04),
        "DMCH": (0.10, 0.05),
        "RMCH": (0.10, 0.04),
        "SOMCH": (0.10, 0.04),
        "SZMCH": (0.10, 0.04),
        "CoxMCH": (-0.18, 0.05),
    })


CFG = Config()


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
    k = _norm_key(user_name)
    if k in ALIASES:
        k = ALIASES[k]
    if k in norm_to_real:
        return norm_to_real[k]
    close = difflib.get_close_matches(k, list(norm_to_real.keys()), n=1, cutoff=0.75)
    if close:
        return norm_to_real[close[0]]
    raise ValueError(f"Could not match name '{user_name}' to district names in the map file.")


def _find_existing_file(candidates: Iterable[str]) -> Path | None:
    for candidate in candidates:
        p = Path(candidate)
        if p.exists():
            return p
    return None


def _pick_column(columns: Iterable[str], preferred: Iterable[str]) -> str | None:
    col_map = {_norm_key(c): c for c in columns}
    for p in preferred:
        key = _norm_key(p)
        if key in col_map:
            return col_map[key]
    return None


def _standardize_district_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    cols = list(gdf.columns)
    div_col = _pick_column(cols, ["NAME_1", "division", "division_name", "adm1_en", "adm1_name", "provname"])
    dist_col = _pick_column(cols, ["NAME_2", "district", "district_name", "adm2_en", "adm2_name", "shapeName"])

    if dist_col is None:
        raise RuntimeError(
            "Could not identify the district-name column in your local GeoJSON. "
            f"Available columns: {cols}"
        )

    if div_col is None:
        gdf = gdf.copy()
        gdf["NAME_1"] = "Bangladesh"
    else:
        gdf = gdf.copy()
        gdf["NAME_1"] = gdf[div_col].astype(str).str.strip()

    gdf["NAME_2"] = gdf[dist_col].astype(str).str.strip()
    return gdf[["NAME_1", "NAME_2", "geometry"]]


def load_bd_districts() -> gpd.GeoDataFrame:
    local_path = _find_existing_file(CFG.local_district_candidates)
    if local_path is not None:
        print(f"[INFO] Using local district map file: {local_path.resolve()}")
        gdf = gpd.read_file(local_path)
        gdf = _standardize_district_columns(gdf)
        return gdf.dissolve(by=["NAME_1", "NAME_2"], as_index=False)

    print("[INFO] Local district GeoJSON not found. Trying online GADM source...")
    gdf = gpd.read_file(CFG.gadm_adm2_url)
    required = {"NAME_1", "NAME_2", "geometry"}
    missing = required - set(gdf.columns)
    if missing:
        raise RuntimeError(f"Map file missing columns: {missing}")
    gdf = gdf[["NAME_1", "NAME_2", "geometry"]].copy()
    gdf["NAME_1"] = gdf["NAME_1"].astype(str).str.strip()
    gdf["NAME_2"] = gdf["NAME_2"].astype(str).str.strip()
    return gdf.dissolve(by=["NAME_1", "NAME_2"], as_index=False)


def division_boundaries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return gdf.dissolve(by="NAME_1", as_index=False)


def set_lonlat_ticks(ax, gdf: gpd.GeoDataFrame) -> None:
    minx, miny, maxx, maxy = gdf.total_bounds
    xt = np.linspace(minx, maxx, CFG.n_ticks)
    yt = np.linspace(miny, maxy, CFG.n_ticks)

    fmt = f"{{:.{CFG.tick_decimals}f}}"
    ax.set_xticks(xt)
    ax.set_yticks(yt)
    ax.set_xticklabels([fmt.format(v) for v in xt], fontsize=CFG.tick_fontsize)
    ax.set_yticklabels([fmt.format(v) for v in yt], fontsize=CFG.tick_fontsize)

    ax.set_xlabel("Longitude", fontsize=CFG.axis_label_fontsize)
    ax.set_ylabel("Latitude", fontsize=CFG.axis_label_fontsize)

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
        pad=3,
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
        gpd.GeoSeries([outline], crs=gdf.crs).boundary.plot(
            ax=ax,
            color=CFG.country_border_color,
            linewidth=CFG.country_border_lw,
            zorder=3,
        )
    except Exception:
        pass


def site_marker_positions(gdf: gpd.GeoDataFrame, site_to_district_resolved: Dict[str, str]) -> Dict[str, Tuple[float, float]]:
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
        dx, dy = CFG.label_offsets.get(site, (0.10, 0.04))
        ha = "left" if dx >= 0 else "right"
        t = ax.text(
            x + dx,
            y + dy,
            site,
            fontsize=CFG.site_label_fs,
            color=CFG.site_label_color,
            fontweight=CFG.site_label_weight,
            ha=ha,
            va="bottom",
            zorder=7,
            clip_on=False,
        )
        t.set_path_effects([
            pe.Stroke(linewidth=CFG.label_halo_lw, foreground="white"),
            pe.Normal(),
        ])


def save_figure(fig, out_base: Path) -> None:
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), dpi=CFG.dpi, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def make_sites_map(gdf: gpd.GeoDataFrame, gdf_div: gpd.GeoDataFrame, site_xy: Dict[str, Tuple[float, float]], out_base: Path) -> None:
    fig, ax = plt.subplots(figsize=CFG.figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    plot_base_layers(ax, gdf, gdf_div)
    draw_sites(ax, site_xy)

    ax.set_title(CFG.title_sites, fontsize=CFG.title_fontsize, fontweight="bold", pad=12)
    set_lonlat_ticks(ax, gdf)

    handles = [
        Line2D(
            [0], [0],
            marker=CFG.site_marker,
            color="none",
            markerfacecolor=CFG.site_color,
            markeredgecolor=CFG.site_edge,
            markersize=9,
            label="Study site",
        ),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=True, framealpha=0.97, fontsize=CFG.legend_fontsize)

    save_figure(fig, out_base)


def main() -> int:
    out_dir = Path(CFG.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gdf = load_bd_districts()
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
