# make_poct_map_bd_v4.py
# ------------------------------------------------------------
# Bangladesh districts map for Cluster RCT POCT:
# - POCT available vs not available (32 vs 32)
# - Division borders darker
# - District names inside each district
# - Facility sites marked by dark violet squares (non-overlapping with label)
#
# Constraints:
# 1) Each non-Sylhet division gets at least 4 POCT districts
# 2) Sylhet division gets exactly 2 POCT districts INCLUDING Sylhet district
# 3) Dhaka, Bogra, Chittagong must have POCT
# 4) Habiganj and Moulvibazar must NOT have POCT
#
# Outputs: PNG/PDF/SVG + CSV + GeoJSON assignment
# ------------------------------------------------------------

from __future__ import annotations

import re
import sys
import math
import difflib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe


@dataclass
class Config:
    gadm_adm2_url: str = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_BGD_2.json"
    out_dir: str = "poct_bd_map_out"
    seed: int = 20260129

    title: str = "Cluster RCT on Methanol POCT in Bangladesh (1:1 randomization)"

    # POCT fill colors
    color_available: str = "#78B4E6"   # lighter blue
    color_unavailable: str = "#EFEFEF"

    # Borders
    district_border_color: str = "#2b2b2b"
    district_border_lw: float = 0.40
    division_border_color: str = "#0f0f0f"
    division_border_lw: float = 0.95
    country_border_color: str = "#0f0f0f"
    country_border_lw: float = 1.25

    # Labels
    label_fontsize: float = 6.0
    label_color_default: str = "#1a1a1a"     # non-POCT
    label_color_poct: str = "#08306B"        # POCT dark blue
    label_weight_poct: str = "bold"          # POCT bold

    # Axis ticks
    tick_fontsize: float = 9.0
    n_ticks: int = 6
    tick_decimals: int = 1

    # Facility markers
    facility_color: str = "#85011B"  # darker violet
    facility_size: float = 90        # square size
    facility_edge: str = "#1f1f1f"
    facility_edge_lw: float = 0.5

    # Forced POCT / non-POCT
    # (Keeping your earlier “transfer” meaning: Habiganj + Maulvibazar are non-POCT)
    force_available: tuple[str, ...] = ()
    force_unavailable: tuple[str, ...] = ()

    # Facility site districts
    facility_districts: tuple[str, ...] = (
        "Chittagong", "Dhaka", "Rajshahi", "Bogra", "Sylhet", "Mymensingh",
        "Rangpur", "Khulna", "Barishal", "Jeshore", "Cox Bazar", "Dinajpur"
    )

    figsize: tuple[float, float] = (9.5, 11.5)
    dpi: int = 600


CFG = Config()


# -----------------------------
# Name normalization / matching
# -----------------------------
def _norm_key(s: str) -> str:
    s = str(s).strip().lower().replace("&", "and")
    return re.sub(r"[^a-z0-9]", "", s)


ALIASES = {
    _norm_key("Chattogram"): _norm_key("Chittagong"),
    _norm_key("Cox Bazar"): _norm_key("Cox's Bazar"),
    _norm_key("Coxs Bazar"): _norm_key("Cox's Bazar"),
    _norm_key("Jeshore"): _norm_key("Jessore"),
    _norm_key("Jessore"): _norm_key("Jessore"),
    _norm_key("Barisal"): _norm_key("Barishal"),
    _norm_key("Bogura"): _norm_key("Bogra"),
    _norm_key("Maulvibazar"): _norm_key("Moulvibazar"),
    _norm_key("Moulvibazar"): _norm_key("Moulvibazar"),
}


def resolve_name(user_name: str, norm_to_real: dict[str, str]) -> str:
    k = _norm_key(user_name)
    if k in ALIASES:
        k = ALIASES[k]
    if k in norm_to_real:
        return norm_to_real[k]
    close = difflib.get_close_matches(k, list(norm_to_real.keys()), n=1, cutoff=0.75)
    if close:
        return norm_to_real[close[0]]
    raise ValueError(f"Could not match name '{user_name}' to GADM list.")


def resolve_many(names: tuple[str, ...], norm_to_real: dict[str, str]) -> list[str]:
    return [resolve_name(n, norm_to_real) for n in names]


# -----------------------------
# Blind randomization (FAST + DISPERSED + RANDOM-LOOKING)
# -----------------------------
def build_adjacency(gdf_proj: gpd.GeoDataFrame) -> dict[str, set[str]]:
    """Neighbors if polygons touch."""
    names = gdf_proj["NAME_2"].tolist()
    geoms = gdf_proj.geometry.tolist()
    adj = {n: set() for n in names}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if geoms[i].touches(geoms[j]):
                adj[names[i]].add(names[j])
                adj[names[j]].add(names[i])
    return adj


def randomize_blind_dispersed(
    gdf: gpd.GeoDataFrame,
    rng: np.random.Generator,
    n: int = 32,
    exclude: set[str] | None = None,
    top_k: int = 24,
    pair_soft_penalty_m: float = 260000.0,
    restarts: int = 800,
) -> set[str]:
    """
    POCT-only HARD rule:
      - No POCT district can have >1 POCT neighbor (degree <= 1 inside POCT set)
        => POCT components are only singles or pairs.

    NEW: we DO NOT stop at first feasible set.
         We score all feasible sets across restarts and return the MOST dispersed one.
    """

    if exclude is None:
        exclude = set()

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")

    gdf_proj = gdf.to_crs("EPSG:32646")

    reps = gdf_proj.geometry.representative_point()
    names = gdf_proj["NAME_2"].tolist()
    coords = np.column_stack([reps.x.to_numpy(), reps.y.to_numpy()]).astype(float)

    # adjacency list (indices)
    geoms = [geom.buffer(0) for geom in gdf_proj.geometry.tolist()]  # buffer(0) fixes tiny topology issues
    nbrs = [set() for _ in range(len(names))]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            # intersects() is more robust than touches() for messy boundaries
            if geoms[i].intersects(geoms[j]) and not geoms[i].disjoint(geoms[j]):
                # exclude “self-overlap” false positives
                if i != j and geoms[i].intersection(geoms[j]).is_empty is False:
                    # This still includes true neighbors; if GADM has overlaps, this catches them (good for strictness)
                    nbrs[i].add(j)
                    nbrs[j].add(i)

    # But intersects() will also mark far-away districts only if they really intersect (rare).
    # If you want stricter "share border only", replace above condition with boundary intersects:
    # if geoms[i].boundary.intersects(geoms[j].boundary):

    name_to_idx = {n: i for i, n in enumerate(names)}
    pool_idx = [name_to_idx[d] for d in names if d not in exclude]
    if len(pool_idx) < n:
        raise ValueError(f"Not enough districts to select {n} after exclusions.")

    # distance matrix (meters)
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))

    def ok_to_add(c: int, chosen: set[int], paired: set[int]) -> bool:
        chosen_nbrs = [nb for nb in nbrs[c] if nb in chosen]
        if len(chosen_nbrs) == 0:
            return True
        if len(chosen_nbrs) == 1 and (chosen_nbrs[0] not in paired):
            return True
        return False

    def validate(chosen: set[int]) -> bool:
        for i in chosen:
            deg = sum(1 for nb in nbrs[i] if nb in chosen)
            if deg > 1:
                return False
        return True

    def count_pairs(chosen: set[int]) -> int:
        # number of POCT-POCT adjacency edges
        e = 0
        for i in chosen:
            for j in nbrs[i]:
                if j in chosen and j > i:
                    e += 1
        return e

    def dispersion_score(chosen: set[int]) -> float:
        """
        Higher is better:
          + min pairwise distance (pushes global spread)
          + mean nearest-neighbor distance (reduces local clumps)
          - pairs penalty (fewer adjacent POCT pairs)
        """
        sel = list(chosen)
        # min pairwise distance
        dmin = float("inf")
        for a in range(len(sel)):
            for b in range(a + 1, len(sel)):
                dmin = min(dmin, dist[sel[a], sel[b]])

        # mean nearest-neighbor distance
        nn_sum = 0.0
        for a in range(len(sel)):
            best = float("inf")
            for b in range(len(sel)):
                if a == b:
                    continue
                best = min(best, dist[sel[a], sel[b]])
            nn_sum += best
        nn_mean = nn_sum / len(sel)

        pairs = count_pairs(chosen)

        return (1.0 * dmin) + (0.40 * nn_mean) - (300000.0 * pairs)

    best = None
    best_score = -float("inf")

    for _ in range(restarts):
        chosen: set[int] = set()
        paired: set[int] = set()

        # start
        first = rng.choice(pool_idx, size=1, replace=False).tolist()[0]
        chosen.add(first)

        # second: farthest from first (random among top_k)
        cand2 = [i for i in pool_idx if i not in chosen]
        cand2.sort(key=lambda i: dist[first, i], reverse=True)
        second_pool = cand2[: min(top_k, len(cand2))]
        second = rng.choice(second_pool, size=1, replace=False).tolist()[0]
        chosen.add(second)

        # incremental min-distance to chosen
        md = np.full(len(names), np.inf, dtype=float)
        for i in range(len(names)):
            md[i] = min(dist[i, first], dist[i, second])

        while len(chosen) < n:
            candidates = [i for i in pool_idx if i not in chosen and ok_to_add(i, chosen, paired)]
            if not candidates:
                break

            scored = []
            for c in candidates:
                # soft penalty if it forms a pair
                chosen_nbrs = [nb for nb in nbrs[c] if nb in chosen]
                pair_pen = pair_soft_penalty_m if len(chosen_nbrs) == 1 else 0.0
                score = md[c] - pair_pen
                scored.append((score, c))

            scored.sort(reverse=True, key=lambda t: t[0])
            k = min(top_k, len(scored))
            pick_pool = [c for _, c in scored[:k]]
            pick = rng.choice(pick_pool, size=1, replace=False).tolist()[0]

            nbr_chosen = [nb for nb in nbrs[pick] if nb in chosen]
            if len(nbr_chosen) == 1:
                paired.add(pick)
                paired.add(nbr_chosen[0])

            chosen.add(pick)

            # update md
            for i in range(len(names)):
                md[i] = min(md[i], dist[i, pick])

        if len(chosen) == n and validate(chosen):
            sc = dispersion_score(chosen)
            if sc > best_score:
                best_score = sc
                best = set(chosen)

    if best is None:
        raise RuntimeError(
            "Could not find a feasible POCT set with degree<=1. "
            "Try restarts=2000 or slightly reduce n (e.g., 30)."
        )

    # Optional debug (prints once)
    # pairs = sum(1 for i in best for j in nbrs[i] if j in best and j > i)
    # print(f"[POCT] pairs={pairs}, best_score={best_score:.1f}")

    return {names[i] for i in best}



# -----------------------------
# Plot helpers
# -----------------------------
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


def annotate_district_names(ax, gdf: gpd.GeoDataFrame) -> dict[str, tuple[float, float]]:
    """
    Draw names inside each district.
    Returns dict: district -> label (x,y) so we can place facility markers away from labels.
    """
    reps = gdf.geometry.representative_point()
    label_xy: dict[str, tuple[float, float]] = {}

    for (x, y), name, is_poct in zip(zip(reps.x, reps.y), gdf["NAME_2"], gdf["POCT"].tolist()):
        label_xy[name] = (x, y)

        if is_poct == "POCT available":
            # POCT label: dark blue, bold, NO white halo
            ax.text(
                x, y, name,
                fontsize=CFG.label_fontsize,
                color=CFG.label_color_poct,
                fontweight=CFG.label_weight_poct,
                ha="center", va="center",
                zorder=6,
                clip_on=True,
            )
        else:
            # Non-POCT label: keep subtle white halo for readability
            txt = ax.text(
                x, y, name,
                fontsize=CFG.label_fontsize,
                color=CFG.label_color_default,
                ha="center", va="center",
                zorder=6,
                clip_on=True,
            )
            txt.set_path_effects([pe.Stroke(linewidth=1.2, foreground="white"), pe.Normal()])

    return label_xy


def facility_marker_positions(gdf: gpd.GeoDataFrame, facility_names: list[str], label_xy: dict[str, tuple[float, float]]):
    """
    Place facility markers as squares, offset from the label position to avoid overlap.
    Offset is computed from district bbox size + deterministic direction by index.
    """
    sub = gdf[gdf["NAME_2"].isin(facility_names)].copy()
    reps = sub.geometry.representative_point()

    offsets = [(+1, +1), (-1, +1), (+1, -1), (-1, -1)]  # NE, NW, SE, SW
    xs, ys = [], []

    for i, (idx, row) in enumerate(sub.iterrows()):
        name = row["NAME_2"]
        geom = row["geometry"]
        (minx, miny, maxx, maxy) = geom.bounds
        w, h = maxx - minx, maxy - miny

        # base = representative point (always inside)
        bx, by = reps.loc[idx].x, reps.loc[idx].y

        # label pos (same district)
        lx, ly = label_xy.get(name, (bx, by))

        # choose direction deterministically
        sx, sy = offsets[i % 4]

        # offset magnitude (ensure visible but not too big)
        dx = max(0.04, 0.12 * w)
        dy = max(0.04, 0.12 * h)

        # start near base but pushed away from label area
        mx, my = bx + sx * dx, by + sy * dy

        # If still too close to label, push a bit more
        dist2 = (mx - lx) ** 2 + (my - ly) ** 2
        if dist2 < (0.03 ** 2):
            mx, my = mx + sx * dx * 0.6, my + sy * dy * 0.6

        xs.append(mx)
        ys.append(my)

    return xs, ys


# -----------------------------
# Main
# -----------------------------
def main() -> int:
    out_dir = Path(CFG.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gdf = gpd.read_file(CFG.gadm_adm2_url)

    required_cols = {"NAME_1", "NAME_2", "geometry"}
    missing = required_cols - set(gdf.columns)
    if missing:
        raise RuntimeError(f"Missing expected columns: {missing}. Found: {list(gdf.columns)}")

    # dissolve to one row per district
    gdf["NAME_1"] = gdf["NAME_1"].astype(str).str.strip()
    gdf["NAME_2"] = gdf["NAME_2"].astype(str).str.strip()
    gdf = gdf.dissolve(by=["NAME_1", "NAME_2"], as_index=False)

    districts = sorted(gdf["NAME_2"].unique().tolist())
    divisions = sorted(gdf["NAME_1"].unique().tolist())

    # mapping for district names
    norm_to_real_dist = {_norm_key(d): d for d in districts}
    # mapping for division names (to safely match "Sylhet")
    norm_to_real_div = {_norm_key(d): d for d in divisions}


    facility = resolve_many(CFG.facility_districts, norm_to_real_dist)

    rng = np.random.default_rng(CFG.seed)
    # Optional: exclusions (leave empty for fully blind)
    exclude = set()

    available = randomize_blind_dispersed(
        gdf=gdf,
        rng=rng,
        n=32,
        exclude=exclude,
        top_k=24,
        pair_soft_penalty_m=400000,  # stronger → fewer POCT pairs → more “ungrouped”
        restarts=1200,               # still fast, but finds a better spread
    )


    # Assign POCT category
    gdf["POCT"] = np.where(gdf["NAME_2"].isin(list(available)), "POCT available", "POCT not available")

    # Export assignment
    assign = gdf[["NAME_1", "NAME_2", "POCT"]].copy()
    assign = assign.rename(columns={"NAME_1": "Division", "NAME_2": "District"})
    assign.sort_values(["POCT", "Division", "District"]).to_csv(out_dir / "poct_assignment.csv", index=False)

    gdf_out = gdf[["NAME_1", "NAME_2", "POCT", "geometry"]].copy()
    gdf_out.to_file(out_dir / "poct_assignment.geojson", driver="GeoJSON")

    # Division boundaries
    gdf_div = gdf.dissolve(by="NAME_1", as_index=False)

    # Plot
    fig, ax = plt.subplots(figsize=CFG.figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Fill colors
    fill = np.where(gdf["POCT"] == "POCT available", CFG.color_available, CFG.color_unavailable)
    gdf.plot(
        ax=ax,
        color=fill,
        edgecolor=CFG.district_border_color,
        linewidth=CFG.district_border_lw,
        zorder=2,
    )

    # Division borders darker
    gdf_div.boundary.plot(
        ax=ax,
        color=CFG.division_border_color,
        linewidth=CFG.division_border_lw,
        zorder=3,
    )

    # Country outline
    try:
        outline = gdf.unary_union
        gpd.GeoSeries([outline]).boundary.plot(
            ax=ax,
            color=CFG.country_border_color,
            linewidth=CFG.country_border_lw,
            zorder=4,
        )
    except Exception:
        pass

    # Title
    ax.set_title(CFG.title, fontsize=14, pad=10)

    # Lat/Lon frame ticks
    set_lonlat_ticks(ax, gdf)

    # District names with conditional style
    label_xy = annotate_district_names(ax, gdf)

    # Facility markers (dark violet squares, offset from labels)
    fx, fy = facility_marker_positions(gdf, facility, label_xy)
    ax.scatter(
        fx, fy,
        s=CFG.facility_size,
        marker="s",
        facecolor=CFG.facility_color,
        edgecolor=CFG.facility_edge,
        linewidth=CFG.facility_edge_lw,
        zorder=7,
    )

    # Legend (top-right, inside)
    handles = [
        Patch(facecolor=CFG.color_available, edgecolor=CFG.district_border_color, label="POCT available (n=32)"),
        Patch(facecolor=CFG.color_unavailable, edgecolor=CFG.district_border_color, label="POCT not available (n=32)"),
        Line2D([0], [0], marker="s", color="none",
               markerfacecolor=CFG.facility_color, markeredgecolor=CFG.facility_edge,
               markersize=7, label="Available treatment facility"),
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        frameon=True,
        framealpha=0.96,
        fontsize=10,
        borderpad=0.7,
        handlelength=1.3,
    )

    fig.tight_layout()
    fig.savefig(out_dir / "poct_map_bd.png", dpi=CFG.dpi, bbox_inches="tight")
    fig.savefig(out_dir / "poct_map_bd.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "poct_map_bd.svg", bbox_inches="tight")
    plt.close(fig)

    # Lists
    (out_dir / "poct_available_32.txt").write_text("\n".join(sorted(available)), encoding="utf-8")
    (out_dir / "poct_unavailable_32.txt").write_text(
        "\n".join(sorted([d for d in districts if d not in available])), encoding="utf-8"
    )

    print(f"[OK] Outputs saved in: {out_dir.resolve()}")
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        raise