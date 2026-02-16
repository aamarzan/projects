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
    force_available: tuple[str, ...] = ("Dhaka", "Bogra", "Chittagong", "Gazipur", "Narayanganj", "Sylhet")
    force_unavailable: tuple[str, ...] = ("Habiganj", "Maulvibazar")

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
# Randomization with constraints (DISPERSED / LESS CLUSTERED)
# -----------------------------
def build_adjacency(gdf_proj: gpd.GeoDataFrame) -> dict[str, set[str]]:
    """
    District adjacency graph: two districts are neighbors if they touch (share boundary).
    n~64, so O(n^2) is totally fine.
    """
    names = gdf_proj["NAME_2"].tolist()
    geoms = gdf_proj.geometry.tolist()
    adj = {n: set() for n in names}

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if geoms[i].touches(geoms[j]):
                adj[names[i]].add(names[j])
                adj[names[j]].add(names[i])
    return adj


def min_distance_to_selected(pt, pts: dict[str, tuple[float, float]], selected: set[str]) -> float:
    """
    Minimum Euclidean distance (meters) from pt to any selected district point.
    """
    if not selected:
        return float("inf")
    x, y = pt
    best = float("inf")
    for nm in selected:
        sx, sy = pts[nm]
        d2 = (x - sx) ** 2 + (y - sy) ** 2
        if d2 < best:
            best = d2
    return best ** 0.5


def pick_by_dispersion(
    candidates: list[str],
    pts: dict[str, tuple[float, float]],
    selected: set[str],
    adj: dict[str, set[str]],
    rng: np.random.Generator,
    top_k: int = 10,
    adj_penalty_m: float = 30000.0,
) -> str:
    """
    Score candidate by:
      score = min_distance_to_selected - adj_penalty_m*(#neighbors already selected)

    Then pick RANDOMLY among the top_k best-scoring candidates.
    This keeps it "random-looking" but avoids clustering.
    """
    scored = []
    for c in candidates:
        d = min_distance_to_selected(pts[c], pts, selected)
        n_adj = sum(1 for nb in adj.get(c, set()) if nb in selected)
        score = d - adj_penalty_m * n_adj
        scored.append((score, c))

    scored.sort(reverse=True, key=lambda x: x[0])
    k = min(top_k, len(scored))
    pool = [c for _, c in scored[:k]]
    return rng.choice(pool, size=1, replace=False).tolist()[0]


# -----------------------------
# Randomization with constraints (ULTRA-UNGROUPED)
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


def _score_set(selected: list[str], pts: dict[str, tuple[float, float]], adj: dict[str, set[str]],
              w_min: float, w_mean: float, adj_penalty_m: float) -> float:
    """
    Higher is better:
      + w_min  * minimum pairwise distance (meters)
      + w_mean * mean pairwise distance (meters)
      - adj_penalty_m * (#adjacent pairs inside selected)
    """
    xy = np.array([pts[n] for n in selected], dtype=float)
    n = xy.shape[0]
    # pairwise distances
    dmin = float("inf")
    s = 0.0
    cnt = 0
    for i in range(n):
        xi, yi = xy[i]
        for j in range(i + 1, n):
            xj, yj = xy[j]
            d = math.hypot(xi - xj, yi - yj)
            if d < dmin:
                dmin = d
            s += d
            cnt += 1
    dmean = s / max(cnt, 1)

    # adjacency count
    sel = set(selected)
    adj_pairs = 0
    for a in selected:
        for b in adj.get(a, set()):
            if b in sel:
                adj_pairs += 1
    adj_pairs //= 2  # each edge counted twice

    return (w_min * dmin) + (w_mean * dmean) - (adj_penalty_m * adj_pairs)


def constrained_randomization_ultra(
    gdf: gpd.GeoDataFrame,
    rng: np.random.Generator,
    sylhet_div_real: str,
    must_in: set[str],
    must_out: set[str],
    total_n: int = 32,
    # knobs (stronger anti-cluster)
    restarts: int = 60,
    steps: int = 9000,
    top_k_greedy: int = 20,
    w_min: float = 1.0,
    w_mean: float = 0.12,
    adj_penalty_m: float = 90000.0,
) -> set[str]:
    """
    Produces a MUCH more ungrouped set:
      1) Build initial feasible set using 'farthest among top_k' greedy
      2) Improve via many random swaps (hill-climb) with random restarts

    Constraints preserved:
      - non-Sylhet divisions >=4
      - Sylhet division exactly 2 (and includes Sylhet if forced)
      - total_n selected
      - must_in always included, must_out never included
    """

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")

    # meters CRS for distances (Bangladesh fits UTM 46N well)
    gdf_proj = gdf.to_crs("EPSG:32646")

    reps = gdf_proj.geometry.representative_point()
    pts = {nm: (float(x), float(y)) for nm, x, y in zip(gdf_proj["NAME_2"], reps.x, reps.y)}
    adj = build_adjacency(gdf_proj)

    # division -> districts
    div_to_districts: dict[str, list[str]] = {}
    for div, sub in gdf.groupby("NAME_1"):
        div_to_districts[div] = sorted(sub["NAME_2"].unique().tolist())
    divisions = sorted(div_to_districts.keys())

    # sanity
    overlap = must_in & must_out
    if overlap:
        raise ValueError(f"District(s) in both force_available and force_unavailable: {sorted(overlap)}")

    # helper counts
    def div_of(d: str) -> str:
        # quick lookup by scanning (small n); could precompute but fine
        for dv, lst in div_to_districts.items():
            if d in lst:
                return dv
        raise KeyError(d)

    district_to_div = {d: div_of(d) for div in divisions for d in div_to_districts[div]}

    def counts_by_div(sel: set[str]) -> dict[str, int]:
        c = {div: 0 for div in divisions}
        for d in sel:
            c[district_to_div[d]] += 1
        return c

    def feasible(sel: set[str]) -> bool:
        if len(sel) != total_n:
            return False
        if not must_in.issubset(sel):
            return False
        if any(d in must_out for d in sel):
            return False
        c = counts_by_div(sel)
        # sylhet exactly 2
        if c[sylhet_div_real] != 2:
            return False
        # others >=4
        for div in divisions:
            if div == sylhet_div_real:
                continue
            if c[div] < 4:
                return False
        return True

    def candidates_in_div(div: str, sel: set[str]) -> list[str]:
        return [d for d in div_to_districts[div] if d not in must_out and d not in sel]

    def min_dist_to_sel(d: str, sel: set[str]) -> float:
        if not sel:
            return float("inf")
        x, y = pts[d]
        best = float("inf")
        for s in sel:
            sx, sy = pts[s]
            best = min(best, math.hypot(x - sx, y - sy))
        return best

    def greedy_build() -> set[str]:
        sel = set(must_in)

        # ensure Sylhet = exactly 2 (but do not exceed)
        while sum(1 for d in sel if district_to_div[d] == sylhet_div_real) < 2:
            cands = candidates_in_div(sylhet_div_real, sel)
            if not cands:
                raise ValueError("Cannot satisfy Sylhet=2 with given exclusions/forced.")
            # farthest selection in Sylhet
            cands.sort(key=lambda d: min_dist_to_sel(d, sel), reverse=True)
            pool = cands[: min(top_k_greedy, len(cands))]
            sel.add(rng.choice(pool))

        # fill other divisions to >=4 (randomize division order)
        non_syl = [d for d in divisions if d != sylhet_div_real]
        rng.shuffle(non_syl)

        for div in non_syl:
            while sum(1 for d in sel if district_to_div[d] == div) < 4:
                cands = candidates_in_div(div, sel)
                if not cands:
                    raise ValueError(f"Cannot satisfy >=4 in division '{div}'.")
                cands.sort(key=lambda d: min_dist_to_sel(d, sel), reverse=True)
                pool = cands[: min(top_k_greedy, len(cands))]
                sel.add(rng.choice(pool))

        # fill remaining slots (never add more from Sylhet)
        while len(sel) < total_n:
            all_cands = []
            for div in divisions:
                if div == sylhet_div_real:
                    continue
                all_cands.extend(candidates_in_div(div, sel))
            if not all_cands:
                raise ValueError("No eligible districts left to reach 32 under constraints.")
            all_cands.sort(key=lambda d: min_dist_to_sel(d, sel), reverse=True)
            pool = all_cands[: min(top_k_greedy, len(all_cands))]
            sel.add(rng.choice(pool))

        # if greedy accidentally violated Sylhet exact (shouldn’t), fix:
        syl_sel = [d for d in sel if district_to_div[d] == sylhet_div_real]
        if len(syl_sel) > 2:
            # keep forced-in Sylhet if any; drop the rest
            keep = [d for d in syl_sel if d in must_in][:2]
            if len(keep) < 2:
                keep += [d for d in syl_sel if d not in keep][: (2 - len(keep))]
            for d in syl_sel:
                if d not in keep:
                    sel.remove(d)

        if len(sel) != total_n:
            # top-up (non-sylhet only)
            while len(sel) < total_n:
                all_cands = []
                for div in divisions:
                    if div == sylhet_div_real:
                        continue
                    all_cands.extend(candidates_in_div(div, sel))
                sel.add(rng.choice(all_cands))

        return sel

    def optimize(sel: set[str]) -> set[str]:
        best = set(sel)
        best_score = _score_set(sorted(best), pts, adj, w_min, w_mean, adj_penalty_m)

        # Precompute allowed add pool (all eligible districts)
        all_eligible = [d for div in divisions for d in div_to_districts[div] if d not in must_out]

        for _ in range(steps):
            cur = set(best)

            # pick a removable district (not forced)
            removable = [d for d in cur if d not in must_in]
            if not removable:
                break
            rem = rng.choice(removable)

            # If removing would break division minima, skip quickly
            c = counts_by_div(cur)
            div_rem = district_to_div[rem]
            if div_rem == sylhet_div_real:
                # must keep Sylhet exactly 2
                continue
            if div_rem != sylhet_div_real and c[div_rem] <= 4:
                # would drop below 4
                continue

            cur.remove(rem)

            # build candidate additions that keep constraints possible
            add_candidates = []
            for cand in all_eligible:
                if cand in cur:
                    continue
                div_c = district_to_div[cand]
                # never add to Sylhet (must stay exactly 2)
                if div_c == sylhet_div_real:
                    continue
                add_candidates.append(cand)

            if not add_candidates:
                continue

            # choose a good add using a "score among top" approach
            # (favors dispersion + low adjacency)
            # Evaluate a sample subset for speed
            sample = add_candidates
            if len(sample) > 40:
                sample = rng.choice(sample, size=40, replace=False).tolist()

            scored = []
            base_list = sorted(cur)
            base_set = set(base_list)
            for cand in sample:
                trial = base_list + [cand]
                sc = _score_set(trial, pts, adj, w_min, w_mean, adj_penalty_m)
                scored.append((sc, cand))
            scored.sort(reverse=True, key=lambda x: x[0])

            cand_pool = [cnd for _, cnd in scored[: min(10, len(scored))]]
            add = rng.choice(cand_pool)

            cur.add(add)

            if not feasible(cur):
                continue

            sc = _score_set(sorted(cur), pts, adj, w_min, w_mean, adj_penalty_m)
            if sc > best_score:
                best = set(cur)
                best_score = sc

        return best

    # multi-restart: keep best overall
    global_best = None
    global_best_score = -float("inf")

    for _ in range(restarts):
        init = greedy_build()
        improved = optimize(init)

        sc = _score_set(sorted(improved), pts, adj, w_min, w_mean, adj_penalty_m)
        if sc > global_best_score:
            global_best = set(improved)
            global_best_score = sc

    if global_best is None or not feasible(global_best):
        raise RuntimeError("Failed to generate a feasible ungrouped randomization set.")

    return global_best

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

    sylhet_div_real = resolve_name("Sylhet", norm_to_real_div)

    force_in = set(resolve_many(CFG.force_available, norm_to_real_dist))
    force_out = set(resolve_many(CFG.force_unavailable, norm_to_real_dist))
    facility = resolve_many(CFG.facility_districts, norm_to_real_dist)

    rng = np.random.default_rng(CFG.seed)
    available = constrained_randomization_ultra(
        gdf=gdf,
        rng=rng,
        sylhet_div_real=sylhet_div_real,
        must_in=force_in,
        must_out=force_out,
        total_n=32,
        restarts=12,
        steps=100,
        top_k_greedy=18,
        adj_penalty_m=150000
    )

    # Must-have check
    must_have = set(resolve_many(("Dhaka", "Bogra", "Chittagong"), norm_to_real_dist))
    if not must_have.issubset(available):
        raise RuntimeError("Constraint failed: Dhaka/Bogra/Chittagong not all POCT-available.")

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