#!/usr/bin/env python3
# bangladesh_case_map.py  (PI-updated + your requested fixes)

from __future__ import annotations
from shapely.geometry import Point
import math
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MultipleLocator
from matplotlib.patches import Circle

# ---- Styles (journal-like) ----
CASE_ALIVE_COLOR = "#1F77B4"   # blue
CASE_DIED_COLOR  = "#D62728"   # red
COMP_ALIVE_COLOR = "#111111"   # black
COMP_DIED_COLOR  = "#D62728"   # red

HOSP_DOT_FACE = "#B00020"
HOSP_DOT_EDGE = "#5A0010"

REQ_COLS = ["Site", "Patient_ID", "Latitude", "Longitude", "Test Result", "Status of the Patient"]

# --- Manual nudges requested by PI/user (number circle positions) ---
# dx, dy are in degrees (lon, lat)
NUMBER_OVERRIDE = {
    "RMCH":  (0.55,  0.04),   # move to the right of RMCH label
    "SOMCH": (-0.22, 0.18),   # slightly left from current
    "SZMCH": (-0.18, 0.10),   # slightly left from current
}

# Site label overrides (keep labels close + avoid masking points)
LABEL_OVERRIDE = {
    "SOMCH": (-0.28, 0.06),   # already requested earlier
    "RMCH":  (-0.18, 0.06),   # helps ensure RMCH number circle stays to the right of the text
}

# --- Division label preferences (only used if needed; offsets are lon/lat degrees) ---
DIV_LABEL_PREF = {
    "Dhaka":      [(0.00, 0.56), (0.00, 0.56), (0.00, 0.56)],
    "Chittagong": [(-0.22, 0.25), (-0.18, 0.25), (-0.12, 0.25)],
    "Rangpur":    [(0.08, -0.18), (0.12, -0.12), (0.00, -0.20)],
    "Khulna":     [(0.12, 0.22), (0.16, 0.28), (0.08, 0.30)],
}


def _norm_status(x) -> str:
    if pd.isna(x) or str(x).strip() == "":
        return "Alive"
    s = str(x).strip().lower()
    return "Died" if ("die" in s or "dead" in s or "death" in s) else "Alive"


def _norm_test(x) -> str:
    if pd.isna(x) or str(x).strip() == "":
        return "Unknown"
    s = str(x).strip().lower()
    if "pos" in s:
        return "Positive"
    if "neg" in s:
        return "Negative"
    return str(x).strip()


def load_site_excels(folder: Path) -> pd.DataFrame:
    files = sorted([p for p in folder.glob("*.xlsx") if not p.name.startswith("~$")])
    if not files:
        raise SystemExit(f"No .xlsx files found in: {folder}")

    frames = []
    for fp in files:
        df = pd.read_excel(fp)
        df = df.rename(columns={c: str(c).strip() for c in df.columns})
        missing = [c for c in REQ_COLS if c not in df.columns]
        if missing:
            raise SystemExit(f"{fp.name}: missing required columns {missing}. Found: {list(df.columns)}")

        df = df.copy()
        df["Site"] = df["Site"].astype(str).str.strip()
        df["Patient_ID"] = df["Patient_ID"].astype(str).str.strip()
        df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
        df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")

        df["Status of the Patient"] = df["Status of the Patient"].apply(_norm_status)
        df["Test Result"] = df["Test Result"].apply(_norm_test)

        if "Role" not in df.columns:
            df["Role"] = "Patient"
        else:
            df["Role"] = df["Role"].fillna("Patient").astype(str).str.strip()
            df.loc[df["Role"] == "", "Role"] = "Patient"

        if "Index_Patient_ID" not in df.columns:
            df["Index_Patient_ID"] = df["Patient_ID"].str.replace(r"_C\d+$", "", regex=True)

        df = df.dropna(subset=["Latitude", "Longitude"]).copy()
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def load_hospitals(path: Path) -> pd.DataFrame:
    hos = pd.read_excel(path)
    hos = hos.rename(columns={c: str(c).strip() for c in hos.columns})
    need = ["Site", "Hospital_Name", "Latitude", "Longitude"]
    miss = [c for c in need if c not in hos.columns]
    if miss:
        raise SystemExit(f"Hospitals file missing columns: {miss}. Found: {list(hos.columns)}")

    hos = hos.copy()
    hos["Site"] = hos["Site"].astype(str).str.strip()
    hos["Hospital_Name"] = hos["Hospital_Name"].astype(str).str.strip()
    hos["Latitude"] = pd.to_numeric(hos["Latitude"], errors="coerce")
    hos["Longitude"] = pd.to_numeric(hos["Longitude"], errors="coerce")
    return hos.dropna(subset=["Latitude", "Longitude"]).copy()

def label_candidates_around_circle(rad):
    # offsets scale with circle radius -> keeps label close but not overlapping
    m = rad + 0.10  # margin outside circle
    return [
        ( m,  0.00),   # right
        (-m,  0.00),   # left
        (0.00,  m),    # up
        (0.00, -m),    # down
        ( m,  0.10),   # right-up
        ( m, -0.10),   # right-down
        (-m,  0.10),   # left-up
        (-m, -0.10),   # left-down
    ]

def km_to_deg_lat(km: float) -> float:
    return float(km) / 111.0


def spread_overlaps(df: pd.DataFrame, min_sep_km: float = 6.0) -> pd.DataFrame:
    out = df.copy()
    d = km_to_deg_lat(min_sep_km)
    if d <= 0:
        out["Plot_Lat"] = out["Latitude"]
        out["Plot_Lon"] = out["Longitude"]
        return out

    key_lat = np.round(out["Latitude"] / d).astype(int)
    key_lon = np.round(out["Longitude"] / d).astype(int)
    out["_cluster"] = list(zip(key_lat, key_lon))

    golden = math.pi * (3 - math.sqrt(5))
    plot_lat = out["Latitude"].to_numpy(float).copy()
    plot_lon = out["Longitude"].to_numpy(float).copy()

    for _, idxs in out.groupby("_cluster").groups.items():
        idxs = list(idxs)
        if len(idxs) <= 1:
            continue

        lat0 = float(np.nanmean(out.loc[idxs, "Latitude"]))
        coslat = max(0.35, math.cos(math.radians(lat0)))

        for k, ix in enumerate(idxs):
            if k == 0:
                plot_lat[ix] = out.loc[ix, "Latitude"]
                plot_lon[ix] = out.loc[ix, "Longitude"]
                continue
            r = d * 0.75 * math.sqrt(k)
            theta = k * golden
            dlat = r * math.sin(theta)
            dlon = (r * math.cos(theta)) / coslat

            plot_lat[ix] = out.loc[ix, "Latitude"] + dlat
            plot_lon[ix] = out.loc[ix, "Longitude"] + dlon

    out["Plot_Lat"] = plot_lat
    out["Plot_Lon"] = plot_lon
    out.drop(columns=["_cluster"], inplace=True)
    return out


def number_circle_radius_deg(n: int) -> float:
    n = max(int(n), 0)
    return 0.075 + 0.012 * math.sqrt(n)


def fmt_lon(x, _):
    return f"{x:.1f}°E"


def fmt_lat(y, _):
    return f"{y:.1f}°N"


def best_offset(hx, hy, candidates, avoid_lon, avoid_lat, used_points, w_used=1.0):
    best = None
    best_score = -1e18

    for dx, dy in candidates:
        x = hx + dx
        y = hy + dy

        if avoid_lon.size:
            d2_pat = np.min((avoid_lon - x) ** 2 + (avoid_lat - y) ** 2)
        else:
            d2_pat = 1e6

        d2_used = 1e6
        if used_points:
            xs = np.array([p[0] for p in used_points], float)
            ys = np.array([p[1] for p in used_points], float)
            d2_used = np.min((xs - x) ** 2 + (ys - y) ** 2)

        score = d2_pat + w_used * d2_used
        if score > best_score:
            best_score = score
            best = (dx, dy)

    return best if best is not None else candidates[0]

def nudge_inside(lon, lat, poly, center_pt, eps=0.002):
    """
    Pull a point inside poly if it falls outside.
    eps ~ degrees (0.002 ≈ 200m). Increase slightly if needed.
    """
    p = Point(lon, lat)
    if poly.contains(p):
        return lon, lat

    cx, cy = center_pt.x, center_pt.y

    # pull towards center progressively until it enters
    for t in [0.98, 0.95, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30]:
        nx = cx + t * (lon - cx)
        ny = cy + t * (lat - cy)
        if poly.contains(Point(nx, ny)):
            return nx, ny

    # fallback: snap near boundary then move slightly inward
    b = poly.boundary
    nearest = b.interpolate(b.project(p))
    vx, vy = cx - nearest.x, cy - nearest.y
    norm = math.hypot(vx, vy) or 1.0
    return nearest.x + eps * (vx / norm), nearest.y + eps * (vy / norm)

def _guess_division_name_col(adm1: gpd.GeoDataFrame) -> str:
    # Try common admin-1 name fields first
    preferred = ["NAME_1", "NAME_EN", "NAME", "Division", "DIVISION", "division", "ADM1_EN", "NAME_ENG"]
    for c in preferred:
        if c in adm1.columns:
            return c

    # Otherwise: first non-geometry object column
    for c in adm1.columns:
        if c != adm1.geometry.name and adm1[c].dtype == "object":
            return c

    raise ValueError("Could not detect a division name column in the divisions file.")


def _largest_poly(geom):
    # Handle MultiPolygon by picking the largest piece
    if geom is None:
        return None
    if geom.geom_type == "Polygon":
        return geom
    if geom.geom_type == "MultiPolygon":
        return max(list(geom.geoms), key=lambda g: g.area)
    return geom


def _best_point_inside_polygon(poly, avoid_lon, avoid_lat, used_points,
                              name=None, n_samples=700, seed=42, inner_buffer=0.06):
    """
    Pick a point inside `poly` that:
    - avoids patient dots + other labels (hard minimum clearance)
    - prefers deep interior (strong border-distance reward)
    - can try preferred offsets for specific divisions (Dhaka/Chittagong/Rangpur/Khulna)
    """
    poly = _largest_poly(poly)
    if poly is None or poly.is_empty:
        return None

    # dynamic buffer so small polygons don't collapse
    minx0, miny0, maxx0, maxy0 = poly.bounds
    w = maxx0 - minx0
    h = maxy0 - miny0
    buf = min(inner_buffer, 0.20 * min(w, h))
    inner = poly.buffer(-buf)
    if inner.is_empty:
        inner = poly

    minx, miny, maxx, maxy = inner.bounds
    rng = np.random.default_rng(seed)

    # prep used label points
    if used_points:
        ux = np.array([p[0] for p in used_points], float)
        uy = np.array([p[1] for p in used_points], float)
    else:
        ux = uy = None

    have_pat = bool(getattr(avoid_lon, "size", 0))

    # HARD minimum clearance (degrees). Tune if needed.
    # 0.08 deg ≈ 8–9 km (enough to avoid visual overlap on this map)
    MIN_CLEAR = 0.08

    def score_point(x, y):
        pt = Point(x, y)

        # must be inside the inner polygon
        if not inner.contains(pt):
            return None

        # nearest patient distance
        if have_pat:
            d2_pat = np.min((avoid_lon - x) ** 2 + (avoid_lat - y) ** 2)
            d_pat = math.sqrt(d2_pat)
            if d_pat < MIN_CLEAR:
                return -1e12  # hard reject
        else:
            d_pat = 999.0

        # nearest used-label distance
        if ux is not None:
            d2_used = np.min((ux - x) ** 2 + (uy - y) ** 2)
            d_used = math.sqrt(d2_used)
            if d_used < (MIN_CLEAR * 0.75):
                return -1e12
        else:
            d_used = 999.0

        # distance from border (bigger is better)
        d_border = poly.boundary.distance(pt)

        # score: prioritize interior strongly + keep away from points/labels
        return (3.0 * d_pat) + (3.0 * d_used) + (80.0 * d_border)

    best_xy = None
    best_score = -1e18

    # 1) Try preferred offsets first (for your 4 problematic divisions)
    if name and name in DIV_LABEL_PREF:
        rp = poly.representative_point()
        for dx, dy in DIV_LABEL_PREF[name]:
            s = score_point(rp.x + dx, rp.y + dy)
            if s is not None and s > best_score:
                best_score = s
                best_xy = (rp.x + dx, rp.y + dy)

    # 2) Random search inside “inner” polygon
    tries = 0
    kept = 0
    while tries < n_samples * 4 and kept < n_samples:
        tries += 1
        x = rng.uniform(minx, maxx)
        y = rng.uniform(miny, maxy)
        if not inner.contains(Point(x, y)):
            continue
        kept += 1

        s = score_point(x, y)
        if s is not None and s > best_score:
            best_score = s
            best_xy = (x, y)

    # 3) Fallback (always inside)
    if best_xy is None:
        rp = poly.representative_point()
        best_xy = (rp.x, rp.y)

    return best_xy


def _div_key(name: str) -> str:
    # robust matching across shapefiles like "Dhaka", "Dhaka Division", "DHAKA"
    s = str(name).strip().lower()
    s = s.replace("division", "").strip()
    return s

def place_division_labels(adm1, ax, avoid_lon, avoid_lat, used_points):
    """
    Place division names inside each polygon, avoiding overlap with plotted points/labels,
    and respecting manual preferences in DIV_LABEL_PREF (dx,dy in degrees).
    """
    name_col = _guess_division_name_col(adm1)

    # subtle styling (smaller + darker, as requested)
    bbox_kw = dict(facecolor="white", edgecolor="none", alpha=0.35, pad=0.9)

    # normalize preference dict keys once
    pref_norm = {_div_key(k): v for k, v in DIV_LABEL_PREF.items()}

    # pre-pack used points
    if used_points:
        ux = np.array([p[0] for p in used_points], float)
        uy = np.array([p[1] for p in used_points], float)
    else:
        ux = uy = None

    have_pat = bool(getattr(avoid_lon, "size", 0))

    for i, row in adm1.iterrows():
        raw_name = str(row[name_col]).strip()
        if not raw_name or raw_name.lower() == "nan":
            continue

        poly = _largest_poly(row.geometry)
        if poly is None or poly.is_empty:
            continue

        # keep labels away from borders if possible
        safe_poly = poly.buffer(-0.08)
        if safe_poly.is_empty:
            safe_poly = poly

        # base blank-space point (your existing "blank space" logic)
        base_xy = _best_point_inside_polygon(
            safe_poly, avoid_lon, avoid_lat, used_points,
            n_samples=450, seed=100 + i
        )
        bx, by = base_xy

        # candidate nudges: try your preferred offsets FIRST (if present), then small fallbacks
        key = _div_key(raw_name)
        nudges = pref_norm.get(key, [])

        # add tiny fallbacks if no pref works
        nudges = nudges + [
            (0.00, 0.00),
            (0.00, 0.10), (0.00, -0.10),
            (0.10, 0.00), (-0.10, 0.00),
        ]

        best = (bx, by)
        best_score = -1e18

        for dx, dy in nudges:
            x = bx + dx
            y = by + dy

            if not safe_poly.contains(Point(x, y)):
                continue

            # far from patient points
            if have_pat:
                d2_pat = np.min((avoid_lon - x) ** 2 + (avoid_lat - y) ** 2)
            else:
                d2_pat = 1e6

            # far from other labels
            if ux is not None and len(ux):
                d2_used = np.min((ux - x) ** 2 + (uy - y) ** 2)
            else:
                d2_used = 1e6

            score = d2_pat + 2.0 * d2_used
            if score > best_score:
                best_score = score
                best = (x, y)

        x, y = best

        ax.text(
            x, y, raw_name,
            ha="center", va="center",
            fontsize=8.5, fontweight="semibold",
            color="#333333", alpha=0.95,
            zorder=3.2,
            bbox=bbox_kw
        )

        used_points.append((x, y))
        if ux is None:
            ux = np.array([x], float)
            uy = np.array([y], float)
        else:
            ux = np.append(ux, x)
            uy = np.append(uy, y)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_folder", required=True, type=Path)
    ap.add_argument("--hospitals", required=True, type=Path)
    ap.add_argument("--boundary", required=True, type=Path)
    ap.add_argument("--divisions", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--include_companions", action="store_true")
    ap.add_argument("--min_sep_km", type=float, default=6.0)
    args = ap.parse_args()

    df = load_site_excels(args.data_folder)
    hos = load_hospitals(args.hospitals)

    if not args.include_companions:
        df = df[df["Role"].astype(str).str.lower() != "companion"].copy()

    df = spread_overlaps(df, min_sep_km=args.min_sep_km)

    bd0 = gpd.read_file(args.boundary)
    if bd0.crs is None:
        bd0 = bd0.set_crs(4326)
    bd0 = bd0.to_crs(4326)

    # --- Bangladesh polygon + safe interior polygon (prevents border spill-out) ---
    bd_poly = bd0.geometry.unary_union
    bd_safe = bd_poly.buffer(-0.01)  # inward buffer (~1 km). If too aggressive, use -0.005
    if bd_safe.is_empty:
        bd_safe = bd_poly  # fallback if buffer collapses
    bd_center = bd_poly.representative_point()

    # --- Force jittered plotting points inside Bangladesh ---
    def _fix_row(r):
        lon, lat = nudge_inside(r["Plot_Lon"], r["Plot_Lat"], bd_safe, bd_center)
        return pd.Series([lon, lat], index=["Plot_Lon", "Plot_Lat"])

    df[["Plot_Lon", "Plot_Lat"]] = df.apply(_fix_row, axis=1)


    adm1 = gpd.read_file(args.divisions)
    if adm1.crs is None:
        adm1 = adm1.set_crs(4326)
    adm1 = adm1.to_crs(4326)

    df["RoleN"] = df["Role"].astype(str).str.strip().str.lower()
    is_comp = df["RoleN"].eq("companion")
    is_case = ~is_comp

    case_alive = df[is_case & df["Status of the Patient"].eq("Alive")]
    case_died  = df[is_case & df["Status of the Patient"].eq("Died")]
    comp_alive = df[is_comp & df["Status of the Patient"].eq("Alive")]
    comp_died  = df[is_comp & df["Status of the Patient"].eq("Died")]

    pos_counts = (
        df[is_case & df["Test Result"].eq("Positive")]
        .groupby("Site")["Index_Patient_ID"]
        .nunique()
        .to_dict()
    )

    avoid_lon = df["Plot_Lon"].to_numpy(float)
    avoid_lat = df["Plot_Lat"].to_numpy(float)

    fig, ax = plt.subplots(figsize=(10.5, 12.0))

    bd0.plot(ax=ax, facecolor="#F7F7F7", edgecolor="#111111", linewidth=1.15, zorder=1)
    adm1.plot(ax=ax, facecolor="none", edgecolor="#444444", linewidth=0.70, zorder=2, alpha=0.95)

    if len(case_alive):
        ax.scatter(case_alive["Plot_Lon"], case_alive["Plot_Lat"],
                   s=22, marker="o",
                   facecolors=CASE_ALIVE_COLOR, edgecolors="#0B3D91",
                   linewidths=0.25, alpha=0.70, zorder=4)

    if len(case_died):
        ax.scatter(case_died["Plot_Lon"], case_died["Plot_Lat"],
                   s=32, marker="o",
                   facecolors=CASE_DIED_COLOR, edgecolors="#7A0010",
                   linewidths=0.35, alpha=0.85, zorder=5)

    if args.include_companions and len(comp_alive):
        ax.scatter(comp_alive["Plot_Lon"], comp_alive["Plot_Lat"],
                   s=42, marker="x",
                   c=COMP_ALIVE_COLOR, linewidths=1.25, alpha=0.85, zorder=6)

    if args.include_companions and len(comp_died):
        ax.scatter(comp_died["Plot_Lon"], comp_died["Plot_Lat"],
                   s=42, marker="x",
                   c=COMP_DIED_COLOR, linewidths=1.25, alpha=0.90, zorder=7)

    # --- Title: increase gap above map box ---
    ax.set_title("Distribution of Methanol Poisoning Cases Across the Sites in Bangladesh",
                 fontsize=14, fontweight="bold", pad=36)

    ax.xaxis.set_major_formatter(FuncFormatter(fmt_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_lat))
    ax.tick_params(axis="both", which="major", labelsize=9, width=1.1, length=6)
    ax.tick_params(top=True, labeltop=True, right=True, labelright=True)

    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.yaxis.set_major_locator(MultipleLocator(1.0))
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.25)

    minx, miny, maxx, maxy = bd0.total_bounds
    ax.set_xlim(minx - 0.4, maxx + 0.7)
    ax.set_ylim(miny - 0.4, maxy + 0.6)

    # Legend (unchanged position request already satisfied)
    legend_items = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=CASE_ALIVE_COLOR,
            markeredgecolor="#0B3D91", markersize=8, label="Case (Alive)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=CASE_DIED_COLOR,
            markeredgecolor="#7A0010", markersize=9, label="Case (Died)"),
        Line2D([0], [0], marker="x", color=COMP_ALIVE_COLOR, markersize=9,
            linewidth=0, label="Companion (Alive)"),
        Line2D([0], [0], marker="x", color=COMP_DIED_COLOR, markersize=9,
            linewidth=0, label="Companion (Died)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
            markeredgecolor=HOSP_DOT_FACE, markersize=10, label="Positive cases (circle size)"),
    ]

    leg = ax.legend(
        handles=legend_items,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        borderaxespad=0.4,
        frameon=True, framealpha=0.96,
        facecolor="white", edgecolor="#DDDDDD",
        fontsize=9, title="Legend"
    )
    leg.get_title().set_fontweight("bold")

    used_label_points = []

    number_candidates = [
        (0.35, 0.18), (0.35, -0.18), (-0.35, 0.18), (-0.35, -0.18),
        (0.45, 0.00), (-0.45, 0.00), (0.00, 0.28), (0.00, -0.28),
    ]
    label_candidates = [
        (0.18, 0.06), (0.18, -0.06), (-0.18, 0.06), (-0.18, -0.06),
        (0.00, 0.16), (0.00, -0.16)
    ]

    used_label_points = []

    for _, r in hos.iterrows():
        site = str(r["Site"]).strip()
        hx, hy = float(r["Longitude"]), float(r["Latitude"])
        npos = int(pos_counts.get(site, 0))

        # ---- Number circle position ----
        if site in NUMBER_OVERRIDE:
            dxn, dyn = NUMBER_OVERRIDE[site]
        else:
            dxn, dyn = best_offset(
                hx, hy,
                candidates=number_candidates,
                avoid_lon=avoid_lon, avoid_lat=avoid_lat,
                used_points=used_label_points,
                w_used=2.0
            )

        nx, ny = hx + dxn, hy + dyn
        rad = number_circle_radius_deg(npos)
        # Keep the number-circle center inside Bangladesh
        nx, ny = nudge_inside(nx, ny, bd_safe, bd_center)

        # Draw the positive-case circle + number
        circ = Circle((nx, ny), radius=rad, facecolor="white",
                    edgecolor=HOSP_DOT_FACE, linewidth=0.9, zorder=12)
        ax.add_patch(circ)

        ax.text(nx, ny, str(npos),
                ha="center", va="center",
                fontsize=10, fontweight="bold", color=HOSP_DOT_FACE, zorder=13)

        used_label_points.append((nx, ny))

        # ---- Site label: place near the number circle (NOT near hospital) ----
        site_label_candidates = label_candidates_around_circle(rad)

        dxl, dyl = best_offset(
            nx, ny,  # IMPORTANT: anchor on the circle center
            candidates=site_label_candidates,
            avoid_lon=avoid_lon, avoid_lat=avoid_lat,
            used_points=used_label_points,
            w_used=3.0
        )
        lx, ly = nx + dxl, ny + dyl
        
        # If label goes outside border, force it BELOW the number circle (fixes SOMCH issue)
        if not bd_safe.contains(Point(lx, ly)):
            lx, ly = nx, ny - (rad + 0.12)

        ax.text(lx, ly, site,
                fontsize=9.5, fontweight="bold", color="#2E7D32",
                ha="center", va="center",
                zorder=13)

        used_label_points.append((lx, ly))

    # ---- Division names (place AFTER hospital circles + site labels so we can avoid them) ----
    # Build a stronger "avoid" list: patient plot points + all label points (circles + site labels)
    avoid_lon2 = np.concatenate([avoid_lon, np.array([p[0] for p in used_label_points], float)]) if used_label_points else avoid_lon
    avoid_lat2 = np.concatenate([avoid_lat, np.array([p[1] for p in used_label_points], float)]) if used_label_points else avoid_lat

    place_division_labels(adm1, ax, avoid_lon2, avoid_lat2, used_label_points)


    # Footnote updated to reflect the change (circle size = positives)
    fig.text(
        0.5, 0.025,
        "Footnote: Circle size is proportional to the number of positive enrolled cases; numeric labels show the same count.",
        ha="center", va="center", fontsize=9, color="#333333"
    )

    # --- Layout: more headroom above map box ---
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.06, top=0.885)

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    print("Saved:")
    print(out.with_suffix(".png"))
    print(out.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
