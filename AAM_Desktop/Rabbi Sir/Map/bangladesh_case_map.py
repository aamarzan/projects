#!/usr/bin/env python3
# bangladesh_case_map.py
# Bangladesh map: methanol poisoning cases (positive/negative, alive/died) + companions,
# with division boundaries and site-level positive-count circles.
#
# Inputs:
#   --data_folder : folder containing site .xlsx files (CMCH.xlsx, DMCH.xlsx, ...)
#   --hospitals   : hospitals.xlsx (Site, Hospital_Name, Latitude, Longitude)
#   --boundary    : bangladesh_boundary.geojson (ADM0 outline)
#   --divisions   : bangladesh_divisions_ADM1.geojson (ADM1 boundaries)
#
# Outputs:
#   PNG (600 dpi) + PDF + SVG

from __future__ import annotations
from pathlib import Path
import argparse
import math

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MultipleLocator
from matplotlib.patches import Circle
from shapely.geometry import Point

# ---- Styles ----
CASE_ALIVE_COLOR = "#1F77B4"   # blue
CASE_DIED_COLOR  = "#D62728"   # red
COMP_ALIVE_COLOR = "#111111"   # black
COMP_DIED_COLOR  = "#D62728"   # red

CIRCLE_EDGE = "#B00020"        # deep red outline for hospital/positive-count circles
CIRCLE_FACE = "white"
SITE_LABEL_COLOR = "#2E7D32"   # green site labels
SITE_LABEL_SIZE = 12.5
DIVISION_LABEL_SIZE = 10.6
DIVISION_LABEL_COLOR = "#444444"

REQ_COLS = ["Site", "Patient_ID", "Latitude", "Longitude", "Test Result", "Status of the Patient"]

# Optional manual nudges for the POSITIVE COUNT circles (lon, lat degrees)
NUMBER_OVERRIDE = {
    # Keep empty unless a site truly needs manual override
}

# Relative nudges for SITE NAME placement relative to the number-circle center.
# These are only used when a site is NOT listed in SITE_TEXT_ABSOLUTE_OVERRIDE.
LABEL_OVERRIDE = {
    "CMCH":  (0.22,  0.03),
    "DMCH":  (-0.52, 0.16),
    "RMCH":  (0.26, -0.18),
    "SOMCH": (0.10, -0.22),
    "SZMCH": (0.24, -0.12),
}

# Absolute manual label positions for ALL site names.
# Edit these directly as (Longitude, Latitude) whenever you want to fine-tune label placement.
# When a site is listed here, these coordinates take precedence over LABEL_OVERRIDE.
SITE_TEXT_ABSOLUTE_OVERRIDE = {
    "CMCH":  (92.09, 23.08),
    "DMCH":  (90.18, 24.08),
    "RMCH":  (88.96, 23.74),
    "SOMCH": (91.32, 24.99),
    "SZMCH": (89.48, 25.03),
}

# Absolute manual positions for ALL division names.
# Edit these directly as (Longitude, Latitude) whenever you want to fine-tune label placement.
DIVISION_TEXT_OVERRIDE = {
    "Rangpur":    (89.35, 25.68),
    "Rajshahi":   (89.25, 24.16),
    "Mymensingh": (90.40, 24.95),
    "Dhaka":      (90.18, 24.20),
    "Khulna":     (89.45, 22.38),
    "Barishal":    (90.36, 22.33),
    "Chittagong": (91.84, 22.82),
    "Sylhet":     (91.74, 24.72),
}

def fmt_lon(x, _): return f"{x:.1f}°E"
def fmt_lat(y, _): return f"{y:.1f}°N"

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

def km_to_deg_lat(km: float) -> float:
    return float(km) / 111.0

def spread_overlaps(df: pd.DataFrame, min_sep_km: float = 6.0) -> pd.DataFrame:
    """
    Deterministic jitter so overlapping points become distinguishable.
    Adds Plot_Lon / Plot_Lat columns.
    """
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

def nudge_inside(lon, lat, poly, center_pt, eps=0.002):
    """Pull a point inside polygon if it falls outside (after jitter/offset)."""
    p = Point(lon, lat)
    if poly.contains(p):
        return lon, lat

    cx, cy = center_pt.x, center_pt.y

    for t in [0.98, 0.95, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40]:
        nx = cx + t * (lon - cx)
        ny = cy + t * (lat - cy)
        if poly.contains(Point(nx, ny)):
            return nx, ny

    b = poly.boundary
    nearest = b.interpolate(b.project(p))
    vx, vy = cx - nearest.x, cy - nearest.y
    norm = math.hypot(vx, vy) or 1.0
    return nearest.x + eps * (vx / norm), nearest.y + eps * (vy / norm)

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

        # Role handling (optional)
        if "Role" not in df.columns:
            df["Role"] = "Patient"
        else:
            df["Role"] = df["Role"].fillna("Patient").astype(str).str.strip()
            df.loc[df["Role"] == "", "Role"] = "Patient"

        # Index patient id (for companions -> ties back to recruited)
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

def best_offset(hx, hy, candidates, avoid_lon, avoid_lat, used_points, w_used=1.0):
    """Pick offset maximizing distance from patient points + already used label points."""
    best = None
    best_score = -1e18

    for dx, dy in candidates:
        x = hx + dx
        y = hy + dy

        if getattr(avoid_lon, "size", 0):
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

def number_circle_radius_deg(n: int) -> float:
    n = max(int(n), 0)
    return 0.075 + 0.012 * math.sqrt(n)

def label_candidates_around_circle(rad):
    # farther away than before to avoid overlap with nearby points/circle
    m = rad + 0.24
    return [
        ( m, 0.00), (-m, 0.00), (0.00, m), (0.00, -m),
        ( m, 0.12), ( m,-0.12), (-m, 0.12), (-m,-0.12),
        ( m+0.08, 0.00), (-m-0.08, 0.00),
        (0.00, m+0.08), (0.00, -m-0.08),
    ]

def find_division_name_col(adm1: gpd.GeoDataFrame):
    preferred = [
        "NAME_1", "NAME_EN", "ADM1_EN", "ADM1_NAME", "Division", "division",
        "ShapeName", "shapeName", "shape_name", "name", "NAME", "Name", "admin1Name"
    ]
    for cand in preferred:
        if cand in adm1.columns:
            vals = adm1[cand].dropna().astype(str).str.strip()
            if len(vals) and vals.nunique() >= 4:
                return cand

    # heuristic fallback: first non-geometry object/string-like column with several unique short labels
    for col in adm1.columns:
        if col == "geometry":
            continue
        vals = adm1[col].dropna().astype(str).str.strip()
        if not len(vals):
            continue
        if 4 <= vals.nunique() <= max(20, len(vals) + 2):
            mean_len = vals.map(len).mean()
            if mean_len <= 25:
                return col
    return None

def canonical_division_name(name: str) -> str:
    s = str(name).strip()
    sl = s.lower()
    mapping = {
        "chattogram": "Chittagong",
        "chittagong": "Chittagong",
        "barisal": "Barishal",
        "barishal": "Barishal",
        "rajshani": "Rajshahi",
        "rajshahi": "Rajshahi",
        "rangpur": "Rangpur",
        "mymensingh": "Mymensingh",
        "dhaka": "Dhaka",
        "khulna": "Khulna",
        "sylhet": "Sylhet",
    }
    return mapping.get(sl, s)

def plot_division_labels(ax, adm1: gpd.GeoDataFrame):
    name_col = find_division_name_col(adm1)
    if name_col is None:
        # hard fallback to known Bangladesh division names and approximate positions
        for name, (x, y) in DIVISION_TEXT_OVERRIDE.items():
            ax.text(
                x, y, name,
                fontsize=DIVISION_LABEL_SIZE, fontweight="bold", color=DIVISION_LABEL_COLOR,
                ha="center", va="center", zorder=3, alpha=0.95,
                bbox=dict(boxstyle="round,pad=0.10", facecolor="white", edgecolor="none", alpha=0.55)
            )
        return

    for _, rr in adm1.iterrows():
        geom = rr.geometry
        if geom is None or geom.is_empty:
            continue
        raw_name = str(rr[name_col]).strip()
        if not raw_name or raw_name.lower() == "nan":
            continue
        name = canonical_division_name(raw_name)
        pt = geom.representative_point()
        x, y = pt.x, pt.y
        if name in DIVISION_TEXT_OVERRIDE:
            x, y = DIVISION_TEXT_OVERRIDE[name]

        ax.text(
            x, y, name,
            fontsize=DIVISION_LABEL_SIZE, fontweight="bold", color=DIVISION_LABEL_COLOR,
            ha="center", va="center", zorder=3, alpha=0.95,
            bbox=dict(boxstyle="round,pad=0.10", facecolor="white", edgecolor="none", alpha=0.55)
        )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_folder", required=True, type=Path)
    ap.add_argument("--hospitals", required=True, type=Path)
    ap.add_argument("--boundary", required=True, type=Path)
    ap.add_argument("--divisions", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--include_companions", action="store_true")
    ap.add_argument("--min_sep_km", type=float, default=6.0)
    ap.add_argument("--show_division_names", action="store_true")
    args = ap.parse_args()

    df = load_site_excels(args.data_folder)
    hos = load_hospitals(args.hospitals)

    if not args.include_companions:
        df = df[df["Role"].astype(str).str.lower() != "companion"].copy()

    # Jitter
    df = spread_overlaps(df, min_sep_km=args.min_sep_km)

    # Boundary + divisions
    bd0 = gpd.read_file(args.boundary)
    if bd0.crs is None:
        bd0 = bd0.set_crs(4326)
    bd0 = bd0.to_crs(4326)

    adm1 = gpd.read_file(args.divisions)
    if adm1.crs is None:
        adm1 = adm1.set_crs(4326)
    adm1 = adm1.to_crs(4326)

    # Bangladesh polygon safety buffer
    bd_poly = bd0.geometry.unary_union
    bd_safe = bd_poly.buffer(-0.01)
    if bd_safe.is_empty:
        bd_safe = bd_poly
    bd_center = bd_poly.representative_point()

    # Force plotted points inside Bangladesh
    def _fix_row(r):
        lon, lat = nudge_inside(float(r["Plot_Lon"]), float(r["Plot_Lat"]), bd_safe, bd_center)
        return pd.Series([lon, lat], index=["Plot_Lon", "Plot_Lat"])
    df[["Plot_Lon", "Plot_Lat"]] = df.apply(_fix_row, axis=1)

    # Role masks
    role = df["Role"].astype(str).str.strip().str.lower()
    is_comp = role.eq("companion")
    is_case = ~is_comp

    # Case categories by status & test
    case_alive_pos = df[is_case & df["Status of the Patient"].eq("Alive") & df["Test Result"].eq("Positive")]
    case_alive_neg = df[is_case & df["Status of the Patient"].eq("Alive") & ~df["Test Result"].eq("Positive")]
    case_died_pos  = df[is_case & df["Status of the Patient"].eq("Died")  & df["Test Result"].eq("Positive")]
    case_died_neg  = df[is_case & df["Status of the Patient"].eq("Died")  & ~df["Test Result"].eq("Positive")]

    comp_alive = df[is_comp & df["Status of the Patient"].eq("Alive")]
    comp_died  = df[is_comp & df["Status of the Patient"].eq("Died")]

    # Positive counts per site (unique recruited/index patients)
    pos_counts = (
        df[is_case & df["Test Result"].eq("Positive")]
        .groupby("Site")["Index_Patient_ID"]
        .nunique()
        .to_dict()
    )

    avoid_lon = df["Plot_Lon"].to_numpy(float)
    avoid_lat = df["Plot_Lat"].to_numpy(float)

    fig, ax = plt.subplots(figsize=(10.8, 12.4))

    # Base map
    bd0.plot(ax=ax, facecolor="#F7F7F7", edgecolor="#111111", linewidth=1.15, zorder=1)
    adm1.plot(ax=ax, facecolor="none", edgecolor="#444444", linewidth=0.70, zorder=2, alpha=0.95)

    # Division names
    if args.show_division_names or True:
        plot_division_labels(ax, adm1)

    # ---- Plot points ----
    # Alive Positive (filled blue)
    if len(case_alive_pos):
        ax.scatter(case_alive_pos["Plot_Lon"], case_alive_pos["Plot_Lat"],
                   s=22, marker="o", facecolors=CASE_ALIVE_COLOR, edgecolors="#0B3D91",
                   linewidths=0.25, alpha=0.75, zorder=4)

    # Alive Negative/Unknown (hollow blue)
    if len(case_alive_neg):
        ax.scatter(case_alive_neg["Plot_Lon"], case_alive_neg["Plot_Lat"],
                   s=22, marker="o", facecolors="none", edgecolors=CASE_ALIVE_COLOR,
                   linewidths=0.9, alpha=0.70, zorder=4)

    # Died Positive (filled red)
    if len(case_died_pos):
        ax.scatter(case_died_pos["Plot_Lon"], case_died_pos["Plot_Lat"],
                   s=32, marker="o", facecolors=CASE_DIED_COLOR, edgecolors="#7A0010",
                   linewidths=0.35, alpha=0.90, zorder=5)

    # Died Negative/Unknown (hollow red)
    if len(case_died_neg):
        ax.scatter(case_died_neg["Plot_Lon"], case_died_neg["Plot_Lat"],
                   s=32, marker="o", facecolors="none", edgecolors=CASE_DIED_COLOR,
                   linewidths=1.1, alpha=0.85, zorder=5)

    # Companions
    if args.include_companions and len(comp_alive):
        ax.scatter(comp_alive["Plot_Lon"], comp_alive["Plot_Lat"],
                   s=42, marker="x", c=COMP_ALIVE_COLOR, linewidths=1.25, alpha=0.85, zorder=6)

    if args.include_companions and len(comp_died):
        ax.scatter(comp_died["Plot_Lon"], comp_died["Plot_Lat"],
                   s=42, marker="x", c=COMP_DIED_COLOR, linewidths=1.25, alpha=0.90, zorder=7)

    # Title
    ax.set_title("Distribution of Methanol Poisoning Cases Across the Sites",
                 fontsize=14, fontweight="bold", pad=36)

    # Axes formatting
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

    # Legend (journal-friendly)
    legend_items = [
        Line2D([0],[0], marker="o", color="none", markerfacecolor=CASE_ALIVE_COLOR,
               markeredgecolor="#0B3D91", markersize=8, label="Case (Alive, Positive)"),
        Line2D([0],[0], marker="o", color="none", markerfacecolor="none",
               markeredgecolor=CASE_ALIVE_COLOR, markersize=8, label="Case (Alive, Negative/Unknown)"),
        Line2D([0],[0], marker="o", color="none", markerfacecolor=CASE_DIED_COLOR,
               markeredgecolor="#7A0010", markersize=9, label="Case (Died, Positive)"),
        Line2D([0],[0], marker="o", color="none", markerfacecolor="none",
               markeredgecolor=CASE_DIED_COLOR, markersize=9, label="Case (Died, Negative/Unknown)"),
    ]
    if args.include_companions:
        legend_items += [
            Line2D([0],[0], marker="x", color=COMP_ALIVE_COLOR, markersize=9, linewidth=0, label="Companion (Alive)"),
            Line2D([0],[0], marker="x", color=COMP_DIED_COLOR, markersize=9, linewidth=0, label="Companion (Died)"),
        ]
    legend_items += [
        Line2D([0],[0], marker="o", color="none", markerfacecolor="white",
               markeredgecolor=CIRCLE_EDGE, markersize=10, label="Positive cases (circle size)"),
    ]

    leg = ax.legend(handles=legend_items, loc="upper right",
                    bbox_to_anchor=(0.985, 0.985),
                    borderaxespad=0.4, frameon=True, framealpha=0.96,
                    facecolor="white", edgecolor="#DDDDDD",
                    fontsize=9, title="Legend")
    leg.get_title().set_fontweight("bold")

    # ---- Site positive-count circles + site labels ----
    used_label_points = []

    number_candidates = [
        (0.38, 0.20), (0.38, -0.20), (-0.38, 0.20), (-0.38, -0.20),
        (0.50, 0.00), (-0.50, 0.00), (0.00, 0.32), (0.00, -0.32),
        (0.28, 0.36), (-0.28, 0.36), (0.28, -0.36), (-0.28, -0.36),
    ]

    for _, r in hos.iterrows():
        site = str(r["Site"]).strip()
        hx, hy = float(r["Longitude"]), float(r["Latitude"])
        npos = int(pos_counts.get(site, 0))

        if site in NUMBER_OVERRIDE:
            dxn, dyn = NUMBER_OVERRIDE[site]
        else:
            dxn, dyn = best_offset(hx, hy, number_candidates, avoid_lon, avoid_lat, used_label_points, w_used=2.0)

        nx, ny = hx + dxn, hy + dyn
        nx, ny = nudge_inside(nx, ny, bd_safe, bd_center)
        rad = number_circle_radius_deg(npos)

        # circle + number
        circ = Circle((nx, ny), radius=rad, facecolor=CIRCLE_FACE, edgecolor=CIRCLE_EDGE,
                      linewidth=0.9, zorder=12)
        ax.add_patch(circ)
        ax.text(nx, ny, str(npos), ha="center", va="center",
                fontsize=10, fontweight="bold", color=CIRCLE_EDGE, zorder=13)
        used_label_points.append((nx, ny))

        # site label near circle
        if site in SITE_TEXT_ABSOLUTE_OVERRIDE:
            lx, ly = SITE_TEXT_ABSOLUTE_OVERRIDE[site]
        else:
            if site in LABEL_OVERRIDE:
                dxl, dyl = LABEL_OVERRIDE[site]
            else:
                dxl, dyl = best_offset(nx, ny, label_candidates_around_circle(rad),
                                       avoid_lon, avoid_lat, used_label_points, w_used=3.0)
            lx, ly = nx + dxl, ny + dyl
            if not bd_safe.contains(Point(lx, ly)):
                lx, ly = nx, ny - (rad + 0.18)

        ax.text(
            lx, ly, site,
            fontsize=SITE_LABEL_SIZE, fontweight="bold", color=SITE_LABEL_COLOR,
            ha="center", va="center", zorder=13,
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.82)
        )
        used_label_points.append((lx, ly))

    # Footnote
    fig.text(0.5, 0.028,
             "Footnote: Circle size is proportional to the number of positive enrolled cases; numeric labels show the same count.",
             ha="center", va="center", fontsize=9.5, color="#444444")

    # Layout
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.08, top=0.885)

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)

    print("[OK] Saved:")
    print(out.with_suffix(".png"))
    print(out.with_suffix(".pdf"))
    print(out.with_suffix(".svg"))

if __name__ == "__main__":
    main()
