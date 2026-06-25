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
import re

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

REQ_COLS = ["Site", "Patient_ID", "Latitude", "Longitude", "Test Result", "Status of the Patient"]

# Optional manual nudges for the POSITIVE COUNT circles (lon, lat degrees)
NUMBER_OVERRIDE = {
    # "RMCH":  (0.55,  0.04),
    # "SOMCH": (-0.22, 0.18),
    # "SZMCH": (-0.18, 0.10),
}

# Optional manual nudges for SITE NAME placement relative to the number-circle center
LABEL_OVERRIDE = {
    # "SOMCH": (-0.10, -0.12),
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


def _first_nonempty(*vals):
    for v in vals:
        if pd.notna(v) and str(v).strip() != "":
            return v
    return np.nan

_WORD_NUMS = {
    "zero": 0, "no": 0, "none": 0,
    "one": 1, "a": 1, "an": 1,
    "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

def _clean_coord_text(s: str) -> str:
    s = str(s or "")
    s = s.replace("\xa0", " ")
    s = s.replace("′", "'").replace("’", "'").replace("‵", "'")
    s = s.replace("″", '"').replace("“", '"').replace("”", '"')
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _dms_to_decimal(deg, minute=None, second=None, hemi=None):
    deg = float(deg)
    minute = float(minute or 0)
    second = float(second or 0)
    val = abs(deg) + minute / 60.0 + second / 3600.0
    hemi = (hemi or "").upper()
    if hemi in {"S", "W"}:
        val = -val
    return val

def _extract_axis_values(text: str, axis: str):
    """
    axis='lat' -> values tagged N/S
    axis='lon' -> values tagged E/W
    Returns list of decimal values, supports decimal or DMS-ish tokens.
    """
    hemi_set = "NS" if axis == "lat" else "EW"
    vals = []

    # DMS / decimal followed by hemisphere
    pat = re.compile(
        rf'(-?\d+(?:\.\d+)?)\s*°?\s*'
        rf'(?:(\d+(?:\.\d+)?)\s*[\'′]?\s*)?'
        rf'(?:(\d+(?:\.\d+)?)\s*["″]?\s*)?'
        rf'([{hemi_set}])',
        flags=re.I
    )
    for m in pat.finditer(text):
        vals.append(_dms_to_decimal(m.group(1), m.group(2), m.group(3), m.group(4)))

    return vals

def parse_lat_lon(value):
    """
    Robust parser for strings like:
      24.8510° N, 89.3730° E
      23°45'0" N 90°42'0" E
      24.90310 , 91.85909
      21.43° N to 21.76° N, 92.00° E to 92.08° E  -> averaged
    Returns (lat, lon)
    """
    if pd.isna(value):
        return (np.nan, np.nan)

    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return (pd.to_numeric(value[0], errors="coerce"), pd.to_numeric(value[1], errors="coerce"))

    s = _clean_coord_text(value)

    lat_vals = _extract_axis_values(s, "lat")
    lon_vals = _extract_axis_values(s, "lon")
    if lat_vals and lon_vals:
        return (float(np.mean(lat_vals)), float(np.mean(lon_vals)))

    # Fallback: plain decimal pair
    nums = re.findall(r'-?\d+(?:\.\d+)?', s)
    nums = [float(x) for x in nums]
    if len(nums) >= 2:
        # If there are ranges, average first half as lat and second half as lon when possible.
        if len(nums) == 4:
            a, b, c, d = nums[:4]
            # common case: lat1 lat2 lon1 lon2
            if max(abs(a), abs(b)) <= 35 and max(abs(c), abs(d)) <= 100:
                return (float((a + b) / 2.0), float((c + d) / 2.0))
        a, b = nums[0], nums[1]
        # Guess order by Bangladesh bounds
        if 20 <= a <= 27 and 88 <= b <= 93:
            return (a, b)
        if 88 <= a <= 93 and 20 <= b <= 27:
            return (b, a)
        # Default to lat, lon
        return (a, b)

    return (np.nan, np.nan)

def infer_site_name(fp: Path, df: pd.DataFrame) -> str:
    if "Site" in df.columns:
        ser = df["Site"].dropna().astype(str).str.strip()
        if not ser.empty:
            return ser.iloc[0]
    stem = fp.stem.strip()
    return stem.split()[0].strip()

def _extract_companion_count(x) -> int:
    if pd.isna(x):
        return 0
    s = str(x).strip()
    if not s:
        return 0
    sl = s.lower()
    if any(k in sl for k in ["no companion", "none", "nil", "0 companion"]) or sl in {"0", "no"}:
        return 0
    if any(k in sl for k in ["could not be traced", "refused to give any information", "no idea"]):
        return 0

    # Explicit linked ID / paired patient style
    if re.search(r'[A-Z]{2,}-\d+', s):
        return 1

    nums = re.findall(r'\d+', s)
    if nums:
        return max(int(nums[0]), 0)

    for w, n in sorted(_WORD_NUMS.items(), key=lambda kv: -len(kv[0])):
        if re.search(rf'\b{re.escape(w)}\b', sl):
            return n
    return 0

def _normalize_patient_status_from_row(recruited_id, not_recruited_id, raw_status=None):
    if pd.notna(recruited_id) and str(recruited_id).strip():
        return "Alive"
    if pd.notna(not_recruited_id) and str(not_recruited_id).strip():
        return "Died"
    return _norm_status(raw_status)

def _transform_present_format(fp: Path, df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    site = infer_site_name(fp, df)

    # column aliases
    recruited_col = next((c for c in df.columns if str(c).strip().lower() in {"recruited", "recruited id", "recruited "} ), None)
    not_recruited_col = next((c for c in df.columns if str(c).strip().lower() in {"not recruited", "not recruited id"} ), None)
    loc_col = next((c for c in df.columns if "location" in str(c).strip().lower() and "latitude" not in str(c).lower() and "longitude" not in str(c).lower() and "lattitude" not in str(c).lower()), None)
    coord_col = next((c for c in df.columns if "longitude" in str(c).lower() or "lattitude" in str(c).lower() or "latitude" in str(c).lower()), None)
    test_col = next((c for c in df.columns if "positive/negative" in str(c).lower() or "status (positive" in str(c).lower()), None)
    comp_count_col = next((c for c in df.columns if "number of companions" in str(c).lower()), None)
    comp_status_col = next((c for c in df.columns if "status of the companions" in str(c).lower()), None)

    rows = []
    for i, r in df.iterrows():
        recruited_id = _first_nonempty(r.get(recruited_col), np.nan)
        not_recruited_id = _first_nonempty(r.get(not_recruited_col), np.nan)
        patient_id = _first_nonempty(recruited_id, not_recruited_id)
        if pd.isna(patient_id) or str(patient_id).strip() == "":
            # skip fully blank rows
            if all(pd.isna(r.get(c)) or str(r.get(c)).strip()=="" for c in df.columns):
                continue
            else:
                continue

        patient_id = str(patient_id).strip()
        lat, lon = parse_lat_lon(r.get(coord_col))
        role = "Patient"
        is_recruited = pd.notna(recruited_id) and str(recruited_id).strip() != ""

        test_result = _norm_test(r.get(test_col)) if is_recruited else "Unknown"
        patient_status = _normalize_patient_status_from_row(recruited_id, not_recruited_id, r.get(comp_status_col))

        row = {
            "Site": site,
            "Patient_ID": patient_id,
            "Latitude": lat,
            "Longitude": lon,
            "Test Result": test_result,
            "Status of the Patient": patient_status,
            "Role": role,
            "Index_Patient_ID": patient_id,
            "Original_Location": _first_nonempty(r.get(loc_col), np.nan),
            "Original_Companion_Count": _first_nonempty(r.get(comp_count_col), np.nan),
            "Original_Companion_Status": _first_nonempty(r.get(comp_status_col), np.nan),
            "Geo_Source": "Parsed from present-format coordinates",
            "Original_Row": i + 2,
        }
        rows.append(row)

        # Companion rows
        ccount = _extract_companion_count(r.get(comp_count_col))
        cstatus_raw = r.get(comp_status_col)
        cstatus = _norm_status(cstatus_raw)
        for j in range(ccount):
            rows.append({
                "Site": site,
                "Patient_ID": f"{patient_id}_C{j+1}",
                "Latitude": lat,
                "Longitude": lon,
                "Test Result": "Unknown",
                "Status of the Patient": cstatus,
                "Role": "Companion",
                "Index_Patient_ID": patient_id,
                "Original_Location": _first_nonempty(r.get(loc_col), np.nan),
                "Original_Companion_Count": _first_nonempty(r.get(comp_count_col), np.nan),
                "Original_Companion_Status": _first_nonempty(cstatus_raw, np.nan),
                "Geo_Source": f"Shared with {patient_id}",
                "Original_Row": i + 2,
            })

    out = pd.DataFrame(rows)
    return out

def load_site_excels(folder: Path) -> pd.DataFrame:
    files = sorted([p for p in folder.glob("*.xlsx") if not p.name.startswith("~$")])
    if not files:
        raise SystemExit(f"No .xlsx files found in: {folder}")

    # If both "present format" and "previous format" versions exist for the same site,
    # prefer the present-format file automatically.
    by_site = {}
    for fp in files:
        site_guess = fp.stem.split()[0].strip().upper()
        bucket = by_site.setdefault(site_guess, [])
        bucket.append(fp)

    selected = []
    for _, fps in by_site.items():
        present = [x for x in fps if "present format" in x.stem.lower()]
        previous = [x for x in fps if "previous format" in x.stem.lower()]
        if present:
            selected.extend(sorted(present))
        elif previous:
            selected.extend(sorted(previous))
        else:
            selected.extend(sorted(fps))

    frames = []
    for fp in selected:
        raw = pd.read_excel(fp)
        raw = raw.rename(columns={c: str(c).strip() for c in raw.columns})

        if all(c in raw.columns for c in REQ_COLS):
            df = raw.copy()
        else:
            df = _transform_present_format(fp, raw)

        missing = [c for c in REQ_COLS if c not in df.columns]
        if missing:
            raise SystemExit(f"{fp.name}: could not standardize required columns {missing}. Found: {list(df.columns)}")

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

    if not frames:
        raise SystemExit("No usable rows were parsed from the Excel files.")
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
    m = rad + 0.10
    return [( m, 0.0), (-m, 0.0), (0.0, m), (0.0,-m), ( m, 0.10), ( m,-0.10), (-m, 0.10), (-m,-0.10)]

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
        (0.35, 0.18), (0.35, -0.18), (-0.35, 0.18), (-0.35, -0.18),
        (0.45, 0.00), (-0.45, 0.00), (0.00, 0.28), (0.00, -0.28),
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
        if site in LABEL_OVERRIDE:
            dxl, dyl = LABEL_OVERRIDE[site]
        else:
            dxl, dyl = best_offset(nx, ny, label_candidates_around_circle(rad),
                                   avoid_lon, avoid_lat, used_label_points, w_used=3.0)
        lx, ly = nx + dxl, ny + dyl
        if not bd_safe.contains(Point(lx, ly)):
            lx, ly = nx, ny - (rad + 0.12)

        ax.text(lx, ly, site, fontsize=9.5, fontweight="bold", color="#222222",
                ha="center", va="center", zorder=13)
        used_label_points.append((lx, ly))

    # Layout
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.06, top=0.885)

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