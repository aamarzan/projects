#!/usr/bin/env python3
# khulna_district_map.py
# Khulna district ONLY map with journal-style callout labels (Place + n).
# Outputs: PNG/JPG (600 dpi) + PDF + SVG
#
# Fixes in this version:
# - Ensures axes border (spines) NEVER overlaps label boxes:
#     -> auto-expands x/y limits based on rendered text bounding boxes + padding
# - Prevents leader lines/arrows from bleeding into tick-label area:
#     -> uses clipping (clip_on=True, annotation_clip=True)
# - No legend, no footnote, no counts inside circles
# - Equal-sized small red circles (alpha=0.60)

from __future__ import annotations
from pathlib import Path
import argparse
import math

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
from shapely.geometry import Point


def fmt_lon(x, _): return f"{x:.1f}°E"
def fmt_lat(y, _): return f"{y:.1f}°N"


def guess_col(df: pd.DataFrame, want: list[str]) -> str:
    cols = {str(c).strip().lower(): c for c in df.columns}
    for w in want:
        if w in cols:
            return cols[w]
    raise ValueError(f"Could not find a required column among: {want}. Found: {list(df.columns)}")


def guess_district_name_col(gdf: gpd.GeoDataFrame) -> str:
    preferred = ["NAME_2", "NAME_EN", "NAME", "district", "District", "DISTRICT", "NAME_2_EN"]
    for c in preferred:
        if c in gdf.columns:
            return c
    for c in gdf.columns:
        if c != gdf.geometry.name and gdf[c].dtype == "object":
            return c
    raise ValueError("Could not detect district name column in districts file.")


def km_to_deg_lat(km: float) -> float:
    return float(km) / 111.0


def spread_overlaps_xy(df: pd.DataFrame, min_sep_km: float = 1.5) -> pd.DataFrame:
    """Small deterministic jitter for overlapping coords. Only affects Plot_Lon/Plot_Lat."""
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


def nudge_inside_polygon(lon: float, lat: float, poly, center_pt, eps=0.0015):
    """If jitter pushes a point outside the district polygon, pull it back in."""
    p = Point(lon, lat)
    if poly.contains(p):
        return lon, lat

    cx, cy = center_pt.x, center_pt.y
    for t in [0.98, 0.95, 0.90, 0.80, 0.70, 0.60, 0.50]:
        nx = cx + t * (lon - cx)
        ny = cy + t * (lat - cy)
        if poly.contains(Point(nx, ny)):
            return nx, ny

    b = poly.boundary
    nearest = b.interpolate(b.project(p))
    vx, vy = cx - nearest.x, cy - nearest.y
    norm = math.hypot(vx, vy) or 1.0
    return nearest.x + eps * (vx / norm), nearest.y + eps * (vy / norm)


def distribute_ys(y_desired, y_min, y_max, min_gap):
    """Spread y values to avoid overlap within [y_min, y_max]."""
    y = np.array(y_desired, dtype=float).copy()
    n = len(y)
    if n <= 1:
        return y

    max_gap = (y_max - y_min) / max(n - 1, 1)
    gap = min(min_gap, max_gap) if max_gap > 0 else min_gap

    order = np.argsort(-y)  # top to bottom
    y_sorted = y[order]

    for i in range(1, n):
        y_sorted[i] = min(y_sorted[i], y_sorted[i - 1] - gap)

    if y_sorted[-1] < y_min:
        y_sorted += (y_min - y_sorted[-1])

    for i in range(n - 2, -1, -1):
        y_sorted[i] = max(y_sorted[i], y_sorted[i + 1] + gap)

    if y_sorted[0] > y_max:
        y_sorted -= (y_sorted[0] - y_max)

    y_out = np.empty_like(y_sorted)
    y_out[order] = y_sorted
    return y_out


def draw_callout_labels(ax, df, poly_bounds):
    """
    - Labels placed on left/right rails (outside polygon but inside axes)
    - Leader lines are straight segments with an elbow
    - Arrowhead ALWAYS at the POINT (unambiguous)
    - Labels are guaranteed inside the axes border by expanding limits based on bbox
    """
    minx, miny, maxx, maxy = poly_bounds
    spanx = max(maxx - minx, 1e-9)
    spany = max(maxy - miny, 1e-9)
    midx = (minx + maxx) / 2.0

    # rails and elbows (initial)
    left_label_x  = minx - 0.22 * spanx
    right_label_x = maxx + 0.22 * spanx
    left_elbow_base  = minx - 0.05 * spanx
    right_elbow_base = maxx + 0.05 * spanx

    # y rail range
    y_min = miny + 0.07 * spany
    y_max = maxy - 0.07 * spany
    min_gap = 0.07 * spany

    left_df = df[df["Plot_Lon"].to_numpy(float) < midx].copy().sort_values("Plot_Lat", ascending=False)
    right_df = df[df["Plot_Lon"].to_numpy(float) >= midx].copy().sort_values("Plot_Lat", ascending=False)

    left_y = distribute_ys(left_df["Plot_Lat"].to_numpy(float), y_min, y_max, min_gap) if len(left_df) else np.array([])
    right_y = distribute_ys(right_df["Plot_Lat"].to_numpy(float), y_min, y_max, min_gap) if len(right_df) else np.array([])

    bbox_kw = dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.0)
    txt_kw = dict(fontsize=10.5, fontweight="bold", color="#111111", zorder=6, bbox=bbox_kw)
    seg_kw = dict(color="#444444", lw=0.9, alpha=0.85, zorder=3)
    arrowprops = dict(arrowstyle="-|>", lw=0.95, color="#444444", mutation_scale=12)

    left_texts = []
    for (_, r), ly in zip(left_df.iterrows(), left_y):
        label = f'{str(r["Place"]).strip()}  (n={int(r["Count"])})'
        t = ax.text(left_label_x, ly, label, ha="right", va="center", clip_on=True, **txt_kw)
        left_texts.append((t, float(r["Plot_Lon"]), float(r["Plot_Lat"]), float(ly)))

    right_texts = []
    for (_, r), ly in zip(right_df.iterrows(), right_y):
        label = f'{str(r["Place"]).strip()}  (n={int(r["Count"])})'
        t = ax.text(right_label_x, ly, label, ha="left", va="center", clip_on=True, **txt_kw)
        right_texts.append((t, float(r["Plot_Lon"]), float(r["Plot_Lat"]), float(ly)))

    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    inv = ax.transData.inverted()

    def elbow_fan(n, max_shift):
        if n <= 1:
            return [0.0]
        return np.linspace(-max_shift, max_shift, n).tolist()

    left_fan = elbow_fan(len(left_texts), 0.018 * spanx)
    right_fan = elbow_fan(len(right_texts), 0.018 * spanx)

    # draw leaders (CLIPPED to axes)
    for i, (t, px, py, ly) in enumerate(left_texts):
        bb = t.get_window_extent(renderer=renderer)
        (x1, _) = inv.transform((bb.x1, bb.y0))
        start_x = x1 + 0.012 * spanx
        elbow_x = left_elbow_base + left_fan[i]

        ax.plot([start_x, elbow_x], [ly, ly], **seg_kw)
        ax.plot([elbow_x, elbow_x], [ly, py], **seg_kw)
        ax.annotate("", xy=(px, py), xytext=(elbow_x, py),
                    arrowprops=arrowprops, annotation_clip=True, zorder=4)

    for i, (t, px, py, ly) in enumerate(right_texts):
        bb = t.get_window_extent(renderer=renderer)
        (x0, _) = inv.transform((bb.x0, bb.y0))
        start_x = x0 - 0.012 * spanx
        elbow_x = right_elbow_base + right_fan[i]

        ax.plot([start_x, elbow_x], [ly, ly], **seg_kw)
        ax.plot([elbow_x, elbow_x], [ly, py], **seg_kw)
        ax.annotate("", xy=(px, py), xytext=(elbow_x, py),
                    arrowprops=arrowprops, annotation_clip=True, zorder=4)

    # === CRITICAL FIX: expand axes limits based on real text bboxes + padding ===
    xmins, xmaxs, ymins, ymaxs = [], [], [], []
    for (t, _, _, _) in (left_texts + right_texts):
        bb = t.get_window_extent(renderer=renderer)
        (x0, y0) = inv.transform((bb.x0, bb.y0))
        (x1, y1) = inv.transform((bb.x1, bb.y1))
        xmins.append(x0); xmaxs.append(x1); ymins.append(y0); ymaxs.append(y1)

    if xmins:
        xmin_lbl = min(xmins); xmax_lbl = max(xmaxs)
        ymin_lbl = min(ymins); ymax_lbl = max(ymaxs)
    else:
        xmin_lbl = minx; xmax_lbl = maxx
        ymin_lbl = miny; ymax_lbl = maxy

    xpad = 0.06 * spanx   # inner padding so border never touches labels
    ypad = 0.05 * spany

    ax.set_xlim(min(minx, xmin_lbl) - xpad, max(maxx, xmax_lbl) + xpad)
    ax.set_ylim(min(miny, ymin_lbl) - ypad, max(maxy, ymax_lbl) + ypad)


def save_all(fig, out_dir: Path, out_prefix: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / out_prefix
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.12, facecolor="white")
    fig.savefig(base.with_suffix(".jpg"), dpi=600, bbox_inches="tight", pad_inches=0.12, facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.12, facecolor="white")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.12, facecolor="white")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel", required=True, type=Path)
    ap.add_argument("--districts", required=True, type=Path)
    ap.add_argument("--district_name", required=True, type=str)
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--out_prefix", required=True, type=str)
    ap.add_argument("--title", default="Khulna District: Case Distribution by Location")
    ap.add_argument("--min_sep_km", type=float, default=1.5)
    args = ap.parse_args()

    if not args.districts.exists():
        raise SystemExit(f"Districts file not found:\n  {args.districts}")

    try:
        df = pd.read_excel(args.excel)
    except PermissionError:
        raise SystemExit(
            f"PermissionError: Excel file is locked:\n  {args.excel}\n\n"
            "Close the Excel file (and OneDrive preview) then run again."
        )

    df = df.rename(columns={c: str(c).strip() for c in df.columns})
    place_col = guess_col(df, ["place", "place name", "place_name", "location", "union", "upazila"])
    count_col = guess_col(df, ["number", "count", "n", "cases"])
    lat_col   = guess_col(df, ["latitude", "lat"])
    lon_col   = guess_col(df, ["longitude", "lon", "lng"])

    df = df[[place_col, count_col, lat_col, lon_col]].copy()
    df.columns = ["Place", "Count", "Latitude", "Longitude"]

    df["Place"] = df["Place"].astype(str).str.strip()
    df.loc[df["Place"].str.lower().isin(["nan", "none", ""]), "Place"] = ""
    df["Count"] = pd.to_numeric(df["Count"], errors="coerce")
    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df = df.dropna(subset=["Count", "Latitude", "Longitude"]).copy()
    df["Count"] = df["Count"].astype(int)

    missing = df["Place"].eq("")
    if missing.any():
        df.loc[missing, "Place"] = [f"Unknown-{i}" for i in df[missing].index]

    agg = (
        df.groupby("Place", as_index=False)
          .agg(Count=("Count", "sum"),
               Latitude=("Latitude", "mean"),
               Longitude=("Longitude", "mean"))
    )

    agg = spread_overlaps_xy(agg, min_sep_km=args.min_sep_km)

    gdf = gpd.read_file(args.districts)
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    gdf = gdf.to_crs(4326)

    name_col = guess_district_name_col(gdf)
    target = args.district_name.strip().lower()
    sel = gdf[gdf[name_col].astype(str).str.strip().str.lower().eq(target)].copy()
    if sel.empty:
        options = sorted(gdf[name_col].dropna().astype(str).unique().tolist())
        raise SystemExit(f'District "{args.district_name}" not found.\nExample names: {options[:30]}')

    district_geom = sel.dissolve().geometry.iloc[0]
    district = gpd.GeoDataFrame({"name": [args.district_name]}, geometry=[district_geom], crs="EPSG:4326")

    # keep jittered points inside polygon
    poly = district_geom
    center_pt = poly.representative_point()
    fixed = agg.apply(lambda r: pd.Series(
        nudge_inside_polygon(float(r["Plot_Lon"]), float(r["Plot_Lat"]), poly, center_pt),
        index=["Plot_Lon", "Plot_Lat"]
    ), axis=1)
    agg[["Plot_Lon", "Plot_Lat"]] = fixed

    fig, ax = plt.subplots(figsize=(9.2, 9.2))
    district.plot(ax=ax, facecolor="#F7F7F7", edgecolor="#111111", linewidth=1.25, zorder=1)

    # Equal-sized red circles (60% transparent)
    ax.scatter(
        agg["Plot_Lon"], agg["Plot_Lat"],
        s=70,
        marker="o",
        facecolors="#D62728",
        edgecolors="#7A0010",
        linewidths=0.7,
        alpha=0.60,
        zorder=3
    )

    minx, miny, maxx, maxy = district.total_bounds
    draw_callout_labels(ax=ax, df=agg, poly_bounds=(minx, miny, maxx, maxy))

    # Axis formatting (push tick labels outward to avoid any visual crowding)
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_lat))
    ax.tick_params(axis="both", which="major", labelsize=9, width=1.0, length=5, pad=18, direction="out")
    ax.tick_params(top=True, labeltop=True, right=True, labelright=True)

    ax.xaxis.set_major_locator(MultipleLocator(0.3))
    ax.yaxis.set_major_locator(MultipleLocator(0.3))
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.25)

    for sp in ax.spines.values():
        sp.set_linewidth(1.15)
        sp.set_color("#111111")

    ax.set_title(args.title, fontsize=13.5, fontweight="bold", pad=24)
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.06, top=0.92)

    save_all(fig, args.out_dir, args.out_prefix)
    plt.close(fig)

    print("[OK] Saved outputs to:")
    for ext in [".png", ".jpg", ".pdf", ".svg"]:
        print(" ", (args.out_dir / (args.out_prefix + ext)))


if __name__ == "__main__":
    main()
