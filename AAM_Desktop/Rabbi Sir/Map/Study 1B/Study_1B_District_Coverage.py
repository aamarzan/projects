#!/usr/bin/env python3
# Study_1B_District_Coverage.py
#
# Updates over the prior script:
#   1) Highlights the top 32 districts from the source Excel (so Chapainawabganj
#      is included in sky blue when it is within the first 32 rows).
#   2) Draws district labels for ALL districts in the GeoJSON, not only the
#      highlighted ones.
#   3) Keeps Excel spellings for labels whenever a GeoJSON district can be
#      matched to a source Excel district name.
#
# Expected output style remains the same: sky-blue highlighted districts,
# neutral non-highlighted districts, local label placement without arrows.
#
# Example usage:
#   python Study_1B_District_Coverage.py \
#       --districts_geojson "bangladesh_districts.geojson" \
#       --source_xlsx "District_Coverage_Table.xlsx" \
#       --out "District_Coverage_Map"

from __future__ import annotations
from pathlib import Path
import argparse
import re
import math

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator


HIGHLIGHT_FACE = "#DAF5FF"
NON_HIGHLIGHT_FACE = "#F7F7F7"
EDGE_COLOR = "#4A4A4A"
TEXT_COLOR = "#0F172A"
LABEL_SIZE = 7.3
DEFAULT_SOURCE_XLSX = Path("District_Coverage_Table.xlsx")
DEFAULT_OUT_STEM = Path("District_Coverage_Map")

# Manual fine-tuning for crowded districts.
# Keys should use the displayed label spellings.
DISTRICT_TEXT_OVERRIDE = {
    "Dhaka":            (90.38, 23.80),
    "Gazipur":          (90.50, 24.10),
    "Narayanganj":      (90.60, 23.72),
    "Munshiganj":       (90.44, 23.50),
    "Narsingdi":        (90.78, 23.98),
    "Manikganj":        (89.98, 23.90),
    "Tangail":          (89.98, 24.32),
    "Mymensingh":       (90.40, 24.74),
    "Kishoreganj":      (91.03, 24.31),
    "Netrakona":        (90.73, 24.89),
    "Sherpur":          (89.98, 25.03),
    "Jamalpur":         (89.97, 24.93),
    "Faridpur":         (89.82, 23.66),
    "Rajbari":          (89.65, 23.78),
    "Madaripur":        (90.17, 23.17),
    "Shariatpur":       (90.37, 23.22),
    "Gopalganj":        (89.82, 23.00),
    "Barisal":          (90.35, 22.72),
    "Barishal":         (90.35, 22.72),
    "Jhalokati":        (90.20, 22.62),
    "Pirojpur":         (89.95, 22.57),
    "Bhola":            (90.67, 22.54),
    "Patuakhali":       (90.36, 22.36),
    "Barguna":          (89.98, 22.14),
    "Khulna":           (89.40, 22.48),
    "Bagerhat":         (89.82, 22.66),
    "Satkhira":         (89.12, 22.47),
    "Narail":           (89.52, 23.17),
    "Magura":           (89.46, 23.42),
    "Jashore":          (89.22, 23.18),
    "Jessore":          (89.22, 23.18),
    "Jhenaidah":        (89.05, 23.44),
    "Chuadanga":        (88.78, 23.53),
    "Meherpur":         (88.63, 23.76),
    "Kushtia":          (88.98, 23.95),
    "Rajshahi":         (88.58, 24.38),
    "Natore":           (88.98, 24.43),
    "Naogaon":          (88.74, 24.90),
    "Joypurhat":        (89.10, 25.06),
    "Bogura":           (89.37, 24.86),
    "Bogra":            (89.37, 24.86),
    "Sirajganj":        (89.52, 24.34),
    "Pabna":            (89.34, 24.08),
    "Chapainawabganj":  (88.26, 24.60),
    "Nawabganj":        (88.26, 24.60),
    "Rangpur":          (89.22, 25.65),
    "Gaibandha":        (89.55, 25.32),
    "Kurigram":         (89.67, 25.80),
    "Lalmonirhat":      (89.46, 25.92),
    "Nilphamari":       (88.86, 25.94),
    "Dinajpur":         (88.73, 25.65),
    "Thakurgaon":       (88.47, 26.03),
    "Panchagarh":       (88.56, 26.33),
    "Sylhet":           (91.98, 24.93),
    "Sunamganj":        (91.34, 24.92),
    "Maulvibazar":      (91.90, 24.42),
    "Habiganj":         (91.46, 24.30),
    "Comilla":          (91.17, 23.45),
    "Cumilla":          (91.17, 23.45),
    "Brahmanbaria":     (91.11, 23.96),
    "Brahamanbaria":    (91.11, 23.96),
    "Chandpur":         (90.71, 23.23),
    "Noakhali":         (91.09, 22.87),
    "Feni":             (91.39, 23.02),
    "Lakshmipur":       (90.83, 22.96),
    "Chattogram":       (91.92, 22.43),
    "Chittagong":       (91.92, 22.43),
    "Cox's Bazar":      (92.05, 21.68),
    "Khagrachhari":     (91.96, 23.12),
    "Rangamati":        (92.18, 22.76),
    "Bandarban":        (92.22, 22.20),
}

# Normalize name spellings between source Excel and GeoJSON.
NAME_MAP_SOURCE_TO_GEOJSON = {
    "bogra": "Bogra",
    "bogura": "Bogra",
    "brahamanbaria": "Brahamanbaria",
    "brahmanbaria": "Brahamanbaria",
    "maulvibazar": "Maulvibazar",
    "moulvibazar": "Maulvibazar",
    "cox'sbazar": "Cox'SBazar",
    "coxs bazar": "Cox'SBazar",
    "coxsbazar": "Cox'SBazar",
    "cox's bazar": "Cox'SBazar",
    "jessore": "Jessore",
    "jashore": "Jessore",
    "chittagong": "Chittagong",
    "chattogram": "Chittagong",
    "nawabganj": "Nawabganj",
    "chapainawabganj": "Nawabganj",
    "netrakona": "Netrakona",
    "netrokona": "Netrakona",
    "barisal": "Barisal",
    "barishal": "Barisal",
    "comilla": "Comilla",
    "cumilla": "Comilla",
}

NAME_MAP_GEOJSON_TO_SOURCE = {
    "bogra": "Bogura",
    "bogura": "Bogura",
    "brahamanbaria": "Brahmanbaria",
    "brahmanbaria": "Brahmanbaria",
    "maulvibazar": "Maulvibazar",
    "moulvibazar": "Maulvibazar",
    "cox'sbazar": "Cox's Bazar",
    "coxs bazar": "Cox's Bazar",
    "coxsbazar": "Cox's Bazar",
    "cox's bazar": "Cox's Bazar",
    "jessore": "Jashore",
    "jashore": "Jashore",
    "chittagong": "Chattogram",
    "chattogram": "Chattogram",
    "nawabganj": "Chapainawabganj",
    "chapainawabganj": "Chapainawabganj",
    "netrakona": "Netrakona",
    "netrokona": "Netrakona",
    "barisal": "Barishal",
    "barishal": "Barishal",
    "comilla": "Comilla",
    "cumilla": "Comilla",
}


def fmt_lon(x, _):
    return f"{x:.1f}°E"


def fmt_lat(y, _):
    return f"{y:.1f}°N"


def _norm_name(s: str) -> str:
    s = str(s).strip().lower()
    s = s.replace("’", "'")
    s = re.sub(r"\s+", " ", s)
    return s


def source_to_geojson_name(s: str) -> str:
    key = _norm_name(s)
    return NAME_MAP_SOURCE_TO_GEOJSON.get(key, str(s).strip())


def geojson_to_label_name(s: str) -> str:
    key = _norm_name(s)
    return NAME_MAP_GEOJSON_TO_SOURCE.get(key, str(s).strip())


def find_name_col(gdf: gpd.GeoDataFrame) -> str:
    preferred = [
        "NAME_2", "ADM2_EN", "ADM2_NAME", "District", "district",
        "DIST_NAME", "DISTRICT", "shapeName", "ShapeName", "name", "NAME"
    ]
    for cand in preferred:
        if cand in gdf.columns:
            return cand

    for col in gdf.columns:
        if col == "geometry":
            continue
        vals = gdf[col].dropna().astype(str).str.strip()
        if len(vals) >= 20 and vals.nunique() >= 20 and vals.map(len).mean() <= 30:
            return col

    raise SystemExit(
        f"Could not identify district name column. Available columns: {list(gdf.columns)}"
    )


def find_source_district_col(src: pd.DataFrame) -> str:
    cols = {str(c).strip().lower(): c for c in src.columns}
    for cand in ["district", "districts", "name", "district name"]:
        if cand in cols:
            return cols[cand]
    raise SystemExit(f"Could not find district column in source Excel. Found: {list(src.columns)}")


def candidate_offsets():
    # More local nudges so all 64 labels have a better chance to remain readable.
    offsets = [(0.00, 0.00)]
    radii = [0.08, 0.12, 0.16, 0.20, 0.24, 0.28, 0.34, 0.40]
    dirs = [
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (-1, 1), (1, -1), (-1, -1),
        (2, 1), (-2, 1), (2, -1), (-2, -1),
        (1, 2), (-1, 2), (1, -2), (-1, -2),
    ]
    for r in radii:
        for dx, dy in dirs:
            mag = math.hypot(dx, dy)
            offsets.append((r * dx / mag, r * dy / mag))
    return offsets


def rects_overlap(a, b, pad=0.0):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (ax1 + pad < bx0 or bx1 + pad < ax0 or ay1 + pad < by0 or by1 + pad < ay0)


def estimate_text_box(x, y, text):
    w = 0.016 * len(str(text)) + 0.034
    h = 0.048
    return (x - w / 2, y - h / 2, x + w / 2, y + h / 2)


def inside_bounds(box, xlim, ylim):
    x0, y0, x1, y1 = box
    return x0 >= xlim[0] and x1 <= xlim[1] and y0 >= ylim[0] and y1 <= ylim[1]


def choose_label_position(anchor_x, anchor_y, name, placed_boxes, xlim, ylim):
    best = None
    best_score = -1e18
    for dx, dy in candidate_offsets():
        x = anchor_x + dx
        y = anchor_y + dy
        box = estimate_text_box(x, y, name)
        if not inside_bounds(box, xlim, ylim):
            continue
        overlap_count = sum(rects_overlap(box, old, pad=0.010) for old in placed_boxes)
        dist = math.hypot(dx, dy)
        score = -overlap_count * 100 - dist
        if overlap_count == 0 and dist == 0:
            return x, y, box
        if score > best_score:
            best_score = score
            best = (x, y, box)
    if best is not None:
        return best
    x = anchor_x
    y = anchor_y
    box = estimate_text_box(x, y, name)
    return x, y, box


def build_excel_label_map(src: pd.DataFrame, district_col: str) -> dict[str, str]:
    """
    Create a GeoJSON-key -> display-label map using the Excel names.
    If multiple Excel rows map to the same GeoJSON district (e.g.,
    Chapainawabganj and Nawabganj), keep the FIRST occurrence so the earlier,
    ranked entry is preserved.
    """
    mapping = {}
    for _, row in src.iterrows():
        excel_name = str(row[district_col]).strip()
        geo_key = source_to_geojson_name(excel_name)
        if geo_key not in mapping:
            mapping[geo_key] = excel_name
    return mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--districts_geojson", required=True, type=Path)
    ap.add_argument("--source_xlsx", required=False, type=Path, default=DEFAULT_SOURCE_XLSX)
    ap.add_argument("--out", required=False, type=Path, default=DEFAULT_OUT_STEM)
    ap.add_argument("--top_n", type=int, default=32)
    ap.add_argument("--sheet_name", default=0)
    args = ap.parse_args()

    src = pd.read_excel(args.source_xlsx, sheet_name=args.sheet_name)
    src.columns = [str(c).strip() for c in src.columns]

    district_col = find_source_district_col(src)
    sort_col = "All_Cases" if "All_Cases" in src.columns else None
    if sort_col is not None:
        src = src.sort_values(sort_col, ascending=False, kind="stable").reset_index(drop=True)

    top = src.head(args.top_n).copy()
    top["District_Excel"] = top[district_col].astype(str).str.strip()
    top["District_GeoJSON_Key"] = top["District_Excel"].apply(source_to_geojson_name)

    # Use all Excel rows for displayed label spellings, but keep first occurrence.
    excel_label_map = build_excel_label_map(src, district_col)

    districts = gpd.read_file(args.districts_geojson)
    if districts.crs is None:
        districts = districts.set_crs(4326)
    districts = districts.to_crs(4326)

    name_col = find_name_col(districts)
    districts["District_Name"] = districts[name_col].astype(str).str.strip()

    highlight_set = set(top["District_GeoJSON_Key"].astype(str).str.strip())
    districts["Highlight"] = districts["District_Name"].isin(highlight_set)

    districts["Label_Name"] = districts["District_Name"].map(excel_label_map).fillna(
        districts["District_Name"].apply(geojson_to_label_name)
    )

    matched = set(districts.loc[districts["Highlight"], "District_Name"])
    missing = sorted(highlight_set - matched)

    fig, ax = plt.subplots(figsize=(10.8, 12.4))

    districts.loc[~districts["Highlight"]].plot(
        ax=ax, facecolor=NON_HIGHLIGHT_FACE, edgecolor=EDGE_COLOR,
        linewidth=0.7, zorder=1
    )
    districts.loc[districts["Highlight"]].plot(
        ax=ax, facecolor=HIGHLIGHT_FACE, edgecolor=EDGE_COLOR,
        linewidth=0.9, zorder=2
    )

    ax.set_title(
        "Top Districts Covering the Majority of Cases",
        fontsize=14, fontweight="bold", pad=20
    )

    ax.xaxis.set_major_formatter(FuncFormatter(fmt_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_lat))
    ax.tick_params(axis="both", which="major", labelsize=9, width=1.0, length=5)
    ax.tick_params(top=True, labeltop=True, right=True, labelright=True)
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.yaxis.set_major_locator(MultipleLocator(1.0))
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.25)

    minx, miny, maxx, maxy = districts.total_bounds
    xlim = (minx - 0.3, maxx + 0.4)
    ylim = (miny - 0.3, maxy + 0.4)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    # Label all districts, not only the highlighted ones.
    label_df = districts.copy()
    label_df["area"] = label_df.geometry.area
    # Place larger districts first; if areas tie, highlighted ones first.
    label_df = label_df.sort_values(["area", "Highlight"], ascending=[False, False]).reset_index(drop=True)

    placed_boxes = []
    placed_names = []
    for _, row in label_df.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        name = row["Label_Name"]

        if name in DISTRICT_TEXT_OVERRIDE:
            lx, ly = DISTRICT_TEXT_OVERRIDE[name]
            box = estimate_text_box(lx, ly, name)
            if not inside_bounds(box, xlim, ylim):
                lx, ly = geom.representative_point().x, geom.representative_point().y
                lx, ly, box = choose_label_position(lx, ly, name, placed_boxes, xlim, ylim)
        else:
            anchor = geom.representative_point()
            lx, ly, box = choose_label_position(anchor.x, anchor.y, name, placed_boxes, xlim, ylim)

        ax.text(
            lx, ly, name,
            fontsize=LABEL_SIZE,
            fontweight="bold",
            color=TEXT_COLOR,
            ha="center", va="center", zorder=5,
            bbox=dict(
                boxstyle="round,pad=0.10",
                facecolor="white",
                edgecolor="none",
                alpha=0.72
            )
        )
        placed_boxes.append(box)
        placed_names.append(name)

    if missing:
        print("[WARNING] These highlighted source districts were not matched in the GeoJSON:")
        for m in missing:
            print(" -", m)

    if len(placed_names) != len(districts):
        print(f"[WARNING] Labels drawn: {len(placed_names)} / {len(districts)} districts")

    fig.subplots_adjust(left=0.06, right=0.97, bottom=0.07, top=0.91)

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
    print(f"Highlighted districts requested: {len(highlight_set)}")
    print(f"District labels drawn: {len(placed_names)}")


if __name__ == "__main__":
    main()
