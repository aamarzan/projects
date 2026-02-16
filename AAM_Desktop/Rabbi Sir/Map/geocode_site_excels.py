#!/usr/bin/env python3
# geocode_site_excels.py
# Fill Latitude/Longitude columns in site Excel files using OpenStreetMap Nominatim.
# - Does NOT modify any original text columns.
# - Writes new files with suffix: _GEOCODED.xlsx
#
# Usage:
#   python geocode_site_excels.py --data_folder "C:\path\to\folder_with_site_excels"
#
# Optional:
#   --country "Bangladesh" (default)
#   --sleep 1.1           (seconds between requests, Nominatim-friendly)
#   --email "you@example.com" (recommended by Nominatim usage policy)
#
# Notes:
# - Geocoding is inherently fuzzy. Always spot-check low-confidence matches.
# - For administrative places (Upazila/District), geocoding returns a centroid.

import argparse
import json
import time
import urllib.parse
from pathlib import Path

import pandas as pd
import requests

DEFAULT_COLS = {
    "lat": "Latitude",
    "lon": "Longitude",
    "src": "LatLon_Source",
    "q": "Geocode_Query",
    "disp": "Geocode_DisplayName",
    "cls": "Geocode_Class",
    "typ": "Geocode_Type",
    "imp": "Geocode_Importance",
    "conf": "Geocode_Confidence",
}

def nominatim_search(query: str, email: str | None = None) -> list[dict]:
    base = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": query,
        "format": "jsonv2",
        "limit": 1,
        "addressdetails": 0,
    }
    headers = {
        "User-Agent": "BangladeshCaseMap/1.0 (academic; contact: {})".format(email or "no-email-provided")
    }
    url = base + "?" + urllib.parse.urlencode(params)
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()

def load_cache(cache_path: Path) -> dict:
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    return {}

def save_cache(cache_path: Path, cache: dict) -> None:
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")

def ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    for k, col in DEFAULT_COLS.items():
        if col not in df.columns:
            df[col] = pd.NA
    return df

def is_missing(x) -> bool:
    return x is None or (isinstance(x, float) and pd.isna(x)) or (isinstance(x, str) and x.strip()=="") or pd.isna(x)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_folder", required=True, help="Folder containing site Excel files (*.xlsx)")
    ap.add_argument("--country", default="Bangladesh")
    ap.add_argument("--sleep", type=float, default=1.1)
    ap.add_argument("--email", default=None, help="Contact email for User-Agent (recommended)")
    args = ap.parse_args()

    folder = Path(args.data_folder)
    if not folder.exists():
        raise SystemExit(f"Folder not found: {folder}")

    cache_path = folder / "_geocode_cache.json"
    cache = load_cache(cache_path)

    xlsx_files = sorted([p for p in folder.glob("*.xlsx") if "_GEOCODED" not in p.stem])
    if not xlsx_files:
        raise SystemExit("No .xlsx files found in the folder.")

    for fp in xlsx_files:
        print(f"\nProcessing: {fp.name}")
        df = pd.read_excel(fp)
        df = ensure_cols(df)

        # If Geocode_Query is missing, try to build it from the first column containing 'Location'
        if DEFAULT_COLS["q"] not in df.columns:
            df[DEFAULT_COLS["q"]] = pd.NA

        if df[DEFAULT_COLS["q"]].isna().all():
            loc_cols = [c for c in df.columns if "location" in str(c).lower()]
            loc_col = loc_cols[0] if loc_cols else None
            if loc_col:
                df[DEFAULT_COLS["q"]] = df[loc_col].astype(str).str.strip() + f", {args.country}"

        # Geocode rows where lat/lon missing but query exists
        updated = 0
        for i, row in df.iterrows():
            q = row.get(DEFAULT_COLS["q"])
            if is_missing(q):
                continue
            lat = row.get(DEFAULT_COLS["lat"])
            lon = row.get(DEFAULT_COLS["lon"])
            if not (is_missing(lat) or is_missing(lon)):
                continue

            q_str = str(q).strip()
            if q_str in cache:
                result = cache[q_str]
            else:
                try:
                    hits = nominatim_search(q_str, email=args.email)
                except Exception as e:
                    print(f"  [WARN] Geocode failed for '{q_str}': {e}")
                    continue
                result = hits[0] if hits else None
                cache[q_str] = result
                save_cache(cache_path, cache)
                time.sleep(args.sleep)

            if not result:
                df.at[i, DEFAULT_COLS["src"]] = "Nominatim OSM (no match)"
                df.at[i, DEFAULT_COLS["conf"]] = 0
                continue

            df.at[i, DEFAULT_COLS["lat"]] = float(result.get("lat"))
            df.at[i, DEFAULT_COLS["lon"]] = float(result.get("lon"))
            df.at[i, DEFAULT_COLS["src"]] = "Nominatim OSM"
            df.at[i, DEFAULT_COLS["disp"]] = result.get("display_name")
            df.at[i, DEFAULT_COLS["cls"]] = result.get("class")
            df.at[i, DEFAULT_COLS["typ"]] = result.get("type")
            df.at[i, DEFAULT_COLS["imp"]] = result.get("importance")

            # Simple confidence heuristic:
            conf = 80
            if args.country.lower() not in q_str.lower():
                conf -= 20
            if any(k in q_str.lower() for k in ["union", "ward", "village", "street", "road", "para"]):
                conf -= 10  # often ambiguous
            df.at[i, DEFAULT_COLS["conf"]] = max(1, min(100, conf))
            updated += 1

        out = fp.with_name(fp.stem + "_GEOCODED.xlsx")
        df.to_excel(out, index=False)
        print(f"  Wrote: {out.name}  (updated rows: {updated})")

    print("\nDone. Check *_GEOCODED.xlsx files and spot-check low-confidence rows.")

if __name__ == "__main__":
    main()
