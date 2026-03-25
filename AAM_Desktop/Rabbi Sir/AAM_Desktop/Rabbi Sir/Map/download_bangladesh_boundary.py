#!/usr/bin/env python3
# download_bangladesh_boundary.py
# Downloads Bangladesh ADM0 boundary GeoJSON from geoBoundaries (CC0).
#
# Usage:
#   python download_bangladesh_boundary.py --out "bangladesh_boundary.geojson"

import argparse
import json
from pathlib import Path

import requests

API = "https://www.geoboundaries.org/api/current/gbOpen/BGD/ADM0/"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output .geojson path")
    ap.add_argument("--simplified", action="store_true", help="Download simplified geometry (smaller file)")
    args = ap.parse_args()

    meta = requests.get(API, timeout=30).json()
    url = meta["simplifiedGeometryGeoJSON"] if args.simplified else meta["gjDownloadURL"]

    geojson_text = requests.get(url, timeout=60).text
    out = Path(args.out)
    out.write_text(geojson_text, encoding="utf-8")
    print(f"Saved: {out.resolve()}")

if __name__ == "__main__":
    main()
