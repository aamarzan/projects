#!/usr/bin/env python3
# download_bangladesh_districts.py
# Downloads Bangladesh ADM2 (districts) GeoJSON and saves locally.

from pathlib import Path
import argparse
import sys
import urllib.request

GADM_ADM2_URL = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_BGD_2.json"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path,
                    help="Output path, e.g. ...\\Map\\bangladesh_districts.geojson")
    args = ap.parse_args()

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        print(f"[INFO] Downloading districts (ADM2) from:\n  {GADM_ADM2_URL}")
        urllib.request.urlretrieve(GADM_ADM2_URL, out)
        print(f"[OK] Saved: {out}")
    except Exception as e:
        print(f"[ERROR] Failed to download: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
