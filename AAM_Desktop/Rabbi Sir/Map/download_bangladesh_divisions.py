# download_bangladesh_divisions.py
from pathlib import Path
import argparse
import requests

ADM1_URL = "https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/BGD/ADM1/geoBoundaries-BGD-ADM1_simplified.geojson"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bangladesh_divisions_ADM1.geojson")
    args = ap.parse_args()

    out = Path(args.out)
    r = requests.get(ADM1_URL, timeout=60)
    r.raise_for_status()
    out.write_bytes(r.content)
    print("Saved:", out.resolve())

if __name__ == "__main__":
    main()
