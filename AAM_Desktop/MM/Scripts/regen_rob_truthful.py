#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)

    domains = [c for c in df.columns if c.startswith("D")]
    # Mark review articles as NA across domains (prevalence-style RoB not applicable)
    is_review = df["Design"].astype(str).str.contains("review", case=False, na=False)

    df.loc[is_review, "Overall"] = "NA"
    for d in domains:
        df.loc[is_review, d] = "NA"

    df.to_csv(args.out_csv, index=False)
    print("Wrote:", Path(args.out_csv).resolve())

if __name__ == "__main__":
    main()