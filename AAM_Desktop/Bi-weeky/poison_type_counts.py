#!/usr/bin/env python3
"""
poison_type_counts.py

Counts each poison type from an Excel column (e.g., column O or "Types of poisoning"),
and also shows a *normalized* count to fix mismatches caused by trailing spaces,
non-breaking spaces, mixed casing, weird hidden characters, etc.

Usage examples:
  python poison_type_counts.py --file "data.xlsx" --sheet "Sheet1" --column O
  python poison_type_counts.py --file "data.xlsx" --sheet 0 --column "Types of poisoning"
  python poison_type_counts.py --file "data.xlsx" --sheet "Sheet1" --column O --out "poison_counts.xlsx"
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import Optional, Dict, List

import pandas as pd


_HIDDEN_CHARS_RE = re.compile(r"[\u200B-\u200D\uFEFF]")  # zero-width chars + BOM
_MULTI_SPACE_RE = re.compile(r"\s+")


def normalize_label(x: str) -> str:
    """
    Normalization that usually fixes Excel count mismatches:
    - converts non-breaking space to normal space
    - removes zero-width chars
    - trims and collapses multiple spaces
    - lowercases (so 'OPC' == 'opc')
    """
    if x is None:
        return ""
    s = str(x)

    # Replace non-breaking spaces etc.
    s = s.replace("\u00A0", " ").replace("\u202F", " ").replace("\t", " ")

    # Remove zero-width / hidden chars
    s = _HIDDEN_CHARS_RE.sub("", s)

    # Strip and collapse whitespace
    s = s.strip()
    s = _MULTI_SPACE_RE.sub(" ", s)

    # Lowercase to unify casing differences
    s = s.lower()

    return s


def looks_like_excel_column_letter(col: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z]{1,3}", col.strip()))


def read_target_column(
    file_path: Path,
    sheet: str,
    column: str,
    header: int,
) -> pd.Series:
    """
    Reads the target column either by Excel letter (e.g., O) or by header name.
    Returns a string Series.
    """
    if looks_like_excel_column_letter(column):
        # Read by Excel letter (works even if header names are messy)
        df = pd.read_excel(
            file_path,
            sheet_name=sheet,
            engine="openpyxl",
            usecols=column,   # e.g., "O"
            header=header,
            dtype=str,
        )
        # If you used a column letter, pandas will still assign some header name;
        # just take the first/only column
        s = df.iloc[:, 0]
    else:
        # Read by exact header name
        df = pd.read_excel(
            file_path,
            sheet_name=sheet,
            engine="openpyxl",
            header=header,
            dtype=str,
        )
        if column not in df.columns:
            # Helpful fallback: case-insensitive match
            lower_map = {c.lower(): c for c in df.columns}
            if column.lower() in lower_map:
                column = lower_map[column.lower()]
            else:
                raise KeyError(
                    f"Column '{column}' not found. Available columns:\n"
                    + "\n".join(map(str, df.columns))
                )
        s = df[column]

    return s.astype("string")


def build_variant_groups(raw: pd.Series) -> pd.DataFrame:
    """
    Shows which raw strings collapse into the same normalized label.
    This is usually where the mismatch comes from (e.g., 'OPC', 'OPC ', 'opc').
    """
    tmp = pd.DataFrame({"raw": raw.fillna("").astype(str)})
    tmp["norm"] = tmp["raw"].map(normalize_label)

    # Remove empty after normalization
    tmp = tmp[tmp["norm"] != ""]

    # Group raw variants per normalized label
    grp = (
        tmp.groupby("norm")["raw"]
        .apply(lambda x: sorted(set(x)))
        .reset_index()
        .rename(columns={"raw": "raw_variants"})
    )
    grp["n_variants"] = grp["raw_variants"].apply(len)

    # Show only those where multiple raw variants exist
    grp = grp[grp["n_variants"] > 1].sort_values(["n_variants", "norm"], ascending=[False, True])
    return grp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="Path to Excel file (.xlsx)")
    ap.add_argument("--sheet", default="Sheet1", help="Sheet name or index (default: Sheet1)")
    ap.add_argument("--column", default="O", help="Excel column letter (e.g., O) OR header name")
    ap.add_argument("--header", type=int, default=0, help="Header row index (default: 0)")
    ap.add_argument("--out", default=None, help="Optional output Excel file to save results")
    args = ap.parse_args()

    file_path = Path(args.file).expanduser().resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Sheet can be int or str
    sheet: str | int
    try:
        sheet = int(args.sheet)
    except ValueError:
        sheet = args.sheet

    raw = read_target_column(file_path, sheet, args.column, args.header)

    # Raw exact counts (no cleaning)
    raw_cleaned_for_count = raw.fillna("").astype(str)
    raw_exact_counts = (
        raw_cleaned_for_count[raw_cleaned_for_count.str.strip() != ""]
        .value_counts(dropna=False)
        .rename_axis("poison_type_raw")
        .reset_index(name="count_raw_exact")
    )

    # Normalized counts (fixes whitespace/case/hidden-char mismatches)
    norm = raw.fillna("").astype(str).map(normalize_label)
    norm_counts = (
        norm[norm != ""]
        .value_counts(dropna=False)
        .rename_axis("poison_type_normalized")
        .reset_index(name="count_normalized")
    )

    # Variant groups (diagnostics for mismatched totals)
    variant_groups = build_variant_groups(raw)

    # Print summaries
    print("\n=== TOTAL (non-empty) ===")
    print(f"Raw exact total:       {int(raw_exact_counts['count_raw_exact'].sum())}")
    print(f"Normalized total:      {int(norm_counts['count_normalized'].sum())}")

    print("\n=== RAW EXACT COUNTS (top 50) ===")
    print(raw_exact_counts.head(50).to_string(index=False))

    print("\n=== NORMALIZED COUNTS (top 50) ===")
    print(norm_counts.head(50).to_string(index=False))

    if not variant_groups.empty:
        print("\n=== VALUES THAT COLLAPSE TO THE SAME NORMALIZED LABEL (top 30) ===")
        # show a compact preview
        preview = variant_groups.copy()
        preview["raw_variants"] = preview["raw_variants"].apply(lambda v: "; ".join(v[:10]) + ("; ..." if len(v) > 10 else ""))
        print(preview.head(30).to_string(index=False))
    else:
        print("\nNo multi-variant groups found. (Counts likely already clean.)")

    # Optional save
    if args.out:
        out_path = Path(args.out).expanduser().resolve()
        with pd.ExcelWriter(out_path, engine="openpyxl") as w:
            raw_exact_counts.to_excel(w, index=False, sheet_name="raw_exact_counts")
            norm_counts.to_excel(w, index=False, sheet_name="normalized_counts")
            variant_groups.to_excel(w, index=False, sheet_name="variant_groups")
        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
