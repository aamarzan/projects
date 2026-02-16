# mixed_op_table_median_full_v2_journal.py
# Journal-style table (WHITE background, thin gridlines, NO black separators)
# Uses MEDIAN stats (median + IQR OR median + 95% bootstrap CI)
# Outputs: PNG + PDF + XLSX (+ CSV as optional)

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from scipy.stats import chi2_contingency, mannwhitneyu, kruskal


# =========================
# USER SETTINGS
# =========================
INPUT_BASENAME = "Table_1_source"      # file name without extension
SHEET_NAME = "Sheet1"

# True => median (95% bootstrap CI)
# False => median (IQR)
USE_MEDIAN_95CI = True
BOOT_N = 3000
RANDOM_SEED = 42

# Use plain ASCII dash so Excel never shows mojibake like "â€“"
DASH = "-"

# --- Factors (ALL rows from your template) ---
FACTORS = [
    {"label": "Age (years)", "kind": "num", "col_candidates": ["Age", "Age (years)"]},
    {"label": "Sex (counts)", "kind": "sex", "col_candidates": ["Sex"]},
    {"label": "Time post ingestion (hours)", "kind": "num", "col_candidates": ["Time post ingestion", "Time post ingestion (hours)", "Time post ingestion (h)"]},
    {"label": "GCS score total", "kind": "num", "col_candidates": ["Gcs score total", "GCS score total"]},
    {"label": "POP score", "kind": "num", "col_candidates": ["Pop score", "POP score"]},
    {"label": "BMI (kg/m²)", "kind": "num", "col_candidates": ["BMI", "BMI (kg/m2)", "BMI (kg/m²)"]},
    {"label": "Poison volume (mL)", "kind": "num", "col_candidates": ["Poison volume", "Poison volume (mL)", "Poison volume (ml)", "Volume", "Volume (mL)"]},
    {"label": "Cholinesterase value (U/g Hb)", "kind": "num", "col_candidates": ["Cholinesterase value", "Cholinesterase value (U/g Hb)"]},
]

OUT_PNG = "MixedOP_Table_median_JOURNAL.png"
OUT_PDF = "MixedOP_Table_median_JOURNAL.pdf"
OUT_XLSX = "MixedOP_Table_summary.xlsx"
OUT_CSV = "MixedOP_Table_summary.csv"  # kept as optional


# =========================
# HELPERS
# =========================
def find_input_file(base_dir: Path) -> Path:
    for ext in (".xlsx", ".xls", ".xlsm"):
        p = base_dir / f"{INPUT_BASENAME}{ext}"
        if p.exists():
            return p
    wildcard = list(base_dir.glob(f"{INPUT_BASENAME}*.xls*"))
    if wildcard:
        return wildcard[0]
    raise FileNotFoundError(f"Could not find '{INPUT_BASENAME}.xlsx' (or .xls/.xlsm) in: {base_dir}")


def canonical(s: str) -> str:
    s = str(s).strip().lower()
    s = s.replace("²", "2")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    return s


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    cmap = {canonical(c): c for c in df.columns}
    for c in candidates:
        key = canonical(c)
        if key in cmap:
            return cmap[key]
    keys = list(cmap.keys())
    for c in candidates:
        key = canonical(c)
        for k in keys:
            if key and key in k:
                return cmap[k]
    return None


def clean_numeric_series(s: pd.Series) -> pd.Series:
    s2 = s.astype(str).str.strip()
    s2 = s2.replace({"": np.nan, "NA": np.nan, "NaN": np.nan, "nan": np.nan, "None": np.nan})
    s2 = s2.apply(lambda x: re.sub(r"[^0-9.\-]", "", x) if isinstance(x, str) else x)
    return pd.to_numeric(s2, errors="coerce")


def normalize_sex(s: pd.Series) -> pd.Series:
    s2 = s.astype(str).str.strip().str.upper()
    s2 = s2.replace({"M": "MALE", "F": "FEMALE"})
    s2 = s2.where(s2.isin(["MALE", "FEMALE"]), np.nan)
    return s2


def assign_group(poison_type: pd.Series) -> pd.Series:
    s = poison_type.astype(str).str.lower().str.strip()
    has_chlor = s.str.contains("chlorpyrifos", na=False)
    has_cyp = s.str.contains("cypermethrin", na=False)

    grp = pd.Series(index=s.index, dtype=object)
    grp[has_chlor & has_cyp] = "Chlorpyrifos + Cypermethrin"
    grp[has_chlor & ~has_cyp] = "Chlorpyrifos"
    grp[~has_chlor & has_cyp] = "Cypermethrin"
    return grp


def median_iqr(x: np.ndarray) -> tuple[float, float, float]:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (np.nan, np.nan, np.nan)
    med = float(np.median(x))
    q1 = float(np.quantile(x, 0.25))
    q3 = float(np.quantile(x, 0.75))
    return med, q1, q3


def bootstrap_median_ci(x: np.ndarray, n_boot: int = 2000, seed: int = 42) -> tuple[float, float, float]:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(x, size=(n_boot, x.size), replace=True), axis=1)
    med = float(np.median(x))
    lo = float(np.quantile(meds, 0.025))
    hi = float(np.quantile(meds, 0.975))
    return med, lo, hi


def fmt_stat(x: np.ndarray, use_ci: bool) -> str:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return "NA"
    if use_ci:
        med, lo, hi = bootstrap_median_ci(x, n_boot=BOOT_N, seed=RANDOM_SEED)
        return f"{med:.2f} ({lo:.2f}{DASH}{hi:.2f})"
    else:
        med, q1, q3 = median_iqr(x)
        return f"{med:.2f} ({q1:.2f}{DASH}{q3:.2f})"


def fmt_stat_with_n(x: np.ndarray, use_ci: bool, n_total_group: int) -> str:
    x_valid = x[np.isfinite(x)]
    n_used = int(x_valid.size)
    if n_used == 0:
        return "NA"
    s = fmt_stat(x_valid, use_ci)
    if n_total_group > 0 and n_used != n_total_group:
        s = f"{s} (n={n_used})"
    return s


def fmt_p(p: float) -> str:
    if p is None or not np.isfinite(p):
        return "NA"
    if p < 0.0001:
        return "<0.0001"
    return f"{p:.5g}"


def p_for_numeric(groups: list[np.ndarray]) -> float | None:
    gg = [g[np.isfinite(g)] for g in groups if g is not None and np.isfinite(g).sum() > 0]
    if len(gg) < 2:
        return None
    if len(gg) == 2:
        return float(mannwhitneyu(gg[0], gg[1], alternative="two-sided").pvalue)
    return float(kruskal(*gg).pvalue)


def p_for_sex(df_sub: pd.DataFrame, group_col: str, sex_col: str) -> float | None:
    tmp = df_sub[[group_col, sex_col]].dropna()
    if tmp.empty:
        return None
    tab = pd.crosstab(tmp[group_col], tmp[sex_col])
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return None
    _, p, _, _ = chi2_contingency(tab)
    return float(p)


# =========================
# BUILD SUMMARY
# =========================
def build_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    poison_col = find_col(df, ["Poison Type", "Poison", "Poison type"])
    if poison_col is None:
        raise KeyError("Could not find a 'Poison Type' column in the Excel sheet.")

    sex_col = find_col(df, ["Sex"])
    if sex_col is not None:
        df[sex_col] = normalize_sex(df[sex_col])

    df["Group"] = assign_group(df[poison_col])

    base_order = ["Chlorpyrifos + Cypermethrin", "Chlorpyrifos", "Cypermethrin"]
    present = [g for g in base_order if (df["Group"] == g).any()]
    df = df[df["Group"].isin(present)].copy()

    group_ns = {g: int((df["Group"] == g).sum()) for g in present}
    meta = {"group_order": present, "group_ns": group_ns}

    rows = []

    for spec in FACTORS:
        label = spec["label"]
        kind = spec["kind"]
        row = {"Factors": label}

        if kind == "sex":
            if sex_col is None:
                row["Overall (n)"] = 0
                row["Overall stat"] = "NA"
                for g in present:
                    row[g] = "NA"
                row["p"] = "NA"
                rows.append(row)
                continue

            n_valid = int(df[sex_col].notna().sum())
            row["Overall (n)"] = n_valid
            male = int((df[sex_col] == "MALE").sum())
            female = int((df[sex_col] == "FEMALE").sum())

            # Keep single-line in spreadsheet; we can wrap visually in figure
            row["Overall stat"] = f"MALE: {male}; FEMALE: {female}"

            for g in present:
                dfg = df[df["Group"] == g]
                m = int((dfg[sex_col] == "MALE").sum())
                f = int((dfg[sex_col] == "FEMALE").sum())
                row[g] = f"Male: {m}, Female: {f}"

            row["p"] = fmt_p(p_for_sex(df, "Group", sex_col))

        elif kind == "num":
            col = find_col(df, spec["col_candidates"])
            if col is None:
                row["Overall (n)"] = 0
                row["Overall stat"] = "NA"
                for g in present:
                    row[g] = "NA"
                row["p"] = "NA"
                rows.append(row)
                continue

            x_all = clean_numeric_series(df[col]).to_numpy()
            row["Overall (n)"] = int(np.isfinite(x_all).sum())
            row["Overall stat"] = fmt_stat(x_all, USE_MEDIAN_95CI)

            group_arrays = []
            for g in present:
                dfg = df[df["Group"] == g]
                xg = clean_numeric_series(dfg[col]).to_numpy()
                row[g] = fmt_stat_with_n(xg, USE_MEDIAN_95CI, n_total_group=group_ns[g])
                group_arrays.append(xg)

            row["p"] = fmt_p(p_for_numeric(group_arrays))

        else:
            raise ValueError(f"Unknown kind: {kind}")

        rows.append(row)

    out = pd.DataFrame(rows)
    out = out[["Factors", "Overall (n)", "Overall stat"] + present + ["p"]]
    return out, meta


# =========================
# DRAW JOURNAL FIGURE
# =========================
def draw_table_figure(summary: pd.DataFrame, meta: dict, out_png: Path, out_pdf: Path):
    group_order = meta["group_order"]
    group_ns = meta["group_ns"]

    headers = [
        "Factors",
        "Overall\n(n)",
        "Overall median\n(95% CI)" if USE_MEDIAN_95CI else "Overall median\n(IQR)",
    ]
    for g in group_order:
        if g == "Chlorpyrifos + Cypermethrin":
            headers.append(f"Chlorpyrifos +\nCypermethrin\nmedian (n={group_ns[g]})")
        elif g == "Chlorpyrifos":
            headers.append(f"Chlorpyrifos\nmedian (n={group_ns[g]})")
        else:
            headers.append(f"Cypermethrin\nmedian (n={group_ns[g]})")
    headers.append("p (test)")

    # Column widths
    if len(group_order) == 2:
        col_w = np.array([0.26, 0.09, 0.22, 0.22, 0.15, 0.06], dtype=float)
    else:
        col_w = np.array([0.23, 0.08, 0.19, 0.18, 0.16, 0.16, 0.08], dtype=float)
    col_w = col_w / col_w.sum()

    # Journal styling
    BG = "white"
    HEADER_BG = "#E9E9E9"
    ROW1 = "white"
    ROW2 = "#F6F6F6"
    GRID = "#6E6E6E"     # subtle grey
    TXT = "black"

    nrows = summary.shape[0]
    fig_w_in = 18
    fig_h_in = max(6.5, 1.4 + nrows * 0.75)

    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), dpi=300)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_axis_off()

    top, bottom = 0.98, 0.04
    left, right = 0.02, 0.98
    usable_h = top - bottom

    header_h = 0.16 * usable_h
    row_h = (usable_h - header_h) / nrows

    # X edges
    x_edges = [left]
    for w in col_w:
        x_edges.append(x_edges[-1] + (right - left) * w)

    # Header band
    ax.add_patch(
        Rectangle(
            (left, top - header_h),
            width=(right - left),
            height=header_h,
            transform=ax.transAxes,
            facecolor=HEADER_BG,
            edgecolor=GRID,
            linewidth=0.9,
        )
    )

    # Header text
    for j, h in enumerate(headers):
        x0, x1 = x_edges[j], x_edges[j + 1]
        ax.text(
            (x0 + x1) / 2,
            top - header_h / 2,
            h,
            ha="center",
            va="center",
            color=TXT,
            fontsize=12.5,
            fontweight="bold",
            linespacing=1.15,
            transform=ax.transAxes,
        )

    # Rows
    y = top - header_h
    for i in range(nrows):
        y0 = y - row_h
        bg = ROW1 if (i % 2 == 0) else ROW2

        ax.add_patch(
            Rectangle(
                (left, y0),
                width=(right - left),
                height=row_h,
                transform=ax.transAxes,
                facecolor=bg,
                edgecolor=GRID,
                linewidth=0.6,
            )
        )

        r = summary.iloc[i]

        # Wrap sex overall in figure only (optional)
        overall_stat = str(r["Overall stat"])
        if r["Factors"] == "Sex (counts)":
            overall_stat = overall_stat.replace("; ", ";\n")

        values = [r["Factors"], str(r["Overall (n)"]), overall_stat]
        for g in group_order:
            values.append(r[g])
        values.append(r["p"])

        # Bold p if significant (<0.05)
        p_txt = str(r["p"])
        try:
            p_val = float(p_txt) if (p_txt not in ["NA"] and not p_txt.startswith("<")) else (0.00005 if p_txt.startswith("<") else np.nan)
        except Exception:
            p_val = np.nan
        p_bold = np.isfinite(p_val) and (p_val < 0.05)

        for j, val in enumerate(values):
            x0, x1 = x_edges[j], x_edges[j + 1]
            ha = "left" if j == 0 else "center"
            x_text = x0 + 0.01 * (right - left) if j == 0 else (x0 + x1) / 2
            fw = "bold" if (j == len(values) - 1 and p_bold) else "normal"

            ax.text(
                x_text,
                y0 + row_h / 2,
                str(val),
                ha=ha,
                va="center",
                color=TXT,
                fontsize=11.3,
                fontweight=fw,
                linespacing=1.1,
                transform=ax.transAxes,
            )

        y = y0

    # Vertical grid lines (thin)
    for xe in x_edges:
        ax.plot([xe, xe], [bottom, top], color=GRID, linewidth=0.6, transform=ax.transAxes)

    # Outer border
    ax.add_patch(
        Rectangle(
            (left, bottom),
            width=(right - left),
            height=(top - bottom),
            transform=ax.transAxes,
            fill=False,
            edgecolor=GRID,
            linewidth=0.9,
        )
    )

    plt.tight_layout(pad=0.2)
    fig.savefig(out_png, dpi=600, facecolor=fig.get_facecolor(), bbox_inches="tight")
    fig.savefig(out_pdf, dpi=600, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def write_excel(summary: pd.DataFrame, out_xlsx: Path):
    # Plain write is enough; Excel will display "-" correctly
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        summary.to_excel(writer, index=False, sheet_name="Table1")


def main():
    base_dir = Path(__file__).resolve().parent
    xlsx_path = find_input_file(base_dir)
    df = pd.read_excel(xlsx_path, sheet_name=SHEET_NAME)

    summary, meta = build_summary(df)

    # XLSX (real Excel)
    write_excel(summary, base_dir / OUT_XLSX)

    # CSV (Excel-friendly UTF-8 with BOM)
    summary.to_csv(base_dir / OUT_CSV, index=False, encoding="utf-8-sig")

    # Journal-style figure
    draw_table_figure(summary, meta, base_dir / OUT_PNG, base_dir / OUT_PDF)

    print("Done.")
    print(f"Input:  {xlsx_path}")
    print(f"Saved:  {base_dir / OUT_PNG}")
    print(f"Saved:  {base_dir / OUT_PDF}")
    print(f"Saved:  {base_dir / OUT_XLSX}")
    print(f"Saved:  {base_dir / OUT_CSV}")


if __name__ == "__main__":
    main()
