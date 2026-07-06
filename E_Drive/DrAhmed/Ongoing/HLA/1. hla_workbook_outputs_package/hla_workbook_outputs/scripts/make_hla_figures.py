#!/usr/bin/env python3
"""
Publication-ready HLA figure builder for:
HLA Class I and II variants associated with COVID-19 severity and mortality.

Expected frequency CSV/XLSX columns:
    locus, allele_label, allele_1_frequency, allele_2_frequency, level
where level is one of:
    allele_group
    two_field

Expected association CSV/XLSX columns:
    comparison, locus, allele, group1_name, group1_n, group1_positive,
    group2_name, group2_n, group2_positive, p_value

How to update input CSV:
    1. Replace the template rows with your real data.
    2. Enter counts where available. The script computes percentages from counts
       for the association forest plot.
    3. For frequency plots, frequencies must be numeric and must represent
       true percentages from 0 to 100. If any value exceeds 100, the script stops.
    4. Use correct HLA spelling: DQB1, not DBQ1.
    5. DQB1 is validated as Class II.

Run examples:
    python hla_publication_figures.py --frequency_csv hla_frequency_template.csv \
        --association_csv hla_association_from_manuscript_tables.csv --outdir figures

    python hla_publication_figures.py --frequency_csv raw_hla_frequency.xlsx \
        --association_csv raw_hla_association.xlsx --outdir figures
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

try:
    from scipy.stats import fisher_exact
except Exception:  # pragma: no cover
    fisher_exact = None


# -----------------------------
# Global style
# -----------------------------
OKABE_ITO = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#000000",
    "gray": "#666666",
}

LOCUS_ORDER = [
    "HLA-A", "HLA-B", "HLA-C", "HLA-DRB1", "HLA-DRB3", "HLA-DRB4", "HLA-DRB5",
    "HLA-DQA1", "HLA-DQB1", "HLA-DPA1", "HLA-DPB1", "HLA-DOA", "HLA-DOB",
    "HLA-DMA", "HLA-DMB"
]

CLASS_II_LOCI = {
    "HLA-DRA", "HLA-DRB1", "HLA-DRB3", "HLA-DRB4", "HLA-DRB5",
    "HLA-DQA1", "HLA-DQB1", "HLA-DPA1", "HLA-DPB1",
    "HLA-DOA", "HLA-DOB", "HLA-DMA", "HLA-DMB"
}

PANEL_LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def set_publication_style() -> None:
    """Use a clean MDPI/Biomedicines-compatible matplotlib style."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.linewidth": 0.6,
        "axes.edgecolor": "#333333",
        "axes.labelsize": 8,
        "axes.titlesize": 10,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 8,
        "figure.titlesize": 10,
        "savefig.dpi": 600,
    })


def read_table(path: str | Path) -> pd.DataFrame:
    """Read CSV or Excel."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input format: {suffix}. Use .csv, .xlsx, or .xls")


def normalize_locus(value: str) -> str:
    """Normalize locus names to HLA- prefix and correct common typo DBQ1 -> DQB1."""
    s = str(value).strip().upper().replace(" ", "")
    s = s.replace("DBQ1", "DQB1")
    if not s.startswith("HLA-"):
        s = "HLA-" + s
    return s


def normalize_allele_label(label: str, locus: Optional[str] = None) -> str:
    """
    Normalize readable HLA labels.

    Examples:
        HLA-A*02:01 -> A*02:01
        A0201 -> A*02:01 is not inferred automatically; provide cleaned labels if possible.
    """
    s = str(label).strip().replace("HLA-", "")
    s = s.replace("DBQ1", "DQB1")
    s = re.sub(r"\s+", "", s)
    # Keep labels such as B*15, B*51, DQB1*03, A*02:01.
    return s


def validate_dqb1_classification(df: pd.DataFrame) -> None:
    """Ensure DQB1 is Class II and typo DBQ1 is corrected/flagged."""
    text = " ".join([str(x) for x in df.astype(str).values.ravel()])
    if "DBQ1" in text.upper():
        raise ValueError("Detected typo 'DBQ1'. Please correct it to 'DQB1' before plotting.")
    if "class" in df.columns:
        mask = df["locus"].astype(str).str.contains("DQB1", case=False, na=False) | df.get("allele", pd.Series("", index=df.index)).astype(str).str.contains("DQB1", case=False, na=False) | df.get("allele_label", pd.Series("", index=df.index)).astype(str).str.contains("DQB1", case=False, na=False)
        bad = df.loc[mask & ~df["class"].astype(str).str.contains("II|2", case=False, na=False)]
        if len(bad):
            raise ValueError("HLA-DQB1 rows must be labeled Class II, not Class I.")


def load_and_validate_data(
    frequency_file: Optional[str | Path] = None,
    association_file: Optional[str | Path] = None,
) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Load and validate frequency and association data."""
    freq_df = None
    assoc_df = None

    if frequency_file:
        freq_df = read_table(frequency_file)
        required = {"locus", "allele_label", "allele_1_frequency", "allele_2_frequency", "level"}
        missing = required - set(freq_df.columns)
        if missing:
            raise ValueError(f"Frequency file missing required columns: {sorted(missing)}")

        freq_df = freq_df.copy()
        freq_df["locus"] = freq_df["locus"].map(normalize_locus)
        freq_df["allele_label"] = [
            normalize_allele_label(a, l) for a, l in zip(freq_df["allele_label"], freq_df["locus"])
        ]
        freq_df["level"] = freq_df["level"].astype(str).str.strip().str.lower()
        invalid_levels = sorted(set(freq_df["level"]) - {"allele_group", "two_field"})
        if invalid_levels:
            raise ValueError(f"Invalid level values: {invalid_levels}")

        for col in ["allele_1_frequency", "allele_2_frequency"]:
            freq_df[col] = pd.to_numeric(freq_df[col], errors="coerce")
            if freq_df[col].isna().any():
                bad_rows = freq_df.loc[freq_df[col].isna(), ["locus", "allele_label", col]]
                raise ValueError(f"Non-numeric frequency values in {col}:\n{bad_rows}")
            too_high = freq_df[freq_df[col] > 100]
            if len(too_high):
                raise ValueError(
                    f"Frequency values exceed 100 in {col}. These cannot be plotted as percentages.\n"
                    f"{too_high[['locus', 'allele_label', col]].to_string(index=False)}"
                )
            too_low = freq_df[freq_df[col] < 0]
            if len(too_low):
                raise ValueError(
                    f"Frequency values below 0 in {col}.\n"
                    f"{too_low[['locus', 'allele_label', col]].to_string(index=False)}"
                )

        validate_dqb1_classification(freq_df)

    if association_file:
        assoc_df = read_table(association_file)
        required = {
            "comparison", "locus", "allele", "group1_name", "group1_n", "group1_positive",
            "group2_name", "group2_n", "group2_positive", "p_value"
        }
        missing = required - set(assoc_df.columns)
        if missing:
            raise ValueError(f"Association file missing required columns: {sorted(missing)}")

        assoc_df = assoc_df.copy()
        assoc_df["locus"] = assoc_df["locus"].map(normalize_locus)
        assoc_df["allele"] = assoc_df["allele"].map(normalize_allele_label)

        for col in ["group1_n", "group1_positive", "group2_n", "group2_positive"]:
            assoc_df[col] = pd.to_numeric(assoc_df[col], errors="raise").astype(int)
        assoc_df["p_value"] = pd.to_numeric(assoc_df["p_value"], errors="raise")

        if (assoc_df["group1_positive"] > assoc_df["group1_n"]).any() or (assoc_df["group2_positive"] > assoc_df["group2_n"]).any():
            raise ValueError("Positive counts cannot exceed group n.")

        validate_dqb1_classification(assoc_df)

    return freq_df, assoc_df


def save_all_formats(fig: plt.Figure, out_prefix: str | Path) -> None:
    """Save a matplotlib figure as SVG, PDF, PNG 600 dpi, and TIFF 600 dpi."""
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    for ext in ["svg", "pdf", "png", "tiff"]:
        fig.savefig(out_prefix.with_suffix(f".{ext}"), bbox_inches="tight", dpi=600)


def _ordered_loci(df: pd.DataFrame) -> list[str]:
    present = list(dict.fromkeys(df["locus"].tolist()))
    return [l for l in LOCUS_ORDER if l in present] + [l for l in present if l not in LOCUS_ORDER]


def plot_frequency_facets(
    df: pd.DataFrame,
    level: str,
    out_prefix: str | Path,
    max_categories_per_locus: Optional[int] = None,
    separate_panels: bool = False,
) -> None:
    """
    Faceted horizontal grouped bar plots for Allele 1 vs Allele 2 frequencies.

    The plot is designed to replace radar/crowded bar charts for unordered HLA categories.
    """
    set_publication_style()
    d = df[df["level"].eq(level)].copy()
    if d.empty:
        raise ValueError(f"No rows found for level='{level}'.")

    # Remove rows where both allele frequencies are zero.
    d = d[(d["allele_1_frequency"] > 0) | (d["allele_2_frequency"] > 0)].copy()
    if d.empty:
        raise ValueError(f"All rows for level='{level}' have zero frequency.")

    d["max_frequency"] = d[["allele_1_frequency", "allele_2_frequency"]].max(axis=1)
    loci = _ordered_loci(d)

    rows_per_locus = {}
    panel_data = {}
    for locus in loci:
        sub = d[d["locus"].eq(locus)].sort_values("max_frequency", ascending=False).copy()
        if max_categories_per_locus is not None:
            sub = sub.head(max_categories_per_locus)
        sub = sub.sort_values("max_frequency", ascending=True)
        rows_per_locus[locus] = len(sub)
        panel_data[locus] = sub

    n_panels = len(loci)
    ncols = 3
    nrows = int(math.ceil(n_panels / ncols))
    height_ratios = []
    for r in range(nrows):
        row_loci = loci[r * ncols:(r + 1) * ncols]
        height_ratios.append(max([max(rows_per_locus[l], 2) for l in row_loci]) * 0.22 + 0.8)

    fig_width = 7.2
    fig_height = max(8.0, sum(height_ratios) + 0.8)
    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
    gs = GridSpec(nrows=nrows, ncols=ncols, figure=fig, height_ratios=height_ratios)

    legend_handles = None
    legend_labels = None
    xmax = min(100, max(5, float(d["max_frequency"].max()) * 1.15))

    for i, locus in enumerate(loci):
        ax = fig.add_subplot(gs[i // ncols, i % ncols])
        sub = panel_data[locus]
        y = np.arange(len(sub))
        h = 0.36

        b1 = ax.barh(y - h/2, sub["allele_1_frequency"], height=h, color=OKABE_ITO["blue"], label="Allele 1", linewidth=0)
        b2 = ax.barh(y + h/2, sub["allele_2_frequency"], height=h, color=OKABE_ITO["orange"], label="Allele 2", linewidth=0)

        ax.set_yticks(y)
        ax.set_yticklabels(sub["allele_label"])
        ax.set_xlim(0, xmax)
        ax.set_title(locus, pad=3, fontweight="bold")
        ax.grid(axis="x", color="#DDDDDD", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="both", length=2, width=0.5)
        ax.text(-0.16, 1.04, PANEL_LETTERS[i], transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="bottom", ha="left")
        if i // ncols == nrows - 1:
            ax.set_xlabel("Frequency (%)")
        else:
            ax.set_xlabel("")
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

        if separate_panels:
            panel_fig, panel_ax = plt.subplots(figsize=(3.4, max(1.8, len(sub) * 0.22 + 0.9)), constrained_layout=True)
            panel_ax.barh(y - h/2, sub["allele_1_frequency"], height=h, color=OKABE_ITO["blue"], label="Allele 1", linewidth=0)
            panel_ax.barh(y + h/2, sub["allele_2_frequency"], height=h, color=OKABE_ITO["orange"], label="Allele 2", linewidth=0)
            panel_ax.set_yticks(y)
            panel_ax.set_yticklabels(sub["allele_label"])
            panel_ax.set_xlim(0, xmax)
            panel_ax.set_title(locus, fontweight="bold")
            panel_ax.set_xlabel("Frequency (%)")
            panel_ax.grid(axis="x", color="#DDDDDD", linewidth=0.5)
            panel_ax.set_axisbelow(True)
            panel_ax.spines[["top", "right"]].set_visible(False)
            panel_ax.legend(frameon=False, loc="lower right")
            panel_prefix = Path(out_prefix).parent / f"{Path(out_prefix).stem}_{locus.replace('HLA-', '').replace('*', '').replace(':', '-')}"
            save_all_formats(panel_fig, panel_prefix)
            plt.close(panel_fig)

    # Hide unused panels.
    for j in range(n_panels, nrows * ncols):
        ax = fig.add_subplot(gs[j // ncols, j % ncols])
        ax.axis("off")

    fig.legend(legend_handles, legend_labels, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.01))
    save_all_formats(fig, out_prefix)
    plt.close(fig)


def compute_or_ci(a: int, n1: int, c: int, n2: int) -> tuple[float, float, float, float]:
    """
    Compute odds ratio and approximate 95% CI from 2x2 counts.
    a = group1 positive, b = group1 negative, c = group2 positive, d = group2 negative.
    Uses Haldane-Anscombe correction when a cell is zero.
    """
    b = n1 - a
    d = n2 - c
    aa, bb, cc, dd = map(float, [a, b, c, d])
    if min(aa, bb, cc, dd) == 0:
        aa += 0.5
        bb += 0.5
        cc += 0.5
        dd += 0.5
    odds_ratio = (aa * dd) / (bb * cc)
    se = math.sqrt(1/aa + 1/bb + 1/cc + 1/dd)
    lo = math.exp(math.log(odds_ratio) - 1.96 * se)
    hi = math.exp(math.log(odds_ratio) + 1.96 * se)
    if fisher_exact is not None:
        _, p_fisher = fisher_exact([[a, b], [c, d]], alternative="two-sided")
    else:
        p_fisher = np.nan
    return odds_ratio, lo, hi, p_fisher


def _section_label(row: pd.Series) -> str:
    comp = str(row["comparison"]).lower()
    if "mortality" in comp or "deceased" in comp or "recovered" in comp:
        return "Mortality outcome"
    return "Clinical severity"


def _direction_label(row: pd.Series) -> str:
    p1 = 100 * row["group1_positive"] / row["group1_n"]
    p2 = 100 * row["group2_positive"] / row["group2_n"]
    higher = row["group1_name"] if p1 >= p2 else row["group2_name"]
    h = str(higher).lower()
    if "deceased" in h:
        return "Higher in deceased"
    if "recovered" in h:
        return "Higher in recovered"
    if "d" in h or "critical" in h or "severe" in h or "c+d" in h or "b+c+d" in h:
        return "Higher in severe"
    return "Higher in non-critical"


def plot_association_forest(df: pd.DataFrame, out_prefix: str | Path) -> None:
    """
    Forest plot of HLA associations with outcomes.
    OR compares group1 vs group2; direction labels show which group has higher frequency.
    """
    set_publication_style()
    d = df.copy()
    rows = []
    for _, r in d.iterrows():
        or_, lo, hi, p_fisher = compute_or_ci(
            int(r["group1_positive"]), int(r["group1_n"]),
            int(r["group2_positive"]), int(r["group2_n"])
        )
        p1 = 100 * int(r["group1_positive"]) / int(r["group1_n"])
        p2 = 100 * int(r["group2_positive"]) / int(r["group2_n"])
        rows.append({
            **r.to_dict(),
            "section": _section_label(r),
            "direction": _direction_label(r),
            "pct1": p1,
            "pct2": p2,
            "odds_ratio": or_,
            "ci_low": lo,
            "ci_high": hi,
            "p_fisher": p_fisher,
            "label": f"{normalize_allele_label(r['allele'])}",
            "detail": f"{r['group1_name']}: {int(r['group1_positive'])}/{int(r['group1_n'])} ({p1:.1f}%)  |  "
                      f"{r['group2_name']}: {int(r['group2_positive'])}/{int(r['group2_n'])} ({p2:.1f}%)"
        })
    p = pd.DataFrame(rows)

    # Order sections and within each by OR.
    p["section_order"] = p["section"].map({"Mortality outcome": 0, "Clinical severity": 1}).fillna(9)
    p = p.sort_values(["section_order", "odds_ratio"], ascending=[True, False]).reset_index(drop=True)

    # Insert visual gaps between sections.
    y_positions = []
    y = 0
    prev = None
    for section in p["section"]:
        if prev is not None and section != prev:
            y += 1.0
        y_positions.append(y)
        y += 1
        prev = section
    p["y"] = y_positions

    height = max(4.8, 0.32 * len(p) + 1.9)
    fig, ax = plt.subplots(figsize=(7.2, height), constrained_layout=True)

    # Color by directional interpretation.
    color_map = {
        "Higher in deceased": OKABE_ITO["vermillion"],
        "Higher in recovered": OKABE_ITO["blue"],
        "Higher in severe": OKABE_ITO["orange"],
        "Higher in non-critical": OKABE_ITO["green"],
    }

    for _, r in p.iterrows():
        color = color_map.get(r["direction"], OKABE_ITO["gray"])
        ax.plot([r["ci_low"], r["ci_high"]], [r["y"], r["y"]], color=color, linewidth=1.2)
        ax.scatter(r["odds_ratio"], r["y"], color=color, s=22, zorder=3, edgecolor="white", linewidth=0.4)

    ax.axvline(1, color="#444444", linewidth=0.8, linestyle="--")
    ax.set_xscale("log")
    xmax = max(10, p["ci_high"].replace([np.inf], np.nan).max() * 1.2)
    xmin = min(0.05, p["ci_low"].replace([0], np.nan).min() / 1.2)
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel("Odds ratio, group 1 vs group 2 (log scale)")
    ax.set_yticks(p["y"])
    ax.set_yticklabels([f"{r.label}   {r.direction}" for r in p.itertuples()])
    ax.invert_yaxis()
    ax.grid(axis="x", which="major", color="#DDDDDD", linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    # Add table-like annotations on the right in axis coordinates.
    trans = ax.get_yaxis_transform()
    x_text = 1.02
    for _, r in p.iterrows():
        annotation = f"OR {r['odds_ratio']:.2f} [{r['ci_low']:.2f}, {r['ci_high']:.2f}], P={r['p_value']:.3g}"
        ax.text(x_text, r["y"], annotation, transform=trans, va="center", ha="left", fontsize=7)

    # Section headers
    for section, sub in p.groupby("section", sort=False):
        y_header = sub["y"].min() - 0.65
        ax.text(0.0, y_header, section, transform=trans, ha="left", va="bottom", fontsize=9, fontweight="bold")

    # Legend only once
    handles = [
        mpl.lines.Line2D([0], [0], marker="o", linestyle="", color=c, label=lab, markersize=5)
        for lab, c in color_map.items() if lab in set(p["direction"])
    ]
    ax.legend(handles=handles, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.20), ncol=2)

    save_all_formats(fig, out_prefix)
    plt.close(fig)

    # Save computed stats table beside figure.
    stats_path = Path(out_prefix).with_name(Path(out_prefix).stem + "_computed_OR_CI.csv")
    p.drop(columns=["section_order", "y"]).to_csv(stats_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate publication-ready HLA figures.")
    parser.add_argument("--frequency_csv", "--frequency_file", "--frequency", dest="frequency_file", default=None,
                        help="CSV/XLSX file with HLA frequency data.")
    parser.add_argument("--association_csv", "--association_file", "--association", dest="association_file", default=None,
                        help="CSV/XLSX file with HLA association counts.")
    parser.add_argument("--outdir", default="figures", help="Output directory.")
    parser.add_argument("--max_categories_per_locus", type=int, default=None,
                        help="Optional cap to show top N categories per locus.")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    freq_df, assoc_df = load_and_validate_data(args.frequency_file, args.association_file)

    if freq_df is not None:
        if freq_df.empty:
            print("Frequency input has no validated exact rows; skipping Figures 1 and 2.")
        else:
            plot_frequency_facets(
                freq_df, "allele_group",
                outdir / "Figure1_HLA_allele_group_frequency",
                max_categories_per_locus=args.max_categories_per_locus,
                separate_panels=False,
            )
            plot_frequency_facets(
                freq_df, "two_field",
                outdir / "Figure2_HLA_two_field_frequency",
                max_categories_per_locus=args.max_categories_per_locus,
                separate_panels=True,
            )

    if assoc_df is not None:
        plot_association_forest(
            assoc_df,
            outdir / "Figure3_HLA_outcome_associations",
        )


if __name__ == "__main__":
    main()
