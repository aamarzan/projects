#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.gridspec import GridSpec


# -----------------------------
# Defaults
# -----------------------------
DEFAULT_MASTER = Path(
    "/mnt/e/DrAhmed/Ongoing/WGS/Result/_BIOLOGY_LAYER_V2/PrimaryResults_v4_withBiology_166.csv"
)
DEFAULT_PLASMID_DIR = Path(
    "/mnt/e/DrAhmed/Ongoing/WGS/Result/Result copy/"
    "tormes_all+plasmid+serotype(Enterobacteriaceae)/plasmids"
)
DEFAULT_OUTDIR = Path(
    "/mnt/e/DrAhmed/Ongoing/WGS/Result/_G4_REMAINING/output/supplementary"
)
DEFAULT_STEM = "SupplementaryFigure_Plasmid_Replicon_BubbleMatrix"


# -----------------------------
# Styling helpers
# -----------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 17,
    "axes.labelsize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

PURPLE_CMAP = LinearSegmentedColormap.from_list(
    "premium_purple",
    ["#ede9f8", "#c8bfe8", "#9b8fcd", "#6c55b2", "#4b2e83", "#5a189a"],
)

SPECIES_SHORT = {
    "Serratia marcescens": "S. marcescens",
    "Serratia nevei": "S. nevei",
    "Acinetobacter baumannii": "A. baumannii",
    "Klebsiella pneumoniae": "K. pneumoniae",
    "Pseudomonas aeruginosa": "P. aeruginosa",
    "Escherichia coli": "E. coli",
    "Homo sapiens": "H. sapiens",
}


# -----------------------------
# Parsing helpers
# -----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a premium plasmid replicon architecture figure."
    )
    parser.add_argument(
        "--master",
        type=Path,
        default=DEFAULT_MASTER,
        help="Path to PrimaryResults_v4_withBiology_166.csv",
    )
    parser.add_argument(
        "--plasmid_dir",
        type=Path,
        default=DEFAULT_PLASMID_DIR,
        help="Directory containing *_plasmids.tab files",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help="Output directory",
    )
    parser.add_argument(
        "--stem",
        type=str,
        default=DEFAULT_STEM,
        help="Output filename stem without extension",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=None,
        help="Optional limit on top N replicons to plot",
    )
    return parser.parse_args()


def clean_str(x: object) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in {"", "na", "nan", "none", "-", "null"}:
        return ""
    return s


def short_species(species: str) -> str:
    return SPECIES_SHORT.get(species, species)


def clean_replicon_name(label: str) -> str:
    """
    Convert raw replicon names into compact, publication-friendly labels.
    """
    label = clean_str(label)

    mapping = {
        "IncFIB(K)_1_Kpn3": "IncFIB(K)",
        "IncFII_1_pKP91": "IncFII",
        "IncA/C2_1": "IncA/C2",
        "Col440I_1": "Col440I",
        "IncL/M(pMU407)_1_pMU407": "IncL/M(pMU407)",
        "Col440II_1": "Col440II",
        "ColRNAI_1": "ColRNAI",
        "FII(pBK30683)_1": "FII(pBK30683)",
        "pSM22_1": "pSM22",
        "pADAP_1": "pADAP",
        "IncFIB(pQil)_1_pQil": "IncFIB(pQil)",
        "IncFII(pCRY)_1_pCRY": "IncFII(pCRY)",
        "IncP6_1": "IncP6",
        "IncQ2_1": "IncQ2",
    }

    if label in mapping:
        return mapping[label]

    # Generic cleanup if unseen but similar pattern appears
    label = re.sub(r"_1$", "", label)
    label = re.sub(r"_\d+$", "", label)
    label = label.replace("_", " ")
    label = re.sub(r"\s+", " ", label).strip()
    return label


def wrap_replicon_label(label: str) -> str:
    """
    Create compact multi-line labels to reduce crowding.
    """
    label = clean_str(label)

    special = {
        "IncFIB(K)": "IncFIB\n(K)",
        "IncL/M(pMU407)": "IncL/M\n(pMU407)",
        "FII(pBK30683)": "FII\n(pBK30683)",
        "IncFIB(pQil)": "IncFIB\n(pQil)",
        "IncFII(pCRY)": "IncFII\n(pCRY)",
    }
    if label in special:
        return special[label]

    if len(label) <= 10:
        return label

    if "(" in label and not label.startswith("("):
        left, right = label.split("(", 1)
        return f"{left.rstrip()}\n({right}"

    if "/" in label and len(label) > 10:
        return label.replace("/", "/\n", 1)

    return label


def parse_numeric(value: object, default: float = 0.0) -> float:
    s = clean_str(value)
    if not s:
        return default
    try:
        return float(s)
    except Exception:
        return default


def normalize_token(token: str) -> str:
    token = clean_str(token)
    token = token.strip(";,|")
    token = re.sub(r"\s+", " ", token)

    # Remove trailing explicit count e.g. IncFIB(K)(2) -> IncFIB(K)
    token = re.sub(r"\s*\((\d+)\)\s*$", "", token)
    token = token.strip()

    token = clean_replicon_name(token)
    return token.strip()


def split_field_tokens(field_value: str) -> list[str]:
    """
    Split a semicolon/comma/pipe-delimited replicon field conservatively.
    Preferred separator is ';'. Falls back to ',' only when ';' absent.
    """
    s = clean_str(field_value)
    if not s:
        return []

    if ";" in s:
        raw = [x.strip() for x in s.split(";")]
    elif "|" in s and "," not in s:
        raw = [x.strip() for x in s.split("|")]
    else:
        raw = [x.strip() for x in s.split(",")]

    return [x for x in raw if clean_str(x)]


def parse_replicon_counter_from_field(field_value: str) -> Counter:
    """
    Parse a master-field style replicon list.

    Handles entries like:
    - IncFIB(K)
    - IncFIB(K)(2)
    - IncFIB(K) (2)
    """
    counter: Counter = Counter()

    for tok in split_field_tokens(field_value):
        tok = clean_str(tok)
        if not tok:
            continue

        count = 1
        m = re.match(r"^(.*?)(?:\s*\((\d+)\))\s*$", tok)
        if m:
            base = m.group(1).strip()
            maybe_n = m.group(2)
            if maybe_n is not None and maybe_n.isdigit():
                base_norm = normalize_token(base)
                if base_norm:
                    tok = base_norm
                    count = int(maybe_n)
                else:
                    tok = normalize_token(tok)
            else:
                tok = normalize_token(tok)
        else:
            tok = normalize_token(tok)

        if not tok:
            continue

        if tok.lower() in {"na", "none", "null", "unknown"}:
            continue

        counter[tok] += count

    return counter


def read_replicons_from_plasmid_tab(fp: Path) -> Counter:
    """
    Parse individual *_plasmids.tab files.
    Expected header includes a GENE column.
    """
    counter: Counter = Counter()

    try:
        with fp.open("r", encoding="utf-8", errors="replace") as f:
            lines = [ln.rstrip("\n") for ln in f if ln.strip()]
    except Exception:
        return counter

    if not lines:
        return counter

    header_idx = None
    header = None

    for i, ln in enumerate(lines):
        cols = ln.split("\t")
        cols_clean = [c.strip().lstrip("#").lower() for c in cols]
        if "gene" in cols_clean:
            header_idx = i
            header = cols
            break

    if header_idx is None or header is None:
        return counter

    header_lookup = {c.strip().lstrip("#").lower(): idx for idx, c in enumerate(header)}
    gene_idx = header_lookup.get("gene")

    if gene_idx is None:
        return counter

    for ln in lines[header_idx + 1:]:
        if ln.startswith("#"):
            continue
        cols = ln.split("\t")
        if gene_idx >= len(cols):
            continue

        gene = normalize_token(cols[gene_idx])
        if not gene:
            continue
        counter[gene] += 1

    return counter


# -----------------------------
# Data loading
# -----------------------------
def load_master_rows(master_fp: Path) -> list[dict[str, str]]:
    if not master_fp.exists():
        raise FileNotFoundError(f"Master file not found: {master_fp}")

    with master_fp.open("r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            rows.append({k: clean_str(v) for k, v in row.items()})

    if not rows:
        raise ValueError(f"Master file is empty: {master_fp}")

    required = {"Sample", "TopSpecies1"}
    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(
            f"Master file is missing required columns: {sorted(missing)}"
        )

    return rows


def build_species_replicon_table(
    master_rows: list[dict[str, str]],
    plasmid_dir: Path | None = None,
) -> tuple[dict[str, Counter], Counter]:
    """
    Preferred route:
      - use master_rows['Plasmid_Replicons'] when available
    Fallback:
      - parse *_plasmids.tab using sample -> TopSpecies1 from master
    """
    species_rep: dict[str, Counter] = defaultdict(Counter)
    total_rep: Counter = Counter()

    for row in master_rows:
        sample = clean_str(row.get("Sample", ""))
        species = clean_str(row.get("TopSpecies1", ""))

        if not sample or not species:
            continue

        plasmid_hits = parse_numeric(row.get("Plasmid_Hits", 0), default=0.0)
        field_counter = Counter()

        if "Plasmid_Replicons" in row:
            field_counter = parse_replicon_counter_from_field(
                clean_str(row.get("Plasmid_Replicons", ""))
            )

        if not field_counter and "Plasmid_Preview" in row:
            field_counter = parse_replicon_counter_from_field(
                clean_str(row.get("Plasmid_Preview", ""))
            )

        if not field_counter and plasmid_dir is not None:
            tab_fp = plasmid_dir / f"{sample}_plasmids.tab"
            if tab_fp.exists():
                field_counter = read_replicons_from_plasmid_tab(tab_fp)

        if not field_counter and plasmid_hits <= 0:
            continue

        for rep, n in field_counter.items():
            if n <= 0:
                continue
            species_rep[species][rep] += int(n)
            total_rep[rep] += int(n)

    return species_rep, total_rep


def order_species_and_replicons(
    species_rep: dict[str, Counter],
    total_rep: Counter,
    top_n: int | None = None,
) -> tuple[list[str], list[str], dict[str, dict[str, int]], dict[str, int]]:
    species_totals = {
        sp: sum(cnt.values())
        for sp, cnt in species_rep.items()
        if sum(cnt.values()) > 0
    }

    replicon_totals = {
        rep: int(n)
        for rep, n in total_rep.items()
        if int(n) > 0
    }

    if not species_totals:
        raise ValueError("No positive plasmid replicon signal was found.")

    replicon_order = sorted(
        replicon_totals,
        key=lambda x: (-replicon_totals[x], x.lower())
    )
    if top_n is not None:
        replicon_order = replicon_order[:top_n]

    species_order = sorted(
        species_totals,
        key=lambda x: (-species_totals[x], x.lower())
    )

    matrix = {
        sp: {rep: int(species_rep[sp].get(rep, 0)) for rep in replicon_order}
        for sp in species_order
    }

    replicon_totals = {rep: sum(matrix[sp][rep] for sp in species_order) for rep in replicon_order}
    species_totals = {sp: sum(matrix[sp][rep] for rep in replicon_order) for sp in species_order}

    return species_order, replicon_order, matrix, species_totals


# -----------------------------
# Plotting
# -----------------------------
def make_figure(
    species_order: list[str],
    replicon_order: list[str],
    matrix: dict[str, dict[str, int]],
    species_totals: dict[str, int],
    outdir: Path,
    stem: str,
) -> None:
    rep_values = [sum(matrix[sp][rep] for sp in species_order) for rep in replicon_order]
    max_count = max(rep_values) if rep_values else 1

    outdir.mkdir(parents=True, exist_ok=True)

    norm_bar = Normalize(vmin=min(rep_values) if rep_values else 0, vmax=max_count)
    bar_colors = [PURPLE_CMAP(0.18 + 0.78 * norm_bar(v)) for v in rep_values]

    # Taller figure + more top spacing
    fig = plt.figure(figsize=(22.0, 15.5))
    gs = GridSpec(
        2, 1,
        height_ratios=[1.32, 1.18],
        hspace=0.42,
        figure=fig
    )

    # -------- Panel A --------
    ax1 = fig.add_subplot(gs[0])

    y_labels_a = [wrap_replicon_label(x) for x in replicon_order[::-1]]
    vals_a = rep_values[::-1]
    cols_a = bar_colors[::-1]

    bars = ax1.barh(
        y_labels_a,
        vals_a,
        color=cols_a,
        edgecolor="#4b2e83",
        linewidth=0.9,
        height=0.50,
        zorder=3,
    )

    ax1.set_title(
        "A. Overall plasmid replicon burden",
        loc="left",
        fontweight="bold",
        pad=14
    )
    ax1.set_xlabel("Total replicon count")
    ax1.set_ylabel("Replicon")

    ax1.grid(axis="x", linestyle="-", alpha=0.18, zorder=0)
    ax1.grid(axis="y", visible=False)
    ax1.spines["left"].set_color("#777777")
    ax1.spines["bottom"].set_color("#777777")

    ax1.tick_params(axis="y", pad=10, labelsize=9.8)
    for lab in ax1.get_yticklabels():
        lab.set_linespacing(1.12)
        lab.set_verticalalignment("center")

    xmax = max(vals_a) if vals_a else 1
    ax1.set_xlim(0, xmax * 1.14)

    for bar, val in zip(bars, vals_a):
        ax1.text(
            bar.get_width() + xmax * 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{val}",
            va="center",
            ha="left",
            fontsize=10.2,
            fontweight="bold",
            color="#35224d",
        )

    # -------- Panel B --------
    ax2 = fig.add_subplot(gs[1])

    # Extra row/column breathing room
    x_step = 1.28
    y_step = 1.95
    x_positions = [i * x_step for i in range(len(replicon_order))]
    y_positions = [i * y_step for i in range(len(species_order))]

    norm_bubble = Normalize(vmin=1, vmax=max_count)

    for yi, sp in enumerate(species_order):
        for xi, rp in enumerate(replicon_order):
            count = matrix[sp][rp]
            if count <= 0:
                continue

            xp = x_positions[xi]
            yp = y_positions[yi]

            size = 170 + 88 * math.sqrt(count)

            ax2.scatter(
                xp,
                yp,
                s=size,
                c=[PURPLE_CMAP(0.20 + 0.78 * norm_bubble(count))],
                edgecolors="#4b2e83",
                linewidths=1.0,
                zorder=3,
            )

            txt_color = "white" if count >= max_count * 0.20 else "#2f244e"
            ax2.text(
                xp,
                yp,
                str(count),
                ha="center",
                va="center",
                fontsize=10.0,
                fontweight="bold",
                color=txt_color,
                zorder=4,
            )

    ax2.set_title(
        "B. Species-by-replicon bubble matrix",
        loc="left",
        fontweight="bold",
        pad=14
    )
    ax2.set_xlabel("Dominant plasmid replicons")
    ax2.set_ylabel("Species")

    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(
        [wrap_replicon_label(rp) for rp in replicon_order],
        rotation=0,
        ha="center",
    )
    ax2.tick_params(axis="x", pad=16)

    ax2.set_yticks(y_positions)
    ax2.set_yticklabels([short_species(sp) for sp in species_order])
    ax2.tick_params(axis="y", pad=7)
    ax2.invert_yaxis()

    ax2.set_xlim(min(x_positions) - 0.78, max(x_positions) + 0.86)
    ax2.set_ylim(max(y_positions) + 0.85, min(y_positions) - 0.65)

    ax2.grid(True, linestyle="-", alpha=0.16, zorder=0)
    ax2.spines["left"].set_color("#777777")
    ax2.spines["bottom"].set_color("#777777")

    # Colorbar
    sm = ScalarMappable(norm=norm_bubble, cmap=PURPLE_CMAP)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax2, fraction=0.028, pad=0.045)
    cbar.set_label("Replicon count")
    cbar.outline.set_edgecolor("#777777")

    # Bubble-size legend
    size_levels = [1, 10, 25, 50]
    handles = []
    for v in size_levels:
        handles.append(
            ax2.scatter(
                [], [], s=170 + 88 * math.sqrt(v),
                facecolor=PURPLE_CMAP(0.45),
                edgecolor="#4b2e83",
                linewidth=0.9
            )
        )

    leg = ax2.legend(
        handles,
        [str(v) for v in size_levels],
        title="Bubble size",
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.10, 1.02),
        borderaxespad=0.0,
    )
    plt.setp(leg.get_title(), fontsize=11, fontweight="bold")

    # Figure title and footnote
    fig.suptitle(
        "Plasmid replicon architecture",
        fontsize=24,
        fontweight="bold",
        y=0.985,
    )

    foot = (
        "Panel A summarizes total replicon burden across the plasmid-positive cohort. "
        "Panel B shows the species-level distribution of individual replicons. "
        "Only species and replicons with positive signal are displayed to improve readability "
        "and reduce unused space."
    )
    fig.text(
        0.5,
        0.045,
        foot,
        ha="center",
        va="center",
        fontsize=10.3,
        color="#444444",
    )

    # Lower panel A by reducing top plotting region and enlarge left margin
    fig.subplots_adjust(
        left=0.17,
        right=0.83,
        top=0.885,
        bottom=0.12,
    )

    png_fp = outdir / f"{stem}.png"
    pdf_fp = outdir / f"{stem}.pdf"
    svg_fp = outdir / f"{stem}.svg"

    fig.savefig(png_fp, dpi=900, bbox_inches="tight")
    fig.savefig(pdf_fp, bbox_inches="tight")
    fig.savefig(svg_fp, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_fp}")
    print(f"Saved: {pdf_fp}")
    print(f"Saved: {svg_fp}")


# -----------------------------
# Main
# -----------------------------
def main() -> int:
    args = parse_args()

    try:
        master_rows = load_master_rows(args.master)
        species_rep, total_rep = build_species_replicon_table(
            master_rows=master_rows,
            plasmid_dir=args.plasmid_dir if args.plasmid_dir.exists() else None,
        )

        species_order, replicon_order, matrix, species_totals = order_species_and_replicons(
            species_rep=species_rep,
            total_rep=total_rep,
            top_n=args.top_n,
        )

        make_figure(
            species_order=species_order,
            replicon_order=replicon_order,
            matrix=matrix,
            species_totals=species_totals,
            outdir=args.outdir,
            stem=args.stem,
        )

        print("\nSpecies totals used:")
        for sp in species_order:
            print(f"  {sp}: {species_totals[sp]}")

        print("\nReplicon totals used:")
        for rp in replicon_order:
            total = sum(matrix[sp][rp] for sp in species_order)
            print(f"  {rp}: {total}")

        return 0

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())