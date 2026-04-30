# table2_analysis_mixedop_v2.py
# ------------------------------------------------------------
# Table 2 analysis from table_2.xlsx:
#   1) AE (patient-level wide): analyze ALL AEs separately
#   2) AE grouping (as many as possible) + export mapping/unmapped
#   3) Mortality (Deceased sheet): parse mixed-layout sheet and compute properly
#
# Outputs:
#   - Table2_AllOutputs.xlsx  (multiple sheets)
#   - CSVs (utf-8-sig) for Excel-safe reading
# ------------------------------------------------------------

from __future__ import annotations

import math
import re
import itertools
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, chi2_contingency


# =========================
# USER SETTINGS
# =========================
INPUT_XLSX = "table_2.xlsx"
SHEET_AE = "AE"
SHEET_DECEASED = "Deceased"

# Use plain ASCII dash everywhere (prevents Excel mojibake like â€“)
DASH = "-"

# If AE sheet is incomplete (rare), you can force denominators here:
DEFAULT_DENOMS = {
    "Mixed OP": 812,
    "Chlorpyrifos": 406,
    "Cypermethrin": 22,
}

# Drop symptoms with 0 events in all 3 groups?
DROP_ALL_ZERO_SYMPTOMS = True

# Output folder settings
OUTPUT_ROOT = "outputs"                 # created under the script folder
OUTPUT_RUN_SUBFOLDER = True             # make a subfolder named after the input xlsx (recommended)

# --- Exploratory grouping explosion controls ---
ENABLE_EXPLORATORY_GROUPS = True

# Build extra groups from:
INCLUDE_BASE_GROUPS = True              # your AE_GROUPS as-is
INCLUDE_KEYWORD_GROUPS = True           # group per keyword phrase
INCLUDE_TOKEN_GROUPS = True             # group per token extracted from symptom names
INCLUDE_UNION_GROUPS = True             # unions of base groups
INCLUDE_INTERSECTION_GROUPS = True      # intersections of base groups

MAX_UNION_SIZE = 3                      # unions of 2..3 base groups
MIN_TOKEN_LEN = 5
TOKEN_STOPWORDS = {
    "acute","since","with","without","related","status","syndrome","shock",
    "respiratory","cardiac","neurological","neuromuscular","ventilation"
}
MIN_COLS_PER_AUTO_GROUP = 2             # skip trivial auto groups
MAX_COLS_PER_GROUP = 60                 # safety cap
MAX_GROUPS_EXPORT = 1500                # cap exported groups (set None for unlimited; not recommended)
MIN_TOTAL_EVENTS = 1                    # skip groups with 0 events in all 3 arms

# Output files
OUT_ALL_XLSX = "Table2_AllOutputs.xlsx"

OUT_AE_ALL_CSV = "Table2_AE_AllSymptoms.csv"
OUT_AE_GROUPED_CSV = "Table2_AE_Grouped.csv"
OUT_AE_MAPPING_CSV = "Table2_AE_SymptomMapping.csv"
OUT_AE_UNMAPPED_CSV = "Table2_AE_Unmapped.csv"
OUT_MORTALITY_CSV = "Table2_Mortality.csv"


# =========================
# GROUPING RULES (expanded to group "as many as possible")
# NOTE: order matters if a symptom matches multiple groups.
# =========================
AE_GROUPS: Dict[str, List[str]] = {
    "Respiratory complications": [
        "respiratory arrest",
        "aspiration pneumonia",
        "acute respiratory distress syndrome",
        "shortness of breath",
        "gasping",
        "basal crepitations",
        "bronchorrhoea",
        "bronchorrhea",
    ],
    "Cardiac complications": [
        "cardiac arrest",
        "bradycardia",
        "bradicardia",
        "dysrhythmia",
        "arrhythmia",
        "shock",
        "hypotension",
        "hypertension",
    ],
    "Neurological complications": [
        "coma",
        "delirium",
        "seizure",
        "semiconscious",
        "unconscious",
    ],
    "Ventilation crisis": [
        "ventilation related",
        "ventilation",
        "ventilator",
        "intubation",
        "mechanical ventilation",
    ],
    "Neuromuscular complications": [
        "neuromuscular weakness",
        "neuromascular weakness",
        "intermediate syndrome",
        "intermediate",
        "fasciculation",
    ],
}

# =========================
# PI / USER-DEFINED GROUPS (ADD YOUR OWN HERE)
# These will be included in:
#   - mapping
#   - AE_Grouped
#   - AE_Grouped_Exploratory
# =========================
PI_GROUPS: Dict[str, List[str]] = {
    # Example (replace with your exact preferred grouping):
    # "PI: Clinical complications (broad)": [
    #     "respiratory arrest", "aspiration pneumonia", "cardiac arrest", "shock",
    #     "coma", "seizure", "intubation", "mechanical ventilation"
    # ],
}

# Prevent accidental overwrite of default group names
_dups = set(AE_GROUPS).intersection(PI_GROUPS)
if _dups:
    raise ValueError(f"Duplicate group names found in PI_GROUPS: {_dups}")

# Master group dictionary used everywhere
ALL_GROUPS: Dict[str, List[str]] = {**AE_GROUPS, **PI_GROUPS}

# =========================
# HELPERS
# =========================
def canonical(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9 %/._-]+", "", s)
    return s

def to_bool_series(s: pd.Series) -> pd.Series:
    # Accepts Yes/No, 1/0, true/false, present/absent
    if s.dtype.kind in "biufc":
        return s.fillna(0).astype(float) > 0
    x = s.astype(str).str.strip().str.lower()
    true_set = {"1", "yes", "y", "true", "present", "positive", "pos"}
    false_set = {"0", "no", "n", "false", "absent", "negative", "neg", "", "nan", "none"}
    return x.apply(lambda v: True if v in true_set else (False if v in false_set else False))

def assign_group_from_poison(poison_type: pd.Series) -> pd.Series:
    s = poison_type.astype(str).str.lower()
    has_chlor = s.str.contains("chlorpyrifos", na=False)
    has_cyp = s.str.contains("cypermethrin", na=False)

    grp = pd.Series(index=s.index, dtype=object)
    grp[has_chlor & has_cyp] = "Mixed OP"
    grp[has_chlor & ~has_cyp] = "Chlorpyrifos"
    grp[~has_chlor & has_cyp] = "Cypermethrin"
    return grp

def wilson_ci(x: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if n <= 0:
        return (np.nan, np.nan)
    p = x / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) / n) + ((z * z) / (4 * n * n))) / denom
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return lo, hi

def rd_newcombe_wilson(x1: int, n1: int, x2: int, n2: int) -> Tuple[float, float, float]:
    # RD = p1 - p2; CI = [L1 - U2, U1 - L2]
    if n1 <= 0 or n2 <= 0:
        return (np.nan, np.nan, np.nan)
    p1 = x1 / n1
    p2 = x2 / n2
    l1, u1 = wilson_ci(x1, n1)
    l2, u2 = wilson_ci(x2, n2)
    rd = p1 - p2
    lo = l1 - u2
    hi = u1 - l2
    return rd, lo, hi

def or_ci_haldane(x1: int, n1: int, x2: int, n2: int) -> Tuple[float, float, float]:
    # OR with Haldane-Anscombe correction if any zero cell
    a = float(x1)
    b = float(n1 - x1)
    c = float(x2)
    d = float(n2 - x2)

    if any(v == 0 for v in [a, b, c, d]):
        a += 0.5; b += 0.5; c += 0.5; d += 0.5

    orv = (a * d) / (b * c)
    se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    lo = math.exp(math.log(orv) - 1.96 * se)
    hi = math.exp(math.log(orv) + 1.96 * se)
    return float(orv), float(lo), float(hi)

def pvals_2x2(x1: int, n1: int, x2: int, n2: int) -> Tuple[float, float, float]:
    # Returns: Fisher, Chi-square (no continuity), Chi-square (Yates)
    a = x1
    b = n1 - x1
    c = x2
    d = n2 - x2
    table = np.array([[a, b], [c, d]], dtype=float)

    # Fisher
    try:
        fp = fisher_exact([[a, b], [c, d]], alternative="two-sided")[1]
    except Exception:
        fp = np.nan

    # Chi-square no continuity
    try:
        _, cp_nc, _, _ = chi2_contingency(table, correction=False)
    except Exception:
        cp_nc = np.nan

    # Chi-square with Yates
    try:
        _, cp_y, _, _ = chi2_contingency(table, correction=True)
    except Exception:
        cp_y = np.nan

    return float(fp), float(cp_nc), float(cp_y)

def fmt_count_pct(x: int, n: int) -> str:
    if n <= 0:
        return "NA"
    return f"{x}/{n} ({(x/n)*100:.1f}%)"

def fmt_rd(rd: float, lo: float, hi: float) -> str:
    if not np.isfinite(rd) or not np.isfinite(lo) or not np.isfinite(hi):
        return "NA"
    # Use "to" for CI range (no special dash problems)
    return f"{rd*100:.1f}% ({lo*100:.1f} to {hi*100:.1f})"

def fmt_or(orv: float, lo: float, hi: float) -> str:
    if not np.isfinite(orv) or not np.isfinite(lo) or not np.isfinite(hi):
        return "NA"
    return f"{orv:.2f} ({lo:.2f}{DASH}{hi:.2f})"

def fmt_p(p: float) -> str:
    if p is None or not np.isfinite(p):
        return "NA"
    if p < 0.0001:
        return "<0.0001"
    return f"{p:.4f}"

def map_symptom_to_groups(symptom: str) -> List[str]:
    """
    Returns ALL matching groups (multi-label).
    A symptom can belong to multiple groups.
    Uses ALL_GROUPS = default + PI groups.
    """
    s = canonical(symptom)
    hits: List[str] = []
    for grp, keys in ALL_GROUPS.items():
        for k in keys:
            if canonical(k) in s:
                hits.append(grp)
                break
    return hits


# =========================
# AE: Extract patient-level wide
# =========================
def extract_ae_patientwide(df_ae: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int], List[str]]:
    """
    Expected (your file): Patient ID | Poison Type | <many symptom columns with Yes/No>
    Returns:
      tmp: includes Group + all symptom columns
      denoms: group Ns (unique patients)
      symptom_cols: list of symptom columns
    """
    cols = list(df_ae.columns)
    if len(cols) < 3:
        raise ValueError("AE sheet looks too small. Expected Patient ID, Poison Type, and symptom columns.")

    pid_col = cols[0]
    poison_col = cols[1]
    symptom_cols = cols[2:]

    tmp = df_ae.copy()
    tmp["Group"] = assign_group_from_poison(tmp[poison_col])
    tmp = tmp.dropna(subset=["Group", pid_col])

    denoms = tmp.groupby("Group")[pid_col].nunique().to_dict()
    denoms = {k: int(v) for k, v in denoms.items()}

    # Ensure defaults exist
    for g, n in DEFAULT_DENOMS.items():
        denoms.setdefault(g, n)

    return tmp, denoms, symptom_cols


def build_ae_all_symptoms(tmp: pd.DataFrame, denoms: Dict[str, int], symptom_cols: List[str]) -> pd.DataFrame:
    rows = []

    for sym in symptom_cols:
        # total counts per group (unique patients)
        for g in ["Mixed OP", "Chlorpyrifos", "Cypermethrin"]:
            if g not in tmp["Group"].unique():
                continue

        xM = int(to_bool_series(tmp.loc[tmp["Group"] == "Mixed OP", sym]).sum())
        xC = int(to_bool_series(tmp.loc[tmp["Group"] == "Chlorpyrifos", sym]).sum())
        xY = int(to_bool_series(tmp.loc[tmp["Group"] == "Cypermethrin", sym]).sum())

        if DROP_ALL_ZERO_SYMPTOMS and (xM + xC + xY == 0):
            continue

        nM = int(denoms["Mixed OP"])
        nC = int(denoms["Chlorpyrifos"])
        nY = int(denoms["Cypermethrin"])

        # Mixed vs Chlor
        rdMC, loMC, hiMC = rd_newcombe_wilson(xM, nM, xC, nC)
        orMC, orloMC, orhiMC = or_ci_haldane(xM, nM, xC, nC)
        pF_MC, pChiNC_MC, pChiY_MC = pvals_2x2(xM, nM, xC, nC)

        # Mixed vs Cyp
        rdMY, loMY, hiMY = rd_newcombe_wilson(xM, nM, xY, nY)
        orMY, orloMY, orhiMY = or_ci_haldane(xM, nM, xY, nY)
        pF_MY, pChiNC_MY, pChiY_MY = pvals_2x2(xM, nM, xY, nY)

        rows.append({
            "Symptom": sym,
            "Mixed OP n (%)": fmt_count_pct(xM, nM),
            "Chlorpyrifos n (%)": fmt_count_pct(xC, nC),
            "Cypermethrin n (%)": fmt_count_pct(xY, nY),

            "Risk diff (Mixed−Chlor) % (95% CI)": fmt_rd(rdMC, loMC, hiMC),
            "OR (Mixed vs Chlor) (95% CI)": fmt_or(orMC, orloMC, orhiMC),
            "p (Fisher) MC": fmt_p(pF_MC),
            "p (Chi2 no cont) MC": fmt_p(pChiNC_MC),
            "p (Chi2 Yates) MC": fmt_p(pChiY_MC),

            "Risk diff (Mixed−Cyp) % (95% CI)": fmt_rd(rdMY, loMY, hiMY),
            "OR (Mixed vs Cyp) (95% CI)": fmt_or(orMY, orloMY, orhiMY),
            "p (Fisher) MY": fmt_p(pF_MY),
            "p (Chi2 no cont) MY": fmt_p(pChiNC_MY),
            "p (Chi2 Yates) MY": fmt_p(pChiY_MY),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Sort: mapped group first, then symptom
    out["AE Group (mapped)"] = out["Symptom"].apply(lambda s: (map_symptom_to_groups(s)[0] if map_symptom_to_groups(s) else None))
    out = out.sort_values(["AE Group (mapped)", "Symptom"], na_position="last").reset_index(drop=True)
    return out


def build_symptom_mapping(symptom_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    mapping_rows = []
    unmapped_rows = []

    for sym in symptom_cols:
        grps = map_symptom_to_groups(sym)
        if grps:
            mapping_rows.append({
                "Symptom": sym,
                "AE Groups (multi)": "; ".join(sorted(set(grps))),
                "Primary (first)": grps[0],
                "n_groups": len(set(grps)),
            })
        else:
            mapping_rows.append({
                "Symptom": sym,
                "AE Groups (multi)": "UNMAPPED",
                "Primary (first)": "UNMAPPED",
                "n_groups": 0,
            })
            unmapped_rows.append({"Symptom": sym})

    map_df = pd.DataFrame(mapping_rows).sort_values(["AE Groups (multi)", "Symptom"]).reset_index(drop=True)
    un_df = pd.DataFrame(unmapped_rows).sort_values("Symptom").reset_index(drop=True)
    return map_df, un_df



def build_ae_grouped(tmp: pd.DataFrame, denoms: Dict[str, int], symptom_cols: List[str]) -> pd.DataFrame:
    """
    Patient-level grouped analysis using ANY-in-group per patient (no double counting).
    Only uses symptoms that map to a group.
    """
    sym_to_grps = {sym: map_symptom_to_groups(sym) for sym in symptom_cols}
    grouped_cols: Dict[str, List[str]] = {}
    for sym, grps in sym_to_grps.items():
        for grp in grps:
            grouped_cols.setdefault(grp, []).append(sym)

    rows = []
    for grp_name, cols in grouped_cols.items():
        # Any symptom within the group
        any_event = tmp[cols].apply(lambda c: to_bool_series(c)).any(axis=1)

        # Count by group
        xM = int(any_event[tmp["Group"] == "Mixed OP"].sum())
        xC = int(any_event[tmp["Group"] == "Chlorpyrifos"].sum())
        xY = int(any_event[tmp["Group"] == "Cypermethrin"].sum())

        nM = int(denoms["Mixed OP"])
        nC = int(denoms["Chlorpyrifos"])
        nY = int(denoms["Cypermethrin"])

        rdMC, loMC, hiMC = rd_newcombe_wilson(xM, nM, xC, nC)
        orMC, orloMC, orhiMC = or_ci_haldane(xM, nM, xC, nC)
        pF_MC, pChiNC_MC, pChiY_MC = pvals_2x2(xM, nM, xC, nC)

        rdMY, loMY, hiMY = rd_newcombe_wilson(xM, nM, xY, nY)
        orMY, orloMY, orhiMY = or_ci_haldane(xM, nM, xY, nY)
        pF_MY, pChiNC_MY, pChiY_MY = pvals_2x2(xM, nM, xY, nY)

        rows.append({
            "AE Group": grp_name,
            "Symptoms included": "; ".join(cols),
            "Mixed OP n (%)": fmt_count_pct(xM, nM),
            "Chlorpyrifos n (%)": fmt_count_pct(xC, nC),
            "Cypermethrin n (%)": fmt_count_pct(xY, nY),

            "Risk diff (Mixed−Chlor) % (95% CI)": fmt_rd(rdMC, loMC, hiMC),
            "OR (Mixed vs Chlor) (95% CI)": fmt_or(orMC, orloMC, orhiMC),
            "p (Fisher) MC": fmt_p(pF_MC),
            "p (Chi2 no cont) MC": fmt_p(pChiNC_MC),
            "p (Chi2 Yates) MC": fmt_p(pChiY_MC),

            "Risk diff (Mixed−Cyp) % (95% CI)": fmt_rd(rdMY, loMY, hiMY),
            "OR (Mixed vs Cyp) (95% CI)": fmt_or(orMY, orloMY, orhiMY),
            "p (Fisher) MY": fmt_p(pF_MY),
            "p (Chi2 no cont) MY": fmt_p(pChiNC_MY),
            "p (Chi2 Yates) MY": fmt_p(pChiY_MY),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("AE Group").reset_index(drop=True)

def bh_fdr(pvals: List[float]) -> List[float]:
    """Benjamini–Hochberg FDR. Keeps NA as NA."""
    arr = np.array([p if (p is not None and np.isfinite(p)) else np.nan for p in pvals], dtype=float)
    out = np.full_like(arr, np.nan, dtype=float)
    ok = np.isfinite(arr)
    if ok.sum() == 0:
        return out.tolist()

    pv = arr[ok]
    m = pv.size
    order = np.argsort(pv)
    ranked = pv[order]
    q = ranked * m / (np.arange(1, m + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)

    tmp = np.empty_like(pv)
    tmp[order] = q
    out[ok] = tmp
    return out.tolist()


def _group_signature(cols: List[str]) -> Tuple[str, ...]:
    return tuple(sorted(set(cols)))


def build_ae_grouped_exploratory(tmp: pd.DataFrame, denoms: Dict[str, int], symptom_cols: List[str]) -> pd.DataFrame:
    """
    Creates MANY overlapping candidate groups:
      - Base groups (multi-label membership)
      - Keyword groups (each keyword becomes its own group)
      - Token groups (tokens from symptom colnames)
      - Unions and intersections of base groups

    Then performs ANY-in-group per patient and computes p-values.
    Includes BH-FDR adjusted p for Fisher (MC and MY) across all groups.
    """

    # ----- 1) build base multi-label mapping: group -> cols -----
    base_grouped: Dict[str, List[str]] = {}
    for sym in symptom_cols:
        hits = map_symptom_to_groups(sym)
        for g in hits:
            base_grouped.setdefault(g, []).append(sym)

    # keep only groups with >=1 col
    base_grouped = {g: sorted(set(cols)) for g, cols in base_grouped.items() if cols}

    variants: List[Tuple[str, str, List[str]]] = []  # (group_type, group_name, cols)

    if INCLUDE_BASE_GROUPS:
        for g, cols in base_grouped.items():
            origin = "PI" if g in PI_GROUPS else "DEFAULT"
            variants.append((f"BASE_{origin}", g, cols))

    # ----- 2) keyword groups -----
    if INCLUDE_KEYWORD_GROUPS:
        for parent_g, keys in ALL_GROUPS.items():
            origin = "PI" if parent_g in PI_GROUPS else "DEFAULT"
            for k in keys:
                cols = [sym for sym in symptom_cols if canonical(k) in canonical(sym)]
                cols = sorted(set(cols))
                if len(cols) >= MIN_COLS_PER_AUTO_GROUP:
                    variants.append((f"KEYWORD_{origin}", f"Keyword: {k}  (from {parent_g})", cols))


    # ----- 3) token groups -----
    if INCLUDE_TOKEN_GROUPS:
        tokens = set()
        for sym in symptom_cols:
            toks = re.split(r"[ _/,\-]+", canonical(sym))
            for t in toks:
                if len(t) >= MIN_TOKEN_LEN and t not in TOKEN_STOPWORDS and not t.isdigit():
                    tokens.add(t)

        for t in sorted(tokens):
            cols = [sym for sym in symptom_cols if t in canonical(sym)]
            cols = sorted(set(cols))
            if len(cols) >= MIN_COLS_PER_AUTO_GROUP:
                variants.append(("TOKEN", f"Token: {t}", cols))

    # ----- 4) unions / intersections of BASE groups -----
    base_names = sorted(base_grouped.keys())

    if INCLUDE_UNION_GROUPS and len(base_names) >= 2:
        for r in range(2, max(2, MAX_UNION_SIZE) + 1):
            for comb in itertools.combinations(base_names, r):
                cols = sorted(set().union(*[base_grouped[c] for c in comb]))
                if len(cols) >= MIN_COLS_PER_AUTO_GROUP:
                    label = []
                    for name in comb:
                        label.append(("PI:" if name in PI_GROUPS else "D:") + name)
                    variants.append(("UNION", f"Union({r}): " + " + ".join(label), cols))


    if INCLUDE_INTERSECTION_GROUPS and len(base_names) >= 2:
        for a, b in itertools.combinations(base_names, 2):
            cols = sorted(set(base_grouped[a]).intersection(base_grouped[b]))
            if len(cols) >= MIN_COLS_PER_AUTO_GROUP:
                variants.append(("INTERSECTION", f"Intersection: {a} ∩ {b}", cols))

    # ----- 5) deduplicate groups by exact symptom-set signature -----
    seen = {}
    deduped: List[Tuple[str, str, List[str]]] = []
    for gtype, gname, cols in variants:
        cols = sorted(set(cols))
        if not cols:
            continue
        if len(cols) > MAX_COLS_PER_GROUP:
            continue
        sig = _group_signature(cols)
        if sig in seen:
            continue
        seen[sig] = (gtype, gname)
        deduped.append((gtype, gname, cols))

    # ----- 6) compute group stats -----
    rows = []
    nM = int(denoms["Mixed OP"])
    nC = int(denoms["Chlorpyrifos"])
    nY = int(denoms["Cypermethrin"])

    for gtype, gname, cols in deduped:
        any_event = tmp[cols].apply(lambda c: to_bool_series(c)).any(axis=1)

        xM = int(any_event[tmp["Group"] == "Mixed OP"].sum())
        xC = int(any_event[tmp["Group"] == "Chlorpyrifos"].sum())
        xY = int(any_event[tmp["Group"] == "Cypermethrin"].sum())

        if (xM + xC + xY) < MIN_TOTAL_EVENTS:
            continue

        rdMC, loMC, hiMC = rd_newcombe_wilson(xM, nM, xC, nC)
        orMC, orloMC, orhiMC = or_ci_haldane(xM, nM, xC, nC)
        pF_MC, pChiNC_MC, pChiY_MC = pvals_2x2(xM, nM, xC, nC)

        rdMY, loMY, hiMY = rd_newcombe_wilson(xM, nM, xY, nY)
        orMY, orloMY, orhiMY = or_ci_haldane(xM, nM, xY, nY)
        pF_MY, pChiNC_MY, pChiY_MY = pvals_2x2(xM, nM, xY, nY)

        rows.append({
            "Group type": gtype,
            "AE Group (exploratory)": gname,
            "n_symptoms": len(cols),
            "Symptoms included": "; ".join(cols),

            "Mixed OP n (%)": fmt_count_pct(xM, nM),
            "Chlorpyrifos n (%)": fmt_count_pct(xC, nC),
            "Cypermethrin n (%)": fmt_count_pct(xY, nY),

            "RD MC % (95% CI)": fmt_rd(rdMC, loMC, hiMC),
            "OR MC (95% CI)": fmt_or(orMC, orloMC, orhiMC),
            "pF_MC": pF_MC,
            "pChiNC_MC": pChiNC_MC,
            "pChiY_MC": pChiY_MC,

            "RD MY % (95% CI)": fmt_rd(rdMY, loMY, hiMY),
            "OR MY (95% CI)": fmt_or(orMY, orloMY, orhiMY),
            "pF_MY": pF_MY,
            "pChiNC_MY": pChiNC_MY,
            "pChiY_MY": pChiY_MY,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # ----- 7) add FDR-adjusted Fisher p-values across ALL exploratory groups -----
    out["pF_MC_FDR"] = bh_fdr(out["pF_MC"].tolist())
    out["pF_MY_FDR"] = bh_fdr(out["pF_MY"].tolist())

    # user-friendly formatted p columns (keep raw numeric columns too)
    out["p (Fisher) MC"] = out["pF_MC"].apply(fmt_p)
    out["p (Fisher) MC FDR"] = out["pF_MC_FDR"].apply(fmt_p)
    out["p (Fisher) MY"] = out["pF_MY"].apply(fmt_p)
    out["p (Fisher) MY FDR"] = out["pF_MY_FDR"].apply(fmt_p)

    # sort by best (smallest) Fisher p (uncorrected), then by size (smaller groups first)
    out["best_p_uncorrected"] = np.nanmin(np.vstack([out["pF_MC"].values, out["pF_MY"].values]), axis=0)
    out = out.sort_values(["best_p_uncorrected", "n_symptoms", "AE Group (exploratory)"]).reset_index(drop=True)

    # optional export cap
    if MAX_GROUPS_EXPORT is not None and len(out) > MAX_GROUPS_EXPORT:
        out = out.iloc[:MAX_GROUPS_EXPORT].copy()

    return out


# =========================
# MORTALITY: Parse "Deceased" mixed-layout sheet properly
# =========================
def parse_deceased_sheet_patientlevel(xlsx_path: Path) -> pd.DataFrame:
    """
    Your Deceased sheet is a mixed layout (not a clean table).
    We read header=None and find the row that contains 'Patient ID' and 'Poison Type' and 'Status'.
    """
    raw = pd.read_excel(xlsx_path, sheet_name=SHEET_DECEASED, header=None)

    # Find header row
    header_idx = None
    for i in range(min(30, raw.shape[0])):
        row = raw.iloc[i].astype(str).str.strip().str.lower().tolist()
        if ("patient id" in row) and ("poison type" in row) and ("status" in row):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("Could not detect the patient-level header row in Deceased sheet.")

    # Identify columns for the patient-level table
    header_row = raw.iloc[header_idx].astype(str).str.strip()
    def find_col_idx(name: str) -> Optional[int]:
        name_l = name.strip().lower()
        for j, v in enumerate(header_row.tolist()):
            if str(v).strip().lower() == name_l:
                return j
        return None

    c_pid = find_col_idx("Patient ID")
    c_poison = find_col_idx("Poison Type")
    c_status = find_col_idx("Status")

    if c_pid is None or c_poison is None or c_status is None:
        raise ValueError("Detected header row but could not locate Patient ID / Poison Type / Status columns reliably.")

    # Build table from rows after header
    df = raw.iloc[header_idx + 1 :, [c_pid, c_poison, c_status]].copy()
    df.columns = ["Patient ID", "Poison Type", "Status"]

    # Drop empty rows
    df = df.dropna(subset=["Patient ID", "Poison Type", "Status"], how="any").copy()
    return df


def build_mortality(xlsx_path: Path) -> pd.DataFrame:
    df = parse_deceased_sheet_patientlevel(xlsx_path)

    df["Group"] = assign_group_from_poison(df["Poison Type"])
    df = df.dropna(subset=["Group"]).copy()

    died = df["Status"].astype(str).str.strip().str.lower().isin(["deceased", "dead", "died", "death"])

    denoms = df.groupby("Group")["Patient ID"].nunique().to_dict()
    denoms = {k: int(v) for k, v in denoms.items()}
    for g, n in DEFAULT_DENOMS.items():
        denoms.setdefault(g, n)

    deaths = df[died].groupby("Group")["Patient ID"].nunique().to_dict()
    deaths = {k: int(v) for k, v in deaths.items()}

    nM = int(denoms["Mixed OP"]); nC = int(denoms["Chlorpyrifos"]); nY = int(denoms["Cypermethrin"])
    xM = int(deaths.get("Mixed OP", 0)); xC = int(deaths.get("Chlorpyrifos", 0)); xY = int(deaths.get("Cypermethrin", 0))

    # Comparisons
    orMC, loMC, hiMC = or_ci_haldane(xM, nM, xC, nC)
    pF_MC, pChiNC_MC, pChiY_MC = pvals_2x2(xM, nM, xC, nC)

    orMY, loMY, hiMY = or_ci_haldane(xM, nM, xY, nY)
    pF_MY, pChiNC_MY, pChiY_MY = pvals_2x2(xM, nM, xY, nY)

    lM, uM = wilson_ci(xM, nM)
    lC, uC = wilson_ci(xC, nC)
    lY, uY = wilson_ci(xY, nY)

    return pd.DataFrame([{
        "Factors": "Mortality",
        "Overall (n)": int(nM + nC + nY),
        "Mixed OP": f"{xM} ({(xM/nM)*100:.2f}%)",
        "Chlorpyrifos": f"{xC} ({(xC/nC)*100:.2f}%)",
        "Cypermethrin": f"{xY} ({(xY/nY)*100:.2f}%)",

        "OR (Mixed vs Chlor) (95% CI)": fmt_or(orMC, loMC, hiMC),
        "95% CI (Mortality rate) Mixed vs Chlor": f"({lM*100:.2f}{DASH}{uM*100:.2f}%) vs ({lC*100:.2f}{DASH}{uC*100:.2f}%)",
        "p (Fisher) MC": fmt_p(pF_MC),
        "p (Chi2 no cont) MC": fmt_p(pChiNC_MC),
        "p (Chi2 Yates) MC": fmt_p(pChiY_MC),

        "OR (Mixed vs Cyp) (95% CI)": fmt_or(orMY, loMY, hiMY),
        "95% CI (Mortality rate) Mixed vs Cyp": f"({lM*100:.2f}{DASH}{uM*100:.2f}%) vs ({lY*100:.2f}{DASH}{uY*100:.2f}%)",
        "p (Fisher) MY": fmt_p(pF_MY),
        "p (Chi2 no cont) MY": fmt_p(pChiNC_MY),
        "p (Chi2 Yates) MY": fmt_p(pChiY_MY),
    }])


# =========================
# MAIN
# =========================
def main():
    base_dir = Path(__file__).resolve().parent
    xlsx_path = base_dir / INPUT_XLSX
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Missing: {xlsx_path}")

    # ---- create output folder ----
    out_dir = base_dir / OUTPUT_ROOT
    if OUTPUT_RUN_SUBFOLDER:
        out_dir = out_dir / xlsx_path.stem  # e.g., outputs/table_2
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- AE ---
    df_ae = pd.read_excel(xlsx_path, sheet_name=SHEET_AE)
    tmp, denoms, symptom_cols = extract_ae_patientwide(df_ae)

    ae_all = build_ae_all_symptoms(tmp, denoms, symptom_cols)
    ae_grouped = build_ae_grouped(tmp, denoms, symptom_cols)
    ae_map, ae_unmapped = build_symptom_mapping(symptom_cols)
    ae_grouped_explore = build_ae_grouped_exploratory(tmp, denoms, symptom_cols)
    
    # --- Mortality ---
    mortality = build_mortality(xlsx_path)

    # ---- outputs (names include input stem for clarity) ----
    out_all_xlsx = out_dir / f"{xlsx_path.stem}_Table2_AllOutputs.xlsx"
    out_ae_all_csv = out_dir / f"{xlsx_path.stem}_Table2_AE_AllSymptoms.csv"
    out_ae_grouped_csv = out_dir / f"{xlsx_path.stem}_Table2_AE_Grouped.csv"
    out_ae_map_csv = out_dir / f"{xlsx_path.stem}_Table2_AE_SymptomMapping.csv"
    out_ae_unmapped_csv = out_dir / f"{xlsx_path.stem}_Table2_AE_Unmapped.csv"
    out_mortality_csv = out_dir / f"{xlsx_path.stem}_Table2_Mortality.csv"
    out_ae_grouped_explore_csv = out_dir / f"{xlsx_path.stem}_Table2_AE_Grouped_Exploratory.csv"
    
    # --- Save combined workbook ---
    with pd.ExcelWriter(out_all_xlsx, engine="openpyxl") as w:
        ae_all.to_excel(w, sheet_name="AE_AllSymptoms", index=False)
        ae_grouped.to_excel(w, sheet_name="AE_Grouped", index=False)
        ae_map.to_excel(w, sheet_name="AE_SymptomMapping", index=False)
        ae_unmapped.to_excel(w, sheet_name="AE_Unmapped", index=False)
        mortality.to_excel(w, sheet_name="Mortality", index=False)
        ae_grouped_explore.to_excel(w, sheet_name="AE_Grouped_Exploratory", index=False)


        meta = pd.DataFrame([{
            "Input file": str(xlsx_path),
            "Output folder": str(out_dir),
            "Denoms_MixedOP": denoms["Mixed OP"],
            "Denoms_Chlorpyrifos": denoms["Chlorpyrifos"],
            "Denoms_Cypermethrin": denoms["Cypermethrin"],
            "Total_AE_Columns": len(symptom_cols),
            "Mapped_AE_Columns": int((ae_map["AE Groups (multi)"] != "UNMAPPED").sum()),
            "Unmapped_AE_Columns": int((ae_map["AE Groups (multi)"] == "UNMAPPED").sum()),

            "DROP_ALL_ZERO_SYMPTOMS": DROP_ALL_ZERO_SYMPTOMS,
        }])
        meta.to_excel(w, sheet_name="Metadata", index=False)

    # --- CSVs (Excel-friendly) ---
    ae_all.to_csv(out_ae_all_csv, index=False, encoding="utf-8-sig")
    ae_grouped.to_csv(out_ae_grouped_csv, index=False, encoding="utf-8-sig")
    ae_map.to_csv(out_ae_map_csv, index=False, encoding="utf-8-sig")
    ae_unmapped.to_csv(out_ae_unmapped_csv, index=False, encoding="utf-8-sig")
    mortality.to_csv(out_mortality_csv, index=False, encoding="utf-8-sig")
    ae_grouped_explore.to_csv(out_ae_grouped_explore_csv, index=False, encoding="utf-8-sig")
    
    print("Done.")
    print("Input:", xlsx_path)
    print("Outputs saved to:", out_dir)
    print("Denominators used:", denoms)
    print("Workbook:", out_all_xlsx)


if __name__ == "__main__":
    main()
