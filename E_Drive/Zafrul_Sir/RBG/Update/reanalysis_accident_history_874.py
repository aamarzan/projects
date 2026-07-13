#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Complete-case accident-history re-analysis among 874 drivers.

This script:
1. Keeps only participants with accident-history data.
2. Screens all available variables for association with accident history.
3. Groups variables into easy reviewer-friendly domains.
4. Tests smoking + betel quid + RBG pathway.
5. Tests license type + betel quid + accident-history pathway.
6. Exports Excel tables and a simple-language summary.
"""

from pathlib import Path
import re
import math
import warnings

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------------------

PROJECT_DIR = Path(r"E:\Zafrul_Sir\RBG\Update")
DATA_SAV = PROJECT_DIR / "Wobaidul_zafrul_RBG STUDY.sav"
OUT_DIR = PROJECT_DIR / "Reanalysis_874"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# 2. Helper functions
# ---------------------------------------------------------------------

def norm_name(x):
    return re.sub(r"[^a-z0-9]", "", str(x).lower())

def clean_label(x):
    if pd.isna(x):
        return np.nan
    return re.sub(r"\s+", " ", str(x).strip())

def find_col(df, candidates, required=True):
    norm_map = {norm_name(c): c for c in df.columns}

    for cand in candidates:
        if cand in df.columns:
            return cand
        key = norm_name(cand)
        if key in norm_map:
            return norm_map[key]

    for cand in candidates:
        key = norm_name(cand)
        hits = [orig for nk, orig in norm_map.items() if key in nk or nk in key]
        if hits:
            return hits[0]

    if required:
        raise KeyError(f"Column not found. Tried: {candidates}")
    return None

def infer_type(s):
    numeric = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() > 0 and numeric.notna().sum() / s.notna().sum() >= 0.85 and s.nunique() > 8:
        return "numeric"
    return "categorical"

def classify_group(var):
    n = norm_name(var)

    if any(k in n for k in ["age", "sex", "marital", "education", "region", "location"]):
        return "Demographic factors"

    if any(k in n for k in ["license", "licence", "vehicle", "driving", "drive"]):
        return "Occupational factors"

    if any(k in n for k in ["smok", "betel", "food", "diet", "sleep", "screen", "physical", "device", "attention", "temper"]):
        return "Lifestyle and behavioral factors"

    if any(k in n for k in ["rbg", "glucose", "diabet", "glycemic", "bmi", "bp", "pulse", "comorbid", "medication"]):
        return "Metabolic and clinical factors"

    if any(k in n for k in ["urine", "urin", "protein", "leuk", "ketone", "bilirubin", "ph", "specific", "uti", "kidney", "burn", "pain", "urgency"]):
        return "Urinary and renal factors"

    return "Other factors"

def p_text(p):
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"

def odds_ratio_2x2(a, b, c, d):
    """
    a = exposed accident
    b = exposed no accident
    c = unexposed accident
    d = unexposed no accident
    """
    p = stats.fisher_exact([[a, b], [c, d]])[1]

    a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    OR = (a * d) / (b * c)
    SE = math.sqrt(1/a + 1/b + 1/c + 1/d)
    low = math.exp(math.log(OR) - 1.96 * SE)
    high = math.exp(math.log(OR) + 1.96 * SE)

    return OR, low, high, p

def yes_no_binary(s):
    x = s.map(clean_label).astype("string").str.lower()

    return np.where(
        x.str.contains("yes|positive|present|smok", regex=True, na=False), 1,
        np.where(
            x.str.contains("no|negative|none|absent|never", regex=True, na=False), 0,
            np.nan
        )
    )

# ---------------------------------------------------------------------
# 3. Load SPSS data
# ---------------------------------------------------------------------

try:
    import pyreadstat
except ImportError:
    raise ImportError("Please install pyreadstat: python -m pip install pyreadstat")

raw, meta = pyreadstat.read_sav(str(DATA_SAV), apply_value_formats=True)

# ---------------------------------------------------------------------
# 4. Detect key columns
# ---------------------------------------------------------------------

accident_col = find_col(raw, ["Accident_History", "Accident History", "AccidentHistory"])
rbg_col = find_col(raw, ["RBG", "Random blood glucose", "Random_Blood_Glucose"], required=False)
license_col = find_col(raw, ["License", "License_Type", "Licence"], required=False)
smoking_col = find_col(raw, ["Smoking", "Smoking_Status", "Smoke"], required=False)
betel_col = find_col(raw, ["B_Quid", "Betel_Quid", "Betel quid intake"], required=False)
age_col = find_col(raw, ["Age"], required=False)
driving_hours_col = find_col(raw, ["Driving_H_D", "Driving hours/day", "Driving_Hours"], required=False)

detected_columns = {
    "accident_col": accident_col,
    "rbg_col": rbg_col,
    "license_col": license_col,
    "smoking_col": smoking_col,
    "betel_col": betel_col,
    "age_col": age_col,
    "driving_hours_col": driving_hours_col,
}

# ---------------------------------------------------------------------
# 5. Keep only accident-history complete cases
# ---------------------------------------------------------------------

df = raw[raw[accident_col].notna()].copy()
df["accident_label"] = df[accident_col].map(clean_label)
df["accident"] = df["accident_label"].str.lower().eq("yes").astype(int)

if rbg_col:
    df["RBG_num"] = pd.to_numeric(df[rbg_col], errors="coerce")

if age_col:
    df["Age_num"] = pd.to_numeric(df[age_col], errors="coerce")

if driving_hours_col:
    df["Driving_hours_num"] = pd.to_numeric(df[driving_hours_col], errors="coerce")

if license_col:
    df["License_clean"] = df[license_col].map(clean_label)

if smoking_col:
    df["Smoking_binary"] = yes_no_binary(df[smoking_col])

if betel_col:
    df["Betel_binary"] = yes_no_binary(df[betel_col])

print("Complete-case N:", len(df))
print("Accident events:", int(df["accident"].sum()))
print("Accident prevalence:", round(df["accident"].mean() * 100, 1), "%")

df.to_csv(OUT_DIR / "complete_case_874_dataset.csv", index=False)

# ---------------------------------------------------------------------
# 6. Screen all variables one by one
# ---------------------------------------------------------------------

screen_rows = []
category_rows = []

exclude_cols = {
    accident_col,
    "accident",
    "accident_label",
    "RBG_num",
    "Age_num",
    "Driving_hours_num",
    "License_clean",
    "Smoking_binary",
    "Betel_binary",
}

for var in df.columns:
    if var in exclude_cols:
        continue

    if df[var].notna().sum() < 40:
        continue

    vtype = infer_type(df[var])

    try:
        if vtype == "numeric":
            x0 = pd.to_numeric(df.loc[df["accident"] == 0, var], errors="coerce").dropna()
            x1 = pd.to_numeric(df.loc[df["accident"] == 1, var], errors="coerce").dropna()

            if len(x0) < 5 or len(x1) < 5:
                continue

            p = stats.mannwhitneyu(x0, x1, alternative="two-sided").pvalue

            screen_rows.append({
                "variable": var,
                "group": classify_group(var),
                "type": "numeric",
                "n_no_accident": len(x0),
                "n_accident": len(x1),
                "no_accident_median_IQR": f"{x0.median():.2f} [{x0.quantile(.25):.2f}, {x0.quantile(.75):.2f}]",
                "accident_median_IQR": f"{x1.median():.2f} [{x1.quantile(.25):.2f}, {x1.quantile(.75):.2f}]",
                "test": "Mann–Whitney U test",
                "p_value": p,
                "simple_interpretation": "Values differ between accident and no-accident groups."
            })

        else:
            temp = pd.DataFrame({
                "accident": df["accident"],
                "x": df[var].map(clean_label)
            }).dropna()

            if temp["x"].nunique() < 2:
                continue

            counts = temp["x"].value_counts()
            rare = counts[counts < 10].index
            temp["x2"] = np.where(temp["x"].isin(rare), "Other/rare", temp["x"])

            if temp["x2"].nunique() > 12:
                top = temp["x2"].value_counts().head(11).index
                temp["x2"] = np.where(temp["x2"].isin(top), temp["x2"], "Other/rare")

            table = pd.crosstab(temp["x2"], temp["accident"])

            if table.shape[0] < 2 or table.shape[1] < 2:
                continue

            chi2, p, dof, exp = stats.chi2_contingency(table)

            screen_rows.append({
                "variable": var,
                "group": classify_group(var),
                "type": "categorical",
                "n_total": len(temp),
                "test": "Chi-square test",
                "p_value": p,
                "simple_interpretation": "Accident-history percentage differs across categories."
            })

            for cat in sorted(temp["x2"].unique()):
                exposed = temp["x2"].eq(cat)

                a = int(((exposed) & (temp["accident"] == 1)).sum())
                b = int(((exposed) & (temp["accident"] == 0)).sum())
                c = int(((~exposed) & (temp["accident"] == 1)).sum())
                d = int(((~exposed) & (temp["accident"] == 0)).sum())

                OR, low, high, p_cat = odds_ratio_2x2(a, b, c, d)

                category_rows.append({
                    "variable": var,
                    "category": cat,
                    "n_category": int(exposed.sum()),
                    "accident_n": a,
                    "no_accident_n": b,
                    "accident_percent": round(100 * a / max(a + b, 1), 2),
                    "OR_category_vs_others": OR,
                    "CI_low": low,
                    "CI_high": high,
                    "p_value": p_cat,
                })

    except Exception as e:
        screen_rows.append({
            "variable": var,
            "group": classify_group(var),
            "type": vtype,
            "p_value": np.nan,
            "error": str(e)
        })

screening = pd.DataFrame(screen_rows)
category_details = pd.DataFrame(category_rows)

if not screening.empty:
    valid = screening["p_value"].notna()
    screening["FDR_q_value"] = np.nan
    screening.loc[valid, "FDR_q_value"] = multipletests(
        screening.loc[valid, "p_value"],
        method="fdr_bh"
    )[1]

    screening["priority"] = np.where(
        screening["FDR_q_value"] <= 0.10,
        "Strong signal after FDR",
        np.where(screening["p_value"] < 0.05, "Nominal p<0.05", "Not significant")
    )

    screening = screening.sort_values(["FDR_q_value", "p_value"], na_position="last")

if not category_details.empty:
    category_details = category_details.sort_values("p_value")

# ---------------------------------------------------------------------
# 7. Dr. Zafrul hypothesis 1:
#    Smoking + betel quid + RBG + accident history
# ---------------------------------------------------------------------

hypothesis_tables = {}

if smoking_col and betel_col:
    df["smoking_betel_group"] = pd.Series(pd.NA, index=df.index, dtype="object")

    df.loc[
        (df["Smoking_binary"] == 0) & (df["Betel_binary"] == 0),
        "smoking_betel_group"
    ] = "Neither smoking nor betel"

    df.loc[
        (df["Smoking_binary"] == 1) & (df["Betel_binary"] == 0),
        "smoking_betel_group"
    ] = "Smoking only"

    df.loc[
        (df["Smoking_binary"] == 0) & (df["Betel_binary"] == 1),
        "smoking_betel_group"
    ] = "Betel only"

    df.loc[
        (df["Smoking_binary"] == 1) & (df["Betel_binary"] == 1),
        "smoking_betel_group"
    ] = "Both smoking and betel"

    temp = df[["accident", "smoking_betel_group", "RBG_num"]].dropna()

    group_summary = temp.groupby("smoking_betel_group").agg(
        n=("accident", "count"),
        accident_n=("accident", "sum"),
        accident_percent=("accident", lambda x: 100 * x.mean()),
        RBG_median=("RBG_num", "median"),
        RBG_mean=("RBG_num", "mean"),
        RBG_sd=("RBG_num", "std")
    ).reset_index()

    rbg_groups = [
        g["RBG_num"].values
        for _, g in temp.groupby("smoking_betel_group")
    ]

    if len(rbg_groups) >= 2:
        group_summary["RBG_group_difference_p"] = stats.kruskal(*rbg_groups).pvalue

    hypothesis_tables["Smoking_betel_group_summary"] = group_summary

    # Model A: smoking/betel group only
    model_a_data = temp.dropna()
    model_a = smf.logit(
        "accident ~ C(smoking_betel_group)",
        data=model_a_data
    ).fit(disp=False)

    # Model B: smoking/betel group adjusted for RBG
    model_b = smf.logit(
        "accident ~ C(smoking_betel_group) + RBG_num",
        data=model_a_data
    ).fit(disp=False)

    hypothesis_tables["Smoking_betel_accident_model_only"] = pd.DataFrame({
        "term": model_a.params.index,
        "OR": np.exp(model_a.params.values),
        "p_value": model_a.pvalues.values
    })

    hypothesis_tables["Smoking_betel_accident_model_adjusted_RBG"] = pd.DataFrame({
        "term": model_b.params.index,
        "OR": np.exp(model_b.params.values),
        "p_value": model_b.pvalues.values
    })

# ---------------------------------------------------------------------
# 8. Dr. Zafrul hypothesis 2:
#    License type -> betel quid -> accident history
# ---------------------------------------------------------------------

if license_col and betel_col:
    lic = df["License_clean"].astype("string").str.lower()

    df["license_renew"] = np.where(
        lic.str.contains("renew", na=False), 1,
        np.where(lic.str.contains("new", na=False), 0, np.nan)
    )

    temp = df[["accident", "license_renew", "Betel_binary"]].dropna()

    license_summary = temp.groupby("license_renew").agg(
        n=("accident", "count"),
        accident_n=("accident", "sum"),
        accident_percent=("accident", lambda x: 100 * x.mean()),
        betel_n=("Betel_binary", "sum"),
        betel_percent=("Betel_binary", lambda x: 100 * x.mean())
    ).reset_index()

    license_summary["license_group"] = license_summary["license_renew"].map({
        0: "New license",
        1: "Renew license"
    })

    hypothesis_tables["License_betel_summary"] = license_summary

    # License predicts betel
    betel_model = smf.logit(
        "Betel_binary ~ license_renew",
        data=temp
    ).fit(disp=False)

    # License predicts accident
    license_model = smf.logit(
        "accident ~ license_renew",
        data=temp
    ).fit(disp=False)

    # License + betel predict accident
    license_betel_model = smf.logit(
        "accident ~ license_renew + Betel_binary",
        data=temp
    ).fit(disp=False)

    # Interaction model
    interaction_model = smf.logit(
        "accident ~ license_renew * Betel_binary",
        data=temp
    ).fit(disp=False)

    hypothesis_tables["License_to_betel_model"] = pd.DataFrame({
        "term": betel_model.params.index,
        "OR": np.exp(betel_model.params.values),
        "p_value": betel_model.pvalues.values
    })

    hypothesis_tables["License_accident_model"] = pd.DataFrame({
        "term": license_model.params.index,
        "OR": np.exp(license_model.params.values),
        "p_value": license_model.pvalues.values
    })

    hypothesis_tables["License_betel_accident_model"] = pd.DataFrame({
        "term": license_betel_model.params.index,
        "OR": np.exp(license_betel_model.params.values),
        "p_value": license_betel_model.pvalues.values
    })

    hypothesis_tables["License_betel_interaction_model"] = pd.DataFrame({
        "term": interaction_model.params.index,
        "OR": np.exp(interaction_model.params.values),
        "p_value": interaction_model.pvalues.values
    })

# ---------------------------------------------------------------------
# 9. Core final model
# ---------------------------------------------------------------------

core_terms = []

if age_col:
    core_terms.append("Age_num")
if license_col:
    core_terms.append("C(License_clean)")
if driving_hours_col:
    core_terms.append("Driving_hours_num")
if rbg_col:
    core_terms.append("RBG_num")
if smoking_col:
    core_terms.append("Smoking_binary")
if betel_col:
    core_terms.append("Betel_binary")

core_formula = "accident ~ " + " + ".join(core_terms)

core_data_cols = ["accident"]
for x in ["Age_num", "License_clean", "Driving_hours_num", "RBG_num", "Smoking_binary", "Betel_binary"]:
    if x in df.columns:
        core_data_cols.append(x)

core_data = df[core_data_cols].dropna()

core_model = smf.logit(core_formula, data=core_data).fit(disp=False)

core_results = pd.DataFrame({
    "term": core_model.params.index,
    "OR": np.exp(core_model.params.values),
    "CI_low": np.exp(core_model.conf_int()[0].values),
    "CI_high": np.exp(core_model.conf_int()[1].values),
    "p_value": core_model.pvalues.values,
})

# ---------------------------------------------------------------------
# 10. Export outputs
# ---------------------------------------------------------------------

with pd.ExcelWriter(OUT_DIR / "accident_history_874_reanalysis_outputs.xlsx", engine="openpyxl") as writer:
    pd.DataFrame([detected_columns]).to_excel(writer, sheet_name="Detected_columns", index=False)
    screening.to_excel(writer, sheet_name="All_variable_screening", index=False)
    category_details.to_excel(writer, sheet_name="Category_specific_ORs", index=False)
    core_results.to_excel(writer, sheet_name="Core_adjusted_model", index=False)

    for name, table in hypothesis_tables.items():
        sheet_name = re.sub(r"[^A-Za-z0-9_]", "_", name)[:31]
        table.to_excel(writer, sheet_name=sheet_name, index=False)

# ---------------------------------------------------------------------
# 11. Simple language interpretation file
# ---------------------------------------------------------------------

summary_text = f"""
# Simple Summary of the 874-driver Accident-History Re-analysis

## Dataset used

Only drivers with accident-history information were included.

Total complete cases: {len(df)}
Drivers with accident history: {int(df['accident'].sum())}
Accident-history prevalence: {df['accident'].mean() * 100:.1f}%

## What this analysis does

This analysis checks all available factors to see which ones are associated with accident history.

The factors are grouped into:

1. Demographic factors
2. Occupational factors
3. Lifestyle and behavioral factors
4. Metabolic and clinical factors
5. Urinary and renal factors

## Smoking, betel quid, and RBG

The analysis checks whether drivers with both smoking and betel quid intake have higher RBG and higher accident-history prevalence.

If smoking/betel groups are associated with accident history, and the association becomes weaker after adding RBG, then this supports a possible RBG-related pathway.

This should be described as an association, not proof of causation.

## License type, betel quid, and accident history

The analysis checks whether renew-license and new-license drivers differ in betel quid intake.

It also checks whether betel quid intake partly explains the relationship between license type and accident history.

Again, this is pathway-consistent evidence, not causal proof.

## Recommended interpretation style

Use simple wording:

'Drivers with this factor had higher accident-history prevalence.'

Avoid hard wording:

'This factor caused accident.'

## Best manuscript title

Factors Associated With Accident History Among Professional Drivers: A Complete-Case Analysis of 874 Drivers
"""

(OUT_DIR / "simple_language_summary.md").write_text(summary_text, encoding="utf-8")

print("Re-analysis complete.")
print("Outputs saved to:", OUT_DIR)