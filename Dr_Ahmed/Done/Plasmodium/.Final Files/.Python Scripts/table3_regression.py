#!/usr/bin/env python
"""
Explore multiple regression approaches for parasitemia (PD_level) vs traits.

Input (same folder as this script):
    Supplementary_File_SF_2b Traits.xlsx  (sheet 'Traits')

Models fitted:
  1) Multivariable linear regression (OLS):
        PD_level ~ Region_num + Age + Sex_num + Nat_num
  2) Univariable linear regressions (each predictor separately)
  3) Ordinal logistic regression (PD_level as ordered outcome)
  4) Binary logistic (PRIMARY): high vs low parasitemia
        PD3–4 (high) vs PD1–2 (low)
  5) Binary logistic (SENSITIVITY): extreme high vs others
        PD4 vs PD1–3

Output:
    Table3_regression_all_models.xlsx  with sheets:
        - OLS_raw
        - OLS_coeffs_tidy
        - OLS_model_stats
        - Univariable_OLS
        - Ordinal_logit_params
        - Ordinal_logit_stats
        - Logit_PD34_vs_12_params      (PRIMARY high vs low model)
        - Logit_PD34_vs_12_stats
        - Logit_PD4_vs_1to3_params     (sensitivity model)
        - Logit_PD4_vs_1to3_stats
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
import statsmodels.api as sm

# Ordinal logistic (if available in your statsmodels)
try:
    from statsmodels.miscmodels.ordinal_model import OrderedModel
    ORDINAL_AVAILABLE = True
except Exception:
    ORDINAL_AVAILABLE = False
    print("Warning: OrderedModel not available; ordinal logistic will be skipped.")


# --------------------------------------------------------------------
# 1. Load and pre-process data
# --------------------------------------------------------------------
here = Path(__file__).resolve().parent

excel_name = "Supplementary_File_SF_2b Traits.xlsx"
xlsx_path = here / excel_name
if not xlsx_path.exists():
    raise FileNotFoundError(
        f"Could not find '{excel_name}' in {here}.\n"
        "Make sure the traits Excel file is in the same folder as this script."
    )

print(f"Loading traits from: {xlsx_path}")
df = pd.read_excel(xlsx_path, sheet_name="Traits")

# Treat literal 'Unknown' as missing
df = df.replace("Unknown", np.nan)


def extract_number(val):
    """
    Extract numeric code from strings like 'Region1', 'Nationality2'.
    Returns float or np.nan.
    """
    if isinstance(val, str):
        m = re.search(r"(\d+(?:\.\d+)?)", val)
        if m:
            return float(m.group(1))
    return np.nan


# Map parasite density categories to ordinal 1–4
pd_map = {"PD 1": 1, "PD 2": 2, "PD 3": 3, "PD 4": 4}
df["PD_level"] = df["Parasite density"].map(pd_map)

df["Region_num"] = df["Region"].apply(extract_number)
df["Nat_num"] = df["Nationality"].apply(extract_number)
df["Sex_num"] = pd.to_numeric(df["Sex"], errors="coerce")
df["Age"] = pd.to_numeric(df["Age"], errors="coerce")

required = ["PD_level", "Region_num", "Age", "Sex_num", "Nat_num"]
sub = df.dropna(subset=required).copy()

print(f"Total rows in Traits sheet               : {len(df)}")
print(f"Rows used in regression (complete cases): {len(sub)}")

y = sub["PD_level"]
X_multi = sub[["Region_num", "Age", "Sex_num", "Nat_num"]]
X_multi = sm.add_constant(X_multi)

# To store all output tables for Excel
sheets = {}


# --------------------------------------------------------------------
# 2. Multivariable linear regression (OLS) = original Table 4 idea
# --------------------------------------------------------------------
model_ols = sm.OLS(y, X_multi).fit()

print("\n[OLS] Multivariable linear regression – parameter table:")
print(model_ols.summary().tables[1])

# Raw statsmodels-style table
summary2 = model_ols.summary2()
ols_raw_df = summary2.tables[1].reset_index()
ols_raw_df.rename(columns={"index": "Variable"}, inplace=True)
sheets["OLS_raw"] = ols_raw_df

# Tidy coefficient table
params = model_ols.params
bse = model_ols.bse
tvals = model_ols.tvalues
pvals = model_ols.pvalues
conf_int = model_ols.conf_int(alpha=0.05)
conf_int.columns = ["CI_lower", "CI_upper"]

ols_param_df = pd.concat([params, bse, tvals, pvals, conf_int], axis=1)
ols_param_df.columns = [
    "Coefficient", "Std_Error", "t_value", "p_value", "CI_lower", "CI_upper"
]
ols_param_df.index.name = "Variable"
ols_param_df.reset_index(inplace=True)

ols_param_df["Coefficient"] = ols_param_df["Coefficient"].round(4)
ols_param_df["Std_Error"] = ols_param_df["Std_Error"].round(4)
ols_param_df["t_value"] = ols_param_df["t_value"].round(3)
ols_param_df["p_value"] = ols_param_df["p_value"].map(lambda x: float(f"{x:.4g}"))
ols_param_df["CI_lower"] = ols_param_df["CI_lower"].round(4)
ols_param_df["CI_upper"] = ols_param_df["CI_upper"].round(4)

print("\n[OLS] Tidy coefficient table:")
print(ols_param_df)
sheets["OLS_coeffs_tidy"] = ols_param_df

# Model-level statistics
ols_stats = {
    "n_observations": int(model_ols.nobs),
    "R_squared": float(model_ols.rsquared),
    "Adj_R_squared": float(model_ols.rsquared_adj),
    "F_statistic": float(model_ols.fvalue) if model_ols.fvalue is not None else np.nan,
    "F_pvalue": float(model_ols.f_pvalue) if model_ols.f_pvalue is not None else np.nan,
    "AIC": float(model_ols.aic),
    "BIC": float(model_ols.bic),
}
ols_stats_df = pd.DataFrame(
    {
        "Metric": list(ols_stats.keys()),
        "Value": [
            round(v, 4) if isinstance(v, (int, float)) and not np.isnan(v) else v
            for v in ols_stats.values()
        ],
    }
)
print("\n[OLS] Model-level statistics:")
print(ols_stats_df)
sheets["OLS_model_stats"] = ols_stats_df


# --------------------------------------------------------------------
# 3. Univariable linear regressions (each predictor separately)
# --------------------------------------------------------------------
uni_rows = []
for var in ["Region_num", "Age", "Sex_num", "Nat_num"]:
    X_uni = sm.add_constant(sub[[var]])
    model_uni = sm.OLS(y, X_uni).fit()
    coef = model_uni.params[var]
    se = model_uni.bse[var]
    tval = model_uni.tvalues[var]
    pval = model_uni.pvalues[var]
    ci = model_uni.conf_int().loc[var]
    ci_low, ci_up = ci[0], ci[1]

    uni_rows.append(
        {
            "Predictor": var,
            "Coefficient": round(coef, 4),
            "Std_Error": round(se, 4),
            "t_value": round(tval, 3),
            "p_value": float(f"{pval:.4g}"),
            "CI_lower": round(ci_low, 4),
            "CI_upper": round(ci_up, 4),
            "R_squared": round(model_uni.rsquared, 4),
            "Adj_R_squared": round(model_uni.rsquared_adj, 4),
            "F_pvalue": float(f"{model_uni.f_pvalue:.4g}")
            if model_uni.f_pvalue is not None
            else np.nan,
        }
    )

uni_df = pd.DataFrame(uni_rows)
print("\n[Univariable OLS] Summary:")
print(uni_df)
sheets["Univariable_OLS"] = uni_df


# --------------------------------------------------------------------
# 4. Ordinal logistic regression (PD1 < PD2 < PD3 < PD4)
# --------------------------------------------------------------------
if ORDINAL_AVAILABLE:
    try:
        exog_ord = sub[["Region_num", "Age", "Sex_num", "Nat_num"]]
        mod_ord = OrderedModel(
            endog=y,
            exog=exog_ord,
            distr="logit",
        )
        res_ord = mod_ord.fit(method="bfgs", disp=False)
        print("\n[Ordinal logit] Fitted successfully.")

        params_o = res_ord.params
        bse_o = res_ord.bse
        zvals_o = params_o / bse_o
        pvals_o = res_ord.pvalues
        ci_o = res_ord.conf_int(alpha=0.05)
        ci_o.columns = ["CI_lower", "CI_upper"]

        ord_df = pd.concat([params_o, bse_o, zvals_o, pvals_o, ci_o], axis=1)
        ord_df.columns = [
            "Coefficient",
            "Std_Error",
            "z_value",
            "p_value",
            "CI_lower",
            "CI_upper",
        ]
        ord_df.index.name = "Parameter"
        ord_df.reset_index(inplace=True)

        ord_df["Coefficient"] = ord_df["Coefficient"].round(4)
        ord_df["Std_Error"] = ord_df["Std_Error"].round(4)
        ord_df["z_value"] = ord_df["z_value"].round(3)
        ord_df["p_value"] = ord_df["p_value"].map(lambda x: float(f"{x:.4g}"))
        ord_df["CI_lower"] = ord_df["CI_lower"].round(4)
        ord_df["CI_upper"] = ord_df["CI_upper"].round(4)

        print("\n[Ordinal logit] Parameter table:")
        print(ord_df)
        sheets["Ordinal_logit_params"] = ord_df

        ord_stats = {
            "n_observations": float(getattr(res_ord, "nobs", np.nan)),
            "log_likelihood": float(getattr(res_ord, "llf", np.nan)),
            "AIC": float(getattr(res_ord, "aic", np.nan)),
            "BIC": float(getattr(res_ord, "bic", np.nan)),
        }
        ord_stats_df = pd.DataFrame(
            {
                "Metric": list(ord_stats.keys()),
                "Value": [
                    round(v, 4)
                    if isinstance(v, (int, float)) and not np.isnan(v)
                    else v
                    for v in ord_stats.values()
                ],
            }
        )
        sheets["Ordinal_logit_stats"] = ord_stats_df

    except Exception as e:
        print(f"[Ordinal logit] Failed to fit model: {e}")
else:
    print("[Ordinal logit] Skipped (OrderedModel not available).")


# --------------------------------------------------------------------
# 5. Binary logistic regressions
#    (PRIMARY) PD3–4 vs PD1–2  = high vs low
#    (SENSITIVITY) PD4 vs PD1–3
# --------------------------------------------------------------------
def fit_logit_binary(name, y_binary, sub_df):
    """
    Fit a binary logistic regression:
        y_binary (0/1) ~ Region_num + Age + Sex_num + Nat_num
    Returns (params_df, stats_df) or (None, None) on failure.
    """
    mask = ~y_binary.isna()
    yb = y_binary[mask]
    Xb = sub_df.loc[mask, ["Region_num", "Age", "Sex_num", "Nat_num"]]
    Xb = sm.add_constant(Xb)

    if yb.nunique() < 2:
        print(f"[{name}] Outcome has <2 classes; skipping.")
        return None, None

    try:
        model_logit = sm.Logit(yb, Xb).fit(disp=False)
    except Exception as e:
        print(f"[{name}] Logit failed: {e}")
        return None, None

    print(f"\n[{name}] Logistic regression fitted.")

    params = model_logit.params
    bse = model_logit.bse
    zvals = params / bse
    pvals = model_logit.pvalues
    ci = model_logit.conf_int(alpha=0.05)
    ci.columns = ["CI_lower", "CI_upper"]

    # Odds ratios and 95% CI in OR space
    or_vals = np.exp(params)
    or_ci = np.exp(ci)
    or_ci.columns = ["OR_CI_lower", "OR_CI_upper"]

    logit_df = pd.concat(
        [params, bse, zvals, pvals, ci, or_vals, or_ci], axis=1
    )
    logit_df.columns = [
        "Coefficient",
        "Std_Error",
        "z_value",
        "p_value",
        "CI_lower",
        "CI_upper",
        "OR",
        "OR_CI_lower",
        "OR_CI_upper",
    ]
    logit_df.index.name = "Variable"
    logit_df.reset_index(inplace=True)

    for col in [
        "Coefficient",
        "Std_Error",
        "z_value",
        "CI_lower",
        "CI_upper",
        "OR",
        "OR_CI_lower",
        "OR_CI_upper",
    ]:
        logit_df[col] = logit_df[col].round(4)
    logit_df["p_value"] = logit_df["p_value"].map(lambda x: float(f"{x:.4g}"))

    llf = float(getattr(model_logit, "llf", np.nan))
    llnull = float(getattr(model_logit, "llnull", np.nan)) if hasattr(
        model_logit, "llnull"
    ) else np.nan
    if not np.isnan(llf) and not np.isnan(llnull):
        pseudo_r2 = 1 - llf / llnull
    else:
        pseudo_r2 = np.nan

    stats = {
        "n_observations": int(model_logit.nobs),
        "log_likelihood": llf,
        "log_likelihood_null": llnull,
        "Pseudo_R2_McFadden": pseudo_r2,
        "AIC": float(model_logit.aic),
        "BIC": float(model_logit.bic),
    }
    stats_df = pd.DataFrame(
        {
            "Metric": list(stats.keys()),
            "Value": [
                round(v, 4)
                if isinstance(v, (int, float)) and not np.isnan(v)
                else v
                for v in stats.values()
            ],
        }
    )

    return logit_df, stats_df


# (PRIMARY) PD3–4 vs PD1–2  → high vs low parasitemia
sub["PD34_vs_12"] = (sub["PD_level"] >= 3).astype(int)
logit_primary_params, logit_primary_stats = fit_logit_binary(
    "Logit_PD34_vs_12", sub["PD34_vs_12"], sub
)
if logit_primary_params is not None:
    sheets["Logit_PD34_vs_12_params"] = logit_primary_params
    sheets["Logit_PD34_vs_12_stats"] = logit_primary_stats

# (SENSITIVITY) PD4 vs PD1–3  → extreme high vs others
sub["PD4_vs_1to3"] = (sub["PD_level"] == 4).astype(int)
logit_sens_params, logit_sens_stats = fit_logit_binary(
    "Logit_PD4_vs_1to3", sub["PD4_vs_1to3"], sub
)
if logit_sens_params is not None:
    sheets["Logit_PD4_vs_1to3_params"] = logit_sens_params
    sheets["Logit_PD4_vs_1to3_stats"] = logit_sens_stats


# --------------------------------------------------------------------
# 6. Save all tables to Excel
# --------------------------------------------------------------------
out_path = here / "Table3_regression_all_models.xlsx"
with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
    for sheet_name, df_out in sheets.items():
        if df_out is None or df_out.empty:
            continue
        safe_name = sheet_name[:31]  # Excel sheet names max length 31
        df_out.to_excel(writer, sheet_name=safe_name, index=False)

print(f"\nSaved all regression/model tables to: {out_path}")
