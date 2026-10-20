# ================================================================
# ROBUSTNESS AND DIAGNOSTIC TESTING SUITE
# Digital Choice Study
#
# Paste this AFTER your main analysis models have been estimated.
# It assumes these objects already exist in your script:
#   df, OUTPUT_FOLDER, robust_lpm, cronbach_alpha,
#   model_1, model_2, model_sq, model_switch,
#   model_platform, model_fl
#
# IMPORTANT:
# - Randomization integrity and treatment contamination cannot be
#   proven by code. They must be documented from how links were sent.
# - Influence flags are for sensitivity checks, NOT automatic deletion.
# ================================================================

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from scipy import stats
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.contingency_tables import cochrans_q, mcnemar


ROBUSTNESS_FOLDER = Path(OUTPUT_FOLDER) / "Robustness_Checks"
ROBUSTNESS_FOLDER.mkdir(exist_ok=True)

print("\n" + "=" * 70)
print("ROBUSTNESS AND DIAGNOSTIC TESTING")
print("=" * 70)


# ================================================================
# R1. RAW MISSING-VALUE AUDIT
# ================================================================

raw_questions = [f"Q{i}" for i in range(1, 18)]

raw_for_missing = (
    df[raw_questions]
    .replace(r"^\s*$", np.nan, regex=True)
)

missing_table = pd.DataFrame({
    "Question": raw_questions,
    "Missing_N": raw_for_missing.isna().sum().values,
    "Missing_Percent": raw_for_missing.isna().mean().values * 100
})

print("\nR1. RAW MISSING-VALUE AUDIT")
print(missing_table.round(2))

missing_table.to_csv(
    ROBUSTNESS_FOLDER / "R01_missing_values.csv",
    index=False
)


# ================================================================
# R2. POTENTIAL EXACT DUPLICATE RESPONSE PATTERNS
# FLAG ONLY. DO NOT AUTOMATICALLY DELETE.
# ================================================================

duplicate_mask = df.duplicated(
    subset=raw_questions,
    keep=False
)

potential_duplicates = df.loc[
    duplicate_mask,
    ["Default", "Form"] + raw_questions
].copy()

potential_duplicates.insert(
    0,
    "Original_Row",
    potential_duplicates.index
)

print("\nR2. POTENTIAL EXACT DUPLICATE RESPONSE PATTERNS")
print("Rows involved:", int(duplicate_mask.sum()))

potential_duplicates.to_csv(
    ROBUSTNESS_FOLDER / "R02_potential_duplicate_patterns.csv",
    index=False
)


# ================================================================
# R3. CODING RANGE / INVALID-VALUE CHECKS
# ================================================================

range_rules = {
    "Default": (0, 1),
    "UPIChoice": (0, 1),
    "ExperienceBreadth": (0, 5),
    "FrequencyOrdinal": (0, 5),
    "PriorUPI": (0, 5),
    "StatusQuo": (0, 1),
    "SwitchingEffort": (0, 1),
    "PlatformReliance": (0, 1),
    "FinancialLiteracy": (0, 3)
}

range_results = []

for variable, (lower, upper) in range_rules.items():
    s = pd.to_numeric(df[variable], errors="coerce")

    invalid = (
        s.notna()
        & ((s < lower) | (s > upper))
    )

    range_results.append({
        "Variable": variable,
        "Expected_Min": lower,
        "Expected_Max": upper,
        "Observed_Min": s.min(),
        "Observed_Max": s.max(),
        "Invalid_N": int(invalid.sum()),
        "Missing_N": int(s.isna().sum())
    })

range_table = pd.DataFrame(range_results)

print("\nR3. CODING RANGE CHECKS")
print(range_table)

range_table.to_csv(
    ROBUSTNESS_FOLDER / "R03_range_checks.csv",
    index=False
)


# ================================================================
# R4. STANDARDIZED MEAN DIFFERENCES (SMD)
# ================================================================

balance_vars = [
    "ExperienceBreadth",
    "FrequencyOrdinal",
    "PriorUPI",
    "StatusQuo",
    "SwitchingEffort",
    "PlatformReliance",
    "FinancialLiteracy"
]

smd_results = []

for variable in balance_vars:
    control = df.loc[df["Default"] == 0, variable].dropna().astype(float)
    treatment = df.loc[df["Default"] == 1, variable].dropna().astype(float)

    mean_c = control.mean()
    mean_t = treatment.mean()
    var_c = control.var(ddof=1)
    var_t = treatment.var(ddof=1)
    pooled_sd = np.sqrt((var_c + var_t) / 2)

    smd = (
        (mean_t - mean_c) / pooled_sd
        if pooled_sd > 0
        else np.nan
    )

    smd_results.append({
        "Variable": variable,
        "Control_Mean": mean_c,
        "Treatment_Mean": mean_t,
        "Difference_T_minus_C": mean_t - mean_c,
        "SMD": smd,
        "Abs_SMD": abs(smd) if pd.notna(smd) else np.nan
    })

smd_table = pd.DataFrame(smd_results)

print("\nR4. STANDARDIZED MEAN DIFFERENCES")
print(smd_table.round(4))

smd_table.to_csv(
    ROBUSTNESS_FOLDER / "R04_standardized_mean_differences.csv",
    index=False
)


# ================================================================
# R5. JOINT BALANCE TEST
# ================================================================

joint_balance_model = smf.ols(
    """
    Default
    ~ ExperienceBreadth
    + FrequencyOrdinal
    + PriorUPI
    + StatusQuo
    + SwitchingEffort
    + PlatformReliance
    + FinancialLiteracy
    """,
    data=df
).fit(cov_type="HC3")

joint_test = joint_balance_model.f_test(
    """
    ExperienceBreadth = 0,
    FrequencyOrdinal = 0,
    PriorUPI = 0,
    StatusQuo = 0,
    SwitchingEffort = 0,
    PlatformReliance = 0,
    FinancialLiteracy = 0
    """
)

joint_balance_table = pd.DataFrame([{
    "Test": "Joint balance test",
    "Statistic": float(np.asarray(joint_test.fvalue).squeeze()),
    "p_value": float(np.asarray(joint_test.pvalue).squeeze()),
    "N": int(joint_balance_model.nobs)
}])

print("\nR5. JOINT BALANCE TEST")
print(joint_balance_table.round(4))

joint_balance_table.to_csv(
    ROBUSTNESS_FOLDER / "R05_joint_balance_test.csv",
    index=False
)


# ================================================================
# R6. SCALE ITEM DIAGNOSTICS
# Corrected item-total correlation + alpha if item deleted
# ================================================================

scale_definitions = {
    "StatusQuo": ["SQ1", "SQ2", "SQ3"],
    "SwitchingEffort": ["Switch2", "Switch4", "Switch6"],
    "PlatformReliance": ["PR1", "PR2", "PR3"]
}

scale_diagnostics = []

for scale_name, items in scale_definitions.items():
    complete = df[items].dropna().astype(float)
    full_alpha = cronbach_alpha(complete)

    for item in items:
        other_items = [x for x in items if x != item]
        other_score = complete[other_items].mean(axis=1)

        if complete[item].std(ddof=1) == 0 or other_score.std(ddof=1) == 0:
            item_total_corr = np.nan
        else:
            item_total_corr = complete[item].corr(other_score)

        alpha_deleted = cronbach_alpha(complete[other_items])

        scale_diagnostics.append({
            "Scale": scale_name,
            "Item": item,
            "Full_Alpha": full_alpha,
            "Corrected_Item_Total_Correlation": item_total_corr,
            "Alpha_If_Item_Deleted": alpha_deleted,
            "N_Complete": len(complete)
        })

scale_diagnostics_table = pd.DataFrame(scale_diagnostics)

print("\nR6. SCALE ITEM DIAGNOSTICS")
print(scale_diagnostics_table.round(4))

scale_diagnostics_table.to_csv(
    ROBUSTNESS_FOLDER / "R06_scale_item_diagnostics.csv",
    index=False
)


# ================================================================
# R7. CORRELATION MATRIX
# ================================================================

correlation_vars = [
    "ExperienceBreadth",
    "FrequencyOrdinal",
    "PriorUPI",
    "StatusQuo",
    "SwitchingEffort",
    "PlatformReliance",
    "FinancialLiteracy"
]

correlation_matrix = df[correlation_vars].astype(float).corr()

print("\nR7. CORRELATION MATRIX")
print(correlation_matrix.round(3))

correlation_matrix.to_csv(
    ROBUSTNESS_FOLDER / "R07_correlation_matrix.csv"
)


# ================================================================
# R8. VARIANCE INFLATION FACTORS (VIF)
# ================================================================

vif_data = (
    df[
        [
            "Default",
            "ExperienceBreadth",
            "FrequencyOrdinal",
            "PriorUPI",
            "StatusQuo",
            "SwitchingEffort",
            "PlatformReliance",
            "FinancialLiteracy"
        ]
    ]
    .dropna()
    .astype(float)
)

X_vif = sm.add_constant(vif_data)

vif_table = pd.DataFrame({
    "Variable": X_vif.columns,
    "VIF": [
        variance_inflation_factor(X_vif.values, i)
        for i in range(X_vif.shape[1])
    ]
})

print("\nR8. VIF")
print(vif_table.round(3))

vif_table.to_csv(
    ROBUSTNESS_FOLDER / "R08_vif.csv",
    index=False
)


# ================================================================
# R9. HETEROSKEDASTICITY DIAGNOSTICS
# HC3 remains the primary correction regardless of these tests.
# ================================================================

hetero_results = []

for name, model in [("Model 1", model_1), ("Model 2", model_2)]:
    resid = np.asarray(model.resid)
    exog = np.asarray(model.model.exog)

    bp = het_breuschpagan(resid, exog)

    hetero_results.append({
        "Model": name,
        "Test": "Breusch-Pagan",
        "Statistic": bp[0],
        "p_value": bp[1]
    })

    try:
        white = het_white(resid, exog)
        hetero_results.append({
            "Model": name,
            "Test": "White",
            "Statistic": white[0],
            "p_value": white[1]
        })
    except Exception as exc:
        hetero_results.append({
            "Model": name,
            "Test": "White",
            "Statistic": np.nan,
            "p_value": np.nan,
            "Note": str(exc)
        })

hetero_table = pd.DataFrame(hetero_results)

print("\nR9. HETEROSKEDASTICITY TESTS")
print(hetero_table.round(4))

hetero_table.to_csv(
    ROBUSTNESS_FOLDER / "R09_heteroskedasticity_tests.csv",
    index=False
)


# ================================================================
# R10. INFLUENCE DIAGNOSTICS FOR FULL ADJUSTED MODEL
# Cook's D, leverage, DFBETA(Default)
# ================================================================

full_ols_for_influence = smf.ols(
    model_2.model.formula,
    data=df
).fit()

influence = full_ols_for_influence.get_influence()
influence_frame = influence.summary_frame()
influence_frame["Original_Row"] = influence_frame.index

n_influence = int(full_ols_for_influence.nobs)
p_influence = int(len(full_ols_for_influence.params))

cook_threshold = 4 / n_influence
leverage_threshold = 2 * p_influence / n_influence
dfbeta_threshold = 2 / np.sqrt(n_influence)

influence_frame["Cook_Flag"] = (
    influence_frame["cooks_d"] > cook_threshold
)

influence_frame["Leverage_Flag"] = (
    influence_frame["hat_diag"] > leverage_threshold
)

default_position = list(full_ols_for_influence.params.index).index("Default")
influence_frame["DFBETA_Default"] = influence.dfbetas[:, default_position]
influence_frame["DFBETA_Default_Flag"] = (
    influence_frame["DFBETA_Default"].abs() > dfbeta_threshold
)

print("\nR10. INFLUENCE DIAGNOSTICS")
print("Cook threshold:", round(cook_threshold, 4))
print("Leverage threshold:", round(leverage_threshold, 4))
print("DFBETA threshold:", round(dfbeta_threshold, 4))
print("Cook flags:", int(influence_frame["Cook_Flag"].sum()))
print("Leverage flags:", int(influence_frame["Leverage_Flag"].sum()))
print(
    "DFBETA(Default) flags:",
    int(influence_frame["DFBETA_Default_Flag"].sum())
)

influence_frame.to_csv(
    ROBUSTNESS_FOLDER / "R10_influence_diagnostics.csv",
    index=False
)


# ================================================================
# R11. SENSITIVITY: FULL MODEL WITHOUT COOK-FLAGGED OBSERVATIONS
# This is a sensitivity check only, not a deletion recommendation.
# ================================================================

cook_flagged_rows = influence_frame.loc[
    influence_frame["Cook_Flag"],
    "Original_Row"
].tolist()

base_ci = model_2.conf_int().loc["Default"]

sensitivity_rows = [{
    "Specification": "Full model - all observations",
    "Default_Coefficient": model_2.params["Default"],
    "Robust_SE": model_2.bse["Default"],
    "p_value": model_2.pvalues["Default"],
    "CI_Lower": base_ci.iloc[0],
    "CI_Upper": base_ci.iloc[1],
    "N": int(model_2.nobs)
}]

if len(cook_flagged_rows) > 0:
    sensitivity_df = df.drop(index=cook_flagged_rows)

    model_2_no_cook = robust_lpm(
        model_2.model.formula,
        sensitivity_df
    )

    ci_no_cook = model_2_no_cook.conf_int().loc["Default"]

    sensitivity_rows.append({
        "Specification": "Full model - Cook flagged excluded",
        "Default_Coefficient": model_2_no_cook.params["Default"],
        "Robust_SE": model_2_no_cook.bse["Default"],
        "p_value": model_2_no_cook.pvalues["Default"],
        "CI_Lower": ci_no_cook.iloc[0],
        "CI_Upper": ci_no_cook.iloc[1],
        "N": int(model_2_no_cook.nobs)
    })

influence_sensitivity_table = pd.DataFrame(sensitivity_rows)

print("\nR11. INFLUENCE SENSITIVITY")
print(influence_sensitivity_table.round(4))

influence_sensitivity_table.to_csv(
    ROBUSTNESS_FOLDER / "R11_influence_sensitivity.csv",
    index=False
)


# ================================================================
# R12. LPM PREDICTED-PROBABILITY CHECK
# ================================================================

prediction_results = []

for name, model in [("Model 1", model_1), ("Model 2", model_2)]:
    pred = np.asarray(model.fittedvalues)

    prediction_results.append({
        "Model": name,
        "Min_Prediction": pred.min(),
        "Max_Prediction": pred.max(),
        "N_Below_0": int((pred < 0).sum()),
        "Percent_Below_0": float((pred < 0).mean() * 100),
        "N_Above_1": int((pred > 1).sum()),
        "Percent_Above_1": float((pred > 1).mean() * 100)
    })

prediction_table = pd.DataFrame(prediction_results)

print("\nR12. LPM PREDICTED PROBABILITY CHECK")
print(prediction_table.round(3))

prediction_table.to_csv(
    ROBUSTNESS_FOLDER / "R12_lpm_prediction_range.csv",
    index=False
)


# ================================================================
# R13. TWO-SAMPLE DIFFERENCE IN PROPORTIONS
# ================================================================

treatment_y = df.loc[
    df["Default"] == 1,
    "UPIChoice"
].dropna().astype(int)

control_y = df.loc[
    df["Default"] == 0,
    "UPIChoice"
].dropna().astype(int)

count = np.array([
    treatment_y.sum(),
    control_y.sum()
])

nobs = np.array([
    len(treatment_y),
    len(control_y)
])

z_stat, z_p = proportions_ztest(
    count=count,
    nobs=nobs,
    alternative="two-sided"
)

observed_ate = treatment_y.mean() - control_y.mean()

prop_test_table = pd.DataFrame([{
    "Test": "Two-sample proportions z-test",
    "Treatment_UPI_Rate": treatment_y.mean(),
    "Control_UPI_Rate": control_y.mean(),
    "Difference": observed_ate,
    "Statistic": z_stat,
    "p_value": z_p
}])

print("\nR13. DIFFERENCE IN PROPORTIONS")
print(prop_test_table.round(4))

prop_test_table.to_csv(
    ROBUSTNESS_FOLDER / "R13_difference_in_proportions.csv",
    index=False
)


# ================================================================
# R14. FISHER EXACT TEST
# ================================================================

two_by_two = np.array([
    [
        int(treatment_y.sum()),
        int(len(treatment_y) - treatment_y.sum())
    ],
    [
        int(control_y.sum()),
        int(len(control_y) - control_y.sum())
    ]
])

fisher_odds, fisher_p = stats.fisher_exact(
    two_by_two,
    alternative="two-sided"
)

fisher_table = pd.DataFrame([{
    "Treatment_UPI": two_by_two[0, 0],
    "Treatment_NonUPI": two_by_two[0, 1],
    "Control_UPI": two_by_two[1, 0],
    "Control_NonUPI": two_by_two[1, 1],
    "Odds_Ratio": fisher_odds,
    "p_value": fisher_p
}])

print("\nR14. FISHER EXACT TEST")
print(fisher_table.round(4))

fisher_table.to_csv(
    ROBUSTNESS_FOLDER / "R14_fisher_exact.csv",
    index=False
)


# ================================================================
# R15. RANDOMIZATION / PERMUTATION INFERENCE
# VALID ONLY IF TREATMENT ASSIGNMENT WAS GENUINELY RANDOMIZED.
# ================================================================

RI_REPETITIONS = 20000
RI_SEED = 20260902
rng_ri = np.random.default_rng(RI_SEED)

ri_data = df[["Default", "UPIChoice"]].dropna().copy()
d_obs = ri_data["Default"].to_numpy(dtype=int)
y_obs = ri_data["UPIChoice"].to_numpy(dtype=float)

observed_ate_ri = (
    y_obs[d_obs == 1].mean()
    - y_obs[d_obs == 0].mean()
)

null_ates = np.empty(RI_REPETITIONS)

for b in range(RI_REPETITIONS):
    permuted_d = rng_ri.permutation(d_obs)

    null_ates[b] = (
        y_obs[permuted_d == 1].mean()
        - y_obs[permuted_d == 0].mean()
    )

ri_p = (
    np.sum(np.abs(null_ates) >= abs(observed_ate_ri)) + 1
) / (RI_REPETITIONS + 1)

ri_table = pd.DataFrame([{
    "Observed_ATE": observed_ate_ri,
    "Permutations": RI_REPETITIONS,
    "Two_Sided_p_value": ri_p,
    "Null_Mean": null_ates.mean(),
    "Null_SD": null_ates.std(ddof=1)
}])

print("\nR15. RANDOMIZATION INFERENCE")
print(ri_table.round(4))

ri_table.to_csv(
    ROBUSTNESS_FOLDER / "R15_randomization_inference.csv",
    index=False
)


# ================================================================
# R16. STRATIFIED BOOTSTRAP OF RAW ATE
# ================================================================

BOOT_REPETITIONS = 10000
BOOT_SEED = 20260903
rng_boot = np.random.default_rng(BOOT_SEED)

treat_array = treatment_y.to_numpy(dtype=float)
control_array = control_y.to_numpy(dtype=float)
bootstrap_ates = np.empty(BOOT_REPETITIONS)

for b in range(BOOT_REPETITIONS):
    boot_t = rng_boot.choice(
        treat_array,
        size=len(treat_array),
        replace=True
    )

    boot_c = rng_boot.choice(
        control_array,
        size=len(control_array),
        replace=True
    )

    bootstrap_ates[b] = boot_t.mean() - boot_c.mean()

boot_ci = np.percentile(bootstrap_ates, [2.5, 97.5])

bootstrap_table = pd.DataFrame([{
    "Observed_ATE": observed_ate,
    "Bootstrap_SE": bootstrap_ates.std(ddof=1),
    "CI_2.5": boot_ci[0],
    "CI_97.5": boot_ci[1],
    "Replications": BOOT_REPETITIONS
}])

print("\nR16. BOOTSTRAP ATE")
print(bootstrap_table.round(4))

bootstrap_table.to_csv(
    ROBUSTNESS_FOLDER / "R16_bootstrap_ate.csv",
    index=False
)


# ================================================================
# R17. MODEL-SPECIFICATION STABILITY
# ================================================================

model_basic_1 = robust_lpm(
    "UPIChoice ~ Default + PriorUPI",
    df
)

model_basic_2 = robust_lpm(
    """
    UPIChoice
    ~ Default
    + PriorUPI
    + ExperienceBreadth
    + C(FrequencyCategory)
    """,
    df
)

stability_models = [
    ("M1 Default only", model_1),
    ("M2 + Prior UPI", model_basic_1),
    ("M3 + Basic payment controls", model_basic_2),
    ("M4 Full adjusted", model_2)
]

stability_rows = []

for model_name, model in stability_models:
    ci = model.conf_int().loc["Default"]

    stability_rows.append({
        "Model": model_name,
        "Default_Coefficient": model.params["Default"],
        "Robust_SE": model.bse["Default"],
        "p_value": model.pvalues["Default"],
        "CI_Lower": ci.iloc[0],
        "CI_Upper": ci.iloc[1],
        "N": int(model.nobs),
        "R_squared": getattr(model, "rsquared", np.nan)
    })

stability_table = pd.DataFrame(stability_rows)

print("\nR17. MODEL SPECIFICATION STABILITY")
print(stability_table.round(4))

stability_table.to_csv(
    ROBUSTNESS_FOLDER / "R17_model_specification_stability.csv",
    index=False
)


# ================================================================
# R18. MULTIPLE-TESTING CORRECTION FOR MODERATOR INTERACTIONS
# ================================================================

interaction_p_values = {
    "Default x StatusQuo": model_sq.pvalues["Default:StatusQuo_c"],
    "Default x SwitchingEffort": model_switch.pvalues["Default:SwitchingEffort_c"],
    "Default x PlatformReliance": model_platform.pvalues["Default:PlatformReliance_c"],
    "Default x FinancialLiteracy": model_fl.pvalues["Default:FinancialLiteracy_c"]
}

interaction_names = list(interaction_p_values.keys())
raw_pvals = np.array(list(interaction_p_values.values()), dtype=float)

holm_reject, holm_p, _, _ = multipletests(
    raw_pvals,
    alpha=0.05,
    method="holm"
)

fdr_reject, fdr_p, _, _ = multipletests(
    raw_pvals,
    alpha=0.05,
    method="fdr_bh"
)

multiple_testing_table = pd.DataFrame({
    "Interaction": interaction_names,
    "Raw_p": raw_pvals,
    "Holm_Adjusted_p": holm_p,
    "Holm_Reject_5pct": holm_reject,
    "BH_FDR_Adjusted_p": fdr_p,
    "BH_FDR_Reject_5pct": fdr_reject
})

print("\nR18. MULTIPLE TESTING CORRECTION")
print(multiple_testing_table.round(4))

multiple_testing_table.to_csv(
    ROBUSTNESS_FOLDER / "R18_multiple_testing_interactions.csv",
    index=False
)


# ================================================================
# R19. COCHRAN'S Q TEST FOR 2/4/6 STEP SWITCHING
# Same respondents answer all three items.
# ================================================================

switch_complete = (
    df[["Switch2", "Switch4", "Switch6"]]
    .dropna()
    .astype(int)
)

cochran_result = cochrans_q(
    switch_complete.to_numpy()
)

cochran_table = pd.DataFrame([{
    "Test": "Cochran Q",
    "Statistic": cochran_result.statistic,
    "p_value": cochran_result.pvalue,
    "N": len(switch_complete)
}])

print("\nR19. COCHRAN Q - SWITCHING FRICTION")
print(cochran_table.round(4))

cochran_table.to_csv(
    ROBUSTNESS_FOLDER / "R19_cochran_q_switching.csv",
    index=False
)


# ================================================================
# R20. PAIRWISE MCNEMAR TESTS + HOLM CORRECTION
# ================================================================

switch_pairs = [
    ("Switch2", "Switch4"),
    ("Switch4", "Switch6"),
    ("Switch2", "Switch6")
]

mcnemar_rows = []

for first, second in switch_pairs:
    pair_data = switch_complete[[first, second]]

    table = pd.crosstab(
        pair_data[first],
        pair_data[second]
    ).reindex(
        index=[0, 1],
        columns=[0, 1],
        fill_value=0
    )

    result = mcnemar(
        table.to_numpy(),
        exact=True
    )

    mcnemar_rows.append({
        "Comparison": f"{first} vs {second}",
        "00": int(table.loc[0, 0]),
        "01": int(table.loc[0, 1]),
        "10": int(table.loc[1, 0]),
        "11": int(table.loc[1, 1]),
        "Raw_p": result.pvalue
    })

mcnemar_table = pd.DataFrame(mcnemar_rows)

mcnemar_holm = multipletests(
    mcnemar_table["Raw_p"].to_numpy(),
    alpha=0.05,
    method="holm"
)

mcnemar_table["Holm_Adjusted_p"] = mcnemar_holm[1]
mcnemar_table["Holm_Reject_5pct"] = mcnemar_holm[0]

print("\nR20. MCNEMAR PAIRWISE TESTS")
print(mcnemar_table.round(4))

mcnemar_table.to_csv(
    ROBUSTNESS_FOLDER / "R20_mcnemar_switching_pairs.csv",
    index=False
)


# ================================================================
# R21. THREE-CATEGORY PAYMENT CHOICE:
# CHI-SQUARE + PERMUTATION P-VALUE
# ================================================================

choice3_data = df[["Default", "PaymentChoice3"]].dropna().copy()

choice3_table = pd.crosstab(
    choice3_data["Default"],
    choice3_data["PaymentChoice3"]
)

chi2_stat, chi2_p, chi2_df, expected = stats.chi2_contingency(
    choice3_table.to_numpy(),
    correction=False
)

CHOICE3_PERMUTATIONS = 20000
CHOICE3_SEED = 20260904
rng_choice3 = np.random.default_rng(CHOICE3_SEED)

d3 = choice3_data["Default"].to_numpy(dtype=int)
y3 = choice3_data["PaymentChoice3"].astype(str).to_numpy()
categories3 = list(choice3_table.columns)

perm_chi2 = np.empty(CHOICE3_PERMUTATIONS)

for b in range(CHOICE3_PERMUTATIONS):
    perm_d3 = rng_choice3.permutation(d3)

    perm_table = pd.crosstab(
        pd.Series(perm_d3),
        pd.Series(y3)
    ).reindex(
        index=[0, 1],
        columns=categories3,
        fill_value=0
    )

    perm_chi2[b] = stats.chi2_contingency(
        perm_table.to_numpy(),
        correction=False
    )[0]

choice3_perm_p = (
    np.sum(perm_chi2 >= chi2_stat) + 1
) / (CHOICE3_PERMUTATIONS + 1)

choice3_test_table = pd.DataFrame([{
    "ChiSquare": chi2_stat,
    "Asymptotic_p": chi2_p,
    "Degrees_of_Freedom": chi2_df,
    "Minimum_Expected_Cell": np.asarray(expected).min(),
    "Permutation_p": choice3_perm_p,
    "Permutations": CHOICE3_PERMUTATIONS
}])

print("\nR21. THREE-CATEGORY OUTCOME TEST")
print(choice3_table)
print(choice3_test_table.round(4))

choice3_table.to_csv(
    ROBUSTNESS_FOLDER / "R21_payment_choice_3category_table.csv"
)

choice3_test_table.to_csv(
    ROBUSTNESS_FOLDER / "R21_payment_choice_3category_test.csv",
    index=False
)


# ================================================================
# R22. STATUS QUO ITEM-LEVEL ROBUSTNESS
# Useful because StatusQuo alpha is weak.
# ================================================================

for item in ["SQ1", "SQ2", "SQ3"]:
    df[item + "_c"] = df[item] - df[item].mean()

statusquo_item_models = []

for item in ["SQ1", "SQ2", "SQ3"]:
    item_c = item + "_c"

    formula = f"""
    UPIChoice
    ~ Default * {item_c}
    + ExperienceBreadth
    + C(FrequencyCategory)
    + PriorUPI
    + SwitchingEffort
    + PlatformReliance
    + FinancialLiteracy
    """

    model_item = robust_lpm(formula, df)
    interaction_name = f"Default:{item_c}"

    statusquo_item_models.append({
        "Item": item,
        "Interaction_Coefficient": model_item.params[interaction_name],
        "Robust_SE": model_item.bse[interaction_name],
        "p_value": model_item.pvalues[interaction_name],
        "N": int(model_item.nobs)
    })

statusquo_item_table = pd.DataFrame(statusquo_item_models)

print("\nR22. STATUS QUO ITEM-LEVEL ROBUSTNESS")
print(statusquo_item_table.round(4))

statusquo_item_table.to_csv(
    ROBUSTNESS_FOLDER / "R22_statusquo_item_robustness.csv",
    index=False
)


# ================================================================
# R23. FINAL ROBUSTNESS SUMMARY
# ================================================================

platform_holm_p = multiple_testing_table.loc[
    multiple_testing_table["Interaction"] == "Default x PlatformReliance",
    "Holm_Adjusted_p"
].iloc[0]

robustness_summary = pd.DataFrame([
    {
        "Check": "Primary LPM Default",
        "Estimate": model_1.params["Default"],
        "p_value": model_1.pvalues["Default"]
    },
    {
        "Check": "Full adjusted LPM Default",
        "Estimate": model_2.params["Default"],
        "p_value": model_2.pvalues["Default"]
    },
    {
        "Check": "Difference in proportions",
        "Estimate": observed_ate,
        "p_value": z_p
    },
    {
        "Check": "Fisher exact",
        "Estimate": fisher_odds,
        "p_value": fisher_p
    },
    {
        "Check": "Randomization inference",
        "Estimate": observed_ate_ri,
        "p_value": ri_p
    },
    {
        "Check": "Default x PlatformReliance raw",
        "Estimate": model_platform.params["Default:PlatformReliance_c"],
        "p_value": model_platform.pvalues["Default:PlatformReliance_c"]
    },
    {
        "Check": "Default x PlatformReliance Holm-adjusted",
        "Estimate": model_platform.params["Default:PlatformReliance_c"],
        "p_value": platform_holm_p
    },
    {
        "Check": "Cochran Q switching friction",
        "Estimate": cochran_result.statistic,
        "p_value": cochran_result.pvalue
    }
])

print("\n" + "=" * 70)
print("FINAL ROBUSTNESS SUMMARY")
print("=" * 70)
print(robustness_summary.round(4))

robustness_summary.to_csv(
    ROBUSTNESS_FOLDER / "R23_final_robustness_summary.csv",
    index=False
)

print("\nRobustness outputs saved to:")
print(ROBUSTNESS_FOLDER)
print("\nROBUSTNESS TESTING COMPLETE")
