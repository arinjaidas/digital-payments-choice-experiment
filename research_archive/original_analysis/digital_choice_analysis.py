
"""
Digital Choice Study - Final Analysis Pipeline

INPUT FILES
-----------
form_a_control.csv    -> Form A / no-default condition
form_b_default.csv    -> Form B / UPI-default condition

Run:
    python digital_choice_analysis.py

OUTPUTS
-------
A folder called analysis_outputs containing:
- cleaned master dataset
- descriptives
- treatment/control balance table
- index diagnostics
- 2/4/6-step friction gradient
- main ATE
- LPM models with HC3 robust SE
- moderator interaction models
- Logit robustness
- manipulation-check summary
- figures
"""

from pathlib import Path
import re
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

BASE = Path(".")
A_FILE = BASE / "form_a_control.csv"
B_FILE = BASE / "form_b_default.csv"
OUT = BASE / "analysis_outputs"
OUT.mkdir(exist_ok=True)

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def norm(x):
    x = str(x).lower().replace("₹", "rs")
    x = x.replace("–", "-").replace("—", "-")
    x = re.sub(r"\s+", " ", x)
    return x.strip()

def find_col(df, terms, label):
    matches = []
    for col in df.columns:
        n = norm(col)
        if all(norm(t) in n for t in terms):
            matches.append(col)

    if len(matches) == 1:
        return matches[0]

    if len(matches) == 0:
        raise KeyError(
            f"\nCould not find {label}.\n"
            f"Expected column containing: {terms}\n\n"
            "Available columns:\n" + "\n".join(df.columns.astype(str))
        )

    raise KeyError(
        f"\nMore than one column matched {label}: {matches}\n"
        "Make the matching rule more specific."
    )

def robust_lpm(formula, data):
    return smf.ols(formula, data=data).fit(cov_type="HC3")

def cronbach_alpha(items):
    x = items.dropna()
    k = x.shape[1]
    if k < 2 or len(x) < 2:
        return np.nan
    item_var_sum = x.var(axis=0, ddof=1).sum()
    total_var = x.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return np.nan
    return (k / (k - 1)) * (1 - item_var_sum / total_var)

# ------------------------------------------------------------
# 1. Load raw data
# ------------------------------------------------------------

if not A_FILE.exists() or not B_FILE.exists():
    raise FileNotFoundError(
        "\nExport the two Google Form response sheets as CSV and name them:\n"
        "  form_a_control.csv\n"
        "  form_b_default.csv\n"
    )

a = pd.read_csv(A_FILE)
b = pd.read_csv(B_FILE)

# Assigned treatment, not self-reported treatment.
a["Default"] = 0
b["Default"] = 1
a["Form"] = "A_Control"
b["Form"] = "B_UPI_Default"

df = pd.concat([a, b], ignore_index=True, sort=False)
df.to_csv(OUT / "00_combined_raw.csv", index=False)

# ------------------------------------------------------------
# 2. Match survey questions to exported columns
# ------------------------------------------------------------

Q1 = find_col(df, ["payment methods", "last 6 months"], "Q1 Experience Breadth")
Q2 = find_col(df, ["last 7 days", "digital payments"], "Q2 Frequency")
Q3 = find_col(df, ["5 most recent", "made using upi"], "Q3 Prior UPI")

Q4 = find_col(df, ["bank account a", "bank account b", "next payment"], "Q4 Status Quo")
Q5 = find_col(df, ["1,000", "2 additional steps"], "Q5 Switching Effort 2")
Q6 = find_col(df, ["method a", "suggested based on your recent payment activity"], "Q6 Platform Reliance")

Q7 = find_col(df, ["payment app b", "payment app a", "which would you use"], "Q7 Status Quo")
Q8 = find_col(df, ["1,000", "4 additional steps"], "Q8 Switching Effort 4")
Q9 = find_col(df, ["linked payment accounts", "account b", "suggested"], "Q9 Platform Reliance")

Q10 = find_col(df, ["recurring digital payment", "bank account a", "bank account b"], "Q10 Status Quo")
Q11 = find_col(df, ["1,000", "6 additional steps"], "Q11 Switching Effort 6")
Q12 = find_col(df, ["payment methods appear equally suitable", "method a", "suggested"], "Q12 Platform Reliance")

Q13 = find_col(df, ["10,000", "5% interest", "one year"], "Q13 Interest")
Q14 = find_col(df, ["savings increase by 3%", "prices increase by 5%"], "Q14 Inflation")
Q15 = find_col(df, ["investor a", "investor b", "investment-specific risk"], "Q15 Diversification")

Q16 = find_col(df, ["2,000", "debit card", "upi", "credit card"], "Q16 Outcome")

# Supports either final wording or earlier wording.
try:
    Q17 = find_col(df, ["before you made your choice", "already selected"], "Q17 Manipulation Check")
except KeyError:
    Q17 = find_col(df, ["previous payment question", "already selected"], "Q17 Manipulation Check")

# ------------------------------------------------------------
# 3. Construct variables
# ------------------------------------------------------------

# Q1 Experience Breadth
methods = ["UPI", "Debit Card", "Credit Card", "Mobile Wallet", "Net Banking"]

def count_methods(x):
    nx = norm(x)
    if nx in ("", "nan", "none", "none of these"):
        return 0
    return sum(norm(m) in nx for m in methods)

df["ExperienceBreadth"] = df[Q1].apply(count_methods)

# Q2 Frequency
freq_map = {
    "0": 0,
    "1-2": 1,
    "3-5": 2,
    "6-10": 3,
    "11-20": 4,
    "more than 20": 5,
}
df["FrequencyOrdinal"] = df[Q2].apply(lambda x: freq_map.get(norm(x), np.nan))
df["FrequencyCategory"] = df[Q2].astype(str).str.strip()

# Q3 Prior UPI
df["PriorUPI"] = pd.to_numeric(df[Q3], errors="coerce")

# Q4, Q7, Q10 Status Quo
df["SQ1"] = df[Q4].apply(lambda x: int("continue with bank account a" in norm(x)))
df["SQ2"] = df[Q7].apply(lambda x: int("continue with payment app b" in norm(x)))
df["SQ3"] = df[Q10].apply(lambda x: int("continue with bank account a" in norm(x)))
df["StatusQuo"] = df[["SQ1", "SQ2", "SQ3"]].mean(axis=1)

# Q5, Q8, Q11 Switching Effort
# Effortful/cheaper choice = 1.
df["ES2"] = df[Q5].apply(lambda x: int("980" in norm(x)))
df["ES4"] = df[Q8].apply(lambda x: int("980" in norm(x)))
df["ES6"] = df[Q11].apply(lambda x: int("980" in norm(x)))
df["SwitchingEffort"] = df[["ES2", "ES4", "ES6"]].mean(axis=1)

# Q6, Q9, Q12 Platform Reliance
# Suggested choice = 1.
df["PR1"] = df[Q6].apply(lambda x: int(norm(x) == "method a"))
df["PR2"] = df[Q9].apply(lambda x: int(norm(x) == "account b"))
df["PR3"] = df[Q12].apply(lambda x: int(norm(x) == "method a"))
df["PlatformReliance"] = df[["PR1", "PR2", "PR3"]].mean(axis=1)

# Q13-Q15 Financial Literacy
def interest_correct(x):
    y = norm(x).replace(",", "").replace("rs", "")
    return int("10500" in y)

df["FL_Interest"] = df[Q13].apply(interest_correct)
df["FL_Inflation"] = df[Q14].apply(lambda x: int(norm(x) == "decrease"))
df["FL_Diversification"] = df[Q15].apply(lambda x: int(norm(x) == "investor b"))
df["FinancialLiteracy"] = df[
    ["FL_Interest", "FL_Inflation", "FL_Diversification"]
].sum(axis=1)

# Q16 Dependent Variable
df["UPIChoice"] = df[Q16].apply(lambda x: int(norm(x) == "upi"))
df["PaymentChoice3"] = df[Q16].astype(str).str.strip()

# Q17 Manipulation Check
df["ManipulationCorrect"] = np.where(
    ((df["Default"] == 0) & df[Q17].apply(lambda x: norm(x) == "none")) |
    ((df["Default"] == 1) & df[Q17].apply(lambda x: norm(x) == "upi")),
    1, 0
)

# ------------------------------------------------------------
# 4. Data-quality summary
# ------------------------------------------------------------

quality = pd.DataFrame([{
    "N_Total": len(df),
    "N_Control": int((df["Default"] == 0).sum()),
    "N_Treatment": int((df["Default"] == 1).sum()),
    "ManipulationCorrect_All": df["ManipulationCorrect"].mean(),
    "ManipulationCorrect_Control": df.loc[df["Default"] == 0, "ManipulationCorrect"].mean(),
    "ManipulationCorrect_Treatment": df.loc[df["Default"] == 1, "ManipulationCorrect"].mean(),
}])
quality.to_csv(OUT / "01_data_quality.csv", index=False)

# ------------------------------------------------------------
# 5. Descriptive statistics
# ------------------------------------------------------------

analysis_vars = [
    "UPIChoice",
    "ExperienceBreadth",
    "FrequencyOrdinal",
    "PriorUPI",
    "StatusQuo",
    "SwitchingEffort",
    "PlatformReliance",
    "FinancialLiteracy",
    "ManipulationCorrect",
]

df[analysis_vars].describe().T.to_csv(OUT / "02_descriptive_statistics.csv")

df.groupby("Default")[analysis_vars].agg(
    ["count", "mean", "std"]
).to_csv(OUT / "03_descriptives_by_treatment.csv")

# ------------------------------------------------------------
# 6. Balance checks on pre-treatment variables
# ------------------------------------------------------------

balance_vars = [
    "ExperienceBreadth",
    "FrequencyOrdinal",
    "PriorUPI",
    "StatusQuo",
    "SwitchingEffort",
    "PlatformReliance",
    "FinancialLiteracy",
]

balance_rows = []
for v in balance_vars:
    temp = df[["Default", v]].dropna()
    m = robust_lpm(f"{v} ~ Default", temp)
    balance_rows.append({
        "Variable": v,
        "Control_Mean": temp.loc[temp["Default"] == 0, v].mean(),
        "Treatment_Mean": temp.loc[temp["Default"] == 1, v].mean(),
        "Difference_T_minus_C": m.params["Default"],
        "Robust_SE": m.bse["Default"],
        "p_value": m.pvalues["Default"],
        "N": int(m.nobs),
    })

pd.DataFrame(balance_rows).to_csv(OUT / "04_balance_checks.csv", index=False)

# ------------------------------------------------------------
# 7. Index diagnostics
# ------------------------------------------------------------

pd.DataFrame({
    "Index": ["StatusQuo", "SwitchingEffort", "PlatformReliance"],
    "Cronbach_alpha": [
        cronbach_alpha(df[["SQ1", "SQ2", "SQ3"]]),
        cronbach_alpha(df[["ES2", "ES4", "ES6"]]),
        cronbach_alpha(df[["PR1", "PR2", "PR3"]]),
    ]
}).to_csv(OUT / "05_index_reliability.csv", index=False)

df[["SQ1", "SQ2", "SQ3"]].corr().to_csv(OUT / "06_statusquo_correlations.csv")
df[["ES2", "ES4", "ES6"]].corr().to_csv(OUT / "07_switching_correlations.csv")
df[["PR1", "PR2", "PR3"]].corr().to_csv(OUT / "08_platform_correlations.csv")

# ------------------------------------------------------------
# 8. 2/4/6-step friction gradient
# ------------------------------------------------------------

friction = pd.DataFrame({
    "AdditionalSteps": [2, 4, 6],
    "ShareChoosingRs980": [
        df["ES2"].mean(),
        df["ES4"].mean(),
        df["ES6"].mean(),
    ],
})
friction.to_csv(OUT / "09_friction_gradient.csv", index=False)

plt.figure(figsize=(6, 4))
plt.plot(friction["AdditionalSteps"], friction["ShareChoosingRs980"], marker="o")
plt.xlabel("Additional steps")
plt.ylabel("Share choosing ₹980 option")
plt.ylim(0, 1)
plt.title("Switching-friction gradient")
plt.tight_layout()
plt.savefig(OUT / "fig_01_friction_gradient.png", dpi=220)
plt.close()

# ------------------------------------------------------------
# 9. Raw treatment effect
# ------------------------------------------------------------

rates = df.groupby("Default")["UPIChoice"].agg(["count", "mean"])
rates.index = ["Control_No_Default", "Treatment_UPI_Default"]
rates.to_csv(OUT / "10_upi_choice_rates.csv")

control_rate = df.loc[df["Default"] == 0, "UPIChoice"].mean()
treatment_rate = df.loc[df["Default"] == 1, "UPIChoice"].mean()
raw_ate = treatment_rate - control_rate

pd.DataFrame([{
    "Control_UPI_Rate": control_rate,
    "Treatment_UPI_Rate": treatment_rate,
    "ATE_Probability": raw_ate,
    "ATE_Percentage_Points": raw_ate * 100,
}]).to_csv(OUT / "11_raw_ATE.csv", index=False)

plt.figure(figsize=(5, 4))
plt.bar(["No Default", "UPI Default"], [control_rate, treatment_rate])
plt.ylabel("Share choosing UPI")
plt.ylim(0, 1)
plt.title("UPI choice by experimental condition")
plt.tight_layout()
plt.savefig(OUT / "fig_02_main_default_effect.png", dpi=220)
plt.close()

# ------------------------------------------------------------
# 10. Main LPM and adjusted LPM
# ------------------------------------------------------------

m1 = robust_lpm("UPIChoice ~ Default", df)

m2 = robust_lpm(
    "UPIChoice ~ Default + ExperienceBreadth + C(FrequencyCategory) + "
    "PriorUPI + StatusQuo + SwitchingEffort + PlatformReliance + "
    "FinancialLiteracy",
    df
)

def model_long(model, model_name):
    ci = model.conf_int()
    return pd.DataFrame({
        "Model": model_name,
        "Variable": model.params.index,
        "Coefficient": model.params.values,
        "Robust_SE": model.bse.values,
        "p_value": model.pvalues.values,
        "CI95_Lower": ci.iloc[:, 0].values,
        "CI95_Upper": ci.iloc[:, 1].values,
        "N": int(model.nobs),
        "R2": getattr(model, "rsquared", np.nan),
    })

pd.concat([
    model_long(m1, "M1 Main ATE"),
    model_long(m2, "M2 Adjusted"),
], ignore_index=True).to_csv(OUT / "12_main_regressions.csv", index=False)

with open(OUT / "12_main_regressions.txt", "w", encoding="utf-8") as f:
    f.write("MODEL 1: MAIN RANDOMIZED DEFAULT EFFECT\n")
    f.write(m1.summary().as_text())
    f.write("\n\nMODEL 2: ADJUSTED MODEL\n")
    f.write(m2.summary().as_text())

# ------------------------------------------------------------
# 11. Heterogeneity / moderator models
# ------------------------------------------------------------

moderators = [
    "StatusQuo",
    "SwitchingEffort",
    "PlatformReliance",
    "FinancialLiteracy",
]

interaction_tables = []

for z in moderators:
    df[z + "_c"] = df[z] - df[z].mean()
    model = robust_lpm(
        f"UPIChoice ~ Default * {z}_c + ExperienceBreadth + "
        f"C(FrequencyCategory) + PriorUPI",
        df
    )
    interaction_tables.append(model_long(model, f"Default x {z}"))

pd.concat(interaction_tables, ignore_index=True).to_csv(
    OUT / "13_interaction_regressions.csv", index=False
)

# ------------------------------------------------------------
# 12. Logit robustness
# ------------------------------------------------------------

try:
    logit = smf.logit(
        "UPIChoice ~ Default + ExperienceBreadth + C(FrequencyCategory) + "
        "PriorUPI + StatusQuo + SwitchingEffort + PlatformReliance + "
        "FinancialLiteracy",
        data=df
    ).fit(disp=False)

    with open(OUT / "14_logit_model.txt", "w", encoding="utf-8") as f:
        f.write(logit.summary().as_text())

    logit.get_margeff(at="overall").summary_frame().to_csv(
        OUT / "15_logit_average_marginal_effects.csv"
    )
except Exception as e:
    (OUT / "14_logit_ERROR.txt").write_text(str(e), encoding="utf-8")

# ------------------------------------------------------------
# 13. Three-category choice, secondary
# ------------------------------------------------------------

pd.crosstab(
    df["Form"],
    df["PaymentChoice3"],
    normalize="index"
).to_csv(OUT / "16_three_category_choice.csv")

# ------------------------------------------------------------
# 14. Manipulation check
# ------------------------------------------------------------

df.groupby("Form")["ManipulationCorrect"].agg(
    ["count", "mean"]
).to_csv(OUT / "17_manipulation_check.csv")

# Do NOT exclude manipulation-check failures from the primary ITT model.

# ------------------------------------------------------------
# 15. Save final clean dataset
# ------------------------------------------------------------

df.to_csv(OUT / "18_clean_master_dataset.csv", index=False)

# ------------------------------------------------------------
# Console summary
# ------------------------------------------------------------

ci = m1.conf_int().loc["Default"]

print("=" * 70)
print("DIGITAL CHOICE STUDY")
print("=" * 70)
print(f"N total     : {len(df)}")
print(f"N control   : {(df.Default == 0).sum()}")
print(f"N treatment : {(df.Default == 1).sum()}")
print()
print(f"Control UPI rate   : {control_rate:.3f}")
print(f"Treatment UPI rate : {treatment_rate:.3f}")
print(f"Raw ATE             : {100*raw_ate:.2f} percentage points")
print()
print("PRIMARY LPM")
print(f"Default coefficient : {m1.params['Default']:.4f}")
print(f"Robust SE           : {m1.bse['Default']:.4f}")
print(f"p-value             : {m1.pvalues['Default']:.4f}")
print(f"95% CI              : [{ci.iloc[0]:.4f}, {ci.iloc[1]:.4f}]")
print()
print(f"Outputs: {OUT.resolve()}")
print("=" * 70)
