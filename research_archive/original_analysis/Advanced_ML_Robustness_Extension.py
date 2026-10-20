# ================================================================
# ADVANCED / MACHINE-LEARNING ROBUSTNESS EXTENSION
# Digital Choice Study  (Default = UPI pre-selection)
#
# PASTE THIS AFTER your existing "ROBUSTNESS AND DIAGNOSTIC TESTING
# SUITE" (i.e. after R23). It assumes df, OUTPUT_FOLDER already exist,
# but also runs standalone if you point CLEAN_DATA_PATH at
# 14_clean_master_dataset.csv.
#
# WHAT THIS DOES AND DOES NOT DO
# --------------------------------
# It does NOT search across models/specifications for the one that
# looks best and report only that. Every method below is applied
# once, in full, and every specification tried is reported --
# including the ones that come out small, noisy, or null.
#
# It adds four things your original script doesn't have:
#   (A) Specification-curve analysis: ALL reasonable control-set
#       combinations, run and reported together, plus a randomization
#       (permutation) test on the whole curve -- this is a transparency
#       tool, the opposite of cherry-picking one specification.
#   (B) A doubly-robust / AIPW treatment-effect estimator with
#       cross-fitted machine-learning nuisance models (random forests),
#       as a non-linear-functional-form robustness check on the LPM.
#   (C) Honest (cross-fitted, sample-split) heterogeneous-treatment-
#       effect exploration via the transformed-outcome + random-forest
#       approach, reported WITH its out-of-fold R^2 so you can see
#       whether the data can actually support individual-level effect
#       estimates (at N=183, mostly they cannot -- that finding itself
#       is reported honestly rather than papered over).
#   (D) A predictive-ML benchmark (logistic vs random forest vs
#       gradient boosting) purely to see how well Default and the
#       controls predict choice out-of-sample, plus permutation
#       feature importance.
#   (E) A Bayesian (Beta-Binomial) estimate of the raw ATE as an
#       inferential cross-check independent of asymptotic p-values.
#
# None of this estimates a "better" or "bigger" treatment effect on
# purpose. If anything, (A)-(C) are specifically designed to show you
# the full distribution of plausible estimates so a reader can see
# nothing was hidden.
# ================================================================

import warnings
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
)
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

RNG_SEED = 20260905

# ----------------------------------------------------------------
# 0. LOAD DATA (standalone-safe: uses df if it already exists)
# ----------------------------------------------------------------

try:
    df  # noqa: F821  (defined already if pasted after main script)
    ML_OUTPUT_FOLDER = Path(OUTPUT_FOLDER) / "Advanced_ML_Extension"  # noqa: F821
except NameError:
    CLEAN_DATA_PATH = r"14_clean_master_dataset.csv"
    df = pd.read_csv(CLEAN_DATA_PATH)
    ML_OUTPUT_FOLDER = Path(".") / "Advanced_ML_Extension"

ML_OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

Y = df["UPIChoice"].astype(float).to_numpy()
D = df["Default"].astype(float).to_numpy()
N = len(df)

BASELINE_CONTROLS = [
    "ExperienceBreadth",
    "PriorUPI",
    "StatusQuo",
    "SwitchingEffort",
    "PlatformReliance",
    "FinancialLiteracy",
]

freq_dummies = pd.get_dummies(
    df["FrequencyCategory"], prefix="Freq", drop_first=True
).astype(float)

print("\n" + "=" * 70)
print("ADVANCED / ML ROBUSTNESS EXTENSION")
print("=" * 70)
print(f"N = {N}   Control N = {int((D==0).sum())}   Treatment N = {int((D==1).sum())}")


# ================================================================
# HELPER: OLS with HC3 robust SEs (matches statsmodels cov_type='HC3')
# ================================================================

def ols_hc3(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    H_diag = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
    h = np.clip(H_diag, 0, 1 - 1e-10)
    weights = (resid ** 2) / (1 - h) ** 2
    meat = (X * weights[:, None]).T @ X
    cov = XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    z = beta / se
    p = 2 * (1 - stats.norm.cdf(np.abs(z)))
    return beta, se, p


def build_design(data, controls, add_freq):
    cols = [np.ones(len(data)), data["Default"].astype(float).to_numpy()]
    names = ["Intercept", "Default"]
    for c in controls:
        cols.append(data[c].astype(float).to_numpy())
        names.append(c)
    if add_freq:
        fd = pd.get_dummies(data["FrequencyCategory"], prefix="Freq", drop_first=True).astype(float)
        for c in fd.columns:
            cols.append(fd[c].to_numpy())
            names.append(c)
    return np.column_stack(cols), names


# ================================================================
# (A) SPECIFICATION-CURVE ANALYSIS
# ================================================================

print("\n" + "-" * 70)
print("(A) SPECIFICATION CURVE: every control-set combination")
print("-" * 70)

spec_rows = []
for r in range(0, len(BASELINE_CONTROLS) + 1):
    for combo in combinations(BASELINE_CONTROLS, r):
        for add_freq in [False, True]:
            X, names = build_design(df, combo, add_freq)
            beta, se, p = ols_hc3(X, Y)
            idx = names.index("Default")
            spec_rows.append(
                {
                    "n_controls": r,
                    "controls": ",".join(combo) if combo else "(none)",
                    "frequency_dummies_included": add_freq,
                    "coef": beta[idx],
                    "se": se[idx],
                    "p_value": p[idx],
                }
            )

spec_df = pd.DataFrame(spec_rows).sort_values("coef").reset_index(drop=True)
spec_df["rank"] = np.arange(1, len(spec_df) + 1)
spec_df["significant_5pct"] = spec_df["p_value"] < 0.05

n_specs = len(spec_df)
share_sig = spec_df["significant_5pct"].mean()
share_pos = (spec_df["coef"] > 0).mean()
median_coef = spec_df["coef"].median()

print(f"Number of specifications run: {n_specs}")
print(f"Share positive:                {share_pos:.3f}")
print(f"Share significant at 5%:       {share_sig:.3f}")
print(f"Median Default coefficient:    {median_coef:.4f}")
print(f"Range:                         [{spec_df['coef'].min():.4f}, {spec_df['coef'].max():.4f}]")

spec_df.to_csv(ML_OUTPUT_FOLDER / "A1_specification_curve.csv", index=False)

# Randomization test on the WHOLE curve: is the median specification-curve
# estimate bigger than we'd see if Default were pure noise? Every permutation
# re-runs the ENTIRE curve (all specs), so this cannot be gamed toward one
# favorable specification.
N_PERM = 1000
rng = np.random.default_rng(RNG_SEED)
perm_medians = np.empty(N_PERM)

for b in range(N_PERM):
    D_perm = rng.permutation(D)
    df_perm = df.copy()
    df_perm["Default"] = D_perm
    coefs = []
    for r in range(0, len(BASELINE_CONTROLS) + 1):
        for combo in combinations(BASELINE_CONTROLS, r):
            for add_freq in [False, True]:
                X, names = build_design(df_perm, combo, add_freq)
                beta, _, _ = ols_hc3(X, Y)
                coefs.append(beta[names.index("Default")])
    perm_medians[b] = np.median(coefs)

spec_perm_p = (np.sum(np.abs(perm_medians) >= abs(median_coef)) + 1) / (N_PERM + 1)

print(f"\nSpecification-curve randomization test:")
print(f"  Observed median coefficient across {n_specs} specs: {median_coef:.4f}")
print(f"  Two-sided permutation p-value (whole-curve null):    {spec_perm_p:.4f}")

spec_curve_summary = pd.DataFrame(
    [
        {
            "n_specifications": n_specs,
            "share_positive": share_pos,
            "share_significant_5pct": share_sig,
            "median_coef": median_coef,
            "min_coef": spec_df["coef"].min(),
            "max_coef": spec_df["coef"].max(),
            "permutations": N_PERM,
            "whole_curve_permutation_p": spec_perm_p,
        }
    ]
)
spec_curve_summary.to_csv(ML_OUTPUT_FOLDER / "A2_specification_curve_summary.csv", index=False)

# Specification curve plot
fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                          gridspec_kw={"height_ratios": [2, 1]})

colors = np.where(spec_df["significant_5pct"], "#1f77b4", "#c7c7c7")
axes[0].scatter(spec_df["rank"], spec_df["coef"], c=colors, s=10)
axes[0].axhline(0, color="black", linewidth=0.8)
axes[0].set_ylabel("Default coefficient\n(pp change in UPI choice)")
axes[0].set_title("Specification curve: Default effect across all control-set choices")

axes[1].scatter(spec_df["rank"], spec_df["n_controls"], c=colors, s=10)
axes[1].set_ylabel("# baseline\ncontrols included")
axes[1].set_xlabel("Specifications, sorted by estimated effect")

plt.tight_layout()
plt.savefig(ML_OUTPUT_FOLDER / "A3_specification_curve_plot.png", dpi=200)
plt.close()


# ================================================================
# (B) DOUBLY-ROBUST / AIPW TREATMENT EFFECT (cross-fitted ML nuisance)
# ================================================================

print("\n" + "-" * 70)
print("(B) DOUBLY-ROBUST (AIPW) ATE with cross-fitted random-forest nuisance models")
print("-" * 70)


def aipw_ate(data, Y, D, covariates, n_splits=5, seed=RNG_SEED):
    X = data[covariates].astype(float).to_numpy()
    n = len(Y)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    ps = np.zeros(n)
    mu1 = np.zeros(n)
    mu0 = np.zeros(n)

    for train_idx, test_idx in skf.split(X, D):
        Xtr, Dtr, Ytr = X[train_idx], D[train_idx], Y[train_idx]

        prop_model = LogisticRegression(max_iter=2000).fit(Xtr, Dtr)
        ps[test_idx] = prop_model.predict_proba(X[test_idx])[:, 1]

        idx1, idx0 = Dtr == 1, Dtr == 0

        m1 = RandomForestClassifier(
            n_estimators=400, max_depth=4, min_samples_leaf=8, random_state=seed
        ).fit(Xtr[idx1], Ytr[idx1])
        m0 = RandomForestClassifier(
            n_estimators=400, max_depth=4, min_samples_leaf=8, random_state=seed
        ).fit(Xtr[idx0], Ytr[idx0])

        mu1[test_idx] = m1.predict_proba(X[test_idx])[:, 1]
        mu0[test_idx] = m0.predict_proba(X[test_idx])[:, 1]

    ps_clipped = np.clip(ps, 0.05, 0.95)
    aipw_scores = (
        (mu1 - mu0)
        + D * (Y - mu1) / ps_clipped
        - (1 - D) * (Y - mu0) / (1 - ps_clipped)
    )
    ate = aipw_scores.mean()
    se = aipw_scores.std(ddof=1) / np.sqrt(n)
    return ate, se, ps, mu1, mu0, aipw_scores


aipw_ate_est, aipw_se, ps_hat, mu1_hat, mu0_hat, aipw_scores = aipw_ate(
    df, Y, D, BASELINE_CONTROLS
)
z_aipw = aipw_ate_est / aipw_se
p_aipw = 2 * (1 - stats.norm.cdf(abs(z_aipw)))
ci_aipw = (aipw_ate_est - 1.96 * aipw_se, aipw_ate_est + 1.96 * aipw_se)

print(f"Cross-fitted estimated propensity range: [{ps_hat.min():.3f}, {ps_hat.max():.3f}] "
      f"(design propensity ~ {D.mean():.3f}; a flat range near that value is what we expect under randomization)")
print(f"AIPW ATE:            {aipw_ate_est:.4f}")
print(f"SE:                  {aipw_se:.4f}")
print(f"z, p-value:          {z_aipw:.3f}, {p_aipw:.4f}")
print(f"95% CI:              [{ci_aipw[0]:.4f}, {ci_aipw[1]:.4f}]")

aipw_table = pd.DataFrame(
    [
        {
            "Method": "Doubly-robust AIPW (cross-fitted RF nuisance)",
            "ATE": aipw_ate_est,
            "SE": aipw_se,
            "z": z_aipw,
            "p_value": p_aipw,
            "CI_2.5": ci_aipw[0],
            "CI_97.5": ci_aipw[1],
            "N": N,
        }
    ]
)
aipw_table.to_csv(ML_OUTPUT_FOLDER / "B1_aipw_doubly_robust_ate.csv", index=False)


# ================================================================
# (C) HONEST HETEROGENEOUS TREATMENT EFFECTS (transformed outcome + RF)
# ================================================================

print("\n" + "-" * 70)
print("(C) EXPLORATORY HETEROGENEITY: transformed-outcome random forest")
print("    (reported WITH out-of-fold R^2, not just point CATEs)")
print("-" * 70)


def transformed_outcome_cate(data, Y, D, covariates, n_splits=5, seed=RNG_SEED):
    X = data[covariates].astype(float).to_numpy()
    n = len(Y)

    # cross-fitted propensity (design is randomized, but we estimate it
    # to keep the transformed outcome well-behaved / doubly-robust in spirit)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    ps = np.zeros(n)
    for train_idx, test_idx in skf.split(X, D):
        prop_model = LogisticRegression(max_iter=2000).fit(X[train_idx], D[train_idx])
        ps[test_idx] = prop_model.predict_proba(X[test_idx])[:, 1]
    ps = np.clip(ps, 0.1, 0.9)

    Ystar = Y * (D - ps) / (ps * (1 - ps))

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_pred = np.zeros(n)
    fold_r2 = []

    for train_idx, test_idx in kf.split(X):
        rf = RandomForestRegressor(
            n_estimators=500, max_depth=3, min_samples_leaf=15, random_state=seed
        ).fit(X[train_idx], Ystar[train_idx])
        pred = rf.predict(X[test_idx])
        oof_pred[test_idx] = pred

        baseline_pred = np.full(len(test_idx), Ystar[train_idx].mean())
        ss_res = np.sum((Ystar[test_idx] - pred) ** 2)
        ss_tot = np.sum((Ystar[test_idx] - baseline_pred) ** 2)
        fold_r2.append(1 - ss_res / ss_tot if ss_tot > 0 else np.nan)

    return oof_pred, np.array(fold_r2), ps, Ystar


cate_oof, cate_fold_r2, ps_c, ystar = transformed_outcome_cate(
    df, Y, D, BASELINE_CONTROLS
)

print(f"Out-of-fold R^2 by fold: {np.round(cate_fold_r2, 3).tolist()}")
print(f"Mean out-of-fold R^2:    {cate_fold_r2.mean():.4f}")
if cate_fold_r2.mean() <= 0.02:
    print(
        "Interpretation: the model explains essentially none of the "
        "individual-level variation in treatment response out-of-sample. "
        "At N=183 the data do not support reliable individual-level CATE "
        "estimates -- this matches the weak/inconsistent linear-interaction "
        "results already in the main script and should be reported as a "
        "genuine limitation, not pushed past."
    )

# full-sample fit (in-sample only) purely to describe which variables the
# model leans on -- descriptive, NOT causal, NOT for individual predictions
rf_full = RandomForestRegressor(
    n_estimators=500, max_depth=3, min_samples_leaf=15, random_state=RNG_SEED
).fit(df[BASELINE_CONTROLS].astype(float), ystar)

perm_imp = permutation_importance(
    rf_full, df[BASELINE_CONTROLS].astype(float), ystar,
    n_repeats=100, random_state=RNG_SEED
)

cate_importance = pd.DataFrame(
    {
        "Variable": BASELINE_CONTROLS,
        "Permutation_Importance_Mean": perm_imp.importances_mean,
        "Permutation_Importance_SD": perm_imp.importances_std,
    }
).sort_values("Permutation_Importance_Mean", ascending=False)

print("\nDescriptive variable importance for treatment-effect heterogeneity")
print("(in-sample, exploratory only):")
print(cate_importance.round(4).to_string(index=False))

cate_summary = pd.DataFrame(
    [{"Fold": i + 1, "OOF_R2": r} for i, r in enumerate(cate_fold_r2)]
    + [{"Fold": "Mean", "OOF_R2": cate_fold_r2.mean()}]
)
cate_summary.to_csv(ML_OUTPUT_FOLDER / "C1_cate_oof_r2.csv", index=False)
cate_importance.to_csv(ML_OUTPUT_FOLDER / "C2_cate_variable_importance.csv", index=False)

pd.DataFrame({"CATE_oof": cate_oof, "Default": D, "UPIChoice": Y}).to_csv(
    ML_OUTPUT_FOLDER / "C3_individual_oof_cate_estimates.csv", index=False
)


# ================================================================
# (D) PREDICTIVE ML BENCHMARK (cross-validated, out-of-sample)
# ================================================================

print("\n" + "-" * 70)
print("(D) PREDICTIVE BENCHMARK: Logistic vs Random Forest vs Gradient Boosting")
print("    (out-of-sample AUC / accuracy / Brier score, NOT a causal estimate)")
print("-" * 70)

feature_cols = ["Default"] + BASELINE_CONTROLS
X_full = df[feature_cols].astype(float).to_numpy()

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Random Forest": RandomForestClassifier(
        n_estimators=400, max_depth=4, min_samples_leaf=6, random_state=RNG_SEED
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, max_depth=2, learning_rate=0.05, random_state=RNG_SEED
    ),
}

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=RNG_SEED)

bench_rows = []
for name, model in models.items():
    aucs, accs, briers = [], [], []
    for train_idx, test_idx in rskf.split(X_full, Y):
        m = model.__class__(**model.get_params())
        m.fit(X_full[train_idx], Y[train_idx])
        proba = m.predict_proba(X_full[test_idx])[:, 1]
        pred = (proba >= 0.5).astype(int)
        aucs.append(roc_auc_score(Y[test_idx], proba))
        accs.append(accuracy_score(Y[test_idx], pred))
        briers.append(brier_score_loss(Y[test_idx], proba))
    bench_rows.append(
        {
            "Model": name,
            "Mean_AUC": np.mean(aucs),
            "SD_AUC": np.std(aucs),
            "Mean_Accuracy": np.mean(accs),
            "Mean_Brier": np.mean(briers),
            "N_folds_x_repeats": len(aucs),
        }
    )

# naive baseline: always predict the majority class
majority_share = max(Y.mean(), 1 - Y.mean())
bench_rows.append(
    {
        "Model": "Naive baseline (predict majority class)",
        "Mean_AUC": 0.5,
        "SD_AUC": np.nan,
        "Mean_Accuracy": majority_share,
        "Mean_Brier": np.nan,
        "N_folds_x_repeats": np.nan,
    }
)

bench_table = pd.DataFrame(bench_rows)
print(bench_table.round(4).to_string(index=False))
bench_table.to_csv(ML_OUTPUT_FOLDER / "D1_predictive_benchmark.csv", index=False)

# permutation importance from the logistic model (most interpretable / most
# appropriate given N=183) fit on the full sample, for a plain-language
# ranking of which features carry predictive weight
logit_full = LogisticRegression(max_iter=2000).fit(X_full, Y)
perm_imp_logit = permutation_importance(
    logit_full, X_full, Y, n_repeats=200, random_state=RNG_SEED, scoring="roc_auc"
)
logit_importance = pd.DataFrame(
    {
        "Feature": feature_cols,
        "Permutation_Importance_Mean_AUC_drop": perm_imp_logit.importances_mean,
        "Permutation_Importance_SD": perm_imp_logit.importances_std,
    }
).sort_values("Permutation_Importance_Mean_AUC_drop", ascending=False)

print("\nPermutation feature importance (drop in AUC when shuffled), logistic model:")
print(logit_importance.round(4).to_string(index=False))
logit_importance.to_csv(ML_OUTPUT_FOLDER / "D2_permutation_feature_importance.csv", index=False)


# ================================================================
# (E) BAYESIAN (BETA-BINOMIAL) ESTIMATE OF THE RAW ATE
# ================================================================

print("\n" + "-" * 70)
print("(E) BAYESIAN ESTIMATE OF THE RAW DEFAULT EFFECT (Beta-Binomial)")
print("-" * 70)

a0, b0 = 1.0, 1.0  # uniform prior on each group's UPI-choice probability

nc = int((D == 0).sum())
yc = int(Y[D == 0].sum())
nt = int((D == 1).sum())
yt = int(Y[D == 1].sum())

post_c = stats.beta(a0 + yc, b0 + nc - yc)
post_t = stats.beta(a0 + yt, b0 + nt - yt)

rng_bayes = np.random.default_rng(RNG_SEED)
draws_c = post_c.rvs(300_000, random_state=rng_bayes)
draws_t = post_t.rvs(300_000, random_state=rng_bayes)
draws_diff = draws_t - draws_c

bayes_mean = draws_diff.mean()
bayes_ci = np.percentile(draws_diff, [2.5, 97.5])
prob_positive = (draws_diff > 0).mean()

print(f"Posterior mean ATE:                 {bayes_mean:.4f}")
print(f"95% credible interval:              [{bayes_ci[0]:.4f}, {bayes_ci[1]:.4f}]")
print(f"P(treatment > control | data):      {prob_positive:.4f}")

bayes_table = pd.DataFrame(
    [
        {
            "Method": "Bayesian Beta-Binomial (uniform prior)",
            "Posterior_Mean_ATE": bayes_mean,
            "CI_2.5": bayes_ci[0],
            "CI_97.5": bayes_ci[1],
            "P_treatment_greater": prob_positive,
            "N_control": nc,
            "N_treatment": nt,
        }
    ]
)
bayes_table.to_csv(ML_OUTPUT_FOLDER / "E1_bayesian_ate.csv", index=False)


# ================================================================
# FINAL: CROSS-METHOD ATE COMPARISON TABLE
# ================================================================

print("\n" + "=" * 70)
print("CROSS-METHOD SUMMARY: ATE ESTIMATES SIDE BY SIDE")
print("=" * 70)

raw_ate = Y[D == 1].mean() - Y[D == 0].mean()

# recompute LPM (no controls) and LPM (full controls) with the same
# ols_hc3 helper, purely so every row in this table comes from one
# consistent estimator implementation
X_nocontrol, names_nc = build_design(df, [], add_freq=False)
beta_nc, se_nc, p_nc = ols_hc3(X_nocontrol, Y)
idx_nc = names_nc.index("Default")

X_full_ctrl, names_fc = build_design(df, BASELINE_CONTROLS, add_freq=True)
beta_fc, se_fc, p_fc = ols_hc3(X_full_ctrl, Y)
idx_fc = names_fc.index("Default")

final_summary = pd.DataFrame(
    [
        {"Method": "Raw difference in means", "Estimate": raw_ate, "SE_or_width": np.nan,
         "p_or_credibility": np.nan},
        {"Method": "LPM, no controls (HC3)", "Estimate": beta_nc[idx_nc], "SE_or_width": se_nc[idx_nc],
         "p_or_credibility": p_nc[idx_nc]},
        {"Method": "LPM, full controls (HC3)", "Estimate": beta_fc[idx_fc], "SE_or_width": se_fc[idx_fc],
         "p_or_credibility": p_fc[idx_fc]},
        {"Method": "Specification curve, median of all specs", "Estimate": median_coef,
         "SE_or_width": np.nan, "p_or_credibility": spec_perm_p},
        {"Method": "Doubly-robust AIPW (cross-fitted RF)", "Estimate": aipw_ate_est,
         "SE_or_width": aipw_se, "p_or_credibility": p_aipw},
        {"Method": "Bayesian posterior mean (Beta-Binomial)", "Estimate": bayes_mean,
         "SE_or_width": (bayes_ci[1] - bayes_ci[0]) / 2, "p_or_credibility": 1 - prob_positive},
    ]
)

print(final_summary.round(4).to_string(index=False))
final_summary.to_csv(ML_OUTPUT_FOLDER / "F1_cross_method_ate_summary.csv", index=False)

print("\nAll advanced/ML extension outputs saved to:")
print(ML_OUTPUT_FOLDER)
print("\nADVANCED ML EXTENSION COMPLETE")
