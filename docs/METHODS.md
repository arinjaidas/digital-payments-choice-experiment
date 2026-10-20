# Methods

## Outcome and covariates

The example's binary outcome is UPIChoice. Default indicates UPI preselection; PriorUPI records prior use. SwitchingEffort and PlatformReliance are numeric covariates. The synthetic generator creates 800 fictional records. Its mechanism is illustrative and is not estimated from the original participant records.

## Three complementary descriptions

Group shares report an unadjusted difference. The linear probability model relates the binary outcome to covariates with an additive conditional mean and HC3 uncertainty. The logistic model constrains predicted probabilities through the logistic link. A logit coefficient is in log-odds units, so it should not be read directly as a probability-point change.

For each observation, the code creates otherwise identical design rows with Default set to one and zero. It averages `p(Default=1, X_i) - p(Default=0, X_i)`. This is an average **discrete change**, appropriate for a binary indicator. It is not the derivative approximation p(1-p)β evaluated at a single representative person.

## Identification

The code demonstrates conditional comparisons. Original-study causal claims would require documented assignment, treatment integrity, measurement, attrition and sampling assumptions. The public demonstration does not establish that the original survey implemented a fully randomised factorial design.
