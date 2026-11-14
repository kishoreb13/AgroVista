# AgroVista methodology

## Agricultural Performance Index
A transparent composite score on a 0–100 scale:
- Yield: 30%
- Profit: 25%
- Water efficiency: 20%
- Inverse disease/pest risk: 25%

Each component is clipped to its 5th and 95th percentile and min-max scaled. The risk component is inverted so lower risk contributes a higher score. These weights are proposed for this project and require domain validation.

## Statistical testing
One-way ANOVA compares seasonal yield means. Before interpreting results, assess independence, residual normality and variance homogeneity. If significant, conduct a suitable post-hoc comparison and adjust for multiple testing.

## Machine learning
The current baseline compares Linear Regression, Random Forest and Gradient Boosting using a single 80/20 holdout. MAE, RMSE and R² are reported. Permutation importance is computed on held-out data. For a thesis-quality evaluation, add repeated cross-validation, residual plots, confidence intervals, and a final untouched test set.

## Data integrity
Demo records are synthetic. Real-data findings must only be reported after importing the actual dataset and validating the data dictionary, units, missingness, duplicates, outliers and feature semantics.
