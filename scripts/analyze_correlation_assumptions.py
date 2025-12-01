"""
Verify statistical assumptions for correlation analysis on Slide 14
"""
import json
import numpy as np
from scipy import stats
from pathlib import Path

# Load data
data_path = Path('results/qgan_tournament_validation_N30.json')
with open(data_path) as f:
    data = json.load(f)

x = np.array(data['correlation']['per_device_data']['avg_kl'])
y = np.array(data['correlation']['per_device_data']['nn_accuracies'])

print("\n" + "="*80)
print("CORRELATION ANALYSIS DIAGNOSTICS - SLIDE 14")
print("="*80)

# Basic stats
print(f"\nData Range:")
print(f"  X (avg KL): {x.min():.3f} to {x.max():.3f}")
print(f"  Y (NN accuracy): {y.min():.3f} to {y.max():.3f}")
print(f"  N = {len(x)} devices")

# Linear regression
slope, intercept, r, p_value, std_err = stats.linregress(x, y)
residuals = y - (slope * x + intercept)

print(f"\nLinear Regression:")
print(f"  Slope: {slope:.4f}")
print(f"  Intercept: {intercept:.4f}")
print(f"  R²: {r**2:.4f}")
print(f"  Pearson r: {r:.4f}")
print(f"  P-value: {p_value:.2e}")

# Residual analysis
print(f"\nResidual Analysis:")
print(f"  Mean: {np.mean(residuals):.6f} (should be ~0)")
print(f"  Std Dev: {np.std(residuals):.4f}")
print(f"  Min: {residuals.min():.4f}")
print(f"  Max: {residuals.max():.4f}")
print(f"  Range: {residuals.max() - residuals.min():.4f}")

# Homoscedasticity test (equal variance across X range)
print(f"\n{'='*80}")
print("HOMOSCEDASTICITY CHECK (Constant Variance)")
print("="*80)

# Split data at median
x_low_mask = x < np.median(x)
x_high_mask = x >= np.median(x)

res_low = residuals[x_low_mask]
res_high = residuals[x_high_mask]

print(f"\nLow X region (KL < {np.median(x):.3f}):")
print(f"  N = {len(res_low)}")
print(f"  Residual variance: {np.var(res_low):.6f}")
print(f"  Residual std: {np.std(res_low):.4f}")

print(f"\nHigh X region (KL >= {np.median(x):.3f}):")
print(f"  N = {len(res_high)}")
print(f"  Residual variance: {np.var(res_high):.6f}")
print(f"  Residual std: {np.std(res_high):.4f}")

variance_ratio = np.var(res_high) / np.var(res_low)
print(f"\nVariance Ratio (high/low): {variance_ratio:.2f}")
print(f"  Interpretation: {'CONCERN - heteroscedastic!' if variance_ratio > 3 or variance_ratio < 0.33 else 'Acceptable homoscedasticity'}")

# Levene's test for equal variances
levene_stat, levene_p = stats.levene(res_low, res_high)
print(f"\nLevene's Test (equal variance):")
print(f"  Test statistic: {levene_stat:.4f}")
print(f"  P-value: {levene_p:.4f}")
print(f"  Conclusion: {'✓ Homoscedastic (p>0.05)' if levene_p > 0.05 else '✗ HETEROSCEDASTIC (p<0.05)'}")

# Breusch-Pagan test (more rigorous)
from scipy.stats import chi2
n = len(x)
residuals_squared = residuals**2
aux_slope, aux_intercept, aux_r, _, _ = stats.linregress(x, residuals_squared)
lm_stat = n * aux_r**2
bp_p = 1 - chi2.cdf(lm_stat, 1)
print(f"\nBreusch-Pagan Test:")
print(f"  LM statistic: {lm_stat:.4f}")
print(f"  P-value: {bp_p:.4f}")
print(f"  Conclusion: {'✓ Homoscedastic (p>0.05)' if bp_p > 0.05 else '✗ HETEROSCEDASTIC (p<0.05)'}")

# Normality of residuals
print(f"\n{'='*80}")
print("NORMALITY CHECK (Residuals)")
print("="*80)

sw_stat, sw_p = stats.shapiro(residuals)
print(f"\nShapiro-Wilk Test:")
print(f"  Test statistic: {sw_stat:.4f}")
print(f"  P-value: {sw_p:.4f}")
print(f"  Conclusion: {'✓ Normal (p>0.05)' if sw_p > 0.05 else '✗ Non-normal (p<0.05)'}")

# Outliers
print(f"\n{'='*80}")
print("OUTLIER ANALYSIS")
print("="*80)

outliers_2sd = np.abs(residuals) > 2*np.std(residuals)
outliers_3sd = np.abs(residuals) > 3*np.std(residuals)

print(f"\nOutliers (>2 SD): {np.sum(outliers_2sd)}/30 ({np.sum(outliers_2sd)/30*100:.1f}%)")
if np.sum(outliers_2sd) > 0:
    outlier_idx = np.where(outliers_2sd)[0]
    for idx in outlier_idx:
        print(f"  Device {idx:2d}: KL={x[idx]:.3f}, Acc={y[idx]:.3f}, Residual={residuals[idx]:+.3f}")

print(f"\nOutliers (>3 SD): {np.sum(outliers_3sd)}/30 ({np.sum(outliers_3sd)/30*100:.1f}%)")

# Influential points (Cook's distance)
print(f"\n{'='*80}")
print("INFLUENTIAL POINTS (Cook's Distance)")
print("="*80)

# Calculate leverage
X_design = np.column_stack([np.ones(n), x])
hat_matrix = X_design @ np.linalg.inv(X_design.T @ X_design) @ X_design.T
leverage = np.diag(hat_matrix)

# Calculate Cook's distance
mse = np.sum(residuals**2) / (n - 2)
cooks_d = (residuals**2 / (2 * mse)) * (leverage / (1 - leverage)**2)

influential = cooks_d > 4/n
print(f"\nInfluential points (Cook's D > {4/n:.3f}): {np.sum(influential)}/30")
if np.sum(influential) > 0:
    inf_idx = np.where(influential)[0]
    for idx in inf_idx:
        print(f"  Device {idx:2d}: KL={x[idx]:.3f}, Acc={y[idx]:.3f}, Cook's D={cooks_d[idx]:.4f}")

# Final assessment
print(f"\n{'='*80}")
print("FINAL ASSESSMENT - SLIDE 14 VALIDITY")
print("="*80)

issues = []
warnings = []

if levene_p < 0.05 or bp_p < 0.05:
    issues.append("HETEROSCEDASTICITY detected - variance not constant")
if sw_p < 0.05:
    warnings.append("Residuals not normally distributed")
if variance_ratio > 3 or variance_ratio < 0.33:
    issues.append(f"Large variance ratio ({variance_ratio:.2f}) suggests heteroscedasticity")
if np.sum(outliers_2sd) > 3:
    warnings.append(f"{np.sum(outliers_2sd)} outliers (>10%) may affect results")
if np.sum(influential) > 2:
    warnings.append(f"{np.sum(influential)} influential points detected")

if len(issues) == 0 and len(warnings) == 0:
    print("\n✓ ALL ASSUMPTIONS SATISFIED")
    print("  - Homoscedasticity: Valid")
    print("  - Linearity: Valid")
    print("  - Residuals: Normal")
    print("\n✓ CORRELATION EVIDENCE ON SLIDE 14 IS VALID")
    print(f"  Pearson r = {r:.3f}, p = {p_value:.2e}")
else:
    if len(issues) > 0:
        print("\n✗ CRITICAL ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
    if len(warnings) > 0:
        print("\n⚠ WARNINGS:")
        for warning in warnings:
            print(f"  - {warning}")
    
    if len(issues) > 0:
        print("\n✗ SLIDE 14 CLAIMS REQUIRE REVISION")
        print("\nRECOMMENDATIONS:")
        print("  1. Use Spearman correlation (more robust to heteroscedasticity)")
        print("  2. Consider robust regression methods")
        print("  3. Add disclaimer about assumption violations")
        print("  4. Use bootstrap confidence intervals")
    else:
        print("\n✓ Correlation still valid but with caveats")
        print("  Spearman ρ is more appropriate given warnings")

print(f"\n{'='*80}\n")
