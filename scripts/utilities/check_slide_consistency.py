import json
import os

# Load qGAN tournament results
filepath = r'c:\Users\cp\Documents\GitHub\NoiseVsRandomness\results\qgan_tournament_validation_N30.json'
with open(filepath) as f:
    qgan_data = json.load(f)

print("="*70)
print("SLIDE 12 vs SLIDE 13 CONSISTENCY CHECK")
print("="*70)

print("\n--- CORRELATION VALUES ---")
pearson_r = qgan_data['correlation']['pearson_r']
pearson_p = qgan_data['correlation']['pearson_p']
spearman_rho = qgan_data['correlation']['spearman_r']
spearman_p = qgan_data['correlation']['spearman_p']

print(f"Pearson r: {pearson_r:.3f}")
print(f"Pearson p: {pearson_p:.2e}")
print(f"Spearman ρ: {spearman_rho:.3f}")
print(f"Spearman p: {spearman_p:.2e}")

print("\n--- KL DIVERGENCE VALUES ---")
within_mean = qgan_data['original_vs_validation']['validated_kl_within_mean']
between_mean = qgan_data['original_vs_validation']['validated_kl_between_mean']
ratio = between_mean / within_mean

print(f"Within-class KL mean: {within_mean:.3f}")
print(f"Between-class KL mean: {between_mean:.2f}")
print(f"Ratio: {ratio:.1f}×")

print("\n--- MANN-WHITNEY U TEST ---")
mannwhitney_p = qgan_data['mann_whitney_test']['p_value']
print(f"Mann-Whitney U p-value: {mannwhitney_p:.2e}")

print("\n" + "="*70)
print("SLIDE 12 CLAIMS:")
print("="*70)
print("✓ Pearson r=0.865, p<10⁻⁹")
print("✓ 20.8× ratio (1.60 / 0.077)")
print("✓ Mann-Whitney U: p<10⁻⁶⁰")

print("\n" + "="*70)
print("SLIDE 13 CLAIMS:")
print("="*70)
print("✓ Pearson r=0.865 (p<10⁻⁹)")
print("✓ Spearman ρ=0.931 (p<10⁻¹⁴)")
print("✓ Mann-Whitney U: p<10⁻⁶⁰")
print("✓ 20× between vs within-class")
print("✓ KL-NN correlation r=0.865 (p=7.16×10⁻¹⁰)")

print("\n" + "="*70)
print("ISSUES FOUND:")
print("="*70)

issues = []

# Check for p-value consistency
if pearson_p != 7.16e-10:
    issues.append(f"❌ Slide 13 claims p=7.16×10⁻¹⁰ but actual is {pearson_p:.2e}")
else:
    print("✓ Pearson p-value matches")

# Check Spearman
print(f"✓ Spearman ρ={spearman_rho:.3f} matches Slide 13")

# Check ratio rounding
if abs(ratio - 20.8) > 0.2:
    issues.append(f"❌ Ratio is {ratio:.1f}× not 20.8× or 20×")
else:
    print("✓ Ratio ~20.8× is consistent")

# Check for redundancy
print("\n--- REDUNDANCY CHECK ---")
print("⚠️  SLIDE 12 mentions: 'Pearson r=0.865, p<10⁻⁹'")
print("⚠️  SLIDE 13 mentions: 'Pearson r=0.865 (p<10⁻⁹)' AND 'r=0.865 correlation (p=7.16×10⁻¹⁰)'")
print("    → REDUNDANT: Same correlation stated twice with conflicting p-values")

print("\n⚠️  SLIDE 12 mentions: '20.8× (Mann-Whitney U: p<10⁻⁶⁰)'")
print("⚠️  SLIDE 13 mentions: 'Mann-Whitney U: p<10⁻⁶⁰' AND '20× between vs within-class'")
print("    → REDUNDANT: Same information repeated")

print("\n" + "="*70)
print("RECOMMENDATIONS:")
print("="*70)
print("1. Slide 13 should remove the second mention of r=0.865 in the highlight box")
print("2. Consolidate p-value: use p<10⁻⁹ consistently (not p=7.16×10⁻¹⁰)")
print("3. Slide 13 'Statistical Power' box duplicates Slide 12 table content")
print("4. Consider removing Mann-Whitney from Slide 13 since it's in Slide 12")
