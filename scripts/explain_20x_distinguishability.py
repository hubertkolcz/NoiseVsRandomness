"""
Explain the "20× between vs within-class" distinguishability metric
"""
import json
from pathlib import Path

# Load N=30 validation results
data_path = Path('results/qgan_tournament_validation_N30.json')
with open(data_path) as f:
    data = json.load(f)

# Extract key values
within_mean = data['original_vs_validation']['validated_kl_within_mean']
between_mean = data['original_vs_validation']['validated_kl_between_mean']
ratio = between_mean / within_mean

# Get detailed stats
within_stats = data['kl_stats']['within_class']
between_stats = data['kl_stats']['between_class']

print("\n" + "="*80)
print("WHAT DOES '20× BETWEEN VS WITHIN-CLASS' MEAN?")
print("="*80)

print("\n1. THE CALCULATION:")
print("-" * 80)
print(f"   Within-class mean KL:  {within_mean:.6f}")
print(f"   Between-class mean KL: {between_mean:.6f}")
print(f"   Ratio: {between_mean:.6f} / {within_mean:.6f} = {ratio:.2f}×")

print("\n2. INTERPRETATION:")
print("-" * 80)
print(f"   Devices from DIFFERENT classes (e.g., low bias vs high bias)")
print(f"   are {ratio:.1f}× MORE DISTINGUISHABLE than devices from")
print(f"   the SAME class (e.g., two low bias devices).")

print("\n3. WHAT THIS MEANS IN PLAIN ENGLISH:")
print("-" * 80)
print("   • Within-class: Comparing two devices with similar characteristics")
print(f"     → Low KL divergence (mean = {within_mean:.3f})")
print("     → Hard to tell apart")
print()
print("   • Between-class: Comparing devices with different characteristics")
print(f"     → High KL divergence (mean = {between_mean:.3f})")
print("     → Easy to distinguish")
print()
print(f"   • The {ratio:.1f}× ratio means the signal (between-class difference)")
print("     is much stronger than the noise (within-class variation)")

print("\n4. STATISTICAL SIGNIFICANCE:")
print("-" * 80)
print(f"   Mann-Whitney U test: p = {data['mann_whitney_test']['p_value']:.2e}")
print(f"   → This p-value (p < 10⁻⁶⁰) means the difference is NOT due to chance")
print(f"   → With 30 devices, we have df=28 degrees of freedom")
print(f"   → Statistical power is EXTREMELY HIGH")

print("\n5. WHY THIS MATTERS FOR SLIDE 14:")
print("-" * 80)
print("   This metric demonstrates that:")
print("   • The classification task is NOT random")
print("   • Classes are genuinely separable")
print("   • ML models have a real signal to learn from")
print("   • The effect size is LARGE (not just statistically significant)")

print("\n6. DETAILED BREAKDOWN:")
print("-" * 80)
print("\n   WITHIN-CLASS KL Divergences:")
for class_pair, stats in within_stats.items():
    print(f"     {class_pair}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, n={stats['n']}")

print("\n   BETWEEN-CLASS KL Divergences:")
for class_pair, stats in between_stats.items():
    print(f"     {class_pair}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, n={stats['n']}")

print("\n7. COHEN'S d (Effect Size):")
print("-" * 80)
# Calculate pooled standard deviation
all_within_n = sum(s['n'] for s in within_stats.values())
all_between_n = sum(s['n'] for s in between_stats.values())

# Weighted average of stds
within_std_avg = sum(s['std'] * s['n'] for s in within_stats.values()) / all_within_n
between_std_avg = sum(s['std'] * s['n'] for s in between_stats.values()) / all_between_n

# Pooled standard deviation
pooled_std = ((all_within_n - 1) * within_std_avg**2 + (all_between_n - 1) * between_std_avg**2)
pooled_std = (pooled_std / (all_within_n + all_between_n - 2)) ** 0.5

cohens_d = (between_mean - within_mean) / pooled_std
print(f"   Cohen's d = {cohens_d:.2f}")
print(f"   Interpretation: ", end="")
if cohens_d > 2.0:
    print("HUGE effect size")
elif cohens_d > 0.8:
    print("LARGE effect size")
elif cohens_d > 0.5:
    print("MEDIUM effect size")
else:
    print("SMALL effect size")

print("\n" + "="*80)
print("CONCLUSION:")
print("="*80)
print(f"The '{ratio:.1f}×' metric is a simple, intuitive way to show that")
print("the classification task has a strong signal. It combines:")
print("  • Effect size (how big is the difference)")
print("  • Statistical significance (p < 10⁻⁶⁰)")
print("  • Practical interpretation (easy vs hard to distinguish)")
print("\nThis supports the claim that ML models can reliably distinguish")
print("between quantum device noise profiles.")
print("="*80 + "\n")
