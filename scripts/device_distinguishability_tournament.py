"""
qGAN-Inspired Device Distinguishability Tournament - FINAL VERSION

Key Insight: Instead of replicating full qGAN training (which was designed for 
a different purpose), we measure the UNDERLYING distinguishability that qGAN 
was designed to capture: How different are the distributions?

We use the same distributional distance metric (KL divergence) but apply it 
directly to compare device outputs, providing a clean tournament scoring system.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
import json
from datetime import datetime
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
DATA_FILE = ROOT_DIR / "AI_2qubits_training_data.txt"
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = ROOT_DIR / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# Reproducibility
np.random.seed(42)

print("="*80)
print("qGAN-INSPIRED DEVICE DISTINGUISHABILITY TOURNAMENT - FINAL")
print("="*80)
print()

# Load data
print("Loading data from AI_2qubits_training_data.txt...")
with open(str(DATA_FILE), 'r') as file:
    data = file.readlines()

device_1_data = data[:2000]
device_2_data = data[2001:4000]
device_3_data = data[4001:6000]

print(f"Device 1: {len(device_1_data)} samples")
print(f"Device 2: {len(device_2_data)} samples")
print(f"Device 3: {len(device_3_data)} samples")
print()

def extract_distribution_features(device_data):
    """
    Extract comprehensive distribution features for KL divergence comparison.
    Uses multiple feature types to capture device characteristics.
    """
    # Method 1: Bit-position frequencies (64 dimensions)
    bit_freq = np.zeros(64)
    for string in device_data:
        for i, bit in enumerate(string[:-39]):  # First 64 bits
            if bit == '1':
                bit_freq[i] += 1
    bit_freq = bit_freq / len(device_data)
    
    # Method 2: 2-bit pattern frequencies (64×64 → 4096 dimensions, matching qGAN)
    # This creates a joint distribution over pairs of bit positions
    pattern_freq = np.zeros((64, 64))
    for string in device_data:
        bits = string[:-39]
        for i in range(64):
            for j in range(64):
                if bits[i] == '1' and bits[j] == '1':
                    pattern_freq[i, j] += 1
    pattern_freq = pattern_freq / len(device_data)
    pattern_freq_flat = pattern_freq.reshape(-1)
    
    # Method 3: Difference-based representation (like original qGAN grid)
    # For a single device, we use autocorrelation structure
    diff_grid = np.zeros((64, 64))
    for i in range(64):
        for j in range(64):
            diff_grid[i, j] = abs(bit_freq[i] - bit_freq[j])
    diff_grid_flat = diff_grid.reshape(-1)
    
    return {
        'bit_freq': bit_freq,
        'pattern_freq': pattern_freq_flat,
        'diff_grid': diff_grid_flat
    }

def compute_kl_divergence(dist1, dist2, method='symmetric'):
    """
    Compute KL divergence with numerical stability.
    
    Args:
        dist1, dist2: Probability distributions (will be normalized)
        method: 'forward', 'reverse', or 'symmetric' (Jensen-Shannon-like)
    """
    # Normalize to probability distributions
    d1 = np.abs(dist1) + 1e-10
    d1 = d1 / d1.sum()
    
    d2 = np.abs(dist2) + 1e-10
    d2 = d2 / d2.sum()
    
    if method == 'forward':
        return entropy(d1, d2)
    elif method == 'reverse':
        return entropy(d2, d1)
    elif method == 'symmetric':
        # Symmetric version: average of both directions
        return (entropy(d1, d2) + entropy(d2, d1)) / 2
    else:
        raise ValueError(f"Unknown method: {method}")

def run_tournament_match(features1, features2, device_names):
    """
    Run a single tournament match between two devices.
    Computes KL divergence across multiple feature representations.
    """
    device_a, device_b = device_names
    
    print(f"Match: {device_a} vs {device_b}")
    print("-" * 60)
    
    results = {}
    
    # Feature type 1: Bit-position frequencies (64-dim)
    kl_bit = compute_kl_divergence(features1['bit_freq'], features2['bit_freq'], method='symmetric')
    results['bit_freq_kl'] = kl_bit
    print(f"  Bit-position KL (64-dim):     {kl_bit:10.6f}")
    
    # Feature type 2: 2-bit pattern frequencies (4096-dim, like qGAN)
    kl_pattern = compute_kl_divergence(features1['pattern_freq'], features2['pattern_freq'], method='symmetric')
    results['pattern_freq_kl'] = kl_pattern
    print(f"  2-bit pattern KL (4096-dim):  {kl_pattern:10.6f}")
    
    # Feature type 3: Difference grid (4096-dim, qGAN-style)
    kl_diff = compute_kl_divergence(features1['diff_grid'], features2['diff_grid'], method='symmetric')
    results['diff_grid_kl'] = kl_diff
    print(f"  Difference grid KL (4096-dim): {kl_diff:10.6f}")
    
    # Composite score: weighted average (emphasize high-dim features)
    composite = (0.2 * kl_bit + 0.4 * kl_pattern + 0.4 * kl_diff)
    results['composite_score'] = composite
    print(f"  Composite Distinguishability:  {composite:10.6f}")
    print()
    
    return results

# Extract features for all devices
print("="*80)
print("FEATURE EXTRACTION")
print("="*80)
print()

print("Extracting distributional features from each device...")
features_1 = extract_distribution_features(device_1_data)
features_2 = extract_distribution_features(device_2_data)
features_3 = extract_distribution_features(device_3_data)

print(f"  Device 1: bit_freq mean={features_1['bit_freq'].mean():.4f}")
print(f"  Device 2: bit_freq mean={features_2['bit_freq'].mean():.4f}")
print(f"  Device 3: bit_freq mean={features_3['bit_freq'].mean():.4f}")
print()

# Run tournament
print("="*80)
print("TOURNAMENT MATCHES")
print("="*80)
print()

tournament_results = {}

# Match 1: Device 1 vs Device 2
results_1v2 = run_tournament_match(features_1, features_2, ("Device 1", "Device 2"))
tournament_results['1v2'] = results_1v2

# Match 2: Device 1 vs Device 3
results_1v3 = run_tournament_match(features_1, features_3, ("Device 1", "Device 3"))
tournament_results['1v3'] = results_1v3

# Match 3: Device 2 vs Device 3
results_2v3 = run_tournament_match(features_2, features_3, ("Device 2", "Device 3"))
tournament_results['2v3'] = results_2v3

# Analysis
print("="*80)
print("TOURNAMENT RESULTS & ANALYSIS")
print("="*80)
print()

print("Distinguishability Ranking (by Composite Score):")
print("-" * 80)

composite_scores = {
    'Device 1 vs 2': tournament_results['1v2']['composite_score'],
    'Device 1 vs 3': tournament_results['1v3']['composite_score'],
    'Device 2 vs 3': tournament_results['2v3']['composite_score']
}

sorted_pairs = sorted(composite_scores.items(), key=lambda x: x[1], reverse=True)

for rank, (pair, score) in enumerate(sorted_pairs, 1):
    if score > 5:
        interp = "HIGHLY DISTINGUISHABLE"
    elif score > 1:
        interp = "MODERATELY DISTINGUISHABLE"
    elif score > 0.1:
        interp = "SOMEWHAT DISTINGUISHABLE"
    else:
        interp = "DIFFICULT TO DISTINGUISH"
    
    print(f"  {rank}. {pair:20s}: Score = {score:10.6f}  [{interp}]")

print()
print("Cross-Validation with Classification Results:")
print("-" * 80)
print("Per-device classification accuracies:")
print("  Device 1: 66.7%")
print("  Device 2: 65.0%")
print("  Device 3: 70.0% (easiest to identify)")
print()

most_distinguishable = sorted_pairs[0]
print(f"Most distinguishable pair: {most_distinguishable[0]}")
print(f"  Score: {most_distinguishable[1]:.6f}")
print()

if 'Device 2 vs 3' in most_distinguishable[0] or 'Device 1 vs 3' in most_distinguishable[0]:
    print("CORRELATION: Pairs involving Device 3 are most distinguishable")
    print("  This aligns with Device 3 having highest classification accuracy (70%)")
else:
    print("[WARN] OBSERVATION: Top distinguishable pair may not involve easiest-to-classify device")
    print("  This could indicate distributional vs. instance-level differences")

print()
print("Methodology:")
print("-" * 80)
print("This tournament uses KL divergence to measure distributional distance,")
print("inspired by qGAN's approach but applied directly to device outputs.")
print("Features include:")
print("  - Bit-position frequencies (64-dim)")
print("  - 2-bit joint patterns (4096-dim, matching qGAN dimensionality)")
print("  - Autocorrelation structure (4096-dim, qGAN-style grid)")
print()

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Composite Scores
ax1 = axes[0, 0]
pairs = ['Device 1 vs 2', 'Device 1 vs 3', 'Device 2 vs 3']
scores = [tournament_results['1v2']['composite_score'], 
          tournament_results['1v3']['composite_score'], 
          tournament_results['2v3']['composite_score']]
colors = ['#3498db', '#e74c3c', '#2ecc71']

bars = ax1.bar(pairs, scores, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax1.set_ylabel('Distinguishability Score', fontsize=12, fontweight='bold')
ax1.set_title('Device Distinguishability Tournament - Composite Scores', fontsize=14, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim(bottom=0)

for bar, val in zip(bars, scores):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

# Plot 2: Feature-specific KL scores
ax2 = axes[0, 1]
feature_names = ['Bit-position\n(64-dim)', '2-bit pattern\n(4096-dim)', 'Diff grid\n(4096-dim)']
x = np.arange(len(feature_names))
width = 0.25

kl_1v2 = [tournament_results['1v2']['bit_freq_kl'], tournament_results['1v2']['pattern_freq_kl'], tournament_results['1v2']['diff_grid_kl']]
kl_1v3 = [tournament_results['1v3']['bit_freq_kl'], tournament_results['1v3']['pattern_freq_kl'], tournament_results['1v3']['diff_grid_kl']]
kl_2v3 = [tournament_results['2v3']['bit_freq_kl'], tournament_results['2v3']['pattern_freq_kl'], tournament_results['2v3']['diff_grid_kl']]

ax2.bar(x - width, kl_1v2, width, label='Device 1 vs 2', color='#3498db', alpha=0.7)
ax2.bar(x, kl_1v3, width, label='Device 1 vs 3', color='#e74c3c', alpha=0.7)
ax2.bar(x + width, kl_2v3, width, label='Device 2 vs 3', color='#2ecc71', alpha=0.7)

ax2.set_ylabel('KL Divergence', fontsize=12, fontweight='bold')
ax2.set_title('Feature-Specific Distinguishability', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(feature_names, fontsize=10)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# Plot 3: Bit-position frequency distributions
ax3 = axes[1, 0]
ax3.plot(features_1['bit_freq'], label='Device 1', color='#3498db', linewidth=2, alpha=0.7)
ax3.plot(features_2['bit_freq'], label='Device 2', color='#e74c3c', linewidth=2, alpha=0.7)
ax3.plot(features_3['bit_freq'], label='Device 3', color='#2ecc71', linewidth=2, alpha=0.7)
ax3.set_xlabel('Bit Position', fontsize=12, fontweight='bold')
ax3.set_ylabel('Frequency of "1"', fontsize=12, fontweight='bold')
ax3.set_title('Device Bit-Position Frequency Profiles', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(alpha=0.3)

# Plot 4: Correlation analysis
ax4 = axes[1, 1]
classification_acc = [66.7, 65.0, 70.0]  # Device 1, 2, 3
device_labels = ['Device 1', 'Device 2', 'Device 3']

# For each device, compute average distinguishability from others
avg_distinguishability = [
    (tournament_results['1v2']['composite_score'] + tournament_results['1v3']['composite_score']) / 2,  # Device 1
    (tournament_results['1v2']['composite_score'] + tournament_results['2v3']['composite_score']) / 2,  # Device 2
    (tournament_results['1v3']['composite_score'] + tournament_results['2v3']['composite_score']) / 2,  # Device 3
]

scatter = ax4.scatter(avg_distinguishability, classification_acc, 
                     c=['#3498db', '#e74c3c', '#2ecc71'], 
                     s=200, alpha=0.7, edgecolors='black', linewidth=2)
for i, label in enumerate(device_labels):
    ax4.annotate(label, (avg_distinguishability[i], classification_acc[i]), 
                fontsize=11, fontweight='bold', 
                xytext=(10, 5), textcoords='offset points')

ax4.set_xlabel('Avg Distinguishability Score', fontsize=12, fontweight='bold')
ax4.set_ylabel('Classification Accuracy (%)', fontsize=12, fontweight='bold')
ax4.set_title('Distinguishability vs Classification Performance', fontsize=14, fontweight='bold')
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(str(FIGURES_DIR / 'fig10_device_distinguishability_final.png'), dpi=300, bbox_inches='tight')
print(f"Figure saved: fig10_device_distinguishability_final.png")
plt.close()

# Save results
output = {
    'timestamp': datetime.now().isoformat(),
    'methodology': 'qGAN-Inspired Distributional Distance Tournament',
    'description': 'Direct KL divergence comparison across multiple feature representations',
    'feature_dimensions': {
        'bit_position': 64,
        'two_bit_pattern': 4096,
        'difference_grid': 4096
    },
    'results': {
        'device_1_vs_2': {
            'composite_score': float(tournament_results['1v2']['composite_score']),
            'bit_freq_kl': float(tournament_results['1v2']['bit_freq_kl']),
            'pattern_freq_kl': float(tournament_results['1v2']['pattern_freq_kl']),
            'diff_grid_kl': float(tournament_results['1v2']['diff_grid_kl'])
        },
        'device_1_vs_3': {
            'composite_score': float(tournament_results['1v3']['composite_score']),
            'bit_freq_kl': float(tournament_results['1v3']['bit_freq_kl']),
            'pattern_freq_kl': float(tournament_results['1v3']['pattern_freq_kl']),
            'diff_grid_kl': float(tournament_results['1v3']['diff_grid_kl'])
        },
        'device_2_vs_3': {
            'composite_score': float(tournament_results['2v3']['composite_score']),
            'bit_freq_kl': float(tournament_results['2v3']['bit_freq_kl']),
            'pattern_freq_kl': float(tournament_results['2v3']['pattern_freq_kl']),
            'diff_grid_kl': float(tournament_results['2v3']['diff_grid_kl'])
        }
    },
    'ranking': [
        {'pair': pair, 'score': float(score)}
        for pair, score in sorted_pairs
    ],
    'most_distinguishable': {
        'pair': sorted_pairs[0][0],
        'score': float(sorted_pairs[0][1])
    },
    'least_distinguishable': {
        'pair': sorted_pairs[-1][0],
        'score': float(sorted_pairs[-1][1])
    },
    'cross_validation': {
        'classification_accuracy_d1': 66.7,
        'classification_accuracy_d2': 65.0,
        'classification_accuracy_d3': 70.0
    }
}

with open(str(RESULTS_DIR / 'device_distinguishability_tournament_final.json'), 'w') as f:
    json.dump(output, f, indent=2)

print(f"Results saved: {RESULTS_DIR / 'device_distinguishability_tournament_final.json'}")
print()

print("="*80)
print("TOURNAMENT COMPLETE")
print("="*80)
print()
print("Summary:")
print(f"  Most distinguishable: {sorted_pairs[0][0]} (Score: {sorted_pairs[0][1]:.6f})")
print(f"  Least distinguishable: {sorted_pairs[-1][0]} (Score: {sorted_pairs[-1][1]:.6f})")
print()
print("This tournament provides a complementary view to classification accuracy,")
print("measuring distributional distinguishability inspired by qGAN methodology.")
