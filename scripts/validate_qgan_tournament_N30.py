"""
Statistical Validation of qGAN Tournament Concept on N=30 Synthetic Devices

This script tests whether the "qGAN tournament" approach (measuring distributional
distinguishability via KL divergence) is statistically valid when scaled from N=3
to N=30 devices.

Key Questions:
1. Do devices with different bias levels have significantly different KL divergences?
2. Is there a correlation between KL divergence and classification accuracy?
3. Are the N=3 results (KL: 0.05, 0.205, 0.202) representative of larger samples?

Original N=3 Claims:
- Device 1 vs 2: KL = 0.050 (low distinguishability)
- Device 1 vs 3: KL = 0.205 (high distinguishability)
- Device 2 vs 3: KL = 0.202 (high distinguishability)
- Correlation with NN accuracy: r = 0.949 (claimed but unvalidated)
"""

import os
import sys

# Fix Intel Fortran/MKL threading issues on Windows
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

import numpy as np
from scipy.stats import entropy, pearsonr, spearmanr, mannwhitneyu
from scipy.special import kl_div
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = ROOT_DIR / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# Set seeds
np.random.seed(42)
torch.manual_seed(42)

print("="*80)
print("qGAN TOURNAMENT VALIDATION: N=30 DEVICES")
print("="*80)
print()

# ============================================================================
# SYNTHETIC DATA GENERATION (Same as validate_framework_synthetic.py)
# ============================================================================

def generate_synthetic_rng(n_samples=2000, bit_length=100, bias_level=0.5, 
                          temporal_correlation=0.0, drift=0.0):
    """Generate synthetic RNG with controlled characteristics"""
    samples = np.zeros((n_samples, bit_length), dtype=int)
    
    for i in range(n_samples):
        current_bias = bias_level + drift * (i / n_samples - 0.5)
        current_bias = np.clip(current_bias, 0.1, 0.9)
        
        for j in range(bit_length):
            if j == 0 or temporal_correlation == 0.0:
                samples[i, j] = 1 if np.random.rand() < current_bias else 0
            else:
                prev_bit = samples[i, j-1]
                if np.random.rand() < temporal_correlation:
                    samples[i, j] = prev_bit
                else:
                    samples[i, j] = 1 if np.random.rand() < current_bias else 0
    
    actual_freq = np.mean(samples)
    actual_entropy = entropy([1 - actual_freq, actual_freq], base=2)
    
    # Markov transitions
    transitions = {'00': 0, '01': 0, '10': 0, '11': 0}
    for sample in samples:
        for k in range(len(sample) - 1):
            pair = f"{sample[k]}{sample[k+1]}"
            transitions[pair] += 1
    total = sum(transitions.values())
    transition_probs = {k: v/total for k, v in transitions.items()}
    
    metadata = {
        'bias_level_target': bias_level,
        'temporal_correlation': temporal_correlation,
        'drift': drift,
        'actual_freq_1': actual_freq,
        'actual_entropy': actual_entropy,
        'transition_probs': transition_probs
    }
    
    return samples, metadata


def create_synthetic_dataset(n_devices_per_class=10):
    """
    Create N=30 device dataset with 3 bias classes:
    - Class 0 (Low bias): 48-52% '1' frequency (10 devices)
    - Class 1 (Med bias): 54-58% '1' frequency (10 devices)
    - Class 2 (High bias): 60-65% '1' frequency (10 devices)
    """
    X = []
    y = []
    device_metadata = []
    device_id = 0
    
    # Class 0: Low bias (similar to certified RNGs)
    print(f"Generating Class 0 devices (low bias)...")
    for i in range(n_devices_per_class):
        bias = np.random.uniform(0.48, 0.52)
        samples, meta = generate_synthetic_rng(
            n_samples=2000,
            bias_level=bias,
            temporal_correlation=np.random.uniform(0.0, 0.05),
            drift=np.random.uniform(-0.01, 0.01)
        )
        X.append(samples)
        y.extend([0] * len(samples))
        device_metadata.append({
            'device_id': device_id,
            'class': 0,
            'class_name': 'low_bias',
            **meta
        })
        device_id += 1
    
    # Class 1: Medium bias
    print(f"Generating Class 1 devices (medium bias)...")
    for i in range(n_devices_per_class):
        bias = np.random.uniform(0.54, 0.58)
        samples, meta = generate_synthetic_rng(
            n_samples=2000,
            bias_level=bias,
            temporal_correlation=np.random.uniform(0.0, 0.1),
            drift=np.random.uniform(-0.02, 0.02)
        )
        X.append(samples)
        y.extend([1] * len(samples))
        device_metadata.append({
            'device_id': device_id,
            'class': 1,
            'class_name': 'medium_bias',
            **meta
        })
        device_id += 1
    
    # Class 2: High bias (exploitable)
    print(f"Generating Class 2 devices (high bias)...")
    for i in range(n_devices_per_class):
        bias = np.random.uniform(0.60, 0.65)
        samples, meta = generate_synthetic_rng(
            n_samples=2000,
            bias_level=bias,
            temporal_correlation=np.random.uniform(0.05, 0.15),
            drift=np.random.uniform(-0.03, 0.03)
        )
        X.append(samples)
        y.extend([2] * len(samples))
        device_metadata.append({
            'device_id': device_id,
            'class': 2,
            'class_name': 'high_bias',
            **meta
        })
        device_id += 1
    
    X = np.vstack(X)
    y = np.array(y)
    
    return X, y, device_metadata


# ============================================================================
# KL DIVERGENCE TOURNAMENT (Full Pairwise)
# ============================================================================

def compute_kl_divergence_advanced(samples1, samples2, method='histogram'):
    """
    Compute KL divergence between two RNG sample sets using multiple methods.
    
    Methods:
    - 'histogram': Distribution of per-sample mean bit frequencies
    - 'bit_position': Per-bit position frequency distributions
    - 'pattern': Two-bit pattern frequencies (00, 01, 10, 11)
    """
    if method == 'histogram':
        # Distribution of sample means (as in original tournament)
        hist1, bins = np.histogram(samples1.mean(axis=1), bins=20, range=(0, 1), density=True)
        hist2, _ = np.histogram(samples2.mean(axis=1), bins=bins, density=True)
        
    elif method == 'bit_position':
        # Per-position bit frequencies (64-dimensional distribution)
        hist1 = samples1.mean(axis=0)  # Frequency of '1' at each position
        hist2 = samples2.mean(axis=0)
        
    elif method == 'pattern':
        # Two-bit pattern frequencies
        def extract_patterns(samples):
            patterns = {'00': 0, '01': 0, '10': 0, '11': 0}
            for sample in samples:
                for i in range(len(sample) - 1):
                    pair = f"{sample[i]}{sample[i+1]}"
                    patterns[pair] += 1
            total = sum(patterns.values())
            return np.array([patterns[k]/total for k in ['00', '01', '10', '11']])
        
        hist1 = extract_patterns(samples1)
        hist2 = extract_patterns(samples2)
    
    # Add epsilon and normalize
    hist1 = hist1 + 1e-10
    hist2 = hist2 + 1e-10
    hist1 = hist1 / hist1.sum()
    hist2 = hist2 / hist2.sum()
    
    # Compute KL divergence
    kl = np.sum(kl_div(hist1, hist2))
    
    return kl


def compute_full_kl_tournament(X, y, device_metadata, method='histogram'):
    """
    Compute KL divergence for ALL device pairs (N*(N-1)/2 comparisons).
    Returns matrix and summary statistics.
    """
    n_devices = len(device_metadata)
    kl_matrix = np.zeros((n_devices, n_devices))
    
    # Group samples by device
    samples_per_device = len(X) // n_devices
    device_samples = []
    for i in range(n_devices):
        start_idx = i * samples_per_device
        end_idx = (i + 1) * samples_per_device
        device_samples.append(X[start_idx:end_idx])
    
    print(f"Computing {n_devices*(n_devices-1)//2} pairwise KL divergences...")
    
    for i in range(n_devices):
        for j in range(i+1, n_devices):
            kl_ij = compute_kl_divergence_advanced(device_samples[i], device_samples[j], method)
            kl_ji = compute_kl_divergence_advanced(device_samples[j], device_samples[i], method)
            
            # Use symmetric KL (average of both directions)
            kl_symmetric = (kl_ij + kl_ji) / 2
            
            kl_matrix[i, j] = kl_symmetric
            kl_matrix[j, i] = kl_symmetric
    
    return kl_matrix, device_samples


# ============================================================================
# NEURAL NETWORK FOR CLASSIFICATION
# ============================================================================

class SimpleNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 30)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(30, 20)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(20, num_classes)
    
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x


def train_neural_network(X_train, y_train, X_test, y_test, epochs=50, batch_size=8):
    """Train neural network for device classification"""
    input_size = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    
    model = SimpleNN(input_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        train_outputs = model(torch.FloatTensor(X_train))
        train_preds = torch.argmax(train_outputs, dim=1).numpy()
        train_acc = accuracy_score(y_train, train_preds)
        
        test_outputs = model(torch.FloatTensor(X_test))
        test_preds = torch.argmax(test_outputs, dim=1).numpy()
        test_acc = accuracy_score(y_test, test_preds)
        
        # Per-device predictions (on full dataset for proper alignment)
        all_outputs = model(torch.FloatTensor(np.vstack([X_train, X_test])))
        all_preds = torch.argmax(all_outputs, dim=1).numpy()
    
    return {
        'model': model,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'all_predictions': all_preds
    }


def compute_per_device_accuracy_from_preds(predictions, device_metadata, samples_per_device):
    """Compute accuracy for each device from model predictions"""
    device_accuracies = []
    
    for i, meta in enumerate(device_metadata):
        start_idx = i * samples_per_device
        end_idx = (i + 1) * samples_per_device
        
        device_preds = predictions[start_idx:end_idx]
        true_label = meta['class']
        
        # Accuracy = fraction of samples correctly classified as this device's class
        correct = np.sum(device_preds == true_label)
        acc = correct / len(device_preds)
        
        device_accuracies.append({
            'device_id': meta['device_id'],
            'class': meta['class'],
            'accuracy': acc,
            'bias_level': meta['actual_freq_1']
        })
    
    return device_accuracies


def train_logistic_regression(X_train, y_train, X_test, y_test):
    """Train logistic regression classifier for device classification"""
    # Scale features for logistic regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train logistic regression with L2 regularization
    lr_model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        solver='lbfgs',
        multi_class='multinomial',
        C=1.0  # Inverse of regularization strength
    )
    
    lr_model.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_preds = lr_model.predict(X_train_scaled)
    train_acc = accuracy_score(y_train, train_preds)
    
    test_preds = lr_model.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, test_preds)
    
    # Per-device predictions (on full dataset)
    X_full_scaled = scaler.transform(np.vstack([X_train, X_test]))
    all_preds = lr_model.predict(X_full_scaled)
    
    return {
        'model': lr_model,
        'scaler': scaler,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'all_predictions': all_preds
    }


# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

def analyze_kl_by_class_pair(kl_matrix, device_metadata):
    """
    Analyze KL divergences grouped by class pairs:
    - Within-class (0-0, 1-1, 2-2)
    - Between-class (0-1, 0-2, 1-2)
    """
    results = {
        'within_class': {'0-0': [], '1-1': [], '2-2': []},
        'between_class': {'0-1': [], '0-2': [], '1-2': []}
    }
    
    n_devices = len(device_metadata)
    for i in range(n_devices):
        for j in range(i+1, n_devices):
            class_i = device_metadata[i]['class']
            class_j = device_metadata[j]['class']
            kl_val = kl_matrix[i, j]
            
            if class_i == class_j:
                # Within-class
                key = f"{class_i}-{class_i}"
                results['within_class'][key].append(kl_val)
            else:
                # Between-class
                key = f"{min(class_i, class_j)}-{max(class_i, class_j)}"
                results['between_class'][key].append(kl_val)
    
    # Compute statistics
    stats = {}
    for category in ['within_class', 'between_class']:
        stats[category] = {}
        for key, values in results[category].items():
            if len(values) > 0:
                stats[category][key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'median': np.median(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'n': len(values)
                }
    
    return results, stats


def test_correlation_with_accuracy(kl_matrix, device_accuracies):
    """
    Test correlation between average KL divergence and classification accuracy.
    This is the key test of the r=0.949 claim.
    """
    n_devices = len(device_accuracies)
    
    # Compute average KL divergence for each device (excluding self)
    avg_kl_per_device = []
    for i in range(n_devices):
        kl_values = [kl_matrix[i, j] for j in range(n_devices) if i != j]
        avg_kl_per_device.append(np.mean(kl_values))
    
    # Extract accuracies
    accuracies = [d['accuracy'] for d in device_accuracies]
    
    # Compute correlations
    pearson_r, pearson_p = pearsonr(avg_kl_per_device, accuracies)
    spearman_r, spearman_p = spearmanr(avg_kl_per_device, accuracies)
    
    return {
        'avg_kl': avg_kl_per_device,
        'accuracies': accuracies,
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'significant': pearson_p < 0.05
    }


# ============================================================================
# MAIN VALIDATION
# ============================================================================

def main():
    print("STEP 1: Generate N=30 synthetic device dataset")
    print("-" * 80)
    X, y, device_metadata = create_synthetic_dataset(n_devices_per_class=10)
    print(f"Total samples: {len(X)}")
    print(f"Total devices: {len(device_metadata)}")
    print(f"Samples per device: {len(X) // len(device_metadata)}")
    print()
    
    print("STEP 2: Compute full KL divergence tournament")
    print("-" * 80)
    kl_matrix, device_samples = compute_full_kl_tournament(X, y, device_metadata, method='histogram')
    print(f"KL matrix shape: {kl_matrix.shape}")
    print(f"Non-zero KL values: {np.sum(kl_matrix > 0)}")
    print(f"Mean KL (all pairs): {kl_matrix[kl_matrix > 0].mean():.6f}")
    print(f"Std KL (all pairs): {kl_matrix[kl_matrix > 0].std():.6f}")
    print()
    
    print("STEP 3: Analyze KL by class pairs")
    print("-" * 80)
    kl_by_class, kl_stats = analyze_kl_by_class_pair(kl_matrix, device_metadata)
    
    print("Within-class KL divergences:")
    for key, stats in kl_stats['within_class'].items():
        print(f"  Class {key}: mean={stats['mean']:.6f}, std={stats['std']:.6f}, n={stats['n']}")
    
    print("\nBetween-class KL divergences:")
    for key, stats in kl_stats['between_class'].items():
        print(f"  Classes {key}: mean={stats['mean']:.6f}, std={stats['std']:.6f}, n={stats['n']}")
    print()
    
    print("STEP 4: Train neural network classifier")
    print("-" * 80)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    nn_results = train_neural_network(X_train, y_train, X_test, y_test, epochs=50, batch_size=8)
    print(f"Train accuracy: {nn_results['train_accuracy']:.4f}")
    print(f"Test accuracy: {nn_results['test_accuracy']:.4f}")
    print()
    
    print("STEP 4B: Train logistic regression classifier")
    print("-" * 80)
    lr_results = train_logistic_regression(X_train, y_train, X_test, y_test)
    print(f"Train accuracy: {lr_results['train_accuracy']:.4f}")
    print(f"Test accuracy: {lr_results['test_accuracy']:.4f}")
    print()
    
    print("STEP 5: Compute per-device classification accuracy")
    print("-" * 80)
    samples_per_device = len(X) // len(device_metadata)
    
    # Get NN predictions on FULL dataset (to match device ordering)
    with torch.no_grad():
        nn_results['model'].eval()
        full_outputs = nn_results['model'](torch.FloatTensor(X))
        full_preds = torch.argmax(full_outputs, dim=1).numpy()
    
    device_accuracies = compute_per_device_accuracy_from_preds(full_preds, device_metadata, samples_per_device)
    
    # Get LR predictions on FULL dataset
    X_full_scaled = lr_results['scaler'].transform(X)
    lr_full_preds = lr_results['model'].predict(X_full_scaled)
    lr_device_accuracies = compute_per_device_accuracy_from_preds(lr_full_preds, device_metadata, samples_per_device)
    
    print("Per-device accuracies (NN):")
    for d in device_accuracies[:5]:  # Show first 5
        print(f"  Device {d['device_id']} (class {d['class']}): {d['accuracy']:.4f}")
    print(f"  ... ({len(device_accuracies)} devices total)")
    
    print("\nPer-device accuracies (LR):")
    for d in lr_device_accuracies[:5]:  # Show first 5
        print(f"  Device {d['device_id']} (class {d['class']}): {d['accuracy']:.4f}")
    print(f"  ... ({len(lr_device_accuracies)} devices total)")
    print()
    
    print("STEP 6: Test correlation between KL and accuracy")
    print("-" * 80)
    correlation_results = test_correlation_with_accuracy(kl_matrix, device_accuracies)
    lr_correlation_results = test_correlation_with_accuracy(kl_matrix, lr_device_accuracies)
    
    print("Neural Network:")
    print(f"  Pearson correlation:  r = {correlation_results['pearson_r']:.6f}")
    print(f"  P-value:              p = {correlation_results['pearson_p']:.6f}")
    print(f"  Spearman correlation: rho = {correlation_results['spearman_r']:.6f}")
    print(f"  P-value:              p = {correlation_results['spearman_p']:.6f}")
    print(f"  Statistically significant (p<0.05): {correlation_results['significant']}")
    
    print("\nLogistic Regression:")
    print(f"  Pearson correlation:  r = {lr_correlation_results['pearson_r']:.6f}")
    print(f"  P-value:              p = {lr_correlation_results['pearson_p']:.6f}")
    print(f"  Spearman correlation: rho = {lr_correlation_results['spearman_r']:.6f}")
    print(f"  P-value:              p = {lr_correlation_results['spearman_p']:.6f}")
    print(f"  Statistically significant (p<0.05): {lr_correlation_results['significant']}")
    print()
    
    print("="*80)
    print("COMPARISON WITH ORIGINAL N=3 STUDY")
    print("="*80)
    print(f"Original claim:")
    print(f"  Device 1 vs 2: KL = 0.050 (low distinguishability)")
    print(f"  Device 1 vs 3: KL = 0.205 (high distinguishability)")
    print(f"  Device 2 vs 3: KL = 0.202 (high distinguishability)")
    print(f"  Correlation (qGAN-NN): r = 0.949")
    print()
    print(f"N=30 validation:")
    print(f"  Within-class mean KL: {np.mean([s['mean'] for s in kl_stats['within_class'].values()]):.6f}")
    print(f"  Between-class mean KL: {np.mean([s['mean'] for s in kl_stats['between_class'].values()]):.6f}")
    print(f"  Correlation (KL-NN): r = {correlation_results['pearson_r']:.6f} (p = {correlation_results['pearson_p']:.6f})")
    print()
    
    # Statistical test: Are between-class KLs significantly larger than within-class?
    all_within = []
    all_between = []
    for values in kl_by_class['within_class'].values():
        all_within.extend(values)
    for values in kl_by_class['between_class'].values():
        all_between.extend(values)
    
    u_stat, u_pval = mannwhitneyu(all_between, all_within, alternative='greater')
    print(f"Mann-Whitney U test (between > within):")
    print(f"  U-statistic = {u_stat:.2f}")
    print(f"  P-value = {u_pval:.6e}")
    print(f"  Conclusion: Between-class KLs ARE {'significantly' if u_pval < 0.05 else 'NOT significantly'} larger")
    print()
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'dataset': {
            'n_devices': len(device_metadata),
            'n_samples': len(X),
            'samples_per_device': samples_per_device
        },
        'kl_stats': kl_stats,
        'classification': {
            'nn_train_accuracy': nn_results['train_accuracy'],
            'nn_test_accuracy': nn_results['test_accuracy'],
            'lr_train_accuracy': lr_results['train_accuracy'],
            'lr_test_accuracy': lr_results['test_accuracy'],
            'train_accuracy': nn_results['train_accuracy'],  # Keep for backward compatibility
            'test_accuracy': nn_results['test_accuracy']      # Keep for backward compatibility
        },
        'correlation': {
            'nn_pearson_r': float(correlation_results['pearson_r']),
            'nn_pearson_p': float(correlation_results['pearson_p']),
            'nn_spearman_r': float(correlation_results['spearman_r']),
            'nn_spearman_p': float(correlation_results['spearman_p']),
            'lr_pearson_r': float(lr_correlation_results['pearson_r']),
            'lr_pearson_p': float(lr_correlation_results['pearson_p']),
            'lr_spearman_r': float(lr_correlation_results['spearman_r']),
            'lr_spearman_p': float(lr_correlation_results['spearman_p']),
            'pearson_r': float(correlation_results['pearson_r']),      # Keep for backward compatibility
            'pearson_p': float(correlation_results['pearson_p']),      # Keep for backward compatibility
            'spearman_r': float(correlation_results['spearman_r']),    # Keep for backward compatibility
            'spearman_p': float(correlation_results['spearman_p']),    # Keep for backward compatibility
            'significant': bool(correlation_results['significant']),
            'per_device_data': {
                'avg_kl': [float(x) for x in correlation_results['avg_kl']],
                'nn_accuracies': [float(x) for x in correlation_results['accuracies']],
                'lr_accuracies': [float(x) for x in lr_correlation_results['accuracies']]
            }
        },
        'original_vs_validation': {
            'original_kl_low': 0.050,
            'original_kl_high': 0.205,
            'original_correlation': 0.949,
            'validated_kl_within_mean': float(np.mean([s['mean'] for s in kl_stats['within_class'].values()])),
            'validated_kl_between_mean': float(np.mean([s['mean'] for s in kl_stats['between_class'].values()])),
            'validated_correlation': correlation_results['pearson_r']
        },
        'mann_whitney_test': {
            'u_statistic': float(u_stat),
            'p_value': float(u_pval),
            'significant': bool(u_pval < 0.05)
        }
    }
    
    with open(str(RESULTS_DIR / 'qgan_tournament_validation_N30.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Results saved to: qgan_tournament_validation_N30.json")
    print()
    
    # Visualization
    print("STEP 7: Generate visualization")
    print("-" * 80)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. KL matrix heatmap
    ax1 = axes[0, 0]
    sns.heatmap(kl_matrix, cmap='YlOrRd', ax=ax1, cbar_kws={'label': 'KL Divergence'})
    ax1.set_title('KL Divergence Matrix (N=30 devices)', fontweight='bold')
    ax1.set_xlabel('Device ID')
    ax1.set_ylabel('Device ID')
    
    # 2. Within vs between class distributions
    ax2 = axes[0, 1]
    ax2.boxplot([all_within, all_between], labels=['Within-class', 'Between-class'])
    ax2.set_ylabel('KL Divergence')
    ax2.set_title('KL Distribution: Within vs Between Class', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Correlation scatter plot
    ax3 = axes[1, 0]
    ax3.scatter(correlation_results['avg_kl'], correlation_results['accuracies'], alpha=0.6)
    ax3.set_xlabel('Average KL Divergence')
    ax3.set_ylabel('Classification Accuracy')
    ax3.set_title(f'KL vs Accuracy (r={correlation_results["pearson_r"]:.3f}, p={correlation_results["pearson_p"]:.3f})', 
                  fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(correlation_results['avg_kl'], correlation_results['accuracies'], 1)
    p = np.poly1d(z)
    ax3.plot(correlation_results['avg_kl'], p(correlation_results['avg_kl']), "r--", alpha=0.5)
    
    # 4. Per-class KL distributions
    ax4 = axes[1, 1]
    class_pairs = list(kl_stats['within_class'].keys()) + list(kl_stats['between_class'].keys())
    means = [kl_stats['within_class'][k]['mean'] for k in kl_stats['within_class']] + \
            [kl_stats['between_class'][k]['mean'] for k in kl_stats['between_class']]
    stds = [kl_stats['within_class'][k]['std'] for k in kl_stats['within_class']] + \
           [kl_stats['between_class'][k]['std'] for k in kl_stats['between_class']]
    
    x_pos = np.arange(len(class_pairs))
    ax4.bar(x_pos, means, yerr=stds, alpha=0.7, color=['blue']*3 + ['red']*3)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(class_pairs, rotation=45)
    ax4.set_ylabel('Mean KL Divergence')
    ax4.set_title('Mean KL by Class Pair', fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(str(FIGURES_DIR / 'qgan_tournament_validation_N30.png'), dpi=150, bbox_inches='tight')
    print("Visualization saved to: qgan_tournament_validation_N30.png")
    print()
    
    print("="*80)
    print("VALIDATION COMPLETE")
    print("="*80)
    
    return results


if __name__ == "__main__":
    results = main()
