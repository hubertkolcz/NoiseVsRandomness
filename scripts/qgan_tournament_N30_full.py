"""
Full qGAN Tournament on N=30 Synthetic Devices

This script runs ACTUAL qGAN training (Generator + Discriminator) on all 435 
device pairs from the N=30 synthetic dataset, genuinely validating the qGAN 
methodology (not just direct KL calculation).

WARNING: This is computationally expensive - approximately:
- 435 device pairs × 100 epochs × ~10 seconds = ~12 hours of computation
- Requires GPU for reasonable runtime
- Output saved incrementally to avoid data loss
"""

import os
import sys

# Fix Intel Fortran/MKL threading issues
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from scipy.stats import entropy
import json
from datetime import datetime
import time
from pathlib import Path
import pickle

# Setup paths
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = ROOT_DIR / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# Progress file for resuming
PROGRESS_FILE = RESULTS_DIR / "qgan_N30_tournament_progress.pkl"
RESULTS_FILE = RESULTS_DIR / "qgan_N30_tournament_results.json"

# Set seeds
np.random.seed(42)
torch.manual_seed(42)

# Device selection
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = True

print("="*80)
print("FULL qGAN TOURNAMENT ON N=30 DEVICES")
print("="*80)
print(f"Device: {device}")
print(f"Estimated time: ~12 hours (435 pairs × 100 epochs)")
print(f"Progress will be saved to: {PROGRESS_FILE}")
print()

# ============================================================================
# DATA GENERATION (reuse from validate_qgan_tournament_N30.py)
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
    
    return samples

def create_synthetic_dataset(n_devices_per_class=10):
    """Create N=30 device dataset with 3 bias classes"""
    X = []
    y = []
    device_metadata = []
    device_id = 0
    
    # Class 0: Low bias
    for i in range(n_devices_per_class):
        bias = np.random.uniform(0.48, 0.52)
        samples = generate_synthetic_rng(
            n_samples=2000, bias_level=bias,
            temporal_correlation=np.random.uniform(0.0, 0.05),
            drift=np.random.uniform(-0.01, 0.01)
        )
        X.append(samples)
        y.extend([0] * len(samples))
        device_metadata.append({'device_id': device_id, 'class': 0, 'bias': bias})
        device_id += 1
    
    # Class 1: Medium bias
    for i in range(n_devices_per_class):
        bias = np.random.uniform(0.54, 0.58)
        samples = generate_synthetic_rng(
            n_samples=2000, bias_level=bias,
            temporal_correlation=np.random.uniform(0.0, 0.1),
            drift=np.random.uniform(-0.02, 0.02)
        )
        X.append(samples)
        y.extend([1] * len(samples))
        device_metadata.append({'device_id': device_id, 'class': 1, 'bias': bias})
        device_id += 1
    
    # Class 2: High bias
    for i in range(n_devices_per_class):
        bias = np.random.uniform(0.60, 0.65)
        samples = generate_synthetic_rng(
            n_samples=2000, bias_level=bias,
            temporal_correlation=np.random.uniform(0.0, 0.15),
            drift=np.random.uniform(-0.03, 0.03)
        )
        X.append(samples)
        y.extend([2] * len(samples))
        device_metadata.append({'device_id': device_id, 'class': 2, 'bias': bias})
        device_id += 1
    
    return np.vstack(X), np.array(y), device_metadata

# ============================================================================
# qGAN IMPLEMENTATION (from qGAN_tournament_evaluation.py)
# ============================================================================

class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        self.linear_input = nn.Linear(input_size, 20)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.linear20 = nn.Linear(20, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        x = self.linear_input(input_tensor)
        x = self.leaky_relu(x)
        x = self.linear20(x)
        x = self.sigmoid(x)
        return x

class Generator(nn.Module):
    def __init__(self, output_size, device=None):
        super(Generator, self).__init__()
        self.device = device if device is not None else torch.device('cpu')
        self.output_size = output_size
        self.linear1 = nn.Linear(output_size, 128)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(128, 256)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(256, output_size)
        self.softmax = nn.Softmax(dim=0)
        
    def forward(self):
        x = torch.randn(self.output_size, device=self.device)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.softmax(x)
        return x

def adversarial_loss(input_val, target, weights):
    """Weighted binary cross entropy loss"""
    bce_loss = target * torch.log(input_val + 1e-8) + (1 - target) * torch.log(1 - input_val + 1e-8)
    weighted_loss = weights * bce_loss
    total_loss = -torch.sum(weighted_loss)
    return total_loss

def extract_bit_frequencies(samples):
    """Extract per-position bit frequencies (first 64 bits)"""
    frequencies = np.zeros(64)
    for sample in samples:
        for i in range(64):
            if sample[i] == 1:
                frequencies[i] += 1
    frequencies = frequencies / len(samples)
    return frequencies

def create_grid_distribution(freq1, freq2):
    """Create 2D grid distribution from two frequency vectors"""
    grid_data = []
    for i in range(64):
        grid_data.append([])
        for j in range(64):
            grid_data[i].append(freq2[j] - freq1[i])
    
    grid_data = np.array(grid_data) - np.min(grid_data)
    grid_data = grid_data / np.sum(grid_data)
    return grid_data

def train_qgan(real_distribution, n_epochs=100, lr=0.01, verbose=False):
    """Train qGAN to match real_distribution"""
    num_qnn_outputs = 4096  # 64x64 grid
    use_amp = torch.cuda.is_available()
    
    generator = Generator(num_qnn_outputs, device=device).to(device)
    discriminator = Discriminator(2).to(device)
    
    b1, b2 = 0.7, 0.999
    generator_optimizer = Adam(generator.parameters(), lr=lr, betas=(b1, b2), weight_decay=0.005)
    discriminator_optimizer = Adam(discriminator.parameters(), lr=lr, betas=(b1, b2), weight_decay=0.005)
    
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    
    real_dist = torch.tensor(real_distribution.reshape(-1, 1), dtype=torch.float32).to(device, non_blocking=use_amp)
    coords = np.linspace(0, 1, 64)
    grid_elements = np.transpose([np.tile(coords, len(coords)), np.repeat(coords, len(coords))])
    samples = torch.tensor(grid_elements, dtype=torch.float32).to(device, non_blocking=use_amp)
    
    valid = torch.ones(num_qnn_outputs, 1, dtype=torch.float32).to(device, non_blocking=use_amp)
    fake = torch.zeros(num_qnn_outputs, 1, dtype=torch.float32).to(device, non_blocking=use_amp)
    
    for epoch in range(n_epochs):
        if use_amp:
            with torch.amp.autocast('cuda'):
                gen_dist = generator().reshape(-1, 1)
                disc_value = discriminator(samples)
                generator_optimizer.zero_grad()
                generator_loss = adversarial_loss(disc_value, valid, gen_dist)
            
            scaler.scale(generator_loss).backward(retain_graph=True)
            scaler.step(generator_optimizer)
            scaler.update()
            
            discriminator_optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                real_loss = adversarial_loss(disc_value, valid, real_dist)
                fake_loss = adversarial_loss(disc_value, fake, gen_dist.detach())
                discriminator_loss = (real_loss + fake_loss) / 2
            
            scaler.scale(discriminator_loss).backward()
            scaler.step(discriminator_optimizer)
            scaler.update()
        else:
            gen_dist = generator().reshape(-1, 1)
            disc_value = discriminator(samples)
            
            generator_optimizer.zero_grad()
            generator_loss = adversarial_loss(disc_value, valid, gen_dist)
            generator_loss.backward(retain_graph=True)
            generator_optimizer.step()
            
            discriminator_optimizer.zero_grad()
            real_loss = adversarial_loss(disc_value, valid, real_dist)
            fake_loss = adversarial_loss(disc_value, fake, gen_dist.detach())
            discriminator_loss = (real_loss + fake_loss) / 2
            discriminator_loss.backward()
            discriminator_optimizer.step()
    
    # Calculate final KL divergence
    gen_dist_np = gen_dist.detach().cpu().squeeze().numpy()
    real_dist_np = real_dist.cpu().squeeze().numpy()
    
    gen_dist_np = np.clip(gen_dist_np, 1e-10, 1.0)
    real_dist_np = np.clip(real_dist_np, 1e-10, 1.0)
    
    gen_dist_np = gen_dist_np / np.sum(gen_dist_np)
    real_dist_np = real_dist_np / np.sum(real_dist_np)
    
    kl_div = entropy(real_dist_np, gen_dist_np)
    
    return kl_div

# ============================================================================
# MAIN TOURNAMENT
# ============================================================================

def run_full_tournament():
    """Run qGAN training on all 435 device pairs"""
    
    # Load or create progress
    if PROGRESS_FILE.exists():
        print("Loading existing progress...")
        with open(PROGRESS_FILE, 'rb') as f:
            progress = pickle.load(f)
        X = progress['X']
        y = progress['y']
        device_metadata = progress['device_metadata']
        kl_matrix = progress['kl_matrix']
        completed_pairs = progress['completed_pairs']
        print(f"Resuming from {completed_pairs} completed pairs")
    else:
        print("Generating N=30 synthetic dataset...")
        X, y, device_metadata = create_synthetic_dataset(n_devices_per_class=10)
        n_devices = len(device_metadata)
        kl_matrix = np.zeros((n_devices, n_devices))
        completed_pairs = 0
        print(f"Total devices: {n_devices}")
        print(f"Total samples: {len(X)}")
    
    n_devices = len(device_metadata)
    total_pairs = n_devices * (n_devices - 1) // 2
    
    # Group samples by device
    samples_per_device = len(X) // n_devices
    device_samples = []
    for i in range(n_devices):
        start_idx = i * samples_per_device
        end_idx = (i + 1) * samples_per_device
        device_samples.append(X[start_idx:end_idx])
    
    # Extract bit frequencies for all devices
    print("Extracting bit frequencies...")
    device_frequencies = []
    for samples in device_samples:
        freq = extract_bit_frequencies(samples)
        device_frequencies.append(freq)
    
    print()
    print(f"Starting qGAN tournament: {total_pairs} device pairs")
    print(f"Progress will be saved every 10 pairs")
    print()
    
    start_time = time.time()
    
    for i in range(n_devices):
        for j in range(i+1, n_devices):
            pair_idx = completed_pairs + 1
            
            # Skip if already computed
            if kl_matrix[i, j] != 0:
                continue
            
            # Create grid distribution for this pair
            grid_dist = create_grid_distribution(device_frequencies[i], device_frequencies[j])
            
            # Train qGAN
            pair_start = time.time()
            kl_ij = train_qgan(grid_dist, n_epochs=100, lr=0.01, verbose=False)
            pair_time = time.time() - pair_start
            
            # Store result (symmetric)
            kl_matrix[i, j] = kl_ij
            kl_matrix[j, i] = kl_ij
            completed_pairs += 1
            
            # Progress update
            elapsed = time.time() - start_time
            avg_time_per_pair = elapsed / completed_pairs
            remaining_pairs = total_pairs - completed_pairs
            eta_seconds = remaining_pairs * avg_time_per_pair
            eta_hours = eta_seconds / 3600
            
            print(f"[{completed_pairs}/{total_pairs}] Device {i} vs {j}: KL={kl_ij:.4f} | "
                  f"Pair time: {pair_time:.1f}s | ETA: {eta_hours:.1f}h")
            
            # Save progress every 10 pairs
            if completed_pairs % 10 == 0:
                progress = {
                    'X': X,
                    'y': y,
                    'device_metadata': device_metadata,
                    'kl_matrix': kl_matrix,
                    'completed_pairs': completed_pairs,
                    'timestamp': datetime.now().isoformat()
                }
                with open(PROGRESS_FILE, 'wb') as f:
                    pickle.dump(progress, f)
                print(f"  -> Progress saved ({completed_pairs}/{total_pairs} pairs)")
    
    total_time = time.time() - start_time
    print()
    print("="*80)
    print(f"TOURNAMENT COMPLETE!")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Average time per pair: {total_time/total_pairs:.1f} seconds")
    print("="*80)
    
    # Save final results
    results = {
        'timestamp': datetime.now().isoformat(),
        'method': 'full_qgan_training',
        'description': 'Complete qGAN tournament with Generator+Discriminator training on 64x64 difference grids',
        'n_devices': n_devices,
        'n_pairs': total_pairs,
        'total_time_hours': total_time / 3600,
        'device_metadata': device_metadata,
        'kl_matrix': kl_matrix.tolist(),
        'statistics': {
            'mean_kl': float(np.mean(kl_matrix[np.triu_indices(n_devices, k=1)])),
            'std_kl': float(np.std(kl_matrix[np.triu_indices(n_devices, k=1)])),
            'min_kl': float(np.min(kl_matrix[kl_matrix > 0])),
            'max_kl': float(np.max(kl_matrix))
        }
    }
    
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {RESULTS_FILE}")
    
    # Clean up progress file
    if PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()
    
    return results

if __name__ == "__main__":
    try:
        results = run_full_tournament()
        print("\nSUCCESS: Full qGAN tournament completed!")
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Progress has been saved.")
        print(f"Resume by running this script again.")
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        print(f"\nProgress saved to: {PROGRESS_FILE}")
