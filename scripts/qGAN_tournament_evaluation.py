"""
qGAN Tournament Evaluation for Device Distinguishability

This script implements a tournament-style evaluation where qGAN is used to:
1. Train on each device pair (1v2, 1v3, 2v3)
2. Measure final KL divergence after fixed epochs
3. Use KL divergence as a "distinguishability score"

Hypothesis: Lower KL = devices are more similar (harder to distinguish)
           Higher KL = devices are more different (easier to distinguish)

This provides an unsupervised metric to complement supervised classification accuracy.
"""

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
import matplotlib.pyplot as plt
from scipy.stats import entropy
import json
from datetime import datetime
import time

# Reproducibility
np.random.seed(42)
torch.manual_seed(42)

print("="*80)
print("qGAN TOURNAMENT EVALUATION")
print("="*80)
print()

# Load data
print("Loading data from AI_2qubits_training_data.txt...")
with open('AI_2qubits_training_data.txt', 'r') as file:
    data = file.readlines()

# Split into three device datasets
device_1_data = data[:2000]
device_2_data = data[2001:4000]
device_3_data = data[4001:6000]

print(f"Device 1: {len(device_1_data)} samples")
print(f"Device 2: {len(device_2_data)} samples")
print(f"Device 3: {len(device_3_data)} samples")
print()

def extract_bit_frequencies(device_data):
    """Extract per-position bit frequencies from device data"""
    delimiter = ' '
    first_column = [row.split(delimiter)[0] for row in device_data]
    
    frequencies = np.zeros(64)
    for string in device_data:
        for i, bit in enumerate(string[:-39]):  # First 64 bits
            if bit == '1':
                frequencies[i] += 1
    frequencies = frequencies / len(device_data)
    
    return frequencies

def create_grid_distribution(freq1, freq2):
    """Create 2D grid distribution from two frequency vectors"""
    grid_data = []
    for i in range(64):
        grid_data.append([])
        for j in range(64):
            grid_data[i].append(freq2[j] - freq1[i])
    
    grid_data = np.array(grid_data) - np.min(grid_data)
    # Normalize to probability distribution
    grid_data = grid_data / np.sum(grid_data)
    
    return grid_data

# Extract frequencies for each device
print("Extracting bit frequency distributions...")
freq_1 = extract_bit_frequencies(device_1_data)
freq_2 = extract_bit_frequencies(device_2_data)
freq_3 = extract_bit_frequencies(device_3_data)

print(f"Device 1 mean frequency: {freq_1.mean():.4f}")
print(f"Device 2 mean frequency: {freq_2.mean():.4f}")
print(f"Device 3 mean frequency: {freq_3.mean():.4f}")
print()

# Simple Discriminator (Classical NN)
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

# Simple Generator (for simplicity, using classical NN instead of quantum)
class Generator(nn.Module):
    def __init__(self, output_size):
        super(Generator, self).__init__()
        self.linear1 = nn.Linear(output_size, 128)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(128, 256)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(256, output_size)
        self.softmax = nn.Softmax(dim=0)
        
    def forward(self):
        # Generator takes no input, learns to generate distribution
        x = torch.randn(4096)  # 64x64 = 4096
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

def train_qgan(real_distribution, n_epochs=100, lr=0.01):
    """
    Train qGAN to match real_distribution
    Returns: final KL divergence, training history
    """
    num_qnn_outputs = 4096  # 64x64 grid
    device = torch.device("cpu")
    
    # Initialize models
    generator = Generator(num_qnn_outputs).to(device)
    discriminator = Discriminator(2).to(device)  # 2D input (x,y coordinates)
    
    # Optimizers
    b1, b2 = 0.7, 0.999
    generator_optimizer = Adam(generator.parameters(), lr=lr, betas=(b1, b2), weight_decay=0.005)
    discriminator_optimizer = Adam(discriminator.parameters(), lr=lr, betas=(b1, b2), weight_decay=0.005)
    
    # Prepare data
    real_dist = torch.tensor(real_distribution.reshape(-1, 1), dtype=torch.float32).to(device)
    coords = np.linspace(0, 1, 64)
    grid_elements = np.transpose([np.tile(coords, len(coords)), np.repeat(coords, len(coords))])
    samples = torch.tensor(grid_elements, dtype=torch.float32).to(device)
    
    valid = torch.ones(num_qnn_outputs, 1, dtype=torch.float32).to(device)
    fake = torch.zeros(num_qnn_outputs, 1, dtype=torch.float32).to(device)
    
    # Training history
    history = {
        'generator_loss': [],
        'discriminator_loss': [],
        'kl_divergence': []
    }
    
    start_time = time.time()
    
    for epoch in range(n_epochs):
        # Generate distribution
        gen_dist = generator().reshape(-1, 1)
        disc_value = discriminator(samples)
        
        # Train Generator
        generator_optimizer.zero_grad()
        generator_loss = adversarial_loss(disc_value, valid, gen_dist)
        generator_loss.backward(retain_graph=True)
        generator_optimizer.step()
        
        # Train Discriminator
        discriminator_optimizer.zero_grad()
        real_loss = adversarial_loss(disc_value, valid, real_dist)
        fake_loss = adversarial_loss(disc_value, fake, gen_dist.detach())
        discriminator_loss = (real_loss + fake_loss) / 2
        discriminator_loss.backward()
        discriminator_optimizer.step()
        
        # Calculate KL divergence
        gen_dist_np = gen_dist.detach().squeeze().numpy()
        real_dist_np = real_dist.squeeze().numpy()
        
        # Ensure no zeros for KL divergence calculation
        gen_dist_np = np.clip(gen_dist_np, 1e-10, 1.0)
        real_dist_np = np.clip(real_dist_np, 1e-10, 1.0)
        
        # Normalize
        gen_dist_np = gen_dist_np / np.sum(gen_dist_np)
        real_dist_np = real_dist_np / np.sum(real_dist_np)
        
        kl_div = entropy(real_dist_np, gen_dist_np)
        
        # Record history
        history['generator_loss'].append(generator_loss.item())
        history['discriminator_loss'].append(discriminator_loss.item())
        history['kl_divergence'].append(kl_div)
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs} | G Loss: {generator_loss.item():.4f} | "
                  f"D Loss: {discriminator_loss.item():.4f} | KL: {kl_div:.4f}")
    
    elapsed = time.time() - start_time
    final_kl = history['kl_divergence'][-1]
    
    print(f"  Training completed in {elapsed:.2f}s | Final KL: {final_kl:.4f}")
    
    return final_kl, history

def run_tournament(n_epochs=100):
    """
    Run tournament-style evaluation:
    Train qGAN on each device pair and measure distinguishability via KL divergence
    """
    
    print("="*80)
    print("TOURNAMENT EVALUATION")
    print("="*80)
    print()
    print(f"Training qGAN for {n_epochs} epochs on each device pair...")
    print(f"Lower KL = More similar distributions (harder to distinguish)")
    print(f"Higher KL = More different distributions (easier to distinguish)")
    print()
    
    results = {}
    
    # Match 1: Device 1 vs Device 2
    print("-" * 80)
    print("MATCH 1: Device 1 vs Device 2")
    print("-" * 80)
    grid_1v2 = create_grid_distribution(freq_1, freq_2)
    kl_1v2, history_1v2 = train_qgan(grid_1v2, n_epochs=n_epochs)
    results['1v2'] = {'kl': kl_1v2, 'history': history_1v2}
    print()
    
    # Match 2: Device 1 vs Device 3
    print("-" * 80)
    print("MATCH 2: Device 1 vs Device 3")
    print("-" * 80)
    grid_1v3 = create_grid_distribution(freq_1, freq_3)
    kl_1v3, history_1v3 = train_qgan(grid_1v3, n_epochs=n_epochs)
    results['1v3'] = {'kl': kl_1v3, 'history': history_1v3}
    print()
    
    # Match 3: Device 2 vs Device 3
    print("-" * 80)
    print("MATCH 3: Device 2 vs Device 3")
    print("-" * 80)
    grid_2v3 = create_grid_distribution(freq_2, freq_3)
    kl_2v3, history_2v3 = train_qgan(grid_2v3, n_epochs=n_epochs)
    results['2v3'] = {'kl': kl_2v3, 'history': history_2v3}
    print()
    
    return results

def analyze_results(results):
    """Analyze tournament results and draw conclusions"""
    
    print("="*80)
    print("TOURNAMENT RESULTS SUMMARY")
    print("="*80)
    print()
    
    print("Final KL Divergence Scores (after training):")
    print("-" * 80)
    kl_scores = {
        'Device 1 vs 2': results['1v2']['kl'],
        'Device 1 vs 3': results['1v3']['kl'],
        'Device 2 vs 3': results['2v3']['kl']
    }
    
    for pair, kl in kl_scores.items():
        print(f"  {pair:20s}: KL = {kl:8.4f}")
    
    print()
    print("Distinguishability Ranking (higher KL = more distinguishable):")
    print("-" * 80)
    sorted_pairs = sorted(kl_scores.items(), key=lambda x: x[1], reverse=True)
    for rank, (pair, kl) in enumerate(sorted_pairs, 1):
        interpretation = ""
        if kl > 10:
            interpretation = "HIGHLY DISTINGUISHABLE"
        elif kl > 5:
            interpretation = "MODERATELY DISTINGUISHABLE"
        elif kl > 1:
            interpretation = "SOMEWHAT DISTINGUISHABLE"
        else:
            interpretation = "DIFFICULT TO DISTINGUISH"
        
        print(f"  {rank}. {pair:20s}: KL = {kl:8.4f}  [{interpretation}]")
    
    print()
    print("Interpretation:")
    print("-" * 80)
    
    most_distinguishable = sorted_pairs[0]
    least_distinguishable = sorted_pairs[-1]
    
    print(f"Most distinguishable pair: {most_distinguishable[0]} (KL={most_distinguishable[1]:.4f})")
    print(f"  -> These devices have the most different statistical signatures")
    print(f"  -> Classification NN should achieve highest accuracy on this pair")
    print()
    print(f"Least distinguishable pair: {least_distinguishable[0]} (KL={least_distinguishable[1]:.4f})")
    print(f"  -> These devices have similar statistical signatures")
    print(f"  -> Classification NN may struggle more on this pair")
    print()
    
    # Compare with classification results (from previous analysis)
    print("Cross-validation with Classification Results:")
    print("-" * 80)
    print("From ML_solution.ipynb:")
    print("  Device 1 accuracy: 66.7%")
    print("  Device 2 accuracy: 65.0%")
    print("  Device 3 accuracy: 70.0% (easiest to identify)")
    print()
    print("Device 3 being easiest to classify suggests it has most distinctive signature.")
    if 'Device 2 vs 3' in [p[0] for p in sorted_pairs[:1]]:
        print("✓ VALIDATED: qGAN tournament confirms Device 2 vs 3 most distinguishable")
    else:
        print("⚠ MISMATCH: qGAN ranking differs from classification results")
    
    print()
    return kl_scores

def plot_results(results):
    """Create visualization of tournament results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: KL Divergence Comparison
    ax1 = axes[0, 0]
    pairs = ['Device 1 vs 2', 'Device 1 vs 3', 'Device 2 vs 3']
    kl_values = [results['1v2']['kl'], results['1v3']['kl'], results['2v3']['kl']]
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    bars = ax1.bar(pairs, kl_values, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('KL Divergence (Final)', fontsize=12, fontweight='bold')
    ax1.set_title('qGAN Tournament: Device Distinguishability', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars, kl_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Training History - KL Divergence
    ax2 = axes[0, 1]
    ax2.plot(results['1v2']['history']['kl_divergence'], label='Device 1 vs 2', color='#3498db', linewidth=2)
    ax2.plot(results['1v3']['history']['kl_divergence'], label='Device 1 vs 3', color='#e74c3c', linewidth=2)
    ax2.plot(results['2v3']['history']['kl_divergence'], label='Device 2 vs 3', color='#2ecc71', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('KL Divergence', fontsize=12, fontweight='bold')
    ax2.set_title('qGAN Training: KL Divergence Convergence', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Plot 3: Generator Loss
    ax3 = axes[1, 0]
    ax3.plot(results['1v2']['history']['generator_loss'], label='Device 1 vs 2', color='#3498db', alpha=0.7)
    ax3.plot(results['1v3']['history']['generator_loss'], label='Device 1 vs 3', color='#e74c3c', alpha=0.7)
    ax3.plot(results['2v3']['history']['generator_loss'], label='Device 2 vs 3', color='#2ecc71', alpha=0.7)
    ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Generator Loss', fontsize=12, fontweight='bold')
    ax3.set_title('qGAN Training: Generator Loss', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # Plot 4: Discriminator Loss
    ax4 = axes[1, 1]
    ax4.plot(results['1v2']['history']['discriminator_loss'], label='Device 1 vs 2', color='#3498db', alpha=0.7)
    ax4.plot(results['1v3']['history']['discriminator_loss'], label='Device 1 vs 3', color='#e74c3c', alpha=0.7)
    ax4.plot(results['2v3']['history']['discriminator_loss'], label='Device 2 vs 3', color='#2ecc71', alpha=0.7)
    ax4.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Discriminator Loss', fontsize=12, fontweight='bold')
    ax4.set_title('qGAN Training: Discriminator Loss', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fig_qgan_tournament_results.png', dpi=300, bbox_inches='tight')
    print(f"Figure saved: fig_qgan_tournament_results.png")
    plt.close()

def save_results(results, kl_scores):
    """Save results to JSON file"""
    
    output = {
        'timestamp': datetime.now().isoformat(),
        'methodology': 'qGAN Tournament Evaluation',
        'description': 'KL divergence after training qGAN on each device pair',
        'interpretation': 'Higher KL = more distinguishable devices',
        'results': {
            'device_1_vs_2': {
                'kl_divergence': float(results['1v2']['kl']),
                'final_generator_loss': float(results['1v2']['history']['generator_loss'][-1]),
                'final_discriminator_loss': float(results['1v2']['history']['discriminator_loss'][-1])
            },
            'device_1_vs_3': {
                'kl_divergence': float(results['1v3']['kl']),
                'final_generator_loss': float(results['1v3']['history']['generator_loss'][-1]),
                'final_discriminator_loss': float(results['1v3']['history']['discriminator_loss'][-1])
            },
            'device_2_vs_3': {
                'kl_divergence': float(results['2v3']['kl']),
                'final_generator_loss': float(results['2v3']['history']['generator_loss'][-1]),
                'final_discriminator_loss': float(results['2v3']['history']['discriminator_loss'][-1])
            }
        },
        'ranking': [
            {'pair': pair, 'kl_divergence': float(kl)} 
            for pair, kl in sorted(kl_scores.items(), key=lambda x: x[1], reverse=True)
        ]
    }
    
    with open('qgan_tournament_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Results saved: qgan_tournament_results.json")

# Main execution
if __name__ == "__main__":
    # Run tournament with configurable epochs
    N_EPOCHS = 100  # Adjust as needed (1000 for full run, 100 for quick test)
    
    print(f"Running qGAN tournament with {N_EPOCHS} epochs per match...")
    print("Note: For full evaluation, use 1000 epochs (this may take significant time)")
    print()
    
    results = run_tournament(n_epochs=N_EPOCHS)
    kl_scores = analyze_results(results)
    plot_results(results)
    save_results(results, kl_scores)
    
    print()
    print("="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print()
    print("Outputs generated:")
    print("  1. fig_qgan_tournament_results.png - Visualization of all results")
    print("  2. qgan_tournament_results.json - Detailed numerical results")
    print()
    print("Key Takeaway:")
    print("  qGAN tournament provides unsupervised distinguishability metric")
    print("  Complements supervised classification accuracy (58.67%)")
    print("  Higher KL divergence indicates devices are more distinguishable")
