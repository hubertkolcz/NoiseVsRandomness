"""
Generate multi-seed runs for all baseline models to compute variance estimates
Trains each model with 5 different seeds and saves mean±std per model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import json
from datetime import datetime
from pathlib import Path
import os

# Setup
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
DATA_FILE = ROOT_DIR / "AI_2qubits_training_data.txt"
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

def set_all_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Load data
print("Loading data from AI_2qubits_training_data.txt...")
data = []
labels = []
with open(DATA_FILE, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 2:
            bitstring = parts[0]
            label = int(parts[1]) - 1
            bits = [int(b) for b in bitstring]
            data.append(bits)
            labels.append(label)

data = np.array(data, dtype=np.float32)
labels = np.array(labels, dtype=np.int64)
print(f"Loaded {len(data)} samples with {data.shape[1]} features\n")

# Define model architectures
class Net_20_20_3(nn.Module):
    """Baseline: 20-20-3 with L2 regularization"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 3)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class Net_20_10_3(nn.Module):
    """Compressed: 20-10-3"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 3)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class Net_30_20_3(nn.Module):
    """Best architecture: 30-20-3 with dropout"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 30)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(30, 20)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(20, 3)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.fc3(x)

def train_model(model, train_loader, test_loader, config, model_name, seed):
    """Train a model with given config"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    if 'l2_lambda' in config:
        optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['l2_lambda'])
    else:
        optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    
    best_test_acc = 0
    
    for epoch in range(config['epochs']):
        model.train()
        for inputs, labels_batch in train_loader:
            inputs, labels_batch = inputs.to(device), labels_batch.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels_batch)
            
            if 'l1_lambda' in config and config['l1_lambda'] > 0:
                l1_norm = sum(p.abs().sum() for p in model.parameters())
                loss += config['l1_lambda'] * l1_norm
            
            loss.backward()
            optimizer.step()
        
        # Evaluate
        if (epoch + 1) % 20 == 0 or epoch == config['epochs'] - 1:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels_batch in test_loader:
                    inputs, labels_batch = inputs.to(device), labels_batch.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    total += labels_batch.size(0)
                    correct += (predicted == labels_batch).sum().item()
            
            test_acc = correct / total
            best_test_acc = max(best_test_acc, test_acc)
    
    return best_test_acc

def count_parameters(model):
    """Count total parameters"""
    return sum(p.numel() for p in model.parameters())

# Multi-seed configuration
NUM_SEEDS = 5
BASE_SEED = 100
SEEDS = [BASE_SEED + i for i in range(NUM_SEEDS)]

results = {
    'metadata': {
        'timestamp': datetime.now().isoformat(),
        'dataset': 'AI_2qubits_training_data.txt',
        'total_samples': len(data),
        'num_seeds': NUM_SEEDS,
        'seeds': SEEDS
    },
    'models': []
}

X_tensor = torch.FloatTensor(data)
y_tensor = torch.LongTensor(labels)
dataset = TensorDataset(X_tensor, y_tensor)

# Model 1: Random Baseline (theoretical, no variance)
print("\n" + "="*60)
print("Model 1: Random Baseline (Theoretical)")
print("="*60)
random_acc = 1/3
results['models'].append({
    'name': 'Random Baseline',
    'architecture': 'None',
    'mean_accuracy': random_acc,
    'std_accuracy': 0.0,
    'accuracies': [random_acc] * NUM_SEEDS,
    'parameters': 0
})

# Model 2: NN (20-20-3) with L2, 40 epochs
print("\n" + "="*60)
print("Model 2: NN (20-20-3) L2 (5 seeds)")
print("="*60)
config2 = {'epochs': 40, 'lr': 0.001, 'l2_lambda': 0.0005}
accs2 = []
for seed in SEEDS:
    set_all_seeds(seed)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8)
    
    model2 = Net_20_20_3()
    acc = train_model(model2, train_loader, test_loader, config2, "NN (20-20-3) L2", seed)
    accs2.append(acc)
    print(f"  Seed {seed}: {acc*100:.2f}%")

results['models'].append({
    'name': 'NN (20-20-3) L2',
    'architecture': '100->20->20->3',
    'mean_accuracy': float(np.mean(accs2)),
    'std_accuracy': float(np.std(accs2, ddof=1)),
    'accuracies': [float(a) for a in accs2],
    'parameters': count_parameters(Net_20_20_3())
})
print(f"Mean: {np.mean(accs2)*100:.2f}% ± {np.std(accs2, ddof=1)*100:.2f}%")

# Model 3: NN (20-10-3) Limited, 300 epochs
print("\n" + "="*60)
print("Model 3: NN (20-10-3) Limited (5 seeds)")
print("="*60)
config3 = {'epochs': 300, 'lr': 0.001, 'l1_lambda': 0.001}
accs3 = []
for seed in SEEDS:
    set_all_seeds(seed)
    train_size = int(0.7 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8)
    
    model3 = Net_20_10_3()
    acc = train_model(model3, train_loader, test_loader, config3, "NN (20-10-3) Limited", seed)
    accs3.append(acc)
    print(f"  Seed {seed}: {acc*100:.2f}%")

results['models'].append({
    'name': 'NN (20-10-3) Limited',
    'architecture': '100->20->10->3',
    'mean_accuracy': float(np.mean(accs3)),
    'std_accuracy': float(np.std(accs3, ddof=1)),
    'accuracies': [float(a) for a in accs3],
    'parameters': count_parameters(Net_20_10_3())
})
print(f"Mean: {np.mean(accs3)*100:.2f}% ± {np.std(accs3, ddof=1)*100:.2f}%")

# Model 4: NN (30-20-3) Batch=4, 100 epochs
print("\n" + "="*60)
print("Model 4: NN (30-20-3) Batch=4 (5 seeds)")
print("="*60)
config4 = {'epochs': 100, 'lr': 0.001, 'l1_lambda': 0.002}
accs4 = []
for seed in SEEDS:
    set_all_seeds(seed)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4)
    
    model4 = Net_30_20_3()
    acc = train_model(model4, train_loader, test_loader, config4, "NN (30-20-3) Batch=4", seed)
    accs4.append(acc)
    print(f"  Seed {seed}: {acc*100:.2f}%")

results['models'].append({
    'name': 'NN (30-20-3) Batch=4',
    'architecture': '100->30->20->3',
    'mean_accuracy': float(np.mean(accs4)),
    'std_accuracy': float(np.std(accs4, ddof=1)),
    'accuracies': [float(a) for a in accs4],
    'parameters': count_parameters(Net_30_20_3())
})
print(f"Mean: {np.mean(accs4)*100:.2f}% ± {np.std(accs4, ddof=1)*100:.2f}%")

# Model 5: Logistic Regression (70-30 split)
print("\n" + "="*60)
print("Model 5: Logistic Regression (5 seeds)")
print("="*60)
accs5 = []
for seed in SEEDS:
    set_all_seeds(seed)
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    train_size = int(0.7 * len(data))
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]
    
    X_train_lr = data[train_idx]
    y_train_lr = labels[train_idx]
    X_test_lr = data[test_idx]
    y_test_lr = labels[test_idx]
    
    lr_model = LogisticRegression(max_iter=1000, random_state=seed)
    lr_model.fit(X_train_lr, y_train_lr)
    y_pred_lr = lr_model.predict(X_test_lr)
    acc = accuracy_score(y_test_lr, y_pred_lr)
    accs5.append(acc)
    print(f"  Seed {seed}: {acc*100:.2f}%")

lr_params = 303  # 100*3 + 3
results['models'].append({
    'name': 'Logistic Regression',
    'architecture': '100->3 (linear)',
    'mean_accuracy': float(np.mean(accs5)),
    'std_accuracy': float(np.std(accs5, ddof=1)),
    'accuracies': [float(a) for a in accs5],
    'parameters': lr_params
})
print(f"Mean: {np.mean(accs5)*100:.2f}% ± {np.std(accs5, ddof=1)*100:.2f}%")

# Model 6: Best NN (30-20-3) with L1, 1000 epochs
print("\n" + "="*60)
print("Model 6: Best NN (30-20-3) L1 (5 seeds)")
print("="*60)
config6 = {'epochs': 1000, 'lr': 0.001, 'l1_lambda': 0.002}
accs6 = []
for seed in SEEDS:
    set_all_seeds(seed)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8)
    
    model6 = Net_30_20_3()
    acc = train_model(model6, train_loader, test_loader, config6, "Best NN (30-20-3) L1", seed)
    accs6.append(acc)
    print(f"  Seed {seed}: {acc*100:.2f}%")

results['models'].append({
    'name': 'Best NN (30-20-3) L1',
    'architecture': '100->30->20->3',
    'mean_accuracy': float(np.mean(accs6)),
    'std_accuracy': float(np.std(accs6, ddof=1)),
    'accuracies': [float(a) for a in accs6],
    'parameters': count_parameters(Net_30_20_3())
})
print(f"Mean: {np.mean(accs6)*100:.2f}% ± {np.std(accs6, ddof=1)*100:.2f}%")

# Save results
output_file = RESULTS_DIR / "multiseed_variance_estimates.json"
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*60)
print("SUMMARY: MULTI-SEED VARIANCE ESTIMATES")
print("="*60)
for model in results['models']:
    mean_pct = model['mean_accuracy'] * 100
    std_pct = model['std_accuracy'] * 100
    print(f"{model['name']:30s} | Mean: {mean_pct:6.2f}% ± {std_pct:5.2f}% | Params: {model['parameters']:6d}")

print(f"\n✓ Results saved to: {output_file}")
