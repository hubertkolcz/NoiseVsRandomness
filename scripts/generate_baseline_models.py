"""
Generate ALL baseline model results to replace hardcoded values
This script trains all 6 models shown in Panel A of Slide 9 figure
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
import sys
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
            label = int(parts[1]) - 1  # Convert 1,2,3 to 0,1,2
            bits = [int(b) for b in bitstring]
            data.append(bits)
            labels.append(label)

data = np.array(data, dtype=np.float32)
labels = np.array(labels, dtype=np.int64)
print(f"Loaded {len(data)} samples with {data.shape[1]} features\n")

# Define all model architectures
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

def train_model(model, train_loader, test_loader, config, model_name):
    """Train a model with given config"""
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"Config: {config}")
    print(f"{'='*60}")
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    # L2 regularization via weight_decay
    if 'l2_lambda' in config:
        optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['l2_lambda'])
    else:
        optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    
    best_test_acc = 0
    
    for epoch in range(config['epochs']):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Add L1 regularization if specified
            if 'l1_lambda' in config and config['l1_lambda'] > 0:
                l1_norm = sum(p.abs().sum() for p in model.parameters())
                loss += config['l1_lambda'] * l1_norm
            
            loss.backward()
            optimizer.step()
        
        # Evaluate
        if (epoch + 1) % 10 == 0 or epoch == config['epochs'] - 1:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            test_acc = correct / total
            best_test_acc = max(best_test_acc, test_acc)
            if (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch+1}/{config['epochs']}: Test Acc = {test_acc*100:.2f}%")
    
    print(f"✓ Best Test Accuracy: {best_test_acc*100:.2f}%")
    return best_test_acc

def count_parameters(model):
    """Count total parameters"""
    return sum(p.numel() for p in model.parameters())

# Run all experiments
set_all_seeds(42)

results = {
    'metadata': {
        'timestamp': datetime.now().isoformat(),
        'dataset': 'AI_2qubits_training_data.txt',
        'total_samples': len(data),
        'seed': 42
    },
    'models': []
}

# Prepare datasets
X_tensor = torch.FloatTensor(data)
y_tensor = torch.LongTensor(labels)
dataset = TensorDataset(X_tensor, y_tensor)

# Model 1: Random Baseline (theoretical)
print("\n" + "="*60)
print("Model 1: Random Baseline (Theoretical)")
print("="*60)
random_acc = 1/3  # 3-class problem
print(f"Theoretical accuracy: {random_acc*100:.2f}%")
results['models'].append({
    'name': 'Random Baseline',
    'architecture': 'None',
    'test_accuracy': random_acc,
    'parameters': 0,
    'config': {'theoretical': True}
})

# Model 2: NN (20-20-3) with L2, 40 epochs
set_all_seeds(42)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)

model2 = Net_20_20_3()
config2 = {'epochs': 40, 'lr': 0.001, 'l2_lambda': 0.0005}
acc2 = train_model(model2, train_loader, test_loader, config2, "NN (20-20-3) L2")
results['models'].append({
    'name': 'NN (20-20-3) L2',
    'architecture': '100->20->20->3',
    'test_accuracy': acc2,
    'parameters': count_parameters(model2),
    'config': config2
})

# Model 3: NN (20-10-3) Limited, 300 epochs
set_all_seeds(42)
train_size = int(0.7 * len(dataset))  # 70-30 split
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)

model3 = Net_20_10_3()
config3 = {'epochs': 300, 'lr': 0.001, 'l1_lambda': 0.001}
acc3 = train_model(model3, train_loader, test_loader, config3, "NN (20-10-3) Limited")
results['models'].append({
    'name': 'NN (20-10-3) Limited',
    'architecture': '100->20->10->3',
    'test_accuracy': acc3,
    'parameters': count_parameters(model3),
    'config': config3
})

# Model 4: NN (30-20-3) Batch=4, 100 epochs
set_all_seeds(42)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)  # Batch=4
test_loader = DataLoader(test_dataset, batch_size=4)

model4 = Net_30_20_3()
config4 = {'epochs': 100, 'lr': 0.001, 'l1_lambda': 0.002}
acc4 = train_model(model4, train_loader, test_loader, config4, "NN (30-20-3) Batch=4")
results['models'].append({
    'name': 'NN (30-20-3) Batch=4',
    'architecture': '100->30->20->3',
    'test_accuracy': acc4,
    'parameters': count_parameters(model4),
    'config': config4
})

# Model 5: Logistic Regression (70-30 split)
set_all_seeds(42)
indices = np.arange(len(data))
np.random.shuffle(indices)
train_size = int(0.7 * len(data))
train_idx = indices[:train_size]
test_idx = indices[train_size:]

X_train_lr = data[train_idx]
y_train_lr = labels[train_idx]
X_test_lr = data[test_idx]
y_test_lr = labels[test_idx]

print("\n" + "="*60)
print("Training: Logistic Regression (70-30 split)")
print("="*60)
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_lr, y_train_lr)
y_pred_lr = lr_model.predict(X_test_lr)
acc5 = accuracy_score(y_test_lr, y_pred_lr)
print(f"✓ Test Accuracy: {acc5*100:.2f}%")

lr_params = lr_model.coef_.size + lr_model.intercept_.size
results['models'].append({
    'name': 'Logistic Regression',
    'architecture': '100->3 (linear)',
    'test_accuracy': acc5,
    'parameters': lr_params,
    'config': {'max_iter': 1000, 'split': '70-30'}
})

# Model 6: Best NN (30-20-3) with L1, 1000 epochs, 80-20 split
set_all_seeds(42)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)

model6 = Net_30_20_3()
config6 = {'epochs': 1000, 'lr': 0.001, 'l1_lambda': 0.002}
acc6 = train_model(model6, train_loader, test_loader, config6, "Best NN (30-20-3) L1")
results['models'].append({
    'name': 'Best NN (30-20-3) L1',
    'architecture': '100->30->20->3',
    'test_accuracy': acc6,
    'parameters': count_parameters(model6),
    'config': config6
})

# Save results
output_file = RESULTS_DIR / "baseline_models_panel_a.json"
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*60)
print("SUMMARY OF ALL MODELS")
print("="*60)
for i, model in enumerate(results['models'], 1):
    print(f"{i}. {model['name']:30s} | Acc: {model['test_accuracy']*100:6.2f}% | Params: {model['parameters']:6d}")

print(f"\n✓ Results saved to: {output_file}")
