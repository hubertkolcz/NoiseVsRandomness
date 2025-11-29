"""
Comprehensive Neural Network Model Evaluation Script
Evaluates all NN architectures from the repository and compares their performance
against the DoraHacks challenge goal and article claims
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import json
from datetime import datetime

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
print("\n" + "="*80)
print("LOADING DATA")
print("="*80)
data = np.loadtxt("AI_2qubits_training_data.txt", dtype=str)
print(f"Total samples: {len(data)}")

def binary_to_bits(binary_list):
    return [[[int(i) for i in str(bit)] for bit in binary_string][0] for binary_string in binary_list]

X = torch.tensor(np.array(binary_to_bits(data[:,:-1])))
Y = torch.tensor(data[:,-1].astype(int))-1

print(f"Features shape: {X.shape}")
print(f"Labels shape: {Y.shape}")
print(f"Device distribution: {torch.bincount(Y)}")

# Store all results
results = {
    "metadata": {
        "timestamp": datetime.now().isoformat(),
        "total_samples": len(data),
        "device_counts": torch.bincount(Y).tolist(),
        "baseline_random_accuracy": 33.33,
        "dorahacks_challenge_goal": 54.0,
        "article_best_reported": 58.67
    },
    "models": []
}

# Function to calculate accuracy
def get_accuracy(loader, model):
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return correct / total, all_preds, all_labels

def evaluate_model(net, train_loader, test_loader, model_name, config):
    """Evaluate a model and return comprehensive metrics"""
    print(f"\n{'='*80}")
    print(f"EVALUATING: {model_name}")
    print(f"{'='*80}")
    print(f"Architecture: {config['architecture']}")
    print(f"Hyperparameters: {config['hyperparameters']}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer_config = config['hyperparameters']['optimizer']
    
    if optimizer_config['type'] == 'Adam':
        if 'weight_decay' in optimizer_config:
            optimizer = optim.Adam(net.parameters(), 
                                 lr=optimizer_config['lr'],
                                 weight_decay=optimizer_config['weight_decay'])
        else:
            optimizer = optim.Adam(net.parameters(), lr=optimizer_config['lr'])
    
    epochs = config['hyperparameters']['epochs']
    l1_lambda = config['hyperparameters'].get('l1_lambda', 0.0)
    
    # Training
    best_train_acc = 0
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            
            # Add L1 regularization if specified
            if l1_lambda > 0:
                l1_norm = sum(p.abs().sum() for p in net.parameters())
                loss += l1_lambda * l1_norm
            
            loss.backward()
            optimizer.step()
        
        if epoch % 10 == 0 or epoch == epochs - 1:
            train_acc, _, _ = get_accuracy(train_loader, net)
            best_train_acc = max(best_train_acc, train_acc)
            if epoch % 50 == 0:
                print(f"  Epoch {epoch:4d}/{epochs} | Loss: {loss.item():.4f} | Train Acc: {train_acc:.4f}")
    
    # Final evaluation
    net.eval()
    train_acc, train_preds, train_labels = get_accuracy(train_loader, net)
    test_acc, test_preds, test_labels = get_accuracy(test_loader, net)
    
    # Calculate confusion matrix and per-class metrics
    cm = confusion_matrix(test_labels, test_preds)
    report = classification_report(test_labels, test_preds, output_dict=True, zero_division=0)
    
    print(f"\n  RESULTS:")
    print(f"  Train Accuracy: {train_acc*100:.2f}%")
    print(f"  Test Accuracy:  {test_acc*100:.2f}%")
    print(f"  Best Train Acc: {best_train_acc*100:.2f}%")
    print(f"\n  Per-Class Metrics:")
    for class_id in range(3):
        if str(class_id) in report:
            print(f"    Device {class_id+1}: Precision={report[str(class_id)]['precision']:.4f}, "
                  f"Recall={report[str(class_id)]['recall']:.4f}, "
                  f"F1={report[str(class_id)]['f1-score']:.4f}")
    
    print(f"\n  Confusion Matrix:")
    print(f"  {cm}")
    
    # Calculate improvement over baselines
    improvement_over_random = ((test_acc - 0.3333) / 0.3333) * 100
    improvement_over_dorahacks = ((test_acc - 0.54) / 0.54) * 100
    
    print(f"\n  Improvements:")
    print(f"    vs Random (33.33%): +{improvement_over_random:.1f}%")
    print(f"    vs DoraHacks (54%): +{improvement_over_dorahacks:.1f}%")
    
    return {
        "name": model_name,
        "config": config,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "best_train_accuracy": best_train_acc,
        "confusion_matrix": cm.tolist(),
        "per_class_metrics": {
            "device_1": report.get('0', {}),
            "device_2": report.get('1', {}),
            "device_3": report.get('2', {})
        },
        "improvement_over_random_pct": improvement_over_random,
        "improvement_over_dorahacks_pct": improvement_over_dorahacks,
        "exceeds_article_claim": test_acc >= 0.5867
    }

# ============================================================================
# MODEL 1: Baseline - Simple Architecture (20-20-3)
# ============================================================================
class Net_Baseline(nn.Module):
    def __init__(self):
        super(Net_Baseline, self).__init__()
        self.fc1 = nn.Linear(100, 20)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(20, 20)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(20, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# ============================================================================
# MODEL 2: Article Model - Optimized Architecture (30-20-3)
# ============================================================================
class Net_Article(nn.Module):
    def __init__(self):
        super(Net_Article, self).__init__()
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
        x = self.fc3(x)
        return x

# ============================================================================
# MODEL 3: Compressed Model (20-10-3)
# ============================================================================
class Net_Compressed(nn.Module):
    def __init__(self):
        super(Net_Compressed, self).__init__()
        self.fc1 = nn.Linear(100, 20)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(20, 10)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(10, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# ============================================================================
# TEST ALL MODELS WITH DIFFERENT CONFIGURATIONS
# ============================================================================

dataset = TensorDataset(X.float(), Y.long())

# Configuration 1: 80-20 split, batch=8 (Article configuration)
print("\n" + "="*80)
print("CONFIGURATION 1: 80-20 Split, Batch=8, 1000 epochs (Article Setup)")
print("="*80)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
torch.manual_seed(89)  # Article seed
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Model 1.1: Article Model (30-20-3) with L1
config = {
    "architecture": "100->30->20->3",
    "hyperparameters": {
        "epochs": 1000,
        "batch_size": 8,
        "optimizer": {"type": "Adam", "lr": 0.001},
        "l1_lambda": 0.002,
        "dropout": 0.2,
        "split": "80-20",
        "seed": 89
    }
}
net = Net_Article().to(device)
result = evaluate_model(net, train_loader, test_loader, 
                       "Article Model (30-20-3, L1=0.002)", config)
results["models"].append(result)

# Model 1.2: Article Model without L1
config["hyperparameters"]["l1_lambda"] = 0.0
net = Net_Article().to(device)
result = evaluate_model(net, train_loader, test_loader,
                       "Article Model (30-20-3, no L1)", config)
results["models"].append(result)

# Model 1.3: Baseline Model (20-20-3) with L2
config = {
    "architecture": "100->20->20->3",
    "hyperparameters": {
        "epochs": 1000,
        "batch_size": 8,
        "optimizer": {"type": "Adam", "lr": 0.001, "weight_decay": 0.0005},
        "l1_lambda": 0.0,
        "dropout": 0.2,
        "split": "80-20",
        "seed": 42
    }
}
torch.manual_seed(42)
net = Net_Baseline().to(device)
result = evaluate_model(net, train_loader, test_loader,
                       "Baseline Model (20-20-3, L2=0.0005)", config)
results["models"].append(result)

# Configuration 2: 80-20 split, batch=4
print("\n" + "="*80)
print("CONFIGURATION 2: 80-20 Split, Batch=4")
print("="*80)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

config = {
    "architecture": "100->30->20->3",
    "hyperparameters": {
        "epochs": 500,
        "batch_size": 4,
        "optimizer": {"type": "Adam", "lr": 0.001},
        "l1_lambda": 0.002,
        "dropout": 0.2,
        "split": "80-20",
        "seed": 0
    }
}
torch.manual_seed(0)
net = Net_Article().to(device)
result = evaluate_model(net, train_loader, test_loader,
                       "Article Model (30-20-3, batch=4)", config)
results["models"].append(result)

# Configuration 3: 70-30 split (Logistic Regression comparison)
print("\n" + "="*80)
print("CONFIGURATION 3: 70-30 Split, Batch=8 (LR Comparison)")
print("="*80)

train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size
torch.manual_seed(42)
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

config = {
    "architecture": "100->30->20->3",
    "hyperparameters": {
        "epochs": 500,
        "batch_size": 8,
        "optimizer": {"type": "Adam", "lr": 0.001},
        "l1_lambda": 0.002,
        "dropout": 0.2,
        "split": "70-30",
        "seed": 42
    }
}
net = Net_Article().to(device)
result = evaluate_model(net, train_loader, test_loader,
                       "Article Model (30-20-3, 70-30 split)", config)
results["models"].append(result)

# Configuration 4: Small training set (30-70 split)
print("\n" + "="*80)
print("CONFIGURATION 4: 30-70 Split (Limited Training Data)")
print("="*80)

train_size = int(0.3 * len(dataset))
test_size = len(dataset) - train_size
torch.manual_seed(42)
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

config = {
    "architecture": "100->20->10->3",
    "hyperparameters": {
        "epochs": 300,
        "batch_size": 8,
        "optimizer": {"type": "Adam", "lr": 0.001},
        "l1_lambda": 0.001,
        "dropout": 0.2,
        "split": "30-70",
        "seed": 42
    }
}
net = Net_Compressed().to(device)
result = evaluate_model(net, train_loader, test_loader,
                       "Compressed Model (20-10-3, limited data)", config)
results["models"].append(result)

# ============================================================================
# LOGISTIC REGRESSION (from accuracy.ipynb)
# ============================================================================
print("\n" + "="*80)
print("BASELINE: Logistic Regression (70-30 split)")
print("="*80)

X_np = X.numpy()
Y_np = Y.numpy()
X_train, X_test, y_train, y_test = train_test_split(X_np, Y_np, test_size=0.3, random_state=42)

classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
lr_accuracy = accuracy_score(y_test, y_pred)
lr_cm = confusion_matrix(y_test, y_pred)
lr_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

print(f"  Test Accuracy: {lr_accuracy*100:.2f}%")
print(f"  Confusion Matrix:\n{lr_cm}")

results["models"].append({
    "name": "Logistic Regression",
    "config": {
        "architecture": "Linear (100->3)",
        "hyperparameters": {
            "split": "70-30",
            "max_iter": 1000,
            "seed": 42
        }
    },
    "train_accuracy": None,
    "test_accuracy": lr_accuracy,
    "confusion_matrix": lr_cm.tolist(),
    "per_class_metrics": {
        "device_1": lr_report.get('0', {}),
        "device_2": lr_report.get('1', {}),
        "device_3": lr_report.get('2', {})
    }
})

# ============================================================================
# SUMMARY AND COMPARISON
# ============================================================================
print("\n" + "="*80)
print("COMPREHENSIVE RESULTS SUMMARY")
print("="*80)

print(f"\nBaseline Random Accuracy: 33.33%")
print(f"DoraHacks Challenge Goal: 54.00%")
print(f"Article Best Reported: 58.67%")
print(f"\n{'Model':<50} {'Test Acc':<12} {'Status'}")
print("-"*80)

for model in results["models"]:
    status = "✓ EXCEEDS ARTICLE" if model.get("exceeds_article_claim", False) else "○"
    print(f"{model['name']:<50} {model['test_accuracy']*100:>10.2f}%  {status}")

# Find best model
best_model = max(results["models"], key=lambda x: x["test_accuracy"])
print(f"\n{'='*80}")
print(f"BEST MODEL: {best_model['name']}")
print(f"Test Accuracy: {best_model['test_accuracy']*100:.2f}%")
print(f"Architecture: {best_model['config']['architecture']}")
print(f"{'='*80}")

# Save results to JSON
with open('model_evaluation_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to: model_evaluation_results.json")

# Generate comparison table for presentation
print("\n" + "="*80)
print("PRESENTATION TABLE DATA")
print("="*80)
print("\nCopy this data for the presentation:")
print("-"*80)

for model in results["models"]:
    config = model['config']
    arch = config['architecture']
    hyper = config['hyperparameters']
    split = hyper.get('split', 'N/A')
    batch = hyper.get('batch_size', 'N/A')
    l1 = hyper.get('l1_lambda', 0.0)
    test_acc = model['test_accuracy'] * 100
    
    print(f"{model['name']:<45} | {arch:<20} | {split:<8} | batch={batch:<4} | L1={l1:<6} | {test_acc:>6.2f}%")

print("\n" + "="*80)
print("EVALUATION COMPLETE")
print("="*80)
