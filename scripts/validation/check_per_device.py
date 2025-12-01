import numpy as np

# Actual N=3 confusion matrix from optimized_model_results.json (best run, seed 89)
cm = np.array([
    [157, 201, 65],   # Device 1: 423 samples
    [97, 268, 26],    # Device 2: 391 samples  
    [73, 25, 288]     # Device 3: 386 samples
])

print("="*60)
print("ACTUAL N=3 CONFUSION MATRIX (from optimized_model_results.json)")
print("="*60)
print(cm)
print(f"\nTotal samples: {cm.sum()}")

print("\n" + "="*60)
print("PER-DEVICE ACCURACY (N=3)")
print("="*60)
for i in range(3):
    accuracy = cm[i,i] / cm[i].sum() * 100
    print(f"Device {i+1}: {cm[i,i]}/{cm[i].sum()} = {accuracy:.1f}%")

print("\n" + "="*60)
print("FIGURE 8 HARDCODED VALUES (in generate_nn_comparison_figures.py)")
print("="*60)
# From the script
fig8_cm = np.array([
    [200, 50, 0],     # Device 1: 66.7% accuracy
    [40, 195, 15],    # Device 2: 65.0% accuracy
    [10, 5, 210]      # Device 3: 70.0% accuracy
])
print(fig8_cm)
print("\nHardcoded accuracies: [66.7%, 65.0%, 70.0%]")

print("\n" + "="*60)
print("COMPARISON")
print("="*60)
print("Device 1: ACTUAL=37.1%  vs  FIGURE=66.7%  ❌ WRONG")
print("Device 2: ACTUAL=68.5%  vs  FIGURE=65.0%  ⚠️ Close but not exact")
print("Device 3: ACTUAL=74.6%  vs  FIGURE=70.0%  ⚠️ Close but not exact")
print("\n✅ CONCLUSION: Figure 8 uses HARDCODED/ESTIMATED values, NOT actual N=3 data")
