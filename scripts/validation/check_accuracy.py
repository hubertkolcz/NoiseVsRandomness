import json
import numpy as np

# Load optimized model results
with open('results/optimized_model_results.json') as f:
    data = json.load(f)

print("="*70)
print("ACCURACY ANALYSIS FROM optimized_model_results.json")
print("="*70)

# Overall statistics
stats = data['statistics']['test_accuracy']
print(f"\nAcross all 4 runs:")
print(f"  Mean accuracy: {stats['mean']*100:.2f}%")
print(f"  Best accuracy: {stats['max']*100:.2f}%")
print(f"  Worst accuracy: {stats['min']*100:.2f}%")
print(f"  Std deviation: {stats['std']*100:.2f}%")

# Best run details
best = data['best_run']
print(f"\n" + "="*70)
print(f"BEST RUN (Seed {best['seed']}):")
print("="*70)
print(f"Test Accuracy: {best['test_accuracy']*100:.2f}%")
print(f"\nConfusion Matrix:")
cm = np.array(best['confusion_matrix'])
print(cm)

print(f"\nPer-class metrics:")
for device, metrics in best['per_class_metrics'].items():
    print(f"\n{device}:")
    print(f"  Precision: {metrics['precision']*100:.1f}%")
    print(f"  Recall: {metrics['recall']*100:.1f}%")
    print(f"  F1-Score: {metrics['f1-score']*100:.1f}%")

# Check article claim
print("\n" + "="*70)
print("COMPARISON WITH ARTICLE CLAIM")
print("="*70)
article_claim = data['metadata']['article_claim']
print(f"Article claims: {article_claim}%")
print(f"Best run achieved: {best['test_accuracy']*100:.2f}%")
print(f"Difference: {abs(article_claim - best['test_accuracy']*100):.2f}% {'✅ CLOSE' if abs(article_claim - best['test_accuracy']*100) < 2 else '⚠️ DIFFERENT'}")
