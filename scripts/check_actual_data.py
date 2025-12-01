#!/usr/bin/env python3
"""Check actual data statistics from AI_2qubits_training_data.txt"""

# Read data
data = open('AI_2qubits_training_data.txt').readlines()

# Separate by device
d1 = [l.split()[0] for l in data if l.split()[1]=='1']
d2 = [l.split()[0] for l in data if l.split()[1]=='2']
d3 = [l.split()[0] for l in data if l.split()[1]=='3']

# Calculate Markov P(1->1)
def markov_p11(bits):
    total = 0
    ones_after_one = 0
    for i in range(len(bits)-1):
        if bits[i] == '1':
            total += 1
            if bits[i+1] == '1':
                ones_after_one += 1
    return ones_after_one/total if total > 0 else 0

# Calculate entropy
import math
def entropy(bits):
    p1 = bits.count('1') / len(bits)
    p0 = 1 - p1
    if p1 == 0 or p1 == 1:
        return 0
    return -(p1 * math.log2(p1) + p0 * math.log2(p0))

# Combine all bits for each device
b1 = ''.join(d1)
b2 = ''.join(d2)
b3 = ''.join(d3)

print("=" * 70)
print("ACTUAL DATA STATISTICS FROM AI_2qubits_training_data.txt")
print("=" * 70)
print()

for i, (bits, label) in enumerate([(b1, 'Device 1'), (b2, 'Device 2'), (b3, 'Device 3')], 1):
    ones = bits.count('1')
    total = len(bits)
    freq = ones / total * 100
    p11 = markov_p11(bits)
    ent = entropy(bits)
    
    print(f"{label}:")
    print(f"  '1' frequency: {freq:.2f}%")
    print(f"  P(1->1):       {p11:.4f} ({p11*100:.2f}%)")
    print(f"  Shannon entropy: {ent:.3f} bits")
    print(f"  Total bits:    {total:,}")
    print()

print("=" * 70)
print("COMPARISON WITH PRESENTATION VALUES")
print("=" * 70)
print()

presentation_values = [
    ("Device 1", 54.8, 0.573, 0.986),
    ("Device 2", 56.5, 0.592, 0.979),
    ("Device 3", 59.2, 0.508, 0.992)  # CURRENT (WRONG) VALUES IN SLIDES
]

actual_values = [
    (b1.count('1')/len(b1)*100, markov_p11(b1), entropy(b1)),
    (b2.count('1')/len(b2)*100, markov_p11(b2), entropy(b2)),
    (b3.count('1')/len(b3)*100, markov_p11(b3), entropy(b3))
]

print(f"{'Device':<10} {'Metric':<20} {'Presentation':<15} {'Actual':<15} {'Difference':<15}")
print("-" * 75)

for i, (dev, pres_freq, pres_p11, pres_ent) in enumerate(presentation_values):
    act_freq, act_p11, act_ent = actual_values[i]
    
    print(f"{dev:<10} {'1 frequency %':<20} {pres_freq:<15.2f} {act_freq:<15.2f} {abs(pres_freq-act_freq):<15.2f}")
    print(f"{'':<10} {'P(1->1)':<20} {pres_p11:<15.4f} {act_p11:<15.4f} {abs(pres_p11-act_p11):<15.4f}")
    print(f"{'':<10} {'Entropy (bits)':<20} {pres_ent:<15.3f} {act_ent:<15.3f} {abs(pres_ent-act_ent):<15.3f}")
    print()

print("=" * 70)
print("ISSUES FOUND:")
print("=" * 70)
print()
print("1. Device 3 '1' frequency: Presentation shows 59.2%, actual is 49.2%")
print("   → ERROR: 10.0 percentage points difference!")
print("   → Device 3 is MOST BALANCED (closest to 50%), not HIGH BIAS")
print()
print("2. All P(1→1) values slightly off due to rounding")
print("   → Should use actual values: 0.5731, 0.5915, 0.5078")
print()
print("3. Device 3 should be relabeled:")
print("   → FROM: 'Device 3 (High Bias): 59.2%'")
print("   → TO:   'Device 3 (Low Bias): 49.2%'")
print()
