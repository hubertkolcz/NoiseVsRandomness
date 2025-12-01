import numpy as np

# Load the actual device data
d1 = np.load('data/machine1_GenericBackendV2.npy')
d2 = np.load('data/machine2_Fake27QPulseV1.npy')

def shannon_entropy(binary_data):
    """Calculate Shannon entropy for binary data"""
    p1 = np.mean(binary_data)
    p0 = 1 - p1
    h = 0
    if p0 > 0:
        h += -p0 * np.log2(p0)
    if p1 > 0:
        h += -p1 * np.log2(p1)
    return h

print("="*60)
print("ACTUAL ENTROPY VALUES FROM DATA FILES")
print("="*60)

# Calculate for each device
entropy1 = shannon_entropy(d1)
entropy2 = shannon_entropy(d2)

print(f"\nDevice 1 (machine1_GenericBackendV2.npy):")
print(f"  '1' frequency: {np.mean(d1)*100:.2f}%")
print(f"  Shannon Entropy: {entropy1:.3f} bits")

print(f"\nDevice 2 (machine2_Fake27QPulseV1.npy):")
print(f"  '1' frequency: {np.mean(d2)*100:.2f}%")
print(f"  Shannon Entropy: {entropy2:.3f} bits")

# Note: The repository appears to use 3 devices from DoraHacks data
# Let's check what the generate_presentation_figures.py script reports
print("\n" + "="*60)
print("CHECKING SCRIPT OUTPUT FOR 3-DEVICE ENTROPY")
print("="*60)
print("Note: The DoraHacks dataset has 3 devices")
print("We need to check the comprehensive_verification_report.json")
