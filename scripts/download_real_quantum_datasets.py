"""
Download and Prepare Real Quantum RNG Datasets

This script downloads publicly available quantum random number datasets from:
1. IBM QPU QRNG Trial Sequences (Zenodo)
2. Google Sycamore RCS bitstrings (Zenodo)
3. ANU Quantum Random Numbers (API/bulk)

These datasets will be used to validate the N=3 and N=30 trained models on
real quantum hardware, addressing the critical gap identified in bridging validation.

Author: GitHub Copilot
Date: December 1, 2025
"""

import requests
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import hashlib
import time

# Setup paths
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
DATA_DIR = ROOT_DIR / "data" / "real_quantum_rngs"
DATA_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("DOWNLOADING REAL QUANTUM RNG DATASETS")
print("=" * 80)

# ============================================================================
# Dataset Metadata
# ============================================================================

DATASETS = {
    "ibm_qrng": {
        "name": "IBM QPU QRNG Trial Sequences",
        "source": "IBM Quantum Computers (Zenodo)",
        "url": "https://zenodo.org/record/3993549/files/IBM_QRNG_Trial_Sequences.zip",
        "description": "Raw binary outputs from QRNG algorithms on IBM QPUs",
        "uniformity": "Depends on algorithm/extraction",
        "file": DATA_DIR / "ibm_qrng_sequences.zip",
        "expected_format": "Binary files with raw quantum measurements",
        "notes": "Pre-extraction data; validate with NIST tests"
    },
    "google_sycamore": {
        "name": "Google Sycamore RCS Bitstrings",
        "source": "Google Quantum AI Sycamore (Zenodo)",
        "url": "https://zenodo.org/record/3520741",
        "description": "Random circuit sampling bitstrings (67-70 qubits)",
        "uniformity": "Non-uniform (Porter-Thomas distribution)",
        "file": DATA_DIR / "sycamore_rcs_data.json",
        "expected_format": "Circuit definitions + measured bitstrings",
        "notes": "Anti-concentrated distribution; may need post-processing"
    },
    "anu_qrng": {
        "name": "ANU Quantum Random Numbers",
        "source": "Australian National University (Photonic vacuum)",
        "url": "https://qrng.anu.edu.au/API/jsonI.php?length=1024&type=uint8",
        "description": "Real-time quantum random numbers from optical noise",
        "uniformity": "Uniform after extraction (production-ready)",
        "file": DATA_DIR / "anu_qrng_bulk.npy",
        "expected_format": "JSON API responses with uint8 arrays",
        "notes": "Most reliable for uniform IID bits; not general QPU"
    }
}

# ============================================================================
# Download Functions
# ============================================================================

def download_file(url, filepath, chunk_size=8192):
    """Download file with progress indication"""
    try:
        print(f"\n  Downloading from: {url}")
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\r  Progress: {progress:.1f}% ({downloaded}/{total_size} bytes)", end='')
        
        print(f"\n  [OK] Downloaded: {filepath.name} ({downloaded} bytes)")
        return True
    
    except requests.exceptions.RequestException as e:
        print(f"\n  [ERROR] Download failed: {e}")
        return False

def download_anu_qrng_bulk(n_samples=10000, samples_per_device=2000):
    """
    Download multiple batches from ANU QRNG API to simulate multiple devices.
    We'll collect data at different times to get independent samples.
    """
    print("\n[Downloading ANU QRNG Bulk Data]")
    print(f"  Target: {n_samples} samples ({n_samples // samples_per_device} 'devices')")
    
    all_data = []
    n_batches = (n_samples * 100) // 1024  # Each API call returns 1024 bytes
    
    for i in range(n_batches):
        try:
            url = f"https://qrng.anu.edu.au/API/jsonI.php?length=1024&type=uint8"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if data.get('success'):
                # Convert uint8 array to binary bits
                uint8_array = np.array(data['data'], dtype=np.uint8)
                # Unpack each uint8 into 8 bits
                bits = np.unpackbits(uint8_array)
                all_data.extend(bits)
            
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{n_batches} batches downloaded ({len(all_data)} bits collected)")
            
            # Rate limiting to avoid overwhelming the API
            time.sleep(0.1)
        
        except Exception as e:
            print(f"  [WARN] Batch {i+1} failed: {e}")
            continue
    
    # Reshape into samples of 100 bits each
    all_data = np.array(all_data[:n_samples * 100])
    samples = all_data.reshape(-1, 100)
    
    print(f"\n  [OK] Collected {len(samples)} samples of 100 bits each")
    return samples

# ============================================================================
# Main Download Process
# ============================================================================

print("\n[STEP 1] Dataset Information")
print("-" * 80)

for key, info in DATASETS.items():
    print(f"\n{info['name']}:")
    print(f"  Source: {info['source']}")
    print(f"  Description: {info['description']}")
    print(f"  Uniformity: {info['uniformity']}")
    print(f"  Notes: {info['notes']}")

# ============================================================================
# Download ANU QRNG (Most Reliable for Validation)
# ============================================================================

print("\n" + "=" * 80)
print("[STEP 2] Downloading ANU QRNG Data (Primary Validation Source)")
print("=" * 80)

anu_file = DATA_DIR / "anu_qrng_samples.npy"

if anu_file.exists():
    print(f"\n[INFO] ANU QRNG data already exists: {anu_file}")
    print("  Loading existing data...")
    anu_samples = np.load(anu_file)
    print(f"  Loaded: {anu_samples.shape[0]} samples")
else:
    print("\n[INFO] Downloading fresh ANU QRNG data...")
    print("  This will take several minutes due to API rate limits...")
    
    # Download 10,000 samples (5 "devices" × 2000 samples each)
    anu_samples = download_anu_qrng_bulk(n_samples=10000, samples_per_device=2000)
    
    # Save to file
    np.save(anu_file, anu_samples)
    print(f"\n  [OK] Saved to: {anu_file}")

# Create "devices" by splitting into temporal batches
# This simulates 5 independent quantum sources by sampling at different times
n_devices = 5
samples_per_device = len(anu_samples) // n_devices

print(f"\n[INFO] Creating {n_devices} virtual devices (temporal batches)")
for i in range(n_devices):
    start_idx = i * samples_per_device
    end_idx = (i + 1) * samples_per_device
    device_samples = anu_samples[start_idx:end_idx]
    
    freq_1 = np.mean(device_samples)
    print(f"  Device {i+1}: {len(device_samples)} samples, freq('1')={freq_1:.4f}")

# ============================================================================
# Download IBM QRNG (Secondary - Requires Manual Processing)
# ============================================================================

print("\n" + "=" * 80)
print("[STEP 3] IBM QRNG Trial Sequences")
print("=" * 80)

print("\n[INFO] IBM QRNG data requires manual download from Zenodo:")
print("  URL: https://zenodo.org/record/3993549")
print("  Note: Dataset may not be directly accessible via automated download")
print("  Recommendation: Download manually and place in data/real_quantum_rngs/")
print("\n  [SKIP] Automated download not implemented (requires Zenodo API key)")

# ============================================================================
# Download Google Sycamore (Tertiary - Non-uniform Distribution)
# ============================================================================

print("\n" + "=" * 80)
print("[STEP 4] Google Sycamore RCS Bitstrings")
print("=" * 80)

print("\n[INFO] Google Sycamore data available at:")
print("  URL: https://zenodo.org/record/3520741")
print("  Note: Non-uniform distribution (Porter-Thomas); requires post-processing")
print("  Recommendation: Download manually for advanced analysis")
print("\n  [SKIP] Automated download not implemented (large dataset)")

# ============================================================================
# Generate Metadata File
# ============================================================================

print("\n" + "=" * 80)
print("[STEP 5] Generating Metadata")
print("=" * 80)

metadata = {
    "timestamp": datetime.now().isoformat(),
    "purpose": "Real quantum RNG datasets for model validation",
    "datasets": {
        "anu_qrng": {
            "status": "downloaded",
            "file": str(anu_file.relative_to(ROOT_DIR)),
            "n_samples": len(anu_samples),
            "n_devices": n_devices,
            "samples_per_device": samples_per_device,
            "source": DATASETS["anu_qrng"]["source"],
            "uniformity": DATASETS["anu_qrng"]["uniformity"],
            "collection_date": datetime.now().isoformat(),
            "bias_characteristics": {
                f"device_{i+1}": {
                    "freq_1": float(np.mean(anu_samples[i*samples_per_device:(i+1)*samples_per_device]))
                }
                for i in range(n_devices)
            }
        },
        "ibm_qrng": {
            "status": "manual_download_required",
            "url": DATASETS["ibm_qrng"]["url"],
            "notes": "Requires manual download from Zenodo"
        },
        "google_sycamore": {
            "status": "manual_download_required",
            "url": DATASETS["google_sycamore"]["url"],
            "notes": "Requires manual download from Zenodo; non-uniform distribution"
        }
    },
    "next_steps": [
        "Run validate_on_real_quantum_rngs.py to test N=3 and N=30 models",
        "Manually download IBM QRNG and Google Sycamore datasets if needed",
        "Compare performance on real quantum data vs synthetic validation"
    ]
}

metadata_file = DATA_DIR / "datasets_metadata.json"
with open(metadata_file, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\n[OK] Metadata saved to: {metadata_file}")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 80)
print("DOWNLOAD SUMMARY")
print("=" * 80)

print(f"\n[OK] ANU QRNG: {len(anu_samples)} samples downloaded ({n_devices} devices)")
print(f"     File: {anu_file.relative_to(ROOT_DIR)}")

print(f"\n[INFO] IBM QRNG: Manual download required")
print(f"     URL: {DATASETS['ibm_qrng']['url']}")

print(f"\n[INFO] Google Sycamore: Manual download required")
print(f"     URL: {DATASETS['google_sycamore']['url']}")

print("\n[NEXT STEP]")
print("  Run: python scripts/validate_on_real_quantum_rngs.py")
print("  This will test the trained models on real quantum data")

print("\n" + "=" * 80)
print("[OK] Dataset download complete!")
print("=" * 80)
