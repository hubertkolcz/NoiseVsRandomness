# Noise versus Randomness
**YQuantum 2024 -- DoraHacks Challenge**

## 📄 Publications & Resources
- **Poster**: [Verification of qRNG Using qGAN and Classification Models](Verification_of_qRNG_Using_qGAN_and_Classification_Models.pdf)
- **Paper**: [ML-Driven Quantum Hacking of CHSH-Based QKD Protocols](ML_Driven_Quantum_Hacking_of_CHSH_Based_QKD_Protocols.pdf)
- **Presentation**: [Presentation](presentations/presentation.pdf)

## 🎯 Project Overview

This project investigates whether quantum random number generators (qRNGs) from different quantum devices produce statistically distinguishable outputs, despite passing standard randomness tests. We explore the "fingerprint" hypothesis: can machine learning models identify which quantum device generated a specific random number sequence?

### Dataset
Training data in `AI_2qubits_training_data.txt` contains 6,000 samples (2,000 per device) from three quantum sources:
- **Rigetti Aspen-M-3**: 79-qubit superconducting QPU (CHSH score 0.8036, gate fidelity 93.6%)
- **IonQ Aria-1**: 25-qubit trapped ion QPU (CHSH score 0.8362, gate fidelity 99.4%)
- **IBM Qiskit Simulator**: Realistic noise injection model

Each device generates 100-bit random number strings using Bell state entanglement, following the protocol in `quantum-randomness-generator/` (based on https://github.com/dorahacksglobal/quantum-randomness-generator).

### Key Findings
- **Device Classification**: Achieved 59.42% accuracy distinguishing between 3 devices (vs 33.3% random baseline)
- **Device Distinguishability Tournament**: Devices 1 vs 3 show highest distinguishability (KL divergence: 0.205), while Devices 1 vs 2 are most similar (KL: 0.049)
- **qGAN Validation**: Quantum Generative Adversarial Networks successfully model device-specific distributions
- **Cross-Scale Validation**: N=3 device fingerprints successfully generalize to N=30 synthetic device scenarios

## 📁 Project Structure

### Core Directories
- **`notebooks/`**: Jupyter notebooks for analysis and experimentation
  - `qGAN_final.ipynb`: Final qGAN implementation for device distribution learning
  - `ML_solution.ipynb`: Machine learning classification approaches
  - `Q_Random_No.ipynb`: Quantum random number generation analysis
  - `accuracy.ipynb`: Model accuracy evaluation
  
- **`scripts/`**: Python scripts for analysis and validation
  - `device_distinguishability_tournament.py`: KL divergence-based device comparison
  - `comprehensive_verification_benchmark.py`: Full validation suite
  - `evaluate_all_models.py`: Multi-model performance comparison
  - `bridge_N3_N30_validation.py`: Cross-scale validation (3 to 30 devices)
  - `validate_on_real_quantum_rngs.py`: Real quantum device validation
  
- **`classification_models/`**: Trained discriminator models
  - `Discriminator-1.pth`, `Discriminator-2.pth`, `Discriminator-3.pth`
  
- **`data/`**: Datasets and real quantum RNG samples
  - `machine1_GenericBackendV2.npy`: IBM simulator data
  - `machine2_Fake27QPulseV1.npy`: Pulse-level simulation data
  - `N30_synthetic_data.npz`: Synthetic 30-device dataset
  - `real_quantum_rngs/`: Real quantum device outputs
  
- **`results/`**: Experimental results in JSON format
  - `device_distinguishability_tournament_final.json`
  - `bridging_validation_N3_N30.json`
  - `comprehensive_verification_report.json`
  - Model evaluation and optimization results
  
- **`docs/`**: Detailed documentation and analysis reports
  - `DEVICE_DISTINGUISHABILITY_FINAL.md`: Tournament methodology and results
  - `VALIDATION_COMPARISON_ANALYSIS.md`: Cross-validation analysis
  - Verification and audit reports
  
- **`presentations/`**: Presentation materials and speaking notes
  - `presentation_20slides.html`: Main presentation
  - Speaking notes and explanatory documents

## 🚀 Getting Started

### Prerequisites
```bash
# Core dependencies
pip install qiskit qiskit-algorithms qiskit-machine-learning
pip install torch pennylane numpy scipy
pip install matplotlib seaborn pandas
```

### Quick Start

1. **Explore qGAN Implementation**
   ```bash
   jupyter notebook notebooks/qGAN_final.ipynb
   ```

2. **Run Device Distinguishability Tournament**
   ```bash
   python scripts/device_distinguishability_tournament.py
   ```

3. **Evaluate Classification Models**
   ```bash
   python scripts/evaluate_all_models.py
   ```

4. **Comprehensive Verification**
   ```bash
   python scripts/comprehensive_verification_benchmark.py
   ```

## 🔬 Key Components

### 1. Quantum GAN (qGAN)
Implemented in `notebooks/qGAN_final.ipynb`, uses quantum circuits to learn device-specific distributions. Achieves KL divergence ≈ 17.0 between Device 2 and Device 3, demonstrating significant distributional differences.

### 2. Device Distinguishability Tournament
Script: `scripts/device_distinguishability_tournament.py`

Compares device pairs using multiple feature representations:
- **Bit-position frequencies** (64-dim)
- **2-bit joint patterns** (4096-dim)
- **Autocorrelation structure** (4096-dim)

Results ranked by composite KL divergence score.

### 3. Classification Models
Three discriminator models trained to classify device outputs:
- Neural network architectures optimized for 100-bit input
- Trained on 6,000 samples with cross-validation
- Best performance: 59.42% accuracy on 3-class problem

### 4. Cross-Scale Validation
Validates that fingerprinting approach scales from N=3 devices to N=30 synthetic devices, demonstrating generalizability of the method.

## 📊 Results Summary

### Classification Performance
- **3-Device Classification**: 59.42% accuracy (33.3% baseline)
- **Best Device Pair**: Devices 1 vs 3 (most distinguishable)
- **Challenging Pair**: Devices 1 vs 2 (similar noise profiles)

### KL Divergence Rankings
| Rank | Device Pair | Composite Score | Interpretation |
|------|-------------|-----------------|----------------|
| 1 | Device 1 vs 3 | 0.2052 | Most distinguishable |
| 2 | Device 2 vs 3 | 0.2018 | Highly distinguishable |
| 3 | Device 1 vs 2 | 0.0495 | Difficult to distinguish |

## 🤝 Contributing

This project was developed for YQuantum 2024 DoraHacks Challenge. Contributions, issues, and feature requests are welcome.

## 📝 License

See `LICENSE` file for details.

## 🔗 References

- Original qRNG implementation: https://github.com/dorahacksglobal/quantum-randomness-generator
- Detailed documentation available in `docs/` directory
- Results and validation reports in `results/` directory
