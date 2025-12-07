# 20-Minute Presentation Speaking Notes
## Machine Learning-Driven Quantum Hacking of CHSH-Based QKD
**Exploiting Entropy Vulnerabilities in Self-Testing Protocols**

**QUEST-IS 2025 | December 3, 2025**  
**Target: 19-20 minutes for 19 slides**

---

## **SLIDE 1: Title Slide (0:00-0:50) [50s]**
**On screen:**
- Machine Learning-Driven Quantum Hacking of CHSH-Based QKD
- Subtitle: Exploiting Entropy Vulnerabilities in Self-Testing Protocols
- Authors: Hubert Kołcz¹, Tushar Pandey², Yug Shah³
- ¹Warsaw University of Technology | ²Texas A&M University | ³University of Toronto

**What to say:**
- Good morning. I'm Hubert Kołcz from Warsaw University of Technology
- Today we present ML-driven analysis of entropy vulnerabilities in CHSH-based quantum key distribution
- Key finding: ML can fingerprint quantum RNGs at 59% accuracy even when CHSH tests pass
- Validated on N=30 synthetic devices with 20× distinguishability between classes

---

## **SLIDE 2: CHSH Inequality: Foundation for QKD Security (0:50-2:10) [80s]**
**On screen:**
- Equation: S = ⟨AB⟩ + ⟨AB'⟩ + ⟨A'B⟩ - ⟨A'B'⟩
- Classical: S ≤ 2 | Quantum: S > 2 (max 2√2 ≈ 2.828)
- Why CHSH Dominates QKD Industry

**What to say:**
- CHSH inequality is foundation for device-independent QKD
- Combines four correlation measurements between Alice and Bob
- Classical physics constrains S ≤ 2, quantum allows up to 2.828
- **Why industry adopts CHSH:**
  - Experimental robustness: tolerates detector imperfections (unlike Bell's perfect correlation requirement)
  - Self-testing capability: simultaneously verifies quantum state AND detects eavesdropping
  - Device-independent security: if S > 2, eavesdropper cannot have complete key information
  - Industry standard in metro QKD networks and commercial implementations

---

## **SLIDE 3: The Critical Security Gap (2:10-3:30) [80s] - KEY SLIDE**
**On screen:**
- The Paradox: CHSH provides mathematical security, but implementations rely on RNGs susceptible to side-channel attacks
- Four attack types: Phase Remapping, Trojan Horse, Time-Shift, Detector Blinding
- RNG Entropy Analysis box

**What to say:**
- The central paradox: CHSH gives mathematical security, but implementations depend on vulnerable RNGs
- **Traditional attacks target transmission/detection:**
  - Phase remapping: manipulates quantum phase relationships
  - Trojan horse: light signal injection for eavesdropping
  - Time-shift: exploits detection timing windows
  - Detector blinding: forces detectors into classical mode
- **Our work focuses on different vulnerability: RNG entropy source itself**
- **Our contribution:** ML-driven framework analyzing RNG noise through:
  - Entropy monitoring (Shannon, min-entropy, KL divergence)
  - Hardware metrics (gate fidelity → Bell correlation)
- **CRITICAL DISCLAIMER:** We demonstrate statistical fingerprinting on simulator data
- Have NOT demonstrated actual key extraction from real QKD systems
- Gap between detecting patterns and cryptanalysis is significant

---

## **SLIDE 4: Multi-Method ML Benchmarking Framework (3:30-5:00) [90s]**
**On screen:**
- Three independent approaches: qGAN (12-qubit), Logistic Regression (baseline), Neural Network (30-20-3)
- Four-stage operation flowchart
- Platform comparison: N=3 IBMQ simulators, N=30 synthetic validation

**What to say:**
- Tested three independent ML approaches for robustness:
  1. **qGAN**: 12-qubit quantum GAN, KL divergence 0.05-0.20
  2. **Logistic Regression**: linear baseline for comparison
  3. **Neural Network**: 30-20-3 architecture, our primary method
- **Four-stage operation:**
  1. Analyze entropy patterns (Markov transitions, autocorrelation, run-length statistics)
  2. Correlate hardware metrics (gate fidelity → Bell correlation, R²=0.977)
  3. Compare platforms (Rigetti 0.8036, IonQ 0.8362, IBM 1.0 baseline)
  4. Extract device-specific fingerprints
- **Cross-method validation:** Pearson r=0.865 (p<10⁻⁹), Spearman ρ=0.931 (p<10⁻¹⁴)
- All three methods converge on same device classifications

---

## **SLIDE 5: Experimental Methodology & Hardware (5:00-6:30) [90s]**
**On screen:**
- Hardware platforms table: Rigetti Aspen-M-3 (80 qubits), IonQ Aria-1 (25 qubits), IBM Qiskit (ideal)
- Dataset transparency: N=3 from IBMQ simulators, N=30 synthetic
- Multi-method benchmarking approach

**What to say:**
- **Hardware platforms:**
  - Rigetti Aspen-M-3: 80 superconducting qubits, Bell correlation 0.8036, gate fidelity 93.6%
  - IonQ Aria-1: 25 trapped ion qubits, Bell 0.8362, gate fidelity 99.4%
  - IBM Qiskit: ideal simulator baseline, Bell 1.0, fidelity 100%
- **Dataset transparency - this is critical:**
  - N=3 dataset: 6,000 samples from IBMQ noise-injected simulators (DoraHacks YQuantum 2024)
  - NOT actual QPU data - realistic noise models but still simulators
  - N=30 dataset: entirely synthetic devices with controlled bias levels
- **Why synthetic?** Proper statistical power requires 30+ samples, commercial QPU access expensive/limited
- **N=3 device fingerprints:**
  - Device 0: 54.7% '1' frequency, P(1→1)=0.573
  - Device 1: 56.5%, P(1→1)=0.592
  - Device 2: 49.2%, P(1→1)=0.508
- Cross-method validation: r=0.865, ρ=0.931, both p<10⁻⁹

---

## **SLIDE 6: Quantitative Analysis: Bit Frequency Distribution (6:30-7:30) [60s]**
**On screen:**
- Figure showing bit frequency histograms for 3 devices
- Three colored boxes showing Device 1, 2, 3 statistics

**What to say:**
- **The entropy paradox - this is fascinating:**
  - Device 1: 54.8% '1' bias, entropy 0.986 bits (low bias)
  - Device 2: 56.5% bias, entropy 0.979 bits (medium bias)
  - Device 3: 59.2% bias, entropy 0.992 bits (HIGHEST bias, HIGHEST entropy!)
- **Why paradoxical?**
  - Device 3 has strongest frequency bias but highest Shannon entropy
  - Yet easiest to classify at 70% accuracy
  - Shannon entropy measures first-order statistics (individual bits)
  - ML detects second-order patterns (Markov transitions, autocorrelation, run-length)
- **Security implication:** Passing NIST chi-square tests doesn't guarantee ML-robustness
- All three devices pass χ² < 3.841, yet distinguishable at 59-70% accuracy

---

## **SLIDE 7: Markov Chain Transition Matrices (7:30-8:20) [50s]**
**On screen:**
- Figure showing 3 Markov transition matrix heatmaps
- Device-specific transition probabilities

**What to say:**
- **Markov transition probabilities reveal device fingerprints:**
  - Device 1: P(1→1)=0.573 (moderate persistence)
  - Device 2: P(1→1)=0.592 (strongest persistence - bit '1' tends to repeat)
  - Device 3: P(1→1)=0.508 (most symmetric, appears random)
- **Key finding:** Device-specific transition biases create ML fingerprints invisible to NIST tests
- **Memory attack connection:**
  - P(1→1) deviation from 0.5 indicates statistical memory
  - Physical origins: detector afterpulsing, gate errors, thermal drift
  - Enables basis prediction with better-than-random probability
- Standard NIST tests check frequency distributions, not temporal correlations

---

## **SLIDE 8: Machine Learning Performance Metrics (8:20-9:50) [90s]**
**On screen:**
- Four-panel figure: confusion matrix, method comparison, N=3 to N=30 bridging, baseline improvement

**What to say:**
- **N=30 Validation Results - four panels:**
  
**Panel A - Confusion Matrix:**
  - 59% overall accuracy balanced across 3 device classes
  - Diagonal shows correct classifications, off-diagonal shows confusion
  
**Panel B - Method Comparison:**
  - Neural Network: 59% accuracy
  - Logistic Regression: 60% (slightly better, simpler model)
  - Random baseline: 33.3% (3-way classification)
  - 77% improvement over random guessing
  
**Panel C - N=3 to N=30 Bridging (critical validation):**
  - N=3: 58.67% accuracy (IBMQ simulators)
  - N=30: 59.21% accuracy (synthetic)
  - Replicates within 0.54 percentage points
  - p<10⁻⁹ with 28 degrees of freedom
  - Model trained on N=3 generalizes to N=30 without modification
  
**Panel D - Baseline Improvement:**
  - All methods 77-80% above random
  - Statistical significance validated

- **Conclusion:** Method works on synthetic data with proper statistical power
- Need validation on 50+ real production QKD RNGs

---

## **SLIDE 9: Neural Network Architecture Analysis (9:50-10:40) [50s]**
**On screen:**
- Architecture diagram or hyperparameter analysis figure

**What to say:**
- **Optimal configuration found through grid search:**
  - Batch size: 8 (stable gradients without overfitting)
  - Architecture: 30-20-3 (wider first layer captures more features, bottleneck for classification)
  - Regularization: L1 λ=0.002 (sparse feature selection, prevents overfitting)
  - Training: 1000 epochs necessary for convergence
- **Generalization evidence:**
  - Hyperparameters optimized on N=3 dataset
  - Applied to N=30 without any modification
  - Performance replicates (58.67% → 59.21%)
  - Indicates not overfitting to specific N=3 noise profiles
  - Model learned generalizable RNG fingerprints

---

## **SLIDE 10: Per-Device Classification Performance (10:40-11:30) [50s]**
**On screen:**
- Bar chart or table showing individual device accuracy

**What to say:**
- **Individual device results (N=3):**
  - Device 1 (54.8% '1' bias): 66.7% classification accuracy
  - Device 2 (56.5% bias): 65.0% accuracy - most challenging because similar to both extremes
  - Device 3 (59.2% bias): 70.0% accuracy - best performance despite highest entropy!
- **Key insight:** Device 3 paradox again
  - Highest Shannon entropy (0.992 bits)
  - Should be "most random" by traditional metrics
  - Yet easiest to classify because ML exploits temporal patterns
- **Security takeaway:** High entropy is necessary but NOT sufficient
  - Need adversarial robustness testing beyond NIST
  - Second-order statistics matter for ML security

---

## **SLIDE 11: qGAN Distributional Analysis: Device Distinguishability (11:30-12:50) [80s]**
**On screen:**
- KL divergence tournament results
- Within-class vs between-class comparison

**What to say:**
- **qGAN tournament methodology:**
  - 435 pairwise KL divergence comparisons
  - All combinations of 30 devices in N=30 validation

- **Within-class KL (similar devices in same bias class):**
  - Class 0-0 (low bias): mean 0.048, std 0.044 (very similar)
  - Class 1-1 (medium): mean 0.083, std 0.072
  - Class 2-2 (high): mean 0.101, std 0.101 (most variable)

- **Between-class KL (different bias classes):**
  - Classes 0-1: mean 0.670, std 0.369
  - Classes 0-2: mean 3.180, std 1.318 (MOST distinguishable)
  - Classes 1-2: mean 0.961, std 0.705

- **Statistical validation:**
  - Mann-Whitney U test: p=3.26×10⁻⁶⁰ (ridiculously significant)
  - 20× higher distinguishability between classes vs within classes
  - Confirms devices cluster by bias level
  
- **Cross-method consistency:**
  - qGAN KL divergence correlates with NN accuracy: r=0.865, ρ=0.931
  - Independent methods converge on same device structure

---

## **SLIDE 12: Proposed DI-QKD Vulnerability Analysis (12:50-14:20) [90s] - KEY TECHNICAL**
**On screen:**
- Two-phase attack methodology (Phase 1: RNG Profiling, Phase 2: Basis Prediction)
- Validated technical foundation section
- Critical finding box

**What to say:**
- **Phase 1: RNG Profiling (passive monitoring)**
  - Collect RNG output during normal QKD operation
  - ML fingerprinting: classify device at 59% accuracy (80% above random)
  - Bias detection: identify exploitable 59% vs 54% '1' frequency threshold
  - Temporal pattern extraction: Markov transitions P(1→1) = 0.508-0.592

- **Phase 2: Measurement Basis Prediction (active exploitation)**
  - Monitor environmental factors: temperature, gate fidelity drift
  - Track CHSH degradation: deviation from ideal 2.828 to exploitable S<2.2
  - Basis inference: use RNG bias patterns to predict Alice/Bob measurement settings
  - Side-channel extraction: combine entropy deviation + hardware signatures

- **Validated technical foundation:**
  - Multi-modal validation: qGAN-NN correlation r=0.865 (N=30, p<10⁻⁹), between-class KL 20× higher (p<10⁻⁶⁰)
  - Hardware correlation: Gate fidelity → CHSH score → RNG quality (R²=0.977 across Rigetti/IonQ/IBM)
  - Attack detection threshold: CHSH<2.2 + bias>59% signals exploitable conditions

- **DI-QKD vulnerability context:**
  - Security proofs assume stable min-entropy bounds
  - Don't account for temporal correlations beyond what min-entropy captures
  - Our framework detects when real RNGs violate theoretical assumptions

- **CRITICAL GAP:** We fingerprint certified QRNGs but haven't demonstrated key extraction on real QKD
  - Detecting statistical patterns ≠ extracting secret keys
  - Real attack requires real-time basis prediction + key correlation
  - Gap between our work and actual security breach is substantial

---

## **SLIDE 13: Hardware Platform CHSH Correlation Analysis (14:20-15:20) [60s]**
**On screen:**
- Table comparing IBM Qiskit, Rigetti Aspen-M-3, IonQ Aria-1
- Hardware comparison figure

**What to say:**
- **Platform comparison table:**
  - IBM Qiskit (simulator): Bell 1.0, gate fidelity 100%, ideal baseline
  - Rigetti Aspen-M-3: 80 superconducting qubits, Bell 0.8036, fidelity 93.6%
  - IonQ Aria-1: 25 trapped ion qubits, Bell 0.8362, fidelity 99.4% (best hardware)

- **Critical finding: R²=0.977 correlation**
  - Gate fidelity strongly predicts Bell correlation coefficient
  - Lower gate fidelity → lower Bell correlation → noisier quantum operations → more exploitable RNG
  - Easier to measure gate fidelity than full CHSH tests
  - Provides early warning system for RNG vulnerability

- **Aggregate vs microstructure:**
  - CHSH verifies aggregate S values (averaged over 100,000 shots)
  - Doesn't verify per-sample microstructure (temporal correlations within 100-bit sequences)
  - CHSH says "average behavior is quantum"
  - ML says "individual patterns are exploitable"
  - Gap between aggregate certification and sample-level security

---

## **SLIDE 14: Statistical Significance & Correlation Analysis (15:20-16:10) [50s]**
**On screen:**
- Correlation analysis figure with scatter plot and confidence intervals
- Two-column boxes: Correlation Evidence, Statistical Power

**What to say:**
- **Correlation evidence:**
  - Pearson r = 0.865 (p<10⁻⁹) - strong linear correlation
  - Spearman ρ = 0.931 (p<10⁻¹⁴) - even stronger, non-parametric, robust to outliers
  - 95% confidence intervals shown in figure
  - Homoscedastic residuals (constant variance)

- **Statistical power:**
  - N=30 devices gives df=28 degrees of freedom
  - Adequate for detecting medium-to-large effects
  - All comparisons p < 0.01
  - Mann-Whitney U: p<10⁻⁶⁰ for between-class vs within-class
  - 20× higher distinguishability between classes

- **NIST test paradox:**
  - All devices pass χ² test (values < 3.841 threshold)
  - All have high Shannon entropy (0.979-0.992 bits, near ideal 1.0)
  - Yet NN classifies at 59% with p<10⁻⁹
  
- **Key message:** Passing NIST tests is insufficient for ML-adversarial security
  - Need second-order statistics evaluation
  - Temporal pattern analysis beyond frequency tests

---

## **SLIDE 15: Proposed Attack Detection Framework (16:10-17:20) [70s]**
**On screen:**
- Two-column comparison: High-Quality RNG vs Degraded RNG
- Attack Type Signatures (4 cards)
- Proposed application note

**What to say:**
- **High-quality RNG profile (safe operation):**
  - Bell correlation ≥ 0.8 (high quantum fidelity)
  - Shannon entropy ≈ 0.99 bits (near ideal)
  - KL divergence ≈ 3.7 (stable baseline distribution)
  - Bit frequency: 50% ± 2% (within tolerances)

- **Degraded RNG profile (attack signatures):**
  - Bell correlation < 0.8 (increasing noise)
  - Entropy deviation > 5% from baseline
  - KL divergence > 17 (massive distribution shift)
  - Bit frequency: 59% (exploitable threshold from our analysis)

- **Attack type signatures (4 categories):**
  - Phase remapping: correlation drop + entropy oscillation pattern
  - Detector blinding: complete loss of quantum correlation (S → 2)
  - Temperature attack: gradual bias accumulation over time
  - RNG compromise: persistent frequency bias + Markov patterns

- **Proposed application:** Real-time statistical monitoring framework
  - Multi-indicator thresholds for early detection
  - Combines entropy + correlation + hardware metrics
  
- **Critical requirement:** Validation on 50+ production QKD RNGs needed
  - Current results on synthetic data only
  - Need long-term studies on real commercial systems

---

## **SLIDE 16: Proposed Application: Metro QKD Security Monitoring (17:20-18:10) [50s]**
**On screen:**
- Validated Methods section
- Two metric boxes: 59% accuracy, 20× distinguishability
- Critical gaps note (highlighted in red)

**What to say:**
- **What's validated (on synthetic data):**
  - Framework tested on N=30 synthetic devices
  - 59% classification accuracy, 80% above random, p<10⁻⁹
  - 20× distinguishability between device classes, p<10⁻⁶⁰
  - qGAN tournament confirms device clustering
  - Statistical signatures detectable despite all devices passing NIST tests

- **Critical gaps for real-world deployment:**
  1. Need 50+ real production QKD RNGs (not synthetic simulators)
  2. Long-term drift monitoring: months to years of continuous operation data
  3. Demonstration of actual key leakage detection (not just statistical patterns)

- **Honest assessment - and this is crucial:**
  - We detect patterns in RNG output
  - Have NOT demonstrated key extraction in practice
  - Gap between statistical fingerprinting and cryptanalysis is substantial
  - Would need: real-time basis prediction, correlation with key bits, information leakage quantification
  - Our work: first step identifying vulnerability, not complete attack

---

## **SLIDE 17: Bridging Theory & Engineering Reality (18:10-19:10) [60s]**
**On screen:**
- The Fundamental Gap box
- Two columns: This Work Addresses, Future Directions

**What to say:**
- **The fundamental gap:**
  - Mathematical excellence: CHSH-based QKD provides device-independent security guarantees with rigorous proofs
  - Engineering compromise: Real-world implementations rely on RNGs vulnerable to side-channel attacks
  - Our solution: Combines CHSH self-testing with ML-driven entropy monitoring to close this gap

- **This work addresses:**
  - Continuous RNG validation (not one-time certification at deployment)
  - Environmental factor monitoring (temperature, electromagnetic interference)
  - Hardware drift detection (gradual degradation over lifetime)
  - Real-time attack identification (before security breach)

- **Future directions:**
  - Photonic & topological qubits: different noise models, new challenges
  - Long-term degradation studies: 6-12 month continuous monitoring campaigns
  - Quantum ML for detection: quantum algorithms for RNG quality assessment
  - NIST/ISO standards development: integrating ML-based monitoring into certification frameworks

- This bridges academic security proofs with operational security requirements

---

## **SLIDE 18: Comprehensive Validation Summary (19:10-19:40) [30s]**
**On screen:**
- Six-panel comprehensive figure showing all main results

**What to say:**
- **Point to comprehensive figure (6 panels):**
  - Top left: bit frequency distributions across 3 devices
  - Top right: Markov transition matrices showing device-specific patterns
  - Middle left: qGAN tournament - 20× distinguishability between classes
  - Middle right: NN confusion matrix - 59% balanced accuracy
  - Bottom left: hardware correlation - R²=0.977 gate fidelity → Bell correlation
  - Bottom right: per-device performance - Device 3 has 70% accuracy despite highest entropy

- **Summary message:** Multi-method consistency, proper statistical power, N=3 to N=30 replication validates approach on synthetic data

---

## **SLIDE 19: Conclusions & Impact (19:40-21:00) [80s] - FINAL**
**On screen:**
- Five key contributions (numbered list)
- Impact statement
- Critical gap reminder (red box)
- Acknowledgments, references, contact

**What to say:**
- **Five key contributions:**
  
  1. **Device fingerprinting:** 59% accuracy on N=30 devices (73.2% balanced), 120% above baseline, r=0.865, p<10⁻⁹
  
  2. **Multi-method consistency:** qGAN, Logistic Regression, and Neural Network converge, ρ=0.931, p<10⁻¹⁴
  
  3. **qGAN tournament:** 20× higher distinguishability between classes (p<10⁻⁶⁰), within-class KL 0.077 vs between-class 1.60
  
  4. **Scalability:** N=3 (58.67%) replicates to N=30 (59%) with strong statistical significance
  
  5. **Proposed application:** ML-based continuous monitoring framework validated on synthetic data, requires real hardware testing

- **Impact statement:**
  - ML-based fingerprinting successfully distinguishes quantum noise profiles on synthetic data
  - Demonstrates vulnerability class that CHSH certification alone doesn't detect
  - Proposes operational security layer beyond theoretical proofs

- **Critical gap reminder (emphasize this):**
  - Detecting patterns ≠ exploiting for QKD attacks
  - We show RNGs have ML-detectable fingerprints
  - Have NOT demonstrated key leakage in production systems
  - Real security impact requires actual key extraction demonstration
  - Our work: vulnerability identification, not complete exploit

- **References:**
  - Zapatero et al. 2023 (Nature npj Quantum Information) - DI-QKD security analysis
  - DoraHacks YQuantum 2024 - quantum randomness generation challenge
  - github.com/hubertkolcz/NoiseVsRandomness - open-source repository

- **Acknowledgments:** Warsaw University of Technology, Texas A&M University, University of Toronto, DoraHacks

- **Contact:** hubert.kolcz.dokt@pw.edu.pl

**Thank you! Questions?**

---

## **TIMING SUMMARY**
- **Total: 19 slides in 21 minutes**
- **Slide 1 (Title):** 50s
- **Slides 2-19 (Content):** 1270 seconds (21 min 10 sec total with title)
- **Key technical slides:** 3 (80s), 4 (90s), 5 (90s), 8 (90s), 12 (90s)
- **Quickest:** Slide 18 (30s) - pointing to comprehensive figure
- **Buffer:** ~1 minute for transitions and unexpected delays

---

## **CRITICAL FRAMING - REPEAT THROUGHOUT**

1. **Data transparency:** N=3 from IBMQ simulators, N=30 synthetic - NOT real QPU data
2. **Gap emphasis:** Detecting patterns ≠ breaking security - key extraction not demonstrated
3. **Validation needs:** Requires 50+ production QKD RNGs for real-world validation
4. **Statistical rigor:** p<10⁻⁹ throughout, proper df=28, multi-method convergence
5. **Honest assessment:** First step identifying vulnerability, not complete attack

---

## **BACKUP Q&A RESPONSES**

**Q: "Have you tested on real quantum hardware?"**
A: N=3 dataset uses IBMQ noise-injected simulators with realistic noise models from DoraHacks 2024 challenge, not actual QPU data. N=30 entirely synthetic with controlled bias levels. Real hardware testing is critical next step - need 50+ production QKD RNGs from commercial vendors for validation.

**Q: "Can you actually extract secret keys?"**
A: No. We demonstrate ML can fingerprint RNGs at 59% accuracy on synthetic data. Gap between statistical fingerprinting and cryptanalysis is substantial. Real attack would require: (1) real-time basis prediction correlating with RNG patterns, (2) mapping basis choices to key bits, (3) demonstrating information leakage exceeding DI-QKD security bounds. We've done (1) on synthetic data only.

**Q: "Why is 59% accuracy significant if it's barely better than random?"**
A: For 3-way classification, random baseline is 33.3%. Our 59% represents 77% improvement (nearly doubling random performance) with p<10⁻⁹. Even small basis prediction improvements create security vulnerabilities. If adversary predicts 59% of basis choices correctly (vs 50% random), that's 18% information leakage potentially weakening DI-QKD security proofs which assume uniformly random basis selection.

**Q: "Devices pass NIST tests - isn't that sufficient for security?"**
A: Precisely our point. All devices pass χ² test (< 3.841 threshold) and have high Shannon entropy (0.979-0.992 bits near ideal 1.0), yet ML classifies at 59% with p<10⁻⁹. NIST tests designed for classical attacks, checking first-order statistics (frequency distributions). ML exploits second-order patterns: Markov transition biases, autocorrelation structures, run-length statistics. These temporal patterns invisible to NIST but exploitable by ML adversaries.

**Q: "What about DI-QKD security proofs accounting for weak randomness?"**
A: DI-QKD proofs DO incorporate weak-randomness analysis (Santha-Vazirani models, finite-size effects, min-entropy bounds like H∞=0.186). However: (1) they assume these bounds remain stable after one-time certification, (2) standard min-entropy doesn't capture temporal correlations (Markov memory, autocorrelation lag-20, multi-bit patterns), (3) no operational framework for continuous monitoring. Our attack exploits monitoring gap: RNGs may degrade from certified H∞=0.186 to exploitable H∞=0.05 without detection. ML framework provides missing continuous validation layer.

**Q: "How long until this becomes a real threat?"**
A: Current work demonstrates statistical vulnerability on synthetic data, not operational threat to deployed systems. Timeline to real threat depends on: (1) validation on 50+ production RNGs (2-3 years research), (2) developing real-time basis prediction from side channels (significant engineering), (3) demonstrating actual key leakage in field conditions (major research program). However, proactive security response should begin now: integrate ML-based continuous monitoring into next-generation QKD certification standards, don't wait for exploit demonstration.

**Q: "Why synthetic data instead of real QPU data?"**
A: Statistical power. Proper validation requires N≥30 samples per class for meaningful hypothesis testing (df=28). Commercial QPU access: expensive ($100-1000/hour), limited availability (weeks of queue time), restricted by vendor policies. DoraHacks challenge provided N=3 IBMQ simulator samples with realistic noise models. We extended to N=30 synthetic for statistical rigor. Trade-off: synthetic data lacks unknown real-world effects, but provides controlled validation of methodology. Next step: collaborate with QKD vendors for production RNG access.
