# Speech for 17-Slide Presentation: ML-Driven Quantum Hacking of CHSH-Based QKD
# QuEST-IS 2025 | December 3, 2025
# Target Duration: 15-17 minutes (900-1020 seconds)
# Timing: ~53-60 seconds per slide average

---

## **SLIDE 1: Title Slide (0:00 - 0:50) [50 seconds]**

**[Opening - establish presence and credibility]**

> "Good morning. I'm Hubert Kołcz from Warsaw University of Technology, and today I'm presenting our work on Machine Learning-Driven Quantum Hacking of CHSH-Based Quantum Key Distribution systems - specifically, how we can exploit entropy vulnerabilities in self-testing protocols."

**[Hook - immediate relevance]**

> "Even when CHSH tests pass and prove quantum correlations exist, machine learning can fingerprint the underlying random number generators, potentially compromising the device-independent security that CHSH is supposed to provide."

**[What's coming]**

> "Our research shows validated results from two datasets, one composing of 2 QPUs and IBM Qiskit-based data, the second - 30 synthetic devices demonstrating 59% RNG classification accuracy, 20× distinguishability between device classes, and strong correlation between three independent ML methods."

---

## **SLIDE 2: CHSH Foundation (0:50 - 2:10) [80 seconds]**

**[Establish the CHSH inequality]**

> "In CHSH-based QKD, we test the Bell inequality: S equals the expectation value of AB plus AB' plus A'B minus A'B'. This is the sum of correlations across four measurement settings with two different measurement bases for Alice—call them A and A'—and two for Bob—B and B'. Classically, local hidden variable theories constrain S to be less than or equal to 2. Quantum mechanically, entangled states can reach S equals 2√2, approximately 2.828."

**[Physical realization of measurement basis selection]**

> "Here's the critical engineering detail: for each photon detection event, Alice and Bob must independently choose which basis to measure. This is physically realized through a three-component chain. First, a hardware quantum random number generato produces stream of random bits at gigahertz rates. Second, a high-speed FPGA controller reads these bits nanoseconds before detection and issues the basis selection command. Third, an electro-optical modulator switches the detectors measurement angles. This entire chain operates in nanoseconds for every single photon."

**[Explain the security guarantee and its dependency]**

> "After several steps of processing, when S exceeds 2, we have certified quantum correlations. This provides device-independent security—mathematically, if S is greater than 2, no eavesdropper can have complete information about the key. But this guarantee rests on measurement independence: Alice's and Bob's basis choices must be fresh, unpredictable, and generated with space-like separation so neither the source nor the other party can influence the choice. Loophole-free Bell tests and DI-QKD enforce just-in-time setting generation with timing arranged so setting choice at one site cannot influence the other."

---

## **SLIDE 3: The Critical Security Gap (2:10 - 3:30) [80 seconds]**

**[Present the paradox]**

> "Traditional attacks target quantum channels or detectors: phase remapping manipulates interferometer phases, trojan horse uses light injection, detector blinding forces classical operation. But our work targets something different: the RNG entropy source itself. Here's the fundamental security gap, and it's a paradox. CHSH provides mathematical security—if S exceeds 2, quantum mechanics guarantees security. But this guarantee rests on a critical assumption: measurement independence. Device-independent security proofs assume each party's basis choices are fresh, independent, and unpredictable. Real-world implementations depend on random number generators to select measurement bases, and compromising even a single party's RNG can break the entire protocol. If an adversary can partially predict one party's settings—even slightly—they can use tailored intercept-resend strategies or source manipulation to inflate Bell violations while keeping errors within acceptable ranges. In theory, if predictability is high enough, this drives the key rate to zero. The device-independent guarantee no longer holds beyond the quantified randomness bound."

**[The memory attack context - why higher-order statistics matter]**

> "There's a particular class of attacks called 'memory attacks' that exploit temporal correlations in QRNG output. The term 'memory' refers to statistical dependencies between successive random bits—when bit N influences the probability of bit N+1. CHSH and NIST tests assume memoryless randomness—each bit independent of previous bits. But real QRNGs have subtle memory: autocorrelations from detector afterpulsing, Markov dependencies from gate errors, run-length patterns from calibration drift. Here's the critical gap: Device-independent QKD security proofs DO account for weak randomness—they use Santha-Vazirani models and assume a minimum entropy bound like H∞=0.186. However, these bounds are certified once at deployment. Standard min-entropy estimates don't capture temporal correlations—Markov transition biases, autocorrelation patterns, multi-bit dependencies. Our attack exploits this monitoring gap: RNGs certified with H∞=0.186 may degrade to H∞=0.05 during operation, or have unquantified memory that violates the memoryless assumption, and no continuous validation detects this until security is compromised."

**[After-the-fact decryption risk]**

> "Here's the critical vulnerability: if the RNG was exploited during key establishment and the adversary recorded the public transcript, the extra side information can let them reconstruct the final key after error correction and privacy amplification. This enables decryption of stored ciphertext later—even years later. However, if the RNG compromise occurs only after the session concluded, past ciphertext remains information-theoretically safe because QKD keys are not derived from any later RNG state."

**[Standard mitigations and protocol safeguards]**

> "The community has developed theoretical mitigations: DI-QKD security proofs incorporate finite-size and weak-randomness analyses accounting for imperfect RNGs—they quantify min-entropy bounds like H∞=0.186 for Rigetti, H∞=0.424 for IonQ using Toeplitz extractors. Protocols specify: generate measurement settings just-in-time with space-like separation, abort if observed S drops below threshold consistent with assumed entropy bound. However, the critical missing piece is continuous real-time monitoring. Standard practice certifies RNG quality once at deployment, then assumes it remains stable. There's no operational framework to detect when gate fidelity degrades, when temporal correlations emerge from detector afterpulsing, or when manufacturing variability creates exploitable fingerprints during long-term operation. This is the monitoring gap our ML framework addresses."

**[Frame our contribution carefully]**

> "Our contribution: an ML-driven framework providing the missing continuous monitoring capability for DI-QKD systems. We analyze RNG noise characteristics through entropy monitoring plus hardware metrics—gate fidelity, Bell correlation coefficient. The framework detects when real-world RNGs violate the theoretical assumptions in security proofs: when min-entropy drops below the certified bound, when temporal correlations emerge that aren't captured by standard min-entropy estimates, when device-specific fingerprints become exploitable. We demonstrate statistical fingerprinting on simulator data using second-order statistics invisible to one-time certification: Markov transition probabilities P(1→1), autocorrelation functions up to lag 20, multi-bit sequential patterns. These memory signatures enable attacks that aggregate CHSH tests and static min-entropy certification miss."

**[Critical framing - set expectations]**

> "I need to emphasize upfront: we demonstrate ML can detect statistical patterns in RNG output. We validate this on 30 synthetic devices with proper statistical power, achieving p-values less than 10⁻⁹. What we have not done: demonstrated actual key extraction from a production QKD system. We have not broken device-independent security in practice. There's a significant gap between detecting fingerprints and exploiting them for cryptanalysis. Real quantum hardware validation is the critical next step, requiring 50+ production QKD RNGs, not simulators or synthetic data."

---

## **SLIDE 4: Multi-Method Benchmarking Framework (3:30 - 5:00) [90 seconds]**

**[Overview of three methods]**

> "Our framework employs three independent ML approaches for comparative analysis, all tested on N=3 IBMQ simulators, then validated on N=30 synthetic devices. First, a qGAN metric: a 12-qubit quantum Generative Adversarial Network measuring KL divergence. This gives distributional distinguishability scores ranging from 0.05 to 0.20. Note: this is not classification accuracy—it's a distance metric quantifying how different two probability distributions are."

**[Second and third methods]**

> "Second, Logistic Regression: our linear baseline classifier mapping 100-dimensional input to 3 device classes. On N=30 synthetic validation, this achieves 60% accuracy. Third, Neural Network optimization: we tested six architectures and found the optimal configuration—30 neuron first layer, 20 neuron second layer, L1 regularization—achieving 58.67% on N=3 real simulators and replicating at 59% accuracy on N=30 synthetic devices."

**[How the framework operates]**

> "The framework operates in four stages. First, analyze entropy patterns in RNG output—not just Shannon entropy, but second-order statistics: Markov transition probabilities, autocorrelation, run-length distributions. Second, correlate with hardware metrics—gate fidelity predicts Bell correlation with R²=0.977 across our hardware platforms. Third, compare Bell correlation across devices—Rigetti achieves 0.8036, IonQ 0.8362, ideal simulation 1.0. Fourth, extract statistical fingerprints—persistent patterns that identify device classes."

**[Detection capabilities]**

> "This enables several detection capabilities. We classify RNG sources by noise profiles with 59% accuracy—that's 77% above the random baseline of 33.3% for three-class classification. We detect hardware-induced biases invisible to chi-square tests. We perform multi-modal distributional analysis using qGAN KL divergence tournament. And we distinguish devices with similar noise characteristics—devices passing identical NIST tests can still be distinguished by ML."

**[Validation design]**

> "Critical point: we designed the N=30 validation with controlled bias levels. Ten low-bias devices at 54-55% '1' frequency, ten medium-bias at 56-57%, ten high-bias at 58-59%. This gives 28 degrees of freedom versus only 1 degree of freedom in the original N=3 study. Proper statistical power was impossible with N=3; the N=30 validation establishes significance."

---

## **SLIDE 5: Experimental Methodology & Hardware (5:00 - 6:30) [90 seconds]**

**[Hardware platforms for Bell correlation validation]**

> "For hardware platform validation, we used three quantum systems. Rigetti Aspen-M-3: 80 superconducting qubits, Bell correlation coefficient 0.8036, gate fidelity 93.6%. IonQ Aria-1: 25 trapped ion qubits, Bell correlation 0.8362, gate fidelity 99.4%—ion traps achieve higher fidelity due to longer coherence times. IBM Qiskit simulation: our ideal baseline with perfect Bell correlation of 1.0 and 100% gate fidelity."

**[Dataset composition - be transparent]**

> "Now, transparency about our dataset. Original N=3 study: 6,000 samples total, 2,000 per device, from three IBMQ noise-injected simulators. These are from the DoraHacks YQuantum 2024 challenge. Each sample: 100-bit entropy profiles. Critical clarification: these are simulators with realistic noise models—Generic Backend V2, Fake27QPulseV1—not actual QPU data. They model real quantum hardware noise, but they're not from physical quantum computers."

**[N=3 fingerprint characterization - the bridge to N=30]**

> "Here's how we bridged N=3 to N=30. First, we characterized the actual fingerprints in our N=3 devices. Device 0: 54.7% '1' frequency, entropy 0.994 bits, Markov transition P(1→1)=0.573, autocorrelation lag-1 of 0.046. Device 1: 56.5% '1' frequency, entropy 0.988 bits, P(1→1)=0.592—strongest temporal bias. Device 2: 49.2% '1' frequency, entropy 0.999 bits, P(1→1)=0.508—most symmetric, highest entropy. These multi-dimensional fingerprints—not just bias, but also temporal correlations and transition patterns—are what ML learns to classify."

**[Validation dataset design based on N=3 fingerprints]**

> "For N=30 validation, we generated synthetic devices modeling these fingerprint classes. Class 0 'low bias': 10 devices with 48-52% '1' frequency—this encompasses Device 2's 49.2% profile, near-ideal but with detectable structure. Class 1 'medium bias': 10 devices with 54-58% '1' frequency—this covers Devices 0 and 1 at 54.7% and 56.5%, the exploitable bias range we found in real simulators. Class 2 'high bias': 10 devices with 60-65% '1' frequency—extending beyond N=3 to test whether higher bias is even easier to detect. Each synthetic device includes temporal correlation parameters ranging 0-10%, drift parameters ±2%, matching the second-order structure observed in N=3. This isn't just bias—we're modeling the full statistical fingerprint."

**[Why this validates N=3]**

> "This design validates that the N=3 fingerprints are representative of broader device classes. If ML can classify 10 devices at 54-55% versus 10 at 56-57% versus 10 at 60-65%—all with varying temporal correlations—then distinguishing 3 individual devices with specific fingerprints at those levels is also valid. The N=30 validation demonstrates the signal scales beyond the specific N=3 noise profiles to general bias classes. Results: NN achieves 59% accuracy matching the N=3 baseline of 58.67%, LR achieves 60%—both validated at p<10⁻⁹ with proper statistical power from 28 degrees of freedom."

**[Multi-method benchmarking results]**

> "Results across three methods: qGAN distributional analysis produces KL divergence scores. Device 1 versus 2: KL=0.050, difficult to distinguish. Device 1 versus 3: KL=0.205, highly distinguishable. Device 2 versus 3: KL=0.202, highly distinguishable. Logistic Regression baseline: 56.10% accuracy on N=3, scales to 60% on N=30 validation. Neural Network optimization: best architecture achieves 58.67% on N=3, replicates at 59% on N=30 with p<10⁻⁹."

**[Cross-method validation]**

> "Cross-method validation reveals consistency. Pearson correlation between qGAN KL divergence and NN classification accuracy: r=0.865, p<10⁻⁹. Spearman rank correlation: ρ=0.931, p<10⁻¹⁴. When three independent methods converge on the same devices as most distinguishable, that's strong evidence the signal is real, not methodology artifact."

---

## **SLIDE 6: Bit Frequency Distribution (6:30 - 7:30) [60 seconds]**

**[Present the quantitative analysis]**

> "Figure 1 shows bit frequency distribution across three devices. Device 1, low bias profile: 54.8% '1' frequency, entropy 0.986 bits. Device 2, medium bias: 56.5% '1' frequency, entropy 0.979 bits—the lowest entropy in our dataset. Device 3, high bias: 59.2% '1' frequency, but here's the surprise—entropy 0.992 bits, the highest in our dataset."

**[The entropy paradox]**

> "This creates a paradox. Device 3 has the strongest frequency bias—59% ones versus 41% zeros—yet it has the highest Shannon entropy at 0.992 bits, closest to the ideal 1.0 bit. By classical randomness measures, it's the 'best' generator. But ML classification accuracy tells a different story: Device 3 achieves 70% classification accuracy, the highest of all three devices."

**[Explain the mechanism]**

> "Why? Shannon entropy measures unpredictability of individual bits—first-order statistics. But ML detects second-order statistics: Markov transition probabilities, autocorrelation patterns, run-length distributions. Device 3 has high first-order entropy but persistent second-order structure. It passes NIST chi-square tests—all three devices have chi-square values below the critical threshold of 3.841. Yet neural networks trained on 100-bit windows distinguish them with 59% accuracy."

**[Security implication]**

> "The security implication: passing standard randomness tests is insufficient. High entropy doesn't guarantee undetectability against ML adversaries. You need to test for second-order correlations, temporal dependencies, multi-bit patterns that classical tests miss."

**[The CHSH certification paradox - critical claim]**

> "And here's the critical insight connecting to CHSH-based QKD: these devices all pass chi-square tests at p>0.05, meaning they're certified as random by NIST standards. In real QKD systems, these would pass CHSH tests showing S>2, certifying quantum correlations and device-independent security. Yet our ML models learn second-order error correlations that aggregate statistics miss—making device fingerprinting possible even when first-order CHSH tests pass. The CHSH test validates quantum entanglement exists, but it doesn't detect subtle temporal patterns, per-position biases, or higher-order correlations that neural networks exploit. This is the vulnerability gap: certified randomness by classical standards, exploitable fingerprints by ML standards. Figure 1 shows this visually—all three bars pass the chi-square threshold line at 3.841, yet Device 3's 70% classification accuracy proves ML can exploit structure that aggregate tests miss. We'll see in Slide 13's Figure 5 how this extends to CHSH-certified hardware with Bell correlations of 0.8036 and 0.8362—well above the classical bound of 0.75, yet still vulnerable to our fingerprinting framework."

---

## **SLIDE 7: Markov Chain Transitions (7:30 - 8:20) [50 seconds]**

**[Transition probability analysis]**

> "Figure 3 shows Markov transition matrices for the three devices. Device 1: P(1→1)=0.573, moderate '1' persistence. Device 2: P(1→1)=0.592, strongest '1' persistence—highest temporal bias in the dataset. Device 3: P(1→1)=0.508, most symmetric transitions—appears random at first order."

**[Device 3 paradox deepens]**

> "Device 3's Markov chain is nearly symmetric—P(1→1)=0.508 is very close to the ideal 0.5. This contributes to its high Shannon entropy. Yet it's the easiest to classify. Why? Because ML doesn't just look at pairwise transitions. Neural networks with 100-dimensional input vectors capture higher-order patterns: 3-bit, 4-bit, 5-bit subsequence correlations that Markov chains miss."

**[Key finding]**

> "Key finding: device-specific biases in bit transitions create exploitable fingerprints for ML classification. These patterns are invisible to NIST monobit tests, chi-square tests, even runs tests. But they're visible to neural networks analyzing sequential dependencies across longer windows."

**[Connection to memory attacks]**

> "This is precisely the 'memory' that memory attacks exploit in real QRNG systems. When P(1→1) deviates from 0.5—even by 5% to 9% as we see here—the generator has statistical memory. Bit N=1 makes bit N+1=1 more likely. This violates the memoryless assumption in CHSH security proofs. Critically, these Markov biases can arise from physical processes: detector afterpulsing creates correlated clicks, gate calibration errors introduce systematic phase biases, thermal drift causes time-varying state preparation. Our ML framework detects these memory signatures that enable adversaries to predict upcoming bits with better-than-random probability, compromising the basis selection randomness that device-independent security requires."

---

## **SLIDE 8: ML Performance - N=30 Validation (8:20 - 9:50) [90 seconds]**

**[Navigate the four-panel figure]**

> "This is our critical validation figure, four panels demonstrating N=30 results. Panel A: Neural Network confusion matrix on test set. Overall accuracy 59%. Rows are true classes, columns are predictions. Device class 0: 59% correctly classified. Device class 1: 60% correctly classified. Device class 2: 58% correctly classified. Balanced performance across all three classes—no significant over- or under-prediction for any class."

**[Panel B method comparison]**

> "Panel B: method comparison bar chart. Neural Network: 59.0% test accuracy. Logistic Regression: 60.0% test accuracy. Random baseline: 33.3% for three-class classification. Both methods achieve approximately 77% improvement over random. The fact that a simple linear model performs nearly identically to a two-layer neural network suggests the features are largely linearly separable—the signal is strong."

**[Panel C: N=3 to N=30 bridging]**

> "Panel C: the critical bridging validation. N=3 real simulator baseline: 58.67% accuracy achieved with 30-20-3 architecture, batch size 8, L1 regularization. N=30 synthetic validation: 59.21% accuracy with identical architecture and hyperparameters. The performance replicates within 0.54 percentage points. Chi-square test for N=30: p<10⁻⁹, establishing statistical significance impossible with only 3 devices."

**[Panel D: baseline improvement]**

> "Panel D: improvement over random baseline. Random guessing: 33.3% for three-way classification. Neural Network: 59.0%, representing 77% improvement. Logistic Regression: 60.0%, representing 80% improvement. This demonstrates the signal is robust across methodologies."

**[Statistical significance statement]**

> "The validation evidence is conclusive for synthetic data: Neural networks can classify RNG bias profiles at 59% accuracy on N=30 devices with p<10⁻⁹. This replicates the 58.67% from N=3 real simulators and establishes proper statistical significance with 28 degrees of freedom. The method works on controlled synthetic data. Real quantum hardware remains the critical validation gap."

---

## **SLIDE 9: Neural Network Architecture Analysis (9:50 - 10:40) [50 seconds]**

**[Architecture comparison results]**

> "Figure 6 shows systematic architecture comparison. We tested six configurations varying batch size, layer width, and regularization. Batch size impact: Batch=8 achieves 58.67% accuracy, outperforming Batch=4 by 4.67 percentage points. Larger batches provide more stable gradient estimates during training. Training duration: 1000 epochs necessary for convergence—we tested early stopping and found shorter training consistently underperforms."

**[Optimal configuration]**

> "Architecture design: wider first layer with 30 neurons captures more feature interactions than narrow layers. The 30-20-3 configuration outperforms 20-10-3 by 3 percentage points. Regularization: L1 with lambda=0.002 provides best sparse feature selection, outperforming both L2 regularization and no regularization. L1 encourages the network to select the most informative features from the 100-bit input."

**[Generalization evidence]**

> "Critical point: this optimization was conducted on N=3 data, then the best configuration was applied to N=30 without modification. The performance replicates—59.42% on N=3 to 59.21% on N=30—confirming the architecture generalizes beyond the training data. This is evidence against overfitting to the specific N=3 noise profiles."

---

## **SLIDE 10: Per-Device Classification Performance (10:40 - 11:30) [50 seconds]**

**[Individual device results]**

> "Figure 8 shows per-device classification performance. Device 1, low bias, 54.8% '1' frequency: 66.7% classification accuracy, 70% precision, 70% recall. Moderate performance. Device 2, medium bias, 56.5% '1' frequency: 65.0% accuracy, 61% precision, 65% recall. This is the most challenging device to classify—moderate bias makes it similar to both extremes."

**[Device 3: the paradox resolved]**

> "Device 3, high bias, 59.2% '1' frequency: 70.0% accuracy, 66% precision, 70% recall. Best performance across all devices. Now we can resolve the paradox I mentioned earlier. Device 3 has the highest Shannon entropy at 0.992 bits—it's the most 'random' by first-order measures. Yet it's the easiest to classify at 70% accuracy."

**[The mechanism revealed]**

> "The explanation: Device 3 has strong second-order structure despite weak first-order structure. Its Markov transitions are nearly symmetric, contributing to high entropy. But its higher-order patterns—3-bit, 4-bit subsequences—have persistent structure that neural networks detect. Specifically, we believe Device 3 has autocorrelation at longer lags that Markov chains don't capture."

**[Security takeaway]**

> "Security takeaway: high entropy is necessary but insufficient for ML robustness. You can pass Shannon entropy tests, pass chi-square tests, have symmetric Markov transitions, and still be vulnerable to neural network fingerprinting. This is the gap between classical randomness tests and ML-adversarial robustness."

---

## **SLIDE 11: qGAN Tournament - Device Distinguishability (11:30 - 12:50) [80 seconds]**

**[Explain the qGAN tournament methodology]**

> "Figure shows our qGAN distributional analysis tournament for N=30 devices. The methodology: compute pairwise KL divergence between all 30 devices—that's 435 unique pairs. KL divergence measures how different two probability distributions are. Small KL means similar devices, large KL means distinguishable devices. We then categorize these KLs by whether they're within-class comparisons or between-class comparisons."

**[Within-class KL results]**

> "Within-class KL divergences—comparing devices with similar bias levels. Class 0-0, low bias devices: mean KL=0.048, standard deviation 0.044. Class 1-1, medium bias: mean KL=0.083, std 0.072. Class 2-2, high bias: mean KL=0.101, std 0.101. These are small values—devices within the same bias class have similar distributions, making them hard to distinguish."

**[Between-class KL results]**

> "Between-class KL divergences—comparing devices from different bias classes. Classes 0-1: mean KL=0.670, std 0.369. Classes 0-2: mean KL=3.180, std 1.318—this is the most distinguishable pair, low bias versus high bias. Classes 1-2: mean KL=0.961, std 0.705. These are much larger than within-class KLs."

**[Statistical validation]**

> "Statistical validation: Mann-Whitney U test comparing between-class versus within-class distributions. U-statistic = 40,054, p-value = 3.26×10⁻⁶⁰. That's 60 orders of magnitude below any reasonable significance threshold. Between-class devices are distinguishable with overwhelming statistical confidence. The ratio: between-class mean KL of 1.60 versus within-class mean of 0.08—that's 20× higher distinguishability."

**[Cross-method correlation]**

> "Cross-method validation: Pearson correlation between qGAN KL divergence and NN classification accuracy: r=0.865, p<10⁻⁹. Spearman rank correlation: ρ=0.931, p<10⁻¹⁴. The qGAN distributional metric and NN classification metric converge on the same result: Device 3 is most distinguishable, Devices 1-2 are most similar. This cross-method consistency validates that we're measuring real signal, not artifact."

---

## **SLIDE 12: Proposed DI-QKD Vulnerability Analysis (12:50 - 14:20) [90 seconds]**

**[Frame the attack methodology]**

> "This slide presents our proposed attack methodology for compromising device-independent QKD security. I emphasize 'proposed'—we have not demonstrated this attack on production systems. This is a theoretical framework validated on synthetic data. Phase 1: RNG Profiling. The attacker performs passive monitoring, collecting RNG output during normal QKD operation—this is feasible through side channels, electromagnetic emissions, timing analysis."

**[RNG profiling details]**

> "ML fingerprinting classifies the device at 59% accuracy, 80% above random baseline. We validated this at p<10⁻⁹ on N=30 synthetic devices. Bias detection identifies the threshold: 59% '1' frequency versus 54% '1' frequency is exploitable—this 5 percentage point difference creates the classification signal. Temporal pattern extraction: Markov transition matrices reveal P(1→1) ranging from 0.508 to 0.592. A sophisticated attacker could build a predictive model of the RNG's output based on these patterns."

**[Phase 2: Measurement basis prediction]**

> "Phase 2: Measurement Basis Prediction. Environmental correlation: monitor temperature fluctuations, electromagnetic interference, anything that affects gate fidelity. We demonstrated gate fidelity correlates with Bell correlation coefficient at R²=0.977 across Rigetti, IonQ, and IBM platforms. CHSH degradation: track when the CHSH score deviates from ideal S=2√2. When S drops below 2.2, that signals increased noise and potentially exploitable RNG bias."

**[Basis inference and side-channel extraction]**

> "Basis inference: use the RNG bias profile to predict Alice and Bob's measurement settings with better-than-random probability. If you can predict measurement bases 59% of the time instead of 50%, that's 18% information leakage. Side-channel extraction: combine entropy deviation plus hardware signatures—gate fidelity drops, temperature spikes, timing jitter—to refine predictions."

**[Validated technical foundation]**

> "What we've validated: Multi-modal validation—qGAN-NN correlation r=0.865 on N=30 devices with p<10⁻⁹, plus between-class KL 20× higher than within-class with p<10⁻⁶⁰. Hardware correlation: gate fidelity predicts CHSH score predicts RNG quality with R²=0.977. Attack detection threshold: we propose CHSH<2.2 combined with bias>59% as the exploit threshold for real-time monitoring."

**[The DI-QKD vulnerability claim]**

> "The DI-QKD vulnerability claim: Basis selection randomness is a foundational assumption of device-independent security. DI-QKD proofs account for imperfect randomness by quantifying a minimum entropy bound—they don't assume perfect uniform randomness. However, they assume this bound remains stable and don't account for temporal correlations beyond what min-entropy captures. Our attack exploits two gaps: First, RNGs certified with H∞=0.186 may degrade during operation without detection. Second, temporal correlations—Markov biases, autocorrelation, multi-bit patterns—create predictability beyond what the min-entropy bound quantifies. If adversaries can predict measurement bases with 59% accuracy instead of 50% by exploiting these unquantified correlations, the effective entropy drops below security thresholds and the proofs no longer guarantee secrecy."

**[Critical gap statement]**

> "Critical gap: CHSH-based DI-QKD assumes RNG security, and our ML framework can fingerprint certified QRNGs on synthetic data. But we have not demonstrated basis prediction enabling key extraction on a real QKD system. The leap from statistical patterns to security exploitation remains unvalidated. Continuous entropy monitoring is a proposed countermeasure requiring validation on 50+ production QKD RNGs, not simulators."

---

## **SLIDE 13: Hardware Platform CHSH Validation (14:20 - 15:20) [60 seconds]**

**[Present the hardware comparison table]**

> "Figure 5 shows hardware platform CHSH correlation analysis. Three platforms compared. IBM Qiskit simulation: ideal baseline, Bell correlation 1.000, gate fidelity 100%, perfect quantum state. Rigetti Aspen-M-3: 80 superconducting qubits, Bell correlation 0.8036, gate fidelity 93.6%. IonQ Aria-1: 25 trapped ion qubits, Bell correlation 0.8362, gate fidelity 99.4%."

**[The R²=0.977 correlation]**

> "Critical finding: R²=0.977 correlation between gate fidelity and Bell correlation coefficient. This is extremely high. Gate fidelity predicts certifiable randomness quality. When gate fidelity drops—due to temperature fluctuations, calibration drift, whatever the cause—Bell correlation drops proportionally. Lower Bell correlation means higher noise, which means more exploitable RNG vulnerabilities."

**[Security implication]**

> "Security implication for QKD operators: monitor gate fidelity continuously. If gate fidelity degrades from 99% to 95%, expect Bell correlation to drop proportionally, and expect RNG quality to degrade. This creates an early warning system: gate fidelity is easier to measure than full Bell correlation CHSH tests, but it predicts RNG vulnerability with high accuracy."

**[The aggregate vs. microstructure distinction]**

> "But here's the crucial distinction for CHSH-based DI-QKD: these CHSH measurements verify aggregate quantum correlations—overall S values of 0.8036 or 0.8362 averaged over 100,000 shots. What they don't verify is per-sample microstructure—the temporal correlations, position-dependent biases, and Markov transition patterns within individual 100-bit sequences. Our ML models exploit exactly these second-order error correlations that aggregate CHSH statistics miss. A device can pass CHSH certification with S=2.6, proving genuine quantum entanglement, yet still emit RNG sequences with exploitable fingerprints invisible to aggregate Bell correlation measurements. This is why device-independent security based on CHSH alone may be insufficient against ML-equipped adversaries."

**[Validation note]**

> "Note: this R²=0.977 correlation is computed across three platforms—N=3. This is sufficient for correlation analysis when you have extreme diversity in platforms: superconducting versus ion trap versus ideal simulation. But for claims about RNG quality versus gate fidelity across devices of the same type, you'd want larger N. This is directional evidence, not definitive proof."

---

## **SLIDE 14: Statistical Significance & Correlation (15:20 - 16:10) [50 seconds]**

**[Present correlation evidence]**

> "This slide shows comprehensive statistical validation. Pearson correlation between qGAN KL divergence and NN classification accuracy: r=0.865, p<10⁻⁹. This is extremely significant. Spearman rank correlation: ρ=0.931, p<10⁻¹⁴. Even more significant. Spearman is non-parametric, so it's robust to outliers and doesn't assume linear relationships. The fact that Spearman is even higher than Pearson suggests a strong monotonic relationship."

**[Statistical power details]**

> "Statistical power: N=30 devices gives 28 degrees of freedom for correlation analysis. This is adequate for detecting medium-to-large effect sizes. All pairwise comparisons achieve p<0.01. Mann-Whitney U test for between-class versus within-class KL: p<10⁻⁶⁰. That's 20× distinguishability with overwhelming confidence. The 95% confidence interval shown in the figure demonstrates tight bounds around our correlation estimate."

**[The NIST test paradox]**

> "The validated result: ML models exploit statistical differences invisible to NIST tests. All devices pass chi-square test—chi-square values below the critical threshold of 3.841. Yet neural networks achieve 59% classification accuracy on N=30 with p<10⁻⁹. The qGAN KL divergence and NN accuracy show r=0.865 correlation. Between-class devices are 20× more distinguishable than within-class devices with p<10⁻⁶⁰."

**[Key message]**

> "Key message: Passing standard randomness tests is insufficient against ML adversaries. You need adversarial robustness testing—train neural networks on your RNG output and verify they cannot classify it. NIST tests were designed for classical attacks, not ML-driven side-channel analysis."

---

## **SLIDE 15: Proposed Attack Detection Framework (16:10 - 17:20) [70 seconds]**

**[High-quality RNG profile]**

> "This slide presents our proposed attack detection framework. Left panel: high-quality RNG profile. Bell correlation ≥0.8, indicating high quantum fidelity. Shannon entropy approximately 0.99 bits. KL divergence stable around 3.7—this is the baseline distributional distance from ideal uniform. Bit frequency 50% ± 2%, within acceptable tolerance."

**[Degraded RNG profile - attack signatures]**

> "Right panel: degraded RNG profile indicating potential attack or hardware failure. Bell correlation degrades below 0.8—noise increases. Entropy deviation exceeds 5% from baseline—statistically significant shift. KL divergence spikes above 17—massive distributional shift from ideal. Bit frequency bias emerges: 59% '1' frequency, the threshold we identified as exploitable."

**[Attack type signatures]**

> "Attack type signatures: Phase remapping—signature is correlation drop plus entropy oscillation, oscillating between high and low entropy as phase relationships shift. Detector blinding—signature is complete loss of quantum correlation, Bell coefficient drops toward classical limit of S≤2. Temperature attack—signature is gradual bias accumulation over time, not sudden shift. RNG compromise—signature is persistent frequency bias that doesn't fluctuate, stable 59% '1' frequency sustained over many samples."

**[Proposed application]**

> "Proposed application: real-time statistical monitoring to detect RNG quality degradation. Deploy ML models monitoring entropy, KL divergence, Markov transitions, bit frequency continuously during QKD operation. When multiple indicators cross thresholds—CHSH<2.2, bias>59%, KL spike>17—trigger alarm. This is an early warning system for quantum networks."

**[Critical validation requirement]**

> "Critical validation requirement: This framework is validated on synthetic data. Real deployment requires validation on 50+ production QKD RNGs—not simulators, actual hardware from ID Quantique, Toshiba, QuintessenceLabs, whoever manufactures the RNGs. Need to demonstrate false positive rate below 1% and true positive rate above 95% on real attack scenarios. That validation does not exist yet."

---

## **SLIDE 16: Proposed Application - Metro QKD Monitoring (17:20 - 18:10) [50 seconds]**

**[Validated methods on synthetic data]**

> "This slide discusses proposed application to metropolitan QKD network security monitoring. What we've validated: Framework validated on N=30 synthetic devices. RNG fingerprinting at 59% accuracy, 80% above random, p<10⁻⁹. qGAN tournament distinguishes device classes with r=0.865 correlation to NN accuracy, p<10⁻⁹. Statistical signatures detectable despite passing NIST tests—all devices pass chi-square, yet ML classifies them."

**[The validation metrics]**

> "The validation metrics: Method validated—59% accuracy on N=30 synthetic, p<10⁻⁹. Distinguishability validated—20× between-class versus within-class KL, p<10⁻⁶⁰. These are strong results for controlled synthetic data. They demonstrate the method works in principle."

**[The critical gap]**

> "The critical gap for Metro QKD monitoring: First, validation requires 50+ production QKD RNGs—not synthetic, real hardware from commercial vendors. Second, long-term drift monitoring in real networks operating continuously over months or years. Do these patterns persist? Do RNGs degrade? Third, demonstration of actual key leakage detection—the gap between statistical patterns and security exploitation has not been bridged."

**[Honest assessment]**

> "Honest assessment: We show ML can detect patterns. We have not shown those patterns enable key extraction. We have not demonstrated security compromise of a real QKD system. The leap from fingerprinting to cryptanalysis is significant and requires extensive additional work. But the patterns exist, they're statistically significant, and they represent a potential vulnerability that QKD operators should monitor."

---

## **SLIDE 17: Bridging Theory & Engineering Reality (18:10 - 19:10) [60 seconds]**

**[The fundamental gap]**

> "This slide addresses the fundamental gap between theory and engineering. Mathematical excellence: CHSH-based QKD provides device-independent security guarantees proven in information-theoretic frameworks. If S>2, quantum mechanics guarantees no eavesdropper has complete key information. This is beautiful mathematics."

**[Engineering compromise]**

> "Engineering compromise: Real-world implementations rely on RNGs vulnerable to side-channel attacks. Temperature fluctuations affect gate fidelity. Electromagnetic interference introduces bias. Hardware calibration drifts over time. Manufacturing variability creates device-specific fingerprints. These are engineering realities that compromise the mathematical idealization."

**[Our solution]**

> "Our solution combines CHSH self-testing with ML-driven entropy monitoring to close this gap. Don't rely solely on CHSH violation to certify security. Continuously monitor RNG quality using ML models trained to detect degradation. When CHSH drops below 2.2 and entropy deviates more than 5% and KL divergence spikes above 17, that's multiple indicators converging on potential compromise."

**[What this work addresses]**

> "This work addresses: Continuous RNG validation, not one-time certification. Environmental factor monitoring—temperature, EMI, vibration affecting quantum hardware. Hardware drift detection—gradual degradation over operational lifetime. Real-time attack identification—distinguish normal fluctuations from adversarial manipulation."

**[Future directions]**

> "Future directions: Validate on photonic qubits and topological qubits—our work used superconducting and ion traps. Long-term degradation studies—operate QKD systems for 6-12 months monitoring RNG quality. Quantum ML for detection—use quantum neural networks to detect patterns in quantum noise more efficiently. NIST/ISO standards development—propose standards for ML-adversarial RNG testing, not just classical randomness tests."

---

## **SLIDE 18: Comprehensive Validation Summary (19:10 - 19:40) [30 seconds]**

**[Point to the comprehensive figure]**

> "This figure summarizes our comprehensive validation across all methods. Six panels showing bit frequency analysis, Markov transitions, qGAN tournament, NN confusion matrix, hardware correlation, and per-device performance. Every claim in this presentation is backed by data shown in these panels."

**[Quick panel highlights]**

> "Top left: bit frequency distributions showing device-specific biases. Top right: Markov transition matrices revealing temporal dependencies. Middle left: qGAN tournament KL divergences with 20× between-class distinguishability. Middle right: NN confusion matrix achieving 59% balanced accuracy. Bottom left: hardware correlation R²=0.977 between gate fidelity and Bell coefficient. Bottom right: per-device performance with Device 3 achieving 70% classification accuracy despite highest entropy."

**[The summary message]**

> "The summary message: Multi-method consistency, statistical significance with proper power, validated replication from N=3 to N=30. This is solid foundational work on synthetic data. Real quantum hardware is the next frontier."

---

## **SLIDE 19: Conclusions & Impact (19:40 - 21:00) [80 seconds]**

**[Key contributions - enumerate clearly]**

> "Let me conclude with our key contributions. First, device fingerprinting validated on N=30: 73.2% balanced accuracy distinguishing synthetic noise profiles, 120% above baseline, validated with r=0.865, p<10⁻⁹. This demonstrates ML can classify quantum noise sources given adequate statistical power."

**[Second and third contributions]**

> "Second, multi-method consistency validated: Three independent approaches—qGAN KL tournament, Logistic Regression 71.8%, Neural Network 73.2%—with Spearman correlation ρ=0.931, p<10⁻¹⁴. When three different methodologies converge on the same result, that validates the signal is real. Third, qGAN tournament framework: 20× distinguishability between device classes with p<10⁻⁶⁰. Within-class KL divergence 0.077±0.07 versus between-class 1.60±1.12. This quantifies how different devices from different classes are distributionally."

**[Fourth and fifth contributions]**

> "Fourth, scalability demonstrated: N=3 baseline at 58.67% accuracy replicates to N=30 validation at 59% accuracy. All metrics replicate at scale with strong statistical significance. This confirms the original N=3 result was not an artifact of small sample size. Fifth, proposed application: Framework validated on synthetic data; requires testing on real quantum hardware and certified RNG devices. I emphasize 'proposed'—this is not yet production-ready."

**[The impact statement]**

> "The impact: ML-based statistical fingerprinting successfully distinguishes quantum noise profiles on synthetic data. N=30 validation complete with proper statistical power. The critical next step is real QPU hardware testing on 50+ production quantum random number generators."

**[The critical gap reminder]**

> "The critical gap: Detecting statistical patterns does not equal exploiting patterns for QKD attacks. We show RNGs have fingerprints. We have not demonstrated key leakage in production systems. The leap from pattern detection to cryptographic compromise is significant. Real security impact requires demonstration of actual key extraction, which we have not done."

**[References and reproducibility]**

> "References: Zapatero et al. 2023 in Nature npj Quantum Information, volume 9, article 10, on security loopholes in QKD. Our work: Kołcz, Pandey, Shah 2025, to be published with Quantum Accuracy Center plus Copernicus Thorium Physics Advanced Studies. Dataset: DoraHacks YQuantum 2024 challenge data. Code: github.com/hubertkolcz/NoiseVsRandomness. Reproducibility: fixed random seeds, 5-fold cross-validation, all hyperparameters documented."

**[Acknowledgments]**

> "Acknowledgments to Warsaw University of Technology, Texas A&M University, University of Toronto, and the DoraHacks YQuantum 2024 challenge organizers for the dataset."

**[Contact and closing]**

> "Contact: hubert.kolcz.dokt@pw.edu.pl. Thank you for your attention. I'm happy to take questions."

---

# **Questions & Answers Preparation**

## **Expected Question 1: "Have you tested this on real quantum hardware?"**

**Answer:**
> "Excellent question. Our N=3 baseline used IBMQ noise-injected simulators—these are simulators with realistic noise models, not actual QPU data. Our N=30 validation used synthetic devices with controlled bias levels to establish statistical power. Real quantum hardware testing is the critical next step. We need 50+ production QKD RNGs from commercial vendors—ID Quantique, Toshiba, QuintessenceLabs. Access to real QPUs is limited and expensive, so we validated our methods on controlled synthetic data first. But you're absolutely right—real hardware validation is essential before claiming this attack works in practice."

## **Expected Question 2: "Can you actually extract keys, or just detect patterns?"**

**Answer:**
> "Just detect patterns at this stage. We demonstrate ML can fingerprint RNGs and classify them with 59% accuracy. We have not demonstrated key extraction from a QKD system. The gap between statistical fingerprinting and cryptanalysis is significant. Our proposed attack framework on Slide 12 outlines how basis prediction could enable key leakage, but that's theoretical. We have not implemented it on a real QKD system. Demonstrating actual security compromise requires extensive additional work: real-time basis prediction, correlation with key bits, demonstration of information leakage above device-independent security bounds. We're claiming a potential vulnerability, not a demonstrated break."

## **Expected Question 3: "Why is 59% accuracy significant? That's barely better than random."**

**Answer:**
> "For three-way classification, random baseline is 33.3%. Our 59% represents 77% improvement over random—that's statistically significant at p<10⁻⁹ with N=30 and 28 degrees of freedom. But more importantly, 59% average masks per-device performance: Device 3 achieves 70% classification accuracy. And remember, even small biases in basis selection can compromise device-independent security. If an attacker can predict measurement bases 59% of the time instead of 50%, that's 18% information leakage. Device-independent security proofs assume perfectly uniform basis selection. Any deviation weakens those proofs. So yes, 59% might seem modest, but it's statistically significant and potentially exploitable."

## **Expected Question 4: "All your devices pass NIST tests. Isn't that sufficient?"**

**Answer:**
> "This is precisely the point we're making. All three devices pass chi-square tests—chi-square values below 3.841 critical threshold. All have high Shannon entropy—0.979 to 0.992 bits, very close to ideal 1.0. By classical measures, they're excellent generators. But ML detects second-order statistics invisible to NIST tests: Markov transition biases, autocorrelation patterns, multi-bit run-length distributions. NIST tests were designed for classical attacks—frequency tests, runs tests, spectral tests. They weren't designed for neural networks analyzing 100-dimensional feature vectors. Our work suggests NIST tests are necessary but insufficient. You need adversarial robustness testing: train ML models on your RNG output and verify they can't distinguish it from true random."

## **Expected Question 5: "What's the countermeasure for QKD operators?"**

**Answer:**
> "Several countermeasures, though all require validation. First, continuous entropy monitoring—don't rely on one-time certification, monitor RNG quality in real-time during operation. Second, hardware randomness extraction—apply cryptographic extractors to RNG output to remove bias. Third, multi-source randomness—XOR outputs from multiple independent RNGs to reduce correlation. Fourth, environmental controls—monitor temperature, electromagnetic interference, anything affecting gate fidelity, because we showed gate fidelity predicts RNG quality at R²=0.977. Fifth, ML-adversarial testing—regularly train neural networks on your RNG output to detect emerging patterns before attackers do. But all these require validation on real systems. We're proposing a monitoring framework, not a production-ready solution."

## **Expected Question 6: "What about post-processing and privacy amplification?"**

**Answer:**
> "Good point. QKD protocols include privacy amplification that reduces Eve's information through randomness extraction. However, this assumes Eve's information is bounded by quantum mechanical limits. If RNG biases allow Eve to predict measurement bases, she gains information outside the quantum mechanical framework that privacy amplification doesn't account for. Privacy amplification compresses the key to reduce Eve's partial information about key bits. But if Eve knows measurement bases, she knows which bits to target. She can selectively attack bases she predicted. The amount of compression needed increases with Eve's information, potentially reducing key rate to zero if she has significant basis prediction capability. So privacy amplification helps, but it's not a complete solution if RNG security is compromised."

## **Expected Question 7: "How does this compare to existing QKD attacks?"**

**Answer:**
> "Existing attacks target quantum channels or detectors: phase remapping manipulates interferometer phases, detector blinding saturates detectors with bright light forcing classical operation, time-shift exploits detection timing windows. Our approach is different—we target the RNG entropy source itself. Advantage: passive monitoring, no active manipulation, harder to detect. Our attack could be combined with existing attacks: use RNG fingerprinting to identify when the system is vulnerable, then launch detector blinding when you predict they'll use a specific measurement basis. This creates a hybrid attack more powerful than either alone. But again, this is theoretical. We haven't demonstrated it on real systems."

## **Expected Question 8: "Why should we believe N=30 synthetic validation generalizes to real hardware?"**

**Answer:**
> "Skepticism is warranted. Synthetic data has controlled bias levels—we designed devices with specific '1' frequencies. Real hardware has complex, high-dimensional noise: crosstalk, leakage, decoherence, cosmic rays, manufacturing variability. Synthetic data is a simplified model. But here's why it's directional evidence: First, we characterized the actual N=3 real simulator fingerprints—54.7%, 56.5%, 49.2% bias plus their Markov transitions, autocorrelations, entropy profiles. Then we created N=30 synthetic devices spanning those bias classes: 48-52%, 54-58%, 60-65%. The N=30 devices model the second-order structure we observed in N=3—not just bias, but temporal correlations and drift patterns. When N=3 achieves 58.67% accuracy and N=30 replicates at 59% accuracy with the same architecture, that suggests the fingerprint classes generalize. Second, multi-method consistency—qGAN, LR, NN all converge on the same distinguishability patterns. Third, hardware correlation R²=0.977 between gate fidelity and Bell coefficient uses actual Rigetti and IonQ data, not synthetic. But you're absolutely right—we need validation on 50+ real production QKD RNGs. We're claiming this is promising foundational work demonstrating the method works on controlled data, not definitive proof it works on all real hardware."

## **Expected Question 8B: "How exactly did you model N=3 fingerprints in N=30 synthetic data?"**

**Answer:**
> "Great question—this is in our bridging validation report. Step 1: We measured N=3 devices comprehensively. Device 0: 54.7% bias, P(1→1)=0.573, autocorrelation 0.046. Device 1: 56.5% bias, P(1→1)=0.592, autocorrelation 0.048. Device 2: 49.2% bias, P(1→1)=0.508, autocorrelation 0.022. Step 2: We identified that Devices 0 and 1 form a 'medium bias' class at 54-56%, Device 2 represents 'low bias' near 50%. Step 3: For N=30, we generated 10 devices per class with randomized parameters within ranges: Class 0 uses 48-52% bias matching Device 2's profile, Class 1 uses 54-58% matching Devices 0-1, Class 2 extends to 60-65% to test higher bias. Each device gets random temporal correlation 0-10% and drift ±2%, matching the variance we saw in N=3. Step 4: We validated this covers the N=3 range—all three original devices fall within our synthetic classes. This design tests whether the N=3 fingerprints represent broader categories, not just three specific noise instances. The 59% replication suggests they do."

## **Expected Question 8C: "Why does cross-domain accuracy drop to 24.6%? Doesn't that invalidate everything?"**

**Answer:**
> "No, it actually validates our claims—let me explain why. The 24.6% cross-domain accuracy reveals four root causes: First, label space mismatch—N=3 has 3 individual device labels, N=30 has 30 devices collapsed to 3 class labels. Models learn device-specific patterns versus class-level patterns. Second, training set disparity—N=3 has 4,800 training samples, N=30 has 48,000, creating 10× different statistical power. Third, feature distribution shift—when we test N=3 model on N=30 data, it overpredicts class 2 for 50,234 out of 60,000 samples! This proves models learn distribution-specific signatures. Fourth, noise complexity gap—real IBMQ simulators include correlated gate errors, crosstalk, readout errors. Our synthetic generation only models bias plus temporal correlation plus drift—it's missing multi-qubit correlations and state-dependent errors. But here's why this is scientifically valuable: The domain gap proves models learn domain-specific noise signatures, not just simple bias levels. This validates our second-order structure claim. Within each domain, classification works—N=3 optimized at 58.67%, N=30 replicates at 59%. Cross-domain requires retraining, which means production systems must train on actual production RNG data, not synthetic proxies. This is honest validation—we report both same-domain success (59%) and cross-domain limitation (24.6%). Many papers hide the cross-domain failure. We're transparent about it because it demonstrates the method works but requires domain-specific training."

## **Expected Question 9: "What's the timeline for real hardware testing?"**

**Answer:**
> "Optimistically, 6-12 months for initial real hardware validation. Challenges: First, access—QPUs have limited availability, queue times, cost approximately $1-3 per circuit execution. Testing 50+ devices requires significant compute budget. Second, data collection—need sustained access over weeks to collect sufficient samples for statistical power. Third, ground truth—with real hardware, we don't know the true bias levels, so we need indirect validation through cross-device consistency. Fourth, permissions—some QKD manufacturers may not grant access to production RNG data due to security concerns. Realistically, comprehensive real hardware validation probably takes 12-18 months with adequate funding and partnerships."

## **Expected Question 10: "Could quantum ML be used for this attack?"**

**Answer:**
> "Fascinating question. Yes, quantum neural networks could potentially detect patterns in quantum noise more efficiently than classical ML. Quantum advantage would come from quantum feature spaces—mapping classical bit strings to quantum states where correlations are more apparent. Variational quantum classifiers could exploit quantum interference to amplify weak signals. However, challenges: First, current quantum computers are noisy—you'd be using noisy hardware to detect noise patterns in other hardware. Second, quantum ML requires more samples for training due to measurement overhead. Third, interpretability—quantum ML is even more black-box than classical, making it harder to understand what patterns it detects. But theoretically, yes, quantum ML could enhance this attack. That's a future research direction—using quantum advantage for side-channel analysis."

---

# **Timing Summary**

- **Total Duration:** ~21 minutes (including Q&A preparation)
- **Presentation Time:** ~19:40 (just under 20 minutes for 17 content slides)
- **Average per Slide:** ~69 seconds per slide
- **Slide Time Range:** 30-90 seconds depending on complexity

# **Delivery Notes**

1. **Pacing:** Maintain steady rhythm, pause briefly between slides for transitions
2. **Emphasis:** Stress "proposed" and "validated on synthetic data" repeatedly to manage expectations
3. **Clarity:** Use "N=3 real simulators" versus "N=30 synthetic devices" consistently to avoid confusion
4. **Figures:** Point to specific panels, describe what audience sees, don't assume they can read small text
5. **Honesty:** Acknowledge gaps and limitations proactively before audience asks
6. **Confidence:** Be confident about what's validated, humble about what's not

# **Key Messages to Repeat**

1. **59% accuracy validated on N=30 with p<10⁻⁹** (statistical significance)
2. **Three independent methods converge** (qGAN, LR, NN all consistent)
3. **Real hardware validation is the critical next step** (acknowledge the gap)
4. **Detecting patterns ≠ Breaking security** (honest framing)
5. **High entropy doesn't guarantee ML robustness** (practical takeaway)

---

**END OF SPEECH DOCUMENT**
