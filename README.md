# Poster: Verification of qRNG Using qGAN and Classification Models
Available under the link: [Poster](hubertkolcz/NoiseVsRandomness/Verification_of_qRNG_Using_qGAN_and_Classification_Models.pdf)

# Noise versus Randomness
YQuantum 2024 -- DoraHacks Challenge

Utilized the output from a quantum random number generator as training data - "AI_2qubits_training_data.txt", containing 6,000 samples (2,000 per device) from three quantum devices: **Rigetti Aspen-M-3** (79-qubit superconducting QPU, CHSH score 0.8036, gate fidelity 93.6%), **IonQ Aria-1** (25-qubit trapped ion QPU, CHSH score 0.8362, gate fidelity 99.4%), and an **IBM Qiskit simulator** with realistic noise injection. Each device generated 100-bit random number strings using Bell state entanglement. 

Explored the possibility of creating a model that classifies new random number output, and identifies which quantum device produced the random number. The quantum random number generator functions as described in the `quantum-randomness-generator/` folder (based on https://github.com/dorahacksglobal/quantum-randomness-generator). We compared different machine learning models and ran several tests to understand whether this classification task is possible. 
