"""
Analysis of Assumptions Behind the "50+ Production QKD RNGs" Requirement
"""

print("\n" + "="*80)
print("ASSUMPTIONS ANALYSIS: Metro QKD Monitoring Requirements")
print("="*80)

print("""
STATEMENT CONTEXT:
"Metro QKD monitoring requires: (1) validation on 50+ production QKD RNGs 
(not synthetic), (2) long-term drift monitoring in real networks, (3) 
demonstration of actual key leakage detection (gap between statistical 
patterns and security exploitation not bridged)"

""")

print("="*80)
print("REQUIREMENT 1: 50+ Production QKD RNGs (Not Synthetic)")
print("="*80)

print("""
UNDERLYING ASSUMPTIONS:

1. STATISTICAL POWER ASSUMPTION:
   • N=30 synthetic devices provide adequate df=28 for validation
   • Production systems need LARGER sample size (50+) because:
     - Real hardware has more complex noise sources
     - Environmental factors add variance
     - Manufacturer differences increase heterogeneity
   • Assumes 50+ is needed to maintain 80% statistical power

2. GENERALIZATION ASSUMPTION:
   • Synthetic data with controlled bias (0.54-0.59) represents reality
   • Real QKD RNGs may have different noise characteristics:
     - Photonic sources (photon number splitting)
     - Thermal noise in detectors
     - Clock jitter in time-binning
     - Side-channel electromagnetic leakage
   • Assumes synthetic → production gap exists and is significant

3. DIVERSITY ASSUMPTION:
   • 50+ systems should cover:
     - Multiple manufacturers (ID Quantique, Toshiba, QuantumCTek, etc.)
     - Different architectures (discrete variable, continuous variable)
     - Various deployment scenarios (fiber, free-space, satellite)
     - Different network conditions (metro, long-haul)
   • Assumes current study's 3 devices insufficient for generalization

4. CERTIFICATION STANDARD ASSUMPTION:
   • 50+ aligns with industry practices:
     - NIST typically validates on 10-100 devices
     - ISO/IEC standards require multi-vendor testing
     - Common Criteria evaluation requires diverse test beds
   • Assumes 50 is threshold for practical acceptance

5. FALSE POSITIVE CONTROL ASSUMPTION:
   • With 30 synthetic devices: p<10⁻⁶⁰ → extremely low false positive
   • With 50+ production devices:
     - More heterogeneity → higher variance
     - Need larger N to maintain same confidence
     - Multiple testing corrections become critical
   • Assumes Bonferroni/FDR corrections require 50+ for robustness

""")

print("="*80)
print("REQUIREMENT 2: Long-term Drift Monitoring in Real Networks")
print("="*80)

print("""
UNDERLYING ASSUMPTIONS:

1. TEMPORAL STATIONARITY ASSUMPTION:
   • Current study: cross-sectional (snapshot at one timepoint)
   • Real systems: non-stationary (drift over weeks/months/years)
   • Assumes short-term patterns ≠ long-term patterns
   • Temperature cycling, component aging, calibration drift

2. ENVIRONMENTAL COUPLING ASSUMPTION:
   • Lab conditions: controlled temperature, EM shielding, stable power
   • Production networks: 
     - Temperature: -20°C to +60°C in telecom vaults
     - Humidity: affects optical coupling efficiency
     - EM interference: from adjacent telecom equipment
     - Power fluctuations: grid quality varies by location
   • Assumes environment significantly affects RNG statistics

3. LONGITUDINAL MONITORING ASSUMPTION:
   • Need time-series data to distinguish:
     - Normal drift (aging, calibration)
     - Malicious manipulation (slow attacks)
     - Sudden failures (component breakdown)
   • Assumes baseline must be established over time, not instantly

4. ATTACK DETECTION TIME WINDOW:
   • Assumes adversary might use "slow poisoning" attacks:
     - Gradually degrade QBER over months
     - Stay below alarm thresholds
     - Exploit accumulated bias in key material
   • Real-time monitoring needed to detect 0.1% daily drift

5. OPERATIONAL REALITY ASSUMPTION:
   • Warsaw network example: 70km, 6 nodes, 24/7 operation
   • Assumes lab validation ≠ operational validation
   • Need to handle:
     - Network topology changes
     - Node failures and recovery
     - Maintenance windows
     - Traffic load variations

""")

print("="*80)
print("REQUIREMENT 3: Demonstration of Actual Key Leakage Detection")
print("="*80)

print("""
UNDERLYING ASSUMPTIONS:

1. DETECTION ≠ EXPLOITATION GAP:
   • Current achievement: 59.42% device classification (N=3)
   • This shows: ML can detect statistical patterns
   • Does NOT show: patterns leak key information
   • Assumes: pattern detection is necessary but not sufficient

2. INFORMATION-THEORETIC SECURITY GAP:
   • QKD security proof: based on QBER, basis disclosure
   • ML fingerprinting: based on RNG bias, entropy, transitions
   • These are DIFFERENT information channels
   • Assumes: need to prove ML-detected patterns → key leakage

3. QUANTIFICATION ASSUMPTION:
   • Need to measure in security-relevant units:
     - "59% accuracy" → X bits of key compromised per Y samples
     - "KL divergence 0.05" → Z bits of mutual information
     - "Markov P(1→1)=0.572" → basis prediction advantage
   • Assumes: must bridge statistics → cryptanalysis

4. ATTACK MODEL COMPLETENESS:
   • Study shows: honest devices can be distinguished
   • Security needs: malicious devices can be detected
   • Gap: honest distinguishability ≠ malicious detectability
   • Assumes: adversarial noise differs from honest noise

5. PRACTICAL EXPLOIT DEMONSTRATION:
   • Academic requirement: show actual attack scenario
   • Need to demonstrate:
     - Adversary learns device fingerprint from public data
     - Uses fingerprint to predict basis choices
     - Extracts partial key information
     - Quantify: "X% accuracy → Y bits leaked per Z rounds"
   • Assumes: theoretical possibility ≠ practical exploit

6. THRESHOLD SETTING ASSUMPTION:
   • When does 59.42% accuracy become a security threat?
   • Need operational thresholds:
     - >54% (DoraHacks goal): "interesting but not alarming"
     - >60%: "investigate device quality"
     - >70%: "halt operations, re-certify device"
     - >80%: "assume compromised, discard keys"
   • Assumes: thresholds must be empirically validated, not guessed

""")

print("="*80)
print("WHY THESE ASSUMPTIONS MATTER")
print("="*80)

print("""
INTELLECTUAL HONESTY:
   The presentation explicitly acknowledges these gaps to:
   • Distinguish exploratory findings from operational claims
   • Set realistic expectations for practical deployment
   • Guide future research directions
   • Maintain scientific credibility

CURRENT STATUS:
   ✓ VALIDATED: ML can detect statistical patterns (N=30, p<10⁻⁹)
   ✓ VALIDATED: Patterns correlate across methods (r=0.865)
   ✓ VALIDATED: Effect size is large (20.8× distinguishability)
   
   ✗ NOT VALIDATED: Patterns exist in production QKD systems
   ✗ NOT VALIDATED: Patterns persist over time
   ✗ NOT VALIDATED: Patterns enable key extraction

RESEARCH ROADMAP:
   Near-term (1-2 years):
   • Secure 50+ production QRNG datasets from QKD vendors
   • Deploy monitoring on test networks (e.g., Warsaw)
   • Establish baseline drift characteristics

   Mid-term (2-5 years):
   • Demonstrate information leakage quantification
   • Develop operational thresholds
   • Integrate into QKD protocols as health check

   Long-term (5+ years):
   • Standardize in ISO/IEC QKD specifications
   • Deploy in commercial metropolitan QKD networks
   • Achieve continuous certification paradigm

""")

print("="*80)
print("BOTTOM LINE")
print("="*80)

print("""
The "50+ production QKD RNGs" requirement is NOT arbitrary. It reflects:

1. Statistical rigor: N=50 provides adequate power across diverse hardware
2. Engineering realism: Lab results must validate in operational networks
3. Security completeness: Pattern detection must prove key leakage risk
4. Industry standards: Aligns with certification body expectations
5. Scientific honesty: Acknowledges gap between proof-of-concept and deployment

The presentation makes a VALIDATED scientific contribution (pattern detection
on synthetic data) while being transparent about the UNVALIDATED security 
application (operational QKD monitoring). This is good science communication.
""")

print("="*80 + "\n")
