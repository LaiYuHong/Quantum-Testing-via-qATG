from scipy.stats import entropy
import numpy as np
from copy import deepcopy
from qiskit_aer import AerSimulator

def normalize_counts(counts, shots):
    all_keys = sorted(counts.keys())
    probs = np.array([counts.get(k, 0) for k in all_keys], dtype=float)
    probs = probs / np.sum(probs)
    return dict(zip(all_keys, probs))

def bootstrap_kl_threshold(qc, reference_distribution, shots, n_bootstrap=100, alpha=0.01):
    """Estimate KL-divergence threshold for normal behavior via bootstrapping."""
    simulator = AerSimulator()
    probs_ref = normalize_counts(reference_distribution, shots)
    all_keys = sorted(probs_ref.keys())

    kl_samples = []
    for _ in range(n_bootstrap):
        # Simulate again from original circuit
        new_counts = simulator.run(deepcopy(qc), shots=shots).result().get_counts()
        probs_new = normalize_counts(new_counts, shots)

        # Align probabilities
        P = np.array([probs_new.get(k, 1e-9) for k in all_keys])
        Q = np.array([probs_ref.get(k, 1e-9) for k in all_keys])
        kl = entropy(P, Q)
        kl_samples.append(kl)

    # Compute percentile threshold (e.g., 99th percentile if alpha=0.01)
    return np.percentile(kl_samples, 100 * (1 - alpha))

def fault_detection2(fault_model, qc, distribution,
                    maximum_test_escape=0.01, maximum_overkill=0.01,
                    shots=100000, n_bootstrap=100):
    # Step 1: Simulate the reference distribution

    reference_distribution = fault_simulation(None, qc, shots=shots)

    # Step 2: Normalize both distributions
    P = normalize_counts(distribution, shots)
    Q = normalize_counts(reference_distribution, shots)
    all_keys = sorted(set(P) | set(Q))

    P_vec = np.array([P.get(k, 1e-9) for k in all_keys])
    Q_vec = np.array([Q.get(k, 1e-9) for k in all_keys])

    # Step 3: Compute KL divergence
    kl_div = entropy(P_vec, Q_vec)

    # Step 4: Estimate safe threshold using bootstrap
    kl_threshold = bootstrap_kl_threshold(qc, reference_distribution, shots, n_bootstrap, alpha=maximum_overkill)

    # Step 5: Fault is detected if KL divergence exceeds the threshold
    return kl_div > kl_threshold