from scipy.stats import chisquare
from qiskit_aer import AerSimulator
from copy import deepcopy
import numpy as np

def fault_detection1(fault_model, qc, distribution,
                    maximum_test_escape = 0.05, maximum_overkill = 0.05,
                    shots=100000):
    # 1. Run  simulation
    reference_distribution = fault_simulation(None, qc, shots=shots)

    # 2. Union of all keys in both distributions
    all_keys = sorted(set(reference_distribution) | set(distribution))

    # 3. Build aligned frequency vectors
    observed = np.array([distribution.get(k, 0) for k in all_keys], dtype=float)
    expected = np.array([reference_distribution.get(k, 0) for k in all_keys], dtype=float)

    # 4. Avoid zero in expected by applying a floor (Laplace smoothing)
    epsilon = 1e-6
    expected += epsilon
    observed += epsilon

    # 5. Normalize to the same number of shots
    observed *= (shots / observed.sum())
    expected *= (shots / expected.sum())

    # 6. Chi-square test
    chi2_stat, p_value = chisquare(f_obs=observed, f_exp=expected)

    # 7. Evaluate detection
    # Lower p => more likely the distributions differ => detect fault
    detect_fault = p_value < maximum_test_escape

    # 8. Optional: Apply overkill guard (only matters if you simulate healthy circuits for testing)
    if not detect_fault and p_value < maximum_overkill:
        # Uncertain region, could be overkill
        return False  # Conservative: don't declare a fault
    return detect_fault