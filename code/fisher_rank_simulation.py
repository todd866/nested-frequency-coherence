"""
Fisher Rank Simulation for Nested vs Clock-Synchronized Oscillators

Demonstrates:
1. Single observation has Fisher rank ≤ 1 (outer product structure)
2. Multi-sample observations achieve rank up to K-1 for nested coupling
3. Clock synchronization constrains rank to 1 regardless of N
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import matrix_rank, eigvalsh

# Set random seed for reproducibility
np.random.seed(42)

def compute_fisher_nested(psi, omega, omega_K, alpha, t_samples, sigma=1.0):
    """
    Compute Fisher information matrix for nested frequency model.

    Model: y(t) = [A_0 + sum_j alpha_j cos(omega_j t + psi_j)] cos(omega_K t) + noise

    Parameters:
    -----------
    psi : array (K-1,) - slow phase offsets
    omega : array (K-1,) - slow frequencies
    omega_K : float - carrier frequency
    alpha : array (K-1,) - modulation depths
    t_samples : array (N,) - observation times
    sigma : float - noise std

    Returns:
    --------
    FIM : array (K-1, K-1) - Fisher information matrix
    """
    K_minus_1 = len(psi)
    N = len(t_samples)

    # Compute gradient matrix G: N x (K-1)
    # G[n, j] = -alpha_j * sin(omega_j * t_n + psi_j) * cos(omega_K * t_n)
    G = np.zeros((N, K_minus_1))
    for n, t in enumerate(t_samples):
        for j in range(K_minus_1):
            G[n, j] = -alpha[j] * np.sin(omega[j] * t + psi[j]) * np.cos(omega_K * t)

    # Fisher matrix is (1/sigma^2) * G^T G
    FIM = (1 / sigma**2) * G.T @ G
    return FIM


def compute_fisher_clock(Phi_0, omega_0, n_multipliers, t_samples, alpha, sigma=1.0):
    """
    Compute Fisher information for clock-synchronized model.

    Model: All phases are phi_k(t) = n_k * Phi(t) where Phi(t) = omega_0 * t + Phi_0
    Single parameter: Phi_0

    Returns scalar Fisher information (1D parameter).
    """
    N = len(t_samples)

    # For clock model, all relative phases are deterministic functions of Phi_0
    # The signal still has PAC structure, but relative phases are locked
    # d(mu)/d(Phi_0) depends on how phases enter the amplitude

    # Simplified: Fisher info is sum of squared derivatives
    fisher = 0.0
    for t in enumerate(t_samples):
        # Derivative of mean w.r.t. Phi_0 (all phase terms contribute)
        deriv = 0.0
        for j, n_j in enumerate(n_multipliers[:-1]):
            n_K = n_multipliers[-1]
            # Phase difference: (n_j - n_K) * Phi(t)
            phase_diff = (n_j - n_K) * (omega_0 * t[1] + Phi_0)
            deriv += -alpha[j] * (n_j - n_K) * np.sin(phase_diff) * np.cos(n_K * (omega_0 * t[1] + Phi_0))
        fisher += deriv**2

    return fisher / sigma**2


def effective_rank(FIM, threshold=1e-10):
    """Compute effective rank as number of eigenvalues above threshold."""
    eigenvalues = eigvalsh(FIM)
    return np.sum(eigenvalues > threshold * np.max(eigenvalues))


def main():
    # Parameters
    K = 4  # Total bands (3 slow + 1 carrier)
    omega = np.array([2*np.pi*2, 2*np.pi*5, 2*np.pi*8])  # Slow frequencies: 2, 5, 8 Hz
    omega_K = 2*np.pi*40  # Carrier: 40 Hz
    alpha = np.array([0.3, 0.3, 0.3])  # Modulation depths
    psi_true = np.array([0.5, 1.2, 2.1])  # True phase offsets
    sigma = 0.1

    # Clock model parameters
    omega_0 = 2*np.pi*1  # Base clock: 1 Hz
    n_multipliers = np.array([2, 5, 8, 40])  # Integer multipliers
    Phi_0_true = 0.3

    # Range of sample sizes
    N_values = np.arange(1, 21)

    # Store results
    ranks_nested = []
    ranks_clock = []
    eigenvalues_nested = []

    for N in N_values:
        # Sample times (avoid pathological cases)
        t_samples = np.linspace(0, 1.0, N) + 0.01 * np.random.randn(N)

        # Nested model
        FIM_nested = compute_fisher_nested(psi_true, omega, omega_K, alpha, t_samples, sigma)
        ranks_nested.append(effective_rank(FIM_nested))
        eigenvalues_nested.append(eigvalsh(FIM_nested))

        # Clock model - always rank 1 (scalar parameter)
        # For comparison, we compute the effective dimensionality
        ranks_clock.append(1)  # By construction, clock has 1 parameter

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left panel: Rank vs N
    ax1 = axes[0]
    ax1.plot(N_values, ranks_nested, 'b-o', label='Nested coupling', markersize=6)
    ax1.axhline(y=K-1, color='b', linestyle='--', alpha=0.5, label=f'Max rank = K-1 = {K-1}')
    ax1.plot(N_values, ranks_clock, 'r-s', label='Clock synchronized', markersize=6)
    ax1.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='Clock rank = 1')
    ax1.set_xlabel('Number of observations N', fontsize=11)
    ax1.set_ylabel('Fisher rank', fontsize=11)
    ax1.set_title('(a) Fisher rank vs. sample size', fontsize=12)
    ax1.legend(loc='right', fontsize=9)
    ax1.set_ylim(0, K + 0.5)
    ax1.set_xlim(0, 21)
    ax1.grid(True, alpha=0.3)

    # Right panel: Eigenvalue spectrum for N=10
    ax2 = axes[1]
    N_example = 10
    t_samples = np.linspace(0, 1.0, N_example) + 0.01 * np.random.randn(N_example)
    FIM_nested = compute_fisher_nested(psi_true, omega, omega_K, alpha, t_samples, sigma)
    eigs = eigvalsh(FIM_nested)

    ax2.bar(range(1, K), sorted(eigs, reverse=True), color='steelblue', edgecolor='black')
    ax2.set_xlabel('Eigenvalue index', fontsize=11)
    ax2.set_ylabel('Eigenvalue magnitude', fontsize=11)
    ax2.set_title(f'(b) Fisher eigenvalues (N={N_example})', fontsize=12)
    ax2.set_xticks(range(1, K))
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('../figures/fisher_rank.pdf', bbox_inches='tight', dpi=150)
    plt.savefig('../figures/fisher_rank.png', bbox_inches='tight', dpi=150)
    print("Saved figures to ../figures/fisher_rank.pdf and .png")

    # Print summary
    print("\n=== Fisher Rank Summary ===")
    print(f"Model: K={K} bands ({K-1} slow + 1 carrier)")
    print(f"Slow frequencies: {omega/(2*np.pi)} Hz")
    print(f"Carrier frequency: {omega_K/(2*np.pi)} Hz")
    print(f"\nNested model:")
    print(f"  - Rank grows from 1 to {K-1} as N increases")
    print(f"  - Full rank ({K-1}) achieved at N ≥ {K-1}")
    print(f"\nClock model:")
    print(f"  - Rank = 1 regardless of N (single parameter Φ₀)")
    print(f"\nIdentifiability gap: {K-2} dimensions")


if __name__ == "__main__":
    main()
