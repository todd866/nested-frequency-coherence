"""
Cerebellar Takens Embedding Simulation

Demonstrates how the cerebellum's parallel fiber architecture could implement
delay-coordinate embedding (Takens' theorem) to reconstruct high-dimensional
dynamics from low-frequency modulation signals.

Key insight:
- Parallel fibers have systematically varying conduction delays (different lengths)
- This is structurally equivalent to delay embedding: x(t), x(t-τ), x(t-2τ), ...
- A single slow-frequency signal, tapped at multiple delays, reconstructs high-D dynamics

This simulation shows:
1. A chaotic system (Lorenz) as ground-truth high-D dynamics
2. A single 1D projection (what a "slow wave" might carry)
3. Delay embedding reconstruction (what parallel fibers could compute)
4. Comparison of reconstructed vs original attractor geometry
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.spatial.distance import pdist, squareform
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(42)


def lorenz(state, t, sigma=10, rho=28, beta=8/3):
    """Lorenz system dynamics."""
    x, y, z = state
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]


def rossler(state, t, a=0.2, b=0.2, c=5.7):
    """Rossler system dynamics."""
    x, y, z = state
    return [-y - z, x + a * y, b + z * (x - c)]


def delay_embed(signal, tau, m):
    """
    Create delay embedding of a 1D signal.

    Parameters:
    -----------
    signal : array (N,) - time series
    tau : int - delay in samples
    m : int - embedding dimension

    Returns:
    --------
    embedded : array (N - (m-1)*tau, m) - delay-embedded signal
    """
    N = len(signal)
    n_points = N - (m - 1) * tau
    embedded = np.zeros((n_points, m))
    for i in range(m):
        embedded[:, i] = signal[i * tau : i * tau + n_points]
    return embedded


def correlation_dimension_estimate(embedded, r_values=None):
    """
    Estimate correlation dimension using correlation integral.
    C(r) ~ r^d as r -> 0
    """
    distances = pdist(embedded)
    if r_values is None:
        r_values = np.logspace(np.log10(distances.min() + 0.01),
                               np.log10(distances.max()), 20)

    C_r = []
    for r in r_values:
        C_r.append(np.mean(distances < r))
    C_r = np.array(C_r)

    # Estimate dimension from log-log slope
    mask = C_r > 0
    log_r = np.log(r_values[mask])
    log_C = np.log(C_r[mask])

    # Linear fit in scaling region (middle portion)
    mid = len(log_r) // 4
    end = 3 * len(log_r) // 4
    if end - mid > 2:
        coeffs = np.polyfit(log_r[mid:end], log_C[mid:end], 1)
        return coeffs[0], r_values, C_r
    return np.nan, r_values, C_r


def manifold_preservation_score(original, reconstructed):
    """
    Measure how well local neighborhoods are preserved.
    Uses k-nearest neighbor consistency.
    """
    k = 10
    n = min(len(original), len(reconstructed))
    original = original[:n]
    reconstructed = reconstructed[:n]

    # Find k-NN in each space
    dist_orig = squareform(pdist(original))
    dist_recon = squareform(pdist(reconstructed))

    nn_orig = np.argsort(dist_orig, axis=1)[:, 1:k+1]
    nn_recon = np.argsort(dist_recon, axis=1)[:, 1:k+1]

    # Count preserved neighbors
    preserved = 0
    for i in range(n):
        preserved += len(set(nn_orig[i]) & set(nn_recon[i]))

    return preserved / (n * k)


def simulate_cerebellar_processing(slow_signal, fiber_delays, dt):
    """
    Simulate parallel fiber delay-tap architecture.

    Parameters:
    -----------
    slow_signal : array - incoming slow oscillation (1D)
    fiber_delays : array - delays for each parallel fiber (in ms)
    dt : float - time step (in ms)

    Returns:
    --------
    fiber_outputs : array (n_timepoints, n_fibers) - tapped signals
    """
    n_fibers = len(fiber_delays)
    delay_samples = (fiber_delays / dt).astype(int)
    max_delay = np.max(delay_samples)
    n_out = len(slow_signal) - max_delay

    fiber_outputs = np.zeros((n_out, n_fibers))
    for i, d in enumerate(delay_samples):
        fiber_outputs[:, i] = slow_signal[max_delay - d : max_delay - d + n_out]

    return fiber_outputs


def main():
    print("=== Cerebellar Takens Embedding Simulation ===\n")

    # Generate Lorenz attractor as "true" high-D dynamics
    print("1. Generating Lorenz attractor (ground truth high-D dynamics)...")
    t_span = np.linspace(0, 100, 10000)
    dt = t_span[1] - t_span[0]
    x0 = [1.0, 1.0, 1.0]
    trajectory = odeint(lorenz, x0, t_span)

    # Extract 1D projection (what slow wave might carry)
    # The x-component is our "slow modulation signal"
    slow_signal = trajectory[:, 0]

    print(f"   - Trajectory shape: {trajectory.shape}")
    print(f"   - Single-channel (slow wave) observation: {slow_signal.shape}")

    # Cerebellar parallel fiber delays (mimicking fiber length variation)
    # Typical granule cell -> Purkinje cell delays: 1-10 ms
    # We work in dimensionless time, so scale appropriately
    print("\n2. Simulating cerebellar parallel fiber architecture...")

    # Embedding parameters
    tau = 8  # Delay in samples (~time constant of slow oscillation)
    m = 3    # Embedding dimension (matches Lorenz attractor dimension)

    # Method 1: Standard Takens delay embedding
    embedded_takens = delay_embed(slow_signal, tau, m)

    # Method 2: Cerebellar parallel fiber simulation
    fiber_delays = np.array([0, tau * dt, 2 * tau * dt]) * 1000  # Convert to "ms-like" units
    fiber_outputs = simulate_cerebellar_processing(slow_signal,
                                                   np.array([0, tau, 2*tau]),
                                                   1.0)  # dt=1 sample

    print(f"   - Delay τ = {tau} samples")
    print(f"   - Embedding dimension m = {m}")
    print(f"   - Parallel fiber delays: {fiber_delays} 'ms'")

    # Trim original trajectory to match embedded length
    original_trimmed = trajectory[(m-1)*tau:, :]

    # Compute manifold preservation
    print("\n3. Evaluating reconstruction quality...")
    preservation = manifold_preservation_score(original_trimmed, embedded_takens)
    print(f"   - Neighborhood preservation: {preservation:.1%}")

    # Estimate correlation dimensions
    dim_orig, _, _ = correlation_dimension_estimate(original_trimmed[::10])
    dim_recon, _, _ = correlation_dimension_estimate(embedded_takens[::10])
    print(f"   - Original correlation dimension: {dim_orig:.2f}")
    print(f"   - Reconstructed correlation dimension: {dim_recon:.2f}")

    # Create visualization
    print("\n4. Generating figures...")

    fig = plt.figure(figsize=(14, 10))

    # Panel A: Original Lorenz attractor
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.plot(trajectory[::5, 0], trajectory[::5, 1], trajectory[::5, 2],
             'b-', alpha=0.6, linewidth=0.5)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('(a) Original Lorenz Attractor\n(Ground truth dynamics)', fontsize=11)

    # Panel B: 1D observation (slow wave)
    ax2 = fig.add_subplot(2, 3, 2)
    t_plot = t_span[:1000]
    ax2.plot(t_plot, slow_signal[:1000], 'b-', linewidth=0.8)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Amplitude')
    ax2.set_title('(b) Single-channel observation\n("slow wave" signal)', fontsize=11)
    ax2.set_xlim(0, t_plot[-1])
    ax2.grid(True, alpha=0.3)

    # Panel C: Parallel fiber schematic
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 6)
    ax3.axis('off')
    ax3.set_title('(c) Cerebellar parallel fiber delays', fontsize=11)

    # Draw schematic
    # Input signal
    ax3.annotate('', xy=(2, 3), xytext=(0.5, 3),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax3.text(0.3, 3, 'x(t)', fontsize=10, va='center')

    # Parallel fibers with different lengths
    fiber_colors = ['#e41a1c', '#377eb8', '#4daf4a']
    for i, (delay, color) in enumerate(zip([0, 1, 2], fiber_colors)):
        y = 4.5 - i * 1.5
        length = 2 + delay * 1.5
        ax3.plot([2, 2 + length], [y, y], color=color, lw=3)
        ax3.plot([2 + length, 8], [y, 3], color=color, lw=1.5, ls='--')
        ax3.text(2 + length/2, y + 0.3, f'τ={delay}', fontsize=9, ha='center')

    # Output (Purkinje cell)
    circle = plt.Circle((8.5, 3), 0.5, color='purple', alpha=0.7)
    ax3.add_patch(circle)
    ax3.text(8.5, 3, 'PC', fontsize=9, ha='center', va='center', color='white')
    ax3.text(8.5, 2, 'Purkinje\ncell', fontsize=8, ha='center', va='top')

    # Label
    ax3.text(5, 0.5, 'Different fiber lengths → delay embedding',
             fontsize=10, ha='center', style='italic')

    # Panel D: Delay-embedded reconstruction
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    ax4.plot(embedded_takens[::5, 0], embedded_takens[::5, 1], embedded_takens[::5, 2],
             'r-', alpha=0.6, linewidth=0.5)
    ax4.set_xlabel('x(t)')
    ax4.set_ylabel(f'x(t-{tau})')
    ax4.set_zlabel(f'x(t-{2*tau})')
    ax4.set_title(f'(d) Takens reconstruction\n(τ={tau}, m={m})', fontsize=11)

    # Panel E: Comparison metric
    ax5 = fig.add_subplot(2, 3, 5)

    # Vary embedding dimension
    m_values = range(1, 8)
    preservation_scores = []
    for m_test in m_values:
        if m_test == 1:
            preservation_scores.append(0)
        else:
            embedded_test = delay_embed(slow_signal, tau, m_test)
            orig_test = trajectory[(m_test-1)*tau:, :]
            score = manifold_preservation_score(orig_test, embedded_test)
            preservation_scores.append(score)

    ax5.plot(m_values, preservation_scores, 'ko-', markersize=8)
    ax5.axvline(x=3, color='red', linestyle='--', alpha=0.7, label='Lorenz dim ≈ 2.06')
    ax5.axhline(y=preservation, color='green', linestyle=':', alpha=0.7)
    ax5.set_xlabel('Embedding dimension m', fontsize=11)
    ax5.set_ylabel('Neighborhood preservation', fontsize=11)
    ax5.set_title('(e) Reconstruction quality vs. m', fontsize=11)
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(0.5, 7.5)
    ax5.set_ylim(0, 1)

    # Panel F: Key point
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    text = """
    KEY INSIGHT

    Takens' theorem: A single scalar observable,
    properly delay-embedded, recovers the full
    attractor topology.

    Cerebellar parallel fibers have graded lengths
    → systematic conduction delays
    → natural implementation of delay embedding

    IMPLICATIONS:

    • Slow cortical oscillations (1-10 Hz) carry
      high-dimensional coordination signals

    • Cerebellum reconstructs full dynamics
      from delay-tapped slow input

    • Loss of slow-wave structure → loss of
      reconstructible coordination manifold
    """

    ax6.text(0.1, 0.95, text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig('../figures/cerebellar_takens.pdf', bbox_inches='tight', dpi=150)
    plt.savefig('../figures/cerebellar_takens.png', bbox_inches='tight', dpi=150)
    print("   Saved: ../figures/cerebellar_takens.pdf and .png")

    # Additional analysis: varying tau
    print("\n5. Analyzing optimal delay selection...")

    fig2, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: reconstruction quality vs tau
    tau_values = range(2, 25, 2)
    preservation_vs_tau = []
    for tau_test in tau_values:
        embedded_test = delay_embed(slow_signal, tau_test, 3)
        orig_test = trajectory[2*tau_test:, :]
        score = manifold_preservation_score(orig_test, embedded_test)
        preservation_vs_tau.append(score)

    axes[0].plot(tau_values, preservation_vs_tau, 'bo-', markersize=6)
    axes[0].set_xlabel('Delay τ (samples)', fontsize=11)
    axes[0].set_ylabel('Neighborhood preservation', fontsize=11)
    axes[0].set_title('(a) Reconstruction quality vs. delay', fontsize=12)
    axes[0].axvline(x=tau, color='red', linestyle='--', alpha=0.7, label=f'Used: τ={tau}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Right: what happens with wrong delays
    axes[1].set_title('(b) Effect of delay mismatch', fontsize=12)

    # Good delay
    embedded_good = delay_embed(slow_signal, 8, 3)
    # Too small delay
    embedded_small = delay_embed(slow_signal, 2, 3)
    # Too large delay
    embedded_large = delay_embed(slow_signal, 25, 3)

    # Plot 2D projections
    axes[1].plot(embedded_small[:500, 0], embedded_small[:500, 1],
                 'g-', alpha=0.5, linewidth=0.5, label=f'τ=2 (too small)')
    axes[1].plot(embedded_good[:500, 0], embedded_good[:500, 1],
                 'b-', alpha=0.7, linewidth=0.8, label=f'τ=8 (optimal)')
    axes[1].plot(embedded_large[:500, 0], embedded_large[:500, 1],
                 'r-', alpha=0.5, linewidth=0.5, label=f'τ=25 (too large)')

    axes[1].set_xlabel('x(t)', fontsize=11)
    axes[1].set_ylabel('x(t-τ)', fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../figures/cerebellar_takens_delay.pdf', bbox_inches='tight', dpi=150)
    plt.savefig('../figures/cerebellar_takens_delay.png', bbox_inches='tight', dpi=150)
    print("   Saved: ../figures/cerebellar_takens_delay.pdf and .png")

    # Summary statistics
    print("\n=== Summary ===")
    print(f"Original system: Lorenz attractor (dim ≈ 2.06)")
    print(f"Observation: 1D projection (x-component)")
    print(f"Reconstruction: Takens embedding with τ={tau}, m=3")
    print(f"Neighborhood preservation: {preservation:.1%}")
    print(f"Correlation dimension (original): {dim_orig:.2f}")
    print(f"Correlation dimension (reconstructed): {dim_recon:.2f}")
    print(f"\nThe parallel fiber architecture naturally implements delay embedding,")
    print(f"allowing reconstruction of high-D dynamics from slow-wave input.")


if __name__ == "__main__":
    main()
