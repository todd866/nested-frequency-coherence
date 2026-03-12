"""
Torus visualization showing clock-synchronized curve vs full parameter space.

For K=3 bands, the parameter space is T^2 (2-torus).
Under clock synchronization, relative phases trace a 1D curve on this torus.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def torus_surface(R=2, r=0.8, n_theta=50, n_phi=50):
    """Generate torus surface coordinates."""
    theta = np.linspace(0, 2*np.pi, n_theta)
    phi = np.linspace(0, 2*np.pi, n_phi)
    theta, phi = np.meshgrid(theta, phi)

    x = (R + r*np.cos(phi)) * np.cos(theta)
    y = (R + r*np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)

    return x, y, z

def clock_curve_on_torus(R=2, r=0.8, n1=2, n2=5, n_points=500):
    """
    Generate clock-synchronized curve on torus.

    Under clock sync with base phase Phi:
    psi_1 = (n_1 - n_3) * Phi
    psi_2 = (n_2 - n_3) * Phi

    For visualization, let n_3 = 8 (gamma), n_1 = 2 (delta), n_2 = 5 (theta).
    Then psi_1 = -6*Phi, psi_2 = -3*Phi.

    The curve winds around the torus with (psi_1, psi_2) = (-6, -3)*Phi.
    """
    n3 = 8  # carrier multiplier
    ratio1 = n1 - n3  # = -6
    ratio2 = n2 - n3  # = -3

    Phi = np.linspace(0, 2*np.pi, n_points)
    psi1 = ratio1 * Phi  # ranges from 0 to -12*pi
    psi2 = ratio2 * Phi  # ranges from 0 to -6*pi

    # Map to torus coordinates
    # psi1 corresponds to theta (around the torus hole)
    # psi2 corresponds to phi (around the tube)
    theta = psi1
    phi = psi2

    x = (R + r*np.cos(phi)) * np.cos(theta)
    y = (R + r*np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)

    return x, y, z

def main():
    fig = plt.figure(figsize=(10, 5))

    # Left panel: Nested coupling (full torus accessible)
    ax1 = fig.add_subplot(121, projection='3d')

    X, Y, Z = torus_surface()
    ax1.plot_surface(X, Y, Z, alpha=0.3, color='steelblue',
                     edgecolor='none', linewidth=0)

    # Add some sample trajectories showing full access
    np.random.seed(42)
    for _ in range(5):
        theta = np.linspace(0, 2*np.pi, 100) + np.random.rand()*2*np.pi
        phi = np.linspace(0, 2*np.pi, 100) * np.random.choice([0.3, 0.5, 0.7, 1.0]) + np.random.rand()*2*np.pi
        R, r = 2, 0.8
        x = (R + r*np.cos(phi)) * np.cos(theta)
        y = (R + r*np.cos(phi)) * np.sin(theta)
        z = r * np.sin(phi)
        ax1.plot(x, y, z, 'b-', alpha=0.6, linewidth=1.5)

    ax1.set_title(r'(a) Nested coupling: $\mathbb{T}^2$ accessible', fontsize=11)
    ax1.set_xlabel(r'$\psi_1$')
    ax1.set_ylabel(r'$\psi_2$')
    ax1.set_zlabel('')
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-3, 3)
    ax1.set_zlim(-1.5, 1.5)
    ax1.view_init(elev=20, azim=45)
    ax1.set_box_aspect([1,1,0.5])

    # Right panel: Clock synchronization (1D curve)
    ax2 = fig.add_subplot(122, projection='3d')

    X, Y, Z = torus_surface()
    ax2.plot_surface(X, Y, Z, alpha=0.15, color='gray',
                     edgecolor='none', linewidth=0)

    # Clock curve
    x, y, z = clock_curve_on_torus()
    ax2.plot(x, y, z, 'r-', linewidth=2.5, label='Clock trajectory')

    ax2.set_title(r'(b) Clock sync: 1D submanifold', fontsize=11)
    ax2.set_xlabel(r'$\psi_1$')
    ax2.set_ylabel(r'$\psi_2$')
    ax2.set_zlabel('')
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-3, 3)
    ax2.set_zlim(-1.5, 1.5)
    ax2.view_init(elev=20, azim=45)
    ax2.set_box_aspect([1,1,0.5])

    plt.tight_layout()
    plt.savefig('../figures/torus_clock.pdf', bbox_inches='tight', dpi=150)
    plt.savefig('../figures/torus_clock.png', bbox_inches='tight', dpi=150)
    print("Saved figures to ../figures/torus_clock.pdf and .png")

if __name__ == "__main__":
    main()
