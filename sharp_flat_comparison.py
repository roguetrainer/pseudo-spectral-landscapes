"""
Sharp vs Flat Minima: Generalization Visualization
=================================================
Demonstrates why flatter minima generalize better through
geometric and spectral analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')


def create_sharp_vs_flat_visualization():
    """
    Create comprehensive visualization showing sharp vs flat minima
    """
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # ============================================================================
    # ROW 1: 1D Comparison
    # ============================================================================
    
    # Create parameter range
    theta = np.linspace(-2, 2, 200)
    
    # Flat minimum: small second derivative
    flat_minimum = 0.1 + 0.5 * theta**2
    flat_lambda = 1.0  # Small eigenvalue
    
    # Sharp minimum: large second derivative
    sharp_minimum = 0.1 + 5 * theta**2
    sharp_lambda = 10.0  # Large eigenvalue
    
    # 1. Flat Minimum
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(theta, flat_minimum, 'b-', linewidth=3)
    ax1.scatter([0], [0.1], color='gold', s=300, marker='*', 
               edgecolors='black', linewidths=2, zorder=5, label='Minimum')
    
    # Show perturbation region
    epsilon = 0.5
    ax1.axvspan(-epsilon, epsilon, alpha=0.2, color='green', 
               label=f'±{epsilon} perturbation')
    
    # Mark loss increase
    loss_at_perturb_flat = 0.1 + 0.5 * epsilon**2
    ax1.plot([epsilon, epsilon], [0.1, loss_at_perturb_flat], 
            'r--', linewidth=2, alpha=0.7)
    ax1.text(epsilon + 0.1, (0.1 + loss_at_perturb_flat)/2, 
            f'ΔL={loss_at_perturb_flat-0.1:.2f}',
            fontsize=10, color='red', fontweight='bold')
    
    ax1.set_xlabel('Parameter θ', fontsize=11)
    ax1.set_ylabel('Loss L(θ)', fontsize=11)
    ax1.set_title(f'Flat Minimum\nλ_max = {flat_lambda:.1f}', 
                 fontsize=13, fontweight='bold', color='blue')
    ax1.set_ylim([0, 2])
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 2. Sharp Minimum
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(theta, sharp_minimum, 'r-', linewidth=3)
    ax2.scatter([0], [0.1], color='gold', s=300, marker='*', 
               edgecolors='black', linewidths=2, zorder=5, label='Minimum')
    
    # Show perturbation region
    ax2.axvspan(-epsilon, epsilon, alpha=0.2, color='orange', 
               label=f'±{epsilon} perturbation')
    
    # Mark loss increase
    loss_at_perturb_sharp = 0.1 + 5 * epsilon**2
    ax2.plot([epsilon, epsilon], [0.1, loss_at_perturb_sharp], 
            'r--', linewidth=2, alpha=0.7)
    ax2.text(epsilon + 0.1, (0.1 + loss_at_perturb_sharp)/2, 
            f'ΔL={loss_at_perturb_sharp-0.1:.2f}',
            fontsize=10, color='red', fontweight='bold')
    
    ax2.set_xlabel('Parameter θ', fontsize=11)
    ax2.set_ylabel('Loss L(θ)', fontsize=11)
    ax2.set_title(f'Sharp Minimum\nλ_max = {sharp_lambda:.1f}', 
                 fontsize=13, fontweight='bold', color='red')
    ax2.set_ylim([0, 2])
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 3. Comparison Summary
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    
    comparison_text = f"""
    ROBUSTNESS COMPARISON
    
    Same perturbation ε = {epsilon}
    
    Flat Minimum:
      • ΔL = {loss_at_perturb_flat-0.1:.3f}
      • Small sensitivity
      • ✓ Robust to noise
      • ✓ Better generalization
    
    Sharp Minimum:  
      • ΔL = {loss_at_perturb_sharp-0.1:.3f}
      • High sensitivity
      • ✗ Sensitive to noise
      • ✗ Poor generalization
    
    Ratio: {(loss_at_perturb_sharp-0.1)/(loss_at_perturb_flat-0.1):.1f}× more sensitive!
    """
    
    ax3.text(0.05, 0.95, comparison_text, ha='left', va='top', 
            fontsize=10, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', 
                     alpha=0.8, edgecolor='black', linewidth=2))
    
    # ============================================================================
    # ROW 2: 2D Landscape Comparison
    # ============================================================================
    
    # Create 2D grid
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    
    # 4. Flat 2D Landscape
    ax4 = fig.add_subplot(gs[1, 0])
    Z_flat = 0.1 + 0.3 * X**2 + 0.3 * Y**2
    
    levels = np.linspace(0.1, 2, 15)
    contour = ax4.contour(X, Y, Z_flat, levels=levels, cmap='Blues', linewidths=2)
    ax4.contourf(X, Y, Z_flat, levels=levels, cmap='Blues', alpha=0.4)
    ax4.scatter([0], [0], color='gold', s=300, marker='*', 
               edgecolors='black', linewidths=2, zorder=5)
    
    # Draw perturbation circle
    circle_theta = np.linspace(0, 2*np.pi, 100)
    circle_r = 0.5
    ax4.plot(circle_r * np.cos(circle_theta), circle_r * np.sin(circle_theta),
            'g--', linewidth=2, label='Test perturbations')
    
    ax4.set_xlabel('Parameter θ₁', fontsize=11)
    ax4.set_ylabel('Parameter θ₂', fontsize=11)
    ax4.set_title('Flat Loss Basin\n(Wide valley)', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=9)
    ax4.set_aspect('equal')
    ax4.grid(True, alpha=0.3)
    
    # 5. Sharp 2D Landscape
    ax5 = fig.add_subplot(gs[1, 1])
    Z_sharp = 0.1 + 3 * X**2 + 3 * Y**2
    
    contour = ax5.contour(X, Y, Z_sharp, levels=levels, cmap='Reds', linewidths=2)
    ax5.contourf(X, Y, Z_sharp, levels=levels, cmap='Reds', alpha=0.4)
    ax5.scatter([0], [0], color='gold', s=300, marker='*', 
               edgecolors='black', linewidths=2, zorder=5)
    
    # Draw perturbation circle
    ax5.plot(circle_r * np.cos(circle_theta), circle_r * np.sin(circle_theta),
            'orange', linestyle='--', linewidth=2, label='Test perturbations')
    
    ax5.set_xlabel('Parameter θ₁', fontsize=11)
    ax5.set_ylabel('Parameter θ₂', fontsize=11)
    ax5.set_title('Sharp Loss Basin\n(Narrow valley)', fontsize=12, fontweight='bold')
    ax5.legend(loc='upper right', fontsize=9)
    ax5.set_aspect('equal')
    ax5.grid(True, alpha=0.3)
    
    # 6. Eigenvalue Comparison
    ax6 = fig.add_subplot(gs[1, 2])
    
    # Eigenvalue spectra for both cases
    eigenvalues_flat = [0.6, 0.6, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02]
    eigenvalues_sharp = [6.0, 6.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.2]
    
    x_pos = np.arange(len(eigenvalues_flat))
    width = 0.35
    
    ax6.bar(x_pos - width/2, eigenvalues_flat, width, 
           label='Flat', color='blue', alpha=0.7, edgecolor='black')
    ax6.bar(x_pos + width/2, eigenvalues_sharp, width, 
           label='Sharp', color='red', alpha=0.7, edgecolor='black')
    
    ax6.set_xlabel('Eigenvalue Index', fontsize=11)
    ax6.set_ylabel('Eigenvalue Magnitude', fontsize=11)
    ax6.set_title('Hessian Eigenvalue Spectra', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=10)
    ax6.set_yscale('log')
    ax6.grid(True, alpha=0.3, which='both')
    
    # ============================================================================
    # ROW 3: Training and Generalization
    # ============================================================================
    
    # 7. Training Curves
    ax7 = fig.add_subplot(gs[2, 0])
    
    epochs = np.linspace(0, 100, 50)
    
    # Both reach similar training loss
    train_loss_flat = 0.5 * np.exp(-epochs/15) + 0.1
    train_loss_sharp = 0.5 * np.exp(-epochs/15) + 0.1
    
    # But test loss diverges
    test_loss_flat = train_loss_flat + 0.05
    test_loss_sharp = train_loss_flat + 0.2 + 0.1 * (epochs / 100)
    
    ax7.plot(epochs, train_loss_flat, 'b-', linewidth=2.5, 
            label='Flat: Train', alpha=0.7)
    ax7.plot(epochs, test_loss_flat, 'b--', linewidth=2.5, 
            label='Flat: Test')
    ax7.plot(epochs, train_loss_sharp, 'r-', linewidth=2.5, 
            label='Sharp: Train', alpha=0.7)
    ax7.plot(epochs, test_loss_sharp, 'r--', linewidth=2.5, 
            label='Sharp: Test')
    
    # Highlight generalization gap
    gap_epoch = 40
    gap_idx = int(gap_epoch / 100 * len(epochs))
    ax7.plot([gap_epoch, gap_epoch], 
            [train_loss_sharp[gap_idx], test_loss_sharp[gap_idx]], 
            'k-', linewidth=3, alpha=0.5, label='Generalization gap')
    
    ax7.set_xlabel('Training Epoch', fontsize=11)
    ax7.set_ylabel('Loss', fontsize=11)
    ax7.set_title('Training vs Test Loss', fontsize=12, fontweight='bold')
    ax7.legend(loc='upper right', fontsize=9)
    ax7.grid(True, alpha=0.3)
    ax7.set_ylim([0, 0.5])
    
    # 8. Generalization Gap Analysis
    ax8 = fig.add_subplot(gs[2, 1])
    
    # Show generalization gap over time
    gen_gap_flat = test_loss_flat - train_loss_flat
    gen_gap_sharp = test_loss_sharp - train_loss_sharp
    
    ax8.fill_between(epochs, 0, gen_gap_flat, 
                     color='blue', alpha=0.4, label='Flat minimum')
    ax8.fill_between(epochs, 0, gen_gap_sharp, 
                     color='red', alpha=0.4, label='Sharp minimum')
    
    ax8.plot(epochs, gen_gap_flat, 'b-', linewidth=2.5)
    ax8.plot(epochs, gen_gap_sharp, 'r-', linewidth=2.5)
    
    ax8.set_xlabel('Training Epoch', fontsize=11)
    ax8.set_ylabel('Generalization Gap\n(Test Loss - Train Loss)', fontsize=11)
    ax8.set_title('Generalization Performance', fontsize=12, fontweight='bold')
    ax8.legend(loc='upper left', fontsize=10)
    ax8.grid(True, alpha=0.3)
    
    # 9. Summary Table
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    summary_text = """
    KEY INSIGHTS
    
    Why Flat Minima Generalize:
    
    1. Robustness
       Small λ → Small ΔL
       Insensitive to parameter noise
    
    2. Basin Width
       Wide valley in parameter space
       Many nearby good solutions
    
    3. Test Perturbations
       Real data ≈ perturbed training
       Flat → consistent performance
    
    4. Optimization
       Implicit regularization
       SGD noise escapes sharp minima
    
    Practical Implications:
       • Use large batch sizes
       • Apply SAM optimizer
       • Monitor λ_max during training
       • Prefer flatter architectures
    """
    
    ax9.text(0.05, 0.95, summary_text, ha='left', va='top', 
            fontsize=9, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', 
                     alpha=0.7, edgecolor='black', linewidth=2))
    
    # Overall title
    fig.suptitle('Sharp vs Flat Minima: Why Geometry Matters for Generalization\n' +
                'Pseudo-Spectral Perspective on Neural Network Optimization',
                fontsize=15, fontweight='bold', y=0.995)
    
    return fig


def main():
    """Create and save the sharp vs flat comparison"""
    print("Creating sharp vs flat minima visualization...")
    
    fig = create_sharp_vs_flat_visualization()
    
    plt.savefig('/mnt/user-data/outputs/sharp_vs_flat_minima.png', 
               dpi=150, bbox_inches='tight', facecolor='white')
    print("✓ Saved: sharp_vs_flat_minima.png")
    
    print("\nThis visualization demonstrates:")
    print("  • 1D comparison of sharp vs flat curvature")
    print("  • 2D loss landscape basins")
    print("  • Eigenvalue spectrum differences")
    print("  • Training vs test loss behavior")
    print("  • Generalization gap analysis")
    print("\nKey takeaway: Flat minima (small λ_max) → Better generalization!")
    
    plt.show()


if __name__ == "__main__":
    main()
