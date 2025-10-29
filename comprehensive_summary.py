"""
Comprehensive Pseudo-Spectral Summary Visualization
===================================================
A single figure that ties together all the key concepts.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.linalg import eigh
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def create_comprehensive_summary():
    """
    Create a single comprehensive figure explaining pseudo-spectral landscapes
    """
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(4, 4, figure=fig, hspace=0.35, wspace=0.3)
    
    # Color scheme
    colors = plt.cm.viridis(np.linspace(0, 1, 5))
    
    # ============================================================================
    # ROW 1: CONCEPTUAL FOUNDATION
    # ============================================================================
    
    # 1. Loss Landscape Intuition
    ax1 = fig.add_subplot(gs[0, :2], projection='3d')
    
    # Create a simple loss surface
    x = np.linspace(-3, 3, 50)
    y = np.linspace(-3, 3, 50)
    X, Y = np.meshgrid(x, y)
    
    # Quadratic bowl with some complexity
    Z = 0.5 * X**2 + 0.2 * Y**2 + 0.1 * np.sin(3*X) * np.sin(3*Y) + 0.3
    
    surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, 
                            linewidth=0, antialiased=True)
    
    # Mark a point
    ax1.scatter([0], [0], [0.3], color='red', s=200, marker='*', 
               edgecolors='black', linewidths=2, zorder=5)
    
    ax1.set_xlabel('Parameter θ₁', fontsize=10)
    ax1.set_ylabel('Parameter θ₂', fontsize=10)
    ax1.set_zlabel('Loss L(θ)', fontsize=10)
    ax1.set_title('Loss Landscape\n(High-dimensional reality)', 
                 fontsize=12, fontweight='bold', pad=10)
    ax1.view_init(elev=25, azim=45)
    
    # 2. Hessian Concept
    ax2 = fig.add_subplot(gs[0, 2:])
    ax2.axis('off')
    
    # Draw concept diagram
    ax2.text(0.5, 0.95, 'The Hessian Matrix H', 
            ha='center', va='top', fontsize=14, fontweight='bold')
    
    ax2.text(0.5, 0.85, 'H[i,j] = ∂²L/∂θᵢ∂θⱼ', 
            ha='center', va='top', fontsize=12, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # Arrow
    ax2.annotate('', xy=(0.5, 0.7), xytext=(0.5, 0.78),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    ax2.text(0.5, 0.65, 'Eigenvalue Decomposition', 
            ha='center', va='top', fontsize=12, fontweight='bold')
    
    ax2.text(0.5, 0.58, 'H = QΛQᵀ', 
            ha='center', va='top', fontsize=11, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    # Key insight boxes
    box_y = 0.35
    box_width = 0.25
    
    insights = [
        ('λ > 0\nMinimum', 'lightgreen'),
        ('λ < 0\nSaddle', 'lightcoral'),
        ('|λ| ≈ 0\nFlat', 'lightyellow')
    ]
    
    for i, (text, color) in enumerate(insights):
        x_pos = 0.15 + i * 0.35
        ax2.text(x_pos, box_y, text, ha='center', va='center',
                fontsize=10, bbox=dict(boxstyle='round', 
                facecolor=color, alpha=0.8, edgecolor='black', linewidth=2))
    
    ax2.text(0.5, 0.05, 'Eigenvalues reveal local geometry!', 
            ha='center', va='bottom', fontsize=11, style='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # ============================================================================
    # ROW 2: EIGENVALUE ANALYSIS
    # ============================================================================
    
    # 3. Example Eigenvalue Spectrum
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Simulate a typical spectrum
    n_params = 20
    eigenvalues = np.concatenate([
        np.array([5.0, 3.0]),  # Few large
        np.random.exponential(0.5, 8),  # Medium decay
        np.random.normal(0, 0.2, 7),  # Near zero
        np.array([-0.5, -0.3, -0.1])  # Few negative
    ])
    eigenvalues = np.sort(eigenvalues)[::-1]
    
    bars = ax3.bar(range(n_params), eigenvalues, 
                   color=[colors[0] if e > 1 else colors[2] if e > 0 else colors[4] 
                          for e in eigenvalues], alpha=0.8, edgecolor='black')
    ax3.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax3.set_xlabel('Eigenvalue Index', fontsize=10)
    ax3.set_ylabel('Eigenvalue λ', fontsize=10)
    ax3.set_title('Typical Eigenvalue Spectrum', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Annotations
    ax3.text(1, 4.5, 'Sharp\ndirections', ha='center', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    ax3.text(10, 0.3, 'Flat\ndirections', ha='center', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    ax3.text(18, -0.4, 'Saddle\ndirections', ha='center', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    
    # 4. Interpretation Guide
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Create interpretation table
    categories = ['Sharp (λ>1)', 'Medium (0<λ<1)', 'Flat (|λ|≈0)', 'Saddle (λ<0)']
    counts = [2, 8, 7, 3]
    colors_cat = [colors[0], colors[1], colors[2], colors[4]]
    
    bars = ax4.barh(categories, counts, color=colors_cat, alpha=0.8, edgecolor='black')
    ax4.set_xlabel('Count', fontsize=10)
    ax4.set_title('Eigenvalue Distribution', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')
    
    # Add percentages
    for i, (cat, count) in enumerate(zip(categories, counts)):
        pct = 100 * count / sum(counts)
        ax4.text(count + 0.3, i, f'{pct:.0f}%', va='center', fontsize=9)
    
    # 5. Key Metrics
    ax5 = fig.add_subplot(gs[1, 2:])
    ax5.axis('off')
    
    # Calculate metrics
    lambda_max = np.max(eigenvalues)
    lambda_min = np.min(eigenvalues[eigenvalues > 0]) if any(eigenvalues > 0) else 1e-6
    condition_num = lambda_max / lambda_min
    trace = np.sum(eigenvalues)
    flatness = 1 / (1 + lambda_max)
    n_negative = np.sum(eigenvalues < 0)
    
    ax5.text(0.5, 0.95, 'Key Metrics', ha='center', fontsize=13, 
            fontweight='bold', va='top')
    
    metrics_text = f"""
    λ_max (sharpest): {lambda_max:.2f}
    λ_min (flattest): {lambda_min:.2f}
    
    Condition κ: {condition_num:.1f}
    Trace Tr(H): {trace:.2f}
    
    Flatness: {flatness:.3f}
    Negative λ: {n_negative}
    
    Geometry: {"Saddle Point" if n_negative > 0 else "Local Minimum"}
    """
    
    ax5.text(0.1, 0.80, metrics_text, ha='left', va='top', 
            fontsize=10, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', 
                     alpha=0.6, edgecolor='black', linewidth=1.5))
    
    # Interpretation
    interpretation = ""
    if condition_num < 10:
        interpretation += "✓ Well-conditioned\n"
    elif condition_num < 100:
        interpretation += "⚠ Moderate conditioning\n"
    else:
        interpretation += "✗ Ill-conditioned\n"
    
    if flatness > 0.3:
        interpretation += "✓ Good flatness\n"
    else:
        interpretation += "✗ Sharp minimum\n"
    
    if n_negative > 0:
        interpretation += f"⚠ {n_negative} escape directions\n"
    
    ax5.text(0.6, 0.50, interpretation, ha='left', va='top', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # ============================================================================
    # ROW 3: LLM CONNECTIONS
    # ============================================================================
    
    # 6. Scale Challenge
    ax6 = fig.add_subplot(gs[2, 0])
    
    model_sizes = ['Toy\nDemo', 'Small\nNN', 'BERT', 'GPT-3', 'GPT-4']
    param_counts = [13, 1e4, 1.1e8, 1.75e11, 1.8e12]
    
    bars = ax6.bar(range(len(model_sizes)), np.log10(param_counts), 
                  color=plt.cm.plasma(np.linspace(0.2, 0.9, len(model_sizes))),
                  alpha=0.8, edgecolor='black')
    ax6.set_xticks(range(len(model_sizes)))
    ax6.set_xticklabels(model_sizes, fontsize=9)
    ax6.set_ylabel('log₁₀(# Parameters)', fontsize=10)
    ax6.set_title('Model Scale Challenge', fontsize=11, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Add actual numbers on bars
    for i, (size, count) in enumerate(zip(model_sizes, param_counts)):
        if count < 1e6:
            label = f'{int(count)}'
        elif count < 1e9:
            label = f'{count/1e6:.0f}M'
        elif count < 1e12:
            label = f'{count/1e9:.0f}B'
        else:
            label = f'{count/1e12:.1f}T'
        ax6.text(i, np.log10(count) + 0.3, label, ha='center', fontsize=8)
    
    # 7. Low-Rank Structure (LoRA Connection)
    ax7 = fig.add_subplot(gs[2, 1])
    
    # Simulate singular value decay (typical for attention matrices)
    n_dims = 16
    true_rank = 4
    singular_values = np.concatenate([
        np.linspace(1.0, 0.3, true_rank),
        np.random.exponential(0.05, n_dims - true_rank)
    ])
    
    ax7.semilogy(range(n_dims), singular_values, 'o-', 
                linewidth=2, markersize=8, color=colors[3])
    ax7.axvline(x=true_rank-0.5, color='red', linestyle='--', 
               linewidth=2, alpha=0.7, label=f'Effective rank ≈ {true_rank}')
    ax7.set_xlabel('Singular Value Index', fontsize=10)
    ax7.set_ylabel('Magnitude (log)', fontsize=10)
    ax7.set_title('Low-Rank Structure\n(Enables LoRA!)', fontsize=11, fontweight='bold')
    ax7.legend(fontsize=9)
    ax7.grid(True, alpha=0.3, which='both')
    
    # 8. Flat vs Sharp Generalization
    ax8 = fig.add_subplot(gs[2, 2:])
    
    # Create illustration
    theta = np.linspace(-2, 2, 100)
    
    # Flat minimum
    flat_loss = 0.1 + 0.5 * theta**2
    # Sharp minimum  
    sharp_loss = 0.1 + 5 * theta**2
    
    ax8.plot(theta, flat_loss, 'b-', linewidth=3, label='Flat (λ_max small)')
    ax8.plot(theta, sharp_loss, 'r-', linewidth=3, label='Sharp (λ_max large)')
    
    # Mark perturbation region
    perturbation = 0.5
    ax8.axvspan(-perturbation, perturbation, alpha=0.2, color='green')
    ax8.text(0, 3, 'Typical test\nperturbations', ha='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    ax8.scatter([0], [0.1], color='gold', s=200, marker='*', 
               edgecolors='black', linewidths=2, zorder=5, label='Minimum')
    
    ax8.set_xlabel('Parameter Perturbation', fontsize=10)
    ax8.set_ylabel('Loss', fontsize=10)
    ax8.set_title('Flat Minima Generalize Better', fontsize=11, fontweight='bold')
    ax8.legend(fontsize=9, loc='upper right')
    ax8.set_ylim([0, 5])
    ax8.grid(True, alpha=0.3)
    
    # ============================================================================
    # ROW 4: PRACTICAL APPLICATIONS
    # ============================================================================
    
    # 9. Training Dynamics
    ax9 = fig.add_subplot(gs[3, :2])
    
    # Simulate training curves
    epochs = np.linspace(0, 100, 50)
    
    # Loss
    loss_good = 2 * np.exp(-epochs/20) + 0.1
    loss_bad = 2 * np.exp(-epochs/40) + 0.5
    
    # Max eigenvalue
    lambda_max_good = 10 * np.exp(-epochs/25) + 2
    lambda_max_bad = 10 * np.exp(-epochs/50) + 8
    
    ax9_2 = ax9.twinx()
    
    l1 = ax9.plot(epochs, loss_good, 'b-', linewidth=2.5, 
                 label='Good: Loss')
    l2 = ax9.plot(epochs, loss_bad, 'b--', linewidth=2.5, alpha=0.6,
                 label='Bad: Loss')
    
    l3 = ax9_2.plot(epochs, lambda_max_good, 'r-', linewidth=2.5,
                   label='Good: λ_max')
    l4 = ax9_2.plot(epochs, lambda_max_bad, 'r--', linewidth=2.5, alpha=0.6,
                   label='Bad: λ_max')
    
    ax9.set_xlabel('Training Epoch', fontsize=10)
    ax9.set_ylabel('Loss', fontsize=10, color='blue')
    ax9_2.set_ylabel('Max Eigenvalue λ_max', fontsize=10, color='red')
    ax9.set_title('Good vs Bad Training Dynamics', fontsize=11, fontweight='bold')
    ax9.tick_params(axis='y', labelcolor='blue')
    ax9_2.tick_params(axis='y', labelcolor='red')
    
    # Combined legend
    lns = l1 + l2 + l3 + l4
    labs = [l.get_label() for l in lns]
    ax9.legend(lns, labs, loc='right', fontsize=8)
    ax9.grid(True, alpha=0.3)
    
    # 10. Application Summary
    ax10 = fig.add_subplot(gs[3, 2:])
    ax10.axis('off')
    
    ax10.text(0.5, 0.95, 'Practical Applications', ha='center', 
             fontsize=13, fontweight='bold', va='top')
    
    applications = """
    🎯 Optimizer Design
       • Learning rate ∝ 1/λ_max
       • Adam adapts to local curvature
       • SAM seeks flat minima
    
    🏗️ Architecture Choices  
       • Skip connections → lower λ_max
       • Layer norm → better conditioning
       • Width affects spectral properties
    
    🔧 Fine-Tuning Strategies
       • LoRA exploits low-rank structure
       • Target high-curvature parameters
       • Preserve flat directions
    
    🐛 Debugging Training
       • Loss spikes → high curvature
       • Slow convergence → ill-conditioned
       • Check eigenvalue spectrum!
    """
    
    ax10.text(0.05, 0.80, applications, ha='left', va='top', 
             fontsize=9, family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', 
                      alpha=0.8, edgecolor='black', linewidth=1.5))
    
    # Overall title
    fig.suptitle('Pseudo-Spectral Landscapes: Complete Guide\n' + 
                'Understanding Neural Network Optimization Through Geometry',
                fontsize=16, fontweight='bold', y=0.995)
    
    return fig


def main():
    """Create and save the comprehensive summary"""
    print("Creating comprehensive pseudo-spectral summary...")
    
    fig = create_comprehensive_summary()
    
    plt.savefig('/mnt/user-data/outputs/pseudo_spectral_analysis.png', 
               dpi=150, bbox_inches='tight', facecolor='white')
    print("✓ Saved: pseudo_spectral_analysis.png")
    
    print("\nThis single visualization ties together all key concepts:")
    print("  • Loss landscape geometry")
    print("  • Hessian eigenvalue analysis")
    print("  • Connections to LLMs")
    print("  • Practical applications")
    print("\nUse this as a quick reference for understanding optimization geometry!")
    
    plt.show()


if __name__ == "__main__":
    main()
