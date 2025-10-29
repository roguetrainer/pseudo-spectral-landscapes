"""
Pseudo-Spectral Analysis for Transformer Attention Mechanisms
==============================================================
This demonstrates how geometric and spectral concepts apply specifically to
the attention mechanisms that power modern LLMs.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy.linalg import eigh, svd
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')


class SimpleAttentionLayer:
    """
    Simplified self-attention mechanism to demonstrate geometric properties.
    Similar to what's in transformers/LLMs but simplified for analysis.
    """
    
    def __init__(self, d_model=8, n_heads=2, seq_len=4, seed=42):
        np.random.seed(seed)
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.seq_len = seq_len
        
        # Query, Key, Value projection matrices (simplified)
        self.W_Q = np.random.randn(d_model, d_model) * 0.1
        self.W_K = np.random.randn(d_model, d_model) * 0.1
        self.W_V = np.random.randn(d_model, d_model) * 0.1
        self.W_O = np.random.randn(d_model, d_model) * 0.1
        
    def attention(self, X):
        """
        Scaled dot-product attention
        X: (seq_len, d_model)
        """
        Q = X @ self.W_Q
        K = X @ self.W_K
        V = X @ self.W_V
        
        # Attention scores
        scores = Q @ K.T / np.sqrt(self.d_k)
        
        # Softmax
        attention_weights = self.softmax(scores)
        
        # Apply attention to values
        attended = attention_weights @ V
        
        # Output projection
        output = attended @ self.W_O
        
        return output, attention_weights, Q, K, V
    
    def softmax(self, x):
        """Numerically stable softmax"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def get_params_vector(self):
        """Flatten all parameters"""
        return np.concatenate([
            self.W_Q.flatten(),
            self.W_K.flatten(),
            self.W_V.flatten(),
            self.W_O.flatten()
        ])
    
    def set_params_vector(self, params):
        """Set parameters from flattened vector"""
        d = self.d_model
        idx = 0
        
        self.W_Q = params[idx:idx + d*d].reshape(d, d)
        idx += d*d
        
        self.W_K = params[idx:idx + d*d].reshape(d, d)
        idx += d*d
        
        self.W_V = params[idx:idx + d*d].reshape(d, d)
        idx += d*d
        
        self.W_O = params[idx:idx + d*d].reshape(d, d)
    
    def loss(self, X, target):
        """Simple MSE loss for demonstration"""
        output, _, _, _, _ = self.attention(X)
        return 0.5 * np.mean((output - target) ** 2)
    
    def compute_gradient_numerical(self, X, target, eps=1e-5):
        """Compute gradient numerically"""
        params = self.get_params_vector()
        grad = np.zeros_like(params)
        
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += eps
            self.set_params_vector(params_plus)
            loss_plus = self.loss(X, target)
            
            params_minus = params.copy()
            params_minus[i] -= eps
            self.set_params_vector(params_minus)
            loss_minus = self.loss(X, target)
            
            grad[i] = (loss_plus - loss_minus) / (2 * eps)
        
        self.set_params_vector(params)
        return grad


def analyze_attention_geometry():
    """
    Comprehensive geometric analysis of attention mechanism
    """
    # Create attention layer and synthetic data
    layer = SimpleAttentionLayer(d_model=8, n_heads=2, seq_len=4)
    
    # Input sequence (e.g., token embeddings)
    X = np.random.randn(4, 8) * 0.5  # (seq_len, d_model)
    target = np.random.randn(4, 8) * 0.5
    
    # Get attention outputs
    output, attn_weights, Q, K, V = layer.attention(X)
    
    # Compute QK^T matrix (before softmax)
    QKT = Q @ K.T / np.sqrt(layer.d_k)
    
    # Analyze spectral properties of different matrices
    results = {
        'X': X,
        'output': output,
        'attention_weights': attn_weights,
        'QKT': QKT,
        'Q': Q,
        'K': K,
        'V': V
    }
    
    # SVD of weight matrices
    U_Q, S_Q, Vt_Q = svd(layer.W_Q)
    U_K, S_K, Vt_K = svd(layer.W_K)
    U_V, S_V, Vt_V = svd(layer.W_V)
    U_O, S_O, Vt_O = svd(layer.W_O)
    
    results['svd'] = {
        'W_Q': (U_Q, S_Q, Vt_Q),
        'W_K': (U_K, S_K, Vt_K),
        'W_V': (U_V, S_V, Vt_V),
        'W_O': (U_O, S_O, Vt_O)
    }
    
    # Eigenanalysis of attention weights matrix
    try:
        eigvals_attn, eigvecs_attn = eigh(attn_weights)
        results['attention_spectrum'] = (eigvals_attn, eigvecs_attn)
    except:
        results['attention_spectrum'] = None
    
    # Compute approximate Hessian using numerical gradients
    print("Computing Hessian approximation (this may take a moment)...")
    params = layer.get_params_vector()
    n_params = len(params)
    
    # For speed, compute only diagonal and near-diagonal of Hessian
    hessian_diag = np.zeros(n_params)
    eps = 1e-4
    
    for i in range(n_params):
        params_pp = params.copy()
        params_pp[i] += eps
        layer.set_params_vector(params_pp)
        loss_plus = layer.loss(X, target)
        
        params_mm = params.copy()
        params_mm[i] -= eps
        layer.set_params_vector(params_mm)
        loss_minus = layer.loss(X, target)
        
        layer.set_params_vector(params)
        loss_center = layer.loss(X, target)
        
        # Second derivative approximation
        hessian_diag[i] = (loss_plus - 2*loss_center + loss_minus) / (eps**2)
    
    results['hessian_diag'] = hessian_diag
    
    layer.set_params_vector(params)
    
    return layer, results


def visualize_attention_geometry(layer, results):
    """
    Create comprehensive visualization of attention mechanism geometry
    """
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Attention Weights Heatmap
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(results['attention_weights'], cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    ax1.set_xlabel('Key Position')
    ax1.set_ylabel('Query Position')
    ax1.set_title('Attention Weight Matrix\n(After Softmax)')
    plt.colorbar(im1, ax=ax1)
    
    # Add values on heatmap
    for i in range(results['attention_weights'].shape[0]):
        for j in range(results['attention_weights'].shape[1]):
            text = ax1.text(j, i, f'{results["attention_weights"][i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    # 2. QK^T Score Matrix (before softmax)
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(results['QKT'], cmap='RdBu_r', aspect='auto',
                     vmin=-np.abs(results['QKT']).max(), 
                     vmax=np.abs(results['QKT']).max())
    ax2.set_xlabel('Key Position')
    ax2.set_ylabel('Query Position')
    ax2.set_title('QK^T Score Matrix\n(Before Softmax)')
    plt.colorbar(im2, ax=ax2)
    
    # 3. Singular Value Spectra
    ax3 = fig.add_subplot(gs[0, 2])
    svd_data = results['svd']
    x_pos = np.arange(len(svd_data['W_Q'][1]))
    width = 0.2
    
    ax3.bar(x_pos - 1.5*width, svd_data['W_Q'][1], width, label='W_Q', alpha=0.8)
    ax3.bar(x_pos - 0.5*width, svd_data['W_K'][1], width, label='W_K', alpha=0.8)
    ax3.bar(x_pos + 0.5*width, svd_data['W_V'][1], width, label='W_V', alpha=0.8)
    ax3.bar(x_pos + 1.5*width, svd_data['W_O'][1], width, label='W_O', alpha=0.8)
    
    ax3.set_xlabel('Singular Value Index')
    ax3.set_ylabel('Singular Value Magnitude')
    ax3.set_title('Singular Value Spectra of Weight Matrices')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Attention Eigenspectrum
    ax4 = fig.add_subplot(gs[0, 3])
    if results['attention_spectrum'] is not None:
        eigvals = results['attention_spectrum'][0]
        ax4.bar(range(len(eigvals)), eigvals, color='purple', alpha=0.7)
        ax4.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax4.set_xlabel('Eigenvalue Index')
        ax4.set_ylabel('Eigenvalue')
        ax4.set_title('Attention Matrix Eigenspectrum')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Eigenanalysis\nNot Available', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Attention Matrix Eigenspectrum')
    
    # 5. Query Matrix Q
    ax5 = fig.add_subplot(gs[1, 0])
    im5 = ax5.imshow(results['Q'], cmap='RdBu_r', aspect='auto',
                     vmin=-np.abs(results['Q']).max(),
                     vmax=np.abs(results['Q']).max())
    ax5.set_xlabel('Dimension')
    ax5.set_ylabel('Sequence Position')
    ax5.set_title('Query Matrix Q\n(Q = X @ W_Q)')
    plt.colorbar(im5, ax=ax5)
    
    # 6. Key Matrix K
    ax6 = fig.add_subplot(gs[1, 1])
    im6 = ax6.imshow(results['K'], cmap='RdBu_r', aspect='auto',
                     vmin=-np.abs(results['K']).max(),
                     vmax=np.abs(results['K']).max())
    ax6.set_xlabel('Dimension')
    ax6.set_ylabel('Sequence Position')
    ax6.set_title('Key Matrix K\n(K = X @ W_K)')
    plt.colorbar(im6, ax=ax6)
    
    # 7. Value Matrix V
    ax7 = fig.add_subplot(gs[1, 2])
    im7 = ax7.imshow(results['V'], cmap='RdBu_r', aspect='auto',
                     vmin=-np.abs(results['V']).max(),
                     vmax=np.abs(results['V']).max())
    ax7.set_xlabel('Dimension')
    ax7.set_ylabel('Sequence Position')
    ax7.set_title('Value Matrix V\n(V = X @ W_V)')
    plt.colorbar(im7, ax=ax7)
    
    # 8. Output
    ax8 = fig.add_subplot(gs[1, 3])
    im8 = ax8.imshow(results['output'], cmap='RdBu_r', aspect='auto',
                     vmin=-np.abs(results['output']).max(),
                     vmax=np.abs(results['output']).max())
    ax8.set_xlabel('Dimension')
    ax8.set_ylabel('Sequence Position')
    ax8.set_title('Output\n(Attention(Q,K,V) @ W_O)')
    plt.colorbar(im8, ax=ax8)
    
    # 9. Hessian Diagonal (Curvature per parameter)
    ax9 = fig.add_subplot(gs[2, :2])
    hess_diag = results['hessian_diag']
    
    # Separate by weight matrix
    d = layer.d_model
    n_per_matrix = d * d
    
    colors = ['red', 'green', 'blue', 'orange']
    labels = ['W_Q', 'W_K', 'W_V', 'W_O']
    
    for i, (color, label) in enumerate(zip(colors, labels)):
        start = i * n_per_matrix
        end = start + n_per_matrix
        ax9.scatter(range(start, end), hess_diag[start:end], 
                   c=color, label=label, alpha=0.6, s=30)
    
    ax9.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax9.set_xlabel('Parameter Index')
    ax9.set_ylabel('Curvature (Diagonal Hessian)')
    ax9.set_title('Loss Curvature per Parameter\n(Pseudo-Spectral View of Optimization Landscape)')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    # 10. Attention Row Statistics
    ax10 = fig.add_subplot(gs[2, 2])
    row_entropies = []
    for row in results['attention_weights']:
        # Compute entropy of attention distribution
        entropy = -np.sum(row * np.log(row + 1e-10))
        row_entropies.append(entropy)
    
    ax10.bar(range(len(row_entropies)), row_entropies, color='teal', alpha=0.7)
    ax10.set_xlabel('Query Position')
    ax10.set_ylabel('Entropy (bits)')
    ax10.set_title('Attention Distribution Entropy\n(Higher = More Dispersed)')
    ax10.grid(True, alpha=0.3)
    
    # 11. Rank Analysis
    ax11 = fig.add_subplot(gs[2, 3])
    ranks = {
        'W_Q': np.sum(results['svd']['W_Q'][1] > 1e-6),
        'W_K': np.sum(results['svd']['W_K'][1] > 1e-6),
        'W_V': np.sum(results['svd']['W_V'][1] > 1e-6),
        'W_O': np.sum(results['svd']['W_O'][1] > 1e-6),
        'Attn': np.linalg.matrix_rank(results['attention_weights'])
    }
    
    ax11.bar(ranks.keys(), ranks.values(), color=['red', 'green', 'blue', 'orange', 'purple'], alpha=0.7)
    ax11.set_ylabel('Effective Rank')
    ax11.set_title('Matrix Ranks\n(Dimensionality of Transformations)')
    ax11.grid(True, alpha=0.3)
    ax11.set_ylim([0, layer.d_model + 1])
    
    # Add text annotations
    for i, (name, rank) in enumerate(ranks.items()):
        ax11.text(i, rank + 0.2, str(rank), ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('Geometric Analysis of Attention Mechanism\n' + 
                 'Pseudo-Spectral Perspective on Transformer Components', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    return fig


def demonstrate_training_dynamics():
    """
    Show how spectral properties evolve during training
    """
    layer = SimpleAttentionLayer(d_model=8, n_heads=2, seq_len=4)
    X = np.random.randn(4, 8) * 0.5
    target = np.random.randn(4, 8) * 0.5
    
    history = {
        'loss': [],
        'sv_Q': [],
        'sv_K': [],
        'sv_V': [],
        'sv_O': [],
        'attention_entropy': []
    }
    
    print("\nTraining attention layer...")
    lr = 0.01
    n_epochs = 200
    
    for epoch in range(n_epochs):
        # Compute loss and gradient
        loss = layer.loss(X, target)
        grad = layer.compute_gradient_numerical(X, target, eps=1e-5)
        
        # Update parameters
        params = layer.get_params_vector()
        params -= lr * grad
        layer.set_params_vector(params)
        
        # Track metrics
        history['loss'].append(loss)
        
        if epoch % 20 == 0:
            # Compute singular values
            U_Q, S_Q, _ = svd(layer.W_Q)
            U_K, S_K, _ = svd(layer.W_K)
            U_V, S_V, _ = svd(layer.W_V)
            U_O, S_O, _ = svd(layer.W_O)
            
            history['sv_Q'].append(S_Q)
            history['sv_K'].append(S_K)
            history['sv_V'].append(S_V)
            history['sv_O'].append(S_O)
            
            # Compute attention entropy
            _, attn_weights, _, _, _ = layer.attention(X)
            entropy = -np.mean(attn_weights * np.log(attn_weights + 1e-10))
            history['attention_entropy'].append(entropy)
            
            print(f"Epoch {epoch}: Loss = {loss:.4f}, Attention Entropy = {entropy:.4f}")
    
    # Visualize training dynamics
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Loss curve
    axes[0, 0].plot(history['loss'], 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Singular values evolution for W_Q
    axes[0, 1].set_title('W_Q Singular Values Evolution')
    for i in range(len(history['sv_Q'][0])):
        sv_trajectory = [sv[i] for sv in history['sv_Q']]
        axes[0, 1].plot(np.arange(0, n_epochs, 20), sv_trajectory, 
                       marker='o', label=f'σ_{i+1}')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Singular Value')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Attention entropy
    axes[0, 2].plot(np.arange(0, n_epochs, 20), history['attention_entropy'], 
                   'g-', marker='o', linewidth=2)
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Entropy')
    axes[0, 2].set_title('Attention Distribution Entropy')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Condition numbers
    axes[1, 0].set_title('Condition Numbers (σ_max / σ_min)')
    for name, svs in [('W_Q', history['sv_Q']), ('W_K', history['sv_K']), 
                      ('W_V', history['sv_V']), ('W_O', history['sv_O'])]:
        cond_numbers = [sv[0] / (sv[-1] + 1e-8) for sv in svs]
        axes[1, 0].plot(np.arange(0, n_epochs, 20), cond_numbers, 
                       marker='o', label=name, linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Condition Number')
    axes[1, 0].set_yscale('log')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, which='both')
    
    # Effective rank evolution
    axes[1, 1].set_title('Effective Rank Evolution')
    threshold = 0.01
    for name, svs in [('W_Q', history['sv_Q']), ('W_K', history['sv_K']), 
                      ('W_V', history['sv_V']), ('W_O', history['sv_O'])]:
        ranks = [np.sum(sv / sv[0] > threshold) for sv in svs]
        axes[1, 1].plot(np.arange(0, n_epochs, 20), ranks, 
                       marker='s', label=name, linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Effective Rank')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Final singular value distribution
    axes[1, 2].set_title('Final Singular Value Distribution')
    axes[1, 2].bar(np.arange(len(history['sv_Q'][-1])) - 0.3, history['sv_Q'][-1], 
                  0.2, label='W_Q', alpha=0.7)
    axes[1, 2].bar(np.arange(len(history['sv_K'][-1])) - 0.1, history['sv_K'][-1], 
                  0.2, label='W_K', alpha=0.7)
    axes[1, 2].bar(np.arange(len(history['sv_V'][-1])) + 0.1, history['sv_V'][-1], 
                  0.2, label='W_V', alpha=0.7)
    axes[1, 2].bar(np.arange(len(history['sv_O'][-1])) + 0.3, history['sv_O'][-1], 
                  0.2, label='W_O', alpha=0.7)
    axes[1, 2].set_xlabel('Singular Value Index')
    axes[1, 2].set_ylabel('Magnitude')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.suptitle('Training Dynamics: Spectral Properties of Attention Layer', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig, layer, history


def main():
    """Main execution"""
    print("=" * 70)
    print("ATTENTION MECHANISM GEOMETRY - PSEUDO-SPECTRAL ANALYSIS")
    print("=" * 70)
    
    # Part 1: Static geometric analysis
    print("\nPart 1: Analyzing geometric structure of attention mechanism...")
    layer, results = analyze_attention_geometry()
    
    fig1 = visualize_attention_geometry(layer, results)
    fig1.savefig('/mnt/user-data/outputs/attention_geometry.png', 
                 dpi=150, bbox_inches='tight')
    print("✓ Saved: attention_geometry.png")
    
    # Part 2: Training dynamics
    print("\nPart 2: Analyzing spectral evolution during training...")
    fig2, trained_layer, history = demonstrate_training_dynamics()
    fig2.savefig('/mnt/user-data/outputs/attention_training_dynamics.png', 
                 dpi=150, bbox_inches='tight')
    print("✓ Saved: attention_training_dynamics.png")
    
    print("\n" + "=" * 70)
    print("KEY INSIGHTS FOR LLMs")
    print("=" * 70)
    
    print("\n1. ATTENTION WEIGHT GEOMETRY:")
    print(f"   - Attention weights sum to 1 per row (softmax constraint)")
    print(f"   - Forms a probability simplex in high-dimensional space")
    print(f"   - Entropy measures how 'spread out' attention is")
    
    print("\n2. WEIGHT MATRIX SPECTRA:")
    svd_Q = results['svd']['W_Q']
    print(f"   - W_Q singular values: {svd_Q[1][:4]}")
    print(f"   - Condition number: {svd_Q[1][0] / (svd_Q[1][-1] + 1e-8):.2f}")
    print(f"   - Low-rank structure emerges during training")
    
    print("\n3. LOSS LANDSCAPE CURVATURE:")
    hess_diag = results['hessian_diag']
    print(f"   - Max curvature: {np.max(hess_diag):.4f}")
    print(f"   - Min curvature: {np.min(hess_diag):.4f}")
    print(f"   - Different matrices have different optimization difficulty")
    
    print("\n4. IMPLICATIONS FOR LLMs:")
    print("   - Attention creates complex loss landscape geometry")
    print("   - Spectral properties determine learning dynamics")
    print("   - Low-rank structure enables efficient fine-tuning (LoRA!)")
    print("   - Condition numbers affect gradient flow through layers")
    
    print("\n" + "=" * 70)
    print("\nVisualization files saved!")
    print("These show the geometric structure underlying transformer attention.")
    plt.show()


if __name__ == "__main__":
    main()
