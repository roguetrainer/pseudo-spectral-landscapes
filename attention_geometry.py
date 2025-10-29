"""
Pseudo-Spectral Analysis for Attention Mechanisms
Demonstrates geometric properties specific to transformer-like architectures
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from matplotlib.gridspec import GridSpec

sns.set_style("whitegrid")

class SimpleAttention:
    """
    Simplified attention mechanism for geometric analysis
    Q, K, V attention: Attention(Q,K,V) = softmax(QK^T/sqrt(d_k))V
    """
    def __init__(self, d_model=4, seed=42):
        np.random.seed(seed)
        self.d_model = d_model
        
        # Query, Key, Value projection matrices
        self.W_q = np.random.randn(d_model, d_model) * 0.1
        self.W_k = np.random.randn(d_model, d_model) * 0.1
        self.W_v = np.random.randn(d_model, d_model) * 0.1
        self.W_o = np.random.randn(d_model, d_model) * 0.1
        
    def softmax(self, x):
        """Numerically stable softmax"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def forward(self, X):
        """
        X: (seq_len, d_model) - input sequence
        Returns: (seq_len, d_model) - output sequence
        """
        Q = X @ self.W_q  # Queries
        K = X @ self.W_k  # Keys
        V = X @ self.W_v  # Values
        
        # Scaled dot-product attention
        scores = (Q @ K.T) / np.sqrt(self.d_model)
        attention_weights = self.softmax(scores)
        
        # Apply attention to values
        attended = attention_weights @ V
        output = attended @ self.W_o
        
        return output, attention_weights
    
    def get_params(self):
        """Flatten all parameters"""
        return np.concatenate([
            self.W_q.flatten(),
            self.W_k.flatten(),
            self.W_v.flatten(),
            self.W_o.flatten()
        ])
    
    def set_params(self, params):
        """Set parameters from flattened vector"""
        d = self.d_model
        self.W_q = params[0:d*d].reshape(d, d)
        self.W_k = params[d*d:2*d*d].reshape(d, d)
        self.W_v = params[2*d*d:3*d*d].reshape(d, d)
        self.W_o = params[3*d*d:4*d*d].reshape(d, d)
    
    def loss(self, X, y):
        """MSE loss between output and target"""
        output, _ = self.forward(X)
        return np.mean((output - y) ** 2)
    
    def compute_gradient(self, X, y, epsilon=1e-5):
        """Numerical gradient computation"""
        params = self.get_params()
        grad = np.zeros_like(params)
        
        base_loss = self.loss(X, y)
        
        for i in range(len(params)):
            params_perturbed = params.copy()
            params_perturbed[i] += epsilon
            self.set_params(params_perturbed)
            
            perturbed_loss = self.loss(X, y)
            grad[i] = (perturbed_loss - base_loss) / epsilon
        
        self.set_params(params)
        return grad


def analyze_attention_rank(attention_weights):
    """
    Analyze the rank and spectral properties of attention matrices
    Low-rank attention → model focuses on few key relationships
    """
    # Compute SVD
    U, s, Vt = np.linalg.svd(attention_weights)
    
    # Effective rank (using entropy-based measure)
    s_normalized = s / np.sum(s)
    entropy = -np.sum(s_normalized * np.log(s_normalized + 1e-10))
    effective_rank = np.exp(entropy)
    
    return s, effective_rank


def compute_attention_hessian_spectrum(model, X, y, num_samples=5):
    """
    Estimate dominant eigenvalues of Hessian for attention mechanism
    Using Lanczos-style approximation with random projections
    """
    params = model.get_params()
    eigenvalue_estimates = []
    
    for _ in range(num_samples):
        # Random direction
        v = np.random.randn(len(params))
        v = v / np.linalg.norm(v)
        
        # Compute Hv (Hessian-vector product) via finite differences
        epsilon = 1e-4
        grad_base = model.compute_gradient(X, y)
        
        model.set_params(params + epsilon * v)
        grad_perturbed = model.compute_gradient(X, y)
        model.set_params(params)
        
        Hv = (grad_perturbed - grad_base) / epsilon
        
        # Rayleigh quotient: v^T H v / v^T v
        eigenvalue_est = np.dot(v, Hv)
        eigenvalue_estimates.append(eigenvalue_est)
    
    return np.array(eigenvalue_estimates)


def visualize_attention_geometry(model, X, y):
    """
    Create comprehensive visualization of attention mechanism geometry
    """
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Get attention weights
    output, attention_weights = model.forward(X)
    
    # 1. Attention pattern visualization
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(attention_weights, cmap='viridis', aspect='auto')
    ax1.set_xlabel('Key Position', fontsize=11)
    ax1.set_ylabel('Query Position', fontsize=11)
    ax1.set_title('Attention Weight Matrix\n(Learned Relationships)', 
                  fontsize=12, fontweight='bold')
    plt.colorbar(im1, ax=ax1, label='Attention Weight')
    
    # 2. Singular values of attention matrix
    ax2 = fig.add_subplot(gs[0, 1])
    singular_values, effective_rank = analyze_attention_rank(attention_weights)
    ax2.bar(range(len(singular_values)), singular_values, 
            color='teal', alpha=0.7, edgecolor='black')
    ax2.axhline(y=singular_values[0] * 0.1, color='r', 
                linestyle='--', label='10% threshold')
    ax2.set_xlabel('Singular Value Index', fontsize=11)
    ax2.set_ylabel('Singular Value', fontsize=11)
    ax2.set_title(f'Attention Matrix Spectrum\nEffective Rank: {effective_rank:.2f}', 
                  fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Attention entropy (sparsity analysis)
    ax3 = fig.add_subplot(gs[0, 2])
    entropies = []
    for i in range(attention_weights.shape[0]):
        probs = attention_weights[i, :]
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        entropies.append(entropy)
    
    ax3.plot(entropies, 'o-', linewidth=2, markersize=8, color='darkblue')
    ax3.axhline(y=np.log(attention_weights.shape[1]), 
                color='r', linestyle='--', label='Uniform (max entropy)')
    ax3.set_xlabel('Query Position', fontsize=11)
    ax3.set_ylabel('Attention Entropy', fontsize=11)
    ax3.set_title('Attention Sparsity Analysis\n(Lower = More Focused)', 
                  fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Parameter space curvature
    ax4 = fig.add_subplot(gs[1, 0])
    eigenvalue_samples = compute_attention_hessian_spectrum(model, X, y, num_samples=20)
    ax4.hist(eigenvalue_samples, bins=15, color='orange', 
             alpha=0.7, edgecolor='black')
    ax4.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax4.set_xlabel('Estimated Eigenvalue', fontsize=11)
    ax4.set_ylabel('Frequency', fontsize=11)
    ax4.set_title('Hessian Eigenvalue Distribution\n(Curvature Profile)', 
                  fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 5. Query/Key correlation structure
    ax5 = fig.add_subplot(gs[1, 1])
    Q = X @ model.W_q
    K = X @ model.W_k
    correlation = np.corrcoef(Q.T)
    im5 = ax5.imshow(correlation, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax5.set_xlabel('Query Dimension', fontsize=11)
    ax5.set_ylabel('Query Dimension', fontsize=11)
    ax5.set_title('Query Space Correlation\n(Feature Redundancy)', 
                  fontsize=12, fontweight='bold')
    plt.colorbar(im5, ax=ax5, label='Correlation')
    
    # 6. Value transformation analysis
    ax6 = fig.add_subplot(gs[1, 2])
    V = X @ model.W_v
    value_norms = np.linalg.norm(V, axis=1)
    ax6.plot(value_norms, 'o-', linewidth=2, markersize=8, color='green')
    ax6.set_xlabel('Sequence Position', fontsize=11)
    ax6.set_ylabel('Value Vector Norm', fontsize=11)
    ax6.set_title('Value Transformation Magnitude\n(Information Flow)', 
                  fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    # 7. Loss landscape slice
    ax7 = fig.add_subplot(gs[2, :], projection='3d')
    
    # Compute loss landscape along two random directions
    params = model.get_params()
    dir1 = np.random.randn(len(params))
    dir1 = dir1 / np.linalg.norm(dir1)
    dir2 = np.random.randn(len(params))
    dir2 = dir2 - np.dot(dir2, dir1) * dir1
    dir2 = dir2 / np.linalg.norm(dir2)
    
    range_val = 0.5
    grid_size = 30
    alphas = np.linspace(-range_val, range_val, grid_size)
    betas = np.linspace(-range_val, range_val, grid_size)
    
    losses = np.zeros((grid_size, grid_size))
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            perturbed_params = params + alpha * dir1 + beta * dir2
            model.set_params(perturbed_params)
            losses[i, j] = model.loss(X, y)
    
    model.set_params(params)
    
    Alpha, Beta = np.meshgrid(alphas, betas)
    surf = ax7.plot_surface(Alpha, Beta, losses.T, cmap=cm.plasma,
                           alpha=0.9, edgecolor='none')
    ax7.scatter([0], [0], [model.loss(X, y)], color='red', 
                s=150, marker='*', label='Current Position')
    
    ax7.set_xlabel('Random Direction 1', fontsize=10)
    ax7.set_ylabel('Random Direction 2', fontsize=10)
    ax7.set_zlabel('Loss', fontsize=10)
    ax7.set_title('Attention Mechanism Loss Landscape\n(2D Slice of Parameter Space)', 
                  fontsize=13, fontweight='bold')
    ax7.view_init(elev=25, azim=135)
    ax7.legend()
    
    plt.suptitle('Geometric Analysis of Attention Mechanisms', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    return fig, attention_weights, singular_values, effective_rank


def train_attention(model, X, y, lr=0.01, epochs=200):
    """Train attention model with gradient descent"""
    losses = []
    attention_ranks = []
    
    for epoch in range(epochs):
        loss = model.loss(X, y)
        losses.append(loss)
        
        # Track attention rank evolution
        _, attention_weights = model.forward(X)
        _, eff_rank = analyze_attention_rank(attention_weights)
        attention_ranks.append(eff_rank)
        
        # Update parameters
        grad = model.compute_gradient(X, y)
        params = model.get_params()
        model.set_params(params - lr * grad)
    
    return losses, attention_ranks


def main():
    """Main demonstration of attention geometry"""
    print("=" * 80)
    print("PSEUDO-SPECTRAL ANALYSIS OF ATTENTION MECHANISMS")
    print("Geometric Properties of Transformer-like Architectures")
    print("=" * 80)
    
    # Create synthetic sequence data
    print("\n1. Generating sequence data...")
    np.random.seed(42)
    seq_len = 8
    d_model = 4
    
    # Input sequence
    X = np.random.randn(seq_len, d_model) * 0.5
    
    # Target: simple transformation (copy with slight modification)
    y = X * 0.8 + np.random.randn(seq_len, d_model) * 0.1
    
    print(f"   Sequence length: {seq_len}")
    print(f"   Model dimension: {d_model}")
    
    # Create attention model
    print("\n2. Initializing attention mechanism...")
    model = SimpleAttention(d_model=d_model, seed=42)
    initial_loss = model.loss(X, y)
    print(f"   Initial loss: {initial_loss:.4f}")
    print(f"   Number of parameters: {len(model.get_params())}")
    
    # Train the model
    print("\n3. Training attention model...")
    losses, attention_ranks = train_attention(model, X, y, lr=0.05, epochs=200)
    final_loss = model.loss(X, y)
    print(f"   Final loss: {final_loss:.4f}")
    print(f"   Loss reduction: {(initial_loss - final_loss)/initial_loss*100:.1f}%")
    
    # Analyze trained model
    print("\n4. Analyzing geometric properties...")
    _, attention_weights = model.forward(X)
    singular_values, effective_rank = analyze_attention_rank(attention_weights)
    
    print(f"   Attention matrix effective rank: {effective_rank:.2f} / {seq_len}")
    print(f"   Rank ratio: {effective_rank/seq_len*100:.1f}%")
    print(f"   Top singular value: {singular_values[0]:.4f}")
    print(f"   Spectral gap: {singular_values[0] - singular_values[1]:.4f}")
    
    # Compute sharpness
    print("\n5. Computing curvature properties...")
    eigenvalue_samples = compute_attention_hessian_spectrum(model, X, y, num_samples=20)
    print(f"   Mean Hessian eigenvalue: {np.mean(eigenvalue_samples):.4f}")
    print(f"   Max estimated curvature: {np.max(eigenvalue_samples):.4f}")
    
    # Create visualizations
    print("\n6. Creating visualizations...")
    fig, att_weights, sing_vals, eff_rank = visualize_attention_geometry(
        model, X, y)
    
    output_path = '/mnt/user-data/outputs/attention_geometry.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   Saved to: {output_path}")
    
    # Additional analysis plot
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Training dynamics
    ax1.plot(losses, linewidth=2, color='blue')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Dynamics', fontsize=13, fontweight='bold')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # Attention rank evolution
    ax2.plot(attention_ranks, linewidth=2, color='green')
    ax2.axhline(y=seq_len, color='r', linestyle='--', 
                label=f'Maximum Rank ({seq_len})')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Effective Rank', fontsize=12)
    ax2.set_title('Attention Rank Evolution\n(Structural Learning)', 
                  fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path2 = '/mnt/user-data/outputs/attention_training_dynamics.png'
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    print(f"   Saved to: {output_path2}")
    
    # Summary
    print("\n" + "=" * 80)
    print("KEY INSIGHTS FOR TRANSFORMER/LLM ARCHITECTURES")
    print("=" * 80)
    
    print("\n📐 ATTENTION GEOMETRY:")
    print(f"   • Attention matrix has low effective rank: {eff_rank:.1f}/{seq_len}")
    print("   • This means attention focuses on few key relationships")
    print("   • Low-rank structure emerges naturally during training")
    
    print("\n🎯 SPECTRAL PROPERTIES:")
    print(f"   • Dominant singular value: {sing_vals[0]:.3f}")
    print("   • Captures most important information flow")
    print("   • Spectral gap indicates clear separation of importance")
    
    print("\n🔍 LOSS LANDSCAPE:")
    print("   • Curvature reveals optimization difficulty")
    print("   • Attention parameters live in high-dimensional space")
    print("   • Geometric understanding → better training strategies")
    
    print("\n💡 IMPLICATIONS FOR LLMs:")
    print("   • Real transformers have multiple attention heads")
    print("   • Each head learns different geometric structures")
    print("   • Pseudo-spectral analysis reveals:")
    print("     - Which directions in parameter space matter most")
    print("     - How attention patterns evolve during training")
    print("     - Why some architectures generalize better")
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)
    
    plt.show()


if __name__ == "__main__":
    main()
