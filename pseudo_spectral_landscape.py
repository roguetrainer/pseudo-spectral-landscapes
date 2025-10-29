"""
Pseudo-Spectral Landscape Analysis for Neural Networks
Demonstrates geometric approaches to understanding neural network optimization
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scipy.linalg import eigh
from sklearn.decomposition import PCA

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

class ToyNeuralNetwork:
    """
    Simple 2-layer neural network for demonstration
    Architecture: input_dim -> hidden_dim -> output_dim
    """
    def __init__(self, input_dim=2, hidden_dim=4, output_dim=1, seed=42):
        np.random.seed(seed)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Initialize weights
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.5
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.5
        self.b2 = np.zeros(output_dim)
        
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def forward(self, X):
        """Forward pass"""
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        return self.z2
    
    def loss(self, X, y):
        """Mean squared error loss"""
        predictions = self.forward(X)
        return np.mean((predictions - y) ** 2)
    
    def get_params(self):
        """Flatten all parameters into a single vector"""
        return np.concatenate([
            self.W1.flatten(),
            self.b1.flatten(),
            self.W2.flatten(),
            self.b2.flatten()
        ])
    
    def set_params(self, params):
        """Set parameters from a flattened vector"""
        idx = 0
        
        # W1
        size = self.input_dim * self.hidden_dim
        self.W1 = params[idx:idx+size].reshape(self.input_dim, self.hidden_dim)
        idx += size
        
        # b1
        size = self.hidden_dim
        self.b1 = params[idx:idx+size]
        idx += size
        
        # W2
        size = self.hidden_dim * self.output_dim
        self.W2 = params[idx:idx+size].reshape(self.hidden_dim, self.output_dim)
        idx += size
        
        # b2
        self.b2 = params[idx:]
    
    def compute_gradients(self, X, y):
        """Compute gradients via backpropagation"""
        m = X.shape[0]
        
        # Forward pass
        predictions = self.forward(X)
        
        # Backward pass
        dz2 = 2 * (predictions - y) / m
        dW2 = self.a1.T @ dz2
        db2 = np.sum(dz2, axis=0)
        
        da1 = dz2 @ self.W2.T
        dz1 = da1 * self.relu_derivative(self.z1)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0)
        
        return {
            'W1': dW1, 'b1': db1,
            'W2': dW2, 'b2': db2
        }
    
    def gradient_vector(self, X, y):
        """Return gradient as a flattened vector"""
        grads = self.compute_gradients(X, y)
        return np.concatenate([
            grads['W1'].flatten(),
            grads['b1'].flatten(),
            grads['W2'].flatten(),
            grads['b2'].flatten()
        ])


class PseudoSpectralAnalyzer:
    """
    Analyzer for pseudo-spectral properties of the loss landscape
    """
    def __init__(self, model, X, y):
        self.model = model
        self.X = X
        self.y = y
        
    def compute_hessian_numerical(self, epsilon=1e-5):
        """
        Compute Hessian matrix numerically using finite differences
        Warning: This is expensive! Only feasible for toy networks.
        """
        params = self.model.get_params()
        n = len(params)
        hessian = np.zeros((n, n))
        
        # Get base gradient
        grad_base = self.model.gradient_vector(self.X, self.y)
        
        for i in range(n):
            # Perturb parameter i
            params_perturbed = params.copy()
            params_perturbed[i] += epsilon
            
            self.model.set_params(params_perturbed)
            grad_perturbed = self.model.gradient_vector(self.X, self.y)
            
            # Finite difference approximation
            hessian[:, i] = (grad_perturbed - grad_base) / epsilon
        
        # Restore original parameters
        self.model.set_params(params)
        
        # Symmetrize
        hessian = (hessian + hessian.T) / 2
        return hessian
    
    def compute_spectrum(self):
        """Compute eigenvalue spectrum of the Hessian"""
        hessian = self.compute_hessian_numerical()
        eigenvalues, eigenvectors = eigh(hessian)
        return eigenvalues, eigenvectors, hessian
    
    def compute_loss_landscape_2d(self, direction1, direction2, 
                                   alpha_range=(-2, 2), beta_range=(-2, 2), 
                                   grid_size=50):
        """
        Compute loss landscape along two directions
        """
        original_params = self.model.get_params()
        
        alphas = np.linspace(alpha_range[0], alpha_range[1], grid_size)
        betas = np.linspace(beta_range[0], beta_range[1], grid_size)
        
        losses = np.zeros((grid_size, grid_size))
        
        for i, alpha in enumerate(alphas):
            for j, beta in enumerate(betas):
                perturbed_params = (original_params + 
                                   alpha * direction1 + 
                                   beta * direction2)
                self.model.set_params(perturbed_params)
                losses[i, j] = self.model.loss(self.X, self.y)
        
        # Restore original parameters
        self.model.set_params(original_params)
        
        return alphas, betas, losses


def train_and_track(model, X, y, learning_rate=0.01, epochs=1000, track_interval=50):
    """
    Train the model and track trajectory through parameter space
    """
    trajectory = []
    losses = []
    
    for epoch in range(epochs):
        # Compute loss
        loss = model.loss(X, y)
        losses.append(loss)
        
        # Track parameters
        if epoch % track_interval == 0:
            trajectory.append(model.get_params().copy())
        
        # Compute gradients
        grads = model.compute_gradients(X, y)
        
        # Update parameters
        model.W1 -= learning_rate * grads['W1']
        model.b1 -= learning_rate * grads['b1']
        model.W2 -= learning_rate * grads['W2']
        model.b2 -= learning_rate * grads['b2']
    
    # Final trajectory point
    trajectory.append(model.get_params().copy())
    
    return np.array(trajectory), np.array(losses)


def visualize_all(model, X, y, trajectory, losses, eigenvalues, eigenvectors):
    """
    Create comprehensive visualization of the pseudo-spectral landscape
    """
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Eigenvalue Spectrum
    ax1 = plt.subplot(2, 3, 1)
    ax1.bar(range(len(eigenvalues)), eigenvalues, color='steelblue', alpha=0.7)
    ax1.axhline(y=0, color='r', linestyle='--', linewidth=1)
    ax1.set_xlabel('Eigenvalue Index', fontsize=12)
    ax1.set_ylabel('Eigenvalue Magnitude', fontsize=12)
    ax1.set_title('Hessian Eigenvalue Spectrum\n(Pseudo-Spectral Signature)', 
                  fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add text annotation about spectrum
    n_positive = np.sum(eigenvalues > 1e-6)
    n_negative = np.sum(eigenvalues < -1e-6)
    ax1.text(0.02, 0.98, f'Positive: {n_positive}\nNegative: {n_negative}\n'
             f'Near-zero: {len(eigenvalues) - n_positive - n_negative}',
             transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', 
             facecolor='wheat', alpha=0.5))
    
    # 2. Loss Surface (Top 2 Principal Directions)
    ax2 = plt.subplot(2, 3, 2, projection='3d')
    
    # Use top 2 eigenvectors (largest magnitude eigenvalues)
    sorted_indices = np.argsort(np.abs(eigenvalues))[::-1]
    direction1 = eigenvectors[:, sorted_indices[0]]
    direction2 = eigenvectors[:, sorted_indices[1]]
    
    analyzer = PseudoSpectralAnalyzer(model, X, y)
    alphas, betas, loss_surface = analyzer.compute_loss_landscape_2d(
        direction1, direction2, alpha_range=(-1.5, 1.5), 
        beta_range=(-1.5, 1.5), grid_size=40
    )
    
    Alpha, Beta = np.meshgrid(alphas, betas)
    surf = ax2.plot_surface(Alpha, Beta, loss_surface.T, cmap=cm.viridis,
                            alpha=0.8, edgecolor='none')
    
    # Mark current position
    ax2.scatter([0], [0], [model.loss(X, y)], color='red', s=100, 
                marker='*', label='Current Position')
    
    ax2.set_xlabel('Principal Direction 1', fontsize=10)
    ax2.set_ylabel('Principal Direction 2', fontsize=10)
    ax2.set_zlabel('Loss', fontsize=10)
    ax2.set_title('Loss Landscape Geometry\n(Top 2 Eigenvector Directions)', 
                  fontsize=13, fontweight='bold')
    ax2.legend()
    
    # 3. Loss Surface Contour
    ax3 = plt.subplot(2, 3, 3)
    contour = ax3.contour(Alpha, Beta, loss_surface.T, levels=20, cmap='viridis')
    ax3.clabel(contour, inline=True, fontsize=8)
    contourf = ax3.contourf(Alpha, Beta, loss_surface.T, levels=20, 
                            cmap='viridis', alpha=0.6)
    plt.colorbar(contourf, ax=ax3, label='Loss')
    
    ax3.plot(0, 0, 'r*', markersize=15, label='Current Position')
    ax3.set_xlabel('Principal Direction 1', fontsize=12)
    ax3.set_ylabel('Principal Direction 2', fontsize=12)
    ax3.set_title('Loss Landscape Contours\n(Geometric Structure)', 
                  fontsize=13, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Training Loss Curve
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(losses, linewidth=2, color='darkblue')
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Loss', fontsize=12)
    ax4.set_title('Training Loss Trajectory', fontsize=13, fontweight='bold')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)
    
    # 5. Trajectory in Parameter Space (PCA projection)
    ax5 = plt.subplot(2, 3, 5)
    
    if len(trajectory) > 2:
        # Project trajectory to 2D using PCA
        pca = PCA(n_components=2)
        trajectory_2d = pca.fit_transform(trajectory)
        
        # Plot trajectory
        ax5.plot(trajectory_2d[:, 0], trajectory_2d[:, 1], 'o-', 
                linewidth=2, markersize=6, alpha=0.7, color='darkgreen')
        ax5.plot(trajectory_2d[0, 0], trajectory_2d[0, 1], 'go', 
                markersize=12, label='Start', zorder=5)
        ax5.plot(trajectory_2d[-1, 0], trajectory_2d[-1, 1], 'r*', 
                markersize=15, label='End', zorder=5)
        
        ax5.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} var)', 
                      fontsize=12)
        ax5.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} var)', 
                      fontsize=12)
        ax5.set_title('Optimization Trajectory\n(PCA Projection)', 
                     fontsize=13, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    
    # 6. Sharpness vs Epoch (Trace of Hessian as proxy)
    ax6 = plt.subplot(2, 3, 6)
    
    # For visualization, show distribution of eigenvalues
    ax6.hist(eigenvalues, bins=20, color='coral', alpha=0.7, edgecolor='black')
    ax6.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero')
    ax6.set_xlabel('Eigenvalue', fontsize=12)
    ax6.set_ylabel('Frequency', fontsize=12)
    ax6.set_title('Eigenvalue Distribution\n(Curvature Profile)', 
                  fontsize=13, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Add sharpness metrics
    trace = np.sum(eigenvalues)
    max_eigenvalue = np.max(eigenvalues)
    ax6.text(0.02, 0.98, f'Trace (Sharpness): {trace:.3f}\n'
             f'Max Eigenvalue: {max_eigenvalue:.3f}',
             transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', 
             facecolor='lightcoral', alpha=0.5))
    
    plt.tight_layout()
    return fig


def main():
    """
    Main demonstration of pseudo-spectral landscape analysis
    """
    print("=" * 70)
    print("PSEUDO-SPECTRAL LANDSCAPE ANALYSIS FOR NEURAL NETWORKS")
    print("=" * 70)
    
    # Generate synthetic dataset
    print("\n1. Generating synthetic dataset...")
    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 2)
    # Simple non-linear function: y = x1^2 + x2^2 + sin(x1)
    y = (X[:, 0]**2 + X[:, 1]**2 + np.sin(X[:, 0])).reshape(-1, 1)
    y += np.random.randn(n_samples, 1) * 0.1  # Add noise
    
    print(f"   Dataset shape: X={X.shape}, y={y.shape}")
    
    # Create and train model
    print("\n2. Training neural network...")
    model = ToyNeuralNetwork(input_dim=2, hidden_dim=4, output_dim=1)
    initial_loss = model.loss(X, y)
    print(f"   Initial loss: {initial_loss:.4f}")
    
    trajectory, losses = train_and_track(model, X, y, learning_rate=0.05, 
                                        epochs=1000, track_interval=50)
    
    final_loss = model.loss(X, y)
    print(f"   Final loss: {final_loss:.4f}")
    print(f"   Loss reduction: {(initial_loss - final_loss) / initial_loss * 100:.2f}%")
    
    # Compute pseudo-spectral properties
    print("\n3. Computing pseudo-spectral properties...")
    analyzer = PseudoSpectralAnalyzer(model, X, y)
    print("   Computing Hessian eigenvalues (this may take a moment)...")
    eigenvalues, eigenvectors, hessian = analyzer.compute_spectrum()
    
    print(f"   Number of parameters: {len(eigenvalues)}")
    print(f"   Eigenvalue range: [{eigenvalues.min():.3f}, {eigenvalues.max():.3f}]")
    print(f"   Trace (sum of eigenvalues): {eigenvalues.sum():.3f}")
    print(f"   Condition number: {eigenvalues.max() / (eigenvalues.min() + 1e-10):.2e}")
    
    # Analyze flatness/sharpness
    positive_eigenvalues = eigenvalues[eigenvalues > 1e-6]
    if len(positive_eigenvalues) > 0:
        avg_positive = np.mean(positive_eigenvalues)
        print(f"   Average positive eigenvalue (sharpness): {avg_positive:.3f}")
    
    # Create visualizations
    print("\n4. Creating visualizations...")
    fig = visualize_all(model, X, y, trajectory, losses, 
                       eigenvalues, eigenvectors)
    
    output_path = '/mnt/user-data/outputs/pseudo_spectral_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   Visualization saved to: {output_path}")
    
    # Summary insights
    print("\n" + "=" * 70)
    print("KEY INSIGHTS FROM PSEUDO-SPECTRAL ANALYSIS")
    print("=" * 70)
    print("\n• EIGENVALUE SPECTRUM:")
    print(f"  - Reveals the curvature structure of the loss landscape")
    print(f"  - {np.sum(eigenvalues > 0)} positive eigenvalues (upward curvature)")
    print(f"  - {np.sum(eigenvalues < 0)} negative eigenvalues (downward curvature)")
    print(f"  - {np.sum(np.abs(eigenvalues) < 1e-6)} near-zero eigenvalues (flat directions)")
    
    print("\n• GEOMETRIC INTERPRETATION:")
    print(f"  - Large eigenvalues → sharp minimum → may not generalize well")
    print(f"  - Small eigenvalues → flat minimum → better generalization")
    print(f"  - Current sharpness (trace): {eigenvalues.sum():.3f}")
    
    print("\n• LOSS LANDSCAPE:")
    print(f"  - 3D surface shows geometry along principal curvature directions")
    print(f"  - Contours reveal basin of attraction structure")
    print(f"  - Trajectory shows path through parameter space during optimization")
    
    print("\n" + "=" * 70)
    print("Analysis complete! Check the visualization for detailed insights.")
    print("=" * 70)
    
    plt.show()
    
    return model, X, y, eigenvalues, eigenvectors


if __name__ == "__main__":
    model, X, y, eigenvalues, eigenvectors = main()
