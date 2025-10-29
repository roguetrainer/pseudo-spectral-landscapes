"""
Pseudo-Spectral Landscapes and Loss Geometry Demonstration
===========================================================
This script demonstrates key concepts in understanding neural network optimization
through geometric and spectral analysis of the loss landscape.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scipy.linalg import eigh
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class ToyNeuralNetwork:
    """
    A simple 2-layer neural network for demonstration.
    We keep it small so we can actually compute and visualize the Hessian.
    """
    
    def __init__(self, input_dim=2, hidden_dim=3, output_dim=1, seed=42):
        np.random.seed(seed)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Initialize weights
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.5
        self.b1 = np.random.randn(hidden_dim) * 0.1
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.5
        self.b2 = np.random.randn(output_dim) * 0.1
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)
    
    def forward(self, X):
        """Forward pass"""
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        return self.z2
    
    def get_params_vector(self):
        """Flatten all parameters into a single vector"""
        return np.concatenate([
            self.W1.flatten(),
            self.b1.flatten(),
            self.W2.flatten(),
            self.b2.flatten()
        ])
    
    def set_params_vector(self, params):
        """Set parameters from a flattened vector"""
        idx = 0
        
        w1_size = self.input_dim * self.hidden_dim
        self.W1 = params[idx:idx + w1_size].reshape(self.input_dim, self.hidden_dim)
        idx += w1_size
        
        self.b1 = params[idx:idx + self.hidden_dim]
        idx += self.hidden_dim
        
        w2_size = self.hidden_dim * self.output_dim
        self.W2 = params[idx:idx + w2_size].reshape(self.hidden_dim, self.output_dim)
        idx += w2_size
        
        self.b2 = params[idx:idx + self.output_dim]
    
    def loss(self, X, y):
        """MSE loss"""
        pred = self.forward(X)
        return 0.5 * np.mean((pred - y) ** 2)
    
    def compute_gradient(self, X, y):
        """Compute gradient via backpropagation"""
        m = X.shape[0]
        
        # Forward pass
        pred = self.forward(X)
        
        # Backward pass
        dz2 = (pred - y) / m
        dW2 = self.a1.T @ dz2
        db2 = np.sum(dz2, axis=0)
        
        da1 = dz2 @ self.W2.T
        dz1 = da1 * self.sigmoid_derivative(self.z1)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0)
        
        return np.concatenate([
            dW1.flatten(),
            db1.flatten(),
            dW2.flatten(),
            db2.flatten()
        ])
    
    def compute_hessian(self, X, y):
        """
        Compute the Hessian matrix numerically using finite differences.
        This is the key for pseudo-spectral analysis!
        """
        n_params = len(self.get_params_vector())
        H = np.zeros((n_params, n_params))
        eps = 1e-5
        
        def loss_fn(params):
            self.set_params_vector(params)
            return self.loss(X, y)
        
        params = self.get_params_vector()
        
        for i in range(n_params):
            for j in range(i, n_params):
                # Compute second derivative using finite differences
                params_pp = params.copy()
                params_pp[i] += eps
                params_pp[j] += eps
                
                params_pm = params.copy()
                params_pm[i] += eps
                params_pm[j] -= eps
                
                params_mp = params.copy()
                params_mp[i] -= eps
                params_mp[j] += eps
                
                params_mm = params.copy()
                params_mm[i] -= eps
                params_mm[j] -= eps
                
                f_pp = loss_fn(params_pp)
                f_pm = loss_fn(params_pm)
                f_mp = loss_fn(params_mp)
                f_mm = loss_fn(params_mm)
                
                H[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * eps * eps)
                H[j, i] = H[i, j]
        
        # Reset to original parameters
        self.set_params_vector(params)
        return H


def generate_synthetic_data(n_samples=100, seed=42):
    """Generate simple synthetic data for demonstration"""
    np.random.seed(seed)
    X = np.random.randn(n_samples, 2)
    # Simple nonlinear function
    y = (np.sin(X[:, 0]) + 0.5 * X[:, 1]**2).reshape(-1, 1)
    y += np.random.randn(n_samples, 1) * 0.1  # Add noise
    return X, y


def visualize_loss_landscape_2d(nn, X, y, param_indices=(0, 1), 
                                 range_scale=2.0, resolution=50):
    """
    Visualize the loss landscape along two parameter directions.
    This is a 2D slice through the high-dimensional loss landscape.
    """
    params = nn.get_params_vector()
    i, j = param_indices
    
    # Create grid around current parameters
    param_range_i = np.linspace(params[i] - range_scale, 
                                 params[i] + range_scale, resolution)
    param_range_j = np.linspace(params[j] - range_scale, 
                                 params[j] + range_scale, resolution)
    
    loss_grid = np.zeros((resolution, resolution))
    
    for idx_i, pi in enumerate(param_range_i):
        for idx_j, pj in enumerate(param_range_j):
            params_temp = params.copy()
            params_temp[i] = pi
            params_temp[j] = pj
            nn.set_params_vector(params_temp)
            loss_grid[idx_j, idx_i] = nn.loss(X, y)
    
    # Reset parameters
    nn.set_params_vector(params)
    
    return param_range_i, param_range_j, loss_grid


def analyze_hessian_spectrum(nn, X, y):
    """
    Compute and analyze the eigenvalue spectrum of the Hessian.
    This is the core of pseudo-spectral analysis!
    """
    print("Computing Hessian (this may take a moment)...")
    H = nn.compute_hessian(X, y)
    
    # Compute eigenvalues
    eigenvalues, eigenvectors = eigh(H)
    
    return eigenvalues, eigenvectors, H


def plot_comprehensive_analysis(nn, X, y, training_history=None):
    """
    Create a comprehensive visualization of the loss landscape and 
    pseudo-spectral properties.
    """
    fig = plt.figure(figsize=(20, 12))
    
    # 1. 2D Loss Landscape Contour (parameters 0 and 1)
    ax1 = plt.subplot(2, 4, 1)
    param_i, param_j, loss_grid = visualize_loss_landscape_2d(
        nn, X, y, param_indices=(0, 1), range_scale=1.5, resolution=40
    )
    contour = ax1.contour(param_i, param_j, loss_grid, levels=20, cmap='viridis')
    ax1.contourf(param_i, param_j, loss_grid, levels=20, cmap='viridis', alpha=0.6)
    plt.colorbar(contour, ax=ax1)
    params = nn.get_params_vector()
    ax1.plot(params[0], params[1], 'r*', markersize=15, label='Current position')
    ax1.set_xlabel('Parameter 0 (W1[0,0])')
    ax1.set_ylabel('Parameter 1 (W1[0,1])')
    ax1.set_title('Loss Landscape (2D Slice)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 3D Loss Landscape Surface
    ax2 = plt.subplot(2, 4, 2, projection='3d')
    Pi, Pj = np.meshgrid(param_i, param_j)
    surf = ax2.plot_surface(Pi, Pj, loss_grid, cmap='viridis', alpha=0.8)
    ax2.set_xlabel('Parameter 0')
    ax2.set_ylabel('Parameter 1')
    ax2.set_zlabel('Loss')
    ax2.set_title('3D Loss Surface')
    
    # 3. Hessian Spectrum Analysis
    eigenvalues, eigenvectors, H = analyze_hessian_spectrum(nn, X, y)
    
    ax3 = plt.subplot(2, 4, 3)
    ax3.bar(range(len(eigenvalues)), eigenvalues, color='steelblue', alpha=0.7)
    ax3.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Eigenvalue Index')
    ax3.set_ylabel('Eigenvalue Magnitude')
    ax3.set_title('Hessian Eigenvalue Spectrum')
    ax3.grid(True, alpha=0.3)
    
    # Add statistics
    n_positive = np.sum(eigenvalues > 1e-6)
    n_negative = np.sum(eigenvalues < -1e-6)
    ax3.text(0.05, 0.95, f'Positive: {n_positive}\nNegative: {n_negative}\nZero: {len(eigenvalues)-n_positive-n_negative}',
             transform=ax3.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 4. Eigenvalue Distribution (Histogram)
    ax4 = plt.subplot(2, 4, 4)
    ax4.hist(eigenvalues, bins=20, color='coral', alpha=0.7, edgecolor='black')
    ax4.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero')
    ax4.set_xlabel('Eigenvalue')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Eigenvalue Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Hessian Matrix Heatmap
    ax5 = plt.subplot(2, 4, 5)
    im = ax5.imshow(H, cmap='RdBu_r', aspect='auto', vmin=-np.abs(H).max(), vmax=np.abs(H).max())
    plt.colorbar(im, ax=ax5)
    ax5.set_xlabel('Parameter Index')
    ax5.set_ylabel('Parameter Index')
    ax5.set_title('Hessian Matrix')
    
    # 6. Loss along principal curvature directions
    ax6 = plt.subplot(2, 4, 6)
    current_params = nn.get_params_vector()
    
    # Plot loss along top 3 eigenvector directions
    alphas = np.linspace(-1, 1, 50)
    colors = ['red', 'green', 'blue']
    
    for idx in [0, len(eigenvalues)//2, -1]:  # Min, middle, max curvature
        losses = []
        for alpha in alphas:
            perturbed = current_params + alpha * eigenvectors[:, idx]
            nn.set_params_vector(perturbed)
            losses.append(nn.loss(X, y))
        nn.set_params_vector(current_params)
        
        label = f'λ={eigenvalues[idx]:.3f}'
        ax6.plot(alphas, losses, label=label, linewidth=2)
    
    ax6.set_xlabel('Step size α')
    ax6.set_ylabel('Loss')
    ax6.set_title('Loss Along Principal Curvature Directions')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Curvature vs Eigenvalue (log scale)
    ax7 = plt.subplot(2, 4, 7)
    sorted_eigs = np.sort(np.abs(eigenvalues))[::-1]
    ax7.semilogy(range(len(sorted_eigs)), sorted_eigs, 'o-', color='purple', markersize=6)
    ax7.set_xlabel('Eigenvalue Rank')
    ax7.set_ylabel('|Eigenvalue| (log scale)')
    ax7.set_title('Curvature Spectrum (Sorted)')
    ax7.grid(True, alpha=0.3, which='both')
    
    # 8. Training History (if provided)
    ax8 = plt.subplot(2, 4, 8)
    if training_history is not None:
        ax8.plot(training_history['loss'], 'b-', linewidth=2, label='Training Loss')
        ax8.set_xlabel('Epoch')
        ax8.set_ylabel('Loss')
        ax8.set_title('Training History')
        ax8.set_yscale('log')
        ax8.legend()
        ax8.grid(True, alpha=0.3, which='both')
    else:
        # Show gradient norm instead
        grad = nn.compute_gradient(X, y)
        grad_norms = []
        for i in range(len(grad)):
            grad_norms.append(np.abs(grad[i]))
        ax8.bar(range(len(grad_norms)), grad_norms, color='teal', alpha=0.7)
        ax8.set_xlabel('Parameter Index')
        ax8.set_ylabel('|Gradient|')
        ax8.set_title('Gradient Magnitudes')
        ax8.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def train_and_analyze(X, y, epochs=100, lr=0.1):
    """
    Train the network and track how the spectral properties evolve.
    """
    nn = ToyNeuralNetwork()
    history = {'loss': [], 'max_eigenvalue': [], 'min_eigenvalue': [], 'trace': []}
    
    print("Training neural network...")
    for epoch in range(epochs):
        # Forward and backward pass
        loss = nn.loss(X, y)
        grad = nn.compute_gradient(X, y)
        
        # Update parameters
        params = nn.get_params_vector()
        params -= lr * grad
        nn.set_params_vector(params)
        
        history['loss'].append(loss)
        
        # Compute Hessian spectrum periodically
        if epoch % 20 == 0:
            eigenvalues, _, _ = analyze_hessian_spectrum(nn, X, y)
            history['max_eigenvalue'].append(np.max(eigenvalues))
            history['min_eigenvalue'].append(np.min(eigenvalues))
            history['trace'].append(np.sum(eigenvalues))
            print(f"Epoch {epoch}: Loss = {loss:.4f}, "
                  f"λ_max = {np.max(eigenvalues):.4f}, "
                  f"λ_min = {np.min(eigenvalues):.4f}")
    
    return nn, history


def plot_spectral_evolution(history):
    """
    Visualize how the spectral properties evolve during training.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs_hessian = np.arange(0, len(history['loss']), 20)
    
    # Loss curve
    axes[0, 0].plot(history['loss'], 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Max and min eigenvalues
    axes[0, 1].plot(epochs_hessian, history['max_eigenvalue'], 'r-', 
                    linewidth=2, label='λ_max', marker='o')
    axes[0, 1].plot(epochs_hessian, history['min_eigenvalue'], 'b-', 
                    linewidth=2, label='λ_min', marker='s')
    axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Eigenvalue')
    axes[0, 1].set_title('Extreme Eigenvalues During Training')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Trace (sum of eigenvalues)
    axes[1, 0].plot(epochs_hessian, history['trace'], 'g-', 
                    linewidth=2, marker='o')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Trace(H)')
    axes[1, 0].set_title('Hessian Trace (Total Curvature)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Sharpness ratio
    sharpness = np.array(history['max_eigenvalue']) / (np.abs(np.array(history['min_eigenvalue'])) + 1e-8)
    axes[1, 1].plot(epochs_hessian, sharpness, 'm-', linewidth=2, marker='d')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('λ_max / |λ_min|')
    axes[1, 1].set_title('Sharpness Ratio (Flat vs Sharp Minimum)')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    return fig


def main():
    """
    Main demonstration of pseudo-spectral landscape concepts.
    """
    print("=" * 70)
    print("PSEUDO-SPECTRAL LANDSCAPES DEMONSTRATION")
    print("=" * 70)
    print()
    
    # Generate data
    print("Generating synthetic data...")
    X, y = generate_synthetic_data(n_samples=50)
    print(f"Data shape: X={X.shape}, y={y.shape}")
    print()
    
    # Train network and analyze evolution
    nn, history = train_and_analyze(X, y, epochs=100, lr=0.05)
    print()
    
    # Create comprehensive analysis plots
    print("Creating comprehensive analysis visualizations...")
    fig1 = plot_comprehensive_analysis(nn, X, y, history)
    plt.savefig('/mnt/user-data/outputs/loss_landscape_analysis.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: loss_landscape_analysis.png")
    
    # Plot spectral evolution
    print("Creating spectral evolution plots...")
    fig2 = plot_spectral_evolution(history)
    plt.savefig('/mnt/user-data/outputs/spectral_evolution.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: spectral_evolution.png")
    
    print()
    print("=" * 70)
    print("KEY INSIGHTS FROM PSEUDO-SPECTRAL ANALYSIS")
    print("=" * 70)
    
    # Final analysis
    eigenvalues, eigenvectors, H = analyze_hessian_spectrum(nn, X, y)
    
    print(f"\n1. HESSIAN SPECTRUM:")
    print(f"   - Number of parameters: {len(eigenvalues)}")
    print(f"   - Max eigenvalue (sharpest direction): {np.max(eigenvalues):.4f}")
    print(f"   - Min eigenvalue (flattest direction): {np.min(eigenvalues):.4f}")
    print(f"   - Condition number: {np.max(eigenvalues) / (np.abs(np.min(eigenvalues)) + 1e-8):.2f}")
    print(f"   - Trace (total curvature): {np.sum(eigenvalues):.4f}")
    
    print(f"\n2. LOSS LANDSCAPE GEOMETRY:")
    n_positive = np.sum(eigenvalues > 1e-6)
    n_negative = np.sum(eigenvalues < -1e-6)
    n_zero = len(eigenvalues) - n_positive - n_negative
    
    if n_negative > 0:
        print(f"   - Type: Saddle point ({n_negative} negative eigenvalues)")
    elif n_positive == len(eigenvalues):
        print(f"   - Type: Local minimum (all positive eigenvalues)")
    else:
        print(f"   - Type: Approximately flat ({n_zero} near-zero eigenvalues)")
    
    print(f"\n3. GENERALIZATION PROPERTIES:")
    flatness_score = 1.0 / (1.0 + np.max(eigenvalues))
    print(f"   - Flatness score: {flatness_score:.4f}")
    print(f"     (Higher values indicate flatter minima → better generalization)")
    
    print(f"\n4. OPTIMIZATION INSIGHTS:")
    print(f"   - Dominant curvature directions: {np.sum(eigenvalues > 1.0)}")
    print(f"   - Nearly flat directions: {n_zero}")
    print(f"   - These flat directions allow easy movement without increasing loss")
    
    print("\n" + "=" * 70)
    print()
    print("Visualization files saved to /mnt/user-data/outputs/")
    print("All plots show different aspects of the loss landscape geometry!")
    
    plt.show()


if __name__ == "__main__":
    main()
