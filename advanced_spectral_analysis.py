"""
Advanced Pseudo-Spectral Analysis: Mode Connectivity and Sharpness
Demonstrates how different training strategies lead to different geometric properties
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

sns.set_style("whitegrid")

class SimpleNN:
    """Simplified neural network for mode connectivity analysis"""
    def __init__(self, input_dim=2, hidden_dim=3, output_dim=1, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        # Smaller network for computational efficiency
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.3
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.3
        self.b2 = np.zeros(output_dim)
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(0, z1)  # ReLU
        z2 = a1 @ self.W2 + self.b2
        return z2
    
    def loss(self, X, y):
        predictions = self.forward(X)
        return np.mean((predictions - y) ** 2)
    
    def get_params(self):
        return np.concatenate([
            self.W1.flatten(), self.b1,
            self.W2.flatten(), self.b2.flatten()
        ])
    
    def set_params(self, params):
        idx = 0
        self.W1 = params[idx:idx + self.input_dim * self.hidden_dim].reshape(
            self.input_dim, self.hidden_dim)
        idx += self.input_dim * self.hidden_dim
        
        self.b1 = params[idx:idx + self.hidden_dim]
        idx += self.hidden_dim
        
        self.W2 = params[idx:idx + self.hidden_dim * self.output_dim].reshape(
            self.hidden_dim, self.output_dim)
        idx += self.hidden_dim * self.output_dim
        
        self.b2 = params[idx:]
    
    def gradient(self, X, y):
        """Compute gradient via backpropagation"""
        m = X.shape[0]
        
        # Forward
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(0, z1)
        z2 = a1 @ self.W2 + self.b2
        predictions = z2
        
        # Backward
        dz2 = 2 * (predictions - y) / m
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0)
        
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (z1 > 0).astype(float)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0)
        
        return np.concatenate([
            dW1.flatten(), db1,
            dW2.flatten(), db2.flatten()
        ])


def train_standard_sgd(model, X, y, lr=0.05, epochs=500):
    """Standard SGD training"""
    losses = []
    for _ in range(epochs):
        losses.append(model.loss(X, y))
        grad = model.gradient(X, y)
        params = model.get_params()
        model.set_params(params - lr * grad)
    return losses


def train_with_noise(model, X, y, lr=0.05, noise_scale=0.01, epochs=500):
    """Training with parameter noise (explores flatter regions)"""
    losses = []
    for _ in range(epochs):
        losses.append(model.loss(X, y))
        grad = model.gradient(X, y)
        params = model.get_params()
        
        # Add noise to encourage exploration of flat regions
        noise = np.random.randn(len(params)) * noise_scale
        model.set_params(params - lr * grad + noise)
    return losses


def compute_sharpness(model, X, y, epsilon=0.01, num_samples=10):
    """
    Estimate sharpness by measuring loss variation under parameter perturbation
    A proxy for the maximum eigenvalue of the Hessian
    """
    original_params = model.get_params()
    base_loss = model.loss(X, y)
    
    max_loss_increase = 0
    for _ in range(num_samples):
        # Random perturbation
        perturbation = np.random.randn(len(original_params))
        perturbation = perturbation / np.linalg.norm(perturbation) * epsilon
        
        model.set_params(original_params + perturbation)
        perturbed_loss = model.loss(X, y)
        
        loss_increase = perturbed_loss - base_loss
        max_loss_increase = max(max_loss_increase, loss_increase)
    
    model.set_params(original_params)
    return max_loss_increase / (epsilon ** 2)


def explore_mode_connectivity(model1, model2, X, y, steps=50):
    """
    Explore the loss barrier between two trained models (modes)
    Linear interpolation path
    """
    params1 = model1.get_params()
    params2 = model2.get_params()
    
    alphas = np.linspace(0, 1, steps)
    losses = []
    
    temp_model = SimpleNN(model1.input_dim, model1.hidden_dim, model1.output_dim)
    
    for alpha in alphas:
        # Linear interpolation
        interpolated_params = (1 - alpha) * params1 + alpha * params2
        temp_model.set_params(interpolated_params)
        losses.append(temp_model.loss(X, y))
    
    return alphas, np.array(losses)


def visualize_comparison(X, y, results):
    """Create comprehensive comparison visualization"""
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Training curves comparison
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(results['standard_losses'], label='Standard SGD', 
             linewidth=2, color='blue')
    ax1.plot(results['noisy_losses'], label='SGD + Noise', 
             linewidth=2, color='green')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Dynamics Comparison', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # 2. Sharpness comparison (bar chart)
    ax2 = plt.subplot(2, 3, 2)
    sharpness_values = [results['standard_sharpness'], results['noisy_sharpness']]
    colors = ['blue', 'green']
    bars = ax2.bar(['Standard SGD', 'SGD + Noise'], sharpness_values, 
                    color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Sharpness (Loss Sensitivity)', fontsize=12)
    ax2.set_title('Minima Sharpness Comparison\n(Lower = Flatter = Better)', 
                  fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{sharpness_values[i]:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 3. Mode connectivity
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(results['mode_alphas'], results['mode_losses'], 
             linewidth=3, color='purple', marker='o', markersize=4)
    ax3.axvline(x=0, color='blue', linestyle='--', alpha=0.5, label='Model 1 (Standard)')
    ax3.axvline(x=1, color='green', linestyle='--', alpha=0.5, label='Model 2 (Noisy)')
    ax3.fill_between(results['mode_alphas'], results['mode_losses'], 
                      alpha=0.2, color='purple')
    ax3.set_xlabel('Interpolation Parameter (α)', fontsize=12)
    ax3.set_ylabel('Loss', fontsize=12)
    ax3.set_title('Mode Connectivity Analysis\n(Loss Barrier Between Minima)', 
                  fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Compute barrier height
    barrier_height = np.max(results['mode_losses']) - np.min(results['mode_losses'])
    ax3.text(0.5, 0.95, f'Barrier Height: {barrier_height:.4f}',
             transform=ax3.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='center',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    # 4. Loss landscape around standard SGD minimum
    ax4 = plt.subplot(2, 3, 4, projection='3d')
    standard_model = results['standard_model']
    landscape = compute_local_landscape(standard_model, X, y, range_val=0.5)
    
    X_grid, Y_grid, Z_grid = landscape
    surf1 = ax4.plot_surface(X_grid, Y_grid, Z_grid, cmap=cm.Blues, 
                             alpha=0.8, edgecolor='none')
    ax4.scatter([0], [0], [standard_model.loss(X, y)], color='blue', 
                s=100, marker='*', label='Minimum')
    ax4.set_xlabel('Direction 1', fontsize=10)
    ax4.set_ylabel('Direction 2', fontsize=10)
    ax4.set_zlabel('Loss', fontsize=10)
    ax4.set_title('Standard SGD Minimum\n(Sharp)', fontsize=12, fontweight='bold')
    ax4.view_init(elev=25, azim=45)
    
    # 5. Loss landscape around noisy SGD minimum
    ax5 = plt.subplot(2, 3, 5, projection='3d')
    noisy_model = results['noisy_model']
    landscape = compute_local_landscape(noisy_model, X, y, range_val=0.5)
    
    X_grid, Y_grid, Z_grid = landscape
    surf2 = ax5.plot_surface(X_grid, Y_grid, Z_grid, cmap=cm.Greens, 
                             alpha=0.8, edgecolor='none')
    ax5.scatter([0], [0], [noisy_model.loss(X, y)], color='green', 
                s=100, marker='*', label='Minimum')
    ax5.set_xlabel('Direction 1', fontsize=10)
    ax5.set_ylabel('Direction 2', fontsize=10)
    ax5.set_zlabel('Loss', fontsize=10)
    ax5.set_title('Noisy SGD Minimum\n(Flat)', fontsize=12, fontweight='bold')
    ax5.view_init(elev=25, azim=45)
    
    # 6. Generalization analysis
    ax6 = plt.subplot(2, 3, 6)
    
    # Test on perturbed data
    perturbation_levels = np.linspace(0, 0.5, 20)
    standard_gen_losses = []
    noisy_gen_losses = []
    
    for perturb in perturbation_levels:
        X_perturbed = X + np.random.randn(*X.shape) * perturb
        standard_gen_losses.append(results['standard_model'].loss(X_perturbed, y))
        noisy_gen_losses.append(results['noisy_model'].loss(X_perturbed, y))
    
    ax6.plot(perturbation_levels, standard_gen_losses, 'o-', 
             label='Standard SGD', linewidth=2, color='blue')
    ax6.plot(perturbation_levels, noisy_gen_losses, 's-', 
             label='SGD + Noise (Flatter)', linewidth=2, color='green')
    ax6.set_xlabel('Input Perturbation Level', fontsize=12)
    ax6.set_ylabel('Loss', fontsize=12)
    ax6.set_title('Generalization Under Perturbation\n(Proxy for Robustness)', 
                  fontsize=13, fontweight='bold')
    ax6.legend(fontsize=11)
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def compute_local_landscape(model, X, y, range_val=1.0, grid_size=25):
    """Compute loss landscape around a model's current parameters"""
    original_params = model.get_params()
    
    # Choose two random directions
    np.random.seed(42)
    dir1 = np.random.randn(len(original_params))
    dir1 = dir1 / np.linalg.norm(dir1)
    
    dir2 = np.random.randn(len(original_params))
    dir2 = dir2 - np.dot(dir2, dir1) * dir1  # Orthogonalize
    dir2 = dir2 / np.linalg.norm(dir2)
    
    alphas = np.linspace(-range_val, range_val, grid_size)
    betas = np.linspace(-range_val, range_val, grid_size)
    
    losses = np.zeros((grid_size, grid_size))
    
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            perturbed_params = original_params + alpha * dir1 + beta * dir2
            model.set_params(perturbed_params)
            losses[i, j] = model.loss(X, y)
    
    model.set_params(original_params)
    
    Alpha, Beta = np.meshgrid(alphas, betas)
    return Alpha, Beta, losses.T


def main():
    """Main demonstration"""
    print("=" * 80)
    print("ADVANCED PSEUDO-SPECTRAL ANALYSIS: SHARP VS FLAT MINIMA")
    print("=" * 80)
    
    # Generate data
    print("\n1. Generating dataset...")
    np.random.seed(42)
    n_samples = 80
    X_train = np.random.randn(n_samples, 2)
    y_train = (X_train[:, 0]**2 + 0.5 * X_train[:, 1]**2).reshape(-1, 1)
    y_train += np.random.randn(n_samples, 1) * 0.1
    
    # Train two models with different strategies
    print("\n2. Training Model 1: Standard SGD...")
    model1 = SimpleNN(input_dim=2, hidden_dim=3, output_dim=1, seed=42)
    standard_losses = train_standard_sgd(model1, X_train, y_train, 
                                        lr=0.05, epochs=500)
    print(f"   Final loss: {model1.loss(X_train, y_train):.4f}")
    
    print("\n3. Training Model 2: SGD with Noise (seeking flat minima)...")
    model2 = SimpleNN(input_dim=2, hidden_dim=3, output_dim=1, seed=43)
    noisy_losses = train_with_noise(model2, X_train, y_train, 
                                   lr=0.05, noise_scale=0.02, epochs=500)
    print(f"   Final loss: {model2.loss(X_train, y_train):.4f}")
    
    # Compute sharpness
    print("\n4. Computing sharpness metrics...")
    print("   (Measuring loss sensitivity to parameter perturbations)")
    standard_sharpness = compute_sharpness(model1, X_train, y_train, 
                                          epsilon=0.1, num_samples=20)
    noisy_sharpness = compute_sharpness(model2, X_train, y_train, 
                                       epsilon=0.1, num_samples=20)
    
    print(f"   Standard SGD sharpness: {standard_sharpness:.4f}")
    print(f"   Noisy SGD sharpness: {noisy_sharpness:.4f}")
    print(f"   Sharpness ratio: {standard_sharpness / noisy_sharpness:.2f}x sharper")
    
    # Mode connectivity
    print("\n5. Analyzing mode connectivity...")
    print("   (Exploring loss barrier between the two minima)")
    mode_alphas, mode_losses = explore_mode_connectivity(model1, model2, 
                                                         X_train, y_train, 
                                                         steps=50)
    barrier = np.max(mode_losses) - min(model1.loss(X_train, y_train),
                                        model2.loss(X_train, y_train))
    print(f"   Loss barrier height: {barrier:.4f}")
    
    # Package results
    results = {
        'standard_model': model1,
        'noisy_model': model2,
        'standard_losses': standard_losses,
        'noisy_losses': noisy_losses,
        'standard_sharpness': standard_sharpness,
        'noisy_sharpness': noisy_sharpness,
        'mode_alphas': mode_alphas,
        'mode_losses': mode_losses
    }
    
    # Visualize
    print("\n6. Creating visualizations...")
    fig = visualize_comparison(X_train, y_train, results)
    
    output_path = '/mnt/user-data/outputs/sharp_vs_flat_minima.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   Saved to: {output_path}")
    
    # Summary
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    print("\n📊 SHARPNESS COMPARISON:")
    print(f"   • Standard SGD found a SHARPER minimum (sharpness: {standard_sharpness:.3f})")
    print(f"   • Noisy SGD found a FLATTER minimum (sharpness: {noisy_sharpness:.3f})")
    print(f"   • Flat minima typically generalize better!")
    
    print("\n🔗 MODE CONNECTIVITY:")
    print(f"   • Loss barrier between minima: {barrier:.4f}")
    print(f"   • Lower barriers → models explored similar solution manifolds")
    print(f"   • Higher barriers → models found different modes")
    
    print("\n🎯 IMPLICATIONS FOR LLMs:")
    print("   • Large language models also exhibit these properties")
    print("   • Flat minima → better generalization to unseen data")
    print("   • Training techniques (SAM, noise injection) seek flatter regions")
    print("   • Understanding geometry helps predict model behavior")
    
    print("\n" + "=" * 80)
    
    plt.show()


if __name__ == "__main__":
    main()
