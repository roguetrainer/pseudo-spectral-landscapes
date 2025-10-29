# Pseudo-Spectral Landscapes: Complete Demonstration

This repository contains a comprehensive exploration of **pseudo-spectral landscapes** and their relevance to understanding Large Language Models (LLMs) through geometric approaches.

![image](./pseudo-spectral-landscapes.png)

## 📁 Files Overview

### Code Files

1. **`pseudo_spectral_demo.py`** - Core demonstration
   - Implements a toy neural network
   - Computes Hessian matrices and eigenvalue spectra
   - Visualizes 2D/3D loss landscapes
   - Tracks spectral evolution during training
   - Generates: `loss_landscape_analysis.png`, `spectral_evolution.png`

2. **`attention_geometry_demo.py`** - Transformer-specific analysis
   - Implements simplified self-attention mechanism
   - Analyzes singular value decomposition of weight matrices
   - Studies attention weight geometry
   - Shows spectral properties during attention training
   - Generates: `attention_geometry.png`, `attention_training_dynamics.png`

### Documentation

3. **`EXPLANATION.md`** - Detailed conceptual guide
4. **`README.md`** (this file) - Complete reference

## 🚀 Quick Start

```bash
python pseudo_spectral_demo.py
python attention_geometry_demo.py
```

## 🎯 What Are Pseudo-Spectral Landscapes?

Pseudo-spectral landscapes represent the **geometry of neural network optimization** through **spectral analysis** (studying eigenvalues/eigenvectors of the Hessian matrix).

### Core Mathematical Foundation:

**The Hessian Matrix:**
```
H[i,j] = ∂²L / ∂θᵢ∂θⱼ
```

**Eigenvalue Spectrum:** {λ₁, λ₂, ..., λₙ}
- **λ > 0**: Upward curvature (local minimum)  
- **λ < 0**: Downward curvature (saddle point)
- **|λ| ≈ 0**: Flat direction

**Taylor Expansion:**
```
L(θ + δ) ≈ L(θ) + ∇L·δ + ½δᵀHδ
```

The Hessian term reveals local geometry!

## 🧠 Why This Matters for LLMs

### 1. **Scale Problem**
- LLMs: billions of parameters → Hessian computation impossible
- Solution: Spectral approximations (power iteration, stochastic sampling)

### 2. **Saddle Points Everywhere**
- High dimensions → almost all critical points are saddles
- Negative eigenvalues = directions to decrease loss
- SGD's noise helps escape saddles

### 3. **Flat Minima Generalize Better**
- Flatness = small eigenvalues
- More robust to perturbations
- Motivates SAM (Sharpness-Aware Minimization)

### 4. **Low-Rank Structure**
- Attention weights develop low-rank structure
- Enables LoRA (Low-Rank Adaptation)
- Fine-tune with far fewer parameters!

## 📊 Visualizations Explained

### loss_landscape_analysis.png (8 panels)
1. **2D Contour**: Slice through high-D landscape
2. **3D Surface**: Visual representation of loss terrain
3. **Eigenvalue Spectrum**: THE KEY PLOT - shows all eigenvalues
4. **Eigenvalue Histogram**: Distribution of curvatures
5. **Hessian Heatmap**: Full curvature matrix
6. **Principal Curvatures**: Loss along eigenvector directions
7. **Sorted Spectrum**: Decay pattern (log scale)
8. **Training History**: Evolution of loss

**Key Insight**: Most eigenvalues cluster near zero → landscape is mostly flat

### spectral_evolution.png (4 panels)
1. **Loss Curve**: Standard training progress
2. **Extreme Eigenvalues**: λ_max and λ_min evolution
3. **Hessian Trace**: Total curvature over time
4. **Sharpness Ratio**: Conditioning metric

**Key Insight**: Networks evolve toward flatter regions

### attention_geometry.png (11 panels)
Focuses on transformer attention mechanism:
- Attention weight matrices
- Q, K, V, O singular value decompositions
- Per-parameter curvature
- Attention entropy and rank analysis

**Key Insight**: Low-rank structure emerges naturally

### attention_training_dynamics.png (6 panels)
Shows how spectral properties evolve:
- Singular value trajectories
- Condition number evolution
- Effective rank changes

**Key Insight**: Training finds low-dimensional subspaces

## 📈 Key Metrics

### Condition Number
```
κ = λ_max / λ_min
```
- κ < 10: Easy optimization
- κ > 100: Difficult optimization
- High κ → need small learning rates

### Flatness Score
```
flatness = 1 / (1 + λ_max)
```
Higher → better generalization

### Effective Rank
```
rank_eff = (Σσᵢ)² / Σσᵢ²
```
Intrinsic dimensionality of transformations

## 💡 Practical Applications

### 1. Optimizer Selection
- Adam/RMSprop: Adapt to local curvature
- SAM: Explicitly seeks flat minima
- Learning rate ∝ 1/λ_max

### 2. Architecture Design
- Skip connections → smoother landscapes
- Layer normalization → better conditioning
- Attention patterns affect geometry

### 3. Fine-Tuning Strategies
- LoRA exploits low-rank structure
- Target parameters with high curvature
- Preserve flat directions

### 4. Debugging Training
- Loss spikes → high curvature regions
- Slow convergence → high condition number
- Oscillations → eigenvalue mismatch with learning rate

## 🔬 Research Connections

### Foundational Papers
- Li et al. "Visualizing Loss Landscapes" (NeurIPS 2018)
- Sagun et al. "Hessian Analysis" (ICLR 2018)
- Dinh et al. "Sharp Minima" (ICLR 2017)

### Modern Methods
- **SAM**: Sharpness-Aware Minimization
- **LoRA**: Low-Rank Adaptation
- **NTK**: Neural Tangent Kernel theory

## 🛠️ Code Examples

### Analyze Your Network
```python
from pseudo_spectral_demo import analyze_hessian_spectrum

eigenvalues, eigenvectors, H = analyze_hessian_spectrum(nn, X, y)
print(f"Condition number: {np.max(eigenvalues)/np.min(eigenvalues)}")
```

### Study Mode Connectivity
```python
# Train two networks
nn1, nn2 = train_from_different_inits()

# Interpolate
for alpha in np.linspace(0, 1, 100):
    params = (1-alpha)*params1 + alpha*params2
    loss = compute_loss(params)
```

### Fast Diagonal Approximation
```python
def approx_hessian_diag(model, X, y, eps=1e-4):
    params = model.get_params_vector()
    diag = []
    for i in range(len(params)):
        # Finite differences for second derivative
        params[i] += eps
        l_plus = model.loss(X, y)
        params[i] -= 2*eps
        l_minus = model.loss(X, y)
        params[i] += eps
        l_center = model.loss(X, y)
        diag.append((l_plus - 2*l_center + l_minus) / eps**2)
    return np.array(diag)
```

## 🎓 Key Takeaways

1. **Hessian eigenvalues are the "fingerprint" of optimization geometry**
2. **Flat minima (small λ) → better generalization**
3. **Saddle points dominate high-dimensional spaces**
4. **Low-rank structure enables efficient fine-tuning**
5. **Spectral methods scale to billion-parameter models**

## 🔮 Future Directions

- Adaptive learning rates from spectral info
- Architecture search for better landscapes
- Geometry-aware data ordering
- Spectral preconditioning for transformers

## 📚 Further Reading

**Books:**
- Goodfellow et al., "Deep Learning" (2016)
- Nocedal & Wright, "Numerical Optimization" (2006)

**Papers:**
- Visualizing Loss Landscapes (Li et al., 2018)
- LoRA (Hu et al., 2022)
- SAM (Foret et al., 2021)

---

**Start exploring:**
```bash
python pseudo_spectral_demo.py
python attention_geometry_demo.py
```

The geometric perspective is fundamental to understanding modern AI!
