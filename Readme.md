# Manifold Representations

A comprehensive framework for analyzing how different training objectives shape the geometric properties of learned representations in neural networks. This project trains models on MNIST using four different objectives (supervised, autoencoder, contrastive, and random labels) and analyzes the resulting manifold structures.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Training Models](#training-models)
- [Extracting Representations](#extracting-representations)
- [Running Analysis](#running-analysis)
- [Visualizing Results](#visualizing-results)
- [Configuration](#configuration)
- [Results Interpretation](#results-interpretation)

## Overview

This project investigates how different learning objectives affect the geometric structure of learned representations:

- **Supervised Learning**: Standard classification with cross-entropy loss
- **Autoencoder**: Reconstruction-based learning with MSE loss
- **Contrastive Learning**: Self-supervised learning with InfoNCE loss
- **Random Labels**: Supervised learning with randomly assigned labels (baseline)

The analysis includes:
- **Intrinsic Dimension**: Effective dimensionality using Participation Ratio (PR)
- **Neighborhood Stability**: Consistency of local structure via kNN overlap
- **Class Geometry**: Intra/inter-class variance and silhouette scores
- **Perturbation Analysis**: Robustness to input perturbations
- **Dynamics Over Time**: Evolution of representations during training

## Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended) or CPU
- pip or conda

### Setup

1. Clone the repository:
```bash
git clone <https://github.com/raghulchandramouli/Manifold-representations.git>
cd Manifold-representations
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. The MNIST dataset will be automatically downloaded on first use.

## Project Structure

```
Manifold-representations/
├── config/
│   └── config.yaml          # Main configuration file
├── data/
│   ├── dataset.py           # Dataset loading logic
│   ├── augmentations.py     # Data augmentation for contrastive learning
│   └── MNIST/              # MNIST data (auto-downloaded)
├── models/
│   ├── mlp.py              # MLP architectures
│   ├── autoencoder.py      # Autoencoder model
│   └── projection_head.py  # Projection head for contrastive learning
├── train/
│   ├── run.py              # Main training entry point
│   ├── base_trainer.py     # Base trainer class
│   ├── supervised_trainer.py
│   ├── autoencoder_trainer.py
│   ├── contrasive_trainer.py
│   └── random_label_trainer.py
├── analysis/
│   ├── extract_representation.py  # Extract representations from checkpoints
│   ├── intrinsic_dimension.py     # Compute intrinsic dimension
│   ├── neighbourhood_stability.py # Compute neighborhood overlap
│   ├── class_geomentry.py         # Compute class geometry metrics
│   ├── perturbation_analysis.py   # Analyze perturbation robustness
│   └── dynamics_over_time.py      # Analyze training dynamics
├── nbs/
│   ├── playbook.ipynb      # Visualization of learned manifolds
│   └── geometry.ipynb      # Visualization of geometric metrics
├── utils/
│   ├── config.py           # Configuration loading utilities
│   └── seed.py             # Random seed management
├── checkpoint/             # Saved model checkpoints (gitignored)
├── representation/         # Extracted representations (gitignored)
└── analysis/results/       # Analysis results (gitignored)
```

## Quick Start

### 1. Train a Model

Train a supervised model:
```bash
python train/run.py --objective supervised --config config/config.yaml
```

### 2. Extract Representations

Extract representations from a checkpoint:
```bash
python analysis/extract_representation.py --objective supervised --checkpoint checkpoint/supervised/final.pt --config config/config.yaml
```

### 3. Run Analysis

Compute intrinsic dimension:
```bash
python analysis/intrinsic_dimension.py
```

### 4. Visualize Results

Open the Jupyter notebooks:
```bash
jupyter notebook nbs/playbook.ipynb
jupyter notebook nbs/geometry.ipynb
```

## Training Models

The project supports four training objectives. All models use the same MLP architecture (784 → 512 → 256 → 128) for fair comparison.

### Supervised Learning

Standard classification with cross-entropy loss:
```bash
python train/run.py --objective supervised --config config/config.yaml
```

### Autoencoder

Reconstruction-based learning:
```bash
python train/run.py --objective autoencoder --config config/config.yaml
```

### Contrastive Learning

Self-supervised learning with InfoNCE loss:
```bash
python train/run.py --objective contrastive --config config/config.yaml
```

### Random Labels

Baseline with randomly assigned labels:
```bash
python train/run.py --objective random_labels --config config/config.yaml
```

### Training Details

- **Checkpoints**: Models are saved at steps 1000, 10000, and 50000 (final)
- **Logging**: Training metrics are logged every 500 steps
- **Device**: Automatically uses CUDA if available, falls back to CPU
- **Determinism**: Training is deterministic (seed=42) for reproducibility

## Extracting Representations

After training, extract representations from checkpoints:

```bash
python analysis/extract_representation.py \
    --objective supervised \
    --checkpoint checkpoint/supervised/step_50000.pt \
    --config config/config.yaml
```

This saves:
- `embeddings.npy`: Final layer representations
- `layer_0.npy`, `layer_1.npy`, `layer_2.npy`: Intermediate layer representations
- `labels.npy`: Class labels (when available)

**Note**: For autoencoder and contrastive objectives, `labels.npy` may contain image data or augmented images rather than class labels. The visualization notebooks handle this by loading true MNIST labels when needed.

## Running Analysis

All analysis scripts read from `representation/mnist/` and save results to `analysis/results/`.

### Intrinsic Dimension

Computes Participation Ratio (PR) for each layer:
```bash
python analysis/intrinsic_dimension.py
```

Output: `analysis/results/intrinsic_dimension/intrinsic_dimension.json`

### Neighborhood Stability

Computes kNN overlap between input space and representation space:
```bash
python analysis/neighbourhood_stability.py
```

Output: `analysis/results/neighborhood_stability/neighborhood_overlap.json`

### Class Geometry

Computes intra-class variance, inter-class distance, and silhouette scores:
```bash
python analysis/class_geomentry.py
```

Output: `analysis/results/class_geometry/class_geometry.json`

### Perturbation Analysis

Analyzes robustness to input perturbations:
```bash
python analysis/perturbation_analysis.py
```

Output: `analysis/results/perturbation/perturbation_drift.json`

### Dynamics Over Time

Analyzes how representations evolve during training:
```bash
python analysis/dynamics_over_time.py
```

Output: `analysis/results/dynamics_over_time/dynamics_summary.json`

## Visualizing Results

### Playbook Notebook (`nbs/playbook.ipynb`)

Visualizes learned manifolds using PCA:
- 2D scatter plots colored by digit class
- Side-by-side comparisons of different objectives
- Quantitative metrics (Adjusted Rand Index, Silhouette Score)

### Geometry Notebook (`nbs/geometry.ipynb`)

Visualizes geometric analysis results:
- Intrinsic dimension bar charts
- Neighborhood stability comparisons
- Class geometry metrics

## Configuration

The main configuration file is `config/config.yaml`. Key settings:

### Experiment Settings
```yaml
experiment:
  name: mnist-manifold-geometry
  seed: 42
  device: cuda
  deterministic: true
```

### Data Settings
```yaml
data:
  dataset: MNIST
  root: ./data
  input_dim: 784
  num_classes: 10
  subset_size: 10000
  shuffle: true
  num_workers: 1
```

### Model Architecture
```yaml
model:
  type: mlp
  input_dim: 784
  hidden_dim: [512, 256, 128]
  output_dim: 10
  use_layer_norm: false
```

### Training Settings
```yaml
training:
  batch_size: 32
  max_steps: 50000
  log_every: 500
  checkpoint_steps: [1000, 10000, 50000]
```

### Objective-Specific Settings
```yaml
objective:
  type: supervised  # Change to: supervised, autoencoder, contrastive, random_labels
  
  contrastive:
    temperature: 0.5
    projection_dim: 64
    gaussian_noise_std: 0.1
    random_crop: true
```

## Results Interpretation

### Intrinsic Dimension (Participation Ratio)

- **Higher PR**: More dimensions are actively used
- **Lower PR**: Representations are more compressed/low-dimensional
- **Expected**: Supervised > Contrastive > Autoencoder > Random Labels

### Neighborhood Stability

- **Higher overlap**: Local structure is preserved from input to representation
- **Lower overlap**: Representations reorganize the data significantly
- **Expected**: Autoencoder > Supervised > Contrastive > Random Labels

### Class Geometry

- **Intra-class variance**: How spread out samples within a class are
- **Inter-class distance**: How separated different classes are
- **Silhouette score**: Overall class separation quality (-1 to 1, higher is better)
- **Expected**: Supervised should have best class separation

### Key Insights

1. **Supervised learning** creates well-separated class clusters with high intrinsic dimension
2. **Contrastive learning** creates semantically meaningful representations without explicit labels
3. **Autoencoder** preserves local structure but may not separate classes well
4. **Random labels** serves as a baseline showing what happens without meaningful supervision

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce `batch_size` in `config.yaml`
2. **FileNotFoundError in notebooks**: Ensure you're running notebooks from the project root or adjust paths with `../`
3. **Labels shape mismatch**: For autoencoder/contrastive, `labels.npy` may contain image data. The notebooks handle this automatically.
4. **Empty plots**: Some analysis results only contain data for `step_50000`. The notebooks use bar charts for single data points.

## Contributing

When contributing:
- Follow the existing code structure
- Add docstrings to new functions
- Update this README if adding new features
- Ensure all analysis scripts save results to `analysis/results/`

## License

[Add your license here]

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{manifold-representations,
  title={Manifold Representations: Analyzing Geometric Properties of Learned Representations},
  author={Your Name},
  year={2024}
}
```

