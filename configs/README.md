# Configuration Files

This directory contains YAML configuration files for different use cases.

## Available Configurations

### `default.yaml`
General-purpose configuration suitable for most problems.
- Moderate feature count (10-100 features)
- MCMC inference for accurate posteriors
- Beta-based feature selection

### `sparse_highdim.yaml`
Optimized for high-dimensional sparse problems.
- Many features (>100), few relevant (<10)
- SVI inference for speed
- Lambda-based selection to filter pure noise
- Lower global scale for strong sparsity

## Usage

```bash
# Specify data in config file
bayesian-feature-selection -c configs/example_with_data.yaml

# Mix of config and CLI arguments
bayesian-feature-selection -c configs/sparse_highdim.yaml --family binomial
```

## Creating Custom Configs

Copy and modify an existing config:

```bash
cp configs/default.yaml configs/my_experiment.yaml
# Edit my_experiment.yaml
bayesian-feature-selection -c configs/my_experiment.yaml
```

## Configuration Structure

```yaml
data:
  data_path: "data.csv"   # Path to CSV file (required)
  target_col: "target"    # Target column name (required)
  feature_cols: null      # Specific features (null = all except target)
  test_size: 0.0          # Train/test split fraction (0 = no split)
  standardize: false      # Standardize features to mean=0, std=1
  random_seed: 42         # Random seed for reproducibility

model:
  family: "gaussian"      # gaussian, binomial, poisson
  scale_global: 1.0       # Global shrinkage (tau scale)

inference:
  method: "mcmc"          # mcmc or svi
  num_warmup: 1000        # MCMC warmup iterations
  num_samples: 2000       # MCMC posterior samples
  num_chains: 4           # MCMC chains
  num_steps: 10000        # SVI optimization steps
  learning_rate: 0.001    # SVI learning rate
  use_gpu: true           # Use GPU acceleration
  progress_bar: true      # Show progress bar

selection:
  method: "beta"          # beta, lambda, or both
  threshold: 0.5          # Inclusion probability threshold

output:
  save_plots: true        # Generate feature importance plots
  save_diagnostics: true  # Generate MCMC diagnostic plots
  save_samples: false     # Save posterior samples (large files)
```

## Parameter Guidelines

### `data.standardize`
- **true**: Recommended for high-dimensional data or when features have different scales
- **false**: Use when features already normalized or for interpretability

### `data.test_size`
- **0.0**: No split, use all data for training (default)
- **0.2-0.3**: Typical test set size for model evaluation

### `scale_global`
- **Sparse** (few relevant features): 0.1 - 0.5
- **Moderate**: 0.5 - 1.0 (default)
- **Dense** (many relevant features): 1.0 - 2.0
- Formula: `p0 / (p - p0) / sqrt(n)` where p0 = expected relevant features

### `selection.method`
- **beta**: Best for effect size and prediction
- **lambda**: Best for filtering pure noise, keeping weak signals
- **both**: Most conservative, combines both metrics

### `selection.threshold`
- Higher (0.7-0.9): More conservative, fewer false positives
- Default (0.5): Balanced
- Lower (0.3-0.4): More inclusive, fewer false negatives
