# Data Loading System

The bayesian_feature_selection package now includes a comprehensive data loading system.

## Key Components

### 1. DataConfig (in `config.py`)
Configuration dataclass for data-related parameters:
- `data_path`: Path to CSV file
- `target_col`: Target column name
- `feature_cols`: Specific features to use (optional)
- `test_size`: Train/test split fraction
- `standardize`: Whether to standardize features
- `random_seed`: Random seed for reproducibility

### 2. DataLoader (in `data_loader.py`)
Handles all data I/O operations:
- `load_data()`: Load data from CSV
- `load_and_split()`: Load and split into train/test
- `save_predictions()`: Save model predictions
- Automatic feature standardization
- Feature name tracking

## Usage Examples

### CLI Usage

```bash
# Data in config file
bayesian-feature-selection -c configs/my_experiment.yaml

# Data in config, override model params
bayesian-feature-selection -c configs/example_with_data.yaml --family binomial
```

### Programmatic Usage

```python
from bayesian_feature_selection import DataLoader, DataConfig, HorseshoeGLM

# Configure data loading
data_config = DataConfig(
    data_path="data/my_data.csv",
    target_col="target",
    standardize=True,
    test_size=0.2
)

# Load and split data
loader = DataLoader(data_config)
X_train, X_test, y_train, y_test, feature_names = loader.load_and_split()

# Fit model
model = HorseshoeGLM()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Save predictions with features
loader.save_predictions(
    predictions,
    output_path="results/predictions.csv",
    include_features=True,
    X=X_test
)
```

### Complete Config File Example

```yaml
data:
  data_path: "data/features.csv"
  target_col: "outcome"
  feature_cols: null  # Use all features
  test_size: 0.2
  standardize: true
  random_seed: 42

model:
  family: "gaussian"
  scale_global: 1.0

inference:
  method: "mcmc"
  num_samples: 2000

selection:
  method: "beta"
  threshold: 0.5

output:
  save_plots: true
  save_diagnostics: true
```

## Benefits

✅ **Separation of concerns** - Data handling separated from modeling
✅ **Reproducibility** - Data preprocessing config saved with results
✅ **Flexibility** - Config file or command line arguments
✅ **Testability** - DataLoader can be tested independently
✅ **Reusability** - Use same data loader for multiple experiments
✅ **Standardization** - Consistent data preprocessing across runs

## Features

- **CSV support**: Read data from CSV files
- **Train/test splitting**: Built-in sklearn integration
- **Feature standardization**: Zero mean, unit variance
- **Feature selection**: Use subset of features
- **Prediction saving**: Export predictions with features
- **Name tracking**: Maintain feature names throughout pipeline

## See Also

- [examples/data_loader_example.py](../examples/data_loader_example.py) - Detailed usage examples
- [configs/README.md](../configs/README.md) - Configuration documentation
- [configs/example_with_data.yaml](../configs/example_with_data.yaml) - Complete example config
