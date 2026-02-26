"""
Example: Using DataLoader for custom workflows
"""

from bayesian_feature_selection import (
    HorseshoeGLM,
    DataLoader,
    DataConfig,
    InferenceConfig,
    ExperimentConfig
)

# Example 1: Simple data loading
print("=" * 60)
print("Example 1: Simple data loading")
print("=" * 60)

data_config = DataConfig(
    data_path="data/my_data.csv",
    target_col="target",
    standardize=True
)

loader = DataLoader(data_config)
X, y, feature_names = loader.load_data()

print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
print(f"Features: {feature_names[:5]}...")

# Example 2: Train/test split
print("\n" + "=" * 60)
print("Example 2: Train/test split")
print("=" * 60)

data_config_split = DataConfig(
    data_path="data/my_data.csv",
    target_col="target",
    test_size=0.2,
    standardize=True,
    random_seed=42
)

loader2 = DataLoader(data_config_split)
X_train, X_test, y_train, y_test, feature_names = loader2.load_and_split()

print(f"Train: {X_train.shape[0]} samples")
print(f"Test: {X_test.shape[0]} samples")

# Fit model on training data
model = HorseshoeGLM(family="gaussian")
model.fit(X_train, y_train, InferenceConfig(num_samples=500))

# Evaluate on test set
if len(X_test) > 0:
    predictions = model.predict(X_test)
    print(f"Made {len(predictions)} predictions on test set")

# Example 3: Using full ExperimentConfig
print("\n" + "=" * 60)
print("Example 3: Complete workflow with config file")
print("=" * 60)

# Load config from YAML
config = ExperimentConfig.from_yaml("configs/example_with_data.yaml")

# Load data
loader3 = DataLoader(config.data)
X, y, feature_names = loader3.load_data()

# Fit model
model = HorseshoeGLM(
    family=config.model.family,
    scale_global=config.model.scale_global
)
model.fit(X, y, config.inference)

# Get feature importance
importance = model.get_feature_importance(
    threshold=config.selection.threshold,
    method=config.selection.method
)

print(f"Selected {importance['selected'].sum()} out of {len(importance)} features")
print("\nTop 5 features:")
print(importance.head()[['feature_idx', 'beta_mean', 'beta_inclusion_prob']])

# Save predictions
predictions = model.predict(X)
loader3.save_predictions(
    predictions,
    output_path="results/predictions.csv",
    include_features=True,
    X=X
)
print("\nPredictions saved to results/predictions.csv")

# Example 4: Custom feature selection
print("\n" + "=" * 60)
print("Example 4: Using specific features")
print("=" * 60)

data_config_subset = DataConfig(
    data_path="data/my_data.csv",
    target_col="target",
    feature_cols=["feature_1", "feature_2", "feature_3"],  # Only these features
    standardize=True
)

loader4 = DataLoader(data_config_subset)
X_subset, y_subset, feature_names_subset = loader4.load_data()

print(f"Using only {len(feature_names_subset)} features: {feature_names_subset}")
