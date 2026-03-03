"""Integration tests for end-to-end bayesian_feature_selection workflows."""

import numpy as np
import pandas as pd
import pytest
import warnings

import numpyro

numpyro.set_platform("cpu")

from bayesian_feature_selection import (
    HorseshoeGLM,
    DataLoader,
    InferenceConfig,
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    SelectionConfig,
    OutputConfig,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TINY_MCMC_CONFIG = InferenceConfig(
    method="mcmc",
    num_warmup=10,
    num_samples=20,
    num_chains=1,
    use_gpu=False,
    progress_bar=False,
)

TINY_SVI_CONFIG = InferenceConfig(
    method="svi",
    num_steps=100,
    use_gpu=False,
    progress_bar=False,
)


def _make_regression_csv(path, n_samples=20, n_features=5, n_relevant=2, seed=42):
    """Create synthetic regression data and save to CSV."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features)
    beta_true = np.zeros(n_features)
    beta_true[:n_relevant] = rng.uniform(1, 3, size=n_relevant)
    y = X @ beta_true + rng.randn(n_samples) * 0.1
    cols = [f"feat_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=cols)
    df["target"] = y
    df.to_csv(path, index=False)
    return cols


def _make_classification_csv(path, n_samples=20, n_features=5, seed=42):
    """Create synthetic binary classification data and save to CSV."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features)
    logits = X[:, 0] * 1.5 - X[:, 1] * 1.0
    y = (logits > 0).astype(float)
    cols = [f"feat_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=cols)
    df["target"] = y
    df.to_csv(path, index=False)
    return cols


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_full_gaussian_mcmc_pipeline(tmp_path):
    """End-to-end Gaussian regression with MCMC inference."""
    csv_path = tmp_path / "regression.csv"
    feat_cols = _make_regression_csv(csv_path, n_samples=20, n_features=5, n_relevant=2)

    # Load data
    data_cfg = DataConfig(data_path=str(csv_path), target_col="target")
    loader = DataLoader(data_cfg)
    X, y, feature_names = loader.load_data()

    assert X.shape == (20, 5)
    assert y.shape == (20,)
    assert len(feature_names) == 5

    # Fit model
    model = HorseshoeGLM(family="gaussian", scale_global=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y, config=TINY_MCMC_CONFIG)

    # Feature importance
    importance = model.get_feature_importance(threshold=0.5, method="beta")
    assert isinstance(importance, pd.DataFrame)
    assert len(importance) == 5

    # Predictions
    predictions = model.predict(X)
    assert predictions.shape == (20,)
    assert np.all(np.isfinite(predictions))


@pytest.mark.integration
def test_full_binomial_svi_pipeline(tmp_path):
    """End-to-end binary classification with SVI inference."""
    csv_path = tmp_path / "classification.csv"
    _make_classification_csv(csv_path, n_samples=20, n_features=3)

    # Load with standardization
    data_cfg = DataConfig(
        data_path=str(csv_path), target_col="target", standardize=True
    )
    loader = DataLoader(data_cfg)
    X, y, feature_names = loader.load_data()

    assert X.shape == (20, 3)
    assert set(np.unique(y)).issubset({0.0, 1.0})

    # Fit binomial model with SVI
    model = HorseshoeGLM(family="binomial", scale_global=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y, config=TINY_SVI_CONFIG)

    # Feature importance
    importance = model.get_feature_importance(threshold=0.5, method="beta")
    assert isinstance(importance, pd.DataFrame)
    assert len(importance) == 3

    # Predictions should be finite
    predictions = model.predict(X)
    assert predictions.shape == (20,)
    assert np.all(np.isfinite(predictions))


@pytest.mark.integration
def test_config_driven_workflow(tmp_path):
    """Full workflow driven by YAML ExperimentConfig."""
    # Create data
    csv_path = tmp_path / "data.csv"
    _make_regression_csv(csv_path, n_samples=20, n_features=4, n_relevant=2)

    # Write YAML config
    yaml_path = tmp_path / "config.yaml"
    yaml_content = f"""\
data:
  data_path: "{csv_path}"
  target_col: "target"
  test_size: 0.0
  standardize: false
  random_seed: 42
model:
  family: "gaussian"
  scale_global: 1.0
inference:
  method: "mcmc"
  num_warmup: 10
  num_samples: 20
  num_chains: 1
  num_steps: 100
  learning_rate: 0.001
  use_gpu: false
  progress_bar: false
selection:
  method: "beta"
  threshold: 0.5
output:
  save_plots: false
  save_diagnostics: false
  save_samples: false
"""
    yaml_path.write_text(yaml_content)

    # Load config
    exp_cfg = ExperimentConfig.from_yaml(yaml_path)
    assert exp_cfg.model.family == "gaussian"
    assert exp_cfg.inference.num_warmup == 10

    # Drive workflow from config
    loader = DataLoader(exp_cfg.data)
    X, y, feature_names = loader.load_data()

    model = HorseshoeGLM(
        family=exp_cfg.model.family, scale_global=exp_cfg.model.scale_global
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y, config=exp_cfg.inference)

    importance = model.get_feature_importance(
        threshold=exp_cfg.selection.threshold, method=exp_cfg.selection.method
    )
    assert isinstance(importance, pd.DataFrame)
    assert len(importance) == 4

    predictions = model.predict(X)
    assert predictions.shape == (20,)


@pytest.mark.integration
def test_train_test_split_workflow(tmp_path):
    """Train/test split, fit on train, predict on test, save predictions."""
    csv_path = tmp_path / "split_data.csv"
    _make_regression_csv(csv_path, n_samples=20, n_features=3, n_relevant=1)

    data_cfg = DataConfig(
        data_path=str(csv_path),
        target_col="target",
        test_size=0.3,
        random_seed=42,
    )
    loader = DataLoader(data_cfg)
    X_train, X_test, y_train, y_test, feature_names = loader.load_and_split()

    assert X_train.shape[0] + X_test.shape[0] == 20
    assert X_train.shape[1] == 3
    assert X_test.shape[1] == 3

    # Fit on training data
    model = HorseshoeGLM(family="gaussian", scale_global=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train, config=TINY_MCMC_CONFIG)

    # Predict on test data
    predictions = model.predict(X_test)
    assert predictions.shape == (X_test.shape[0],)
    assert np.all(np.isfinite(predictions))

    # Save predictions
    output_path = tmp_path / "predictions.csv"
    loader.save_predictions(predictions, output_path)

    # Verify saved file
    saved = pd.read_csv(output_path)
    assert "prediction" in saved.columns
    assert len(saved) == X_test.shape[0]


@pytest.mark.integration
def test_feature_selection_consistency(tmp_path):
    """All feature importance methods return valid DataFrames with same length."""
    csv_path = tmp_path / "consistency.csv"
    n_features = 4
    _make_regression_csv(csv_path, n_samples=20, n_features=n_features, n_relevant=2)

    data_cfg = DataConfig(data_path=str(csv_path), target_col="target")
    loader = DataLoader(data_cfg)
    X, y, feature_names = loader.load_data()

    model = HorseshoeGLM(family="gaussian", scale_global=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y, config=TINY_MCMC_CONFIG)

    for method in ("beta", "lambda", "both"):
        importance = model.get_feature_importance(threshold=0.5, method=method)
        assert isinstance(importance, pd.DataFrame), f"method={method} not a DataFrame"
        assert len(importance) == n_features, (
            f"method={method}: expected {n_features} rows, got {len(importance)}"
        )
