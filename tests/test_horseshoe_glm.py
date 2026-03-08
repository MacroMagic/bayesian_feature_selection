"""Unit tests for HorseshoeGLM class."""

import numpy as np
import pandas as pd
import pytest
import warnings

import numpyro

numpyro.set_platform("cpu")

from bayesian_feature_selection import HorseshoeGLM, InferenceConfig  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_SAMPLES = 20
N_FEATURES = 3


@pytest.fixture
def synthetic_data():
    """Small synthetic regression dataset."""
    rng = np.random.RandomState(42)
    X = rng.randn(N_SAMPLES, N_FEATURES)
    beta_true = np.array([1.0, 0.0, -0.5])
    y = X @ beta_true + rng.randn(N_SAMPLES) * 0.1
    return X, y


@pytest.fixture
def mcmc_config():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return InferenceConfig(
            method="mcmc",
            num_warmup=10,
            num_samples=20,
            num_chains=1,
            use_gpu=False,
            progress_bar=False,
        )


@pytest.fixture
def svi_config():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return InferenceConfig(
            method="svi",
            num_steps=100,
            use_gpu=False,
            progress_bar=False,
        )


@pytest.fixture
def fitted_mcmc_model(synthetic_data, mcmc_config):
    """Return a HorseshoeGLM already fitted with MCMC."""
    X, y = synthetic_data
    model = HorseshoeGLM(family="gaussian", scale_global=1.0)
    model.fit(X, y, config=mcmc_config)
    return model


# ---------------------------------------------------------------------------
# 1. Initialization
# ---------------------------------------------------------------------------


class TestInit:
    def test_default_init(self):
        model = HorseshoeGLM()
        assert model.family == "gaussian"
        assert model.scale_global == 1.0
        assert model.mcmc is None
        assert model.svi_result is None
        assert model.X_train is None
        assert model.y_train is None

    def test_custom_init(self):
        model = HorseshoeGLM(family="binomial", scale_global=0.5)
        assert model.family == "binomial"
        assert model.scale_global == 0.5

    def test_poisson_init(self):
        model = HorseshoeGLM(family="poisson", scale_global=2.0)
        assert model.family == "poisson"
        assert model.scale_global == 2.0


# ---------------------------------------------------------------------------
# 2 & 4. Fit with MCMC
# ---------------------------------------------------------------------------


class TestFitMCMC:
    def test_mcmc_sets_mcmc(self, synthetic_data, mcmc_config):
        X, y = synthetic_data
        model = HorseshoeGLM()
        model.fit(X, y, config=mcmc_config)
        assert model.mcmc is not None

    def test_mcmc_stores_training_data(self, synthetic_data, mcmc_config):
        X, y = synthetic_data
        model = HorseshoeGLM()
        model.fit(X, y, config=mcmc_config)
        assert model.X_train is not None
        assert model.y_train is not None

    def test_fit_returns_self(self, synthetic_data, mcmc_config):
        X, y = synthetic_data
        model = HorseshoeGLM()
        result = model.fit(X, y, config=mcmc_config)
        assert result is model


# ---------------------------------------------------------------------------
# 3. Fit with SVI
# ---------------------------------------------------------------------------


class TestFitSVI:
    def test_svi_sets_svi_result(self, synthetic_data, svi_config):
        X, y = synthetic_data
        model = HorseshoeGLM()
        model.fit(X, y, config=svi_config)
        assert model.svi_result is not None

    def test_svi_returns_self(self, synthetic_data, svi_config):
        X, y = synthetic_data
        model = HorseshoeGLM()
        result = model.fit(X, y, config=svi_config)
        assert result is model


# ---------------------------------------------------------------------------
# 5. Fit with invalid method
# ---------------------------------------------------------------------------


class TestFitInvalidMethod:
    def test_invalid_method_raises(self, synthetic_data):
        X, y = synthetic_data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            config = InferenceConfig(
                method="mcmc",
                num_warmup=10,
                num_samples=20,
                num_chains=1,
                use_gpu=False,
                progress_bar=False,
            )
        # Bypass dataclass validation by setting after init
        object.__setattr__(config, "method", "invalid")

        model = HorseshoeGLM()
        with pytest.raises(ValueError, match="Unknown method"):
            model.fit(X, y, config=config)


# ---------------------------------------------------------------------------
# 6. get_feature_importance before fit
# ---------------------------------------------------------------------------


class TestFeatureImportanceBeforeFit:
    def test_raises_before_fit(self):
        model = HorseshoeGLM()
        with pytest.raises(ValueError, match="Model must be fitted first"):
            model.get_feature_importance()


# ---------------------------------------------------------------------------
# 7. get_feature_importance after MCMC fit
# ---------------------------------------------------------------------------

EXPECTED_COLS_BETA = [
    "feature_idx",
    "beta_mean",
    "beta_std",
    "beta_lower_95",
    "beta_upper_95",
    "beta_inclusion_prob",
    "ci_excludes_zero",
    "lambda_mean",
    "lambda_median",
    "lambda_inclusion_prob",
    "selected",
]


class TestFeatureImportanceMCMC:
    def test_beta_method_columns(self, fitted_mcmc_model):
        df = fitted_mcmc_model.get_feature_importance(method="beta")
        assert isinstance(df, pd.DataFrame)
        for col in EXPECTED_COLS_BETA:
            assert col in df.columns, f"Missing column: {col}"

    def test_beta_method_rows(self, fitted_mcmc_model):
        df = fitted_mcmc_model.get_feature_importance(method="beta")
        assert len(df) == N_FEATURES

    def test_lambda_method_columns(self, fitted_mcmc_model):
        df = fitted_mcmc_model.get_feature_importance(method="lambda")
        for col in EXPECTED_COLS_BETA:
            assert col in df.columns

    def test_both_method_has_combined_column(self, fitted_mcmc_model):
        df = fitted_mcmc_model.get_feature_importance(method="both")
        assert "combined_inclusion_prob" in df.columns
        for col in EXPECTED_COLS_BETA:
            assert col in df.columns

    def test_both_method_rows(self, fitted_mcmc_model):
        df = fitted_mcmc_model.get_feature_importance(method="both")
        assert len(df) == N_FEATURES

    def test_custom_threshold(self, fitted_mcmc_model):
        df = fitted_mcmc_model.get_feature_importance(threshold=0.99)
        assert "selected" in df.columns
        # With a very high threshold, likely fewer (or zero) features selected
        assert df["selected"].dtype == bool


# ---------------------------------------------------------------------------
# 8. Predict after MCMC fit
# ---------------------------------------------------------------------------


class TestPredictMCMC:
    def test_predict_shape(self, fitted_mcmc_model, synthetic_data):
        X, _ = synthetic_data
        preds = fitted_mcmc_model.predict(X)
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (N_SAMPLES,)

    def test_predict_return_samples(self, fitted_mcmc_model, synthetic_data):
        X, _ = synthetic_data
        preds = fitted_mcmc_model.predict(X, return_samples=True)
        assert isinstance(preds, np.ndarray)
        # Should be (num_samples, N_SAMPLES) – num_samples=20 from mcmc_config
        assert preds.ndim == 2
        assert preds.shape[1] == N_SAMPLES

    def test_predict_new_data(self, fitted_mcmc_model):
        rng = np.random.RandomState(99)
        X_new = rng.randn(5, N_FEATURES)
        preds = fitted_mcmc_model.predict(X_new)
        assert preds.shape == (5,)


# ---------------------------------------------------------------------------
# 9. Different families
# ---------------------------------------------------------------------------


class TestFamilies:
    def test_binomial_family(self, mcmc_config):
        rng = np.random.RandomState(42)
        X = rng.randn(N_SAMPLES, N_FEATURES)
        y = (X @ np.array([1.0, 0.0, -0.5]) > 0).astype(float)

        model = HorseshoeGLM(family="binomial")
        model.fit(X, y, config=mcmc_config)
        assert model.mcmc is not None

    def test_poisson_family(self, mcmc_config):
        rng = np.random.RandomState(42)
        X = rng.randn(N_SAMPLES, N_FEATURES) * 0.5
        y = rng.poisson(lam=np.exp(X @ np.array([0.5, 0.0, -0.3])))

        model = HorseshoeGLM(family="poisson")
        model.fit(X, y, config=mcmc_config)
        assert model.mcmc is not None
