"""Tests for the config module."""

import warnings

import pytest
import yaml

from bayesian_feature_selection.config import (
    DataConfig,
    ExperimentConfig,
    InferenceConfig,
    ModelConfig,
    OutputConfig,
    SelectionConfig,
)


# ---------------------------------------------------------------------------
# DataConfig
# ---------------------------------------------------------------------------

class TestDataConfig:
    def test_defaults(self):
        cfg = DataConfig()
        assert cfg.data_path is None
        assert cfg.target_col is None
        assert cfg.feature_cols is None
        assert cfg.test_size == 0.0
        assert cfg.standardize is False
        assert cfg.random_seed == 42

    def test_custom_values(self):
        cfg = DataConfig(
            data_path="/data/file.csv",
            target_col="y",
            feature_cols=["a", "b"],
            test_size=0.2,
            standardize=True,
            random_seed=123,
        )
        assert cfg.data_path == "/data/file.csv"
        assert cfg.target_col == "y"
        assert cfg.feature_cols == ["a", "b"]
        assert cfg.test_size == 0.2
        assert cfg.standardize is True
        assert cfg.random_seed == 123


# ---------------------------------------------------------------------------
# InferenceConfig – defaults & custom
# ---------------------------------------------------------------------------

class TestInferenceConfig:
    def test_defaults(self):
        cfg = InferenceConfig()
        assert cfg.method == "mcmc"
        assert cfg.num_warmup == 1000
        assert cfg.num_samples == 2000
        assert cfg.num_chains == 4
        assert cfg.num_steps == 10000
        assert cfg.learning_rate == 0.001
        assert cfg.use_gpu is True
        assert cfg.progress_bar is True

    def test_custom_values(self):
        cfg = InferenceConfig(
            method="svi",
            num_warmup=500,
            num_samples=1000,
            num_chains=2,
            num_steps=5000,
            learning_rate=0.01,
            use_gpu=False,
            progress_bar=False,
        )
        assert cfg.method == "svi"
        assert cfg.num_warmup == 500
        assert cfg.num_samples == 1000
        assert cfg.num_chains == 2
        assert cfg.num_steps == 5000
        assert cfg.learning_rate == 0.01
        assert cfg.use_gpu is False
        assert cfg.progress_bar is False


# ---------------------------------------------------------------------------
# InferenceConfig – validation errors
# ---------------------------------------------------------------------------

class TestInferenceConfigValidation:
    def test_negative_num_warmup(self):
        with pytest.raises(ValueError, match="num_warmup must be positive"):
            InferenceConfig(num_warmup=-1)

    def test_zero_num_warmup(self):
        with pytest.raises(ValueError, match="num_warmup must be positive"):
            InferenceConfig(num_warmup=0)

    def test_negative_num_samples(self):
        with pytest.raises(ValueError, match="num_samples must be positive"):
            InferenceConfig(num_samples=-1)

    def test_zero_num_samples(self):
        with pytest.raises(ValueError, match="num_samples must be positive"):
            InferenceConfig(num_samples=0)

    def test_negative_num_chains(self):
        with pytest.raises(ValueError, match="num_chains must be positive"):
            InferenceConfig(num_chains=-1)

    def test_zero_num_chains(self):
        with pytest.raises(ValueError, match="num_chains must be positive"):
            InferenceConfig(num_chains=0)

    def test_negative_num_steps(self):
        with pytest.raises(ValueError, match="num_steps must be positive"):
            InferenceConfig(num_steps=-1)

    def test_zero_num_steps(self):
        with pytest.raises(ValueError, match="num_steps must be positive"):
            InferenceConfig(num_steps=0)

    def test_learning_rate_zero(self):
        with pytest.raises(ValueError, match="learning_rate must be in"):
            InferenceConfig(learning_rate=0.0)

    def test_learning_rate_one(self):
        with pytest.raises(ValueError, match="learning_rate must be in"):
            InferenceConfig(learning_rate=1.0)

    def test_learning_rate_negative(self):
        with pytest.raises(ValueError, match="learning_rate must be in"):
            InferenceConfig(learning_rate=-0.1)

    def test_mcmc_samples_less_than_warmup(self):
        with pytest.raises(ValueError, match="num_samples .* should be >= .*num_warmup"):
            InferenceConfig(method="mcmc", num_warmup=2000, num_samples=1000)


# ---------------------------------------------------------------------------
# InferenceConfig – warnings
# ---------------------------------------------------------------------------

class TestInferenceConfigWarnings:
    def test_high_chains_warning(self):
        with pytest.warns(UserWarning, match="num_chains=.*is high"):
            InferenceConfig(num_chains=11)

    def test_high_samples_warning(self):
        with pytest.warns(UserWarning, match="num_samples=.*is very high"):
            InferenceConfig(num_samples=10001)

    def test_svi_low_steps_warning(self):
        with pytest.warns(UserWarning, match="num_steps=.*may be too low"):
            InferenceConfig(method="svi", num_steps=500)

    def test_svi_high_learning_rate_warning(self):
        with pytest.warns(UserWarning, match="learning_rate=.*is quite high"):
            InferenceConfig(method="svi", learning_rate=0.2)

    def test_no_warning_for_normal_mcmc(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            InferenceConfig(method="mcmc", num_chains=4, num_samples=2000)

    def test_no_warning_for_normal_svi(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            InferenceConfig(method="svi", num_steps=5000, learning_rate=0.01)


# ---------------------------------------------------------------------------
# ModelConfig
# ---------------------------------------------------------------------------

class TestModelConfig:
    def test_defaults(self):
        cfg = ModelConfig()
        assert cfg.family == "gaussian"
        assert cfg.scale_global == 1.0

    def test_custom_values(self):
        cfg = ModelConfig(family="binomial", scale_global=0.5)
        assert cfg.family == "binomial"
        assert cfg.scale_global == 0.5


# ---------------------------------------------------------------------------
# SelectionConfig
# ---------------------------------------------------------------------------

class TestSelectionConfig:
    def test_defaults(self):
        cfg = SelectionConfig()
        assert cfg.method == "beta"
        assert cfg.threshold == 0.5

    def test_custom_values(self):
        cfg = SelectionConfig(method="both", threshold=0.8)
        assert cfg.method == "both"
        assert cfg.threshold == 0.8


# ---------------------------------------------------------------------------
# OutputConfig
# ---------------------------------------------------------------------------

class TestOutputConfig:
    def test_defaults(self):
        cfg = OutputConfig()
        assert cfg.save_plots is True
        assert cfg.save_diagnostics is True
        assert cfg.save_samples is False

    def test_custom_values(self):
        cfg = OutputConfig(save_plots=False, save_diagnostics=False, save_samples=True)
        assert cfg.save_plots is False
        assert cfg.save_diagnostics is False
        assert cfg.save_samples is True


# ---------------------------------------------------------------------------
# ExperimentConfig
# ---------------------------------------------------------------------------

class TestExperimentConfig:
    def test_defaults(self):
        cfg = ExperimentConfig()
        assert isinstance(cfg.data, DataConfig)
        assert isinstance(cfg.model, ModelConfig)
        assert isinstance(cfg.inference, InferenceConfig)
        assert isinstance(cfg.selection, SelectionConfig)
        assert isinstance(cfg.output, OutputConfig)

    def test_from_yaml(self, tmp_path):
        yaml_content = {
            "data": {"data_path": "/tmp/data.csv", "target_col": "target"},
            "model": {"family": "poisson", "scale_global": 2.0},
            "inference": {"method": "svi", "num_steps": 5000, "learning_rate": 0.01},
            "selection": {"method": "lambda", "threshold": 0.7},
            "output": {"save_plots": False, "save_samples": True},
        }
        yaml_file = tmp_path / "config.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump(yaml_content, f)

        cfg = ExperimentConfig.from_yaml(yaml_file)
        assert cfg.data.data_path == "/tmp/data.csv"
        assert cfg.data.target_col == "target"
        assert cfg.model.family == "poisson"
        assert cfg.model.scale_global == 2.0
        assert cfg.inference.method == "svi"
        assert cfg.inference.num_steps == 5000
        assert cfg.inference.learning_rate == 0.01
        assert cfg.selection.method == "lambda"
        assert cfg.selection.threshold == 0.7
        assert cfg.output.save_plots is False
        assert cfg.output.save_samples is True

    def test_from_yaml_empty_file(self, tmp_path):
        yaml_file = tmp_path / "empty.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump({}, f)

        cfg = ExperimentConfig.from_yaml(yaml_file)
        assert cfg.data.data_path is None
        assert cfg.inference.method == "mcmc"

    def test_to_yaml(self, tmp_path):
        cfg = ExperimentConfig()
        yaml_file = tmp_path / "output.yaml"
        cfg.to_yaml(yaml_file)

        with open(yaml_file, "r") as f:
            loaded = yaml.safe_load(f)

        assert loaded["data"]["random_seed"] == 42
        assert loaded["inference"]["method"] == "mcmc"
        assert loaded["model"]["family"] == "gaussian"

    def test_yaml_round_trip(self, tmp_path):
        original = ExperimentConfig(
            data=DataConfig(data_path="data.csv", target_col="y", test_size=0.3),
            model=ModelConfig(family="binomial", scale_global=0.5),
            inference=InferenceConfig(method="svi", num_steps=8000, learning_rate=0.005),
            selection=SelectionConfig(method="both", threshold=0.6),
            output=OutputConfig(save_plots=False, save_diagnostics=False, save_samples=True),
        )

        yaml_file = tmp_path / "roundtrip.yaml"
        original.to_yaml(yaml_file)
        reloaded = ExperimentConfig.from_yaml(yaml_file)

        assert reloaded.data.data_path == original.data.data_path
        assert reloaded.data.target_col == original.data.target_col
        assert reloaded.data.test_size == original.data.test_size
        assert reloaded.model.family == original.model.family
        assert reloaded.model.scale_global == original.model.scale_global
        assert reloaded.inference.method == original.inference.method
        assert reloaded.inference.num_steps == original.inference.num_steps
        assert reloaded.inference.learning_rate == original.inference.learning_rate
        assert reloaded.selection.method == original.selection.method
        assert reloaded.selection.threshold == original.selection.threshold
        assert reloaded.output.save_plots == original.output.save_plots
        assert reloaded.output.save_diagnostics == original.output.save_diagnostics
        assert reloaded.output.save_samples == original.output.save_samples

    def test_update_from_dict(self):
        cfg = ExperimentConfig()
        cfg.update_from_dict({
            "data": {"data_path": "updated.csv", "standardize": True},
            "inference": {"method": "svi", "learning_rate": 0.01},
        })
        assert cfg.data.data_path == "updated.csv"
        assert cfg.data.standardize is True
        assert cfg.inference.method == "svi"
        assert cfg.inference.learning_rate == 0.01
        # Unchanged fields keep defaults
        assert cfg.data.random_seed == 42
        assert cfg.model.family == "gaussian"

    def test_update_from_dict_returns_self(self):
        cfg = ExperimentConfig()
        result = cfg.update_from_dict({"data": {"target_col": "y"}})
        assert result is cfg

    def test_update_from_dict_unknown_section_ignored(self):
        cfg = ExperimentConfig()
        cfg.update_from_dict({"nonexistent": {"key": "value"}})
        # Should not raise; config unchanged
        assert cfg.data.data_path is None

    def test_update_from_dict_unknown_field_ignored(self):
        cfg = ExperimentConfig()
        cfg.update_from_dict({"data": {"nonexistent_field": 999}})
        # Should not raise; existing fields unchanged
        assert cfg.data.data_path is None
