"""Tests for the data_loader module."""

import numpy as np
import pandas as pd
import pytest

from bayesian_feature_selection.config import DataConfig
from bayesian_feature_selection.data_loader import DataLoader, load_data_from_config


@pytest.fixture
def sample_csv(tmp_path):
    """Create a simple CSV file for testing."""
    df = pd.DataFrame({
        "feat1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "feat2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
        "feat3": [5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0],
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })
    path = tmp_path / "test_data.csv"
    df.to_csv(path, index=False)
    return path


class TestLoadData:
    """Tests for DataLoader.load_data."""

    def test_load_data_basic(self, sample_csv):
        config = DataConfig(data_path=str(sample_csv), target_col="target")
        loader = DataLoader(config)
        X, y, feature_names = loader.load_data()

        assert X.shape == (10, 3)
        assert y.shape == (10,)
        assert feature_names == ["feat1", "feat2", "feat3"]
        assert loader.feature_names == feature_names

    def test_load_data_specific_feature_cols(self, sample_csv):
        config = DataConfig(
            data_path=str(sample_csv),
            target_col="target",
            feature_cols=["feat1", "feat3"],
        )
        loader = DataLoader(config)
        X, y, feature_names = loader.load_data()

        assert X.shape == (10, 2)
        assert feature_names == ["feat1", "feat3"]
        np.testing.assert_array_equal(X[:, 0], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    def test_load_data_standardize(self, sample_csv):
        config = DataConfig(
            data_path=str(sample_csv), target_col="target", standardize=True
        )
        loader = DataLoader(config)
        X, y, feature_names = loader.load_data()

        assert X.shape == (10, 3)
        # Standardized data should have ~zero mean and ~unit variance
        np.testing.assert_allclose(X.mean(axis=0), 0.0, atol=1e-10)
        np.testing.assert_allclose(X.std(axis=0, ddof=0), 1.0, atol=1e-10)
        assert loader.scaler is not None

    def test_load_data_missing_data_path(self):
        config = DataConfig(target_col="target")
        loader = DataLoader(config)
        with pytest.raises(ValueError, match="data_path must be provided"):
            loader.load_data()

    def test_load_data_missing_target_col(self, sample_csv):
        config = DataConfig(data_path=str(sample_csv))
        loader = DataLoader(config)
        with pytest.raises(ValueError, match="target_col must be provided"):
            loader.load_data()

    def test_load_data_target_not_in_data(self, sample_csv):
        config = DataConfig(data_path=str(sample_csv), target_col="nonexistent")
        loader = DataLoader(config)
        with pytest.raises(ValueError, match="Target column 'nonexistent' not found"):
            loader.load_data()

    def test_load_data_feature_cols_not_in_data(self, sample_csv):
        config = DataConfig(
            data_path=str(sample_csv),
            target_col="target",
            feature_cols=["feat1", "missing_col"],
        )
        loader = DataLoader(config)
        with pytest.raises(ValueError, match="Feature columns not found"):
            loader.load_data()

    def test_load_data_path_override(self, sample_csv):
        config = DataConfig(target_col="target")
        loader = DataLoader(config)
        X, y, feature_names = loader.load_data(data_path=sample_csv)

        assert X.shape == (10, 3)
        assert y.shape == (10,)

    def test_load_data_target_col_override(self, sample_csv):
        config = DataConfig(data_path=str(sample_csv))
        loader = DataLoader(config)
        X, y, feature_names = loader.load_data(target_col="target")

        assert X.shape == (10, 3)
        assert y.shape == (10,)

    def test_load_data_both_overrides(self, tmp_path):
        other_csv = tmp_path / "other.csv"
        pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(other_csv, index=False)

        config = DataConfig(data_path="nonexistent.csv", target_col="wrong")
        loader = DataLoader(config)
        X, y, feature_names = loader.load_data(data_path=other_csv, target_col="b")

        assert X.shape == (2, 1)
        assert feature_names == ["a"]
        np.testing.assert_array_equal(y, [3, 4])


class TestLoadAndSplit:
    """Tests for DataLoader.load_and_split."""

    def test_split_with_test_size(self, sample_csv):
        config = DataConfig(
            data_path=str(sample_csv),
            target_col="target",
            test_size=0.3,
            random_seed=42,
        )
        loader = DataLoader(config)
        X_train, X_test, y_train, y_test, feature_names = loader.load_and_split()

        assert X_train.shape[0] + X_test.shape[0] == 10
        assert y_train.shape[0] + y_test.shape[0] == 10
        assert X_test.shape[0] == 3  # 30% of 10
        assert X_train.shape[1] == 3
        assert X_test.shape[1] == 3
        assert feature_names == ["feat1", "feat2", "feat3"]

    def test_split_with_zero_test_size(self, sample_csv):
        config = DataConfig(
            data_path=str(sample_csv), target_col="target", test_size=0.0
        )
        loader = DataLoader(config)
        X_train, X_test, y_train, y_test, feature_names = loader.load_and_split()

        assert X_train.shape == (10, 3)
        assert y_train.shape == (10,)
        assert X_test.shape == (0,)
        assert y_test.shape == (0,)

    def test_split_reproducibility(self, sample_csv):
        config = DataConfig(
            data_path=str(sample_csv),
            target_col="target",
            test_size=0.3,
            random_seed=123,
        )
        loader1 = DataLoader(config)
        result1 = loader1.load_and_split()

        loader2 = DataLoader(config)
        result2 = loader2.load_and_split()

        np.testing.assert_array_equal(result1[0], result2[0])
        np.testing.assert_array_equal(result1[1], result2[1])


class TestSavePredictions:
    """Tests for DataLoader.save_predictions."""

    def test_save_predictions_basic(self, sample_csv, tmp_path):
        config = DataConfig(data_path=str(sample_csv), target_col="target")
        loader = DataLoader(config)
        loader.load_data()

        predictions = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5, 0.5])
        output_path = tmp_path / "predictions.csv"
        loader.save_predictions(predictions, output_path)

        result = pd.read_csv(output_path)
        assert "prediction" in result.columns
        assert len(result) == 10
        np.testing.assert_allclose(result["prediction"].values, predictions)

    def test_save_predictions_with_features(self, sample_csv, tmp_path):
        config = DataConfig(data_path=str(sample_csv), target_col="target")
        loader = DataLoader(config)
        X, y, feature_names = loader.load_data()

        predictions = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5, 0.5])
        output_path = tmp_path / "predictions_with_features.csv"
        loader.save_predictions(predictions, output_path, include_features=True, X=X)

        result = pd.read_csv(output_path)
        assert list(result.columns) == ["prediction", "feat1", "feat2", "feat3"]
        assert len(result) == 10
        np.testing.assert_allclose(result["feat1"].values, X[:, 0])

    def test_save_predictions_include_features_no_X(self, sample_csv, tmp_path):
        config = DataConfig(data_path=str(sample_csv), target_col="target")
        loader = DataLoader(config)
        loader.load_data()

        predictions = np.array([0.1, 0.9])
        output_path = tmp_path / "out.csv"
        with pytest.raises(ValueError, match="X must be provided"):
            loader.save_predictions(
                predictions, output_path, include_features=True, X=None
            )

    def test_save_predictions_include_features_no_feature_names(self, tmp_path):
        config = DataConfig()
        loader = DataLoader(config)
        # feature_names is None since load_data was never called

        predictions = np.array([0.1, 0.9])
        X = np.array([[1, 2], [3, 4]])
        output_path = tmp_path / "out.csv"
        with pytest.raises(ValueError, match="Feature names not available"):
            loader.save_predictions(
                predictions, output_path, include_features=True, X=X
            )


class TestStandardize:
    """Tests for DataLoader._standardize."""

    def test_standardize_fit_transform(self, sample_csv):
        config = DataConfig(data_path=str(sample_csv), target_col="target")
        loader = DataLoader(config)

        X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        X_scaled = loader._standardize(X)

        assert loader.scaler is not None
        np.testing.assert_allclose(X_scaled.mean(axis=0), 0.0, atol=1e-10)
        np.testing.assert_allclose(X_scaled.std(axis=0, ddof=0), 1.0, atol=1e-10)

    def test_standardize_transform_reuses_scaler(self, sample_csv):
        config = DataConfig(data_path=str(sample_csv), target_col="target")
        loader = DataLoader(config)

        X_train = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        loader._standardize(X_train)

        # Second call should use existing scaler (transform, not fit_transform)
        X_new = np.array([[4.0, 40.0]])
        X_new_scaled = loader._standardize(X_new)

        # Value should be scaled using training statistics, not its own
        assert X_new_scaled.shape == (1, 2)
        # mean=2, std=0.8165 for col0 => (4-2)/0.8165 ≈ 2.449
        assert X_new_scaled[0, 0] > 1.0  # clearly above training mean


class TestLoadDataFromConfig:
    """Tests for the load_data_from_config convenience function."""

    def test_load_data_from_config(self, sample_csv):
        config = DataConfig(data_path=str(sample_csv), target_col="target")
        X, y, feature_names = load_data_from_config(config)

        assert X.shape == (10, 3)
        assert y.shape == (10,)
        assert feature_names == ["feat1", "feat2", "feat3"]

    def test_load_data_from_config_with_overrides(self, sample_csv):
        config = DataConfig()
        X, y, feature_names = load_data_from_config(
            config, data_path=sample_csv, target_col="target"
        )

        assert X.shape == (10, 3)
        assert y.shape == (10,)
