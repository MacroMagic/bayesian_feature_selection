"""Data loading and preprocessing utilities."""

from pathlib import Path
from typing import Tuple, Optional, List
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import DataConfig


class DataLoader:
    """
    Handle data loading and preprocessing for Bayesian feature selection.
    
    Supports:
    - CSV file loading
    - Train/test splitting
    - Feature standardization
    - Feature selection
    """
    
    def __init__(self, config: DataConfig):
        """
        Initialize DataLoader.
        
        Parameters
        ----------
        config : DataConfig
            Data configuration
        """
        self.config = config
        self.scaler = None
        self.feature_names = None
        
    def load_data(
        self,
        data_path: Optional[Path] = None,
        target_col: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load data from CSV file.
        
        Parameters
        ----------
        data_path : Path, optional
            Path to CSV file (overrides config)
        target_col : str, optional
            Target column name (overrides config)
            
        Returns
        -------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target vector
        feature_names : List[str]
            Feature names
        """
        # Use provided arguments or fall back to config
        path = data_path or self.config.data_path
        target = target_col or self.config.target_col
        
        if path is None:
            raise ValueError("data_path must be provided either in config or as argument")
        if target is None:
            raise ValueError("target_col must be provided either in config or as argument")
        
        # Load CSV
        df = pd.read_csv(path)
        
        # Extract target
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in data")
        
        y = df[target].values
        
        # Extract features
        if self.config.feature_cols is not None:
            # Use specified features
            missing_cols = [col for col in self.config.feature_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Feature columns not found: {missing_cols}")
            X = df[self.config.feature_cols].values
            feature_names = self.config.feature_cols
        else:
            # Use all columns except target
            feature_names = [col for col in df.columns if col != target]
            X = df[feature_names].values
        
        self.feature_names = feature_names
        
        # Standardize if requested
        if self.config.standardize:
            X = self._standardize(X)
        
        return X, y, feature_names
    
    def load_and_split(
        self,
        data_path: Optional[Path] = None,
        target_col: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Load data and split into train/test sets.
        
        Parameters
        ----------
        data_path : Path, optional
            Path to CSV file (overrides config)
        target_col : str, optional
            Target column name (overrides config)
            
        Returns
        -------
        X_train : np.ndarray
            Training features
        X_test : np.ndarray
            Test features
        y_train : np.ndarray
            Training targets
        y_test : np.ndarray
            Test targets
        feature_names : List[str]
            Feature names
        """
        X, y, feature_names = self.load_data(data_path, target_col)
        
        if self.config.test_size > 0:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.config.test_size,
                random_state=self.config.random_seed
            )
        else:
            # No split, use all data for training
            X_train, y_train = X, y
            X_test, y_test = np.array([]), np.array([])
        
        return X_train, X_test, y_train, y_test, feature_names
    
    def _standardize(self, X: np.ndarray) -> np.ndarray:
        """
        Standardize features to zero mean and unit variance.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
            
        Returns
        -------
        np.ndarray
            Standardized features
        """
        if self.scaler is None:
            self.scaler = StandardScaler()
            return self.scaler.fit_transform(X)
        else:
            return self.scaler.transform(X)
    
    def save_predictions(
        self,
        predictions: np.ndarray,
        output_path: Path,
        include_features: bool = False,
        X: Optional[np.ndarray] = None
    ) -> None:
        """
        Save predictions to CSV file.
        
        Parameters
        ----------
        predictions : np.ndarray
            Model predictions
        output_path : Path
            Output file path
        include_features : bool
            Include feature values in output
        X : np.ndarray, optional
            Feature matrix (required if include_features=True)
        """
        df_dict = {"prediction": predictions}
        
        if include_features:
            if X is None:
                raise ValueError("X must be provided when include_features=True")
            if self.feature_names is None:
                raise ValueError("Feature names not available")
            
            for i, name in enumerate(self.feature_names):
                df_dict[name] = X[:, i]
        
        df = pd.DataFrame(df_dict)
        df.to_csv(output_path, index=False)


def load_data_from_config(
    config: DataConfig,
    data_path: Optional[Path] = None,
    target_col: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Convenience function to load data from config.
    
    Parameters
    ----------
    config : DataConfig
        Data configuration
    data_path : Path, optional
        Path to CSV file (overrides config)
    target_col : str, optional
        Target column name (overrides config)
        
    Returns
    -------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target vector
    feature_names : List[str]
        Feature names
    """
    loader = DataLoader(config)
    return loader.load_data(data_path, target_col)
