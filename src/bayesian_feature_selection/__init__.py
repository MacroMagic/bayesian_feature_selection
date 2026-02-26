"""Top-level package for bayesian feature selection."""

__author__ = """Hong Mu"""
__email__ = 'hm761@nyu.edu'
__version__ = '0.1.0'

from .bayesian_feature_selection import HorseshoeGLM
from .config import (
    InferenceConfig,
    ExperimentConfig,
    ModelConfig,
    SelectionConfig,
    OutputConfig,
)

__all__ = [
    "HorseshoeGLM",
    "InferenceConfig",
    "ExperimentConfig",
    "ModelConfig",
    "SelectionConfig",
    "OutputConfig",
]
