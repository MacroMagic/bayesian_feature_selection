"""Configuration management for bayesian_feature_selection."""

from dataclasses import dataclass, field, asdict
from typing import Literal, Optional, Dict, Any, List
from pathlib import Path
import yaml
import warnings


@dataclass
class DataConfig:
    """Data configuration."""
    data_path: Optional[str] = None
    target_col: Optional[str] = None
    feature_cols: Optional[List[str]] = None  # If None, use all except target
    test_size: float = 0.0  # Fraction for test split (0 = no split)
    standardize: bool = False  # Standardize features
    random_seed: int = 42


@dataclass
class InferenceConfig:
    """Configuration for inference."""
    method: Literal["mcmc", "svi"] = "mcmc"
    num_warmup: int = 1000
    num_samples: int = 2000
    num_chains: int = 4
    # SVI specific
    num_steps: int = 10000
    learning_rate: float = 0.001
    # Performance
    use_gpu: bool = True
    progress_bar: bool = True
    
    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        # Validate MCMC parameters
        if self.num_warmup <= 0:
            raise ValueError(
                f"num_warmup must be positive, got {self.num_warmup}"
            )
        
        if self.num_samples <= 0:
            raise ValueError(
                f"num_samples must be positive, got {self.num_samples}"
            )
        
        if self.num_chains <= 0:
            raise ValueError(
                f"num_chains must be positive, got {self.num_chains}"
            )
        
        # Validate SVI parameters
        if self.num_steps <= 0:
            raise ValueError(
                f"num_steps must be positive, got {self.num_steps}"
            )
        
        if not 0 < self.learning_rate < 1:
            raise ValueError(
                f"learning_rate must be in (0, 1), got {self.learning_rate}"
            )
        
        # Cross-parameter validation
        if self.method == "mcmc":
            if self.num_samples < self.num_warmup:
                raise ValueError(
                    f"num_samples ({self.num_samples}) should be >= "
                    f"num_warmup ({self.num_warmup}) for MCMC"
                )
            
            # Warn about potentially slow configurations
            if self.num_chains > 10:
                warnings.warn(
                    f"num_chains={self.num_chains} is high and may be slow. "
                    "Consider using fewer chains for faster inference.",
                    UserWarning
                )
            
            if self.num_samples > 10000:
                warnings.warn(
                    f"num_samples={self.num_samples} is very high. "
                    "This may take a long time to run.",
                    UserWarning
                )
        
        elif self.method == "svi":
            # Warn about potentially suboptimal SVI configurations
            if self.num_steps < 1000:
                warnings.warn(
                    f"num_steps={self.num_steps} may be too low for SVI convergence. "
                    "Consider using at least 1000 steps.",
                    UserWarning
                )
            
            if self.learning_rate > 0.1:
                warnings.warn(
                    f"learning_rate={self.learning_rate} is quite high for SVI. "
                    "Consider using a smaller value (e.g., 0.001-0.01).",
                    UserWarning
                )


@dataclass
class ModelConfig:
    """Model configuration."""
    family: Literal["gaussian", "binomial", "poisson"] = "gaussian"
    scale_global: float = 1.0


@dataclass
class SelectionConfig:
    """Feature selection configuration."""
    method: Literal["beta", "lambda", "both"] = "beta"
    threshold: float = 0.5


@dataclass
class OutputConfig:
    """Output configuration."""
    save_plots: bool = True
    save_diagnostics: bool = True
    save_samples: bool = False


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "ExperimentConfig":
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        data_cfg = DataConfig(**config_dict.get("data", {}))
        model_cfg = ModelConfig(**config_dict.get("model", {}))
        inference_cfg = InferenceConfig(**config_dict.get("inference", {}))
        selection_cfg = SelectionConfig(**config_dict.get("selection", {}))
        output_cfg = OutputConfig(**config_dict.get("output", {}))
        
        return cls(
            data=data_cfg,
            model=model_cfg,
            inference=inference_cfg,
            selection=selection_cfg,
            output=output_cfg
        )
    
    def to_yaml(self, yaml_path: Path) -> None:
        """Save configuration to YAML file."""
        
        config_dict = {
            "data": asdict(self.data),
            "model": asdict(self.model),
            "inference": asdict(self.inference),
            "selection": asdict(self.selection),
            "output": asdict(self.output)
        }
        
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    def update_from_dict(self, updates: Dict[str, Any]) -> "ExperimentConfig":
        """Update configuration from dictionary (e.g., CLI overrides)."""
        for section, params in updates.items():
            if hasattr(self, section):
                section_obj = getattr(self, section)
                for key, value in params.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)
        return self
