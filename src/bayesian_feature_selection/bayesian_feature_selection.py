"""Main module for Bayesian feature selection with horseshoe prior."""

from typing import Optional, Dict, Any, Tuple, Literal
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, autoguide
import pandas as pd
import numpy as np
from dataclasses import dataclass


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


class HorseshoeGLM:
    """
    Bayesian GLM with horseshoe prior for feature selection.
    
    The horseshoe prior provides strong shrinkage for irrelevant features
    while keeping relevant features relatively unshrunk.
    
    Supports:
    - Linear regression (family='gaussian')
    - Logistic regression (family='binomial')
    - Poisson regression (family='poisson')
    """
    
    def __init__(
        self,
        family: Literal["gaussian", "binomial", "poisson"] = "gaussian",
        scale_global: float = 1.0,
        scale_slab: float = 2.0,
    ):
        """
        Initialize HorseshoeGLM.
        
        Parameters
        ----------
        family : str
            Distribution family for GLM
        scale_global : float
            Global shrinkage parameter (tau)
        scale_slab : float
            Slab scale parameter
        """
        self.family = family
        self.scale_global = scale_global
        self.scale_slab = scale_slab
        self.mcmc = None
        self.svi_result = None
        self.X_train = None
        self.y_train = None
        
    def model(self, X: jnp.ndarray, y: Optional[jnp.ndarray] = None):
        """
        Horseshoe prior GLM model.
        
        Parameters
        ----------
        X : jnp.ndarray
            Design matrix (n_samples, n_features)
        y : jnp.ndarray, optional
            Response variable
        """
        n_samples, n_features = X.shape
        
        # Global shrinkage parameter
        tau = numpyro.sample("tau", dist.HalfCauchy(self.scale_global))
        
        # Local shrinkage parameters (per feature)
        lambda_local = numpyro.sample(
            "lambda_local",
            dist.HalfCauchy(jnp.ones(n_features))
        )
        
        # Regularized horseshoe (optional slab)
        c2 = numpyro.sample("c2", dist.InverseGamma(1.0, 1.0))
        lambda_tilde = jnp.sqrt(
            (c2 * lambda_local**2) / (c2 + tau**2 * lambda_local**2)
        )
        
        # Coefficients with horseshoe prior
        beta = numpyro.sample(
            "beta",
            dist.Normal(0, tau * lambda_tilde)
        )
        
        # Intercept
        intercept = numpyro.sample("intercept", dist.Normal(0, 10))
        
        # Linear predictor
        eta = intercept + X @ beta
        
        # Likelihood based on family
        if self.family == "gaussian":
            sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
            numpyro.sample("y", dist.Normal(eta, sigma), obs=y)
        elif self.family == "binomial":
            logits = eta
            numpyro.sample("y", dist.Bernoulli(logits=logits), obs=y)
        elif self.family == "poisson":
            numpyro.sample("y", dist.Poisson(jnp.exp(eta)), obs=y)
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        config: Optional[InferenceConfig] = None
    ) -> "HorseshoeGLM":
        """
        Fit the model using MCMC or SVI.
        
        Parameters
        ----------
        X : np.ndarray
            Design matrix
        y : np.ndarray
            Response variable
        config : InferenceConfig, optional
            Inference configuration
            
        Returns
        -------
        self
        """
        if config is None:
            config = InferenceConfig()
        
        # Convert to JAX arrays
        X_jax = jnp.array(X)
        y_jax = jnp.array(y)
        
        self.X_train = X_jax
        self.y_train = y_jax
        
        # Set device
        if config.use_gpu and jax.devices("gpu"):
            numpyro.set_platform("gpu")
        else:
            numpyro.set_platform("cpu")
        
        if config.method == "mcmc":
            self._fit_mcmc(X_jax, y_jax, config)
        elif config.method == "svi":
            self._fit_svi(X_jax, y_jax, config)
        else:
            raise ValueError(f"Unknown method: {config.method}")
        
        return self
    
    def _fit_mcmc(self, X: jnp.ndarray, y: jnp.ndarray, config: InferenceConfig):
        """Fit using MCMC (NUTS)."""
        rng_key = jax.random.PRNGKey(0)
        kernel = NUTS(self.model)
        
        self.mcmc = MCMC(
            kernel,
            num_warmup=config.num_warmup,
            num_samples=config.num_samples,
            num_chains=config.num_chains,
            progress_bar=config.progress_bar
        )
        
        self.mcmc.run(rng_key, X, y)
    
    def _fit_svi(self, X: jnp.ndarray, y: jnp.ndarray, config: InferenceConfig):
        """Fit using Stochastic Variational Inference."""
        rng_key = jax.random.PRNGKey(0)
        
        # Automatic guide (mean-field approximation)
        guide = autoguide.AutoNormal(self.model)
        
        optimizer = numpyro.optim.Adam(step_size=config.learning_rate)
        svi = SVI(self.model, guide, optimizer, loss=Trace_ELBO())
        
        svi_result = svi.run(
            rng_key,
            config.num_steps,
            X,
            y,
            progress_bar=config.progress_bar
        )
        
        self.svi_result = svi_result
        self.guide = guide
    
    def get_feature_importance(self, threshold: float = 0.5) -> pd.DataFrame:
        """
        Extract feature importance based on posterior inclusion probabilities.
        
        Parameters
        ----------
        threshold : float
            Threshold for feature selection (based on |beta| credible interval)
            
        Returns
        -------
        pd.DataFrame
            Feature importance metrics
        """
        if self.mcmc is None and self.svi_result is None:
            raise ValueError("Model must be fitted first")
        
        if self.mcmc is not None:
            samples = self.mcmc.get_samples()
            beta_samples = samples["beta"]
        else:
            # Get posterior samples from SVI
            guide_samples = self.guide.sample_posterior(
                jax.random.PRNGKey(1),
                self.svi_result.params,
                sample_shape=(1000,)
            )
            beta_samples = guide_samples["beta"]
        
        # Calculate statistics
        beta_mean = jnp.mean(beta_samples, axis=0)
        beta_std = jnp.std(beta_samples, axis=0)
        
        # Credible intervals
        beta_lower = jnp.percentile(beta_samples, 2.5, axis=0)
        beta_upper = jnp.percentile(beta_samples, 97.5, axis=0)
        
        # Inclusion probability (CI doesn't contain 0)
        inclusion_prob = jnp.mean(
            (beta_lower > 0) | (beta_upper < 0),
            axis=0
        )
        
        # Feature selection
        selected = inclusion_prob > threshold
        
        importance_df = pd.DataFrame({
            "feature_idx": range(len(beta_mean)),
            "beta_mean": np.array(beta_mean),
            "beta_std": np.array(beta_std),
            "beta_lower_95": np.array(beta_lower),
            "beta_upper_95": np.array(beta_upper),
            "inclusion_prob": np.array(inclusion_prob),
            "selected": np.array(selected)
        })
        
        return importance_df.sort_values("inclusion_prob", ascending=False)
    
    def predict(
        self,
        X_new: np.ndarray,
        return_samples: bool = False
    ) -> np.ndarray:
        """
        Make predictions on new data.
        
        Parameters
        ----------
        X_new : np.ndarray
            New design matrix
        return_samples : bool
            If True, return posterior predictive samples
            
        Returns
        -------
        np.ndarray
            Predictions (mean or samples)
        """
        from numpyro.infer import Predictive
        
        X_new_jax = jnp.array(X_new)
        
        if self.mcmc is not None:
            predictive = Predictive(self.model, self.mcmc.get_samples())
        else:
            predictive = Predictive(
                self.model,
                guide=self.guide,
                params=self.svi_result.params,
                num_samples=1000
            )
        
        rng_key = jax.random.PRNGKey(2)
        predictions = predictive(rng_key, X_new_jax, None)["y"]
        
        if return_samples:
            return np.array(predictions)
        else:
            return np.array(jnp.mean(predictions, axis=0))
