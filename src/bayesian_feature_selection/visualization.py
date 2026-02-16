"""Visualization utilities for Bayesian feature selection."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import arviz as az
from typing import List, Optional


def plot_feature_importance(
    importance_df: pd.DataFrame,
    output_dir: Path,
    feature_names: Optional[List[str]] = None,
    top_n: int = 20
):
    """
    Plot feature importance with credible intervals.
    
    Parameters
    ----------
    importance_df : pd.DataFrame
        Feature importance dataframe
    output_dir : Path
        Output directory for plots
    feature_names : List[str], optional
        Feature names
    top_n : int
        Number of top features to plot
    """
    # Select top features
    top_features = importance_df.head(top_n).copy()
    
    if feature_names is not None:
        top_features["feature_name"] = [
            feature_names[i] for i in top_features["feature_idx"]
        ]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    y_pos = np.arange(len(top_features))
    
    # Plot means
    ax.barh(
        y_pos,
        top_features["beta_mean"],
        color=top_features["selected"].map({True: "green", False: "gray"}),
        alpha=0.6
    )
    
    # Plot credible intervals
    ax.errorbar(
        top_features["beta_mean"],
        y_pos,
        xerr=[
            top_features["beta_mean"] - top_features["beta_lower_95"],
            top_features["beta_upper_95"] - top_features["beta_mean"]
        ],
        fmt='none',
        ecolor='black',
        capsize=3
    )
    
    ax.set_yticks(y_pos)
    if feature_names is not None:
        ax.set_yticklabels(top_features["feature_name"])
    else:
        ax.set_yticklabels([f"Feature {i}" for i in top_features["feature_idx"]])
    
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel("Coefficient Value")
    ax.set_title(f"Top {top_n} Features by Inclusion Probability")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Plot inclusion probabilities
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = top_features["selected"].map({True: "green", False: "red"})
    ax.barh(y_pos, top_features["inclusion_prob"], color=colors, alpha=0.6)
    
    ax.set_yticks(y_pos)
    if feature_names is not None:
        ax.set_yticklabels(top_features["feature_name"])
    else:
        ax.set_yticklabels([f"Feature {i}" for i in top_features["feature_idx"]])
    
    ax.set_xlabel("Inclusion Probability")
    ax.set_title("Feature Inclusion Probabilities")
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_dir / "inclusion_probabilities.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_diagnostics(mcmc, output_dir: Path):
    """
    Plot MCMC diagnostics using ArviZ.
    
    Parameters
    ----------
    mcmc : MCMC
        Fitted MCMC object
    output_dir : Path
        Output directory
    """
    # Convert to ArviZ InferenceData
    inference_data = az.from_numpyro(mcmc)
    
    # Trace plots
    az.plot_trace(inference_data, var_names=["beta", "tau"])
    plt.savefig(output_dir / "trace_plot.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Posterior plots
    az.plot_posterior(inference_data, var_names=["tau", "c2"])
    plt.savefig(output_dir / "posterior_plot.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # R-hat and ESS
    summary = az.summary(inference_data, var_names=["beta"])
    summary.to_csv(output_dir / "mcmc_summary.csv")