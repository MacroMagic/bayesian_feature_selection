"""Console script for bayesian_feature_selection."""
import typer
from rich.console import Console
from rich.table import Table
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Optional

from .bayesian_feature_selection import HorseshoeGLM, InferenceConfig
from .visualization import plot_feature_importance, plot_diagnostics

app = typer.Typer()
console = Console()


@app.command()
def fit(
    data_path: Path = typer.Argument(..., help="Path to CSV data file"),
    target_col: str = typer.Option(..., help="Target column name"),
    output_dir: Path = typer.Option("./results", help="Output directory"),
    family: str = typer.Option("gaussian", help="GLM family: gaussian, binomial, poisson"),
    method: str = typer.Option("mcmc", help="Inference method: mcmc or svi"),
    num_samples: int = typer.Option(2000, help="Number of MCMC samples"),
    num_warmup: int = typer.Option(1000, help="Number of warmup samples"),
    num_chains: int = typer.Option(4, help="Number of MCMC chains"),
    threshold: float = typer.Option(0.5, help="Feature selection threshold"),
    use_gpu: bool = typer.Option(True, help="Use GPU if available"),
):
    """
    Fit Bayesian GLM with horseshoe prior for feature selection.
    """
    console.print(f"[bold blue]Loading data from {data_path}...[/bold blue]")
    
    # Load data
    df = pd.read_csv(data_path)
    y = df[target_col].values
    X = df.drop(columns=[target_col]).values
    feature_names = df.drop(columns=[target_col]).columns.tolist()
    
    console.print(f"Data shape: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Configure inference
    config = InferenceConfig(
        method=method,
        num_samples=num_samples,
        num_warmup=num_warmup,
        num_chains=num_chains,
        use_gpu=use_gpu
    )
    
    # Fit model
    console.print(f"[bold green]Fitting {family} GLM with {method.upper()}...[/bold green]")
    model = HorseshoeGLM(family=family)
    model.fit(X, y, config=config)
    
    # Get feature importance
    importance = model.get_feature_importance(threshold=threshold)
    importance["feature_name"] = [feature_names[i] for i in importance["feature_idx"]]
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    importance.to_csv(output_dir / "feature_importance.csv", index=False)
    console.print(f"[bold green]Results saved to {output_dir}[/bold green]")
    
    # Display summary
    table = Table(title="Selected Features")
    table.add_column("Feature", style="cyan")
    table.add_column("Beta (Mean)", style="magenta")
    table.add_column("Inclusion Prob", style="green")
    
    selected_features = importance[importance["selected"]]
    for _, row in selected_features.iterrows():
        table.add_row(
            row["feature_name"],
            f"{row['beta_mean']:.4f}",
            f"{row['inclusion_prob']:.4f}"
        )
    
    console.print(table)
    
    # Generate plots
    if method == "mcmc":
        console.print("[bold blue]Generating diagnostic plots...[/bold blue]")
        plot_diagnostics(model.mcmc, output_dir)
    
    plot_feature_importance(importance, output_dir, feature_names)
    console.print(f"[bold green]Plots saved to {output_dir}[/bold green]")


@app.command()
def predict(
    model_path: Path = typer.Argument(..., help="Path to saved model"),
    data_path: Path = typer.Argument(..., help="Path to new data CSV"),
    output_path: Path = typer.Option("predictions.csv", help="Output predictions path"),
):
    """
    Make predictions using fitted model.
    """
    console.print("[bold yellow]Prediction command - to be implemented[/bold yellow]")


if __name__ == "__main__":
    app()
