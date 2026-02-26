"""Console script for bayesian_feature_selection."""
import typer
from rich.console import Console
from rich.table import Table
from pathlib import Path
from typing import Optional

from .bayesian_feature_selection import HorseshoeGLM
from .visualization import plot_feature_importance, plot_diagnostics
from .config import ExperimentConfig
from .data_loader import DataLoader

app = typer.Typer()
console = Console()


@app.command()
def main(
    config_path: Optional[Path] = typer.Option(
        None, 
        "--config", 
        "-c",
        help="Path to YAML config file (required or use default)"
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o", 
        help="Output directory (default: ./results)"
    ),
    # Quick overrides for common parameters
    family: Optional[str] = typer.Option(None, "--family", help="GLM family (overrides config)"),
    method: Optional[str] = typer.Option(None, "--method", help="Inference method (overrides config)"),
    use_gpu: Optional[bool] = typer.Option(None, "--use-gpu", help="Use GPU (overrides config)"),
):
    """
    Fit Bayesian GLM with horseshoe prior for feature selection.
    
    Examples:
        # Use config file with data paths specified
        bayesian-feature-selection -c configs/example_with_data.yaml
        
        # Use default config (must have data paths set)
        bayesian-feature-selection
        
        # Override model parameters
        bayesian-feature-selection -c configs/default.yaml --family binomial --method svi
        
        # Specify output directory
        bayesian-feature-selection -c configs/my_experiment.yaml -o results/experiment1
    """
    # Load configuration
    if config_path is None:
        # Use default config
        default_config = Path(__file__).parent.parent.parent / "configs" / "default.yaml"
        if default_config.exists():
            config = ExperimentConfig.from_yaml(default_config)
            console.print(f"[dim]Using default config: {default_config}[/dim]")
        else:
            # Fallback to code defaults
            config = ExperimentConfig()
            console.print("[dim]Using built-in defaults[/dim]")
    else:
        config = ExperimentConfig.from_yaml(config_path)
        console.print(f"[bold blue]Loaded config from {config_path}[/bold blue]")
    
    # Apply CLI overrides
    if family is not None:
        config.model.family = family
    if method is not None:
        config.inference.method = method
    if use_gpu is not None:
        config.inference.use_gpu = use_gpu
    
    # Validate data config
    if config.data.data_path is None:
        console.print("[bold red]Error: data_path must be specified in config or command line[/bold red]")
        raise typer.Exit(1)
    if config.data.target_col is None:
        console.print("[bold red]Error: target_col must be specified in config or command line[/bold red]")
        raise typer.Exit(1)
    
    # Set output directory
    if output_dir is None:
        output_dir = Path("./results")
    
    console.print(f"[bold blue]Loading data from {config.data.data_path}...[/bold blue]")
    
    # Load data using DataLoader
    data_loader = DataLoader(config.data)
    X, y, feature_names = data_loader.load_data()
    
    console.print(f"Data shape: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Display configuration summary
    console.print(f"\n[bold]Configuration:[/bold]")
    console.print(f"  Model: {config.model.family}, scale_global={config.model.scale_global}")
    console.print(f"  Inference: {config.inference.method}, samples={config.inference.num_samples}")
    console.print(f"  Selection: method={config.selection.method}, threshold={config.selection.threshold}")
    
    # Fit model
    console.print(f"\n[bold green]Fitting {config.model.family} GLM with {config.inference.method.upper()}...[/bold green]")
    model = HorseshoeGLM(
        family=config.model.family,
        scale_global=config.model.scale_global
    )
    model.fit(X, y, config=config.inference)
    
    # Get feature importance
    importance = model.get_feature_importance(
        threshold=config.selection.threshold,
        method=config.selection.method
    )
    importance["feature_name"] = [feature_names[i] for i in importance["feature_idx"]]
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration used for this run
    config.to_yaml(output_dir / "config.yaml")
    
    # Save results
    importance.to_csv(output_dir / "feature_importance.csv", index=False)
    console.print(f"\n[bold green]Results saved to {output_dir}[/bold green]")
    
    # Display summary
    table = Table(title="Selected Features")
    table.add_column("Feature", style="cyan")
    table.add_column("Beta (Mean)", style="magenta")
    
    # Determine which inclusion prob to show
    if config.selection.method == "beta":
        inc_col = "beta_inclusion_prob"
    elif config.selection.method == "lambda":
        inc_col = "lambda_inclusion_prob"
    else:
        inc_col = "combined_inclusion_prob"
    
    table.add_column("Inclusion Prob", style="green")
    
    selected_features = importance[importance["selected"]]
    for _, row in selected_features.iterrows():
        table.add_row(
            row["feature_name"],
            f"{row['beta_mean']:.4f}",
            f"{row[inc_col]:.4f}"
        )
    
    console.print(table)
    console.print(f"\nSelected {len(selected_features)} out of {len(importance)} features")
    
    # Generate plots
    if config.output.save_plots:
        console.print("\n[bold blue]Generating plots...[/bold blue]")
        plot_feature_importance(importance, output_dir, feature_names)
    
    if config.output.save_diagnostics and config.inference.method == "mcmc":
        console.print("[bold blue]Generating diagnostic plots...[/bold blue]")
        plot_diagnostics(model.mcmc, output_dir)
    
    console.print(f"[bold green]✓ Complete! Results in {output_dir}[/bold green]")

if __name__ == "__main__":
    app()
