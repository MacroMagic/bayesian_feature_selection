"""Console script for bayesian_feature_selection."""
import bayesian_feature_selection

import typer
from rich.console import Console

app = typer.Typer()
console = Console()


@app.command()
def main():
    """Console script for bayesian_feature_selection."""
    console.print("Replace this message by putting your code into "
               "bayesian_feature_selection.cli.main")
    console.print("See Typer documentation at https://typer.tiangolo.com/")
    


if __name__ == "__main__":
    app()
