import typer
from pathlib import Path
from CART.src.main import run_pipeline  # <-- Import your orchestrator

# Create a Typer app with a clearly defined command structure
app = typer.Typer(help="CART pipeline CLI")

@app.callback()
def callback():
    """
    CART pipeline: AI system for CAR-T sequence augmentation, modeling, and cytotoxicity prediction.
    """
    pass

@app.command()
def run(
    steps: list[str] = typer.Option(['all'], help="Steps to run"),
    fasta_dir: Path = typer.Option(Path('fasta'), help="Directory containing FASTA files"),
    output_dir: Path = typer.Option(Path('output'), help="Directory for output files"),
    model_dir: Path = typer.Option(None, help="Directory for model files (default: output_dir/models)"),
    domain_1: Path = typer.Option(..., help="Path to fasta of domain 1 (eg hinge and transmembrane domains of CAR from CD28)"),
    domain_2: Path = typer.Option(..., help="Path to fasta of domain 2 (eg intracellular domain of CAR from CD3z)"),
    cytotox_csv: Path = typer.Option(..., help="Path to cytotoxicity csv"),
    esm_model: str = typer.Option("facebook/esm2_t6_8M_UR50D", help="Path to ESM model or Huggingface model id"),
    uniprot_db: Path = typer.Option(..., help="Path to Uniprot database FASTA file (required)"),
    wt_car: Path = typer.Option(None, help="Path to CAR sequence FASTA file (optional, for mutations)")
):
    """Run the CART pipeline with specified steps and parameters."""
    run_pipeline(
        steps, fasta_dir, output_dir, model_dir,
        domain_1, domain_2, cytotox_csv, esm_model,
        uniprot_db, wt_car
    )


if __name__ == "__main__":
    app()