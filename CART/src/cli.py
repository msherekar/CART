import typer
from pathlib import Path

from car_ai_system.data.augmentation import run_augmentation
from car_ai_system.data.mds import run_mds
from car_ai_system.modeling.finetune import fine_tune
from car_ai_system.modeling.pll import compute_pll
from car_ai_system.modeling.extract_embeddings import extract_embeddings
from car_ai_system.modeling.pred import run_prediction, evaluate_metrics
from car_ai_system.utils import get_logger

logger = get_logger(__name__)
app = typer.Typer()

@app.command()
def augment(
    wt_cd28: Path = typer.Option(..., exists=True, help="WT CD28 FASTA path"),
    wt_cd3z: Path = typer.Option(..., exists=True, help="WT CD3ζ FASTA path"),
    db: Path = typer.Option(..., exists=True, help="Uniprot DB FASTA path"),
    outdir: Path = typer.Option("augmented_seqs", help="Output directory for augmented sequences"),
):
    """1) Sequence augmentation via HMMER, clustering, and generation."""
    logger.info("Starting augmentation...")
    run_augmentation(str(wt_cd28), str(wt_cd3z), str(db), str(outdir))
    logger.info("Augmentation complete.")

@app.command()
def mds(
    high_fasta: Path = typer.Option(..., exists=True, help="High-diversity FASTA"),
    low_fasta: Path = typer.Option(..., exists=True, help="Low-diversity FASTA"),
    frac: float = typer.Option(0.02, help="Fraction to sample for MDS"),
    outplot: Path = typer.Option("mds.png", help="Output plot file path"),
):
    """2) Perform MDS and save scatterplot."""
    logger.info("Running MDS analysis...")
    run_mds(str(high_fasta), str(low_fasta), frac, str(outplot))
    logger.info(f"MDS plot saved to {outplot}")

@app.command()
def finetune(
    group: str = typer.Option(..., help="Which set to fine-tune: 'high' or 'low'"),
    device: str = typer.Option("auto", help="Device: 'auto', 'cuda', 'mps', or 'cpu'"),
    use_wandb: bool = typer.Option(False, help="Enable Weights & Biases logging"),
    use_mlflow: bool = typer.Option(False, help="Enable MLflow logging"),
    patience: int = typer.Option(5, help="Early stopping patience in epochs"),
):
    """3) Fine-tune ESM2 on high/low diversity sequences."""
    fine_tune(
        group=group,
        device=device,
        use_wandb=use_wandb,
        use_mlflow=use_mlflow,
        patience=patience,
    )

@app.command()
def pll(
    mutants_fasta: Path = typer.Option(..., exists=True, help="FASTA of CAR mutant sequences"),
    models: list[str] = typer.Option(..., help="List of model paths or identifiers"),
):
    """4) Compute pseudo log-likelihood for each model."""
    compute_pll(str(mutants_fasta), models)

@app.command()
def embed(
    fasta: Path = typer.Option(..., exists=True, help="FASTA file for embedding extraction"),
    model: str = typer.Option(..., help="ESM model path or Hugging Face identifier"),
    out_emb: Path = typer.Option("embeddings.npy", help="Output embeddings file"),
):
    """5) Extract [CLS] embeddings from a model."""
    extract_embeddings(str(fasta), model, str(out_emb))

@app.command()
def predict(
    embeddings: Path = typer.Option(..., exists=True, help="Embeddings .npy file"),
    labels: Path = typer.Option(..., exists=True, help="Labels .csv or .npy file"),
):
    """6) Nested-CV Ridge regression prediction."""
    run_prediction(str(embeddings), str(labels))

@ app.command()
def evaluate_cmd(
    y_trues: list[Path] = typer.Option(..., exists=True, help="List of ground-truth .npy files"),
    y_preds: list[Path] = typer.Option(..., exists=True, help="List of prediction .npy files"),
):
    """7) Compute Recall@K, Precision@K, and Spearman’s ρ."""
    # Convert Path objects to strings
    y_true_paths = [str(p) for p in y_trues]
    y_pred_paths = [str(p) for p in y_preds]
    evaluate_metrics(y_true_paths, y_pred_paths)

if __name__ == "__main__":
    app()