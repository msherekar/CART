#!/usr/bin/env python3
"""
CAR-T Cell Activity Prediction Pipeline

This script serves as the main entry point for the CAR-T prediction pipeline.
It orchestrates the entire workflow from data preparation to model evaluation.
"""

import argparse
import logging
from pathlib import Path
from typing import Optional, List

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from CART.src.data.augmentation import run_augmentation
from CART.src.data.mds import run_mds_analysis
from CART.src.data.mutants import run_mutants
from CART.src.modeling.finetuning import run_finetuning
from CART.src.modeling.pll import run_pll
from CART.src.modeling.embeddings import run_embeddings
from CART.src.modeling.prediction import run_prediction
from CART.src.modeling.evaluation import run_evaluation
from CART.src.modeling.score import run_score
from CART.src.modeling.visualization import (
    plot_metrics_from_mlflow,
    plot_training_metrics,
    plot_correlation_comparison
)
from CART.src.utils import get_project_root, get_relative_path, DEFAULT_OUTPUT_DIR

# Default paths
project_root = get_project_root()
logger.info(f"Project root: {project_root}")

DEFAULT_WT_CD28 = project_root / "fasta" / "wt_cd28.fasta"
DEFAULT_WT_CD3Z = project_root / "fasta" / "wt_cd3z.fasta"
DEFAULT_UNIPROT_DB = project_root / "fasta" / "uniprot_trembl.fasta"

logger.info(f"Default WT CD28 path: {DEFAULT_WT_CD28}")
logger.info(f"Default WT CD3Z path: {DEFAULT_WT_CD3Z}")
logger.info(f"Default Uniprot DB path: {DEFAULT_UNIPROT_DB}")

def setup_directories(base_dir: Path) -> dict:
    """Create necessary directories for the pipeline"""
    dirs = {
        'data': base_dir / 'data',
        'models': base_dir / 'models',
        'results': base_dir / 'results',
        'plots': base_dir / 'plots',
        'embeddings': base_dir / 'embeddings',
        'mutants': base_dir / 'mutants',
        'augmented': base_dir / 'augmented'
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs

def run_pipeline(
    wt_cd28: Path,
    wt_cd3z: Path,
    uniprot_db: Path,
    output_dir: Path,
    device: str = "auto",
    batch_size: int = 32,
    max_length: int = 256,
    grad_accum: int = 4,
    patience: int = 5,
    use_wandb: bool = False,
    use_mlflow: bool = False,
    num_cores: Optional[int] = None,
    skip_steps: Optional[List[str]] = None
):
    """
    Run the complete CAR-T prediction pipeline.
    
    Args:
        wt_cd28: Path to WT CD28 FASTA file
        wt_cd3z: Path to WT CD3ζ FASTA file
        uniprot_db: Path to Uniprot database FASTA file
        output_dir: Base directory for all outputs
        device: Compute device to use
        batch_size: Training batch size
        max_length: Maximum sequence length
        grad_accum: Gradient accumulation steps
        patience: Early stopping patience
        use_wandb: Whether to use Weights & Biases
        use_mlflow: Whether to use MLflow
        num_cores: Number of CPU cores to use
        skip_steps: List of steps to skip
    """
    if skip_steps is None:
        skip_steps = []
    
    # Convert all input paths to absolute paths
    wt_cd28 = Path(wt_cd28).resolve()
    wt_cd3z = Path(wt_cd3z).resolve()
    uniprot_db = Path(uniprot_db).resolve()
    output_dir = Path(output_dir).resolve()
    
    # Validate input files exist
    for path in [wt_cd28, wt_cd3z, uniprot_db]:
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
    
    # Setup directories
    dirs = setup_directories(output_dir)
    logger.info(f"Created output directories in {output_dir}")
    
    # Step 1: Sequence Augmentation
    if 'augmentation' not in skip_steps:
        logger.info("Starting sequence augmentation...")
        run_augmentation()  # This function uses command line arguments
        logger.info("Sequence augmentation complete")
    
    # Step 2: MDS Analysis
    if 'mds' not in skip_steps:
        logger.info("Running MDS analysis...")
        high_fasta = dirs['augmented'] / "high_diversity.fasta"
        low_fasta = dirs['augmented'] / "low_diversity.fasta"
        run_mds_analysis(
            high_fasta_path=str(high_fasta),
            low_fasta_path=str(low_fasta),
            output_dir=str(dirs['plots']),
            sample_frac=0.02  # Default fraction
        )
        logger.info("MDS analysis complete")
    
    # Step 3: Generate Mutants
    if 'mutants' not in skip_steps:
        logger.info("Generating mutant sequences...")
        run_mutants(
            wt_cd28=str(wt_cd28),
            wt_cd3z=str(wt_cd3z),
            output_path=str(dirs['mutants'] / "mutants.fasta")
        )
        logger.info("Mutant generation complete")
    
    # Step 4: Fine-tune Models
    if 'finetune' not in skip_steps:
        logger.info("Fine-tuning models...")
        for group in ['high', 'low']:
            run_finetuning(
                group=group,
                device=device,
                high_fasta=str(dirs['augmented'] / "high_diversity.fasta"),
                low_fasta=str(dirs['augmented'] / "low_diversity.fasta"),
                output_dir=str(dirs['models']),
                batch_size=batch_size,
                max_length=max_length,
                grad_accum=grad_accum,
                use_wandb=use_wandb,
                use_mlflow=use_mlflow,
                patience=patience
            )
        logger.info("Model fine-tuning complete")
    
    # Step 5: Compute PLLs
    if 'pll' not in skip_steps:
        logger.info("Computing pseudo-log-likelihoods...")
        run_pll(
            mutant_fasta_path=str(dirs['mutants'] / "mutants.fasta"),
            model_paths={
                "pretrained": "facebook/esm2_t6_8M_UR50D",
                "finetuned_high": str(dirs['models'] / "high"),
                "finetuned_low": str(dirs['models'] / "low")
            },
            device=device,
            num_cores=num_cores,
            output_dir=str(dirs['results']),
            plot_distribution=True
        )
        logger.info("PLL computation complete")
    
    # Step 6: Extract Embeddings
    if 'embeddings' not in skip_steps:
        logger.info("Extracting embeddings...")
        for model_type in ['pretrained', 'finetuned_high', 'finetuned_low']:
            run_embeddings(
                fasta_path=str(dirs['mutants'] / "mutants.fasta"),
                output_path=str(dirs['embeddings'] / f"{model_type}_embeddings.npy"),
                model=model_type,
                device=device
            )
        logger.info("Embedding extraction complete")
    
    # Step 7: Predict Cytotoxicity
    if 'predict' not in skip_steps:
        logger.info("Predicting cytotoxicity...")
        run_prediction(
            embedding_paths=[
                str(dirs['embeddings'] / f"{model_type}_embeddings.npy")
                for model_type in ['pretrained', 'finetuned_high', 'finetuned_low']
            ],
            output_dir=str(dirs['results']),
            use_real_labels=False  # Set to True if you have real labels
        )
        logger.info("Cytotoxicity prediction complete")
    
    # Step 8: Evaluate Models
    if 'evaluate' not in skip_steps:
        logger.info("Evaluating models...")
        run_evaluation(
            embedding_paths=[
                str(dirs['embeddings'] / f"{model_type}_embeddings.npy")
                for model_type in ['pretrained', 'finetuned_high', 'finetuned_low']
            ],
            output_dir=str(dirs['results']),
            use_real_labels=False  # Set to True if you have real labels
        )
        logger.info("Model evaluation complete")
    
    # Step 9: Score Predictions
    if 'score' not in skip_steps:
        logger.info("Scoring predictions...")
        run_score(
            predictions_dir=str(dirs['results']),
            output_dir=str(dirs['results'])
        )
        logger.info("Prediction scoring complete")
    
    logger.info("Pipeline completed successfully!")

def parse_args():
    parser = argparse.ArgumentParser(description="CAR-T Cell Activity Prediction Pipeline")
    
    # Get project root for default paths
    project_root = get_project_root()
    
    # Required arguments with sensible defaults
    parser.add_argument(
        "--wt_cd28", 
        type=Path, 
        default=project_root / "fasta" / "wt_cd28.fasta",
        help="Path to WT CD28 FASTA file"
    )
    parser.add_argument(
        "--wt_cd3z", 
        type=Path, 
        default=project_root / "fasta" / "wt_cd3z.fasta",
        help="Path to WT CD3ζ FASTA file"
    )
    parser.add_argument(
        "--uniprot_db", 
        type=Path, 
        default=project_root / "fasta" / "uniprot_trembl.fasta",
        help="Path to Uniprot database FASTA file"
    )
    
    # Optional arguments
    parser.add_argument(
        "--output_dir", 
        type=Path, 
        default=DEFAULT_OUTPUT_DIR, 
        help="Base directory for outputs"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto", 
        help="Compute device to use"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=32, 
        help="Training batch size"
    )
    parser.add_argument(
        "--max_length", 
        type=int, 
        default=256, 
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--grad_accum", 
        type=int, 
        default=4, 
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--patience", 
        type=int, 
        default=5, 
        help="Early stopping patience"
    )
    parser.add_argument(
        "--use_wandb", 
        action="store_true", 
        help="Use Weights & Biases"
    )
    parser.add_argument(
        "--use_mlflow", 
        action="store_true", 
        help="Use MLflow"
    )
    parser.add_argument(
        "--num_cores", 
        type=int, 
        help="Number of CPU cores to use"
    )
    parser.add_argument(
        "--skip", 
        nargs="+", 
        choices=[
            "augmentation", "mds", "mutants", "finetune", "pll", "embeddings", 
            "predict", "evaluate", "score"
        ], 
        help="Steps to skip"
    )
    
    args = parser.parse_args()
    
    # Validate that input files exist if using defaults
    for path, name in [
        (args.wt_cd28, "WT CD28 FASTA"),
        (args.wt_cd3z, "WT CD3ζ FASTA"),
        (args.uniprot_db, "Uniprot database FASTA")
    ]:
        if not path.exists():
            logger.warning(
                f"{name} file not found at {path}. "
                "Please provide correct paths using command line arguments."
            )
    
    return args

def main():
    args = parse_args()
    
    # Run the pipeline
    run_pipeline(
        wt_cd28=args.wt_cd28,
        wt_cd3z=args.wt_cd3z,
        uniprot_db=args.uniprot_db,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        grad_accum=args.grad_accum,
        patience=args.patience,
        use_wandb=args.use_wandb,
        use_mlflow=args.use_mlflow,
        num_cores=args.num_cores,
        skip_steps=args.skip
    )

if __name__ == "__main__":
    main() 