#!/usr/bin/env python3
"""
Fine-tune ESM-2 with masked language modeling on high/low-diversity CAR sequence sets.
Supports GPU (CUDA), Apple Metal (MPS), or CPU via --device.
Adds:
  - wandb logging (--use_wandb, --wandb_project)
  - mlflow logging (--use_mlflow, --mlflow_experiment)
  - early stopping (--patience)
  - visualization of training metrics (--plot_metrics)
"""
import argparse
import os
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from transformers import (AutoTokenizer,AutoModelForMaskedLM,DataCollatorForLanguageModeling,get_linear_schedule_with_warmup)
from Bio import SeqIO
import wandb
import mlflow
import numpy as np
from scipy.stats import spearmanr

# Import visualization module for plotting metrics
try:
    from .visualization import (
        plot_training_metrics,
        plot_spearman_vs_epoch,
        plot_model_comparison,
        plot_recall_precision
    )
except ImportError:
    from visualization import (
        plot_training_metrics,
        plot_spearman_vs_epoch,
        plot_model_comparison,
        plot_recall_precision
    )


class SequenceDataset(Dataset):
    def __init__(self, fasta_path: Path, tokenizer, max_length: int = 512):
        self.seqs = [str(rec.seq) for rec in SeqIO.parse(str(fasta_path), "fasta")]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.seqs[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }

def select_device(choice: str) -> torch.device:
    if choice == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(choice)

def parse_args():
    # Define project root for relative paths
    project_root = Path("/Users/mukulsherekar/pythonProject/CART-Project")
    checkpoints_dir = project_root / "checkpoints"
    mlruns_dir = project_root / "mlruns"
    wandb_dir = project_root / "wandb"
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--high_fasta",  type=Path, default=project_root / "CART/homologs/high_diversity.fasta")
    parser.add_argument("--low_fasta",   type=Path, default=project_root / "CART/homologs/low_diversity.fasta")
    parser.add_argument("--group",       choices=["high", "low"], required=True, help="Which set to fine-tune on")                        
    parser.add_argument("--output_dir",  type=Path, default=checkpoints_dir, help="Directory to save models")
    parser.add_argument("--device",      choices=["auto", "cuda", "mps", "cpu"], default="mps", help="Device to run on")
    parser.add_argument("--batch_size",  type=int, default=32, help="Batch size for training and validation")
    parser.add_argument("--max_length",  type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--grad_accum",  type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--no_pin_memory", action="store_true", help="Disable pin_memory in DataLoader to save memory")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate for the optimizer")
    # Model selection
    parser.add_argument("--model_name", type=str, default="facebook/esm2_t6_8M_UR50D", 
                       help="ESM2 model to use (default: 8M parameter model)")
    # Training parameters
    parser.add_argument("--max_epochs", type=int, default=50, help="Maximum number of epochs")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (in epochs)")
    # Evaluation parameters
    parser.add_argument("--top_k_percent", type=float, default=25.0, 
                       help="Top percentage of sequences to consider for binary evaluation")
    # wandb & mlflow flags
    parser.add_argument("--use_wandb", action="store_true", help="Log to Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="esm2-cart", help="wandb project name")
    parser.add_argument("--use_mlflow", action="store_true", help="Log to MLflow")
    parser.add_argument("--mlflow_experiment", type=str, default="esm2-cart", help="mlflow experiment name")
    parser.add_argument("--view_mlflow", action="store_true", help="Start MLflow UI after training")
    
    return parser.parse_args()

def compute_metrics(model, val_loader, device):
    """Compute evaluation metrics including binary classification for top K%."""
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in val_loader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            predictions = outputs.logits.argmax(dim=-1)
            targets = inputs['labels']
            
            # Filter out masked tokens (-100)
            mask = targets != -100
            predictions = predictions[mask]
            targets = targets[mask]
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Convert to numpy arrays
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)
    
    # Compute Spearman correlation
    spearman = spearmanr(predictions, targets)[0]
    
    # Compute binary metrics for top K%
    k = int(len(targets) * (args.top_k_percent / 100))
    top_k_indices = np.argsort(targets)[-k:]
    
    # Sort predictions and get top K
    sorted_pred_indices = np.argsort(predictions)[-k:]
    
    # Compute Recall@K and Precision@K
    recall_at_k = len(set(top_k_indices) & set(sorted_pred_indices)) / k
    precision_at_k = len(set(top_k_indices) & set(sorted_pred_indices)) / k
    
    return {
        'spearman': spearman,
        'recall_at_k': recall_at_k,
        'precision_at_k': precision_at_k,
        'predictions': predictions.tolist(),
        'targets': targets.tolist()
    }

def run_finetuning(args):
    # Define project root for relative paths
    project_root = Path("/Users/mukulsherekar/pythonProject/CART-Project")
    checkpoints_dir = project_root / "checkpoints"
    mlruns_dir = project_root / "mlruns"
    wandb_dir = project_root / "wandb"

    device = select_device(args.device)
    print(f"[INFO] Using device: {device}")

    # ─── Init wandb ──────────────────────────────────────────────────────────────
    if args.use_wandb:
        if wandb is None:
            raise ImportError("wandb not installed; `pip install wandb` to use --use_wandb")
        # Ensure wandb directory exists
        wandb_dir.mkdir(parents=True, exist_ok=True)
        # Set WANDB_DIR environment variable to specify where to store W&B files
        os.environ["WANDB_DIR"] = str(wandb_dir)
        wandb.init(
            project=args.wandb_project,
            name=f"{args.group}_finetune",
            config=vars(args),
        )

    # ─── Init MLflow ────────────────────────────────────────────────────────────
    if args.use_mlflow:
        if mlflow is None:
            raise ImportError("mlflow not installed; `pip install mlflow` to use --use_mlflow")
        # Ensure mlruns directory exists
        mlruns_dir.mkdir(parents=True, exist_ok=True)
        # Set MLFLOW_TRACKING_URI environment variable to store runs in project directory
        os.environ["MLFLOW_TRACKING_URI"] = f"file://{mlruns_dir}"
        mlflow.set_experiment(args.mlflow_experiment)
        mlflow.start_run(run_name=f"{args.group}_finetune")
        # log hyperparameters
        mlflow.log_params({
            "group": args.group,
            "lr": args.learning_rate,
            "batch_size": args.batch_size,
            "epochs": args.max_epochs,
            "device": str(device),
            "patience": args.patience,
            "max_length": args.max_length,
            "grad_accum": args.grad_accum,
            "model": args.model_name
        })

    # ─── Prepare tokenizer + dataset ──────────────────────────────────────────────
    print(f"[INFO] Loading model and tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, do_lower_case=False)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name)
    model.to(device)

    fasta = args.high_fasta if args.group == "high" else args.low_fasta
    print(f"[INFO] Loading sequences from: {fasta}")
    full_ds = SequenceDataset(fasta, tokenizer, max_length=args.max_length)
    train_size = int(0.8 * len(full_ds))
    val_size   = len(full_ds) - train_size
    train_ds, val_ds = random_split(
        full_ds, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    print(f"[INFO] Dataset sizes - Train: {len(train_ds)}, Val: {len(val_ds)}")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=data_collator, num_workers=2, pin_memory=not args.no_pin_memory
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=data_collator, num_workers=2, pin_memory=not args.no_pin_memory
    )

    # ─── Load optimizer and scheduler ────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_loader) * args.max_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    # Ensure checkpoint directory exists
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Track metrics
    train_losses = []
    val_losses = []
    spearman_scores = []
    recall_scores = {5: [], 10: []}
    precision_scores = {5: [], 10: []}
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    print(f"[INFO] Starting training for {args.max_epochs} epochs")
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            # Scale the loss to account for gradient accumulation
            loss = loss / args.grad_accum
            loss.backward()
            
            # Only update weights after accumulating gradients for specified steps
            if (batch_idx + 1) % args.grad_accum == 0 or batch_idx == len(train_loader) - 1:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
            train_loss += loss.item() * args.grad_accum
        avg_train = train_loss / len(train_loader)
        train_losses.append(avg_train)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                val_loss += model(**batch).loss.item()
        avg_val = val_loss / len(val_loader)
        val_losses.append(avg_val)

        # Compute metrics
        metrics = compute_metrics(model, val_loader, device)
        spearman_scores.append(metrics['spearman'])
        recall_scores[5].append(metrics['recall_at_k'])
        recall_scores[10].append(metrics['recall_at_k'])
        precision_scores[5].append(metrics['precision_at_k'])
        precision_scores[10].append(metrics['precision_at_k'])
        
        print(f"Epoch {epoch:2d}  Train: {avg_train:.4f}  Val: {avg_val:.4f}  Spearman: {metrics['spearman']:.4f}")

        # ─── Log metrics ────────────────────────────────────────────────────────
        if args.use_wandb:
            wandb.log({
                "epoch": epoch, 
                "train_loss": avg_train, 
                "val_loss": avg_val,
                "spearman": metrics['spearman'],
                "recall_at_k": metrics['recall_at_k'],
                "precision_at_k": metrics['precision_at_k']
            })
        if args.use_mlflow:
            mlflow.log_metrics({
                "train_loss": avg_train,
                "val_loss": avg_val,
                "spearman": metrics['spearman'],
                "recall_at_k": metrics['recall_at_k'],
                "precision_at_k": metrics['precision_at_k']
            }, step=epoch)

        # ─── Check early stopping ───────────────────────────────────────────────
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            epochs_no_improve = 0
            # ─── Save best checkpoint ────────────────────────────────────────
            ckpt = args.output_dir / f"{args.group}_best.pth"
            torch.save(model.state_dict(), ckpt)
            if args.use_wandb:
                wandb.save(str(ckpt))
            if args.use_mlflow:
                mlflow.log_artifact(str(ckpt))
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"[INFO] Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                break

        # ─── Periodic saves every 5 epochs ──────────────────────────────────
        if epoch % 5 == 0:
            ckpt_dir = args.output_dir / f"{args.group}_epoch_{epoch}"
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            if args.use_wandb:
                artifact = wandb.Artifact(f"{args.group}_model_epoch_{epoch}", type="model")
                artifact.add_dir(str(ckpt_dir))
                wandb.log_artifact(artifact)
            if args.use_mlflow:
                mlflow.pytorch.log_model(model, artifact_path=f"{args.group}_epoch_{epoch}")

    # ─── Finish runs ───────────────────────────────────────────────────────────
    if args.use_wandb:
        wandb.finish()
    if args.use_mlflow:
        mlflow.end_run()

    # ─── View MLflow data ──────────────────────────────────────────────────────
    if args.view_mlflow:
        if mlflow is None:
            raise ImportError("mlflow not installed; `pip install mlflow` to use --view_mlflow")
        import subprocess
        print("[INFO] Starting MLflow UI. View the results at http://localhost:5000")
        print("[INFO] Press Ctrl+C to stop the MLflow UI")
        try:
            subprocess.run(["mlflow", "ui", "--port", "5000"], check=True)
        except KeyboardInterrupt:
            print("[INFO] MLflow UI stopped")

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'spearman_scores': spearman_scores,
        'recall_scores': recall_scores,
        'precision_scores': precision_scores
    }

def main():
    args = parse_args()
    run_finetuning(args)

if __name__ == "__main__":
    main()
