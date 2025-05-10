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
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
)
from Bio import SeqIO
import wandb
import mlflow
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import pandas as pd
import sys
from CART.src.modeling.embeddings import get_model_size_mb, check_memory_requirements


def parse_args(args_list=None):
    parser = argparse.ArgumentParser(description="Fine-tune ESM2 model on CAR-T sequences")

    # Input FASTA files
    parser.add_argument("--high_fasta",type=Path,default=Path('../../output/augmented/high_diversity.fasta'),help="Path to high-diversity FASTA file")
    parser.add_argument("--low_fasta",type=Path,default=Path('../../output/augmented/low_diversity.fasta'),help="Path to low-diversity FASTA file")
    parser.add_argument("--groups",nargs='+',choices=["high","low"],default=["high","low"],help="One or more groups to fine-tune (default: both)")
    parser.add_argument("--output_dir",type=Path,default=Path('../../output/models'),help="Directory to save fine-tuned models")
    parser.add_argument("--device",type=str,choices=["auto","cuda","mps","cpu"],default="auto",help="Compute device to use")
    parser.add_argument("--batch_size",type=int,default=32,help="Training batch size")
    parser.add_argument("--max_length",type=int,default=256,help="Maximum sequence length")
    parser.add_argument("--grad_accum",type=int,default=4,help="Gradient accumulation steps")
    parser.add_argument("--no_pin_memory",action="store_true",help="Disable pin memory for data loading")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate")
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/esm2_t6_8M_UR50D",
        help="Model name or path"
    )
    parser.add_argument("--max_epochs", type=int, default=50, help="Maximum number of epochs")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument(
        "--top_k_percent",
        type=float,
        default=25.0,
        help="Top K percent for evaluation"
    )

    # Logging options
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases")
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="esm2-cart",
        help="Weights & Biases project name"
    )
    parser.add_argument("--use_mlflow", action="store_true", help="Use MLflow")
    parser.add_argument(
        "--mlflow_experiment",
        type=str,
        default="esm2-cart",
        help="MLflow experiment name"
    )
    parser.add_argument(
        "--view_mlflow", action="store_true", help="View MLflow results"
    )

    if args_list is None and __name__ == "__main__":
        return parser.parse_args()
    else:
        return parser.parse_args(args_list)


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


def compute_metrics(model, val_loader, device):
    """Compute Spearman correlation between predictions and targets."""
    model.eval()
    all_preds, all_targs = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = outputs.logits.argmax(dim=-1)
            targs = batch['labels']
            mask = targs != -100
            all_preds.extend(preds[mask].cpu().numpy())
            all_targs.extend(targs[mask].cpu().numpy())
    return spearmanr(all_preds, all_targs)[0]


def plot_metrics(train_losses, val_losses, spearman_scores, output_dir: Path, group: str):
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.plot(epochs, train_losses, label='Train Loss')
    ax1.plot(epochs, val_losses, label='Val Loss')
    ax1.set(title=f'Loss ({group})', xlabel='Epoch', ylabel='Loss')
    ax1.legend(); ax1.grid(True)
    ax2.plot(epochs, spearman_scores, label='Spearman')
    ax2.set(title=f'Spearman ({group})', xlabel='Epoch', ylabel='Spearman')
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / f'{group}_metrics.png')
    plt.close()


def run_finetuning(args, group: str):
    """Run fine-tuning for a specified diversity group."""
    root = Path(__file__).parent.parent.resolve()
    checkpoints_dir = root / "output" / "checkpoints"
    mlruns_dir = root / "output" / "mlruns"
    wandb_dir = root / "output" / "wandb"

    device = select_device(args.device)
    print(f"[INFO] Using device: {device}")

    # Check if we have enough memory for the model
    if check_memory_requirements(args.model_name, device):
        print(f"[INFO] Memory check passed for model: {args.model_name}")
    
    is_high = (group == "high")
    model_dir = args.output_dir / group
    if model_dir.exists():
        print(f"[INFO] Model dir exists: {model_dir}, clearing...")
        for f in model_dir.glob('*'):
            if f.is_file(): f.unlink()
    else:
        model_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Saving models under: {model_dir}")

    # W&B setup
    if args.use_wandb:
        os.environ["WANDB_DIR"] = str(wandb_dir)
        wandb.init(project=args.wandb_project, name=f"{group}_finetune", config=vars(args))

    # MLflow setup
    if args.use_mlflow:
        os.environ["MLFLOW_TRACKING_URI"] = f"file://{mlruns_dir}"
        mlflow.set_experiment(args.mlflow_experiment)
        mlflow.start_run(run_name=f"{group}_finetune")
        mlflow.log_params({
            "group": group,
            "lr": args.learning_rate,
            "batch_size": args.batch_size,
            "epochs": args.max_epochs,
            "device": str(device),
            "patience": args.patience,
        })

    # Data + model
    try:
        print(f"[INFO] Loading {args.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModelForMaskedLM.from_pretrained(args.model_name).to(device)
        
        # Adjust batch size based on model size to avoid OOM errors
        model_size_mb = get_model_size_mb(args.model_name)
        actual_batch_size = args.batch_size
        
        if model_size_mb > 150 and device.type == 'cuda':
            # For larger models, reduce batch size
            size_factor = min(150 / model_size_mb, 1.0)
            adjusted_batch_size = max(1, int(args.batch_size * size_factor))
            
            if adjusted_batch_size < args.batch_size:
                print(f"[WARNING] Reducing batch size from {args.batch_size} to {adjusted_batch_size} for large model")
                actual_batch_size = adjusted_batch_size
        
        fasta = args.high_fasta if is_high else args.low_fasta
        print(f"[INFO] Loading sequences from: {fasta}")
        full_ds = SequenceDataset(fasta, tokenizer, max_length=args.max_length)
        train_size = int(0.8 * len(full_ds))
        val_size = len(full_ds) - train_size
        train_ds, val_ds = random_split(full_ds, [train_size, val_size],
                                        generator=torch.Generator().manual_seed(42))
        
        # Use the adjusted batch size
        train_loader = DataLoader(
            train_ds, 
            batch_size=actual_batch_size, 
            shuffle=True, 
            pin_memory=not args.no_pin_memory
        )
        val_loader = DataLoader(
            val_ds, 
            batch_size=actual_batch_size, 
            shuffle=False, 
            pin_memory=not args.no_pin_memory
        )
        
        optimizer = AdamW(model.parameters(), lr=args.learning_rate)
        total_steps = len(train_loader) * args.max_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)

        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        args.output_dir.mkdir(parents=True, exist_ok=True)

        train_losses, val_losses, spearman_scores = [], [], []
        best_spear, no_improve, best_ep = float('-inf'), 0, 0

        for epoch in range(1, args.max_epochs + 1):
            model.train()
            running_loss = 0.0
            for i, batch in enumerate(train_loader):
                batch = {k: v.to(device) for k, v in batch.items()}
                loss = model(**batch).loss / args.grad_accum
                loss.backward()
                if (i + 1) % args.grad_accum == 0 or i == len(train_loader) - 1:
                    optimizer.step(); scheduler.step(); optimizer.zero_grad()
                running_loss += loss.item() * args.grad_accum
            avg_train = running_loss / len(train_loader)
            train_losses.append(avg_train)

            model.eval()
            val_loss = sum(model(**{k: v.to(device) for k, v in b.items()}).loss.item()
                           for b in val_loader) / len(val_loader)
            val_losses.append(val_loss)

            spear = compute_metrics(model, val_loader, device)
            spearman_scores.append(spear)
            print(f"Epoch {epoch} — train: {avg_train:.4f}, val: {val_loss:.4f}, spearman: {spear:.4f}")

            # Logging
            if args.use_wandb:
                wandb.log({"epoch": epoch, "train_loss": avg_train,
                           "val_loss": val_loss, "spearman": spear})
            if args.use_mlflow:
                mlflow.log_metrics({"train_loss": avg_train,
                                    "val_loss": val_loss,
                                    "spearman": spear}, step=epoch)

            # Early stopping & checkpoint
            if spear > best_spear:
                best_spear, best_ep, no_improve = spear, epoch, 0
                model.save_pretrained(model_dir)
                tokenizer.save_pretrained(model_dir)
                print(f"[INFO] Saved best model at epoch {epoch}")
            else:
                no_improve += 1
                if no_improve >= args.patience:
                    print(f"[INFO] Early stopping at epoch {epoch}")
                    break

            if epoch % 5 == 0:
                chk = model_dir / f"epoch_{epoch}"
                chk.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(chk)
                tokenizer.save_pretrained(chk)

        # Save metrics and plots
        df = pd.DataFrame({"epoch": range(1, len(train_losses)+1),
                           "train_loss": train_losses,
                           "val_loss": val_losses,
                           "spearman": spearman_scores})
        df.to_csv(model_dir / f"{group}_metrics.csv", index=False)
        plot_metrics(train_losses, val_losses, spearman_scores, model_dir, group)

        if args.use_wandb: wandb.finish()
        if args.use_mlflow: mlflow.end_run()
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            print(f"[ERROR] GPU out of memory when loading model {args.model_name}")
            print(f"[ERROR] Try using a smaller model, reducing batch size, or moving to CPU with --device cpu")
            return
        else:
            print(f"[ERROR] Failed to initialize model or data: {e}")
            return
    except Exception as e:
        print(f"[ERROR] Unexpected error during model/data initialization: {e}")
        return


def main():
    args = parse_args()
    for grp in args.groups:
        print(f"\n=== Training {grp.title()} Diversity Model ===")
        run_finetuning(args, grp)


if __name__ == "__main__":
    main()
