#!/usr/bin/env python3
"""
Fine-tune ESM-2 with masked language modeling on high/low-diversity CAR sequence sets.
Supports GPU (CUDA), Apple Metal (MPS), or CPU via --device.
Adds:
  - wandb logging (--use_wandb, --wandb_project)
  - mlflow logging (--use_mlflow, --mlflow_experiment)
  - early stopping (--patience)
"""
import argparse
import os
import random
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

# optional imports
try:
    import wandb
except ImportError:
    wandb = None

try:
    import mlflow
    import mlflow.pytorch
except ImportError:
    mlflow = None

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--high_fasta",  type=Path, default="augmented_seqs/high_diversity.fasta")
    parser.add_argument("--low_fasta",   type=Path, default="augmented_seqs/low_diversity.fasta")
    parser.add_argument("--group",       choices=["high", "low"], required=True,
                        help="Which set to fine-tune on")
    parser.add_argument("--output_dir",  type=Path, default=Path("fine_tuned_models"))
    parser.add_argument("--device",      choices=["auto", "cuda", "mps", "cpu"],
                        default="mps", help="Device to run on")
    parser.add_argument("--batch_size",  type=int, default=32, help="Batch size for training and validation")
    parser.add_argument("--max_length",  type=int, default=256, help="Maximum sequence length (was 512)")
    parser.add_argument("--grad_accum",  type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--no_pin_memory", action="store_true", help="Disable pin_memory in DataLoader to save memory")
    # wandb & mlflow flags
    parser.add_argument("--use_wandb", action="store_true", help="Log to Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="esm2-cart", help="wandb project name")
    parser.add_argument("--use_mlflow", action="store_true", help="Log to MLflow")
    parser.add_argument("--mlflow_experiment", type=str, default="esm2-cart", help="mlflow experiment name")
    parser.add_argument("--view_mlflow", action="store_true", help="Start MLflow UI after training")
    # early stopping
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (in epochs)")
    args = parser.parse_args()

    device = select_device(args.device)
    print(f"[INFO] Using device: {device}")

    # ─── Init wandb ──────────────────────────────────────────────────────────────
    if args.use_wandb:
        if wandb is None:
            raise ImportError("wandb not installed; `pip install wandb` to use --use_wandb")
        wandb.init(
            project=args.wandb_project,
            name=f"{args.group}_finetune",
            config=vars(args),
        )

    # ─── Init MLflow ────────────────────────────────────────────────────────────
    if args.use_mlflow:
        if mlflow is None:
            raise ImportError("mlflow not installed; `pip install mlflow` to use --use_mlflow")
        mlflow.set_experiment(args.mlflow_experiment)
        mlflow.start_run(run_name=f"{args.group}_finetune")
        # log hyperparameters
        mlflow.log_params({
            "group": args.group,
            "lr": 5e-6,
            "batch_size": args.batch_size,
            "epochs": 50,
            "device": str(device),
            "patience": args.patience,
            "max_length": args.max_length,
            "grad_accum": args.grad_accum,
        })

    # ─── Prepare tokenizer + dataset ──────────────────────────────────────────────
    model_name = "facebook/esm2_t6_8M_UR50D"
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)

    fasta = args.high_fasta if args.group == "high" else args.low_fasta
    full_ds = SequenceDataset(fasta, tokenizer, max_length=args.max_length)
    train_size = int(0.8 * len(full_ds))
    val_size   = len(full_ds) - train_size
    train_ds, val_ds = random_split(
        full_ds, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

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

    # ─── Load model, optimizer, scheduler ────────────────────────────────────────
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=5e-6)
    total_steps = len(train_loader) * 50  # 50 epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ─── Training loop w/ early stopping ────────────────────────────────────────
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, 51):
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

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                val_loss += model(**batch).loss.item()
        avg_val = val_loss / len(val_loader)

        print(f"Epoch {epoch:2d}  Train: {avg_train:.4f}  Val: {avg_val:.4f}")

        # ─── Log metrics ────────────────────────────────────────────────────────
        if args.use_wandb:
            wandb.log({"epoch": epoch, "train_loss": avg_train, "val_loss": avg_val})
        if args.use_mlflow:
            mlflow.log_metric("train_loss", avg_train, step=epoch)
            mlflow.log_metric("val_loss",   avg_val,   step=epoch)

        # ─── Check early stopping ───────────────────────────────────────────────
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            epochs_no_improve = 0
            # ─── Save best checkpoint ────────────────────────────────────────
            ckpt = args.output_dir / f"{args.group}_best.pth"
            torch.save(model.state_dict(), ckpt)
            if args.use_wandb:
                wandb.save(os.path.basename(ckpt))
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

if __name__ == "__main__":
    main()
