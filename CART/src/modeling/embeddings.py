#!/usr/bin/env python3
"""
extract_embeddings.py

Extracts per-sequence [CLS] embeddings from an ESM-2 model (pretrained or fine-tuned),
skipping any sequences longer than the model's max length, and writes them out as
a NumPy array plus a matching IDs file.
"""
import argparse
import json
from pathlib import Path

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM
from Bio import SeqIO


def select_device(choice: str) -> torch.device:
    if choice == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(choice)


def parse_args(args_list=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract CLS embeddings from ESM2 models and finetuned models")
    # Required arguments with default paths
    parser.add_argument(
        "--mutant_fasta", 
        type=Path, 
        default=Path('output/mutants/CAR_mutants.fasta'),
        help="Path to mutant FASTA file, relative to project root"
    )
    parser.add_argument(
        "--pretrained", 
        type=str, 
        default="facebook/esm2_t6_8M_UR50D", 
        help="Pretrained model name or path"
    )
    parser.add_argument(
        "--finetuned_high", 
        type=Path, 
        default=Path('../../output/models/high'), 
        help="Path to high-diversity fine-tuned model"
    )
    parser.add_argument(
        "--finetuned_low", 
        type=Path, 
        default=Path('../../output/models/low'), 
        help="Path to low-diversity fine-tuned model"
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("output/embeddings"),
        help="Output directory (relative to project root)",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Compute device",
    )
    return parser.parse_args(args_list)


def get_model_size_mb(model_name):
    """Estimate the model size in MB based on the model name."""
    if '8M' in model_name:
        return 8
    elif '35M' in model_name:
        return 35
    elif '150M' in model_name:
        return 150
    elif '650M' in model_name:
        return 650
    elif '3B' in model_name:
        return 3000
    else:
        # Default case for unknown models
        return 100


def check_memory_requirements(model_name, device):
    """Check if the device has enough memory for the model."""
    model_size_mb = get_model_size_mb(model_name)
    
    if device.type == 'cuda':
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], 
                                   capture_output=True, text=True)
            free_memory_mb = int(result.stdout.strip())
            
            # Check if we have at least 2x the model size available
            required_mb = model_size_mb * 2
            if free_memory_mb < required_mb:
                print(f"[WARNING] Your GPU has {free_memory_mb}MB free memory, but the model {model_name} requires approximately {required_mb}MB.")
                print("[WARNING] The pipeline may run out of memory. Consider using a smaller model.")
        except Exception as e:
            print(f"[INFO] Could not check GPU memory: {e}")
            print(f"[INFO] Model {model_name} requires approximately {model_size_mb * 2}MB of GPU memory.")
    
    return True


def run_embeddings(args: argparse.Namespace):
    # 1. locate project root (four levels up from this script)
    root = Path(__file__).parent.parent.parent.parent.resolve()

    # 2. resolve FASTA path
    fasta_path = args.mutant_fasta if args.mutant_fasta.is_absolute() else root / args.mutant_fasta
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA not found at {fasta_path}")

    # 3. read sequences once
    seqs = [str(r.seq) for r in SeqIO.parse(str(fasta_path), "fasta")]
    ids  = [r.id        for r in SeqIO.parse(str(fasta_path), "fasta")]
    print(f"[INFO] Loaded {len(seqs)} sequences from {fasta_path}")

    # 4. select device
    device = select_device(args.device)
    print(f"[INFO] Using device: {device}")

    # 5. iterate over all three models
    models = {
        'pretrained': args.pretrained,
        'finetuned_high': args.finetuned_high,
        'finetuned_low': args.finetuned_low
    }

    for label, spec in models.items():
        # resolve model directory or hub name
        if isinstance(spec, Path) and (root / spec).exists():
            model_dir = root / spec
            print(f"[INFO] Loading fine-tuned model ({label}) from {model_dir}")
        else:
            model_dir = str(spec)
            print(f"[INFO] Loading model ({label}) from hub: {model_dir}")
            
            # Check memory requirements for pretrained models
            if label == 'pretrained':
                check_memory_requirements(model_dir, device)

        try:
            # load model + tokenizer
            model_obj = AutoModelForMaskedLM.from_pretrained(str(model_dir))
            tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
            model_obj = model_obj.to(device).eval()

            # determine max sequence length
            max_tok    = model_obj.config.max_position_embeddings
            max_aa_len = max_tok - 2
            print(f"[INFO] [{label}] Model max tokens = {max_tok}, so max AA = {max_aa_len}")

            # extract embeddings
            embeddings = []
            with torch.no_grad():
                for seq in seqs:
                    if len(seq) > max_aa_len:
                        print(f"[WARNING] Sequence of length {len(seq)} truncated to {max_aa_len}")
                        seq = seq[:max_aa_len]
                    enc = tokenizer(seq, return_tensors="pt", truncation=True, padding=False)
                    enc = {k: v.to(device) for k, v in enc.items()}
                    
                    try:
                        out = model_obj(**enc, output_hidden_states=True)
                        h = out.hidden_states[-1][0, 1 : 1 + len(seq), :].mean(dim=0).cpu().numpy()
                        embeddings.append(h)
                    except RuntimeError as e:
                        if 'out of memory' in str(e).lower():
                            print(f"[ERROR] GPU out of memory when processing sequence of length {len(seq)}")
                            print(f"[ERROR] Try using a smaller model or moving to CPU with --device cpu")
                            raise
                        else:
                            raise
                        
            emb_arr = np.stack(embeddings, axis=0)
            print(f"[INFO] [{label}] Computed embeddings shape {emb_arr.shape}")

            # save embeddings + IDs
            out_emb_path = args.out_dir / f"{label}_embeddings.npy"
            full_out_emb = out_emb_path if out_emb_path.is_absolute() else root / out_emb_path
            full_out_emb.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(full_out_emb), emb_arr)
            print(f"[INFO] Saved embeddings to {full_out_emb}")

            ids_path = full_out_emb.with_suffix(".ids.txt")
            with open(ids_path, "w") as f:
                f.write("\n".join(ids))
            print(f"[INFO] Saved sequence IDs to {ids_path}")
            
        except Exception as e:
            print(f"[ERROR] Failed to process model {label}: {e}")
            print("[INFO] Continuing with the next model...")
            continue


def main():
    args = parse_args()
    run_embeddings(args)


if __name__ == "__main__":
    main()
