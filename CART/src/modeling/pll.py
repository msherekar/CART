#!/usr/bin/env python3
"""
Compute pseudo-log-likelihood scores for protein sequences using ESM2 models and finetuned models.
PLL scores indicate how well different models can predict masked amino acids in the sequence.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM
from pathlib import Path
from Bio import SeqIO
from tqdm import tqdm
import time
import os
import json
import argparse

def parse_args():
    # Define project root for relative paths
    project_root = Path(__file__).resolve().parents[3]  # Go up 3 levels from this file
    
    parser = argparse.ArgumentParser(
        description="Compute pseudo-log-likelihood scores for protein sequences"
    )
    
    # Input file
    parser.add_argument(
        "--mutant_fasta", 
        type=Path, 
        default=project_root / "CART/mutants/CAR_mutants.fasta", 
        help="FASTA file containing sequences to evaluate"
    )
    
    # Model paths
    parser.add_argument(
        "--pretrained", 
        type=str, 
        default="facebook/esm2_t6_8M_UR50D", 
        help="Pretrained model name or path"
    )
    parser.add_argument(
        "--finetuned_high", 
        type=Path, 
        default=project_root / "checkpoints/high_best.pth", 
        help="Path to high-diversity fine-tuned model"
    )
    parser.add_argument(
        "--finetuned_low", 
        type=Path, 
        default=project_root / "checkpoints/low_best.pth", 
        help="Path to low-diversity fine-tuned model"
    )
    
    # Processing options
    parser.add_argument(
        "--device", 
        type=str, 
        choices=["auto", "cuda", "mps", "cpu"], 
        default="auto", 
        help="Compute device to use"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=4, 
        help="Batch size for processing sequences"
    )
    parser.add_argument(
        "--save_interval", 
        type=int, 
        default=10, 
        help="Save results every N sequences"
    )
    parser.add_argument(
        "--max_tokens", 
        type=int, 
        default=512, 
        help="Maximum tokens for tokenizer"
    )
    parser.add_argument(
        "--output_dir", 
        type=Path, 
        default=project_root / "results", 
        help="Directory to save results"
    )
    parser.add_argument(
        "--use_subset", 
        action="store_true", 
        help="Use a subset of sequences for testing"
    )
    parser.add_argument(
        "--test_size", 
        type=int, 
        default=20, 
        help="Number of sequences to use when use_subset is True"
    )
    
    return parser.parse_args()

def select_device(choice: str) -> torch.device:
    if choice == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(choice)

def compute_pll(sequence, model, tokenizer, device, max_tokens):
    """Compute pseudo-log-likelihood for a single sequence - standard method"""
    # Tokenize the sequence
    enc = tokenizer(sequence,
                   return_tensors="pt",
                   truncation=True,
                   padding=False,
                   max_length=max_tokens)
    input_ids = enc.input_ids.to(device)
    attention_mask = enc.attention_mask.to(device)
    L = input_ids.size(1)
    
    # For each position, mask & get log-prob of true token
    log_probs = []
    for i in range(L):
        masked = input_ids.clone()
        masked[0, i] = tokenizer.mask_token_id
        with torch.no_grad():
            outputs = model(input_ids=masked, attention_mask=attention_mask)
            logits = outputs.logits
        log_soft = torch.log_softmax(logits[0, i], dim=-1)
        true_id = input_ids[0, i]
        log_p = log_soft[true_id].item()
        log_probs.append(log_p)
    
    return float(np.mean(log_probs))

def save_results(results, model_name, seq_idx, total_count, output_dir):
    """Save results to disk with progress information"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"pll_results_{model_name}.npy"
    progress_file = output_dir / f"progress_{model_name}.json"
    
    # Save numpy array with results
    np.save(output_file, np.array(results))
    
    # Save progress information
    progress = {
        "model": model_name,
        "completed": seq_idx,
        "total": total_count,
        "average_pll": float(np.mean(results)),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(progress_file, "w") as f:
        json.dump(progress, f, indent=2)
    
    print(f"[SAVE] Results saved: {seq_idx}/{total_count} sequences processed")

def load_progress(model_name, total_sequences, output_dir):
    """Load previous progress if available"""
    output_dir = Path(output_dir)
    output_file = output_dir / f"pll_results_{model_name}.npy"
    progress_file = output_dir / f"progress_{model_name}.json"
    
    if os.path.exists(output_file) and os.path.exists(progress_file):
        try:
            # Load progress info
            with open(progress_file, "r") as f:
                progress = json.load(f)
            
            # Load saved results
            results = np.load(output_file).tolist()
            
            # Verify count matches
            if len(results) == progress["completed"]:
                print(f"[INFO] Resuming from previous run: {len(results)}/{total_sequences} sequences already processed")
                return results, progress["completed"]
            else:
                print(f"[WARNING] Inconsistent saved data. Starting fresh.")
        except Exception as e:
            print(f"[WARNING] Error loading previous progress: {e}")
    
    # Start fresh
    return [], 0

def run_pll(args):
    """Main function to compute PLL scores for sequences with different models"""
    # Set up device
    device = select_device(args.device)
    print(f"[INFO] Device: {device}")
    
    # Calculate max sequence length based on max tokens
    max_seq_len = args.max_tokens - 2
    print(f"[INFO] Model max tokens = {args.max_tokens}")
    print(f"[INFO] Max amino-acid length = {max_seq_len}")
    
    # Make sure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create model paths dictionary
    model_paths = {
        "pretrained": args.pretrained,
        "finetuned_high": str(args.finetuned_high),
        "finetuned_low": str(args.finetuned_low),
    }
    
    # Load all sequences
    print("[INFO] Loading sequences...")
    sequences = [str(rec.seq) for rec in SeqIO.parse(args.mutant_fasta, "fasta")]

    # Filter sequences that exceed max length
    sequences = [seq for seq in sequences if len(seq) <= max_seq_len]
    print(f"[INFO] Total sequences: {len(sequences)}")
    
    # Use subset for testing if enabled
    if args.use_subset:
        test_size = min(args.test_size, len(sequences))
        print(f"[INFO] Using subset of {test_size} sequences for testing")
        sequences = sequences[:test_size]
    
    # Process each model
    for name, path in model_paths.items():
        print(f"\n[INFO] Processing with {name} model")
        
        # Check for previous progress
        results, start_idx = load_progress(name, len(sequences), args.output_dir)
        
        # Skip if already completed
        if start_idx >= len(sequences):
            print(f"[INFO] Model {name} already completed. Skipping.")
            continue
        
        # Load model and tokenizer
        print(f"[INFO] Loading {name} model and tokenizer...")
        start_time = time.time()
        
        if name == "pretrained":
            # Load pretrained model and tokenizer from Hugging Face Hub
            tokenizer = AutoTokenizer.from_pretrained(path, do_lower_case=False)
            model = AutoModelForMaskedLM.from_pretrained(path)
        else:
            # Load tokenizer from the pretrained model
            tokenizer = AutoTokenizer.from_pretrained(model_paths["pretrained"], do_lower_case=False)
            # Load model weights from local checkpoint
            model = AutoModelForMaskedLM.from_pretrained(model_paths["pretrained"])
            model.load_state_dict(torch.load(path, map_location=device))
        
        model.to(device).eval()
        print(f"[INFO] Model loaded in {time.time() - start_time:.1f} seconds")
        
        # Calculate estimated time
        if start_idx > 0:
            # Calculate average processing time from previous test
            avg_time_per_seq = 23.0  # Based on previous test run showing ~23 seconds per sequence
            est_time = avg_time_per_seq * (len(sequences) - start_idx) / 60.0
            print(f"[INFO] Estimated time remaining: {est_time:.1f} minutes ({est_time/60:.1f} hours)")
        
        # Process sequences in batches from the starting point
        for i in range(start_idx, len(sequences), args.batch_size):
            batch = sequences[i:i+args.batch_size]
            print(f"[INFO] Processing batch {i//args.batch_size + 1}/{(len(sequences)+args.batch_size-1)//args.batch_size}")
            
            for j, seq in enumerate(batch):
                seq_idx = i + j
                print(f"  Processing sequence {seq_idx+1}/{len(sequences)}")
                
                try:
                    seq_start_time = time.time()
                    pll = compute_pll(seq, model, tokenizer, device, args.max_tokens)
                    duration = time.time() - seq_start_time
                    results.append(pll)
                    print(f"  ✓ Sequence {seq_idx+1}: PLL = {pll:.4f} ({duration:.1f}s, length: {len(seq)})")
                except Exception as e:
                    print(f"  ✗ Error processing sequence {seq_idx+1}: {e}")
                    results.append(float('nan'))  # Store NaN for failed sequences
                
                # Save periodically
                current_idx = seq_idx + 1
                if current_idx % args.save_interval == 0 or current_idx == len(sequences):
                    save_results(results, name, current_idx, len(sequences), args.output_dir)
        
        # Calculate final average for this model
        avg_pll = np.nanmean(results)  # Use nanmean to ignore NaN values
        print(f"\n{name:>16} average PLL: {avg_pll:.4f}")
        
        # Clear memory
        del model, tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Print final summary
    print("\nFinal Results:")
    for name in model_paths:
        output_file = output_dir / f"pll_results_{name}.npy"
        if os.path.exists(output_file):
            results = np.load(output_file)
            avg_pll = np.nanmean(results)
            print(f"{name:>16} average PLL: {avg_pll:.4f}")

def main():
    args = parse_args()
    run_pll(args)

if __name__ == "__main__":
    main()