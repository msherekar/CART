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
import matplotlib.pyplot as plt
import pandas as pd



def parse_args(args_list=None):
    parser = argparse.ArgumentParser(description="Compute pseudo-log-likelihoods for CAR-T sequences")
    
    # Get project root for relative paths
    #project_root = Path(__file__).parent.parent.parent.resolve()
    
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
    
    # Optional arguments
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
        help="Batch size for inference"
    )
    parser.add_argument(
        "--save_interval", 
        type=int, 
        default=10, 
        help="Save interval for checkpoints"
    )
    parser.add_argument(
        "--max_tokens", 
        type=int, 
        default=512, 
        help="Maximum number of tokens per sequence"
    )
    parser.add_argument(
        "--output_dir", 
        type=Path, 
        default=Path('../../output/results'), 
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
        help="Number of sequences to use for testing"
    )
    
    # Only parse command line arguments if this module is run directly
    if args_list is None and __name__ == "__main__":
        return parser.parse_args()
    else:
        # When imported, use the provided args_list or an empty list
        return parser.parse_args(args_list or [])

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

def plot_pll_results(results_dict, output_path):
    """Plot bar graph comparing PLL scores across models and plot perplexity."""
    plt.figure(figsize=(10, 6))
    
    # Prepare data for plotting
    models = list(results_dict.keys())
    avg_plls = [np.nanmean(results_dict[model]) for model in models]
    
    # Compute perplexity for each model
    perplexities = [np.exp(-pll) for pll in avg_plls]
    
    # Save PLL data as CSV for custom plotting
    results_dir = Path(os.path.dirname(output_path))
    csv_path = results_dir / "pll_scores.csv"
    
    # Create DataFrame for all sequence scores
    seq_data = {}
    for model in models:
        seq_data[model] = results_dict[model]
    
    # Ensure all columns have the same length by padding with NaN
    max_len = max(len(scores) for scores in seq_data.values())
    for model, scores in seq_data.items():
        if len(scores) < max_len:
            seq_data[model] = scores + [np.nan] * (max_len - len(scores))
    
    # Save sequence-level scores
    seq_df = pd.DataFrame(seq_data)
    seq_df.to_csv(csv_path, index=False)
    print(f"[INFO] Sequence-level PLL scores saved to {csv_path}")
    
    # Save summary statistics including perplexity
    summary_df = pd.DataFrame({
        'model': models,
        'avg_pll': avg_plls,
        'perplexity': perplexities,
        'min_pll': [np.nanmin(results_dict[model]) for model in models],
        'max_pll': [np.nanmax(results_dict[model]) for model in models],
        'std_pll': [np.nanstd(results_dict[model]) for model in models],
    })
    summary_path = results_dir / "pll_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"[INFO] PLL summary statistics (with perplexity) saved to {summary_path}")
    
    # Create bar plot for PLL
    bars = plt.bar(models, avg_plls, color=['skyblue', 'lightgreen', 'salmon'])
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom')
    plt.title('Average Pseudo-Log-Likelihood Scores by Model')
    plt.ylabel('Average PLL Score')
    plt.xlabel('Model')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(results_dir / 'pll_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] PLL plot saved to {results_dir / 'pll_comparison.png'}")
    
    # Create bar plot for Perplexity
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, perplexities, color=['skyblue', 'lightgreen', 'salmon'])
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    plt.title('Perplexity by Model (lower is better)')
    plt.ylabel('Perplexity')
    plt.xlabel('Model')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(results_dir / 'perplexity_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Perplexity plot saved to {results_dir / 'perplexity_comparison.png'}")

def load_model_and_tokenizer(model_name, device):
    """Load model and tokenizer based on model name"""
    if model_name == "pretrained":
        # Load ESM model from Hugging Face
        model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D")
        tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    else:
        # Load finetuned model
        if model_name == "finetuned_high":
            path = os.path.join(get_project_root(), "output", "models", "high")
        elif model_name == "finetuned_low":
            path = os.path.join(get_project_root(), "output", "models", "low")
        else:
            raise ValueError(f"Unknown model name: {model_name}")
            
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model checkpoint not found: {path}")
            
        print(f"Loading model from {path}")
            
        # Load the model directly using from_pretrained
        model = AutoModelForMaskedLM.from_pretrained(path)
        tokenizer = AutoTokenizer.from_pretrained(path)
    
    # Move model to device after loading
    model = model.to(device)
    return model, tokenizer

def compute_plls(sequences, model, tokenizer, device, batch_size, max_tokens):
    """Compute pseudo-log-likelihoods for a batch of sequences."""
    plls = []
    for seq in sequences:
        # Tokenize the sequence
        enc = tokenizer(
            seq,
            return_tensors="pt",
            truncation=True,
            padding=False,
            max_length=max_tokens
        )
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
        
        plls.append(float(np.mean(log_probs)))
    return plls

def run_pll(args):
    root = Path(__file__).parent.parent.parent.parent.resolve()
   # now resolve the FASTA path:
    mutant_fasta = args.mutant_fasta
    if not mutant_fasta.is_absolute():
        mutant_fasta = root / mutant_fasta

    # same for output_dir, finetuned_high, etc.
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = root / output_dir

    
    """Run PLL computation pipeline."""
    # Select device
    device = select_device(args.device)
    print(f"Using device: {device}")
    
    # Load sequences
    sequences = []
    sequence_ids = []
    for record in SeqIO.parse(mutant_fasta, "fasta"):
        sequences.append(str(record.seq))
        sequence_ids.append(record.id)
    print(f"Loaded {len(sequences)} sequences from {mutant_fasta}")
    
    # Process with each model
    pll_results = {}
    for model_name in ['pretrained', 'finetuned_high', 'finetuned_low']:
        print(f"\n[INFO] Processing with {model_name} model")
        print(f"[INFO] Loading {model_name} model and tokenizer...")
        
        try:
            if model_name == 'pretrained':
                model_path = args.pretrained
                print(f"Loading pretrained model from {model_path}")
            elif model_name == 'finetuned_high':
                model_path = args.finetuned_high
                print(f"Loading finetuned high-diversity model from {model_path}")
            else:
                model_path = args.finetuned_low
                print(f"Loading finetuned low-diversity model from {model_path}")
            if os.path.exists(model_path):
                print(f"Loading model from {model_path}")
                # Load the model directly using from_pretrained
                model = AutoModelForMaskedLM.from_pretrained(model_path)
                tokenizer = AutoTokenizer.from_pretrained(model_path)
            else:
                print(f"Loading model from Hugging Face Hub: {model_path}")
                model = AutoModelForMaskedLM.from_pretrained(model_path)
                tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            model = model.to(device)
            model.eval()
            
            # Compute PLLs using the existing compute_pll function
            plls = []
            for seq in tqdm(sequences, desc=f"Computing PLL for {model_name}"):
                pll = compute_pll(seq, model, tokenizer, device, args.max_tokens)
                plls.append(pll)
            
            pll_results[model_name] = plls
            
            # Save results as numpy array
            output_path = os.path.join(args.output_dir, f"{model_name}_pll.npy")
            os.makedirs(args.output_dir, exist_ok=True)
            np.save(output_path, plls)
            print(f"Saved PLL results to {output_path}")
            
            # Save results as CSV with sequence IDs
            csv_path = os.path.join(args.output_dir, f"{model_name}_pll.csv")
            pll_df = pd.DataFrame({
                'sequence_id': sequence_ids,
                'pll_score': plls
            })
            pll_df.to_csv(csv_path, index=False)
            print(f"Saved PLL results as CSV to {csv_path}")
            
        except Exception as e:
            print(f"Warning: Could not load model with CPU map_location: {str(e)}")
            print(f"Error loading model: {str(e)}")
            print(f"Error processing {model_name}: {str(e)}")
            continue
    
    # Plot comparison if we have results
    if pll_results:
        plot_path = root / 'output' / 'results' / 'pll_comparison.png'
        plot_pll_results(pll_results, plot_path)

def main():
    args = parse_args()
    run_pll(args)

if __name__ == "__main__":
    main()