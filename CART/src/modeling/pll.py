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

def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def parse_args():
    # Define project root for relative paths
    project_root = get_project_root()
    
    parser = argparse.ArgumentParser(
        description="Compute pseudo-log-likelihood scores for protein sequences"
    )
    
    # Input file
    parser.add_argument(
        "--mutant_fasta", 
        type=str, 
        default=os.path.join(project_root, "CART", "mutants", "CAR_mutants.fasta"), 
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
        type=str, 
        default=os.path.join(project_root, "CART", "checkpoints", "high_best.pth"), 
        help="Path to high-diversity fine-tuned model"
    )
    parser.add_argument(
        "--finetuned_low", 
        type=str, 
        default=os.path.join(project_root, "CART", "checkpoints", "low_best.pth"), 
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
        type=str, 
        default=os.path.join(project_root, "CART", "results"), 
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

def plot_pll_results(results_dict, output_dir):
    """Plot bar graph comparing PLL scores across models"""
    plt.figure(figsize=(10, 6))
    
    # Prepare data for plotting
    models = list(results_dict.keys())
    avg_plls = [np.nanmean(results_dict[model]) for model in models]
    
    # Create bar plot
    bars = plt.bar(models, avg_plls, color=['skyblue', 'lightgreen', 'salmon'])
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom')
    
    plt.title('Average Pseudo-Log-Likelihood Scores by Model')
    plt.ylabel('Average PLL Score')
    plt.xlabel('Model')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save the plot
    plot_path = Path(output_dir) / 'pll_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Plot saved to {plot_path}")

def load_model_and_tokenizer(model_name, device):
    """Load model and tokenizer based on model name"""
    if model_name == "pretrained":
        # Load ESM model from Hugging Face
        model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D")
        tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    else:
        # Load finetuned model
        if model_name == "finetuned_high":
            path = os.path.join(get_project_root(), "checkpoints", "high_best.pth")
        elif model_name == "finetuned_low":
            path = os.path.join(get_project_root(), "checkpoints", "low_best.pth")
        else:
            raise ValueError(f"Unknown model name: {model_name}")
            
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model checkpoint not found: {path}")
            
        print(f"Loading model from {path}")
            
        # First load the model architecture
        model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D")
        tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
        
        # Then load the state dict with proper device handling
        try:
            # Try loading directly to CPU first
            state_dict = torch.load(path, map_location='cpu')
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Warning: Could not load model with CPU map_location: {e}")
            try:
                # Try loading without map_location
                state_dict = torch.load(path)
                model.load_state_dict(state_dict)
            except Exception as e:
                print(f"Error loading model: {e}")
                raise
    
    # Move model to device after loading
    model = model.to(device)
    return model, tokenizer

def run_pll(args):
    """Run PLL computation for all models"""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load FASTA file
    fasta_path = args.mutant_fasta
    if not os.path.exists(fasta_path):
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")
    
    sequences = [str(rec.seq) for rec in SeqIO.parse(fasta_path, "fasta")]
    print(f"Loaded {len(sequences)} sequences from {fasta_path}")
    
    # Process each model
    all_results = {}
    for model_name in ["pretrained", "finetuned_high", "finetuned_low"]:
        print(f"\n[INFO] Processing with {model_name} model")
        print(f"[INFO] Loading {model_name} model and tokenizer...")
        
        try:
            model, tokenizer = load_model_and_tokenizer(model_name, device)
            results = []
            for seq in sequences:
                pll = compute_pll(seq, model, tokenizer, device, args.max_tokens)
                results.append(pll)
            
            all_results[model_name] = results
            
            # Save results
            output_path = os.path.join(args.output_dir, f"{model_name}_pll.npy")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            np.save(output_path, results)
            print(f"Saved PLL results to {output_path}")
            
        except Exception as e:
            print(f"Error processing {model_name}: {e}")
            continue
    
    # Generate plots
    if all_results:
        plot_pll_results(all_results, args.output_dir)
    else:
        print("No results to plot")

def main():
    args = parse_args()
    run_pll(args)

if __name__ == "__main__":
    main()