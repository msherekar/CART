# PSEUDO LOG LIKELIHOOD - FULL VERSION
# Process all sequences with periodic saving

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM
from pathlib import Path
from Bio import SeqIO
from tqdm import tqdm
import time
import os
import json

# — CONFIGURATION —————————————————————————————————————————————————————
MODEL_NAMES = {
    "pretrained": "facebook/esm2_t6_8M_UR50D",
    "finetuned_high": "/Users/mukulsherekar/pythonProject/Finetuning_Activity_Prediction/results/high_best.pth",
    "finetuned_low":  "/Users/mukulsherekar/pythonProject/Finetuning_Activity_Prediction/results/low_best.pth",
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                      "mps"  if torch.backends.mps.is_available() else
                      "cpu")

# Process all sequences
USE_SUBSET = False
TEST_SIZE = 20  # Only used if USE_SUBSET is True

# Set batch size for processing multiple sequences at once
BATCH_SIZE = 4  # Small batch size to reduce memory issues
# Save results periodically (every N sequences)
SAVE_INTERVAL = 10

# Set maximum sequence length
max_tokens = 512  # Set reasonable upper limit
max_seq_len = max_tokens - 2

def compute_pll(sequence, model, tokenizer):
    """Compute pseudo-log-likelihood for a single sequence - standard method"""
    # Tokenize the sequence
    enc = tokenizer(sequence,
                   return_tensors="pt",
                   truncation=True,
                   padding=False,
                   max_length=max_tokens)
    input_ids = enc.input_ids.to(DEVICE)
    attention_mask = enc.attention_mask.to(DEVICE)
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

def save_results(results, model_name, seq_idx, total_count):
    """Save results to disk with progress information"""
    output_file = f"pll_results_{model_name}.npy"
    progress_file = f"progress_{model_name}.json"
    
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

def load_progress(model_name, total_sequences):
    """Load previous progress if available"""
    output_file = f"pll_results_{model_name}.npy"
    progress_file = f"progress_{model_name}.json"
    
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

def main():
    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Model max tokens = {max_tokens}")
    print(f"[INFO] Max amino-acid length = {max_seq_len}")
    
    # Load all sequences
    print("[INFO] Loading sequences...")
    mutant_fasta = Path("/Users/mukulsherekar/pythonProject/Finetuning_Activity_Prediction/CART_AI_SYSTEM/mutants/CAR_mutants.fasta")
    sequences = [str(rec.seq) for rec in SeqIO.parse(mutant_fasta, "fasta")]

    # Filter sequences that exceed max length
    sequences = [seq for seq in sequences if len(seq) <= max_seq_len]
    print(f"[INFO] Total sequences: {len(sequences)}")
    
    # Use subset for testing if enabled
    if USE_SUBSET:
        test_size = min(TEST_SIZE, len(sequences))
        print(f"[INFO] Using subset of {test_size} sequences for testing")
        sequences = sequences[:test_size]
    
    # Process each model
    for name, path in MODEL_NAMES.items():
        print(f"\n[INFO] Processing with {name} model")
        
        # Check for previous progress
        results, start_idx = load_progress(name, len(sequences))
        
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
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAMES["pretrained"], do_lower_case=False)
            # Load model weights from local checkpoint
            model = AutoModelForMaskedLM.from_pretrained(MODEL_NAMES["pretrained"])
            model.load_state_dict(torch.load(path, map_location=DEVICE))
        
        model.to(DEVICE).eval()
        print(f"[INFO] Model loaded in {time.time() - start_time:.1f} seconds")
        
        # Calculate estimated time
        if start_idx > 0:
            # Calculate average processing time from previous test
            avg_time_per_seq = 23.0  # Based on your test run showing ~23 seconds per sequence
            est_time = avg_time_per_seq * (len(sequences) - start_idx) / 60.0
            print(f"[INFO] Estimated time remaining: {est_time:.1f} minutes ({est_time/60:.1f} hours)")
        
        # Process sequences in batches from the starting point
        for i in range(start_idx, len(sequences), BATCH_SIZE):
            batch = sequences[i:i+BATCH_SIZE]
            print(f"[INFO] Processing batch {i//BATCH_SIZE + 1}/{(len(sequences)+BATCH_SIZE-1)//BATCH_SIZE}")
            
            for j, seq in enumerate(batch):
                seq_idx = i + j
                print(f"  Processing sequence {seq_idx+1}/{len(sequences)}")
                
                try:
                    seq_start_time = time.time()
                    pll = compute_pll(seq, model, tokenizer)
                    duration = time.time() - seq_start_time
                    results.append(pll)
                    print(f"  ✓ Sequence {seq_idx+1}: PLL = {pll:.4f} ({duration:.1f}s, length: {len(seq)})")
                except Exception as e:
                    print(f"  ✗ Error processing sequence {seq_idx+1}: {e}")
                    results.append(float('nan'))  # Store NaN for failed sequences
                
                # Save periodically
                current_idx = seq_idx + 1
                if current_idx % SAVE_INTERVAL == 0 or current_idx == len(sequences):
                    save_results(results, name, current_idx, len(sequences))
        
        # Calculate final average for this model
        avg_pll = np.nanmean(results)  # Use nanmean to ignore NaN values
        print(f"\n{name:>16} average PLL: {avg_pll:.4f}")
        
        # Clear memory
        del model, tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Print final summary
    print("\nFinal Results:")
    for name in MODEL_NAMES:
        output_file = f"pll_results_{name}.npy"
        if os.path.exists(output_file):
            results = np.load(output_file)
            avg_pll = np.nanmean(results)
            print(f"{name:>16} average PLL: {avg_pll:.4f}")

if __name__ == "__main__":
    main()