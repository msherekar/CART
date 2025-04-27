# PSEUDO LOG LIKELIHOOD
# Notes:Make sure car_mutants.fasta contains your 382 mutants in FASTA format.
# Adjust finetuned_high/finetuned_low paths to where you saved your best checkpoints.

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM
from pathlib import Path
from Bio import SeqIO
from tqdm import tqdm
import os
import multiprocessing as mp
from functools import partial
from packaging import version

# — CONFIGURATION —————————————————————————————————————————————————————
MODEL_NAMES = {
    "pretrained": "facebook/esm2_t6_8M_UR50D",
    "finetuned_high": "/Users/mukulsherekar/pythonProject/Finetuning_Activity_Prediction/results/high_best.pth",  # path to your checkpoint
    "finetuned_low":  "/Users/mukulsherekar/pythonProject/Finetuning_Activity_Prediction/results/low_best.pth",
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                      "mps"  if torch.backends.mps.is_available() else
                      "cpu")

# For Mac M1, ensure we're using the right multiprocessing method
if DEVICE.type == "mps":
    # Use fork on Mac for better MPS compatibility
    mp_context = mp.get_context('fork')
else:
    # Use spawn on other platforms for better CUDA compatibility
    mp_context = mp.get_context('spawn')

# Batch processing parameters
BATCH_SIZE = 16  # Adjust based on your memory capacity
NUM_CORES = 8    # Number of cores on Mac M1

# Load models & tokenizers upfront (these will be used in the main process only)
models = {}
tokenizers = {}
for name, path in MODEL_NAMES.items():
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
    
    # Apply torch.compile for model speedup if PyTorch 2.x is available
    if version.parse(torch.__version__) >= version.parse("2.0"):
        print(f"[INFO] Using torch.compile for model acceleration")
        model = torch.compile(model)
    
    tokenizers[name] = tokenizer
    models[name] = model

# Determine model's max tokens and compute max sequence length
max_tokens = min(tokenizer.model_max_length, 512)  # Set a reasonable upper limit
max_seq_len = max_tokens - 2
print(f"[INFO] Model max tokens     = {max_tokens}")
print(f"[INFO] Max amino-acid length = {max_seq_len}")

def compute_pll(sequence, model, tokenizer):
    """Compute pseudo-log-likelihood for a single sequence"""
    # Tokenize the sequence
    enc = tokenizer(sequence,
                   return_tensors="pt",
                   truncation=True,
                   padding=False,
                   max_length=max_tokens)  # Explicitly set max_length
    input_ids = enc.input_ids.to(DEVICE)  # shape [1, L]
    attention_mask = enc.attention_mask.to(DEVICE)
    L = input_ids.size(1)
    
    # For each position, mask & get log-prob of true token
    log_probs = []
    for i in range(L):
        masked = input_ids.clone()
        masked[0, i] = tokenizer.mask_token_id
        with torch.no_grad():
            outputs = model(input_ids=masked, attention_mask=attention_mask)
            logits = outputs.logits  # [1, L, V]
        # Get log softmax at position i
        log_soft = torch.log_softmax(logits[0, i], dim=-1)
        true_id = input_ids[0, i]
        log_p = log_soft[true_id].item()
        log_probs.append(log_p)
    
    return float(np.mean(log_probs))

def compute_pll_batched(sequence, model, tokenizer):
    """Compute pseudo-log-likelihood for a single sequence in a batched manner"""
    enc = tokenizer(sequence, 
                   return_tensors="pt", 
                   truncation=True, 
                   padding=False, 
                   max_length=max_tokens)
    input_ids = enc.input_ids[0]
    L = input_ids.shape[0]

    # Build [L x L] masked input where each row has one masked token
    masked_input = input_ids.repeat(L, 1)
    masked_input[torch.arange(L), torch.arange(L)] = tokenizer.mask_token_id

    masked_input = masked_input.to(DEVICE)
    with torch.no_grad():
        logits = model(masked_input).logits  # [L, L, V]

    log_probs = torch.log_softmax(logits[torch.arange(L), torch.arange(L)], dim=-1)
    true_ids = input_ids.to(DEVICE)
    pll = log_probs[torch.arange(L), true_ids].mean().item()
    return pll

def process_chunk(chunk_data):
    """Process a chunk of sequences for a given model"""
    chunk_id, sequences, model_name = chunk_data
    results = []
    
    # Use the global models and tokenizers
    model = models[model_name]
    tokenizer = tokenizers[model_name]
    
    for seq in sequences:
        try:
            # Use the batched PLL calculation for speed
            pll = compute_pll_batched(seq, model, tokenizer)
            results.append(pll)
        except Exception as e:
            print(f"Error in chunk {chunk_id} processing sequence: {e}")
            # Fallback to regular PLL if batched version fails
            try:
                pll = compute_pll(seq, model, tokenizer)
                results.append(pll)
                print(f"  -> Recovered using non-batched method")
            except Exception as e2:
                print(f"  -> Failed with fallback method: {e2}")
                results.append(0.0)
    
    return results

# — MAIN —————————————————————————————————————————————————————————————
# load your 382 mutant sequences (FASTA or list)
mutant_fasta = Path("/Users/mukulsherekar/pythonProject/Finetuning_Activity_Prediction/CART_AI_SYSTEM/mutants/CAR_mutants.fasta")
sequences = [str(rec.seq) for rec in SeqIO.parse(mutant_fasta, "fasta")]

# Filter sequences that exceed max length
sequences = [seq for seq in sequences if len(seq) <= max_seq_len]
print(f"[INFO] Processing {len(sequences)} sequences")

# For each model, compute PLLs
pll_results = {name: [] for name in MODEL_NAMES}

for name in MODEL_NAMES:
    print(f"Computing PLLs with {name}...")
    
    # Divide work into chunks (one chunk per worker)
    chunk_size = max(1, len(sequences) // NUM_CORES)
    chunks = []
    for i in range(0, len(sequences), chunk_size):
        chunk_seqs = sequences[i:i+chunk_size]
        chunks.append((i//chunk_size, chunk_seqs, name))
    
    # Single-process mode for testing/debugging
    if False:  # Set to True to debug without multiprocessing
        all_results = []
        for chunk in tqdm(chunks):
            results = process_chunk(chunk)
            all_results.extend(results)
    else:
        # Process chunks in parallel
        with mp_context.Pool(processes=min(NUM_CORES, len(chunks))) as pool:
            all_results = []
            for results in tqdm(pool.imap(process_chunk, chunks), total=len(chunks)):
                all_results.extend(results)
    
    pll_results[name] = all_results
    print(f"[INFO] Completed model {name}")

# average across all mutants
for name, values in pll_results.items():
    avg_pll = np.mean(values)
    print(f"{name:>16} average PLL: {avg_pll:.4f}")