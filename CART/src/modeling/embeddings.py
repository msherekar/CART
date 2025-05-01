#!/usr/bin/env python3
"""
extract_embeddings.py
CLS stands for Class token.
Extracts [CLS] embeddings from an ESM-2 model for all sequences in a FASTA,
and saves them as a NumPy array of shape (n_sequences, hidden_size).
Supports CUDA, MPS, or CPU via --device.
Skips any sequence longer than the model's maximum allowable length.
"""
import argparse
import os
import sys

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM
from Bio import SeqIO

# Get the project root directory (3 levels up from this file)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def select_device(choice: str) -> torch.device:
    if choice == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(choice)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fasta",
        type=str,
        default="CART/mutants/CAR_mutants.fasta",
        help="Input FASTA file with one sequence per record (relative to project root)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/esm2_t6_8M_UR50D",
        help="HuggingFace model name or path (e.g. facebook/esm2_t6_8M_UR50D or ./fine_tuned_models/high_best)",
    )
    parser.add_argument(
        "--out_emb",     
        type=str,
        default="CART/embeddings/esm2_t6_8M_UR50D_embeddings.npy",
        help="Output .npy file to save embeddings (relative to project root)",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Compute device to use",
    )
    return parser.parse_args()

def run_embeddings(fasta_path, output_path, model="facebook/esm2_t6_8M_UR50D", device="auto"):
    """
    Extract embeddings from sequences in a FASTA file.
    
    Args:
        fasta_path (str): Path to input FASTA file (relative to project root)
        output_path (str): Path to save output embeddings (.npy) (relative to project root)
        model (str): Model name or path
        device (str): Compute device ("auto", "cuda", "mps", "cpu")
        
    Returns:
        np.ndarray: Embeddings array of shape (n_sequences, hidden_size)
    """
    # Convert relative paths to absolute paths
    fasta_path = os.path.join(project_root, fasta_path)
    output_path = os.path.join(project_root, output_path)
    
    # Print paths for debugging
    print(f"[INFO] Input FASTA: {fasta_path}")
    print(f"[INFO] Output embeddings: {output_path}")
    
    device = select_device(device)
    print(f"[INFO] Using device: {device}")

    # Load tokenizer & model (with hidden states)
    if not model.startswith(("facebook/", "http://", "https://")):
        # Local path
        model_path = model
        # Use default pretrained tokenizer
        tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D", do_lower_case=False)
        model_obj = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D", output_hidden_states=True)
        model_obj.load_state_dict(torch.load(model_path, map_location=device))
    else:
        # Load from Hugging Face Hub
        tokenizer = AutoTokenizer.from_pretrained(model, do_lower_case=False)
        model_obj = AutoModelForMaskedLM.from_pretrained(model, output_hidden_states=True)
    model_obj.to(device).eval()

    # Correctly determine model's max tokens and compute max sequence length
    max_tokens = min(tokenizer.model_max_length, 512)  # Set a reasonable upper limit
    max_seq_len = max_tokens - 2
    print(f"[INFO] Model max tokens     = {max_tokens}")
    print(f"[INFO] Max amino-acid length = {max_seq_len}")

    # Ensure input file exists
    if not os.path.exists(fasta_path):
        raise FileNotFoundError(f"Input FASTA file not found: {fasta_path}")

    embeddings = []
    seq_ids = []

    for rec in SeqIO.parse(fasta_path, "fasta"):
        seq = str(rec.seq)
        L = len(seq)
        if L > max_seq_len:
            print(f"[WARNING] Skipping {rec.id!r}: length {L} > {max_seq_len}")
            continue

        enc = tokenizer(
            seq,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=max_tokens,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model_obj(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        # last hidden layer: (1, max_tokens, H)
        last_hidden = outputs.hidden_states[-1]
        # [CLS] token is at position 0
        cls_emb = last_hidden[:, 0, :].squeeze(0).cpu().numpy()  # (H,)
        embeddings.append(cls_emb)
        seq_ids.append(rec.id)

    if not embeddings:
        raise RuntimeError("No sequences processed; all were too long or FASTA was empty.")

    embeddings = np.vstack(embeddings)  # (n_seqs, H)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    np.save(output_path, embeddings)
    print(f"[INFO] Saved embeddings array {embeddings.shape} to {output_path}")

    # Optionally, save seq_ids for later mapping
    id_file = os.path.splitext(output_path)[0] + ".ids.txt"
    with open(id_file, "w") as fh:
        for sid in seq_ids:
            fh.write(f"{sid}\n")
    print(f"[INFO] Saved sequence IDs to {id_file}")
    
    return embeddings

def main():
    args = parse_args()
    run_embeddings(args.fasta, args.out_emb, args.model, args.device)

if __name__ == "__main__":
    main()
