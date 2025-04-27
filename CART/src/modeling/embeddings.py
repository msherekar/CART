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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fasta",
        type=Path,
        required=True,
        help="Input FASTA file with one sequence per record",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model name or path (e.g. facebook/esm2_t6_8M_UR50D or ./fine_tuned_models/high_best)",
    )
    parser.add_argument(
        "--out-emb",
        type=Path,
        default=Path("embeddings.npy"),
        help="Output .npy file to save embeddings",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Compute device to use",
    )
    args = parser.parse_args()

    device = select_device(args.device)
    print(f"[INFO] Using device: {device}")

    # Load tokenizer & model (with hidden states)
    if args.model.startswith("/Users/mukulsherekar"):
        # Assuming local path for finetuned model
        tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D", do_lower_case=False)
        model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D", output_hidden_states=True)
        model.load_state_dict(torch.load(args.model, map_location=device))
    else:
        # Load from Hugging Face Hub
        tokenizer = AutoTokenizer.from_pretrained(args.model, do_lower_case=False)
        model = AutoModelForMaskedLM.from_pretrained(args.model, output_hidden_states=True)
    model.to(device).eval()

    # Correctly determine model's max tokens and compute max sequence length
    max_tokens = min(tokenizer.model_max_length, 512)  # Set a reasonable upper limit
    max_seq_len = max_tokens - 2
    print(f"[INFO] Model max tokens     = {max_tokens}")
    print(f"[INFO] Max amino-acid length = {max_seq_len}")

    embeddings = []
    seq_ids = []

    for rec in SeqIO.parse(str(args.fasta), "fasta"):
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
            outputs = model(
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
    np.save(args.out_emb, embeddings)
    print(f"[INFO] Saved embeddings array {embeddings.shape} to {args.out_emb}")

    # Optionally, save seq_ids for later mapping
    id_file = args.out_emb.with_suffix(".ids.txt")
    with open(id_file, "w") as fh:
        for sid in seq_ids:
            fh.write(f"{sid}\n")
    print(f"[INFO] Saved sequence IDs to {id_file}")

if __name__ == "__main__":
    main()
