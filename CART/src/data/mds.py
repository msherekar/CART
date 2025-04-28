#!/usr/bin/env python3
"""
MDS visualization of CAR sequences

Generates a 2D MDS plot to visualize the difference between high-diversity 
and low-diversity CAR sequences using one-hot encoding and multidimensional scaling.
Also generates plots for sequence length distribution and Levenshtein distances.
"""

import random
import numpy as np
import argparse
from pathlib import Path
from Bio import SeqIO
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Union
import seaborn as sns
from Levenshtein import distance as levenshtein_distance
import matplotlib.gridspec as gridspec

# Use relative paths instead of absolute paths
def get_project_root() -> Path:
    """Get project root directory relative to this file"""
    return Path(__file__).parent.parent.parent.parent

# --- default parameters ---
PROJECT_ROOT = get_project_root()
DEFAULT_HIGH_FASTA = "CART/homologs/high_diversity.fasta"
DEFAULT_LOW_FASTA = "CART/homologs/low_diversity.fasta"
DEFAULT_OUTPUT_DIR = "plots"
DEFAULT_SAMPLE_FRAC = 1  # 100
DEFAULT_RANDOM_STATE = 42

# --- one-hot setup ---
AA_LIST = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}

def one_hot_padded(sequences: List[str]) -> Tuple[np.ndarray, int]:
    """
    Convert sequences to one-hot encodings with padding to make all sequences the same length.
    
    Args:
        sequences: List of amino acid sequences
        
    Returns:
        Tuple of (one-hot encoded sequences array, max_length)
    """
    # Find max length to ensure consistent dimensions
    max_length = max(len(seq) for seq in sequences)
    print(f"Max sequence length: {max_length}")
    
    # Create one-hot encodings
    one_hot_vectors = []
    for seq in sequences:
        # Create matrix of zeros with shape (max_length, len(AA_LIST))
        one_hot = np.zeros((max_length, len(AA_LIST)), dtype=int)
        
        # Fill with one-hot encodings for each amino acid in the sequence
        for i, aa in enumerate(seq[:max_length]):  # Truncate if longer than max_length
            idx = AA_TO_IDX.get(aa)
            if idx is not None:
                one_hot[i, idx] = 1
                
        # Flatten and add to list
        one_hot_vectors.append(one_hot.flatten())
    
    return np.vstack(one_hot_vectors), max_length

def load_and_sample(fasta_path: Path, frac: float, seed: int) -> Tuple[List[str], Optional[str]]:
    """
    Load and sample sequences from a FASTA file
    
    Args:
        fasta_path: Path to the FASTA file
        frac: Fraction of sequences to sample
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (sampled sequences, wild-type sequence)
    """
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")
        
    seqs = [str(rec.seq) for rec in SeqIO.parse(fasta_path, "fasta")]
    print(f"Loaded {len(seqs)} sequences from {fasta_path}")
    
    # If this is the first sequence (assuming it's the wild-type)
    wild_type = seqs[0] if seqs else None
    
    random.Random(seed).shuffle(seqs)
    n = max(1, int(len(seqs) * frac))
    
    # Make sure wild_type is included in the sampled sequences
    sampled = seqs[:n]
    if wild_type and wild_type not in sampled:
        sampled[0] = wild_type
        
    return sampled, wild_type

def plot_mds_with_kde(high_emb, low_emb, wt_emb, output_path):
    """Plot MDS with KDE contours and wild-type marker"""
    plt.figure(figsize=(10, 8))
    
    # Plot KDE contours
    for data, color, label in [(high_emb, 'orange', 'High-diversity'), 
                              (low_emb, 'blue', 'Low-diversity')]:
        sns.kdeplot(x=data[:, 0], y=data[:, 1], 
                   levels=5, fill=True, alpha=0.3, 
                   color=color, label=f"{label} KDE")
    
    # Plot scatter points
    plt.scatter(high_emb[:, 0], high_emb[:, 1], c='orange', 
               label='High-diversity', alpha=0.6, edgecolor='k', linewidth=0.5)
    plt.scatter(low_emb[:, 0], low_emb[:, 1], c='blue', 
               label='Low-diversity', alpha=0.6, edgecolor='k', linewidth=0.5)
    
    # Plot wild-type as yellow star
    if wt_emb is not None:
        plt.scatter(wt_emb[0], wt_emb[1], c='yellow', marker='*', s=300, 
                   label='Wild-type', edgecolor='k', linewidth=1, zorder=10)
    
    plt.title("(A) MDS of CAR Sequences")
    plt.xlabel("MDS Dimension 1")
    plt.ylabel("MDS Dimension 2")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_sequence_lengths(high_seqs, low_seqs, output_path):
    """Plot distribution of sequence lengths"""
    high_lengths = [len(seq) for seq in high_seqs]
    low_lengths = [len(seq) for seq in low_seqs]
    
    plt.figure(figsize=(10, 6))
    
    # Combine data for histogram
    all_lengths = high_lengths + low_lengths
    bins = np.linspace(min(all_lengths), max(all_lengths), 30)
    
    plt.hist(high_lengths, bins=bins, alpha=0.7, color='orange', label='High-diversity')
    plt.hist(low_lengths, bins=bins, alpha=0.7, color='blue', label='Low-diversity')
    
    plt.title("(B) Distribution of Sequence Lengths")
    plt.xlabel("Sequence Length")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_levenshtein_distances(high_seqs, low_seqs, wild_type, output_path):
    """Plot Levenshtein distances from wild-type sequence"""
    if not wild_type:
        print("Warning: Wild-type sequence not available, skipping Levenshtein plot")
        return
    
    high_distances = [levenshtein_distance(wild_type, seq) for seq in high_seqs]
    low_distances = [levenshtein_distance(wild_type, seq) for seq in low_seqs]
    
    plt.figure(figsize=(10, 6))
    
    # Combine data for histogram
    all_distances = high_distances + low_distances
    bins = np.linspace(min(all_distances), max(all_distances), 30)
    
    plt.hist(high_distances, bins=bins, alpha=0.7, color='orange', label='High-diversity')
    plt.hist(low_distances, bins=bins, alpha=0.7, color='blue', label='Low-diversity')
    
    plt.title("(C) Levenshtein Distances from Wild-type Sequence")
    plt.xlabel("Levenshtein Distance")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300)
    plt.close()

def create_combined_plot(high_seqs, low_seqs, high_emb, low_emb, wt_emb, wild_type, output_path):
    """Create a combined figure with all three plots"""
    fig = plt.figure(figsize=(18, 6))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.2, 1, 1])
    
    # Plot A: MDS with KDE
    ax1 = plt.subplot(gs[0])
    
    # Plot KDE contours
    for data, color, label in [(high_emb, 'orange', 'High-diversity'), 
                              (low_emb, 'blue', 'Low-diversity')]:
        sns.kdeplot(x=data[:, 0], y=data[:, 1], 
                   levels=5, fill=True, alpha=0.3, 
                   color=color, label=f"{label} KDE", ax=ax1)
    
    # Plot scatter points
    ax1.scatter(high_emb[:, 0], high_emb[:, 1], c='orange', 
               label='High-diversity', alpha=0.6, edgecolor='k', linewidth=0.5)
    ax1.scatter(low_emb[:, 0], low_emb[:, 1], c='blue', 
               label='Low-diversity', alpha=0.6, edgecolor='k', linewidth=0.5)
    
    # Plot wild-type as yellow star
    if wt_emb is not None:
        ax1.scatter(wt_emb[0], wt_emb[1], c='yellow', marker='*', s=300, 
                   label='Wild-type', edgecolor='k', linewidth=1, zorder=10)
    
    ax1.set_title("(A) MDS of CAR Sequences")
    ax1.set_xlabel("MDS Dimension 1")
    ax1.set_ylabel("MDS Dimension 2")
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot B: Sequence lengths
    ax2 = plt.subplot(gs[1])
    high_lengths = [len(seq) for seq in high_seqs]
    low_lengths = [len(seq) for seq in low_seqs]
    
    # Combine data for histogram
    all_lengths = high_lengths + low_lengths
    bins = np.linspace(min(all_lengths), max(all_lengths), 30)
    
    ax2.hist(high_lengths, bins=bins, alpha=0.7, color='orange', label='High-diversity')
    ax2.hist(low_lengths, bins=bins, alpha=0.7, color='blue', label='Low-diversity')
    
    ax2.set_title("(B) Distribution of Sequence Lengths")
    ax2.set_xlabel("Sequence Length")
    ax2.set_ylabel("Frequency")
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Plot C: Levenshtein distances
    ax3 = plt.subplot(gs[2])
    
    if wild_type:
        high_distances = [levenshtein_distance(wild_type, seq) for seq in high_seqs]
        low_distances = [levenshtein_distance(wild_type, seq) for seq in low_seqs]
        
        # Combine data for histogram
        all_distances = high_distances + low_distances
        bins = np.linspace(min(all_distances), max(all_distances), 30)
        
        ax3.hist(high_distances, bins=bins, alpha=0.7, color='orange', label='High-diversity')
        ax3.hist(low_distances, bins=bins, alpha=0.7, color='blue', label='Low-diversity')
        
        ax3.set_title("(C) Levenshtein Distances from Wild-type")
        ax3.set_xlabel("Levenshtein Distance")
        ax3.set_ylabel("Frequency")
        ax3.legend()
        ax3.grid(alpha=0.3)
    else:
        ax3.text(0.5, 0.5, "Wild-type sequence not available", 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title("(C) Levenshtein Distances from Wild-type")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def run_mds_analysis(
    high_fasta_path: Union[str, Path], 
    low_fasta_path: Union[str, Path], 
    output_dir: Union[str, Path],
    sample_frac: float = DEFAULT_SAMPLE_FRAC,
    random_state: int = DEFAULT_RANDOM_STATE,
    return_data: bool = False
) -> Optional[dict]:
    """
    Run MDS analysis and generate all plots. This function can be imported into pipeline.py.
    
    Args:
        high_fasta_path: Path to high diversity FASTA file
        low_fasta_path: Path to low diversity FASTA file
        output_dir: Directory to save plots
        sample_frac: Fraction of sequences to sample (default: 0.02)
        random_state: Random seed for reproducibility (default: 42)
        return_data: Whether to return the computed data (default: False)
        
    Returns:
        Dictionary with computed data if return_data is True, None otherwise
    """
    # Convert string paths to Path objects if needed
    high_fasta = Path(high_fasta_path)
    low_fasta = Path(low_fasta_path)
    output_dir = Path(output_dir)
    
    print(f"Running MDS analysis with sampling fraction: {sample_frac}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # --- load & sample sequences ---
    high_seqs, high_wt = load_and_sample(high_fasta, sample_frac, random_state)
    low_seqs, low_wt = load_and_sample(low_fasta, sample_frac, random_state)
    
    # Use the high diversity wild-type as the reference
    wild_type = high_wt if high_wt else low_wt
    
    print(f"Sampled {len(high_seqs)} high-diversity sequences")
    print(f"Sampled {len(low_seqs)} low-diversity sequences")
    print(f"Wild-type sequence available: {wild_type is not None}")
    
    # --- vectorize with consistent dimensions ---
    # Process all sequences together to ensure consistent dimensionality
    all_seqs = high_seqs + low_seqs
    all_vecs, max_length = one_hot_padded(all_seqs)
    
    # Split back into high and low
    n_high = len(high_seqs)
    high_vecs = all_vecs[:n_high]
    low_vecs = all_vecs[n_high:]
    
    # --- MDS ---
    print("Computing MDS projection...")
    mds = MDS(n_components=2, random_state=random_state, n_init=1, verbose=1, normalized_stress='auto')
    embedding = mds.fit_transform(all_vecs)
    print(f"MDS stress: {mds.stress_}")
    
    # --- split embeddings ---
    high_emb = embedding[:n_high]
    low_emb = embedding[n_high:]
    
    # --- Get wild-type embedding ---
    wt_emb = None
    if wild_type and wild_type in high_seqs:
        wt_idx = high_seqs.index(wild_type)
        wt_emb = high_emb[wt_idx]
    elif wild_type and wild_type in low_seqs:
        wt_idx = low_seqs.index(wild_type)
        wt_emb = low_emb[wt_idx]
    
    # --- Generate individual plots ---
    plot_mds_with_kde(high_emb, low_emb, wt_emb, output_dir / "mds_visualization.png")
    plot_sequence_lengths(high_seqs, low_seqs, output_dir / "sequence_lengths.png")
    plot_levenshtein_distances(high_seqs, low_seqs, wild_type, output_dir / "levenshtein_distances.png")
    
    # --- Generate combined plot ---
    create_combined_plot(high_seqs, low_seqs, high_emb, low_emb, wt_emb, wild_type, 
                        output_dir / "combined_analysis.png")
    
    print(f"All plots saved to {output_dir}")
    
    # Return data if requested
    if return_data:
        return {
            "high_seqs": high_seqs,
            "low_seqs": low_seqs,
            "high_emb": high_emb,
            "low_emb": low_emb,
            "wild_type": wild_type,
            "wt_emb": wt_emb,
            "max_length": max_length
        }
    return None

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="MDS visualization of CAR sequences")
    parser.add_argument(
        "--high-fasta", 
        type=str,
        default=DEFAULT_HIGH_FASTA,
        help=f"Path to high diversity FASTA file (default: {DEFAULT_HIGH_FASTA})"
    )
    parser.add_argument(
        "--low-fasta", 
        type=str,
        default=DEFAULT_LOW_FASTA,
        help=f"Path to low diversity FASTA file (default: {DEFAULT_LOW_FASTA})"
    )
    parser.add_argument(
        "--output-dir", 
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save plots (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--sample-fraction", 
        type=float,
        default=DEFAULT_SAMPLE_FRAC,
        help=f"Fraction of sequences to sample (default: {DEFAULT_SAMPLE_FRAC})"
    )
    parser.add_argument(
        "--random-state", 
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help=f"Random seed for reproducibility (default: {DEFAULT_RANDOM_STATE})"
    )
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Get project root for resolving relative paths
    root = get_project_root()
    
    # Resolve paths relative to project root if they're not absolute
    high_fasta = Path(args.high_fasta)
    if not high_fasta.is_absolute():
        high_fasta = root / high_fasta
        
    low_fasta = Path(args.low_fasta)
    if not low_fasta.is_absolute():
        low_fasta = root / low_fasta
        
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    
    # Run MDS analysis
    run_mds_analysis(
        high_fasta_path=high_fasta,
        low_fasta_path=low_fasta,
        output_dir=output_dir,
        sample_frac=args.sample_fraction,
        random_state=args.random_state
    )


# Analysis of generated CAR sequences.

# (A) MDS plot of generated sequences, comparing the high-diversity and low-diversity groups. 
# The orange plots represent sequences belonging to the high-diversity group, 
# while the blue plots represent sequences belonging to the low-diversity group. 
# The filled contours represent the kernel density estimate (KDE). 
# The yellow star indicates the wild-type (= original sequence before introducing mutations) sequence. 
# (B) Distribution of the sequence lengths. 
# (C) Levenshtein distances from the wild-type sequence.
# A stress value of 0 indicates “perfect” fit, 0.025 excellent, 0.05 good, 0.1 fair, and 0.2 poor
