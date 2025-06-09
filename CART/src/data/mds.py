#!/usr/bin/env python3
"""
MDS visualization of CAR-T sequences

Generates 2D MDS plots, sequence-length distributions, and Levenshtein-distance histograms
for high- and low-diversity CAR sequences.
"""
import argparse
import logging
import random
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

import numpy as np
from Bio import SeqIO
from sklearn.manifold import MDS
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from Levenshtein import distance as levenshtein_distance

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)

# --- Constants ---
AA_LIST = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}
DEFAULT_SAMPLE_FRAC = 1.0
DEFAULT_RANDOM_STATE = 42


def one_hot_padded(seqs: List[str]) -> np.ndarray:
    """
    One-hot encode a list of sequences, padding to common max length.
    Returns a 2D array of shape (n_seqs, max_len*20).
    """
    max_len = max(len(s) for s in seqs)
    log.info(f"Padding sequences to length {max_len}")
    arr = np.zeros((len(seqs), max_len * len(AA_LIST)), dtype=int)
    for i, seq in enumerate(seqs):
        for j, aa in enumerate(seq[:max_len]):
            idx = AA_TO_IDX.get(aa)
            if idx is not None:
                arr[i, j * len(AA_LIST) + idx] = 1
    return arr


def load_and_sample(
    fasta: Path, frac: float, seed: int
) -> (List[str], Optional[str]):
    """
    Load sequences from FASTA, shuffle, and sample fraction.
    Returns sampled list and the first (wild-type) sequence.
    """
    if not fasta.exists():
        log.error(f"FASTA not found: {fasta}")
        raise FileNotFoundError(fasta)
    seqs = [str(r.seq) for r in SeqIO.parse(str(fasta), "fasta")]
    if not seqs:
        log.warning(f"No sequences in {fasta}")
        return [], None
    wt = seqs[0]
    rnd = random.Random(seed)
    rnd.shuffle(seqs)
    n = max(1, int(len(seqs) * frac))
    sampled = seqs[:n]
    if wt not in sampled:
        sampled[0] = wt
    log.info(f"Sampled {len(sampled)} of {len(seqs)} from {fasta.name}")
    return sampled, wt


def plot_mds_with_kde(
    high_emb: np.ndarray,
    low_emb: np.ndarray,
    wt_emb: Optional[np.ndarray],
    out: Path
) -> None:
    """Plot 2D MDS embedding with KDE contours and wild-type marker."""
    fig, ax = plt.subplots(figsize=(10, 8))
    def kde_contour(data: np.ndarray, cmap_name: str):
        kde = gaussian_kde(data.T)
        x_min, x_max = data[:, 0].min(), data[:, 0].max()
        y_min, y_max = data[:, 1].min(), data[:, 1].max()
        xi, yi = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
        zi = kde(np.vstack([xi.ravel(), yi.ravel()]))
        ax.contourf(xi, yi, zi.reshape(xi.shape), levels=5, cmap=cmap_name, alpha=0.3)
    kde_contour(high_emb, plt.cm.Oranges)
    kde_contour(low_emb, plt.cm.Blues)
    ax.scatter(high_emb[:, 0], high_emb[:, 1], label='High-diversity', alpha=0.6,
               edgecolor='k', linewidth=0.5, s=50, c='orange')
    ax.scatter(low_emb[:, 0], low_emb[:, 1], label='Low-diversity', alpha=0.6,
               edgecolor='k', linewidth=0.5, s=50, c='blue')
    if wt_emb is not None:
        ax.scatter(wt_emb[0], wt_emb[1], label='Wild-type', marker='*',
                   s=300, edgecolor='k', linewidth=1, c='yellow', zorder=10)
    ax.set_title('MDS of CAR Sequences')
    ax.set_xlabel('MDS 1')
    ax.set_ylabel('MDS 2')
    ax.legend()
    ax.grid(alpha=0.3)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    log.info(f"Saved MDS plot to {out}")


def plot_histogram(
    data1: List[int],
    data2: List[int],
    labels: List[str],
    colors: List[str],
    title: str,
    xlabel: str,
    out: Path
) -> None:
    """Plot overlaid histogram of two datasets."""
    fig, ax = plt.subplots(figsize=(10, 6))
    all_vals = data1 + data2
    bins = np.linspace(min(all_vals), max(all_vals), 30)
    ax.hist(data1, bins=bins, alpha=0.7, label=labels[0], color=colors[0])
    ax.hist(data2, bins=bins, alpha=0.7, label=labels[1], color=colors[1])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(alpha=0.3)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    log.info(f"Saved histogram to {out}")


def create_combined_plot(
    high_seqs: List[str],
    low_seqs: List[str],
    high_emb: np.ndarray,
    low_emb: np.ndarray,
    wt_emb: Optional[np.ndarray],
    out: Path
) -> None:
    # Combined plotting handled via separate functions
    pass  # Can be implemented if needed


def run_mds_analysis(
    high_fasta: Path,
    low_fasta: Path,
    output_dir: Path,
    sample_frac: float = DEFAULT_SAMPLE_FRAC,
    random_state: int = DEFAULT_RANDOM_STATE
) -> Dict[str, Any]:
    """Run full MDS analysis pipeline and generate plots."""
    high_seqs, wt_high = load_and_sample(high_fasta, sample_frac, random_state)
    low_seqs, wt_low = load_and_sample(low_fasta, sample_frac, random_state)
    wt = wt_high or wt_low
    combined = high_seqs + low_seqs
    X = one_hot_padded(combined)
    log.info("Computing MDS embedding...")
    mds = MDS(n_components=2, random_state=random_state, n_init=1)
    emb = mds.fit_transform(X)
    high_emb = emb[: len(high_seqs)]
    low_emb = emb[len(high_seqs):]
    wt_emb = None
    if wt:
        if wt in high_seqs:
            wt_emb = high_emb[high_seqs.index(wt)]
        else:
            wt_emb = low_emb[low_seqs.index(wt)]
    plot_mds_with_kde(high_emb, low_emb, wt_emb, output_dir / "mds.png")
    plot_histogram([len(s) for s in high_seqs], [len(s) for s in low_seqs],
                   ['High','Low'], ['orange','blue'],
                   'Sequence Lengths', 'Length', output_dir / 'lengths.png')
    plot_histogram([levenshtein_distance(wt, s) for s in high_seqs],
                   [levenshtein_distance(wt, s) for s in low_seqs],
                   ['High','Low'], ['orange','blue'],
                   'Levenshtein Distances', 'Distance', output_dir / 'levenshtein.png')
    return {
        "high_seqs": high_seqs,
        "low_seqs": low_seqs,
        "high_emb": high_emb,
        "low_emb": low_emb,
        "wt": wt,
        "wt_emb": wt_emb
    }

def parse_args(args_list=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MDS analysis of CAR sequences")
    parser.add_argument("--high_fasta", type=Path, default=Path('../../output/augmented/high_diversity.fasta'),
                        help="High-diversity FASTA path")
    parser.add_argument("--low_fasta", type=Path, default=Path('../../output/augmented/low_diversity.fasta'),
                        help="Low-diversity FASTA path")
    parser.add_argument("--output_dir", type=Path, default=Path("../../output/plots"),
                        help="Directory to save plots")
    parser.add_argument("--sample_frac", type=float, default=DEFAULT_SAMPLE_FRAC,
                        help="Fraction of sequences to sample")
    parser.add_argument("--random_state", type=int, default=DEFAULT_RANDOM_STATE,
                        help="Random seed")
    if args_list is None and __name__ == "__main__":
        return parser.parse_args()
    return parser.parse_args(args_list)

def run_mds(args: argparse.Namespace) -> None:
    """Wrap run_mds_analysis for CLI invocation"""
    root = Path(__file__).parent.parent.resolve()
    hf = args.high_fasta if args.high_fasta.is_absolute() else root / args.high_fasta
    lf = args.low_fasta if args.low_fasta.is_absolute() else root / args.low_fasta
    out = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    run_mds_analysis(hf, lf, out, args.sample_frac, args.random_state)

def main():
    args = parse_args()
    run_mds(args)

if __name__ == "__main__":
    main()
