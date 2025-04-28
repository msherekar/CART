# Script to generate augmented CAR sequences by recombining homologous intracellular domains
# (4-1BB ICD and CD3ζ ICD) via HMMER, clustering, and sampling.

import subprocess
import random
import os
from pathlib import Path
from Bio import SeqIO, SearchIO
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from collections import Counter

# Amino acid alphabet (for one-hot encoding)
AA_LIST = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}

# --- plotting constants ---
PLOT_BACKEND = 'kmeans'  # Options: 'kmeans', 'pca', 'tsne'
DEFAULT_PLOTS_DIR = 'plots'  # Relative to project root


def get_project_root() -> Path:
    """
    Get the project root directory.
    Returns a Path object representing the project root.
    """
    # Start with the directory of this script
    script_dir = Path(__file__).parent.absolute()
    
    # Navigate up to find project root (where CART is a directory)
    current_dir = script_dir
    while current_dir.name != "CART" and current_dir.parent != current_dir:
        current_dir = current_dir.parent
    
    # If we found CART directory, return its parent
    if current_dir.name == "CART":
        return current_dir.parent
    
    # If we couldn't find it, use the current working directory as fallback
    return Path.cwd()


def resolve_path(path_str: str) -> Path:
    """
    Resolve a path string relative to the project root.
    Absolute paths are returned unchanged.
    """
    path = Path(path_str)
    if path.is_absolute():
        return path
    return get_project_root() / path


def run_phmmer(query_fasta: Path, db_fasta: Path, evalue: float = 1e-4) -> Path:
    """
    Run phmmer to search homologs of query in the uniprot database.
    Returns domtblout path.
    """
    tblout = query_fasta.with_suffix('.tblout')
    domtblout = query_fasta.with_suffix('.domtblout')
    cmd = [
        'phmmer',
        '--tblout', str(tblout),
        '--domtblout', str(domtblout),
        '-E', str(evalue),
        str(query_fasta),
        str(db_fasta)
    ]
    subprocess.run(cmd, check=True)
    return domtblout



def extract_domain_hits(domtblout: Path, db_fasta: Path, evalue_cutoff: float = 1e-4) -> list:
    """
    Extract domain-aligned subsequences from domtblout.
    """
    seqdb = SeqIO.index(str(db_fasta), "fasta")
    seqs = []
    for qresult in SearchIO.parse(str(domtblout), "phmmer3-domtab"):
        for hit in qresult.hits:
            for hsp in hit.hsps:
                if hsp.evalue <= evalue_cutoff:
                    start, end = hsp.hit_start, hsp.hit_end
                    subseq = seqdb[hit.id].seq[start:end]
                    seqs.append(str(subseq))
    return seqs


def remove_duplicates(seqs):
    """Remove duplicate sequences while preserving order."""
    unique_seqs = []
    seen = set()
    for seq in seqs:
        if seq not in seen:
            unique_seqs.append(seq)
            seen.add(seq)
    return unique_seqs


def filter_by_length(seqs, ref_len, tol):
    """Filter sequences by length based on reference length and tolerance."""
    min_len = int(ref_len * (1 - tol))
    max_len = int(ref_len * (1 + tol))
    return [s for s in seqs if min_len <= len(s) <= max_len]


def one_hot(seq: str, length: int) -> np.ndarray:
    """Convert sequence to one-hot encoding."""
    arr = np.zeros((length, len(AA_LIST)), dtype=int)
    for i, aa in enumerate(seq):
        if i >= length:
            break
        if aa in AA_TO_IDX:
            arr[i, AA_TO_IDX[aa]] = 1
    return arr.flatten()


def determine_optimal_k(seqs, wt_seq, target_percentage=0.25, max_k=20, tolerance=0.05):
    """
    Determine optimal K for clustering to get as close as possible to target_percentage
    of sequences in the wild-type cluster.
    
    Args:
        seqs: List of sequences
        wt_seq: Wild-type sequence
        target_percentage: Target percentage of sequences in WT cluster (default: 0.25)
        max_k: Maximum number of clusters to try
        tolerance: Acceptable deviation from target percentage
        
    Returns:
        Optimal k value
    """
    if not seqs:
        return 1
        
    max_len = max(len(s) for s in seqs)
    wt_encoded = one_hot(wt_seq, max_len)
    encoded = [one_hot(s, max_len) for s in seqs]
    X = np.vstack(encoded)
    
    best_k = 1
    best_wt_percentage = 0
    best_diff = float('inf')
    
    for k in range(2, min(max_k + 1, len(seqs))):
        km = KMeans(n_clusters=k, random_state=0, n_init=10).fit(X)
        wt_cluster = int(km.predict([wt_encoded])[0])
        
        # Count sequences in each cluster
        cluster_counts = Counter(km.labels_)
        wt_cluster_size = cluster_counts[wt_cluster]
        wt_percentage = wt_cluster_size / len(seqs)
        
        # Calculate how close we are to target percentage
        diff = abs(wt_percentage - target_percentage)
        
        # Update best k if:
        # 1. We're closer to target percentage than before
        # 2. The percentage is within tolerance of target
        # 3. The percentage is not too high (upper limit)
        if (diff < best_diff and 
            abs(wt_percentage - target_percentage) <= tolerance and
            wt_percentage <= target_percentage + tolerance):
            best_k = k
            best_wt_percentage = wt_percentage
            best_diff = diff
            
        # If we've found a perfect match, stop searching
        if abs(wt_percentage - target_percentage) < 0.01:
            break
    
    if best_k == 1:
        print(f"Warning: Could not find k value that achieves {target_percentage:.1%} ± {tolerance:.1%} sequences in WT cluster")
        print(f"Using k=1 with {best_wt_percentage:.1%} sequences in WT cluster")
    else:
        print(f"Selected k={best_k} with {best_wt_percentage:.1%} sequences in WT cluster")
    
    return best_k


def cluster_and_assign_seqs(seqs, k: int, wt_seq: str):
    """
    Cluster sequences using K-means and assign them to clusters.
    
    Returns:
        clusters: Dictionary mapping cluster IDs to sequences
        km: K-means model
        wt_cluster: Cluster ID containing the wild-type sequence
        X: Feature matrix
    """
    if not seqs:
        raise ValueError("No sequences provided for clustering.")
    max_len = max(len(s) for s in seqs)
    encoded = [one_hot(s, max_len) for s in seqs if s]
    X = np.vstack(encoded)
    km = KMeans(n_clusters=min(k, len(X)), random_state=0, n_init=10).fit(X)
    clusters = {i: [] for i in range(km.n_clusters)}
    for seq, label in zip(seqs, km.labels_):
        clusters[label].append(seq)
    wt_cluster = int(km.predict([one_hot(wt_seq, max_len)])[0])
    
    # Calculate percentage of sequences in wild-type cluster
    total_seqs = len(seqs)
    wt_cluster_size = len(clusters[wt_cluster])
    wt_percentage = wt_cluster_size / total_seqs
    print(f"Wild-type cluster {wt_cluster} contains {wt_percentage:.1%} of sequences ({wt_cluster_size}/{total_seqs})")
    
    return clusters, km, wt_cluster, X


def plot_clusters(X, labels, wt_idx, title, output_path, max_points=500, plot_method=PLOT_BACKEND):
    """
    Plot clusters using KMeans centroids directly or dimensionality reduction techniques.
    
    Args:
        X: Feature matrix (one-hot encoded sequences)
        labels: Cluster labels
        wt_idx: Index of wild-type sequence
        title: Plot title
        output_path: Path to save the plot
        max_points: Maximum number of points to plot (for large datasets)
        plot_method: Which visualization method to use ('kmeans', 'pca', or 'tsne')
    """
    # Sample points if there are too many
    if X.shape[0] > max_points:
        # Always include wild-type
        indices = list(range(X.shape[0]))
        indices.remove(wt_idx)
        random.seed(42)  # For reproducibility
        sample_indices = random.sample(indices, max_points - 1)
        sample_indices.append(wt_idx)
        
        X_sample = X[sample_indices]
        labels_sample = labels[sample_indices]
        wt_idx_sample = sample_indices.index(wt_idx)
    else:
        X_sample = X
        labels_sample = labels
        wt_idx_sample = wt_idx
    
    # Prepare the 2D representation based on selected method
    if plot_method == 'kmeans':
        # For KMeans-based plotting, we use the first two features with most variance
        # First, we calculate the variance of each feature
        feature_vars = np.var(X_sample, axis=0)
        # Get the two indices with highest variance
        top_features = np.argsort(feature_vars)[-2:]
        # Extract those features
        X_2d = X_sample[:, top_features]
        plot_title = f"{title} (using top 2 variance features)"
        x_label = f"Feature {top_features[0]}"
        y_label = f"Feature {top_features[1]}"
    elif plot_method == 'pca':
        # For PCA, we reduce to 2 dimensions
        pca = PCA(n_components=2, random_state=42)
        X_2d = pca.fit_transform(X_sample)
        explained_var = pca.explained_variance_ratio_
        plot_title = title
        x_label = f"PC1 ({explained_var[0]:.2%} variance)"
        y_label = f"PC2 ({explained_var[1]:.2%} variance)"
    elif plot_method == 'tsne':
        # For t-SNE, we reduce to 2 dimensions
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, X_sample.shape[0]-1))
        X_2d = tsne.fit_transform(X_sample)
        plot_title = f"{title} (t-SNE projection)"
        x_label = "t-SNE 1"
        y_label = "t-SNE 2"
    else:
        raise ValueError(f"Unknown plot method: {plot_method}")
    
    # Plot clusters
    plt.figure(figsize=(10, 8))
    
    # Create colormap with distinct colors for clusters
    unique_labels = np.unique(labels_sample)
    colors = sns.color_palette("husl", len(unique_labels))
    
    # Plot each cluster
    for i, label in enumerate(unique_labels):
        mask = (labels_sample == label)
        plt.scatter(
            X_2d[mask, 0], 
            X_2d[mask, 1], 
            c=[colors[i]], 
            label=f'Cluster {label}',
            alpha=0.7, 
            s=50,
            edgecolor='k',
            linewidth=0.5
        )
    
    # Highlight wild-type sequence with larger star and outline
    plt.scatter(
        X_2d[wt_idx_sample, 0], 
        X_2d[wt_idx_sample, 1], 
        marker='*', 
        c='yellow', 
        s=500,  # Larger size
        label='Wild-type',
        edgecolor='black',
        linewidth=2,  # Thicker border
        zorder=10
    )
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(plot_title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Cluster plot saved to {output_path}")


def generate_sequences(clusters_cd28, clusters_cd3z, sel_clusters_cd28, sel_clusters_cd3z, ratio=1.0):
    """
    Generate sequences by combining clusters.
    The number of sequences is determined by the possible combinations.
    
    Args:
        clusters_cd28: Dictionary of CD28 clusters
        clusters_cd3z: Dictionary of CD3ζ clusters
        sel_clusters_cd28: List of selected CD28 cluster IDs
        sel_clusters_cd3z: List of selected CD3ζ cluster IDs
        ratio: Ratio of possible combinations to generate (default: 1.0)
        
    Returns:
        List of combined sequences
    """
    # Count total possible combinations
    total_seqs_cd28 = sum(len(clusters_cd28[c]) for c in sel_clusters_cd28)
    total_seqs_cd3z = sum(len(clusters_cd3z[c]) for c in sel_clusters_cd3z)
    
    # Calculate total possible combinations (upper bound)
    max_combinations = total_seqs_cd28 * total_seqs_cd3z
    
    # Calculate number of sequences to generate based on ratio
    num_to_generate = min(int(max_combinations * ratio), 10000)  # Cap at 10,000 to avoid excessive memory use
    
    print(f"Generating {num_to_generate} sequences from {total_seqs_cd28} CD28 and {total_seqs_cd3z} CD3ζ sequences")
    
    # Generate sequences
    combined = []
    for _ in range(num_to_generate):
        c28 = random.choice(sel_clusters_cd28)
        c3 = random.choice(sel_clusters_cd3z)
        seq28 = random.choice(clusters_cd28[c28])
        seq3 = random.choice(clusters_cd3z[c3])
        combined.append(seq28 + seq3)
    
    # Remove duplicates
    unique_combined = remove_duplicates(combined)
    print(f"Generated {len(unique_combined)} unique sequences after removing duplicates")
    
    return unique_combined


def write_fasta(seqs, out_fasta: Path) -> None:
    """Write sequences to FASTA file."""
    with open(out_fasta, 'w') as fh:
        for i, seq in enumerate(seqs):
            fh.write(f'>aug_{i}\n{seq}\n')


def make_test_fasta(output_path: Path, sequences: list = None):
    """
    Create a test FASTA file with some example sequences.
    Used if the input files don't exist.
    """
    if not sequences:
        # Default test sequences (simple artificial sequences)
        sequences = [
            "YMNMTPRRPGPTRKHYQPYAPPRDFAAYRS",  # CD28-like sequence
            "KSRKGQRDLYSGLNQRRI"               # CD3ζ-like sequence
        ]
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for i, seq in enumerate(sequences):
            f.write(f">seq_{i}\n{seq}\n")
    print(f"Created test FASTA file at {output_path}")

def parse_args():
    parser = argparse.ArgumentParser(description=(
        "Generate augmented CAR sequences by recombining homologous domains. "
        "Uses HMMER to search for homologs and K-means for clustering."
    ))
    parser.add_argument('--wt_cd28', type=str, default='CART/fasta/cd28.fasta',
                       help='Path to wild-type CD28 FASTA file (position: 115-220)')
    parser.add_argument('--wt_cd3z', type=str, default='CART/fasta/cd3z.fasta',
                       help='Path to wild-type CD3ζ FASTA file (position: 52-164)')
    parser.add_argument('--uniprot_db', type=str, default='CART/fasta/uniprot_trembl.fasta',
                       help='Path to Uniprot database FASTA file')
    parser.add_argument('--output_dir', type=str, default='CART/homologs', 
                       help='Directory to save output files (default: CART/homologs)')
    parser.add_argument('--plots_dir', type=str, default=DEFAULT_PLOTS_DIR,
                       help=f'Directory to save plot files (default: {DEFAULT_PLOTS_DIR})')
    parser.add_argument('--evalue', type=float, default=1e-4, 
                       help='E-value threshold for HMMER (default: 1e-4)')
    parser.add_argument('--tol', type=float, default=0.25, 
                       help='Length tolerance factor (default: 0.25, i.e., 0.75-1.25×)')
    parser.add_argument('--min_wt_percentage', type=float, default=0.25, 
                       help='Minimum percentage of sequences in wild-type cluster (default: 0.25)')
    parser.add_argument('--high_ratio', type=float, default=1.0, 
                       help='Ratio of possible combinations for high-diversity group (default: 1.0)')
    parser.add_argument('--low_ratio', type=float, default=1.0, 
                       help='Ratio of possible combinations for low-diversity group (default: 1.0)')
    parser.add_argument('--plot_clusters', action='store_true', 
                       help='Generate cluster plots')
    parser.add_argument('--plot_method', type=str, choices=['kmeans', 'pca', 'tsne'], default='kmeans',
                       help='Method for visualizing clusters (default: kmeans)')
    parser.add_argument('--create_test_data', action='store_true',
                       help='Create test FASTA files if input files do not exist')
    return parser.parse_args()

def run_augmentation():
    # Get project root to handle relative paths
    project_root = get_project_root()
    
    args = parse_args()

    # Resolve all paths relative to project root
    wt_cd28_path = resolve_path(args.wt_cd28)
    wt_cd3z_path = resolve_path(args.wt_cd3z)
    uniprot_db_path = resolve_path(args.uniprot_db)
    output_dir = resolve_path(args.output_dir)
    plots_dir = resolve_path(args.plots_dir)
    
    # Create test files if requested and files don't exist
    if args.create_test_data:
        if not wt_cd28_path.exists():
            make_test_fasta(wt_cd28_path)
        if not wt_cd3z_path.exists():
            make_test_fasta(wt_cd3z_path)
        if not uniprot_db_path.exists():
            # Create a simple uniprot db with a few sequences
            make_test_fasta(uniprot_db_path, [
                "YMNMTPRRPGPTRKHYQPYAPPRDFAAYRSLPGPTRKHYQPYAPPRDFAA",
                "KSRKGQRDLYSGLNQRRIKSRKGQRDLYSGLNQRRI",
                "YMNMTPRRPGPTRKHYQPYAPPRDFAAYRSLPGPTRKH",
                "KSRKGQRDLYSGLNQRRIKSRKGQ",
                "YMNMTPRRPGPTREEEHYQPYAPPRDFAAYRSLPGPTRKHYQPYAPPRD",
                "KSRKGQRDLYSGLEEEEENQRRIKSRKGQRDLYSGLNQRRI"
            ])
    
    # Check if required input files exist
    for path, name in [(wt_cd28_path, "CD28"), (wt_cd3z_path, "CD3ζ"), (uniprot_db_path, "Uniprot DB")]:
        if not path.exists():
            print(f"⚠️ {name} file not found at {path}")
            print(f"Run with --create_test_data to create test files or provide correct paths.")
            exit(1)
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Print working paths
    print(f"Project root: {project_root}")
    print(f"Working with:")
    print(f"  CD28 file: {wt_cd28_path}")
    print(f"  CD3ζ file: {wt_cd3z_path}")
    print(f"  Uniprot DB: {uniprot_db_path}")
    print(f"  Output directory: {output_dir}")
    print(f"  Plots directory: {plots_dir}")

    # Extract wild-type sequences
    wt28 = str(next(SeqIO.parse(wt_cd28_path, 'fasta')).seq)
    wt3z = str(next(SeqIO.parse(wt_cd3z_path, 'fasta')).seq)
    print(f"Wild-type CD28 length: {len(wt28)}")
    print(f"Wild-type CD3ζ length: {len(wt3z)}")

    # Run HMMER
    domtbl28 = run_phmmer(wt_cd28_path, uniprot_db_path, evalue=args.evalue)
    domtbl3z = run_phmmer(wt_cd3z_path, uniprot_db_path, evalue=args.evalue)

    # Extract hits
    seqs28_raw = extract_domain_hits(domtbl28, uniprot_db_path, evalue_cutoff=args.evalue)
    seqs3z_raw = extract_domain_hits(domtbl3z, uniprot_db_path, evalue_cutoff=args.evalue)
    print(f"Extracted {len(seqs28_raw)} CD28 sequences and {len(seqs3z_raw)} CD3ζ sequences")

    # Remove duplicates
    seqs28_unique = remove_duplicates(seqs28_raw)
    seqs3z_unique = remove_duplicates(seqs3z_raw)
    print(f"After removing duplicates: {len(seqs28_unique)} CD28 sequences and {len(seqs3z_unique)} CD3ζ sequences")

    # Filter by length
    seqs28 = filter_by_length(seqs28_unique, len(wt28), args.tol)
    seqs3z = filter_by_length(seqs3z_unique, len(wt3z), args.tol)
    print(f"After length filtering: {len(seqs28)} CD28 sequences and {len(seqs3z)} CD3ζ sequences")

    # Ensure wild-type sequences are in the datasets
    if wt28 not in seqs28:
        print("⚠️ Wild-type CD28 sequence not found in filtered results, adding it.")
        seqs28.append(wt28)
    if wt3z not in seqs3z:
        print("⚠️ Wild-type CD3ζ sequence not found in filtered results, adding it.")
        seqs3z.append(wt3z)

    # Determine optimal K for clustering
    k_cd28 = determine_optimal_k(seqs28, wt28, target_percentage=args.min_wt_percentage)
    k_cd3z = determine_optimal_k(seqs3z, wt3z, target_percentage=args.min_wt_percentage)

    # Cluster sequences
    clusters28, km28, wt_cluster28, X28 = cluster_and_assign_seqs(seqs28, k_cd28, wt28)
    clusters3z, km3z, wt_cluster3z, X3z = cluster_and_assign_seqs(seqs3z, k_cd3z, wt3z)

    # Find wild-type index for plotting
    wt_idx28 = seqs28.index(wt28)
    wt_idx3z = seqs3z.index(wt3z)
    
    # Generate plots if requested
    if args.plot_clusters:
        # Generate multiple visualizations if needed
        for plot_method in [args.plot_method]:
            method_suffix = "" if plot_method == 'kmeans' else f"_{plot_method}"
            
            plot_clusters(
                X28, 
                km28.labels_, 
                wt_idx28, 
                f'CD28 Sequence Clusters (k={k_cd28})', 
                plots_dir / f'cd28_clusters{method_suffix}.png',
                plot_method=plot_method
            )
            
            plot_clusters(
                X3z, 
                km3z.labels_, 
                wt_idx3z, 
                f'CD3ζ Sequence Clusters (k={k_cd3z})', 
                plots_dir / f'cd3z_clusters{method_suffix}.png',
                plot_method=plot_method
            )

    # Generate high-diversity sequences (using all clusters)
    high = generate_sequences(
        clusters28, clusters3z,
        list(clusters28.keys()), list(clusters3z.keys()),
        ratio=args.high_ratio
    )
    
    # Generate low-diversity sequences (using only wild-type clusters)
    low = generate_sequences(
        clusters28, clusters3z,
        [wt_cluster28], [wt_cluster3z],
        ratio=args.low_ratio
    )

    # Write sequences to FASTA files
    high_fasta = output_dir / 'high_diversity.fasta'
    low_fasta = output_dir / 'low_diversity.fasta'
    write_fasta(high, high_fasta)
    write_fasta(low, low_fasta)

    print(f'✅ Sequence augmentation completed:')
    print(f'   - {len(high)} high-diversity sequences saved to {high_fasta}')
    print(f'   - {len(low)} low-diversity sequences saved to {low_fasta}')


if __name__ == '__main__':
    run_augmentation()