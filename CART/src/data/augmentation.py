#!/usr/bin/env python3
"""
Augment CAR-T sequences by recombining homologous intracellular domains
(CD28 ICD and CD3ζ ICD) via HMMER, clustering, and sampling.
"""
import argparse
import logging
import subprocess
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
from Bio import SeqIO, SearchIO
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# --- Configuration constants ---
AA_LIST = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}
MAX_GENERATE = 10000  # cap on generated sequences

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)


def run_phmmer(query: Path, db: Path, out_domtbl: Path, evalue: float = 1e-4) -> Path:
    """
    Run phmmer to search query against db, writing domain table to out_domtbl.
    """
    cmd = [
        'phmmer',
        '--domtblout', str(out_domtbl),
        '-E', str(evalue),
        str(query),
        str(db)
    ]
    subprocess.run(cmd, check=True)
    log.info(f"phmmer domtblout written to {out_domtbl}")
    return out_domtbl


def extract_domain_hits(domtblout: Path, db_fasta: Path, evalue_cutoff: float = 1e-4) -> List[str]:
    """
    Parse HMMER domtblout and extract subsequences with evalue <= cutoff.
    """
    seqdb = SeqIO.index(str(db_fasta), "fasta")
    seqs = []
    for qres in SearchIO.parse(str(domtblout), "phmmer3-domtab"):
        for hit in qres.hits:
            for hsp in hit.hsps:
                if hsp.evalue <= evalue_cutoff:
                    subseq = seqdb[hit.id].seq[hsp.hit_start:hsp.hit_end]
                    seqs.append(str(subseq))
    log.info(f"Extracted {len(seqs)} raw hits from {domtblout.name}")
    return seqs


def remove_duplicates(seqs: List[str]) -> List[str]:
    seen = set()
    unique = []
    for s in seqs:
        if s not in seen:
            unique.append(s)
            seen.add(s)
    return unique


def filter_by_length(seqs: List[str], ref_len: int, tol: float) -> List[str]:
    min_len = int(ref_len * (1 - tol))
    max_len = int(ref_len * (1 + tol))
    filtered = [s for s in seqs if min_len <= len(s) <= max_len]
    log.info(f"Filtered to {len(filtered)} sequences in [{min_len},{max_len}]")
    return filtered


def one_hot_encode(seq: str, length: int) -> np.ndarray:
    arr = np.zeros((length, len(AA_LIST)), dtype=int)
    for i, aa in enumerate(seq[:length]):
        if aa in AA_TO_IDX:
            arr[i, AA_TO_IDX[aa]] = 1
    return arr.flatten()


def determine_optimal_k(
    seqs: List[str], wt_seq: str,
    target_pct: float = 0.25, max_k: int = 20,
    tol: float = 0.05
) -> int:
    if not seqs:
        return 1
    max_len = max(map(len, seqs))
    X = np.vstack([one_hot_encode(s, max_len) for s in seqs])
    wt_vec = one_hot_encode(wt_seq, max_len)

    best_k, best_diff = 1, float('inf')
    best_pct = 0.0
    for k in range(2, min(max_k, len(seqs)) + 1):
        km = KMeans(n_clusters=k, random_state=0, n_init=10).fit(X)
        wt_label = int(km.predict([wt_vec])[0])
        counts = np.bincount(km.labels_)
        pct = counts[wt_label] / len(seqs)
        diff = abs(pct - target_pct)
        if diff < best_diff:
            best_k, best_diff, best_pct = k, diff, pct
            if diff <= tol:
                break
    log.info(f"Optimal k={best_k} ({best_pct:.1%} WT cluster)")
    return best_k


def cluster_and_project(
    seqs: List[str], k: int, wt_seq: str
) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    max_len = max(map(len, seqs))
    X = np.vstack([one_hot_encode(s, max_len) for s in seqs])
    km = KMeans(n_clusters=k, random_state=0, n_init=10).fit(X)
    wt_idx = seqs.index(wt_seq)
    labels = km.labels_
    return X, labels, wt_idx, km.cluster_centers_


def reduce_to_2d(
    X: np.ndarray, method: str = 'kmeans'
) -> Tuple[np.ndarray, str, str]:
    if method == 'pca':
        pca = PCA(n_components=2, random_state=42)
        X2d = pca.fit_transform(X)
        var = pca.explained_variance_ratio_
        return X2d, f"PC1 ({var[0]:.1%})", f"PC2 ({var[1]:.1%})"
    if method == 'tsne':
        tsne = TSNE(n_components=2, random_state=42)
        X2d = tsne.fit_transform(X)
        return X2d, 't-SNE1', 't-SNE2'
    # kmeans fallback: top-2 variance features
    vars_ = np.var(X, axis=0)
    idx = np.argsort(vars_)[-2:]
    return X[:, idx], f"Feat{idx[0]}", f"Feat{idx[1]}"


def plot_2d(
    X2d: np.ndarray, labels: np.ndarray, wt_idx: int,
    xlabel: str, ylabel: str, title: str, out: Path
) -> None:
    n_clusters = len(np.unique(labels))
    colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))
    plt.figure(figsize=(8, 6))
    for cl in range(n_clusters):
        mask = labels == cl
        plt.scatter(X2d[mask, 0], X2d[mask, 1],
                    label=f"Cluster {cl}",
                    alpha=0.7, s=40,
                    c=[colors[cl]])
    plt.scatter(
        X2d[wt_idx, 0], X2d[wt_idx, 1],
        marker='*', s=200,
        edgecolor='k', linewidth=1,
        label='Wild-type', c='yellow'
    )
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(alpha=0.3);
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(out, dpi=300)
    plt.close()
    log.info(f"Saved plot: {out}")


def generate_sequences(
    clusters: List[List[str]],
    sel_idxs: List[int], ratio: float
) -> List[str]:
    choices = [(i, s) for i in sel_idxs for s in clusters[i]]
    max_comb = len(choices)
    n = min(int(max_comb * ratio), MAX_GENERATE)
    combined = []
    for _ in range(n):
        c1 = random.choice(choices)
        c2 = random.choice(choices)
        combined.append(c1[1] + c2[1])
    unique = remove_duplicates(combined)
    log.info(f"Generated {len(unique)} unique sequences")
    return unique


def write_fasta(seqs: List[str], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as fh:
        for i, s in enumerate(seqs):
            fh.write(f">seq_{i}\n{s}\n")
    log.info(f"Wrote {len(seqs)} sequences to {out}")


def parse_args(args_list=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Augment CAR-T sequences")
    parser.add_argument('--wt_cd28', type=Path, default=Path('../fasta/wt_cd28.fasta'))
    parser.add_argument('--wt_cd3z', type=Path, default=Path('../fasta/wt_cd3z.fasta'))
    parser.add_argument('--uniprot_db', type=Path, default=Path('../fasta/uniprot_trembl.fasta'))
    parser.add_argument('--output_dir', type=Path, default=Path('../../output/augmented'))
    parser.add_argument('--plots_dir', type=Path, default=Path('../../output/plots'))
    parser.add_argument('--tol', type=float, default=0.1)
    parser.add_argument('--min_wt_pct', type=float, default=0.5)
    parser.add_argument('--high_ratio', type=float, default=0.7)
    parser.add_argument('--low_ratio', type=float, default=0.3)
    parser.add_argument('--plot_clusters', action='store_true')
    parser.add_argument('--plot_method', choices=['kmeans','pca','tsne'], default='kmeans')
    parser.add_argument('--create_test_data', action='store_true')
    args = parser.parse_args()
    if args_list is None and __name__ == "__main__":
        return parser.parse_args()
    return parser.parse_args(args_list)
    

def run_augmentation(args: argparse.Namespace) -> None:
    root = Path(__file__).parent.parent.resolve()
    wt28 = args.wt_cd28 if args.wt_cd28.is_absolute() else root/args.wt_cd28
    wt3z = args.wt_cd3z if args.wt_cd3z.is_absolute() else root/args.wt_cd3z
    db = args.uniprot_db if args.uniprot_db.is_absolute() else root/args.uniprot_db
    out = args.output_dir if args.output_dir.is_absolute() else root/args.output_dir
    plots = args.plots_dir if args.plots_dir.is_absolute() else root/args.plots_dir

    # Optional test data
    if args.create_test_data:
        from tempfile import NamedTemporaryFile
        if not wt28.exists(): SeqIO.write([SeqIO.SeqRecord(SeqIO.Seq('A'*10), 'wt28', '')], wt28, 'fasta')
        if not wt3z.exists(): SeqIO.write([SeqIO.SeqRecord(SeqIO.Seq('C'*10), 'wt3z', '')], wt3z, 'fasta')
        if not db.exists(): SeqIO.write([], db, 'fasta')

    for p, name in [(wt28,'CD28'),(wt3z,'CD3ζ'),(db,'Uniprot DB')]:
        if not p.exists():
            log.error(f"Missing {name} file: {p}")
            return

    out.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    
    seq_wt28 = str(next(SeqIO.parse(wt28, 'fasta')).seq)
    seq_wt3z = str(next(SeqIO.parse(wt3z, 'fasta')).seq)

    dom28 = run_phmmer(wt28, db, out/'cd28.domtblout')
    dom3z = run_phmmer(wt3z, db, out/'cd3z.domtblout')

    hits28 = extract_domain_hits(dom28, db, args.tol)
    hits3z = extract_domain_hits(dom3z, db, args.tol)
    uniq28 = remove_duplicates(hits28)
    uniq3z = remove_duplicates(hits3z)

    filt28 = filter_by_length(uniq28, len(seq_wt28), args.tol)
    filt3z = filter_by_length(uniq3z, len(seq_wt3z), args.tol)
    if seq_wt28 not in filt28: filt28.append(seq_wt28)
    if seq_wt3z not in filt3z: filt3z.append(seq_wt3z)

    k28 = determine_optimal_k(filt28, seq_wt28, args.min_wt_pct)
    k3z = determine_optimal_k(filt3z, seq_wt3z, args.min_wt_pct)

    X28, lbl28, iwt28, _ = cluster_and_project(filt28, k28, seq_wt28)
    X3z, lbl3z, iwt3z, _ = cluster_and_project(filt3z, k3z, seq_wt3z)

    if args.plot_clusters:
        X2d_28, xl_28, yl_28 = reduce_to_2d(X28, args.plot_method)
        plot_2d(X2d_28, lbl28, iwt28, xl_28, yl_28,
                f"CD28 clusters ({args.plot_method})", plots/'cd28.png')
        X2d_3z, xl_3z, yl_3z = reduce_to_2d(X3z, args.plot_method)
        plot_2d(X2d_3z, lbl3z, iwt3z, xl_3z, yl_3z,
                f"CD3ζ clusters ({args.plot_method})", plots/'cd3z.png')

    # Generate sequences
    # treat clusters as list of lists
    clusters28 = {i:[] for i in range(k28)}
    for s,l in zip(filt28, lbl28): clusters28[l].append(s)
    clusters3z = {i:[] for i in range(k3z)}
    for s,l in zip(filt3z, lbl3z): clusters3z[l].append(s)

    high = generate_sequences(list(clusters28.values()), list(clusters28.keys()), args.high_ratio)
    low  = generate_sequences(list(clusters3z.values()), list(clusters3z.keys()), args.low_ratio)

    write_fasta(high, out/'high_diversity.fasta')
    write_fasta(low, out/'low_diversity.fasta')

    log.info("Augmentation complete.")

def main():
    args = parse_args()
    run_augmentation(args)

if __name__ == "__main__":
    main()

    
