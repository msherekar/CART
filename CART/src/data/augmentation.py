# Script to generate augmented CAR sequences by recombining homologous intracellular domains
# (4-1BB ICD and CD3ζ ICD) via HMMER, clustering, and sampling.

import subprocess
import random
from pathlib import Path
from Bio import SeqIO, SearchIO
import numpy as np
from sklearn.cluster import KMeans
import argparse

# Amino acid alphabet (for one-hot encoding)
AA_LIST = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}


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


def filter_by_length(seqs, ref_len, tol):
    min_len = int(ref_len * (1 - tol))
    max_len = int(ref_len * (1 + tol))
    return [s for s in seqs if min_len <= len(s) <= max_len]


def one_hot(seq: str, length: int) -> np.ndarray:
    arr = np.zeros((length, len(AA_LIST)), dtype=int)
    for i, aa in enumerate(seq):
        if i >= length:
            break
        if aa in AA_TO_IDX:
            arr[i, AA_TO_IDX[aa]] = 1
    return arr.flatten()


def cluster_and_assign_seqs(seqs, k: int, wt_seq: str):
    if not seqs:
        raise ValueError("No sequences provided for clustering.")
    max_len = max(len(s) for s in seqs)
    encoded = [one_hot(s, max_len) for s in seqs if s]
    X = np.vstack(encoded)
    km = KMeans(n_clusters=min(k, len(X)), random_state=0).fit(X)
    clusters = {i: [] for i in range(km.n_clusters)}
    for seq, label in zip(seqs, km.labels_):
        clusters[label].append(seq)
    wt_cluster = int(km.predict([one_hot(wt_seq, max_len)])[0])
    return clusters, km, wt_cluster


def generate_sequences(clusters_cd28, clusters_cd3z, sel_clusters_cd28, sel_clusters_cd3z, num: int):
    combined = []
    for _ in range(num):
        c28 = random.choice(sel_clusters_cd28)
        c3 = random.choice(sel_clusters_cd3z)
        seq28 = random.choice(clusters_cd28[c28])
        seq3 = random.choice(clusters_cd3z[c3])
        combined.append(seq28 + seq3)
    return combined


def write_fasta(seqs, out_fasta: Path) -> None:
    with open(out_fasta, 'w') as fh:
        for i, seq in enumerate(seqs):
            fh.write(f'>aug_{i}\n{seq}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wt_cd28', type=Path, required=True)
    parser.add_argument('--wt_cd3z', type=Path, required=True)
    parser.add_argument('--uniprot_db', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, default=Path('augmented_seqs'))
    parser.add_argument('--evalue', type=float, default=1e-4)
    parser.add_argument('--tol', type=float, default=0.25)
    parser.add_argument('--k_cd28', type=int, default=4)
    parser.add_argument('--k_cd3z', type=int, default=3)
    parser.add_argument('--n_high', type=int, default=5500)
    parser.add_argument('--n_low', type=int, default=5250)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    wt28 = str(next(SeqIO.parse(args.wt_cd28, 'fasta')).seq)
    wt3z = str(next(SeqIO.parse(args.wt_cd3z, 'fasta')).seq)

    domtbl28 = run_phmmer(args.wt_cd28, args.uniprot_db, evalue=args.evalue)
    domtbl3z = run_phmmer(args.wt_cd3z, args.uniprot_db, evalue=args.evalue)

    seqs28 = extract_domain_hits(domtbl28, args.uniprot_db, evalue_cutoff=args.evalue)
    seqs3z = extract_domain_hits(domtbl3z, args.uniprot_db, evalue_cutoff=args.evalue)

    seqs28 = filter_by_length(seqs28, len(wt28), args.tol)
    seqs3z = filter_by_length(seqs3z, len(wt3z), args.tol)

    if not seqs28:
        print("⚠️ No CD28 sequences found, using WT.")
        seqs28 = [wt28]
    if not seqs3z:
        print("⚠️ No CD3ζ sequences found, using WT.")
        seqs3z = [wt3z]

    clusters28, km28, wt_cluster28 = cluster_and_assign_seqs(seqs28, args.k_cd28, wt28)
    clusters3z, km3z, wt_cluster3z = cluster_and_assign_seqs(seqs3z, args.k_cd3z, wt3z)

    high = generate_sequences(
        clusters28, clusters3z,
        list(clusters28.keys()), list(clusters3z.keys()),
        num=args.n_high
    )
    low = generate_sequences(
        clusters28, clusters3z,
        [wt_cluster28], [wt_cluster3z],
        num=args.n_low
    )

    write_fasta(high, args.output_dir / 'high_diversity.fasta')
    write_fasta(low, args.output_dir / 'low_diversity.fasta')

    print('✅ Sequence augmentation completed.')
