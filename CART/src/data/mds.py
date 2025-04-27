import random
import numpy as np
from pathlib import Path
from Bio import SeqIO
from sklearn.manifold import MDS
import matplotlib.pyplot as plt

# --- parameters ---
HIGH_FASTA = Path("augmented_seqs/high_diversity.fasta")
LOW_FASTA  = Path("augmented_seqs/low_diversity.fasta")
SAMPLE_FRAC = 0.02           # 2%
RANDOM_STATE = 42

# --- one-hot setup ---
AA_LIST = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}

def one_hot(seq: str) -> np.ndarray:
    """Flattened one-hot encoding of amino-acid sequence."""
    L = len(seq)
    arr = np.zeros((L, len(AA_LIST)), dtype=int)
    for i, aa in enumerate(seq):
        idx = AA_TO_IDX.get(aa)
        if idx is not None:
            arr[i, idx] = 1
    return arr.flatten()

# --- load & sample sequences ---
def load_and_sample(fasta_path, frac, seed):
    seqs = [str(rec.seq) for rec in SeqIO.parse(fasta_path, "fasta")]
    random.Random(seed).shuffle(seqs)
    n = max(1, int(len(seqs) * frac))
    return seqs[:n]

high_seqs = load_and_sample(HIGH_FASTA, SAMPLE_FRAC, RANDOM_STATE)
low_seqs  = load_and_sample(LOW_FASTA,  SAMPLE_FRAC, RANDOM_STATE)

# --- vectorize ---
high_vecs = np.vstack([one_hot(s) for s in high_seqs])
low_vecs  = np.vstack([one_hot(s) for s in low_seqs])

# --- MDS ---
mds = MDS(n_components=2, random_state=RANDOM_STATE, dissimilarity="euclidean")
all_vecs = np.vstack([high_vecs, low_vecs])
embedding = mds.fit_transform(all_vecs)

# --- split embeddings & plot ---
n_high = high_vecs.shape[0]
high_emb = embedding[:n_high]
low_emb  = embedding[n_high:]

plt.figure(figsize=(8,6))
plt.scatter(high_emb[:,0], high_emb[:,1], c='red',   label='High-diversity', alpha=0.6)
plt.scatter(low_emb[:,0],  low_emb[:,1],  c='blue',  label='Low-diversity',  alpha=0.6)
plt.title("2D MDS of Augmented CAR Sequences")
plt.xlabel("MDS Dimension 1")
plt.ylabel("MDS Dimension 2")
plt.legend()
plt.tight_layout()
plt.show()
