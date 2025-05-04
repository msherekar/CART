#!/usr/bin/env python3
"""
Generate FMC63-based CAR mutants with mutations in CD28 and CD3ζ domains.
"""
import argparse
import logging
import random
import csv
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from typing import Optional

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)

# --- Constants ---
FMC63_SCFV = (
    "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVAYISSGGGSTYYADSVKGRFT"
    "ISRDNSKNTLYLQMNSLRAEDTAVYYCAKYPHGYWYFDVWGQGTLVTVSSGGGGSGGGGSGGGGSEIVL"
    "TQSPGTLSLSPGERATLSCRASQSVSSSYLAWYQQKPGQAPRLLIYDASTRATGIPDRFSGSGSGTDFTL"
    "TISSLQPEDFATYYCQQYNSYPLTFGAGTKLEIK"
)
CD28_SEQ = "EVMYPPPYLDNEKSNGTIIHVKGKHLCPSPLFPGPSKPFWVLVVVGGVLACYSLLVTVAFIIFWVRSKRSLLHSDYMNMTPRRPGPTRKHYQPYAPPRDFAAYRS"
CD3Z_SEQ = "RVKFSRSADAPAYQQGQNQLYNELNLGRREEYDVLDKRRGRDPEMGGKPQRRKNPQEGLYNELQKDKMAEAYSEIGMKGERRRGKGHDGLYQGLSTATKDTYDALHMQALPPR"
AA_LIST = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}
DEFAULT_N_MUTANTS = 382
DEFAULT_MAX_MUTATIONS = 10
DEFAULT_OUTPUT_DIR = Path("../../output/mutants")
DEFAULT_PLOTS_DIR = Path("../../output/plots")


def parse_args(args_list=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate FMC63-based CAR mutants"
    )
    parser.add_argument(
        "--n_mutants",
        type=int,
        default=DEFAULT_N_MUTANTS,
        help=f"Number of mutants to generate (default: {DEFAULT_N_MUTANTS})"
    )
    parser.add_argument(
        "--max_mutations",
        type=int,
        default=DEFAULT_MAX_MUTATIONS,
        help=f"Max number of mutations per CAR (default: {DEFAULT_MAX_MUTATIONS})"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save mutant CSV/TSV/FASTA files"
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="CAR_mutants",
        help="Base name for output files"
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--baseline_cytox",
        type=float,
        default=50.0,
        help="Baseline cytotoxicity (default: 50.0)"
    )
    parser.add_argument(
        "--cytox_std_dev",
        type=float,
        default=20.0,
        help="Std dev for cytotoxicity Gaussian (default: 20.0)"
    )
    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="Skip generating plots"
    )
    parser.add_argument(
        "--plots_dir",
        type=Path,
        default=DEFAULT_PLOTS_DIR,
        help="Directory to save plots"
    )
    return parser.parse_args(args_list) if args_list else parser.parse_args()


def mutate_sequence(seq: str, num_mutations: int) -> str:
    if num_mutations <= 0:
        return seq
    seq_list = list(seq)
    positions = random.sample(range(len(seq)), num_mutations)
    for pos in positions:
        choices = [aa for aa in AA_LIST if aa != seq[pos]]
        seq_list[pos] = random.choice(choices)
    return ''.join(seq_list)


def generate_car_mutants(
    n: int,
    cd28: str,
    cd3z: str,
    scfv: str,
    max_mut: int,
    seed: Optional[int] = None
) -> List[Tuple[str, str]]:
    if seed is not None:
        random.seed(seed)
    mutants = []
    for i in range(1, n+1):
        total = random.randint(1, max_mut)
        m28 = random.randint(0, total)
        m3z = total - m28
        seq28 = mutate_sequence(cd28, m28)
        seq3z = mutate_sequence(cd3z, m3z)
        full = scfv + seq28 + seq3z
        mutants.append((f"CAR_mutant_{i}", full))
    log.info(f"Generated {len(mutants)} CAR mutants")
    return mutants


def save_mutants(
    mutants: List[Tuple[str, str]],
    base_path: Path
) -> Tuple[Path, Path, Path]:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    csv_p = base_path.with_suffix('.csv')
    tsv_p = base_path.with_suffix('.tsv')
    fasta_p = base_path.with_suffix('.fasta')
    with open(csv_p, 'w', newline='') as cf:
        w = csv.writer(cf)
        w.writerow(['mutant_id','sequence'])
        w.writerows(mutants)
    with open(tsv_p, 'w', newline='') as tf:
        w = csv.writer(tf, delimiter='\t')
        w.writerow(['mutant_id','sequence'])
        w.writerows(mutants)
    with open(fasta_p, 'w') as ff:
        for mid, seq in mutants:
            ff.write(f">{mid}\n{seq}\n")
    log.info(f"Saved mutants to {csv_p}, {tsv_p}, {fasta_p}")
    return csv_p, tsv_p, fasta_p


def dummy_cytox_data(
    mutants: List[Tuple[str, str]],
    out_path: Path,
    baseline: float,
    std_dev: float,
    seed: Optional[int] = None
) -> List[Tuple[str, float]]:
    if seed is not None:
        random.seed(seed)
    data = []
    for mid, _ in mutants:
        val = max(0.0, random.gauss(baseline, std_dev))
        data.append((mid, val))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='') as cf:
        w = csv.writer(cf)
        w.writerow(['mutant_id','cytotoxicity'])
        w.writerows(data)
    log.info(f"Saved cytotoxicity to {out_path}")
    return data


def plot_mutants(
    mutants: List[Tuple[str, str]],
    out_path: Path
) -> None:
    counts = [sum(1 for a,b in zip(seq, FMC63_SCFV+CD28_SEQ+CD3Z_SEQ) if a!=b) for _,seq in mutants]
    plt.figure(figsize=(8,6))
    plt.hist(counts, bins=range(min(counts), max(counts)+2), edgecolor='k', alpha=0.7)
    plt.title('Mutation Counts')
    plt.xlabel('Number of Mutations')
    plt.ylabel('Frequency')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(out_path, dpi=300); plt.close()
    log.info(f"Saved mutation plot to {out_path}")


def plot_cytotoxicity(
    data: List[Tuple[str,float]],
    out_path: Path
) -> None:
    vals = [v for _,v in data]
    mu, sd = norm.fit(vals)
    plt.figure(figsize=(8,6))
    n,b,p = plt.hist(vals, bins=30, edgecolor='k', alpha=0.7)
    x = np.linspace(min(vals), max(vals), 200)
    plt.plot(x, norm.pdf(x, mu, sd)*len(vals)*(b[1]-b[0]), 'r-', lw=2)
    plt.title('Cytotoxicity Distribution')
    plt.xlabel('Cytotoxicity'); plt.ylabel('Count')
    plt.text(0.02,0.95, f"Mean={mu:.2f}\nStd={sd:.2f}", transform=plt.gca().transAxes,
             va='top', bbox=dict(facecolor='white', alpha=0.8))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(out_path, dpi=300); plt.close()
    log.info(f"Saved cytotoxicity plot to {out_path}")


def run_mutants(args: argparse.Namespace) -> None:
    # Resolve paths
    root = Path(__file__).parent.parent.resolve()
    out_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    plot_dir = args.plots_dir if args.plots_dir.is_absolute() else root / args.plots_dir
    out_dir.mkdir(parents=True, exist_ok=True); plot_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir/args.output_name

    log.info(f"Generating {args.n_mutants} CAR mutants (max {args.max_mutations} mutations)")
    mutants = generate_car_mutants(
        args.n_mutants, CD28_SEQ, CD3Z_SEQ, FMC63_SCFV,
        args.max_mutations, args.random_seed
    )
    csv_p, tsv_p, fasta_p = save_mutants(mutants, base)

    cytox_path = out_dir/f"{args.output_name}_cytox.csv"
    cytox = dummy_cytox_data(
        mutants, cytox_path,
        args.baseline_cytox, args.cytox_std_dev,
        args.random_seed
    )

    if not args.no_plots:
        plot_mutants(mutants, plot_dir/"mutation_counts.png")
        plot_cytotoxicity(cytox, plot_dir/"cytotoxicity.png")


def main():
    args = parse_args()
    run_mutants(args)


if __name__ == "__main__":
    main()
