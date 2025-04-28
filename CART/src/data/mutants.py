#!/usr/bin/env python3
"""
Generate FMC63-based CAR mutants with mutations in hinge, TM, and ICD of CD28 and CD3ζ
"""

import random
import csv
import argparse
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import norm

# Constant FMC63 scFv sequence (VH-linker-VL format)
FMC63_SCFV = ("EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVAYISSGGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKYPHGYWYFDVWGQGTLVTVSSGGGGSGGGGSGGGGSEIVLTQSPGTLSLSPGERATLSCRASQSVSSSYLAWYQQKPGQAPRLLIYDASTRATGIPDRFSGSGSGTDFTLTISSLQPEDFATYYCQQYNSYPLTFGAGTKLEIK")

# Domain sequences: CD28(115–228), CD3ζ(52–164)
CD28_SEQ = "EVMYPPPYLDNEKSNGTIIHVKGKHLCPSPLFPGPSKPFWVLVVVGGVLACYSLLVTVAFIIFWVRSKRSRLLHSDYMNMTPRRPGPTRKHYQPYAPPRDFAAYRS"
CD3Z_SEQ = "RVKFSRSADAPAYQQGQNQLYNELNLGRREEYDVLDKRRGRDPEMGGKPQRRKNPQEGLYNELQKDKMAEAYSEIGMKGERRRGKGHDGLYQGLSTATKDTYDALHMQALPPR"

AA_LIST = "ACDEFGHIKLMNPQRSTVWY"
DEFAULT_N_MUTANTS = 382
DEFAULT_MAX_MUTATIONS = 10
DEFAULT_OUTPUT_DIR = "CART/mutants"
DEFAULT_PLOTS_DIR = "CART/plots"


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


def mutate_sequence(seq, num_mutations):
    """
    Create a mutated version of a sequence with specified number of amino acid substitutions.
    
    Args:
        seq: Original amino acid sequence
        num_mutations: Number of mutations to introduce (can be 0)
        
    Returns:
        Mutated sequence
    """
    if num_mutations == 0:
        return seq
        
    positions = random.sample(range(len(seq)), num_mutations)
    mutated = list(seq)
    for pos in positions:
        current = seq[pos]
        options = [aa for aa in AA_LIST if aa != current]
        mutated[pos] = random.choice(options)
    return ''.join(mutated)


def generate_car_mutants(n, cd28_seq, cd3z_seq, scfv, max_mutations, random_seed=None):
    """
    Generate CAR mutants by introducing random mutations to CD28 and CD3ζ domains.
    The total number of mutations across both domains will not exceed max_mutations.
    
    Args:
        n: Number of mutants to generate
        cd28_seq: CD28 domain sequence
        cd3z_seq: CD3ζ domain sequence
        scfv: scFv sequence (constant)
        max_mutations: Maximum number of mutations per CAR (across both domains)
        random_seed: Seed for random number generator (optional)
        
    Returns:
        List of (mutant_id, mutant_sequence) tuples
    """
    if random_seed is not None:
        random.seed(random_seed)
        
    mutants = []
    for i in range(n):
        # Determine how many mutations to apply to each domain
        total_mutations = random.randint(1, max_mutations)
        # Randomly split mutations between domains
        cd28_mutations = random.randint(0, total_mutations)
        cd3z_mutations = total_mutations - cd28_mutations
        
        # Apply mutations to each domain
        mut_cd28 = mutate_sequence(cd28_seq, cd28_mutations)
        mut_cd3z = mutate_sequence(cd3z_seq, cd3z_mutations)
        
        # Combine into full sequence
        full_seq = scfv + mut_cd28 + mut_cd3z
        mutants.append((f"CAR_mutant_{i+1}", full_seq))
    return mutants

def dummy_cytox_data(mutants, output_path: Path, baseline_cytox=50, std_dev=20, random_seed=None):
    """
    Generate dummy cytotoxicity data for the mutants using a Gaussian distribution.
    
    Args:
        mutants: List of (mutant_id, mutant_sequence) tuples
        output_path: Path to save the cytotoxicity data
        baseline_cytox: Baseline cytotoxicity value (default: 50)
        std_dev: Standard deviation for the Gaussian distribution (default: 20)
        random_seed: Random seed for reproducibility (optional)
        
    Returns:
        List of (mutant_id, cytotoxicity) tuples
    """
    if random_seed is not None:
        random.seed(random_seed)
    
    # Generate cytotoxicity values from Gaussian distribution
    cytox_data = []
    for mutant_id, _ in mutants:
        # Generate value from Gaussian distribution
        cytox = random.gauss(baseline_cytox, std_dev)
        # Ensure cytotoxicity is non-negative
        cytox = max(0, cytox)
        cytox_data.append((mutant_id, cytox))
    
    # Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["mutant_id", "cytotoxicity"])
        writer.writerows(cytox_data)
    
    return cytox_data

def save_mutants(mutants, output_path: Path):
    """
    Save mutants to CSV, TSV, and FASTA files.
    
    Args:
        mutants: List of (mutant_id, mutant_sequence) tuples
        output_path: Base path for output files (without extension)
    """
    # Ensure directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_path.with_suffix('.csv')
    tsv_path = output_path.with_suffix('.tsv')
    fasta_path = output_path.with_suffix('.fasta')

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["mutant_id", "mutant_sequence"])
        writer.writerows(mutants)

    with open(tsv_path, 'w', newline='') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t')
        writer.writerow(["mutant_id", "mutant_sequence"])
        writer.writerows(mutants)
    
    # Write FASTA
    with open(fasta_path, 'w') as fastafile:
        for mutant_id, mutant_seq in mutants:
            fastafile.write(f">{mutant_id}\n{mutant_seq}\n")
    
    return csv_path, tsv_path, fasta_path

def plot_mutants(mutants, output_path: Path):
    """
    Plot the distribution of mutations in the mutants.
    
    Args:
        mutants: List of (mutant_id, mutant_sequence) tuples
        output_path: Path to save the plot
    """
    if not mutants:
        raise ValueError("No mutants provided for plotting")
    
    # Count mutations in each sequence
    mutation_counts = []
    for _, seq in mutants:
        # Count differences from wild-type sequence
        mutations = sum(1 for a, b in zip(seq, FMC63_SCFV + CD28_SEQ + CD3Z_SEQ) if a != b)
        mutation_counts.append(mutations)
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(mutation_counts, bins=range(min(mutation_counts), max(mutation_counts) + 2), 
             edgecolor='black', alpha=0.7)
    
    # Add labels and title
    plt.title("Distribution of Mutations in CAR Mutants", fontsize=14)
    plt.xlabel("Number of Mutations", fontsize=12)
    plt.ylabel("Number of Mutants", fontsize=12)
    
    # Add grid and adjust layout
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_cytotoxicity(cytox_data, output_path: Path):
    """
    Plot the distribution of cytotoxicity in the mutants.
    
    Args:
        cytox_data: List of (mutant_id, cytotoxicity) tuples
        output_path: Path to save the plot
    """
    if not cytox_data:
        raise ValueError("No cytotoxicity data provided for plotting")
    
    # Extract cytotoxicity values
    cytox_values = [cytox for _, cytox in cytox_data]
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    n, bins, patches = plt.hist(cytox_values, bins=30, edgecolor='black', alpha=0.7)
    
    # Add Gaussian curve
    mu, std = norm.fit(cytox_values)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p * len(cytox_values) * (bins[1] - bins[0]), 'r-', linewidth=2)
    
    # Add labels and title
    plt.title("Distribution of Cytotoxicity in CAR Mutants", fontsize=14)
    plt.xlabel("Cytotoxicity", fontsize=12)
    plt.ylabel("Number of Mutants", fontsize=12)
    
    # Add legend and grid
    plt.legend(['Gaussian Fit', 'Data'])
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    plt.text(0.02, 0.98, 
             f'Mean: {mu:.2f}\nStd Dev: {std:.2f}',
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate FMC63-based CAR mutants with mutations in CD28 and CD3ζ domains")
    
    parser.add_argument('--n_mutants', type=int, default=DEFAULT_N_MUTANTS,
                       help=f'Number of mutants to generate (default: {DEFAULT_N_MUTANTS})')
    
    parser.add_argument('--max_mutations', type=int, default=DEFAULT_MAX_MUTATIONS,
                       help=f'Maximum number of mutations per domain (default: {DEFAULT_MAX_MUTATIONS})')
    
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                       help=f'Directory for output files (default: {DEFAULT_OUTPUT_DIR})')
    
    parser.add_argument('--output_name', type=str, default="CAR_mutants",
                       help='Base name for output files (default: CAR_mutants)')
    
    parser.add_argument('--random_seed', type=int, default=None,
                       help='Random seed for reproducibility (default: None)')
    
    # Cytotoxicity parameters
    parser.add_argument('--baseline_cytox', type=float, default=50.0,
                       help='Baseline cytotoxicity value for Gaussian distribution (default: 50.0)')
    
    parser.add_argument('--cytox_std_dev', type=float, default=20.0,
                       help='Standard deviation for cytotoxicity Gaussian distribution (default: 20.0)')
    
    # Plotting options
    parser.add_argument('--no_plots', action='store_true',
                       help='Skip generating plots')
    
    parser.add_argument('--plots_dir', type=str, default=DEFAULT_PLOTS_DIR,
                       help='Directory to save plots (default: plots)')
    
    return parser.parse_args()


def run_mutants(args):
    """
    Run the mutant generation with the provided arguments.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Tuple of paths to the CSV, TSV, and FASTA output files
    """
    # Show project root for path reference
    project_root = get_project_root()
    print(f"Project root: {project_root}")
    
    # Resolve output paths
    output_dir_path = resolve_path(args.output_dir)
    output_path = output_dir_path / args.output_name
    plots_dir_path = resolve_path(args.plots_dir)
    
    print(f"Generating {args.n_mutants} FMC63-CAR mutants with up to {args.max_mutations} mutations per domain...")
    
    # Generate mutants
    mutants = generate_car_mutants(
        args.n_mutants, 
        CD28_SEQ, 
        CD3Z_SEQ, 
        FMC63_SCFV, 
        args.max_mutations,
        args.random_seed
    )
    
    # Save mutants to files
    csv_path, tsv_path, fasta_path = save_mutants(mutants, output_path)
    
    print(f"✅ Generated {args.n_mutants} FMC63-CAR mutants")
    print(f"   CSV saved to: {csv_path}")
    print(f"   TSV saved to: {tsv_path}")
    print(f"   FASTA saved to: {fasta_path}")
    
    # Generate cytotoxicity data
    cytox_path = output_dir_path / f"{args.output_name}_cytox.csv"
    cytox_data = dummy_cytox_data(
        mutants,
        output_path=cytox_path,
        baseline_cytox=args.baseline_cytox,
        std_dev=args.cytox_std_dev,
        random_seed=args.random_seed
    )
    print(f"✅ Generated cytotoxicity data")
    print(f"   Cytotoxicity data saved to: {cytox_path}")
    
    # Generate plots if not disabled
    if not args.no_plots:
        # Create plots directory
        plots_dir_path.mkdir(parents=True, exist_ok=True)
        
        # Plot mutation distribution
        mutation_plot_path = plots_dir_path / "mutation_distribution.png"
        plot_mutants(mutants, mutation_plot_path)
        print(f"   Mutation distribution plot saved to: {mutation_plot_path}")
        
        # Plot cytotoxicity distribution
        cytox_plot_path = plots_dir_path / "cytotoxicity_distribution.png"
        plot_cytotoxicity(cytox_data, cytox_plot_path)
        print(f"   Cytotoxicity distribution plot saved to: {cytox_plot_path}")
    
    return csv_path, tsv_path, fasta_path, cytox_path


def main():
    args = parse_args()
    run_mutants(args)


if __name__ == "__main__":
    main()
