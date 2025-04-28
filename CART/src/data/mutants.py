#!/usr/bin/env python3
"""
Generate FMC63-based CAR mutants with mutations in hinge, TM, and ICD of CD28 and CD3ζ
"""

import random
import csv
import argparse
from pathlib import Path

# Constant FMC63 scFv sequence (VH-linker-VL format)
FMC63_SCFV = ("EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVAYISSGGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKYPHGYWYFDVWGQGTLVTVSSGGGGSGGGGSGGGGSEIVLTQSPGTLSLSPGERATLSCRASQSVSSSYLAWYQQKPGQAPRLLIYDASTRATGIPDRFSGSGSGTDFTLTISSLQPEDFATYYCQQYNSYPLTFGAGTKLEIK")

# Domain sequences: CD28(115–228), CD3ζ(52–164)
CD28_SEQ = "EVMYPPPYLDNEKSNGTIIHVKGKHLCPSPLFPGPSKPFWVLVVVGGVLACYSLLVTVAFIIFWVRSKRSRLLHSDYMNMTPRRPGPTRKHYQPYAPPRDFAAYRS"
CD3Z_SEQ = "RVKFSRSADAPAYQQGQNQLYNELNLGRREEYDVLDKRRGRDPEMGGKPQRRKNPQEGLYNELQKDKMAEAYSEIGMKGERRRGKGHDGLYQGLSTATKDTYDALHMQALPPR"

AA_LIST = "ACDEFGHIKLMNPQRSTVWY"
DEFAULT_N_MUTANTS = 382
DEFAULT_MAX_MUTATIONS = 10
DEFAULT_OUTPUT_DIR = "CART/mutants"


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


def mutate_sequence(seq, max_mutations):
    """
    Create a mutated version of a sequence with random amino acid substitutions.
    
    Args:
        seq: Original amino acid sequence
        max_mutations: Maximum number of mutations to introduce
        
    Returns:
        Mutated sequence
    """
    num_mutations = random.randint(1, max_mutations)
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
    
    Args:
        n: Number of mutants to generate
        cd28_seq: CD28 domain sequence
        cd3z_seq: CD3ζ domain sequence
        scfv: scFv sequence (constant)
        max_mutations: Maximum number of mutations per domain
        random_seed: Seed for random number generator (optional)
        
    Returns:
        List of (mutant_id, mutant_sequence) tuples
    """
    if random_seed is not None:
        random.seed(random_seed)
        
    mutants = []
    for i in range(n):
        mut_cd28 = mutate_sequence(cd28_seq, max_mutations)
        mut_cd3z = mutate_sequence(cd3z_seq, max_mutations)
        full_seq = scfv + mut_cd28 + mut_cd3z
        mutants.append((f"CAR_mutant_{i+1}", full_seq))
    return mutants


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
    
    # Resolve output path
    output_dir_path = resolve_path(args.output_dir)
    output_path = output_dir_path / args.output_name
    
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
    
    # Save to files
    csv_path, tsv_path, fasta_path = save_mutants(mutants, output_path)
    
    print(f"✅ Generated {args.n_mutants} FMC63-CAR mutants")
    print(f"   CSV saved to: {csv_path}")
    print(f"   TSV saved to: {tsv_path}")
    print(f"   FASTA saved to: {fasta_path}")
    
    return csv_path, tsv_path, fasta_path


def main():
    args = parse_args()
    run_mutants(args)


if __name__ == "__main__":
    main()
