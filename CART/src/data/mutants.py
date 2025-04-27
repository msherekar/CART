# Generate FMC63-based CAR mutants with mutations in hinge, TM, and ICD of CD28 and CD3ζ

import random
import csv
from pathlib import Path

# Constant FMC63 scFv sequence (VH-linker-VL format)
FMC63_SCFV = (
    "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVAYISSGGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKYPHGYWYFDVWGQGTLVTVSSGGGGSGGGGSGGGGSEIVLTQSPGTLSLSPGERATLSCRASQSVSSSYLAWYQQKPGQAPRLLIYDASTRATGIPDRFSGSGSGTDFTLTISSLQPEDFATYYCQQYNSYPLTFGAGTKLEIK"
)

# Domain sequences: CD28(115–228), CD3ζ(52–164)
CD28_SEQ = "EVMYPPPYLDNEKSNGTIIHVKGKHLCPSPLFPGPSKPFWVLVVVGGVLACYSLLVTVAFIIFWVRSKRSRLLHSDYMNMTPRRPGPTRKHYQPYAPPRDFAAYRS"
CD3Z_SEQ = "RVKFSRSADAPAYQQGQNQLYNELNLGRREEYDVLDKRRGRDPEMGGKPQRRKNPQEGLYNELQKDKMAEAYSEIGMKGERRRGKGHDGLYQGLSTATKDTYDALHMQALPPR"

AA_LIST = "ACDEFGHIKLMNPQRSTVWY"
N_MUTANTS = 382
MAX_MUTATIONS = 10

def mutate_sequence(seq, max_mutations):
    num_mutations = random.randint(1, max_mutations)
    positions = random.sample(range(len(seq)), num_mutations)
    mutated = list(seq)
    for pos in positions:
        current = seq[pos]
        options = [aa for aa in AA_LIST if aa != current]
        mutated[pos] = random.choice(options)
    return ''.join(mutated)

def generate_car_mutants(n, cd28_seq, cd3z_seq, scfv, max_mutations):
    mutants = []
    for i in range(n):
        mut_cd28 = mutate_sequence(cd28_seq, max_mutations)
        mut_cd3z = mutate_sequence(cd3z_seq, max_mutations)
        full_seq = scfv + mut_cd28 + mut_cd3z
        mutants.append((f"CAR_mutant_{i+1}", full_seq))
    return mutants

def save_mutants(mutants, output_path: Path):
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


if __name__ == "__main__":
    output_base = Path("CAR_mutants")
    mutants = generate_car_mutants(N_MUTANTS, CD28_SEQ, CD3Z_SEQ, FMC63_SCFV, MAX_MUTATIONS)
    save_mutants(mutants, output_base)
    print(f"✅ Generated {N_MUTANTS} FMC63-CAR mutants to 'CAR_mutants.csv' and 'CAR_mutants.tsv'")
