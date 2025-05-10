import re
from typing import Union, List
from pathlib import Path

def validate_fasta(sequence: str) -> bool:
    """
    Validate if a string is in FASTA format.
    
    Args:
        sequence: String to validate
        
    Returns:
        bool: True if valid FASTA format, False otherwise
    """
    # Check if sequence starts with '>'
    if not sequence.startswith('>'):
        return False
    
    # Split into header and sequence
    parts = sequence.split('\n', 1)
    if len(parts) != 2:
        return False
    
    header, seq = parts
    
    # Validate header
    if not header.startswith('>'):
        return False
    
    # Validate sequence (only allow standard amino acids)
    valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
    seq = seq.replace('\n', '').strip()
    if not all(aa in valid_aa for aa in seq):
        return False
    
    return True

def read_fasta(file_path: Union[str, Path]) -> List[tuple]:
    """
    Read sequences from a FASTA file.
    
    Args:
        file_path: Path to FASTA file
        
    Returns:
        List of tuples containing (header, sequence)
    """
    sequences = []
    current_header = None
    current_sequence = []
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_header is not None:
                    sequences.append((current_header, ''.join(current_sequence)))
                current_header = line[1:]
                current_sequence = []
            else:
                current_sequence.append(line)
        
        if current_header is not None:
            sequences.append((current_header, ''.join(current_sequence)))
    
    return sequences

def write_fasta(sequences: List[tuple], output_path: Union[str, Path]) -> None:
    """
    Write sequences to a FASTA file.
    
    Args:
        sequences: List of tuples containing (header, sequence)
        output_path: Path to write FASTA file
    """
    with open(output_path, 'w') as f:
        for header, sequence in sequences:
            f.write(f'>{header}\n')
            # Write sequence in chunks of 60 characters
            for i in range(0, len(sequence), 60):
                f.write(sequence[i:i+60] + '\n') 