# tests/test_data.py
from pathlib import Path
import numpy as np
import pytest
from CART.src.data.augmentation import remove_duplicates, filter_by_length
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
import tempfile

@pytest.fixture
def small_fasta(tmp_path):
    # create FASTA with duplicates and varied lengths
    records = [
        SeqRecord(Seq("AAA"), id="a1"),
        SeqRecord(Seq("AAA"), id="a2"),
        SeqRecord(Seq("AAAA"), id="b1"),
        SeqRecord(Seq("AA"), id="c1"),
    ]
    path = tmp_path/"test.fa"
    SeqIO.write(records, path, "fasta")
    return path

def test_dedupe_and_filter(small_fasta, tmp_path):
    # Read sequences from FASTA
    records = list(SeqIO.parse(small_fasta, "fasta"))
    sequences = [str(record.seq) for record in records]
    
    # Test remove_duplicates
    unique_seqs = remove_duplicates(sequences)
    assert len(unique_seqs) == 3  # Should collapse AAA duplicates to one
    
    # Test filter_by_length
    filtered_seqs = filter_by_length(unique_seqs, ref_len=3, tol=0.5)
    assert all(2 <= len(s) <= 4 for s in filtered_seqs)  # Length should be within 50% of reference length
