#!/usr/bin/env python3
"""
CART-Project pipeline orchestrator (main.py)

CLI-driven entrypoint to run one or more steps of the CAR-T pipeline.
Receives all paths from CLI (typer) or uses defaults if needed.
"""

import sys
from pathlib import Path
import argparse
from Bio import SeqIO

# Import each module's parser and runner
from CART.src.data.augmentation import parse_args as aug_parse_args_orig, run_augmentation
from CART.src.data.mds import parse_args as mds_parse_args_orig, run_mds
from CART.src.data.mutants import parse_args as mut_parse_args_orig, run_mutants
from CART.src.modeling.finetuning import parse_args as ft_parse_args_orig, run_finetuning
from CART.src.modeling.embeddings import parse_args as emb_parse_args_orig, run_embeddings
from CART.src.modeling.prediction import parse_args as pred_parse_args_orig, run_prediction
from CART.src.modeling.evaluation import parse_args as eval_parse_args_orig, run_evaluation
from CART.src.modeling.score import parse_args as score_parse_args_orig, run_score

# --- Safe argument parser for each module ---
def safe_parse_args(parser_func, args_list):
    orig_argv = sys.argv
    try:
        sys.argv = ['dummy_program']
        return parser_func(args_list)
    finally:
        sys.argv = orig_argv

aug_parse_args = lambda args_list: safe_parse_args(aug_parse_args_orig, args_list)
mds_parse_args = lambda args_list: safe_parse_args(mds_parse_args_orig, args_list)
mut_parse_args = lambda args_list: safe_parse_args(mut_parse_args_orig, args_list)
ft_parse_args = lambda args_list: safe_parse_args(ft_parse_args_orig, args_list)
emb_parse_args = lambda args_list: safe_parse_args(emb_parse_args_orig, args_list)
pred_parse_args = lambda args_list: safe_parse_args(pred_parse_args_orig, args_list)
eval_parse_args = lambda args_list: safe_parse_args(eval_parse_args_orig, args_list)
score_parse_args = lambda args_list: safe_parse_args(score_parse_args_orig, args_list)

# --- Main orchestrator ---
def run_pipeline(
    steps=['all'],
    fasta_dir=Path('fasta'),
    output_dir=Path('output'),
    model_dir=Path('output/models'),
    domain_1=None,
    domain_2=None,
    cytotox_csv=None,
    esm_model="facebook/esm2_t6_8M_UR50D",
    uniprot_db=None,
    wt_car=None,
):
    # Validate required files
    if not domain_1 or not domain_2 or not cytotox_csv:
        raise ValueError("All input files (domain_1, domain_2, cytotox_csv) must be provided.")

    # Validate Uniprot DB path
    if not uniprot_db:
        raise ValueError("Uniprot database path (uniprot_db) must be provided. This file should be on your local machine.")
    
    if not Path(uniprot_db).exists():
        raise ValueError(f"Uniprot database file not found at {uniprot_db}. Please provide a valid path.")
    
    print(f"Using Uniprot database: {uniprot_db}")
    
    # Read sequence files if provided
    domain_1_seq = None
    domain_2_seq = None
    wt_car_seq = None
    
    if domain_1 and Path(domain_1).exists():
        
        try:
            domain_1_seq = str(next(SeqIO.parse(domain_1, "fasta")).seq)
            print(f"Loaded domain 1 sequence: {domain_1_seq[:20]}...")
        except Exception as e:
            print(f"Warning: Could not read domain 1 sequence from {domain_1}: {e}")
    
    if domain_2 and Path(domain_2).exists():
        
        try:
            domain_2_seq = str(next(SeqIO.parse(domain_2, "fasta")).seq)
            print(f"Loaded domain 2 sequence: {domain_2_seq[:20]}...")
        except Exception as e:
            print(f"Warning: Could not read domain 2 sequence from {domain_2}: {e}")
    
    if wt_car and Path(wt_car).exists():
        
        try:
            wt_car_seq = str(next(SeqIO.parse(wt_car, "fasta")).seq)
            print(f"Loaded CAR sequence: {wt_car_seq[:20]}...")
        except Exception as e:
            print(f"Warning: Could not read CAR sequence from {wt_car}: {e}")

    selected = set(steps)
    if 'all' in selected:
        selected = {'augmentation', 'mds', 'mutants', 'finetuning', 'embeddings', 'prediction', 'evaluation', 'score'}

    # --- Create output folders if needed ---
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir/'augmentation').mkdir(parents=True, exist_ok=True)
    (output_dir/'mds').mkdir(parents=True, exist_ok=True)
    (output_dir/'mutants').mkdir(parents=True, exist_ok=True)
    (output_dir/'plots').mkdir(parents=True, exist_ok=True)
    (output_dir/'models').mkdir(parents=True, exist_ok=True)
    (output_dir/'embeddings').mkdir(parents=True, exist_ok=True)
    (output_dir/'results').mkdir(parents=True, exist_ok=True)
    
    print(f"Using ESM model: {esm_model}")

    # --- Pipeline steps ---
    if 'augmentation' in selected:
        print('==> [1/8] Data augmentation')
        aug_args = aug_parse_args([
            '--domain_1', str(domain_1),
            '--domain_2', str(domain_2),
            '--uniprot_db', str(uniprot_db),
            '--output_dir', str(output_dir/'augmentation'),
            '--plots_dir', str(output_dir/'augmentation'/'plots')
        ])
        run_augmentation(aug_args)

    if 'mds' in selected:
        print('==> [2/8] MDS analysis')
        mds_args = mds_parse_args([
            '--high_fasta', str(output_dir/'augmentation'/'high_diversity.fasta'),
            '--low_fasta', str(output_dir/'augmentation'/'low_diversity.fasta'),
            '--output_dir', str(output_dir/'mds')
        ])
        run_mds(mds_args)

    if 'mutants' in selected:
        print('==> [3/8] Mutants processing')
        mut_args = mut_parse_args([
            '--output_dir', str(output_dir/'mutants'),
            '--plots_dir', str(output_dir/'plots')
        ])
        run_mutants(mut_args, wt_car_seq, domain_1_seq, domain_2_seq)

    if 'finetuning' in selected:
        print('==> [4/8] Model finetuning')
        for group in ['high', 'low']:
            ft_args = ft_parse_args([
                '--high_fasta', str(output_dir/'augmentation'/'high_diversity.fasta'),
                '--low_fasta', str(output_dir/'augmentation'/'low_diversity.fasta'),
                '--output_dir', str(model_dir),
                '--model_name', str(esm_model),
                '--max_epochs', '51'
            ])
            run_finetuning(ft_args, group)

    if 'embeddings' in selected:
        print('==> [5/8] Extract embeddings')
        emb_args = emb_parse_args([
            '--mutant_fasta', str(output_dir/'mutants'/'CAR_mutants.fasta'),
            '--pretrained', str(esm_model),
            '--finetuned_high', str(model_dir/'high'),
            '--finetuned_low', str(model_dir/'low'),
            '--out_dir', str(output_dir/'embeddings'),
            '--device', 'auto'
        ])
        run_embeddings(emb_args)

    if 'prediction' in selected:
        print('==> [6/8] Running predictions')
        pred_args = pred_parse_args([
            '--embedding_dir', str(output_dir/'embeddings'),
            '--labels_path', str(cytotox_csv),
            '--output_dir', str(output_dir/'results'),
            '--n_splits', '5',
            '--random_seed', '42'
        ])
        run_prediction(pred_args)

    if 'evaluation' in selected:
        print('==> [7/8] Running evaluation')
        eval_args = eval_parse_args([
            '--embed_paths', 
            str(output_dir/'embeddings'/'pretrained_embeddings.npy'), 
            str(output_dir/'embeddings'/'finetuned_high_embeddings.npy'),
            str(output_dir/'embeddings'/'finetuned_low_embeddings.npy'),
            '--output_dir', str(output_dir/'evaluation'),
            '--labels_path', str(cytotox_csv),
            '--use_real_labels',
            '--random_seed', '42',
            '--n_splits', '5'
        ])
        run_evaluation(eval_args)

    if 'score' in selected:
        print('==> [8/8] Running model scoring')
        score_args = score_parse_args([
            '--predictions_dir', str(output_dir/'results'),
            '--output_file', str(output_dir/'model_scores.json'),
            '--output_dir', str(output_dir/'plots'),
            '--k_values', '5', '10', '20'
        ])
        run_score(score_args)


if __name__ == "__main__":
    run_pipeline()


