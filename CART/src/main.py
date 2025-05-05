#!/usr/bin/env python3
"""
CART-Project pipeline orchestrator (main.py)

CLI-driven entrypoint to run one or more steps of the CAR-T pipeline.
You can override paths via flags. Each submodule has its own `parse_args` and `run_<step>` functions.
"""
import argparse
from pathlib import Path
import sys

# Import each module's parser and runner
from CART.src.data.augmentation import parse_args as aug_parse_args_orig, run_augmentation
from CART.src.data.mds import parse_args as mds_parse_args_orig, run_mds
from CART.src.data.mutants import parse_args as mut_parse_args_orig, run_mutants
from CART.src.modeling.finetuning import parse_args as ft_parse_args_orig, run_finetuning
from CART.src.modeling.embeddings import parse_args as emb_parse_args_orig, run_embeddings
from CART.src.modeling.prediction import parse_args as pred_parse_args_orig, run_prediction
from CART.src.modeling.evaluation import parse_args as eval_parse_args_orig, run_evaluation
from CART.src.modeling.score import parse_args as score_parse_args_orig, run_score
# Fix for the buggy parse_args functions in each module
def safe_parse_args(parser_func, args_list):
    """Safely call a module's parse_args function with our own args list"""
    # Save the original sys.argv
    orig_argv = sys.argv
    try:
        # Temporarily replace sys.argv with a minimal argv that won't interfere
        sys.argv = ['dummy_program']
        # Create a fresh parser and parse our args_list directly
        return parser_func(args_list)
    finally:
        # Restore original sys.argv
        sys.argv = orig_argv

# Use the safe wrapper for each module's parse_args
aug_parse_args = lambda args_list: safe_parse_args(aug_parse_args_orig, args_list)
mds_parse_args = lambda args_list: safe_parse_args(mds_parse_args_orig, args_list)
mut_parse_args = lambda args_list: safe_parse_args(mut_parse_args_orig, args_list)
ft_parse_args = lambda args_list: safe_parse_args(ft_parse_args_orig, args_list)
emb_parse_args = lambda args_list: safe_parse_args(emb_parse_args_orig, args_list)
pred_parse_args = lambda args_list: safe_parse_args(pred_parse_args_orig, args_list)
eval_parse_args = lambda args_list: safe_parse_args(eval_parse_args_orig, args_list)
score_parse_args = lambda args_list: safe_parse_args(score_parse_args_orig, args_list)


def run_pipeline():
    # project root is the absolute path to CART-Project
    project_root = Path(__file__).resolve().parents[2].absolute()
    print(f"Project root: {project_root}")
    
    # Make sure output directories exist
    (project_root/'output').mkdir(exist_ok=True)
    (project_root/'output'/'augmentation').mkdir(exist_ok=True, parents=True)
    (project_root/'output'/'mds').mkdir(exist_ok=True, parents=True)
    (project_root/'output'/'mutants').mkdir(exist_ok=True, parents=True)
    (project_root/'output'/'plots').mkdir(exist_ok=True, parents=True)
    (project_root/'output'/'models').mkdir(exist_ok=True, parents=True)

    # top-level CLI for selecting steps and base paths
    parser = argparse.ArgumentParser(description="Run CART pipeline steps")
    parser.add_argument(
        '-s', '--steps', nargs='+', choices=['all','augmentation','mds','mutants','finetuning','embeddings','prediction','evaluation','score'],
        default=['all'], help='Which steps to run'
    )
    parser.add_argument(
        '--fasta-dir', type=Path,
        default=project_root/'CART'/'fasta', help='Folder of input FASTA files'
    )
    parser.add_argument(
        '--output-dir', type=Path,
        default=project_root/'output', help='Root for all outputs'
    )
    parser.add_argument(
        '--model-dir', type=Path,
        default=project_root/'output'/'models', help='Folder for models'
    )
    args = parser.parse_args()
    
    print(f"FASTA directory: {args.fasta_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Model directory: {args.model_dir}")

    selected = set(args.steps)
    if 'all' in selected:
        selected = {'augmentation', 'mds', 'mutants', 'finetuning', 
                   'embeddings', 'prediction', 'evaluation', 'score'}

    # Step 1: augmentation
    if 'augmentation' in selected:
        print('==> [1/8] Data augmentation')
        aug_args = aug_parse_args([
            '--wt_cd28', str(args.fasta_dir/'wt_cd28.fasta'),
            '--wt_cd3z', str(args.fasta_dir/'wt_cd3z.fasta'),
            '--uniprot_db', str(args.fasta_dir/'uniprot_trembl.fasta'),
            '--output_dir', str(args.output_dir/'augmentation'),
            '--plots_dir', str(args.output_dir/'augmentation'/'plots')
        ])
        run_augmentation(aug_args)

    # Step 2: MDS
    if 'mds' in selected:
        print('==> [2/8] MDS analysis')
        mds_args = mds_parse_args([
            '--high_fasta', str(args.output_dir/'augmentation'/'high_diversity.fasta'),
            '--low_fasta', str(args.output_dir/'augmentation'/'low_diversity.fasta'),
            '--output_dir', str(args.output_dir/'mds')
        ])
        run_mds(mds_args)

    # Step 3: mutants
    if 'mutants' in selected:
        print('==> [3/8] Mutants processing')
        mut_args = mut_parse_args([
            '--output_dir', str(args.output_dir/'mutants'),
            '--plots_dir', str(args.output_dir/'plots')
        ])
        run_mutants(mut_args)

    # Step 4: finetuning
    if 'finetuning' in selected:
        print('==> [4/8] Model finetuning')
        ft_args = ft_parse_args([
            '--high_fasta', str(args.output_dir/'augmentation'/'high_diversity.fasta'),
            '--low_fasta', str(args.output_dir/'augmentation'/'low_diversity.fasta'),
            '--output_dir', str(args.model_dir),
            '--max_epochs', '51'
        ])
        for group in ['high', 'low']:
            run_finetuning(ft_args, group)
    
    # Step 5: Extract Embeddings
    if 'embeddings' in selected:
        print('==> [5/8] Extract embeddings')
        emb_args = emb_parse_args([
            '--mutant_fasta', str(args.output_dir/'mutants'/'CAR_mutants.fasta'),
            '--pretrained', 'facebook/esm2_t6_8M_UR50D',
            '--finetuned_high', str(args.model_dir/'high'),
            '--finetuned_low', str(args.model_dir/'low'),
            '--out_dir', str(args.output_dir/'embeddings'),
            '--device', 'auto'
        ])
        run_embeddings(emb_args)
        
    # Step 6: Prediction
    if 'prediction' in selected:
        print('==> [6/8] Running predictions')
        pred_args = pred_parse_args([
            '--embedding_dir', str(args.output_dir/'embeddings'),
            '--labels_path', str(args.output_dir/'mutants'/'CAR_mutants_cytox.csv'),
            '--output_dir', str(args.output_dir/'results'),
            '--n_splits', '5',
            '--random_seed', '42'
        ])
        run_prediction(pred_args)
    
    # Step 7: Evaluation
    if 'evaluation' in selected:
        print('==> [7/8] Running evaluation')
        # Find all embedding files in the embeddings directory
        embeddings_dir = args.output_dir/'embeddings'
        
        # Create embeddings_files list as absolute paths to .npy files
        eval_args = eval_parse_args([
            '--embed_paths', 
            str(embeddings_dir/'pretrained_embeddings.npy'), 
            str(embeddings_dir/'finetuned_high_embeddings.npy'),
            str(embeddings_dir/'finetuned_low_embeddings.npy'),
            '--output_dir', str(args.output_dir/'evaluation'),
            '--labels_path', str(args.output_dir/'mutants'/'CAR_mutants_cytox.csv'),
            '--use_real_labels',
            '--random_seed', '42',
            '--n_splits', '5'
        ])
        run_evaluation(eval_args)
        
    # Step 8: Model Scoring
    if 'score' in selected:
        print('==> [8/8] Running model scoring')
        score_args = score_parse_args([
            '--predictions_dir', str(args.output_dir/'results'),
            '--output_file', str(args.output_dir/'model_scores.json'),
            '--output_dir', str(args.output_dir/'plots'),
            '--k_values', '5', '10', '20'
        ])
        run_score(score_args)


if __name__ == '__main__':
    run_pipeline()
