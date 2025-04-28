#!/usr/bin/env python3
"""
Run cross-validation experiments for ESM2-8M models and generate comparison plots.
"""
import argparse
from pathlib import Path
import numpy as np
from typing import Dict, List
import torch
from .finetuning import run_finetuning, parse_args
from .visualization import (
    plot_training_metrics,
    plot_spearman_vs_epoch,
    plot_model_comparison,
    plot_recall_precision
)

def run_cross_validation(args, n_folds: int = 5) -> Dict[str, Dict[str, List[float]]]:
    """Run cross-validation experiments."""
    results = {
        'pretrained': {'spearman': [], 'recall': {5: [], 10: []}, 'precision': {5: [], 10: []}},
        'high': {'spearman': [], 'recall': {5: [], 10: []}, 'precision': {5: [], 10: []}},
        'low': {'spearman': [], 'recall': {5: [], 10: []}, 'precision': {5: [], 10: []}}
    }
    
    for fold in range(n_folds):
        print(f"\nRunning fold {fold + 1}/{n_folds}")
        
        # Run pretrained model
        print("\nEvaluating pretrained model...")
        pretrained_metrics = evaluate_pretrained(args)
        for metric in ['spearman', 'recall', 'precision']:
            if metric == 'spearman':
                results['pretrained'][metric].append(pretrained_metrics[metric])
            else:
                for k in [5, 10]:
                    results['pretrained'][metric][k].append(pretrained_metrics[metric][k])
        
        # Run high diversity fine-tuning
        print("\nFine-tuning on high diversity sequences...")
        args.group = 'high'
        high_metrics = run_finetuning(args)
        for metric in ['spearman', 'recall', 'precision']:
            if metric == 'spearman':
                results['high'][metric].append(high_metrics[metric])
            else:
                for k in [5, 10]:
                    results['high'][metric][k].append(high_metrics[metric][k])
        
        # Run low diversity fine-tuning
        print("\nFine-tuning on low diversity sequences...")
        args.group = 'low'
        low_metrics = run_finetuning(args)
        for metric in ['spearman', 'recall', 'precision']:
            if metric == 'spearman':
                results['low'][metric].append(low_metrics[metric])
            else:
                for k in [5, 10]:
                    results['low'][metric][k].append(low_metrics[metric][k])
    
    return results

def evaluate_pretrained(args) -> Dict[str, float]:
    """Evaluate the pretrained model."""
    # Load pretrained model
    model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    
    # Prepare dataset
    dataset = SequenceDataset(args.high_fasta, tokenizer, max_length=args.max_length)
    val_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    )
    
    # Compute metrics
    return compute_metrics(model, val_loader, args.device)

def generate_plots(results: Dict[str, Dict[str, List[float]]], plots_dir: Path) -> None:
    """Generate all comparison plots."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Training and Validation Loss
    for model in ['high', 'low']:
        plot_training_metrics(
            results[model]['train_loss'],
            results[model]['val_loss'],
            str(plots_dir / f'{model}_training_loss.png'),
            f'ESM2-8M ({model.capitalize()} Diversity)'
        )
    
    # 2. Spearman Correlation vs Epoch
    for model in ['high', 'low']:
        spearman_data = {
            'mean': np.mean(results[model]['spearman'], axis=0),
            'std': np.std(results[model]['spearman'], axis=0)
        }
        plot_spearman_vs_epoch(
            spearman_data,
            str(plots_dir / f'{model}_spearman_vs_epoch.png'),
            f'ESM2-8M ({model.capitalize()} Diversity)'
        )
    
    # 3. Model Comparison
    model_scores = {
        'Pretrained': results['pretrained']['spearman'],
        'High Diversity': results['high']['spearman'],
        'Low Diversity': results['low']['spearman']
    }
    plot_model_comparison(
        model_scores,
        str(plots_dir / 'model_comparison.png')
    )
    
    # 4. Recall and Precision
    metrics_data = {
        'recall': {
            'Pretrained': [np.mean(results['pretrained']['recall'][5]), np.mean(results['pretrained']['recall'][10])],
            'High Diversity': [np.mean(results['high']['recall'][5]), np.mean(results['high']['recall'][10])],
            'Low Diversity': [np.mean(results['low']['recall'][5]), np.mean(results['low']['recall'][10])]
        },
        'precision': {
            'Pretrained': [np.mean(results['pretrained']['precision'][5]), np.mean(results['pretrained']['precision'][10])],
            'High Diversity': [np.mean(results['high']['precision'][5]), np.mean(results['high']['precision'][10])],
            'Low Diversity': [np.mean(results['low']['precision'][5]), np.mean(results['low']['precision'][10])]
        }
    }
    plot_recall_precision(
        metrics_data,
        str(plots_dir / 'recall_precision.png')
    )

def parse_args():
    parser = argparse.ArgumentParser(description="Run cross-validation experiments for ESM2-8M models")
    parser.add_argument('--n_folds', type=int, default=5, help="Number of cross-validation folds")
    parser.add_argument('--plots_dir', type=Path, default=Path("plots"), help="Directory to save plots")
    return parser.parse_args()

def main():
    args = parse_args()
    # Run cross-validation experiments
    results = run_cross_validation(args, n_folds=args.n_folds)
    # Generate plots
    generate_plots(results, args.plots_dir)

if __name__ == "__main__":
    main() 