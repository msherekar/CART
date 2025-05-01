#!/usr/bin/env python3
"""
score.py

Computes evaluation metrics for CAR-T cell activity predictions:
- Spearman's correlation
- Recall@K
- Precision@K

Can be used to evaluate prediction results stored in files or directly process arrays.
"""

import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import json

def get_project_root():
    """Get project root directory relative to this file"""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def recall_precision_at_k(y_true, y_pred, K):
    """
    y_true: 1D array of true continuous scores
    y_pred: 1D array of predicted continuous scores
    K:      integer top-K to evaluate
    """
    n = len(y_true)
    # label positives = top 25% by true value
    cutoff = np.percentile(y_true, 75)
    positives = (y_true >= cutoff)
    n_pos = positives.sum()

    # get indices of top-K predictions
    topk_idx = np.argsort(y_pred)[-K:]
    tp = positives[topk_idx].sum()

    recall = tp / n_pos if n_pos > 0 else 0.0
    precision = tp / K
    return recall, precision

def parse_args():
    project_root = get_project_root()
    results_dir = os.path.join(project_root, "CART", "predictions")
    
    parser = argparse.ArgumentParser(description="Score predictions with various metrics")
    
    # Input options
    parser.add_argument(
        "--predictions_file", 
        type=str, 
        help="Path to .npz file containing predictions and actual values"
    )
    parser.add_argument(
        "--predictions_dir", 
        type=str, 
        default=results_dir,
        help="Directory containing multiple prediction files to evaluate"
    )
    parser.add_argument(
        "--model_names", 
        nargs="+", 
        help="Model names to score (used with --predictions_dir)"
    )
    
    # Output options
    parser.add_argument(
        "--output_file", 
        type=str, 
        help="Path to save results (JSON format)"
    )
    
    # Scoring options
    parser.add_argument(
        "--k_values", 
        type=int, 
        nargs="+", 
        default=[5, 10, 20],
        help="K values for Recall@K and Precision@K metrics"
    )
    
    return parser.parse_args()

def score_predictions(y_true, y_pred, k_values):
    """
    Score predictions with various metrics
    
    Args:
        y_true: True values
        y_pred: Predicted values
        k_values: List of K values for recall@K and precision@K
        
    Returns:
        Dictionary with scores
    """
    # 1) Spearman's ρ
    rho, pval = spearmanr(y_true, y_pred)
    
    # 2) Recall@K and Precision@K for each K value
    recall_at_k = {}
    precision_at_k = {}
    
    for k in k_values:
        r, p = recall_precision_at_k(y_true, y_pred, K=k)
        recall_at_k[k] = r
        precision_at_k[k] = p
    
    return {
        "spearman_rho": float(rho),
        "spearman_pval": float(pval),
        "recall_at_k": recall_at_k,
        "precision_at_k": precision_at_k
    }

def plot_recall_precision(scores, model_name, output_dir):
    """
    Plot Recall@K and Precision@K metrics for different K values
    
    Args:
        scores: Dictionary containing recall_at_k and precision_at_k
        model_name: Name of the model being evaluated
        output_dir: Directory to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Get K values and corresponding metrics
    k_values = sorted(scores['recall_at_k'].keys())
    recalls = [scores['recall_at_k'][k] for k in k_values]
    precisions = [scores['precision_at_k'][k] for k in k_values]
    
    # Plot Recall and Precision
    plt.plot(k_values, recalls, 'o-', label='Recall@K', linewidth=2, markersize=8)
    plt.plot(k_values, precisions, 's-', label='Precision@K', linewidth=2, markersize=8)
    
    # Add value labels
    for k, r, p in zip(k_values, recalls, precisions):
        plt.text(k, r, f'{r:.2f}', ha='center', va='bottom')
        plt.text(k, p, f'{p:.2f}', ha='center', va='top')
    
    # Customize plot
    plt.title(f'Recall and Precision Metrics - {model_name}', fontsize=14)
    plt.xlabel('Top-K', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.xticks(k_values, [f'Top-{k}' for k in k_values])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    # Adjust layout and save
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{model_name}_recall_precision.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved recall/precision plot to {plot_path}")

def run_score(args):
    """
    Run scoring based on command line arguments
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Dictionary with scores for each model
    """
    all_scores = {}
    
    # Create output directory for plots if it doesn't exist
    plot_dir = os.path.join(os.path.dirname(args.predictions_dir), "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    # Case 1: Single predictions file
    if args.predictions_file:
        file_path = args.predictions_file
        if not os.path.isabs(file_path):
            file_path = os.path.join(get_project_root(), file_path)
            
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Predictions file not found: {file_path}")
            
        print(f"Scoring predictions from {file_path}")
        
        # Load predictions and actuals
        data = np.load(file_path)
        
        # Different files might have different keys
        # Try common options for actuals and predictions
        actual_keys = ['actuals', 'y_test', 'y_tests', 'y_true']
        pred_keys = ['predictions', 'y_pred', 'y_preds']
        
        y_true = None
        for key in actual_keys:
            if key in data:
                y_true = data[key]
                break
                
        y_pred = None
        for key in pred_keys:
            if key in data:
                y_pred = data[key]
                break
                
        if y_true is None or y_pred is None:
            print(f"Could not find prediction/actual arrays in {file_path}")
            print(f"Available keys: {list(data.keys())}")
            return {}
            
        # Score predictions
        model_name = os.path.splitext(os.path.basename(file_path))[0].replace("_predictions", "")
        all_scores[model_name] = score_predictions(y_true, y_pred, args.k_values)
        
    # Case 2: Directory with multiple prediction files
    elif args.predictions_dir:
        dir_path = args.predictions_dir
        if not os.path.isabs(dir_path):
            dir_path = os.path.join(get_project_root(), dir_path)
            
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"Predictions directory not found: {dir_path}")
            
        # Get prediction files
        if args.model_names:
            pred_files = [os.path.join(dir_path, f"{model}_predictions.npz") for model in args.model_names]
        else:
            pred_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) 
                         if f.endswith('_predictions.npz')]
            
        if not pred_files:
            print(f"No prediction files found in {dir_path}")
            return {}
            
        print(f"Found {len(pred_files)} prediction files in {dir_path}")
        
        # Score each model
        for file_path in pred_files:
            if not os.path.exists(file_path):
                print(f"Warning: File not found: {file_path}")
                continue
                
            print(f"Scoring predictions from {file_path}")
            
            # Load predictions and actuals
            data = np.load(file_path)
            
            # Different files might have different keys
            # Try common options for actuals and predictions
            actual_keys = ['actuals', 'y_test', 'y_tests', 'y_true']
            pred_keys = ['predictions', 'y_pred', 'y_preds']
            
            y_true = None
            for key in actual_keys:
                if key in data:
                    y_true = data[key]
                    break
                    
            y_pred = None
            for key in pred_keys:
                if key in data:
                    y_pred = data[key]
                    break
                    
            if y_true is None or y_pred is None:
                print(f"Could not find prediction/actual arrays in {file_path}")
                print(f"Available keys: {list(data.keys())}")
                continue
                
            # Score predictions
            model_name = os.path.splitext(os.path.basename(file_path))[0].replace("_predictions", "")
            all_scores[model_name] = score_predictions(y_true, y_pred, args.k_values)
    
    # Print scores to console and generate plots
    for model_name, scores in all_scores.items():
        print(f"\nScores for {model_name}:")
        print(f"  Spearman's ρ  : {scores['spearman_rho']:.3f} (p={scores['spearman_pval']:.3g})")
        
        for k, r in scores['recall_at_k'].items():
            p = scores['precision_at_k'][k]
            print(f"  Recall@{k}      : {r:.3f}")
            print(f"  Precision@{k}   : {p:.3f}")
        
        # Generate recall/precision plot
        plot_recall_precision(scores, model_name, plot_dir)
    
    # Save results to file if specified
    if args.output_file:
        output_path = args.output_file
        if not os.path.isabs(output_path):
            output_path = os.path.join(get_project_root(), output_path)
            
        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert to serializable format
        serializable_scores = {}
        for model, scores in all_scores.items():
            serializable_scores[model] = {
                "spearman_rho": float(scores["spearman_rho"]),
                "spearman_pval": float(scores["spearman_pval"]),
                "recall_at_k": {str(k): float(v) for k, v in scores["recall_at_k"].items()},
                "precision_at_k": {str(k): float(v) for k, v in scores["precision_at_k"].items()},
            }
        
        with open(output_path, 'w') as f:
            json.dump(serializable_scores, f, indent=2)
        print(f"\nSaved scores to {output_path}")
    
    return all_scores

def main():
    args = parse_args()
    run_score(args)

if __name__ == "__main__":
    main()
