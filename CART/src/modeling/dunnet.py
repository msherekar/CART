#!/usr/bin/env python3
"""
Dunnett's test for multiple comparison procedure.

This script compares Spearman correlation scores between a control model
(typically a pretrained model) and one or more treatment models (typically finetuned models).
It loads results from the predictions directory and performs statistical significance testing.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import dunnett
import glob
import matplotlib.pyplot as plt
import seaborn as sns

def get_project_root() -> Path:
    """Get project root directory relative to this file"""
    return Path(__file__).resolve().parents[3]  # Go up 3 levels from this file

def parse_args(args_list=None):
    project_root = get_project_root()
    results_dir = project_root / "CART" / "output" / "results"
    
    parser = argparse.ArgumentParser(description="Run Dunnett's test for multiple comparison procedure")
    parser.add_argument(
        "--results_dir", 
        type=Path, 
        default=results_dir,
        help="Directory containing model prediction results"
    )
    parser.add_argument(
        "--control_model", 
        type=str, 
        default="pretrained",
        help="Name of the control model (without _spearman.npy suffix)"
    )
    parser.add_argument(
        "--treatment_models", 
        type=str, 
        nargs="+",
        help="Names of treatment models to compare against control (without _spearman.npy suffix)"
    )
    parser.add_argument(
        "--output_dir", 
        type=Path, 
        default=None,
        help="Output directory for test results (defaults to results_dir)"
    )
    parser.add_argument(
        "--alpha", 
        type=float, 
        default=0.05,
        help="Significance level for Dunnett's test"
    )
    parser.add_argument(
        "--plot", 
        action="store_true",
        help="Generate comparison plots"
    )
    
    # Only parse command line arguments if this module is run directly
    if args_list is None and __name__ == "__main__":
        return parser.parse_args()
    else:
        # When imported, use the provided args_list or an empty list
        return parser.parse_args(args_list or [])

def load_spearman_scores(results_dir, model_name):
    """Load Spearman correlation scores for a model"""
    # Try multiple possible file patterns
    possible_patterns = [
        f"{model_name}_spearman.npy",
        f"{model_name}_correlations.npy",
        f"pll_results_{model_name}_spearman.npy"
    ]
    
    for pattern in possible_patterns:
        file_path = results_dir / pattern
        if file_path.exists():
            scores = np.load(file_path)
            if isinstance(scores, np.ndarray):
                if scores.dtype == np.dtype('O'):  # Object array, need to extract the right field
                    return scores.item().get('spearman', scores)
                else:
                    return scores
            else:
                return scores
    
    # Also check if scores are within a .npz file
    npz_patterns = [
        f"{model_name}_predictions.npz",
        f"{model_name}_correlations.npz"
    ]
    
    for pattern in npz_patterns:
        file_path = results_dir / pattern
        if file_path.exists():
            data = np.load(file_path)
            if 'spearman' in data:
                return data['spearman']
            if 'spearman_rhos' in data:
                return data['spearman_rhos']
    
    raise FileNotFoundError(f"Could not find Spearman scores for {model_name} in {results_dir}")

def find_available_models(results_dir):
    """Find all available model results in the predictions directory"""
    models = set()
    
    # Check for spearman.npy files
    pattern = str(results_dir / "*_spearman.npy")
    files = glob.glob(pattern)
    for f in files:
        models.add(Path(f).stem.replace("_spearman", ""))
    
    # Check for predictions.npz files
    pattern = str(results_dir / "*_predictions.npz")
    files = glob.glob(pattern)
    for f in files:
        models.add(Path(f).stem.replace("_predictions", ""))
    
    # Check for correlations.npy files
    pattern = str(results_dir / "*_correlations.npy")
    files = glob.glob(pattern)
    for f in files:
        models.add(Path(f).stem.replace("_correlations", ""))
    
    # Check for pll files
    pattern = str(results_dir / "pll_results_*.npy")
    files = glob.glob(pattern)
    for f in files:
        models.add(Path(f).stem.replace("pll_results_", ""))
    
    return list(models)

def plot_comparison(control_name, treatment_names, statistics, p_values, output_dir):
    """Generate comparison plots for Dunnett's test results"""
    # Create a dataframe for plotting
    result_df = pd.DataFrame({
        'Model': treatment_names,
        'Test Statistic': statistics,
        'p-value': p_values,
        'Significant': p_values < 0.05
    })
    
    # Plot test statistics
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y='Test Statistic', hue='Significant', data=result_df)
    plt.title(f'Dunnett Test Statistics (vs {control_name})')
    plt.axhline(y=0, color='r', linestyle='-')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_dir / 'dunnett_statistics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot p-values
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='Model', y='p-value', data=result_df)
    plt.title(f'P-values (vs {control_name})')
    plt.axhline(y=0.05, color='r', linestyle='-', label='α=0.05')
    
    # Add significance indicators
    for i, p in enumerate(p_values):
        significance = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        plt.text(i, p + 0.01, significance, ha='center')
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'dunnett_pvalues.png', dpi=300, bbox_inches='tight')
    plt.close()

def run_dunnet(args):
    # Ensure results directory exists
    args.results_dir.mkdir(parents=True, exist_ok=True)
    
    # Set default output directory if not provided
    if args.output_dir is None:
        args.output_dir = args.results_dir
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a CSV directory for custom plotting data
    csv_dir = args.output_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    
    # Find available models if treatment models not specified
    if args.treatment_models is None:
        available_models = find_available_models(args.results_dir)
        args.treatment_models = [m for m in available_models if m != args.control_model]
        if not args.treatment_models:
            print(f"Error: No treatment models found in {args.results_dir}")
            return
    
    print(f"Control model: {args.control_model}")
    print(f"Treatment models: {args.treatment_models}")
    
    # Load control model scores
    try:
        control_scores = load_spearman_scores(args.results_dir, args.control_model)
        print(f"Control scores: {control_scores}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Prepare data array for Dunnett's test
    all_scores = [control_scores]
    model_names = [args.control_model]
    
    # Load treatment model scores
    for model in args.treatment_models:
        try:
            scores = load_spearman_scores(args.results_dir, model)
            all_scores.append(scores)
            model_names.append(model)
            print(f"{model} scores: {scores}")
        except FileNotFoundError as e:
            print(f"Warning: {e}")
    
    # Stack scores into shape (n_splits, n_groups)
    # group 0 = control, groups 1+ = treatments
    # Ensure all arrays have the same length by padding with NaN
    max_len = max(len(s) for s in all_scores)
    padded_scores = []
    for scores in all_scores:
        if len(scores) < max_len:
            padded = np.pad(scores, (0, max_len - len(scores)), 
                           'constant', constant_values=np.nan)
            padded_scores.append(padded)
        else:
            padded_scores.append(scores)
    
    data = np.column_stack(padded_scores)
    
    # Run Dunnett's test, control index = 0
    result = dunnett(data, control=0)
    
    # Save raw scores to CSV
    scores_df = pd.DataFrame({name: scores for name, scores in zip(model_names, padded_scores)})
    scores_df.to_csv(csv_dir / "spearman_scores_by_model.csv", index=False)
    print(f"Saved raw Spearman scores to {csv_dir / 'spearman_scores_by_model.csv'}")
    
    # Create results dataframe
    results_list = []
    for i, comp in enumerate(result.comparisons):
        treatment_idx = comp[0]
        results_list.append({
            'control': args.control_model,
            'treatment': model_names[treatment_idx],
            'statistic': result.statistic[i],
            'p_value': result.pvalue[i],
            'significant': result.pvalue[i] < args.alpha,
            'mean_control': np.nanmean(control_scores),
            'mean_treatment': np.nanmean(padded_scores[treatment_idx]),
            'difference': np.nanmean(padded_scores[treatment_idx]) - np.nanmean(control_scores)
        })
    
    results_df = pd.DataFrame(results_list)
    results_df.to_csv(csv_dir / "dunnett_test_results.csv", index=False)
    print(f"Saved Dunnett's test results to {csv_dir / 'dunnett_test_results.csv'}")
    
    # Print results
    print("\nDunnett's Test Results:")
    print("=====================")
    for i, comp in enumerate(result.comparisons):
        treatment_idx = comp[0]
        treatment_name = model_names[treatment_idx]
        print(f"{treatment_name} vs {args.control_model}:")
        print(f"  Test statistic: {result.statistic[i]:.4f}")
        print(f"  p-value: {result.pvalue[i]:.4f}")
        significance = "significant" if result.pvalue[i] < args.alpha else "not significant"
        print(f"  Result: {significance} (α={args.alpha})")
    
    # Save results to text file
    with open(args.output_dir / "dunnett_results.txt", 'w') as f:
        f.write("Dunnett's Test Results\n")
        f.write("=====================\n\n")
        f.write(f"Control model: {args.control_model}\n")
        f.write(f"Control scores: {control_scores}\n\n")
        
        for i, comp in enumerate(result.comparisons):
            treatment_idx = comp[0]
            treatment_name = model_names[treatment_idx]
            f.write(f"{treatment_name} vs {args.control_model}:\n")
            f.write(f"  Treatment scores: {padded_scores[treatment_idx]}\n")
            f.write(f"  Test statistic: {result.statistic[i]:.4f}\n")
            f.write(f"  p-value: {result.pvalue[i]:.4f}\n")
            significance = "significant" if result.pvalue[i] < args.alpha else "not significant"
            f.write(f"  Result: {significance} (α={args.alpha})\n\n")
    
    print(f"\nResults saved to {args.output_dir / 'dunnett_results.txt'}")
    
    # Generate comparison plots if requested
    if args.plot:
        plot_comparison(
            args.control_model, 
            args.treatment_models, 
            result.statistic, 
            result.pvalue, 
            args.output_dir
        )
    
    return {
        "comparisons": result.comparisons,
        "statistic": result.statistic,
        "pvalue": result.pvalue,
        "model_names": model_names,
        "results_df": results_df
    }

def main():
    args = parse_args()
    run_dunnet(args)

if __name__ == "__main__":
    main()

#Notes:
#data must be an array of shape (n_observations, n_groups).
#control=0 tells SciPy that column 0 (your pre-trained group) is the reference.
#result typically exposes .comparisons, .statistic, and .pvalue, which you can print or log.