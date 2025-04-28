#!/usr/bin/env python3
"""
Dunnett's test for multiple comparison procedure.

This script compares Spearman correlation scores between a control model
(typically a pretrained model) and one or more treatment models (typically finetuned models).
It loads results from the predictions directory and performs statistical significance testing.
"""

import argparse
import numpy as np
from pathlib import Path
from scipy.stats import dunnett
import glob

def get_project_root() -> Path:
    """Get project root directory relative to this file"""
    return Path(__file__).resolve().parents[3]  # Go up 3 levels from this file

def parse_args():
    project_root = get_project_root()
    predictions_dir = project_root / "CART/predictions"
    
    parser = argparse.ArgumentParser(description="Run Dunnett's test for multiple comparison procedure")
    parser.add_argument(
        "--predictions_dir", 
        type=Path, 
        default=predictions_dir,
        help="Directory containing model prediction results"
    )
    parser.add_argument(
        "--control_model", 
        type=str, 
        default="pll_results_pretrained",
        help="Name of the control model (without _spearman.npy suffix)"
    )
    parser.add_argument(
        "--treatment_models", 
        type=str, 
        nargs="+",
        help="Names of treatment models to compare against control (without _spearman.npy suffix)"
    )
    parser.add_argument(
        "--output_file", 
        type=Path, 
        default=None,
        help="Output file for test results (defaults to predictions_dir/dunnett_results.txt)"
    )
    return parser.parse_args()

def load_spearman_scores(predictions_dir, model_name):
    """Load Spearman correlation scores for a model"""
    file_path = predictions_dir / f"{model_name}_spearman.npy"
    if not file_path.exists():
        raise FileNotFoundError(f"Could not find results file: {file_path}")
    return np.load(file_path)

def find_available_models(predictions_dir):
    """Find all available model results in the predictions directory"""
    pattern = str(predictions_dir / "*_spearman.npy")
    files = glob.glob(pattern)
    models = [Path(f).stem.replace("_spearman", "") for f in files]
    return models

def run_dunnet(args):
    # Ensure output directory exists
    args.predictions_dir.mkdir(parents=True, exist_ok=True)
    
    # Set default output file if not provided
    if args.output_file is None:
        args.output_file = args.predictions_dir / "dunnett_results.txt"
    
    # Find available models if treatment models not specified
    if args.treatment_models is None:
        available_models = find_available_models(args.predictions_dir)
        args.treatment_models = [m for m in available_models if m != args.control_model]
        if not args.treatment_models:
            print(f"Error: No treatment models found in {args.predictions_dir}")
            return
    
    print(f"Control model: {args.control_model}")
    print(f"Treatment models: {args.treatment_models}")
    
    # Load control model scores
    try:
        control_scores = load_spearman_scores(args.predictions_dir, args.control_model)
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
            scores = load_spearman_scores(args.predictions_dir, model)
            all_scores.append(scores)
            model_names.append(model)
            print(f"{model} scores: {scores}")
        except FileNotFoundError as e:
            print(f"Warning: {e}")
    
    # Stack scores into shape (n_splits, n_groups)
    # group 0 = control, groups 1+ = treatments
    data = np.column_stack(all_scores)
    
    # Run Dunnett's test, control index = 0
    result = dunnett(data, control=0)
    
    # Print results
    print("\nDunnett's Test Results:")
    print("=====================")
    for i, comp in enumerate(result.comparisons):
        treatment_idx = comp[0]
        treatment_name = model_names[treatment_idx]
        print(f"{treatment_name} vs {args.control_model}:")
        print(f"  Test statistic: {result.statistic[i]:.4f}")
        print(f"  p-value: {result.pvalue[i]:.4f}")
        significance = "significant" if result.pvalue[i] < 0.05 else "not significant"
        print(f"  Result: {significance} (α=0.05)")
    
    # Save results to file
    with open(args.output_file, 'w') as f:
        f.write("Dunnett's Test Results\n")
        f.write("=====================\n\n")
        f.write(f"Control model: {args.control_model}\n")
        f.write(f"Control scores: {control_scores}\n\n")
        
        for i, comp in enumerate(result.comparisons):
            treatment_idx = comp[0]
            treatment_name = model_names[treatment_idx]
            f.write(f"{treatment_name} vs {args.control_model}:\n")
            f.write(f"  Treatment scores: {all_scores[treatment_idx]}\n")
            f.write(f"  Test statistic: {result.statistic[i]:.4f}\n")
            f.write(f"  p-value: {result.pvalue[i]:.4f}\n")
            significance = "significant" if result.pvalue[i] < 0.05 else "not significant"
            f.write(f"  Result: {significance} (α=0.05)\n\n")
    
    print(f"\nResults saved to {args.output_file}")
    
    return {
        "comparisons": result.comparisons,
        "statistic": result.statistic,
        "pvalue": result.pvalue,
        "model_names": model_names
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