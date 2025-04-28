#!/usr/bin/env python3
"""
predict_cytotoxicity.py

Loads embedding files, generates dummy cytotoxicity scores or loads real ones if available,
and runs nested 5-fold CV (outer) with 3-fold RidgeCV (inner)
to compute Spearman's ρ for each embedding set.

Outputs prediction results to a directory for further analysis.
"""
import argparse
import numpy as np
from pathlib import Path
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from scipy.stats import spearmanr

def get_project_root() -> Path:
    """Get project root directory relative to this file"""
    return Path(__file__).resolve().parents[3]  # Go up 3 levels from this file

def parse_args():
    project_root = get_project_root()
    predictions_dir = project_root / "CART/predictions"
    
    parser = argparse.ArgumentParser(description="Run prediction model on embeddings and evaluate performance")
    parser.add_argument(
        "--embedding_paths", 
        nargs="+", 
        type=Path,
        help="Paths to embedding .npy files"
    )
    parser.add_argument(
        "--labels_path", 
        type=Path, 
        help="Path to real labels file (optional, uses dummy labels if not provided)"
    )
    parser.add_argument(
        "--output_dir", 
        type=Path, 
        default=predictions_dir,
        help="Directory to save prediction results"
    )
    parser.add_argument(
        "--random_seed", 
        type=int, 
        default=42, 
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--n_splits", 
        type=int, 
        default=5, 
        help="Number of splits for cross-validation"
    )
    parser.add_argument(
        "--n_inner_splits", 
        type=int, 
        default=3, 
        help="Number of splits for inner cross-validation"
    )
    return parser.parse_args()

def run_prediction(args):
    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Resolve embedding paths if they're relative
    embedding_paths = []
    for path in args.embedding_paths:
        if not path.is_absolute():
            path = get_project_root() / path
        embedding_paths.append(path)
    
    # Hyperparameter grid for RidgeCV: 10^-6 … 10^6 (log-spaced)
    alphas = np.logspace(-6, 6, num=13)
    
    # Outer k-fold splitter
    outer_cv = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_seed)
    
    # Fix random seed for dummy labels
    rng = np.random.RandomState(args.random_seed)
    
    all_results = {}
    
    # Process each embedding file
    for emb_path in embedding_paths:
        print(f"\n=== Evaluating embeddings: {emb_path} ===")
        X = np.load(emb_path)  # shape (n_samples, H)
        n = X.shape[0]
        
        # Generate dummy continuous cytotoxicity scores or load real ones
        if args.labels_path:
            try:
                y = np.load(args.labels_path)
                print(f"Loaded {len(y)} labels from {args.labels_path}")
                # Make sure dimensions match
                if len(y) != n:
                    print(f"Warning: Label count ({len(y)}) doesn't match sample count ({n})")
                    y = y[:n] if len(y) > n else np.pad(y, (0, n-len(y)), mode='constant')
            except:
                print(f"Failed to load labels from {args.labels_path}, using dummy labels")
                y = rng.rand(n) * 100
        else:
            # Generate dummy continuous cytotoxicity scores (e.g., in range [0, 100])
            y = rng.rand(n) * 100
            print(f"Using {n} dummy labels (no real labels provided)")
        
        # Store predictions and actual values for each fold
        all_predictions = []
        all_actuals = []
        spearman_scores = []
        
        for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X), start=1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Inner k-fold RidgeCV to pick best alpha
            ridge_cv = RidgeCV(alphas=alphas, cv=args.n_inner_splits)
            ridge_cv.fit(X_train, y_train)
            
            # Predict on held-out fold
            y_pred = ridge_cv.predict(X_test)
            
            # Store predictions and actual values
            all_predictions.extend(y_pred)
            all_actuals.extend(y_test)
            
            # Compute Spearman correlation
            rho, pval = spearmanr(y_test, y_pred)
            spearman_scores.append(rho)
            print(
                f" Fold {fold}: best α={ridge_cv.alpha_:.1e}, "
                f"Spearman's ρ={rho:.3f} (p={pval:.3g})"
            )
        
        mean_rho = np.mean(spearman_scores)
        std_rho = np.std(spearman_scores)
        print(
            f" --> {args.n_splits}-fold outer Spearman's ρ: "
            f"mean={mean_rho:.3f} ± {std_rho:.3f}"
        )
        
        # Save results for this embedding
        model_name = emb_path.stem
        results_file = args.output_dir / f"{model_name}_spearman.npy"
        np.save(results_file, np.array(spearman_scores))
        print(f"Saved Spearman scores to {results_file}")
        
        # Save predictions and actual values
        predictions_file = args.output_dir / f"{model_name}_predictions.npz"
        np.savez(
            predictions_file, 
            predictions=np.array(all_predictions), 
            actuals=np.array(all_actuals),
            spearman=np.array(spearman_scores),
            mean_spearman=mean_rho,
            std_spearman=std_rho
        )
        print(f"Saved predictions to {predictions_file}")
        
        # Store results for summary
        all_results[model_name] = {
            "spearman_scores": spearman_scores,
            "mean_rho": mean_rho,
            "std_rho": std_rho
        }
    
    # Save summary of all results
    summary_file = args.output_dir / "prediction_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Prediction Results Summary\n")
        f.write(f"=========================\n")
        for model_name, results in all_results.items():
            f.write(f"\n{model_name}:\n")
            f.write(f"  Individual fold Spearman ρ: {[f'{x:.3f}' for x in results['spearman_scores']]}\n")
            f.write(f"  Mean Spearman ρ: {results['mean_rho']:.3f} ± {results['std_rho']:.3f}\n")
    
    print(f"\nSummary of all results saved to {summary_file}")
    return all_results

def main():
    args = parse_args()
    run_prediction(args)

if __name__ == "__main__":
    main()
