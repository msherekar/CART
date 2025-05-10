#!/usr/bin/env python3
"""
Evaluates embeddings with nested cross-validation to compute:
- Spearman's correlation
- Precision and recall at different k values
- Generates comparison plots for multiple models
"""

import numpy as np
import argparse
import pandas as pd
from pathlib import Path
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from scipy.stats import spearmanr
from scipy.stats import dunnett
#from ..score import recall_precision_at_k

# Import visualization module for plotting
try:
    from .visualization import (
        plot_correlation_comparison, 
        plot_confusion_matrix, 
        plot_precision_recall_at_k
    )
except ImportError:
    from visualization import (
        plot_correlation_comparison, 
        plot_confusion_matrix, 
        plot_precision_recall_at_k
    )

def get_project_root() -> Path:
    """Get project root directory relative to this file"""
    return Path(__file__).resolve().parents[3]  # Go up 3 levels from this file

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

def nested_ridge_cv(X: np.ndarray, y: np.ndarray, alphas, outer_cv, inner_cv=3):
    """
    Runs nested CV:
      outer_cv splits for train/test,
      RidgeCV(cv=inner_cv) on the train split.
    Returns:
      all_y_tests, all_y_preds, spearman_scores, best_alphas
    """
    all_y_tests = []
    all_y_preds = []
    spearman_scores = []
    best_alphas = []

    for train_idx, test_idx in outer_cv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Reshape if needed
        if len(X_train.shape) == 1:
            X_train = X_train.reshape(-1, 1)
            X_test = X_test.reshape(-1, 1)

        # inner CV to pick α
        ridge = RidgeCV(alphas=alphas, cv=inner_cv)
        ridge.fit(X_train, y_train)

        # predict
        y_pred = ridge.predict(X_test)

        # record
        all_y_tests.append(y_test)
        all_y_preds.append(y_pred)
        best_alphas.append(ridge.alpha_)

        rho, _ = spearmanr(y_test, y_pred)
        spearman_scores.append(rho)

    return all_y_tests, all_y_preds, spearman_scores, best_alphas

def parse_args(args_list=None):
    project_root = get_project_root()
    plots_dir = project_root / "CART" / "output" / "plots"
    
    parser = argparse.ArgumentParser(description="Evaluate embeddings and generate comparison plots")
    parser.add_argument(
        "--embed_paths", 
        nargs="+", 
        required=True, 
        help="Paths to embedding .npy files"
    )
    parser.add_argument(
        "--output_dir", 
        type=Path, 
        default=plots_dir,
        help="Directory to save plots"
    )
    parser.add_argument(
        "--use_real_labels", 
        action="store_true", 
        help="Use real labels instead of dummy data"
    )
    parser.add_argument(
        "--labels_path", 
        type=Path, 
        help="Path to real labels file (.npy or .csv)"
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
        help="Number of splits for outer cross-validation"
    )
    parser.add_argument(
        "--inner_cv", 
        type=int, 
        default=3, 
        help="Number of splits for inner cross-validation"
    )
    
    # Only parse command line arguments if this module is run directly
    if args_list is None and __name__ == "__main__":
        return parser.parse_args()
    else:
        # When imported, use the provided args_list or an empty list
        return parser.parse_args(args_list or [])

def run_evaluation(args):
    """
    Evaluate multiple embedding models and generate comparison plots
    
    Args:
        args: Parsed command line arguments
    """
    # Create output directory if specified
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a subdirectory for CSV files
    csv_dir = args.output_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert strings to Path objects
    embedding_paths = []
    for path in args.embed_paths:
        p = Path(path)
        if not p.is_absolute():
            p = get_project_root() / p
        embedding_paths.append(p)
    
    # Grid for RidgeCV α
    alphas = np.logspace(-6, 6, num=13)
    
    # outer CV splitter
    outer_cv = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_seed)
    
    # For reproducible dummy labels
    rng = np.random.RandomState(args.random_seed)
    
    results = {}
    model_names = []
    
    for path in embedding_paths:
        print(f"\n--- Processing {path} ---")
        model_name = path.stem
        model_names.append(model_name)
        
        X = np.load(path)
        n = X.shape[0]
        
        # Generate or load labels
        if args.use_real_labels and args.labels_path:
            labels_path = args.labels_path
            if not labels_path.is_absolute():
                labels_path = get_project_root() / labels_path
                
            if str(labels_path).endswith('.npy'):
                y = np.load(labels_path)
            elif str(labels_path).endswith('.csv'):
                df = pd.read_csv(labels_path)
                # Only use the cytotoxicity column, not all columns
                if 'cytotoxicity' in df.columns:
                    y = df['cytotoxicity'].values
                else:
                    # If cytotoxicity column not found, use the second column 
                    # (assuming format is ID, value)
                    y = df.iloc[:, 1].values
            else:
                raise ValueError(f"Unsupported label file format: {labels_path}")
            
            if len(y) != n:
                raise ValueError(f"Label count ({len(y)}) doesn't match embedding count ({n})")
        else:
            # dummy cytotoxicity in [0,100)
            y = rng.rand(n) * 100
        
        (all_y_tests,
         all_y_preds,
         spearman_scores,
         best_alphas) = nested_ridge_cv(X, y, alphas, outer_cv, inner_cv=args.inner_cv)
        
        # store
        results[path] = {
            "model_name": model_name,
            "y_tests": all_y_tests,
            "y_preds": all_y_preds,
            "spearman_rhos": spearman_scores,
            "best_alphas": best_alphas,
        }
        
        # quick printout
        for i, rho in enumerate(spearman_scores, 1):
            print(f" Fold {i}: α={best_alphas[i-1]:.1e}, ρ={rho:.3f}")
        mean_rho, std_rho = np.mean(spearman_scores), np.std(spearman_scores)
        print(f" → mean ρ = {mean_rho:.3f} ± {std_rho:.3f}")
    
    # Compute precision/recall metrics
    for path, res in results.items():
        model_name = res["model_name"]
        print(f"\nMetrics for {model_name}:")
        r5, p5, r10, p10 = [], [], [], []
        
        for y_t, y_p in zip(res["y_tests"], res["y_preds"]):
            rec5, prec5 = recall_precision_at_k(y_t, y_p, K=5)
            rec10, prec10 = recall_precision_at_k(y_t, y_p, K=10)
            r5.append(rec5); p5.append(prec5)
            r10.append(rec10); p10.append(prec10)
            
        res["recall_at_5"] = r5
        res["precision_at_5"] = p5
        res["recall_at_10"] = r10
        res["precision_at_10"] = p10
        
        print(f" Recall@5:  {np.mean(r5):.3f} ± {np.std(r5):.3f}")
        print(f" Precision@5: {np.mean(p5):.3f} ± {np.std(p5):.3f}")
        print(f" Recall@10: {np.mean(r10):.3f} ± {np.std(r10):.3f}")
        print(f" Precision@10: {np.mean(p10):.3f} ± {np.std(p10):.3f}")
    
    # Save results to npy files for use with dunnet.py
    for path, res in results.items():
        model_name = res["model_name"]
        results_dir = args.output_dir / "results"
        results_dir.mkdir(exist_ok=True, parents=True)
        
        # Save metrics as CSV
        metrics_df = pd.DataFrame({
            'fold': range(1, len(res["spearman_rhos"]) + 1),
            'spearman': res["spearman_rhos"],
            'alpha': res["best_alphas"],
            'recall_at_5': res["recall_at_5"],
            'precision_at_5': res["precision_at_5"],
            'recall_at_10': res["recall_at_10"],
            'precision_at_10': res["precision_at_10"]
        })
        metrics_csv = csv_dir / f"{model_name}_metrics.csv"
        metrics_df.to_csv(metrics_csv, index=False)
        print(f"Saved metrics to {metrics_csv}")
        
        # Save Spearman scores
        spearman_file = results_dir / f"{model_name}_spearman.npy"
        np.save(spearman_file, np.array(res["spearman_rhos"]))
        print(f"Saved Spearman scores to {spearman_file}")
        
        # Find maximum length among all folds
        max_len = max(len(y_test) for y_test in res["y_tests"])
        
        # Pad arrays to make them the same length
        padded_y_tests = np.array([np.pad(y_test, (0, max_len - len(y_test)), mode='constant', constant_values=np.nan) 
                                 for y_test in res["y_tests"]])
        padded_y_preds = np.array([np.pad(y_pred, (0, max_len - len(y_pred)), mode='constant', constant_values=np.nan) 
                                 for y_pred in res["y_preds"]])
        
        # Save predictions and labels
        pred_file = results_dir / f"{model_name}_predictions.npz"
        np.savez(
            pred_file,
            y_tests=padded_y_tests,
            y_preds=padded_y_preds,
            spearman=np.array(res["spearman_rhos"]),
            best_alphas=np.array(res["best_alphas"]),
            recall_at_5=np.array(res["recall_at_5"]),
            precision_at_5=np.array(res["precision_at_5"]),
            recall_at_10=np.array(res["recall_at_10"]),
            precision_at_10=np.array(res["precision_at_10"])
        )
        print(f"Saved predictions to {pred_file}")
    
    # Generate comparison plots
    if len(results) > 1:
        # 1. Plot Spearman correlation comparison
        spearman_values = [results[path]["spearman_rhos"] for path in embedding_paths]
        model_names = [results[path]["model_name"] for path in embedding_paths]
        
        # Save comparison data as CSV
        comparison_data = {}
        for i, model in enumerate(model_names):
            comparison_data[model] = spearman_values[i]
        
        # Create DataFrame with variable number of rows (use the max length)
        max_folds = max(len(vals) for vals in spearman_values)
        comparison_df = pd.DataFrame({model: pd.Series(vals) for model, vals in zip(model_names, spearman_values)})
        comparison_df.index = [f"fold_{i+1}" for i in range(max_folds)]
        comparison_csv = csv_dir / "spearman_comparison.csv"
        comparison_df.to_csv(comparison_csv)
        print(f"Saved Spearman comparison to {comparison_csv}")
        
        corr_plot_path = None
        if args.output_dir:
            corr_plot_path = args.output_dir / "spearman_correlation_comparison.png"
        
        plot_correlation_comparison(
            model_names=model_names,
            spearman_values=spearman_values,
            output_path=corr_plot_path,
            title="Spearman Correlation Comparison"
        )
        
        # 2. Plot precision/recall@k comparison
        k_values = [1, 5, 10, 15, 20]
        precision_at_k = []
        recall_at_k = []
        
        for path in embedding_paths:
            # Compute precision/recall at each k value
            prec_values = []
            recall_values = []
            
            for k in k_values:
                prec = []
                rec = []
                for y_test, y_pred in zip(results[path]["y_tests"], results[path]["y_preds"]):
                    r, p = recall_precision_at_k(y_test, y_pred, K=k)
                    prec.append(p)
                    rec.append(r)
                prec_values.append(np.mean(prec))
                recall_values.append(np.mean(rec))
            
            precision_at_k.append(prec_values)
            recall_at_k.append(recall_values)
        
        # Save precision/recall at k data as CSV
        pr_data = {'k': k_values}
        for i, model in enumerate(model_names):
            pr_data[f"{model}_precision"] = precision_at_k[i]
            pr_data[f"{model}_recall"] = recall_at_k[i]
        
        pr_df = pd.DataFrame(pr_data)
        pr_csv = csv_dir / "precision_recall_at_k.csv"
        pr_df.to_csv(pr_csv, index=False)
        print(f"Saved precision/recall at k data to {pr_csv}")
        
        pr_plot_path = None
        if args.output_dir:
            pr_plot_path = args.output_dir / "precision_recall_at_k.png"
        
        plot_precision_recall_at_k(
            model_names=model_names,
            precision_values=precision_at_k,
            recall_values=recall_at_k,
            k_values=k_values,
            output_path=pr_plot_path
        )
        
        # 3. For each model, plot a sample confusion matrix from the first fold
        for i, path in enumerate(embedding_paths):
            model_name = results[path]["model_name"]
            
            # Find the fold with the highest Spearman correlation
            spearman_scores = results[path]["spearman_rhos"]
            best_fold_idx = np.argmax(spearman_scores)
            best_spearman = spearman_scores[best_fold_idx]
            
            # Use the best fold's predictions instead of the first fold
            y_test = results[path]["y_tests"][best_fold_idx]
            y_pred = results[path]["y_preds"][best_fold_idx]
            
            # Save confusion matrix data
            cm_data = pd.DataFrame({'y_true': y_test, 'y_pred': y_pred})
            cm_csv = csv_dir / f"confusion_matrix_{model_name}_best_fold.csv"
            cm_data.to_csv(cm_csv, index=False)
            print(f"Saved best fold (ρ={best_spearman:.3f}) confusion matrix data to {cm_csv}")
            
            cm_plot_path = None
            if args.output_dir:
                cm_plot_path = args.output_dir / f"confusion_matrix_{model_name}_best_fold.png"
            
            plot_confusion_matrix(
                y_true=y_test,
                y_pred=y_pred,
                output_path=cm_plot_path,
                title=f"Confusion Matrix: {model_name} (Best Fold, ρ={best_spearman:.3f})"
            )
    
    return results

def main():
    args = parse_args()
    run_evaluation(args)

if __name__ == "__main__":
    main()
