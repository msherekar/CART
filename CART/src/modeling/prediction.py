#!/usr/bin/env python3
"""
Code to predict cytotoxicity of CAR-T cells.

Inputs embedding files, generates dummy cytotoxicity scores or loads real ones if available,
and runs nested 5-fold CV (outer) with 3-fold RidgeCV (inner)
to compute Spearman's ρ for each embedding set.

Outputs prediction results to a directory for further analysis.
"""
import argparse
import numpy as np
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from scipy.stats import spearmanr

def get_project_root():
    """Get project root directory relative to this file"""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def parse_args():
    project_root = get_project_root()
    predictions_dir = os.path.join(project_root, "CART", "predictions")
    embeddings_dir = os.path.join(project_root, "CART", "embeddings")
    
    parser = argparse.ArgumentParser(description="Run prediction model on embeddings and evaluate performance")
    parser.add_argument(
        "--embedding_dir", 
        type=str,
        default=embeddings_dir,
        help="Directory containing embedding .npy files"
    )
    parser.add_argument(
        "--embedding_files", 
        nargs="+", 
        type=str,
        help="Specific embedding files to use (optional, uses all .npy files in directory if not provided)"
    )
    parser.add_argument(
        "--labels_path", 
        type=str, 
        help="Path to CSV file containing cytotoxicity labels"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
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

def load_labels_and_match_ids(labels_path, embedding_path):
    """Load labels from CSV and match with embedding IDs"""
    # Load the CSV file
    df = pd.read_csv(labels_path)
    print(f"Loaded {len(df)} labels from {labels_path}")
    
    # Get the corresponding IDs file
    ids_file = embedding_path.replace('.npy', '.ids.txt')
    if not os.path.exists(ids_file):
        raise FileNotFoundError(f"IDs file not found: {ids_file}")
    
    # Load sequence IDs
    with open(ids_file, 'r') as f:
        sequence_ids = [line.strip() for line in f]
    print(f"Loaded {len(sequence_ids)} sequence IDs from {ids_file}")
    
    # Match labels with sequence IDs
    matched_labels = []
    matched_ids = []
    unmatched_ids = []
    
    for seq_id in sequence_ids:
        # Try to find matching label
        match = df[df['sequence_id'] == seq_id]  # Adjust column name if different
        if not match.empty:
            matched_labels.append(match['cytotoxicity'].values[0])  # Adjust column name if different
            matched_ids.append(seq_id)
        else:
            unmatched_ids.append(seq_id)
    
    if unmatched_ids:
        print(f"Warning: {len(unmatched_ids)} sequences without matching labels:")
        for seq_id in unmatched_ids[:5]:  # Show first 5 unmatched IDs
            print(f"  - {seq_id}")
        if len(unmatched_ids) > 5:
            print(f"  ... and {len(unmatched_ids) - 5} more")
    
    print(f"Matched {len(matched_labels)} sequences with labels")
    return np.array(matched_labels), matched_ids

def get_embedding_files(embedding_dir, specific_files=None):
    """Get list of embedding files to process"""
    if specific_files:
        # If specific files are provided, use those
        embedding_files = []
        for file in specific_files:
            if not os.path.isabs(file):
                file = os.path.join(embedding_dir, file)
            if not file.endswith('.npy'):
                file += '.npy'
            if os.path.exists(file):
                embedding_files.append(file)
            else:
                print(f"Warning: Embedding file not found: {file}")
    else:
        # Otherwise, use all .npy files in the directory
        embedding_files = glob.glob(os.path.join(embedding_dir, "*.npy"))
    
    if not embedding_files:
        raise ValueError(f"No embedding files found in {embedding_dir}")
    
    print(f"Found {len(embedding_files)} embedding files:")
    for file in embedding_files:
        print(f"  - {os.path.basename(file)}")
    
    return embedding_files

def plot_correlation(y_test, y_pred, fold, model_name, output_dir):
    """Plot correlation between actual and predicted values"""
    plt.figure(figsize=(10, 8))
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Scatter plot
    ax1.scatter(y_pred, y_test, alpha=0.5)
    
    # Add regression line
    z = np.polyfit(y_pred, y_test, 1)
    p = np.poly1d(z)
    ax1.plot(y_pred, p(y_pred), "r--", alpha=0.8)
    
    ax1.set_title(f'Fold {fold} - {model_name}\nActual vs Predicted Values')
    ax1.set_xlabel('Predicted Values')
    ax1.set_ylabel('Actual Values')
    
    # Add correlation coefficients
    rho, pval = spearmanr(y_test, y_pred)
    pearson_r = np.corrcoef(y_test, y_pred)[0, 1]
    
    stats_text = (f'Spearman ρ = {rho:.3f} (p={pval:.3g})\n'
                 f'Pearson r = {pearson_r:.3f}')
    ax1.text(0.05, 0.95, stats_text, 
             transform=ax1.transAxes, 
             bbox=dict(facecolor='white', alpha=0.8))
    
    # Add distribution plots
    sns.histplot(y_test, color='blue', alpha=0.5, label='Actual', ax=ax2)
    sns.histplot(y_pred, color='red', alpha=0.5, label='Predicted', ax=ax2)
    ax2.set_title('Distribution of Actual vs Predicted Values')
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Count')
    ax2.legend()
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, f"{model_name}_fold{fold}_correlation.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return rho, pval, pearson_r

def plot_spearman_whisker(spearman_scores, model_name, output_dir):
    """Plot whisker plot of Spearman correlation scores across folds"""
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=spearman_scores)
    plt.title(f'Spearman Correlation Scores - {model_name}')
    plt.ylabel('Spearman ρ')
    plt.xlabel('Folds')
    
    # Add individual points
    x = np.random.normal(0, 0.04, size=len(spearman_scores))
    plt.scatter(x, spearman_scores, alpha=0.6, color='red')
    
    # Add mean line
    mean_rho = np.mean(spearman_scores)
    plt.axhline(y=mean_rho, color='r', linestyle='--', alpha=0.5)
    plt.text(0.5, mean_rho, f'Mean: {mean_rho:.3f}', 
             ha='left', va='bottom', color='r')
    
    # Save plot
    plot_path = os.path.join(output_dir, f"{model_name}_spearman_whisker.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

def run_prediction(args):
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get list of embedding files to process
    embedding_files = get_embedding_files(args.embedding_dir, args.embedding_files)
    
    # Hyperparameter grid for RidgeCV: 10^-6 … 10^6 (log-spaced)
    alphas = np.logspace(-6, 6, num=13)
    
    # Outer k-fold splitter
    outer_cv = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_seed)
    
    # Fix random seed for dummy labels
    rng = np.random.RandomState(args.random_seed)
    
    all_results = {}
    
    # Process each embedding file
    for emb_path in embedding_files:
        print(f"\n=== Evaluating embeddings: {os.path.basename(emb_path)} ===")
        X = np.load(emb_path)  # shape (n_samples, H)
        
        # Load and match labels if provided
        if args.labels_path:
            try:
                y, matched_ids = load_labels_and_match_ids(args.labels_path, emb_path)
                print(f"Using {len(y)} matched labels")
            except Exception as e:
                print(f"Error loading labels: {e}")
                print("Using dummy labels instead")
                y = rng.rand(X.shape[0]) * 100
        else:
            y = rng.rand(X.shape[0]) * 100
            print(f"Using {len(y)} dummy labels (no real labels provided)")
        
        # Store predictions and actual values for each fold
        all_predictions = []
        all_actuals = []
        spearman_scores = []
        pearson_scores = []
        
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
            
            # Compute correlations and plot
            model_name = os.path.splitext(os.path.basename(emb_path))[0]
            rho, pval, pearson_r = plot_correlation(y_test, y_pred, fold, model_name, args.output_dir)
            spearman_scores.append(rho)
            pearson_scores.append(pearson_r)
            
            print(
                f" Fold {fold}: best α={ridge_cv.alpha_:.1e}\n"
                f"  Spearman's ρ={rho:.3f} (p={pval:.3g})\n"
                f"  Pearson's r={pearson_r:.3f}"
            )
        
        # Plot whisker plot for all folds
        plot_spearman_whisker(spearman_scores, model_name, args.output_dir)
        
        mean_rho = np.mean(spearman_scores)
        std_rho = np.std(spearman_scores)
        mean_pearson = np.mean(pearson_scores)
        std_pearson = np.std(pearson_scores)
        
        print(
            f" --> {args.n_splits}-fold outer correlations:\n"
            f"  Spearman's ρ: mean={mean_rho:.3f} ± {std_rho:.3f}\n"
            f"  Pearson's r: mean={mean_pearson:.3f} ± {std_pearson:.3f}"
        )
        
        # Save results for this embedding
        results_file = os.path.join(args.output_dir, f"{model_name}_correlations.npy")
        np.save(results_file, {
            'spearman': np.array(spearman_scores),
            'pearson': np.array(pearson_scores)
        })
        print(f"Saved correlation scores to {results_file}")
        
        # Save predictions and actual values
        predictions_file = os.path.join(args.output_dir, f"{model_name}_predictions.npz")
        np.savez(
            predictions_file, 
            predictions=np.array(all_predictions), 
            actuals=np.array(all_actuals),
            spearman=np.array(spearman_scores),
            pearson=np.array(pearson_scores),
            mean_spearman=mean_rho,
            std_spearman=std_rho,
            mean_pearson=mean_pearson,
            std_pearson=std_pearson
        )
        print(f"Saved predictions to {predictions_file}")
        
        # Store results for summary
        all_results[model_name] = {
            "spearman_scores": spearman_scores,
            "pearson_scores": pearson_scores,
            "mean_rho": mean_rho,
            "std_rho": std_rho,
            "mean_pearson": mean_pearson,
            "std_pearson": std_pearson
        }
    
    # Save summary of all results
    summary_file = os.path.join(args.output_dir, "prediction_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Prediction Results Summary\n")
        f.write(f"=========================\n")
        for model_name, results in all_results.items():
            f.write(f"\n{model_name}:\n")
            f.write(f"  Individual fold Spearman ρ: {[f'{x:.3f}' for x in results['spearman_scores']]}\n")
            f.write(f"  Individual fold Pearson r: {[f'{x:.3f}' for x in results['pearson_scores']]}\n")
            f.write(f"  Mean Spearman ρ: {results['mean_rho']:.3f} ± {results['std_rho']:.3f}\n")
            f.write(f"  Mean Pearson r: {results['mean_pearson']:.3f} ± {results['std_pearson']:.3f}\n")
    
    print(f"\nSummary of all results saved to {summary_file}")
    return all_results

def main():
    args = parse_args()
    run_prediction(args)

if __name__ == "__main__":
    main()
