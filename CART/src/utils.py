import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
import CART
from pathlib import Path


def get_project_root() -> Path:
    """Get project root directory relative to this file"""
    # Get the directory containing this file
    current_file = Path(__file__).resolve()
    # Go up two levels to reach the project root (src -> CART -> project root)
    return current_file.parent.parent

def get_relative_path(*path_components) -> Path:
    """Get path relative to project root"""
    return get_project_root().joinpath(*path_components)

DEFAULT_OUTPUT_DIR = get_project_root() / "output"


def load_labels_and_match_ids(labels_path: Path, embedding_path: Path):
    """Load labels from CSV and match with embedding IDs"""
    df = pd.read_csv(labels_path)
    print(f"Loaded {len(df)} labels from {labels_path}")
    
    # Get the corresponding IDs file
    ids_file = embedding_path.with_suffix('.ids.txt')
    if not ids_file.exists():
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
        match = df[df['sequence_id'] == seq_id]
        if not match.empty:
            matched_labels.append(match['cytotoxicity'].values[0])
            matched_ids.append(seq_id)
        else:
            unmatched_ids.append(seq_id)
    
    if unmatched_ids:
        print(f"Warning: {len(unmatched_ids)} sequences without matching labels")
    
    print(f"Matched {len(matched_labels)} sequences with labels")
    return np.array(matched_labels), matched_ids

def recall_precision_at_k(y_true, y_pred, K):
    """Compute Recall@K and Precision@K metrics"""
    n = len(y_true)
    cutoff = np.percentile(y_true, 75)
    positives = (y_true >= cutoff)
    n_pos = positives.sum()

    topk_idx = np.argsort(y_pred)[-K:]
    tp = positives[topk_idx].sum()

    recall = tp / n_pos if n_pos > 0 else 0.0
    precision = tp / K
    return recall, precision

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

def plot_recall_precision(scores, model_name, output_dir):
    """Plot Recall@K and Precision@K metrics for different K values"""
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

# Use package-relative paths for default values
