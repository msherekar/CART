#!/usr/bin/env python3
"""
visualization.py

Visualization utilities for CAR-T cell activity prediction system.
- Plot training/validation metrics over epochs
- Plot Spearman correlation and other evaluation metrics
- Compare performance of different models
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from scipy.stats import spearmanr
from scipy import stats
from statsmodels.stats.multitest import multipletests

# Set style for all plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

def plot_training_metrics(
    train_metrics: Dict[str, List[float]],
    val_metrics: Dict[str, List[float]],
    output_path: Optional[str] = None,
    model_name: str = "Model",
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Plot training and validation metrics over epochs.
    
    Args:
        train_metrics: Dictionary of training metrics (keys are metric names, values are lists of metric values per epoch)
        val_metrics: Dictionary of validation metrics (keys are metric names, values are lists of metric values per epoch)
        output_path: Path to save the figure (if None, figure is displayed)
        model_name: Name of the model for the plot title
        figsize: Figure size as (width, height)
    """
    num_metrics = len(train_metrics)
    epochs = range(1, len(list(train_metrics.values())[0]) + 1)
    
    fig, axes = plt.subplots(num_metrics, 1, figsize=figsize, sharex=True)
    if num_metrics == 1:
        axes = [axes]
    
    for i, (metric_name, train_values) in enumerate(train_metrics.items()):
        val_values = val_metrics[metric_name]
        ax = axes[i]
        
        ax.plot(epochs, train_values, 'b-', label=f'Training {metric_name}')
        ax.plot(epochs, val_values, 'r-', label=f'Validation {metric_name}')
        
        ax.set_title(f'{metric_name.capitalize()} over Epochs')
        ax.set_ylabel(metric_name)
        ax.legend()
        
        # Annotate best validation metric
        best_epoch = np.argmin(val_values) if 'loss' in metric_name.lower() else np.argmax(val_values)
        best_value = val_values[best_epoch]
        ax.scatter(best_epoch + 1, best_value, c='red', s=100, zorder=5)
        ax.annotate(f'Best: {best_value:.4f}', 
                   (best_epoch + 1, best_value),
                   xytext=(10, 0), textcoords='offset points')
    
    axes[-1].set_xlabel('Epochs')
    fig.suptitle(f'{model_name} Training Metrics', fontsize=16)
    plt.tight_layout()
    fig.subplots_adjust(top=0.9)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved training metrics plot to {output_path}")
    else:
        plt.show()
    plt.close()

def plot_correlation_comparison(
    model_names: List[str],
    spearman_values: List[List[float]],
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    title: str = "Model Performance Comparison"
) -> None:
    """
    Create a boxplot comparing Spearman correlations across different models.
    
    Args:
        model_names: List of model names
        spearman_values: List of lists containing Spearman correlation values for each model
        output_path: Path to save the figure (if None, figure is displayed)
        figsize: Figure size as (width, height)
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create a DataFrame for seaborn
    data = []
    for model_name, values in zip(model_names, spearman_values):
        for value in values:
            data.append({"Model": model_name, "Spearman Correlation": value})
    df = pd.DataFrame(data)
    
    # Create the boxplot
    sns.boxplot(x="Model", y="Spearman Correlation", data=df, palette="Set2", ax=ax)
    sns.stripplot(x="Model", y="Spearman Correlation", data=df, color='black', size=4, alpha=0.6, ax=ax)
    
    # Calculate and display mean values
    for i, model_name in enumerate(model_names):
        mean_val = np.mean(spearman_values[i])
        ax.text(i, mean_val, f"μ={mean_val:.3f}", ha='center', va='bottom', fontweight='bold')
    
    ax.set_title(title, fontsize=16)
    ax.set_ylim(-0.1, 1.0)  # Adjust as needed
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved correlation comparison plot to {output_path}")
    else:
        plt.show()
    plt.close()

def plot_metric_vs_epoch(
    metric_values: List[Dict[str, List[float]]],
    metric_name: str,
    model_names: List[str],
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    title: Optional[str] = None
) -> None:
    """
    Plot a specific metric across epochs for multiple models.
    
    Args:
        metric_values: List of dictionaries containing metrics for each model
        metric_name: Name of the metric to plot
        model_names: List of model names corresponding to metric_values
        output_path: Path to save the figure (if None, figure is displayed)
        figsize: Figure size as (width, height)
        title: Plot title (if None, a default title is used)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for i, (model_metrics, model_name) in enumerate(zip(metric_values, model_names)):
        if metric_name in model_metrics:
            epochs = range(1, len(model_metrics[metric_name]) + 1)
            ax.plot(epochs, model_metrics[metric_name], label=model_name, color=COLORS[i % len(COLORS)])
    
    ax.set_xlabel('Epochs')
    ax.set_ylabel(metric_name.capitalize())
    ax.set_title(title or f'{metric_name.capitalize()} vs. Epoch')
    ax.legend()
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved {metric_name} plot to {output_path}")
    else:
        plt.show()
    plt.close()

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Optional[str] = None,
    threshold: float = 0.75,
    figsize: Tuple[int, int] = (8, 8),
    title: str = "Binary Classification Confusion Matrix"
) -> None:
    """
    Plot confusion matrix for binary classification using percentile threshold.
    
    Args:
        y_true: True continuous values
        y_pred: Predicted continuous values
        output_path: Path to save the figure (if None, figure is displayed)
        threshold: Percentile threshold for binary classification (e.g., 0.75 for top 25%)
        figsize: Figure size as (width, height)
        title: Plot title
    """
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    
    # Convert continuous values to binary using percentile threshold
    cutoff_true = np.percentile(y_true, threshold * 100)
    cutoff_pred = np.percentile(y_pred, threshold * 100)
    
    y_true_binary = (y_true >= cutoff_true).astype(int)
    y_pred_binary = (y_pred >= cutoff_pred).astype(int)
    
    cm = confusion_matrix(y_true_binary, y_pred_binary)
    
    fig, ax = plt.subplots(figsize=figsize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Low Activity', 'High Activity'])
    disp.plot(cmap='Blues', ax=ax)
    
    ax.set_title(title)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix to {output_path}")
    else:
        plt.show()
    plt.close()

def plot_pll_distribution(
    pll_values: Dict[str, np.ndarray],
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
    bins: int = 30
) -> None:
    """
    Plot distribution of pseudo-log-likelihood scores for different models.
    
    Args:
        pll_values: Dictionary mapping model names to arrays of PLL values
        output_path: Path to save the figure (if None, figure is displayed)
        figsize: Figure size as (width, height)
        bins: Number of histogram bins
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for i, (model_name, values) in enumerate(pll_values.items()):
        sns.kdeplot(values, label=model_name, ax=ax, color=COLORS[i % len(COLORS)])
        
    ax.set_xlabel('Pseudo Log Likelihood')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Pseudo Log Likelihood Scores')
    ax.legend()
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved PLL distribution plot to {output_path}")
    else:
        plt.show()
    plt.close()

def plot_precision_recall_at_k(
    model_names: List[str],
    precision_values: List[List[float]],
    recall_values: List[List[float]],
    k_values: List[int],
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 7)
) -> None:
    """
    Plot precision@k and recall@k for different models.
    
    Args:
        model_names: List of model names
        precision_values: List of lists of precision values for each model at different k
        recall_values: List of lists of recall values for each model at different k
        k_values: List of k values corresponding to precision/recall measurements
        output_path: Path to save the figure (if None, figure is displayed)
        figsize: Figure size as (width, height)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    for i, model_name in enumerate(model_names):
        ax1.plot(k_values, precision_values[i], 'o-', label=model_name, color=COLORS[i % len(COLORS)])
        ax2.plot(k_values, recall_values[i], 'o-', label=model_name, color=COLORS[i % len(COLORS)])
    
    ax1.set_xlabel('k')
    ax1.set_ylabel('Precision@k')
    ax1.set_title('Precision at k')
    ax1.legend()
    ax1.grid(True)
    
    ax2.set_xlabel('k')
    ax2.set_ylabel('Recall@k')
    ax2.set_title('Recall at k')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved precision/recall plot to {output_path}")
    else:
        plt.show()
    plt.close()

def load_mlflow_metrics(run_ids: List[str], metric_names: List[str]) -> Dict[str, Dict[str, List[float]]]:
    """
    Load metrics from MLflow runs.
    
    Args:
        run_ids: List of MLflow run IDs
        metric_names: List of metric names to load
        
    Returns:
        Dictionary mapping run_ids to dictionaries of metrics
    """
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except ImportError:
        raise ImportError("MLflow not installed. Install with 'pip install mlflow'")
    
    client = MlflowClient()
    result = {}
    
    for run_id in run_ids:
        run_metrics = {}
        for metric_name in metric_names:
            metrics = client.get_metric_history(run_id, metric_name)
            values = [m.value for m in metrics]
            run_metrics[metric_name] = values
        result[run_id] = run_metrics
    
    return result

def plot_metrics_from_mlflow(
    experiment_name: str,
    metric_names: List[str],
    output_dir: str = "plots",
    model_name_pattern: Optional[str] = None
) -> None:
    """
    Load metrics from MLflow and create plots.
    
    Args:
        experiment_name: MLflow experiment name
        metric_names: List of metric names to plot
        output_dir: Directory to save plots
        model_name_pattern: Optional pattern to filter runs by name
    """
    try:
        import mlflow
    except ImportError:
        raise ImportError("MLflow not installed. Install with 'pip install mlflow'")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get experiment ID
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")
    
    # Get runs
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    
    if model_name_pattern:
        runs = runs[runs["tags.mlflow.runName"].str.contains(model_name_pattern, na=False)]
    
    if len(runs) == 0:
        print(f"No runs found for experiment '{experiment_name}'")
        return
    
    # Group runs by model type (assuming run names follow pattern "type_finetune")
    model_types = []
    run_ids = []
    
    for _, run in runs.iterrows():
        run_name = run["tags.mlflow.runName"]
        model_type = run_name.split("_")[0] if "_" in run_name else run_name
        model_types.append(model_type)
        run_ids.append(run["run_id"])
    
    # Load metrics
    metrics_by_run = load_mlflow_metrics(run_ids, metric_names)
    
    # Create plots for each metric
    for metric_name in metric_names:
        metric_values = []
        valid_model_types = []
        
        for run_id, model_type in zip(run_ids, model_types):
            if metric_name in metrics_by_run[run_id] and len(metrics_by_run[run_id][metric_name]) > 0:
                metric_values.append({metric_name: metrics_by_run[run_id][metric_name]})
                valid_model_types.append(model_type)
        
        if len(metric_values) > 0:
            output_path = os.path.join(output_dir, f"{metric_name}_vs_epoch.png")
            plot_metric_vs_epoch(
                metric_values,
                metric_name,
                valid_model_types,
                output_path=output_path,
                title=f"{metric_name.capitalize()} vs. Epoch by Model Type"
            )

def plot_training_metrics(train_metrics: Dict[str, List[float]], 
                         val_metrics: Dict[str, List[float]], 
                         output_path: str,
                         model_name: str) -> None:
    """Plot training and validation metrics over epochs."""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_metrics['loss']) + 1)
    
    plt.plot(epochs, train_metrics['loss'], label='Training Loss', color='blue')
    plt.plot(epochs, val_metrics['loss'], label='Validation Loss', color='red')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss - {model_name}')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_spearman_vs_epoch(spearman_scores: Dict[str, List[float]], 
                          output_path: str,
                          model_name: str) -> None:
    """Plot Spearman correlation vs epochs with mean and std."""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(spearman_scores['mean']) + 1)
    
    # Plot mean with shaded std
    plt.plot(epochs, spearman_scores['mean'], color='blue', label='Mean')
    plt.fill_between(epochs, 
                    np.array(spearman_scores['mean']) - np.array(spearman_scores['std']),
                    np.array(spearman_scores['mean']) + np.array(spearman_scores['std']),
                    alpha=0.2, color='blue')
    
    # Find and mark maximum
    max_idx = np.argmax(spearman_scores['mean'])
    plt.arrow(epochs[max_idx], spearman_scores['mean'][max_idx] - 0.1,
              0, 0.1, color='red', head_width=0.5, head_length=0.05)
    
    plt.xlabel('Epoch')
    plt.ylabel("Spearman's Correlation")
    plt.title(f"Spearman's Correlation vs Epoch - {model_name}")
    plt.grid(True)
    
    # Save plot
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_model_comparison(scores: Dict[str, List[float]], 
                         output_path: str) -> None:
    """Plot comparison of Spearman correlation between models."""
    plt.figure(figsize=(10, 6))
    
    # Prepare data
    data = []
    for model, values in scores.items():
        data.extend([(model, v) for v in values])
    df = pd.DataFrame(data, columns=['Model', 'Score'])
    
    # Create boxplot
    sns.boxplot(x='Model', y='Score', data=df, showfliers=False)
    sns.stripplot(x='Model', y='Score', data=df, color='white', size=8)
    
    # Add significance markers
    models = list(scores.keys())
    for i in range(1, len(models)):
        p_value = stats.ttest_ind(scores[models[0]], scores[models[i]])[1]
        if p_value < 0.01:
            plt.text(i, max(scores[models[i]]) + 0.05, '**', ha='center')
        elif p_value < 0.05:
            plt.text(i, max(scores[models[i]]) + 0.05, '*', ha='center')
    
    plt.ylabel("Spearman's Correlation")
    plt.title("Model Comparison: Spearman's Correlation")
    plt.grid(True)
    
    # Save plot
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_recall_precision(metrics: Dict[str, Dict[str, List[float]]], 
                         output_path: str) -> None:
    """Plot Recall@K and Precision@K for K=5 and K=10."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot Recall@K
    for model, values in metrics['recall'].items():
        ax1.plot(['Top5', 'Top10'], values, marker='o', label=model)
    ax1.set_title('Recall@K')
    ax1.set_ylabel('Recall')
    ax1.grid(True)
    ax1.legend()
    
    # Plot Precision@K
    for model, values in metrics['precision'].items():
        ax2.plot(['Top5', 'Top10'], values, marker='o', label=model)
    ax2.set_title('Precision@K')
    ax2.set_ylabel('Precision')
    ax2.grid(True)
    ax2.legend()
    
    # Add significance markers
    for ax, metric in [(ax1, 'recall'), (ax2, 'precision')]:
        for i, k in enumerate(['Top5', 'Top10']):
            values = [metrics[metric][model][i] for model in metrics[metric].keys()]
            p_values = multipletests([stats.ttest_ind(values[0], values[j])[1] 
                                    for j in range(1, len(values))], 
                                   method='fdr_bh')[1]
            for j, p in enumerate(p_values):
                if p < 0.01:
                    ax.text(i + j*0.1, max(values) + 0.05, '**', ha='center')
                elif p < 0.05:
                    ax.text(i + j*0.1, max(values) + 0.05, '*', ha='center')
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Example usage
    # Generate sample data
    np.random.seed(42)
    epochs = 30
    train_loss = [np.random.uniform(0.5, 1.0) - 0.015 * e for e in range(epochs)]
    val_loss = [np.random.uniform(0.6, 1.1) - 0.014 * e for e in range(epochs)]
    train_acc = [0.6 + np.random.uniform(0, 0.05) + 0.01 * e for e in range(epochs)]
    val_acc = [0.55 + np.random.uniform(0, 0.05) + 0.01 * e for e in range(epochs)]
    
    # Plot training metrics
    train_metrics = {"loss": train_loss, "accuracy": train_acc}
    val_metrics = {"loss": val_loss, "accuracy": val_acc}
    plot_training_metrics(train_metrics, val_metrics, "sample_training_metrics.png", model_name="ESM2 Fine-tuned")
    
    # Generate and plot Spearman correlation for different models
    model_names = ["Pretrained", "Fine-tuned (Low)", "Fine-tuned (High)"]
    spearman_values = [
        np.random.uniform(0.2, 0.4, size=5),  # Pretrained
        np.random.uniform(0.3, 0.5, size=5),  # Fine-tuned (Low)
        np.random.uniform(0.4, 0.6, size=5),  # Fine-tuned (High)
    ]
    plot_correlation_comparison(model_names, spearman_values, "sample_correlation_comparison.png") 