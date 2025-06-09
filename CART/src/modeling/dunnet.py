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
    results_dir = project_root / "output" / "results"
    
    parser = argparse.ArgumentParser(description="Run Dunnett's test for protein language model comparison")
    parser.add_argument(
        "--results_dir", 
        type=Path, 
        default=results_dir,
        help="Directory containing model prediction results"
    )
    parser.add_argument(
        "--baseline_model", 
        type=str, 
        default="pretrained",
        help="Name of the baseline model (typically pretrained, without file suffix)"
    )
    parser.add_argument(
        "--finetuned_models", 
        type=str, 
        nargs="+",
        help="Names of fine-tuned models to compare against baseline (without file suffix)"
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
        help="Generate comprehensive comparison plots"
    )
    
    # Only parse command line arguments if this module is run directly
    if args_list is None and __name__ == "__main__":
        return parser.parse_args()
    else:
        # When imported, use the provided args_list or an empty list
        return parser.parse_args(args_list or [])

def load_spearman_scores(results_dir, model_name):
    """Load Spearman correlation scores for a model with consolidated file pattern checking."""
    # Consolidated file patterns - order by preference
    file_patterns = [
        # Direct spearman files
        f"{model_name}_spearman.npy",
        f"{model_name}_correlations.npy",
        f"pll_results_{model_name}_spearman.npy",
        # NPZ files
        f"{model_name}_predictions.npz",
        f"{model_name}_correlations.npz"
    ]
    
    # Check regular files first
    for pattern in file_patterns[:3]:
        file_path = results_dir / pattern
        if file_path.exists():
            scores = np.load(file_path)
            # Handle different array types
            if isinstance(scores, np.ndarray):
                if scores.dtype == np.dtype('O'):  # Object array
                    return scores.item().get('spearman', scores)
                return scores
            return scores
    
    # Check NPZ files
    npz_keys = ['spearman', 'spearman_rhos']
    for pattern in file_patterns[3:]:
        file_path = results_dir / pattern
        if file_path.exists():
            data = np.load(file_path)
            for key in npz_keys:
                if key in data:
                    return data[key]
    
    raise FileNotFoundError(f"Could not find Spearman scores for {model_name} in {results_dir}")

def find_available_models(results_dir):
    """Find all available model results with consolidated pattern matching."""
    models = set()
    
    # Consolidated file patterns and their corresponding model name extraction
    patterns_and_replacements = [
        ("*_spearman.npy", "_spearman"),
        ("*_correlations.npy", "_correlations"),
        ("*_predictions.npz", "_predictions"),
        ("pll_results_*.npy", "pll_results_")
    ]
    
    for pattern, replacement in patterns_and_replacements:
        files = glob.glob(str(results_dir / pattern))
        for f in files:
            if replacement.startswith("pll_results_"):
                # Special handling for pll_results prefix
                model_name = Path(f).stem.replace(replacement, "")
            else:
                # Standard suffix removal
                model_name = Path(f).stem.replace(replacement, "")
            if model_name:  # Only add non-empty model names
                models.add(model_name)
    
    return sorted(list(models))  # Return sorted for consistency

def create_results_dataframe(control_name, model_names, padded_scores, result, alpha):
    """Create a comprehensive results dataframe."""
    results_list = []
    control_scores = padded_scores[0]
    
    # DunnettResult has statistic and pvalue arrays, one for each treatment vs control comparison
    for i in range(len(result.statistic)):
        treatment_idx = i + 1  # Treatment models start at index 1 (control is at 0)
        treatment_scores = padded_scores[treatment_idx]
        
        # Calculate Cohen's d (effect size)
        mean_control = np.nanmean(control_scores)
        mean_treatment = np.nanmean(treatment_scores)
        std_control = np.nanstd(control_scores)
        std_treatment = np.nanstd(treatment_scores)
        
        # Pooled standard deviation for Cohen's d
        n_control = np.sum(~np.isnan(control_scores))
        n_treatment = np.sum(~np.isnan(treatment_scores))
        pooled_std = np.sqrt(((n_control - 1) * std_control**2 + (n_treatment - 1) * std_treatment**2) / 
                            (n_control + n_treatment - 2))
        
        cohens_d = (mean_treatment - mean_control) / pooled_std if pooled_std > 0 else 0
        
        results_list.append({
            'baseline_model': control_name,
            'finetuned_model': model_names[treatment_idx],
            'statistic': result.statistic[i],
            'p_value': result.pvalue[i],
            'significant': result.pvalue[i] < alpha,
            'mean_baseline': mean_control,
            'mean_finetuned': mean_treatment,
            'improvement': mean_treatment - mean_control,
            'std_baseline': std_control,
            'std_finetuned': std_treatment,
            'cohens_d': cohens_d,
            'effect_size': 'large' if abs(cohens_d) >= 0.8 else 'medium' if abs(cohens_d) >= 0.5 else 'small'
        })
    
    return pd.DataFrame(results_list)

def save_results(results_df, model_names, padded_scores, output_dir):
    """Save all results in a consolidated manner."""
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save raw scores
    scores_df = pd.DataFrame({name: scores for name, scores in zip(model_names, padded_scores)})
    scores_df.to_csv(output_dir / "spearman_correlations_by_model.csv", index=False)
    
    # Save test results
    results_df.to_csv(output_dir / "dunnett_test_results.csv", index=False)
    
    # Save summary text file
    with open(output_dir / "dunnett_results.txt", 'w') as f:
        f.write("Dunnett's Test Results - Protein Language Model Comparison\n")
        f.write("=" * 60 + "\n\n")
        f.write("Statistical comparison of Spearman correlations between baseline and fine-tuned models\n")
        f.write("for protein sequence-function prediction tasks.\n\n")
        
        for _, row in results_df.iterrows():
            f.write(f"{row['finetuned_model']} vs {row['baseline_model']}:\n")
            f.write(f"  Test statistic: {row['statistic']:.4f}\n")
            f.write(f"  p-value: {row['p_value']:.4f}\n")
            f.write(f"  Mean correlation improvement: {row['improvement']:.4f}\n")
            f.write(f"  Cohen's d (effect size): {row['cohens_d']:.4f} ({row['effect_size']})\n")
            significance = "significant" if row['significant'] else "not significant"
            f.write(f"  Statistical significance: {significance}\n\n")
    
    return scores_df

def plot_comparison(results_df, model_names, padded_scores, output_dir):
    """Generate comprehensive comparison plots for protein language model evaluation."""
    # Calculate means and standard deviations for all models
    means = [np.nanmean(s) for s in padded_scores]
    stds = [np.nanstd(s) for s in padded_scores]
    
    # Create individual plots first
    
    # Plot 1: Model Performance Comparison with Error Bars (Individual)
    plt.figure(figsize=(10, 6))
    colors = ['steelblue'] + ['lightcoral' if sig else 'lightblue' 
                              for sig in list(results_df['significant'])]
    bars = plt.bar(model_names, means, yerr=stds, capsize=5, 
                   color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    plt.title('Protein Language Model Performance\n(Spearman Correlation)', fontsize=14, fontweight='bold')
    plt.ylabel('Spearman Correlation (ρ)', fontsize=12)
    plt.xlabel('Model Type', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Dunnett Test Statistics (Individual) - Include baseline as reference
    plt.figure(figsize=(10, 6))
    # Add baseline model with statistic = 0 (reference point)
    all_models = [model_names[0]] + results_df['finetuned_model'].tolist()
    test_stats = [0] + results_df['statistic'].tolist()  # Baseline has 0 statistic
    sig_colors = ['steelblue'] + ['red' if sig else 'lightblue' for sig in results_df['significant']]
    
    bars = plt.bar(all_models, test_stats, color=sig_colors, alpha=0.8, 
                   edgecolor='black', linewidth=1)
    plt.title('Statistical Significance Test\n(Dunnett Test)', fontsize=14, fontweight='bold')
    plt.ylabel('Test Statistic', fontsize=12)
    plt.xlabel('Model', fontsize=12)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xticks(rotation=45)
    
    # Add value labels
    for bar, stat in zip(bars, test_stats):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{stat:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'dunnett_test_statistics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: P-values (Individual) - Only for fine-tuned models
    plt.figure(figsize=(10, 6))
    finetuned_models = results_df['finetuned_model'].tolist()
    p_values = results_df['p_value'].tolist()
    sig_colors = ['red' if sig else 'lightblue' for sig in results_df['significant']]
    
    bars = plt.bar(finetuned_models, p_values, color=sig_colors, alpha=0.8,
                   edgecolor='black', linewidth=1)
    plt.title('Statistical Significance\n(p-values vs Baseline)', fontsize=14, fontweight='bold')
    plt.ylabel('p-value', fontsize=12)
    plt.xlabel('Fine-tuned Model', fontsize=12)
    plt.axhline(y=0.05, color='red', linestyle='--', label='α=0.05', alpha=0.7, linewidth=2)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xticks(rotation=45)
    plt.legend()
    
    # Add significance indicators
    for bar, p, sig in zip(bars, p_values, results_df['significant']):
        height = bar.get_height()
        symbol = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                symbol, ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'p_values_significance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 4: Cohen's d (Individual) - Include baseline as reference
    plt.figure(figsize=(10, 6))
    # Add baseline model with Cohen's d = 0 (reference point)
    all_models = [model_names[0]] + results_df['finetuned_model'].tolist()
    cohens_d_values = [0] + results_df['cohens_d'].tolist()  # Baseline has 0 effect size
    effect_colors = ['steelblue'] + ['darkgreen' if abs(d) >= 0.8 else 'orange' if abs(d) >= 0.5 else 'lightgray' 
                                     for d in results_df['cohens_d'].tolist()]
    
    bars = plt.bar(all_models, cohens_d_values, color=effect_colors, alpha=0.8,
                   edgecolor='black', linewidth=1)
    plt.title('Effect Size Analysis\n(Cohen\'s d)', fontsize=14, fontweight='bold')
    plt.ylabel('Cohen\'s d', fontsize=12)
    plt.xlabel('Model', fontsize=12)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    plt.axhline(y=0.5, color='orange', linestyle=':', alpha=0.7, label='Medium effect')
    plt.axhline(y=0.8, color='green', linestyle=':', alpha=0.7, label='Large effect')
    plt.axhline(y=-0.5, color='orange', linestyle=':', alpha=0.7)
    plt.axhline(y=-0.8, color='green', linestyle=':', alpha=0.7)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xticks(rotation=45)
    plt.legend()
    
    # Add value labels
    for bar, d in zip(bars, cohens_d_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{d:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cohens_d_effect_size.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 5: Performance Improvement (Individual) - Show absolute performance + improvement
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left: Absolute performance with baseline
    ax1.bar(model_names, means, yerr=stds, capsize=5, 
            color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax1.set_title('Absolute Performance', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Spearman Correlation (ρ)', fontsize=11)
    ax1.set_xlabel('Model', fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    for bar, mean, std in zip(ax1.patches, means, stds):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Right: Improvement relative to baseline
    improvements = results_df['improvement'].tolist()
    improvement_colors = ['green' if imp > 0 else 'red' for imp in improvements]
    
    ax2.bar(finetuned_models, improvements, color=improvement_colors, alpha=0.8,
            edgecolor='black', linewidth=1)
    ax2.set_title('Performance Improvement\n(vs Baseline)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Δ Spearman Correlation', fontsize=11)
    ax2.set_xlabel('Fine-tuned Model', fontsize=11)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    for bar, imp in zip(ax2.patches, improvements):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                f'{imp:+.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_improvement.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 6: Summary Statistics Table (Individual)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Create summary table including baseline
    table_data = []
    # Add baseline row
    baseline_mean = means[0]
    baseline_std = stds[0]
    table_data.append([
        model_names[0] + " (baseline)",
        f"{baseline_mean:.3f}",
        "0.000",  # No improvement vs itself
        "0.000",  # No effect size vs itself
        "—",      # No p-value vs itself
        "—"       # No significance vs itself
    ])
    
    # Add fine-tuned model rows
    for _, row in results_df.iterrows():
        table_data.append([
            row['finetuned_model'],
            f"{row['mean_finetuned']:.3f}",
            f"{row['improvement']:+.3f}",
            f"{row['cohens_d']:.3f}",
            f"{row['p_value']:.3f}",
            "✓" if row['significant'] else "✗"
        ])
    
    table = ax.table(cellText=table_data,
                     colLabels=['Model', 'Mean ρ', 'Δρ', 'Cohen\'s d', 'p-value', 'Sig.'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.3, 2)
    ax.set_title('Summary Statistics - Protein Language Model Comparison', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Color code the significance column (skip baseline row)
    for i in range(1, len(table_data)):
        if table_data[i][5] == "✓":
            table[(i+1, 5)].set_facecolor('#90EE90')  # Light green
        elif table_data[i][5] == "✗":
            table[(i+1, 5)].set_facecolor('#FFB6C1')  # Light red
    
    # Highlight baseline row
    for j in range(6):
        table[(1, j)].set_facecolor('#E6E6FA')  # Light lavender for baseline
    
    plt.savefig(output_dir / 'summary_statistics_table.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Now create the comprehensive figure with all subplots
    fig = plt.figure(figsize=(20, 12))
    
    # Plot 1: Model Performance Comparison with Error Bars
    ax1 = plt.subplot(2, 3, 1)
    bars1 = ax1.bar(model_names, means, yerr=stds, capsize=5, 
                    color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax1.set_title('Model Performance\n(Spearman Correlation)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Spearman Correlation (ρ)', fontsize=11)
    ax1.set_xlabel('Model Type', fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, mean, std in zip(bars1, means, stds):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Dunnett Test Statistics (with baseline)
    ax2 = plt.subplot(2, 3, 2)
    all_models_short = [model_names[0]] + results_df['finetuned_model'].tolist()
    test_stats_all = [0] + results_df['statistic'].tolist()
    sig_colors_all = ['steelblue'] + ['red' if sig else 'lightblue' for sig in results_df['significant']]
    
    bars2 = ax2.bar(all_models_short, test_stats_all, color=sig_colors_all, alpha=0.8, 
                    edgecolor='black', linewidth=1)
    ax2.set_title('Dunnett Test Statistics', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Test Statistic', fontsize=11)
    ax2.set_xlabel('Model', fontsize=11)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # Plot 3: P-values
    ax3 = plt.subplot(2, 3, 3)
    bars3 = ax3.bar(finetuned_models, p_values, color=sig_colors[1:], alpha=0.8,
                    edgecolor='black', linewidth=1)
    ax3.set_title('P-values', fontsize=12, fontweight='bold')
    ax3.set_ylabel('p-value', fontsize=11)
    ax3.set_xlabel('Fine-tuned Model', fontsize=11)
    ax3.axhline(y=0.05, color='red', linestyle='--', label='α=0.05', alpha=0.7, linewidth=2)
    ax3.grid(True, linestyle='--', alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)
    ax3.legend()
    
    # Plot 4: Cohen's d (with baseline)
    ax4 = plt.subplot(2, 3, 4)
    cohens_d_all = [0] + results_df['cohens_d'].tolist()
    effect_colors_all = ['steelblue'] + ['darkgreen' if abs(d) >= 0.8 else 'orange' if abs(d) >= 0.5 else 'lightgray' 
                                         for d in results_df['cohens_d'].tolist()]
    
    bars4 = ax4.bar(all_models_short, cohens_d_all, color=effect_colors_all, alpha=0.8,
                    edgecolor='black', linewidth=1)
    ax4.set_title('Effect Size (Cohen\'s d)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Cohen\'s d', fontsize=11)
    ax4.set_xlabel('Model', fontsize=11)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax4.grid(True, linestyle='--', alpha=0.3)
    ax4.tick_params(axis='x', rotation=45)
    
    # Plot 5: Performance Improvement
    ax5 = plt.subplot(2, 3, 5)
    improvements = results_df['improvement'].tolist()
    improvement_colors = ['green' if imp > 0 else 'red' for imp in improvements]
    
    bars5 = ax5.bar(finetuned_models, improvements, color=improvement_colors, alpha=0.8,
                    edgecolor='black', linewidth=1)
    ax5.set_title('Performance Improvement', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Δ Spearman Correlation', fontsize=11)
    ax5.set_xlabel('Fine-tuned Model', fontsize=11)
    ax5.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax5.grid(True, linestyle='--', alpha=0.3)
    ax5.tick_params(axis='x', rotation=45)
    
    # Plot 6: Summary Statistics Table (simplified for subplot)
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('tight')
    ax6.axis('off')
    
    # Simplified table for subplot
    simple_table_data = []
    for _, row in results_df.iterrows():
        simple_table_data.append([
            row['finetuned_model'],
            f"{row['mean_finetuned']:.3f}",
            f"{row['improvement']:+.3f}",
            f"{row['cohens_d']:.3f}",
            "✓" if row['significant'] else "✗"
        ])
    
    table = ax6.table(cellText=simple_table_data,
                     colLabels=['Model', 'Mean ρ', 'Δρ', 'Cohen\'s d', 'Sig.'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax6.set_title('Summary Statistics', fontsize=12, fontweight='bold')
    
    plt.suptitle('Protein Language Model Fine-tuning Evaluation\nSpearman Correlation Analysis', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(output_dir / 'comprehensive_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def run_dunnett(args):
    """Run optimized Dunnett's test analysis."""
    # Setup directories
    args.results_dir.mkdir(parents=True, exist_ok=True)
    if args.output_dir is None:
        args.output_dir = args.results_dir
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Auto-detect treatment models if not specified
    if args.finetuned_models is None:
        available_models = find_available_models(args.results_dir)
        args.finetuned_models = [m for m in available_models if m != args.baseline_model]
        if not args.finetuned_models:
            print(f"Error: No fine-tuned models found in {args.results_dir}")
            print(f"Available models: {available_models}")
            return None
    
    print(f"Baseline model: {args.baseline_model}")
    print(f"Fine-tuned models: {args.finetuned_models}")
    
    # Load all scores
    try:
        control_scores = load_spearman_scores(args.results_dir, args.baseline_model)
        all_scores = [control_scores]
        model_names = [args.baseline_model]
        
        for model in args.finetuned_models:
            try:
                scores = load_spearman_scores(args.results_dir, model)
                all_scores.append(scores)
                model_names.append(model)
            except FileNotFoundError as e:
                print(f"Warning: {e}")
        
        if len(all_scores) < 2:
            print("Error: Need at least one fine-tuned model with valid scores")
            return None
            
    except FileNotFoundError as e:
        print(f"Error loading baseline model: {e}")
        return None
    
    # Prepare data for Dunnett's test
    max_len = max(len(s) for s in all_scores)
    padded_scores = []
    for scores in all_scores:
        if len(scores) < max_len:
            padded = np.pad(scores, (0, max_len - len(scores)), 
                           'constant', constant_values=np.nan)
            padded_scores.append(padded)
        else:
            padded_scores.append(scores)
    
    # Run Dunnett's test
    # scipy.stats.dunnett expects: dunnett(*samples, control=control_array)
    # where samples are individual 1D arrays, not a 2D matrix
    control_data = padded_scores[0]
    treatment_data = padded_scores[1:]  # All treatment groups
    
    # Remove NaN values for the test
    control_clean = control_data[~np.isnan(control_data)]
    treatment_clean = []
    for treatment in treatment_data:
        clean_treatment = treatment[~np.isnan(treatment)]
        treatment_clean.append(clean_treatment)
    
    # Run Dunnett's test with proper format
    result = dunnett(*treatment_clean, control=control_clean)
    
    # Create and save results
    results_df = create_results_dataframe(args.baseline_model, model_names, padded_scores, result, args.alpha)
    scores_df = save_results(results_df, model_names, padded_scores, args.output_dir)
    
    # Print summary
    print(f"\nProtein Language Model Evaluation Results:")
    print("=" * 60)
    for _, row in results_df.iterrows():
        print(f"{row['finetuned_model']} vs {row['baseline_model']}:")
        print(f"  Correlation improvement: {row['improvement']:+.4f}")
        print(f"  Effect size (Cohen's d): {row['cohens_d']:.4f} ({row['effect_size']})")
        print(f"  Statistical significance: p={row['p_value']:.4f} ({'significant' if row['significant'] else 'not significant'})")
    
    # Generate plots if requested
    if args.plot:
        plot_comparison(results_df, model_names, padded_scores, args.output_dir)
        print(f"\nComprehensive plots saved to {args.output_dir}")
    
    print(f"\nResults saved to {args.output_dir}")
    
    return {
        "results_df": results_df,
        "scores_df": scores_df,
        "test_result": result
    }

def main():
    args = parse_args()
    run_dunnett(args)

if __name__ == "__main__":
    main()

#Notes:
#data must be an array of shape (n_observations, n_groups).
#control=0 tells SciPy that column 0 (your pre-trained group) is the reference.
#result typically exposes .comparisons, .statistic, and .pvalue, which you can print or log.