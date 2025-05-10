#!/usr/bin/env python3
"""
score.py
Scores for Ridge Models
Computes evaluation metrics for CAR-T cell activity predictions:
- Spearman's correlation
- Recall@K
- Precision@K

Can score a single .npz predictions file or all "*_predictions.npz" in a directory,
and outputs JSON summaries plus per-model recall/precision plots.
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from pathlib import Path
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

# constants for key lookup in .npz files
ACTUAL_KEYS = ('actuals', 'y_test', 'y_tests', 'y_true')
PRED_KEYS   = ('predictions', 'y_pred', 'y_preds')
DEFAULT_K   = [5, 10, 20]


def parse_args(args_list=None):
    
    parser = argparse.ArgumentParser(description="Score CAR-T cytotoxicity predictions")
    parser.add_argument(
        "--predictions_file",
        type=Path,
        help="Path to a single .npz predictions file"
    )
    parser.add_argument(
        "--predictions_dir",
        type=Path,
        default=Path('output/results'),
        help="Directory of *_predictions.npz files (if --predictions_file omitted)"
    )
    parser.add_argument(
        "--model_names",
        nargs="+",
        help="Basenames (without suffix) of models to score in predictions_dir"
    )
    parser.add_argument(
        "--k_values",
        nargs="+",
        type=int,
        default=DEFAULT_K,
        help="List of K values for Recall@K / Precision@K"
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        default=Path('output/results/scores.json'),
        help="Where to write JSON summary"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path('output/plots'),
        help="Where to save recall/precision plots"
    )
    # Only parse command line arguments if this module is run directly
    if args_list is None and __name__ == "__main__":
        return parser.parse_args()
    else:
        # When imported, use the provided args_list or an empty list
        return parser.parse_args(args_list or [])

    


def load_preds(path: Path):
    """Load y_true and y_pred arrays from a .npz, trying common key names."""
    data = np.load(path)
    for k in ACTUAL_KEYS:
        if k in data:
            y_true = data[k]
            break
    else:
        raise KeyError(f"No actuals key in {path}: {list(data.keys())}")

    for k in PRED_KEYS:
        if k in data:
            y_pred = data[k]
            break
    else:
        raise KeyError(f"No predictions key in {path}: {list(data.keys())}")

    return y_true, y_pred


def collect_prediction_files(args) -> list[Path]:
    """Return a list of .npz files to score based on args."""
    if args.predictions_file:
        return [args.predictions_file]
    base = args.predictions_dir
    if args.model_names:
        return [base / f"{name}_predictions.npz" for name in args.model_names]
    return sorted(base.glob("*_predictions.npz"))


def recall_precision_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int, cutoff: float):
    """Compute Recall@k and Precision@k using a precomputed cutoff."""
    positives = y_true >= cutoff
    n_pos = positives.sum()
    topk = np.argsort(y_pred)[-k:]
    tp = positives[topk].sum()
    recall = tp / n_pos if n_pos > 0 else 0.0
    precision = tp / k
    return recall, precision


def score_predictions(y_true: np.ndarray,
                      y_pred: np.ndarray,
                      k_values: list[int]) -> dict:
    """Compute Spearman ρ and Recall/Precision@K for given arrays."""
    rho, pval = spearmanr(y_true, y_pred)
    cutoff = np.percentile(y_true, 75)

    recall_at_k = {}
    precision_at_k = {}
    for k in k_values:
        r, p = recall_precision_at_k(y_true, y_pred, k, cutoff)
        recall_at_k[k] = float(r)
        precision_at_k[k] = float(p)

    
    return {
        "spearman_rho": float(rho),
        "spearman_pval": float(pval),
        "recall_at_k": recall_at_k,
        "precision_at_k": precision_at_k
    }


def plot_recall_precision(scores: dict, model_name: str, out_dir: Path):
    """Plot Recall@K and Precision@K versus K."""
    out_dir.mkdir(parents=True, exist_ok=True)

    ks = sorted(scores["recall_at_k"])
    rec = [scores["recall_at_k"][k] for k in ks]
    pre = [scores["precision_at_k"][k] for k in ks]

    # Save data as CSV for custom plotting
    data = {
        'k': ks,
        'recall': rec,
        'precision': pre
    }
    df = pd.DataFrame(data)
    csv_path = out_dir / f"{model_name}_recall_precision.csv"
    df.to_csv(csv_path, index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(ks, rec, 'o-', label="Recall@K")
    plt.plot(ks, pre, 's-', label="Precision@K")
    for k, r, p in zip(ks, rec, pre):
        plt.text(k, r, f"{r:.2f}", ha='center', va='bottom')
        plt.text(k, p, f"{p:.2f}", ha='center', va='top')
    plt.title(f"Recall/Precision @K — {model_name}")
    plt.xlabel("K")
    plt.ylabel("Score")
    plt.xticks(ks, [f"Top-{k}" for k in ks])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{model_name}_recall_precision.png", dpi=300)
    plt.close()


def run_score(args):
    root = Path(__file__).parent.parent.parent.parent.resolve()
    files = collect_prediction_files(args)
    out_plots = args.output_dir.resolve()
    if not out_plots.is_absolute():
        out_plots = root / out_plots
    out_plots.mkdir(parents=True, exist_ok=True)

    all_scores: dict[str, dict] = {}
    for fp in files:
        if not fp.exists():
            print(f"Warning: file not found, skipping: {fp}")
            continue
        print(f"Scoring {fp.name}...")
        try:
            y_true, y_pred = load_preds(fp)
        except KeyError as e:
            print(f"  ERROR: {e}")
            continue

        scores = score_predictions(y_true, y_pred, args.k_values)
        model_name = fp.stem.replace("_predictions", "")
        all_scores[model_name] = scores
        plot_recall_precision(scores, model_name, out_plots)

    # write JSON summary
    out_json = args.output_file.resolve()
    if not out_json.is_absolute():
        out_json = root / out_json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(all_scores, f, indent=2)
    print(f"\nSaved all scores to {out_json}")

    return all_scores


def main():
    args = parse_args()
    run_score(args)


if __name__ == "__main__":
    main()
