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
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from scipy.stats import spearmanr
import glob


def parse_args(args_list=None) -> argparse.Namespace:
    
    parser = argparse.ArgumentParser(
            description="Run prediction model on embeddings and evaluate performance"
        )
    parser.add_argument(
        "--embedding_dir",
        type=Path,
        default=Path('output/embeddings'),
        help="Directory containing embedding .npy files"
    )
    parser.add_argument(
        "--embedding_files",
        nargs="+",
        type=str,
        help="Specific embedding files to use (basename, without path). "
             "If omitted, all .npy in embedding_dir are used"
    )
    parser.add_argument(
        "--labels_path",
        type=Path,
        default=Path('output/mutants/CAR_mutants_cytox.csv'),
        help="Path to CSV file containing cytotoxicity labels"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path('output/results'),
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
        help="Number of splits for outer cross-validation"
    )
    parser.add_argument(
        "--n_inner_splits",
        type=int,
        default=3,
        help="Number of splits for inner cross-validation"
    )
    return parser.parse_args(args_list)


def load_labels_and_match_ids(labels_path: Path, emb_path: Path):
    """Load labels and match to the order in emb_path.ids.txt"""
    df = pd.read_csv(labels_path)
    if 'sequence_id' not in df.columns or 'cytotoxicity' not in df.columns:
        raise ValueError("Labels CSV must have 'sequence_id' and 'cytotoxicity' columns")

    ids_file = emb_path.with_suffix('.ids.txt')
    if not ids_file.exists():
        raise FileNotFoundError(f"IDs file not found: {ids_file}")

    seq_ids = [line.strip() for line in ids_file.read_text().splitlines()]

    matched_labels = []
    matched_ids = []
    unmatched = []
    for sid in seq_ids:
        hit = df[df['sequence_id'] == sid]
        if not hit.empty:
            matched_labels.append(hit['cytotoxicity'].values[0])
            matched_ids.append(sid)
        else:
            unmatched.append(sid)

    if unmatched:
        print(f"Warning: {len(unmatched)} sequences without labels, e.g.: {unmatched[:5]}")

    print(f"Matched {len(matched_labels)} / {len(seq_ids)} sequences with labels")
    return np.array(matched_labels), matched_ids


def get_embedding_files(emb_dir: Path, specific: list[str] | None):
    """Return list of full Paths to .npy embeddings to process."""
    if specific:
        files = []
        for name in specific:
            p = emb_dir / name
            if not name.endswith('.npy'):
                p = p.with_suffix('.npy')
            if p.exists():
                files.append(p)
            else:
                print(f"Warning: embedding file not found: {p}")
        if not files:
            raise FileNotFoundError(f"No specified embedding files found in {emb_dir}")
    else:
        files = list(emb_dir.glob("*.npy"))
        if not files:
            raise FileNotFoundError(f"No .npy files found in {emb_dir}")
    print(f"Found {len(files)} embeddings: {[p.name for p in files]}")
    return files


def plot_correlation(y_test, y_pred, fold, model_name, out_dir: Path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # Save correlation data as CSV for custom plotting
    corr_data = pd.DataFrame({
        'actual': y_test,
        'predicted': y_pred
    })
    csv_path = out_dir / f"{model_name}_fold{fold}_correlation.csv"
    corr_data.to_csv(csv_path, index=False)
    
    # scatter + fit
    ax1.scatter(y_pred, y_test, alpha=0.5)
    z = np.polyfit(y_pred, y_test, 1)
    ax1.plot(y_pred, np.poly1d(z)(y_pred), 'r--', alpha=0.8)
    rho, pval = spearmanr(y_test, y_pred)
    pearson_r = np.corrcoef(y_test, y_pred)[0, 1]
    ax1.set_title(f"Fold {fold} – {model_name}")
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")
    ax1.text(0.05, 0.95,
             f"Spearman ρ={rho:.3f} (p={pval:.2g})\nPearson r={pearson_r:.3f}",
             transform=ax1.transAxes, bbox=dict(facecolor='white', alpha=0.8))

    # distributions
    sns.histplot(y_test, label="Actual", ax=ax2, alpha=0.5)
    sns.histplot(y_pred, label="Predicted", ax=ax2, alpha=0.5)
    ax2.set_title("Value Distributions")
    ax2.legend()

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{model_name}_fold{fold}_correlation.png", dpi=300)
    plt.close(fig)

    return rho, pval, pearson_r


def plot_spearman_whisker(scores, model_name, out_dir: Path):
    # Save spearman scores as CSV for custom plotting
    scores_df = pd.DataFrame({
        'fold': range(1, len(scores) + 1),
        'spearman_rho': scores
    })
    csv_path = out_dir / f"{model_name}_spearman_scores.csv"
    scores_df.to_csv(csv_path, index=False)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot(scores, labels=[model_name])
    ax.scatter(np.ones_like(scores), scores, color='red', alpha=0.6)
    mean = np.mean(scores)
    ax.axhline(mean, linestyle='--', label=f"Mean={mean:.3f}")
    ax.set_ylabel("Spearman ρ")
    ax.set_title(f"Spearman Scores – {model_name}")
    ax.legend()

    fig.savefig(out_dir / f"{model_name}_spearman_whisker.png", dpi=300)
    plt.close(fig)


def run_prediction(args):
    # resolve absolute paths
    root = Path(__file__).parent.parent.parent.parent.resolve()
    emb_dir    = args.embedding_dir if args.embedding_dir.is_absolute() else root / args.embedding_dir
    labels_path = args.labels_path if args.labels_path.is_absolute() else root / args.labels_path
    out_dir    = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    embedding_files = get_embedding_files(emb_dir, args.embedding_files)

    alphas   = np.logspace(-6, 6, 13)
    outer_cv = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_seed)
    rng      = np.random.RandomState(args.random_seed)

    summary = {}
    for emb_path in embedding_files:
        model_name = emb_path.stem
        print(f"\n=== Evaluating {model_name} ===")
        X = np.load(emb_path)

        # match labels
        try:
            y, matched_ids = load_labels_and_match_ids(labels_path, emb_path)
        except Exception as e:
            print(f"Label load error: {e}\nUsing dummy labels")
            y = rng.rand(X.shape[0]) * 100

        preds, acts = [], []
        spearman_scores, pearson_scores = [], []

        for fold, (tr, te) in enumerate(outer_cv.split(X), start=1):
            Xtr, Xte = X[tr], X[te]
            ytr, yte = y[tr], y[te]
            clf = RidgeCV(alphas=alphas, cv=args.n_inner_splits)
            clf.fit(Xtr, ytr)
            ypred = clf.predict(Xte)

            preds.extend(ypred); acts.extend(yte)
            rho, pval, pr = plot_correlation(yte, ypred, fold, model_name, out_dir)
            spearman_scores.append(rho); pearson_scores.append(pr)

            print(f" Fold {fold}: α={clf.alpha_:.1e}, ρ={rho:.3f}, r={pr:.3f}")

        plot_spearman_whisker(spearman_scores, model_name, out_dir)

        summary[model_name] = {
            "spearman": np.array(spearman_scores),
            "pearson": np.array(pearson_scores),
            "mean_spearman": np.mean(spearman_scores),
            "mean_pearson": np.mean(pearson_scores)
        }

        # save per-model
        np.save(out_dir / f"{model_name}_correlations.npy", summary[model_name])
        np.savez(out_dir / f"{model_name}_predictions.npz",
                 predictions=np.array(preds),
                 actuals=np.array(acts),
                 spearman=spearman_scores,
                 pearson=pearson_scores)

    # write summary
    with open(out_dir / "prediction_summary.txt", "w") as f:
        for mn, res in summary.items():
            f.write(f"{mn}: mean ρ={res['mean_spearman']:.3f}, r={res['mean_pearson']:.3f}\n")

    print(f"\nSummary saved to {out_dir/'prediction_summary.txt'}")
    return summary


def main():
    args = parse_args()
    run_prediction(args)


if __name__ == "__main__":
    main()
