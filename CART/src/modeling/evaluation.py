#!/usr/bin/env python3
"""
predict_cytotoxicity_with_dummy.py

Loads embedding files, generates dummy cytotoxicity scores,
and runs nested CV to compute per‐fold predictions and Spearman's ρ.
"""

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from scipy.stats import spearmanr
from scipy.stats import dunnett
#from ..score import recall_precision_at_k
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
# — your three embeddings —
EMBED_PATHS = [
    "/Users/mukulsherekar/pythonProject/Finetuning_Activity_Prediction/embeddings/pll_results_pretrained.npy",
    "/Users/mukulsherekar/pythonProject/Finetuning_Activity_Prediction/embeddings/pll_results_finetuned_low.npy",
    "/Users/mukulsherekar/pythonProject/Finetuning_Activity_Prediction/embeddings/pll_results_finetuned_high.npy",
]

# grid for RidgeCV α
ALPHAS = np.logspace(-6, 6, num=13)

# outer CV splitter
OUTER_CV = KFold(n_splits=5, shuffle=True, random_state=42)

# for reproducible dummy labels
RNG = np.random.RandomState(0)


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


if __name__ == "__main__":
    results = {}

    for path in EMBED_PATHS:
        print(f"\n--- Processing {path} ---")
        X = np.load(path)
        n = X.shape[0]

        # dummy cytotoxicity in [0,100)
        y = RNG.rand(n) * 100

        (all_y_tests,
         all_y_preds,
         spearman_scores,
         best_alphas) = nested_ridge_cv(X, y, ALPHAS, OUTER_CV, inner_cv=3)

        # store
        results[path] = {
            "y_tests": spearman_scores,      # <-- careful: these are the ρ values
            "y_preds": all_y_preds,
            "y_tests_arr": all_y_tests,
            "spearman_rhos": spearman_scores,
            "best_alphas": best_alphas,
        }

        # quick printout
        for i, rho in enumerate(spearman_scores, 1):
            print(f" Fold {i}: α={best_alphas[i-1]:.1e}, ρ={rho:.3f}")
        mean_rho, std_rho = np.mean(spearman_scores), np.std(spearman_scores)
        print(f" → mean ρ = {mean_rho:.3f} ± {std_rho:.3f}")

    # Now `results` holds, for each embedding file:
    #  - results[...]['y_tests_arr']: list of y_test arrays per fold
    #  - results[...]['y_preds']:   list of y_pred arrays per fold
    #  - results[...]['spearman_rhos']: list of ρ per fold

    # === example: feeding into recall/precision ===
    #from scipy.stats import recall_precision_at_k  # assuming you have this util

    for path, res in results.items():
        print(f"\nMetrics for {path}:")
        r5, p5, r10, p10 = [], [], [], []
        for y_t, y_p in zip(res["y_tests_arr"], res["y_preds"]):
            rec5, prec5 = recall_precision_at_k(y_t, y_p, K=5)
            rec10, prec10 = recall_precision_at_k(y_t, y_p, K=10)
            r5.append(rec5); p5.append(prec5)
            r10.append(rec10); p10.append(prec10)
        print(f" Recall@5:  {np.mean(r5):.3f} ± {np.std(r5):.3f}")
        print(f" Precision@5: {np.mean(p5):.3f} ± {np.std(p5):.3f}")
        print(f" Recall@10: {np.mean(r10):.3f} ± {np.std(r10):.3f}")
        print(f" Precision@10: {np.mean(p10):.3f} ± {np.std(p10):.3f}")

    # === example: Dunnett's test between two sets of rhos ===
    pre = np.array(results[EMBED_PATHS[0]]["spearman_rhos"])
    high = np.array(results[EMBED_PATHS[2]]["spearman_rhos"])

    # Verify the contents of pre and high arrays
    print("pre array:", pre)
    print("high array:", high)
    if pre.size == 0 or high.size == 0:
        raise ValueError("One of the arrays is empty. Please check the data.")

    # Call dunnett function
    dun = dunnett(pre, high, control=0)
    print("\nDunnett's test (high vs pre):")
    print(" comparisons:", dun.comparisons)
    print(" statistics :", dun.statistic)
    print(" p-values   :", dun.pvalue)
