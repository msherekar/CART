#!/usr/bin/env python3
"""
predict_cytotoxicity_with_dummy.py

Loads embedding files, generates dummy cytotoxicity scores,
and runs nested 5-fold CV (outer) with 3-fold RidgeCV (inner)
to compute Spearman's ρ for each embedding set.
"""
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from scipy.stats import spearmanr

# Paths to your three embedding files
embedding_paths = [
    "/Users/mukulsherekar/pythonProject/Finetuning_Activity_Prediction/embeddings/pll_results_pretrained.npy",
    "/Users/mukulsherekar/pythonProject/Finetuning_Activity_Prediction/embeddings/pll_results_finetuned_low.npy",
    "/Users/mukulsherekar/pythonProject/Finetuning_Activity_Prediction/embeddings/pll_results_finetuned_high.npy",
]

# hyperparameter grid for RidgeCV: 10^-6 … 10^6 (log-spaced)
alphas = np.logspace(-6, 6, num=13)

# outer 5-fold splitter
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)

# fix random seed for dummy labels
rng = np.random.RandomState(0)

for emb_path in embedding_paths:
    print(f"\n=== Evaluating embeddings: {emb_path} ===")
    X = np.load(emb_path)  # shape (n_samples, H)
    n = X.shape[0]

    # Generate dummy continuous cytotoxicity scores (e.g., in range [0, 100])
    y = rng.rand(n) * 100

    spearman_scores = []

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train = X_train.reshape(-1, 1)
        X_test = X_test.reshape(-1, 1)

        # inner 3-fold RidgeCV to pick best alpha
        ridge_cv = RidgeCV(alphas=alphas, cv=3)
        ridge_cv.fit(X_train, y_train)

        # predict on held-out fold
        y_pred = ridge_cv.predict(X_test)

        # compute Spearman correlation
        rho, pval = spearmanr(y_test, y_pred)
        spearman_scores.append(rho)
        print(
            f" Fold {fold}: best α={ridge_cv.alpha_:.1e}, "
            f"Spearman's ρ={rho:.3f} (p={pval:.3g})"
        )

    mean_rho = np.mean(spearman_scores)
    std_rho  = np.std(spearman_scores)
    print(
        f" --> 5-fold outer Spearman's ρ: "
        f"mean={mean_rho:.3f} ± {std_rho:.3f}"
    )
