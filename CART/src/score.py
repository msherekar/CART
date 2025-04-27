import numpy as np
from scipy.stats import spearmanr

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

# — example usage inside your 5-fold loop ——————————————
spearman_scores = []
recall5, precision5 = [], []
recall10, precision10 = [], []

# suppose you have lists of y_test and y_pred for each fold:
for fold, (y_test, y_pred) in enumerate(zip(all_y_tests, all_y_preds), start=1):
    # 1) Spearman’s ρ
    rho, pval = spearmanr(y_test, y_pred)
    spearman_scores.append(rho)

    # 2) Recall@K and Precision@K
    r5, p5   = recall_precision_at_k(y_test, y_pred, K=5)
    r10, p10 = recall_precision_at_k(y_test, y_pred, K=10)
    recall5.append(r5);    precision5.append(p5)
    recall10.append(r10);  precision10.append(p10)

    print(f"Fold {fold}: ρ={rho:.3f}, Recall@5={r5:.3f}, Precision@5={p5:.3f}, "
          f"Recall@10={r10:.3f}, Precision@10={p10:.3f}")

# — summary across folds —————————————————————————————
print("\nOverall (mean ± std over folds):")
print(f"  Spearman’s ρ  : {np.mean(spearman_scores):.3f} ± {np.std(spearman_scores):.3f}")
print(f"  Recall@5      : {np.mean(recall5):.3f} ± {np.std(recall5):.3f}")
print(f"  Precision@5   : {np.mean(precision5):.3f} ± {np.std(precision5):.3f}")
print(f"  Recall@10     : {np.mean(recall10):.3f} ± {np.std(recall10):.3f}")
print(f"  Precision@10  : {np.mean(precision10):.3f} ± {np.std(precision10):.3f}")
