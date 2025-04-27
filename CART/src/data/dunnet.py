import numpy as np
from scipy.stats import dunnett

# --- replace these with your actual 5 Spearman ρs per split ---
pretrained_rho = np.array([0.45, 0.52, 0.48, 0.50, 0.47])
finetuned_rho  = np.array([0.60, 0.65, 0.63, 0.64, 0.62])

# Stack into shape (n_splits, n_groups)
#   group 0 = pre-trained (control)
#   group 1 = fine-tuned (treatment)
data = np.vstack([pretrained_rho, finetuned_rho]).T

# Run Dunnett’s test, control index = 0
result = dunnett(data, control=0)

# Inspect results
print("Comparisons:", result.comparisons)   # e.g. [(1 vs. 0)]
print("Statistic :", result.statistic)     # test statistic(s)
print("p-values   :", result.pvalue)       # adjusted p-value(s)


#Notes:

    #data must be an array of shape (n_observations, n_groups).

    #control=0 tells SciPy that column 0 (your pre-trained group) is the reference.

    #result typically exposes .comparisons, .statistic, and .pvalue, which you can print or log.