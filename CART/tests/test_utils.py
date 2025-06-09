# tests/test_utils.py
import numpy as np
import pytest
from pathlib import Path
import tempfile
from CART.src.utils import recall_precision_at_k, plot_correlation, plot_spearman_whisker

def test_recall_precision_at_k():
    """Test the recall and precision at K calculation"""
    # Create test data where top 25% are positives (values >= 7.5)
    # We need to ensure the values are properly ordered for the test
    y_true = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # For perfect prediction, y_pred should be in the same order as y_true
    y_pred = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    
    # Test with K=5
    r5, p5 = recall_precision_at_k(y_true, y_pred, K=5)
    # Top 25% are [8,9,10], and top-5 preds include [6,7,8,9,10]
    # 3 of the top 5 predictions are positives
    assert r5 == 1.0  # All positives retrieved (3/3)
    assert p5 == 0.6  # 3 positives in top 5 (3/5)
    
    # Test with K=10
    r10, p10 = recall_precision_at_k(y_true, y_pred, K=10)
    assert r10 == 1.0  # All positives retrieved (3/3)
    assert p10 == 0.3  # 3 positives in 10 total (3/10)

def test_plot_functions(tmp_path):
    """Test the plotting functions"""
    # Create test data
    y_test = np.random.randn(50)
    y_pred = np.random.randn(50)
    spearman_scores = np.random.randn(5)
    
    # Create output directory
    output_dir = tmp_path / "plots"
    output_dir.mkdir()
    
    # Test correlation plot
    plot_correlation(y_test, y_pred, fold=1, model_name="test", output_dir=str(output_dir))
    assert (output_dir / "test_fold1_correlation.png").exists()
    
    # Test spearman whisker plot
    plot_spearman_whisker(spearman_scores, "test", str(output_dir))
    assert (output_dir / "test_spearman_whisker.png").exists()
