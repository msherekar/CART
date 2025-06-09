# tests/test_prediction.py
import numpy as np
import pytest
from pathlib import Path
import tempfile
from CART.src.modeling.prediction import run_prediction

@pytest.fixture
def test_data(tmp_path):
    """Create test data for prediction pipeline"""
    # Create dummy embedding file
    X = np.random.randn(50, 10)
    embedding_path = tmp_path / "test_embeddings.npy"
    np.save(embedding_path, X)
    
    # Create dummy labels file
    y = np.random.randn(50)
    labels_path = tmp_path / "test_labels.csv"
    np.savetxt(labels_path, y, delimiter=",")
    
    # Create output directory
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    return {
        'embedding_path': embedding_path,
        'labels_path': labels_path,
        'output_dir': output_dir
    }

def test_prediction_pipeline(test_data):
    """Test the prediction pipeline with dummy data"""
    # Create Args object with test data
    args = type('Args', (), {
        'embedding_dir': str(test_data['output_dir']),
        'embedding_files': [str(test_data['embedding_path'])],
        'labels_path': str(test_data['labels_path']),
        'output_dir': str(test_data['output_dir']),
        'n_splits': 5,
        'n_inner_splits': 3,
        'random_seed': 42,
        'model_name': 'test_model',
        'device': 'cpu'
    })
    
    # Run prediction
    run_prediction(args)
    
    # Check if output files were created with correct names
    assert (test_data['output_dir'] / "test_embeddings_correlations.npy").exists()
    assert (test_data['output_dir'] / "test_embeddings_predictions.npz").exists()
    assert (test_data['output_dir'] / "prediction_summary.txt").exists()
