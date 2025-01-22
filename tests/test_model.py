import pytest
import pandas as pd
import numpy as np
from src.model import SalesPredictor

@pytest.fixture
def sample_data():
    X = pd.DataFrame({
        'Feature1': np.random.rand(100),
        'Feature2': np.random.rand(100)
    })
    y = np.random.rand(100)
    return X, y

def test_model_training(sample_data):
    X, y = sample_data
    predictor = SalesPredictor()
    training_info = predictor.train(X, y)
    
    assert predictor.model is not None
    assert 'feature_importance' in training_info
    assert len(training_info['feature_importance']) == X.shape[1]

def test_model_prediction(sample_data):
    X, y = sample_data
    predictor = SalesPredictor()
    predictor.train(X, y)
    predictions = predictor.predict(X)
    
    assert len(predictions) == len(y)
    assert isinstance(predictions, np.ndarray)