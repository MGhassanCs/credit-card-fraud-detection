import joblib
import numpy as np
import pytest

def test_model_inference():
    # Load model and test data
    model = joblib.load('models/randomforest_model_final.pkl')
    _, X_test, _, _ = joblib.load('data/processed/split_data.pkl')
    # Use a small sample
    X_sample = X_test[:5]
    # Predict
    preds = model.predict(X_sample)
    # Check output shape and type
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == X_sample.shape[0]
    # Check that predictions are 0 or 1
    assert set(preds).issubset({0, 1}) 