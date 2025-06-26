import pytest
import pandas as pd
from src.data_preprocessing import load_data, preprocess_data, split_and_balance

def test_preprocessing_pipeline():
    # Use a small sample of the dataset for testing
    df = load_data('data/creditcard.csv').head(100)
    X, y, scaler = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_and_balance(X, y)
    # Check shapes
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    # Check that scaler is fitted
    assert hasattr(scaler, 'mean_')
    # Check that no NaNs are present
    assert not X_train.isnull().any().any()
    assert not X_test.isnull().any().any() 