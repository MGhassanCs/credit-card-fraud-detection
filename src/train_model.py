"""
train_model.py
--------------
Trains the final production model pipeline (preprocessing + classifier) using the best hyperparameters found in model selection. Saves the trained pipeline for inference and deployment.
"""
import joblib
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# Load preprocessed data and transformer
split_data_path = "data/processed/split_data.pkl"
ct_path = "data/processed/amount_scaler_ct.pkl"
X_train, X_test, y_train, y_test = joblib.load(split_data_path)
ct = joblib.load(ct_path)

# Best hyperparameters found from model_selector.py
best_params = {
    "n_estimators": 50,
    "max_depth": 10,
    "random_state": 42
}
# Only keep valid RandomForestClassifier params
rf_valid_keys = {"n_estimators", "max_depth", "random_state", "min_samples_split", "min_samples_leaf", "max_features", "criterion", "bootstrap", "oob_score", "warm_start", "class_weight"}
best_params = {k: v for k, v in best_params.items() if k in rf_valid_keys}

def train_final_pipeline(X_train, y_train, ct, best_params):
    """
    Train a pipeline with preprocessing and RandomForest classifier.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        ct (ColumnTransformer): Preprocessing transformer.
        best_params (dict): Best hyperparameters for RandomForest.

    Returns:
        Pipeline: Trained pipeline.
    """
    pipeline = Pipeline([
        ("preprocessor", ct),
        ("classifier", RandomForestClassifier(**best_params))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

try:
    pipeline = train_final_pipeline(X_train, y_train, ct, best_params)
    logging.info("RandomForest pipeline trained.")

    # Save the trained pipeline
    os.makedirs("models", exist_ok=True)
    pipeline_path = "models/randomforest_pipeline.pkl"
    joblib.dump(pipeline, pipeline_path)
    logging.info(f"Pipeline saved to {pipeline_path}")
except Exception as e:
    logging.error(f"Training pipeline failed: {e}")
