"""
rf_with_anomaly_feature.py
-------------------------
Retrains the best classifier (e.g., Random Forest) using the anomaly score from Isolation Forest as an additional feature. Evaluates and compares performance to the original model.
"""
import joblib
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def add_anomaly_score_feature(X_train, X_test, iso_forest):
    """
    Compute raw anomaly scores and add as a new feature to X_train and X_test.

    Args:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Test features.
        iso_forest (IsolationForest): Trained anomaly detector.

    Returns:
        X_train_new (pd.DataFrame): Training features with anomaly score.
        X_test_new (pd.DataFrame): Test features with anomaly score.
    """
    X_train_new = X_train.copy()
    X_test_new = X_test.copy()
    X_train_new["anomaly_score"] = -iso_forest.score_samples(X_train)
    X_test_new["anomaly_score"] = -iso_forest.score_samples(X_test)
    return X_train_new, X_test_new

def retrain_and_evaluate_with_anomaly(X_train, X_test, y_train, y_test, best_model_info):
    """
    Retrain the best model with the anomaly score feature and evaluate performance.

    Args:
        X_train (pd.DataFrame): Training features with anomaly score.
        X_test (pd.DataFrame): Test features with anomaly score.
        y_train (pd.Series): Training labels.
        y_test (pd.Series): Test labels.
        best_model_info (dict): Info about the best model and its parameters.

    Returns:
        model: Trained model with anomaly feature.
    """
    model_map = {
        "RandomForest": RandomForestClassifier,
        "LogisticRegression": LogisticRegression,
        "XGBoost": XGBClassifier,
        "LightGBM": LGBMClassifier
    }
    model_name = best_model_info["model_name"]
    model_params = best_model_info["params"]
    ModelClass = model_map[model_name]
    model = ModelClass(**model_params)
    model.fit(X_train, y_train)
    logging.info(f"{model_name} retrained with anomaly score feature.")

    # Evaluate the updated model
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    print("\nClassification Report (with anomaly score feature):")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    if y_proba is not None:
        auc = roc_auc_score(y_test, y_proba)
        print(f"AUC-ROC: {auc:.4f}")
    return model

def compare_to_previous_rf(X_test, y_test):
    """
    Compare the new model to the previous Random Forest model (without anomaly feature).

    Args:
        X_test (pd.DataFrame): Test features with anomaly score.
        y_test (pd.Series): Test labels.
    """
    prev_rf_path = "models/randomforest_model_final.pkl"
    if os.path.exists(prev_rf_path):
        prev_rf = joblib.load(prev_rf_path)
        # Get the original feature names (all columns except 'anomaly_score')
        orig_features = [col for col in X_test.columns if col != "anomaly_score"]
        X_test_no_anom = X_test[orig_features]  # Ensures correct order and only original features
        y_pred_prev = prev_rf.predict(X_test_no_anom)
        y_proba_prev = prev_rf.predict_proba(X_test_no_anom)[:, 1]
        print("\nPrevious Random Forest Model (no anomaly feature):")
        print(classification_report(y_test, y_pred_prev, digits=4))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred_prev))
        auc_prev = roc_auc_score(y_test, y_proba_prev)
        print(f"AUC-ROC: {auc_prev:.4f}")

if __name__ == "__main__":
    # 1. Load preprocessed data and anomaly detection model
    split_data_path = "data/processed/split_data.pkl"
    iso_path = "models/isolationforest_model.pkl"
    best_model_info_path = "models/best_model_info.pkl"
    X_train, X_test, y_train, y_test = joblib.load(split_data_path)
    iso_obj = joblib.load(iso_path)
    iso_forest = iso_obj["model"]
    best_model_info = joblib.load(best_model_info_path)

    # 2. Compute and add anomaly score feature
    X_train_new, X_test_new = add_anomaly_score_feature(X_train, X_test, iso_forest)

    # 3. Retrain and evaluate the best model with anomaly feature
    model = retrain_and_evaluate_with_anomaly(X_train_new, X_test_new, y_train, y_test, best_model_info)

    # 4. Compare to previous Random Forest model (if available)
    compare_to_previous_rf(X_test_new, y_test)

    # 5. Save the new model
    os.makedirs("models", exist_ok=True)
    model_path = f"models/{best_model_info['model_name'].lower()}_with_anomaly_feature.pkl"
    joblib.dump(model, model_path)
    logging.info(f"{best_model_info['model_name']} with anomaly feature saved to {model_path}") 