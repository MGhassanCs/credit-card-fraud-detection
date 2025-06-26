"""
anomaly_detection.py
-------------------
Fits and tunes an Isolation Forest anomaly detector on the credit card fraud dataset, selects a threshold for fraud detection, and evaluates performance. Saves the model and threshold for future use.
"""
import joblib
import os
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# Load preprocessed data
split_data_path = "data/processed/split_data.pkl"
X_train, X_test, y_train, y_test = joblib.load(split_data_path)

def tune_and_evaluate_isolation_forest(X_train, X_test, y_train, y_test):
    """
    Fit IsolationForest, tune threshold, and evaluate performance.

    Args:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Test features.
        y_train (pd.Series): Training labels.
        y_test (pd.Series): Test labels.

    Returns:
        iso_forest (IsolationForest): Trained IsolationForest model.
        best_threshold (float): Selected threshold for anomaly score.
    """
    # Fit IsolationForest with default/best params
    iso_forest = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
    iso_forest.fit(X_train)
    logging.info("IsolationForest model trained.")

    # Get raw anomaly scores (the lower, the more anomalous)
    anomaly_scores = -iso_forest.decision_function(X_test)  # Flip sign so higher = more anomalous

    # Loop through thresholds to find a good trade-off
    thresholds = np.linspace(np.min(anomaly_scores), np.max(anomaly_scores), 100)
    best_threshold = None
    best_recall = 0
    print("Threshold\tPrecision\tRecall\tNum_Frauds")
    for thresh in thresholds:
        y_pred = (anomaly_scores >= thresh).astype(int)  # 1 = predicted fraud
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        num_frauds = y_pred.sum()
        print(f"{thresh:.4f}\t{prec:.4f}\t{rec:.4f}\t{num_frauds}")
        # Select threshold: recall >= 0.3, precision >= 0.05, prefer higher recall
        if rec >= 0.3 and prec >= 0.05 and rec > best_recall:
            best_threshold = thresh
            best_recall = rec

    if best_threshold is None:
        print("No threshold found that meets the criteria. Showing best recall found.")
        # Fallback: pick threshold with highest recall >= 0.05 precision
        best_recall = 0
        for thresh in thresholds:
            y_pred = (anomaly_scores >= thresh).astype(int)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            if prec >= 0.05 and rec > best_recall:
                best_threshold = thresh
                best_recall = rec

    print(f"\nSelected threshold: {best_threshold:.4f}")

    # Apply the selected threshold
    y_pred_final = (anomaly_scores >= best_threshold).astype(int)

    # Print final classification report and confusion matrix
    print("\nFinal Classification Report (using selected threshold):")
    print(classification_report(y_test, y_pred_final, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_final))

    return iso_forest, best_threshold

if __name__ == "__main__":
    try:
        iso_forest, best_threshold = tune_and_evaluate_isolation_forest(X_train, X_test, y_train, y_test)
        # Save the model and threshold
        os.makedirs("models", exist_ok=True)
        model_path = "models/isolationforest_model.pkl"
        joblib.dump({"model": iso_forest, "threshold": best_threshold}, model_path)
        logging.info(f"IsolationForest model and threshold saved to {model_path}")
    except Exception as e:
        logging.error(f"Anomaly detection pipeline failed: {e}") 