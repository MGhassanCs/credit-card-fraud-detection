import joblib
import pandas as pd
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define paths for model, data, plots, and results
MODEL_PATH = "models/randomforest_model.pkl"
TEST_DATA_PATH = "data/processed/split_data.pkl"
PLOTS_DIR = "plots"
RESULTS_DIR = "results"

# Ensure output directories exist
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def main():
    # Load test data and trained model
    _, X_test, _, y_test = joblib.load(TEST_DATA_PATH)
    model = joblib.load(MODEL_PATH)

    # Generate predictions and prediction probabilities
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Compute and print classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    print(classification_report(y_test, y_pred))

    # Compute and print ROC AUC score
    auc = roc_auc_score(y_test, y_proba)
    print(f"ROC AUC: {auc:.4f}")

    # Plot and save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"{PLOTS_DIR}/confusion_matrix_final.png")
    plt.close()

    # Plot and save ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig(f"{PLOTS_DIR}/roc_curve_final.png")
    plt.close()

    # Plot and save Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label="Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.savefig(f"{PLOTS_DIR}/pr_curve_final.png")
    plt.close()

    # Save classification report to CSV
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(f"{RESULTS_DIR}/final_classification_report.csv")

    print(f"✅ Evaluation complete. Plots saved to {PLOTS_DIR}, report saved to {RESULTS_DIR}")

if __name__ == "__main__":
    main()
