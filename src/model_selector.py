"""
model_selector.py
----------------
Performs model selection and hyperparameter tuning for multiple classifiers (Logistic Regression, Random Forest, XGBoost, LightGBM) on the credit card fraud dataset. Saves the best model and its parameters for downstream use.
"""
import pandas as pd
import joblib
import os
import warnings
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score

# Suppress LightGBM split warnings for cleaner output
warnings.filterwarnings("ignore", message="No further splits with positive gain")

# Load preprocessed training and test data
data_path = "data/processed/split_data.pkl"
X_train, X_test, y_train, y_test = joblib.load(data_path)

# Ensure output directories exist
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Define models and their hyperparameter grids for tuning
models = {
    "LogisticRegression": {
        "model": LogisticRegression(max_iter=1000, random_state=42),
        "params": {
            "C": [0.1, 1, 10],
            "solver": ["liblinear"]
        }
    },
    "RandomForest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": [50],
            "max_depth": [10]
        }
    },
    "XGBoost": {
        "model": XGBClassifier(eval_metric='logloss', random_state=42),
        "params": {
            "n_estimators": [50, 100],
            "max_depth": [3, 5],
            "learning_rate": [0.01, 0.1]
        }
    },
    "LightGBM": {
        "model": LGBMClassifier(is_unbalance=True, random_state=42, verbose=-1),
        "params": {
            "n_estimators": [50, 100],
            "max_depth": [5, 10],
            "learning_rate": [0.01, 0.1],
            "min_child_samples": [5],
            "num_leaves": [31, 63]
        }
    }
}

# Store results for all models
results = []

print("\n🚀 Starting model training and tuning...")

def run_model_selection():
    """
    Run model selection and hyperparameter tuning for all defined models.
    Saves the best model and its parameters for downstream use.
    """
    best_auc = -1
    best_model_name = None
    best_model_params = None
    for name, config in models.items():
        print(f"\n🔍 Training {name}...")

        # Grid search for best hyperparameters using ROC-AUC as scoring
        grid = GridSearchCV(
            estimator=config["model"],
            param_grid=config["params"],
            scoring='roc_auc',
            cv=3,
            n_jobs=-1,
            verbose=1
        )

        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_

        # Evaluate best model on test set
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        report = classification_report(y_test, y_pred, output_dict=True)

        print(f"✅ {name} AUC-ROC: {auc:.4f}")
        print(classification_report(y_test, y_pred))

        # Save the trained model to disk
        model_path = f"models/{name.lower()}_model.pkl"
        joblib.dump(best_model, model_path)

        # Log key metrics and best parameters
        report_dict = report if isinstance(report, dict) else eval(report)
        results.append({
            "model": name,
            "best_params": grid.best_params_,
            "auc_roc": auc,
            "precision": report_dict.get("1", {}).get("precision", None),
            "recall": report_dict.get("1", {}).get("recall", None),
            "f1": report_dict.get("1", {}).get("f1-score", None)
        })

        if auc > best_auc:
            best_auc = auc
            best_model_name = name
            best_model_params = grid.best_params_

    # Save all results to a timestamped CSV file
    results_df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(f"results/model_metrics_{timestamp}.csv", index=False)

    # Save best model info for final training
    best_model_info = {
        "model_name": best_model_name,
        "params": best_model_params
    }
    joblib.dump(best_model_info, "models/best_model_info.pkl")
    print(f"\n🏆 Best model: {best_model_name} (AUC: {best_auc:.4f})")
    print("Best model info saved to models/best_model_info.pkl")

    print("\n✅ All models trained and saved. Metrics logged successfully.")

if __name__ == "__main__":
    run_model_selection()
