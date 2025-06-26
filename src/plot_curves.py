import joblib
import os
from evaluate import plot_model_curves  # Assumes plot_model_curves is defined in evaluate.py

# Load test data for evaluation
# Only test set is needed for plotting model performance

data_path = "data/processed/split_data.pkl"
_, X_test, _, y_test = joblib.load(data_path)

# List of model names to evaluate
model_names = ["logisticregression", "randomforest", "xgboost", "lightgbm"]
model_preds = []
loaded_models = []

# Load each trained model and generate probability predictions
for name in model_names:
    model = joblib.load(f"models/{name}_model.pkl")
    loaded_models.append(model)
    y_proba = model.predict_proba(X_test)[:, 1]
    model_preds.append(y_proba)

# Ensure the plots directory exists
os.makedirs("plots", exist_ok=True)

# Plot ROC and Precision-Recall curves for all models
plot_model_curves(y_test, model_preds, model_names, save_dir="plots")
