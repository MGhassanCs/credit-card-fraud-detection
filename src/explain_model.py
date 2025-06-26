"""
explain_model.py
----------------
Generates SHAP explanations for the trained model to interpret feature importance and visualize model behavior on the credit card fraud dataset.
"""
import joblib
import shap
import matplotlib.pyplot as plt
import os
import numpy as np

# Load model and test data
model = joblib.load('models/randomforest_model_final.pkl')
_, X_test, _, _ = joblib.load('data/processed/split_data.pkl')

# Use a sample for speed
X_sample = X_test[:100]

# Create SHAP explainer and compute SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

# Debug print shapes and types
print('shap_values type:', type(shap_values))
if isinstance(shap_values, list):
    print('shap_values[1] shape:', shap_values[1].shape)
elif isinstance(shap_values, np.ndarray):
    print('shap_values shape:', shap_values.shape)
print('X_sample shape:', X_sample.shape)

# Ensure plots directory exists
os.makedirs('plots', exist_ok=True)

# Robustly select the correct SHAP values for binary classification
if isinstance(shap_values, list):
    # Old SHAP: list of arrays, use class 1
    shap_to_plot = shap_values[1]
elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
    # New SHAP: 3D array (samples, features, classes)
    shap_to_plot = shap_values[:, :, 1]
else:
    # Fallback: use as is
    shap_to_plot = shap_values

# Plot and save SHAP summary plot for the positive class (fraud)
def plot_shap_summary(shap_values, X_sample):
    """
    Plot and save a SHAP summary plot for the given SHAP values and feature sample.

    Args:
        shap_values (np.ndarray): SHAP values for the positive class.
        X_sample (pd.DataFrame): Feature sample used for SHAP computation.
    """
    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig('plots/shap_summary_rf.png')
    plt.close()
    print('SHAP summary plot saved to plots/shap_summary_rf.png')

plot_shap_summary(shap_to_plot, X_sample) 