# 🛡️ Credit Card Fraud Detection

## Overview
This project implements a robust, end-to-end pipeline for detecting credit card fraud using both supervised and unsupervised machine learning methods. It includes data preprocessing, model selection, training, evaluation, explainability, and advanced hybrid modeling. The codebase is modular, well-documented, and ready for extension or deployment.

---

## Dataset
- **Source:** [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Features:**
  - `Time`: Seconds elapsed between each transaction and the first transaction.
  - `V1`–`V28`: Principal components from PCA (anonymized, original meanings not public).
  - `Amount`: Transaction amount.
  - `Class`: Target variable (1 = Fraud, 0 = Legitimate).

---

## Problem Statement & Goals
- **Goal:** Detect fraudulent credit card transactions with high recall and reasonable precision.
- **Approach:**
  - Use supervised models (Random Forest, Logistic Regression, XGBoost, LightGBM) for classification.
  - Use unsupervised anomaly detection (Isolation Forest) to flag outliers.
  - Combine anomaly scores with supervised models for improved detection.

---

## Setup & Installation

```bash
git clone <repo-url>
cd credit-card-fraud-detection
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## How to Run

### 1. **Data Preprocessing**
```bash
python src/data_preprocessing.py
```

### 2. **Model Selection & Training**
```bash
python src/model_selector.py
python src/train_model.py
```

### 3. **Evaluation & Explainability**
```bash
python src/evaluate.py
python src/explain_model.py
```

### 4. **Anomaly Detection**
```bash
python src/anomaly_detection.py
```

### 5. **Hybrid Model (Classifier + Anomaly Score)**
```bash
python src/rf_with_anomaly_feature.py
```

---

## Folder Structure
```
credit-card-fraud-detection/
├── data/                # Raw and processed data
│   └── processed/
├── models/              # Saved models and pipelines
├── notebooks/           # Jupyter notebooks for EDA, experiments
├── src/                 # Source code (all .py files)
│   ├── data_preprocessing.py
│   ├── model_selector.py
│   ├── train_model.py
│   ├── evaluate.py
│   ├── explain_model.py
│   ├── anomaly_detection.py
│   ├── rf_with_anomaly_feature.py
│   └── plot_curves.py
├── tests/               # Unit and integration tests
├── logs/                # Log files
├── results/             # Evaluation results, reports
├── plots/               # Plots and visualizations
├── requirements.txt
├── README.md
└── future_works/   # Features not currently in use (API, frontend)
```

---

## Methods Used
- **Supervised Models:** Random Forest, Logistic Regression, XGBoost, LightGBM
- **Unsupervised Model:** Isolation Forest (anomaly detection)
- **Hybrid:** Random Forest with anomaly score as an additional feature
- **Explainability:** SHAP for feature importance
- **Evaluation:** ROC-AUC, Precision, Recall, F1, Confusion Matrix, PR/ROC curves

---

## Interpreting Results
- **Classification Report:** Shows precision, recall, F1-score for each class.
- **ROC-AUC:** Measures overall model discrimination.
- **Confusion Matrix:** Shows true/false positives/negatives.
- **SHAP Plots:** Visualize feature importance and model behavior.

---

## Next Steps / Future Improvements
- Deploy API and frontend for real-time fraud detection (see `future_works.md`).
- Integrate with MLflow for experiment tracking.
- Add more advanced anomaly detection methods.
- Automate retraining with new data.
- Explore deep learning models.

---

## References
- [Kaggle Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [Papers: Credit Card Fraud Detection, Anomaly Detection]

---

## Note
The API and frontend are not currently in use. For more information, see `future_works.md`.
