"""
data_preprocessing.py
---------------------
Preprocesses the credit card fraud dataset: loads data, scales the 'Amount' feature, splits into train/test, and applies SMOTE for class balancing. Saves processed data and transformer for downstream modeling.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
import joblib
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def load_data(path):
    """
    Load the credit card dataset from a CSV file.

    Args:
        path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    try:
        df = pd.read_csv(path)
        logging.info(f"Data shape: {df.shape}")
        logging.info(f"Fraudulent transactions: {df[df['Class'] == 1].shape[0]}")
        return df
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        raise

def preprocess_data(df):
    """
    Prepare features and target for modeling. Drops 'Time', scales 'Amount' using a ColumnTransformer.

    Args:
        df (pd.DataFrame): Raw dataset.

    Returns:
        X (pd.DataFrame): Preprocessed features.
        y (pd.Series): Target labels.
        ct (ColumnTransformer): Fitted transformer for scaling.
    """
    try:
        df = df.drop('Time', axis=1)
        X = df.drop('Class', axis=1)
        y = df['Class']
        # Only scale 'Amount' for modeling
        ct = ColumnTransformer(
            transformers=[
                ('amount_scaler', StandardScaler(), ['Amount'])
            ],
            remainder='passthrough',
            verbose_feature_names_out=False
        )
        X_transformed = pd.DataFrame(ct.fit_transform(X), columns=ct.get_feature_names_out(X.columns), index=X.index)
        logging.info("Data preprocessing complete with ColumnTransformer.")
        return X_transformed, y, ct
    except Exception as e:
        logging.error(f"Preprocessing failed: {e}")
        raise

def split_and_balance(X, y):
    """
    Split data into train/test sets and balance the training set using SMOTE.

    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Target labels.

    Returns:
        X_resampled (pd.DataFrame): Balanced training features.
        X_test (pd.DataFrame): Test features.
        y_resampled (pd.Series): Balanced training labels.
        y_test (pd.Series): Test labels.
    """
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        logging.info(f"Before SMOTE: {y_train.sum()} fraud / {len(y_train)} total")
        logging.info(f"After SMOTE: {y_resampled.sum()} fraud / {len(y_resampled)} total")
        return X_resampled, X_test, y_resampled, y_test
    except Exception as e:
        logging.error(f"Split and balance failed: {e}")
        raise

if __name__ == "__main__":
    DATA_PATH = "data/creditcard.csv"
    try:
        df = load_data(DATA_PATH)
        X, y, ct = preprocess_data(df)
        X_train, X_test, y_train, y_test = split_and_balance(X, y)
        os.makedirs("data/processed", exist_ok=True)
        joblib.dump((X_train, X_test, y_train, y_test), "data/processed/split_data.pkl")
        joblib.dump(ct, "data/processed/amount_scaler_ct.pkl")
        logging.info("✅ Data preprocessing complete and saved.")
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
