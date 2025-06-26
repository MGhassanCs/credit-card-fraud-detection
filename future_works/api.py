from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import logging
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# Initialize FastAPI app for serving predictions
app = FastAPI(title="Credit Card Fraud Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or ["http://localhost:3000"] for more security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load pipeline once at startup for efficiency
try:
    pipeline = joblib.load("models/randomforest_pipeline.pkl")
    logging.info("RandomForest pipeline loaded successfully.")
except Exception as e:
    pipeline = None
    logging.error(f"Pipeline could not be loaded: {e}")

# Define request schema for transaction data using Pydantic
class Transaction(BaseModel):
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

class PredictionResponse(BaseModel):
    fraud_probability: float

@app.get("/", response_model=dict)
async def root():
    """Health check endpoint."""
    logging.info("Health check endpoint called.")
    return {"message": "Credit Card Fraud Detection API is running"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(transaction: Transaction):
    """
    Predict the probability of a transaction being fraudulent.
    - **transaction**: Transaction features (V1-V28, Amount)
    - **returns**: Probability of fraud (0.0-1.0)
    """
    if pipeline is None:
        logging.error("Pipeline not loaded.")
        raise HTTPException(status_code=503, detail="Pipeline not loaded.")
    try:
        # Prepare input as a DataFrame with correct column names
        feature_names = [
            "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
            "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
            "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"
        ]
        data = pd.DataFrame(
            [[
                transaction.V1, transaction.V2, transaction.V3, transaction.V4,
                transaction.V5, transaction.V6, transaction.V7, transaction.V8,
                transaction.V9, transaction.V10, transaction.V11, transaction.V12,
                transaction.V13, transaction.V14, transaction.V15, transaction.V16,
                transaction.V17, transaction.V18, transaction.V19, transaction.V20,
                transaction.V21, transaction.V22, transaction.V23, transaction.V24,
                transaction.V25, transaction.V26, transaction.V27, transaction.V28,
                transaction.Amount
            ]],
            columns=pd.Index(feature_names)
        )
        # Predict probability of fraud (class 1) using the pipeline
        proba = pipeline.predict_proba(data)[0, 1]
        logging.info(f"Prediction made. Fraud probability: {proba:.4f}")
        return {"fraud_probability": proba}
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")
