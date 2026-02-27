import numpy as np
from joblib import load
from fastapi import FastAPI, HTTPException
from src.schemas import LoanInput

app = FastAPI(title="Loan Approval Prediction API")

# Load model when server starts
try:
    model = load("models/model.joblib")
except Exception:
    model = None

@app.get("/")
def home():
    return {"message": "Loan Approval API is running"}

@app.get("/health")
def health():
    return {"status": "ok" if model is not None else "model_not_loaded"}

@app.post("/predict-loan")
def predict_loan(data: LoanInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Run: python src/train.py")

    features = np.array([[
        data.applicant_income,
        data.coapplicant_income,
        data.loan_amount,
        data.loan_term,
        data.credit_history
    ]], dtype=float)

    pred = int(model.predict(features)[0])

    return {
        "prediction": pred,
        "meaning": "Loan Approved" if pred == 1 else "Loan Rejected"
    }