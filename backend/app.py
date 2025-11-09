from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import numpy as np
import joblib
import os

app = FastAPI(title="Food Access Prediction API")

# Load pipelines/models for each algorithm
models = {
    "logistic": {
        "imputer": joblib.load("models/imputer_logistic.pkl"),
        "scaler":  joblib.load("models/scaler_logistic.pkl"),
        "model":   joblib.load("models/model_logistic.pkl")
    },
    "rf": {
        "imputer": joblib.load("models/imputer_rf.pkl"),
        "scaler":  joblib.load("models/scaler_rf.pkl"),
        "model":   joblib.load("models/model_rf.pkl")
    },
    "xgb": {
        "imputer": joblib.load("models/imputer_xgb.pkl"),
        "scaler":  joblib.load("models/scaler_xgb.pkl"),
        "model":   joblib.load("models/model_xgb.pkl")
    },
}

class Features3(BaseModel):
    HUNVFlag:    float
    PovertyRate: float
    LA1and10:    float

class PredictionResponse(BaseModel):
    predicted_class: int
    probability:     float
    model_used:      str

@app.post("/predict/simple", response_model=PredictionResponse)
def predict_simple(
    data: Features3,
    model_type: str = Query("logistic", enum=["logistic","rf","xgb"])
):
    if model_type not in models:
        raise HTTPException(status_code=400, detail=f"Unknown model_type: {model_type}")
    pipeline = models[model_type]
    try:
        X_new = np.array([[data.HUNVFlag, data.PovertyRate, data.LA1and10]])
        X_imp = pipeline["imputer"].transform(X_new)
        X_scaled = pipeline["scaler"].transform(X_imp)
        prob = pipeline["model"].predict_proba(X_scaled)[0][1]
        pred = int(pipeline["model"].predict(X_scaled)[0])
        return PredictionResponse(predicted_class=pred, probability=prob, model_used=model_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
