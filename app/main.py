from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np

bundle = joblib.load("model.joblib")
model = bundle["model"]
feature_names = bundle["feature_names"]
n_features = len(feature_names)

class PredictRequest(BaseModel):
    # List of rows, each row = list of floats with length n_features
    instances: List[List[float]]

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok", "features": feature_names}

@app.post("/predict")
def predict(req: PredictRequest):
    X = req.instances
    if any(len(row) != n_features for row in X):
        raise HTTPException(status_code=400, detail=f"Each row must have {n_features} features: {feature_names}")
    preds = model.predict(np.array(X)).tolist()
    return {"predictions": preds}
