from fastapi import FastAPI
import joblib
import pandas as pd
from api.m import CustomerFeatures, PredictionResponse

app = FastAPI(title="Olist CLV & Churn API")

churn_model = joblib.load("src/churn_model.pkl")
clv_model = joblib.load("src/clv_model.pkl")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict(features: CustomerFeatures):
    X = pd.DataFrame([features.dict()])
    churn_prob = 1 - churn_model.predict_proba(X)[0][1]
    predicted_clv = clv_model.predict(X)[0]

    if churn_prob > 0.7 and predicted_clv > 200:
        action = "High-value at risk — send retention offer"
    elif churn_prob > 0.7:
        action = "Low priority — likely one-time buyer"
    else:
        action = "Healthy — no action needed"

    return PredictionResponse(
        churn_risk=round(float(churn_prob), 3),
        predicted_clv=round(float(predicted_clv), 2),
        recommended_action=action
    )