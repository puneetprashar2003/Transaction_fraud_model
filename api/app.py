from fastapi import FastAPI
import joblib
import pandas as pd
import uvicorn
from pydantic import BaseModel
import xgboost as xgb
import numpy as np  

model = joblib.load("xgboost_fraud_model.pkl")  
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")  


type_mapping = {"PAYMENT": 0, "TRANSFER": 1, "CASH_OUT": 2, "DEBIT": 3, "CASH_IN": 4}


app = FastAPI(title="Fraud Detection API", version="1.0")

class Transaction(BaseModel):
    step: int
    type: str
    amount: float
    oldbalanceSender: float
    newbalanceSender: float
    oldbalanceReceiver: float
    newbalanceReceiver: float
    isFlaggedFraud: int
    prev_transactions_sender: int
    prev_receives_receiver: int



def predict_fraud(transaction_data: dict):
    df = pd.DataFrame([transaction_data])

   
    df["type"] = df["type"].map(type_mapping)

    df["balance_change_sender"] = df["newbalanceSender"] - df["oldbalanceSender"]
    df["balance_change_receiver"] = df["newbalanceReceiver"] - df["oldbalanceReceiver"]
    df["mismatch_flag"] = (df["balance_change_sender"] != -1 * df["balance_change_receiver"])


    missing_features = set(feature_names) - set(df.columns)
    if missing_features:
        return {"error": f"Missing features: {missing_features}"}

    X_test = df[feature_names]


    X_test_scaled = scaler.transform(X_test)

  
    prediction = model.predict(X_test_scaled)[0]
    probability = model.predict_proba(X_test_scaled)[0][1]

    
    return {
        "fraud_prediction": "Fraudulent" if int(prediction) == 1 else "Not Fraudulent",
        "fraud_probability": float(probability)  
    }

# API Endpoint
@app.post("/predict")
def predict(transaction: Transaction):
    result = predict_fraud(transaction.dict())
    return result

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
