import os, json, joblib, pandas as pd
from model.train import train, ensure_data, MODEL_DIR

def test_train_produces_artifacts():
    metrics = train()
    assert "roc_auc" in metrics and metrics["roc_auc"] > 0.70
    assert os.path.exists(os.path.join(MODEL_DIR, "model.pkl"))
    assert os.path.exists(os.path.join(MODEL_DIR, "feature_names.json"))

def test_predict_one():
    pipe = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
    sample = {
        "amount": 1299.9,
        "channel": "PIX",
        "hour": 1,
        "is_new_device": 1,
        "device_trust_score": 0.23,
        "days_since_last_tx": 0.5,
        "tx_velocity_1h": 7,
        "merchant_risk_score": 0.88,
        "customer_age": 28,
        "has_chargeback_history": 1,
        "country_risk_score": 0.67,
    }
    df = pd.DataFrame([sample])
    proba = pipe.predict_proba(df)[:,1][0]
    assert 0.0 <= proba <= 1.0
