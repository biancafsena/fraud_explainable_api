from fastapi import FastAPI
from app.schemas import Transaction, PredictResponse, ExplanationResponse
from app import model as mdl

app = FastAPI(title="Fraud Scoring API — Explainable AI", version="1.0.0")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {"message": "Fraud Scoring API — use /docs para explorar."}

@app.post("/predict", response_model=PredictResponse)
def predict(tx: Transaction):
    label, proba = mdl.predict(tx.model_dump())
    return {"is_fraud": label, "proba": round(proba, 6), "model_version": "1.0.0"}

@app.post("/explain/shap", response_model=ExplanationResponse)
def explain_shap(tx: Transaction, top_k: int = 5):
    items = mdl.shap_explain(tx.model_dump(), top_k=top_k)
    return {"top_k": top_k, "items": items}

@app.post("/explain/lime", response_model=ExplanationResponse)
def explain_lime(tx: Transaction, top_k: int = 5):
    items = mdl.lime_explain(tx.model_dump(), top_k=top_k)
    return {"top_k": top_k, "items": items}
