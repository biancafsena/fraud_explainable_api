from enum import Enum
from pydantic import BaseModel, Field, validator

class Channel(str, Enum):
    PIX = "PIX"
    CARD = "CARD"

class Transaction(BaseModel):
    amount: float = Field(ge=0, description="Valor da transação")
    channel: Channel
    hour: int = Field(ge=0, le=23)
    is_new_device: int = Field(ge=0, le=1)
    device_trust_score: float = Field(ge=0, le=1)
    days_since_last_tx: float = Field(ge=0)
    tx_velocity_1h: int = Field(ge=0)
    merchant_risk_score: float = Field(ge=0, le=1)
    customer_age: int = Field(ge=18, le=120)
    has_chargeback_history: int = Field(ge=0, le=1)
    country_risk_score: float = Field(ge=0, le=1)

class PredictResponse(BaseModel):
    is_fraud: int
    proba: float
    model_version: str = "1.0.0"

class ExplanationItem(BaseModel):
    feature: str
    value: float | int | str
    shap_value: float | None = None
    weight: float | None = None
    impact: str | None = None  # increase/decrease

class ExplanationResponse(BaseModel):
    top_k: int
    items: list[ExplanationItem]
