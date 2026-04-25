from pydantic import BaseModel, Field
from typing import Optional
from typing import List

class TransactionRequest(BaseModel):
    TransactionAmt: float = Field(..., gt=0, description="Transaction amount in USD")
    addr1: Optional[float] = None
    addr2: Optional[float] = None
    dist1: Optional[float] = None
    dist2: Optional[float] = None
    C1: Optional[float] = None
    C2: Optional[float] = None
    C3: Optional[float] = None
    C4: Optional[float] = None
    C5: Optional[float] = None
    C6: Optional[float] = None
    C7: Optional[float] = None
    C8: Optional[float] = None
    C9: Optional[float] = None
    C10: Optional[float] = None
    C11: Optional[float] = None
    D1: Optional[float] = None
    D2: Optional[float] = None
    D3: Optional[float] = None
    D4: Optional[float] = None
    D5: Optional[float] = None
    D10: Optional[float] = None
    D11: Optional[float] = None
    D15: Optional[float] = None
    ProductCD: Optional[str] = "W"
    card4: Optional[str] = "visa"
    card6: Optional[str] = "debit"
    P_emaildomain: Optional[str] = "gmail.com"
    R_emaildomain: Optional[str] = "gmail.com"
    M4: Optional[str] = "M0"
    M5: Optional[str] = "T"
    M6: Optional[str] = "T"
    DeviceType: Optional[str] = "desktop"

    class Config:
        json_schema_extra = {
            "example": {
                "TransactionAmt": 500.00,
                "C1": 3.0,
                "C2": 1.0,
                "D1": 30.0,
                "ProductCD": "W",
                "card4": "visa",
                "card6": "debit",
                "P_emaildomain": "gmail.com",
                "R_emaildomain": "yahoo.com",
                "DeviceType": "mobile"
            }
        }


class PredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    risk_level: str
    model_round: int
    threshold_used: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_round: int
    input_dim: int


class ModelInfoResponse(BaseModel):
    model_round: int
    input_dim: int
    auc: float
    f1: float
    precision: float
    recall: float

class FeatureAttribution(BaseModel):
    feature: str
    shap_value: float
    direction: str
    raw_value: float


class ExplanationResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    risk_level: str
    model_round: int
    threshold_used: float
    explanation: dict
    english_explanation: str


class PredictionWithExplanationResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    risk_level: str
    risk_level: str
    model_round: int
    threshold_used: float
    top_features: List[FeatureAttribution]
    english_explanation: str
    total_fraud_push: float
    total_fraud_pull: float