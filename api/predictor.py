import torch
import numpy as np
import pandas as pd
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../src/simulator"))

from model_loader import model_loader
from schemas import TransactionRequest


CATEGORICAL_FEATURES = [
    "ProductCD", "card4", "card6",
    "P_emaildomain", "R_emaildomain",
    "M4", "M5", "M6", "DeviceType",
]

NUMERICAL_FEATURES = [
    "TransactionAmt", "addr1", "addr2",
    "dist1", "dist2",
    "C1", "C2", "C3", "C4", "C5", "C6",
    "C7", "C8", "C9", "C10", "C11",
    "D1", "D2", "D3", "D4", "D5",
    "D10", "D11", "D15",
]


def get_risk_level(probability: float) -> str:
    if probability >= 0.8:
        return "critical"
    elif probability >= 0.6:
        return "high"
    elif probability >= 0.4:
        return "medium"
    else:
        return "low"


def request_to_dataframe(request: TransactionRequest) -> pd.DataFrame:
    """Convert a Pydantic request object to a single-row DataFrame."""
    row = {}

    for col in NUMERICAL_FEATURES:
        val = getattr(request, col, None)
        row[col] = val if val is not None else 0.0

    for col in CATEGORICAL_FEATURES:
        val = getattr(request, col, None)
        row[col] = val if val is not None else "unknown"

    return pd.DataFrame([row])


def align_features(df: pd.DataFrame) -> pd.DataFrame:
    df = pd.get_dummies(df, columns=CATEGORICAL_FEATURES,
                        prefix=CATEGORICAL_FEATURES)

    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    scaler_features = model_loader.scaler.feature_names_in_

    # Build all missing columns at once instead of inserting one by one
    missing = {col: [0] for col in scaler_features if col not in df.columns}
    if missing:
        df = pd.concat([df, pd.DataFrame(missing, index=df.index)], axis=1)

    return df[scaler_features]


def predict(request: TransactionRequest,
            threshold: float = 0.5) -> dict:
    """
    Run a single transaction through the model.
    Returns fraud probability, label, and risk level.
    """
    df = request_to_dataframe(request)
    df = align_features(df)

    X = model_loader.scaler.transform(df).astype(np.float32)
    X_tensor = torch.tensor(X, dtype=torch.float32).to(model_loader.device)

    with torch.no_grad():
        probability = model_loader.model(X_tensor).cpu().numpy()[0]

    probability = float(probability)

    return {
        "fraud_probability": round(probability, 4),
        "is_fraud": probability >= threshold,
        "risk_level": get_risk_level(probability),
        "model_round": model_loader.model_round,
        "threshold_used": threshold,
    }

def predict_with_explanation(request: TransactionRequest,
                              threshold: float = 0.5,
                              top_k: int = 10) -> dict:
    """
    Run prediction AND compute SHAP explanation in one call.
    This is what the dashboard will use.
    """
    from explainer import fraud_explainer

    df = request_to_dataframe(request)
    df = align_features(df)

    X = model_loader.scaler.transform(df).astype(np.float32)
    X_tensor = torch.tensor(
        X, dtype=torch.float32
    ).to(model_loader.device)

    with torch.no_grad():
        probability = model_loader.model(X_tensor).cpu().numpy()[0]

    probability = float(probability)

    # Get SHAP explanation
    explanation = fraud_explainer.explain(X, top_k=top_k)
    english = fraud_explainer.explain_in_english(
        explanation["top_features"]
    )

    return {
        "fraud_probability": round(probability, 4),
        "is_fraud": probability >= threshold,
        "risk_level": get_risk_level(probability),
        "model_round": model_loader.model_round,
        "threshold_used": threshold,
        "top_features": explanation["top_features"],
        "english_explanation": english,
        "total_fraud_push": explanation["total_fraud_push"],
        "total_fraud_pull": explanation["total_fraud_pull"],
    }