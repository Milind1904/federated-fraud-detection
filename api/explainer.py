import torch
import torch.nn as nn
import shap
import numpy as np
import pandas as pd
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../src/simulator"))

from model_loader import model_loader


class ModelWrapper(nn.Module):
    """
    Wraps FraudMLP to return 2D output (batch, 1)
    instead of 1D (batch,) — required by shap.DeepExplainer.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out.unsqueeze(1)


class FraudExplainer:
    def __init__(self):
        self.explainer = None
        self.feature_names = None
        self.background_size = 200

    def load(self):
        print("Initializing SHAP explainer...")

        test_path = os.path.join(
            os.path.dirname(__file__),
            "../data/clients/global_test.csv"
        )
        df = pd.read_csv(test_path)

        label_col = "isFraud"
        feature_cols = [c for c in df.columns if c != label_col]
        self.feature_names = feature_cols

        legit = df[df[label_col] == 0].sample(
            n=min(180, len(df[df[label_col] == 0])),
            random_state=42
        )
        fraud = df[df[label_col] == 1].sample(
            n=min(20, len(df[df[label_col] == 1])),
            random_state=42
        )
        background_df = pd.concat([legit, fraud])[feature_cols]

        background_scaled = model_loader.scaler.transform(
            background_df
        ).astype(np.float32)

        background_tensor = torch.tensor(
            background_scaled,
            dtype=torch.float32
        ).to(model_loader.device)

        # Wrap model to fix output shape for SHAP
        wrapped_model = ModelWrapper(model_loader.model)
        wrapped_model.eval()

        self.explainer = shap.DeepExplainer(
            wrapped_model,
            background_tensor
        )

        print(f"SHAP explainer ready — "
              f"background size: {len(background_df)} samples, "
              f"{len(feature_cols)} features")

    def is_loaded(self) -> bool:
        return self.explainer is not None

    def explain(self, X_scaled: np.ndarray, top_k: int = 10) -> dict:
        X_tensor = torch.tensor(
            X_scaled,
            dtype=torch.float32
        ).to(model_loader.device)

        shap_values = self.explainer.shap_values(X_tensor)

        # shap_values is a list with one element for our single output
        if isinstance(shap_values, list):
            shap_vals = np.array(shap_values[0]).flatten()
        else:
            shap_vals = np.array(shap_values).flatten()

        attributions = []
        for i, (fname, sval) in enumerate(
            zip(self.feature_names, shap_vals)
        ):
            attributions.append({
                "feature": fname,
                "shap_value": round(float(sval), 6),
                "direction": "increases_fraud" if sval > 0 else "decreases_fraud",
                "raw_value": round(float(X_scaled[0][i]), 4),
            })

        attributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        top_features = attributions[:top_k]

        total_positive = sum(
            a["shap_value"] for a in attributions if a["shap_value"] > 0
        )
        total_negative = sum(
            a["shap_value"] for a in attributions if a["shap_value"] < 0
        )

        return {
            "top_features": top_features,
            "total_fraud_push": round(float(total_positive), 6),
            "total_fraud_pull": round(float(total_negative), 6),
            "n_features_total": len(self.feature_names),
        }

    def explain_in_english(self, top_features: list) -> str:
        pushers = [
            f for f in top_features if f["direction"] == "increases_fraud"
        ][:3]
        pullers = [
            f for f in top_features if f["direction"] == "decreases_fraud"
        ][:2]

        parts = []
        if pushers:
            pusher_names = [p["feature"].replace("_", " ") for p in pushers]
            parts.append(
                f"Flagged primarily due to: {', '.join(pusher_names)}"
            )
        if pullers:
            puller_names = [p["feature"].replace("_", " ") for p in pullers]
            parts.append(
                f"Partially offset by: {', '.join(puller_names)}"
            )

        if not parts:
            return "Insufficient signal for explanation."

        return ". ".join(parts) + "."


fraud_explainer = FraudExplainer()
