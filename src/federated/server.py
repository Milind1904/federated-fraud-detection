import torch
import numpy as np
import pandas as pd
import copy
import json
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../simulator"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../models"))
from mlp import FraudMLP
from preprocess import preprocess
from sklearn.metrics import (
    roc_auc_score, f1_score,
    precision_score, recall_score,
    classification_report
)


def fedavg(client_updates: list, global_model: FraudMLP) -> FraudMLP:
    total_samples = sum(u["n_samples"] for u in client_updates)
    global_state = copy.deepcopy(global_model.state_dict())

    for key in global_state.keys():
        global_state[key] = torch.zeros_like(global_state[key], dtype=torch.float32).cpu()
        for update in client_updates:
            weight = update["n_samples"] / total_samples
            global_state[key] += weight * update["state_dict"][key].float().cpu()

    global_model.load_state_dict(global_state)
    return global_model

def evaluate(model: FraudMLP,
             df: pd.DataFrame,
             scaler,
             threshold: float = 0.5) -> dict:
    """
    Evaluate the global model on a held-out dataset.
    Returns AUC, F1, precision, recall.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    X, y_true = preprocess(df, scaler)
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    with torch.no_grad():
        y_prob = model(X_tensor).cpu().numpy()

    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "auc":       round(float(roc_auc_score(y_true, y_prob)), 4),
        "f1":        round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
    }
    return metrics


def save_global_model(model: FraudMLP,
                      round_num: int,
                      metrics: dict,
                      output_dir: str = "../../data/models"):
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, f"global_model_round_{round_num}.pt")
    torch.save({
        "round": round_num,
        "state_dict": model.state_dict(),
        "metrics": metrics,
    }, model_path)

    # Always keep a copy as "latest" for the API to load
    latest_path = os.path.join(output_dir, "global_model_latest.pt")
    torch.save({
        "round": round_num,
        "state_dict": model.state_dict(),
        "metrics": metrics,
    }, latest_path)

    print(f"  Model saved: {model_path}")
    return model_path


def save_round_history(history: list,
                       path: str = "../../logs/round_history.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(history, f, indent=2)