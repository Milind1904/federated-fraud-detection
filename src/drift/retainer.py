import sys
import os
import copy
import pandas as pd
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "../simulator"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../models"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../federated"))

from mlp import FraudMLP
from client import local_train
from server import fedavg, evaluate, save_global_model
from preprocess import fit_scaler, save_scaler, load_scaler
from schema import LABEL_COLUMN, NUM_CLIENTS

DATA_DIR   = os.path.join(os.path.dirname(__file__), "../../data/clients")
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "../../data/models")
SCALER_PATH = os.path.join(os.path.dirname(__file__),
                           "../../data/processed/scaler.pkl")


def retrain(
    trigger_reason: str = "drift_detected",
    num_rounds: int = 3,
    local_epochs: int = 2,
) -> dict:
    """
    Run a quick federated retraining cycle.
    Uses fewer rounds than the initial training — just enough
    to adapt to the new distribution.

    Returns metrics of the newly trained model.
    """
    print(f"\n{'='*50}")
    print(f"RETRAINING TRIGGERED — reason: {trigger_reason}")
    print(f"Rounds: {num_rounds} | Local epochs: {local_epochs}")
    print(f"{'='*50}")

    # Load current global model as starting point
    latest_path = os.path.join(MODEL_DIR, "global_model_latest.pt")
    checkpoint = torch.load(
        latest_path, map_location="cpu", weights_only=False
    )
    current_round = checkpoint["round"]
    input_dim = checkpoint["state_dict"]["network.0.weight"].shape[1]

    global_model = FraudMLP(input_dim=input_dim)
    global_model.load_state_dict(checkpoint["state_dict"])

    # Load client data
    client_datasets = []
    for i in range(NUM_CLIENTS):
        path = os.path.join(DATA_DIR, f"client_{i}.csv")
        df = pd.read_csv(path)
        client_datasets.append(df)

    # Load test set
    test_df = pd.read_csv(os.path.join(DATA_DIR, "global_test.csv"))

    # Load scaler
    scaler = load_scaler(SCALER_PATH)

    best_metrics = None
    best_model_state = None

    for round_num in range(1, num_rounds + 1):
        new_round = current_round + round_num
        print(f"\nRetraining round {round_num}/{num_rounds} "
              f"(global round {new_round})")

        client_updates = []
        for client_id, df in enumerate(client_datasets):
            client_model = copy.deepcopy(global_model)
            update = local_train(
                model=client_model,
                df=df,
                scaler=scaler,
                client_id=client_id,
                epochs=local_epochs,
                lr=5e-4,
            )
            client_updates.append(update)

        global_model = fedavg(client_updates, global_model)
        metrics = evaluate(global_model, test_df, scaler)

        print(f"  AUC: {metrics['auc']} | "
              f"F1: {metrics['f1']} | "
              f"Recall: {metrics['recall']}")

        if best_metrics is None or metrics["auc"] > best_metrics["auc"]:
            best_metrics = metrics
            best_model_state = copy.deepcopy(global_model.state_dict())

    # Save best model from retraining
    global_model.load_state_dict(best_model_state)
    save_global_model(global_model, current_round + num_rounds, best_metrics)

    print(f"\nRetraining complete.")
    print(f"New model AUC: {best_metrics['auc']} "
          f"(was {checkpoint['metrics']['auc']})")
    print(f"{'='*50}\n")

    return {
        "previous_auc": checkpoint["metrics"]["auc"],
        "new_auc": best_metrics["auc"],
        "improved": best_metrics["auc"] > checkpoint["metrics"]["auc"],
        "new_round": current_round + num_rounds,
        "metrics": best_metrics,
    }