import pandas as pd
import numpy as np
import copy
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../simulator"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../models"))
sys.path.append(os.path.dirname(__file__))

from preprocess import fit_scaler, save_scaler, load_scaler
from mlp import FraudMLP
from client import local_train
from server import fedavg, evaluate, save_global_model, save_round_history
from schema import LABEL_COLUMN, NUM_CLIENTS

DATA_DIR   = "../../data/clients"
NUM_ROUNDS = 10
LOCAL_EPOCHS = 3
LEARNING_RATE = 1e-3


def load_client_data() -> list:
    datasets = []
    for i in range(NUM_CLIENTS):
        path = os.path.join(DATA_DIR, f"client_{i}.csv")
        df = pd.read_csv(path)
        datasets.append(df)
        print(f"Loaded client {i}: {len(df):,} rows")
    return datasets


def load_test_data() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "global_test.csv")
    df = pd.read_csv(path)
    print(f"Loaded global test: {len(df):,} rows")
    return df


def main():
    print("=" * 60)
    print("Federated Fraud Detection — Training")
    print("=" * 60)

    # Load all data
    print("\nLoading client datasets...")
    client_datasets = load_client_data()
    test_df = load_test_data()

    # Fit scaler on combined client data (in real FL this would be done
    # locally and only statistics shared — fine for simulation)
    print("\nFitting scaler...")
    combined = pd.concat(client_datasets)
    scaler = fit_scaler(combined)
    save_scaler(scaler)

    # Get input dimension from data
    input_dim = len([c for c in combined.columns if c != LABEL_COLUMN])
    print(f"Input dimension: {input_dim} features")

    # Initialise global model
    global_model = FraudMLP(input_dim=input_dim)
    print(f"Model parameters: {sum(p.numel() for p in global_model.parameters()):,}")

    history = []

    print(f"\nStarting federated training: {NUM_ROUNDS} rounds, "
          f"{LOCAL_EPOCHS} local epochs each")
    print("=" * 60)

    for round_num in range(1, NUM_ROUNDS + 1):
        print(f"\nRound {round_num}/{NUM_ROUNDS}")
        print("-" * 40)

        client_updates = []

        # Each client trains locally on their own data
        for client_id, df in enumerate(client_datasets):
            client_model = copy.deepcopy(global_model)
            update = local_train(
                model=client_model,
                df=df,
                scaler=scaler,
                client_id=client_id,
                epochs=LOCAL_EPOCHS,
                lr=LEARNING_RATE,
            )
            client_updates.append(update)

        # Aggregate on server using FedAvg
        global_model = fedavg(client_updates, global_model)

        # Evaluate global model on held-out test set
        metrics = evaluate(global_model, test_df, scaler)

        print(f"\n  Round {round_num} Global Metrics:")
        print(f"  AUC:       {metrics['auc']}")
        print(f"  F1:        {metrics['f1']}")
        print(f"  Precision: {metrics['precision']}")
        print(f"  Recall:    {metrics['recall']}")

        # Save model checkpoint
        save_global_model(global_model, round_num, metrics)

        # Track history for drift monitoring (Days 7-9)
        round_log = {
            "round": round_num,
            "metrics": metrics,
            "client_losses": [u["avg_loss"] for u in client_updates],
        }
        history.append(round_log)
        save_round_history(history)

    print("\n" + "=" * 60)
    print("Training complete.")
    print(f"Best AUC: {max(r['metrics']['auc'] for r in history)}")
    print(f"Final F1: {history[-1]['metrics']['f1']}")
    print("=" * 60)


if __name__ == "__main__":
    main()