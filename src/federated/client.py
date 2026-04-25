import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../simulator"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../models"))
from preprocess import preprocess, fit_scaler
from mlp import FraudMLP


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def df_to_dataloader(df: pd.DataFrame,
                     scaler,
                     batch_size: int = 512,
                     shuffle: bool = True) -> DataLoader:
    X, y = preprocess(df, scaler)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def compute_class_weights(df: pd.DataFrame) -> torch.Tensor:
    """
    Fraud is rare (2-5%) so we weight the loss function to penalise
    missing a fraud transaction more than a false alarm.
    """
    from schema import LABEL_COLUMN
    n_legit = (df[LABEL_COLUMN] == 0).sum()
    n_fraud = (df[LABEL_COLUMN] == 1).sum()
    weight = n_legit / n_fraud
    return torch.tensor(weight, dtype=torch.float32)


def local_train(model: FraudMLP,
                df: pd.DataFrame,
                scaler,
                client_id: int,
                epochs: int = 3,
                lr: float = 1e-3,
                batch_size: int = 512) -> dict:
    """
    Train the model locally on one client's data.
    Returns the updated state_dict — raw data never leaves this function.
    """
    device = get_device()
    model = model.to(device)
    model.train()

    dataloader = df_to_dataloader(df, scaler, batch_size=batch_size)
    pos_weight = compute_class_weights(df).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Use BCEWithLogitsLoss for numerical stability — remove Sigmoid from model
    # when using this loss. We'll handle this by wrapping forward pass.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    total_loss = 0.0
    total_batches = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            # Get raw logits (bypass sigmoid for BCEWithLogitsLoss)
            logits = model.network[:-1](X_batch).squeeze(1)
            loss = criterion(logits, y_batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            total_batches += 1

        scheduler.step()
        avg_epoch_loss = epoch_loss / len(dataloader)
        total_loss += epoch_loss
        print(f"    Client {client_id} | Epoch {epoch+1}/{epochs} "
              f"| Loss: {avg_epoch_loss:.4f}")

    avg_loss = total_loss / total_batches
    return {
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "n_samples": len(df),
        "avg_loss": avg_loss,
        "client_id": client_id,
    }