import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(__file__))
from schema import (
    LABEL_COLUMN, NUM_CLIENTS,
    NUMERICAL_FEATURES, CATEGORICAL_FEATURES
)

RAW_TRANSACTION = "../../data/raw/train_transaction.csv"
RAW_IDENTITY    = "../../data/raw/train_identity.csv"
OUTPUT_DIR      = "../../data/clients"
np.random.seed(42)


def load_raw_data() -> pd.DataFrame:
    print("Loading transaction data...")
    df_tx = pd.read_csv(RAW_TRANSACTION)
    print(f"Transactions: {df_tx.shape[0]:,} rows, {df_tx.shape[1]} columns")

    print("Loading identity data...")
    df_id = pd.read_csv(RAW_IDENTITY)
    print(f"Identity:     {df_id.shape[0]:,} rows, {df_id.shape[1]} columns")

    print("Merging on TransactionID...")
    df = df_tx.merge(df_id[["TransactionID", "DeviceType"]], 
                     on="TransactionID", how="left")
    print(f"Merged:       {df.shape[0]:,} rows, {df.shape[1]} columns")
    print(f"Overall fraud rate: {df[LABEL_COLUMN].mean():.2%}")
    return df


def select_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    print("\nCleaning data...")

    cols_to_keep = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + [LABEL_COLUMN]
    
    # Only keep columns that actually exist in the dataframe
    cols_to_keep = [c for c in cols_to_keep if c in df.columns]
    df = df[cols_to_keep].copy()

    # Fill numerical nulls with median
    for col in NUMERICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Fill categorical nulls with 'unknown'
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna("unknown")

    # One-hot encode categoricals
    df = pd.get_dummies(df, columns=[c for c in CATEGORICAL_FEATURES if c in df.columns],
                        prefix=[c for c in CATEGORICAL_FEATURES if c in df.columns])

    # Convert bool columns to int
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    print(f"After cleaning: {df.shape[0]:,} rows, {df.shape[1]} columns")
    print(f"Nulls remaining: {df.isnull().sum().sum()}")
    return df


def partition_into_clients(df: pd.DataFrame) -> dict:
    """
    Split into 5 clients with different fraud rate profiles.
    No transaction appears in more than one client.
    """
    print("\nPartitioning into clients...")

    fraud_df = df[df[LABEL_COLUMN] == 1].sample(frac=1, random_state=42).reset_index(drop=True)
    legit_df = df[df[LABEL_COLUMN] == 0].sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Total fraud: {len(fraud_df):,}")
    print(f"Total legit: {len(legit_df):,}")

    # Different fraud exposure per client — simulates different bank risk profiles
    # fraud_shares must sum to <= 1.0 (remainder goes to global test set)
    fraud_shares = [0.15, 0.25, 0.10, 0.25, 0.15]  # sums to 0.90
    legit_shares = [0.18, 0.18, 0.18, 0.18, 0.18]  # sums to 0.90

    clients = {}
    fraud_idx = 0
    legit_idx = 0

    for client_id in range(NUM_CLIENTS):
        n_fraud = int(len(fraud_df) * fraud_shares[client_id])
        n_legit = int(len(legit_df) * legit_shares[client_id])

        client_fraud = fraud_df.iloc[fraud_idx : fraud_idx + n_fraud]
        client_legit = legit_df.iloc[legit_idx : legit_idx + n_legit]

        fraud_idx += n_fraud
        legit_idx += n_legit

        client_df = pd.concat([client_fraud, client_legit])
        client_df = client_df.sample(frac=1, random_state=42).reset_index(drop=True)
        clients[client_id] = client_df

    return clients, fraud_df.iloc[fraud_idx:], legit_df.iloc[legit_idx:]


def save_clients(clients: dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    for client_id, df in clients.items():
        path = os.path.join(output_dir, f"client_{client_id}.csv")
        df.to_csv(path, index=False)
        fraud_count = df[LABEL_COLUMN].sum()
        print(f"Client {client_id}: {len(df):,} transactions | "
              f"{fraud_count:,} fraud ({fraud_count/len(df)*100:.1f}%)")


def save_global_test(remaining_fraud: pd.DataFrame,
                     remaining_legit: pd.DataFrame,
                     output_dir: str):
    """
    Global test set = the 10% of fraud + legit not assigned to any client.
    This gives a clean held-out set with no overlap with any client.
    """
    global_df = pd.concat([remaining_fraud, remaining_legit])
    global_df = global_df.sample(frac=1, random_state=99).reset_index(drop=True)

    path = os.path.join(output_dir, "global_test.csv")
    global_df.to_csv(path, index=False)
    fraud_count = global_df[LABEL_COLUMN].sum()
    print(f"\nGlobal test: {len(global_df):,} transactions | "
          f"{fraud_count:,} fraud ({fraud_count/len(global_df)*100:.1f}%)")


if __name__ == "__main__":
    df_raw   = load_raw_data()
    df_clean = select_and_clean(df_raw)
    clients, rem_fraud, rem_legit = partition_into_clients(df_clean)
    save_clients(clients, OUTPUT_DIR)
    save_global_test(rem_fraud, rem_legit, OUTPUT_DIR)
    print("\nDay 1 complete.")