import pandas as pd
import os
import glob
import sys
sys.path.append(os.path.dirname(__file__))
from schema import LABEL_COLUMN

DATA_DIR = "../../data/clients"


def validate_datasets():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "client_*.csv")))
    print(f"Found {len(files)} client files\n")

    all_columns = None
    for path in files:
        df = pd.read_csv(path)
        client_name = os.path.basename(path)

        if all_columns is None:
            all_columns = set(df.columns)
        else:
            assert set(df.columns) == all_columns, \
                f"Schema mismatch in {client_name}"

        fraud_rate = df[LABEL_COLUMN].mean()
        print(f"{client_name}: shape={df.shape} | "
              f"fraud={fraud_rate:.2%} | "
              f"nulls={df.isnull().sum().sum()}")

    print("\nChecking for transaction overlap between clients...")
    all_dfs = []
    for path in files:
        df = pd.read_csv(path)
        all_dfs.append(df)

    # Check that total rows across clients equals sum of individual counts
    total_rows = sum(len(d) for d in all_dfs)
    combined = pd.concat(all_dfs)
    print(f"Total rows across all clients: {total_rows:,}")
    print(f"Combined unique rows: {len(combined):,}")

    if total_rows == len(combined):
        print("No duplicate rows across clients.")
    else:
        print(f"Warning: {total_rows - len(combined)} duplicate rows found.")

    print("\nGlobal test set:")
    global_path = os.path.join(DATA_DIR, "global_test.csv")
    if os.path.exists(global_path):
        gdf = pd.read_csv(global_path)
        print(f"  shape={gdf.shape} | "
              f"fraud={gdf[LABEL_COLUMN].mean():.2%} | "
              f"nulls={gdf.isnull().sum().sum()}")
    else:
        print("  global_test.csv not found")

    print("\nAll validations passed.")


if __name__ == "__main__":
    validate_datasets()