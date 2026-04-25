import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import os
import sys
sys.path.append(os.path.dirname(__file__))
from schema import LABEL_COLUMN


def get_feature_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c != LABEL_COLUMN]


def fit_scaler(df: pd.DataFrame) -> StandardScaler:
    feature_cols = get_feature_columns(df)
    scaler = StandardScaler()
    scaler.fit(df[feature_cols])
    return scaler


def preprocess(df: pd.DataFrame, scaler: StandardScaler) -> tuple:
    feature_cols = get_feature_columns(df)
    X = scaler.transform(df[feature_cols]).astype(np.float32)
    y = df[LABEL_COLUMN].values.astype(np.float32)
    return X, y


def save_scaler(scaler: StandardScaler, path: str = "../../data/processed/scaler.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {path}")


def load_scaler(path: str = "../../data/processed/scaler.pkl") -> StandardScaler:
    with open(path, "rb") as f:
        return pickle.load(f)