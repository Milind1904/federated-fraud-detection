import torch
import pickle
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../src/models"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../src/simulator"))

from mlp import FraudMLP

MODEL_PATH  = os.path.join(os.path.dirname(__file__), "../data/models/global_model_latest.pt")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "../data/processed/scaler.pkl")


class ModelLoader:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.model_round = None
        self.metrics = None
        self.input_dim = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(self):
        print(f"Loading model from {MODEL_PATH}")
        checkpoint = torch.load(MODEL_PATH, map_location=self.device, weights_only=False)
        
        self.model_round = checkpoint["round"]
        self.metrics     = checkpoint["metrics"]
        self.input_dim   = checkpoint["state_dict"]["network.0.weight"].shape[1]

        self.model = FraudMLP(input_dim=self.input_dim)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device)
        self.model.eval()

        print(f"Loading scaler from {SCALER_PATH}")
        with open(SCALER_PATH, "rb") as f:
            self.scaler = pickle.load(f)

        print(f"Model loaded — Round {self.model_round} | "
              f"AUC: {self.metrics['auc']} | "
              f"Input dim: {self.input_dim}")

    def is_loaded(self) -> bool:
        return self.model is not None and self.scaler is not None


# Single global instance — loaded once at API startup
model_loader = ModelLoader()