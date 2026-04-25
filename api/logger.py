import numpy as np
import threading
from database import (
    log_prediction,
    get_recent_predictions,
    get_total_prediction_count,
    log_performance_window,
    get_predictions_window,
)
from sklearn.metrics import (
    roc_auc_score, f1_score,
    precision_score, recall_score
)
import json


ROLLING_WINDOW_SIZE = 500
PERFORMANCE_EVAL_EVERY = 100


class PredictionLogger:
    def __init__(self):
        self._lock = threading.Lock()
        self._prediction_count = 0

    def log(
        self,
        transaction_amt: float,
        fraud_probability: float,
        is_fraud_predicted: bool,
        risk_level: str,
        threshold_used: float,
        model_round: int,
        shap_values: list = None,
        english_explanation: str = None,
        ground_truth: int = None,
    ) -> int:
        """
        Log a prediction to the database.
        Triggers rolling performance evaluation every N predictions.
        """
        with self._lock:
            row_id = log_prediction(
                transaction_amt=transaction_amt,
                fraud_probability=fraud_probability,
                is_fraud_predicted=is_fraud_predicted,
                risk_level=risk_level,
                threshold_used=threshold_used,
                model_round=model_round,
                shap_values=shap_values,
                english_explanation=english_explanation,
                ground_truth=ground_truth,
            )

            self._prediction_count += 1

            # Every 100 predictions, compute rolling performance
            if self._prediction_count % PERFORMANCE_EVAL_EVERY == 0:
                self._evaluate_rolling_window(row_id)

            return row_id

    def _evaluate_rolling_window(self, latest_id: int):
        """
        Compute AUC/F1 over the last ROLLING_WINDOW_SIZE predictions
        that have ground truth labels.
        """
        start_id = max(1, latest_id - ROLLING_WINDOW_SIZE + 1)
        rows = get_predictions_window(start_id, latest_id)

        # Only evaluate rows with ground truth
        labeled = [r for r in rows if r["ground_truth"] is not None]

        if len(labeled) < 50:
            return

        y_true = [r["ground_truth"] for r in labeled]
        y_prob = [r["fraud_probability"] for r in labeled]
        y_pred = [r["is_fraud_predicted"] for r in labeled]

        # Need both classes present for AUC
        if len(set(y_true)) < 2:
            return

        try:
            auc = round(roc_auc_score(y_true, y_prob), 4)
            f1  = round(f1_score(y_true, y_pred, zero_division=0), 4)
            prec = round(precision_score(y_true, y_pred, zero_division=0), 4)
            rec  = round(recall_score(y_true, y_pred, zero_division=0), 4)
            fraud_rate = round(sum(y_true) / len(y_true), 4)
            avg_prob   = round(sum(y_prob) / len(y_prob), 4)

            log_performance_window(
                window_start=start_id,
                window_end=latest_id,
                window_size=len(labeled),
                auc=auc, f1=f1,
                precision=prec, recall=rec,
                fraud_rate=fraud_rate,
                avg_probability=avg_prob,
            )

            print(f"Rolling window [{start_id}-{latest_id}]: "
                  f"AUC={auc} F1={f1} "
                  f"fraud_rate={fraud_rate:.2%}")

        except Exception as e:
            print(f"Rolling eval error: {e}")

    def get_stats(self) -> dict:
        """Quick stats for the /stats endpoint."""
        total = get_total_prediction_count()
        recent = get_recent_predictions(limit=500)

        if not recent:
            return {"total_predictions": total, "recent_fraud_rate": 0.0}

        fraud_count = sum(1 for r in recent if r["is_fraud_predicted"])
        avg_prob = sum(r["fraud_probability"] for r in recent) / len(recent)

        risk_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        for r in recent:
            level = r["risk_level"]
            if level in risk_counts:
                risk_counts[level] += 1

        return {
            "total_predictions": total,
            "recent_fraud_rate": round(fraud_count / len(recent), 4),
            "avg_fraud_probability": round(avg_prob, 4),
            "risk_distribution": risk_counts,
            "window_size": len(recent),
        }


prediction_logger = PredictionLogger()