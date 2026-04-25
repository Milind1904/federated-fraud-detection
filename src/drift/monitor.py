import sys
import os
import json
import time
import threading
import sqlite3

sys.path.append(os.path.join(os.path.dirname(__file__), "../../api"))
sys.path.append(os.path.dirname(__file__))

from detector import DDMDetector, SlidingWindowDetector, SHAPDriftDetector

DB_PATH = os.path.join(
    os.path.dirname(__file__),
    "../../data/processed/predictions.db"
)

MONITOR_INTERVAL_SECONDS = 30
WINDOW_SIZE = 200
MIN_PREDICTIONS_TO_START = 50


class DriftMonitor:
    def __init__(self):
        self.ddm = DDMDetector(
            warning_threshold=2.0,
            drift_threshold=3.0,
            min_samples=30
        )
        self.sliding = SlidingWindowDetector(
            window_size=WINDOW_SIZE,
            auc_drop_threshold=0.05,
            fraud_rate_shift_threshold=0.02
        )
        self.shap_detector = SHAPDriftDetector(
            window_size=WINDOW_SIZE,
            js_threshold=0.15
        )

        self._thread = None
        self._running = False
        self._last_processed_id = 0
        self.retraining_in_progress = False

    def start(self):
        """Start background monitoring thread."""
        self._running = True
        self._thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self._thread.start()
        print("Drift monitor started — "
              f"checking every {MONITOR_INTERVAL_SECONDS}s")

    def stop(self):
        self._running = False
        print("Drift monitor stopped")

    def _get_new_predictions(self) -> list:
        """Fetch predictions we haven't processed yet."""
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM predictions
            WHERE id > ?
            ORDER BY id ASC
        """, (self._last_processed_id,))
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return rows

    def _get_recent_window(self, n: int = WINDOW_SIZE) -> list:
        """Fetch the most recent N predictions."""
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM predictions
            ORDER BY id DESC
            LIMIT ?
        """, (n,))
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return rows[::-1]

    def _log_drift_event(self, result, retraining_triggered: bool = False):
        """Write drift event to database."""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO drift_events (
                timestamp, drift_type, severity,
                metric_name, metric_value, threshold_value,
                description, retraining_triggered
            ) VALUES (datetime('now'), ?, ?, ?, ?, ?, ?, ?)
        """, (
            "concept_drift" if result.drift_detected else "warning",
            result.severity,
            result.metric_name,
            result.metric_value,
            result.threshold_value,
            result.description,
            int(retraining_triggered),
        ))
        conn.commit()
        conn.close()

    def _monitor_loop(self):
        """Main monitoring loop — runs in background thread."""
        while self._running:
            try:
                self._run_checks()
            except Exception as e:
                print(f"Drift monitor error: {e}")
            time.sleep(MONITOR_INTERVAL_SECONDS)

    def _run_checks(self):
        total = self._get_total_count()
        if total < MIN_PREDICTIONS_TO_START:
            return

        new_preds = self._get_new_predictions()
        if not new_preds:
            return

        # Update DDM with labeled predictions
        labeled = [p for p in new_preds if p["ground_truth"] is not None]
        for pred in labeled:
            error = int(pred["is_fraud_predicted"] != pred["ground_truth"])
            result = self.ddm.update(error)
            if result.drift_detected or result.warning_detected:
                print(f"DDM alert: {result.description}")
                self._log_drift_event(result)
                if result.drift_detected and not self.retraining_in_progress:
                    self._trigger_retraining("DDM drift detected")

        # Update last processed ID
        if new_preds:
            self._last_processed_id = new_preds[-1]["id"]

        # Sliding window check every WINDOW_SIZE predictions
        if total % WINDOW_SIZE == 0:
            window = self._get_recent_window(WINDOW_SIZE)
            labeled_window = [
                p for p in window if p["ground_truth"] is not None
            ]

            if len(labeled_window) >= 50:
                fraud_rate = sum(
                    p["ground_truth"] for p in labeled_window
                ) / len(labeled_window)
                probs = [p["fraud_probability"] for p in labeled_window]
                avg_prob = sum(probs) / len(probs)

                result = self.sliding.check(avg_prob, fraud_rate)
                if result.drift_detected or result.warning_detected:
                    print(f"Sliding window alert: {result.description}")
                    self._log_drift_event(result)
                    if result.drift_detected and \
                            not self.retraining_in_progress:
                        self._trigger_retraining(
                            "Sliding window drift detected"
                        )

            # SHAP drift check
            shap_data = [
                json.loads(p["shap_values_json"])
                for p in window
                if p["shap_values_json"] is not None
            ]
            if shap_data:
                result = self.shap_detector.update(shap_data)
                if result and (result.drift_detected or
                               result.warning_detected):
                    print(f"SHAP drift alert: {result.description}")
                    self._log_drift_event(result)

    def _trigger_retraining(self, reason: str):
        """Trigger retraining in a separate thread."""
        def retrain_job():
            self.retraining_in_progress = True
            try:
                from retrainer import retrain
                result = retrain(trigger_reason=reason, num_rounds=3)

                # Reload model in API after retraining
                sys.path.append(
                    os.path.join(os.path.dirname(__file__), "../../api")
                )
                from model_loader import model_loader
                model_loader.load()
                print(f"Model reloaded after retraining. "
                      f"New AUC: {result['new_auc']}")
            except Exception as e:
                print(f"Retraining error: {e}")
            finally:
                self.retraining_in_progress = False

        thread = threading.Thread(target=retrain_job, daemon=True)
        thread.start()

    def _get_total_count(self) -> int:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM predictions")
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def get_status(self) -> dict:
        return {
            "running": self._running,
            "last_processed_id": self._last_processed_id,
            "retraining_in_progress": self.retraining_in_progress,
            "ddm_error_rate": round(self.ddm.mean, 4),
            "ddm_in_warning": self.ddm.in_warning,
            "monitor_interval_seconds": MONITOR_INTERVAL_SECONDS,
            "min_predictions_to_start": MIN_PREDICTIONS_TO_START,
        }


drift_monitor = DriftMonitor()