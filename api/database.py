import sqlite3
import json
import os
import numpy as np
from datetime import datetime


DB_PATH = os.path.join(os.path.dirname(__file__), "../data/processed/predictions.db")


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """
    Create all tables if they don't exist.
    Called once at API startup.
    """
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = get_connection()
    cursor = conn.cursor()

    # Main predictions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp           TEXT NOT NULL,
            transaction_amt     REAL,
            fraud_probability   REAL NOT NULL,
            is_fraud_predicted  INTEGER NOT NULL,
            risk_level          TEXT NOT NULL,
            threshold_used      REAL NOT NULL,
            model_round         INTEGER NOT NULL,
            shap_values_json    TEXT,
            english_explanation TEXT,
            ground_truth        INTEGER,
            correct             INTEGER
        )
    """)

    # Rolling performance table — one row per evaluation window
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS performance_windows (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT NOT NULL,
            window_start    INTEGER NOT NULL,
            window_end      INTEGER NOT NULL,
            window_size     INTEGER NOT NULL,
            auc             REAL,
            f1              REAL,
            precision_score REAL,
            recall_score    REAL,
            fraud_rate      REAL,
            avg_probability REAL
        )
    """)

    # Drift events table — records when drift is detected
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS drift_events (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT NOT NULL,
            drift_type      TEXT NOT NULL,
            severity        TEXT NOT NULL,
            metric_name     TEXT,
            metric_value    REAL,
            threshold_value REAL,
            description     TEXT,
            retraining_triggered INTEGER DEFAULT 0
        )
    """)

    conn.commit()
    conn.close()
    print(f"Database initialized at {DB_PATH}")


def log_prediction(
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
    Insert a single prediction into the database.
    Returns the new row ID.
    """
    conn = get_connection()
    cursor = conn.cursor()

    correct = None
    if ground_truth is not None:
        correct = int(int(is_fraud_predicted) == ground_truth)

    shap_json = json.dumps(shap_values) if shap_values else None

    cursor.execute("""
        INSERT INTO predictions (
            timestamp, transaction_amt, fraud_probability,
            is_fraud_predicted, risk_level, threshold_used,
            model_round, shap_values_json, english_explanation,
            ground_truth, correct
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.utcnow().isoformat(),
        transaction_amt,
        fraud_probability,
        int(is_fraud_predicted),
        risk_level,
        threshold_used,
        model_round,
        shap_json,
        english_explanation,
        ground_truth,
        correct,
    ))

    row_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return row_id


def get_recent_predictions(limit: int = 100) -> list:
    """Fetch the most recent N predictions."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM predictions
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_predictions_window(start_id: int, end_id: int) -> list:
    """Fetch predictions between two IDs — used for drift detection."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM predictions
        WHERE id BETWEEN ? AND ?
        ORDER BY id ASC
    """, (start_id, end_id))
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_total_prediction_count() -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM predictions")
    count = cursor.fetchone()[0]
    conn.close()
    return count


def log_performance_window(
    window_start: int,
    window_end: int,
    window_size: int,
    auc: float = None,
    f1: float = None,
    precision: float = None,
    recall: float = None,
    fraud_rate: float = None,
    avg_probability: float = None,
):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO performance_windows (
            timestamp, window_start, window_end, window_size,
            auc, f1, precision_score, recall_score,
            fraud_rate, avg_probability
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.utcnow().isoformat(),
        window_start, window_end, window_size,
        auc, f1, precision, recall,
        fraud_rate, avg_probability,
    ))
    conn.commit()
    conn.close()


def log_drift_event(
    drift_type: str,
    severity: str,
    metric_name: str = None,
    metric_value: float = None,
    threshold_value: float = None,
    description: str = None,
    retraining_triggered: bool = False,
) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO drift_events (
            timestamp, drift_type, severity,
            metric_name, metric_value, threshold_value,
            description, retraining_triggered
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.utcnow().isoformat(),
        drift_type, severity,
        metric_name, metric_value, threshold_value,
        description, int(retraining_triggered),
    ))
    row_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return row_id


def get_drift_events(limit: int = 50) -> list:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM drift_events
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_performance_history() -> list:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM performance_windows
        ORDER BY id ASC
    """)
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows