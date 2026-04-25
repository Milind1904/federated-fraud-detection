from fastapi import FastAPI, HTTPException, Query
from contextlib import asynccontextmanager
import time
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../src/models"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../src/simulator"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../src/drift"))

from schemas import (
    TransactionRequest, PredictionResponse,
    HealthResponse, ModelInfoResponse
)
from model_loader import model_loader
from explainer import fraud_explainer
from predictor import predict, predict_with_explanation
from database import (
    init_db, get_recent_predictions,
    get_drift_events, get_performance_history,
    log_drift_event
)
from logger import prediction_logger
from monitor import drift_monitor


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_loader.load()
    fraud_explainer.load()
    init_db()
    drift_monitor.start()
    yield
    drift_monitor.stop()


app = FastAPI(
    title="Federated Fraud Detection API",
    description="Real-time transaction scoring with federated learning",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health():
    return {
        "status": "ok",
        "model_loaded": model_loader.is_loaded(),
        "model_round": model_loader.model_round or 0,
        "input_dim": model_loader.input_dim or 0,
    }


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    if not model_loader.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "model_round": model_loader.model_round,
        "input_dim":   model_loader.input_dim,
        "auc":         model_loader.metrics["auc"],
        "f1":          model_loader.metrics["f1"],
        "precision":   model_loader.metrics["precision"],
        "recall":      model_loader.metrics["recall"],
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_transaction(
    request: TransactionRequest,
    threshold: float = Query(default=0.5, ge=0.0, le=1.0),
    ground_truth: int = Query(default=None, ge=0, le=1),
):
    if not model_loader.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.time()
    result = predict(request, threshold=threshold)
    latency_ms = round((time.time() - start) * 1000, 2)

    prediction_logger.log(
        transaction_amt=request.TransactionAmt,
        fraud_probability=result["fraud_probability"],
        is_fraud_predicted=result["is_fraud"],
        risk_level=result["risk_level"],
        threshold_used=threshold,
        model_round=result["model_round"],
        ground_truth=ground_truth,
    )

    print(f"Prediction: {result['fraud_probability']:.4f} "
          f"({result['risk_level']}) | {latency_ms}ms | logged")
    return result


@app.post("/predict/batch")
def predict_batch(
    requests: list[TransactionRequest],
    threshold: float = Query(default=0.5, ge=0.0, le=1.0)
):
    if not model_loader.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    if len(requests) > 100:
        raise HTTPException(status_code=400, detail="Max 100 per batch")

    results = []
    for req in requests:
        result = predict(req, threshold=threshold)
        prediction_logger.log(
            transaction_amt=req.TransactionAmt,
            fraud_probability=result["fraud_probability"],
            is_fraud_predicted=result["is_fraud"],
            risk_level=result["risk_level"],
            threshold_used=threshold,
            model_round=result["model_round"],
        )
        results.append(result)
    return {"predictions": results, "count": len(results)}


@app.post("/explain")
def explain_transaction(
    request: TransactionRequest,
    threshold: float = Query(default=0.5, ge=0.0, le=1.0),
    top_k: int = Query(default=10, ge=1, le=50),
    ground_truth: int = Query(default=None, ge=0, le=1),
):
    if not model_loader.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not fraud_explainer.is_loaded():
        raise HTTPException(status_code=503, detail="Explainer not loaded")

    start = time.time()
    result = predict_with_explanation(
        request, threshold=threshold, top_k=top_k
    )
    latency_ms = round((time.time() - start) * 1000, 2)

    prediction_logger.log(
        transaction_amt=request.TransactionAmt,
        fraud_probability=result["fraud_probability"],
        is_fraud_predicted=result["is_fraud"],
        risk_level=result["risk_level"],
        threshold_used=threshold,
        model_round=result["model_round"],
        shap_values=result["top_features"],
        english_explanation=result["english_explanation"],
        ground_truth=ground_truth,
    )

    print(f"Explain: {result['fraud_probability']:.4f} "
          f"({result['risk_level']}) | {latency_ms}ms | logged")
    return result


@app.get("/explain/features")
def get_feature_importance():
    if not fraud_explainer.is_loaded():
        raise HTTPException(status_code=503, detail="Explainer not loaded")
    return {
        "feature_names": fraud_explainer.feature_names,
        "n_features": len(fraud_explainer.feature_names),
        "model_round": model_loader.model_round,
    }


@app.get("/stats")
def get_stats():
    return prediction_logger.get_stats()


@app.get("/predictions/recent")
def recent_predictions(limit: int = Query(default=50, ge=1, le=500)):
    rows = get_recent_predictions(limit=limit)
    return {"predictions": rows, "count": len(rows)}


@app.get("/drift/status")
def drift_status():
    return drift_monitor.get_status()


@app.get("/drift/events")
def get_drift_event_log(limit: int = Query(default=50, ge=1, le=200)):
    events = get_drift_events(limit=limit)
    return {"events": events, "count": len(events)}


@app.post("/drift/simulate")
def simulate_drift():
    event_id = log_drift_event(
        drift_type="simulated",
        severity="critical",
        metric_name="auc",
        metric_value=0.72,
        threshold_value=0.83,
        description="Manually simulated drift event for testing",
        retraining_triggered=False,
    )
    return {"message": "Drift event simulated", "event_id": event_id}


@app.post("/retrain")
def manual_retrain():
    if drift_monitor.retraining_in_progress:
        raise HTTPException(
            status_code=409,
            detail="Retraining already in progress"
        )
    drift_monitor._trigger_retraining("manual_trigger")
    return {"message": "Retraining triggered", "status": "started"}


@app.get("/performance/history")
def performance_history():
    history = get_performance_history()
    return {"history": history, "count": len(history)}