# Federated Fraud Detection with Explainable AI

A production-style federated learning system for real-time fraud detection,
built on 590,540 real transactions from the IEEE-CIS dataset.
Every prediction is explainable via SHAP feature attributions.
Concept drift is monitored continuously and triggers automatic retraining.

---

## Architecture overview

    5 Bank Clients (IEEE-CIS partitions)
            │
            │  local training (no raw data shared)
            ▼
      FedAvg Server  ──►  Global Model (PyTorch MLP)
            │
            ▼
      FastAPI Scoring Engine
            │
            ├──►  /predict      — fraud probability + risk level
            ├──►  /explain      — SHAP feature attributions per prediction
            └──►  /drift/status — concept drift monitor
                    │
                    ▼
            SQLite Prediction Log
                    │
                    ▼
            Streamlit Dashboard
            ├── Live feed      — real-time scoring + SHAP waterfall
            ├── Model health   — AUC/F1 across federated rounds
            ├── Drift monitor  — DDM + sliding window + SHAP drift
            └── Federated view — per-client loss convergence

## Key features

**Federated learning**
- 5 simulated bank clients train collaboratively without sharing raw data
- FedAvg aggregation weighted by dataset size
- Raw transactions never leave each client partition
- GDPR-compatible architecture — only float32 weight tensors exchanged

**Explainable AI**
- SHAP DeepExplainer wraps the PyTorch model
- Every prediction returns ranked feature attributions
- Natural language explanation generated per transaction
- Explanation drift tracked via Jensen-Shannon divergence

**Concept drift monitoring**
- DDM (Drift Detection Method) monitors prediction error stream
- Sliding window detector tracks AUC and fraud rate shifts
- SHAP drift detector catches explanation distribution changes
- Automatic federated retraining triggered on drift detection

**Real-time scoring**
- FastAPI endpoint scores transactions in under 50ms
- Adjustable decision threshold via query parameter
- Batch scoring endpoint (up to 100 transactions)
- Full prediction audit log in SQLite

---

## Dataset

IEEE-CIS Fraud Detection dataset (Kaggle)
- 590,540 real transactions from Vesta Corporation
- 394 original features — 173 used after cleaning
- 3.5% fraud rate
- Partitioned into 5 clients with fraud rates from 2% to 4.8%

Download from:
https://www.kaggle.com/competitions/ieee-fraud-detection/data

Required files: `train_transaction.csv`, `train_identity.csv`

---

## Project structure

    federated-fraud-detection/
    ├── data/
    │   ├── raw/                    # IEEE-CIS CSVs
    │   ├── clients/                # 5 client partitions + global test
    │   ├── models/                 # federated model checkpoints
    │   └── processed/              # scaler.pkl + predictions.db
    ├── src/
    │   ├── simulator/              # data pipeline
    │   ├── models/                 # PyTorch FraudMLP
    │   ├── federated/              # FedAvg training
    │   └── drift/                  # drift detection + retraining
    ├── api/                        # FastAPI scoring engine
    ├── dashboard/                  # Streamlit UI
    │   └── tabs/                   # live feed, model health, drift, federated
    ├── logs/                       # round_history.json
    ├── requirements.txt
    ├── run_demo.sh
    └── README.md

## Setup

### 1. Clone and create environment

```bash
git clone https://github.com/YOUR_USERNAME/federated-fraud-detection.git
cd federated-fraud-detection

conda create -n fraud-detection python=3.11 -y
conda activate fraud-detection
pip install -r requirements.txt
```

### 2. Download the dataset

Download from Kaggle:
https://www.kaggle.com/competitions/ieee-fraud-detection/data

Place these two files in `data/raw/`:
- `train_transaction.csv`
- `train_identity.csv`

### 3. Generate client partitions

```bash
cd src/simulator
python generate.py
python validate.py
cd ../..
```

### 4. Train the federated model

```bash
cd src/federated
python train.py
cd ../..
```

Training takes 10-20 minutes. Runs 10 federated rounds across 5 clients.

### 5. Start the system

```bash
bash run_demo.sh
```

This starts both the FastAPI backend and Streamlit dashboard.

- API: http://localhost:8000
- Dashboard: http://localhost:8501
- API docs: http://localhost:8000/docs

---

## API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | API and model status |
| GET | /model-info | Current model metrics |
| POST | /predict | Score a transaction |
| POST | /predict/batch | Score up to 100 transactions |
| POST | /explain | Score + SHAP explanation |
| GET | /explain/features | Global feature list |
| GET | /stats | Live prediction statistics |
| GET | /predictions/recent | Recent prediction log |
| GET | /drift/status | Drift monitor state |
| GET | /drift/events | Drift event log |
| POST | /drift/simulate | Inject a test drift event |
| POST | /retrain | Manually trigger retraining |
| GET | /performance/history | Rolling performance windows |

---

## Model performance

After 10 federated rounds on IEEE-CIS data:

| Metric | Value |
|--------|-------|
| AUC | 0.8805 |
| Recall | 0.7994 |
| F1 | 0.2273 |
| Precision | 0.1325 |

Recall is prioritized over precision — missing a fraud transaction
is more costly than a false alarm in production fraud detection.

---

## Design decisions

**Why federated learning?**
Banks cannot share raw customer transaction data due to privacy
regulations (GDPR, PCI-DSS). Federated learning allows collaborative
model training without centralizing sensitive data.

**Why SHAP per prediction?**
Regulatory frameworks (EU AI Act, GDPR Article 22) require that
automated decisions affecting individuals must be explainable.
SHAP provides a mathematically grounded attribution for every
prediction the model makes.

**Why explanation drift?**
Accuracy metrics lag behind real distribution shifts. Tracking
SHAP importance distributions catches pattern changes before
model performance degrades — an earlier warning signal.

**Why SQLite?**
Zero-dependency setup for a single-node deployment. The database
layer is abstracted behind `database.py` — swapping to PostgreSQL
requires only changing the connection string.

---

## Tech stack

| Component | Technology |
|-----------|------------|
| Federated learning | PyTorch + manual FedAvg |
| Model architecture | MLP (256→128→64→1) |
| Explainability | SHAP DeepExplainer |
| API | FastAPI + Uvicorn |
| Database | SQLite |
| Dashboard | Streamlit + Plotly |
| Drift detection | DDM + sliding window + JS divergence |
| Data | IEEE-CIS (590k transactions) |

---

## Author

Milind K— AI/ML, BMSIT Bangalore 