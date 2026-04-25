#!/bin/bash

echo "============================================"
echo "  Federated Fraud Detection — Demo Startup"
echo "============================================"

# Check conda environment
if [[ "$CONDA_DEFAULT_ENV" != "fraud-detection" ]]; then
    echo "Activating fraud-detection environment..."
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate fraud-detection
fi

# Check model exists
if [ ! -f "data/models/global_model_latest.pt" ]; then
    echo "ERROR: No trained model found."
    echo "Run: cd src/federated && python train.py"
    exit 1
fi

# Check data exists
if [ ! -f "data/clients/client_0.csv" ]; then
    echo "ERROR: No client data found."
    echo "Run: cd src/simulator && python generate.py"
    exit 1
fi

echo ""
echo "Starting FastAPI backend on port 8000..."
cd api
uvicorn main:app --port 8000 &
API_PID=$!
cd ..

echo "Waiting for API to initialize (SHAP takes ~30s)..."
sleep 35

echo ""
echo "Starting Streamlit dashboard on port 8501..."
cd dashboard
streamlit run app.py --server.port 8501 &
DASH_PID=$!
cd ..

echo ""
echo "============================================"
echo "  System running"
echo "  API:       http://localhost:8000"
echo "  Dashboard: http://localhost:8501"
echo "  API docs:  http://localhost:8000/docs"
echo ""
echo "  Press Ctrl+C to stop everything"
echo "============================================"

# Keep running and kill both on Ctrl+C
trap "echo 'Stopping...'; kill $API_PID $DASH_PID 2>/dev/null; exit" INT
wait