#!/bin/bash
# EduAI Local Development Startup Script
# Usage: chmod +x scripts/start_dev.sh && ./scripts/start_dev.sh

set -e
cd "$(dirname "$0")/.."

echo "═══════════════════════════════════════════════"
echo "  🎓 EduAI — Starting Development Environment"
echo "═══════════════════════════════════════════════"

# Check .env
if [ ! -f ".env" ]; then
    echo "⚠️  .env not found. Copying from .env.example..."
    cp .env.example .env
    echo "📝 Edit .env with your actual credentials before continuing."
    exit 1
fi

# Check Python
python3 --version || { echo "❌ Python 3 not found"; exit 1; }

# Create virtual env if needed
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

# Install deps
echo "📦 Installing dependencies..."
pip install -r requirements.txt -q

# Setup DB (indexes + seed)
echo "🗄️  Setting up MongoDB..."
python scripts/setup_db.py

# Train dropout model if needed
echo "🤖 Checking dropout model..."
python -c "
import sys; sys.path.insert(0, '.')
from backend.modules.dropout.model import _ensure_models
_ensure_models()
print('✅ Dropout model ready')
"

# Start MLflow in background
echo "📊 Starting MLflow server on port 5001..."
mlflow server \
    --host 0.0.0.0 \
    --port 5001 \
    --backend-store-uri ./mlflow_data/mlruns \
    --default-artifact-root ./mlflow_data/artifacts &
MLFLOW_PID=$!
echo "  MLflow PID: $MLFLOW_PID"

sleep 2

# Start FastAPI backend in background
echo "🚀 Starting FastAPI backend on port 8000..."
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
echo "  Backend PID: $BACKEND_PID"

sleep 3

# Start Streamlit frontend
echo "🌐 Starting Streamlit frontend on port 8501..."
echo ""
echo "═══════════════════════════════════════════════"
echo "  ✅ EduAI is running!"
echo ""
echo "  🌐 Frontend:  http://localhost:8501"
echo "  🔌 Backend:   http://localhost:8000/docs"
echo "  📊 MLflow:    http://localhost:5001"
echo "═══════════════════════════════════════════════"
echo ""

streamlit run frontend/Home.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --browser.gatherUsageStats false

# Cleanup on exit
trap "kill $MLFLOW_PID $BACKEND_PID 2>/dev/null; echo 'Servers stopped.'" EXIT
