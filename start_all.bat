@echo off
echo ============================================
echo   🎓 Starting EduAI (All Services)
echo ============================================

REM Activate venv
call .venv\Scripts\activate

REM Start MLflow
start cmd /k "call .venv\Scripts\activate && mlflow server --host 0.0.0.0 --port 5001 --backend-store-uri ./mlflow_data/mlruns --default-artifact-root ./mlflow_data/artifacts"

REM Start Backend
start cmd /k "call .venv\Scripts\activate && uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload"

REM Start Frontend
start cmd /k "call .venv\Scripts\activate && streamlit run frontend/Home.py --server.port 8501"

echo.
echo ✅ All services started!
echo 🌐 Frontend: http://localhost:8501
echo 🔌 Backend:  http://localhost:8000/docs
echo 📊 MLflow:   http://localhost:5001
echo.
pause