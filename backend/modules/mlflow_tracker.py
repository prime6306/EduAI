"""
modules/mlflow_tracker.py – Centralised MLflow logging helper.
Every module calls log_run() to record params + metrics + artifacts.
"""
import mlflow
from backend.config import get_settings


def init_mlflow():
    s = get_settings()
    mlflow.set_tracking_uri(s.mlflow_tracking_uri)
    mlflow.set_experiment(s.mlflow_experiment)


def log_run(
    run_name: str,
    params: dict | None = None,
    metrics: dict | None = None,
    tags: dict | None = None,
    artifact_paths: list[str] | None = None,
):
    """
    Log a single run to MLflow.
    Safe to call even if MLflow server is not running (warns but doesn't crash).
    """
    try:
        init_mlflow()
        with mlflow.start_run(run_name=run_name):
            if params:
                mlflow.log_params(params)
            if metrics:
                mlflow.log_metrics(metrics)
            if tags:
                mlflow.set_tags(tags)
            if artifact_paths:
                for p in artifact_paths:
                    mlflow.log_artifact(p)
    except Exception as e:
        print(f"[MLflow] Warning: could not log run '{run_name}': {e}")
