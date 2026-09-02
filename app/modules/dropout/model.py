"""
Dropout Risk Predictor — model training/loading.

Trains two classifiers on synthetic, UCI "student performance"-style data
(the public UCI dataset's schema, not its rows — we generate our own 1500
synthetic samples so the app has no external data dependency):
  - Random Forest (100 estimators) — primary
  - Logistic Regression — secondary comparison
14 raw features, SelectKBest(chi2, k=10) for feature selection.

Auto-trains on first app startup if the joblib artifacts are absent, and
never re-trains on every boot after that (checks file existence first).
Retraining is also exposed on-demand via POST /api/dropout/retrain in the
routes module (added in the Dropout build phase).
"""
import os
import numpy as np
import pandas as pd
from joblib import dump, load

FEATURE_NAMES = [
    "age", "study_time", "past_failures", "absences", "g1_grade", "g2_grade",
    "sex", "address", "school_support", "family_support", "free_time",
    "social_activity", "health", "family_relationships",
]


def _generate_synthetic_dataset(n_samples: int = 1500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.integers(15, 22, n_samples),
        "study_time": rng.integers(1, 5, n_samples),
        "past_failures": rng.integers(0, 4, n_samples),
        "absences": rng.integers(0, 30, n_samples),
        "g1_grade": rng.integers(0, 20, n_samples),
        "g2_grade": rng.integers(0, 20, n_samples),
        "sex": rng.integers(0, 2, n_samples),
        "address": rng.integers(0, 2, n_samples),
        "school_support": rng.integers(0, 2, n_samples),
        "family_support": rng.integers(0, 2, n_samples),
        "free_time": rng.integers(1, 5, n_samples),
        "social_activity": rng.integers(1, 5, n_samples),
        "health": rng.integers(1, 5, n_samples),
        "family_relationships": rng.integers(1, 5, n_samples),
    })

    # Synthetic label: higher failures/absences/low grades/low support -> higher dropout risk.
    risk_score = (
        0.18 * df["past_failures"]
        + 0.05 * df["absences"]
        - 0.12 * df["g1_grade"]
        - 0.12 * df["g2_grade"]
        - 0.4 * df["family_support"]
        - 0.3 * df["school_support"]
        + 0.1 * (5 - df["study_time"])
        + rng.normal(0, 1.5, n_samples)
    )
    threshold = np.percentile(risk_score, 70)
    df["dropout"] = (risk_score > threshold).astype(int)
    return df


def _artifacts_exist(app) -> bool:
    return (
        os.path.exists(app.config["DROPOUT_RF_MODEL_PATH"])
        and os.path.exists(app.config["DROPOUT_LR_MODEL_PATH"])
        and os.path.exists(app.config["DROPOUT_SELECTOR_PATH"])
    )


def train_and_save(app) -> dict:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_selection import SelectKBest, chi2
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import MinMaxScaler

    df = _generate_synthetic_dataset()
    X = df[FEATURE_NAMES]
    y = df["dropout"]

    # chi2 requires non-negative features.
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    selector = SelectKBest(chi2, k=10)
    X_selected = selector.fit_transform(X_scaled, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test))

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    lr_acc = accuracy_score(y_test, lr.predict(X_test))

    os.makedirs(os.path.dirname(app.config["DROPOUT_RF_MODEL_PATH"]), exist_ok=True)
    dump(rf, app.config["DROPOUT_RF_MODEL_PATH"])
    dump(lr, app.config["DROPOUT_LR_MODEL_PATH"])
    dump(
        {"selector": selector, "scaler": scaler, "feature_names": FEATURE_NAMES},
        app.config["DROPOUT_SELECTOR_PATH"],
    )

    metrics = {"rf_accuracy": rf_acc, "lr_accuracy": lr_acc}

    try:
        import mlflow
        mlflow.set_tracking_uri(app.config["MLFLOW_TRACKING_URI"])
        mlflow.set_experiment(app.config["MLFLOW_EXPERIMENT"])
        with mlflow.start_run(run_name="dropout_model_training"):
            mlflow.log_metric("rf_accuracy", rf_acc)
            mlflow.log_metric("lr_accuracy", lr_acc)
            mlflow.log_param("n_samples", len(df))
    except Exception:  # noqa: BLE001
        app.logger.info("MLflow not reachable — skipping dropout training run log.")

    app.logger.info(
        "Dropout models trained: RF acc=%.3f, LR acc=%.3f", rf_acc, lr_acc
    )
    return metrics


def ensure_models_trained(app) -> None:
    if _artifacts_exist(app):
        app.logger.info("Dropout models already present — skipping training.")
        return
    app.logger.info("Dropout model artifacts not found — auto-training on startup...")
    train_and_save(app)


def load_models(app):
    return (
        load(app.config["DROPOUT_RF_MODEL_PATH"]),
        load(app.config["DROPOUT_LR_MODEL_PATH"]),
        load(app.config["DROPOUT_SELECTOR_PATH"]),
    )
