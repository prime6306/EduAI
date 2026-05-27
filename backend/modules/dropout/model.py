"""
modules/dropout/model.py
Student dropout risk classifier.
Rebuilt from scratch using sklearn (Logistic Regression + Random Forest).
Trained on synthetic UCI-style student data if no model exists.
"""
import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from backend.config import get_settings
from backend.db.mongodb import get_sync_db
from backend.modules.mlflow_tracker import log_run

MODEL_DIR = "./models"
LR_PATH = os.path.join(MODEL_DIR, "dropout_lr.pkl")
RF_PATH = os.path.join(MODEL_DIR, "dropout_rf.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "dropout_scaler.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "dropout_encoders.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "dropout_features.pkl")

FEATURE_COLS = [
    "age", "studytime", "failures", "absences", "G1", "G2",
    "sex", "address", "schoolsup", "famsup",
    "freetime", "goout", "health", "famrel",
]
TARGET_COL = "dropout"
CAT_COLS = ["sex", "address", "schoolsup", "famsup"]


def _generate_synthetic_data(n: int = 1000) -> pd.DataFrame:
    """Generate realistic synthetic student data for training."""
    np.random.seed(42)
    df = pd.DataFrame({
        "age": np.random.randint(15, 22, n),
        "studytime": np.random.randint(1, 5, n),
        "failures": np.random.randint(0, 4, n),
        "absences": np.random.randint(0, 30, n),
        "G1": np.random.randint(0, 20, n),
        "G2": np.random.randint(0, 20, n),
        "sex": np.random.choice(["M", "F"], n),
        "address": np.random.choice(["U", "R"], n),
        "schoolsup": np.random.choice(["yes", "no"], n),
        "famsup": np.random.choice(["yes", "no"], n),
        "freetime": np.random.randint(1, 6, n),
        "goout": np.random.randint(1, 6, n),
        "health": np.random.randint(1, 6, n),
        "famrel": np.random.randint(1, 6, n),
    })
    # Dropout probability — correlated with grades, absences, failures
    score = (
        -0.3 * df["G1"] - 0.3 * df["G2"]
        + 0.4 * df["failures"]
        + 0.15 * df["absences"]
        - 0.2 * df["studytime"]
        + 0.1 * df["goout"]
        + np.random.randn(n) * 2
    )
    df[TARGET_COL] = (score > score.quantile(0.35)).astype(int)
    return df


def _preprocess(df: pd.DataFrame):
    encoders = {}
    for col in CAT_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    return df, encoders


def train():
    """Train models, save artifacts, log to MLflow."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = _generate_synthetic_data(1500)
    df, encoders = _preprocess(df)

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    selector = SelectKBest(chi2, k=10)
    X_sel = selector.fit_transform(X_scaled, y)

    X_train, X_test, y_train, y_test = train_test_split(X_sel, y, test_size=0.2, random_state=42)

    lr = LogisticRegression(max_iter=500, random_state=42)
    lr.fit(X_train, y_train)
    lr_acc = accuracy_score(y_test, lr.predict(X_test))

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test))

    joblib.dump(lr, LR_PATH)
    joblib.dump(rf, RF_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(encoders, ENCODER_PATH)
    joblib.dump(selector.get_support(indices=True), FEATURES_PATH)

    log_run("dropout_training",
            params={"model": "LR+RF", "n_samples": 1500, "features": 10},
            metrics={"lr_accuracy": lr_acc, "rf_accuracy": rf_acc})
    print(f"[Dropout] LR acc: {lr_acc:.3f} | RF acc: {rf_acc:.3f}")
    return {"lr_accuracy": lr_acc, "rf_accuracy": rf_acc}


def _ensure_models():
    if not os.path.exists(LR_PATH):
        print("[Dropout] Models not found — training now...")
        train()


def predict_dropout(features: dict, model_type: str = "rf") -> dict:
    """
    features: dict with keys matching FEATURE_COLS.
    Returns risk label, probability, and key risk factors.
    """
    _ensure_models()
    scaler = joblib.load(SCALER_PATH)
    encoders = joblib.load(ENCODER_PATH)
    feature_idx = joblib.load(FEATURES_PATH)
    model = joblib.load(RF_PATH if model_type == "rf" else LR_PATH)

    row = {}
    for col in FEATURE_COLS:
        val = features.get(col, 0)
        if col in CAT_COLS and col in encoders:
            le = encoders[col]
            try:
                val = le.transform([str(val)])[0]
            except Exception:
                val = 0
        row[col] = val

    X = np.array([[row[c] for c in FEATURE_COLS]], dtype=float)
    X_scaled = scaler.transform(X)
    X_sel = X_scaled[:, feature_idx]

    prob = model.predict_proba(X_sel)[0]
    pred = int(model.predict(X_sel)[0])

    # Risk factors (top features by importance for RF)
    risk_factors = []
    if model_type == "rf":
        importances = model.feature_importances_
        top = np.argsort(-importances)[:3]
        sel_cols = [FEATURE_COLS[i] for i in feature_idx]
        risk_factors = [sel_cols[i] for i in top if i < len(sel_cols)]

    # Save to MongoDB
    db = get_sync_db()
    db.dropout_predictions.insert_one({
        "features": features,
        "prediction": pred,
        "dropout_probability": round(float(prob[1]), 3),
        "risk_level": "High" if prob[1] > 0.6 else "Medium" if prob[1] > 0.35 else "Low",
        "risk_factors": risk_factors,
        "model_type": model_type,
        "timestamp": datetime.utcnow(),
    })

    return {
        "prediction": pred,
        "dropout_probability": round(float(prob[1]), 3),
        "safe_probability": round(float(prob[0]), 3),
        "risk_level": "High" if prob[1] > 0.6 else "Medium" if prob[1] > 0.35 else "Low",
        "risk_factors": risk_factors,
        "model": model_type,
    }
