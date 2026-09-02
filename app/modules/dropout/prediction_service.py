"""
Prediction logic for the Dropout Risk Predictor. Training lives in
model.py (already auto-trains on startup); this module only loads those
artifacts and runs inference.
"""
import io
import csv
from datetime import datetime

import pandas as pd

from app.extensions import db
from .model import FEATURE_NAMES, load_models

FEATURE_FORM_SPEC = [
    {"name": "age", "label": "Age", "type": "slider", "min": 15, "max": 22, "default": 18},
    {"name": "study_time", "label": "Weekly Study Time (1=<2hrs, 4=10hrs+)", "type": "slider", "min": 1, "max": 4, "default": 2},
    {"name": "past_failures", "label": "Past Class Failures", "type": "slider", "min": 0, "max": 3, "default": 0},
    {"name": "absences", "label": "Absences (days this term)", "type": "slider", "min": 0, "max": 30, "default": 2},
    {"name": "g1_grade", "label": "First Term Grade (0-20)", "type": "slider", "min": 0, "max": 20, "default": 12},
    {"name": "g2_grade", "label": "Second Term Grade (0-20)", "type": "slider", "min": 0, "max": 20, "default": 12},
    {"name": "sex", "label": "Sex", "type": "select", "options": [("0", "Female"), ("1", "Male")]},
    {"name": "address", "label": "Address Type", "type": "select", "options": [("0", "Urban"), ("1", "Rural")]},
    {"name": "school_support", "label": "Extra School Support", "type": "select", "options": [("0", "No"), ("1", "Yes")]},
    {"name": "family_support", "label": "Family Support", "type": "select", "options": [("0", "No"), ("1", "Yes")]},
    {"name": "free_time", "label": "Free Time After School (1=low, 4=high)", "type": "slider", "min": 1, "max": 4, "default": 3},
    {"name": "social_activity", "label": "Social Activity (1=low, 4=high)", "type": "slider", "min": 1, "max": 4, "default": 3},
    {"name": "health", "label": "Health Status (1=poor, 4=excellent)", "type": "slider", "min": 1, "max": 4, "default": 3},
    {"name": "family_relationships", "label": "Family Relationship Quality (1=poor, 4=excellent)", "type": "slider", "min": 1, "max": 4, "default": 4},
]

RECOMMENDATIONS = {
    "High": [
        "Meet with your academic advisor this week to make a plan",
        "Consider joining or forming a peer study group",
        "If stress is a factor, the Wellness Companion is a good place to start",
        "Prioritise attendance - it's one of the strongest predictors of staying on track",
        "Talk to family or a trusted mentor about what support could help",
    ],
    "Medium": [
        "Check in with a teacher about topics you're finding difficult",
        "Set a realistic weekly study schedule and stick to it",
        "Small, consistent attendance improvements add up fast",
        "Use the Doubt Solver for quick help between classes",
        "Revisit your last two term grades and target the weakest area first",
    ],
    "Low": [
        "Keep up your current study habits - they're working",
        "Consider mentoring a peer who's struggling; teaching reinforces learning",
        "Use spare time to get ahead on upcoming topics",
        "Maintain your attendance streak",
        "Set a stretch goal for next term",
    ],
}


def _risk_level(probability: float) -> str:
    if probability >= 0.6:
        return "High"
    if probability >= 0.3:
        return "Medium"
    return "Low"


def _build_row(features: dict) -> pd.DataFrame:
    row = {name: float(features.get(name, 0)) for name in FEATURE_NAMES}
    return pd.DataFrame([row], columns=FEATURE_NAMES)


def _top_factors(rf_model, selector_bundle, n: int = 3) -> list[str]:
    selector = selector_bundle["selector"]
    selected_mask = selector.get_support()
    selected_names = [name for name, keep in zip(FEATURE_NAMES, selected_mask) if keep]
    importances = rf_model.feature_importances_
    ranked = sorted(zip(selected_names, importances), key=lambda x: x[1], reverse=True)
    return [name.replace("_", " ").title() for name, _ in ranked[:n]]


def predict(app, features: dict, model_choice: str = "rf") -> dict:
    rf, lr, selector_bundle = load_models(app)
    model = rf if model_choice == "rf" else lr

    X = _build_row(features)
    X_scaled = selector_bundle["scaler"].transform(X)
    X_selected = selector_bundle["selector"].transform(X_scaled)

    probability = float(model.predict_proba(X_selected)[0][1])
    risk_level = _risk_level(probability)
    top_factors = _top_factors(rf, selector_bundle) if model_choice == "rf" else []

    return {
        "probability": round(probability * 100, 1),
        "risk_level": risk_level,
        "top_factors": top_factors,
        "recommendations": RECOMMENDATIONS[risk_level],
        "model_used": "Random Forest" if model_choice == "rf" else "Logistic Regression",
    }


def save_prediction(user_id: str, features: dict, result: dict) -> None:
    db.dropout_predictions.insert_one({
        "user_id": user_id,
        "features": features,
        "probability": result["probability"],
        "risk_level": result["risk_level"],
        "model_used": result["model_used"],
        "created_at": datetime.utcnow(),
    })


def batch_predict(app, csv_stream) -> dict:
    df = pd.read_csv(csv_stream)
    missing = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")

    results = []
    for _, row in df.iterrows():
        features = {name: row[name] for name in FEATURE_NAMES}
        result = predict(app, features, model_choice="rf")
        results.append({**{"student_id": row.get("student_id", "")}, **features, **{
            "probability": result["probability"], "risk_level": result["risk_level"],
        }})

    risk_counts = {"Low": 0, "Medium": 0, "High": 0}
    for r in results:
        risk_counts[r["risk_level"]] += 1

    return {"results": results, "risk_counts": risk_counts, "total": len(results)}


def batch_results_to_csv(results: list[dict]) -> str:
    if not results:
        return ""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(results[0].keys()))
    writer.writeheader()
    writer.writerows(results)
    return buf.getvalue()


def get_latest_prediction(user_id: str) -> dict | None:
    return db.dropout_predictions.find_one({"user_id": user_id}, sort=[("created_at", -1)])
