from flask import Blueprint, render_template, request, jsonify, current_app
from flask_login import login_required, current_user

from app.auth.utils import role_required
from app.extensions import logger
from app.utils.audit import log_action
from . import prediction_service as pred
from .model import train_and_save

dropout_bp = Blueprint("dropout", __name__, url_prefix="/dropout")
dropout_api_bp = Blueprint("dropout_api", __name__, url_prefix="/api/dropout")


@dropout_bp.route("", methods=["GET"])
@login_required
def index():
    latest = pred.get_latest_prediction(current_user.id)
    return render_template(
        "dropout/index.html",
        feature_spec=pred.FEATURE_FORM_SPEC,
        latest=latest,
    )


@dropout_api_bp.route("/predict", methods=["POST"])
@login_required
def predict():
    data = request.get_json(silent=True) or {}
    model_choice = data.get("model_choice", "rf")
    features = {f["name"]: data.get(f["name"]) for f in pred.FEATURE_FORM_SPEC}

    missing = [k for k, v in features.items() if v is None]
    if missing:
        return jsonify({"error": f"Missing values for: {', '.join(missing)}"}), 400

    try:
        result = pred.predict(current_app._get_current_object(), features, model_choice)
    except Exception:  # noqa: BLE001
        logger.exception("Dropout prediction failed")
        return jsonify({"error": "Prediction failed. Please try again."}), 500

    pred.save_prediction(current_user.id, features, result)
    log_action("dropout.predicted", {"risk_level": result["risk_level"]})
    return jsonify(result)


@dropout_api_bp.route("/batch", methods=["POST"])
@login_required
@role_required("teacher")
def batch():
    file = request.files.get("csv_file")
    if not file:
        return jsonify({"error": "No CSV file provided."}), 400

    try:
        result = pred.batch_predict(current_app._get_current_object(), file.stream)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:  # noqa: BLE001
        logger.exception("Batch dropout prediction failed")
        return jsonify({"error": "Couldn't process that CSV. Check the format and try again."}), 500

    result["csv"] = pred.batch_results_to_csv(result["results"])
    log_action("dropout.batch_predicted", {"total": result["total"]})
    return jsonify(result)


@dropout_api_bp.route("/retrain", methods=["POST"])
@login_required
@role_required("teacher")
def retrain():
    try:
        metrics = train_and_save(current_app._get_current_object())
    except Exception:  # noqa: BLE001
        logger.exception("Dropout retrain failed")
        return jsonify({"error": "Retraining failed. Please try again."}), 500

    log_action("dropout.retrained", metrics)
    return jsonify(metrics)
