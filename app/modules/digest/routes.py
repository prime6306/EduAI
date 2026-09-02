from datetime import timedelta

from flask import Blueprint, render_template, redirect, url_for, jsonify, request, flash
from flask_login import login_required, current_user

from app.auth.utils import role_required
from app.utils.audit import log_action
from app.extensions import logger, db
from . import digest_service

digest_bp = Blueprint("digest", __name__, url_prefix="/digest")
digest_api_bp = Blueprint("digest_api", __name__, url_prefix="/api/digest")


@digest_bp.route("")
@login_required
@role_required("teacher")
def index():
    return redirect(url_for("digest.history"))


@digest_bp.route("/history")
@login_required
@role_required("teacher")
def history():
    digests = digest_service.list_digests(current_user.id)
    email_enabled = digest_service.is_email_enabled(current_user.id)
    return render_template(
        "digest/history.html", digests=digests, email_enabled=email_enabled, timedelta=timedelta
    )


@digest_bp.route("/<digest_id>")
@login_required
@role_required("teacher")
def view(digest_id):
    doc = digest_service.get_digest(digest_id, current_user.id)
    if not doc:
        flash("That digest could not be found.", "danger")
        return redirect(url_for("digest.history"))
    return render_template("digest/view.html", digest=doc, timedelta=timedelta)


@digest_api_bp.route("/send-now", methods=["POST"])
@login_required
@role_required("teacher")
def send_now():
    week_start, week_end = digest_service.get_last_completed_week()
    # Manual trigger always produces a fresh digest for the last completed
    # week, even if the scheduled job already generated one.
    db.digest_reports.delete_many({"teacher_id": current_user.id, "week_start": week_start})
    try:
        doc = digest_service.generate_and_save(
            current_user.id, current_user.email, week_start, week_end
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Manual digest generation failed")
        return jsonify({"error": f"Couldn't generate the digest: {exc}"}), 500

    log_action("digest.sent_now", {"digest_id": str(doc["_id"])})
    return jsonify({
        "generated": True,
        "email_sent": doc["email_sent"],
        "redirect_url": url_for("digest.view", digest_id=str(doc["_id"])),
    })


@digest_api_bp.route("/toggle-email", methods=["POST"])
@login_required
@role_required("teacher")
def toggle_email():
    data = request.get_json(silent=True) or {}
    enabled = bool(data.get("enabled", True))
    digest_service.set_email_enabled(current_user.id, enabled)
    log_action("digest.email_toggled", {"enabled": enabled})
    return jsonify({"enabled": enabled})
