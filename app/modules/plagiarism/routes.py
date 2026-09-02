import os
import uuid

from flask import Blueprint, render_template, request, flash, current_app
from flask_login import login_required, current_user

from app.auth.utils import role_required
from app.extensions import logger
from app.utils.audit import log_action
from app.utils import rate_limiter
from . import similarity_service

plagiarism_bp = Blueprint("plagiarism", __name__, url_prefix="/plagiarism")


@plagiarism_bp.route("", methods=["GET", "POST"])
@login_required
@role_required("teacher")
def index():
    if request.method != "POST":
        return render_template(
            "plagiarism/index.html", result=None, threshold=similarity_service.DEFAULT_THRESHOLD,
            status=rate_limiter.get_status("plagiarism_check", current_user.id, current_user.role),
            usage_feature_label=rate_limiter.FEATURE_LABELS["plagiarism_check"],
        )

    threshold = float(request.form.get("threshold", similarity_service.DEFAULT_THRESHOLD))
    mode = request.form.get("mode", "files")
    names, texts = [], []

    if mode == "files":
        files = request.files.getlist("files")
        if len(files) < 2:
            flash("Upload at least 2 files to compare.", "danger")
            return render_template("plagiarism/index.html", result=None, threshold=threshold)

        for f in files:
            if not f.filename:
                continue
            tmp_path = os.path.join(current_app.config["UPLOAD_FOLDER"], f"plag_{uuid.uuid4().hex}{os.path.splitext(f.filename)[1]}")
            f.save(tmp_path)
            try:
                text = similarity_service.extract_text_from_upload(tmp_path, f.filename)
                names.append(os.path.splitext(f.filename)[0])
                texts.append(text)
            except ValueError as exc:
                flash(f"Skipped {f.filename}: {exc}", "warning")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
    else:
        paste_names = request.form.getlist("paste_name[]")
        paste_texts = request.form.getlist("paste_text[]")
        for n, t in zip(paste_names, paste_texts):
            if n.strip() and t.strip():
                names.append(n.strip())
                texts.append(t.strip())

    if len(texts) < 2:
        flash("Need at least 2 valid submissions to compare.", "danger")
        return render_template("plagiarism/index.html", result=None, threshold=threshold)

    allowed, rl_status = rate_limiter.check_and_increment("plagiarism_check", current_user.id, current_user.role)
    if not allowed:
        flash(rate_limiter.limit_message("plagiarism_check", rl_status), "warning")
        return render_template("plagiarism/index.html", result=None, threshold=threshold)

    try:
        result = similarity_service.find_suspicious_pairs(names, texts, threshold)
    except ValueError as exc:
        flash(str(exc), "danger")
        return render_template("plagiarism/index.html", result=None, threshold=threshold)
    except Exception:  # noqa: BLE001
        logger.exception("Plagiarism check failed")
        flash("Something went wrong running the comparison. Please try again.", "danger")
        return render_template("plagiarism/index.html", result=None, threshold=threshold)

    log_action("plagiarism.checked", {
        "submissions": result["total_submissions"], "suspicious_pairs": len(result["suspicious_pairs"]),
    })
    return render_template("plagiarism/index.html", result=result, threshold=threshold)
