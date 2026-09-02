import os
import uuid
from datetime import datetime

from flask import Blueprint, render_template, request, jsonify, redirect, url_for, flash, current_app, send_from_directory
from flask_login import login_required, current_user

from app.auth.utils import role_required
from app.extensions import db
from app.utils.audit import log_action
from . import corrections_service as svc

corrections_bp = Blueprint("corrections", __name__, url_prefix="/attendance/corrections")
corrections_api_bp = Blueprint("corrections_api", __name__, url_prefix="/api/corrections")

ALLOWED_PROOF_EXT = {".jpg", ".jpeg", ".png", ".pdf"}


@corrections_bp.route("/new", methods=["GET", "POST"])
@login_required
def new():
    if current_user.is_teacher:
        flash("Teachers don't submit correction requests.", "info")
        return redirect(url_for("attendance.index"))
    if not current_user.student_id:
        flash("Add your Student ID in your profile first.", "danger")
        return redirect(url_for("auth.profile"))

    if request.method == "POST":
        date_str = request.form.get("requested_date")
        reason_category = request.form.get("reason_category", "Other")
        explanation = request.form.get("explanation", "")

        try:
            requested_date = datetime.strptime(date_str, "%Y-%m-%d")
        except (ValueError, TypeError):
            flash("Please provide a valid date.", "danger")
            return redirect(url_for("corrections.new"))

        proof_filename = None
        proof = request.files.get("proof")
        if proof and proof.filename:
            ext = os.path.splitext(proof.filename)[1].lower()
            if ext not in ALLOWED_PROOF_EXT:
                flash("Unsupported proof file type.", "danger")
                return redirect(url_for("corrections.new"))
            stored_name = f"{uuid.uuid4().hex}{ext}"
            dest_dir = os.path.join(current_app.config["UPLOAD_FOLDER"], "corrections")
            os.makedirs(dest_dir, exist_ok=True)
            proof.save(os.path.join(dest_dir, stored_name))
            proof_filename = stored_name

        try:
            doc = svc.submit_correction(
                current_user.student_id, requested_date, reason_category, explanation, proof_filename,
            )
        except ValueError as exc:
            flash(str(exc), "danger")
            return redirect(url_for("corrections.new"))

        log_action("corrections.submitted", {"correction_id": str(doc["_id"])})
        flash("Correction request submitted. Your teacher will review it soon.", "success")
        return redirect(url_for("corrections.my_requests"))

    return render_template("attendance/correction_new.html", reasons=svc.REASON_CATEGORIES)


@corrections_bp.route("")
@login_required
def my_requests():
    if current_user.is_teacher:
        return redirect(url_for("corrections.queue"))
    requests_list = svc.list_for_student(current_user.student_id) if current_user.student_id else []
    for r in requests_list:
        if not r.get("seen_by_student", True):
            svc.mark_seen(str(r["_id"]), current_user.student_id)
    return render_template("attendance/correction_my_requests.html", requests=requests_list)


@corrections_bp.route("/queue")
@login_required
@role_required("teacher")
def queue():
    status = request.args.get("status", "Pending")
    items = svc.list_queue(status=status if status != "All" else None)
    students = {s["student_id"]: s["name"] for s in db.students.find({}, {"student_id": 1, "name": 1})}
    return render_template("attendance/correction_queue.html", items=items, status=status, students=students, reasons=svc.REASON_CATEGORIES)


@corrections_bp.route("/proof/<filename>")
@login_required
@role_required("teacher")
def proof(filename):
    dest_dir = os.path.join(current_app.config["UPLOAD_FOLDER"], "corrections")
    return send_from_directory(dest_dir, filename)


@corrections_api_bp.route("/approve/<correction_id>", methods=["POST"])
@login_required
@role_required("teacher")
def approve(correction_id):
    ok = svc.approve(correction_id, current_user.id)
    if ok:
        log_action("corrections.approved", {"correction_id": correction_id})
    return jsonify({"approved": ok})


@corrections_api_bp.route("/reject/<correction_id>", methods=["POST"])
@login_required
@role_required("teacher")
def reject(correction_id):
    data = request.get_json(silent=True) or {}
    try:
        ok = svc.reject(correction_id, current_user.id, data.get("reason", ""))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    if ok:
        log_action("corrections.rejected", {"correction_id": correction_id})
    return jsonify({"rejected": ok})


@corrections_api_bp.route("/bulk-approve", methods=["POST"])
@login_required
@role_required("teacher")
def bulk_approve():
    data = request.get_json(silent=True) or {}
    count = svc.bulk_approve(data.get("ids", []), current_user.id)
    log_action("corrections.bulk_approved", {"count": count})
    return jsonify({"approved_count": count})


@corrections_api_bp.route("/bulk-reject", methods=["POST"])
@login_required
@role_required("teacher")
def bulk_reject():
    data = request.get_json(silent=True) or {}
    count = svc.bulk_reject(data.get("ids", []), current_user.id, data.get("reason", "Bulk rejected"))
    log_action("corrections.bulk_rejected", {"count": count})
    return jsonify({"rejected_count": count})


@corrections_api_bp.route("/seen/<correction_id>", methods=["POST"])
@login_required
def mark_seen(correction_id):
    svc.mark_seen(correction_id, current_user.student_id)
    return jsonify({"seen": True})


@corrections_api_bp.route("/message/<correction_id>", methods=["POST"])
@login_required
def message(correction_id):
    data = request.get_json(silent=True) or {}
    content = (data.get("content") or "").strip()
    if not content:
        return jsonify({"error": "Message is required."}), 400
    sender = "teacher" if current_user.is_teacher else "student"
    ok = svc.add_message(correction_id, sender, content)
    return jsonify({"sent": ok})
