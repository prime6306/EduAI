import os
import uuid
from datetime import datetime

from flask import (
    Blueprint, render_template, request, jsonify, redirect, url_for, flash,
    current_app, send_from_directory,
)
from flask_login import login_required, current_user

from app.auth.utils import role_required
from app.utils.audit import log_action
from app.extensions import db
from . import announcement_service as svc

announcements_bp = Blueprint("announcements", __name__, url_prefix="/announcements")
announcements_api_bp = Blueprint("announcements_api", __name__, url_prefix="/api/announcements")

ALLOWED_ATTACHMENT_EXT = {".pdf", ".docx", ".xlsx", ".jpg", ".jpeg", ".png"}


def _parse_datetime(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


@announcements_bp.route("")
@login_required
def index():
    if current_user.is_teacher:
        return redirect(url_for("announcements.manage"))

    category = request.args.get("category", "All")
    items = svc.list_for_student(current_user.id, category)
    return render_template(
        "announcements/feed.html", items=items, category=category, categories=svc.CATEGORIES,
    )


@announcements_bp.route("/create", methods=["GET", "POST"])
@login_required
@role_required("teacher")
def create():
    if request.method == "POST":
        title = (request.form.get("title") or "").strip()
        body_html = request.form.get("body_html") or ""
        category = request.form.get("category", "General")
        visibility_type = request.form.get("visibility_type", "all")
        visible_to_students = [s.strip() for s in request.form.getlist("visible_to_students[]") if s.strip()]
        scheduled_for = _parse_datetime(request.form.get("scheduled_for"))
        expires_at = _parse_datetime(request.form.get("expires_at"))

        if not title or not body_html.strip():
            flash("Title and body are required.", "danger")
            return redirect(url_for("announcements.create"))

        attachment_filename = None
        file = request.files.get("attachment")
        if file and file.filename:
            ext = os.path.splitext(file.filename)[1].lower()
            if ext not in ALLOWED_ATTACHMENT_EXT:
                flash("Unsupported attachment type.", "danger")
                return redirect(url_for("announcements.create"))
            stored_name = f"{uuid.uuid4().hex}{ext}"
            dest_dir = os.path.join(current_app.config["UPLOAD_FOLDER"], "announcements")
            os.makedirs(dest_dir, exist_ok=True)
            file.save(os.path.join(dest_dir, stored_name))
            attachment_filename = stored_name

        doc = svc.create_announcement(
            current_user.id, title, body_html, category, visibility_type,
            visible_to_students, scheduled_for, expires_at, attachment_filename,
        )
        log_action("announcements.created", {"announcement_id": str(doc["_id"]), "category": category})
        flash("Announcement published.", "success")
        return redirect(url_for("announcements.manage"))

    students = list(db.students.find({}, {"student_id": 1, "name": 1}))
    return render_template("announcements/create.html", categories=svc.CATEGORIES, students=students)


@announcements_bp.route("/manage")
@login_required
@role_required("teacher")
def manage():
    items = svc.list_for_teacher(current_user.id)
    return render_template("announcements/manage.html", items=items)


@announcements_bp.route("/attachment/<filename>")
@login_required
def attachment(filename):
    dest_dir = os.path.join(current_app.config["UPLOAD_FOLDER"], "announcements")
    return send_from_directory(dest_dir, filename)


@announcements_api_bp.route("/delete/<announcement_id>", methods=["POST"])
@login_required
@role_required("teacher")
def delete(announcement_id):
    ok = svc.delete_announcement(announcement_id, current_user.id)
    if ok:
        log_action("announcements.deleted", {"announcement_id": announcement_id})
    return jsonify({"deleted": ok})


@announcements_api_bp.route("/update/<announcement_id>", methods=["POST"])
@login_required
@role_required("teacher")
def update(announcement_id):
    data = request.get_json(silent=True) or {}
    allowed_fields = {"title", "body_html", "category", "expires_at"}
    fields = {k: v for k, v in data.items() if k in allowed_fields}
    if "expires_at" in fields:
        fields["expires_at"] = _parse_datetime(fields["expires_at"])
    ok = svc.update_announcement(announcement_id, current_user.id, **fields)
    if ok:
        log_action("announcements.updated", {"announcement_id": announcement_id})
    return jsonify({"updated": ok})


@announcements_api_bp.route("/read/<announcement_id>", methods=["POST"])
@login_required
def read(announcement_id):
    ok = svc.mark_read(announcement_id, current_user.id)
    return jsonify({"read": ok})


@announcements_api_bp.route("/unread-count")
@login_required
def unread_count():
    if current_user.is_teacher:
        return jsonify({"count": 0})
    return jsonify({"count": svc.unread_count(current_user.id)})
