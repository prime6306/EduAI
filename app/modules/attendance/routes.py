"""
Face Attendance System (/attendance) and Student Management (/students,
teacher-only). Image decoding (face_recognition/PIL) is deferred into the
route bodies so the module still imports cleanly without those native
deps installed - matches the pattern used across face_engine/antispoof.
"""
import os
import uuid

from flask import (
    Blueprint, render_template, request, jsonify, current_app, flash, redirect, url_for, Response,
)
from flask_login import login_required, current_user

from app.auth.forms import BRANCH_CHOICES, YEAR_CHOICES
from app.auth.utils import role_required
from app.extensions import logger
from app.utils.audit import log_action
from app.utils import rate_limiter
from . import face_engine, attendance_service, enrollment_service

attendance_bp = Blueprint("attendance", __name__, url_prefix="/attendance")
attendance_api_bp = Blueprint("attendance_api", __name__, url_prefix="/api/attendance")
students_bp = Blueprint("students", __name__, url_prefix="/students")
students_api_bp = Blueprint("students_api", __name__, url_prefix="/api/students")

STATUS_LABELS = {
    "marked": "Marked",
    "spoof": "Spoof detected",
    "duplicate": "Already marked",
    "unknown": "Unknown face",
    "antispoof_unavailable": "Anti-spoof model unavailable",
}


def _load_image_array(filepath):
    import face_recognition
    return face_recognition.load_image_file(filepath)


def _save_upload(file_storage, prefix="attendance"):
    ext = os.path.splitext(file_storage.filename or "")[1].lower() or ".jpg"
    tmp_name = f"{prefix}_{uuid.uuid4().hex}{ext}"
    tmp_path = os.path.join(current_app.config["UPLOAD_FOLDER"], tmp_name)
    file_storage.save(tmp_path)
    return tmp_path


# ═══════════════════════════════════════════════════════════════════
#  Attendance
# ═══════════════════════════════════════════════════════════════════

@attendance_bp.route("")
@login_required
def index():
    if current_user.is_teacher:
        report = attendance_service.get_report_metrics()
        recent_logs = attendance_service.get_recent_logs(30)
        return render_template("attendance/teacher.html", report=report, recent_logs=recent_logs)

    stats = attendance_service.get_student_stats(current_user.student_id) if current_user.student_id else None
    return render_template("attendance/student.html", stats=stats)


@attendance_api_bp.route("/mark-photo", methods=["POST"])
@login_required
@role_required("teacher")
def mark_photo():
    file = request.files.get("photo")
    if not file or not file.filename:
        return jsonify({"error": "No photo provided."}), 400

    tmp_path = _save_upload(file, "group")
    try:
        image_array = _load_image_array(tmp_path)
        faces = face_engine.detect_faces(image_array)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception:  # noqa: BLE001
        logger.exception("Group photo face detection failed")
        return jsonify({"error": "Couldn't process that photo. Try a clearer image."}), 500
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    if not faces:
        return jsonify({"results": [], "message": "No faces detected in the photo."})

    results = []
    for face in faces:
        result = attendance_service.process_face(face["encoding"], face["crop_160"])
        result["label"] = STATUS_LABELS.get(result["status"], result["status"])
        results.append(result)

    log_action("attendance.group_photo_processed", {"face_count": len(faces)})
    return jsonify({"results": results})


@attendance_api_bp.route("/mark-webcam", methods=["POST"])
@login_required
@role_required("teacher")
def mark_webcam():
    file = request.files.get("frame")
    if not file:
        return jsonify({"error": "No frame provided."}), 400

    tmp_path = _save_upload(file, "webcam")
    try:
        image_array = _load_image_array(tmp_path)
        faces = face_engine.detect_faces(image_array)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception:  # noqa: BLE001
        logger.exception("Webcam frame face detection failed")
        return jsonify({"error": "Couldn't process that frame. Try again."}), 500
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    if not faces:
        return jsonify({"status": "unknown", "label": "No face detected"})

    result = attendance_service.process_face(faces[0]["encoding"], faces[0]["crop_160"])
    result["label"] = STATUS_LABELS.get(result["status"], result["status"])
    log_action("attendance.webcam_processed", {"status": result["status"]})
    return jsonify(result)


@attendance_api_bp.route("/logs")
@login_required
@role_required("teacher")
def logs():
    recent = attendance_service.get_recent_logs(50)
    return jsonify([
        {
            "student_id": r["student_id"], "name": r.get("name", ""),
            "timestamp": r["timestamp"].isoformat() if r.get("timestamp") else None,
            "total_attendance": r.get("total_attendance"),
        }
        for r in recent
    ])


@attendance_api_bp.route("/my-stats")
@login_required
def my_stats():
    if not current_user.student_id:
        return jsonify({"error": "No Student ID linked to your profile."}), 400
    stats = attendance_service.get_student_stats(current_user.student_id)
    return jsonify({
        "total_attendance": stats["total_attendance"],
        "last_attendance_date": stats["last_attendance_date"].isoformat() if stats["last_attendance_date"] else None,
        "logs": [
            {"timestamp": l["timestamp"].isoformat() if l.get("timestamp") else None, "total_attendance": l.get("total_attendance")}
            for l in stats["logs"]
        ],
    })


# ═══════════════════════════════════════════════════════════════════
#  Student Management (teacher only)
# ═══════════════════════════════════════════════════════════════════

@students_bp.route("")
@login_required
@role_required("teacher")
def list_students():
    search = request.args.get("q", "").strip()
    students = enrollment_service.list_students(search)
    return render_template("students/list.html", students=students, search=search)


@students_bp.route("/add", methods=["GET", "POST"])
@login_required
@role_required("teacher")
def add_student():
    if request.method == "POST":
        student_id = (request.form.get("student_id") or "").strip()
        name = (request.form.get("name") or "").strip()
        branch = request.form.get("branch") or ""
        year = request.form.get("year") or ""
        email = (request.form.get("email") or "").strip()

        if not student_id or not name:
            flash("Student ID and name are required.", "danger")
            return redirect(url_for("students.add_student"))

        try:
            enrollment_service.add_student(student_id, name, branch, year, email)
        except ValueError as exc:
            flash(str(exc), "danger")
            return redirect(url_for("students.add_student"))

        log_action("students.added", {"student_id": student_id})

        photo = request.files.get("photo")
        if photo and photo.filename:
            tmp_path = _save_upload(photo, "enroll")
            try:
                enrollment_service.enroll_face(current_app._get_current_object(), student_id, tmp_path)
                flash(f"{name} added and face enrolled.", "success")
            except ValueError as exc:
                flash(f"{name} added, but face enrollment failed: {exc}", "warning")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        else:
            flash(f"{name} added. Enroll a face photo to enable attendance matching.", "success")

        return redirect(url_for("students.list_students"))

    return render_template("students/add.html", branches=BRANCH_CHOICES, years=YEAR_CHOICES)


@students_api_bp.route("/enroll-face", methods=["POST"])
@login_required
@role_required("teacher")
def enroll_face():
    student_id = request.form.get("student_id")
    photo = request.files.get("photo")
    if not student_id or not photo:
        return jsonify({"error": "student_id and photo are required."}), 400

    tmp_path = _save_upload(photo, "enroll")
    try:
        enrollment_service.enroll_face(current_app._get_current_object(), student_id, tmp_path)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    log_action("students.face_enrolled", {"student_id": student_id})
    return jsonify({"enrolled": True})


@students_api_bp.route("/reencode", methods=["POST"])
@login_required
@role_required("teacher")
def reencode():
    result = enrollment_service.reencode_all(current_app._get_current_object())
    log_action("students.reencoded", {"succeeded": len(result["succeeded"]), "failed": len(result["failed"])})
    return jsonify(result)


@students_api_bp.route("/<student_id>", methods=["DELETE"])
@login_required
@role_required("teacher")
def delete_student(student_id):
    ok = enrollment_service.delete_student(current_app._get_current_object(), student_id)
    if ok:
        log_action("students.deleted", {"student_id": student_id})
    return jsonify({"deleted": ok})


@students_api_bp.route("/<student_id>/rate-limit-bypass", methods=["POST"])
@login_required
@role_required("teacher")
def grant_rate_limit_bypass(student_id):
    """Infra Feature 1 override: a 24-hour AI-feature quota bypass for one
    student — useful before exams. Looks up the linked user account by the
    roster student_id, since the attendance roster and login accounts are
    separate collections."""
    from app.extensions import db
    user = db.users.find_one({"student_id": student_id, "role": "student"})
    if not user:
        return jsonify({"error": "No login account is linked to this student ID yet."}), 404

    rate_limiter.grant_bypass(str(user["_id"]), hours=24)
    log_action("admin.rate_limit_override", {"student_id": student_id, "hours": 24})
    return jsonify({"granted": True, "student_id": student_id})


@students_bp.route("/export-csv")
@login_required
@role_required("teacher")
def export_csv():
    csv_data = enrollment_service.export_attendance_csv()
    return Response(
        csv_data, mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=attendance_report.csv"},
    )
