import io

from flask import Blueprint, render_template, request, jsonify, redirect, url_for, flash, send_file
from flask_login import login_required, current_user

from app.auth.utils import role_required
from app.utils.audit import log_action
from app.extensions import logger
from . import analytics_service

analytics_bp = Blueprint("analytics", __name__, url_prefix="/analytics")


# ═══════════════════════════════════════════════════════════════════
#  Main Analytics Dashboard
# ═══════════════════════════════════════════════════════════════════

@analytics_bp.route("")
@login_required
@role_required("teacher")
def index():
    attendance = analytics_service.get_attendance_analytics()
    quiz = analytics_service.get_quiz_analytics()
    dropout = analytics_service.get_dropout_analytics()
    wellness = analytics_service.get_wellness_analytics()
    return render_template(
        "analytics/index.html",
        attendance=attendance, quiz=quiz, dropout=dropout, wellness=wellness,
    )


@analytics_bp.route("/api/attendance")
@login_required
@role_required("teacher")
def api_attendance():
    data = analytics_service.get_attendance_analytics()
    return jsonify({
        "sessions_held": data["sessions_held"],
        "bar_chart": data["bar_chart"],
        "trend_chart": data["trend_chart"],
        "low_attendance": data["low_attendance"],
    })


@analytics_bp.route("/api/quiz")
@login_required
@role_required("teacher")
def api_quiz():
    data = analytics_service.get_quiz_analytics()
    return jsonify(data)


@analytics_bp.route("/api/dropout")
@login_required
@role_required("teacher")
def api_dropout():
    data = analytics_service.get_dropout_analytics()
    return jsonify(data)


@analytics_bp.route("/export/attendance.xlsx")
@login_required
@role_required("teacher")
def export_attendance_xlsx():
    try:
        xlsx_bytes = analytics_service.export_attendance_xlsx()
    except Exception:  # noqa: BLE001
        logger.exception("Attendance XLSX export failed")
        flash("Couldn't generate the attendance report right now.", "danger")
        return redirect(url_for("analytics.index"))
    log_action("analytics.attendance_exported", {})
    return send_file(
        io.BytesIO(xlsx_bytes),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True, download_name="attendance_report.xlsx",
    )


@analytics_bp.route("/export/quiz.csv")
@login_required
@role_required("teacher")
def export_quiz_csv():
    csv_data = analytics_service.export_quiz_csv()
    log_action("analytics.quiz_report_exported", {})
    return send_file(
        io.BytesIO(csv_data.encode("utf-8")), mimetype="text/csv",
        as_attachment=True, download_name="quiz_performance_report.csv",
    )


# ═══════════════════════════════════════════════════════════════════
#  Topic Difficulty Heatmap (Module 17)
# ═══════════════════════════════════════════════════════════════════

def _heatmap_filters():
    return {
        "date_range": request.args.get("date_range", "all"),
        "subject": request.args.get("subject") or None,
        "branch": request.args.get("branch") or None,
        "year": request.args.get("year") or None,
    }


@analytics_bp.route("/heatmap")
@login_required
@role_required("teacher")
def heatmap():
    filters = _heatmap_filters()
    data = analytics_service.get_heatmap_data(**filters)
    subjects = analytics_service.get_available_subjects()
    return render_template("analytics/heatmap.html", heatmap=data, subjects=subjects, filters=filters)


@analytics_bp.route("/api/heatmap-data")
@login_required
@role_required("teacher")
def api_heatmap_data():
    filters = _heatmap_filters()
    data = analytics_service.get_heatmap_data(**filters)
    return jsonify(data)


@analytics_bp.route("/api/heatmap/drilldown/<path:topic>")
@login_required
@role_required("teacher")
def api_heatmap_drilldown(topic):
    subject = request.args.get("subject", "")
    filters = _heatmap_filters()
    data = analytics_service.get_heatmap_drilldown(
        subject, topic, date_range=filters["date_range"], branch=filters["branch"], year=filters["year"]
    )
    return jsonify(data)


@analytics_bp.route("/api/heatmap/reteach", methods=["POST"])
@login_required
@role_required("teacher")
def api_heatmap_reteach():
    data = request.get_json(silent=True) or {}
    subject = (data.get("subject") or "").strip()
    topic = (data.get("topic") or "").strip()
    if not topic:
        return jsonify({"error": "Missing topic."}), 400

    try:
        pipeline = analytics_service.trigger_reteach(
            current_user.id, subject, topic, current_user.branch, current_user.year
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Re-teach material generation failed")
        return jsonify({"error": f"Couldn't generate material: {exc}"}), 500

    log_action("analytics.reteach_generated", {"topic": topic, "subject": subject})
    return jsonify({
        "generated": True,
        "redirect_url": url_for("nlp.study_material_result", pipeline_id=str(pipeline["_id"])),
    })


@analytics_bp.route("/heatmap/export-pdf")
@login_required
@role_required("teacher")
def heatmap_export_pdf():
    filters = _heatmap_filters()
    data = analytics_service.get_heatmap_data(**filters)
    try:
        pdf_bytes = analytics_service.render_heatmap_pdf(data)
    except Exception:  # noqa: BLE001
        logger.exception("Heatmap PDF export failed")
        flash("PDF export isn't available in this environment right now.", "warning")
        return redirect(url_for("analytics.heatmap"))
    log_action("analytics.heatmap_exported", {})
    return send_file(
        io.BytesIO(pdf_bytes), mimetype="application/pdf",
        as_attachment=True, download_name="topic_difficulty_heatmap.pdf",
    )
