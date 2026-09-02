from datetime import datetime, timedelta

from flask import Blueprint, render_template
from flask_login import login_required, current_user

from app.extensions import db, logger

dashboard_bp = Blueprint("dashboard", __name__, template_folder="../templates/dashboard")


def _safe_find(collection_name, query=None, sort=None, limit=None):
    """Best-effort Mongo read that degrades to [] instead of 500ing a
    dashboard when a collection (from a not-yet-built module) is empty
    or Mongo is briefly unreachable."""
    try:
        cursor = db[collection_name].find(query or {})
        if sort:
            cursor = cursor.sort(*sort)
        if limit:
            cursor = cursor.limit(limit)
        return list(cursor)
    except Exception:  # noqa: BLE001
        logger.warning("Dashboard read failed on collection '%s'", collection_name, exc_info=True)
        return []


def _safe_count(collection_name, query=None):
    try:
        return db[collection_name].count_documents(query or {})
    except Exception:  # noqa: BLE001
        return 0


@dashboard_bp.route("")
@dashboard_bp.route("/")
@login_required
def home():
    if current_user.is_teacher:
        return _teacher_dashboard()
    return _student_dashboard()


def _student_dashboard():
    user_doc = current_user.to_doc()
    student_id = current_user.student_id

    attendance_records = []
    total_sessions = 0
    last_marked = None
    if student_id:
        student_doc = db.students.find_one({"student_id": student_id}) if _try_mongo() else None
        if student_doc:
            total_sessions = student_doc.get("total_attendance", 0)
            last_marked = student_doc.get("last_attendance_date")
        attendance_records = _safe_find(
            "attendance_logs", {"student_id": student_id}, sort=("timestamp", -1), limit=20
        )

    quiz_results = _safe_find(
        "quiz_results", {"user_id": current_user.id}, sort=("timestamp", -1), limit=10
    )
    quiz_results = list(reversed(quiz_results))  # chronological for the line chart

    dropout_predictions = _safe_find(
        "dropout_predictions", {"user_id": current_user.id}, sort=("created_at", -1), limit=1
    )
    latest_risk = dropout_predictions[0] if dropout_predictions else None

    wellness_sessions = _safe_find(
        "wellness_sessions", {"user_id": current_user.id}, sort=("timestamp", -1), limit=1
    )
    days_since_wellness = None
    if wellness_sessions and wellness_sessions[0].get("timestamp"):
        delta = datetime.utcnow() - wellness_sessions[0]["timestamp"]
        days_since_wellness = delta.days

    announcements = []
    urgent_unread = []
    try:
        from app.modules.announcements import announcement_service as ann_svc
        if current_user.id:
            all_items = ann_svc.list_for_student(current_user.id)
            announcements = all_items[:3]
            urgent_unread = [a for a in all_items if a["category"] == "Urgent" and not a["is_read"]]
    except Exception:  # noqa: BLE001
        logger.warning("Could not load announcements for student dashboard", exc_info=True)

    unseen_corrections = []
    try:
        from app.modules.attendance import corrections_service as corr_svc
        if student_id:
            unseen_corrections = corr_svc.get_unseen_for_student(student_id)
    except Exception:  # noqa: BLE001
        logger.warning("Could not load corrections for student dashboard", exc_info=True)

    return render_template(
        "dashboard/student.html",
        student_id_linked=bool(student_id),
        total_sessions=total_sessions,
        last_marked=last_marked,
        attendance_records=attendance_records,
        quiz_results=quiz_results,
        latest_risk=latest_risk,
        days_since_wellness=days_since_wellness,
        announcements=announcements,
        urgent_unread=urgent_unread,
        unseen_corrections=unseen_corrections,
    )


def _teacher_dashboard():
    total_students = _safe_count("students")
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    sessions_today = _safe_count("attendance_logs", {"timestamp": {"$gte": today_start}})
    quizzes_taken = _safe_count("quiz_results")

    students = _safe_find("students")
    avg_attendance = (
        round(sum(s.get("total_attendance", 0) for s in students) / len(students), 1)
        if students
        else 0
    )

    attendance_chart = {
        "labels": [s.get("name", s.get("student_id", "?")) for s in students][:20],
        "chart_values": [s.get("total_attendance", 0) for s in students][:20],
    }

    dropout_predictions = _safe_find("dropout_predictions", sort=("created_at", -1), limit=500)
    risk_counts = {"High": 0, "Medium": 0, "Low": 0}
    for p in dropout_predictions:
        level = p.get("risk_level", "Low")
        if level in risk_counts:
            risk_counts[level] += 1

    quiz_results_all = _safe_find("quiz_results", sort=("timestamp", -1), limit=500)
    score_buckets = {"0-20": 0, "21-40": 0, "41-60": 0, "61-80": 0, "81-100": 0}
    for r in quiz_results_all:
        score = r.get("score_percent", 0)
        if score <= 20:
            score_buckets["0-20"] += 1
        elif score <= 40:
            score_buckets["21-40"] += 1
        elif score <= 60:
            score_buckets["41-60"] += 1
        elif score <= 80:
            score_buckets["61-80"] += 1
        else:
            score_buckets["81-100"] += 1

    recent_logs = _safe_find("attendance_logs", sort=("timestamp", -1), limit=30)

    pending_corrections = _safe_count("attendance_corrections", {"status": "Pending"})
    ungraded_tests = _safe_count("test_attempts", {"grading_status": "pending"})
    scheduled_announcements = _safe_count("announcements", {"status": "scheduled"})

    return render_template(
        "dashboard/teacher.html",
        total_students=total_students,
        sessions_today=sessions_today,
        avg_attendance=avg_attendance,
        quizzes_taken=quizzes_taken,
        attendance_chart=attendance_chart,
        risk_counts=risk_counts,
        score_buckets=score_buckets,
        recent_logs=recent_logs,
        pending_corrections=pending_corrections,
        ungraded_tests=ungraded_tests,
        scheduled_announcements=scheduled_announcements,
    )


def _try_mongo():
    try:
        db.client.admin.command("ping")
        return True
    except Exception:  # noqa: BLE001
        return False
