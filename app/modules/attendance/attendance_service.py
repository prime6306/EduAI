"""
Orchestrates one face through the full attendance pipeline: match ->
anti-spoof -> cooldown -> mark. Used by both the group-photo route (once
per detected face) and the webcam route (once, for the single frame).
"""
import time
from datetime import datetime

from flask import current_app

from app.extensions import db
from . import face_engine, antispoof

ANTISPOOF_REAL_THRESHOLD = 0.5


def check_cooldown(student_doc: dict):
    """Returns (allowed, seconds_since_last_mark)."""
    last = student_doc.get("last_attendance_date")
    if not last:
        return True, 0
    elapsed = int((datetime.utcnow() - last).total_seconds())
    cooldown = current_app.config["ATTENDANCE_COOLDOWN_SECONDS"]
    return elapsed >= cooldown, elapsed


def mark_attendance(student_id: str, name: str, session_name: str = "Class Session") -> int:
    now = datetime.utcnow()
    db.students.update_one(
        {"student_id": student_id},
        {"$inc": {"total_attendance": 1}, "$set": {"last_attendance_date": now}},
    )
    updated = db.students.find_one({"student_id": student_id})
    total = updated.get("total_attendance", 1)
    db.attendance_logs.insert_one({
        "student_id": student_id, "name": name, "timestamp": now,
        "total_attendance": total, "session_name": session_name,
    })
    return total


def log_spoof_attempt(student_id, confidence: float) -> None:
    db.spoof_attempts.insert_one({
        "student_id": student_id, "confidence": confidence, "timestamp": datetime.utcnow(),
    })


def process_face(encoding, crop_160) -> dict:
    """Runs the full pipeline for a single detected face and returns a
    result dict with a `status` of marked / spoof / duplicate / unknown /
    antispoof_unavailable, plus per-stage timings in milliseconds."""
    timings = {}

    t0 = time.time()
    student_id, distance = face_engine.match_encoding(encoding)
    timings["recognition_ms"] = round((time.time() - t0) * 1000, 1)

    if not student_id:
        return {"status": "unknown", "timings": timings}

    student_doc = db.students.find_one({"student_id": student_id})
    name = student_doc.get("name", student_id) if student_doc else student_id

    t1 = time.time()
    try:
        real_prob = antispoof.predict_is_real(crop_160)
    except (RuntimeError, FileNotFoundError) as exc:
        return {
            "status": "antispoof_unavailable", "student_id": student_id, "name": name,
            "timings": timings, "error": str(exc),
        }
    timings["antispoof_ms"] = round((time.time() - t1) * 1000, 1)

    if real_prob < ANTISPOOF_REAL_THRESHOLD:
        log_spoof_attempt(student_id, real_prob)
        return {
            "status": "spoof", "student_id": student_id, "name": name,
            "confidence": round(real_prob, 3), "timings": timings,
        }

    allowed, elapsed = check_cooldown(student_doc or {})
    if not allowed:
        return {
            "status": "duplicate", "student_id": student_id, "name": name,
            "seconds_elapsed": elapsed, "timings": timings,
        }

    t2 = time.time()
    total = mark_attendance(student_id, name)
    timings["db_ms"] = round((time.time() - t2) * 1000, 1)

    return {
        "status": "marked", "student_id": student_id, "name": name,
        "total_attendance": total, "timings": timings,
    }


def get_report_metrics() -> dict:
    total_sessions = db.attendance_logs.count_documents({})
    unique_students = len(db.attendance_logs.distinct("student_id"))
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    today_count = db.attendance_logs.count_documents({"timestamp": {"$gte": today_start}})

    students = list(db.students.find())
    class_avg = (
        round(sum(s.get("total_attendance", 0) for s in students) / len(students), 1)
        if students else 0
    )
    return {
        "total_sessions": total_sessions,
        "unique_students": unique_students,
        "today_count": today_count,
        "class_avg": class_avg,
        "chart_labels": [s.get("name", s.get("student_id")) for s in students][:20],
        "chart_values": [s.get("total_attendance", 0) for s in students][:20],
    }


def get_recent_logs(limit: int = 30) -> list[dict]:
    return list(db.attendance_logs.find().sort("timestamp", -1).limit(limit))


def get_student_stats(student_id: str) -> dict:
    student_doc = db.students.find_one({"student_id": student_id})
    logs = list(db.attendance_logs.find({"student_id": student_id}).sort("timestamp", -1).limit(20))
    return {
        "total_attendance": student_doc.get("total_attendance", 0) if student_doc else 0,
        "last_attendance_date": student_doc.get("last_attendance_date") if student_doc else None,
        "logs": logs,
    }
