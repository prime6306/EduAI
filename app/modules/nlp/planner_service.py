"""
Study Planner (Module 12): AI-generated personalised weekly study schedule.

Mirrors the two-step pattern used elsewhere in the app (question paper,
CO-PO): generation produces a result the user reviews first; an explicit
"Save Plan" action is what persists it to Mongo. This matches the spec's
distinct "Save Plan" button rather than auto-saving every generation like
Study Material does.
"""
from datetime import datetime, date

from bson import ObjectId
from bson.errors import InvalidId
from flask import current_app, render_template

from app.extensions import db, logger
from . import prompts
from .llm_client import chat_json

VALID_PRIORITIES = {"high", "medium", "low"}
DAY_LABELS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _duration_label(minutes: int) -> str:
    hours, mins = divmod(minutes, 60)
    if hours and mins:
        return f"{hours}h {mins}m"
    if hours:
        return f"{hours}h"
    return f"{mins} min"


def _normalize_schedule(raw: dict) -> tuple[list[dict], list[str]]:
    schedule_raw = raw.get("schedule", [])
    by_day = {d.get("day"): d.get("sessions", []) for d in schedule_raw if d.get("day")}

    schedule = []
    for label in DAY_LABELS:
        sessions_raw = by_day.get(label, [])
        sessions = []
        for s in sessions_raw:
            subject = (s.get("subject") or "").strip()
            if not subject:
                continue
            priority = (s.get("priority") or "medium").lower()
            if priority not in VALID_PRIORITIES:
                priority = "medium"
            try:
                duration = max(15, min(240, int(s.get("duration_minutes"))))
            except (TypeError, ValueError):
                duration = 60
            sessions.append({
                "subject": subject,
                "topic": (s.get("topic") or "").strip(),
                "duration_minutes": duration,
                "duration_label": _duration_label(duration),
                "priority": priority,
            })
        schedule.append({"day": label, "sessions": sessions})

    if not any(d["sessions"] for d in schedule):
        raise ValueError("The AI didn't return a usable schedule — try again.")

    tips = [t.strip() for t in raw.get("tips", []) if isinstance(t, str) and t.strip()][:5]
    if not tips:
        tips = ["Review yesterday's topics for five minutes before starting a new session."]
    return schedule, tips


def _generate_with_retry(messages: list[dict]) -> tuple[list[dict], list[str]]:
    try:
        raw = chat_json(messages)
        return _normalize_schedule(raw)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Study plan generation malformed, retrying once: %s", exc)
        raw = chat_json(messages)
        return _normalize_schedule(raw)


def generate_plan(subjects: list[str], exam_date_str: str, hours_per_day: int, branch: str, year: str) -> dict:
    days_until_exam = None
    if exam_date_str:
        try:
            exam_d = datetime.strptime(exam_date_str, "%Y-%m-%d").date()
            days_until_exam = (exam_d - date.today()).days
        except ValueError:
            exam_date_str = ""

    messages = prompts.planner_prompt(subjects, exam_date_str, days_until_exam, hours_per_day, branch, year)
    schedule, tips = _generate_with_retry(messages)

    return {
        "subjects": subjects,
        "exam_date": exam_date_str,
        "days_until_exam": days_until_exam,
        "hours_per_day": hours_per_day,
        "branch": branch,
        "year": year,
        "schedule": schedule,
        "tips": tips,
    }


def save_plan(user_id: str, plan: dict) -> dict:
    doc = dict(plan)
    doc["user_id"] = user_id
    doc["created_at"] = datetime.utcnow()
    result = db.study_plans.insert_one(doc)
    doc["_id"] = result.inserted_id
    _log_to_mlflow(doc)
    return doc


def _log_to_mlflow(doc: dict) -> None:
    try:
        import mlflow
        mlflow.set_tracking_uri(current_app.config["MLFLOW_TRACKING_URI"])
        mlflow.set_experiment(current_app.config["MLFLOW_EXPERIMENT"])
        with mlflow.start_run(run_name="study_planner"):
            mlflow.log_param("subjects", ", ".join(doc.get("subjects", [])))
            mlflow.log_param("hours_per_day", doc.get("hours_per_day"))
    except Exception:  # noqa: BLE001
        logger.info("MLflow not reachable — skipping study-planner run log.")


def get_plan(plan_id: str, user_id: str) -> dict | None:
    try:
        oid = ObjectId(plan_id)
    except (InvalidId, TypeError):
        return None
    return db.study_plans.find_one({"_id": oid, "user_id": user_id})


def list_saved_plans(user_id: str) -> list[dict]:
    return list(db.study_plans.find({"user_id": user_id}).sort("created_at", -1))


def delete_plan(plan_id: str, user_id: str) -> bool:
    try:
        oid = ObjectId(plan_id)
    except (InvalidId, TypeError):
        return False
    result = db.study_plans.delete_one({"_id": oid, "user_id": user_id})
    return result.deleted_count > 0


def render_pdf(plan: dict) -> bytes:
    """Deferred import so a missing WeasyPrint native dep can't break app boot."""
    from weasyprint import HTML
    html_string = render_template("planner/pdf.html", plan=plan)
    return HTML(string=html_string).write_pdf()
