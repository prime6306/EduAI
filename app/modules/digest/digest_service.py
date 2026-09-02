"""
Weekly Class Digest (Module 18).

Every Sunday at 23:00, assembles a data-driven summary of the past week
across attendance, quizzes, dropout risk, wellness, and platform activity
for each teacher, optionally emails it, and stores it for later viewing at
/digest/history. A startup check also catches up on any week that was
missed while the server was offline, per spec.

The six numeric sections always get built from real data. Only the "AI
Summary Paragraph" section depends on Groq being configured — if it isn't,
a plain templated paragraph built from the same numbers is used instead,
so a digest is never blocked on an optional AI feature.
"""
from collections import Counter
from datetime import datetime, timedelta

from bson import ObjectId
from bson.errors import InvalidId

from app.extensions import db, logger
from app.modules.analytics.analytics_service import _users_by_id
from app.modules.nlp.llm_client import chat_completion, LLMNotConfigured


# ═══════════════════════════════════════════════════════════════════
#  Week bounds
# ═══════════════════════════════════════════════════════════════════

def get_last_completed_week(reference: datetime | None = None) -> tuple[datetime, datetime]:
    """(week_start, week_end) for the most recently *completed* Mon-Sun
    week as of `reference` (defaults to now). week_end is exclusive."""
    reference = reference or datetime.utcnow()
    days_since_monday = reference.weekday()  # Monday == 0
    this_week_start = datetime(reference.year, reference.month, reference.day) - timedelta(days=days_since_monday)
    week_start = this_week_start - timedelta(days=7)
    week_end = this_week_start
    return week_start, week_end


# ═══════════════════════════════════════════════════════════════════
#  Section builders
# ═══════════════════════════════════════════════════════════════════

def _attendance_section(week_start, week_end, prev_start, prev_end) -> dict:
    students = list(db.students.find())
    logs = list(db.attendance_logs.find({"timestamp": {"$gte": week_start, "$lt": week_end}}))
    prev_logs = list(db.attendance_logs.find({"timestamp": {"$gte": prev_start, "$lt": prev_end}}))

    sessions = len({l["timestamp"].date() for l in logs if l.get("timestamp")})
    prev_sessions = len({l["timestamp"].date() for l in prev_logs if l.get("timestamp")})

    by_student = Counter(l.get("student_id") for l in logs)
    possible = sessions * len(students) if students else 0
    rate = round((len(logs) / possible) * 100, 1) if possible else 0.0

    prev_possible = prev_sessions * len(students) if students else 0
    prev_rate = round((len(prev_logs) / prev_possible) * 100, 1) if prev_possible else 0.0

    absent_students = []
    if sessions:
        for s in students:
            if by_student.get(s["student_id"], 0) < sessions * 0.5:
                absent_students.append(s.get("name", s["student_id"]))

    return {
        "sessions": sessions, "rate": rate, "wow_change": round(rate - prev_rate, 1),
        "absent_students": sorted(absent_students),
    }


def _quiz_section(week_start, week_end, prev_start, prev_end) -> dict:
    users = _users_by_id()
    results = list(db.quiz_results.find({"timestamp": {"$gte": week_start, "$lt": week_end}}))
    prev_results = list(db.quiz_results.find({"timestamp": {"$gte": prev_start, "$lt": prev_end}}))

    avg = round(sum(r.get("score_percent", 0) for r in results) / len(results), 1) if results else 0.0
    prev_avg = round(sum(r.get("score_percent", 0) for r in prev_results) / len(prev_results), 1) if prev_results else 0.0

    by_topic = {}
    for r in results:
        by_topic.setdefault(r.get("topic", "Unknown"), []).append(r.get("score_percent", 0))
    topic_avgs = {t: sum(v) / len(v) for t, v in by_topic.items()}
    top_topics = sorted(topic_avgs.items(), key=lambda kv: kv[1], reverse=True)[:3]
    weakest_topic = min(topic_avgs.items(), key=lambda kv: kv[1])[0] if topic_avgs else None

    last_attempt = {}
    for r in db.quiz_results.find({}, {"user_id": 1, "timestamp": 1}):
        uid, ts = r.get("user_id"), r.get("timestamp")
        if uid and ts and (uid not in last_attempt or ts > last_attempt[uid]):
            last_attempt[uid] = ts
    cutoff = week_end - timedelta(days=7)
    inactive = [
        u.get("name", "Unknown") for uid, u in users.items()
        if uid not in last_attempt or last_attempt[uid] < cutoff
    ]

    return {
        "total_taken": len(results), "avg_score": avg, "wow_change": round(avg - prev_avg, 1),
        "top_topics": [t for t, _ in top_topics], "weakest_topic": weakest_topic,
        "inactive_students": sorted(inactive)[:20],
    }


def _dropout_section(week_start, week_end) -> dict:
    def latest_risk_as_of(cutoff):
        latest = {}
        for p in db.dropout_predictions.find({"created_at": {"$lte": cutoff}}).sort("created_at", 1):
            latest[p.get("user_id")] = p.get("risk_level")
        return latest

    users = _users_by_id()
    before, current = latest_risk_as_of(week_start), latest_risk_as_of(week_end)

    newly_high = [users.get(uid, {}).get("name", "Unknown") for uid, lvl in current.items() if lvl == "High" and before.get(uid) != "High"]
    improved = [users.get(uid, {}).get("name", "Unknown") for uid, lvl in before.items() if lvl == "High" and current.get(uid) != "High"]

    return {
        "high_count": sum(1 for lvl in current.values() if lvl == "High"),
        "newly_high": sorted(newly_high), "improved": sorted(improved),
    }


def _wellness_section(week_start, week_end) -> dict:
    assessments = list(db.wellness_assessments.find({"timestamp": {"$gte": week_start, "$lt": week_end}}))
    sessions = list(db.wellness_sessions.find({"created_at": {"$gte": week_start, "$lt": week_end}}))
    return {
        "sessions": len(sessions),
        "moderate_severe_count": sum(1 for a in assessments if a.get("severity") in ("Moderate", "Severe")),
        "crisis_detected": any(s.get("crisis_flagged") for s in sessions),
    }


def _activity_section(week_start, week_end) -> dict:
    convos = list(db.doubt_conversations.find({"updated_at": {"$gte": week_start, "$lt": week_end}}))
    doubt_questions = sum(
        1 for c in convos for m in c.get("messages", [])
        if m.get("role") == "user" and week_start <= (m.get("timestamp") or week_start) < week_end
    )
    subject_counts = Counter(c.get("subject") for c in convos if c.get("subject"))

    return {
        "doubt_questions": doubt_questions,
        "pdfs_uploaded": db.pdfs.count_documents({"created_at": {"$gte": week_start, "$lt": week_end}}),
        "pipelines_generated": db.topic_pipelines.count_documents({"created_at": {"$gte": week_start, "$lt": week_end}}),
        "top_subject": subject_counts.most_common(1)[0][0] if subject_counts else None,
    }


def _fallback_summary(sections: dict) -> str:
    a, q, d = sections["attendance"], sections["quiz"], sections["dropout"]
    direction = "up" if a["wow_change"] >= 0 else "down"
    parts = [
        f"Attendance was {a['rate']}% this week, {direction} {abs(a['wow_change'])} points from last week.",
        f"{q['total_taken']} quiz(zes) were taken with an average score of {q['avg_score']}%"
        + (f", and {q['weakest_topic']} was the weakest topic." if q["weakest_topic"] else "."),
        f"{d['high_count']} student(s) are currently flagged as high dropout risk"
        + (f", including {len(d['newly_high'])} new this week." if d["newly_high"] else "."),
    ]
    return " ".join(parts)


def _ai_summary(sections: dict) -> str:
    fallback = _fallback_summary(sections)
    a, q, d, w = sections["attendance"], sections["quiz"], sections["dropout"], sections["wellness"]
    prompt = (
        "Write a 3-4 sentence plain-language summary paragraph for a teacher's weekly class "
        "digest. Be direct and specific, referencing the numbers given. No greeting, no "
        "sign-off, no markdown formatting — just the paragraph.\n\n"
        f"Attendance: {a['rate']}% this week ({a['wow_change']:+} pts vs last week). "
        f"{len(a['absent_students'])} student(s) missed over half this week's sessions.\n"
        f"Quizzes: {q['total_taken']} taken, avg {q['avg_score']}% ({q['wow_change']:+} pts vs last week). "
        f"Weakest topic: {q['weakest_topic'] or 'none'}. Top topics: {', '.join(q['top_topics']) or 'none'}.\n"
        f"Dropout: {d['high_count']} currently high-risk, {len(d['newly_high'])} newly high-risk, "
        f"{len(d['improved'])} improved out of high-risk this week.\n"
        f"Wellness: {w['sessions']} session(s), {w['moderate_severe_count']} moderate/severe, "
        f"crisis flag: {'yes' if w['crisis_detected'] else 'no'}."
    )
    try:
        text = chat_completion([
            {"role": "system", "content": "You write concise, factual weekly summaries for teachers. Output only the paragraph, nothing else."},
            {"role": "user", "content": prompt},
        ])
        return text.strip() if text and text.strip() else fallback
    except LLMNotConfigured:
        return fallback
    except Exception:  # noqa: BLE001
        logger.warning("Digest AI summary generation failed — using fallback paragraph.")
        return fallback


# ═══════════════════════════════════════════════════════════════════
#  Assembly, persistence, email
# ═══════════════════════════════════════════════════════════════════

def generate_digest(teacher_id: str, week_start: datetime, week_end: datetime) -> dict:
    prev_start, prev_end = week_start - timedelta(days=7), week_start
    sections = {
        "attendance": _attendance_section(week_start, week_end, prev_start, prev_end),
        "quiz": _quiz_section(week_start, week_end, prev_start, prev_end),
        "dropout": _dropout_section(week_start, week_end),
        "wellness": _wellness_section(week_start, week_end),
        "activity": _activity_section(week_start, week_end),
    }
    summary_paragraph = _ai_summary(sections)

    return {
        "teacher_id": teacher_id, "week_start": week_start, "week_end": week_end,
        "generated_at": datetime.utcnow(),
        "attendance": sections["attendance"], "quiz": sections["quiz"],
        "dropout": sections["dropout"], "wellness": sections["wellness"],
        "activity": sections["activity"], "summary_paragraph": summary_paragraph,
        "email_sent": False, "email_sent_at": None,
    }


def digest_exists(teacher_id: str, week_start: datetime) -> bool:
    return db.digest_reports.count_documents({"teacher_id": teacher_id, "week_start": week_start}) > 0


def save_digest(doc: dict) -> dict:
    result = db.digest_reports.insert_one(doc)
    doc["_id"] = result.inserted_id
    return doc


def list_digests(teacher_id: str) -> list[dict]:
    return list(db.digest_reports.find({"teacher_id": teacher_id}).sort("week_start", -1))


def get_digest(digest_id: str, teacher_id: str) -> dict | None:
    try:
        oid = ObjectId(digest_id)
    except (InvalidId, TypeError):
        return None
    return db.digest_reports.find_one({"_id": oid, "teacher_id": teacher_id})


def is_email_enabled(teacher_id: str) -> bool:
    user = db.users.find_one({"_id": ObjectId(teacher_id)}, {"digest_email_enabled": 1})
    return bool(user.get("digest_email_enabled", True)) if user else True


def set_email_enabled(teacher_id: str, enabled: bool) -> None:
    db.users.update_one({"_id": ObjectId(teacher_id)}, {"$set": {"digest_email_enabled": enabled}})


def send_digest_email(digest_doc: dict, teacher_email: str) -> bool:
    """Best-effort — returns False (and logs) rather than raising, so a
    misconfigured mail server never breaks digest generation itself."""
    if not teacher_email:
        logger.info("No teacher email on file — skipping digest email.")
        return False
    try:
        from flask import current_app, render_template
        from flask_mail import Message
        from app.extensions import mail

        if not current_app.config.get("MAIL_SERVER"):
            logger.info("MAIL_SERVER not configured — skipping digest email send.")
            return False

        html_body = render_template("digest/_email.html", digest=digest_doc, timedelta=timedelta)
        msg = Message(
            subject=f"EduAI Weekly Digest — {digest_doc['week_start'].strftime('%b %d').replace(' 0', ' ')} to {(digest_doc['week_end'] - timedelta(days=1)).strftime('%b %d, %Y').replace(' 0', ' ')}",
            recipients=[teacher_email],
            html=html_body,
        )
        mail.send(msg)
        return True
    except Exception:  # noqa: BLE001
        logger.warning("Digest email send failed — digest was still generated and saved.")
        return False


def generate_and_save(teacher_id: str, teacher_email: str, week_start: datetime, week_end: datetime, send_email: bool = True) -> dict:
    doc = generate_digest(teacher_id, week_start, week_end)
    if send_email and is_email_enabled(teacher_id):
        sent = send_digest_email(doc, teacher_email)
        doc["email_sent"] = sent
        doc["email_sent_at"] = datetime.utcnow() if sent else None
    saved = save_digest(doc)
    return saved


# ═══════════════════════════════════════════════════════════════════
#  Scheduling — weekly cron job + startup catch-up
# ═══════════════════════════════════════════════════════════════════

def run_weekly_job(app) -> int:
    """Generate this week's digest for every teacher who doesn't already
    have one. Safe to call repeatedly (idempotent) — used both by the
    Sunday 23:00 cron job and the startup catch-up check."""
    with app.app_context():
        week_start, week_end = get_last_completed_week()
        generated = 0
        for teacher in db.users.find({"role": "teacher"}):
            teacher_id = str(teacher["_id"])
            if digest_exists(teacher_id, week_start):
                continue
            try:
                generate_and_save(teacher_id, teacher.get("email", ""), week_start, week_end)
                generated += 1
            except Exception:  # noqa: BLE001
                logger.exception("Weekly digest generation failed for teacher %s", teacher_id)
        if generated:
            logger.info("Weekly digest job: generated %d digest(s).", generated)
        return generated
