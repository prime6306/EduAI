"""
Progress and pace are derived entirely from `total_lectures`, `exam_date`,
and each topic's `covered`/`covered_date` - no extra scheduling input is
required from the teacher. "Lectures used so far" is approximated as the
count of distinct dates on which at least one topic was marked covered
(one teaching session per day is a reasonable assumption for this signal).
"""
from datetime import datetime, timedelta

from bson import ObjectId
from bson.errors import InvalidId

from app.extensions import db, logger


def create_syllabus(teacher_id, subject, branch, year, semester, academic_year,
                     exam_date, total_lectures, units):
    doc = {
        "teacher_id": teacher_id,
        "subject": subject,
        "branch": branch,
        "year": year,
        "semester": semester,
        "academic_year": academic_year,
        "exam_date": exam_date,
        "total_lectures": total_lectures,
        "units": units,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }
    result = db.syllabuses.insert_one(doc)
    doc["_id"] = result.inserted_id
    return doc


def list_for_teacher(teacher_id):
    return list(db.syllabuses.find({"teacher_id": teacher_id}).sort("created_at", -1))


def get(syllabus_id, teacher_id=None):
    try:
        oid = ObjectId(syllabus_id)
    except (InvalidId, TypeError):
        return None
    query = {"_id": oid}
    if teacher_id:
        query["teacher_id"] = teacher_id
    return db.syllabuses.find_one(query)


def toggle_topic(syllabus_id, teacher_id, unit_index, topic_index, note=None):
    doc = get(syllabus_id, teacher_id)
    if not doc:
        return None
    try:
        unit = doc["units"][unit_index]
        topic = unit["topics"][topic_index]
    except (IndexError, KeyError):
        return None

    now_covered = not topic.get("covered", False)
    topic["covered"] = now_covered
    topic["covered_date"] = datetime.utcnow() if now_covered else None
    if note is not None:
        topic["note"] = note

    db.syllabuses.update_one(
        {"_id": doc["_id"]},
        {"$set": {"units": doc["units"], "updated_at": datetime.utcnow()}},
    )
    return topic


def compute_progress(doc):
    all_topics = [t for u in doc["units"] for t in u["topics"]]
    total_topics = len(all_topics)
    covered_topics = sum(1 for t in all_topics if t.get("covered"))
    overall_pct = round((covered_topics / total_topics) * 100, 1) if total_topics else 0.0

    per_unit = []
    for u in doc["units"]:
        u_total = len(u["topics"])
        u_covered = sum(1 for t in u["topics"] if t.get("covered"))
        per_unit.append({
            "number": u["number"], "title": u["title"],
            "covered": u_covered, "total": u_total,
            "pct": round((u_covered / u_total) * 100, 1) if u_total else 0.0,
        })

    covered_dates = sorted({
        t["covered_date"].date() for t in all_topics if t.get("covered") and t.get("covered_date")
    })
    lectures_used = len(covered_dates)
    total_lectures = doc.get("total_lectures", 0) or 0
    lectures_remaining = max(total_lectures - lectures_used, 0)

    expected_pct = round((lectures_used / total_lectures) * 100, 1) if total_lectures else 0.0
    pace_diff_pct = overall_pct - expected_pct

    if pace_diff_pct >= 5:
        lectures_ahead = round(abs(pace_diff_pct) / 100 * total_lectures) if total_lectures else 0
        pace_status = f"{lectures_ahead} lecture{'s' if lectures_ahead != 1 else ''} ahead of schedule"
    elif pace_diff_pct <= -5:
        lectures_behind = round(abs(pace_diff_pct) / 100 * total_lectures) if total_lectures else 0
        topics_remaining = total_topics - covered_topics
        topics_per_class_needed = (
            round(topics_remaining / lectures_remaining, 1) if lectures_remaining > 0 else topics_remaining
        )
        pace_status = f"{lectures_behind} lecture{'s' if lectures_behind != 1 else ''} behind - need ~{topics_per_class_needed} topics/class to catch up"
    else:
        pace_status = "On schedule"

    projected_finish_date = None
    exam_date = doc.get("exam_date")
    if lectures_used > 0 and covered_topics > 0:
        topics_per_lecture = covered_topics / lectures_used
        topics_remaining = total_topics - covered_topics
        lectures_needed = topics_remaining / topics_per_lecture if topics_per_lecture > 0 else None
        if lectures_needed is not None:
            first_date = doc.get("created_at", datetime.utcnow()).date()
            days_elapsed = max((datetime.utcnow().date() - first_date).days, 1)
            avg_days_per_lecture = days_elapsed / lectures_used
            extra_days = lectures_needed * avg_days_per_lecture
            projected_finish_date = datetime.utcnow().date() + timedelta(days=extra_days)

    exam_overshoot_days = None
    if projected_finish_date and exam_date and projected_finish_date > exam_date.date():
        exam_overshoot_days = (projected_finish_date - exam_date.date()).days

    uncovered = []
    for u in doc["units"]:
        for t in u["topics"]:
            if not t.get("covered"):
                uncovered.append({"unit": u["title"], "topic": t["text"]})

    heatmap = {}
    for t in all_topics:
        if t.get("covered") and t.get("covered_date"):
            key = t["covered_date"].strftime("%Y-%m-%d")
            heatmap[key] = heatmap.get(key, 0) + 1

    return {
        "overall_pct": overall_pct,
        "total_topics": total_topics,
        "covered_topics": covered_topics,
        "per_unit": per_unit,
        "lectures_used": lectures_used,
        "lectures_remaining": lectures_remaining,
        "pace_status": pace_status,
        "pace_behind": pace_diff_pct <= -5,
        "projected_finish_date": projected_finish_date,
        "exam_overshoot_days": exam_overshoot_days,
        "uncovered_topics": uncovered,
        "heatmap": heatmap,
    }


def generate_naac_summary(doc, progress):
    """Groq-written pace analysis paragraph for the NAAC export. Falls back
    to a templated sentence if the LLM isn't configured."""
    try:
        from app.modules.nlp.llm_client import chat_completion
        messages = [
            {"role": "system", "content": "You write concise, factual academic progress summaries for NAAC documentation. One paragraph, formal tone, no markdown."},
            {"role": "user", "content": (
                f"Subject: {doc['subject']}. {progress['covered_topics']} of {progress['total_topics']} topics "
                f"covered ({progress['overall_pct']}%). {progress['lectures_used']} of {doc.get('total_lectures', 0)} "
                f"lecture slots used. Pace status: {progress['pace_status']}. Write a 3-4 sentence progress summary."
            )},
        ]
        return chat_completion(messages, max_tokens=200)
    except Exception:  # noqa: BLE001
        logger.info("Groq not available for NAAC summary - using templated fallback.")
        return (
            f"As of this report, {progress['covered_topics']} of {progress['total_topics']} topics "
            f"({progress['overall_pct']}%) have been covered in {doc['subject']}, using "
            f"{progress['lectures_used']} of {doc.get('total_lectures', 0)} available lecture slots. "
            f"Current pace status: {progress['pace_status']}."
        )
